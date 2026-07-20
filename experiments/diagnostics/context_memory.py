from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from seqmem.encoding import SSTDDiscreteEncoder


@dataclass
class RunningMean:
    total: float = 0.0
    count: float = 0.0

    def add(self, value: float, weight: float = 1.0) -> None:
        self.total += value * weight
        self.count += weight

    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0


class SparseContextMemory:
    """Fast associative memory for multi-field sparse contexts.

    This is an optimized Fig. 9-oriented memory. Instead of pushing each field
    through a long generic sequence, it encodes all fields with SSTD and stores
    direct context-to-target evidence. It is useful for testing the paper's
    time-series setting where day, time, and value jointly predict a future
    value.
    """

    def __init__(
        self,
        encoder: SSTDDiscreteEncoder,
        max_order: int = 3,
        decay: float = 1.0,
    ) -> None:
        self.encoder = encoder
        self.max_order = max_order
        self.decay = decay
        self._tables: list[dict[tuple[int, ...], dict[str, float]]] = [
            defaultdict(dict) for _ in range(max_order)
        ]

    def observe(self, fields: list[str], target: str) -> None:
        active_sets = [set(self.encoder.encode(field).columns) for field in fields]
        for order, table in enumerate(self._tables, start=1):
            for key in self._keys(active_sets, order):
                bucket = table[key]
                if self.decay < 1.0:
                    for symbol in list(bucket):
                        bucket[symbol] *= self.decay
                        if bucket[symbol] < 1e-6:
                            del bucket[symbol]
                bucket[target] = bucket.get(target, 0.0) + 1.0

    def predict(self, fields: list[str], candidates: set[str]) -> str | None:
        best_symbol, _best_score, _margin = self.predict_with_confidence(
            fields, candidates
        )
        return best_symbol

    def predict_with_confidence(
        self, fields: list[str], candidates: set[str]
    ) -> tuple[str | None, float, float]:
        active_sets = [set(self.encoder.encode(field).columns) for field in fields]
        scores = {candidate: 0.0 for candidate in candidates}

        for order, table in enumerate(self._tables, start=1):
            order_weight = float(order * order)
            for key in self._keys(active_sets, order):
                bucket = table.get(key)
                if not bucket:
                    continue
                for symbol, value in bucket.items():
                    if symbol in scores:
                        scores[symbol] += order_weight * value

        best_symbol = max(scores, key=scores.get)
        if scores[best_symbol] <= 0.0:
            return None, 0.0, 0.0
        ordered = sorted(scores.values(), reverse=True)
        runner_up = ordered[1] if len(ordered) > 1 else 0.0
        return best_symbol, scores[best_symbol], scores[best_symbol] - runner_up

    def _keys(
        self, active_sets: list[set[int]], order: int
    ) -> list[tuple[int, ...]]:
        keys: list[tuple[int, ...]] = []
        for field_index, active in enumerate(active_sets):
            for column in active:
                keys.append((field_index, column))

        if order == 1:
            return keys

        if order >= 2:
            pair_keys: list[tuple[int, ...]] = []
            for left_index in range(len(active_sets)):
                for right_index in range(left_index + 1, len(active_sets)):
                    for left_column in active_sets[left_index]:
                        for right_column in active_sets[right_index]:
                            pair_keys.append(
                                (left_index, left_column, right_index, right_column)
                            )
            if order == 2:
                return pair_keys
            keys = pair_keys

        triple_keys: list[tuple[int, ...]] = []
        for a in range(len(active_sets)):
            for b in range(a + 1, len(active_sets)):
                for c in range(b + 1, len(active_sets)):
                    for col_a in active_sets[a]:
                        for col_b in active_sets[b]:
                            for col_c in active_sets[c]:
                                triple_keys.append((a, col_a, b, col_b, c, col_c))
        return triple_keys


class SparseContextRegressor:
    """Sparse distributed context memory that directly predicts a number."""

    def __init__(
        self,
        encoder: SSTDDiscreteEncoder,
        max_order: int = 3,
        decay: float = 1.0,
    ) -> None:
        self.encoder = encoder
        self.max_order = max_order
        self.decay = decay
        self._tables: list[dict[tuple[int, ...], RunningMean]] = [
            defaultdict(RunningMean) for _ in range(max_order)
        ]

    def observe(self, fields: list[str], target: float) -> None:
        active_sets = [set(self.encoder.encode(field).columns) for field in fields]
        for order, table in enumerate(self._tables, start=1):
            for key in self._keys(active_sets, order):
                stat = table[key]
                if self.decay < 1.0:
                    stat.total *= self.decay
                    stat.count *= self.decay
                stat.add(target)

    def predict(self, fields: list[str]) -> tuple[float | None, float]:
        active_sets = [set(self.encoder.encode(field).columns) for field in fields]

        for order in range(len(self._tables), 0, -1):
            table = self._tables[order - 1]
            order_weight = float(order * order)
            weighted_total = 0.0
            weighted_count = 0.0
            for key in self._keys(active_sets, order):
                stat = table.get(key)
                if stat is None or stat.count <= 0:
                    continue
                weight = order_weight * stat.count
                weighted_total += stat.mean() * weight
                weighted_count += weight

            if weighted_count > 0:
                return weighted_total / weighted_count, weighted_count

        return None, 0.0

    def _keys(
        self, active_sets: list[set[int]], order: int
    ) -> list[tuple[int, ...]]:
        return SparseContextMemory._keys(self, active_sets, order)


class SparseKNNRegressor:
    """Nearest-neighbor regression over SSTD-encoded multi-field contexts."""

    def __init__(self, encoder: SSTDDiscreteEncoder) -> None:
        self.encoder = encoder
        self._samples: list[tuple[frozenset[tuple[int, int]], float]] = []
        self._inverted: dict[tuple[int, int], list[int]] = defaultdict(list)

    def observe(self, fields: list[str], target: float) -> None:
        features = self._features(fields)
        sample_index = len(self._samples)
        self._samples.append((features, target))
        for feature in features:
            self._inverted[feature].append(sample_index)

    def predict(
        self,
        fields: list[str],
        neighbors: int = 25,
        min_overlap: int = 1,
        max_candidates: int = 2000,
    ) -> tuple[float | None, int]:
        features = self._features(fields)
        candidate_counts: dict[int, int] = {}
        for feature in features:
            for sample_index in self._inverted.get(feature, []):
                candidate_counts[sample_index] = candidate_counts.get(sample_index, 0) + 1

        if not candidate_counts:
            return None, 0

        ranked_candidates = sorted(
            candidate_counts.items(), key=lambda item: item[1], reverse=True
        )[:max_candidates]

        scored: list[tuple[float, float]] = []
        for sample_index, rough_overlap in ranked_candidates:
            if rough_overlap < min_overlap:
                continue
            sample_features, target = self._samples[sample_index]
            overlap = len(features.intersection(sample_features))
            if overlap >= min_overlap:
                scored.append((float(overlap), target))

        if not scored:
            return None, 0

        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[:neighbors]
        total_weight = sum(weight for weight, _target in top)
        if total_weight <= 0:
            return None, 0
        prediction = sum(weight * target for weight, target in top) / total_weight
        return prediction, len(top)

    def _features(self, fields: list[str]) -> frozenset[tuple[int, int]]:
        features: set[tuple[int, int]] = set()
        for field_index, field in enumerate(fields):
            for column in self.encoder.encode(field).columns:
                features.add((field_index, column))
        return frozenset(features)
