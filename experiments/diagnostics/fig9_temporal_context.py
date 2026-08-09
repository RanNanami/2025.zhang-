"""Bounded, causal temporal summaries for Fig.9 candidates.

This module is deliberately diagnostic-only.  It records the temporal state
that was available before a prediction, but never participates in matching,
selection, competition, learning, or RNG calls.  One fixed-width row is
written per candidate; source histories are reduced in O(number of
contributors) time and are never expanded into source-pair rows.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .fig9_candidate_context_oracle import stable_candidate_event_id


VERSION = "fig9-temporal-context-v1"
TRAJECTORY_ACTUAL = "actual_observation"
TRAJECTORY_AUTONOMOUS = "autonomous_rollout"
FIELDS = ("passenger", "time", "weekday")
RECENT_WINDOWS = (1, 3, 5, 10, 20)
NA = "NA"


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _mean(values: Sequence[float]) -> float | str:
    return statistics.fmean(values) if values else NA


def _median(values: Sequence[float]) -> float | str:
    return statistics.median(values) if values else NA


def _std(values: Sequence[float]) -> float | str:
    return statistics.pstdev(values) if len(values) > 1 else (0.0 if values else NA)


def _entropy(values: Sequence[float]) -> float | str:
    if not values:
        return NA
    counts: dict[float, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    total = len(values)
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _age_metrics(ages: Sequence[int]) -> dict[str, object]:
    ordered = sorted(int(age) for age in ages if int(age) >= 0)
    if not ordered:
        return {
            "mean_age": NA,
            "median_age": NA,
            "min_age": NA,
            "max_age": NA,
            "std_age": NA,
            "iqr_age": NA,
            "age_span": NA,
            "age_entropy": NA,
            "recent_1_fraction": 0.0,
            "recent_3_fraction": 0.0,
            "recent_5_fraction": 0.0,
            "recent_10_fraction": 0.0,
            "recent_20_fraction": 0.0,
            "recentness_concentration": 0.0,
            "temporal_compactness": NA,
            "recent_to_old_ratio": 0.0,
            "old_to_recent_ratio": 0.0,
            "adjacent_age_gap_mean": NA,
            "adjacent_age_gap_median": NA,
            "adjacent_age_gap_max": NA,
            "age_gap_entropy": NA,
            "recent_cluster_size": 0,
            "longest_recent_run": 0,
        }
    gaps = [right - left for left, right in zip(ordered, ordered[1:])]
    recent = {window: sum(age < window for age in ordered) for window in RECENT_WINDOWS}
    recent_fraction = {
        window: _ratio(count, len(ordered)) for window, count in recent.items()
    }
    old_count = sum(age >= 10 for age in ordered)
    recent_count = recent[5]
    recent_run = 0
    longest_run = 0
    for age in ordered:
        if age < 5:
            recent_run += 1
            longest_run = max(longest_run, recent_run)
        else:
            recent_run = 0
    span = ordered[-1] - ordered[0]
    age_std = statistics.pstdev(ordered) if len(ordered) > 1 else 0.0
    return {
        "mean_age": statistics.fmean(ordered),
        "median_age": statistics.median(ordered),
        "min_age": ordered[0],
        "max_age": ordered[-1],
        "std_age": age_std,
        "iqr_age": _quantile(ordered, 0.75) - _quantile(ordered, 0.25),
        "age_span": span,
        "age_entropy": _entropy(ordered),
        "recent_1_fraction": recent_fraction[1],
        "recent_3_fraction": recent_fraction[3],
        "recent_5_fraction": recent_fraction[5],
        "recent_10_fraction": recent_fraction[10],
        "recent_20_fraction": recent_fraction[20],
        "recentness_concentration": recent_fraction[5],
        "temporal_compactness": 1.0 / (1.0 + age_std),
        "recent_to_old_ratio": _ratio(recent_count, old_count),
        "old_to_recent_ratio": _ratio(old_count, recent_count),
        "adjacent_age_gap_mean": statistics.fmean(gaps) if gaps else 0.0,
        "adjacent_age_gap_median": statistics.median(gaps) if gaps else 0.0,
        "adjacent_age_gap_max": max(gaps, default=0),
        "age_gap_entropy": _entropy(gaps),
        "recent_cluster_size": recent[5],
        "longest_recent_run": longest_run,
    }


def source_id_fingerprint(source_ids: Iterable[int]) -> str:
    payload = ",".join(str(int(source)) for source in sorted(set(source_ids)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class _CsvAppendSink:
    path: Path
    compress: bool
    _handle: object | None = None
    _writer: csv.DictWriter | None = None
    row_count: int = 0

    def write(self, row: Mapping[str, object]) -> None:
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.compress:
                self._handle = gzip.open(self.path, "at", encoding="utf-8", newline="")
            else:
                self._handle = self.path.open("a", encoding="utf-8", newline="")
            needs_header = self.path.stat().st_size == 0
            self._writer = csv.DictWriter(
                self._handle,
                fieldnames=list(row.keys()),
                extrasaction="ignore",
            )
            if needs_header:
                self._writer.writeheader()
        self._writer.writerow(row)
        self._handle.flush()  # type: ignore[union-attr]
        self.row_count += 1

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()  # type: ignore[union-attr]


@dataclass
class TemporalContextTracker:
    """Keep causal actual/predicted activation state and bounded summaries."""

    actual_last_activation: dict[int, int] = field(default_factory=dict)
    autonomous_last_activation: dict[int, int] = field(default_factory=dict)
    actual_transition_count: int = 0
    autonomous_transition_count: int = 0
    candidate_row_count: int = 0
    actual_row_count: int = 0
    _sink: _CsvAppendSink | None = None
    _pending_rows: list[dict[str, object]] = field(default_factory=list)
    _flush_every: int = 256
    _current_record_index: int | None = None

    def enable_streaming(self, output_dir: Path, *, compress: bool = True) -> None:
        suffix = ".csv.gz" if compress else ".csv"
        self._sink = _CsvAppendSink(output_dir / f"temporal_candidate_trace{suffix}", compress)

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "version": VERSION,
            "actual_last_activation": {str(key): value for key, value in self.actual_last_activation.items()},
            "actual_transition_count": self.actual_transition_count,
            "candidate_row_count": self.candidate_row_count,
            "actual_row_count": self.actual_row_count,
        }

    @classmethod
    def from_checkpoint_payload(cls, payload: object) -> "TemporalContextTracker":
        if not isinstance(payload, Mapping) or payload.get("version") != VERSION:
            return cls()
        return cls(
            actual_last_activation={int(key): int(value) for key, value in dict(payload.get("actual_last_activation", {})).items()},
            actual_transition_count=_int(payload.get("actual_transition_count")),
            candidate_row_count=_int(payload.get("candidate_row_count")),
            actual_row_count=_int(payload.get("actual_row_count")),
        )

    def begin_record(self, record_index: int) -> None:
        self._current_record_index = record_index
        self.autonomous_last_activation = {}
        self.autonomous_transition_count = 0

    def record_actual_observation(self, winners: Iterable[int], record_index: int) -> None:
        for source in winners:
            self.actual_last_activation[int(source)] = int(record_index)
        self.actual_transition_count += 1

    def record_autonomous_state(self, active_cells: Iterable[int], horizon_step: int) -> None:
        for source in active_cells:
            self.autonomous_last_activation[int(source)] = int(horizon_step)
        self.autonomous_transition_count = max(self.autonomous_transition_count, int(horizon_step))

    def _emit(self, row: dict[str, object]) -> None:
        if self._sink is None:
            self._pending_rows.append(row)
            return
        self._sink.write(row)

    def _source_age_sets(
        self,
        source_ids: Sequence[int],
        *,
        record_index: int,
        trajectory_kind: str,
        horizon_step: int,
    ) -> tuple[list[int], list[int], list[int], int]:
        actual_ages: list[int] = []
        predicted_ages: list[int] = []
        active_ages: list[int] = []
        unknown = 0
        for source in source_ids:
            actual = self.actual_last_activation.get(source)
            predicted = self.autonomous_last_activation.get(source)
            if actual is not None and actual < record_index:
                actual_ages.append(record_index - actual)
            if predicted is not None and predicted < horizon_step:
                predicted_ages.append(horizon_step - predicted)
            if trajectory_kind == TRAJECTORY_ACTUAL:
                if actual is not None and actual < record_index:
                    active_ages.append(record_index - actual)
                else:
                    unknown += 1
            elif predicted is not None and predicted < horizon_step:
                active_ages.append(horizon_step - predicted)
            elif actual is not None and actual < record_index:
                # The first autonomous step legitimately starts from the
                # actual prefix.  Keep this separate from predicted history.
                active_ages.append(record_index - actual)
            else:
                unknown += 1
        return actual_ages, predicted_ages, active_ages, unknown

    def _base_row(
        self,
        *,
        trajectory_kind: str,
        actual_record_index: int,
        horizon_step: int,
        field: str,
        observed_encoded_column: object,
        candidate_target_column: int,
        candidate_target_neuron: int,
        segment_id: str,
        ordinal: int,
        candidate_score: object,
        source_ids: Sequence[int],
        contribution_items: Sequence[object],
        segment: object,
        candidate_time: object,
        target_columns: set[int],
        oracle_target_column: object,
        response_peak: object,
    ) -> dict[str, object]:
        actual_ages, predicted_ages, ages, unknown = self._source_age_sets(
            source_ids,
            record_index=actual_record_index,
            trajectory_kind=trajectory_kind,
            horizon_step=horizon_step,
        )
        age = _age_metrics(ages)
        actual_metrics = _age_metrics(actual_ages)
        predicted_metrics = _age_metrics(predicted_ages)
        delays = [
            float(getattr(item, "arrival_time")) - float(getattr(item, "source_time"))
            for item in contribution_items
            if _finite(getattr(item, "arrival_time", None)) is not None
            and _finite(getattr(item, "source_time", None)) is not None
        ]
        segment_age_values = [
            int(getattr(synapse, "age", 0))
            for synapse in getattr(segment, "synapses", {}).values()
        ]
        creation = getattr(segment, "creation_transition_index", None)
        reinforcement_count = int(
            getattr(segment, "scenario1_reinforcements", 0)
            + getattr(segment, "scenario2_reinforcements", 0)
        )
        result: dict[str, object] = {
            "row_unit": "CANDIDATE_TEMPORAL_SUMMARY",
            "diagnostic_version": VERSION,
            "temporal_semantics": "OFFLINE_POSTHOC_TEMPORAL; causal state frozen before selection",
            "candidate_event_id": stable_candidate_event_id(
                trajectory_kind=trajectory_kind,
                actual_record_index=actual_record_index,
                horizon_step=horizon_step,
                field=field,
                observed_encoded_column=observed_encoded_column,
                candidate_target_column=candidate_target_column,
                candidate_target_neuron=candidate_target_neuron,
                stable_segment_provenance_id=segment_id,
                candidate_generation_ordinal=ordinal,
                observed_event_time=candidate_time,
            ),
            "actual_record_index": actual_record_index,
            "trajectory_kind": trajectory_kind,
            "horizon_step": horizon_step,
            "field": field,
            "candidate_target_column": candidate_target_column,
            "candidate_target_neuron": candidate_target_neuron,
            "stable_segment_provenance_id": segment_id,
            "oracle_target_column": oracle_target_column,
            "is_target_column_candidate": candidate_target_column in target_columns,
            "candidate_score": candidate_score,
            "candidate_score_available": _finite(candidate_score) is not None,
            "contributor_count": len(source_ids),
            "actual_history_source_count": len(actual_ages),
            "predicted_history_source_count": len(predicted_ages),
            "unknown_history_source_count": unknown,
            "source_id_fingerprint": source_id_fingerprint(source_ids),
            "mean_actual_age": actual_metrics["mean_age"],
            "median_actual_age": actual_metrics["median_age"],
            "min_actual_age": actual_metrics["min_age"],
            "max_actual_age": actual_metrics["max_age"],
            "std_actual_age": actual_metrics["std_age"],
            "iqr_actual_age": actual_metrics["iqr_age"],
            "actual_age_span": actual_metrics["age_span"],
            "mean_predicted_age": predicted_metrics["mean_age"],
            "median_predicted_age": predicted_metrics["median_age"],
            "mean_any_age": age["mean_age"],
            "median_any_age": age["median_age"],
            "min_any_age": age["min_age"],
            "max_any_age": age["max_age"],
            "std_any_age": age["std_age"],
            "iqr_any_age": age["iqr_age"],
            "actual_age_entropy": actual_metrics["age_entropy"],
            "age_entropy": age["age_entropy"],
            "actual_age_span": actual_metrics["age_span"],
            "recent_1_fraction": age["recent_1_fraction"],
            "recent_3_fraction": age["recent_3_fraction"],
            "recent_5_fraction": age["recent_5_fraction"],
            "recent_10_fraction": age["recent_10_fraction"],
            "recent_20_fraction": age["recent_20_fraction"],
            "recentness_concentration": age["recentness_concentration"],
            "temporal_compactness": age["temporal_compactness"],
            # Source IDs are not an activation order.  Without an explicit
            # model template, reporting a monotonicity score would invent
            # temporal structure.
            "source_age_monotonicity": NA,
            "activation_order_consistency": NA,
            "historical_order_agreement": "TEMPORAL_TEMPLATE_UNAVAILABLE",
            "recent_to_old_ratio": age["recent_to_old_ratio"],
            "old_to_recent_ratio": age["old_to_recent_ratio"],
            "adjacent_age_gap_mean": age["adjacent_age_gap_mean"],
            "adjacent_age_gap_median": age["adjacent_age_gap_median"],
            "adjacent_age_gap_max": age["adjacent_age_gap_max"],
            "age_gap_entropy": age["age_gap_entropy"],
            "recent_cluster_size": age["recent_cluster_size"],
            "longest_recent_run": age["longest_recent_run"],
            "segment_creation_index": creation if creation is not None else NA,
            "segment_age": max(segment_age_values, default=0),
            "time_since_last_segment_reinforcement": NA,
            "segment_reinforcement_count": reinforcement_count,
            "segment_recent_reinforcement_count": NA,
            "candidate_predicted_time": candidate_time,
            "response_peak_time": response_peak,
            "mean_contributor_delay": _mean(delays),
            "std_contributor_delay": _std(delays),
            "delay_span": (max(delays) - min(delays)) if delays else NA,
            "recurrent_trajectory_divergence": trajectory_kind == TRAJECTORY_AUTONOMOUS and horizon_step > 1,
        }
        return result

    def record_autonomous_candidates(
        self,
        *,
        model: object,
        registry: object | None,
        actual_record_index: int,
        horizon_step: int,
        field_for_column,
        target_columns: Iterable[int],
        target_column_by_field: Mapping[str, int],
        input_index: int,
        input_timestamp: str,
        candidates: Mapping[int, Sequence[object]],
        emitted_candidate_ids: Iterable[int] = (),
    ) -> None:
        target_set = {int(value) for value in target_columns}
        emitted = set(int(value) for value in emitted_candidate_ids)
        ordinal = 0
        for column in sorted(candidates):
            for candidate in candidates[column]:
                positive = tuple(
                    item for item in getattr(candidate, "crossing_synapse_contributions", ())
                    if float(getattr(item, "psp_contribution", 0.0)) > 0.0
                )
                source_ids = tuple(sorted({int(item.source_cell_id) for item in positive}))
                if not source_ids:
                    source_ids = tuple(sorted(int(source) for source in getattr(candidate.segment, "synapses", {})))
                origin = registry.provenance_for(candidate.segment) if registry is not None else None
                segment_id = origin.segment_provenance_id if origin is not None else ""
                decision_id = id(candidate)
                row = self._base_row(
                    trajectory_kind=TRAJECTORY_AUTONOMOUS,
                    actual_record_index=actual_record_index,
                    horizon_step=horizon_step,
                    field=field_for_column(int(column)),
                    observed_encoded_column="",
                    candidate_target_column=int(column),
                    candidate_target_neuron=int(candidate.neuron_index),
                    segment_id=segment_id,
                    ordinal=ordinal,
                    candidate_score=candidate.score,
                    source_ids=source_ids,
                    contribution_items=positive,
                    segment=candidate.segment,
                    candidate_time=candidate.time,
                    target_columns=target_set,
                    oracle_target_column=target_column_by_field.get(field_for_column(int(column)), NA),
                    response_peak=(candidate.peak_dendritic_potential if candidate.peak_dendritic_potential is not None else NA),
                )
                row.update({
                    "input_index": input_index,
                    "input_timestamp": input_timestamp,
                    "emitted_candidate": decision_id in emitted,
                    "source_context_domain": (
                        "actual_prefix_or_autonomous_suffix"
                    ),
                })
                self._emit(row)
                self.candidate_row_count += 1
                ordinal += 1

    def record_actual_source_rows(
        self,
        *,
        source_rows: Sequence[Mapping[str, object]],
        actual_record_index: int,
        target_field_by_column,
        l_match: int,
    ) -> None:
        groups: dict[tuple[str, int, int, str], list[Mapping[str, object]]] = defaultdict(list)
        for row in source_rows:
            if not _bool(row.get("contributes_to_actual_overlap")):
                continue
            key = (
                str(row.get("segment_provenance_id", "")),
                _int(row.get("encoded_column")),
                _int(row.get("segment_target_neuron")),
                str(row.get("field", "")),
            )
            groups[key].append(row)
        for ordinal, (key, rows) in enumerate(sorted(groups.items())):
            segment_id, column, neuron, field = key
            source_ids = tuple(sorted({_int(row.get("source_cell_stable_id")) for row in rows}))
            ages = [
                actual_record_index - self.actual_last_activation[source]
                for source in source_ids
                if source in self.actual_last_activation
                and self.actual_last_activation[source] < actual_record_index
            ]
            age = _age_metrics(ages)
            delays = [
                _finite(row.get("source_synaptic_delay"))
                for row in rows
                if _finite(row.get("source_synaptic_delay")) is not None
            ]
            delays = [float(value) for value in delays if value is not None]
            overlap_count = len(rows)
            row: dict[str, object] = {
                "row_unit": "CANDIDATE_TEMPORAL_SUMMARY",
                "diagnostic_version": VERSION,
                "temporal_semantics": "OFFLINE_POSTHOC_TEMPORAL; actual pre-observation state",
                "candidate_event_id": stable_candidate_event_id(
                    trajectory_kind=TRAJECTORY_ACTUAL,
                    actual_record_index=actual_record_index,
                    horizon_step=0,
                    field=field,
                    observed_encoded_column=column,
                    candidate_target_column=column,
                    candidate_target_neuron=neuron,
                    stable_segment_provenance_id=segment_id,
                    candidate_generation_ordinal=ordinal,
                    observed_event_time=rows[0].get("actual_observation_time", ""),
                ),
                "actual_record_index": actual_record_index,
                "trajectory_kind": TRAJECTORY_ACTUAL,
                "horizon_step": 0,
                "field": field,
                "candidate_target_column": column,
                "candidate_target_neuron": neuron,
                "stable_segment_provenance_id": segment_id,
                "oracle_target_column": column,
                "is_target_column_candidate": True,
                "candidate_score": NA,
                "candidate_score_available": False,
                "contributor_count": len(source_ids),
                "actual_history_source_count": len(ages),
                "predicted_history_source_count": 0,
                "unknown_history_source_count": len(source_ids) - len(ages),
                "source_id_fingerprint": source_id_fingerprint(source_ids),
                "mean_actual_age": age["mean_age"],
                "median_actual_age": age["median_age"],
                "min_actual_age": age["min_age"],
                "max_actual_age": age["max_age"],
                "std_actual_age": age["std_age"],
                "iqr_actual_age": age["iqr_age"],
                "actual_age_span": age["age_span"],
                "mean_predicted_age": NA,
                "median_predicted_age": NA,
                "mean_any_age": age["mean_age"],
                "median_any_age": age["median_age"],
                "min_any_age": age["min_age"],
                "max_any_age": age["max_age"],
                "std_any_age": age["std_age"],
                "iqr_any_age": age["iqr_age"],
                "actual_age_entropy": age["age_entropy"],
                "age_entropy": age["age_entropy"],
                "recent_1_fraction": age["recent_1_fraction"],
                "recent_3_fraction": age["recent_3_fraction"],
                "recent_5_fraction": age["recent_5_fraction"],
                "recent_10_fraction": age["recent_10_fraction"],
                "recent_20_fraction": age["recent_20_fraction"],
                "recentness_concentration": age["recentness_concentration"],
                "temporal_compactness": age["temporal_compactness"],
                "source_age_monotonicity": NA,
                "activation_order_consistency": NA,
                "historical_order_agreement": "TEMPORAL_TEMPLATE_UNAVAILABLE",
                "recent_to_old_ratio": age["recent_to_old_ratio"],
                "old_to_recent_ratio": age["old_to_recent_ratio"],
                "adjacent_age_gap_mean": age["adjacent_age_gap_mean"],
                "adjacent_age_gap_median": age["adjacent_age_gap_median"],
                "adjacent_age_gap_max": age["adjacent_age_gap_max"],
                "age_gap_entropy": age["age_gap_entropy"],
                "recent_cluster_size": age["recent_cluster_size"],
                "longest_recent_run": age["longest_recent_run"],
                "segment_creation_index": rows[0].get("segment_creation_transition_index", NA),
                "segment_age": rows[0].get("segment_age", NA),
                "time_since_last_segment_reinforcement": NA,
                "segment_reinforcement_count": NA,
                "segment_recent_reinforcement_count": NA,
                "candidate_predicted_time": NA,
                "response_peak_time": NA,
                "mean_contributor_delay": _mean(delays),
                "std_contributor_delay": _std(delays),
                "delay_span": max(delays) - min(delays) if delays else NA,
                "overlap_count": overlap_count,
                "L2_ONLY_STRICT": overlap_count == 2,
                "L2_RELAXED_VS_L4": overlap_count in {2, 3},
                "L3_ONLY_VS_L4": overlap_count == 3,
                "match_regime": (
                    "L2_ONLY_STRICT" if overlap_count == 2 else
                    "L3_ONLY_VS_L4" if overlap_count == 3 else
                    "OVERLAP_OTHER"
                ),
                "recurrent_trajectory_divergence": False,
                "source_context_domain": "actual_observation_prefix",
            }
            self._emit(row)
            self.actual_row_count += 1

    def close(self) -> None:
        if self._sink is None and self._pending_rows:
            raise RuntimeError("temporal tracker rows were not assigned an output sink")
        if self._sink is not None:
            self._sink.close()

    def pending_rows(self) -> list[dict[str, object]]:
        rows = self._pending_rows
        self._pending_rows = []
        return rows


def read_temporal_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def protocol() -> dict[str, object]:
    return {
        "version": VERSION,
        "diagnostic_only": True,
        "default_enabled": False,
        "row_unit": "CANDIDATE_TEMPORAL_SUMMARY",
        "complexity": "O(number_of_contributors) per candidate",
        "trajectory_kinds": [TRAJECTORY_ACTUAL, TRAJECTORY_AUTONOMOUS],
        "causal_state": "records < current prediction; current observation updates after prediction",
        "source_ordering": "TEMPORAL_TEMPLATE_UNAVAILABLE unless model exposes an explicit template",
        "future_information_used_for_selection": False,
        "ground_truth_used_for_selection": False,
        "strict_default_changed": False,
    }
