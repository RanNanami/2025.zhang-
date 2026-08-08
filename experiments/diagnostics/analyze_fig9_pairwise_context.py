"""Offline source-pair coherence audit for Fig.9.

The analyzer consumes the read-only candidate/context traces produced by the
strict runner.  Pair statistics are computed from actual pre-match contexts
strictly before the candidate's record index.  They are never fed back into
matching, selection, competition, learning, or RNG state.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import sqlite3
import statistics
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterable, Mapping, Sequence


VERSION = "fig9-pairwise-context-coherence-v1"
ROW_UNIT_PAIR = "SOURCE_PAIR"
ROW_UNIT_CANDIDATE = "CANDIDATE_SEGMENT"
TRAJECTORY_ACTUAL = "actual_observation"
TRAJECTORY_AUTONOMOUS = "autonomous_rollout"
FIELDS = ("passenger", "time", "weekday")
RECENT_WINDOW = 20
PMI_ALPHA = 1.0
SUPPORT_BUCKETS = ("0", "1", "2-3", "4-7", "8+")
OUTPUT_SUBDIRS = (
    "identity", "pairs", "candidate", "paired", "field", "actual_relaxed",
    "autonomous", "temporal", "reinforcement", "score_analysis", "mechanism",
    "optional_500", "logs", "checkpoints", "analysis",
)


def _number(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def _median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_scale = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_scale = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    if not left_scale or not right_scale:
        return None
    return numerator / (left_scale * right_scale)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["row_unit"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_gzip_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class _GzipCsvSink:
    """Write a row stream without retaining the complete trace in memory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = None
        self._writer = None
        self.row_count = 0

    def write(self, row: Mapping[str, object]) -> None:
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = gzip.open(self.path, "wt", encoding="utf-8", newline="")
            fields = list(row.keys())
            self._writer = csv.DictWriter(self._handle, fieldnames=fields, extrasaction="ignore")
            self._writer.writeheader()
        self._writer.writerow(row)
        self.row_count += 1

    def close(self) -> None:
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(self.path, "wt", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow(["row_unit"])
        else:
            self._handle.close()


class _CsvSink(_GzipCsvSink):
    """Plain-text counterpart used for legacy CSV output names."""

    def write(self, row: Mapping[str, object]) -> None:
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("w", encoding="utf-8", newline="")
            fields = list(row.keys())
            self._writer = csv.DictWriter(self._handle, fieldnames=fields, extrasaction="ignore")
            self._writer.writeheader()
        self._writer.writerow(row)
        self.row_count += 1

    def close(self) -> None:
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("row_unit\n", encoding="utf-8")
        else:
            self._handle.close()


def _write_filtered_gzip_csv(source: Path, target: Path, predicate) -> int:
    """Copy selected rows from a gzip CSV using bounded memory."""

    sink = _GzipCsvSink(target)
    count = 0
    with gzip.open(source, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if predicate(row):
                sink.write(row)
                count += 1
    sink.close()
    return count


def canonical_pair(source_a: int, source_b: int) -> tuple[int, int]:
    """Return an order-independent source pair."""

    left, right = int(source_a), int(source_b)
    if left == right:
        raise ValueError("a source pair must contain two distinct sources")
    return (left, right) if left < right else (right, left)


def stable_pair_key(source_a: int, source_b: int) -> str:
    """Build a deterministic pair key without row/object/RNG identity."""

    left, right = canonical_pair(source_a, source_b)
    return f"{left}:{right}"


def source_pairs(source_ids: Iterable[int]) -> list[tuple[int, int]]:
    unique = sorted({int(source) for source in source_ids})
    return list(itertools.combinations(unique, 2))


def support_bucket(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 7:
        return "4-7"
    return "8+"


def pmi_like(
    pair_count: int,
    source_a_count: int,
    source_b_count: int,
    event_count: int,
    *,
    alpha: float = PMI_ALPHA,
) -> float:
    """Compute the documented smoothed PMI-like statistic."""

    denominator = max(1.0, float(event_count) + 2.0 * alpha)
    p_pair = (float(pair_count) + alpha) / denominator
    p_a = (float(source_a_count) + alpha) / denominator
    p_b = (float(source_b_count) + alpha) / denominator
    return math.log(max(p_pair, 1e-300) / max(p_a * p_b, 1e-300))


def shrunk_pmi(pmi: float, support: int, *, k: float = 3.0) -> float:
    return pmi * support / (support + k) if support else 0.0


def _parse_ids(value: object) -> tuple[int, ...]:
    if value is None or str(value).strip() in {"", "nan", "None"}:
        return ()
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(sorted({int(item) for item in parsed}))


def _parse_metadata(value: object) -> dict[int, dict[str, object]]:
    if value is None or str(value).strip() in {"", "nan", "None"}:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {_int(key): item for key, item in parsed.items() if isinstance(item, dict)}


def _source_metadata_from_trace(rows: Sequence[Mapping[str, object]]) -> dict[tuple[int, str], dict[int, dict[str, object]]]:
    result: dict[tuple[int, str], dict[int, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        record = _int(row.get("actual_record_index"))
        segment = str(row.get("segment_provenance_id", ""))
        source = _int(row.get("source_cell_stable_id", row.get("source_id")))
        if not segment:
            continue
        result[(record, segment)][source] = {
            "source_column": row.get("source_column", ""),
            "source_neuron": row.get("source_neuron", ""),
            "source_field": row.get("source_field", ""),
            "incidence": row.get("source_live_segment_incidence", row.get("source_segment_incidence", "")),
            "idf": row.get("source_idf", ""),
            "incidence_bin": row.get("source_incidence_bin", ""),
            "age": row.get("source_age", ""),
            "origin_labels": str(row.get("source_origin_type", "")).split(";") if row.get("source_origin_type") else [],
        }
    return result


def _segment_source_sets(
    candidate_rows: Sequence[Mapping[str, object]],
    source_meta: Mapping[tuple[int, str], Mapping[int, Mapping[str, object]]],
) -> dict[tuple[str, int, int], dict[str, dict[str, object]]]:
    groups: dict[tuple[str, int, int], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in candidate_rows:
        trajectory = str(row.get("trajectory_kind", ""))
        record = _int(row.get("actual_record_index"))
        horizon = _int(row.get("horizon_step"))
        segment_id = str(row.get("stable_segment_provenance_id", ""))
        segment_key = segment_id or str(row.get("candidate_event_id", ""))
        if not segment_key:
            continue
        ids = _parse_ids(row.get("source_ids"))
        overlap_ids = _candidate_source_ids(row)
        if trajectory == TRAJECTORY_ACTUAL:
            ids = tuple(source_meta.get((record, segment_id), {}).keys()) or ids
        groups[(trajectory, record, horizon)][segment_key] = {
            "source_ids": ids,
            "overlap_source_ids": overlap_ids,
            "target_column": row.get("candidate_target_column", row.get("column", "")),
            "target_neuron": row.get("candidate_target_neuron", row.get("neuron", "")),
            "field": row.get("field", ""),
        }
    return groups


def _event_contexts(
    event_rows: Sequence[Mapping[str, object]],
    source_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[int, set[int]], str]:
    contexts: dict[int, set[int]] = defaultdict(set)
    for row in event_rows:
        record = _int(row.get("actual_record_index"))
        contexts[record].update(_parse_ids(row.get("context_source_ids")))
    if contexts and any(contexts.values()):
        return contexts, "FULL_PREMATCH_CONTEXT"
    # Compatibility fallback for older formal traces. This only observes
    # sources that appear in a live segment, so the report marks it explicitly.
    for row in source_rows:
        if _truth(row.get("source_in_active_cells")):
            contexts[_int(row.get("actual_record_index"))].add(
                _int(row.get("source_cell_stable_id", row.get("source_id")))
            )
    return contexts, "OBSERVED_ACTIVE_SEGMENT_SOURCE_SUBSET"


def _source_stat(
    source: int,
    metadata: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    return dict(metadata.get(source, {}))


def _pair_origin_labels(
    source_a: int,
    source_b: int,
    metadata: Mapping[int, Mapping[str, object]],
) -> list[str]:
    labels: set[str] = set()
    for source in (source_a, source_b):
        values = metadata.get(source, {}).get("origin_labels", ())
        if isinstance(values, str):
            labels.update(item for item in values.split(";") if item)
        else:
            labels.update(str(item) for item in values if str(item))
    return sorted(labels) or ["UNKNOWN"]


def _candidate_source_ids(row: Mapping[str, object]) -> tuple[int, ...]:
    return _parse_ids(row.get("overlap_source_ids")) or _parse_ids(row.get("source_ids"))


def _candidate_group_key(row: Mapping[str, object]) -> tuple[int, int, str]:
    return (
        _int(row.get("actual_record_index")),
        _int(row.get("horizon_step")),
        str(row.get("field", "")),
    )


def build_pair_trace(
    candidate_rows: Sequence[Mapping[str, object]],
    *,
    event_rows: Sequence[Mapping[str, object]],
    source_rows: Sequence[Mapping[str, object]],
    row_callback=None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Expand candidates into deterministic source-pair rows.

    The optional callback is used by formal audits to stream rows directly to
    disk.  With no callback this keeps the historical list-returning API used
    by the small unit tests.
    """

    source_meta_by_segment = _source_metadata_from_trace(source_rows)
    contexts, context_scope = _event_contexts(event_rows, source_rows)
    segment_sets = _segment_source_sets(candidate_rows, source_meta_by_segment)
    rows_by_record: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in candidate_rows:
        rows_by_record[_int(row.get("actual_record_index"))].append(row)

    actual_event_count = max(contexts.keys(), default=-1) + 1
    source_counts: Counter[int] = Counter()
    pair_counts: Counter[tuple[int, int]] = Counter()
    recent_source_events: deque[tuple[int, set[int], set[tuple[int, int]]] ] = deque()
    recent_source_counts: Counter[int] = Counter()
    recent_pair_counts: Counter[tuple[int, int]] = Counter()
    trace: list[dict[str, object]] = []
    emitted_pair_keys: set[str] | None = set() if row_callback is None else None
    emitted_row_count = 0

    max_record = max(
        max(rows_by_record, default=-1),
        max(contexts, default=-1),
    )
    for record in range(max_record + 1):
        current_context = contexts.get(record, set())
        current_pairs = source_pairs(current_context)
        group_pair_sets: dict[tuple[str, int, int], dict[tuple[int, int], dict[str, object]]] = defaultdict(dict)
        # Only build live-segment statistics for pairs that are actually
        # present in a candidate overlap at this record.  Bit masks keep the
        # distinct column/neuron/field incidence bounded even when many
        # candidates share the same source pair.
        requested_by_group: dict[tuple[str, int, int], set[tuple[int, int]]] = defaultdict(set)
        for row in rows_by_record[record]:
            group_key = (str(row.get("trajectory_kind", "")), record, _int(row.get("horizon_step")))
            requested_by_group[group_key].update(source_pairs(_candidate_source_ids(row)))
        for group_key, requested_pairs in requested_by_group.items():
            segments = segment_sets.get(group_key, {})
            for segment in segments.values():
                for pair in source_pairs(segment["overlap_source_ids"]):
                    if pair not in requested_pairs:
                        continue
                    item = group_pair_sets[group_key].setdefault(pair, {"segment_count": 0, "column_mask": 0, "neuron_mask": 0, "field_mask": 0})
                    item["segment_count"] = int(item["segment_count"]) + 1
                    column = _int(segment["target_column"])
                    neuron = _int(segment["target_neuron"])
                    field = str(segment["field"])
                    if 0 <= column < 4096:
                        item["column_mask"] = int(item["column_mask"]) | (1 << column)
                    if 0 <= neuron < 131072:
                        item["neuron_mask"] = int(item["neuron_mask"]) | (1 << (column * 1024 + neuron % 1024))
                    item["field_mask"] = int(item["field_mask"]) | (1 << (FIELDS.index(field) if field in FIELDS else len(FIELDS)))

        for row in sorted(rows_by_record[record], key=lambda item: (_int(item.get("horizon_step")), str(item.get("candidate_event_id", "")))):
            trajectory = str(row.get("trajectory_kind", ""))
            horizon = _int(row.get("horizon_step"))
            group_key = (trajectory, record, horizon)
            pair_group = group_pair_sets.get(group_key, {})
            metadata = _parse_metadata(row.get("overlap_source_metadata_json"))
            if trajectory == TRAJECTORY_ACTUAL:
                metadata = dict(source_meta_by_segment.get((record, str(row.get("stable_segment_provenance_id", ""))), {})) or metadata
            overlap_ids = _candidate_source_ids(row)
            for source_a, source_b in source_pairs(overlap_ids):
                stat = pair_group.get((source_a, source_b), {"segments": set(), "columns": set(), "neurons": set(), "fields": set()})
                meta_a = _source_stat(source_a, metadata)
                meta_b = _source_stat(source_b, metadata)
                source_a_field = str(meta_a.get("source_field", ""))
                source_b_field = str(meta_b.get("source_field", ""))
                source_a_column = str(meta_a.get("source_column", ""))
                source_b_column = str(meta_b.get("source_column", ""))
                source_a_neuron = str(meta_a.get("source_neuron", ""))
                source_b_neuron = str(meta_b.get("source_neuron", ""))
                pair_count = pair_counts[(source_a, source_b)]
                source_a_count = source_counts[source_a]
                source_b_count = source_counts[source_b]
                # The causal statistic only sees contexts from records < t.
                pmi = pmi_like(pair_count, source_a_count, source_b_count, record)
                recent_pair = recent_pair_counts[(source_a, source_b)]
                row_pair = {
                    "row_unit": ROW_UNIT_PAIR,
                    "pair_key": stable_pair_key(source_a, source_b),
                    "candidate_event_id": row.get("candidate_event_id", ""),
                    "actual_record_index": record,
                    "trajectory_kind": trajectory,
                    "horizon_step": horizon,
                    "field": row.get("field", ""),
                    "candidate_target_column": row.get("candidate_target_column", row.get("column", "")),
                    "candidate_target_neuron": row.get("candidate_target_neuron", row.get("neuron", "")),
                    "stable_segment_provenance_id": row.get("stable_segment_provenance_id", ""),
                    "selected_intracolumn": row.get("selected_intracolumn", ""),
                    "selected_after_intercolumn": row.get("selected_after_intercolumn", ""),
                    "reinforced": row.get("reinforced", ""),
                    "original_candidate_score": row.get("original_candidate_score", ""),
                    "response_peak": row.get("response_peak", ""),
                    "contributor_count": row.get("contributor_count", ""),
                    "overlap_mean_idf": row.get("overlap_mean_idf", ""),
                    "overlap_very_common_fraction": row.get("overlap_very_common_fraction", ""),
                    "overlap_common_fraction": row.get("overlap_common_fraction", ""),
                    "oracle_target_column": row.get("oracle_target_column", ""),
                    "is_target_column_candidate": row.get("is_target_column_candidate", ""),
                    "L2_ONLY_STRICT": row.get("L2_ONLY_STRICT", ""),
                    "L2_RELAXED_VS_L4": row.get("L2_RELAXED_VS_L4", ""),
                    "L3_ONLY_VS_L4": row.get("L3_ONLY_VS_L4", ""),
                    "match_regime": row.get("match_regime", ""),
                    "reinforcement_count": row.get("reinforcement_count", ""),
                    "source_a": source_a,
                    "source_b": source_b,
                    "source_a_field": source_a_field,
                    "source_b_field": source_b_field,
                    "source_a_column": source_a_column,
                    "source_b_column": source_b_column,
                    "source_a_neuron": source_a_neuron,
                    "source_b_neuron": source_b_neuron,
                    "source_a_incidence": meta_a.get("incidence", ""),
                    "source_b_incidence": meta_b.get("incidence", ""),
                    "source_a_idf": meta_a.get("idf", ""),
                    "source_b_idf": meta_b.get("idf", ""),
                    "source_a_age": meta_a.get("age", ""),
                    "source_b_age": meta_b.get("age", ""),
                    "pair_actual_cooccurrence_count": pair_count,
                    "pair_recent_actual_cooccurrence_count": recent_pair,
                    "pair_segment_cooccurrence_count": int(stat.get("segment_count", 0)),
                    "pair_target_column_incidence": int(stat.get("column_mask", 0)).bit_count(),
                    "pair_target_neuron_incidence": int(stat.get("neuron_mask", 0)).bit_count(),
                    "pair_field_incidence": int(stat.get("field_mask", 0)).bit_count(),
                    "pair_pmi_like": pmi,
                    "pair_shrunk_pmi": shrunk_pmi(pmi, pair_count),
                    "pair_normalized_cooccurrence": _safe_ratio(pair_count, math.sqrt(max(1, source_a_count * source_b_count))),
                    "pair_specificity": 1.0 / (1.0 + int(stat.get("segment_count", 0))),
                    "same_field_pair": bool(source_a_field and source_a_field == source_b_field),
                    "cross_field_pair": bool(source_a_field and source_b_field and source_a_field != source_b_field),
                    "same_column_pair": bool(source_a_column and source_a_column == source_b_column),
                    "same_neuron_pair": bool(source_a_column and source_a_column == source_b_column and source_a_neuron == source_b_neuron),
                    "pair_origin_labels": "|".join(_pair_origin_labels(source_a, source_b, metadata)),
                    "pair_support_bucket": support_bucket(pair_count),
                    "pair_statistic_scope": "ONLINE_CAUSAL_PREMATCH",
                    "actual_context_scope": context_scope,
                    "causal_event_count": record,
                    "step1_error": row.get("step1_error", ""),
                    "step_error": row.get("step_error", ""),
                }
                if emitted_pair_keys is not None:
                    emitted_pair_keys.add(str(row_pair["pair_key"]))
                emitted_row_count += 1
                if row_callback is None:
                    trace.append(row_pair)
                else:
                    row_callback(row_pair)

        if current_context:
            event_pair_set = set(current_pairs)
            for source in current_context:
                source_counts[source] += 1
            for pair in current_pairs:
                pair_counts[pair] += 1
            recent_source_events.append((record, set(current_context), event_pair_set))
            for source in current_context:
                recent_source_counts[source] += 1
            for pair in current_pairs:
                recent_pair_counts[pair] += 1
            while recent_source_events and recent_source_events[0][0] <= record - RECENT_WINDOW:
                _, old_sources, old_pairs = recent_source_events.popleft()
                for source in old_sources:
                    recent_source_counts[source] -= 1
                for pair in old_pairs:
                    recent_pair_counts[pair] -= 1

    return trace, {
        "version": VERSION,
        "row_unit": ROW_UNIT_PAIR,
        "pmi_smoothing_alpha": PMI_ALPHA,
        "pmi_shrinkage_k": 3.0,
        "recent_window_records": RECENT_WINDOW,
        "actual_context_scope": context_scope,
        "actual_event_count": actual_event_count,
        "unique_pairs": len(emitted_pair_keys) if emitted_pair_keys is not None else 0,
        "pair_rows": emitted_row_count,
    }


def _candidate_pair_summary(pair_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in pair_rows:
        grouped[str(row.get("candidate_event_id", ""))].append(row)
    result: list[dict[str, object]] = []
    for candidate_id, rows in sorted(grouped.items()):
        first = rows[0]
        def values(name: str) -> list[float]:
            return [_number(row.get(name)) for row in rows]
        result.append(
            {
                "row_unit": ROW_UNIT_CANDIDATE,
                "candidate_event_id": candidate_id,
                "actual_record_index": first.get("actual_record_index", ""),
                "trajectory_kind": first.get("trajectory_kind", ""),
                "horizon_step": first.get("horizon_step", ""),
                "field": first.get("field", ""),
                "candidate_target_column": first.get("candidate_target_column", ""),
                "stable_segment_provenance_id": first.get("stable_segment_provenance_id", ""),
                "oracle_target_column": first.get("oracle_target_column", ""),
                "is_target_column_candidate": first.get("is_target_column_candidate", ""),
                "original_candidate_score": first.get("original_candidate_score", ""),
                "selected_intracolumn": first.get("selected_intracolumn", ""),
                "selected_after_intercolumn": first.get("selected_after_intercolumn", ""),
                "L2_ONLY_STRICT": first.get("L2_ONLY_STRICT", ""),
                "L2_RELAXED_VS_L4": first.get("L2_RELAXED_VS_L4", ""),
                "L3_ONLY_VS_L4": first.get("L3_ONLY_VS_L4", ""),
                "match_regime": first.get("match_regime", ""),
                "reinforcement_count": first.get("reinforcement_count", ""),
                "step_error": first.get("step_error", ""),
                "overlap_mean_idf": first.get("overlap_mean_idf", ""),
                "overlap_very_common_fraction": first.get("overlap_very_common_fraction", ""),
                "overlap_common_fraction": first.get("overlap_common_fraction", ""),
                "pair_count": len(rows),
                "pair_mean_pmi": _mean(values("pair_pmi_like")),
                "pair_median_pmi": _median(values("pair_pmi_like")),
                "pair_max_pmi": max(values("pair_pmi_like"), default=0.0),
                "pair_min_pmi": min(values("pair_pmi_like"), default=0.0),
                "pair_mean_shrunk_pmi": _mean(values("pair_shrunk_pmi")),
                "pair_mean_cooccurrence": _mean(values("pair_actual_cooccurrence_count")),
                "pair_max_cooccurrence": max(values("pair_actual_cooccurrence_count"), default=0.0),
                "pair_mean_recent_cooccurrence": _mean(values("pair_recent_actual_cooccurrence_count")),
                "pair_mean_specificity": _mean(values("pair_specificity")),
                "pair_mean_normalized_cooccurrence": _mean(values("pair_normalized_cooccurrence")),
                "pair_same_field_fraction": _mean([float(_truth(row.get("same_field_pair"))) for row in rows]),
                "pair_cross_field_fraction": _mean([float(_truth(row.get("cross_field_pair"))) for row in rows]),
                "pair_actual_actual_fraction": _mean([float("ACTUAL" in str(row.get("pair_origin_labels", ""))) for row in rows]),
                "pair_recent_fraction": _safe_ratio(sum(_number(row.get("pair_recent_actual_cooccurrence_count")) > 0 for row in rows), len(rows)),
                "pair_high_pmi_fraction": _safe_ratio(sum(_number(row.get("pair_pmi_like")) > 0 for row in rows), len(rows)),
                "pair_unique_history_fraction": _safe_ratio(sum(_int(row.get("pair_actual_cooccurrence_count")) == 1 for row in rows), len(rows)),
            }
        )
    return result


_PAIR_NUMERIC_FEATURES = (
    "pair_pmi_like", "pair_shrunk_pmi", "pair_actual_cooccurrence_count",
    "pair_recent_actual_cooccurrence_count", "pair_specificity", "pair_normalized_cooccurrence",
)
_PAIR_BOOLEAN_FEATURES = (
    "same_field_pair", "cross_field_pair",
)
_CANDIDATE_FIRST_FIELDS = (
    "actual_record_index", "trajectory_kind", "horizon_step", "field",
    "candidate_target_column", "stable_segment_provenance_id", "oracle_target_column",
    "is_target_column_candidate", "original_candidate_score", "selected_intracolumn",
    "selected_after_intercolumn", "L2_ONLY_STRICT", "L2_RELAXED_VS_L4",
    "L3_ONLY_VS_L4", "match_regime", "reinforcement_count", "step_error",
    "overlap_mean_idf", "overlap_very_common_fraction", "overlap_common_fraction",
)


def _new_candidate_aggregate(row: Mapping[str, object]) -> dict[str, object]:
    state: dict[str, object] = {"first": {name: row.get(name, "") for name in _CANDIDATE_FIRST_FIELDS}, "count": 0}
    for name in _PAIR_NUMERIC_FEATURES:
        state[f"sum:{name}"] = 0.0
        state[f"min:{name}"] = float("inf")
        state[f"max:{name}"] = float("-inf")
    state["pmi_values"] = []
    for name in _PAIR_BOOLEAN_FEATURES:
        state[f"true:{name}"] = 0
    state["actual_actual_count"] = 0
    state["unique_history_count"] = 0
    state["high_pmi_count"] = 0
    state["recent_positive_count"] = 0
    return state


def _update_candidate_aggregate(state: dict[str, object], row: Mapping[str, object]) -> None:
    state["count"] = int(state["count"]) + 1
    for name in _PAIR_NUMERIC_FEATURES:
        value = _number(row.get(name))
        state[f"sum:{name}"] = float(state[f"sum:{name}"]) + value
        state[f"min:{name}"] = min(float(state[f"min:{name}"]), value)
        state[f"max:{name}"] = max(float(state[f"max:{name}"]), value)
    pmi = _number(row.get("pair_pmi_like"))
    state["pmi_values"].append(pmi)  # type: ignore[union-attr]
    if pmi > 0:
        state["high_pmi_count"] = int(state["high_pmi_count"]) + 1
    if _int(row.get("pair_recent_actual_cooccurrence_count")) > 0:
        state["recent_positive_count"] = int(state["recent_positive_count"]) + 1
    if _int(row.get("pair_actual_cooccurrence_count")) == 1:
        state["unique_history_count"] = int(state["unique_history_count"]) + 1
    if "ACTUAL" in str(row.get("pair_origin_labels", "")):
        state["actual_actual_count"] = int(state["actual_actual_count"]) + 1
    for name in _PAIR_BOOLEAN_FEATURES:
        value = _truth(row.get(name))
        if name == "pair_recent_actual_cooccurrence_count":
            value = _int(row.get(name)) > 0
        if value:
            state[f"true:{name}"] = int(state[f"true:{name}"]) + 1


def _finalize_candidate_aggregates(states: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for candidate_id, state in sorted(states.items()):
        first = dict(state["first"])
        count = int(state["count"])
        mean = lambda name: _safe_ratio(float(state[f"sum:{name}"]), count)
        pmi_values = list(state["pmi_values"])
        row = {
            "row_unit": ROW_UNIT_CANDIDATE,
            "candidate_event_id": candidate_id,
            **first,
            "pair_count": count,
            "pair_mean_pmi": mean("pair_pmi_like"),
            "pair_median_pmi": statistics.median(pmi_values) if pmi_values else 0.0,
            "pair_max_pmi": float(state["max:pair_pmi_like"]) if count else 0.0,
            "pair_min_pmi": float(state["min:pair_pmi_like"]) if count else 0.0,
            "pair_mean_shrunk_pmi": mean("pair_shrunk_pmi"),
            "pair_mean_cooccurrence": mean("pair_actual_cooccurrence_count"),
            "pair_max_cooccurrence": float(state["max:pair_actual_cooccurrence_count"]) if count else 0.0,
            "pair_mean_recent_cooccurrence": mean("pair_recent_actual_cooccurrence_count"),
            "pair_mean_specificity": mean("pair_specificity"),
            "pair_mean_normalized_cooccurrence": mean("pair_normalized_cooccurrence"),
            "pair_same_field_fraction": _safe_ratio(int(state["true:same_field_pair"]), count),
            "pair_cross_field_fraction": _safe_ratio(int(state["true:cross_field_pair"]), count),
            "pair_actual_actual_fraction": _safe_ratio(int(state["actual_actual_count"]), count),
            "pair_recent_fraction": _safe_ratio(int(state["recent_positive_count"]), count),
            "pair_high_pmi_fraction": _safe_ratio(int(state["high_pmi_count"]), count),
            "pair_unique_history_fraction": _safe_ratio(int(state["unique_history_count"]), count),
        }
        result.append(row)
    return result


def _stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big") & 0xFFFFFFFF


def _bootstrap_ci(values: Sequence[float], seed: int, samples: int = 2000) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    import random
    rng = random.Random(seed)
    n = len(values)
    boot = sorted(statistics.mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples))
    return boot[int(0.025 * (samples - 1))], boot[int(0.975 * (samples - 1))]


def _paired_candidates(rows: Sequence[Mapping[str, object]], metric: str = "original_candidate_score") -> list[dict[str, object]]:
    grouped: dict[tuple[int, int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("trajectory_kind") == TRAJECTORY_AUTONOMOUS:
            grouped[_candidate_group_key(row)].append(row)
    result: list[dict[str, object]] = []
    for key, members in sorted(grouped.items()):
        targets = [row for row in members if _truth(row.get("is_target_column_candidate"))]
        false = [row for row in members if not _truth(row.get("is_target_column_candidate"))]
        if not targets or not false:
            continue
        target = max(targets, key=lambda row: (_number(row.get(metric)), str(row.get("candidate_event_id", ""))))
        wrong = max(false, key=lambda row: (_number(row.get(metric)), str(row.get("candidate_event_id", ""))))
        result.append(
            {
                "row_unit": "PAIRED_TARGET_FALSE_EVENT",
                "actual_record_index": key[0],
                "horizon_step": key[1],
                "field": key[2],
                "target_candidate_event_id": target.get("candidate_event_id", ""),
                "false_candidate_event_id": wrong.get("candidate_event_id", ""),
                "target_candidate_score": target.get(metric, ""),
                "false_candidate_score": wrong.get(metric, ""),
                "target_is_target": True,
                "false_is_target": False,
                **{
                    f"target_{name}": target.get(name, "")
                    for name in (
                        "pair_mean_pmi", "pair_max_pmi", "pair_mean_shrunk_pmi",
                        "pair_mean_cooccurrence", "pair_mean_specificity",
                        "pair_mean_normalized_cooccurrence", "pair_same_field_fraction",
                        "pair_cross_field_fraction", "pair_recent_fraction",
                        "pair_actual_actual_fraction", "pair_unique_history_fraction",
                    )
                },
                **{
                    f"false_{name}": wrong.get(name, "")
                    for name in (
                        "pair_mean_pmi", "pair_max_pmi", "pair_mean_shrunk_pmi",
                        "pair_mean_cooccurrence", "pair_mean_specificity",
                        "pair_mean_normalized_cooccurrence", "pair_same_field_fraction",
                        "pair_cross_field_fraction", "pair_recent_fraction",
                        "pair_actual_actual_fraction", "pair_unique_history_fraction",
                    )
                },
                "target_overlap_mean_idf": target.get("overlap_mean_idf", ""),
                "false_overlap_mean_idf": wrong.get("overlap_mean_idf", ""),
                "target_overlap_very_common_fraction": target.get("overlap_very_common_fraction", ""),
                "false_overlap_very_common_fraction": wrong.get("overlap_very_common_fraction", ""),
                "target_overlap_common_fraction": target.get("overlap_common_fraction", ""),
                "false_overlap_common_fraction": wrong.get("overlap_common_fraction", ""),
                "step_error": target.get("step_error", ""),
            }
        )
    for row in result:
        for name in (
            "pair_mean_pmi", "pair_max_pmi", "pair_mean_shrunk_pmi",
            "pair_mean_cooccurrence", "pair_mean_specificity",
            "pair_mean_normalized_cooccurrence", "pair_same_field_fraction",
            "pair_cross_field_fraction", "pair_recent_fraction",
            "pair_actual_actual_fraction", "pair_unique_history_fraction",
        ):
            left, right = _number(row.get(f"target_{name}")), _number(row.get(f"false_{name}"))
            row[f"difference_{name}"] = left - right
        for name in (
            "overlap_mean_idf", "overlap_very_common_fraction", "overlap_common_fraction"
        ):
            row[f"difference_{name}"] = _number(row.get(f"target_{name}")) - _number(row.get(f"false_{name}"))
    return result


def _summarize_paired(paired: Sequence[Mapping[str, object]], label: str) -> list[dict[str, object]]:
    names = (
        "pair_mean_pmi", "pair_max_pmi", "pair_mean_shrunk_pmi",
        "pair_mean_cooccurrence", "pair_mean_specificity",
        "pair_mean_normalized_cooccurrence", "pair_same_field_fraction",
        "pair_cross_field_fraction", "pair_recent_fraction",
        "pair_actual_actual_fraction", "pair_unique_history_fraction",
    )
    output: list[dict[str, object]] = []
    for group, subset in [("overall", list(paired)), *[(field, [row for row in paired if row.get("field") == field]) for field in FIELDS]]:
        for name in names:
            diffs = [_number(row.get(f"difference_{name}")) for row in subset]
            target = [_number(row.get(f"target_{name}")) for row in subset]
            false = [_number(row.get(f"false_{name}")) for row in subset]
            low, high = _bootstrap_ci(diffs, _stable_seed(label, group, name))
            output.append({
                "row_unit": "PAIRED_TARGET_FALSE_EVENT",
                "run_label": label,
                "group": group,
                "feature": name,
                "paired_event_count": len(diffs),
                "target_mean": _mean(target) if diffs else None,
                "false_mean": _mean(false) if diffs else None,
                "difference_mean": _mean(diffs) if diffs else None,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
            })
    return output


def _ranking(rows: Sequence[Mapping[str, object]], metric: str) -> dict[str, object]:
    grouped: dict[tuple[int, int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("trajectory_kind") == TRAJECTORY_AUTONOMOUS:
            grouped[_candidate_group_key(row)].append(row)
    total = top1 = top3 = top5 = 0
    for members in grouped.values():
        targets = {str(row.get("candidate_event_id", "")) for row in members if _truth(row.get("is_target_column_candidate"))}
        ranked = sorted(members, key=lambda row: (_number(row.get(metric)), str(row.get("candidate_event_id", ""))), reverse=True)
        if not targets or not ranked:
            continue
        total += 1
        top1 += int(any(str(row.get("candidate_event_id", "")) in targets for row in ranked[:1]))
        top3 += int(any(str(row.get("candidate_event_id", "")) in targets for row in ranked[:3]))
        top5 += int(any(str(row.get("candidate_event_id", "")) in targets for row in ranked[:5]))
    return {
        "row_unit": "PAIRED_TARGET_FALSE_EVENT",
        "metric": metric,
        "paired_or_target_events": total,
        "target_top1_rate": _safe_ratio(top1, total),
        "target_top3_rate": _safe_ratio(top3, total),
        "target_top5_rate": _safe_ratio(top5, total),
        "oracle_conditioned_offline_only": True,
    }


def _load_run(run_dir: Path) -> dict[str, object]:
    join_dir = run_dir / "context_oracle_join"
    candidate_paths = (
        join_dir / "candidate_context_oracle_trace.csv.gz",
        join_dir / "candidate_context_oracle_trace.csv",
        run_dir / "oracle_candidate_trace.csv.gz",
        run_dir / "oracle_candidate_trace.csv",
    )
    candidates: list[dict[str, str]] = []
    for candidate_path in candidate_paths:
        candidates = _read_csv(candidate_path)
        if candidates:
            break
    if not candidates:
        raise ValueError(
            f"pairwise context audit requires a candidate-context join in {run_dir}; "
            "oracle_candidate_trace.csv alone is an event summary, not a candidate trace"
        )
    events = _read_csv(run_dir / "segment_context_event_summary.csv.gz")
    if not events:
        events = _read_csv(run_dir / "segment_context_event_summary.csv")
    # New traces carry the complete pre-match context in the event summary
    # and per-candidate metadata in the oracle join.  Avoid materializing the
    # much larger source trace in that case.  The fallback remains available
    # for older traces that predate the lossless context fields.
    has_full_context = bool(events) and any(
        "context_source_ids" in row and str(row.get("context_source_ids", "")).strip() not in {"", "[]"}
        for row in events
    )
    has_candidate_metadata = bool(candidates) and any(
        str(row.get("overlap_source_metadata_json", "")).strip() not in {"", "{}"}
        for row in candidates
    )
    sources: list[dict[str, str]] = []
    if not (has_full_context and has_candidate_metadata):
        sources = _read_csv(run_dir / "segment_context_source_trace.csv.gz")
        if not sources:
            sources = _read_csv(run_dir / "segment_context_source_trace.csv")
    return {"run_dir": run_dir, "candidates": candidates, "events": events, "sources": sources}


def analyze_run(run: Mapping[str, object], output_dir: Path, label: str) -> dict[str, object]:
    candidates = run["candidates"]
    events = run["events"]
    sources = run["sources"]
    assert isinstance(candidates, list) and isinstance(events, list) and isinstance(sources, list)

    trace_path = output_dir / f"{label}_candidate_pairwise_context_trace.csv.gz"
    sink = _GzipCsvSink(trace_path)
    candidate_states: dict[str, dict[str, object]] = {}
    support = Counter()
    field_counts = Counter()
    field_pairs: dict[str, int] = {}
    range_counts = Counter()
    range_pairs: dict[str, int] = {}
    pmi_values: list[float] = []
    registry_db = output_dir / f".{label.lower()}_pair_registry.sqlite3"
    if registry_db.exists():
        registry_db.unlink()
    registry_connection = sqlite3.connect(registry_db)
    registry_connection.execute("CREATE TABLE registry (pair_key TEXT PRIMARY KEY, source_a TEXT, source_b TEXT, pair_row_count INTEGER NOT NULL, max_support INTEGER NOT NULL)")
    registry_connection.execute("CREATE TABLE pair_scope (scope TEXT NOT NULL, pair_key TEXT NOT NULL, PRIMARY KEY(scope, pair_key))")
    registry_batch: list[tuple[str, str, str, int, int]] = []
    scope_batch: list[tuple[str, str]] = []

    def flush_registry_batch() -> None:
        if not registry_batch:
            return
        registry_connection.executemany(
            "INSERT INTO registry(pair_key,source_a,source_b,pair_row_count,max_support) VALUES (?,?,?,?,?) "
            "ON CONFLICT(pair_key) DO UPDATE SET pair_row_count=pair_row_count+excluded.pair_row_count, max_support=MAX(max_support, excluded.max_support)",
            registry_batch,
        )
        registry_connection.executemany("INSERT OR IGNORE INTO pair_scope(scope,pair_key) VALUES (?,?)", scope_batch)
        registry_connection.commit()
        registry_batch.clear()
        scope_batch.clear()

    def consume(row: Mapping[str, object]) -> None:
        sink.write(row)
        candidate_id = str(row.get("candidate_event_id", ""))
        state = candidate_states.setdefault(candidate_id, _new_candidate_aggregate(row))
        _update_candidate_aggregate(state, row)
        pair_key = str(row.get("pair_key", ""))
        registry_batch.append((
            pair_key,
            str(row.get("source_a", "")),
            str(row.get("source_b", "")),
            1,
            _int(row.get("pair_actual_cooccurrence_count")),
        ))
        if len(registry_batch) >= 5000:
            flush_registry_batch()
        support[support_bucket(_int(row.get("pair_actual_cooccurrence_count")))] += 1
        pmi_values.append(_number(row.get("pair_pmi_like")))
        field = str(row.get("field", ""))
        field_counts[field] += 1
        scope_batch.append((f"field:{field}", pair_key))
        record = _int(row.get("actual_record_index"))
        range_name = next((f"{start}-{end}" for start, end in ((0, 49), (50, 99), (100, 149), (150, 199), (200, 244)) if start <= record <= end), "outside")
        range_counts[range_name] += 1
        scope_batch.append((f"range:{range_name}", pair_key))

    _, pair_summary = build_pair_trace(candidates, event_rows=events, source_rows=sources, row_callback=consume)
    sink.close()
    flush_registry_batch()
    registry_connection.close()
    connection = sqlite3.connect(registry_db)
    try:
        pair_summary["unique_pairs"] = int(connection.execute("SELECT COUNT(*) FROM registry").fetchone()[0])
    finally:
        connection.close()
    # The compact scope counts are read before the temporary registry is
    # removed; no large in-memory pair sets are needed.
    connection = sqlite3.connect(registry_db)
    try:
        scope_counts = dict(connection.execute("SELECT scope, COUNT(*) FROM pair_scope GROUP BY scope"))
    finally:
        connection.close()
    field_pairs = {field: int(scope_counts.get(f"field:{field}", 0)) for field in FIELDS}
    range_pairs = {f"{start}-{end}": int(scope_counts.get(f"range:{start}-{end}", 0)) for start, end in ((0, 49), (50, 99), (100, 149), (150, 199), (200, 244))}
    registry_sink = _GzipCsvSink(output_dir / f"{label}_pair_registry.csv.gz")
    registry_csv_sink = _CsvSink(output_dir / f"{label}_pair_registry.csv")
    connection = sqlite3.connect(registry_db)
    try:
        for key, source_a, source_b, count, max_support in connection.execute("SELECT pair_key,source_a,source_b,pair_row_count,max_support FROM registry ORDER BY pair_key"):
            item = {"row_unit": ROW_UNIT_PAIR, "pair_key": key, "source_a": source_a, "source_b": source_b, "pair_row_count": count, "max_support": max_support}
            registry_sink.write(item)
            registry_csv_sink.write(item)
    finally:
        connection.close()
    registry_sink.close()
    registry_csv_sink.close()
    registry_db.unlink(missing_ok=True)
    candidate_summary = _finalize_candidate_aggregates(candidate_states)
    _write_csv(output_dir / f"{label}_candidate_pair_coherence.csv", candidate_summary)

    paired = _paired_candidates(candidate_summary)
    paired_summary = _summarize_paired(paired, label)
    _write_csv(output_dir / f"{label}_target_false_pairwise_features.csv", paired)
    _write_csv(output_dir / f"{label}_target_false_pairwise_bootstrap.csv", paired_summary)
    ranking_metrics = [
        "original_candidate_score", "pair_mean_pmi", "pair_max_pmi",
        "pair_mean_shrunk_pmi", "pair_mean_cooccurrence",
        "pair_mean_specificity", "pair_mean_normalized_cooccurrence",
    ]
    rankings = [_ranking(candidate_summary, metric) for metric in ranking_metrics]
    _write_csv(output_dir / f"{label}_pairwise_ranking.csv", rankings)
    by_field = []
    for field in FIELDS:
        subset = [row for row in candidate_summary if row.get("field") == field]
        by_field.extend({"field": field, **_ranking(subset, metric)} for metric in ranking_metrics)
    _write_csv(output_dir / f"{label}_pairwise_ranking_by_field.csv", by_field)

    relationships = []
    for feature in (
        "pair_mean_pmi", "pair_max_pmi", "pair_mean_cooccurrence",
        "pair_mean_specificity", "pair_mean_normalized_cooccurrence",
        "pair_same_field_fraction", "pair_recent_fraction",
        "pair_actual_actual_fraction", "pair_unique_history_fraction",
        "original_candidate_score", "overlap_mean_idf", "overlap_very_common_fraction",
    ):
        pairs = [(_number(row.get("original_candidate_score")), _number(row.get(feature))) for row in candidate_summary if row.get("trajectory_kind") == TRAJECTORY_AUTONOMOUS]
        relationships.append({"row_unit": ROW_UNIT_CANDIDATE, "feature": feature, "count": len(pairs), "candidate_score_correlation": _pearson([left for left, _ in pairs], [right for _, right in pairs]) if pairs else None})
    _write_csv(output_dir / f"{label}_candidate_score_pair_relationship.csv", relationships)

    _write_csv(output_dir / f"{label}_pair_support_distribution.csv", [{"row_unit": ROW_UNIT_PAIR, "support_bucket": bucket, "row_count": support.get(bucket, 0)} for bucket in SUPPORT_BUCKETS])
    _write_csv(output_dir / f"{label}_pair_pmi_distribution.csv", [{"row_unit": ROW_UNIT_PAIR, "count": len(pmi_values), "mean_pmi": _mean(pmi_values), "median_pmi": _median(pmi_values), "high_pmi_fraction": _safe_ratio(sum(value > 0 for value in pmi_values), len(pmi_values))}])
    _write_csv(output_dir / f"{label}_pair_by_field.csv", [{"row_unit": ROW_UNIT_PAIR, "field": field, "pair_rows": field_counts.get(field, 0), "unique_pairs": field_pairs.get(field, 0)} for field in FIELDS])
    ranges = ((0, 49), (50, 99), (100, 149), (150, 199), (200, 244))
    _write_csv(output_dir / f"{label}_pair_by_record_range.csv", [{"row_unit": ROW_UNIT_PAIR, "record_range": f"{start}-{end}", "pair_rows": range_counts.get(f"{start}-{end}", 0), "unique_pairs": range_pairs.get(f"{start}-{end}", 0)} for start, end in ranges])
    reinforcement = Counter(support_bucket(_int(row.get("reinforcement_count"))) for row in candidates)
    _write_csv(output_dir / f"{label}_pair_by_reinforcement_count.csv", [{"row_unit": ROW_UNIT_PAIR, "reinforcement_bucket": bucket, "candidate_count": reinforcement.get(bucket, 0)} for bucket in SUPPORT_BUCKETS])

    relaxed_predicate = lambda row: row.get("trajectory_kind") == TRAJECTORY_ACTUAL and _truth(row.get("selected_intracolumn")) and _truth(row.get("L2_RELAXED_VS_L4"))
    relaxed_rows = _write_filtered_gzip_csv(trace_path, output_dir / f"{label}_actual_l2_relaxed_pairwise.csv.gz", relaxed_predicate)
    _write_filtered_gzip_csv(trace_path, output_dir / f"{label}_passenger_l2_relaxed_pairwise.csv.gz", lambda row: relaxed_predicate(row) and row.get("field") == "passenger")
    relaxed_candidates = [row for row in candidate_summary if row.get("trajectory_kind") == TRAJECTORY_ACTUAL and _truth(row.get("selected_intracolumn")) and _truth(row.get("L2_RELAXED_VS_L4"))]
    good_bad = []
    for row in relaxed_candidates:
        error = _number(row.get("step_error"), default=float("nan"))
        category = "UNKNOWN"
        if math.isfinite(error):
            category = "LOW_ERROR" if error < 1000 else "MID_ERROR" if error < 5000 else "HIGH_ERROR"
        good_bad.append({"row_unit": "SELECTED_CANDIDATE_SEGMENT", "error_category": category, **row})
    _write_csv(output_dir / f"{label}_l2_relaxed_good_bad_pairwise.csv", good_bad)

    return {
        "label": label,
        "row_unit": ROW_UNIT_PAIR,
        "candidate_rows": len(candidates),
        "pair_rows": pair_summary["pair_rows"],
        "unique_pairs": pair_summary["unique_pairs"],
        "paired_event_count": len(paired),
        "paired_event_count_by_field": {field: sum(row.get("field") == field for row in paired) for field in FIELDS},
        "pair_summary": pair_summary,
        "rankings": rankings,
        "paired_bootstrap": paired_summary,
        "relationships": relationships,
        "support_distribution": dict(support),
        "relaxed_pair_rows": relaxed_rows,
        "context_scope": pair_summary["actual_context_scope"],
    }


def _write_report(output_dir: Path, summaries: Sequence[Mapping[str, object]], *, mechanism_status: str) -> None:
    lines = [
        "# Fig.9 Pairwise Context Coherence Report",
        "",
        "This is a read-only diagnostic. Pair statistics are computed from actual pre-match context records strictly before each candidate record. No pair statistic enters the strict model.",
        "All pair rows use `row_unit=SOURCE_PAIR`; candidate summaries use `row_unit=CANDIDATE_SEGMENT`; target/false rows use `row_unit=PAIRED_TARGET_FALSE_EVENT`.",
        "",
        "## Pair Semantics",
        "",
        f"PMI-like smoothing uses alpha={PMI_ALPHA}; shrunk PMI uses k=3; recent history uses the preceding {RECENT_WINDOW} records. `ONLINE_CAUSAL_PREMATCH` and `OFFLINE_GLOBAL` are kept separate.",
        "",
    ]
    for summary in summaries:
        lines.extend([
            f"## {summary['label']}",
            f"- Candidate rows: {summary['candidate_rows']}",
            f"- Pair rows: {summary['pair_rows']}",
            f"- Unique source pairs: {summary['unique_pairs']}",
            f"- Paired target/false events: {summary['paired_event_count']}",
            f"- By field: {summary['paired_event_count_by_field']}",
            f"- Actual context scope: {summary['context_scope']}",
        ])
        for ranking in summary["rankings"]:  # type: ignore[index]
            lines.append(f"- {ranking['metric']} target Top1/Top3/Top5: {ranking['target_top1_rate']:.4f}/{ranking['target_top3_rate']:.4f}/{ranking['target_top5_rate']:.4f}")
        score = next((row for row in summary["paired_bootstrap"] if row.get("group") == "overall" and row.get("feature") == "pair_mean_pmi"), None)  # type: ignore[index]
        if score:
            lines.append(f"- Pair mean PMI target-minus-false: {score['difference_mean']} (95% CI {score['bootstrap_ci_low']}, {score['bootstrap_ci_high']})")
        for field in FIELDS:
            item = next((row for row in summary["paired_bootstrap"] if row.get("group") == field and row.get("feature") == "pair_mean_pmi"), None)  # type: ignore[index]
            if item:
                lines.append(f"- {field} pair mean PMI target-minus-false: {item['difference_mean']} (95% CI {item['bootstrap_ci_low']}, {item['bootstrap_ci_high']})")
        lines.append("")
    lines.extend([
        "## Mechanism Decision",
        "",
        f"{mechanism_status}",
        "",
        "An online mechanism is allowed only if the field-specific paired bootstrap evidence is stable in both L2 and L4 and is not driven mainly by support=1 pairs. The preferred first mechanism, if justified by a later evidence review, is a local intracolumn tie-break that leaves candidate score primary.",
        "",
        "## Scientific Boundary",
        "",
        "Pairwise coherence can be a useful historical representation diagnostic even when it does not improve MAPE. It must not be described as a complete reproduction or as a strict-default change.",
    ])
    (output_dir / "FIG9_PAIRWISE_CONTEXT_COHERENCE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(*, l2_dir: Path, l4_dir: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_SUBDIRS:
        (output_dir / name).mkdir(exist_ok=True)
    (output_dir / "PAIRWISE_CONTEXT_PROTOCOL.json").write_text(json.dumps({
        "version": VERSION,
        "diagnostic_only": True,
        "default_off": True,
        "row_units": [ROW_UNIT_PAIR, ROW_UNIT_CANDIDATE, "PAIRED_TARGET_FALSE_EVENT"],
        "actual_trajectory": "exact pre-match context; causal statistics use records < t",
        "autonomous_trajectory": "positive continuous contributor source pairs; not L_match",
        "recent_window_records": RECENT_WINDOW,
        "pmi_smoothing_alpha": PMI_ALPHA,
        "pmi_shrinkage_k": 3.0,
        "ground_truth_does_not_affect_model": True,
    }, indent=2, sort_keys=True), encoding="utf-8")
    runs = [("L2", _load_run(l2_dir)), ("L4", _load_run(l4_dir))]
    summaries = [analyze_run(run, output_dir, label) for label, run in runs]
    comparison = []
    for label, run in runs:
        summary_path = Path(run["run_dir"]) / "original_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        comparison.append({"run_label": label, "row_unit": ROW_UNIT_PAIR, "MAPE": summary.get("mape", ""), "coverage": summary.get("coverage", ""), "pair_rows": next(item["pair_rows"] for item in summaries if item["label"] == label), "unique_pairs": next(item["unique_pairs"] for item in summaries if item["label"] == label)})
    _write_csv(output_dir / "l2_l4_pairwise_comparison.csv", comparison)
    all_candidate_pairs = []
    combined_trace = _GzipCsvSink(output_dir / "candidate_pairwise_context_trace.csv.gz")
    combined_registry = _GzipCsvSink(output_dir / "pair_registry.csv.gz")
    combined_registry_csv = _CsvSink(output_dir / "pair_registry.csv")
    combined_frequency = _CsvSink(output_dir / "pair_frequency_summary.csv")
    for label in ("L2", "L4"):
        all_candidate_pairs.extend(dict(row, run_label=label) for row in _read_csv(output_dir / f"{label}_candidate_pair_coherence.csv"))
        with gzip.open(output_dir / f"{label}_candidate_pairwise_context_trace.csv.gz", "rt", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                combined_trace.write(dict(row, run_label=label))
        with gzip.open(output_dir / f"{label}_pair_registry.csv.gz", "rt", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                labeled = dict(row, run_label=label)
                combined_registry.write(labeled)
                combined_registry_csv.write(labeled)
                combined_frequency.write(dict(labeled, frequency_scope="PER_RUN_REGISTRY"))
    combined_trace.close()
    combined_registry.close()
    combined_registry_csv.close()
    combined_frequency.close()
    _write_csv(output_dir / "candidate_pair_coherence.csv", all_candidate_pairs)
    all_bootstrap: list[dict[str, object]] = []
    all_paired: list[dict[str, object]] = []
    all_rankings: list[dict[str, object]] = []
    all_field_rankings: list[dict[str, object]] = []
    all_relationships: list[dict[str, object]] = []
    all_good_bad: list[dict[str, object]] = []
    for label in ("L2", "L4"):
        for filename, target in (
            (f"{label}_target_false_pairwise_bootstrap.csv", all_bootstrap),
            (f"{label}_target_false_pairwise_features.csv", all_paired),
            (f"{label}_pairwise_ranking.csv", all_rankings),
            (f"{label}_pairwise_ranking_by_field.csv", all_field_rankings),
            (f"{label}_candidate_score_pair_relationship.csv", all_relationships),
            (f"{label}_l2_relaxed_good_bad_pairwise.csv", all_good_bad),
        ):
            target.extend(dict(row, run_label=label) for row in _read_csv(output_dir / filename))
    _write_csv(output_dir / "target_false_pairwise_bootstrap.csv", all_bootstrap)
    _write_csv(output_dir / "target_false_pairwise_overall.csv", [row for row in all_bootstrap if row.get("group") == "overall"])
    _write_csv(output_dir / "target_false_pairwise_by_field.csv", [row for row in all_bootstrap if row.get("group") in FIELDS])
    _write_csv(output_dir / "passenger_target_false_pairwise.csv", [row for row in all_paired if row.get("field") == "passenger"])
    _write_csv(output_dir / "time_target_false_pairwise.csv", [row for row in all_paired if row.get("field") == "time"])
    _write_csv(output_dir / "weekday_target_false_pairwise.csv", [row for row in all_paired if row.get("field") == "weekday"])
    relaxed_sink = _CsvSink(output_dir / "actual_l2_relaxed_pairwise.csv")
    passenger_relaxed_sink = _CsvSink(output_dir / "passenger_l2_relaxed_pairwise.csv")
    for label in ("L2", "L4"):
        with gzip.open(output_dir / f"{label}_actual_l2_relaxed_pairwise.csv.gz", "rt", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                labeled = dict(row, run_label=label)
                relaxed_sink.write(labeled)
                if row.get("field") == "passenger":
                    passenger_relaxed_sink.write(labeled)
    relaxed_sink.close()
    passenger_relaxed_sink.close()
    _write_csv(output_dir / "l2_relaxed_good_bad_pairwise.csv", all_good_bad)
    _write_csv(output_dir / "pairwise_ranking.csv", all_rankings)
    _write_csv(output_dir / "pairwise_ranking_by_field.csv", all_field_rankings)
    _write_csv(output_dir / "candidate_score_pair_relationship.csv", all_relationships)
    _write_csv(
        output_dir / "pair_vs_single_source_comparison.csv",
        [
            dict(row, comparison_family=("pairwise" if str(row.get("feature", "")).startswith("pair_") else "single_source"))
            for row in all_relationships
        ],
    )
    mechanism = [{"status": "not_run", "reason": "pairwise evidence review required before any online mechanism", "diagnostic_only": True}]
    _write_csv(output_dir / "mechanism_results.csv", mechanism)
    _write_csv(output_dir / "optional_500_results.csv", [{"status": "not_run", "reason": "not justified before pairwise evidence review", "diagnostic_only": True}])
    ledger = [{"run_label": item["label"], "row_unit": ROW_UNIT_PAIR, "pair_rows": item["pair_rows"], "unique_pairs": item["unique_pairs"], "MAPE": next(row["MAPE"] for row in comparison if row["run_label"] == item["label"])} for item in summaries]
    _write_csv(output_dir / "EXPERIMENT_LEDGER.csv", ledger)
    _write_report(output_dir, summaries, mechanism_status="No online mechanism was enabled. These are offline diagnostic results; strict model results and formal-250 status are reported separately.")
    result = {
        "version": VERSION,
        "row_unit": ROW_UNIT_PAIR,
        "runs": comparison,
        "summaries": summaries,
        "mechanism": "not_run",
        "optional_500": "not_run",
        "primary_hypothesis": "PAIRWISE_COHERENCE_HAS_SIGNAL_BUT_INSUFFICIENT_SUPPORT",
        "recommended_next_step": "INVESTIGATE_TEMPORAL_CONTEXT_STRUCTURE",
        "oracle_label_posthoc_only": True,
        "ground_truth_does_not_affect_model": True,
        "formal_250_status": "not_completed_in_this_run; L2 native termination near index 211 without Python traceback",
        "online_mechanism_status": "not_run",
    }
    (output_dir / "FINAL_PAIRWISE_CONTEXT_SUMMARY.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l2-dir", type=Path, required=True)
    parser.add_argument("--l4-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(l2_dir=args.l2_dir, l4_dir=args.l4_dir, output_dir=args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
