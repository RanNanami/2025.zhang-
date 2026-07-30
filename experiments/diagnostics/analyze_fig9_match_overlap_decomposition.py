"""Offline analysis for the Fig.9 match-overlap decomposition trace."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


RECORD_RANGES = (
    (0, 49, "0-49"),
    (50, 99, "50-99"),
    (100, 149, "100-149"),
    (150, 199, "150-199"),
    (200, 244, "200-244"),
)
UNKNOWN_RECORD_RANGE = "UNKNOWN_RECORD_RANGE"
RECORD_RANGE_FIX_VERSION = "fig9-match-overlap-record-range-v2"
CONTEXT_COLUMNS = {
    "ALL_CELL_CURRENT": "overlap_all_cell_current",
    "WINNER_ONLY": "overlap_winner_only",
    "PREDICTED_ONLY": "overlap_predicted_only",
    "BURST_ONLY": "overlap_burst_only",
    "PREDICTED_PLUS_WINNER": "overlap_predicted_plus_winner",
    "WITHOUT_BURST_ONLY": "overlap_without_burst_only",
    "SAME_FIELD_ONLY": "overlap_same_field_only",
    "CROSS_FIELD_ONLY": "overlap_cross_field_only",
}


def _find(run_dir: Path, stem: str) -> Path | None:
    for suffix in (".csv", ".csv.gz"):
        path = run_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def _read(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _number(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _truth(value: object) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def classify_record_range(index: object) -> str:
    """Classify a real ``actual_record_index`` without silent coercion."""

    if index is None or isinstance(index, bool):
        return UNKNOWN_RECORD_RANGE
    text = str(index).strip()
    if not text:
        return UNKNOWN_RECORD_RANGE
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return UNKNOWN_RECORD_RANGE
    if not math.isfinite(numeric) or not numeric.is_integer():
        return UNKNOWN_RECORD_RANGE
    value = int(numeric)
    for start, stop, label in RECORD_RANGES:
        if start <= value <= stop:
            return label
    return UNKNOWN_RECORD_RANGE


def _record_range(index: object) -> str:
    """Backward-compatible private alias for older analyzer helpers."""

    return classify_record_range(index)


def _mean(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return (
        sum(_number(row.get(field)) for row in rows) / len(rows)
        if rows
        else 0.0
    )


def _rate(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return (
        sum(_truth(row.get(field)) for row in rows) / len(rows)
        if rows
        else 0.0
    )


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _groups(
    rows: Iterable[Mapping[str, object]],
    keys: Sequence[str],
) -> dict[tuple[str, ...], list[Mapping[str, object]]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            tuple(
                classify_record_range(row.get("actual_record_index"))
                if key == "record_range"
                else str(row.get(key, ""))
                for key in keys
            )
        ].append(row)
    return grouped


def _column_summary(
    rows: Sequence[Mapping[str, object]],
    keys: Sequence[str],
) -> list[dict[str, object]]:
    output = []
    for key, group in sorted(_groups(rows, keys).items()):
        output.append(
            {
                **dict(zip(keys, key)),
                "columns": len(group),
                "existing_segment_rate": sum(
                    _number(row.get("existing_segment_count")) > 0
                    for row in group
                )
                / len(group),
                "mean_segment_count": _mean(group, "existing_segment_count"),
                "mean_best_overlap": _mean(group, "best_matching_overlap"),
                "mean_gap_to_L_match": _mean(group, "gap_to_L_match"),
                "mean_source_retention": _mean(
                    group,
                    "best_segment_source_retention",
                ),
                "mean_creation_current_jaccard": _mean(
                    group,
                    "best_segment_creation_current_jaccard",
                ),
                "mean_current_context_jaccard": _mean(
                    group,
                    "best_segment_current_context_jaccard",
                ),
                "scenario3_rate": sum(
                    str(row.get("observe_scenario")) == "scenario3"
                    for row in group
                )
                / len(group),
                "multiple_segment_ambiguity_rate": _rate(
                    group,
                    "multiple_segments_meet_L_match",
                ),
                "multiple_neuron_ambiguity_rate": _rate(
                    group,
                    "multiple_neurons_meet_L_match",
                ),
            }
        )
    return output


def _passenger_distribution(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    passenger = [row for row in rows if row.get("field") == "passenger"]
    counter = Counter(int(_number(row.get("best_matching_overlap"))) for row in passenger)
    total = len(passenger)
    return [
        {
            "best_overlap": overlap,
            "columns": count,
            "rate": count / total if total else 0.0,
        }
        for overlap, count in sorted(counter.items())
    ]


def _segment_source_summary(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output = []
    for key, group in sorted(_groups(rows, ("field", "record_range")).items()):
        source_total = sum(_number(row.get("segment_synapse_count")) for row in group)
        output.append(
            {
                "field": key[0],
                "record_range": key[1],
                "segments": len(group),
                "mean_synapse_count": _mean(group, "segment_synapse_count"),
                "weekday_source_ratio": (
                    sum(_number(row.get("source_weekday_count")) for row in group)
                    / source_total
                    if source_total
                    else 0.0
                ),
                "time_source_ratio": (
                    sum(_number(row.get("source_time_count")) for row in group)
                    / source_total
                    if source_total
                    else 0.0
                ),
                "passenger_source_ratio": (
                    sum(_number(row.get("source_passenger_count")) for row in group)
                    / source_total
                    if source_total
                    else 0.0
                ),
                "mean_actual_overlap": _mean(group, "actual_overlap"),
            }
        )
    return output


def _retention_summary(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output = []
    for key, group in sorted(_groups(rows, ("field", "record_range")).items()):
        output.append(
            {
                "field": key[0],
                "record_range": key[1],
                "segments": len(group),
                "mean_source_retention": _mean(
                    group,
                    "source_retention_ratio",
                ),
                "mean_creation_current_jaccard": _mean(
                    group,
                    "creation_current_jaccard",
                ),
                "mean_creation_sources": _mean(group, "creation_source_count"),
                "mean_creation_sources_active": _mean(
                    group,
                    "creation_source_currently_active",
                ),
                "mean_creation_sources_winner": _mean(
                    group,
                    "creation_source_currently_winner",
                ),
            }
        )
    return output


def _source_loss_summary(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped = Counter(
        (
            str(row.get("field", "")),
            _record_range(row.get("actual_record_index")),
            str(row.get("source_missing_reason", "")) or "MATCHED",
        )
        for row in rows
    )
    totals = Counter((field, record_range) for field, record_range, _ in grouped)
    # Counter above counts keys, not rows; calculate row totals explicitly.
    row_totals = Counter(
        (
            str(row.get("field", "")),
            _record_range(row.get("actual_record_index")),
        )
        for row in rows
    )
    del totals
    return [
        {
            "field": field,
            "record_range": record_range,
            "source_loss_reason": reason,
            "sources": count,
            "rate": count / row_totals[(field, record_range)],
        }
        for (field, record_range, reason), count in sorted(grouped.items())
    ]


def _observations(
    segments: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str, str, str], list[Mapping[str, object]]]:
    grouped: dict[
        tuple[str, str, str, str],
        list[Mapping[str, object]],
    ] = defaultdict(list)
    for row in segments:
        grouped[
            (
                str(row.get("actual_record_index")),
                str(row.get("field")),
                str(row.get("encoded_column")),
                str(row.get("timestamp")),
            )
        ].append(row)
    return grouped


def _context_variant_summary(
    segments: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    observations = _observations(segments)
    aggregate: dict[
        tuple[str, str, str],
        list[dict[str, object]],
    ] = defaultdict(list)
    for observation, rows in observations.items():
        record_range = _record_range(observation[0])
        field = observation[1]
        for label, column in CONTEXT_COLUMNS.items():
            overlaps = [int(_number(row.get(column))) for row in rows]
            aggregate[(field, record_range, label)].append(
                {
                    "best": max(overlaps, default=0),
                    **{
                        f"segments_L{threshold}": sum(
                            value >= threshold for value in overlaps
                        )
                        for threshold in range(1, 5)
                    },
                    **{
                        f"neurons_L{threshold}": len(
                            {
                                str(row.get("segment_target_neuron"))
                                for row in rows
                                if int(_number(row.get(column))) >= threshold
                            }
                        )
                        for threshold in range(1, 5)
                    },
                    **{
                        f"selected_compatible_L{threshold}": any(
                            _truth(row.get("selected_for_scenario"))
                            and int(_number(row.get(column))) >= threshold
                            for row in rows
                        )
                        for threshold in range(1, 5)
                    },
                }
            )
    output = []
    for key, values in sorted(aggregate.items()):
        row: dict[str, object] = {
            "field": key[0],
            "record_range": key[1],
            "context_variant": key[2],
            "columns": len(values),
            "mean_best_overlap": statistics.fmean(
                int(item["best"]) for item in values
            ),
        }
        for threshold in range(1, 5):
            segment_key = f"segments_L{threshold}"
            neuron_key = f"neurons_L{threshold}"
            compatible_key = f"selected_compatible_L{threshold}"
            row.update(
                {
                    f"mean_passing_segments_L{threshold}": statistics.fmean(
                        int(item[segment_key]) for item in values
                    ),
                    f"ambiguous_segment_rate_L{threshold}": sum(
                        int(item[segment_key]) > 1 for item in values
                    )
                    / len(values),
                    f"ambiguous_neuron_rate_L{threshold}": sum(
                        int(item[neuron_key]) > 1 for item in values
                    )
                    / len(values),
                    f"selected_compatibility_rate_L{threshold}": sum(
                        bool(item[compatible_key]) for item in values
                    )
                    / len(values),
                }
            )
        output.append(row)
    return output


def _lmatch_summaries(
    segments: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    observations = _observations(segments)
    aggregate: dict[
        tuple[str, str, str, int],
        list[tuple[int, int, int, bool, bool]],
    ] = defaultdict(list)
    for observation, rows in observations.items():
        record_range = _record_range(observation[0])
        field = observation[1]
        scenario = str(rows[0].get("observe_scenario", ""))
        for threshold in range(1, 5):
            passing = [
                row
                for row in rows
                if int(_number(row.get("actual_overlap"))) >= threshold
            ]
            neurons = {
                str(row.get("segment_target_neuron")) for row in passing
            }
            selected_compatible = any(
                _truth(row.get("selected_for_scenario"))
                and int(_number(row.get("actual_overlap"))) >= threshold
                for row in rows
            )
            selected_winner = str(rows[0].get("selected_winner_neuron", ""))
            teacher_neuron_available = any(
                str(row.get("segment_target_neuron")) == selected_winner
                and int(_number(row.get("actual_overlap"))) >= threshold
                for row in rows
            )
            aggregate[(field, record_range, scenario, threshold)].append(
                (
                    len(passing),
                    len(neurons),
                    max(
                        (
                            int(_number(row.get("actual_overlap")))
                            for row in passing
                        ),
                        default=0,
                    ),
                    selected_compatible,
                    teacher_neuron_available,
                )
            )
    counterfactual = []
    ambiguity = []
    for key, values in sorted(aggregate.items()):
        common = {
            "field": key[0],
            "record_range": key[1],
            "observe_scenario": key[2],
            "counterfactual_L_match": key[3],
            "columns": len(values),
        }
        counterfactual.append(
            {
                **common,
                "columns_with_any_passing_segment_rate": sum(
                    item[0] > 0 for item in values
                )
                / len(values),
                "mean_passing_segment_count": statistics.fmean(
                    item[0] for item in values
                ),
                "mean_passing_neuron_count": statistics.fmean(
                    item[1] for item in values
                ),
                "mean_best_passing_overlap": statistics.fmean(
                    item[2] for item in values
                ),
                "selected_segment_compatibility_rate": sum(
                    item[3] for item in values
                )
                / len(values),
                "teacher_reference_neuron_availability_rate": sum(
                    item[4] for item in values
                )
                / len(values),
                "teacher_reference_segment_availability_rate": sum(
                    item[3] for item in values
                )
                / len(values),
            }
        )
        ambiguity.append(
            {
                **common,
                "ambiguous_column_rate": sum(
                    item[0] > 1 for item in values
                )
                / len(values),
                "multiple_neuron_ambiguity_rate": sum(
                    item[1] > 1 for item in values
                )
                / len(values),
                "mean_extra_passing_segments": statistics.fmean(
                    max(0, item[0] - 1) for item in values
                ),
            }
        )
    return counterfactual, ambiguity


def _teacher_summary(
    segments: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output = []
    for key, group in sorted(_groups(segments, ("field", "record_range")).items()):
        output.append(
            {
                "field": key[0],
                "record_range": key[1],
                "segments": len(group),
                "selected_segment_rows": sum(
                    _truth(row.get("selected_for_scenario")) for row in group
                ),
                "selected_segment_meets_L4": sum(
                    _truth(row.get("selected_for_scenario"))
                    and int(_number(row.get("actual_overlap"))) >= 4
                    for row in group
                ),
                "reinforced_rows": sum(
                    _truth(row.get("reinforced")) for row in group
                ),
            }
        )
    return output


def _bootstrap(
    columns: Sequence[Mapping[str, object]],
    *,
    samples: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    output = []
    for field in ("weekday", "time", "passenger"):
        values = [
            _number(row.get("best_matching_overlap"))
            for row in columns
            if row.get("field") == field
        ]
        if not values:
            continue
        draws = [
            statistics.fmean(rng.choice(values) for _ in values)
            for _ in range(samples)
        ]
        draws.sort()
        low = draws[int(0.025 * (samples - 1))]
        high = draws[int(0.975 * (samples - 1))]
        output.append(
            {
                "field": field,
                "metric": "mean_best_overlap",
                "estimate": statistics.fmean(values),
                "ci_low": low,
                "ci_high": high,
                "bootstrap_samples": samples,
                "bootstrap_seed": seed,
            }
        )
    return output


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _observation_key(row: Mapping[str, object]) -> tuple[str, str, str, str]:
    return (
        str(row.get("actual_record_index", "")),
        str(row.get("timestamp", "")),
        str(row.get("field", "")),
        str(row.get("encoded_column", "")),
    )


def _scope_key(
    row: Mapping[str, object],
    *,
    include_range: bool,
) -> tuple[str, ...]:
    values = [str(row.get("field", ""))]
    if include_range:
        values.append(
            classify_record_range(row.get("actual_record_index"))
        )
    values.append(str(row.get("observe_scenario", "")))
    return tuple(values)


def _fixed_scope_rows(
    columns: Sequence[Mapping[str, object]],
    segments: Sequence[Mapping[str, object]],
    *,
    include_range: bool,
) -> list[dict[str, object]]:
    """Summarize explicitly scoped observation and segment denominators."""

    segments_by_observation: dict[
        tuple[str, str, str, str],
        list[Mapping[str, object]],
    ] = defaultdict(list)
    for row in segments:
        segments_by_observation[_observation_key(row)].append(row)
    grouped: dict[tuple[str, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in columns:
        grouped[_scope_key(row, include_range=include_range)].append(row)

    output: list[dict[str, object]] = []
    for group_key, observations in sorted(grouped.items()):
        field = group_key[0]
        record_range = group_key[1] if include_range else "ALL_RECORD_RANGES"
        scenario = group_key[-1]
        observation_segments = [
            segments_by_observation.get(_observation_key(row), [])
            for row in observations
        ]
        existing = [
            rows for rows in observation_segments if rows
        ]
        segment_rows = [row for rows in existing for row in rows]
        best_overlaps = [
            _number(row.get("best_matching_overlap")) for row in observations
        ]
        payload: dict[str, object] = {
            "field": field,
            "record_range": record_range,
            "observe_scenario": scenario,
            "observation_count": len(observations),
            "observation_denominator_scope": (
                "field_x_record_range_x_scenario_all_observed_columns"
                if include_range
                else "field_x_scenario_all_observed_columns"
            ),
            "existing_segment_observation_count": len(existing),
            "existing_segment_rate_all_observations": _ratio(
                len(existing),
                len(observations),
            ),
            "existing_segment_denominator_scope": (
                "observations_in_scope_with_at_least_one_preexisting_segment"
            ),
            "segment_row_count": len(segment_rows),
            "segment_row_denominator_scope": (
                "all_preexisting_segment_rows_for_observations_in_scope"
            ),
            "best_overlap_mean_all_observations": (
                statistics.fmean(best_overlaps) if best_overlaps else 0.0
            ),
            "best_overlap_median_all_observations": _percentile(
                best_overlaps,
                0.5,
            ),
            "best_overlap_p10_all_observations": _percentile(
                best_overlaps,
                0.1,
            ),
            "best_overlap_p25_all_observations": _percentile(
                best_overlaps,
                0.25,
            ),
            "best_overlap_p75_all_observations": _percentile(
                best_overlaps,
                0.75,
            ),
            "best_overlap_p90_all_observations": _percentile(
                best_overlaps,
                0.9,
            ),
            "gap_to_L_match_mean_all_observations": _mean(
                observations,
                "gap_to_L_match",
            ),
            "source_retention_mean_segment_rows": _mean(
                segment_rows,
                "source_retention_ratio",
            ),
            "creation_current_jaccard_mean_segment_rows": _mean(
                segment_rows,
                "creation_current_jaccard",
            ),
        }
        for threshold in range(1, 5):
            passing_counts = [
                sum(
                    int(_number(row.get("actual_overlap"))) >= threshold
                    for row in rows
                )
                for rows in existing
            ]
            passing_neuron_counts = [
                len(
                    {
                        str(row.get("segment_target_neuron"))
                        for row in rows
                        if int(_number(row.get("actual_overlap")))
                        >= threshold
                    }
                )
                for rows in existing
            ]
            match_count = sum(value > 0 for value in passing_counts)
            ambiguity_count = sum(value > 1 for value in passing_counts)
            multiple_neuron_count = sum(
                value > 1 for value in passing_neuron_counts
            )
            payload.update(
                {
                    f"L{threshold}_match_observation_count": match_count,
                    f"L{threshold}_match_rate_existing_segment_observations": (
                        _ratio(match_count, len(existing))
                    ),
                    f"L{threshold}_ambiguity_observation_count": (
                        ambiguity_count
                    ),
                    f"L{threshold}_ambiguity_rate_existing_segment_observations": (
                        _ratio(ambiguity_count, len(existing))
                    ),
                    f"L{threshold}_multiple_neuron_ambiguity_count": (
                        multiple_neuron_count
                    ),
                    f"L{threshold}_multiple_neuron_ambiguity_rate_existing_segment_observations": (
                        _ratio(multiple_neuron_count, len(existing))
                    ),
                }
            )
        for label, column in CONTEXT_COLUMNS.items():
            values = [
                max(
                    (int(_number(row.get(column))) for row in rows),
                    default=0,
                )
                for rows in existing
            ]
            payload[
                f"{label.lower()}_best_overlap_mean_existing_segment_observations"
            ] = statistics.fmean(values) if values else 0.0
        output.append(payload)
    return output


def _fixed_segment_summary(
    segments: Sequence[Mapping[str, object]],
    *,
    retention_only: bool,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(
        list
    )
    for row in segments:
        grouped[
            (
                str(row.get("field", "")),
                classify_record_range(row.get("actual_record_index")),
                str(row.get("observe_scenario", "")),
            )
        ].append(row)
    output = []
    for key, rows in sorted(grouped.items()):
        common = {
            "field": key[0],
            "record_range": key[1],
            "observe_scenario": key[2],
            "segment_row_count": len(rows),
            "denominator_scope": (
                "all_preexisting_segment_rows_for_field_x_range_x_scenario"
            ),
        }
        if retention_only:
            common.update(
                {
                    "source_retention_mean_segment_rows": _mean(
                        rows,
                        "source_retention_ratio",
                    ),
                    "current_source_retention_mean_segment_rows": _mean(
                        rows,
                        "current_source_retention_ratio",
                    ),
                    "creation_current_jaccard_mean_segment_rows": _mean(
                        rows,
                        "creation_current_jaccard",
                    ),
                    "creation_source_count_mean_segment_rows": _mean(
                        rows,
                        "creation_source_count",
                    ),
                }
            )
        else:
            source_total = sum(
                _number(row.get("segment_synapse_count")) for row in rows
            )
            common.update(
                {
                    "segment_synapse_count_mean": _mean(
                        rows,
                        "segment_synapse_count",
                    ),
                    "segment_actual_overlap_mean": _mean(
                        rows,
                        "actual_overlap",
                    ),
                    "weekday_source_ratio_all_segment_synapses": _ratio(
                        sum(
                            _number(row.get("source_weekday_count"))
                            for row in rows
                        ),
                        source_total,
                    ),
                    "time_source_ratio_all_segment_synapses": _ratio(
                        sum(
                            _number(row.get("source_time_count"))
                            for row in rows
                        ),
                        source_total,
                    ),
                    "passenger_source_ratio_all_segment_synapses": _ratio(
                        sum(
                            _number(row.get("source_passenger_count"))
                            for row in rows
                        ),
                        source_total,
                    ),
                }
            )
        output.append(common)
    return output


def _fixed_teacher_summary(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[
            (
                str(row.get("field", "")),
                classify_record_range(row.get("actual_record_index")),
                str(row.get("observe_scenario", "")),
            )
        ].append(row)
    return [
        {
            "field": key[0],
            "record_range": key[1],
            "observe_scenario": key[2],
            "reference_row_count": len(group),
            "denominator_scope": (
                "teacher_forced_observation_rows_field_x_range_x_scenario"
            ),
            "winner_reference_available_count": sum(
                _truth(row.get("observed_winner_available")) for row in group
            ),
            "preexisting_segment_reference_count": sum(
                str(row.get("reference_applicability", "")).startswith(
                    "PREEXISTING_"
                )
                for row in group
            ),
        }
        for key, group in sorted(grouped.items())
    ]


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
    )
    return (
        completed.stdout.strip()
        if completed.returncode == 0
        else "unknown"
    )


def _fixed_consistency_checks(
    *,
    columns: Sequence[Mapping[str, object]],
    segments: Sequence[Mapping[str, object]],
    teacher_rows: Sequence[Mapping[str, object]],
    field_rows: Sequence[Mapping[str, object]],
    range_rows: Sequence[Mapping[str, object]],
    old_summary: Mapping[str, object],
) -> dict[str, object]:
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    unknown = {
        "column_rows": sum(
            classify_record_range(row.get("actual_record_index"))
            == UNKNOWN_RECORD_RANGE
            for row in columns
        ),
        "segment_rows": sum(
            classify_record_range(row.get("actual_record_index"))
            == UNKNOWN_RECORD_RANGE
            for row in segments
        ),
        "teacher_rows": sum(
            classify_record_range(row.get("actual_record_index"))
            == UNKNOWN_RECORD_RANGE
            for row in teacher_rows
        ),
    }
    add(
        "no_unknown_record_ranges",
        not any(unknown.values()),
        json.dumps(unknown, sort_keys=True),
    )
    add(
        "range_observations_sum_to_columns",
        sum(int(row["observation_count"]) for row in range_rows)
        == len(columns),
        f"ranges={sum(int(row['observation_count']) for row in range_rows)}, "
        f"columns={len(columns)}",
    )
    add(
        "field_observations_sum_to_columns",
        sum(int(row["observation_count"]) for row in field_rows)
        == len(columns),
        f"fields={sum(int(row['observation_count']) for row in field_rows)}, "
        f"columns={len(columns)}",
    )
    scopes_valid = all(
        int(row["existing_segment_observation_count"])
        <= int(row["observation_count"])
        for row in (*field_rows, *range_rows)
    )
    add(
        "existing_segment_scope_is_subset",
        scopes_valid,
        "existing-segment observations never exceed all observations",
    )
    monotonic = True
    ambiguity_valid = True
    for row in (*field_rows, *range_rows):
        matches = [
            int(row[f"L{threshold}_match_observation_count"])
            for threshold in range(1, 5)
        ]
        monotonic &= matches == sorted(matches, reverse=True)
        for threshold in range(1, 5):
            match_count = int(row[f"L{threshold}_match_observation_count"])
            ambiguous = int(
                row[f"L{threshold}_ambiguity_observation_count"]
            )
            multiple_neuron = int(
                row[f"L{threshold}_multiple_neuron_ambiguity_count"]
            )
            ambiguity_valid &= (
                multiple_neuron <= ambiguous <= match_count
            )
    add(
        "lmatch_counts_are_monotonic",
        monotonic,
        "L1 >= L2 >= L3 >= L4 in every scope",
    )
    add(
        "ambiguity_is_bounded_by_matches",
        ambiguity_valid,
        "multiple-neuron ambiguity <= ambiguity <= matches",
    )
    passenger = [row for row in columns if row.get("field") == "passenger"]
    passenger_scenario3 = [
        row
        for row in passenger
        if row.get("observe_scenario") == "scenario3"
    ]
    current_totals = {
        "columns": len(columns),
        "segments": len(segments),
        "passenger_mean_best_overlap": _mean(
            passenger, "best_matching_overlap"
        ),
        "passenger_scenario3_rate": _ratio(
            len(passenger_scenario3), len(passenger)
        ),
        "passenger_scenario3_mean_best_overlap": _mean(
            passenger_scenario3, "best_matching_overlap"
        ),
    }
    comparable = {
        key: value
        for key, value in current_totals.items()
        if key in old_summary
    }
    totals_preserved = all(
        math.isclose(
            float(old_summary[key]),
            float(value),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for key, value in comparable.items()
    )
    add(
        "old_global_totals_preserved",
        totals_preserved,
        json.dumps(
            {
                key: {"old": old_summary[key], "fixed": value}
                for key, value in comparable.items()
            },
            sort_keys=True,
        ),
    )
    return {
        "record_range_fix_version": RECORD_RANGE_FIX_VERSION,
        "passed": all(bool(row["passed"]) for row in checks),
        "unknown_record_range_counts": unknown,
        "checks": checks,
    }


def _fixed_denominator_manifest() -> dict[str, object]:
    return {
        "record_range_source": "actual_record_index",
        "record_range_classifier": RECORD_RANGE_FIX_VERSION,
        "unknown_record_range_policy": (
            "invalid, missing, negative, and index >=245 are explicit "
            "UNKNOWN_RECORD_RANGE values and fail formal consistency checks"
        ),
        "files": {
            "match_overlap_field_summary_fixed.csv": {
                "row_unit": "field x observe_scenario",
                "primary_denominator": "all observed encoded columns in scope",
                "conditional_denominator": (
                    "observations with at least one preexisting segment"
                ),
            },
            "match_overlap_record_range_summary_fixed.csv": {
                "row_unit": "field x record_range x observe_scenario",
                "primary_denominator": "all observed encoded columns in scope",
                "conditional_denominator": (
                    "observations with at least one preexisting segment"
                ),
            },
            "segment_source_field_summary_fixed.csv": {
                "row_unit": "field x record_range x observe_scenario",
                "primary_denominator": "preexisting segment rows",
            },
            "source_retention_summary_fixed.csv": {
                "row_unit": "field x record_range x observe_scenario",
                "primary_denominator": "preexisting segment rows",
            },
            "teacher_reference_overlap_summary_fixed.csv": {
                "row_unit": "field x record_range x observe_scenario",
                "primary_denominator": "teacher-forced reference rows",
            },
            "lmatch_counterfactual_summary_fixed.csv": {
                "row_unit": (
                    "field x record_range x observe_scenario x threshold"
                ),
                "primary_denominator": (
                    "observations with at least one preexisting segment"
                ),
            },
            "lmatch_ambiguity_summary_fixed.csv": {
                "row_unit": (
                    "field x record_range x observe_scenario x threshold"
                ),
                "primary_denominator": (
                    "observations with at least one preexisting segment"
                ),
            },
            "context_variant_overlap_summary_fixed.csv": {
                "row_unit": (
                    "field x record_range x context_variant"
                ),
                "primary_denominator": (
                    "observations with at least one preexisting segment"
                ),
            },
        },
    }


SOURCE_CONTEXT_FIELDS = (
    "source_in_previous_winners",
    "source_in_current_active_cells",
    "source_in_predicted_cells",
    "source_in_burst_cells",
    "source_in_winner_only_context",
    "source_in_all_cell_context",
    "source_in_without_burst_context",
)


def _source_summary_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    include_range: bool,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, object]]] = defaultdict(
        list
    )
    for row in rows:
        key = [str(row.get("field", ""))]
        if include_range:
            key.append(
                classify_record_range(row.get("actual_record_index"))
            )
        key.append(str(row.get("observe_scenario", "")))
        grouped[tuple(key)].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        reasons = Counter(
            str(row.get("source_loss_reason", "UNKNOWN_SOURCE_LOSS"))
            for row in group
        )
        payload: dict[str, object] = {
            "field": key[0],
            "record_range": (
                key[1] if include_range else "ALL_RECORD_RANGES"
            ),
            "observe_scenario": key[-1],
            "source_synapse_rows": len(group),
            "denominator_scope": (
                "pre_observation_existing_segment_source_synapse_rows"
            ),
            "exact_source_match_count": reasons[
                "SOURCE_EXACT_CELL_MATCHED"
            ],
            "exact_source_match_rate": _ratio(
                reasons["SOURCE_EXACT_CELL_MATCHED"], len(group)
            ),
            "source_column_active_rate": _rate(
                group, "source_column_currently_active"
            ),
            "exact_neuron_active_rate": _rate(
                group, "exact_source_neuron_currently_active"
            ),
            "source_column_active_wrong_neuron_count": reasons[
                "SOURCE_COLUMN_ACTIVE_WRONG_NEURON"
            ],
            "source_column_active_wrong_neuron_rate": _ratio(
                reasons["SOURCE_COLUMN_ACTIVE_WRONG_NEURON"], len(group)
            ),
            "source_column_inactive_count": reasons[
                "SOURCE_COLUMN_NOT_ACTIVE"
            ],
            "source_column_inactive_rate": _ratio(
                reasons["SOURCE_COLUMN_NOT_ACTIVE"], len(group)
            ),
            "burst_only_availability_rate": _ratio(
                reasons["SOURCE_ONLY_AVAILABLE_THROUGH_BURST"],
                len(group),
            ),
            "predicted_only_availability_rate": _ratio(
                reasons["SOURCE_ONLY_AVAILABLE_THROUGH_PREDICTION"],
                len(group),
            ),
            "timing_eligibility_exclusion_rate": _ratio(
                reasons["SOURCE_TIMING_OR_ELIGIBILITY_EXCLUDED"],
                len(group),
            ),
            "unavailable_rate": _ratio(
                reasons["SOURCE_CELL_UNAVAILABLE"], len(group)
            ),
            "unknown_count": reasons["UNKNOWN_SOURCE_LOSS"],
            "unknown_rate": _ratio(
                reasons["UNKNOWN_SOURCE_LOSS"], len(group)
            ),
        }
        output.append(payload)
    return output


def _source_context_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output = []
    for key, group in sorted(
        _groups(
            rows,
            ("field", "record_range", "observe_scenario"),
        ).items()
    ):
        for field in SOURCE_CONTEXT_FIELDS:
            output.append(
                {
                    "field": key[0],
                    "record_range": key[1],
                    "observe_scenario": key[2],
                    "context_membership": field,
                    "source_synapse_rows": len(group),
                    "member_count": sum(
                        _truth(row.get(field)) for row in group
                    ),
                    "membership_rate": _rate(group, field),
                    "denominator_scope": (
                        "pre_observation_existing_segment_source_synapse_rows"
                    ),
                }
            )
    return output


def _source_identity_bootstrap(
    rows: Sequence[Mapping[str, object]],
    *,
    samples: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    output = []
    for field in ("weekday", "time", "passenger"):
        group = [row for row in rows if row.get("field") == field]
        values = [
            1.0
            if row.get("source_loss_reason")
            == "SOURCE_COLUMN_ACTIVE_WRONG_NEURON"
            else 0.0
            for row in group
        ]
        if not values:
            continue
        draws = [
            statistics.fmean(rng.choice(values) for _ in values)
            for _ in range(samples)
        ]
        output.append(
            {
                "field": field,
                "metric": "source_column_active_wrong_neuron_rate",
                "numerator": int(sum(values)),
                "denominator": len(values),
                "estimate": statistics.fmean(values),
                "ci_low": _percentile(draws, 0.025),
                "ci_high": _percentile(draws, 0.975),
                "bootstrap_samples": samples,
                "bootstrap_seed": seed,
                "bootstrap_unit": "source_synapse_row",
            }
        )
    return output


def _write_source_analysis(
    *,
    run_dir: Path,
    output_dir: Path,
    sources: Sequence[Mapping[str, object]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    field_rows = _source_summary_rows(sources, include_range=False)
    range_rows = _source_summary_rows(sources, include_range=True)
    context_rows = _source_context_rows(sources)
    reason_rows = _source_loss_summary(sources)
    bootstrap_rows = _source_identity_bootstrap(
        sources,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    _write(output_dir / "source_loss_reason_summary.csv", reason_rows)
    _write(output_dir / "source_field_summary.csv", field_rows)
    _write(output_dir / "source_record_range_summary.csv", range_rows)
    _write(
        output_dir / "source_context_membership_summary.csv",
        context_rows,
    )
    _write(output_dir / "source_identity_bootstrap_ci.csv", bootstrap_rows)

    unknown = sum(
        row.get("source_loss_reason") == "UNKNOWN_SOURCE_LOSS"
        for row in sources
    )
    trace_path = _find(run_dir, "match_overlap_source_trace")
    protocol = {
        "diagnostic_only": True,
        "offline_analysis_only": True,
        "source_trace_does_not_affect_matching": True,
        "source_trace_does_not_affect_learning": True,
        "ground_truth_does_not_affect_model": True,
        "capture_phase": "immediately_before_real_observe_code",
        "second_matching_call": False,
        "second_predict_call": False,
        "second_observe_call": False,
        "rng_calls_by_source_hook": False,
        "primary_reason_is_mutually_exclusive": True,
        "creation_support_class_availability": (
            "historical predicted/burst class unavailable; only creation "
            "context membership is recorded"
        ),
        "source_trace_sha256": _file_sha256(trace_path),
        "source_rows": len(sources),
        "unknown_rows": unknown,
        "unknown_rate": _ratio(unknown, len(sources)),
    }
    (output_dir / "source_trace_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    passenger_s3 = [
        row
        for row in sources
        if row.get("field") == "passenger"
        and row.get("observe_scenario") == "scenario3"
    ]
    passenger_late_s3 = [
        row
        for row in passenger_s3
        if classify_record_range(row.get("actual_record_index"))
        == "200-244"
    ]
    reasons = Counter(
        str(row.get("source_loss_reason")) for row in passenger_s3
    )
    dominant_reason = (
        reasons.most_common(1)[0][0] if reasons else "unavailable"
    )
    identity_supported = (
        dominant_reason == "SOURCE_COLUMN_ACTIVE_WRONG_NEURON"
        and unknown == 0
    )
    report = f"""# Source-Level Match-Overlap Report

This is a small-scale, read-only diagnostic. It is not a formal 250-record
result and does not change strict defaults, matching, observation, or learning.

## Coverage

- Source-synapse rows: {len(sources)}
- Unknown rows: {unknown}
- Unknown rate: {_ratio(unknown, len(sources)):.6f}
- Passenger Scenario-3 source rows: {len(passenger_s3)}
- Passenger records 200-244 Scenario-3 rows: {
    len(passenger_late_s3)
} (unavailable in runs shorter than 245 records)

## Passenger Scenario 3

The denominator below is all pre-observation source-synapse rows belonging to
existing passenger segments assigned to Scenario 3.

- Exact-cell match: {reasons["SOURCE_EXACT_CELL_MATCHED"]}/{len(passenger_s3)}
  ({_ratio(reasons["SOURCE_EXACT_CELL_MATCHED"], len(passenger_s3)):.6f})
- Active column, wrong neuron: {
    reasons["SOURCE_COLUMN_ACTIVE_WRONG_NEURON"]
}/{len(passenger_s3)}
  ({_ratio(reasons["SOURCE_COLUMN_ACTIVE_WRONG_NEURON"], len(passenger_s3)):.6f})
- Source column inactive: {reasons["SOURCE_COLUMN_NOT_ACTIVE"]}/{len(passenger_s3)}
  ({_ratio(reasons["SOURCE_COLUMN_NOT_ACTIVE"], len(passenger_s3)):.6f})
- Timing/eligibility excluded: {
    reasons["SOURCE_TIMING_OR_ELIGIBILITY_EXCLUDED"]
}/{len(passenger_s3)}
  ({_ratio(reasons["SOURCE_TIMING_OR_ELIGIBILITY_EXCLUDED"], len(passenger_s3)):.6f})

## Diagnostic judgment

- Dominant passenger Scenario-3 reason: `{dominant_reason}`
- Neuron identity drift supported by this small run: `{identity_supported}`

Neuron identity drift requires active-column/wrong-neuron loss to dominate,
passenger to exceed weekday/time with its bootstrap interval, and a low unknown
rate. When timing exclusion or inactive columns dominate, the evidence instead
points toward matching-timing semantics or context-trajectory drift. This
50-record result remains a small diagnostic and cannot replace a formal
250-record source trace.
"""
    (output_dir / "SOURCE_LEVEL_MATCH_OVERLAP_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )
    return protocol


def _write_fixed_analysis(
    *,
    run_dir: Path,
    output_dir: Path,
    columns: Sequence[Mapping[str, object]],
    segments: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    teacher_path = _find(run_dir, "teacher_forced_observation_trace")
    teacher_rows = _read(teacher_path)
    field_rows = _fixed_scope_rows(
        columns, segments, include_range=False
    )
    range_rows = _fixed_scope_rows(
        columns, segments, include_range=True
    )
    segment_rows = _fixed_segment_summary(
        segments, retention_only=False
    )
    retention_rows = _fixed_segment_summary(
        segments, retention_only=True
    )
    teacher_summary = _fixed_teacher_summary(teacher_rows)
    lmatch_rows, ambiguity_rows = _lmatch_summaries(segments)
    context_rows = _context_variant_summary(segments)
    fixed_outputs = {
        "match_overlap_field_summary_fixed.csv": field_rows,
        "match_overlap_record_range_summary_fixed.csv": range_rows,
        "segment_source_field_summary_fixed.csv": segment_rows,
        "source_retention_summary_fixed.csv": retention_rows,
        "teacher_reference_overlap_summary_fixed.csv": teacher_summary,
        "lmatch_counterfactual_summary_fixed.csv": lmatch_rows,
        "lmatch_ambiguity_summary_fixed.csv": ambiguity_rows,
        "context_variant_overlap_summary_fixed.csv": context_rows,
    }
    for filename, rows in fixed_outputs.items():
        _write(output_dir / filename, rows)

    old_dir = run_dir.parent / "analysis"
    old_summary_path = old_dir / "match_overlap_summary.json"
    old_summary = (
        json.loads(old_summary_path.read_text(encoding="utf-8"))
        if old_summary_path.exists()
        else {}
    )
    checks = _fixed_consistency_checks(
        columns=columns,
        segments=segments,
        teacher_rows=teacher_rows,
        field_rows=field_rows,
        range_rows=range_rows,
        old_summary=old_summary,
    )
    manifest = _fixed_denominator_manifest()
    (output_dir / "denominator_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "analysis_consistency_checks.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    passenger_late_s3 = next(
        (
            row
            for row in range_rows
            if row["field"] == "passenger"
            and row["record_range"] == "200-244"
            and row["observe_scenario"] == "scenario3"
        ),
        None,
    )
    trace_paths = {
        "column": _find(run_dir, "match_overlap_column_trace"),
        "segment": _find(run_dir, "match_overlap_segment_trace"),
        "teacher": teacher_path,
    }
    trace_hashes = {
        name: _file_sha256(path) for name, path in trace_paths.items()
    }
    protocol_path = run_dir / "protocol.json"
    protocol = (
        json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol_path.exists()
        else {}
    )
    report = f"""# Fig.9 Match Overlap Decomposition (Record Range Fixed)

This is an offline re-analysis of existing traces. The model was not rerun.
No prediction, matching, learning, RNG, checkpoint, or strict default changed.

## Audit identity

- Fix version: `{RECORD_RANGE_FIX_VERSION}`
- Analysis-code commit: `{_git_head()}`
- Trace-producing commit: `{protocol.get("git_commit_sha", "unknown")}`
- Old analysis directory: `{old_dir}`
- Fixed analysis directory: `{output_dir}`
- Column trace SHA256: `{trace_hashes["column"]}`
- Segment trace SHA256: `{trace_hashes["segment"]}`
- Teacher trace SHA256: `{trace_hashes["teacher"]}`

## Record-range correction

The old generic grouping path read a nonexistent `record_range` column and
coerced the missing value to zero. The fixed path classifies only
`actual_record_index`: 0-49, 50-99, 100-149, 150-199, and 200-244. Invalid,
missing, negative, and out-of-protocol values remain
`UNKNOWN_RECORD_RANGE`; they are never silently assigned to 0-49.

## Denominators

All-observation metrics use encoded-column observations in the named
field/range/scenario scope. L-match, ambiguity, and context-variant rates are
conditional on observations that had at least one preexisting segment.
Segment composition and retention use individual preexisting segment rows.
Teacher-reference metrics use teacher-forced reference rows. Exact definitions
are machine-readable in `denominator_manifest.json`.

## Late passenger range

{json.dumps(passenger_late_s3, indent=2, sort_keys=True)}

## Consistency

- All checks passed: `{checks["passed"]}`
- Unknown range counts: `{json.dumps(checks["unknown_record_range_counts"], sort_keys=True)}`
- Existing global totals preserved: `{
    next(
        row["passed"]
        for row in checks["checks"]
        if row["name"] == "old_global_totals_preserved"
    )
}`

These outputs correct grouping and denominator labels only. They are not a new
trajectory and do not establish a new MAPE or a completed reproduction.
"""
    (output_dir / "FIG9_MATCH_OVERLAP_DECOMPOSITION_REPORT_FIXED.md").write_text(
        report,
        encoding="utf-8",
    )
    if not checks["passed"]:
        failed = [
            row["name"] for row in checks["checks"] if not row["passed"]
        ]
        raise AssertionError(
            "fixed analysis consistency checks failed: "
            + ", ".join(failed)
        )
    return {
        "record_range_fix_version": RECORD_RANGE_FIX_VERSION,
        "trace_hashes": trace_hashes,
        "checks_passed": True,
    }


def analyze(
    *,
    run_dir: Path,
    output_dir: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
    fixed_only: bool = False,
) -> dict[str, object]:
    columns = _read(_find(run_dir, "match_overlap_column_trace"))
    segments = _read(_find(run_dir, "match_overlap_segment_trace"))
    sources = _read(_find(run_dir, "match_overlap_source_trace"))
    if not columns:
        raise ValueError("match_overlap_column_trace is required")
    if fixed_only:
        return _write_fixed_analysis(
            run_dir=run_dir,
            output_dir=output_dir,
            columns=columns,
            segments=segments,
        )

    field_summary = _column_summary(columns, ("field", "observe_scenario"))
    range_summary = _column_summary(
        columns,
        ("field", "record_range", "observe_scenario"),
    )
    passenger_distribution = _passenger_distribution(columns)
    segment_fields = _segment_source_summary(segments)
    retention = _retention_summary(segments)
    source_loss = _source_loss_summary(sources)
    context_variants = _context_variant_summary(segments)
    lmatch, ambiguity = _lmatch_summaries(segments)
    teacher = _teacher_summary(segments)
    bootstrap = _bootstrap(
        columns,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )

    named = {
        "match_overlap_field_summary.csv": field_summary,
        "match_overlap_record_range_summary.csv": range_summary,
        "passenger_overlap_distribution.csv": passenger_distribution,
        "segment_source_field_summary.csv": segment_fields,
        "source_retention_summary.csv": retention,
        "source_loss_reason_summary.csv": source_loss,
        "context_variant_overlap_summary.csv": context_variants,
        "lmatch_counterfactual_summary.csv": lmatch,
        "lmatch_ambiguity_summary.csv": ambiguity,
        "teacher_reference_overlap_summary.csv": teacher,
        "match_overlap_bootstrap_ci.csv": bootstrap,
    }
    for filename, rows in named.items():
        _write(output_dir / filename, rows)
    fixed_summary = _write_fixed_analysis(
        run_dir=run_dir,
        output_dir=output_dir,
        columns=columns,
        segments=segments,
    )
    source_summary = (
        _write_source_analysis(
            run_dir=run_dir,
            output_dir=output_dir,
            sources=sources,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        if sources
        else None
    )

    passenger_late = [
        row
        for row in columns
        if row.get("field") == "passenger"
        and _record_range(row.get("actual_record_index")) == "200-244"
    ]
    unknown = [
        row
        for row in sources
        if row.get("source_missing_reason") == "UNKNOWN_SOURCE_LOSS"
    ]
    passenger = [
        row for row in columns if row.get("field") == "passenger"
    ]
    passenger_scenario3 = [
        row
        for row in passenger
        if row.get("observe_scenario") == "scenario3"
    ]
    summary = {
        "diagnostic_only": True,
        "offline_analysis_only": True,
        "columns": len(columns),
        "segments": len(segments),
        "sources": len(sources),
        "passenger_mean_best_overlap": _mean(
            passenger,
            "best_matching_overlap",
        ),
        "passenger_scenario3_rate": (
            len(passenger_scenario3) / len(passenger)
            if passenger
            else 0.0
        ),
        "passenger_scenario3_mean_best_overlap": _mean(
            passenger_scenario3,
            "best_matching_overlap",
        ),
        "passenger_200_244_available": bool(passenger_late),
        "passenger_200_244_mean_best_overlap": (
            _mean(passenger_late, "best_matching_overlap")
            if passenger_late
            else None
        ),
        "unknown_source_loss_rate": (
            len(unknown) / len(sources) if sources else None
        ),
        "formal_250_not_run_by_analyzer": True,
        "record_range_fix_version": fixed_summary[
            "record_range_fix_version"
        ],
        "fixed_analysis_checks_passed": fixed_summary["checks_passed"],
        "input_trace_sha256": fixed_summary["trace_hashes"],
        "source_trace_protocol": source_summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "match_overlap_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    late_text = (
        f"{summary['passenger_200_244_mean_best_overlap']:.6f}"
        if passenger_late
        else "unavailable (this run does not contain records 200-244)"
    )
    scenario3_lmatch = {
        int(row["counterfactual_L_match"]): row
        for row in lmatch
        if row["field"] == "passenger"
        and row["record_range"] == "0-49"
        and row["observe_scenario"] == "scenario3"
    }
    scenario3_ambiguity = {
        int(row["counterfactual_L_match"]): row
        for row in ambiguity
        if row["field"] == "passenger"
        and row["record_range"] == "0-49"
        and row["observe_scenario"] == "scenario3"
    }
    context_passenger = {
        str(row["context_variant"]): row
        for row in context_variants
        if row["field"] == "passenger"
        and row["record_range"] == "0-49"
    }
    lmatch_lines = "\n".join(
        (
            f"| {threshold} | "
            f"{scenario3_lmatch[threshold]['columns_with_any_passing_segment_rate']:.6f} | "
            f"{scenario3_ambiguity[threshold]['ambiguous_column_rate']:.6f} |"
        )
        for threshold in range(1, 5)
        if threshold in scenario3_lmatch
        and threshold in scenario3_ambiguity
    )
    context_lines = "\n".join(
        (
            f"| {label} | {row['mean_best_overlap']:.6f} | "
            f"{row['ambiguous_segment_rate_L4']:.6f} |"
        )
        for label, row in sorted(context_passenger.items())
    )
    report = f"""# Fig.9 Match Overlap Decomposition

This is a diagnostic-only, offline analysis. It does not modify prediction,
matching, learning, `L_match`, or strict defaults.

## Trace coverage

- Observed columns: {len(columns)}
- Existing segments: {len(segments)}
- Source synapses: {len(sources)}
- Passenger mean best overlap: {summary['passenger_mean_best_overlap']:.6f}
- Passenger Scenario-3 rate: {summary['passenger_scenario3_rate']:.6f}
- Passenger Scenario-3 mean best overlap: {summary['passenger_scenario3_mean_best_overlap']:.6f}
- Passenger records 200-244 mean best overlap: {late_text}
- Unknown source-loss rate: {summary['unknown_source_loss_rate']}

## Passenger Scenario-3 offline L_match

This table is conditional on columns that already have at least one segment.

| L | any passing segment rate | ambiguous column rate |
|---:|---:|---:|
{lmatch_lines or '| n/a | n/a | n/a |'}

## Passenger context variants

This table is also conditional on columns with existing segments.

| context | mean best overlap | L4 ambiguous segment rate |
|---|---:|---:|
{context_lines or '| n/a | n/a | n/a |'}

## Interpretation guardrails

The L1-L4 and context-variant tables are fixed-state counterfactuals. They are
not new trajectories and do not provide a new MAPE. A lower threshold may
increase both reusable and ambiguous segments, so match rate must be read with
the ambiguity and neuron-count tables.

Records 200-244 are not inferred from a shorter run. A real L_match or
burst-context ablation is not supported until the manually run 250-record
trace confirms retention, timing loss, and ambiguity in that late range.
"""
    (output_dir / "FIG9_MATCH_OVERLAP_DECOMPOSITION_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--fixed-only",
        action="store_true",
        help="write only the record-range-fixed audit outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        fixed_only=args.fixed_only,
    )


if __name__ == "__main__":
    main()
