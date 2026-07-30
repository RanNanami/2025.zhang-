"""Offline analysis for the Fig.9 match-overlap decomposition trace."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import statistics
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


def _record_range(index: object) -> str:
    value = int(_number(index))
    for start, stop, label in RECORD_RANGES:
        if start <= value <= stop:
            return label
    return "other"


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


def _groups(
    rows: Iterable[Mapping[str, object]],
    keys: Sequence[str],
) -> dict[tuple[str, ...], list[Mapping[str, object]]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            tuple(
                _record_range(row.get(key))
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


def analyze(
    *,
    run_dir: Path,
    output_dir: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    columns = _read(_find(run_dir, "match_overlap_column_trace"))
    segments = _read(_find(run_dir, "match_overlap_segment_trace"))
    sources = _read(_find(run_dir, "match_overlap_source_trace"))
    if not columns:
        raise ValueError("match_overlap_column_trace is required")

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()
