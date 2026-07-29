"""Offline summary of real-observation Scenario 1/2/3 assignments."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


MARKERS = {
    "diagnostic_only": True,
    "offline_analysis_only": True,
    "observe_scenario_assignment_only": True,
    "selection_behavior_changed": False,
    "ground_truth_does_not_affect_prediction": True,
    "diagnostic_trace_does_not_affect_selection": True,
    "future_observation_does_not_affect_past_prediction": True,
}


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _number(value: object) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _record_range(record_index: int) -> str:
    start = min((record_index // 50) * 50, 200)
    end = 244 if start == 200 else start + 49
    return f"{start}-{end}"


def _percentile(values: Sequence[float], fraction: float) -> float | str:
    if not values:
        return ""
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def normalize_assignment_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output = []
    for row in rows:
        record_index = int(row["actual_record_index"])
        reason = str(row.get("scenario_assignment_reason", ""))
        output.append(
            {
                **MARKERS,
                **row,
                "record_range": _record_range(record_index),
                "scenario3_reason_available": bool(reason),
                "scenario3_reason": (
                    reason
                    if reason
                    else "TRACE_FIELD_UNAVAILABLE"
                ),
            }
        )
    return output


def summarize_assignments(
    rows: Sequence[Mapping[str, object]],
    *,
    group_fields: Sequence[str] = ("field", "record_range", "observe_scenario"),
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[field]) for field in group_fields)].append(row)
    output = []
    total_fields = tuple(
        field for field in group_fields if field != "observe_scenario"
    )
    totals: Counter[tuple[str, ...]] = Counter(
        tuple(str(row[field]) for field in total_fields) for row in rows
    )
    for group, items in sorted(grouped.items()):
        payload = dict(zip(group_fields, group))
        overlaps = [
            value
            for row in items
            if (
                value := _number(
                    row.get("best_matching_segment_overlap")
                )
            )
            is not None
        ]
        gaps = [
            value
            for row in items
            if (value := _number(row.get("gap_to_L_match"))) is not None
        ]
        contexts = [
            value
            for row in items
            if (value := _number(row.get("previous_context_size"))) is not None
        ]
        jaccards = [
            value
            for row in items
            if (value := _number(row.get("matching_source_jaccard"))) is not None
        ]
        output.append(
            {
                **payload,
                "scenario": payload.get("observe_scenario", ""),
                "count": len(items),
                "scenario_rate_within_group": (
                    len(items)
                    / totals[
                        tuple(str(items[0][field]) for field in total_fields)
                    ]
                ),
                "no_predictive_cell_rate": statistics.fmean(
                    not _bool(row.get("active_column_was_predicted"))
                    for row in items
                ),
                "no_segment_rate": statistics.fmean(
                    row.get("scenario3_reason")
                    == "NO_EXISTING_SEGMENT_IN_COLUMN"
                    for row in items
                ),
                "below_L_match_rate": statistics.fmean(
                    row.get("scenario3_reason")
                    == "EXISTING_SEGMENTS_BELOW_L_MATCH"
                    for row in items
                ),
                "best_overlap_mean": (
                    statistics.fmean(overlaps) if overlaps else ""
                ),
                "best_overlap_median": (
                    statistics.median(overlaps) if overlaps else ""
                ),
                "best_overlap_p10": _percentile(overlaps, 0.10),
                "best_overlap_p90": _percentile(overlaps, 0.90),
                "gap_to_L_match_mean": (
                    statistics.fmean(gaps) if gaps else ""
                ),
                "previous_context_size_mean": (
                    statistics.fmean(contexts) if contexts else ""
                ),
                "source_context_jaccard_mean": (
                    statistics.fmean(jaccards) if jaccards else ""
                ),
                "newly_created_segment_rate": statistics.fmean(
                    _bool(row.get("created_new_segment"))
                    or _bool(row.get("segment_created_after_observation"))
                    or bool(row.get("created_segment_id"))
                    for row in items
                ),
                "newly_selected_burst_neuron_rate": statistics.fmean(
                    _bool(row.get("created_new_neuron_identity"))
                    for row in items
                ),
                "predictive_cell_available_rate": statistics.fmean(
                    _bool(row.get("active_column_was_predicted"))
                    for row in items
                ),
                "any_existing_segment_rate": statistics.fmean(
                    _bool(row.get("column_has_any_segment"))
                    for row in items
                ),
                "matching_segment_rate": statistics.fmean(
                    int(row.get("matching_segment_count", 0) or 0) > 0
                    for row in items
                ),
                "invalid_predicted_time_rate": statistics.fmean(
                    row.get("scenario3_reason") == "PREDICTED_TIME_INVALID"
                    for row in items
                ),
                "segment_reuse_rate": statistics.fmean(
                    bool(row.get("reinforced_segment_id"))
                    for row in items
                ),
                "reason_trace_available_rate": statistics.fmean(
                    _bool(row.get("scenario3_reason_available"))
                    for row in items
                ),
            }
        )
    return output


def bootstrap_scenario3(
    rows: Sequence[Mapping[str, object]],
    *,
    samples: int,
    seed: int,
) -> list[dict[str, object]]:
    by_record: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_record[int(row["actual_record_index"])].append(row)
    record_ids = sorted(by_record)
    if not record_ids:
        return []
    rng = random.Random(seed)
    output = []
    for field in ("weekday", "time", "passenger"):
        draws = []
        for _ in range(samples):
            sampled = [
                row
                for _record in record_ids
                for row in by_record[
                    record_ids[rng.randrange(len(record_ids))]
                ]
                if row["field"] == field
            ]
            draws.append(
                statistics.fmean(
                    row["observe_scenario"] == "scenario3"
                    for row in sampled
                )
            )
        draws.sort()
        output.append(
            {
                "analysis": "observe_scenario3_rate",
                "field": field,
                "bootstrap_unit": "actual_record_index",
                "samples": samples,
                "seed": seed,
                "mean": statistics.fmean(draws),
                "ci_2_5": draws[int(0.025 * (samples - 1))],
                "ci_97_5": draws[int(0.975 * (samples - 1))],
            }
        )
    return output


def analyze(
    *,
    observation_trace: Path,
    output_dir: Path,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    rows = normalize_assignment_rows(_read(observation_trace))
    field_summary = summarize_assignments(
        rows,
        group_fields=("field", "observe_scenario"),
    )
    record_range_summary = summarize_assignments(rows)
    scenario3_rows = [
        row for row in rows if row["observe_scenario"] == "scenario3"
    ]
    passenger_scenario3 = [
        row
        for row in rows
        if row["field"] == "passenger"
        and row["observe_scenario"] == "scenario3"
    ]
    reason_summary = [
        {
            "field": field,
            "reason": reason,
            "count": count,
            "fraction": (
                count
                / sum(
                    1 for row in scenario3_rows if row["field"] == field
                )
                if scenario3_rows
                else 0.0
            ),
        }
        for field in ("weekday", "time", "passenger")
        for reason, count in sorted(
            Counter(
                str(row["scenario3_reason"])
                for row in scenario3_rows
                if row["field"] == field
            ).items()
        )
    ]
    passenger_reason_summary = [
        row for row in reason_summary if row["field"] == "passenger"
    ]
    overlap_summary = []
    segment_summary = []
    for field in ("weekday", "time", "passenger"):
        items = [row for row in scenario3_rows if row["field"] == field]
        overlaps = [
            value
            for row in items
            if (
                value := _number(
                    row.get(
                        "best_matching_overlap",
                        row.get("best_matching_segment_overlap"),
                    )
                )
            )
            is not None
        ]
        counts = [
            value
            for row in items
            if (
                value := _number(
                    row.get("existing_segment_count_in_column")
                )
            )
            is not None
        ]
        overlap_summary.append(
            {
                "field": field,
                "rows": len(items),
                "mean": statistics.fmean(overlaps) if overlaps else "",
                "median": statistics.median(overlaps) if overlaps else "",
                "p10": _percentile(overlaps, 0.10),
                "p90": _percentile(overlaps, 0.90),
            }
        )
        segment_summary.append(
            {
                "field": field,
                "rows": len(items),
                "no_existing_segment_rate": (
                    statistics.fmean(value == 0 for value in counts)
                    if counts
                    else ""
                ),
                "existing_segment_count_mean": (
                    statistics.fmean(counts) if counts else ""
                ),
                "existing_segment_count_median": (
                    statistics.median(counts) if counts else ""
                ),
                "forgetting_evidence_available_rate": (
                    statistics.fmean(
                        _bool(row.get("forgetting_evidence_available"))
                        for row in items
                    )
                    if items
                    else 0.0
                ),
            }
        )
    bootstrap = bootstrap_scenario3(
        rows,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    _write(output_dir / "observe_scenario_assignment_rows.csv", rows)
    _write(output_dir / "observe_scenario_field_summary.csv", field_summary)
    _write(
        output_dir / "observe_scenario_record_range_summary.csv",
        record_range_summary,
    )
    _write(output_dir / "scenario3_reason_summary.csv", reason_summary)
    _write(
        output_dir / "scenario3_matching_overlap_summary.csv",
        overlap_summary,
    )
    _write(
        output_dir / "scenario3_segment_availability_summary.csv",
        segment_summary,
    )
    _write(output_dir / "scenario3_bootstrap_ci.csv", bootstrap)
    _write(
        output_dir / "passenger_scenario3_reason_summary.csv",
        passenger_reason_summary,
    )
    summary = {
        **MARKERS,
        "observation_rows": len(rows),
        "scenario_counts": dict(
            Counter(str(row["observe_scenario"]) for row in rows)
        ),
        "passenger_scenario3_rows": len(passenger_scenario3),
        "passenger_scenario3_reasons": passenger_reason_summary,
        "reason_trace_available_rate": (
            statistics.fmean(
                _bool(row["scenario3_reason_available"]) for row in rows
            )
            if rows
            else 0.0
        ),
        "bootstrap": bootstrap,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "observe_scenario_assignment_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_lines = [
        "# Observe Scenario Reason Diagnostic",
        "",
        "This is a read-only, nonpaper diagnostic. Observation reasons are "
        "copied from the branch that actually assigned Scenario 1/2/3.",
        "",
        f"Rows: {len(rows)}",
        f"Scenario counts: {summary['scenario_counts']}",
        f"Reason trace available rate: "
        f"{summary['reason_trace_available_rate']:.6f}",
        "",
        "Forgetting evidence is reported as unavailable unless the trace "
        "contains direct deletion history; no cause is inferred from the "
        "final model.",
        "",
        "## Passenger Scenario 3 Reasons",
        "",
    ]
    report_lines.extend(
        f"- `{row['reason']}`: {row['count']} ({row['fraction']:.6f})"
        for row in passenger_reason_summary
    )
    (output_dir / "OBSERVE_SCENARIO_REASON_REPORT.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-trace", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze(
        observation_trace=Path(args.observation_trace),
        output_dir=Path(args.output_dir),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()
