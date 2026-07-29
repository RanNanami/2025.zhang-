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
                "record_range": (
                    f"{(record_index // 25) * 25}-"
                    f"{(record_index // 25) * 25 + 24}"
                ),
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
) -> list[dict[str, object]]:
    grouped: dict[
        tuple[str, str, str],
        list[Mapping[str, object]],
    ] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["field"]),
                str(row["record_range"]),
                str(row["observe_scenario"]),
            )
        ].append(row)
    output = []
    for (field, record_range, scenario), items in sorted(grouped.items()):
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
                "field": field,
                "record_range": record_range,
                "scenario": scenario,
                "count": len(items),
                "scenario_rate_within_group": "",
                "no_predictive_cell_rate": statistics.fmean(
                    not _bool(row.get("correctly_predicted_cell_available"))
                    for row in items
                ),
                "no_segment_rate": statistics.fmean(
                    row.get("scenario3_reason") == "NO_EXISTING_SEGMENT"
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
                    _bool(row.get("segment_created_after_observation"))
                    or bool(row.get("created_segment_id"))
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
    totals: Counter[tuple[str, str]] = Counter(
        (str(row["field"]), str(row["record_range"])) for row in rows
    )
    for row in output:
        row["scenario_rate_within_group"] = (
            int(row["count"])
            / totals[(str(row["field"]), str(row["record_range"]))]
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
    field_summary = summarize_assignments(rows)
    passenger_scenario3 = [
        row
        for row in rows
        if row["field"] == "passenger"
        and row["observe_scenario"] == "scenario3"
    ]
    reason_counts = Counter(
        str(row["scenario3_reason"]) for row in passenger_scenario3
    )
    reason_summary = [
        {
            "reason": reason,
            "count": count,
            "fraction": (
                count / len(passenger_scenario3)
                if passenger_scenario3
                else 0.0
            ),
        }
        for reason, count in sorted(reason_counts.items())
    ]
    bootstrap = bootstrap_scenario3(
        rows,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    _write(output_dir / "observe_scenario_assignment_rows.csv", rows)
    _write(output_dir / "observe_scenario_field_summary.csv", field_summary)
    _write(
        output_dir / "passenger_scenario3_reason_summary.csv",
        reason_summary,
    )
    summary = {
        **MARKERS,
        "observation_rows": len(rows),
        "scenario_counts": dict(
            Counter(str(row["observe_scenario"]) for row in rows)
        ),
        "passenger_scenario3_rows": len(passenger_scenario3),
        "passenger_scenario3_reasons": reason_summary,
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
