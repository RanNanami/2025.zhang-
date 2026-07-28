"""Compare sequential and batched Fig.9 oracle diagnostics offline.

Ground truth is read only after both prediction runs have completed. The tool
strictly aligns rows and never imports or calls the competition implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


ALIGNMENT_FIELDS = (
    "prediction_input_index",
    "horizon_step",
    "target_timestamp",
)


def read_trace(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def align_traces(
    sequential_rows: Sequence[Mapping[str, str]],
    batched_rows: Sequence[Mapping[str, str]],
) -> list[tuple[Mapping[str, str], Mapping[str, str]]]:
    """Return one-to-one aligned rows or reject any missing/duplicate key."""

    sequential = _index_rows(sequential_rows, "sequential")
    batched = _index_rows(batched_rows, "batched")
    if set(sequential) != set(batched):
        missing_batched = sorted(set(sequential) - set(batched))
        missing_sequential = sorted(set(batched) - set(sequential))
        raise ValueError(
            "oracle trace alignment mismatch: "
            f"missing_batched={missing_batched[:5]}, "
            f"missing_sequential={missing_sequential[:5]}"
        )
    return [(sequential[key], batched[key]) for key in sorted(sequential)]


def compare_policy_rows(
    sequential_rows: Sequence[Mapping[str, str]],
    batched_rows: Sequence[Mapping[str, str]],
    *,
    sequential_runtime: float | None = None,
    batched_runtime: float | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Build aligned row evidence and horizon-level comparison metrics."""

    aligned = align_traces(sequential_rows, batched_rows)
    comparison_rows: list[dict[str, object]] = []
    for sequential, batched in aligned:
        sequential_targets = _columns(
            sequential.get("emitted_target_total_columns", "")
        )
        batched_targets = _columns(
            batched.get("emitted_target_total_columns", "")
        )
        sequential_false = _columns(
            sequential.get("emitted_false_columns", "")
        )
        batched_false = _columns(batched.get("emitted_false_columns", ""))
        rescued_targets = batched_targets - sequential_targets
        rescued_false = batched_false - sequential_false
        comparison_rows.append(
            {
                "diagnostic_only": True,
                "competition_is_local_choice": True,
                "uses_ground_truth_for_analysis_only": True,
                "prediction_input_index": int(
                    sequential["prediction_input_index"]
                ),
                "horizon_step": int(sequential["horizon_step"]),
                "target_timestamp": sequential["target_timestamp"],
                "sequential_emitted_target_hits": len(sequential_targets),
                "batched_emitted_target_hits": len(batched_targets),
                "target_rescued_by_batching": len(rescued_targets),
                "false_rescued_by_batching": len(rescued_false),
                "target_rescued_columns": _columns_text(rescued_targets),
                "false_rescued_columns": _columns_text(rescued_false),
                "target_suppressed_only_due_to_same_batch_order": _integer(
                    sequential.get(
                        "target_suppressed_only_due_to_same_batch_order",
                        "0",
                    )
                ),
                "target_suppressed_by_earlier_batch": _integer(
                    batched.get("target_suppressed_by_earlier_batch", "0")
                ),
                "target_best_candidate_batch_index": batched.get(
                    "target_best_candidate_batch_index", ""
                ),
                "false_best_candidate_batch_index": batched.get(
                    "false_best_candidate_batch_index", ""
                ),
            }
        )

    step_summary = []
    for step in sorted(
        {int(row["horizon_step"]) for row in comparison_rows}
    ):
        indexes = [
            index
            for index, row in enumerate(comparison_rows)
            if int(row["horizon_step"]) == step
        ]
        sequential = [aligned[index][0] for index in indexes]
        batched = [aligned[index][1] for index in indexes]
        paired = [comparison_rows[index] for index in indexes]
        step_summary.append(
            {
                "horizon_step": step,
                "sequential": _step_metrics(
                    sequential,
                    runtime=sequential_runtime,
                ),
                "batched": _step_metrics(
                    batched,
                    runtime=batched_runtime,
                ),
                "target_rescued_by_batching_count": sum(
                    int(row["target_rescued_by_batching"])
                    for row in paired
                ),
                "false_rescued_by_batching_count": sum(
                    int(row["false_rescued_by_batching"])
                    for row in paired
                ),
                "target_rescued_by_batching_rate": _ratio(
                    sum(
                        int(row["target_rescued_by_batching"])
                        for row in paired
                    ),
                    sum(
                        _integer(
                            row.get("target_total_column_count", "0")
                        )
                        for row in batched
                    ),
                ),
                "false_rescued_by_batching_rate": _ratio(
                    sum(
                        int(row["false_rescued_by_batching"])
                        for row in paired
                    ),
                    sum(
                        _integer(row.get("raw_false_column_count", "0"))
                        for row in batched
                    ),
                ),
            }
        )

    return comparison_rows, {
        "diagnostic_only": True,
        "competition_is_local_choice": True,
        "uses_ground_truth_for_analysis_only": True,
        "alignment_fields": list(ALIGNMENT_FIELDS),
        "aligned_rows": len(aligned),
        "target_rescued_by_batching_count": sum(
            int(row["target_rescued_by_batching"])
            for row in comparison_rows
        ),
        "false_rescued_by_batching_count": sum(
            int(row["false_rescued_by_batching"])
            for row in comparison_rows
        ),
        "step_summary": step_summary,
    }


def write_comparison(
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "competition_policy_comparison.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "competition_policy_comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _step_metrics(
    rows: Sequence[Mapping[str, str]],
    *,
    runtime: float | None,
) -> dict[str, object]:
    available = [
        row for row in rows if _boolean(row.get("prediction_available", ""))
    ]
    absolute_error = sum(
        _number(row.get("absolute_error", ""))
        for row in rows
        if row.get("absolute_error", "") != ""
    )
    targets = sum(
        abs(_number(row.get("target_passenger", "")))
        for row in rows
        if row.get("absolute_error", "") != ""
    )
    raw_columns = [
        _number(row.get("raw_candidate_column_count", "0")) for row in rows
    ]
    emitted_columns = [
        _number(row.get("emitted_column_count", "0")) for row in rows
    ]
    raw_total = sum(raw_columns)
    emitted_total = sum(emitted_columns)
    return {
        "rows": len(rows),
        "mape": _ratio(absolute_error, targets),
        "coverage": _ratio(len(available), len(rows)),
        "raw_column_mean": _ratio(raw_total, len(rows)),
        "emitted_column_mean": _ratio(emitted_total, len(rows)),
        "suppression_ratio": (
            1.0 - emitted_total / raw_total if raw_total else 0.0
        ),
        "raw_target_recall": _field_recall(rows, "raw", "total"),
        "emitted_target_recall": _field_recall(
            rows, "emitted", "total"
        ),
        "passenger_emitted_target_recall": _field_recall(
            rows, "emitted", "passenger"
        ),
        "target_suppression_ratio": _mean(
            rows, "target_suppression_ratio"
        ),
        "false_emitted_ratio": _mean(
            rows, "emitted_false_column_ratio"
        ),
        "target_present_but_suppressed_rate": _ratio(
            sum(
                row.get("classification")
                == "TARGET_PRESENT_BUT_SUPPRESSED"
                for row in rows
            ),
            len(rows),
        ),
        "runtime_seconds": runtime,
    }


def _index_rows(
    rows: Sequence[Mapping[str, str]],
    label: str,
) -> dict[tuple[int, int, str], Mapping[str, str]]:
    indexed: dict[tuple[int, int, str], Mapping[str, str]] = {}
    for row in rows:
        try:
            key = (
                int(row["prediction_input_index"]),
                int(row["horizon_step"]),
                row["target_timestamp"],
            )
        except KeyError as error:
            raise ValueError(
                f"{label} trace lacks alignment field {error.args[0]}"
            ) from error
        if key in indexed:
            raise ValueError(f"duplicate {label} oracle key: {key}")
        indexed[key] = row
    return indexed


def _field_recall(
    rows: Sequence[Mapping[str, str]],
    prefix: str,
    field: str,
) -> float:
    return _ratio(
        sum(
            _integer(row.get(f"{prefix}_target_{field}_hits", "0"))
            for row in rows
        ),
        sum(
            _integer(row.get(f"target_{field}_column_count", "0"))
            for row in rows
        ),
    )


def _mean(rows: Sequence[Mapping[str, str]], field: str) -> float:
    values = [
        _number(row[field])
        for row in rows
        if row.get(field, "") != ""
    ]
    return sum(values) / len(values) if values else 0.0


def _columns(value: object) -> set[int]:
    if value in {"", None}:
        return set()
    return {int(item) for item in str(value).split()}


def _columns_text(columns: set[int]) -> str:
    return " ".join(str(column) for column in sorted(columns))


def _integer(value: object) -> int:
    return int(float(value)) if value not in {"", None} else 0


def _number(value: object) -> float:
    return float(value) if value not in {"", None} else 0.0


def _boolean(value: object) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _runtime(path: Path | None) -> float | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "runtime_seconds" in payload:
        return float(payload["runtime_seconds"])
    summaries = payload.get("summaries", {})
    original = summaries.get("original", {}) if isinstance(summaries, dict) else {}
    value = original.get("runtime_seconds") if isinstance(original, dict) else None
    return float(value) if value is not None else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline sequential/batched Fig.9 oracle comparison."
    )
    parser.add_argument("--sequential-trace", required=True)
    parser.add_argument("--batched-trace", required=True)
    parser.add_argument("--sequential-summary", default="")
    parser.add_argument("--batched-summary", default="")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = compare_policy_rows(
        read_trace(Path(args.sequential_trace)),
        read_trace(Path(args.batched_trace)),
        sequential_runtime=_runtime(
            Path(args.sequential_summary)
            if args.sequential_summary
            else None
        ),
        batched_runtime=_runtime(
            Path(args.batched_summary) if args.batched_summary else None
        ),
    )
    write_comparison(Path(args.output_dir), rows, summary)
    print(
        Path(args.output_dir) / "competition_policy_comparison.json"
    )


if __name__ == "__main__":
    main()
