"""Assemble read-only Fig.9 traces split by a checkpoint resume.

The model checkpoint already carries predictions and density rows, but large
diagnostic streams are written outside the checkpoint.  A native crash can
therefore leave the formal window split across two directories.  This utility
joins the pre-checkpoint rows from the failed run with post-checkpoint rows from
the successful resume without replaying the model.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
from pathlib import Path
from typing import Iterator


def open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def rows(path: Path) -> Iterator[dict[str, str]]:
    with open_text(path, "rt") as handle:
        yield from csv.DictReader(handle)


def trace_path(run_dir: Path, stem: str) -> Path:
    plain = run_dir / stem
    compressed = plain.with_suffix(plain.suffix + ".gz")
    if compressed.exists():
        return compressed
    if plain.exists():
        return plain
    raise FileNotFoundError(f"missing trace {stem} in {run_dir}")


def merge_trace(
    before: Path,
    after: Path,
    output: Path,
    *,
    index_field: str,
    resume_index: int,
    minimum_index: int | None = None,
    maximum_index_exclusive: int | None = None,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    fieldnames: list[str] | None = None
    with open_text(output, "wt") as handle:
        writer: csv.DictWriter | None = None
        for source, keep in (
            (before, lambda value: value < resume_index),
            (after, lambda value: value >= resume_index),
        ):
            for row in rows(source):
                value = int(row[index_field])
                if not keep(value):
                    continue
                if minimum_index is not None and value < minimum_index:
                    continue
                if maximum_index_exclusive is not None and value >= maximum_index_exclusive:
                    continue
                if writer is None:
                    fieldnames = list(row)
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                elif list(row) != fieldnames:
                    raise ValueError(f"trace schema mismatch in {source}")
                writer.writerow(row)
                count += 1
    return count


def scenario_events_from_context(path: Path) -> dict[tuple[int, str, int], str]:
    events: dict[tuple[int, str, int], str] = {}
    for row in rows(path):
        key = (
            int(row["actual_record_index"]),
            row["observed_field"],
            int(row["observed_column"]),
        )
        scenario = row["observe_scenario"]
        previous = events.setdefault(key, scenario)
        if previous != scenario:
            raise ValueError(f"inconsistent scenario for {key}: {previous}/{scenario}")
    return events


def scenario_events_from_observe(path: Path) -> dict[tuple[int, str, int], str]:
    events: dict[tuple[int, str, int], str] = {}
    for row in rows(path):
        key = (
            int(row["actual_record_index"]),
            row["field"],
            int(row["encoded_column"]),
        )
        scenario = row["observe_scenario"]
        previous = events.setdefault(key, scenario)
        if previous != scenario:
            raise ValueError(f"inconsistent observe scenario for {key}")
    return events


def timing_by_event(path: Path) -> dict[tuple[int, str, int], tuple[bool, bool]]:
    result: dict[tuple[int, str, int], tuple[bool, bool]] = {}
    for row in rows(path):
        key = (
            int(row["actual_record_index"]),
            row["field"],
            int(row["encoded_column"]),
        )
        available = row.get("predicted_time_available", "").lower() == "true"
        valid = row.get("predicted_time_valid", "").lower() == "true"
        old_available, old_valid = result.get(key, (False, False))
        result[key] = (old_available or available, old_valid or valid)
    return result


def write_scenario_trace(
    path: Path,
    events: dict[tuple[int, str, int], str],
    timing: dict[tuple[int, str, int], tuple[bool, bool]],
) -> None:
    fields = [
        "actual_record_index",
        "field",
        "encoded_column",
        "observe_scenario",
        "created_segment_id",
        "reinforced_segment_id",
        "predicted_time_available",
        "predicted_time_valid",
        "assembled_from_context_trajectory",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, scenario in sorted(events.items()):
            record_index, field, column = key
            available, valid = timing.get(key, (False, False))
            event_id = f"assembled:{record_index}:{field}:{column}"
            writer.writerow(
                {
                    "actual_record_index": record_index,
                    "field": field,
                    "encoded_column": column,
                    "observe_scenario": scenario,
                    "created_segment_id": event_id if scenario == "scenario3" else "",
                    "reinforced_segment_id": event_id if scenario in {"scenario1", "scenario2"} else "",
                    "predicted_time_available": available,
                    "predicted_time_valid": valid,
                    "assembled_from_context_trajectory": True,
                }
            )


def assemble_runtime(
    before_run: Path,
    output_dir: Path,
    *,
    resume_index: int,
) -> dict[str, float]:
    prefix_runtime = None
    progress_path = before_run / "progress.jsonl"
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if int(row["current_index"]) == resume_index:
            prefix_runtime = float(row["elapsed_seconds"])
    if prefix_runtime is None:
        raise ValueError(f"no progress runtime for checkpoint index {resume_index}")

    summary_path = output_dir / "original_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    resumed_runtime = float(summary["runtime_seconds"])
    cumulative_runtime = prefix_runtime + resumed_runtime
    components = {
        "prefix_through_checkpoint_seconds": prefix_runtime,
        "resumed_suffix_seconds": resumed_runtime,
        "logical_cumulative_seconds": cumulative_runtime,
    }
    summary["runtime_seconds"] = cumulative_runtime
    summary["records_per_second"] = float(summary["records_used"]) / cumulative_runtime
    summary["runtime_assembly"] = components
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    runtime_path = output_dir / "runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["runtime_assembly"] = components
    runtime["summaries"]["original"] = summary
    runtime_path.write_text(
        json.dumps(runtime, indent=2, sort_keys=True), encoding="utf-8"
    )
    return components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-run", type=Path, required=True)
    parser.add_argument("--resumed-run", type=Path, required=True)
    parser.add_argument("--resume-index", type=int, required=True)
    parser.add_argument("--formal-start-index", type=int, default=200)
    parser.add_argument("--formal-end-index", type=int, default=245)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")
    shutil.copytree(args.resumed_run, args.output_dir)
    runtime_components = assemble_runtime(
        args.before_run,
        args.output_dir,
        resume_index=args.resume_index,
    )

    match_output = args.output_dir / "match_overlap_segment_trace.csv.gz"
    match_count = merge_trace(
        trace_path(args.before_run, "match_overlap_segment_trace.csv"),
        trace_path(args.resumed_run, "match_overlap_segment_trace.csv"),
        match_output,
        index_field="actual_record_index",
        resume_index=args.resume_index,
        minimum_index=args.formal_start_index,
        maximum_index_exclusive=args.formal_end_index,
    )
    context_output = args.output_dir / "context_trajectory_column_trace.csv.gz"
    context_count = merge_trace(
        trace_path(args.before_run, "context_trajectory_column_trace.csv"),
        trace_path(args.resumed_run, "context_trajectory_column_trace.csv"),
        context_output,
        index_field="actual_record_index",
        resume_index=args.resume_index,
        minimum_index=args.formal_start_index,
        maximum_index_exclusive=args.formal_end_index,
    )

    context_events = scenario_events_from_context(context_output)
    resumed_events = scenario_events_from_observe(
        args.resumed_run / "observe_scenario_trace.csv"
    )
    events = {
        key: scenario
        for key, scenario in context_events.items()
        if args.formal_start_index <= key[0] < args.resume_index
    }
    events.update(
        {
            key: scenario
            for key, scenario in resumed_events.items()
            if args.resume_index <= key[0] < args.formal_end_index
        }
    )
    expected_events = (args.formal_end_index - args.formal_start_index) * 30
    if len(events) != expected_events:
        raise ValueError(
            f"expected {expected_events} formal scenario events, found {len(events)}"
        )
    timing = timing_by_event(match_output)
    write_scenario_trace(args.output_dir / "observe_scenario_trace.csv", events, timing)

    manifest = {
        "before_run": str(args.before_run.resolve()),
        "resumed_run": str(args.resumed_run.resolve()),
        "resume_index": args.resume_index,
        "formal_record_range": [args.formal_start_index, args.formal_end_index - 1],
        "scenario_event_count": len(events),
        "match_segment_row_count": match_count,
        "context_trajectory_row_count": context_count,
        "model_was_replayed": False,
        "runtime_components": runtime_components,
        "scenario_identity_note": (
            "created/reinforced IDs are deterministic event IDs used only for "
            "count aggregation; scenario labels come from the original read-only trace"
        ),
    }
    (args.output_dir / "resume_trace_assembly.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
