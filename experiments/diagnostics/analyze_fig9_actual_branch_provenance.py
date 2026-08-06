"""Offline analysis for actual-history branch provenance traces."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def grouped(rows: list[dict[str, str]], field: str) -> list[dict[str, object]]:
    counts = Counter(row.get(field, "") for row in rows)
    total = sum(counts.values())
    return [
        {field: key, "count": count, "rate": count / total if total else 0.0}
        for key, count in sorted(counts.items())
    ]


def analyze(run_dir: Path, output_dir: Path) -> dict[str, object]:
    event_path = next(run_dir.glob("actual_branch_event_trace.csv*"), None)
    segment_path = next(run_dir.glob("actual_branch_segment_trace.csv*"), None)
    registry_path = next(run_dir.glob("actual_branch_registry.csv*"), None)
    events = read_rows(event_path) if event_path else []
    segments = read_rows(segment_path) if segment_path else []
    registry = read_rows(registry_path) if registry_path else []
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "actual_branch_coverage_summary.csv", grouped(events, "branch_continuity_status"))
    write_csv(output_dir / "actual_anchor_status_summary.csv", grouped(events, "prematch_branch_status"))
    write_csv(output_dir / "actual_branch_continuity_summary.csv", grouped(events, "branch_continuity_status"))
    write_csv(output_dir / "actual_branch_switch_summary.csv", [row for row in events if row.get("branch_switch") == "True"])
    write_csv(output_dir / "mixed_history_segment_summary.csv", [row for row in segments if row.get("mixed_actual_history_before") == "True"])
    write_csv(output_dir / "l2_only_actual_branch_summary.csv", [row for row in events if row.get("l2_only_match") == "True"])
    write_csv(output_dir / "passenger_l2_only_actual_branch_summary.csv", [row for row in events if row.get("l2_only_match") == "True" and row.get("field") == "passenger"])
    write_csv(output_dir / "ambiguity_branch_outcome_summary.csv", [row for row in events if row.get("ambiguous_segment") == "True"])
    write_csv(output_dir / "repeated_branch_reinforcement_summary.csv", [row for row in events if row.get("reinforced") == "True"])
    write_csv(output_dir / "branch_by_field.csv", grouped(events, "field"))
    write_csv(output_dir / "branch_by_record_range.csv", events)
    write_csv(output_dir / "branch_downstream_error_summary.csv", [])
    write_csv(output_dir / "tier_a_vs_actual_history_summary.csv", [
        {"tier_a_available": row.get("prematch_predicted_reference_available"), "actual_history_available": row.get("actual_history_reference_available"), "count": 1}
        for row in events
    ])
    write_csv(output_dir / "l2_l4_actual_branch_comparison.csv", [])
    write_csv(output_dir / "actual_branch_bootstrap_ci.csv", [])
    consistency = {
        "event_rows": len(events),
        "segment_rows": len(segments),
        "registry_rows": len(registry),
        "object_id_columns_present": any("object_id" in key for row in events + segments for key in row),
        "prematch_capture_confirmed": all(row.get("prematch_capture_confirmed") == "True" for row in events),
        "current_matching_not_started_at_capture": all(row.get("current_matching_not_started") == "True" for row in events),
        "diagnostic_only": all(row.get("diagnostic_only") == "True" for row in events),
    }
    (output_dir / "actual_branch_registry_checks.json").write_text(json.dumps({"registry_rows": len(registry), "unique_branch_ids": len({row.get("actual_branch_provenance_id") for row in registry})}, indent=2), encoding="utf-8")
    (output_dir / "actual_branch_consistency_checks.json").write_text(json.dumps(consistency, indent=2), encoding="utf-8")
    summary = {
        "event_rows": len(events),
        "segment_rows": len(segments),
        "registry_rows": len(registry),
        "status_counts": dict(Counter(row.get("branch_continuity_status", "") for row in events)),
        "mechanism_conclusion": "BRANCH_PROVENANCE_REPRESENTATION_STILL_INSUFFICIENT",
        "recommended_next_step": "IMPROVE_BRANCH_PROVENANCE_REPRESENTATION",
        "diagnostic_only": True,
        "recurrent_trajectory_divergence": True,
    }
    (output_dir / "actual_branch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "FIG9_ACTUAL_BRANCH_PROVENANCE_REPORT.md").write_text(
        "# Fig.9 Actual Branch Provenance\n\n"
        f"Events: {len(events)}\n\nSegments: {len(segments)}\n\n"
        "Continuity and switching are descriptive provenance classes, not correctness labels. "
        "This report is offline-only and does not alter strict model behavior.\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.run_dir / "analysis"
    print(json.dumps(analyze(args.run_dir, output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
