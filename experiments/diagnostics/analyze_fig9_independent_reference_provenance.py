"""Offline summaries for the independent Fig.9 reference traces."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
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
    return [{field: key, "count": value} for key, value in sorted(counts.items())]


def analyze(run_dir: Path, output_dir: Path) -> dict[str, object]:
    event_path = next(run_dir.glob("independent_reference_event_trace.csv*"), None)
    segment_path = next(run_dir.glob("independent_reference_segment_trace.csv*"), None)
    events = read_rows(event_path) if event_path else []
    segments = read_rows(segment_path) if segment_path else []
    write_csv(output_dir / "independent_reference_coverage_summary.csv", grouped(events, "reference_status"))
    write_csv(output_dir / "independent_reference_tier_summary.csv", grouped(events, "reference_tiers"))
    write_csv(output_dir / "reference_compatibility_summary.csv", grouped(events, "reference_compatible"))
    write_csv(output_dir / "l2_only_reference_summary.csv", [row for row in events if row.get("observed_column", "")])
    write_csv(output_dir / "passenger_l2_only_reference_summary.csv", [])
    write_csv(output_dir / "ambiguity_reference_outcome_summary.csv", [row for row in events if "AMBIGUOUS" in row.get("reference_status", "")])
    write_csv(output_dir / "repeated_incompatible_reinforcement_summary.csv", [])
    write_csv(output_dir / "reference_by_record_range.csv", events)
    write_csv(output_dir / "reference_by_field.csv", grouped(events, "source_field"))
    write_csv(output_dir / "reference_downstream_error_summary.csv", [])
    write_csv(output_dir / "l2_l4_reference_comparison.csv", [])
    write_csv(output_dir / "reference_bootstrap_ci.csv", [])
    consistency = {
        "event_rows": len(events),
        "segment_rows": len(segments),
        "duplicate_event_keys": len(events) - len({(r.get("actual_record_index"), r.get("event_index")) for r in events}),
        "reference_captured_before_current_matching": all(r.get("reference_captured_before_current_matching") == "True" for r in events),
        "ground_truth_used_for_analysis_only": all(r.get("ground_truth_used_for_analysis_only") == "True" for r in events),
    }
    (output_dir / "reference_consistency_checks.json").write_text(json.dumps(consistency, indent=2), encoding="utf-8")
    summary = {
        "event_rows": len(events),
        "segment_rows": len(segments),
        "status_counts": dict(Counter(row.get("reference_status", "") for row in events)),
        "recommended_next_step": "IMPROVE_REFERENCE_PROVENANCE_TRACE",
        "final_conclusion": "REFERENCE_PROVENANCE_STILL_INSUFFICIENT",
        "diagnostic_only": True,
    }
    (output_dir / "reference_coverage_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = output_dir / "FIG9_INDEPENDENT_REFERENCE_PROVENANCE_REPORT.md"
    report.write_text(
        "# Fig.9 Independent Reference Provenance\n\n"
        f"Events: {len(events)}\n\nSegments: {len(segments)}\n\n"
        "This report is offline-only. It does not change prediction, selection, reinforcement, learning, RNG, or strict defaults.\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_dir or args.run_dir / "analysis_independent_reference"
    print(json.dumps(analyze(args.run_dir, output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
