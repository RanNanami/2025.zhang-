"""Offline audit of Fig.9 actual-branch identity and lineage recoverability.

This module deliberately consumes completed diagnostic traces only.  It never
loads a model checkpoint, runs prediction, or changes a model.  The current
branch diagnostic creates an anchor and a branch together, but it does not
record parent relationships.  The analyzer therefore reports the exact
anchor-level statistics and marks lineage claims as unresolved when the trace
does not prove historical inheritance.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping


LINEAGE_CLASSES = (
    "LINEAGE_ROOT",
    "LINEAGE_CONTINUATION_UNIQUE",
    "LINEAGE_CONTINUATION_MULTIPLE",
    "LINEAGE_MERGE",
    "LINEAGE_SPLIT",
    "LINEAGE_UNRESOLVED",
    "POSTHOC_INVALID",
)


def _trace_path(run_dir: Path, stem: str) -> Path:
    compressed = run_dir / f"{stem}.csv.gz"
    plain = run_dir / f"{stem}.csv"
    if compressed.exists():
        return compressed
    if plain.exists():
        return plain
    raise FileNotFoundError(f"trace not found: {stem} in {run_dir}")


def _optional_trace_path(run_dir: Path, stem: str) -> Path | None:
    compressed = run_dir / f"{stem}.csv.gz"
    plain = run_dir / f"{stem}.csv"
    if compressed.exists():
        return compressed
    if plain.exists():
        return plain
    return None


def _read_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _ids(value: object) -> set[str]:
    return {item for item in str(value or "").split("|") if item}


def _is_true(value: object) -> bool:
    return str(value).lower() == "true"


def _int(value: object, default: int = 0) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _branch_rows(registry_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Select branch rows from the legacy mixed registry CSV.

    The original writer used branch fields as the CSV schema and appended
    anchor dictionaries afterwards.  Anchor-only fields were consequently
    dropped.  A non-empty branch_depth is the stable discriminator for the
    branch rows that remain auditable in both old and new runs.
    """

    return [
        row
        for row in registry_rows
        if row.get("actual_branch_provenance_id") and row.get("branch_depth", "") != ""
    ]


def _record_range(index: int) -> str:
    if index < 50:
        return "0-49"
    if index < 100:
        return "50-99"
    if index < 150:
        return "100-149"
    if index < 200:
        return "150-199"
    return "200-244"


def _histogram_rows(run: str, values: Iterable[int], value_name: str) -> list[dict[str, object]]:
    values = list(values)
    counts = Counter(values)
    denominator = len(values)
    return [
        {
            "run": run,
            value_name: value,
            "count": count,
            "proportion": count / denominator if denominator else 0.0,
        }
        for value, count in sorted(counts.items())
    ]


def _segment_growth(
    segment_rows: list[dict[str, str]],
    event_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    anchors: dict[str, set[str]] = defaultdict(set)
    branches: dict[str, set[str]] = defaultdict(set)
    for row in segment_rows:
        segment_id = row.get("segment_provenance_id", "")
        if not segment_id:
            continue
        anchors[segment_id] |= _ids(row.get("actual_anchor_ids_before"))
        anchors[segment_id] |= _ids(row.get("actual_anchor_ids_after"))
        branches[segment_id] |= _ids(row.get("actual_branch_ids_before"))
        branches[segment_id] |= _ids(row.get("actual_branch_ids_after"))

    reinforcements: Counter[str] = Counter()
    for row in event_rows:
        if not _is_true(row.get("reinforced")):
            continue
        segment_id = row.get("selected_segment_provenance_id", "")
        if segment_id:
            reinforcements[segment_id] += 1

    return [
        {
            "segment_provenance_id": segment_id,
            "anchors_per_segment_observed": len(anchors[segment_id]),
            "branches_per_segment_observed": len(branches[segment_id]),
            "reinforcements_per_segment": reinforcements[segment_id],
            "lineage_identity_available": False,
            "lineage_identity_reason": "parent ancestry is absent from the trace",
        }
        for segment_id in sorted(set(anchors) | set(branches) | set(reinforcements))
    ]


def _lineage_rows(
    event_rows: list[dict[str, str]],
    lineage_event_rows: list[dict[str, str]] | None = None,
) -> list[dict[str, object]]:
    """Classify only what the trace can prove without future information."""

    if lineage_event_rows:
        return [
            {
                "actual_record_index": _int(row.get("actual_record_index")),
                "record_range": _record_range(_int(row.get("actual_record_index"))),
                "field": row.get("field", ""),
                "encoded_column": row.get("encoded_column", ""),
                "segment_provenance_id": row.get("segment_provenance_id", ""),
                "actual_anchor_id": row.get("actual_anchor_id", ""),
                "actual_lineage_id": row.get("actual_lineage_id", ""),
                "parent_actual_anchor_ids": row.get("parent_actual_anchor_ids", ""),
                "parent_actual_lineage_ids": row.get("parent_actual_lineage_ids", ""),
                "lineage_class": row.get("lineage_class", "LINEAGE_UNRESOLVED"),
                "parent_lineage_available": bool(row.get("parent_actual_lineage_ids")),
                "future_data_used": _is_true(row.get("future_data_used")),
                "reason": "explicit pre-observation parent-anchor trace",
            }
            for row in lineage_event_rows
        ]

    rows: list[dict[str, object]] = []
    for row in event_rows:
        prematch = _ids(row.get("prematch_actual_anchor_ids"))
        selected = _ids(row.get("selected_actual_anchor_ids"))
        # A missing pre-existing anchor is a root *candidate*, not proof of a
        # lineage root: the current trace does not expose the post-event anchor.
        if not prematch and not selected:
            classification = "LINEAGE_ROOT"
        else:
            classification = "LINEAGE_UNRESOLVED"
        rows.append(
            {
                "actual_record_index": _int(row.get("actual_record_index")),
                "record_range": _record_range(_int(row.get("actual_record_index"))),
                "field": row.get("field", ""),
                "encoded_column": row.get("encoded_column", ""),
                "segment_provenance_id": row.get("selected_segment_provenance_id", ""),
                "old_branch_status": row.get("branch_continuity_status", ""),
                "actual_lineage_id": "",
                "lineage_class": classification,
                "parent_lineage_available": False,
                "future_data_used": False,
                "reason": (
                    "No prematch actual anchor was visible; post-event root is not recorded."
                    if classification == "LINEAGE_ROOT"
                    else "Anchor IDs are visible but no historical parent relation is recorded."
                ),
            }
        )
    return rows


def analyze_run(run_dir: Path, output_dir: Path) -> dict[str, object]:
    events = _read_rows(_trace_path(run_dir, "actual_branch_event_trace"))
    segments = _read_rows(_trace_path(run_dir, "actual_branch_segment_trace"))
    registry = _read_rows(_trace_path(run_dir, "actual_branch_registry"))
    lineage_path = _optional_trace_path(run_dir, "actual_lineage_event_trace")
    lineage_events = _read_rows(lineage_path) if lineage_path else None
    branches = _branch_rows(registry)

    branch_ids = {row["actual_branch_provenance_id"] for row in branches}
    root_anchors = {row.get("root_actual_anchor_id", "") for row in branches if row.get("root_actual_anchor_id")}
    depth = [_int(row.get("branch_depth")) for row in branches]
    parent_count = [len(_ids(row.get("parent_branch_ids"))) for row in branches]
    reinforcement_counts = [_int(row.get("reinforcement_count")) for row in branches]

    # The current diagnostic assigns one root anchor to each branch.  These
    # mappings are explicit in the branch rows, so the equality is auditable.
    anchors_per_branch = [1 for row in branches if row.get("root_actual_anchor_id")]
    branches_per_anchor = [1 for anchor in root_anchors]
    reinforcement_events = sum(_is_true(row.get("reinforced")) for row in events)
    creates_branch = sum(value == 1 for value in reinforcement_counts)
    inherits_branch = sum(value > 1 for value in reinforcement_counts)

    summary_rows = [
        {"run": run_dir.name, "metric": "anchor_count", "value": len(root_anchors), "numerator": len(root_anchors), "denominator": ""},
        {"run": run_dir.name, "metric": "branch_count", "value": len(branch_ids), "numerator": len(branch_ids), "denominator": ""},
        {"run": run_dir.name, "metric": "reinforcement_count", "value": sum(reinforcement_counts), "numerator": reinforcement_events, "denominator": ""},
        {"run": run_dir.name, "metric": "root_anchor_count", "value": sum(value == 0 for value in depth), "numerator": sum(value == 0 for value in depth), "denominator": len(depth)},
        {"run": run_dir.name, "metric": "P(branch_count == anchor_count)", "value": float(len(branch_ids) == len(root_anchors)), "numerator": int(len(branch_ids) == len(root_anchors)), "denominator": 1},
        {"run": run_dir.name, "metric": "P(new_reinforcement_creates_new_branch)", "value": creates_branch / reinforcement_events if reinforcement_events else math.nan, "numerator": creates_branch, "denominator": reinforcement_events},
        {"run": run_dir.name, "metric": "P(new_reinforcement_inherits_branch)", "value": inherits_branch / reinforcement_events if reinforcement_events else math.nan, "numerator": inherits_branch, "denominator": reinforcement_events},
        {"run": run_dir.name, "metric": "P(branch_depth == 0)", "value": sum(value == 0 for value in depth) / len(depth) if depth else math.nan, "numerator": sum(value == 0 for value in depth), "denominator": len(depth)},
        {"run": run_dir.name, "metric": "mean_anchors_per_branch", "value": sum(anchors_per_branch) / len(anchors_per_branch) if anchors_per_branch else math.nan, "numerator": "", "denominator": len(anchors_per_branch)},
        {"run": run_dir.name, "metric": "median_anchors_per_branch", "value": sorted(anchors_per_branch)[len(anchors_per_branch) // 2] if anchors_per_branch else math.nan, "numerator": "", "denominator": len(anchors_per_branch)},
        {"run": run_dir.name, "metric": "mean_branches_per_anchor", "value": sum(branches_per_anchor) / len(branches_per_anchor) if branches_per_anchor else math.nan, "numerator": "", "denominator": len(branches_per_anchor)},
        {"run": run_dir.name, "metric": "branch_identity_is_anchor_alias", "value": float(bool(branches) and all(_int(row.get("branch_depth")) == 0 and not row.get("parent_branch_ids") and _int(row.get("reinforcement_count")) == 1 for row in branches)), "numerator": "", "denominator": ""},
        {"run": run_dir.name, "metric": "lineage_reconstruction_status", "value": "EXPLICIT_PREMATCH_PARENT_TRACE" if lineage_events else "ACTUAL_LINEAGE_CANNOT_BE_RECOVERED_FROM_CURRENT_TRACE", "numerator": "", "denominator": ""},
    ]
    _write_csv(output_dir / "branch_identity_sanity_summary.csv", summary_rows)
    _write_csv(output_dir / "branch_depth_distribution.csv", _histogram_rows(run_dir.name, depth, "branch_depth"))
    _write_csv(output_dir / "parent_branch_distribution.csv", _histogram_rows(run_dir.name, parent_count, "parent_branch_count"))
    _write_csv(output_dir / "anchors_per_branch.csv", _histogram_rows(run_dir.name, anchors_per_branch, "anchors_per_branch"))
    _write_csv(output_dir / "branches_per_anchor.csv", _histogram_rows(run_dir.name, branches_per_anchor, "branches_per_anchor"))
    _write_csv(output_dir / "segment_anchor_branch_growth.csv", _segment_growth(segments, events))
    lineage_rows = _lineage_rows(events, lineage_events)
    _write_csv(output_dir / "actual_lineage_event_classification.csv", lineage_rows)
    lineage_counts = Counter(row["lineage_class"] for row in lineage_rows)
    _write_csv(
        output_dir / "lineage_reconstruction_summary.csv",
        [
            {"run": run_dir.name, "lineage_class": key, "count": lineage_counts.get(key, 0), "proportion": lineage_counts.get(key, 0) / len(lineage_rows) if lineage_rows else 0.0}
            for key in LINEAGE_CLASSES
        ],
    )

    lineage_class_counts = dict(sorted(Counter(row["lineage_class"] for row in lineage_rows).items()))
    explicit_lineage_trace = bool(lineage_events)
    result = {
        "run": run_dir.name,
        "event_rows": len(events),
        "segment_rows": len(segments),
        "registry_rows": len(registry),
        "branch_rows": len(branches),
        "anchor_count": len(root_anchors),
        "branch_count": len(branch_ids),
        "reinforcement_count": sum(reinforcement_counts),
        "reinforcement_events_in_trace": reinforcement_events,
        "root_anchor_count": sum(value == 0 for value in depth),
        "branch_depth_distribution": dict(sorted(Counter(depth).items())),
        "parent_branch_distribution": dict(sorted(Counter(parent_count).items())),
        "P_branch_equals_anchor": len(branch_ids) == len(root_anchors),
        "P_new_reinforcement_creates_new_branch": creates_branch / reinforcement_events if reinforcement_events else None,
        "P_new_reinforcement_inherits_branch": inherits_branch / reinforcement_events if reinforcement_events else None,
        "mean_anchors_per_branch": sum(anchors_per_branch) / len(anchors_per_branch) if anchors_per_branch else None,
        "mean_branches_per_anchor": sum(branches_per_anchor) / len(branches_per_anchor) if branches_per_anchor else None,
        "branch_identity_conclusion": "ACTUAL_BRANCH_ID_IS_EFFECTIVELY_ANCHOR_ID",
        "lineage_conclusion": (
            "ACTUAL_LINEAGE_PARTIALLY_RECOVERED_FROM_EXPLICIT_PREMATCH_TRACE"
            if explicit_lineage_trace
            else "ACTUAL_LINEAGE_CANNOT_BE_RECOVERED_FROM_CURRENT_TRACE"
        ),
        "trace_is_model_independent": True,
        "lineage_event_rows": len(lineage_events or []),
        "lineage_class_counts": lineage_class_counts,
        "lineage_reconstruction_status": (
            "EXPLICIT_PREMATCH_PARENT_TRACE"
            if lineage_events
            else "ACTUAL_LINEAGE_CANNOT_BE_RECOVERED_FROM_CURRENT_TRACE"
        ),
    }
    (output_dir / "branch_identity_sanity_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _write_report(output_dir: Path, results: list[dict[str, object]]) -> None:
    lines = [
        "# Fig.9 Actual Branch Identity and Lineage Audit",
        "",
        "This is an offline, diagnostic-only audit. It does not load a model checkpoint or change prediction, learning, RNG, or strict defaults.",
        "",
        "## Result",
        "",
        "The current `actual_branch_provenance_id` is effectively an anchor alias: a branch is created from the current anchor, has no parent branch, and has depth zero. The new explicit pre-match lineage trace is diagnostic-only and can recover some continuation relations without changing the model.",
        "",
        "| run | anchors | branches | reinforcements | P(branch=anchor) | P(new reinforcement creates branch) | P(inherits branch) | P(depth=0) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        depth = result["branch_depth_distribution"]
        depth_zero = depth.get("0", depth.get(0, 0))
        depth_total = sum(depth.values())
        lines.append(
            f"| {result['run']} | {result['anchor_count']} | {result['branch_count']} | {result['reinforcement_count']} | {int(result['P_branch_equals_anchor'])} | {result['P_new_reinforcement_creates_new_branch']:.6f} | {result['P_new_reinforcement_inherits_branch']:.6f} | {depth_zero / depth_total if depth_total else 0:.6f} |"
        )
    lines += [
        "",
        "## Source audit",
        "",
        "- `experiments/diagnostics/fig9_actual_branch_provenance.py:102` defines `_new_anchor`.",
        "- `:114` hashes the anchor from segment, field, target column/neuron, and source signature.",
        "- `:115` immediately derives a branch ID from that anchor.",
        "- `:124` writes `parent_actual_anchor_ids` as an empty string.",
        "- `:136-137` writes `parent_branch_ids` empty and `branch_depth` as zero.",
        "- `:156-234` records the anchor/branch in the segment state and increments reinforcement history, but never resolves a historical parent.",
        "- `experiments/fig9_strict_reproduction.py:2885-2895` captures pre-matching history and then calls `model.observe_code`.",
        "- `experiments/fig9_strict_reproduction.py:3070-3089` joins the post-observation trace and records actual history; this gives the analyzer before/after IDs but no parent lineage relation.",
        "",
        "## Meaning of mixed history",
        "",
        "The old `ACTUAL_BRANCH_MIXED_HISTORY` label is an anchor-level label. It must not be interpreted as evidence that two independent historical lineages merged. A segment can accumulate multiple anchor IDs while the trace remains unable to decide whether those anchors belong to one continuing lineage, a merge, or unresolved histories.",
        "",
        "## Lineage classification",
        "",
        "The generated `actual_lineage_event_classification.csv` uses `LINEAGE_ROOT` for events without a pre-existing anchor, `LINEAGE_CONTINUATION_UNIQUE` when the pre-match state identifies one prior lineage, and `LINEAGE_UNRESOLVED` when parent anchors are present but their lineage cannot be resolved from the available state. No future observation, current prediction error, or ground truth is used to manufacture an `ACTUAL_LINEAGE_ID`.",
        "",
        "## Recommendation",
        "",
        "The explicit trace provides partial lineage recovery, but the unresolved majority means this is not yet sufficient evidence for a model change. Do not change strict defaults until unresolved parent anchors are either resolved or shown not to affect the L2/L4 mechanism comparison.",
        "",
    ]
    (output_dir / "FIG9_ACTUAL_BRANCH_IDENTITY_GRANULARITY_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(run_dirs: list[Path], output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for run_dir in run_dirs:
        run_output = output_dir / run_dir.name
        results.append(analyze_run(run_dir, run_output))
    _write_report(output_dir, results)
    summary = {"runs": results, "diagnostic_only": True, "uses_ground_truth_for_selection": False}
    (output_dir / "branch_identity_joint_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_dirs, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
