"""Analyze read-only Fig.9 context trajectory traces."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


STAGES = (
    "segment_inspected",
    "overlap_positive",
    "response_computed",
    "threshold_crossed",
    "firing_time_valid",
    "saved_candidate",
    "intracolumn_selected",
    "intercolumn_survived",
    "emitted_winner",
    "entered_previous_winners",
    "present_in_next_context",
)


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _latest_loss(row: dict[str, str]) -> str:
    checks = (
        ("segment_inspected", "NO_INSPECTED_SEGMENT"),
        ("overlap_positive", "INSPECTED_ZERO_OVERLAP"),
        ("response_computed", "RESPONSE_BELOW_THRESHOLD"),
        ("threshold_crossed", "RESPONSE_BELOW_THRESHOLD"),
        ("firing_time_valid", "CROSSING_INVALID_TIME"),
        ("saved_candidate", "CANDIDATE_NOT_SAVED"),
        ("intracolumn_selected", "LOST_INTRACOLUMN"),
        ("intercolumn_survived", "LOST_INTERCOLUMN"),
        ("emitted_winner", "LOST_INTERCOLUMN"),
        ("entered_previous_winners", "EMITTED_NOT_PROPAGATED"),
        ("present_in_next_context", "EMITTED_NOT_PROPAGATED"),
    )
    for field, reason in checks:
        if not _bool(row.get(field, "")):
            return reason
    return "NONE"


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(materialized[0]) if materialized else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _rate_rows(
    rows: list[dict[str, str]],
    *,
    key: str,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    kinds = sorted({row["trajectory_kind"] for row in rows})
    for kind in kinds:
        scoped = [row for row in rows if row["trajectory_kind"] == kind]
        counts = Counter(row.get(key, "") or "NONE" for row in scoped)
        for value, count in sorted(counts.items()):
            output.append(
                {
                    "trajectory_kind": kind,
                    key: value,
                    "numerator": count,
                    "denominator": len(scoped),
                    "rate": count / len(scoped) if scoped else 0.0,
                    "denominator_scope": (
                        "source-column dependencies with "
                        "SOURCE_COLUMN_NOT_ACTIVE"
                    ),
                }
            )
    return output


def _bootstrap_reason_rows(
    rows: list[dict[str, str]],
    *,
    samples: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    output: list[dict[str, object]] = []
    for kind in sorted({row["trajectory_kind"] for row in rows}):
        scoped = [row for row in rows if row["trajectory_kind"] == kind]
        grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
        for row in scoped:
            grouped[
                (
                    row["actual_record_index"],
                    row["observed_column"],
                    row["segment_id"],
                )
            ].append(row)
        clusters = list(grouped.values())
        reasons = sorted(
            {
                row["trajectory_loss_primary_reason"]
                for row in scoped
            }
        )
        for reason in reasons:
            estimates: list[float] = []
            for _ in range(samples):
                sampled = [
                    rng.choice(clusters)
                    for _index in range(len(clusters))
                ] if clusters else []
                flat = [row for cluster in sampled for row in cluster]
                estimates.append(
                    sum(
                        row["trajectory_loss_primary_reason"] == reason
                        for row in flat
                    )
                    / len(flat)
                    if flat
                    else 0.0
                )
            estimates.sort()
            low_index = int(0.025 * max(0, len(estimates) - 1))
            high_index = int(0.975 * max(0, len(estimates) - 1))
            observed = sum(
                row["trajectory_loss_primary_reason"] == reason
                for row in scoped
            )
            output.append(
                {
                    "trajectory_kind": kind,
                    "trajectory_loss_primary_reason": reason,
                    "numerator": observed,
                    "denominator": len(scoped),
                    "rate": observed / len(scoped) if scoped else 0.0,
                    "bootstrap_low_95": (
                        estimates[low_index] if estimates else 0.0
                    ),
                    "bootstrap_high_95": (
                        estimates[high_index] if estimates else 0.0
                    ),
                    "bootstrap_unit": (
                        "actual_record-observed_column-segment cluster"
                    ),
                }
            )
    return output


def _dominant(rows: list[dict[str, str]], field: str) -> str:
    counts = Counter(row.get(field, "") or "NONE" for row in rows)
    return counts.most_common(1)[0][0] if counts else "NONE"


def _recommend(rows: list[dict[str, str]]) -> str:
    autonomous = [
        row
        for row in rows
        if row["trajectory_kind"] == "autonomous_rollout"
    ]
    actual = [
        row
        for row in rows
        if row["trajectory_kind"] == "actual_observation"
    ]
    autonomous_present_rate = (
        sum(_bool(row.get("present_in_next_context", "")) for row in autonomous)
        / len(autonomous)
        if autonomous
        else 0.0
    )
    actual_emitted_not_propagated_rate = (
        sum(
            row.get("latest_loss_stage") == "EMITTED_NOT_PROPAGATED"
            for row in actual
        )
        / len(actual)
        if actual
        else 0.0
    )
    if (
        autonomous_present_rate >= 0.5
        and actual_emitted_not_propagated_rate >= 0.5
    ):
        return "SEGMENT_CONTEXT_REPRESENTATION_FIX_REQUIRED"

    scope = autonomous or rows
    dominant = _dominant(scope, "latest_loss_stage")
    mapping = {
        "NO_INSPECTED_SEGMENT": "SEGMENT_CONTEXT_REPRESENTATION_FIX_REQUIRED",
        "INSPECTED_ZERO_OVERLAP": "SEGMENT_CONTEXT_REPRESENTATION_FIX_REQUIRED",
        "RESPONSE_BELOW_THRESHOLD": "CANDIDATE_GENERATION_DIAGNOSTIC_REQUIRED",
        "CROSSING_INVALID_TIME": "CONTINUOUS_TIMING_DIAGNOSTIC_REQUIRED",
        "CANDIDATE_NOT_SAVED": "CANDIDATE_GENERATION_DIAGNOSTIC_REQUIRED",
        "LOST_INTRACOLUMN": "INTRACOLUMN_RETENTION_FIX_REQUIRED",
        "LOST_INTERCOLUMN": "INTERCOLUMN_RETENTION_FIX_REQUIRED",
        "EMITTED_NOT_PROPAGATED": "STATE_PROPAGATION_FIX_REQUIRED",
        "LOST_BEFORE_CURRENT_TRANSITION": "AUTONOMOUS_CONTEXT_DRIFT_FIX_REQUIRED",
        "NEVER_REACTIVATED_AFTER_CREATION": (
            "SEGMENT_CONTEXT_REPRESENTATION_FIX_REQUIRED"
        ),
        "NONE": "REAL_LMATCH2_ABLATION",
    }
    return mapping.get(
        dominant,
        "CANDIDATE_GENERATION_DIAGNOSTIC_REQUIRED",
    )


def analyze(
    *,
    run_dir: Path,
    output_dir: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    trace_path = run_dir / "context_trajectory_column_trace.csv.gz"
    if not trace_path.exists():
        raise FileNotFoundError(trace_path)
    rows = _read_rows(trace_path)
    for row in rows:
        if not row.get("latest_loss_stage"):
            row["latest_loss_stage"] = _latest_loss(row)
    output_dir.mkdir(parents=True, exist_ok=True)

    funnel: list[dict[str, object]] = []
    for kind in sorted({row["trajectory_kind"] for row in rows}):
        scoped = [row for row in rows if row["trajectory_kind"] == kind]
        for stage in STAGES:
            count = sum(_bool(row.get(stage, "")) for row in scoped)
            funnel.append(
                {
                    "trajectory_kind": kind,
                    "stage": stage,
                    "numerator": count,
                    "denominator": len(scoped),
                    "rate": count / len(scoped) if scoped else 0.0,
                }
            )
    _write_csv(output_dir / "context_trajectory_stage_funnel.csv", funnel)
    _write_csv(
        output_dir / "context_trajectory_loss_reason_summary.csv",
        _rate_rows(rows, key="trajectory_loss_primary_reason"),
    )
    _write_csv(
        output_dir / "context_trajectory_first_loss_summary.csv",
        _rate_rows(rows, key="first_loss_stage"),
    )
    _write_csv(
        output_dir / "context_trajectory_recent_loss_summary.csv",
        _rate_rows(rows, key="most_recent_loss_stage"),
    )
    _write_csv(
        output_dir / "context_trajectory_latest_loss_summary.csv",
        _rate_rows(rows, key="latest_loss_stage"),
    )
    _write_csv(
        output_dir / "context_trajectory_pattern_summary.csv",
        _rate_rows(rows, key="trajectory_pattern"),
    )
    _write_csv(
        output_dir / "context_trajectory_source_field_summary.csv",
        _rate_rows(rows, key="source_field"),
    )
    _write_csv(
        output_dir / "context_trajectory_horizon_summary.csv",
        _rate_rows(rows, key="most_recent_loss_horizon_step"),
    )

    recovery_rows: list[dict[str, object]] = []
    for kind in sorted({row["trajectory_kind"] for row in rows}):
        scoped = [row for row in rows if row["trajectory_kind"] == kind]
        values = [int(row.get("recovery_count", "0") or 0) for row in scoped]
        recovery_rows.append(
            {
                "trajectory_kind": kind,
                "dependency_count": len(scoped),
                "dependencies_with_recovery": sum(value > 0 for value in values),
                "recovery_rate": (
                    sum(value > 0 for value in values) / len(values)
                    if values
                    else 0.0
                ),
                "mean_recovery_count": (
                    sum(values) / len(values) if values else 0.0
                ),
                "maximum_recovery_count": max(values, default=0),
            }
        )
    _write_csv(
        output_dir / "context_trajectory_recovery_summary.csv",
        recovery_rows,
    )
    bootstrap = _bootstrap_reason_rows(
        rows,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    _write_csv(
        output_dir / "context_trajectory_bootstrap_ci.csv",
        bootstrap,
    )

    dependency_keys = {
        (
            row["actual_record_index"],
            row["observed_column"],
            row["segment_id"],
            row["source_column"],
            row.get("source_neuron", ""),
        )
        for row in rows
    }
    kinds_by_dependency: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for row in rows:
        key = (
            row["actual_record_index"],
            row["observed_column"],
            row["segment_id"],
            row["source_column"],
            row.get("source_neuron", ""),
        )
        kinds_by_dependency[key].add(row["trajectory_kind"])
    paired = sum(
        kinds == {"actual_observation", "autonomous_rollout"}
        for kinds in kinds_by_dependency.values()
    )
    manifest = {
        "trace_rows": len(rows),
        "unique_dependency_count": len(dependency_keys),
        "unique_observation_count": len(
            {
                (row["actual_record_index"], row["observed_column"])
                for row in rows
            }
        ),
        "unique_segment_count": len(
            {
                (
                    row["actual_record_index"],
                    row["observed_column"],
                    row["segment_id"],
                )
                for row in rows
            }
        ),
        "dependencies_with_both_trajectory_kinds": paired,
        "trajectory_pair_coverage": (
            paired / len(dependency_keys) if dependency_keys else 0.0
        ),
        "history_available_rate": (
            sum(_bool(row.get("history_available", "")) for row in rows)
            / len(rows)
            if rows
            else 0.0
        ),
        "unknown_reason_count": sum(
            row["trajectory_loss_primary_reason"] == "UNKNOWN"
            for row in rows
        ),
        "unknown_reason_rate": (
            sum(
                row["trajectory_loss_primary_reason"] == "UNKNOWN"
                for row in rows
            )
            / len(rows)
            if rows
            else 0.0
        ),
    }
    _write_json(
        output_dir / "context_trajectory_join_coverage.json",
        manifest,
    )

    by_kind: dict[str, dict[str, str]] = {}
    for kind in ("actual_observation", "autonomous_rollout"):
        scoped = [row for row in rows if row["trajectory_kind"] == kind]
        by_kind[kind] = {
            "dominant_first_loss_stage": _dominant(
                scoped, "first_loss_stage"
            ),
            "dominant_recent_loss_stage": _dominant(
                scoped, "most_recent_loss_stage"
            ),
            "dominant_latest_loss_stage": _dominant(
                scoped, "latest_loss_stage"
            ),
            "dominant_trajectory_pattern": _dominant(
                scoped, "trajectory_pattern"
            ),
            "dominant_primary_reason": _dominant(
                scoped, "trajectory_loss_primary_reason"
            ),
        }
    recommended = _recommend(rows)
    summary = {
        **manifest,
        "by_trajectory_kind": by_kind,
        "recommended_next_step": recommended,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "diagnostic_only": True,
        "model_trajectory_changed": False,
    }
    _write_json(output_dir / "context_trajectory_summary.json", summary)

    report = [
        "# Fig.9 Context Trajectory Decomposition",
        "",
        "This diagnostic is read-only and keeps actual-observation and "
        "autonomous-rollout trajectories separate.",
        "",
        f"- Trace rows: {len(rows)}",
        f"- Unique dependencies: {len(dependency_keys)}",
        f"- Unique observations: {manifest['unique_observation_count']}",
        f"- Unique segments: {manifest['unique_segment_count']}",
        f"- Trajectory pair coverage: {manifest['trajectory_pair_coverage']:.6f}",
        f"- Unknown rate: {manifest['unknown_reason_rate']:.6f}",
        "",
    ]
    for kind, values in by_kind.items():
        report.extend(
            [
                f"## {kind}",
                "",
                f"- Dominant first loss: `{values['dominant_first_loss_stage']}`",
                f"- Dominant recent loss: `{values['dominant_recent_loss_stage']}`",
                f"- Dominant latest state loss: `{values['dominant_latest_loss_stage']}`",
                f"- Dominant pattern: `{values['dominant_trajectory_pattern']}`",
                f"- Dominant primary reason: `{values['dominant_primary_reason']}`",
                "",
            ]
        )
    report.extend(
        [
            "## Recommendation",
            "",
            f"`recommended_next_step = {recommended}`",
            "",
            "The recommendation is diagnostic, not a strict-default change.",
        ]
    )
    (output_dir / "FIG9_CONTEXT_TRAJECTORY_DECOMPOSITION_REPORT.md").write_text(
        "\n".join(report) + "\n",
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
