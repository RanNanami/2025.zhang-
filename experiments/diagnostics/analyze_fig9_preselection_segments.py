"""Offline analysis for the read-only Fig.9 preselection segment funnel."""

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
from typing import Callable, Iterable, Mapping, Sequence

from experiments.diagnostics.analyze_fig9_candidate_score_separability import (
    average_precision,
    roc_auc,
)


METRICS = (
    "response_peak",
    "threshold_margin",
    "first_crossing_time",
    "predicted_time",
    "candidate_score_value",
    "contributor_count",
    "current_context_jaccard",
    "creation_current_source_jaccard",
)


def _float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bool(value: object) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _read_csv(run_dir: Path, stem: str) -> list[dict[str, str]]:
    plain = run_dir / f"{stem}.csv"
    compressed = run_dir / f"{stem}.csv.gz"
    if plain.exists():
        handle_context = plain.open("r", encoding="utf-8", newline="")
    elif compressed.exists():
        handle_context = gzip.open(
            compressed,
            "rt",
            encoding="utf-8",
            newline="",
        )
    else:
        return []
    with handle_context as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(rows[0]) if rows else list(fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not names:
            return
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _group(
    rows: Iterable[dict[str, str]],
    keys: Sequence[str],
) -> dict[tuple[str, ...], list[dict[str, str]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    return grouped


def _stage_rows(
    funnel: list[dict[str, str]],
    *,
    policy: str,
) -> list[dict[str, object]]:
    grouped = _group(funnel, ("horizon_step",))
    rows: list[dict[str, object]] = []
    for (step,), members in sorted(grouped.items()):
        totals = {
            key: sum(int(float(row.get(key, 0) or 0)) for row in members)
            for key in (
                "inspected_segment_count",
                "response_computed_count",
                "positive_response_count",
                "threshold_crossing_count",
                "event_winner_count",
                "saved_candidate_count",
                "emitted_candidate_count",
                "target_column_crossing_count",
                "target_column_candidate_count",
                "target_column_emitted_count",
                "false_column_crossing_count",
                "false_column_candidate_count",
                "false_column_emitted_count",
                "unknown_elimination_count",
            )
        }
        crossed = totals["threshold_crossing_count"]
        candidates = totals["saved_candidate_count"]
        rows.append(
            {
                "policy": policy,
                "horizon_step": step,
                "rollout_steps": len(members),
                **totals,
                "crossing_rate_per_inspected": (
                    crossed / totals["inspected_segment_count"]
                    if totals["inspected_segment_count"]
                    else 0.0
                ),
                "crossing_to_candidate_survival": (
                    candidates / crossed if crossed else 0.0
                ),
                "candidate_to_emitted_survival": (
                    totals["emitted_candidate_count"] / candidates
                    if candidates
                    else 0.0
                ),
            }
        )
    return rows


def _field_rows(
    segments: list[dict[str, str]],
    *,
    policy: str,
) -> list[dict[str, object]]:
    grouped = _group(segments, ("horizon_step", "field"))
    rows: list[dict[str, object]] = []
    for (step, field), members in sorted(grouped.items()):
        target = [row for row in members if _bool(row["is_target_column"])]
        false = [row for row in members if not _bool(row["is_target_column"])]
        candidate = [
            row
            for row in members
            if _bool(row["became_prediction_candidate"])
        ]
        emitted = [
            row for row in candidate if _bool(row["emitted_after_competition"])
        ]
        rows.append(
            {
                "policy": policy,
                "horizon_step": step,
                "field": field,
                "threshold_segments": len(members),
                "target_segments": len(target),
                "false_segments": len(false),
                "target_prevalence": (
                    len(target) / len(members) if members else 0.0
                ),
                "candidate_segments": len(candidate),
                "emitted_segments": len(emitted),
                "target_segment_candidate_survival": (
                    sum(_bool(row["became_prediction_candidate"]) for row in target)
                    / len(target)
                    if target
                    else 0.0
                ),
                "false_segment_candidate_survival": (
                    sum(_bool(row["became_prediction_candidate"]) for row in false)
                    / len(false)
                    if false
                    else 0.0
                ),
            }
        )
    return rows


def _target_survival_rows(
    funnel: list[dict[str, str]],
    *,
    policy: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    survival: list[dict[str, object]] = []
    for row in funnel:
        outcomes = json.loads(row.get("target_column_outcomes", "[]"))
        for outcome in outcomes:
            survival.append(
                {
                    "policy": policy,
                    "input_index": row["input_index"],
                    "horizon_step": row["horizon_step"],
                    "field": outcome["field"],
                    "target_column": outcome["column"],
                    "outcome": outcome["outcome"],
                }
            )
    grouped = _group(
        [
            {key: str(value) for key, value in row.items()}
            for row in survival
        ],
        ("horizon_step", "field", "outcome"),
    )
    losses = [
        {
            "policy": policy,
            "horizon_step": step,
            "field": field,
            "outcome": outcome,
            "count": len(members),
        }
        for (step, field, outcome), members in sorted(grouped.items())
    ]
    return survival, losses


def _standardized_difference(
    target: Sequence[float],
    false: Sequence[float],
) -> float:
    if not target or not false:
        return 0.0
    target_variance = statistics.pvariance(target) if len(target) > 1 else 0.0
    false_variance = statistics.pvariance(false) if len(false) > 1 else 0.0
    pooled = math.sqrt((target_variance + false_variance) / 2.0)
    return (_mean(target) - _mean(false)) / pooled if pooled else 0.0


def _separability_rows(
    segments: list[dict[str, str]],
    metrics: Sequence[str],
    *,
    policy: str,
) -> list[dict[str, object]]:
    grouped = _group(segments, ("horizon_step", "field"))
    rows: list[dict[str, object]] = []
    for (step, field), members in sorted(grouped.items()):
        for metric in metrics:
            selected = [
                (row, _float(row.get(metric)))
                for row in members
            ]
            selected = [
                (row, value)
                for row, value in selected
                if value is not None
            ]
            labels = [_bool(row["is_target_column"]) for row, _ in selected]
            values = [value for _row, value in selected]
            target = [
                value
                for (row, value) in selected
                if _bool(row["is_target_column"])
            ]
            false = [
                value
                for (row, value) in selected
                if not _bool(row["is_target_column"])
            ]
            descending = metric not in {
                "first_crossing_time",
                "predicted_time",
            }
            ranked = sorted(
                selected,
                key=lambda pair: pair[1],
                reverse=descending,
            )
            hit = {
                budget: any(
                    _bool(row["is_target_column"])
                    for row, _value in ranked[:budget]
                )
                for budget in (1, 3, 5, 10)
            }
            rows.append(
                {
                    "policy": policy,
                    "horizon_step": step,
                    "field": field,
                    "metric": metric,
                    "count": len(values),
                    "positive_prevalence": (
                        sum(labels) / len(labels) if labels else 0.0
                    ),
                    "roc_auc": roc_auc(labels, values) or "",
                    "pr_auc": average_precision(labels, values) or "",
                    "target_mean": _mean(target),
                    "target_median": statistics.median(target) if target else 0.0,
                    "target_p10": _percentile(target, 0.10),
                    "target_p90": _percentile(target, 0.90),
                    "false_mean": _mean(false),
                    "false_median": statistics.median(false) if false else 0.0,
                    "false_p10": _percentile(false, 0.10),
                    "false_p90": _percentile(false, 0.90),
                    "standardized_mean_difference": (
                        _standardized_difference(target, false)
                    ),
                    "hit_at_1": hit[1],
                    "hit_at_3": hit[3],
                    "hit_at_5": hit[5],
                    "hit_at_10": hit[10],
                }
            )
    return rows


def _internal_group_rows(
    groups: list[dict[str, str]],
    segments: list[dict[str, str]],
    *,
    policy: str,
) -> list[dict[str, object]]:
    by_group = _group(segments, ("selection_group_id",))
    rows: list[dict[str, object]] = []
    rank_metrics: dict[str, tuple[str, bool]] = {
        "existing_score": ("candidate_score_value", True),
        "highest_response_peak": ("response_peak", True),
        "earliest_crossing": ("first_crossing_time", False),
        "highest_contributor_count": ("contributor_count", True),
        "highest_current_context_jaccard": (
            "current_context_jaccard",
            True,
        ),
        "highest_creation_current_jaccard": (
            "creation_current_source_jaccard",
            True,
        ),
    }
    for group in groups:
        members = by_group.get((group["selection_group_id"],), [])
        output: dict[str, object] = {
            "policy": policy,
            **group,
        }
        for label, (metric, descending) in rank_metrics.items():
            available = [
                (row, _float(row.get(metric)))
                for row in members
                if _float(row.get(metric)) is not None
            ]
            selected = (
                sorted(
                    available,
                    key=lambda pair: pair[1],  # type: ignore[arg-type]
                    reverse=descending,
                )[0][0]
                if available
                else None
            )
            output[f"{label}_winner_is_target"] = (
                _bool(selected["is_target_column"])
                if selected is not None
                else ""
            )
        rows.append(output)
    return rows


def _winner_discarded_rows(
    groups: list[dict[str, str]],
    segments: list[dict[str, str]],
    *,
    policy: str,
) -> list[dict[str, object]]:
    by_group = _group(segments, ("selection_group_id",))
    output: list[dict[str, object]] = []
    for group in groups:
        if not _bool(group.get("mixed_target_false_group", "")):
            continue
        members = by_group.get((group["selection_group_id"],), [])
        winner = next(
            (
                row
                for row in members
                if row["segment_provenance_id"]
                == group.get("winner_segment_id", "")
            ),
            None,
        )
        discarded_targets = [
            row
            for row in members
            if _bool(row["is_target_column"]) and row is not winner
        ]
        best_target = max(
            discarded_targets,
            key=lambda row: _float(row.get("candidate_score_value"))
            or float("-inf"),
            default=None,
        )
        if winner is None or best_target is None:
            continue
        row: dict[str, object] = {
            "policy": policy,
            "selection_group_id": group["selection_group_id"],
            "horizon_step": group["horizon_step"],
            "field": group["field"],
            "winner_is_target": _bool(winner["is_target_column"]),
        }
        for metric in METRICS:
            row[f"winner_{metric}"] = winner.get(metric, "")
            row[f"discarded_target_{metric}"] = best_target.get(metric, "")
        output.append(row)
    return output


def _bootstrap_rows(
    segments: list[dict[str, str]],
    *,
    policy: str,
    samples: int,
    seed: int,
) -> list[dict[str, object]]:
    passenger = [
        row for row in segments if row.get("field") == "passenger"
    ]
    rollouts = _group(passenger, ("input_index",))
    keys = list(rollouts)
    if not keys or samples <= 0:
        return []
    rng = random.Random(seed)
    output: list[dict[str, object]] = []
    for metric in ("candidate_score_value", "current_context_jaccard"):
        values: list[float] = []
        for _ in range(samples):
            sampled = [
                row
                for _index in keys
                for row in rollouts[rng.choice(keys)]
            ]
            pairs = [
                (row, _float(row.get(metric)))
                for row in sampled
                if _float(row.get(metric)) is not None
            ]
            labels = [_bool(row["is_target_column"]) for row, _ in pairs]
            scores = [value for _row, value in pairs]
            result = average_precision(labels, scores)
            if result is not None:
                values.append(result)
        output.append(
            {
                "policy": policy,
                "field": "passenger",
                "metric": metric,
                "statistic": "pr_auc",
                "bootstrap_samples": len(values),
                "estimate_mean": _mean(values),
                "ci_lower": _percentile(values, 0.025),
                "ci_upper": _percentile(values, 0.975),
            }
        )
    return output


def analyze(
    *,
    run_dir: Path,
    output_dir: Path,
    policy_label: str,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    """Analyze existing CSV traces without loading or mutating a model."""

    funnel = _read_csv(run_dir, "preselection_funnel_trace")
    segments = _read_csv(run_dir, "preselection_segment_trace")
    groups = _read_csv(run_dir, "preselection_group_trace")
    replacements = _read_csv(run_dir, "preselection_replacement_trace")
    stage = _stage_rows(funnel, policy=policy_label)
    field = _field_rows(segments, policy=policy_label)
    survival, losses = _target_survival_rows(
        funnel,
        policy=policy_label,
    )
    score = _separability_rows(
        segments,
        (
            "response_peak",
            "threshold_margin",
            "first_crossing_time",
            "predicted_time",
            "candidate_score_value",
            "contributor_count",
        ),
        policy=policy_label,
    )
    context = _separability_rows(
        segments,
        (
            "current_context_jaccard",
            "creation_current_source_jaccard",
        ),
        policy=policy_label,
    )
    internal = _internal_group_rows(
        groups,
        segments,
        policy=policy_label,
    )
    winner_discarded = _winner_discarded_rows(
        groups,
        segments,
        policy=policy_label,
    )
    bootstrap = _bootstrap_rows(
        segments,
        policy=policy_label,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "preselection_stage_summary.csv", stage)
    _write_csv(output_dir / "preselection_field_summary.csv", field)
    _write_csv(output_dir / "preselection_horizon_summary.csv", stage)
    _write_csv(output_dir / "target_segment_survival.csv", survival)
    _write_csv(output_dir / "target_loss_stage_summary.csv", losses)
    _write_csv(output_dir / "internal_group_competition.csv", internal)
    _write_csv(output_dir / "segment_score_separability.csv", score)
    _write_csv(output_dir / "segment_context_separability.csv", context)
    _write_csv(
        output_dir / "winner_vs_discarded_summary.csv",
        winner_discarded,
    )
    _write_csv(output_dir / "preselection_bootstrap_ci.csv", bootstrap)
    outcome_counts = Counter(row["outcome"] for row in survival)
    unknown = sum(
        int(float(row.get("unknown_elimination_count", 0) or 0))
        for row in funnel
    )
    inspected = sum(
        int(float(row.get("inspected_segment_count", 0) or 0))
        for row in funnel
    )
    crossed = sum(
        int(float(row.get("threshold_crossing_count", 0) or 0))
        for row in funnel
    )
    candidates = sum(
        int(float(row.get("saved_candidate_count", 0) or 0))
        for row in funnel
    )
    summary = {
        "diagnostic_only": True,
        "offline_analysis_only": True,
        "preselection_segment_diagnostic": True,
        "uses_ground_truth_for_analysis_only": True,
        "ground_truth_does_not_affect_prediction": True,
        "segment_trace_does_not_affect_selection": True,
        "policy": policy_label,
        "run_dir": str(run_dir),
        "funnel_steps": len(funnel),
        "segment_rows": len(segments),
        "group_rows": len(groups),
        "replacement_rows": len(replacements),
        "inspected_segments": inspected,
        "threshold_crossing_segments": crossed,
        "saved_candidates": candidates,
        "crossing_to_candidate_ratio": (
            candidates / crossed if crossed else 0.0
        ),
        "unknown_elimination_count": unknown,
        "unknown_elimination_rate": unknown / inspected if inspected else 0.0,
        "target_outcomes": dict(outcome_counts),
        "target_vs_false_within_group_testable": False,
        "target_vs_false_within_group_reason": (
            "existing event and column groups contain one column; the "
            "oracle defines target columns but not target neurons"
        ),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_unit": "rollout input_index",
        "empty_trace": not funnel,
    }
    (output_dir / "preselection_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    passenger_losses = Counter(
        row["outcome"]
        for row in survival
        if row["field"] == "passenger"
    )
    report = [
        "# Fig.9 Preselection Segment Diagnostic",
        "",
        "This is a nonpaper, offline, oracle-labelled diagnostic.",
        "Ground truth never enters prediction or selection.",
        (
            "Target-vs-false competition within an internal group is not "
            "testable: every member has the same column and the encoder "
            "does not define a target neuron."
        ),
        "",
        "## Funnel",
        "",
        f"- Inspected segments: {inspected}",
        f"- Threshold-crossing segments: {crossed}",
        f"- Saved PredictionCandidates: {candidates}",
        (
            "- Crossing to candidate ratio: "
            f"{summary['crossing_to_candidate_ratio']:.6f}"
        ),
        f"- Unknown elimination rate: {summary['unknown_elimination_rate']:.6f}",
        "",
        "## Passenger Target Outcomes",
        "",
    ]
    report.extend(
        f"- {name}: {count}"
        for name, count in sorted(passenger_losses.items())
    )
    report.extend(
        [
            "",
            "No new selector or MAPE result is produced by this analysis.",
            "",
        ]
    )
    (output_dir / "FIG9_PRESELECTION_SEGMENT_REPORT.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy-label", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze(
        run_dir=Path(args.run_dir),
        output_dir=Path(args.output_dir),
        policy_label=args.policy_label,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
