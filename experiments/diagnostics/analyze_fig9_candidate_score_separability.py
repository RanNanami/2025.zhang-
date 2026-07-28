"""Offline score-separability analysis for Fig.9 competition candidates.

The tool reads completed diagnostic traces. Ground truth labels are never
passed back into prediction, competition, decoding, or learning.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.diagnostics.fig9_candidate_score_trace import (
    TRACE_FIELDS,
    validate_trace_schema,
)


ANALYSIS_MARKERS = {
    "diagnostic_only": True,
    "uses_ground_truth_for_analysis_only": True,
    "ground_truth_does_not_affect_prediction": True,
    "offline_analysis_only": True,
}
FIELDS = ("weekday", "time", "passenger", "total")
SCORE_FIELDS = (
    "original_score",
    "effective_score",
    "predicted_time",
    "accumulated_inhibition",
)
RECALL_TARGETS = (0.30, 0.50, 0.70, 0.90)
COLUMN_BUDGETS = (10, 20, 50, 75, 100)


def read_candidate_trace(path: Path, policy: str) -> list[dict[str, object]]:
    """Read and type-check one candidate-level trace."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_trace_schema(reader.fieldnames)
        rows = [
            {
                **row,
                "policy": policy,
                "input_index": int(row["input_index"]),
                "horizon_step": int(row["horizon_step"]),
                "candidate_original_index": int(
                    row["candidate_original_index"]
                ),
                "column": int(row["column"]),
                "neuron": int(row["neuron"]),
                "batch_index": int(row["batch_index"]),
                "predicted_time": float(row["predicted_time"]),
                "original_score": float(row["original_score"]),
                "accumulated_inhibition": float(
                    row["accumulated_inhibition"]
                ),
                "effective_score": float(row["effective_score"]),
                "emitted": _boolean(row["emitted"]),
                "is_target_candidate": _boolean(
                    row["is_target_candidate"]
                ),
            }
            for row in reader
        ]
    return rows


def validate_protocol(protocol: Mapping[str, object]) -> dict[str, tuple[int, int]]:
    """Return contiguous field ranges and verify their protocol total."""

    sizes = protocol.get("encoder_sizes")
    if not isinstance(sizes, Mapping):
        raise ValueError("protocol is missing encoder_sizes")
    try:
        weekday = int(sizes["weekday"])
        time_size = int(sizes["time"])
        passenger = int(sizes["passenger"])
        total = int(sizes["total"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("protocol encoder_sizes are incomplete") from error
    if weekday + time_size + passenger != total:
        raise ValueError(
            "protocol field sizes do not sum to total: "
            f"{weekday}+{time_size}+{passenger}!={total}"
        )
    return {
        "weekday": (0, weekday),
        "time": (weekday, weekday + time_size),
        "passenger": (weekday + time_size, total),
        "total": (0, total),
    }


def aggregate_columns(
    candidate_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Select each column's highest-original-score mapped candidate.

    This mirrors the existing raw candidate mapping: score is maximized and a
    stable original event index resolves equal scores. Scores are never
    averaged across neurons.
    """

    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(
        list
    )
    for row in candidate_rows:
        grouped[
            (
                row["policy"],
                row["input_index"],
                row["target_timestamp"],
                row["horizon_step"],
                row["field"],
                row["column"],
            )
        ].append(row)

    result: list[dict[str, object]] = []
    for key in sorted(grouped, key=lambda item: tuple(map(str, item))):
        candidates = grouped[key]
        selected = min(
            candidates,
            key=lambda row: (
                -float(row["original_score"]),
                int(row["candidate_original_index"]),
                int(row["neuron"]),
            ),
        )
        result.append(
            {
                **ANALYSIS_MARKERS,
                "policy": selected["policy"],
                "input_index": selected["input_index"],
                "target_timestamp": selected["target_timestamp"],
                "horizon_step": selected["horizon_step"],
                "field": selected["field"],
                "column": selected["column"],
                "selected_neuron": selected["neuron"],
                "candidate_count_in_column": len(candidates),
                "max_original_score": selected["original_score"],
                "corresponding_effective_score": selected["effective_score"],
                "corresponding_predicted_time": selected["predicted_time"],
                "corresponding_inhibition": selected[
                    "accumulated_inhibition"
                ],
                "any_emitted": any(bool(row["emitted"]) for row in candidates),
                "is_target_column": bool(selected["is_target_candidate"]),
            }
        )
    return result


def roc_auc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    """Tie-aware ROC-AUC using the Mann-Whitney rank identity."""

    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = _average_ranks(scores)
    positive_rank_sum = sum(
        rank for rank, label in zip(ranks, labels) if label
    )
    u = positive_rank_sum - positives * (positives + 1) / 2
    return u / (positives * negatives)


def average_precision(
    labels: Sequence[bool],
    scores: Sequence[float],
) -> float | None:
    """Tie-grouped average precision with prevalence behavior for all ties."""

    positives = sum(labels)
    if positives == 0:
        return None
    ordered = sorted(
        zip(scores, labels),
        key=lambda item: item[0],
        reverse=True,
    )
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    area = 0.0
    cursor = 0
    while cursor < len(ordered):
        score = ordered[cursor][0]
        end = cursor
        group_true = 0
        group_false = 0
        while end < len(ordered) and ordered[end][0] == score:
            if ordered[end][1]:
                group_true += 1
            else:
                group_false += 1
            end += 1
        true_positive += group_true
        false_positive += group_false
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        cursor = end
    return area


def threshold_sweep(
    rows: Sequence[Mapping[str, object]],
    score_field: str,
) -> list[dict[str, object]]:
    """Sweep all unique column scores from strictest to loosest."""

    if not rows:
        return []
    positives = sum(bool(row["is_target_column"]) for row in rows)
    negatives = len(rows) - positives
    ordered = sorted(
        rows,
        key=lambda row: float(row[score_field]),
        reverse=True,
    )
    output: list[dict[str, object]] = []
    true_count = 0
    false_count = 0
    cursor = 0
    while cursor < len(ordered):
        threshold = float(ordered[cursor][score_field])
        end = cursor
        while (
            end < len(ordered)
            and float(ordered[end][score_field]) == threshold
        ):
            if bool(ordered[end]["is_target_column"]):
                true_count += 1
            else:
                false_count += 1
            end += 1
        retained_count = true_count + false_count
        output.append(
            {
                "threshold": threshold,
                "target_recall": _ratio(true_count, positives),
                "precision": _ratio(true_count, retained_count),
                "false_positive_rate": _ratio(false_count, negatives),
                "false_emitted_ratio": _ratio(
                    false_count,
                    retained_count,
                ),
                "true_target_columns_retained": true_count,
                "false_columns_retained": false_count,
                "total_columns_retained": retained_count,
            }
        )
        cursor = end
    return output


def fixed_budget_recall(
    rows: Sequence[Mapping[str, object]],
    score_field: str,
    budget: int,
) -> float:
    positives = sum(bool(row["is_target_column"]) for row in rows)
    if not positives:
        return 0.0
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row[score_field]),
            int(row["column"]),
        ),
    )
    retained = ordered[:budget]
    return sum(bool(row["is_target_column"]) for row in retained) / positives


def bootstrap_by_rollout(
    rows: Sequence[Mapping[str, object]],
    *,
    samples: int,
    seed: int,
    metric: Callable[[Sequence[Mapping[str, object]]], float | None],
) -> list[float]:
    """Resample complete rollout starts; candidates are never IID units."""

    by_rollout: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_rollout[int(row["input_index"])].append(row)
    keys = sorted(by_rollout)
    if not keys or samples <= 0:
        return []
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        sampled: list[Mapping[str, object]] = []
        for _draw in keys:
            sampled.extend(by_rollout[rng.choice(keys)])
        value = metric(sampled)
        if value is not None and math.isfinite(value):
            estimates.append(value)
    return estimates


def bootstrap_rollout_metric_means(
    rows: Sequence[Mapping[str, object]],
    *,
    samples: int,
    seed: int,
    metric: Callable[[Sequence[Mapping[str, object]]], float | None],
) -> list[float]:
    """Bootstrap the mean of metrics computed on complete rollout units."""

    by_rollout: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_rollout[int(row["input_index"])].append(row)
    rollout_values = [
        value
        for key in sorted(by_rollout)
        for value in [metric(by_rollout[key])]
        if value is not None and math.isfinite(value)
    ]
    if not rollout_values or samples <= 0:
        return []
    rng = random.Random(seed)
    return [
        _mean(
            [
                rollout_values[rng.randrange(len(rollout_values))]
                for _ in rollout_values
            ]
        )
        for _ in range(samples)
    ]


def compare_policy_candidate_pools(
    sequential_rows: Sequence[Mapping[str, object]],
    batched_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Audit fixed Step-1 pools and flag later recurrent divergence."""

    output: list[dict[str, object]] = []
    for step in range(1, 6):
        sequential = _pool_signatures(sequential_rows, step)
        batched = _pool_signatures(batched_rows, step)
        common = sorted(set(sequential) & set(batched))
        matches = sum(
            sequential[key] == batched[key]
            for key in common
        )
        output.append(
            {
                "horizon_step": step,
                "direct_candidate_pool_comparison_allowed": step == 1,
                "trajectory_divergence": step > 1,
                "aligned_rollouts": len(common),
                "matching_pool_rollouts": matches,
                "matching_pool_rate": _ratio(matches, len(common)),
            }
        )
    return output


def analyze(
    policy_inputs: Sequence[tuple[str, Path]],
    *,
    output_dir: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    """Run all requested offline tables for one or two policies."""

    all_candidates: list[dict[str, object]] = []
    input_manifest: list[dict[str, object]] = []
    protocol_ranges: dict[str, dict[str, tuple[int, int]]] = {}
    for policy, directory in policy_inputs:
        trace_path = directory / "candidate_separability_trace.csv"
        competition_path = directory / "competition_trace.csv"
        oracle_path = directory / "oracle_candidate_trace.csv"
        protocol_path = directory / "original_protocol.json"
        predictions_path = directory / "original_predictions.csv"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol_ranges[policy] = validate_protocol(protocol)
        rows = read_candidate_trace(trace_path, policy)
        _validate_trace_columns(rows, protocol_ranges[policy])
        all_candidates.extend(rows)
        input_manifest.append(
            {
                "policy": policy,
                "input_directory": str(directory.resolve()),
                "git_commit_sha": protocol.get(
                    "git_commit_sha",
                    "unavailable",
                ),
                "candidate_trace_sha256": _sha256(trace_path),
                "competition_trace_sha256": _sha256(competition_path),
                "oracle_trace_sha256": _sha256(oracle_path),
                "predictions_sha256": _sha256(predictions_path),
                "protocol_sha256": _sha256(protocol_path),
            }
        )

    columns = aggregate_columns(all_candidates)
    distributions = _distribution_tables(all_candidates, columns)
    separability = _separability_tables(all_candidates, columns)
    curves, budgets = _threshold_and_budget_tables(columns)
    batches = _batch_tables(all_candidates)
    field_scale = _field_scale_tables(all_candidates)
    horizon = _horizon_tables(separability, budgets, batches)
    bootstrap = _bootstrap_tables(
        all_candidates,
        columns,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "candidate_score_rows.csv", all_candidates)
    _write_csv(output_dir / "column_score_rows.csv", columns)
    _write_csv(
        output_dir / "score_distribution_summary.csv",
        distributions,
    )
    _write_csv(output_dir / "separability_metrics.csv", separability)
    _write_csv(output_dir / "threshold_curves.csv", curves)
    _write_csv(output_dir / "recall_budget_summary.csv", budgets)
    _write_csv(output_dir / "batch_separability_summary.csv", batches)
    _write_csv(output_dir / "field_scale_summary.csv", field_scale)
    _write_csv(
        output_dir / "horizon_degradation_summary.csv",
        horizon,
    )
    _write_csv(
        output_dir / "bootstrap_confidence_intervals.csv",
        bootstrap,
    )

    summary = {
        **ANALYSIS_MARKERS,
        "git_commit": _common_manifest_value(
            input_manifest,
            "git_commit_sha",
        ),
        "inputs": input_manifest,
        "protocol_field_ranges": protocol_ranges,
        "candidate_aggregation_rule": (
            "one saved CompetitionDecision is one candidate sample"
        ),
        "column_aggregation_rule": (
            "highest original_score; ties use candidate_original_index then neuron"
        ),
        "bootstrap_unit": "rollout input_index",
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "candidate_rows": len(all_candidates),
        "column_rows": len(columns),
        "provenance_available": False,
        "headline_metrics": _headline_metrics(separability, budgets, batches),
        "policy_candidate_pool_comparison": (
            compare_policy_candidate_pools(
                [
                    row
                    for row in all_candidates
                    if row["policy"] == "sequential"
                ],
                [
                    row
                    for row in all_candidates
                    if row["policy"] == "batched"
                ],
            )
            if {row["policy"] for row in all_candidates}
            == {"sequential", "batched"}
            else []
        ),
        "trajectory_comparison_rule": (
            "step 1 candidate pools may be compared directly; steps 2-5 are "
            "policy-level distributions after recurrent trajectory divergence"
        ),
    }
    (output_dir / "candidate_score_separability_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "CANDIDATE_SCORE_SEPARABILITY_REPORT.md").write_text(
        _render_report(summary),
        encoding="utf-8",
    )
    return summary


def _distribution_tables(
    candidates: Sequence[Mapping[str, object]],
    columns: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    level_specs = (
        ("candidate", candidates, "is_target_candidate", {
            "original_score": "original_score",
            "effective_score": "effective_score",
            "predicted_time": "predicted_time",
            "accumulated_inhibition": "accumulated_inhibition",
        }),
        ("column", columns, "is_target_column", {
            "original_score": "max_original_score",
            "effective_score": "corresponding_effective_score",
            "predicted_time": "corresponding_predicted_time",
            "accumulated_inhibition": "corresponding_inhibition",
        }),
    )
    for level, source, label_field, metric_fields in level_specs:
        for key, selected in _analysis_groups(source):
            labels = [bool(row[label_field]) for row in selected]
            for metric, source_field in metric_fields.items():
                positive = [
                    float(row[source_field])
                    for row in selected
                    if bool(row[label_field])
                ]
                negative = [
                    float(row[source_field])
                    for row in selected
                    if not bool(row[label_field])
                ]
                output.append(
                    {
                        **ANALYSIS_MARKERS,
                        "level": level,
                        "policy": key[0],
                        "horizon_step": key[1],
                        "field": key[2],
                        "metric": metric,
                        "count": len(selected),
                        "positive_count": sum(labels),
                        "negative_count": len(labels) - sum(labels),
                        "positive_prevalence": _ratio(
                            sum(labels), len(labels)
                        ),
                        **_prefixed_stats("target", positive),
                        **_prefixed_stats("false", negative),
                        "target_mean_minus_false_mean": (
                            _mean(positive) - _mean(negative)
                            if positive and negative
                            else ""
                        ),
                        "standardized_mean_difference": _standardized_difference(
                            positive, negative
                        ),
                        "median_difference": (
                            statistics.median(positive)
                            - statistics.median(negative)
                            if positive and negative
                            else ""
                        ),
                        "distribution_overlap_coefficient": _overlap(
                            positive, negative
                        ),
                    }
                )
    return output


def _separability_tables(
    candidates: Sequence[Mapping[str, object]],
    columns: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    level_specs = (
        ("candidate", candidates, "is_target_candidate", {
            "original_score": lambda row: float(row["original_score"]),
            "effective_score": lambda row: float(row["effective_score"]),
            "negative_predicted_time": lambda row: -float(row["predicted_time"]),
        }),
        ("column", columns, "is_target_column", {
            "original_score": lambda row: float(row["max_original_score"]),
            "effective_score": lambda row: float(
                row["corresponding_effective_score"]
            ),
            "negative_predicted_time": lambda row: -float(
                row["corresponding_predicted_time"]
            ),
        }),
    )
    for level, source, label_field, indicators in level_specs:
        for key, selected in _analysis_groups(source):
            labels = [bool(row[label_field]) for row in selected]
            original = [indicators["original_score"](row) for row in selected]
            negative_time = [
                indicators["negative_predicted_time"](row) for row in selected
            ]
            combined = [
                left + right
                for left, right in zip(
                    _z_scores(original),
                    _z_scores(negative_time),
                )
            ]
            expanded = dict(indicators)
            expanded["score_time_2d"] = lambda row: 0.0
            for indicator, getter in expanded.items():
                scores = (
                    combined
                    if indicator == "score_time_2d"
                    else [getter(row) for row in selected]
                )
                auc = roc_auc(labels, scores)
                ap = average_precision(labels, scores)
                prevalence = _ratio(sum(labels), len(labels))
                ranking = _ranking_metrics(
                    selected,
                    labels,
                    scores,
                )
                output.append(
                    {
                        **ANALYSIS_MARKERS,
                        "level": level,
                        "policy": key[0],
                        "horizon_step": key[1],
                        "field": key[2],
                        "indicator": indicator,
                        "sample_count": len(labels),
                        "positive_count": sum(labels),
                        "positive_prevalence": prevalence,
                        "roc_auc": auc if auc is not None else "",
                        "pr_auc": ap if ap is not None else "",
                        "pr_auc_prevalence_ratio": (
                            ap / prevalence
                            if ap is not None and prevalence
                            else ""
                        ),
                        "mann_whitney_u": (
                            auc * sum(labels) * (len(labels) - sum(labels))
                            if auc is not None
                            else ""
                        ),
                        **ranking,
                    }
                )
    return output


def _threshold_and_budget_tables(
    columns: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    curves: list[dict[str, object]] = []
    budgets: list[dict[str, object]] = []
    score_fields = {
        "original_score": "max_original_score",
        "effective_score": "corresponding_effective_score",
    }
    for key, selected in _analysis_groups(columns):
        for indicator, source_field in score_fields.items():
            sweep = threshold_sweep(selected, source_field)
            curves.extend(
                {
                    **ANALYSIS_MARKERS,
                    "policy": key[0],
                    "horizon_step": key[1],
                    "field": key[2],
                    "indicator": indicator,
                    **row,
                }
                for row in sweep
            )
            for target_recall in RECALL_TARGETS:
                feasible = [
                    row
                    for row in sweep
                    if float(row["target_recall"]) >= target_recall
                ]
                best = min(
                    feasible,
                    key=lambda row: (
                        int(row["false_columns_retained"]),
                        int(row["total_columns_retained"]),
                        -float(row["threshold"]),
                    ),
                ) if feasible else None
                budgets.append(
                    {
                        **ANALYSIS_MARKERS,
                        "policy": key[0],
                        "horizon_step": key[1],
                        "field": key[2],
                        "indicator": indicator,
                        "summary_type": "target_recall",
                        "target_recall_requirement": target_recall,
                        "column_budget": "",
                        "achieved_target_recall": (
                            best["target_recall"] if best else ""
                        ),
                        "false_columns_retained": (
                            best["false_columns_retained"] if best else ""
                        ),
                        "false_emitted_ratio": (
                            best["false_emitted_ratio"] if best else ""
                        ),
                        "total_columns_retained": (
                            best["total_columns_retained"] if best else ""
                        ),
                        "feasible": best is not None,
                    }
                )
            for budget in COLUMN_BUDGETS:
                budgets.append(
                    {
                        **ANALYSIS_MARKERS,
                        "policy": key[0],
                        "horizon_step": key[1],
                        "field": key[2],
                        "indicator": indicator,
                        "summary_type": "fixed_budget",
                        "target_recall_requirement": "",
                        "column_budget": budget,
                        "achieved_target_recall": fixed_budget_recall(
                            selected,
                            source_field,
                            budget,
                        ),
                        "false_columns_retained": "",
                        "false_emitted_ratio": "",
                        "total_columns_retained": min(
                            budget, len(selected)
                        ),
                        "feasible": True,
                    }
                )
    return curves, budgets


def _batch_tables(
    candidates: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    batched = [row for row in candidates if row["policy"] == "batched"]
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(
        list
    )
    for row in batched:
        grouped[
            (
                row["input_index"],
                row["horizon_step"],
                row["target_timestamp"],
                row["batch_index"],
            )
        ].append(row)
    output: list[dict[str, object]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        target = [row for row in rows if bool(row["is_target_candidate"])]
        false = [row for row in rows if not bool(row["is_target_candidate"])]
        ordered = sorted(
            rows,
            key=lambda row: (
                -float(row["original_score"]),
                int(row["column"]),
                int(row["neuron"]),
                int(row["candidate_original_index"]),
            ),
        )
        target_ranks = [
            rank
            for rank, row in enumerate(ordered, start=1)
            if bool(row["is_target_candidate"])
        ]
        target_best = max(
            (float(row["original_score"]) for row in target),
            default=None,
        )
        false_best = max(
            (float(row["original_score"]) for row in false),
            default=None,
        )
        mixed = bool(target and false)
        output.append(
            {
                **ANALYSIS_MARKERS,
                "row_type": "batch",
                "policy": "batched",
                "input_index": key[0],
                "horizon_step": key[1],
                "target_timestamp": key[2],
                "batch_index": key[3],
                "candidate_count": len(rows),
                "target_candidate_count": len(target),
                "false_candidate_count": len(false),
                "target_best_score": target_best if target_best is not None else "",
                "false_best_score": false_best if false_best is not None else "",
                "target_best_score_rank": min(target_ranks) if target_ranks else "",
                "highest_score_is_target": bool(ordered and ordered[0]["is_target_candidate"]),
                "target_false_score_margin": (
                    target_best - false_best
                    if target_best is not None and false_best is not None
                    else ""
                ),
                "mixed_target_false": mixed,
                "target_emitted": any(bool(row["emitted"]) for row in target),
                "false_emitted": any(bool(row["emitted"]) for row in false),
                "target_is_top3": bool(
                    target_ranks and min(target_ranks) <= 3
                ),
                "false_outranks_all_targets": bool(
                    mixed and false_best is not None and target_best is not None
                    and false_best > target_best
                ),
                "score_can_separate": bool(
                    mixed and target_best is not None and false_best is not None
                    and target_best > false_best
                ),
                "complete_score_overlap": bool(
                    mixed
                    and {
                        float(row["original_score"]) for row in target
                    }
                    == {
                        float(row["original_score"]) for row in false
                    }
                ),
            }
        )
    for step in sorted({int(row["horizon_step"]) for row in output}):
        selected = [
            row for row in output if int(row["horizon_step"]) == step
        ]
        mixed = [row for row in selected if row["mixed_target_false"]]
        multi_candidate_total = sum(
            int(row["candidate_count"])
            for row in selected
            if int(row["candidate_count"]) > 1
        )
        candidate_total = sum(
            int(row["candidate_count"]) for row in selected
        )
        output.append(
            {
                **ANALYSIS_MARKERS,
                "row_type": "summary",
                "policy": "batched",
                "input_index": "",
                "horizon_step": step,
                "target_timestamp": "",
                "batch_index": "",
                "candidate_count": _mean(
                    [float(row["candidate_count"]) for row in selected]
                ),
                "target_candidate_count": "",
                "false_candidate_count": "",
                "target_best_score": "",
                "false_best_score": "",
                "target_best_score_rank": "",
                "highest_score_is_target": _rate(
                    mixed, "highest_score_is_target"
                ),
                "target_false_score_margin": "",
                "mixed_target_false": _ratio(len(mixed), len(selected)),
                "target_emitted": _rate(mixed, "target_emitted"),
                "false_emitted": _rate(mixed, "false_emitted"),
                "target_is_top3": _rate(mixed, "target_is_top3"),
                "false_outranks_all_targets": _rate(
                    mixed, "false_outranks_all_targets"
                ),
                "score_can_separate": _rate(mixed, "score_can_separate"),
                "complete_score_overlap": _rate(
                    mixed, "complete_score_overlap"
                ),
                "multi_candidate_batch_candidate_fraction": _ratio(
                    multi_candidate_total,
                    candidate_total,
                ),
            }
        )
    return output


def _field_scale_tables(
    candidates: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    base_groups: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(
        list
    )
    for row in candidates:
        base_groups[(str(row["policy"]), int(row["horizon_step"]))].append(row)
    for (policy, step), step_rows in sorted(base_groups.items()):
        normalized: dict[int, dict[str, float]] = {}
        for field in FIELDS[:-1]:
            field_rows = [row for row in step_rows if row["field"] == field]
            values = [float(row["original_score"]) for row in field_rows]
            z_values = _z_scores(values)
            percentile = _percentile_ranks(values)
            median = statistics.median(values) if values else 0.0
            for row, z_value, percentile_value in zip(
                field_rows, z_values, percentile
            ):
                normalized[id(row)] = {
                    "field_z_score": z_value,
                    "field_percentile": percentile_value,
                    "score_over_field_median": (
                        float(row["original_score"]) / median
                        if median
                        else 0.0
                    ),
                }
        for field in FIELDS:
            selected = (
                step_rows
                if field == "total"
                else [row for row in step_rows if row["field"] == field]
            )
            labels = [bool(row["is_target_candidate"]) for row in selected]
            for indicator in (
                "original_score",
                "field_z_score",
                "field_percentile",
                "score_over_field_median",
            ):
                scores = [
                    float(row["original_score"])
                    if indicator == "original_score"
                    else normalized[id(row)][indicator]
                    for row in selected
                ]
                output.append(
                    {
                        **ANALYSIS_MARKERS,
                        "policy": policy,
                        "horizon_step": step,
                        "field": field,
                        "indicator": indicator,
                        "count": len(selected),
                        "score_mean": _mean(scores),
                        "score_variance": _variance(scores),
                        "score_p10": _quantile(scores, 0.10),
                        "score_p50": _quantile(scores, 0.50),
                        "score_p90": _quantile(scores, 0.90),
                        "threshold_pass_rate": _ratio(
                            sum(score >= 1.0 for score in scores),
                            len(scores),
                        ) if indicator == "original_score" else "",
                        "emitted_rate": _ratio(
                            sum(bool(row["emitted"]) for row in selected),
                            len(selected),
                        ),
                        "positive_prevalence": _ratio(
                            sum(labels), len(labels)
                        ),
                        "pr_auc": average_precision(labels, scores) or 0.0,
                        "target_inhibition_mean": _mean(
                            [
                                float(row["accumulated_inhibition"])
                                for row in selected
                                if bool(row["is_target_candidate"])
                            ]
                        ),
                    }
                )
    return output


def _horizon_tables(
    separability: Sequence[Mapping[str, object]],
    budgets: Sequence[Mapping[str, object]],
    batches: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in separability:
        if (
            row["level"] == "column"
            and row["indicator"] in {"original_score", "effective_score"}
            and row["field"] in {"total", "passenger"}
        ):
            budget = next(
                (
                    item
                    for item in budgets
                    if item["policy"] == row["policy"]
                    and item["horizon_step"] == row["horizon_step"]
                    and item["field"] == row["field"]
                    and item["indicator"] == row["indicator"]
                    and item["summary_type"] == "fixed_budget"
                    and item["column_budget"] == 50
                ),
                None,
            )
            batch = next(
                (
                    item
                    for item in batches
                    if item["row_type"] == "summary"
                    and item["policy"] == row["policy"]
                    and item["horizon_step"] == row["horizon_step"]
                ),
                None,
            )
            output.append(
                {
                    **ANALYSIS_MARKERS,
                    "policy": row["policy"],
                    "horizon_step": row["horizon_step"],
                    "field": row["field"],
                    "indicator": row["indicator"],
                    "pr_auc": row["pr_auc"],
                    "positive_prevalence": row["positive_prevalence"],
                    "target_best_rank": row["target_candidate_best_rank"],
                    "top50_target_recall": (
                        budget["achieved_target_recall"] if budget else ""
                    ),
                    "mixed_batch_rate": (
                        batch["mixed_target_false"] if batch else ""
                    ),
                }
            )
    return output


def _bootstrap_tables(
    candidates: Sequence[Mapping[str, object]],
    columns: Sequence[Mapping[str, object]],
    *,
    samples: int,
    seed: int,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    policies = sorted({str(row["policy"]) for row in candidates})
    for policy in policies:
        for step in range(1, 6):
            for field in ("total", "passenger"):
                candidate_selected = _select_group(
                    candidates, policy, step, field
                )
                column_selected = _select_group(columns, policy, step, field)
                metrics: dict[
                    str,
                    tuple[
                        Sequence[Mapping[str, object]],
                        Callable[[Sequence[Mapping[str, object]]], float | None],
                    ],
                ] = {
                    "pr_auc": (
                        column_selected,
                        lambda rows: average_precision(
                            [bool(row["is_target_column"]) for row in rows],
                            [float(row["max_original_score"]) for row in rows],
                        ),
                    ),
                    "roc_auc": (
                        column_selected,
                        lambda rows: roc_auc(
                            [bool(row["is_target_column"]) for row in rows],
                            [float(row["max_original_score"]) for row in rows],
                        ),
                    ),
                    "top50_target_recall": (
                        column_selected,
                        lambda rows: fixed_budget_recall(
                            rows, "max_original_score", 50
                        ),
                    ),
                    "target_suppression_rate": (
                        column_selected,
                        _target_suppression,
                    ),
                    "false_retained_at_recall_0.5": (
                        column_selected,
                        lambda rows: _false_at_recall(
                            rows, "max_original_score", 0.5
                        ),
                    ),
                    "false_retained_at_recall_0.7": (
                        column_selected,
                        lambda rows: _false_at_recall(
                            rows, "max_original_score", 0.7
                        ),
                    ),
                }
                if policy == "batched":
                    metrics["target_is_top1_rate"] = (
                        candidate_selected,
                        _target_batch_top1_rate,
                    )
                for offset, (metric_name, (source, function)) in enumerate(
                    metrics.items()
                ):
                    estimates = bootstrap_rollout_metric_means(
                        source,
                        samples=samples,
                        seed=seed + step * 100 + offset,
                        metric=function,
                    )
                    output.append(
                        {
                            **ANALYSIS_MARKERS,
                            "policy": policy,
                            "horizon_step": step,
                            "field": field,
                            "metric": metric_name,
                            "bootstrap_unit": "input_index",
                            "bootstrap_samples_requested": samples,
                            "bootstrap_samples_valid": len(estimates),
                            "estimate_mean": _mean(estimates),
                            "ci95_low": _quantile(estimates, 0.025),
                            "ci95_high": _quantile(estimates, 0.975),
                        }
                    )
    return output


def _headline_metrics(
    separability: Sequence[Mapping[str, object]],
    budgets: Sequence[Mapping[str, object]],
    batches: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = [
        row
        for row in separability
        if row["level"] == "column"
        and row["field"] in {"total", "passenger"}
        and row["indicator"] in {"original_score", "effective_score"}
    ]
    return [
        {
            "policy": row["policy"],
            "horizon_step": row["horizon_step"],
            "field": row["field"],
            "indicator": row["indicator"],
            "pr_auc": row["pr_auc"],
            "positive_prevalence": row["positive_prevalence"],
            "roc_auc": row["roc_auc"],
            "target_best_rank": row["target_candidate_best_rank"],
            "false_columns_at_recall_0.5": _budget_value(
                budgets, row, 0.5
            ),
            "false_columns_at_recall_0.7": _budget_value(
                budgets, row, 0.7
            ),
        }
        for row in rows
    ]


def _analysis_groups(
    rows: Sequence[Mapping[str, object]],
) -> Iterable[tuple[tuple[str, int, str], list[Mapping[str, object]]]]:
    grouped: dict[tuple[str, int, str], list[Mapping[str, object]]] = defaultdict(
        list
    )
    for row in rows:
        key = (str(row["policy"]), int(row["horizon_step"]), str(row["field"]))
        grouped[key].append(row)
        grouped[(key[0], key[1], "total")].append(row)
    for key in sorted(grouped):
        yield key, grouped[key]


def _pool_signatures(
    rows: Sequence[Mapping[str, object]],
    step: int,
) -> dict[tuple[int, str], tuple[tuple[object, ...], ...]]:
    grouped: dict[tuple[int, str], list[tuple[object, ...]]] = defaultdict(list)
    for row in rows:
        if int(row["horizon_step"]) != step:
            continue
        grouped[
            (int(row["input_index"]), str(row["target_timestamp"]))
        ].append(
            (
                int(row["candidate_original_index"]),
                str(row["field"]),
                int(row["column"]),
                int(row["neuron"]),
                float(row["predicted_time"]),
                float(row["original_score"]),
            )
        )
    return {
        key: tuple(sorted(values))
        for key, values in grouped.items()
    }


def _select_group(
    rows: Sequence[Mapping[str, object]],
    policy: str,
    step: int,
    field: str,
) -> list[Mapping[str, object]]:
    return [
        row
        for row in rows
        if row["policy"] == policy
        and int(row["horizon_step"]) == step
        and (field == "total" or row["field"] == field)
    ]


def _ranking_metrics(
    rows: Sequence[Mapping[str, object]],
    labels: Sequence[bool],
    scores: Sequence[float],
) -> dict[str, float]:
    grouped: dict[int, list[tuple[Mapping[str, object], bool, float]]] = defaultdict(
        list
    )
    for row, label, score in zip(rows, labels, scores):
        grouped[int(row["input_index"])].append((row, label, score))
    mean_ranks: list[float] = []
    best_ranks: list[float] = []
    reciprocal: list[float] = []
    hits = {budget: [] for budget in (1, 5, 10, 20, 50)}
    for group in grouped.values():
        ordered = sorted(
            group,
            key=lambda item: (
                -item[2],
                int(item[0]["column"]),
                int(item[0].get("neuron", item[0].get("selected_neuron", 0))),
            ),
        )
        ranks = [
            rank
            for rank, (_, label, _) in enumerate(ordered, start=1)
            if label
        ]
        if not ranks:
            continue
        mean_ranks.append(_mean(ranks))
        best_ranks.append(min(ranks))
        reciprocal.append(1.0 / min(ranks))
        for budget in hits:
            hits[budget].append(float(min(ranks) <= budget))
    return {
        "target_candidate_mean_rank": _mean(mean_ranks),
        "target_candidate_best_rank": _mean(best_ranks),
        "reciprocal_rank": _mean(reciprocal),
        **{
            f"hit_at_{budget}": _mean(values)
            for budget, values in hits.items()
        },
    }


def _validate_trace_columns(
    rows: Sequence[Mapping[str, object]],
    ranges: Mapping[str, tuple[int, int]],
) -> None:
    for row in rows:
        field = str(row["field"])
        if field not in ranges or field == "total":
            raise ValueError(f"unexpected candidate field: {field}")
        column = int(row["column"])
        start, stop = ranges[field]
        if not start <= column < stop:
            raise ValueError(
                f"candidate column {column} does not belong to {field} "
                f"range [{start}, {stop})"
            )


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average = (cursor + 1 + end) / 2
        for index, _ in ordered[cursor:end]:
            ranks[index] = average
        cursor = end
    return ranks


def _percentile_ranks(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    ranks = _average_ranks(values)
    return [
        (rank - 1) / (len(values) - 1) if len(values) > 1 else 0.5
        for rank in ranks
    ]


def _z_scores(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    mean = _mean(values)
    std = math.sqrt(_variance(values))
    return [(value - mean) / std if std else 0.0 for value in values]


def _prefixed_stats(prefix: str, values: Sequence[float]) -> dict[str, object]:
    return {
        f"{prefix}_mean": _mean(values) if values else "",
        f"{prefix}_std": math.sqrt(_variance(values)) if values else "",
        f"{prefix}_median": _quantile(values, 0.50),
        f"{prefix}_p10": _quantile(values, 0.10),
        f"{prefix}_p25": _quantile(values, 0.25),
        f"{prefix}_p75": _quantile(values, 0.75),
        f"{prefix}_p90": _quantile(values, 0.90),
        f"{prefix}_min": min(values) if values else "",
        f"{prefix}_max": max(values) if values else "",
    }


def _standardized_difference(
    positive: Sequence[float],
    negative: Sequence[float],
) -> float | str:
    if not positive or not negative:
        return ""
    pooled = math.sqrt((_variance(positive) + _variance(negative)) / 2)
    return (_mean(positive) - _mean(negative)) / pooled if pooled else 0.0


def _overlap(
    positive: Sequence[float],
    negative: Sequence[float],
    bins: int = 32,
) -> float | str:
    if not positive or not negative:
        return ""
    minimum = min(min(positive), min(negative))
    maximum = max(max(positive), max(negative))
    if minimum == maximum:
        return 1.0
    width = (maximum - minimum) / bins
    positive_hist = [0] * bins
    negative_hist = [0] * bins
    for values, histogram in (
        (positive, positive_hist),
        (negative, negative_hist),
    ):
        for value in values:
            index = min(int((value - minimum) / width), bins - 1)
            histogram[index] += 1
    return sum(
        min(left / len(positive), right / len(negative))
        for left, right in zip(positive_hist, negative_hist)
    )


def _target_suppression(rows: Sequence[Mapping[str, object]]) -> float:
    target = [row for row in rows if bool(row["is_target_column"])]
    return 1.0 - _ratio(
        sum(bool(row["any_emitted"]) for row in target),
        len(target),
    )


def _false_at_recall(
    rows: Sequence[Mapping[str, object]],
    score_field: str,
    recall: float,
) -> float | None:
    feasible = [
        row
        for row in threshold_sweep(rows, score_field)
        if float(row["target_recall"]) >= recall
    ]
    if not feasible:
        return None
    return float(
        min(
            feasible,
            key=lambda row: (
                int(row["false_columns_retained"]),
                int(row["total_columns_retained"]),
            ),
        )["false_columns_retained"]
    )


def _target_batch_top1_rate(
    rows: Sequence[Mapping[str, object]],
) -> float:
    grouped: dict[tuple[int, int], list[Mapping[str, object]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[(int(row["input_index"]), int(row["batch_index"]))].append(row)
    mixed = [
        group
        for group in grouped.values()
        if any(bool(row["is_target_candidate"]) for row in group)
        and any(not bool(row["is_target_candidate"]) for row in group)
    ]
    if not mixed:
        return 0.0
    return _ratio(
        sum(
            bool(
                max(
                    group,
                    key=lambda row: (
                        float(row["original_score"]),
                        -int(row["column"]),
                    ),
                )["is_target_candidate"]
            )
            for group in mixed
        ),
        len(mixed),
    )


def _budget_value(
    budgets: Sequence[Mapping[str, object]],
    metric: Mapping[str, object],
    recall: float,
) -> object:
    row = next(
        (
            item
            for item in budgets
            if item["policy"] == metric["policy"]
            and item["horizon_step"] == metric["horizon_step"]
            and item["field"] == metric["field"]
            and item["indicator"] == metric["indicator"]
            and item["summary_type"] == "target_recall"
            and item["target_recall_requirement"] == recall
        ),
        None,
    )
    return row["false_columns_retained"] if row else ""


def _render_report(summary: Mapping[str, object]) -> str:
    lines = [
        "# Candidate Score Separability Report",
        "",
        "This report is offline and diagnostic only. Ground truth labels do not "
        "affect prediction, competition, decoding, or learning.",
        "",
        "## Inputs",
        "",
    ]
    for item in summary["inputs"]:  # type: ignore[index]
        lines.extend(
            [
                f"- Policy: `{item['policy']}`",  # type: ignore[index]
                f"- Directory: `{item['input_directory']}`",  # type: ignore[index]
                f"- Candidate trace SHA256: `{item['candidate_trace_sha256']}`",  # type: ignore[index]
                f"- Competition trace SHA256: `{item['competition_trace_sha256']}`",  # type: ignore[index]
                f"- Oracle trace SHA256: `{item['oracle_trace_sha256']}`",  # type: ignore[index]
                f"- Predictions SHA256: `{item['predictions_sha256']}`",  # type: ignore[index]
                f"- Protocol SHA256: `{item['protocol_sha256']}`",  # type: ignore[index]
                "",
            ]
        )
    lines.extend(
        [
            "## Method",
            "",
            f"- Candidate rows: {summary['candidate_rows']}",
            f"- Column rows: {summary['column_rows']}",
            f"- Column aggregation: {summary['column_aggregation_rule']}",
            f"- Bootstrap unit: {summary['bootstrap_unit']}",
            f"- Bootstrap samples: {summary['bootstrap_samples']}",
            "- Step 1 may be compared as a fixed raw candidate pool. Steps 2-5 "
            "are policy-level comparisons after recurrent trajectory divergence.",
            "- Candidate-to-segment provenance is unavailable.",
            "",
            "## Headline Metrics",
            "",
            "| Policy | Step | Field | Indicator | PR-AUC | Prevalence | ROC-AUC | False @ R=0.5 | False @ R=0.7 |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["headline_metrics"]:  # type: ignore[index]
        lines.append(
            f"| {row['policy']} | {row['horizon_step']} | "
            f"{row['field']} | {row['indicator']} | "
            f"{row['pr_auc']} | {row['positive_prevalence']} | "
            f"{row['roc_auc']} | "
            f"{row['false_columns_at_recall_0.5']} | "
            f"{row['false_columns_at_recall_0.7']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- PR-AUC and fixed-budget recall are primary because target columns "
            "are rare; ROC-AUC is supplementary.",
            "- No threshold in this report is applied to the model.",
            "- Field normalization is offline only.",
            "- Formal conclusions require candidate traces generated from the "
            "corresponding 250-record runs.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _common_manifest_value(
    rows: Sequence[Mapping[str, object]],
    field: str,
) -> object:
    values = {row.get(field, "unavailable") for row in rows}
    return next(iter(values)) if len(values) == 1 else "mixed"


def _boolean(value: object) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _quantile(values: Sequence[float], quantile: float) -> float | str:
    if not values:
        return ""
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _rate(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return _ratio(sum(bool(row[field]) for row in rows), len(rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline Fig.9 candidate score separability analysis."
    )
    parser.add_argument("--sequential-dir", default="")
    parser.add_argument("--batched-dir", default="")
    parser.add_argument("--competition-trace", default="")
    parser.add_argument("--oracle-trace", default="")
    parser.add_argument("--predictions", default="")
    parser.add_argument("--protocol", default="")
    parser.add_argument("--candidate-trace", default="")
    parser.add_argument("--policy-label", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_inputs: list[tuple[str, Path]] = []
    if args.sequential_dir:
        policy_inputs.append(("sequential", Path(args.sequential_dir)))
    if args.batched_dir:
        policy_inputs.append(("batched", Path(args.batched_dir)))
    if args.candidate_trace:
        if not args.policy_label or not args.protocol or not args.predictions:
            raise ValueError(
                "single-trace mode requires --policy-label, --protocol, "
                "and --predictions"
            )
        candidate_path = Path(args.candidate_trace).resolve()
        directory = candidate_path.parent
        expected = {
            "candidate trace": directory / "candidate_separability_trace.csv",
            "protocol": directory / "original_protocol.json",
            "predictions": directory / "original_predictions.csv",
        }
        supplied = {
            "candidate trace": candidate_path,
            "protocol": Path(args.protocol).resolve(),
            "predictions": Path(args.predictions).resolve(),
        }
        for label, expected_path in expected.items():
            if supplied[label] != expected_path.resolve():
                raise ValueError(
                    f"{label} must use the standard output filename "
                    f"{expected_path.name} in one result directory"
                )
        for optional in (args.competition_trace, args.oracle_trace):
            if optional and not Path(optional).is_file():
                raise ValueError(f"input file does not exist: {optional}")
        policy_inputs.append((args.policy_label, directory))
    if not policy_inputs:
        raise ValueError("provide --sequential-dir and/or --batched-dir")
    summary = analyze(
        policy_inputs,
        output_dir=Path(args.output_dir),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir).resolve()),
                "candidate_rows": summary["candidate_rows"],
                "column_rows": summary["column_rows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
