"""Offline analysis for Fig.9 branch provenance traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from experiments.diagnostics.analyze_fig9_candidate_score_separability import (
    average_precision,
    roc_auc,
)
from experiments.diagnostics.fig9_branch_provenance import (
    DIAGNOSTIC_MARKERS,
)


INDICATORS: dict[str, Callable[[Mapping[str, object]], float]] = {
    "original_score": lambda row: float(row["original_score"]),
    "effective_score": lambda row: _float(row["effective_score"]),
    "dominant_branch_share": lambda row: float(
        row["dominant_branch_share"]
    ),
    "negative_normalized_branch_entropy": lambda row: -float(
        row["normalized_branch_entropy"]
    ),
    "contribution_hhi": lambda row: float(row["contribution_hhi"]),
    "current_context_jaccard": lambda row: float(
        row["current_context_jaccard"]
    ),
    "creation_current_source_jaccard": lambda row: _float(
        row.get("creation_current_source_jaccard")
    ),
    "score_x_dominant_branch_share": lambda row: float(
        row["original_score"]
    )
    * float(row["dominant_branch_share"]),
    "effective_x_dominant_branch_share": lambda row: _float(
        row["effective_score"]
    )
    * float(row["dominant_branch_share"]),
    "score_x_contribution_hhi": lambda row: float(row["original_score"])
    * float(row["contribution_hhi"]),
    "score_x_current_context_jaccard": lambda row: float(
        row["original_score"]
    )
    * float(row["current_context_jaccard"]),
}

COMPARISON_FIELDS = (
    "unique_branch_count",
    "dominant_branch_share",
    "normalized_branch_entropy",
    "contribution_hhi",
    "mean_source_jaccard",
    "current_context_jaccard",
    "creation_current_source_jaccard",
    "creation_source_retention",
    "source_set_growth_count",
    "predicted_supported_contribution",
    "burst_supported_contribution",
    "candidate_segment_count",
    "dominant_branch_contribution",
    "original_score",
    "effective_score",
)


def _float(value: object) -> float:
    if value in {"", None}:
        return 0.0
    return float(value)


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes"}


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": _mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values) if values else 0.0,
        "p10": _percentile(values, 0.10),
        "p25": _percentile(values, 0.25),
        "p75": _percentile(values, 0.75),
        "p90": _percentile(values, 0.90),
    }


def _standardized_difference(
    positive: Sequence[float],
    negative: Sequence[float],
) -> float:
    variance = (
        (statistics.pvariance(positive) if len(positive) > 1 else 0.0)
        + (statistics.pvariance(negative) if len(negative) > 1 else 0.0)
    ) / 2.0
    return (
        (_mean(positive) - _mean(negative)) / math.sqrt(variance)
        if variance > 0.0
        else 0.0
    )


def _read_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _groups(
    rows: Sequence[Mapping[str, object]],
) -> Iterable[tuple[tuple[str, int, str], list[Mapping[str, object]]]]:
    grouped: dict[
        tuple[str, int, str],
        list[Mapping[str, object]],
    ] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["policy"]),
                int(row["horizon_step"]),
                str(row["field"]),
            )
        ].append(row)
    return sorted(grouped.items())


def _metric_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for (policy, step, field), selected in _groups(rows):
        labels = [_bool(row["is_target_candidate"]) for row in selected]
        prevalence = sum(labels) / len(labels) if labels else 0.0
        for name, getter in INDICATORS.items():
            scores = [getter(row) for row in selected]
            output.append(
                {
                    **DIAGNOSTIC_MARKERS,
                    "oracle_evaluation_only": True,
                    "not_used_for_prediction": True,
                    "policy": policy,
                    "horizon_step": step,
                    "field": field,
                    "indicator": name,
                    "sample_count": len(selected),
                    "positive_prevalence": prevalence,
                    "roc_auc": roc_auc(labels, scores),
                    "pr_auc": average_precision(labels, scores),
                }
            )
    return output


def _comparison_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for (policy, step, field), selected in _groups(rows):
        positive_rows = [
            row for row in selected if _bool(row["is_target_candidate"])
        ]
        negative_rows = [
            row for row in selected if not _bool(row["is_target_candidate"])
        ]
        for name in COMPARISON_FIELDS:
            positive = [_float(row.get(name)) for row in positive_rows]
            negative = [_float(row.get(name)) for row in negative_rows]
            output.append(
                {
                    **DIAGNOSTIC_MARKERS,
                    "policy": policy,
                    "horizon_step": step,
                    "field": field,
                    "metric": name,
                    **{
                        f"target_{key}": value
                        for key, value in _distribution(positive).items()
                    },
                    **{
                        f"false_{key}": value
                        for key, value in _distribution(negative).items()
                    },
                    "target_mean_minus_false_mean": (
                        _mean(positive) - _mean(negative)
                    ),
                    "standardized_mean_difference": (
                        _standardized_difference(positive, negative)
                    ),
                }
            )
    return output


def _threshold_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for (policy, step, field), selected in _groups(rows):
        for indicator in (
            "dominant_branch_share",
            "current_context_jaccard",
            "creation_current_source_jaccard",
            "score_x_current_context_jaccard",
        ):
            getter = INDICATORS[indicator]
            ordered = sorted(
                selected,
                key=getter,
                reverse=True,
            )
            positives = sum(
                _bool(row["is_target_candidate"]) for row in selected
            )
            true_count = 0
            false_count = 0
            cursor = 0
            while cursor < len(ordered):
                threshold = getter(ordered[cursor])
                end = cursor
                while end < len(ordered) and getter(ordered[end]) == threshold:
                    if _bool(ordered[end]["is_target_candidate"]):
                        true_count += 1
                    else:
                        false_count += 1
                    end += 1
                output.append(
                    {
                        **DIAGNOSTIC_MARKERS,
                        "policy": policy,
                        "horizon_step": step,
                        "field": field,
                        "indicator": indicator,
                        "threshold": threshold,
                        "target_recall": (
                            true_count / positives if positives else 0.0
                        ),
                        "true_candidates_retained": true_count,
                        "false_candidates_retained": false_count,
                        "total_candidates_retained": (
                            true_count + false_count
                        ),
                    }
                )
                cursor = end
    return output


def _chimera_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for (policy, step, field), selected in _groups(rows):
        score_cutoff = _percentile(
            [_float(row["original_score"]) for row in selected],
            0.80,
        )
        counts: dict[tuple[bool, str], int] = defaultdict(int)
        for row in selected:
            target = _bool(row["is_target_candidate"])
            if not _bool(row["provenance_available"]):
                label = "PROVENANCE_UNAVAILABLE"
            else:
                share = _float(row["dominant_branch_share"])
                if (
                    _float(row["original_score"]) >= score_cutoff
                    and share < 0.5
                ):
                    label = "HIGH_SCORE_LOW_COHERENCE"
                elif share >= 0.8:
                    label = "COHERENT_SINGLE_BRANCH"
                elif share >= 0.5:
                    label = "MOSTLY_COHERENT"
                else:
                    label = "MIXED_BRANCH"
            counts[(target, label)] += 1
        for (target, label), count in sorted(counts.items()):
            output.append(
                {
                    **DIAGNOSTIC_MARKERS,
                    "policy": policy,
                    "horizon_step": step,
                    "field": field,
                    "is_target_candidate": target,
                    "chimera_class": label,
                    "count": count,
                    "rate": count
                    / sum(
                        amount
                        for (is_target, _label), amount in counts.items()
                        if is_target == target
                    ),
                }
            )
    return output


def _bootstrap_ci(
    rows: Sequence[Mapping[str, object]],
    *,
    samples: int,
    seed: int,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    rng = random.Random(seed)
    for (policy, step, field), selected in _groups(rows):
        if field != "passenger":
            continue
        by_rollout: dict[int, list[Mapping[str, object]]] = defaultdict(list)
        for row in selected:
            by_rollout[int(row["input_index"])].append(row)
        rollout_ids = sorted(by_rollout)
        if not rollout_ids:
            continue
        for indicator in (
            "original_score",
            "dominant_branch_share",
            "current_context_jaccard",
            "creation_current_source_jaccard",
            "score_x_current_context_jaccard",
        ):
            values: list[float] = []
            getter = INDICATORS[indicator]
            for _ in range(samples):
                sampled: list[Mapping[str, object]] = []
                for _rollout in rollout_ids:
                    sampled.extend(
                        by_rollout[rng.choice(rollout_ids)]
                    )
                metric = average_precision(
                    [
                        _bool(row["is_target_candidate"])
                        for row in sampled
                    ],
                    [getter(row) for row in sampled],
                )
                if metric is not None:
                    values.append(metric)
            output.append(
                {
                    **DIAGNOSTIC_MARKERS,
                    "bootstrap_unit": "rollout input_index",
                    "bootstrap_samples": samples,
                    "bootstrap_seed": seed,
                    "policy": policy,
                    "horizon_step": step,
                    "field": field,
                    "indicator": indicator,
                    "metric": "pr_auc",
                    "mean": _mean(values),
                    "ci_low": _percentile(values, 0.025),
                    "ci_high": _percentile(values, 0.975),
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
    candidate_path = run_dir / "branch_candidate_trace.csv"
    rows = _read_csv(candidate_path)
    if not rows:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "branch_candidate_summary.csv",
            "branch_field_summary.csv",
            "branch_horizon_summary.csv",
            "branch_target_vs_false.csv",
            "branch_separability_metrics.csv",
            "branch_threshold_curves.csv",
            "branch_chimera_summary.csv",
            "branch_bootstrap_ci.csv",
        ):
            (output_dir / name).write_text("", encoding="utf-8")
        summary = {
            **DIAGNOSTIC_MARKERS,
            "policy": policy_label,
            "run_dir": str(run_dir.resolve()),
            "candidate_rows": 0,
            "provenance_available_rate": 0.0,
            "empty_trace": True,
            "multi_segment_candidate_chimera_testable": False,
            "source_context_mixing_testable": False,
        }
        (output_dir / "branch_provenance_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (output_dir / "FIG9_BRANCH_PROVENANCE_REPORT.md").write_text(
            "# Fig.9 Branch Provenance Diagnostic\n\n"
            "No candidate rows were available; no branch conclusion was made.\n",
            encoding="utf-8",
        )
        return summary
    for row in rows:
        row["policy"] = policy_label or row.get("policy", "")
    comparison = _comparison_rows(rows)
    metrics = _metric_rows(rows)
    thresholds = _threshold_rows(rows)
    chimera = _chimera_rows(rows)
    bootstrap = _bootstrap_ci(
        rows,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    candidate_summary = [
        {
            **DIAGNOSTIC_MARKERS,
            "policy": policy,
            "horizon_step": step,
            "field": field,
            "candidate_count": len(selected),
            "target_count": sum(
                _bool(row["is_target_candidate"]) for row in selected
            ),
            "provenance_available_rate": _mean(
                [
                    float(_bool(row["provenance_available"]))
                    for row in selected
                ]
            ),
            "mean_segments_per_candidate": _mean(
                [_float(row["candidate_segment_count"]) for row in selected]
            ),
            "mean_current_context_jaccard": _mean(
                [_float(row["current_context_jaccard"]) for row in selected]
            ),
        }
        for (policy, step, field), selected in _groups(rows)
    ]
    field_summary = [
        row for row in candidate_summary
    ]
    horizon_summary = [
        {
            **DIAGNOSTIC_MARKERS,
            "policy": policy,
            "horizon_step": step,
            "candidate_count": len(selected),
            "provenance_available_rate": _mean(
                [
                    float(_bool(row["provenance_available"]))
                    for row in selected
                ]
            ),
            "mean_current_context_jaccard": _mean(
                [_float(row["current_context_jaccard"]) for row in selected]
            ),
        }
        for (policy, step, _field), selected in _groups(
            [
                {**row, "field": "total"}
                for row in rows
            ]
        )
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "branch_candidate_summary.csv", candidate_summary)
    _write_csv(output_dir / "branch_field_summary.csv", field_summary)
    _write_csv(output_dir / "branch_horizon_summary.csv", horizon_summary)
    _write_csv(output_dir / "branch_target_vs_false.csv", comparison)
    _write_csv(output_dir / "branch_separability_metrics.csv", metrics)
    _write_csv(output_dir / "branch_threshold_curves.csv", thresholds)
    _write_csv(output_dir / "branch_chimera_summary.csv", chimera)
    _write_csv(output_dir / "branch_bootstrap_ci.csv", bootstrap)
    summary = {
        **DIAGNOSTIC_MARKERS,
        "policy": policy_label,
        "run_dir": str(run_dir.resolve()),
        "candidate_rows": len(rows),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_unit": "rollout input_index",
        "candidate_segment_semantics": (
            "one PredictionCandidate references exactly one winning Segment"
        ),
        "multi_segment_candidate_chimera_testable": False,
        "source_context_mixing_testable": True,
        "provenance_available_rate": _mean(
            [
                float(_bool(row["provenance_available"]))
                for row in rows
            ]
        ),
    }
    (output_dir / "branch_provenance_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report = [
        "# Fig.9 Branch Provenance Diagnostic",
        "",
        "This is a nonpaper, offline-only diagnostic. Ground truth labels",
        "candidates after prediction and never changes prediction or learning.",
        "",
        "## Structural Finding",
        "",
        "A saved `PredictionCandidate` contains one winning dendritic segment.",
        "Its score is not a sum across multiple segments. Consequently, exact",
        "creation-branch count, entropy, and HHI are structurally 1/0/1 when",
        "provenance is available. A multi-segment candidate chimera cannot be",
        "tested in this implementation. Source-context mixing inside that one",
        "segment remains testable through creation/current source fingerprints",
        "and context Jaccard.",
        "",
        "## Run",
        "",
        f"- Policy: `{policy_label}`",
        f"- Candidate rows: {len(rows)}",
        (
            "- Provenance available rate: "
            f"{summary['provenance_available_rate']:.6f}"
        ),
        "- Bootstrap unit: rollout `input_index`",
        "",
        "See the CSV outputs for field, horizon, target/false, separability,",
        "threshold, chimera, and bootstrap statistics.",
    ]
    (output_dir / "FIG9_BRANCH_PROVENANCE_REPORT.md").write_text(
        "\n".join(report) + "\n",
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
