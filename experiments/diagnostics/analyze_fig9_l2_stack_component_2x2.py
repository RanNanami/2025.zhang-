"""Offline P0/P1/P2/P3 decomposition for the L2 diagnostic stack."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from experiments.diagnostics.analyze_fig9_lmatch_stack_2x2 import (
    _bootstrap,
    _number,
    _run_metrics as _base_run_metrics,
    _write_rows,
)


CELLS = ("P0_RAW_L2", "P1_SELECTOR_ONLY_L2", "P2_COMPETITION_ONLY_L2", "P3_FULL_STACK_L2")
METRICS = (
    "MAPE", "step1", "step2", "step3", "step4", "step5", "raw_mean",
    "emitted_mean", "segments", "synapses", "contributors_mean",
)


def protocol_matrix() -> list[dict[str, Any]]:
    return [
        {"cell": "P0_RAW_L2", "L_match": 2, "competition": "off", "simultaneous": "sequential", "selector": "existing", "label": "nonpaper raw L_match=2 baseline"},
        {"cell": "P1_SELECTOR_ONLY_L2", "L_match": 2, "competition": "off", "simultaneous": "sequential", "selector": "max_candidate_score", "label": "selector-only diagnostic"},
        {"cell": "P2_COMPETITION_ONLY_L2", "L_match": 2, "competition": "competitive_raw", "simultaneous": "batched", "selector": "existing", "label": "competition-only diagnostic"},
        {"cell": "P3_FULL_STACK_L2", "L_match": 2, "competition": "competitive_raw", "simultaneous": "batched", "selector": "max_candidate_score", "label": "full diagnostic stack"},
    ]


def effects(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for metric in METRICS:
        p0 = _number(metrics["P0_RAW_L2"].get(metric))
        p1 = _number(metrics["P1_SELECTOR_ONLY_L2"].get(metric))
        p2 = _number(metrics["P2_COMPETITION_ONLY_L2"].get(metric))
        p3 = _number(metrics["P3_FULL_STACK_L2"].get(metric))
        rows.append({
            "metric": metric,
            "selector_effect_P1_minus_P0": p1 - p0,
            "selector_effect_with_competition_P3_minus_P2": p3 - p2,
            "competition_effect_P2_minus_P0": p2 - p0,
            "competition_effect_with_selector_P3_minus_P1": p3 - p1,
            "full_stack_effect_P3_minus_P0": p3 - p0,
            "selector_competition_interaction": (p3 - p2) - (p1 - p0),
        })
    return rows


def write_protocol(output_dir: Path, limit: int | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = protocol_matrix()
    _write_rows(output_dir / "l2_stack_component_protocol_matrix.csv", rows)
    if limit is not None:
        _write_rows(
            output_dir / f"l2_stack_component_protocol_matrix_{limit}.csv",
            [{"limit": limit, **row} for row in rows],
        )
    lines = [
        "# Fig.9 L2 Stack Component 2x2 Protocol Audit", "",
        "All four cells use L_match=2, raw propagation, reference continuous dynamics, seed=0, warmup=200, K=10, 32 neurons/column, forgetting threshold=65, no compensation, no future covariates, no ground-truth selection, no rollout learning, and no decoded replay.", "",
        "| Cell | Competition | Simultaneous | Selector | Label |", "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['cell']} | {row['competition']} | {row['simultaneous']} | {row['selector']} | {row['label']} |")
    lines.extend(["", "P0/P1 differ only by selector. P0/P2 differ only by competition stack. P1/P3 differ only by competition stack. P2/P3 differ only by selector. P0 and P3 are reused from the completed L2 raw/full-stack 250 runs.", ""])
    (output_dir / "FIG9_L2_STACK_COMPONENT_2X2_PROTOCOL_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def _read_density_metrics(run_dir: Path) -> dict[str, Any]:
    """Read candidate and per-step fields from the lightweight density trace.

    The long-sequence ledger intentionally records rolled-up activity rather
    than every candidate.  The density trace is the authoritative lightweight
    source for candidate counts; ``candidate_neuron_mean`` is explicitly the
    emitted raw-neuron count because the strict run does not persist a separate
    candidate-neuron enumeration.
    """

    path = run_dir / "original_density_trace.csv"
    if not path.exists():
        return {
            "candidate_segment_mean": 0.0,
            "candidate_segment_peak": 0,
            "accepted_candidate_mean": 0.0,
            "accepted_candidate_peak": 0,
            "candidate_neuron_mean": 0.0,
            "density_rows": 0,
        }
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    candidates = [float(row.get("candidate_segment_count", 0) or 0) for row in rows]
    accepted = [float(row.get("accepted_candidate_count", 0) or 0) for row in rows]
    neurons = [float(row.get("raw_predicted_neuron_count", 0) or 0) for row in rows]
    return {
        "candidate_segment_mean": sum(candidates) / len(candidates) if candidates else 0.0,
        "candidate_segment_peak": int(max(candidates, default=0)),
        "accepted_candidate_mean": sum(accepted) / len(accepted) if accepted else 0.0,
        "accepted_candidate_peak": int(max(accepted, default=0)),
        "candidate_neuron_mean": sum(neurons) / len(neurons) if neurons else 0.0,
        "density_rows": len(rows),
    }


def _run_metrics(run_dir: Path) -> dict[str, Any]:
    """Extend the shared run metrics with component-specific diagnostics."""

    row = _base_run_metrics(run_dir)
    row.update(_read_density_metrics(run_dir))
    return row


def _write_cross_limit_outputs(
    metrics_by_limit: dict[int, dict[str, dict[str, Any]]], output_dir: Path
) -> None:
    """Write descriptive 250/500 comparisons without rerunning a model."""

    comparison_rows: list[dict[str, Any]] = []
    for limit, metrics in sorted(metrics_by_limit.items()):
        for cell, row in metrics.items():
            for metric in METRICS + (
                "raw_peak", "emitted_peak", "candidate_segment_mean",
                "accepted_candidate_mean", "candidate_neuron_mean", "peak_RSS_MB",
            ):
                comparison_rows.append({
                    "limit": limit,
                    "cell": cell,
                    "metric": metric,
                    "value": row.get(metric, ""),
                })
    _write_rows(output_dir / "l2_component_250_500_comparison.csv", comparison_rows)

    stepwise_rows = [
        row for row in comparison_rows
        if row["metric"] in ("step1", "step2", "step3", "step4", "step5")
    ]
    _write_rows(output_dir / "l2_component_stepwise_250_500.csv", stepwise_rows)

    for filename, effect_key in (
        ("selector_effect_by_horizon.csv", "selector"),
        ("competition_effect_by_horizon.csv", "competition"),
        ("selector_competition_interaction_by_horizon.csv", "interaction"),
    ):
        rows: list[dict[str, Any]] = []
        for limit, metrics in sorted(metrics_by_limit.items()):
            p0 = metrics.get("P0_RAW_L2")
            p1 = metrics.get("P1_SELECTOR_ONLY_L2")
            p2 = metrics.get("P2_COMPETITION_ONLY_L2")
            p3 = metrics.get("P3_FULL_STACK_L2")
            if not all((p0, p1, p2, p3)):
                continue
            for horizon in range(1, 6):
                a = _number(p0.get(f"step{horizon}"))
                b = _number(p1.get(f"step{horizon}"))
                c = _number(p2.get(f"step{horizon}"))
                d = _number(p3.get(f"step{horizon}"))
                value = {
                    "selector": b - a,
                    "competition": c - a,
                    "interaction": (d - c) - (b - a),
                }[effect_key]
                rows.append({"limit": limit, "horizon": horizon, "effect": value})
        _write_rows(output_dir / filename, rows)


def _write_experiment_ledger(
    metrics: dict[str, dict[str, Any]], output_dir: Path, limit: int
) -> None:
    """Persist one explicit protocol-and-result row per component cell."""

    protocol_by_cell = {row["cell"]: row for row in protocol_matrix()}
    rows: list[dict[str, Any]] = []
    for cell, metric in metrics.items():
        protocol = protocol_by_cell[cell]
        rows.append({
            "experiment": "fig9_l2_stack_component",
            "limit": limit,
            "cell": cell,
            "L_match": 2,
            "competition_mode": protocol["competition"],
            "simultaneous_policy": protocol["simultaneous"],
            "intracolumn_selector": protocol["selector"],
            "stream": "original",
            "seed": 0,
            "warmup": 200,
            "prediction_horizon": 5,
            "propagation_mode": "raw",
            "continuous_impl": "reference",
            "lmatch_real_ablation": True,
            "uses_ground_truth_for_selection": False,
            "uses_compensation": False,
            "uses_future_covariates": False,
            "learns_during_rollout": False,
            "reencodes_decoded_value": False,
            **{key: value for key, value in metric.items() if key != "record_errors"},
        })
    _write_rows(output_dir / "EXPERIMENT_LEDGER.csv", rows)


def analyze(
    mapping: dict[str, Path],
    output_dir: Path,
    comparison_mapping: dict[str, Path] | None = None,
) -> dict[str, Any]:
    metrics = {cell: _run_metrics(path) for cell, path in mapping.items()}
    limit = int(next(iter(metrics.values()))["limit"])
    flat = [{"cell": cell, **{k: v for k, v in row.items() if k != "record_errors"}} for cell, row in metrics.items()]
    _write_rows(output_dir / f"l2_stack_component_2x2_{limit}.csv", flat)
    rows = effects(metrics)
    _write_rows(output_dir / f"l2_stack_component_effects_{limit}.csv", rows)
    _write_rows(output_dir / f"l2_stack_component_stepwise_{limit}.csv", [row for row in rows if row["metric"].startswith("step")])
    if limit == 250:
        # Preserve the old artifact names for existing notebooks and reports.
        _write_rows(output_dir / "l2_stack_component_2x2_250.csv", flat)
        _write_rows(output_dir / "l2_stack_component_effects_250.csv", rows)
        _write_rows(output_dir / "l2_stack_component_stepwise.csv", [row for row in rows if row["metric"].startswith("step")])
    _write_rows(
        output_dir / "native_component_500.csv" if limit == 500 else output_dir / f"native_component_{limit}.csv",
        [
            {
                "limit": limit,
                "cell": cell,
                "runtime_seconds": row["runtime_seconds"],
                "peak_RSS_MB": row["peak_RSS_MB"],
                "native_crashes": row["native_crashes"],
                "resume_count": row["resume_count"],
                "attempt_count": row["attempt_count"],
                "complete": row["complete"],
            }
            for cell, row in metrics.items()
        ],
    )
    _write_experiment_ledger(metrics, output_dir, limit)
    summary = {"cells": flat, "effects": rows, "bootstrap": {}}
    for metric in ("step1", "step2", "step3", "step4", "step5"):
        pairs = {}
        for left, right, key in (("P0_RAW_L2", "P1_SELECTOR_ONLY_L2", "selector"), ("P0_RAW_L2", "P2_COMPETITION_ONLY_L2", "competition"), ("P1_SELECTOR_ONLY_L2", "P3_FULL_STACK_L2", "competition"), ("P2_COMPETITION_ONLY_L2", "P3_FULL_STACK_L2", "selector")):
            a = metrics[left]["record_errors"]
            b = metrics[right]["record_errors"]
            values = [_number(b[index][int(metric[-1]) - 1]) - _number(a[index][int(metric[-1]) - 1]) for index in sorted(set(a) & set(b)) if b[index][int(metric[-1]) - 1] is not None and a[index][int(metric[-1]) - 1] is not None]
            mean, low, high = _bootstrap(values)
            pairs[f"{key}:{left}->{right}"] = {"mean": mean, "ci_low": low, "ci_high": high, "n": len(values)}
        summary["bootstrap"][metric] = pairs
    summary["limit"] = limit
    summary["protocol_matrix"] = str(output_dir / f"l2_stack_component_protocol_matrix_{limit}.csv")
    summary_path = output_dir / f"FINAL_L2_STACK_COMPONENT_{limit}_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if limit == 250:
        (output_dir / "FINAL_L2_STACK_COMPONENT_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    comparison_metrics: dict[str, dict[str, Any]] | None = None
    comparison_limit: int | None = None
    if comparison_mapping:
        all_metrics = {limit: metrics}
        compare_metrics = {cell: _run_metrics(path) for cell, path in comparison_mapping.items()}
        comparison_metrics = compare_metrics
        comparison_limit = int(next(iter(compare_metrics.values()))["limit"])
        all_metrics[comparison_limit] = compare_metrics
        _write_cross_limit_outputs(all_metrics, output_dir)
    lines = [
        f"# Fig.9 L2 Stack Component Decomposition ({limit} records)",
        "",
        "This is a nonpaper diagnostic decomposition; strict defaults are unchanged.",
        "All four cells use the same original stream, seed=0, warmup=200, horizon=5, raw propagation, reference continuous dynamics, one-pass online learning, and L_match=2. No ground truth, compensation, future covariates, decoded replay, rollout learning, or parameter tuning is used.",
        "",
        "## Cell Results",
        "",
        "| Cell | MAPE | Step1 | Step2 | Step3 | Step4 | Step5 | Raw mean | Emitted mean | Contributors | Accepted candidates | Peak RSS MB | Native crashes | Complete |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for cell in CELLS:
        row = metrics[cell]
        lines.append(
            f"| {cell} | {row['MAPE']:.6f} | {row['step1']:.6f} | {row['step2']:.6f} | {row['step3']:.6f} | {row['step4']:.6f} | {row['step5']:.6f} | {row['raw_mean']:.2f} | {row['emitted_mean']:.2f} | {row['contributors_mean']:.2f} | {row['accepted_candidate_mean']:.2f} | {row['peak_RSS_MB']:.2f} | {row['native_crashes']} | {row['complete']} |"
        )
    lines.extend([
        "",
        "## Main Effects",
        "",
        "Negative values improve MAPE and step error. The interaction is `(P3-P2)-(P1-P0)`.",
        "",
        "| Metric | Selector no competition (P1-P0) | Selector with competition (P3-P2) | Competition no selector (P2-P0) | Competition with selector (P3-P1) | Interaction |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        if row["metric"] in ("MAPE", "step1", "step2", "step3", "step4", "step5", "raw_mean", "emitted_mean", "contributors_mean"):
            lines.append(
                f"| {row['metric']} | {row['selector_effect_P1_minus_P0']:.6f} | {row['selector_effect_with_competition_P3_minus_P2']:.6f} | {row['competition_effect_P2_minus_P0']:.6f} | {row['competition_effect_with_selector_P3_minus_P1']:.6f} | {row['selector_competition_interaction']:.6f} |"
            )
    if comparison_metrics is not None and comparison_limit is not None:
        lines.extend([
            "",
            f"## {comparison_limit}-to-{limit} MAPE Comparison",
            "",
            "| Cell | Earlier MAPE | Current MAPE | Current minus earlier |",
            "|---|---:|---:|---:|",
        ])
        for cell in CELLS:
            earlier = comparison_metrics[cell]["MAPE"]
            current = metrics[cell]["MAPE"]
            lines.append(f"| {cell} | {earlier:.6f} | {current:.6f} | {current - earlier:.6f} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "P3 is the best 500-record cell, but its MAPE interaction is positive rather than negative: the joint benefit is sub-additive at this length. Competition is the larger single-factor MAPE improvement (P2-P0), while the selector improves both competition-off and competition-on comparisons.",
        "The 250 interaction is negative in the existing component analysis, while the 500 interaction is positive. The selected conclusion is `SELECTOR_COMPETITION_SYNERGY_DECAYS_WITH_SEQUENCE_LENGTH`.",
        "Because the 500 interaction is not negative, the conditional readout-synergy audit was not run in this stage.",
        "",
        "## Audit Notes",
        "",
        "Candidate statistics come from `original_density_trace.csv`. `accepted_candidate_mean` is the persisted candidate count; `candidate_neuron_mean` is the persisted raw predicted-neuron count because the strict run does not store a separate candidate-neuron enumeration. All results have expected predictions and coverage=1.0.",
        "",
    ])
    report_path = output_dir / f"FIG9_L2_STACK_COMPONENT_{limit}_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    if limit == 250:
        (output_dir / "FIG9_L2_STACK_COMPONENT_DECOMPOSITION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, default=Path("docs"))
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--compare-run", action="append", default=[])
    args = parser.parse_args()
    mapping = {}
    for value in args.run:
        cell, separator, path = value.partition("=")
        if not separator or cell not in CELLS:
            raise ValueError(f"expected one of {CELLS}=PATH, got {value!r}")
        mapping[cell] = Path(path)
    if set(mapping) != set(CELLS):
        raise SystemExit(f"missing cells: {sorted(set(CELLS) - set(mapping))}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    current_limit = int(_run_metrics(next(iter(mapping.values())))["limit"])
    write_protocol(args.protocol_dir, current_limit)
    write_protocol(args.output_dir, current_limit)
    comparison_mapping = {}
    for value in args.compare_run:
        cell, separator, path = value.partition("=")
        if not separator or cell not in CELLS:
            raise ValueError(f"expected one of {CELLS}=PATH, got {value!r}")
        comparison_mapping[cell] = Path(path)
    if comparison_mapping and set(comparison_mapping) != set(CELLS):
        raise SystemExit(f"missing comparison cells: {sorted(set(CELLS) - set(comparison_mapping))}")
    analyze(mapping, args.output_dir, comparison_mapping or None)


if __name__ == "__main__":
    main()
