"""Offline P0/P1/P2/P3 decomposition for the L2 diagnostic stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.diagnostics.analyze_fig9_lmatch_stack_2x2 import (
    _bootstrap,
    _number,
    _run_metrics,
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
            "competition_effect_P2_minus_P0": p2 - p0,
            "full_stack_effect_P3_minus_P0": p3 - p0,
            "selector_competition_interaction": (p3 - p2) - (p1 - p0),
        })
    return rows


def write_protocol(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = protocol_matrix()
    _write_rows(output_dir / "l2_stack_component_protocol_matrix.csv", rows)
    lines = [
        "# Fig.9 L2 Stack Component 2x2 Protocol Audit", "",
        "All four cells use L_match=2, raw propagation, reference continuous dynamics, seed=0, warmup=200, K=10, 32 neurons/column, forgetting threshold=65, no compensation, no future covariates, no ground-truth selection, no rollout learning, and no decoded replay.", "",
        "| Cell | Competition | Simultaneous | Selector | Label |", "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['cell']} | {row['competition']} | {row['simultaneous']} | {row['selector']} | {row['label']} |")
    lines.extend(["", "P0/P1 differ only by selector. P0/P2 differ only by competition stack. P1/P3 differ only by competition stack. P2/P3 differ only by selector. P0 and P3 are reused from the completed L2 raw/full-stack 250 runs.", ""])
    (output_dir / "FIG9_L2_STACK_COMPONENT_2X2_PROTOCOL_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(mapping: dict[str, Path], output_dir: Path) -> dict[str, Any]:
    metrics = {cell: _run_metrics(path) for cell, path in mapping.items()}
    flat = [{"cell": cell, **{k: v for k, v in row.items() if k != "record_errors"}} for cell, row in metrics.items()]
    _write_rows(output_dir / "l2_stack_component_2x2_250.csv", flat)
    rows = effects(metrics)
    _write_rows(output_dir / "l2_stack_component_effects_250.csv", rows)
    _write_rows(output_dir / "l2_stack_component_stepwise.csv", [row for row in rows if row["metric"].startswith("step")])
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
    (output_dir / "FINAL_L2_STACK_COMPONENT_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# Fig.9 L2 Stack Component Decomposition", "", "This is a nonpaper diagnostic decomposition; strict defaults are unchanged.", "", "| Cell | MAPE | Step1 | Step2 | Step3 | Step4 | Step5 | Segments | Raw mean | Complete |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for cell in CELLS:
        row = metrics[cell]
        lines.append(f"| {cell} | {row['MAPE']:.6f} | {row['step1']:.6f} | {row['step2']:.6f} | {row['step3']:.6f} | {row['step4']:.6f} | {row['step5']:.6f} | {row['segments']} | {row['raw_mean']:.2f} | {row['complete']} |")
    lines.extend(["", "## Interpretation", "", "The component conclusion must be based on P1/P2/P3, not on component names. Single-seed effects are descriptive and are not universal superiority claims.", ""])
    (output_dir / "FIG9_L2_STACK_COMPONENT_DECOMPOSITION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, default=Path("docs"))
    parser.add_argument("--run", action="append", default=[])
    args = parser.parse_args()
    write_protocol(args.protocol_dir)
    mapping = {}
    for value in args.run:
        cell, separator, path = value.partition("=")
        if not separator or cell not in CELLS:
            raise ValueError(f"expected one of {CELLS}=PATH, got {value!r}")
        mapping[cell] = Path(path)
    if set(mapping) != set(CELLS):
        raise SystemExit(f"missing cells: {sorted(set(CELLS) - set(mapping))}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analyze(mapping, args.output_dir)


if __name__ == "__main__":
    main()
