"""Orchestrate scenario-aware teacher-reference analysis from existing traces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

from experiments.diagnostics.analyze_fig9_observe_scenario_assignment import (
    analyze as analyze_scenarios,
)
from experiments.diagnostics.analyze_fig9_reference_neuron_selection import (
    analyze as analyze_selection,
)
from experiments.diagnostics.analyze_fig9_teacher_forced_identity import (
    analyze as analyze_identity,
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _metric(
    rows: Sequence[Mapping[str, str]],
    metric: str,
    *,
    scope: str,
    field: str = "",
    horizon_step: str = "",
) -> Mapping[str, str]:
    return next(
        row
        for row in rows
        if row["metric"] == metric
        and row["reference_scope"] == scope
        and row.get("field", "") == field
        and row.get("horizon_step", "") == horizon_step
    )


def _write_report(
    output_dir: Path,
    *,
    identity_summary: Mapping[str, object],
    selection_summary: Mapping[str, object],
    scenario_summary: Mapping[str, object],
) -> None:
    scoped = _read(output_dir / "teacher_reference_scoped_metrics.csv")
    operational_wrong = _metric(
        scoped,
        "correct_column_wrong_neuron_rate",
        scope="ALL_OPERATIONAL_REFERENCES",
        field="passenger",
    )
    preexisting_wrong = _metric(
        scoped,
        "correct_column_wrong_neuron_rate",
        scope="PREEXISTING_NEURON_REFERENCES",
        field="passenger",
    )
    passenger_step1 = {
        metric: _metric(
            scoped,
            metric,
            scope="PREEXISTING_NEURON_REFERENCES",
            field="passenger",
            horizon_step="1",
        )
        for metric in (
            "target_column_candidate_recall",
            "target_column_emitted_recall",
            "reference_neuron_crossing_recall",
            "reference_neuron_candidate_recall",
            "reference_neuron_emitted_recall",
            "correct_column_wrong_neuron_rate",
        )
    }
    applicability = {
        row["reference_applicability"]: int(row["count"])
        for row in _read(
            output_dir / "teacher_reference_applicability_summary.csv"
        )
    }
    passenger_reasons = scenario_summary[
        "passenger_scenario3_reasons"
    ]
    ranking = sorted(
        selection_summary["ranking_metrics"],
        key=lambda row: float(row["hit_at_1"]),
        reverse=True,
    )
    lines = [
        "# Fig.9 Scenario-Aware Teacher Reference Audit",
        "",
        "Scenario 3 operational winners are not valid pre-existing neuron or "
        "segment references. Primary neuron-identity metrics use only "
        "`PREEXISTING_NEURON_REFERENCES`; exact segment metrics use only "
        "`PREEXISTING_SEGMENT_REFERENCES`.",
        "",
        f"Boundary censoring was not filled by shadow learning. Exactly "
        f"{identity_summary['boundary_unavailable_rows']} target-column rows "
        "point to the final five records that the main loop never observed. "
        "They remain explicitly unavailable so the original model, RNG, "
        "predictions, checkpoint, and fingerprints remain untouched.",
        "",
        "Steps 2-5 include autonomous trajectory divergence. Step 1 is the "
        "primary initial-selection diagnostic.",
        "",
        "## Reference Applicability",
        "",
    ]
    for label, count in sorted(applicability.items()):
        lines.append(f"- `{label}`: {count}")
    lines.extend(
        [
            "",
            "## Corrected Passenger Interpretation",
            "",
            "Among all post-observation operational winners, wrong-neuron "
            f"given emitted correct column is "
            f"{float(operational_wrong['rate']):.6f} "
            f"({operational_wrong['numerator']}/"
            f"{operational_wrong['denominator']}). This scope includes "
            "post-observation-created Scenario 3 identities and is not a "
            "selector-error estimate.",
            "",
            "Among pre-existing Scenario 1/2 neuron references, the same rate "
            f"is {float(preexisting_wrong['rate']):.6f} "
            f"({preexisting_wrong['numerator']}/"
            f"{preexisting_wrong['denominator']}).",
            "",
            "## Passenger Step 1, Pre-Existing References",
            "",
            "| metric | numerator | denominator | rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric, row in passenger_step1.items():
        lines.append(
            f"| {metric} | {row['numerator']} | {row['denominator']} | "
            f"{float(row['rate']):.6f} |"
        )
    lines.extend(
        [
            "",
            "The formal crossing-level trace shows the dominant Step-1 losses "
            "after threshold crossing. It cannot distinguish no-segment, "
            "nonpositive-response, and below-threshold causes before crossing; "
            "those fields require the new read-only hook in a future run.",
            "",
            "## Offline Local Ranking",
            "",
            "| metric | Hit@1 | MRR | pairwise win rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in ranking:
        lines.append(
            f"| {row['metric']} | {float(row['hit_at_1']):.6f} | "
            f"{float(row['mrr']):.6f} | "
            f"{float(row['pairwise_win_rate']):.6f} |"
        )
    lines.extend(
        [
            "",
            "These rules are oracle-conditioned on the already-known target "
            "column and are offline diagnostics only. They do not produce a "
            "SymbolCode, MAPE, or deployable selector.",
            "",
            "## Scenario 3 Cause Availability",
            "",
        ]
    )
    if float(scenario_summary["reason_trace_available_rate"]) == 0.0:
        lines.append(
            "The existing formal 250 observation trace predates the Scenario-3 "
            "reason hook. Exact no-segment versus below-L_match causes are "
            "therefore unavailable and are not inferred from the final model."
        )
    else:
        for row in passenger_reasons:
            lines.append(
                f"- `{row['reason']}`: {row['count']} "
                f"({float(row['fraction']):.6f})"
            )
    (output_dir / "FIG9_SCENARIO_AWARE_IDENTITY_REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def analyze(
    *,
    run_dir: Path,
    output_dir: Path,
    policy_label: str = "batched",
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    identity = analyze_identity(
        run_dir=run_dir,
        output_dir=output_dir,
        policy_label=policy_label,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    selection = analyze_selection(
        run_dir=run_dir,
        identity_dir=output_dir,
        output_dir=output_dir,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    scenarios = analyze_scenarios(
        observation_trace=run_dir / "teacher_forced_observation_trace.csv",
        output_dir=output_dir,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    summary = {
        "diagnostic_only": True,
        "offline_analysis_only": True,
        "teacher_reference": identity,
        "reference_neuron_selection": selection,
        "observe_scenario_assignment": scenarios,
        "formal_model_rerun": False,
        "strict_defaults_changed": False,
    }
    (output_dir / "teacher_reference_scenario_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir,
        identity_summary=identity,
        selection_summary=selection,
        scenario_summary=scenarios,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy-label", default="batched")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze(
        run_dir=Path(args.run_dir),
        output_dir=Path(args.output_dir),
        policy_label=args.policy_label,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()
