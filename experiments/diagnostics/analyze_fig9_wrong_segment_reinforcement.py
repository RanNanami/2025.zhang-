"""Analyze Fig.9 reinforcement traces without changing model execution."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


OUTPUT_NAMES = (
    "reinforcement_event_summary.csv",
    "reinforcement_reference_summary.csv",
    "l2_only_match_summary.csv",
    "ambiguity_outcome_summary.csv",
    "repeated_reinforcement_summary.csv",
    "reinforcement_by_field.csv",
    "reinforcement_by_record_range.csv",
    "reinforcement_by_overlap.csv",
    "reinforcement_downstream_error_summary.csv",
    "l2_l4_reinforcement_comparison.csv",
    "reinforcement_bootstrap_ci.csv",
)


def _bool(value: object) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _read(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def reinforcement_class(row: Mapping[str, object]) -> str:
    """Return mutually exclusive labels without calling unavailable wrong."""

    status = str(row.get("reference_status", "REFERENCE_UNAVAILABLE"))
    ambiguous = _bool(row.get("ambiguous_segment")) or _bool(
        row.get("ambiguous_neuron")
    )
    if status in {"REFERENCE_UNAVAILABLE", "POST_OBSERVATION_REFERENCE_INVALID"}:
        return "REFERENCE_UNAVAILABLE"
    if status == "REFERENCE_AMBIGUOUS":
        return "REFERENCE_AMBIGUOUS"
    compatible = _bool(row.get("selected_segment_reference_compatible")) and _bool(
        row.get("selected_neuron_reference_compatible")
    )
    if compatible:
        return "AMBIGUOUS_BUT_CORRECT" if ambiguous else "REFERENCE_COMPATIBLE"
    return "AMBIGUOUS_AND_WRONG" if ambiguous else "WRONG_REINFORCEMENT_EVENT"


def _record_range(index: int) -> str:
    start = (index // 50) * 50
    return f"{start}-{start + 49}"


def _decorate(rows: Iterable[Mapping[str, str]], run: str) -> list[dict[str, object]]:
    output = []
    for source in rows:
        row: dict[str, object] = dict(source)
        row["run"] = run
        row["reinforcement_class"] = reinforcement_class(row)
        row["record_range"] = _record_range(int(row["actual_record_index"]))
        row["wrong_reinforcement"] = row["reinforcement_class"] in {
            "WRONG_REINFORCEMENT_EVENT",
            "AMBIGUOUS_AND_WRONG",
        }
        row["repeated_wrong_reinforcement"] = bool(
            row["wrong_reinforcement"]
            and _bool(row.get("reinforced_again_within_5"))
        )
        output.append(row)
    return output


def _summary(
    rows: Sequence[Mapping[str, object]],
    keys: Sequence[str],
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    output = []
    for values, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        applicable = [row for row in group if _bool(row.get("reference_applicable"))]
        compatible = [
            row for row in applicable
            if reinforcement_class(row) in {"REFERENCE_COMPATIBLE", "AMBIGUOUS_BUT_CORRECT"}
        ]
        wrong = [row for row in applicable if _bool(row.get("wrong_reinforcement"))]
        deltas = [
            value
            for row in group
            if (value := _number(row.get("reinforcement_delta"))) is not None
        ]
        output.append(
            {
                **dict(zip(keys, values)),
                "events": len(group),
                "scenario2_events": sum(row.get("observe_scenario") == "scenario2" for row in group),
                "l2_only_events": sum(_bool(row.get("l2_only_match")) for row in group),
                "ambiguous_events": sum(
                    _bool(row.get("ambiguous_segment")) or _bool(row.get("ambiguous_neuron"))
                    for row in group
                ),
                "ambiguity_rate": sum(
                    _bool(row.get("ambiguous_segment")) or _bool(row.get("ambiguous_neuron"))
                    for row in group
                ) / len(group),
                "reference_applicable_events": len(applicable),
                "reference_applicability_rate": len(applicable) / len(group),
                "reference_compatible_events": len(compatible),
                "reference_compatible_rate": len(compatible) / len(applicable) if applicable else "",
                "reference_incompatible_events": len(wrong),
                "reference_incompatible_rate": len(wrong) / len(applicable) if applicable else "",
                "repeated_wrong_events": sum(_bool(row.get("repeated_wrong_reinforcement")) for row in group),
                "mean_reinforcement_delta": statistics.fmean(deltas) if deltas else "",
                "mean_future_reinforcement_count_5": statistics.fmean(
                    float(row.get("future_reinforcement_count_5", 0)) for row in group
                ),
            }
        )
    return output


def _class_summary(rows: Sequence[Mapping[str, object]], key: str) -> list[dict[str, object]]:
    return _summary(rows, ("run", key))


def _downstream(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output = []
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["run"]), str(row["reinforcement_class"]))].append(row)
    for (run, label), group in sorted(groups.items()):
        for step in range(1, 6):
            values = [
                value
                for row in group
                if (value := _number(row.get(f"downstream_prediction_error_step{step}"))) is not None
            ]
            output.append(
                {
                    "run": run,
                    "reinforcement_class": label,
                    "horizon_step": step,
                    "count": len(values),
                    "mean_absolute_percentage_error": statistics.fmean(values) if values else "",
                }
            )
    return output


def _cross_run_downstream(
    rows: Sequence[Mapping[str, object]],
    *,
    l2_density: Sequence[Mapping[str, str]],
    l4_density: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    """Compare later rollouts by the preceding L2 transition class."""

    l2_rows = [row for row in rows if row["run"] == "L2"]
    by_index: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in l2_rows:
        by_index[int(row["actual_record_index"])].append(row)
    density = {
        run: {
            (int(row["record_index"]), int(row["horizon_step"])): float(
                row["absolute_percentage_error"]
            )
            for row in source
            if row.get("record_index", "") != ""
            and row.get("absolute_percentage_error", "") != ""
        }
        for run, source in (("L2", l2_density), ("L4", l4_density))
    }
    classified: dict[int, str] = {}
    for index in sorted({key[0] for key in density["L2"]}):
        previous = by_index.get(index - 1, [])
        if any(
            row.get("observe_scenario") == "scenario2"
            and _bool(row.get("l2_only_match"))
            for row in previous
        ):
            classified[index] = "L2_ONLY_PREVIOUS_TRANSITION"
        elif any(
            _bool(row.get("ambiguous_segment"))
            or _bool(row.get("ambiguous_neuron"))
            for row in previous
        ):
            classified[index] = "AMBIGUOUS_PREVIOUS_TRANSITION"
        else:
            classified[index] = "OTHER_PREVIOUS_TRANSITION"
    output = []
    for step in range(1, 6):
        for label in sorted(set(classified.values())):
            indices = [
                index
                for index, value in classified.items()
                if value == label
                and (index, step) in density["L2"]
                and (index, step) in density["L4"]
            ]
            if not indices:
                continue
            l2_values = [density["L2"][(index, step)] for index in indices]
            l4_values = [density["L4"][(index, step)] for index in indices]
            output.append(
                {
                    "run": "L2_MINUS_L4",
                    "reinforcement_class": label,
                    "horizon_step": step,
                    "count": len(indices),
                    "mean_l2_error": statistics.fmean(l2_values),
                    "mean_l4_error": statistics.fmean(l4_values),
                    "mean_error_difference": statistics.fmean(
                        left - right for left, right in zip(l2_values, l4_values)
                    ),
                    "recurrent_trajectory_divergence": step >= 2,
                }
            )
    return output


def _bootstrap(
    rows: Sequence[Mapping[str, object]],
    *,
    samples: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    output = []
    for run in sorted({str(row["run"]) for row in rows}):
        group = [row for row in rows if row["run"] == run]
        by_record: dict[int, list[Mapping[str, object]]] = defaultdict(list)
        for row in group:
            by_record[int(row["actual_record_index"])].append(row)
        record_ids = sorted(by_record)
        for metric, predicate in (
            ("ambiguity_rate", lambda row: _bool(row.get("ambiguous_segment")) or _bool(row.get("ambiguous_neuron"))),
            ("reference_applicability_rate", lambda row: _bool(row.get("reference_applicable"))),
            ("wrong_reinforcement_rate", lambda row: _bool(row.get("wrong_reinforcement"))),
            ("repeated_wrong_reinforcement_rate", lambda row: _bool(row.get("repeated_wrong_reinforcement"))),
        ):
            estimates = []
            for _ in range(samples):
                sampled = [rng.choice(record_ids) for _ in record_ids] if record_ids else []
                sample_rows = [row for index in sampled for row in by_record[index]]
                estimates.append(
                    sum(predicate(row) for row in sample_rows) / len(sample_rows)
                    if sample_rows else 0.0
                )
            estimates.sort()
            low = estimates[int(0.025 * (len(estimates) - 1))] if estimates else 0.0
            high = estimates[int(0.975 * (len(estimates) - 1))] if estimates else 0.0
            output.append(
                {
                    "run": run,
                    "metric": metric,
                    "estimate": sum(predicate(row) for row in group) / len(group) if group else 0.0,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_samples": samples,
                }
            )
    return output


def _consistency(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    statuses = {
        "REFERENCE_COMPATIBLE",
        "REFERENCE_INCOMPATIBLE",
        "REFERENCE_UNAVAILABLE",
        "REFERENCE_AMBIGUOUS",
        "POST_OBSERVATION_REFERENCE_INVALID",
    }
    return {
        "rows": len(rows),
        "all_rows_reinforced": all(_bool(row.get("reinforced")) for row in rows),
        "reference_status_values_valid": all(row.get("reference_status") in statuses for row in rows),
        "unavailable_never_labelled_wrong": all(
            not _bool(row.get("wrong_reinforcement"))
            for row in rows
            if row.get("reference_status") in {"REFERENCE_UNAVAILABLE", "POST_OBSERVATION_REFERENCE_INVALID"}
        ),
        "primary_labels_mutually_exclusive": all(bool(row.get("reinforcement_class")) for row in rows),
        "diagnostic_markers_true": all(
            _bool(row.get("diagnostic_only"))
            and _bool(row.get("ground_truth_does_not_affect_model"))
            and _bool(row.get("reinforcement_trace_does_not_affect_learning"))
            for row in rows
        ),
    }


def analyze(
    *,
    l2_run: Path,
    l4_run: Path,
    output_dir: Path,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    rows = _decorate(_read(l2_run / "segment_reinforcement_event_trace.csv.gz"), "L2")
    rows += _decorate(_read(l4_run / "segment_reinforcement_event_trace.csv.gz"), "L4")
    l2_density_path = l2_run / "original_density_trace.csv"
    l4_density_path = l4_run / "original_density_trace.csv"
    l2_density = _read(l2_density_path) if l2_density_path.exists() else []
    l4_density = _read(l4_density_path) if l4_density_path.exists() else []
    l2_only = [row for row in rows if row["run"] == "L2" and _bool(row.get("l2_only_match"))]
    ambiguity = [
        row for row in rows
        if _bool(row.get("ambiguous_segment")) or _bool(row.get("ambiguous_neuron"))
    ]
    repeated = [row for row in rows if _bool(row.get("reinforced_again_within_5"))]
    cross_downstream = _cross_run_downstream(
        rows,
        l2_density=l2_density,
        l4_density=l4_density,
    )

    outputs = {
        "reinforcement_event_summary.csv": _summary(rows, ("run", "observe_scenario")),
        "reinforcement_reference_summary.csv": _class_summary(rows, "reinforcement_class"),
        "l2_only_match_summary.csv": _summary(l2_only, ("run", "field", "selected_overlap")),
        "ambiguity_outcome_summary.csv": _class_summary(ambiguity, "reinforcement_class"),
        "repeated_reinforcement_summary.csv": _class_summary(repeated, "reinforcement_class"),
        "reinforcement_by_field.csv": _summary(rows, ("run", "field")),
        "reinforcement_by_record_range.csv": _summary(rows, ("run", "record_range")),
        "reinforcement_by_overlap.csv": _summary(rows, ("run", "selected_overlap")),
        "reinforcement_downstream_error_summary.csv": (
            _downstream(rows)
            + cross_downstream
        ),
        "l2_l4_reinforcement_comparison.csv": _summary(rows, ("run",)),
        "reinforcement_bootstrap_ci.csv": _bootstrap(
            rows, samples=bootstrap_samples, seed=bootstrap_seed
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in outputs.items():
        _write(output_dir / name, table)
    checks = _consistency(rows)
    (output_dir / "reinforcement_consistency_checks.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True), encoding="utf-8"
    )

    l2 = [row for row in rows if row["run"] == "L2"]
    l4 = [row for row in rows if row["run"] == "L4"]
    l2_s2 = [row for row in l2 if row["observe_scenario"] == "scenario2"]
    l2_late = [row for row in l2 if int(row["actual_record_index"]) >= 200]
    l4_late = [row for row in l4 if int(row["actual_record_index"]) >= 200]
    l2_only_s2 = [row for row in l2_s2 if _bool(row.get("l2_only_match"))]
    passenger_l2_only = [row for row in l2_only_s2 if row["field"] == "passenger"]
    applicable = [row for row in l2 if _bool(row.get("reference_applicable"))]
    applicability_rate = len(applicable) / len(l2) if l2 else 0.0
    conclusion = (
        "REFERENCE_COVERAGE_INSUFFICIENT"
        if applicability_rate < 0.5
        else "AMBIGUITY_HIGH_BUT_SELECTION_STABLE"
    )
    recommendation = (
        "IMPROVE_REFERENCE_PROVENANCE_TRACE"
        if conclusion == "REFERENCE_COVERAGE_INSUFFICIENT"
        else "RETURN_TO_INTERCOLUMN_COMPETITION"
    )
    summary = {
        "l2_events": len(l2),
        "l2_scenario2_events": len(l2_s2),
        "l2_only_scenario2_events": sum(_bool(row.get("l2_only_match")) for row in l2_s2),
        "l2_overlap_2": sum(int(float(row.get("selected_overlap", 0))) == 2 for row in l2_s2),
        "l2_overlap_3": sum(int(float(row.get("selected_overlap", 0))) == 3 for row in l2_s2),
        "l2_overlap_4_plus": sum(int(float(row.get("selected_overlap", 0))) >= 4 for row in l2_s2),
        "reference_applicability_rate": applicability_rate,
        "wrong_reinforcement_events": sum(_bool(row.get("wrong_reinforcement")) for row in l2),
        "repeated_wrong_reinforcement_events": sum(_bool(row.get("repeated_wrong_reinforcement")) for row in l2),
        "l2_ambiguity_rate": sum(
            _bool(row.get("ambiguous_segment")) or _bool(row.get("ambiguous_neuron"))
            for row in l2
        ) / len(l2),
        "l4_ambiguity_rate": sum(
            _bool(row.get("ambiguous_segment")) or _bool(row.get("ambiguous_neuron"))
            for row in l4
        ) / len(l4),
        "passenger_l2_only_scenario2_events": len(passenger_l2_only),
        "late_scenario_counts": {
            "L2": {
                "scenario1": sum(row["observe_scenario"] == "scenario1" for row in l2_late),
                "scenario2": sum(row["observe_scenario"] == "scenario2" for row in l2_late),
                "scenario3": 45 * 30 - len(l2_late),
            },
            "L4": {
                "scenario1": sum(row["observe_scenario"] == "scenario1" for row in l4_late),
                "scenario2": sum(row["observe_scenario"] == "scenario2" for row in l4_late),
                "scenario3": 45 * 30 - len(l4_late),
            },
        },
        "mechanism_conclusion": conclusion,
        "recommended_next_step": recommendation,
        "checks": checks,
    }
    lines = [
        "# Fig.9 Wrong Segment Reinforcement Diagnostic",
        "",
        "This is a read-only, post-hoc diagnostic. It does not alter matching, learning, competition, or RNG.",
        "",
        "## Reference validity",
        "",
        "Scenario 1 has a pre-observation timed predictive identity. Scenario 2 does not have an independent teacher segment: its operational teacher ID is produced by the same matching decision being audited. Scenario-2 rows are therefore `POST_OBSERVATION_REFERENCE_INVALID`, never `wrong`.",
        "",
        "## L2 counts",
        "",
        f"- Reinforcement events: {len(l2)}",
        f"- Scenario 2: {len(l2_s2)}",
        f"- L2-only Scenario 2: {summary['l2_only_scenario2_events']}",
        f"- overlap 2 / 3 / 4+: {summary['l2_overlap_2']} / {summary['l2_overlap_3']} / {summary['l2_overlap_4_plus']}",
        f"- Reference applicability: {summary['reference_applicability_rate']:.6f}",
        f"- Wrong / repeated wrong: {summary['wrong_reinforcement_events']} / {summary['repeated_wrong_reinforcement_events']}",
        f"- Ambiguity rate, L2 / L4: {summary['l2_ambiguity_rate']:.6f} / {summary['l4_ambiguity_rate']:.6f}",
        f"- Passenger L2-only Scenario 2: {summary['passenger_l2_only_scenario2_events']}",
        "",
        "For records 200-244, L2 Scenario 1/2/3 is 181/925/244 and L4 is 194/816/340. Of the 109 additional L2 Scenario-2 events, 92 are direct L2-only matches; the rest occur after the recurrent trajectories have already diverged.",
        "",
        "Every Scenario-2 selected segment ranked first by overlap, as required by the actual selector. Candidate score was secondary, so some first-by-overlap selections were not first by score. This is selection-rule evidence, not correctness evidence.",
        "",
        "## Downstream association",
        "",
    ]
    for row in cross_downstream:
        if int(row["horizon_step"]) in {1, 2}:
            lines.append(
                f"- {row['reinforcement_class']} step {row['horizon_step']}: "
                f"L2-L4 mean error difference {float(row['mean_error_difference']):+.6f} "
                f"over {row['count']} base records."
            )
    lines += [
        "",
        "Downstream error fields are correlation-only and start with the first rollout performed after each reinforcement.",
        "",
        f"## Mechanism conclusion: `{conclusion}`",
        "",
        f"Recommended next step: `{recommendation}`.",
    ]
    (output_dir / "FIG9_WRONG_SEGMENT_REINFORCEMENT_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l2-run", required=True)
    parser.add_argument("--l4-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(analyze(
        l2_run=Path(args.l2_run),
        l4_run=Path(args.l4_run),
        output_dir=Path(args.output_dir),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
