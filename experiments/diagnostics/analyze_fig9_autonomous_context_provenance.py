"""Offline analysis for the Fig.9 autonomous-context provenance audit.

The analyzer consumes candidate-level CSV/GZ traces only.  It never imports
the model runner, never selects a candidate, and never changes a checkpoint.
The input may be one run or a pair of L2/L4 runs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


PRIMARY_CONCLUSIONS = (
    "SELF_GENERATED_CONTEXT_TAKES_OVER_AUTONOMOUS_ROLLOUT",
    "L2_SLOWS_SELF_GENERATED_CONTEXT_EXPANSION",
    "L2_PRESERVES_MORE_ACTUAL_HISTORY",
    "L2_ONLY_REDUCES_TOTAL_AUTONOMOUS_ACTIVITY",
    "INTERCOLUMN_COMPETITION_REMOVES_ACTUAL_HISTORY",
    "RECENCY_COLLAPSE_IS_MAINLY_A_METRIC_ARTIFACT",
    "AUTONOMOUS_CONTEXT_PROVENANCE_DOES_NOT_EXPLAIN_L2_ADVANTAGE",
    "AUTONOMOUS_CONTEXT_PROVENANCE_TRACE_IS_INSUFFICIENT",
)
NEXT_STEPS = (
    "KEEP_L2_AS_EXPERIMENTAL_BASELINE",
    "ADD_HISTORICAL_CONTEXT_RETENTION_LOCAL_BIAS",
    "ADD_GENERATED_DEPTH_LOCAL_PENALTY",
    "ADD_ROLLOUT_GENERATED_PROPAGATION_DAMPING",
    "FIX_INTERCOLUMN_ACTUAL_HISTORY_SURVIVAL",
    "INVESTIGATE_ROLLOUT_STATE_RESET",
    "INVESTIGATE_PROPAGATION_STATE_SEMANTICS",
    "INVESTIGATE_AUTONOMOUS_SOURCE_ORDERING",
    "RETURN_TO_INTRACOLUMN_SELECTION",
    "RETURN_TO_INTERCOLUMN_COMPETITION",
    "TEST_LONGER_SEQUENCE",
)


def _number(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _boolean(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[float]) -> float:
    clean = list(values)
    return statistics.fmean(clean) if clean else 0.0


def _bootstrap_ci(values: Sequence[float], *, seed: int = 11) -> tuple[float, float]:
    """Deterministic percentile bootstrap without touching model RNG."""

    if not values:
        return 0.0, 0.0
    state = seed & 0x7FFFFFFF
    samples: list[float] = []
    for _ in range(500):
        picked: list[float] = []
        for _ in values:
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            picked.append(values[state % len(values)])
        samples.append(_mean(picked))
    samples.sort()
    return samples[int(0.025 * (len(samples) - 1))], samples[int(0.975 * (len(samples) - 1))]


def _label_for_path(path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    text = str(path).lower()
    if "l2" in text:
        return "L2"
    if "l4" in text:
        return "L4"
    return path.name


def load_candidate_runs(
    paths: Sequence[Path], labels: Sequence[str] | None = None
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    labels = labels or []
    for ordinal, directory in enumerate(paths):
        trace = directory / "autonomous_candidate_provenance_trace.csv.gz"
        if not trace.exists():
            raise FileNotFoundError(trace)
        label = labels[ordinal] if ordinal < len(labels) else _label_for_path(directory, None)
        for row in _read_rows(trace):
            rows.append({**row, "run_label": label})
    return rows


def _group(rows: Sequence[Mapping[str, object]], *keys: str):
    grouped: dict[tuple[str, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key, "")) for key in keys)].append(row)
    return grouped


def _metric_row(
    label: str, horizon: str, field: str, rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    total = sum(_number(row.get("contributor_count")) for row in rows)
    actual = sum(_number(row.get("source_identity_actual_history_count")) for row in rows)
    generated = sum(_number(row.get("activation_rollout_generated_count")) for row in rows)
    selected = sum(_boolean(row.get("selected_after_intercolumn")) for row in rows)
    emitted = sum(_boolean(row.get("emitted")) for row in rows)
    actual_fraction = actual / total if total else 0.0
    generated_fraction = generated / total if total else 0.0
    depths = [
        _number(row.get("self_generated_depth_mean"))
        for row in rows
        if str(row.get("self_generated_depth_mean", "")) not in {"", "NA"}
    ]
    return {
        "run_label": label,
        "horizon_step": int(_number(horizon)),
        "field": field,
        "candidate_count": len(rows),
        "total_contributors": total,
        "actual_history_contributors": actual,
        "generated_contributors": generated,
        "actual_history_retention_R_h": actual_fraction,
        "self_generated_fraction_S_h": generated_fraction,
        "actual_history_ci_low": _bootstrap_ci(
            [_number(row.get("source_identity_actual_history_fraction")) for row in rows]
        )[0],
        "actual_history_ci_high": _bootstrap_ci(
            [_number(row.get("source_identity_actual_history_fraction")) for row in rows]
        )[1],
        "mean_self_generated_depth": _mean(depths),
        "selected_after_intercolumn_count": selected,
        "emitted_count": emitted,
        "emitted_fraction": emitted / len(rows) if rows else 0.0,
        "mean_raw_column_count": _mean(_number(row.get("raw_column_count")) for row in rows),
        "mean_emitted_column_count": _mean(_number(row.get("emitted_column_count")) for row in rows),
    }


def analyze_runs(
    *, input_paths: Sequence[Path], output_dir: Path, labels: Sequence[str] | None = None
) -> dict[str, object]:
    rows = load_candidate_runs(input_paths, labels)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = _group(rows, "run_label", "horizon_step", "field")
    metric_rows = [
        _metric_row(label, horizon, field, values)
        for (label, horizon, field), values in sorted(grouped.items())
    ]
    _write_rows(output_dir / "source_provenance_by_field.csv", metric_rows)
    _write_rows(output_dir / "source_provenance_by_horizon.csv", metric_rows)
    _write_rows(output_dir / "actual_history_retention.csv", metric_rows)
    _write_rows(output_dir / "self_generated_fraction.csv", metric_rows)
    _write_rows(output_dir / "self_generated_depth.csv", metric_rows)

    step_rows = [
        _metric_row(label, horizon, "ALL", values)
        for (label, horizon), values in sorted(_group(rows, "run_label", "horizon_step").items())
    ]
    _write_rows(output_dir / "autonomous_rollout_provenance_by_step.csv", step_rows)
    origin_rows: list[dict[str, object]] = []
    for row in step_rows:
        source_total = row["total_contributors"] or 1
        source_rows = [
            source for source in ("pre_rollout_actual", "pre_rollout_predicted", "step1", "step2", "step3", "step4", "step5", "unknown")
        ]
        original = [
            value for value in rows
            if value.get("run_label") == row["run_label"]
            and int(_number(value.get("horizon_step"))) == int(row["horizon_step"])
        ]
        for origin in source_rows:
            key = {
                "pre_rollout_actual": "activation_pre_rollout_actual_count",
                "pre_rollout_predicted": "activation_pre_rollout_predicted_count",
                "step1": "origin_step1_count",
                "step2": "origin_step2_count",
                "step3": "origin_step3_count",
                "step4": "origin_step4_count",
                "step5": "origin_step5_count",
                "unknown": "activation_unknown_count",
            }[origin]
            count = sum(_number(value.get(key)) for value in original)
            origin_rows.append({
                "run_label": row["run_label"],
                "horizon_step": row["horizon_step"],
                "origin": origin,
                "count": count,
                "fraction": count / source_total,
            })
    _write_rows(output_dir / "rollout_origin_step_matrix.csv", origin_rows)

    survival_rows: list[dict[str, object]] = []
    for label, label_rows in _group(rows, "run_label").items():
        for source_horizon in range(1, 6):
            source_ids = {
                str(row.get("source_id_fingerprint"))
                for row in label_rows
                if int(_number(row.get("horizon_step"))) == source_horizon
                and _number(row.get("source_identity_actual_history_count")) > 0
            }
            for horizon in range(source_horizon, 6):
                later = {
                    str(row.get("source_id_fingerprint"))
                    for row in label_rows
                    if int(_number(row.get("horizon_step"))) == horizon
                    and _number(row.get("source_identity_actual_history_count")) > 0
                }
                survival_rows.append({
                    "run_label": label,
                    "from_step": source_horizon,
                    "to_step": horizon,
                    "source_count_from": len(source_ids),
                    "source_count_survived": len(source_ids & later),
                    "survival_fraction": len(source_ids & later) / len(source_ids) if source_ids else 0.0,
                })
    _write_rows(output_dir / "actual_source_survival_matrix.csv", survival_rows)

    target_false: list[dict[str, object]] = []
    for (label, horizon, field), values in sorted(grouped.items()):
        for role, subset in (("target", [row for row in values if _boolean(row.get("is_target_column_candidate"))]), ("false", [row for row in values if not _boolean(row.get("is_target_column_candidate"))])):
            item = _metric_row(label, horizon, field, subset)
            item["candidate_role"] = role
            item["mean_candidate_score"] = _mean(_number(row.get("candidate_score")) for row in subset)
            item["mean_reactivation_fraction"] = _mean(_number(row.get("same_rollout_reactivation_fraction")) for row in subset)
            target_false.append(item)
    _write_rows(output_dir / "target_false_provenance.csv", target_false)
    _write_rows(output_dir / "passenger_target_false_provenance.csv", [row for row in target_false if row.get("field") == "passenger"])

    intercolumn_rows: list[dict[str, object]] = []
    for (label, horizon), values in sorted(_group(rows, "run_label", "horizon_step").items()):
        for state, subset in (("actual_identity", [row for row in values if _number(row.get("source_identity_actual_history_count")) > 0]), ("generated_activation", [row for row in values if _number(row.get("activation_rollout_generated_count")) > 0])):
            intercolumn_rows.append({
                "run_label": label,
                "horizon_step": horizon,
                "state": state,
                "candidate_count": len(subset),
                "selected_after_intercolumn_count": sum(_boolean(row.get("selected_after_intercolumn")) for row in subset),
                "emitted_count": sum(_boolean(row.get("emitted")) for row in subset),
                "survival_fraction": sum(_boolean(row.get("emitted")) for row in subset) / len(subset) if subset else 0.0,
            })
    _write_rows(output_dir / "intercolumn_provenance_survival.csv", intercolumn_rows)

    recency_rows: list[dict[str, object]] = []
    compactness_rows: list[dict[str, object]] = []
    for (label, horizon, field), values in sorted(grouped.items()):
        recency_rows.append({
            "run_label": label,
            "horizon_step": horizon,
            "field": field,
            "age_definition": "AGE_FROM_LAST_ANY_ACTIVATION",
            "recent5_fraction": _mean(_number(row.get("recent5_any_fraction")) for row in values),
        })
        recency_rows.extend([
            {"run_label": label, "horizon_step": horizon, "field": field, "age_definition": "AGE_FROM_LAST_ACTUAL_OBSERVATION", "recent5_fraction": _mean(_number(row.get("recent5_actual_fraction")) for row in values)},
            {"run_label": label, "horizon_step": horizon, "field": field, "age_definition": "AGE_FROM_PRE_ROLLOUT_ACTUAL_ROOT", "recent5_fraction": _mean(_number(row.get("recent5_root_fraction")) for row in values)},
        ])
        compactness_rows.extend([
            {"run_label": label, "horizon_step": horizon, "field": field, "compactness_definition": "ANY_ACTIVATION", "compactness": _mean(_number(row.get("compactness_any_activation")) for row in values)},
            {"run_label": label, "horizon_step": horizon, "field": field, "compactness_definition": "ACTUAL_HISTORY_AGE", "compactness": _mean(_number(row.get("compactness_actual_history_age")) for row in values)},
            {"run_label": label, "horizon_step": horizon, "field": field, "compactness_definition": "ACTUAL_ROOT_AGE", "compactness": _mean(_number(row.get("compactness_actual_root_age")) for row in values)},
        ])
    _write_rows(output_dir / "recency_metric_semantics.csv", recency_rows)
    _write_rows(output_dir / "compactness_metric_semantics.csv", compactness_rows)

    comparison_rows: list[dict[str, object]] = []
    labels_present = sorted({str(row["run_label"]) for row in rows})
    for horizon in range(1, 6):
        values = {label: next((item for item in step_rows if item["run_label"] == label and item["horizon_step"] == horizon), None) for label in labels_present}
        comparison_rows.append({
            "horizon_step": horizon,
            "L2_actual_history_retention": (values.get("L2") or {}).get("actual_history_retention_R_h", "NA"),
            "L4_actual_history_retention": (values.get("L4") or {}).get("actual_history_retention_R_h", "NA"),
            "L2_self_generated_fraction": (values.get("L2") or {}).get("self_generated_fraction_S_h", "NA"),
            "L4_self_generated_fraction": (values.get("L4") or {}).get("self_generated_fraction_S_h", "NA"),
            "L2_candidate_count": (values.get("L2") or {}).get("candidate_count", "NA"),
            "L4_candidate_count": (values.get("L4") or {}).get("candidate_count", "NA"),
        })
    _write_rows(output_dir / "l2_l4_provenance_comparison.csv", comparison_rows)

    _write_rows(output_dir / "source_provenance_funnel.csv", step_rows)
    _write_rows(output_dir / "optional_500_results.csv", [{"status": "NOT_RUN", "reason": "This diagnostic does not authorize a 500-record run."}])
    _write_rows(output_dir / "EXPERIMENT_LEDGER.csv", [{"analysis": "autonomous_context_provenance", "model_changed": False, "ground_truth_used_for_selection": False, "online_mechanism_implemented": False}])

    recent_any = _mean(_number(row.get("recent5_any_fraction")) for row in rows if int(_number(row.get("horizon_step"))) >= 2)
    recent_actual = _mean(_number(row.get("recent5_actual_fraction")) for row in rows if int(_number(row.get("horizon_step"))) >= 2)
    generated = _mean(_number(row.get("activation_rollout_generated_fraction")) for row in rows if int(_number(row.get("horizon_step"))) >= 2)
    l2_steps = [row for row in step_rows if row["run_label"] == "L2" and int(row["horizon_step"]) >= 2]
    l4_steps = [row for row in step_rows if row["run_label"] == "L4" and int(row["horizon_step"]) >= 2]
    l2_total = _mean(_number(row.get("total_contributors")) for row in l2_steps)
    l4_total = _mean(_number(row.get("total_contributors")) for row in l4_steps)
    l2_actual = _mean(_number(row.get("actual_history_retention_R_h")) for row in l2_steps)
    l4_actual = _mean(_number(row.get("actual_history_retention_R_h")) for row in l4_steps)
    actual_retention_difference = l2_actual - l4_actual
    recency_artifact = recent_any > 0.95 and recent_actual < 0.5
    mechanism_rows = [
        {
            "hypothesis": "H1_SELF_GENERATED_CONTEXT_TAKES_OVER",
            "supported": generated > 0.5,
            "evidence": "activation_rollout_generated_fraction_step_ge_2",
            "value": generated,
        },
        {
            "hypothesis": "H3_L2_PRESERVES_MORE_ACTUAL_HISTORY",
            "supported": actual_retention_difference > 0.02,
            "evidence": "L2_minus_L4_actual_history_retention",
            "value": actual_retention_difference,
        },
        {
            "hypothesis": "H4_L2_ONLY_REDUCES_TOTAL_ACTIVITY",
            "supported": bool(l2_steps and l4_steps and l2_total < l4_total and actual_retention_difference <= 0.02),
            "evidence": "mean_total_contributors_step_ge_2",
            "value": l2_total - l4_total,
        },
        {
            "hypothesis": "H6_RECENCY_IS_SELF_REFRESH_ARTIFACT",
            "supported": recency_artifact,
            "evidence": "recent5_any_minus_recent5_actual",
            "value": recent_any - recent_actual,
        },
    ]
    _write_rows(output_dir / "mechanism_results.csv", mechanism_rows)
    if l2_steps and l4_steps and l2_total < l4_total and actual_retention_difference <= 0.02:
        primary = "L2_ONLY_REDUCES_TOTAL_AUTONOMOUS_ACTIVITY"
    elif generated > 0.5 and recent_actual < 0.5:
        primary = "SELF_GENERATED_CONTEXT_TAKES_OVER_AUTONOMOUS_ROLLOUT"
    elif recency_artifact:
        primary = "RECENCY_COLLAPSE_IS_MAINLY_A_METRIC_ARTIFACT"
    else:
        primary = "AUTONOMOUS_CONTEXT_PROVENANCE_TRACE_IS_INSUFFICIENT"
    if primary not in PRIMARY_CONCLUSIONS:
        primary = "AUTONOMOUS_CONTEXT_PROVENANCE_TRACE_IS_INSUFFICIENT"
    recommended = "KEEP_L2_AS_EXPERIMENTAL_BASELINE" if {"L2", "L4"}.issubset(labels_present) else "TEST_LONGER_SEQUENCE"
    summary = {
        "diagnostic_only": True,
        "read_only": True,
        "candidate_rows": len(rows),
        "run_labels": labels_present,
        "recent5_any_step_ge_2": recent_any,
        "recent5_actual_step_ge_2": recent_actual,
        "generated_activation_fraction_step_ge_2": generated,
        "recency_metric_self_refresh_artifact": recency_artifact,
        "L2_mean_total_contributors_step_ge_2": l2_total,
        "L4_mean_total_contributors_step_ge_2": l4_total,
        "L2_minus_L4_actual_history_retention": actual_retention_difference,
        "primary_conclusion": primary,
        "recommended_next_step": recommended,
        "online_mechanism_implemented": False,
        "ground_truth_used_for_selection": False,
    }
    (output_dir / "FINAL_AUTONOMOUS_CONTEXT_PROVENANCE_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report = [
        "# Fig.9 Autonomous Context Provenance Report",
        "",
        "This is a read-only post-hoc diagnostic. It does not alter prediction, learning, RNG, or strict defaults.",
        "",
        f"- Candidate rows: `{len(rows)}`",
        f"- Runs: `{', '.join(labels_present)}`",
        f"- Step >= 2 recent-any mean: `{recent_any:.6f}`",
        f"- Step >= 2 recent-actual mean: `{recent_actual:.6f}`",
        f"- Step >= 2 generated activation fraction: `{generated:.6f}`",
        f"- L2/L4 mean total contributors (step >= 2): `{l2_total:.2f}` / `{l4_total:.2f}`",
        f"- L2 minus L4 actual-history retention: `{actual_retention_difference:.6f}`",
        f"- `recent5=1` self-refresh artifact: `{recency_artifact}`",
        f"- Primary conclusion: `{primary}`",
        f"- Recommended next step: `{recommended}`",
        "",
        "## Interpretation",
        "",
        "`recent5_any_fraction` is measured from the current rollout activation, while `recent5_actual_fraction` and `recent5_root_fraction` retain the actual-history reference. They must not be conflated.",
        "",
        "No online mechanism is implemented in this phase.",
    ]
    (output_dir / "FIG9_AUTONOMOUS_CONTEXT_PROVENANCE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze_runs(input_paths=args.input, output_dir=args.output, labels=args.label), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
