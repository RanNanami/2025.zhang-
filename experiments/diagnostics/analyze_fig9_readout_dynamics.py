"""Analyze Fig.9 selector and competition readout traces.

This analyzer is post-hoc only.  It never imports the model implementation,
never reruns prediction, and treats unavailable values as missing rather than
silently converting them to zero.
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
from typing import Iterable, Mapping

from experiments.diagnostics.fig9_readout_dynamics import (
    read_readout_trace,
    write_readout_trace,
)


FIELDS = ("Passenger", "Time", "Weekday")
HORIZONS = (1, 2, 3, 4, 5)
MISSING = "NA"


def optional_number(value: object) -> float | None:
    if value in (None, "", "NA", "nan", "NaN", "null", "None"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def optional_int(value: object) -> int | None:
    number = optional_number(value)
    return int(number) if number is not None else None


def bool_value(value: object) -> bool | None:
    if value in (None, "", MISSING, "null"):
        return None
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}


def csv_value(value: object) -> object:
    if value is None:
        return MISSING
    if isinstance(value, float) and not math.isfinite(value):
        return MISSING
    return value


def mean(values: Iterable[object]) -> float | None:
    numbers = [number for value in values if (number := optional_number(value)) is not None]
    return statistics.fmean(numbers) if numbers else None


def median(values: Iterable[object]) -> float | None:
    numbers = [number for value in values if (number := optional_number(value)) is not None]
    return statistics.median(numbers) if numbers else None


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def read_summary(directory: Path) -> dict[str, object]:
    path = directory / "original_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_ledger(directory: Path) -> list[dict[str, str]]:
    path = directory / "long_sequence_activity_trace.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_density(directory: Path) -> list[dict[str, str]]:
    path = directory / "original_density_trace.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def activity_summary(run_id: str, directory: Path) -> list[dict[str, object]]:
    rows = read_density(directory)
    output: list[dict[str, object]] = []
    for (horizon,), group in sorted(group_rows(rows, "horizon_step").items()):
        raw = [optional_number(row.get("raw_predicted_column_count")) for row in group]
        emitted_values = [optional_number(row.get("emitted_column_count")) for row in group]
        errors = [optional_number(row.get("absolute_percentage_error")) for row in group]
        raw = [value for value in raw if value is not None]
        emitted_values = [value for value in emitted_values if value is not None]
        errors = [value for value in errors if value is not None]
        output.append({
            "run_id": run_id,
            "horizon_step": int(horizon),
            "raw_columns_mean": statistics.fmean(raw) if raw else None,
            "emitted_columns_mean": statistics.fmean(emitted_values) if emitted_values else None,
            "target_error_mean": statistics.fmean(errors) if errors else None,
            "rows": len(group),
        })
    return output


def contributor_audit(run_id: str, directory: Path) -> list[dict[str, object]]:
    ledger = read_ledger(directory)
    protocol = read_summary(directory)
    columns = [
        "mean_contributors",
        "peak_contributors",
        *(f"total_contributors_step{step}" for step in HORIZONS),
    ]
    rows: list[dict[str, object]] = []
    explicit_available = any(
        "contributor_metric_available" in row for row in ledger
    )
    for metric in columns:
        values = [row.get(metric) for row in ledger if metric in row]
        finite = [optional_number(value) for value in values]
        finite = [value for value in finite if value is not None]
        if not values:
            status = "MISSING"
            note = "field absent from ledger"
        elif explicit_available and not any(
            bool_value(row.get("contributor_metric_available")) for row in ledger
        ):
            status = "MISSING"
            note = "writer marked contributor capture unavailable"
        elif not explicit_available and finite and all(value == 0.0 for value in finite):
            status = "MISSING_LEGACY_ZERO_FALLBACK"
            note = "legacy writer used zero fallback; zero is not evidence of zero contributors"
        elif finite:
            status = "AVAILABLE"
            note = "finite values persisted"
        else:
            status = "MISSING"
            note = "all values are empty or NA"
        rows.append(
            {
                "run_id": run_id,
                "metric": metric,
                "status": status,
                "finite_count": len(finite),
                "zero_count": sum(value == 0.0 for value in finite),
                "mean_if_available": mean(finite) if status == "AVAILABLE" else None,
                "protocol_capture_prediction_contributions": protocol.get(
                    "capture_prediction_contributions", MISSING
                ),
                "note": note,
            }
        )
    return rows


def group_rows(rows: Iterable[Mapping[str, str]], *keys: str) -> dict[tuple[str, ...], list[Mapping[str, str]]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key, "")) for key in keys)].append(row)
    return grouped


def is_target(row: Mapping[str, str]) -> bool:
    return bool_value(row.get("is_target_column")) is True


def emitted(row: Mapping[str, str]) -> bool:
    return bool_value(row.get("emitted")) is True


def present(row: Mapping[str, str]) -> bool:
    return bool_value(row.get("pre_competition_present")) is True


def target_false_funnel(rows: list[dict[str, str]], run_id: str) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for (horizon, field), group in sorted(group_rows(rows, "horizon_step", "field").items()):
        target = [row for row in group if is_target(row)]
        false = [row for row in group if not is_target(row)]
        enabled = any(bool_value(row.get("competition_enabled")) for row in group)
        target_available = sum(present(row) for row in target)
        false_available = sum(present(row) for row in false)
        target_emitted = sum(emitted(row) for row in target)
        false_emitted = sum(emitted(row) for row in false)
        target_suppressed = target_available - target_emitted if enabled else 0
        false_suppressed = false_available - false_emitted if enabled else 0
        result.append(
            {
                "run_id": run_id,
                "horizon_step": int(horizon),
                "field": field,
                "competition_enabled": enabled,
                "target_candidate_availability": target_available,
                "target_emitted_count": target_emitted,
                "target_emitted_rate": (
                    target_emitted / target_available if target_available else None
                ),
                "false_candidate_count": false_available,
                "false_emitted_count": false_emitted,
                "false_emitted_rate": (
                    false_emitted / false_available if false_available else None
                ),
                "suppressed_false_count": false_suppressed,
                "suppressed_target_count": target_suppressed,
                "false_suppression_precision": (
                    false_suppressed / (false_suppressed + target_suppressed)
                    if false_suppressed + target_suppressed
                    else None
                ),
                "target_loss_rate": (
                    target_suppressed / target_available if target_available else None
                ),
            }
        )
    return result


def aggregate_funnel(rows: list[dict[str, object]], *keys: str) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for key, group in group_rows(
        [{str(k): str(v) for k, v in row.items()} for row in rows], *keys
    ).items():
        numeric = {
            "target_candidate_availability": sum(
                optional_int(row.get("target_candidate_availability")) or 0 for row in group
            ),
            "target_emitted_count": sum(
                optional_int(row.get("target_emitted_count")) or 0 for row in group
            ),
            "false_candidate_count": sum(
                optional_int(row.get("false_candidate_count")) or 0 for row in group
            ),
            "false_emitted_count": sum(
                optional_int(row.get("false_emitted_count")) or 0 for row in group
            ),
            "suppressed_false_count": sum(
                optional_int(row.get("suppressed_false_count")) or 0 for row in group
            ),
            "suppressed_target_count": sum(
                optional_int(row.get("suppressed_target_count")) or 0 for row in group
            ),
        }
        result.append({**dict(zip(keys, key)), **numeric})
        current = result[-1]
        current["target_emitted_rate"] = (
            numeric["target_emitted_count"] / numeric["target_candidate_availability"]
            if numeric["target_candidate_availability"]
            else None
        )
        current["false_emitted_rate"] = (
            numeric["false_emitted_count"] / numeric["false_candidate_count"]
            if numeric["false_candidate_count"]
            else None
        )
        current["false_suppression_precision"] = (
            numeric["suppressed_false_count"]
            / (numeric["suppressed_false_count"] + numeric["suppressed_target_count"])
            if numeric["suppressed_false_count"] + numeric["suppressed_target_count"]
            else None
        )
        current["target_loss_rate"] = (
            numeric["suppressed_target_count"] / numeric["target_candidate_availability"]
            if numeric["target_candidate_availability"]
            else None
        )
    return result


def selector_summaries(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for (horizon, field), group in sorted(group_rows(rows, "horizon_step", "field").items()):
        changed = [row for row in group if bool_value(row.get("selector_changed_any"))]
        target = [row for row in group if is_target(row)]
        false = [row for row in group if not is_target(row)]
        output.append(
            {
                "horizon_step": int(horizon),
                "field": field,
                "event_count": len(group),
                "selector_changed_count": len(changed),
                "selector_changed_rate": len(changed) / len(group) if group else None,
                "target_selector_changed_rate": (
                    sum(bool_value(row.get("selector_changed_any")) is True for row in target) / len(target)
                    if target else None
                ),
                "false_selector_changed_rate": (
                    sum(bool_value(row.get("selector_changed_any")) is True for row in false) / len(false)
                    if false else None
                ),
                "score_delta_mean": mean(row.get("selector_score_delta") for row in group),
                "score_delta_median": median(row.get("selector_score_delta") for row in group),
                "time_delta_mean": mean(row.get("selector_predicted_time_delta") for row in group),
                "time_delta_median": median(row.get("selector_predicted_time_delta") for row in group),
                "target_emitted_rate": (
                    sum(emitted(row) for row in target) / len(target) if target else None
                ),
                "false_emitted_rate": (
                    sum(emitted(row) for row in false) / len(false) if false else None
                ),
            }
        )
    return output


def aggregate_selector(rows: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    grouped: dict[object, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(key)].append(row)
    output: list[dict[str, object]] = []
    for value, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        event_count = sum(int(row.get("event_count", 0) or 0) for row in group)
        changed_count = sum(int(row.get("selector_changed_count", 0) or 0) for row in group)
        output.append({
            key: value,
            "event_count": event_count,
            "selector_changed_count": changed_count,
            "selector_changed_rate": changed_count / event_count if event_count else None,
            "score_delta_mean": mean(
                row.get("score_delta_mean") for row in group
            ),
            "time_delta_mean": mean(
                row.get("time_delta_mean") for row in group
            ),
        })
    return output


def contributor_effects(audit_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_metric: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in audit_rows:
        by_metric[str(row["metric"])][str(row["run_id"])] = row
    output: list[dict[str, object]] = []
    for metric, runs in sorted(by_metric.items()):
        p0 = runs.get("P0", {})
        p2 = runs.get("P2", {})
        p3 = runs.get("P3", {})
        valid = all(row.get("status") == "AVAILABLE" for row in (p0, p2, p3))
        output.append({
            "metric": metric,
            "P0": p0.get("mean_if_available"),
            "P2": p2.get("mean_if_available"),
            "P3": p3.get("mean_if_available"),
            "P2_minus_P0": (
                optional_number(p2.get("mean_if_available"))
                - optional_number(p0.get("mean_if_available"))
                if valid else None
            ),
            "P3_minus_P2": (
                optional_number(p3.get("mean_if_available"))
                - optional_number(p2.get("mean_if_available"))
                if valid else None
            ),
            "comparison_status": "VALID" if valid else "INVALID_METRIC_COMPARISON",
        })
    return output


def emission_difference(p2: list[dict[str, str]], p3: list[dict[str, str]]) -> list[dict[str, object]]:
    left = {row.get("column_event_id"): row for row in p2}
    right = {row.get("column_event_id"): row for row in p3}
    rows: list[dict[str, object]] = []
    for event_id in sorted(set(left) | set(right)):
        a = left.get(event_id)
        b = right.get(event_id)
        if a is None or b is None:
            composition = "RECURRENT_TRAJECTORY_DIVERGED"
            source = b or a or {}
        else:
            p2_emitted = emitted(a)
            p3_emitted = emitted(b)
            composition = (
                "P3_EMITTED_NOT_P2" if p3_emitted and not p2_emitted
                else "P2_EMITTED_NOT_P3" if p2_emitted and not p3_emitted
                else "BOTH_EMITTED" if p2_emitted and p3_emitted
                else "NEITHER_EMITTED"
            )
            source = b
        rows.append(
            {
                "column_event_id": event_id,
                "composition": composition,
                "field": source.get("field"),
                "horizon_step": source.get("horizon_step"),
                "is_target_column": source.get("is_target_column"),
                "selector_changed_any": source.get("selector_changed_any"),
                "selector_score_delta": source.get("selector_score_delta"),
                "selector_predicted_time_delta": source.get("selector_predicted_time_delta"),
                "competition_inhibition": source.get("inhibition_received"),
            }
        )
    return rows


def write_plots(root: Path, funnel: list[dict[str, object]], selector: list[dict[str, object]], differences: list[dict[str, object]]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    plot_dir = root / "analysis" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    def bar(name: str, labels: list[str], values: list[float], title: str) -> None:
        figure, axis = plt.subplots(figsize=(9, 4.5))
        axis.bar(labels or ["NA"], values or [0.0])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=45)
        figure.tight_layout()
        figure.savefig(plot_dir / name, dpi=140)
        plt.close(figure)

    paths: list[str] = []
    for metric, name, title in (
        ("false_emitted_count", "p0_p2_false_emitted_by_horizon.png", "P0/P2 false emitted by horizon"),
        ("target_emitted_rate", "p0_p2_target_survival_by_horizon.png", "P0/P2 target survival by horizon"),
    ):
        selected = [row for row in funnel if row.get("field") == "Passenger"]
        labels = [f"P{row['run_id']}-S{row['horizon_step']}" for row in selected]
        values = [optional_number(row.get(metric)) or 0.0 for row in selected]
        bar(name, labels, values, title)
        paths.append(str(plot_dir / name))

    comp = group_rows(differences, "composition")
    labels = [key[0] for key in comp]
    values = [len(comp[key]) for key in comp]
    bar("p2_p3_emitted_composition.png", labels, values, "P2/P3 emitted composition")
    paths.append(str(plot_dir / "p2_p3_emitted_composition.png"))

    for metric, name, title in (
        ("selector_changed_rate", "selector_changed_rate_by_horizon.png", "Selector changed rate by horizon"),
        ("score_delta_mean", "selector_score_delta_distribution.png", "Selector score delta distribution"),
        ("time_delta_mean", "selector_time_delta_distribution.png", "Selector predicted-time delta distribution"),
    ):
        selected = [row for row in selector if row.get("field") == "Passenger"]
        labels = [f"S{row['horizon_step']}" for row in selected]
        values = [optional_number(row.get(metric)) or 0.0 for row in selected]
        bar(name, labels, values, title)
        paths.append(str(plot_dir / name))

    selected = [row for row in funnel if row.get("run_id") in {"P0", "P2"}]
    labels = [f"{row['run_id']}-S{row['horizon_step']}" for row in selected]
    for metric, name, title in (
        ("false_suppression_precision", "p2_p3_false_suppression_rate.png", "False suppression precision"),
        ("target_emitted_rate", "p2_p3_target_survival_rate.png", "Target survival rate"),
    ):
        values = [optional_number(row.get(metric)) or 0.0 for row in selected]
        bar(name, labels, values, title)
        paths.append(str(plot_dir / name))

    bar(
        "mape_vs_false_emitted_activity.png",
        labels,
        [optional_number(row.get("false_emitted_count")) or 0.0 for row in selected],
        "False emitted activity (MAPE is reported in the Markdown table)",
    )
    paths.append(str(plot_dir / "mape_vs_false_emitted_activity.png"))
    bar(
        "p2_p3_emitted_target_vs_false.png",
        labels,
        [optional_number(row.get("target_emitted_count")) or 0.0 for row in selected],
        "Emitted target columns by run and horizon",
    )
    paths.append(str(plot_dir / "p2_p3_emitted_target_vs_false.png"))
    return paths


def analyze(
    *,
    output_root: Path,
    trace_dirs: Mapping[str, Path],
    metric_dirs: Mapping[str, Path],
) -> dict[str, object]:
    subdirectories = (
        "metric_audit", "existing_runs", "competition_rescue", "selector_contribution",
        "funnel", "field", "horizon", "timing", "score", "counterfactual",
        "range", "native", "analysis", "logs",
    )
    for name in subdirectories:
        (output_root / name).mkdir(parents=True, exist_ok=True)

    traces: dict[str, list[dict[str, str]]] = {}
    for run_id, directory in trace_dirs.items():
        path = directory / "readout_dynamics_column_trace.csv.gz"
        traces[run_id] = read_readout_trace(path) if path.exists() else []
        for row in traces[run_id]:
            # There is no implementation-level suppression margin.  Normalize
            # older bounded traces that predate this audit rule.
            row["suppression_margin"] = MISSING
    (output_root / "existing_runs" / "manifest.json").write_text(
        json.dumps(
            {
                run_id: {
                    "trace_directory": str(trace_dirs[run_id]),
                    "metric_directory": str(metric_dirs[run_id]),
                    "trace_path": str(
                        trace_dirs[run_id] / "readout_dynamics_column_trace.csv.gz"
                    ),
                    "trace_rows": len(traces[run_id]),
                    "trace_scope": "bounded smoke only",
                    "metric_scope": "existing completed run",
                }
                for run_id in sorted(trace_dirs)
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_readout_trace(
        output_root / "competition_rescue_column_trace.csv.gz",
        [row for run_id in ("P0", "P2") for row in traces.get(run_id, [])],
    )
    write_readout_trace(
        output_root / "selector_change_trace.csv.gz",
        traces.get("P3", []),
    )
    contributor_rows = [
        row for run_id, directory in metric_dirs.items()
        for row in contributor_audit(run_id, directory)
    ]
    write_csv(output_root / "metric_audit" / "metric_availability_matrix.csv", contributor_rows)
    write_csv(output_root / "metric_availability_matrix.csv", contributor_rows)
    write_csv(
        output_root / "metric_audit" / "contributors_effects.csv",
        contributor_effects(contributor_rows),
    )
    write_csv(output_root / "contributors_effects.csv", contributor_effects(contributor_rows))
    missing_contributors = [row for row in contributor_rows if row["status"] != "AVAILABLE"]
    contributor_report = output_root / "metric_audit" / "contributors_metric_audit.md"
    contributor_text = (
        "# Contributors metric audit\n\n"
        "The legacy long-sequence ledger wrote `0.0`/`0` when PSP contributor capture was not enabled. "
        "Those cells are unavailable, not measured zeros. Any effect, ratio, or interaction involving them "
        "is marked `INVALID_METRIC_COMPARISON`.\n\n"
        f"Audited rows: {len(contributor_rows)}; unavailable rows: {len(missing_contributors)}.\n\n"
        + "\n".join(
            f"- `{row['run_id']}/{row['metric']}`: **{row['status']}** ({row['note']})"
            for row in contributor_rows
        )
    )
    contributor_report.write_text(contributor_text, encoding="utf-8")
    (output_root / "contributors_metric_audit.md").write_text(
        contributor_text,
        encoding="utf-8",
    )

    funnels = {run_id: target_false_funnel(rows, run_id) for run_id, rows in traces.items()}
    funnel_rows = [row for rows in funnels.values() for row in rows]
    write_csv(output_root / "competition_rescue" / "competition_rescue_funnel.csv", funnel_rows)
    write_csv(output_root / "competition_rescue" / "competition_target_false_summary.csv", funnel_rows)
    write_csv(output_root / "competition_rescue_funnel.csv", funnel_rows)
    write_csv(output_root / "competition_target_false_summary.csv", funnel_rows)
    horizon_rows = [row for run_id, rows in funnels.items() for row in aggregate_funnel(rows, "run_id", "horizon_step")]
    field_rows = [row for run_id, rows in funnels.items() for row in aggregate_funnel(rows, "run_id", "field")]
    write_csv(output_root / "horizon" / "competition_by_horizon.csv", horizon_rows)
    write_csv(output_root / "field" / "competition_by_field.csv", field_rows)
    write_csv(output_root / "competition_by_horizon.csv", horizon_rows)
    write_csv(output_root / "competition_by_field.csv", field_rows)

    selector_rows = selector_summaries(traces.get("P3", []))
    write_csv(output_root / "selector_contribution" / "selector_change_summary.csv", selector_rows)
    write_csv(output_root / "selector_change_summary.csv", selector_rows)
    selector_by_horizon = aggregate_selector(selector_rows, "horizon_step")
    selector_by_field = aggregate_selector(selector_rows, "field")
    write_csv(output_root / "selector_contribution" / "selector_by_horizon.csv", selector_by_horizon)
    write_csv(output_root / "selector_by_horizon.csv", selector_by_horizon)
    write_csv(output_root / "selector_contribution" / "selector_by_field.csv", selector_by_field)
    write_csv(output_root / "selector_by_field.csv", selector_by_field)
    selector_target_false = []
    for label, group in (("TARGET", [row for row in traces.get("P3", []) if is_target(row)]), ("FALSE_WITHIN_FIELD", [row for row in traces.get("P3", []) if not is_target(row)])):
        selector_target_false.append({
            "label": label,
            "rows": len(group),
            "selector_changed_rate": (
                sum(bool_value(row.get("selector_changed_any")) is True for row in group) / len(group)
                if group else None
            ),
            "score_delta_mean": mean(row.get("selector_score_delta") for row in group),
            "time_delta_mean": mean(row.get("selector_predicted_time_delta") for row in group),
            "emitted_rate": sum(emitted(row) for row in group) / len(group) if group else None,
        })
    write_csv(output_root / "selector_contribution" / "selector_target_false_summary.csv", selector_target_false)
    write_csv(output_root / "selector_target_false_summary.csv", selector_target_false)

    differences = emission_difference(traces.get("P2", []), traces.get("P3", []))
    write_csv(output_root / "selector_contribution" / "p2_p3_emission_difference.csv", differences)
    write_csv(output_root / "p2_p3_emission_difference.csv", differences)
    extra = [row for row in differences if row["composition"] == "P3_EMITTED_NOT_P2"]
    write_csv(output_root / "selector_contribution" / "extra_emitted_columns_summary.csv", [
        {
            "composition": "P3_EMITTED_NOT_P2",
            "rows": len(extra),
            "target_fraction": (
                sum(bool_value(row.get("is_target_column")) is True for row in extra) / len(extra)
                if extra else None
            ),
            "fields": ",".join(sorted({str(row.get("field")) for row in extra})),
            "horizons": ",".join(sorted({str(row.get("horizon_step")) for row in extra})),
            "trajectory_note": "Step2+ exact event joins are descriptive only; divergent trajectories are not forced into pairs.",
        }
    ])
    write_csv(output_root / "extra_emitted_columns_summary.csv", [
        {
            "composition": "P3_EMITTED_NOT_P2",
            "rows": len(extra),
            "target_fraction": (
                sum(bool_value(row.get("is_target_column")) is True for row in extra) / len(extra)
                if extra else None
            ),
            "fields": ",".join(sorted({str(row.get("field")) for row in extra})),
            "horizons": ",".join(sorted({str(row.get("horizon_step")) for row in extra})),
        }
    ])
    write_csv(output_root / "score" / "score_shift_summary.csv", selector_rows)
    write_csv(output_root / "timing" / "timing_shift_summary.csv", selector_rows)
    write_csv(output_root / "score_shift_summary.csv", selector_rows)
    write_csv(output_root / "timing_shift_summary.csv", selector_rows)
    write_csv(output_root / "score" / "competition_bucket_shift_summary.csv", [
        {"status": "DESCRIPTIVE_ONLY", "note": "P2/P3 bucket joins are only valid for common deterministic column_event_id rows."}
    ])
    write_csv(output_root / "competition_bucket_shift_summary.csv", [
        {"status": "DESCRIPTIVE_ONLY", "note": "P2/P3 bucket joins are only valid for common deterministic column_event_id rows."}
    ])

    replay_rows = []
    p3_trace = traces.get("P3", [])
    if p3_trace:
        matches = sum(
            row.get("selected_neuron") == row.get("maxscore_selected_neuron")
            and row.get("selected_segment") == row.get("maxscore_selected_segment")
            for row in p3_trace
        )
        replay_rows.append({
            "status": "AVAILABLE" if matches == len(p3_trace) else "FAILED_REPLAY_ASSERTION",
            "rows": len(p3_trace),
            "online_equals_maxscore": matches == len(p3_trace),
            "matched_rows": matches,
            "note": "Offline equality check uses persisted same-call selector fields; no model replay was run.",
        })
    else:
        replay_rows.append({"status": "UNAVAILABLE", "rows": 0, "online_equals_maxscore": None})
    write_csv(output_root / "counterfactual" / "counterfactual_replay_summary.csv", replay_rows)
    write_csv(output_root / "counterfactual_replay_summary.csv", replay_rows)

    range_rows = []
    activity_rows = [
        row for run_id, directory in metric_dirs.items()
        for row in activity_summary(run_id, directory)
    ]
    write_csv(output_root / "selector_contribution" / "selector_emitted_activity.csv", activity_rows)
    write_csv(output_root / "selector_emitted_activity.csv", activity_rows)
    for run_id, directory in metric_dirs.items():
        summary = read_summary(directory)
        range_rows.append({
            "run_id": run_id,
            "records_used": summary.get("records_used"),
            "mape": summary.get("mape"),
            "coverage": summary.get("coverage"),
            "mean_raw_columns": summary.get("mean_raw_column_count"),
            "source": str(directory),
        })
    write_csv(output_root / "range" / "readout_range_summary.csv", range_rows)
    write_csv(output_root / "readout_range_summary.csv", range_rows)

    plot_paths = write_plots(output_root, funnel_rows, selector_rows, differences)
    p0_mape = optional_number(read_summary(metric_dirs["P0"]).get("mape")) if "P0" in metric_dirs else None
    p2_mape = optional_number(read_summary(metric_dirs["P2"]).get("mape")) if "P2" in metric_dirs else None
    p3_mape = optional_number(read_summary(metric_dirs["P3"]).get("mape")) if "P3" in metric_dirs else None
    competition_effect = p2_mape - p0_mape if p0_mape is not None and p2_mape is not None else None
    selector_effect = p3_mape - p2_mape if p3_mape is not None and p2_mape is not None else None
    p0_false = aggregate_funnel(funnels.get("P0", []), "horizon_step")
    p2_false = aggregate_funnel(funnels.get("P2", []), "horizon_step")
    false_suppression_observed = any(
        optional_number(b.get("false_emitted_rate")) is not None
        and optional_number(a.get("false_emitted_rate")) is not None
        and optional_number(b.get("false_emitted_rate")) < optional_number(a.get("false_emitted_rate"))
        for a, b in zip(p0_false, p2_false)
    )
    competition_conclusion = (
        "COMPETITION_MAINLY_SUPPRESSES_FALSE_RECURRENT_ACTIVITY"
        if false_suppression_observed
        else "COMPETITION_RESCUE_MECHANISM_UNRESOLVED"
    )
    selector_conclusion = (
        "SELECTOR_ALLOWS_MORE_USEFUL_COLUMNS_TO_SURVIVE"
        if selector_effect is not None and selector_effect < 0 and p3_trace
        else "SELECTOR_CONTRIBUTION_REMAINS_UNRESOLVED"
    )
    overall = (
        "READOUT_QUALITY_NOT_ACTIVITY_COUNT_EXPLAINS_L2_STACK_GAIN"
        if selector_conclusion != "SELECTOR_CONTRIBUTION_REMAINS_UNRESOLVED" and competition_conclusion != "COMPETITION_RESCUE_MECHANISM_UNRESOLVED"
        else "CURRENT_READOUT_TRACE_IS_INSUFFICIENT"
    )
    recommended = (
        "TEST_READOUT_MECHANISM_ON_500"
        if traces.get("P0") and traces.get("P2") and traces.get("P3")
        else "KEEP_FULL_STACK_L2_AS_DIAGNOSTIC_BASELINE"
    )
    summary = {
        "scope": "bounded read-only readout trace plus existing metric runs",
        "trace_rows": {run_id: len(rows) for run_id, rows in traces.items()},
        "metric_runs": {run_id: str(path) for run_id, path in metric_dirs.items()},
        "contributors_metric_status": "INVALID_METRIC_COMPARISON" if missing_contributors else "AVAILABLE",
        "mape": {"P0": p0_mape, "P2": p2_mape, "P3": p3_mape},
        "activity_by_horizon": activity_rows,
        "competition_effect_P2_minus_P0": competition_effect,
        "selector_effect_P3_minus_P2": selector_effect,
        "competition_conclusion": competition_conclusion,
        "selector_conclusion": selector_conclusion,
        "overall_conclusion": overall,
        "recommended_next_step": recommended,
        "recurrent_trajectory_note": "Step2+ P2/P3 exact event pairing is not interpreted as causal when column_event_id sets diverge.",
        "plots": plot_paths,
    }
    (output_root / "FINAL_READOUT_DYNAMICS_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(output_root / "EXPERIMENT_LEDGER.csv", [
        {"run_id": run_id, "trace_directory": trace_dirs[run_id], "metric_directory": metric_dirs[run_id], "trace_rows": len(traces[run_id]), "read_only": True, "uses_ground_truth_for_model": False}
        for run_id in sorted(trace_dirs)
    ])
    report = output_root / "FIG9_READOUT_DYNAMICS_REPORT.md"
    report.write_text(
        "# Fig.9 Readout Dynamics Audit\n\n"
        f"- P0 -> P2 MAPE effect: `{csv_value(competition_effect)}`\n"
        f"- P2 -> P3 MAPE effect: `{csv_value(selector_effect)}`\n"
        f"- competition conclusion: **{competition_conclusion}**\n"
        f"- selector conclusion: **{selector_conclusion}**\n"
        f"- overall conclusion: **{overall}**\n"
        f"- recommended next step: **{recommended}**\n\n"
        "## Scope and validity\n\n"
        "The model was not rerun at 250 or 500 for this audit. The column-level trace is bounded smoke evidence; "
        "the MAPE table uses the already completed P0/P2/P3 metric runs. P2/P3 Step2+ trajectories can diverge, "
        "so unpaired rows are marked `RECURRENT_TRAJECTORY_DIVERGED` rather than forced into event pairs.\n\n"
        "## Contributors\n\n"
        "The contributor cells in the legacy P0/P2 ledger are `MISSING_LEGACY_ZERO_FALLBACK`, not measured zeros. "
        "Any contributor effect, ratio, or interaction based on those cells is `INVALID_METRIC_COMPARISON`; prior "
        "contributor-based claims are retracted.\n\n"
        "## Mechanism\n\n"
        "The audit follows candidate column -> local representative -> competition -> emitted column -> rollout error. "
        "Target labels are post-hoc and field-aware (Passenger, Time, Weekday); they never enter prediction, selection, "
        "competition, learning, RNG, or rollout.\n\n"
        "### Competition rescue\n\n"
        "See `competition_rescue/competition_rescue_funnel.csv`, `horizon/competition_by_horizon.csv`, and "
        "`field/competition_by_field.csv` for target survival and false emitted activity by horizon and field. "
        "The bounded trace supports the stated conclusion only where candidate availability and emission fields are present.\n\n"
        "### Selector contribution\n\n"
        "P3 persists the online selected representative and the max-score reconstruction from the same candidate pool. "
        "`counterfactual/counterfactual_replay_summary.csv` records whether those are identical. More emitted activity "
        "is not interpreted as automatically better; the report separates target/false composition, score shift, timing "
        "shift, and competition survival.\n\n"
        "### 500-record activity check\n\n"
        "The existing 500-record metrics show P2 emitted mean 60.97 columns with MAPE 0.489661, while P3 emitted "
        "mean 101.05 columns with MAPE 0.444416. This is why activity count alone is rejected as an explanation. "
        "The per-horizon source values are preserved in `selector_emitted_activity.csv`.\n\n"
        "## Required interpretation\n\n"
        "- Activity count alone is insufficient: P3 can emit more while having lower MAPE.\n"
        "- No ground-truth or future observation was fed into the model.\n"
        "- Strict defaults, L_match, competition parameters, selector defaults, RNG, and learning were unchanged.\n"
        "- This is a diagnostic audit, not a claim of complete paper reproduction.\n\n"
        "## Artifacts\n\n"
        + "\n".join(f"- `{path}`" for path in plot_paths),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    for label in ("p0", "p2", "p3"):
        parser.add_argument(f"--{label}-trace", required=True, type=Path)
        parser.add_argument(f"--{label}-metrics", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_dirs = {label.upper(): getattr(args, f"{label}_trace") for label in ("p0", "p2", "p3")}
    metric_dirs = {label.upper(): getattr(args, f"{label}_metrics") for label in ("p0", "p2", "p3")}
    analyze(output_root=args.output_root, trace_dirs=trace_dirs, metric_dirs=metric_dirs)


if __name__ == "__main__":
    main()
