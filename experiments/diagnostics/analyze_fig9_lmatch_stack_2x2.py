"""Protocol audit and offline analysis for the Fig.9 L_match x stack study.

The four cells are deliberately described here as an auditable matrix.  This
module never loads a model and never changes a prediction.  It reads completed
run artifacts, reuses compatible historical runs, and computes descriptive
factor effects from the recorded predictions and activity ledgers.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
from typing import Any


CELL_NAMES = (
    "STRICT_RAW_L4",
    "STRICT_RAW_L2",
    "DIAGNOSTIC_STACK_L4",
    "DIAGNOSTIC_STACK_L2",
)

COMMON_FIELDS = (
    "cell",
    "stack",
    "L_match",
    "competition_mode",
    "simultaneous_policy",
    "inhibition_strength",
    "inhibition_tau",
    "simultaneous_bin_width",
    "intracolumn_selector",
    "propagation_mode",
    "continuous_impl",
    "burst_context",
    "prediction_horizon",
    "RNG_seed",
    "warmup",
    "stream",
    "forgetting_threshold",
    "K",
    "neurons_per_column",
    "learning_behavior",
    "rollout_learning",
    "future_covariates",
    "compensation",
    "decoded_replay",
    "strict_reproduction",
    "lmatch_real_ablation",
    "diagnostic_stack_active",
    "competition_is_local_choice",
    "intracolumn_selector_is_local_choice",
    "label_note",
)


def _strict_defaults() -> dict[str, Any]:
    # Importing the config keeps this audit tied to the real strict source.
    from experiments.fig9_strict_reproduction import Fig9StrictConfig

    config = Fig9StrictConfig()
    return {
        "propagation_mode": config.propagation_mode,
        "continuous_impl": config.continuous_prediction_impl,
        "burst_context": "all-cell" if config.burst_context else "winner-only",
        "prediction_horizon": config.horizon,
        "RNG_seed": config.seed,
        "forgetting_threshold": config.forgetting_threshold,
        "K": config.k,
        "neurons_per_column": config.neurons_per_column,
        "learning_behavior": f"{config.scenario1_rule}; one-pass online",
        "rollout_learning": config.rollout_learning,
        "future_covariates": config.use_future_covariates,
        "compensation": False,
        "decoded_replay": config.reencode_decoded_value,
        # Formal 250/500 runs explicitly use warmup=200.
        "warmup": 200,
        "stream": "original",
    }


def protocol_matrix() -> list[dict[str, Any]]:
    common = _strict_defaults()
    rows: list[dict[str, Any]] = []
    for cell, l_match, stack in (
        ("STRICT_RAW_L4", 4, "STRICT_RAW"),
        ("STRICT_RAW_L2", 2, "STRICT_RAW"),
        ("DIAGNOSTIC_STACK_L4", 4, "DIAGNOSTIC_STACK"),
        ("DIAGNOSTIC_STACK_L2", 2, "DIAGNOSTIC_STACK"),
    ):
        is_diag = stack == "DIAGNOSTIC_STACK"
        is_default = cell == "STRICT_RAW_L4"
        row = {
            **common,
            "cell": cell,
            "stack": stack,
            "L_match": l_match,
            "competition_mode": "competitive_raw" if is_diag else "off",
            "simultaneous_policy": "batched" if is_diag else "sequential",
            "inhibition_strength": 0.1 if is_diag else "inactive",
            "inhibition_tau": 0.02 if is_diag else "inactive",
            "simultaneous_bin_width": 0.005 if is_diag else "inactive",
            "intracolumn_selector": "max_candidate_score" if is_diag else "existing",
            "strict_reproduction": is_default,
            "lmatch_real_ablation": not is_default,
            "diagnostic_stack_active": is_diag,
            "competition_is_local_choice": is_diag,
            "intracolumn_selector_is_local_choice": is_diag,
            "label_note": (
                "strict reference"
                if is_default
                else "single-factor L_match ablation on strict/raw path"
                if stack == "STRICT_RAW"
                else "nonpaper diagnostic stack"
            ),
        }
        rows.append({field: row.get(field, "") for field in COMMON_FIELDS})
    return rows


def _write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fields:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_protocol_audit(output_dir: Path, docs_dir: Path) -> None:
    rows = protocol_matrix()
    _write_rows(output_dir / "2x2_protocol_matrix.csv", rows, list(COMMON_FIELDS))
    docs_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fig.9 L_match x Diagnostic-Stack Protocol Audit",
        "",
        "This audit is generated from the strict configuration source and the explicit cell definitions.",
        "It is a protocol document, not an experimental result.",
        "",
        "## Source Path",
        "",
        "`experiments/fig9_strict_reproduction.py::Fig9StrictConfig` is the source of the common strict values.",
        "The strict/raw path uses `competition_mode=off`, `simultaneous_policy=sequential`, and `intracolumn_selection_policy=existing`.",
        "The diagnostic stack is enabled only by explicit CLI settings: `competitive_raw`, `batched`, and `max_candidate_score`.",
        "",
        "## Matrix",
        "",
        "| Cell | L_match | Stack | Competition | Simultaneous | Selector | Strict reproduction |",
        "|---|---:|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['cell']} | {row['L_match']} | {row['stack']} | "
            f"{row['competition_mode']} | {row['simultaneous_policy']} | "
            f"{row['intracolumn_selector']} | {row['strict_reproduction']} |"
        )
    lines.extend(
        [
            "",
            "## Isolation Checks",
            "",
            "- STRICT_RAW_L4 vs STRICT_RAW_L2 differs in model mechanism only by `L_match: 4 -> 2`; the L2 row is explicitly non-strict.",
            "- DIAGNOSTIC_STACK_L4 vs DIAGNOSTIC_STACK_L2 differs in model mechanism only by `L_match: 4 -> 2`.",
            "- STRICT_RAW_L4 vs DIAGNOSTIC_STACK_L4 differs by the explicitly labeled experimental competition/selection stack.",
            "- STRICT_RAW_L2 vs DIAGNOSTIC_STACK_L2 has the same stack distinction and keeps the same L2 ablation label.",
            "- All cells use raw propagation, continuous reference dynamics, all-cell burst context, K=10, 32 neurons/column, forgetting threshold 65, seed 0, original stream, and no rollout learning.",
            "- No cell uses compensation, future covariates, ground-truth selection, decoded replay, or target correction.",
            "",
            "## Naming Rule",
            "",
            "`STRICT_RAW_L4` is the strict reference. `STRICT_RAW_L2` is a single-factor nonpaper ablation, never a strict paper reproduction.",
        ]
    )
    (docs_dir / "FIG9_LMATCH_STACK_2X2_PROTOCOL_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _prediction_sha(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_predictions(limit: int, warmup: int, horizon: int) -> int:
    return max(0, limit - warmup - horizon)


def _native_attempts(run_dir: Path) -> tuple[int, int, int]:
    attempts = list(run_dir.glob("process_*.json"))
    native = 0
    resumes = 0
    for path in attempts:
        payload = _json(path)
        code = int(payload.get("exit_code", 0) or 0)
        native += code == 3221225477
        resumes += code != 0
    return len(attempts), native, resumes


def _native_attempts_for_paths(paths: list[Path]) -> tuple[int, int, int]:
    """Combine process histories when a completed run was resumed elsewhere."""

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(path)
    attempts = native = resumes = 0
    for path in unique_paths:
        current_attempts, current_native, current_resumes = _native_attempts(path)
        attempts += current_attempts
        native += current_native
        resumes += current_resumes
    return attempts, native, resumes


def inventory_runs(root: Path, output_dir: Path, docs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.rglob("original_summary.json")):
        run_dir = summary_path.parent
        summary = _json(summary_path)
        if summary.get("L_match") not in (2, 4):
            continue
        runtime = _json(run_dir / "runtime.json")
        protocol = _json(run_dir / "protocol.json")
        merged = {**protocol, **runtime, **summary}
        limit = int(merged.get("limit", merged.get("records_used", 0)) or 0)
        warmup = int(merged.get("warmup", merged.get("warmup_length", 0)) or 0)
        horizon = int(merged.get("prediction_horizon", merged.get("autonomous_rollout_steps", 5)) or 5)
        expected = _expected_predictions(limit, warmup, horizon)
        actual = int(summary.get("predictions", 0) or 0)
        attempts, native, resumes = _native_attempts(run_dir)
        competition = merged.get("competition", {})
        mode = competition.get("mode", merged.get("competition_mode", "off")) if isinstance(competition, dict) else merged.get("competition_mode", "off")
        complete = actual == expected and _number(summary.get("coverage")) == 1.0
        row = {
            "run_directory": str(run_dir),
            "git_commit": merged.get("git_commit_sha", ""),
            "limit": limit,
            "warmup": warmup,
            "L_match": merged.get("L_match", ""),
            "stack": "DIAGNOSTIC_STACK" if mode == "competitive_raw" else "STRICT_RAW",
            "selector": "max_candidate_score" if mode == "competitive_raw" else "existing",
            "competition": mode,
            "continuous_impl": merged.get("continuous_impl", ""),
            "seed": merged.get("tie_break_seed", merged.get("RNG_seeds", {}).get("tie_break_seed", "")),
            "data_sha": merged.get("data_file_sha256", ""),
            "expected_predictions": expected,
            "actual_predictions": actual,
            "coverage": summary.get("coverage", ""),
            "MAPE": summary.get("mape", ""),
            "prediction_sha": _prediction_sha(run_dir / "original_predictions.csv"),
            "model_fingerprint": summary.get("final_model_fingerprint", ""),
            "RNG_fingerprint": summary.get("final_rng_fingerprint", ""),
            "complete": complete,
            "native_crash_count": native,
            "resume_count": resumes,
            "attempt_count": attempts,
            "reuse_eligible": complete and limit in (250, 500) and merged.get("continuous_impl", "") == "reference",
        }
        rows.append(row)
    fields = list(rows[0]) if rows else []
    _write_rows(output_dir / "existing_run_inventory.csv", rows, fields)
    eligible = [row for row in rows if row["reuse_eligible"]]
    docs_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Existing Fig.9 Run Inventory",
        "",
        f"Scanned `{root}`. Found {len(rows)} run artifacts and {len(eligible)} complete reference-dynamics candidates.",
        "Only rows with expected prediction count and coverage=1 are marked `reuse_eligible`; protocol stack still requires the explicit audit below.",
        "",
        "| L_match | limit | stack | MAPE | coverage | predictions | reusable | run directory |",
        "|---:|---:|---|---:|---:|---:|---|---|",
    ]
    for row in eligible:
        lines.append(
            f"| {row['L_match']} | {row['limit']} | {row['stack']} | {row['MAPE']} | "
            f"{row['coverage']} | {row['actual_predictions']}/{row['expected_predictions']} | "
            f"{row['reuse_eligible']} | `{row['run_directory']}` |"
        )
    (docs_dir / "EXISTING_RUN_INVENTORY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return rows


def _read_activity(run_dir: Path) -> tuple[dict[int, dict[str, Any]], dict[int, dict[int, dict[str, Any]]]]:
    activity: dict[int, dict[str, Any]] = {}
    steps: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    path = run_dir / "long_sequence_activity_trace.csv"
    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                index = int(row.get("record_index", 0))
                activity[index] = row
                for step in range(1, 6):
                    steps[index][step] = {
                        "error": row.get(f"prediction_error_step{step}", ""),
                        "raw": row.get(f"raw_predicted_columns_step{step}", ""),
                        "emitted": row.get(f"emitted_columns_step{step}", ""),
                        "contributors": row.get(f"total_contributors_step{step}", ""),
                    }
        return activity, steps
    path = run_dir / "original_density_trace.csv"
    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                index = int(row.get("record_index", 0))
                step = int(row.get("horizon_step", 0))
                if step not in range(1, 6):
                    continue
                steps[index][step] = {
                    "error": row.get("absolute_percentage_error", ""),
                    "raw": row.get("raw_predicted_column_count", ""),
                    "emitted": row.get("emitted_column_count", ""),
                    "contributors": "",
                }
    return activity, steps


def _step_values(steps: dict[int, dict[int, dict[str, Any]]], key: str, step: int | None = None) -> list[float]:
    values: list[float] = []
    for row in steps.values():
        selected = [row.get(step, {})] if step else list(row.values())
        for item in selected:
            value = item.get(key, "")
            if value not in ("", None):
                values.append(_number(value))
    return values


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _run_metrics(run_dir: Path) -> dict[str, Any]:
    summary = _json(run_dir / "original_summary.json")
    runtime = _json(run_dir / "runtime.json")
    protocol = _json(run_dir / "protocol.json")
    merged = {**protocol, **runtime, **summary}
    limit = int(merged.get("limit", merged.get("records_used", 0)) or 0)
    warmup = int(merged.get("warmup", merged.get("warmup_length", 0)) or 0)
    horizon = int(merged.get("prediction_horizon", merged.get("autonomous_rollout_steps", 5)) or 5)
    activity, steps = _read_activity(run_dir)
    stepwise = {f"step{step}": _mean(_step_values(steps, "error", step)) for step in range(1, 6)}
    raw = _step_values(steps, "raw")
    emitted = _step_values(steps, "emitted")
    contributors = _step_values(steps, "contributors")
    scenario = {
        f"scenario{number}_mean": _mean([_number(row.get(f"scenario{number}_count", "")) for row in activity.values()])
        for number in range(1, 4)
    }
    rss = [_number(row.get("memory_RSS", "")) for row in activity.values() if row.get("memory_RSS", "") != ""]
    attempts, native, resumes = _native_attempts(run_dir)
    return {
        "run_directory": str(run_dir),
        "limit": limit,
        "warmup": warmup,
        "expected_predictions": _expected_predictions(limit, warmup, horizon),
        "predictions": int(summary.get("predictions", 0) or 0),
        "coverage": _number(summary.get("coverage")),
        "MAPE": _number(summary.get("mape")),
        "final_rolling_MAPE": _number(summary.get("final_rolling_mape")),
        "segments": int(summary.get("final_segment_count", 0) or 0),
        "synapses": int(summary.get("final_synapse_count", 0) or 0),
        "raw_mean": _number(summary.get("mean_raw_column_count"), _mean(raw)),
        "raw_peak": int(summary.get("peak_raw_column_count", max(raw, default=0)) or 0),
        "emitted_mean": _mean(emitted),
        "emitted_peak": int(max(emitted, default=0)),
        "contributors_mean": _mean(contributors),
        **scenario,
        **stepwise,
        "runtime_seconds": _number(summary.get("runtime_seconds")),
        "peak_RSS_MB": max(rss, default=0.0),
        "native_crashes": native,
        "resume_count": resumes,
        "attempt_count": attempts,
        "prediction_sha": _prediction_sha(run_dir / "original_predictions.csv"),
        "model_fingerprint": summary.get("final_model_fingerprint", ""),
        "RNG_fingerprint": summary.get("final_rng_fingerprint", ""),
        "complete": int(summary.get("predictions", 0) or 0) == _expected_predictions(limit, warmup, horizon) and _number(summary.get("coverage")) == 1.0,
        "record_errors": {
            index: [
                (
                    _number(steps[index].get(step, {}).get("error"))
                    if steps[index].get(step, {}).get("error", "") != ""
                    else None
                )
                for step in range(1, 6)
            ]
            for index in steps
        },
    }


def _bootstrap(values: list[float], samples: int = 2000, seed: int = 11) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(statistics.fmean(rng.choice(values) for _ in values))
    means.sort()
    return statistics.fmean(values), means[int(0.025 * samples)], means[int(0.975 * samples)]


def _effects(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for limit in (250, 500):
        cells = {}
        for run_id, row in metrics.items():
            if row["limit"] == limit:
                cells[run_id.rsplit("_", 1)[0]] = row
        if any(name not in cells for name in CELL_NAMES):
            continue
        for metric in (
            "MAPE", "step1", "step2", "step3", "step4", "step5",
            "segments", "synapses", "raw_mean", "emitted_mean",
            "contributors_mean", "scenario1_mean", "scenario2_mean",
            "scenario3_mean",
        ):
            a = _number(cells["STRICT_RAW_L4"].get(metric))
            b = _number(cells["STRICT_RAW_L2"].get(metric))
            c = _number(cells["DIAGNOSTIC_STACK_L4"].get(metric))
            d = _number(cells["DIAGNOSTIC_STACK_L2"].get(metric))
            output.append({
                "limit": limit,
                "metric": metric,
                "L2_effect_raw_B_minus_A": b - a,
                "L2_effect_stack_D_minus_C": d - c,
                "stack_effect_L4_C_minus_A": c - a,
                "stack_effect_L2_D_minus_B": d - b,
                "interaction_D_minus_C_minus_B_minus_A": (d - c) - (b - a),
            })
    return output


def _cell_rows(metrics: dict[str, dict[str, Any]], limit: int) -> dict[str, dict[str, Any]]:
    """Return one run for each named cell at a single record limit."""
    rows: dict[str, dict[str, Any]] = {}
    for run_id, row in metrics.items():
        if row["limit"] == limit:
            rows[run_id.rsplit("_", 1)[0]] = row
    return rows


def _write_derived_tables(
    output_dir: Path,
    metrics: dict[str, dict[str, Any]],
    effects: list[dict[str, Any]],
    native_history: dict[str, list[Path]] | None = None,
) -> None:
    """Write stable, small tables used by the report and later analyses."""
    for limit in (250, 500):
        rows = [
            {"limit": limit, "cell": cell, **{
                key: value for key, value in row.items() if key != "record_errors"
            }}
            for cell, row in _cell_rows(metrics, limit).items()
        ]
        _write_rows(output_dir / f"lmatch_stack_2x2_{limit}.csv", rows)
        _write_rows(
            output_dir / f"lmatch_stack_2x2_effects_{limit}.csv",
            [row for row in effects if row["limit"] == limit],
        )

    _write_rows(output_dir / "stepwise_interaction_summary.csv", [
        row for row in effects if row["metric"].startswith("step")
    ])
    _write_rows(output_dir / "scenario_interaction_summary.csv", [
        row for row in effects if row["metric"].startswith("scenario")
    ])
    _write_rows(output_dir / "activity_interaction_summary.csv", [
        row for row in effects if row["metric"] in {
            "segments", "synapses", "raw_mean", "emitted_mean",
            "contributors_mean",
        }
    ])
    _write_rows(output_dir / "mechanism_decomposition_summary.csv", [
        row for row in effects if row["metric"] in {
            "MAPE", "step1", "step2", "step3", "step4", "step5",
            "segments", "raw_mean", "emitted_mean",
        }
    ])
    range_rows: list[dict[str, Any]] = []
    range_edges = ((200, 250), (250, 300), (300, 350), (350, 400), (400, 450), (450, 495))
    for run_id, row in metrics.items():
        if row["limit"] != 500:
            continue
        activity, steps = _read_activity(Path(row["run_directory"]))
        cell = run_id.rsplit("_", 1)[0]
        for start, end in range_edges:
            indices = [index for index in activity if start <= index < end]
            step_errors = [
                _number(steps[index][step]["error"])
                for index in indices
                for step in range(1, 6)
                if steps[index].get(step, {}).get("error", "") not in ("", None)
            ]
            raw_values = _step_values(
                {index: steps[index] for index in indices}, "raw"
            )
            emitted_values = _step_values(
                {index: steps[index] for index in indices}, "emitted"
            )
            range_rows.append({
                "cell": cell,
                "limit": 500,
                "record_range": f"{start}-{end - 1}",
                "records": len(indices),
                "MAPE_or_step_error_mean": _mean(step_errors),
                "step1_mean": _mean([
                    _number(steps[i][1]["error"]) for i in indices
                    if steps[i].get(1, {}).get("error", "") not in ("", None)
                ]),
                "step2_mean": _mean([
                    _number(steps[i][2]["error"]) for i in indices
                    if steps[i].get(2, {}).get("error", "") not in ("", None)
                ]),
                "step3_to_5_mean": _mean([
                    _number(steps[i][step]["error"])
                    for i in indices for step in (3, 4, 5)
                    if steps[i].get(step, {}).get("error", "") not in ("", None)
                ]),
                "segments_mean": _mean([
                    _number(activity[i].get("live_segments", "")) for i in indices
                ]),
                "synapses_mean": _mean([
                    _number(activity[i].get("live_synapses", "")) for i in indices
                ]),
                "raw_mean": _mean(raw_values),
                "emitted_mean": _mean(emitted_values),
                "contributors_mean": _mean(_step_values(
                    {index: steps[index] for index in indices}, "contributors"
                )),
                "scenario2_mean": _mean([
                    _number(activity[i].get("scenario2_count", "")) for i in indices
                ]),
                "scenario3_mean": _mean([
                    _number(activity[i].get("scenario3_count", "")) for i in indices
                ]),
            })
    _write_rows(output_dir / "range_interaction_summary.csv", range_rows or [{
        "cell": "", "limit": "", "record_range": "", "records": 0,
        "MAPE_or_step_error_mean": "", "step1_mean": "", "step2_mean": "",
        "step3_to_5_mean": "", "segments_mean": "", "synapses_mean": "",
        "raw_mean": "", "emitted_mean": "", "contributors_mean": "",
        "scenario2_mean": "", "scenario3_mean": "",
    }])
    native_rows = []
    ledger_rows = []
    for run_id, row in metrics.items():
        cell = run_id.rsplit("_", 1)[0]
        history_paths = [Path(row["run_directory"])]
        if native_history and run_id in native_history:
            history_paths.extend(native_history[run_id])
        attempts, native, resumes = _native_attempts_for_paths(history_paths)
        native_rows.append({
            "cell": cell,
            "limit": row["limit"],
            "run_directory": row["run_directory"],
            "complete": row["complete"],
            "runtime_seconds": row["runtime_seconds"],
            "native_crashes": native,
            "attempt_count": attempts,
            "resume_count": resumes,
            "history_directories": ";".join(str(path) for path in history_paths),
            "peak_RSS_MB": row["peak_RSS_MB"],
        })
        ledger_rows.append({
            "run_id": run_id,
            "cell": cell,
            "limit": row["limit"],
            "purpose": "strict/raw versus diagnostic-stack L_match decomposition",
            "L_match": 4 if cell.endswith("L4") else 2,
            "competition": "competitive_raw" if cell.startswith("DIAGNOSTIC") else "off",
            "selector": "max_candidate_score" if cell.startswith("DIAGNOSTIC") else "existing",
            "seed": 0,
            "commit": _json(Path(row["run_directory"]) / "protocol.json").get("git_commit_sha", ""),
            "attempt_count": attempts,
            "resume_count": resumes,
            "native_crashes": native,
            "expected_predictions": row["expected_predictions"],
            "actual_predictions": row["predictions"],
            "coverage": row["coverage"],
            "MAPE": row["MAPE"],
            "step1": row["step1"],
            "step2": row["step2"],
            "step3": row["step3"],
            "step4": row["step4"],
            "step5": row["step5"],
            "segments": row["segments"],
            "synapses": row["synapses"],
            "raw_columns": row["raw_mean"],
            "emitted_columns": row["emitted_mean"],
            "runtime_seconds": row["runtime_seconds"],
            "peak_RSS_MB": row["peak_RSS_MB"],
            "run_directory": row["run_directory"],
            "complete": row["complete"],
            "prediction_sha": row["prediction_sha"],
            "model_fingerprint": row["model_fingerprint"],
            "RNG_fingerprint": row["RNG_fingerprint"],
            "valid_for_comparison": row["complete"],
            "reuse_or_new": "historical/reused" if "context_representation" in row["run_directory"] or "ambiguity_accumulation" in row["run_directory"] else "new/current",
        })
    _write_rows(output_dir / "native_stability_2x2.csv", native_rows)
    _write_rows(output_dir / "native_stability_raw_vs_stack.csv", [
        {
            "limit": row["limit"],
            "cell": row["cell"],
            "stack": "DIAGNOSTIC_STACK" if row["cell"].startswith("DIAGNOSTIC") else "STRICT_RAW",
            "complete": row["complete"],
            "native_crashes": row["native_crashes"],
            "resume_count": row["resume_count"],
            "attempt_count": row["attempt_count"],
            "runtime_seconds_last_attempt": row["runtime_seconds"],
            "peak_RSS_MB": row["peak_RSS_MB"],
            "history_directories": row["history_directories"],
        }
        for row in native_rows
    ])
    _write_rows(output_dir / "EXPERIMENT_LEDGER.csv", ledger_rows)

    length_rows: list[dict[str, Any]] = []
    for cell in CELL_NAMES:
        row_250 = _cell_rows(metrics, 250).get(cell)
        row_500 = _cell_rows(metrics, 500).get(cell)
        if not row_250 or not row_500:
            continue
        for metric in ("MAPE", "final_rolling_MAPE", "step1", "step2", "step3", "step4", "step5", "segments", "synapses", "raw_mean", "emitted_mean"):
            value_250 = _number(row_250.get(metric))
            value_500 = _number(row_500.get(metric))
            length_rows.append({
                "cell": cell,
                "metric": metric,
                "value_250": value_250,
                "value_500": value_500,
                "delta_500_minus_250": value_500 - value_250,
            })
    _write_rows(output_dir / "lmatch_stack_250_500_length_interaction.csv", length_rows)

    network_metrics = {"segments", "synapses", "raw_mean", "emitted_mean", "scenario1_mean", "scenario2_mean", "scenario3_mean"}
    _write_rows(output_dir / "lmatch_network_main_effect.csv", [
        {
            "limit": row["limit"],
            "metric": row["metric"],
            "raw_L2_minus_L4": row["L2_effect_raw_B_minus_A"],
            "stack_L2_minus_L4": row["L2_effect_stack_D_minus_C"],
            "stack_minus_raw": row["L2_effect_stack_D_minus_C"] - row["L2_effect_raw_B_minus_A"],
        }
        for row in effects
        if row["metric"] in network_metrics
    ])
    _write_rows(output_dir / "lmatch_error_interaction.csv", [
        {
            "limit": row["limit"],
            "metric": row["metric"],
            "raw_L2_effect": row["L2_effect_raw_B_minus_A"],
            "stack_L2_effect": row["L2_effect_stack_D_minus_C"],
            "raw_stack_interaction": row["interaction_D_minus_C_minus_B_minus_A"],
            "stack_effect_L4": row["stack_effect_L4_C_minus_A"],
            "stack_effect_L2": row["stack_effect_L2_D_minus_B"],
        }
        for row in effects
        if row["metric"] == "MAPE" or row["metric"].startswith("step")
    ])
    _write_rows(output_dir / "stepwise_raw_stack_interaction.csv", [
        {
            "limit": row["limit"],
            "step": row["metric"].replace("step", ""),
            "raw_L2_effect": row["L2_effect_raw_B_minus_A"],
            "stack_L2_effect": row["L2_effect_stack_D_minus_C"],
            "interaction": row["interaction_D_minus_C_minus_B_minus_A"],
        }
        for row in effects
        if row["metric"].startswith("step")
    ])


def analyze_runs(
    run_mapping: dict[str, Path],
    output_dir: Path,
    native_history: dict[str, list[Path]] | None = None,
) -> dict[str, Any]:
    metrics = {cell: _run_metrics(path) for cell, path in run_mapping.items()}
    flat_rows = []
    for cell, row in metrics.items():
        flat_rows.append({"cell": cell, **{k: v for k, v in row.items() if k != "record_errors"}})
    _write_rows(output_dir / "cell_metrics.csv", flat_rows)
    effects = _effects(metrics)
    _write_rows(output_dir / "lmatch_stack_2x2_effects.csv", effects)
    tables: list[dict[str, Any]] = []
    for limit in (250, 500):
        for cell, row in _cell_rows(metrics, limit).items():
            tables.append({"limit": limit, "cell": cell, **{k: v for k, v in row.items() if k != "record_errors"}})
    _write_rows(output_dir / "lmatch_stack_2x2_results.csv", tables)
    supplied = {
        (run_id.rsplit("_", 1)[0], row["limit"])
        for run_id, row in metrics.items()
    }
    missing = [
        {"cell": cell, "limit": limit}
        for limit in (250, 500)
        for cell in CELL_NAMES
        if (cell, limit) not in supplied
    ]
    summary: dict[str, Any] = {"cells": {}, "effects": effects, "missing": missing}
    for cell, row in metrics.items():
        summary["cells"][cell] = {k: v for k, v in row.items() if k != "record_errors"}
    for limit in (250, 500):
        for metric in ("step1", "step2", "step3", "step4", "step5"):
            pairs = []
            cells = _cell_rows(metrics, limit)
            for left, right in (("STRICT_RAW_L4", "STRICT_RAW_L2"), ("DIAGNOSTIC_STACK_L4", "DIAGNOSTIC_STACK_L2")):
                if left in cells and right in cells:
                    left_errors = cells[left]["record_errors"]
                    right_errors = cells[right]["record_errors"]
                    keys = sorted(set(left_errors) & set(right_errors))
                    step = int(metric[-1])
                    values = [
                        _number(right_errors[k][step - 1]) - _number(left_errors[k][step - 1])
                        for k in keys
                        if right_errors[k][step - 1] is not None
                        and left_errors[k][step - 1] is not None
                    ]
                    mean, low, high = _bootstrap(values)
                    pairs.append({"limit": limit, "path": "raw" if left.startswith("STRICT") else "stack", "metric": metric, "mean_L4_minus_L2": -mean, "ci_low": -high, "ci_high": -low, "n": len(values)})
            summary.setdefault("bootstrap", []).extend(pairs)
    _write_derived_tables(output_dir, metrics, effects, native_history)
    (output_dir / "FINAL_LMATCH_STACK_2X2_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report = _report(metrics, effects, summary)
    (output_dir / "FIG9_LMATCH_STACK_2X2_DECOMPOSITION_REPORT.md").write_text(report, encoding="utf-8")
    return summary


def _report(metrics: dict[str, dict[str, Any]], effects: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Fig.9 L_match x Diagnostic-Stack Decomposition",
        "",
        "This is a factor decomposition, not a strict paper reproduction claim.",
        "STRICT_RAW_L4 is the strict reference; STRICT_RAW_L2 is a nonpaper single-factor ablation.",
        "",
        "## Cell Results",
        "",
        "| Limit | Cell | MAPE | Step1 | Step2 | Step3 | Step4 | Step5 | Coverage | Predictions | Segments | Raw mean | Native crashes |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for limit in (250, 500):
        for cell in CELL_NAMES:
            row = _cell_rows(metrics, limit).get(cell)
            if not row:
                lines.append(f"| {limit} | {cell} | NOT AVAILABLE | - | - | - | - | - | - | -/- | - | - | - |")
                continue
            lines.append("| {} | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.3f} | {}/{} | {} | {:.2f} | {} |".format(limit, cell, row["MAPE"], row["step1"], row["step2"], row["step3"], row["step4"], row["step5"], row["coverage"], row["predictions"], row["expected_predictions"], row["segments"], row["raw_mean"], row["native_crashes"]))
    lines.extend(["", "## Interpretation Gate", "", "The final primary conclusion must be selected only after both raw and diagnostic paths are complete. A lower MAPE in one cell alone is not evidence of mechanism correctness.", "", "## Runtime Integrity", "", "No record may be skipped. A cell is invalid unless predictions equal the expected count and coverage is 1.0. Native crashes and resume attempts are reported separately from model quality.", "", "## Missing Cells", "", ""])
    if summary.get("missing"):
        lines.append("The following cell/limit combinations were not supplied and are excluded from interaction effects:")
        for item in summary["missing"]:
            lines.append(f"- `{item['cell']}_{item['limit']}`")
    else:
        lines.append("All requested cell/limit combinations were supplied.")
    lines.extend(["", "## Bootstrap", "", "Bootstrap intervals describe paired prediction-error differences; they do not establish mechanism causality.", ""])
    if summary.get("bootstrap"):
        lines.append("| Limit | Path | Metric | Mean L4-L2 | 95% low | 95% high | n |")
        lines.append("|---:|---|---|---:|---:|---:|---:|")
        for row in summary["bootstrap"]:
            lines.append(f"| {row['limit']} | {row['path']} | {row['metric']} | {row['mean_L4_minus_L2']:.6f} | {row['ci_low']:.6f} | {row['ci_high']:.6f} | {row['n']} |")
    return "\n".join(lines) + "\n"


def _parse_mapping(values: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for value in values:
        cell, separator, path = value.partition("=")
        parts = cell.rsplit("_", 1)
        base = parts[0] if len(parts) == 2 and parts[1] in {"250", "500"} else cell
        if not separator or base not in CELL_NAMES:
            raise ValueError(f"expected CELL[_250|_500]=PATH, got {value!r}")
        run_id = cell if cell != base else f"{base}_{_run_metrics(Path(path)).get('limit', '')}"
        if run_id.endswith("_"):
            raise ValueError(f"could not infer record limit from {value!r}")
        mapping[run_id] = Path(path)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inventory-root", type=Path, default=Path("results/fig9_diagnostics"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--run", action="append", default=[], help="CELL=PATH; may be repeated for completed cells")
    parser.add_argument(
        "--native-history",
        action="append",
        default=[],
        help="CELL[_250|_500]=PATH; add a prior process-attempt directory to stability counts",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_protocol_audit(args.output_dir / "protocol_audit", args.docs_dir)
    inventory_runs(args.inventory_root, args.output_dir / "existing_runs", args.docs_dir)
    mapping = _parse_mapping(args.run)
    if mapping:
        native_history: dict[str, list[Path]] = defaultdict(list)
        for value in args.native_history:
            run_id, separator, path = value.partition("=")
            if not separator or run_id not in mapping:
                raise ValueError(f"native history key must match a supplied run id, got {value!r}")
            native_history[run_id].append(Path(path))
        analyze_runs(mapping, args.output_dir / "analysis", native_history)


if __name__ == "__main__":
    main()
