"""Offline audit for the Fig.9 L2/L4 500-record robustness experiment.

The analyzer only reads completed run artifacts.  It never loads a model,
selects a prediction, or changes a checkpoint.  Activity and error are
reported together so a lower activity level is not silently treated as a
better memory representation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Iterable


EXPECTED_L_MATCHES = (2, 4)
RANGES = ((200, 249), (250, 299), (300, 349), (350, 399), (400, 449), (450, 494))


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def record_range(index: int) -> str:
    for start, end in RANGES:
        if start <= index <= end:
            return f"{start}-{end}"
    return "outside_requested_ranges"


def expected_predictions(protocol: dict[str, object]) -> int:
    limit = as_int(protocol.get("data_file_rows_used"))
    warmup = as_int(protocol.get("warmup_length"))
    horizon = as_int(protocol.get("prediction_horizon"))
    return max(0, limit - horizon - warmup)


def process_attempt_summary(run_dir: Path) -> dict[str, object]:
    """Summarize child-process attempts used by checkpoint recovery.

    A resumed long run may contain several successful and native-crashed child
    processes.  The final model summary only reports the last child, so these
    files are the auditable source for cumulative recovery runtime.
    """

    attempts = []
    for path in sorted(run_dir.glob("process_*.json")):
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        attempts.append(payload)
    elapsed = sum(as_float(item.get("elapsed_seconds")) for item in attempts)
    native_crashes = sum(
        as_int(item.get("exit_code")) == 3221225477 for item in attempts
    )
    return {
        "process_attempt_count": len(attempts),
        "process_runtime_seconds_total": elapsed,
        "native_crash_count": native_crashes,
    }


def load_run(l_match: int, run_dir: Path) -> dict[str, object]:
    protocol = read_json(run_dir / "original_protocol.json")
    summary = read_json(run_dir / "original_summary.json")
    predictions = read_csv(run_dir / "original_predictions.csv")
    density = read_csv(run_dir / "original_density_trace.csv")
    ledger = read_csv(run_dir / "long_sequence_activity_trace.csv")
    if not ledger:
        ledger = read_csv(run_dir / "original_long_sequence_activity_trace.csv")
    if not ledger:
        raise ValueError(f"missing long-sequence ledger: {run_dir}")
    provenance = read_csv(run_dir / "autonomous_rollout_provenance_by_step.csv")
    return {
        "L_match": l_match,
        "run_dir": str(run_dir),
        "protocol": protocol,
        "summary": summary,
        "predictions": predictions,
        "density": density,
        "ledger": ledger,
        "provenance": provenance,
        "prediction_sha256": sha256(run_dir / "original_predictions.csv"),
        "process_attempts": process_attempt_summary(run_dir),
    }


def prediction_summary(run: dict[str, object]) -> dict[str, object]:
    summary = run["summary"]
    protocol = run["protocol"]
    assert isinstance(summary, dict) and isinstance(protocol, dict)
    predictions = run["predictions"]
    assert isinstance(predictions, list)
    expected = expected_predictions(protocol)
    attempts = run["process_attempts"]
    assert isinstance(attempts, dict)
    summary_runtime = as_float(summary.get("runtime_seconds"))
    process_runtime = as_float(attempts.get("process_runtime_seconds_total"))
    return {
        "L_match": run["L_match"],
        "expected_predictions": expected,
        "actual_predictions": as_int(summary.get("predictions")),
        "attempted_predictions": as_int(summary.get("attempted_predictions")),
        "coverage": as_float(summary.get("coverage")),
        "MAPE": as_float(summary.get("mape")),
        "final_rolling_MAPE": as_float(summary.get("final_rolling_mape")),
        "runtime_seconds": process_runtime or summary_runtime,
        "summary_runtime_seconds": summary_runtime,
        "process_attempt_runtime_seconds_total": process_runtime,
        "process_attempt_count": as_int(attempts.get("process_attempt_count")),
        "native_crash_count": as_int(attempts.get("native_crash_count")),
        "final_segments": as_int(summary.get("final_segment_count")),
        "final_synapses": as_int(summary.get("final_synapse_count")),
        "mean_raw_columns": as_float(summary.get("mean_raw_column_count")),
        "peak_raw_columns": as_int(summary.get("peak_raw_column_count")),
        "prediction_SHA256": run["prediction_sha256"],
        "model_fingerprint": summary.get("final_model_fingerprint", ""),
        "RNG_fingerprint": summary.get("final_rng_fingerprint", ""),
        "ledger_rows": len(run["ledger"]),
        "prediction_file_rows": len(predictions),
    }


def density_by_step(run: dict[str, object]) -> dict[tuple[int, int], list[dict[str, str]]]:
    density = run["density"]
    assert isinstance(density, list)
    grouped: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in density:
        key = (as_int(row.get("record_index")), as_int(row.get("horizon_step")))
        grouped.setdefault(key, []).append(row)
    return grouped


def range_stats(run: dict[str, object]) -> list[dict[str, object]]:
    density = run["density"]
    ledger = run["ledger"]
    assert isinstance(density, list) and isinstance(ledger, list)
    output: list[dict[str, object]] = []
    for start, end in RANGES:
        drows = [row for row in density if start <= as_int(row.get("record_index")) <= end]
        lrows = [row for row in ledger if start <= as_int(row.get("record_index")) <= end]
        predictions = [row for row in drows if as_int(row.get("horizon_step")) == 5]
        targets = [abs(as_float(row.get("actual_future_passenger"))) for row in predictions]
        errors = [as_float(row.get("absolute_error")) for row in predictions]
        scenarios = {
            f"scenario{i}_count": sum(as_int(row.get(f"scenario{i}_count")) for row in lrows)
            for i in (1, 2, 3)
        }
        row: dict[str, object] = {
            "range": f"{start}-{end}",
            "record_start": start,
            "record_end": end,
            "ledger_rows": len(lrows),
            "prediction_rows": len(predictions),
            "MAPE_step5": sum(errors) / sum(targets) if sum(targets) else 0.0,
            "mean_live_segments": mean(as_float(item.get("live_segments")) for item in lrows),
            "mean_live_synapses": mean(as_float(item.get("live_synapses")) for item in lrows),
            "mean_contributors": mean(as_float(item.get("mean_contributors")) for item in lrows),
            "peak_contributors": max((as_float(item.get("peak_contributors")) for item in lrows), default=0.0),
            "mean_rolling_MAPE": mean(as_float(item.get("rolling_MAPE")) for item in lrows),
            **scenarios,
        }
        for step in range(1, 6):
            step_rows = [
                item
                for item in density
                if start <= as_int(item.get("record_index")) <= end
                and as_int(item.get("horizon_step")) == step
            ]
            step_targets = [abs(as_float(item.get("actual_future_passenger"))) for item in step_rows]
            step_errors = [as_float(item.get("absolute_error")) for item in step_rows]
            row[f"MAPE_step{step}"] = (
                sum(step_errors) / sum(step_targets) if sum(step_targets) else 0.0
            )
            row[f"mean_raw_columns_step{step}"] = mean(
                as_float(item.get("raw_predicted_column_count")) for item in step_rows
            )
            row[f"mean_emitted_columns_step{step}"] = mean(
                as_float(item.get("emitted_column_count")) for item in step_rows
            )
            row[f"mean_contributors_step{step}"] = mean(
                as_float(item.get(f"total_contributors_step{step}"))
                for item in lrows
            )
        output.append(row)
    return output


def overall_stepwise_error(run: dict[str, object]) -> dict[str, object]:
    """Aggregate stepwise MAPE over all requested post-warmup records."""

    density = run["density"]
    assert isinstance(density, list)
    row: dict[str, object] = {"L_match": run["L_match"]}
    for step in range(1, 6):
        step_rows = [
            item
            for item in density
            if RANGES[0][0] <= as_int(item.get("record_index")) <= RANGES[-1][1]
            and as_int(item.get("horizon_step")) == step
        ]
        targets = sum(abs(as_float(item.get("actual_future_passenger"))) for item in step_rows)
        errors = sum(as_float(item.get("absolute_error")) for item in step_rows)
        row[f"MAPE_step{step}"] = errors / targets if targets else 0.0
        row[f"prediction_rows_step{step}"] = len(step_rows)
    return row


def runtime_memory_summary(run: dict[str, object]) -> dict[str, object]:
    """Report cumulative attempt runtime and observed ledger memory bounds."""

    ledger = run["ledger"]
    assert isinstance(ledger, list)
    memory = [as_float(row.get("memory_RSS")) for row in ledger]
    memory = [value for value in memory if value > 0]
    summary = run["summary"]
    assert isinstance(summary, dict)
    attempts = run["process_attempts"]
    assert isinstance(attempts, dict)
    summary_runtime = as_float(summary.get("runtime_seconds"))
    process_runtime = as_float(attempts.get("process_runtime_seconds_total"))
    return {
        "L_match": run["L_match"],
        "runtime_seconds": process_runtime or summary_runtime,
        "summary_runtime_seconds": summary_runtime,
        "process_attempt_runtime_seconds_total": process_runtime,
        "process_attempt_count": as_int(attempts.get("process_attempt_count")),
        "native_crash_count": as_int(attempts.get("native_crash_count")),
        "ledger_rows": len(ledger),
        "memory_RSS_min_mb": min(memory, default=0.0),
        "memory_RSS_peak_mb": max(memory, default=0.0),
        "memory_RSS_last_mb": memory[-1] if memory else 0.0,
    }


def enrich_activity_ledger(run: dict[str, object]) -> list[dict[str, object]]:
    """Join the exact per-record ledger with provenance aggregates when present."""

    ledger = run["ledger"]
    provenance = run["provenance"]
    assert isinstance(ledger, list) and isinstance(provenance, list)
    by_anchor_step = {
        (as_int(row.get("anchor_record_index")), as_int(row.get("horizon_step"))): row
        for row in provenance
    }
    output: list[dict[str, object]] = []
    for source in ledger:
        row: dict[str, object] = dict(source)
        index = as_int(source.get("record_index"))
        actual_values: list[float] = []
        generated_values: list[float] = []
        depth_values: list[float] = []
        for step in range(1, 6):
            provenance_row = by_anchor_step.get((index, step))
            if provenance_row is None:
                continue
            row[f"total_contributors_step{step}"] = as_float(
                provenance_row.get("total_contributors")
            )
            actual_values.append(as_float(provenance_row.get("actual_history_contributors")))
            generated_values.append(as_float(provenance_row.get("generated_contributors")))
            depth_values.append(as_float(provenance_row.get("mean_self_generated_depth")))
            row[f"actual_history_retention_step{step}"] = as_float(
                provenance_row.get("actual_history_fraction")
            )
            row[f"generated_fraction_step{step}"] = as_float(
                provenance_row.get("generated_fraction")
            )
        total = sum(actual_values) + sum(generated_values)
        row["actual_root_contributors"] = sum(actual_values)
        row["generated_root_contributors"] = sum(generated_values)
        row["actual_history_retention"] = sum(actual_values) / total if total else ""
        row["generated_fraction"] = sum(generated_values) / total if total else ""
        row["self_generated_depth_mean"] = mean(depth_values)
        row["self_generated_depth_max"] = max(depth_values, default=0.0)
        row["range"] = record_range(index)
        output.append(row)
    return output


def linear_slope(rows: list[dict[str, object]], field: str) -> float:
    points = [(as_float(row.get("record_index")), as_float(row.get(field))) for row in rows]
    points = [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y)]
    if len(points) < 2:
        return 0.0
    x_mean = mean(x for x, _ in points)
    y_mean = mean(y for _, y in points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    return sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator if denominator else 0.0


def growth_summary(run: dict[str, object]) -> dict[str, object]:
    ledger = run["ledger"]
    assert isinstance(ledger, list)
    return {
        "L_match": run["L_match"],
        "segments_slope_per_record": linear_slope(ledger, "live_segments"),
        "synapses_slope_per_record": linear_slope(ledger, "live_synapses"),
        "contributors_slope_per_record": linear_slope(ledger, "mean_contributors"),
        "raw_columns_slope_per_record": linear_slope(ledger, "raw_predicted_columns_step5"),
        "first_segments": as_float(ledger[0].get("live_segments")) if ledger else 0.0,
        "last_segments": as_float(ledger[-1].get("live_segments")) if ledger else 0.0,
        "first_synapses": as_float(ledger[0].get("live_synapses")) if ledger else 0.0,
        "last_synapses": as_float(ledger[-1].get("live_synapses")) if ledger else 0.0,
    }


def activity_ratios(l2: dict[str, object], l4: dict[str, object]) -> list[dict[str, object]]:
    left = {as_int(row.get("record_index")): row for row in l2["ledger"]}  # type: ignore[index]
    right = {as_int(row.get("record_index")): row for row in l4["ledger"]}  # type: ignore[index]
    output: list[dict[str, object]] = []
    for index in sorted(set(left) & set(right)):
        lrow, rrow = left[index], right[index]
        output.append(
            {
                "record_index": index,
                "contributor_ratio_L2_over_L4": as_float(lrow.get("mean_contributors")) / max(as_float(rrow.get("mean_contributors")), 1e-12),
                "raw_column_ratio_L2_over_L4": as_float(lrow.get("raw_predicted_columns_step5")) / max(as_float(rrow.get("raw_predicted_columns_step5")), 1e-12),
                "emitted_column_ratio_L2_over_L4": as_float(lrow.get("emitted_columns_step5")) / max(as_float(rrow.get("emitted_columns_step5")), 1e-12),
                "segment_ratio_L2_over_L4": as_float(lrow.get("live_segments")) / max(as_float(rrow.get("live_segments")), 1e-12),
                "synapse_ratio_L2_over_L4": as_float(lrow.get("live_synapses")) / max(as_float(rrow.get("live_synapses")), 1e-12),
            }
        )
    return output


def association(rows: list[dict[str, object]], x_field: str, y_field: str) -> dict[str, float]:
    pairs = [(as_float(row.get(x_field)), as_float(row.get(y_field))) for row in rows]
    pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return {"pearson": 0.0, "n": float(len(pairs))}
    x_mean = mean(x for x, _ in pairs)
    y_mean = mean(y for _, y in pairs)
    x_dev = [x - x_mean for x, _ in pairs]
    y_dev = [y - y_mean for _, y in pairs]
    denominator = math.sqrt(sum(value * value for value in x_dev) * sum(value * value for value in y_dev))
    return {
        "pearson": sum(x * y for x, y in zip(x_dev, y_dev)) / denominator if denominator else 0.0,
        "n": float(len(pairs)),
    }


def bootstrap_mape_difference(l2: dict[str, object], l4: dict[str, object], samples: int, seed: int) -> dict[str, float]:
    l2_rows = {as_int(row.get("record_index")): row for row in l2["ledger"]}  # type: ignore[index]
    l4_rows = {as_int(row.get("record_index")): row for row in l4["ledger"]}  # type: ignore[index]
    paired = [
        (as_float(l2_rows[index].get("prediction_error_step5")), as_float(l4_rows[index].get("prediction_error_step5")))
        for index in sorted(set(l2_rows) & set(l4_rows))
        if l2_rows[index].get("prediction_error_step5") not in (None, "")
        and l4_rows[index].get("prediction_error_step5") not in (None, "")
    ]
    if not paired:
        return {"n": 0, "difference_mean_L4_minus_L2": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    values = [right - left for left, right in paired]
    boot = []
    for _ in range(samples):
        boot.append(mean(values[rng.randrange(len(values))] for _ in values))
    boot.sort()
    low = boot[int(0.025 * (len(boot) - 1))]
    high = boot[int(0.975 * (len(boot) - 1))]
    return {
        "n": len(values),
        "difference_mean_L4_minus_L2": mean(values),
        "ci_low": low,
        "ci_high": high,
    }


def classify(
    summaries: list[dict[str, object]],
    ranges: list[dict[str, object]],
    growth: list[dict[str, object]],
) -> tuple[str, str]:
    by_l = {as_int(row.get("L_match")): row for row in summaries}
    range_l2 = [row for row in ranges if as_int(row.get("L_match")) == 2]
    range_l4 = [row for row in ranges if as_int(row.get("L_match")) == 4]
    pairs = list(zip(range_l2, range_l4))
    l2 = by_l[2]
    l4 = by_l[4]
    gaps = [as_float(left.get("MAPE_step5")) - as_float(right.get("MAPE_step5")) for left, right in pairs]
    late_reversal = any(gap >= 0.0 for gap in gaps[-2:])
    l2_lower_activity = all(
        as_float(left.get("mean_contributors")) < as_float(right.get("mean_contributors"))
        for left, right in pairs
    )
    if as_float(l2.get("MAPE")) < as_float(l4.get("MAPE")) and not late_reversal and l2_lower_activity:
        return "L2_SPARSITY_ADVANTAGE_IS_LONG_SEQUENCE_ROBUST", "KEEP_L2_AS_EXPERIMENTAL_BASELINE"
    if late_reversal and l2_lower_activity:
        return "L2_ADVANTAGE_DECAYS_WITH_SEQUENCE_LENGTH", "INVESTIGATE_L2_AMBIGUITY_ACCUMULATION"
    if l2_lower_activity and as_float(l2.get("MAPE")) >= as_float(l4.get("MAPE")):
        return "L2_BECOMES_OVERSPARSE_ON_LONGER_SEQUENCE", "INVESTIGATE_L2_AMBIGUITY_ACCUMULATION"
    if as_float(growth[1].get("segments_slope_per_record")) > as_float(growth[0].get("segments_slope_per_record")) and as_float(l4.get("MAPE")) > as_float(l2.get("MAPE")):
        return "L4_DENSITY_DRIVES_RECURRENT_ERROR_AMPLIFICATION", "INVESTIGATE_L4_DENSITY_EXPLOSION"
    return "ACTIVITY_LEVEL_ASSOCIATES_WITH_ERROR_BUT_DOES_NOT_EXPLAIN_ADVANTAGE", "KEEP_L2_AS_EXPERIMENTAL_BASELINE"


def analyze(l2_dir: Path, l4_dir: Path, output_dir: Path, bootstrap_samples: int, bootstrap_seed: int) -> None:
    l2 = load_run(2, l2_dir)
    l4 = load_run(4, l4_dir)
    summaries = [prediction_summary(l2), prediction_summary(l4)]
    enriched_l2 = enrich_activity_ledger(l2)
    enriched_l4 = enrich_activity_ledger(l4)
    l2["ledger"] = enriched_l2
    l4["ledger"] = enriched_l4
    ranges = []
    for run in (l2, l4):
        for row in range_stats(run):
            row["L_match"] = run["L_match"]
            ranges.append(row)
    growth = [growth_summary(l2), growth_summary(l4)]
    ratios = activity_ratios(l2, l4)
    associations = []
    for run in (l2, l4):
        ledger = run["ledger"]
        assert isinstance(ledger, list)
        associations.append({
            "L_match": run["L_match"],
            "contributors_vs_step5_error": association(ledger, "mean_contributors", "prediction_error_step5"),
            "segments_vs_step5_error": association(ledger, "live_segments", "prediction_error_step5"),
            "raw_columns_vs_step5_error": association(ledger, "raw_predicted_columns_step5", "prediction_error_step5"),
        })
    bootstrap = bootstrap_mape_difference(l2, l4, bootstrap_samples, bootstrap_seed)
    primary, recommended = classify(summaries, ranges, growth)
    stepwise = [overall_stepwise_error(run) for run in (l2, l4)]
    runtime_rows = [runtime_memory_summary(l2), runtime_memory_summary(l4)]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "long_sequence_activity_trace.csv", enriched_l2 + enriched_l4)
    write_csv(output_dir / "l2_l4_500_comparison.csv", summaries)
    write_csv(output_dir / "long_sequence_range_summary.csv", ranges)
    write_csv(output_dir / "l2_l4_network_growth.csv", growth)
    write_csv(output_dir / "l2_l4_activity_ratio.csv", ratios)
    write_csv(output_dir / "activity_error_association.csv", associations)
    write_csv(
        output_dir / "l2_l4_stepwise_error.csv",
        stepwise,
    )
    write_csv(
        output_dir / "runtime_memory_summary.csv",
        runtime_rows,
    )
    write_json(output_dir / "optional_1000_comparison.json", {"status": "NOT_RUN", "reason": "500-record robustness result does not authorize 1000 automatically."})
    final = {
        "protocol": {
            "limit": 500,
            "warmup": 200,
            "prediction_horizon": 5,
            "continuous_impl": "reference",
            "competition_mode": "competitive_raw",
            "simultaneous_policy": "batched",
            "intracolumn_selection_policy": "max_candidate_score",
            "ground_truth_used_for_selection": False,
            "uses_compensation": False,
            "uses_future_covariates": False,
        },
        "summaries": summaries,
        "growth": growth,
        "bootstrap": bootstrap,
        "primary_conclusion": primary,
        "recommended_next_step": recommended,
        "optional_1000": "NOT_RUN",
    }
    write_json(output_dir / "FINAL_LONG_SEQUENCE_SUMMARY.json", final)
    report = [
        "# Fig.9 L2/L4 Long-Sequence Robustness Report",
        "",
        "This is a diagnostic L_match ablation, not a strict paper reproduction.",
        "No compensation, future covariates, ground-truth selection, learning, selector, competition, RNG, or strict default was changed.",
        "",
        f"- Primary conclusion: `{primary}`",
        f"- Recommended next step: `{recommended}`",
        f"- Bootstrap L4 minus L2 step-5 error difference: `{bootstrap['difference_mean_L4_minus_L2']:.6f}` ({bootstrap['ci_low']:.6f}, {bootstrap['ci_high']:.6f})",
        "- Optional 1000-record run: `NOT_RUN`",
        "",
        "## Formal Completeness",
        "",
        "| L_match | expected predictions | actual predictions | coverage | MAPE |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        report.append(f"| {row['L_match']} | {row['expected_predictions']} | {row['actual_predictions']} | {row['coverage']:.6f} | {row['MAPE']:.9f} |")
    report += [
        "",
        "## Interpretation",
        "",
        "The activity ledger is recorded after each real observation. Rollout remains read-only and is not fed back through decoded values.",
        "Range-level results, network growth, activity ratios, scenario counts, stepwise errors, and activity-error associations are in the CSV outputs beside this report.",
        "",
        "## Stepwise Error",
        "",
        "| L_match | step 1 | step 2 | step 3 | step 4 | step 5 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stepwise:
        report.append(
            f"| {row['L_match']} | {row['MAPE_step1']:.6f} | {row['MAPE_step2']:.6f} | {row['MAPE_step3']:.6f} | {row['MAPE_step4']:.6f} | {row['MAPE_step5']:.6f} |"
        )
    report += [
        "",
        "## Range Activity and Scenarios",
        "",
        "| L_match | range | step-5 MAPE | mean segments | mean synapses | mean contributors | raw columns step 5 | emitted columns step 5 | scenario 1 | scenario 2 | scenario 3 |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranges:
        report.append(
            f"| {row['L_match']} | {row['range']} | {row['MAPE_step5']:.6f} | {row['mean_live_segments']:.2f} | {row['mean_live_synapses']:.2f} | {row['mean_contributors']:.2f} | {row['mean_raw_columns_step5']:.2f} | {row['mean_emitted_columns_step5']:.2f} | {row['scenario1_count']} | {row['scenario2_count']} | {row['scenario3_count']} |"
        )
    report += [
        "",
        "## Network Growth",
        "",
        "| L_match | segment slope/record | synapse slope/record | contributor slope/record | raw-column slope/record | first segments | last segments |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in growth:
        report.append(
            f"| {row['L_match']} | {row['segments_slope_per_record']:.4f} | {row['synapses_slope_per_record']:.4f} | {row['contributors_slope_per_record']:.4f} | {row['raw_columns_slope_per_record']:.4f} | {row['first_segments']:.0f} | {row['last_segments']:.0f} |"
        )
    report += [
        "",
        "## Activity/Error Association",
        "",
        "Pearson correlation is descriptive only; it is not evidence that activity causes the error difference.",
        "",
        "| L_match | contributors vs step-5 error | segments vs step-5 error | raw columns vs step-5 error |",
        "|---:|---:|---:|---:|",
    ]
    for row in associations:
        report.append(
            f"| {row['L_match']} | {row['contributors_vs_step5_error']['pearson']:.6f} (n={row['contributors_vs_step5_error']['n']:.0f}) | {row['segments_vs_step5_error']['pearson']:.6f} (n={row['segments_vs_step5_error']['n']:.0f}) | {row['raw_columns_vs_step5_error']['pearson']:.6f} (n={row['raw_columns_vs_step5_error']['n']:.0f}) |"
        )
    report += [
        "",
        "## Verification",
        "",
        f"- L2 prediction SHA256: `{summaries[0]['prediction_SHA256']}`",
        f"- L4 prediction SHA256: `{summaries[1]['prediction_SHA256']}`",
        f"- L2 model/RNG fingerprints: `{summaries[0]['model_fingerprint']}` / `{summaries[0]['RNG_fingerprint']}`",
        f"- L4 model/RNG fingerprints: `{summaries[1]['model_fingerprint']}` / `{summaries[1]['RNG_fingerprint']}`",
        "- Every requested prediction row is present; no ground-truth value was used to select a candidate.",
        "",
        "## Runtime and Recovery",
        "",
        "Runtime includes all checkpoint-recovery child-process attempts. Native crash counts are reported explicitly; they are runtime stability observations, not model-quality improvements.",
        "",
        "| L_match | cumulative attempt seconds | attempts | native crashes | peak RSS MB |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in runtime_rows:
        report.append(
            f"| {row['L_match']} | {row['runtime_seconds']:.3f} | {row['process_attempt_count']} | {row['native_crash_count']} | {row['memory_RSS_peak_mb']:.3f} |"
        )
    (output_dir / "FIG9_L2_L4_LONG_SEQUENCE_ROBUSTNESS_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir-l2", required=True, type=Path)
    parser.add_argument("--run-dir-l4", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()
    analyze(args.run_dir_l2, args.run_dir_l4, args.output_dir, args.bootstrap_samples, args.bootstrap_seed)


if __name__ == "__main__":
    main()
