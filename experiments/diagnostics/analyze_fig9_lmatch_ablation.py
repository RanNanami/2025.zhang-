"""Paired offline analysis for the real Fig.9 L_match diagnostic grid."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


EXPECTED_L_MATCHES = (4, 3, 2)
MARKERS = {
    "diagnostic_only": True,
    "lmatch_real_ablation": True,
    "strict_default_unchanged": True,
    "uses_ground_truth_for_selection": False,
    "uses_future_covariates": False,
    "uses_compensation": False,
}


def _open_csv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def read_rows(run_dir: Path, stem: str) -> list[dict[str, str]]:
    for name in (stem, f"{stem}.gz"):
        path = run_dir / name
        if path.exists():
            with _open_csv(path) as handle:
                return list(csv.DictReader(handle))
    return []


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else 0.0


def _median(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.median(materialized) if materialized else 0.0


def record_range(index: int) -> str:
    if 200 <= index <= 244:
        return "200-244"
    start = (index // 50) * 50
    return f"{start}-{start + 49}"


def validate_protocols(
    protocols: dict[int, dict[str, object]],
) -> dict[str, object]:
    # A commit identifies code provenance, not a model protocol parameter.
    # Keep differing commits auditable without invalidating an otherwise
    # identical ablation after a logging-only recovery patch.
    ignored = {"L_match", "git_commit_sha"}
    provenance = {
        "git_commit_sha": {
            str(l_match): protocols[l_match].get("git_commit_sha")
            for l_match in protocols
        }
    }
    mismatches: dict[str, dict[str, object]] = {}
    keys = set().union(*(protocol.keys() for protocol in protocols.values()))
    for key in sorted(keys - ignored):
        values = {str(l_match): protocols[l_match].get(key) for l_match in protocols}
        encoded = {json.dumps(value, sort_keys=True) for value in values.values()}
        if len(encoded) > 1:
            mismatches[key] = values
    marker_errors = {
        str(l_match): {
            key: protocol.get(key)
            for key, expected in MARKERS.items()
            if protocol.get(key) != expected
        }
        for l_match, protocol in protocols.items()
    }
    marker_errors = {key: value for key, value in marker_errors.items() if value}
    return {
        "comparison_valid": not mismatches and not marker_errors,
        "only_intended_model_variable": "L_match",
        "protocol_mismatches_excluding_L_match": mismatches,
        "provenance_differences": provenance,
        "marker_errors": marker_errors,
        "recurrent_trajectory_divergence": True,
    }


def paired_bootstrap(
    left: dict[int, tuple[float, float]],
    right: dict[int, tuple[float, float]],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    common = sorted(set(left) & set(right))
    pairs = [(left[index], right[index]) for index in common]

    def difference(sample: list[tuple[tuple[float, float], tuple[float, float]]]) -> float:
        left_error = sum(pair[0][0] for pair in sample)
        left_target = sum(abs(pair[0][1]) for pair in sample)
        right_error = sum(pair[1][0] for pair in sample)
        right_target = sum(abs(pair[1][1]) for pair in sample)
        left_mape = left_error / left_target if left_target else 0.0
        right_mape = right_error / right_target if right_target else 0.0
        return right_mape - left_mape

    observed = difference(pairs)
    rng = random.Random(seed)
    boot = []
    if pairs:
        for _ in range(samples):
            boot.append(difference([pairs[rng.randrange(len(pairs))] for _ in pairs]))
    boot.sort()
    low_index = max(0, int(0.025 * len(boot)) - 1)
    high_index = min(len(boot) - 1, int(0.975 * len(boot)))
    return {
        "paired_records": len(pairs),
        "mean_difference": observed,
        "ci95_low": boot[low_index] if boot else 0.0,
        "ci95_high": boot[high_index] if boot else 0.0,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def summarize_run(l_match: int, run_dir: Path) -> dict[str, object]:
    protocol = read_json(run_dir / "original_protocol.json")
    summary = read_json(run_dir / "original_summary.json")
    runtime = read_json(run_dir / "runtime.json")
    predictions_path = run_dir / "original_predictions.csv"
    predictions = read_rows(run_dir, "original_predictions.csv")
    density = read_rows(run_dir, "original_density_trace.csv")
    scenarios = read_rows(run_dir, "observe_scenario_trace.csv")
    segments = read_rows(run_dir, "match_overlap_segment_trace.csv")
    trajectory = read_rows(run_dir, "context_trajectory_column_trace.csv")
    oracle = read_rows(run_dir, "oracle_candidate_trace.csv")
    return {
        "L_match": l_match,
        "run_dir": str(run_dir),
        "protocol": protocol,
        "summary": summary,
        "runtime": runtime,
        "predictions": predictions,
        "density": density,
        "scenarios": scenarios,
        "segments": segments,
        "trajectory": trajectory,
        "oracle": oracle,
        "predictions_sha256": file_sha256(predictions_path),
    }


def summary_row(run: dict[str, object]) -> dict[str, object]:
    summary = run["summary"]
    assert isinstance(summary, dict)
    return {
        "L_match": run["L_match"],
        "MAPE": summary.get("mape", 0.0),
        "final_rolling_MAPE": summary.get("final_rolling_mape", ""),
        "coverage": summary.get("coverage", 0.0),
        "attempted_predictions": summary.get("attempted_predictions", 0),
        "runtime_seconds": summary.get("runtime_seconds", 0.0),
        "mean_raw_columns": summary.get("mean_raw_column_count", 0.0),
        "peak_raw_columns": summary.get("peak_raw_column_count", 0),
        "final_segment_count": summary.get("final_segment_count", ""),
        "model_fingerprint": summary.get("final_model_fingerprint", ""),
        "RNG_fingerprint": summary.get("final_rng_fingerprint", ""),
        "predictions_SHA256": run["predictions_sha256"],
        "diagnostic_only": True,
        "strict_default_unchanged": True,
    }


def horizon_rows(run: dict[str, object]) -> list[dict[str, object]]:
    density = run["density"]
    assert isinstance(density, list)
    output = []
    for step in range(1, 6):
        rows = [row for row in density if _int(row.get("horizon_step")) == step]
        valid = [row for row in rows if not _bool(row.get("prediction_missing"))]
        error_sum = sum(_float(row.get("absolute_error")) for row in valid)
        target_sum = sum(abs(_float(row.get("actual_future_passenger"))) for row in valid)
        output.append(
            {
                "L_match": run["L_match"],
                "horizon_step": step,
                "MAPE": error_sum / target_sum if target_sum else 0.0,
                "coverage": len(valid) / len(rows) if rows else 0.0,
                "mean_raw_columns": _mean(_float(row.get("raw_predicted_column_count")) for row in rows),
                "mean_emitted_columns": _mean(_float(row.get("emitted_column_count")) for row in rows),
                "recurrent_trajectory_divergence": step > 1,
            }
        )
    return output


def scenario_rows(run: dict[str, object]) -> list[dict[str, object]]:
    rows = run["scenarios"]
    assert isinstance(rows, list)
    total = len(rows)
    counts = Counter(row.get("observe_scenario", "") for row in rows)
    return [
        {
            "L_match": run["L_match"],
            "scenario": scenario,
            "count": counts[scenario],
            "rate": counts[scenario] / total if total else 0.0,
            "observations": total,
        }
        for scenario in ("scenario1", "scenario2", "scenario3")
    ]


def grouped_scenario_rows(
    run: dict[str, object], field_name: str
) -> list[dict[str, object]]:
    rows = run["scenarios"]
    assert isinstance(rows, list)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            record_range(_int(row.get("actual_record_index")))
            if field_name == "record_range"
            else str(row.get(field_name, ""))
        )
        grouped[key].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        counts = Counter(row.get("observe_scenario", "") for row in group)
        output.append(
            {
                "L_match": run["L_match"],
                field_name: key,
                "observations": len(group),
                "scenario1_count": counts["scenario1"],
                "scenario2_count": counts["scenario2"],
                "scenario3_count": counts["scenario3"],
                "scenario3_rate": counts["scenario3"] / len(group),
                "created_segment_count": sum(
                    bool(row.get("created_segment_id")) for row in group
                ),
                "reinforced_segment_count": sum(
                    bool(row.get("reinforced_segment_id")) for row in group
                ),
            }
        )
    return output


def segment_row(run: dict[str, object]) -> dict[str, object]:
    rows = run["segments"]
    scenarios = run["scenarios"]
    summary = run["summary"]
    protocol = run["protocol"]
    assert isinstance(rows, list) and isinstance(scenarios, list)
    assert isinstance(summary, dict) and isinstance(protocol, dict)
    unique = {row.get("segment_provenance_id") for row in rows if row.get("segment_provenance_id")}
    reinforced = {row.get("segment_provenance_id") for row in rows if _bool(row.get("reinforced"))}
    deleted = {row.get("segment_provenance_id") for row in rows if _bool(row.get("segment_deleted_before_end"))}
    created = {row.get("created_segment_id") for row in scenarios if row.get("created_segment_id")}
    final_count = _int(summary.get("final_segment_count"))
    total_neurons = _int(protocol.get("encoder_sizes", {}).get("total")) * _int(protocol.get("neurons_per_column"))  # type: ignore[union-attr]
    total_columns = _int(protocol.get("encoder_sizes", {}).get("total"))  # type: ignore[union-attr]
    return {
        "L_match": run["L_match"],
        "existing_segment_rows": len(rows),
        "existing_segment_reuse_count": sum(_bool(row.get("selected_for_scenario")) for row in rows),
        "reinforced_existing_segment_count": len(reinforced - {None}),
        "created_segment_count": len(created - {None, ""}),
        "deleted_segment_count": len(deleted - {None}),
        "final_segment_count": final_count,
        "mean_segments_per_neuron": final_count / total_neurons if total_neurons else 0.0,
        "mean_segments_per_column": final_count / total_columns if total_columns else 0.0,
        "mean_segment_age": _mean(_float(row.get("segment_age")) for row in rows),
        "mean_segment_weight_sum": _mean(_float(row.get("segment_weight_sum")) for row in rows),
        "unique_segments_inspected": len(unique),
    }


def density_row(run: dict[str, object]) -> dict[str, object]:
    rows = run["density"]
    oracle = run["oracle"]
    scenarios = run["scenarios"]
    assert isinstance(rows, list) and isinstance(oracle, list)
    assert isinstance(scenarios, list)
    raw = [_float(row.get("raw_predicted_column_count")) for row in rows]
    emitted = [_float(row.get("emitted_column_count")) for row in rows]
    raw_total = sum(raw)
    emitted_total = sum(emitted)
    nonfinite = 0
    for row in rows:
        for field in ("decoded_passenger", "absolute_percentage_error"):
            value = row.get(field)
            if value not in (None, ""):
                try:
                    nonfinite += not math.isfinite(float(value))
                except (TypeError, ValueError):
                    nonfinite += 1
    return {
        "L_match": run["L_match"],
        "mean_raw_candidate_columns": _mean(raw),
        "mean_emitted_columns": _mean(emitted),
        "peak_emitted_columns": max(emitted, default=0.0),
        "suppression_ratio": 1.0 - emitted_total / raw_total if raw_total else 0.0,
        "weekday_emitted_count": sum(_float(row.get("weekday_emitted_column_count")) for row in rows),
        "time_emitted_count": sum(_float(row.get("time_emitted_column_count")) for row in rows),
        "passenger_emitted_count": sum(_float(row.get("passenger_emitted_column_count")) for row in rows),
        "mean_posthoc_false_emitted_ratio": _mean(
            _float(row.get("emitted_false_column_ratio")) for row in oracle
        ),
        "invalid_firing_time_count": sum(
            _bool(row.get("predicted_time_available"))
            and not _bool(row.get("predicted_time_valid"))
            for row in scenarios
        ),
        "NaN_or_Inf_count": nonfinite,
    }


def ambiguity_row(run: dict[str, object]) -> dict[str, object]:
    rows = run["segments"]
    assert isinstance(rows, list)
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("actual_record_index")), str(row.get("field")), str(row.get("encoded_column")))].append(row)
    eligible_counts = []
    neuron_counts = []
    best_overlaps = []
    selected_overlaps = []
    for group in groups.values():
        eligible = [row for row in group if _bool(row.get("eligible"))]
        eligible_counts.append(len(eligible))
        neuron_counts.append(len({row.get("segment_target_neuron") for row in eligible}))
        best_overlaps.append(max((_int(row.get("actual_overlap")) for row in group), default=0))
        selected_overlaps.extend(_int(row.get("actual_overlap")) for row in group if _bool(row.get("selected_as_best_matching")))
    return {
        "L_match": run["L_match"],
        "observed_column_events": len(groups),
        "mean_best_overlap": _mean(best_overlaps),
        "median_best_overlap": _median(best_overlaps),
        "matching_segment_count": sum(eligible_counts),
        "eligible_segment_count": sum(eligible_counts),
        "multiple_segment_ambiguity_rate": _mean(count > 1 for count in eligible_counts),
        "multiple_neuron_ambiguity_rate": _mean(count > 1 for count in neuron_counts),
        "mean_selected_segment_overlap": _mean(selected_overlaps),
        "selected_segment_count": len(selected_overlaps),
    }


def trajectory_row(run: dict[str, object]) -> dict[str, object]:
    rows = run["trajectory"]
    assert isinstance(rows, list)
    actual = [row for row in rows if row.get("trajectory_kind") == "actual_observation"]
    autonomous = [row for row in rows if row.get("trajectory_kind") == "autonomous_rollout"]
    def rate(group: list[dict[str, str]], field: str) -> float:
        return _mean(_bool(row.get(field)) for row in group)
    return {
        "L_match": run["L_match"],
        "actual_rows": len(actual),
        "autonomous_rows": len(autonomous),
        "actual_source_column_retention": rate(actual, "present_in_next_context"),
        "autonomous_source_column_retention": rate(autonomous, "present_in_next_context"),
        "actual_LOST_INTERCOLUMN_rate": _mean(row.get("most_recent_loss_stage") == "LOST_INTERCOLUMN" for row in actual),
        "autonomous_LOST_INTERCOLUMN_rate": _mean(row.get("most_recent_loss_stage") == "LOST_INTERCOLUMN" for row in autonomous),
        "actual_EMITTED_NOT_PROPAGATED_rate": _mean(row.get("most_recent_loss_stage") == "EMITTED_NOT_PROPAGATED" for row in actual),
        "autonomous_EMITTED_NOT_PROPAGATED_rate": _mean(row.get("most_recent_loss_stage") == "EMITTED_NOT_PROPAGATED" for row in autonomous),
        "mean_recovery_count": _mean(_float(row.get("recovery_count")) for row in rows),
    }


def classify_result(
    summaries: list[dict[str, object]],
    horizons: list[dict[str, object]],
    scenarios: list[dict[str, object]],
    density: list[dict[str, object]],
    ambiguity: list[dict[str, object]],
    bootstrap_rows: list[dict[str, object]],
) -> tuple[str, str]:
    """Apply the preregistered decision tree without changing any run."""

    by_l = {_int(row["L_match"]): row for row in summaries}
    horizon = {
        (_int(row["L_match"]), _int(row["horizon_step"])): _float(row["MAPE"])
        for row in horizons
    }
    scenario = {
        (_int(row["L_match"]), str(row["scenario"])): _float(row["rate"])
        for row in scenarios
    }
    density_by_l = {_int(row["L_match"]): row for row in density}
    ambiguity_by_l = {_int(row["L_match"]): row for row in ambiguity}
    bootstrap = {str(row["comparison"]): row for row in bootstrap_rows}
    l4 = _float(by_l[4]["MAPE"])
    l3 = _float(by_l[3]["MAPE"])
    l2 = _float(by_l[2]["MAPE"])
    l2_ci_high = _float(bootstrap["L4_vs_L2"]["ci95_high"])
    l3_ci_high = _float(bootstrap["L4_vs_L3"]["ci95_high"])

    if l3 < l2 and l3 < l4 and l3_ci_high < 0.0:
        return "LMATCH3_PREFERRED_DIAGNOSTIC", "TEST_LMATCH3_AT_500"
    if l2 > l4 * 1.02:
        return "LMATCH2_REJECTED", "RETURN_TO_INTERCOLUMN_COMPETITION"
    if abs(l2 - l4) <= max(1e-12, l4 * 0.01) and l2_ci_high >= 0.0:
        return "LMATCH_NOT_PRIMARY_BOTTLENECK", "LMATCH_NOT_PRIMARY_BOTTLENECK"

    later_improvements = sum(horizon[(2, step)] < horizon[(4, step)] for step in range(2, 6))
    scenario2_increased = scenario[(2, "scenario2")] > scenario[(4, "scenario2")]
    scenario3_decreased = scenario[(2, "scenario3")] < scenario[(4, "scenario3")]
    density_growth = _float(density_by_l[2]["mean_emitted_columns"]) / max(
        _float(density_by_l[4]["mean_emitted_columns"]), 1e-12
    )
    ambiguity_growth = _float(ambiguity_by_l[2]["multiple_neuron_ambiguity_rate"]) / max(
        _float(ambiguity_by_l[4]["multiple_neuron_ambiguity_rate"]), 1e-12
    )
    segment_growth = _float(by_l[2]["final_segment_count"]) / max(
        _float(by_l[4]["final_segment_count"]), 1e-12
    )
    ambiguity_increase = (
        _float(ambiguity_by_l[2]["multiple_neuron_ambiguity_rate"])
        - _float(ambiguity_by_l[4]["multiple_neuron_ambiguity_rate"])
    )
    ambiguity_unstable = ambiguity_growth > 1.5 and ambiguity_increase > 0.05
    growth_unstable = density_growth > 1.25 or segment_growth > 1.25
    if l2 < l4 and (ambiguity_unstable or growth_unstable):
        return (
            "LMATCH2_IMPROVES_ERROR_WITH_INSTABILITY",
            (
                "DIAGNOSE_WRONG_SEGMENT_REINFORCEMENT"
                if ambiguity_unstable
                else "DIAGNOSE_SEGMENT_GROWTH_INSTABILITY"
            ),
        )
    if horizon[(2, 1)] < horizon[(4, 1)] and later_improvements < 3 and scenario2_increased:
        return (
            "LMATCH2_CAUSES_RECURRENT_ERROR_REINFORCEMENT",
            "DIAGNOSE_WRONG_SEGMENT_REINFORCEMENT",
        )
    if (
        l2 < l4
        and l2_ci_high < 0.0
        and later_improvements >= 3
        and scenario3_decreased
        and _float(by_l[2]["coverage"]) == 1.0
    ):
        return "LMATCH2_MECHANISM_SUPPORTED", "EXTEND_LMATCH2_TO_500"
    if l2 < l4 and scenario2_increased:
        return (
            "LMATCH2_CAUSES_RECURRENT_ERROR_REINFORCEMENT",
            "DIAGNOSE_WRONG_SEGMENT_REINFORCEMENT",
        )
    return "LMATCH_NOT_PRIMARY_BOTTLENECK", "LMATCH_NOT_PRIMARY_BOTTLENECK"


def analyze(
    run_dirs: dict[int, Path],
    output_dir: Path,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    runs = {l_match: summarize_run(l_match, path) for l_match, path in run_dirs.items()}
    protocols = {l_match: run["protocol"] for l_match, run in runs.items()}
    protocols = {key: value for key, value in protocols.items() if isinstance(value, dict)}
    checks = validate_protocols(protocols)
    checks["run_output_sha256"] = {
        str(l_match): {
            path.name: file_sha256(path)
            for path in sorted(run_dirs[l_match].iterdir())
            if path.is_file()
            and path.suffix in {".csv", ".json"}
        }
        for l_match in EXPECTED_L_MATCHES
    }
    checkpoint_checks: dict[str, object] = {}
    try:
        from experiments.fig9_strict_reproduction import (
            load_strict_checkpoint,
            model_long_term_fingerprint,
            model_rng_fingerprint,
        )

        for l_match in EXPECTED_L_MATCHES:
            checkpoint_path = run_dirs[l_match] / "checkpoint.pkl"
            if not checkpoint_path.exists():
                checkpoint_checks[str(l_match)] = {"available": False}
                continue
            first = load_strict_checkpoint(checkpoint_path)
            second = load_strict_checkpoint(checkpoint_path)
            first_model = first["model"]
            second_model = second["model"]
            first_long = model_long_term_fingerprint(first_model)  # type: ignore[arg-type]
            first_rng = model_rng_fingerprint(first_model)  # type: ignore[arg-type]
            checkpoint_checks[str(l_match)] = {
                "available": True,
                "recorded_L_match": first.get("L_match"),
                "model_fingerprint": first_long,
                "RNG_fingerprint": first_rng,
                "repeat_reload_model_identical": first_long
                == model_long_term_fingerprint(second_model),  # type: ignore[arg-type]
                "repeat_reload_RNG_identical": first_rng
                == model_rng_fingerprint(second_model),  # type: ignore[arg-type]
            }
    except (ImportError, KeyError, ValueError, TypeError) as error:
        checkpoint_checks["analysis_error"] = str(error)
    checks["checkpoint_reload"] = checkpoint_checks

    summaries = [summary_row(runs[l_match]) for l_match in EXPECTED_L_MATCHES]
    baseline_mape = _float(summaries[0]["MAPE"])
    for row in summaries:
        difference = _float(row["MAPE"]) - baseline_mape
        row["absolute_difference_vs_L4"] = difference
        row["relative_difference_vs_L4"] = (
            difference / baseline_mape if baseline_mape else 0.0
        )
    horizons = [row for l_match in EXPECTED_L_MATCHES for row in horizon_rows(runs[l_match])]
    scenarios = [row for l_match in EXPECTED_L_MATCHES for row in scenario_rows(runs[l_match])]
    fields = [row for l_match in EXPECTED_L_MATCHES for row in grouped_scenario_rows(runs[l_match], "field")]
    ranges = [row for l_match in EXPECTED_L_MATCHES for row in grouped_scenario_rows(runs[l_match], "record_range")]
    segments = [segment_row(runs[l_match]) for l_match in EXPECTED_L_MATCHES]
    density = [density_row(runs[l_match]) for l_match in EXPECTED_L_MATCHES]
    ambiguity = [ambiguity_row(runs[l_match]) for l_match in EXPECTED_L_MATCHES]
    trajectory = [trajectory_row(runs[l_match]) for l_match in EXPECTED_L_MATCHES]

    bootstrap_rows = []
    prediction_errors: dict[int, dict[int, tuple[float, float]]] = {}
    for l_match, run in runs.items():
        rows = run["predictions"]
        assert isinstance(rows, list)
        prediction_errors[l_match] = {
            _int(row.get("input_index")): (
                _float(row.get("absolute_error")),
                _float(row.get("target")),
            )
            for row in rows
        }
    for left, right in ((4, 3), (4, 2), (3, 2)):
        result = paired_bootstrap(
            prediction_errors[left], prediction_errors[right],
            samples=bootstrap_samples, seed=bootstrap_seed,
        )
        bootstrap_rows.append({"comparison": f"L{left}_vs_L{right}", **result})

    formal_inputs = all(
        _int(protocol.get("data_file_limit")) == 250
        and _int(protocol.get("warmup_length")) == 200
        for protocol in protocols.values()
    )
    checks["formal_250_inputs"] = formal_inputs
    conclusion = "COMPARISON_INVALID"
    recommendation = "WITHHELD_PROTOCOL_MISMATCH"
    if checks["comparison_valid"] and not formal_inputs:
        conclusion = "SMOKE_ONLY_NO_MECHANISM_CONCLUSION"
        recommendation = "WITHHELD_SMOKE_ONLY"
    elif checks["comparison_valid"]:
        conclusion, recommendation = classify_result(
            summaries, horizons, scenarios, density, ambiguity, bootstrap_rows
        )
    checks["decision"] = {
        "conclusion": conclusion,
        "recommended_next_step": recommendation,
        "based_on_formal_results_only_when_formal_inputs_are_supplied": True,
    }

    write_csv(output_dir / "lmatch_ablation_summary.csv", summaries)
    write_csv(output_dir / "lmatch_ablation_horizon_summary.csv", horizons)
    write_csv(output_dir / "lmatch_ablation_scenario_summary.csv", scenarios)
    write_csv(output_dir / "lmatch_ablation_field_summary.csv", fields)
    write_csv(output_dir / "lmatch_ablation_record_range_summary.csv", ranges)
    write_csv(output_dir / "lmatch_ablation_segment_summary.csv", segments)
    write_csv(output_dir / "lmatch_ablation_density_summary.csv", density)
    write_csv(output_dir / "lmatch_ablation_ambiguity_summary.csv", ambiguity)
    write_csv(output_dir / "lmatch_ablation_trajectory_summary.csv", trajectory)
    write_csv(output_dir / "lmatch_ablation_bootstrap_ci.csv", bootstrap_rows)
    write_json(output_dir / "lmatch_ablation_protocol_comparison.json", {"protocols": protocols, **checks})
    write_json(output_dir / "lmatch_ablation_consistency_checks.json", checks)

    report = [
        "# Fig.9 real L_match ablation",
        "",
        "This is a nonpaper diagnostic. Strict defaults remain unchanged.",
        "Ground truth and future covariates are not used for model selection.",
        "",
        f"Comparison valid: `{checks['comparison_valid']}`",
        f"Decision: `{conclusion}`",
        f"Recommended next step: `{recommendation}`",
        "",
        "## Main results",
        "",
        "| L_match | MAPE | coverage | runtime (s) | final segments |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        report.append(
            f"| {row['L_match']} | {row['MAPE']} | {row['coverage']} | "
            f"{row['runtime_seconds']} | {row['final_segment_count']} |"
        )
    report.extend([
        "",
        "## Interpretation guardrail",
        "",
        "Step 2 onward follows recurrently diverged model trajectories. Differences "
        "therefore are paired outcomes, not fixed-state local threshold effects.",
        "",
        "No 50-record smoke result is a formal mechanism conclusion.",
    ])
    (output_dir / "FIG9_LMATCH_ABLATION_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir-l4", type=Path, required=True)
    parser.add_argument("--run-dir-l3", type=Path, required=True)
    parser.add_argument("--run-dir-l2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    checks = analyze(
        {4: args.run_dir_l4, 3: args.run_dir_l3, 2: args.run_dir_l2},
        args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(checks, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
