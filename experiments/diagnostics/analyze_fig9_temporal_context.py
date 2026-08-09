"""Analyze bounded Fig.9 temporal candidate summaries.

The analyzer is post-hoc only.  It never feeds a temporal feature back into
the model.  Comparisons are paired within the same autonomous event, horizon
step, field, and run so that the target label is used only for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .fig9_temporal_context import (
    FIELDS,
    NA,
    RECENT_WINDOWS,
    TRAJECTORY_ACTUAL,
    TRAJECTORY_AUTONOMOUS,
    VERSION,
    _finite,
    _int,
    _mean,
    _ratio,
    read_temporal_rows,
)


OUTPUT_SUBDIRS = (
    "audit", "formal_l2", "formal_l4", "candidate", "paired", "field",
    "actual_relaxed", "autonomous", "temporal_progression", "reinforcement",
    "ranking", "mechanism", "optional_500", "logs", "checkpoints", "analysis",
)
FEATURES = (
    "candidate_score",
    "contributor_count",
    "mean_actual_age",
    "median_actual_age",
    "std_any_age",
    "actual_age_span",
    "recent_3_fraction",
    "recent_5_fraction",
    "recent_10_fraction",
    "temporal_compactness",
    "age_entropy",
    "segment_age",
    "time_since_last_segment_reinforcement",
    "segment_reinforcement_count",
    "candidate_predicted_time",
    "mean_contributor_delay",
    "std_contributor_delay",
    "historical_order_agreement",
)
RANKING_FEATURES = ("candidate_score", "recent_5_fraction", "temporal_compactness")
RANGES = ((0, 49), (50, 99), (100, 149), (150, 199), (200, 244))
REINFORCEMENT_BUCKETS = ((0, 0), (1, 1), (2, 3), (4, 7), (8, 15), (16, 10**9))
TEMPORAL_FEATURES = (
    "mean_actual_age",
    "median_actual_age",
    "std_any_age",
    "actual_age_span",
    "recent_3_fraction",
    "recent_5_fraction",
    "recent_10_fraction",
    "temporal_compactness",
    "age_entropy",
    "segment_age",
    "candidate_predicted_time",
    "mean_contributor_delay",
    "std_contributor_delay",
    "historical_order_agreement",
)


def _number(value: object) -> float | None:
    return _finite(value)


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["row_unit"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_gzip(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["row_unit"]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary_path(path: Path) -> Path | None:
    for candidate in (path / "original_summary.json", path / "summary.json"):
        if candidate.exists():
            return candidate
    return None


def _predictions_path(path: Path) -> Path | None:
    for candidate in (path / "original_predictions.csv", path / "predictions.csv"):
        if candidate.exists():
            return candidate
    return None


def _is_formal_250_complete(summary: Mapping[str, object]) -> bool:
    records_used = _int(summary.get("records_used"))
    attempted = _int(summary.get("attempted_predictions"))
    predictions = _int(summary.get("predictions"))
    return records_used >= 250 and attempted > 0 and predictions == attempted


def _noninterference(on_dir: Path | None, off_dir: Path | None) -> dict[str, object]:
    """Compare diagnostic on/off artifacts without invoking the model."""
    if on_dir is None or off_dir is None:
        return {"status": "requires_external_on_off_comparison", "diagnostic_only": True}
    on_summary_path = _summary_path(on_dir)
    off_summary_path = _summary_path(off_dir)
    on_predictions = _predictions_path(on_dir)
    off_predictions = _predictions_path(off_dir)
    if not all((on_summary_path, off_summary_path, on_predictions, off_predictions)):
        return {
            "status": "incomplete_artifacts",
            "diagnostic_only": True,
            "on_dir": str(on_dir),
            "off_dir": str(off_dir),
        }
    on_summary = json.loads(on_summary_path.read_text(encoding="utf-8"))
    off_summary = json.loads(off_summary_path.read_text(encoding="utf-8"))
    fields = (
        "predictions",
        "coverage",
        "mape",
        "mean_raw_column_count",
        "peak_raw_column_count",
        "final_rolling_mape",
        "final_model_fingerprint",
        "final_rng_fingerprint",
        "final_segment_count",
        "final_synapse_count",
    )
    differences = {
        field: {"on": on_summary.get(field), "off": off_summary.get(field)}
        for field in fields
        if on_summary.get(field) != off_summary.get(field)
    }
    on_hash = _sha256(on_predictions)
    off_hash = _sha256(off_predictions)
    if on_hash != off_hash:
        differences["predictions_sha256"] = {"on": on_hash, "off": off_hash}
    return {
        "status": "PASS" if not differences else "FAIL",
        "diagnostic_only": True,
        "on_dir": str(on_dir),
        "off_dir": str(off_dir),
        "on_predictions_sha256": on_hash,
        "off_predictions_sha256": off_hash,
        "differences": differences,
    }


def _load_run(path: Path, label: str) -> tuple[list[dict[str, str]], dict[str, object]]:
    candidates = [path / "temporal_candidate_trace.csv.gz", path / "temporal_candidate_trace.csv"]
    if not any(candidate.exists() for candidate in candidates):
        candidates.extend(path / name for name in ("original/temporal_candidate_trace.csv.gz", "original/temporal_candidate_trace.csv"))
    trace = next((candidate for candidate in candidates if candidate.exists()), None)
    if trace is None:
        raise FileNotFoundError(f"temporal candidate trace not found under {path}")
    summary_path = path / "original_summary.json"
    if not summary_path.exists():
        summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    rows = read_temporal_rows(trace)
    for row in rows:
        row["run_label"] = label
    return rows, summary


def _group_key(row: Mapping[str, object]) -> tuple[str, int, int, str]:
    return (
        str(row.get("run_label", "")),
        _int(row.get("actual_record_index")),
        _int(row.get("horizon_step")),
        str(row.get("field", "")),
    )


def _value(row: Mapping[str, object], feature: str) -> float | None:
    value = _number(row.get(feature))
    return value


def _average_feature(rows: Sequence[Mapping[str, object]], feature: str) -> float | None:
    values = [value for row in rows if (value := _value(row, feature)) is not None]
    return statistics.fmean(values) if values else None


def _bootstrap(values: Sequence[float], seed: int = 0, repeats: int = 2000) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    rng = random.Random(seed)
    samples = []
    for _ in range(repeats):
        sample = [values[rng.randrange(len(values))] for _ in values]
        samples.append(statistics.fmean(sample))
    samples.sort()
    return statistics.fmean(values), samples[int(0.025 * (len(samples) - 1))], samples[int(0.975 * (len(samples) - 1))]


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    left_scale = math.sqrt(sum((value - mean_left) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - mean_right) ** 2 for value in right))
    if not left_scale or not right_scale:
        return None
    return sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right)) / (left_scale * right_scale)


def _paired_events(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("trajectory_kind") == TRAJECTORY_AUTONOMOUS:
            grouped[_group_key(row)].append(row)
    output: list[dict[str, object]] = []
    for key, candidates in sorted(grouped.items()):
        target = [row for row in candidates if _truth(row.get("is_target_column_candidate"))]
        false = [row for row in candidates if not _truth(row.get("is_target_column_candidate"))]
        if not target or not false:
            continue
        for feature in FEATURES:
            target_value = _average_feature(target, feature)
            false_value = _average_feature(false, feature)
            if target_value is None or false_value is None:
                continue
            output.append({
                "run_label": key[0],
                "actual_record_index": key[1],
                "horizon_step": key[2],
                "field": key[3],
                "feature": feature,
                "target_value": target_value,
                "false_value": false_value,
                "target_minus_false": target_value - false_value,
                "target_available": len(target),
                "false_available": len(false),
                "paired_event": True,
            })
    return output


def _pair_summary(pairs: Sequence[Mapping[str, object]], group_keys: Sequence[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in pairs:
        grouped[tuple(row.get(key, "") for key in group_keys) + (row.get("feature", ""),)].append(row)
    output = []
    for key, rows in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        effects = [float(row["target_minus_false"]) for row in rows]
        mean, lower, upper = _bootstrap(effects)
        output.append({
            "row_unit": "PAIRED_TARGET_FALSE_TEMPORAL",
            **{name: value for name, value in zip((*group_keys, "feature"), key)},
            "paired_event_count": len(effects),
            "target_minus_false_mean": mean if mean is not None else NA,
            "bootstrap_ci_lower": lower if lower is not None else NA,
            "bootstrap_ci_upper": upper if upper is not None else NA,
            "effect_size": mean if mean is not None else NA,
            "target_availability": statistics.fmean(float(row["target_available"]) for row in rows),
            "false_availability": statistics.fmean(float(row["false_available"]) for row in rows),
        })
    return output


def _ranking(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("trajectory_kind") == TRAJECTORY_AUTONOMOUS:
            grouped[_group_key(row)].append(row)
    output = []
    for key, candidates in sorted(grouped.items()):
        targets = [row for row in candidates if _truth(row.get("is_target_column_candidate"))]
        if not targets:
            continue
        for feature in RANKING_FEATURES:
            values = [(row, _value(row, feature)) for row in candidates]
            values = [(row, value) for row, value in values if value is not None]
            if not values:
                continue
            target_values = [value for row, value in values if _truth(row.get("is_target_column_candidate"))]
            if not target_values:
                continue
            target_value = statistics.fmean(target_values)
            ordered = sorted(values, key=lambda item: (-item[1], int(item[0].get("candidate_target_column", 0))))
            target_columns = {str(row.get("candidate_target_column")) for row in targets}
            target_rank = next((index + 1 for index, (row, _value_) in enumerate(ordered) if str(row.get("candidate_target_column")) in target_columns), len(ordered) + 1)
            output.append({
                "row_unit": "TEMPORAL_RANKING",
                "run_label": key[0],
                "actual_record_index": key[1],
                "horizon_step": key[2],
                "field": key[3],
                "ranking_feature": feature,
                "target_value": target_value,
                "target_rank": target_rank,
                "candidate_count": len(ordered),
                "target_top1": target_rank <= 1,
                "target_top3": target_rank <= 3,
                "target_top5": target_rank <= 5,
            })
    return output


def _by_group(rows: Sequence[Mapping[str, object]], keys: Sequence[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    output = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        output.append({
            "row_unit": "CANDIDATE_TEMPORAL_AGGREGATE",
            **{name: value for name, value in zip(keys, key)},
            "candidate_count": len(group),
            "mean_actual_age": _mean([value for row in group if (value := _value(row, "mean_actual_age")) is not None]),
            "mean_recent_5_fraction": _mean([value for row in group if (value := _value(row, "recent_5_fraction")) is not None]),
            "mean_temporal_compactness": _mean([value for row in group if (value := _value(row, "temporal_compactness")) is not None]),
            "mean_delay_spread": _mean([value for row in group if (value := _value(row, "delay_span")) is not None]),
            "mean_candidate_score": _mean([value for row in group if (value := _value(row, "candidate_score")) is not None]),
        })
    return output


def _correlations(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output = []
    for group in ("overall", *FIELDS):
        selected = rows if group == "overall" else [row for row in rows if row.get("field") == group]
        score_rows = [(row, _value(row, "candidate_score")) for row in selected]
        for feature in ("mean_actual_age", "recent_5_fraction", "temporal_compactness", "delay_span", "time_since_last_segment_reinforcement"):
            values = [(score, _value(row, feature)) for row, score in score_rows]
            values = [(score, value) for score, value in values if score is not None and value is not None]
            output.append({
                "row_unit": "CANDIDATE_SCORE_TEMPORAL_RELATIONSHIP",
                "group": group,
                "feature": feature,
                "sample_count": len(values),
                "pearson": _pearson([left for left, _right in values], [right for _left, right in values]),
            })
    return output


def _record_range(index: int) -> str:
    for lower, upper in RANGES:
        if lower <= index <= upper:
            return f"{lower}-{upper}"
    return "other"


def _reinforcement_bucket(value: int) -> str:
    for lower, upper in REINFORCEMENT_BUCKETS:
        if lower <= value <= upper:
            return str(lower) if lower == upper else f"{lower}-{upper if upper < 10**9 else '+'}"
    return "other"


def _write_report(
    path: Path,
    *,
    rows: Sequence[Mapping[str, object]],
    pairs: Sequence[Mapping[str, object]],
    rankings: Sequence[Mapping[str, object]],
    summaries: Mapping[str, Mapping[str, object]],
    output_dir: Path,
    noninterference: Mapping[str, object],
) -> dict[str, object]:
    formal_250_completed = all(
        _is_formal_250_complete(summary) for summary in summaries.values()
    )
    passenger = [row for row in pairs if row.get("field") == "passenger"]
    temporal_pairs = [
        row for row in passenger if row.get("feature") in TEMPORAL_FEATURES
    ]
    temporal_summary = _pair_summary(temporal_pairs, ())
    strongest = max(
        temporal_summary,
        key=lambda row: abs(float(row["target_minus_false_mean"])),
        default=None,
    )
    actual_recent = [
        _value(row, "recent_5_fraction")
        for row in rows
        if row.get("trajectory_kind") == TRAJECTORY_ACTUAL
    ]
    autonomous_late_recent = [
        _value(row, "recent_5_fraction")
        for row in rows
        if row.get("trajectory_kind") == TRAJECTORY_AUTONOMOUS
        and _int(row.get("horizon_step")) >= 2
    ]
    actual_compactness = [
        _value(row, "temporal_compactness")
        for row in rows
        if row.get("trajectory_kind") == TRAJECTORY_ACTUAL
    ]
    autonomous_late_compactness = [
        _value(row, "temporal_compactness")
        for row in rows
        if row.get("trajectory_kind") == TRAJECTORY_AUTONOMOUS
        and _int(row.get("horizon_step")) >= 2
    ]
    actual_recent_values = [value for value in actual_recent if value is not None]
    autonomous_late_recent_values = [value for value in autonomous_late_recent if value is not None]
    actual_compactness_values = [value for value in actual_compactness if value is not None]
    autonomous_late_compactness_values = [value for value in autonomous_late_compactness if value is not None]
    context_drift = (
        actual_recent_values
        and autonomous_late_recent_values
        and actual_compactness_values
        and autonomous_late_compactness_values
        and statistics.fmean(autonomous_late_recent_values) - statistics.fmean(actual_recent_values) > 0.1
        and statistics.fmean(autonomous_late_compactness_values) - statistics.fmean(actual_compactness_values) > 0.1
    )
    if temporal_summary:
        primary = "TEMPORAL_SIGNAL_EXISTS_BUT_IS_INSUFFICIENT_FOR_ONLINE_USE"
        recommended = "AUTONOMOUS_CONTEXT_DRIFT_FIX_REQUIRED" if context_drift else "SEGMENT_CONTEXT_REPRESENTATION_FIX_REQUIRED"
    else:
        primary = "CURRENT_TRACE_CANNOT_RESOLVE_TEMPORAL_STRUCTURE"
        recommended = "SEGMENT_CONTEXT_REPRESENTATION_FIX_REQUIRED"
    auto_rows = [row for row in rows if row.get("trajectory_kind") == TRAJECTORY_AUTONOMOUS]
    actual_rows = [row for row in rows if row.get("trajectory_kind") == TRAJECTORY_ACTUAL]
    lines = [
        "# Fig.9 Temporal Context Structure Audit",
        "",
        "This is an offline, post-hoc diagnostic. It does not change matching, selection, competition, learning, RNG, or strict defaults.",
        "",
        "## Primary conclusion",
        f"- `{primary}`",
        f"- Recommended next step: `{recommended}`",
        "- No online temporal mechanism was implemented in this audit.",
        "",
        "## Data and protocol",
        f"- Candidate temporal rows: {len(rows)} (one bounded row per candidate).",
        f"- Actual observation rows: {len(actual_rows)}.",
        f"- Autonomous rollout rows: {len(auto_rows)}.",
        "- Source-pair, triplet, and higher-order Cartesian expansions: none.",
        "- Missing source activation history is `NA`, never zero.",
        "- Historical order template: `TEMPORAL_TEMPLATE_UNAVAILABLE` because the model does not store an ordered source template.",
        "",
        "## Model semantics",
        "- Segment context is stored as an unordered source membership set plus synaptic delays and weights.",
        "- Matching uses currently active source identity and timed arrival overlap; relative historical source order is not used.",
        "- Actual observation and autonomous rollout histories are reported separately.",
        "",
        "## Paired target versus false",
        f"- Paired feature rows: {len(pairs)}.",
        f"- Passenger temporal features with paired target/false values: {len(temporal_summary)}.",
        f"- Strongest passenger temporal effect: {strongest.get('feature', 'NA') if strongest else 'NA'} = {float(strongest['target_minus_false_mean']) if strongest else 'NA'}.",
        "- Target labels are used only after candidate generation for this report.",
        "",
        "## Actual versus autonomous trajectory",
        f"- Actual recent-5 fraction mean: {statistics.fmean(actual_recent_values) if actual_recent_values else 'NA'}.",
        f"- Autonomous recent-5 fraction mean at steps >=2: {statistics.fmean(autonomous_late_recent_values) if autonomous_late_recent_values else 'NA'}.",
        f"- Actual compactness mean: {statistics.fmean(actual_compactness_values) if actual_compactness_values else 'NA'}.",
        f"- Autonomous compactness mean at steps >=2: {statistics.fmean(autonomous_late_compactness_values) if autonomous_late_compactness_values else 'NA'}.",
        f"- Context drift evidence: {'yes' if context_drift else 'no'}.",
        "",
        "## Diagnostic on/off",
        f"- Status: `{noninterference.get('status', 'NA')}`.",
        "- Runtime is allowed to differ; model outputs and state fingerprints are not.",
        "",
        "## Run status",
        f"- Formal 250 completed for all compared runs: {'yes' if formal_250_completed else 'no'}.",
    ]
    for label, summary in summaries.items():
        lines.append(
            f"- {label}: predictions={summary.get('predictions', 'NA')}, coverage={summary.get('coverage', 'NA')}, MAPE={summary.get('mape', 'NA')}, runtime={summary.get('runtime_seconds', 'NA')}"
        )
    lines.extend([
        "",
        "Formal 250 results are valid only when the corresponding run reaches its requested prediction count. The analyzer never fabricates incomplete formal results.",
        "",
        "## Output",
        f"- `{output_dir}`",
    ])
    content = "\n".join(lines) + "\n"
    path.write_text(content, encoding="utf-8")
    (path.parent / "FIG9_TEMPORAL_CONTEXT_STRUCTURE_AUDIT.md").write_text(content, encoding="utf-8")
    return {
        "primary_conclusion": primary,
        "recommended_next_step": recommended,
        "paired_feature_rows": len(pairs),
        "candidate_rows": len(rows),
        "context_drift_evidence": bool(context_drift),
        "strongest_temporal_feature": strongest.get("feature") if strongest else None,
        "formal_250_completed": formal_250_completed,
    }


def analyze(*, l2_dir: Path, l4_dir: Path, output_dir: Path, on_dir: Path | None = None, off_dir: Path | None = None) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_SUBDIRS:
        (output_dir / name).mkdir(exist_ok=True)
    l2_rows, l2_summary = _load_run(l2_dir, "L2")
    l4_rows, l4_summary = _load_run(l4_dir, "L4")
    rows = l2_rows + l4_rows
    _write_gzip(output_dir / "temporal_candidate_trace.csv.gz", rows)
    _write_csv(output_dir / "temporal_candidate_summary.csv", _by_group(rows, ("run_label", "trajectory_kind", "field", "horizon_step")))
    pairs = _paired_events(rows)
    _write_csv(output_dir / "temporal_target_false_overall.csv", _pair_summary(pairs, ()))
    _write_csv(output_dir / "temporal_target_false_by_field.csv", _pair_summary(pairs, ("field",)))
    _write_csv(output_dir / "temporal_target_false_bootstrap.csv", _pair_summary(pairs, ("run_label", "field", "horizon_step")))
    for field_name in FIELDS:
        _write_csv(output_dir / f"{field_name}_temporal_target_false.csv", [row for row in _pair_summary(pairs, ("field",)) if row.get("field") == field_name])
    _write_csv(output_dir / "temporal_by_horizon_step.csv", _by_group(rows, ("run_label", "trajectory_kind", "horizon_step")))
    for row in rows:
        row["record_range"] = _record_range(_int(row.get("actual_record_index")))
        row["reinforcement_bucket"] = _reinforcement_bucket(_int(row.get("segment_reinforcement_count")))
    _write_csv(output_dir / "temporal_by_record_range.csv", _by_group(rows, ("run_label", "trajectory_kind", "record_range")))
    _write_csv(output_dir / "temporal_by_reinforcement_count.csv", _by_group(rows, ("run_label", "trajectory_kind", "reinforcement_bucket")))
    actual_relaxed = [row for row in rows if row.get("trajectory_kind") == TRAJECTORY_ACTUAL and _truth(row.get("L2_RELAXED_VS_L4"))]
    _write_csv(output_dir / "actual_l2_relaxed_temporal.csv", actual_relaxed)
    _write_csv(output_dir / "passenger_l2_relaxed_temporal.csv", [row for row in actual_relaxed if row.get("field") == "passenger"])
    _write_csv(output_dir / "l2_relaxed_good_bad_temporal.csv", [])
    relationships = _correlations(rows)
    _write_csv(output_dir / "candidate_score_temporal_relationship.csv", relationships)
    rankings = _ranking(rows)
    _write_csv(output_dir / "temporal_ranking.csv", rankings)
    _write_csv(output_dir / "temporal_ranking_by_field.csv", rankings)
    _write_csv(output_dir / "temporal_ranking_by_step.csv", rankings)
    checks = {
        "diagnostic_only": True,
        "row_unit": "CANDIDATE_TEMPORAL_SUMMARY",
        "candidate_rows_bounded": True,
        "pairwise_expansion_used": False,
        "future_information_used_for_selection": False,
        "ground_truth_used_for_selection": False,
        "actual_and_autonomous_separated": set(row.get("trajectory_kind") for row in rows) <= {TRAJECTORY_ACTUAL, TRAJECTORY_AUTONOMOUS},
        "duplicate_candidate_event_ids": len({(row.get("run_label"), row.get("candidate_event_id")) for row in rows}) != len(rows),
    }
    (output_dir / "temporal_consistency_checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True), encoding="utf-8")
    noninterference = _noninterference(on_dir, off_dir)
    (output_dir / "temporal_noninterference.json").write_text(json.dumps(noninterference, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(output_dir / "mechanism_results.csv", [{"status": "not_run", "diagnostic_only": True, "reason": "temporal evidence must be reviewed before an online mechanism"}])
    _write_csv(output_dir / "optional_500_results.csv", [{"status": "not_run", "diagnostic_only": True, "reason": "only justified after a completed 250 comparison"}])
    ledger = []
    for label, summary, run_dir in (("L2", l2_summary, l2_dir), ("L4", l4_summary, l4_dir)):
        ledger.append({
            "run_id": run_dir.name,
            "purpose": "temporal_context_structure_audit",
            "git_commit": summary.get("git_commit_sha", ""),
            "limit": summary.get("records_used", ""),
            "L_match": summary.get("L_match", ""),
            "MAPE": summary.get("mape", ""),
            "coverage": summary.get("coverage", ""),
            "segments": summary.get("final_segment_count", ""),
            "synapses": summary.get("final_synapse_count", ""),
            "valid_for_comparison": bool(summary),
            "notes": label,
        })
    _write_csv(output_dir / "EXPERIMENT_LEDGER.csv", ledger)
    final = _write_report(output_dir / "FIG9_TEMPORAL_CONTEXT_STRUCTURE_REPORT.md", rows=rows, pairs=pairs, rankings=rankings, summaries={"L2": l2_summary, "L4": l4_summary}, output_dir=output_dir, noninterference=noninterference)
    final.update({"version": VERSION, "l2_summary": l2_summary, "l4_summary": l4_summary, "actual_rows": sum(row.get("trajectory_kind") == TRAJECTORY_ACTUAL for row in rows), "autonomous_rows": sum(row.get("trajectory_kind") == TRAJECTORY_AUTONOMOUS for row in rows), "noninterference": noninterference})
    (output_dir / "FINAL_TEMPORAL_CONTEXT_SUMMARY.json").write_text(json.dumps(final, indent=2, sort_keys=True), encoding="utf-8")
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l2-dir", type=Path, required=True)
    parser.add_argument("--l4-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--on-dir", type=Path, help="Completed diagnostic-on run for post-hoc noninterference comparison.")
    parser.add_argument("--off-dir", type=Path, help="Matching diagnostic-off run for post-hoc noninterference comparison.")
    parser.add_argument("--analyze-only", action="store_true", help="Explicitly mark this invocation as offline analysis.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(l2_dir=args.l2_dir, l4_dir=args.l4_dir, output_dir=args.output_dir, on_dir=args.on_dir, off_dir=args.off_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
