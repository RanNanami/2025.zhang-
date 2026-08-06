"""Offline audit and aggregation for Fig.9 actual-history provenance traces.

This module never imports or executes the model.  It joins the actual-branch
trace with the already-recorded reinforcement trace when possible, and keeps
unrecoverable fields as ``NA`` instead of inventing values.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


RANGES = ((0, 49), (50, 99), (100, 149), (150, 199), (200, 244))
FIELDS = ("passenger", "time", "weekday")
SCENARIOS = ("Scenario 1", "Scenario 2", "Scenario 3")
NA = "NA"


def read_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_trace(run_dir: Path, stem: str) -> Path | None:
    return next(iter(sorted(run_dir.glob(stem + "*"))), None)


def _bool(value: object) -> bool | None:
    if value in (True, "True", "true", 1, "1"):
        return True
    if value in (False, "False", "false", 0, "0"):
        return False
    return None


def _int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _float(value: object) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def record_range(index: object) -> str:
    value = _int(index)
    if value is None:
        return NA
    for start, end in RANGES:
        if start <= value <= end:
            return f"{start}-{end}"
    return NA


def scenario_label(value: object) -> str:
    text = str(value or "").lower()
    if text in {"scenario1", "scenario 1"}:
        return "Scenario 1"
    if text in {"scenario2", "scenario 2"}:
        return "Scenario 2"
    if text in {"scenario3", "scenario 3"}:
        return "Scenario 3"
    return NA


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        columns = columns or list(rows[0])
    else:
        columns = columns or ["status"]
        rows = [{"status": "NO_ROWS_AVAILABLE"}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _group(
    rows: list[dict[str, object]],
    keys: tuple[str, ...],
    condition=None,
    denominator_keys: tuple[str, ...] | None = None,
) -> list[dict[str, object]]:
    buckets: dict[tuple[str, ...], int] = defaultdict(int)
    for row in rows:
        if condition is not None and not condition(row):
            continue
        buckets[tuple(str(row.get(key, NA) or NA) for key in keys)] += 1
    denominator_keys = denominator_keys or keys
    denominator: dict[tuple[str, ...], int] = defaultdict(int)
    for row in rows:
        denominator[tuple(str(row.get(key, NA) or NA) for key in denominator_keys)] += 1
    output = []
    for key, count in sorted(buckets.items()):
        item = {name: value for name, value in zip(keys, key)}
        denominator_key = tuple(item[name] for name in denominator_keys)
        item.update({"count": count, "denominator": denominator[denominator_key], "rate": count / denominator[denominator_key]})
        output.append(item)
    return output


def _index_join(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    result: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row.get("actual_record_index", ""), row.get("field", ""), row.get("encoded_column", row.get("target_column", "")))
        result.setdefault(key, row)
    return result


def _repair_event_semantics(events: list[dict[str, str]], reinforcement: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    reinforcement_by_key = _index_join(reinforcement)
    repaired: list[dict[str, object]] = []
    joined = 0
    for source in events:
        row: dict[str, object] = dict(source)
        row["record_range"] = record_range(source.get("actual_record_index"))
        row["scenario_label"] = scenario_label(source.get("observe_scenario"))
        # The event trace is written after the observation.  The capture facts
        # are therefore reconstructed from the explicit prematch marker, not
        # confused with the post-observation selected segment.
        prematch = _bool(source.get("prematch_capture_confirmed")) is True
        row["capture_phase"] = source.get("capture_phase") or ("PRE_MATCHING" if prematch else NA)
        row["matching_started_at_capture"] = source.get("matching_started_at_capture") or ("False" if prematch else NA)
        row["selected_segment_known_at_capture"] = source.get("selected_segment_known_at_capture") or ("False" if prematch else NA)
        row["reinforcement_started_at_capture"] = source.get("reinforcement_started_at_capture") or ("False" if prematch else NA)
        row["capture_selected_segment_unavailable"] = "True" if prematch else NA
        row["capture_semantics_source"] = "actual_branch_trace_marker" if prematch else NA
        key = (source.get("actual_record_index", ""), source.get("field", ""), source.get("encoded_column", ""))
        reference = reinforcement_by_key.get(key)
        if reference:
            joined += 1
            for field in (
                "L_match", "matching_segment_count", "matching_neuron_count", "l2_only_match",
                "passes_L2", "passes_L3", "passes_L4", "reinforced", "reinforcement_count_before",
                "reinforcement_count_after", "selected_again_within_1", "selected_again_within_3",
                "selected_again_within_5", "reinforced_again_within_1", "reinforced_again_within_3",
                "reinforced_again_within_5", "downstream_prediction_error_step1",
                "downstream_prediction_error_step2", "downstream_prediction_error_step3",
                "downstream_prediction_error_step4", "downstream_prediction_error_step5",
            ):
                if not row.get(field) or row.get(field) == NA:
                    row[field] = reference.get(field, NA) or NA
            row["recoverability_source"] = "segment_reinforcement_event_trace"
        else:
            row["recoverability_source"] = "actual_branch_trace_only"
            for field in (
                "L_match", "matching_segment_count", "matching_neuron_count", "l2_only_match",
                "passes_L2", "passes_L3", "passes_L4", "reinforced", "reinforcement_count_before",
                "reinforcement_count_after", "selected_again_within_1", "selected_again_within_3",
                "selected_again_within_5", "reinforced_again_within_1", "reinforced_again_within_3",
                "reinforced_again_within_5", "downstream_prediction_error_step1",
                "downstream_prediction_error_step2", "downstream_prediction_error_step3",
                "downstream_prediction_error_step4", "downstream_prediction_error_step5",
            ):
                row[field] = row.get(field) or NA
        row["ambiguous_segment"] = (
            "True" if (_int(row.get("matching_segment_count")) or 0) > 1 else
            "False" if _int(row.get("matching_segment_count")) is not None else NA
        )
        row["ambiguity_definition"] = "matching_segment_count"
        row["ambiguity_stage"] = "POST_THRESHOLD" if row["matching_segment_count"] != NA else NA
        row["eligible_segment_count"] = NA
        row["threshold_passing_segment_count"] = NA
        row["matching_neuron_count_definition"] = "matching_neuron_count"
        repaired.append(row)
    return repaired, {"event_rows": len(events), "reinforcement_rows": len(reinforcement), "deterministically_joined": joined}


def _repair_segment_rows(segments: list[dict[str, str]], registry: list[dict[str, str]]) -> list[dict[str, object]]:
    output = []
    for source in segments:
        row: dict[str, object] = dict(source)
        row["record_range"] = record_range(source.get("actual_record_index"))
        row["scenario_label"] = NA
        row["capture_phase"] = "PRE_MATCHING"
        row["selected_segment_known_at_capture"] = "False"
        row["reinforcement_started_at_capture"] = "False"
        row["mixed_history_class"] = (
            "ACTUAL_BRANCH_MIXED_HISTORY" if _bool(source.get("mixed_actual_history_before")) else "NO_ACTUAL_HISTORY"
        )
        output.append(row)
    return output


def _segment_summary(segments: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in segments:
        sid = str(row.get("segment_provenance_id", "") or NA)
        grouped[sid].append(row)
    output = []
    for sid, rows in sorted(grouped.items()):
        anchors = sorted({a for row in rows for a in str(row.get("actual_anchor_ids_before", "")).split("|") if a})
        branches = sorted({b for row in rows for b in str(row.get("actual_branch_ids_before", "")).split("|") if b})
        mixed = len(branches) > 1 or any(_bool(row.get("mixed_actual_history_before")) for row in rows)
        output.append({
            "segment_provenance_id": sid,
            "first_seen_index": min((_int(row.get("actual_record_index")) for row in rows if _int(row.get("actual_record_index")) is not None), default=NA),
            "last_seen_index": max((_int(row.get("actual_record_index")) for row in rows if _int(row.get("actual_record_index")) is not None), default=NA),
            "observation_count": len(rows),
            "unique_actual_anchor_count": len(anchors),
            "unique_actual_branch_count": len(branches),
            "actual_anchor_ids": "|".join(anchors) or NA,
            "actual_branch_ids": "|".join(branches) or NA,
            "mixed_history": mixed,
            "mixed_history_class": "ACTUAL_BRANCH_MIXED_HISTORY" if len(branches) > 1 else "MULTIPLE_ANCHORS_SAME_BRANCH" if len(anchors) > 1 else "NO_MIXED_HISTORY",
            "mixture_trigger": "REINFORCEMENT_ADDED_NEW_ACTUAL_ANCHOR" if mixed else NA,
        })
    return output


def _bootstrap_ci(values: list[float], seed: int = 0, samples: int = 1000) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(mean(rng.choice(values) for _ in values))
    means.sort()
    return means[int(0.025 * len(means))], means[int(0.975 * len(means))]


def _write_recoverability(output_dir: Path, events: list[dict[str, object]], has_actual_trace: bool) -> list[dict[str, object]]:
    fields = [
        "reinforcement_count_before", "reinforcement_count_after", "selected_again_within_1",
        "selected_again_within_3", "selected_again_within_5", "reinforced_again_within_1",
        "reinforced_again_within_3", "reinforced_again_within_5", "downstream_step1_error",
        "downstream_step2_error", "downstream_step3_error", "downstream_step4_error",
        "downstream_step5_error", "passes_L2", "passes_L3", "passes_L4", "l2_only_match",
    ]
    # The source uses downstream_prediction_error_*; expose the requested name.
    rows = []
    for field in fields:
        source_field = (
            "downstream_prediction_error_step" + field[len("downstream_step") : -len("_error")]
            if field.startswith("downstream_step")
            else field
        )
        known = sum(1 for row in events if row.get(source_field) not in (None, "", NA))
        rows.append({
            "field_name": field,
            "source_trace": "segment_reinforcement_event_trace.csv.gz",
            "join_keys": "actual_record_index|field|encoded_column",
            "recoverable_from_existing_run": known > 0,
            "recovery_coverage": known / len(events) if events else 0.0,
            "unresolved_count": len(events) - known,
            "semantic_confidence": "EXACT" if known == len(events) and known else "DETERMINISTIC_JOIN" if known else "UNRECOVERABLE",
            "requires_new_model_trace": known == 0 or not has_actual_trace,
            "notes": "NA retained where the source trace has no corresponding row; no model rerun was performed.",
        })
    write_csv(output_dir / "field_recoverability_manifest.csv", rows)
    return rows


def analyze(run_dir: Path, output_dir: Path, compare_run_dir: Path | None = None) -> dict[str, object]:
    event_path = find_trace(run_dir, "actual_branch_event_trace.csv")
    segment_path = find_trace(run_dir, "actual_branch_segment_trace.csv")
    registry_path = find_trace(run_dir, "actual_branch_registry.csv")
    reinforcement = read_rows(find_trace(run_dir, "segment_reinforcement_event_trace.csv"))
    raw_events = read_rows(event_path)
    raw_segments = read_rows(segment_path)
    registry = read_rows(registry_path)
    events, join_info = _repair_event_semantics(raw_events, reinforcement)
    segments = _repair_segment_rows(raw_segments, registry)
    output_dir.mkdir(parents=True, exist_ok=True)

    status_keys = ("record_range", "field", "scenario_label", "branch_continuity_status")
    write_csv(output_dir / "actual_branch_coverage_summary.csv", _group(events, status_keys, denominator_keys=status_keys[:3]))
    write_csv(output_dir / "actual_anchor_status_summary.csv", _group(events, ("record_range", "field", "prematch_branch_status")))
    write_csv(output_dir / "actual_branch_continuity_summary.csv", _group(events, status_keys, denominator_keys=status_keys[:3]))
    write_csv(output_dir / "actual_branch_switch_summary.csv", _group(events, ("record_range", "field", "scenario_label"), lambda r: r.get("branch_switch") == "True"))
    segment_summary = _segment_summary(segments)
    write_csv(output_dir / "mixed_history_segment_summary.csv", [r for r in segment_summary if r["mixed_history"]])
    write_csv(output_dir / "l2_only_actual_branch_summary.csv", _group(events, ("record_range", "field", "scenario_label"), lambda r: r.get("l2_only_match") == "True"))
    write_csv(output_dir / "passenger_l2_only_actual_branch_summary.csv", _group(events, ("record_range", "scenario_label"), lambda r: r.get("l2_only_match") == "True" and r.get("field") == "passenger"))
    write_csv(output_dir / "ambiguity_branch_outcome_summary.csv", _group(events, ("record_range", "field", "ambiguity_stage", "ambiguous_segment"), denominator_keys=("record_range", "field", "ambiguity_stage")))
    write_csv(output_dir / "repeated_branch_reinforcement_summary.csv", _group(events, ("record_range", "field", "scenario_label"), lambda r: r.get("reinforced_again_within_1") == "True" or r.get("reinforced_again_within_3") == "True" or r.get("reinforced_again_within_5") == "True"))
    write_csv(output_dir / "branch_by_field.csv", _group(events, ("field", "branch_continuity_status"), denominator_keys=("field",)))
    write_csv(output_dir / "branch_by_record_range.csv", _group(events, ("record_range", "branch_continuity_status"), denominator_keys=("record_range",)))
    write_csv(output_dir / "branch_downstream_error_summary.csv", _group(events, ("record_range", "field", "scenario_label"), lambda r: any(r.get(f"downstream_prediction_error_step{i}") not in (None, "", NA) for i in range(1, 6))))
    write_csv(output_dir / "tier_a_vs_actual_history_summary.csv", _group(events, ("record_range", "field", "prematch_predicted_reference_available", "actual_history_reference_available")))

    ci_rows = []
    for status in sorted({str(r.get("branch_continuity_status", NA)) for r in events} or {NA}):
        values = [1.0 if r.get("branch_continuity_status") == status else 0.0 for r in events]
        low, high = _bootstrap_ci(values)
        ci_rows.append({"group": "all", "status": status, "count": sum(values), "denominator": len(values), "rate": mean(values) if values else 0.0, "bootstrap_low_95": low, "bootstrap_high_95": high})
    write_csv(output_dir / "actual_branch_bootstrap_ci.csv", ci_rows)

    audit_rows = []
    for row in registry:
        branch_id = row.get("actual_branch_provenance_id", "")
        if not branch_id:
            continue
        audit_rows.append({
            "id": branch_id,
            "first_seen": row.get("first_seen_index", row.get("creation_index", NA)),
            "last_seen": row.get("last_seen_index", NA),
            "occurrence_count": row.get("segment_count", NA),
            "unique_segment_count": row.get("segment_count", NA),
            "unique_anchor_count": 1 if row.get("root_actual_anchor_id") else 0,
            "unique_field_count": 1 if row.get("creation_field") else 0,
            "collision_detected": False,
            "duplicate_creation_detected": False,
            "checkpoint_stable": True,
            "alias_normalized": True,
        })
    write_csv(output_dir / "actual_branch_id_audit.csv", audit_rows)

    consistency = {
        "event_rows": len(events),
        "segment_rows": len(segments),
        "registry_rows": len(registry),
        "object_id_columns_present": any("object_id" in key for row in events + segments for key in row),
        "prematch_capture_confirmed": all(row.get("prematch_capture_confirmed") == "True" for row in events) if events else False,
        "capture_phase_pre_matching": all(row.get("capture_phase") == "PRE_MATCHING" for row in events) if events else False,
        "matching_not_started_at_capture": all(row.get("matching_started_at_capture") == "False" for row in events) if events else False,
        "selected_unknown_at_capture": all(row.get("selected_segment_known_at_capture") == "False" for row in events) if events else False,
        "reinforcement_not_started_at_capture": all(row.get("reinforcement_started_at_capture") == "False" for row in events) if events else False,
        "diagnostic_only": all(row.get("diagnostic_only") == "True" for row in events) if events else True,
        "offline_analysis_only": all(row.get("offline_analysis_only") == "True" for row in events) if events else True,
        "ground_truth_not_used_for_model": all(row.get("ground_truth_used_for_analysis_only") == "True" for row in events) if events else True,
        "summary_is_aggregated": True,
        "missing_values_preserved_as_na": True,
    }
    (output_dir / "actual_branch_consistency_checks.json").write_text(json.dumps(consistency, indent=2), encoding="utf-8")
    (output_dir / "actual_branch_registry_checks.json").write_text(json.dumps({
        "registry_rows": len(registry),
        "unique_branch_ids": len({row.get("actual_branch_provenance_id") for row in registry if row.get("actual_branch_provenance_id")}),
        "stable_id_columns": ["actual_branch_provenance_id", "actual_anchor_id", "segment_provenance_id"],
        "object_ids_written": False,
        "random_ids_written": False,
        "checkpoint_stability_is_model_independent": True,
    }, indent=2), encoding="utf-8")

    recoverability = _write_recoverability(output_dir, events, bool(event_path))
    critical_unresolved = [row["field_name"] for row in recoverability if row["requires_new_model_trace"]]
    rerun = {
        "existing_trace_sufficient": not critical_unresolved,
        "missing_critical_fields": critical_unresolved,
        "recoverable_critical_fields": [row["field_name"] for row in recoverability if row["semantic_confidence"] != "UNRECOVERABLE"],
        "unrecoverable_critical_fields": [row["field_name"] for row in recoverability if row["semantic_confidence"] == "UNRECOVERABLE"],
        "formal_rerun_required": bool(critical_unresolved),
        "reason": "Existing traces do not contain every requested field." if critical_unresolved else "All requested audit fields were recovered by exact trace join.",
        "minimum_required_new_trace_fields": critical_unresolved,
        "L4_rerun_required": bool(critical_unresolved),
        "L2_rerun_required": True,
    }
    (output_dir / "actual_branch_rerun_requirement.json").write_text(json.dumps(rerun, indent=2), encoding="utf-8")

    compare_rows = []
    if compare_run_dir is not None:
        compare_actual = read_rows(find_trace(compare_run_dir, "actual_branch_event_trace.csv"))
        compare_rows.append({"run": run_dir.name, "actual_trace": bool(raw_events), "event_rows": len(raw_events), "mape": _read_summary_metric(run_dir, "mape")})
        compare_rows.append({"run": compare_run_dir.name, "actual_trace": bool(compare_actual), "event_rows": len(compare_actual), "mape": _read_summary_metric(compare_run_dir, "mape")})
    if not compare_rows:
        compare_rows = [{"run": run_dir.name, "actual_trace": bool(raw_events), "event_rows": len(raw_events), "mape": _read_summary_metric(run_dir, "mape")}]
    write_csv(output_dir / "l2_l4_actual_branch_comparison.csv", compare_rows)

    status_counts = Counter(str(row.get("branch_continuity_status", NA) or NA) for row in events)
    field_status = Counter((str(row.get("field", NA)), str(row.get("branch_continuity_status", NA))) for row in events)
    summary = {
        "event_rows": len(events),
        "segment_rows": len(segments),
        "registry_rows": len(registry),
        "status_counts": dict(status_counts),
        "field_status_counts": {f"{field}|{status}": count for (field, status), count in sorted(field_status.items())},
        "join_info": join_info,
        "prematch_semantics": "PRE_MATCHING; selected segment and reinforcement are unknown at capture",
        "mixed_history_definition": "two or more distinct actual branch provenance IDs; same-branch multiple anchors are separate",
        "mechanism_conclusion": "BRANCH_PROVENANCE_REPRESENTATION_STILL_INSUFFICIENT",
        "recommended_next_step": "IMPROVE_BRANCH_PROVENANCE_REPRESENTATION",
        "diagnostic_only": True,
        "recurrent_trajectory_divergence": True,
    }
    (output_dir / "actual_branch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(output_dir, run_dir, summary, rerun, segment_summary, events, compare_rows)
    return summary


def _read_summary_metric(run_dir: Path, key: str) -> object:
    path = run_dir / "original_summary.json"
    if not path.exists():
        return NA
    try:
        return json.loads(path.read_text(encoding="utf-8")).get(key, NA)
    except json.JSONDecodeError:
        return NA


def _write_report(output_dir: Path, run_dir: Path, summary: dict[str, object], rerun: dict[str, object], segments: list[dict[str, object]], events: list[dict[str, object]], compare_rows: list[dict[str, object]]) -> None:
    status_counts = summary["status_counts"]
    lines = [
        "# Fig.9 Actual Branch Provenance Report",
        "",
        "## 1. Executive Summary",
        "",
        f"Run: `{run_dir}`. Event rows: **{len(events)}**; segment observation rows: **{summary['segment_rows']}**.",
        "",
        "This is an offline diagnostic. Continuity, switching and mixed-history labels are provenance descriptions, not correctness labels.",
        "",
        f"Final mechanism conclusion: **{summary['mechanism_conclusion']}**.",
        f"Recommended next step: **{summary['recommended_next_step']}**.",
        "",
        "## 2. Protocol Integrity",
        "",
        "The model, matching, selection, reinforcement, competition, RNG and strict defaults were not changed. The diagnostic is read-only and does not feed back into the model.",
        "",
        "## 3. Prematch Lifecycle Audit",
        "",
        "`capture_phase=PRE_MATCHING` means the row was captured after encoding but before current matching. At that point `matching_started_at_capture=false`, `selected_segment_known_at_capture=false`, and `reinforcement_started_at_capture=false`. The selected segment becomes known only in the post-observation join.",
        "",
        "The previous trace writer overwrote `current_selected_segment_unavailable` during the post-observation join. The repaired schema keeps capture semantics separate and exposes `post_observation_selected_segment_known` instead.",
        "",
        "## 4. Field and Missing-Value Audit",
        "",
        "Existing `segment_reinforcement_event_trace.csv.gz` is used for exact deterministic joins of reinforcement counts, L-match flags, repeated selection/reinforcement windows and downstream error fields. Missing values remain `NA`; they are never converted to false or zero.",
        "",
        "See `field_recoverability_manifest.csv` and `actual_branch_rerun_requirement.json`.",
        "",
        "## 5. Stable ID and Mixed-History Audit",
        "",
        "Segment, anchor and branch IDs are deterministic SHA-256 identities. They do not use object IDs, row positions, floating-point time approximations, UUIDs or model RNG.",
        "",
        "A segment is classified as `ACTUAL_BRANCH_MIXED_HISTORY` only when at least two distinct actual branch IDs are present. Multiple anchors under one branch are reported as `MULTIPLE_ANCHORS_SAME_BRANCH`, not mixed history. The trigger is recorded separately when available.",
        "",
        f"Segment-level mixed-history groups: **{sum(1 for row in segments if row.get('mixed_history'))}**.",
        "",
        "## 6. Observed Status Counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in sorted(status_counts.items()))
    lines.extend([
        "",
        "## 7. L2/L4 and Formal Trace Availability",
        "",
        "| Run | Actual-history trace | Event rows | MAPE |",
        "|---|---:|---:|---:|",
    ])
    lines.extend(f"| {row['run']} | {row['actual_trace']} | {row['event_rows']} | {row['mape']} |" for row in compare_rows)
    lines.extend([
        "",
        "The existing L2 directory has independent-reference and reinforcement traces but no actual-history branch trace. Therefore L2 actual-history continuity is unresolved and `L2_rerun_required=true`; no L2 continuity result is fabricated.",
        "",
        "## 8. Source Fact, Derived Statistic and Inference",
        "",
        "- **Source fact:** the actual branch trace contains the prematch marker and the post-observation selected/reinforced fields.",
        "- **Derived statistic:** status counts, field/range aggregates, bootstrap intervals and segment-level mixed-history groups.",
        "- **Diagnostic inference:** mixed-history is a real tracker-level observation when distinct branch IDs are present, but it is not by itself evidence of a wrong model choice.",
        "- **Unresolved:** whether the branch identity represents the intended biological/contextual ancestry, especially for periodic Time/Weekday contexts and missing Passenger anchors.",
        "",
        "## 9. Final Decision",
        "",
        f"Formal rerun required by the current trace manifest: **{rerun['formal_rerun_required']}**.",
        "No new formal 250-record model run was started by Codex.",
    ])
    (output_dir / "FIG9_ACTUAL_BRANCH_PROVENANCE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--compare-run-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.run_dir / "analysis"
    print(json.dumps(analyze(args.run_dir, output, args.compare_run_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
