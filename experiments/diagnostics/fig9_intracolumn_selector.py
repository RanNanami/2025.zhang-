"""Read-only serialization helpers for Fig.9 intracolumn local choice."""

from __future__ import annotations

import hashlib
import json
import csv
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

from experiments.diagnostics.fig9_candidate_score_trace import field_for_column
from experiments.diagnostics.fig9_ambiguity import stable_context_signature
from experiments.diagnostics.fig9_oracle_candidate import FieldColumnRanges
from seqmem.model import IntracolumnSelectionTrace


SELECTOR_VERSION = "fig9-intracolumn-local-choice-v1"
DIAGNOSTIC_MARKERS = {
    "diagnostic_only": True,
    "competition_is_local_choice": True,
    "intracolumn_selector_is_local_choice": True,
    "ground_truth_does_not_affect_prediction": True,
    "future_observation_does_not_affect_selection": True,
    "teacher_reference_does_not_affect_selection": True,
    "uses_compensation": False,
}
SELECTION_TRACE_FIELDS = [
    *DIAGNOSTIC_MARKERS,
    "selector_version",
    "policy",
    "input_index",
    "input_timestamp",
    "horizon_step",
    "column",
    "field",
    "group_candidate_count",
    "within_column_candidate_segment_count",
    "within_column_candidate_neuron_count",
    "local_top1_score",
    "local_top2_score",
    "local_score_margin",
    "normalized_score_margin",
    "candidate_score_entropy",
    "top_score_tie_count",
    "selected_candidate_original_index",
    "selected_neuron",
    "selected_segment_id",
    "selected_predicted_time",
    "selected_candidate_score",
    "selected_response_peak",
    "selected_contributor_count",
    "selected_positive_contributor_count",
    "selected_current_context_jaccard",
    "existing_policy_candidate_index",
    "existing_policy_neuron",
    "existing_policy_segment_id",
    "selected_differs_from_existing",
    "column_present_under_all_policies",
    "emitted_after_competition",
    "suppressed_intercolumn",
    "candidate_pool_fingerprint",
    "context_signature",
    "context_source_count",
    "selector_primary_key",
    "selector_secondary_keys",
    "tie_count",
    "tie_break_used",
    "post_hoc_labels_present",
]


def selection_trace_rows(
    *,
    trace: IntracolumnSelectionTrace,
    policy: str,
    input_index: int,
    input_timestamp: str,
    horizon_step: int,
    ranges: FieldColumnRanges,
    active_sources: Mapping[int, float] | None = None,
) -> list[dict[str, object]]:
    """Project the trace captured by the same predict_code call into CSV rows."""

    rows: list[dict[str, object]] = []
    for group in trace.groups:
        candidates = {
            candidate.original_index: candidate for candidate in group.candidates
        }
        selected = candidates[group.selected_candidate_original_index]
        existing = candidates[group.existing_policy_candidate_index]
        scores = sorted(
            (float(candidate.candidate_score) for candidate in group.candidates),
            reverse=True,
        )
        top1 = scores[0] if scores else math.nan
        top2 = scores[1] if len(scores) > 1 else math.nan
        margin = top1 - top2 if len(scores) > 1 else math.nan
        denominator = max(abs(top1), abs(top2)) if len(scores) > 1 else math.nan
        normalized_margin = (
            margin / (denominator + 1e-12) if len(scores) > 1 else math.nan
        )
        if scores:
            shifted = [math.exp(score - top1) for score in scores]
            total = sum(shifted)
            entropy = -sum(
                (value / total) * math.log(value / total)
                for value in shifted
                if value > 0.0
            )
        else:
            entropy = math.nan
        rows.append(
            {
                **DIAGNOSTIC_MARKERS,
                "selector_version": SELECTOR_VERSION,
                "policy": policy,
                "input_index": input_index,
                "input_timestamp": input_timestamp,
                "horizon_step": horizon_step,
                "column": group.column,
                "field": field_for_column(group.column, ranges),
                "group_candidate_count": len(group.candidates),
                "within_column_candidate_segment_count": len(group.candidates),
                "within_column_candidate_neuron_count": len(
                    {candidate.neuron_index for candidate in group.candidates}
                ),
                "local_top1_score": top1,
                "local_top2_score": top2,
                "local_score_margin": margin,
                "normalized_score_margin": normalized_margin,
                "candidate_score_entropy": entropy,
                "top_score_tie_count": sum(
                    score == top1 for score in scores
                ),
                "selected_candidate_original_index": selected.original_index,
                "selected_neuron": selected.neuron_index,
                "selected_segment_id": (
                    f"{selected.column}:{selected.neuron_index}:"
                    f"{selected.segment_index}"
                ),
                "selected_predicted_time": selected.predicted_time,
                "selected_candidate_score": selected.candidate_score,
                "selected_response_peak": (
                    selected.response_peak
                    if selected.response_peak is not None
                    else ""
                ),
                "selected_contributor_count": selected.contributor_count,
                "selected_positive_contributor_count": (
                    selected.positive_contributor_count
                ),
                "selected_current_context_jaccard": (
                    selected.current_context_jaccard
                ),
                "existing_policy_candidate_index": existing.original_index,
                "existing_policy_neuron": existing.neuron_index,
                "existing_policy_segment_id": (
                    f"{existing.column}:{existing.neuron_index}:"
                    f"{existing.segment_index}"
                ),
                "selected_differs_from_existing": (
                    selected.original_index != existing.original_index
                ),
                "column_present_under_all_policies": True,
                "emitted_after_competition": "",
                "suppressed_intercolumn": "",
                "candidate_pool_fingerprint": (
                    group.candidate_pool_fingerprint
                ),
                "context_signature": (
                    stable_context_signature(active_sources)
                    if active_sources is not None
                    else ""
                ),
                "context_source_count": (
                    len(active_sources) if active_sources is not None else ""
                ),
                "selector_primary_key": group.selector_primary_key,
                "selector_secondary_keys": "|".join(
                    group.selector_secondary_keys
                ),
                "tie_count": group.tie_count,
                "tie_break_used": group.tie_break_used,
                "post_hoc_labels_present": False,
            }
        )
    return rows


def summarize_selection_rows(
    rows: Iterable[dict[str, object]],
    *,
    policy: str,
) -> dict[str, object]:
    materialized = list(rows)
    multi = [
        row
        for row in materialized
        if int(row["group_candidate_count"]) > 1
    ]
    changed = [
        row for row in materialized if bool(row["selected_differs_from_existing"])
    ]
    ties = [row for row in materialized if bool(row["tie_break_used"])]
    missing_context = [
        row
        for row in materialized
        if row["selected_current_context_jaccard"] == ""
    ]
    fingerprint_payload = [
        (
            int(row["input_index"]),
            int(row["horizon_step"]),
            int(row["column"]),
            str(row["candidate_pool_fingerprint"]),
        )
        for row in materialized
    ]
    pool_fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    count = len(materialized)
    return {
        **DIAGNOSTIC_MARKERS,
        "policy": policy,
        "selector_version": SELECTOR_VERSION,
        "candidate_groups": count,
        "groups_with_more_than_one_candidate": len(multi),
        "groups_differing_from_existing": len(changed),
        "fraction_differing_from_existing": (
            len(changed) / count if count else 0.0
        ),
        "mean_candidates_per_group": (
            sum(int(row["group_candidate_count"]) for row in materialized)
            / count
            if count
            else 0.0
        ),
        "tie_rate": len(ties) / count if count else 0.0,
        "missing_context_metric_rate": (
            len(missing_context) / count if count else 0.0
        ),
        "raw_reachable_column_set_parity": all(
            bool(row["column_present_under_all_policies"])
            for row in materialized
        ),
        "candidate_pool_fingerprint": pool_fingerprint,
        "contributor_metric": "contributor_count",
        "uses_ground_truth": False,
        "uses_future_observation": False,
    }


def mark_competition_outcomes(
    rows: Iterable[dict[str, object]],
    *,
    emitted_columns: set[int],
) -> None:
    """Attach post-selection competition labels without changing selection."""

    for row in rows:
        emitted = int(row["column"]) in emitted_columns
        row["emitted_after_competition"] = emitted
        row["suppressed_intercolumn"] = not emitted


def write_selection_trace(
    path: Path,
    rows: Iterable[dict[str, object]],
) -> None:
    """Write a schema-complete trace even when no candidate group exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SELECTION_TRACE_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
