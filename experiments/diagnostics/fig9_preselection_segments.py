"""Read-only Fig.9 segment funnel rows built after prediction is fixed."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Iterable, Mapping

from experiments.diagnostics.fig9_branch_provenance import (
    BranchProvenanceRegistry,
    stable_source_fingerprint,
)
from experiments.diagnostics.fig9_candidate_score_trace import field_for_column
from experiments.diagnostics.fig9_competitive_inhibition import (
    CompetitionResult,
)
from experiments.diagnostics.fig9_oracle_candidate import FieldColumnRanges
from seqmem.encoding import SymbolCode
from seqmem.model import (
    PreselectionReplacementTrace,
    PreselectionSegmentTrace,
    PreselectionTrace,
    SequentialMemory,
)


PRESELECTION_LEVELS = {"summary", "crossing", "full"}
DIAGNOSTIC_MARKERS = {
    "diagnostic_only": True,
    "offline_analysis_only": True,
    "preselection_segment_diagnostic": True,
    "uses_ground_truth_for_analysis_only": True,
    "ground_truth_does_not_affect_prediction": True,
    "segment_trace_does_not_affect_selection": True,
}


def stable_selection_group_id(
    *,
    input_index: int,
    horizon_step: int,
    group_type: str,
    group_column: int,
    group_neuron: int | None,
    stable_time_key: float | None,
) -> str:
    """Return a process-independent ID from the real selection group keys."""

    payload = {
        "input_index": input_index,
        "horizon_step": horizon_step,
        "group_type": group_type,
        "group_column": group_column,
        "group_neuron": group_neuron,
        "stable_time_key": stable_time_key,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _segment_id(
    item: PreselectionSegmentTrace,
    registry: BranchProvenanceRegistry,
) -> tuple[str, object | None]:
    origin = registry.provenance_for(item.segment)
    return (
        origin.segment_provenance_id if origin is not None else "",
        origin,
    )


def _support_metrics(
    item: PreselectionSegmentTrace,
    *,
    predicted_sources: set[int],
    burst_sources: set[int],
    threshold_without_rest: float,
) -> tuple[float, float, str]:
    positive = [
        contribution
        for contribution in item.crossing_synapse_contributions
        if contribution.psp_contribution > 0.0
    ]
    predicted = sum(
        contribution.psp_contribution
        for contribution in positive
        if contribution.source_cell_id in predicted_sources
    )
    burst = sum(
        contribution.psp_contribution
        for contribution in positive
        if contribution.source_cell_id in burst_sources
    )
    unlabelled = sum(
        contribution.psp_contribution
        for contribution in positive
        if contribution.source_cell_id not in predicted_sources
        and contribution.source_cell_id not in burst_sources
    )
    if not predicted_sources and not burst_sources:
        support_class = "unavailable"
    elif predicted + unlabelled >= threshold_without_rest:
        support_class = "predicted-supported"
    elif burst >= threshold_without_rest:
        support_class = "burst-only"
    else:
        support_class = "burst-assisted"
    return predicted, burst, support_class


def _competition_maps(
    competition_result: CompetitionResult | None,
) -> tuple[set[int], dict[int, object]]:
    if competition_result is None:
        return set(), {}
    emitted = {
        id(decision.candidate.candidate.segment)
        for decision in competition_result.decisions
        if decision.emitted
    }
    decisions = {
        id(decision.candidate.candidate.segment): decision
        for decision in competition_result.decisions
    }
    return emitted, decisions


def _event_group_id(
    item: PreselectionSegmentTrace,
    *,
    input_index: int,
    horizon_step: int,
) -> str:
    return stable_selection_group_id(
        input_index=input_index,
        horizon_step=horizon_step,
        group_type="column_time_event",
        group_column=item.target_column,
        group_neuron=None,
        stable_time_key=item.event_time_key,
    )


def _column_group_id(
    item: PreselectionSegmentTrace,
    *,
    input_index: int,
    horizon_step: int,
) -> str:
    return stable_selection_group_id(
        input_index=input_index,
        horizon_step=horizon_step,
        group_type="column",
        group_column=item.target_column,
        group_neuron=None,
        stable_time_key=None,
    )


def _segment_rows(
    *,
    model: SequentialMemory,
    registry: BranchProvenanceRegistry,
    trace: PreselectionTrace,
    stream_label: str,
    policy: str,
    input_index: int,
    target_timestamp: str,
    horizon_step: int,
    target_code: SymbolCode,
    ranges: FieldColumnRanges,
    active_sources: Mapping[int, float],
    competition_result: CompetitionResult | None,
    level: str,
) -> list[dict[str, object]]:
    target_columns = set(target_code.columns)
    active_source_ids = set(active_sources)
    emitted_segments, _decisions = _competition_maps(competition_result)
    rows: list[dict[str, object]] = []
    for item in trace.segments:
        if level == "crossing" and not item.crossed_threshold:
            continue
        if level == "summary":
            continue
        segment_id, origin = _segment_id(item, registry)
        segment_sources = set(item.segment.synapses)
        creation_sources = (
            set(origin.creation_source_cell_ids) if origin is not None else set()
        )
        current_intersection = len(creation_sources & segment_sources)
        predicted_psp, burst_psp, support_class = _support_metrics(
            item,
            predicted_sources=registry.current_predicted_sources,
            burst_sources=registry.current_burst_sources,
            threshold_without_rest=(
                model.params.dendrite_threshold - model._dynamics.v_rest
            ),
        )
        winner_id = ""
        if item.winner_segment_identity is not None:
            winner = next(
                (
                    candidate
                    for candidate in trace.segments
                    if candidate.segment_identity
                    == item.winner_segment_identity
                ),
                None,
            )
            if winner is not None:
                winner_id = _segment_id(winner, registry)[0]
        response_peak = item.response_peak
        rows.append(
            {
                **DIAGNOSTIC_MARKERS,
                "stream_label": stream_label,
                "policy": policy,
                "input_index": input_index,
                "target_timestamp": target_timestamp,
                "horizon_step": horizon_step,
                "trace_level": level,
                "segment_provenance_id": segment_id,
                "selection_group_id": (
                    _event_group_id(
                        item,
                        input_index=input_index,
                        horizon_step=horizon_step,
                    )
                    if item.event_time_key is not None
                    else ""
                ),
                "column_selection_group_id": _column_group_id(
                    item,
                    input_index=input_index,
                    horizon_step=horizon_step,
                ),
                "field": field_for_column(item.target_column, ranges),
                "target_column": item.target_column,
                "target_neuron": item.target_neuron,
                "response_available": item.response_computed,
                "response_peak": (
                    response_peak if response_peak is not None else ""
                ),
                "response_at_selected_time": (
                    item.response_at_selected_time
                    if item.response_at_selected_time is not None
                    else ""
                ),
                "threshold": item.dendritic_threshold,
                "threshold_margin": (
                    response_peak - item.dendritic_threshold
                    if response_peak is not None
                    else ""
                ),
                "first_crossing_time": (
                    item.first_threshold_crossing_time
                    if item.first_threshold_crossing_time is not None
                    else ""
                ),
                "predicted_time": (
                    item.predicted_soma_firing_time
                    - model.params.cycle_period
                    if item.predicted_soma_firing_time is not None
                    else ""
                ),
                "candidate_score_value": (
                    item.candidate_score_value
                    if item.candidate_score_value is not None
                    else ""
                ),
                "contributor_count": item.contributor_count,
                "positive_contributor_count": (
                    item.positive_contributor_count
                ),
                "synapse_count": len(item.segment.synapses),
                "segment_weight_sum": sum(
                    synapse.weight
                    for synapse in item.segment.synapses.values()
                ),
                "segment_age": max(
                    (
                        synapse.age
                        for synapse in item.segment.synapses.values()
                    ),
                    default=0,
                ),
                "crossed_threshold": item.crossed_threshold,
                "became_event_winner": item.became_event_winner,
                "became_neuron_winner": "",
                "became_column_winner": item.became_column_winner,
                "became_prediction_candidate": (
                    item.became_prediction_candidate
                ),
                "appeared_in_raw_code": item.became_prediction_candidate,
                "emitted_after_competition": (
                    item.segment_identity in emitted_segments
                ),
                "furthest_stage_reached": item.furthest_stage_reached,
                "elimination_stage": item.elimination_stage,
                "elimination_reason": item.elimination_reason,
                "winner_segment_id": winner_id,
                "rank_within_actual_group": (
                    item.rank_within_event_group
                    if item.rank_within_event_group is not None
                    else ""
                ),
                "rank_within_column_group": (
                    item.rank_within_column_group
                    if item.rank_within_column_group is not None
                    else ""
                ),
                "winner_score_gap": (
                    item.winner_score_gap
                    if item.winner_score_gap is not None
                    else ""
                ),
                "winner_time_gap": (
                    item.winner_time_gap
                    if item.winner_time_gap is not None
                    else ""
                ),
                "tie_break_used": item.tie_break_used,
                "creation_transition_index": (
                    origin.creation_transition_index
                    if origin is not None
                    else ""
                ),
                "creation_source_fingerprint": (
                    origin.creation_source_fingerprint
                    if origin is not None
                    else ""
                ),
                "current_source_fingerprint": (
                    stable_source_fingerprint(segment_sources)
                ),
                "creation_current_source_jaccard": (
                    _jaccard(creation_sources, segment_sources)
                    if origin is not None
                    else ""
                ),
                "creation_source_retention": (
                    _ratio(current_intersection, len(creation_sources))
                    if origin is not None
                    else ""
                ),
                "current_context_overlap": len(
                    segment_sources & active_source_ids
                ),
                "current_context_jaccard": _jaccard(
                    segment_sources,
                    active_source_ids,
                ),
                "historical_winner_match": (
                    creation_sources == active_source_ids
                    if origin is not None
                    else ""
                ),
                "predicted_supported_contribution": predicted_psp,
                "burst_supported_contribution": burst_psp,
                "source_support_class": support_class,
                "provenance_available": origin is not None,
                "provenance_missing_reason": (
                    "" if origin is not None else "segment_not_seen_at_creation"
                ),
                "is_target_field": (
                    field_for_column(item.target_column, ranges)
                    in {"weekday", "time", "passenger"}
                ),
                "is_target_column": item.target_column in target_columns,
                "is_target_neuron_when_definable": "",
                "target_label_definition": (
                    "post-prediction target SSTD column; target neuron "
                    "is not defined by the encoder"
                ),
            }
        )
    return rows


def _group_rows(
    *,
    trace: PreselectionTrace,
    registry: BranchProvenanceRegistry,
    input_index: int,
    horizon_step: int,
    ranges: FieldColumnRanges,
    target_columns: set[int],
) -> list[dict[str, object]]:
    groups: dict[
        tuple[str, int, float | None],
        list[PreselectionSegmentTrace],
    ] = defaultdict(list)
    for item in trace.segments:
        if item.valid_firing_time:
            groups[
                ("column_time_event", item.target_column, item.event_time_key)
            ].append(item)
        if item.became_event_winner:
            groups[("column", item.target_column, None)].append(item)
    rows: list[dict[str, object]] = []
    for (group_type, column, time_key), members in groups.items():
        if group_type == "column_time_event":
            winner = next(
                (item for item in members if item.became_event_winner),
                None,
            )
            ordered = sorted(
                members,
                key=lambda item: (
                    -(
                        item.candidate_score_value
                        if item.candidate_score_value is not None
                        else float("-inf")
                    ),
                    item.inspection_order,
                ),
            )
            selection_reason = "highest_score_strict_greater"
        else:
            winner = next(
                (item for item in members if item.became_column_winner),
                None,
            )
            ordered = sorted(
                members,
                key=lambda item: (
                    (
                        item.predicted_soma_firing_time
                        if item.predicted_soma_firing_time is not None
                        else float("inf")
                    ),
                    item.inspection_order,
                ),
            )
            selection_reason = "earliest_predicted_time_strict_less"
        target_members = [
            item for item in members if item.target_column in target_columns
        ]
        best_target = next(
            (item for item in ordered if item in target_members),
            None,
        )
        winner_id = (
            _segment_id(winner, registry)[0] if winner is not None else ""
        )
        best_target_id = (
            _segment_id(best_target, registry)[0]
            if best_target is not None
            else ""
        )
        winning_value = (
            winner.candidate_score_value
            if group_type == "column_time_event" and winner is not None
            else (
                winner.predicted_soma_firing_time
                if winner is not None
                else None
            )
        )
        tie_count = sum(
            (
                item.candidate_score_value
                if group_type == "column_time_event"
                else item.predicted_soma_firing_time
            )
            == winning_value
            for item in members
        )
        rows.append(
            {
                **DIAGNOSTIC_MARKERS,
                "selection_group_id": stable_selection_group_id(
                    input_index=input_index,
                    horizon_step=horizon_step,
                    group_type=group_type,
                    group_column=column,
                    group_neuron=None,
                    stable_time_key=time_key,
                ),
                "group_type": group_type,
                "input_index": input_index,
                "horizon_step": horizon_step,
                "field": field_for_column(column, ranges),
                "group_column": column,
                "group_neuron": "",
                "stable_time_key": (
                    time_key if time_key is not None else ""
                ),
                "group_fingerprint_inputs": json.dumps(
                    {
                        "input_index": input_index,
                        "horizon_step": horizon_step,
                        "group_type": group_type,
                        "group_column": column,
                        "group_neuron": None,
                        "stable_time_key": time_key,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "segment_count": len(members),
                "threshold_crossing_count": sum(
                    item.crossed_threshold for item in members
                ),
                "target_segment_count": len(target_members),
                "false_segment_count": len(members) - len(target_members),
                "winner_segment_id": winner_id,
                "winner_is_target_column": (
                    winner.target_column in target_columns
                    if winner is not None
                    else False
                ),
                "winner_response_peak": (
                    winner.response_peak if winner is not None else ""
                ),
                "winner_score": (
                    winner.candidate_score_value
                    if winner is not None
                    else ""
                ),
                "winner_crossing_time": (
                    winner.first_threshold_crossing_time
                    if winner is not None
                    else ""
                ),
                "best_target_segment_id": best_target_id,
                "best_target_score": (
                    best_target.candidate_score_value
                    if best_target is not None
                    else ""
                ),
                "best_target_peak": (
                    best_target.response_peak
                    if best_target is not None
                    else ""
                ),
                "best_target_crossing_time": (
                    best_target.first_threshold_crossing_time
                    if best_target is not None
                    else ""
                ),
                "target_best_rank_by_existing_rule": (
                    ordered.index(best_target) + 1
                    if best_target is not None
                    else ""
                ),
                "false_outranks_all_targets": (
                    winner is not None
                    and winner.target_column not in target_columns
                    and bool(target_members)
                ),
                "target_crossed_but_lost": (
                    bool(target_members)
                    and (
                        winner is None
                        or winner.target_column not in target_columns
                    )
                ),
                "mixed_target_false_group": (
                    bool(target_members)
                    and len(target_members) < len(members)
                ),
                "selection_reason": selection_reason,
                "tie_count": tie_count,
            }
        )
    return rows


def _replacement_rows(
    *,
    replacements: Iterable[PreselectionReplacementTrace],
    trace: PreselectionTrace,
    registry: BranchProvenanceRegistry,
    input_index: int,
    horizon_step: int,
    target_columns: set[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in replacements:
        previous = next(
            candidate
            for candidate in trace.segments
            if candidate.segment_identity == item.previous_segment_identity
        )
        replacement = next(
            candidate
            for candidate in trace.segments
            if candidate.segment_identity == item.replacement_segment_identity
        )
        rows.append(
            {
                **DIAGNOSTIC_MARKERS,
                "selection_group_id": stable_selection_group_id(
                    input_index=input_index,
                    horizon_step=horizon_step,
                    group_type=item.group_type,
                    group_column=item.target_column,
                    group_neuron=item.target_neuron,
                    stable_time_key=item.stable_time_key,
                ),
                "previous_winner_id": _segment_id(previous, registry)[0],
                "replacement_winner_id": _segment_id(
                    replacement,
                    registry,
                )[0],
                "comparison_field": item.comparison_field,
                "previous_value": item.previous_value,
                "replacement_value": item.replacement_value,
                "previous_tie_break_values": json.dumps(
                    item.previous_tie_break_value
                ),
                "replacement_tie_break_values": json.dumps(
                    item.replacement_tie_break_value
                ),
                "previous_is_target": (
                    previous.target_column in target_columns
                ),
                "replacement_is_target": (
                    replacement.target_column in target_columns
                ),
                "replacement_stage": item.replacement_stage,
            }
        )
    return rows


def build_preselection_rows(
    *,
    model: SequentialMemory,
    registry: BranchProvenanceRegistry,
    trace: PreselectionTrace,
    stream_label: str,
    policy: str,
    input_index: int,
    input_timestamp: str,
    target_timestamp: str,
    horizon_step: int,
    target_code: SymbolCode,
    ranges: FieldColumnRanges,
    active_sources: Mapping[int, float],
    competition_result: CompetitionResult | None,
    raw_code: SymbolCode | None,
    runtime_seconds: float,
    level: str,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Label one already-completed prediction without affecting selection."""

    if level not in PRESELECTION_LEVELS:
        raise ValueError(f"unsupported preselection level: {level}")
    target_columns = set(target_code.columns)
    emitted_segments, _decisions = _competition_maps(competition_result)
    crossed = [item for item in trace.segments if item.crossed_threshold]
    candidates = [
        item for item in trace.segments if item.became_prediction_candidate
    ]
    emitted = [
        item
        for item in candidates
        if item.segment_identity in emitted_segments
    ]
    target_field_segment_counts = {
        field: sum(
            field_for_column(item.target_column, ranges) == field
            for item in trace.segments
        )
        for field in ("weekday", "time", "passenger", "unknown")
    }
    target_column_outcomes: list[dict[str, object]] = []
    for target_column in sorted(target_columns):
        members = [
            item
            for item in trace.segments
            if item.target_column == target_column
        ]
        if not members:
            outcome = "TARGET_NO_SEGMENT_INSPECTED"
        elif not any(item.response_computed for item in members):
            outcome = "TARGET_SEGMENT_BELOW_THRESHOLD"
        elif not any(
            item.response_peak is not None
            and item.response_peak > model._dynamics.v_rest
            for item in members
        ):
            outcome = "TARGET_SEGMENT_RESPONSE_NONPOSITIVE"
        elif not any(item.crossed_threshold for item in members):
            outcome = "TARGET_SEGMENT_BELOW_THRESHOLD"
        elif not any(item.valid_firing_time for item in members):
            outcome = (
                "TARGET_SEGMENT_CROSSED_BUT_NO_VALID_FIRING_TIME"
            )
        elif not any(item.became_prediction_candidate for item in members):
            outcome = (
                "TARGET_SEGMENT_CROSSED_BUT_LOST_INTERNAL_SELECTION"
            )
        elif not any(
            item.segment_identity in emitted_segments for item in members
        ):
            outcome = (
                "TARGET_SEGMENT_BECAME_CANDIDATE_BUT_"
                "SUPPRESSED_INTERCOLUMN"
            )
        else:
            outcome = "TARGET_SEGMENT_EMITTED"
        target_column_outcomes.append(
            {
                "column": target_column,
                "field": field_for_column(target_column, ranges),
                "outcome": outcome,
            }
        )
    funnel = {
        **DIAGNOSTIC_MARKERS,
        "stream_label": stream_label,
        "policy": policy,
        "input_index": input_index,
        "input_timestamp": input_timestamp,
        "target_timestamp": target_timestamp,
        "horizon_step": horizon_step,
        "inspected_segment_count": len(trace.segments),
        "response_computed_count": sum(
            item.response_computed for item in trace.segments
        ),
        "positive_response_count": sum(
            item.response_peak is not None
            and item.response_peak > model._dynamics.v_rest
            for item in trace.segments
        ),
        "threshold_crossing_count": len(crossed),
        "event_winner_count": sum(
            item.became_event_winner for item in trace.segments
        ),
        "neuron_winner_count": "",
        "column_winner_count": len(candidates),
        "saved_candidate_count": len(candidates),
        "raw_event_count": len(raw_code.events) if raw_code is not None else 0,
        "emitted_candidate_count": len(emitted),
        "unique_threshold_columns": len(
            {item.target_column for item in crossed}
        ),
        "unique_threshold_neurons": len(
            {(item.target_column, item.target_neuron) for item in crossed}
        ),
        "unique_candidate_columns": len(
            {item.target_column for item in candidates}
        ),
        "target_field_segment_counts": json.dumps(
            target_field_segment_counts,
            sort_keys=True,
        ),
        "target_column_outcomes": json.dumps(
            target_column_outcomes,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "target_column_crossing_count": sum(
            item.target_column in target_columns for item in crossed
        ),
        "target_column_candidate_count": sum(
            item.target_column in target_columns for item in candidates
        ),
        "target_column_emitted_count": sum(
            item.target_column in target_columns for item in emitted
        ),
        "false_column_crossing_count": sum(
            item.target_column not in target_columns for item in crossed
        ),
        "false_column_candidate_count": sum(
            item.target_column not in target_columns for item in candidates
        ),
        "false_column_emitted_count": sum(
            item.target_column not in target_columns for item in emitted
        ),
        "unknown_elimination_count": sum(
            not item.became_prediction_candidate
            and not item.elimination_reason
            for item in trace.segments
        ),
        "runtime_seconds": runtime_seconds,
    }
    segment_rows = _segment_rows(
        model=model,
        registry=registry,
        trace=trace,
        stream_label=stream_label,
        policy=policy,
        input_index=input_index,
        target_timestamp=target_timestamp,
        horizon_step=horizon_step,
        target_code=target_code,
        ranges=ranges,
        active_sources=active_sources,
        competition_result=competition_result,
        level=level,
    )
    group_rows = (
        _group_rows(
            trace=trace,
            registry=registry,
            input_index=input_index,
            horizon_step=horizon_step,
            ranges=ranges,
            target_columns=target_columns,
        )
        if level != "summary"
        else []
    )
    replacement_rows = (
        _replacement_rows(
            replacements=trace.replacements,
            trace=trace,
            registry=registry,
            input_index=input_index,
            horizon_step=horizon_step,
            target_columns=target_columns,
        )
        if level != "summary"
        else []
    )
    return funnel, segment_rows, group_rows, replacement_rows
