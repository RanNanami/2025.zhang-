"""Read-only Scenario-3 match-overlap decomposition for Fig.9.

The capture functions copy state immediately before the normal proximal
observation.  They never invoke prediction, matching, observation, or RNG.
Counterfactual contexts are therefore measurements of the fixed state, not
alternative model trajectories.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

from experiments.diagnostics.fig9_branch_provenance import (
    BranchProvenanceRegistry,
)
from experiments.diagnostics.fig9_candidate_score_trace import field_for_column
from experiments.diagnostics.fig9_oracle_candidate import FieldColumnRanges
from experiments.fig9 import TaxiRecord, record_values
from seqmem.encoding import SymbolCode
from seqmem.model import ObservationTrace, Segment, SequentialMemory


MATCH_OVERLAP_LEVELS = {"summary", "segment", "source"}
DIAGNOSTIC_MARKERS = {
    "diagnostic_only": True,
    "offline_analysis_only": True,
    "match_overlap_decomposition": True,
    "uses_ground_truth_for_analysis_only": True,
    "ground_truth_does_not_affect_matching": True,
    "ground_truth_does_not_affect_prediction": True,
    "counterfactual_L_match_does_not_affect_model": True,
    "context_variants_do_not_affect_model": True,
    "uses_compensation": False,
}


@dataclass(frozen=True)
class MatchOverlapCapture:
    """Immutable rows copied before one real observation."""

    column_rows: tuple[dict[str, object], ...]
    segment_rows: tuple[dict[str, object], ...]
    source_rows: tuple[dict[str, object], ...]


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _median(values: Sequence[int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _segment_id(
    registry: BranchProvenanceRegistry,
    segment: Segment,
) -> str:
    origin = registry.provenance_for(segment)
    if origin is not None:
        return origin.segment_provenance_id
    if segment.diagnostic_id is not None:
        return f"diagnostic:{segment.diagnostic_id}"
    return ""


def _creation_sources(
    registry: BranchProvenanceRegistry,
    segment: Segment,
) -> tuple[set[int], int | str]:
    origin = registry.provenance_for(segment)
    if origin is not None:
        return (
            set(origin.creation_source_cell_ids),
            origin.creation_transition_index,
        )
    if segment.creation_source_cell_ids:
        return (
            set(segment.creation_source_cell_ids),
            segment.creation_transition_index
            if segment.creation_transition_index is not None
            else "",
        )
    return set(segment.synapses), ""


def _cell_column(model: SequentialMemory, cell_id: int) -> int:
    neurons_per_column = len(model.columns[0].neurons)
    return cell_id // neurons_per_column


def _cell_neuron(model: SequentialMemory, cell_id: int) -> int:
    neurons_per_column = len(model.columns[0].neurons)
    return cell_id % neurons_per_column


def _timed_members(
    segment: Segment,
    context: Mapping[int, float],
    *,
    dendritic_time: float,
    tolerance: float,
) -> set[int]:
    """Mirror Segment.timed_overlap without calling the matching selector."""

    return {
        source
        for source, source_time in context.items()
        if (synapse := segment.synapses.get(source)) is not None
        and abs(source_time + synapse.delay - dendritic_time) <= tolerance
    }


def _source_loss_reason(
    *,
    source: int,
    contributes: bool,
    segment_active: bool,
    active: set[int],
    active_columns: set[int],
    model: SequentialMemory,
    timing_matched: bool,
) -> str:
    if contributes:
        return ""
    if not segment_active:
        return "SOURCE_SEGMENT_DELETED"
    source_column = _cell_column(model, source)
    if source in active and not timing_matched:
        return "TIMING_OR_ELIGIBILITY_EXCLUDED"
    if source_column in active_columns:
        return "SOURCE_COLUMN_ACTIVE_WRONG_NEURON"
    if source not in active:
        return "SOURCE_COLUMN_NOT_ACTIVE"
    return "UNKNOWN_SOURCE_LOSS"


def capture_match_overlap(
    *,
    model: SequentialMemory,
    code: SymbolCode,
    registry: BranchProvenanceRegistry,
    stream_label: str,
    actual_record_index: int,
    actual_record: TaxiRecord,
    ranges: FieldColumnRanges,
    level: str,
) -> MatchOverlapCapture:
    """Copy match inputs before the one real ``observe_code`` call."""

    if level not in MATCH_OVERLAP_LEVELS:
        raise ValueError("match overlap level must be summary, segment, or source")

    # This exactly mirrors SequentialMemory._active_sources without calling a
    # selector or changing the ordering of the real matching traversal.
    winners = dict(model.previous_winners)
    active = (
        dict(model.previous_active_cells or model.previous_winners)
        if model.params.burst_context
        else dict(model.previous_winners)
    )
    # The registry preserves the exact split recovered after the previous
    # observation even when model-level branch capture is disabled.
    predicted = set(registry.current_predicted_sources)
    burst = set(registry.current_burst_sources)
    winner_ids = set(winners)
    active_ids = set(active)
    predicted_plus_winner = predicted | winner_ids
    burst_only = burst - predicted - winner_ids
    without_burst_only = active_ids - burst_only
    active_columns = {_cell_column(model, source) for source in active_ids}
    values = record_values(actual_record)
    timestamp = actual_record.timestamp.isoformat(sep=" ")

    column_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []

    for event in code.events:
        column = model.columns[event.column]
        field = field_for_column(event.column, ranges)
        field_index = {"weekday": 0, "time": 1, "passenger": 2}[field]
        dendritic_time = (
            model.params.cycle_period / 2.0 + event.time
        )
        segments = [
            (neuron_index, segment)
            for neuron_index, neuron in enumerate(column.neurons)
            for segment in neuron.segments
            if segment.active
        ]
        overlaps: list[int] = []
        segment_payloads: list[dict[str, object]] = []

        for neuron_index, segment in segments:
            source_ids = set(segment.synapses)
            creation_sources, creation_transition = _creation_sources(
                registry,
                segment,
            )
            actual_members = _timed_members(
                segment,
                active,
                dendritic_time=dendritic_time,
                tolerance=model.params.timing_tolerance,
            )
            overlap = len(actual_members)
            overlaps.append(overlap)
            source_fields = Counter(
                field_for_column(_cell_column(model, source), ranges)
                for source in source_ids
            )
            active_source_fields = Counter(
                field_for_column(_cell_column(model, source), ranges)
                for source in actual_members
            )
            same_field = {
                source: source_time
                for source, source_time in active.items()
                if field_for_column(_cell_column(model, source), ranges) == field
            }
            cross_field = {
                source: source_time
                for source, source_time in active.items()
                if field_for_column(_cell_column(model, source), ranges) != field
            }
            variants = {
                "overlap_all_cell_current": overlap,
                "overlap_winner_only": len(
                    _timed_members(
                        segment,
                        winners,
                        dendritic_time=dendritic_time,
                        tolerance=model.params.timing_tolerance,
                    )
                ),
                "overlap_predicted_only": len(
                    _timed_members(
                        segment,
                        {key: active[key] for key in predicted if key in active},
                        dendritic_time=dendritic_time,
                        tolerance=model.params.timing_tolerance,
                    )
                ),
                "overlap_burst_only": len(
                    _timed_members(
                        segment,
                        {key: active[key] for key in burst if key in active},
                        dendritic_time=dendritic_time,
                        tolerance=model.params.timing_tolerance,
                    )
                ),
                "overlap_predicted_plus_winner": len(
                    _timed_members(
                        segment,
                        {
                            key: (active[key] if key in active else winners[key])
                            for key in predicted_plus_winner
                            if key in active or key in winners
                        },
                        dendritic_time=dendritic_time,
                        tolerance=model.params.timing_tolerance,
                    )
                ),
                "overlap_without_burst_only": len(
                    _timed_members(
                        segment,
                        {
                            key: active[key]
                            for key in without_burst_only
                            if key in active
                        },
                        dendritic_time=dendritic_time,
                        tolerance=model.params.timing_tolerance,
                    )
                ),
                "overlap_same_field_only": len(
                    _timed_members(
                        segment,
                        same_field,
                        dendritic_time=dendritic_time,
                        tolerance=model.params.timing_tolerance,
                    )
                ),
                "overlap_cross_field_only": len(
                    _timed_members(
                        segment,
                        cross_field,
                        dendritic_time=dendritic_time,
                        tolerance=model.params.timing_tolerance,
                    )
                ),
            }
            segment_id = _segment_id(registry, segment)
            segment_row: dict[str, object] = {
                **DIAGNOSTIC_MARKERS,
                "stream_label": stream_label,
                "actual_record_index": actual_record_index,
                "timestamp": timestamp,
                "field": field,
                "encoded_column": event.column,
                "encoded_value": values[field_index],
                "observe_scenario": "",
                "scenario_assignment_reason": "",
                "segment_provenance_id": segment_id,
                "segment_target_neuron": neuron_index,
                "segment_creation_transition_index": creation_transition,
                "segment_age": max(
                    (synapse.age for synapse in segment.synapses.values()),
                    default=0,
                ),
                "segment_synapse_count": len(source_ids),
                "segment_weight_sum": sum(
                    synapse.weight for synapse in segment.synapses.values()
                ),
                "current_source_active_count": len(source_ids & active_ids),
                "current_source_retention_ratio": _ratio(
                    len(source_ids & active_ids),
                    len(source_ids),
                ),
                "actual_overlap": overlap,
                "meets_L_match": overlap >= model.params.l_match,
                "eligible": overlap >= model.params.l_match,
                "eligibility_failure_reason": (
                    "" if overlap >= model.params.l_match else "BELOW_L_MATCH"
                ),
                "selected_as_best_matching": False,
                "reinforced": False,
                "selected_for_scenario": False,
                "selected_winner_neuron": "",
                # A current-state trace cannot infer whether this segment is
                # deleted later in the run.
                "segment_deleted_before_end": "",
                "predicted_time_available": bool(
                    model.last_prediction_candidates.get(event.column)
                ),
                "predicted_time_valid": any(
                    abs(candidate.time - event.time)
                    <= model.params.timing_tolerance
                    for candidate in model.last_prediction_candidates.get(
                        event.column,
                        (),
                    )
                ),
                "creation_source_count": len(creation_sources),
                "creation_source_currently_active": len(
                    creation_sources & active_ids
                ),
                "creation_source_currently_winner": len(
                    creation_sources & winner_ids
                ),
                "creation_source_currently_predicted": len(
                    creation_sources & predicted
                ),
                "creation_source_currently_burst": len(
                    creation_sources & burst
                ),
                "creation_source_missing": len(creation_sources - active_ids),
                "creation_current_jaccard": _jaccard(
                    creation_sources,
                    active_ids,
                ),
                "source_retention_ratio": _ratio(
                    len(creation_sources & active_ids),
                    len(creation_sources),
                ),
                "source_weekday_count": source_fields["weekday"],
                "source_time_count": source_fields["time"],
                "source_passenger_count": source_fields["passenger"],
                "active_source_weekday_count": active_source_fields["weekday"],
                "active_source_time_count": active_source_fields["time"],
                "active_source_passenger_count": active_source_fields["passenger"],
                **variants,
                "passes_L1": overlap >= 1,
                "passes_L2": overlap >= 2,
                "passes_L3": overlap >= 3,
                "passes_L4": overlap >= 4,
                "_segment_object_id": id(segment),
            }
            segment_payloads.append(segment_row)

            if level == "source":
                for source, synapse in segment.synapses.items():
                    source_column = _cell_column(model, source)
                    source_time = active.get(source)
                    timing_matched = bool(
                        source_time is not None
                        and abs(
                            source_time + synapse.delay - dendritic_time
                        )
                        <= model.params.timing_tolerance
                    )
                    contributes = source in actual_members
                    source_rows.append(
                        {
                            **DIAGNOSTIC_MARKERS,
                            "stream_label": stream_label,
                            "actual_record_index": actual_record_index,
                            "timestamp": timestamp,
                            "field": field,
                            "encoded_column": event.column,
                            "segment_provenance_id": segment_id,
                            "source_column": source_column,
                            "source_neuron": _cell_neuron(model, source),
                            "source_field": field_for_column(
                                source_column,
                                ranges,
                            ),
                            "source_synaptic_weight": synapse.weight,
                            "source_synaptic_delay": synapse.delay,
                            "source_in_creation_context": (
                                source in creation_sources
                            ),
                            "source_in_current_previous_winners": (
                                source in winner_ids
                            ),
                            "source_currently_active": source in active_ids,
                            "source_currently_predicted": source in predicted,
                            "source_currently_burst": source in burst,
                            "source_currently_winner": source in winner_ids,
                            "source_contributes_to_actual_overlap": contributes,
                            "source_missing_reason": _source_loss_reason(
                                source=source,
                                contributes=contributes,
                                segment_active=segment.active,
                                active=active_ids,
                                active_columns=active_columns,
                                model=model,
                                timing_matched=timing_matched,
                            ),
                            "source_cell_still_exists": (
                                0 <= source_column < len(model.columns)
                            ),
                            "source_segment_still_exists": segment.active,
                            "_segment_object_id": id(segment),
                        }
                    )

        best_overlap = max(overlaps, default=0)
        best_segments = [
            row
            for row in segment_payloads
            if int(row["actual_overlap"]) == best_overlap
        ]
        best_neurons = {
            int(row["segment_target_neuron"]) for row in best_segments
        }
        passing = [
            row
            for row in segment_payloads
            if bool(row["meets_L_match"])
        ]
        passing_neurons = {
            int(row["segment_target_neuron"]) for row in passing
        }
        best = best_segments[0] if best_segments else None
        synapse_counts = [len(segment.synapses) for _, segment in segments]
        ages = [
            max((synapse.age for synapse in segment.synapses.values()), default=0)
            for _, segment in segments
        ]
        weight_sums = [
            sum(synapse.weight for synapse in segment.synapses.values())
            for _, segment in segments
        ]
        column_rows.append(
            {
                **DIAGNOSTIC_MARKERS,
                "stream_label": stream_label,
                "actual_record_index": actual_record_index,
                "timestamp": timestamp,
                "field": field,
                "encoded_column": event.column,
                "encoded_value": values[field_index],
                "observe_scenario": "",
                "scenario_assignment_reason": "",
                "previous_winner_count": len(winner_ids),
                "previous_winner_unique_column_count": len(
                    {_cell_column(model, source) for source in winner_ids}
                ),
                "previous_winner_unique_neuron_count": len(
                    {_cell_neuron(model, source) for source in winner_ids}
                ),
                "predicted_cell_count": len(predicted),
                "burst_cell_count": len(burst),
                "active_cell_count": len(active_ids),
                "winner_only_context_count": len(winner_ids),
                "predicted_only_context_count": len(predicted),
                "burst_only_context_count": len(burst),
                "predicted_and_burst_context_count": len(predicted & burst),
                "neuron_count": len(column.neurons),
                "neurons_with_segments": sum(
                    bool(neuron.segments) for neuron in column.neurons
                ),
                "existing_segment_count": len(segments),
                "existing_synapse_count": sum(synapse_counts),
                "segment_age_mean": (
                    sum(ages) / len(ages) if ages else 0.0
                ),
                "segment_age_max": max(ages, default=0),
                "segment_weight_sum_mean": (
                    sum(weight_sums) / len(weight_sums)
                    if weight_sums
                    else 0.0
                ),
                "segment_overlap_max": best_overlap,
                "segment_overlap_mean": (
                    sum(overlaps) / len(overlaps) if overlaps else 0.0
                ),
                "segment_overlap_median": _median(overlaps),
                "segments_overlap_0": overlaps.count(0),
                "segments_overlap_1": overlaps.count(1),
                "segments_overlap_2": overlaps.count(2),
                "segments_overlap_3": overlaps.count(3),
                "segments_overlap_4_or_more": sum(
                    value >= 4 for value in overlaps
                ),
                "segments_meeting_L_match": len(passing),
                "eligible_matching_segments": len(passing),
                "best_matching_segment_id": (
                    best["segment_provenance_id"] if best else ""
                ),
                "best_matching_neuron": (
                    best["segment_target_neuron"] if best else ""
                ),
                "best_matching_overlap": best_overlap,
                "L_match": model.params.l_match,
                "gap_to_L_match": max(
                    0,
                    model.params.l_match - best_overlap,
                ),
                "matching_ambiguity_count": len(passing),
                "multiple_segments_meet_L_match": len(passing) > 1,
                "multiple_neurons_meet_L_match": len(passing_neurons) > 1,
                "best_segment_source_count": (
                    best["segment_synapse_count"] if best else 0
                ),
                "best_segment_source_active_count": (
                    best["current_source_active_count"] if best else 0
                ),
                "best_segment_source_retention": (
                    best["current_source_retention_ratio"] if best else 0.0
                ),
                "best_segment_creation_current_jaccard": (
                    best["creation_current_jaccard"] if best else 0.0
                ),
                "best_segment_current_context_jaccard": (
                    _jaccard(
                        set(
                            next(
                                segment.synapses
                                for neuron_index, segment in segments
                                if best
                                and id(segment)
                                == best["_segment_object_id"]
                            )
                        ),
                        active_ids,
                    )
                    if best
                    else 0.0
                ),
                "selected_winner_neuron": "",
                "selected_segment_id": "",
                "reinforced_segment_id": "",
                "created_segment_id": "",
                "created_new_segment": False,
                "selected_existing_segment": False,
                "reference_applicability": "",
                "_best_segment_object_ids": json.dumps(
                    sorted(
                        int(row["_segment_object_id"])
                        for row in best_segments
                    )
                ),
                "_passing_segment_object_ids": json.dumps(
                    sorted(
                        int(row["_segment_object_id"])
                        for row in passing
                    )
                ),
            }
        )
        if level != "summary":
            segment_rows.extend(segment_payloads)

    return MatchOverlapCapture(
        column_rows=tuple(column_rows),
        segment_rows=tuple(segment_rows),
        source_rows=tuple(source_rows),
    )


def finalize_match_overlap(
    capture: MatchOverlapCapture,
    observation_trace: ObservationTrace,
    registry: BranchProvenanceRegistry,
) -> MatchOverlapCapture:
    """Attach outcomes from the already-completed real observation."""

    events = {event.target_column: event for event in observation_trace.events}
    column_rows: list[dict[str, object]] = []
    for original in capture.column_rows:
        row = dict(original)
        event = events.get(int(row["encoded_column"]))
        if event is not None:
            if event.best_matching_segment is not None:
                row["best_matching_segment_id"] = _segment_id(
                    registry,
                    event.best_matching_segment,
                )
                row["best_matching_neuron"] = event.best_matching_neuron
                row["best_matching_overlap"] = event.best_matching_overlap
                row["gap_to_L_match"] = max(
                    0,
                    int(row["L_match"]) - event.best_matching_overlap,
                )
            row.update(
                {
                    "observe_scenario": event.scenario,
                    "scenario_assignment_reason": (
                        event.scenario_assignment_reason
                    ),
                    "selected_winner_neuron": event.winner_neuron,
                    "selected_segment_id": (
                        _segment_id(registry, event.selected_segment)
                        if event.selected_segment is not None
                        else ""
                    ),
                    "reinforced_segment_id": (
                        _segment_id(registry, event.reinforced_segment)
                        if event.reinforced_segment is not None
                        else ""
                    ),
                    "created_segment_id": (
                        _segment_id(registry, event.created_segment)
                        if event.created_segment is not None
                        else ""
                    ),
                    "created_new_segment": event.created_segment is not None,
                    "selected_existing_segment": (
                        event.selected_segment is not None
                        and event.created_segment is None
                    ),
                    "reference_applicability": (
                        "PREEXISTING"
                        if event.created_segment is None
                        else "POST_OBSERVATION_CREATED"
                    ),
                }
            )
        column_rows.append(_public_row(row))

    segment_rows: list[dict[str, object]] = []
    for original in capture.segment_rows:
        row = dict(original)
        event = events.get(int(row["encoded_column"]))
        object_id = int(row["_segment_object_id"])
        if event is not None:
            row["observe_scenario"] = event.scenario
            row["scenario_assignment_reason"] = (
                event.scenario_assignment_reason
            )
            row["selected_winner_neuron"] = event.winner_neuron
            row["selected_as_best_matching"] = (
                event.best_matching_segment is not None
                and id(event.best_matching_segment) == object_id
            )
            row["reinforced"] = (
                event.reinforced_segment is not None
                and id(event.reinforced_segment) == object_id
            )
            row["selected_for_scenario"] = (
                event.selected_segment is not None
                and id(event.selected_segment) == object_id
            )
        segment_rows.append(_public_row(row))

    return MatchOverlapCapture(
        column_rows=tuple(column_rows),
        segment_rows=tuple(segment_rows),
        source_rows=tuple(_public_row(dict(row)) for row in capture.source_rows),
    )


def _public_row(row: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if not key.startswith("_")}
