"""Read-only Fig.9 segment-selection and reinforcement trace helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

from experiments.diagnostics.fig9_branch_provenance import BranchProvenanceRegistry
from experiments.diagnostics.fig9_candidate_score_trace import field_for_column
from experiments.diagnostics.fig9_oracle_candidate import FieldColumnRanges
from experiments.fig9 import TaxiRecord
from seqmem.model import ObservationTrace, ReinforcementTrace, SequentialMemory


SEGMENT_REINFORCEMENT_LEVELS = {"summary", "event", "segment"}
REFERENCE_STATUSES = {
    "REFERENCE_COMPATIBLE",
    "REFERENCE_INCOMPATIBLE",
    "REFERENCE_UNAVAILABLE",
    "REFERENCE_AMBIGUOUS",
    "POST_OBSERVATION_REFERENCE_INVALID",
}
DIAGNOSTIC_MARKERS = {
    "diagnostic_only": True,
    "ground_truth_used_for_analysis_only": True,
    "ground_truth_does_not_affect_model": True,
    "reinforcement_trace_does_not_affect_learning": True,
    "counterfactual_does_not_affect_model": True,
    "uses_compensation": False,
}


def _segment_id(registry: BranchProvenanceRegistry, segment: object) -> str:
    origin = registry.provenance_for(segment)  # type: ignore[arg-type]
    if origin is not None:
        return origin.segment_provenance_id
    diagnostic_id = getattr(segment, "diagnostic_id", None)
    return f"diagnostic:{diagnostic_id}" if diagnostic_id is not None else ""


def _rank(value: float, values: Sequence[float]) -> int:
    """Return competition rank with ties sharing the same rank."""

    return 1 + sum(candidate > value for candidate in values)


def _reference_fields(*, scenario: str, candidate_count: int) -> dict[str, object]:
    """Label only references that existed independently before observation.

    Scenario 1 has a time-matched predictive identity before the proximal input.
    Scenario 2's so-called teacher reference is created by the same matching
    decision being audited, so using it would be circular and is prohibited.
    """

    if scenario == "scenario1" and candidate_count == 1:
        return {
            "reference_applicable": True,
            "reference_segment_available": True,
            "reference_neuron_available": True,
            "selected_segment_reference_compatible": True,
            "selected_neuron_reference_compatible": True,
            "reference_status": "REFERENCE_COMPATIBLE",
        }
    if scenario == "scenario1" and candidate_count > 1:
        return {
            "reference_applicable": True,
            "reference_segment_available": False,
            "reference_neuron_available": True,
            "selected_segment_reference_compatible": "",
            "selected_neuron_reference_compatible": "",
            "reference_status": "REFERENCE_AMBIGUOUS",
        }
    return {
        "reference_applicable": False,
        "reference_segment_available": False,
        "reference_neuron_available": False,
        "selected_segment_reference_compatible": "",
        "selected_neuron_reference_compatible": "",
        "reference_status": "POST_OBSERVATION_REFERENCE_INVALID",
    }


def build_segment_reinforcement_rows(
    *,
    model: SequentialMemory,
    registry: BranchProvenanceRegistry,
    observation_trace: ObservationTrace,
    reinforcement_traces: Sequence[ReinforcementTrace],
    actual_record_index: int,
    actual_record: TaxiRecord,
    ranges: FieldColumnRanges,
    level: str,
) -> list[dict[str, object]]:
    """Join callbacks to the exact observation events that triggered them."""

    if level not in SEGMENT_REINFORCEMENT_LEVELS:
        raise ValueError("segment reinforcement level must be summary, event, or segment")
    callbacks: dict[tuple[int, int, int], list[ReinforcementTrace]] = defaultdict(list)
    for trace in reinforcement_traces:
        callbacks[(trace.target_column, trace.target_neuron, trace.segment_identity)].append(trace)

    timestamp = actual_record.timestamp.isoformat(sep=" ")
    rows: list[dict[str, object]] = []
    for event in observation_trace.events:
        segment = event.reinforced_segment
        if segment is None:
            continue
        key = (event.target_column, event.winner_neuron, id(segment))
        if not callbacks[key]:
            raise RuntimeError("reinforced observation event has no matching callback trace")
        reinforcement = callbacks[key].pop(0)

        if event.scenario == "scenario1":
            candidates = [
                candidate
                for candidate in model.last_prediction_candidates.get(event.target_column, ())
                if abs(candidate.time - event.target_time) <= model.params.timing_tolerance
            ]
            candidate_payload = [
                (candidate.segment, candidate.neuron_index, candidate.score)
                for candidate in candidates
            ]
        else:
            candidate_payload = [
                (candidate.segment, candidate.target_neuron, candidate.candidate_score)
                for candidate in event.matching_candidates
                if candidate.timed_overlap >= model.params.l_match
            ]

        selected_id = _segment_id(registry, segment)
        overlap_by_id = {
            _segment_id(registry, candidate.segment): candidate.timed_overlap
            for candidate in event.matching_candidates
        }
        if selected_id not in overlap_by_id:
            overlap_by_id[selected_id] = segment.timed_overlap(
                observation_trace.pre_observe_active_sources,
                model.params.cycle_period / 2.0 + event.target_time,
                model.params.timing_tolerance,
            )
        for candidate_segment, _neuron, _score in candidate_payload:
            candidate_id = _segment_id(registry, candidate_segment)
            if candidate_id not in overlap_by_id:
                overlap_by_id[candidate_id] = candidate_segment.timed_overlap(
                    observation_trace.pre_observe_active_sources,
                    model.params.cycle_period / 2.0 + event.target_time,
                    model.params.timing_tolerance,
                )
        selected_overlap = overlap_by_id[selected_id]
        candidate_scores = [float(item[2]) for item in candidate_payload]
        candidate_overlaps = [
            overlap_by_id.get(_segment_id(registry, item[0]), selected_overlap)
            for item in candidate_payload
        ]
        selected_score = float(event.selected_candidate_score)
        matching_segment_count = len({id(item[0]) for item in candidate_payload})
        matching_neuron_count = len({int(item[1]) for item in candidate_payload})
        origin = registry.provenance_for(segment)
        reference = _reference_fields(
            scenario=event.scenario,
            candidate_count=matching_segment_count,
        )
        reinforcement_before = (
            reinforcement.scenario1_count_before
            + reinforcement.scenario2_count_before
        )
        reinforcement_after = (
            reinforcement.scenario1_count_after
            + reinforcement.scenario2_count_after
        )
        weight_before = sum(reinforcement.weights_before)
        weight_after = sum(reinforcement.weights_after)
        rows.append(
            {
                **DIAGNOSTIC_MARKERS,
                "segment_reinforcement_level": level,
                "actual_record_index": actual_record_index,
                "timestamp": timestamp,
                "field": field_for_column(event.target_column, ranges),
                "encoded_column": event.target_column,
                "observe_scenario": event.scenario,
                "L_match": model.params.l_match,
                "segment_provenance_id": selected_id,
                "target_neuron": event.winner_neuron,
                "segment_creation_index": (
                    origin.creation_transition_index if origin is not None else ""
                ),
                "segment_age": max(
                    (synapse.age for synapse in segment.synapses.values()),
                    default=0,
                ),
                "selected_overlap": selected_overlap,
                "selected_candidate_score": selected_score,
                "selected_response_peak": selected_score,
                "matching_segment_count": matching_segment_count,
                "matching_neuron_count": matching_neuron_count,
                "max_overlap": max(candidate_overlaps, default=selected_overlap),
                "max_candidate_score": max(candidate_scores, default=selected_score),
                "overlap_rank": _rank(float(selected_overlap), [float(v) for v in candidate_overlaps]),
                "score_rank": _rank(selected_score, candidate_scores),
                "ambiguous_segment": matching_segment_count > 1,
                "ambiguous_neuron": matching_neuron_count > 1,
                "l2_only_match": selected_overlap in {2, 3},
                "passes_L2": selected_overlap >= 2,
                "passes_L3": selected_overlap >= 3,
                "passes_L4": selected_overlap >= 4,
                "reinforced": True,
                "weight_sum_before": weight_before,
                "weight_sum_after": weight_after,
                "reinforcement_delta": weight_after - weight_before,
                "source_count_before": reinforcement.source_count_before,
                "source_count_after": reinforcement.source_count_after,
                "reinforcement_count_before": reinforcement_before,
                "reinforcement_count_after": reinforcement_after,
                **reference,
                "selected_again_within_1": False,
                "selected_again_within_3": False,
                "selected_again_within_5": False,
                "reinforced_again_within_1": False,
                "reinforced_again_within_3": False,
                "reinforced_again_within_5": False,
                "future_reinforcement_count_5": 0,
                "future_scenario3_count_5": 0,
                "segment_deleted_within_5": "",
                "downstream_prediction_error_step1": "",
                "downstream_prediction_error_step2": "",
                "downstream_prediction_error_step3": "",
                "downstream_prediction_error_step4": "",
                "downstream_prediction_error_step5": "",
            }
        )
    if any(callbacks.values()):
        raise RuntimeError("reinforcement callback trace was not joined to an observation event")
    return rows


def enrich_segment_reinforcement_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    observation_rows: Sequence[Mapping[str, object]],
    density_rows: Sequence[Mapping[str, object]],
    final_segment_ids: set[str],
) -> list[dict[str, object]]:
    """Add future associations after the run without changing model execution."""

    reinforced_by_segment: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        reinforced_by_segment[str(row["segment_provenance_id"])].append(
            int(row["actual_record_index"])
        )
    scenario3_by_index: dict[int, int] = defaultdict(int)
    for row in observation_rows:
        if row.get("observe_scenario") == "scenario3":
            scenario3_by_index[int(row["actual_record_index"])] += 1
    errors = {
        (int(row["record_index"]), int(row["horizon_step"])): row.get(
            "absolute_percentage_error", ""
        )
        for row in density_rows
        if row.get("record_index", "") != ""
        and row.get("horizon_step", "") != ""
    }
    output: list[dict[str, object]] = []
    for source in rows:
        row = dict(source)
        index = int(row["actual_record_index"])
        future = [
            candidate
            for candidate in reinforced_by_segment[str(row["segment_provenance_id"])]
            if index < candidate <= index + 5
        ]
        for horizon in (1, 3, 5):
            seen = any(candidate <= index + horizon for candidate in future)
            row[f"selected_again_within_{horizon}"] = seen
            row[f"reinforced_again_within_{horizon}"] = seen
        row["future_reinforcement_count_5"] = len(future)
        row["future_scenario3_count_5"] = sum(
            scenario3_by_index[candidate]
            for candidate in range(index + 1, index + 6)
        )
        row["segment_deleted_within_5"] = (
            str(row["segment_provenance_id"]) not in final_segment_ids
        )
        for horizon in range(1, 6):
            # The next base-record rollout is the first prediction made after
            # this reinforcement. This is correlation-only, not causation.
            row[f"downstream_prediction_error_step{horizon}"] = errors.get(
                (index + 1, horizon), ""
            )
        output.append(row)
    return output
