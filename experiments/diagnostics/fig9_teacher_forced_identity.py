"""Future-observed winner reference capture for Fig.9 diagnostics.

The rows are created only after the normal online observation has completed.
They are an operational teacher-forced reference, not a selector and not a
biological ground-truth neuron assignment.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

from experiments.diagnostics.fig9_branch_provenance import (
    BranchProvenanceRegistry,
    stable_source_fingerprint,
)
from experiments.diagnostics.fig9_candidate_score_trace import field_for_column
from experiments.diagnostics.fig9_oracle_candidate import FieldColumnRanges
from experiments.fig9 import TaxiRecord, record_values
from seqmem.model import ObservationEventTrace, ObservationTrace, Segment, SequentialMemory


TEACHER_FORCED_LEVELS = {"summary", "cell", "segment"}
DIAGNOSTIC_MARKERS = {
    "diagnostic_only": True,
    "offline_analysis_only": True,
    "teacher_forced_reference_only": True,
    "future_observation_does_not_affect_past_prediction": True,
    "ground_truth_does_not_affect_prediction": True,
    "identity_metrics_do_not_affect_selection": True,
}


def stable_observation_reference_id(
    *,
    stream_label: str,
    actual_record_index: int,
    actual_timestamp: str,
    target_column: int,
) -> str:
    """Return a checkpoint-independent ID for one observed target column."""

    payload = (
        stream_label,
        actual_record_index,
        actual_timestamp,
        target_column,
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _segment_count(model: SequentialMemory) -> int:
    return sum(
        len(neuron.segments)
        for column in model.columns
        for neuron in column.neurons
    )


def _cell_ids(value: Iterable[int]) -> str:
    return " ".join(str(cell_id) for cell_id in sorted(set(value)))


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _segment_fields(
    segment: Segment | None,
    registry: BranchProvenanceRegistry,
    *,
    current_context: set[int],
) -> dict[str, object]:
    if segment is None:
        return {
            "segment_provenance_id": "",
            "segment_creation_transition_index": "",
            "creation_source_fingerprint": "",
            "creation_source_cell_ids": "",
            "current_source_fingerprint": "",
            "current_source_cell_ids": "",
            "source_cell_count": 0,
            "source_context_jaccard": "",
            "creation_current_source_jaccard": "",
            "segment_provenance_available": False,
            "segment_provenance_missing_reason": "no_selected_segment",
        }
    current_sources = set(segment.synapses)
    origin = registry.provenance_for(segment)
    if origin is None:
        return {
            "segment_provenance_id": "",
            "segment_creation_transition_index": "",
            "creation_source_fingerprint": "",
            "creation_source_cell_ids": "",
            "current_source_fingerprint": stable_source_fingerprint(
                current_sources
            ),
            "current_source_cell_ids": _cell_ids(current_sources),
            "source_cell_count": len(current_sources),
            "source_context_jaccard": _jaccard(
                current_sources,
                current_context,
            ),
            "creation_current_source_jaccard": "",
            "segment_provenance_available": False,
            "segment_provenance_missing_reason": "registry_origin_unavailable",
        }
    creation_sources = set(origin.creation_source_cell_ids)
    return {
        "segment_provenance_id": origin.segment_provenance_id,
        "segment_creation_transition_index": (
            origin.creation_transition_index
        ),
        "creation_source_fingerprint": (
            origin.creation_source_fingerprint
        ),
        "creation_source_cell_ids": _cell_ids(creation_sources),
        "current_source_fingerprint": stable_source_fingerprint(
            current_sources
        ),
        "current_source_cell_ids": _cell_ids(current_sources),
        "source_cell_count": len(current_sources),
        "source_context_jaccard": _jaccard(
            current_sources,
            current_context,
        ),
        "creation_current_source_jaccard": _jaccard(
            creation_sources,
            current_sources,
        ),
        "segment_provenance_available": True,
        "segment_provenance_missing_reason": "",
    }


def _source_support_class(
    segment: Segment | None,
    *,
    predicted_sources: set[int],
    burst_sources: set[int],
) -> str:
    if segment is None:
        return "unavailable"
    sources = set(segment.synapses)
    predicted = bool(sources & predicted_sources)
    burst = bool(sources & burst_sources)
    if predicted and burst:
        return "burst-assisted"
    if predicted:
        return "predicted-supported"
    if burst:
        return "burst-only"
    return "unavailable"


def build_teacher_forced_observation_rows(
    *,
    model: SequentialMemory,
    registry: BranchProvenanceRegistry,
    trace: ObservationTrace,
    stream_label: str,
    actual_record_index: int,
    actual_record: TaxiRecord,
    ranges: FieldColumnRanges,
    pre_observe_segment_count: int,
    level: str,
    predicted_sources: set[int] | None = None,
    burst_sources: set[int] | None = None,
) -> list[dict[str, object]]:
    """Serialize identities chosen by the already-completed real observation."""

    if level not in TEACHER_FORCED_LEVELS:
        raise ValueError("teacher-forced winner level must be summary, cell, or segment")
    timestamp = actual_record.timestamp.isoformat(sep=" ")
    values = record_values(actual_record)
    events_by_column: dict[int, list[ObservationEventTrace]] = defaultdict(list)
    for event in trace.events:
        events_by_column[event.target_column].append(event)
    pre_winners = set(trace.pre_observe_previous_winners)
    predicted_sources = set(predicted_sources or ())
    burst_sources = set(burst_sources or ())
    resulting_fingerprint = stable_source_fingerprint(model.previous_winners)
    post_segment_count = _segment_count(model)
    rows: list[dict[str, object]] = []
    for target_column, events in sorted(events_by_column.items()):
        winner_cells = {event.winner_cell_id for event in events}
        winner_neurons = {event.winner_neuron for event in events}
        representative = events[0]
        field = field_for_column(target_column, ranges)
        field_index = {"weekday": 0, "time": 1, "passenger": 2}[field]
        selected_segment = representative.selected_segment
        reinforced_segment = representative.reinforced_segment
        created_segment = representative.created_segment
        selected_fields = _segment_fields(
            selected_segment,
            registry,
            current_context=pre_winners,
        )
        reinforced_fields = _segment_fields(
            reinforced_segment,
            registry,
            current_context=pre_winners,
        )
        created_fields = _segment_fields(
            created_segment,
            registry,
            current_context=pre_winners,
        )
        segment_available = bool(
            selected_fields["segment_provenance_available"]
        )
        missing_reason = str(
            selected_fields["segment_provenance_missing_reason"]
        )
        if level != "segment":
            selected_fields = {
                key: "" if key != "source_cell_count" else 0
                for key in selected_fields
            }
            reinforced_fields["segment_provenance_id"] = ""
            created_fields["segment_provenance_id"] = ""
            segment_available = False
            missing_reason = "not_captured_at_requested_level"
        winner_available = bool(winner_cells)
        rows.append(
            {
                **DIAGNOSTIC_MARKERS,
                "observation_reference_id": stable_observation_reference_id(
                    stream_label=stream_label,
                    actual_record_index=actual_record_index,
                    actual_timestamp=timestamp,
                    target_column=target_column,
                ),
                "stream_label": stream_label,
                "actual_record_index": actual_record_index,
                "actual_timestamp": timestamp,
                "field": field,
                "target_column": target_column,
                "target_value": values[field_index],
                "observed_winner_column": (
                    target_column if winner_available else ""
                ),
                "observed_winner_neuron": (
                    next(iter(winner_neurons))
                    if len(winner_neurons) == 1
                    else ""
                ),
                "observed_winner_neurons": _cell_ids(winner_neurons),
                "observed_winner_cell_id": (
                    next(iter(winner_cells))
                    if len(winner_cells) == 1
                    else ""
                ),
                "observed_winner_cell_ids": _cell_ids(winner_cells),
                "observed_winner_available": winner_available,
                "observed_winner_count_in_column": len(winner_cells),
                "pre_observe_previous_winner_count": len(pre_winners),
                "pre_observe_context_fingerprint": (
                    stable_source_fingerprint(pre_winners)
                ),
                "pre_observe_source_cell_ids": (
                    _cell_ids(pre_winners) if level != "summary" else ""
                ),
                "pre_observe_segment_count": pre_observe_segment_count,
                "observe_scenario": representative.scenario,
                "scenario1": representative.scenario == "scenario1",
                "scenario2": representative.scenario == "scenario2",
                "scenario3": representative.scenario == "scenario3",
                "reinforced_segment_id": (
                    reinforced_fields["segment_provenance_id"]
                ),
                "created_segment_id": (
                    created_fields["segment_provenance_id"]
                ),
                "selected_segment_id": (
                    selected_fields["segment_provenance_id"]
                ),
                "segment_provenance_available": segment_available,
                "source_support_class": _source_support_class(
                    selected_segment,
                    predicted_sources=predicted_sources,
                    burst_sources=burst_sources,
                ),
                "resulting_winner_fingerprint": resulting_fingerprint,
                "post_observe_segment_count": post_segment_count,
                "segment_creation_transition_index": (
                    selected_fields["segment_creation_transition_index"]
                ),
                "creation_source_fingerprint": (
                    selected_fields["creation_source_fingerprint"]
                ),
                "creation_source_cell_ids": (
                    selected_fields["creation_source_cell_ids"]
                ),
                "current_source_fingerprint": (
                    selected_fields["current_source_fingerprint"]
                ),
                "current_source_cell_ids": (
                    selected_fields["current_source_cell_ids"]
                ),
                "source_cell_count": selected_fields["source_cell_count"],
                "source_context_jaccard": (
                    selected_fields["source_context_jaccard"]
                ),
                "creation_current_source_jaccard": (
                    selected_fields["creation_current_source_jaccard"]
                ),
                "provenance_missing_reason": missing_reason,
            }
        )
    return rows


def winner_set(row: Mapping[str, object]) -> set[int]:
    """Parse the explicit winner set used by offline identity matching."""

    raw = str(row.get("observed_winner_neurons", "")).strip()
    return {int(value) for value in raw.split()} if raw else set()


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
