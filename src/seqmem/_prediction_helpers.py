"""Small deterministic packaging helpers for the prediction pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar


CandidateT = TypeVar("CandidateT")
EventT = TypeVar("EventT")


def segment_upper_bound_is_below_threshold(
    v_rest: float,
    upper_bound: float,
    threshold: float,
) -> bool:
    """Preserve the existing upper-bound elimination expression exactly."""

    return v_rest + upper_bound < threshold


def prediction_time_is_outside_window(
    target_time: float,
    cycle_period: float,
    timing_tolerance: float,
) -> bool:
    """Preserve the existing short-circuit soma-window rejection test."""

    return (
        target_time < cycle_period
        or target_time
        > cycle_period + cycle_period / 2.0 + timing_tolerance
    )


def prediction_event_key(
    column_id: int,
    target_time: float,
) -> tuple[int, float]:
    """Package the established column/time key with its exact rounding."""

    return (column_id, round(target_time, 12))


def build_prediction_candidate(
    *,
    candidate_factory: Callable[..., CandidateT],
    contribution_type: type[object],
    neuron_index: int,
    score: float,
    target_time: float,
    cycle_period: float,
    segment: object,
    metadata: Mapping[str, object],
    dendrite_threshold: float,
) -> CandidateT:
    """Construct one candidate from the selected segment's existing metadata."""

    crossing_time = metadata.get("first_threshold_crossing_time")
    peak_potential = metadata.get("peak_dendritic_potential")
    soma_time = metadata.get("predicted_soma_firing_time")
    return candidate_factory(
        neuron_index=neuron_index,
        score=score,
        time=target_time - cycle_period,
        segment=segment,
        dendritic_crossing_time=(
            crossing_time if isinstance(crossing_time, float) else None
        ),
        crossing_synapse_contributions=tuple(
            item
            for item in metadata.get("crossing_synapse_contributions", ())
            if isinstance(item, contribution_type)
        ),
        peak_dendritic_potential=(
            peak_potential if isinstance(peak_potential, float) else None
        ),
        threshold_margin=(
            peak_potential - dendrite_threshold
            if isinstance(peak_potential, float)
            else None
        ),
        predicted_soma_firing_time=(
            soma_time if isinstance(soma_time, float) else None
        ),
    )


def build_prediction_stats(
    *,
    active_source_count: int,
    candidate_segment_count: int,
    threshold_crossing_segment_count: int,
    accepted_candidate_count: int,
    raw_event_count: int,
    raw_predicted_column_count: int,
) -> dict[str, int | float]:
    """Publish prediction counters with the established key order."""

    return {
        "active_source_count": active_source_count,
        "candidate_segment_count": candidate_segment_count,
        "threshold_crossing_segment_count": threshold_crossing_segment_count,
        "accepted_candidate_count": accepted_candidate_count,
        "raw_event_count": raw_event_count,
        "raw_predicted_column_count": raw_predicted_column_count,
    }


def build_emitted_spike_events(
    selected_events: Sequence[tuple[int, tuple[int, float, float, object]]],
    *,
    cycle_period: float,
    event_factory: Callable[..., EventT],
) -> tuple[EventT, ...]:
    """Construct emitted events once, preserving selected-event order."""

    return tuple(
        event_factory(
            column=column_id,
            time=target_time - cycle_period,
        )
        for column_id,
        (_neuron_index, _score, target_time, _segment) in selected_events
    )
