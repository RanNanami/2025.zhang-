"""Pure names for the existing observation-side learning decisions."""

from __future__ import annotations

from types import MappingProxyType
from typing import Iterable, Protocol


class MutableSynapse(Protocol):
    """Structural type for the two fields changed by learning rules."""

    weight: float
    age: int


class MutableSegment(Protocol):
    """Structural type for a segment passed explicitly by its model owner."""

    synapses: dict[int, MutableSynapse]


NO_LEARNING_BRANCH = ""
SCENARIO1_BRANCH = "scenario1"
SCENARIO2_BRANCH = "scenario2"
SCENARIO3_BRANCH = "scenario3"

# Documentation/test mapping only. Existing model counters and artifacts keep
# their historical labels, including observation-side ``scenario3`` for S2B.
CANONICAL_PAPER_BRANCH_BY_INTERNAL = MappingProxyType(
    {
        NO_LEARNING_BRANCH: "NONE",
        SCENARIO1_BRANCH: "PAPER_S1",
        SCENARIO2_BRANCH: "PAPER_S2A",
        SCENARIO3_BRANCH: "PAPER_S2B",
        "wrong_prediction_punishment": "PAPER_S3",
    }
)


def classify_observation_learning_branch(
    *,
    learning_enabled: bool,
    has_matching_segment: bool,
    was_predicted: bool,
    matching_segment_eligible: bool,
) -> str:
    """Return the historical branch label from already-resolved evidence.

    The no-match path intentionally keeps returning ``scenario3`` even when
    learning is disabled. That is the current observable behavior; this helper
    names it without correcting or reinterpreting it.
    """

    if not has_matching_segment:
        return SCENARIO3_BRANCH
    if not learning_enabled:
        return NO_LEARNING_BRANCH
    if was_predicted:
        return SCENARIO1_BRANCH
    if matching_segment_eligible:
        return SCENARIO2_BRANCH
    return SCENARIO3_BRANCH


def apply_selected_segment_updates(
    synapses: dict[int, MutableSynapse],
    contributed_sources: set[int],
    *,
    delta_w: float,
    depress_noncontributing: bool,
) -> None:
    """Mutate one already-selected segment in existing dictionary order."""

    for source, synapse in synapses.items():
        if source in contributed_sources:
            synapse.weight = min(1.0, synapse.weight + delta_w)
            synapse.age = 0
        elif depress_noncontributing:
            synapse.weight = max(0.0, synapse.weight - delta_w)
            synapse.age += 1


def depress_other_segment_updates(
    segments: Iterable[MutableSegment],
    selected_segment: MutableSegment,
    *,
    delta_w: float,
    enabled: bool,
) -> None:
    """Depress other segments after the selected-segment update."""

    for segment in segments:
        if segment is selected_segment:
            continue
        for synapse in segment.synapses.values():
            if enabled:
                synapse.weight = max(0.0, synapse.weight - delta_w)
                synapse.age += 1


def apply_failed_prediction_updates(
    synapses: dict[int, MutableSynapse],
    contributed_sources: set[int],
    *,
    delta_w_bad: float,
) -> None:
    """Punish already-resolved contributing sources in set iteration order."""

    for source in contributed_sources:
        synapse = synapses[source]
        synapse.weight = max(0.0, synapse.weight - delta_w_bad)
        synapse.age += 1
