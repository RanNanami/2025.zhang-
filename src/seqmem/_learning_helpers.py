"""Pure names for the existing observation-side learning decisions."""

from __future__ import annotations

from types import MappingProxyType


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
