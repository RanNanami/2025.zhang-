"""Behavior-neutral option values shared by the strict Fig.9 CLI.

These values mirror the public diagnostic level sets. Keeping CLI validation
in a lightweight module prevents parser construction from importing complete
trace and report implementations.
"""

from __future__ import annotations


BRANCH_LEVELS = frozenset({"summary", "candidate", "full"})
CONTEXT_TRAJECTORY_LEVELS = frozenset({"summary", "column", "cell"})
MATCH_OVERLAP_LEVELS = frozenset({"summary", "segment", "source"})
OBSERVE_SCENARIO_LEVELS = frozenset({"summary", "column", "full"})
PRESELECTION_LEVELS = frozenset({"summary", "crossing", "full"})
REFERENCE_NEURON_SELECTION_LEVELS = frozenset(
    {"summary", "crossing", "full"}
)
SEGMENT_CONTEXT_COMPOSITION_LEVELS = frozenset(
    {"summary", "match", "source"}
)
SEGMENT_REINFORCEMENT_LEVELS = frozenset({"summary", "event", "segment"})
TEACHER_FORCED_LEVELS = frozenset({"summary", "cell", "segment"})
