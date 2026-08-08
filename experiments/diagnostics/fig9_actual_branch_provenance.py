"""Read-only actual-history branch provenance for Fig.9.

This tracker is deliberately outside ``SequentialMemory``.  It observes the
completed actual transition and is queried before the next transition starts.
It never participates in prediction, matching, selection, reinforcement, or
random-number generation.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

from experiments.diagnostics.fig9_branch_provenance import BranchProvenanceRegistry
from experiments.diagnostics.fig9_candidate_score_trace import field_for_column
from experiments.diagnostics.fig9_oracle_candidate import FieldColumnRanges
from seqmem.encoding import SymbolCode
from seqmem.model import ObservationTrace, Segment, SequentialMemory


MARKERS = {
    "diagnostic_only": True,
    "offline_analysis_only": True,
    "branch_provenance_does_not_affect_model": True,
    "branch_provenance_does_not_affect_matching": True,
    "branch_provenance_does_not_affect_selection": True,
    "branch_provenance_does_not_affect_reinforcement": True,
    "branch_provenance_does_not_affect_prediction": True,
    "branch_provenance_does_not_affect_rng": True,
    "ground_truth_used_for_analysis_only": True,
    "uses_compensation": False,
}

STATUSES = {
    "UNIQUE_ACTUAL_BRANCH_CONTINUATION",
    "MULTIPLE_ACTUAL_BRANCH_COMPATIBLE",
    "ACTUAL_BRANCH_SWITCH",
    "ACTUAL_BRANCH_NOT_FOUND",
    "ACTUAL_BRANCH_MIXED_HISTORY",
    "ACTUAL_BRANCH_CHAIN_BROKEN",
    "CURRENT_MATCH_CIRCULAR_INVALID",
    "POST_OBSERVATION_BRANCH_INVALID",
}

LINEAGE_CLASSES = {
    "LINEAGE_ROOT",
    "LINEAGE_CONTINUATION_UNIQUE",
    "LINEAGE_CONTINUATION_MULTIPLE",
    "LINEAGE_MERGE",
    "LINEAGE_SPLIT",
    "LINEAGE_UNRESOLVED",
    "POSTHOC_INVALID",
}


def stable_id(prefix: str, *parts: object) -> str:
    payload = json.dumps([prefix, *parts], sort_keys=False, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_signature(source_cells: Mapping[int, float] | set[int] | tuple[int, ...]) -> str:
    if isinstance(source_cells, Mapping):
        values = sorted(int(value) for value in source_cells)
    else:
        values = sorted(int(value) for value in source_cells)
    return stable_id("actual-source", *values)


def _ids(value: object) -> set[str]:
    return {item for item in str(value or "").split("|") if item}


def _iter_column_segments(model: SequentialMemory, column: int):
    if 0 <= column < len(model.columns):
        for neuron_index, neuron in enumerate(model.columns[column].neurons):
            for segment in neuron.segments:
                yield neuron_index, segment


class ActualBranchProvenanceTracker:
    """Checkpointable history keyed only by stable segment provenance IDs."""

    VERSION = "fig9-actual-branch-provenance-v2"

    def __init__(self) -> None:
        self.segments: dict[str, dict[str, object]] = {}
        self.anchors: dict[str, dict[str, object]] = {}
        self.branches: dict[str, dict[str, object]] = {}
        self.lineages: dict[str, dict[str, object]] = {}
        self.lineage_events: list[dict[str, object]] = []

    def state(self, segment_id: str) -> dict[str, object] | None:
        return self.segments.get(segment_id)

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "version": self.VERSION,
            "segments": self.segments,
            "anchors": self.anchors,
            "branches": self.branches,
            "lineages": self.lineages,
            "lineage_events": self.lineage_events,
        }

    @classmethod
    def from_checkpoint_payload(cls, payload: Mapping[str, object] | None) -> "ActualBranchProvenanceTracker":
        tracker = cls()
        if not payload:
            return tracker
        for name in ("segments", "anchors", "branches", "lineages"):
            raw = payload.get(name, {})
            if isinstance(raw, Mapping):
                setattr(tracker, name, {str(key): dict(value) for key, value in raw.items() if isinstance(value, Mapping)})
        raw_events = payload.get("lineage_events", [])
        if isinstance(raw_events, list):
            tracker.lineage_events = [dict(value) for value in raw_events if isinstance(value, Mapping)]
        return tracker

    def _new_anchor(
        self,
        *,
        segment_id: str,
        index: int,
        field: str,
        column: int,
        neuron: int,
        source_sig: str,
        parent_anchor_ids: tuple[str, ...] = (),
    ) -> tuple[str, str, str, str]:
        # The first observed index is metadata, not part of identity. Repeated
        # reinforcement of the same segment/source must retain one anchor.
        anchor_id = stable_id("actual-anchor", segment_id, field, column, neuron, source_sig)
        branch_id = stable_id("actual-branch", anchor_id, segment_id, source_sig)
        parent_anchor_ids = tuple(sorted(set(parent_anchor_ids)))
        parent_lineage_ids = tuple(
            sorted(
                {
                    str(self.anchors[parent_id].get("actual_lineage_id", ""))
                    for parent_id in parent_anchor_ids
                    if parent_id in self.anchors
                    and self.anchors[parent_id].get("actual_lineage_id")
                }
            )
        )
        if not parent_anchor_ids:
            lineage_class = "LINEAGE_ROOT"
            lineage_id = stable_id(
                "actual-lineage-root", segment_id, field, column, neuron, source_sig
            )
        elif len(parent_lineage_ids) != len(parent_anchor_ids):
            lineage_class = "LINEAGE_UNRESOLVED"
            lineage_id = ""
        elif len(parent_lineage_ids) == 1:
            lineage_class = (
                "LINEAGE_CONTINUATION_UNIQUE"
                if len(parent_anchor_ids) == 1
                else "LINEAGE_CONTINUATION_MULTIPLE"
            )
            lineage_id = parent_lineage_ids[0]
        else:
            lineage_class = "LINEAGE_MERGE"
            lineage_id = stable_id("actual-lineage-merge", *parent_lineage_ids)
        if anchor_id not in self.anchors:
            self.anchors[anchor_id] = {
                "record_type": "ANCHOR",
                "actual_anchor_id": anchor_id,
                "anchor_creation_index": index,
                "anchor_field": field,
                "anchor_column": column,
                "anchor_neuron": neuron,
                "anchor_segment_provenance_id": segment_id,
                "parent_actual_anchor_ids": "|".join(parent_anchor_ids),
                "parent_actual_lineage_ids": "|".join(parent_lineage_ids),
                "actual_lineage_id": lineage_id,
                "lineage_class": lineage_class,
                "source_provenance_signature": source_sig,
                "first_verified_index": index,
                "last_seen_index": index,
                "actual_branch_provenance_id": branch_id,
            }
        else:
            self.anchors[anchor_id]["last_seen_index"] = index
            # Keep the first causal assignment stable when the same anchor is
            # observed again with a different surrounding diagnostic state.
            lineage_id = str(self.anchors[anchor_id].get("actual_lineage_id", lineage_id))
            lineage_class = str(self.anchors[anchor_id].get("lineage_class", lineage_class))
        if branch_id not in self.branches:
            self.branches[branch_id] = {
                "record_type": "OLD_BRANCH",
                "actual_branch_provenance_id": branch_id,
                "root_actual_anchor_id": anchor_id,
                "parent_branch_ids": "",
                "branch_depth": 0,
                "branch_source_signature": source_sig,
                "branch_is_unique": True,
                "branch_is_mixed": False,
                "branch_chain_broken": False,
                "creation_index": index,
                "creation_field": field,
                "creation_column": column,
                "creation_neuron": neuron,
                "first_seen_index": index,
                "last_seen_index": index,
                "segment_count": 1,
                "reinforcement_count": 0,
                "mixed_with_other_branch": False,
                "chain_broken": False,
                "deletion_status": "live",
            }
        if lineage_id:
            lineage = self.lineages.setdefault(
                lineage_id,
                {
                    "record_type": "LINEAGE",
                    "actual_lineage_id": lineage_id,
                    "lineage_class": lineage_class,
                    "parent_actual_lineage_ids": "|".join(parent_lineage_ids),
                    "anchor_ids": "",
                    "segment_ids": "",
                    "first_seen_index": index,
                    "last_seen_index": index,
                    "anchor_count": 0,
                    "event_count": 0,
                },
            )
            lineage["last_seen_index"] = index
            lineage["lineage_class"] = lineage_class
            anchors = set(_ids(lineage.get("anchor_ids")))
            anchors.add(anchor_id)
            lineage["anchor_ids"] = "|".join(sorted(anchors))
            lineage["anchor_count"] = len(anchors)
        return anchor_id, branch_id, lineage_id, lineage_class

    def record_actual(
        self,
        *,
        segment_id: str,
        index: int,
        field: str,
        column: int,
        neuron: int,
        source_cells: Mapping[int, float] | set[int] | tuple[int, ...],
        reinforced: bool,
        parent_anchor_ids: Iterable[str] = (),
    ) -> dict[str, object]:
        sig = source_signature(source_cells)
        parent_anchor_ids = tuple(sorted(set(str(value) for value in parent_anchor_ids if value)))
        current_anchor_id = stable_id(
            "actual-anchor", segment_id, field, column, neuron, sig
        )
        # A repeated observation of the exact same anchor is not its own
        # parent.  Excluding the current identity keeps the causal graph
        # acyclic while preserving the existing stable anchor ID.
        parent_anchor_ids = tuple(
            value for value in parent_anchor_ids if value != current_anchor_id
        )
        state = self.segments.setdefault(segment_id, {
            "segment_provenance_id": segment_id,
            "segment_creation_index": index,
            "segment_creation_field": field,
            "segment_creation_column": column,
            "segment_creation_neuron": neuron,
            "creation_scenario": "actual_observation",
            "creation_source_signature": sig,
            "first_actual_anchor_id": "",
            "actual_anchor_ids_seen": [],
            "actual_branch_ids_seen": [],
            "actual_anchor_count": 0,
            "actual_branch_count": 0,
            "actual_lineage_ids_seen": [],
            "actual_lineage_count": 0,
            "lineage_level_mixed_history": False,
            "autonomous_anchor_ids_seen": [],
            "reinforcement_history_count": 0,
            "actual_reinforcement_count": 0,
            "autonomous_reinforcement_count": 0,
            "mixed_actual_history": False,
            "mixed_history_class": "NO_ACTUAL_HISTORY",
            "first_mixed_index": "",
            "event_that_created_mixture": "",
            "mixture_trigger": "",
            "deleted_index": "",
            "recreated_from_prior_id": "",
        })
        anchor_id, branch_id, lineage_id, lineage_class = self._new_anchor(
            segment_id=segment_id, index=index, field=field, column=column,
            neuron=neuron, source_sig=sig, parent_anchor_ids=parent_anchor_ids,
        )
        parent_lineage_ids = tuple(
            sorted(
                {
                    str(self.anchors[parent_id].get("actual_lineage_id", ""))
                    for parent_id in parent_anchor_ids
                    if parent_id in self.anchors
                    and self.anchors[parent_id].get("actual_lineage_id")
                }
            )
        )
        anchors = list(state.get("actual_anchor_ids_seen", []))
        previous_anchor_count = len(anchors)
        previous_branch_ids = set(str(value) for value in state.get("actual_branch_ids_seen", []))
        if anchor_id not in anchors:
            anchors.append(anchor_id)
        state["actual_anchor_ids_seen"] = anchors
        state["actual_anchor_count"] = len(anchors)
        branch_ids = list(state.get("actual_branch_ids_seen", []))
        if branch_id not in branch_ids:
            branch_ids.append(branch_id)
        state["actual_branch_ids_seen"] = branch_ids
        state["actual_branch_count"] = len(branch_ids)
        lineages = list(state.get("actual_lineage_ids_seen", []))
        if lineage_id and lineage_id not in lineages:
            lineages.append(lineage_id)
        state["actual_lineage_ids_seen"] = lineages
        state["actual_lineage_count"] = len(lineages)
        state["lineage_level_mixed_history"] = len(lineages) > 1
        state["first_actual_anchor_id"] = anchors[0]
        state["mixed_actual_history"] = len(branch_ids) > 1
        if len(anchors) > 1 and len(branch_ids) == 1:
            state["mixed_history_class"] = "MULTIPLE_ANCHORS_SAME_BRANCH"
        elif len(branch_ids) > 1:
            state["mixed_history_class"] = "ACTUAL_BRANCH_MIXED_HISTORY"
        elif previous_anchor_count == 0:
            state["mixed_history_class"] = "SINGLE_ACTUAL_ANCHOR"
        if len(branch_ids) > 1 and not state.get("first_mixed_index"):
            state["first_mixed_index"] = index
            state["event_that_created_mixture"] = index
            state["mixture_trigger"] = (
                "CREATION_WITH_MULTIPLE_ACTUAL_SOURCES"
                if previous_anchor_count == 0
                else "REINFORCEMENT_ADDED_NEW_ACTUAL_ANCHOR"
            )
        elif previous_anchor_count and len(anchors) > previous_anchor_count and len(branch_ids) == 1:
            state["mixture_trigger"] = "DUPLICATE_ANCHOR_ENTRY"
        state["reinforcement_history_count"] = int(state.get("reinforcement_history_count", 0)) + 1
        if reinforced:
            state["actual_reinforcement_count"] = int(state.get("actual_reinforcement_count", 0)) + 1
        branch = self.branches[branch_id]
        branch["last_seen_index"] = index
        branch["reinforcement_count"] = int(branch.get("reinforcement_count", 0)) + int(reinforced)
        branch["mixed_with_other_branch"] = bool(state["mixed_actual_history"])
        branch["branch_is_mixed"] = bool(state["mixed_actual_history"])
        if lineage_id in self.lineages:
            lineage = self.lineages[lineage_id]
            segment_ids = set(_ids(lineage.get("segment_ids")))
            segment_ids.add(segment_id)
            lineage["segment_ids"] = "|".join(sorted(segment_ids))
            lineage["event_count"] = int(lineage.get("event_count", 0)) + 1
        self.lineage_events.append(
            {
                **MARKERS,
                "actual_record_index": index,
                "field": field,
                "encoded_column": column,
                "target_neuron": neuron,
                "segment_provenance_id": segment_id,
                "actual_anchor_id": anchor_id,
                "actual_branch_provenance_id": branch_id,
                "parent_actual_anchor_ids": "|".join(parent_anchor_ids),
                "parent_actual_lineage_ids": "|".join(parent_lineage_ids),
                "actual_lineage_id": lineage_id,
                "lineage_class": lineage_class,
                "reinforced": reinforced,
                "future_data_used": False,
            }
        )
        return state

    def live_state_for_segment(self, registry: BranchProvenanceRegistry, segment: Segment) -> dict[str, object] | None:
        origin = registry.provenance_for(segment)
        return self.segments.get(origin.segment_provenance_id) if origin else None


def _state_ids(state: Mapping[str, object] | None, field_name: str) -> list[str]:
    if not state:
        return []
    return [str(value) for value in state.get(field_name, [])]


def capture_prematch(
    *,
    model: SequentialMemory,
    registry: BranchProvenanceRegistry,
    tracker: ActualBranchProvenanceTracker,
    code: SymbolCode,
    actual_record_index: int,
    timestamp: str,
    ranges: FieldColumnRanges,
    l_match: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], set[int]]:
    """Capture only history that existed before the current matching call."""
    event_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    preexisting_ids: set[int] = set()
    for event_index, event in enumerate(code.events):
        column = int(event.column)
        field = field_for_column(column, ranges)
        candidates: list[tuple[int, Segment, dict[str, object]]] = []
        for neuron, segment in _iter_column_segments(model, column):
            preexisting_ids.add(id(segment))
            origin = registry.provenance_for(segment)
            state = tracker.state(origin.segment_provenance_id) if origin else None
            if origin and state and _state_ids(state, "actual_anchor_ids_seen"):
                candidates.append((neuron, segment, state))
            origin_fields = {
                "segment_provenance_id": origin.segment_provenance_id if origin else "",
                "segment_creation_index": origin.creation_transition_index if origin else "",
            }
            segment_rows.append({
                **MARKERS,
                "actual_record_index": actual_record_index,
                "timestamp": timestamp,
                "field": field,
                "encoded_column": column,
                **origin_fields,
                "target_neuron": neuron,
                "existed_before_current_matching": True,
                "eligible_before_current_matching": "",
                "selected_by_current_matching": False,
                "reinforced_by_current_observation": False,
                "actual_anchor_ids_before": "|".join(_state_ids(state, "actual_anchor_ids_seen")),
                "actual_anchor_count_before": len(_state_ids(state, "actual_anchor_ids_seen")),
                "actual_branch_ids_before": "|".join(_state_ids(state, "actual_branch_ids_seen")),
                "actual_branch_count_before": len(_state_ids(state, "actual_branch_ids_seen")),
                "mixed_actual_history_before": bool(state and state.get("mixed_actual_history")),
                "mixed_history_class_before": state.get("mixed_history_class", "NO_ACTUAL_HISTORY") if state else "NO_ACTUAL_HISTORY",
                "first_mixed_index_before": state.get("first_mixed_index", "") if state else "",
                "event_that_created_mixture_before": state.get("event_that_created_mixture", "") if state else "",
                "mixture_trigger_before": state.get("mixture_trigger", "") if state else "",
                "capture_phase": "PRE_MATCHING",
                "matching_started_at_capture": False,
                "selected_segment_known_at_capture": False,
                "reinforcement_started_at_capture": False,
                "actual_anchor_ids_after": "",
                "actual_anchor_count_after": "",
                "mixed_actual_history_after": "",
                "current_overlap": "",
                "current_candidate_score": "",
                "l2_only_match": "",
                "branch_continuation_candidate": False,
                "branch_switch_candidate": False,
                "_segment_object_id": id(segment),
            })
        anchor_ids = sorted({anchor for _, _, state in candidates for anchor in _state_ids(state, "actual_anchor_ids_seen")})
        branch_ids = sorted({str(item.get("actual_branch_provenance_id", "")) for anchor in anchor_ids for item in tracker.anchors.values() if item.get("actual_anchor_id") == anchor})
        event_rows.append({
            **MARKERS,
            "actual_record_index": actual_record_index,
            "timestamp": timestamp,
            "field": field,
            "encoded_column": column,
            "observe_scenario": "",
            "L_match": l_match,
            "prematch_capture_confirmed": True,
            "current_matching_not_started": True,
            "current_selected_segment_unavailable": True,
            "current_reinforcement_not_started": True,
            "capture_phase": "PRE_MATCHING",
            "matching_started_at_capture": False,
            "selected_segment_known_at_capture": False,
            "reinforcement_started_at_capture": False,
            "prematch_actual_anchor_count": len(anchor_ids),
            "prematch_unique_actual_anchor": len(anchor_ids) == 1,
            "prematch_actual_anchor_ids": "|".join(anchor_ids),
            "prematch_branch_ids": "|".join(branch_ids),
            "prematch_branch_status": "unique" if len(branch_ids) == 1 else "multiple" if branch_ids else "unresolved",
            "selected_segment_provenance_id": "",
            "selected_target_neuron": "",
            "selected_overlap": "",
            "selected_candidate_score": "",
            "matching_segment_count": "",
            "matching_neuron_count": "",
            "ambiguous_segment": len(anchor_ids) > 1,
            "ambiguity_definition": "distinct_actual_anchor_ids_before",
            "ambiguity_stage": "PRE_THRESHOLD",
            "ambiguous_neuron": "",
            "l2_only_match": "",
            "passes_L2": "",
            "passes_L3": "",
            "passes_L4": "",
            "selected_actual_anchor_ids": "",
            "selected_actual_anchor_count": "",
            "selected_actual_branch_ids": "",
            "selected_actual_branch_count": "",
            "selected_branch_is_unique": "",
            "selected_branch_is_mixed": "",
            "branch_continuity_status": "",
            "unique_branch_continuation": False,
            "branch_switch": False,
            "multiple_history_compatible": False,
            "provenance_chain_broken": False,
            "reinforced": False,
            "reinforcement_count_before": "",
            "reinforcement_count_after": "",
            "actual_reinforcement_count_before": "",
            "actual_reinforcement_count_after": "",
            "segment_became_mixed_after_reinforcement": False,
            "new_actual_anchor_added": False,
            "prematch_predicted_reference_available": bool(model.last_prediction_candidates.get(column)),
            "prematch_predicted_reference_compatible": "",
            "actual_history_reference_available": bool(anchor_ids),
            "actual_history_reference_compatible": "",
            "selected_again_within_1": "",
            "selected_again_within_3": "",
            "selected_again_within_5": "",
            "reinforced_again_within_1": "",
            "reinforced_again_within_3": "",
            "reinforced_again_within_5": "",
            "branch_seen_again_within_1": "",
            "branch_seen_again_within_3": "",
            "branch_seen_again_within_5": "",
            "downstream_step1_error": "",
            "downstream_step2_error": "",
            "downstream_step3_error": "",
            "downstream_step4_error": "",
            "downstream_step5_error": "",
            "prematch_selected_segment_unavailable": True,
            "post_observation_selected_segment_known": False,
            "_event_index": event_index,
        })
    return event_rows, segment_rows, preexisting_ids


def _branch_ids(tracker: ActualBranchProvenanceTracker, state: Mapping[str, object] | None) -> list[str]:
    anchors = _state_ids(state, "actual_anchor_ids_seen")
    return sorted({str(item.get("actual_branch_provenance_id")) for item in tracker.anchors.values() if item.get("actual_anchor_id") in anchors})


def join_after_observation(
    *,
    event_rows: list[dict[str, object]],
    segment_rows: list[dict[str, object]],
    trace: ObservationTrace,
    registry: BranchProvenanceRegistry,
    tracker: ActualBranchProvenanceTracker,
    preexisting_ids: set[int],
    actual_record_index: int,
    source_cells: Mapping[int, float],
) -> None:
    events = {event.target_column: event for event in trace.events}
    selected_ids: set[str] = set()
    reinforced_ids: set[str] = set()
    for row in event_rows:
        event = events.get(int(row["encoded_column"]))
        if event is None:
            continue
        selected = event.selected_segment or event.best_matching_segment
        reinforced = event.reinforced_segment
        selected_origin = registry.provenance_for(selected) if selected else None
        selected_state = tracker.state(selected_origin.segment_provenance_id) if selected_origin else None
        before_ids = set(str(row["prematch_actual_anchor_ids"]).split("|")) - {""}
        selected_anchor_ids = _state_ids(selected_state, "actual_anchor_ids_seen")
        selected_branch_ids = _branch_ids(tracker, selected_state)
        if selected is None:
            status = "POST_OBSERVATION_BRANCH_INVALID"
        elif selected_state and selected_state.get("mixed_actual_history"):
            status = "ACTUAL_BRANCH_MIXED_HISTORY"
        elif before_ids and selected_anchor_ids and set(selected_anchor_ids) == before_ids and len(before_ids) == 1:
            status = "UNIQUE_ACTUAL_BRANCH_CONTINUATION"
        elif before_ids and selected_anchor_ids and not (set(selected_anchor_ids) & before_ids):
            status = "ACTUAL_BRANCH_SWITCH"
        elif len(before_ids) > 1 and set(selected_anchor_ids) & before_ids:
            status = "MULTIPLE_ACTUAL_BRANCH_COMPATIBLE"
        elif not before_ids or not selected_anchor_ids:
            status = "ACTUAL_BRANCH_NOT_FOUND"
        else:
            status = "ACTUAL_BRANCH_CHAIN_BROKEN"
        row.update({
            "observe_scenario": event.scenario,
            # These fields describe the capture point and must not be rewritten
            # with post-observation values.  Post state has separate fields.
            "post_observation_selected_segment_known": selected is not None,
            "post_observation_matching_started": True,
            "post_observation_reinforcement_started": bool(reinforced),
            "selected_segment_provenance_id": selected_origin.segment_provenance_id if selected_origin else "",
            "selected_target_neuron": event.winner_neuron,
            "selected_overlap": event.best_matching_overlap,
            "selected_candidate_score": event.best_matching_score,
            "matching_segment_count": event.matching_segment_count,
            "matching_neuron_count": len(event.predictive_neuron_ids),
            "selected_actual_anchor_ids": "|".join(selected_anchor_ids),
            "selected_actual_anchor_count": len(selected_anchor_ids),
            "selected_actual_branch_ids": "|".join(selected_branch_ids),
            "selected_actual_branch_count": len(selected_branch_ids),
            "selected_branch_is_unique": len(selected_branch_ids) == 1,
            "selected_branch_is_mixed": bool(selected_state and selected_state.get("mixed_actual_history")),
            "branch_continuity_status": status,
            "unique_branch_continuation": status == "UNIQUE_ACTUAL_BRANCH_CONTINUATION",
            "branch_switch": status == "ACTUAL_BRANCH_SWITCH",
            "multiple_history_compatible": status == "MULTIPLE_ACTUAL_BRANCH_COMPATIBLE",
            "provenance_chain_broken": status == "ACTUAL_BRANCH_CHAIN_BROKEN",
            "reinforced": bool(reinforced),
            "actual_history_reference_compatible": bool(before_ids & set(selected_anchor_ids)),
            "prematch_predicted_reference_compatible": "",
        })
        if selected_origin:
            selected_ids.add(selected_origin.segment_provenance_id)
        reinforced_origin = registry.provenance_for(reinforced) if reinforced else None
        if reinforced_origin:
            reinforced_ids.add(reinforced_origin.segment_provenance_id)
    for row in segment_rows:
        sid = str(row.get("segment_provenance_id", ""))
        selected = sid in selected_ids if sid else False
        reinforced = sid in reinforced_ids if sid else False
        row["selected_by_current_matching"] = selected
        row["reinforced_by_current_observation"] = reinforced
        row["l2_only_match"] = ""  # Matching-derived detail is joined only when a matching trace supplies it.
        row["branch_continuation_candidate"] = selected and bool(row.get("actual_anchor_count_before"))
        row["branch_switch_candidate"] = selected and not bool(row.get("actual_anchor_count_before"))
        state = tracker.state(sid) if sid else None
        row["actual_anchor_ids_after"] = "|".join(_state_ids(state, "actual_anchor_ids_seen"))
        row["actual_anchor_count_after"] = len(_state_ids(state, "actual_anchor_ids_seen"))
        row["actual_branch_ids_after"] = "|".join(_state_ids(state, "actual_branch_ids_seen"))
        row["actual_branch_count_after"] = len(_state_ids(state, "actual_branch_ids_seen"))
        row["mixed_actual_history_after"] = bool(state and state.get("mixed_actual_history"))
        row["mixed_history_class_after"] = state.get("mixed_history_class", "NO_ACTUAL_HISTORY") if state else "NO_ACTUAL_HISTORY"
        row["first_mixed_index_after"] = state.get("first_mixed_index", "") if state else ""
        row["event_that_created_mixture_after"] = state.get("event_that_created_mixture", "") if state else ""
        row["mixture_trigger_after"] = state.get("mixture_trigger", "") if state else ""


def record_actual_transition(
    tracker: ActualBranchProvenanceTracker,
    *,
    trace: ObservationTrace,
    registry: BranchProvenanceRegistry,
    preexisting_ids: set[int],
    actual_record_index: int,
    ranges: FieldColumnRanges,
    source_cells: Mapping[int, float],
) -> None:
    for event in trace.events:
        segment = event.selected_segment or event.reinforced_segment
        if segment is None or id(segment) not in preexisting_ids:
            continue
        origin = registry.provenance_for(segment)
        if origin is None:
            continue
        tracker.record_actual(
            segment_id=origin.segment_provenance_id,
            index=actual_record_index,
            field=field_for_column(event.target_column, ranges),
            column=event.target_column,
            neuron=event.winner_neuron,
            source_cells=source_cells,
            reinforced=event.reinforced_segment is not None,
            parent_anchor_ids=_state_ids(
                tracker.state(origin.segment_provenance_id),
                "actual_anchor_ids_seen",
            ),
        )


def write_rows(path: Path, rows: list[dict[str, object]], compress: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    destination = path.with_suffix(path.suffix + ".gz") if compress else path
    opener = gzip.open if compress else open
    clean = [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]
    with opener(destination, "wt", encoding="utf-8", newline="") as handle:
        if clean:
            # Branch and anchor records have different fields.  Building the
            # schema from only the first branch row silently discarded all
            # anchor-only lineage fields from the historical CSV.
            fieldnames: list[str] = []
            for row in clean:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(clean)
    return destination


def write_registry(path: Path, tracker: ActualBranchProvenanceTracker, compress: bool) -> Path:
    rows = list(tracker.branches.values()) + list(tracker.anchors.values())
    return write_rows(path, rows, compress)


def write_lineage_registry(path: Path, tracker: ActualBranchProvenanceTracker, compress: bool) -> Path:
    return write_rows(path, list(tracker.lineages.values()), compress)
