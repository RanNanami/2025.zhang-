"""Independent, read-only reference provenance for strict Fig.9 observations.

The capture side deliberately runs before the current observation is matched.
Only stable provenance fields are serialized; Python object identities are used
internally for the post-observation join and never leave this module.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from experiments.diagnostics.fig9_branch_provenance import (
    BranchProvenanceRegistry,
    SegmentOrigin,
)
from experiments.diagnostics.fig9_candidate_score_trace import field_for_column
from experiments.diagnostics.fig9_oracle_candidate import FieldColumnRanges
from seqmem.encoding import SymbolCode
from seqmem.model import ObservationTrace, Segment, SequentialMemory


MARKERS = {
    "diagnostic_only": True,
    "offline_analysis_only": True,
    "reference_captured_before_current_matching": True,
    "reference_does_not_affect_selection": True,
    "reference_does_not_affect_reinforcement": True,
    "reference_does_not_affect_prediction": True,
    "ground_truth_used_for_analysis_only": True,
    "uses_compensation": False,
}
REFERENCE_STATUSES = {
    "STRICT_INDEPENDENT_REFERENCE",
    "PREMATCH_REFERENCE_AMBIGUOUS",
    "MULTIPLE_INDEPENDENT_REFERENCES",
    "REFERENCE_NOT_FOUND",
    "CIRCULAR_REFERENCE_INVALID",
    "POST_OBSERVATION_REFERENCE_INVALID",
    "PROVENANCE_CHAIN_BROKEN",
    "REFERENCE_KEY_UNAVAILABLE",
}


def _stable_branch_id(origin: SegmentOrigin) -> str:
    payload = (
        origin.segment_provenance_id,
        origin.creation_transition_index,
        origin.target_column,
        origin.target_neuron,
        origin.creation_source_fingerprint,
    )
    return hashlib.sha256(repr(payload).encode("ascii")).hexdigest()


def _iter_column_segments(model: SequentialMemory, column: int):
    if column < 0 or column >= len(model.columns):
        return
    for neuron_index, neuron in enumerate(model.columns[column].neurons):
        for segment in neuron.segments:
            yield neuron_index, segment


def _origin_fields(origin: SegmentOrigin | None) -> dict[str, object]:
    if origin is None:
        return {
            "segment_provenance_id": "",
            "branch_provenance_id": "",
            "creation_transition_index": "",
            "creation_target_neuron": "",
            "creation_source_fingerprint": "",
        }
    return {
        "segment_provenance_id": origin.segment_provenance_id,
        "branch_provenance_id": _stable_branch_id(origin),
        "creation_transition_index": origin.creation_transition_index,
        "creation_target_neuron": origin.target_neuron,
        "creation_source_fingerprint": origin.creation_source_fingerprint,
    }


@dataclass
class IndependentReferenceTracker:
    """Stable history of actual observations, never autonomous rollout."""

    by_column: dict[int, list[dict[str, object]]] = field(default_factory=dict)

    def record_actual(
        self,
        *,
        actual_record_index: int,
        target_column: int,
        segment: Segment | None,
        registry: BranchProvenanceRegistry,
        preexisting_ids: set[int],
    ) -> None:
        if segment is None or id(segment) not in preexisting_ids:
            return
        origin = registry.provenance_for(segment)
        if origin is None:
            return
        value = {
            **_origin_fields(origin),
            "actual_record_index": actual_record_index,
            "target_column": target_column,
            "segment_object_id": id(segment),
        }
        values = self.by_column.setdefault(target_column, [])
        if not any(
            item["segment_provenance_id"] == value["segment_provenance_id"]
            for item in values
        ):
            values.append(value)

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "version": "fig9-independent-reference-v1",
            "by_column": {
                str(column): [
                    {key: value for key, value in row.items() if key != "segment_object_id"}
                    for row in rows
                ]
                for column, rows in self.by_column.items()
            },
        }

    @classmethod
    def from_checkpoint_payload(cls, payload: Mapping[str, object] | None):
        tracker = cls()
        if not payload:
            return tracker
        raw = payload.get("by_column", {})
        if isinstance(raw, Mapping):
            for column, rows in raw.items():
                if isinstance(rows, list):
                    tracker.by_column[int(column)] = [dict(row) for row in rows if isinstance(row, Mapping)]
        return tracker


def _reference_rows(
    *,
    model: SequentialMemory,
    registry: BranchProvenanceRegistry,
    tracker: IndependentReferenceTracker,
    code: SymbolCode,
    actual_record_index: int,
    ranges: FieldColumnRanges,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[int, list[dict[str, object]]]]:
    """Capture references before current matching and return private join data."""
    event_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    private: dict[int, list[dict[str, object]]] = {}
    for event_index, event in enumerate(code.events):
        column = int(event.column)
        candidates = []
        for candidate in model.last_prediction_candidates.get(column, ()):
            origin = registry.provenance_for(candidate.segment)
            if origin is not None:
                candidates.append({
                    **_origin_fields(origin),
                    "tier": "TIER_A_PREMATCH_PREDICTED_SEGMENT_REFERENCE",
                    "source": "last_prediction_candidates",
                    "segment_object_id": id(candidate.segment),
                })
        history = tracker.by_column.get(column, [])
        for item in history:
            candidates.append({
                **item,
                "tier": "TIER_B_PREVIOUS_ACTUAL_BRANCH_REFERENCE",
                "source": "previous_actual_observation",
                "segment_object_id": "",
            })
        unique = {item["segment_provenance_id"]: item for item in candidates}
        if len(unique) == 1:
            status = "STRICT_INDEPENDENT_REFERENCE"
        elif len(unique) > 1:
            status = "PREMATCH_REFERENCE_AMBIGUOUS" if candidates and all(
                item["tier"].startswith("TIER_A") for item in candidates
            ) else "MULTIPLE_INDEPENDENT_REFERENCES"
        else:
            status = "REFERENCE_NOT_FOUND"
        row = {
            **MARKERS,
            "actual_record_index": actual_record_index,
            "event_index": event_index,
            "observed_column": column,
            "observed_time": event.time,
            "source_field": field_for_column(column, ranges),
            "reference_status": status,
            "reference_count": len(unique),
            "reference_tiers": "|".join(sorted({str(item["tier"]) for item in unique.values()})),
            "reference_segment_provenance_ids": "|".join(sorted(unique)),
            "reference_branch_provenance_ids": "|".join(sorted(str(item["branch_provenance_id"]) for item in unique.values())),
            "reference_captured_before_current_matching": True,
            "current_matching_selected": False,
            "current_matching_score": "",
            "current_reinforced": False,
            "reference_compatible": "",
            "future_selected_1": "",
            "future_selected_3": "",
            "future_selected_5": "",
            "downstream_error_step_1": "",
            "downstream_error_step_2": "",
        }
        event_rows.append(row)
        private[event_index] = candidates
        for neuron_index, segment in _iter_column_segments(model, column):
            origin = registry.provenance_for(segment)
            segment_rows.append({
                **MARKERS,
                "actual_record_index": actual_record_index,
                "event_index": event_index,
                "observed_column": column,
                "source_field": field_for_column(column, ranges),
                **_origin_fields(origin),
                "target_neuron": neuron_index,
                "existed_before_current_matching": True,
                "prematch_predicted": any(item.get("segment_object_id") == id(segment) for item in candidates),
                "part_of_previous_actual_branch": any(item.get("segment_provenance_id") == (origin.segment_provenance_id if origin else None) for item in history),
                "selected_by_current_matching": False,
                "reinforced_by_current_observation": False,
                "current_overlap": "",
                "current_candidate_score": "",
                "reference_compatible": "",
                "posthoc_future_compatible": "",
            })
    return event_rows, segment_rows, private


def capture_before_matching(*args, **kwargs):
    return _reference_rows(*args, **kwargs)


def join_after_observation(
    event_rows: list[dict[str, object]],
    segment_rows: list[dict[str, object]],
    *,
    trace: ObservationTrace,
    registry: BranchProvenanceRegistry,
    private: dict[int, list[dict[str, object]]],
    actual_record_index: int,
) -> None:
    events = {event.target_column: event for event in trace.events}
    selected_ids: set[str] = set()
    reinforced_ids: set[str] = set()
    for event_index, row in enumerate(event_rows):
        event = events.get(int(row["observed_column"]))
        if event is None:
            continue
        selected = event.selected_segment or event.best_matching_segment
        reinforced = event.reinforced_segment
        selected_origin = registry.provenance_for(selected) if selected else None
        reinforced_origin = registry.provenance_for(reinforced) if reinforced else None
        selected_id = selected_origin.segment_provenance_id if selected_origin else ""
        reinforced_id = reinforced_origin.segment_provenance_id if reinforced_origin else ""
        row.update({
            "current_matching_selected": bool(selected),
            "current_matching_score": event.best_matching_score,
            "current_reinforced": bool(reinforced),
            "reference_compatible": selected_id in str(row["reference_segment_provenance_ids"]).split("|") if selected_id else False,
        })
        selected_ids.add(selected_id)
        reinforced_ids.add(reinforced_id)
    for row in segment_rows:
        sid = str(row["segment_provenance_id"])
        row["selected_by_current_matching"] = sid in selected_ids if sid else False
        row["reinforced_by_current_observation"] = sid in reinforced_ids if sid else False


def record_actual_history(
    tracker: IndependentReferenceTracker,
    *,
    trace: ObservationTrace,
    registry: BranchProvenanceRegistry,
    preexisting_ids: set[int],
    actual_record_index: int,
) -> None:
    for event in trace.events:
        tracker.record_actual(
            actual_record_index=actual_record_index,
            target_column=event.target_column,
            segment=event.selected_segment or event.reinforced_segment,
            registry=registry,
            preexisting_ids=preexisting_ids,
        )


def write_trace(path: Path, rows: list[dict[str, object]], compress: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    actual = path.with_suffix(path.suffix + ".gz") if compress else path
    opener = gzip.open if compress else open
    with opener(actual, "wt", encoding="utf-8", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return actual


def protocol(fingerprint: Mapping[str, object], level: str, compress: bool) -> dict[str, object]:
    return {
        **MARKERS,
        "version": "fig9-independent-reference-v1",
        "level": level,
        "compressed": compress,
        "strict_protocol_sha256": hashlib.sha256(json.dumps(fingerprint, sort_keys=True, default=str).encode()).hexdigest(),
        "reference_tiers": [
            "TIER_A_PREMATCH_PREDICTED_SEGMENT_REFERENCE",
            "TIER_B_PREVIOUS_ACTUAL_BRANCH_REFERENCE",
            "TIER_C_PREEXISTING_PROVENANCE_CHAIN_REFERENCE",
            "TIER_D_POSTHOC_FUTURE_COMPATIBILITY",
        ],
        "forbidden_reference_inputs": ["current winner", "current reinforcement", "current score", "ground-truth suffix"],
    }
