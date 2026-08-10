"""Read-only Fig.9 ambiguity and segment-reuse trace projections.

The helpers in this module consume traces that the model has already produced.
They do not inspect future records, choose candidates, or mutate model state.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from experiments.diagnostics.fig9_candidate_score_trace import field_for_column
from experiments.diagnostics.fig9_oracle_candidate import FieldColumnRanges
from seqmem.model import ObservationTrace


DIAGNOSTIC_MARKERS = {
    "diagnostic_only": True,
    "offline_analysis_only": True,
    "ground_truth_does_not_affect_model": True,
    "selection_behavior_changed": False,
    "learning_behavior_changed": False,
    "uses_compensation": False,
    "uses_future_covariates": False,
}


@dataclass
class AmbiguityReuseState:
    """Small serializable accumulator for actual-history reuse diagnostics."""

    counts: dict[str, int] = field(default_factory=dict)
    contexts: dict[str, set[str]] = field(default_factory=dict)
    record_indices: dict[str, list[int]] = field(default_factory=dict)
    context_history: dict[str, list[tuple[int, str]]] = field(default_factory=dict)

    def annotate(self, rows: Sequence[dict[str, object]]) -> None:
        for row in rows:
            if row.get("trajectory_kind") != "actual_observation":
                continue
            segment_id = str(row.get("candidate_segment_id", ""))
            if not segment_id:
                continue
            record_index = int(row.get("actual_record_index", 0))
            context = str(row.get("context_signature", ""))
            prior_indices = self.record_indices.setdefault(segment_id, [])
            count = self.counts.get(segment_id, 0) + 1
            context_set = self.contexts.setdefault(segment_id, set())
            context_set.add(context)
            prior_indices.append(record_index)
            history = self.context_history.setdefault(segment_id, [])
            history.append((record_index, context))
            row["candidate_match_count_running"] = count
            row["unique_context_count_running"] = len(context_set)
            row["recent_50_match_count_running"] = sum(
                record_index - 49 <= value <= record_index for value in prior_indices
            )
            row["recent_50_unique_context_count_running"] = len(
                {
                    value
                    for history_index, value in history
                    if record_index - 49 <= history_index <= record_index
                }
            )
            self.counts[segment_id] = count

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "counts": dict(self.counts),
            "contexts": {
                key: sorted(values) for key, values in self.contexts.items()
            },
            "record_indices": {
                key: list(values) for key, values in self.record_indices.items()
            },
            "context_history": {
                key: [[index, value] for index, value in values]
                for key, values in self.context_history.items()
            },
        }

    @classmethod
    def from_checkpoint_payload(cls, payload: object) -> "AmbiguityReuseState":
        if not isinstance(payload, Mapping):
            return cls()
        return cls(
            counts={str(key): int(value) for key, value in dict(payload.get("counts", {})).items()},
            contexts={
                str(key): {str(item) for item in value}
                for key, value in dict(payload.get("contexts", {})).items()
            },
            record_indices={
                str(key): [int(item) for item in value]
                for key, value in dict(payload.get("record_indices", {})).items()
            },
            context_history={
                str(key): [(int(item[0]), str(item[1])) for item in value]
                for key, value in dict(payload.get("context_history", {})).items()
            },
        )


def stable_context_signature(active_sources: Mapping[int, float]) -> str:
    """Hash the causal source-cell/time set available at this transition."""

    payload = [
        (int(source), round(float(source_time), 12))
        for source, source_time in sorted(active_sources.items())
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def score_summary(scores: Sequence[float]) -> dict[str, float | int]:
    """Return descriptive top-score separation and entropy statistics."""

    finite = sorted(
        (float(value) for value in scores if math.isfinite(float(value))),
        reverse=True,
    )
    top1 = finite[0] if finite else math.nan
    top2 = finite[1] if len(finite) > 1 else math.nan
    margin = top1 - top2 if len(finite) > 1 else math.nan
    denominator = max(abs(top1), abs(top2)) if len(finite) > 1 else math.nan
    normalized = margin / (denominator + 1e-12) if len(finite) > 1 else math.nan
    if finite:
        shifted = [math.exp(value - finite[0]) for value in finite]
        normalizer = sum(shifted)
        probabilities = [value / normalizer for value in shifted]
        entropy = -sum(
            probability * math.log(probability)
            for probability in probabilities
            if probability > 0.0
        )
    else:
        entropy = math.nan
    return {
        "local_top1_score": top1,
        "local_top2_score": top2,
        "local_score_margin": margin,
        "normalized_score_margin": normalized,
        "candidate_score_entropy": entropy,
        "top_score_tie_count": sum(value == top1 for value in finite) if finite else 0,
    }


def _segment_id(registry: Any, segment: Any) -> str:
    origin = registry.provenance_for(segment)
    if origin is not None:
        return str(origin.segment_provenance_id)
    diagnostic_id = getattr(segment, "diagnostic_id", None)
    if diagnostic_id is None:
        return ""
    return f"diagnostic:{diagnostic_id}"


def _segment_age(segment: Any) -> int:
    return max((int(synapse.age) for synapse in segment.synapses.values()), default=0)


def _reinforcement_count(segment: Any) -> int:
    return int(
        getattr(segment, "scenario1_reinforcements", 0)
        + getattr(segment, "scenario2_reinforcements", 0)
    )


def _match_class(overlap: int) -> str:
    if overlap == 2:
        return "L2_ONLY_STRICT"
    if overlap == 3:
        return "L3_ONLY_VS_L4"
    if overlap >= 4:
        return "L4_OR_ABOVE"
    return "BELOW_L2"


def build_actual_observation_rows(
    *,
    observation_trace: ObservationTrace,
    registry: Any,
    active_sources: Mapping[int, float],
    actual_record_index: int,
    input_timestamp: str,
    ranges: FieldColumnRanges,
) -> list[dict[str, object]]:
    """Project actual matching candidates into one row per candidate event."""

    context_signature = stable_context_signature(active_sources)
    rows: list[dict[str, object]] = []
    for event_number, event in enumerate(observation_trace.events):
        candidates = list(event.matching_candidates)
        segment_ids = {
            _segment_id(registry, candidate.segment) for candidate in candidates
        }
        neuron_ids = {int(candidate.target_neuron) for candidate in candidates}
        scores = [float(candidate.candidate_score) for candidate in candidates]
        score_stats = score_summary(scores)
        selected_id = (
            _segment_id(registry, event.selected_segment)
            if event.selected_segment is not None
            else ""
        )
        selected_overlap = int(event.best_matching_overlap)
        candidates_or_empty = candidates or [None]
        for candidate_number, candidate in enumerate(candidates_or_empty):
            candidate_id = (
                _segment_id(registry, candidate.segment)
                if candidate is not None
                else ""
            )
            segment = candidate.segment if candidate is not None else None
            overlap = int(candidate.timed_overlap) if candidate is not None else 0
            candidate_score = (
                float(candidate.candidate_score) if candidate is not None else math.nan
            )
            rows.append(
                {
                    **DIAGNOSTIC_MARKERS,
                    "trajectory_kind": "actual_observation",
                    "actual_record_index": actual_record_index,
                    "input_index": actual_record_index + 1,
                    "input_timestamp": input_timestamp,
                    "horizon_step": 0,
                    "event_number": event_number,
                    "candidate_number": candidate_number if candidate is not None else "",
                    "field": field_for_column(event.target_column, ranges),
                    "column": event.target_column,
                    "scenario": event.scenario,
                    "candidate_segment_id": candidate_id,
                    "selected_segment_id": selected_id,
                    "candidate_neuron": (
                        candidate.target_neuron if candidate is not None else ""
                    ),
                    "candidate_score": candidate_score,
                    "candidate_overlap": overlap,
                    "selected_candidate": candidate_id != "" and candidate_id == selected_id,
                    "reinforced_segment": candidate_id != "" and candidate_id == selected_id,
                    "matching_segment_count": len(segment_ids),
                    "matching_neuron_count": len(neuron_ids),
                    "matching_column_count": 1 if candidates else 0,
                    "within_column_candidate_segment_count": len(segment_ids),
                    "within_column_candidate_neuron_count": len(neuron_ids),
                    **score_stats,
                    "selected_segment_reinforcement_count": (
                        _reinforcement_count(event.selected_segment)
                        if event.selected_segment is not None
                        else ""
                    ),
                    "candidate_segment_reinforcement_count": (
                        _reinforcement_count(segment) if segment is not None else ""
                    ),
                    "segment_age": _segment_age(segment) if segment is not None else "",
                    "context_signature": context_signature,
                    "context_source_count": len(active_sources),
                    "l2_match_class": _match_class(overlap),
                    "l2_relaxed_vs_l4": overlap in (2, 3),
                    "selected_overlap": selected_overlap,
                    "candidate_match_count_running": "",
                    "unique_context_count_running": "",
                    "recent_50_match_count_running": "",
                    "recent_50_unique_context_count_running": "",
                }
            )
    return rows


def build_autonomous_rollout_rows(
    *,
    selection_rows: Sequence[Mapping[str, object]],
    actual_record_index: int,
    input_timestamp: str,
) -> list[dict[str, object]]:
    """Project same-call intracolumn selection rows as rollout diagnostics."""

    output: list[dict[str, object]] = []
    for source in selection_rows:
        row = {
            **DIAGNOSTIC_MARKERS,
            "trajectory_kind": "autonomous_rollout",
            "actual_record_index": actual_record_index,
            "input_index": source.get("input_index", actual_record_index + 1),
            "input_timestamp": input_timestamp,
            "horizon_step": source.get("horizon_step", ""),
            "event_number": "",
            "candidate_number": "",
            "field": source.get("field", ""),
            "column": source.get("column", ""),
            "scenario": "",
            "candidate_segment_id": source.get("selected_segment_id", ""),
            "selected_segment_id": source.get("selected_segment_id", ""),
            "candidate_neuron": source.get("selected_neuron", ""),
            "candidate_score": source.get("selected_candidate_score", ""),
            "candidate_overlap": "",
            "selected_candidate": True,
            "reinforced_segment": False,
            "matching_segment_count": "",
            "matching_neuron_count": "",
            "matching_column_count": "",
            "within_column_candidate_segment_count": source.get(
                "within_column_candidate_segment_count",
                source.get("group_candidate_count", ""),
            ),
            "within_column_candidate_neuron_count": source.get(
                "within_column_candidate_neuron_count", ""
            ),
            "local_top1_score": source.get("local_top1_score", ""),
            "local_top2_score": source.get("local_top2_score", ""),
            "local_score_margin": source.get("local_score_margin", ""),
            "normalized_score_margin": source.get("normalized_score_margin", ""),
            "candidate_score_entropy": source.get("candidate_score_entropy", ""),
            "top_score_tie_count": source.get("top_score_tie_count", ""),
            "selected_segment_reinforcement_count": "",
            "candidate_segment_reinforcement_count": "",
            "segment_age": "",
            "context_signature": source.get("context_signature", ""),
            "context_source_count": source.get("context_source_count", ""),
            "l2_match_class": "",
            "l2_relaxed_vs_l4": "",
            "selected_overlap": "",
            "candidate_match_count_running": "",
            "unique_context_count_running": "",
            "recent_50_match_count_running": "",
            "recent_50_unique_context_count_running": "",
        }
        output.append(row)
    return output


def protocol(fingerprint: Mapping[str, object]) -> dict[str, object]:
    return {
        **DIAGNOSTIC_MARKERS,
        "version": "fig9-ambiguity-v1",
        "trace_unit": "one actual candidate row or one autonomous column row",
        "trace_complexity": "O(candidate)",
        "context_signature": "SHA256(sorted causal active source-cell/time pairs)",
        "strict_protocol_sha256": hashlib.sha256(
            json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
    }
