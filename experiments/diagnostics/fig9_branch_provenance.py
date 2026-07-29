"""Read-only candidate-to-segment provenance capture for Fig.9 diagnostics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

from experiments.diagnostics.fig9_candidate_score_trace import field_for_column
from experiments.diagnostics.fig9_competitive_inhibition import (
    CompetitionDecision,
    CompetitionResult,
)
from experiments.diagnostics.fig9_oracle_candidate import FieldColumnRanges
from seqmem.encoding import SymbolCode
from seqmem.model import (
    PredictionCandidate,
    Segment,
    SequentialMemory,
    SynapsePSPContribution,
)


DIAGNOSTIC_MARKERS = {
    "diagnostic_only": True,
    "offline_analysis_only": True,
    "capture_branch_provenance": True,
    "uses_ground_truth_for_analysis_only": True,
    "ground_truth_does_not_affect_prediction": True,
    "branch_metrics_do_not_affect_prediction": True,
}

BRANCH_LEVELS = {"summary", "candidate", "full"}
SOURCE_SUPPORT_CLASSES = {
    "predicted-supported",
    "burst-assisted",
    "burst-only",
    "unavailable",
}


def stable_source_fingerprint(source_cells: Iterable[int]) -> str:
    """Hash a source-cell set without depending on input iteration order."""

    payload = ",".join(str(value) for value in sorted(set(source_cells)))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def stable_segment_provenance_id(
    *,
    creation_transition_index: int,
    target_column: int,
    target_neuron: int,
    creation_source_fingerprint: str,
    stable_creation_ordinal: int,
) -> str:
    """Build a process-independent ID from immutable creation metadata."""

    payload = {
        "creation_transition_index": creation_transition_index,
        "target_column": target_column,
        "target_neuron": target_neuron,
        "creation_source_fingerprint": creation_source_fingerprint,
        "stable_creation_ordinal": stable_creation_ordinal,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SegmentOrigin:
    segment_provenance_id: str
    creation_transition_index: int
    target_column: int
    target_neuron: int
    creation_target_time: float | None
    creation_source_cell_ids: tuple[int, ...]
    creation_source_fingerprint: str
    stable_creation_ordinal: int


def _iter_segments(
    model: SequentialMemory,
) -> Iterable[tuple[int, int, Segment]]:
    for column_index, column in enumerate(model.columns):
        for neuron_index, neuron in enumerate(column.neurons):
            for segment in neuron.segments:
                yield column_index, neuron_index, segment


def _current_segment_signature(
    column: int,
    neuron: int,
    segment: Segment,
) -> str:
    """Checkpoint rebinding signature; never used as the persistent ID."""

    payload = (
        column,
        neuron,
        segment.target_time,
        tuple(
            sorted(
                (source, synapse.delay)
                for source, synapse in segment.synapses.items()
            )
        ),
    )
    return hashlib.sha256(repr(payload).encode("ascii")).hexdigest()


class BranchProvenanceRegistry:
    """Experiment-side creation registry that does not mutate model objects."""

    def __init__(self) -> None:
        self._by_object: dict[int, SegmentOrigin] = {}
        self.current_predicted_sources: set[int] = set()
        self.current_burst_sources: set[int] = set()

    def segment_object_ids(self, model: SequentialMemory) -> set[int]:
        return {id(segment) for _column, _neuron, segment in _iter_segments(model)}

    def capture_new_segments(
        self,
        model: SequentialMemory,
        *,
        previous_segment_ids: set[int],
        creation_transition_index: int,
        creation_sources: Mapping[int, float],
    ) -> list[SegmentOrigin]:
        """Record segments created by one already-completed learning call."""

        source_ids = tuple(sorted(creation_sources))
        source_fingerprint = stable_source_fingerprint(source_ids)
        created = [
            (column, neuron, segment)
            for column, neuron, segment in _iter_segments(model)
            if id(segment) not in previous_segment_ids
        ]
        origins: list[SegmentOrigin] = []
        for ordinal, (column, neuron, segment) in enumerate(created):
            origin = SegmentOrigin(
                segment_provenance_id=stable_segment_provenance_id(
                    creation_transition_index=creation_transition_index,
                    target_column=column,
                    target_neuron=neuron,
                    creation_source_fingerprint=source_fingerprint,
                    stable_creation_ordinal=ordinal,
                ),
                creation_transition_index=creation_transition_index,
                target_column=column,
                target_neuron=neuron,
                creation_target_time=(
                    segment.target_time - model.params.cycle_period
                    if segment.target_time is not None
                    else None
                ),
                creation_source_cell_ids=source_ids,
                creation_source_fingerprint=source_fingerprint,
                stable_creation_ordinal=ordinal,
            )
            self._by_object[id(segment)] = origin
            origins.append(origin)
        return origins

    def provenance_for(self, segment: Segment) -> SegmentOrigin | None:
        return self._by_object.get(id(segment))

    def update_source_labels(
        self,
        model: SequentialMemory,
        observed_code: SymbolCode,
    ) -> None:
        """Recover the source identity split already chosen by observe_code."""

        predicted = set(model.prediction_active_cells(observed_code))
        active = set(model.previous_active_cells)
        self.current_predicted_sources = predicted.intersection(active)
        self.current_burst_sources = active - predicted

    def set_autonomous_sources(self, sources: Iterable[int]) -> None:
        """Autonomous rollout sources all come from emitted prediction cells."""

        self.current_predicted_sources = set(sources)
        self.current_burst_sources = set()

    def checkpoint_payload(self, model: SequentialMemory) -> dict[str, object]:
        rows: list[dict[str, object]] = []
        for column, neuron, segment in _iter_segments(model):
            origin = self.provenance_for(segment)
            if origin is None:
                continue
            rows.append(
                {
                    **asdict(origin),
                    "rebind_signature": _current_segment_signature(
                        column,
                        neuron,
                        segment,
                    ),
                }
            )
        return {
            "version": "fig9-branch-provenance-v1",
            "segments": rows,
            "current_predicted_sources": sorted(self.current_predicted_sources),
            "current_burst_sources": sorted(self.current_burst_sources),
        }

    @classmethod
    def from_checkpoint_payload(
        cls,
        model: SequentialMemory,
        payload: Mapping[str, object] | None,
    ) -> "BranchProvenanceRegistry":
        registry = cls()
        if not payload:
            return registry
        records_by_signature: dict[str, list[SegmentOrigin]] = {}
        for raw in payload.get("segments", ()):  # type: ignore[union-attr]
            if not isinstance(raw, Mapping):
                continue
            origin = SegmentOrigin(
                segment_provenance_id=str(raw["segment_provenance_id"]),
                creation_transition_index=int(
                    raw["creation_transition_index"]
                ),
                target_column=int(raw["target_column"]),
                target_neuron=int(raw["target_neuron"]),
                creation_target_time=(
                    float(raw["creation_target_time"])
                    if raw.get("creation_target_time") is not None
                    else None
                ),
                creation_source_cell_ids=tuple(
                    int(value)
                    for value in raw["creation_source_cell_ids"]  # type: ignore[index]
                ),
                creation_source_fingerprint=str(
                    raw["creation_source_fingerprint"]
                ),
                stable_creation_ordinal=int(raw["stable_creation_ordinal"]),
            )
            records_by_signature.setdefault(
                str(raw["rebind_signature"]),
                [],
            ).append(origin)
        segments_by_signature: dict[str, list[Segment]] = {}
        for column, neuron, segment in _iter_segments(model):
            segments_by_signature.setdefault(
                _current_segment_signature(column, neuron, segment),
                [],
            ).append(segment)
        for signature, origins in records_by_signature.items():
            segments = segments_by_signature.get(signature, ())
            for segment, origin in zip(
                segments,
                sorted(
                    origins,
                    key=lambda item: item.segment_provenance_id,
                ),
            ):
                registry._by_object[id(segment)] = origin
        registry.current_predicted_sources = {
            int(value)
            for value in payload.get("current_predicted_sources", ())  # type: ignore[union-attr]
        }
        registry.current_burst_sources = {
            int(value)
            for value in payload.get("current_burst_sources", ())  # type: ignore[union-attr]
        }
        return registry


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _candidate_key(
    *,
    policy: str,
    input_index: int,
    horizon_step: int,
    column: int,
    candidate: PredictionCandidate,
    original_index: int,
) -> str:
    payload = (
        policy,
        input_index,
        horizon_step,
        column,
        candidate.neuron_index,
        round(candidate.time, 12),
        original_index,
    )
    return hashlib.sha256(repr(payload).encode("ascii")).hexdigest()


def _support_metrics(
    contributions: Sequence[SynapsePSPContribution],
    *,
    predicted_sources: set[int],
    burst_sources: set[int],
    threshold_without_rest: float,
) -> tuple[float, float, str]:
    predicted = sum(
        item.psp_contribution
        for item in contributions
        if item.source_cell_id in predicted_sources
    )
    burst = sum(
        item.psp_contribution
        for item in contributions
        if item.source_cell_id in burst_sources
    )
    unlabelled = sum(
        item.psp_contribution
        for item in contributions
        if item.source_cell_id not in predicted_sources
        and item.source_cell_id not in burst_sources
    )
    if not predicted_sources and not burst_sources:
        classification = "unavailable"
    elif predicted + unlabelled >= threshold_without_rest:
        classification = "predicted-supported"
    elif burst >= threshold_without_rest:
        classification = "burst-only"
    else:
        classification = "burst-assisted"
    return predicted, burst, classification


def branch_trace_rows(
    *,
    model: SequentialMemory,
    registry: BranchProvenanceRegistry,
    stream_label: str,
    policy: str,
    input_index: int,
    input_timestamp: str,
    target_timestamp: str,
    horizon_step: int,
    target_code: SymbolCode,
    ranges: FieldColumnRanges,
    competition_result: CompetitionResult,
    active_sources: Mapping[int, float],
    level: str,
    source_top_n: int | None = None,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Copy provenance from the same saved PredictionCandidate objects."""

    if level not in BRANCH_LEVELS:
        raise ValueError(f"unsupported branch provenance level: {level}")
    target_columns = set(target_code.columns)
    decisions_by_candidate = {
        id(decision.candidate.candidate): decision
        for decision in competition_result.decisions
    }
    candidates: list[tuple[int, PredictionCandidate]] = [
        (column, candidate)
        for column, values in model.last_prediction_candidates.items()
        for candidate in values
    ]
    active_source_ids = set(active_sources)
    candidate_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    for fallback_index, (column, candidate) in enumerate(candidates):
        decision = decisions_by_candidate.get(id(candidate))
        original_index = (
            decision.candidate.original_order
            if decision is not None
            else fallback_index
        )
        key = _candidate_key(
            policy=policy,
            input_index=input_index,
            horizon_step=horizon_step,
            column=column,
            candidate=candidate,
            original_index=original_index,
        )
        origin = registry.provenance_for(candidate.segment)
        positive = [
            item
            for item in candidate.crossing_synapse_contributions
            if item.psp_contribution > 0.0
        ]
        total_contribution = sum(
            item.psp_contribution for item in positive
        )
        predicted_contribution, burst_contribution, support_class = (
            _support_metrics(
                positive,
                predicted_sources=registry.current_predicted_sources,
                burst_sources=registry.current_burst_sources,
                threshold_without_rest=(
                    model.params.dendrite_threshold - model._dynamics.v_rest
                ),
            )
        )
        segment_sources = set(candidate.segment.synapses)
        creation_sources = (
            set(origin.creation_source_cell_ids) if origin is not None else set()
        )
        context_overlap = len(segment_sources & active_source_ids)
        creation_current_intersection = len(
            creation_sources & segment_sources
        )
        creation_current_jaccard = (
            _jaccard(creation_sources, segment_sources)
            if origin is not None
            else 0.0
        )
        candidate_row = {
            **DIAGNOSTIC_MARKERS,
            "stream_label": stream_label,
            "policy": policy,
            "input_index": input_index,
            "input_timestamp": input_timestamp,
            "target_timestamp": target_timestamp,
            "horizon_step": horizon_step,
            "candidate_key": key,
            "candidate_original_index": original_index,
            "candidate_level": "neural",
            "is_column_representative": decision is not None,
            "preselection_candidate_segment_count": (
                model.last_prediction_stats.get(
                    "candidate_segment_count",
                    0,
                )
            ),
            "threshold_crossing_segment_count": (
                model.last_prediction_stats.get(
                    "threshold_crossing_segment_count",
                    0,
                )
            ),
            "accepted_prediction_candidate_count": (
                model.last_prediction_stats.get(
                    "accepted_candidate_count",
                    0,
                )
            ),
            "field": field_for_column(column, ranges),
            "column": column,
            "neuron": candidate.neuron_index,
            "predicted_time": candidate.time,
            "batch_index": decision.batch_index if decision else "",
            "original_score": candidate.score,
            "accumulated_inhibition": (
                decision.accumulated_inhibition if decision else ""
            ),
            "effective_score": decision.effective_score if decision else "",
            "emitted": decision.emitted if decision else False,
            "is_target_candidate": column in target_columns,
            "candidate_segment_count": 1,
            "contributing_segment_count": int(bool(positive)),
            "unique_branch_count": int(origin is not None),
            "total_positive_segment_contribution": total_contribution,
            "dominant_branch_contribution": total_contribution,
            "dominant_branch_share": 1.0 if origin and positive else 0.0,
            "branch_contribution_entropy": 0.0,
            "normalized_branch_entropy": 0.0,
            "contribution_hhi": 1.0 if origin and positive else 0.0,
            "mean_source_jaccard": 1.0 if origin else 0.0,
            "source_cell_union_count": len(segment_sources),
            "source_fingerprint_count": int(origin is not None),
            "creation_source_count": (
                len(creation_sources) if origin is not None else ""
            ),
            "current_segment_source_count": len(segment_sources),
            "source_set_growth_count": (
                len(segment_sources - creation_sources)
                if origin is not None
                else ""
            ),
            "creation_source_retention": (
                _ratio(
                    creation_current_intersection,
                    len(creation_sources),
                )
                if origin is not None
                else ""
            ),
            "creation_current_source_jaccard": (
                creation_current_jaccard if origin is not None else ""
            ),
            "current_context_overlap": context_overlap,
            "current_context_jaccard": _jaccard(
                segment_sources,
                active_source_ids,
            ),
            "historical_winner_match": (
                creation_sources == active_source_ids
                if origin is not None
                else ""
            ),
            "predicted_supported_contribution": predicted_contribution,
            "burst_supported_contribution": burst_contribution,
            "source_support_class": support_class,
            "provenance_available": origin is not None,
            "provenance_missing_reason": (
                "" if origin is not None else "segment_not_seen_at_creation"
            ),
        }
        candidate_rows.append(candidate_row)
        if level == "summary":
            continue
        segment_row = {
            **DIAGNOSTIC_MARKERS,
            "candidate_key": key,
            "segment_provenance_id": (
                origin.segment_provenance_id if origin else ""
            ),
            "segment_target_column": column,
            "segment_target_neuron": candidate.neuron_index,
            "segment_creation_transition_index": (
                origin.creation_transition_index if origin else ""
            ),
            "segment_creation_target_column": (
                origin.target_column if origin else ""
            ),
            "segment_creation_target_neuron": (
                origin.target_neuron if origin else ""
            ),
            "creation_source_fingerprint": (
                origin.creation_source_fingerprint if origin else ""
            ),
            "current_source_fingerprint": stable_source_fingerprint(
                segment_sources
            ),
            "synapse_count": len(candidate.segment.synapses),
            "contributor_synapse_count": len(positive),
            "segment_weight_sum": sum(
                synapse.weight
                for synapse in candidate.segment.synapses.values()
            ),
            "segment_age": (
                max(
                    (
                        synapse.age
                        for synapse in candidate.segment.synapses.values()
                    ),
                    default=0,
                )
            ),
            "segment_current_response": candidate.score,
            "segment_contribution_at_candidate_time": total_contribution,
            "contribution_fraction_of_candidate": 1.0 if positive else 0.0,
            "source_context_overlap": context_overlap,
            "source_context_jaccard": _jaccard(
                segment_sources,
                active_source_ids,
            ),
            "supports_target_candidate": column in target_columns,
            "segment_was_dominant": True,
            "provenance_available": origin is not None,
        }
        segment_rows.append(segment_row)
        if level != "full":
            continue
        selected_sources = sorted(
            positive,
            key=lambda item: (
                -item.psp_contribution,
                item.source_cell_id,
            ),
        )
        if source_top_n is not None:
            selected_sources = selected_sources[:source_top_n]
        for item in selected_sources:
            if item.source_cell_id in registry.current_predicted_sources:
                source_type = "predicted"
            elif item.source_cell_id in registry.current_burst_sources:
                source_type = "burst"
            else:
                source_type = "unavailable"
            synapse = candidate.segment.synapses[item.source_cell_id]
            source_rows.append(
                {
                    **DIAGNOSTIC_MARKERS,
                    "candidate_key": key,
                    "segment_provenance_id": (
                        origin.segment_provenance_id if origin else ""
                    ),
                    "source_column": (
                        item.source_cell_id
                        // len(model.columns[0].neurons)
                    ),
                    "source_neuron": (
                        item.source_cell_id
                        % len(model.columns[0].neurons)
                    ),
                    "source_event_time": item.source_time,
                    "source_type": source_type,
                    "synaptic_weight": synapse.weight,
                    "synaptic_delay": synapse.delay,
                    "arrival_time": item.arrival_time,
                    "response_at_candidate_time": (
                        _ratio(item.psp_contribution, item.weight)
                    ),
                    "contribution": item.psp_contribution,
                    "source_in_current_context": (
                        item.source_cell_id in active_source_ids
                    ),
                    "source_in_historical_creation_context": (
                        item.source_cell_id in creation_sources
                        if origin is not None
                        else ""
                    ),
                }
            )
    return candidate_rows, segment_rows, source_rows


def summarize_branch_steps(
    candidate_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Produce compact run-level counts without applying an oracle selector."""

    by_step: dict[int, list[Mapping[str, object]]] = {}
    for row in candidate_rows:
        by_step.setdefault(int(row["horizon_step"]), []).append(row)
    steps = []
    for step, rows in sorted(by_step.items()):
        available = sum(bool(row["provenance_available"]) for row in rows)
        representatives = sum(
            bool(row["is_column_representative"]) for row in rows
        )
        rollout_first_rows: dict[int, Mapping[str, object]] = {}
        for row in rows:
            rollout_first_rows.setdefault(int(row["input_index"]), row)
        preselection_segments = sum(
            int(row["preselection_candidate_segment_count"])
            for row in rollout_first_rows.values()
        )
        threshold_crossings = sum(
            int(row["threshold_crossing_segment_count"])
            for row in rollout_first_rows.values()
        )
        accepted_candidates = sum(
            int(row["accepted_prediction_candidate_count"])
            for row in rollout_first_rows.values()
        )
        steps.append(
            {
                "horizon_step": step,
                "preselection_segment_candidate_count": (
                    preselection_segments
                ),
                "threshold_crossing_segment_count": threshold_crossings,
                "neural_candidate_count": len(rows),
                "unique_column_count": len(
                    {int(row["column"]) for row in rows}
                ),
                "column_representative_count": representatives,
                "discarded_by_column_aggregation": len(rows) - representatives,
                "discarded_before_prediction_candidate": (
                    threshold_crossings - accepted_candidates
                ),
                "provenance_available_rate": _ratio(available, len(rows)),
                "mean_candidates_per_column": _ratio(
                    len(rows),
                    len({int(row["column"]) for row in rows}),
                ),
            }
        )
    return {
        **DIAGNOSTIC_MARKERS,
        "candidate_semantics": (
            "last_prediction_candidates after best-by-event and "
            "intracolumn inhibition"
        ),
        "steps": steps,
    }
