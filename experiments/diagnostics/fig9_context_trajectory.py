"""Read-only context trajectory tracing for Fig.9 diagnostics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from experiments.diagnostics.fig9_competitive_inhibition import (
    CompetitionResult,
)
from experiments.diagnostics.fig9_oracle_candidate import FieldColumnRanges
from experiments.diagnostics.fig9_candidate_score_trace import field_for_column
from seqmem.encoding import SymbolCode
from seqmem.model import PreselectionTrace, SequentialMemory


CONTEXT_TRAJECTORY_LEVELS = {"summary", "column", "cell"}
CONTEXT_TRAJECTORY_MARKERS = {
    "diagnostic_only": True,
    "offline_analysis_only": True,
    "context_trajectory_decomposition": True,
    "diagnostic_does_not_affect_prediction": True,
    "diagnostic_does_not_affect_learning": True,
    "ground_truth_does_not_affect_model": True,
    "actual_and_autonomous_trajectories_separated": True,
    "uses_compensation": False,
}

LOSS_REASONS = {
    "NO_INSPECTED_SEGMENT",
    "INSPECTED_ZERO_OVERLAP",
    "RESPONSE_BELOW_THRESHOLD",
    "CROSSING_INVALID_TIME",
    "CANDIDATE_NOT_SAVED",
    "LOST_INTRACOLUMN",
    "LOST_INTERCOLUMN",
    "EMITTED_NOT_PROPAGATED",
    "LOST_BEFORE_CURRENT_TRANSITION",
    "NEVER_REACTIVATED_AFTER_CREATION",
    "OTHER_EXPLICIT_CODE_PATH",
    "UNKNOWN",
}


def _columns_from_cells(model: SequentialMemory, cells: Iterable[int]) -> set[int]:
    width = len(model.columns[0].neurons)
    return {int(cell) // width for cell in cells}


def _columns_from_code(code: SymbolCode | None) -> set[int]:
    return {event.column for event in code.events} if code is not None else set()


@dataclass(frozen=True)
class ColumnTransition:
    """One column's progress through a real prediction transition."""

    trajectory_kind: str
    actual_record_index: int
    effective_record_index: int
    horizon_step: int
    column: int
    segment_inspected: bool
    overlap_positive: bool
    response_computed: bool
    threshold_crossed: bool
    firing_time_valid: bool
    saved_candidate: bool
    intracolumn_selected: bool
    intercolumn_survived: bool
    emitted_winner: bool
    entered_previous_winners: bool
    present_in_next_context: bool


@dataclass(frozen=True)
class TransitionSnapshot:
    """Compact stage sets for one transition, including absent columns."""

    trajectory_kind: str
    actual_record_index: int
    effective_record_index: int
    horizon_step: int
    segment_inspected: frozenset[int]
    overlap_positive: frozenset[int]
    response_computed: frozenset[int]
    threshold_crossed: frozenset[int]
    firing_time_valid: frozenset[int]
    saved_candidate: frozenset[int]
    intracolumn_selected: frozenset[int]
    intercolumn_survived: frozenset[int]
    emitted_winner: frozenset[int]
    entered_previous_winners: frozenset[int]
    present_in_next_context: frozenset[int]

    def for_column(self, column: int) -> ColumnTransition:
        return ColumnTransition(
            trajectory_kind=self.trajectory_kind,
            actual_record_index=self.actual_record_index,
            effective_record_index=self.effective_record_index,
            horizon_step=self.horizon_step,
            column=column,
            segment_inspected=column in self.segment_inspected,
            overlap_positive=column in self.overlap_positive,
            response_computed=column in self.response_computed,
            threshold_crossed=column in self.threshold_crossed,
            firing_time_valid=column in self.firing_time_valid,
            saved_candidate=column in self.saved_candidate,
            intracolumn_selected=column in self.intracolumn_selected,
            intercolumn_survived=column in self.intercolumn_survived,
            emitted_winner=column in self.emitted_winner,
            entered_previous_winners=column in self.entered_previous_winners,
            present_in_next_context=column in self.present_in_next_context,
        )


def _stage_loss(row: ColumnTransition) -> str:
    if not row.segment_inspected:
        return "NO_INSPECTED_SEGMENT"
    if not row.overlap_positive:
        return "INSPECTED_ZERO_OVERLAP"
    if not row.response_computed:
        return "RESPONSE_BELOW_THRESHOLD"
    if not row.threshold_crossed:
        return "RESPONSE_BELOW_THRESHOLD"
    if not row.firing_time_valid:
        return "CROSSING_INVALID_TIME"
    if not row.saved_candidate:
        return "CANDIDATE_NOT_SAVED"
    if not row.intracolumn_selected:
        return "LOST_INTRACOLUMN"
    if not row.intercolumn_survived:
        return "LOST_INTERCOLUMN"
    if not row.emitted_winner:
        return "LOST_INTERCOLUMN"
    if not row.entered_previous_winners or not row.present_in_next_context:
        return "EMITTED_NOT_PROPAGATED"
    return ""


class ContextTrajectoryTracker:
    """Accumulate column-level transitions without touching model state."""

    def __init__(self) -> None:
        self._transitions: dict[str, list[TransitionSnapshot]] = defaultdict(list)

    def record_transition(
        self,
        *,
        model: SequentialMemory,
        trajectory_kind: str,
        actual_record_index: int,
        horizon_step: int,
        preselection_trace: PreselectionTrace,
        raw_code: SymbolCode | None,
        competition_result: CompetitionResult | None,
        next_active_cells: Mapping[int, float],
        next_winners: Mapping[int, float],
    ) -> None:
        """Record stages produced by the same prediction and propagation."""

        if trajectory_kind not in {"actual_observation", "autonomous_rollout"}:
            raise ValueError("unsupported trajectory kind")
        by_column: dict[int, list[object]] = defaultdict(list)
        for item in preselection_trace.segments:
            by_column[item.target_column].append(item)

        raw_columns = _columns_from_code(raw_code)
        if competition_result is None:
            survived_columns = set(raw_columns)
        else:
            survived_columns = {
                decision.candidate.column_index
                for decision in competition_result.decisions
                if decision.emitted
            }
        active_columns = _columns_from_cells(model, next_active_cells)
        winner_columns = _columns_from_cells(model, next_winners)
        effective_index = actual_record_index + horizon_step
        inspected = frozenset(by_column)
        overlap = frozenset(
            column
            for column, segments in by_column.items()
            if any(
                int(getattr(item, "active_matched_synapse_count", 0)) > 0
                for item in segments
            )
        )
        response = frozenset(
            column
            for column, segments in by_column.items()
            if any(
                bool(getattr(item, "response_computed", False))
                for item in segments
            )
        )
        crossed = frozenset(
            column
            for column, segments in by_column.items()
            if any(
                bool(getattr(item, "crossed_threshold", False))
                for item in segments
            )
        )
        valid = frozenset(
            column
            for column, segments in by_column.items()
            if any(
                bool(getattr(item, "valid_firing_time", False))
                for item in segments
            )
        )
        saved = frozenset(
            column
            for column, segments in by_column.items()
            if any(
                bool(getattr(item, "became_prediction_candidate", False))
                for item in segments
            )
        )
        self._transitions[trajectory_kind].append(
            TransitionSnapshot(
                trajectory_kind=trajectory_kind,
                actual_record_index=actual_record_index,
                effective_record_index=effective_index,
                horizon_step=horizon_step,
                segment_inspected=inspected,
                overlap_positive=overlap,
                response_computed=response,
                threshold_crossed=crossed,
                firing_time_valid=valid,
                saved_candidate=saved,
                intracolumn_selected=frozenset(saved & raw_columns),
                intercolumn_survived=frozenset(survived_columns),
                emitted_winner=frozenset(survived_columns),
                entered_previous_winners=frozenset(winner_columns),
                present_in_next_context=frozenset(active_columns),
            )
        )

    def dependency_rows(
        self,
        *,
        source_rows: Sequence[dict[str, object]],
        ranges: FieldColumnRanges,
        level: str,
    ) -> list[dict[str, object]]:
        """Join inactive historical sources to prior actual/rollout histories."""

        if level not in CONTEXT_TRAJECTORY_LEVELS:
            raise ValueError("context trajectory level must be summary, column, or cell")
        inactive = [
            row
            for row in source_rows
            if row.get("source_loss_reason_primary")
            == "SOURCE_COLUMN_NOT_ACTIVE"
        ]
        if level == "summary":
            return []

        grouped: dict[tuple[object, ...], dict[str, object]] = {}
        for source in inactive:
            cell_suffix: tuple[object, ...] = ()
            if level == "cell":
                cell_suffix = (
                    source.get("source_neuron", ""),
                    source.get("source_cell_stable_id", ""),
                )
            key = (
                source.get("actual_record_index"),
                source.get("timestamp"),
                source.get("field"),
                source.get("encoded_column"),
                source.get("observe_scenario"),
                source.get("segment_provenance_id"),
                source.get("segment_creation_transition_index"),
                source.get("source_field"),
                source.get("source_column"),
                *cell_suffix,
            )
            grouped.setdefault(key, source)

        output: list[dict[str, object]] = []
        history_cache: dict[tuple[str, int, int, int], dict[str, object]] = {}
        for source in grouped.values():
            current_index = int(source["actual_record_index"])
            source_column = int(source["source_column"])
            creation_raw = source.get("segment_creation_transition_index", "")
            creation_index = (
                int(creation_raw)
                if creation_raw not in {"", None}
                else -1
            )
            for kind in ("actual_observation", "autonomous_rollout"):
                cache_key = (kind, source_column, creation_index, current_index)
                history_summary = history_cache.get(cache_key)
                if history_summary is None:
                    history = [
                        snapshot.for_column(source_column)
                        for snapshot in self._transitions.get(kind, ())
                        if snapshot.effective_record_index < current_index
                        and snapshot.effective_record_index >= creation_index
                    ]
                    history_summary = self._summarize_history(history)
                    history_cache[cache_key] = history_summary
                output.append(
                    {
                        **CONTEXT_TRAJECTORY_MARKERS,
                        "actual_record_index": source["actual_record_index"],
                        "timestamp": source.get("timestamp", ""),
                        "observed_field": source.get("field", ""),
                        "observed_column": source.get("encoded_column", ""),
                        "observe_scenario": source.get("observe_scenario", ""),
                        "segment_id": source.get("segment_provenance_id", ""),
                        "segment_creation_transition_index": source.get(
                            "segment_creation_transition_index", ""
                        ),
                        "source_field": source.get(
                            "source_field",
                            field_for_column(source_column, ranges),
                        ),
                        "source_column": source_column,
                        "source_neuron": source.get("source_neuron", ""),
                        "source_cell_stable_id": source.get(
                            "source_cell_stable_id", ""
                        ),
                        "trajectory_kind": kind,
                        **history_summary,
                    }
                )
        return output

    @staticmethod
    def _summarize_dependency(
        *,
        source: dict[str, object],
        history: Sequence[ColumnTransition],
        trajectory_kind: str,
        ranges: FieldColumnRanges,
    ) -> dict[str, object]:
        source_column = int(source["source_column"])
        return {
            **CONTEXT_TRAJECTORY_MARKERS,
            "actual_record_index": source["actual_record_index"],
            "timestamp": source.get("timestamp", ""),
            "observed_field": source.get("field", ""),
            "observed_column": source.get("encoded_column", ""),
            "observe_scenario": source.get("observe_scenario", ""),
            "segment_id": source.get("segment_provenance_id", ""),
            "segment_creation_transition_index": source.get(
                "segment_creation_transition_index", ""
            ),
            "source_field": source.get(
                "source_field", field_for_column(source_column, ranges)
            ),
            "source_column": source_column,
            "source_neuron": source.get("source_neuron", ""),
            "source_cell_stable_id": source.get("source_cell_stable_id", ""),
            "trajectory_kind": trajectory_kind,
            **ContextTrajectoryTracker._summarize_history(history),
        }

    @staticmethod
    def _summarize_history(
        history: Sequence[ColumnTransition],
    ) -> dict[str, object]:
        losses = [_stage_loss(item) for item in history]
        explicit_losses = [reason for reason in losses if reason]
        loss_items = [
            item for item, reason in zip(history, losses) if reason
        ]
        present = [item.present_in_next_context for item in history]
        last_active = max(
            (
                item.effective_record_index
                for item in history
                if item.present_in_next_context
            ),
            default=None,
        )
        recovery_count = sum(
            not previous and current
            for previous, current in zip(present, present[1:])
        )
        if not history:
            pattern = "NO_TRAJECTORY_HISTORY"
            primary = "LOST_BEFORE_CURRENT_TRANSITION"
        elif not any(present):
            pattern = "NEVER_PRESENT_AFTER_CREATION"
            primary = "NEVER_REACTIVATED_AFTER_CREATION"
        elif all(present):
            pattern = "CONSISTENTLY_PRESENT"
            primary = "LOST_BEFORE_CURRENT_TRANSITION"
        elif recovery_count and not present[-1]:
            pattern = "INTERMITTENT_THEN_LOST"
            primary = explicit_losses[-1] if explicit_losses else "LOST_BEFORE_CURRENT_TRANSITION"
        elif recovery_count:
            pattern = "LOST_THEN_RECOVERED"
            primary = "LOST_BEFORE_CURRENT_TRANSITION"
        else:
            pattern = "ACTIVE_THEN_LOST"
            primary = explicit_losses[-1] if explicit_losses else "LOST_BEFORE_CURRENT_TRANSITION"
        if primary not in LOSS_REASONS:
            primary = "OTHER_EXPLICIT_CODE_PATH"

        latest = history[-1] if history else None
        return {
            "history_transition_count": len(history),
            "history_available": bool(history),
            "latest_horizon_step": (
                latest.horizon_step if latest is not None else ""
            ),
            "segment_inspected": bool(latest and latest.segment_inspected),
            "overlap_positive": bool(latest and latest.overlap_positive),
            "response_computed": bool(latest and latest.response_computed),
            "threshold_crossed": bool(latest and latest.threshold_crossed),
            "firing_time_valid": bool(latest and latest.firing_time_valid),
            "saved_candidate": bool(latest and latest.saved_candidate),
            "intracolumn_selected": bool(
                latest and latest.intracolumn_selected
            ),
            "intercolumn_survived": bool(
                latest and latest.intercolumn_survived
            ),
            "emitted_winner": bool(latest and latest.emitted_winner),
            "entered_previous_winners": bool(
                latest and latest.entered_previous_winners
            ),
            "present_in_next_context": bool(
                latest and latest.present_in_next_context
            ),
            "last_active_index": last_active if last_active is not None else "",
            "first_loss_stage": (
                explicit_losses[0]
                if explicit_losses
                else (
                    "LOST_BEFORE_CURRENT_TRANSITION"
                    if not history
                    else ""
                )
            ),
            "first_loss_horizon_step": (
                loss_items[0].horizon_step if loss_items else ""
            ),
            "most_recent_loss_stage": (
                explicit_losses[-1]
                if explicit_losses
                else (
                    "LOST_BEFORE_CURRENT_TRANSITION"
                    if not history
                    else ""
                )
            ),
            "latest_loss_stage": (
                _stage_loss(latest) if latest is not None else
                "LOST_BEFORE_CURRENT_TRANSITION"
            ),
            "most_recent_loss_horizon_step": (
                loss_items[-1].horizon_step if loss_items else ""
            ),
            "recovery_count": recovery_count,
            "trajectory_pattern": pattern,
            "trajectory_loss_primary_reason": primary,
            "source_loss_reason_joined": "SOURCE_COLUMN_NOT_ACTIVE",
        }
