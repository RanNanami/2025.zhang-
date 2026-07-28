"""Read-only candidate rows for offline Fig.9 score diagnostics."""

from __future__ import annotations

from typing import Sequence

from experiments.diagnostics.fig9_competitive_inhibition import (
    CompetitionResult,
)
from experiments.diagnostics.fig9_oracle_candidate import FieldColumnRanges
from seqmem.encoding import SymbolCode


TRACE_FIELDS = (
    "diagnostic_only",
    "uses_ground_truth_for_analysis_only",
    "ground_truth_does_not_affect_prediction",
    "offline_analysis_only",
    "policy",
    "input_index",
    "target_timestamp",
    "horizon_step",
    "candidate_original_index",
    "field",
    "column",
    "neuron",
    "predicted_time",
    "batch_index",
    "original_score",
    "accumulated_inhibition",
    "effective_score",
    "emitted",
    "is_target_candidate",
)


def candidate_score_trace_rows(
    *,
    policy: str,
    input_index: int,
    target_timestamp: str,
    horizon_step: int,
    competition_result: CompetitionResult,
    target_code: SymbolCode,
    ranges: FieldColumnRanges,
) -> list[dict[str, object]]:
    """Label saved competition decisions without changing their outcome."""

    target_columns = set(target_code.columns)
    return [
        {
            "diagnostic_only": True,
            "uses_ground_truth_for_analysis_only": True,
            "ground_truth_does_not_affect_prediction": True,
            "offline_analysis_only": True,
            "policy": policy,
            "input_index": input_index,
            "target_timestamp": target_timestamp,
            "horizon_step": horizon_step,
            "candidate_original_index": decision.candidate.original_order,
            "field": field_for_column(
                decision.candidate.column_index,
                ranges,
            ),
            "column": decision.candidate.column_index,
            "neuron": decision.candidate.candidate.neuron_index,
            "predicted_time": decision.candidate.candidate.time,
            "batch_index": decision.batch_index,
            "original_score": decision.original_score,
            "accumulated_inhibition": decision.accumulated_inhibition,
            "effective_score": decision.effective_score,
            "emitted": decision.emitted,
            "is_target_candidate": (
                decision.candidate.column_index in target_columns
            ),
        }
        for decision in competition_result.decisions
    ]


def field_for_column(
    column: int,
    ranges: FieldColumnRanges,
) -> str:
    """Map a mini-column using protocol-derived contiguous field offsets."""

    if 0 <= column < ranges.weekday_stop:
        return "weekday"
    if ranges.weekday_stop <= column < ranges.time_stop:
        return "time"
    if ranges.time_stop <= column < ranges.passenger_stop:
        return "passenger"
    raise ValueError(
        f"column {column} is outside protocol total {ranges.passenger_stop}"
    )


def validate_trace_schema(fieldnames: Sequence[str] | None) -> None:
    """Raise a focused error when an input is not a candidate-level trace."""

    available = set(fieldnames or ())
    missing = [field for field in TRACE_FIELDS if field not in available]
    if missing:
        raise ValueError(
            "candidate trace is missing required fields: "
            + ", ".join(missing)
        )
