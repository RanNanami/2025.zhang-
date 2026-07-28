"""Diagnostic continuous intercolumn competition for Fig.9 rollout.

This module is intentionally independent from ``SequentialMemory`` state
mutation. It consumes candidates saved by one ``predict_code()`` call and
returns a local propagation choice. It is not part of the strict paper path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from seqmem.encoding import SpikeEvent, SymbolCode
from seqmem.model import PredictionCandidate


@dataclass(frozen=True)
class CompetitionSettings:
    """Nonpaper rollout settings kept outside the strict model config."""

    mode: str = "off"
    inhibition_strength: float = 0.0
    inhibition_tau: float = 0.02
    simultaneous_tolerance: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.mode == "competitive_raw"

    def validate(self) -> None:
        if self.mode not in {"off", "competitive_raw"}:
            raise ValueError(f"unsupported competition mode: {self.mode}")
        if self.inhibition_strength < 0.0:
            raise ValueError("inhibition_strength must be nonnegative")
        if self.inhibition_tau <= 0.0:
            raise ValueError("inhibition_tau must be positive")
        if self.simultaneous_tolerance < 0.0:
            raise ValueError("simultaneous_tolerance must be nonnegative")


@dataclass(frozen=True)
class CompetitionCandidate:
    """A prediction candidate paired with its mini-column and stable order."""

    column_index: int
    candidate: PredictionCandidate
    original_order: int

    @property
    def candidate_id(self) -> str:
        return (
            f"{self.column_index}:{self.candidate.neuron_index}:"
            f"{self.candidate.time:.12f}:{self.original_order}"
        )


@dataclass(frozen=True)
class CompetitionDecision:
    """Auditable competition outcome for one candidate."""

    candidate: CompetitionCandidate
    original_score: float
    accumulated_inhibition: float
    effective_score: float
    emitted: bool
    reason: str
    winning_predecessor_ids: tuple[str, ...]


@dataclass(frozen=True)
class CompetitionResult:
    """Stable decisions produced without changing the model or RNG."""

    decisions: tuple[CompetitionDecision, ...]
    emitted_candidates: tuple[CompetitionCandidate, ...]
    inhibited_candidates: tuple[CompetitionCandidate, ...]
    raw_candidate_count: int
    raw_column_count: int
    emitted_column_count: int


def candidates_for_prediction(
    predicted: SymbolCode,
    candidates_by_column: Mapping[int, Sequence[PredictionCandidate]],
    *,
    timing_tolerance: float,
) -> tuple[CompetitionCandidate, ...]:
    """Bind each raw event to the neuron selected by the existing raw path.

    ``prediction_active_cells`` resolves an event to the highest-score matching
    candidate. Reusing that rule preserves the candidate identity that raw
    propagation would activate.
    """

    selected: list[CompetitionCandidate] = []
    for original_order, event in enumerate(predicted.events):
        matching = [
            candidate
            for candidate in candidates_by_column.get(event.column, ())
            if abs(candidate.time - event.time) <= timing_tolerance
        ]
        if not matching:
            continue
        winner = max(matching, key=lambda candidate: candidate.score)
        selected.append(
            CompetitionCandidate(
                column_index=event.column,
                candidate=winner,
                original_order=original_order,
            )
        )
    return tuple(selected)


def compete_prediction_candidates(
    candidates: Sequence[CompetitionCandidate],
    *,
    threshold: float,
    inhibition_strength: float,
    inhibition_tau: float,
    simultaneous_tolerance: float,
) -> CompetitionResult:
    """Apply event-driven inhibition from previously emitted other columns.

    For candidate ``i``, every earlier emitted candidate ``j`` from another
    mini-column contributes

    ``strength * exp(-max(0, t_i - t_j) / tau)``.

    Time differences within ``simultaneous_tolerance`` are treated as zero.
    Stable ordering makes this local same-window choice deterministic.
    """

    if inhibition_strength < 0.0:
        raise ValueError("inhibition_strength must be nonnegative")
    if inhibition_tau <= 0.0:
        raise ValueError("inhibition_tau must be positive")
    if simultaneous_tolerance < 0.0:
        raise ValueError("simultaneous_tolerance must be nonnegative")

    ordered = sorted(
        candidates,
        key=lambda item: (
            item.candidate.time,
            item.column_index,
            item.candidate.neuron_index,
            item.original_order,
        ),
    )
    emitted: list[CompetitionCandidate] = []
    inhibited: list[CompetitionCandidate] = []
    decisions: list[CompetitionDecision] = []

    for item in ordered:
        predecessor_ids: list[str] = []
        inhibition = 0.0
        for predecessor in emitted:
            if predecessor.column_index == item.column_index:
                continue
            delta = item.candidate.time - predecessor.candidate.time
            if delta < -simultaneous_tolerance:
                continue
            effective_delta = 0.0 if delta <= simultaneous_tolerance else delta
            inhibition += inhibition_strength * math.exp(
                -effective_delta / inhibition_tau
            )
            predecessor_ids.append(predecessor.candidate_id)

        effective_score = item.candidate.score - inhibition
        does_emit = effective_score >= threshold
        if does_emit:
            emitted.append(item)
            reason = "effective_score_at_or_above_threshold"
        else:
            inhibited.append(item)
            reason = "effective_score_below_threshold"
        decisions.append(
            CompetitionDecision(
                candidate=item,
                original_score=item.candidate.score,
                accumulated_inhibition=inhibition,
                effective_score=effective_score,
                emitted=does_emit,
                reason=reason,
                winning_predecessor_ids=tuple(predecessor_ids),
            )
        )

    return CompetitionResult(
        decisions=tuple(decisions),
        emitted_candidates=tuple(emitted),
        inhibited_candidates=tuple(inhibited),
        raw_candidate_count=len(ordered),
        raw_column_count=len({item.column_index for item in ordered}),
        emitted_column_count=len({item.column_index for item in emitted}),
    )


def emitted_prediction_code(
    raw_prediction: SymbolCode,
    result: CompetitionResult,
) -> SymbolCode | None:
    """Build the propagation code selected by competition.

    When every raw candidate emits, the original ``SymbolCode`` is returned
    verbatim so zero-strength competition is exactly the raw path.
    """

    if not result.emitted_candidates:
        return None
    if (
        not result.inhibited_candidates
        and len(result.emitted_candidates) == len(raw_prediction.events)
    ):
        return raw_prediction
    return SymbolCode(
        events=tuple(
            SpikeEvent(
                column=item.column_index,
                time=item.candidate.time,
            )
            for item in result.emitted_candidates
        )
    )


def summarize_competition(
    rows: Sequence[Mapping[str, object]],
    *,
    horizon: int,
    attempted_rollouts: int,
    segment_count: int,
    runtime_seconds: float,
) -> dict[str, object]:
    """Summarize candidate suppression and prediction quality by horizon."""

    step_summary: list[dict[str, object]] = []
    for step in range(1, horizon + 1):
        selected = [row for row in rows if int(row["horizon_step"]) == step]
        raw_columns = [
            float(row["raw_predicted_column_count"]) for row in selected
        ]
        emitted_columns = [
            float(row["emitted_column_count"]) for row in selected
        ]
        decoded = [
            row for row in selected if row.get("decoded_passenger", "") != ""
        ]
        absolute_errors = [float(row["absolute_error"]) for row in decoded]
        targets = [
            abs(float(row["actual_future_passenger"])) for row in decoded
        ]
        paired_density = [
            (
                float(row["emitted_column_count"]),
                float(row["absolute_percentage_error"]),
            )
            for row in decoded
            if row.get("absolute_percentage_error", "") != ""
        ]
        density_error_correlation = _correlation(paired_density)
        raw_total = sum(raw_columns)
        emitted_total = sum(emitted_columns)
        step_summary.append(
            {
                "horizon_step": step,
                "rows": len(selected),
                "raw_column_mean": (
                    raw_total / len(raw_columns) if raw_columns else 0.0
                ),
                "emitted_column_mean": (
                    emitted_total / len(emitted_columns)
                    if emitted_columns
                    else 0.0
                ),
                "suppression_ratio": (
                    1.0 - emitted_total / raw_total if raw_total else 0.0
                ),
                "coverage": (
                    len(decoded) / attempted_rollouts
                    if attempted_rollouts
                    else 0.0
                ),
                "mape": (
                    sum(absolute_errors) / sum(targets)
                    if absolute_errors and sum(targets)
                    else 0.0
                ),
                "density_error_correlation": density_error_correlation,
                "segment_count": segment_count,
                "runtime_seconds": runtime_seconds,
            }
        )
    return {
        "diagnostic_only": True,
        "competition_is_local_choice": True,
        "attempted_rollouts": attempted_rollouts,
        "trace_rows": len(rows),
        "step_summary": step_summary,
    }


def _correlation(pairs: Sequence[tuple[float, float]]) -> float:
    if len(pairs) < 2:
        return 0.0
    left = [pair[0] for pair in pairs]
    right = [pair[1] for pair in pairs]
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right)
    )
    left_variance = sum((value - left_mean) ** 2 for value in left)
    right_variance = sum((value - right_mean) ** 2 for value in right)
    if left_variance == 0.0 or right_variance == 0.0:
        return 0.0
    return numerator / math.sqrt(left_variance * right_variance)
