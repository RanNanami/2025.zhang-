"""Diagnostic continuous intercolumn competition for Fig.9 rollout.

This module is intentionally independent from ``SequentialMemory`` state
mutation. It consumes candidates saved by one ``predict_code()`` call and
returns a local propagation choice. It is not part of the strict paper path.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from seqmem.encoding import SpikeEvent, SymbolCode  # noqa: E402
from seqmem.model import PredictionCandidate  # noqa: E402


@dataclass(frozen=True)
class CompetitionSettings:
    """Nonpaper rollout settings kept outside the strict model config."""

    mode: str = "off"
    inhibition_strength: float = 0.0
    inhibition_tau: float = 0.02
    simultaneous_tolerance: float = 0.0
    simultaneous_policy: str = "sequential"
    simultaneous_bin_width: float = 0.005

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
        if self.simultaneous_policy not in {"sequential", "batched"}:
            raise ValueError(
                f"unsupported simultaneous policy: {self.simultaneous_policy}"
            )
        if self.simultaneous_bin_width <= 0.0:
            raise ValueError("simultaneous_bin_width must be positive")


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
    batch_index: int = 0
    batch_start_time: float = 0.0
    batch_end_time: float = 0.0
    batch_candidate_count: int = 1
    batch_emitted_count: int = 0
    earlier_batch_inhibition: float = 0.0
    same_batch_inhibition: float = 0.0


@dataclass(frozen=True)
class CompetitionBatch:
    """Aggregate metadata for one deterministic diagnostic time bucket."""

    batch_index: int
    batch_start_time: float
    batch_end_time: float
    candidate_count: int
    emitted_count: int


@dataclass(frozen=True)
class CompetitionResult:
    """Stable decisions produced without changing the model or RNG."""

    decisions: tuple[CompetitionDecision, ...]
    emitted_candidates: tuple[CompetitionCandidate, ...]
    inhibited_candidates: tuple[CompetitionCandidate, ...]
    raw_candidate_count: int
    raw_column_count: int
    emitted_column_count: int
    simultaneous_policy: str = "sequential"
    simultaneous_bin_width: float = 0.005
    batches: tuple[CompetitionBatch, ...] = ()


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
    simultaneous_policy: str = "sequential",
    simultaneous_bin_width: float = 0.005,
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
    if simultaneous_policy not in {"sequential", "batched"}:
        raise ValueError(
            f"unsupported simultaneous policy: {simultaneous_policy}"
        )
    if simultaneous_bin_width <= 0.0:
        raise ValueError("simultaneous_bin_width must be positive")

    if simultaneous_policy == "sequential":
        decisions, emitted, inhibited = _compete_sequential(
            candidates,
            threshold=threshold,
            inhibition_strength=inhibition_strength,
            inhibition_tau=inhibition_tau,
            simultaneous_tolerance=simultaneous_tolerance,
            simultaneous_bin_width=simultaneous_bin_width,
        )
    else:
        decisions, emitted, inhibited = _compete_batched(
            candidates,
            threshold=threshold,
            inhibition_strength=inhibition_strength,
            inhibition_tau=inhibition_tau,
            simultaneous_bin_width=simultaneous_bin_width,
        )

    decisions, batches = _attach_batch_counts(
        decisions,
        simultaneous_bin_width=simultaneous_bin_width,
    )
    return CompetitionResult(
        decisions=decisions,
        emitted_candidates=emitted,
        inhibited_candidates=inhibited,
        raw_candidate_count=len(decisions),
        raw_column_count=len(
            {decision.candidate.column_index for decision in decisions}
        ),
        emitted_column_count=len({item.column_index for item in emitted}),
        simultaneous_policy=simultaneous_policy,
        simultaneous_bin_width=simultaneous_bin_width,
        batches=batches,
    )


def _compete_sequential(
    candidates: Sequence[CompetitionCandidate],
    *,
    threshold: float,
    inhibition_strength: float,
    inhibition_tau: float,
    simultaneous_tolerance: float,
    simultaneous_bin_width: float,
) -> tuple[
    tuple[CompetitionDecision, ...],
    tuple[CompetitionCandidate, ...],
    tuple[CompetitionCandidate, ...],
]:
    """Preserve the original immediate-emission competition exactly."""

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
        earlier_batch_inhibition = 0.0
        same_batch_inhibition = 0.0
        item_batch = _batch_index(
            item.candidate.time,
            simultaneous_bin_width,
        )
        for predecessor in emitted:
            if predecessor.column_index == item.column_index:
                continue
            delta = item.candidate.time - predecessor.candidate.time
            if delta < -simultaneous_tolerance:
                continue
            effective_delta = 0.0 if delta <= simultaneous_tolerance else delta
            contribution = inhibition_strength * math.exp(
                -effective_delta / inhibition_tau
            )
            inhibition += contribution
            if (
                _batch_index(
                    predecessor.candidate.time,
                    simultaneous_bin_width,
                )
                == item_batch
            ):
                same_batch_inhibition += contribution
            else:
                earlier_batch_inhibition += contribution
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
                batch_index=item_batch,
                batch_start_time=_batch_start(
                    item_batch,
                    simultaneous_bin_width,
                ),
                batch_end_time=_batch_end(
                    item_batch,
                    simultaneous_bin_width,
                ),
                earlier_batch_inhibition=earlier_batch_inhibition,
                same_batch_inhibition=same_batch_inhibition,
            )
        )

    return tuple(decisions), tuple(emitted), tuple(inhibited)


def _compete_batched(
    candidates: Sequence[CompetitionCandidate],
    *,
    threshold: float,
    inhibition_strength: float,
    inhibition_tau: float,
    simultaneous_bin_width: float,
) -> tuple[
    tuple[CompetitionDecision, ...],
    tuple[CompetitionCandidate, ...],
    tuple[CompetitionCandidate, ...],
]:
    """Decide one time bucket before exposing its winners to later buckets."""

    ordered = sorted(
        candidates,
        key=lambda item: (
            _batch_index(item.candidate.time, simultaneous_bin_width),
            item.candidate.time,
            item.column_index,
            item.candidate.neuron_index,
            item.original_order,
        ),
    )
    emitted: list[CompetitionCandidate] = []
    inhibited: list[CompetitionCandidate] = []
    decisions: list[CompetitionDecision] = []
    cursor = 0
    while cursor < len(ordered):
        batch_index = _batch_index(
            ordered[cursor].candidate.time,
            simultaneous_bin_width,
        )
        end = cursor + 1
        while (
            end < len(ordered)
            and _batch_index(
                ordered[end].candidate.time,
                simultaneous_bin_width,
            )
            == batch_index
        ):
            end += 1

        batch_decisions: list[CompetitionDecision] = []
        for item in ordered[cursor:end]:
            predecessor_ids: list[str] = []
            inhibition = 0.0
            for predecessor in emitted:
                if predecessor.column_index == item.column_index:
                    continue
                delta = item.candidate.time - predecessor.candidate.time
                if delta < 0.0:
                    continue
                contribution = inhibition_strength * math.exp(
                    -delta / inhibition_tau
                )
                inhibition += contribution
                predecessor_ids.append(predecessor.candidate_id)

            effective_score = item.candidate.score - inhibition
            does_emit = effective_score >= threshold
            batch_decisions.append(
                CompetitionDecision(
                    candidate=item,
                    original_score=item.candidate.score,
                    accumulated_inhibition=inhibition,
                    effective_score=effective_score,
                    emitted=does_emit,
                    reason=(
                        "effective_score_at_or_above_threshold"
                        if does_emit
                        else "effective_score_below_threshold"
                    ),
                    winning_predecessor_ids=tuple(predecessor_ids),
                    batch_index=batch_index,
                    batch_start_time=_batch_start(
                        batch_index,
                        simultaneous_bin_width,
                    ),
                    batch_end_time=_batch_end(
                        batch_index,
                        simultaneous_bin_width,
                    ),
                    earlier_batch_inhibition=inhibition,
                    same_batch_inhibition=0.0,
                )
            )

        batch_emitted = [
            decision.candidate
            for decision in batch_decisions
            if decision.emitted
        ]
        emitted.extend(batch_emitted)
        inhibited.extend(
            decision.candidate
            for decision in batch_decisions
            if not decision.emitted
        )
        decisions.extend(batch_decisions)
        cursor = end

    return tuple(decisions), tuple(emitted), tuple(inhibited)


def _batch_index(predicted_time: float, bin_width: float) -> int:
    """Map time to a deterministic nearest-grid bucket."""

    return round(predicted_time / bin_width)


def _batch_start(batch_index: int, bin_width: float) -> float:
    return (batch_index - 0.5) * bin_width


def _batch_end(batch_index: int, bin_width: float) -> float:
    return (batch_index + 0.5) * bin_width


def _attach_batch_counts(
    decisions: Sequence[CompetitionDecision],
    *,
    simultaneous_bin_width: float,
) -> tuple[tuple[CompetitionDecision, ...], tuple[CompetitionBatch, ...]]:
    grouped: dict[int, list[CompetitionDecision]] = {}
    for decision in decisions:
        grouped.setdefault(decision.batch_index, []).append(decision)

    batches: list[CompetitionBatch] = []
    updated: list[CompetitionDecision] = []
    for batch_index in sorted(grouped):
        members = grouped[batch_index]
        emitted_count = sum(member.emitted for member in members)
        batches.append(
            CompetitionBatch(
                batch_index=batch_index,
                batch_start_time=_batch_start(
                    batch_index,
                    simultaneous_bin_width,
                ),
                batch_end_time=_batch_end(
                    batch_index,
                    simultaneous_bin_width,
                ),
                candidate_count=len(members),
                emitted_count=emitted_count,
            )
        )
        updated.extend(
            replace(
                member,
                batch_candidate_count=len(members),
                batch_emitted_count=emitted_count,
            )
            for member in members
        )
    return tuple(updated), tuple(batches)


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
        batch_counts = [
            _integer_sequence(row.get("batch_candidate_count", ""))
            for row in selected
        ]
        emitted_batch_counts = [
            _integer_sequence(row.get("batch_emitted_count", ""))
            for row in selected
        ]
        flat_batch_counts = [
            count for counts in batch_counts for count in counts
        ]
        flat_emitted_batch_counts = [
            count for counts in emitted_batch_counts for count in counts
        ]
        total_batched_candidates = sum(flat_batch_counts)
        same_batch_candidates = sum(
            count for count in flat_batch_counts if count > 1
        )
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
                "batch_count_mean": (
                    sum(len(counts) for counts in batch_counts) / len(selected)
                    if selected
                    else 0.0
                ),
                "candidates_per_batch_mean": (
                    total_batched_candidates / len(flat_batch_counts)
                    if flat_batch_counts
                    else 0.0
                ),
                "emitted_per_batch_mean": (
                    sum(flat_emitted_batch_counts)
                    / len(flat_emitted_batch_counts)
                    if flat_emitted_batch_counts
                    else 0.0
                ),
                "same_batch_candidate_fraction": (
                    same_batch_candidates / total_batched_candidates
                    if total_batched_candidates
                    else 0.0
                ),
                # These require paired ground-truth oracle traces and are
                # populated by compare_fig9_competition_policies.py.
                "target_candidates_sharing_batch_with_earlier_false": None,
                "target_candidates_rescued_by_batched": None,
                "false_candidates_additionally_emitted_by_batched": None,
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


def _integer_sequence(value: object) -> list[int]:
    if value in {"", None}:
        return []
    return [int(item) for item in str(value).split()]
