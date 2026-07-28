"""Ground-truth candidate survival analysis for competitive Fig.9 rollout.

Ground truth enters only these post-selection functions. Nothing in this
module constructs competition winners or a propagation code.
"""

from __future__ import annotations

import math
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.diagnostics.fig9_competitive_inhibition import CompetitionResult
from seqmem.encoding import SymbolCode  # noqa: E402


@dataclass(frozen=True)
class FieldColumnRanges:
    weekday_stop: int
    time_stop: int
    passenger_stop: int

    @classmethod
    def from_sizes(
        cls,
        weekday_columns: int,
        time_columns: int,
        passenger_columns: int,
    ) -> "FieldColumnRanges":
        weekday_stop = weekday_columns
        time_stop = weekday_stop + time_columns
        return cls(
            weekday_stop=weekday_stop,
            time_stop=time_stop,
            passenger_stop=time_stop + passenger_columns,
        )


CLASSIFICATION_THRESHOLDS = {
    "mostly_suppressed_survival_ratio_below": 0.5,
    "high_recall_at_or_above": 0.7,
    "decode_wrong_relative_error_above": 0.25,
    "false_dominates_ratio_above": 0.5,
    "branch_incoherent_coherence_below": 0.5,
}


def split_field_columns(
    code: SymbolCode | None,
    ranges: FieldColumnRanges,
) -> dict[str, set[int]]:
    """Split unique code columns into the three fixed Fig.9 fields."""

    columns = {event.column for event in code.events} if code is not None else set()
    return {
        "weekday": {
            column for column in columns if 0 <= column < ranges.weekday_stop
        },
        "time": {
            column
            for column in columns
            if ranges.weekday_stop <= column < ranges.time_stop
        },
        "passenger": {
            column
            for column in columns
            if ranges.time_stop <= column < ranges.passenger_stop
        },
        "all": {
            column for column in columns if 0 <= column < ranges.passenger_stop
        },
    }


def analyze_oracle_step(
    *,
    prediction_input_index: int,
    input_timestamp: str,
    horizon_step: int,
    target_timestamp: str,
    target_passenger: float,
    decoded_prediction: float | str,
    absolute_error: float | str,
    raw_prediction: SymbolCode | None,
    emitted_prediction: SymbolCode | None,
    competition_result: CompetitionResult | None,
    target_code: SymbolCode,
    ranges: FieldColumnRanges,
) -> dict[str, object]:
    """Compare an already-fixed competition result with the true target code."""

    target = split_field_columns(target_code, ranges)
    raw = split_field_columns(raw_prediction, ranges)
    emitted = split_field_columns(emitted_prediction, ranges)

    raw_hits = {field: target[field] & raw[field] for field in target}
    emitted_hits = {
        field: target[field] & emitted[field] for field in target
    }
    suppressed = {
        field: raw_hits[field] - emitted_hits[field] for field in target
    }

    decisions = competition_result.decisions if competition_result is not None else ()
    target_decisions = [
        decision
        for decision in decisions
        if decision.candidate.column_index in target["all"]
    ]
    false_decisions = [
        decision
        for decision in decisions
        if decision.candidate.column_index not in target["all"]
    ]
    target_ranks = _target_score_ranks(decisions, target["all"])

    target_emitted = [
        decision for decision in target_decisions if decision.emitted
    ]
    false_emitted = [
        decision for decision in false_decisions if decision.emitted
    ]
    target_best = _best_score_decision(target_decisions)
    false_best = _best_score_decision(false_decisions)
    target_same_batch_order_suppressed = [
        decision
        for decision in target_decisions
        if (
            not decision.emitted
            and decision.same_batch_inhibition > 0.0
            and (
                decision.effective_score
                + decision.same_batch_inhibition
                >= 1.0
            )
        )
    ]
    target_earlier_batch_suppressed = [
        decision
        for decision in target_decisions
        if (
            not decision.emitted
            and decision.earlier_batch_inhibition > 0.0
        )
    ]
    target_sharing_batch_with_earlier_false = [
        decision
        for decision in target_decisions
        if any(
            false.batch_index == decision.batch_index
            and (
                false.candidate.candidate.time,
                false.candidate.column_index,
                false.candidate.candidate.neuron_index,
                false.candidate.original_order,
            )
            < (
                decision.candidate.candidate.time,
                decision.candidate.column_index,
                decision.candidate.candidate.neuron_index,
                decision.candidate.original_order,
            )
            for false in false_decisions
        )
    ]
    provenance = _provenance_metrics(target_emitted, false_emitted)

    target_total_recall = _recall(raw_hits["all"], target["all"])
    emitted_total_recall = _recall(emitted_hits["all"], target["all"])
    target_survival_ratio = (
        len(emitted_hits["all"]) / len(raw_hits["all"])
        if raw_hits["all"]
        else 0.0
    )
    target_suppression_ratio = (
        1.0 - target_survival_ratio if raw_hits["all"] else 0.0
    )
    emitted_false_columns = emitted["all"] - target["all"]
    raw_false_columns = raw["all"] - target["all"]
    emitted_false_ratio = (
        len(emitted_false_columns) / len(emitted["all"])
        if emitted["all"]
        else 0.0
    )
    relative_error = (
        float(absolute_error) / abs(target_passenger)
        if absolute_error != "" and target_passenger
        else None
    )
    classification = classify_step(
        raw_target_recall=target_total_recall,
        emitted_target_recall=emitted_total_recall,
        passenger_recall=_recall(
            emitted_hits["passenger"], target["passenger"]
        ),
        target_survival_ratio=target_survival_ratio,
        emitted_false_ratio=emitted_false_ratio,
        relative_error=relative_error,
        provenance_available=bool(provenance["provenance_available"]),
        target_branch_coherence=provenance["target_branch_coherence"],
        target_score_max=_maximum(
            [decision.original_score for decision in target_decisions]
        ),
        false_score_max=_maximum(
            [decision.original_score for decision in false_decisions]
        ),
    )

    row: dict[str, object] = {
        "diagnostic_only": True,
        "uses_ground_truth_for_analysis_only": True,
        "ground_truth_does_not_affect_prediction": True,
        "prediction_input_index": prediction_input_index,
        "input_timestamp": input_timestamp,
        "horizon_step": horizon_step,
        "target_timestamp": target_timestamp,
        "target_passenger": target_passenger,
        "decoded_prediction": decoded_prediction,
        "absolute_error": absolute_error,
        "prediction_available": emitted_prediction is not None,
        "raw_candidate_neuron_count": len(decisions),
        "raw_candidate_column_count": len(
            {decision.candidate.column_index for decision in decisions}
        ),
        "emitted_candidate_neuron_count": sum(
            decision.emitted for decision in decisions
        ),
        "emitted_column_count": len(emitted["all"]),
        "inhibited_candidate_count": sum(
            not decision.emitted for decision in decisions
        ),
    }
    for field in ("weekday", "time", "passenger", "all"):
        label = "total" if field == "all" else field
        row[f"target_{label}_column_count"] = len(target[field])
        row[f"raw_target_{label}_hits"] = len(raw_hits[field])
        row[f"raw_target_{label}_recall"] = _recall(
            raw_hits[field], target[field]
        )
        row[f"emitted_target_{label}_hits"] = len(emitted_hits[field])
        row[f"emitted_target_{label}_recall"] = _recall(
            emitted_hits[field], target[field]
        )
        row[f"suppressed_target_{label}_columns"] = _columns_text(
            suppressed[field]
        )
    row.update(
        {
            "target_survival_ratio": target_survival_ratio,
            "target_suppression_ratio": target_suppression_ratio,
            "emitted_false_column_count": len(emitted_false_columns),
            "emitted_false_column_ratio": emitted_false_ratio,
            "emitted_false_columns": _columns_text(emitted_false_columns),
            "raw_false_column_count": len(raw_false_columns),
            "emitted_target_total_columns": _columns_text(
                emitted_hits["all"]
            ),
            "target_best_candidate_batch_index": (
                target_best.batch_index
                if target_best is not None
                else ""
            ),
            "false_best_candidate_batch_index": (
                false_best.batch_index
                if false_best is not None
                else ""
            ),
            "target_suppressed_only_due_to_same_batch_order": len(
                target_same_batch_order_suppressed
            ),
            "target_suppressed_by_earlier_batch": len(
                target_earlier_batch_suppressed
            ),
            "target_candidates_sharing_batch_with_earlier_false": len(
                target_sharing_batch_with_earlier_false
            ),
            "target_candidate_count": len(target_decisions),
            "target_candidate_score_mean": _mean(
                [decision.original_score for decision in target_decisions]
            ),
            "target_candidate_score_max": _maximum(
                [decision.original_score for decision in target_decisions]
            ),
            "target_candidate_score_min": _minimum(
                [decision.original_score for decision in target_decisions]
            ),
            "target_candidate_best_rank": min(target_ranks)
            if target_ranks
            else "",
            "target_candidate_median_rank": statistics.median(target_ranks)
            if target_ranks
            else "",
            "target_candidate_effective_score_mean": _mean(
                [decision.effective_score for decision in target_decisions]
            ),
            "target_candidate_inhibition_mean": _mean(
                [
                    decision.accumulated_inhibition
                    for decision in target_decisions
                ]
            ),
            "target_candidate_emitted_count": len(target_emitted),
            "target_candidate_suppressed_count": (
                len(target_decisions) - len(target_emitted)
            ),
            "false_candidate_score_mean": _mean(
                [decision.original_score for decision in false_decisions]
            ),
            "false_candidate_score_max": _maximum(
                [decision.original_score for decision in false_decisions]
            ),
            "false_candidate_effective_score_mean": _mean(
                [decision.effective_score for decision in false_decisions]
            ),
            "false_candidate_inhibition_mean": _mean(
                [
                    decision.accumulated_inhibition
                    for decision in false_decisions
                ]
            ),
            **provenance,
            "classification": classification,
        }
    )
    return row


def classify_step(
    *,
    raw_target_recall: float,
    emitted_target_recall: float,
    passenger_recall: float,
    target_survival_ratio: float,
    emitted_false_ratio: float,
    relative_error: float | None,
    provenance_available: bool,
    target_branch_coherence: object,
    target_score_max: float | str,
    false_score_max: float | str,
) -> str:
    """Apply documented diagnostic thresholds without affecting prediction."""

    thresholds = CLASSIFICATION_THRESHOLDS
    if raw_target_recall == 0.0:
        return "TARGET_ABSENT_FROM_RAW"
    if (
        target_survival_ratio
        < thresholds["mostly_suppressed_survival_ratio_below"]
    ):
        return "TARGET_PRESENT_BUT_SUPPRESSED"
    if (
        provenance_available
        and target_branch_coherence != ""
        and float(target_branch_coherence)
        < thresholds["branch_incoherent_coherence_below"]
    ):
        return "TARGET_BRANCH_INCOHERENT"
    if (
        passenger_recall >= thresholds["high_recall_at_or_above"]
        and relative_error is not None
        and relative_error > thresholds["decode_wrong_relative_error_above"]
    ):
        return "TARGET_SURVIVES_BUT_DECODE_WRONG"
    false_score_dominates = (
        target_score_max != ""
        and false_score_max != ""
        and float(false_score_max) > float(target_score_max)
    )
    if (
        emitted_false_ratio > thresholds["false_dominates_ratio_above"]
        or false_score_dominates
        or emitted_target_recall < thresholds["high_recall_at_or_above"]
    ):
        return "TARGET_SURVIVES_BUT_FALSE_DOMINATES"
    return "TARGET_PRESERVED"


def summarize_oracle_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    horizon: int,
) -> dict[str, object]:
    """Aggregate oracle survival metrics and first-loss steps."""

    step_summary: list[dict[str, object]] = []
    for step in range(1, horizon + 1):
        selected = [row for row in rows if int(row["horizon_step"]) == step]
        available = [row for row in selected if row["prediction_available"]]
        absolute_errors = [
            float(row["absolute_error"])
            for row in selected
            if row["absolute_error"] != ""
        ]
        targets = [
            abs(float(row["target_passenger"]))
            for row in selected
            if row["absolute_error"] != ""
        ]
        step_summary.append(
            {
                "horizon_step": step,
                "rows": len(selected),
                "coverage": len(available) / len(selected) if selected else 0.0,
                "mape": (
                    sum(absolute_errors) / sum(targets)
                    if absolute_errors and sum(targets)
                    else 0.0
                ),
                "raw_target_recall": _field_recall_summary(
                    selected, "raw"
                ),
                "emitted_target_recall": _field_recall_summary(
                    selected, "emitted"
                ),
                "target_suppression_ratio": _row_mean(
                    selected, "target_suppression_ratio"
                ),
                "false_emitted_ratio": _row_mean(
                    selected, "emitted_false_column_ratio"
                ),
                "target_absent_from_raw_rate": _classification_rate(
                    selected, "TARGET_ABSENT_FROM_RAW"
                ),
                "target_present_but_suppressed_rate": _classification_rate(
                    selected, "TARGET_PRESENT_BUT_SUPPRESSED"
                ),
                "target_survives_but_decode_wrong_rate": _classification_rate(
                    selected, "TARGET_SURVIVES_BUT_DECODE_WRONG"
                ),
                "target_branch_incoherent_rate": _classification_rate(
                    selected, "TARGET_BRANCH_INCOHERENT"
                ),
                "target_candidate_mean_rank": _row_mean(
                    selected, "target_candidate_median_rank"
                ),
                "target_vs_false_score_gap": _mean_gap(
                    selected,
                    "target_candidate_score_mean",
                    "false_candidate_score_mean",
                ),
                "target_vs_false_effective_score_gap": _mean_gap(
                    selected,
                    "target_candidate_effective_score_mean",
                    "false_candidate_effective_score_mean",
                ),
                "correlations": {
                    "target_recall_vs_absolute_error": _correlation(
                        selected,
                        "emitted_target_total_recall",
                        "absolute_error",
                    ),
                    "target_suppression_vs_absolute_error": _correlation(
                        selected,
                        "target_suppression_ratio",
                        "absolute_error",
                    ),
                    "branch_coherence_vs_absolute_error": _correlation(
                        selected,
                        "target_branch_coherence",
                        "absolute_error",
                    ),
                    "false_emitted_ratio_vs_absolute_error": _correlation(
                        selected,
                        "emitted_false_column_ratio",
                        "absolute_error",
                    ),
                },
            }
        )

    first_loss = Counter()
    by_input: dict[int, list[Mapping[str, object]]] = {}
    for row in rows:
        by_input.setdefault(int(row["prediction_input_index"]), []).append(row)
    for input_rows in by_input.values():
        lost = [
            int(row["horizon_step"])
            for row in sorted(
                input_rows, key=lambda item: int(item["horizon_step"])
            )
            if (
                float(row["emitted_target_total_recall"])
                < CLASSIFICATION_THRESHOLDS["mostly_suppressed_survival_ratio_below"]
                or (
                    row["provenance_available"]
                    and row["target_branch_coherence"] != ""
                    and float(row["target_branch_coherence"])
                    < CLASSIFICATION_THRESHOLDS[
                        "branch_incoherent_coherence_below"
                    ]
                )
            )
        ]
        first_loss[str(lost[0]) if lost else "not_lost"] += 1

    return {
        "diagnostic_only": True,
        "uses_ground_truth_for_analysis_only": True,
        "ground_truth_does_not_affect_prediction": True,
        "classification_thresholds": CLASSIFICATION_THRESHOLDS,
        "step_summary": step_summary,
        "first_clear_target_loss_step_distribution": dict(first_loss),
    }


def _target_score_ranks(
    decisions: Sequence[object],
    target_columns: set[int],
) -> list[int]:
    ordered = sorted(
        decisions,
        key=lambda decision: (
            -decision.original_score,  # type: ignore[attr-defined]
            decision.candidate.candidate.time,  # type: ignore[attr-defined]
            decision.candidate.column_index,  # type: ignore[attr-defined]
            decision.candidate.candidate.neuron_index,  # type: ignore[attr-defined]
            decision.candidate.original_order,  # type: ignore[attr-defined]
        ),
    )
    return [
        rank
        for rank, decision in enumerate(ordered, start=1)
        if decision.candidate.column_index in target_columns  # type: ignore[attr-defined]
    ]


def _best_score_decision(decisions: Sequence[object]) -> object | None:
    if not decisions:
        return None
    return min(
        decisions,
        key=lambda decision: (
            -decision.original_score,  # type: ignore[attr-defined]
            decision.candidate.candidate.time,  # type: ignore[attr-defined]
            decision.candidate.column_index,  # type: ignore[attr-defined]
            decision.candidate.candidate.neuron_index,  # type: ignore[attr-defined]
            decision.candidate.original_order,  # type: ignore[attr-defined]
        ),
    )


def _provenance_metrics(
    target_emitted: Sequence[object],
    false_emitted: Sequence[object],
) -> dict[str, object]:
    all_emitted = [*target_emitted, *false_emitted]
    target_keys = [_provenance_key(item) for item in target_emitted]
    false_keys = [_provenance_key(item) for item in false_emitted]
    all_keys = [*target_keys, *false_keys]
    provenance_available = bool(all_keys) and all(
        key is not None for key in all_keys
    )
    if not provenance_available:
        return {
            "provenance_available": False,
            "emitted_target_branch_count": "",
            "emitted_false_branch_count": "",
            "emitted_unique_source_fingerprint_count": "",
            "target_branch_coherence": "",
            "emitted_branch_coherence": "",
            "target_and_winner_same_branch_rate": "",
            "target_segment_diagnostic_ids": _segment_values(
                target_emitted, "diagnostic_id"
            ),
            "target_creation_sentence_indices": _segment_values(
                target_emitted, "creation_sentence_index"
            ),
            "target_creation_transition_indices": _segment_values(
                target_emitted, "creation_transition_index"
            ),
            "target_creation_target_columns": _segment_values(
                target_emitted, "creation_target_column"
            ),
            "target_creation_target_neurons": _segment_values(
                target_emitted, "creation_target_neuron"
            ),
            "target_creation_source_fingerprints": _segment_values(
                target_emitted, "creation_source_fingerprint"
            ),
            "target_source_cell_ids": _source_cell_values(target_emitted),
        }

    concrete_target = [key for key in target_keys if key is not None]
    concrete_false = [key for key in false_keys if key is not None]
    concrete_all = [key for key in all_keys if key is not None]
    all_counter = Counter(concrete_all)
    dominant_key, _count = all_counter.most_common(1)[0]
    return {
        "provenance_available": True,
        "emitted_target_branch_count": len(set(concrete_target)),
        "emitted_false_branch_count": len(set(concrete_false)),
        "emitted_unique_source_fingerprint_count": len(
            {key[-1] for key in concrete_all}
        ),
        "target_branch_coherence": _dominant_fraction(concrete_target),
        "emitted_branch_coherence": _dominant_fraction(concrete_all),
        "target_and_winner_same_branch_rate": (
            sum(key == dominant_key for key in concrete_target)
            / len(concrete_target)
            if concrete_target
            else 0.0
        ),
        "target_segment_diagnostic_ids": _segment_values(
            target_emitted, "diagnostic_id"
        ),
        "target_creation_sentence_indices": _segment_values(
            target_emitted, "creation_sentence_index"
        ),
        "target_creation_transition_indices": _segment_values(
            target_emitted, "creation_transition_index"
        ),
        "target_creation_target_columns": _segment_values(
            target_emitted, "creation_target_column"
        ),
        "target_creation_target_neurons": _segment_values(
            target_emitted, "creation_target_neuron"
        ),
        "target_creation_source_fingerprints": _segment_values(
            target_emitted, "creation_source_fingerprint"
        ),
        "target_source_cell_ids": _source_cell_values(target_emitted),
    }


def _provenance_key(item: object) -> tuple[object, ...] | None:
    segment = item.candidate.candidate.segment  # type: ignore[attr-defined]
    if segment.creation_source_fingerprint is None:
        return None
    return (
        segment.creation_sentence_index,
        segment.creation_transition_index,
        segment.creation_target_column,
        segment.creation_target_neuron,
        segment.creation_source_fingerprint,
    )


def _segment_values(items: Sequence[object], attribute: str) -> str:
    values = [
        getattr(item.candidate.candidate.segment, attribute)  # type: ignore[attr-defined]
        for item in items
    ]
    return "|".join("" if value is None else str(value) for value in values)


def _source_cell_values(items: Sequence[object]) -> str:
    return "|".join(
        " ".join(
            map(
                str,
                sorted(
                    item.candidate.candidate.segment.synapses  # type: ignore[attr-defined]
                ),
            )
        )
        for item in items
    )


def _dominant_fraction(keys: Sequence[tuple[object, ...]]) -> float:
    if not keys:
        return 0.0
    return Counter(keys).most_common(1)[0][1] / len(keys)


def _field_recall_summary(
    rows: Sequence[Mapping[str, object]],
    prefix: str,
) -> dict[str, float]:
    return {
        field: _row_mean(rows, f"{prefix}_target_{field}_recall")
        for field in ("weekday", "time", "passenger", "total")
    }


def _classification_rate(
    rows: Sequence[Mapping[str, object]],
    classification: str,
) -> float:
    return (
        sum(row["classification"] == classification for row in rows) / len(rows)
        if rows
        else 0.0
    )


def _row_mean(rows: Sequence[Mapping[str, object]], field: str) -> float:
    values = [
        float(row[field]) for row in rows if row.get(field, "") != ""
    ]
    return sum(values) / len(values) if values else 0.0


def _mean_gap(
    rows: Sequence[Mapping[str, object]],
    left: str,
    right: str,
) -> float:
    gaps = [
        float(row[left]) - float(row[right])
        for row in rows
        if row.get(left, "") != "" and row.get(right, "") != ""
    ]
    return sum(gaps) / len(gaps) if gaps else 0.0


def _correlation(
    rows: Sequence[Mapping[str, object]],
    left: str,
    right: str,
) -> float:
    pairs = [
        (float(row[left]), float(row[right]))
        for row in rows
        if row.get(left, "") != "" and row.get(right, "") != ""
    ]
    if len(pairs) < 2:
        return 0.0
    left_mean = sum(pair[0] for pair in pairs) / len(pairs)
    right_mean = sum(pair[1] for pair in pairs) / len(pairs)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in pairs
    )
    left_variance = sum((x - left_mean) ** 2 for x, _y in pairs)
    right_variance = sum((y - right_mean) ** 2 for _x, y in pairs)
    if left_variance == 0.0 or right_variance == 0.0:
        return 0.0
    return numerator / math.sqrt(left_variance * right_variance)


def _columns_text(columns: set[int]) -> str:
    return " ".join(map(str, sorted(columns)))


def _recall(hits: set[int], target: set[int]) -> float:
    return len(hits) / len(target) if target else 0.0


def _mean(values: Sequence[float]) -> float | str:
    return sum(values) / len(values) if values else ""


def _maximum(values: Sequence[float]) -> float | str:
    return max(values) if values else ""


def _minimum(values: Sequence[float]) -> float | str:
    return min(values) if values else ""
