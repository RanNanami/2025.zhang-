"""Causal time semantics and post-hoc metrics for the Fig.9 anchor audit.

This module deliberately contains no model calls.  A prediction is produced
without a target, then the helpers here attach the appropriate timestamp and
score it.  Keeping that boundary explicit makes future-value leakage easy to
test and audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import sqrt
from statistics import median
from typing import Iterable, Sequence

from .data import TaxiRecord
from .metrics import mape as repository_mape


CELL_ORDER = ("A", "B", "C", "D")
CELL_NAMES = {
    "A": "CURRENT_BUGGED",
    "B": "OBSERVE_CURRENT_THEN_5",
    "C": "REALIGN_TARGET_ONLY",
    "D": "SIX_STEP_SAME_TARGET",
}


@dataclass(frozen=True)
class AnchorCell:
    """One isolated prediction-anchor interpretation."""

    cell: str
    name: str
    observe_current: bool
    rollout_steps: int
    causal_endpoint_offset: int
    target_offset: int


CELLS = {
    "A": AnchorCell("A", CELL_NAMES["A"], False, 5, 4, 5),
    "B": AnchorCell("B", CELL_NAMES["B"], True, 5, 5, 5),
    "C": AnchorCell("C", CELL_NAMES["C"], False, 5, 4, 4),
    "D": AnchorCell("D", CELL_NAMES["D"], False, 6, 5, 5),
}


def cell_timing(
    records: Sequence[TaxiRecord],
    anchor_index: int,
    cell: str,
) -> dict[str, object]:
    """Return the causal timestamps for a cell without reading target values."""

    spec = CELLS[cell]
    if anchor_index <= 0:
        raise ValueError("an anchor requires an observed record at t-1")
    if anchor_index + spec.target_offset >= len(records):
        raise ValueError("the evaluation target is outside the supplied records")
    labelled = records[anchor_index].timestamp
    before = records[anchor_index - 1].timestamp
    rollout_origin = labelled if spec.observe_current else before
    endpoint = records[anchor_index + spec.causal_endpoint_offset].timestamp
    target = records[anchor_index + spec.target_offset].timestamp
    return {
        "anchor_index": anchor_index,
        "labelled_anchor_timestamp": labelled,
        "last_observed_timestamp_before_cell": before,
        "last_observed_timestamp_for_rollout": rollout_origin,
        "causal_endpoint_timestamp": endpoint,
        "evaluation_target_timestamp": target,
        "causal_offset_minutes": int((target - endpoint).total_seconds() / 60),
        "target_offset_from_label_minutes": int(
            (target - labelled).total_seconds() / 60
        ),
        "last_observed_to_target_minutes": int(
            (target - rollout_origin).total_seconds() / 60
        ),
    }


def score_prediction(
    *,
    records: Sequence[TaxiRecord],
    anchor_index: int,
    cell: str,
    prediction: float | None,
    raw_event_counts: Sequence[int] = (),
    raw_column_counts: Sequence[int] = (),
    pre_model_fingerprint: str = "",
    pre_rng_fingerprint: str = "",
    pre_transient_fingerprint: str = "",
) -> dict[str, object]:
    """Attach ground truth after prediction and return an auditable row."""

    spec = CELLS[cell]
    timing = cell_timing(records, anchor_index, cell)
    target_record = records[anchor_index + spec.target_offset]
    absolute_error = (
        abs(prediction - target_record.value) if prediction is not None else None
    )
    standard_ape = (
        absolute_error / abs(target_record.value)
        if absolute_error is not None and target_record.value != 0.0
        else None
    )
    return {
        "anchor_index": anchor_index,
        "anchor_ordinal": anchor_index + 1,
        "labelled_anchor_timestamp": _format_time(
            timing["labelled_anchor_timestamp"]
        ),
        "last_observed_timestamp_before_cell": _format_time(
            timing["last_observed_timestamp_before_cell"]
        ),
        "last_observed_timestamp_for_rollout": _format_time(
            timing["last_observed_timestamp_for_rollout"]
        ),
        "cell": cell,
        "cell_name": spec.name,
        "current_record_observed": spec.observe_current,
        "rollout_steps": spec.rollout_steps,
        "causal_endpoint_timestamp": _format_time(
            timing["causal_endpoint_timestamp"]
        ),
        "evaluation_target_timestamp": _format_time(
            timing["evaluation_target_timestamp"]
        ),
        "causal_offset_minutes": timing["causal_offset_minutes"],
        "target_offset_from_label_minutes": timing[
            "target_offset_from_label_minutes"
        ],
        "last_observed_to_target_minutes": timing[
            "last_observed_to_target_minutes"
        ],
        "prediction": prediction if prediction is not None else "",
        "ground_truth": target_record.value,
        "abs_error": absolute_error if absolute_error is not None else "",
        "standard_APE": standard_ape if standard_ape is not None else "",
        "current_repo_metric_contribution": "",
        "raw_rollout_event_counts": " ".join(map(str, raw_event_counts)),
        "raw_rollout_column_counts": " ".join(map(str, raw_column_counts)),
        "pre_model_fingerprint": pre_model_fingerprint,
        "pre_rng_fingerprint": pre_rng_fingerprint,
        "pre_transient_fingerprint": pre_transient_fingerprint,
        "in_common_anchor_set": False,
    }


def finalize_prediction_rows(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], set[int]]:
    """Mark the common valid anchors and add repository-metric contributions."""

    valid_by_anchor: dict[int, set[str]] = {}
    for row in rows:
        if row["prediction"] != "":
            valid_by_anchor.setdefault(int(row["anchor_index"]), set()).add(
                str(row["cell"])
            )
    common = {
        anchor
        for anchor, cells in valid_by_anchor.items()
        if cells == set(CELL_ORDER)
    }
    denominators = {
        cell: sum(
            abs(float(row["ground_truth"]))
            for row in rows
            if row["cell"] == cell and int(row["anchor_index"]) in common
        )
        for cell in CELL_ORDER
    }
    for row in rows:
        included = int(row["anchor_index"]) in common
        row["in_common_anchor_set"] = included
        if included and row["abs_error"] != "":
            denominator = denominators[str(row["cell"])]
            row["current_repo_metric_contribution"] = (
                float(row["abs_error"]) / denominator if denominator else 0.0
            )
    return rows, common


def standard_mape(predictions: Iterable[float], targets: Iterable[float]) -> float:
    """Mean pointwise percentage error, omitting zero-valued targets."""

    apes = [
        abs(prediction - target) / abs(target)
        for prediction, target in zip(predictions, targets)
        if target != 0.0
    ]
    return sum(apes) / len(apes) if apes else 0.0


def summarize_cells(
    rows: Sequence[dict[str, object]],
    common: set[int],
) -> list[dict[str, object]]:
    """Summarize all cells on the exact same set of valid anchors."""

    attempted = len({int(row["anchor_index"]) for row in rows})
    summaries: list[dict[str, object]] = []
    for cell in CELL_ORDER:
        all_cell_rows = [row for row in rows if row["cell"] == cell]
        cell_rows = [
            row
            for row in all_cell_rows
            if int(row["anchor_index"]) in common and row["prediction"] != ""
        ]
        predictions = [float(row["prediction"]) for row in cell_rows]
        targets = [float(row["ground_truth"]) for row in cell_rows]
        errors = [abs(left - right) for left, right in zip(predictions, targets)]
        apes = [
            error / abs(target)
            for error, target in zip(errors, targets)
            if target != 0.0
        ]
        valid_count = sum(row["prediction"] != "" for row in all_cell_rows)
        summaries.append(
            {
                "cell": cell,
                "cell_name": CELL_NAMES[cell],
                "predictions": valid_count,
                "metrics_anchor_count": len(cell_rows),
                "coverage": valid_count / attempted if attempted else 0.0,
                "standard_mape": sum(apes) / len(apes) if apes else 0.0,
                "repo_metric": repository_mape(predictions, targets),
                "mae": sum(errors) / len(errors) if errors else 0.0,
                "median_ape": median(apes) if apes else 0.0,
                "p90_ape": percentile(apes, 0.90),
                "max_ape": max(apes, default=0.0),
                "zero_target_count": sum(target == 0.0 for target in targets),
                "relative_improvement_vs_A": 0.0,
                "causal_endpoint_matches_target": all(
                    int(row["causal_offset_minutes"]) == 0
                    for row in cell_rows
                ),
                "current_record_observed": CELLS[cell].observe_current,
                "rollout_steps": CELLS[cell].rollout_steps,
            }
        )
    baseline_standard = float(summaries[0]["standard_mape"])
    for summary in summaries[1:]:
        value = float(summary["standard_mape"])
        summary["relative_improvement_vs_A"] = relative_improvement(
            baseline_standard, value
        )
    return summaries


def pairwise_effects(
    rows: Sequence[dict[str, object]],
    summaries: Sequence[dict[str, object]],
    common: set[int],
) -> list[dict[str, object]]:
    """Compare every pair using pointwise APE and both aggregate metrics."""

    by_cell_anchor = {
        (str(row["cell"]), int(row["anchor_index"])): row
        for row in rows
        if int(row["anchor_index"]) in common
    }
    summary_by_cell = {str(row["cell"]): row for row in summaries}
    output: list[dict[str, object]] = []
    for left_index, left in enumerate(CELL_ORDER):
        for right in CELL_ORDER[left_index + 1 :]:
            differences: list[float] = []
            right_wins = ties = right_losses = 0
            for anchor in sorted(common):
                left_ape = by_cell_anchor[(left, anchor)]["standard_APE"]
                right_ape = by_cell_anchor[(right, anchor)]["standard_APE"]
                if left_ape == "" or right_ape == "":
                    continue
                difference = float(right_ape) - float(left_ape)
                differences.append(difference)
                if abs(difference) <= 1e-15:
                    ties += 1
                elif difference < 0.0:
                    right_wins += 1
                else:
                    right_losses += 1
            left_summary = summary_by_cell[left]
            right_summary = summary_by_cell[right]
            output.append(
                {
                    "comparison": f"{left}_vs_{right}",
                    "left_cell": left,
                    "right_cell": right,
                    "right_minus_left_mean_APE": (
                        sum(differences) / len(differences) if differences else 0.0
                    ),
                    "right_minus_left_median_APE": (
                        median(differences) if differences else 0.0
                    ),
                    "right_win_count": right_wins,
                    "tie_count": ties,
                    "right_loss_count": right_losses,
                    "relative_standard_mape_improvement": relative_improvement(
                        float(left_summary["standard_mape"]),
                        float(right_summary["standard_mape"]),
                    ),
                    "relative_repo_metric_improvement": relative_improvement(
                        float(left_summary["repo_metric"]),
                        float(right_summary["repo_metric"]),
                    ),
                }
            )
    return output


def anchor_level_rows(
    rows: Sequence[dict[str, object]],
    records: Sequence[TaxiRecord],
    common: set[int],
) -> list[dict[str, object]]:
    """Pivot cell predictions and attach post-hoc local-change features."""

    by_cell_anchor = {
        (str(row["cell"]), int(row["anchor_index"])): row
        for row in rows
        if int(row["anchor_index"]) in common
    }
    output: list[dict[str, object]] = []
    for anchor in sorted(common):
        local_change = abs(records[anchor + 5].value - records[anchor + 4].value)
        relative_change = local_change / max(abs(records[anchor + 5].value), 1e-12)
        row: dict[str, object] = {
            "anchor_index": anchor,
            "labelled_anchor_timestamp": _format_time(records[anchor].timestamp),
            "local_target_change": local_change,
            "relative_local_target_change": relative_change,
            "time_of_day": time_of_day(records[anchor].timestamp),
        }
        best_cell = ""
        best_ape = float("inf")
        for cell in CELL_ORDER:
            source = by_cell_anchor[(cell, anchor)]
            row[f"{cell}_prediction"] = source["prediction"]
            row[f"{cell}_ground_truth"] = source["ground_truth"]
            row[f"{cell}_abs_error"] = source["abs_error"]
            row[f"{cell}_standard_APE"] = source["standard_APE"]
            if source["standard_APE"] != "" and float(source["standard_APE"]) < best_ape:
                best_ape = float(source["standard_APE"])
                best_cell = cell
        row["best_cell"] = best_cell
        row["A_minus_B_APE"] = float(row["A_standard_APE"]) - float(
            row["B_standard_APE"]
        )
        output.append(row)
    return output


def local_change_analysis(
    anchor_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Correlate A's excess error over B with target movement."""

    absolute_change = [float(row["local_target_change"]) for row in anchor_rows]
    relative_change = [
        float(row["relative_local_target_change"]) for row in anchor_rows
    ]
    excess = [float(row["A_minus_B_APE"]) for row in anchor_rows]
    return {
        "pearson_absolute_change_vs_A_minus_B_APE": pearson(
            absolute_change, excess
        ),
        "spearman_absolute_change_vs_A_minus_B_APE": spearman(
            absolute_change, excess
        ),
        "pearson_relative_change_vs_A_minus_B_APE": pearson(
            relative_change, excess
        ),
        "spearman_relative_change_vs_A_minus_B_APE": spearman(
            relative_change, excess
        ),
    }


def quartile_rows(
    anchor_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Summarize cells in deterministic local-change quartiles."""

    ordered = sorted(
        anchor_rows,
        key=lambda row: (float(row["local_target_change"]), int(row["anchor_index"])),
    )
    buckets: dict[str, list[dict[str, object]]] = {f"Q{i}": [] for i in range(1, 5)}
    for rank, row in enumerate(ordered):
        quartile = min(4, int(rank * 4 / max(1, len(ordered))) + 1)
        buckets[f"Q{quartile}"].append(row)
    output: list[dict[str, object]] = []
    for quartile, bucket in buckets.items():
        summary: dict[str, object] = {
            "quartile": quartile,
            "anchors": len(bucket),
            "min_local_target_change": min(
                (float(row["local_target_change"]) for row in bucket), default=0.0
            ),
            "max_local_target_change": max(
                (float(row["local_target_change"]) for row in bucket), default=0.0
            ),
        }
        for cell in CELL_ORDER:
            apes = [float(row[f"{cell}_standard_APE"]) for row in bucket]
            summary[f"{cell}_standard_mape"] = (
                sum(apes) / len(apes) if apes else 0.0
            )
        output.append(summary)
    return output


def timeofday_rows(
    anchor_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Summarize errors in fixed, target-independent time bands."""

    output: list[dict[str, object]] = []
    for band in ("night", "morning_rise", "midday", "evening"):
        selected = [row for row in anchor_rows if row["time_of_day"] == band]
        summary: dict[str, object] = {"time_of_day": band, "anchors": len(selected)}
        for cell in CELL_ORDER:
            apes = [float(row[f"{cell}_standard_APE"]) for row in selected]
            summary[f"{cell}_standard_mape"] = (
                sum(apes) / len(apes) if apes else 0.0
            )
            summary[f"{cell}_mae"] = (
                sum(float(row[f"{cell}_abs_error"]) for row in selected)
                / len(selected)
                if selected
                else 0.0
            )
        output.append(summary)
    return output


def effect_classification(relative_improvement_value: float) -> str:
    """Classify the requested diagnostic effect size."""

    if relative_improvement_value >= 0.40:
        return "VERY_LARGE"
    if relative_improvement_value >= 0.20:
        return "LARGE"
    if relative_improvement_value >= 0.10:
        return "MODERATE"
    if relative_improvement_value >= 0.02:
        return "SMALL"
    return "NEGLIGIBLE"


def select_conclusion(
    summaries: Sequence[dict[str, object]],
    *,
    prerequisites_met: bool,
) -> tuple[str, str]:
    """Choose one bounded conclusion and one next step from the allowed sets."""

    if not prerequisites_met:
        return "ANCHOR_ABCD_RESULT_INCONCLUSIVE", "DO_NOT_CHANGE_PROTOCOL_YET"
    by_cell = {str(row["cell"]): row for row in summaries}
    improvements = {
        cell: float(by_cell[cell]["relative_improvement_vs_A"])
        for cell in ("B", "C", "D")
    }
    best = max(improvements, key=improvements.get)
    best_improvement = improvements[best]
    if best_improvement < 0.02:
        return (
            "ANCHOR_FIX_DOES_NOT_EXPLAIN_PAPER_GAP",
            "AUDIT_EVALUATION_WINDOW_AND_MAPE",
        )
    if best_improvement < 0.10:
        return (
            "ANCHOR_MISMATCH_HAS_SMALL_NUMERICAL_EFFECT",
            "AUDIT_EVALUATION_WINDOW_AND_MAPE",
        )
    if best_improvement < 0.20:
        return (
            "ANCHOR_MISMATCH_IS_MODERATE_MAPE_GAP_SOURCE",
            "DO_NOT_CHANGE_PROTOCOL_YET",
        )
    conclusions = {
        "B": "OBSERVING_CURRENT_RECORD_IS_PRIMARY_CORRECTION",
        "C": "TARGET_REALIGNMENT_IS_PRIMARY_CORRECTION",
        "D": "EXTRA_CAUSAL_STEP_IS_PRIMARY_CORRECTION",
    }
    if best == "B":
        next_step = "ADOPT_OBSERVE_CURRENT_FIVE_STEP_PROTOCOL"
    elif best == "C":
        next_step = "ADOPT_CAUSALLY_REALIGNED_TARGET_PROTOCOL"
    else:
        next_step = "DO_NOT_CHANGE_PROTOCOL_YET"
    return conclusions[best], next_step


def relative_improvement(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline if baseline else 0.0


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_scale = sum((value - left_mean) ** 2 for value in left)
    right_scale = sum((value - right_mean) ** 2 for value in right)
    if left_scale == 0.0 or right_scale == 0.0:
        return 0.0
    return numerator / sqrt(left_scale * right_scale)


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    return pearson(_average_ranks(left), _average_ranks(right))


def time_of_day(timestamp: datetime) -> str:
    if timestamp.hour < 6:
        return "night"
    if timestamp.hour < 10:
        return "morning_rise"
    if timestamp.hour < 17:
        return "midday"
    return "evening"


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and ordered[stop][1] == ordered[start][1]:
            stop += 1
        average = (start + 1 + stop) / 2.0
        for position in range(start, stop):
            ranks[ordered[position][0]] = average
        start = stop
    return ranks


def _format_time(value: object) -> str:
    if not isinstance(value, datetime):
        raise TypeError("expected a datetime")
    return value.isoformat(sep=" ")


def timeline_steps(
    records: Sequence[TaxiRecord], anchor_index: int, cell: str
) -> list[str]:
    """Return the concrete autonomous timestamps used in a report example."""

    spec = CELLS[cell]
    first_index = anchor_index + 1 if spec.observe_current else anchor_index
    return [
        _format_time(records[first_index + step].timestamp)
        for step in range(spec.rollout_steps)
    ]


def expected_interval() -> timedelta:
    """Expose the fixed sampling interval for tests and report provenance."""

    return timedelta(minutes=30)
