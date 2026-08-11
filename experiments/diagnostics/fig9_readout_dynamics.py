"""Read-only Fig.9 readout dynamics trace helpers.

The trace follows one predicted column from the same ``predict_code`` call
through local selection, optional inter-column competition, and emission.
It is deliberately post-hoc: target labels are attached only after the
prediction has already been produced, and this module never calls model code.
"""

from __future__ import annotations

import csv
import gzip
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from seqmem.encoding import SymbolCode


READOUT_TRACE_FIELDS = [
    "run_id",
    "trajectory_kind",
    "record_index",
    "anchor_index",
    "horizon_step",
    "field",
    "column_id",
    "column_event_id",
    "is_target_column",
    "target_label_source",
    "pre_competition_present",
    "pre_selector_candidate_count",
    "pre_selector_neuron_count",
    "pre_selector_segment_count",
    "candidate_score_max",
    "candidate_score_second",
    "candidate_score_mean",
    "candidate_score_margin",
    "response_peak_max",
    "contributor_count_if_available",
    "predicted_time_min",
    "predicted_time_max",
    "predicted_time_spread",
    "existing_selected_neuron",
    "existing_selected_segment",
    "existing_selected_score",
    "existing_selected_response_peak",
    "existing_selected_predicted_time",
    "existing_selected_contributor_count",
    "maxscore_selected_neuron",
    "maxscore_selected_segment",
    "maxscore_selected_score",
    "maxscore_selected_response_peak",
    "maxscore_selected_predicted_time",
    "maxscore_selected_contributor_count",
    "selected_neuron",
    "selected_segment",
    "selected_score",
    "selected_response_peak",
    "selected_predicted_time",
    "selected_contributor_count",
    "selector_changed_neuron",
    "selector_changed_segment",
    "selector_changed_score",
    "selector_score_delta",
    "selector_response_peak_delta",
    "selector_predicted_time_delta",
    "selector_changed_any",
    "selection_policy",
    "competition_enabled",
    "competition_bucket",
    "same_bucket_candidate_count",
    "prior_competitor_count",
    "inhibition_received",
    "effective_score",
    "competition_survived",
    "competition_suppressed",
    "emitted",
    "competition_reason",
    "suppression_margin",
]


def column_event_id(
    record_index: int,
    anchor_index: int,
    horizon_step: int,
    field: str,
    candidate_column: int,
    trajectory_kind: str = "autonomous_rollout",
) -> str:
    """Return a stable identity made only from observable event coordinates."""

    return (
        f"{trajectory_kind}:r{int(record_index)}:a{int(anchor_index)}:"
        f"h{int(horizon_step)}:{field}:c{int(candidate_column)}"
    )


def _number(value: object) -> float | None:
    if value in (None, "", "NA", "null", "None"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: object) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _value(value: object) -> object:
    """Serialize missing diagnostic values explicitly rather than as zero."""

    if value is None:
        return "NA"
    if isinstance(value, float) and not math.isfinite(value):
        return "NA"
    return value


def _delta(left: object, right: object) -> float | None:
    left_number = _number(left)
    right_number = _number(right)
    if left_number is None or right_number is None:
        return None
    return left_number - right_number


def _bool(value: object) -> bool | None:
    if value in (None, "", "NA", "null"):
        return None
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes"}


def _candidate_columns(
    raw_code: SymbolCode | None,
    selection_rows: Sequence[Mapping[str, object]],
) -> list[int]:
    columns = {int(row["column"]) for row in selection_rows if row.get("column") not in (None, "")}
    if raw_code is not None:
        columns.update(int(event.column) for event in raw_code.events)
    return sorted(columns)


def _target_columns_by_field(
    target_code: SymbolCode | None,
    field_for_column: Callable[[int], str],
) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    if target_code is None:
        return result
    for event in target_code.events:
        field = field_for_column(int(event.column))
        result.setdefault(field, set()).add(int(event.column))
    return result


def build_readout_column_rows(
    *,
    run_id: str,
    record_index: int,
    anchor_index: int,
    horizon_step: int,
    field_for_column: Callable[[int], str],
    selection_rows: Sequence[Mapping[str, object]],
    raw_code: SymbolCode | None,
    competition_result: object | None,
    target_code: SymbolCode | None,
    trajectory_kind: str = "autonomous_rollout",
) -> list[dict[str, object]]:
    """Build O(candidate columns) rows from already-computed readout objects."""

    from experiments.diagnostics.fig9_competitive_inhibition import CompetitionResult

    selections = {
        int(row["column"]): row
        for row in selection_rows
        if row.get("column") not in (None, "")
    }
    target_columns = _target_columns_by_field(target_code, field_for_column)
    raw_columns = set(
        int(event.column) for event in raw_code.events
    ) if raw_code is not None else set()
    decisions = {}
    batches = {}
    competition_enabled = isinstance(competition_result, CompetitionResult)
    if competition_enabled:
        decisions = {
            int(item.candidate.column_index): item
            for item in competition_result.decisions
        }
        batches = {
            int(batch.batch_index): batch
            for batch in competition_result.batches
        }

    rows: list[dict[str, object]] = []
    for column in _candidate_columns(raw_code, selection_rows):
        field = field_for_column(column)
        selection = selections.get(column, {})
        existing_score = selection.get("existing_selected_score", selection.get("existing_policy_score"))
        existing_neuron = selection.get("existing_selected_neuron", selection.get("existing_policy_neuron"))
        existing_segment = selection.get("existing_selected_segment", selection.get("existing_policy_segment_id"))
        existing_peak = selection.get("existing_selected_response_peak")
        existing_time = selection.get("existing_selected_predicted_time")
        existing_contributors = selection.get("existing_selected_contributor_count")
        max_score = selection.get("maxscore_selected_score")
        max_neuron = selection.get("maxscore_selected_neuron")
        max_segment = selection.get("maxscore_selected_segment")
        max_peak = selection.get("maxscore_selected_response_peak")
        max_time = selection.get("maxscore_selected_predicted_time")
        max_contributors = selection.get("maxscore_selected_contributor_count")
        selected_score = selection.get("selected_candidate_score")
        selected_neuron = selection.get("selected_neuron")
        selected_segment = selection.get("selected_segment_id")
        selected_peak = selection.get("selected_response_peak")
        selected_time = selection.get("selected_predicted_time")
        selected_contributors = selection.get("selected_contributor_count")

        decision = decisions.get(column)
        if competition_enabled and decision is not None:
            candidate = decision.candidate.candidate
            bucket = decision.batch_index
            batch = batches.get(bucket)
            prior_count = len(decision.winning_predecessor_ids)
            inhibition = decision.accumulated_inhibition
            effective_score = decision.effective_score
            survived = decision.emitted
            suppressed = not decision.emitted
            emitted = decision.emitted
            reason = decision.reason
            # The implementation does not expose a separate suppression
            # margin.  Do not relabel effective_score as a new metric.
            suppression_margin = None
            same_bucket_count = batch.candidate_count if batch is not None else None
        elif competition_enabled:
            bucket = None
            prior_count = None
            inhibition = None
            effective_score = None
            survived = None
            suppressed = None
            emitted = False
            reason = "no_competition_decision"
            suppression_margin = None
            same_bucket_count = None
        else:
            bucket = None
            prior_count = None
            inhibition = 0.0
            effective_score = selected_score
            survived = True if column in raw_columns else None
            suppressed = False if column in raw_columns else None
            emitted = column in raw_columns
            reason = "competition_disabled"
            suppression_margin = None
            same_bucket_count = None

        is_target = column in target_columns.get(field, set())
        row = {
            "run_id": run_id,
            "trajectory_kind": trajectory_kind,
            "record_index": record_index,
            "anchor_index": anchor_index,
            "horizon_step": horizon_step,
            "field": field,
            "column_id": column,
            "column_event_id": column_event_id(
                record_index, anchor_index, horizon_step, field, column, trajectory_kind
            ),
            "is_target_column": is_target,
            "target_label_source": "posthoc_future_code",
            "pre_competition_present": column in raw_columns,
            "pre_selector_candidate_count": selection.get("pre_selector_candidate_count", selection.get("group_candidate_count")),
            "pre_selector_neuron_count": selection.get("pre_selector_neuron_count", selection.get("within_column_candidate_neuron_count")),
            "pre_selector_segment_count": selection.get("pre_selector_segment_count", selection.get("within_column_candidate_segment_count")),
            "candidate_score_max": selection.get("candidate_score_max", selection.get("local_top1_score")),
            "candidate_score_second": selection.get("candidate_score_second", selection.get("local_top2_score")),
            "candidate_score_mean": selection.get("candidate_score_mean"),
            "candidate_score_margin": selection.get("candidate_score_margin", selection.get("local_score_margin")),
            "response_peak_max": selection.get("response_peak_max"),
            "contributor_count_if_available": selection.get("contributor_count_if_available"),
            "predicted_time_min": selection.get("predicted_time_min"),
            "predicted_time_max": selection.get("predicted_time_max"),
            "predicted_time_spread": selection.get("predicted_time_spread"),
            "existing_selected_neuron": existing_neuron,
            "existing_selected_segment": existing_segment,
            "existing_selected_score": existing_score,
            "existing_selected_response_peak": existing_peak,
            "existing_selected_predicted_time": existing_time,
            "existing_selected_contributor_count": existing_contributors,
            "maxscore_selected_neuron": max_neuron,
            "maxscore_selected_segment": max_segment,
            "maxscore_selected_score": max_score,
            "maxscore_selected_response_peak": max_peak,
            "maxscore_selected_predicted_time": max_time,
            "maxscore_selected_contributor_count": max_contributors,
            "selector_changed_neuron": _bool(max_neuron != existing_neuron),
            "selector_changed_segment": _bool(max_segment != existing_segment),
            "selector_changed_score": _bool(_delta(max_score, existing_score) not in (None, 0.0)),
            "selector_score_delta": _delta(max_score, existing_score),
            "selector_response_peak_delta": _delta(max_peak, existing_peak),
            "selector_predicted_time_delta": _delta(max_time, existing_time),
            "selector_changed_any": _bool(
                max_neuron != existing_neuron
                or max_segment != existing_segment
                or _delta(max_score, existing_score) not in (None, 0.0)
            ),
            "selection_policy": selection.get("policy", "NA"),
            "competition_enabled": competition_enabled,
            "competition_bucket": bucket,
            "same_bucket_candidate_count": same_bucket_count,
            "prior_competitor_count": prior_count,
            "inhibition_received": inhibition,
            "effective_score": effective_score,
            "competition_survived": survived,
            "competition_suppressed": suppressed,
            "emitted": emitted,
            "competition_reason": reason,
            "suppression_margin": suppression_margin,
            # The real selector is the candidate that survives local selection;
            # retaining these names makes the online/offline comparison explicit.
            "selected_neuron": selected_neuron,
            "selected_segment": selected_segment,
            "selected_score": selected_score,
            "selected_response_peak": selected_peak,
            "selected_predicted_time": selected_time,
            "selected_contributor_count": selected_contributors,
        }
        rows.append({field_name: _value(row.get(field_name)) for field_name in READOUT_TRACE_FIELDS})
    return rows


def write_readout_trace(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Write a schema-stable gzip CSV with explicit ``NA`` missing values."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=READOUT_TRACE_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _value(row.get(field)) for field in READOUT_TRACE_FIELDS})


def read_readout_trace(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
