"""Offline within-column loss analysis for pre-existing Fig.9 references."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence


DIAGNOSTIC_MARKERS = {
    "diagnostic_only": True,
    "offline_analysis_only": True,
    "teacher_forced_reference_only": True,
    "target_column_is_oracle_conditioned_for_analysis": True,
    "not_a_deployable_prediction_result": True,
    "reference_neuron_selection_diagnostic": True,
    "ground_truth_does_not_affect_prediction": True,
    "diagnostic_trace_does_not_affect_selection": True,
    "future_observation_does_not_affect_past_prediction": True,
}


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _integer(value: object) -> int | None:
    try:
        if value in ("", None):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _find_trace(run_dir: Path, stem: str) -> Path:
    for suffix in (".csv.gz", ".csv"):
        candidate = run_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing {stem}.csv[.gz] in {run_dir}")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _winner_set(row: Mapping[str, object]) -> set[int]:
    return {
        int(value)
        for value in str(row.get("reference_winner_neurons", "")).split()
    }


def _key(row: Mapping[str, object]) -> tuple[int, str, int, int]:
    return (
        int(row["input_index"]),
        str(row["target_timestamp"]),
        int(row["horizon_step"]),
        int(row["target_column"]),
    )


def build_reference_selection_trace(
    identity_rows: Sequence[Mapping[str, object]],
    segment_rows: Sequence[Mapping[str, object]],
    *,
    input_timestamps: Mapping[int, str] | None = None,
) -> list[dict[str, object]]:
    """Join valid future references to the real saved selection stages."""

    references = {
        _key(row): row
        for row in identity_rows
        if _bool(row.get("neuron_reference_valid_before_observation"))
    }
    input_timestamps = input_timestamps or {}
    by_key: dict[
        tuple[int, str, int, int],
        list[Mapping[str, object]],
    ] = defaultdict(list)
    for row in segment_rows:
        key = _key(row)
        if key in references:
            by_key[key].append(row)
    output: list[dict[str, object]] = []
    for key, reference in sorted(references.items()):
        winners = _winner_set(reference)
        rows = by_key.get(key, [])
        by_segment = {
            str(row.get("segment_provenance_id", "")): row for row in rows
        }
        for row in rows:
            segment_id = str(row.get("segment_provenance_id", ""))
            winner_id = str(row.get("winner_segment_id", ""))
            winner = by_segment.get(winner_id)
            is_reference_neuron = (
                _integer(row.get("target_neuron")) in winners
            )
            reference_segment_id = str(
                reference.get("teacher_forced_segment_id", "")
            )
            output.append(
                {
                    **DIAGNOSTIC_MARKERS,
                    "policy": row.get("policy", reference.get("policy", "")),
                    "input_index": key[0],
                    "input_timestamp": input_timestamps.get(key[0], ""),
                    "target_record_index": reference.get(
                        "target_record_index",
                        "",
                    ),
                    "target_timestamp": key[1],
                    "horizon_step": key[2],
                    "field": reference["field"],
                    "target_column": key[3],
                    "reference_neuron": " ".join(
                        map(str, sorted(winners))
                    ),
                    "reference_segment_id": reference_segment_id,
                    "reference_applicability": reference[
                        "reference_applicability"
                    ],
                    "segment_id": segment_id,
                    "segment_neuron": row.get("target_neuron", ""),
                    "is_reference_neuron": is_reference_neuron,
                    "is_reference_segment": (
                        bool(reference_segment_id)
                        and segment_id == reference_segment_id
                    ),
                    "inspected": True,
                    "response_computed": _bool(
                        row.get("response_available")
                    ),
                    "response_positive": (
                        _number(row.get("response_peak"), -math.inf) > 0.0
                    ),
                    "crossed_threshold": _bool(
                        row.get("crossed_threshold")
                    ),
                    "valid_firing_time": (
                        row.get("predicted_time", "") != ""
                    ),
                    "became_event_winner": _bool(
                        row.get("became_event_winner")
                    ),
                    "became_column_candidate": _bool(
                        row.get("became_prediction_candidate")
                    ),
                    "appeared_in_raw_code": _bool(
                        row.get("appeared_in_raw_code")
                    ),
                    "emitted_after_competition": _bool(
                        row.get("emitted_after_competition")
                    ),
                    "response_peak": row.get("response_peak", ""),
                    "response_at_selected_time": row.get(
                        "response_at_selected_time",
                        "",
                    ),
                    "first_crossing_time": row.get(
                        "first_crossing_time",
                        "",
                    ),
                    "predicted_time": row.get("predicted_time", ""),
                    "threshold_margin": row.get("threshold_margin", ""),
                    "candidate_score": row.get(
                        "candidate_score_value",
                        "",
                    ),
                    "contributor_count": row.get("contributor_count", ""),
                    "positive_contributor_count": row.get(
                        "positive_contributor_count",
                        "",
                    ),
                    "synapse_count": row.get("synapse_count", ""),
                    "weight_sum": row.get("segment_weight_sum", ""),
                    "segment_age": row.get("segment_age", ""),
                    "current_context_jaccard": row.get(
                        "current_context_jaccard",
                        "",
                    ),
                    "creation_current_source_jaccard": row.get(
                        "creation_current_source_jaccard",
                        "",
                    ),
                    "historical_winner_match": row.get(
                        "historical_winner_match",
                        "",
                    ),
                    "predicted_supported_contribution": row.get(
                        "predicted_supported_contribution",
                        "",
                    ),
                    "burst_supported_contribution": row.get(
                        "burst_supported_contribution",
                        "",
                    ),
                    "support_class": row.get("source_support_class", ""),
                    "selection_group_id": row.get(
                        "selection_group_id",
                        "",
                    ),
                    "group_type": (
                        "EVENT_SCORE_SELECTION"
                        if row.get("selection_group_id")
                        else "COLUMN_EARLIEST_SELECTION"
                    ),
                    "actual_selector_primary_key": (
                        "candidate_score"
                        if row.get("selection_group_id")
                        else "predicted_time"
                    ),
                    "actual_selector_secondary_key": "inspection_order",
                    "actual_selector_tie_break_key": (
                        "captured_in_preselection_replacement_trace"
                    ),
                    "rank_by_existing_selector": row.get(
                        "rank_within_column_group",
                        row.get("rank_within_actual_group", ""),
                    ),
                    "winning_segment_id": winner_id,
                    "winning_neuron": (
                        winner.get("target_neuron", "")
                        if winner is not None
                        else ""
                    ),
                    "winner_is_reference_neuron": (
                        _integer(winner.get("target_neuron"))
                        in winners
                        if winner is not None
                        else False
                    ),
                    "winner_is_reference_segment": (
                        bool(reference_segment_id)
                        and winner_id == reference_segment_id
                    ),
                    "comparison_lost_at_stage": row.get(
                        "elimination_stage",
                        "",
                    ),
                    "comparison_loss_reason": row.get(
                        "elimination_reason",
                        "",
                    ),
                    "score_gap_to_winner": row.get(
                        "winner_score_gap",
                        "",
                    ),
                    "crossing_time_gap_to_winner": (
                        (
                            _number(
                                row.get("first_crossing_time"),
                                math.nan,
                            )
                            - _number(
                                winner.get("first_crossing_time"),
                                math.nan,
                            )
                        )
                        if winner is not None
                        else ""
                    ),
                    "peak_gap_to_winner": (
                        (
                            _number(row.get("response_peak"), math.nan)
                            - _number(
                                winner.get("response_peak"),
                                math.nan,
                            )
                        )
                        if winner is not None
                        else ""
                    ),
                    "contributor_gap_to_winner": (
                        (
                            _number(row.get("contributor_count"), math.nan)
                            - _number(
                                winner.get("contributor_count"),
                                math.nan,
                            )
                        )
                        if winner is not None
                        else ""
                    ),
                    "context_jaccard_gap_to_winner": (
                        (
                            _number(
                                row.get("current_context_jaccard"),
                                math.nan,
                            )
                            - _number(
                                winner.get("current_context_jaccard"),
                                math.nan,
                            )
                        )
                        if winner is not None
                        else ""
                    ),
                    "replacement_event_count": 0,
                }
            )
    return output


def classify_reference_loss(
    reference_segments: Sequence[Mapping[str, object]],
) -> str:
    """Classify only stages represented by the actual trace level."""

    if not reference_segments:
        return "REFERENCE_NEURON_PRE_CROSSING_DETAIL_UNAVAILABLE"
    if not any(row.get("predicted_time", "") != "" for row in reference_segments):
        return "REFERENCE_NEURON_CROSSED_NO_VALID_TIME"
    if not any(_bool(row.get("became_event_winner")) for row in reference_segments):
        return "REFERENCE_NEURON_VALID_EVENT_BUT_LOST_EVENT_SELECTION"
    if not any(
        _bool(row.get("became_prediction_candidate"))
        for row in reference_segments
    ):
        return "REFERENCE_NEURON_EVENT_WINNER_BUT_LOST_COLUMN_SELECTION"
    if not any(
        _bool(row.get("emitted_after_competition"))
        for row in reference_segments
    ):
        return "REFERENCE_NEURON_CANDIDATE_BUT_SUPPRESSED_INTERCOLUMN"
    return "REFERENCE_NEURON_EMITTED"


def _best_row(
    rows: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            _bool(row.get("became_prediction_candidate")),
            _bool(row.get("became_event_winner")),
            _number(row.get("candidate_score_value"), -math.inf),
            -_number(row.get("predicted_time"), math.inf),
            -int(row.get("inspection_order", 0) or 0),
        ),
    )


def build_reference_funnel(
    identity_rows: Sequence[Mapping[str, object]],
    segment_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build one real-stage funnel row and optional wrong-winner pair per key."""

    references = {
        _key(row): row
        for row in identity_rows
        if _bool(row.get("neuron_reference_valid_before_observation"))
    }
    by_key: dict[
        tuple[int, str, int, int],
        list[Mapping[str, object]],
    ] = defaultdict(list)
    for row in segment_rows:
        key = _key(row)
        if key in references:
            by_key[key].append(row)
    funnels: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    for key, reference in sorted(references.items()):
        rows = by_key.get(key, [])
        winners = _winner_set(reference)
        reference_rows = [
            row
            for row in rows
            if _integer(row.get("target_neuron")) in winners
        ]
        candidates = [
            row
            for row in rows
            if _bool(row.get("became_prediction_candidate"))
        ]
        emitted = [
            row
            for row in rows
            if _bool(row.get("emitted_after_competition"))
        ]
        selected = _best_row(candidates)
        reference_best = _best_row(reference_rows)
        selected_neuron = (
            _integer(selected.get("target_neuron"))
            if selected is not None
            else None
        )
        loss_stage = classify_reference_loss(reference_rows)
        common = {
            **DIAGNOSTIC_MARKERS,
            "input_index": key[0],
            "target_timestamp": key[1],
            "horizon_step": key[2],
            "field": reference["field"],
            "target_column": key[3],
            "observe_scenario": reference[
                "teacher_forced_observe_scenario"
            ],
            "reference_applicability": reference[
                "reference_applicability"
            ],
            "reference_neurons": " ".join(map(str, sorted(winners))),
        }
        funnels.append(
            {
                **common,
                "reference_neuron_inspected_segment_count": "",
                "reference_neuron_positive_response_segment_count": "",
                "reference_neuron_threshold_crossing_segment_count": len(
                    reference_rows
                ),
                "reference_neuron_valid_firing_time_segment_count": sum(
                    row.get("predicted_time", "") != ""
                    for row in reference_rows
                ),
                "reference_neuron_event_winner_count": sum(
                    _bool(row.get("became_event_winner"))
                    for row in reference_rows
                ),
                "reference_neuron_prediction_candidate_count": sum(
                    _bool(row.get("became_prediction_candidate"))
                    for row in reference_rows
                ),
                "reference_neuron_raw_code_presence": any(
                    _bool(row.get("appeared_in_raw_code"))
                    for row in reference_rows
                ),
                "reference_neuron_emitted_presence": any(
                    _bool(row.get("emitted_after_competition"))
                    for row in reference_rows
                ),
                "total_inspected_neurons": "",
                "total_crossing_neurons": len(
                    {
                        _integer(row.get("target_neuron"))
                        for row in rows
                    }
                ),
                "total_candidate_neurons": len(
                    {
                        _integer(row.get("target_neuron"))
                        for row in candidates
                    }
                ),
                "total_emitted_neurons": len(
                    {
                        _integer(row.get("target_neuron"))
                        for row in emitted
                    }
                ),
                "actual_selected_autonomous_neuron": (
                    selected_neuron if selected_neuron is not None else ""
                ),
                "selected_autonomous_segment": (
                    selected.get("segment_provenance_id", "")
                    if selected is not None
                    else ""
                ),
                "selected_autonomous_score": (
                    selected.get("candidate_score_value", "")
                    if selected is not None
                    else ""
                ),
                "selected_autonomous_predicted_time": (
                    selected.get("predicted_time", "")
                    if selected is not None
                    else ""
                ),
                "loss_stage": loss_stage,
                "reference_segment_inspected_count": "",
                "reference_positive_response_count": "",
                "reference_threshold_crossing_count": len(reference_rows),
                "reference_valid_time_count": sum(
                    row.get("predicted_time", "") != ""
                    for row in reference_rows
                ),
                "reference_event_winner_count": sum(
                    _bool(row.get("became_event_winner"))
                    for row in reference_rows
                ),
                "reference_column_candidate_count": sum(
                    _bool(row.get("became_prediction_candidate"))
                    for row in reference_rows
                ),
                "reference_raw_code_count": sum(
                    _bool(row.get("appeared_in_raw_code"))
                    for row in reference_rows
                ),
                "reference_emitted_count": sum(
                    _bool(row.get("emitted_after_competition"))
                    for row in reference_rows
                ),
                "same_column_nonreference_crossing_count": sum(
                    _integer(row.get("target_neuron")) not in winners
                    for row in rows
                ),
                "same_column_nonreference_candidate_count": sum(
                    _integer(row.get("target_neuron")) not in winners
                    and _bool(row.get("became_prediction_candidate"))
                    for row in rows
                ),
                "selected_autonomous_neuron": (
                    selected_neuron if selected_neuron is not None else ""
                ),
                "selected_autonomous_segment": (
                    selected.get("segment_provenance_id", "")
                    if selected is not None
                    else ""
                ),
                "first_reference_loss_stage": loss_stage,
                "first_reference_loss_reason": (
                    reference_best.get("elimination_reason", "")
                    if reference_best is not None
                    else "trace_level_does_not_include_pre_crossing_segments"
                ),
                "pre_crossing_detail_available": False,
                "pre_crossing_unavailable_reason": (
                    "formal trace level is crossing"
                ),
            }
        )
        if (
            reference_best is None
            or selected is None
            or selected_neuron in winners
        ):
            continue
        pair = {
            **common,
            "reference_neuron": reference_best["target_neuron"],
            "selected_wrong_neuron": selected["target_neuron"],
            "reference_segment_id": reference_best[
                "segment_provenance_id"
            ],
            "wrong_winner_segment_id": selected[
                "segment_provenance_id"
            ],
        }
        for label, row in (
            ("reference", reference_best),
            ("winner", selected),
        ):
            for field in (
                "response_peak",
                "response_at_selected_time",
                "first_crossing_time",
                "predicted_time",
                "threshold_margin",
                "candidate_score_value",
                "contributor_count",
                "positive_contributor_count",
                "synapse_count",
                "segment_weight_sum",
                "segment_age",
                "creation_transition_index",
                "current_context_overlap",
                "current_context_jaccard",
                "creation_current_source_jaccard",
                "historical_winner_match",
                "predicted_supported_contribution",
                "burst_supported_contribution",
                "source_support_class",
                "selection_group_id",
                "column_selection_group_id",
            ):
                pair[f"{label}_{field}"] = row.get(field, "")
        reference_time = _number(
            reference_best.get("predicted_time"),
            math.inf,
        )
        winner_time = _number(selected.get("predicted_time"), math.inf)
        reference_score = _number(
            reference_best.get("candidate_score_value"),
            -math.inf,
        )
        winner_score = _number(
            selected.get("candidate_score_value"),
            -math.inf,
        )
        pair.update(
            {
                "winner_was_earlier": winner_time < reference_time,
                "winner_had_higher_score": winner_score > reference_score,
                "predicted_time_gap_winner_minus_reference": (
                    winner_time - reference_time
                ),
                "score_gap_winner_minus_reference": (
                    winner_score - reference_score
                ),
                "replacement_observed": "",
                "comparison_field": "",
                "comparison_previous_value": "",
                "comparison_replacement_value": "",
                "reference_failure_comparison": "",
            }
        )
        pairs.append(pair)
    return funnels, pairs


def attach_replacement_evidence(
    pairs: list[dict[str, object]],
    replacement_rows: Sequence[Mapping[str, object]],
) -> None:
    """Attach only comparisons that explicitly replaced the reference segment."""

    by_previous: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in replacement_rows:
        by_previous[str(row.get("previous_winner_id", ""))].append(row)
    for pair in pairs:
        reference_id = str(pair["reference_segment_id"])
        winner_id = str(pair["wrong_winner_segment_id"])
        match = next(
            (
                row
                for row in by_previous.get(reference_id, ())
                if str(row.get("replacement_winner_id", "")) == winner_id
            ),
            None,
        )
        if match is None:
            continue
        pair["replacement_observed"] = True
        pair["comparison_field"] = match.get("comparison_field", "")
        pair["comparison_previous_value"] = match.get(
            "previous_value",
            "",
        )
        pair["comparison_replacement_value"] = match.get(
            "replacement_value",
            "",
        )
        pair["reference_failure_comparison"] = match.get(
            "replacement_stage",
            "",
        )


def build_reference_replacement_trace(
    identity_rows: Sequence[Mapping[str, object]],
    segment_rows: Sequence[Mapping[str, object]],
    replacement_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Retain explicit replacement comparisons involving a valid reference."""

    references = {
        _key(row): row
        for row in identity_rows
        if _bool(row.get("neuron_reference_valid_before_observation"))
    }
    segment_by_id = {
        str(row.get("segment_provenance_id", "")): row
        for row in segment_rows
    }
    output = []
    for replacement in replacement_rows:
        previous_id = str(replacement.get("previous_winner_id", ""))
        new_id = str(replacement.get("replacement_winner_id", ""))
        previous = segment_by_id.get(previous_id)
        new = segment_by_id.get(new_id)
        representative = previous or new
        if representative is None:
            continue
        key = _key(representative)
        reference = references.get(key)
        if reference is None:
            continue
        winners = _winner_set(reference)
        previous_reference = (
            previous is not None
            and _integer(previous.get("target_neuron")) in winners
        )
        new_reference = (
            new is not None
            and _integer(new.get("target_neuron")) in winners
        )
        if not (previous_reference or new_reference):
            continue
        output.append(
            {
                **DIAGNOSTIC_MARKERS,
                "selection_group_id": replacement.get(
                    "selection_group_id",
                    "",
                ),
                "input_index": key[0],
                "horizon_step": key[2],
                "target_column": key[3],
                "previous_winner_segment": previous_id,
                "previous_winner_neuron": (
                    previous.get("target_neuron", "")
                    if previous is not None
                    else ""
                ),
                "new_winner_segment": new_id,
                "new_winner_neuron": (
                    new.get("target_neuron", "")
                    if new is not None
                    else ""
                ),
                "previous_is_reference": previous_reference,
                "new_is_reference": new_reference,
                "comparison_key": replacement.get(
                    "comparison_field",
                    "",
                ),
                "previous_value": replacement.get("previous_value", ""),
                "new_value": replacement.get("replacement_value", ""),
                "tie_break_values": json.dumps(
                    {
                        "previous": replacement.get(
                            "previous_tie_break_values",
                            "",
                        ),
                        "new": replacement.get(
                            "replacement_tie_break_values",
                            "",
                        ),
                    },
                    separators=(",", ":"),
                ),
                "replacement_reason": replacement.get(
                    "replacement_stage",
                    "",
                ),
            }
        )
    return output


MetricKey = Callable[[Mapping[str, object]], tuple[object, ...]]


def _metric_keys() -> dict[str, MetricKey]:
    descending = lambda field: (
        lambda row: (_number(row.get(field), -math.inf),)
    )
    ascending = lambda field: (
        lambda row: (-_number(row.get(field), math.inf),)
    )
    return {
        "existing_selector": lambda row: (
            _bool(row.get("became_prediction_candidate")),
            _bool(row.get("became_event_winner")),
            -_number(row.get("predicted_time"), math.inf),
            _number(row.get("candidate_score_value"), -math.inf),
        ),
        "response_peak": descending("response_peak"),
        "candidate_score": descending("candidate_score_value"),
        "earliest_crossing_time": ascending("first_crossing_time"),
        "predicted_time": ascending("predicted_time"),
        "contributor_count": descending("contributor_count"),
        "current_context_jaccard": descending("current_context_jaccard"),
        "creation_current_source_jaccard": descending(
            "creation_current_source_jaccard"
        ),
        "historical_winner_match": lambda row: (
            _bool(row.get("historical_winner_match")),
        ),
        "segment_age": ascending("segment_age"),
        "weight_sum": descending("segment_weight_sum"),
        "context_then_score": lambda row: (
            _number(row.get("current_context_jaccard"), -math.inf),
            _number(row.get("candidate_score_value"), -math.inf),
        ),
        "score_then_context": lambda row: (
            _number(row.get("candidate_score_value"), -math.inf),
            _number(row.get("current_context_jaccard"), -math.inf),
        ),
    }


def build_ranking_observations(
    identity_rows: Sequence[Mapping[str, object]],
    segment_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    references = {
        _key(row): row
        for row in identity_rows
        if _bool(row.get("neuron_reference_valid_before_observation"))
    }
    by_key: dict[
        tuple[int, str, int, int],
        list[Mapping[str, object]],
    ] = defaultdict(list)
    for row in segment_rows:
        key = _key(row)
        if key in references and _bool(row.get("crossed_threshold")):
            by_key[key].append(row)
    observations: list[dict[str, object]] = []
    for key, rows in sorted(by_key.items()):
        reference = references[key]
        winners = _winner_set(reference)
        if not rows or not winners:
            continue
        for metric, key_function in _metric_keys().items():
            ordered = sorted(
                rows,
                key=lambda row: (
                    key_function(row),
                    -int(row["target_neuron"]),
                    str(row.get("segment_provenance_id", "")),
                ),
                reverse=True,
            )
            ranks = [
                index
                for index, row in enumerate(ordered, start=1)
                if _integer(row.get("target_neuron")) in winners
            ]
            rank = min(ranks) if ranks else len(ordered) + 1
            selected = ordered[0]
            observations.append(
                {
                    **DIAGNOSTIC_MARKERS,
                    "input_index": key[0],
                    "target_timestamp": key[1],
                    "horizon_step": key[2],
                    "field": reference["field"],
                    "target_column": key[3],
                    "observe_scenario": reference[
                        "teacher_forced_observe_scenario"
                    ],
                    "metric": metric,
                    "reference_rank": rank,
                    "candidate_count": len(ordered),
                    "hit_at_1": rank <= 1,
                    "hit_at_2": rank <= 2,
                    "hit_at_3": rank <= 3,
                    "hit_at_5": rank <= 5,
                    "reciprocal_rank": 1.0 / rank,
                    "pairwise_win_rate": (
                        (len(ordered) - rank) / (len(ordered) - 1)
                        if len(ordered) > 1
                        else 1.0
                    ),
                    "selected_neuron": selected["target_neuron"],
                    "selected_segment_id": selected[
                        "segment_provenance_id"
                    ],
                    "reference_segment_exact_match": (
                        selected.get("segment_provenance_id", "")
                        == reference.get("teacher_forced_segment_id", "")
                    ),
                }
            )
    return observations


def summarize_rankings(
    observations: Sequence[Mapping[str, object]],
    *,
    group_fields: Sequence[str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in observations:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for group, rows in sorted(grouped.items(), key=lambda item: repr(item[0])):
        ranks = [_number(row["reference_rank"]) for row in rows]
        payload = dict(zip(group_fields, group))
        payload.update(
            {
                "rows": len(rows),
                "reference_rank_mean": statistics.fmean(ranks),
                "reference_rank_median": statistics.median(ranks),
                "hit_at_1": statistics.fmean(
                    _bool(row["hit_at_1"]) for row in rows
                ),
                "hit_at_2": statistics.fmean(
                    _bool(row["hit_at_2"]) for row in rows
                ),
                "hit_at_3": statistics.fmean(
                    _bool(row["hit_at_3"]) for row in rows
                ),
                "hit_at_5": statistics.fmean(
                    _bool(row["hit_at_5"]) for row in rows
                ),
                "mrr": statistics.fmean(
                    _number(row["reciprocal_rank"]) for row in rows
                ),
                "pairwise_win_rate": statistics.fmean(
                    _number(row["pairwise_win_rate"]) for row in rows
                ),
                "reference_segment_exact_match": statistics.fmean(
                    _bool(row["reference_segment_exact_match"])
                    for row in rows
                ),
            }
        )
        output.append(payload)
    return output


def bootstrap_ranking(
    observations: Sequence[Mapping[str, object]],
    *,
    samples: int,
    seed: int,
) -> list[dict[str, object]]:
    """Bootstrap Hit@1 by rollout input, preserving correlated columns."""

    by_metric: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in observations:
        by_metric[str(row["metric"])].append(row)
    rng = random.Random(seed)
    output = []
    for metric, rows in sorted(by_metric.items()):
        by_input: dict[int, list[Mapping[str, object]]] = defaultdict(list)
        for row in rows:
            by_input[int(row["input_index"])].append(row)
        inputs = sorted(by_input)
        draws = []
        for _ in range(samples):
            sampled = [
                row
                for _input in inputs
                for row in by_input[inputs[rng.randrange(len(inputs))]]
            ]
            draws.append(
                statistics.fmean(_bool(row["hit_at_1"]) for row in sampled)
            )
        draws.sort()
        output.append(
            {
                "analysis": "reference_neuron_ranking",
                "metric": metric,
                "bootstrap_unit": "rollout_input_index",
                "samples": samples,
                "seed": seed,
                "hit_at_1_mean": statistics.fmean(draws),
                "ci_2_5": draws[int(0.025 * (len(draws) - 1))],
                "ci_97_5": draws[int(0.975 * (len(draws) - 1))],
            }
        )
    return output


def analyze(
    *,
    run_dir: Path,
    identity_dir: Path,
    output_dir: Path,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    identity_rows = _read_csv(
        identity_dir / "teacher_forced_identity_rows.csv"
    )
    segment_rows = _read_csv(
        _find_trace(run_dir, "preselection_segment_trace")
    )
    input_timestamps: dict[int, str] = {}
    try:
        branch_rows = _read_csv(_find_trace(run_dir, "branch_candidate_trace"))
        input_timestamps = {
            int(row["input_index"]): str(row.get("input_timestamp", ""))
            for row in branch_rows
        }
    except FileNotFoundError:
        pass
    selection_trace = build_reference_selection_trace(
        identity_rows,
        segment_rows,
        input_timestamps=input_timestamps,
    )
    funnels, pairs = build_reference_funnel(identity_rows, segment_rows)
    try:
        replacement_rows = _read_csv(
            _find_trace(run_dir, "preselection_replacement_trace")
        )
    except FileNotFoundError:
        replacement_rows = []
    if replacement_rows:
        attach_replacement_evidence(pairs, replacement_rows)
    reference_replacements = build_reference_replacement_trace(
        identity_rows,
        segment_rows,
        replacement_rows,
    )
    replacement_events_by_segment: Counter[str] = Counter()
    for row in reference_replacements:
        replacement_events_by_segment[
            str(row["previous_winner_segment"])
        ] += 1
        replacement_events_by_segment[
            str(row["new_winner_segment"])
        ] += 1
    for row in selection_trace:
        row["replacement_event_count"] = replacement_events_by_segment[
            str(row["segment_id"])
        ]
    replacement_counts = Counter(
        (
            _bool(row["previous_is_reference"]),
            _bool(row["new_is_reference"]),
            str(row["replacement_reason"]),
        )
        for row in reference_replacements
    )
    replacement_summary = [
        {
            "previous_is_reference": previous_reference,
            "new_is_reference": new_reference,
            "replacement_reason": reason,
            "count": count,
        }
        for (
            previous_reference,
            new_reference,
            reason,
        ), count in sorted(replacement_counts.items())
    ]
    ranking_observations = build_ranking_observations(
        identity_rows,
        segment_rows,
    )
    ranking_summary = summarize_rankings(
        ranking_observations,
        group_fields=("metric",),
    )
    counterfactual = (
        summarize_rankings(
            ranking_observations,
            group_fields=("metric", "field"),
        )
        + summarize_rankings(
            ranking_observations,
            group_fields=("metric", "horizon_step"),
        )
        + summarize_rankings(
            ranking_observations,
            group_fields=("metric", "observe_scenario"),
        )
        + summarize_rankings(
            ranking_observations,
            group_fields=(
                "metric",
                "field",
                "horizon_step",
                "observe_scenario",
            ),
        )
    )
    loss_counts = Counter(str(row["loss_stage"]) for row in funnels)
    loss_summary = [
        {
            "loss_stage": stage,
            "count": count,
            "fraction": count / len(funnels) if funnels else 0.0,
        }
        for stage, count in sorted(loss_counts.items())
    ]
    bootstrap = bootstrap_ranking(
        ranking_observations,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    _write_csv(output_dir / "reference_neuron_funnel.csv", funnels)
    _write_csv(
        output_dir / "reference_neuron_funnel_trace.csv",
        funnels,
    )
    _write_csv(
        output_dir / "reference_neuron_selection_trace.csv",
        selection_trace,
    )
    _write_csv(
        output_dir / "reference_neuron_replacement_trace.csv",
        reference_replacements,
    )
    _write_csv(
        output_dir / "reference_neuron_loss_stage_summary.csv",
        loss_summary,
    )
    _write_csv(output_dir / "reference_vs_wrong_winner_pairs.csv", pairs)
    _write_csv(
        output_dir / "reference_neuron_ranking_metrics.csv",
        ranking_summary,
    )
    _write_csv(
        output_dir / "reference_selector_rank_summary.csv",
        counterfactual,
    )
    _write_csv(
        output_dir / "reference_replacement_summary.csv",
        replacement_summary,
    )
    _write_csv(
        output_dir / "reference_metric_separability.csv",
        ranking_summary,
    )
    _write_csv(
        output_dir / "reference_neuron_counterfactual_summary.csv",
        counterfactual,
    )
    _write_csv(output_dir / "scenario_bootstrap_ci.csv", bootstrap)
    _write_csv(
        output_dir / "reference_metric_bootstrap_ci.csv",
        bootstrap,
    )
    summary = {
        **DIAGNOSTIC_MARKERS,
        "preselection_trace_level": "crossing",
        "pre_crossing_loss_detail_available": False,
        "pre_crossing_unavailable_reason": (
            "the formal 250 trace contains only threshold-crossing segments"
        ),
        "preexisting_reference_rows": len(funnels),
        "wrong_winner_pair_rows": len(pairs),
        "selection_trace_rows": len(selection_trace),
        "reference_replacement_rows": len(reference_replacements),
        "loss_stages": loss_summary,
        "ranking_metrics": ranking_summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reference_neuron_selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_lines = [
        "# Reference Neuron Selection Diagnostic",
        "",
        "This analysis is oracle-conditioned on the target column and a "
        "future-observed pre-existing reference identity. It is read-only, "
        "nonpaper, and not a deployable selector.",
        "",
        f"Pre-existing reference rows: {len(funnels)}",
        f"Participating segment rows: {len(selection_trace)}",
        f"Explicit reference replacement events: "
        f"{len(reference_replacements)}",
        "",
        "## Loss Stages",
        "",
    ]
    report_lines.extend(
        f"- `{row['loss_stage']}`: {row['count']} "
        f"({row['fraction']:.6f})"
        for row in loss_summary
    )
    report_lines.extend(
        [
            "",
            "## Offline Ranking",
            "",
            "Ranking metrics use the real same-column crossing set. They do "
            "not generate a SymbolCode or alter competition.",
            "",
        ]
    )
    report_lines.extend(
        f"- `{row['metric']}` Hit@1={row['hit_at_1']:.6f}, "
        f"MRR={row['mrr']:.6f}"
        for row in ranking_summary
    )
    (output_dir / "REFERENCE_NEURON_SELECTION_REPORT.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--identity-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze(
        run_dir=Path(args.run_dir),
        identity_dir=Path(args.identity_dir),
        output_dir=Path(args.output_dir),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()
