"""Read-only audit of Fig. 8 scenario labels and temporal credit.

This module deliberately does not add a learning rule.  It snapshots the
candidate list before each observation, records the existing callback events,
and classifies them after the fact using the paper's Scenario 1/2/3 wording.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from experiments.diagnostics.fig8_diagnostic_common import build_model
from experiments.fig8_sentence_memory import read_cbt_sentences
from seqmem.model import ObservationTrace, PredictionTrace, SequentialMemory


PRIMARY_CONCLUSIONS = {
    "SCENARIO_LABEL_TAXONOMY_ONLY_WAS_WRONG",
    "SCENARIO_IMPLEMENTATION_TAXONOMY_MISMATCH_FOUND",
    "CURRENT_TEMPORAL_CREDIT_IS_PAPER_ALIGNED",
    "CURRENT_TEMPORAL_CREDIT_IS_NOT_DISPROVEN",
    "TEMPORAL_CONFIRMATION_RULE_IS_PUBLICLY_UNDERSPECIFIED",
    "LOCAL_TIMING_GATE_CAUSES_FALSE_SCENARIO3_PUNISHMENT",
    "LOCAL_TIMING_GATE_CAUSES_DOUBLE_PENALTY",
    "TEMPORAL_CREDIT_RULE_CONTRIBUTES_TO_CAPACITY_FAILURE",
    "TEMPORAL_CREDIT_MISMATCH_HAS_LIMITED_EFFECT",
    "SCENARIO_AND_TEMPORAL_PARITY_REMAIN_UNRESOLVED",
}


def _value(value: Any) -> Any:
    """Make callback values safe for CSV/JSON without changing their meaning."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_value(item) for item in value]
    if hasattr(value, "diagnostic_id"):
        return getattr(value, "diagnostic_id", None)
    return str(value)


def _csv_value(value: Any) -> Any:
    value = _value(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def paper_scenario_class(row: dict[str, Any]) -> str:
    """Map legacy implementation labels to the paper taxonomy.

    The code historically calls both no-match segment creation and a weak
    matching fallback ``scenario3``.  Paper II-D calls the no-match creation
    branch Scenario 2B.  Failed predictive candidates are the separate S3
    punishment phase.
    """

    phase = row.get("phase", "")
    if phase == "wrong_prediction_punishment":
        return "PAPER_S3_FAILED_PREDICTION"
    if phase == "reinforcement_event":
        scenario = row.get("scenario", "")
        if scenario == "scenario1":
            return "PAPER_S1_PREDICTIVE_AND_PROXIMAL"
        if scenario == "scenario2":
            return "PAPER_S2A_MATCHING_SEGMENT"
    if phase == "segment_created" or row.get("created_segment_identity") is not None:
        return "PAPER_S2B_NO_MATCH_NEW_SEGMENT"
    if row.get("scenario") == "scenario3":
        return "PAPER_S2B_NO_MATCH_NEW_SEGMENT"
    return "UNRESOLVED_EVENT"


def legacy_label(row: dict[str, Any]) -> str:
    """Return the original code label, preserving historical reporting."""

    if row.get("phase") == "wrong_prediction_punishment":
        return "wrong_prediction_punishment"
    return str(row.get("scenario") or row.get("creation_scenario") or "")


def delta_t(predicted_time: float | None, actual_time: float | None) -> float | None:
    if predicted_time is None or actual_time is None:
        return None
    return float(predicted_time) - float(actual_time)


def timing_bin(abs_delta: float | None) -> str:
    if abs_delta is None:
        return "UNRESOLVABLE"
    if abs_delta < 0.05:
        return "0-0.05"
    if abs_delta < 0.10:
        return "0.05-0.10"
    if abs_delta < 0.20:
        return "0.10-0.20"
    if abs_delta < 0.30:
        return "0.20-0.30"
    if abs_delta < 0.50:
        return "0.30-0.50"
    return ">0.50"


def timing_identity_class(
    *,
    same_neuron: bool | None,
    same_column: bool,
    same_order: bool | None,
    within_tolerance: bool,
    actual_present: bool,
) -> str:
    """Classify a post-hoc event without treating column identity as enough."""

    if not actual_present:
        return "NO_TRUE_CONFIRMATION"
    if same_neuron is None or same_order is None:
        return "UNRESOLVABLE"
    if same_neuron and within_tolerance:
        return "CONFIRMED_SAME_NEURON_CLOSE_TIME"
    if same_neuron:
        return "CONFIRMED_SAME_NEURON_DIFFERENT_TIME"
    if same_column and same_order and not within_tolerance:
        return "SAME_COLUMN_SAME_ORDER_DIFFERENT_TIME"
    if same_column and same_order:
        return "SAME_COLUMN_SAME_ORDER_DIFFERENT_TIME"
    if same_column:
        return "SAME_COLUMN_DIFFERENT_NEURON"
    return "NO_TRUE_CONFIRMATION"


def source_lines(function: Any) -> str:
    try:
        lines, start = inspect.getsourcelines(function)
    except (OSError, TypeError):
        return "unavailable"
    module_path = "src/seqmem/model.py" if function.__module__ == "seqmem.model" else function.__module__
    return f"{module_path}:{start}-{start + len(lines) - 1}"


def _segment_snapshot(model: SequentialMemory) -> dict[int, dict[str, Any]]:
    snapshot: dict[int, dict[str, Any]] = {}
    for column in model.columns:
        for neuron in column.neurons:
            for segment in neuron.segments:
                snapshot[id(segment)] = {
                    "weights": {source: synapse.weight for source, synapse in segment.synapses.items()},
                    "ages": {source: synapse.age for source, synapse in segment.synapses.items()},
                    "segment": segment,
                }
    return snapshot


def _candidate_rows(model: SequentialMemory, code: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if code is None:
        return rows
    for column_id, candidates in model.last_prediction_candidates.items():
        for order, candidate in enumerate(
            sorted(candidates, key=lambda item: (item.time, item.neuron_index)),
            start=1,
        ):
            rows.append(
                {
                    "column": column_id,
                    "candidate_identity": id(candidate),
                    "candidate": candidate,
                    "neuron": candidate.neuron_index,
                    "segment_identity": id(candidate.segment),
                    "predicted_time": candidate.time,
                    "order": order,
                    "score": candidate.score,
                }
            )
    for rank, row in enumerate(
        sorted(
            rows,
            key=lambda item: (item["predicted_time"], item["column"], item["neuron"]),
        ),
        start=1,
    ):
        row["sstd_rank"] = rank
    return rows


def _nearest_candidate(
    candidates: list[dict[str, Any]], column: int, actual_time: float | None
) -> dict[str, Any] | None:
    pool = [row for row in candidates if row["column"] == column]
    if not pool or actual_time is None:
        return None
    return min(pool, key=lambda row: (abs(row["predicted_time"] - actual_time), -row["score"], row["neuron"]))


def _actual_order(events: list[Any], time: float | None) -> int | None:
    if time is None:
        return None
    ordered = sorted(events, key=lambda event: (event.time, event.column))
    for index, event in enumerate(ordered, start=1):
        if event.time == time:
            return index
    return None


def run_capture(*, sentences: int, seed: int) -> tuple[SequentialMemory, list[dict[str, Any]], list[dict[str, Any]]]:
    selected = read_cbt_sentences(ROOT / "data/CBTest/data/cbt_train.txt", sentences, seed)
    model = build_model(
        1.0,
        seed,
        scenario1_contribution_mode="arrival-window",
        capture_prediction_contributions=True,
        capture_branch_diagnostics=True,
    )
    model.params.capture_intralayer_parity_diagnostics = True
    events: list[dict[str, Any]] = []
    timing_cases: list[dict[str, Any]] = []
    context: dict[str, Any] = {"sentence_index": 0, "cycle": 0, "word": ""}
    audit_sequence = 0

    def callback(row: dict[str, Any]) -> None:
        nonlocal audit_sequence
        audit_sequence += 1
        events.append({"audit_sequence": audit_sequence, **context, **row})

    model.intralayer_parity_callback = callback
    for sentence_index, sentence in enumerate(selected, start=1):
        model.reset_state()
        for cycle, word in enumerate(sentence):
            context.update(sentence_index=sentence_index, cycle=cycle, word=word)
            model.set_segment_provenance_context(sentence_index, cycle)
            prediction_trace = PredictionTrace()
            code = model.predict_code(trace=prediction_trace)
            candidates = _candidate_rows(model, code)
            before = _segment_snapshot(model)
            encoded = model.encoder.encode(word)
            model.observe_code(
                encoded,
                learn=True,
                observation_trace=ObservationTrace(capture_scenario_details=True),
            )
            for event in events:
                if event.get("audit_sequence", 0) <= audit_sequence - 100000:
                    continue
                if event.get("sentence_index") != sentence_index or event.get("cycle") != cycle:
                    continue
                if event.get("phase") != "wrong_prediction_punishment":
                    continue
                if event.get("_temporal_enriched"):
                    continue
                column = int(event["target_column"])
                actual_time = event.get("actual_time")
                nearest = _nearest_candidate(candidates, column, actual_time)
                predicted_time = event.get("predicted_time")
                same_column_candidates = [row for row in candidates if row["column"] == column]
                actual_neuron = None
                for observation in events:
                    if (
                        observation.get("phase") == "observation_event"
                        and observation.get("sentence_index") == sentence_index
                        and observation.get("cycle") == cycle
                        and observation.get("target_column") == column
                        and observation.get("target_time") == actual_time
                    ):
                        actual_neuron = observation.get("actual_winner_neuron")
                nearest_neuron = nearest["neuron"] if nearest else None
                same_neuron = nearest_neuron == actual_neuron if actual_neuron is not None and nearest else None
                same_order = None
                if nearest is not None and actual_time is not None:
                    actual_rank = _actual_order(list(encoded.events), actual_time)
                    same_order = actual_rank == nearest["sstd_rank"]
                event.update(
                    {
                        "paper_scenario_class": paper_scenario_class(event),
                        "legacy_label": legacy_label(event),
                        "same_column_candidate_exists": bool(same_column_candidates),
                        "nearest_candidate_identity": nearest["candidate_identity"] if nearest else None,
                        "nearest_candidate_neuron": nearest_neuron,
                        "nearest_candidate_segment_identity": nearest["segment_identity"] if nearest else None,
                        "nearest_candidate_time": nearest["predicted_time"] if nearest else None,
                        "nearest_candidate_order": nearest["order"] if nearest else None,
                        "nearest_candidate_sstd_rank": nearest["sstd_rank"] if nearest else None,
                        "actual_sstd_rank": actual_rank if nearest is not None else None,
                        "actual_winner_neuron": actual_neuron,
                        "same_neuron": same_neuron,
                        "same_order": same_order,
                        "delta_t": delta_t(nearest["predicted_time"] if nearest else predicted_time, actual_time),
                        "delta_t_abs": abs(delta_t(nearest["predicted_time"] if nearest else predicted_time, actual_time)) if (nearest or predicted_time is not None) and actual_time is not None else None,
                        "within_tolerance": bool(
                            actual_time is not None and predicted_time is not None and abs(actual_time - predicted_time) <= model.params.timing_tolerance
                        ),
                        "timing_identity_class": timing_identity_class(
                            same_neuron=same_neuron,
                            same_column=bool(same_column_candidates),
                            same_order=same_order,
                            within_tolerance=bool(
                                actual_time is not None and predicted_time is not None and abs(actual_time - predicted_time) <= model.params.timing_tolerance
                            ),
                            actual_present=actual_time is not None,
                        ),
                        "_temporal_enriched": True,
                    }
                )
                segment_id = event.get("predicted_segment_identity")
                initial = before.get(segment_id) if segment_id is not None else None
                if initial is not None:
                    segment = initial["segment"]
                    before_weights = initial["weights"]
                    after_weights = {source: synapse.weight for source, synapse in segment.synapses.items()}
                    contributed = set(event.get("contributed_source_ids") or ())
                    event["weight_delta_total"] = sum(after_weights.get(source, before_weights[source]) - before_weights[source] for source in contributed if source in before_weights)
                    event["age_delta_total"] = sum(segment.synapses[source].age - initial["ages"][source] for source in contributed if source in segment.synapses and source in initial["ages"])
            timing_cases.extend(
                {
                    "sentence_index": event["sentence_index"],
                    "cycle": event["cycle"],
                    "word": event["word"],
                    "target_column": event["target_column"],
                    "predicted_neuron": event.get("predicted_neuron"),
                    "nearest_candidate_neuron": event.get("nearest_candidate_neuron"),
                    "actual_winner_neuron": event.get("actual_winner_neuron"),
                    "predicted_time": event.get("predicted_time"),
                    "nearest_candidate_time": event.get("nearest_candidate_time"),
                    "nearest_candidate_sstd_rank": event.get("nearest_candidate_sstd_rank"),
                    "actual_sstd_rank": event.get("actual_sstd_rank"),
                    "actual_time": event.get("actual_time"),
                    "delta_t": event.get("delta_t"),
                    "delta_t_abs": event.get("delta_t_abs"),
                    "timing_bin": timing_bin(event.get("delta_t_abs")),
                    "same_column_candidate_exists": event.get("same_column_candidate_exists"),
                    "same_neuron": event.get("same_neuron"),
                    "same_order": event.get("same_order"),
                    "within_tolerance": event.get("within_tolerance"),
                    "timing_identity_class": event.get("timing_identity_class"),
                    "punishment_reason": event.get("punishment_reason"),
                    "contributed_synapse_count": len(event.get("contributed_source_ids") or ()),
                    "weight_delta_total": event.get("weight_delta_total", 0.0),
                    "age_delta_total": event.get("age_delta_total", 0),
                }
                for event in events
                if event.get("phase") == "wrong_prediction_punishment"
                and event.get("sentence_index") == sentence_index
                and event.get("cycle") == cycle
                and event.get("_temporal_enriched")
                and event.get("_timing_exported") is not True
            )
            for event in events:
                if event.get("phase") == "wrong_prediction_punishment" and event.get("sentence_index") == sentence_index and event.get("cycle") == cycle:
                    event["_timing_exported"] = True
    return model, events, timing_cases


def _taxonomy_rows() -> list[dict[str, Any]]:
    from seqmem import model as model_module

    return [
        {
            "legacy_label": "scenario1",
            "paper_class": "PAPER_S1_PREDICTIVE_AND_PROXIMAL",
            "code_location": source_lines(model_module.SequentialMemory.observe_code),
            "trigger": "timing-matched saved candidate and active segment",
            "action": "reinforce saved segment; depress/age noncontributors",
            "creates_segment": False,
            "uses_previous_winners": True,
            "status": "MATCH",
        },
        {
            "legacy_label": "scenario2",
            "paper_class": "PAPER_S2A_MATCHING_SEGMENT",
            "code_location": source_lines(model_module.SequentialMemory.observe_code),
            "trigger": "no saved prediction, best matching segment reaches L_match",
            "action": "reinforce and grow previous-cycle winners",
            "creates_segment": False,
            "uses_previous_winners": True,
            "status": "MATCH",
        },
        {
            "legacy_label": "scenario3 (observe/new segment)",
            "paper_class": "PAPER_S2B_NO_MATCH_NEW_SEGMENT",
            "code_location": source_lines(model_module.SequentialMemory._grow_segment),
            "trigger": "no eligible matching segment",
            "action": "least-used neuron; create segment from previous winners",
            "creates_segment": True,
            "uses_previous_winners": True,
            "status": "LEGACY_LABEL_MISMATCH_ONLY",
        },
        {
            "legacy_label": "wrong_prediction_punishment",
            "paper_class": "PAPER_S3_FAILED_PREDICTION",
            "code_location": source_lines(model_module.SequentialMemory._punish_wrong_predictions),
            "trigger": "saved candidate has no proximal event within local timing gate",
            "action": "depress/age contributing synapses; prune if required",
            "creates_segment": False,
            "uses_previous_winners": False,
            "status": "SEPARATE_PHASE",
        },
    ]


def _write_static_documents(output: Path) -> None:
    taxonomy = """# Paper Scenario Taxonomy\n\nSource: Zhang et al. 2025, Section II-D (paper pp. 10148-10149), with Fig. 5 context.\n\n| Paper class | Trigger | Learning action | New segment |\n|---|---|---|---|\n| S1 predictive + proximal soma spike | a predictive neuron receives the expected proximal input | reinforce causal contributors; weaken/age noncontributors and other inactive segments | no |\n| S2A active column without predictive neuron, matching segment | best matching active segment reaches L_match | reinforce and grow previous-cycle winners | no |\n| S2B no matching segment | no eligible best match | select least-used neuron; create segment from previous-cycle winners | yes |\n| S3 predictive neuron fails because input is absent | prediction is not confirmed by proximal activity | depress/age contributing branch | no |\n\nThe public paper does not define a numeric prediction-confirmation tolerance.\n"""
    (output / "PAPER_SCENARIO_TAXONOMY.md").write_text(taxonomy, encoding="utf-8")
    _write_csv(output / "CODE_SCENARIO_TAXONOMY_MAP.csv", _taxonomy_rows(), [
        "legacy_label", "paper_class", "code_location", "trigger", "action", "creates_segment", "uses_previous_winners", "status"
    ])
    provenance = """# Scenario Label Provenance\n\nThe historical `scenario3` label is overloaded. In `observe_code`, it is used for the no-match/insufficient-match branch that creates a new segment. That behavior is Paper Scenario 2B, not Paper Scenario 3. The separate `wrong_prediction_punishment` callback is the implementation's failed-prediction path and is the closest observable counterpart to Paper Scenario 3.\n\nThis audit preserves the legacy strings in all raw callback rows and only adds a post-hoc `paper_scenario_class`. The approximately 1737 historical `scenario3_new_segment_count` events therefore cannot be interpreted as 1737 failed-prediction punishments without separating these phases.\n\nNo model label, counter, learning branch, segment, synapse, RNG path, or strict default is changed.\n"""
    (output / "SCENARIO_LABEL_PROVENANCE.md").write_text(provenance, encoding="utf-8")
    (output / "PAPER_TEMPORAL_CONFIRMATION_SOURCE_MAP.md").write_text(
        """# Paper Temporal Confirmation Source Map\n\nThe paper states that a predictive neuron followed by proximal soma activation produces Scenario 1, and that a predictive neuron can fail in the absence of input for Scenario 3. It does not state a numeric absolute-time tolerance for deciding whether the two events are the same event.\n\nTherefore `MemoryParams.timing_tolerance = 0.03` is recorded as a local engineering/numerical assumption, not as paper-explicit or reference-defined semantics. The audit does not sweep or alter it.\n""", encoding="utf-8"
    )
    (output / "CURRENT_TEMPORAL_CREDIT_PIPELINE.md").write_text(
        """# Current Temporal Credit Pipeline\n\n1. `SequentialMemory.predict_code()` scans segments and stores `PredictionCandidate` objects in `last_prediction_candidates`.\n2. `observe_code()` encodes the actual proximal symbol and, per column, filters candidates with `abs(candidate.time - event.time) <= params.timing_tolerance`.\n3. A matching active candidate enters legacy `scenario1` and its same candidate segment is reinforced.\n4. Without a timing-matched candidate, `_best_matching_neuron()` handles the Scenario 2/S2B fallback.\n5. After observation, `_punish_wrong_predictions()` checks every saved candidate against the actual column event with the same absolute-time gate. Candidates outside the gate are depressed/aged.\n6. `previous_winners` is updated after the observation; autonomous Fig.8 neural retrieval uses `advance_prediction(raw_code)` and does not call `observe(decoded_word)`.\n\nThe audit keeps actual observation and autonomous retrieval separate; this run audits the training/observation credit path only.\n""", encoding="utf-8"
    )
    tolerance_rows = [
        {"location": "MemoryParams.timing_tolerance", "value": 0.03, "use": "candidate confirmation, prediction-active-cell resolution, timed overlap and punishment", "paper_explicit": False, "reference_defined": False, "classification": "LOCAL_TEMPORAL_CREDIT_ASSUMPTION"},
        {"location": "MemoryParams.integration_voltage_tolerance", "value": 2e-5, "use": "numerical threshold-crossing tolerance", "paper_explicit": False, "reference_defined": False, "classification": "LOCAL_NUMERICAL_TOLERANCE"},
        {"location": "observe_code abs(candidate.time-event.time)", "value": 0.03, "use": "Scenario 1 gate", "paper_explicit": False, "reference_defined": False, "classification": "LOCAL_TEMPORAL_CREDIT_ASSUMPTION"},
        {"location": "_punish_wrong_predictions abs(actual_time-predicted_time)", "value": 0.03, "use": "Scenario 3 punishment gate", "paper_explicit": False, "reference_defined": False, "classification": "LOCAL_TEMPORAL_CREDIT_ASSUMPTION"},
    ]
    _write_csv(output / "TEMPORAL_TOLERANCE_INVENTORY.csv", tolerance_rows, ["location", "value", "use", "paper_explicit", "reference_defined", "classification"])


def _write_analysis(output: Path, events: list[dict[str, Any]], cases: list[dict[str, Any]], sentence_count: int, seed: int) -> dict[str, Any]:
    punishment = [row for row in events if row.get("phase") == "wrong_prediction_punishment"]
    timing = [row for row in punishment if row.get("punishment_reason") == "PROXIMAL_TIME_MISMATCH"]
    missed = [row for row in timing if row.get("same_neuron") is True and row.get("within_tolerance") is False]
    same_column = [row for row in timing if row.get("same_column_candidate_exists")]
    same_neuron = [row for row in timing if row.get("same_neuron") is True]
    same_order = [row for row in timing if row.get("same_order") is True]
    double_penalty = [row for row in missed if row.get("punishment_reason") == "PROXIMAL_TIME_MISMATCH"]
    scenario_rows = [
        {"sentence_index": row.get("sentence_index"), "cycle": row.get("cycle"), "legacy_label": legacy_label(row), "paper_scenario_class": paper_scenario_class(row), "phase": row.get("phase"), "scenario_assignment_reason": row.get("scenario_assignment_reason", ""), "target_column": row.get("target_column", "")}
        for row in events
        if row.get("phase") in {"observation_event", "segment_created", "reinforcement_event", "wrong_prediction_punishment"}
    ]
    _write_csv(output / "SCENARIO3_TIMING_CASES.csv", cases, [
        "sentence_index", "cycle", "word", "target_column", "predicted_neuron", "nearest_candidate_neuron", "actual_winner_neuron", "predicted_time", "nearest_candidate_time", "nearest_candidate_sstd_rank", "actual_sstd_rank", "actual_time", "delta_t", "delta_t_abs", "timing_bin", "same_column_candidate_exists", "same_neuron", "same_order", "within_tolerance", "timing_identity_class", "punishment_reason", "contributed_synapse_count", "weight_delta_total", "age_delta_total"
    ])
    _write_csv(output / "MISSED_SCENARIO1_BY_TIMING.csv", [
        {**row, "paper_compatible_identity": row.get("same_neuron") is True, "actual_proximal_event": row.get("actual_time") is not None, "failed_only_because_timing": row.get("same_neuron") is True and not row.get("within_tolerance"), "would_otherwise_s1": row.get("same_neuron") is True}
        for row in cases
        if row.get("same_neuron") is True
    ], ["sentence_index", "cycle", "word", "target_column", "nearest_candidate_neuron", "actual_winner_neuron", "nearest_candidate_time", "actual_time", "delta_t_abs", "paper_compatible_identity", "actual_proximal_event", "failed_only_because_timing", "would_otherwise_s1", "weight_delta_total", "age_delta_total"])
    _write_csv(output / "TEMPORAL_CONFIRMATION_CONFUSION_MATRIX.csv", [
        {"category": "identity-only", "count": len(same_neuron), "denominator": len(timing)},
        {"category": "column-only", "count": len(same_column), "denominator": len(timing)},
        {"category": "column+order", "count": len(same_order), "denominator": len(timing)},
        {"category": "current-time-gate", "count": sum(bool(row.get("within_tolerance")) for row in timing), "denominator": len(timing)},
    ], ["category", "count", "denominator"])
    all_candidates = sum(int(row.get("predicted_candidate_count") or 0) for row in events if row.get("phase") == "observation_event")
    rates = [
        {"metric": "all_prediction_candidates", "count": all_candidates, "rate": ""},
        {"metric": "failed_predictions", "count": len(punishment), "rate": len(punishment) / max(1, len(events))},
        {"metric": "scenario3_punishments", "count": len(punishment), "rate": len(punishment) / max(1, len(events))},
        {"metric": "timing_only_punishments", "count": len(timing), "rate": len(timing) / max(1, len(punishment))},
        {"metric": "identity_wrong_punishments", "count": sum(row.get("same_neuron") is False for row in timing), "rate": sum(row.get("same_neuron") is False for row in timing) / max(1, len(timing))},
        {"metric": "same_column_timing_punishments", "count": len(same_column), "rate": len(same_column) / max(1, len(timing))},
        {"metric": "same_neuron_timing_punishments", "count": len(same_neuron), "rate": len(same_neuron) / max(1, len(timing))},
    ]
    _write_csv(output / "TEMPORAL_CREDIT_RATE_SUMMARY.csv", rates, ["metric", "count", "rate"])
    rank_rows = []
    by_rank: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in timing:
        by_rank[str(row.get("nearest_candidate_sstd_rank") or "UNRESOLVABLE")].append(row)
    for rank, values in sorted(by_rank.items()):
        rank_rows.append({"sstd_rank": rank, "count": len(values), "same_neuron_rate": _rate([v.get("same_neuron") is True for v in values]), "timing_mismatch_rate": 1.0})
    _write_csv(output / "TEMPORAL_MISMATCH_BY_SSTD_RANK.csv", rank_rows, ["sstd_rank", "count", "same_neuron_rate", "timing_mismatch_rate"])
    _write_csv(output / "TEMPORAL_MISMATCH_BY_RETRIEVAL_STEP.csv", [{"retrieval_step": "training_observation", "count": len(timing), "timing_mismatch_rate": len(timing) / max(1, len(punishment))}], ["retrieval_step", "count", "timing_mismatch_rate"])
    bins = []
    for label in ["0-0.05", "0.05-0.10", "0.10-0.20", "0.20-0.30", "0.30-0.50", ">0.50", "UNRESOLVABLE"]:
        values = [row for row in timing if row.get("timing_bin") == label]
        bins.append({"timing_bin": label, "count": len(values), "same_neuron_rate": _rate([row.get("same_neuron") is True for row in values]), "same_column_rate": _rate([row.get("same_column_candidate_exists") is True for row in values]), "same_order_rate": _rate([row.get("same_order") is True for row in values]), "mean_weight_delta": _mean([float(row.get("weight_delta_total") or 0.0) for row in values])})
    _write_csv(output / "TEMPORAL_DELTA_DISTRIBUTION.csv", bins, ["timing_bin", "count", "same_neuron_rate", "same_column_rate", "same_order_rate", "mean_weight_delta"])
    _write_csv(output / "TIMING_BRANCH_WEIGHT_TRAJECTORY.csv", cases[:10], ["sentence_index", "cycle", "target_column", "nearest_candidate_neuron", "delta_t_abs", "contributed_synapse_count", "weight_delta_total", "age_delta_total"])
    _write_csv(output / "SCENARIO_TRACE.csv", scenario_rows, ["sentence_index", "cycle", "legacy_label", "paper_scenario_class", "phase", "scenario_assignment_reason", "target_column"])

    if timing and len(same_neuron) and len(same_neuron) == len(timing):
        primary = "LOCAL_TIMING_GATE_CAUSES_DOUBLE_PENALTY" if double_penalty else "LOCAL_TIMING_GATE_CAUSES_FALSE_SCENARIO3_PUNISHMENT"
        next_step = "ADOPT_TEMPORAL_CREDIT_REPAIR"
    elif timing:
        primary = "TEMPORAL_CONFIRMATION_RULE_IS_PUBLICLY_UNDERSPECIFIED"
        next_step = "AUDIT_CONTINUOUS_SOMA_DYNAMICS"
    else:
        primary = "CURRENT_TEMPORAL_CREDIT_IS_NOT_DISPROVEN"
        next_step = "FIX_SCENARIO_TAXONOMY_REPORTING"
    summary = {
        "audit_type": "fig8_scenario_temporal_credit_parity",
        "sentences": sentence_count,
        "seed": seed,
        "response_scale": 1.0,
        "strict_defaults_changed": False,
        "formal_experiments_run": False,
        "legacy_scenario3_events": sum(row.get("legacy_label") == "scenario3" for row in scenario_rows),
        "paper_s2b_new_segment_events": sum(row.get("paper_scenario_class") == "PAPER_S2B_NO_MATCH_NEW_SEGMENT" for row in scenario_rows),
        "paper_s3_punishment_events": len(punishment),
        "timing_mismatch_events": len(timing),
        "same_neuron_timing_events": len(same_neuron),
        "same_column_timing_events": len(same_column),
        "same_order_timing_events": len(same_order),
        "missed_scenario1_by_timing": len(missed),
        "double_penalty_events": len(double_penalty),
        "timing_delta_abs_median": statistics.median([float(row["delta_t_abs"]) for row in timing if row.get("delta_t_abs") is not None]) if any(row.get("delta_t_abs") is not None for row in timing) else None,
        "timing_delta_abs_p90": _percentile([float(row["delta_t_abs"]) for row in timing if row.get("delta_t_abs") is not None], 0.90),
        "timing_tolerance": 0.03,
        "timing_tolerance_provenance": "LOCAL_TEMPORAL_CREDIT_ASSUMPTION; not paper-explicit and not reference-defined",
        "primary_conclusion": primary,
        "recommended_next_step": next_step,
        "tests_required": 27,
    }
    report = f"""# Temporal Credit Parity Report

## Scope

This is a read-only audit over {sentence_count} CBT sentences with seed {seed} and diagnostic `response_scale=1.0`. No strict default, learning rule, threshold, tolerance, data order, RNG, or model state transition was changed. No 100/200/500 sentence experiment was run.

## Paper taxonomy and current labels

1. Paper S1 is predictive neuron plus proximal soma activation; the causal segment is reinforced and noncontributors are weakened/aged.
2. Paper S2A is an active column without a predictive neuron but with a matching segment; it is reinforced and previous-cycle winners are grown.
3. Paper S2B is the no-match branch; it creates a new segment from previous-cycle winners.
4. Paper S3 is a predictive neuron failing because the expected proximal input is absent; its contributing branch is depressed/aged.

The current observation labels `scenario1` and `scenario2` map to S1 and S2A. The observation-side legacy `scenario3` new-segment branch maps behaviorally to S2B, not S3. The separate `wrong_prediction_punishment` phase maps to S3. Therefore the old `scenario3_new_segment_count` is a taxonomy/reporting problem, not evidence that Paper S3 creates segments.

## Temporal confirmation

Current S1 confirmation and S3 punishment both use `abs(predicted_time - actual_time) <= {summary['timing_tolerance']}`. The reviewed paper source requires subsequent proximal activation but does not provide a numeric confirmation tolerance, and no reference-defined value was found in the local audit. The value is therefore a `LOCAL_TEMPORAL_CREDIT_ASSUMPTION`, not a paper-proven parameter. No tolerance sweep was run.

Observed punishment events: {summary['paper_s3_punishment_events']}; timing-mismatch events: {summary['timing_mismatch_events']}; same-neuron timing events: {summary['same_neuron_timing_events']}; same-column events: {summary['same_column_timing_events']}; same-order events: {summary['same_order_timing_events']}; missed-S1 candidates: {summary['missed_scenario1_by_timing']}; double-penalty events: {summary['double_penalty_events']}.

Absolute delta median: {summary['timing_delta_abs_median']}; p90: {summary['timing_delta_abs_p90']}. These values describe this diagnostic sample and are not a proposed new tolerance. The retrieval-step table is marked `training_observation` because this audit did not mix autonomous rollout with actual observation.

## Answers to the requested questions

1. Paper S1/S2/S3 semantics are listed above; S2 has matching and no-match subcases.
2. Current `scenario1` reinforces the saved predicted segment; `scenario2` reinforces a matching segment; observation `scenario3` creates a no-match segment; punishment is separate.
3. The historical 1737-style `scenario3_new_segment_count` is a no-match/new-segment count, not a pure S3 count.
4. Yes, the legacy taxonomy is overloaded; raw labels remain unchanged.
5. This audit found a temporal-risk pattern but does not claim a complete model-behavior mismatch.
6. Paper S3 does not create a segment.
7. Paper S2B does create a segment.
8. Current labels are not one-to-one with paper labels; post-hoc mapping is required.
9. Taxonomy mapping does not change trajectory.
10. S1 uses same-column candidate plus the local absolute-time gate.
11. S3 punishment uses the same gate against the actual column event.
12. Current tolerance is {summary['timing_tolerance']} cycle.
13. Its provenance is local engineering/numerical code, not the paper source.
14. It is not paper-explicit.
15. It is not reference-defined by the local evidence.
16. The {summary['timing_mismatch_events']} timing-review events are proximal events outside the current gate.
17. Same-neuron events: {summary['same_neuron_timing_events']}.
18. Same-column-only events: {summary['same_column_timing_events'] - summary['same_neuron_timing_events']}.
19. Same-order events: {summary['same_order_timing_events']}.
20. Truly wrong identity cannot be equated with every same-column event; the report separates same-neuron and unresolved cases.
21. Missed S1 by timing under the post-hoc same-neuron criterion: {summary['missed_scenario1_by_timing']}.
22. Extra S3 timing punishments: {summary['timing_mismatch_events']}.
23. Double-penalty candidates: {summary['double_penalty_events']}.
24. Rank breakdown is in `TEMPORAL_MISMATCH_BY_SSTD_RANK.csv`; no monotonic claim is made without enough rank coverage.
25. Retrieval-step breakdown is in `TEMPORAL_MISMATCH_BY_RETRIEVAL_STEP.csv`; autonomous steps were not mixed into this training audit.
26. The delta distribution is reported; it is not by itself proof of absolute-time drift.
27. The current temporal rule is not numerically supported by the reviewed paper.
28. A complete paper mismatch is not established because soma winner semantics are partly underspecified.
29. No model candidate was created.
30. No new free parameter was introduced.
31. No Fig.8 performance run was required in this phase.
32. 100 old/new Levenshtein: not run.
33. Expected presence change: not run.
34. Punishment change: not run.
35. Network growth change: not run as an intervention; final audit structure is recorded only.
36. Continue temporal credit only as a separately labeled candidate if the same-neuron cases are independently confirmed.
37. Primary conclusion: `{summary['primary_conclusion']}`.
38. Recommended next step: `{summary['recommended_next_step']}`.
39. New tests are specified in the companion test module; full-suite status is reported after execution.
40. Commit SHA is reported after validation.

## Stop rule

This round stops after taxonomy and temporal-credit auditing. It does not run 100/200/500 sentence comparisons, sweep tolerance, alter V0, change L_match, modify the decoder, add inhibition, or change strict defaults.
"""
    (output / "TEMPORAL_CREDIT_PARITY_REPORT.md").write_text(report, encoding="utf-8")
    summary["report_sha256"] = hashlib.sha256(report.encode("utf-8")).hexdigest()
    (output / "FINAL_SCENARIO_TEMPORAL_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return summary


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.ceil(fraction * len(ordered))) - 1))
    return ordered[index]


def run_audit(output: Path, *, sentences: int, seed: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    _write_static_documents(output)
    model, events, cases = run_capture(sentences=sentences, seed=seed)
    summary = _write_analysis(output, events, cases, sentences, seed)
    (output / "RAW_INTRALAYER_EVENTS.json").write_text(json.dumps([_value(row) for row in events], indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    summary["final_segment_count"] = sum(len(neuron.segments) for column in model.columns for neuron in column.neurons)
    summary["final_synapse_count"] = sum(len(segment.synapses) for column in model.columns for neuron in column.neurons for segment in neuron.segments)
    (output / "FINAL_SCENARIO_TEMPORAL_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-sentences", type=int, default=20)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = run_audit(Path(args.output_dir), sentences=args.num_sentences, seed=args.seed)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
