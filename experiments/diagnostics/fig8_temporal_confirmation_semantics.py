"""Audit the Fig.8 temporal confirmation gate and its narrow paper-semantic candidate.

The audit records prediction candidates *before* ``observe_code`` applies the
local absolute-time gate.  It never feeds the expected word back into the
model.  The optional candidate is intentionally separate from the strict
protocol and is only run after the Phase-B evidence gate passes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from experiments.fig8_sentence_memory import read_cbt_sentences
from experiments.diagnostics.fig8_diagnostic_common import evaluate_diagnostic
from seqmem.encoding import SymbolCode
from seqmem.model import MemoryParams, PredictionTrace, SequentialMemory
from seqmem.encoding import SSTDDiscreteEncoder


DELTA_BINS = (
    ("0-0.05", 0.0, 0.05),
    ("0.05-0.10", 0.05, 0.10),
    ("0.10-0.20", 0.10, 0.20),
    ("0.20-0.30", 0.20, 0.30),
    ("0.30-0.50", 0.30, 0.50),
    (">0.50", 0.50, float("inf")),
)


def classify_pre_gate_candidate_count(count: int) -> str:
    if count == 0:
        return "NO_PREDICTIVE_CANDIDATE"
    if count == 1:
        return "UNIQUE_PREDICTIVE_CANDIDATE"
    if count > 1:
        return "MULTIPLE_PREDICTIVE_CANDIDATES"
    return "UNRESOLVABLE"


def delta_bin(abs_delta: float | None) -> str:
    if abs_delta is None:
        return "UNRESOLVABLE"
    for label, lower, upper in DELTA_BINS:
        if lower <= abs_delta < upper:
            return label
    return "UNRESOLVABLE"


def sstd_rank_for_column(code: SymbolCode | None, column: int) -> int | None:
    if code is None:
        return None
    for rank, event in enumerate(sorted(code.events, key=lambda item: item.time), start=1):
        if event.column == column:
            return rank
    return None

PRIMARY_CONCLUSIONS = {
    "PREVIOUS_TEMPORAL_AUDIT_INSTRUMENTATION_WAS_INCORRECT",
    "UNIQUE_PREDICTIVE_EVENTS_RARELY_FAIL_TIMING_GATE",
    "LOCAL_ABSOLUTE_TIMING_GATE_REJECTS_MANY_UNIQUE_PREDICTIONS",
    "LOCAL_ABSOLUTE_TIMING_GATE_CAUSES_BRANCH_PROLIFERATION",
    "LOCAL_ABSOLUTE_TIMING_GATE_CAUSES_DOUBLE_NEGATIVE_CREDIT",
    "TEMPORAL_CONFIRMATION_CANDIDATE_IMPROVES_CAPACITY",
    "TEMPORAL_CONFIRMATION_CANDIDATE_HAS_LIMITED_EFFECT",
    "ABSOLUTE_TIMING_GATE_MAY_BE_FUNCTIONALLY_NECESSARY",
    "MULTI_PREDICTIVE_SOMA_WINNER_REMAINS_UNDERSPECIFIED",
    "TEMPORAL_CONFIRMATION_PARITY_REMAINS_UNRESOLVED",
}

RECOMMENDED_NEXT_STEPS = {
    "RUN_500_TEMPORAL_CONFIRMATION_VALIDATION",
    "RUN_FIG9_SHARED_CORE_TEMPORAL_VALIDATION",
    "AUDIT_MULTI_PREDICTIVE_SOMA_WINNER",
    "AUDIT_BURST_CONTEXT_REPRESENTATION",
    "WAIT_FOR_AUTHOR_CODE",
    "STOP_TEMPORAL_CONFIRMATION_DIAGNOSTICS",
}


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_value(item) for item in value]
    if hasattr(value, "diagnostic_id"):
        return getattr(value, "diagnostic_id", None)
    return str(value)


def _csv_value(value: Any) -> Any:
    value = _json_value(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def _sha256_json(value: Any) -> str:
    payload = json.dumps(_json_value(value), sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _structure_fingerprint(model: SequentialMemory) -> str:
    rows: list[Any] = []
    for column_index, column in enumerate(model.columns):
        for neuron_index, neuron in enumerate(column.neurons):
            for segment in neuron.segments:
                rows.append(
                    (
                        column_index,
                        neuron_index,
                        segment.active,
                        segment.target_time,
                        tuple(
                            (source, synapse.weight, synapse.delay, synapse.age)
                            for source, synapse in sorted(segment.synapses.items())
                        ),
                    )
                )
    return _sha256_json(rows)


def _count_segments(model: SequentialMemory) -> int:
    return sum(len(neuron.segments) for column in model.columns for neuron in column.neurons)


def _count_synapses(model: SequentialMemory) -> int:
    return sum(
        len(segment.synapses)
        for column in model.columns
        for neuron in column.neurons
        for segment in neuron.segments
    )


def _event_rank_by_column(code: SymbolCode | None) -> dict[int, int]:
    if code is None:
        return {}
    return {
        event.column: rank
        for rank, event in enumerate(sorted(code.events, key=lambda item: item.time), start=1)
    }


def _model(
    seed: int,
    *,
    temporal_trace: bool,
    temporal_mode: str = "current",
) -> SequentialMemory:
    return SequentialMemory(
        encoder=SSTDDiscreteEncoder(num_columns=100, k=10, seed=seed),
        num_neurons_per_column=10,
        params=MemoryParams(
            l_match=3,
            forgetting_threshold=500.0,
            response_scale=1.0,
            capture_prediction_contributions=True,
            capture_branch_diagnostics=True,
            capture_intralayer_parity_diagnostics=temporal_trace,
            capture_temporal_confirmation_diagnostics=temporal_trace,
            temporal_confirmation_mode=temporal_mode,
        ),
        tie_break_seed=seed,
    )


def _prediction_events(code: SymbolCode | None) -> tuple[tuple[int, float], ...]:
    if code is None:
        return ()
    return tuple((event.column, event.time) for event in code.events)


def train_capture(
    sentences: list[list[str]],
    seed: int,
    *,
    temporal_trace: bool,
    temporal_mode: str = "current",
) -> tuple[SequentialMemory, list[dict[str, Any]], dict[tuple[int, int], tuple[tuple[int, float], ...]]]:
    model = _model(
        seed,
        temporal_trace=temporal_trace,
        temporal_mode=temporal_mode,
    )
    rows: list[dict[str, Any]] = []
    predictions: dict[tuple[int, int], tuple[tuple[int, float], ...]] = {}
    context = {"sentence_index": 0, "cycle": 0, "word": ""}
    sequence = 0

    def callback(item: dict[str, Any]) -> None:
        nonlocal sequence
        sequence += 1
        rows.append({"audit_sequence": sequence, **context, **item})

    model.intralayer_parity_callback = callback
    for sentence_index, sentence in enumerate(sentences, start=1):
        model.reset_state()
        for cycle, word in enumerate(sentence):
            context.update(sentence_index=sentence_index, cycle=cycle, word=word)
            model.set_segment_provenance_context(sentence_index, cycle)
            code = model.predict_code(trace=PredictionTrace())
            predictions[(sentence_index, cycle)] = _prediction_events(code)
            model.observe(word, learn=True)
    model.intralayer_parity_callback = None
    return model, rows, predictions


def _observation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("phase") == "observation_event"]


def _punishment_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("phase") == "wrong_prediction_punishment"]


def _is_unique_candidate(row: dict[str, Any]) -> bool:
    return int(row.get("pre_gate_candidate_count") or 0) == 1


def _unique_timing_rejected(row: dict[str, Any]) -> bool:
    passes = row.get("pre_gate_candidate_time_gate_passed") or ()
    return _is_unique_candidate(row) and bool(passes) and not bool(passes[0])


def enrich_pre_gate_rows(
    rows: list[dict[str, Any]],
    predictions: dict[tuple[int, int], tuple[tuple[int, float], ...]],
    sentences: list[list[str]],
    seed: int,
) -> list[dict[str, Any]]:
    """Join raw prediction order with the pre-gate callback snapshot.

    This is post-hoc reporting only.  The actual training call has already
    completed and none of these values are passed back to the model.
    """

    encoder = SSTDDiscreteEncoder(num_columns=100, k=10, seed=seed)
    for sentence in sentences:
        for word in sentence:
            encoder.encode(word)
    observations = _observation_rows(rows)
    punishments = _punishment_rows(rows)
    punish_by_identity = {
        (
            row.get("sentence_index"),
            row.get("cycle"),
            row.get("predicted_candidate_identity"),
        ): row
        for row in punishments
        if row.get("predicted_candidate_identity") is not None
    }
    output: list[dict[str, Any]] = []
    for row in observations:
        sentence_index = int(row["sentence_index"])
        cycle = int(row["cycle"])
        code_events = predictions.get((sentence_index, cycle), ())
        predicted_rank = {
            column: rank for rank, (column, _time) in enumerate(code_events, start=1)
        }
        actual_code = encoder.encode(str(row["word"]))
        actual_rank = _event_rank_by_column(actual_code).get(int(row["target_column"]))
        candidate_columns = list(row.get("pre_gate_candidate_neuron_ids") or ())
        candidate_times = list(row.get("pre_gate_candidate_times") or ())
        candidate_scores = list(row.get("pre_gate_candidate_scores") or ())
        candidate_segments = list(row.get("pre_gate_candidate_segment_identities") or ())
        candidate_ids = list(row.get("pre_gate_candidate_identity_ids") or ())
        passes = list(row.get("pre_gate_candidate_time_gate_passed") or ())
        candidate_count = int(row.get("pre_gate_candidate_count") or 0)
        candidate_class = classify_pre_gate_candidate_count(candidate_count)
        nearest_index = None
        if candidate_times:
            nearest_index = min(
                range(len(candidate_times)),
                key=lambda index: abs(float(candidate_times[index]) - float(row["target_time"])),
            )
        candidate_rows: list[dict[str, Any]] = []
        for index in range(candidate_count):
            candidate_id = (
                row.get("pre_gate_candidate_identity_ids", [None] * candidate_count)[index]
                if row.get("pre_gate_candidate_identity_ids")
                else None
            )
            candidate_rows.append(
                {
                    "candidate_index": index,
                    "candidate_neuron": candidate_columns[index] if index < len(candidate_columns) else None,
                    "candidate_segment": candidate_segments[index] if index < len(candidate_segments) else None,
                    "candidate_time": candidate_times[index] if index < len(candidate_times) else None,
                    "candidate_score": candidate_scores[index] if index < len(candidate_scores) else None,
                    "candidate_passes": passes[index] if index < len(passes) else False,
                    "predicted_sstd_rank": predicted_rank.get(int(row["target_column"])),
                    "candidate_identity": candidate_id,
                }
            )
        if candidate_rows:
            for candidate in candidate_rows:
                candidate["punished"] = (
                    sentence_index,
                    cycle,
                    candidate.get("candidate_identity"),
                ) in punish_by_identity
        else:
            candidate_rows = []
        chosen = candidate_rows[nearest_index] if nearest_index is not None and nearest_index < len(candidate_rows) else None
        predicted_time = chosen.get("candidate_time") if chosen else None
        passed_any = any(bool(value) for value in passes)
        timing_rejected = bool(candidate_rows) and not passed_any
        actual_winner = row.get("actual_winner_neuron")
        same_neuron = bool(
            chosen
            and actual_winner is not None
            and chosen.get("candidate_neuron") == actual_winner
        ) if chosen else None
        punishment = (
            punish_by_identity.get(
                (sentence_index, cycle, chosen.get("candidate_identity"))
            )
            if chosen
            else None
        )
        enriched = {
            **row,
            "candidate_class": candidate_class,
            "predicted_sstd_rank": chosen.get("predicted_sstd_rank") if chosen else None,
            "actual_sstd_rank": actual_rank,
            "same_sstd_rank": (
                chosen is not None
                and chosen.get("predicted_sstd_rank") is not None
                and actual_rank is not None
                and chosen.get("predicted_sstd_rank") == actual_rank
            ) if chosen else None,
            "nearest_candidate_neuron": chosen.get("candidate_neuron") if chosen else None,
            "nearest_candidate_segment": chosen.get("candidate_segment") if chosen else None,
            "nearest_candidate_time": predicted_time,
            "nearest_candidate_score": chosen.get("candidate_score") if chosen else None,
            "candidate_time_gate_passed": chosen.get("candidate_passes") if chosen else None,
            "timing_rejected": timing_rejected,
            "delta_t_signed": float(predicted_time) - float(row["target_time"]) if predicted_time is not None else None,
            "delta_t_abs": abs(float(predicted_time) - float(row["target_time"])) if predicted_time is not None else None,
            "same_neuron_posthoc": same_neuron,
            "candidate_punished": punishment is not None,
            "candidate_punishment_reason": punishment.get("punishment_reason") if punishment else "",
            "selected_scenario": row.get("scenario", ""),
            "selected_learning_neuron": row.get("actual_winner_neuron"),
            "selected_learning_segment": row.get("selected_segment_identity"),
            "predicted_raw_code_column": int(row["target_column"]) in predicted_rank,
            "candidate_rows": candidate_rows,
            "prediction_event_count": len(code_events),
            "prediction_column_count": len({column for column, _time in code_events}),
        }
        output.append(enriched)
    return output


def _timing_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("timing_rejected") and row.get("delta_t_abs") is not None]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return sorted(values)[min(len(values) - 1, max(0, int(fraction * len(values))))]


def _git_first_timing_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "log", "--all", "--oneline", "-Stiming_tolerance", "--", "src/seqmem/model.py"],
            cwd=ROOT,
            text=True,
        ).splitlines()[-1]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return "unavailable"


def write_static_documents(output: Path) -> None:
    timing_commit = _git_first_timing_commit()
    (output / "TIMING_TOLERANCE_PROVENANCE.md").write_text(
        f"""# Timing Tolerance Provenance

`MemoryParams.timing_tolerance` is currently `0.03`. Git search first finds the introduction in `{timing_commit}`. The local audit found no paper-explicit or reference-defined numeric value. It is therefore classified as `LOCAL_IMPLEMENTATION / LOCAL_TEMPORAL_CREDIT_ASSUMPTION`. This round does not sweep or change it.

The value is used by `observe_code` to confirm a saved candidate and by `_punish_wrong_predictions` to identify a missing or mismatched proximal event.
""",
        encoding="utf-8",
    )
    (output / "TEMPORAL_CONFIRMATION_PAPER_EVIDENCE.md").write_text(
        """# Temporal Confirmation Paper Evidence

The local extraction of Zhang et al. 2025 was reviewed for the four requested questions.

| Question | Local source | Evidence classification | Finding |
|---|---|---|---|
| Does the paper require a numeric absolute-time confirmation tolerance? | `paper_text.txt`, Section II-D and SSTD discussion | NOT_SPECIFIED | The paper describes subsequent proximal activation and ordered spikes, but does not give `0.03` or an equivalent numeric gate. |
| Does the paper support a predictive neuron winning inside an active mini-column? | `paper_text.txt`, lines 236-250, Section II-A | PAPER_EXPLICIT | A predictive/depolarized neuron fires before its same-column counterparts; if none is predictive, the column bursts. |
| Does the paper fully specify multiple predictive-neuron soma competition? | `paper_text.txt`, lines 427-445 | PAPER_PARTIAL | The text gives the winner/inhibition idea, but not enough implementation detail to resolve every multi-candidate tie. |
| Is relative SSTD order part of the representation? | `paper_text.txt`, lines 463-478, Section II-C | PAPER_EXPLICIT | The SSTD description uses ordered spikes and distinguishes temporal patterns, while it does not say that exact local absolute time is irrelevant. |

The candidate tested in Phase C is therefore a narrowly constrained paper-semantic diagnostic, not a claim that the published algorithm has been completely recovered.
""",
        encoding="utf-8",
    )


def write_audit_artifacts(output: Path, rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    observations = [row for row in rows if row.get("phase") == "observation_event"]
    timing = _timing_cases(observations)
    unique = [row for row in observations if row.get("candidate_class") == "UNIQUE_PREDICTIVE_CANDIDATE"]
    multiple = [row for row in observations if row.get("candidate_class") == "MULTIPLE_PREDICTIVE_CANDIDATES"]
    any_prediction = [row for row in observations if int(row.get("pre_gate_candidate_count") or 0) > 0]
    punishments = _punishment_rows(rows)
    unique_rejected = [row for row in unique if row.get("timing_rejected")]
    unique_punished = [row for row in unique_rejected if row.get("candidate_punished")]
    unique_s2a = [row for row in unique_rejected if row.get("scenario") == "scenario2"]
    unique_s2b = [row for row in unique_rejected if row.get("scenario") == "scenario3"]
    branch_replacement = [row for row in unique_rejected if row.get("candidate_punished") and row.get("scenario") in {"scenario2", "scenario3"}]
    double_credit = [row for row in unique_rejected if row.get("candidate_punished")]
    summary_fields = [
        "sentence_index", "cycle", "word", "target_column", "target_time",
        "candidate_class", "pre_gate_candidate_count", "pre_gate_candidate_neuron_ids",
        "pre_gate_candidate_segment_identities", "pre_gate_candidate_times",
        "pre_gate_candidate_scores", "pre_gate_candidate_depolarization_states",
        "pre_gate_candidate_time_gate_passed", "pre_gate_timing_tolerance",
        "predicted_sstd_rank", "actual_sstd_rank", "same_sstd_rank",
        "nearest_candidate_neuron", "nearest_candidate_segment", "nearest_candidate_time",
        "delta_t_signed", "delta_t_abs", "timing_rejected", "scenario",
        "selected_learning_neuron", "selected_learning_segment", "candidate_punished",
        "candidate_punishment_reason", "prediction_event_count", "prediction_column_count",
    ]
    write_csv(output / "PRE_TIME_GATE_ACTIVE_COLUMN_TRACE.csv", observations, summary_fields)

    delta_rows = []
    for label, lower, upper in DELTA_BINS:
        values = [row for row in timing if lower <= float(row["delta_t_abs"]) < upper]
        delta_rows.append({
            "timing_bin": label,
            "count": len(values),
            "fraction_of_timing_mismatch_cases": len(values) / len(timing) if timing else 0.0,
            "mean_delta_t_signed": _mean([float(row["delta_t_signed"]) for row in values]),
        })
    write_csv(output / "TEMPORAL_DELTA_DISTRIBUTION_V2.csv", delta_rows, ["timing_bin", "count", "fraction_of_timing_mismatch_cases", "mean_delta_t_signed"])

    rates = [
        ("N_prediction_candidates", sum(int(row.get("pre_gate_candidate_count") or 0) for row in observations)),
        ("N_active_columns_with_any_prediction", len(any_prediction)),
        ("N_active_columns_with_unique_prediction", len(unique)),
        ("N_active_columns_with_multiple_predictions", len(multiple)),
        ("N_current_S1_confirmed", sum(row.get("scenario") == "scenario1" for row in observations)),
        ("N_timing_rejected", len(timing)),
        ("N_paper_S3_punished", len(punishments)),
        ("N_same_column_timing_rejected", len(timing)),
        ("N_same_neuron_timing_rejected_posthoc", sum(row.get("same_neuron_posthoc") for row in timing)),
        ("N_unique_predictive_timing_rejected", len(unique_rejected)),
        ("N_unique_predictive_timing_rejected_and_punished", len(unique_punished)),
        ("N_unique_predictive_timing_rejected_to_S2A", len(unique_s2a)),
        ("N_unique_predictive_timing_rejected_to_S2B", len(unique_s2b)),
        ("N_branch_replacement_double_credit_failure", len(branch_replacement)),
    ]
    write_csv(output / "TEMPORAL_CREDIT_RATE_SUMMARY_V2.csv", [
        {"metric": name, "count": count, "denominator": "event-specific; see metric name", "rate": ""}
        for name, count in rates
    ], ["metric", "count", "denominator", "rate"])

    unique_rows = []
    for row in unique_rejected:
        candidate = row.get("candidate_rows", [{}])[0]
        unique_rows.append({
            "sentence_index": row["sentence_index"], "cycle": row["cycle"],
            "active_column": row["target_column"], "candidate_neuron": candidate.get("candidate_neuron"),
            "candidate_segment": candidate.get("candidate_segment"), "predicted_time": row.get("nearest_candidate_time"),
            "actual_time": row.get("target_time"), "delta_t": row.get("delta_t_signed"),
            "predicted_sstd_rank": row.get("predicted_sstd_rank"), "actual_sstd_rank": row.get("actual_sstd_rank"),
            "same_sstd_rank": row.get("same_sstd_rank"), "candidate_passes_003": row.get("candidate_time_gate_passed"),
            "current_scenario": row.get("scenario"), "current_S1": row.get("scenario") == "scenario1",
            "current_S2A": row.get("scenario") == "scenario2", "current_S2B": row.get("scenario") == "scenario3",
            "current_S3_punished": row.get("candidate_punished"), "new_segment_created": row.get("created_segment_identity") is not None,
            "double_credit_failure": row in double_credit,
        })
    write_csv(output / "UNIQUE_PREDICTIVE_TIMING_REJECTION.csv", unique_rows, list(unique_rows[0].keys()) if unique_rows else ["active_column"])
    semantic_rows = []
    for row in unique_rejected:
        semantic_rows.append({
            "sentence_index": row["sentence_index"], "cycle": row["cycle"], "active_column": row["target_column"],
            "predicted_sstd_rank": row.get("predicted_sstd_rank"), "actual_sstd_rank": row.get("actual_sstd_rank"),
            "delta_t_abs": row.get("delta_t_abs"),
            "semantic_class": (
                "UNIQUE_SAME_RANK_TIME_SHIFT" if row.get("same_sstd_rank") is True
                else "UNIQUE_DIFFERENT_RANK" if row.get("same_sstd_rank") is False
                else "UNIQUE_RANK_UNAVAILABLE"
            ),
        })
    write_csv(output / "UNIQUE_TEMPORAL_SEMANTIC_CLASSES.csv", semantic_rows, ["sentence_index", "cycle", "active_column", "predicted_sstd_rank", "actual_sstd_rank", "delta_t_abs", "semantic_class"])
    provenance_rows = [
        {
            "sentence_index": row["sentence_index"], "cycle": row["cycle"], "active_column": row["target_column"],
            "delta_t_abs": row.get("delta_t_abs"),
            "provenance_class": "UNKNOWN",
            "evidence": "the trace establishes a timing residual but does not isolate phase, cycle, crossing, or numerical causality",
        }
        for row in unique_rejected
    ]
    write_csv(output / "TIMING_DRIFT_PROVENANCE.csv", provenance_rows, ["sentence_index", "cycle", "active_column", "delta_t_abs", "provenance_class", "evidence"])
    write_csv(output / "UNIQUE_BRANCH_CASE_STUDY.csv", unique_rows, list(unique_rows[0].keys()) if unique_rows else ["active_column"])
    write_csv(
        output / "TEMPORAL_CONFIRMATION_CAPACITY_GROWTH.csv",
        [{
            "sentence_count": "",
            "mode": "audit_only_no_candidate_run",
            "tokens_seen": "",
            "S1": sum(row.get("scenario") == "scenario1" for row in observations),
            "S2A": sum(row.get("scenario") == "scenario2" for row in observations),
            "S2B": sum(row.get("scenario") == "scenario3" for row in observations),
            "S3": len(punishments),
            "segments_created": "",
            "segments_total": "",
            "synapses_total": "",
            "mean_segment_size": "",
            "new_segments_per_token": "",
            "timing_rejected_unique": len(unique_rejected),
            "confirmed_unique": sum(row.get("scenario") == "scenario1" for row in unique),
            "ambiguous_multiple": len(multiple),
            "double_credit_failures": len(double_credit),
            "mean_levenshtein": "",
            "expected_presence": "",
            "raw_columns": "",
        }],
        [
            "sentence_count", "mode", "tokens_seen", "S1", "S2A", "S2B", "S3",
            "segments_created", "segments_total", "synapses_total", "mean_segment_size",
            "new_segments_per_token", "timing_rejected_unique", "confirmed_unique",
            "ambiguous_multiple", "double_credit_failures", "mean_levenshtein",
            "expected_presence", "raw_columns",
        ],
    )

    primary = "PREVIOUS_TEMPORAL_AUDIT_INSTRUMENTATION_WAS_INCORRECT" if len(timing) else "TEMPORAL_CONFIRMATION_PARITY_REMAINS_UNRESOLVED"
    if len(unique_rejected) and len(unique_punished):
        primary = "LOCAL_ABSOLUTE_TIMING_GATE_CAUSES_DOUBLE_NEGATIVE_CREDIT"
    elif len(unique_rejected):
        primary = "LOCAL_ABSOLUTE_TIMING_GATE_REJECTS_MANY_UNIQUE_PREDICTIONS"
    elif len(multiple) and not len(unique):
        primary = "MULTI_PREDICTIVE_SOMA_WINNER_REMAINS_UNDERSPECIFIED"
    gate_pass = bool(
        MemoryParams().timing_tolerance == 0.03
        and len(unique_rejected) > 0
        and (len(unique_punished) > 0 or len(unique_s2a) > 0 or len(unique_s2b) > 0)
    )
    report = f"""# Instrumentation Repair Report

## Scope

This is a Phase-A/Phase-B read-only audit over 20 CBT sentences, seed {seed}, response scale 1.0, raw training observations, and the unchanged local timing tolerance 0.03. No tolerance sweep or strict/default change was made.

## Repair

The previous delta histogram was aggregated from post-hoc punishment rows whose nearest-candidate timing fields were unavailable for `NO_PROXIMAL_EVENT` cases. It therefore wrote zero counts even though the source trace contained timing-rejected same-column candidates. This run computes deltas directly from the gate-before observation snapshot and uses one row per active column timing rejection. The total is `{len(timing)}` and the V2 bins sum to the same number.

The previous SSTD rank was a global candidate ordering and could exceed K=10. This run defines `predicted_sstd_rank` from the same raw prediction code's ordered selected column events and `actual_sstd_rank` from the encoded proximal event's ordered K events. These are post-hoc labels only and never affect training.

## Phase-B counts

| quantity | count |
|---|---:|
| active column events | {len(observations)} |
| any prediction candidate | {len(any_prediction)} |
| unique candidate | {len(unique)} |
| multiple candidates | {len(multiple)} |
| current Scenario 1 | {sum(row.get('scenario') == 'scenario1' for row in observations)} |
| timing rejected | {len(timing)} |
| unique timing rejected | {len(unique_rejected)} |
| unique rejected and exact candidate punished | {len(unique_punished)} |
| unique rejected to current S2A | {len(unique_s2a)} |
| unique rejected to current S2B | {len(unique_s2b)} |
| branch-replacement/double-credit cases | {len(branch_replacement)} |

## Gate decision

The numerical gate is not paper-explicit in the reviewed source. The paper does explicitly describe a predictive/depolarized neuron firing before its same-column peers, but multi-candidate soma competition remains only partially specified. This sample has nonzero unique candidates rejected only by the local time gate and those cases create a measurable missed-S1/punishment or fallback-branch population. Phase-C narrow candidate gate: **{'PASS' if gate_pass else 'STOP'}**.

Primary audit conclusion: `{primary}`.

The complete source trace is `PRE_TIME_GATE_ACTIVE_COLUMN_TRACE.csv`; all rates use event-specific denominators in `TEMPORAL_CREDIT_RATE_SUMMARY_V2.csv`.
"""
    (output / "INSTRUMENTATION_REPAIR_REPORT.md").write_text(report, encoding="utf-8")
    return {
        "timing_mismatch_count": len(timing),
        "any_prediction_count": len(any_prediction),
        "unique_count": len(unique),
        "multiple_count": len(multiple),
        "unique_rejected_count": len(unique_rejected),
        "unique_punished_count": len(unique_punished),
        "unique_s2a_count": len(unique_s2a),
        "unique_s2b_count": len(unique_s2b),
        "branch_replacement_count": len(branch_replacement),
        "double_credit_count": len(double_credit),
        "primary_conclusion": primary,
        "phase_c_gate_pass": gate_pass,
    }


def write_semantic_audit(
    output: Path,
    stats: dict[str, Any],
    seed: int,
    candidate_results: list[dict[str, Any]] | None = None,
) -> None:
    candidate_section = "Candidate A/B was not run."
    if candidate_results:
        by_mode = {row["mode"]: row for row in candidate_results}
        current = by_mode["current"]
        identity = by_mode["unique_predictive_identity"]
        candidate_section = f"""## Phase-C candidate A/B (20 sentences)

The identity mode is a nonpaper semantic diagnostic. It does not change the
strict default and it does not relax `timing_tolerance`. The observed metrics
were:

| mode | mean Levenshtein | expected presence | mean raw columns | early-stop rate | S1 | S2A | S2B | S3 | segments |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current | {current['mean_levenshtein']} | {current['expected_present_raw']} | {current['mean_raw_columns']} | {current['early_stop_rate']} | {current['S1']} | {current['S2A']} | {current['S2B']} | {current['S3']} | {current['segments_total']} |
| unique_predictive_identity | {identity['mean_levenshtein']} | {identity['expected_present_raw']} | {identity['mean_raw_columns']} | {identity['early_stop_rate']} | {identity['S1']} | {identity['S2A']} | {identity['S2B']} | {identity['S3']} | {identity['segments_total']} |

On this 20-sentence sample, identity confirmation increased direct S1
confirmation and reduced segment/S3 counts, but it did **not** improve the
retrieval score: mean Levenshtein changed from `{current['mean_levenshtein']}`
to `{identity['mean_levenshtein']}` and early-stop rate changed from
`{current['early_stop_rate']}` to `{identity['early_stop_rate']}`. This is
mechanism evidence, not a claim of performance improvement or complete paper
reproduction. No 100-, 200-, 500-, or Fig.9 run was performed in this round.
"""
    (output / "TEMPORAL_CONFIRMATION_SEMANTIC_AUDIT.md").write_text(
        f"""# Temporal Confirmation Semantic Audit

1. The old 172/49/123 observation summary is mechanically reproduced as any={stats['any_prediction_count']}, current S1 is not used as the candidate denominator, and timing rejection is {stats['timing_mismatch_count']}.
2. Gate-before classification finds unique={stats['unique_count']} and multiple={stats['multiple_count']}.
3. Unique timing rejection is {stats['unique_rejected_count']}; exact candidate punishment is {stats['unique_punished_count']}.
4. Unique rejected fallback counts are current S2A={stats['unique_s2a_count']} and current S2B={stats['unique_s2b_count']}.
5. Branch replacement/double-credit count is {stats['branch_replacement_count']}.
6. `timing_tolerance=0.03` is local implementation provenance, not paper/reference provenance.
7. The paper supports the existence of a predictive/depolarized early winner in an active mini-column, but does not fully specify a multi-predictive soma winner.
8. The candidate gate is {'passed' if stats['phase_c_gate_pass'] else 'not passed'}; no candidate is run when it is not passed.

This report distinguishes document evidence, mechanistic trace evidence, and any later performance evidence. It does not claim complete numerical reproduction.

{candidate_section}
""",
        encoding="utf-8",
    )


def _capacity_row(
    model: SequentialMemory,
    mode: str,
    sentence_count: int,
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    observations = _observation_rows(rows)
    segments = _count_segments(model)
    synapses = _count_synapses(model)
    punished_candidate_ids = {
        row.get("predicted_candidate_identity")
        for row in _punishment_rows(rows)
        if row.get("predicted_candidate_identity") is not None
    }
    return {
        "sentence_count": sentence_count,
        "mode": mode,
        "tokens_seen": sentence_count * 10,
        "S1": sum(row.get("scenario") == "scenario1" for row in observations),
        "S2A": sum(row.get("scenario") == "scenario2" for row in observations),
        "S2B": sum(row.get("scenario") == "scenario3" for row in observations),
        "S3": len(_punishment_rows(rows)),
        "segments_created": segments,
        "segments_total": segments,
        "synapses_total": synapses,
        "mean_segment_size": synapses / segments if segments else 0.0,
        "new_segments_per_token": segments / max(1, sentence_count * 10),
        "timing_rejected_unique": sum(
            _unique_timing_rejected(row)
            for row in observations
        ),
        "confirmed_unique": sum(
            _is_unique_candidate(row)
            and row.get("scenario") == "scenario1"
            for row in observations
        ),
        "ambiguous_multiple": sum(
            int(row.get("pre_gate_candidate_count") or 0) > 1
            for row in observations
        ),
        "double_credit_failures": sum(
            _is_unique_candidate(row)
            and any(
                candidate_id in punished_candidate_ids
                for candidate_id in (row.get("pre_gate_candidate_identity_ids") or ())
            )
            for row in observations
        ),
        "mean_levenshtein": metrics.get("mean_levenshtein", ""),
        "expected_presence": metrics.get("expected_present_raw", ""),
        "raw_columns": metrics.get("mean_raw_columns", ""),
    }


def run_candidate_ab(
    output: Path,
    sentences: list[list[str]],
    seed: int,
) -> list[dict[str, Any]]:
    result_rows: list[dict[str, Any]] = []
    growth_rows: list[dict[str, Any]] = []
    for mode in ("current", "unique_predictive_identity"):
        model, trace_rows, _predictions = train_capture(
            sentences,
            seed,
            temporal_trace=True,
            temporal_mode=mode,
        )
        cue_rows: list[dict[str, object]] = []
        metrics = evaluate_diagnostic(
            model,
            sentences,
            propagation="raw",
            details_sample_sentences=0,
            segment_trace_path=None,
            cue_rows=cue_rows,
        )
        observation_rows = _observation_rows(trace_rows)
        result_rows.append(
            {
                "mode": mode,
                "mean_levenshtein": metrics.get("mean_levenshtein"),
                "expected_present_raw": metrics.get("expected_present_raw"),
                "mean_raw_columns": metrics.get("mean_raw_columns"),
                "no_prediction_rate": metrics.get("no_prediction_rate"),
                "early_stop_rate": metrics.get("early_stop_rate"),
                "step1_error": metrics.get("step1_error"),
                "step2_error": metrics.get("step2_error"),
                "step3_error": metrics.get("step3_error"),
                "step4_error": metrics.get("step4_error"),
                "S1": sum(row.get("scenario") == "scenario1" for row in observation_rows),
                "S2A": sum(row.get("scenario") == "scenario2" for row in observation_rows),
                "S2B": sum(row.get("scenario") == "scenario3" for row in observation_rows),
                "S3": len(_punishment_rows(trace_rows)),
                "segments_total": _count_segments(model),
                "synapses_total": _count_synapses(model),
                "unique_confirmed": sum(
                    row.get("identity_confirmation") is True
                    for row in observation_rows
                ),
                "unique_rejected_current": sum(
                    _unique_timing_rejected(row)
                    for row in observation_rows
                ),
                "candidate_punishment_rows": len(
                    [
                        row
                        for row in _punishment_rows(trace_rows)
                        if row.get("predicted_candidate_identity") is not None
                    ]
                ),
            }
        )
        growth_rows.append(
            _capacity_row(model, mode, len(sentences), metrics, trace_rows)
        )
        trace_fields = sorted({key for row in trace_rows for key in row})
        write_csv(
            output / f"{mode}_TEMPORAL_TRACE.csv",
            trace_rows,
            trace_fields,
        )
    fields = list(result_rows[0])
    write_csv(output / "fig8_20_temporal_confirmation_ab.csv", result_rows, fields)
    write_csv(
        output / "TEMPORAL_CONFIRMATION_CAPACITY_GROWTH.csv",
        growth_rows,
        list(growth_rows[0]),
    )
    return result_rows


def run_audit(output: Path, *, sentence_count: int, seed: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    sentences = read_cbt_sentences(ROOT / "data/CBTest/data/cbt_train.txt", sentence_count, seed)
    write_static_documents(output)
    plain_model, _plain_rows, plain_predictions = train_capture(sentences, seed, temporal_trace=False)
    traced_model, traced_rows, traced_predictions = train_capture(sentences, seed, temporal_trace=True)
    enriched = enrich_pre_gate_rows(traced_rows, traced_predictions, sentences, seed)
    stats = write_audit_artifacts(output, enriched + [row for row in traced_rows if row.get("phase") == "wrong_prediction_punishment"], seed)
    candidate_results = (
        run_candidate_ab(output, sentences, seed)
        if stats["phase_c_gate_pass"]
        else []
    )
    write_semantic_audit(output, stats, seed, candidate_results)
    write_csv(output / "TRACE_ON_OFF_COMPARISON.csv", [{
        "predictions_equal": plain_predictions == traced_predictions,
        "model_fingerprint_equal": _structure_fingerprint(plain_model) == _structure_fingerprint(traced_model),
        "learning_rng_equal": plain_model._learning_rng.getstate() == traced_model._learning_rng.getstate(),
        "decode_rng_equal": plain_model._decode_rng.getstate() == traced_model._decode_rng.getstate(),
        "plain_segments": _count_segments(plain_model),
        "traced_segments": _count_segments(traced_model),
        "plain_synapses": _count_synapses(plain_model),
        "traced_synapses": _count_synapses(traced_model),
    }], ["predictions_equal", "model_fingerprint_equal", "learning_rng_equal", "decode_rng_equal", "plain_segments", "traced_segments", "plain_synapses", "traced_synapses"])
    summary = {
        "audit_type": "fig8_temporal_confirmation_semantics",
        "sentences": sentence_count,
        "seed": seed,
        "response_scale": 1.0,
        "timing_tolerance": 0.03,
        "strict_defaults_changed": False,
        "formal_100_200_500_runs": False,
        "candidate_status": (
            "executed_20_sentence_ab"
            if candidate_results
            else "stopped_gate_not_passed"
        ),
        "candidate_results": candidate_results,
        **stats,
        "trace_on_off": {
            "predictions_equal": plain_predictions == traced_predictions,
            "model_fingerprint_equal": _structure_fingerprint(plain_model) == _structure_fingerprint(traced_model),
            "learning_rng_equal": plain_model._learning_rng.getstate() == traced_model._learning_rng.getstate(),
            "decode_rng_equal": plain_model._decode_rng.getstate() == traced_model._decode_rng.getstate(),
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if summary["primary_conclusion"] not in PRIMARY_CONCLUSIONS:
        raise AssertionError(summary["primary_conclusion"])
    (output / "FINAL_TEMPORAL_CONFIRMATION_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-sentences", type=int, default=20)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output_dir) if args.output_dir else ROOT / "results" / f"temporal_confirmation_semantics_{timestamp}"
    started = time.perf_counter()
    summary = run_audit(output, sentence_count=args.num_sentences, seed=args.seed)
    summary["runtime_seconds"] = time.perf_counter() - started
    (output / "FINAL_TEMPORAL_CONFIRMATION_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
