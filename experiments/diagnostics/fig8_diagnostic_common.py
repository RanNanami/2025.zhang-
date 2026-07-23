from __future__ import annotations

import gzip
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

from experiments.fig8_sentence_memory import levenshtein
from seqmem.dynamics import (
    kernel_peak_value,
    minimum_synchronous_synapses,
    unscaled_kernel_peak,
)
from seqmem.encoding import SSTDDiscreteEncoder, SymbolCode
from seqmem.model import (
    MemoryParams,
    PredictionTrace,
    ReinforcementTrace,
    SequentialMemory,
)


def parse_response_scale(value: str) -> float | None:
    return None if value == "normalized" else float(value)


def response_scale_label(value: float | None) -> str:
    return "normalized" if value is None else f"{value:g}"


def kernel_diagnostic(scale: float | None) -> dict[str, object]:
    tau_m = 0.10
    tau_s = 0.02
    _peak_time, unscaled_peak = unscaled_kernel_peak(tau_m, tau_s)
    peak = kernel_peak_value(tau_m, tau_s, scale)
    contribution = 0.5 * peak
    return {
        "response_scale": response_scale_label(scale),
        "tau_m": tau_m,
        "tau_s": tau_s,
        "unscaled_kernel_peak": unscaled_peak,
        "kernel_peak": peak,
        "single_w0_peak_contribution": contribution,
        "min_initial_synapses_to_threshold": minimum_synchronous_synapses(
            weight=0.5,
            threshold=1.0,
            kernel_peak=peak,
            voltage_tolerance=2e-5,
        ),
    }


def build_model(
    scale: float | None,
    seed: int,
    *,
    scenario1_contribution_mode: str = "arrival-window",
    capture_prediction_contributions: bool = False,
    synapse_delay_mode: str = "current-delay",
    capture_branch_diagnostics: bool = False,
) -> SequentialMemory:
    return SequentialMemory(
        encoder=SSTDDiscreteEncoder(num_columns=100, k=10, seed=seed),
        num_neurons_per_column=10,
        params=MemoryParams(
            l_match=3,
            forgetting_threshold=500.0,
            response_scale=scale,
            scenario1_contribution_mode=scenario1_contribution_mode,
            capture_prediction_contributions=capture_prediction_contributions,
            synapse_delay_mode=synapse_delay_mode,
            capture_branch_diagnostics=capture_branch_diagnostics,
        ),
        tie_break_seed=seed,
    )


class ReinforcementAccumulator:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []
        self.repetitions: Counter[int] = Counter()

    def __call__(self, item: ReinforcementTrace) -> None:
        if item.scenario != "scenario1":
            return
        self.repetitions[item.segment_identity] += 1
        self.rows.append(
            {
                "scenario": item.scenario,
                "target_column": item.target_column,
                "target_neuron": item.target_neuron,
                "segment_identity": item.segment_identity,
                "contributing_synapse_count": item.contributing_synapse_count,
                "mean_weight_before": _mean(item.weights_before),
                "mean_weight_after": _mean(item.weights_after),
                "reinforcement_number": self.repetitions[item.segment_identity],
                "contribution_mode": item.contribution_mode,
                "prediction_candidate_identity": item.prediction_candidate_identity,
                "prediction_crossing_time": item.prediction_crossing_time,
                "actual_positive_synapse_count": (
                    item.actual_positive_synapse_count
                ),
                "actual_positive_weakened_count": (
                    item.actual_positive_weakened_count
                ),
                "strengthened_synapse_count": item.strengthened_synapse_count,
                "weakened_synapse_count": item.weakened_synapse_count,
            }
        )

    def summary(self) -> dict[str, float]:
        if not self.rows:
            empty = {
                "scenario1_events": 0.0,
                "scenario1_two_contributor_fraction": 0.0,
                "scenario1_mean_weight_before": 0.0,
                "scenario1_mean_weight_after": 0.0,
                "scenario1_mean_reinforcement_number": 0.0,
                "scenario1_max_reinforcement_number": 0.0,
                "scenario1_actual_positive_weakened_fraction": 0.0,
                "scenario1_mean_strengthened_synapses": 0.0,
                "scenario1_mean_weakened_synapses": 0.0,
                "scenario1_missing_prediction_candidate_events": 0.0,
                "scenario1_contributor_5_plus_fraction": 0.0,
            }
            empty.update(
                {
                    f"scenario1_contributor_{label}_fraction": 0.0
                    for label in [*map(str, range(10)), "10_plus"]
                }
            )
            return empty
        contributor_counts = [
            int(row["contributing_synapse_count"]) for row in self.rows
        ]
        reinforcement_numbers = [
            float(row["reinforcement_number"]) for row in self.rows
        ]
        actual_positive = sum(
            int(row["actual_positive_synapse_count"]) for row in self.rows
        )
        actual_positive_weakened = sum(
            int(row["actual_positive_weakened_count"]) for row in self.rows
        )
        summary = {
            "scenario1_events": float(len(self.rows)),
            "scenario1_two_contributor_fraction": _fraction(
                count == 2 for count in contributor_counts
            ),
            "scenario1_mean_weight_before": _mean(
                float(row["mean_weight_before"]) for row in self.rows
            ),
            "scenario1_mean_weight_after": _mean(
                float(row["mean_weight_after"]) for row in self.rows
            ),
            "scenario1_mean_reinforcement_number": _mean(
                reinforcement_numbers
            ),
            "scenario1_max_reinforcement_number": max(reinforcement_numbers),
            "scenario1_actual_positive_weakened_fraction": (
                actual_positive_weakened / actual_positive
                if actual_positive
                else 0.0
            ),
            "scenario1_mean_strengthened_synapses": _mean(
                float(row["strengthened_synapse_count"]) for row in self.rows
            ),
            "scenario1_mean_weakened_synapses": _mean(
                float(row["weakened_synapse_count"]) for row in self.rows
            ),
            "scenario1_missing_prediction_candidate_events": float(
                sum(
                    row["prediction_candidate_identity"] is None
                    for row in self.rows
                )
            ),
            "scenario1_contributor_5_plus_fraction": _fraction(
                count >= 5 for count in contributor_counts
            ),
        }
        summary.update(
            {
                f"scenario1_contributor_{count}_fraction": _fraction(
                    value == count for value in contributor_counts
                )
                for count in range(10)
            }
        )
        summary["scenario1_contributor_10_plus_fraction"] = _fraction(
            value >= 10 for value in contributor_counts
        )
        return summary


def train_once(
    model: SequentialMemory,
    sentences: list[list[str]],
) -> ReinforcementAccumulator:
    reinforcement = ReinforcementAccumulator()
    model.reinforcement_trace_callback = reinforcement
    try:
        for sentence in sentences:
            model.reset_state()
            for word in sentence:
                model.predict_code()
                model.observe(word)
    finally:
        model.reinforcement_trace_callback = None
    return reinforcement


def _decode_without_side_effect(
    model: SequentialMemory,
    code: SymbolCode,
) -> tuple[str | None, list[tuple[int, int, str]]]:
    rng_state = model._decode_rng.getstate()
    ranking_before = model.last_symbol_ranking.copy()
    try:
        selected = model.decode_symbol_from_prediction(code)
        return selected, model.last_symbol_ranking.copy()
    finally:
        model._decode_rng.setstate(rng_state)
        model.last_symbol_ranking = ranking_before


def _word_rank(
    ranking: list[tuple[int, int, str]], word: str
) -> tuple[int | None, int, int]:
    for rank, (timed, columns, symbol) in enumerate(ranking, start=1):
        if symbol == word:
            return rank, timed, columns
    return None, 0, 0


def _trace_stats(trace: PredictionTrace) -> dict[str, float]:
    accepted = [item for item in trace.segments if item.entered_raw_prediction]
    counts = [item.temporally_contributing_synapse_count for item in accepted]
    margins = [
        item.threshold_margin
        for item in accepted
        if item.threshold_margin is not None
    ]
    return {
        "accepted_segment_count": float(len(accepted)),
        "mean_contributors": _mean(counts),
        "median_contributors": float(statistics.median(counts)) if counts else 0.0,
        "two_contributor_fraction": _fraction(count == 2 for count in counts),
        "three_contributor_fraction": _fraction(count == 3 for count in counts),
        "four_plus_contributor_fraction": _fraction(count >= 4 for count in counts),
        "average_threshold_margin": _mean(margins),
    }


def _stream_trace(
    handle: TextIO | None,
    trace: PredictionTrace,
    **context: object,
) -> None:
    if handle is None:
        return
    for item in trace.segments:
        row = {"diagnostic_label": "nonpaper diagnostic", **context, **asdict(item)}
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_diagnostic(
    model: SequentialMemory,
    sentences: list[list[str]],
    *,
    propagation: str,
    details_sample_sentences: int,
    segment_trace_path: Path | None,
    cue_rows: list[dict[str, object]],
) -> dict[str, object]:
    if propagation not in {"raw", "eventwise-inhibited"}:
        raise ValueError(f"unsupported propagation: {propagation}")
    snapshot = model.snapshot_transient_state()
    started = time.perf_counter()
    trace_handle: TextIO | None = None
    if segment_trace_path is not None:
        segment_trace_path.parent.mkdir(parents=True, exist_ok=True)
        if segment_trace_path.suffix == ".gz":
            trace_handle = gzip.open(
                segment_trace_path, "wt", encoding="utf-8"
            )
        else:
            trace_handle = segment_trace_path.open("w", encoding="utf-8")

    distances: list[int] = []
    step_wrong: Counter[int] = Counter()
    step_total: Counter[int] = Counter()
    expected_absent_raw = 0
    expected_absent_after = 0
    expected_total = 0
    raw_columns: list[int] = []
    raw_events: list[int] = []
    propagated_events: list[int] = []
    active_cells: list[int] = []
    ranked_counts: list[int] = []
    cue_burst_cells: list[int] = []
    accepted_contributors: list[int] = []
    accepted_margins: list[float] = []
    no_prediction_steps = 0
    total_requested_steps = len(sentences) * 4
    early_stops = 0
    peak_cache_size = 0
    word_density: defaultdict[str, list[int]] = defaultdict(list)
    all_cue_rows: list[dict[str, object]] = []

    try:
        for sentence_index, sentence in enumerate(sentences, start=1):
            sampled = sentence_index <= details_sample_sentences
            model.reset_state()
            model.predict_code()
            model.observe(sentence[0], learn=False)
            pending: tuple[SymbolCode, PredictionTrace] | None = None

            for current_index in range(6):
                expected_index = current_index + 1
                trace = PredictionTrace()
                previous_active_count = len(model.previous_active_cells)
                previous_winner_count = len(model.previous_winners)
                raw = model.predict_code(trace=trace)
                peak_cache_size = max(peak_cache_size, len(model._spike_response_cache))
                if raw is None:
                    selected = None
                    ranking: list[tuple[int, int, str]] = []
                else:
                    selected, ranking = _decode_without_side_effect(model, raw)
                rank, timed, columns = _word_rank(ranking, sentence[expected_index])
                predicted_cells = (
                    model.prediction_active_cells(raw) if raw is not None else {}
                )
                burst_cells = 0
                burst_columns = 0
                if expected_index < 6:
                    model.observe(sentence[expected_index], learn=False)
                    extra = set(model.previous_active_cells) - set(predicted_cells)
                    burst_cells = len(extra)
                    neurons_per_column = len(model.columns[0].neurons)
                    burst_columns = len(
                        {cell // neurons_per_column for cell in extra}
                    )
                    cue_burst_cells.append(burst_cells)
                stats = _trace_stats(trace)
                cue_row = {
                    "diagnostic_label": "nonpaper diagnostic",
                    "sentence_index": sentence_index,
                    "cue_transition": f"{current_index + 1}->{expected_index + 1}",
                    "current_word": sentence[current_index],
                    "expected_next_word": sentence[expected_index],
                    "previous_active_cell_count": previous_active_count,
                    "previous_winner_count": previous_winner_count,
                    "burst_column_count": burst_columns,
                    "burst_cell_count": burst_cells,
                    "raw_event_count": len(raw.events) if raw is not None else 0,
                    "raw_predicted_column_count": (
                        len({event.column for event in raw.events}) if raw else 0
                    ),
                    "prediction_active_cell_count": len(predicted_cells),
                    "number_of_ranked_candidates": len(ranking),
                    "expected_word_rank": rank if rank is not None else "",
                    "timed_overlap_of_expected_word": timed,
                    "column_overlap_of_expected_word": columns,
                    "expected_word_present": rank is not None,
                    "selected_word": selected or "",
                    "selected_word_correct": selected == sentence[expected_index],
                    **stats,
                }
                all_cue_rows.append(cue_row)
                if sampled:
                    cue_rows.append(cue_row)
                    _stream_trace(
                        trace_handle,
                        trace,
                        sentence_index=sentence_index,
                        phase="cue",
                        step=current_index + 1,
                    )
                word_density[sentence[current_index]].append(
                    int(cue_row["raw_predicted_column_count"])
                )
                if current_index == 5 and raw is not None:
                    pending = (raw, trace)

            recalled: list[str] = []
            for suffix_step in range(4):
                if suffix_step == 0 and pending is not None:
                    raw, trace = pending
                else:
                    trace = PredictionTrace()
                    raw = model.predict_code(trace=trace)
                    peak_cache_size = max(
                        peak_cache_size, len(model._spike_response_cache)
                    )
                if raw is None:
                    remaining = 4 - suffix_step
                    no_prediction_steps += remaining
                    early_stops += 1
                    expected_total += remaining
                    expected_absent_raw += remaining
                    expected_absent_after += remaining
                    for missing_step in range(suffix_step + 1, 5):
                        step_total[missing_step] += 1
                        step_wrong[missing_step] += 1
                    break

                expected = sentence[6 + suffix_step]
                propagated = raw
                if propagation == "eventwise-inhibited":
                    propagated = model.select_prediction_events(raw, trace=trace)
                    if propagated is None:
                        _raw_selected, ranking_raw = _decode_without_side_effect(
                            model, raw
                        )
                        raw_rank, _raw_timed, _raw_columns = _word_rank(
                            ranking_raw, expected
                        )
                        remaining = 4 - suffix_step
                        no_prediction_steps += remaining
                        early_stops += 1
                        expected_total += remaining
                        expected_absent_raw += (raw_rank is None) + remaining - 1
                        expected_absent_after += remaining
                        for missing_step in range(suffix_step + 1, 5):
                            step_total[missing_step] += 1
                            step_wrong[missing_step] += 1
                        break

                if propagation == "raw":
                    selected = model.decode_symbol_from_prediction(raw)
                    ranking_raw = model.last_symbol_ranking.copy()
                    ranking_after = ranking_raw
                else:
                    _raw_selected, ranking_raw = _decode_without_side_effect(
                        model, raw
                    )
                    selected = model.decode_symbol_from_prediction(propagated)
                    ranking_after = model.last_symbol_ranking.copy()
                raw_rank, _raw_timed, _raw_columns = _word_rank(
                    ranking_raw, expected
                )
                after_rank, _after_timed, _after_columns = _word_rank(
                    ranking_after, expected
                )
                expected_total += 1
                expected_absent_raw += raw_rank is None
                expected_absent_after += after_rank is None
                step_total[suffix_step + 1] += 1
                step_wrong[suffix_step + 1] += selected != expected
                recalled.append(selected or "")

                raw_events.append(len(raw.events))
                raw_columns.append(len({event.column for event in raw.events}))
                propagated_events.append(len(propagated.events))
                active = model.prediction_active_cells(propagated)
                active_cells.append(len(active))
                ranked_counts.append(len(ranking_after))
                accepted = [
                    item for item in trace.segments if item.entered_raw_prediction
                ]
                accepted_contributors.extend(
                    item.temporally_contributing_synapse_count for item in accepted
                )
                accepted_margins.extend(
                    item.threshold_margin
                    for item in accepted
                    if item.threshold_margin is not None
                )
                if sampled:
                    _stream_trace(
                        trace_handle,
                        trace,
                        sentence_index=sentence_index,
                        phase="suffix",
                        step=suffix_step + 1,
                        propagation=propagation,
                    )
                if not model.advance_prediction(propagated) and suffix_step < 3:
                    remaining = 3 - suffix_step
                    no_prediction_steps += remaining
                    early_stops += 1
                    expected_total += remaining
                    expected_absent_raw += remaining
                    expected_absent_after += remaining
                    for missing_step in range(suffix_step + 2, 5):
                        step_total[missing_step] += 1
                        step_wrong[missing_step] += 1
                    break

            distances.append(levenshtein(recalled, sentence[6:]))
    finally:
        if trace_handle is not None:
            trace_handle.close()
        model.restore_transient_state(snapshot)

    cue_bursts = [float(row["burst_cell_count"]) for row in all_cue_rows]
    cue_densities = [
        float(row["raw_predicted_column_count"]) for row in all_cue_rows
    ]
    word_frequency = Counter(word for sentence in sentences for word in sentence)
    density_frequency_pairs = [
        (float(word_frequency[word]), float(value))
        for word, values in word_density.items()
        for value in values
    ]
    return {
        "diagnostic_label": "nonpaper diagnostic",
        "propagation_mode": propagation,
        "sentences": len(sentences),
        "mean_levenshtein": _mean(distances),
        **{
            f"step{step}_error": (
                step_wrong[step] / step_total[step] if step_total[step] else 1.0
            )
            for step in range(1, 5)
        },
        "expected_absent_rate": (
            expected_absent_raw / expected_total if expected_total else 1.0
        ),
        "expected_present_raw": (
            1.0 - expected_absent_raw / expected_total if expected_total else 0.0
        ),
        "expected_present_after_inhibition": (
            1.0 - expected_absent_after / expected_total
            if expected_total
            else 0.0
        ),
        "mean_raw_events": _mean(raw_events),
        "mean_propagated_events": _mean(propagated_events),
        "mean_raw_columns": _mean(raw_columns),
        "p50_raw_columns": _percentile(raw_columns, 0.50),
        "p90_raw_columns": _percentile(raw_columns, 0.90),
        "p99_raw_columns": _percentile(raw_columns, 0.99),
        "mean_active_cells": _mean(active_cells),
        "mean_ranked_candidate_count": _mean(ranked_counts),
        "mean_burst_cells_after_cue": _mean(cue_burst_cells),
        "two_contributor_crossing_fraction": _fraction(
            count == 2 for count in accepted_contributors
        ),
        "three_contributor_crossing_fraction": _fraction(
            count == 3 for count in accepted_contributors
        ),
        "four_plus_contributor_fraction": _fraction(
            count >= 4 for count in accepted_contributors
        ),
        "average_threshold_margin": _mean(accepted_margins),
        "proportion_columns_with_prediction": _mean(raw_columns) / 100.0,
        "no_prediction_rate": no_prediction_steps / total_requested_steps,
        "early_stop_rate": early_stops / len(sentences),
        "burst_density_correlation": _correlation(cue_bursts, cue_densities),
        "word_frequency_density_correlation": _pair_correlation(
            density_frequency_pairs
        ),
        "runtime_seconds": time.perf_counter() - started,
        "peak_cache_size": peak_cache_size,
    }


def _mean(values) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def _fraction(values) -> float:
    materialized = list(values)
    return (
        sum(bool(value) for value in materialized) / len(materialized)
        if materialized
        else 0.0
    )


def _percentile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return float(ordered[low])
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_var = sum((x - left_mean) ** 2 for x in left)
    right_var = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_var * right_var)
    return numerator / denominator if denominator else 0.0


def _pair_correlation(pairs: list[tuple[float, float]]) -> float:
    return _correlation(
        [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    )
