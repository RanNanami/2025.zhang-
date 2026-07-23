from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.diagnostics.fig8_diagnostic_common import (  # noqa: E402
    build_model,
    kernel_diagnostic,
    train_once,
)
from experiments.fig8_sentence_memory import (  # noqa: E402
    levenshtein,
    read_cbt_sentences,
)
from seqmem.encoding import SpikeEvent, SymbolCode  # noqa: E402
from seqmem.model import (  # noqa: E402
    PredictionCandidate,
    PredictionTrace,
    SequentialMemory,
)


RANKING_MODES = (
    "current-score",
    "actual-peak",
    "earliest-crossing",
)
DELAY_MODES = (
    "current-delay",
    "peak-aligned-delay",
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
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


def memory_structure(model: SequentialMemory) -> tuple[object, ...]:
    return tuple(
        (
            segment.active,
            segment.target_time,
            tuple(
                sorted(
                    (source, synapse.weight, synapse.delay, synapse.age)
                    for source, synapse in segment.synapses.items()
                )
            ),
        )
        for column in model.columns
        for neuron in column.neurons
        for segment in neuron.segments
    )


def _all_candidates(
    model: SequentialMemory,
) -> list[tuple[int, PredictionCandidate]]:
    return [
        (column, candidate)
        for column, candidates in model.last_prediction_candidates.items()
        for candidate in candidates
    ]


def _candidate_order(
    model: SequentialMemory,
    item: tuple[int, PredictionCandidate],
    mode: str,
) -> tuple[float, int, int, float]:
    column, candidate = item
    return (
        *model.prediction_candidate_sort_key(candidate, mode),
        column,
        candidate.neuron_index,
        candidate.time,
    )


def _candidate_rank(
    model: SequentialMemory,
    candidates: list[tuple[int, PredictionCandidate]],
    expected: SpikeEvent,
    mode: str,
) -> int | None:
    competition = [
        item
        for item in candidates
        if abs(item[1].time - expected.time) <= model.params.timing_tolerance
    ]
    competition.sort(key=lambda item: _candidate_order(model, item, mode))
    for rank, (column, candidate) in enumerate(competition, start=1):
        if (
            column == expected.column
            and abs(candidate.time - expected.time)
            <= model.params.timing_tolerance
        ):
            return rank
    return None


def _selected_candidate_ids(
    model: SequentialMemory,
    propagated: SymbolCode | None,
) -> set[int]:
    if propagated is None:
        return set()
    selected: set[int] = set()
    for event in propagated.events:
        for candidate in model.last_prediction_candidates.get(event.column, []):
            if round(candidate.time, 12) == round(event.time, 12):
                selected.add(id(candidate))
    return selected


def _event_json(code: SymbolCode | None) -> str:
    if code is None:
        return "[]"
    return json.dumps(
        [
            {"column": event.column, "time": event.time}
            for event in code.events
        ],
        separators=(",", ":"),
    )


class CompetitionAccumulator:
    def __init__(self, model: SequentialMemory) -> None:
        self.model = model
        self.current_scores: list[float] = []
        self.actual_peaks: list[float] = []
        self.crossing_times: list[float] = []
        self.contributor_counts: list[float] = []
        self.score_values: set[float] = set()
        self.group_unique_scores: list[int] = []
        self.group_top_ties: list[int] = []
        self.correct_event_rows: list[dict[str, object]] = []
        self.cue_rows: list[dict[str, object]] = []

    def add_candidates(
        self,
        candidates: list[tuple[int, PredictionCandidate]],
    ) -> None:
        for _column, candidate in candidates:
            if (
                candidate.peak_dendritic_potential is None
                or candidate.dendritic_crossing_time is None
            ):
                raise RuntimeError("accepted candidate is missing competition trace")
            contributors = sum(
                item.psp_contribution > 0.0
                for item in candidate.crossing_synapse_contributions
            )
            self.current_scores.append(candidate.score)
            self.actual_peaks.append(candidate.peak_dendritic_potential)
            self.crossing_times.append(candidate.dendritic_crossing_time)
            self.contributor_counts.append(float(contributors))
            self.score_values.add(round(candidate.score, 12))
        for event_time in self.model.encoder.event_times:
            group = [
                candidate
                for _column, candidate in candidates
                if abs(candidate.time - event_time)
                <= self.model.params.timing_tolerance
            ]
            if not group:
                continue
            unique = {round(candidate.score, 12) for candidate in group}
            top = max(candidate.score for candidate in group)
            ties = sum(abs(candidate.score - top) <= 1e-12 for candidate in group)
            self.group_unique_scores.append(len(unique))
            self.group_top_ties.append(ties)

    def summary(self, kernel_peak_time: float) -> dict[str, object]:
        residuals = [
            float(row["timing_residual"])
            for row in self.correct_event_rows
            if row["timing_residual"] != ""
        ]
        return {
            "accepted_candidate_count": len(self.current_scores),
            "score_exact_threshold_fraction": _fraction(
                abs(score - self.model.params.dendrite_threshold) <= 1e-12
                for score in self.current_scores
            ),
            "score_near_threshold_fraction": _fraction(
                abs(score - self.model.params.dendrite_threshold)
                <= self.model.params.integration_voltage_tolerance
                for score in self.current_scores
            ),
            "score_unique_value_count": len(self.score_values),
            "mean_group_score_unique_values": _mean(self.group_unique_scores),
            "top_score_tie_group_fraction": _fraction(
                count > 1 for count in self.group_top_ties
            ),
            "mean_top_score_tie_count": _mean(self.group_top_ties),
            "score_actual_peak_correlation": _correlation(
                self.current_scores, self.actual_peaks
            ),
            "score_crossing_time_correlation": _correlation(
                self.current_scores, self.crossing_times
            ),
            "score_contributor_count_correlation": _correlation(
                self.current_scores, self.contributor_counts
            ),
            "correct_event_column_missing_rate": _fraction(
                not bool(row["correct_column_present"])
                for row in self.correct_event_rows
            ),
            "correct_event_timing_mismatch_rate": _fraction(
                bool(row["correct_column_present"])
                and bool(row["column_present_but_timing_mismatch"])
                for row in self.correct_event_rows
            ),
            "correct_event_timed_presence_rate": _fraction(
                int(row["matching_candidate_count"]) > 0
                for row in self.correct_event_rows
            ),
            "timing_residual_mean": _mean(residuals),
            "timing_residual_p50": _percentile(residuals, 0.50),
            "timing_residual_p90": _percentile(residuals, 0.90),
            "timing_residual_p99": _percentile(residuals, 0.99),
            "timing_residual_kernel_peak_offset_mean": _mean(
                residual - kernel_peak_time for residual in residuals
            ),
            "timing_residual_near_kernel_peak_fraction": _fraction(
                abs(residual - kernel_peak_time)
                <= self.model.params.timing_tolerance
                for residual in residuals
            ),
            **{
                f"correct_event_mean_rank_{mode.replace('-', '_')}": _mean(
                    float(row[f"rank_{mode.replace('-', '_')}"])
                    for row in self.correct_event_rows
                    if row[f"rank_{mode.replace('-', '_')}"] != ""
                )
                for mode in RANKING_MODES
            },
        }


def _trace_prediction(
    model: SequentialMemory,
    raw: SymbolCode | None,
    propagated: SymbolCode | None,
    expected_code: SymbolCode,
    accumulator: CompetitionAccumulator,
    expected_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    *,
    sentence_index: int,
    phase: str,
    step: int,
    ranking_mode: str,
    delay_mode: str,
) -> None:
    candidates = _all_candidates(model) if raw is not None else []
    selected_ids = _selected_candidate_ids(model, propagated)
    accumulator.add_candidates(candidates)
    common = {
        "diagnostic_label": "nonpaper diagnostic",
        "sentences": "",
        "sentence_index": sentence_index,
        "phase": phase,
        "step": step,
        "ranking_mode": ranking_mode,
        "delay_mode": delay_mode,
    }

    for expected in expected_code.events:
        same_column = [
            candidate
            for column, candidate in candidates
            if column == expected.column
        ]
        nearest = min(
            same_column,
            key=lambda candidate: abs(candidate.time - expected.time),
            default=None,
        )
        matching = [
            candidate
            for candidate in same_column
            if abs(candidate.time - expected.time)
            <= model.params.timing_tolerance
        ]
        selected = [candidate for candidate in matching if id(candidate) in selected_ids]
        row = {
            **common,
            "expected_column": expected.column,
            "expected_time": expected.time,
            "correct_column_present": bool(same_column),
            "column_present_but_timing_mismatch": bool(same_column)
            and not matching,
            "nearest_predicted_time": nearest.time if nearest else "",
            "timing_residual": (
                nearest.time - expected.time if nearest is not None else ""
            ),
            "matching_candidate_count": len(matching),
            "distinct_neuron_count": len(
                {candidate.neuron_index for candidate in matching}
            ),
            "selected_neuron": selected[0].neuron_index if selected else "",
            **{
                f"rank_{mode.replace('-', '_')}": (
                    rank
                    if (
                        rank := _candidate_rank(
                            model, candidates, expected, mode
                        )
                    )
                    is not None
                    else ""
                )
                for mode in RANKING_MODES
            },
        }
        expected_rows.append(row)
        accumulator.correct_event_rows.append(row)

    for event in raw.events if raw is not None else ():
        event_candidates = [
            candidate
            for column, candidate in candidates
            if column == event.column
            and round(candidate.time, 12) == round(event.time, 12)
        ]
        if not event_candidates:
            continue
        candidate = event_candidates[0]
        nearest_event_time = min(
            model.encoder.event_times,
            key=lambda value: abs(value - event.time),
        )
        competition = [
            item
            for item in candidates
            if abs(item[1].time - nearest_event_time)
            <= model.params.timing_tolerance
        ]
        scores = [item[1].score for item in competition]
        top = max(scores) if scores else candidate.score
        prediction_rows.append(
            {
                **common,
                "predicted_column": event.column,
                "predicted_time": event.time,
                "candidate_count": len(competition),
                "score_unique_value_count": len(
                    {round(score, 12) for score in scores}
                ),
                "top_score_tie_count": sum(
                    abs(score - top) <= 1e-12 for score in scores
                ),
                "current_returned_score": candidate.score,
                "actual_peak_dendritic_potential": (
                    candidate.peak_dendritic_potential
                ),
                "threshold_margin": candidate.threshold_margin,
                "first_crossing_time": candidate.dendritic_crossing_time,
                "soma_firing_time": candidate.predicted_soma_firing_time,
                "contributor_count": sum(
                    item.psp_contribution > 0.0
                    for item in candidate.crossing_synapse_contributions
                ),
                "selected_by_ranking": id(candidate) in selected_ids,
            }
        )


def evaluate_competition(
    model: SequentialMemory,
    sentences: list[list[str]],
    *,
    ranking_mode: str,
    delay_mode: str,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    snapshot = model.snapshot_transient_state()
    structure_before = memory_structure(model)
    accumulator = CompetitionAccumulator(model)
    expected_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    distances: list[int] = []
    step_wrong: Counter[int] = Counter()
    raw_columns: list[int] = []
    propagated_columns: list[int] = []
    expected_word_present: list[bool] = []
    no_prediction_steps = 0
    early_stops = 0
    started = time.perf_counter()

    try:
        for sentence_index, sentence in enumerate(sentences, start=1):
            model.reset_state()
            model.predict_code()
            model.observe(sentence[0], learn=False)

            for cue_step in range(1, 6):
                expected_code = model.encoder.encode(sentence[cue_step])
                previous_active = len(model.previous_active_cells)
                previous_winners = len(model.previous_winners)
                trace = PredictionTrace()
                raw = model.predict_code(trace=trace)
                predicted_cells = (
                    model.prediction_active_cells(raw) if raw is not None else {}
                )
                _trace_prediction(
                    model,
                    raw,
                    raw,
                    expected_code,
                    accumulator,
                    expected_rows,
                    prediction_rows,
                    sentence_index=sentence_index,
                    phase="cue",
                    step=cue_step,
                    ranking_mode=ranking_mode,
                    delay_mode=delay_mode,
                )
                model.observe(sentence[cue_step], learn=False)
                extra = set(model.previous_active_cells) - set(predicted_cells)
                neurons_per_column = len(model.columns[0].neurons)
                transition_rows.append(
                    {
                        "diagnostic_label": "nonpaper diagnostic",
                        "sentence_index": sentence_index,
                        "phase": "cue",
                        "step": cue_step,
                        "transition": f"{cue_step}->{cue_step + 1}",
                        "ranking_mode": ranking_mode,
                        "delay_mode": delay_mode,
                        "previous_active_cell_count": previous_active,
                        "previous_winner_count": previous_winners,
                        "expected_column_time_events": _event_json(expected_code),
                        "raw_predicted_events": _event_json(raw),
                        "predicted_event_count": len(raw.events) if raw else 0,
                        "raw_predicted_column_count": len(
                            {event.column for event in raw.events}
                        )
                        if raw
                        else 0,
                        "burst_column_count": len(
                            {cell // neurons_per_column for cell in extra}
                        ),
                        "burst_cell_count": len(extra),
                        "active_cell_count": len(model.previous_active_cells),
                        "stopped_reason": "completed",
                    }
                )
                accumulator.cue_rows.append(transition_rows[-1])

            recalled: list[str] = []
            for suffix_step in range(1, 5):
                expected_word = sentence[5 + suffix_step]
                expected_code = model.encoder.encode(expected_word)
                previous_active = len(model.previous_active_cells)
                previous_winners = len(model.previous_winners)
                trace = PredictionTrace()
                raw = model.predict_code(trace=trace)
                if raw is None:
                    remaining = 5 - suffix_step
                    no_prediction_steps += remaining
                    early_stops += 1
                    expected_word_present.extend(False for _ in range(remaining))
                    for missing_step in range(suffix_step, 5):
                        missing_code = model.encoder.encode(sentence[5 + missing_step])
                        _trace_prediction(
                            model,
                            None,
                            None,
                            missing_code,
                            accumulator,
                            expected_rows,
                            prediction_rows,
                            sentence_index=sentence_index,
                            phase="suffix",
                            step=missing_step,
                            ranking_mode=ranking_mode,
                            delay_mode=delay_mode,
                        )
                        step_wrong[missing_step] += 1
                    transition_rows.append(
                        {
                            "diagnostic_label": "nonpaper diagnostic",
                            "sentence_index": sentence_index,
                            "phase": "suffix",
                            "step": suffix_step,
                            "ranking_mode": ranking_mode,
                            "delay_mode": delay_mode,
                            "previous_active_cell_count": previous_active,
                            "previous_winner_count": previous_winners,
                            "expected_column_time_events": _event_json(expected_code),
                            "raw_predicted_events": "[]",
                            "predicted_event_count": 0,
                            "raw_predicted_column_count": 0,
                            "burst_column_count": 0,
                            "burst_cell_count": 0,
                            "active_cell_count": previous_active,
                            "stopped_reason": "no_raw_prediction",
                        }
                    )
                    break

                propagated = model.select_prediction_events(
                    raw,
                    trace=trace,
                    ranking_mode=ranking_mode,
                )
                _trace_prediction(
                    model,
                    raw,
                    propagated,
                    expected_code,
                    accumulator,
                    expected_rows,
                    prediction_rows,
                    sentence_index=sentence_index,
                    phase="suffix",
                    step=suffix_step,
                    ranking_mode=ranking_mode,
                    delay_mode=delay_mode,
                )
                raw_column_count = len({event.column for event in raw.events})
                raw_columns.append(raw_column_count)
                propagated_column_count = (
                    len({event.column for event in propagated.events})
                    if propagated is not None
                    else 0
                )
                propagated_columns.append(propagated_column_count)
                decoded = None
                ranking: list[tuple[int, int, str]] = []
                if propagated is not None:
                    decoded = model.decode_symbol_from_prediction(propagated)
                    ranking = model.last_symbol_ranking.copy()
                expected_word_present.append(
                    any(item[2] == expected_word for item in ranking)
                )
                step_wrong[suffix_step] += decoded != expected_word
                if decoded is not None:
                    recalled.append(decoded)
                advance_success = (
                    decoded is not None
                    and propagated is not None
                    and model.advance_prediction(propagated)
                )
                transition_rows.append(
                    {
                        "diagnostic_label": "nonpaper diagnostic",
                        "sentence_index": sentence_index,
                        "phase": "suffix",
                        "step": suffix_step,
                        "ranking_mode": ranking_mode,
                        "delay_mode": delay_mode,
                        "previous_active_cell_count": previous_active,
                        "previous_winner_count": previous_winners,
                        "expected_column_time_events": _event_json(expected_code),
                        "raw_predicted_events": _event_json(raw),
                        "predicted_event_count": len(raw.events),
                        "raw_predicted_column_count": raw_column_count,
                        "propagated_column_count": propagated_column_count,
                        "burst_column_count": 0,
                        "burst_cell_count": 0,
                        "active_cell_count": len(model.previous_active_cells),
                        "decoded_word": decoded or "",
                        "expected_word_present": expected_word_present[-1],
                        "advance_success": advance_success,
                        "stopped_reason": (
                            "completed" if advance_success else "no_prediction_active_cells"
                        ),
                    }
                )
                if not advance_success and suffix_step < 4:
                    remaining = 4 - suffix_step
                    no_prediction_steps += remaining
                    early_stops += 1
                    expected_word_present.extend(False for _ in range(remaining))
                    for missing_step in range(suffix_step + 1, 5):
                        step_wrong[missing_step] += 1
                    break
            distances.append(levenshtein(recalled, sentence[6:]))
    finally:
        model.restore_transient_state(snapshot)

    if memory_structure(model) != structure_before:
        raise RuntimeError("competition evaluation modified long-term memory")
    summary = {
        "diagnostic_label": "nonpaper diagnostic",
        "sentences": len(sentences),
        "ranking_mode": ranking_mode,
        "delay_mode": delay_mode,
        "kernel_peak_time": model._kernel_peak_time,
        "mean_levenshtein": _mean(distances),
        **{
            f"step{step}_error": step_wrong[step] / len(sentences)
            for step in range(1, 5)
        },
        "expected_word_present_rate": _fraction(expected_word_present),
        "mean_raw_predicted_columns": _mean(raw_columns),
        "mean_propagated_columns": _mean(propagated_columns),
        "mean_cue_burst_cells": _mean(
            float(row["burst_cell_count"]) for row in accumulator.cue_rows
        ),
        "no_prediction_rate": no_prediction_steps / (len(sentences) * 4),
        "early_stop_rate": early_stops / len(sentences),
        "runtime_seconds": time.perf_counter() - started,
        **accumulator.summary(model._kernel_peak_time),
    }
    for transition in range(1, 6):
        rows = [
            row
            for row in accumulator.cue_rows
            if int(row["step"]) == transition
        ]
        summary[f"cue_{transition}_{transition + 1}_mean_burst_cells"] = _mean(
            float(row["burst_cell_count"]) for row in rows
        )
        summary[f"cue_{transition}_{transition + 1}_mean_raw_columns"] = _mean(
            float(row["raw_predicted_column_count"]) for row in rows
        )
    return summary, transition_rows, expected_rows, prediction_rows


def _negative_delay_fraction(model: SequentialMemory) -> float:
    delays = [
        synapse.delay
        for column in model.columns
        for neuron in column.neurons
        for segment in neuron.segments
        for synapse in segment.synapses.values()
    ]
    return _fraction(delay < 0.0 for delay in delays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/CBTest/data/cbt_train.txt")
    parser.add_argument("--num-sentences", type=int, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--ranking-modes",
        nargs="+",
        choices=RANKING_MODES,
        default=RANKING_MODES,
    )
    parser.add_argument(
        "--delay-modes",
        nargs="+",
        choices=DELAY_MODES,
        default=("current-delay",),
    )
    args = parser.parse_args()

    sentences = read_cbt_sentences(Path(args.data), args.num_sentences, args.seed)
    output_dir = Path(args.output_dir)
    summaries: list[dict[str, object]] = []
    for delay_mode in args.delay_modes:
        print(f"training delay_mode={delay_mode}", flush=True)
        model = build_model(
            1.0,
            args.seed,
            synapse_delay_mode=delay_mode,
        )
        train_once(model, sentences)
        negative_fraction = _negative_delay_fraction(model)
        for ranking_mode in args.ranking_modes:
            print(f"evaluating ranking_mode={ranking_mode}", flush=True)
            summary, transitions, expected, predictions = evaluate_competition(
                model,
                sentences,
                ranking_mode=ranking_mode,
                delay_mode=delay_mode,
            )
            summary = {
                **kernel_diagnostic(1.0),
                "negative_synapse_delay_fraction": negative_fraction,
                **summary,
            }
            summaries.append(summary)
            stem = f"{delay_mode}_{ranking_mode}"
            write_csv(output_dir / f"transitions_{stem}.csv", transitions)
            write_csv(output_dir / f"expected_events_{stem}.csv", expected)
            write_csv(output_dir / f"prediction_events_{stem}.csv", predictions)
            write_csv(output_dir / "summary.csv", summaries)
            print(
                f"{stem}: distance={summary['mean_levenshtein']:.3f}, "
                f"raw={summary['mean_raw_predicted_columns']:.3f}, "
                f"burst={summary['mean_cue_burst_cells']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
