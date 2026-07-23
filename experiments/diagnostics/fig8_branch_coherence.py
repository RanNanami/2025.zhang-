from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.diagnostics.fig8_diagnostic_common import build_model  # noqa: E402
from experiments.fig8_sentence_memory import (  # noqa: E402
    levenshtein,
    read_cbt_sentences,
)
from seqmem.encoding import SymbolCode  # noqa: E402
from seqmem.model import PredictionCandidate, SequentialMemory  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def fraction(values) -> float:
    items = list(values)
    return sum(bool(item) for item in items) / len(items) if items else 0.0


def train_with_provenance(
    model: SequentialMemory,
    sentences: list[list[str]],
) -> set[frozenset[int]]:
    winner_sets: set[frozenset[int]] = set()
    for sentence_index, sentence in enumerate(sentences, start=1):
        model.reset_state()
        for word_index, word in enumerate(sentence):
            model.set_segment_provenance_context(sentence_index, word_index)
            model.predict_code()
            model.observe(word)
            winner_sets.add(frozenset(model.previous_winners))
    model.set_segment_provenance_context(None, None)
    return winner_sets


def matching_expected_candidates(
    model: SequentialMemory,
    expected: SymbolCode,
) -> tuple[list[PredictionCandidate], list[dict[str, object]]]:
    selected: list[PredictionCandidate] = []
    event_rows: list[dict[str, object]] = []
    for event in expected.events:
        candidates = model.last_prediction_candidates.get(event.column, [])
        matching = [
            candidate
            for candidate in candidates
            if abs(candidate.time - event.time) <= model.params.timing_tolerance
        ]
        winner = max(matching, key=lambda item: item.score) if matching else None
        if winner is not None:
            selected.append(winner)
        event_rows.append(
            {
                "expected_column": event.column,
                "expected_time": event.time,
                "candidate_count": len(candidates),
                "matching_candidate_count": len(matching),
                "distinct_neuron_count": len(
                    {candidate.neuron_index for candidate in matching}
                ),
                "selected_neuron": winner.neuron_index if winner else "",
                "selected_segment_id": winner.segment.diagnostic_id if winner else "",
            }
        )
    return selected, event_rows


def source_coherence(candidates: list[PredictionCandidate]) -> tuple[float, float]:
    similarities: list[float] = []
    sets = [set(candidate.segment.synapses) for candidate in candidates]
    for index, left in enumerate(sets):
        for right in sets[index + 1 :]:
            union = left | right
            similarities.append(len(left & right) / len(union) if union else 1.0)
    return (
        mean(similarities),
        min(similarities) if similarities else 0.0,
    )


def provenance_metrics(
    model: SequentialMemory,
    candidates: list[PredictionCandidate],
    winner_sets: set[frozenset[int]],
) -> dict[str, object]:
    origins = [
        (
            candidate.segment.creation_sentence_index,
            candidate.segment.creation_transition_index,
        )
        for candidate in candidates
        if candidate.segment.creation_sentence_index is not None
    ]
    counts = Counter(origins)
    maximum = max(counts.values(), default=0)
    probabilities = [count / len(origins) for count in counts.values()] if origins else []
    entropy = -sum(value * math.log(value) for value in probabilities)
    normalized_entropy = (
        entropy / math.log(len(counts)) if len(counts) > 1 else 0.0
    )
    average_jaccard, minimum_jaccard = source_coherence(candidates)
    selected_cells = frozenset(
        model._cell_id(
            candidate.segment.creation_target_column
            if candidate.segment.creation_target_column is not None
            else 0,
            candidate.neuron_index,
        )
        for candidate in candidates
        if candidate.segment.creation_target_column is not None
    )
    return {
        "selected_event_count": len(candidates),
        "distinct_creation_transitions": len(counts),
        "maximum_same_origin_events": maximum,
        "provenance_coherent_fraction": maximum / model.encoder.k,
        "branch_entropy": normalized_entropy,
        "source_coherence": average_jaccard,
        "minimum_source_jaccard": minimum_jaccard,
        "matches_historical_winner_set": selected_cells in winner_sets,
    }


def burst_rows(
    model: SequentialMemory,
    expected: SymbolCode,
) -> list[dict[str, object]]:
    timing_tolerance = model.params.timing_tolerance
    rows: list[dict[str, object]] = []
    for column, candidates in model.last_prediction_candidates.items():
        for candidate in candidates:
            support = model.candidate_burst_psp_trace(candidate)
            is_expected_event = any(
                column == event.column
                and abs(candidate.time - event.time) <= timing_tolerance
                for event in expected.events
            )
            rows.append(
                {
                    "column": column,
                    "time": candidate.time,
                    "segment_id": candidate.segment.diagnostic_id,
                    "is_expected_event": is_expected_event,
                    "total_psp": support.total_psp,
                    "predicted_source_psp": support.predicted_source_psp,
                    "burst_only_psp": support.burst_only_psp,
                    "unlabelled_psp": support.unlabelled_psp,
                    "predicted_without_burst_crosses": (
                        support.predicted_without_burst_crosses
                    ),
                    "burst_only_crosses": support.burst_only_crosses,
                    "classification": support.classification,
                    "contributor_count": sum(
                        item.psp_contribution > 0.0
                        for item in candidate.crossing_synapse_contributions
                    ),
                }
            )
    return rows


def selected_candidates(
    model: SequentialMemory,
    code: SymbolCode | None,
) -> list[PredictionCandidate]:
    if code is None:
        return []
    selected: list[PredictionCandidate] = []
    for event in code.events:
        selected.extend(
            candidate
            for candidate in model.last_prediction_candidates.get(event.column, [])
            if round(candidate.time, 12) == round(event.time, 12)
        )
    return selected


def select_code(
    model: SequentialMemory,
    raw: SymbolCode,
    selection: str,
    beam_width: int,
    lambda_coherence: float,
    lambda_support: float,
) -> SymbolCode | None:
    if selection == "raw":
        return raw
    if selection == "local-top1":
        return model.select_prediction_events(raw)
    if selection == "coherent-beam":
        return model.select_coherent_prediction_events(
            raw,
            beam_width=beam_width,
            lambda_coherence=lambda_coherence,
            lambda_support=lambda_support,
        )
    raise ValueError(f"unsupported neural selection: {selection}")


def evaluate(
    model: SequentialMemory,
    sentences: list[list[str]],
    winner_sets: set[frozenset[int]],
    *,
    selection: str,
    beam_width: int = 4,
    lambda_coherence: float = 0.5,
    lambda_support: float = 0.0,
    collect_details: bool = True,
    burst_analysis: str = "all",
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    snapshot = model.snapshot_transient_state()
    step_rows: list[dict[str, object]] = []
    all_burst_rows: list[dict[str, object]] = []
    false_burst_class_counts: Counter[str] = Counter()
    false_burst_count = 0
    distances: list[int] = []
    raw_columns: list[int] = []
    propagated_columns: list[int] = []
    presence: list[bool] = []
    early_stops = 0
    try:
        for sentence_index, sentence in enumerate(sentences, start=1):
            model.reset_state()
            model.predict_code()
            model.observe(sentence[0], learn=False)
            for cue_index in range(1, 6):
                expected = model.encoder.encode(sentence[cue_index])
                raw = model.predict_code()
                candidates, events = matching_expected_candidates(model, expected)
                metrics = provenance_metrics(model, candidates, winner_sets)
                candidate_bursts = (
                    burst_rows(model, expected) if burst_analysis == "all" else []
                )
                predicted_cells = model.prediction_active_cells(raw) if raw else {}
                model.observe(sentence[cue_index], learn=False)
                burst_cells = set(model.previous_active_cells) - set(predicted_cells)
                if collect_details:
                    for row in events:
                        row.update(
                            {
                                "diagnostic_label": "nonpaper diagnostic",
                                "sentence_index": sentence_index,
                                "phase": "cue",
                                "step": cue_index,
                            }
                        )
                    for row in candidate_bursts:
                        row.update(
                            {
                                "diagnostic_label": "nonpaper diagnostic",
                                "sentence_index": sentence_index,
                                "phase": "cue",
                                "step": cue_index,
                            }
                        )
                    all_burst_rows.extend(candidate_bursts)
                step_rows.append(
                    {
                        "diagnostic_label": "nonpaper diagnostic",
                        "sentence_index": sentence_index,
                        "phase": "cue",
                        "step": cue_index,
                        "selection": selection,
                        **metrics,
                        "raw_predicted_columns": (
                            len({event.column for event in raw.events}) if raw else 0
                        ),
                        "burst_columns": len(
                            {
                                cell // len(model.columns[0].neurons)
                                for cell in burst_cells
                            }
                        ),
                        "burst_cells": len(burst_cells),
                        "active_cells_after_observe": len(model.previous_active_cells),
                        "winners_after_observe": len(model.previous_winners),
                    }
                )

            recalled: list[str] = []
            for suffix_step in range(1, 5):
                expected_word = sentence[5 + suffix_step]
                expected = model.encoder.encode(expected_word)
                raw = model.predict_code()
                if raw is None:
                    early_stops += 1
                    presence.extend(False for _ in range(5 - suffix_step))
                    break
                propagated = select_code(
                    model,
                    raw,
                    selection,
                    beam_width,
                    lambda_coherence,
                    lambda_support,
                )
                if propagated is None:
                    early_stops += 1
                    presence.extend(False for _ in range(5 - suffix_step))
                    break
                analysis_code = (
                    model.select_prediction_events(raw)
                    if selection == "raw"
                    else propagated
                )
                candidates = selected_candidates(model, analysis_code)
                metrics = provenance_metrics(model, candidates, winner_sets)
                candidate_bursts = (
                    burst_rows(model, expected) if burst_analysis == "all" else []
                )
                false_candidates = [
                    row for row in candidate_bursts if not row["is_expected_event"]
                ]
                false_burst_count += len(false_candidates)
                false_burst_class_counts.update(
                    str(row["classification"]) for row in false_candidates
                )
                decoded = model.decode_symbol_from_prediction(propagated)
                ranking = model.last_symbol_ranking.copy()
                is_correct = decoded == expected_word
                presence.append(any(item[2] == expected_word for item in ranking))
                recalled.append(decoded or "")
                raw_columns.append(len({event.column for event in raw.events}))
                propagated_columns.append(
                    len({event.column for event in propagated.events})
                )
                if collect_details:
                    for row in candidate_bursts:
                        row.update(
                            {
                                "diagnostic_label": "nonpaper diagnostic",
                                "sentence_index": sentence_index,
                                "phase": "suffix",
                                "step": suffix_step,
                                "decoded_correct": is_correct,
                            }
                        )
                    all_burst_rows.extend(candidate_bursts)
                step_rows.append(
                    {
                        "diagnostic_label": "nonpaper diagnostic",
                        "sentence_index": sentence_index,
                        "phase": "suffix",
                        "step": suffix_step,
                        "selection": selection,
                        "beam_width": beam_width,
                        "lambda_coherence": lambda_coherence,
                        "lambda_support": lambda_support,
                        "decoded_correct": is_correct,
                        "expected_word_present": presence[-1],
                        **metrics,
                        "raw_predicted_columns": raw_columns[-1],
                        "propagated_columns": propagated_columns[-1],
                    }
                )
                if decoded is None or not model.advance_prediction(propagated):
                    early_stops += 1
                    presence.extend(False for _ in range(4 - suffix_step))
                    break
            distances.append(levenshtein(recalled, sentence[6:]))
    finally:
        model.restore_transient_state(snapshot)

    suffix_rows = [row for row in step_rows if row["phase"] == "suffix"]
    correct_rows = [row for row in suffix_rows if row.get("decoded_correct")]
    wrong_rows = [row for row in suffix_rows if not row.get("decoded_correct")]
    summary = {
        "diagnostic_label": "nonpaper diagnostic",
        "sentences": len(sentences),
        "selection": selection,
        "beam_width": beam_width,
        "lambda_coherence": lambda_coherence,
        "lambda_support": lambda_support,
        "burst_analysis": burst_analysis,
        "mean_levenshtein": mean(distances),
        "expected_word_presence": fraction(presence),
        "early_stop_rate": early_stops / len(sentences),
        "mean_raw_columns": mean(raw_columns),
        "mean_propagated_columns": mean(propagated_columns),
        "mean_cue_burst_cells": mean(
            float(row["burst_cells"]) for row in step_rows if row["phase"] == "cue"
        ),
        "correct_provenance_coherent_fraction": mean(
            float(row["provenance_coherent_fraction"]) for row in correct_rows
        ),
        "wrong_provenance_coherent_fraction": mean(
            float(row["provenance_coherent_fraction"]) for row in wrong_rows
        ),
        "correct_source_coherence": mean(
            float(row["source_coherence"]) for row in correct_rows
        ),
        "wrong_source_coherence": mean(
            float(row["source_coherence"]) for row in wrong_rows
        ),
        "correct_branch_entropy": mean(
            float(row["branch_entropy"]) for row in correct_rows
        ),
        "wrong_branch_entropy": mean(
            float(row["branch_entropy"]) for row in wrong_rows
        ),
        "historical_winner_set_fraction": fraction(
            bool(row["matches_historical_winner_set"]) for row in suffix_rows
        ),
        "false_prediction_predicted_supported_fraction": (
            false_burst_class_counts["predicted-supported"] / false_burst_count
            if false_burst_count else 0.0
        ),
        "false_prediction_burst_assisted_fraction": (
            false_burst_class_counts["burst-assisted"] / false_burst_count
            if false_burst_count else 0.0
        ),
        "false_prediction_burst_only_fraction": (
            false_burst_class_counts["burst-only"] / false_burst_count
            if false_burst_count else 0.0
        ),
    }
    return summary, step_rows, all_burst_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/CBTest/data/cbt_train.txt")
    parser.add_argument("--num-sentences", type=int, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--neural-selection",
        choices=("raw", "local-top1", "coherent-beam"),
        nargs="+",
        default=("raw",),
    )
    parser.add_argument("--beam-widths", nargs="+", type=int, default=(4,))
    parser.add_argument(
        "--lambda-coherence", nargs="+", type=float, default=(0.5,)
    )
    parser.add_argument("--lambda-support", type=float, default=0.0)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="skip candidate-level burst CSV while retaining aggregate metrics",
    )
    parser.add_argument(
        "--burst-analysis",
        choices=("all", "none"),
        default="all",
        help="skip all-candidate PSP burst decomposition for large selection grids",
    )
    args = parser.parse_args()

    sentences = read_cbt_sentences(Path(args.data), args.num_sentences, args.seed)
    model = build_model(
        1.0,
        args.seed,
        capture_prediction_contributions=True,
        capture_branch_diagnostics=True,
    )
    winner_sets = train_with_provenance(model, sentences)
    output_dir = Path(args.output_dir)
    summaries: list[dict[str, object]] = []
    for selection in args.neural_selection:
        settings = (
            [
                (width, coherence)
                for width in args.beam_widths
                for coherence in args.lambda_coherence
            ]
            if selection == "coherent-beam"
            else [(1, 0.0)]
        )
        for beam_width, coherence in settings:
            summary, steps, bursts = evaluate(
                model,
                sentences,
                winner_sets,
                selection=selection,
                beam_width=beam_width,
                lambda_coherence=coherence,
                lambda_support=args.lambda_support,
                collect_details=not args.summary_only,
                burst_analysis=args.burst_analysis,
            )
            summaries.append(summary)
            stem = (
                f"{selection}_b{beam_width}_c{coherence:g}"
                f"_s{args.lambda_support:g}"
            )
            write_csv(output_dir / f"steps_{stem}.csv", steps)
            if not args.summary_only:
                write_csv(output_dir / f"burst_candidates_{stem}.csv", bursts)
            write_csv(output_dir / "summary.csv", summaries)
            print(
                f"{stem}: distance={summary['mean_levenshtein']:.3f}, "
                f"presence={summary['expected_word_presence']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
