"""Capture stable prediction-call sequences for Phase 08 regression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.fig8_sentence_memory import read_cbt_sentences
from experiments.fig9 import read_records, record_values
from experiments.fig9_strict_reproduction import (
    Fig9StrictConfig,
    build_fig9_encoder,
    build_strict_model,
)
from scripts.capture_phase06_branch_sequence import stable_sha256
from seqmem.encoding import SSTDDiscreteEncoder, SymbolCode
from seqmem.model import (
    MemoryParams,
    PredictionCandidate,
    PredictionTrace,
    Segment,
    SequentialMemory,
)


ROOT = Path(__file__).resolve().parents[1]


def segment_key(model: SequentialMemory, target: Segment) -> str:
    for column_id, column in enumerate(model.columns):
        for neuron_index, neuron in enumerate(column.neurons):
            for segment_index, segment in enumerate(neuron.segments):
                if segment is target:
                    return f"{column_id}:{neuron_index}:{segment_index}"
    return "detached"


def candidate_row(
    model: SequentialMemory,
    column: int,
    candidate_index: int,
    candidate: PredictionCandidate,
) -> dict[str, object]:
    return {
        "column": column,
        "candidate_index": candidate_index,
        "neuron": candidate.neuron_index,
        "segment_key": segment_key(model, candidate.segment),
        "score": candidate.score,
        "time": candidate.time,
        "dendritic_crossing_time": candidate.dendritic_crossing_time,
        "predicted_soma_firing_time": candidate.predicted_soma_firing_time,
        "peak_dendritic_potential": candidate.peak_dendritic_potential,
        "threshold_margin": candidate.threshold_margin,
        "crossing_synapse_contributions": [
            {
                "source_cell_id": item.source_cell_id,
                "source_time": item.source_time,
                "arrival_time": item.arrival_time,
                "weight": item.weight,
                "psp_contribution": item.psp_contribution,
            }
            for item in candidate.crossing_synapse_contributions
        ],
    }


def capture_prediction_call(
    model: SequentialMemory,
    fixture: str,
    call_index: int,
) -> tuple[SymbolCode | None, dict[str, object]]:
    source_context = list(model._active_sources().items())
    trace = PredictionTrace()
    predicted = model.predict_code(trace=trace)
    candidates = [
        candidate_row(model, column, candidate_index, candidate)
        for column, column_candidates in model.last_prediction_candidates.items()
        for candidate_index, candidate in enumerate(column_candidates)
    ]
    emitted_events = (
        [[event.column, event.time] for event in predicted.events]
        if predicted is not None
        else []
    )
    emitted_keys = {(column, event_time) for column, event_time in emitted_events}
    selected_candidate_keys = [
        [row["column"], row["candidate_index"], row["segment_key"]]
        for row in candidates
        if (row["column"], row["time"]) in emitted_keys
    ]
    evaluated_segments = [
        {
            "target_column": item.target_column,
            "target_neuron": item.target_neuron,
            "segment_key": item.segment_id,
            "sum_active_weights": item.sum_active_weights,
            "peak_dendritic_potential": item.peak_dendritic_potential,
            "threshold_margin": item.threshold_margin,
            "first_threshold_crossing_time": (
                item.first_threshold_crossing_time
            ),
            "predicted_soma_firing_time": item.predicted_soma_firing_time,
            "crossed_threshold": item.crossed_threshold,
            "entered_raw_prediction": item.entered_raw_prediction,
            "inhibited_intracolumn": item.inhibited_intracolumn,
        }
        for item in trace.segments
        if item.crossed_threshold or item.entered_raw_prediction
    ]
    return predicted, {
        "fixture": fixture,
        "call_index": call_index,
        "source_context": source_context,
        "candidate_segment_count": model.last_prediction_stats.get(
            "candidate_segment_count", 0
        ),
        "accepted_candidate_count": len(candidates),
        "candidates": candidates,
        "evaluated_crossing_segments": evaluated_segments,
        "emitted_events": emitted_events,
        "selected_candidate_keys": selected_candidate_keys,
        "last_prediction_stats": dict(model.last_prediction_stats),
    }


def summarize(fixture: str, rows: list[dict[str, object]]) -> dict[str, object]:
    float_sequence = [
        [
            candidate["score"],
            candidate["time"],
            candidate["dendritic_crossing_time"],
            candidate["predicted_soma_firing_time"],
            candidate["peak_dendritic_potential"],
            candidate["threshold_margin"],
            candidate["crossing_synapse_contributions"],
        ]
        for row in rows
        for candidate in row["candidates"]  # type: ignore[union-attr]
    ]
    return {
        "fixture": fixture,
        "prediction_call_count": len(rows),
        "accepted_candidate_count": sum(
            int(row["accepted_candidate_count"]) for row in rows
        ),
        "emitted_event_count": sum(
            len(row["emitted_events"]) for row in rows  # type: ignore[arg-type]
        ),
        "source_context_sequence_sha256": stable_sha256(
            [row["source_context"] for row in rows]
        ),
        "candidate_structure_sequence_sha256": stable_sha256(
            [row["candidates"] for row in rows]
        ),
        "candidate_float_sequence_sha256": stable_sha256(float_sequence),
        "emitted_event_sequence_sha256": stable_sha256(
            [row["emitted_events"] for row in rows]
        ),
        "selected_candidate_sequence_sha256": stable_sha256(
            [row["selected_candidate_keys"] for row in rows]
        ),
        "prediction_stats_sequence_sha256": stable_sha256(
            [row["last_prediction_stats"] for row in rows]
        ),
        "rows": rows,
    }


def capture_fig8(size: int) -> dict[str, object]:
    fixture = f"fig8_{size}"
    seed = 11
    sentences = read_cbt_sentences(
        ROOT / "data/CBTest/data/cbt_train.txt", size, seed
    )
    model = SequentialMemory(
        SSTDDiscreteEncoder(num_columns=100, k=10, seed=seed),
        num_neurons_per_column=10,
        params=MemoryParams(l_match=3, forgetting_threshold=500.0),
        tie_break_seed=seed,
    )
    rows: list[dict[str, object]] = []
    call_index = 0
    for sentence in sentences:
        model.reset_state()
        for word in sentence:
            _predicted, row = capture_prediction_call(
                model, fixture, call_index
            )
            rows.append(row)
            model.observe(word)
            call_index += 1
    return summarize(fixture, rows)


def capture_fig9(size: int) -> dict[str, object]:
    fixture = f"fig9_{size}"
    config = Fig9StrictConfig(
        warmup={20: 5, 50: 10}[size],
        seed=0,
        l_match=4,
        continuous_prediction_impl="reference",
    )
    encoder = build_fig9_encoder(config)
    model = build_strict_model(encoder, config)
    rows: list[dict[str, object]] = []
    records = read_records(ROOT / "data/paper_nyc_taxi.csv", size)
    for call_index, record in enumerate(records):
        _predicted, row = capture_prediction_call(model, fixture, call_index)
        rows.append(row)
        model.observe_code(
            encoder.encode(record_values(record)),
            learn=True,
        )
    return summarize(fixture, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fixtures = [
        capture_fig8(20),
        capture_fig8(50),
        capture_fig9(20),
        capture_fig9(50),
    ]
    result: dict[str, Any] = {
        "classification": "PHASE08_PREDICTION_SEQUENCE_REGRESSION",
        "fixtures": fixtures,
    }
    path = output / "prediction_sequences.json"
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(path)


if __name__ == "__main__":
    main()
