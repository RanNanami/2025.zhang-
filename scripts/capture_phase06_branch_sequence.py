"""Capture stable observation-branch sequences for Phase 06 regression."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.fig8_sentence_memory import read_cbt_sentences
from experiments.fig9 import learn_actual_code, read_records, record_values
from experiments.fig9_strict_reproduction import (
    Fig9StrictConfig,
    build_fig9_encoder,
    build_strict_model,
)
from seqmem.encoding import SSTDDiscreteEncoder
from seqmem.model import MemoryParams, ObservationTrace, Segment, SequentialMemory


ROOT = Path(__file__).resolve().parents[1]


def stable_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def segment_key(model: SequentialMemory, segment: Segment | None) -> str:
    if segment is None:
        return ""
    for column_id, column in enumerate(model.columns):
        for neuron_index, neuron in enumerate(column.neurons):
            for segment_index, current in enumerate(neuron.segments):
                if current is segment:
                    return f"{column_id}:{neuron_index}:{segment_index}"
    return "detached"


def install_punishment_capture(
    model: SequentialMemory,
    fixture: str,
    punishment_rows: list[dict[str, object]],
) -> None:
    model.params.capture_intralayer_parity_diagnostics = True

    def capture(payload: dict[str, object]) -> None:
        if payload.get("phase") != "wrong_prediction_punishment":
            return
        segment_identity = payload.get("predicted_segment_identity")
        selected: Segment | None = None
        for column in model.columns:
            for neuron in column.neurons:
                for segment in neuron.segments:
                    if id(segment) == segment_identity:
                        selected = segment
                        break
        punishment_rows.append(
            {
                "fixture": fixture,
                "target_column": payload.get("target_column"),
                "predicted_neuron": payload.get("predicted_neuron"),
                "segment_key": segment_key(model, selected),
                "predicted_time": payload.get("predicted_time"),
                "actual_time": payload.get("actual_time"),
                "punishment_reason": payload.get("punishment_reason"),
            }
        )

    model.intralayer_parity_callback = capture


def append_trace(
    model: SequentialMemory,
    fixture: str,
    observation_index: int,
    trace: ObservationTrace,
    rows: list[dict[str, object]],
) -> None:
    for event_index, event in enumerate(trace.events):
        rows.append(
            {
                "fixture": fixture,
                "observation_index": observation_index,
                "event_index": event_index,
                "target_column": event.target_column,
                "target_time": event.target_time,
                "current_internal_branch": event.scenario,
                "winner_neuron": event.winner_neuron,
                "selected_segment_key": segment_key(model, event.selected_segment),
                "reinforced_segment_key": segment_key(
                    model, event.reinforced_segment
                ),
                "created_segment_key": segment_key(model, event.created_segment),
                "scenario_assignment_reason": event.scenario_assignment_reason,
                "was_predicted": event.was_predicted,
            }
        )


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
    punishments: list[dict[str, object]] = []
    install_punishment_capture(model, fixture, punishments)
    observation_index = 0
    for sentence in sentences:
        model.reset_state()
        for word in sentence:
            trace = ObservationTrace()
            model.predict_code()
            model.observe_code(model.encoder.encode(word), observation_trace=trace)
            append_trace(model, fixture, observation_index, trace, rows)
            observation_index += 1
    return summarize(fixture, rows, punishments)


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
    punishments: list[dict[str, object]] = []
    install_punishment_capture(model, fixture, punishments)
    records = read_records(ROOT / "data/paper_nyc_taxi.csv", size)
    for observation_index, record in enumerate(records):
        trace = ObservationTrace()
        learn_actual_code(
            model,
            encoder.encode(record_values(record)),
            observation_trace=trace,
        )
        append_trace(model, fixture, observation_index, trace, rows)
    return summarize(fixture, rows, punishments)


def summarize(
    fixture: str,
    rows: list[dict[str, object]],
    punishments: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "fixture": fixture,
        "event_count": len(rows),
        "punishment_count": len(punishments),
        "branch_sequence_sha256": stable_sha256(
            [row["current_internal_branch"] for row in rows]
        ),
        "winner_sequence_sha256": stable_sha256(
            [row["winner_neuron"] for row in rows]
        ),
        "selected_segment_sequence_sha256": stable_sha256(
            [row["selected_segment_key"] for row in rows]
        ),
        "created_segment_sequence_sha256": stable_sha256(
            [row["created_segment_key"] for row in rows]
        ),
        "punishment_sequence_sha256": stable_sha256(punishments),
        "rows": rows,
        "punishments": punishments,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    captures = [
        capture_fig8(20),
        capture_fig8(50),
        capture_fig9(20),
        capture_fig9(50),
    ]
    result: dict[str, Any] = {
        "classification": "PHASE06_BRANCH_SEQUENCE_REGRESSION",
        "fixtures": captures,
    }
    path = output / "branch_sequences.json"
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(path)


if __name__ == "__main__":
    main()
