"""Capture stable learning-mutation sequences for Phase 07 regression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from experiments.fig8_sentence_memory import read_cbt_sentences
from experiments.fig9 import learn_actual_code, read_records, record_values
from experiments.fig9_strict_reproduction import (
    Fig9StrictConfig,
    build_fig9_encoder,
    build_strict_model,
)
from scripts.capture_phase06_branch_sequence import stable_sha256
from seqmem.encoding import SSTDDiscreteEncoder
from seqmem.model import MemoryParams, SequentialMemory


ROOT = Path(__file__).resolve().parents[1]


def segment_key_from_identity(model: SequentialMemory, identity: object) -> str:
    """Translate a runtime identity to a structural key before persisting it."""

    for column_id, column in enumerate(model.columns):
        for neuron_index, neuron in enumerate(column.neurons):
            for segment_index, segment in enumerate(neuron.segments):
                if id(segment) == identity:
                    return f"{column_id}:{neuron_index}:{segment_index}"
    return "detached"


def persistent_snapshot(model: SequentialMemory) -> dict[str, dict[str, object]]:
    snapshot: dict[str, dict[str, object]] = {}
    for column_id, column in enumerate(model.columns):
        for neuron_index, neuron in enumerate(column.neurons):
            for segment_index, segment in enumerate(neuron.segments):
                key = f"{column_id}:{neuron_index}:{segment_index}"
                snapshot[key] = {
                    "active": segment.active,
                    "target_time": segment.target_time,
                    "synapses": {
                        str(source): {
                            "weight": synapse.weight,
                            "age": synapse.age,
                            "delay": synapse.delay,
                        }
                        for source, synapse in segment.synapses.items()
                    },
                }
    return snapshot


def snapshot_diff(
    before: dict[str, dict[str, object]],
    after: dict[str, dict[str, object]],
) -> dict[str, object]:
    before_keys = set(before)
    after_keys = set(after)
    created_segments = [
        [key, after[key]] for key in sorted(after_keys - before_keys)
    ]
    removed_segments = [
        [key, before[key]] for key in sorted(before_keys - after_keys)
    ]
    weight_age_updates: list[list[object]] = []
    new_synapses: list[list[object]] = []
    removed_synapses: list[list[object]] = []
    for key in sorted(before_keys & after_keys):
        before_synapses = before[key]["synapses"]
        after_synapses = after[key]["synapses"]
        assert isinstance(before_synapses, dict)
        assert isinstance(after_synapses, dict)
        before_sources = set(before_synapses)
        after_sources = set(after_synapses)
        for source in sorted(after_sources - before_sources, key=int):
            new_synapses.append([key, source, after_synapses[source]])
        for source in sorted(before_sources - after_sources, key=int):
            removed_synapses.append([key, source, before_synapses[source]])
        for source in sorted(before_sources & after_sources, key=int):
            old = before_synapses[source]
            new = after_synapses[source]
            if old != new:
                weight_age_updates.append([key, source, old, new])
    return {
        "created_segments": created_segments,
        "removed_segments": removed_segments,
        "new_synapses": new_synapses,
        "removed_synapses": removed_synapses,
        "weight_age_updates": weight_age_updates,
    }


class MutationCapture:
    def __init__(self, model: SequentialMemory, fixture: str) -> None:
        self.model = model
        self.fixture = fixture
        self.observation_index = -1
        self.reinforcements: list[dict[str, object]] = []
        self.creations: list[dict[str, object]] = []
        self.punishment_targets: list[dict[str, object]] = []
        self.punishment_mutations: list[dict[str, object]] = []
        self.prune_calls: list[dict[str, object]] = []
        self.observation_mutations: list[dict[str, object]] = []
        self._install()

    def _install(self) -> None:
        self.model.params.capture_intralayer_parity_diagnostics = True

        def reinforcement(trace: object) -> None:
            identity = getattr(trace, "segment_identity")
            self.reinforcements.append(
                {
                    "fixture": self.fixture,
                    "observation_index": self.observation_index,
                    "scenario": getattr(trace, "scenario"),
                    "segment_key": segment_key_from_identity(
                        self.model, identity
                    ),
                    "weights_before": getattr(trace, "weights_before"),
                    "weights_after": getattr(trace, "weights_after"),
                    "contributed_source_ids": getattr(
                        trace, "contributed_source_ids"
                    ),
                    "strengthened_source_ids": getattr(
                        trace, "strengthened_source_ids"
                    ),
                    "weakened_source_ids": getattr(
                        trace, "weakened_source_ids"
                    ),
                    "added_source_ids": getattr(trace, "added_source_ids"),
                    "other_segment_weakened_count": getattr(
                        trace, "other_segment_weakened_count"
                    ),
                    "other_segment_aged_count": getattr(
                        trace, "other_segment_aged_count"
                    ),
                }
            )

        def intralayer(payload: dict[str, object]) -> None:
            phase = payload.get("phase")
            if phase not in {"segment_created", "wrong_prediction_punishment"}:
                return
            identity = (
                payload.get("segment_identity")
                if phase == "segment_created"
                else payload.get("predicted_segment_identity")
            )
            row = {
                "fixture": self.fixture,
                "observation_index": self.observation_index,
                "phase": phase,
                "segment_key": segment_key_from_identity(self.model, identity),
                "target_column": payload.get("target_column"),
            }
            if phase == "segment_created":
                row.update(
                    {
                        "target_neuron": payload.get("target_neuron"),
                        "creation_scenario": payload.get("creation_scenario"),
                        "source_ids": payload.get("source_ids"),
                    }
                )
                self.creations.append(row)
            else:
                row.update(
                    {
                        "predicted_neuron": payload.get("predicted_neuron"),
                        "predicted_time": payload.get("predicted_time"),
                        "actual_time": payload.get("actual_time"),
                        "contributed_source_ids": payload.get(
                            "contributed_source_ids"
                        ),
                        "punishment_reason": payload.get("punishment_reason"),
                    }
                )
                self.punishment_targets.append(row)

        self.model.reinforcement_trace_callback = reinforcement
        self.model.intralayer_parity_callback = intralayer

        original_prune = self.model._prune_neuron

        def wrapped_prune(
            neuron: object,
            *,
            target_column: int | None = None,
            neuron_index: int | None = None,
            candidate_segment: object | None = None,
        ) -> None:
            segment_key = segment_key_from_identity(
                self.model,
                id(candidate_segment) if candidate_segment is not None else None,
            )
            self.prune_calls.append(
                {
                    "fixture": self.fixture,
                    "observation_index": self.observation_index,
                    "phase": "PRUNE_NEURON_ENTER",
                    "target_column": target_column,
                    "neuron_index": neuron_index,
                    "candidate_segment_key": segment_key,
                    "segment_count": len(getattr(neuron, "segments")),
                }
            )
            original_prune(
                neuron,  # type: ignore[arg-type]
                target_column=target_column,
                neuron_index=neuron_index,
                candidate_segment=candidate_segment,  # type: ignore[arg-type]
            )
            self.prune_calls.append(
                {
                    "fixture": self.fixture,
                    "observation_index": self.observation_index,
                    "phase": "PRUNE_NEURON_EXIT",
                    "target_column": target_column,
                    "neuron_index": neuron_index,
                    "candidate_segment_key": segment_key,
                    "segment_count": len(getattr(neuron, "segments")),
                }
            )

        self.model._prune_neuron = wrapped_prune  # type: ignore[method-assign]

        original = self.model._punish_wrong_predictions

        def wrapped_punishment(
            active_events: dict[int, float],
            confirmed_candidate_ids: set[int] | None = None,
        ) -> None:
            before = persistent_snapshot(self.model)
            original(
                active_events,
                confirmed_candidate_ids=confirmed_candidate_ids,
            )
            after = persistent_snapshot(self.model)
            self.punishment_mutations.append(
                {
                    "fixture": self.fixture,
                    "observation_index": self.observation_index,
                    **snapshot_diff(before, after),
                }
            )

        self.model._punish_wrong_predictions = wrapped_punishment  # type: ignore[method-assign]

    def observe(self, index: int, operation: Callable[[], object]) -> None:
        self.observation_index = index
        before = persistent_snapshot(self.model)
        operation()
        after = persistent_snapshot(self.model)
        self.observation_mutations.append(
            {
                "fixture": self.fixture,
                "observation_index": index,
                **snapshot_diff(before, after),
            }
        )

    def summary(self) -> dict[str, object]:
        new_synapse_sequence = [
            {
                "fixture": row["fixture"],
                "observation_index": row["observation_index"],
                "new_synapses": row["new_synapses"],
            }
            for row in self.observation_mutations
        ]
        prune_sequence = {
            "calls": self.prune_calls,
            "removed": [
                {
                    "observation_index": row["observation_index"],
                    "removed_segments": row["removed_segments"],
                    "removed_synapses": row["removed_synapses"],
                }
                for row in self.observation_mutations
            ],
        }
        return {
            "fixture": self.fixture,
            "reinforcement_count": len(self.reinforcements),
            "creation_count": len(self.creations),
            "punishment_target_count": len(self.punishment_targets),
            "prune_call_count": len(self.prune_calls),
            "reinforced_segment_sequence_sha256": stable_sha256(
                [row["segment_key"] for row in self.reinforcements]
            ),
            "weight_update_sequence_sha256": stable_sha256(
                self.reinforcements
            ),
            "new_segment_sequence_sha256": stable_sha256(self.creations),
            "new_synapse_sequence_sha256": stable_sha256(
                new_synapse_sequence
            ),
            "punishment_update_sequence_sha256": stable_sha256(
                {
                    "targets": self.punishment_targets,
                    "mutations": self.punishment_mutations,
                }
            ),
            "forgetting_prune_sequence_sha256": stable_sha256(prune_sequence),
            "reinforcements": self.reinforcements,
            "creations": self.creations,
            "punishment_targets": self.punishment_targets,
            "punishment_mutations": self.punishment_mutations,
            "prune_calls": self.prune_calls,
            "observation_mutations": self.observation_mutations,
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
    capture = MutationCapture(model, fixture)
    observation_index = 0
    for sentence in sentences:
        model.reset_state()
        for word in sentence:
            model.predict_code()
            code = model.encoder.encode(word)
            capture.observe(
                observation_index,
                lambda code=code: model.observe_code(code),
            )
            observation_index += 1
    return capture.summary()


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
    capture = MutationCapture(model, fixture)
    records = read_records(ROOT / "data/paper_nyc_taxi.csv", size)
    for observation_index, record in enumerate(records):
        code = encoder.encode(record_values(record))
        capture.observe(
            observation_index,
            lambda code=code: learn_actual_code(model, code),
        )
    return capture.summary()


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
        "classification": "PHASE07_MUTATION_SEQUENCE_REGRESSION",
        "fixtures": fixtures,
    }
    path = output / "mutation_sequences.json"
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(path)


if __name__ == "__main__":
    main()
