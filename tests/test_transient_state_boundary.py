from __future__ import annotations

import unittest

from experiments.fig8_sentence_memory import evaluate, train_sentence
from seqmem.encoding import SSTDDiscreteEncoder, SpikeEvent, SymbolCode
from seqmem.model import (
    MemoryParams,
    PredictionCandidate,
    Segment,
    SequentialMemory,
    Synapse,
    TransientStateSnapshot,
)


class TransientStateBoundaryTests(unittest.TestCase):
    def _model_with_candidate(
        self,
    ) -> tuple[SequentialMemory, Segment, Synapse, PredictionCandidate]:
        model = SequentialMemory(
            SSTDDiscreteEncoder(num_columns=8, k=2, seed=3),
            num_neurons_per_column=2,
            params=MemoryParams(continuous_dynamics=False),
            tie_break_seed=17,
        )
        synapse = Synapse(source=2, delay=0.5, weight=0.7, age=4)
        segment = Segment(synapses={2: synapse}, target_time=0.0)
        model.columns[0].neurons[1].segments.append(segment)
        candidate = PredictionCandidate(
            neuron_index=1,
            score=1.2,
            time=0.0,
            segment=segment,
        )
        model.previous_active_cells = {2: 0.0, 3: 0.5}
        model.previous_winners = {2: 0.0}
        model.last_prediction_candidates = {0: [candidate]}
        model.last_prediction_stats = {"candidate_segment_count": 1}
        model.last_observe_stats = {"scenario1": 1}
        model.last_symbol_ranking = [(2, 1, "A")]
        model.previous_predicted_sources = {2}
        model.previous_burst_only_sources = {3}
        return model, segment, synapse, candidate

    @staticmethod
    def _snapshot_signature(snapshot: TransientStateSnapshot) -> tuple[object, ...]:
        return (
            snapshot.previous_active_cells,
            snapshot.previous_winners,
            snapshot.last_prediction_candidates,
            snapshot.last_prediction_stats,
            snapshot.last_observe_stats,
            snapshot.last_symbol_ranking,
            snapshot.previous_predicted_sources,
            snapshot.previous_burst_only_sources,
            snapshot.decode_rng_state,
            snapshot.learning_rng_state,
        )

    @staticmethod
    def _persistent_identity_signature(model: SequentialMemory) -> tuple:
        return tuple(
            (
                id(segment),
                segment.active,
                segment.target_time,
                tuple(
                    (
                        source,
                        id(synapse),
                        synapse.delay,
                        synapse.weight,
                        synapse.age,
                    )
                    for source, synapse in segment.synapses.items()
                ),
            )
            for column in model.columns
            for neuron in column.neurons
            for segment in neuron.segments
        )

    def test_snapshot_is_exact_field_by_field_and_shallow(self) -> None:
        model, segment, _, candidate = self._model_with_candidate()

        snapshot = model.snapshot_transient_state()

        self.assertEqual(snapshot.previous_active_cells, model.previous_active_cells)
        self.assertEqual(snapshot.previous_winners, model.previous_winners)
        self.assertEqual(snapshot.last_prediction_stats, model.last_prediction_stats)
        self.assertEqual(snapshot.last_observe_stats, model.last_observe_stats)
        self.assertEqual(snapshot.last_symbol_ranking, model.last_symbol_ranking)
        self.assertEqual(
            snapshot.previous_predicted_sources,
            model.previous_predicted_sources,
        )
        self.assertEqual(
            snapshot.previous_burst_only_sources,
            model.previous_burst_only_sources,
        )
        self.assertEqual(snapshot.decode_rng_state, model._decode_rng.getstate())
        self.assertEqual(snapshot.learning_rng_state, model._learning_rng.getstate())
        self.assertIsNot(snapshot.previous_active_cells, model.previous_active_cells)
        self.assertIsNot(
            snapshot.last_prediction_candidates,
            model.last_prediction_candidates,
        )
        self.assertIsNot(
            snapshot.last_prediction_candidates[0],
            model.last_prediction_candidates[0],
        )
        self.assertIs(snapshot.last_prediction_candidates[0][0], candidate)
        self.assertIs(snapshot.last_prediction_candidates[0][0].segment, segment)

    def test_restore_is_exact_and_preserves_persistent_identity(self) -> None:
        model, segment, synapse, candidate = self._model_with_candidate()
        snapshot = model.snapshot_transient_state()

        model.reset_state()
        model._decode_rng.random()
        model._learning_rng.random()
        model.restore_transient_state(snapshot)

        self.assertEqual(
            self._snapshot_signature(model.snapshot_transient_state()),
            self._snapshot_signature(snapshot),
        )
        self.assertIs(model.columns[0].neurons[1].segments[0], segment)
        self.assertIs(segment.synapses[2], synapse)
        self.assertIs(model.last_prediction_candidates[0][0], candidate)
        self.assertIs(model.last_prediction_candidates[0][0].segment, segment)

    def test_restore_replays_future_rng_sequences(self) -> None:
        model, _, _, _ = self._model_with_candidate()
        snapshot = model.snapshot_transient_state()

        expected_decode = [model._decode_rng.random() for _ in range(8)]
        expected_learning = [model._learning_rng.random() for _ in range(8)]
        model.restore_transient_state(snapshot)

        self.assertEqual(
            [model._decode_rng.random() for _ in range(8)],
            expected_decode,
        )
        self.assertEqual(
            [model._learning_rng.random() for _ in range(8)],
            expected_learning,
        )

    def test_advance_prediction_then_restore_is_transient_exact(self) -> None:
        model, segment, synapse, _ = self._model_with_candidate()
        snapshot = model.snapshot_transient_state()
        code = SymbolCode(events=(SpikeEvent(column=0, time=0.0),))

        self.assertTrue(model.advance_prediction(code))
        self.assertNotEqual(model.previous_active_cells, snapshot.previous_active_cells)
        model.restore_transient_state(snapshot)

        self.assertEqual(
            self._snapshot_signature(model.snapshot_transient_state()),
            self._snapshot_signature(snapshot),
        )
        self.assertIs(model.columns[0].neurons[1].segments[0], segment)
        self.assertIs(segment.synapses[2], synapse)

    def test_reset_uses_fresh_empty_containers_without_resetting_rng(self) -> None:
        model, segment, synapse, _ = self._model_with_candidate()
        old_containers = (
            model.previous_active_cells,
            model.previous_winners,
            model.last_prediction_candidates,
            model.last_prediction_stats,
            model.last_observe_stats,
            model.last_symbol_ranking,
            model.previous_predicted_sources,
            model.previous_burst_only_sources,
        )
        decode_state = model._decode_rng.getstate()
        learning_state = model._learning_rng.getstate()

        model.reset_state()

        new_containers = (
            model.previous_active_cells,
            model.previous_winners,
            model.last_prediction_candidates,
            model.last_prediction_stats,
            model.last_observe_stats,
            model.last_symbol_ranking,
            model.previous_predicted_sources,
            model.previous_burst_only_sources,
        )
        self.assertTrue(all(not value for value in new_containers))
        for old, new in zip(old_containers, new_containers):
            self.assertIsNot(old, new)
        self.assertEqual(model._decode_rng.getstate(), decode_state)
        self.assertEqual(model._learning_rng.getstate(), learning_state)
        self.assertIs(model.columns[0].neurons[1].segments[0], segment)
        self.assertIs(segment.synapses[2], synapse)

    def test_no_learning_evaluation_restores_science_and_transient_state(self) -> None:
        model = SequentialMemory(
            SSTDDiscreteEncoder(num_columns=16, k=3, seed=5),
            num_neurons_per_column=3,
            params=MemoryParams(continuous_dynamics=False),
            tie_break_seed=19,
        )
        sentences = [["A", "B", "C"]]
        for _ in range(4):
            train_sentence(model, sentences[0])
        model.predict_code()
        before_transient = model.snapshot_transient_state()
        before_persistent = self._persistent_identity_signature(model)

        evaluate(
            model,
            sentences,
            prefix_length=1,
            eval_samples=1,
            seed=23,
            retrieval_mode="neural",
        )

        self.assertEqual(
            self._snapshot_signature(model.snapshot_transient_state()),
            self._snapshot_signature(before_transient),
        )
        self.assertEqual(
            self._persistent_identity_signature(model),
            before_persistent,
        )

    def test_pickle_class_module_paths_are_unchanged(self) -> None:
        expected = "seqmem.model"
        for value in (
            Synapse,
            Segment,
            MemoryParams,
            SequentialMemory,
            TransientStateSnapshot,
        ):
            self.assertEqual(value.__module__, expected)


if __name__ == "__main__":
    unittest.main()
