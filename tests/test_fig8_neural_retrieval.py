from __future__ import annotations

import pickle
import unittest
from unittest.mock import Mock

from experiments.fig8_sentence_memory import evaluate, recall_suffix, train_sentence
from seqmem.encoding import SSTDDiscreteEncoder
from seqmem.model import MemoryParams, SequentialMemory


def make_model(seed: int = 7) -> SequentialMemory:
    return SequentialMemory(
        encoder=SSTDDiscreteEncoder(num_columns=30, k=4, seed=seed),
        num_neurons_per_column=4,
        params=MemoryParams(l_match=2, forgetting_threshold=500.0),
        tie_break_seed=seed,
    )


def memory_structure(model: SequentialMemory) -> tuple[object, ...]:
    return tuple(
        (
            column_index,
            neuron_index,
            tuple(
                (
                    segment.active,
                    segment.target_time,
                    tuple(
                        sorted(
                            (
                                source,
                                synapse.weight,
                                synapse.delay,
                                synapse.age,
                            )
                            for source, synapse in segment.synapses.items()
                        )
                    ),
                )
                for segment in neuron.segments
            ),
        )
        for column_index, column in enumerate(model.columns)
        for neuron_index, neuron in enumerate(column.neurons)
    )


class Fig8NeuralRetrievalTests(unittest.TestCase):
    def test_spike_response_cache_is_bounded_without_changing_values(self) -> None:
        model = make_model()
        model.SPIKE_RESPONSE_CACHE_LIMIT = 2

        model._cached_spike_response(0.01)
        model._cached_spike_response(0.02)
        first_uncached = model._cached_spike_response(0.03)
        second_uncached = model._cached_spike_response(0.03)

        self.assertLessEqual(len(model._spike_response_cache), 2)
        self.assertEqual(first_uncached, second_uncached)

    def test_checkpoint_excludes_reproducible_response_cache(self) -> None:
        model = make_model()
        model._cached_spike_response(0.01)

        restored = pickle.loads(pickle.dumps(model))

        self.assertTrue(model._spike_response_cache)
        self.assertEqual(restored._spike_response_cache, {})

    def test_decode_from_prediction_does_not_predict_again(self) -> None:
        model = make_model()
        predicted = model.encoder.encode("B")
        model.predict_code = Mock(  # type: ignore[method-assign]
            side_effect=AssertionError("decode called predict_code")
        )

        decoded = model.decode_symbol_from_prediction(predicted)

        self.assertEqual(decoded, "B")
        model.predict_code.assert_not_called()  # type: ignore[union-attr]

    def test_a_b_c_retrieves_autonomously_without_observing_b(self) -> None:
        model = make_model()
        train_sentence(model, ["A", "B", "C"])
        original_observe = model.observe
        observed: list[tuple[str, bool]] = []

        def record_observe(symbol: str, learn: bool = True) -> dict[int, float]:
            observed.append((symbol, learn))
            return original_observe(symbol, learn=learn)

        model.observe = record_observe  # type: ignore[method-assign]
        recalled = recall_suffix(model, ["A"], 2, retrieval_mode="neural")

        self.assertEqual(recalled, ["B", "C"])
        self.assertEqual(observed, [("A", False)])

    def test_neural_recall_does_not_modify_long_term_memory(self) -> None:
        model = make_model()
        train_sentence(model, ["A", "B", "C", "D"])
        before = memory_structure(model)

        self.assertEqual(
            recall_suffix(model, ["A"], 3, retrieval_mode="neural"),
            ["B", "C", "D"],
        )

        self.assertEqual(memory_structure(model), before)

    def test_advance_contains_only_prediction_active_cells(self) -> None:
        model = make_model()
        train_sentence(model, ["A", "B", "C"])
        model.reset_state()
        model.predict_code()
        model.observe("A", learn=False)
        raw_code = model.predict_code()
        self.assertIsNotNone(raw_code)
        assert raw_code is not None
        expected_active = model.prediction_active_cells(raw_code)
        model.decode_symbol_from_prediction(raw_code)

        self.assertTrue(model.advance_prediction(raw_code))
        self.assertEqual(model.previous_active_cells, expected_active)
        self.assertEqual(model.previous_winners, expected_active)

    def test_proximal_replay_mode_preserves_legacy_behavior(self) -> None:
        model = make_model()
        train_sentence(model, ["A", "B", "C"])
        original_observe = model.observe
        observed: list[tuple[str, bool]] = []

        def record_observe(symbol: str, learn: bool = True) -> dict[int, float]:
            observed.append((symbol, learn))
            return original_observe(symbol, learn=learn)

        model.observe = record_observe  # type: ignore[method-assign]
        recalled = recall_suffix(
            model, ["A"], 2, retrieval_mode="proximal-replay"
        )

        self.assertEqual(recalled, ["B", "C"])
        self.assertEqual(observed, [("A", False), ("B", False), ("C", False)])

    def test_predict_symbol_apis_call_predict_code_once_each(self) -> None:
        model = make_model()
        predicted = model.encoder.encode("B")
        predictor = Mock(return_value=predicted)
        model.predict_code = predictor  # type: ignore[method-assign]

        self.assertEqual(model.predict_symbol(), "B")
        self.assertEqual(predictor.call_count, 1)
        predictor.reset_mock()
        self.assertEqual(model.predict_symbols(max_predictions=2), ["B"])
        self.assertEqual(predictor.call_count, 1)

    def test_checkpoint_evaluation_does_not_change_training_or_rng(self) -> None:
        sentences = [
            [f"s{sentence}-{word}" for word in range(10)]
            for sentence in range(4)
        ]
        frequent = make_model(seed=13)
        final_only = make_model(seed=13)

        for index, sentence in enumerate(sentences, start=1):
            train_sentence(frequent, sentence)
            evaluate(frequent, sentences[:index], 6, 0, 100 + index)
        for sentence in sentences:
            train_sentence(final_only, sentence)

        before = final_only.snapshot_transient_state()
        frequent_result = evaluate(frequent, sentences, 6, 0, 999)
        final_result = evaluate(final_only, sentences, 6, 0, 999)
        after = final_only.snapshot_transient_state()

        self.assertEqual(memory_structure(frequent), memory_structure(final_only))
        self.assertEqual(frequent_result, final_result)
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
