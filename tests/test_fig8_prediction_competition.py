from __future__ import annotations

import inspect
import unittest
from dataclasses import asdict

from experiments.diagnostics.fig8_prediction_competition import (
    evaluate_competition,
    memory_structure,
)
from experiments.fig8_sentence_memory import train_sentence
from seqmem.encoding import SSTDDiscreteEncoder
from seqmem.model import MemoryParams, PredictionTrace, SequentialMemory


def make_model(
    *,
    seed: int = 29,
    delay_mode: str = "current-delay",
) -> SequentialMemory:
    return SequentialMemory(
        encoder=SSTDDiscreteEncoder(num_columns=30, k=4, seed=seed),
        num_neurons_per_column=4,
        params=MemoryParams(
            l_match=2,
            forgetting_threshold=500.0,
            response_scale=1.0,
            synapse_delay_mode=delay_mode,
        ),
        tie_break_seed=seed,
    )


def candidate_core(model: SequentialMemory) -> tuple[object, ...]:
    return tuple(
        (
            column,
            candidate.neuron_index,
            candidate.score,
            candidate.time,
        )
        for column, candidates in sorted(model.last_prediction_candidates.items())
        for candidate in candidates
    )


def train_with_optional_trace(
    model: SequentialMemory,
    sentence: list[str],
    traced: bool,
) -> None:
    model.reset_state()
    for word in sentence:
        model.predict_code(trace=PredictionTrace() if traced else None)
        model.observe(word)


class Fig8PredictionCompetitionTests(unittest.TestCase):
    def test_trace_switch_preserves_prediction_training_and_rng(self) -> None:
        plain_model = make_model()
        traced_model = make_model()
        sentence = ["A", "B", "C", "D"]
        train_with_optional_trace(plain_model, sentence, False)
        train_with_optional_trace(traced_model, sentence, True)

        self.assertEqual(memory_structure(traced_model), memory_structure(plain_model))
        self.assertEqual(
            traced_model._decode_rng.getstate(), plain_model._decode_rng.getstate()
        )
        self.assertEqual(
            traced_model._learning_rng.getstate(),
            plain_model._learning_rng.getstate(),
        )

        plain_model.reset_state()
        traced_model.reset_state()
        for model in (plain_model, traced_model):
            model.predict_code()
            model.observe("A", learn=False)
        raw_plain = plain_model.predict_code()
        raw_traced = traced_model.predict_code(trace=PredictionTrace())
        self.assertEqual(raw_traced, raw_plain)
        self.assertEqual(candidate_core(traced_model), candidate_core(plain_model))

    def test_peak_and_crossing_metadata_belong_to_selected_candidate(self) -> None:
        model = make_model()
        train_sentence(model, ["A", "B", "C"])
        model.reset_state()
        model.predict_code()
        model.observe("A", learn=False)
        trace = PredictionTrace()
        raw = model.predict_code(trace=trace)
        self.assertIsNotNone(raw)

        accepted = {
            item.segment_identity: item
            for item in trace.segments
            if item.entered_raw_prediction
        }
        for candidates in model.last_prediction_candidates.values():
            for candidate in candidates:
                item = accepted[id(candidate.segment)]
                self.assertEqual(
                    candidate.peak_dendritic_potential,
                    item.peak_dendritic_potential,
                )
                self.assertEqual(
                    candidate.dendritic_crossing_time,
                    item.first_threshold_crossing_time,
                )
                self.assertEqual(
                    candidate.predicted_soma_firing_time,
                    item.predicted_soma_firing_time,
                )

    def test_ranking_apis_have_no_ground_truth_parameters(self) -> None:
        for method in (
            SequentialMemory.select_prediction_events,
            SequentialMemory.prediction_candidate_sort_key,
        ):
            parameters = inspect.signature(method).parameters
            for forbidden in (
                "expected",
                "expected_word",
                "expected_suffix",
                "ground_truth",
                "candidate_symbols",
            ):
                self.assertNotIn(forbidden, parameters)

    def test_all_rankings_are_read_only(self) -> None:
        sentence = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        model = make_model()
        train_sentence(model, sentence)
        before = memory_structure(model)
        for mode in SequentialMemory.PREDICTION_RANKING_MODES:
            evaluate_competition(
                model,
                [sentence],
                ranking_mode=mode,
                delay_mode="current-delay",
            )
            self.assertEqual(memory_structure(model), before)

    def test_peak_aligned_mode_only_changes_new_synapse_delay(self) -> None:
        current = make_model(delay_mode="current-delay")
        aligned = make_model(delay_mode="peak-aligned-delay")
        current_params = asdict(current.params)
        aligned_params = asdict(aligned.params)
        current_params.pop("synapse_delay_mode")
        aligned_params.pop("synapse_delay_mode")

        self.assertEqual(current_params, aligned_params)
        self.assertEqual(current.encoder.encode("A"), aligned.encoder.encode("A"))
        self.assertEqual(current.encoder.encode("B"), aligned.encoder.encode("B"))
        for target_time, source_time in ((0.0, 0.5), (0.25, 0.25), (0.5, 0.0)):
            current_delay = current._new_synapse_delay(target_time, source_time)
            aligned_delay = aligned._new_synapse_delay(target_time, source_time)
            self.assertAlmostEqual(
                current_delay - aligned_delay,
                current._kernel_peak_time,
            )

    def test_a_b_c_autonomous_recall_survives_reasonable_modes(self) -> None:
        for delay_mode in SequentialMemory.SYNAPSE_DELAY_MODES:
            for ranking_mode in SequentialMemory.PREDICTION_RANKING_MODES:
                model = make_model(delay_mode=delay_mode)
                # The aligned diagnostic advances events earlier; this toy uses
                # the smallest tolerance that covers both timing conventions.
                model.params.timing_tolerance = 0.05
                model.params.w0 = 0.7
                train_sentence(model, ["A", "B", "C"])
                model.reset_state()
                model.predict_code()
                model.observe("A", learn=False)
                recalled: list[str] = []
                for _step in range(2):
                    raw = model.predict_code(trace=PredictionTrace())
                    self.assertIsNotNone(raw)
                    assert raw is not None
                    propagated = model.select_prediction_events(
                        raw,
                        ranking_mode=ranking_mode,
                    )
                    self.assertIsNotNone(propagated)
                    assert propagated is not None
                    recalled.append(
                        model.decode_symbol_from_prediction(propagated) or ""
                    )
                    self.assertTrue(model.advance_prediction(propagated))
                self.assertEqual(recalled, ["B", "C"])

    def test_strict_defaults_remain_current_score_and_current_delay(self) -> None:
        self.assertEqual(MemoryParams().synapse_delay_mode, "current-delay")
        model = SequentialMemory(
            encoder=SSTDDiscreteEncoder(num_columns=10, k=2, seed=1)
        )
        self.assertIn("current-score", model.PREDICTION_RANKING_MODES)
        self.assertEqual(model.params.synapse_delay_mode, "current-delay")


if __name__ == "__main__":
    unittest.main()
