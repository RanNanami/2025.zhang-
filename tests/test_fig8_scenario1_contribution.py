from __future__ import annotations

import inspect
import unittest

from experiments.fig8_sentence_memory import evaluate, train_sentence
from seqmem.encoding import SSTDDiscreteEncoder
from seqmem.model import (
    MemoryParams,
    PredictionCandidate,
    PredictionTrace,
    ReinforcementTrace,
    SequentialMemory,
)


def make_model(mode: str = "arrival-window") -> SequentialMemory:
    return SequentialMemory(
        encoder=SSTDDiscreteEncoder(num_columns=30, k=4, seed=23),
        num_neurons_per_column=4,
        params=MemoryParams(
            l_match=2,
            forgetting_threshold=500.0,
            response_scale=1.0,
            scenario1_contribution_mode=mode,
            capture_prediction_contributions=True,
        ),
        tie_break_seed=23,
    )


def structure(model: SequentialMemory) -> tuple[object, ...]:
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


def cue_a_prediction(
    model: SequentialMemory,
) -> tuple[object, PredictionCandidate]:
    model.reset_state()
    model.predict_code()
    model.observe("A", learn=False)
    raw = model.predict_code()
    assert raw is not None
    candidate = next(
        candidate
        for candidates in model.last_prediction_candidates.values()
        for candidate in candidates
    )
    return raw, candidate


def candidate_core_signature(model: SequentialMemory) -> tuple[object, ...]:
    return tuple(
        (
            column,
            candidate.neuron_index,
            candidate.score,
            candidate.time,
            id(candidate.segment),
        )
        for column, candidates in sorted(model.last_prediction_candidates.items())
        for candidate in candidates
    )


class Fig8Scenario1ContributionTests(unittest.TestCase):
    def test_trace_toggle_preserves_prediction_metadata_and_rng(self) -> None:
        model = make_model()
        train_sentence(model, ["A", "B", "C"])
        model.params.capture_prediction_contributions = False
        model.reset_state()
        model.predict_code()
        model.observe("A", learn=False)
        snapshot = model.snapshot_transient_state()

        raw_without = model.predict_code()
        signature_without = candidate_core_signature(model)
        decode_rng_without = model._decode_rng.getstate()
        learning_rng_without = model._learning_rng.getstate()

        model.restore_transient_state(snapshot)
        trace = PredictionTrace()
        raw_with = model.predict_code(trace=trace)

        self.assertEqual(raw_with, raw_without)
        self.assertEqual(candidate_core_signature(model), signature_without)
        self.assertEqual(model._decode_rng.getstate(), decode_rng_without)
        self.assertEqual(model._learning_rng.getstate(), learning_rng_without)

    def test_continuous_modes_only_select_real_active_synapses(self) -> None:
        model = make_model()
        train_sentence(model, ["A", "B", "C"])
        _raw, candidate = cue_a_prediction(model)
        real_active = set(model.previous_active_cells)
        captured = {
            item.source_cell_id
            for item in candidate.crossing_synapse_contributions
            if item.psp_contribution > 0.0
        }

        for mode in ("continuous-positive", "continuous-causal"):
            selected = model.scenario1_contributing_sources(
                candidate,
                mode=mode,
            )
            self.assertTrue(selected)
            self.assertLessEqual(selected, real_active)
            self.assertLessEqual(selected, captured)
            self.assertTrue(
                all(source in candidate.segment.synapses for source in selected)
            )

    def test_causal_subset_is_minimal_at_threshold(self) -> None:
        model = make_model("continuous-causal")
        train_sentence(model, ["A", "B", "C"])
        _raw, candidate = cue_a_prediction(model)
        selected = model.scenario1_contributing_sources(candidate)
        by_source = {
            item.source_cell_id: item.psp_contribution
            for item in candidate.crossing_synapse_contributions
        }
        effective_threshold = (
            model.params.dendrite_threshold
            - model.params.integration_voltage_tolerance
        )
        total = model._dynamics.v_rest + sum(
            by_source[source] for source in selected
        )
        smallest = min(by_source[source] for source in selected)

        self.assertGreaterEqual(total, effective_threshold)
        self.assertLess(
            total - smallest,
            effective_threshold + model.params.integration_voltage_tolerance,
        )

    def test_scenario1_uses_the_matching_prediction_candidate(self) -> None:
        model = make_model("continuous-positive")
        train_sentence(model, ["A", "B", "C"])
        model.reset_state()
        model.predict_code()
        model.observe("A", learn=False)
        model.predict_code()
        candidate_ids = {
            id(candidate)
            for candidates in model.last_prediction_candidates.values()
            for candidate in candidates
        }
        traces: list[ReinforcementTrace] = []
        model.reinforcement_trace_callback = traces.append
        try:
            model.observe("B", learn=True)
        finally:
            model.reinforcement_trace_callback = None

        scenario1 = [item for item in traces if item.scenario == "scenario1"]
        self.assertTrue(scenario1)
        self.assertTrue(
            all(
                item.prediction_candidate_identity in candidate_ids
                for item in scenario1
            )
        )
        self.assertTrue(
            all(item.actual_positive_weakened_count == 0 for item in scenario1)
        )

    def test_evaluation_is_read_only_for_all_contribution_modes(self) -> None:
        sentence = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        for mode in SequentialMemory.SCENARIO1_CONTRIBUTION_MODES:
            model = make_model(mode)
            train_sentence(model, sentence)
            before = structure(model)
            evaluate(model, [sentence], 6, 0, 31)
            self.assertEqual(structure(model), before)

    def test_contribution_selection_has_no_ground_truth_parameter(self) -> None:
        parameters = inspect.signature(
            SequentialMemory.scenario1_contributing_sources
        ).parameters
        for forbidden in (
            "expected",
            "expected_word",
            "expected_suffix",
            "candidate_symbols",
            "ground_truth",
        ):
            self.assertNotIn(forbidden, parameters)

    def test_strict_default_remains_arrival_window(self) -> None:
        self.assertEqual(
            MemoryParams().scenario1_contribution_mode,
            "arrival-window",
        )
        self.assertFalse(MemoryParams().capture_prediction_contributions)


if __name__ == "__main__":
    unittest.main()
