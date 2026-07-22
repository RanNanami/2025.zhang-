from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.diagnostics.fig8_diagnostic_common import (
    ReinforcementAccumulator,
    evaluate_diagnostic,
    kernel_diagnostic,
)
from experiments.fig8_sentence_memory import (
    evaluate,
    parse_args,
    recall_suffix,
    train_sentence,
)
from seqmem.encoding import SSTDDiscreteEncoder
from seqmem.model import (
    MemoryParams,
    PredictionTrace,
    ReinforcementTrace,
    SequentialMemory,
)


def make_model(seed: int = 23) -> SequentialMemory:
    return SequentialMemory(
        encoder=SSTDDiscreteEncoder(num_columns=30, k=4, seed=seed),
        num_neurons_per_column=4,
        params=MemoryParams(l_match=2, forgetting_threshold=500.0),
        tie_break_seed=seed,
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


def prediction_signature(model: SequentialMemory) -> tuple[object, ...]:
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


class Fig8FalsePositiveDiagnosticTests(unittest.TestCase):
    def test_response_scale_semantics(self) -> None:
        normalized = kernel_diagnostic(None)
        literal = kernel_diagnostic(1.0)

        self.assertAlmostEqual(float(normalized["kernel_peak"]), 1.0)
        self.assertAlmostEqual(
            float(literal["kernel_peak"]),
            float(literal["unscaled_kernel_peak"]),
        )
        self.assertEqual(normalized["min_initial_synapses_to_threshold"], 2)
        self.assertEqual(literal["min_initial_synapses_to_threshold"], 4)
        self.assertEqual(normalized["tau_m"], literal["tau_m"])
        self.assertEqual(normalized["tau_s"], literal["tau_s"])

    def test_trace_does_not_change_prediction_decode_or_rng(self) -> None:
        model = make_model()
        train_sentence(model, ["A", "B", "C"])
        model.reset_state()
        model.predict_code()
        model.observe("A", learn=False)
        before = model.snapshot_transient_state()

        plain = model.predict_code()
        assert plain is not None
        plain_candidates = prediction_signature(model)
        plain_decoded = model.decode_symbol_from_prediction(plain)
        plain_rng = model._decode_rng.getstate()

        model.restore_transient_state(before)
        trace = PredictionTrace()
        traced = model.predict_code(trace=trace)
        assert traced is not None
        traced_candidates = prediction_signature(model)
        traced_decoded = model.decode_symbol_from_prediction(traced)

        self.assertEqual(traced, plain)
        self.assertEqual(traced_candidates, plain_candidates)
        self.assertEqual(traced_decoded, plain_decoded)
        self.assertEqual(model._decode_rng.getstate(), plain_rng)
        self.assertTrue(trace.segments)

    def test_eventwise_inhibition_preserves_real_prediction_neurons(self) -> None:
        model = make_model()
        train_sentence(model, ["A", "B", "C"])
        model.reset_state()
        model.predict_code()
        model.observe("A", learn=False)
        raw = model.predict_code()
        assert raw is not None

        inhibited = model.select_prediction_events(raw)
        assert inhibited is not None
        active = model.prediction_active_cells(inhibited)

        self.assertEqual(len(active), len(inhibited.events))
        for event in inhibited.events:
            matches = [
                candidate
                for candidate in model.last_prediction_candidates[event.column]
                if abs(candidate.time - event.time)
                <= model.params.timing_tolerance
            ]
            self.assertTrue(matches)
            winner = max(matches, key=lambda candidate: candidate.score)
            cell = event.column * len(model.columns[0].neurons) + winner.neuron_index
            self.assertIn(cell, active)

    def test_eventwise_inhibited_code_is_sparse(self) -> None:
        model = make_model()
        train_sentence(model, ["A", "B", "C"])
        model.reset_state()
        model.predict_code()
        model.observe("A", learn=False)
        raw = model.predict_code()
        assert raw is not None
        inhibited = model.select_prediction_events(raw)
        assert inhibited is not None

        self.assertLessEqual(len(inhibited.events), model.encoder.k)
        raw_columns = {event.column for event in raw.events}
        self.assertTrue(all(event.column in raw_columns for event in inhibited.events))
        for event_time in model.encoder.event_times:
            near = [
                event
                for event in inhibited.events
                if abs(event.time - event_time) <= model.params.timing_tolerance
            ]
            self.assertLessEqual(len({event.column for event in near}), 1)

    def test_a_b_c_works_with_raw_and_eventwise_without_replay(self) -> None:
        for propagation in ("raw", "eventwise-inhibited"):
            model = make_model()
            train_sentence(model, ["A", "B", "C"])
            before = structure(model)
            observed: list[str] = []
            original = model.observe

            def record(symbol: str, learn: bool = True):
                observed.append(symbol)
                return original(symbol, learn=learn)

            model.observe = record  # type: ignore[method-assign]
            recalled = recall_suffix(
                model,
                ["A"],
                2,
                neural_propagation=propagation,
            )
            self.assertEqual(recalled, ["B", "C"])
            self.assertEqual(observed, ["A"])
            self.assertEqual(structure(model), before)

    def test_raw_and_inhibited_evaluate_same_trained_state_order_independently(
        self,
    ) -> None:
        model = make_model()
        sentences = [["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]]
        train_sentence(model, sentences[0])
        before = structure(model)

        raw_first = evaluate(model, sentences, 6, 0, 5, neural_propagation="raw")
        inhibited_second = evaluate(
            model,
            sentences,
            6,
            0,
            5,
            neural_propagation="eventwise-inhibited",
        )
        inhibited_first = evaluate(
            model,
            sentences,
            6,
            0,
            5,
            neural_propagation="eventwise-inhibited",
        )
        raw_second = evaluate(model, sentences, 6, 0, 5, neural_propagation="raw")

        self.assertEqual(raw_first, raw_second)
        self.assertEqual(inhibited_first, inhibited_second)
        self.assertEqual(structure(model), before)

    def test_prefix_diagnostics_are_read_only(self) -> None:
        model = make_model()
        sentence = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        train_sentence(model, sentence)
        before = structure(model)
        expected = evaluate(model, [sentence], 6, 0, 9)
        cue_rows: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as directory:
            evaluate_diagnostic(
                model,
                [sentence],
                propagation="raw",
                details_sample_sentences=1,
                segment_trace_path=Path(directory) / "trace.jsonl",
                cue_rows=cue_rows,
            )
        actual = evaluate(model, [sentence], 6, 0, 9)

        self.assertEqual(len(cue_rows), 6)
        self.assertEqual(actual, expected)
        self.assertEqual(structure(model), before)

    def test_event_selection_has_no_ground_truth_input(self) -> None:
        parameters = inspect.signature(
            SequentialMemory.select_prediction_events
        ).parameters
        self.assertNotIn("expected_word", parameters)
        self.assertNotIn("expected_suffix", parameters)
        self.assertNotIn("candidate_symbols", parameters)

    def test_main_fig8_cli_keeps_raw_propagation_as_default(self) -> None:
        with patch("sys.argv", ["fig8_sentence_memory.py"]):
            self.assertEqual(parse_args().neural_propagation, "raw")

    def test_scenario1_summary_keeps_full_contributor_distribution(self) -> None:
        accumulator = ReinforcementAccumulator()
        for contributors in (2, 4, 10, 12):
            accumulator(
                ReinforcementTrace(
                    scenario="scenario1",
                    target_column=0,
                    target_neuron=0,
                    segment_identity=7,
                    contributing_synapse_count=contributors,
                    weights_before=(0.5,),
                    weights_after=(0.6,),
                )
            )

        summary = accumulator.summary()
        self.assertEqual(summary["scenario1_contributor_2_fraction"], 0.25)
        self.assertEqual(summary["scenario1_contributor_4_fraction"], 0.25)
        self.assertEqual(summary["scenario1_contributor_0_fraction"], 0.0)
        self.assertEqual(summary["scenario1_contributor_10_plus_fraction"], 0.5)
        self.assertEqual(summary["scenario1_max_reinforcement_number"], 4.0)


if __name__ == "__main__":
    unittest.main()
