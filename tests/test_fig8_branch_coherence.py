from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

from experiments.diagnostics.fig8_branch_coherence import (
    evaluate,
    train_with_provenance,
)
from experiments.diagnostics.fig8_prediction_competition import memory_structure
from experiments.fig8_sentence_memory import parse_args, train_sentence
from seqmem.encoding import SSTDDiscreteEncoder
from seqmem.model import (
    MemoryParams,
    PredictionCandidate,
    PredictionTrace,
    Segment,
    SequentialMemory,
    Synapse,
    SynapsePSPContribution,
)


def make_model(capture: bool = True) -> SequentialMemory:
    return SequentialMemory(
        encoder=SSTDDiscreteEncoder(num_columns=30, k=4, seed=37),
        num_neurons_per_column=4,
        params=MemoryParams(
            l_match=2,
            forgetting_threshold=500.0,
            response_scale=1.0,
            capture_prediction_contributions=True,
            capture_branch_diagnostics=capture,
        ),
        tie_break_seed=37,
    )


def trained_model() -> tuple[SequentialMemory, set[frozenset[int]]]:
    model = make_model()
    winners = train_with_provenance(model, [["A", "B", "C"]])
    return model, winners


class Fig8BranchCoherenceTests(unittest.TestCase):
    def test_provenance_switch_preserves_training_prediction_and_rng(self) -> None:
        plain = make_model(False)
        traced = make_model(True)
        sentence = ["A", "B", "C", "D"]
        train_sentence(plain, sentence)
        train_with_provenance(traced, [sentence])
        self.assertEqual(memory_structure(plain), memory_structure(traced))
        self.assertEqual(plain._learning_rng.getstate(), traced._learning_rng.getstate())
        self.assertEqual(plain._decode_rng.getstate(), traced._decode_rng.getstate())

    def test_burst_labels_preserve_active_cell_identity(self) -> None:
        model = make_model()
        model.reset_state()
        model.predict_code()
        model.observe("A", learn=False)
        self.assertEqual(
            set(model.previous_active_cells),
            model.previous_predicted_sources | model.previous_burst_only_sources,
        )
        self.assertFalse(
            model.previous_predicted_sources & model.previous_burst_only_sources
        )

    def _candidate(
        self,
        predicted: tuple[float, ...],
        burst: tuple[float, ...],
    ) -> tuple[SequentialMemory, PredictionCandidate]:
        model = make_model()
        model.previous_predicted_sources = set(range(len(predicted)))
        model.previous_burst_only_sources = set(
            range(len(predicted), len(predicted) + len(burst))
        )
        values = [*predicted, *burst]
        segment = Segment(
            synapses={
                source: Synapse(source=source, weight=1.0)
                for source in range(len(values))
            }
        )
        candidate = PredictionCandidate(
            neuron_index=0,
            score=1.0,
            time=0.0,
            segment=segment,
            crossing_synapse_contributions=tuple(
                SynapsePSPContribution(
                    source_cell_id=source,
                    source_time=0.0,
                    arrival_time=0.0,
                    weight=1.0,
                    psp_contribution=value,
                )
                for source, value in enumerate(values)
            ),
        )
        return model, candidate

    def test_psp_decomposition_sums_to_total(self) -> None:
        model, candidate = self._candidate((0.4, 0.2), (0.3, 0.1))
        trace = model.candidate_burst_psp_trace(candidate)
        self.assertAlmostEqual(
            trace.total_psp,
            trace.predicted_source_psp
            + trace.burst_only_psp
            + trace.unlabelled_psp,
        )

    def test_burst_classification(self) -> None:
        cases = (
            ((0.6, 0.5), (0.1,), "predicted-supported"),
            ((0.4,), (0.7,), "burst-assisted"),
            ((0.1,), (0.6, 0.5), "burst-only"),
        )
        for predicted, burst, expected in cases:
            model, candidate = self._candidate(predicted, burst)
            self.assertEqual(
                model.candidate_burst_psp_trace(candidate).classification,
                expected,
            )

    def test_beam_has_no_ground_truth_or_provenance_inputs(self) -> None:
        method = SequentialMemory.select_coherent_prediction_events
        parameters = inspect.signature(method).parameters
        for forbidden in ("expected", "expected_word", "provenance", "sentence"):
            self.assertNotIn(forbidden, parameters)
        source = inspect.getsource(method)
        self.assertNotIn("creation_sentence", source)
        self.assertNotIn("creation_transition", source)

    def test_beam_uses_real_candidates_without_encoding(self) -> None:
        model, _winners = trained_model()
        model.reset_state()
        model.predict_code()
        model.observe("A", learn=False)
        raw = model.predict_code(trace=PredictionTrace())
        assert raw is not None
        candidate_events = {
            (column, round(candidate.time, 12))
            for column, candidates in model.last_prediction_candidates.items()
            for candidate in candidates
        }
        with patch.object(model.encoder, "encode", side_effect=AssertionError):
            selected = model.select_coherent_prediction_events(
                raw,
                beam_width=4,
                lambda_coherence=0.5,
            )
        assert selected is not None
        self.assertTrue(
            all(
                (event.column, round(event.time, 12)) in candidate_events
                for event in selected.events
            )
        )

    def test_beam_is_read_only_and_order_independent(self) -> None:
        sentence = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        model = make_model()
        winners = train_with_provenance(model, [sentence])
        before = memory_structure(model)
        beam_first = evaluate(
            model,
            [sentence],
            winners,
            selection="coherent-beam",
            collect_details=False,
        )[0]
        local = evaluate(
            model,
            [sentence],
            winners,
            selection="local-top1",
            collect_details=False,
        )[0]
        beam_second = evaluate(
            model,
            [sentence],
            winners,
            selection="coherent-beam",
            collect_details=False,
        )[0]
        self.assertEqual(beam_first, beam_second)
        self.assertEqual(memory_structure(model), before)
        self.assertEqual(local["diagnostic_label"], "nonpaper diagnostic")

    def test_a_b_c_remains_autonomous(self) -> None:
        for selection in ("local-top1", "coherent-beam"):
            model, _winners = trained_model()
            train_with_provenance(model, [["A", "B", "C"], ["A", "B", "C"]])
            model.reset_state()
            model.predict_code()
            model.observe("A", learn=False)
            recalled: list[str] = []
            for _step in range(2):
                raw = model.predict_code(trace=PredictionTrace())
                assert raw is not None
                if selection == "local-top1":
                    selected = model.select_prediction_events(raw)
                else:
                    selected = model.select_coherent_prediction_events(
                        raw,
                        beam_width=4,
                        lambda_coherence=0.5,
                    )
                assert selected is not None
                recalled.append(model.decode_symbol_from_prediction(selected) or "")
                self.assertTrue(model.advance_prediction(selected))
            self.assertEqual(recalled, ["B", "C"])

    def test_strict_defaults_remain_unchanged(self) -> None:
        self.assertFalse(MemoryParams().capture_branch_diagnostics)
        with patch("sys.argv", ["fig8_sentence_memory.py"]):
            args = parse_args()
        self.assertEqual(args.neural_propagation, "raw")
        self.assertEqual(args.retrieval_mode, "neural")


if __name__ == "__main__":
    unittest.main()
