from __future__ import annotations

import unittest
from copy import deepcopy

from seqmem.dynamics import DSNeuronState
from seqmem.encoding import SSTDDiscreteEncoder, SpikeEvent, SymbolCode
from seqmem.model import MemoryParams, PredictionCandidate, Segment, SequentialMemory, Synapse


class PaperCoreInhibitionAuditTests(unittest.TestCase):
    def make_model(self, neurons: int = 3) -> SequentialMemory:
        return SequentialMemory(
            encoder=SSTDDiscreteEncoder(num_columns=4, k=1, seed=3),
            num_neurons_per_column=neurons,
            params=MemoryParams(intracolumn_inhibition=True),
        )

    def test_predictive_candidate_is_not_implicitly_a_burst(self) -> None:
        model = self.make_model()
        segment = Segment(synapses={1: Synapse(1, weight=1.0)}, target_time=1.0)
        model.last_prediction_candidates = {
            0: [PredictionCandidate(2, 1.0, 0.0, segment)]
        }
        active = model.prediction_active_cells(
            SymbolCode((SpikeEvent(0, 0.0),))
        )
        self.assertEqual(set(active), {2})

    def test_prediction_advance_propagates_only_selected_cell(self) -> None:
        model = self.make_model()
        segment = Segment()
        model.last_prediction_candidates = {
            0: [PredictionCandidate(1, 1.0, 0.0, segment)]
        }
        self.assertTrue(model.advance_prediction(SymbolCode((SpikeEvent(0, 0.0),))))
        self.assertEqual(set(model.previous_active_cells), {1})
        self.assertEqual(set(model.previous_winners), {1})

    def test_unpredicted_external_column_preserves_paper_burst(self) -> None:
        model = self.make_model()
        active = model.external_input_active_cells(
            SymbolCode((SpikeEvent(0, 0.0),))
        )
        self.assertEqual(len(active), 3)

    def test_refractory_blocks_same_cycle_refire(self) -> None:
        state = DSNeuronState()
        params = MemoryParams().dynamics()
        state.trigger_dendritic_spike(0.0)
        self.assertTrue(state.try_fire(0.5, params))
        self.assertFalse(state.try_fire(0.6, params))

    def test_refractory_resets_after_cycle(self) -> None:
        state = DSNeuronState()
        params = MemoryParams().dynamics()
        state.trigger_dendritic_spike(0.0)
        self.assertTrue(state.try_fire(0.5, params))
        self.assertFalse(state.is_refractory(1.5, params))

    def test_encoding_k_is_independent_of_retrieval_state(self) -> None:
        encoder = SSTDDiscreteEncoder(num_columns=10, k=3, seed=7)
        self.assertEqual(len(encoder.encode("x").events), 3)

    def test_ground_truth_is_not_an_argument_to_prediction(self) -> None:
        self.assertNotIn("expected", SequentialMemory.predict_code.__code__.co_varnames)

    def test_candidate_score_is_not_claimed_as_soma_potential(self) -> None:
        candidate = PredictionCandidate(0, 1.0, 0.0, Segment())
        self.assertIsNone(candidate.predicted_soma_firing_time)

    def test_same_column_prediction_resolves_to_one_selected_cell(self) -> None:
        model = self.make_model()
        segment_a = Segment()
        segment_b = Segment()
        model.last_prediction_candidates = {
            0: [
                PredictionCandidate(0, 1.0, 0.0, segment_a),
                PredictionCandidate(2, 0.9, 0.0, segment_b),
            ]
        }
        active = model.prediction_active_cells(
            SymbolCode((SpikeEvent(0, 0.0),))
        )
        self.assertEqual(len(active), 1)
        self.assertIn(next(iter(active)), {0, 2})

    def test_advance_prediction_is_transient_and_does_not_learn(self) -> None:
        model = self.make_model()
        model.previous_active_cells = {99: 0.0}
        model.previous_winners = {99: 0.0}
        before = deepcopy(model.columns)
        learning_rng = model._learning_rng.getstate()
        model.last_prediction_candidates = {
            0: [PredictionCandidate(1, 1.0, 0.0, Segment())]
        }

        self.assertTrue(model.advance_prediction(SymbolCode((SpikeEvent(0, 0.0),))))

        self.assertEqual(model._learning_rng.getstate(), learning_rng)
        self.assertEqual(
            [tuple(len(n.segments) for n in c.neurons) for c in model.columns],
            [tuple(len(n.segments) for n in c.neurons) for c in before],
        )

    def test_prediction_time_tolerance_does_not_create_an_external_burst(self) -> None:
        model = self.make_model()
        model.last_prediction_candidates = {
            0: [PredictionCandidate(1, 1.0, 0.0, Segment())]
        }
        active = model.prediction_active_cells(
            SymbolCode((SpikeEvent(0, 0.02),))
        )
        self.assertEqual(set(active), {1})


if __name__ == "__main__":
    unittest.main()
