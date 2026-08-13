from __future__ import annotations

import inspect
import unittest

from experiments.diagnostics.fig8_diagnostic_common import build_model
from experiments.diagnostics.fig8_scenario_temporal_credit_parity import (
    delta_t,
    legacy_label,
    paper_scenario_class,
    timing_bin,
    timing_identity_class,
)
from experiments.fig8_sentence_memory import train_sentence
from seqmem.model import PredictionCandidate, Segment, Synapse


def structure(model):
    return tuple(
        (
            column_index,
            neuron_index,
            tuple(
                (
                    segment.active,
                    segment.target_time,
                    tuple(
                        (source, synapse.weight, synapse.delay, synapse.age)
                        for source, synapse in sorted(segment.synapses.items())
                    ),
                )
                for segment in neuron.segments
            ),
        )
        for column_index, column in enumerate(model.columns)
        for neuron_index, neuron in enumerate(column.neurons)
    )


def make_pair():
    return (
        build_model(1.0, 211, capture_prediction_contributions=True, capture_branch_diagnostics=True),
        build_model(1.0, 211, capture_prediction_contributions=True, capture_branch_diagnostics=True),
    )


class Fig8ScenarioTemporalCreditParityTests(unittest.TestCase):
    def test_taxonomy_trace_is_posthoc_only(self):
        self.assertEqual(paper_scenario_class({"phase": "wrong_prediction_punishment"}), "PAPER_S3_FAILED_PREDICTION")

    def test_legacy_labels_are_preserved(self):
        self.assertEqual(legacy_label({"phase": "observation_event", "scenario": "scenario3"}), "scenario3")

    def test_paper_class_is_deterministic(self):
        row = {"phase": "reinforcement_event", "scenario": "scenario1"}
        self.assertEqual(paper_scenario_class(row), paper_scenario_class(row))

    def test_s2_new_segment_is_not_paper_s3(self):
        self.assertEqual(paper_scenario_class({"phase": "segment_created", "creation_scenario": "scenario3"}), "PAPER_S2B_NO_MATCH_NEW_SEGMENT")

    def test_s3_punishment_does_not_create_segment(self):
        row = {"phase": "wrong_prediction_punishment"}
        self.assertEqual(paper_scenario_class(row), "PAPER_S3_FAILED_PREDICTION")

    def test_trace_on_off_predictions_are_equal(self):
        plain, traced = make_pair()
        traced.params.capture_intralayer_parity_diagnostics = True
        traced.intralayer_parity_callback = lambda _row: None
        for model in (plain, traced):
            train_sentence(model, ["A", "B", "C"])
            model.reset_state()
        first = plain.predict_code()
        second = traced.predict_code()
        self.assertEqual(first, second)

    def test_trace_on_off_weights_are_equal(self):
        plain, traced = make_pair()
        traced.params.capture_intralayer_parity_diagnostics = True
        traced.intralayer_parity_callback = lambda _row: None
        train_sentence(plain, ["A", "B", "C"])
        train_sentence(traced, ["A", "B", "C"])
        self.assertEqual(structure(plain), structure(traced))

    def test_trace_on_off_learning_rng_is_equal(self):
        plain, traced = make_pair()
        traced.params.capture_intralayer_parity_diagnostics = True
        traced.intralayer_parity_callback = lambda _row: None
        train_sentence(plain, ["A", "B", "C"])
        train_sentence(traced, ["A", "B", "C"])
        self.assertEqual(plain._learning_rng.getstate(), traced._learning_rng.getstate())

    def test_trace_on_off_decode_rng_is_equal(self):
        plain, traced = make_pair()
        traced.params.capture_intralayer_parity_diagnostics = True
        traced.intralayer_parity_callback = lambda _row: None
        train_sentence(plain, ["A", "B", "C"])
        train_sentence(traced, ["A", "B", "C"])
        self.assertEqual(plain._decode_rng.getstate(), traced._decode_rng.getstate())

    def test_same_candidate_identity_is_a_distinct_field(self):
        model = build_model(1.0, 211, capture_prediction_contributions=True, capture_branch_diagnostics=True)
        model.params.capture_intralayer_parity_diagnostics = True
        rows = []
        model.intralayer_parity_callback = rows.append
        source = model._cell_id(0, 0)
        segment = Segment(
            synapses={source: Synapse(source, delay=0.5, weight=0.5)},
            target_time=1.0,
        )
        model.columns[1].neurons[0].segments.append(segment)
        candidate = PredictionCandidate(0, 1.0, 0.0, segment)
        model.previous_active_cells = {source: 0.0}
        model.last_prediction_candidates = {1: [candidate]}
        model._punish_wrong_predictions({})
        punishments = [row for row in rows if row.get("phase") == "wrong_prediction_punishment"]
        self.assertTrue(punishments)
        self.assertTrue(all(row.get("predicted_candidate_identity") for row in punishments))

    def test_same_neuron_matching_is_deterministic(self):
        args = dict(same_neuron=True, same_column=True, same_order=True, within_tolerance=False, actual_present=True)
        self.assertEqual(timing_identity_class(**args), "CONFIRMED_SAME_NEURON_DIFFERENT_TIME")

    def test_delta_t_signed(self):
        self.assertAlmostEqual(delta_t(0.2, 0.1), 0.1)

    def test_timing_bin_boundaries(self):
        self.assertEqual(timing_bin(0.049), "0-0.05")
        self.assertEqual(timing_bin(0.05), "0.05-0.10")
        self.assertEqual(timing_bin(0.5), ">0.50")

    def test_missed_s1_by_timing_classification(self):
        result = timing_identity_class(same_neuron=True, same_column=True, same_order=True, within_tolerance=False, actual_present=True)
        self.assertEqual(result, "CONFIRMED_SAME_NEURON_DIFFERENT_TIME")

    def test_s3_timing_only_classification(self):
        self.assertEqual(timing_identity_class(same_neuron=False, same_column=True, same_order=True, within_tolerance=False, actual_present=True), "SAME_COLUMN_SAME_ORDER_DIFFERENT_TIME")

    def test_double_penalty_requires_same_neuron_and_mismatch(self):
        row = {"same_neuron": True, "within_tolerance": False, "punishment_reason": "PROXIMAL_TIME_MISMATCH"}
        self.assertTrue(row["same_neuron"] and not row["within_tolerance"] and row["punishment_reason"] == "PROXIMAL_TIME_MISMATCH")

    def test_audit_has_no_ground_truth_model_argument(self):
        from experiments.diagnostics.fig8_scenario_temporal_credit_parity import run_capture
        self.assertNotIn("expected_word", inspect.signature(run_capture).parameters)

    def test_audit_does_not_take_future_input(self):
        from experiments.diagnostics.fig8_scenario_temporal_credit_parity import run_capture
        self.assertNotIn("future", inspect.getsource(run_capture).lower())

    def test_strict_default_is_unchanged(self):
        from seqmem.model import MemoryParams
        self.assertIsNone(MemoryParams().response_scale)

    def test_v0_default_is_unchanged(self):
        self.assertIsNone(build_model(None, 211).params.response_scale)

    def test_diagnostic_v0_is_explicit(self):
        self.assertEqual(build_model(1.0, 211).params.response_scale, 1.0)

    def test_no_tolerance_sweep_argument(self):
        from experiments.diagnostics.fig8_scenario_temporal_credit_parity import main
        self.assertNotIn("tolerance", inspect.getsource(main).lower())

    def test_callback_does_not_change_long_term_state(self):
        plain, traced = make_pair()
        traced.params.capture_intralayer_parity_diagnostics = True
        traced.intralayer_parity_callback = lambda _row: None
        train_sentence(plain, ["A", "B", "C", "D"])
        train_sentence(traced, ["A", "B", "C", "D"])
        self.assertEqual(structure(plain), structure(traced))

    def test_actual_observation_and_rollout_are_separate_apis(self):
        from seqmem.model import SequentialMemory
        self.assertIn("advance_prediction", inspect.getsource(SequentialMemory.advance_prediction))
        self.assertNotIn("observe(", inspect.getsource(SequentialMemory.advance_prediction))

    def test_same_column_is_not_same_neuron(self):
        self.assertEqual(timing_identity_class(same_neuron=False, same_column=True, same_order=False, within_tolerance=True, actual_present=True), "SAME_COLUMN_DIFFERENT_NEURON")

    def test_missing_proximal_event_is_not_timing_match(self):
        self.assertEqual(timing_identity_class(same_neuron=None, same_column=True, same_order=None, within_tolerance=False, actual_present=False), "NO_TRUE_CONFIRMATION")

    def test_paper_s3_has_no_new_segment(self):
        self.assertFalse(paper_scenario_class({"phase": "wrong_prediction_punishment"}) == "PAPER_S2B_NO_MATCH_NEW_SEGMENT")


if __name__ == "__main__":
    unittest.main()
