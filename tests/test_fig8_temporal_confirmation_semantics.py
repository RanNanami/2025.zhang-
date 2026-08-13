from __future__ import annotations

import unittest

from experiments.diagnostics.fig8_diagnostic_common import build_model
from experiments.diagnostics.fig8_temporal_confirmation_semantics import (
    classify_pre_gate_candidate_count,
    delta_bin,
    sstd_rank_for_column,
    train_capture,
)
from seqmem.encoding import SSTDDiscreteEncoder, SpikeEvent, SymbolCode
from seqmem.model import MemoryParams, PredictionCandidate, Segment


def _manual_time_mismatch_model(mode: str):
    model = build_model(1.0, 19, temporal_confirmation_mode=mode)
    model.params.capture_intralayer_parity_diagnostics = True
    rows = []
    model.intralayer_parity_callback = rows.append
    traces = []
    model.reinforcement_trace_callback = traces.append
    segment = Segment(target_time=0.0)
    model.columns[1].neurons[0].segments.append(segment)
    candidate = PredictionCandidate(
        neuron_index=0,
        score=1.0,
        time=0.0,
        segment=segment,
    )
    model.last_prediction_candidates = {1: [candidate]}
    model.observe_code(
        SymbolCode(events=(SpikeEvent(column=1, time=0.5),)),
        learn=True,
    )
    return model, rows, traces, candidate, segment


class Fig8TemporalConfirmationSemanticsTests(unittest.TestCase):
    def test_candidate_classification(self):
        self.assertEqual(classify_pre_gate_candidate_count(0), "NO_PREDICTIVE_CANDIDATE")
        self.assertEqual(classify_pre_gate_candidate_count(1), "UNIQUE_PREDICTIVE_CANDIDATE")
        self.assertEqual(classify_pre_gate_candidate_count(2), "MULTIPLE_PREDICTIVE_CANDIDATES")
        self.assertEqual(classify_pre_gate_candidate_count(-1), "UNRESOLVABLE")

    def test_delta_bins_have_fixed_boundaries(self):
        self.assertEqual(delta_bin(0.0), "0-0.05")
        self.assertEqual(delta_bin(0.05), "0.05-0.10")
        self.assertEqual(delta_bin(0.10), "0.10-0.20")
        self.assertEqual(delta_bin(0.50), ">0.50")
        self.assertEqual(delta_bin(None), "UNRESOLVABLE")

    def test_sstd_rank_is_local_k_rank(self):
        encoder = SSTDDiscreteEncoder(num_columns=20, k=10, seed=3)
        code = encoder.encode("A")
        self.assertTrue(all(1 <= sstd_rank_for_column(code, event.column) <= 10 for event in code.events))
        missing_column = next(column for column in range(20) if column not in code.columns)
        self.assertIsNone(sstd_rank_for_column(code, missing_column))

    def test_trace_on_off_replays_exact_training(self):
        sentences = [["A", "B", "C"]]
        plain, _plain_rows, plain_predictions = train_capture(sentences, 17, temporal_trace=False)
        traced, traced_rows, traced_predictions = train_capture(sentences, 17, temporal_trace=True)
        self.assertEqual(plain_predictions, traced_predictions)
        self.assertEqual(plain._learning_rng.getstate(), traced._learning_rng.getstate())
        self.assertEqual(plain._decode_rng.getstate(), traced._decode_rng.getstate())
        self.assertTrue(any(row.get("phase") == "observation_event" for row in traced_rows))

    def test_trace_is_default_off(self):
        self.assertFalse(MemoryParams().capture_temporal_confirmation_diagnostics)

    def test_temporal_confirmation_mode_defaults_to_current(self):
        self.assertEqual(MemoryParams().temporal_confirmation_mode, "current")
        with self.assertRaises(ValueError):
            build_model(1.0, 19, temporal_confirmation_mode="invalid")

    def test_unique_identity_reuses_same_candidate_segment_after_time_rejection(self):
        model, rows, traces, candidate, segment = _manual_time_mismatch_model(
            "unique_predictive_identity"
        )
        observation = next(row for row in rows if row["phase"] == "observation_event")
        self.assertTrue(observation["identity_confirmation"])
        self.assertEqual(observation["scenario"], "scenario1")
        self.assertEqual(observation["selected_segment_identity"], id(segment))
        self.assertEqual(observation["predicted_candidate_identity"], id(candidate))
        self.assertEqual(
            [trace.scenario for trace in traces],
            ["scenario1"],
        )
        self.assertEqual(traces[0].segment_identity, id(segment))
        punishments = [
            row for row in rows if row["phase"] == "wrong_prediction_punishment"
        ]
        self.assertFalse(punishments)

    def test_identity_mode_does_not_create_s2_or_s3_for_timing_only_failure(self):
        _model, rows, _traces, _candidate, _segment = _manual_time_mismatch_model(
            "unique_predictive_identity"
        )
        observations = [row for row in rows if row["phase"] == "observation_event"]
        self.assertEqual([row["scenario"] for row in observations], ["scenario1"])
        self.assertFalse(any(row["scenario"] in {"scenario2", "scenario3"} for row in observations))

    def test_multiple_candidates_remain_current_behavior(self):
        model = build_model(1.0, 19, temporal_confirmation_mode="unique_predictive_identity")
        model.params.capture_intralayer_parity_diagnostics = True
        rows = []
        model.intralayer_parity_callback = rows.append
        first = Segment(target_time=0.0)
        second = Segment(target_time=0.0)
        model.columns[1].neurons[0].segments.append(first)
        model.columns[1].neurons[1].segments.append(second)
        model.last_prediction_candidates = {
            1: [
                PredictionCandidate(0, 1.0, 0.0, first),
                PredictionCandidate(1, 1.0, 0.0, second),
            ]
        }
        model.observe_code(
            SymbolCode(events=(SpikeEvent(column=1, time=0.5),)),
            learn=True,
        )
        observation = next(row for row in rows if row["phase"] == "observation_event")
        self.assertFalse(observation["identity_confirmation"])
        self.assertNotEqual(observation["scenario"], "scenario1")

    def test_identity_mode_is_read_only_when_learning_is_disabled(self):
        model = build_model(1.0, 19, temporal_confirmation_mode="unique_predictive_identity")
        model.params.capture_intralayer_parity_diagnostics = True
        rows = []
        model.intralayer_parity_callback = rows.append
        segment = Segment(target_time=0.0)
        model.columns[1].neurons[0].segments.append(segment)
        candidate = PredictionCandidate(0, 1.0, 0.0, segment)
        model.last_prediction_candidates = {1: [candidate]}
        model.observe_code(
            SymbolCode(events=(SpikeEvent(column=1, time=0.5),)),
            learn=False,
        )
        observation = next(row for row in rows if row["phase"] == "observation_event")
        self.assertFalse(observation["identity_confirmation"])


if __name__ == "__main__":
    unittest.main()
