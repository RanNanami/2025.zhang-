from __future__ import annotations

import unittest

from experiments.diagnostics.fig8_temporal_confirmation_semantics import (
    classify_pre_gate_candidate_count,
    delta_bin,
    sstd_rank_for_column,
    train_capture,
)
from seqmem.encoding import SSTDDiscreteEncoder


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
        from seqmem.model import MemoryParams

        self.assertFalse(MemoryParams().capture_temporal_confirmation_diagnostics)


if __name__ == "__main__":
    unittest.main()
