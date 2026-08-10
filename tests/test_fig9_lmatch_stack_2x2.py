import tempfile
import unittest
from pathlib import Path

from experiments.diagnostics.analyze_fig9_lmatch_stack_2x2 import (
    CELL_NAMES,
    _effects,
    _expected_predictions,
    _parse_mapping,
    protocol_matrix,
)


class Fig9LMatchStackProtocolTests(unittest.TestCase):
    def test_matrix_has_four_named_cells(self):
        rows = protocol_matrix()
        self.assertEqual([row["cell"] for row in rows], list(CELL_NAMES))

    def test_raw_pair_only_changes_l_match_and_labels(self):
        rows = {row["cell"]: row for row in protocol_matrix()}
        allowed = {"L_match", "cell", "strict_reproduction", "lmatch_real_ablation", "label_note"}
        for field in rows["STRICT_RAW_L4"]:
            if field not in allowed:
                self.assertEqual(rows["STRICT_RAW_L4"][field], rows["STRICT_RAW_L2"][field], field)
        self.assertEqual(rows["STRICT_RAW_L4"]["L_match"], 4)
        self.assertEqual(rows["STRICT_RAW_L2"]["L_match"], 2)
        self.assertTrue(rows["STRICT_RAW_L4"]["strict_reproduction"])
        self.assertFalse(rows["STRICT_RAW_L2"]["strict_reproduction"])
        self.assertTrue(rows["STRICT_RAW_L2"]["lmatch_real_ablation"])

    def test_stack_pair_only_changes_l_match_and_labels(self):
        rows = {row["cell"]: row for row in protocol_matrix()}
        allowed = {"L_match", "cell", "strict_reproduction", "lmatch_real_ablation", "label_note"}
        for field in rows["DIAGNOSTIC_STACK_L4"]:
            if field not in allowed:
                self.assertEqual(rows["DIAGNOSTIC_STACK_L4"][field], rows["DIAGNOSTIC_STACK_L2"][field], field)
        self.assertEqual(rows["DIAGNOSTIC_STACK_L4"]["competition_mode"], "competitive_raw")
        self.assertEqual(rows["DIAGNOSTIC_STACK_L4"]["simultaneous_policy"], "batched")
        self.assertEqual(rows["DIAGNOSTIC_STACK_L4"]["intracolumn_selector"], "max_candidate_score")
        self.assertEqual(rows["DIAGNOSTIC_STACK_L4"]["L_match"], 4)
        self.assertEqual(rows["DIAGNOSTIC_STACK_L2"]["L_match"], 2)

    def test_forbidden_mechanisms_are_off(self):
        for row in protocol_matrix():
            self.assertFalse(row["rollout_learning"])
            self.assertFalse(row["future_covariates"])
            self.assertFalse(row["compensation"])
            self.assertFalse(row["decoded_replay"])

    def test_expected_prediction_counts(self):
        self.assertEqual(_expected_predictions(250, 200, 5), 45)
        self.assertEqual(_expected_predictions(500, 200, 5), 295)

    def test_effects_use_difference_in_differences(self):
        def row(limit, value):
            return {
                "limit": limit,
                "MAPE": value,
                "step1": value,
                "step2": value,
                "step3": value,
                "step4": value,
                "step5": value,
                "segments": value,
                "synapses": value,
                "raw_mean": value,
                "emitted_mean": value,
                "contributors_mean": value,
                "scenario1_mean": value,
                "scenario2_mean": value,
                "scenario3_mean": value,
            }

        metrics = {
            "STRICT_RAW_L4_250": row(250, 10),
            "STRICT_RAW_L2_250": row(250, 12),
            "DIAGNOSTIC_STACK_L4_250": row(250, 7),
            "DIAGNOSTIC_STACK_L2_250": row(250, 8),
        }
        effects = _effects(metrics)
        mape = next(item for item in effects if item["metric"] == "MAPE")
        self.assertEqual(mape["L2_effect_raw_B_minus_A"], 2)
        self.assertEqual(mape["L2_effect_stack_D_minus_C"], 1)
        self.assertEqual(mape["interaction_D_minus_C_minus_B_minus_A"], -1)

    def test_mapping_accepts_explicit_limit_suffix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            (path / "original_summary.json").write_text(
                '{"records_used": 250, "predictions": 45, "coverage": 1.0, "L_match": 4}',
                encoding="utf-8",
            )
            mapping = _parse_mapping([f"STRICT_RAW_L4_250={path}"])
            self.assertIn("STRICT_RAW_L4_250", mapping)


if __name__ == "__main__":
    unittest.main()
