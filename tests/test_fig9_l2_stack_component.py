import tempfile
import unittest
from pathlib import Path

from experiments.diagnostics.analyze_fig9_l2_stack_component_2x2 import (
    CELLS,
    effects,
    protocol_matrix,
)
from experiments.diagnostics.analyze_fig9_lmatch_stack_2x2 import (
    _native_attempts_for_paths,
)


class Fig9L2StackComponentTests(unittest.TestCase):
    def test_component_matrix_has_expected_factor_cells(self):
        rows = protocol_matrix()
        self.assertEqual([row["cell"] for row in rows], list(CELLS))
        self.assertEqual(rows[0]["competition"], "off")
        self.assertEqual(rows[1]["selector"], "max_candidate_score")
        self.assertEqual(rows[2]["competition"], "competitive_raw")
        self.assertEqual(rows[3]["selector"], "max_candidate_score")
        self.assertTrue(all(row["L_match"] == 2 for row in rows))

    def test_component_effects_use_p0_as_common_baseline(self):
        def row(value):
            return {
                "record_errors": {},
                "MAPE": value,
                "step1": value,
                "step2": value,
                "step3": value,
                "step4": value,
                "step5": value,
                "raw_mean": value,
                "emitted_mean": value,
                "segments": value,
                "synapses": value,
                "contributors_mean": value,
            }

        metrics = {
            "P0_RAW_L2": row(10),
            "P1_SELECTOR_ONLY_L2": row(9),
            "P2_COMPETITION_ONLY_L2": row(8),
            "P3_FULL_STACK_L2": row(6),
        }
        mape = next(item for item in effects(metrics) if item["metric"] == "MAPE")
        self.assertEqual(mape["selector_effect_P1_minus_P0"], -1)
        self.assertEqual(mape["competition_effect_P2_minus_P0"], -2)
        self.assertEqual(mape["full_stack_effect_P3_minus_P0"], -4)
        self.assertEqual(mape["selector_competition_interaction"], -1)

    def test_native_history_combines_distinct_attempt_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            (first / "process_attempt_01.json").write_text('{"exit_code": 3221225477}', encoding="utf-8")
            (second / "process_attempt_01.json").write_text('{"exit_code": 0}', encoding="utf-8")
            self.assertEqual(_native_attempts_for_paths([first, second]), (2, 1, 1))


if __name__ == "__main__":
    unittest.main()
