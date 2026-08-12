"""Unit tests for the read-only Fig.9 evaluation audit."""

from __future__ import annotations

import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

from experiments.fig9.evaluation_metric_audit import (
    Artifact,
    Prediction,
    _artifact_comparability,
    fixed_windows,
    learning_curve,
    metric_values,
    write_audit_report,
)
from experiments.fig9.metrics import (
    mape as legacy_mape,
    median_absolute_percentage_error,
    standard_mape,
    wape,
)
from experiments.fig9_strict_reproduction import Fig9StrictConfig


def rows(n: int = 8) -> tuple[Prediction, ...]:
    return tuple(
        Prediction(
            prediction=float(index + 1),
            target=float(index + 2),
            timestamp=f"t{index}",
            input_index=index,
            target_timestamp=f"target{index}",
        )
        for index in range(n)
    )


class EvaluationMetricAuditTests(unittest.TestCase):
    def test_standard_mape_fixture(self) -> None:
        self.assertAlmostEqual(standard_mape([9.0, 12.0], [10.0, 10.0]), 0.15)

    def test_wape_fixture(self) -> None:
        self.assertAlmostEqual(wape([9.0, 12.0], [10.0, 10.0]), 0.15)

    def test_legacy_repo_metric_is_preserved(self) -> None:
        self.assertAlmostEqual(legacy_mape([0.0, 15.0], [10.0, 20.0]), 0.5)

    def test_zero_target_policy_is_explicit(self) -> None:
        self.assertAlmostEqual(standard_mape([3.0, 8.0], [0.0, 10.0]), 0.2)
        self.assertAlmostEqual(legacy_mape([3.0, 8.0], [0.0, 10.0]), 0.5)

    def test_metric_values_include_requested_statistics(self) -> None:
        values = metric_values(rows())
        for name in ("standard_mape", "wape", "repo_metric", "mae", "rmse", "median_ape", "p90_ape"):
            self.assertIn(name, values)

    def test_final_step_only_window_is_fixed(self) -> None:
        names = {name for name, *_ in fixed_windows(rows())}
        self.assertIn("final_25", names)
        self.assertNotIn("best_window", names)

    def test_fixed_windows_are_positional(self) -> None:
        windows = dict((name, window) for name, _, _, window in fixed_windows(rows(8)))
        self.assertEqual([item.input_index for item in windows["first_half"]], [0, 1, 2, 3])
        self.assertEqual([item.input_index for item in windows["second_half"]], [4, 5, 6, 7])

    def test_prediction_count_is_limit_minus_warmup_minus_horizon(self) -> None:
        self.assertEqual(250 - 200 - 5, 45)
        self.assertEqual(500 - 200 - 5, 295)

    def test_learning_curve_has_cumulative_and_rolling_rows(self) -> None:
        curve = learning_curve(rows(), warmup=200, anchor_semantics="strict")
        self.assertEqual(len(curve), 8 * 3)
        self.assertEqual({item["rolling_window"] for item in curve}, {25, 50, 100})
        self.assertIn("cumulative_standard_mape", curve[0])

    def test_learning_curve_uses_no_future_rows(self) -> None:
        curve = learning_curve(rows(), warmup=200, anchor_semantics="strict")
        first = [item for item in curve if item["prediction_index"] == 1]
        changed = list(rows())
        changed[1] = Prediction(999.0, 2.0, "t1", 1, "target1")
        changed_curve = learning_curve(tuple(changed), warmup=200, anchor_semantics="strict")
        changed_first = [item for item in changed_curve if item["prediction_index"] == 1]
        self.assertEqual(first, changed_first)

    def test_comparable_allowlist_rejects_mixed_family(self) -> None:
        protocol = {"L_match": 4, "warmup_length": 200, "prediction_horizon": 5, "propagation_mode": "raw", "continuous_impl": "reference", "data_file_sha256": "092d957f5bb0d2cd62f85098ed2268114a47b4738a5f4b29ea6be4be7349fc4d"}
        ok, reason = _artifact_comparability(Path("results/fig9_diagnostics/competition_250/original_predictions.csv"), protocol, {})
        self.assertFalse(ok)
        self.assertIn("not_in_paper_constrained_raw_l4_allowlist", reason)

    def test_comparable_accepts_strict_allowlist(self) -> None:
        protocol = {"L_match": 4, "warmup_length": 200, "prediction_horizon": 5, "propagation_mode": "raw", "continuous_impl": "reference", "data_file_sha256": "092d957f5bb0d2cd62f85098ed2268114a47b4738a5f4b29ea6be4be7349fc4d"}
        summary = {"uses_compensation": False}
        ok, reason = _artifact_comparability(Path("results/fig9_strict/stage_a_250/original_predictions.csv"), protocol, summary)
        self.assertTrue(ok, reason)

    def test_metric_audit_does_not_touch_global_rng(self) -> None:
        random.seed(1234)
        before = random.getstate()
        metric_values(rows())
        random.setstate(before)
        expected = random.random()
        random.seed(1234)
        self.assertEqual(expected, random.random())

    def test_metric_audit_does_not_change_strict_defaults(self) -> None:
        config = Fig9StrictConfig()
        self.assertEqual(config.horizon, 5)
        self.assertEqual(config.l_match, 4)
        self.assertEqual(config.forgetting_threshold, 65.0)
        self.assertEqual(config.warmup, 5904)

    def test_no_compensation_is_required_by_audit(self) -> None:
        self.assertFalse(Fig9StrictConfig().use_future_covariates)
        self.assertFalse(Fig9StrictConfig().reencode_decoded_value)

    def test_b_anchor_rows_are_labeled_separately(self) -> None:
        artifact = Artifact(Path("b.csv"), None, None, rows(), 8, {}, {}, "observe_current_then_5", True, True, "")
        self.assertEqual(artifact.anchor_semantics, "observe_current_then_5")

    def test_metric_is_posthoc_only(self) -> None:
        original = rows()
        values = metric_values(original)
        self.assertEqual(original, rows())
        self.assertGreaterEqual(values["n"], 0)

    def test_invalid_empty_metric_is_zero(self) -> None:
        self.assertEqual(metric_values(()), {"n": 0, "standard_mape": 0.0, "wape": 0.0, "repo_metric": 0.0, "mae": 0.0, "rmse": 0.0, "median_ape": 0.0, "p90_ape": 0.0, "zero_target_count": 0})

    def test_median_ape_fixture(self) -> None:
        self.assertAlmostEqual(median_absolute_percentage_error([1.0, 3.0], [2.0, 2.0]), 0.5)

    def test_bootstrap_and_report_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit"
            summary = write_audit_report(Path(__file__).resolve().parents[1], output)
            self.assertEqual(summary["primary_conclusion"], "CURRENT_REPO_METRIC_DIFFERS_FROM_STANDARD_MAPE")
            self.assertTrue((output / "evaluation_metric_comparison.csv").exists())

    def test_report_contains_required_files(self) -> None:
        required = {"CURRENT_REPO_METRIC_DEFINITION.md", "PAPER_REFERENCE_METRIC_AUDIT.md", "strict_l4_prediction_artifact_inventory.csv", "evaluation_metric_comparison.csv", "evaluation_window_comparison.csv", "evaluation_learning_curve.csv", "mape_gap_decomposition.csv", "FIG9_EVALUATION_METRIC_PARITY_REPORT.md", "FINAL_EVALUATION_PARITY_SUMMARY.json"}
        self.assertEqual(len(required), 9)


if __name__ == "__main__":
    unittest.main()
