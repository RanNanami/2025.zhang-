from __future__ import annotations

import inspect
import pickle
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from experiments.diagnostics.fig9_anchor_parity_abcd import (
    EXPECTED_DATA_SHA256,
    HISTORICAL_MODEL_FINGERPRINT,
    HISTORICAL_PREDICTIONS,
    HISTORICAL_REPO_METRIC,
    HISTORICAL_RNG_FINGERPRINT,
    clone_model_from_blob,
    observe_current_then_predict,
    predict_without_target,
)
from experiments.fig9.anchor_parity import (
    CELLS,
    CELL_ORDER,
    anchor_level_rows,
    cell_timing,
    effect_classification,
    expected_interval,
    finalize_prediction_rows,
    local_change_analysis,
    pairwise_effects,
    pearson,
    quartile_rows,
    relative_improvement,
    score_prediction,
    select_conclusion,
    spearman,
    standard_mape,
    summarize_cells,
    time_of_day,
    timeline_steps,
    timeofday_rows,
)
from experiments.fig9.data import TaxiRecord
from experiments.fig9.metrics import mape as repository_mape
from experiments.fig9.parity import file_sha256
from experiments.fig9_strict_reproduction import Fig9StrictConfig
from seqmem.encoding import SSTDDiscreteEncoder
from seqmem.model import MemoryParams, SequentialMemory


ROOT = Path(__file__).resolve().parents[1]


class Fig9AnchorParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        start = datetime(2014, 7, 1, 0, 0)
        cls.records = [
            TaxiRecord(start + timedelta(minutes=30 * index), float(100 + index * 10))
            for index in range(20)
        ]

    def _rows(self) -> tuple[list[dict[str, object]], set[int]]:
        rows = []
        for anchor in (2, 3):
            for cell, prediction in zip(CELL_ORDER, (140.0, 150.0, 140.0, 160.0)):
                rows.append(
                    score_prediction(
                        records=self.records,
                        anchor_index=anchor,
                        cell=cell,
                        prediction=prediction + anchor,
                    )
                )
        return finalize_prediction_rows(rows)

    def test_01_cell_a_is_historical_anchor(self) -> None:
        self.assertFalse(CELLS["A"].observe_current)
        self.assertEqual(CELLS["A"].rollout_steps, 5)
        self.assertEqual(CELLS["A"].target_offset, 5)

    def test_02_cell_b_observes_current(self) -> None:
        self.assertTrue(CELLS["B"].observe_current)

    def test_03_cell_b_target_is_t_plus_5(self) -> None:
        self.assertEqual(CELLS["B"].target_offset, 5)

    def test_04_cell_b_last_observed_to_target_is_2_5_hours(self) -> None:
        timing = cell_timing(self.records, 2, "B")
        self.assertEqual(timing["last_observed_to_target_minutes"], 150)

    def test_05_cell_c_does_not_observe_current(self) -> None:
        self.assertFalse(CELLS["C"].observe_current)

    def test_06_cell_c_target_is_t_plus_4(self) -> None:
        self.assertEqual(CELLS["C"].target_offset, 4)

    def test_07_cell_c_last_observed_to_target_is_2_5_hours(self) -> None:
        timing = cell_timing(self.records, 2, "C")
        self.assertEqual(timing["last_observed_to_target_minutes"], 150)

    def test_08_cell_d_does_not_observe_current(self) -> None:
        self.assertFalse(CELLS["D"].observe_current)

    def test_09_cell_d_uses_six_steps(self) -> None:
        self.assertEqual(CELLS["D"].rollout_steps, 6)

    def test_10_cell_d_target_is_t_plus_5(self) -> None:
        self.assertEqual(CELLS["D"].target_offset, 5)

    def test_11_cell_d_endpoint_matches_target(self) -> None:
        self.assertEqual(cell_timing(self.records, 2, "D")["causal_offset_minutes"], 0)

    def test_12_cell_a_endpoint_does_not_match_target(self) -> None:
        self.assertEqual(cell_timing(self.records, 2, "A")["causal_offset_minutes"], 30)

    def test_13_cell_b_has_no_off_by_one(self) -> None:
        self.assertEqual(CELLS["B"].causal_endpoint_offset, CELLS["B"].target_offset)

    def test_14_cell_c_has_no_off_by_one(self) -> None:
        self.assertEqual(CELLS["C"].causal_endpoint_offset, CELLS["C"].target_offset)

    def test_15_cell_d_has_no_off_by_one(self) -> None:
        self.assertEqual(CELLS["D"].causal_endpoint_offset, CELLS["D"].target_offset)

    def test_16_predict_api_cannot_receive_future_ground_truth(self) -> None:
        self.assertEqual(
            list(inspect.signature(predict_without_target).parameters),
            ["model", "rollout_steps"],
        )

    def test_17_observe_then_predict_orders_calls(self) -> None:
        calls: list[str] = []
        with (
            patch(
                "experiments.diagnostics.fig9_anchor_parity_abcd.learn_actual_code",
                side_effect=lambda *_args: calls.append("observe"),
            ),
            patch(
                "experiments.diagnostics.fig9_anchor_parity_abcd.predict_without_target",
                side_effect=lambda *_args: (calls.append("rollout") or (1.0, (), ())),
            ),
        ):
            result = observe_current_then_predict(object(), object(), 5)  # type: ignore[arg-type]
        self.assertEqual(calls, ["observe", "rollout"])
        self.assertEqual(result, (1.0, (), ()))

    def test_18_score_api_receives_prediction_before_ground_truth(self) -> None:
        parameters = inspect.signature(score_prediction).parameters
        self.assertIn("prediction", parameters)
        self.assertNotIn("ground_truth", parameters)

    def test_19_all_cells_share_prestate_fields(self) -> None:
        rows = [
            score_prediction(
                records=self.records,
                anchor_index=2,
                cell=cell,
                prediction=1.0,
                pre_model_fingerprint="model",
                pre_rng_fingerprint="rng",
                pre_transient_fingerprint="state",
            )
            for cell in CELL_ORDER
        ]
        self.assertEqual({row["pre_model_fingerprint"] for row in rows}, {"model"})
        self.assertEqual({row["pre_rng_fingerprint"] for row in rows}, {"rng"})
        self.assertEqual({row["pre_transient_fingerprint"] for row in rows}, {"state"})

    def test_20_model_clone_preserves_rng_and_state(self) -> None:
        encoder = SSTDDiscreteEncoder(num_columns=4, k=1, seed=2)
        model = SequentialMemory(
            encoder=encoder,
            num_neurons_per_column=2,
            params=MemoryParams(l_match=1),
            tie_break_seed=7,
        )
        model.observe("A", learn=True)
        clone = clone_model_from_blob(pickle.dumps(model), model)
        self.assertEqual(pickle.dumps(clone._decode_rng.getstate()), pickle.dumps(model._decode_rng.getstate()))
        self.assertEqual(clone.previous_active_cells, model.previous_active_cells)
        self.assertEqual(clone.previous_winners, model.previous_winners)

    def test_21_standard_mape_fixture(self) -> None:
        self.assertAlmostEqual(standard_mape([9.0, 12.0], [10.0, 10.0]), 0.15)

    def test_22_repository_metric_fixture(self) -> None:
        self.assertAlmostEqual(repository_mape([0.0, 15.0], [10.0, 20.0]), 0.5)

    def test_23_zero_target_handling_differs_explicitly(self) -> None:
        self.assertEqual(standard_mape([3.0, 8.0], [0.0, 10.0]), 0.2)
        self.assertEqual(repository_mape([3.0, 8.0], [0.0, 10.0]), 0.5)

    def test_24_local_target_change_is_posthoc(self) -> None:
        rows, common = self._rows()
        anchors = anchor_level_rows(rows, self.records, common)
        self.assertEqual(
            anchors[0]["local_target_change"],
            abs(self.records[7].value - self.records[6].value),
        )

    def test_25_a_and_c_can_share_exact_prediction(self) -> None:
        prediction = 123.0
        a = score_prediction(records=self.records, anchor_index=2, cell="A", prediction=prediction)
        c = score_prediction(records=self.records, anchor_index=2, cell="C", prediction=prediction)
        self.assertEqual(a["prediction"], c["prediction"])
        self.assertNotEqual(a["ground_truth"], c["ground_truth"])

    def test_26_strict_default_is_unchanged(self) -> None:
        self.assertEqual(Fig9StrictConfig().horizon, 5)

    def test_27_lmatch_remains_four(self) -> None:
        self.assertEqual(Fig9StrictConfig().l_match, 4)

    def test_28_forgetting_remains_65(self) -> None:
        self.assertEqual(Fig9StrictConfig().forgetting_threshold, 65.0)

    def test_29_no_compensation_or_future_context(self) -> None:
        config = Fig9StrictConfig()
        self.assertEqual(config.propagation_mode, "raw")
        self.assertFalse(config.use_future_covariates)
        self.assertFalse(config.reencode_decoded_value)

    def test_30_historical_asset_and_constants_exist(self) -> None:
        self.assertTrue(HISTORICAL_PREDICTIONS.exists())
        self.assertEqual(HISTORICAL_REPO_METRIC, 0.5054500721931258)
        self.assertEqual(len(HISTORICAL_MODEL_FINGERPRINT), 64)
        self.assertEqual(len(HISTORICAL_RNG_FINGERPRINT), 64)

    def test_31_current_data_hash_is_fixed(self) -> None:
        self.assertEqual(
            file_sha256(ROOT / "data/paper_nyc_taxi.csv"), EXPECTED_DATA_SHA256
        )

    def test_32_common_anchor_set_requires_all_cells(self) -> None:
        rows, common = self._rows()
        self.assertEqual(common, {2, 3})
        rows[0]["prediction"] = ""
        _, reduced = finalize_prediction_rows(rows)
        self.assertEqual(reduced, {3})

    def test_33_summary_uses_common_anchor_set(self) -> None:
        rows, common = self._rows()
        summaries = summarize_cells(rows, common)
        self.assertEqual({row["metrics_anchor_count"] for row in summaries}, {2})

    def test_34_pairwise_table_has_six_rows(self) -> None:
        rows, common = self._rows()
        summaries = summarize_cells(rows, common)
        self.assertEqual(len(pairwise_effects(rows, summaries, common)), 6)

    def test_35_relative_improvement_fixture(self) -> None:
        self.assertAlmostEqual(relative_improvement(0.5, 0.4), 0.2)

    def test_36_effect_classification_boundaries(self) -> None:
        self.assertEqual(effect_classification(0.40), "VERY_LARGE")
        self.assertEqual(effect_classification(0.20), "LARGE")
        self.assertEqual(effect_classification(0.10), "MODERATE")
        self.assertEqual(effect_classification(0.02), "SMALL")
        self.assertEqual(effect_classification(0.019), "NEGLIGIBLE")

    def test_37_correlation_fixtures(self) -> None:
        self.assertAlmostEqual(pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [9, 8, 7]), -1.0)

    def test_38_quartiles_cover_every_anchor(self) -> None:
        rows, common = self._rows()
        anchors = anchor_level_rows(rows, self.records, common)
        quartiles = quartile_rows(anchors)
        self.assertEqual(sum(int(row["anchors"]) for row in quartiles), len(anchors))
        self.assertEqual(len(quartiles), 4)

    def test_39_time_of_day_fixed_boundaries(self) -> None:
        base = datetime(2014, 1, 1)
        self.assertEqual(time_of_day(base.replace(hour=5)), "night")
        self.assertEqual(time_of_day(base.replace(hour=6)), "morning_rise")
        self.assertEqual(time_of_day(base.replace(hour=10)), "midday")
        self.assertEqual(time_of_day(base.replace(hour=17)), "evening")

    def test_40_timeofday_rows_always_use_fixed_four_bands(self) -> None:
        rows, common = self._rows()
        anchors = anchor_level_rows(rows, self.records, common)
        self.assertEqual(len(timeofday_rows(anchors)), 4)

    def test_41_timeline_step_counts(self) -> None:
        self.assertEqual(len(timeline_steps(self.records, 2, "A")), 5)
        self.assertEqual(len(timeline_steps(self.records, 2, "B")), 5)
        self.assertEqual(len(timeline_steps(self.records, 2, "D")), 6)

    def test_42_sampling_interval_is_30_minutes(self) -> None:
        self.assertEqual(expected_interval(), timedelta(minutes=30))

    def test_43_missing_prediction_has_blank_error(self) -> None:
        row = score_prediction(
            records=self.records, anchor_index=2, cell="A", prediction=None
        )
        self.assertEqual(row["prediction"], "")
        self.assertEqual(row["abs_error"], "")
        self.assertEqual(row["standard_APE"], "")

    def test_44_repository_contributions_sum_to_metric(self) -> None:
        rows, common = self._rows()
        summaries = summarize_cells(rows, common)
        for cell in CELL_ORDER:
            contributions = sum(
                float(row["current_repo_metric_contribution"])
                for row in rows
                if row["cell"] == cell and row["in_common_anchor_set"]
            )
            metric = next(row["repo_metric"] for row in summaries if row["cell"] == cell)
            self.assertAlmostEqual(contributions, float(metric))

    def test_45_local_change_analysis_is_deterministic(self) -> None:
        rows, common = self._rows()
        anchors = anchor_level_rows(rows, self.records, common)
        self.assertEqual(local_change_analysis(anchors), local_change_analysis(anchors))

    def test_46_inconclusive_when_prerequisites_fail(self) -> None:
        rows, common = self._rows()
        summaries = summarize_cells(rows, common)
        self.assertEqual(
            select_conclusion(summaries, prerequisites_met=False),
            ("ANCHOR_ABCD_RESULT_INCONCLUSIVE", "DO_NOT_CHANGE_PROTOCOL_YET"),
        )

    def test_47_b_target_offset_from_label_is_150_minutes(self) -> None:
        self.assertEqual(
            cell_timing(self.records, 2, "B")["target_offset_from_label_minutes"],
            150,
        )

    def test_48_c_target_offset_from_label_is_120_minutes(self) -> None:
        self.assertEqual(
            cell_timing(self.records, 2, "C")["target_offset_from_label_minutes"],
            120,
        )

    def test_49_small_best_effect_is_not_promoted_to_primary_cause(self) -> None:
        summaries = [
            {"cell": "A", "relative_improvement_vs_A": 0.0},
            {"cell": "B", "relative_improvement_vs_A": 0.0588},
            {"cell": "C", "relative_improvement_vs_A": -0.05},
            {"cell": "D", "relative_improvement_vs_A": 0.029},
        ]
        self.assertEqual(
            select_conclusion(summaries, prerequisites_met=True),
            (
                "ANCHOR_MISMATCH_HAS_SMALL_NUMERICAL_EFFECT",
                "AUDIT_EVALUATION_WINDOW_AND_MAPE",
            ),
        )


if __name__ == "__main__":
    unittest.main()
