from __future__ import annotations

import csv
import hashlib
import pickle
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from experiments.diagnostics.analyze_fig9_candidate_score_separability import (
    aggregate_columns,
    analyze,
    average_precision,
    bootstrap_by_rollout,
    bootstrap_rollout_metric_means,
    compare_policy_candidate_pools,
    fixed_budget_recall,
    roc_auc,
    threshold_sweep,
    validate_protocol,
)
from experiments.diagnostics.fig9_candidate_score_trace import (
    field_for_column,
    validate_trace_schema,
)
from experiments.diagnostics.fig9_competitive_inhibition import (
    CompetitionSettings,
)
from experiments.diagnostics.fig9_oracle_candidate import FieldColumnRanges
from experiments.fig9 import TaxiRecord
from experiments.fig9_strict_reproduction import (
    Fig9StrictConfig,
    model_long_term_fingerprint,
    model_rng_fingerprint,
    run_strict_stream,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Fig9CandidateSeparabilityMetricTests(unittest.TestCase):
    def test_auc_known_samples(self) -> None:
        self.assertEqual(roc_auc([False, True], [0.0, 1.0]), 1.0)
        self.assertEqual(roc_auc([True, False], [0.0, 1.0]), 0.0)

    def test_average_precision_known_sample(self) -> None:
        value = average_precision(
            [True, False, True],
            [3.0, 2.0, 1.0],
        )
        self.assertAlmostEqual(value or 0.0, (1.0 + 2.0 / 3.0) / 2.0)

    def test_all_tied_scores_reduce_to_prevalence(self) -> None:
        labels = [True, False, False, True]
        scores = [1.0] * 4

        self.assertEqual(roc_auc(labels, scores), 0.5)
        self.assertEqual(average_precision(labels, scores), 0.5)

    def test_single_class_metrics_are_defined_as_unavailable(self) -> None:
        self.assertIsNone(roc_auc([True, True], [1.0, 2.0]))
        self.assertIsNone(average_precision([False, False], [1.0, 2.0]))

    def test_duplicate_column_uses_max_score_without_averaging(self) -> None:
        rows = [
            self._candidate(column=7, neuron=0, score=1.1, order=0),
            self._candidate(column=7, neuron=1, score=1.4, order=1),
        ]

        columns = aggregate_columns(rows)

        self.assertEqual(len(columns), 1)
        self.assertEqual(columns[0]["max_original_score"], 1.4)
        self.assertEqual(columns[0]["selected_neuron"], 1)
        self.assertEqual(columns[0]["candidate_count_in_column"], 2)

    def test_protocol_ranges_and_field_mapping(self) -> None:
        ranges = validate_protocol(
            {
                "encoder_sizes": {
                    "weekday": 30,
                    "time": 58,
                    "passenger": 482,
                    "total": 570,
                }
            }
        )
        field_ranges = FieldColumnRanges.from_sizes(30, 58, 482)

        self.assertEqual(ranges["passenger"], (88, 570))
        self.assertEqual(field_for_column(0, field_ranges), "weekday")
        self.assertEqual(field_for_column(30, field_ranges), "time")
        self.assertEqual(field_for_column(88, field_ranges), "passenger")
        with self.assertRaisesRegex(ValueError, "outside protocol"):
            field_for_column(570, field_ranges)

    def test_protocol_rejects_invalid_total(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not sum"):
            validate_protocol(
                {
                    "encoder_sizes": {
                        "weekday": 30,
                        "time": 58,
                        "passenger": 482,
                        "total": 571,
                    }
                }
            )

    def test_threshold_sweep_is_monotonic(self) -> None:
        rows = [
            self._column(1, 3.0, True),
            self._column(2, 2.0, False),
            self._column(3, 1.0, True),
        ]

        sweep = threshold_sweep(rows, "max_original_score")

        recalls = [float(row["target_recall"]) for row in sweep]
        retained = [int(row["total_columns_retained"]) for row in sweep]
        self.assertEqual(recalls, sorted(recalls))
        self.assertEqual(retained, sorted(retained))

    def test_threshold_sweep_keeps_tied_scores_together(self) -> None:
        rows = [
            self._column(1, 2.0, True),
            self._column(2, 2.0, False),
            self._column(3, 1.0, True),
        ]

        sweep = threshold_sweep(rows, "max_original_score")

        self.assertEqual(len(sweep), 2)
        self.assertEqual(sweep[0]["threshold"], 2.0)
        self.assertEqual(sweep[0]["true_target_columns_retained"], 1)
        self.assertEqual(sweep[0]["false_columns_retained"], 1)
        self.assertEqual(sweep[0]["total_columns_retained"], 2)

    def test_fixed_budget_recall(self) -> None:
        rows = [
            self._column(1, 3.0, True),
            self._column(2, 2.0, False),
            self._column(3, 1.0, True),
        ]

        self.assertEqual(
            fixed_budget_recall(rows, "max_original_score", 1),
            0.5,
        )
        self.assertEqual(
            fixed_budget_recall(rows, "max_original_score", 3),
            1.0,
        )

    def test_bootstrap_is_repeatable_and_uses_whole_rollouts(self) -> None:
        rows = [
            {"input_index": 1},
            {"input_index": 1},
            {"input_index": 2},
            {"input_index": 2},
            {"input_index": 2},
        ]
        first = bootstrap_by_rollout(
            rows,
            samples=20,
            seed=7,
            metric=lambda sampled: float(len(sampled)),
        )
        second = bootstrap_by_rollout(
            rows,
            samples=20,
            seed=7,
            metric=lambda sampled: float(len(sampled)),
        )

        self.assertEqual(first, second)
        self.assertTrue(set(first).issubset({4.0, 5.0, 6.0}))

        rollout_means = bootstrap_rollout_metric_means(
            rows,
            samples=20,
            seed=7,
            metric=lambda sampled: float(len(sampled)),
        )
        repeated_means = bootstrap_rollout_metric_means(
            rows,
            samples=20,
            seed=7,
            metric=lambda sampled: float(len(sampled)),
        )
        self.assertEqual(rollout_means, repeated_means)
        self.assertTrue(set(rollout_means).issubset({2.0, 2.5, 3.0}))

    def test_step_one_alignment_and_later_divergence_marker(self) -> None:
        sequential = [
            self._candidate(step=1, column=1),
            self._candidate(step=2, column=2),
        ]
        batched = [
            self._candidate(step=1, column=1),
            self._candidate(step=2, column=3),
        ]

        comparison = compare_policy_candidate_pools(sequential, batched)

        self.assertTrue(
            comparison[0]["direct_candidate_pool_comparison_allowed"]
        )
        self.assertEqual(comparison[0]["matching_pool_rate"], 1.0)
        self.assertTrue(comparison[1]["trajectory_divergence"])
        self.assertFalse(
            comparison[1]["direct_candidate_pool_comparison_allowed"]
        )

    def test_empty_candidates_are_safe(self) -> None:
        self.assertEqual(aggregate_columns([]), [])
        self.assertEqual(threshold_sweep([], "max_original_score"), [])
        self.assertEqual(
            fixed_budget_recall([], "max_original_score", 10),
            0.0,
        )

    def test_missing_csv_schema_is_explicit(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "candidate trace is missing required fields",
        ):
            validate_trace_schema(["column", "original_score"])

    def _candidate(
        self,
        *,
        step: int = 1,
        column: int = 1,
        neuron: int = 0,
        score: float = 1.0,
        order: int = 0,
    ) -> dict[str, object]:
        return {
            "policy": "sequential",
            "input_index": 1,
            "target_timestamp": f"2015-01-01 0{step}:00:00",
            "horizon_step": step,
            "candidate_original_index": order,
            "field": "weekday",
            "column": column,
            "neuron": neuron,
            "predicted_time": 0.01,
            "batch_index": 2,
            "original_score": score,
            "accumulated_inhibition": 0.0,
            "effective_score": score,
            "emitted": True,
            "is_target_candidate": True,
        }

    def _column(
        self,
        column: int,
        score: float,
        target: bool,
    ) -> dict[str, object]:
        return {
            "column": column,
            "max_original_score": score,
            "is_target_column": target,
        }


class Fig9CandidateSeparabilityTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.data_path = cls.root / "tiny.csv"
        cls.data_path.write_text(
            "timestamp,passenger_count\n",
            encoding="utf-8",
        )
        start = datetime(2014, 7, 1)
        records = [
            TaxiRecord(
                start + timedelta(minutes=30 * index),
                1000.0 + index * 20.0,
            )
            for index in range(18)
        ]
        cls.results: dict[tuple[str, str], dict[str, object]] = {}
        for policy in ("sequential", "batched"):
            settings = CompetitionSettings(
                mode="competitive_raw",
                inhibition_strength=0.1,
                simultaneous_policy=policy,
            )
            for trace_label, enabled in (("off", False), ("on", True)):
                directory = cls.root / policy / trace_label
                checkpoint = directory / "checkpoint.pkl"
                cls.results[(policy, trace_label)] = run_strict_stream(
                    records=records,
                    data_path=cls.data_path,
                    stream_label="original",
                    output_dir=directory,
                    config=Fig9StrictConfig(warmup=6),
                    limit=0,
                    print_fingerprint=False,
                    competition_settings=settings,
                    oracle_candidate_diagnostic=True,
                    candidate_separability_trace=enabled,
                    checkpoint_path=checkpoint,
                    checkpoint_at_index=10,
                )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_trace_switch_preserves_predictions_and_fingerprints(self) -> None:
        for policy in ("sequential", "batched"):
            off = self.root / policy / "off"
            on = self.root / policy / "on"
            self.assertEqual(
                _sha256(off / "original_predictions.csv"),
                _sha256(on / "original_predictions.csv"),
            )
            for field in (
                "final_model_fingerprint",
                "final_rng_fingerprint",
            ):
                self.assertEqual(
                    self.results[(policy, "off")][field],
                    self.results[(policy, "on")][field],
                )

    def test_trace_switch_preserves_stable_competition_trace(self) -> None:
        runtime_fields = {
            "prediction_runtime_seconds",
            "decode_runtime_seconds",
            "competition_runtime_seconds",
            "observe_runtime",
        }
        for policy in ("sequential", "batched"):
            stable = []
            for label in ("off", "on"):
                path = (
                    self.root / policy / label / "competition_trace.csv"
                )
                with path.open("r", encoding="utf-8", newline="") as handle:
                    stable.append(
                        [
                            {
                                key: value
                                for key, value in row.items()
                                if key not in runtime_fields
                            }
                            for row in csv.DictReader(handle)
                        ]
                    )
            self.assertEqual(stable[0], stable[1])

    def test_trace_switch_preserves_checkpoint_v1(self) -> None:
        for policy in ("sequential", "batched"):
            payloads = []
            for label in ("off", "on"):
                with (
                    self.root / policy / label / "checkpoint.pkl"
                ).open("rb") as handle:
                    payloads.append(pickle.load(handle))
            self.assertEqual(
                payloads[0]["checkpoint_format"],
                "fig9-strict-v1",
            )
            self.assertEqual(set(payloads[0]), set(payloads[1]))
            self.assertEqual(
                model_long_term_fingerprint(payloads[0]["model"]),
                model_long_term_fingerprint(payloads[1]["model"]),
            )
            self.assertEqual(
                model_rng_fingerprint(payloads[0]["model"]),
                model_rng_fingerprint(payloads[1]["model"]),
            )

    def test_trace_has_analysis_only_markers_and_candidate_fields(self) -> None:
        for policy in ("sequential", "batched"):
            path = (
                self.root
                / policy
                / "on"
                / "candidate_separability_trace.csv"
            )
            with path.open("r", encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["diagnostic_only"], "True")
            self.assertEqual(row["offline_analysis_only"], "True")
            self.assertEqual(
                row["ground_truth_does_not_affect_prediction"],
                "True",
            )
            self.assertEqual(row["policy"], policy)

    def test_trace_requires_oracle_and_competition(self) -> None:
        records = [
            TaxiRecord(datetime(2014, 7, 1), 1000.0),
            TaxiRecord(datetime(2014, 7, 1, 0, 30), 1010.0),
            TaxiRecord(datetime(2014, 7, 1, 1, 0), 1020.0),
            TaxiRecord(datetime(2014, 7, 1, 1, 30), 1030.0),
            TaxiRecord(datetime(2014, 7, 1, 2, 0), 1040.0),
            TaxiRecord(datetime(2014, 7, 1, 2, 30), 1050.0),
        ]
        with self.assertRaisesRegex(
            ValueError,
            "requires oracle candidate diagnostics",
        ):
            run_strict_stream(
                records=records,
                data_path=self.data_path,
                stream_label="original",
                output_dir=self.root / "invalid",
                config=Fig9StrictConfig(warmup=1),
                limit=0,
                print_fingerprint=False,
                competition_settings=CompetitionSettings(
                    mode="competitive_raw",
                ),
                candidate_separability_trace=True,
            )

    def test_end_to_end_analysis_writes_all_required_outputs(self) -> None:
        output = self.root / "analysis"
        summary = analyze(
            (
                ("sequential", self.root / "sequential" / "on"),
                ("batched", self.root / "batched" / "on"),
            ),
            output_dir=output,
            bootstrap_samples=5,
            bootstrap_seed=0,
        )
        expected = {
            "candidate_score_rows.csv",
            "column_score_rows.csv",
            "score_distribution_summary.csv",
            "separability_metrics.csv",
            "threshold_curves.csv",
            "recall_budget_summary.csv",
            "batch_separability_summary.csv",
            "field_scale_summary.csv",
            "horizon_degradation_summary.csv",
            "bootstrap_confidence_intervals.csv",
            "candidate_score_separability_summary.json",
            "CANDIDATE_SCORE_SEPARABILITY_REPORT.md",
        }

        self.assertEqual(
            expected,
            {path.name for path in output.iterdir()},
        )
        self.assertTrue(summary["offline_analysis_only"])
        self.assertEqual(summary["bootstrap_unit"], "rollout input_index")


if __name__ == "__main__":
    unittest.main()
