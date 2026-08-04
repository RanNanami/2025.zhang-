from __future__ import annotations

import csv
import contextlib
import io
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from experiments.diagnostics.analyze_fig9_lmatch_ablation import (
    analyze,
    classify_result,
    paired_bootstrap,
    validate_protocols,
)
from experiments.fig9 import TaxiRecord
from experiments.fig9_strict_reproduction import (
    Fig9StrictConfig,
    build_fig9_encoder,
    build_strict_model,
    load_strict_checkpoint,
    parse_args,
    protocol_fingerprint,
    run_strict_stream,
    validate_checkpoint_l_match,
)
from seqmem.encoding import SSTDDiscreteEncoder, SpikeEvent, SymbolCode
from seqmem.model import MemoryParams, Segment, SequentialMemory, Synapse


class Fig9LMatchConfigurationTests(unittest.TestCase):
    def test_cli_accepts_only_real_ablation_values(self) -> None:
        for value in (2, 3, 4):
            with patch.object(sys, "argv", ["fig9", "--l-match", str(value)]):
                self.assertEqual(parse_args().l_match, value)
        for value in ("0", "1", "-2", "bad"):
            with self.subTest(value=value), patch.object(
                sys, "argv", ["fig9", "--l-match", value]
            ), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args()

    def test_strict_default_remains_four(self) -> None:
        with patch.object(sys, "argv", ["fig9"]):
            self.assertEqual(parse_args().l_match, 4)
        self.assertEqual(Fig9StrictConfig().l_match, 4)
        encoder = build_fig9_encoder(Fig9StrictConfig())
        self.assertEqual(build_strict_model(encoder, Fig9StrictConfig()).params.l_match, 4)

    def test_real_lmatch_reaches_model_params(self) -> None:
        for value in (2, 3, 4):
            config = Fig9StrictConfig(l_match=value)
            model = build_strict_model(build_fig9_encoder(config), config)
            self.assertEqual(model.params.l_match, value)

    def test_real_lmatch_changes_scenario2_gate(self) -> None:
        code = SymbolCode((SpikeEvent(0, 0.0),))
        outcomes = {}
        for value in (2, 4):
            model = SequentialMemory(
                encoder=SSTDDiscreteEncoder(num_columns=1, k=1),
                num_neurons_per_column=2,
                params=MemoryParams(l_match=value),
                tie_break_seed=0,
            )
            source = 1
            segment = Segment(
                synapses={source: Synapse(source, 0.5, 0.5, 0)},
                target_time=1.0,
            )
            model.columns[0].neurons[0].segments.append(segment)
            model.previous_active_cells = {source: 0.0}
            model.previous_winners = {source: 0.0}
            with patch.object(
                model, "_best_matching_neuron", return_value=(0, segment)
            ), patch.object(Segment, "timed_overlap", return_value=2):
                model.observe_code(code, learn=True)
            outcomes[value] = dict(model.last_observe_stats)
        self.assertEqual(outcomes[2]["scenario2"], 1)
        self.assertEqual(outcomes[2]["scenario3"], 0)
        self.assertEqual(outcomes[4]["scenario2"], 0)
        self.assertEqual(outcomes[4]["scenario3"], 1)

    def test_protocol_records_real_lmatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "taxi.csv"
            path.write_text(
                "timestamp,passenger_count\n2014-07-01 00:00:00,10\n",
                encoding="utf-8",
            )
            records = [TaxiRecord(datetime(2014, 7, 1), 10.0)]
            protocol = protocol_fingerprint(
                data_path=path,
                records=records,
                stream_label="original",
                config=Fig9StrictConfig(l_match=2),
                limit=1,
            )
        self.assertEqual(protocol["L_match"], 2)
        self.assertFalse(protocol["uses_future_covariates"])
        self.assertFalse(protocol["learns_during_rollout"])

    def test_checkpoint_records_and_rejects_mismatched_lmatch(self) -> None:
        records = [
            TaxiRecord(datetime(2014, 7, 1) + timedelta(minutes=30 * i), 10.0 + i)
            for i in range(14)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "taxi.csv"
            data_path.write_text("timestamp,passenger_count\n", encoding="utf-8")
            checkpoint = root / "state.pkl"
            run_strict_stream(
                records=records,
                data_path=data_path,
                stream_label="original",
                output_dir=root / "run",
                config=Fig9StrictConfig(warmup=20, l_match=2),
                limit=0,
                print_fingerprint=False,
                checkpoint_path=checkpoint,
                checkpoint_at_index=7,
                stop_after_index=7,
                lmatch_real_ablation=True,
            )
            payload = load_strict_checkpoint(checkpoint)
            self.assertEqual(payload["L_match"], 2)
            validate_checkpoint_l_match(payload, Fig9StrictConfig(l_match=2))
            with self.assertRaisesRegex(ValueError, "L_match mismatch"):
                validate_checkpoint_l_match(payload, Fig9StrictConfig(l_match=3))

    def test_default_and_explicit_l4_predictions_are_identical(self) -> None:
        records = [
            TaxiRecord(datetime(2014, 7, 1) + timedelta(minutes=30 * i), 20.0 + i)
            for i in range(18)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "taxi.csv"
            data_path.write_text("timestamp,passenger_count\n", encoding="utf-8")
            outputs = []
            for label, config in (
                ("default", Fig9StrictConfig(warmup=8)),
                ("explicit", Fig9StrictConfig(warmup=8, l_match=4)),
            ):
                outputs.append(
                    run_strict_stream(
                        records=records,
                        data_path=data_path,
                        stream_label="original",
                        output_dir=root / label,
                        config=config,
                        limit=0,
                        print_fingerprint=False,
                    )
                )
            first = (root / "default" / "original_predictions.csv").read_bytes()
            second = (root / "explicit" / "original_predictions.csv").read_bytes()
        self.assertEqual(first, second)
        self.assertEqual(outputs[0]["final_model_fingerprint"], outputs[1]["final_model_fingerprint"])
        self.assertEqual(outputs[0]["final_rng_fingerprint"], outputs[1]["final_rng_fingerprint"])

    def test_manual_script_excludes_lmatch_one_and_requires_opt_in(self) -> None:
        script = Path("scripts/fig9_lmatch_ablation_250.ps1").read_text(encoding="utf-8")
        self.assertIn("ValidateSet(4, 3, 2)", script)
        self.assertNotIn("ValidateSet(4, 3, 2, 1)", script)
        self.assertIn("RunFormal250", script)
        self.assertIn("--lmatch-real-ablation", script)


class Fig9LMatchAnalysisTests(unittest.TestCase):
    def _write_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _fixture(self, root: Path, l_match: int) -> Path:
        run = root / f"l{l_match}"
        run.mkdir()
        protocol = {
            "git_commit_sha": "same",
            "data_file_path": "data.csv",
            "data_file_sha256": "same-data",
            "L_match": l_match,
            "encoder_sizes": {"weekday": 1, "time": 1, "passenger": 1, "total": 3},
            "neurons_per_column": 2,
            "diagnostic_only": True,
            "lmatch_real_ablation": True,
            "strict_default_unchanged": True,
            "uses_ground_truth_for_selection": False,
            "uses_future_covariates": False,
            "uses_compensation": False,
        }
        (run / "original_protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
        (run / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
        summary = {
            "mape": 0.1 * l_match,
            "final_rolling_mape": 0.2,
            "coverage": 1.0,
            "attempted_predictions": 2,
            "runtime_seconds": 1.0,
            "mean_raw_column_count": 2.0,
            "peak_raw_column_count": 3,
            "final_segment_count": 6,
            "final_model_fingerprint": f"model-{l_match}",
            "final_rng_fingerprint": f"rng-{l_match}",
        }
        (run / "original_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (run / "runtime.json").write_text(json.dumps({"L_match": l_match}), encoding="utf-8")
        self._write_csv(
            run / "original_predictions.csv",
            [
                {"input_index": index, "target": 10, "prediction": 10 - l_match, "absolute_error": l_match, "normalized_absolute_error": l_match / 10}
                for index in (1, 2)
            ],
        )
        self._write_csv(
            run / "original_density_trace.csv",
            [
                {
                    "horizon_step": step,
                    "prediction_missing": False,
                    "absolute_error": l_match,
                    "actual_future_passenger": 10,
                    "absolute_percentage_error": l_match / 10,
                    "raw_predicted_column_count": 3,
                    "emitted_column_count": 2,
                    "weekday_emitted_column_count": 1,
                    "time_emitted_column_count": 1,
                    "passenger_emitted_column_count": 0,
                    "decoded_passenger": 10 - l_match,
                }
                for step in range(1, 6)
            ],
        )
        self._write_csv(
            run / "observe_scenario_trace.csv",
            [
                {"actual_record_index": 200, "field": "passenger", "observe_scenario": "scenario2", "created_segment_id": "", "L_match": l_match},
                {"actual_record_index": 201, "field": "passenger", "observe_scenario": "scenario3", "created_segment_id": f"new-{l_match}", "L_match": l_match},
            ],
        )
        self._write_csv(
            run / "match_overlap_segment_trace.csv",
            [
                {
                    "actual_record_index": 200,
                    "field": "passenger",
                    "encoded_column": 2,
                    "segment_provenance_id": f"segment-{l_match}",
                    "segment_target_neuron": 0,
                    "eligible": True,
                    "actual_overlap": l_match,
                    "selected_as_best_matching": True,
                    "selected_for_scenario": True,
                    "reinforced": True,
                    "segment_deleted_before_end": False,
                    "segment_age": 1,
                    "segment_weight_sum": 2,
                }
            ],
        )
        self._write_csv(
            run / "context_trajectory_column_trace.csv",
            [
                {
                    "trajectory_kind": "actual_observation",
                    "present_in_next_context": True,
                    "most_recent_loss_stage": "",
                    "recovery_count": 0,
                },
                {
                    "trajectory_kind": "autonomous_rollout",
                    "present_in_next_context": False,
                    "most_recent_loss_stage": "LOST_INTERCOLUMN",
                    "recovery_count": 1,
                },
            ],
        )
        return run

    def test_protocol_comparison_allows_only_lmatch_difference(self) -> None:
        protocols = {
            value: {**{key: expected for key, expected in {
                "diagnostic_only": True,
                "lmatch_real_ablation": True,
                "strict_default_unchanged": True,
                "uses_ground_truth_for_selection": False,
                "uses_future_covariates": False,
                "uses_compensation": False,
            }.items()}, "git_commit_sha": "same", "L_match": value}
            for value in (4, 3, 2)
        }
        self.assertTrue(validate_protocols(protocols)["comparison_valid"])
        protocols[2]["git_commit_sha"] = "different"
        checks = validate_protocols(protocols)
        self.assertTrue(checks["comparison_valid"])
        self.assertEqual(
            checks["provenance_differences"]["git_commit_sha"]["2"],
            "different",
        )

    def test_paired_bootstrap_is_reproducible(self) -> None:
        left = {1: (1.0, 10.0), 2: (2.0, 20.0)}
        right = {1: (2.0, 10.0), 2: (3.0, 20.0)}
        first = paired_bootstrap(left, right, samples=100, seed=7)
        second = paired_bootstrap(left, right, samples=100, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["paired_records"], 2)

    def test_improved_error_with_ambiguity_growth_is_unstable(self) -> None:
        summaries = [
            {"L_match": 4, "MAPE": 0.41, "coverage": 1.0, "final_segment_count": 2300},
            {"L_match": 3, "MAPE": 0.44, "coverage": 1.0, "final_segment_count": 2200},
            {"L_match": 2, "MAPE": 0.39, "coverage": 1.0, "final_segment_count": 1950},
        ]
        horizons = [
            {"L_match": level, "horizon_step": step, "MAPE": value}
            for level, values in {
                4: (0.33, 0.44, 0.46, 0.42, 0.41),
                3: (0.35, 0.48, 0.32, 0.41, 0.44),
                2: (0.40, 0.34, 0.39, 0.40, 0.39),
            }.items()
            for step, value in enumerate(values, 1)
        ]
        scenarios = [
            {"L_match": level, "scenario": scenario, "rate": rate}
            for level, values in {
                4: (0.14, 0.60, 0.26),
                3: (0.17, 0.60, 0.23),
                2: (0.13, 0.69, 0.18),
            }.items()
            for scenario, rate in zip(("scenario1", "scenario2", "scenario3"), values)
        ]
        density = [
            {"L_match": 4, "mean_emitted_columns": 124},
            {"L_match": 3, "mean_emitted_columns": 115},
            {"L_match": 2, "mean_emitted_columns": 92},
        ]
        ambiguity = [
            {"L_match": 4, "multiple_neuron_ambiguity_rate": 0.155},
            {"L_match": 3, "multiple_neuron_ambiguity_rate": 0.203},
            {"L_match": 2, "multiple_neuron_ambiguity_rate": 0.285},
        ]
        bootstrap = [
            {"comparison": "L4_vs_L3", "ci95_high": 0.16},
            {"comparison": "L4_vs_L2", "ci95_high": 0.11},
        ]
        self.assertEqual(
            classify_result(summaries, horizons, scenarios, density, ambiguity, bootstrap),
            (
                "LMATCH2_IMPROVES_ERROR_WITH_INSTABILITY",
                "DIAGNOSE_WRONG_SEGMENT_REINFORCEMENT",
            ),
        )

    def test_analyzer_writes_complete_auditable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = {value: self._fixture(root, value) for value in (4, 3, 2)}
            output = root / "analysis"
            checks = analyze(runs, output, bootstrap_samples=100, bootstrap_seed=0)
            expected = {
                "lmatch_ablation_summary.csv",
                "lmatch_ablation_horizon_summary.csv",
                "lmatch_ablation_scenario_summary.csv",
                "lmatch_ablation_field_summary.csv",
                "lmatch_ablation_record_range_summary.csv",
                "lmatch_ablation_segment_summary.csv",
                "lmatch_ablation_density_summary.csv",
                "lmatch_ablation_ambiguity_summary.csv",
                "lmatch_ablation_trajectory_summary.csv",
                "lmatch_ablation_bootstrap_ci.csv",
                "lmatch_ablation_protocol_comparison.json",
                "lmatch_ablation_consistency_checks.json",
                "FIG9_LMATCH_ABLATION_REPORT.md",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            self.assertTrue(checks["comparison_valid"])
            with (output / "lmatch_ablation_horizon_summary.csv").open(encoding="utf-8") as handle:
                horizon = list(csv.DictReader(handle))
            self.assertTrue(any(row["horizon_step"] == "2" and row["recurrent_trajectory_divergence"] == "True" for row in horizon))
            with (output / "lmatch_ablation_scenario_summary.csv").open(encoding="utf-8") as handle:
                scenario = list(csv.DictReader(handle))
            self.assertTrue(any(row["scenario"] == "scenario2" and row["count"] == "1" for row in scenario))


if __name__ == "__main__":
    unittest.main()
