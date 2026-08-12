from __future__ import annotations

import inspect
import sys
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from experiments.fig9.data import read_records
from experiments.fig9.metrics import mape
from experiments.fig9.parity import (
    CURRENT_DATA_RELATIVE,
    classify_run_family,
    current_protocol_fingerprint,
    dataset_inventory,
    decoder_partial_pattern_rows,
    encoding_fixture_rows,
    file_sha256,
    ideal_roundtrip_rows,
    paper_protocol_fingerprint,
    parity_matrix_rows,
    prediction_alignment,
    severity_sort_key,
)
from experiments.fig9_strict_reproduction import (
    Fig9StrictConfig,
    build_fig9_encoder,
    build_strict_model,
    model_long_term_fingerprint,
    model_rng_fingerprint,
    parse_args,
)
from seqmem.encoding import SSTDRealValueEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / CURRENT_DATA_RELATIVE


class Fig9PaperCodeParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = read_records(DATA_PATH, 0)
        cls.inventory = dataset_inventory(ROOT)
        cls.paper = paper_protocol_fingerprint()
        cls.current = current_protocol_fingerprint(ROOT)
        cls.matrix = parity_matrix_rows(ROOT)
        cls.encoder = build_fig9_encoder(Fig9StrictConfig())

    def test_01_dataset_row_count_audit(self) -> None:
        self.assertEqual(len(self.records), 17520)
        self.assertEqual(self.inventory[0]["row_count"], 17520)

    def test_02_timestamp_interval_audit(self) -> None:
        intervals = {
            right.timestamp - left.timestamp
            for left, right in zip(self.records, self.records[1:])
        }
        self.assertEqual(intervals, {timedelta(minutes=30)})

    def test_03_missing_interval_audit(self) -> None:
        self.assertEqual(self.inventory[0]["missing_timestamps"], 0)
        self.assertTrue(self.inventory[0]["continuous_30_minute_grid"])

    def test_04_dataset_duplicate_audit(self) -> None:
        self.assertEqual(self.inventory[0]["duplicates"], 0)

    def test_05_sha_capture(self) -> None:
        self.assertEqual(
            file_sha256(DATA_PATH),
            "092d957f5bb0d2cd62f85098ed2268114a47b4738a5f4b29ea6be4be7349fc4d",
        )

    def test_06_current_strict_actual_data_path(self) -> None:
        with patch.object(sys, "argv", ["fig9_strict_reproduction.py"]):
            args = parse_args()
        self.assertEqual(args.data.replace("\\", "/"), "data/paper_nyc_taxi.csv")

    def test_07_current_data_is_reference_asset_copy(self) -> None:
        self.assertEqual(self.inventory[0]["sha256"], self.inventory[3]["sha256"])

    def test_08_short_nab_file_is_exact_prefix(self) -> None:
        short = read_records(ROOT / "data/nyc_taxi.csv", 0)
        self.assertEqual(len(short), 10320)
        self.assertEqual(short, self.records[: len(short)])

    def test_09_paper_year_range_stays_unknown(self) -> None:
        protocol = self.paper["protocol"]
        self.assertIsNone(protocol["exact_start_timestamp"])
        self.assertIsNone(protocol["exact_end_timestamp"])

    def test_10_paper_aggregation_stays_unknown(self) -> None:
        self.assertIsNone(self.paper["protocol"]["passenger_aggregation_rule"])

    def test_11_no_current_value_backfill_into_paper_fingerprint(self) -> None:
        paper = self.paper["protocol"]
        current = self.current["protocol"]
        for field in ("passenger_min", "passenger_max", "tau_m", "tau_s"):
            self.assertIsNone(paper[field])
            self.assertIsNotNone(current[field])

    def test_12_k10_fixture(self) -> None:
        self.assertEqual([encoder.k for encoder in self.encoder.encoders], [10, 10, 10])

    def test_13_field_column_count_fixture(self) -> None:
        self.assertEqual(
            [encoder.num_columns for encoder in self.encoder.encoders],
            [30, 58, 482],
        )
        self.assertEqual(self.encoder.num_columns, 570)

    def test_14_encoding_fixture_event_count(self) -> None:
        rows = encoding_fixture_rows(ROOT)
        self.assertTrue(rows)
        self.assertTrue(all(row["event_count"] == 10 for row in rows))

    def test_15_weekday_periodic_boundary(self) -> None:
        weekday = self.encoder.encoders[0]
        monday = set(weekday.encode(0).columns)  # type: ignore[attr-defined]
        sunday = set(weekday.encode(6).columns)  # type: ignore[attr-defined]
        self.assertGreaterEqual(len(monday & sunday), 5)

    def test_16_time_periodic_boundary(self) -> None:
        time = self.encoder.encoders[1]
        midnight = set(time.encode(0).columns)  # type: ignore[attr-defined]
        late = set(time.encode(47).columns)  # type: ignore[attr-defined]
        self.assertGreaterEqual(len(midnight & late), 5)

    def test_17_passenger_neighboring_value_overlap(self) -> None:
        passenger = self.encoder.encoders[2]
        left = set(passenger.encode(15000).columns)  # type: ignore[attr-defined]
        right = set(passenger.encode(15001).columns)  # type: ignore[attr-defined]
        self.assertGreaterEqual(len(left & right), 9)

    def test_18_encoder_is_deterministic(self) -> None:
        first = self.encoder.encode((1.0, 14.0, 12345.0))
        second = self.encoder.encode((1.0, 14.0, 12345.0))
        self.assertEqual(first, second)

    def test_19_spike_order_is_deterministic(self) -> None:
        passenger = self.encoder.encoders[2]
        code = passenger.encode(12345.0)  # type: ignore[attr-defined]
        self.assertEqual(tuple(event.time for event in code.events), passenger.event_times)
        self.assertEqual(code, passenger.encode(12345.0))  # type: ignore[attr-defined]

    def test_20_spike_order_affects_decode(self) -> None:
        rows = decoder_partial_pattern_rows()
        reversed_rows = [row for row in rows if row["order_reversed"]]
        self.assertTrue(any(row["absolute_error"] > 100 for row in reversed_rows))

    def test_21_ideal_encode_decode_roundtrip_sample(self) -> None:
        passenger = self.encoder.encoders[2]
        errors = []
        for value in (0.0, 10000.0, 12345.0, 20000.0, 40000.0):
            code = passenger.encode(value)  # type: ignore[attr-defined]
            decoded = passenger.decode_likelihood(code)  # type: ignore[attr-defined]
            errors.append(abs(decoded - value))
        self.assertLessEqual(max(errors), 83.2)

    def test_22_decoder_boundary_values(self) -> None:
        passenger = self.encoder.encoders[2]
        low = passenger.decode_likelihood(passenger.encode(0.0))  # type: ignore[attr-defined]
        high = passenger.decode_likelihood(passenger.encode(40000.0))  # type: ignore[attr-defined]
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 40000.0)

    def test_23_decoder_api_has_no_future_target(self) -> None:
        signature = inspect.signature(SSTDRealValueEncoder.decode_likelihood)
        self.assertEqual(list(signature.parameters), ["self", "code", "timing_tolerance"])

    def test_24_horizon_reported_timestamp_is_plus_2_5_hours(self) -> None:
        audit = prediction_alignment(self.records)
        self.assertEqual(audit["reported_anchor_to_target_hours"], 2.5)

    def test_25_step5_causal_target_alignment_is_audited(self) -> None:
        audit = prediction_alignment(self.records)
        self.assertEqual(audit["causal_step5_expected_zero_based_index"], 204)
        self.assertEqual(audit["compared_target_zero_based_index"], 205)

    def test_26_current_off_by_one_is_detected(self) -> None:
        audit = prediction_alignment(self.records)
        self.assertTrue(audit["off_by_one_detected"])
        self.assertEqual(audit["target_index_offset_from_causal_step5"], 1)

    def test_27_desired_five_step_fixture_has_no_off_by_one(self) -> None:
        anchor = 200
        causal_step5 = anchor + 5
        target = anchor + 5
        self.assertEqual(causal_step5, target)

    def test_28_mape_definition_fixture(self) -> None:
        self.assertAlmostEqual(mape([9.0, 12.0], [10.0, 10.0]), 3.0 / 20.0)
        conventional = ((1.0 / 10.0) + (2.0 / 10.0)) / 2.0
        self.assertAlmostEqual(conventional, 0.15)

    def test_29_warmup_provenance_is_reported(self) -> None:
        current = self.current["protocol"]
        self.assertEqual(current["warmup_records_default"], 5904)
        self.assertEqual(current["diagnostic_250_500_warmup"], 200)
        self.assertIsNone(self.paper["protocol"]["warmup_records"])

    def test_30_parity_matrix_has_at_least_100_items(self) -> None:
        self.assertGreaterEqual(len(self.matrix), 100)

    def test_31_mismatch_severity_order_is_deterministic(self) -> None:
        first = [row["item"] for row in sorted(self.matrix, key=severity_sort_key)]
        second = [row["item"] for row in sorted(self.matrix, key=severity_sort_key)]
        self.assertEqual(first, second)
        severities = [row["severity"] for row in sorted(self.matrix, key=severity_sort_key)]
        self.assertEqual(severities[0], "CRITICAL")

    def test_32_run_family_classification(self) -> None:
        self.assertEqual(
            classify_run_family("results/fig9_strict/stage_a_250"),
            "PAPER_STRICT_ATTEMPT",
        )
        self.assertEqual(
            classify_run_family("results/fig9_diagnostics/lmatch_ablation"),
            "DIAGNOSTIC_ABLATION",
        )
        self.assertEqual(
            classify_run_family("experiments/diagnostics/weather_paper_snn.py"),
            "ALTERNATIVE_MODEL",
        )

    def test_33_audit_primitives_do_not_alter_model(self) -> None:
        config = Fig9StrictConfig()
        encoder = build_fig9_encoder(config)
        model = build_strict_model(encoder, config)
        long_term = model_long_term_fingerprint(model)
        rng = model_rng_fingerprint(model)
        _ = parity_matrix_rows(ROOT)
        _ = encoding_fixture_rows(ROOT)
        self.assertEqual(model_long_term_fingerprint(model), long_term)
        self.assertEqual(model_rng_fingerprint(model), rng)

    def test_34_no_compensation_in_strict_defaults(self) -> None:
        config = Fig9StrictConfig()
        self.assertTrue(config.burst_context)
        self.assertTrue(config.intracolumn_inhibition)
        self.assertEqual(config.propagation_mode, "raw")

    def test_35_no_future_covariates_in_strict_defaults(self) -> None:
        config = Fig9StrictConfig()
        self.assertFalse(config.use_future_covariates)
        self.assertFalse(config.reencode_decoded_value)
        self.assertFalse(config.rollout_learning)

    def test_36_strict_default_core_values_are_unchanged(self) -> None:
        config = Fig9StrictConfig()
        self.assertEqual(config.horizon, 5)
        self.assertEqual(config.l_match, 4)
        self.assertEqual(config.forgetting_threshold, 65.0)
        self.assertEqual(config.continuous_prediction_impl, "reference")

    def test_37_full_roundtrip_summary_regression(self) -> None:
        _rows, summary = ideal_roundtrip_rows(ROOT)
        self.assertEqual(summary["records"], 17520)
        self.assertAlmostEqual(
            summary["ideal_roundtrip_mape_repository_formula"],
            0.0014126706909136887,
        )
        self.assertLess(summary["mae"], 22.0)


if __name__ == "__main__":
    unittest.main()
