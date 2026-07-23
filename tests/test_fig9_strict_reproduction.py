from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from experiments.fig9_paper_snn import TaxiRecord, mape, read_records
from experiments.fig9_strict_reproduction import (
    Fig9StrictConfig,
    build_fig9_encoder,
    build_strict_model,
    compare_pre_change_predictions,
    protocol_fingerprint,
    rollout_raw_autonomous,
    run_strict_stream,
    strict_summary_paths,
    validate_strict_fingerprint,
)
from seqmem.encoding import SSTDDiscreteEncoder
from seqmem.model import MemoryParams, SequentialMemory


class Fig9StrictReproductionTests(unittest.TestCase):
    def _long_term_signature(self, model: SequentialMemory) -> tuple:
        return tuple(
            tuple(
                tuple(
                    (
                        segment.active,
                        segment.target_time,
                        tuple(
                            sorted(
                                (
                                    source,
                                    synapse.delay,
                                    synapse.weight,
                                    synapse.age,
                                )
                                for source, synapse in segment.synapses.items()
                            )
                        ),
                    )
                    for segment in neuron.segments
                )
                for neuron in column.neurons
            )
            for column in model.columns
        )

    def test_strict_defaults_are_all_cell_burst_and_raw(self) -> None:
        config = Fig9StrictConfig()

        self.assertTrue(config.burst_context)
        self.assertEqual(config.propagation_mode, "raw")
        self.assertFalse(config.use_future_covariates)
        self.assertFalse(config.reencode_decoded_value)
        self.assertFalse(config.rollout_learning)

    def test_encoder_sizes_match_paper(self) -> None:
        config = Fig9StrictConfig()
        encoder = build_fig9_encoder(config)

        self.assertEqual([field.num_columns for field in encoder.encoders], [30, 58, 482])
        self.assertEqual(sum(field.num_columns for field in encoder.encoders), 570)
        self.assertTrue(all(field.k == 10 for field in encoder.encoders))
        model = build_strict_model(encoder, config)
        self.assertEqual(len(model.columns[0].neurons), 32)
        self.assertEqual(model.params.l_match, 4)
        self.assertEqual(model.params.forgetting_threshold, 65.0)

    def test_horizon_is_strictly_five(self) -> None:
        self.assertEqual(Fig9StrictConfig().horizon, 5)

    def test_validate_strict_rejects_compensation(self) -> None:
        fingerprint = {
            "burst_context": "winner-only",
            "propagation_mode": "eventwise",
            "uses_future_covariates": True,
            "reencodes_decoded_value": True,
            "learns_during_rollout": True,
            "strict_label": "nonpaper diagnostic",
        }

        with self.assertRaises(ValueError):
            validate_strict_fingerprint(fingerprint)

    def test_protocol_fingerprint_matches_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "taxi.csv"
            path.write_text(
                "timestamp,passenger_count\n2014-07-01 00:00:00,10\n",
                encoding="utf-8",
            )
            records = read_records(path, 0)
            config = Fig9StrictConfig(warmup=5)

            fingerprint = protocol_fingerprint(
                data_path=path,
                records=records,
                stream_label="original",
                config=config,
                limit=0,
            )

        self.assertEqual(fingerprint["encoder_sizes"], {
            "weekday": 30,
            "time": 58,
            "passenger": 482,
            "total": 570,
        })
        self.assertEqual(fingerprint["prediction_horizon"], 5)
        self.assertEqual(fingerprint["warmup_length"], 5)
        self.assertEqual(fingerprint["burst_context"], "all-cell")
        self.assertEqual(fingerprint["propagation_mode"], "raw")

    def test_rollout_does_not_learn_or_reencode(self) -> None:
        encoder = SSTDDiscreteEncoder(num_columns=4, k=1, seed=2)
        model = SequentialMemory(encoder=encoder, params=MemoryParams())
        for _ in range(3):
            model.reset_state()
            model.observe("A")
            model.observe("B")
        model.reset_state()
        model.observe("A", learn=False)
        before_long_term = self._long_term_signature(model)
        before_transient = model.snapshot_transient_state()

        def forbidden_observe(*_args, **_kwargs) -> None:
            raise AssertionError("rollout must not observe decoded values")

        model.observe_code = forbidden_observe  # type: ignore[method-assign]
        rollout_raw_autonomous(model, 1)
        after_long_term = self._long_term_signature(model)

        self.assertEqual(before_long_term, after_long_term)
        self.assertEqual(
            model.snapshot_transient_state(),
            before_transient,
        )

    def test_run_predicts_before_observe(self) -> None:
        records = [
            TaxiRecord(datetime(2014, 7, 1, 0, 0), 10.0 + index)
            for index in range(12)
        ]
        events: list[tuple[str, int]] = []
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "tiny.csv"
            data_path.write_text(
                "timestamp,passenger_count\n"
                + "\n".join(
                    f"{record.timestamp:%Y-%m-%d %H:%M:%S},{record.value}"
                    for record in records
                )
                + "\n",
                encoding="utf-8",
            )
            run_strict_stream(
                records=records,
                data_path=data_path,
                stream_label="original",
                output_dir=Path(directory) / "out",
                config=Fig9StrictConfig(warmup=5),
                limit=0,
                event_hook=lambda event, index: events.append((event, index)),
                print_fingerprint=False,
            )

        per_index: dict[int, list[str]] = {}
        for event, index in events:
            per_index.setdefault(index, []).append(event)
        self.assertEqual(per_index[5][:2], ["predict", "observe"])

    def test_perturbed_file_matches_original_before_april_first(self) -> None:
        original = read_records(Path("data/paper_nyc_taxi.csv"), 0)
        perturbed = read_records(Path("data/paper_nyc_taxi_perturb.csv"), 0)

        for left, right in zip(original, perturbed):
            if left.timestamp >= datetime(2015, 4, 1):
                break
            self.assertEqual(left.timestamp, right.timestamp)
            self.assertEqual(left.value, right.value)

    def test_perturbation_only_affects_paper_windows(self) -> None:
        original = read_records(Path("data/paper_nyc_taxi.csv"), 0)
        perturbed = read_records(Path("data/paper_nyc_taxi_perturb.csv"), 0)
        changed = 0
        for left, right in zip(original, perturbed):
            if left.timestamp < datetime(2015, 4, 1):
                continue
            hour = left.timestamp.hour + left.timestamp.minute / 60.0
            in_window = (
                left.timestamp.weekday() < 5
                and (7.0 <= hour < 11.0 or 21.0 <= hour < 24.0)
            )
            if in_window:
                changed += left.value != right.value
            else:
                self.assertEqual(left.value, right.value)
        self.assertGreater(changed, 0)

    def test_mape_reference_formula_small_sample(self) -> None:
        self.assertAlmostEqual(mape([9.0, 12.0], [10.0, 10.0]), 3.0 / 20.0)

    def test_pre_change_comparison_detects_identical_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            left = Path(directory) / "left.csv"
            right = Path(directory) / "right.csv"
            rows = [
                {
                    "target_timestamp": "2015-03-31 23:30:00",
                    "target": "1",
                    "prediction": "2",
                    "rolling_mape": "0.1",
                    "raw_rollout_column_counts": "1 2 3",
                }
            ]
            for path in (left, right):
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)

            comparison = compare_pre_change_predictions(left, right)

        self.assertTrue(comparison["pre_change_predictions_identical"])

    def test_historical_compensated_is_not_strict_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            strict = root / "fig9_strict"
            historical = root / "fig9_historical_compensated"
            strict.mkdir()
            historical.mkdir()
            (strict / "original_summary.json").write_text("{}", encoding="utf-8")
            (historical / "old_summary.json").write_text("{}", encoding="utf-8")

            paths = strict_summary_paths(root)

        self.assertEqual(paths, [strict / "original_summary.json"])


if __name__ == "__main__":
    unittest.main()
