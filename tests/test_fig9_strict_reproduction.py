from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from experiments.fig9_paper_snn import TaxiRecord, mape, read_records
from experiments.fig9_strict_reproduction import (
    Fig9StrictConfig,
    build_fig9_encoder,
    build_strict_model,
    compare_pre_change_predictions,
    field_column_counts,
    find_timestamp_split,
    load_strict_checkpoint,
    model_long_term_fingerprint,
    model_rng_fingerprint,
    protocol_fingerprint,
    rollout_raw_autonomous,
    run_strict_stream,
    summarize_density,
    strict_summary_paths,
    validate_strict_fingerprint,
)
from experiments.fig9_paper_snn import learn_actual_code, record_values
from seqmem.encoding import SSTDDiscreteEncoder, SpikeEvent, SymbolCode
from seqmem.model import MemoryParams, Segment, SequentialMemory, Synapse


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

    def _tiny_records(self, count: int = 24) -> list[TaxiRecord]:
        start = datetime(2014, 7, 1, 0, 0)
        return [
            TaxiRecord(start + timedelta(minutes=30 * index), 10.0 + index)
            for index in range(count)
        ]

    def _stable_summary(self, summary: dict[str, object]) -> dict[str, object]:
        clone = dict(summary)
        clone.pop("runtime_seconds", None)
        clone.pop("records_per_second", None)
        clone.pop("density_trace_path", None)
        clone.pop("density_summary_path", None)
        return clone

    def _candidate_signature(self, model: SequentialMemory) -> tuple:
        return tuple(
            (
                column,
                tuple(
                    (
                        candidate.neuron_index,
                        candidate.score,
                        candidate.time,
                        candidate.dendritic_crossing_time,
                        candidate.peak_dendritic_potential,
                        candidate.threshold_margin,
                        candidate.predicted_soma_firing_time,
                        tuple(
                            (
                                item.source_cell_id,
                                item.source_time,
                                item.arrival_time,
                                item.weight,
                                item.psp_contribution,
                            )
                            for item in candidate.crossing_synapse_contributions
                        ),
                    )
                    for candidate in candidates
                ),
            )
            for column, candidates in sorted(model.last_prediction_candidates.items())
        )

    def _transient_signature(self, model: SequentialMemory) -> tuple:
        snapshot = model.snapshot_transient_state()
        return (
            sorted(snapshot.previous_active_cells.items()),
            sorted(snapshot.previous_winners.items()),
            self._candidate_signature(model),
            snapshot.last_prediction_stats,
            snapshot.last_observe_stats,
            snapshot.decode_rng_state,
            snapshot.learning_rng_state,
        )

    def _run_tiny_manual_fig9(
        self,
        records: list[TaxiRecord],
        *,
        continuous_impl: str,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], tuple, tuple]:
        config = Fig9StrictConfig(
            warmup=max(0, len(records) - 10),
            continuous_prediction_impl=continuous_impl,
        )
        encoder = build_fig9_encoder(config)
        model = build_strict_model(encoder, config)
        passenger_encoder = encoder.encoders[2]
        predictions: list[dict[str, object]] = []
        density_rows: list[dict[str, object]] = []
        for index, record in enumerate(records[:-config.horizon]):
            code = encoder.encode(record_values(record))
            if index >= config.warmup:
                rollout = rollout_raw_autonomous(
                    model,
                    config.horizon,
                    config=config,
                    passenger_encoder=passenger_encoder,  # type: ignore[arg-type]
                    record_index=index,
                    timestamp=record.timestamp,
                    future_records=records[index + 1 : index + config.horizon + 1],
                )
                density_rows.extend(rollout.diagnostics)
                if rollout.code is None:
                    predictions.append({"index": index, "prediction": ""})
                else:
                    predictions.append(
                        {
                            "index": index,
                            "prediction": passenger_encoder.decode_likelihood(rollout.code),  # type: ignore[attr-defined]
                            "raw_counts": rollout.raw_column_counts,
                        }
                    )
            learn_actual_code(model, code)
        stable_density_rows = []
        for row in density_rows:
            stable_density_rows.append(
                {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "prediction_runtime_seconds",
                        "decode_runtime_seconds",
                    }
                }
            )
        return (
            predictions,
            stable_density_rows,
            self._long_term_signature(model),
            self._transient_signature(model),
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
        self.assertEqual(fingerprint["continuous_impl"], "reference")
        self.assertEqual(fingerprint["continuous_impl_version"], "reference-v1")

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

    def test_field_level_column_accounting(self) -> None:
        config = Fig9StrictConfig()
        code = SymbolCode(
            (
                SpikeEvent(0, 0.0),
                SpikeEvent(29, 0.1),
                SpikeEvent(30, 0.0),
                SpikeEvent(87, 0.1),
                SpikeEvent(88, 0.0),
                SpikeEvent(569, 0.1),
            )
        )

        counts = field_column_counts(code, config)

        self.assertEqual(counts, {"weekday": 2, "time": 2, "passenger": 2})

    def test_step_level_density_summary(self) -> None:
        rows = [
            {
                "horizon_step": 1,
                "raw_predicted_column_count": 10,
                "active_source_count": 4,
                "candidate_segment_count": 7,
                "accepted_candidate_count": 5,
                "prediction_runtime_seconds": 0.1,
                "passenger_predicted_column_count": 3,
                "absolute_error": 2.0,
                "absolute_percentage_error": 0.2,
                "actual_future_passenger": 10.0,
                "stopped_reason": "",
                "weekday_predicted_column_count": 2,
                "time_predicted_column_count": 1,
            },
            {
                "horizon_step": 1,
                "raw_predicted_column_count": 20,
                "active_source_count": 6,
                "candidate_segment_count": 9,
                "accepted_candidate_count": 7,
                "prediction_runtime_seconds": 0.3,
                "passenger_predicted_column_count": 5,
                "absolute_error": 4.0,
                "absolute_percentage_error": 0.4,
                "actual_future_passenger": 10.0,
                "stopped_reason": "",
                "weekday_predicted_column_count": 2,
                "time_predicted_column_count": 2,
            },
        ]

        summary = summarize_density(rows)

        self.assertEqual(summary["rows"], 2)
        step = summary["step_summary"][0]  # type: ignore[index]
        self.assertEqual(step["raw_columns_mean"], 15)
        self.assertEqual(step["passenger_field_density_mean"], 4)
        self.assertEqual(step["raw_columns_p50"], 10)
        self.assertEqual(step["raw_columns_p90"], 20)
        self.assertEqual(step["raw_columns_p99"], 20)
        self.assertEqual(step["raw_columns_max"], 20)

    def test_density_trace_does_not_change_predictions(self) -> None:
        records = self._tiny_records()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "tiny.csv"
            data_path.write_text("timestamp,passenger_count\n", encoding="utf-8")
            plain = run_strict_stream(
                records=records,
                data_path=data_path,
                stream_label="original",
                output_dir=root / "plain",
                config=Fig9StrictConfig(warmup=8),
                limit=0,
                print_fingerprint=False,
            )
            traced = run_strict_stream(
                records=records,
                data_path=data_path,
                stream_label="original",
                output_dir=root / "traced",
                config=Fig9StrictConfig(warmup=8),
                limit=0,
                print_fingerprint=False,
                density_trace_path=root / "trace.csv",
                density_summary_path=root / "trace.json",
            )

        self.assertEqual(self._stable_summary(plain), self._stable_summary(traced))

    def test_density_trace_csv_field_sums_match_total(self) -> None:
        records = self._tiny_records()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "tiny.csv"
            data_path.write_text("timestamp,passenger_count\n", encoding="utf-8")
            trace_path = root / "trace.csv"
            run_strict_stream(
                records=records,
                data_path=data_path,
                stream_label="original",
                output_dir=root / "traced",
                config=Fig9StrictConfig(warmup=8),
                limit=0,
                print_fingerprint=False,
                density_trace_path=trace_path,
                density_summary_path=root / "trace.json",
            )

            with trace_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertGreater(len(rows), 0)
        for row in rows:
            total = int(row["raw_predicted_column_count"])
            fields = (
                int(row["weekday_predicted_column_count"])
                + int(row["time_predicted_column_count"])
                + int(row["passenger_predicted_column_count"])
            )
            self.assertEqual(fields, total)

    def test_debug_record_output_does_not_change_predictions(self) -> None:
        records = self._tiny_records()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "tiny.csv"
            data_path.write_text("timestamp,passenger_count\n", encoding="utf-8")
            plain = run_strict_stream(
                records=records,
                data_path=data_path,
                stream_label="original",
                output_dir=root / "plain",
                config=Fig9StrictConfig(warmup=8),
                limit=0,
                print_fingerprint=False,
            )
            debug = run_strict_stream(
                records=records,
                data_path=data_path,
                stream_label="original",
                output_dir=root / "debug",
                config=Fig9StrictConfig(warmup=8),
                limit=0,
                print_fingerprint=False,
                debug_record_index=8,
                debug_output_json=root / "debug_record.json",
            )
            payload = json.loads((root / "debug_record.json").read_text(encoding="utf-8"))

        self.assertEqual(self._stable_summary(plain), self._stable_summary(debug))
        self.assertEqual(payload["record_index"], 8)
        self.assertEqual(len(payload["rollout_steps"]), 5)
        self.assertIn("scenario_counts", payload["observe_after"])

    def test_checkpoint_resume_identity(self) -> None:
        records = self._tiny_records(26)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "tiny.csv"
            data_path.write_text("timestamp,passenger_count\n", encoding="utf-8")
            config = Fig9StrictConfig(warmup=8)
            continuous = run_strict_stream(
                records=records,
                data_path=data_path,
                stream_label="original",
                output_dir=root / "continuous",
                config=config,
                limit=0,
                print_fingerprint=False,
            )
            checkpoint = root / "state.pkl"
            run_strict_stream(
                records=records,
                data_path=data_path,
                stream_label="original",
                output_dir=root / "prefix",
                config=config,
                limit=0,
                print_fingerprint=False,
                checkpoint_path=checkpoint,
                checkpoint_at_index=13,
                stop_after_index=13,
            )
            resumed = run_strict_stream(
                records=records,
                data_path=data_path,
                stream_label="original",
                output_dir=root / "resumed",
                config=config,
                limit=0,
                print_fingerprint=False,
                resume_checkpoint=checkpoint,
            )

        self.assertEqual(
            self._stable_summary(continuous),
            self._stable_summary(resumed),
        )

    def test_checkpoint_restores_transient_rng_and_metrics(self) -> None:
        records = self._tiny_records(20)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "tiny.csv"
            data_path.write_text("timestamp,passenger_count\n", encoding="utf-8")
            checkpoint = root / "state.pkl"
            run_strict_stream(
                records=records,
                data_path=data_path,
                stream_label="original",
                output_dir=root / "prefix",
                config=Fig9StrictConfig(warmup=8),
                limit=0,
                print_fingerprint=False,
                checkpoint_path=checkpoint,
                checkpoint_at_index=12,
                stop_after_index=12,
            )
            payload = load_strict_checkpoint(checkpoint)
            model = payload["model"]

        snapshot = model.snapshot_transient_state()
        self.assertEqual(snapshot.decode_rng_state, model._decode_rng.getstate())
        self.assertEqual(snapshot.learning_rng_state, model._learning_rng.getstate())
        self.assertEqual(payload["next_index"], 12)
        self.assertIn("predictions", payload)
        self.assertIn("targets", payload)

    def test_checkpoint_rejects_incompatible_prefix(self) -> None:
        records = self._tiny_records(20)
        changed = [TaxiRecord(record.timestamp, record.value) for record in records]
        changed[0] = TaxiRecord(changed[0].timestamp, changed[0].value + 100.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "tiny.csv"
            changed_path = root / "changed.csv"
            data_path.write_text("timestamp,passenger_count\n1,1\n", encoding="utf-8")
            changed_path.write_text("timestamp,passenger_count\n2,2\n", encoding="utf-8")
            checkpoint = root / "state.pkl"
            run_strict_stream(
                records=records,
                data_path=data_path,
                stream_label="original",
                output_dir=root / "prefix",
                config=Fig9StrictConfig(warmup=8),
                limit=0,
                print_fingerprint=False,
                checkpoint_path=checkpoint,
                checkpoint_at_index=12,
                stop_after_index=12,
            )
            with self.assertRaises(ValueError):
                run_strict_stream(
                    records=changed,
                    data_path=changed_path,
                    stream_label="original",
                    output_dir=root / "bad",
                    config=Fig9StrictConfig(warmup=8),
                    limit=0,
                    print_fingerprint=False,
                    resume_checkpoint=checkpoint,
                )

    def test_timestamp_based_april_split(self) -> None:
        records = [
            TaxiRecord(datetime(2015, 3, 31, 23, 30), 1.0),
            TaxiRecord(datetime(2015, 4, 1, 0, 0), 2.0),
        ]

        self.assertEqual(find_timestamp_split(records), 1)

    def test_strict_fingerprint_unchanged_by_phase2_options(self) -> None:
        config = Fig9StrictConfig()
        self.assertTrue(config.burst_context)
        self.assertEqual(config.propagation_mode, "raw")
        self.assertFalse(config.use_future_covariates)
        self.assertFalse(config.reencode_decoded_value)
        self.assertFalse(config.rollout_learning)
        self.assertEqual(config.continuous_prediction_impl, "reference")

    def test_continuous_prediction_implementations_are_identical(self) -> None:
        model = SequentialMemory(
            encoder=SSTDDiscreteEncoder(num_columns=8, k=2, seed=4),
            params=MemoryParams(continuous_dynamics=True),
        )
        arrivals = [(0.1, 0.5), (0.2, 0.5), (0.2, 0.25), (0.35, 0.75)]
        reference_diagnostic: dict[str, object] = {}
        optimized_v1_diagnostic: dict[str, object] = {}
        optimized_v2_diagnostic: dict[str, object] = {}

        reference = model.reference_continuous_prediction(
            arrivals, diagnostic=reference_diagnostic
        )
        optimized_v1 = model.optimized_v1_continuous_prediction(
            arrivals, diagnostic=optimized_v1_diagnostic
        )
        optimized_v2 = model.optimized_continuous_prediction(
            arrivals, diagnostic=optimized_v2_diagnostic
        )

        self.assertEqual(reference, optimized_v1)
        self.assertEqual(reference, optimized_v2)
        self.assertEqual(
            reference_diagnostic["first_threshold_crossing_time"],
            optimized_v1_diagnostic["first_threshold_crossing_time"],
        )
        self.assertEqual(
            reference_diagnostic["first_threshold_crossing_time"],
            optimized_v2_diagnostic["first_threshold_crossing_time"],
        )
        self.assertEqual(reference_diagnostic, optimized_v1_diagnostic)
        self.assertEqual(reference_diagnostic, optimized_v2_diagnostic)

    def test_non_grid_arrivals_use_equivalent_general_path(self) -> None:
        model = SequentialMemory(
            encoder=SSTDDiscreteEncoder(num_columns=8, k=2, seed=5),
            params=MemoryParams(continuous_dynamics=True),
        )
        arrivals = [(0.1017, 0.5), (0.2134, 0.5), (0.3329, 0.25)]

        self.assertEqual(
            model.reference_continuous_prediction(arrivals, diagnostic={}),
            model.optimized_continuous_prediction(arrivals, diagnostic={}),
        )

    def test_optimized_v2_memo_reuses_only_identical_arrivals(self) -> None:
        model = SequentialMemory(
            encoder=SSTDDiscreteEncoder(num_columns=8, k=2, seed=7),
            params=MemoryParams(continuous_prediction_impl="optimized_v2"),
        )
        first = Segment(
            synapses={
                1: Synapse(source=1, delay=0.1, weight=0.5),
                2: Synapse(source=2, delay=0.1, weight=0.5),
            }
        )
        same_arrivals = Segment(
            synapses={
                1: Synapse(source=1, delay=0.1, weight=0.5),
                2: Synapse(source=2, delay=0.1, weight=0.5),
            }
        )
        different_arrivals = Segment(
            synapses={
                1: Synapse(source=1, delay=0.1, weight=0.5),
                2: Synapse(source=2, delay=0.2, weight=0.5),
            }
        )
        active_sources = {1: 0.0, 2: 0.0}
        calls: list[tuple[tuple[float, float], ...]] = []
        original = model._continuous_prediction_from_arrivals

        def counted(arrivals, diagnostic=None):
            calls.append(tuple(arrivals))
            return original(arrivals, diagnostic=diagnostic)

        model._continuous_prediction_from_arrivals = counted  # type: ignore[method-assign]
        memo: dict[
            tuple[tuple[float, float], ...],
            tuple[tuple[float, float] | None, dict[str, object]],
        ] = {}

        first_result = model._continuous_segment_prediction(
            first,
            active_sources,
            result_memo=memo,
        )
        same_result = model._continuous_segment_prediction(
            same_arrivals,
            active_sources,
            result_memo=memo,
        )
        model._continuous_segment_prediction(
            different_arrivals,
            active_sources,
            result_memo=memo,
        )

        self.assertEqual(first_result, same_result)
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            calls,
            [
                ((0.1, 0.5), (0.1, 0.5)),
                ((0.1, 0.5), (0.2, 0.5)),
            ],
        )

    def test_prediction_candidates_match_reference_path(self) -> None:
        encoder = SSTDDiscreteEncoder(num_columns=8, k=2, seed=3)
        reference = SequentialMemory(
            encoder=encoder,
            num_neurons_per_column=4,
            params=MemoryParams(continuous_prediction_impl="reference"),
        )
        optimized_v1 = SequentialMemory(
            encoder=encoder,
            num_neurons_per_column=4,
            params=MemoryParams(continuous_prediction_impl="optimized_v1"),
        )
        optimized_v2 = SequentialMemory(encoder=encoder, num_neurons_per_column=4)
        for model in (reference, optimized_v1, optimized_v2):
            for _ in range(4):
                model.reset_state()
                model.observe("A")
                model.predict_code()
                model.observe("B")
            model.reset_state()
            model.observe("A", learn=False)

        reference_code = reference.predict_code()
        optimized_v1_code = optimized_v1.predict_code()
        optimized_v2_code = optimized_v2.predict_code()

        self.assertEqual(reference_code, optimized_v1_code)
        self.assertEqual(reference_code, optimized_v2_code)
        self.assertEqual(
            self._candidate_signature(reference),
            self._candidate_signature(optimized_v1),
        )
        self.assertEqual(
            self._candidate_signature(reference),
            self._candidate_signature(optimized_v2),
        )

    def test_ten_and_fifty_record_streams_match_reference_path(self) -> None:
        for count in (10, 50):
            records = self._tiny_records(count)

            reference = self._run_tiny_manual_fig9(
                records,
                continuous_impl="reference",
            )
            optimized_v1 = self._run_tiny_manual_fig9(
                records,
                continuous_impl="optimized_v1",
            )
            optimized_v2 = self._run_tiny_manual_fig9(
                records,
                continuous_impl="optimized_v2",
            )

            self.assertEqual(reference[0], optimized_v1[0])
            self.assertEqual(reference[1], optimized_v1[1])
            self.assertEqual(reference[2], optimized_v1[2])
            self.assertEqual(reference[3], optimized_v1[3])
            self.assertEqual(reference[0], optimized_v2[0])
            self.assertEqual(reference[1], optimized_v2[1])
            self.assertEqual(reference[2], optimized_v2[2])
            self.assertEqual(reference[3], optimized_v2[3])

    def test_strict_summary_fingerprints_are_stable(self) -> None:
        records = self._tiny_records(18)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "synthetic.csv"
            data_path.write_text(
                "timestamp,passenger_count\n",
                encoding="utf-8",
            )
            summaries = []
            for impl in ("reference", "optimized_v1", "optimized_v2"):
                config = Fig9StrictConfig(
                    warmup=8,
                    continuous_prediction_impl=impl,
                )
                summaries.append(
                    run_strict_stream(
                        records=records,
                        data_path=data_path,
                        stream_label=impl,
                        output_dir=root / impl,
                        config=config,
                        limit=len(records),
                        print_fingerprint=False,
                    )
                )

        baseline = summaries[0]
        for summary in summaries[1:]:
            self.assertEqual(
                baseline["final_model_fingerprint"],
                summary["final_model_fingerprint"],
            )
            self.assertEqual(
                baseline["final_rng_fingerprint"],
                summary["final_rng_fingerprint"],
            )

    def test_model_fingerprint_helpers_are_stable(self) -> None:
        encoder = SSTDDiscreteEncoder(num_columns=8, k=2, seed=6)
        model = SequentialMemory(encoder=encoder, num_neurons_per_column=4)
        for _ in range(3):
            model.reset_state()
            model.observe("A")
            model.predict_code()
            model.observe("B")

        self.assertEqual(
            model_long_term_fingerprint(model),
            model_long_term_fingerprint(model),
        )
        self.assertEqual(model_rng_fingerprint(model), model_rng_fingerprint(model))


if __name__ == "__main__":
    unittest.main()
