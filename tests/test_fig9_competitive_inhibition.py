from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from experiments.diagnostics.fig9_competitive_inhibition import (
    CompetitionCandidate,
    CompetitionSettings,
    candidates_for_prediction,
    compete_prediction_candidates,
    emitted_prediction_code,
)
from experiments.fig9 import TaxiRecord, learn_actual_code, record_values
from experiments.fig9_strict_reproduction import (
    Fig9StrictConfig,
    build_fig9_encoder,
    build_strict_model,
    model_long_term_fingerprint,
    model_rng_fingerprint,
    protocol_fingerprint,
    rollout_raw_autonomous,
    run_strict_stream,
)
from seqmem.encoding import SpikeEvent, SymbolCode
from seqmem.model import PredictionCandidate, Segment


class Fig9CompetitiveInhibitionTests(unittest.TestCase):
    def _candidate(
        self,
        column: int,
        neuron: int,
        score: float,
        predicted_time: float,
        order: int,
    ) -> CompetitionCandidate:
        return CompetitionCandidate(
            column_index=column,
            candidate=PredictionCandidate(
                neuron_index=neuron,
                score=score,
                time=predicted_time,
                segment=Segment(),
            ),
            original_order=order,
        )

    def _compete(
        self,
        candidates: list[CompetitionCandidate],
        *,
        strength: float = 0.1,
        tau: float = 0.02,
    ):
        return compete_prediction_candidates(
            candidates,
            threshold=1.0,
            inhibition_strength=strength,
            inhibition_tau=tau,
            simultaneous_tolerance=0.0,
        )

    def _records(self, count: int = 18) -> list[TaxiRecord]:
        start = datetime(2014, 7, 1)
        return [
            TaxiRecord(start + timedelta(minutes=30 * index), 1000.0 + index)
            for index in range(count)
        ]

    def test_off_mode_matches_raw_rollout(self) -> None:
        config = Fig9StrictConfig(warmup=4)
        encoder = build_fig9_encoder(config)
        model = build_strict_model(encoder, config)
        for record in self._records(10):
            learn_actual_code(model, encoder.encode(record_values(record)))

        raw = rollout_raw_autonomous(model, 2)
        off = rollout_raw_autonomous(
            model,
            2,
            competition_settings=CompetitionSettings(mode="off"),
        )

        self.assertEqual(off, raw)

    def test_zero_strength_returns_raw_code_verbatim(self) -> None:
        raw = SymbolCode(
            events=(
                SpikeEvent(column=5, time=0.1),
                SpikeEvent(column=2, time=0.0),
            )
        )
        mapping = {
            5: [self._candidate(5, 1, 1.2, 0.1, 0).candidate],
            2: [self._candidate(2, 0, 1.1, 0.0, 1).candidate],
        }
        candidates = candidates_for_prediction(
            raw,
            mapping,
            timing_tolerance=0.03,
        )
        result = self._compete(list(candidates), strength=0.0)

        self.assertEqual(emitted_prediction_code(raw, result), raw)
        self.assertEqual(result.emitted_column_count, 2)

    def test_zero_strength_predictions_csv_matches_raw_bytes(self) -> None:
        records = self._records(14)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "tiny.csv"
            data_path.write_text("timestamp,passenger_count\n", encoding="utf-8")
            common = {
                "records": records,
                "data_path": data_path,
                "stream_label": "original",
                "config": Fig9StrictConfig(warmup=6),
                "limit": 0,
                "print_fingerprint": False,
            }
            raw = run_strict_stream(
                output_dir=root / "raw",
                **common,
            )
            zero = run_strict_stream(
                output_dir=root / "zero",
                competition_settings=CompetitionSettings(
                    mode="competitive_raw",
                    inhibition_strength=0.0,
                ),
                **common,
            )
            raw_csv = (root / "raw" / "original_predictions.csv").read_bytes()
            zero_csv = (root / "zero" / "original_predictions.csv").read_bytes()

        self.assertEqual(zero_csv, raw_csv)
        self.assertEqual(
            zero["final_model_fingerprint"],
            raw["final_model_fingerprint"],
        )
        self.assertEqual(
            zero["final_rng_fingerprint"],
            raw["final_rng_fingerprint"],
        )

    def test_repeated_and_shuffled_inputs_are_deterministic(self) -> None:
        candidates = [
            self._candidate(4, 2, 1.2, 0.02, 2),
            self._candidate(1, 3, 1.1, 0.00, 0),
            self._candidate(2, 1, 1.3, 0.01, 1),
        ]

        first = self._compete(candidates)
        repeated = self._compete(candidates)
        shuffled = self._compete(list(reversed(candidates)))

        self.assertEqual(first, repeated)
        self.assertEqual(first, shuffled)

    def test_early_winner_suppresses_nearby_weaker_candidate(self) -> None:
        early = self._candidate(0, 0, 1.2, 0.0, 0)
        weak = self._candidate(1, 0, 1.05, 0.0, 1)

        result = self._compete([weak, early], strength=0.1)

        self.assertEqual(result.emitted_candidates, (early,))
        self.assertEqual(result.inhibited_candidates, (weak,))
        self.assertEqual(
            result.decisions[1].winning_predecessor_ids,
            (early.candidate_id,),
        )

    def test_inhibition_decays_with_time_distance(self) -> None:
        early = self._candidate(0, 0, 2.0, 0.0, 0)
        near = self._candidate(1, 0, 2.0, 0.01, 1)
        far = self._candidate(2, 0, 2.0, 0.20, 2)

        result = self._compete([early, near, far], strength=0.1, tau=0.05)

        self.assertGreater(
            result.decisions[1].accumulated_inhibition,
            0.1 * pow(2.718281828459045, -0.01 / 0.05) - 1e-12,
        )
        early_effect_on_far = 0.1 * pow(2.718281828459045, -0.20 / 0.05)
        self.assertGreater(
            result.decisions[2].accumulated_inhibition,
            early_effect_on_far,
        )
        self.assertLess(
            early_effect_on_far,
            result.decisions[1].accumulated_inhibition,
        )

    def test_same_column_does_not_create_inhibition(self) -> None:
        first = self._candidate(3, 0, 1.2, 0.0, 0)
        second = self._candidate(3, 1, 1.05, 0.0, 1)

        result = self._compete([first, second], strength=1.0)

        self.assertEqual(len(result.emitted_candidates), 2)
        self.assertEqual(result.decisions[1].accumulated_inhibition, 0.0)

    def test_competition_is_read_only_for_model_and_rng(self) -> None:
        config = Fig9StrictConfig(warmup=4)
        encoder = build_fig9_encoder(config)
        model = build_strict_model(encoder, config)
        for record in self._records(12):
            learn_actual_code(model, encoder.encode(record_values(record)))
        memory_before = model_long_term_fingerprint(model)
        rng_before = model_rng_fingerprint(model)

        rollout_raw_autonomous(
            model,
            3,
            competition_settings=CompetitionSettings(
                mode="competitive_raw",
                inhibition_strength=0.1,
            ),
        )

        self.assertEqual(model_long_term_fingerprint(model), memory_before)
        self.assertEqual(model_rng_fingerprint(model), rng_before)

    def test_density_trace_switch_preserves_competitive_result(self) -> None:
        records = self._records()
        settings = CompetitionSettings(
            mode="competitive_raw",
            inhibition_strength=0.1,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "tiny.csv"
            data_path.write_text("timestamp,passenger_count\n", encoding="utf-8")
            plain = run_strict_stream(
                records=records,
                data_path=data_path,
                stream_label="original",
                output_dir=root / "plain",
                config=Fig9StrictConfig(warmup=6),
                limit=0,
                print_fingerprint=False,
                competition_settings=settings,
            )
            traced = run_strict_stream(
                records=records,
                data_path=data_path,
                stream_label="original",
                output_dir=root / "traced",
                config=Fig9StrictConfig(warmup=6),
                limit=0,
                print_fingerprint=False,
                density_trace_path=root / "density.csv",
                density_summary_path=root / "density.json",
                competition_settings=settings,
            )
            left = (root / "plain" / "original_predictions.csv").read_bytes()
            right = (root / "traced" / "original_predictions.csv").read_bytes()

        for summary in (plain, traced):
            summary.pop("runtime_seconds", None)
            summary.pop("records_per_second", None)
            summary.pop("density_trace_path", None)
            summary.pop("density_summary_path", None)
            summary.pop("competition_trace_path", None)
            summary.pop("competition_summary_path", None)
        self.assertEqual(left, right)
        self.assertEqual(plain, traced)

    def test_competition_api_has_no_ground_truth_input(self) -> None:
        for function in (
            compete_prediction_candidates,
            candidates_for_prediction,
            emitted_prediction_code,
        ):
            parameters = inspect.signature(function).parameters
            for forbidden in ("target", "expected", "ground_truth", "future"):
                self.assertNotIn(forbidden, parameters)

    def test_raw_checkpoint_and_protocol_remain_strict(self) -> None:
        records = self._records(14)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "tiny.csv"
            data_path.write_text("timestamp,passenger_count\n", encoding="utf-8")
            checkpoint = root / "state.pkl"
            run_strict_stream(
                records=records,
                data_path=data_path,
                stream_label="original",
                output_dir=root / "raw",
                config=Fig9StrictConfig(warmup=6),
                limit=0,
                print_fingerprint=False,
                checkpoint_path=checkpoint,
                checkpoint_at_index=8,
                stop_after_index=8,
            )
            protocol = json.loads(
                (root / "raw" / "original_protocol.json").read_text(
                    encoding="utf-8"
                )
            )
            import pickle

            with checkpoint.open("rb") as handle:
                payload = pickle.load(handle)

        self.assertEqual(payload["checkpoint_format"], "fig9-strict-v1")
        self.assertNotIn("competition", payload)
        self.assertEqual(protocol["propagation_mode"], "raw")
        self.assertNotIn("diagnostic_only", protocol)
        self.assertNotIn("competition_is_local_choice", protocol)

    def test_strict_protocol_fingerprint_has_no_competition_flags(self) -> None:
        records = self._records(8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.csv"
            path.write_text("timestamp,passenger_count\n", encoding="utf-8")
            fingerprint = protocol_fingerprint(
                data_path=path,
                records=records,
                stream_label="original",
                config=Fig9StrictConfig(),
                limit=0,
            )

        self.assertEqual(fingerprint["propagation_mode"], "raw")
        self.assertNotIn("competition_mode", fingerprint)
        self.assertNotIn("diagnostic_only", fingerprint)

    def test_competitive_outputs_are_explicitly_nonpaper(self) -> None:
        records = self._records(14)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "tiny.csv"
            data_path.write_text("timestamp,passenger_count\n", encoding="utf-8")
            run_strict_stream(
                records=records,
                data_path=data_path,
                stream_label="original",
                output_dir=root,
                config=Fig9StrictConfig(warmup=6),
                limit=0,
                print_fingerprint=False,
                competition_settings=CompetitionSettings(
                    mode="competitive_raw",
                    inhibition_strength=0.1,
                ),
            )
            protocol = json.loads(
                (root / "competition_protocol.json").read_text(
                    encoding="utf-8"
                )
            )
            summary = json.loads(
                (root / "competition_summary.json").read_text(
                    encoding="utf-8"
                )
            )

        for payload in (protocol, summary):
            self.assertTrue(payload["diagnostic_only"])
            self.assertTrue(payload["competition_is_local_choice"])


if __name__ == "__main__":
    unittest.main()
