from __future__ import annotations

import csv
import hashlib
import inspect
import json
import pickle
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from experiments.diagnostics.analyze_fig9_branch_provenance import analyze
from experiments.diagnostics.fig9_branch_provenance import (
    BranchProvenanceRegistry,
    branch_trace_rows,
    stable_segment_provenance_id,
    stable_source_fingerprint,
)
from experiments.diagnostics.fig9_competitive_inhibition import (
    CompetitionSettings,
)
from experiments.fig9 import TaxiRecord
from experiments.fig9_strict_reproduction import (
    Fig9StrictConfig,
    branch_checkpoint_sidecar_path,
    load_strict_checkpoint,
    model_long_term_fingerprint,
    model_rng_fingerprint,
    run_strict_stream,
)
from seqmem.encoding import SSTDDiscreteEncoder
from seqmem.model import MemoryParams, SequentialMemory


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_csv(path: Path) -> list[dict[str, str]]:
    runtime_fields = {
        "prediction_runtime_seconds",
        "decode_runtime_seconds",
        "competition_runtime_seconds",
        "observe_runtime",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {
                key: value
                for key, value in row.items()
                if key not in runtime_fields
            }
            for row in csv.DictReader(handle)
        ]


class Fig9BranchProvenanceUnitTests(unittest.TestCase):
    def test_trace_mapping_does_not_predict_observe_or_learn(self) -> None:
        source = inspect.getsource(branch_trace_rows)
        forbidden_calls = (
            "predict_code(",
            ".observe(",
            ".observe_code(",
            "learn_actual_code(",
        )
        for forbidden in forbidden_calls:
            self.assertNotIn(forbidden, source)

    def test_source_fingerprint_is_order_independent(self) -> None:
        self.assertEqual(
            stable_source_fingerprint([7, 2, 5]),
            stable_source_fingerprint({5, 7, 2}),
        )

    def test_stable_segment_id_is_repeatable(self) -> None:
        arguments = {
            "creation_transition_index": 9,
            "target_column": 3,
            "target_neuron": 2,
            "creation_source_fingerprint": stable_source_fingerprint(
                [4, 8]
            ),
            "stable_creation_ordinal": 1,
        }
        self.assertEqual(
            stable_segment_provenance_id(**arguments),
            stable_segment_provenance_id(**arguments),
        )

    def test_empty_trace_analysis_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = analyze(
                run_dir=root / "missing",
                output_dir=root / "analysis",
                policy_label="batched",
                bootstrap_samples=2,
                bootstrap_seed=0,
            )
            self.assertTrue(summary["empty_trace"])
            self.assertEqual(summary["candidate_rows"], 0)

    def test_registry_does_not_mutate_model_and_rebinds_after_pickle(
        self,
    ) -> None:
        model = SequentialMemory(
            encoder=SSTDDiscreteEncoder(num_columns=12, k=3, seed=4),
            num_neurons_per_column=3,
            params=MemoryParams(response_scale=1.0),
            tie_break_seed=4,
        )
        registry = BranchProvenanceRegistry()
        before = registry.segment_object_ids(model)
        model._grow_segment(0, 0, {3: 0.1, 1: 0.2}, 0.3)
        registry.capture_new_segments(
            model,
            previous_segment_ids=before,
            creation_transition_index=5,
            creation_sources={3: 0.1, 1: 0.2},
        )
        segment = model.columns[0].neurons[0].segments[0]
        origin = registry.provenance_for(segment)
        self.assertIsNotNone(origin)
        self.assertIsNone(segment.diagnostic_id)
        payload = registry.checkpoint_payload(model)
        restored_model = pickle.loads(pickle.dumps(model))
        restored = BranchProvenanceRegistry.from_checkpoint_payload(
            restored_model,
            payload,
        )
        restored_segment = restored_model.columns[0].neurons[0].segments[0]
        self.assertEqual(
            restored.provenance_for(restored_segment),
            origin,
        )

    def test_exact_created_segment_capture_matches_full_scan(self) -> None:
        model = SequentialMemory(
            encoder=SSTDDiscreteEncoder(num_columns=12, k=3, seed=4),
            num_neurons_per_column=3,
            params=MemoryParams(response_scale=1.0),
            tie_break_seed=4,
        )
        before: set[int] = set()
        model._grow_segment(2, 1, {3: 0.1, 1: 0.2}, 0.3)
        segment = model.columns[2].neurons[1].segments[0]
        scanned = BranchProvenanceRegistry()
        exact = BranchProvenanceRegistry()
        scanned.capture_new_segments(
            model,
            previous_segment_ids=before,
            creation_transition_index=5,
            creation_sources={3: 0.1, 1: 0.2},
        )
        exact.capture_created_segments(
            model,
            created=[(2, 1, segment)],
            creation_transition_index=5,
            creation_sources={3: 0.1, 1: 0.2},
        )
        self.assertEqual(
            scanned.provenance_for(segment), exact.provenance_for(segment)
        )


class Fig9BranchProvenanceIntegrationTests(unittest.TestCase):
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
        cls.records = [
            TaxiRecord(
                start + timedelta(minutes=30 * index),
                1000.0 + index * 20.0,
            )
            for index in range(18)
        ]
        settings = CompetitionSettings(
            mode="competitive_raw",
            inhibition_strength=0.1,
            inhibition_tau=0.02,
            simultaneous_policy="batched",
            simultaneous_bin_width=0.005,
        )
        cls.results = {}
        for label, enabled in (("off", False), ("on", True)):
            directory = cls.root / label
            checkpoint = directory / "checkpoint.pkl"
            cls.results[label] = run_strict_stream(
                records=cls.records,
                data_path=cls.data_path,
                stream_label="original",
                output_dir=directory,
                config=Fig9StrictConfig(warmup=6),
                limit=0,
                print_fingerprint=False,
                competition_settings=settings,
                oracle_candidate_diagnostic=True,
                branch_provenance_diagnostic=enabled,
                branch_provenance_level="full",
                checkpoint_path=checkpoint,
                checkpoint_at_index=10,
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_diagnostic_preserves_predictions_model_rng_and_traces(
        self,
    ) -> None:
        off = self.root / "off"
        on = self.root / "on"
        self.assertEqual(
            _sha256(off / "original_predictions.csv"),
            _sha256(on / "original_predictions.csv"),
        )
        self.assertEqual(
            self.results["off"]["final_model_fingerprint"],
            self.results["on"]["final_model_fingerprint"],
        )
        self.assertEqual(
            self.results["off"]["final_rng_fingerprint"],
            self.results["on"]["final_rng_fingerprint"],
        )
        self.assertEqual(
            _stable_csv(off / "competition_trace.csv"),
            _stable_csv(on / "competition_trace.csv"),
        )
        self.assertEqual(
            (off / "oracle_candidate_trace.csv").read_bytes(),
            (on / "oracle_candidate_trace.csv").read_bytes(),
        )

    def test_trace_schema_and_same_candidate_segment_mapping(self) -> None:
        directory = self.root / "on"
        with (directory / "branch_candidate_trace.csv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            candidates = list(csv.DictReader(handle))
        with (directory / "branch_segment_trace.csv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            segments = list(csv.DictReader(handle))
        self.assertTrue(candidates)
        self.assertEqual(
            {row["candidate_key"] for row in candidates},
            {row["candidate_key"] for row in segments},
        )
        self.assertTrue(
            all(row["candidate_segment_count"] == "1" for row in candidates)
        )
        self.assertTrue(
            all(row["creation_current_source_jaccard"] != ""
                for row in candidates)
        )
        self.assertTrue(
            all(row["branch_metrics_do_not_affect_prediction"] == "True"
                for row in candidates)
        )

    def test_checkpoint_keeps_v1_and_rebindable_sidecar(self) -> None:
        checkpoint_path = self.root / "on" / "checkpoint.pkl"
        checkpoint = load_strict_checkpoint(checkpoint_path)
        self.assertEqual(checkpoint["checkpoint_format"], "fig9-strict-v1")
        self.assertIsInstance(checkpoint["branch_provenance"], dict)
        sidecar_path = branch_checkpoint_sidecar_path(checkpoint_path)
        self.assertTrue(sidecar_path.exists())
        registry = BranchProvenanceRegistry.from_checkpoint_payload(
            checkpoint["model"],  # type: ignore[arg-type]
            json.loads(sidecar_path.read_text(encoding="utf-8")),
        )
        self.assertTrue(
            any(
                registry.provenance_for(segment) is not None
                for column in checkpoint["model"].columns  # type: ignore[union-attr]
                for neuron in column.neurons
                for segment in neuron.segments
            )
        )

    def test_off_checkpoint_has_no_diagnostic_sidecar(self) -> None:
        off_path = self.root / "off" / "checkpoint.pkl"
        on_path = self.root / "on" / "checkpoint.pkl"
        off = load_strict_checkpoint(off_path)
        on = load_strict_checkpoint(on_path)
        self.assertEqual(off["checkpoint_format"], on["checkpoint_format"])
        self.assertEqual(
            model_long_term_fingerprint(off["model"]),  # type: ignore[arg-type]
            model_long_term_fingerprint(on["model"]),  # type: ignore[arg-type]
        )
        self.assertEqual(
            model_rng_fingerprint(off["model"]),  # type: ignore[arg-type]
            model_rng_fingerprint(on["model"]),  # type: ignore[arg-type]
        )
        self.assertFalse(
            branch_checkpoint_sidecar_path(off_path).exists()
        )

    def test_offline_analysis_writes_required_outputs(self) -> None:
        output = self.root / "analysis"
        summary = analyze(
            run_dir=self.root / "on",
            output_dir=output,
            policy_label="batched",
            bootstrap_samples=20,
            bootstrap_seed=0,
        )
        self.assertFalse(summary["multi_segment_candidate_chimera_testable"])
        for name in (
            "branch_candidate_summary.csv",
            "branch_field_summary.csv",
            "branch_horizon_summary.csv",
            "branch_target_vs_false.csv",
            "branch_separability_metrics.csv",
            "branch_threshold_curves.csv",
            "branch_chimera_summary.csv",
            "branch_bootstrap_ci.csv",
            "branch_provenance_summary.json",
            "FIG9_BRANCH_PROVENANCE_REPORT.md",
        ):
            self.assertTrue((output / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
