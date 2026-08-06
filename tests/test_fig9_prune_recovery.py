from __future__ import annotations

import csv
import gzip
import hashlib
import json
import pickle
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from experiments.fig9 import TaxiRecord
from experiments.fig9_strict_reproduction import (
    Fig9StrictConfig,
    build_fig9_encoder,
    build_strict_model,
    load_strict_checkpoint,
    model_long_term_fingerprint,
    model_rng_fingerprint,
    run_strict_stream,
)
from scripts.prepare_resume_from_checkpoint import trim_csv
from scripts.run_logged_process import _checkpoint_index, run as run_logged_process


class Fig9PruneRecoveryTests(unittest.TestCase):
    def test_checkpoint_metadata_is_atomic_and_has_boundary(self) -> None:
        records = [
            TaxiRecord(
                datetime(2014, 7, 1) + timedelta(minutes=30 * index),
                100.0 + index,
            )
            for index in range(12)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data.csv"
            data.write_text("placeholder\n", encoding="utf-8")
            checkpoint = root / "checkpoint.pkl"
            run_strict_stream(
                records=records,
                data_path=data,
                stream_label="original",
                output_dir=root / "run",
                config=Fig9StrictConfig(
                    horizon=2,
                    warmup=0,
                    weekday_columns=3,
                    time_columns=4,
                    passenger_columns=8,
                    neurons_per_column=3,
                    k=2,
                    rolling_window=4,
                ),
                limit=len(records),
                print_fingerprint=False,
                checkpoint_path=checkpoint,
                checkpoint_every=2,
            )
            metadata_path = root / "checkpoint.metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertTrue(metadata["atomic_complete"])
            self.assertEqual(metadata["next_index"], 10)
            self.assertEqual(
                metadata["checkpoint_sha256"],
                hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            )
            loaded = load_strict_checkpoint(checkpoint)
            self.assertEqual(loaded["next_index"], metadata["next_index"])
            self.assertEqual(
                metadata["model_fingerprint"],
                model_long_term_fingerprint(loaded["model"]),
            )
            self.assertEqual(
                metadata["rng_fingerprint"],
                model_rng_fingerprint(loaded["model"]),
            )

    def test_supervisor_reads_metadata_without_unpickling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pkl"
            checkpoint.write_bytes(b"this is intentionally not a pickle")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            metadata = root / "checkpoint.metadata.json"
            metadata.write_text(
                json.dumps(
                    {
                        "atomic_complete": True,
                        "next_index": 17,
                        "checkpoint_sha256": digest,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(_checkpoint_index(checkpoint), 17)

    def test_progress_is_not_reported_as_completed_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pkl"
            checkpoint.write_bytes(b"durable checkpoint bytes")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            (root / "checkpoint.metadata.json").write_text(
                json.dumps(
                    {
                        "atomic_complete": True,
                        "next_index": 17,
                        "checkpoint_sha256": digest,
                    }
                ),
                encoding="utf-8",
            )
            (root / "progress.jsonl").write_text(
                json.dumps({"current_index": 99}) + "\n", encoding="utf-8"
            )
            result = root / "result.json"
            failure = root / "failure.json"
            exit_code = run_logged_process(
                [sys.executable, "-c", "raise RuntimeError('stop')"],
                result=result,
                failure=failure,
                checkpoint=checkpoint,
            )
            self.assertEqual(exit_code, 1)
            payload = json.loads(failure.read_text(encoding="utf-8"))
            self.assertEqual(payload["last_completed_index"], 17)
            self.assertEqual(payload["last_progress_index"], 99)

    def test_gzip_resume_trim_is_eof_valid_and_semantic_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.csv.gz"
            with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["actual_record_index", "segment_id", "value"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"actual_record_index": "0", "segment_id": "a", "value": "1"},
                        {"actual_record_index": "0", "segment_id": "a", "value": "1"},
                        {"actual_record_index": "1", "segment_id": "b", "value": "2"},
                        {"actual_record_index": "2", "segment_id": "c", "value": "3"},
                    ]
                )
            stats = trim_csv(path, 2)
            self.assertEqual(stats["rows_kept"], 2)
            with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["segment_id"] for row in rows], ["a", "b"])

    def test_prune_breadcrumb_hook_is_scalar_and_non_interfering(self) -> None:
        encoder = build_fig9_encoder(
            Fig9StrictConfig(
                weekday_columns=3,
                time_columns=4,
                passenger_columns=8,
                neurons_per_column=3,
                k=2,
                forgetting_threshold=65.0,
            )
        )
        plain = build_strict_model(encoder, Fig9StrictConfig(
            weekday_columns=3,
            time_columns=4,
            passenger_columns=8,
            neurons_per_column=3,
            k=2,
            forgetting_threshold=65.0,
        ))
        traced = pickle.loads(pickle.dumps(plain))
        plain._grow_segment(0, 0, {1: 0.1, 2: 0.2}, 0.3)
        traced._grow_segment(0, 0, {1: 0.1, 2: 0.2}, 0.3)
        events: list[dict[str, object]] = []
        traced.prune_diagnostic_callback = events.append
        plain._prune_neuron(plain.columns[0].neurons[0], target_column=0, neuron_index=0)
        traced._prune_neuron(traced.columns[0].neurons[0], target_column=0, neuron_index=0)
        self.assertEqual(model_long_term_fingerprint(plain), model_long_term_fingerprint(traced))
        self.assertEqual(model_rng_fingerprint(plain), model_rng_fingerprint(traced))
        self.assertGreaterEqual(len(events), 8)
        for event in events:
            self.assertTrue(all(not isinstance(value, (list, dict, set, tuple))
                                for key, value in event.items()
                                if key != "phase"))
        restored = pickle.loads(pickle.dumps(traced))
        self.assertIsNone(getattr(restored, "prune_diagnostic_callback", None))


if __name__ == "__main__":
    unittest.main()
