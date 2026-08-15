from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path

from experiments.fig9 import read_records
from experiments.fig9_strict_reproduction import (
    Fig9StrictConfig,
    build_fig9_encoder,
    build_strict_model,
    load_strict_checkpoint,
    model_long_term_fingerprint,
    model_rng_fingerprint,
    save_strict_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
OLD_CHECKPOINT = (
    ROOT
    / "results"
    / "fig9_diagnostics"
    / "independent_reference_resume_smoke_part1"
    / "checkpoint.pkl"
)
CURRENT_PAYLOAD_KEYS = {
    "L_match",
    "branch_provenance",
    "checkpoint_format",
    "common_prefix_hash",
    "config",
    "data_file_sha256",
    "data_path",
    "density_rows",
    "diagnostic_state",
    "encoder",
    "fingerprint",
    "git_commit_sha",
    "limit",
    "long_sequence_rows",
    "missing",
    "model",
    "next_index",
    "perturbation_split_timestamp",
    "predictions",
    "prefix_end_index",
    "prefix_end_timestamp",
    "raw_column_counts",
    "raw_event_counts",
    "recent_errors",
    "rows",
    "stream_label",
    "targets",
}
OLD_PAYLOAD_KEYS = CURRENT_PAYLOAD_KEYS - {"long_sequence_rows"}


class Phase05CheckpointCompatibilityTests(unittest.TestCase):
    def test_pre_refactor_checkpoint_loads_with_stable_model_path(self) -> None:
        if not OLD_CHECKPOINT.exists():
            self.skipTest("local pre-refactor checkpoint is not available")

        payload = load_strict_checkpoint(OLD_CHECKPOINT)

        self.assertEqual(payload["checkpoint_format"], "fig9-strict-v1")
        self.assertEqual(set(payload), OLD_PAYLOAD_KEYS)
        self.assertEqual(type(payload["model"]).__module__, "seqmem.model")

    def test_current_checkpoint_round_trip_is_exact_in_fresh_process(self) -> None:
        config = Fig9StrictConfig(warmup=6)
        encoder = build_fig9_encoder(config)
        model = build_strict_model(encoder, config)
        data_path = ROOT / "data" / "paper_nyc_taxi.csv"
        records = read_records(data_path, limit=12)

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pkl"
            save_strict_checkpoint(
                checkpoint,
                encoder=encoder,
                model=model,
                fingerprint={"phase": "phase05-checkpoint-test"},
                config=config,
                stream_label="original",
                data_path=data_path,
                records=records,
                limit=12,
                next_index=0,
                predictions=[],
                targets=[],
                rows=[],
                recent_errors=deque(),
                missing=0,
                raw_event_counts=[],
                raw_column_counts=[],
                density_rows=[],
            )
            payload = load_strict_checkpoint(checkpoint)
            expected = {
                "payload_keys": sorted(payload),
                "model_module": type(payload["model"]).__module__,
                "model_fingerprint": model_long_term_fingerprint(payload["model"]),
                "rng_fingerprint": model_rng_fingerprint(payload["model"]),
            }
            code = """
import json
import sys
from pathlib import Path
from experiments.fig9_strict_reproduction import (
    load_strict_checkpoint,
    model_long_term_fingerprint,
    model_rng_fingerprint,
)
payload = load_strict_checkpoint(Path(sys.argv[1]))
model = payload["model"]
print(json.dumps({
    "payload_keys": sorted(payload),
    "model_module": type(model).__module__,
    "model_fingerprint": model_long_term_fingerprint(model),
    "rng_fingerprint": model_rng_fingerprint(model),
}, sort_keys=True))
"""
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join(
                [str(ROOT / "src"), str(ROOT)]
            )
            completed = subprocess.run(
                [sys.executable, "-c", code, str(checkpoint)],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout), expected)
        self.assertEqual(set(expected["payload_keys"]), CURRENT_PAYLOAD_KEYS)
        self.assertEqual(expected["model_module"], "seqmem.model")


if __name__ == "__main__":
    unittest.main()
