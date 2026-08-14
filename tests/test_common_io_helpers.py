from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from experiments.common.hashing import sha256_file
from experiments.common.jsonio import write_json, write_json_atomic
from scripts.compare_phase01_baselines import BEHAVIOR_FIELDS


class CommonHashingTests(unittest.TestCase):
    def test_sha256_file_matches_direct_digest_across_chunk_boundary(self) -> None:
        payload = (b"phase02-hash-fixture" * 60_000) + b"tail"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "payload.bin"
            path.write_bytes(payload)
            self.assertEqual(
                sha256_file(path),
                hashlib.sha256(payload).hexdigest(),
            )

    def test_phase01_comparison_covers_all_frozen_behavior_fields(self) -> None:
        self.assertEqual(
            BEHAVIOR_FIELDS,
            (
                "prediction_sha256",
                "metric",
                "scenario_summary",
                "SCIENCE_MODEL_SHA256",
                "FULL_STATE_SHA256",
                "RNG_SHA256",
                "LEARNING_RNG_SHA256",
                "DECODE_RNG_SHA256",
                "PREVIOUS_ACTIVE_SHA256",
                "PREVIOUS_WINNER_SHA256",
                "CANDIDATE_SHA256",
                "structure",
            ),
        )


class CommonJsonIoTests(unittest.TestCase):
    def test_write_json_preserves_strict_runner_bytes(self) -> None:
        payload = {"z": "中文", "a": [1, None, 2.5]}
        expected = json.dumps(payload, indent=2, sort_keys=True).replace(
            "\n", os.linesep
        ).encode("utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "payload.json"
            write_json(path, payload)
            self.assertEqual(path.read_bytes(), expected)

    def test_write_json_atomic_preserves_bytes_and_removes_temporary(self) -> None:
        payload = {"b": True, "a": 3}
        expected = json.dumps(payload, indent=2, sort_keys=True).replace(
            "\n", os.linesep
        ).encode("utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "payload.json"
            write_json_atomic(path, payload)
            self.assertEqual(path.read_bytes(), expected)
            self.assertFalse(path.with_name(path.name + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
