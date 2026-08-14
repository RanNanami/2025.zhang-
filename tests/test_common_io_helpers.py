from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from experiments.common.hashing import sha256_file
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


if __name__ == "__main__":
    unittest.main()
