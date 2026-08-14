from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from scripts.capture_phase01_baselines import stable_sha256


ROOT = Path(__file__).resolve().parents[1]
BASELINES = (
    ROOT
    / "reports/refactor/phase01_20260814_233129/baselines/pre_refactor"
)


class Phase01BaselineManifestTests(unittest.TestCase):
    def test_canonical_hash_is_independent_of_dictionary_order(self) -> None:
        left = {"a": 1, "b": {"x": 2, "y": 3}}
        right = {"b": {"y": 3, "x": 2}, "a": 1}
        self.assertEqual(stable_sha256(left), stable_sha256(right))

    def test_all_fixtures_are_present_and_explicitly_nonpaper(self) -> None:
        manifest = json.loads(
            (BASELINES / "baseline_manifest.json").read_text(encoding="utf-8")
        )
        runs = {row["name"]: row for row in manifest["runs"]}
        self.assertEqual(
            set(runs),
            {"fig8_10", "fig8_20", "fig8_50", "fig9_10", "fig9_20", "fig9_50"},
        )
        for name, run in runs.items():
            self.assertEqual(run["protocol"]["classification"], "REGRESSION_FIXTURE")
            self.assertFalse(run["protocol"]["paper_result"])
            if name.startswith("fig9_"):
                self.assertTrue(run["protocol"]["reduced_warmup"])

    def test_prediction_files_match_recorded_sha256(self) -> None:
        manifest = json.loads(
            (BASELINES / "baseline_manifest.json").read_text(encoding="utf-8")
        )
        for run in manifest["runs"]:
            path = BASELINES / run["name"] / "predictions.csv"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, run["prediction_sha256"], run["name"])

    def test_repeat_checks_cover_every_fixture(self) -> None:
        rows = json.loads(
            (BASELINES / "reproducibility_check.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["attempts"] == 2 for row in rows))
        self.assertTrue(all(row["all_behavior_fingerprints_equal"] for row in rows))


if __name__ == "__main__":
    unittest.main()
