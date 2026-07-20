from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "diagnostics"))

from etth1_paper_snn import scores, seasonal_source_index


class ETTh1TransferTests(unittest.TestCase):
    def test_empty_model_coverage_has_defined_metrics(self) -> None:
        self.assertTrue(all(value != value for value in scores([], [])))

    def test_seasonal_source_never_uses_future_data(self) -> None:
        for horizon in (1, 24, 96, 192):
            source = seasonal_source_index(500, horizon)
            self.assertLessEqual(source, 500)
            self.assertEqual((source - (500 + horizon)) % 24, 0)


if __name__ == "__main__":
    unittest.main()
