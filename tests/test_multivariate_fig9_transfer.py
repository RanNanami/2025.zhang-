from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "diagnostics"))

from benchmark_real_timeseries import SPECS
from multivariate_fig9_transfer import (
    build_encoder,
    make_record_values,
    select_feature_indices,
)
from multihorizon_fig9_transfer import parse_horizons


class MultivariateFig9TransferTests(unittest.TestCase):
    def test_selection_keeps_target_last(self) -> None:
        columns = ["a", "b", "target"]
        rows = [[float(i), float(i % 3), float(i + 1)] for i in range(12)]
        selected = select_feature_indices(columns, rows, "target", 10, 1, 2)
        self.assertEqual(selected[-1], 2)
        self.assertEqual(len(selected), 2)

    def test_etth1_record_contains_time_and_selected_fields(self) -> None:
        values = make_record_values(
            datetime(2020, 1, 1, 13),
            [1.0, 2.0, 3.0],
            [0, 2],
            SPECS["etth1"],
        )
        self.assertEqual(values, [2.0, 13.0, 1.0, 3.0])

    def test_eight_numeric_fields_use_976_columns(self) -> None:
        rows = [[float(row + column) for column in range(8)] for row in range(12)]
        encoder, target = build_encoder(
            rows,
            list(range(8)),
            target_index=7,
            warmup=10,
            seasonal_period=24,
            feature_columns=58,
            target_columns=482,
            active_columns=10,
        )
        self.assertEqual(encoder.num_columns, 976)
        self.assertEqual(target.num_columns, 482)
        self.assertEqual(encoder.k, 100)

    def test_multihorizon_parser_sorts_and_deduplicates(self) -> None:
        self.assertEqual(parse_horizons("96,12,24,12,48"), (12, 24, 48, 96))


if __name__ == "__main__":
    unittest.main()
