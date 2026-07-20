from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "diagnostics"))

from etth1_paper_snn import seasonal_source_index
from weather_paper_snn import WeatherRecord, record_values


class WeatherTransferTests(unittest.TestCase):
    def test_record_values_use_ten_minute_slot(self) -> None:
        record = WeatherRecord(datetime(2020, 1, 1, 23, 50), 3.5)
        weekday, slot, temperature = record_values(record)
        self.assertEqual(weekday, 2.0)
        self.assertEqual(slot, 143.0)
        self.assertEqual(temperature, 3.5)

    def test_daily_seasonal_source_never_uses_future_data(self) -> None:
        for horizon in (1, 144, 288):
            source = seasonal_source_index(500, horizon, period=144)
            self.assertLessEqual(source, 500)
            self.assertEqual((source - (500 + horizon)) % 144, 0)


if __name__ == "__main__":
    unittest.main()
