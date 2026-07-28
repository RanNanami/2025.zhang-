"""Fig.9 出租车数据的读取和三字段转换。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class TaxiRecord:
    """一条半小时粒度的出租车客流记录。"""

    timestamp: datetime
    value: float


def read_records(path: Path, limit: int) -> list[TaxiRecord]:
    """读取 Fig.9 使用的时间戳和 passenger count。

    NuPIC 原始文件开头包含两行类型/标志元数据；它们无法解析为数值记录，
    因此保持旧实现行为并跳过。函数不修改模型状态，也不使用 RNG。
    """

    records: list[TaxiRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value_text = row.get("passenger_count") or row.get("value")
            try:
                timestamp = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                value = float(value_text)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            records.append(TaxiRecord(timestamp=timestamp, value=value))
            if limit > 0 and len(records) >= limit:
                break
    return records


def record_values(record: TaxiRecord) -> tuple[float, float, float]:
    """转换为 weekday、半小时槽位和 passenger value 三个字段。"""

    slot = record.timestamp.hour * 2 + record.timestamp.minute // 30
    return float(record.timestamp.weekday()), float(slot), record.value
