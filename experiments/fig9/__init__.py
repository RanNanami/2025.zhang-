"""Fig.9 正式实验共享的稳定、无补偿功能。

strict 入口只应从这个包读取数据、指标、输出和学习 helper。历史补偿、未来
上下文和投影诊断不在这里导出。
"""

from .data import TaxiRecord, read_records, record_values

__all__ = [
    "TaxiRecord",
    "read_records",
    "record_values",
]
