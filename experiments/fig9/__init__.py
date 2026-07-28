"""Fig.9 正式实验共享的稳定、无补偿功能。

strict 入口只应从这个包读取数据、指标、输出和学习 helper。历史补偿、未来
上下文和投影诊断不在这里导出。
"""

from .data import TaxiRecord, read_records, record_values
from .learning import learn_actual_code
from .metrics import error_ratio, mape, reference_rolling_mape
from .outputs import plot_adaptation, write_predictions

__all__ = [
    "TaxiRecord",
    "error_ratio",
    "learn_actual_code",
    "mape",
    "plot_adaptation",
    "read_records",
    "record_values",
    "reference_rolling_mape",
    "write_predictions",
]
