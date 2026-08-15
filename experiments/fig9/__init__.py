"""Fig.9 正式实验共享的稳定、无补偿功能。

strict 入口只应从这个包读取数据、指标、输出和学习 helper。历史补偿、未来
上下文和投影诊断不在这里导出。
"""

from .data import TaxiRecord, read_records, record_values
from .learning import learn_actual_code
from .metrics import error_ratio, mape, reference_rolling_mape
from .outputs import (
    StrictStreamOutputPaths,
    plot_adaptation,
    strict_summary_paths,
    write_density_trace,
    write_initial_stream_artifacts,
    write_long_sequence_activity_trace,
    write_optional_stream_results,
    write_predictions,
    write_runtime_artifacts,
    write_stream_adaptation_plots,
    write_stream_metadata,
    write_stream_summary,
)

__all__ = [
    "TaxiRecord",
    "StrictStreamOutputPaths",
    "error_ratio",
    "learn_actual_code",
    "mape",
    "plot_adaptation",
    "read_records",
    "record_values",
    "reference_rolling_mape",
    "strict_summary_paths",
    "write_density_trace",
    "write_initial_stream_artifacts",
    "write_long_sequence_activity_trace",
    "write_optional_stream_results",
    "write_predictions",
    "write_runtime_artifacts",
    "write_stream_adaptation_plots",
    "write_stream_metadata",
    "write_stream_summary",
]
