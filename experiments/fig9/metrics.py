"""Fig.9 strict 和 historical 路径共享的评价公式。"""

from __future__ import annotations

from collections.abc import Iterable


def error_ratio(errors: Iterable[float], targets: Iterable[float]) -> float:
    """返回总误差与目标绝对值总和之比。"""

    absolute_error = sum(errors)
    target_scale = sum(abs(target) for target in targets)
    return absolute_error / target_scale if target_scale else 0.0


def mape(predictions: list[float], targets: list[float]) -> float:
    """返回参考实现 [58] 使用的全局目标归一化误差。"""

    absolute_error = sum(
        abs(prediction - target)
        for prediction, target in zip(predictions, targets)
    )
    return error_ratio((absolute_error,), targets)


def reference_rolling_mape(
    errors: Iterable[float],
    target_scale: float,
) -> float:
    """返回参考实现 [58] 绘图代码使用的滚动误差。"""

    values = list(errors)
    if not values or target_scale == 0.0:
        return 0.0
    return sum(values) / len(values) / target_scale
