"""Fig.9 strict 和 historical 路径共享的评价公式。"""

from __future__ import annotations

from collections.abc import Iterable
from math import sqrt
from statistics import median


def error_ratio(errors: Iterable[float], targets: Iterable[float]) -> float:
    """返回总误差与目标绝对值总和之比。"""

    absolute_error = sum(errors)
    target_scale = sum(abs(target) for target in targets)
    return absolute_error / target_scale if target_scale else 0.0


def mape(predictions: list[float], targets: list[float]) -> float:
    """Return the legacy Fig.9 ratio-of-sums percentage error.

    PAPER STATUS: REFERENCE-SUPPORTED.  Zhang names MAPE and cites [58] without
    restating the equation; [58] uses this same ratio of sums.  It differs from
    pointwise standard MAPE, which remains available below for metric audits.
    """

    absolute_error = sum(
        abs(prediction - target)
        for prediction, target in zip(predictions, targets)
    )
    return error_ratio((absolute_error,), targets)


def standard_mape(
    predictions: Iterable[float],
    targets: Iterable[float],
    *,
    zero_target_policy: str = "omit",
) -> float:
    """Return pointwise MAPE without changing the legacy ``mape`` helper.

    The paper-reference definition is a mean of per-sample percentage errors.
    Zero-valued targets have no finite percentage error, so the default audit
    policy omits those samples.  ``raise`` is available when a caller wants a
    strict data-quality check instead.
    """

    apes: list[float] = []
    for prediction, target in zip(predictions, targets):
        if target == 0.0:
            if zero_target_policy == "raise":
                raise ZeroDivisionError("standard MAPE received a zero target")
            if zero_target_policy == "omit":
                continue
            raise ValueError(f"unknown zero_target_policy: {zero_target_policy}")
        apes.append(abs(prediction - target) / abs(target))
    return sum(apes) / len(apes) if apes else 0.0


def wape(predictions: Iterable[float], targets: Iterable[float]) -> float:
    """Return weighted absolute percentage error (ratio of sums)."""

    paired = list(zip(predictions, targets))
    return error_ratio(
        (abs(prediction - target) for prediction, target in paired),
        (target for _, target in paired),
    )


def absolute_percentage_errors(
    predictions: Iterable[float],
    targets: Iterable[float],
) -> list[float]:
    """Return nonzero-target pointwise absolute percentage errors."""

    return [
        abs(prediction - target) / abs(target)
        for prediction, target in zip(predictions, targets)
        if target != 0.0
    ]


def mean_absolute_error(
    predictions: Iterable[float], targets: Iterable[float]
) -> float:
    """Return the mean absolute error for paired values."""

    errors = [
        abs(prediction - target)
        for prediction, target in zip(predictions, targets)
    ]
    return sum(errors) / len(errors) if errors else 0.0


def root_mean_squared_error(
    predictions: Iterable[float], targets: Iterable[float]
) -> float:
    """Return the root mean squared error for paired values."""

    squared = [
        (prediction - target) ** 2
        for prediction, target in zip(predictions, targets)
    ]
    return sqrt(sum(squared) / len(squared)) if squared else 0.0


def median_absolute_percentage_error(
    predictions: Iterable[float], targets: Iterable[float]
) -> float:
    """Return the median pointwise percentage error, omitting zero targets."""

    values = absolute_percentage_errors(predictions, targets)
    return median(values) if values else 0.0


def percentile_absolute_percentage_error(
    predictions: Iterable[float],
    targets: Iterable[float],
    percentile: float,
) -> float:
    """Return a linear-interpolated percentile of nonzero-target APE values."""

    values = sorted(absolute_percentage_errors(predictions, targets))
    if not values:
        return 0.0
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be between 0 and 1")
    if len(values) == 1:
        return values[0]
    position = percentile * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + fraction * (values[upper] - values[lower])


def reference_rolling_mape(
    errors: Iterable[float],
    target_scale: float,
) -> float:
    """返回参考实现 [58] 绘图代码使用的滚动误差。"""

    values = list(errors)
    if not values or target_scale == 0.0:
        return 0.0
    return sum(values) / len(values) / target_scale
