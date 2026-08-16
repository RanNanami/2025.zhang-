"""论文 SSTD 编码和解码实现。

离散、连续、周期和多字段编码器都输出按脉冲时间排序的 SymbolCode。编码器
负责符号到 column/time event 的映射，不负责推进 SequentialMemory 状态；
解码只读取预测代码，不应把解码值重新作为 proximal input。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class SpikeEvent:
    column: int
    time: float


@dataclass(frozen=True)
class SymbolCode:
    """A symbol represented by ordered mini-column spikes.

    In the paper's SSTD encoding, a discrete symbol is encoded by K selected
    mini-columns. The mini-columns fire in a fixed order during the first half
    of an oscillation cycle.
    """

    events: tuple[SpikeEvent, ...]

    @property
    def columns(self) -> tuple[int, ...]:
        return tuple(event.column for event in self.events)


class SSTDEncoder(Protocol):
    num_columns: int
    k: int

    @property
    def event_times(self) -> tuple[float, ...]: ...


class SSTDDiscreteEncoder:
    """Sparse spatiotemporal distributed encoder for discrete symbols.

    中文调试提示：一个 symbol 会被固定映射成 K 个 mini-column/time event。
    第一次 encode(symbol) 会消耗 encoder RNG；之后同一个 symbol 会复用缓存。
    """

    def __init__(self, num_columns: int, k: int, seed: int = 0) -> None:
        if not 0 < k <= num_columns:
            raise ValueError("k must be in the range [1, num_columns].")
        self.num_columns = num_columns
        self.k = k
        self._rng = random.Random(seed)
        self._codes: dict[str, SymbolCode] = {}
        self._symbols_by_event: dict[tuple[int, float], set[str]] = {}

    def encode(self, symbol: str) -> SymbolCode:
        """Assign one paper SSTD mini-column/time code to a discrete symbol.

        Inputs/outputs: symbol -> cached ``SymbolCode``.  State mutation: first
        encounter consumes encoder RNG and stores the code.  Paper status: the
        random K-column ordered representation matches Section II-C3.
        """

        # PAPER STATUS: MATCH
        # The paper specifies a fixed random ordered K-column code per symbol;
        # the seed and first-encounter order are reproducibility choices.
        # DEBUG WATCH: 若复现实验不稳定，检查这里是否在不同顺序下首次遇到
        # 新 symbol；首次编码顺序会决定离散词的随机 SSTD code。
        if symbol not in self._codes:
            columns = tuple(self._rng.sample(range(self.num_columns), self.k))
            if self.k == 1:
                times = (0.0,)
            else:
                times = tuple(0.5 * i / (self.k - 1) for i in range(self.k))
            self._codes[symbol] = SymbolCode(
                events=tuple(
                    SpikeEvent(column=column, time=time)
                    for column, time in zip(columns, times)
                )
            )
            for event in self._codes[symbol].events:
                self._symbols_by_event.setdefault((event.column, event.time), set()).add(
                    symbol
                )
        return self._codes[symbol]

    @property
    def event_times(self) -> tuple[float, ...]:
        return _event_times(self.k)

    def known_symbols(self) -> list[str]:
        return sorted(self._codes)

    def symbols_for_event(self, column: int, time: float) -> set[str]:
        """Return symbols containing an exact SSTD column/time event."""

        return self._symbols_by_event.get((column, time), set())


def _event_times(k: int) -> tuple[float, ...]:
    if k == 1:
        return (0.0,)
    return tuple(0.5 * index / (k - 1) for index in range(k))


def _ordered_code(columns: list[int]) -> SymbolCode:
    times = _event_times(len(columns))
    return SymbolCode(
        events=tuple(
            SpikeEvent(column=column, time=time)
            for column, time in zip(columns, times)
        )
    )


class SSTDRealValueEncoder:
    """Gaussian population encoder for real values as described in Fig. 3(a).

    中文调试提示：Fig.9 的 passenger_count 使用 482 列、K=10。encode()
    取响应最高的 10 个中心列；decode_likelihood() 再从预测列/时间中寻找
    最可能的合法人口数编码。
    """

    def __init__(
        self,
        num_columns: int,
        k: int,
        minimum: float,
        maximum: float,
        sigma: float | None = None,
        column_offset: int = 0,
    ) -> None:
        if num_columns < 2:
            raise ValueError("num_columns must be at least 2.")
        if not 0 < k <= num_columns:
            raise ValueError("k must be in the range [1, num_columns].")
        if maximum <= minimum:
            raise ValueError("maximum must be greater than minimum.")
        self.num_columns = num_columns
        self.k = k
        self.minimum = minimum
        self.maximum = maximum
        self.column_offset = column_offset
        # PAPER GAP: LOCAL IMPLEMENTATION
        # Section II-C1 specifies Gaussian fields spaced by l and resolution
        # l/2, but not Fig.9's numeric range, endpoint placement, or sigma.
        self.spacing = (maximum - minimum) / (num_columns - 1)
        self.sigma = sigma if sigma is not None else self.spacing
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.centers = tuple(
            minimum + index * self.spacing for index in range(num_columns)
        )
        self._likelihood_grid: tuple[tuple[float, SymbolCode], ...] | None = None
        self._likelihood_codes: tuple[SymbolCode, ...] | None = None
        self._codes_by_event: dict[tuple[int, float], tuple[SymbolCode, ...]] | None = None

    def encode(self, value: float) -> SymbolCode:
        """Encode a real value with the paper Gaussian Top-K SSTD mechanism.

        Inputs/outputs: scalar -> ordered ``SymbolCode``.  State mutation: none.
        Paper uncertainty: clipping, centers, range, sigma, and tie behavior are
        local choices where the public paper is underspecified.
        """

        # DEBUG WATCH: passenger 会先被截断到 [minimum, maximum]，再按高斯响应
        # 选列。column_offset 保证 passenger 列从 88 开始，不与 weekday/time 混淆。
        clipped = min(self.maximum, max(self.minimum, float(value)))
        ranked = sorted(
            range(self.num_columns),
            key=lambda index: (
                self._response(clipped, self.centers[index]),
                -index,
            ),
            reverse=True,
        )[: self.k]
        return _ordered_code([self.column_offset + index for index in ranked])

    @property
    def event_times(self) -> tuple[float, ...]:
        return _event_times(self.k)

    def decode(self, code: SymbolCode) -> float:
        local_columns = [
            event.column - self.column_offset
            for event in sorted(code.events, key=lambda item: item.time)
            if self.column_offset <= event.column < self.column_offset + self.num_columns
        ][: self.k]
        if not local_columns:
            raise ValueError("code contains no columns from this encoder.")
        # Earlier spikes correspond to stronger Gaussian responses.
        weights = [len(local_columns) - rank for rank in range(len(local_columns))]
        return sum(self.centers[column] * weight for column, weight in zip(local_columns, weights)) / sum(weights)

    def decode_likelihood(
        self,
        code: SymbolCode,
        timing_tolerance: float = 0.03,
    ) -> float:
        """Decode a raw prediction through the local likelihood codebook.

        Inputs/outputs: predicted code -> scalar.  State mutation: lazy readout
        caches only.  Paper uncertainty: half-spacing search, overlap priority,
        timing tolerance, and averaging tied values are not specified.
        """

        # PAPER GAP: LOCAL IMPLEMENTATION
        # The paper says "most likely prediction" but does not publish this
        # decoder.  DEBUG WATCH: dense raw codes create many tied best_values.
        predicted_times: dict[int, list[float]] = {}
        for event in code.events:
            if self.column_offset <= event.column < self.column_offset + self.num_columns:
                predicted_times.setdefault(event.column, []).append(event.time)
        if not predicted_times:
            raise ValueError("code contains no columns from this encoder.")

        grid = self.likelihood_grid()
        # DEBUG WATCH: best_score=(timed_overlap, column_overlap)。timed overlap
        # 优先，column overlap 次之；若 timed 全低，说明可能是 timing mismatch。
        best_score = (-1, -1)
        best_values: list[float] = []
        for value, candidate_code in grid:
            column_overlap = sum(
                event.column in predicted_times for event in candidate_code.events
            )
            timed_overlap = sum(
                event.column in predicted_times
                and any(
                    abs(predicted_time - event.time) <= timing_tolerance
                    for predicted_time in predicted_times[event.column]
                )
                for event in candidate_code.events
            )
            score = (timed_overlap, column_overlap)
            if score > best_score:
                best_score = score
                best_values = [value]
            elif score == best_score:
                best_values.append(value)
        return sum(best_values) / len(best_values)

    def likelihood_grid(self) -> tuple[tuple[float, SymbolCode], ...]:
        """Return candidate values used by likelihood decoding.

        中文调试提示：这是 passenger decode 的搜索网格；它不是训练数据，也不会
        把预测值重新编码送回网络。
        """

        if self._likelihood_grid is None:
            self._likelihood_grid = tuple(
                (value, self.encode(value)) for value in self._likelihood_values()
            )
        return self._likelihood_grid

    def likelihood_codes(self) -> tuple[SymbolCode, ...]:
        """Return each legal ordered population code once."""

        if self._likelihood_codes is None:
            unique: dict[tuple[SpikeEvent, ...], SymbolCode] = {}
            for _value, code in self.likelihood_grid():
                unique.setdefault(code.events, code)
            self._likelihood_codes = tuple(unique.values())
        return self._likelihood_codes

    def likelihood_codes_for_events(
        self, events: set[tuple[int, float]]
    ) -> tuple[SymbolCode, ...]:
        """Return legal codes sharing at least one exact column/time event."""

        if self._codes_by_event is None:
            mutable: dict[tuple[int, float], list[SymbolCode]] = {}
            for code in self.likelihood_codes():
                for event in code.events:
                    mutable.setdefault(
                        (event.column, round(event.time, 12)), []
                    ).append(code)
            self._codes_by_event = {
                event: tuple(codes) for event, codes in mutable.items()
            }
        selected: dict[tuple[SpikeEvent, ...], SymbolCode] = {}
        for event in events:
            for code in self._codes_by_event.get(event, ()):
                selected.setdefault(code.events, code)
        return tuple(selected.values())

    def _likelihood_values(self) -> tuple[float, ...]:
        return tuple(
            self.minimum + index * self.spacing / 2.0
            for index in range(2 * (self.num_columns - 1) + 1)
        )

    def _response(self, value: float, center: float) -> float:
        distance = value - center
        return math.exp(-(distance * distance) / (2.0 * self.sigma * self.sigma))


class SSTDPeriodicEncoder(SSTDRealValueEncoder):
    """Circular Gaussian population encoder for periodic values (Fig. 3(b)).

    中文调试提示：Fig.9 中 weekday 用 period=7，time slot 用 period=48。
    周期编码会让周日/周一、23:30/00:00 在列空间中相邻。
    """

    def __init__(
        self,
        num_columns: int,
        k: int,
        period: float,
        origin: float = 0.0,
        sigma: float | None = None,
        column_offset: int = 0,
    ) -> None:
        if period <= 0:
            raise ValueError("period must be positive.")
        self.period = period
        super().__init__(
            num_columns=num_columns,
            k=k,
            minimum=origin,
            maximum=origin + period,
            sigma=sigma if sigma is not None else period / num_columns,
            column_offset=column_offset,
        )
        self.spacing = period / num_columns
        self.centers = tuple(origin + index * self.spacing for index in range(num_columns))

    def encode(self, value: float) -> SymbolCode:
        wrapped = self.minimum + ((float(value) - self.minimum) % self.period)
        ranked = sorted(
            range(self.num_columns),
            key=lambda index: (
                self._response(wrapped, self.centers[index]),
                -index,
            ),
            reverse=True,
        )[: self.k]
        return _ordered_code([self.column_offset + index for index in ranked])

    def _response(self, value: float, center: float) -> float:
        direct = abs(value - center)
        distance = min(direct, self.period - direct)
        return math.exp(-(distance * distance) / (2.0 * self.sigma * self.sigma))

    def _likelihood_values(self) -> tuple[float, ...]:
        # Include the final midpoint across the circular last/first boundary.
        return tuple(
            self.minimum + index * self.spacing / 2.0
            for index in range(2 * self.num_columns)
        )

    def decode(self, code: SymbolCode) -> float:
        local_columns = [
            event.column - self.column_offset
            for event in sorted(code.events, key=lambda item: item.time)
            if self.column_offset <= event.column < self.column_offset + self.num_columns
        ][: self.k]
        if not local_columns:
            raise ValueError("code contains no columns from this encoder.")
        weights = [len(local_columns) - rank for rank in range(len(local_columns))]
        angles = [
            2.0 * math.pi * (self.centers[column] - self.minimum) / self.period
            for column in local_columns
        ]
        x = sum(weight * math.cos(angle) for weight, angle in zip(weights, angles))
        y = sum(weight * math.sin(angle) for weight, angle in zip(weights, angles))
        angle = math.atan2(y, x) % (2.0 * math.pi)
        return self.minimum + self.period * angle / (2.0 * math.pi)


class SSTDCompositeEncoder:
    """Combine disjoint SSTD fields into one simultaneous record code.

    中文调试提示：Fig.9 一个 record = weekday + time slot + passenger。
    三个字段各自 K=10，因此 composite code 共 30 个 event；column_offset
    是定位字段串扰和 raw-column explosion 的第一处断点。
    """

    def __init__(self, encoders: Sequence[SSTDEncoder]) -> None:
        if not encoders:
            raise ValueError("at least one field encoder is required.")
        self.encoders = tuple(encoders)
        self.num_columns = max(
            getattr(encoder, "column_offset", 0) + encoder.num_columns
            for encoder in self.encoders
        )
        self.k = sum(encoder.k for encoder in self.encoders)

    @property
    def event_times(self) -> tuple[float, ...]:
        return tuple(sorted({time for encoder in self.encoders for time in encoder.event_times}))

    def encode(self, values: Sequence[float]) -> SymbolCode:
        # DEBUG WATCH: 如果 len(events) 不是三个字段 K 之和，先检查 values
        # 数量或某个 field encoder 是否被错误替换。
        if len(values) != len(self.encoders):
            raise ValueError("one value is required for each field encoder.")
        events = tuple(
            event
            for encoder, value in zip(self.encoders, values)
            for event in encoder.encode(value).events  # type: ignore[attr-defined]
        )
        return SymbolCode(events=events)
