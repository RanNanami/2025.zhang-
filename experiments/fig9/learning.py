"""Fig.9 当前真实记录的在线学习入口。"""

from __future__ import annotations

from seqmem.encoding import SymbolCode
from seqmem.model import ObservationTrace, SequentialMemory


def learn_actual_code(
    model: SequentialMemory,
    code: SymbolCode,
    *,
    observation_trace: ObservationTrace | None = None,
) -> None:
    """先预测，再按论文 Scenario 1/2/3 学习真实记录。

    状态影响
    --------
    - `predict_code` 更新本周期预测候选和 transient prediction state。
    - `observe_code(..., learn=True)` 可能修改 segment、synapse、权重和 age。
    - 两次调用的先后顺序属于 strict protocol，不得交换或合并。
    """

    # STRICT PROTOCOL: distal prediction must exist before learning chooses
    # Scenario 1, 2, or 3 for the actual proximal code.
    model.predict_code()
    if observation_trace is None:
        model.observe_code(code, learn=True)
    else:
        model.observe_code(
            code,
            learn=True,
            observation_trace=observation_trace,
        )
