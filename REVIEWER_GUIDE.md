# Reviewer 阅读指南

这是第一次阅读本仓库时唯一需要先看的文件。建议先根据这里的代码导航理解模型，再决定哪些诊断和历史结果值得继续追踪。当前 strict 实现包含论文明确的机制，也包含论文没有公开的本地数值选择；不要把 `strict` 理解成已经证明与作者代码逐式相同。

> **重要术语说明**
>
> 本仓库中的 `strict` 指“受论文约束、没有额外补偿的当前实现”。它不表示“与作者代码完全相同”，也不表示所有论文未公开的实现细节都已知。论文没有充分指定的部分会明确标注为 paper gap。

## 模型主流程

1. **编码**：输入被编码为 `SymbolCode`，其中每个 `SpikeEvent` 保存 mini-column 和半周期内的时间顺序。
2. **预测**：`SequentialMemory.predict_code()` 根据上一周期的 context cells 找到可达的 distal segment。
3. 每个 segment 根据突触 delay 和双指数 PSP 计算 dendritic threshold crossing。
4. crossing 触发 predictive/depolarized state；phase-precessed soma 轨迹给出预测 spike time。
5. 同一事件和同一 mini-column 的候选被收缩成输出 `SymbolCode`，原始 `PredictionCandidate` 同时保留下来。
6. **观察**：`observe_code()` 接收当前真实的 proximal SSTD 输入，并按列和时间与刚才的候选匹配。
7. **Scenario 1**：真实输入确认预测细胞，强化引发该预测的 segment，并处理非贡献突触。
8. **Scenario 2**：没有提前预测成功，但存在达到 `L_match` 的 segment，强化旧 segment 并补长缺失的 winner synapses。
9. **Scenario 2B / 新建 segment**：没有合格 segment 时，在 least-used neuron 上创建 segment；代码的 legacy counter 把该分支记作 `scenario3`，不要与论文中的 Scenario 3 punishment 混淆。
10. **发布状态**：真实 predicted cells 或整列 burst cells 写入 `previous_active_cells`；学习 winner 写入 `previous_winners`。
11. **自主预测**：Fig.8/Fig.9 的 raw rollout 使用 `advance_prediction()` 直接推进真实预测细胞，不把 decoder 输出重新编码。
12. **遗忘**：强化、减弱和 age 更新后，`_prune_neuron()` 根据 weight/age forgetting score 删除失效 synapses/segments。

## 论文概念 → 代码位置

| 论文概念 | 主要代码位置 | 阅读提示 |
|---|---|---|
| SSTD 离散编码 | `src/seqmem/encoding.py::SSTDDiscreteEncoder.encode` | 固定随机列集合和半周期 spike order。 |
| SSTD 实数编码 | `src/seqmem/encoding.py::SSTDRealValueEncoder.encode` | Gaussian centers/range/sigma 含有本地选择。 |
| Eq.(1) PSP kernel | `src/seqmem/dynamics.py::spike_response` | `V0` 由 `response_scale/kernel_scale` 实现；unit-peak normalization 不是论文确认参数。 |
| Dendritic potential | `src/seqmem/dynamics.py::dendritic_potential`；`src/seqmem/model.py::reference_continuous_prediction` | 论文中的 dendritic potential 实际对应 Eq.(1)；Eq.(2) 实际是 oscillation。 |
| Eq.(2) oscillation | `src/seqmem/dynamics.py::DSNeuronState.oscillation` | predictive phase reset 是部分对齐的本地补全。 |
| Eq.(3) soma potential | `src/seqmem/dynamics.py::DSNeuronState.membrane_potential` | `V_inh(t)` 的实现不完整；公开论文也没有给出完整数值轨迹，等价性未证实。 |
| phase precession | `src/seqmem/dynamics.py::DSNeuronState.oscillation` | 当前实现使用 crossing 后的 `-A*cos(...)`。 |
| prediction | `src/seqmem/model.py::SequentialMemory.predict_code` | 主入口；candidate、crossing、event selection 都从这里串起。 |
| intracolumn winner | `src/seqmem/model.py::SequentialMemory.predict_code` 中的 `winner_by_column` block | strict 取 earliest firing；与论文 largest-soma winner 的等价性未证实。 |
| burst context | `src/seqmem/model.py::SequentialMemory._active_sources`；`observe_code` 的 burst block | all-neuron burst 是论文明确的；整列如何作为完整下一步 context 传播仍未充分指定。 |
| Scenario 1 | `src/seqmem/model.py::SequentialMemory.scenario1_contributing_sources`；`_reinforce_segment` | contributor 判定含有本地 arrival window。 |
| Scenario 2A：matching | `src/seqmem/model.py::SequentialMemory._best_matching_neuron` | `timed_overlap >= L_match` 决定是否复用旧 segment。 |
| Scenario 2B：segment creation | `src/seqmem/model.py::SequentialMemory._grow_segment` | 从 previous winners 建突触；delay 由 `_new_synapse_delay` 计算。 |
| Scenario 3 punishment | `src/seqmem/model.py::SequentialMemory._punish_wrong_predictions` | 未兑现预测的 contributing synapses 被减权。 |
| synaptic delay | `src/seqmem/model.py::SequentialMemory._new_synapse_delay` | strict `current-delay` 是论文公式的本地数值实现。 |
| forgetting | `src/seqmem/model.py::SequentialMemory._prune_neuron`；`src/seqmem/_forgetting_helpers.py` | 根据 weight、age 和 experiment-specific threshold。 |
| Fig.8 protocol | `experiments/fig8_sentence_memory.py::run`；`recall_suffix` | 10 words、6-word cue、4-word raw-neural recall。 |
| Fig.9 protocol | `experiments/fig9_strict_reproduction.py::run_strict_stream`；`rollout_raw_autonomous` | 5-step raw rollout；warmup/evaluation 定义含有本地选择。 |
| decoder | `src/seqmem/encoding.py::SSTDRealValueEncoder.decode_likelihood` | half-spacing grid、timed overlap 和 tie averaging 没有由论文完整指定。 |
| Fig.9 error metric | `experiments/fig9/metrics.py::mape` | ratio-of-sums 不同于普通 pointwise MAPE，但有 Zhang 引用的 reference [58] 支持。 |

## 第一次 Review 只看这 12 个函数

1. `SSTDDiscreteEncoder.encode` — `src/seqmem/encoding.py`：理解 symbol 如何变成 ordered mini-columns。
2. `SSTDRealValueEncoder.encode` — `src/seqmem/encoding.py`：理解 Fig.9 passenger Gaussian population code。
3. `spike_response` — `src/seqmem/dynamics.py`：核对 Eq.(1) 的 PSP、`V0` 和 time constants。
4. `DSNeuronState.membrane_potential` — `src/seqmem/dynamics.py`：核对 Eq.(3)、phase precession、refractory 和不完整的 inhibition。
5. `SequentialMemory.predict_code` — `src/seqmem/model.py`：沿 active context → segment → candidate → event winner 阅读。
6. `SequentialMemory.observe_code` — `src/seqmem/model.py`：沿真实输入 → Scenario branch → burst → publish state 阅读。
7. `SequentialMemory._best_matching_neuron` — `src/seqmem/model.py`：检查 Scenario 2 的 `L_match` 和 timing eligibility。
8. `SequentialMemory._grow_segment` — `src/seqmem/model.py`：检查 Scenario 2B 的 segment、synapse 和 delay 创建。
9. `SequentialMemory.scenario1_contributing_sources` — `src/seqmem/model.py`：检查“贡献突触”的本地定义。
10. `SequentialMemory._reinforce_segment` — `src/seqmem/model.py`：检查 strengthen/rejuvenate/weaken/age 的真实写入。
11. `SequentialMemory._punish_wrong_predictions` — `src/seqmem/model.py`：检查未兑现预测如何被处罚。
12. `SequentialMemory._prune_neuron` — `src/seqmem/model.py`：检查 forgetting 如何真正删除长期结构。

之后再从协议入口阅读 `fig8_sentence_memory.py::recall_suffix` 和 `fig9_strict_reproduction.py::rollout_raw_autonomous`，确认调用方没有重编码预测或读取未来真值。

## FIRST REVIEW HOTSPOTS

| 重点 | 论文位置 | 代码函数 | 当前状态 |
|---|---|---|---|
| `kernel_scale` / `V0` / `tau_m` / `tau_s` | Sec. II-B1, Eq.(1) | `DSDynamicsParams.kernel_scale`；`spike_response`；`MemoryParams` | **本地实现（LOCAL IMPLEMENTATION）** |
| `V_inh(t)` | Sec. II-B5, Eq.(3) | `DSNeuronState.membrane_potential` | **实现不完整 / 等价性未证实** |
| intracolumn winner semantics | Sec. II-B5 | `SequentialMemory.predict_code` | **等价性未证实** |
| `timing_tolerance=.03` | Sec. II-D Scenario matching | `MemoryParams.timing_tolerance` 及其调用位置 | **高影响本地语义选择** |
| Scenario 1 contributor semantics | Sec. II-D1, Scenario 1 | `scenario1_contributing_sources` | **本地实现 / 操作化定义** |
| synaptic delay realization | Sec. II-D1, Scenario 2 | `_new_synapse_delay` | **部分对齐** |

## 第一次 Review 可以忽略

- `experiments/diagnostics/` 和 `src/seqmem/*diagnostic*`：trace、join、forensics 与 analyzer。
- `runtime_debug_callback`、native crash breadcrumbs、faulthandler 和进程监督代码。
- CSV/JSON/PNG/ZIP、checkpoint sidecar、artifact packaging 和报告生成 I/O。
- `experiments/historical/`：兼容旧结果，不是 strict 主路径。
- `results/`、`reports/refactor/` 和历史运行报告：用于证据追溯，不用于第一遍理解模型。
- oracle、teacher-forced、competition、alternative selector、L-match ablation 等显式 diagnostic modes。
- `optimized_v1/optimized_v2` continuous paths：等价性对照，strict 默认仍为 `reference`。

## Review 输出

逐项核对时使用 `reports/reviewer/PAPER_CODE_REVIEW_CHECKLIST.md`。已确认的高风险差距见 `reports/paper_code_diff/HIGH_IMPACT_SCIENTIFIC_GAPS.md`；它们是 review 入口，不代表本轮已经修复。
