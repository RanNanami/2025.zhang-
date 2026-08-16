# Reviewer Guide

这是第一次阅读本仓库时唯一需要先看的文件。目标是先找到论文机制对应的代码，再决定哪些诊断和历史结果值得继续追踪。当前 strict 实现包含已公开机制，也包含论文未公开的本地数值选择；不要把 `strict` 理解为已经证明逐式等价。

> **IMPORTANT TERMINOLOGY**
>
> 本仓库的 `strict` 指 **paper-constrained、non-compensated current implementation**。它不表示 author-code-identical，也不表示所有未公开实现细节已经获知。论文未充分指定的部分会明确标为 paper gap。

## 模型主流程

1. **encode**：输入被编码为 `SymbolCode`，其中每个 `SpikeEvent` 保存 mini-column 和半周期内的时间顺序。
2. **predict**：`SequentialMemory.predict_code()` 从上一周期 context cells 找到可达 distal segments。
3. 每个 segment 按突触 delay 和双指数 PSP 计算 dendritic threshold crossing。
4. crossing 触发 predictive/depolarized state；phase-precessed soma 轨迹给出预测 spike time。
5. 同一事件和同一 mini-column 的候选被收缩为输出 `SymbolCode`，原始 `PredictionCandidate` 同时保留。
6. **observe**：`observe_code()` 接收当前真实 proximal SSTD 输入，并与刚才的候选按列和时间匹配。
7. **Scenario 1**：真实输入确认预测细胞，强化引发该预测的 segment，并处理非贡献突触。
8. **Scenario 2**：未被预测但存在 `L_match` 合格 segment，强化旧 segment 并补长缺失 winner synapses。
9. **Scenario 2B / new segment**：没有合格 segment 时，在 least-used neuron 上创建 segment；代码的 legacy counter 把该分支记作 `scenario3`，不要与论文 Scenario 3 punishment 混淆。
10. **publish state**：真实 predicted cells 或整列 burst cells 写入 `previous_active_cells`；学习 winner 写入 `previous_winners`。
11. **autonomous prediction**：Fig.8/Fig.9 raw rollout 用 `advance_prediction()` 将真实预测细胞直接推进，不重编码 decoder 输出。
12. **forgetting**：强化、减弱和 age 更新后，`_prune_neuron()` 按 weight/age forgetting score 删除失效 synapses/segments。

## Paper Concept -> Code Location

| Paper concept | Primary code location | Review note |
|---|---|---|
| SSTD discrete encoding | `src/seqmem/encoding.py::SSTDDiscreteEncoder.encode` | 固定随机列集合和半周期 spike order。 |
| SSTD real encoding | `src/seqmem/encoding.py::SSTDRealValueEncoder.encode` | Gaussian centers/range/sigma 含本地选择。 |
| Eq.(1) PSP kernel | `src/seqmem/dynamics.py::spike_response` | `V0` 被 `response_scale/kernel_scale` 实现；unit-peak normalization 非论文确认参数。 |
| Dendritic potential | `src/seqmem/dynamics.py::dendritic_potential`; `src/seqmem/model.py::reference_continuous_prediction` | 论文中 dendritic potential 是 Eq.(1)；Eq.(2) 实际是 oscillation。 |
| Eq.(2) oscillation | `src/seqmem/dynamics.py::DSNeuronState.oscillation` | predictive phase reset 是部分对齐、本地补全。 |
| Eq.(3) soma potential | `src/seqmem/dynamics.py::DSNeuronState.membrane_potential` | `V_inh(t)` 实现不完整；公开论文也未给数值轨迹，exact equivalence 未证明。 |
| phase precession | `src/seqmem/dynamics.py::DSNeuronState.oscillation` | 当前使用 crossing 后 `-A*cos(...)`。 |
| prediction | `src/seqmem/model.py::SequentialMemory.predict_code` | 主入口；候选、crossing、event selection 都从这里串起。 |
| intracolumn winner | `src/seqmem/model.py::SequentialMemory.predict_code` 的 `winner_by_column` block | strict 取 earliest firing；与 paper largest-soma winner 的等价性未证明。 |
| burst context | `src/seqmem/model.py::SequentialMemory._active_sources`; `observe_code` burst block | all-neuron burst 是 paper-explicit；整列如何作为完整下一步 context 传播仍未充分指定，状态为 PARTIAL。 |
| Scenario 1 | `src/seqmem/model.py::SequentialMemory.scenario1_contributing_sources`; `_reinforce_segment` | contributor 判定含本地 arrival window。 |
| Scenario 2A: matching | `src/seqmem/model.py::SequentialMemory._best_matching_neuron` | `timed_overlap >= L_match` 决定是否复用旧 segment。 |
| Scenario 2B: segment creation | `src/seqmem/model.py::SequentialMemory._grow_segment` | 从 previous winners 建突触；delay 由 `_new_synapse_delay` 计算。 |
| Scenario 3 punishment | `src/seqmem/model.py::SequentialMemory._punish_wrong_predictions` | 未兑现预测的 contributing synapses 被减权。 |
| synaptic delay | `src/seqmem/model.py::SequentialMemory._new_synapse_delay` | strict `current-delay` 是论文公式的本地数值实现。 |
| forgetting | `src/seqmem/model.py::SequentialMemory._prune_neuron`; `src/seqmem/_forgetting_helpers.py` | 依据 weight、age 和 experiment-specific threshold。 |
| Fig.8 protocol | `experiments/fig8_sentence_memory.py::run`; `recall_suffix` | 10 words、6-word cue、4-word raw-neural recall。 |
| Fig.9 protocol | `experiments/fig9_strict_reproduction.py::run_strict_stream`; `rollout_raw_autonomous` | 5-step raw rollout；warmup/evaluation 定义含本地选择。 |
| decoder | `src/seqmem/encoding.py::SSTDRealValueEncoder.decode_likelihood` | half-spacing grid、timed overlap 和 tie averaging 未由论文完整指定。 |
| Fig.9 error metric | `experiments/fig9/metrics.py::mape` | ratio-of-sums 不同于普通 pointwise MAPE，但与 Zhang 引用的 reference [58] 一致，状态为 REFERENCE-SUPPORTED。 |

## FIRST REVIEW HOTSPOTS

| Hotspot | Paper location | Code function | Current status |
|---|---|---|---|
| `kernel_scale` / `V0` / `tau_m` / `tau_s` | Sec. II-B1, Eq.(1) | `DSDynamicsParams.kernel_scale`; `spike_response`; `MemoryParams` | **LOCAL IMPLEMENTATION** |
| `V_inh(t)` | Sec. II-B5, Eq.(3) | `DSNeuronState.membrane_potential` | **IMPLEMENTATION INCOMPLETE / EQUIVALENCE UNPROVEN** |
| intracolumn winner semantics | Sec. II-B5 | `SequentialMemory.predict_code` | **EQUIVALENCE UNPROVEN** |
| `timing_tolerance=.03` | Sec. II-D Scenario matching | `MemoryParams.timing_tolerance` and its call sites | **HIGH-IMPACT LOCAL SEMANTIC** |
| Scenario 1 contributor semantics | Sec. II-D1, Scenario 1 | `scenario1_contributing_sources` | **LOCAL IMPLEMENTATION / OPERATIONALIZATION** |
| synaptic delay realization | Sec. II-D1, Scenario 2 | `_new_synapse_delay` | **PARTIAL** |

## 第一次 Review 只看这 12 个函数

1. `SSTDDiscreteEncoder.encode` — `src/seqmem/encoding.py`：先理解 symbol 如何变成 ordered mini-columns。
2. `SSTDRealValueEncoder.encode` — `src/seqmem/encoding.py`：理解 Fig.9 passenger Gaussian population code。
3. `spike_response` — `src/seqmem/dynamics.py`：核对 Eq.(1) 的 PSP、`V0` 和 time constants。
4. `DSNeuronState.membrane_potential` — `src/seqmem/dynamics.py`：核对 Eq.(3)、phase precession、refractory 和缺失的完整 inhibition。
5. `SequentialMemory.predict_code` — `src/seqmem/model.py`：沿 active context -> segment -> candidate -> event winner 阅读。
6. `SequentialMemory.observe_code` — `src/seqmem/model.py`：沿真实输入 -> Scenario branch -> burst -> publish state 阅读。
7. `SequentialMemory._best_matching_neuron` — `src/seqmem/model.py`：检查 Scenario 2 的 `L_match` 和 timing eligibility。
8. `SequentialMemory._grow_segment` — `src/seqmem/model.py`：检查 Scenario 2B/3 的 segment、synapse 和 delay 创建。
9. `SequentialMemory.scenario1_contributing_sources` — `src/seqmem/model.py`：检查“贡献突触”的本地定义。
10. `SequentialMemory._reinforce_segment` — `src/seqmem/model.py`：检查 strengthen/rejuvenate/weaken/age 的真实写入。
11. `SequentialMemory._punish_wrong_predictions` — `src/seqmem/model.py`：检查未兑现预测如何被处罚。
12. `SequentialMemory._prune_neuron` — `src/seqmem/model.py`：检查 forgetting 如何真正删除长期结构。

随后只需从协议入口阅读 `fig8_sentence_memory.py::recall_suffix` 和 `fig9_strict_reproduction.py::rollout_raw_autonomous`，确认调用方没有重编码预测或读取未来真值。

## 第一次 Review 可以忽略

- `experiments/diagnostics/` 和 `src/seqmem/*diagnostic*`：trace、join、forensics 与 analyzer。
- `runtime_debug_callback`、native crash breadcrumbs、faulthandler 和进程监督代码。
- CSV/JSON/PNG/ZIP、checkpoint sidecar、artifact packaging 和报告生成 I/O。
- `experiments/historical/`：兼容旧结果，不是 strict 主路径。
- `results/`、`reports/refactor/` 和历史运行报告：用于证据追溯，不用于理解第一遍模型。
- oracle、teacher-forced、competition、alternative selector、L-match ablation 等显式 diagnostic modes。
- `optimized_v1/optimized_v2` continuous paths：等价性对照，strict 默认仍为 `reference`。

## Review 输出

逐项核对时使用 `reports/reviewer/PAPER_CODE_REVIEW_CHECKLIST.md`。已确认的高风险差距见 `reports/paper_code_diff/HIGH_IMPACT_SCIENTIFIC_GAPS.md`；它们是 review 入口，不代表本轮已修复。
