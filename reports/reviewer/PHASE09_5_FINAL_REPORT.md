# Phase09.5 最终报告

科学基线 commit：`d33b02506417c1fda7bf0484ce03c68959046991`
分支：`refactor/architecture-cleanup-v2`
范围：仅审阅者说明和文档；没有修改可执行科学代码。

## 1. `burst_context` 是否已经改为 PARTIAL？

是。文档和 `MemoryParams` / `_active_sources` 附近现在明确区分：all-neuron burst 是论文明确规定的行为；但按当前语义把全部 burst neurons 传播为完整的下一步 distal context，并没有在公开论文中被充分说明。

## 2. metric 是否已经改为 REFERENCE-SUPPORTED？

是。Zhang 将指标称为 MAPE 并引用 [58]，但没有重新写出公式；[58] 支持仓库现有的 ratio-of-sums 定义。文档同时保留提醒：该定义不同于普通 pointwise MAPE，但它不是没有依据的本地指标，也不应被标记为直接 mismatch。

## 3. `V_inh` 是否避免了过度声称 MISMATCH？

是。统一状态为 **实现不完整 / 等价性未证实（IMPLEMENTATION INCOMPLETE / EQUIVALENCE UNPROVEN）**。Eq.(3) 明确包含 `V_inh(t)`，而 strict 没有模拟完整的连续轨迹；同时公开论文也没有给出其数值形式，因此 exact author implementation 仍然是未知的。

## 4. winner 是否已经改为 EQUIVALENCE UNPROVEN？

是。论文描述的是先选择 soma membrane-potential 最大的 winner，再进行抑制；strict 的 `existing` 路径选择最早预测发放的事件。文档不再把二者描述为等价，也不直接断言最终科学行为必然错误。

## 5. 是否明确解释了 `kernel_scale` 中 `None` 的真实含义？

是。`response_scale=None` 明确表示 `kernel_scale=1/unscaled_kernel_peak`，不是“没有 scale”。在当前 `tau_m=.10`、`tau_s=.02` 下，effective `V0` 约为 1.87，PSP peak 恰好为 1；状态为 **本地实现（LOCAL IMPLEMENTATION）**。

## 6. 是否明确说明 tau 是本地数值默认值？

是。Eq.(1) 包含 `tau_m/tau_s`，但公开论文和 Table I 都没有给出具体数值；`.10/.02` 被明确标为 repository assumptions。

## 7. 是否明确说明 timing tolerance 是科学语义？

是。`.03` 被标为 **高影响本地语义（HIGH-IMPACT LOCAL SEMANTIC）**，它影响 prediction confirmation、burst/predicted 分类、Scenario matching 和 Scenario-1 contributor attribution。

## 8. 是否把 voltage tolerance 与 timing tolerance 区分开？

是。`integration_voltage_tolerance=2e-5` 被标为 **数值实现参数（NUMERICAL IMPLEMENTATION PARAMETER）**，用于 threshold crossing 的数值 epsilon；它不定义 column/time event 的科学分类。

## 9. 是否加入 strict 术语免责声明？

是。`REVIEWER_GUIDE.md` 开头新增 **IMPORTANT TERMINOLOGY**：strict 指 paper-constrained、non-compensated 的当前实现，不表示 author-code-identical，也不表示所有未公开细节都已知。

## 10. 是否加入 FIRST REVIEW HOTSPOTS？

是。按顺序列出了六项：V0/tau、`V_inh(t)`、winner、timing tolerance、Scenario-1 contributors、synaptic delay，并为每项提供论文位置、代码函数和当前状态。

## 11. 可执行 AST 是否完全不变？

是。相对于基线，剥离 comments/docstrings 后：

- `src/seqmem/dynamics.py`：**EXACT**
- `src/seqmem/model.py`：**EXACT**
- `src/seqmem/encoding.py`：**EXACT**
- `experiments/fig8_sentence_memory.py`：**EXACT**
- `experiments/fig9_strict_reproduction.py`：**EXACT**
- `experiments/fig9/metrics.py`：**EXACT**

## 12. 参数是否完全不变？

是。默认值、CLI、阈值、RNG、时间网格、decoder、learning 和 checkpoint 行为均未修改。

## 13. scientific behavior 是否完全不变？

是。没有修改 executable expression、函数顺序、import、dataclass、状态写入或 strict protocol。

## 验证结果

- 可执行 AST：**6/6 EXACT**
- `python -m compileall -q src experiments tests`：**通过（PASS）**
- `git diff --check`：**通过（PASS）**
- 新科学实验：**无（none）**

## 最终状态

**可以交给资深审阅者继续检查（READY_FOR_SENIOR_REVIEW）**
