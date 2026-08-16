# Phase09.5 Final Report

Baseline scientific commit: `d33b02506417c1fda7bf0484ce03c68959046991`
Branch: `refactor/architecture-cleanup-v2`
Scope: reviewer annotations and documentation only.

## 1. burst_context status 是否已改为 PARTIAL？

是。文档与 `MemoryParams`/`_active_sources` 附近均明确区分：all-neuron burst 是 paper-explicit；将全部 burst neurons 按当前语义传播为完整下一步 distal context 未被公开论文充分指定。

## 2. metric 是否改为 REFERENCE-SUPPORTED？

是。Zhang 将指标称为 MAPE 并引用 [58]，没有重述公式；[58] 支持仓库现有 ratio-of-sums 定义。文档同时保留提醒：它不同于普通 pointwise MAPE，但不是 unsupported local metric，也不应标作直接 mismatch。

## 3. V_inh 是否避免过度声称 MISMATCH？

是。统一状态为 **IMPLEMENTATION INCOMPLETE / EQUIVALENCE UNPROVEN**。Eq.(3) 明确包含 `V_inh(t)`，strict 未模拟完整连续轨迹；同时公开论文未给出其数值形式，所以 exact author implementation 仍是 unknown。

## 4. winner 是否改为 EQUIVALENCE UNPROVEN？

是。论文描述 largest soma membrane-potential winner followed by inhibition；strict `existing` 路径选择 earliest predicted firing。两者不再被描述为等价，也不直接断言最终科学行为必然错误。

## 5. kernel_scale 是否明确解释 None 的真实含义？

是。`response_scale=None` 明确表示 `kernel_scale=1/unscaled_kernel_peak`，不是“无 scale”。在当前 `tau_m=.10`、`tau_s=.02` 下，effective `V0` 约为 1.87，PSP peak 恰为 1；状态为 **LOCAL IMPLEMENTATION**。

## 6. tau 是否明确为 local numeric defaults？

是。Eq.(1) 包含 `tau_m/tau_s`，但公开论文和 Table I 均未给数值；`.10/.02` 被明确标为 repository assumptions。

## 7. timing tolerance 是否明确为 scientific semantic？

是。`.03` 被标为 **HIGH-IMPACT LOCAL SEMANTIC**，影响 prediction confirmation、burst/predicted classification、Scenario matching 与 Scenario-1 contributor attribution。

## 8. voltage tolerance 是否与 timing tolerance 区分？

是。`integration_voltage_tolerance=2e-5` 被标为 **NUMERICAL IMPLEMENTATION PARAMETER**，用于 threshold crossing 数值 epsilon；它不定义 column/time event 的科学分类。

## 9. strict 术语免责声明是否加入？

是。`REVIEWER_GUIDE.md` 开头新增 **IMPORTANT TERMINOLOGY**：strict 是 paper-constrained、non-compensated current implementation，不表示 author-code-identical 或所有未公开细节已知。

## 10. FIRST REVIEW HOTSPOTS 是否加入？

是。按顺序仅列出六项：V0/tau、`V_inh(t)`、winner、timing tolerance、Scenario-1 contributors、synaptic delay，并为每项提供 paper location、code function 和 status。

## 11. executable AST 是否完全不变？

是。相对 baseline，剥离 comments/docstrings 后：

- `src/seqmem/dynamics.py`: **EXACT**
- `src/seqmem/model.py`: **EXACT**
- `src/seqmem/encoding.py`: **EXACT**
- `experiments/fig8_sentence_memory.py`: **EXACT**
- `experiments/fig9_strict_reproduction.py`: **EXACT**
- `experiments/fig9/metrics.py`: **EXACT**

## 12. 参数是否完全不变？

是。默认值、CLI、阈值、RNG、时间网格、decoder、learning 和 checkpoint 行为均未修改。

## 13. scientific behavior 是否完全不变？

是。没有修改 executable expression、函数顺序、import、dataclass、状态写入或 strict protocol。

## Validation

- Executable AST: **6/6 EXACT**
- `python -m compileall -q src experiments tests`: **PASS**
- `git diff --check`: **PASS**
- New scientific experiments: **none**

## Final State

**READY_FOR_SENIOR_REVIEW**
