# Phase09 Reviewability Pass Report

Baseline: `d33b02506417c1fda7bf0484ce03c68959046991`
Scope: documentation, comments, docstrings, and visual section headers only.

## 1. 师兄应该从哪个文件开始看？

从根目录 `REVIEWER_GUIDE.md` 开始。它先给出完整模型主流程，再把论文概念映射到代码位置，并明确哪些 strict 行为仍是本地实现。逐项审查时使用 `reports/reviewer/PAPER_CODE_REVIEW_CHECKLIST.md`。

## 2. 推荐阅读的函数顺序是什么？

1. `SSTDDiscreteEncoder.encode`
2. `SSTDRealValueEncoder.encode`
3. `spike_response`
4. `DSNeuronState.membrane_potential`
5. `SequentialMemory.predict_code`
6. `SequentialMemory.observe_code`
7. `SequentialMemory._best_matching_neuron`
8. `SequentialMemory._grow_segment`
9. `SequentialMemory.scenario1_contributing_sources`
10. `SequentialMemory._reinforce_segment`
11. `SequentialMemory._punish_wrong_predictions`
12. `SequentialMemory._prune_neuron`

完成核心阅读后，再看 `fig8_sentence_memory.py::recall_suffix` 和 `fig9_strict_reproduction.py::rollout_raw_autonomous` 两个协议入口。

## 3. 哪些模块第一次可以忽略？

第一次阅读可以忽略 diagnostic traces、runtime/native crash debug、artifact I/O、checkpoint sidecars、historical experiments、`results/`、历史 reports、oracle/teacher-forced/ablation modes，以及非默认 continuous optimization paths。它们没有被搬动或删除。

## 4. 标出了多少 PAPER GAP？

在六个科学/协议文件中加入 **21 个 `PAPER GAP` 标记**，并加入 **15 个 `PAPER STATUS` 标记**。覆盖附件要求的 V0 normalization、tau、积分网格/epsilon、timing tolerance、cycle、phase、refractory、V_inh、winner、burst、Scenario 1、delay、Fig.9 Gaussian/decoder/warmup/rollout，并额外标出 Fig.9 metric definition。

## 5. 有没有修改 executable scientific code？

没有。对以下六个文件执行了 `HEAD` 与工作树的 AST 对比，剥离 comments/docstrings 后全部 `executable_ast_equal=True`：

- `src/seqmem/dynamics.py`
- `src/seqmem/model.py`
- `src/seqmem/encoding.py`
- `experiments/fig8_sentence_memory.py`
- `experiments/fig9_strict_reproduction.py`
- `experiments/fig9/metrics.py`

## 6. 有没有修改参数？

没有。参数值、默认值、CLI 参数、RNG、阈值、时间网格、decoder 规则和 checkpoint 类型均未修改。

## 7. 有没有改变 strict behavior？

没有。没有修改表达式、分支、调用、函数顺序或模型状态写入。`git diff --check` 通过。Six Golden EXACT 未运行，因为 production scientific executable AST diff 为零。

## 8. Validation 结果？

- `python -m compileall -q src experiments tests`: **PASS**
- Focused unit tests: **155 tests PASS**, 0 failures, 0 errors, 78.742 s
- Test modules: paper alignment, paper inhibition audit, Fig.8 neural retrieval, Fig.8 Scenario-1 contribution, Fig.9 strict reproduction, Fig.9 evaluation metric audit, Fig.9 paper-code parity
- Executable AST equality: **6/6 PASS**
- `git diff --check`: **PASS**
- New scientific experiments: **none run**

Phase09 stops here. The marked gaps are navigation and review evidence; no paper/code gap was silently fixed.
