# 仓库重构计划

本计划只改变结构和可读性，不改变论文协议、模型参数、RNG 调用次数、浮点累加顺序、候选顺序、序列化字段或默认 CLI。

## 1. 拟议目录

目录结构按职责组织，但避免创建只有几行代码的小文件：

```text
src/seqmem/
    __init__.py                 # 保留兼容导出
    model.py                    # 第一阶段保留 pickle 类路径
    encoding.py                 # 第一阶段保留兼容入口
    dynamics.py                 # 第一阶段保留兼容入口
    state/
        checkpoint.py           # 显式 schema、保存与迁移
        fingerprint.py          # 稳定序列化和哈希
        transient.py            # snapshot/restore helper
    diagnostics/
        traces.py               # 只读 trace 数据结构和导出

experiments/
    fig7/
        run.py
        repeated_trials.py
        comparison.py
        diagnostics/
    fig8/
        run.py
        poem.py
        data.py
        diagnostics/
    fig9/
        run_strict.py
        config.py
        data.py
        rollout.py
        outputs.py
        checkpoint.py
        diagnostics/
    historical/
        fig9_compensated/
        fig9_paper_snn.py
    transfer/
        datasets.py
        etth1.py
        weather.py
        multivariate.py
        multihorizon.py

scripts/
    test_all.ps1
    fig7/
    fig8/
    fig9/

tests/
    unit/
    regression/
    integration/
    diagnostics/
```

第一阶段不移动 `Synapse`、`Segment`、`Neuron`、`MiniColumn`、`MemoryParams` 和 `SequentialMemory` 的定义位置。它们继续从 `seqmem.model` 反序列化，直到 checkpoint migration 测试和 compatibility shim 完成。

## 2. 分阶段实施

### 阶段 A：备份与审计

状态：已完成。

- 三层备份。
- 记录 15,611 行 Python、125 个测试和 358 个跟踪文件。
- 建立删除候选和高风险结果清单。
- 改进 `.gitignore`，让本地生成目录不再污染 status。

验收：bundle、分支、标签和 checkpoint 哈希均验证成功。

### 阶段 B：架构护栏

新增静态测试：

- `src/seqmem` 不 import `experiments`。
- strict 模块不 import `experiments.historical`。
- 核心预测路径不 import diagnostics。
- import diagnostics 不改变 `MemoryParams` 默认值。
- 旧 checkpoint 可以读取并保持 fingerprint。

每个测试先在当前结构通过，再做迁移。

### 阶段 C：抽取 Fig.9 纯逻辑

优先从 `fig9_strict_reproduction.py` 抽取不影响 RNG 和模型状态的职责：

1. `config.py`：strict 常量、dataclass、版本标识。
2. `outputs.py`：JSON/CSV 写入和 summary path。
3. `fingerprint.py`：纯哈希与协议 fingerprint。
4. `checkpoint.py`：保存、读取、schema 校验。

旧入口继续 re-export，现有命令不失效。每抽取一个职责就运行 compileall、125 测试、checkpoint identity 和 `git diff --check`。

### 阶段 D：隔离 historical 与 diagnostics

- 将 nonpaper/compensated 入口放入明确目录。
- strict 不得默认 import historical。
- 为旧文件保留薄 compatibility shim 和 deprecation message。
- 阶段报告引用的 diagnostics 只迁移，不删除。

这一阶段不修改算法实现，不合并“看起来相似但协议不同”的函数。

### 阶段 E：拆分 SequentialMemory

这是最高风险阶段，必须在 C、D 完成后开始。

建议先抽取无状态 helper：

- continuous prediction 的纯时间搜索 helper。
- contribution selection 的纯函数。
- trace 构造与格式化。

之后再考虑 state/checkpoint。所有会改变 RNG、集合遍历、浮点累加或 segment mutation 顺序的抽取都需要逐行等价测试。`model.py` 保留兼容类定义，必要时仅委托 helper。

### 阶段 F：统一脚本

- 建立一个短小的 PowerShell 公共执行模板。
- 使用 `$PythonArgs`。
- 设置 `PYTHONUNBUFFERED` 和 `PYTHONFAULTHANDLER`。
- 运行前创建日志目录。
- 检查 `$LASTEXITCODE` 并传播失败。
- 长实验必须要求显式开关。
- 旧脚本先变成兼容入口，确认文档和调用方迁移后再删除。

### 阶段 G：文档与结果资产

- 新增 `ARCHITECTURE.md` 和 `REFACTOR_REPORT.md`。
- 更新 `PAPER_ALIGNMENT.md`、`RESULTS_STATUS.md`、`FIG9_MANUAL_DEBUG_GUIDE.md`。
- 为每个正式入口给出“数据 -> 编码 -> predict -> observe/rollout -> metrics -> output”的中文调用链。
- compact summary 保留在 Git；大体积逐事件 trace 迁至 Release/外部归档需要用户确认。

## 3. 每次提交的固定验证

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD"
.\.venv\Scripts\python.exe -m compileall -q src experiments tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
git diff --check
git status -sb
```

涉及 Fig.9 strict 时额外检查：

- strict protocol fingerprint；
- predictions SHA；
- model fingerprint；
- RNG fingerprint；
- checkpoint save/resume identity；
- density trace 开关不改变结果；
- reference 为默认 continuous implementation。

不自动执行正式 250、500、1000、2000 或全年实验。

## 4. 提交顺序

1. `Document pre-refactor backup`
2. `Audit repository structure`
3. `Add architecture dependency guards`
4. `Extract Fig9 protocol and output helpers`
5. `Separate strict diagnostic and historical entrypoints`
6. `Consolidate checkpoint and fingerprint handling`
7. `Refactor sequential memory helpers`
8. `Clean experiment entrypoints and scripts`
9. `Remove user-approved dead code`
10. `Add Chinese API documentation`
11. `Finalize regression and architecture report`

任何阶段出现无法小范围修复的全量测试失败、checkpoint 不兼容、strict/RNG fingerprint 变化，立即停止后续重构并写 `REFACTOR_BLOCKERS.md`。

## 5. 需要用户确认

以下项目不在今晚自动执行：

- 是否删除三个无静态引用的 CLI，共 299 行。
- 是否把 123 MB tracked results 的逐事件 CSV 迁出普通 Git。
- 是否保留 `fig9_compare_runs.py` 作为正式结果审计工具。
- `tmp/fig8_neural_500_model.pkl` 和 `tmp/fig8_neural_1000_model.pkl` 是否需要提升为正式 checkpoint 资产。
- 是否最终删除 v1/v2 连续预测实验路径；当前要求仍是保留其等价性测试，因此默认不删。
- 是否在 checkpoint migration 完成后移动核心类的模块路径。

## 6. 完成定义

重构完成必须同时满足：

- 原有 125 个测试和新增架构测试全部通过。
- strict 数值基线与 fingerprint 保持不变。
- checkpoint 可读取、resume 一致。
- formal、diagnostic、historical 在目录和 import 上可区分。
- 初学者能从 README 顺着一个入口读到编码、预测、学习、评估和输出。
- 所有删除都有引用搜索、替代实现和恢复位置记录。
- 只 push `refactor/cleanup-readability`，不自动合并 `main`。
