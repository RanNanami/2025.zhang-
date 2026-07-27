# 仓库重构审计

审计日期：2026-07-27
基线 commit：`7e0020ae4267961ca95baaeaab953f02a08f46db`
工作分支：`refactor/cleanup-readability`

## 1. 基线

| 项目 | 数值 |
| --- | ---: |
| Git 跟踪文件 | 358 |
| Python 文件 | 48 |
| Python 总行数（含空行） | 15,611 |
| Python 函数/方法 | 538 |
| 超过 80 行的函数 | 30 |
| 缺少 docstring 的 public 或较长对象（静态近似） | 467 |
| 测试文件 | 11 |
| 单元测试 | 125 |
| 基线测试耗时 | 53.104 s |
| 跟踪的 `results/` 文件 | 254 |
| 跟踪的 `results/` 大小 | 123,247,538 bytes |

基线命令：

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD"
.\.venv\Scripts\python.exe -m compileall -q src experiments tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

结果：compileall 成功，`Ran 125 tests ... OK`。

现有 `test_ten_and_fifty_record_streams_match_reference_path`、strict fingerprint、RNG fingerprint、checkpoint resume identity 和 diagnostics invariance 测试已经覆盖关键行为护栏。本轮没有运行正式 250、500、1000、2000 或全年实验。

## 2. 规模与耦合

| 目录 | 跟踪文件 | 跟踪大小 | 判断 |
| --- | ---: | ---: | --- |
| `src/` | 4 | 117,893 B | 核心代码很集中，`model.py` 过长 |
| `experiments/` | 33 | 375,660 B | formal、diagnostic、transfer、historical 边界不够直观 |
| `tests/` | 11 | 105,090 B | 覆盖充分，但按阶段堆积在根目录 |
| `scripts/` | 9 | 13,801 B | Fig.9 脚本较多，命名和安全模板不统一 |
| `results/` | 254 | 123,247,538 B | GitHub 体积和浏览噪声的主要来源 |
| `data/` | 15 | 38,765,280 B | 复现输入，不能按普通生成物处理 |

当前 `src/seqmem` 不 import `experiments`，这是正确的依赖方向。主要耦合点是：

- `src/seqmem/model.py` 同时包含结构对象、预测、学习、transient state、诊断 trace 和三条连续预测实现。
- `experiments/fig9_strict_reproduction.py` 同时包含 strict 配置、checkpoint、fingerprint、rollout、主训练循环、输出、CLI 和 profiling。
- `experiments/fig9_strict_reproduction.py` 从 `fig9_paper_snn.py` 复用数据和指标；后者同时保留 historical/diagnostic 选项，边界需要显式化。
- Fig.8 的公共诊断逻辑已经集中到 `fig8_diagnostic_common.py`，但文件仍有 573 行和 310 行的 `evaluate_diagnostic`。
- checkpoint 使用 pickle；移动 `SequentialMemory`、`Synapse`、`Segment` 等类的模块路径会破坏旧 checkpoint，因此今晚禁止搬类。

## 3. 超长文件

| 文件 | 行数 | 当前职责 | 处理建议 |
| --- | ---: | --- | --- |
| `src/seqmem/model.py` | 2,352 | 数据结构、预测、学习、状态、诊断、连续实现 | 高风险；先加边界测试，再逐职责抽取 |
| `experiments/fig9_strict_reproduction.py` | 1,491 | strict 全流程 | 优先抽取纯配置、输出和 fingerprint |
| `tests/test_fig9_strict_reproduction.py` | 878 | strict 全部回归 | 后续按 checkpoint/protocol/dynamics 拆分 |
| `experiments/diagnostics/fig8_prediction_competition.py` | 761 | trace、统计、CLI | 拆 trace 数据与报告输出 |
| `experiments/diagnostics/fig8_diagnostic_common.py` | 639 | 训练、评估、统计 | 拆只读评估与训练统计 |
| `tests/test_paper_alignment.py` | 633 | 多层论文对齐回归 | 后续按 Fig.7/Fig.9/塑性拆分 |
| `experiments/fig9_paper_snn.py` | 623 | historical Fig.9 与共享 helper | 移入 historical，并保留兼容入口 |
| `experiments/diagnostics/fig9_taxi_prediction.py` | 607 | 非论文出租车诊断 | 保留在 diagnostics，减少默认可见性 |
| `experiments/diagnostics/fig7_nonpaper_baselines.py` | 535 | 多个非论文基线 | 保留在 diagnostics，后续拆模型与 CLI |
| `experiments/diagnostics/fig8_branch_coherence.py` | 524 | branch trace、统计、CLI | 拆 trace/report |

最长函数：

| 函数 | 行数 | 风险 |
| --- | ---: | --- |
| `run_strict_stream` | 378 | 训练、输出、checkpoint 和统计交织 |
| `SequentialMemory.predict_code` | 329 | strict 浮点/RNG/候选顺序敏感 |
| `evaluate_diagnostic` | 310 | trace 与评估耦合 |
| `evaluate_competition` | 258 | 诊断统计和文件输出耦合 |
| `fig8_branch_coherence.evaluate` | 219 | 多种诊断职责 |
| `fig9_paper_snn.run` | 183 | historical 主循环 |
| `recall_suffix` | 169 | 两种检索协议与细节输出交织 |
| `rollout_raw_autonomous` | 162 | strict 五步自主检索 |

## 4. 文件与模块分类

表中的“删除”表示本轮判断，不代表已经执行。

| 文件/模块 | 当前用途 | 调用方 | 保留 | 迁移 | 删除 | 理由 |
| --- | --- | --- | --- | --- | --- | --- |
| `src/seqmem/__init__.py` | 包入口 | tests/experiments | 是 | 暂否 | 否 | checkpoint/import 稳定性 |
| `src/seqmem/encoding.py` | SSTD 离散、标量、周期、组合编码 | 全部正式实验 | 是 | 后续拆 `encoding/` | 否 | 正式核心 |
| `src/seqmem/dynamics.py` | PSP 与 dendritic dynamics | model/tests | 是 | 后续拆 `dynamics/` | 否 | 正式核心 |
| `src/seqmem/model.py` | SequentialMemory 全部行为 | 全部模型实验 | 是 | 分阶段抽取 | 否 | strict 与 pickle 核心 |
| `experiments/fig7_sequence_prediction.py` | Fig.7 正式入口 | scripts/tests/diagnostics | 是 | `experiments/fig7/` | 否 | 正式入口 |
| `experiments/fig7_repeated_trials.py` | Fig.7 十次试验 | run_all/docs | 是 | `experiments/fig7/` | 否 | 正式评估 |
| `experiments/fig7_reference_comparison.py` | Fig.7 对照绘图 | run_all/docs | 是 | `experiments/fig7/` | 否 | 报告引用 |
| `experiments/fig8_sentence_memory.py` | Fig.8(a) 正式入口 | scripts/tests/diagnostics | 是 | `experiments/fig8/` | 否 | 正式入口 |
| `experiments/fig8_retrieval_ablation.py` | neural/proximal 对照 | README | 是 | Fig.8 diagnostics | 否 | 协议证据 |
| `experiments/fig8_resource_sweep.py` | Fig.8 资源扫描 | run_all/docs | 是 | Fig.8 diagnostics | 否 | 已有复现流程 |
| `experiments/fig8c_poem_memory.py` | Fig.8(c) 诗句实验 | tests/docs | 是 | `experiments/fig8/` | 否 | 正式扩展 |
| `experiments/prepare_fig8c_poems.py` | 数据准备 | tests/docs | 是 | `experiments/fig8/data.py` | 否 | 数据可追溯 |
| `experiments/generate_fig8c_stress_dataset.py` | 压力集生成 | tests/docs | 是 | Fig.8 diagnostics | 否 | 诊断可复现 |
| `experiments/plot_fig8c_stress_comparison.py` | 压力集绘图 | docs | 是 | Fig.8 diagnostics | 否 | 报告入口 |
| `experiments/fig9_strict_reproduction.py` | Fig.9 strict 入口 | scripts/tests/docs | 是 | 优先拆分 | 否 | strict 核心 |
| `experiments/fig9_paper_snn.py` | historical/共享 Fig.9 helper | strict/transfer/docs | 是 | historical + compatibility shim | 否 | 仍有真实调用方 |
| `experiments/fig9_reference_comparison.py` | Fig.9 论文曲线对照 | docs | 是 | Fig.9 evaluation | 否 | 报告引用 |
| `experiments/fig9_recompute_rolling.py` | 重算 rolling 指标 | docs | 是 | Fig.9 tools | 否 | 结果审计 |
| `experiments/fig9_compare_runs.py` | original/perturbed 对比 | 无静态引用 | 待确认 | Fig.9 tools | 待确认 | 128 行独立 CLI，可能是人工入口 |
| `experiments/diagnostics/fig7_context_audit.py` | Fig.7 上下文审计 | README/FIG7_DEBUG | 是 | Fig.7 diagnostics | 否 | 文档引用 |
| `experiments/diagnostics/fig7_nonpaper_baselines.py` | 非论文基线 | README | 是 | Fig.7 diagnostics | 否 | 明确 nonpaper |
| `experiments/diagnostics/fig7_prediction_trace.py` | Fig.7 trace | 无静态引用 | 待确认 | Fig.7 diagnostics | 待确认 | 81 行人工诊断入口 |
| `experiments/diagnostics/fig8_diagnostic_common.py` | Fig.8 公共诊断 | 5 个诊断入口/tests | 是 | 拆分内部职责 | 否 | 多调用方 |
| `experiments/diagnostics/fig8_false_positive_ablation.py` | 假阳性对照 | docs/tests | 是 | Fig.8 diagnostics | 否 | 阶段报告证据 |
| `experiments/diagnostics/fig8_response_scale_sweep.py` | response scale 扫描 | docs | 是 | Fig.8 diagnostics | 否 | 阶段报告证据 |
| `experiments/diagnostics/fig8_scenario1_diagnostic.py` | Scenario 1 初始诊断 | docs | 是 | Fig.8 diagnostics | 否 | 报告引用 |
| `experiments/diagnostics/fig8_scenario1_contribution.py` | 三种贡献定义 CLI | 无静态引用 | 待确认 | Fig.8 diagnostics | 待确认 | 90 行，结果报告存在但入口未被引用 |
| `experiments/diagnostics/fig8_prediction_competition.py` | 排序/timing 诊断 | tests | 是 | 拆 trace/report | 否 | 论文差距证据 |
| `experiments/diagnostics/fig8_branch_coherence.py` | 分支一致性诊断 | tests/docs | 是 | 拆 trace/report | 否 | 论文差距证据 |
| `experiments/diagnostics/context_memory.py` | 非论文 context-table helper | taxi diagnostic | 是 | diagnostics 内部 | 否 | 有调用方 |
| `experiments/diagnostics/fig9_taxi_prediction.py` | historical/非论文出租车诊断 | README | 是 | historical/diagnostics | 否 | 结果对照入口 |
| `experiments/diagnostics/benchmark_real_timeseries.py` | ETTh1/weather 简单基线 | scripts/tests/docs | 是 | transfer diagnostics | 否 | 用户汇报所需 |
| `experiments/diagnostics/etth1_paper_snn.py` | ETTh1 迁移 | tests/docs | 是 | transfer diagnostics | 否 | 用户实验 |
| `experiments/diagnostics/weather_paper_snn.py` | Weather 迁移 | tests/docs | 是 | transfer diagnostics | 否 | 用户实验 |
| `experiments/diagnostics/multivariate_fig9_transfer.py` | 多字段迁移 | scripts/tests/docs | 是 | transfer diagnostics | 否 | 用户实验 |
| `experiments/diagnostics/multihorizon_fig9_transfer.py` | 多步迁移 | scripts/tests/docs | 是 | transfer diagnostics | 否 | 用户实验 |
| `tests/test_paper_alignment.py` | 编码、动力学、塑性、协议回归 | unittest | 是 | 按主题拆分 | 否 | 核心护栏 |
| `tests/test_fig9_strict_reproduction.py` | strict/checkpoint/fingerprint 回归 | unittest | 是 | 按主题拆分 | 否 | 核心护栏 |
| `tests/test_fig8_*.py` | Fig.8 多阶段诊断回归 | unittest | 是 | `tests/diagnostics/` | 否 | 诊断不污染行为的证据 |
| `tests/test_*transfer.py` | ETTh1/weather/multivariate 回归 | unittest | 是 | `tests/integration/` | 否 | 迁移入口护栏 |
| `tests/test_fig8c_poem_memory.py` | Fig.8(c) 回归 | unittest | 是 | `tests/integration/` | 否 | 正式扩展护栏 |

## 5. 脚本审计

- `check_project.ps1`、`run_all.ps1` 和五个根目录 transfer 脚本都被 Markdown 引用，不能按“看起来重复”直接删除。
- `scripts/test_all.ps1` 与 `check_project.ps1` 有测试职责重叠，但前者只跑测试，后者还校验数据并跑 smoke；建议统一公共执行模板，不直接二选一删除。
- 多个旧 Fig.9 脚本通过 `cmd.exe /c` 拼接命令行；这降低可读性并增加路径转义风险。应改为 PowerShell argument array，但每改一个脚本都要做语法和 smoke 验证。
- `run_all.ps1` 中 native command 失败通过 pipeline 后不一定被稳定传播，属于后续小修候选。
- 正式长实验脚本必须保留显式人工开关，不能在测试或 import 时自动运行。

## 6. 删除候选与证据

### 可安全删除但未执行

本地发现约 419,992,111 bytes 的已忽略、可再生成 ZIP、pstats、`__pycache__` 和 `.pytest_cache`。删除命令被桌面安全策略拒绝两次，确认没有发生删除。这些内容不在 Git 跟踪中，不影响 GitHub commit；后续可由用户在文件管理器或本机终端清理。

`tmp/` 中存在两个大型 Fig.8 model pickle 和论文 PDF/页面渲染。虽然目录被忽略，但用途不能确认，因此没有删除，也没有把它们误归类为普通缓存。

### 等待用户确认的 299 行代码

| 文件 | 行数 | 全库静态引用 | 风险 |
| --- | ---: | ---: | --- |
| `experiments/diagnostics/fig7_prediction_trace.py` | 81 | 0 | 可能是人工诊断 CLI |
| `experiments/diagnostics/fig8_scenario1_contribution.py` | 90 | 0 | 对应已完成的阶段实验 |
| `experiments/fig9_compare_runs.py` | 128 | 0 | 可能是手工结果比较工具 |

“0 静态引用”不等于死代码。上述入口可被命令行直接运行，今晚不删除。

### 等待用户确认的历史结果

`results/` 中有 254 个跟踪文件、约 123 MB。大量逐事件 CSV 可以重新生成，但它们也可能是论文差距诊断证据。建议下一阶段按以下标准迁出 Git 主树，而不是今晚删除：

1. 每项实验保留 compact summary、protocol、README 和必要图。
2. 原始逐事件 trace 放 GitHub Release、外部归档或压缩资产仓库。
3. 文档链接改好并验证后，再从普通 Git 历史的未来 commit 中移除。
4. 不做 history rewrite，不 force push。

### 明确保留

- strict `reference` 连续预测实现。
- `optimized_v1` / `optimized_v2` 及其等价性测试；正式 250 A/B 已证明它们更慢，但它们仍是审计对象。
- historical compensated 最终可复现实例。
- checkpoint schema、pickle 类路径和 resume 测试。
- 所有 strict fingerprint、RNG 和 prediction SHA 护栏。

## 7. 注释与类型审计

已有关键位置使用 `STRICT PROTOCOL`、`STATE MUTATION`、`PAPER-EXPLICIT`、`LOCAL CHOICE` 和 `DEBUG WATCH`，方向正确。问题是覆盖不均：

- 静态扫描得到 467 个缺少 docstring 的 public 或较长对象，其中包含测试类；不能机械套模板。
- `predict_code`、`observe_code`、`advance_prediction` 和 strict rollout 已有较好的状态影响说明。
- diagnostics 的大型 `evaluate*` 和 CLI `main` 普遍缺少输入、输出、RNG、长期状态影响说明。
- 应优先给跨模块 public API 和状态修改函数加中文 docstring；简单 helper 只写一两行。
- 当前环境没有安装 ruff，因此“未使用 import”尚未形成可靠删除清单。不能只用字符串匹配批量删 import。

## 8. 今晚结论

- 核心行为没有修改。
- strict 默认、浮点顺序、RNG 顺序和 checkpoint 类路径没有修改。
- GitHub 可读性问题的最大来源是结果资产体积、入口分层不清和两个超长主模块。物理行基线为 15,611 行。
- 目前可证明的代码删除候选为 299 行，但都属于可能被人工调用的 CLI，必须由用户确认。
- 高收益且低风险的下一步是先抽取 Fig.9 的纯配置/输出/fingerprint，再处理 `model.py`；不要反过来。
