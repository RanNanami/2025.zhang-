# 高影响论文—代码科学差距

审计基线：commit `d33b02506417c1fda7bf0484ce03c68959046991`。

范围仅限 Zhang et al. (2025)、`src/seqmem/dynamics.py`、`src/seqmem/model.py` 中的科学路径，以及 Fig.8/Fig.9 strict/default 协议。`UNKNOWN` 表示公开论文没有提供足够信息来判断等价性。

| 项目 | 论文证据 | 当前代码 | 状态 | 风险 | 可能影响 |
|---|---|---|---|---|---|
| `kernel_scale` | Eq. (1) 使用 `V0[exp(-t/tau_m)-exp(-t/tau_s)]`；Eq. (1) 和 Table I 都没有规定峰值归一化。 | 当 `response_scale is None` 时，`DSDynamicsParams.kernel_scale` 返回 `1 / unscaled_kernel_peak(...)`（`dynamics.py:93-97`）。Fig.8 和 Fig.9 strict 都会使用这个默认值。 | 本地实现（LOCAL IMPLEMENTATION） | 极高（CRITICAL） | 每个单位权重 PSP 的峰值都被固定为 1，因此有多少个 `w0=0.5` 突触能够越过阈值会被改变；兴奋性和 segment 歧义可能发生定性变化。 |
| `response_scale` / `V0` | Eq. (1) 中符号上存在 `V0`，但公开正文和 Table I 没有数值。 | strict 默认值为 `response_scale=None`（`model.py:431`；Fig.9 配置 `:205`），但 `None` 表示自动单位峰值归一化，不是没有乘数。在 `tau_m=.10`、`tau_s=.02` 下，它隐含 `V0` 约为 1.87。 | 本地实现（LOCAL IMPLEMENTATION） | 极高（CRITICAL） | strict 模型包含一个论文未公开的振幅选择；它可能使两个同步的初始突触就足以达到阈值 1。 |
| `tau_m` 和 `tau_s` | Eq. (1) 说明相关常数决定 kernel 形状；Table I 没有公布它们的数值。 | `DSDynamicsParams` 和 `MemoryParams` 中均为 `tau_m=.10`、`tau_s=.02`（`dynamics.py:68-69`；`model.py:429-430`）。 | 本地实现（LOCAL IMPLEMENTATION） | 极高（CRITICAL） | 改变 PSP 峰值时间、时间叠加、阈值 crossing、预测发放时间，以及学习到的 delay 的有效含义。 |
| `integration_step` | 论文定义了连续膜电位方程，但没有规定数值积分网格。 | strict 连续预测使用 `integration_step=.005` 扫描（`model.py:435`；Fig.9 配置 `:207`）。 | 本地实现（LOCAL IMPLEMENTATION） | 高（HIGH） | 网格量化可能改变首次 crossing 和 soma 发放时间，进而影响事件顺序、decoder overlap 和 bursting。 |
| `integration_voltage_tolerance` | 论文在 Eq. (1)/(3) 中使用精确不等式 `V_d >= theta_d` 和 soma voltage `>= theta`；没有说明电压 epsilon。 | 有效树突和胞体阈值会因 `2e-5` 而降低（`model.py:436`、`:3198`、`:3285-3297`）。 | 数值实现参数（NUMERICAL IMPLEMENTATION PARAMETER） | 高（HIGH） | 这是数值 crossing epsilon：接近阈值的 segment 可能在略低于打印阈值时被接受。它不是下面的语义时间窗口。 |
| `timing_tolerance` | SSTD 强调顺序而非精确时间，Scenario 2 说 spikes 在同一时间到达，但论文没有公布 `0.03` 的接受窗口。 | `timing_tolerance=.03` 控制预测确认、burst/predicted 分类、Scenario matching、贡献归因、active-cell recovery 和 decoding（`model.py:425`；`encoding.py:184`、`:212`）。 | 高影响本地语义（HIGH-IMPACT LOCAL SEMANTIC） | 极高（CRITICAL） | 决定哪个科学分支会执行，以及哪些突触获得 credit；它不是浮点计算 epsilon。 |
| `cycle_period` | 论文定义了分层振荡频率、半周期去极化，并在 Table I 给出相对频率约束；没有规定通用的无量纲周期为 1。 | `cycle_period=1.0`；去极化使用 `cycle/2`，频率为 `1/cycle`，refractory 持续一个 cycle（`model.py:437`、`:459-461`）。 | 本地实现（LOCAL IMPLEMENTATION） | 高（HIGH） | 它建立了所有 delay、时间常数、积分步长和容差所使用的单位；如果归一化不正确，会改变这些量的相对尺度。 |
| phase-precession 实现 | Sec. II-B4 说预测状态下的振荡会改变 phase，使神经元在没有 feedforward input 时发放，然后回到 `phi0`；没有给出可执行的 reset 轨迹。 | 树突 crossing 时，oscillation 被替换为从 trough 开始的 `-A*cos(2*pi*f*elapsed)`（`dynamics.py:157-168`）；连续 solver 为每个 candidate 创建新状态。 | 部分对齐（PARTIAL） | 高（HIGH） | 所选 reset phase 决定 soma latency 和发放顺序；它合理，但不是论文唯一推出的实现。 |
| refractory 实现 | Eq. (3) 定义神经元发放后，在当前振荡周期剩余时间内 `eta=-theta`。 | 如果 elapsed 小于配置的一个 cycle，`DSNeuronState` 返回 `-inf`（`dynamics.py:150-155`、`:178-180`）；strict selection 还结构性地保证每列只有一个事件。 | 部分对齐（PARTIAL） | 高（HIGH） | 抑制强度和 cycle 边界语义可能更强或不同，可能移除论文模型本来会保留的事件。 |
| `V_inh(t)` | Eq. (3) 明确加入抑制反馈，正文也描述了列内抑制；但其完整数值轨迹没有公开。 | `membrane_potential()` 接收 inhibition，但 strict prediction 以零 inhibition 调用它，没有模拟完整连续的 `V_inh(t)` 轨迹（`model.py:3279-3297`、`:3409-3423`）。 | 实现不完整 / 等价性未证实（IMPLEMENTATION INCOMPLETE / EQUIVALENCE UNPROVEN） | 极高（CRITICAL） | candidate competition 可能不同；但论文也没有公开 exact author implementation，因此仅凭这些证据不能直接证明最终科学行为错误。 |
| intracolumn winner 语义 | 论文选择 mini-column 中 soma membrane-potential 最大的神经元，并随后进行抑制。 | strict `existing` policy 保留每列最早的 predicted firing event（`model.py:1544-1606`）；其他 score selector 属于 diagnostic。 | 等价性未证实（EQUIVALENCE UNPROVEN） | 极高（CRITICAL） | 最早发放不一定等价于 Eq. (3) 中电压最大；差异是合理怀疑，但公开证据不足以证明必然失败。 |
| `burst_context` | 如果 active mini-column 中没有预测神经元，论文明确说所有神经元 burst/fire。 | `burst_context=True` 还会把每个 burst neuron 都传播为完整的下一步 distal context，同时单独记录一个 learning winner（`model.py:780-787`；Fig.9 配置 `:208`）。 | 部分对齐（PARTIAL） | 高（HIGH） | burst 发放本身一致；但完整传播所有 burst neuron 的语义没有被公开论文充分指定，可能扩大后续 candidate activity。 |
| `scenario1_contribution_mode` | Scenario 1 强化所有对产生预测的 segment 有贡献的突触，并减弱/老化其他突触；没有给出可执行贡献判据。 | strict 将贡献操作化为 `arrival-window`：source arrival 位于目标树突时间 `+/- .03` 内（`model.py:3562-3589`）。continuous-positive/causal 只作为 diagnostic。 | 本地实现 / 操作化定义（LOCAL IMPLEMENTATION / OPERATIONALIZATION） | 极高（CRITICAL） | 一个真正促成连续 PSP crossing 的突触，可能被判为无贡献并减弱；这会改变学习机制，而不只是改变统计报告。 |
| 突触 delay 实现 | Scenario 2 打印新突触的 `d_ij = t_j - t_i - pi`，但公开文字没有给出 phase 项到当前归一化时间轴的完整映射。 | strict 使用 `cycle_period/2 + target_time - source_time`；减去 PSP peak time 只是 opt-in 的 nonpaper diagnostic（`model.py:2990-3004`）。 | 部分对齐（PARTIAL） | 高（HIGH） | PSP 峰值可能系统性晚于预期 SSTD event，造成 timing mismatch、burst 和错误的强化归因。 |
| `L_match` | Sec. III-A 明确：taxi 使用 4，其他实验使用 3。 | Fig.8 默认 3；Fig.9 strict 默认 4（`fig8_sentence_memory.py:441`；`fig9_strict_reproduction.py:201`、`:5179-5188`）。 | 一致（MATCH） | 高（HIGH） | 公开数值一致；剩余风险来自同一时间/active synapse 的本地定义，已由 `timing_tolerance` 覆盖。 |
| Fig.9 passenger Gaussian centers/range/sigma | Sec. II-C 规定 Gaussian receptive fields 按 `l` 间隔覆盖编码范围、取 top `K`，分辨率为 `l/2`；Fig.9 指定 482 columns 和 `K=10`。没有公布 taxi 范围、中心端点或 Gaussian sigma。 | 中心点均匀覆盖 `[0, 40000]`；间距为 `range/(482-1)`；默认 `sigma=spacing`，数值会裁剪到范围内（`encoding.py:120-163`；Fig.9 配置 `:198-204`）。 | 本地实现（LOCAL IMPLEMENTATION） | 极高（CRITICAL） | 会改变 passenger count 对应的十个 columns、相邻数值的 code overlap、裁剪行为和有效预测分辨率。 |
| Fig.9 decoder | 论文说 MAPE 使用 most likely prediction，但没有给出 inverse-SSTD likelihood、tie-break 或缺失预测算法。 | decoder 扫描 half-spacing grid，按 `(timed_overlap, column_overlap)` 排序，并对所有并列值取平均（`encoding.py:181-236`、`:269-277`）。 | 本地实现（LOCAL IMPLEMENTATION） | 极高（CRITICAL） | 稠密或不完整 raw code 可能产生大量 tie 和有偏的平均值；报告的 passenger value 可能主要由 decoder 设计决定，而不是 memory state。 |
| Fig.9 autonomous rollout | Fig.9 要求从历史数据进行五步/2.5 小时预测，并在线处理；精确的多步状态管理协议没有公开。 | strict 执行五步 raw-neural prediction，不重新编码 decoded value，rollout 期间不学习，并在 observe 当前记录前恢复 transient state（`fig9_strict_reproduction.py:1490-1529`、`:3189-3219`）。 | 部分对齐（PARTIAL） | 高（HIGH） | horizon 一致且避免 ground-truth leakage；但未公开的 rollout 状态约定可能实质性改变多步预测。 |
| Fig.9 warmup | Zhang 论文说明一年 / 17,520 个半小时记录和五步目标；没有说明评分前排除 5,904 条记录。 | strict/default CLI 使用 `warmup=5904`（`fig9_strict_reproduction.py:193`、`:5175`）。 | 本地实现（LOCAL IMPLEMENTATION） | 高（HIGH） | 选择了不同的评价人群，并让模型获得数月未计分的适应时间，因此无法仅凭论文重现该数值 MAPE。 |
| Fig.9 error metric 公式 | Zhang 只命名 MAPE 并引用 [58]，没有重述方程；参考文献 [58] 使用 ratio-of-sums 定义。 | `mape` 计算 `sum(abs(pred-target))/sum(abs(target))`（`metrics.py:10-25`）。 | 参考文献支持（REFERENCE-SUPPORTED） | 中（MEDIUM） | 它不同于普通 pointwise MAPE，因此报告必须写清定义；但该实现有引用链支持，不应被当作没有依据的 mismatch。 |
| Fig.9 rolling/evaluation window | Zhang 正文没有公布 400-record rolling window。 | rolling output 使用 400 个 error，并除以 target scale（`metrics.py:131-140`；Fig.9 配置 `:194`）。 | 本地实现（LOCAL IMPLEMENTATION） | 高（HIGH） | 会改变 adaptation curve 的平滑程度和可比性，尽管底层 ratio-of-sums 公式有参考文献支持。 |
| nonpaper diagnostic 泄漏到 strict | 论文没有 competitive-raw selector、peak-aligned delay、continuous-positive/causal attribution、oracle candidate 或 alternative continuous implementation。 | 检查到的新增项都是 opt-in/nondefault：competition 为 `off`，selector 为 `existing`，delay 为 `current-delay`，contribution 为 `arrival-window`，continuous implementation 为 `reference`（`model.py:438-451`；Fig.9 `:217`、`:5195-5198`）。 | 非论文诊断（NONPAPER DIAGNOSTIC） | 低（LOW） | 在本次检查的默认值中没有发现诊断模式意外进入 strict；启用这些模式得到的结果不能称作 strict reproduction。 |

## TOP 10 最高风险差距

1. **PSP 振幅没有公开：** strict 的 `response_scale=None` 会静默把 kernel 峰值归一化为 1；在当前 tau 下等价于选择 `V0` 约为 1.87。
2. **`tau_m=.10` 和 `tau_s=.02` 是本地值：** Eq. (1) 或 Table I 中都找不到这两个数值，但它们决定时间叠加和 PSP 峰值延迟。
3. **`V_inh(t)` 等价性未证实：** strict 没有完整连续轨迹，而公开论文也没有给出其数值形式。
4. **winner 等价性未证实：** strict 的最早发放不一定等价于论文中的 soma membrane-potential 最大 winner。
5. **`timing_tolerance=.03` 是高影响本地语义：** 它控制预测确认、burst 分类、Scenario matching 和 contributor credit。
6. **Scenario 1 归因是本地操作化定义：** arrival-window 可能减弱真正具有连续 PSP 因果作用的突触。
7. **突触 delay 映射不完整：** strict 的 current-delay 无法证明与论文打印的 delay 规则等价，也没有补偿 PSP 峰值延迟。
8. **Passenger SSTD 几何没有充分约束：** `[0,40000]`、中心端点、裁剪和 `sigma=spacing` 都是本地选择。
9. **Passenger decoder 完全是本地定义：** half-spacing 搜索、overlap 排序和 tie averaging 都没有在 Zhang 论文中规定。
10. **Fig.9 评价窗口无法由论文重建：** warmup 5,904 和 rolling window 400 未公开；ratio-of-sums 指标有参考文献支持，但不能与 pointwise MAPE 混淆。

## 必须明确的结论

1. **`kernel_scale` 还存在吗？** 存在。它位于 `DSDynamicsParams.kernel_scale`，并作用于每个 distal PSP。
2. **它是否等价于未公开的 `V0` 选择？** 是。在 strict `response_scale=None` 下，它选择 `V0=1/unscaled_peak`；当前 tau 下约为 1.87，使 PSP peak 恰好为 1。
3. **当前 kernel peak normalization 有论文依据吗？** 没有找到明确依据。Eq. (1) 暴露了 `V0`，而 Table I 没有规定单位峰值归一化。
4. **`tau_m/tau_s` 有论文数值依据吗？** 没有。论文把它们称为决定形状的 layer-related parameters，但没有在方程、正文或 Table I 中给出数值。
5. **`timing_tolerance=.03` 有论文依据吗？** 没有找到数值依据。它是对 approximate/same-time matching 的本地操作化。
6. **`V_inh` 是否完整实现？** 不是。strict 没有模拟完整连续轨迹；由于公开论文也没有给出数值形式，exact author-code equivalence 仍然未知，而不是已经证明 mismatch。
7. **soma winner 是否严格等价论文？** 等价性未证实。strict 选择最早预测发放，论文描述的是 soma membrane potential 最大后再抑制；当前证据无法证明二者相同，也无法证明一定失败。
8. **最值得做的三个后续实验：** (a) 预先注册 `V0 x tau_m x tau_s` sensitivity grid，把关闭 kernel normalization 作为独立 diagnostic；(b) 实现只读的完整 Eq. (3) candidate replay，用 explicit `V_inh(t)` 和 largest-soma winner 对比 earliest firing；(c) 联合审计 `timing_tolerance` 和 peak-aligned delay，统计 column-correct/time-wrong、burst、Scenario-1 attribution 和 five-step rollout。除非有独立依据，否则这些实验都不应替换 strict 默认值。
