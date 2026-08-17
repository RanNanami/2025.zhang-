# 论文—代码 Review 检查清单

审阅基线：`d33b02506417c1fda7bf0484ce03c68959046991`。建议先阅读 `REVIEWER_GUIDE.md`。勾选一项只表示审阅者已经对照论文文字和可执行行为完成比较，不表示该实现已经与论文完全等价。

| 审阅项目 | 论文位置 | 代码位置 | 当前状态 |
|---|---|---|---|
| [ ] Eq.(1) PSP 振幅 / `V0` | Sec. II-B1、Eq.(1)；Table I 未给出 | `src/seqmem/dynamics.py::DSDynamicsParams.kernel_scale`；`spike_response`；`src/seqmem/model.py::MemoryParams.response_scale` | **本地实现（LOCAL IMPLEMENTATION）**：`None` 表示 `1/unscaled_peak`，不是“没有缩放”；当前有效 `V0` 约为 1.87，PSP 峰值恰为 1。 |
| [ ] `tau_m` / `tau_s` | Sec. II-B1、Eq.(1)；Table I 未给出具体数值 | `src/seqmem/dynamics.py::DSDynamicsParams`；`src/seqmem/model.py::MemoryParams` | **本地实现（LOCAL IMPLEMENTATION）**：`.10/.02` 是仓库假设，不能确认是 Table I 的数值。 |
| [ ] 树突阈值 | Sec. II-B1、Eq.(1)；Table I 的 `theta_d=1.0` | `MemoryParams.dendrite_threshold`；`reference_continuous_prediction` | **部分对齐（PARTIAL）**：数值一致；代码还会用电压容差降低有效阈值。 |
| [ ] 相位前移（phase precession） | Sec. II-B4；Fig. 2(e)-2(f)、Fig. 5(d)-5(e) | `src/seqmem/dynamics.py::DSNeuronState.oscillation` | **部分对齐（PARTIAL）**：机制一致；具体的、以 crossing 为锚点的 `-cos` 轨迹属于本地实现。 |
| [ ] 胞体阈值 | Sec. II-B5、Eq.(3)；Table I 的 `theta=1.0` | `DSDynamicsParams.soma_threshold`；`model.py` 中的连续胞体搜索 | **部分对齐（PARTIAL）**：数值一致；电压 epsilon 和候选状态处理属于本地实现。 |
| [ ] `V_inh` | Sec. II-B5、Eq.(3) | `DSNeuronState.membrane_potential`；`SequentialMemory` 中的 strict 调用 | **实现不完整 / 等价性未证实**：strict 没有完整的连续轨迹；公开论文也没有给出其数值形式。 |
| [ ] winner 语义 | Sec. II-B5：选择 mini-column 中 soma 电压最大的神经元并抑制同列其他神经元 | `SequentialMemory.predict_code`、`winner_by_column` 代码块 | **等价性未证实**：strict 选择最早发放；不要声称二者等价，也不要直接断言最终科学行为必然错误。 |
| [ ] 时间容差 | Scenario 2 的“同一时间”；论文未公布数值窗口 | `MemoryParams.timing_tolerance`；prediction、observe、Scenario matching 和 contributors | **高影响本地语义（HIGH-IMPACT LOCAL SEMANTIC）**：`.03` 同时影响预测确认、burst 分类、匹配和 Scenario-1 贡献归因。 |
| [ ] 积分电压容差 | Eq.(1)/(3) 中的精确阈值不等式 | `MemoryParams.integration_voltage_tolerance`；连续 crossing solver | **数值实现参数（NUMERICAL IMPLEMENTATION PARAMETER）**：`2e-5` 是数值阈值 epsilon，与语义时间容差不同。 |
| [ ] Scenario 1 贡献突触 | Sec. II-D1、Scenario 1 | `scenario1_contributing_sources`；`_reinforce_segment` | **本地实现 / 操作化定义**：论文说“产生贡献的突触”，但没有给出可执行判据。 |
| [ ] Scenario 2 匹配 | Sec. II-D1、Scenario 2、matching segment 和 `L_match` | `_best_matching_neuron`；`Segment.timed_overlap` | **部分对齐（PARTIAL）**：`L_match` 数值一致；同一时间的资格判定属于本地实现。 |
| [ ] Scenario 2B 建立 segment | Sec. II-D1、Scenario 2、没有 matching segment 时 | `observe_code`；`_grow_segment` | **部分对齐（PARTIAL）**：机制可以对应；旧代码的计数器把该分支称为 `scenario3`。 |
| [ ] Scenario 3 惩罚 | Sec. II-D1、Scenario 3 | `_punish_wrong_predictions` | **部分对齐（PARTIAL）**：权重减小规则一致；贡献突触的时间判据属于本地实现。 |
| [ ] 突触延迟 | Scenario 2 打印出的新突触 delay 规则 | `_new_synapse_delay` | **部分对齐（PARTIAL）**：strict 的 normalized `current-delay` 映射无法完全由公开文字推出。 |
| [ ] burst context | Sec. II-A/II-B2：没有预测神经元时所有细胞 burst | `_active_sources`；`observe_code` 中的 burst 代码块 | **部分对齐（PARTIAL）**：全神经元发放是论文明确内容；将每个 burst 神经元作为完整下一步 distal context 的语义没有被论文充分指定。 |
| [ ] 遗忘 | Sec. II-D；Sec. III-A；Table I | `_reinforce_segment`；`_punish_wrong_predictions`；`_prune_neuron`；`_forgetting_helpers.py` | **部分对齐（PARTIAL）**：公开的权重、年龄和阈值可以对应；确切事件顺序属于实现细节。 |
| [ ] Fig.9 encoder | Sec. II-C1/2；Sec. III-D：30/58/482、`K=10` | `build_fig9_encoder`；`SSTDRealValueEncoder` | **本地实现（LOCAL IMPLEMENTATION）**：乘客范围、中心点、裁剪和 sigma 未在论文中公开。 |
| [ ] Fig.9 decoder | Sec. III-D：“most likely prediction”；没有 decoder 方程 | `SSTDRealValueEncoder.decode_likelihood` | **本地实现（LOCAL IMPLEMENTATION）**：半间距网格、overlap 排序和并列平均都是本地规则。 |
| [ ] Fig.9 指标 | Sec. III-D：MAPE 并引用 [58] | `experiments/fig9/metrics.py::mape` | **参考文献支持（REFERENCE-SUPPORTED）**：[58] 使用代码中的 ratio-of-sums；它不同于普通 pointwise MAPE。 |
| [ ] rollout | Sec. III-D：五步 / 2.5 小时，在线数据流 | `rollout_raw_autonomous`；`run_strict_stream` | **部分对齐（PARTIAL）**：预测步数和 raw autonomy 一致；snapshot、warmup 和 evaluation 状态约定属于本地实现。 |

## Reviewer 签字确认

- [ ] 我已经区分论文明确规定的行为和仓库的 strict 默认值。
- [ ] 我没有把 oracle / diagnostic 模式当作 strict 等价性的证据。
- [ ] 我把每一个拟议的科学改动都记录为独立实验或独立补丁。
- [ ] 在接受数值结论前，我已经检查 `reports/paper_code_diff/HIGH_IMPACT_SCIENTIFIC_GAPS.md`。
