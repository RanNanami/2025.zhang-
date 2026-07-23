# Fig9 手动调试指南

这份指南只解释当前 strict Fig.9 复现代码怎么单步看，不改变任何算法。实际入口是 `experiments/fig9_strict_reproduction.py`，旧的 `experiments/fig9_paper_snn.py` 主要保留历史对照和诊断模式。

## 1. 单条 strict record 的完整调用链

1. `main()`
2. `read_records()` 读取 `timestamp` 和 `passenger_count`
3. `run_strict_stream()`
4. `record_values(record)` 得到 `(weekday, half_hour_slot, passenger_count)`
5. `encoder.encode(...)` 生成三字段 composite SSTD code
6. warmup 之后先调用 `rollout_raw_autonomous(model, horizon=5)`
7. `passenger_encoder.decode_likelihood(rollout.code)` 只把第 5 步 raw code 解码成数值
8. `learn_actual_code(model, code)` 才把当前真实 record 写入模型
9. `learn_actual_code()` 内部先 `model.predict_code()`，再 `model.observe_code(code, learn=True)`

关键验证点：预测发生在 observe 当前真实记录之前。若把 `learn_actual_code()` 移到预测前面，就会把当前 target 附近信息泄漏给模型。

## 2. 5-step rollout 调用链

`rollout_raw_autonomous()` 每一步都是：

1. `raw = model.predict_code()`
2. 记录 `len(raw.events)` 和 raw predicted columns
3. `active = model.prediction_active_cells(raw)`
4. `model.previous_active_cells = active`
5. `model.previous_winners = active.copy()`

这条链路中不能调用 `observe(decoded_value, learn=False)`，也不能把 passenger 解码值重新编码。rollout 结束后 `model.restore_transient_state(snapshot)` 会恢复临时状态和 RNG。

## 3. 长期状态和瞬时状态

长期状态：

- `columns`
- `Neuron.segments`
- `Segment.synapses`
- `Synapse.weight`
- `Synapse.delay`
- `Synapse.age`
- `_incoming_index`

瞬时状态：

- `previous_active_cells`
- `previous_winners`
- `last_prediction_candidates`
- `last_symbol_ranking`
- `previous_predicted_sources`
- `previous_burst_only_sources`
- `_decode_rng` 状态
- `_learning_rng` 状态

rollout/evaluate 只能改瞬时状态，并且结束后必须恢复；`observe_code(..., learn=True)` 才允许改长期状态。

## 4. 关键类和变量速查

- `SymbolCode`: 一组 `(column, time)` spikes。
- `SpikeEvent`: 一个 mini-column 在 SSTD 周期中的发放时间。
- `SSTDCompositeEncoder`: 把 weekday/time/passenger 三字段合成一个 record code。
- `PredictionCandidate`: 一次 `predict_code()` 中真实 crossing 的候选 neuron。
- `Segment`: distal dendritic memory branch。
- `Synapse`: 从旧 active cell 到 segment 的延迟连接。
- `previous_active_cells`: 上一步所有 active cells；burst 时可能包含整列。
- `previous_winners`: 学习 winners；通常比 active cells 稀疏。
- `last_prediction_candidates`: 本次预测候选，decode、advance、Scenario 1 都应复用它。

## 5. 推荐断点

- `experiments/fig9_strict_reproduction.py`: `run_strict_stream()` 中 `event_hook("predict", index)` 前。
- `experiments/fig9_strict_reproduction.py`: `rollout_raw_autonomous()` 的 horizon step 开始。
- `src/seqmem/model.py`: `predict_code()` 清空并重建 `last_prediction_candidates` 后。
- `experiments/fig9_strict_reproduction.py`: `decode_likelihood()` 调用前后。
- `experiments/fig9_strict_reproduction.py`: `rollout_raw_autonomous()` 的 `finally` 恢复前后。
- `experiments/fig9_strict_reproduction.py`: `learn_actual_code(model, code)` 前。
- `src/seqmem/model.py`: `observe_code()` 的 Scenario 1/2/3 三个分支。
- `src/seqmem/model.py`: `_grow_segment()` 创建 segment 处。
- `src/seqmem/model.py`: `_continuous_prediction_from_arrivals()` threshold crossing 处。
- `src/seqmem/model.py`: `snapshot_transient_state()` 和 `restore_transient_state()`。

## 6. 每个断点建议观察的变量

- 预测前：`index`, `record.timestamp`, `target_record.timestamp`, `len(model.previous_active_cells)`。
- horizon step：`_step_index`, `len(raw.events)`, raw column 数。
- `predict_code()` 后：`len(model.last_prediction_candidates)`, 每列 candidate 数、candidate score/peak/crossing time。
- passenger decode 前后：`rollout.code.events`, passenger 字段列数、`prediction`。
- observe 前：确认当前 record 还没被 `learn_actual_code()` 写入。
- Scenario 分支：`was_predicted`, `matched`, `timed_overlap`, `learn`。
- segment creation：`column_id`, `neuron_index`, `len(active_sources)`, `target_time`。
- threshold crossing：`crossing`, `peak_dendritic_potential`, `predicted_soma_firing_time`。
- restore 前后：`previous_active_cells`, `last_prediction_candidates`, RNG state 是否回到 snapshot。

## 7. 定位 raw columns explosion

先看 `rollout_raw_autonomous()` 输出的 `raw_rollout_column_counts`。如果第 1 步正常、第 2 到第 5 步快速变大，继续进 `predict_code()` 看：

- `active_sources` 是否过大；
- `best_by_event` 是否覆盖大量不同 column；
- `last_prediction_candidates` 是否很多列都有接近阈值的 candidate；
- `_active_sources()` 是否因为 burst_context=True 返回了大量 previous_active_cells。

## 8. 定位 burst expansion

进入 `observe_code()`，看真实 event 是否命中 `matching_predictions`。若没有命中：

- 未预测列会进入 burst；
- `active_cells` 会加入该列所有 neuron；
- 下一步 `_active_sources()` 会把这些 burst cells 作为 context。

Fig.8 诊断中还可以看 `burst_cell_count`、`burst_column_count` 和 cue transition 的 raw column 密度。

## 9. 判断 passenger decode 是否被无关列污染

在 `SSTDRealValueEncoder.decode_likelihood()` 看：

- `predicted_times` 中 passenger 范围内列数；
- `best_score=(timed_overlap, column_overlap)` 是否很低；
- `best_values` 是否很多；
- raw code 是否混入大量 weekday/time 之外的 passenger 无关列。

如果 raw passenger 列太密，likelihood decode 可能得到很多同分候选，最后平均值会偏离真实 passenger_count。

## 10. 验证 prediction-before-observe

在 `run_strict_stream()` 中同一条 record 应先进入 `rollout_raw_autonomous()`，再执行 `learn_actual_code(model, code)`。在 `learn_actual_code()` 内部也应先 `model.predict_code()`，再 `model.observe_code(code, learn=True)`。

## 11. 验证 rollout 无学习

在 `rollout_raw_autonomous()` 期间观察：

- `len(neuron.segments)` 不应增加；
- `Synapse.weight` 不应改变；
- `Synapse.age` 不应改变；
- `_incoming_index` 不应新增 source；
- 不能调用 `observe_code(..., learn=True)`。

## 12. 验证 checkpoint/restore 一致性

当前 strict runner 主要通过 protocol/summary/fingerprint 记录协议。若你在诊断脚本里加 checkpoint，保存前后要比对：

- protocol JSON 中 strict flags；
- `snapshot_transient_state()` 覆盖的字段；
- restore 后下一次 `predict_code()` 是否与未评估路径一致；
- learning RNG 是否没有被评估消耗。

## 13. 常用运行命令

PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD"
.\.venv\Scripts\python.exe -m compileall -q src experiments tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe experiments\fig9_strict_reproduction.py --limit 250 --warmup 50 --streams original --output-dir tmp\fig9_debug_original
.\.venv\Scripts\python.exe experiments\fig9_strict_reproduction.py --limit 250 --warmup 50 --streams original perturbed --output-dir tmp\fig9_debug_pair
```

## 14. strict 与 historical compensated 的区别

strict 路径保持：

- all-cell burst；
- raw neural propagation；
- no future covariates；
- no decoded replay；
- no rollout learning；
- horizon=5；
- weekday/time/passenger = 30/58/482；
- neurons/column=32；
- `L_match=4`；
- forgetting threshold=65。

historical compensated 结果是旧阶段为了诊断或追齐数值做过的补偿/对照，不应和 strict Fig.9 混在一起汇报。
