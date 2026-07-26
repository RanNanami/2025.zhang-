# Fig9 Strict Density 调试指南

这份文档对应 `experiments/fig9_strict_reproduction.py` 的第二阶段调试功能。新增功能只提升可观察性和可恢复运行能力，不改变 strict 协议参数。

## 1. 一条 Fig.9 record 的调用链

`run_strict_stream()` 逐条读取 taxi record。每条记录先用 `record_values()` 转成 weekday、半小时 time slot、passenger_count，再由 `SSTDCompositeEncoder.encode()` 编成 30 个 SSTD events。

warmup 之后，当前 record 的真实值不会先进入模型，而是先调用 `rollout_raw_autonomous()` 做 5 步预测。5 步预测完成后，只对最终 raw code 调用 `passenger_encoder.decode_likelihood()` 得到 passenger 预测值。最后才调用 `learn_actual_code()`，让当前真实 record 触发 `predict_code()` + `observe_code(..., learn=True)` 的在线学习。

## 2. 五步 rollout 调用链

每个 horizon step 都是：

1. 保存当前 `previous_active_cells` / `previous_winners`。
2. 调用一次 `model.predict_code()`。
3. 从同一次 `last_prediction_candidates` 计算 raw predicted cells。
4. 用这些真实 predictive cells 推进 `previous_active_cells` 和 `previous_winners`。
5. 不 observe decoded passenger，不学习，不新建 segment。

`finally` 会调用 `restore_transient_state()`，所以 rollout 对训练主线是只读的。

## 3. density trace 在哪里采集

density trace 在 `rollout_raw_autonomous()` 每个 horizon step 采集。关键字段：

- `active_source_count`: 本步预测实际使用的 lateral context 数。
- `candidate_segment_count`: active sources 查到的候选 segment 数。
- `threshold_crossing_segment_count`: 连续 PSP 过 dendritic threshold 的 segment 数。
- `accepted_candidate_count`: 经过列内抑制后保留的 prediction candidate 数。
- `raw_predicted_column_count`: raw prediction 覆盖的不同 column 数。
- `weekday_predicted_column_count` / `time_predicted_column_count` / `passenger_predicted_column_count`: 按 composite offset 拆开的字段密度。
- `passenger_candidate_count`: passenger likelihood decode 的同分候选数量。
- `prediction_runtime_seconds` / `decode_runtime_seconds`: 每步耗时。

三个字段列数相加应该等于 `raw_predicted_column_count`。

## 4. 长期状态与 transient state

长期状态包括 `columns`、`segments`、`synapses`、`weight`、`delay`、`age` 和 `_incoming_index`。这些只应在 `observe_code(..., learn=True)` 中改变。

transient state 包括 `previous_active_cells`、`previous_winners`、`last_prediction_candidates`、`last_symbol_ranking`、`last_prediction_stats`、`last_observe_stats` 和两套 RNG state。rollout、density trace、debug JSON 和 evaluation 都不能把 transient 污染带回训练主线。

## 5. checkpoint 保存了什么

`save_strict_checkpoint()` 保存：

- encoder state；
- `SequentialMemory` 完整长期和瞬时状态；
- `last_prediction_candidates`；
- decode RNG 和 learning RNG；
- 当前 record index；
- 已累计 predictions、targets、rows、rolling errors；
- raw event/column 统计；
- protocol fingerprint；
- data file hash；
- checkpoint schema version；
- common prefix hash 和 prefix end timestamp。

恢复时会检查 schema；如果数据文件 hash 不同，会进一步检查恢复点之前的 common prefix hash，允许 original/perturbed 在 2015-04-01 后分叉。

## 6. 推荐断点

- `run_strict_stream()` 中调用 `rollout_raw_autonomous()` 前：看 prediction-before-observe。
- `rollout_raw_autonomous()` 每个 step 开始：看 `previous_active_cell_count`。
- `model.predict_code()` 返回后：看 `last_prediction_stats` 和 `last_prediction_candidates`。
- `passenger_decode_details()` 前后：看 passenger 字段是否太密。
- `model.restore_transient_state()` 前后：看 rollout 是否污染主线。
- `model.observe_code()` 的 Scenario 1/2/3 分支：看真实输入如何学习。
- `_grow_segment()`：看新 segment 是否过快增长。
- `_continuous_prediction_from_arrivals()` threshold crossing：看 PSP 是否大量过阈值。

## 7. 怎么运行 250 density

PowerShell:

```powershell
scripts/fig9_strict_250_density.ps1
```

脚本会创建类似 `results/fig9_strict/phase2_250_density_YYYYMMDD_HHMMSS/` 的新目录，不覆盖旧结果。

## 8. 怎么读 density summary

打开 `original_density_summary.json`。先看 `step_summary`：

- 如果 step 1 的 `raw_columns_mean` 已经很大，说明第一步就过密。
- 如果 step 1 到 step 5 单调增加，说明 autonomous rollout 在逐步扩散。
- 如果 `passenger_field_density_mean` 占主要部分，说明 passenger 482 列贡献最大。
- 如果 `prediction_runtime_mean` 随 step 增加，说明候选段/PSP 积分负担在扩张。

再看 `final_step_raw_density_error_correlation`，它描述第 5 步 raw density 和误差是否同向变化。

## 9. 什么现象表示 stepwise explosion

- `raw_predicted_column_count` 接近 570 的大比例；
- `passenger_predicted_column_count` 接近 482 的大比例；
- `candidate_segment_count` 和 `threshold_crossing_segment_count` 随 step 增长；
- `passenger_candidate_count` 很大，表示解码有大量同分候选；
- 第 5 步 MAPE 明显高于前几步。

## 10. strict 协议保护点

这些默认不能为了追数值而改：

- all-cell burst；
- raw neural propagation；
- prediction-before-observe；
- no future covariates；
- no decoded-value replay；
- no learning during rollout；
- horizon=5；
- 30/58/482 columns；
- neurons_per_column=32；
- `L_match=4`；
- forgetting threshold=65。

本阶段的 density trace、debug JSON、profile 和 checkpoint 都是可观察性/工程运行能力，不是补偿机制。
