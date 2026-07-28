# Fig.9 模块边界

这个包只放 strict 与其他实验都能安全复用的稳定功能：

| 模块 | 职责 | 是否修改模型 |
| --- | --- | --- |
| `data.py` | `TaxiRecord`、CSV 读取、三字段转换 | 否 |
| `metrics.py` | 参考实现的全局和 rolling 误差 | 否 |
| `outputs.py` | 预测 CSV 和适应曲线 | 否 |
| `learning.py` | prediction-before-observe 在线学习入口 | 是 |

strict 入口 `experiments/fig9_strict_reproduction.py` 只能从
`experiments.fig9` 导入这些 helper，不能 import historical runner。

以下行为属于历史或 nonpaper 诊断，位于
`experiments/historical/fig9_paper_snn.py`：

- compensated/growth-only learning；
- known future weekday/time context；
- sparse、coherent 和 eventwise projection；
- beam 和 proximal replay；
- 旧 historical `run`/`rollout`。

`experiments/fig9_paper_snn.py` 是兼容 shim。已有 import 和 CLI 仍能使用，
但新代码应显式选择稳定包或 historical 包，避免再次混合两种协议。
