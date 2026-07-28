"""Fig.9 预测 CSV 和适应曲线输出。"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


def write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    """按首行字段顺序写出预测记录；空记录不创建文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_adaptation(
    path: Path,
    rows: list[dict[str, object]],
    plot_start: datetime,
) -> bool:
    """绘制 rolling MAPE 曲线；matplotlib 不可用时返回 False。"""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    selected = [
        row
        for row in rows
        if row["rolling_mape"] != ""
        and datetime.fromisoformat(str(row["target_timestamp"])) >= plot_start
    ]
    if not selected:
        return False
    x = [datetime.fromisoformat(str(row["target_timestamp"])) for row in selected]
    y = [float(row["rolling_mape"]) for row in selected]
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.plot(x, y, linewidth=1.2, label="DS memory")
    axis.axvline(
        datetime(2015, 4, 1),
        color="tab:orange",
        linestyle="--",
        label="perturbation",
    )
    axis.set(xlabel="Target time", ylabel="MAPE over last 400 predictions")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True
