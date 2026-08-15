"""Fig.9 预测 CSV 和适应曲线输出。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from experiments.common.jsonio import write_json


@dataclass(frozen=True)
class StrictStreamOutputPaths:
    """Ordinary artifact paths for one strict Fig.9 stream."""

    output_dir: Path
    stream_label: str

    @property
    def predictions(self) -> Path:
        return self.output_dir / f"{self.stream_label}_predictions.csv"

    @property
    def summary(self) -> Path:
        return self.output_dir / f"{self.stream_label}_summary.json"

    @property
    def protocol(self) -> Path:
        return self.output_dir / f"{self.stream_label}_protocol.json"

    @property
    def interval_summary(self) -> Path:
        return self.output_dir / f"{self.stream_label}_interval_summary.csv"

    @property
    def debug_record(self) -> Path:
        return self.output_dir / f"{self.stream_label}_debug_record.json"

    def ensure_directory(self) -> None:
        """Create the shared artifact directory without touching model state."""

        self.output_dir.mkdir(parents=True, exist_ok=True)


def strict_summary_paths(output_dir: Path) -> list[Path]:
    """Return strict summaries while excluding historical compensated output."""

    historical = output_dir / "fig9_historical_compensated"
    return [
        path
        for path in (output_dir / "fig9_strict").glob("*summary.json")
        if historical not in path.parents
    ]


def write_stream_metadata(
    paths: StrictStreamOutputPaths,
    summary: dict[str, object],
    protocol: dict[str, object],
) -> None:
    """Write completed stream metadata without constructing protocol content."""

    paths.ensure_directory()
    write_json(paths.summary, summary)
    write_json(paths.protocol, protocol)


def write_initial_stream_artifacts(
    paths: StrictStreamOutputPaths,
    prediction_rows: list[dict[str, object]],
    summary: dict[str, object],
    protocol: dict[str, object],
) -> None:
    """Publish ordinary artifacts available before diagnostic finalization."""

    paths.ensure_directory()
    write_predictions(paths.predictions, prediction_rows)
    write_stream_metadata(paths, summary, protocol)


def write_stream_summary(
    paths: StrictStreamOutputPaths,
    summary: dict[str, object],
) -> None:
    """Republish a completed summary after optional diagnostics enrich it."""

    write_json(paths.summary, summary)


def write_optional_stream_results(
    paths: StrictStreamOutputPaths,
    *,
    interval_rows: list[dict[str, object]],
    debug_payload: dict[str, object] | None,
    debug_output_path: Path | None,
) -> None:
    """Write completed interval/debug results without deriving their content."""

    if interval_rows:
        write_predictions(paths.interval_summary, interval_rows)
    if debug_payload is not None:
        write_json(debug_output_path or paths.debug_record, debug_payload)


def write_runtime_artifacts(
    output_dir: Path,
    runtime: dict[str, object],
    *,
    canonical_stream: str | None = "original",
) -> None:
    """Write run metadata and preserve the canonical protocol byte copy."""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "runtime.json", runtime)
    if canonical_stream is None:
        return
    source = output_dir / f"{canonical_stream}_protocol.json"
    if source.exists():
        (output_dir / "protocol.json").write_text(
            source.read_text(encoding="utf-8"),
            encoding="utf-8",
        )


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
