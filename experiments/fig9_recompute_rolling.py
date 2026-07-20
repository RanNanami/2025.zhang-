from __future__ import annotations

import argparse
import csv
from collections import deque
from datetime import datetime
from pathlib import Path

try:
    from experiments.fig9_paper_snn import plot_adaptation
except ModuleNotFoundError:
    from fig9_paper_snn import plot_adaptation


def recompute(path: Path, window: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No prediction rows in {path}")

    normalized_errors: deque[float] = deque(maxlen=window)
    for row in rows:
        if not row["prediction"]:
            row["rolling_mape"] = ""
            continue
        normalized = row.get("normalized_absolute_error", "")
        if normalized == "":
            raise ValueError(
                "normalized_absolute_error is required to recover the "
                "global target scale used by reference [58]"
            )
        normalized_errors.append(float(normalized))
        row["rolling_mape"] = str(
            sum(normalized_errors) / len(normalized_errors)
        )

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        default=[
            "results/fig9_corrected_original.csv",
            "results/fig9_corrected_perturbed.csv",
        ],
    )
    parser.add_argument("--window", type=int, default=400)
    parser.add_argument("--plot-start", default="2015-03-25")
    args = parser.parse_args()

    for text in args.paths:
        path = Path(text)
        rows = recompute(path, args.window)
        plot_path = path.with_suffix(".png")
        plotted = plot_adaptation(
            plot_path, rows, datetime.fromisoformat(args.plot_start)
        )
        print(f"updated {path}: rows={len(rows)}, plot={plotted}")


if __name__ == "__main__":
    main()
