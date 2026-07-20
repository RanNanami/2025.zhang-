from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def wape(rows: list[dict[str, str]]) -> float:
    selected = [row for row in rows if row["prediction"]]
    error = sum(
        abs(float(row["prediction"]) - float(row["target"])) for row in selected
    )
    scale = sum(abs(float(row["target"])) for row in selected)
    return error / scale if scale else 0.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_rows(
    original: list[dict[str, str]], perturbed: list[dict[str, str]]
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for text in (
        "2015-04-01",
        "2015-04-08",
        "2015-04-15",
        "2015-04-22",
        "2015-04-29",
        "2015-05-06",
    ):
        date = datetime.fromisoformat(text)
        original_row = next(
            row
            for row in original
            if datetime.fromisoformat(row["target_timestamp"]) >= date
        )
        perturbed_row = next(
            row
            for row in perturbed
            if datetime.fromisoformat(row["target_timestamp"]) >= date
        )
        original_value = float(original_row["rolling_mape"])
        perturbed_value = float(perturbed_row["rolling_mape"])
        output.append(
            {
                "date": text,
                "original_rolling_mape": original_value,
                "perturbed_rolling_mape": perturbed_value,
                "perturbation_delta": perturbed_value - original_value,
            }
        )
    return output


def plot(
    path: Path,
    original: list[dict[str, str]],
    perturbed: list[dict[str, str]],
) -> None:
    import matplotlib.pyplot as plt

    start = datetime(2015, 3, 25)
    stop = datetime(2015, 5, 8)
    figure, axis = plt.subplots(figsize=(10, 4.8))
    for label, rows in (("original", original), ("perturbed", perturbed)):
        selected = [
            row
            for row in rows
            if start <= datetime.fromisoformat(row["target_timestamp"]) <= stop
        ]
        axis.plot(
            [datetime.fromisoformat(row["target_timestamp"]) for row in selected],
            [float(row["rolling_mape"]) for row in selected],
            linewidth=1.25,
            label=label,
        )
    axis.axvline(datetime(2015, 4, 1), color="tab:orange", linestyle="--")
    axis.set(xlabel="Target time", ylabel="MAPE over last 400 predictions")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", default="results/fig9_corrected_original.csv")
    parser.add_argument(
        "--perturbed", default="results/fig9_corrected_perturbed.csv"
    )
    parser.add_argument(
        "--output-dir", default="results/fig9_corrected_comparison"
    )
    args = parser.parse_args()

    original = read_rows(Path(args.original))
    perturbed = read_rows(Path(args.perturbed))
    change = datetime(2015, 4, 1)
    original_post = [
        row for row in original if datetime.fromisoformat(row["target_timestamp"]) >= change
    ]
    perturbed_post = [
        row for row in perturbed if datetime.fromisoformat(row["target_timestamp"]) >= change
    ]
    prechange_mismatches = sum(
        left["prediction"] != right["prediction"]
        for left, right in zip(original, perturbed)
        if datetime.fromisoformat(left["target_timestamp"]) < change
    )
    summary = [
        {
            "original_overall_mape": wape(original),
            "perturbed_overall_mape": wape(perturbed),
            "original_postchange_mape": wape(original_post),
            "perturbed_postchange_mape": wape(perturbed_post),
            "original_final_rolling_mape": float(original[-1]["rolling_mape"]),
            "perturbed_final_rolling_mape": float(perturbed[-1]["rolling_mape"]),
            "prechange_prediction_mismatches": prechange_mismatches,
            "prediction_rows": len(original),
        }
    ]
    output = Path(args.output_dir)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "checkpoints.csv", checkpoint_rows(original, perturbed))
    plot(output / "comparison.png", original, perturbed)
    print(summary[0])


if __name__ == "__main__":
    main()
