from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path


PAPER_MODELS = [
    "HTM",
    "ELM",
    "TDNN",
    "LSTM-1000",
    "LSTM-3000",
    "LSTM-9000",
    "LSTM-online100",
]


def load_reference(path: Path) -> dict[str, dict[str, object]]:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def reference_rows(reference: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in PAPER_MODELS:
        result = reference[model]
        for element, mean, std in zip(
            result["x"], result["meanAccuracy"], result["stdAccuracy"]  # type: ignore[arg-type]
        ):
            if int(element) <= 20000:
                rows.append(
                    {
                        "model": model,
                        "elements": int(element),
                        "mean_moving_accuracy": float(mean),
                        "std_moving_accuracy": float(std),
                        "source": "Cui et al. [58] processed result",
                    }
                )
    return rows


def load_snn_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {
                "model": "DS-memory reproduction",
                "elements": int(row["elements"]),
                "mean_moving_accuracy": float(row["mean_moving_accuracy"]),
                "std_moving_accuracy": float(row["std_moving_accuracy"]),
                "source": "local strict reproduction",
            }
            for row in csv.DictReader(handle)
            if row["task"] == "single"
        ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def endpoint_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for model in PAPER_MODELS + ["DS-memory reproduction"]:
        selected = sorted(
            (row for row in rows if row["model"] == model),
            key=lambda row: int(row["elements"]),
        )
        if not selected:
            continue
        pre = [row for row in selected if int(row["elements"]) <= 10000][-1]
        final = [row for row in selected if int(row["elements"]) <= 20000][-1]
        summary.append(
            {
                "model": model,
                "pre_change_elements": pre["elements"],
                "pre_change_accuracy": pre["mean_moving_accuracy"],
                "pre_change_std": pre["std_moving_accuracy"],
                "final_elements": final["elements"],
                "final_accuracy": final["mean_moving_accuracy"],
                "final_std": final["std_moving_accuracy"],
                "source": final["source"],
            }
        )
    return summary


def plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 5.4))
    models = PAPER_MODELS + ["DS-memory reproduction"]
    for model in models:
        selected = [row for row in rows if row["model"] == model]
        if not selected:
            continue
        x = [int(row["elements"]) for row in selected]
        mean = [float(row["mean_moving_accuracy"]) for row in selected]
        std = [float(row["std_moving_accuracy"]) for row in selected]
        width = 2.2 if model == "DS-memory reproduction" else 1.1
        axis.plot(x, mean, linewidth=width, label=model)
        if model == "DS-memory reproduction":
            axis.fill_between(
                x,
                [max(0.0, value - spread) for value, spread in zip(mean, std)],
                [min(1.0, value + spread) for value, spread in zip(mean, std)],
                alpha=0.2,
            )
    axis.axvline(10000, color="tab:orange", linestyle="--", label="modification")
    axis.set(
        xlim=(0, 20000),
        ylim=(0, 1.02),
        xlabel="Number of elements seen",
        ylabel="Moving accuracy over 100 sequences",
    )
    axis.grid(alpha=0.22)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        default=".deps/ContinuousLearnExperiment.pkl",
        help="Processed result committed by the reference [58] authors.",
    )
    parser.add_argument(
        "--snn-curves",
        default="results/fig7_strict_corrected/curves.csv",
    )
    parser.add_argument("--output-dir", default="results/fig7_reference_comparison")
    args = parser.parse_args()

    rows = reference_rows(load_reference(Path(args.reference)))
    rows.extend(load_snn_rows(Path(args.snn_curves)))
    output = Path(args.output_dir)
    write_csv(output / "curves.csv", rows)
    write_csv(output / "summary.csv", endpoint_summary(rows))
    plot(output / "comparison.png", rows)
    print(f"saved {len(rows)} points to {output}")


if __name__ == "__main__":
    main()
