from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRICS = ("goal_based", "contextual", "feedback_character")


def read_rows(path: Path) -> dict[int, dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            int(row["poems"]): {
                key: float(value) for key, value in row.items() if key != "poems"
            }
            for row in csv.DictReader(handle)
        }


def compare(
    control: dict[int, dict[str, float]], fixed: dict[int, dict[str, float]]
) -> list[dict[str, float | int]]:
    checkpoints = sorted(set(control) & set(fixed))
    if not checkpoints:
        raise ValueError("Control and fixed result files have no common checkpoints.")
    rows: list[dict[str, float | int]] = []
    for poems in checkpoints:
        row: dict[str, float | int] = {"poems": poems}
        for metric in METRICS:
            before = control[poems][metric]
            after = fixed[poems][metric]
            row[f"control_{metric}"] = before
            row[f"fixed_{metric}"] = after
            row[f"improvement_{metric}"] = before - after
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(path: Path, rows: list[dict[str, float | int]]) -> None:
    import matplotlib.pyplot as plt

    labels = {
        "goal_based": "Goal-based character retrieval",
        "contextual": "Contextual character retrieval",
        "feedback_character": "True-line character feedback",
    }
    x = [int(row["poems"]) for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    for axis, metric in zip(axes, METRICS):
        axis.plot(
            x,
            [float(row[f"control_{metric}"]) for row in rows],
            color="tab:gray",
            marker="o",
            label="1 source neuron",
        )
        axis.plot(
            x,
            [float(row[f"fixed_{metric}"]) for row in rows],
            color="tab:green",
            marker="o",
            label="10 source neurons",
        )
        axis.set(title=labels[metric], xlabel="Number of poems", xticks=x)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean Levenshtein distance")
    axes[0].legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", default="results/fig8c_stress_control.csv")
    parser.add_argument("--fixed", default="results/fig8c_stress_fixed.csv")
    parser.add_argument("--output-csv", default="results/fig8c_stress_comparison.csv")
    parser.add_argument("--output-plot", default="results/fig8c_stress_comparison.png")
    args = parser.parse_args()

    rows = compare(read_rows(Path(args.control)), read_rows(Path(args.fixed)))
    write_csv(Path(args.output_csv), rows)
    plot(Path(args.output_plot), rows)
    print(f"Compared {len(rows)} shared capacity checkpoints.")


if __name__ == "__main__":
    main()
