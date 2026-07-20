from __future__ import annotations

import csv
from pathlib import Path


PAPER_ORIGINAL = [
    ("ARIAM", 0.280),
    ("ELM", 0.165),
    ("TDNN", 0.110),
    ("ESN", 0.100),
    ("LSTM-online", 0.103),
    ("LSTM-1000", 0.102),
    ("LSTM-3000", 0.090),
    ("LSTM-6000", 0.085),
    ("HTM", 0.080),
    ("Ours (paper)", 0.100),
]

PAPER_MODIFIED = [
    ("ELM", 0.170),
    ("ESN", 0.150),
    ("LSTM-1000", 0.150),
    ("LSTM-3000", 0.140),
    ("LSTM-6000", 0.130),
    ("HTM", 0.115),
    ("Ours (paper)", 0.100),
]


def read_local_summary(path: Path) -> tuple[float, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    return float(row["original_overall_mape"]), float(
        row["perturbed_postchange_mape"]
    )


def write_table(
    path: Path, original_local: float, modified_local: float
) -> None:
    original = dict(PAPER_ORIGINAL)
    modified = dict(PAPER_MODIFIED)
    models = list(original) + [model for model in modified if model not in original]
    rows = [
        {
            "model": model,
            "paper_original_mape_approx": original.get(model, ""),
            "paper_modified_mape_approx": modified.get(model, ""),
            "source": "digitized from Zhang Fig.9(b,c)",
        }
        for model in models
    ]
    rows.append(
        {
            "model": "Ours (local reproduction)",
            "paper_original_mape_approx": original_local,
            "paper_modified_mape_approx": modified_local,
            "source": "local original/post-change CSV",
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_panel(axis, values: list[tuple[str, float]], local: float, title: str) -> None:
    import matplotlib.pyplot as plt

    labels = [label for label, _value in values] + ["Ours (local)"]
    heights = [value for _label, value in values] + [local]
    colors = ["#6c8ebf"] * (len(values) - 1) + ["#2f6b45", "#c44e52"]
    axis.bar(range(len(labels)), heights, color=colors)
    axis.set_title(title)
    axis.set_ylabel("MAPE")
    axis.set_ylim(0.0, 0.30)
    axis.set_xticks(range(len(labels)), labels, rotation=55, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.text(
        0.01,
        0.98,
        "Paper bars are approximate digitizations",
        transform=axis.transAxes,
        va="top",
        fontsize=8,
        color="#555555",
    )


def main() -> None:
    import matplotlib.pyplot as plt

    output = Path("results/fig9_reference_comparison")
    local = read_local_summary(
        Path("results/fig9_corrected_comparison/summary.csv")
    )
    write_table(output / "bars.csv", *local)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    plot_panel(axes[0], PAPER_ORIGINAL, local[0], "Original data")
    plot_panel(axes[1], PAPER_MODIFIED, local[1], "Modified data")
    figure.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "comparison.png", dpi=180)
    plt.close(figure)
    print(f"saved {output / 'bars.csv'}")
    print(f"saved {output / 'comparison.png'}")


if __name__ == "__main__":
    main()
