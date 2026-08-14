from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.common.csvio import write_csv_rows  # noqa: E402
from experiments.diagnostics.fig8_diagnostic_common import (  # noqa: E402
    build_model,
    evaluate_diagnostic,
    kernel_diagnostic,
    parse_response_scale,
    response_scale_label,
    train_once,
)
from experiments.fig8_sentence_memory import read_cbt_sentences  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    write_csv_rows(path, rows)


def plot_comparison(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    labels = [str(row["response_scale"]) for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    series = (
        ("mean_levenshtein", "Mean Levenshtein"),
        ("mean_raw_columns", "Mean raw predicted columns"),
        ("two_contributor_crossing_fraction", "Two-contributor fraction"),
        ("expected_absent_rate", "Expected-word absent rate"),
    )
    for axis, (key, title) in zip(axes.flat, series):
        axis.plot(labels, [float(row[key]) for row in rows], marker="o")
        axis.set_title(title)
        axis.set_xlabel("response scale (nonpaper diagnostic)")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Fig.8 Response-Scale Sensitivity",
        "",
        "All numeric response scales are nonpaper diagnostics. The strict default",
        "remains `normalized` (`response_scale=None`). Each sentence is trained once.",
        "",
        "| scale | kernel peak | minimum initial synapses | Levenshtein | "
        "raw columns | expected absent | two contributors | no prediction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {response_scale} | {kernel_peak:.6f} | "
            "{min_initial_synapses_to_threshold} | {mean_levenshtein:.3f} | "
            "{mean_raw_columns:.3f} | {expected_absent_rate:.3f} | "
            "{two_contributor_crossing_fraction:.3f} | "
            "{no_prediction_rate:.3f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/CBTest/data/cbt_train.txt")
    parser.add_argument("--num-sentences", type=int, required=True)
    parser.add_argument("--response-scales", nargs="+", required=True)
    parser.add_argument("--checkpoints", nargs="*", type=int, default=[])
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--details-sample-sentences", type=int, default=20)
    parser.add_argument("--trace-segments", action="store_true")
    args = parser.parse_args()

    if args.checkpoints and args.num_sentences not in args.checkpoints:
        parser.error(
            "the current diagnostic runner requires the final count in --checkpoints"
        )
    sentences = read_cbt_sentences(
        Path(args.data), args.num_sentences, args.seed
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    kernel_rows: list[dict[str, object]] = []

    for raw_scale in args.response_scales:
        scale = parse_response_scale(raw_scale)
        label = response_scale_label(scale)
        print(f"training response_scale={label}", flush=True)
        model = build_model(scale, args.seed)
        reinforcement = train_once(model, sentences)
        cue_rows: list[dict[str, object]] = []
        trace_path = (
            output_dir / f"segments_{label}.jsonl.gz"
            if args.trace_segments
            else None
        )
        metrics = evaluate_diagnostic(
            model,
            sentences,
            propagation="raw",
            details_sample_sentences=args.details_sample_sentences,
            segment_trace_path=trace_path,
            cue_rows=cue_rows,
        )
        kernel = kernel_diagnostic(scale)
        row = {**kernel, **metrics, **reinforcement.summary()}
        rows.append(row)
        kernel_rows.append(kernel)
        write_csv(output_dir / f"cue_transitions_{label}.csv", cue_rows)
        print(
            f"{label}: distance={metrics['mean_levenshtein']:.3f}, "
            f"raw_columns={metrics['mean_raw_columns']:.3f}",
            flush=True,
        )

    write_csv(output_dir / "summary.csv", rows)
    write_csv(output_dir / "kernel_peaks.csv", kernel_rows)
    canonical = output_dir.parent / f"response_scale_{args.num_sentences}_summary.csv"
    write_csv(canonical, rows)
    if args.num_sentences == 100:
        plot_comparison(output_dir.parent / "response_scale_comparison.png", rows)
        write_report(output_dir.parent / "response_scale_report.md", rows)


if __name__ == "__main__":
    main()
