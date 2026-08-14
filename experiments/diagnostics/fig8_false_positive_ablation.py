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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/CBTest/data/cbt_train.txt")
    parser.add_argument("--num-sentences", type=int, required=True)
    parser.add_argument("--response-scale", default="normalized")
    parser.add_argument(
        "--neural-propagations",
        nargs="+",
        choices=("raw", "eventwise-inhibited"),
        default=("raw", "eventwise-inhibited"),
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--details-sample-sentences", type=int, default=20)
    parser.add_argument("--trace-segments", action="store_true")
    args = parser.parse_args()

    scale = parse_response_scale(args.response_scale)
    label = response_scale_label(scale)
    sentences = read_cbt_sentences(
        Path(args.data), args.num_sentences, args.seed
    )
    model = build_model(scale, args.seed)
    reinforcement = train_once(model, sentences)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for propagation in args.neural_propagations:
        cue_rows: list[dict[str, object]] = []
        trace_path = (
            output_dir / f"segments_{propagation}.jsonl.gz"
            if args.trace_segments
            else None
        )
        metrics = evaluate_diagnostic(
            model,
            sentences,
            propagation=propagation,
            details_sample_sentences=args.details_sample_sentences,
            segment_trace_path=trace_path,
            cue_rows=cue_rows,
        )
        rows.append(
            {
                **kernel_diagnostic(scale),
                **metrics,
                **reinforcement.summary(),
            }
        )
        write_csv(output_dir / f"cue_transitions_{propagation}.csv", cue_rows)
        print(
            f"{label}/{propagation}: "
            f"distance={metrics['mean_levenshtein']:.3f}, "
            f"raw={metrics['mean_raw_events']:.3f}, "
            f"propagated={metrics['mean_propagated_events']:.3f}",
            flush=True,
        )

    write_csv(output_dir / "summary.csv", rows)


if __name__ == "__main__":
    main()
