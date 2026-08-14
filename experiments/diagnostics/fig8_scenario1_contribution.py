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
    train_once,
)
from experiments.fig8_sentence_memory import read_cbt_sentences  # noqa: E402


MODES = (
    "arrival-window",
    "continuous-positive",
    "continuous-causal",
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    write_csv_rows(path, rows, skip_empty=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/CBTest/data/cbt_train.txt")
    parser.add_argument("--num-sentences", type=int, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--details-sample-sentences", type=int, default=20)
    args = parser.parse_args()

    sentences = read_cbt_sentences(Path(args.data), args.num_sentences, args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []

    for mode in args.modes:
        print(f"training scenario1_contribution_mode={mode}", flush=True)
        model = build_model(
            1.0,
            args.seed,
            scenario1_contribution_mode=mode,
            capture_prediction_contributions=True,
        )
        reinforcement = train_once(model, sentences)
        cue_rows: list[dict[str, object]] = []
        metrics = evaluate_diagnostic(
            model,
            sentences,
            propagation="raw",
            details_sample_sentences=args.details_sample_sentences,
            segment_trace_path=None,
            cue_rows=cue_rows,
        )
        summary = {
            "diagnostic_label": "nonpaper diagnostic",
            "scenario1_contribution_mode": mode,
            **kernel_diagnostic(1.0),
            **reinforcement.summary(),
            **metrics,
        }
        summaries.append(summary)
        write_csv(output_dir / f"cue_transitions_{mode}.csv", cue_rows)
        write_csv(
            output_dir / f"scenario1_events_{mode}.csv",
            [
                {
                    "diagnostic_label": "nonpaper diagnostic",
                    "sentences": args.num_sentences,
                    **row,
                }
                for row in reinforcement.rows
            ],
        )
        write_csv(output_dir / "summary.csv", summaries)
        print(
            f"{mode}: distance={metrics['mean_levenshtein']:.3f}, "
            f"raw_columns={metrics['mean_raw_columns']:.3f}, "
            "actual_positive_weakened="
            f"{summary['scenario1_actual_positive_weakened_fraction']:.3f}",
            flush=True,
        )

if __name__ == "__main__":
    main()
