from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.diagnostics.fig8_diagnostic_common import (  # noqa: E402
    build_model,
    kernel_diagnostic,
    parse_response_scale,
    response_scale_label,
    train_once,
)
from experiments.fig8_sentence_memory import read_cbt_sentences  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/CBTest/data/cbt_train.txt")
    parser.add_argument("--num-sentences", type=int, required=True)
    parser.add_argument("--response-scales", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    sentences = read_cbt_sentences(Path(args.data), args.num_sentences, args.seed)
    rows: list[dict[str, object]] = []
    for raw_scale in args.response_scales:
        scale = parse_response_scale(raw_scale)
        label = response_scale_label(scale)
        print(f"training response_scale={label}", flush=True)
        model = build_model(scale, args.seed)
        reinforcement = train_once(model, sentences)
        rows.append(
            {
                "diagnostic_label": "nonpaper diagnostic",
                "sentences": args.num_sentences,
                **kernel_diagnostic(scale),
                **reinforcement.summary(),
            }
        )

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
