from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from fig7_sequence_prediction import (
    build_sequence_set,
    ending_count,
    make_stream,
    paper_forgetting_threshold,
)
from seqmem.encoding import SSTDDiscreteEncoder
from seqmem.model import MemoryParams, SequentialMemory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["single", "multiple-2", "multiple-4"], default="multiple-4")
    parser.add_argument("--elements", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    sequences = build_sequence_set(args.task)
    stream = make_stream(
        sequences,
        args.elements,
        50000,
        random.Random(args.seed),
        -1,
    )
    encoder = SSTDDiscreteEncoder(200, 10, seed=args.seed)
    vocabulary = {symbol for sequence in sequences for symbol in sequence}
    for symbol in sorted(vocabulary, key=int):
        encoder.encode(symbol)
    model = SequentialMemory(
        encoder,
        num_neurons_per_column=10,
        tie_break_seed=args.seed,
        params=MemoryParams(
            l_match=3,
            forgetting_threshold=paper_forgetting_threshold(args.task),
        ),
    )
    for symbol in stream.symbols:
        model.predict_code()
        model.observe(symbol)

    expected_by_prefix: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for sequence in sequences:
        expected_by_prefix[tuple(sequence[:-1])].add(sequence[-1])

    all_recalled = True
    for prefix, expected in expected_by_prefix.items():
        model.reset_state()
        for symbol in prefix:
            model.predict_code()
            model.observe(symbol, learn=False)
        recalled = set(
            model.predict_symbols(
                max_predictions=ending_count(args.task),
                candidate_symbols=vocabulary,
            )
        )
        correct = expected <= recalled
        all_recalled = all_recalled and correct
        print(
            f"prefix={' '.join(prefix)} expected={sorted(expected)} "
            f"recalled={sorted(recalled)} complete={correct}"
        )
        if not correct:
            print(f"  ranking={model.last_symbol_ranking[:10]}")
    print(f"all contexts complete: {all_recalled}")


if __name__ == "__main__":
    main()
