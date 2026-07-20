from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seqmem.encoding import SSTDDiscreteEncoder
from seqmem.model import MemoryParams, SequentialMemory


def levenshtein(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, start=1):
        current = [i]
        for j, right in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            substitute = previous[j - 1] + (left != right)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


def read_cbt_sentences(path: Path, num_sentences: int, seed: int) -> list[list[str]]:
    """Select CBT sentences exactly as required by the Fig. 8 protocol."""

    eligible: list[list[str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            words = line.strip().split()
            if len(words) >= 10 and not words[0].startswith("_BOOK_"):
                eligible.append(words[:10])
    if len(eligible) < num_sentences:
        raise ValueError(
            f"CBT source has only {len(eligible)} eligible sentences; "
            f"requested {num_sentences}."
        )
    return random.Random(seed).sample(eligible, num_sentences)


def train_sentence(model: SequentialMemory, sentence: list[str]) -> None:
    model.reset_state()
    for word in sentence:
        model.predict_code()
        model.observe(word)


def recall_suffix(
    model: SequentialMemory,
    prefix: list[str],
    suffix_length: int,
) -> list[str]:
    model.reset_state()
    for word in prefix:
        model.predict_code()
        model.observe(word, learn=False)

    recalled: list[str] = []
    for _ in range(suffix_length):
        prediction = model.predict_symbol()
        if prediction is None:
            break
        recalled.append(prediction)
        model.observe(prediction, learn=False)
    return recalled


def evaluate(
    model: SequentialMemory,
    sentences: list[list[str]],
    prefix_length: int,
    eval_samples: int,
    seed: int,
) -> float:
    if eval_samples > 0 and len(sentences) > eval_samples:
        rng = random.Random(seed)
        sentences = rng.sample(sentences, eval_samples)

    distances: list[int] = []

    for sentence in sentences:
        prefix = sentence[:prefix_length]
        expected = sentence[prefix_length:]
        recalled = recall_suffix(
            model,
            prefix,
            suffix_length=len(expected),
        )
        distance = levenshtein(recalled, expected)
        distances.append(distance)
    return sum(distances) / len(distances) if distances else 0.0


def plot_capacity(path: Path, checkpoints: list[tuple[int, float]]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(
        [sentences * 10 for sentences, _distance in checkpoints],
        [distance for _sentences, distance in checkpoints],
        marker="o",
        label="DS memory",
    )
    axis.set(
        xlabel="Number of symbols",
        ylabel="Mean Levenshtein distance",
        ylim=(0, 4.05),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def run(args: argparse.Namespace) -> None:
    sentences = read_cbt_sentences(Path(args.data), args.num_sentences, args.seed)

    encoder = SSTDDiscreteEncoder(num_columns=args.num_columns, k=args.k, seed=args.seed)
    params = MemoryParams(
        l_match=args.l_match,
        forgetting_threshold=args.forgetting_threshold,
    )
    model = SequentialMemory(
        encoder=encoder,
        num_neurons_per_column=args.neurons_per_column,
        params=params,
        tie_break_seed=args.seed,
    )

    checkpoints: list[tuple[int, float]] = []
    for index, sentence in enumerate(sentences, start=1):
        train_sentence(model, sentence)
        if index % args.report_every == 0 or index == len(sentences):
            subset = sentences[:index]
            mean_distance = evaluate(
                model,
                subset,
                args.prefix_length,
                args.eval_samples,
                args.seed + index,
            )
            checkpoints.append((index, mean_distance))
            print(
                f"after {index:5d} sentences: "
                f"levenshtein={mean_distance:.3f}",
                flush=True,
            )
            if args.output_csv:
                output = Path(args.output_csv)
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        ["sentences", "mean_levenshtein"]
                    )
                    writer.writerows(checkpoints)

    _final_count, final_distance = checkpoints[-1]
    print("Fig. 8 CBT sentence memory")
    print(f"source: {args.data}")
    print(f"sentences: {len(sentences)}")
    print("sentence length: 10")
    print(f"prefix length: {args.prefix_length}")
    print(f"encoded vocabulary size: {len(encoder.known_symbols())}")
    print(f"mini-columns: {args.num_columns}")
    print(f"neurons per mini-column: {args.neurons_per_column}")
    print(f"mean Levenshtein distance: {final_distance:.3f}")
    if args.eval_samples > 0:
        print(f"evaluation samples: {min(args.eval_samples, len(sentences))}")
    print()
    print("checkpoints:")
    for index, mean_distance in checkpoints[-10:]:
        print(f"  after {index:5d} sentences: levenshtein={mean_distance:.3f}")
    if args.output_plot:
        generated = plot_capacity(Path(args.output_plot), checkpoints)
        print(f"capacity plot generated: {generated}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--data", default="data/CBTest/data/cbt_train.txt")
    parser.add_argument("--num-sentences", type=int, default=1000)
    parser.add_argument("--prefix-length", type=int, default=6)
    parser.add_argument("--num-columns", type=int, default=100)
    parser.add_argument("--neurons-per-column", type=int, default=10)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--l-match", type=int, default=3)
    parser.add_argument("--forgetting-threshold", type=float, default=500.0)
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=0,
        help="Number of sentences sampled for a diagnostic evaluation; 0 evaluates the full paper set.",
    )
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-plot", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
