from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from fig8_sentence_memory import (  # noqa: E402
    DETAIL_FIELDS,
    evaluate,
    plot_capacity,
    read_cbt_sentences,
    train_sentence,
)
from seqmem.encoding import SSTDDiscreteEncoder  # noqa: E402
from seqmem.model import MemoryParams, SequentialMemory  # noqa: E402


def write_summary(path: Path, sentence_count: int, distance: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sentences", "mean_levenshtein"])
        writer.writerow([sentence_count, distance])


def write_details(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path,
    model: SequentialMemory,
    sentences: list[list[str]],
    trained_sentences: int,
    settings: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(
            {
                "model": model,
                "sentences": sentences,
                "trained_sentences": trained_sentences,
                "settings": settings,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    temporary.replace(path)


def load_checkpoint(
    path: Path, expected_settings: dict[str, object]
) -> tuple[SequentialMemory, list[list[str]], int]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload["settings"] != expected_settings:
        raise ValueError(
            "checkpoint settings do not match this command: "
            f"{payload['settings']!r} != {expected_settings!r}"
        )
    return (
        payload["model"],
        payload["sentences"],
        payload["trained_sentences"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/CBTest/data/cbt_train.txt")
    parser.add_argument("--num-sentences", type=int, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--prefix-length", type=int, default=6)
    parser.add_argument("--num-columns", type=int, default=100)
    parser.add_argument("--neurons-per-column", type=int, default=10)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--l-match", type=int, default=3)
    parser.add_argument("--forgetting-threshold", type=float, default=500.0)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--neural-plot", default="")
    parser.add_argument("--model-checkpoint", default="")
    parser.add_argument("--resume-model", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--evaluation-progress-every", type=int, default=50)
    parser.add_argument(
        "--retrieval-modes",
        nargs="+",
        choices=("proximal-replay", "neural"),
        default=("proximal-replay", "neural"),
    )
    args = parser.parse_args()

    settings = {
        "data": str(Path(args.data).resolve()),
        "num_sentences": args.num_sentences,
        "seed": args.seed,
        "prefix_length": args.prefix_length,
        "num_columns": args.num_columns,
        "neurons_per_column": args.neurons_per_column,
        "k": args.k,
        "l_match": args.l_match,
        "forgetting_threshold": args.forgetting_threshold,
    }
    checkpoint_path = Path(args.model_checkpoint) if args.model_checkpoint else None
    if args.resume_model:
        if checkpoint_path is None:
            parser.error("--resume-model requires --model-checkpoint")
        model, sentences, trained_sentences = load_checkpoint(
            checkpoint_path, settings
        )
        print(
            f"loaded checkpoint after {trained_sentences}/{len(sentences)} sentences",
            flush=True,
        )
    else:
        sentences = read_cbt_sentences(
            Path(args.data), args.num_sentences, args.seed
        )
        model = SequentialMemory(
            encoder=SSTDDiscreteEncoder(
                num_columns=args.num_columns,
                k=args.k,
                seed=args.seed,
            ),
            num_neurons_per_column=args.neurons_per_column,
            params=MemoryParams(
                l_match=args.l_match,
                forgetting_threshold=args.forgetting_threshold,
            ),
            tie_break_seed=args.seed,
        )
        trained_sentences = 0

    for index, sentence in enumerate(
        sentences[trained_sentences:], start=trained_sentences + 1
    ):
        train_sentence(model, sentence)
        if index % 100 == 0 or index == len(sentences):
            print(f"trained {index}/{len(sentences)}", flush=True)
            if checkpoint_path is not None:
                save_checkpoint(
                    checkpoint_path, model, sentences, index, settings
                )

    if args.train_only:
        if checkpoint_path is None:
            parser.error("--train-only requires --model-checkpoint")
        print(f"training complete; checkpoint={checkpoint_path}", flush=True)
        return

    prefix = Path(args.output_prefix)
    for mode, label in (
        ("proximal-replay", "proximal"),
        ("neural", "neural"),
    ):
        if mode not in args.retrieval_modes:
            continue
        details: list[dict[str, object]] = []

        def report_progress(completed: int, total: int) -> None:
            if (
                args.evaluation_progress_every > 0
                and (
                    completed % args.evaluation_progress_every == 0
                    or completed == total
                )
            ):
                print(
                    f"{mode}: evaluated {completed}/{total}", flush=True
                )

        distance = evaluate(
            model,
            sentences,
            args.prefix_length,
            eval_samples=0,
            seed=args.seed + args.num_sentences,
            retrieval_mode=mode,
            details_rows=details,
            checkpoint_sentences=args.num_sentences,
            progress_callback=report_progress,
        )
        summary_path = Path(f"{prefix}_{label}_{args.num_sentences}.csv")
        details_path = Path(
            f"{prefix}_{label}_{args.num_sentences}_details.csv"
        )
        write_summary(summary_path, args.num_sentences, distance)
        write_details(details_path, details)
        print(f"{mode}: mean Levenshtein={distance:.3f}", flush=True)

        if mode == "neural" and args.neural_plot:
            plot_capacity(
                Path(args.neural_plot),
                [(args.num_sentences, distance)],
            )


if __name__ == "__main__":
    main()
