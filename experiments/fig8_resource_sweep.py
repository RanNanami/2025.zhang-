from __future__ import annotations

import argparse
import csv
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from fig8_sentence_memory import evaluate, read_cbt_sentences, train_sentence
from seqmem.encoding import SSTDDiscreteEncoder
from seqmem.model import MemoryParams, SequentialMemory


def run_trial(job: tuple[str, int, int, int, int, int]) -> dict[str, float | int]:
    data, num_sentences, columns, neurons, data_seed, seed = job
    sentences = read_cbt_sentences(Path(data), num_sentences, data_seed)
    encoder = SSTDDiscreteEncoder(num_columns=columns, k=10, seed=seed)
    model = SequentialMemory(
        encoder=encoder,
        num_neurons_per_column=neurons,
        tie_break_seed=seed,
        params=MemoryParams(
            l_match=3,
            forgetting_threshold=500.0,
        ),
    )
    for sentence in sentences:
        train_sentence(model, sentence)
    distance = evaluate(model, sentences, 6, 0, seed)
    return {
        "columns": columns,
        "neurons": neurons,
        "seed": seed,
        "mean_levenshtein": distance,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(path: Path, rows: list[dict[str, object]]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for neurons in sorted({int(row["neurons"]) for row in rows}):
        selected = sorted(
            (row for row in rows if int(row["neurons"]) == neurons),
            key=lambda row: int(row["columns"]),
            reverse=True,
        )
        x = [int(row["columns"]) for row in selected]
        mean = [float(row["mean_levenshtein"]) for row in selected]
        std = [float(row["std_levenshtein"]) for row in selected]
        axis.errorbar(x, mean, yerr=std, marker="o", capsize=3, label=f"M={neurons}")
    axis.invert_xaxis()
    axis.set(xlabel="Number of mini-columns", ylabel="Mean Levenshtein distance")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/CBTest/data/cbt_train.txt")
    parser.add_argument("--num-sentences", type=int, default=1000)
    parser.add_argument("--columns", nargs="+", type=int, default=[500, 400, 300, 200, 100, 50])
    parser.add_argument("--neurons", nargs="+", type=int, default=[4, 8, 12])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=11,
        help="Fixed CBT sample seed shared by every network-size trial.",
    )
    parser.add_argument("--seed", type=int, default=800)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output-dir", default="results/fig8_resource_sweep")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only the jobs recorded in trials.partial.csv.",
    )
    args = parser.parse_args()

    all_jobs = [
        (
            args.data,
            args.num_sentences,
            columns,
            neurons,
            args.data_seed,
            args.seed + trial,
        )
        for neurons in args.neurons
        for columns in args.columns
        for trial in range(args.trials)
    ]
    output = Path(args.output_dir)
    partial_path = output / "trials.partial.csv"
    trials: list[dict[str, float | int]] = []
    if args.resume and partial_path.exists():
        with partial_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                trials.append(
                    {
                        "columns": int(row["columns"]),
                        "neurons": int(row["neurons"]),
                        "seed": int(row["seed"]),
                        "mean_levenshtein": float(row["mean_levenshtein"]),
                    }
                )
    completed_keys = {
        (int(row["columns"]), int(row["neurons"]), int(row["seed"]))
        for row in trials
    }
    jobs = [
        job
        for job in all_jobs
        if (job[2], job[3], job[5]) not in completed_keys
    ]
    if trials:
        print(f"resuming with {len(trials)}/{len(all_jobs)} completed jobs", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_trial, job) for job in jobs]
        for completed, future in enumerate(
            as_completed(futures), start=len(trials) + 1
        ):
            row = future.result()
            trials.append(row)
            write_csv(
                partial_path,
                sorted(
                    trials,
                    key=lambda item: (
                        int(item["neurons"]),
                        -int(item["columns"]),
                        int(item["seed"]),
                    ),
                ),
            )
            print(f"completed {completed}/{len(all_jobs)}: {row}", flush=True)

    summary: list[dict[str, object]] = []
    for neurons in args.neurons:
        for columns in args.columns:
            selected = [
                row for row in trials
                if row["neurons"] == neurons and row["columns"] == columns
            ]
            distances = [float(row["mean_levenshtein"]) for row in selected]
            summary.append({
                "columns": columns,
                "neurons": neurons,
                "trials": len(selected),
                "mean_levenshtein": statistics.fmean(distances),
                "std_levenshtein": statistics.pstdev(distances),
            })

    write_csv(output / "trials.csv", trials)
    write_csv(output / "summary.csv", summary)
    generated = plot(output / "fig8b.png", summary)
    partial_path.unlink(missing_ok=True)
    print(f"saved: {output}; plot generated: {generated}")


if __name__ == "__main__":
    main()
