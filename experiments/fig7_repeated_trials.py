from __future__ import annotations

import argparse
import csv
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from fig7_sequence_prediction import execute


def trial_args(args: argparse.Namespace, task: str, seed: int) -> argparse.Namespace:
    return argparse.Namespace(
        seed=seed,
        task=task,
        max_elements=args.max_elements,
        noise_symbols=50000,
        modify_after_elements=(args.modify_after_elements if task == "single" else -1),
        forgetting_threshold=(25.0 if task == "single" else 50.0),
        recovery_accuracy=args.recovery_accuracy,
        report_every=args.report_every,
        output_csv="",
    )


def aggregate_task(
    args: argparse.Namespace,
    task: str,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str]]:
    results = []
    jobs = [
        (args, task, args.seed + trial, trial + 1)
        for trial in range(args.trials)
    ]
    if args.workers == 1:
        for job in jobs:
            results.append(run_trial(job))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_trial, job): job for job in jobs}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"completed task={task}, {len(results)}/{args.trials}",
                    flush=True,
                )

    by_element: dict[int, list[float]] = {}
    for result in results:
        for element, moving_accuracy in result.checkpoints:
            by_element.setdefault(element, []).append(moving_accuracy)

    rows: list[dict[str, float | int | str]] = []
    for element in sorted(by_element):
        values = by_element[element]
        rows.append(
            {
                "task": task,
                "elements": element,
                "mean_moving_accuracy": statistics.fmean(values),
                "std_moving_accuracy": statistics.pstdev(values),
                "completed_trials": len(values),
            }
        )

    final_values = [result.final_moving_accuracy for result in results]
    recovery_values = [
        result.recovery_elements
        for result in results
        if result.recovery_elements is not None
    ]
    summary: dict[str, float | int | str] = {
        "task": task,
        "trials": args.trials,
        "mean_final_moving_accuracy": statistics.fmean(final_values),
        "std_final_moving_accuracy": statistics.pstdev(final_values),
        "recovered_trials": len(recovery_values),
        "mean_recovery_elements": (
            statistics.fmean(recovery_values) if recovery_values else -1.0
        ),
        "std_recovery_elements": (
            statistics.pstdev(recovery_values) if recovery_values else -1.0
        ),
    }
    return rows, summary


def run_trial(job: tuple[argparse.Namespace, str, int, int]):
    args, task, seed, trial = job
    print(
        f"running task={task}, trial={trial}/{args.trials}, seed={seed}",
        flush=True,
    )
    return execute(trial_args(args, task, seed))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(path: Path, rows: list[dict[str, object]], modification: int) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, axis = plt.subplots(figsize=(9, 5.2))
    for task in sorted({str(row["task"]) for row in rows}):
        selected = [row for row in rows if row["task"] == task]
        x = [int(row["elements"]) for row in selected]
        mean = [float(row["mean_moving_accuracy"]) for row in selected]
        std = [float(row["std_moving_accuracy"]) for row in selected]
        axis.plot(x, mean, label=task)
        axis.fill_between(
            x,
            [max(0.0, value - spread) for value, spread in zip(mean, std)],
            [min(1.0, value + spread) for value, spread in zip(mean, std)],
            alpha=0.18,
        )
    if modification >= 0:
        axis.axvline(modification, color="tab:orange", linestyle="--", label="modification")
    axis.set(xlabel="Processed elements", ylabel="Moving accuracy (last 100)", ylim=(0, 1.02))
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--tasks", nargs="+", choices=["single", "multiple-2", "multiple-4"], default=["single", "multiple-2", "multiple-4"])
    parser.add_argument("--max-elements", type=int, default=0)
    parser.add_argument("--modify-after-elements", type=int, default=10000)
    parser.add_argument("--recovery-accuracy", type=float, default=1.0)
    parser.add_argument("--report-every", type=int, default=250)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", default="results/fig7_strict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for task in args.tasks:
        task_rows, summary = aggregate_task(args, task)
        rows.extend(task_rows)
        summaries.append(summary)
        print(summary, flush=True)

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "curves.csv", rows)
    write_csv(output_dir / "summary.csv", summaries)
    plotted = plot_curves(
        output_dir / "curves.png", rows, args.modify_after_elements
    )
    print(f"saved: {output_dir}")
    print(f"plot generated: {plotted}")


if __name__ == "__main__":
    main()
