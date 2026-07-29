"""CLI-compatible entry point for exact Fig.9 observation reasons."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.diagnostics.analyze_fig9_observe_scenario_assignment import (
    analyze,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    trace = run_dir / "observe_scenario_trace.csv"
    if not trace.exists():
        trace = run_dir / "teacher_forced_observation_trace.csv"
    analyze(
        observation_trace=trace,
        output_dir=Path(args.output_dir),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()
