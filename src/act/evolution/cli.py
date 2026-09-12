"""Run and persist a low-fidelity ACT Experiment 0."""

import argparse
import json
from pathlib import Path

from .experiment import ExperimentConfig, run_experiment
from .registry import append_result, append_study
from .study import run_gate_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments.jsonl"))
    parser.add_argument(
        "--seeds",
        help="comma-separated seeds for a gate study instead of one experiment",
    )
    args = parser.parse_args()
    config = ExperimentConfig(seed=args.seed, steps=args.steps)
    if args.seeds:
        seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
        study = run_gate_study(config, seeds=seeds)
        print(json.dumps(append_study(args.output, study), sort_keys=True))
        return
    result = run_experiment(config)
    print(json.dumps(append_result(args.output, result), sort_keys=True))


if __name__ == "__main__":
    main()