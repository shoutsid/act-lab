"""Run and persist a low-fidelity ACT Experiment 0."""

import argparse
import json
from pathlib import Path

from .experiment import ExperimentConfig, run_experiment
from .registry import append_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments.jsonl"))
    args = parser.parse_args()
    result = run_experiment(ExperimentConfig(seed=args.seed, steps=args.steps))
    print(json.dumps(append_result(args.output, result), sort_keys=True))


if __name__ == "__main__":
    main()