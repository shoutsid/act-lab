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
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments.jsonl"))
    parser.add_argument(
        "--arms",
        default="fixed,random,adaptive",
        help="comma-separated arm selection: fixed, random, adaptive, guided",
    )
    parser.add_argument(
        "--adaptive-policy",
        choices=("probe_loss", "signal"),
        default="probe_loss",
        help="adaptive arm decision rule",
    )
    parser.add_argument(
        "--probe-candidate-cap",
        type=int,
        default=4,
        help="max mutation candidates probed per adaptive mutation",
    )
    parser.add_argument(
        "--probe-batch-count",
        type=int,
        default=2,
        help="probe batches used per candidate evaluation",
    )
    parser.add_argument(
        "--seeds",
        help="comma-separated seeds for a gate study instead of one experiment",
    )
    parser.add_argument(
        "--treatment",
        choices=("adaptive", "guided"),
        default="adaptive",
        help="arm scored against random in a gate study",
    )
    args = parser.parse_args()
    config = ExperimentConfig(
        seed=args.seed,
        steps=args.steps,
        device=args.device,
        arms=args.arms,
        adaptive_policy=args.adaptive_policy,
        probe_candidate_cap=args.probe_candidate_cap,
        probe_batch_count=args.probe_batch_count,
    )
    if args.seeds:
        seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
        study = run_gate_study(config, seeds=seeds, treatment=args.treatment)
        print(json.dumps(append_study(args.output, study), sort_keys=True))
        return
    result = run_experiment(config)
    print(json.dumps(append_result(args.output, result), sort_keys=True))


if __name__ == "__main__":
    main()
