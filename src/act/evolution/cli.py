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
        "--mutation-interval",
        type=int,
        default=12,
        help="steps between structural mutation events",
    )
    parser.add_argument(
        "--mutation-budget",
        type=int,
        default=3,
        help="maximum mutations per arm",
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
        "--probe-candidate-policy",
        choices=("prefix", "spread"),
        default="prefix",
        help="how a cap-bounded probe picks candidates: 'prefix' takes the "
        "first N of the layer-major enumeration (historical), 'spread' draws "
        "a deterministic subset from the whole feasible set",
    )
    parser.add_argument(
        "--independent-arm-rng",
        action="store_true",
        help="seed rebuild init per (arm, event) so the random baseline is the "
        "same control under any treatment (substrate audit, report section "
        "9.8); changes numbers relative to the published shared-RNG studies",
    )
    parser.add_argument(
        "--probe-selection",
        choices=("argmin", "argmax", "random"),
        default="argmin",
        help="decision rule over probed candidates: 'argmin' (registered), "
        "'argmax' (inverted control), 'random' (pool-matched random control)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="number of MoE layers in the toy genome (historical studies used 1)",
    )
    parser.add_argument(
        "--expert-count",
        type=int,
        default=4,
        help="experts per layer in the toy genome (historical studies used 4)",
    )
    parser.add_argument(
        "--probe-landscape",
        action="store_true",
        help="record every candidate's probe loss in mutation history",
    )
    parser.add_argument(
        "--probe-force-candidate",
        type=int,
        default=None,
        help="force applying the k-th best candidate at every mutation event",
    )
    parser.add_argument(
        "--probe-force-events",
        type=int,
        default=None,
        help="limit forced selection to the first N mutation events",
    )
    parser.add_argument(
        "--seeds",
        help="comma-separated seeds for a gate study instead of one experiment",
    )
    parser.add_argument(
        "--treatment",
        choices=("adaptive", "guided", "null"),
        default="adaptive",
        help="arm scored against random in a gate study",
    )
    args = parser.parse_args()
    config = ExperimentConfig(
        seed=args.seed,
        steps=args.steps,
        device=args.device,
        arms=args.arms,
        mutation_interval=args.mutation_interval,
        mutation_budget=args.mutation_budget,
        adaptive_policy=args.adaptive_policy,
        probe_candidate_cap=args.probe_candidate_cap,
        probe_batch_count=args.probe_batch_count,
        probe_candidate_policy=args.probe_candidate_policy,
        probe_selection=args.probe_selection,
        independent_arm_rng=args.independent_arm_rng,
        depth=args.depth,
        expert_count=args.expert_count,
        probe_landscape=args.probe_landscape,
        probe_force_candidate=args.probe_force_candidate,
        probe_force_events=args.probe_force_events,
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
