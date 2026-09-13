"""Command-line entry points for baseline and declared V0 protocol runs."""

import argparse
import json
from pathlib import Path

import torch

from act.model import DenseBaselineConfig, DenseCausalLM
from act.training.runtime import (
    TrainingConfig,
    TrainingState,
    make_optimizer,
    save_checkpoint,
    train_steps,
)
from act.training.v0_protocol import aggregate_arm_cost_ratio, harm_guard_rejects, sign_pairs
from act.training.v0_session import V0SessionConfig, run_v0_session


def _run_dense_baseline(args: argparse.Namespace) -> None:
    torch.manual_seed(19)
    model = DenseCausalLM(DenseBaselineConfig())
    config = TrainingConfig(device=args.device)
    state = TrainingState()
    optimizer = make_optimizer(model, config)
    losses = train_steps(model, optimizer, state, config, args.steps)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.checkpoint, model, optimizer, state)
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "parameters": model.parameter_count,
                "step": state.step,
                "tokens": state.tokens,
                "loss": losses[-1],
                "device": str(next(model.parameters()).device),
            },
            sort_keys=True,
        )
    )


def _run_v0(args: argparse.Namespace) -> None:
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    config = V0SessionConfig(
        arms=("random", "adaptive"),
        probe_batch_count=args.probe_batch_count,
        device=args.device,
        total_steps=args.steps,
    )
    results = []
    metrics = []
    for seed in seeds:
        seed_config = V0SessionConfig(
            arms=("random", "adaptive"),
            seed=seed,
            probe_batch_count=args.probe_batch_count,
            device=args.device,
            total_steps=args.steps,
        )
        result = run_v0_session(seed_config, checkpoint_dir=args.checkpoint_dir)
        results.append(result)
        arms = {arm.mode: arm for arm in result.arms}
        metrics.append(
            {
                "seed": seed,
                "adaptive_loss": arms["adaptive"].final_loss,
                "random_loss": arms["random"].final_loss,
                "search_flops_by_arm": {
                    arm.mode: arm.search_flops for arm in result.arms
                },
                "training_flops_by_arm": {
                    arm.mode: arm.training_flops for arm in result.arms
                },
                "wall_time_seconds": {
                    arm.mode: arm.wall_time_seconds for arm in result.arms
                },
                "tokens_per_second": {
                    arm.mode: arm.tokens_per_second for arm in result.arms
                },
            }
        )
    cost_ratio = aggregate_arm_cost_ratio(results)
    pairs = sign_pairs(results)
    summary = {
        "protocol": "v0-compute-efficiency-5.1",
        "probe_batch_count": args.probe_batch_count,
        "seeds": seeds,
        "predicted_cost_ratio": results[0].predicted_cost_ratio,
        "measured_cost_ratio": cost_ratio,
        "measured_cost_percent": 100.0 * cost_ratio,
        "adaptive_wins": sum(t <= r for t, r in pairs),
        "harm_guard_rejects": harm_guard_rejects(pairs),
        "per_seed": metrics,
    }
    output = args.output or (Path(args.checkpoint_dir) / "summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "per_seed"}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "v0"), default="baseline")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="training horizon (baseline default 8; v0 default 6000 = declared)",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/dense-baseline.pt"))
    # V0 protocol options.
    parser.add_argument("--seeds", default="9000")
    parser.add_argument("--probe-batch-count", type=int, default=1)
    parser.add_argument("--checkpoint-dir", default="artifacts/v0-sessions")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.mode == "v0":
        if args.steps is None:
            args.steps = 6_000
        _run_v0(args)
    else:
        if args.steps is None:
            args.steps = 8
        _run_dense_baseline(args)


if __name__ == "__main__":
    main()
