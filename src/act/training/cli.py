"""Command-line entry point for a small deterministic dense baseline run."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/dense-baseline.pt"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

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


if __name__ == "__main__":
    main()