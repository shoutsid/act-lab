"""A tiny deterministic training loop for validating baseline checkpoints."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from act.model.dense import DenseCausalLM
from act.model.moe import StaticMoECausalLM

BaselineModel = DenseCausalLM | StaticMoECausalLM


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 4
    sequence_length: int = 16
    learning_rate: float = 3e-4
    seed: int = 7


@dataclass
class TrainingState:
    step: int = 0
    tokens: int = 0
    generator_state: Tensor | None = None
    dataset_position: int = 0
    genome: dict[str, object] | None = None
    mutation_history: list[dict[str, object]] = field(default_factory=list)
    importance_state: dict[str, object] = field(default_factory=dict)
    scheduler_state: dict[str, object] = field(default_factory=dict)
    experiment_config: dict[str, object] = field(default_factory=dict)
    code_revision: str = "unknown"


def train_steps(
    model: BaselineModel,
    optimizer: torch.optim.Optimizer,
    state: TrainingState,
    config: TrainingConfig,
    count: int,
) -> list[float]:
    """Train for ``count`` synthetic batches and return their losses."""
    if count < 0:
        raise ValueError("count must be non-negative")
    generator = torch.Generator(device="cpu")
    if state.generator_state is None:
        generator.manual_seed(config.seed)
    else:
        generator.set_state(state.generator_state)
    model.train()
    losses: list[float] = []
    for _ in range(count):
        tokens = torch.randint(
            0,
            model.config.vocab_size,
            (config.batch_size, config.sequence_length),
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(tokens)
        loss.backward()
        optimizer.step()
        state.step += 1
        state.tokens += tokens.numel()
        losses.append(float(loss.detach()))
    state.generator_state = generator.get_state()
    return losses


def make_optimizer(model: BaselineModel, config: TrainingConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


def save_checkpoint(
    path: str | Path,
    model: BaselineModel,
    optimizer: torch.optim.Optimizer,
    state: TrainingState,
) -> None:
    torch.save(
        {
            "version": 2,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "training": {
                "step": state.step,
                "tokens": state.tokens,
                "generator_state": state.generator_state,
                "dataset_position": state.dataset_position,
                "genome": state.genome,
                "mutation_history": state.mutation_history,
                "importance_state": state.importance_state,
                "scheduler_state": state.scheduler_state,
                "experiment_config": state.experiment_config,
                "code_revision": state.code_revision,
            },
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: BaselineModel,
    optimizer: torch.optim.Optimizer,
) -> TrainingState:
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("version") not in {1, 2}:
        raise ValueError(f"unsupported checkpoint version {payload.get('version')}")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    training = payload["training"]
    return TrainingState(
        step=int(training["step"]),
        tokens=int(training["tokens"]),
        generator_state=training["generator_state"].clone(),
        dataset_position=int(training.get("dataset_position", training["tokens"])),
        genome=training.get("genome"),
        mutation_history=list(training.get("mutation_history", [])),
        importance_state=dict(training.get("importance_state", {})),
        scheduler_state=dict(training.get("scheduler_state", {})),
        experiment_config=dict(training.get("experiment_config", {})),
        code_revision=str(training.get("code_revision", "unknown")),
    )