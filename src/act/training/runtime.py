"""A tiny deterministic training loop for validating baseline checkpoints."""

from dataclasses import dataclass, field
from contextlib import nullcontext
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
    device: str = "auto"
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    total_steps: int = 0
    mixed_precision: str = "off"
    activation_checkpointing: bool = False


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
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> list[float]:
    """Train for ``count`` synthetic batches and return their losses."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if config.warmup_steps < 0 or config.total_steps < 0:
        raise ValueError("scheduler steps must be non-negative")
    if config.mixed_precision not in {"off", "fp16", "bf16"}:
        raise ValueError("mixed_precision must be off, fp16, or bf16")
    device = resolve_device(config.device)
    model.to(device)
    if hasattr(model, "activation_checkpointing"):
        model.activation_checkpointing = config.activation_checkpointing
    move_optimizer_state(optimizer, device)
    generator = torch.Generator(device="cpu")
    if state.generator_state is None:
        generator.manual_seed(config.seed)
    else:
        generator.set_state(state.generator_state)
    model.train()
    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    for batch_index in range(count):
        tokens = torch.randint(
            0,
            model.config.vocab_size,
            (config.batch_size, config.sequence_length),
            generator=generator,
        ).to(device)
        with _autocast_context(device, config.mixed_precision):
            loss = model.loss(tokens)
        (loss / config.gradient_accumulation_steps).backward()
        should_step = (
            (batch_index + 1) % config.gradient_accumulation_steps == 0
            or batch_index == count - 1
        )
        if should_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            state.step += 1
            if scheduler is not None:
                scheduler.step()
            state.scheduler_state = {
                "step": state.step,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        state.tokens += tokens.numel()
        losses.append(float(loss.detach()))
    state.generator_state = generator.get_state()
    return losses


def make_optimizer(model: BaselineModel, config: TrainingConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a linear warmup/decay schedule; zero total steps keeps LR constant."""
    if config.warmup_steps < 0 or config.total_steps < 0:
        raise ValueError("scheduler steps must be non-negative")
    total_steps = max(config.total_steps, config.warmup_steps, 1)

    def scale(step: int) -> float:
        if config.warmup_steps and step <= config.warmup_steps:
            return step / config.warmup_steps
        if config.total_steps <= config.warmup_steps:
            return 1.0
        return max(
            0.0,
            (config.total_steps - step) / (config.total_steps - config.warmup_steps),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale, last_epoch=-1)


def _autocast_context(device: torch.device, mode: str):
    if mode == "off":
        return nullcontext()
    if device.type == "cpu" and mode == "fp16":
        raise ValueError("fp16 mixed precision requires CUDA")
    dtype = torch.float16 if mode == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit device or prefer CUDA when the driver supports it."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no compatible device is available")
    return torch.device(requested)


def gpu_inventory() -> dict[str, object]:
    """Every CUDA device the host exposed, not just the one training uses.

    ``resolve_device("auto")`` resolves to plain ``cuda``, i.e. device 0, so a
    second GPU is invisible in a throughput measurement unless the inventory is
    recorded beside it. Kaggle's free T4 offering is exactly that case: the
    ``machine_shape`` string is ``NvidiaTeslaT4`` for both the single-GPU and
    the account's "T4 x2" notebooks, so the artifact - not the kernel metadata
    - has to say how many devices the measured rate came from.

    This deliberately reports only what the host offers: which device training
    actually ran on is the caller's ``device``, and stating it here would let
    the two disagree (``--device cpu`` on a CUDA host).
    """
    if not torch.cuda.is_available():
        return {"count": 0, "names": []}
    count = torch.cuda.device_count()
    return {
        "count": count,
        "names": [torch.cuda.get_device_name(index) for index in range(count)],
    }


def move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """Move tensor state after loading a CPU checkpoint into a device model."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def save_checkpoint(
    path: str | Path,
    model: BaselineModel,
    optimizer: torch.optim.Optimizer,
    state: TrainingState,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 3,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
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
        target,
    )


def load_checkpoint(
    path: str | Path,
    model: BaselineModel,
    optimizer: torch.optim.Optimizer,
) -> TrainingState:
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint_payload(payload)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    move_optimizer_state(optimizer, next(model.parameters()).device)
    if "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"])
    training = payload["training"]
    generator_state = training.get("generator_state")
    return TrainingState(
        step=int(training["step"]),
        tokens=int(training["tokens"]),
        generator_state=generator_state.clone() if generator_state is not None else None,
        dataset_position=int(training.get("dataset_position", training["tokens"])),
        genome=training.get("genome"),
        mutation_history=list(training.get("mutation_history", [])),
        importance_state=dict(training.get("importance_state", {})),
        scheduler_state=dict(training.get("scheduler_state", {})),
        experiment_config=dict(training.get("experiment_config", {})),
        code_revision=str(training.get("code_revision", "unknown")),
    )


def validate_checkpoint_payload(payload: object) -> None:
    """Reject malformed or unsupported checkpoint structures before loading state."""
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    version = payload.get("version")
    if version not in {1, 2, 3, 4, 5}:
        raise ValueError(f"unsupported checkpoint version {version}")
    for key in ("model", "optimizer", "training"):
        if key not in payload:
            raise ValueError(f"checkpoint is missing {key}")
    if not isinstance(payload["model"], dict):
        raise ValueError("checkpoint model must be a state mapping")
    if not isinstance(payload["optimizer"], dict):
        raise ValueError("checkpoint optimizer must be a state mapping")
    if not isinstance(payload["training"], dict):
        raise ValueError("checkpoint training must be a mapping")
    training = payload["training"]
    for key in ("step", "tokens"):
        if key not in training:
            raise ValueError(f"checkpoint training is missing {key}")