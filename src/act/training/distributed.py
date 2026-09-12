"""Small, explicit helpers for single-process and torchrun execution."""

from dataclasses import dataclass
import os

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedConfig:
    backend: str = "auto"
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1


def distributed_config(
    *,
    backend: str = "auto",
    environ: dict[str, str] | None = None,
) -> DistributedConfig:
    """Read torchrun variables without mutating process state."""
    values = os.environ if environ is None else environ
    rank = int(values.get("RANK", "0"))
    world_size = int(values.get("WORLD_SIZE", "1"))
    local_rank = int(values.get("LOCAL_RANK", "0"))
    if rank < 0 or world_size < 1 or rank >= world_size:
        raise ValueError("invalid distributed rank/world size")
    if backend not in {"auto", "gloo", "nccl"}:
        raise ValueError("backend must be auto, gloo, or nccl")
    if backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("NCCL requires CUDA")
    return DistributedConfig(backend, rank, world_size, local_rank)


def initialize_distributed(config: DistributedConfig) -> None:
    """Initialize a process group only for an actual multi-process launch."""
    if not config.enabled or torch.distributed.is_initialized():
        return
    backend = "nccl" if config.backend == "auto" and torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend, rank=config.rank, world_size=config.world_size)


def wrap_distributed(model: nn.Module, config: DistributedConfig) -> nn.Module:
    """Wrap a model after initialization, otherwise preserve single-process behavior."""
    if not config.enabled:
        return model
    if not torch.distributed.is_initialized():
        raise RuntimeError("initialize_distributed must run before wrapping a model")
    device_ids = [config.local_rank] if torch.cuda.is_available() else None
    return DistributedDataParallel(model, device_ids=device_ids)
