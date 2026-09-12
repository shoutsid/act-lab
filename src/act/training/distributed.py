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
    local_world_size: int = 1
    master_addr: str = "127.0.0.1"
    master_port: int = 29500

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
    if backend not in {"auto", "gloo", "nccl"}:
        raise ValueError("backend must be auto, gloo, or nccl")

    world_size_raw = values.get("WORLD_SIZE")
    world_size = 1 if world_size_raw in (None, "") else int(world_size_raw)
    rank = int(values.get("RANK", "0"))
    local_rank = int(values.get("LOCAL_RANK", "0"))
    local_world_size_raw = values.get("LOCAL_WORLD_SIZE")
    local_world_size = world_size if local_world_size_raw in (None, "") else int(local_world_size_raw)

    if rank < 0 or world_size < 1 or rank >= world_size:
        raise ValueError("invalid distributed rank/world size")
    if local_rank < 0 or local_world_size < 1 or local_rank >= local_world_size:
        raise ValueError("invalid distributed local rank/world size")
    if backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("NCCL requires CUDA")

    master_addr = values.get("MASTER_ADDR", "127.0.0.1")
    master_port_raw = values.get("MASTER_PORT", "29500")
    try:
        master_port = int(master_port_raw)
    except ValueError as exc:  # pragma: no cover - guarded by explicit validation.
        raise ValueError("MASTER_PORT must be an integer") from exc
    if not 0 < master_port <= 65535:
        raise ValueError("MASTER_PORT must be in the range 1..65535")

    return DistributedConfig(
        backend=backend,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        local_world_size=local_world_size,
        master_addr=master_addr,
        master_port=master_port,
    )


def initialize_distributed(config: DistributedConfig) -> None:
    """Initialize a process group only for an actual multi-process launch."""
    if not config.enabled or torch.distributed.is_initialized():
        return
    if config.backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("NCCL requires CUDA")

    backend = config.backend
    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    os.environ.setdefault("MASTER_ADDR", config.master_addr)
    os.environ.setdefault("MASTER_PORT", str(config.master_port))
    if torch.cuda.is_available():
        torch.cuda.set_device(config.local_rank)
    torch.distributed.init_process_group(
        backend=backend,
        rank=config.rank,
        world_size=config.world_size,
    )


def wrap_distributed(model: nn.Module, config: DistributedConfig) -> nn.Module:
    """Wrap a model after initialization, otherwise preserve single-process behavior."""
    if not config.enabled:
        return model
    if not torch.distributed.is_initialized():
        raise RuntimeError("initialize_distributed must run before wrapping a model")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{config.local_rank}")
        model.to(device)
        return DistributedDataParallel(model, device_ids=[config.local_rank], output_device=config.local_rank)
    model.to(torch.device("cpu"))
    return DistributedDataParallel(model)
