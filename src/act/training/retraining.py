"""Explicit initialization policies for topology retraining comparisons."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


InitializationMode = Literal["fresh", "evolved", "rewound", "teacher"]


@dataclass(frozen=True)
class InitializationReport:
    mode: InitializationMode
    copied_parameters: int
    skipped_parameters: int


def initialize_retraining(
    target: nn.Module,
    *,
    mode: InitializationMode,
    source: nn.Module | None = None,
    rewind_state: dict[str, torch.Tensor] | None = None,
) -> InitializationReport:
    """Initialize target from a policy while copying only compatible tensors."""
    if mode == "fresh":
        if source is not None or rewind_state is not None:
            raise ValueError("fresh initialization cannot receive a source")
        return InitializationReport(mode, 0, 0)
    if mode == "rewound":
        if rewind_state is None or source is not None:
            raise ValueError("rewound initialization requires rewind_state only")
        copied, skipped = _copy_state(target, rewind_state)
    elif mode in {"evolved", "teacher"}:
        if source is None or rewind_state is not None:
            raise ValueError(f"{mode} initialization requires source only")
        copied, skipped = _copy_state(target, source.state_dict())
    else:
        raise ValueError(f"unsupported initialization mode: {mode}")
    return InitializationReport(mode, copied, skipped)


def _copy_state(
    target: nn.Module,
    source: dict[str, torch.Tensor],
) -> tuple[int, int]:
    state = target.state_dict()
    copied = 0
    skipped = 0
    for name, value in state.items():
        candidate = source.get(name)
        if candidate is None or candidate.shape != value.shape:
            skipped += 1
            continue
        value.copy_(candidate.to(device=value.device, dtype=value.dtype))
        copied += 1
    target.load_state_dict(state)
    return copied, skipped
