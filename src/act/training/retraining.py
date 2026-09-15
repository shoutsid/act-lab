"""Explicit initialization policies for topology retraining comparisons."""

from collections.abc import Mapping
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
    renames: Mapping[str, str | None] | None = None,
) -> InitializationReport:
    """Initialize target from a policy while copying only compatible tensors.

    ``renames`` maps a *target* parameter name to the source name that should
    fill it, or to ``None`` to leave the target's current (freshly drawn) value
    untouched. It exists because name-and-shape matching is not identity
    matching: plan section 11 requires *final topology + original
    initialization* "without ambiguity", and in a topology that grew or pruned
    experts the same tensor name can refer to a different component in the
    source (``blocks.0.experts.1.*`` is a different expert once expert 1 was
    pruned). Without a rename map that copy is silently wrong - it hands a
    survivor another component's weights - so callers rewinding across a
    topology change must say how components correspond.
    """
    if mode == "fresh":
        if source is not None or rewind_state is not None or renames is not None:
            raise ValueError("fresh initialization cannot receive a source")
        return InitializationReport(mode, 0, 0)
    if mode == "rewound":
        if rewind_state is None or source is not None:
            raise ValueError("rewound initialization requires rewind_state only")
        copied, skipped = _copy_state(target, rewind_state, renames)
    elif mode in {"evolved", "teacher"}:
        if source is None or rewind_state is not None:
            raise ValueError(f"{mode} initialization requires source only")
        copied, skipped = _copy_state(target, source.state_dict(), renames)
    else:
        raise ValueError(f"unsupported initialization mode: {mode}")
    return InitializationReport(mode, copied, skipped)


def _copy_state(
    target: nn.Module,
    source: dict[str, torch.Tensor],
    renames: Mapping[str, str | None] | None = None,
) -> tuple[int, int]:
    state = target.state_dict()
    copied = 0
    skipped = 0
    for name, value in state.items():
        key: str | None = name
        if renames is not None and name in renames:
            key = renames[name]
        candidate = None if key is None else source.get(key)
        if candidate is None or candidate.shape != value.shape:
            skipped += 1
            continue
        value.copy_(candidate.to(device=value.device, dtype=value.dtype))
        copied += 1
    target.load_state_dict(state)
    return copied, skipped
