from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from torch import nn


@contextmanager
def evaluation_mode(model: nn.Module) -> Iterator[None]:
    """Temporarily evaluate a model, then restore every module's prior training flag."""
    training_flags = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        # Restore child flags directly: calling train() recursively on the root would erase a
        # caller's intentionally mixed train/eval configuration among submodules.
        for module, training in training_flags:
            module.training = training
