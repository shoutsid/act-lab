from __future__ import annotations

import torch
from torch import Tensor


def validate_token_batch(
    tokens: Tensor,
    *,
    min_sequence_length: int = 1,
    max_sequence_length: int | None = None,
) -> None:
    """Validate token tensor metadata before embedding or loss operations."""
    if not isinstance(tokens, Tensor):
        raise ValueError("tokens must be a tensor")
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [batch, sequence]")
    if tokens.shape[0] < 1:
        raise ValueError("tokens batch dimension must be positive")
    if tokens.shape[1] < min_sequence_length:
        raise ValueError(
            f"tokens sequence length must be at least {min_sequence_length}"
        )
    if max_sequence_length is not None and tokens.shape[1] > max_sequence_length:
        raise ValueError(f"tokens sequence exceeds maximum length {max_sequence_length}")
    if tokens.dtype not in {torch.int32, torch.int64}:
        raise ValueError("tokens must use an integer dtype (int32 or int64)")
