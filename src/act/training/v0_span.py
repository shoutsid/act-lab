"""Random-window sampling from flat uint16 token streams (real-data sessions).

The section 5.1 session machinery draws uniform-random token tensors from
seeded CPU generators. When a real corpus stream is supplied
(``V0SessionConfig.data_path``), the same generators instead pick random
window *positions* into the stream: one seeded, checkpointed generator draw
per batch, exactly as the synthetic path draws one token tensor per batch.
Batch shapes, generator state persistence, and FLOPs booking are unchanged,
so the bit-identical resume contract holds within each path and the FLOPs
cost identity applies unchanged. Tokens are read with
``np.load(..., mmap_mode="r")`` so a 40M-token TinyStories stream costs
~80 MB of RAM regardless of training length.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_stream(path: str | Path) -> np.ndarray:
    """Memory-map a flat uint16 token stream produced by train_real_lm.py."""
    stream = np.load(path, mmap_mode="r")
    if stream.dtype != np.uint16:
        raise ValueError(f"token stream must be uint16 (got {stream.dtype}): {path}")
    if stream.ndim != 1:
        raise ValueError(f"token stream must be flat (got shape {stream.shape}): {path}")
    if stream.shape[0] < 2:
        raise ValueError(f"token stream too small: {path}")
    return stream


def sample_windows(
    stream: np.ndarray,
    generator: torch.Generator,
    count: int,
    width: int,
) -> torch.Tensor:
    """Draw ``count`` random windows of ``width`` tokens from a uint16 stream.

    ``generator`` is the caller's CPU ``torch.Generator``: one seeded draw
    per batch, checkpointed by the session exactly as the synthetic path
    checkpoints its token-tensor draws, so a resumed session continues its
    own stream identically. Windows may span story boundaries (token 0),
    matching the real-LM training script's window semantics.
    """
    high = max(1, stream.shape[0] - width)
    starts = torch.randint(0, high, (count,), generator=generator).tolist()
    windows = np.stack([stream[s : s + width] for s in starts]).astype(np.int64)
    return torch.from_numpy(windows)


class WindowSampler:
    """Per-stream window source bound to the session's vocab contract.

    Validates that the stream's token ids fit inside the model's vocabulary
    (the real-LM stream was produced with a vocab-8192 tokenizer and id 0
    reserved as the story boundary) so a mismatch fails fast instead of
    crashing inside embedding lookup.
    """

    def __init__(self, path: str | Path, vocab_size: int) -> None:
        self.path = Path(path)
        self.stream = load_stream(self.path)
        self.vocab_size = int(vocab_size)
        if int(self.stream.max()) >= self.vocab_size:
            raise ValueError(
                f"token stream {self.path} contains ids >= vocab_size "
                f"{self.vocab_size}; rebuild the stream for this model"
            )

    def draw(self, generator: torch.Generator, count: int, width: int) -> torch.Tensor:
        return sample_windows(self.stream, generator, count, width)
