"""Bounded V0 construction and numerical smoke probes."""

from dataclasses import dataclass
import math

import torch

from .factory import v0_model_from_genome
from act.topology import ArchitectureGenome


@dataclass(frozen=True)
class V0SmokeResult:
    parameter_count: int
    loss: float
    logits_shape: tuple[int, ...]


def run_v0_smoke(
    genome: ArchitectureGenome,
    *,
    batch_size: int = 1,
    sequence_length: int = 8,
) -> V0SmokeResult:
    if batch_size < 1 or sequence_length < 2:
        raise ValueError("batch_size must be positive and sequence_length must be at least 2")
    model = v0_model_from_genome(genome)
    tokens = torch.randint(0, genome.vocab_size, (batch_size, sequence_length))
    with torch.no_grad():
        logits = model(tokens[:, :-1])
        loss = float(model.loss(tokens))
    if not math.isfinite(loss) or not torch.isfinite(logits).all():
        raise RuntimeError("V0 smoke produced non-finite values")
    return V0SmokeResult(model.parameter_count, loss, tuple(logits.shape))
