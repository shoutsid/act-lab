"""Language-model perplexity evaluation over bounded token batches."""

from dataclasses import dataclass
import math
import sys

import torch

from act.model.dense import DenseCausalLM
from act.model.moe import StaticMoECausalLM

LanguageModel = DenseCausalLM | StaticMoECausalLM


@dataclass(frozen=True)
class PerplexityResult:
    mean_loss: float
    perplexity: float
    batches: int
    tokens: int


def evaluate_perplexity(
    model: LanguageModel,
    batches: tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> PerplexityResult:
    if not batches:
        raise ValueError("at least one evaluation batch is required")
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for tokens in batches:
            if tokens.ndim != 2 or tokens.shape[1] < 2:
                raise ValueError("evaluation batches must have shape [batch, sequence>=2]")
            loss = model.loss(tokens)
            count = (tokens.shape[0] * (tokens.shape[1] - 1))
            total_loss += float(loss) * count
            total_tokens += count
    mean_loss = total_loss / total_tokens
    perplexity = math.exp(mean_loss) if mean_loss < math.log(sys.float_info.max) else float("inf")
    return PerplexityResult(mean_loss, perplexity, len(batches), total_tokens)
