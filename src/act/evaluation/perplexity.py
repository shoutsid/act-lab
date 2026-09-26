"""Language-model perplexity evaluation over bounded token batches."""

from dataclasses import dataclass
import math
import sys

import torch

from act.model.dense import DenseCausalLM
from act.model.moe import StaticMoECausalLM
from act.model.v0_moe import V0MoECausalLM
from ._mode import evaluation_mode

#: Any model exposing ``forward(tokens) -> logits`` and ``loss(tokens) -> scalar``.
#: The architecture-search path builds ``V0MoECausalLM``, so it belongs here: a
#: finalist that could not be probed would silently skip section 19's evaluation.
LanguageModel = DenseCausalLM | StaticMoECausalLM | V0MoECausalLM


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
    total_loss = 0.0
    total_tokens = 0
    with evaluation_mode(model), torch.no_grad():
        for tokens in batches:
            if tokens.ndim != 2 or tokens.shape[1] < 2:
                raise ValueError("evaluation batches must have shape [batch, sequence>=2]")
            loss = model.loss(tokens)
            if not torch.isfinite(loss).all():
                raise ValueError("evaluation produced a non-finite loss")
            count = tokens.shape[0] * (tokens.shape[1] - 1)
            total_loss += float(loss) * count
            total_tokens += count
    mean_loss = total_loss / total_tokens
    if not math.isfinite(mean_loss):
        raise ValueError("evaluation produced a non-finite mean loss")
    perplexity = math.exp(mean_loss) if mean_loss < math.log(sys.float_info.max) else float("inf")
    return PerplexityResult(mean_loss, perplexity, len(batches), total_tokens)
