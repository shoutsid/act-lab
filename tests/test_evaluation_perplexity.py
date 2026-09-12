import math

import pytest
import torch

from act.evaluation import evaluate_perplexity
from act.model.dense import DenseBaselineConfig, DenseCausalLM


def test_perplexity_reports_weighted_loss_and_token_count() -> None:
    model = DenseCausalLM(
        DenseBaselineConfig(vocab_size=16, context_size=8, hidden_size=8, num_layers=1)
    )
    batches = [torch.randint(0, 16, (2, 8)), torch.randint(0, 16, (1, 8))]
    result = evaluate_perplexity(model, batches)
    assert result.batches == 2
    assert result.tokens == 21
    assert math.isfinite(result.mean_loss)
    assert math.isfinite(result.perplexity)
    assert result.perplexity == pytest.approx(math.exp(result.mean_loss))


def test_perplexity_rejects_empty_or_short_batches() -> None:
    model = DenseCausalLM(DenseBaselineConfig(hidden_size=8, num_heads=2))
    with pytest.raises(ValueError, match="at least one"):
        evaluate_perplexity(model, [])
    with pytest.raises(ValueError, match="sequence"):
        evaluate_perplexity(model, [torch.ones(1, 1, dtype=torch.long)])
