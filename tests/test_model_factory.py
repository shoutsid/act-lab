import pytest
import torch

from act.model import DenseCausalLM, StaticMoECausalLM, model_from_genome
from act.topology import create_genome


@pytest.mark.parametrize("ffn, expected", [("dense", DenseCausalLM), ("moe", StaticMoECausalLM)])
def test_factory_builds_model_from_genome(ffn: str, expected: type[torch.nn.Module]) -> None:
    genome = create_genome(4, hidden_size=24, vocab_size=32, depth=2, ffn=ffn)
    model = model_from_genome(genome, context_size=12)
    tokens = torch.randint(0, 32, (2, 12))
    assert isinstance(model, expected)
    assert model(tokens).shape == (2, 12, 32)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"num_heads": 0}, "num_heads must be a positive integer"),
        ({"num_heads": -1}, "num_heads must be a positive integer"),
        ({"num_heads": 1.5}, "num_heads must be a positive integer"),
        ({"context_size": 1}, "context_size must be an integer"),
        ({"context_size": 8.5}, "context_size must be an integer"),
    ],
)
def test_factory_rejects_invalid_context_and_head_arguments(kwargs, message) -> None:
    genome = create_genome(4, hidden_size=24, vocab_size=32, depth=1, ffn="dense")

    with pytest.raises(ValueError, match=message):
        model_from_genome(genome, **kwargs)


def test_factory_rejects_mixed_baseline_layout() -> None:
    genome = create_genome(4, ffn="dense")
    genome.layers[0].ffn = "moe"
    genome.layers[0].experts = create_genome(4, ffn="moe").layers[0].experts
    genome.layers[0].active_experts = 2
    with pytest.raises(ValueError, match="mixed dense/MoE"):
        model_from_genome(genome)