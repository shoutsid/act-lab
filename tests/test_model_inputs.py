import pytest
import torch

from act.model import (
    DenseBaselineConfig,
    DenseCausalLM,
    StaticMoECausalLM,
    StaticMoEConfig,
    V0CausalLM,
    V0ModelConfig,
    V0MoECausalLM,
    V0MoEConfig,
)


def _models():
    return [
        DenseCausalLM(
            DenseBaselineConfig(
                vocab_size=16,
                context_size=8,
                hidden_size=8,
                num_layers=1,
                num_heads=2,
                ffn_size=16,
            )
        ),
        StaticMoECausalLM(
            StaticMoEConfig(
                vocab_size=16,
                context_size=8,
                hidden_size=8,
                num_layers=1,
                num_heads=2,
                ffn_size=16,
                num_experts=2,
                active_experts=1,
            )
        ),
        V0CausalLM(
            V0ModelConfig(
                vocab_size=16,
                context_size=8,
                hidden_size=8,
                depth=1,
                attention_heads=2,
                ffn_width=16,
            )
        ),
        V0MoECausalLM(
            V0MoEConfig(
                vocab_size=16,
                context_size=8,
                hidden_size=8,
                depth=1,
                attention_heads=2,
                experts_per_layer=(2,),
                active_experts=2,
                expert_width=4,
            )
        ),
    ]


@pytest.mark.parametrize("model", _models(), ids=lambda model: type(model).__name__)
def test_model_forward_rejects_invalid_token_batch_metadata(model) -> None:
    with pytest.raises(ValueError, match="batch dimension"):
        model(torch.empty(0, 4, dtype=torch.long))
    with pytest.raises(ValueError, match="sequence length"):
        model(torch.empty(1, 0, dtype=torch.long))
    with pytest.raises(ValueError, match="integer dtype"):
        model(torch.zeros(1, 4, dtype=torch.float32))
    with pytest.raises(ValueError, match="exceeds maximum length"):
        model(torch.zeros(1, 9, dtype=torch.long))


@pytest.mark.parametrize("model", _models(), ids=lambda model: type(model).__name__)
def test_model_loss_rejects_invalid_token_sequence_before_cross_entropy(model) -> None:
    with pytest.raises(ValueError, match="sequence length must be at least 2"):
        model.loss(torch.zeros(1, 1, dtype=torch.long))
    with pytest.raises(ValueError, match="exceeds maximum length"):
        model.loss(torch.zeros(1, 10, dtype=torch.long))
