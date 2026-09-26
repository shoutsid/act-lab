import pytest
import torch

from act.model import V0CausalLM, V0ModelConfig
from act.model.v0_moe import V0MoEConfig


def test_v0_hybrid_model_produces_finite_logits_and_loss() -> None:
    model = V0CausalLM(
        V0ModelConfig(
            vocab_size=32,
            context_size=12,
            hidden_size=24,
            depth=3,
            attention_heads=4,
            ffn_width=48,
            module_types=("attention", "gdn", "hybrid"),
        )
    )
    tokens = torch.randint(0, 32, (2, 12))
    logits = model(tokens[:, :-1])
    loss = model.loss(tokens)
    assert logits.shape == (2, 11, 32)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    assert model.parameter_count == model.active_parameter_count


@pytest.mark.parametrize(
    "override",
    [
        {"attention_heads": 0},
        {"hidden_size": 0},
        {"context_size": 0},
        {"vocab_size": 0},
        {"depth": 0, "experts_per_layer": ()},
        {"expert_width": 0},
        {"rotary_base": float("inf")},
        {"rotary_base": 0.0},
        {"norm_epsilon": float("nan")},
        {"norm_epsilon": 0.0},
    ],
)
def test_v0_moe_config_rejects_nonpositive_or_nonfinite_dimensions(override: dict) -> None:
    with pytest.raises(ValueError):
        V0MoEConfig(**override)


def test_v0_rejects_invalid_module_schedule() -> None:
    try:
        V0CausalLM(
            V0ModelConfig(
                hidden_size=24,
                attention_heads=4,
                depth=3,
                module_types=("attention", "gdn"),
            )
        )
    except ValueError as error:
        assert "module_types" in str(error)
    else:
        raise AssertionError("invalid module schedule should fail")
