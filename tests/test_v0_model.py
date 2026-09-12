import torch

from act.model import V0CausalLM, V0ModelConfig


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
