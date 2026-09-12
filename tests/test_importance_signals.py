import torch

from act.importance import cosine_similarity, measure_expert_signals
from act.model import StaticMoECausalLM, StaticMoEConfig


def test_cosine_similarity_is_bounded_and_shape_safe() -> None:
    assert cosine_similarity(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])) == 1.0
    assert cosine_similarity(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])) == -1.0


def test_expert_signals_are_finite_after_a_training_step() -> None:
    torch.manual_seed(4)
    model = StaticMoECausalLM(
        StaticMoEConfig(
            vocab_size=16,
            context_size=8,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            ffn_size=16,
            num_experts=3,
            active_experts=1,
        )
    )
    tokens = torch.randint(0, 16, (2, 8))
    model.loss(tokens).backward()
    signals = measure_expert_signals(model)
    assert len(signals) == 3
    assert all(signal.temporal_stability == 1.0 for signal in signals)
    assert all(torch.isfinite(torch.tensor(signal.gradient_norm)) for signal in signals)
