import torch

from act.importance import (
    cosine_similarity,
    expert_ablation_delta,
    expert_functional_sketches,
    fisher_ema,
    measure_expert_signals,
)
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


def test_fisher_ema_and_functional_sketches_are_deterministic() -> None:
    assert fisher_ema(0.0, 2.0, decay=0.5) == 2.0
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
    hidden = torch.ones(5, 8)
    sketches = expert_functional_sketches(model, hidden)
    assert set(sketches) == {"L0E0", "L0E1", "L0E2"}
    assert all(sketch.shape == (5 * 8,) for sketch in sketches.values())


def test_expert_ablation_restores_parameters() -> None:
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
    before = [parameter.detach().clone() for parameter in model.blocks[0].experts[0].parameters()]
    delta = expert_ablation_delta(model, torch.randint(0, 16, (2, 8)), "L0E0")
    after = list(model.blocks[0].experts[0].parameters())
    assert torch.isfinite(torch.tensor(delta))
    assert all(torch.equal(expected, actual) for expected, actual in zip(before, after))
