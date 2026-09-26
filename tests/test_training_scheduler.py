import pytest
import torch

from act.model.dense import DenseBaselineConfig, DenseCausalLM
from act.training import (
    TrainingConfig,
    TrainingState,
    make_optimizer,
    make_scheduler,
    train_steps,
)

@pytest.mark.parametrize(
    "override, message",
    [
        ({"batch_size": 0}, "batch_size"),
        ({"sequence_length": 1}, "sequence_length"),
        ({"gradient_accumulation_steps": 0}, "gradient_accumulation_steps"),
        ({"warmup_steps": -1}, "scheduler steps"),
        ({"total_steps": -1}, "scheduler steps"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"learning_rate": float("nan")}, "learning_rate"),
        ({"batch_size": 1.5}, "must be integers"),
        ({"mixed_precision": "float8"}, "mixed_precision"),
    ],
)
def test_training_config_rejects_invalid_settings(override, message) -> None:
    with pytest.raises(ValueError, match=message):
        TrainingConfig(**override)


def test_training_config_allows_zero_scheduler_bounds() -> None:
    config = TrainingConfig(warmup_steps=0, total_steps=0)
    assert config.warmup_steps == config.total_steps == 0


def test_gradient_accumulation_counts_optimizer_steps() -> None:
    model = DenseCausalLM(
        DenseBaselineConfig(vocab_size=16, context_size=8, hidden_size=8, num_layers=1)
    )
    config = TrainingConfig(
        batch_size=1,
        sequence_length=8,
        gradient_accumulation_steps=2,
        device="cpu",
    )
    optimizer = make_optimizer(model, config)
    state = TrainingState()
    losses = train_steps(model, optimizer, state, config, 4)
    assert len(losses) == 4
    assert state.step == 2
    assert state.tokens == 32


def test_incomplete_accumulation_group_uses_its_actual_microbatch_count() -> None:
    config_accumulated = TrainingConfig(
        batch_size=1,
        sequence_length=8,
        learning_rate=1e-3,
        gradient_accumulation_steps=4,
        device="cpu",
    )
    config_single = TrainingConfig(
        batch_size=1,
        sequence_length=8,
        learning_rate=1e-3,
        gradient_accumulation_steps=1,
        device="cpu",
    )
    torch.manual_seed(31)
    accumulated = DenseCausalLM(
        DenseBaselineConfig(vocab_size=16, context_size=8, hidden_size=8, num_layers=1)
    )
    single = DenseCausalLM(
        DenseBaselineConfig(vocab_size=16, context_size=8, hidden_size=8, num_layers=1)
    )
    single.load_state_dict(accumulated.state_dict())

    accumulated_state = TrainingState()
    single_state = TrainingState()
    train_steps(
        accumulated,
        make_optimizer(accumulated, config_accumulated),
        accumulated_state,
        config_accumulated,
        1,
    )
    train_steps(
        single,
        make_optimizer(single, config_single),
        single_state,
        config_single,
        1,
    )

    assert accumulated_state.step == single_state.step == 1
    for actual, expected in zip(accumulated.parameters(), single.parameters(), strict=True):
        assert torch.equal(actual, expected)


def test_scheduler_warms_up_and_decays() -> None:
    model = DenseCausalLM(
        DenseBaselineConfig(vocab_size=16, context_size=8, hidden_size=8, num_layers=1)
    )
    config = TrainingConfig(
        batch_size=1,
        sequence_length=8,
        learning_rate=1e-3,
        warmup_steps=2,
        total_steps=4,
        device="cpu",
    )
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)
    state = TrainingState()
    train_steps(model, optimizer, state, config, 4, scheduler)
    assert state.step == 4
    assert state.scheduler_state["learning_rate"] == 0.0
    assert optimizer.param_groups[0]["lr"] == 0.0
