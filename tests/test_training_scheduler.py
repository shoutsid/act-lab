from act.model.dense import DenseBaselineConfig, DenseCausalLM
from act.training import (
    TrainingConfig,
    TrainingState,
    make_optimizer,
    make_scheduler,
    train_steps,
)


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
