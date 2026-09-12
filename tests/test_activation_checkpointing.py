import torch

from act.model.dense import DenseBaselineConfig, DenseCausalLM
from act.training import TrainingConfig, TrainingState, make_optimizer, train_steps


def test_activation_checkpointing_preserves_finite_training() -> None:
    torch.manual_seed(3)
    model = DenseCausalLM(
        DenseBaselineConfig(vocab_size=16, context_size=8, hidden_size=8, num_layers=2)
    )
    config = TrainingConfig(
        batch_size=1,
        sequence_length=8,
        activation_checkpointing=True,
        device="cpu",
    )
    state = TrainingState()
    losses = train_steps(model, make_optimizer(model, config), state, config, 2)
    assert len(losses) == 2
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
