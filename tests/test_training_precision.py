import pytest

from act.model.dense import DenseBaselineConfig, DenseCausalLM
from act.training import TrainingConfig, TrainingState, make_optimizer, train_steps


def test_cpu_bfloat16_training_is_explicit_and_finite() -> None:
    model = DenseCausalLM(
        DenseBaselineConfig(vocab_size=16, context_size=8, hidden_size=8, num_layers=1)
    )
    config = TrainingConfig(
        batch_size=1,
        sequence_length=8,
        mixed_precision="bf16",
        device="cpu",
    )
    losses = train_steps(model, make_optimizer(model, config), TrainingState(), config, 1)
    assert len(losses) == 1


def test_cpu_fp16_precision_is_rejected() -> None:
    model = DenseCausalLM(
        DenseBaselineConfig(vocab_size=16, context_size=8, hidden_size=8, num_layers=1)
    )
    config = TrainingConfig(mixed_precision="fp16", device="cpu")
    with pytest.raises(ValueError, match="requires CUDA"):
        train_steps(model, make_optimizer(model, config), TrainingState(), config, 1)
