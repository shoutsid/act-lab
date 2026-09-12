from pathlib import Path

import torch

from act.model import DenseBaselineConfig, DenseCausalLM, StaticMoECausalLM, StaticMoEConfig
from act.training import (
    TrainingConfig,
    TrainingState,
    load_checkpoint,
    make_optimizer,
    save_checkpoint,
    train_steps,
)


def _build() -> tuple[DenseCausalLM, torch.optim.Optimizer, TrainingState]:
    torch.manual_seed(19)
    model = DenseCausalLM(
        DenseBaselineConfig(
            vocab_size=32,
            context_size=12,
            hidden_size=24,
            num_layers=1,
            num_heads=4,
            ffn_size=48,
        )
    )
    config = TrainingConfig(batch_size=2, sequence_length=12, seed=31)
    return model, make_optimizer(model, config), TrainingState()


def test_checkpoint_resume_matches_continuous_training(tmp_path: Path) -> None:
    config = TrainingConfig(batch_size=2, sequence_length=12, seed=31)
    continuous, continuous_optimizer, continuous_state = _build()
    train_steps(continuous, continuous_optimizer, continuous_state, config, 6)

    paused, paused_optimizer, paused_state = _build()
    paused_state.dataset_position = 3
    paused_state.genome = {"version": 1, "layers": []}
    paused_state.mutation_history = [{"step": 3, "operation": "grow"}]
    paused_state.importance_state = {"expert": 0.5}
    paused_state.scheduler_state = {"last_epoch": 3}
    paused_state.experiment_config = {"seed": 31}
    paused_state.code_revision = "test-revision"
    train_steps(paused, paused_optimizer, paused_state, config, 3)
    checkpoint = tmp_path / "dense-baseline.pt"
    save_checkpoint(checkpoint, paused, paused_optimizer, paused_state)

    resumed, resumed_optimizer, _ = _build()
    resumed_state = load_checkpoint(checkpoint, resumed, resumed_optimizer)
    assert resumed_state.dataset_position == paused_state.dataset_position
    assert resumed_state.genome == paused_state.genome
    assert resumed_state.mutation_history == paused_state.mutation_history
    assert resumed_state.importance_state == paused_state.importance_state
    assert resumed_state.scheduler_state == paused_state.scheduler_state
    assert resumed_state.experiment_config == paused_state.experiment_config
    assert resumed_state.code_revision == paused_state.code_revision
    train_steps(resumed, resumed_optimizer, resumed_state, config, 3)

    assert resumed_state.step == continuous_state.step == 6
    assert resumed_state.tokens == continuous_state.tokens
    for actual, expected in zip(resumed.parameters(), continuous.parameters()):
        assert torch.equal(actual, expected)


def test_static_moe_trains_with_shared_runtime() -> None:
    torch.manual_seed(23)
    model = StaticMoECausalLM(
        StaticMoEConfig(
            vocab_size=32,
            context_size=12,
            hidden_size=24,
            num_layers=1,
            num_heads=4,
            ffn_size=48,
            num_experts=4,
            active_experts=2,
        )
    )
    config = TrainingConfig(batch_size=2, sequence_length=12, seed=37)
    state = TrainingState()
    losses = train_steps(model, make_optimizer(model, config), state, config, 2)

    assert model.parameter_count > 0
    assert state.step == 2
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)


def test_device_resolution_rejects_unavailable_cuda() -> None:
    from act.training.runtime import resolve_device

    if torch.cuda.is_available():
        assert resolve_device("cuda").type == "cuda"
    else:
        try:
            resolve_device("cuda")
        except RuntimeError:
            pass
        else:
            raise AssertionError("CUDA request should fail when CUDA is unavailable")