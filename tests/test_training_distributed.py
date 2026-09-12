import pytest

from act.model.dense import DenseBaselineConfig, DenseCausalLM
from act.training import (
    distributed_config,
    wrap_distributed,
)


def test_single_process_distributed_config_is_a_noop() -> None:
    config = distributed_config(environ={})
    model = DenseCausalLM(DenseBaselineConfig(hidden_size=8, num_heads=2))
    assert not config.enabled
    assert wrap_distributed(model, config) is model


def test_distributed_config_rejects_invalid_rank() -> None:
    with pytest.raises(ValueError, match="rank"):
        distributed_config(environ={"RANK": "2", "WORLD_SIZE": "2"})
