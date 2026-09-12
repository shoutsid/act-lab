import pytest

from act.model.dense import DenseBaselineConfig, DenseCausalLM
from act.training import (
    DistributedConfig,
    distributed_config,
    initialize_distributed,
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


def test_distributed_config_reads_torchrun_environment() -> None:
    config = distributed_config(
        environ={
            "RANK": "1",
            "WORLD_SIZE": "2",
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": "2",
            "MASTER_ADDR": "10.0.0.5",
            "MASTER_PORT": "29501",
        }
    )
    assert config == DistributedConfig(
        backend="auto",
        rank=1,
        world_size=2,
        local_rank=0,
        local_world_size=2,
        master_addr="10.0.0.5",
        master_port=29501,
    )
    assert config.enabled


def test_initialize_distributed_sets_env_and_uses_gloo(monkeypatch: pytest.MonkeyPatch) -> None:
    saw: dict[str, object] = {}

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)

    def fake_init_process_group(*, backend: str, rank: int, world_size: int, **kwargs: object) -> None:
        saw["backend"] = backend
        saw["rank"] = rank
        saw["world_size"] = world_size

    monkeypatch.setattr("torch.distributed.init_process_group", fake_init_process_group)

    initialize_distributed(
        DistributedConfig(
            backend="auto",
            rank=1,
            world_size=2,
            local_rank=0,
            local_world_size=2,
            master_addr="127.0.0.1",
            master_port=29501,
        )
    )

    assert saw["backend"] == "gloo"
    assert saw["rank"] == 1
    assert saw["world_size"] == 2
    assert __import__("os").environ["MASTER_PORT"] == "29501"
