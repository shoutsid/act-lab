import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from act.model.dense import DenseBaselineConfig, DenseCausalLM
from act.training import (
    DistributedConfig,
    distributed_config,
    gpu_inventory,
    initialize_distributed,
    wrap_distributed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = REPO_ROOT / "scripts" / "v0_throughput_benchmark.py"


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


def test_gpu_inventory_counts_every_visible_device() -> None:
    inventory = gpu_inventory()

    assert set(inventory) == {"count", "names"}
    if torch.cuda.is_available():
        assert inventory["count"] == torch.cuda.device_count()
        assert inventory["count"] >= 1
        assert len(inventory["names"]) == inventory["count"]
    else:
        assert inventory == {"count": 0, "names": []}


def test_benchmark_artifact_carries_the_gpu_inventory(tmp_path) -> None:
    """The section 5.2 rate is only interpretable next to the device count."""
    output = tmp_path / "v0-benchmark.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(BENCHMARK),
            "--profile",
            "--device",
            "cpu",
            "--steps",
            "1",
            "--warmup",
            "0",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert completed.returncode == 0, completed.stderr

    report = json.loads(output.read_text())
    # The artifact states the host inventory and the training device separately,
    # so a rate measured on one device next to an idle second one is visible.
    assert report["gpu_inventory"] == gpu_inventory()
    assert report["device"] == "cpu"
    assert report["measured_tokens_per_second"] > 0
