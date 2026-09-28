import json
import os
import subprocess
import sys
import types
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
from act.training import runtime as runtime_module
from act.training.runtime import resolve_device, step_marker

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


def test_resolve_device_accepts_an_explicit_cuda_index() -> None:
    """The pair model addresses one GPU per arm, so ``cuda:N`` must work.

    Kaggle's T4 shape hands out two devices (measured 2026-09-15) and nothing
    could name the second one: ``resolve_device`` only understood ``cuda``,
    which torch reads as device 0.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this host")
    assert resolve_device("cuda:0").index == 0
    # An index the host does not have must fail at startup, not mid-step.
    with pytest.raises(ValueError, match="does not exist"):
        resolve_device(f"cuda:{torch.cuda.device_count()}")
    with pytest.raises(ValueError, match="integer"):
        resolve_device("cuda:x")


def test_resolve_device_rejects_nonsense_and_keeps_cpu() -> None:
    assert resolve_device("cpu").type == "cpu"
    with pytest.raises(ValueError, match="device must be"):
        resolve_device("gpu")
    with pytest.raises(ValueError, match="integer"):
        resolve_device("cuda:-1")


def test_resolve_device_accepts_xla_or_names_the_missing_package() -> None:
    """Kaggle's TPU quota is separate from the GPU's, so a session may be rehearsed on it.

    ``scripts/tpu_session_rehearsal.py`` runs the registered session path against
    this device type before any queue moves there (2026-09-27), which means the
    string has to be resolvable. The half worth pinning is the refusal: without
    ``torch_xla`` the name would otherwise resolve and fail inside the first
    ``.to()``, which on a queued kernel is hours after the run started.
    """
    try:
        import torch_xla  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="torch_xla is not installed"):
            resolve_device("xla")
    else:
        assert resolve_device("xla").type == "xla"


def test_step_marker_is_a_no_op_where_a_step_is_already_a_step() -> None:
    """The CPU and CUDA paths must not pay for XLA's execution model.

    ``torch`` runs eagerly there, so the step *is* the step; a marker that did
    anything on those devices would change the campaign's measured loop to
    rehearse a TPU. Identity rather than behaviour: the same no-op object, so a
    later branch cannot quietly become a synchronize call.
    """
    assert step_marker(torch.device("cpu")) is runtime_module._no_step_marker
    assert step_marker(torch.device("cuda")) is runtime_module._no_step_marker
    assert step_marker(torch.device("cpu"))() is None


def test_step_marker_is_the_xla_sync_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """On XLA the marker is ``torch_xla.sync`` - the call ``xm.mark_step()`` aliases.

    Measured 2026-09-28: the session rehearsal's step loop never marked a step,
    and its session phase produced no checkpoint inside three hours on a v5e-8
    where the same twenty development-fidelity steps take 24 s on a CPU. The
    probe's own measured loop marks after ``optimizer.step()``; this is the same
    call, reached without a second import.
    """
    calls: list[int] = []
    fake = types.ModuleType("torch_xla")
    fake.sync = lambda: calls.append(1)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch_xla", fake)

    marker = step_marker(torch.device("xla"))
    marker()

    assert marker is fake.sync
    assert calls == [1]


def test_step_marker_falls_back_to_the_older_torch_xla_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The platform's torch_xla version is not ours to pin.

    ``sync`` is the 2.8 name; an image that predates it still has
    ``xla_model.mark_step``. A missing marker would put the campaign back in the
    unmarked loop this fix exists for, on hardware nobody can debug from here.
    """
    calls: list[int] = []
    fake = types.ModuleType("torch_xla")  # deliberately no `sync`
    core = types.ModuleType("torch_xla.core")
    xla_model = types.ModuleType("torch_xla.core.xla_model")
    xla_model.mark_step = lambda: calls.append(1)  # type: ignore[attr-defined]
    core.xla_model = xla_model  # type: ignore[attr-defined]
    fake.core = core  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch_xla", fake)
    monkeypatch.setitem(sys.modules, "torch_xla.core", core)
    monkeypatch.setitem(sys.modules, "torch_xla.core.xla_model", xla_model)

    marker = step_marker(torch.device("xla"))
    marker()

    assert calls == [1]


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

    # The schedule names the rate it was derived from, so a run that trains a
    # different dtype than the rule read cannot hide that in the artifact.
    assert report["schedule_dtype"] == report["dtype_decision"]
    assert report["schedule_tokens_per_second"] > 0
    assert report["tokens_per_second_by_dtype"]["fp32"] == report[
        "measured_tokens_per_second"
    ]
    if report["device"] == "cpu":
        # No CPU fp16 path exists, so the schedule must fall back to fp32.
        assert report["schedule_tokens_per_second"] == report["measured_tokens_per_second"]
