import sys
import types

import pytest
import torch

from act.model import dense as dense_module
from act.model.checkpointing import activation_checkpoint
from act.model.dense import DenseBaselineConfig, DenseCausalLM
from act.training import TrainingConfig, TrainingState, make_optimizer, train_steps


def _install_fake_torch_xla(monkeypatch: pytest.MonkeyPatch, calls: list) -> None:
    """A torch_xla whose ``checkpoint`` records how it was called.

    The real one is a separate package with a fork of torch's utility, and the property
    under test is *which* call the dispatch makes rather than what the fork computes, so a
    recording stand-in pins it without a TPU. The parents go in ``sys.modules`` too: a
    ``from torch_xla.utils.checkpoint import checkpoint`` imports them, and stubbing only
    the leaf is an ImportError rather than the call under test.
    """
    root = types.ModuleType("torch_xla")
    utils = types.ModuleType("torch_xla.utils")
    checkpoint_module = types.ModuleType("torch_xla.utils.checkpoint")

    def checkpoint(function, *args, **kwargs):
        calls.append({"function": function, "args": args, "kwargs": kwargs})
        return function(*args)

    checkpoint_module.checkpoint = checkpoint
    utils.checkpoint = checkpoint_module
    root.utils = utils
    for name, module in (
        ("torch_xla", root),
        ("torch_xla.utils", utils),
        ("torch_xla.utils.checkpoint", checkpoint_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_every_other_backend_keeps_torchs_non_reentrant_checkpoint(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    """The GPU campaign's call, unchanged: same function, same ``use_reentrant=False``.

    ``cuda:0`` is resolved by name rather than by hardware - the recorder replaces the
    call, so no device is touched and the test says the same thing on a CPU-only host.
    """
    seen: dict = {}

    def recorder(function, *args, **kwargs):
        seen.update(function=function, args=args, kwargs=kwargs)
        return function(*args)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", recorder)

    def block(value: torch.Tensor) -> torch.Tensor:
        return value * 2

    tensor = torch.ones(3, requires_grad=True)
    result = activation_checkpoint(torch.device(device), block, tensor)

    assert torch.equal(result, torch.full((3,), 2.0))
    assert seen["function"] is block
    assert seen["args"] == (tensor,)
    assert seen["kwargs"] == {"use_reentrant": False}


def test_the_xla_call_is_torch_xlas_fork_and_never_passes_use_reentrant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a TPU the call has to be the fork, and the fork only has the reentrant path.

    Measured 2026-09-29: the declared-shape session died 4.5 s into its first step on
    ``AttributeError: module 'torch' has no attribute 'xla'``, raised by
    ``torch/utils/checkpoint.py``'s ``_get_device_module`` on the ``use_reentrant=False``
    path. ``torch_xla.utils.checkpoint`` is the fork that resolves that module for XLA -
    and its public entry point *rejects* ``use_reentrant=False``, so passing upstream's
    keyword through would turn the fix into a ``ValueError`` on the one device it exists
    for. This pins both halves.
    """
    calls: list = []
    _install_fake_torch_xla(monkeypatch, calls)

    def block(value: torch.Tensor) -> torch.Tensor:
        return value + 1

    tensor = torch.ones(2, requires_grad=True)
    result = activation_checkpoint(torch.device("xla"), block, tensor)

    assert torch.equal(result, torch.tensor([2.0, 2.0]))
    assert len(calls) == 1
    assert calls[0]["function"] is block
    assert calls[0]["args"] == (tensor,)
    assert "use_reentrant" not in calls[0]["kwargs"]


def test_the_xla_call_names_the_missing_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ``xla`` device without the package is an error that says so.

    The alternative is falling through to the upstream call and letting the reader find
    the failure at the first training step of a queued run, which is what this cost on
    2026-09-29. ``None`` in ``sys.modules`` is Python's own "this import is refused", so
    the test means the same thing on a host that has torch_xla installed.
    """
    monkeypatch.setitem(sys.modules, "torch_xla.utils.checkpoint", None)

    with pytest.raises(RuntimeError, match="torch_xla is not installed"):
        activation_checkpoint(torch.device("xla"), lambda value: value, torch.ones(1))


def test_a_checkpointed_model_routes_every_block_through_the_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The blocks call the dispatch, not ``torch.utils.checkpoint`` directly.

    The device rule lives in one place only if the call sites go through it: a model that
    kept importing ``checkpoint`` would train on the CPU and CUDA and die on a TPU, which
    is exactly the shape this defect had. Recorded per call, and one call per block per
    step - two layers, two steps.
    """
    devices: list[str] = []
    real = dense_module.activation_checkpoint

    def recorder(device: torch.device, function, *args):
        devices.append(device.type)
        return real(device, function, *args)

    monkeypatch.setattr(dense_module, "activation_checkpoint", recorder)

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
    train_steps(model, make_optimizer(model, config), TrainingState(), config, 2)

    assert devices == ["cpu"] * 4


def test_the_v0_session_path_checkpoints_its_blocks_through_the_dispatch(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The path that would run on a TPU is the one that has to take the dispatch.

    The training-runtime test above covers the dense baseline, and the liveness suite covers the
    V0 session only for *trajectory* inertness - a checkpointing call that silently did nothing
    would leave every loss identical and pass there. This is the V0 session itself, one **null**
    arm (the queue section 5.14's option 2 moves), with the flag the declared config sets: one
    call per block per step, on the tensors' own device.
    """
    from act.model import v0_moe as v0_moe_module
    from act.training.v0_session import V0SessionConfig, run_v0_session

    devices: list[str] = []
    real = v0_moe_module.activation_checkpoint

    def recorder(device: torch.device, function, *args):
        devices.append(device.type)
        return real(device, function, *args)

    monkeypatch.setattr(v0_moe_module, "activation_checkpoint", recorder)

    run_v0_session(
        V0SessionConfig(
            seed=11,
            device="cpu",
            arms=("null",),
            vocab_size=64,
            context_size=32,
            hidden_size=32,
            depth=2,
            attention_heads=4,
            expert_count=4,
            expert_width=16,
            total_steps=2,
            checkpoint_interval=2,
            validation_batches=1,
            mutation_interval=100,
            activation_checkpointing=True,
        ),
        checkpoint_dir=tmp_path / "session",
    )

    assert devices == ["cpu"] * 4


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
