"""The real-LM checkpoint's optimizer state is bound to names, not positions.

The defect these exist for was found by trying to train, not by reading: the
2026-09-14 real-LM run resumed on 2026-09-25 into a one-position shift between
the model's parameters and its saved Adam moments, and the first step died
inside ``torch._foreach_lerp_`` with

    RuntimeError: output with shape [384] doesn't match the broadcast shape [1152, 384]

which names no parameter, no cause, and no remedy. The shift is the interesting
part, because it is *invisible at load time*: ``Optimizer.state_dict()`` is
position-keyed and ``nn.Module.state_dict()`` is name-keyed, so a model whose
parameter **order** changed while its names and shapes stayed the same loads its
weights cleanly and attaches its moments to the wrong tensors. Nothing raises
until the first optimizer step, and if the neighbouring shapes happened to agree
it would not raise at all - it would train with another parameter's momentum.

So the moments are stored by name and checked against the parameter they claim
to belong to, and a positional payload is refused with both remedies named.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import train_real_lm as real_lm  # noqa: E402

from act.model import V0MoECausalLM, V0MoEConfig  # noqa: E402

# Small enough to build and step on CPU in milliseconds; the checkpoint contract
# does not depend on the size, and the real shape is exercised by the local
# checkpoint test at the bottom rather than by ten of these.
SMALL = dict(
    vocab_size=128,
    context_size=16,
    hidden_size=32,
    depth=2,
    attention_heads=2,
    experts_per_layer=(4, 4),
    active_experts=2,
    expert_width=32,
)


def _trained(seed: int = 0) -> tuple[V0MoECausalLM, torch.optim.Optimizer]:
    """A model and an optimizer with populated moments (two real steps)."""
    torch.manual_seed(seed)
    model = V0MoECausalLM(V0MoEConfig(**SMALL))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.randint(0, SMALL["vocab_size"], (2, SMALL["context_size"]))
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        model.loss(tokens).backward()
        optimizer.step()
    return model, optimizer


def _payload() -> dict:
    model, optimizer = _trained()
    return real_lm.checkpoint_payload(model, optimizer, 3, [{"step": 3, "val_loss": 1.0}])


def _restore(payload: dict, *, reset: bool = False, seed: int = 1):
    torch.manual_seed(seed)
    model = V0MoECausalLM(V0MoEConfig(**SMALL))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # The weights always come back; `reset` governs only the moments, which is the
    # half the historical checkpoint could not carry.
    model.load_state_dict(payload["model"])
    real_lm.restore_optimizer(
        model, optimizer, payload, Path("checkpoint.pt"), reset=reset
    )
    return model, optimizer


def test_the_moments_round_trip_by_parameter_name() -> None:
    model, optimizer = _trained()
    payload = real_lm.checkpoint_payload(model, optimizer, 5, [])
    assert payload["optimizer"]["moments"], "no moments were captured - nothing to test"

    restored_model, restored = _restore(payload)

    by_name = dict(restored_model.named_parameters())
    for name, moment in payload["optimizer"]["moments"].items():
        state = restored.state[by_name[name]]
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(state[key], moment[key]), f"{name}.{key} did not survive"
    # And nothing was attached to a parameter the payload did not name.
    assert len(restored.state) == len(payload["optimizer"]["moments"])


def test_the_payload_carries_no_positional_parameter_list() -> None:
    """``params`` is what made the state positional; it is what must not be saved."""
    payload = _payload()
    for group in payload["optimizer"]["groups"]:
        assert "params" not in group


def test_a_positional_payload_is_refused_and_both_remedies_are_named() -> None:
    """The legacy format, refused rather than guessed at."""
    model, optimizer = _trained()
    payload = real_lm.checkpoint_payload(model, optimizer, 5, [])
    payload["optimizer"] = optimizer.state_dict()  # the old, position-keyed shape

    with pytest.raises(SystemExit) as info:
        _restore(payload)
    message = str(info.value)
    assert "by position" in message
    assert "--reset-optimizer" in message
    assert "delete the checkpoint" in message


def test_reset_optimizer_resumes_weights_with_fresh_moments() -> None:
    model, optimizer = _trained()
    payload = real_lm.checkpoint_payload(model, optimizer, 5, [])
    restored_model, optimizer = _restore(payload, reset=True)

    assert optimizer.state == {}, "reset must leave no moments behind"
    # The weights are the checkpoint's, which is the half reset is meant to keep.
    for name, weight in model.state_dict().items():
        assert torch.equal(restored_model.state_dict()[name], weight)


def test_a_shifted_moment_is_caught_by_name_rather_than_by_luck() -> None:
    """The defect, reproduced: a one-position shift between names and moments.

    This is what a positional resume produces when the parameter order moves. The
    loader accepts it silently; the name-keyed one compares each moment to the
    parameter it claims and refuses, naming both.
    """
    payload = _payload()
    names = list(payload["optimizer"]["moments"])
    by_name = dict(V0MoECausalLM(V0MoEConfig(**SMALL)).named_parameters())
    # A neighbour whose shape differs, so the shift is detectable rather than
    # accidentally shape-compatible - which is exactly the case that used to slip
    # through and train on another parameter's momentum.
    shifted_from = next(
        name
        for name in names
        if tuple(by_name[name].shape) != tuple(by_name[names[(names.index(name) + 1) % len(names)]].shape)
    )
    neighbour = names[(names.index(shifted_from) + 1) % len(names)]
    payload["optimizer"]["moments"][shifted_from] = payload["optimizer"]["moments"][neighbour]

    with pytest.raises(SystemExit) as info:
        _restore(payload)
    message = str(info.value)
    assert "shape mismatches" in message
    assert shifted_from in message


def test_a_parameter_the_checkpoint_does_not_know_is_named() -> None:
    payload = _payload()
    payload["optimizer"]["moments"]["not.a.real.parameter"] = {
        "exp_avg": torch.zeros(4),
        "exp_avg_sq": torch.zeros(4),
    }
    with pytest.raises(SystemExit) as info:
        _restore(payload)
    assert "not.a.real.parameter" in str(info.value)
    assert "missing" in str(info.value)


def test_the_local_checkpoint_restores_by_name() -> None:
    """The actual artifact, when this machine has it.

    The checkpoint is 250 MB of trained weights and moments and is deliberately
    not committed, so this is one of the checks that can only run where the run
    happened; a fresh clone skips it by name.
    """
    path = ROOT / "artifacts" / "real-lm" / "checkpoints" / "latest.pt"
    if not path.is_file():
        pytest.skip("artifacts/real-lm/checkpoints/latest.pt not present (local run output)")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    report = real_lm.json.loads((ROOT / "artifacts" / "real-lm" / "report.json").read_text())
    spec = report["model"]
    model = V0MoECausalLM(
        V0MoEConfig(
            vocab_size=spec["vocab_size"],
            context_size=spec["context_size"],
            hidden_size=spec["hidden_size"],
            depth=spec["depth"],
            attention_heads=spec["attention_heads"],
            experts_per_layer=(spec["experts_per_layer"],) * spec["depth"],
            active_experts=spec["active_experts"],
            expert_width=spec["expert_width"],
        )
    )
    model.load_state_dict(payload["model"])
    assert "moments" in payload["optimizer"], (
        "the checkpoint on disk predates the name-keyed fix; re-run the documented "
        "training command to rewrite it"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    real_lm.restore_optimizer(model, optimizer, payload, path, reset=False)
    by_name = dict(model.named_parameters())
    for name, moment in payload["optimizer"]["moments"].items():
        for key in ("exp_avg", "exp_avg_sq"):
            assert tuple(moment[key].shape) == tuple(by_name[name].shape), name
