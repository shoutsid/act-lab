import pytest
import torch

from act.model import DenseBaselineConfig, DenseCausalLM
from act.training import load_checkpoint, validate_checkpoint_payload


def _payload() -> dict[str, object]:
    return {
        "version": 3,
        "model": {},
        "optimizer": {},
        "training": {"step": 0, "tokens": 0},
    }


def test_checkpoint_compatibility_accepts_legacy_versions() -> None:
    for version in (1, 2, 3):
        payload = _payload()
        payload["version"] = version
        validate_checkpoint_payload(payload)


@pytest.mark.parametrize(
    "payload, message",
    [
        (None, "mapping"),
        ({"version": 9}, "unsupported"),
        ({"version": 3}, "missing model"),
        ({**_payload(), "training": {"step": 0}}, "missing tokens"),
    ],
)
def test_checkpoint_compatibility_rejects_malformed_payloads(
    payload: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_checkpoint_payload(payload)


def test_shape_mismatch_is_rejected_before_checkpoint_mutates_model(tmp_path) -> None:
    torch.manual_seed(17)
    model = DenseCausalLM(
        DenseBaselineConfig(vocab_size=16, context_size=8, hidden_size=8, num_layers=1)
    )
    optimizer = torch.optim.AdamW(model.parameters())
    before = {name: value.clone() for name, value in model.state_dict().items()}
    state = {name: value.clone() for name, value in before.items()}
    state["token_embedding.weight"] = torch.full_like(
        state["token_embedding.weight"], 42.0
    )
    state["blocks.0.attn.in_proj_weight"] = torch.zeros(1)
    checkpoint = tmp_path / "bad-shape.pt"
    torch.save(
        {
            **_payload(),
            "model": state,
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="has shape"):
        load_checkpoint(checkpoint, model, optimizer)

    assert all(
        torch.equal(value, model.state_dict()[name]) for name, value in before.items()
    )


def test_checkpoint_model_state_rejects_missing_and_unexpected_keys(tmp_path) -> None:
    model = DenseCausalLM(DenseBaselineConfig(hidden_size=8, num_heads=2))
    optimizer = torch.optim.AdamW(model.parameters())
    state = {name: value.clone() for name, value in model.state_dict().items()}
    state.pop(next(iter(state)))
    state["unexpected"] = torch.zeros(1)
    checkpoint = tmp_path / "bad-keys.pt"
    torch.save({**_payload(), "model": state}, checkpoint)

    with pytest.raises(ValueError, match="keys do not match"):
        load_checkpoint(checkpoint, model, optimizer)
