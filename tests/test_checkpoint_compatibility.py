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


@pytest.mark.parametrize(
    "training, message",
    [
        ({"step": "zero", "tokens": 0}, "step must be a non-negative integer"),
        ({"step": 0, "tokens": -1}, "tokens must be a non-negative integer"),
        ({"step": 0, "tokens": 0, "dataset_position": 1.5}, "dataset_position"),
        ({"step": 0, "tokens": 0, "importance_state": []}, "importance_state"),
    ],
)
def test_invalid_training_metadata_is_rejected_without_model_mutation(
    tmp_path, training, message
) -> None:
    model = DenseCausalLM(DenseBaselineConfig(hidden_size=8, num_heads=2))
    optimizer = torch.optim.AdamW(model.parameters())
    before = {name: value.clone() for name, value in model.state_dict().items()}
    checkpoint_state = {
        name: torch.full_like(value, 42.0) for name, value in before.items()
    }
    checkpoint = tmp_path / "bad-metadata.pt"
    checkpoint_state = {
        **model.state_dict(),
        **{name: torch.full_like(value, 42.0) for name, value in before.items()},
    }
    torch.save(
        {**_payload(), "model": checkpoint_state, "training": training},
        checkpoint,
    )

    with pytest.raises(ValueError, match=message):
        load_checkpoint(checkpoint, model, optimizer)

    assert all(
        torch.equal(value, model.state_dict()[name]) for name, value in before.items()
    )


def test_invalid_checkpoint_rng_is_rejected_before_global_rng_changes(tmp_path) -> None:
    model = DenseCausalLM(DenseBaselineConfig(hidden_size=8, num_heads=2))
    optimizer = torch.optim.AdamW(model.parameters())
    rng_before = torch.get_rng_state().clone()
    checkpoint = tmp_path / "bad-rng.pt"
    torch.save(
        {
            **_payload(),
            "model": model.state_dict(),
            "torch_rng_state": torch.tensor([1, 2], dtype=torch.uint8),
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="torch_rng_state is invalid"):
        load_checkpoint(checkpoint, model, optimizer)

    assert torch.equal(torch.get_rng_state(), rng_before)


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
