import pytest

from act.training import validate_checkpoint_payload


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
