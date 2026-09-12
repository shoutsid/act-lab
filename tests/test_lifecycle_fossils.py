import json

import pytest

from act.importance import LifecycleLedger


def test_lifecycle_ledger_is_append_only_and_jsonl_serializable() -> None:
    ledger = LifecycleLedger()
    fossil = ledger.record(
        step=4,
        operation="clone",
        expert_id="L0E4",
        layer_id="L0",
        architecture_hash_before="before",
        architecture_hash_after="after",
    )
    assert fossil.expert_id == "L0E4"
    assert len(ledger.records) == 1
    assert json.loads(ledger.to_jsonl())["operation"] == "clone"


def test_lifecycle_ledger_rejects_invalid_events() -> None:
    with pytest.raises(ValueError, match="step"):
        LifecycleLedger().record(
            step=-1,
            operation="birth",
            expert_id="L0E0",
            layer_id="L0",
            architecture_hash_before="a",
            architecture_hash_after="b",
        )
