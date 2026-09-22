"""Pin the two-seed concurrency rehearsal, and that its comparison can see a difference.

The rehearsal is the blocker on the null queue's two-seeds-per-pair amendment: it measures
whether a seed's result moves when another seed trains beside it. A check like that has one
failure mode worth guarding, and it is not a false negative - it is **passing because it
cannot fail**. Two ways that could happen here, both pinned below: a result field that is
compared by nothing, and a record whose verdict was not actually computed from the payloads
it carries.

The committed record is read as an artifact rather than trusted: its own ``identical`` flag is
recomputed from its own two payloads, so an edited flag and a real difference cannot both go
unnoticed. That is the same discipline the record check applies to every published number.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import two_seed_concurrency_rehearsal as rehearsal  # noqa: E402

RECORD = ROOT / "artifacts" / "null-pair-rehearsal" / "rehearsal.json"


def _payload(seed: int = 15024, loss: float = 9.061052083969116) -> dict:
    return {
        "seed": seed,
        "arms": [
            {
                "mode": "null",
                "final_loss": loss,
                "validation_loss": loss,
                "quality": loss,
                "mutation_count": 3,
                "parameter_count": 1_000,
                "active_parameter_count": 500,
                "training_flops": 4_891_921_612_800,
                "search_flops": 0,
                "architecture_hash": "b9fc55adb199",
                "mutation_history": [{"step": 100, "operation": "null"}],
            }
        ],
    }


def test_the_compared_set_is_what_the_arm_carries_minus_the_metadata() -> None:
    """Derived from the arm, not chosen: a silently excluded field is an invisible difference.

    ``wall_time_seconds``, ``tokens_per_second`` and ``device`` are metadata the arm carries so
    a reader can tell a concurrent pair from a sequential one; comparing them would make the
    rehearsal fail for the one difference it is *supposed* to produce. Everything else has to
    be compared, so the partition is asserted against the dataclass itself - the arm gains a
    field and this test fails until someone decides which side it belongs on.
    """
    from dataclasses import fields

    from act.training.v0_session import V0ArmResult

    carried = {field.name for field in fields(V0ArmResult)}
    assert carried == set(rehearsal.RESULT_FIELDS) | set(rehearsal.METADATA_FIELDS)
    assert not set(rehearsal.RESULT_FIELDS) & set(rehearsal.METADATA_FIELDS)


@pytest.mark.parametrize("field", rehearsal.RESULT_FIELDS)
def test_the_comparison_sees_every_compared_field_move(field: str) -> None:
    """No field is compared by nothing: a moved value is reported by name, for all of them."""
    left = _payload()
    right = deepcopy(left)
    moved = {"mutated": True} if field == "mutation_history" else "moved-value"
    right["arms"][0][field] = moved
    differences = rehearsal.compare(left, right)
    assert differences, f"{field} moved and the comparison returned no difference"
    assert any(field in line for line in differences), differences


def test_the_comparison_is_silent_when_nothing_moved() -> None:
    """The other direction, so the pins above cannot be satisfied by always reporting."""
    assert rehearsal.compare(_payload(), _payload()) == []


def test_the_record_recomputes_its_own_verdict() -> None:
    """The artifact is evidence only if its flag follows from its payloads.

    Re-run here rather than trusted, so an edited ``identical`` - or a run whose two scenarios
    shared their output directory and therefore compared a payload with itself - has to show
    up as a disagreement between the flag and the data.
    """
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    sequential = record["runs"]["sequential"]
    concurrent = record["runs"]["concurrent"]
    assert sorted(sequential) == sorted(concurrent) == sorted(str(s) for s in record["seeds"])

    recomputed: list[str] = []
    for key in sequential:
        assert "arms" in sequential[key] and "arms" in concurrent[key]
        recomputed += rehearsal.compare(sequential[key], concurrent[key])
    assert recomputed == record["differences"]
    assert record["identical"] is (recomputed == [])

    # The two scenarios have to be two runs: the seeds differ from each other, so the payloads
    # are not one result copied into both keys.
    losses = {key: sequential[key]["arms"][0]["validation_loss"] for key in sequential}
    assert len(set(losses.values())) == len(losses)
