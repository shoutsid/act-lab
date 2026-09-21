"""Pin the section 5.63 Gate-5 transfer rehearsal and the primitive it exercises.

Gate 5's wording - "fresh architecture remains competitive" - asks whether a discovered
*topology* is worth carrying without the training it accumulated. The published checkpoint
carries both, and until section 5.63 the only reader of its genome was the session's resume
branch, which rebuilds from that genome and then restores ``model``/``optimizer``/RNG state
over it. An architecture could therefore only travel with its training, and the claim had no
path to be made on.

These tests pin both halves: that ``genome_from_checkpoint`` and
``fresh_init_from_checkpoint`` do what such a read needs, and that the fresh path stays
observably distinct from the restored one. That last property is the one that matters -
if the two were the same object, a weights-carrying number could be read as a
fresh-architecture number and nothing would notice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import gate5_rehearsal  # noqa: E402

from act.topology.genome import ArchitectureGenome  # noqa: E402
from act.training.v0_handoff import newest_arm_checkpoint  # noqa: E402
from act.training.v0_session import (  # noqa: E402
    V0SessionConfig,
    fresh_init_from_checkpoint,
    genome_from_checkpoint,
    run_v0_session,
)

RECORD = ROOT / "artifacts" / "study-summaries" / "gate5-transfer-rehearsal.json"


def _record() -> dict:
    return json.loads(RECORD.read_text())


def test_the_record_is_present_and_every_check_holds() -> None:
    checks = _record()["built_in_checks"]
    assert checks, "the rehearsal recorded no checks"
    assert all(checks.values()), [name for name, ok in checks.items() if not ok]


def test_the_record_says_it_is_not_a_gate5_result() -> None:
    record = _record()
    assert record["record_type"] == "transfer-rehearsal"
    assert record["report_section"] == "5.63"
    assert record["synthetic"] is True
    assert "not a Gate-5 result" in record["what_this_is"]


def test_the_record_states_the_two_paths_are_distinguishable() -> None:
    paths = _record()["paths"]
    assert paths["paths_are_distinguishable"] is True
    # The rehearsal must never record the fresh path as reproducing the published weights:
    # that is the reading the whole section exists to rule out.
    assert paths["fresh_weights_equal_published"] is False
    assert paths["restore_reproduces_published_exactly"] is True


def test_every_transferred_parameter_differs_from_the_published_one() -> None:
    transferred = _record()["transferred"]
    assert transferred["parameters"] > 0
    assert transferred["parameters_differing"] == transferred["parameters"]
    assert transferred["max_abs_difference"] > 0.0
    assert transferred["genome_layers"] == transferred["genome_depth"]


@pytest.fixture(scope="module")
def published(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """One real tiny session checkpoint, shared across the functional tests."""
    directory = tmp_path_factory.mktemp("gate5-transfer")
    config = V0SessionConfig(**gate5_rehearsal.REHEARSAL_CONFIG)
    run_v0_session(config, checkpoint_dir=directory)
    arm_dir = directory / f"seed-{config.seed}" / f"arm-{config.arms[0]}"
    checkpoint = newest_arm_checkpoint(arm_dir)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return {"config": config, "checkpoint": checkpoint, "payload": payload}


def test_genome_from_checkpoint_returns_exactly_the_recorded_genome(published) -> None:
    recorded = ArchitectureGenome.from_dict(published["payload"]["training"]["genome"])
    assert genome_from_checkpoint(published["checkpoint"]).to_dict() == recorded.to_dict()


def test_fresh_init_keeps_topology_and_replaces_every_value(published) -> None:
    model, genome = fresh_init_from_checkpoint(published["checkpoint"], published["config"])
    assert genome.depth > 0
    weights = published["payload"]["model"]
    state = model.state_dict()
    # Topology transferred: same parameter names, same shapes.
    assert sorted(state) == sorted(weights)
    assert all(tuple(state[name].shape) == tuple(weights[name].shape) for name in state)
    # Training did not: every learned value differs.
    for name, tensor in model.named_parameters():
        assert not torch.equal(tensor.detach(), weights[name]), name


def test_a_restore_reproduces_the_published_weights_exactly(published) -> None:
    """The path the fresh one has to be distinguishable from."""
    model, _ = fresh_init_from_checkpoint(published["checkpoint"], published["config"])
    model.load_state_dict(published["payload"]["model"])
    weights = published["payload"]["model"]
    for name, tensor in model.named_parameters():
        assert torch.equal(tensor.detach(), weights[name]), name


def test_fresh_init_is_deterministic_at_a_fixed_seed(published) -> None:
    from dataclasses import replace

    first, _ = fresh_init_from_checkpoint(published["checkpoint"], published["config"])
    second, _ = fresh_init_from_checkpoint(published["checkpoint"], published["config"])
    other, _ = fresh_init_from_checkpoint(
        published["checkpoint"], replace(published["config"], seed=published["config"].seed + 1)
    )
    # At a fixed seed the initialization is reproduced exactly, parameter for parameter.
    for name, tensor in first.named_parameters():
        assert torch.equal(tensor.detach(), second.state_dict()[name]), name
    # A different seed moves the initialized weights. Not *every* parameter is
    # seed-dependent - a normalization scale starts at exactly one - so the assertion is
    # that the only parameters a seed cannot move are the normalization scales. A weight
    # matrix that stopped responding to the seed would be a real defect, and this fails on
    # it rather than tolerating it.
    unchanged = [
        name
        for name, tensor in first.named_parameters()
        if torch.equal(tensor.detach(), other.state_dict()[name])
    ]
    assert all("norm" in name for name in unchanged), unchanged
    assert len(unchanged) < len(list(first.named_parameters()))


def test_a_payload_without_a_genome_raises(tmp_path: Path) -> None:
    path = tmp_path / "no-genome.pt"
    torch.save({"version": 5, "training": {"genome": None}}, path)
    with pytest.raises(ValueError, match="carries no genome"):
        genome_from_checkpoint(path)


@pytest.mark.parametrize("version", [3, None])
def test_a_payload_that_is_not_a_session_checkpoint_raises(
    tmp_path: Path, version: int | None
) -> None:
    path = tmp_path / "not-a-session.pt"
    payload: dict = {"training": {"genome": {}}}
    if version is not None:
        payload["version"] = version
    torch.save(payload, path)
    with pytest.raises(ValueError):
        genome_from_checkpoint(path)
