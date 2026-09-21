"""Tests for the section 5.2 versioned-dataset checkpoint handoff.

Publish must be atomic and all-or-nothing (manifest written last, swap after),
restore must verify every checksum before anything moves, and a seed that
travels working-dir -> dataset -> working-dir must resume bit-identically -
the plan's section 38 equivalence the 5.2 handoff exists to guarantee.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from act.training import V0SessionConfig, run_v0_session
from act.training.v0_handoff import (
    HANDOFF_DATASET,
    load_seed_summary,
    latest_published_seed,
    publish_seed,
    published_seeds,
    restore_seed,
    seed_directory,
    session_checkpoint_root,
)


def _config(tmp_path: Path, **overrides) -> V0SessionConfig:
    values = dict(
        seed=42,
        arms=("random", "adaptive"),
        declared=False,
        device="cpu",
        batch_size=2,
        vocab_size=64,
        context_size=256,
        hidden_size=64,
        depth=2,
        attention_heads=4,
        expert_count=4,
        expert_width=32,
        total_steps=12,
        mutation_interval=6,
        mutation_budget=2,
        probe_candidate_cap=4,
        probe_batch_count=1,
        validation_batches=2,
        checkpoint_interval=4,
        max_checkpoints=2,
    )
    values.update(overrides)
    return V0SessionConfig(**values)


def _newest_checkpoint(arm_dir: Path) -> Path:
    return max(arm_dir.glob("checkpoint-*.pt"), key=lambda p: int(p.stem.split("-")[-1]))


def test_publish_seed_writes_checksummed_layout(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions"
    dataset_root = tmp_path / "dataset"
    result = run_v0_session(_config(tmp_path), checkpoint_dir=session_dir)

    destination = publish_seed(
        dataset_root, result, session_dir=session_dir, code_revision="test-rev"
    )

    assert destination == dataset_root / HANDOFF_DATASET / "seed-0042"
    manifest = json.loads((destination / "manifests" / "manifest.json").read_text())
    assert manifest["seed"] == 42
    assert manifest["dataset_id"] == HANDOFF_DATASET
    assert manifest["code_revision"] == "test-rev"
    assert set(manifest["artifacts"]) == {
        "checkpoints/random/checkpoint-000012.pt",
        "checkpoints/adaptive/checkpoint-000012.pt",
        "reports/seed-summary.json",
    }
    for relative, checksum in manifest["checksums"].items():
        payload = destination / relative
        assert payload.is_file()
        assert len(checksum) == 64  # sha256 hex digest
    summary = json.loads((destination / "reports" / "seed-summary.json").read_text())
    assert summary["seed"] == 42
    assert set(summary["arms"]) == {"random", "adaptive"}
    assert summary["arms"]["adaptive"]["checkpoint"] == (
        "checkpoints/adaptive/checkpoint-000012.pt"
    )
    # Section 5.2 prices a pair with its arms running on separate devices, so
    # the published summary has to record where each arm trained: without it a
    # completed two-device pair reads exactly like a sequential two-arm one.
    assert summary["arms"]["adaptive"]["device"] == "cpu"


def test_publish_refuses_unmatched_mutation_budgets(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions"
    dataset_root = tmp_path / "dataset"
    result = run_v0_session(_config(tmp_path), checkpoint_dir=session_dir)
    object.__setattr__(result.arms[1], "mutation_count", result.arms[0].mutation_count + 1)
    with pytest.raises(ValueError, match="mutation budgets"):
        publish_seed(dataset_root, result, session_dir=session_dir, code_revision="rev")


def test_republish_replaces_seed_atomically(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions"
    dataset_root = tmp_path / "dataset"
    config = _config(tmp_path, total_steps=8)
    result = run_v0_session(config, checkpoint_dir=session_dir)
    publish_seed(dataset_root, result, session_dir=session_dir, code_revision="rev-1")

    continued = run_v0_session(_config(tmp_path), checkpoint_dir=session_dir)
    publish_seed(dataset_root, continued, session_dir=session_dir, code_revision="rev-2")

    manifest = json.loads(
        (seed_directory(dataset_root, 42) / "manifests" / "manifest.json").read_text()
    )
    assert manifest["code_revision"] == "rev-2"
    # No staging leftovers: the seed directory is the only entry for the seed.
    siblings = sorted(p.name for p in (dataset_root / HANDOFF_DATASET).iterdir())
    assert siblings == ["seed-0042"]


def test_restore_verifies_checksums_before_touching_session_dir(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions"
    dataset_root = tmp_path / "dataset"
    result = run_v0_session(_config(tmp_path), checkpoint_dir=session_dir)
    publish_seed(dataset_root, result, session_dir=session_dir, code_revision="rev")

    destination = seed_directory(dataset_root, 42)
    checkpoint = destination / "checkpoints" / "random" / "checkpoint-000012.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corruption")

    with pytest.raises(ValueError, match="checksum"):
        restore_seed(dataset_root, 42, session_dir=tmp_path / "fresh")
    # Nothing was re-materialized into the fresh session directory.
    assert not (tmp_path / "fresh").exists()


def test_restore_rejects_missing_version(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="seed 7"):
        restore_seed(tmp_path, 7, session_dir=tmp_path / "sessions")


def test_latest_published_seed_ignores_incomplete_versions(tmp_path: Path) -> None:
    assert latest_published_seed(tmp_path) is None
    session_dir = tmp_path / "sessions"
    dataset_root = tmp_path / "dataset"
    result = run_v0_session(_config(tmp_path), checkpoint_dir=session_dir)
    publish_seed(dataset_root, result, session_dir=session_dir, code_revision="rev")
    assert latest_published_seed(dataset_root) == (42, seed_directory(dataset_root, 42))

    # A directory without a manifest is an incomplete publish, not a version.
    incomplete = dataset_root / HANDOFF_DATASET / "seed-0100"
    (incomplete / "checkpoints").mkdir(parents=True)
    assert latest_published_seed(dataset_root) == (42, seed_directory(dataset_root, 42))

    result2 = run_v0_session(_config(tmp_path, seed=200), checkpoint_dir=session_dir)
    publish_seed(dataset_root, result2, session_dir=session_dir, code_revision="rev")
    assert latest_published_seed(dataset_root) == (200, seed_directory(dataset_root, 200))


def test_clean_working_dir_removes_seed_state(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions"
    dataset_root = tmp_path / "dataset"
    result = run_v0_session(_config(tmp_path), checkpoint_dir=session_dir)
    seed_root = session_dir / "seed-42"
    assert seed_root.exists()
    publish_seed(
        dataset_root, result, session_dir=session_dir,
        code_revision="rev", clean_working_dir=True,
    )
    assert not seed_root.exists()
    # The dataset still restores it.
    restored = restore_seed(dataset_root, 42, session_dir=session_dir)
    assert (restored / "arm-random" / "checkpoint-000012.pt").exists()
    assert (restored / "arm-adaptive" / "checkpoint-000012.pt").exists()


def test_publish_restore_resume_is_bit_identical(tmp_path: Path) -> None:
    """Working-dir -> dataset -> working-dir travel preserves the exact run.

    This is the section 5.2 reason the handoff exists: a restarted session
    must equal an uninterrupted one (plan section 38).
    """
    config = _config(tmp_path, total_steps=16, checkpoint_interval=4)
    session_dir = tmp_path / "sessions"
    dataset_root = tmp_path / "dataset"

    # Reference: one uninterrupted run.
    reference = run_v0_session(config, checkpoint_dir=tmp_path / "reference")

    # Interrupted: run partway, publish, wipe the working dir, restore, finish.
    partial_config = _config(tmp_path, total_steps=8, checkpoint_interval=4)
    run_v0_session(partial_config, checkpoint_dir=session_dir)
    partial_result = run_v0_session(partial_config, checkpoint_dir=session_dir)
    publish_seed(
        dataset_root, partial_result, session_dir=session_dir,
        code_revision="rev", clean_working_dir=True,
    )
    assert not (session_dir / "seed-42").exists()
    restore_seed(dataset_root, 42, session_dir=session_dir)
    resumed = run_v0_session(config, checkpoint_dir=session_dir)

    for before, after in zip(reference.arms, resumed.arms):
        assert after.final_loss == before.final_loss
        assert after.architecture_hash == before.architecture_hash
        assert (after.training_flops, after.search_flops) == (
            before.training_flops,
            before.search_flops,
        )
        assert json.dumps(after.mutation_history, sort_keys=True) == json.dumps(
            before.mutation_history, sort_keys=True
        )
        assert after.parameter_count == before.parameter_count


def test_load_seed_summary_round_trips(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions"
    dataset_root = tmp_path / "dataset"
    result = run_v0_session(_config(tmp_path), checkpoint_dir=session_dir)
    publish_seed(dataset_root, result, session_dir=session_dir, code_revision="rev")
    summary = load_seed_summary(dataset_root, 42)
    assert summary["seed"] == 42
    assert summary["arms"]["adaptive"]["final_loss"] == result.arms[1].final_loss


def test_session_checkpoint_root_layout(tmp_path: Path) -> None:
    assert session_checkpoint_root(tmp_path, 42) == tmp_path / "seed-42"


def test_published_seeds_lists_the_versions_that_carry_a_summary(tmp_path: Path) -> None:
    """The block's membership is what is *present*, not what an invocation asked for.

    ``check_block_treatment_uniform`` reads this instead of its own seed queue, because a
    resumed campaign is launched ``--seeds 15004-15011`` and the versions that disagree
    with it are the ones already finished - the case a reader of the requested queue is
    blind to by construction.
    """
    dataset_root = tmp_path / "dataset"
    session_dir = tmp_path / "sessions"
    for seed in (42, 7):  # published newest-first, so membership is not the write order
        publish_seed(
            dataset_root,
            run_v0_session(_config(tmp_path, seed=seed), checkpoint_dir=session_dir),
            session_dir=session_dir,
            code_revision="rev",
        )
    handoff = dataset_root / HANDOFF_DATASET
    # Decoys: a seed directory with no summary is not a version, and neither is a
    # directory that is not named like a seed.
    (handoff / "seed-0043").mkdir(parents=True)
    (handoff / "seed-notanumber").mkdir(parents=True)
    (handoff / "notes").mkdir(parents=True)

    assert sorted(published_seeds(dataset_root)) == [7, 42]
    assert published_seeds(tmp_path / "nothing-here") == []


def test_published_seeds_is_ascending_whatever_the_filesystem_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Path.glob`` returns the filesystem's order, so the reader has to impose one.

    Stated separately from membership because the first version of the membership test
    asserted ``[7, 42]`` and *passed a mutant that removed the sort* - the directory
    happened to be walked in the right order, which is exactly the kind of assertion that
    looks like a pin and is not one. Here glob is forced to hand the seeds back in
    descending order, so the sort is the only thing that can produce the answer.
    """
    dataset_root = tmp_path / "dataset"
    for seed in (42, 7, 13):
        reports = seed_directory(dataset_root, seed) / "reports"
        reports.mkdir(parents=True)
        (reports / "seed-summary.json").write_text("{}")

    real_glob = Path.glob
    monkeypatch.setattr(
        Path,
        "glob",
        lambda self, pattern: iter(
            sorted(real_glob(self, pattern), reverse=True)
        ),
    )

    assert published_seeds(dataset_root) == [7, 13, 42]
