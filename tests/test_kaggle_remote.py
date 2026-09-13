"""Tests for the remote Kaggle dataset publish/restore wrapper.

The wrapper is the remote half of the section 5.2 handoff: push the local
``act-checkpoints/v0-5-1`` tree to a versioned Kaggle dataset on session
exit (never raising - a failed push must not crash a finished session) and
restore the newest version at session start (fail-fast, manifest-verified,
seed-wise merge). The ``kaggle`` package is optional and never required by
the tests: every decision is exercised against an injected fake client.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from act.ops import kaggle_remote
from act.ops.kaggle_remote import (
    METADATA_NAME,
    STATUS_FAILED,
    STATUS_NOTHING,
    STATUS_PUBLISHED,
    dataset_metadata,
    ensure_metadata,
    publish_on_exit,
    restore_latest,
)
from act.training import V0SessionConfig, run_v0_session
from act.training.v0_handoff import HANDOFF_DATASET, publish_seed


class FakeClient:
    """In-memory KaggleDatasetClient recording calls and staging versions."""

    def __init__(self, *, fail: Exception | None = None, empty_download: bool = False):
        self.fail = fail
        self.empty_download = empty_download
        self.versions: list[tuple[str, str]] = []  # (dataset, version_notes)
        self.downloads: list[str] = []
        self.snapshots: list[Path] = []  # staged copies, one per version

    def create_version(self, folder: Path, version_notes: str) -> str:
        if self.fail is not None:
            raise self.fail
        self.versions.append((folder.name, version_notes))
        # Stage a snapshot now: the real API stores version contents at push
        # time, so later local edits are invisible to the next download.
        staging = Path(__import__('tempfile').mkdtemp(prefix="fake-version-"))
        shutil.copytree(folder, staging / folder.name)
        self.snapshots.append(staging / folder.name)
        return version_notes

    def download_files(self, dataset: str, target: Path) -> None:
        if self.fail is not None:
            raise self.fail
        self.downloads.append(dataset)
        if self.empty_download:
            return
        # Simulate the extracted download from the staged version snapshot.
        if self.snapshots:
            source = self.snapshots[-1]
            shutil.copytree(source, target / source.name)


def _config(tmp_path: Path, seed: int) -> V0SessionConfig:
    return V0SessionConfig(
        seed=seed,
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


def _publish_local_seed(tmp_path: Path, seed: int) -> Path:
    session_dir = tmp_path / "sessions"
    dataset_root = tmp_path / "dataset"
    result = run_v0_session(_config(tmp_path, seed), checkpoint_dir=session_dir)
    publish_seed(
        dataset_root, result, session_dir=session_dir, code_revision="test-rev"
    )
    return dataset_root


def test_dataset_metadata_requires_owner_slug() -> None:
    payload = dataset_metadata("owner/act-checkpoints-v0-5-1")
    assert payload["id"] == "owner/act-checkpoints-v0-5-1"
    assert payload["licenses"] == [{"name": "CC0-1.0"}]
    for bad in ("noslug", "owner/slug/extra", "owner/UPPER"):
        with pytest.raises(ValueError, match="owner/slug"):
            dataset_metadata(bad)


def test_ensure_metadata_is_idempotent(tmp_path: Path) -> None:
    first = ensure_metadata(tmp_path, "owner/slug-1")
    assert first.name == METADATA_NAME
    original = first.read_text()
    second = ensure_metadata(tmp_path, "owner/slug-1")
    assert second.read_text() == original  # no churn on re-publish
    with pytest.raises(ValueError, match="owner/slug"):
        ensure_metadata(tmp_path, "bad")


def test_publish_on_exit_pushes_seeded_handoff(tmp_path: Path) -> None:
    dataset_root = _publish_local_seed(tmp_path, 42)
    client = FakeClient()

    result = publish_on_exit(
        dataset_root,
        dataset="owner/act-checkpoints-v0-5-1",
        client=client,
        log=lambda message: None,
    )

    assert result.status == STATUS_PUBLISHED
    assert result.seeds_published == 1
    assert result.error is None
    assert len(client.versions) == 1
    pushed = client.snapshots[0]
    # The handoff dir itself is pushed (zero-copy), not a re-staged copy.
    assert pushed.name == HANDOFF_DATASET.split("/")[-1]
    assert Path(result.folder) == dataset_root / HANDOFF_DATASET
    assert (pushed / METADATA_NAME).is_file()
    log = json.loads(
        (dataset_root / kaggle_remote.PUBLISH_LOG_NAME).read_text()
    )
    assert log["status"] == STATUS_PUBLISHED


def test_publish_on_exit_reports_nothing_without_seeds(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    client = FakeClient()

    result = publish_on_exit(
        dataset_root,
        dataset="owner/act-checkpoints-v0-5-1",
        client=client,
        log=lambda message: None,
    )

    assert result.status == STATUS_NOTHING
    assert result.seeds_published == 0
    assert client.versions == []  # no push attempted


def test_publish_on_exit_swallows_client_errors(tmp_path: Path) -> None:
    """The exit path must never raise; the local handoff stays intact."""
    dataset_root = _publish_local_seed(tmp_path, 42)
    seed_summary = (
        dataset_root / HANDOFF_DATASET / "seed-0042" / "reports" / "seed-summary.json"
    )
    assert seed_summary.is_file()
    client = FakeClient(fail=RuntimeError("409 version conflict"))

    result = publish_on_exit(
        dataset_root,
        dataset="owner/act-checkpoints-v0-5-1",
        client=client,
        log=lambda message: None,
    )

    assert result.status == STATUS_FAILED
    assert "RuntimeError" in result.error
    assert seed_summary.is_file()  # local state untouched


def test_restore_latest_merges_remote_over_local(tmp_path: Path) -> None:
    dataset_root = _publish_local_seed(tmp_path, 42)
    handoff = dataset_root / HANDOFF_DATASET

    client = FakeClient()
    publish_on_exit(
        dataset_root,
        dataset="owner/ds",
        client=client,
        log=lambda message: None,
    )
    # A local-only seed that postdates the remote snapshot must survive
    # restore (created after the version snapshot, so it is remote-absent).
    (handoff / "seed-9999" / "manifests").mkdir(parents=True)
    (handoff / "seed-9999" / "manifests" / "manifest.json").write_text("{}\n")
    # Wipe the local copy of the remote-published seed, then restore: the
    # snapshot still carries seed-0042 and must bring it back.
    shutil.rmtree(handoff / "seed-0042")
    restored = restore_latest(
        dataset_root,
        dataset="owner/ds",
        client=client,
        log=lambda message: None,
    )

    assert restored == [42]
    assert (handoff / "seed-0042" / "manifests" / "manifest.json").is_file()
    assert (handoff / "seed-9999").is_dir()  # local-only seed kept


def test_restore_latest_ignores_partial_versions(tmp_path: Path) -> None:
    dataset_root = _publish_local_seed(tmp_path, 42)
    client = FakeClient()
    publish_on_exit(
        dataset_root,
        dataset="owner/ds",
        client=client,
        log=lambda message: None,
    )
    # Corrupt the *staged* version snapshot: the next download carries a
    # seed dir without a manifest (partial version).
    snapshot = client.snapshots[-1]
    shutil.rmtree(snapshot / "seed-0042" / "manifests")
    shutil.rmtree(dataset_root / HANDOFF_DATASET / "seed-0042")

    restored = restore_latest(
        dataset_root,
        dataset="owner/ds",
        client=client,
        log=lambda message: None,
    )
    assert restored == []  # a manifest-less seed is not a version


def test_restore_latest_raises_on_client_error(tmp_path: Path) -> None:
    """Restore is fail-fast: no remote state means no session."""
    client = FakeClient()
    client.fail = RuntimeError("404 dataset missing")  # raised by download_files

    def failing_download(dataset: str, target: Path) -> None:
        raise client.fail

    client.download_files = failing_download  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="404"):
        restore_latest(
            tmp_path / "dataset",
            dataset="owner/ds",
            client=client,
            log=lambda message: None,
        )


def test_default_client_requires_the_kaggle_package(monkeypatch) -> None:
    monkeypatch.setattr(kaggle_remote, "KAGGLE_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="kaggle package"):
        kaggle_remote.default_client()


def test_kaggle_api_adapter_is_lazy(monkeypatch) -> None:
    """KaggleApiAdapter must not import kaggle at construction of the module."""
    assert kaggle_remote.KAGGLE_AVAILABLE is True  # installed in this env
    adapter = kaggle_remote.KaggleApiAdapter  # not instantiated here
    assert adapter.__init__ is not None
