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
import os
import shutil
import subprocess
import sys
import types
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
import tarfile

from act.training import V0SessionConfig, run_v0_session
from act.training.v0_handoff import HANDOFF_DATASET, publish_seed, restore_seed


class StubResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class StubHTTPError(Exception):
    def __init__(self, message: str, response: StubResponse | None = None) -> None:
        super().__init__(message)
        self.response = response


def stub_requests(monkeypatch, status_code: int) -> None:
    """Stand in for the parts of `requests` the adapter touches.

    `requests` is an optional Kaggle-side dependency (it ships with `kaggle`), so
    the tests must not need it installed - which the module docstring already
    promises and this file previously broke by importing it directly. The adapter
    detects a missing dataset from the exception *type*, so a stub module with
    that type is enough to exercise the real branch, and it keeps the test
    independent of what happens to be installed in the running interpreter.
    """
    module = types.ModuleType("requests")
    module.exceptions = types.SimpleNamespace(HTTPError=StubHTTPError)  # type: ignore[attr-defined]
    module.models = types.SimpleNamespace(  # type: ignore[attr-defined]
        Response=lambda: StubResponse(status_code)
    )
    monkeypatch.setitem(sys.modules, "requests", module)


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


class _ExtractingService:
    """A client that behaves like the real service's extraction.

    ``stage_push`` uploads one tar per seed directory; Kaggle stores those as
    flat files and expands them server-side, and keeps the archives'
    *internal* paths. ``download_files`` therefore re-materialises internal
    paths, not the archive names - which is exactly what makes the arcname
    prefix load-bearing.
    """

    def __init__(self, staging: Path) -> None:
        self.staging = staging
        self.downloads: list[str] = []

    def download_files(self, dataset: str, target: Path) -> None:
        self.downloads.append(dataset)
        target.mkdir(parents=True, exist_ok=True)
        for entry in sorted(self.staging.iterdir()):
            if entry.suffix == ".tar":
                with tarfile.open(entry) as tar:
                    tar.extractall(target)
            else:
                shutil.copy2(entry, target / entry.name)


def test_the_publish_round_trip_rebuilds_the_handoff_tree(tmp_path: Path) -> None:
    """Stage, upload, extract, download and merge - the whole remote joint.

    The adapter's docstring records why the staging exists at all (the naive
    upload silently produced an *empty* dataset, and per-directory tarring
    flattened every seed to the archive root where they collided), but
    nothing pinned it. A regression here corrupts the published dataset while
    still reporting ``published``, and the run would spend its remaining
    quota re-training seeds that were already done.
    """
    dataset_root = _publish_local_seed(tmp_path, 15000)
    _publish_local_seed(tmp_path, 15001)
    handoff = dataset_root / HANDOFF_DATASET
    dataset = "owner/act-v0-5-1"
    ensure_metadata(handoff, dataset)

    staging = tmp_path / "staging"
    kaggle_remote.stage_push(handoff, staging)

    # One archive per seed, each rooted at its own seed directory: without the
    # prefix both seeds' checkpoints would land on the same paths.
    archives = sorted(path.name for path in staging.glob("*.tar"))
    assert archives == ["seed-15000.tar", "seed-15001.tar"]
    with tarfile.open(staging / "seed-15000.tar") as tar:
        names = tar.getnames()
    assert names and all(name.startswith("seed-15000/") for name in names)

    # Restore into a *fresh* root, as a later session does.
    fresh = tmp_path / "fresh"
    service = _ExtractingService(staging)
    restored = restore_latest(fresh, dataset=dataset, client=service)

    assert restored == [15000, 15001]
    assert service.downloads == [dataset]
    # Byte-exact, not merely present: restore_seed verifies every checksum the
    # pre-upload manifest recorded, so a mangled archive fails here.
    session_dir = tmp_path / "restored-session"
    assert restore_seed(fresh, 15000, session_dir=session_dir).is_dir()
    assert restore_seed(fresh, 15001, session_dir=session_dir).is_dir()


def test_stage_push_copies_plain_files_and_the_metadata(tmp_path: Path) -> None:
    source = tmp_path / "handoff"
    (source / "seed-15000").mkdir(parents=True)
    (source / "seed-15000" / "seed-summary.json").write_text("{}")
    (source / "revision.txt").write_text("abc\n")
    (source / METADATA_NAME).write_text('{"id": "owner/slug"}')

    staging = tmp_path / "staging"
    kaggle_remote.stage_push(source, staging)

    # A loose file stays a loose file; the metadata is re-copied from source.
    assert (staging / "revision.txt").read_text() == "abc\n"
    assert json.loads((staging / METADATA_NAME).read_text())["id"] == "owner/slug"
    assert (staging / "seed-15000.tar").is_file()


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
        sleep=lambda seconds: None,
    )

    assert result.status == STATUS_FAILED
    assert "RuntimeError" in result.error
    assert seed_summary.is_file()  # local state untouched
    # The error text says how many attempts were spent, so a deterministic
    # failure is distinguishable from a flaky one.
    assert result.error.count("409 version conflict") == kaggle_remote.PUBLISH_ATTEMPTS


def test_publish_on_exit_retries_a_transient_push_failure(tmp_path: Path) -> None:
    """A flaky push must not cost the session its pairs.

    The handoff roots live in the container overlay, so when the kernel ends
    there is no local copy left to re-push: a failed push here is the one step
    whose failure discards a whole session's training, and the usual cause is
    transient (an API error, a dropped connection) rather than deterministic.
    """
    dataset_root = _publish_local_seed(tmp_path, 42)
    client = FakeClient()
    real_create_version = client.create_version
    attempts = {"count": 0}
    slept: list[float] = []

    def flaky_create_version(folder: Path, version_notes: str) -> str:
        attempts["count"] += 1
        if attempts["count"] < kaggle_remote.PUBLISH_ATTEMPTS:
            raise RuntimeError("500 server error")
        return real_create_version(folder, version_notes)

    client.create_version = flaky_create_version  # type: ignore[method-assign]
    messages: list[str] = []

    result = publish_on_exit(
        dataset_root,
        dataset="owner/act-checkpoints-v0-5-1",
        client=client,
        log=messages.append,
        sleep=slept.append,
    )

    assert result.status == STATUS_PUBLISHED
    assert result.seeds_published == 1
    assert attempts["count"] == kaggle_remote.PUBLISH_ATTEMPTS
    # Backing off between attempts, once per failure before the successful one.
    assert len(slept) == kaggle_remote.PUBLISH_ATTEMPTS - 1
    assert any("retrying" in message for message in messages)


def test_publish_on_exit_survives_a_metadata_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The metadata write is inside the guard, not an unguarded first step.

    It writes *into* the handoff folder, so it can fail (a full disk) - and it
    used to run before the try, which made the "never raises" exit path raise.
    """
    dataset_root = _publish_local_seed(tmp_path, 42)

    def failing_metadata(folder, dataset):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(kaggle_remote, "ensure_metadata", failing_metadata)

    result = publish_on_exit(
        dataset_root,
        dataset="owner/act-checkpoints-v0-5-1",
        client=FakeClient(),
        log=lambda message: None,
        sleep=lambda seconds: None,
    )

    assert result.status == STATUS_FAILED
    assert "OSError" in result.error and "No space left on device" in result.error


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


def test_restore_latest_prices_its_own_download(tmp_path: Path) -> None:
    """The startup restore is the one cost that grows with the queue: name it.

    Measured 2026-09-21: a session ended ERROR after 1.32 h with no log and no
    output, and this restore - the whole published dataset, downloaded before a
    token is trained - is one of the two things known to occupy a session for
    that long. It is ~22.7 GiB at four seeds and roughly 6.4 GiB more per seed
    published, it was priced nowhere, and an unpriced cost cannot be reasoned
    about when a week's quota is finite.
    """
    dataset_root = _publish_local_seed(tmp_path, 42)
    client = FakeClient()
    publish_on_exit(
        dataset_root, dataset="owner/ds", client=client, log=lambda message: None
    )
    shutil.rmtree(dataset_root / HANDOFF_DATASET / "seed-0042")
    lines: list[str] = []

    restored = restore_latest(
        dataset_root,
        dataset="owner/ds",
        client=client,
        log=lines.append,
    )

    assert restored == [42]
    cost = [line for line in lines if "remote restore:" in line]
    assert len(cost) == 1
    assert "owner/ds downloaded in" in cost[0]
    assert "GiB on disk" in cost[0]


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


def test_restore_latest_treats_missing_dataset_as_bootstrap(
    tmp_path: Path, monkeypatch
) -> None:
    """Session 1 has nothing to restore: absence warns and continues."""
    stub_requests(monkeypatch, status_code=404)
    client = FakeClient()

    def missing_download(dataset: str, target: Path) -> None:
        raise StubHTTPError("404 not found", response=StubResponse(404))

    client.download_files = missing_download  # type: ignore[method-assign]
    messages: list[str] = []
    restored = restore_latest(
        tmp_path / "dataset",
        dataset="owner/ds",
        client=client,
        log=messages.append,
    )
    assert restored == []
    assert any("training from scratch" in message for message in messages)


def test_restore_latest_raises_on_unexpected_client_error(tmp_path: Path) -> None:
    """Fail-fast: errors other than 'no version yet' must stop the launch."""
    client = FakeClient()

    def failing_download(dataset: str, target: Path) -> None:
        raise RuntimeError("507 insufficient storage")

    client.download_files = failing_download  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="507"):
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


def test_kaggle_api_adapter_is_lazy() -> None:
    """The adapter module must not import `kaggle` when it is imported.

    The old form of this test asserted `KAGGLE_AVAILABLE is True`, with the
    comment "installed in this env" - which made a statement about the machine
    into a test, and failed on every interpreter without the optional package
    even though the *code* is explicitly built to run without it (the flag is an
    `importlib.util.find_spec` probe, not an import).

    The property worth pinning is the import graph, and it is measured in a fresh
    interpreter for two reasons: the answer does not depend on whether `kaggle`
    happens to be installed, and it cannot be polluted by anything else in this
    session that already imported it.
    """
    probe = (
        "import sys, act.ops.kaggle_remote as m;"
        " print(m.KAGGLE_AVAILABLE, 'kaggle' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": "src"},
        cwd=Path(__file__).resolve().parent.parent,
    )
    available, imported = result.stdout.split()
    # `available` may be either value depending on the interpreter; what must hold
    # in both is that probing for the package did not import it.
    assert imported == "False", result.stdout
    assert available == ("True" if kaggle_remote.KAGGLE_AVAILABLE else "False")
    assert kaggle_remote.KaggleApiAdapter.__init__ is not None
