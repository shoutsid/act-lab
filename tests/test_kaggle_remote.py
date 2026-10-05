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
    STATUS_REUSED,
    dataset_metadata,
    ensure_metadata,
    publish_handoff,
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

    def __init__(
        self,
        *,
        fail: Exception | None = None,
        empty_download: bool = False,
        owns: bool = False,
    ):
        self.fail = fail
        self.empty_download = empty_download
        self.owns = owns
        self.versions: list[tuple[str, str]] = []  # (dataset, version_notes)
        self.downloads: list[str] = []
        self.snapshots: list[Path] = []  # staged copies, one per version
        self.existence_checks: list[str] = []

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

    def dataset_exists(self, dataset: str) -> bool:
        self.existence_checks.append(dataset)
        return self.owns


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


def test_publish_handoff_does_not_re_upload_a_state_it_already_pushed(
    tmp_path: Path,
) -> None:
    """A second push of an unchanged handoff must not touch the API at all.

    Measured 2026-10-05: four sessions in a row ended ERROR inside the exit
    upload of the whole handoff, each losing the three pairs it had trained and
    leaving no attempt line behind. A session now pushes after every pair, so
    by the exit call the folder is normally unchanged since the last successful
    push - and re-uploading ~38 GiB for nothing is exactly the work that was
    being killed. The fingerprint of the seed versions is what makes the second
    call free.
    """
    dataset_root = _publish_local_seed(tmp_path, 42)
    first = FakeClient()

    published = publish_handoff(
        dataset_root,
        dataset="owner/act-checkpoints-v0-5-1",
        client=first,
        version_notes="after seed 42",
        log=lambda message: None,
    )

    assert published.status == STATUS_PUBLISHED
    assert published.fingerprint
    assert len(first.versions) == 1

    second = FakeClient()
    messages: list[str] = []
    reused = publish_on_exit(
        dataset_root,
        dataset="owner/act-checkpoints-v0-5-1",
        client=second,
        log=messages.append,
    )

    assert reused.status == STATUS_REUSED
    assert reused.seeds_published == 1
    assert reused.fingerprint == published.fingerprint
    # The version that already holds this state is the one the notes describe.
    assert reused.version_notes == "after seed 42"
    assert second.versions == []  # nothing was uploaded
    assert any("already published" in message for message in messages)
    # The driver reads this file for the session's outcome, so the last write
    # has to describe the call that just happened rather than the pair-time
    # push that made it free - and it still has to carry the seed count.
    log = json.loads(
        (dataset_root / kaggle_remote.PUBLISH_LOG_NAME).read_text(encoding="utf-8")
    )
    assert log["status"] == STATUS_REUSED
    assert log["seeds_published"] == 1
    assert log["fingerprint"] == published.fingerprint


def test_a_new_seed_makes_the_handoff_publish_again(tmp_path: Path) -> None:
    """The fingerprint tracks the seed versions, not the elapsed time.

    A queue exists so that the next pair changes the state: if adding a seed
    left the fingerprint alone, the session would report itself already
    published while sitting on a pair the dataset does not have.
    """
    dataset_root = _publish_local_seed(tmp_path, 42)
    publish_handoff(
        dataset_root,
        dataset="owner/act-checkpoints-v0-5-1",
        client=FakeClient(),
        log=lambda message: None,
    )
    # A second pair lands in the same dataset root, the way the pair loop
    # writes it: same session dir, one more seed directory in the handoff.
    session_dir = tmp_path / "sessions"
    second = run_v0_session(_config(tmp_path, 43), checkpoint_dir=session_dir)
    publish_seed(
        dataset_root, second, session_dir=session_dir, code_revision="test-rev"
    )

    client = FakeClient()
    result = publish_handoff(
        dataset_root,
        dataset="owner/act-checkpoints-v0-5-1",
        client=client,
        log=lambda message: None,
    )

    assert result.status == STATUS_PUBLISHED
    assert result.seeds_published == 2
    assert len(client.versions) == 1


def test_a_failed_push_is_never_reused(tmp_path: Path) -> None:
    """The failure path writes the same log, so reuse has to check its status.

    If a failed attempt could make the next call skip, a session would end
    believing in a version that does not exist - the one way this optimisation
    could turn a recoverable push failure into a silent loss.
    """
    dataset_root = _publish_local_seed(tmp_path, 42)
    failed = publish_handoff(
        dataset_root,
        dataset="owner/act-checkpoints-v0-5-1",
        client=FakeClient(fail=RuntimeError("500 server error")),
        log=lambda message: None,
        sleep=lambda seconds: None,
    )
    assert failed.status == STATUS_FAILED

    client = FakeClient()
    retried = publish_handoff(
        dataset_root,
        dataset="owner/act-checkpoints-v0-5-1",
        client=client,
        log=lambda message: None,
    )

    assert retried.status == STATUS_PUBLISHED
    assert retried.seeds_published == 1
    assert len(client.versions) == 1


def test_the_cli_builds_the_hook_the_runner_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two halves of the per-pair publish have to be wired to each other.

    ``run_pair_session`` accepts the callback and ``cli`` builds it, so a real
    session's per-pair durability rests on that wiring and not on either half
    alone: the failure mode of a missing hook is a session that trains pairs,
    publishes only at exit, and loses them all - the one this change fixes.
    """
    from act.training import cli

    dataset_root = _publish_local_seed(tmp_path, 42)
    client = FakeClient()
    monkeypatch.setattr(kaggle_remote, "default_client", lambda: client)
    args = types.SimpleNamespace(
        kaggle_dataset="owner/act-checkpoints-v0-5-1",
        dataset_root=dataset_root,
        dtype="fp16",
    )

    # No --kaggle-dataset, no hook: a workstation run publishes nowhere.
    args_without = types.SimpleNamespace(**{**vars(args), "kaggle_dataset": None})
    assert cli._remote_publisher(args_without, tokens_per_second=1000.0) is None

    hook = cli._remote_publisher(args, tokens_per_second=1000.0)
    assert hook is not None
    outcome = hook(42)

    assert outcome["status"] == STATUS_PUBLISHED
    assert outcome["seeds_published"] == 1
    assert len(client.versions) == 1
    # The notes name the pair, so the dataset's own history says which push a
    # version came from rather than only that a session made it.
    assert client.versions[0][1].endswith("after seed 42")


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


def test_restore_finds_every_seed_at_the_depth_the_service_actually_returns(
    tmp_path: Path,
) -> None:
    """The published tree's depth is the *service's* choice, not ours to assume.

    Measured 2026-09-21 by the startup-restore probe: the live dataset stores
    each seed at ``seed-15000/seed-15000/...`` (a seed is uploaded as
    ``seed-NNNN.tar`` whose members already carry that prefix, and the service
    keeps both), so the handoff parent's children have no manifest at all and
    the old locator fell through to matching a single *seed* directory. The
    probe restored **1 of the 4** published seeds, and a session that cannot see
    three completed seeds re-trains them - ~1.5 h each - while the queue reports
    the same 4/12 either way. This pins the reader rather than the writer:
    whatever depth the service returns, every complete seed comes back.
    """
    dataset_root = tmp_path / "dataset"
    handoff = dataset_root / HANDOFF_DATASET
    staging = tmp_path / "staging"
    for seed in (42, 43):
        name = f"seed-{seed:04d}"
        (handoff / name / "manifests").mkdir(parents=True)
        (handoff / name / "manifests" / "manifest.json").write_text(
            json.dumps({"seed": seed, "artifacts": [], "checksums": {}}) + "\n"
        )
        (handoff / name / "reports").mkdir()
        (handoff / name / "reports" / "seed-summary.json").write_text("{}\n")
        shutil.copytree(handoff / name, staging / name / name)
    shutil.rmtree(handoff)  # nothing local: everything below comes from remote

    class DoubledClient:
        """The nesting the live service returned, not the layout we asked for."""

        def download_files(self, dataset: str, target: Path) -> None:
            shutil.copytree(staging, target, dirs_exist_ok=True)

        def create_version(self, folder: Path, version_notes: str) -> str:
            raise AssertionError("a restore must never publish")

    restored = restore_latest(
        dataset_root, dataset="owner/ds", client=DoubledClient(), log=lambda m: None
    )

    assert restored == [42, 43]
    for seed in (42, 43):
        assert (handoff / f"seed-{seed:04d}" / "manifests" / "manifest.json").is_file()


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
    client = FakeClient()  # owns=False: the account has no such dataset yet

    def missing_download(dataset: str, target: Path) -> None:
        raise StubHTTPError("404 not found", response=StubResponse(404))

    client.download_files = missing_download  # type: ignore[method-assign]
    messages: list[str] = []
    slept: list[float] = []
    restored = restore_latest(
        tmp_path / "dataset",
        dataset="owner/ds",
        client=client,
        log=messages.append,
        sleep=slept.append,
    )
    assert restored == []
    assert any("training from scratch" in message for message in messages)
    # A dataset that does not exist is not a failure to retry: one download
    # attempt, one existence check, no backoff.
    assert client.existence_checks == ["owner/ds"]
    assert slept == []
    assert not any("retrying" in message for message in messages)


def test_restore_latest_refuses_to_train_over_a_dataset_it_cannot_download(
    tmp_path: Path, monkeypatch
) -> None:
    """An existing dataset that will not download must stop the session.

    The download endpoint answers 403/404 both for "no such dataset" and for a
    dataset it will not serve, and the old reading - warn and train from scratch
    - is what let a whole session's quota go to re-training seeds the queue had
    already published, over their own complete versions. Measured 2026-09-26/27
    on the primary queue: two sessions in a row restored nothing, re-trained
    seeds 15024-15026, and the driver saw no progress (4/12 -> 3/12).
    """
    stub_requests(monkeypatch, status_code=403)
    client = FakeClient(owns=True)

    def forbidden_download(dataset: str, target: Path) -> None:
        raise StubHTTPError("403 Forbidden", response=StubResponse(403))

    client.download_files = forbidden_download  # type: ignore[method-assign]
    messages: list[str] = []
    slept: list[float] = []
    with pytest.raises(RuntimeError, match="cannot restore owner/ds"):
        restore_latest(
            tmp_path / "dataset",
            dataset="owner/ds",
            client=client,
            log=messages.append,
            sleep=slept.append,
        )
    # It retried before giving up, and every failure named its cause - the
    # silent version of this cost a campaign three sessions of progress.
    assert len(client.existence_checks) == kaggle_remote.RESTORE_ATTEMPTS
    assert len(slept) == kaggle_remote.RESTORE_ATTEMPTS - 1
    assert any("403 Forbidden" in message for message in messages)


def test_restore_latest_recovers_from_a_transient_download_failure(
    tmp_path: Path, monkeypatch
) -> None:
    """A 403 that clears is not a reason to re-train a published block."""
    stub_requests(monkeypatch, status_code=403)
    dataset_root = _publish_local_seed(tmp_path, 42)
    client = FakeClient(owns=True)
    publish_on_exit(
        dataset_root,
        dataset="owner/ds",
        client=client,
        log=lambda message: None,
    )
    # Nothing local, so the restore below is the only source of seed 42.
    shutil.rmtree(dataset_root / HANDOFF_DATASET / "seed-0042")
    real_download = client.download_files
    attempts = {"count": 0}
    messages: list[str] = []

    def flaky_download(dataset: str, target: Path) -> None:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise StubHTTPError("403 Forbidden", response=StubResponse(403))
        real_download(dataset, target)

    client.download_files = flaky_download  # type: ignore[method-assign]
    restored = restore_latest(
        dataset_root,
        dataset="owner/ds",
        client=client,
        log=messages.append,
        sleep=lambda seconds: None,
    )

    assert restored == [42]
    assert attempts["count"] == 2
    assert any("retrying" in message for message in messages)


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


def test_dataset_exists_reads_the_owner_s_own_listing_by_paging() -> None:
    """The check that tells a missing dataset from one that will not download.

    The download endpoint cannot answer that question - it answers 403 in both
    cases - so the classifier asks the owner's listing instead. A check that
    stopped on page 1 would read a dataset listed on page 2 as missing, which is
    exactly the reading ("no remote version yet") that re-trains published
    seeds.
    """
    adapter = kaggle_remote.KaggleApiAdapter.__new__(kaggle_remote.KaggleApiAdapter)
    pages = {
        1: [types.SimpleNamespace(ref="owner/other")],
        2: [types.SimpleNamespace(ref="owner/ds")],
    }
    asked: list[int] = []

    class Api:
        def dataset_list(self, *, mine: bool, page: int):
            assert mine is True, "only the owner's own listing shows private datasets"
            asked.append(page)
            return pages.get(page, [])

    adapter._api = Api()

    assert adapter.dataset_exists("owner/ds") is True
    assert asked == [1, 2]
    assert adapter.dataset_exists("owner/missing") is False


class _ArchiveWillNotServe:
    """The platform state measured 2026-10-03 on `act-v0-5-2`.

    The whole-archive endpoint answered 404 for the current version (the
    dataset's listing reported `total_bytes: 0`) while every individual file
    streamed normally, so the archive *bundle* was broken, not the dataset. A
    session's restore is the only caller, and without a fallback it cannot
    start: it stops before training rather than re-training published seeds -
    which is what every session of that GPU week did.
    """

    def __init__(self, *, archive_error: Exception | None = None) -> None:
        self.archive_calls: list[str] = []
        self.file_calls: list[str] = []
        self.list_calls: list[tuple[str, str | None, int | None]] = []
        self.archive_error = archive_error or StubHTTPError(
            "404 not found", response=StubResponse(404)
        )
        # Same basename under two seeds: the collision the per-file scratch dirs exist for.
        self.files = [
            "seed-15024/seed-15024/manifests/manifest.json",
            "seed-15024/seed-15024/reports/seed-summary.json",
            "seed-15025/seed-15025/manifests/manifest.json",
        ]

    def dataset_download_files(self, dataset, path=None, unzip=False, quiet=True, **kw):
        self.archive_calls.append(dataset)
        raise self.archive_error

    def dataset_list_files(self, dataset, page_token=None, page_size=None):
        self.list_calls.append((dataset, page_token, page_size))
        return types.SimpleNamespace(
            dataset_files=[types.SimpleNamespace(name=name) for name in self.files],
            next_page_token=None,
        )

    def dataset_download_file(self, dataset, file_name, path=None, quiet=True, **kw):
        self.file_calls.append(file_name)
        # The real API saves under the basename and a fresh scratch dir is the
        # only thing keeping two seeds' `manifest.json`s apart.
        target = Path(path) / Path(file_name).name
        target.write_text(json.dumps({"name": file_name}) + "\n")
        return True


def test_the_adapter_falls_back_to_per_file_downloads_when_the_archive_will_not_serve(
    tmp_path: Path, monkeypatch
) -> None:
    """A broken archive bundle must not cost the session its restore.

    Measured 2026-10-03: `act-v0-5-2`'s current version 404s on the archive
    endpoint (`total_bytes: 0`) while every file is downloadable, so the GPU
    session died at restore before training - twice in a row, on the week's
    refresh. The fallback rebuilds the exact tree the archive would have
    unpacked, at the exact relative paths `published_seed_dirs` searches.
    """
    stub_requests(monkeypatch, status_code=404)
    adapter = kaggle_remote.KaggleApiAdapter.__new__(kaggle_remote.KaggleApiAdapter)
    api = _ArchiveWillNotServe()
    adapter._api = api
    target = tmp_path / "download"

    adapter.download_files("owner/ds", target)

    assert api.archive_calls == ["owner/ds"]
    assert api.file_calls == api.files
    # Each file landed at its own dataset-relative path, not flattened.
    for name in api.files:
        path = target / name
        assert path.is_file(), name
        assert json.loads(path.read_text()) == {"name": name}


def test_the_per_file_fallback_pages_the_listing(tmp_path: Path, monkeypatch) -> None:
    """A block bigger than one page must not be silently truncated.

    Six seeds is 30 files today, but the queue grows by a seed per session; a
    fallback that stopped on page 1 would restore a prefix of the block and
    re-train the rest, which is the failure the restore path exists to prevent.
    """
    stub_requests(monkeypatch, status_code=404)
    adapter = kaggle_remote.KaggleApiAdapter.__new__(kaggle_remote.KaggleApiAdapter)

    class Api(_ArchiveWillNotServe):
        def dataset_list_files(self, dataset, page_token=None, page_size=None):
            self.list_calls.append((dataset, page_token, page_size))
            if page_token is None:
                return types.SimpleNamespace(
                    dataset_files=[types.SimpleNamespace(name=self.files[0])],
                    next_page_token="page-2",
                )
            return types.SimpleNamespace(
                dataset_files=[types.SimpleNamespace(name=self.files[1])],
                next_page_token=None,
            )

    api = Api()
    adapter._api = api
    target = tmp_path / "download"

    adapter.download_files("owner/ds", target)

    assert [call[1] for call in api.list_calls] == [None, "page-2"]
    assert api.file_calls == api.files[:2]


def test_the_per_file_fallback_keeps_the_archive_error_shape(
    tmp_path: Path, monkeypatch
) -> None:
    """When both paths fail, the 404/403 reading has to survive.

    `restore_latest` classifies that status to tell "no such dataset yet" (the
    normal first-session bootstrap, train from scratch) from "will not serve"
    (stop before training). A fallback that replaced the archive's error with
    its own shape would turn a missing dataset into a hard failure, or worse,
    hide a real one.
    """
    stub_requests(monkeypatch, status_code=404)
    adapter = kaggle_remote.KaggleApiAdapter.__new__(kaggle_remote.KaggleApiAdapter)
    api = _ArchiveWillNotServe()

    def also_fails(dataset, file_name, path=None, quiet=True, **kw):
        raise StubHTTPError("404 not found", response=StubResponse(404))

    api.dataset_download_file = also_fails  # type: ignore[method-assign]
    adapter._api = api

    with pytest.raises(StubHTTPError) as caught:
        adapter.download_files("owner/ds", tmp_path / "download")
    assert caught.value.response.status_code == 404


def test_the_adapter_does_not_fall_back_on_an_unclassified_error(
    tmp_path: Path, monkeypatch
) -> None:
    """Fail-fast for real errors: a 500 is not a reason to ask for 30 files."""
    adapter = kaggle_remote.KaggleApiAdapter.__new__(kaggle_remote.KaggleApiAdapter)
    api = _ArchiveWillNotServe(archive_error=RuntimeError("507 insufficient storage"))
    adapter._api = api

    with pytest.raises(RuntimeError, match="507"):
        adapter.download_files("owner/ds", tmp_path / "download")
    assert api.file_calls == []
    assert api.list_calls == []


def test_default_client_requires_the_kaggle_package(monkeypatch) -> None:
    monkeypatch.setattr(kaggle_remote, "KAGGLE_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="kaggle package"):
        kaggle_remote.default_client()


def test_the_adapter_names_a_kaggle_that_is_present_but_broken(monkeypatch) -> None:
    """A resolvable name is not an importable API - the failure the TPU image had.

    Measured 2026-09-30, the null queue's first fire: on the TPU image `kaggle` resolves as a
    name while `kaggle.api` does not, so the run died at restore with `ModuleNotFoundError: No
    module named 'kaggle.api'` - a message that names no fix. `KAGGLE_AVAILABLE` cannot catch it
    without importing the package, and this module is deliberately import-time lazy, so the
    adapter's own import is what reports it - and it says what replaces a broken release.
    """
    monkeypatch.setattr(kaggle_remote, "KAGGLE_AVAILABLE", True)
    # A `kaggle` that is present but is not a package: the shape a broken install has.
    monkeypatch.setitem(sys.modules, "kaggle", types.ModuleType("kaggle"))
    monkeypatch.delitem(sys.modules, "kaggle.api", raising=False)

    with pytest.raises(RuntimeError, match=r"pip install -U kaggle"):
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
