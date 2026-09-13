"""Remote Kaggle dataset publish/restore for the section 5.2 checkpoint handoff.

The local half lives in ``act.training.v0_handoff``: every completed seed (or
deadline-stop snapshot) is staged, checksummed, and swapped into
``act-checkpoints/v0-5-1/seed-NNNN/`` under the dataset root. This module is
the remote half the status tracker lists as the remaining piece: pushing that
handoff directory to a versioned Kaggle dataset when a session exits, and
restoring it at session start (plan sections 38 and 40: a restarted session
is equivalent to an uninterrupted run, and a session never ends without
publishing its state).

Design constraints, pre-committed:

- **The exit path never raises.** ``publish_on_exit`` catches every client
  error and records the outcome; a failed remote push must not turn twelve
  hours of completed local training into a crashed session. The local
  handoff stays intact either way - the next session can re-push it.
- **Restore fails fast.** ``restore_latest`` raises on client errors: a
  session that requested remote state and cannot get it must stop before
  training, not mid-pair.
- **Zero-copy publish.** The pushed folder is the handoff directory itself
  plus a generated ``dataset-metadata.json``; no gigabyte-scale staging
  copies inside the 20 GB Kaggle working directory.
- **No hard kaggle dependency.** The package is optional (preinstalled on
  Kaggle images, absent elsewhere); the module imports cleanly without it
  and every testable decision is exercised against an injected fake client.

The merge rule for restore: a remote seed version replaces a matching local
one, and local-only seeds are kept (a locally-published seed that postdates
the remote snapshot is never rolled back).
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol

from .kaggle import _sha256  # shared checksum helper (same contract)

KAGGLE_AVAILABLE = importlib.util.find_spec("kaggle") is not None

METADATA_NAME = "dataset-metadata.json"
DEFAULT_LICENSE = "CC0-1.0"
PUBLISH_LOG_NAME = "remote-publish-log.json"

STATUS_PUBLISHED = "published"
STATUS_FAILED = "failed"
STATUS_NOTHING = "nothing_to_publish"


def _handoff_dirname() -> str:
    """The handoff dataset layout, imported lazily (avoids an import cycle)."""
    from act.training.v0_handoff import HANDOFF_DATASET

    return HANDOFF_DATASET


def _manifest_seeds(folder: Path) -> list[Path]:
    """Seed directories in a handoff tree that carry a complete manifest."""
    return sorted(
        path
        for path in folder.glob("seed-*")
        if (path / "manifests" / "manifest.json").is_file()
    )


def dataset_metadata(dataset: str) -> dict[str, object]:
    """The ``dataset-metadata.json`` payload the Kaggle API requires.

    ``dataset`` is ``owner/slug``; the slug must satisfy Kaggle's naming
    rules (lowercase letters, digits, dashes).
    """
    owner, _, slug = dataset.partition("/")
    if not owner or not slug or slug != slug.lower() or "/" in slug:
        raise ValueError(
            f"dataset must be 'owner/slug' with a lowercase slug: {dataset!r}"
        )
    return {
        "id": dataset,
        "title": f"ACT checkpoints {slug}",
        "licenses": [{"name": DEFAULT_LICENSE}],
    }


def ensure_metadata(folder: str | Path, dataset: str) -> Path:
    """Write (or refresh) the metadata file inside the pushed folder.

    Idempotent: an existing file describing the same dataset is left
    untouched so repeated publishes do not churn the version contents.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / METADATA_NAME
    payload = json.dumps(dataset_metadata(dataset), indent=2, sort_keys=True) + "\n"
    if not path.is_file() or path.read_text() != payload:
        path.write_text(payload)
    return path


@dataclass(frozen=True)
class RemotePublishResult:
    """Outcome of one exit-time publish attempt."""

    dataset: str
    folder: str
    seeds_published: int
    status: str
    version_notes: str
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class KaggleDatasetClient(Protocol):
    """The slice of the Kaggle API this wrapper needs; injectable for tests."""

    def create_version(self, folder: Path, version_notes: str) -> str:
        """Push ``folder`` as a new dataset version; return a version ref."""
        ...

    def download_files(self, dataset: str, target: Path) -> None:
        """Download and extract the dataset's latest version into ``target``."""
        ...


class KaggleApiAdapter:
    """Production client wrapping the optional ``kaggle`` package.

    Constructed lazily (see :func:`default_client`) so importing this module
    never requires the package.

    Upload layout (established by live probes against the real API, 2026-09):
    Kaggle datasets are flat, but the service auto-extracts uploaded
    ``.tar``/``.zip`` blobs and keeps the archive's *internal* paths as
    dataset files. The default ``upload_files(dir_mode=...)`` handling is
    wrong for nested trees: ``"skip"`` silently drops every subdirectory
    (an empty checkpoint dataset), and ``"tar"`` archives each
    subdirectory's *contents* without a name prefix, so multiple seeds all
    flatten to the archive root and collide. This adapter therefore stages
    one tar per seed directory with internal paths rooted at
    ``seed-NNNN/...``, pushes the archives as flat files, and lets the
    service-side extraction rebuild the handoff tree verbatim (verified
    checksum-identical through the round trip).

    The first-version bootstrap is best-effort: ``dataset_create_version``
    rejects a dataset that does not exist yet (observed as 403/404), and
    the adapter falls back to ``dataset_create_new`` for that case only.
    """

    def __init__(self) -> None:
        if not KAGGLE_AVAILABLE:
            raise RuntimeError(
                "the kaggle package is required for remote dataset access; "
                "it is preinstalled on Kaggle images (pip install kaggle elsewhere)"
            )
        import tarfile

        self._tarfile = tarfile
        from kaggle.api.kaggle_api_extended import KaggleApi

        self._api = KaggleApi()
        self._api.authenticate()

    def _stage_push(self, folder: Path, staging: Path) -> None:
        """Stage the pushed folder: one tar per seed dir, files copied flat."""
        for entry in sorted(folder.iterdir()):
            if entry.name == METADATA_NAME:
                continue  # re-copied last, from its canonical source
            if entry.is_dir():
                archive = staging / f"{entry.name}.tar"
                with self._tarfile.open(archive, "w") as tar:
                    for path in sorted(entry.rglob("*")):
                        if path.is_file():
                            tar.add(path, arcname=str(Path(entry.name) / path.relative_to(entry)))
            elif entry.is_file():
                shutil.copy2(entry, staging / entry.name)
        metadata = folder / METADATA_NAME
        if metadata.is_file():
            shutil.copy2(metadata, staging / METADATA_NAME)

    def create_version(self, folder: Path, version_notes: str) -> str:
        folder = Path(folder)
        with tempfile.TemporaryDirectory(prefix="act-kaggle-push-") as scratch:
            staging = Path(scratch) / "push"
            staging.mkdir()
            self._stage_push(folder, staging)
            try:
                self._api.dataset_create_version(
                    str(staging), version_notes, dir_mode="skip", quiet=True
                )
            except Exception as error:  # noqa: BLE001 - API error types vary
                if not _is_missing_dataset_error(error):
                    raise
                # First publish: the dataset does not exist yet.
                self._api.dataset_create_new(
                    str(staging), dir_mode="skip", quiet=True
                )
        return version_notes

    def download_files(self, dataset: str, target: Path) -> None:
        self._api.dataset_download_files(dataset, path=str(target), unzip=True)


def _is_missing_dataset_error(error: Exception) -> bool:
    """True when a request failed because no remote version exists.

    Session 1 has nothing to restore (plan section 40: "locate the latest
    checkpoint dataset" finds nothing and continues); a missing dataset is
    that normal bootstrap state, not a failure. Any other error is real.
    """
    try:
        import requests
    except ImportError:  # pragma: no cover - requests ships with kaggle
        return False
    response = getattr(error, "response", None)
    return (
        isinstance(error, requests.exceptions.HTTPError)
        and response is not None
        and response.status_code in (403, 404)
    )


def default_client() -> KaggleDatasetClient:
    """The real API client; raises a helpful error when kaggle is missing."""
    return KaggleApiAdapter()


def publish_on_exit(
    dataset_root: str | Path,
    *,
    dataset: str,
    client: KaggleDatasetClient,
    version_notes: str | None = None,
    log: Callable[[str], None] | None = None,
    log_path: str | Path | None = None,
) -> RemotePublishResult:
    """Push the local handoff dataset to Kaggle; never raises.

    Called on the session exit path (normal return or deadline stop) after
    the local per-seed handoff has run. The pushed folder is
    ``<dataset_root>/<handoff>/`` (zero-copy) with its metadata file
    ensured; the outcome is written beside the dataset root for the
    artifact trail.
    """
    logger = log if log is not None else (lambda message: None)
    folder = Path(dataset_root) / _handoff_dirname()
    seeds = _manifest_seeds(folder) if folder.is_dir() else []
    if version_notes is None:
        version_notes = f"session publish: {len(seeds)} seed version(s)"
    if not seeds:
        result = RemotePublishResult(
            dataset=dataset,
            folder=str(folder),
            seeds_published=0,
            status=STATUS_NOTHING,
            version_notes=version_notes,
        )
        logger(f"remote publish skipped: no complete seed versions under {folder}")
        _write_log(log_path, dataset_root, result)
        return result
    ensure_metadata(folder, dataset)
    try:
        client.create_version(folder, version_notes)
        result = RemotePublishResult(
            dataset=dataset,
            folder=str(folder),
            seeds_published=len(seeds),
            status=STATUS_PUBLISHED,
            version_notes=version_notes,
        )
        logger(
            f"remote publish: {dataset} version pushed "
            f"({len(seeds)} seed version(s))"  # type: ignore[decimal]
        )
    except Exception as error:  # noqa: BLE001 - the exit path never raises
        result = RemotePublishResult(
            dataset=dataset,
            folder=str(folder),
            seeds_published=len(seeds),
            status=STATUS_FAILED,
            version_notes=version_notes,
            error=f"{type(error).__name__}: {error}",
        )
        logger(f"remote publish FAILED (local handoff intact): {result.error}")
    _write_log(log_path, dataset_root, result)
    return result


def _write_log(
    log_path: str | Path | None,
    dataset_root: str | Path,
    result: RemotePublishResult,
) -> None:
    """Record the publish outcome beside the dataset root (best-effort)."""
    target = Path(log_path) if log_path is not None else (
        Path(dataset_root) / PUBLISH_LOG_NAME
    )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
    except OSError:
        pass  # logging must never break the exit path


def restore_latest(
    dataset_root: str | Path,
    *,
    dataset: str,
    client: KaggleDatasetClient,
    log: Callable[[str], None] | None = None,
) -> list[int]:
    """Restore the newest remote version's seeds into the local handoff tree.

    Downloads the dataset, keeps every extracted seed directory that carries
    a complete manifest (a manifest-less directory is a partial version and
    is ignored, mirroring the handoff's own atomicity rule), and merges it
    into the local tree: remote replaces matching local seeds, local-only
    seeds are kept. Raises on client errors - a session that asked for
    remote state and cannot get it must fail before training starts.
    Returns the restored seed numbers.
    """
    logger = log if log is not None else (lambda message: None)
    handoff_name = _handoff_dirname()
    local = Path(dataset_root) / handoff_name
    with tempfile.TemporaryDirectory(prefix="act-kaggle-restore-") as scratch:
        extracted = Path(scratch) / "download"
        try:
            client.download_files(dataset, extracted)
        except Exception as error:
            if not _is_missing_dataset_error(error):
                raise
            # No remote version yet (session 1, or a dataset not created):
            # training from scratch is always safe under the atomic-pair
            # rule, so absence warns instead of blocking the launch.
            logger(
                f"no remote version of {dataset} available yet; "
                "training from scratch (atomic pairs make this safe)"
            )
            return []
        source = _locate_handoff_tree(extracted, handoff_name)
        if source is None:
            # Defense in depth: if the service ever stops auto-extracting
            # uploaded archives, the blobs arrive as files - extract them
            # ourselves into the expected tree.
            source = _extract_archive_blobs(extracted)
        if source is None:
            logger(f"remote dataset {dataset} carries no seed versions; nothing restored")
            return []
        restored: list[int] = []
        for seed_dir in _manifest_seeds(source):
            seed = int(seed_dir.name.split("-")[-1])
            target = local / seed_dir.name
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(seed_dir, target)
            restored.append(seed)
    restored.sort()
    logger(
        f"restored {len(restored)} seed version(s) from {dataset}: "
        f"{restored if restored else 'none'}"
    )
    return restored


def _extract_archive_blobs(extracted: Path) -> Path | None:
    """Extract any tar/zip blobs at the archive root into ``extracted``.

    Only used when no seed tree is found already-extracted (the service
    currently unpacks archives server-side). Returns the located tree, or
    None when no archives yield seed versions.
    """
    import tarfile
    import zipfile

    for blob in sorted(extracted.glob("*.tar")) + sorted(extracted.glob("*.zip")):
        if blob.name.endswith(".zip"):
            with zipfile.ZipFile(blob) as archive:
                archive.extractall(extracted)
        else:
            with tarfile.open(blob) as archive:
                archive.extractall(extracted, filter="data")
    return _locate_handoff_tree(extracted, _handoff_dirname().split("/")[-1])


def _locate_handoff_tree(extracted: Path, handoff_name: str) -> Path | None:
    """Find the extracted tree holding the seed directories.

    Layouts seen in the wild, all accepted: the archive flattened to its
    payload (seed directories at the root), the pushed folder name preserved
    (``<dirname>/seed-*``), or the full handoff path nested under the
    archive root (``<handoff>/seed-*``).
    """
    candidates = [extracted / handoff_name, extracted]
    if extracted.is_dir():
        candidates.extend(path for path in extracted.iterdir() if path.is_dir())
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_dir():
            continue
        seen.add(candidate)
        if _manifest_seeds(candidate):
            return candidate
    return None
