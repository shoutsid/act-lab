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
  hours of completed local training into a crashed session. It retries a
  bounded number of times first, because this push is the last step before the
  container (and with it every local pair) is gone - there is no next session
  that could re-push a *container's* handoff, only one that starts again from
  the last published version.
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
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol

from .kaggle import _sha256  # shared checksum helper (same contract)

KAGGLE_AVAILABLE = importlib.util.find_spec("kaggle") is not None

METADATA_NAME = "dataset-metadata.json"
DEFAULT_LICENSE = "CC0-1.0"
PUBLISH_LOG_NAME = "remote-publish-log.json"

# How many times the exit publish re-attempts its push, and the base delay
# between attempts (multiplied by the attempt number). The push is the only
# durable copy of a session's pairs - the handoff roots live in the container
# overlay, which ends with the kernel - so it is worth a few retries against a
# transient API or network error, which is what a failed push usually is.
PUBLISH_ATTEMPTS = 3
PUBLISH_RETRY_SECONDS = 5.0

# How many times the startup restore re-attempts a download whose failure looks
# like "no such dataset yet", and the base delay (multiplied by the attempt
# number). The download endpoint answers 403 both for a dataset that does not
# exist and for one it will not serve, and a version pushed minutes ago can be
# in that second state while the service finishes indexing it - measured
# 2026-09-27, a pushed version's files appear in the listing 5-8 min after the
# push (a 03:27:40Z version carried 03:32-03:34 file times). So a single 403 is
# not enough to conclude that the queue has published nothing, and the wait is
# cheap against a session window: 60 + 120 + 180 s is 6 min of a 7 h window.
RESTORE_ATTEMPTS = 4
RESTORE_RETRY_SECONDS = 60.0

# How many pages of the owner's own dataset listing the existence check reads
# before giving up. The listing pages at 20 datasets and the campaign's account
# holds a handful, so this is slack rather than a limit anyone reaches; the cap
# exists so a misbehaving listing cannot make a session's startup unbounded.
DATASET_LIST_PAGES = 5

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


def _seed_dirname(seed: int) -> str:
    """The handoff's own name for a seed directory, imported lazily (cycle).

    The name does not depend on the root, which is the point of asking for it
    rather than formatting it here: a restore that spelled the name itself could
    place a seed somewhere the handoff's own readers never look, and the seed
    would read as missing and be trained again.
    """
    from act.training.v0_handoff import seed_directory

    return seed_directory(Path("."), seed).name


def published_seed_dirs(extracted: str | Path) -> dict[int, Path]:
    """Every complete seed version anywhere under an extracted dataset.

    Found by search rather than by locating one tree and globbing its children.
    Measured 2026-09-21 by the startup-restore probe: the published dataset's
    seeds sit one level *deeper* than the handoff layout names them
    (``seed-15000/seed-15000/checkpoints/...``), because each seed is uploaded
    as ``seed-NNNN.tar`` whose members already carry the ``seed-NNNN/`` prefix
    and the service keeps both. So the handoff parent's children have no
    ``manifests/manifest.json`` at all, ``_locate_handoff_tree`` fell through to
    matching a single *seed* directory, and the probe restored **1 of the 4**
    published seeds. A session that treats three completed seeds as missing
    re-trains them at ~1.5 h each, and the queue still reports 4/12 either way.

    A manifest-less directory is a partial publish and is not a version, and
    where a seed matches at more than one depth the shallowest match wins (a
    seed nested inside another seed's directory must not shadow the real one).
    """
    found: dict[int, Path] = {}
    root = Path(extracted)
    if not root.is_dir():
        return found
    for manifest in root.rglob("manifest.json"):
        if manifest.parent.name != "manifests":
            continue
        seed_dir = manifest.parent.parent
        stem = seed_dir.name.split("-")[-1]
        if not seed_dir.name.startswith("seed-") or not stem.isdigit():
            continue
        seed = int(stem)
        existing = found.get(seed)
        if existing is None or len(seed_dir.parts) < len(existing.parts):
            found[seed] = seed_dir
    return found


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

    def dataset_exists(self, dataset: str) -> bool:
        """Whether the authenticated account owns a dataset with this id."""
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
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as error:
            # The name resolving is not the API being callable. Measured
            # 2026-09-30 on the TPU image: `find_spec("kaggle")` succeeds while
            # this import raises `No module named 'kaggle.api'`, so the run died
            # at restore before a token was trained - the null queue's first
            # fire, `errors: ["restore"]` - and the message named no fix. The
            # import is the only check that tells a present package from a
            # broken one, so it is the check that gets to report.
            raise RuntimeError(
                "the kaggle package is installed but its dataset API cannot be "
                f"imported ({error}); on a Kaggle image this is a broken package "
                "release, and `pip install -U kaggle` replaces it"
            ) from error

        self._api = KaggleApi()
        self._api.authenticate()

    def create_version(self, folder: Path, version_notes: str) -> str:
        folder = Path(folder)
        with tempfile.TemporaryDirectory(prefix="act-kaggle-push-") as scratch:
            staging = Path(scratch) / "push"
            staging.mkdir()
            stage_push(folder, staging)
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

    def dataset_exists(self, dataset: str) -> bool:
        """Is this dataset one of the account's own? Asked of the owner's listing.

        The download endpoint answers 403 both for "no such dataset" and for a
        dataset it will not serve, so the error alone cannot tell a first
        session (nothing published yet) from a session whose restore is broken.
        The owner's own listing can: a dataset the account published is always
        listed back to it, and ``mine=True`` is the only listing a *private*
        dataset appears in - which every queue's dataset is.

        Raises when the listing itself fails, which is the fail-fast direction:
        a session that cannot establish whether remote state exists must not
        start training over it. The page walk is capped so a listing that never
        ends cannot make a session's startup unbounded.
        """
        owner, _, slug = dataset.partition("/")
        wanted = f"{owner}/{slug}" if owner and slug else dataset
        for page in range(1, DATASET_LIST_PAGES + 1):
            listing = self._api.dataset_list(mine=True, page=page)
            if not listing:
                return False
            if any(getattr(item, "ref", None) == wanted for item in listing):
                return True
        return False


def stage_push(folder: str | Path, staging: str | Path) -> None:
    """Stage the pushed folder: one tar per seed dir, plain files copied flat.

    This is the upload layout the live probes established (see the adapter's
    docstring): the service extracts uploaded archives and keeps their
    *internal* paths, so each seed is archived under ``seed-NNNN/...``. Get
    the prefix wrong and every seed's files land at the archive root and
    collide - a silently corrupt dataset that still reports as published.
    Module-level and free of the API client so that behaviour is testable
    without authenticating.
    """
    source = Path(folder)
    target = Path(staging)
    target.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source.iterdir()):
        if entry.name == METADATA_NAME:
            continue  # re-copied last, from its canonical source
        if entry.is_dir():
            archive = target / f"{entry.name}.tar"
            with tarfile.open(archive, "w") as tar:
                for path in sorted(entry.rglob("*")):
                    if path.is_file():
                        tar.add(
                            path,
                            arcname=str(Path(entry.name) / path.relative_to(entry)),
                        )
        elif entry.is_file():
            shutil.copy2(entry, target / entry.name)
    metadata = source / METADATA_NAME
    if metadata.is_file():
        shutil.copy2(metadata, target / METADATA_NAME)


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
    attempts: int = PUBLISH_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> RemotePublishResult:
    """Push the local handoff dataset to Kaggle; never raises.

    Called on the session exit path (normal return or deadline stop) after
    the local per-seed handoff has run. The pushed folder is
    ``<dataset_root>/<handoff>/`` (zero-copy) with its metadata file
    ensured; the outcome is written beside the dataset root for the
    artifact trail.

    This push is the only durable copy of the session's work: the handoff
    roots live in the container overlay, which goes away with the kernel, so a
    transient API or network error here costs every pair the session trained.
    It is therefore retried a bounded number of times before being recorded as
    a failure - and the metadata write, which writes *into* the handoff folder
    and so can fail on a full disk, is inside the same guard rather than
excepted from it.
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

    failures: list[str] = []
    for attempt in range(1, max(1, attempts) + 1):
        try:
            ensure_metadata(folder, dataset)
            client.create_version(folder, version_notes)
        except Exception as error:  # noqa: BLE001 - the exit path never raises
            failures.append(f"{type(error).__name__}: {error}")
            logger(
                f"remote publish attempt {attempt}/{attempts} failed "
                f"({failures[-1]})"
            )
            if attempt < attempts:
                logger("  retrying: a failed push here loses the session's pairs")
                sleep(PUBLISH_RETRY_SECONDS * attempt)
            continue
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
            + (f" on attempt {attempt}" if attempt > 1 else "")
        )
        _write_log(log_path, dataset_root, result)
        return result

    result = RemotePublishResult(
        dataset=dataset,
        folder=str(folder),
        seeds_published=len(seeds),
        status=STATUS_FAILED,
        version_notes=version_notes,
        error="; ".join(failures),
    )
    logger(
        f"remote publish FAILED after {attempts} attempt(s) (local handoff "
        f"intact, but this container ends with the session): {result.error}"
    )
    _write_log(log_path, dataset_root, result)
    return result


def _tree_gib(root: Path) -> float:
    """Bytes under ``root`` in GiB, so a logged download cost is comparable."""
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total / 2**30


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
    attempts: int = RESTORE_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> list[int]:
    """Restore the newest remote version's seeds into the local handoff tree.

    Downloads the dataset, keeps every extracted seed directory that carries
    a complete manifest (a manifest-less directory is a partial version and
    is ignored, mirroring the handoff's own atomicity rule), and merges it
    into the local tree: remote replaces matching local seeds, local-only
    seeds are kept. Raises on client errors - a session that asked for
    remote state and cannot get it must fail before training starts.
    Returns the restored seed numbers.

    A download failure that *looks* like "no such dataset yet" is classified by
    asking the owner's own listing rather than by trusting the error: the
    download endpoint answers 403 both for a dataset that does not exist and for
    one it will not serve, and the second case must not be read as the first.
    Measured 2026-09-26/27, that misreading cost a whole session's quota at a
    time and, worse, published a re-trained seed *over* its own complete version:
    the sessions restored nothing, trained seeds 15024+ from scratch and
    republished them, so the queue never advanced past the seeds already done
    while the driver read the same partial progress each time. A missing dataset
    (session 1) is still the normal bootstrap state and is not retried; an
    existing one is retried, and then raises rather than training over it.

    The download's size and duration are logged because this cost is *unpriced*
    and grows with every seed the queue publishes: it is the whole published
    dataset (~22.7 GiB at four seeds, ~6.4 GiB per seed added) downloaded before
    a token is trained, so a session's startup gets more expensive for as long as
    the queue runs. Measured 2026-09-21: a session ended ERROR after 1.32 h with
    no log and no output, and this startup cost is one of the two things known to
    occupy a session for that long - so the next session's log has to carry the
    number rather than an expectation.
    """
    logger = log if log is not None else (lambda message: None)
    handoff_name = _handoff_dirname()
    local = Path(dataset_root) / handoff_name
    with tempfile.TemporaryDirectory(prefix="act-kaggle-restore-") as scratch:
        extracted = Path(scratch) / "download"
        download_started = time.monotonic()
        failure: Exception | None = None
        for attempt in range(1, max(1, attempts) + 1):
            try:
                client.download_files(dataset, extracted)
                failure = None
                break
            except Exception as error:  # noqa: BLE001 - classified below
                if not _is_missing_dataset_error(error):
                    raise
                failure = error
                # The error type alone cannot say whether the dataset exists,
                # and the two answers need opposite responses: a dataset that
                # has no version yet is the normal first-session state (train
                # from scratch), while one that exists and will not download
                # must stop the session before it re-trains published seeds.
                if not client.dataset_exists(dataset):
                    logger(
                        f"no remote version of {dataset} available yet; "
                        "training from scratch (atomic pairs make this safe)"
                    )
                    return []
                if attempt < attempts:
                    logger(
                        f"the restore of {dataset} failed ({type(error).__name__}: "
                        f"{error}) although the dataset exists; attempt "
                        f"{attempt}/{attempts}, retrying in "
                        f"{RESTORE_RETRY_SECONDS * attempt:.0f} s (published seeds "
                        "are not re-trained over)"
                    )
                    sleep(RESTORE_RETRY_SECONDS * attempt)
        if failure is not None:
            # Every attempt failed on a dataset the account owns: this is not
            # "nothing published yet", and training from scratch here would
            # re-train - and republish over - the seeds the queue already has.
            raise RuntimeError(
                f"cannot restore {dataset}: the dataset exists but its download "
                f"failed on all {attempts} attempt(s) "
                f"({type(failure).__name__}: {failure}); training nothing rather "
                "than re-training seeds the queue already published"
            ) from failure
        logger(
            f"remote restore: {dataset} downloaded in "
            f"{(time.monotonic() - download_started) / 60.0:.1f} min "
            f"({_tree_gib(extracted):.2f} GiB on disk)"
        )
        seeds = published_seed_dirs(extracted)
        if not seeds:
            # Defense in depth: if the service ever stops auto-extracting
            # uploaded archives, the blobs arrive as files - extract them
            # ourselves, then look again.
            _extract_archive_blobs(extracted)
            seeds = published_seed_dirs(extracted)
        if not seeds:
            logger(f"remote dataset {dataset} carries no seed versions; nothing restored")
            return []
        restored: list[int] = []
        for seed in sorted(seeds):
            # Named by the handoff, not by whatever the extraction called it, so
            # the local tree the handoff's own readers expect is what comes out.
            target = local / _seed_dirname(seed)
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(seeds[seed], target)
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
