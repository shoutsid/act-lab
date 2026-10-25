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
- **Durable per pair, not only per session.** ``publish_handoff`` runs after
  every pair the session completes, not only on the exit path, and skips a
  state it has already pushed. Measured 2026-10-05: four sessions in a row
  ended ERROR inside the *exit* upload - the live log stopped at ``session
  done`` in each one, with no attempt, receipt or retry line from the publish
  path ever flushed - and each one discarded the three pairs it had just
  trained, because until that exit push finished nothing was durable. A pair
  that reaches the dataset mid-session is progress the next session skips
  instead of re-training it.
- **No hard kaggle dependency.** The package is optional (preinstalled on
  Kaggle images, absent elsewhere); the module imports cleanly without it
  and every testable decision is exercised against an injected fake client.

The merge rule for restore: a remote seed version replaces a matching local
one, and local-only seeds are kept (a locally-published seed that postdates
the remote snapshot is never rolled back).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import tarfile
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

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

#: HTTP statuses worth another upload. A 4xx that is not in this set is a
#: refusal of the request itself - a rejected version, a quota, a bad payload -
#: and re-uploading the same gigabytes cannot change it. Measured 2026-10-25:
#: three 25-32 GiB retries per pair (nine uploads, ~45 min) were spent against
#: a 400 that was the account's private-dataset quota, and the session lost all
#: three pairs anyway.
PUBLISH_RETRY_STATUSES = frozenset({408, 409, 425, 429})

# How often a push that is still uploading says so. Measured 2026-10-06: the
# two captures of a session killed during a push each stop on the kaggle CLI's
# version warning - the first line it prints - so the only account of the
# failure is a start with no size, no attempt and no elapsed time beside it.
# The push is a single blocking client call that prints nothing while it runs,
# so without a heartbeat a kill sixteen minutes in and a kill one minute in
# leave the same record. This is the line that tells them apart.
PUBLISH_HEARTBEAT_SECONDS = 60.0


@contextmanager
def _publish_heartbeat(
    logger: Callable[[str], None],
    *,
    seeds: int,
    size_gib: float,
    started: float,
    interval: float = PUBLISH_HEARTBEAT_SECONDS,
):
    """Log that a push is still running, every ``interval`` seconds.

    A context manager so the beat stops on the way out whether the push
    returned or raised: a thread left printing after the result is written
    would put lines in the capture that belong to no attempt.
    """
    import threading

    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(interval):
            elapsed = (time.monotonic() - started) / 60.0
            logger(
                f"remote publish: still pushing {seeds} seed version(s) "
                f"({size_gib:.1f} GiB) after {elapsed:.1f} min"
            )

    thread = threading.Thread(target=beat, name="act-publish-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)


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

# How many files the per-file fallback asks for per page. The API accepts 200,
# and the campaign's own datasets carry tens of files (six seeds is 30), so one
# page is the normal case; the loop exists so a bigger block cannot be silently
# truncated.
DATASET_FILE_PAGE_SIZE = 200

STATUS_PUBLISHED = "published"
#: The folder's state is already in the dataset: a previous push of this exact
#: fingerprint succeeded, so this call uploaded nothing.
STATUS_REUSED = "already_published"
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


def _seed_number(path: Path) -> int | None:
    """The seed a handoff directory names (``seed-15031`` -> 15031).

    ``None`` for a ``seed-*`` directory that is not a seed (a scratch name a
    handoff cleanup left behind): a shard filter must skip it rather than turn
    one stray directory into a crash on the publish path.
    """
    try:
        return int(path.name.split("-", 1)[1])
    except (IndexError, ValueError):
        return None


def shard_dataset(base: str, index: int) -> str:
    """The dataset id that carries shard ``index`` of a queue's handoff.

    Shard 0 keeps the registered id, so the version already published stays
    exactly where the record names it. Later shards are derived from it rather
    than chosen per run: a reader can compute every name from the queue.
    """
    return base if index == 0 else f"{base}-p{index + 1}"


def shard_index(seed: int, seeds: Sequence[int], shard_seeds: int) -> int:
    """Which shard a seed belongs to, by its place in the declared queue.

    Membership is positional and declared, not measured: a rule that grouped
    seeds by their actual checkpoint sizes would make a dataset's contents
    depend on the run that produced them, and a seed could move between
    shards after the fact. ``shard_seeds <= 0`` means no sharding.
    """
    if shard_seeds <= 0:
        return 0
    return seeds.index(seed) // shard_seeds


def shard_members(
    seeds: Sequence[int], shard_seeds: int
) -> list[tuple[int, tuple[int, ...]]]:
    """``(index, seed set)`` for every shard of a declared queue, in order."""
    if shard_seeds <= 0 or shard_seeds >= len(seeds):
        return [(0, tuple(seeds))]
    return [
        (index, tuple(seeds[start : start + shard_seeds]))
        for index, start in enumerate(range(0, len(seeds), shard_seeds))
    ]


def shard_datasets(
    base: str, seeds: Sequence[int], shard_seeds: int
) -> list[tuple[str, tuple[int, ...]]]:
    """Every ``(dataset id, its seeds)`` for a queue, in shard order."""
    return [
        (shard_dataset(base, index), members)
        for index, members in shard_members(seeds, shard_seeds)
    ]


def shard_for_seed(
    base: str, seed: int, seeds: Sequence[int], shard_seeds: int
) -> tuple[str, tuple[int, ...]]:
    """The ``(dataset id, seed set)`` a seed's push belongs to."""
    index = shard_index(seed, seeds, shard_seeds)
    members = shard_members(seeds, shard_seeds)[index][1]
    return shard_dataset(base, index), members


def handoff_fingerprint(
    folder: str | Path, *, seeds: set[int] | None = None
) -> str:
    """A digest of the complete seed versions a push of ``folder`` would carry.

    Each seed's manifest is written last, after its payloads, and checksums them
    (see ``act.training.v0_handoff``: staging, then a manifest baked from the
    checksums, then an atomic swap). So the *set* of manifests is the published
    state: a folder whose fingerprint matches the last successful push has
    nothing new to upload, and re-uploading it is the only thing standing
    between a finished session and its exit.

    The seed's directory name is hashed alongside its manifest digest so that a
    seed added, removed or renamed changes the fingerprint even when every
    manifest that remains is byte-identical.
    """
    digest = hashlib.sha256()
    for seed_dir in _manifest_seeds(Path(folder)):
        if seeds is not None and _seed_number(seed_dir) not in seeds:
            continue
        digest.update(seed_dir.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(seed_dir / "manifests" / "manifest.json").encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


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
    """Outcome of one publish attempt."""

    dataset: str
    folder: str
    seeds_published: int
    status: str
    version_notes: str
    error: str | None = None
    #: Digest of the seed versions this push carried (or, for ``reused``, of the
    #: versions an earlier push carried). What makes a redundant push
    #: detectable: a later call whose folder fingerprints the same has nothing
    #: new to upload. Empty for a state with no seeds to describe.
    fingerprint: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class KaggleDatasetClient(Protocol):
    """The slice of the Kaggle API this wrapper needs; injectable for tests."""

    def create_version(
        self,
        folder: Path,
        version_notes: str,
        *,
        seeds: Sequence[int] | None = None,
    ) -> str:
        """Push ``folder`` as a new dataset version; return a version ref.

        ``seeds`` restricts the upload to one handoff shard. ``None`` pushes
        the whole folder, which is the single-dataset behaviour every caller
        had before sharding existed.
        """
        ...

    def download_files(self, dataset: str, target: Path) -> None:
        """Download and extract the dataset's latest version into ``target``."""
        ...

    def list_files(self, dataset: str) -> list[str]:
        """Every file name in the dataset's latest version, as the upload named it."""
        ...

    def download_file(self, dataset: str, file_name: str, target: Path) -> Path:
        """Download one named file into ``target``, keeping its remote path.

        Returns the path written. This exists because a caller can need
        kilobytes out of a dataset that is tens of gigabytes: the Gate-4
        verdict reads one ~5 KB ``reports/seed-summary.json`` per seed and never
        opens a checkpoint, while each queue's published version carries every
        checkpoint the queue has ever trained (measured 2026-10-24: ~190 GiB
        across the three queues, against a container writable layer that died at
        95.0 GiB on 2026-10-19).
        """
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

    def create_version(
        self,
        folder: Path,
        version_notes: str,
        *,
        seeds: Sequence[int] | None = None,
    ) -> str:
        folder = Path(folder)
        with tempfile.TemporaryDirectory(prefix="act-kaggle-push-") as scratch:
            staging = Path(scratch) / "push"
            staging.mkdir()
            stage_push(folder, staging, seeds=seeds)
            try:
                self._api.dataset_create_version(
                    str(staging), version_notes, dir_mode="skip", quiet=True
                )
            except Exception as error:  # noqa: BLE001 - API error types vary
                if not _is_missing_dataset_error(error):
                    detail = _response_detail(error)
                    if detail and detail not in str(error):
                        # Keep the type and the response: ``restore_latest`` and
                        # the publish retry both classify on them.
                        error.args = (
                            f"{error} - response body: {detail}",
                        )
                    raise
                # First publish: the dataset does not exist yet.
                self._api.dataset_create_new(
                    str(staging), dir_mode="skip", quiet=True
                )
        return version_notes

    def download_files(self, dataset: str, target: Path) -> None:
        target = Path(target)
        try:
            self._api.dataset_download_files(dataset, path=str(target), unzip=True)
            return
        except Exception as error:  # noqa: BLE001 - classified below
            if not _is_missing_dataset_error(error):
                raise
            archive_error = error
        # The whole-archive endpoint can fail for a version whose files are all
        # individually downloadable. Measured 2026-10-03 on `act-v0-5-2`: every
        # archive request answered 404 and the dataset's listing reported
        # `total_bytes: 0`, while each file streamed normally - so the archive
        # *bundle* was broken, not the dataset. The session's restore is the only
        # caller, and without this it cannot start: it stops before training
        # rather than re-training published seeds, which is what every session of
        # the 2026-10-03 GPU week did.
        try:
            self._download_each_file(dataset, target)
        except Exception as per_file_error:  # noqa: BLE001 - reported, not swallowed
            # Keep the archive error's own shape: `restore_latest` classifies a
            # 403/404 to tell "no such dataset yet" from "will not serve", and
            # that reading has to survive the fallback failing too.
            raise archive_error from per_file_error

    def list_files(self, dataset: str) -> list[str]:
        """Every file name in the dataset's latest version, following page tokens."""
        names: list[str] = []
        page_token: str | None = None
        while True:
            response = self._api.dataset_list_files(
                dataset, page_token=page_token, page_size=DATASET_FILE_PAGE_SIZE
            )
            names.extend(str(entry.name) for entry in response.dataset_files)
            page_token = getattr(response, "next_page_token", None) or None
            if page_token is None:
                return names

    def download_file(self, dataset: str, file_name: str, target: Path) -> Path:
        """Download one named file under ``target``, preserving its remote path.

        Each file gets its **own** scratch directory because the API saves it
        under its basename and a dataset has many ``seed-summary.json`` /
        ``manifest.json`` files: a shared directory would let one seed's file
        stand in for the next seed's (the second request would read the first
        file as an already-downloaded one and skip it).
        """
        destination = Path(target) / file_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="act-kaggle-file-") as scratch:
            self._api.dataset_download_file(
                dataset, file_name, path=scratch, quiet=True
            )
            # Where it lands depends on the client version, both measured here:
            # 2.2.5 preserves the remote path (`<scratch>/seed-N/.../summary.json`),
            # while 2.0.2 - what the kernels run - writes the basename.
            saved = Path(scratch) / file_name
            if not saved.is_file():
                saved = Path(scratch) / Path(file_name).name
            if not saved.is_file():
                raise RuntimeError(
                    f"{dataset}: {file_name} did not arrive under {scratch}"
                )
            shutil.move(str(saved), str(destination))
        return destination

    def _download_each_file(self, dataset: str, target: Path) -> None:
        """Fetch a dataset file by file, for a version whose archive will not serve.

        The startup restore needs the same tree the archive would have unpacked,
        so every file the listing names is fetched and moved to that exact
        relative path.
        """
        target.mkdir(parents=True, exist_ok=True)
        names = self.list_files(dataset)
        if not names:
            raise RuntimeError(f"{dataset} lists no files to download")
        for name in names:
            self.download_file(dataset, name, target)

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


def stage_push(
    folder: str | Path,
    staging: str | Path,
    *,
    seeds: Sequence[int] | None = None,
) -> None:
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
    wanted = None if seeds is None else {_seed_dirname(seed) for seed in seeds}
    for entry in sorted(source.iterdir()):
        if entry.name == METADATA_NAME:
            continue  # re-copied last, from its canonical source
        if wanted is not None and entry.is_dir() and entry.name not in wanted:
            # A shard's version must carry only its own seeds: the point of the
            # split is that no version approaches the measured payload wall.
            continue
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


def _response_detail(error: Exception, *, limit: int = 500) -> str | None:
    """The API's own explanation of a rejected request, when it carried one.

    ``requests.HTTPError`` knows only the status line; the body is what names
    the reason. The 2026-10-25 publish failure was a bare ``400 Bad Request``
    recorded nine times while the server's answer - the private-dataset quota -
    was discarded, so the wall had to be inferred from a file-size sum instead
    of read. Truncated because a proxy's error page can be large.
    """
    response = getattr(error, "response", None)
    text = getattr(response, "text", None)
    if not text:
        return None
    compact = " ".join(str(text).split())
    if not compact:
        return None
    if len(compact) > limit:
        compact = compact[: limit - 1] + "\u2026"
    return compact


def _is_retryable_publish_error(error: Exception) -> bool:
    """Whether re-uploading could plausibly succeed.

    No status at all means a network/transport error and is retried, as before.
    A status outside :data:`PUBLISH_RETRY_STATUSES` below 500 is a refusal of
    the request, so the next attempt would upload the same bytes to the same
    answer.
    """
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if not isinstance(status, int):
        return True
    if status >= 500:
        return True
    return status in PUBLISH_RETRY_STATUSES


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


def publish_handoff(
    dataset_root: str | Path,
    *,
    dataset: str,
    client: KaggleDatasetClient,
    version_notes: str | None = None,
    log: Callable[[str], None] | None = None,
    log_path: str | Path | None = None,
    attempts: int = PUBLISH_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
    seeds: Sequence[int] | None = None,
) -> RemotePublishResult:
    """Push the local handoff dataset to Kaggle; never raises.

    Called after every pair the session completes (through
    ``run_pair_session``'s ``remote_publish`` hook, which is what makes a
    session's progress durable *during* the session) and again on the exit path
    (:func:`publish_on_exit`). The pushed folder is
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

    A state already in the dataset is not uploaded again: when the recorded log
    shows a successful push of this folder's fingerprint, this returns
    ``STATUS_REUSED`` without calling the API. That is what stops a session
    from spending its final ~40 minutes re-uploading ~38 GiB it pushed after
    the last pair - the window in which four sessions in a row were killed on
    2026-10-05, losing every pair they had completed.

    ``seeds`` restricts the push to one declared shard of the handoff, which is
    how a queue publishes without ever approaching the measured per-version
    payload wall (a 7-seed / 44.13 GiB version landed 2026-10-05; an 8-seed /
    ~50.4 GiB one killed two containers). ``None`` pushes the whole folder,
    including its fingerprint and log format, exactly as before sharding.
    """
    logger = log if log is not None else (lambda message: None)
    folder = Path(dataset_root) / _handoff_dirname()
    wanted = None if seeds is None else set(seeds)
    seed_dirs = [
        path
        for path in (_manifest_seeds(folder) if folder.is_dir() else [])
        if wanted is None or _seed_number(path) in wanted
    ]
    if version_notes is None:
        version_notes = f"session publish: {len(seed_dirs)} seed version(s)"
    if not seed_dirs:
        result = RemotePublishResult(
            dataset=dataset,
            folder=str(folder),
            seeds_published=0,
            status=STATUS_NOTHING,
            version_notes=version_notes,
        )
        logger(f"remote publish skipped: no complete seed versions under {folder}")
        _write_log(log_path, dataset_root, result, sharded=wanted is not None)
        return result

    fingerprint = handoff_fingerprint(folder, seeds=wanted)
    size_gib = sum(_tree_gib(path) for path in seed_dirs)
    previous_all = _read_log(log_path, dataset_root)
    previous = _shard_record(previous_all, dataset, sharded=wanted is not None)
    if (
        previous is not None
        and previous.get("fingerprint") == fingerprint
        and previous.get("status") in (STATUS_PUBLISHED, STATUS_REUSED)
    ):
        # The last push that succeeded carried exactly these seed versions, so
        # the dataset already holds this state. Re-uploading it would be the
        # only work left in the session, and it is the work that has been
        # getting killed - the log entry below is what the exit path and the
        # driver read, so it is rewritten for *this* call rather than left as
        # the pair-time push's.
        result = RemotePublishResult(
            dataset=dataset,
            folder=str(folder),
            seeds_published=len(seed_dirs),
            status=STATUS_REUSED,
            version_notes=str(previous.get("version_notes") or version_notes),
            fingerprint=fingerprint,
        )
        logger(
            f"remote publish skipped: {len(seed_dirs)} seed version(s) are already "
            f"published as this state ({fingerprint[:12]}); re-uploading them "
            "would spend the session's last minutes on a copy that exists"
        )
        _write_log(log_path, dataset_root, result, sharded=wanted is not None)
        return result

    failures: list[str] = []
    started = time.monotonic()
    logger(
        f"remote publish: pushing {len(seed_dirs)} seed version(s), "
        f"{size_gib:.1f} GiB of handoff"
    )
    for attempt in range(1, max(1, attempts) + 1):
        attempt_started = time.monotonic()
        logger(
            f"remote publish: attempt {attempt}/{attempts} starting "
            f"({len(seed_dirs)} seed version(s), {size_gib:.1f} GiB)"
        )
        try:
            ensure_metadata(folder, dataset)
            with _publish_heartbeat(
                logger,
                seeds=len(seed_dirs),
                size_gib=size_gib,
                started=attempt_started,
            ):
                client.create_version(folder, version_notes, seeds=seeds)
        except Exception as error:  # noqa: BLE001 - a push must never end the session
            failures.append(f"{type(error).__name__}: {error}")
            logger(
                f"remote publish attempt {attempt}/{attempts} failed "
                f"({failures[-1]})"
            )
            if not _is_retryable_publish_error(error):
                logger(
                    "  not retrying: the API refused this version rather than "
                    "failing transiently; re-uploading the same bytes cannot "
                    "change the answer. Read the response beside the error."
                )
                break
            if attempt < attempts:
                logger("  retrying: a failed push here loses the session's pairs")
                sleep(PUBLISH_RETRY_SECONDS * attempt)
            continue
        result = RemotePublishResult(
            dataset=dataset,
            folder=str(folder),
            seeds_published=len(seed_dirs),
            status=STATUS_PUBLISHED,
            version_notes=version_notes,
            fingerprint=fingerprint,
        )
        # The duration is part of the account, not decoration: this push is
        # unpriced in the session plan and grows with every seed the queue
        # publishes, so a session that pushes after each pair has to be able to
        # read what that cost it (the same reason the restore prices itself).
        logger(
            f"remote publish: {dataset} version pushed "
            f"({len(seed_dirs)} seed version(s)) in "  # type: ignore[decimal]
            f"{(time.monotonic() - started) / 60.0:.1f} min"
            + (f" on attempt {attempt}" if attempt > 1 else "")
        )
        _write_log(log_path, dataset_root, result, sharded=wanted is not None)
        return result

    result = RemotePublishResult(
        dataset=dataset,
        folder=str(folder),
        seeds_published=len(seed_dirs),
        status=STATUS_FAILED,
        version_notes=version_notes,
        error="; ".join(failures),
        fingerprint=fingerprint,
    )
    logger(
        f"remote publish FAILED after {attempts} attempt(s) (local handoff "
        f"intact, but this container ends with the session): {result.error}"
    )
    _write_log(log_path, dataset_root, result, sharded=wanted is not None)
    return result


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
    seeds: Sequence[int] | None = None,
) -> RemotePublishResult:
    """The exit-path entry point for :func:`publish_handoff`.

    Named apart because the exit path is the one the driver reads: it prints
    this call's recorded outcome as ``exit publish: <status>`` and stops the
    campaign when that status is ``failed``. The session publishes after every
    pair first, so by the time this runs the state is usually already in the
    dataset and the call returns ``already_published`` without uploading
    anything - which is the point: the exit upload is where four sessions in a
    row were killed, with no attempt line ever flushed, and there is nothing
    left to upload once the pairs have published themselves.
    """
    return publish_handoff(
        dataset_root,
        dataset=dataset,
        client=client,
        version_notes=version_notes,
        log=log,
        log_path=log_path,
        attempts=attempts,
        sleep=sleep,
        seeds=seeds,
    )


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


def _log_target(log_path: str | Path | None, dataset_root: str | Path) -> Path:
    """Where this session's publish outcome is recorded.

    One place, because the reader (``_read_log``) has to look where the writer
    put it: a reuse decision that read a different file than the pushes write
    would either re-upload everything or, worse, skip a state nobody pushed.
    """
    if log_path is not None:
        return Path(log_path)
    return Path(dataset_root) / PUBLISH_LOG_NAME


def _read_log(
    log_path: str | Path | None, dataset_root: str | Path
) -> dict[str, object] | None:
    """The recorded publish outcome, or ``None`` when there is not one.

    A missing, empty or unreadable log is not an error: it is the first push of
    a container, which has nothing to compare a fingerprint against.
    """
    try:
        loaded = json.loads(_log_target(log_path, dataset_root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _shard_record(
    previous: dict | None, dataset: str, *, sharded: bool
) -> dict | None:
    """The part of a publish log that describes ``dataset``'s last push.

    A sharded push must not treat another shard's success as its own: the
    top-level entry is whatever pushed last, and only the ``shards`` map says
    which dataset that was. A log written before sharding existed has no map;
    its top level describes the single dataset it names, which is shard 0.
    """
    if previous is None:
        return None
    if not sharded:
        return previous
    entries = previous.get("shards")
    if isinstance(entries, dict) and isinstance(entries.get(dataset), dict):
        return entries[dataset]
    if previous.get("dataset") == dataset:
        return previous
    return None


def _write_log(
    log_path: str | Path | None,
    dataset_root: str | Path,
    result: RemotePublishResult,
    *,
    sharded: bool = False,
) -> None:
    """Record the publish outcome beside the dataset root (best-effort).

    The top level is always the push that just happened, because that is what
    the driver reads. When the push was one shard of a queue's handoff, the
    other shards' last outcomes are carried forward under ``shards`` so a
    second shard's success does not erase the first's fingerprint - which is
    what would make the next session re-upload a forty-gigabyte version.
    """
    target = _log_target(log_path, dataset_root)
    payload = result.to_dict()
    if sharded:
        existing = _read_log(log_path, dataset_root)
        entries = dict((existing or {}).get("shards") or {})
        if existing and "shards" not in existing and existing.get("dataset"):
            # The pre-shard log describes shard 0; keep it as shard 0's record
            # rather than dropping it when the first sharded push rewrites
            # the file.
            entries.setdefault(str(existing["dataset"]), dict(existing))
        entries[result.dataset] = dict(payload)
        payload = {**payload, "shards": entries}
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except OSError:
        pass  # logging must never break the exit path


def restore_latest(
    dataset_root: str | Path,
    *,
    dataset: str | None = None,
    datasets: Sequence[str] | None = None,
    client: KaggleDatasetClient,
    log: Callable[[str], None] | None = None,
    attempts: int = RESTORE_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> list[int]:
    """Restore one dataset, or a queue's shards, into the local handoff tree.

    ``datasets`` is the sharded form: each shard is downloaded into the same
    root in the order given, and the merge is the same one a single dataset
    has always used (disjoint membership, so order is not load-bearing). The
    chain stops at the first shard that does not exist: a queue's shards are
    contiguous, so a missing shard means there is nothing after it either, and
    a session that treated a *served-but-empty* shard as bootstrap would train
    over seeds the previous shard already published.
    """
    logger = log if log is not None else (lambda message: None)
    if datasets is None:
        if dataset is None:
            raise ValueError("restore_latest needs dataset= or datasets=")
        names = [dataset]
    else:
        if dataset is not None:
            raise ValueError("pass dataset= or datasets=, not both")
        names = list(datasets)
    restored: set[int] = set()
    for name in names:
        seeds = _restore_one(
            dataset_root,
            dataset=name,
            client=client,
            log=logger,
            attempts=attempts,
            sleep=sleep,
        )
        if seeds is None:
            break
        restored.update(seeds)
    return sorted(restored)


def _restore_one(
    dataset_root: str | Path,
    *,
    dataset: str,
    client: KaggleDatasetClient,
    log: Callable[[str], None] | None = None,
    attempts: int = RESTORE_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> list[int] | None:
    """Restore one dataset's newest version; ``None`` when it does not exist.

    The missing/no-version state is distinguished from the restored-but-empty
    one because a shard chain has to stop on the first, while the second is a
    dataset that exists and carries nothing a session may train over.

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
                    return None
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
