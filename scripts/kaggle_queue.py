"""Drive one declared section 5.1 queue to completion across Kaggle sessions.

``T4_RUNBOOK.md`` section 3 says "re-run the same command until the queue
completes; no scheduling decisions are made by hand". At the measured T4 rate
that is two sessions for the primary queue, but ~47 GPU-h for the three queues
against a 30 GPU-h week - so the re-push loop is mechanised here, with a quota
gate, instead of depending on an operator remembering to click push and to
stop.

Per session this stages the identical kernel (``scripts/kaggle_t4.py`` plus the
metadata :func:`kaggle_t4.metadata_for` generates), pushes it, waits for a
terminal status, and reads the session's own log. No scheduling decision lives
here: the queue, the clock, the seeds and the deadline rules are the runner's.

Progress is read from the session log rather than from the dataset. The runner
walks the declared queue in order and labels each seed ``skipped_complete``
(a complete version already published) or ``completed`` (trained this session)
before it stops at the deadline, so counting those labels over its ``pairs``
entries gives cumulative completion without a multi-gigabyte dataset download
per session.

Exit codes: 0 = queue complete, 1 = the session failed, 2 = the session budget
ran out with seeds still missing (re-run it, or raise ``--max-sessions``), 3 =
the weekly quota cannot cover another session (re-run after the refresh), 4 =
the last session completed without advancing the queue (fix the cause, then
re-run - another push would repeat it).

Usage:

    python scripts/kaggle_queue.py --queue session
    python scripts/kaggle_queue.py --queue control --max-sessions 8 --poll 120
"""

from __future__ import annotations

import argparse
import fcntl
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import sys
import tempfile
import time
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import kaggle_t4

#: How long the live log tailer is given to exit before it is killed. A capture
#: that is still draining when the session ends must not hold up the exit path.
LIVE_LOG_STOP_SECONDS = 5.0

# A seed the runner counts as done: published before this session, or trained
# and published during it. "interrupted" (deadline stop mid-pair) is not done.
COMPLETE_ACTIONS = ("completed", "skipped_complete")
# kagglesdk KernelWorkerStatus; NEW_SCRIPT and QUEUED mean "not started yet".
TERMINAL_STATUSES = ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED")
SUCCESS_STATUS = "COMPLETE"

#: Where the per-queue driver locks live. Inside ``artifacts/`` (git-ignored) so a
#: lock can never be mistaken for a result.
LOCK_DIR_NAME = ".locks"

#: A log-less failure at or above this many hours is the platform's own session
#: window (measured 2026-09-20: a 9 h window died at 8 h 00 m). Below it, the
#: session died at or before startup, which is a different failure with a
#: different cause and must not be reported as the window.
PLATFORM_KILL_HOURS = 7.5

EXIT_COMPLETE = 0
EXIT_SESSION_FAILED = 1
EXIT_BUDGET_EXHAUSTED = 2
EXIT_QUOTA_INSUFFICIENT = 3
EXIT_NO_PROGRESS = 4
# A second driver for the same queue: refused rather than raced. Added after
# 2026-09-21, when two drivers drove one queue, both found "nothing live", both
# pushed, and the second push replaced the running kernel's version - which the
# platform reports as an ERROR with no log and no output at all.
EXIT_ALREADY_RUNNING = 6
# scripts/v0_verdict.py's distinct code for "no verdict was produced". It is
# deliberately not 1, which means a real cost FAIL, so an operator error can
# never be recorded as the hypothesis being false.
EXIT_NO_VERDICT = 3

# The weekly accelerator line the session kernels bill against.
QUOTA_RESOURCE = "GPU"

STATUS_RE = re.compile(r"KernelWorkerStatus\.([A-Z_]+)")
SESSION_LOG_NAME = "session-log.json"
VERDICT_NAME = "v0-5-1-verdict.json"
# The exit-publish audit trail the kernel keeps beside the session log. Must
# agree with ``act.ops.kaggle_remote.PUBLISH_LOG_NAME``; a test asserts it,
# because a drifted name would make the driver report every session as having
# published nothing.
PUBLISH_LOG_NAME = "remote-publish-log.json"
# Single-job kernels that are pushed once and reported, rather than queues that
# are re-pushed until a block completes. ``verdict`` is the section 4 decision
# - the reason the queues were run at all.
# ``selfcheck`` is the session battery: it trains tiny non-declared pairs, so
# unlike the verdict it wants a GPU - but only for a few minutes, and it is
# what confirms the target host really does place one arm per device.
# ``archzoo`` is the architecture-search horizon arm: one push, and its ledgers
# come back in the kernel output and are imported into ``artifacts/``.
ONE_SHOT_MODES = ("preflight", "verdict", "dtypeprobe", "selfcheck", "archzoo")


def parse_status(text: str) -> str:
    """Extract the status token from ``kaggle kernels status`` output."""
    match = STATUS_RE.search(text)
    return match.group(1) if match else ""


def is_terminal(status: str) -> bool:
    return status in TERMINAL_STATUSES


def should_adopt(status: str) -> bool:
    """Is a session already live, so the driver must join it rather than push?

    Pushing a new version does **not** supersede the running one (measured
    2026-09-15: a second push left the first session running and doubled the
    GPU burn, 2.00x against 1.00x). A driver started while a session is live
    must therefore adopt that session instead of adding a second - and an
    unknown/empty status (the kernel does not exist yet) means push.
    """
    return bool(status) and not is_terminal(status)


def queue_progress(session_log: Mapping[str, object]) -> tuple[int, int]:
    """``(cumulative complete seeds, declared queue size)`` from one log.

    The labels are cumulative by construction: a seed the runner did not train
    because a complete version already existed is labelled
    ``skipped_complete``, so the count never depends on which session it was.
    """
    declared = list(session_log.get("declared_seed_queue") or [])
    pairs = list(session_log.get("pairs") or [])
    done = sum(1 for entry in pairs if entry.get("action") in COMPLETE_ACTIONS)
    return done, len(declared)


def failed_pair_note(session_log: Mapping[str, object]) -> str | None:
    """Describe a pair that raised, or ``None`` when the session had none.

    The runner records ``training_failed`` and stops the queue so the seeds it
    did complete reach the dataset through the exit publish, then the CLI (and
    so the kernel) still exits 0. The driver is what has to notice: a session
    that failed a pair would otherwise be followed by another that repeats the
    failure, because the pairs it completes before failing look like ordinary
    progress.
    """
    for entry in list(session_log.get("pairs") or []):
        if entry.get("action") == "training_failed":
            return (
                f"seed {entry.get('seed')} failed to train "
                f"({entry.get('error')})"
            )
    return None


MAIN_GUARD = 'if __name__ == "__main__":\n    raise SystemExit(main())\n'

# What the kernel actually executes: the package, the scripts it drives, their
# tests, the notebooks the report cites, and the docs the runbook references.
# Deliberately **not** ``artifacts/`` - a session must start with no
# ``artifacts/`` directory, so the launch command cannot read a benchmark
# artifact this session did not measure.
CODE_PATHS = (
    "pyproject.toml",
    "README.md",
    "docs",
    "src",
    "scripts",
    "tests",
    "notebooks",
)
CODE_DATASET_SLUG = "act-lab-code"
REVISION_MARKER = "revision.txt"
# The tree is uploaded as one archive, not as directories. Kaggle's uploader
# with its default ``dir_mode`` **silently drops every subdirectory**, so a
# folder-shaped push published a code dataset containing only the loose files
# at the top level - no ``src/``, no ``scripts/`` - while still reporting a
# successful version. Every kernel then died at bootstrap on "no code dataset
# under /kaggle/input". The service extracts uploaded archives and keeps their
# internal paths, which is the same route the checkpoint handoff uses.
CODE_ARCHIVE_NAME = "code.tar"
# How long a publish's read-back is retried. A ``datasets download`` issued
# right after a version push can still serve the *previous* version (measured
# 2026-09-15), which is indistinguishable from a rejected publish - so the
# check polls instead of failing on the first read.
CODE_VERIFY_TIMEOUT_SECONDS = 120.0
CODE_VERIFY_POLL_SECONDS = 5.0


def code_dataset_id() -> str:
    owner = os.environ.get(kaggle_t4.OWNER_ENV, kaggle_t4.DEFAULT_OWNER)
    return f"{owner}/{CODE_DATASET_SLUG}"


def code_revision(repo: str | Path) -> str:
    """The commit the code dataset must match.

    Everything that ships is read from **HEAD**, not from the working tree
    (`committed_tar`, `committed_text`), so a dirty checkout cannot publish
    uncommitted edits under a committed revision's name - which is the defect
    this function used to detect by refusing to work at all.

    That refusal is gone deliberately, and the reason is measured. A dirty tree
    is the normal state of a tree someone is working in, and the driver calls
    this before every push: on a two-week unattended campaign, one uncommitted
    edit during a poll would end the driver *after* a session had already
    trained and published, with no session 3 ever pushed. Sourcing the payload
    from HEAD removes the failure instead of detecting it - the dataset is now
    exactly a commit by construction - and a dirty tree is *reported* rather
    than fatal (see `dirty_paths` and `ensure_code_dataset`).
    """
    head = _run(["git", "-C", str(repo), "rev-parse", "HEAD"])
    if head.returncode != 0:
        raise SystemExit(
            f"{repo} is not a git checkout, so the code dataset cannot be pinned "
            "to a revision"
        )
    return head.stdout.strip()


def dirty_paths(repo: str | Path) -> list[str]:
    """Tracked paths the working tree differs from HEAD in.

    Not an error: whatever is listed here is simply not what ships, because the
    payload comes from HEAD. It is worth saying out loud, because the opposite
    assumption - that the tree you are editing is the code the session runs - is
    exactly what the old refusal was protecting.
    """
    dirty = _run(["git", "-C", str(repo), "status", "--porcelain"])
    if dirty.returncode != 0:
        return []
    return [
        line[3:].strip()
        for line in dirty.stdout.splitlines()
        if len(line) > 3
    ]


def committed_tar(repo: str | Path) -> dict[str, tuple[tarfile.TarInfo, bytes]]:
    """Every tracked file's bytes and mode at HEAD, in one ``git archive`` call.

    The single source of the code that ships. Using ``git archive`` rather than
    reading the working tree keeps the executable bits git recorded, needs no
    per-file subprocess, and cannot see an uncommitted edit at all.

    The whole committed tree is archived and the caller keeps the paths it wants,
    rather than passing ``CODE_PATHS`` as pathspecs: a pathspec that HEAD happens
    not to carry (a fresh repository, or a tree without ``notebooks/``) makes
    ``git archive`` fail outright, while a missing path is simply absent from the
    members and is reported per-name by ``stage_code_dataset``.
    """
    result = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", "HEAD"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"git archive HEAD failed in {repo}: "
            f"{result.stderr.decode('utf-8', 'replace').strip()}"
        )
    members: dict[str, tuple[tarfile.TarInfo, bytes]] = {}
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as tar:
        for info in tar.getmembers():
            if not info.isfile():
                continue
            handle = tar.extractfile(info)
            members[info.name] = (info, handle.read() if handle is not None else b"")
    return members


def committed_text(repo: str | Path, name: str) -> str:
    """One file's text at HEAD - what the kernel body is embedded from."""
    result = _run(["git", "-C", str(repo), "show", f"HEAD:{name}"])
    if result.returncode != 0:
        raise SystemExit(
            f"HEAD does not carry {name}: "
            f"{result.stderr.strip() or 'git show failed'}"
        )
    return result.stdout


# Where the kernel driver is read from. A module constant, not ``__file__``, so
# pointing ``__file__`` at a stub cannot move which tree the body comes from.
KERNEL_BODY_NAME = "scripts/kaggle_t4.py"
REPO_ROOT = Path(__file__).resolve().parent.parent


def kernel_body(repo: str | Path) -> str:
    """The text of the kernel driver that the pushed body embeds.

    It has to be the same revision as the package the session imports, and that
    package now comes from HEAD. So this file's *committed* copy ships whenever
    the working copy differs from it - and the working copy is used when it does
    not, because then the two are the same text and a local edit stays testable
    without a commit. Either way a body from one revision can never be embedded
    against a package from another, which is the mismatch the old dirty-tree
    refusal was there to prevent.
    """
    root = Path(repo)
    if KERNEL_BODY_NAME not in dirty_paths(root):
        candidate = root / KERNEL_BODY_NAME
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    return committed_text(root, KERNEL_BODY_NAME)


def code_files(repo: str | Path) -> list[str]:
    """Files under :data:`CODE_PATHS` as of HEAD, in git's order.

    HEAD rather than the index: the dataset's marker names a commit, so the
    list has to be that commit's tree - which also means a file deleted in the
    working tree still ships, because it is in the revision being published.
    """
    listed = _run(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", "HEAD"])
    if listed.returncode != 0:
        raise SystemExit(
            f"git ls-tree HEAD failed in {repo}: {listed.stderr.strip()}"
        )
    paths = []
    for line in listed.stdout.splitlines():
        name = line.strip()
        if not name:
            continue
        if name == CODE_PATHS[0] or any(
            name == prefix or name.startswith(f"{prefix}/")
            for prefix in CODE_PATHS[1:]
        ):
            paths.append(name)
    return paths


def stage_code_dataset(repo: str | Path, directory: str | Path, revision: str) -> Path:
    """Stage the code the kernel runs as one archive, plus its metadata.

    ``pyproject.toml`` and ``src/act`` are also read back out of the archive
    before returning, so a staging bug fails here rather than on Kaggle - the
    kernel's own mount search looks for exactly that pair.
    """
    import tarfile

    root = Path(repo)
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    names = code_files(root)
    if "pyproject.toml" not in names or not any(
        name.startswith("src/act/") for name in names
    ):
        raise SystemExit(
            "staged code dataset lacks pyproject.toml + src/act, which is exactly "
            "what the kernel's mount search looks for"
        )
    committed = committed_tar(root)
    archive = folder / CODE_ARCHIVE_NAME
    # The marker is written into the *staging* folder and never the source
    # tree: a marker left at the repo root is an untracked file, so the very
    # next ``code_revision`` call refuses the tree as dirty - a publish that
    # sabotages the following publish.
    marker = folder / REVISION_MARKER
    marker.write_text(f"{revision}\n", encoding="utf-8")
    with tarfile.open(archive, "w") as tar:
        for name in names:
            entry = committed.get(name)
            if entry is None:
                raise SystemExit(
                    f"HEAD does not carry {name}, which the dataset is supposed to "
                    "ship: the archive's marker names a revision, so its payload "
                    "has to be that revision's"
                )
            info, content = entry
            # Normalised so the archive is a function of the revision alone.
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(content))
        tar.add(marker, arcname=REVISION_MARKER)
    with tarfile.open(archive) as tar:
        archived = set(tar.getnames())
    if "pyproject.toml" not in archived or not any(
        name.startswith("src/act/") for name in archived
    ):
        raise SystemExit(f"{archive} does not carry the package tree")
    (folder / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": CODE_DATASET_SLUG,
                "id": code_dataset_id(),
                "licenses": [{"name": "CC0-1.0"}],
                "isPrivate": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return folder


def download_code_dataset(dataset: str, *, work: str | Path) -> Path | None:
    """Download and unzip the published code dataset into ``work``.

    A few hundred KB, which is cheap enough to do before every push - and the
    only way to know what the kernel will actually run. Returns ``None`` when
    the download fails.
    """
    here = Path(work)
    result = _run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", str(here), "--unzip"]
    )
    return here if result.returncode == 0 else None


def code_payload_defect(directory: str | Path) -> str | None:
    """What a downloaded code dataset is missing to run the kernels, if any.

    The revision marker alone is not evidence: it is a loose top-level file,
    so it survives the uploader's silent drop of subdirectories while the
    package tree does not - which is exactly how a code dataset with no
    ``src/`` was read as current.

    The check is the kernel's own mount search, called directly rather than
    re-implemented: the service wraps an uploaded archive in a directory named
    after it, so the tree is nested differently from the pushed folder and only
    that search gets the layout right.
    """
    if kaggle_t4.find_code_root(directory) is not None:
        return None
    return "a pyproject.toml with a sibling src/act"


def read_revision_marker(directory: str | Path) -> str | None:
    """The revision marker inside an already-downloaded code dataset."""
    marker = Path(directory) / REVISION_MARKER
    if not marker.is_file():
        return None
    return marker.read_text(encoding="utf-8").strip() or None


def published_code_revision(
    dataset: str, *, work: str | Path | None = None
) -> str | None:
    """The revision marker in the published dataset, or ``None`` if unknown."""
    here = Path(work) if work is not None else Path(tempfile.mkdtemp(prefix="act-code-check-"))
    if download_code_dataset(dataset, work=here) is None:
        return None
    return read_revision_marker(here)


def ensure_code_dataset(repo: str | Path, *, log=print) -> str:
    """Publish the code dataset when it lags the committed tree, or refuse.

    Nothing else in the tooling notices this: the kernel body is embedded from
    the working tree while ``src/`` is copied from the mount, so a stale
    dataset makes the run silently execute old training code - measured
    2026-09-15, the dataset was a commit behind and every fix in between
    (including the guarded handoff publish) was absent from the session that
    was running.
    """
    dataset = code_dataset_id()
    revision = code_revision(repo)
    dirty = dirty_paths(repo)
    if dirty:
        # Not an error - the payload is HEAD's either way - but the opposite
        # assumption is the one worth correcting out loud.
        log(
            f"  the working tree differs from {revision[:12]} in "
            f"{len(dirty)} path(s) (e.g. {dirty[0]}); HEAD is what ships",
            flush=True,
        )
    fetch = Path(tempfile.mkdtemp(prefix="act-code-fetch-"))
    if download_code_dataset(dataset, work=fetch) is not None:
        published = read_revision_marker(fetch)
        defect = code_payload_defect(fetch)
    else:
        published, defect = None, None
    if published == revision and defect is None:
        log(f"  code dataset {dataset} is at {revision[:12]}", flush=True)
        return revision
    why = (
        f"is at {(published or 'unknown')[:12]}"
        if defect is None
        else f"carries {defect}"
    )
    log(
        f"  code dataset {dataset} {why}, republishing from {revision[:12]}",
        flush=True,
    )
    staging = Path(tempfile.mkdtemp(prefix="act-code-publish-"))
    folder = stage_code_dataset(repo, staging / "code", revision)
    message = f"code at {revision[:12]}"
    result = _run(["kaggle", "datasets", "version", "-p", str(folder), "-m", message])
    if result.returncode != 0:
        created = _run(["kaggle", "datasets", "create", "-p", str(folder)])
        if created.returncode != 0:
            raise SystemExit(
                f"could not publish the code dataset {dataset}: "
                f"{(result.stderr or result.stdout).strip()} / "
                f"{(created.stderr or created.stdout).strip()}"
            )
    # Read it back: a version that dropped the package tree still reports as
    # published, and every kernel then dies at bootstrap. The read-back can
    # serve the version that was current *before* this push, which looks
    # exactly like a rejected publish, so it is polled rather than judged on
    # one read.
    deadline = time.monotonic() + CODE_VERIFY_TIMEOUT_SECONDS
    while True:
        verify = Path(tempfile.mkdtemp(prefix="act-code-verify-"))
        if download_code_dataset(dataset, work=verify) is None:
            break
        still = code_payload_defect(verify)
        marker = read_revision_marker(verify)
        if still is None and marker == revision:
            break
        if time.monotonic() >= deadline:
            raise SystemExit(
                f"published {dataset} is not usable after "
                f"{CODE_VERIFY_TIMEOUT_SECONDS:.0f}s: "
                f"{'carries ' + still if still else f'is at {marker}'} "
                f"(expected {revision[:12]} with a findable package tree)"
            )
        time.sleep(CODE_VERIFY_POLL_SECONDS)
    log(f"  published code dataset {dataset} at {revision[:12]}", flush=True)
    return revision




# Every knob the kernel resolves from its *own* environment. A script kernel
# receives only ``code_file``, so anything set on the pushing side has to be
# written into the body or it silently falls back to a default - how both
# ``--hours`` and an ``ACT_T4_OWNER`` override became no-ops that only looked
# right because the default happened to match. Names, not values: the body
# defines these as module constants and the staged script uses them directly.
# ``QUEUE_ENV`` is deliberately absent - the kernel only reads it in the
# ``metadata`` mode, which :func:`stage` performs locally via
# ``kaggle_t4.metadata_for``.
PINNED_ENV_NAMES = ("MODE_ENV", "OWNER_ENV", "SLUG_ENV")


def pinned_environment(queue_mode: str, hours: float | None) -> list[str]:
    """The ``__main__`` lines that make the pushed kernel self-contained."""
    values = {
        "MODE_ENV": queue_mode,
        "OWNER_ENV": os.environ.get(kaggle_t4.OWNER_ENV, kaggle_t4.DEFAULT_OWNER),
        "SLUG_ENV": os.environ.get(kaggle_t4.SLUG_ENV, kaggle_t4.DEFAULT_SLUG),
    }
    lines = [
        f"    os.environ.setdefault({name}, {values[name]!r})"
        for name in PINNED_ENV_NAMES
    ]
    if hours is not None:
        # The quota gate compares this same number, so a silent fallback to the
        # kernel's default would make the gate unsound.
        lines.append(f"    os.environ.setdefault(HOURS_ENV, {str(hours)!r})")
    return lines


def stage(
    directory: str | Path, queue_mode: str, hours: float | None = None
) -> Path:
    """Write a push folder whose kernel body is self-contained.

    A Kaggle script kernel receives only ``code_file``: the CLI sends that
    file's text and nothing else from the folder, and there is no field for
    environment variables. So the entry script embeds the driver verbatim with
    the mode - and the declared window, when one was given - pinned into its
    ``__main__`` guard, rather than copying ``kaggle_t4.py`` and hoping
    ``ACT_T4_MODE``/``ACT_T4_HOURS`` are set on the other side. Both are the
    same trap: exporting either on the pushing side only sets it on the pushing
    side. Staging fails loudly if that guard ever moves, rather than pushing a
    kernel that would exit on the mode check without doing any work.
    """
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    # Matched to the dataset's revision, like the dataset itself: the body and
    # the mounted ``src/`` have to be one revision. See `kernel_body`.
    body = kernel_body(REPO_ROOT)
    if MAIN_GUARD not in body:
        raise SystemExit(
            "kaggle_t4.py's __main__ guard drifted; kaggle_queue.stage cannot "
            "pin the queue mode into the pushed kernel body"
        )
    for name in (*PINNED_ENV_NAMES, "HOURS_ENV"):
        if f"\n{name} = " not in body:
            raise SystemExit(
                f"kaggle_t4.py no longer defines {name}; staging cannot pin it "
                "into the pushed kernel body"
            )
    pinned_lines = ['if __name__ == "__main__":', *pinned_environment(queue_mode, hours)]
    pinned_lines.append("    raise SystemExit(main())")
    pinned = "\n".join(pinned_lines) + "\n"
    entry = folder / f"run-{queue_mode}.py"
    entry.write_text(body.replace(MAIN_GUARD, pinned), encoding="utf-8")
    metadata = kaggle_t4.metadata_for(queue_mode)
    metadata["code_file"] = entry.name
    (folder / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return folder


def find_session_log(output_dir: str | Path) -> Path | None:
    """The session log in a downloaded kernel output, preferring the copy the
    kernel kept at the output root over the working-dir original."""
    candidates = sorted(Path(output_dir).rglob(SESSION_LOG_NAME), key=lambda p: len(p.parts))
    return candidates[0] if candidates else None


def load_session_log(output_dir: str | Path) -> dict[str, object] | None:
    path = find_session_log(output_dir)
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_publish_log(output_dir: str | Path) -> dict[str, object] | None:
    """The session's exit-publish outcome, or ``None`` when it is absent.

    This is whether the session's work survived at all: the handoff roots live
    in the container overlay, so a failed push leaves nothing for a later
    session to pick up. The kernel keeps the log in its output for exactly
    this reason, and the driver now reports it rather than leaving the
    operator to fetch it.
    """
    candidates = sorted(
        Path(output_dir).rglob(PUBLISH_LOG_NAME), key=lambda path: len(path.parts)
    )
    if not candidates:
        return None
    return json.loads(candidates[0].read_text(encoding="utf-8"))


# Kaggle CLI 2.2.4 with OAuth credentials: an expired access token makes a write
# call fail with this text rather than refreshing, while ``kaggle auth
# print-access-token`` does refresh (measured 2026-09-19: ``datasets create`` failed
# five minutes after expiry, worked after one print-access-token). A driver that
# waits hours for a kernel would otherwise die on the output download it waited for.
AUTH_REQUIRED = "Authentication required to call the Kaggle API"


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(argv, capture_output=True, text=True)
    if (
        argv[:1] == ["kaggle"]
        and result.returncode != 0
        and AUTH_REQUIRED in (result.stdout + result.stderr)
    ):
        subprocess.run(
            ["kaggle", "auth", "print-access-token"], capture_output=True, text=True
        )
        result = subprocess.run(argv, capture_output=True, text=True)
    return result


def active_status(kernel: str) -> str:
    """The kernel's current status, or ``""`` when it cannot be queried."""
    result = _run(["kaggle", "kernels", "status", kernel])
    return parse_status(result.stdout + result.stderr)


def parse_quota(payload: str, resource: str = QUOTA_RESOURCE) -> dict[str, object] | None:
    """One resource's line from ``kaggle quota --format json``.

    Returns ``{"remaining_hours": float, "used_hours": float|None,
    "total_hours": float|None, "refresh_at": str}``, or ``None`` when the
    resource is absent or the payload is not the expected shape. ``None``
    means "cannot tell", and the caller must treat it as *do not block*: an
    unreadable quota line is not a reason to stall a run that is otherwise
    ready to go.

    ``used_hours`` and ``total_hours`` are reported for context - a health
    report that shows 6 h left without showing 24 of 30 already spent invites
    the wrong decision - and are ``None`` rather than absent when the payload
    omits them, so nothing has to guess whether a missing number is zero.
    """
    try:
        rows = json.loads(payload)
    except (TypeError, ValueError):
        return None
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict) or row.get("resource") != resource:
            continue
        raw = str(row.get("remaining", "")).strip().rstrip("hH").strip()
        try:
            remaining = float(raw)
        except ValueError:
            return None
        def hours(key: str) -> float | None:
            raw = str(row.get(key, "")).strip().rstrip("hH").strip()
            try:
                return float(raw)
            except ValueError:
                return None

        return {
            "remaining_hours": remaining,
            "used_hours": hours("used"),
            "total_hours": hours("total"),
            "refresh_at": str(row.get("refreshAt", "")),
        }
    return None


def active_quota(resource: str = QUOTA_RESOURCE) -> dict[str, object] | None:
    """The weekly quota line for ``resource``, or ``None`` when unreadable."""
    result = _run(["kaggle", "quota", "--format", "json"])
    return parse_quota(result.stdout or result.stderr, resource)


def quota_shortfall(
    quota: Mapping[str, object] | None, declared_hours: float
) -> str | None:
    """Why a session must not be pushed now, or ``None`` when it may be.

    A session that runs out of weekly quota part-way still bills every hour it
    used and produces nothing, so the gate is the whole declared window, not a
    partial one. This is the one scheduling decision the driver does make, and
    it makes it because the alternative is spending the quota: the three queues
    at the measured rate are ~47 GPU-h against a 30 GPU-h week, so a session
    pushed into the tail of a week is a session that bills and never finishes.
    """
    if quota is None:
        return None
    remaining = float(quota["remaining_hours"])
    if remaining >= declared_hours:
        return None
    refresh = str(quota.get("refresh_at") or "unknown")
    return (
        f"only {remaining:g} {QUOTA_RESOURCE}-h of the weekly quota is left and a "
        f"session declares {declared_hours:g} h; a partial session bills what it "
        f"uses and produces nothing. Quota refreshes {refresh}"
    )


def burned_note(
    before: Mapping[str, object] | None,
    after: Mapping[str, object] | None,
) -> str:
    """" (it used N GPU-h of the quota)", for a session that produced nothing.

    A session that does no work still bills whatever it ran for, and the only
    way to see that is to difference the quota line across it - the same
    measurement that caught two sessions running at once.
    """
    if before is None or after is None:
        return ""
    try:
        burned = float(before["remaining_hours"]) - float(after["remaining_hours"])
    except (KeyError, TypeError, ValueError):
        return ""
    return f" (it used {burned:.2f} GPU-h of the quota)" if burned > 0 else ""


# The weekly quota is what bounds this campaign: ~55 GPU-hours of work against
# a 30 GPU-hour week. A driver that exits 3 and waits for an operator to re-run
# it spends wall-clock as well as quota, so ``--wait-for-quota`` sleeps to the
# API's own refresh time instead. The margin is there because the refresh is
# reported to the minute, not to the second: waking one second early would read
# the same exhausted week and give up again.
QUOTA_REFRESH_MARGIN_SECONDS = 600.0


def seconds_until_refresh(
    quota: Mapping[str, object] | None, *, now: float | None = None
) -> float | None:
    """Seconds to wait for this quota's refresh, or ``None`` when unreadable.

    ``None`` means "cannot tell", and the caller must not sleep on it - a blind
    sleep of unknown length is how a campaign stalls silently for days. A
    refresh time that has already passed still returns the margin rather than
    zero, so a stale payload cannot turn the wait into a spin.
    """
    if not quota:
        return None
    raw = str(quota.get("refresh_at") or "").strip()
    if not raw:
        return None
    try:
        refresh = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if refresh.tzinfo is None:
        # The API reports a naive UTC timestamp, e.g. 2026-09-26T00:00:00.
        refresh = refresh.replace(tzinfo=timezone.utc)
    current = datetime.fromtimestamp(
        time.time() if now is None else now, tz=timezone.utc
    )
    return max((refresh - current).total_seconds(), 0.0) + QUOTA_REFRESH_MARGIN_SECONDS


def wait_for_refresh(
    quota: Mapping[str, object] | None, *, log=print, sleep=time.sleep
) -> bool:
    """Sleep until ``quota`` refreshes. ``False`` when it cannot be read."""
    wait = seconds_until_refresh(quota)
    if wait is None:
        log(
            "  the weekly quota's refresh time is unreadable, and sleeping for an "
            "unknown length is how a campaign stalls silently; re-run after the "
            "week rolls over instead",
            flush=True,
        )
        return False
    log(
        f"  the week refreshes at {quota.get('refresh_at')} ({wait / 3600:.1f} h "
        "away); waiting rather than billing a partial session",
        flush=True,
    )
    sleep(wait)
    return True


def _wait_for_terminal(kernel: str, poll: float, timeout: float) -> str:
    """Poll the kernel until a terminal status, then return it."""
    return _wait_for_terminal_detailed(kernel, poll, timeout)[0]


def _wait_for_terminal_detailed(
    kernel: str, poll: float, timeout: float
) -> tuple[str, float | None]:
    """Poll to a terminal status; also report when it was first seen RUNNING.

    The *run* duration is what a log-less ending has to be read against, and the
    wall clock from the push also contains the queue wait - so a session that
    spent an hour queued and then died at startup was being reported as an hour
    of training. That is the same class of error as reporting a 0.29 h launch
    failure as the platform's 8 h window, and it chose the wrong explanation for
    the 2026-09-21 failure for exactly this reason.
    """
    deadline = time.monotonic() + timeout
    last = ""
    running_since: float | None = None
    while True:
        result = _run(["kaggle", "kernels", "status", kernel])
        status = parse_status(result.stdout + result.stderr)
        if status != last:
            print(f"  status: {status or 'unknown'}", flush=True)
            last = status
        if status == "RUNNING" and running_since is None:
            running_since = time.monotonic()
        if is_terminal(status):
            return status, running_since
        if time.monotonic() >= deadline:
            raise SystemExit(f"timed out waiting for {kernel} (last status {last!r})")
        time.sleep(poll)


@contextmanager
def capture_live_log(kernel: str, path: Path):
    """Stream a running kernel's log to local disk, so a failure keeps its words.

    ``kaggle kernels logs -f`` streams the execution log while the kernel runs,
    which puts a copy on *this* machine before the platform decides what to keep.
    That matters because of what the platform keeps for a failed session:
    measured 2026-09-21, a session ended ERROR after 1.32 h and ``kaggle kernels
    logs`` returned nothing at all for it afterwards (while the same command
    returns full logs for kernels that completed), so three failures of this
    queue have now been diagnosed from side effects alone - and two of those
    explanations were wrong.

    Best effort by construction: if the tailer cannot start the session still
    runs, and the exit path falls back to the post-hoc fetch.
    """
    handle = None
    process = None
    try:
        handle = open(path, "w", encoding="utf-8")  # noqa: SIM115 - closed below
        process = subprocess.Popen(
            ["kaggle", "kernels", "logs", "-f", kernel],
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as error:
        if handle is not None:
            handle.close()
        print(f"  live log capture unavailable ({error})", flush=True)
    try:
        yield
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=LIVE_LOG_STOP_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
        if handle is not None:
            handle.close()


def fetch_kernel_log(kernel: str) -> str:
    """Ask the platform for a kernel's execution log after the fact.

    ``kaggle kernels output`` returns the kernel's *files* and has never
    returned a log, so a driver that only ran that command could not tell an
    empty log from a log it never requested - and reported the second as the
    first. Measured 2026-09-21: this returns nothing for a failed run and a full
    log for a completed one.
    """
    result = _run(["kaggle", "kernels", "logs", kernel])
    return (result.stdout or "").strip()


def session_failure_log(
    output_dir: Path, kernel: str, live_path: Path | None = None
) -> tuple[str, str]:
    """The session's own account, from the first channel that has one.

    Three channels, in order of fidelity: the live capture streamed while the
    kernel ran (the only one that exists when the platform keeps nothing for a
    failed run), the log file the platform placed in the kernel's output, and
    ``kaggle kernels logs`` asked afterwards. Returns the text and the channel
    it came from, because a log that is merely empty and a log that was never
    requested otherwise read alike - which is the defect that made the 2026-09
    losses undiagnosable.
    """
    if live_path is not None and live_path.is_file():
        live = live_path.read_text(encoding="utf-8", errors="replace").strip()
        if live:
            return live, "live stream captured while it ran"
    log = output_dir / f"{kernel.split('/')[-1]}.log"
    if log.is_file():
        text = log.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            return text, log.name
    fetched = fetch_kernel_log(kernel)
    if fetched:
        return fetched, "`kaggle kernels logs` after the fact"
    return "", ""


def ran_note(hours: float | None) -> str:
    """Name the run duration beside the wall duration, when the two differ."""
    if hours is None:
        return ""
    return f" ({hours:.2f} h of it after the first RUNNING poll)"


class QueueLockedError(RuntimeError):
    """Another driver already holds this queue's kernel."""


def logless_failure_note(elapsed_hours: float) -> str:
    """Explain a session that ended with no log from any of the three channels.

    The reading depends on how long it *ran*, and the two readings are different
    failures with different causes. This text used to be one hardcoded paragraph
    asserting the platform's 8 h kill and "8.05 GPU-h burned" for *any* log-less
    ending, so a 0.29 h launch failure was reported to an operator as an
    eight-hour platform loss - quoting hours this run never spent, at the moment
    they are deciding whether to spend more.

    The number it is given is the run, measured from the first RUNNING poll where
    one was seen, because wall time from the push also contains the queue wait.

    Measured 2026-09-21: a session ended ERROR after 1.32 h with no log from any
    channel, and neither earlier explanation applied - the code dataset had not
    moved (its publish time was 15 minutes before the push) and the queue lock
    was held, so no second driver could have replaced the running version. So the
    branch below names what is *known* to cost a session that long instead of
    asserting a cause: the startup restore, which downloads the whole published
    dataset before any training (22.7 GiB at four seeds, growing ~6.4 GiB per
    seed, and never priced anywhere), and a hard kill, which is what an ERROR
    with nothing flushed and no log retained looks like.
    """
    if elapsed_hours >= PLATFORM_KILL_HOURS:
        return (
            f"  At {elapsed_hours:.2f} h it ran into the platform's session window: "
            "measured 2026-09-20, a session declared a 9 h window died at 8 h 00 m "
            "before its own deadline. The quota cost is the "
            f"{elapsed_hours:.2f} GPU-h reported above."
        )
    return (
        f"  At {elapsed_hours:.2f} h of run it died with nothing flushed, so the "
        "quota cost is the "
        f"{elapsed_hours:.2f} GPU-h reported above, not a full window. Read the "
        "startup path before the training path, and note that two causes are "
        "already excluded for this failure whatever the last one was: a second "
        "driver pushing this same kernel is impossible while the queue lock is "
        "held (exit 6), and the code dataset cannot move under a live mount "
        "without the launch re-checking it. What remains is the session's own "
        "startup restore - it downloads the entire published dataset "
        "(~22.7 GiB and growing) before training one token, so its cost grows "
        "with every seed the queue publishes - and a hard kill that flushed "
        "nothing, which is what an ERROR with no log and no output looks like "
        "and which /kaggle/working (19.5 GiB) is too small to explain."
    )


def lock_directory() -> Path:
    """Where per-queue locks live, overridable for callers that are not a campaign.

    The default sits in the repository's git-ignored ``artifacts/``. ``ACT_QUEUE_LOCK_DIR``
    exists for the test suite, where two workers calling ``main(["--queue", "session"])``
    look exactly like two drivers to a real lock - which is what the lock is for, and also
    what would serialize the suite.
    """
    override = os.environ.get("ACT_QUEUE_LOCK_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "artifacts" / LOCK_DIR_NAME


@contextmanager
def queue_lock(kernel_id: str):
    """Hold an exclusive lock for one queue's kernel, and record who holds it.

    Two drivers on one queue is not a theoretical race. On 2026-09-21 both checked
    ``active_status``, both found no live session, both pushed, and the second push
    replaced the running kernel's version - which the platform reports as an ERROR
    with no kernel log and no output files at all, after 0.29 h and the loss of
    everything that session had trained. The adopt-instead-of-push rule cannot close
    it, because the gap between the status check and the push is exactly where the
    second driver arrives, so the exclusion has to be mutual rather than a check.

    The lock is per kernel, not global: driving one queue must not block another's
    work. The code dataset is the resource they share, and it is guarded where it is
    published instead.
    """
    lock_dir = lock_directory()
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / f"{kernel_id.split('/')[-1]}.lock"
    handle = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        os.lseek(handle, 0, os.SEEK_SET)
        holder = os.read(handle, 4096).decode("utf-8", "replace").strip()
        os.close(handle)
        raise QueueLockedError(holder or "an unidentified process") from error
    stamp = json.dumps(
        {
            "pid": os.getpid(),
            "kernel": kernel_id,
            "since": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    os.truncate(handle, 0)
    os.lseek(handle, 0, os.SEEK_SET)
    os.write(handle, stamp.encode("utf-8"))
    try:
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        os.close(handle)


def drive(
    queue_mode: str,
    *,
    hours: float | None,
    poll: float,
    timeout: float,
    max_sessions: int,
) -> int:
    """Push the queue's kernel until its declared block is complete."""
    queue = kaggle_t4.QUEUES[queue_mode]
    declared_hours = float(hours) if hours is not None else kaggle_t4.DEFAULT_HOURS
    print(f"=== driving queue {queue_mode!r}: {queue.dataset} ===", flush=True)
    # The kernel runs the mounted code, so what it executes has to be the
    # committed tree - checked before any quota is spent, not after.
    repo = Path(__file__).resolve().parent.parent
    staged_revision = ensure_code_dataset(repo)
    work = Path(tempfile.mkdtemp(prefix=f"act-queue-{queue_mode}-"))
    push_dir = stage(work / "push", queue_mode, declared_hours)
    output_dir = work / "output"
    done = declared = 0

    for session in range(1, max_sessions + 1):
        print(f"--- session {session}/{max_sessions} ---", flush=True)
        previous = done
        quota_before = active_quota()
        live = active_status(queue.kernel_id)
        if should_adopt(live):
            # Never run two sessions of one queue: they would double the GPU
            # burn and race for the same dataset and handoff roots.
            print(
                f"  a session is already {live}; adopting it instead of "
                "pushing a second",
                flush=True,
            )
        else:
            shortfall = quota_shortfall(quota_before, declared_hours)
            if shortfall is not None:
                print(f"  {shortfall}", flush=True)
                return EXIT_QUOTA_INSUFFICIENT
            # A session staged now may be pushed hours from now, and the code
            # the kernel actually imports comes from the dataset, not from the
            # body - so re-check the dataset on every push. Without this, a
            # publish in between (another queue's drive, or a fix) silently
            # runs the kernel body against different training code than the
            # revision the session will record.
            current = ensure_code_dataset(repo)
            if current != staged_revision:
                print(
                    f"  code dataset moved to {current[:12]} since staging; "
                    "re-staging the kernel body from the same tree",
                    flush=True,
                )
                push_dir = stage(work / f"push-{session}", queue_mode, declared_hours)
                staged_revision = current
            push = _run(["kaggle", "kernels", "push", "-p", str(push_dir)])
            print(push.stdout.strip() or push.stderr.strip(), flush=True)
            if push.returncode != 0:
                return EXIT_SESSION_FAILED

        session_started = time.monotonic()
        # Captured for the whole of the wait, not fetched afterwards: this is the
        # only channel that survives a run the platform keeps nothing for.
        live_log = work / f"live-{queue_mode}.log"
        with capture_live_log(queue.kernel_id, live_log):
            status, running_since = _wait_for_terminal_detailed(
                queue.kernel_id, poll, timeout
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        _run(["kaggle", "kernels", "output", queue.kernel_id, "-p", str(output_dir)])

        if status != SUCCESS_STATUS:
            ended = time.monotonic()
            elapsed = (ended - session_started) / 3600.0
            ran = None if running_since is None else (ended - running_since) / 3600.0
            print(
                f"session ended with {status} after {elapsed:.2f} h"
                f"{ran_note(ran)}"
                f"{burned_note(quota_before, active_quota())}",
                flush=True,
            )
            kept = (
                sorted(path.name for path in output_dir.iterdir())
                if output_dir.is_dir()
                else []
            )
            text, source = session_failure_log(output_dir, queue.kernel_id, live_log)
            if text:
                print(f"  --- session log ({source}) ---", flush=True)
                print(text[-8000:], flush=True)
            elif kept:
                # The session kept its audit trail but no channel returned a
                # log: worth naming, because it means the run got far enough to
                # write output and the loss is partial.
                print(
                    "  no log came back from the live stream, the kernel output "
                    "or `kaggle kernels logs`; what the output did return: "
                    + ", ".join(kept),
                    flush=True,
                )
            else:
                # Silence here is what made the 2026-09-20 loss undiagnosable:
                # the driver printed "kernel log follows" and then nothing, so
                # an operator could not tell an empty log from a missing one.
                print(
                    "  no log came back from any channel (the live stream, the "
                    "kernel output, or `kaggle kernels logs`) and the output "
                    "carries no files at all, so the session's own account of "
                    "itself is lost with it. Nothing it trained is durable (the "
                    "exit publish is the only copy and the session never "
                    "reached it), so the queue restarts from its last published "
                    "seed.",
                    flush=True,
                )
                print(
                    logless_failure_note(ran if ran is not None else elapsed),
                    flush=True,
                )
            return EXIT_SESSION_FAILED

        session_log = load_session_log(output_dir)
        if session_log is None:
            print("no session-log.json in the kernel output", flush=True)
            return EXIT_SESSION_FAILED
        done, declared = queue_progress(session_log)
        print(
            f"  queue progress: {done}/{declared} seeds complete "
            f"(session completed {session_log.get('pairs_completed')} pair(s))",
            flush=True,
        )
        publish = load_publish_log(output_dir)
        if publish is not None:
            print(
                f"  exit publish: {publish.get('status')} "
                f"({publish.get('seeds_published')} seed version(s))",
                flush=True,
            )
            if publish.get("status") == "failed":
                # The session's pairs are gone with its container: there is no
                # local copy for the next session to push. Re-training them to
                # re-attempt the push costs a whole window, so stop and let the
                # recorded error be read first - the usual cause is transient,
                # which is why the push already retried.
                print(
                    f"  exit publish FAILED ({publish.get('error')}); this "
                    "session's pairs are lost with the container; stopping "
                    "rather than re-training them",
                    flush=True,
                )
                return EXIT_SESSION_FAILED

        failure = failed_pair_note(session_log)
        if failure is not None:
            # The session's completed seeds are published (the runner stops the
            # queue rather than crashing precisely so the exit publish runs),
            # but re-pushing would train into the same failure. Stop, so the
            # cause is fixed instead of being paid for by the hour.
            print(f"  {failure}; stopping (another push would repeat it)", flush=True)
            return EXIT_SESSION_FAILED
        if declared and done >= declared:
            print(f"=== queue {queue_mode!r} COMPLETE ===", flush=True)
            return EXIT_COMPLETE
        if done <= previous:
            # A successful session that advanced nothing. It usually means the
            # clock never fit a pair (rule 1 idles), so re-pushing repeats it
            # verbatim and each repeat bills quota - the one loop that can
            # spend a week's budget without producing evidence.
            burned = burned_note(quota_before, active_quota())
            print(
                f"  session completed but the queue did not advance "
                f"({previous} -> {done} of {declared} seeds){burned}; another "
                "push would repeat it, so stopping",
                flush=True,
            )
            return EXIT_NO_PROGRESS

    print(
        f"queue {queue_mode!r} still incomplete at {done}/{declared} after "
        f"{max_sessions} session(s); re-run to continue",
        flush=True,
    )
    return EXIT_BUDGET_EXHAUSTED


def import_archzoo_output(
    output_dir: str | Path, repo: str | Path, *, log=print
) -> dict[str, object] | None:
    """Copy the horizon arm's ledgers out of the kernel output into ``artifacts/``.

    The kernel keeps ``<experiment>/`` whole - design files, per-seed ledgers, the
    per-seed run logs and ``archzoo-status.json`` - so importing is a directory
    merge into the repo's own artifacts root, where ``arch_zoo_analyze.py`` and
    ``verify_record.py`` read every other arm from. Returns the kernel's status
    record, or ``None`` when the output holds no experiment directory (a kernel
    that died before bootstrapping writes nothing).
    """
    source = Path(output_dir) / kaggle_t4.ARCHZOO_EXPERIMENT
    if not source.is_dir():
        log(f"  no {kaggle_t4.ARCHZOO_EXPERIMENT}/ in the kernel output", flush=True)
        return None
    target = Path(repo) / "artifacts" / kaggle_t4.ARCHZOO_EXPERIMENT
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(source.iterdir()):
        if path.is_file():
            shutil.copy2(path, target / path.name)
            copied.append(path.name)
    log(f"  imported {len(copied)} file(s) into {target}: {', '.join(copied)}", flush=True)
    status_path = target / kaggle_t4.ARCHZOO_STATUS_NAME
    if not status_path.is_file():
        return None
    status = json.loads(status_path.read_text())
    rows = status.get("ledger_rows", {})
    log(
        f"  archzoo complete={status.get('complete')} "
        f"stopped_at_deadline={status.get('stopped_at_deadline')} "
        f"hours={float(status.get('hours', 0.0)):.2f} rows={rows}",
        flush=True,
    )
    missing = status.get("missing") or []
    if missing:
        # The kernel exits 0 whatever happened, so an incomplete arm has to say so
        # here or it reads as a finished one. Naming the absent runs is the
        # difference between "the design rule could not be read" and "re-push".
        shown = ", ".join(str(item) for item in missing[:8])
        log(
            f"  WARNING: the arm is incomplete - {len(missing)} requested run(s) "
            f"absent ({shown}{' ...' if len(missing) > 8 else ''}); the design rule "
            "cannot be read until they exist",
            flush=True,
        )
    return status


def report_verdict(output_dir: str | Path) -> int:
    """Print the decision the verdict kernel computed and mirror its code.

    The kernel exits 0 whatever the decision, so the artifact is the only
    carrier of the answer. Returning the registered code makes
    ``--queue verdict`` scriptable: 0 PASS, 1 FAIL, 2 PASS-with-qualification
    (a withholding clause fired), 3 no verdict was produced.
    """
    candidates = sorted(Path(output_dir).rglob(VERDICT_NAME), key=lambda p: len(p.parts))
    if not candidates:
        print(f"no {VERDICT_NAME} in the kernel output - no verdict produced", flush=True)
        return EXIT_NO_VERDICT
    verdict = json.loads(candidates[0].read_text(encoding="utf-8"))
    print(json.dumps(verdict, indent=2, sort_keys=True), flush=True)
    code = verdict.get("exit_code")
    if not isinstance(code, int):
        print("verdict artifact carries no exit_code", flush=True)
        return EXIT_NO_VERDICT
    # The script legitimately reports a provisional verdict on an incomplete
    # block (it records `seeds_missing` and withholds Gate 4), but this
    # function's code is about the *run*, and a block with declared seeds
    # missing has not produced the registered decision. Returning the
    # provisional PASS would let a scripted caller read a partial block as the
    # answer, so it reports "no verdict" instead - the artifact still carries
    # the provisional numbers for a human.
    if verdict.get("queue_complete") is False:
        missing = verdict.get("seeds_missing") or []
        print(
            f"VERDICT (provisional, NOT the registered decision): "
            f"{verdict.get('verdict')} on an incomplete block - "
            f"{len(missing)} declared seed(s) missing; keep driving the queues",
            flush=True,
        )
        return EXIT_NO_VERDICT
    print(f"VERDICT: {verdict.get('verdict')} (exit {code})", flush=True)
    return code


def _print_log_tail(output_dir: Path, kernel: str, limit: int = 4000) -> None:
    """Print the end of a kernel's log, where its result lives.

    Falls back to ``kaggle kernels logs``, because for a script kernel the
    execution log is not one of the output files: the ``<slug>.log`` this used
    to look for is usually absent, so the diagnostics that call this - whose
    whole result *is* the log - printed nothing at all.
    """
    text, source = session_failure_log(Path(output_dir), kernel)
    if not text:
        return
    # Named even on success: a reader who cannot tell which channel produced the
    # lines cannot tell a live capture from a post-hoc one.
    print(f"  --- kernel log ({source}) ---", flush=True)
    print(text[-limit:], flush=True)


def run_once(mode: str, *, poll: float, timeout: float) -> int:
    """Push a single-job kernel and report what it produced.

    No re-push loop and no quota gate: the verdict runs on CPU and bills
    nothing, and pre-flight is one measurement. Reporting the verdict from the
    same tool as the queues is the point - it is the step whose output decides
    whether the whole run was worth its quota.
    """
    kernel = str(kaggle_t4.metadata_for(mode)["id"])
    print(f"=== running {mode} ({kernel}) ===", flush=True)
    # One-shot kernels run the mounted code too, so the same staleness rule
    # applies: the verdict reads the runner's own summaries, and a dataset that
    # lags the tree would compute the decision with an older reader than the
    # one that wrote them.
    ensure_code_dataset(Path(__file__).resolve().parent.parent)
    push_dir = stage(Path(tempfile.mkdtemp(prefix=f"act-{mode}-")) / "push", mode)
    push = _run(["kaggle", "kernels", "push", "-p", str(push_dir)])
    print(push.stdout.strip() or push.stderr.strip(), flush=True)
    if push.returncode != 0:
        return EXIT_SESSION_FAILED

    status = _wait_for_terminal(kernel, poll, timeout)
    output_dir = Path(tempfile.mkdtemp(prefix=f"act-{mode}-out-"))
    _run(["kaggle", "kernels", "output", kernel, "-p", str(output_dir)])
    if mode == "archzoo":
        # Imported before the status check: a kernel that ended with an error
        # after hours of training may still have kept its ledgers, and those
        # rows cost GPU-hours to regenerate.
        imported = import_archzoo_output(
            output_dir, Path(__file__).resolve().parent.parent
        )
        _print_log_tail(output_dir, kernel)
        if status != SUCCESS_STATUS or imported is None:
            print(f"{mode} ended with {status}", flush=True)
            return EXIT_SESSION_FAILED
        if not imported.get("complete"):
            print(f"{mode} kernel completed but the arm did not", flush=True)
            return EXIT_BUDGET_EXHAUSTED
        print(f"=== {mode} complete ({kernel}) ===", flush=True)
        return EXIT_COMPLETE
    if status != SUCCESS_STATUS:
        print(f"{mode} ended with {status}; kernel log follows", flush=True)
        _print_log_tail(output_dir, kernel)
        return EXIT_SESSION_FAILED
    if mode == "verdict":
        return report_verdict(output_dir)
    # The diagnostics *are* the result here - the measured rate, the device
    # count, the placement a battery verified - so report the kernel log rather
    # than only the fact that the kernel succeeded, which is what an operator
    # driving this from a terminal would otherwise have to fetch by hand.
    _print_log_tail(output_dir, kernel)
    print(f"=== {mode} complete ({kernel}) ===", flush=True)
    return EXIT_COMPLETE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue", choices=sorted((*kaggle_t4.QUEUES, *ONE_SHOT_MODES)), default="session",
        help="which declared queue to drive, or a one-shot job to run once "
        f"({', '.join(ONE_SHOT_MODES)}); default session",
    )
    parser.add_argument("--hours", type=float, default=None,
                        help="the session window, pinned into the pushed kernel "
                             "(default: the kernel's own ACT_T4_HOURS default)")
    parser.add_argument("--poll", type=float, default=60.0,
                        help="seconds between status polls (default 60)")
    parser.add_argument("--timeout", type=float, default=13 * 3600.0,
                        help="seconds to wait for one session (default 13 h)")
    parser.add_argument("--max-sessions", type=int, default=8,
                        help="push at most this many sessions (default 8)")
    parser.add_argument(
        "--wait-for-quota",
        action="store_true",
        help="when the week is short, sleep to the quota's refresh and re-run "
             "instead of exiting 3 (the campaign needs ~55 GPU-h against a "
             "30 GPU-h week, so this is the difference between a two-week run "
             "and one that waits for an operator every week)",
    )
    parser.add_argument("--max-waits", type=int, default=2,
                        help="refreshes to wait through before giving up (default 2)")
    args = parser.parse_args(argv)

    # Deliberately *not* set in this process's environment: the kernel runs on
    # Kaggle and would never see it. ``drive`` pins it into the staged body.
    def attempt() -> int:
        return drive(
            args.queue,
            hours=args.hours,
            poll=args.poll,
            timeout=args.timeout,
            max_sessions=args.max_sessions,
        )

    def campaign() -> int:
        if not args.wait_for_quota:
            return attempt()
        # One wait per weekly refresh. ``drive`` reads the quota before every
        # push, so this loop only decides whether to sleep between attempts -
        # the scheduling rules stay where they already are.
        for waited in range(args.max_waits + 1):
            code = attempt()
            if code != EXIT_QUOTA_INSUFFICIENT or waited == args.max_waits:
                return code
            if not wait_for_refresh(active_quota()):
                return code
        return EXIT_QUOTA_INSUFFICIENT

    # One driver per kernel, held for the whole run including the quota waits - a
    # waiter that released its lock would be exactly the second driver this
    # prevents. One-shot jobs take the lock too, so two of them cannot race each
    # other's push.
    kernel = str(kaggle_t4.metadata_for(args.queue)["id"])
    try:
        with queue_lock(kernel):
            if args.queue in ONE_SHOT_MODES:
                return run_once(args.queue, poll=args.poll, timeout=args.timeout)
            return campaign()
    except QueueLockedError as locked:
        print(
            f"another driver already holds {kernel}: {locked}",
            flush=True,
        )
        print(
            "  refusing to race it. Two drivers on one queue both find no live "
            "session and both push, and the second push replaces the running "
            "kernel's version - a logless early ERROR that costs the session that "
            "was training. Wait for it, or stop it first.",
            flush=True,
        )
        return EXIT_ALREADY_RUNNING


if __name__ == "__main__":
    raise SystemExit(main())
