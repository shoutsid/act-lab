"""Drive one declared section 5.1 queue to completion across Kaggle sessions.

``T4_RUNBOOK.md`` section 3 says "re-run the same command until the queue
completes; no scheduling decisions are made by hand". At the measured T4 rate
that is six sessions for the primary queue alone, so the re-push loop is
mechanised here instead of depending on an operator remembering to click push.

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
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

import kaggle_t4

# A seed the runner counts as done: published before this session, or trained
# and published during it. "interrupted" (deadline stop mid-pair) is not done.
COMPLETE_ACTIONS = ("completed", "skipped_complete")
# kagglesdk KernelWorkerStatus; NEW_SCRIPT and QUEUED mean "not started yet".
TERMINAL_STATUSES = ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED")
SUCCESS_STATUS = "COMPLETE"

EXIT_COMPLETE = 0
EXIT_SESSION_FAILED = 1
EXIT_BUDGET_EXHAUSTED = 2
EXIT_QUOTA_INSUFFICIENT = 3
EXIT_NO_PROGRESS = 4
# scripts/v0_verdict.py's distinct code for "no verdict was produced". It is
# deliberately not 1, which means a real cost FAIL, so an operator error can
# never be recorded as the hypothesis being false.
EXIT_NO_VERDICT = 3

# The weekly accelerator line the session kernels bill against.
QUOTA_RESOURCE = "GPU"

STATUS_RE = re.compile(r"KernelWorkerStatus\.([A-Z_]+)")
SESSION_LOG_NAME = "session-log.json"
VERDICT_NAME = "v0-5-1-verdict.json"
# Single-job kernels that are pushed once and reported, rather than queues that
# are re-pushed until a block completes. ``verdict`` is the section 4 decision
# - the reason the queues were run at all.
# ``selfcheck`` is the session battery: it trains tiny non-declared pairs, so
# unlike the verdict it wants a GPU - but only for a few minutes, and it is
# what confirms the target host really does place one arm per device.
ONE_SHOT_MODES = ("preflight", "verdict", "dtypeprobe", "selfcheck")


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


MAIN_GUARD = 'if __name__ == "__main__":\n    raise SystemExit(main())\n'

# What the kernel actually executes: the package, the scripts it drives, their
# tests, the notebooks the report cites, and the docs the runbook references.
# Deliberately **not** ``artifacts/`` - a session must start with no
# ``artifacts/`` directory, so the launch command cannot read a benchmark
# artifact this session did not measure - and not the web side
# (``public/``, ``server/``, the vite/ts configs), which the kernel never
# imports.
CODE_PATHS = (
    "pyproject.toml",
    "README.md",
    "T4_RUNBOOK.md",
    "IMPLEMENTATION_STATUS.md",
    "RESEARCH_REPORT.md",
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

    The kernel runs the *mounted dataset*, not this working tree, while the
    staged kernel body is embedded from this working tree - and the handoff
    manifest records ``code_revision``. So a tree that is not a clean commit
    cannot be published reproducibly, and a dataset that lags the tree runs
    different code than the run claims.
    """
    head = _run(["git", "-C", str(repo), "rev-parse", "HEAD"])
    if head.returncode != 0:
        raise SystemExit(
            f"{repo} is not a git checkout, so the code dataset cannot be pinned "
            "to a revision"
        )
    dirty = _run(["git", "-C", str(repo), "status", "--porcelain"])
    if dirty.stdout.strip():
        raise SystemExit(
            "the working tree has uncommitted changes; the published code dataset "
            "must equal the revision the handoff records, so commit first"
        )
    return head.stdout.strip()


def code_files(repo: str | Path) -> list[str]:
    """Tracked files under :data:`CODE_PATHS`, in git's order."""
    listed = _run(["git", "-C", str(repo), "ls-files"])
    if listed.returncode != 0:
        raise SystemExit(f"git ls-files failed in {repo}: {listed.stderr.strip()}")
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
    archive = folder / CODE_ARCHIVE_NAME
    # The marker is written into the *staging* folder and never the source
    # tree: a marker left at the repo root is an untracked file, so the very
    # next ``code_revision`` call refuses the tree as dirty - a publish that
    # sabotages the following publish.
    marker = folder / REVISION_MARKER
    marker.write_text(f"{revision}\n", encoding="utf-8")
    with tarfile.open(archive, "w") as tar:
        for name in names:
            tar.add(root / name, arcname=name)
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
    body = Path(__file__).with_name("kaggle_t4.py").read_text(encoding="utf-8")
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


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True)


def active_status(kernel: str) -> str:
    """The kernel's current status, or ``""`` when it cannot be queried."""
    result = _run(["kaggle", "kernels", "status", kernel])
    return parse_status(result.stdout + result.stderr)


def parse_quota(payload: str, resource: str = QUOTA_RESOURCE) -> dict[str, object] | None:
    """One resource's line from ``kaggle quota --format json``.

    Returns ``{"remaining_hours": float, "refresh_at": str}``, or ``None`` when
    the resource is absent or the payload is not the expected shape. ``None``
    means "cannot tell", and the caller must treat it as *do not block*: an
    unreadable quota line is not a reason to stall a run that is otherwise
    ready to go.
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
        return {
            "remaining_hours": remaining,
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
    it makes it because the alternative is spending the quota: at the measured
    rate the primary queue alone is six sessions against a 30 GPU-h week.
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


def _wait_for_terminal(kernel: str, poll: float, timeout: float) -> str:
    """Poll the kernel until a terminal status, then return it."""
    deadline = time.monotonic() + timeout
    last = ""
    while True:
        result = _run(["kaggle", "kernels", "status", kernel])
        status = parse_status(result.stdout + result.stderr)
        if status != last:
            print(f"  status: {status or 'unknown'}", flush=True)
            last = status
        if is_terminal(status):
            return status
        if time.monotonic() >= deadline:
            raise SystemExit(f"timed out waiting for {kernel} (last status {last!r})")
        time.sleep(poll)


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
    ensure_code_dataset(Path(__file__).resolve().parent.parent)
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
            push = _run(["kaggle", "kernels", "push", "-p", str(push_dir)])
            print(push.stdout.strip() or push.stderr.strip(), flush=True)
            if push.returncode != 0:
                return EXIT_SESSION_FAILED

        status = _wait_for_terminal(queue.kernel_id, poll, timeout)
        output_dir.mkdir(parents=True, exist_ok=True)
        _run(["kaggle", "kernels", "output", queue.kernel_id, "-p", str(output_dir)])

        if status != SUCCESS_STATUS:
            print(f"session ended with {status}; kernel log follows", flush=True)
            log = output_dir / f"{queue.kernel_id.split('/')[-1]}.log"
            if log.is_file():
                print(log.read_text(encoding="utf-8")[-4000:], flush=True)
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
    print(f"VERDICT: {verdict.get('verdict')} (exit {code})", flush=True)
    return code


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
    if status != SUCCESS_STATUS:
        print(f"{mode} ended with {status}; kernel log follows", flush=True)
        log = output_dir / f"{kernel.split('/')[-1]}.log"
        if log.is_file():
            print(log.read_text(encoding="utf-8")[-4000:], flush=True)
        return EXIT_SESSION_FAILED
    if mode == "verdict":
        return report_verdict(output_dir)
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
    args = parser.parse_args(argv)

    if args.queue in ONE_SHOT_MODES:
        return run_once(args.queue, poll=args.poll, timeout=args.timeout)
    # Deliberately *not* set in this process's environment: the kernel runs on
    # Kaggle and would never see it. ``drive`` pins it into the staged body.
    return drive(
        args.queue,
        hours=args.hours,
        poll=args.poll,
        timeout=args.timeout,
        max_sessions=args.max_sessions,
    )


if __name__ == "__main__":
    raise SystemExit(main())
