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
ran out with seeds still missing (re-run it, or raise ``--max-sessions``).

Usage:

    python scripts/kaggle_queue.py --queue session
    python scripts/kaggle_queue.py --queue control --max-sessions 8 --poll 120
"""

from __future__ import annotations

import argparse
import json
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

STATUS_RE = re.compile(r"KernelWorkerStatus\.([A-Z_]+)")
SESSION_LOG_NAME = "session-log.json"


def parse_status(text: str) -> str:
    """Extract the status token from ``kaggle kernels status`` output."""
    match = STATUS_RE.search(text)
    return match.group(1) if match else ""


def is_terminal(status: str) -> bool:
    return status in TERMINAL_STATUSES


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


def stage(directory: str | Path, queue_mode: str) -> Path:
    """Write a push folder whose kernel body is self-contained.

    A Kaggle script kernel receives only ``code_file``: the CLI sends that
    file's text and nothing else from the folder, and there is no field for
    environment variables. So the entry script embeds the driver verbatim with
    the mode pinned into its ``__main__`` guard, rather than copying
    ``kaggle_t4.py`` and hoping ``ACT_T4_MODE`` is set on the other side.
    Staging fails loudly if that guard ever moves, rather than pushing a
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
    pinned = (
        'if __name__ == "__main__":\n'
        f"    os.environ.setdefault(MODE_ENV, {queue_mode!r})\n"
        "    raise SystemExit(main())\n"
    )
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
    print(f"=== driving queue {queue_mode!r}: {queue.dataset} ===", flush=True)
    work = Path(tempfile.mkdtemp(prefix=f"act-queue-{queue_mode}-"))
    push_dir = stage(work / "push", queue_mode)
    output_dir = work / "output"
    done = declared = 0

    for session in range(1, max_sessions + 1):
        print(f"--- session {session}/{max_sessions} ---", flush=True)
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

    print(
        f"queue {queue_mode!r} still incomplete at {done}/{declared} after "
        f"{max_sessions} session(s); re-run to continue",
        flush=True,
    )
    return EXIT_BUDGET_EXHAUSTED


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue", choices=sorted(kaggle_t4.QUEUES), default="session",
        help="which declared queue to drive (default session)",
    )
    parser.add_argument("--hours", type=float, default=None,
                        help="override ACT_T4_HOURS for the session kernels")
    parser.add_argument("--poll", type=float, default=60.0,
                        help="seconds between status polls (default 60)")
    parser.add_argument("--timeout", type=float, default=13 * 3600.0,
                        help="seconds to wait for one session (default 13 h)")
    parser.add_argument("--max-sessions", type=int, default=8,
                        help="push at most this many sessions (default 8)")
    args = parser.parse_args(argv)

    if args.hours is not None:
        import os

        os.environ[kaggle_t4.HOURS_ENV] = str(args.hours)
    return drive(
        args.queue,
        hours=args.hours,
        poll=args.poll,
        timeout=args.timeout,
        max_sessions=args.max_sessions,
    )


if __name__ == "__main__":
    raise SystemExit(main())
