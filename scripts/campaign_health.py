"""One command that says whether the Gate-4 campaign is running, waiting, or stalled.

Both 2026-09 session losses were expensive partly for a reason that has nothing to do with
Kaggle: nobody could tell, without investigating by hand, that the campaign had stopped.
The campaign is driven by a process on a workstation, and a process on a workstation dies
when the terminal, the shell or the machine does - so the campaign's *state* has to be
readable from something that outlives it.

A watchdog process is the obvious answer and it does not work, for the same reason: it dies
with the session it is watching. So this reads durable remote state instead - the published
dataset (the queue's real progress), the kernel's status (is a session live), and the
weekly quota (can another session even fit) - and answers the only question that matters
from a distance: **is anything driving this, or has it stopped?**

The verdict is deliberately one of five, because the right action differs for each:

* ``RUNNING`` - a session is live; nothing to do.
* ``COMPLETE`` - every declared seed is published; go on to the next queue.
* ``FAILED`` - the last session ended in ERROR and nothing is live: read the cause.
* ``WAITING_ON_QUOTA`` - the week cannot fit another session; the driver's own
  ``--wait-for-quota`` handles it, and unused hours expire at the refresh.
* ``STALLED`` - no session is live, the quota could cover one, and nothing is driving it.
  This is the state both losses sat in.

    PYTHONPATH=src python scripts/campaign_health.py                 # human report
    PYTHONPATH=src python scripts/campaign_health.py --json          # for a script or cron
    PYTHONPATH=src python scripts/campaign_health.py --queue control

Exit codes are the verdict, so a later session can act on it: 0 RUNNING, 1 STALLED,
2 WAITING_ON_QUOTA, 3 FAILED, 4 COMPLETE.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import kaggle_queue  # noqa: E402
import kaggle_t4  # noqa: E402
import v0_verdict  # noqa: E402

#: The registered run's per-arm step count, i.e. the step a *complete* seed is published at.
#: Taken from the verdict's own registered config rather than restated, so a published seed
#: cannot be called complete against a step count nobody registered.
DECLARED_STEPS: int = int(v0_verdict.REGISTERED_RUN_CONFIG["total_steps"])  # type: ignore[arg-type]
DECLARED_ARMS: tuple[str, ...] = tuple(
    str(mode) for mode in v0_verdict.REGISTERED_RUN_CONFIG["arms"]  # type: ignore[union-attr]
)
DECLARED_SEEDS: tuple[int, ...] = tuple(v0_verdict.DECLARED_SEEDS)

VERDICT_EXIT = {
    "RUNNING": 0,
    "STALLED": 1,
    "WAITING_ON_QUOTA": 2,
    "FAILED": 3,
    "COMPLETE": 4,
}

#: A published seed directory, with the arm and step of one of its checkpoints:
#: ``seed-15000/seed-15000/checkpoints/adaptive/checkpoint-006000.pt``.
SEED_PATH = re.compile(r"^seed-(\d+)/.*?/checkpoints/([^/]+)/checkpoint-(\d+)\.pt$")


#: The CLI prints this above the table when a listing has more pages. It is not a
#: file row, so it is harmless to the parser below - it simply never matches
#: ``SEED_PATH`` - and it is the only signal that a page is not the whole answer.
PAGE_TOKEN_RE = re.compile(r"^Next Page Token = (\S+)\s*$", re.MULTILINE)


def parse_seed_listing(listing: str) -> dict[int, dict]:
    """The seeds in a ``kaggle datasets files`` listing, with the arms and steps each has.

    Split out from the CLI call because the listing's shape is the fragile part: it is a
    padded table whose name column is followed by size and creationDate.
    """
    seeds: dict[int, dict] = {}
    for line in listing.splitlines():
        # The listing is a padded table: the name column is followed by size and
        # creationDate, and the name itself is space-padded to a fixed width. So the name is
        # the first whitespace-delimited field, not the stripped line - `.strip()` leaves
        # `...checkpoint-006000.pt  3399209661  2026-09-21 ...` in one piece (103 characters
        # against the path's 63) and nothing matches.
        fields = line.split()
        if not fields:
            continue
        match = SEED_PATH.match(fields[0])
        if not match:
            continue
        seed, arm, step = int(match.group(1)), match.group(2), int(match.group(3))
        entry = seeds.setdefault(seed, {"arms": {}, "max_step": 0})
        entry["arms"][arm] = max(entry["arms"].get(arm, 0), step)
        entry["max_step"] = max(entry["max_step"], step)
    return seeds


def dataset_listing(
    dataset_id: str, *, timeout: float = 180.0, page_size: int = 200
) -> str:
    """The *whole* file listing, following the CLI's page tokens.

    ``kaggle datasets files`` pages at 20 names and prints ``Next Page Token = ...`` when
    there are more, so one call sees only the first page. Measured 2026-09-21: the session
    dataset holds exactly 20 files at four seeds (five per seed), and the listing is ordered
    by name - so the fifth published seed would sit entirely on page 2 and read as *not
    published*, which is a stalled report at the exact moment the queue advanced. That is
    the one wrong answer this report exists to avoid, so the pages are joined here and every
    reader sees the same complete listing.
    """
    pages: list[str] = []
    token: str | None = None
    seen: set[str] = set()
    while True:
        argv = ["kaggle", "datasets", "files", dataset_id, "--page-size", str(page_size)]
        if token is not None:
            argv += ["--page-token", token]
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"kaggle datasets files {dataset_id} failed: {proc.stderr.strip()[:400]}"
            )
        pages.append(proc.stdout)
        match = PAGE_TOKEN_RE.search(proc.stdout)
        token = match.group(1) if match else None
        if token is None:
            break
        # A service that repeats a token would otherwise spin here forever, and
        # a health report that hangs is worse than one that under-reports.
        if token in seen:
            raise RuntimeError(
                f"kaggle datasets files {dataset_id} repeated page token {token!r}"
            )
        seen.add(token)
    return "\n".join(pages)


def published_seeds(dataset_id: str, *, timeout: float = 180.0) -> dict[int, dict]:
    """Every seed in the published dataset, with the arms and steps it has.

    Read from the dataset listing rather than downloaded: the session dataset is ~24 GB and
    the listing is a few kilobytes of exactly the information needed to tell a finished seed
    from one whose snapshot stopped part-way.
    """
    return parse_seed_listing(dataset_listing(dataset_id, timeout=timeout))


def seed_progress(seeds: dict[int, dict]) -> dict:
    """Split the published seeds into complete and partial, and say what is next."""
    complete, partial = [], []
    for seed in sorted(seeds):
        entry = seeds[seed]
        finished = all(
            entry["arms"].get(arm, 0) >= DECLARED_STEPS for arm in DECLARED_ARMS
        )
        (complete if finished else partial).append(seed)
    expected = list(DECLARED_SEEDS)
    return {
        "complete": complete,
        "partial": partial,
        "declared": expected,
        "missing": [seed for seed in expected if seed not in complete],
        "complete_count": len(complete),
        "partial_count": len(partial),
        "declared_count": len(expected),
        "next_seed": next((seed for seed in expected if seed not in complete), None),
    }


def lock_holder(kernel_id: str) -> dict | None:
    """Who holds the driver's lock for this kernel, if anyone.

    Probing the lock means briefly taking it, which is safe: a probe that acquires and
    releases cannot block a real driver, and finding it held is the signal that a driver is
    alive on this machine.
    """
    try:
        with kaggle_queue.queue_lock(kernel_id):
            return None
    except kaggle_queue.QueueLockedError as holder:
        try:
            return json.loads(str(holder))
        except json.JSONDecodeError:
            return {"detail": str(holder)}


def health(queue_mode: str, *, window: float | None = None) -> dict:
    """Assemble the campaign's state and pick the verdict."""
    queue = kaggle_t4.QUEUES[queue_mode]
    status = kaggle_queue.active_status(queue.kernel_id)
    quota = kaggle_queue.active_quota() or {}
    remaining = quota.get("remaining_hours")
    declared_window = float(window if window is not None else kaggle_t4.DEFAULT_HOURS)
    try:
        seeds = published_seeds(queue.dataset)
        progress = seed_progress(seeds)
    except RuntimeError as error:
        seeds, progress = {}, {"error": str(error)}
    holder = lock_holder(queue.kernel_id)

    live = kaggle_queue.should_adopt(status)
    if live:
        verdict = "RUNNING"
    elif progress.get("missing") == [] and progress.get("declared_count"):
        verdict = "COMPLETE"
    elif status == "ERROR":
        verdict = "FAILED"
    elif remaining is not None and float(remaining) < declared_window:
        verdict = "WAITING_ON_QUOTA"
    else:
        verdict = "STALLED"

    return {
        "queue": queue_mode,
        "dataset": queue.dataset,
        "kernel": queue.kernel_id,
        "session_status": status,
        "driver_lock": holder,
        "quota": {
            "remaining_hours": remaining,
            "used_hours": quota.get("used_hours"),
            "total_hours": quota.get("total_hours"),
            "refresh_at": quota.get("refresh_at"),
            "declared_window_hours": declared_window,
        },
        "seeds": {
            "complete_count": progress.get("complete_count"),
            "declared_count": progress.get("declared_count"),
            "complete": progress.get("complete"),
            "partial": progress.get("partial"),
            "next": progress.get("next_seed"),
        },
        "verdict": verdict,
    }


ACTIONS = {
    "RUNNING": "a session is training; the queue resumes from seed {next}",
    "COMPLETE": "every declared seed is published; drive the next queue",
    "FAILED": "the last session ended in ERROR and nothing is live: read the cause before pushing",
    "WAITING_ON_QUOTA": (
        "the week cannot fit a {window:.1f} h session (refresh {refresh}); the driver's "
        "--wait-for-quota covers this, and hours left at the refresh are lost"
    ),
    "STALLED": (
        "no session is live, {remaining} h of quota remains, and nothing holds the driver "
        "lock - this is the state both 2026-09 losses sat in"
    ),
}


def report(state: dict) -> str:
    quota = state["quota"]
    seeds = state["seeds"]
    holder = state["driver_lock"]
    lines = [
        f"Gate-4 campaign health ({state['queue']} queue)",
        f"  published seeds : {seeds['complete_count']} of {seeds['declared_count']} complete"
        + (f" ({', '.join(str(s) for s in seeds['complete'])})" if seeds["complete"] else "")
        + (f"; partial {seeds['partial']}" if seeds["partial"] else ""),
        f"  session kernel  : {state['session_status']}"
        + (f", driver lock held by pid {holder['pid']}" if holder and "pid" in holder else ""),
        f"  quota           : {quota['remaining_hours']} h remaining of {quota['total_hours']}"
        f" (refresh {quota['refresh_at']})",
    ]
    action = ACTIONS[state["verdict"]].format(
        next=seeds["next"],
        window=quota["declared_window_hours"],
        refresh=quota["refresh_at"],
        remaining=quota["remaining_hours"],
    )
    lines.append(f"  verdict         : {state['verdict']} - {action}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue", choices=sorted(kaggle_t4.QUEUES), default="session",
        help="which declared queue to report on (default session)",
    )
    parser.add_argument(
        "--window", type=float, default=None,
        help="the session window to test the quota against "
             "(default: the kernel's declared 7 h)",
    )
    parser.add_argument("--json", action="store_true", help="print the state as JSON")
    args = parser.parse_args()

    state = health(args.queue, window=args.window)
    if args.json:
        print(json.dumps(state, indent=1, sort_keys=True))
    else:
        print(report(state))
    return VERDICT_EXIT[state["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
