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
* ``COMPLETE`` - every declared seed is published *and its own record says ``complete``*;
  go on to the next queue. A seed whose listing reaches the declared step but whose record
  still says ``in-progress`` is reported as stepped-but-not-finished and keeps this verdict
  away, because the next session will re-train it (measured on seed 15027, 2026-09-27).
* ``FAILED`` - the last session ended in ERROR and nothing is live: read the cause.
* ``WAITING_ON_QUOTA`` - the week cannot fit another session, and unused hours expire at
  the refresh. The driver's own ``--wait-for-quota`` handles it *when one is waiting*, so
  this verdict says which of the two it is: a driver sleeping to the refresh, or no
  driver at all - in which case nothing resumes until the campaign is relaunched
  (measured 2026-09-27: a workstation reboot killed the waiting driver, the session it
  had already pushed completed unattended, and the report said WAITING_ON_QUOTA while
  the lock was free).
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
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from act.training.v0_handoff import COMPLETE_STATUS  # noqa: E402

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

#: The registered vocabulary, so a report can place each arm against the loss a model with
#: no signal must reach. Uniform over the vocabulary is ``ln(vocab_size)`` nats, and the
#: declared block trains on synthetic tokens (``config.data_path`` is null), so this is the
#: floor the arms sit on rather than a target - which is why it is worth printing beside
#: differences of 1e-4.
DECLARED_VOCAB: int = int(v0_verdict.REGISTERED_RUN_CONFIG["vocab_size"])  # type: ignore[arg-type]
UNIFORM_PRIOR_NATS: float = math.log(DECLARED_VOCAB)

VERDICT_EXIT = {
    "RUNNING": 0,
    "STALLED": 1,
    "WAITING_ON_QUOTA": 2,
    "FAILED": 3,
    "COMPLETE": 4,
}

#: What the report prints when ``kaggle kernels status`` returns no token. Empty is a state,
#: not a hole: every queue but the primary starts with a kernel that has not been pushed yet,
#: and the driver's push decision reads the same emptiness (``should_adopt("")`` means push).
#: It is *also* what a failed query returns, and the two are indistinguishable from here, so
#: the line names both instead of printing nothing - the same reason the seed count prints
#: "unreadable" with its reason rather than a blank or a zero.
UNKNOWN_KERNEL_NOTE = "unknown (not pushed yet, or the status query failed)"

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


def published_seeds(
    dataset_id: str | Sequence[str], *, timeout: float = 180.0
) -> dict[int, dict]:
    """Every seed the published dataset(s) carry, with the arms and steps.

    Read from the dataset listing rather than downloaded: the session dataset is ~24 GB and
    the listing is a few kilobytes of exactly the information needed to tell a finished seed
    from one whose snapshot stopped part-way.

    A queue whose handoff is sharded (see `docs/handoff-sharding-design.md`) passes its
    datasets in order. The first must be readable - if it is not, the report cannot say
    whether the queue started. A later shard that does not exist yet is not an error: it is
    the state before the first seed of that shard is published, and reading it as anything
    else would make the report fail at the moment it is most needed. Reading it as empty
    under-reports at worst, and an under-reported block can never reach the COMPLETE verdict
    the report exists to protect.
    """
    names = [dataset_id] if isinstance(dataset_id, str) else list(dataset_id)
    merged: dict[int, dict] = {}
    for index, name in enumerate(names):
        try:
            listing = dataset_listing(name, timeout=timeout)
        except RuntimeError:
            if index == 0:
                raise
            continue
        for seed, entry in parse_seed_listing(listing).items():
            if seed in merged:
                raise RuntimeError(
                    f"seed {seed} is published in more than one shard "
                    f"({names[index - 1]!r} and {name!r}); the shard rule forbids "
                    "overlap, so this dataset layout cannot be read"
                )
            merged[seed] = entry
    return merged


def stepped_seeds(seeds: dict[int, dict]) -> list[int]:
    """The seeds the *listing* says reached the declared step on every declared arm.

    That is a candidate, not a finished seed. Measured on the live block, 2026-09-27: seed
    15027's listing carries ``checkpoint-006000.pt`` for both arms - a snapshot is taken after
    each checkpoint, and the last one landed at the declared step - while the seed's own
    published record says ``status: in-progress`` and carries no final loss for either arm,
    because the evaluation that follows training never ran before the window ended. So the
    listing cannot tell a finished seed from one whose window closed after its last
    checkpoint; only the seed's own record can, which is why ``seed_progress`` asks it.
    """
    return [
        seed
        for seed in sorted(seeds)
        if all(seeds[seed]["arms"].get(arm, 0) >= DECLARED_STEPS for arm in DECLARED_ARMS)
    ]


def seed_progress(
    seeds: dict[int, dict], *, statuses: Mapping[int, str | None]
) -> dict:
    """Split the published seeds into complete, partial and stepped-but-unfinished.

    ``statuses`` is required rather than optional, and it is the *seed's own published
    status* (``fetch_seed_status``), not anything derived from the listing: the runner's
    skip rule reads exactly this field, so a report that skipped it would be able to call a
    seed finished that the next session will re-train - which is what happened to seed 15027
    on 2026-09-27. A caller that only has the listing can pass ``{}`` and every candidate
    reads as unconfirmed, which is the honest answer rather than a default.

    Three buckets, because "did not finish the steps" and "finished the steps but its record
    does not say complete" are different states with the same consequence for the queue -
    neither is a seed the campaign can move past:

    * ``partial`` - the listing shows at least one arm short of the declared step.
    * ``unconfirmed`` - the listing shows every arm at the declared step, and the seed's own
      record says something else (or cannot be read; ``unconfirmed_status`` distinguishes
      the two, and ``None`` means unreadable).
    * ``complete`` - every declared arm at the declared step *and* the record says
      ``complete``.

    ``missing`` and ``next_seed`` count both incomplete buckets, so they name the seed the
    next session will actually start at.
    """
    complete, partial, unconfirmed = [], [], []
    for seed in sorted(seeds):
        if not all(
            seeds[seed]["arms"].get(arm, 0) >= DECLARED_STEPS for arm in DECLARED_ARMS
        ):
            partial.append(seed)
        elif statuses.get(seed) == COMPLETE_STATUS:
            complete.append(seed)
        else:
            unconfirmed.append(seed)
    expected = list(DECLARED_SEEDS)
    unfinished = [seed for seed in expected if seed not in complete]
    return {
        "complete": complete,
        "partial": partial,
        "unconfirmed": unconfirmed,
        "unconfirmed_status": {seed: statuses.get(seed) for seed in unconfirmed},
        "declared": expected,
        "missing": unfinished,
        "complete_count": len(complete),
        "partial_count": len(partial),
        "unconfirmed_count": len(unconfirmed),
        "declared_count": len(expected),
        "next_seed": next(iter(unfinished), None),
    }


def fetch_seed_status(dataset_id: str, seed: int, *, timeout: float = 180.0) -> str | None:
    """One published seed's own status, or ``None`` when it cannot be read.

    This is the field the runner's skip rule tests (``published_seed_status`` -> the same
    ``status`` key), so reading it here is what keeps the report and the queue telling the
    same story. ``None`` covers both "no summary published" and "the fetch failed": both
    mean the report cannot call the seed finished, and neither is a completed seed.
    """
    summary = fetch_seed_summary(dataset_id, seed, timeout=timeout)
    status = (summary or {}).get("status")
    return status if isinstance(status, str) else None


def seed_statuses(
    dataset_id: str | Mapping[int, str], seeds, *, timeout: float = 180.0
) -> dict[int, str | None]:
    """Each candidate seed's published status: one small download per candidate.

    Asked only for the seeds the listing calls stepped (``stepped_seeds``), not for every
    published seed: a seed whose last checkpoint is short of the declared step is already
    unfinished, so its record cannot change the answer. Measured 2026-09-27: 1.1-1.3 s per
    seed against the live dataset - the block's twelve seeds cost well under a minute at a
    queue boundary, and none of them is the ~24 GB session dataset.
    """
    return {seed: fetch_seed_status(dataset_id, seed, timeout=timeout) for seed in seeds}


def fetch_seed_summary(
    dataset_id: str | Mapping[int, str], seed: int, *, timeout: float = 180.0
) -> dict | None:
    """One published seed's summary: a kilobyte carrying both arms' final losses.

    Fetched as a **single file**, not by downloading the session dataset - that is ~24 GB
    and this needs five kilobytes per seed. Two things about the CLI are load-bearing here
    and both were measured on 2026-09-21: ``-f`` serves one path out of a dataset, and the
    CLI writes the file under its own basename, so seeds fetched into *one* directory leave
    **one** file (the last one) - which would read as the others being unreadable rather
    than as a fetch that overwrote them. This function therefore owns a directory per call
    and the seed is in its name, so a download that fails names the seed it was for.

    The remote path carries the doubled first component the upload created, because a seed
    is pushed as ``seed-N.tar`` whose members already begin ``seed-N/``.
    """
    if not isinstance(dataset_id, str):
        # Sharded: the seed's record lives in the shard that carries it.
        try:
            dataset_id = dataset_id[seed]
        except KeyError:
            return None
    name = f"seed-{seed}/seed-{seed}/reports/seed-summary.json"
    with tempfile.TemporaryDirectory(prefix=f"act-seed-{seed}-") as directory:
        try:
            subprocess.run(
                [
                    "kaggle", "datasets", "download", "-d", dataset_id,
                    "-f", name, "-p", directory, "-o", "-q",
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        path = Path(directory) / "seed-summary.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None


def seed_readout(
    dataset_id: str | Mapping[int, str], seeds, *, timeout: float = 180.0
) -> dict:
    """What the published seeds *show* - which is not the verdict.

    The queue is incomplete for most of the campaign's life and the registered verdict
    withholds itself until every declared seed is published, so this reports the inputs
    only: each seed's two arm losses, the adaptive-minus-random difference, and how many
    seeds the adaptive arm won. **No threshold is applied and no verdict is implied** - a
    provisional verdict is exactly what a pre-registered rule exists to prevent.

    It also places the arms against ``ln(vocab_size)``, the loss a signal-free model must
    reach on the declared block's synthetic stream, because an arm sitting on that floor
    turns a 1e-4 difference from a result into a reading.
    """
    rows: list[dict] = []
    unread: list[int] = []
    for seed in sorted(seeds):
        summary = fetch_seed_summary(dataset_id, seed, timeout=timeout)
        arms = (summary or {}).get("arms") or {}
        losses = {
            arm: (arms.get(arm) or {}).get("final_loss") for arm in DECLARED_ARMS
        }
        if any(value is None for value in losses.values()):
            unread.append(seed)
            continue
        delta = losses["adaptive"] - losses["random"]
        rows.append(
            {
                "seed": seed,
                "losses": losses,
                "adaptive_minus_random": round(delta, 6),
                "adaptive_wins": delta < 0,
            }
        )
    wins = sum(1 for row in rows if row["adaptive_wins"])
    return {
        "seeds_read": [row["seed"] for row in rows],
        "seeds_unread": unread,
        "rows": rows,
        "adaptive_wins": wins,
        "adaptive_losses": len(rows) - wins,
        "mean_adaptive_minus_random": (
            round(sum(row["adaptive_minus_random"] for row in rows) / len(rows), 6)
            if rows
            else None
        ),
        "uniform_prior_nats": round(UNIFORM_PRIOR_NATS, 4),
        "largest_gap_from_uniform": (
            round(
                max(
                    abs(loss - UNIFORM_PRIOR_NATS)
                    for row in rows
                    for loss in row["losses"].values()
                ),
                6,
            )
            if rows
            else None
        ),
    }


#: The clause a quota wait's action text needs, in both states. They read as the same
#: sentence on purpose - the only difference is *who* is waiting - because the failure this
#: states is remembered as a report that looked identical either way (2026-09-27).
DRIVER_WAITING = "a driver is waiting on it (pid {pid})"
DRIVER_ABSENT = (
    "nothing holds the driver's lock, so nothing resumes at the refresh until the campaign "
    "is relaunched"
)


def driver_lock_line(holder: Mapping[str, object] | None) -> str:
    """Who, if anyone, is driving this queue - stated when nobody is, too.

    The lock used to appear only as a suffix on the kernel line *when it was held*, so "a
    driver is sleeping to the refresh" and "nothing will resume this campaign" printed the
    same way. Measured 2026-09-27: the workstation rebooted while the driver was waiting on
    quota, the session it had already pushed completed unattended (three more seeds
    published, 6/12), and this report said ``WAITING_ON_QUOTA`` - whose action text named
    the driver's own wait - with no lock held at all.
    """
    if holder is None:
        return "not held - nothing is waiting on this queue"
    if "pid" in holder:
        return f"held by pid {holder['pid']}"
    return f"held by an unidentified process ({holder.get('detail', 'no detail')})"


def driver_note(holder: Mapping[str, object] | None) -> str:
    """The quota-wait clause: whether anybody is actually waiting on the refresh."""
    if holder is None:
        return DRIVER_ABSENT
    if "pid" in holder:
        return DRIVER_WAITING.format(pid=holder["pid"])
    return "a driver holds the lock and is waiting on it"


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


def health(
    queue_mode: str, *, window: float | None = None, readout: bool = False
) -> dict:
    """Assemble the campaign's state and pick the verdict.

    ``readout`` additionally fetches each published seed's summary, which costs one small
    network call per seed - so it is opt-in: the health check itself is meant to be cheap
    enough to run at every queue boundary, and the substance of the campaign is a
    different question from whether it is running.
    """
    queue = kaggle_t4.QUEUES[queue_mode]
    status = kaggle_queue.active_status(queue.kernel_id)
    quota = kaggle_queue.active_quota() or {}
    remaining = quota.get("remaining_hours")
    declared_window = float(window if window is not None else kaggle_t4.DEFAULT_HOURS)
    owner: dict[int, str] = {}
    try:
        shards = queue.shards
        seeds = published_seeds([name for name, _ in shards])
        # Which shard owns a seed, for the per-seed summary fetches below.
        owner = {seed: name for name, members in shards for seed in members}
        # Candidates are confirmed against their own published record before any of them is
        # called complete, because the listing cannot tell a finished seed from one whose
        # window closed just after its last checkpoint (see ``stepped_seeds``).
        statuses = seed_statuses(owner, stepped_seeds(seeds))
        progress = seed_progress(seeds, statuses=statuses)
        seeds_error = None
    except RuntimeError as error:
        # "Cannot tell" and "zero" are different states and the report has to say which
        # one it is in. It did not: a dataset that has no versions yet (the live block,
        # `act-v0-5-2`, before its first session) raised here, the counts became ``None``,
        # and the *reason* was dropped - so the line read "published seeds : None of None
        # complete" at exactly the moment an operator decides whether to push, which looks
        # like a broken report rather than like a block that has not started.
        seeds, progress, seeds_error = {}, {}, str(error)
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
            "unconfirmed": progress.get("unconfirmed"),
            "unconfirmed_status": progress.get("unconfirmed_status"),
            "next": progress.get("next_seed"),
            "error": seeds_error,
            # The denominator is a registered constant, so it survives a read failure even
            # though the numerator does not - and a reader who is told one of the two can
            # tell an unreadable dataset from an empty one.
            "declared": len(DECLARED_SEEDS),
        },
        "readout": (
            seed_readout(owner, progress["complete"])
            if readout and progress.get("complete")
            else None
        ),
        "verdict": verdict,
    }


ACTIONS = {
    "RUNNING": "a session is training; the queue resumes from seed {next}",
    "COMPLETE": "every declared seed is published *complete* (its own record says so); drive the next queue",
    "FAILED": "the last session ended in ERROR and nothing is live: read the cause before pushing",
    "WAITING_ON_QUOTA": (
        "the week cannot fit a {window:.1f} h session (refresh {refresh}), and {driver}; "
        "hours left at the refresh are lost"
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
    # "Cannot tell" is printed as such, with its reason: see the note in `health` - this
    # line read "None of None complete" for a live block whose dataset has no versions yet,
    # which looks like a broken report rather than like a block whose first session has not
    # run, and it printed no reason at all.
    published = (
        f"unreadable - {seeds['error']} (of {seeds.get('declared')} declared)"
        if seeds.get("error")
        else f"{seeds['complete_count']} of {seeds['declared_count']} complete"
        + (f" ({', '.join(str(s) for s in seeds['complete'])})" if seeds["complete"] else "")
        + (f"; partial {seeds['partial']}" if seeds["partial"] else "")
        # Named apart from ``partial``: this seed finished its declared steps and its record
        # still does not say complete, so it is neither a seed to skip nor a seed the queue
        # has moved past - and the next session will re-train it.
        + (
            f"; stepped but not finished "
            + ", ".join(
                f"{seed} ({seeds.get('unconfirmed_status', {}).get(seed) or 'record unreadable'})"
                for seed in seeds["unconfirmed"]
            )
            if seeds.get("unconfirmed")
            else ""
        )
    )
    lines = [
        f"Gate-4 campaign health ({state['queue']} queue)",
        f"  published seeds : {published}",
        f"  session kernel  : {state['session_status'] or UNKNOWN_KERNEL_NOTE}",
        f"  driver lock     : {driver_lock_line(holder)}",
        f"  quota           : {quota['remaining_hours']} h remaining of {quota['total_hours']}"
        f" (refresh {quota['refresh_at']})",
    ]
    action = ACTIONS[state["verdict"]].format(
        next=seeds["next"],
        window=quota["declared_window_hours"],
        refresh=quota["refresh_at"],
        remaining=quota["remaining_hours"],
        driver=driver_note(holder),
    )
    lines.append(f"  verdict         : {state['verdict']} - {action}")
    lines.extend(_readout_lines(state.get("readout")))
    return "\n".join(lines)


def _readout_lines(read: dict | None) -> list[str]:
    """What the published seeds show, laid out so a difference of 1e-4 is readable.

    The floor line is the point of the block: the declared block trains on synthetic tokens,
    so ``ln(vocab_size)`` is the loss a model with nothing to learn arrives at, and an arm
    sitting on it is a fact that has to be visible next to the differences.
    """
    if not read:
        return []
    if not read["rows"]:
        return [
            f"  seed readout    : {len(read['seeds_unread'])} published seed(s) could not be "
            "read; no verdict is computed here regardless"
        ]
    lines = [
        f"  seed readout    : {len(read['rows'])} published seed(s) read. These are the "
        "verdict's inputs, not a verdict - the queue is incomplete and the registered "
        "rule withholds itself until it is; no threshold is applied here."
    ]
    for row in read["rows"]:
        losses = row["losses"]
        lines.append(
            f"    seed {row['seed']}      : random {losses['random']:.4f}  "
            f"adaptive {losses['adaptive']:.4f}  "
            f"adaptive-random {row['adaptive_minus_random']:+.6f}"
        )
    lines.append(
        f"    adaptive won {read['adaptive_wins']} of {len(read['rows'])}; "
        f"mean adaptive-random {read['mean_adaptive_minus_random']:+.6f} nats"
    )
    lines.append(
        f"    arms sit within {read['largest_gap_from_uniform']:.6f} nats of "
        f"ln({DECLARED_VOCAB}) = {read['uniform_prior_nats']}, the loss a signal-free "
        "model reaches on this block's synthetic stream"
    )
    return lines


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
    parser.add_argument(
        "--readout", action="store_true",
        help="also fetch each published seed's summary (one small download per seed) and "
             "report the arms' losses; still not a verdict, which an incomplete queue "
             "cannot produce",
    )
    args = parser.parse_args()

    state = health(args.queue, window=args.window, readout=args.readout)
    if args.json:
        print(json.dumps(state, indent=1, sort_keys=True))
    else:
        print(report(state))
    return VERDICT_EXIT[state["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
