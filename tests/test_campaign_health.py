"""Pin the campaign health report: what it reads, and which verdict each state gets.

The report exists because two sessions were lost while nobody could tell the campaign had
stopped without investigating by hand - so the part worth pinning is the *verdict*, which is
what decides whether somebody is paged to relaunch it. The parsing is pinned too, with the
real listing shape: it is a padded table, and reading the stripped line rather than the first
field silently matches nothing, which reads as "no seeds published" - the one wrong answer
that would look like a healthy stalled campaign.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import campaign_health  # noqa: E402

# Copied from the live listing (2026-09-21): the name column is padded to 63 characters,
# then size, then creationDate. Two complete seeds, one whose adaptive arm stopped early.
LISTING = """name                                                                   size  creationDate
---------------------------------------------------------------  ----------  --------------------------
seed-15000/seed-15000/checkpoints/adaptive/checkpoint-006000.pt  3399209661  2026-09-21 08:05:15.249000
seed-15000/seed-15000/checkpoints/random/checkpoint-006000.pt    3399209277  2026-09-21 08:05:32.497000
seed-15000/seed-15000/manifests/manifest.json                           596  2026-09-21 08:04:02.743000
seed-15002/seed-15002/checkpoints/adaptive/checkpoint-003000.pt  3399209661  2026-09-21 08:05:36.710000
seed-15002/seed-15002/checkpoints/random/checkpoint-006000.pt    3399209277  2026-09-21 08:05:46.588000
"""


def test_the_listing_follows_page_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """One CLI call sees 20 names, and the dataset is already that big.

    Measured 2026-09-21: the session dataset holds **exactly 20 files at four seeds**
    (five per seed) and the listing is ordered by name, so the fifth published seed
    would sit entirely on page 2. A reader that ignores the token reports it as not
    published - a *stalled* verdict at the exact moment the queue advanced, which is
    the one wrong answer this report exists to avoid.
    """
    second = """name                                                                   size  creationDate
---------------------------------------------------------------  ----------  --------------------------
seed-15004/seed-15004/checkpoints/adaptive/checkpoint-006000.pt  3399209661  2026-09-21 09:05:15.249000
seed-15004/seed-15004/checkpoints/random/checkpoint-006000.pt    3399209277  2026-09-21 09:05:32.497000
"""
    pages = ["Next Page Token = TOK2\n" + LISTING, second]
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, pages[len(calls) - 1], "")

    monkeypatch.setattr(campaign_health.subprocess, "run", fake_run)

    seeds = campaign_health.parse_seed_listing(campaign_health.dataset_listing("o/ds"))

    assert len(calls) == 2
    assert "--page-token" not in calls[0]
    assert calls[1][calls[1].index("--page-token") + 1] == "TOK2"
    # The seed that was only on the second page must be visible, or the report
    # says the queue has not advanced.
    assert set(seeds) == {15000, 15002, 15004}


def test_published_seeds_reads_the_paginated_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wiring is the pin: a paginated helper nothing calls restores the defect.

    Testing ``dataset_listing`` alone leaves ``published_seeds`` free to keep
    shelling out for one page - which is what it did, and the mutation that
    reverted it passed the suite until this test existed.
    """
    called: list[str] = []

    def fake_listing(dataset_id: str, **kwargs) -> str:
        called.append(dataset_id)
        return LISTING

    def forbid_run(*args, **kwargs):
        raise AssertionError("published_seeds must not shell out for one page")

    monkeypatch.setattr(campaign_health, "dataset_listing", fake_listing)
    monkeypatch.setattr(campaign_health.subprocess, "run", forbid_run)

    seeds = campaign_health.published_seeds("o/ds")

    assert called == ["o/ds"]
    assert sorted(seeds) == [15000, 15002]


def test_published_seeds_reads_every_shard_and_forgives_a_missing_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shard that does not exist yet is the state before its first seed.

    The block now publishes across two datasets (7 seeds / 44.13 GiB landed;
    8 / ~50.4 GiB killed two containers). The first shard must be readable -
    if it is not, the report cannot say whether the queue started - while the
    continuation legitimately has no version until its first push. Reading
    that as an error would make the report fail at the moment it is most
    needed, and reading the shards as one merged set is what keeps the counts
    and the COMPLETE verdict honest.
    """

    def fake_listing(dataset_id: str, **kwargs) -> str:
        if dataset_id == "o/ds":
            return LISTING  # seeds 15000 and 15002
        if dataset_id == "o/ds-p2":
            return (
                "name  size  creationDate\n"
                "seed-15031/seed-15031/checkpoints/adaptive/checkpoint-006000.pt  1  2026-10-05\n"
            )
        raise RuntimeError(f"Dataset not found: {dataset_id}")

    monkeypatch.setattr(campaign_health, "dataset_listing", fake_listing)

    merged = campaign_health.published_seeds(["o/ds", "o/ds-p2", "o/ds-p3"])
    assert sorted(merged) == [15000, 15002, 15031]

    # A seed present in two shards means the rule leaked overlap, and the
    # report refuses to invent a priority between two published versions.
    monkeypatch.setattr(
        campaign_health,
        "dataset_listing",
        lambda dataset_id, **kwargs: LISTING,
    )
    with pytest.raises(RuntimeError, match="more than one shard"):
        campaign_health.published_seeds(["o/ds", "o/ds-p2"])


def test_a_repeated_page_token_raises_instead_of_spinning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hanging health report is worse than one that under-reports."""

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, "Next Page Token = SAME\n" + LISTING, "")

    monkeypatch.setattr(campaign_health.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="repeated page token"):
        campaign_health.dataset_listing("o/ds")


def test_the_padded_listing_parses_by_field_not_by_stripped_line() -> None:
    seeds = campaign_health.parse_seed_listing(LISTING)
    assert sorted(seeds) == [15000, 15002]
    assert seeds[15000]["arms"] == {"adaptive": 6000, "random": 6000}
    assert seeds[15002]["arms"] == {"adaptive": 3000, "random": 6000}


def test_a_seed_is_complete_only_with_every_arm_at_the_declared_step() -> None:
    """A snapshot that stopped part-way must not read as a finished seed."""
    progress = campaign_health.seed_progress(
        campaign_health.parse_seed_listing(LISTING), statuses={15000: "complete"}
    )
    assert progress["complete"] == [15000]
    assert progress["partial"] == [15002]
    assert progress["complete_count"] == 1
    # ``LISTING`` stays the 2026-09-21 capture of the superseded 15000-15011 block - it is
    # the evidence for the doubled upload layout, and section 5.65 re-reserved the block
    # rather than rewriting what was observed. ``next_seed`` is therefore about the *live*
    # block, none of which is complete, so it is the live block's first seed. The gap
    # semantics inside a partly-complete block are pinned separately, below.
    assert progress["next_seed"] == campaign_health.DECLARED_SEEDS[0]
    assert progress["declared"] == list(campaign_health.DECLARED_SEEDS)


def test_the_declared_step_is_necessary_but_not_enough_on_its_own() -> None:
    """Measured on the live block, 2026-09-27: seed 15027, stepped to 6000 and unfinished.

    Its listing carries ``checkpoint-006000.pt`` for **both** arms - a snapshot is written at
    every checkpoint and the window closed after the last one - while its own published record
    says ``status: in-progress`` with no final loss, because the evaluation that follows
    training never ran. The runner skips a seed only when that same record says ``complete``,
    so a report that called 15027 finished would disagree with the queue it describes: it
    said ``4 of 12 complete`` and ``resumes from seed 15028`` while the next session would
    start at 15027.
    """
    listing = "\n".join(
        f"seed-{seed}/seed-{seed}/checkpoints/{arm}/checkpoint-{step:06d}.pt"
        f"  1  2026-09-27 03:33:09.095000"
        for seed in (15026, 15027)
        for arm, step in (("adaptive", 6000), ("random", 6000))
    )

    progress = campaign_health.seed_progress(
        campaign_health.parse_seed_listing(listing),
        statuses={15026: "complete", 15027: "in-progress"},
    )

    assert progress["complete"] == [15026]
    assert progress["partial"] == []
    assert progress["unconfirmed"] == [15027]
    assert progress["unconfirmed_status"] == {15027: "in-progress"}
    assert progress["complete_count"] == 1
    # The seed the next session will actually start at, which is the seed the report has to
    # name: an unfinished-but-stepped seed is not one the queue has moved past.
    assert 15027 in progress["missing"]
    # The listing still calls both seeds *candidates* - the distinction the report needs to
    # make, and the reason the confirmation step exists rather than a stricter parser.
    assert campaign_health.stepped_seeds(campaign_health.parse_seed_listing(listing)) == [
        15026,
        15027,
    ]


def test_a_stepped_seed_whose_record_cannot_be_read_is_unconfirmed_not_complete() -> None:
    """A fetch that fails must not read as a finished seed - the two are different states.

    ``None`` is what both "no summary published yet" and "the download failed" answer with,
    and neither is evidence that the seed finished its evaluation. The bucket distinguishes
    it from an ``in-progress`` record so the report can say which one it is looking at.
    """
    seeds = {15027: {"arms": {"random": 6000, "adaptive": 6000}, "max_step": 6000}}
    progress = campaign_health.seed_progress(seeds, statuses={15027: None})

    assert progress["complete"] == []
    assert progress["unconfirmed"] == [15027]
    assert progress["unconfirmed_status"] == {15027: None}
    assert progress["next_seed"] == campaign_health.DECLARED_SEEDS[0]


def test_the_report_names_a_stepped_unfinished_seed_and_its_own_status() -> None:
    """The line an operator reads has to say which seed and *why* it is not counted."""
    seeds = {15027: {"arms": {"random": 6000, "adaptive": 6000}, "max_step": 6000}}
    progress = campaign_health.seed_progress(seeds, statuses={15027: "in-progress"})
    state = {
        "queue": "session",
        "session_status": "COMPLETE",
        "driver_lock": None,
        "quota": {
            "remaining_hours": 6.57,
            "used_hours": 23.43,
            "total_hours": 30.0,
            "refresh_at": "2026-10-03T00:00:00",
            "declared_window_hours": 6.5,
        },
        "seeds": {
            "complete": progress["complete"],
            "complete_count": progress["complete_count"],
            "declared_count": progress["declared_count"],
            "partial": progress["partial"],
            "unconfirmed": progress["unconfirmed"],
            "unconfirmed_status": progress["unconfirmed_status"],
            "next": progress["next_seed"],
            "error": None,
            "declared": len(campaign_health.DECLARED_SEEDS),
        },
        "readout": None,
        "verdict": "STALLED",
    }

    text = campaign_health.report(state)

    assert "0 of 12 complete" in text
    assert "stepped but not finished 15027 (in-progress)" in text
    assert "15027" in text


def test_next_seed_is_the_lowest_live_block_seed_not_complete() -> None:
    """Built from the live block, so a re-reservation cannot leave this stale."""
    first, second, third = campaign_health.DECLARED_SEEDS[:3]
    listing = "\n".join(
        f"seed-{seed}/seed-{seed}/checkpoints/{arm}/checkpoint-{step:06d}.pt"
        f"  1  2026-09-21 08:05:15.249000"
        for seed, step, arm in (
            (first, 6000, "random"),
            (first, 6000, "adaptive"),
            (third, 6000, "random"),  # the middle seed was never published at all
        )
    )

    progress = campaign_health.seed_progress(
        campaign_health.parse_seed_listing(listing), statuses={first: "complete"}
    )

    assert progress["complete"] == [first]
    assert progress["next_seed"] == second
    assert progress["missing"] == list(campaign_health.DECLARED_SEEDS[1:])


def test_the_completeness_threshold_comes_from_the_registered_config() -> None:
    """Not restated here: a seed is complete against the *registered* run's step count."""
    assert campaign_health.DECLARED_STEPS == 6000
    assert campaign_health.DECLARED_ARMS == ("random", "adaptive")
    assert len(campaign_health.DECLARED_SEEDS) == 12


def _state(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: str,
    remaining: float,
    complete: list[int] | None = None,
    window: float = 7.0,
    statuses: dict[int, str | None] | None = None,
    driver: dict | None = None,
    queue: str = "session",
    arms: tuple[str, ...] = ("random", "adaptive"),
    readout: bool = False,
) -> dict:
    seeds = {
        seed: {"arms": {arm: 6000 for arm in arms}, "max_step": 6000}
        for seed in (complete or [])
    }
    # Every candidate's *published record* has to be read before a seed can be called
    # complete, so the state fixture owns that answer too - and it defaults to the state that
    # published seeds are in on a healthy queue rather than to whatever the listing says.
    monkeypatch.setattr(
        campaign_health,
        "seed_statuses",
        lambda dataset_id, seeds, **kwargs: {
            seed: (statuses or {}).get(seed, "complete") for seed in seeds
        },
    )
    monkeypatch.setattr(campaign_health.kaggle_queue, "active_status", lambda kernel: status)
    monkeypatch.setattr(
        campaign_health.kaggle_queue,
        "active_quota",
        lambda *a, **k: {
            "remaining_hours": remaining,
            "used_hours": 30.0 - remaining,
            "total_hours": 30.0,
            "refresh_at": "2026-09-26T00:00:00",
        },
    )
    monkeypatch.setattr(campaign_health, "published_seeds", lambda dataset_id: seeds)
    # ``holder`` is the answer ``lock_holder`` gives: ``None`` when the probe took the lock
    # itself, i.e. when nobody is driving this queue. The fixture defaults to "nobody", and
    # a test that needs a waiting driver passes its pid.
    monkeypatch.setattr(
        campaign_health, "lock_holder", lambda kernel: driver
    )
    return campaign_health.health(queue, window=window, readout=readout)


def test_a_live_session_is_running(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state(monkeypatch, status="RUNNING", remaining=6.0, complete=[15000])
    assert state["verdict"] == "RUNNING"
    assert campaign_health.VERDICT_EXIT["RUNNING"] == 0


def test_the_health_state_names_every_shard_the_queue_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 7-of-12 block now comes from two dataset versions, and the report says so.

    The JSON used to name only the registered id, so a reader could not tell
    that a continuation exists (or that the counts above it were merged from
    two listings). The registered id is kept as ``dataset`` for the callers
    that already read it.
    """
    state = _state(
        monkeypatch,
        status="COMPLETE",
        remaining=6.0,
        complete=list(campaign_health.DECLARED_SEEDS),
    )

    assert state["dataset"] == "sid123123234/act-v0-5-2"
    assert state["datasets"] == [
        "sid123123234/act-v0-5-2",
        "sid123123234/act-v0-5-2-p2",
    ]


def test_every_declared_seed_published_is_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state(
        monkeypatch,
        status="COMPLETE",
        remaining=6.0,
        complete=list(campaign_health.DECLARED_SEEDS),
    )
    assert state["verdict"] == "COMPLETE"


def test_the_null_queue_is_complete_against_its_own_single_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The null queue is single-arm by construction, so a paired read is a false stall.

    Measured 2026-10-24: the null queue's first GPU session published four seeds, each with
    one ``null`` checkpoint at the declared step, and this report called all four ``partial``
    because it required ``adaptive`` and ``random``. The session's own log said ``4/12``.
    Two consequences follow from that disagreement: the launcher's ``step_is_complete`` can
    never skip a finished null queue (it would spend a restore-only session at the end), and
    the report an operator reads says the do-nothing clause has not started when it has.
    """
    block = list(campaign_health.DECLARED_SEEDS)
    state = _state(
        monkeypatch,
        status="COMPLETE",
        remaining=6.0,
        complete=block,
        queue="null",
        arms=("null",),
    )

    assert state["seeds"]["complete"] == block
    assert state["seeds"]["partial"] == []
    assert state["verdict"] == "COMPLETE"
    assert f"{len(block)} of {len(block)} complete" in campaign_health.report(state)


def test_a_stepped_but_unfinished_seed_keeps_the_complete_verdict_away(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The consequence of the 15027 misreading, at the level that decides to move on.

    The last seed a session leaves behind is the one whose snapshot lands after its final
    checkpoint, so this is the state a queue reaches *most often* - and ``verdict:
    COMPLETE`` is what says the next queue may be driven. It must not fire while the block's
    own record still calls a seed unfinished, or the control queue would start on top of an
    incomplete primary block.
    """
    block = list(campaign_health.DECLARED_SEEDS)
    state = _state(
        monkeypatch,
        status="COMPLETE",
        remaining=6.0,
        complete=block,
        statuses={block[-1]: "in-progress"},
    )

    assert state["seeds"]["complete"] == block[:-1]
    assert state["seeds"]["unconfirmed"] == [block[-1]]
    assert state["seeds"]["next"] == block[-1]
    assert state["verdict"] != "COMPLETE"
    assert "stepped but not finished" in campaign_health.report(state)


def test_health_confirms_the_candidates_and_leaves_the_partial_seeds_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wiring is the pin: confirming *every* published seed pays a download for seeds
    whose steps already answer the question, and confirming *none* is the defect this
    fixed.

    Measured 2026-09-27: a candidate's status costs 1.1-1.3 s (one small ``-f`` download),
    so the report stays cheap by asking only about seeds whose listing reaches the declared
    step on every arm.
    """
    stepped = list(campaign_health.DECLARED_SEEDS[:2])
    listing = "\n".join(
        f"seed-{seed}/seed-{seed}/checkpoints/{arm}/checkpoint-{step:06d}.pt"
        f"  1  2026-09-21 08:05:15.249000"
        for seed, arms in (
            (stepped[0], (("random", 6000), ("adaptive", 6000))),
            (stepped[1], (("random", 6000), ("adaptive", 6000))),
            # A seed that stopped part-way: already unfinished, so its record cannot change
            # the answer and it must not be fetched.
            (campaign_health.DECLARED_SEEDS[2], (("random", 6000), ("adaptive", 3000))),
        )
        for arm, step in arms
    )
    asked: list[list[int]] = []

    def fake_statuses(dataset_id: str, seeds, **kwargs) -> dict:
        asked.append(list(seeds))
        return {seed: "complete" for seed in seeds}

    monkeypatch.setattr(campaign_health, "published_seeds", lambda dataset_id: campaign_health.parse_seed_listing(listing))
    monkeypatch.setattr(campaign_health, "seed_statuses", fake_statuses)
    monkeypatch.setattr(campaign_health.kaggle_queue, "active_status", lambda kernel: "COMPLETE")
    monkeypatch.setattr(
        campaign_health.kaggle_queue,
        "active_quota",
        lambda *a, **k: {
            "remaining_hours": 6.0,
            "used_hours": 24.0,
            "total_hours": 30.0,
            "refresh_at": "2026-10-03T00:00:00",
        },
    )
    monkeypatch.setattr(campaign_health, "lock_holder", lambda kernel: None)

    state = campaign_health.health("session")

    assert asked == [stepped]
    assert state["seeds"]["complete"] == stepped
    assert state["seeds"]["partial"] == [campaign_health.DECLARED_SEEDS[2]]
    assert state["seeds"]["unconfirmed"] == []


def test_an_error_with_nothing_live_is_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The state both 2026-09 losses ended in: read the cause before pushing again."""
    state = _state(monkeypatch, status="ERROR", remaining=6.0, complete=[15000])
    assert state["verdict"] == "FAILED"


def test_an_unreadable_dataset_says_cannot_tell_and_names_the_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measured 2026-09-25, on this report's own output, at the moment of deciding to push.

    The live block's dataset (`act-v0-5-2`) has no versions until its first session
    publishes, so the listing raised, the counts became ``None`` and the reason was dropped:
    the line read "published seeds : None of None complete". That is not a state - it is a
    report bug - and the two states it conflated are the ones an operator has to tell apart:
    a dataset that is *unreadable* (ask again, or read the cause) and one that is *empty*
    (nothing has run yet, which is expected for a re-reserved block).
    """

    def unreadable(dataset_id: str) -> dict:
        raise RuntimeError(f"act-v0-5-2 has no versions yet")

    monkeypatch.setattr(campaign_health, "published_seeds", unreadable)
    monkeypatch.setattr(campaign_health.kaggle_queue, "active_status", lambda kernel: "ERROR")
    monkeypatch.setattr(
        campaign_health.kaggle_queue,
        "active_quota",
        lambda: {
            "remaining_hours": 1.07,
            "used_hours": 28.93,
            "total_hours": 30.0,
            "refresh_at": "2026-09-26T00:00:00",
        },
    )
    monkeypatch.setattr(campaign_health, "lock_holder", lambda kernel: None)

    state = campaign_health.health("session")
    text = campaign_health.report(state)
    assert "unreadable" in text
    assert "act-v0-5-2 has no versions yet" in text
    # The denominator is a registered constant, so it survives the read failure; without it
    # an operator cannot tell how far the block is *meant* to go.
    assert f"of {len(campaign_health.DECLARED_SEEDS)} declared" in text
    assert "None of None" not in text
    assert "None" not in text
    # The verdict still follows the kernel's status, which is what it has to act on.
    assert state["verdict"] == "FAILED"


def test_nothing_live_with_no_quota_is_waiting_not_stalled(monkeypatch: pytest.MonkeyPatch) -> None:
    """A week that cannot fit a session is expected; the driver's own wait covers it."""
    state = _state(monkeypatch, status="COMPLETE", remaining=3.0, complete=[15000])
    assert state["verdict"] == "WAITING_ON_QUOTA"
    assert campaign_health.VERDICT_EXIT["WAITING_ON_QUOTA"] == 2


def test_a_quota_wait_says_whether_a_driver_is_actually_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measured 2026-09-27: the state a reboot leaves behind read like a healthy wait.

    The workstation rebooted while the driver was sleeping to the refresh, and the session
    that driver had already pushed completed unattended - three more seeds published, 6/12.
    Nothing was waiting for the refresh any more, and the report still said
    ``WAITING_ON_QUOTA`` with the action text that names the driver's own wait, because the
    lock only appeared on the kernel line when it was *held*. Both states print the same
    verdict (the quota is short either way, and the exit code is already documented as 2),
    so the difference has to be in the words and in the lock line.
    """
    idle = _state(monkeypatch, status="COMPLETE", remaining=1.65, complete=[15024])
    text = campaign_health.report(idle)
    assert idle["verdict"] == "WAITING_ON_QUOTA"
    assert "driver lock     : not held" in text
    assert "nothing resumes at the refresh until the campaign is relaunched" in text

    waiting = _state(
        monkeypatch,
        status="COMPLETE",
        remaining=1.65,
        complete=[15024],
        driver={"pid": 4242},
    )
    text = campaign_health.report(waiting)
    assert "driver lock     : held by pid 4242" in text
    assert "a driver is waiting on it (pid 4242)" in text
    assert "relaunched" not in text


def test_a_lock_whose_holder_cannot_be_read_still_says_a_driver_is_there(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The name is the useful part, not the condition: an unreadable holder is still held."""
    state = _state(
        monkeypatch,
        status="COMPLETE",
        remaining=1.65,
        complete=[15024],
        driver={"detail": "an unidentified process"},
    )
    text = campaign_health.report(state)
    assert "driver lock     : held by an unidentified process" in text
    assert "a driver holds the lock and is waiting on it" in text


def test_nothing_live_with_quota_available_is_stalled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one verdict that means somebody has to act, and why it names the quota."""
    state = _state(monkeypatch, status="COMPLETE", remaining=22.0, complete=[15000])
    assert state["verdict"] == "STALLED"
    assert campaign_health.VERDICT_EXIT["STALLED"] == 1
    assert "STALLED" in campaign_health.report(state)


def _summary(*, random: float, adaptive: float) -> dict:
    return {
        "arms": {
            "random": {"final_loss": random},
            "adaptive": {"final_loss": adaptive},
        }
    }


def test_the_readout_fetches_the_path_the_upload_created(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The remote path carries the doubled component, and each seed needs its own directory.

    Both were measured on 2026-09-21: a seed is pushed as ``seed-N.tar`` whose members
    already begin ``seed-N/``, so the summary lives at ``seed-N/seed-N/reports/...``; and
    because the CLI writes a fetched file under its own basename, four seeds downloaded
    into one directory leave **one** file - the last one - which would have read as three
    unknown seeds.
    """
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        directory = Path(argv[argv.index("-p") + 1])
        (directory / "seed-summary.json").write_text(
            json.dumps(_summary(random=1.0, adaptive=0.5)), encoding="utf-8"
        )
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(campaign_health.subprocess, "run", fake_run)
    read = campaign_health.seed_readout("o/d", [15000, 15001])

    assert [call[call.index("-f") + 1] for call in calls] == [
        "seed-15000/seed-15000/reports/seed-summary.json",
        "seed-15001/seed-15001/reports/seed-summary.json",
    ]
    directories = [call[call.index("-p") + 1] for call in calls]
    assert len(set(directories)) == 2, "both seeds were fetched into one directory"
    # And the seed is visible in the directory name, so a failed download names its seed.
    for seed, directory in zip((15000, 15001), directories, strict=True):
        assert str(seed) in Path(directory).name
    assert read["seeds_read"] == [15000, 15001]
    assert read["adaptive_wins"] == 2
    assert read["mean_adaptive_minus_random"] == -0.5


def test_the_readout_reads_the_nested_layout_the_current_cli_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI 2.2.4 preserves the remote path under ``-p``; 2.0.2 wrote the basename.

    Measured 2026-10-24: the workstation's CLI wrote
    ``<dir>/seed-15024/seed-15024/reports/seed-summary.json``, the flat-only reader looked
    for ``<dir>/seed-summary.json``, and all twelve complete control seeds read as "record
    unreadable" - a report that says a finished block has not advanced. Both layouts are
    live in this project (the sessions run 2.0.2), so both are read.
    """
    def fake_run(argv, **kwargs):
        directory = Path(argv[argv.index("-p") + 1])
        name = argv[argv.index("-f") + 1]
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_summary(random=1.0, adaptive=0.5)), encoding="utf-8"
        )
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(campaign_health.subprocess, "run", fake_run)
    read = campaign_health.seed_readout("o/d", [15000])

    assert read["seeds_read"] == [15000]
    assert read["seeds_unread"] == []
    assert read["mean_adaptive_minus_random"] == -0.5


def test_the_readout_reports_inputs_and_says_it_is_not_a_verdict(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """An incomplete queue cannot produce the registered verdict, so it is not attempted.

    The thresholds live in the pre-registered rule, and applying one here would be the
    provisional verdict that pre-registration exists to prevent - so the block reports the
    differences and the win count and stops.
    """
    summaries = {
        15000: _summary(random=10.3977, adaptive=10.3975),
        15001: _summary(random=10.3976, adaptive=10.3982),
    }
    monkeypatch.setattr(
        campaign_health, "fetch_seed_summary", lambda *a, **k: summaries.get(a[1])
    )
    read = campaign_health.seed_readout("o/d", [15000, 15001])
    lines = "\n".join(campaign_health._readout_lines(read))

    assert read["adaptive_wins"] == 1 and read["adaptive_losses"] == 1
    assert "not a verdict" in lines
    assert "no threshold is applied here" in lines
    assert "adaptive won 1 of 2" in lines
    assert "10.3977" in lines


def test_the_null_readout_reports_its_single_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The do-nothing queue has no adaptive-minus-random difference to report.

    This goes through ``health`` rather than calling ``seed_readout`` directly, because the
    wiring is the part that broke: ``health`` has to pass the queue's own arms down. Read as
    a pair, every published null seed comes back unreadable - the same false stall that made
    the queue's completeness wrong.
    """
    summaries = {15000: {"arms": {"null": {"final_loss": 10.3980}}}}
    monkeypatch.setattr(
        campaign_health, "fetch_seed_summary", lambda *a, **k: summaries.get(a[1])
    )
    state = _state(
        monkeypatch,
        status="COMPLETE",
        remaining=6.0,
        complete=[15000],
        queue="null",
        arms=("null",),
        readout=True,
    )
    lines = campaign_health.report(state)

    assert state["readout"]["arms"] == ["null"]
    assert state["readout"]["adaptive_wins"] is None
    assert state["readout"]["mean_adaptive_minus_random"] is None
    assert "published null seed(s)" in lines
    assert "null 10.3980" in lines
    assert "not a verdict" in lines


def test_a_complete_single_arm_queue_points_at_the_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'Drive the next queue' is wrong once the null block is finished.

    The null queue is the last training queue, so the instruction after it is the free CPU
    verdict over all three. Firing another session would spend GPU hours on nothing.
    """
    block = list(campaign_health.DECLARED_SEEDS)
    state = _state(
        monkeypatch,
        status="COMPLETE",
        remaining=6.0,
        complete=block,
        queue="null",
        arms=("null",),
    )
    lines = campaign_health.report(state)

    assert state["verdict"] == "COMPLETE"
    assert "last training queue" in lines
    assert "drive the next queue" not in lines


def test_the_floor_is_the_registered_vocabulary() -> None:
    """The block trains on synthetic tokens, so ln(vocab) is the loss with nothing to learn."""
    assert campaign_health.DECLARED_VOCAB == 32768
    assert campaign_health.UNIFORM_PRIOR_NATS == pytest.approx(10.3972, abs=5e-5)
    assert "ln(32768)" in "\n".join(
        campaign_health._readout_lines(
            {
                "rows": [
                    {
                        "seed": 15000,
                        "losses": {"random": 10.3977, "adaptive": 10.3975},
                        "adaptive_minus_random": -0.0002,
                    }
                ],
                "seeds_unread": [],
                "adaptive_wins": 1,
                "mean_adaptive_minus_random": -0.0002,
                "largest_gap_from_uniform": 0.0005,
                "uniform_prior_nats": 10.3972,
            }
        )
    )


def test_a_seed_that_cannot_be_read_is_named_not_silently_dropped(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """A seed with no readable summary must not read as a seed with no difference."""
    monkeypatch.setattr(campaign_health, "fetch_seed_summary", lambda *a, **k: None)
    read = campaign_health.seed_readout("o/d", [15000, 15001])

    assert read["rows"] == [] and read["seeds_unread"] == [15000, 15001]
    assert "could not be read" in "\n".join(campaign_health._readout_lines(read))


def test_the_readout_is_opt_in() -> None:
    """The health check is meant to be cheap; the readout costs a download per seed."""
    source = (ROOT / "scripts" / "campaign_health.py").read_text(encoding="utf-8")
    assert '"--readout"' in source
    assert "readout: bool = False" in source


def test_a_queue_whose_kernel_is_not_pushed_yet_says_so_instead_of_printing_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control and null queues start with a kernel that does not exist yet.

    ``kaggle kernels status`` then yields no token, and the line printed nothing at all -
    which reads as a rendering fault rather than as the state the driver is about to act on
    (``should_adopt("")`` means *push*). An empty status is also what a failed query returns,
    and the report cannot tell those apart, so it names both.
    """
    state = _state(monkeypatch, status="", remaining=6.41, window=6.5)
    text = campaign_health.report(state)

    assert "session kernel  : " + campaign_health.UNKNOWN_KERNEL_NOTE in text
    assert "session kernel  : \n" not in text


def test_the_campaign_script_reports_the_window_the_sessions_will_use() -> None:
    """The launcher fits HOURS to the week; a report testing the declared 7 h would lie.

    Measured 2026-09-27, with the week's tail at 6.41 h and the driver pushing 6.5 h
    windows: the boundary report said "cannot fit a 7.0 h session" - a verdict computed
    against a window no session uses, printed at the moment an operator decides whether the
    campaign is stuck.
    """
    source = (ROOT / "scripts" / "run_gate4_campaign.sh").read_text(encoding="utf-8")
    health = source.split("campaign_health.py", 1)[1].split("2>&1", 1)[0]
    assert '"${health[@]}"' in health
    assert 'health+=(--window "$HOURS")' in source


def test_the_verdict_is_the_exit_code() -> None:
    """A cron or a later session acts on the code, not on the prose."""
    assert set(campaign_health.VERDICT_EXIT) == {
        "RUNNING",
        "STALLED",
        "WAITING_ON_QUOTA",
        "FAILED",
        "COMPLETE",
    }
    assert len(set(campaign_health.VERDICT_EXIT.values())) == 5
