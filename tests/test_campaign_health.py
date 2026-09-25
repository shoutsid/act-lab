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
    progress = campaign_health.seed_progress(campaign_health.parse_seed_listing(LISTING))
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

    progress = campaign_health.seed_progress(campaign_health.parse_seed_listing(listing))

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
) -> dict:
    seeds = {
        seed: {"arms": {"random": 6000, "adaptive": 6000}, "max_step": 6000}
        for seed in (complete or [])
    }
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
    monkeypatch.setattr(campaign_health, "lock_holder", lambda kernel: None)
    return campaign_health.health("session", window=window)


def test_a_live_session_is_running(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state(monkeypatch, status="RUNNING", remaining=6.0, complete=[15000])
    assert state["verdict"] == "RUNNING"
    assert campaign_health.VERDICT_EXIT["RUNNING"] == 0


def test_every_declared_seed_published_is_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state(
        monkeypatch,
        status="COMPLETE",
        remaining=6.0,
        complete=list(campaign_health.DECLARED_SEEDS),
    )
    assert state["verdict"] == "COMPLETE"


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
