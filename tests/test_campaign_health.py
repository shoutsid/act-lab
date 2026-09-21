"""Pin the campaign health report: what it reads, and which verdict each state gets.

The report exists because two sessions were lost while nobody could tell the campaign had
stopped without investigating by hand - so the part worth pinning is the *verdict*, which is
what decides whether somebody is paged to relaunch it. The parsing is pinned too, with the
real listing shape: it is a padded table, and reading the stripped line rather than the first
field silently matches nothing, which reads as "no seeds published" - the one wrong answer
that would look like a healthy stalled campaign.
"""

from __future__ import annotations

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
    assert progress["next_seed"] == 15001  # the lowest declared seed not complete


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
