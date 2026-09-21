"""Tests for the launch artifact's plan fields (``scripts/v0_throughput_benchmark.py``).

The artifact is launch input: a session reads its pair time and schedule from
it, and the campaign's total cost is quoted from it. The two things worth
pinning are that the campaign fields are *derived* from the measured rate and
placement rather than restated, and that the null queue is priced at declared
fidelity - the 2026-09-20 correction, where a "~90 s/seed" figure from a
dev-fidelity study had been carried as the declared queue's cost.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import v0_throughput_benchmark  # noqa: E402

from act.training.session_plan import (  # noqa: E402
    DECLARED_SESSION_HOURS,
    PairSchedule,
    session_pair_capacity,
)


def test_the_artifact_campaign_is_three_symmetric_queues() -> None:
    """At the measured T4 placement rate the artifact states nine sessions.

    The rate and placement are the ones session 1 measured (2026-09-20,
    `--queue dtypeprobe` on the T4 x2 shape): a pair is 1.443 h, the runner's
    start gate packs 5 per session (5+5+2), and the null queue - a single arm
    over the same 24.576M tokens - packs and costs the same. The training floor
    is 3 x 12 x 1.443 h = 51.9 h, which is above the superseded "~47 GPU-hour
    plan"; any composition that keeps that number fails here.
    """
    schedule = PairSchedule(tokens_per_second=5_083.4, arms=2, arm_slots=2)
    fields = v0_throughput_benchmark.campaign_plan_fields(schedule)
    assert fields["three_queue_sessions_primary"] == 3
    assert fields["three_queue_sessions_null"] == 3
    assert fields["three_queue_sessions"] == 9
    assert math.isclose(fields["three_queue_training_hours"], 51.95, abs_tol=5e-2)
    assert fields["three_queue_training_hours"] > 47.0


def test_the_artifact_plans_at_the_declared_window() -> None:
    """The artifact's plan is the window sessions are given (2026-09-20).

    Section 5.2's registered 9 h window stays the registered arithmetic the
    schedule table is pinned to, but a plan composed at it describes sessions
    that will not be run: the platform killed session 1 at 8 h, before its own
    9 h deadline, and took every pair it had trained. The artifact therefore
    prices the campaign at the declared clock - four pairs per session instead
    of five - and records both the window and the measured cap it came from, so
    a plan can be read without knowing which constants were live on the day.
    """
    schedule = PairSchedule(tokens_per_second=5_083.4, arms=2, arm_slots=2)
    fields = v0_throughput_benchmark.campaign_plan_fields(schedule)
    assert fields["declared_session_hours"] == DECLARED_SESSION_HOURS == 7.0
    assert fields["platform_session_cap_hours"] == 8.0
    assert session_pair_capacity(schedule.pair_time_hours, DECLARED_SESSION_HOURS) == 4
    # A pair per session less, and the same campaign: ceil(12/4) == ceil(12/5)
    # == 3 sessions for each of the three queues, over the same 51.95 h floor.
    assert fields["three_queue_sessions_primary"] == 3
    assert fields["three_queue_sessions"] == 9
    assert math.isclose(fields["three_queue_training_hours"], 51.95, abs_tol=5e-2)


def test_the_artifact_campaign_keeps_the_null_queue_at_one_wave() -> None:
    """On a one-device host the pair queues halve, the null queue does not.

    Sequential arms cost two waves per pair (2 x 1.3429 + 0.1 = 2.7859 h,
    capacity 2 per session, six sessions each) while the null seed is one arm
    and keeps five per session over three sessions. Deriving the null time from
    the pair time - rather than from the pair schedule - is what this pins, and
    it is why the campaign total cannot be read off the primary queue alone.
    """
    schedule = PairSchedule(tokens_per_second=5_083.4, arms=2, arm_slots=1)
    fields = v0_throughput_benchmark.campaign_plan_fields(schedule)
    assert fields["three_queue_sessions_primary"] == 6
    assert fields["three_queue_sessions_null"] == 3
    assert fields["three_queue_sessions"] == 15
    assert math.isclose(fields["three_queue_training_hours"], 84.2, abs_tol=5e-2)


def test_the_artifact_campaign_refuses_a_schedule_that_starts_nothing() -> None:
    # The benchmark refuses an unusable schedule with a clean SystemExit before
    # this helper is reached; called directly it raises rather than inventing a
    # zero-session plan.
    unusable = PairSchedule(tokens_per_second=100.0, arms=2, arm_slots=2)
    with pytest.raises(ValueError, match="no pair per session"):
        v0_throughput_benchmark.campaign_plan_fields(unusable)
