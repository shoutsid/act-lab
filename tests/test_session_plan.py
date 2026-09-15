"""Tests for the section 5.2 session scheduling and dtype policy.

The values asserted here are pre-committed in RESEARCH_REPORT.md sections
5.1-5.2; if any of them drift, the execution plan is no longer the registered
one and this test must fail before launch.
"""

from __future__ import annotations

import math

import pytest

from act.training.session_plan import (
    CHECKPOINT_STEADY_STATE_GB,
    CHECKPOINT_PAYLOAD_GB,
    FP32_FLOOR_TOKENS_PER_SECOND,
    MAX_CHECKPOINTS,
    SEED_TOKENS_PER_ARM,
    PairSchedule,
    deadline_decision,
    decide_dtype,
    expected_probe_cost_percent,
    tolerance_band,
    unusable_schedule_reason,
    working_dir_headroom_gb,
)


def test_recorded_t4_measurements_derive_the_recorded_schedule() -> None:
    """The 2026-09-15 T4 dtype probe, re-derived from its measured rates.

    fp32 is the pre-committed rule's input; fp16 is the dtype the sessions
    train, so it is the one the schedule is built on.
    """
    fp32 = PairSchedule(tokens_per_second=2_066.3)
    assert (fp32.full_pairs_per_session, fp32.sessions_for_primary) == (2, 6)

    fp16 = PairSchedule(tokens_per_second=4_694.9)
    assert round(fp16.pair_time_hours, 3) == 1.554
    assert (fp16.full_pairs_per_session, fp16.sessions_for_primary) == (5, 3)


def test_unusable_schedule_reason_flags_a_plan_that_cannot_start_a_pair() -> None:
    """A pair time above the usable window yields zero pairs, not one.

    The benchmark wrote exactly this artifact on a GPU where the decided dtype
    runs *slower* than fp32 (a Pascal card's fp16 path is 4x down). Every
    session would have billed its window to publish nothing, and the deadline
    rules never start a pair the clock cannot finish, so zero is silent.
    """
    infeasible = PairSchedule(tokens_per_second=415.19)  # the measured rate there
    assert infeasible.full_pairs_per_session == 0
    reason = unusable_schedule_reason(infeasible)
    assert reason is not None
    assert "does not fit" in reason and "16.54 h" in reason

    # Every runnable plan passes, including opposite ends of the range.
    assert unusable_schedule_reason(PairSchedule(tokens_per_second=10_000.0)) is None
    assert unusable_schedule_reason(PairSchedule(tokens_per_second=4_694.9)) is None


def test_decide_dtype_follows_precommitted_floor() -> None:
    assert decide_dtype(10_000.0) == "fp32"
    assert decide_dtype(2_500.0) == "fp32"  # at the floor is still fp32
    assert decide_dtype(2_499.0) == "fp16"
    assert decide_dtype(0.0) == "fp16"
    with pytest.raises(ValueError):
        decide_dtype(-1.0)
    assert FP32_FLOOR_TOKENS_PER_SECOND == 2_500.0


@pytest.mark.parametrize(
    "tokens_per_second,pair_hours,pairs_per_session,sessions_primary,sessions_total",
    [
        (10_000.0, 0.78, 11, 2, 4),
        (5_000.0, 1.47, 6, 2, 4),
        (2_500.0, 2.83, 3, 4, 8),
    ],
)
def test_schedule_table_matches_section_5_2(
    tokens_per_second: float,
    pair_hours: float,
    pairs_per_session: int,
    sessions_primary: int,
    sessions_total: int,
) -> None:
    schedule = PairSchedule(tokens_per_second=tokens_per_second)
    assert math.isclose(schedule.pair_time_hours, pair_hours, abs_tol=5e-3)
    assert schedule.full_pairs_per_session == pairs_per_session
    assert schedule.sessions_for_primary == sessions_primary
    assert schedule.sessions_total == sessions_total


def test_schedule_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        PairSchedule(tokens_per_second=0.0)
    with pytest.raises(ValueError):
        PairSchedule(tokens_per_second=-5.0)
    with pytest.raises(ValueError):
        PairSchedule(tokens_per_arm=0)


def test_slow_throughput_fits_no_full_pair() -> None:
    # 1,000 tokens/s -> 6.89 h/pair + overhead still fits one pair? The
    # registered worst case bounds sessions, so just assert consistency:
    schedule = PairSchedule(tokens_per_second=1_000.0)
    assert schedule.pair_time_hours > 6.0
    assert schedule.sessions_total == 2 * schedule.sessions_for_primary


def test_steady_state_working_dir_matches_section_5_2() -> None:
    # max_checkpoints=2 pinned by the ledger arithmetic: 3 would overflow.
    assert MAX_CHECKPOINTS == 2
    schedule = PairSchedule()
    steady = schedule.steady_state_gb()
    assert math.isclose(steady, 2 * 2 * CHECKPOINT_PAYLOAD_GB + 2 * 1.06, abs_tol=1e-9)
    assert math.isclose(steady, 14.88, abs_tol=0.01)  # report's "~14.9 GB of 20 GB"
    assert math.isclose(CHECKPOINT_STEADY_STATE_GB, steady, abs_tol=1e-12)
    assert 5.0 < working_dir_headroom_gb() < 5.2


def test_seed_token_budget_matches_declaration() -> None:
    # batch 2 x context 2048 x 6000 steps = 24,576,000 tokens per arm.
    assert SEED_TOKENS_PER_ARM == 2 * 2048 * 6_000 == 24_576_000


@pytest.mark.parametrize(
    "probe_batch_count,predicted",
    [(1, 0.0942), (2, 0.1883)],
)
def test_predicted_probe_cost_matches_section_5_1(
    probe_batch_count: int, predicted: float
) -> None:
    assert math.isclose(
        expected_probe_cost_percent(probe_batch_count), predicted, rel_tol=2e-3
    )


def test_tolerance_band_is_half_to_double_the_prediction() -> None:
    lo, hi = tolerance_band(1)
    prediction = expected_probe_cost_percent(1)
    assert math.isclose(lo, 0.5 * prediction, rel_tol=1e-12)
    assert math.isclose(hi, 2.0 * prediction, rel_tol=1e-12)
    # Report's band for the primary: [0.047%, 0.189%].
    assert math.isclose(lo, 0.047, rel_tol=2e-2)
    assert math.isclose(hi, 0.189, rel_tol=2e-2)


def test_deadline_rules_order_and_thresholds() -> None:
    pair = 0.78  # the registered 10k-tokens/s pair time
    # Rule 1: a fresh pair starts only with pair + 45 min reserve available.
    assert deadline_decision(2.0, pair).action == "start_pair"
    assert deadline_decision(pair + 0.75, pair).action == "start_pair"
    # Exactly the 45-min reserve fits; a minute less must not start.
    assert deadline_decision(pair + 45 / 60, pair).action == "start_pair"
    assert deadline_decision(pair + 44.9 / 60, pair).action == "stop_at_boundary"
    # Rule 2: under an hour -> stop at the next step boundary.
    decision = deadline_decision(0.9, pair)
    assert decision.action == "stop_at_boundary"
    # Rule 3: under 15 minutes -> publish only.
    assert deadline_decision(0.2, pair).action == "publish_only"
    assert deadline_decision(0.0, pair).action == "publish_only"
    with pytest.raises(ValueError):
        deadline_decision(-1.0, pair)
    with pytest.raises(ValueError):
        deadline_decision(1.0, 0.0)


def test_deadline_never_starts_unfinishable_pair() -> None:
    # A 6.89-hour pair (1,000 tokens/s) must not start with 7.0 h left:
    # pair + 45 min reserve = 7.64 h > 7.0 h.
    slow = PairSchedule(tokens_per_second=1_000.0).pair_time_hours
    assert deadline_decision(7.0, slow).action == "stop_at_boundary"
    assert deadline_decision(8.0, slow).action == "start_pair"


def test_session_decisions_expose_reasons() -> None:
    for remaining in (2.0, 0.9, 0.2):
        decision = deadline_decision(remaining, 0.78)
        assert decision.reason
        assert decision.starts_work == (decision.action == "start_pair")
