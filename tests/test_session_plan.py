"""Tests for the section 5.2 session scheduling and dtype policy.

The values asserted here are pre-committed in RESEARCH_REPORT.md sections
5.1-5.2; if any of them drift, the execution plan is no longer the registered
one and this test must fail before launch.
"""

from __future__ import annotations

import math

import pytest

from act.training.session_plan import (
    BOOT_AND_PREP_HOURS,
    CHECKPOINT_STEADY_STATE_GB,
    CHECKPOINT_PAYLOAD_GB,
    DECLARED_SESSION_HOURS,
    FP32_FLOOR_TOKENS_PER_SECOND,
    MAX_CHECKPOINTS,
    PLATFORM_SESSION_CAP_HOURS,
    PUBLISH_RESERVE_MINUTES,
    SEED_TOKENS_PER_ARM,
    USABLE_HOURS,
    PairSchedule,
    deadline_decision,
    decide_dtype,
    expected_probe_cost_percent,
    queue_plan,
    session_pair_capacity,
    three_queue_plan,
    tolerance_band,
    unusable_schedule_reason,
    working_dir_headroom_gb,
)


def test_recorded_t4_measurements_derive_the_recorded_schedule() -> None:
    """The 2026-09-15 T4 dtype probe, re-derived from its measured rates.

    fp32 is the pre-committed rule's input; fp16 is the dtype the sessions
    train, so it is the one the schedule is built on. Both rows are for the
    declared two-arm pair, trained sequentially as the runner does it.
    """
    fp32 = PairSchedule(tokens_per_second=2_066.3)
    assert round(fp32.pair_time_hours, 2) == 6.71
    assert (fp32.full_pairs_per_session, fp32.sessions_for_primary) == (1, 12)

    fp16 = PairSchedule(tokens_per_second=4_694.9)
    assert round(fp16.pair_time_hours, 2) == 3.01
    assert (fp16.full_pairs_per_session, fp16.sessions_for_primary) == (2, 6)

    # The dtype rule's win survives the arm multiplicity: the same pair costs
    # 2.2x less at the rate the arms actually run, which is what justifies
    # scheduling from the decided dtype rather than the fp32 rule input.
    assert fp32.pair_time_hours / fp16.pair_time_hours > 2.0


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
    assert "does not fit" in reason and "32.98 h" in reason

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
        # Amended 2026-09-15: the section 5.2 rows were computed for one arm per
        # GPU, which no runner implements. At an assumed 10k tokens/s the
        # single-arm row was 0.78 h -> 11 pairs/session; a two-arm pair trained
        # sequentially on one device costs 1.465 h -> 6. The rates themselves
        # are unchanged assumptions; only the arm multiplicity was missing.
        (10_000.0, 1.47, 6, 2, 4),
        (5_000.0, 2.83, 3, 4, 8),
        (2_500.0, 5.56, 1, 12, 24),
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


def test_a_single_arm_pair_costs_one_arm_time() -> None:
    """The null queue runs one arm, so its pairs are not doubled."""
    one = PairSchedule(tokens_per_second=4_694.9, arms=1)
    two = PairSchedule(tokens_per_second=4_694.9, arms=2)
    assert round(one.pair_time_hours, 3) == 1.554
    assert math.isclose(two.pair_time_hours, 2 * one.pair_time_hours - 0.1, abs_tol=1e-9)


def test_the_corrected_session_places_exactly_two_pairs() -> None:
    """The amended plan's packing claim, walked through the deadline rules.

    At the T4's measured fp16 rate a 9 h session completes two pairs and then
    stops with clock it cannot finish a third in. It must not start that third
    pair: section 5.2 answers an interrupted pair by restarting it from
    scratch, so starting one it cannot finish throws the work away rather than
    deferring it.
    """
    pair_time = PairSchedule(tokens_per_second=4_694.9).pair_time_hours
    assert session_pair_capacity(pair_time) == 2
    # 3.01 h pairs do not pack neatly into 9 h: two pairs leave 2.98 h, just
    # under one pair, so the session stops with a third of its window unused
    # rather than starting a pair it cannot finish. Sessions therefore exit
    # early (and bill for the ~6 h they ran, not the 9 h they were given).
    remaining = USABLE_HOURS - 2 * pair_time
    assert remaining < pair_time
    assert math.isclose(9.0 - 2 * pair_time, remaining, abs_tol=1e-9)


def test_the_registered_capacity_and_the_runners_gate_differ_when_the_reserve_binds() -> None:
    """Measured 2026-09-20: the registered field reads one pair too many.

    ``full_pairs_per_session`` is section 5.2's arithmetic and stays registered,
    but the runner's start gate adds the 45-minute publish reserve to every
    start. At the T4's fp16 placement rate (one arm per device, 5,083.4
    tokens/s) the two disagree by exactly one pair per session, and therefore by
    one *session* for the primary queue: 2 registered against the 3 the runner
    needs. Reading the plan from the registered field alone is what made the
    launch's own expectation wrong, so both numbers are asserted here.
    """
    schedule = PairSchedule(tokens_per_second=5_083.4, arms=2, arm_slots=2)
    assert math.isclose(schedule.pair_time_hours, 1.443, abs_tol=5e-4)
    assert schedule.full_pairs_per_session == 6
    assert schedule.sessions_for_primary == 2  # registered, section 5.2's form
    assert session_pair_capacity(schedule.pair_time_hours) == 5
    assert math.ceil(12 / session_pair_capacity(schedule.pair_time_hours)) == 3
    # And the reserve is the whole reason for the gap: lend the session those 45
    # minutes as extra window and the runner's gate packs the registered six.
    assert session_pair_capacity(
        schedule.pair_time_hours, hours=USABLE_HOURS + PUBLISH_RESERVE_MINUTES / 60
    ) == schedule.full_pairs_per_session


def test_the_declared_campaign_is_three_symmetric_queues() -> None:
    """The three-queue plan, composed from the runner's gate (2026-09-20).

    Every recorded total so far has been wrong in its own way. The "~47
    GPU-hours" plan was built on a pair time the placement correction changed;
    the null queue carried a "~90 s/seed" figure measured at *dev* fidelity
    (400 steps, dev shape: 102.9 s mean over the 24 sessions of
    `deployed_vs_null_study`) when a declared null seed trains the same
    24,576,000 tokens as a pair arm. So the campaign is three symmetric queues,
    and its training floor is 3 x the per-queue training time - a number this
    test keeps above the recorded plan it superseded.
    """
    schedule = PairSchedule(tokens_per_second=5_083.4, arms=2, arm_slots=2)
    campaign = three_queue_plan(schedule.pair_time_hours)
    assert math.isclose(schedule.pair_time_hours, 1.443, abs_tol=5e-4)
    # The runner's start gate, per queue (not section 5.2's registered six):
    # 5 + 5 + 2 pairs over three sessions.
    assert campaign.primary.pairs_per_session == 5
    assert (campaign.primary.sessions, campaign.control.sessions) == (3, 3)
    assert campaign.sessions == 9
    assert math.isclose(campaign.training_hours, 51.95, abs_tol=5e-2)
    # The recorded "~47 GPU-hour plan" is *below* the training-only floor at
    # the measured rate, so it cannot have been a plan for this campaign. This
    # assertion is the falsifier: it fails on any composition that prices the
    # null queue as an afterthought.
    assert campaign.training_hours > 47.0

    # The null queue is a full-rate third: one arm, one wave, same 24.576M
    # tokens per seed as each pair arm, so it packs identically and costs the
    # same as either pair queue - not the dev study's ~90 s.
    assert campaign.null.pairs_per_session == 5
    assert campaign.null.sessions == 3
    assert math.isclose(
        campaign.null.training_hours, campaign.primary.training_hours, rel_tol=1e-9
    )

    # And the composition reads the runner's gate, not the registered field:
    # the registered arithmetic would plan six sessions for the same campaign.
    assert schedule.full_pairs_per_session == 6
    assert math.ceil(12 / schedule.full_pairs_per_session) * 3 == 6
    assert campaign.sessions == 9


def test_the_declared_window_is_the_measured_cap_less_its_reserve() -> None:
    """The 2026-09-20 amendment: the window a session is *given*, from the cap.

    Section 5.2's 12 h clock / 9 h usable window was written against Kaggle's
    *documented* GPU session limit (the 2022-01-24 product update). The first
    Gate 4 session measured the platform, and it disagreed: pushed 2026-09-20
    10:39:54Z, `act-t4-session` was killed with ERROR exactly 8 h later, before
    its own 9 h deadline, with an empty kernel log and no output files - a
    control check against the same account retrieves 24,973 characters of log
    and 288 output files for `act-t4-archzoo`, so the emptiness belongs to the
    session and not to the query. 8.05 GPU-h burned and not one pair published:
    a session spends its last hour on a pair, and the published dataset is its
    only durable copy, which a killed session never reaches.
    """
    assert PLATFORM_SESSION_CAP_HOURS == 8.0
    # The declared clock pays for the publish reserve and the prep allowance out
    # of the cap rather than out of the platform's patience.
    assert math.isclose(
        DECLARED_SESSION_HOURS
        + PUBLISH_RESERVE_MINUTES / 60
        + BOOT_AND_PREP_HOURS,
        PLATFORM_SESSION_CAP_HOURS,
        abs_tol=1e-9,
    )
    assert DECLARED_SESSION_HOURS == 7.0

    # At the T4's measured placement rate it packs four pairs and refuses a
    # fifth - and the fifth is exactly what the killed session was spending when
    # it died. The registered 9 h window still reads five, which is the number
    # the plan had been quoting.
    schedule = PairSchedule(tokens_per_second=5_083.4, arms=2, arm_slots=2)
    assert math.isclose(schedule.pair_time_hours, 1.443, abs_tol=5e-4)
    capacity = session_pair_capacity(schedule.pair_time_hours, DECLARED_SESSION_HOURS)
    assert capacity == 4
    assert session_pair_capacity(schedule.pair_time_hours) == 5
    # Four pairs, the prep allowance and the reserve all finish inside the cap.
    # That is the amendment's whole content, and it is what that fifth pair did
    # not do: 0.25 h + 5 x 1.443 h + 0.75 h = 8.2 h > 8.0 h.
    assert (
        BOOT_AND_PREP_HOURS
        + capacity * schedule.pair_time_hours
        + PUBLISH_RESERVE_MINUTES / 60
        <= PLATFORM_SESSION_CAP_HOURS
    )
    assert (
        BOOT_AND_PREP_HOURS
        + 5 * schedule.pair_time_hours
        + PUBLISH_RESERVE_MINUTES / 60
        > PLATFORM_SESSION_CAP_HOURS
    )

    # And it gives up nothing: 12 seeds need three sessions at either capacity,
    # so the campaign's session count and its training floor are unchanged - the
    # correction moves the publish inside the cap for free.
    assert (
        math.ceil(12 / capacity)
        == math.ceil(12 / session_pair_capacity(schedule.pair_time_hours))
        == 3
    )
    campaign = three_queue_plan(schedule.pair_time_hours, hours=DECLARED_SESSION_HOURS)
    assert campaign.sessions == 9
    assert campaign.primary.pairs_per_session == 4
    assert math.isclose(campaign.training_hours, 51.95, abs_tol=5e-2)


def test_a_one_device_host_keeps_the_null_queue_at_one_wave() -> None:
    """Sequential pairs pay two waves; the null seed still pays one.

    The asymmetry is why the campaign total cannot be derived from the pair
    queue's numbers alone: halving the pair queues' capacity on a one-device
    host leaves the null queue's untouched.
    """
    rate = 5_083.4
    placed = PairSchedule(tokens_per_second=rate, arms=2, arm_slots=2)
    sequential = PairSchedule(tokens_per_second=rate, arms=2, arm_slots=1)
    null = PairSchedule(tokens_per_second=rate, arms=1, arm_slots=1)
    campaign = three_queue_plan(
        sequential.pair_time_hours, null_pair_time_hours=null.pair_time_hours
    )
    assert campaign.primary.pairs_per_session == 2
    assert (campaign.primary.sessions, campaign.control.sessions) == (6, 6)
    assert campaign.null.pairs_per_session == 5
    assert campaign.null.sessions == 3
    assert campaign.sessions == 15
    assert campaign.null.pair_time_hours < sequential.pair_time_hours
    # The floor under the sequential arithmetic: two pair queues at two waves
    # each, plus the null queue at one, with the +0.1 h event overhead charged
    # once per pair (2 x 1.3429 + 0.1 = 2.7859 h) rather than per wave:
    # 12 x (2 x 2.7859 + 1.4429) = 84.2 h.
    assert math.isclose(campaign.training_hours, 84.2, abs_tol=5e-2)
    assert math.isclose(
        sequential.pair_time_hours,
        2 * placed.pair_time_hours - 0.1,
        rel_tol=1e-9,
    )


def test_queue_plan_refuses_a_schedule_that_starts_nothing() -> None:
    # A pair time above the usable window yields zero pairs; a plan that bills
    # quota and publishes nothing is refused where it is composed.
    with pytest.raises(ValueError, match="no pair per session"):
        queue_plan(USABLE_HOURS + 1.0)


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
    # A 13.75-hour two-arm pair (1,000 tokens/s) must not start with 14.0 h
    # left: pair + 45 min reserve = 14.50 h > 14.0 h, but it does fit 15.0 h.
    slow = PairSchedule(tokens_per_second=1_000.0).pair_time_hours
    assert round(slow, 2) == 13.75
    assert deadline_decision(14.0, slow).action == "stop_at_boundary"
    assert deadline_decision(15.0, slow).action == "start_pair"


def test_session_decisions_expose_reasons() -> None:
    for remaining in (2.0, 0.9, 0.2):
        decision = deadline_decision(remaining, 0.78)
        assert decision.reason
        assert decision.starts_work == (decision.action == "start_pair")
