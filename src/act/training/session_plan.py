"""Session scheduling and dtype policy for the pre-committed section 5.2 plan.

Every constant and rule here is pre-committed in RESEARCH_REPORT.md section
5.2 (the operational companion to the 5.1 compute-efficiency protocol); this
module encodes them exactly so the Kaggle session runner executes the
registered arithmetic rather than re-deriving it on the day. Standard
library only: scheduling decisions must be checkable anywhere, without torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Declared protocol budget (section 5.1): batch 2 x context 2048 x 6000 steps.
SEED_TOKENS_PER_ARM = 24_576_000
# The declared V0 model's sizes, named rather than inlined so that anything
# re-deriving the identity's scale factor reads the same numbers the
# prediction does (`expected_probe_cost_percent` below).
DECLARED_TOTAL_PARAMETERS = 285_336_576
DECLARED_ACTIVE_PARAMETERS = 134_341_632
# Per-batch probe token shape and the declared search shape (section 5.1).
PROBE_TOKENS_PER_BATCH = 4_096
DECLARED_EVENTS = 3
DECLARED_CANDIDATES = 4

# Session model (section 5.2). A Kaggle GPU session offers 12 hours; after
# boot, restore, and the publish reserve, ~9.0 h of usable work remain. That is
# the *documented* limit and stays the registered model; the shape the Gate 4
# sessions run on measured 8 h, which is why the clock a session is actually
# declared is derived separately below (:data:`DECLARED_SESSION_HOURS`).
SESSION_HOURS = 12.0
USABLE_HOURS = 9.0
# Fixed per-pair overhead of event/rebuild work (section 5.2's "+ ~0.1 h").
PAIR_OVERHEAD_HOURS = 0.1
# Deadline-rule constants (section 5.2).
PUBLISH_RESERVE_MINUTES = 45.0
BOUNDARY_STOP_MINUTES = 60.0
PUBLISH_ONLY_MINUTES = 15.0

# --- The declared session clock (amended 2026-09-20) ------------------------
# The constants above are section 5.2's *registered* model, written against
# Kaggle's *documented* GPU session limit (the 2022-01-24 product update raised
# notebook sessions from 9 h to 12 h for CPU and GPU). The first Gate 4 session
# measured the platform rather than the documentation and the two disagreed:
# `sid123123234/act-t4-session`, pushed 2026-09-20 10:39:54Z, was killed with
# ERROR exactly 8 h later, before its own 9 h deadline. The kill was total - an
# empty kernel log, no output files, and nothing published. The emptiness is
# the session's and not the query's, which a control check settles: the same
# call returns 24,973 characters of log and 288 output files for
# `act-t4-archzoo`, and 5,157/249 for `act-t4-selfcheck`.
#
# It cost 8.05 GPU-hours and every pair the session trained, because a session
# spends its last hour on a pair (rule 1 starts one whenever the pair time plus
# the 45-minute reserve fits) and the published dataset is its only durable
# copy, which a killed session never reaches. So the clock a session is
# *declared* is taken from the measured cap instead of the documented one, and
# is short enough that the session's last pair ends and its exit publish
# completes inside the window the platform actually grants.
PLATFORM_SESSION_CAP_HOURS = 8.0
# The part of the window that is neither training nor the exit publish: boot,
# the queue restore, the in-session throughput benchmark, and the import.
BOOT_AND_PREP_HOURS = 0.25
# The declared clock: the measured cap less the publish reserve and the prep
# allowance (8.0 - 0.75 - 0.25 = 7.0 h). At the T4's measured 1.443 h placement
# rate that packs 4 pairs and refuses a fifth - and the fifth is exactly what
# the killed session was spending when it died. It gives up nothing: 12 seeds
# need three sessions either way, because ceil(12/4) == ceil(12/5) == 3, so no
# session, training hour or quota is lost, and the publish moves ~1.5 h inside
# the cap. Read a plan composed at this window, not at ``USABLE_HOURS``.
DECLARED_SESSION_HOURS = PLATFORM_SESSION_CAP_HOURS - (
    PUBLISH_RESERVE_MINUTES / 60.0 + BOOT_AND_PREP_HOURS
)
# Working-directory ledger (20 GB auto-saved): a checkpoint payload is
# parameters + AdamW moments = 3.19 GB per arm, and pairs are atomic, so
# an in-flight pair holds 2 checkpoints x 2 arms x payload + 2 finals.
WORKING_DIR_GB = 20.0
CHECKPOINT_PAYLOAD_GB = 3.19
MAX_CHECKPOINTS = 2  # pinned: 3 would overflow the working dir with finals

# Declared protocol footprint (fp32, section 5.2 "Fit and footprint"):
# an in-flight pair holds max_checkpoints x 2 arms x payload, plus the
# finished pair's two final states (2 x 1.06 GB) until publish.
CHECKPOINT_STEADY_STATE_GB = MAX_CHECKPOINTS * 2 * CHECKPOINT_PAYLOAD_GB + 2 * 1.06

# Throughput-gated dtype rule (pre-committed): below the fp32 floor the
# primary config runs in fp16 with GradScaler (T4 has no bf16).
FP32_FLOOR_TOKENS_PER_SECOND = 2_500.0
FP16_FLOOR_TOKENS_PER_SECOND = 2_500.0


def decide_dtype(tokens_per_second: float) -> str:
    """Return ``"fp32"`` or ``"fp16"`` from the measured throughput.

    The pre-committed rule: if measured fp32 throughput is below the 2,500
    tokens/s floor, the primary config runs in fp16 with GradScaler. Dtype is
    an execution detail touching no pre-registered criterion (FLOPs counters
    book operations, not time); the measured dtype is recorded in the artifact.
    """
    if tokens_per_second < 0:
        raise ValueError("tokens_per_second must be non-negative")
    return "fp32" if tokens_per_second >= FP32_FLOOR_TOKENS_PER_SECOND else "fp16"


@dataclass(frozen=True)
class PairSchedule:
    """Schedule arithmetic for one atomic two-arm pair."""

    tokens_per_arm: int = SEED_TOKENS_PER_ARM
    tokens_per_second: float = FP32_FLOOR_TOKENS_PER_SECOND
    arms: int = 2
    # How many arms train at once: 1 when they run one after the other on one
    # device, ``arms`` when each gets its own GPU. Section 5.2's original rows
    # were computed for full concurrency; that is only reachable when the host
    # really has a device per arm (Kaggle's T4 shape does - two, measured
    # 2026-09-15). Defaulting to 1 keeps the arithmetic honest on a one-device
    # host instead of quietly assuming hardware nobody checked for.
    arm_slots: int = 1

    def __post_init__(self) -> None:
        if self.tokens_per_arm < 1:
            raise ValueError("tokens_per_arm must be positive")
        if self.tokens_per_second <= 0:
            raise ValueError("tokens_per_second must be positive")
        if self.arms < 1:
            raise ValueError("arms must be positive")
        if not 1 <= self.arm_slots <= self.arms:
            raise ValueError(
                f"arm_slots must be between 1 and {self.arms}, got {self.arm_slots}"
            )

    @property
    def pair_time_hours(self) -> float:
        """Wall time for one seed's pair at this placement.

        Section 5.2: pair time = tokens / throughput + the fixed event/rebuild
        overhead, counted once per *wave* of arms. With ``arm_slots == arms``
        every arm has its own device and the waves overlap completely, so the
        pair costs one arm-time - the form section 5.2's rows were written for.
        With ``arm_slots == 1`` the arms queue up on one device and the pair
        costs one arm-time each. Measured 2026-09-15: sequencing the arms added
        their wall times (0.531 s + 0.298 s) rather than overlapping them, and
        ignoring that made the deadline rule start pairs it could not finish -
        which section 5.2 answers by restarting the pair from scratch.
        """
        waves = math.ceil(self.arms / self.arm_slots)
        return (
            waves * self.tokens_per_arm / self.tokens_per_second / 3600.0
            + PAIR_OVERHEAD_HOURS
        )

    @property
    def full_pairs_per_session(self) -> int:
        """Floor(usable hours / pair time); a pair is never started unless it fits."""
        if self.pair_time_hours > USABLE_HOURS:
            return 0
        return int(USABLE_HOURS / self.pair_time_hours)

    @property
    def sessions_for_primary(self) -> int:
        """Sessions needed for the declared 12 seed pairs, primary config only."""
        pairs = self.full_pairs_per_session
        return math.ceil(12 / pairs) if pairs else 0

    @property
    def sessions_total(self) -> int:
        """Sessions including the secondary cap-4/2-batch run.

        The section 5.2 table schedules the secondary run (its own 12 pairs)
        as its own sessions: total = 2 x primary sessions. This matches the
        registered rows (10k tokens/s -> 2 primary, 4 total) even where a
        greedy 24-pair packing would fit in fewer.
        """
        return 2 * self.sessions_for_primary

    def steady_state_gb(self) -> float:
        """Working-directory bytes for an in-flight pair plus finished finals.

        Section 5.2: max_checkpoints x arms x 3.19 GB of resume state, plus
        the finished pair's 2 x 1.06 GB finals until publish (~14.9 GB for
        the declared two-arm pair with max_checkpoints=2).
        """
        return MAX_CHECKPOINTS * self.arms * CHECKPOINT_PAYLOAD_GB + 2 * 1.06


def window_left(declared_hours: float, spent_hours: float) -> float:
    """The session clock at entry: the declared window less what startup spent.

    ``v0_runner.run_pair_session``'s contract is explicit that the window it is
    handed is the platform limit "minus boot/restore", and both halves of that
    subtraction happen *outside* it: the launcher subtracts boot, the repo copy
    and the throughput benchmark, and ``act.training.cli`` subtracts the startup
    restore, which downloads the whole published dataset before a token is
    trained and grows with every seed the queue publishes.

    Both call sites used to pass the declared window unchanged, so a session
    believed it had the full window at entry, scheduled pairs against time it did
    not have, and reached its deadline stop that much late. Measured 2026-09-20: a
    session declared a 9 h window and the platform killed it at 8 h 00 m, before
    its own deadline. One spelling of the subtraction, so the clock the deadline
    rules act on cannot drift from the clock the window was declared with.
    """
    if spent_hours < 0:
        raise ValueError("spent_hours must be non-negative")
    return declared_hours - spent_hours


def working_dir_headroom_gb() -> float:
    """Headroom under the 20 GB auto-saved working directory at steady state."""
    return WORKING_DIR_GB - CHECKPOINT_STEADY_STATE_GB


@dataclass(frozen=True)
class SessionDecision:
    """What a session clock should do right now (deadline rules 1-3)."""

    action: str  # "start_pair", "stop_at_boundary", "publish_only", "idle"
    reason: str

    @property
    def starts_work(self) -> bool:
        return self.action == "start_pair"


def deadline_decision(
    remaining_hours: float,
    pair_time_hours: float,
) -> SessionDecision:
    """Apply the section 5.2 deadline rules to the remaining session clock.

    Rules, in order:
    1. Never start a new pair unless the measured pair time + 45 min publish
       reserve fits the remaining clock.
    2. Remaining time < 60 min mid-pair: stop at the next step boundary, save
       both arms, publish immediately.
    3. Remaining time < 15 min: publish only - event checkpoints are already
       saved.
    """
    if pair_time_hours <= 0:
        raise ValueError("pair_time_hours must be positive")
    if remaining_hours < 0:
        raise ValueError("remaining_hours must be non-negative")
    if remaining_hours < PUBLISH_ONLY_MINUTES / 60.0:
        return SessionDecision(
            "publish_only",
            f"remaining {remaining_hours:.2f} h < {PUBLISH_ONLY_MINUTES:.0f} min: "
            "publish only (rule 3)",
        )
    if remaining_hours < BOUNDARY_STOP_MINUTES / 60.0:
        return SessionDecision(
            "stop_at_boundary",
            f"remaining {remaining_hours:.2f} h < {BOUNDARY_STOP_MINUTES:.0f} min: "
            "stop at next step boundary, save, publish (rule 2)",
        )
    needed = pair_time_hours + PUBLISH_RESERVE_MINUTES / 60.0
    if remaining_hours >= needed:
        return SessionDecision(
            "start_pair",
            f"remaining {remaining_hours:.2f} h fits pair {pair_time_hours:.2f} h "
            f"+ {PUBLISH_RESERVE_MINUTES:.0f} min reserve (rule 1)",
        )
    return SessionDecision(
        "stop_at_boundary",
        f"remaining {remaining_hours:.2f} h does not fit pair {pair_time_hours:.2f} h "
        f"+ {PUBLISH_RESERVE_MINUTES:.0f} min reserve (rule 1)",
    )


def session_pair_capacity(
    pair_time_hours: float, hours: float = USABLE_HOURS
) -> int:
    """Pairs a session can *start*, by walking the deadline rules themselves.

    ``PairSchedule.full_pairs_per_session`` is section 5.2's registered
    arithmetic - ``floor(usable hours / pair time)`` - and stays exactly as
    registered, because the section 5.2 table's rows are computed from it and
    pinned by tests. The runner's start gate is stricter: rule 1 adds the
    45-minute publish reserve to every start, so the registered field reads one
    more pair per session than the runner will begin whenever the reserve binds.
    Measured 2026-09-20 at the T4's fp16 placement rate (1.443 h/pair): the
    registered field says 6 pairs and 2 sessions for the primary queue, while
    the runner starts 5 per session and therefore needs 3 sessions (5+5+2).
    Reading the plan from the registered field is what made that expectation
    wrong, so the launch reports this number beside it.
    """
    if pair_time_hours <= 0:
        raise ValueError("pair_time_hours must be positive")
    remaining, pairs = hours, 0
    while deadline_decision(remaining, pair_time_hours).action == "start_pair":
        remaining -= pair_time_hours
        pairs += 1
    return pairs


@dataclass(frozen=True)
class QueuePlan:
    """One queue's sessions and training time at the runner's own start gate.

    ``sessions`` is what a driver's re-push loop needs (one push is one
    session), and ``training_hours`` is the token-derived time the queue spends
    training - a *floor* on what it bills, because a session that exits when
    rule 1 refuses the next pair still pays boot, restore and its exit publish.
    """

    seeds: int
    pair_time_hours: float
    pairs_per_session: int

    def __post_init__(self) -> None:
        if self.seeds < 1:
            raise ValueError("seeds must be positive")
        if self.pairs_per_session < 1:
            raise ValueError(
                "a queue plan with no pair per session describes a run that "
                "bills quota and publishes nothing; refuse it before launch"
            )
        if self.pair_time_hours <= 0:
            raise ValueError("pair_time_hours must be positive")

    @property
    def sessions(self) -> int:
        return math.ceil(self.seeds / self.pairs_per_session)

    @property
    def training_hours(self) -> float:
        return self.seeds * self.pair_time_hours


def queue_plan(
    pair_time_hours: float,
    *,
    seeds: int = 12,
    hours: float = USABLE_HOURS,
) -> QueuePlan:
    """Plan one queue from the runner's start gate, not the registered field.

    The capacity comes from :func:`session_pair_capacity`, so a session that
    the 45-minute reserve keeps one pair short is priced as the runner will run
    it (see that function for the measured 2026-09-20 divergence).
    """
    return QueuePlan(
        seeds=seeds,
        pair_time_hours=pair_time_hours,
        pairs_per_session=session_pair_capacity(pair_time_hours, hours),
    )


@dataclass(frozen=True)
class ThreeQueuePlan:
    """The declared campaign: primary, pool-matched control, and null queue.

    Three symmetric queues over the same 12-seed block. Composed here rather
    than in prose because every recorded total so far has been wrong in its own
    way: the "~47 GPU-hours" plan was built on a pair time the placement
    correction changed, and the null queue carried a "~90 s/seed" figure
    measured at *dev* fidelity (400 steps, dev shape, `deployed_vs_null_study`
    2026-09-20: 102.9 s mean over 24 sessions) even though a declared null seed
    trains the same 24,576,000 tokens as a pair arm and costs a pair's
    arm-time. ``training_hours`` is the floor under the campaign's bill.
    """

    primary: QueuePlan
    control: QueuePlan
    null: QueuePlan

    @property
    def sessions(self) -> int:
        return self.primary.sessions + self.control.sessions + self.null.sessions

    @property
    def training_hours(self) -> float:
        return (
            self.primary.training_hours
            + self.control.training_hours
            + self.null.training_hours
        )


def three_queue_plan(
    pair_time_hours: float,
    *,
    null_pair_time_hours: float | None = None,
    seeds_per_queue: int = 12,
    hours: float = USABLE_HOURS,
) -> ThreeQueuePlan:
    """The three declared queues at these pair times.

    ``null_pair_time_hours`` defaults to the pair time because on a host with a
    device per arm both are one wave: a two-arm pair overlaps its arms and a
    null seed is a single arm, so they cost the same. On a one-device host the
    pair queues pay two waves and the null queue one, so the caller passes the
    sequential pair time's null cost explicitly rather than inheriting it.
    """
    null_time = pair_time_hours if null_pair_time_hours is None else null_pair_time_hours
    return ThreeQueuePlan(
        primary=queue_plan(pair_time_hours, seeds=seeds_per_queue, hours=hours),
        control=queue_plan(pair_time_hours, seeds=seeds_per_queue, hours=hours),
        null=queue_plan(null_time, seeds=seeds_per_queue, hours=hours),
    )


def expected_probe_cost_percent(probe_batch_count: int) -> float:
    """Section 5.1's declared-scale prediction for the chosen probe budget.

    cap-4/1-batch predicts 0.0942%, cap-4/2-batch 0.1883%; the tolerance band
    is [0.5, 2] x prediction. Values follow from the identity with the
    declared counts (P = 285,336,576; N_active = 134,341,632; D = 24,576,000
    tokens/arm; E = 3 events; k = 4 candidates; probe tokens = 4,096 per
    batch), not from a re-fit.
    """
    search = (
        DECLARED_EVENTS
        * DECLARED_CANDIDATES
        * probe_batch_count
        * PROBE_TOKENS_PER_BATCH
        * 2
        * DECLARED_ACTIVE_PARAMETERS
    )
    training = 2 * DECLARED_TOTAL_PARAMETERS * SEED_TOKENS_PER_ARM
    return 100.0 * search / training


def tolerance_band(probe_batch_count: int) -> tuple[float, float]:
    """Pre-committed [0.5, 2] x predicted-cost band for the primary config."""
    prediction = expected_probe_cost_percent(probe_batch_count)
    return 0.5 * prediction, 2.0 * prediction


def unusable_schedule_reason(schedule: PairSchedule) -> str | None:
    """Why this schedule can never start a pair, or ``None`` if it can.

    A pair is atomic and is never begun unless the whole thing fits the usable
    window, so a pair time above :data:`USABLE_HOURS` gives
    ``full_pairs_per_session == 0`` - an artifact describing a session that does
    nothing at all while billing quota. Measured 2026-09-15: the benchmark wrote
    exactly that on a GPU whose decided dtype runs *slower* than fp32, because
    the pre-committed rule selects fp16 on the fp32 rate and says nothing about
    whether fp16 is faster on the device in hand.
    """
    if schedule.full_pairs_per_session < 1:
        return (
            f"a {schedule.pair_time_hours:.2f} h pair does not fit the "
            f"{USABLE_HOURS:g} h usable window, so no pair can ever start"
        )
    return None
