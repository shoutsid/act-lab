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
# boot, restore, and the publish reserve, ~9.0 h of usable work remain.
SESSION_HOURS = 12.0
USABLE_HOURS = 9.0
# Fixed per-pair overhead of event/rebuild work (section 5.2's "+ ~0.1 h").
PAIR_OVERHEAD_HOURS = 0.1
# Deadline-rule constants (section 5.2).
PUBLISH_RESERVE_MINUTES = 45.0
BOUNDARY_STOP_MINUTES = 60.0
PUBLISH_ONLY_MINUTES = 15.0
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
