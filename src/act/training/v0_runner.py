"""Per-seed pair runner: one Kaggle session of the section 5.2 plan.

Ties the three execution pieces into the registered loop:

- ``act.training.session_plan`` supplies the dtype rule and the deadline
  ladder (never start a pair without the 45-min publish reserve, boundary
  stop under 60 minutes, publish-only under 15 minutes);
- ``act.training.v0_session`` trains a pair cooperatively, stopping only at
  checkpoint boundaries when the session clock runs out;
- ``act.training.v0_handoff`` carries every seed across the session boundary
  (``complete`` versions resume-ready, ``in-progress`` snapshots discarded).

Registered semantics (section 5.2, pre-committed):

- A pair is **atomic**: it is never started unless measured pair time + the
  publish reserve fits the remaining clock, and it is **never resumed across
  sessions** - a pair split by the session boundary restarts from scratch,
  so each seed's 24,576,000 training tokens per arm come from one
  uninterrupted execution per arm.
- Completed pairs publish ``complete`` versions; a deadline stop mid-pair
  publishes an ``in-progress`` safety snapshot (replacing any stale
  in-progress version, never a complete one) purely for crash forensics.
- The loop is idempotent: seeds with a ``complete`` version are skipped, so
  "further sessions are scheduled until seed 9011 completes" is literally
  re-running the same command.

The runner never trains here; it decides, delegates, publishes, and reports.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from act.training.session_plan import (
    PAIR_OVERHEAD_HOURS,
    PUBLISH_RESERVE_MINUTES,
    SEED_TOKENS_PER_ARM,
    deadline_decision,
)
from act.training.v0_handoff import (
    COMPLETE_STATUS,
    IN_PROGRESS_STATUS,
    latest_published_seed,
    newest_arm_checkpoint,
    publish_seed,
    publish_seed_state,
    published_seed_status,
)
from act.training.v0_session import (
    V0SessionConfig,
    V0SessionInterrupted,
    run_v0_session,
)

DECLARED_SEEDS = tuple(range(9000, 9012))


def _make_log(log: Callable[[str], None] | None) -> Callable[[str], None]:
    return log if log is not None else (lambda message: None)


def run_pair_session(
    *,
    seeds: tuple[int, ...] = DECLARED_SEEDS,
    config: V0SessionConfig,
    session_dir: str | Path,
    dataset_root: str | Path,
    code_revision: str,
    remaining_hours: float,
    measured_tokens_per_second: float,
    output: str | Path | None = None,
    log: Callable[[str], None] | None = None,
    clock: Callable[[], float] | None = None,
) -> dict:
    """Run the section 5.2 session loop over the seed queue.

    ``remaining_hours`` is the session clock at entry (12-hour Kaggle limit
    minus boot/restore). ``measured_tokens_per_second`` is the session-1
    benchmark's number: it parameterizes pair time, and the dtype rule lives
    with the benchmark, not here. ``clock`` optionally supplies wall time for
    tests; production uses ``time.monotonic``.
    """
    logger = _make_log(log)
    if measured_tokens_per_second <= 0:
        raise ValueError("measured_tokens_per_second must be positive")
    if remaining_hours <= 0:
        raise ValueError("remaining_hours must be positive")
    started = clock() if clock is not None else time.monotonic()

    def hours_left() -> float:
        now = clock() if clock is not None else time.monotonic()
        return remaining_hours - (now - started) / 3600.0

    seeds = tuple(int(seed) for seed in seeds)
    seed_config = V0SessionConfig(
        **{key: value for key, value in config.__dict__.items() if key != "seed"}
    )
    tokens_per_arm = seed_config.training_tokens()

    pair_time = tokens_per_arm / measured_tokens_per_second / 3600.0 + PAIR_OVERHEAD_HOURS
    logger(
        f"pair time {pair_time:.2f} h "
        f"({tokens_per_arm} tokens at {measured_tokens_per_second:.0f} tokens/s + "
        f"{PAIR_OVERHEAD_HOURS} h event overhead)"
    )

    pair_results: list[dict[str, object]] = []
    for seed in seeds:
        # Skip-completed is checked before any gating: re-running the same
        # command to advance the queue must work on any remaining clock.
        status = published_seed_status(dataset_root, seed)
        if status == COMPLETE_STATUS:
            logger(f"seed {seed}: complete version exists, skipping")
            pair_results.append({"seed": seed, "action": "skipped_complete"})
            continue

        remaining = hours_left()
        decision = deadline_decision(remaining, pair_time)
        if decision.action in {"publish_only", "idle"}:
            logger(f"seed {seed}: {decision.reason}")
            break

        # Rule 1 is the only start gate: a pair may begin whenever the
        # measured pair time + the 45-min publish reserve fits the clock.
        # The raw "stop_at_boundary" verdict is rule 2's advice for an
        # in-flight pair, not a start gate - rule 2's session-side analogue
        # is the intra-pair floor inside ``_run_one_pair``, which stops a
        # started pair at the next checkpoint boundary once the clock passes
        # the reserve. For the declared pair (~0.79 h) rule 1 and rule 2
        # agree; only a tiny pair approved by rule 1 can run under rule 2.
        if remaining < pair_time + PUBLISH_RESERVE_MINUTES / 60.0:
            logger(f"seed {seed}: {decision.reason}")
            break

        # Start (or restart) this pair from scratch on this session's clock.
        # A stale in-progress snapshot is discarded by the restart semantics.
        logger(f"seed {seed}: starting pair ({decision.reason})")
        pair_result = _run_one_pair(
            seed=seed,
            config=seed_config,
            session_dir=session_dir,
            dataset_root=dataset_root,
            code_revision=code_revision,
            hours_left=hours_left,
            tokens_per_arm=tokens_per_arm,
            logger=logger,
        )
        pair_results.append(pair_result)

    completed = sum(1 for entry in pair_results if entry.get("action") == "completed")
    latest = latest_published_seed(dataset_root)
    summary = {
        "protocol": "v0-compute-efficiency-5.2",
        "seeds_requested": list(seeds),
        "declared_seed_queue": list(DECLARED_SEEDS),
        "remaining_hours_at_entry": remaining_hours,
        "measured_tokens_per_second": measured_tokens_per_second,
        "pair_time_hours": pair_time,
        "pairs_completed": completed,
        "pairs": pair_results,
        "latest_published_seed": latest[0] if latest is not None else None,
    }
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    logger(
        f"session done: {completed} pair(s) completed, "
        f"{hours_left():.2f} h remaining"
    )
    return summary


def _run_one_pair(
    *,
    seed: int,
    config: V0SessionConfig,
    session_dir: Path,
    dataset_root: Path,
    code_revision: str,
    hours_left: Callable[[], float],
    tokens_per_arm: int,
    logger: Callable[[str], None],
) -> dict[str, object]:
    """Train, publish, and bookkeep one atomic pair."""
    pair_config = V0SessionConfig(**{**config.__dict__, "seed": seed})

    # Pairs are atomic and never resumed across sessions: any working-dir
    # state for this seed (e.g. from a crashed prior session) must go before
    # a fresh start, or the session runner would silently resume it.
    seed_root = Path(session_dir) / f"seed-{seed}"
    if seed_root.exists():
        logger(f"seed {seed}: discarding stale working-dir state (pairs restart from scratch)")
        shutil.rmtree(seed_root)

    # Deadline rule 2's cooperative stop: the session runner raises
    # V0SessionInterrupted at the next checkpoint boundary once the clock
    # passes the intra-pair floor (the 45-min publish reserve; the 60-minute
    # rule-2 threshold would needlessly kill a small pair rule 1 just
    # approved). The pair then restarts from scratch in a later session.
    def deadline_reached() -> bool:
        return hours_left() < 45.0 / 60.0

    try:
        result = run_v0_session(
            pair_config,
            checkpoint_dir=session_dir,
            log=logger,
            deadline_check=deadline_reached,
        )
    except V0SessionInterrupted as interrupt:
        logger(f"seed {seed}: {interrupt}; publishing in-progress snapshot")
        _publish_snapshot(
            seed=seed,
            interrupt=interrupt,
            pair_config=pair_config,
            session_dir=session_dir,
            dataset_root=dataset_root,
            code_revision=code_revision,
            logger=logger,
        )
        return {
            "seed": seed,
            "action": "interrupted",
            "interrupted_step": interrupt.step,
        }

    publish_seed(
        dataset_root,
        result,
        session_dir=session_dir,
        code_revision=code_revision,
        clean_working_dir=True,
    )
    arms = {arm.mode: arm.final_loss for arm in result.arms}
    logger(f"seed {seed}: pair complete ({arms})")
    return {
        "seed": seed,
        "action": "completed",
        "random_final_loss": arms["random"],
        "adaptive_final_loss": arms["adaptive"],
        "training_flops_by_arm": {
            arm.mode: arm.training_flops for arm in result.arms
        },
        "search_flops_by_arm": {arm.mode: arm.search_flops for arm in result.arms},
    }


def _publish_snapshot(
    *,
    seed: int,
    interrupt: V0SessionInterrupted,
    pair_config: V0SessionConfig,
    session_dir: Path,
    dataset_root: Path,
    code_revision: str,
    logger: Callable[[str], None],
) -> None:
    """Publish a deadline stop's safety snapshot (never a complete version).

    An in-progress snapshot replaces only a previous in-progress version; a
    complete version is left untouched (this snapshot cannot coexist with a
    completed pair anyway - the seed would have been skipped). Arms without
    any checkpoint (an interrupt during the first arm leaves the second
    untrained) are omitted: the snapshot is forensic only, since the
    atomic-pair rule restarts the pair from scratch in the next session.
    """
    seed_root = Path(session_dir) / f"seed-{seed}"
    checkpoint_paths: dict[str, str | Path] = {}
    arm_payloads: dict[str, dict[str, object]] = {}
    for arm_mode in pair_config.arms:
        arm_dir = seed_root / f"arm-{arm_mode}"
        if not any(arm_dir.glob("checkpoint-*.pt")):
            logger(f"seed {seed}: arm {arm_mode} has no checkpoint; omitted from snapshot")
            continue
        checkpoint = newest_arm_checkpoint(arm_dir)
        checkpoint_paths[arm_mode] = checkpoint
        payload = torch_load_summary(checkpoint)
        arm_payloads[arm_mode] = {
            "mode": arm_mode,
            "checkpoint_step": payload["training"]["step"],
            "training_flops": payload["v0_session"]["training_flops"],
            "search_flops": payload["v0_session"]["search_flops"],
            "mutations": payload["v0_session"]["mutations"],
        }
    if not arm_payloads:
        raise RuntimeError(
            f"seed {seed}: deadline stop happened before any checkpoint existed"
        )
    publish_seed_state(
        dataset_root,
        seed=seed,
        session_dir=session_dir,
        checkpoint_paths=checkpoint_paths,
        arm_payloads=arm_payloads,
        config=asdict(pair_config),
        predicted_cost_ratio=0.0,
        code_revision=code_revision,
        status=IN_PROGRESS_STATUS,
        interrupted_step=interrupt.step,
        clean_working_dir=True,
    )


def torch_load_summary(checkpoint: str | Path) -> dict:
    """Load the metric subset of a checkpoint payload without torch models.

    Kept torch-free at import time so the runner's scheduling logic stays
    importable anywhere; torch is only needed when a snapshot is published.
    """
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("version") != 4:
        raise ValueError(f"{checkpoint} is not a version-4 V0 session payload")
    return payload
