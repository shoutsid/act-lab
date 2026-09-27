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
  "further sessions are scheduled until seed 15011 completes" is literally
  re-running the same command.

The runner never trains here; it decides, delegates, publishes, and reports.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from act.training.runtime import resolve_device
from act.training.session_plan import (
    PAIR_OVERHEAD_HOURS,
    PUBLISH_RESERVE_MINUTES,
    SEED_TOKENS_PER_ARM,
    PairSchedule,
    deadline_decision,
)
from act.training.v0_session import arm_slots
from act.training.v0_handoff import (
    COMPLETE_STATUS,
    IN_PROGRESS_STATUS,
    latest_published_seed,
    load_seed_summary,
    newest_arm_checkpoint,
    publish_seed,
    publish_seed_state,
    published_seed_status,
    published_seeds,
)
from act.training.v0_session import (
    V0SessionConfig,
    V0SessionInterrupted,
    run_v0_session,
)

# The declared section 5.1 queue. Re-reserved twice, for the same reason both times:
# the block must be blind and it must be the registered configuration.
#
# 9000-9011 -> 15000-15011 (section 5.5 change 4): the original block was spent by the
# section 5.3 real-data diagnostic, which ran the whole queue through this same runner at
# reduced fidelity, and a section 5.1 verdict already existed over it - so the harm guard,
# one of the three verdict branches, was no longer blind.
#
# 15000-15011 -> 15024-15035 (section 5.65): its first four seeds were published before
# the launch path carried the section 5.47 rebuild treatment, and their harm numbers were
# read and reported while the campaign was stopped - so that block is neither the
# registered configuration nor blind over the seeds already seen. See sections 5.47 and
# 5.65. 15012-15023 is skipped deliberately: 15012 is the benchmark's reserved data seed.
DECLARED_SEEDS = tuple(range(15024, 15036))


def _make_log(log: Callable[[str], None] | None) -> Callable[[str], None]:
    return log if log is not None else (lambda message: None)


#: The two configuration fields the section 5.47 amendment fixes, and the only ones the
#: treated and untreated declared runs disagree on. Named here rather than imported from
#: the verdict because the question this answers is "does the block I am about to extend
#: agree with itself", not "does it match the registration" - the second is the verdict's,
#: and its conformance block is descriptive by design.
TREATMENT_FIELDS = ("preserve_router_on_rebuild", "new_expert_initialization")


class BlockTreatmentMismatch(RuntimeError):
    """A session would append seeds under a treatment the published block does not record."""


def check_block_treatment_uniform(
    *,
    dataset_root: str | Path,
    config: V0SessionConfig,
    log: Callable[[str], None] | None = None,
) -> None:
    """Refuse to extend a published block with seeds trained under a different treatment.

    A block is the unit the verdict judges: its seeds are compared *within* the block, so
    a block whose seeds disagree on the rebuild treatment is not a block but two
    experiments sharing a seed range - and no per-seed field records which half a seed
    belongs to, so the split could only be reconstructed by hand. Nothing prevented it. The
    treated and untreated configurations differ on exactly the two fields below, both are
    valid, neither is a gate input, and neither is rejected by an assertion - so a session
    launched either way would run to completion and publish.

    Measured, on the declared block itself: seeds 15000-15003 recorded the *untreated*
    configuration, because the Kaggle launch builder never passed the two flags the runbook
    prints. The block is therefore untreated, and a resumed campaign on the amended command
    would have appended treated seeds 15004 onward to it. This raises before the first pair
    instead - the cheap failure. The block can still be finished uniformly, and *which*
    uniform configuration it finishes under stays a registration decision rather than
    becoming an accident of when the next session ran.

    The published *versions* are read, not the seeds this invocation was handed: the
    conflict is with the block, and a narrower ``--seeds`` appends to the same block.

    A field no published seed recorded is skipped, matching the verdict's ``not_recorded``
    treatment, so artifacts from before these fields existed are not read as conflicts.
    """
    logger = _make_log(log)
    recorded: dict[str, dict[str, list[int]]] = {}
    for seed in published_seeds(dataset_root):
        block = load_seed_summary(dataset_root, seed).get("config")
        if not isinstance(block, dict):
            continue
        for field in TREATMENT_FIELDS:
            if field in block:
                recorded.setdefault(field, {}).setdefault(repr(block[field]), []).append(seed)
    conflicts = []
    for field in TREATMENT_FIELDS:
        observed = recorded.get(field)
        if not observed:
            continue
        mine = repr(getattr(config, field))
        if set(observed) != {mine}:
            conflicts.append((field, mine, observed))
    if conflicts:
        lines = [
            "this session would train under a rebuild treatment the published block "
            "does not record, and the block cannot be split after the fact:",
        ]
        for field, mine, observed in conflicts:
            published = "; ".join(
                f"{value} (seed(s) {', '.join(str(s) for s in seeds_at)})"
                for value, seeds_at in sorted(observed.items())
            )
            lines.append(f"  {field}: this session {mine}, published {published}")
        lines.append(
            "Either finish the block under the configuration it already records (drop the "
            "section 5.47 flags from the launch command), or start a fresh block on an "
            "unspent seed range so the treatment is uniform. Refusing rather than mixing."
        )
        raise BlockTreatmentMismatch("\n".join(lines))
    agreed = {
        field: repr(getattr(config, field))
        for field in TREATMENT_FIELDS
        if recorded.get(field)
    }
    if agreed:
        logger(
            "block treatment matches the published seed(s): "
            + ", ".join(f"{field}={value}" for field, value in agreed.items())
        )
    else:
        logger(
            "no published seed records a rebuild treatment yet; this session sets it: "
            + ", ".join(
                f"{field}={getattr(config, field)!r}" for field in TREATMENT_FIELDS
            )
        )


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
    summary_extra: dict[str, object] | None = None,
) -> dict:
    """Run the section 5.2 session loop over the seed queue.

    ``remaining_hours`` is the session clock at entry (12-hour Kaggle limit
    minus boot/restore). ``measured_tokens_per_second`` parameterizes pair
    time, so it must be the rate of the dtype the session will train - the
    benchmark records that as ``schedule_tokens_per_second`` - and not the fp32
    rate the pre-committed dtype rule reads. The dtype rule itself lives with
    the benchmark, not here. ``clock`` optionally supplies wall time for tests;
    production uses ``time.monotonic``.
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

    # Pair time comes from the registered arithmetic, parameterized by how many
    # arms this host can actually train at once - one per device when it has
    # them, otherwise one at a time. Deriving it here instead of keeping a
    # second copy of the formula is what stopped the two from disagreeing: a
    # copy that ignored the arm multiplicity made the deadline rule start pairs
    # the clock could not finish, and section 5.2 restarts such a pair from
    # scratch rather than resuming it.
    arms = len(seed_config.arms)
    slots = arm_slots(arms, resolve_device(seed_config.device))
    pair_time = PairSchedule(
        tokens_per_arm=tokens_per_arm,
        tokens_per_second=measured_tokens_per_second,
        arms=arms,
        arm_slots=slots,
    ).pair_time_hours
    logger(
        f"pair time {pair_time:.2f} h "
        f"({arms} arm(s) in {math.ceil(arms / slots)} wave(s) of {tokens_per_arm} "
        f"tokens at {measured_tokens_per_second:.0f} tokens/s + "
        f"{PAIR_OVERHEAD_HOURS} h event overhead)"
    )

    # The block must be single-treatment - its seeds are compared within the block and
    # the treatment is not a field the verdict can partition on - but the check runs
    # where it protects a *write*: immediately before the first pair starts, and never
    # on a run whose queue is gated shut. A session that trains nothing cannot mix
    # treatments, so it must not fail either. See check_block_treatment_uniform.
    treatment_checked = False

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

        if not treatment_checked:
            check_block_treatment_uniform(
                dataset_root=dataset_root, config=seed_config, log=logger
            )
            treatment_checked = True

        # Start (or restart) this pair from scratch on this session's clock.
        # A stale in-progress snapshot is discarded by the restart semantics.
        logger(f"seed {seed}: starting pair ({decision.reason})")
        try:
            pair_result = _run_one_pair(
                seed=seed,
                config=seed_config,
                session_dir=session_dir,
                dataset_root=dataset_root,
                code_revision=code_revision,
                hours_left=hours_left,
                remaining_at_start=remaining,
                tokens_per_arm=tokens_per_arm,
                logger=logger,
            )
        except Exception as error:  # noqa: BLE001 - must not discard earlier seeds
            # The same principle the local-publish failure follows: a pair that
            # cannot finish must not abort the session before the seeds that
            # *did* finish reach the dataset. The durable copy is the exit
            # publish, which a crash never reaches - so a pair that raises
            # (an OOM, a killed arm process, a checkpoint write failure) stops
            # the queue instead, and the session exits with what it completed.
            # Recorded as ``training_failed`` - never as ``completed`` - so the
            # queue accounting stays honest and the driver stops rather than
            # re-pushing a failure it would repeat.
            logger(
                f"seed {seed}: pair FAILED ({type(error).__name__}: {error}); "
                "stopping the queue so the session can exit-publish completed seeds"
            )
            pair_results.append(
                {
                    "seed": seed,
                    "action": "training_failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            break
        pair_results.append(pair_result)
        if pair_result.get("action") == "publish_failed":
            # Stop rather than train a pair that cannot be published.
            break

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
    # Extra provenance from the caller (the CLI records the dtype the rule
    # decided and the rate the schedule used beside the one pair time used),
    # recorded here so the durable log is written by one writer in one format.
    summary.update(summary_extra or {})
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
    remaining_at_start: float,
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

    # The same floor as a duration, so a per-arm child process can enforce it:
    # a closure over this session's clock cannot cross a process boundary. This
    # is what lets the pair's arms run one per device, which is how section
    # 5.2 prices a pair. Taken from the clock reading that approved this pair's
    # start rather than a fresh read: the budget is that approval's, and an
    # extra read would also shift the intra-pair burn-down by one tick.
    arm_budget_seconds = max(0.0, remaining_at_start - 45.0 / 60.0) * 3600.0

    try:
        result = run_v0_session(
            pair_config,
            checkpoint_dir=session_dir,
            log=logger,
            deadline_check=deadline_reached,
            deadline_budget_seconds=arm_budget_seconds,
            # Carried into every checkpoint, not only into the manifest: the
            # manifest and its dataset are the durable record, but a checkpoint
            # that leaves them behind had nothing saying which tree wrote it
            # (2026-09-27, measured on all six published seeds).
            code_revision=code_revision,
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

    try:
        publish_seed(
            dataset_root,
            result,
            session_dir=session_dir,
            code_revision=code_revision,
            clean_working_dir=True,
        )
    except Exception as error:  # noqa: BLE001 - must not discard earlier seeds
        # The same principle the exit publish follows (v0_handoff/ops: a failed
        # push must never destroy completed work): a failed *local* publish for
        # seed N must not abort the session before the seeds already published
        # reach the dataset. Recorded as ``publish_failed`` - never as
        # ``completed``, so skip-completed and the queue accounting stay honest
        # - and the caller stops the queue rather than spending more GPU on a
        # substrate that cannot publish.
        logger(
            f"seed {seed}: publish FAILED ({type(error).__name__}: {error}); "
            "stopping the queue so the session can exit-publish completed seeds"
        )
        return {
            "seed": seed,
            "action": "publish_failed",
            "error": f"{type(error).__name__}: {error}",
        }
    arms = {arm.mode: arm.final_loss for arm in result.arms}
    logger(f"seed {seed}: pair complete ({arms})")
    # Single-arm sessions (gated, report section 5.4; null, report section
    # 5.8; heuristic, report section 5.13) publish one arm; the baseline
    # comparison lives in the paired-block summaries, not here.
    return {
        "seed": seed,
        "action": "completed",
        "random_final_loss": arms.get("random"),
        "adaptive_final_loss": arms.get("adaptive"),
        "gated_final_loss": arms.get("gated"),
        "null_final_loss": arms.get("null"),
        "heuristic_final_loss": arms.get("heuristic"),
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
    if not isinstance(payload, dict) or payload.get("version") not in {4, 5}:
        raise ValueError(f"{checkpoint} is not a version-4/5 V0 session payload")
    return payload
