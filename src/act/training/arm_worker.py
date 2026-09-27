"""Train exactly one arm of one pair, in its own process.

Section 5.2 defines a pair as two arms running concurrently, one per GPU, and
the pair-time arithmetic has always been written that way: ``PairSchedule``
charges one arm's training time for the pair. ``run_v0_session`` trained the
arms one after the other on device 0, so every pair actually cost the sum of
its arms - the deadline rule then started pairs the session clock could not
finish, and section 5.2 answers an interrupted pair by restarting it from
scratch rather than resuming it. Kaggle's T4 shape hands out two devices
(measured 2026-09-15), so the declared model is reachable.

A process per arm is the isolation the determinism contract needs. Model init
consumes global torch RNG and ``torch.manual_seed`` is global, so two threads
would race on it; two processes cannot. Each arm already seeds itself from the
session seed and writes only its own ``arm-<mode>`` directory, so the arms
share nothing but the seed - which is why the result must be identical however
they are scheduled.

Invoked by ``act.training.v0_session``, not by hand:

    python -m act.training.arm_worker JOB.json

The job file carries ``mode``, ``config`` (``V0SessionConfig`` fields),
``session_root``, ``device`` (``cuda:N``), ``deadline_seconds`` (how long the
arm may run, or ``null`` for no deadline), ``output`` (where to write the arm
result as JSON) and ``code_revision`` (what to record in the checkpoints this
child writes, or ``null``).

A child that stops at its **deadline** exits ``INTERRUPTED_EXIT_CODE`` and
writes where it stopped to ``output`` instead of an arm result. The parent
cannot tell a deadline stop from a crash by exit status alone, and the two are
handled very differently - a stop publishes an in-progress snapshot and is not
recorded as a failure, while a crash must stop the queue - so the distinction
is carried explicitly rather than inferred.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from act.training.runtime import resolve_device
from act.training.v0_session import (
    V0SessionConfig,
    V0SessionInterrupted,
    _train_arm,
)

# Distinct from any ordinary failure status, so the parent can recognise a
# cooperative deadline stop without parsing the child's stderr.
INTERRUPTED_EXIT_CODE = 10


def _deadline_check(budget: object) -> Callable[[], bool] | None:
    """The arm's own deadline, translated from the session's clock.

    The parent's check is a closure over the session clock, which cannot cross
    a process boundary, so the parent passes the seconds the arm may take and
    the child measures that interval from its own start. The gap between the
    two instants is subprocess spin-up - under a second against an hours-long
    budget - and the parent applies the same floor before spawning.
    """
    if budget is None:
        return None
    started = time.monotonic()
    limit = float(budget)

    def reached() -> bool:
        return time.monotonic() - started > limit

    return reached


def arm_job(
    *,
    mode: str,
    config: V0SessionConfig,
    session_root: Path,
    device: str,
    deadline_seconds: float | None,
    output: Path,
    code_revision: str | None = None,
) -> dict[str, object]:
    """The job payload for one arm (JSON-native, so it survives a process)."""
    return {
        "mode": mode,
        "config": asdict(config),
        "session_root": str(session_root),
        "device": device,
        "deadline_seconds": deadline_seconds,
        "output": str(output),
        "code_revision": code_revision,
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("usage: python -m act.training.arm_worker JOB.json")
    job = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    settings = dict(job["config"])
    # ``asdict`` turns the arm names back into a list; the config wants a tuple
    # (and every downstream mode lookup is by name, so a list would still work
    # - but the rebuilt config should equal the one the parent described).
    settings["arms"] = tuple(settings["arms"])
    config = V0SessionConfig(**settings)
    mode = str(job["mode"])
    output = Path(str(job["output"]))
    try:
        arm = _train_arm(
            mode,
            config,
            Path(str(job["session_root"])),
            resolve_device(str(job["device"])),
            log=lambda line: print(f"[{mode}] {line}", flush=True),
            deadline_check=_deadline_check(job.get("deadline_seconds")),
            # Absent in a job written before this key existed, which is why it is
            # read with ``.get`` and may be None - the same shape the parent uses.
            code_revision=job.get("code_revision"),
        )
    except V0SessionInterrupted as interrupt:
        # The same cooperative stop the sequential path raises through the
        # caller's own clock - just reported as a status the parent can read.
        output.write_text(
            json.dumps(
                {"mode": mode, "interrupted_at_step": interrupt.step},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[{mode}] {interrupt}", flush=True)
        return INTERRUPTED_EXIT_CODE
    payload = asdict(arm)
    payload["mutation_history"] = list(payload["mutation_history"])
    output.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
