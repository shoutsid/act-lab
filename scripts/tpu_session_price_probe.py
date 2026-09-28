"""Price the **declared** session path on the TPU: one null seed, the real config.

Section 5.14's option 2 moves the do-nothing (null) queue - twelve single-arm seeds,
24,576,000 tokens each - onto Kaggle's separate 20 TPU-hours/week, and it is priced at
**8.06 h** by the port rule's own arithmetic. That price comes from `tpu_probe.py` v6, which
measured a **hand-built step loop** at the declared shape (15,034 aggregate tokens/s, 2.96x
the T4 pair). The rehearsal (`tpu_session_rehearsal.py`) then proved the *registered* session
path runs on XLA - but at **development** fidelity, and its own record says so:
``not_measured`` carries "declared-fidelity throughput: the shape here is the development
one, so its tokens/s is a path measurement and not a schedule input".

So the one number the amendment's price rests on has never been measured on the path that
would run it. This measures it. ``run_v0_session`` on ``declared_session_config()`` - 285M
total / 134M active parameters, batch 2 x context 2048, fp16, checkpointed activations - as a
single **null** arm on one device, and the result is priced through
``tpu_probe.queue_tpu_hours``, the campaign's own arithmetic, so this figure and the port
rule's 8.06 h cannot be computed two different ways.

**What it is not.** It is not a declared seed: it trains ``--steps`` steps (240 by default,
enough that the steady phase dominates the compile), not 6,000, and it publishes nothing to
the campaign's datasets - its handoff goes to a scratch root, exactly as the rehearsal's
does, which is what the record's ``scratch_root`` field is for. It is not a Gate-4 result, not
a port, and not evidence about the hypothesis.

**Why its figure is a floor, not a flattering rate.** The first step of the loop compiles the
declared graph, and that compile is inside the wall the rate is divided by - whereas a
6,000-step seed pays it once and amortizes it. So the number here prices the queue
*pessimistically*: if this fits the week, the queue does. ``per_session_loop_seconds`` beside
it is the loop's own clock, so a reader can see how much of the wall was the compile.

    PYTHONPATH=src .venv/bin/python scripts/tpu_session_price_probe.py --device cpu --steps 2
    PYTHONPATH=src .venv/bin/python scripts/tpu_session_price_probe.py --device xla   # on a TPU
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# Reused rather than re-implemented, which is the point of the three imports: the campaign's
# price arithmetic, the mount search and per-phase containment, and the *proven* session
# phases. A second copy of the phases is how the declared probe would end up measuring a
# slightly different path than the one that was rehearsed.
import tpu_probe  # noqa: E402
import tpu_session_rehearsal as rehearsal  # noqa: E402

RECORD_NAME = "tpu-session-price-probe.json"
OUTPUT = Path(os.environ.get("TPU_PRICE_OUTPUT", f"/kaggle/working/{RECORD_NAME}"))

#: The same four phases the rehearsal runs, and `run_phases` reads this order out of the
#: record rather than taking it as an argument.
PHASES = rehearsal.PHASES

#: The chips a v5e-8 gives one process, and so the number the port rule's *aggregate* assumes:
#: `tpu_probe` v6 measured eight independent seeds at once, one per chip. This run measures
#: **one** session, so the aggregate below is that measurement times this figure - the same
#: assumption the port rule makes, named here rather than buried in a constant, and reported
#: twice so a reader can see both the parallel price and the price if the chips do not scale.
XLA_CHIPS = 8

#: The declared shape is ~19x the development model per step, so the rehearsal's leashes are
#: not leashes here: the session gets an hour, and the phases that write and hash a ~3.4 GB
#: declared checkpoint get half of one each.
PHASE_DEADLINE_SECONDS = {
    "environment": 900.0,
    "session": 3600.0,
    "resume": 1800.0,
    "handoff": 1800.0,
}

#: Outside every reserved block (15000-15011, 15024-15035): this is a rate measurement, and
#: a seed that could belong to the campaign is a seed whose record a reader would have to
#: place. The benchmark uses the same one for the same reason.
SEED = 15012

NOT_MEASURED = (
    "a declared seed: this trains the declared *shape* for --steps steps, not the protocol's "
    "24,576,000 tokens, so it prices the queue without producing any of its records",
    "the eight-chips-at-once placement: one session ran on one device, and the aggregate "
    "below multiplies it by eight rather than measuring eight",
    "mutation and rebuild events: the null arm never fires one, and --steps stays below the "
    "declared mutation interval, so the rebuild cost the port probe priced is not in this "
    "figure",
)


def price_config(device: str, *, steps: int, seed: int):
    """The declared session shape, with only the step budget and cadence this run needs.

    Everything that *defines* the shape comes from `declared_session_config` - 285M total /
    134M active, batch 2 x context 2048, fp16, checkpointed activations, the declared vocab -
    because a price measured on a shape nobody will run is not a price. Overridden: the step
    budget (a rate does not need 6,000 steps), the seed (see :data:`SEED`), the checkpoint
    cadence, and the arm set (the null queue is single-arm by construction).
    """
    from act.training.v0_session import declared_session_config

    return declared_session_config(
        device=device,
        seed=seed,
        arms=("null",),
        total_steps=steps,
        checkpoint_interval=max(1, steps // 2),
        max_checkpoints=2,
    )


def price_block(session: dict[str, Any]) -> dict[str, Any]:
    """The campaign's own arithmetic, applied to what the session path measured.

    ``queue_tpu_hours`` is the campaign's single spelling of "twelve single-arm seeds through
    the TPU's eight-core aggregate", so this figure and the port rule's 8.06 h cannot come
    apart - which is the whole reason to take a price from a live run instead of writing a
    second formula for it.
    """
    arm = (session.get("arms") or {}).get("null") or {}
    rate = float(arm.get("tokens_per_second") or 0.0)
    if rate <= 0:
        raise RuntimeError("the session phase left no throughput to price the queue with")
    aggregate = rate * XLA_CHIPS
    multiple = aggregate / tpu_probe.T4_PAIR_TOKENS_PER_SECOND
    tokens = tpu_probe.DECLARED_SEEDS_PER_QUEUE * tpu_probe.DECLARED_TOKENS_PER_ARM
    hours = tpu_probe.queue_tpu_hours(multiple, arms=1)
    bar = tpu_probe.queue_tpu_hours(tpu_probe.PORT_THRESHOLD_MULTIPLE, arms=1)
    return {
        "per_session_tokens_per_second": round(rate, 1),
        "per_session_wall_seconds": arm.get("wall_time_seconds"),
        "assumed_chips": XLA_CHIPS,
        "implied_aggregate_tokens_per_second": round(aggregate, 1),
        "multiple_of_t4_pair": round(multiple, 3),
        "port_threshold_multiple": tpu_probe.PORT_THRESHOLD_MULTIPLE,
        "null_queue_seeds": tpu_probe.DECLARED_SEEDS_PER_QUEUE,
        "null_queue_tokens": tokens,
        "null_queue_hours_at_this_rate": round(hours, 3),
        "null_queue_hours_at_the_2_0x_bar": round(bar, 3),
        # Without the eight-chip assumption, for a reader who will not take scaling on trust:
        # one session after another, twelve seeds, at the rate this run actually measured.
        "null_queue_hours_if_the_chips_do_not_scale": round(tokens / rate / 3600.0, 3),
        "tpu_week_hours": tpu_probe.TPU_QUOTA_HOURS,
        "fits_the_week": hours <= tpu_probe.TPU_QUOTA_HOURS,
        "pessimistic_because": (
            "the loop's first step compiles the declared graph and that compile is inside "
            "the wall this rate divides by; a 6,000-step seed pays it once"
        ),
    }


def base_record(args: argparse.Namespace) -> dict[str, Any]:
    """This record's labels, before any phase has run.

    The rehearsal's labels would be wrong here in the one direction that matters: it says
    ``declared_scale: false``, and this run *is* at the declared shape while still not being
    a declared seed. So every field that could be read as a result says what it is, and
    ``not_measured`` names the three things a reader could otherwise supply from memory.
    """
    return {
        "record_type": "session-price-probe",
        "kind": "tpu-session-price-probe",
        "rehearsal": False,
        "declared_shape": True,
        "declared_seed": False,
        "declared_scale": False,
        "device_requested": args.device,
        "steps": args.steps,
        "seed": args.seed,
        "scratch_root": "the handoff publishes into a temporary root under --output's parent, "
        "not into any campaign dataset",
        "phase_order": list(PHASES),
        "completed": [],
        "errors": [],
        "phase_deadlines": {
            name: rehearsal.phase_deadline_seconds(name, PHASE_DEADLINE_SECONDS)
            for name in PHASES
        },
        "watch_seconds": rehearsal.watch_seconds(),
        "not_measured": list(NOT_MEASURED),
        "what_this_is": (
            "A price: the registered V0 session path, at the declared 285M shape but for "
            f"{args.steps} steps rather than a seed's 24,576,000 tokens, on the device named "
            "in `device_requested`. It reports what one session path sustains there, and "
            "what twelve null-queue seeds would therefore cost. It is not a declared seed, "
            "not a Gate-4 result, and not evidence about the hypothesis; no queue moves on "
            "this record."
        ),
    }


def verdict(record: dict[str, Any]) -> str:
    """One line: what the session path priced, and whether the queue fits the week.

    The comparison the amendment needs, in the order it needs it: the measured rate, the
    multiple it implies, the queue's hours, and the bar it is being held to.
    """
    price = record.get("price") or {}
    rate = price.get("per_session_tokens_per_second")
    if not rate:
        return "no price: the session phase did not leave a throughput to read"
    return (
        f"the declared session path sustains {rate:,.1f} tokens/s on one chip "
        f"({price['multiple_of_t4_pair']:.2f}x the T4 pair across {XLA_CHIPS} chips: "
        f"{price['implied_aggregate_tokens_per_second']:,.0f} tokens/s), so the null queue is "
        f"{price['null_queue_hours_at_this_rate']:.2f} h against the "
        f"{price['null_queue_hours_at_the_2_0x_bar']:.2f} h at the rule's 2.0x bar and a "
        f"{price['tpu_week_hours']:.0f} h week "
        f"({'fits' if price['fits_the_week'] else 'does not fit'})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="xla", help="xla (a TPU host) or cpu (a dry run)")
    parser.add_argument(
        "--steps", type=int, default=240, help="declared-shape training steps (one arm)"
    )
    parser.add_argument("--seed", type=int, default=SEED, help="see SEED in this module")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    record = base_record(args)
    work = Path(args.output).parent / f"tpu-price-work-{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)

    def session_phase() -> None:
        rehearsal.phase_session(
            record,
            work,
            device=args.device,
            steps=args.steps,
            seed=args.seed,
            build=price_config,
        )
        # Priced the moment the session lands rather than after the phases that follow it:
        # the price is this run's whole product, and a later phase that dies must not take it
        # with it. The three phases after this one are the rehearsal's own evidence that the
        # path around the rate still works at this shape - the resume loads a ~3.4 GB declared
        # checkpoint back, and the handoff writes and hashes one.
        record["price"] = price_block(record["session"])
        tpu_probe._save(record, Path(args.output))

    phases: dict[str, Callable[[], None]] = {
        "environment": lambda: rehearsal.phase_environment(record, work),
        "session": session_phase,
        "resume": lambda: rehearsal.phase_resume(
            record,
            work,
            device=args.device,
            steps=args.steps,
            seed=args.seed,
            build=price_config,
        ),
        "handoff": lambda: rehearsal.phase_handoff(
            record,
            work,
            device=args.device,
            steps=args.steps,
            seed=args.seed,
            build=price_config,
        ),
    }

    started = time.time()
    rehearsal.run_phases(record, Path(args.output), phases, deadlines=PHASE_DEADLINE_SECONDS)
    record["wall_seconds"] = time.time() - started
    tpu_probe._save(record, Path(args.output))

    print(json.dumps(record, indent=1, sort_keys=True, default=str), flush=True)
    # The exit status is a signal, not the result: the record is the result, and it was
    # written as each phase landed - so a nonzero exit here still leaves a fetchable record.
    return 0 if len(record["completed"]) == len(PHASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
