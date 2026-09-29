"""Run the registered **null queue** on a TPU: one declared seed per chip, all at once.

Section 5.14's option 2 is the campaign's own priced route onto the TPU week: the null queue is
**12 single-arm seeds** of 24,576,000 tokens each, and the port rule put it at **8.06 h** of TPU
time at its own 2.0x bar (5.7 h at the 2.96x the probe measured). What makes that a *week's* worth
of work rather than six weeks is the shape of the queue: one arm, no pairing, no probe, no
collective - so the twelve seeds are independent, and a v5e-8 can carry **eight of them at once,
one per chip**. Sequentially, at the probe's measured ~1,879 tokens/s per chip, twelve seeds would
need ~43 h against a 20 h week; eight-wide, two waves land them in ~5.4 h of *wall clock*, which is
what Kaggle bills (v1 of the rehearsal cost 3.02 TPU-h for a 3.02 h run; its 2.5-minute rerun cost
0.04). That parallelism is the whole reason the TPU is worth anything here, and it is the one piece
the session CLI does not have: `act.training.cli --mode pair-session` walks seeds **in turn on one
device**.

So this is the queue's registered command with the one substitution the port needs - `--device xla`
- plus a thread per chip. What it reuses rather than re-implements:

* ``run_v0_session`` and ``publish_seed``, the registered path (rehearsed on XLA at development
  fidelity on 2026-09-28, and since 2026-09-29 with ``activation_checkpointing`` on, which is the
  declared config's field and the one whose absence from the rehearsal let the checkpointing defect
  reach a queue);
* ``tpu_probe.bind_thread_device``, because ``xm.mark_step()`` is ``torch_xla.sync()`` and acts on
  the **calling thread's** current device - a fresh thread inherits the process default, so without
  the binding seven of the eight ranks would mark chip 0's steps and their graphs would never
  execute;
* the section 5.47 rebuild treatment (``--preserve-router-on-rebuild --new-expert clone``), carried
  on every arm of every queue by the registration. It is inert for the null arm - a do-nothing arm
  fires no rebuild - and it is carried anyway, because "the queue was launched without the flags"
  is the exact defect that invalidated the campaign's first block;
* ``tpu_probe._save``, so the record is on disk as each seed lands rather than at the end: a kernel
  that dies at hour five still carries the seeds that finished.

**The dtype is fp32, and it is the one field where this differs from the registered queue** (which
is fp16). ``V0SessionConfig`` admits ``fp32`` and ``fp16`` only - there is no bf16 session to run -
and fp16 on a TPU needs ``torch.autocast(device_type="xla", ...)`` plus a ``GradScaler``, neither of
which any record in this repository shows working there. fp32 is registered, needs no such
machinery, and is the slower end of the two dtypes ``tpu_probe`` v5 put within 6% of each other, so
it errs toward slower rather than faster. The record says which dtype produced it.

Each seed checkpoints on the registered cadence, so a kernel that hits the platform's wall can be
followed by a second run that resumes from those checkpoints rather than restarting the seed.

    # development fidelity, one worker - the shape a local XLA:CPU dry run can carry
    PJRT_DEVICE=CPU python scripts/tpu_null_queue.py --device xla --fidelity development \
        --seeds 15024-15025 --workers 1 --steps 20
    # the queue itself, on a TPU
    python scripts/tpu_null_queue.py --device xla --seeds 15024-15035 --workers 8
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# Reused rather than re-implemented: the probe's per-thread device binding (and the reason for it),
# the record saver, and the rehearsal's development-fidelity config and revision lookup.
import tpu_probe  # noqa: E402
import tpu_session_rehearsal as rehearsal  # noqa: E402

RECORD_NAME = "tpu-null-queue.json"
OUTPUT = Path(os.environ.get("TPU_NULL_OUTPUT", f"/kaggle/working/{RECORD_NAME}"))

#: The registered null queue's own dataset root: `kaggle_t4.QUEUES["null"]` is the primary block's
#: slug plus `-null`, so these seeds publish where the campaign's null clause already looks rather
#: than into a root of this script's invention.
DATASET_ROOT = "act-v0-5-2-null"

#: The registered block, and the same twelve seeds the T4 null queue is defined over. A seed is a
#: seed: this run has to produce the queue's records, not a rate about them.
DEFAULT_SEEDS = "15024-15035"

#: Section 5.47, carried by every arm of every queue. Inert for a do-nothing arm, and carried
#: anyway so the published manifest's treatment matches the registration.
REBUILD_TREATMENT: dict[str, Any] = {
    "preserve_router_on_rebuild": True,
    "new_expert_initialization": "clone",
}

#: See the module docstring: the one field where this differs from the registered fp16 queue.
DTYPE = "fp32"

#: Checkpoints every this many steps, so a kernel ended by the platform's wall is followed by a
#: resume rather than a re-run. The declared cadence is 500; at ~1,879 tokens/s per chip a step is
#: 4,096 tokens, so this is a few minutes of work per checkpoint.
CHECKPOINT_INTERVAL = 500


def parse_seeds(spec: str) -> list[int]:
    """``15024-15035`` or ``15024,15030`` -> the seeds, in the order given."""
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, _, stop = part.partition("-")
            low, high = int(start), int(stop)
            if high < low:
                raise ValueError(f"seed range runs backwards: {part}")
            seeds.extend(range(low, high + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("no seeds given")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"a seed appears twice: {spec}")
    return seeds


def seed_config(device: str, *, seed: int, steps: int, fidelity: str):
    """The queue's seed: the declared config for a real seed, the dev one for a dry run."""
    if fidelity == "development":
        return rehearsal._dev_null_config(device, steps=steps, seed=seed)
    from act.training.v0_session import declared_session_config

    return declared_session_config(
        device=device,
        seed=seed,
        arms=("null",),
        dtype=DTYPE,
        total_steps=steps,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        max_checkpoints=2,
        **REBUILD_TREATMENT,
    )


def run_seed(
    rank: int,
    seed: int,
    *,
    device: str,
    steps: int,
    fidelity: str,
    dataset_root: Path,
    work: Path,
    revision: str,
    record: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    """Train and publish one seed on the rank's chip, keeping any failure in the record.

    One seed's failure is one seed's failure: the other seven chips are mid-seed and must not be
    taken down with it, which is why the containment is here rather than around the pool. A failed
    seed lands under `errors` with its **traceback** - the frames, not only the message - because
    this is a run a reader gets one chance to diagnose.
    """
    from act.training.v0_handoff import publish_seed
    from act.training.v0_session import run_v0_session

    entry: dict[str, Any] = {"rank": rank, "seed": seed, "device": None, "binding": None}
    started = time.time()

    def log(line: str) -> None:
        # Printed as it arrives: a kernel killed by the platform still carries these in its log.
        print(f"[seed {seed}] {line}", flush=True)

    try:
        import torch_xla

        # The chip this thread owns, bound before anything is built or stepped - see
        # `tpu_probe.bind_thread_device` for what a step marker does without it.
        before, after = tpu_probe.bind_thread_device(torch_xla, torch_xla.device(rank))
        entry["binding"] = {"before": before, "after": after}
        entry["device"] = after

        config = seed_config(device, seed=seed, steps=steps, fidelity=fidelity)
        session_dir = work / f"seed-{seed}" / "session"
        log(f"training {config.total_steps} steps, dtype {config.dtype}, on {after}")
        result = run_v0_session(
            config,
            checkpoint_dir=session_dir,
            log=log,
            code_revision=revision,
        )
        arm = result.arms[0]
        entry["arm"] = {
            "mode": arm.mode,
            "final_loss": arm.final_loss,
            "validation_loss": arm.validation_loss,
            # The arm books a rate and a wall, not a token count: the tokens are the config's own
            # (batch x context x steps, 24,576,000 at the declared shape), so the figure the queue
            # is defined in comes from the config rather than from multiplying a rate back out.
            "tokens": config.batch_size * config.context_size * config.total_steps,
            "tokens_per_second": arm.tokens_per_second,
            "wall_time_seconds": arm.wall_time_seconds,
            "search_flops": arm.search_flops,
            "mutation_count": arm.mutation_count,
        }
        publish_seed(
            dataset_root,
            result,
            session_dir=session_dir,
            code_revision=revision,
        )
        entry["published"] = True
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["traceback"] = traceback.format_exc()
        log(f"failed: {entry['error']}")
    entry["wall_seconds"] = time.time() - started

    with _lock:
        record["seeds"][str(seed)] = entry
        (record["completed"] if "error" not in entry else record["errors"]).append(seed)
        # Saved as each seed lands, not at the end: a kernel that dies at hour five still carries
        # the seeds that finished, which is the property the whole record exists for.
        tpu_probe._save(record, path)
    print(
        f"[seed {seed}] {'done' if 'error' not in entry else 'FAILED'} "
        f"in {entry['wall_seconds']:.1f}s",
        flush=True,
    )
    return entry


_lock = threading.Lock()


def device_count() -> int:
    """The chips this host actually exposes - the pool size is measured, never assumed."""
    try:
        from torch_xla.core import xla_model

        return len(xla_model.get_xla_supported_devices())
    except Exception:  # noqa: BLE001 - a diagnostic; the pool falls back to one worker
        return 1


def base_record(args: argparse.Namespace, revision: str) -> dict[str, Any]:
    """This record's labels, before any seed has run.

    ``declared_seed: true`` here is not the rehearsal's disclaimer read backwards - these *are* the
    queue's seeds, at the queue's shape, published to the queue's dataset root. What the record has
    to say instead is everything a reader needs to decide whether to import them: which dtype, which
    fidelity, which treatment, and which revision wrote them.
    """
    return {
        "record_type": "tpu-null-queue",
        "kind": "tpu-null-queue",
        "declared_seed": args.fidelity == "declared",
        "fidelity": args.fidelity,
        "declared_scale": args.fidelity == "declared",
        "queue": "null",
        "device_requested": args.device,
        "dtype": DTYPE,
        "steps": args.steps,
        "workers_requested": args.workers,
        "seeds_requested": parse_seeds(args.seeds),
        "dataset_root": str(DATASET_ROOT),
        "tpu_dataset_root_name": DATASET_ROOT,
        "rebuild_treatment": dict(REBUILD_TREATMENT),
        "code_revision": revision,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "scratch_root": (
            "each seed's session checkpoints live under --output's parent; the published seed "
            "artifacts go to the dataset root named in `tpu_dataset_root_name`"
        ),
        "completed": [],
        "errors": [],
        "seeds": {},
        "what_this_is": (
            "The registered campaign's null queue (section 5.14), one seed per chip on a TPU, "
            "trained with run_v0_session and published with publish_seed - the same path the T4 "
            "null queue runs. It differs from that queue in exactly one field, named in `dtype`: "
            "the session's registered dtypes are fp32 and fp16, and fp16 on a TPU needs an "
            "autocast and scaler path no record here shows working."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="xla", help="xla (a TPU host) or cpu")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS, help="15024-15035, or a comma list")
    parser.add_argument("--steps", type=int, default=6_000, help="the declared seed's 6,000 steps")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="chips to use; 0 means every chip the host exposes",
    )
    parser.add_argument(
        "--fidelity",
        choices=("declared", "development"),
        default="declared",
        help="development is the local dry run: the dev shape, not the queue's",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    revision = rehearsal._revision()
    record = base_record(args, revision)
    work = Path(args.output).parent
    dataset_root = work / DATASET_ROOT
    dataset_root.mkdir(parents=True, exist_ok=True)
    path = Path(args.output)

    seeds = parse_seeds(args.seeds)
    available = device_count()
    workers = args.workers if args.workers > 0 else available
    workers = max(1, min(workers, len(seeds)))
    record["devices_available"] = available
    record["workers"] = workers
    print(
        f"[queue] {len(seeds)} seed(s), {workers} worker(s) of {available} device(s), "
        f"fidelity {args.fidelity}, {args.steps} steps -> {dataset_root}",
        flush=True,
    )
    tpu_probe._save(record, path)

    started = time.time()
    # Seeds are handed to workers in order, so with more seeds than chips the waves are the
    # queue's own order rather than whichever thread happened to ask first.
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                run_seed,
                rank % workers,
                seed,
                device=args.device,
                steps=args.steps,
                fidelity=args.fidelity,
                dataset_root=dataset_root,
                work=work,
                revision=revision,
                record=record,
                path=path,
            )
            for rank, seed in enumerate(seeds)
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    record["wall_seconds"] = time.time() - started
    record["seeds_completed"] = len(record["completed"])
    tpu_probe._save(record, path)
    print(json.dumps({k: v for k, v in record.items() if k != "seeds"}, indent=1, default=str))
    return 0 if len(record["completed"]) == len(seeds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
