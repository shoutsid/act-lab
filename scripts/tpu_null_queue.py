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

**Where the roots live, and where the record lives.** A Kaggle kernel's ``/kaggle/working`` is a
20 GB loopback mount (measured: 19.5 GiB on ``/dev/loop1``) while the container's overlay has about
a terabyte free, and one completed seed publishes roughly 3.2 GiB of resume state - so the handoff
roots go to the overlay, through ``kaggle_t4.workspace_root``, which is the campaign's own answer
to exactly this (``/tmp`` on a kernel, repo-relative locally), and never beside the record. Twelve
seeds written into the working mount would fill it and lose the **record**, which is the one
artifact the fetch reads. ``--scratch`` names the root; the record stays at ``--output``.

**What a kernel does that the dry run does not.** With ``--dataset`` given (a kernel passes the
registered null queue's own id) the run follows section 5.2 the way a session does: it **restores**
the newest published version before training, **skips the seeds that version already carries**
rather than re-training over them, and **publishes** at exit through
``kaggle_remote.publish_on_exit`` - which retries and never raises, because the container ends with
the kernel and that push is the only durable copy of the block. The per-seed summaries are copied
flat into the output directory as well (``kaggle_t4._keep_seed_summaries``), so a failed push still
leaves every number the verdict reads.

    # development fidelity, one worker - the shape a local XLA:CPU dry run can carry
    PJRT_DEVICE=CPU python scripts/tpu_null_queue.py --device xla --fidelity development \
        --seeds 15024-15025 --workers 1 --steps 20
    # the queue itself, on a TPU
    python scripts/tpu_null_queue.py --device xla --seeds 15024-15035 --workers 8
    # what the pushed kernel entry runs: record into the kernel output, roots onto the overlay
    python scripts/tpu_null_queue.py --device xla --seeds 15024-15035 --workers 8 \
        --dataset sid123123234/act-v0-5-2-null \
        --output /kaggle/working/tpu-null-queue.json
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
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# Reused rather than re-implemented: the probe's per-thread device binding (and the reason for it)
# and its record saver, the rehearsal's development-fidelity config and revision lookup, and the
# campaign's own answer to "where do handoff roots live on a kernel" (`kaggle_t4.workspace_root`)
# plus the summary-keeping it does on a session's exit path.
import kaggle_t4  # noqa: E402
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

#: The campaign dataset this run publishes into, in the form ``kaggle_t4.Queue.dataset`` builds
#: (``<owner>/<slug>-null``). Unset means no remote half at all: the local dry run's behaviour,
#: which is why it is not defaulted to a dataset rather than to nothing.
DATASET: str | None = None

#: Where the working mount is, when there is one. A module constant rather than a literal inside
#: the function so the placement rule is testable without a Kaggle host.
KAGGLE_WORKING = Path("/kaggle/working")


def scratch_root(output: Path) -> Path:
    """Where this run's handoff roots go - the overlay on a kernel, beside the output locally.

    See the module docstring: the roots are gigabytes and the working mount is 20 GB, so they
    cannot share it with the record. On a kernel the answer is ``kaggle_t4.workspace_root``'s
    (``/tmp/act-lab-null-queue-<run token>``, token-scoped because the overlay is host-shared
    and two runs must never share a directory). Off a kernel it is the output's own parent, so
    a dry run stays inspectable where the caller put it.
    """
    if kaggle_t4.run_token():
        return kaggle_t4.workspace_root("null-queue")
    return Path(output).parent / f"{DATASET_ROOT}-scratch"


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
        # ``clean_working_dir``: the published copy is the durable one and a resume restores
        # from it, so holding the session's own second copy of the same checkpoint would cost
        # the block another ~3.2 GiB per seed of overlay for nothing.
        publish_seed(
            dataset_root,
            result,
            session_dir=session_dir,
            code_revision=revision,
            clean_working_dir=True,
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


def restore_published(
    dataset_root: Path, dataset: str | None, *, log: Callable[[str], None] | None = None
) -> dict[str, Any]:
    """Section 5.2's startup half: bring the block's already-published seeds local.

    A run that asked for remote state and cannot get it must stop **before** it trains, and
    this one does: ``restore_latest`` raises rather than letting a fresh block be trained and
    published over a complete one. Measured 2026-09-26/27, that misreading cost a session's
    quota and republished a seed over its own version while the driver read the same partial
    progress every time. Without ``--dataset`` there is nothing remote to ask for, and a dry
    run restores nothing.
    """
    if not dataset:
        return {"dataset": None, "restored": [], "error": None}
    from act.ops.kaggle_remote import default_client, restore_latest

    try:
        restored = restore_latest(
            dataset_root, dataset=dataset, client=default_client(), log=log
        )
    except Exception as exc:  # noqa: BLE001 - reported in the record, not swallowed
        return {"dataset": dataset, "restored": [], "error": f"{type(exc).__name__}: {exc}"}
    return {"dataset": dataset, "restored": sorted(int(seed) for seed in restored), "error": None}


def skipped_entry(dataset_root: Path, seed: int) -> dict[str, Any]:
    """A seed the restored version already carries, read back from its published summary.

    Read rather than merely counted: a block assembled over two runs still has to report every
    seed's losses, and the restored version is where the earlier run's numbers live.
    """
    entry: dict[str, Any] = {
        "seed": seed,
        "skipped": "already published in the restored version",
        "published": True,
    }
    try:
        from act.training.v0_handoff import load_seed_summary

        arm = (load_seed_summary(dataset_root, seed).get("arms") or {}).get("null") or {}
        entry["arm"] = {
            "mode": "null",
            "final_loss": arm.get("final_loss"),
            "validation_loss": arm.get("validation_loss"),
            "tokens_per_second": arm.get("tokens_per_second"),
            "wall_time_seconds": arm.get("wall_time_seconds"),
            "search_flops": arm.get("search_flops"),
            "mutation_count": arm.get("mutation_count"),
        }
    except Exception as exc:  # noqa: BLE001 - the skip still counts as complete
        entry["summary_read"] = f"{type(exc).__name__}: {exc}"
    return entry


def keep_summaries(dataset_root: Path, output_dir: Path) -> int:
    """Copy every seed's summary flat into the output directory, as a session does.

    The push is the durable copy and it is also the one step that can fail *after* hours of
    training; every number the verdict reads per seed is in these kilobyte files. Reuses the
    campaign's own keeper rather than a second spelling of "which summary is whose" - the
    publisher writes each payload twice (under the seed and under its ``reports/``), and the
    flat name has to carry the seed because they are all called ``seed-summary.json``.
    """
    return kaggle_t4._keep_seed_summaries(dataset_root, output=output_dir)


def publish_remote(
    dataset_root: Path,
    dataset: str | None,
    *,
    seeds: int,
    log: Callable[[str], None] | None = None,
    log_path: Path | None = None,
) -> dict[str, Any]:
    """Push the block to its dataset on the exit path; never raises.

    This is the only durable copy of a twelve-seed block - the roots live in the container
    overlay, which ends with the kernel - so the outcome is *recorded* rather than trusted, and
    a failure here is a fact the record carries, not an exception that loses the run's account
    of itself. ``publish_on_exit`` already retries and never raises; the guard here is for the
    steps around it (building the client can fail with no credentials at all).
    """
    if not dataset:
        return {"dataset": None, "status": "not_requested"}
    try:
        from act.ops.kaggle_remote import default_client, publish_on_exit

        result = publish_on_exit(
            dataset_root,
            dataset=dataset,
            client=default_client(),
            version_notes=f"tpu null queue: {seeds} seed version(s)",
            log=log,
            log_path=log_path,
        )
        return result.to_dict()
    except Exception as exc:  # noqa: BLE001 - the exit path never raises
        return {
            "dataset": dataset,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def verdict(record: dict[str, Any]) -> str:
    """One line: how many of the queue's seeds landed, and what they trained to.

    Stated in the null clause's own terms - this is the do-nothing arm, so the fields that say
    so are ``search_flops`` and ``mutation_count``, and the losses are what the primary record
    gets paired against. It is the fetch's read, so it cannot be left to a second hand-run
    command (`artifacts/tpu-probe/NOTES.md` records what that costs).
    """
    seeds = record.get("seeds") or {}
    if not seeds:
        return "no seeds: the run left no per-seed entry to read"
    arms = [entry["arm"] for entry in seeds.values() if isinstance(entry.get("arm"), dict)]
    losses = [float(arm["final_loss"]) for arm in arms if arm.get("final_loss") is not None]
    zeroed = sum(
        1 for arm in arms if arm.get("search_flops") == 0 and arm.get("mutation_count") == 0
    )
    rates = [float(arm.get("tokens_per_second") or 0.0) for arm in arms]
    line = (
        f"{record.get('seeds_completed', 0)} of {len(record.get('seeds_requested') or [])} "
        f"seed(s) landed in {record.get('dtype', 'unknown')} on XLA; {zeroed} of {len(arms)} "
        "arm(s) do nothing (search_flops 0, mutation_count 0)"
    )
    if losses:
        line += (
            f"; final loss mean {sum(losses) / len(losses):.4f}, min {min(losses):.4f}, "
            f"max {max(losses):.4f}"
        )
    if rates and max(rates) > 0:
        line += f"; {sum(rates) / len(rates):,.1f} tokens/s per chip"
    errors = record.get("errors") or []
    if errors:
        line += f"; {len(errors)} error(s): {errors[:4]}"
    return line


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
        "dataset": args.dataset,
        "rebuild_treatment": dict(REBUILD_TREATMENT),
        "code_revision": revision,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "scratch_root": (
            "the seed checkpoints and the dataset root live under --scratch (the container "
            "overlay on a kernel), never beside the record: the block is ~38 GiB and the "
            "working mount is 20 GB"
        ),
        "clean_working_dir_after_publish": True,
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
    parser.add_argument(
        "--dataset",
        default=DATASET,
        help="campaign dataset to restore from and publish into; unset = no remote half",
    )
    parser.add_argument(
        "--scratch",
        type=Path,
        default=None,
        help="where the handoff roots go; default keeps them off the kernel's working mount",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    def log(line: str) -> None:
        # Printed as it arrives, because a kernel the platform kills still carries these.
        print(f"[queue] {line}", flush=True)

    revision = rehearsal._revision()
    record = base_record(args, revision)
    path = Path(args.output)
    scratch = Path(args.scratch) if args.scratch else scratch_root(path)
    scratch.mkdir(parents=True, exist_ok=True)
    dataset_root = scratch / DATASET_ROOT
    dataset_root.mkdir(parents=True, exist_ok=True)
    record["scratch"] = str(scratch)

    seeds = parse_seeds(args.seeds)
    # Section 5.2 before a token is trained: what the queue already carries is restored, and a
    # restore that fails on a dataset the account owns stops the run rather than re-training
    # over it. A block assembled over two runs reports the earlier run's seeds too.
    record["restore"] = restore_published(dataset_root, args.dataset, log=log)
    if record["restore"]["error"]:
        record["errors"].append("restore")
        record["seeds_completed"] = 0
        tpu_probe._save(record, path)
        print(json.dumps({k: v for k, v in record.items() if k != "seeds"}, indent=1, default=str))
        return 2
    published = set(record["restore"]["restored"])
    pending = [seed for seed in seeds if seed not in published]
    for seed in seeds:
        if seed in published:
            record["seeds"][str(seed)] = skipped_entry(dataset_root, seed)
            record["completed"].append(seed)

    available = device_count()
    workers = args.workers if args.workers > 0 else available
    workers = max(1, min(workers, len(pending)))
    record["devices_available"] = available
    record["workers"] = workers
    log(
        f"{len(seeds)} seed(s) requested, {len(published)} already published, "
        f"{len(pending)} to train; {workers} worker(s) of {available} device(s), "
        f"fidelity {args.fidelity}, {args.steps} steps -> {dataset_root}"
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
                work=scratch,
                revision=revision,
                record=record,
                path=path,
            )
            for rank, seed in enumerate(pending)
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    record["wall_seconds"] = time.time() - started
    record["seeds_completed"] = len(record["completed"])
    # Exit path, in the order a session's is: keep the kilobyte summaries in the output first
    # (so a failed push still leaves the losses), then push, then write the outcome down.
    kept = keep_summaries(dataset_root, path.parent)
    log(f"kept {kept} seed summaries in {path.parent}")
    publish_log = path.parent / "tpu-null-queue-publish-log.json"
    record["publish"] = publish_remote(
        dataset_root, args.dataset, seeds=len(pending), log=log, log_path=publish_log
    )
    log(f"publish: {record['publish'].get('status')}")
    tpu_probe._save(record, path)
    print(json.dumps({k: v for k, v in record.items() if k != "seeds"}, indent=1, default=str))
    print(verdict(record))
    return 0 if len(record["completed"]) == len(seeds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
