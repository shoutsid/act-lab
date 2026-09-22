"""Rehearse the null queue's two-seeds-per-pair amendment: does concurrency move a seed?

The null queue is the campaign's only **one-arm** queue (`arms=1, arm_slots=1`), so its
session trains one seed on one device while the second T4 is idle for the whole pair: 12
seeds x 1.4143 h = **16.97 GPU-h** for a third of the campaign's sessions and a quarter of its
arms. Two seeds per pair - one arm each, on the two devices - costs the same pair time and
completes two seeds, so the recorded amendment Option 1 is worth **8.49 GPU-h**, and it is
also what lets a *single* queue on a TPU still deliver a one-week block (see
``tpu_probe.week_capacity``).

What the amendment needs is not a new measurement of the hardware but an answer to one
question: **does a seed's result depend on what else is running beside it?** Today's pairs
already run two arms in two processes, and `run_v0_session` trains each arm in its own
subprocess, so the mechanism is in place - an arm is bit-identical with and without its pair
(measured 2026-09-19 at the dev shape). What is *not* measured is the amendment's own
configuration: two **seeds** rather than one seed's two arms, over the **whole** result rather
than the validation loss alone. That is what this rehearsal adds, and it is free.

Method: run two seeds' null arms in separate processes, once one after the other and once at
the same time, and compare every field that *is* the result. Timing and placement
(``wall_time_seconds``, ``tokens_per_second``, ``device``) are metadata the arm carries so a
reader can tell a concurrent pair from a sequential one - they are exactly what concurrency is
allowed to move, so they are excluded. Threads are pinned identically in both scenarios, so
concurrency is the only variable: on CPU a different thread count would change a reduction's
blocking and could move a loss by itself, which would make "sequentially and concurrently
agree" untrue for the wrong reason.

The declared host is two T4s, one seed per device with no shared compute; this runs on CPU,
where the two processes *do* contend, so its passing is the conservative case. A failure here
means the amendment's premise is wrong and the option is dead until it is understood.

    python scripts/two_seed_concurrency_rehearsal.py --steps 40

Writes ``artifacts/null-pair-rehearsal/rehearsal.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUTPUT = ROOT / "artifacts" / "null-pair-rehearsal" / "rehearsal.json"

#: What an arm carries about *how* it ran rather than what it produced. Concurrency is
#: allowed to move every one of these, which is why they are named once, here, and why the
#: compared set is asserted to be exactly what :class:`V0ArmResult` carries minus this.
METADATA_FIELDS = (
    "wall_time_seconds",
    "tokens_per_second",
    "device",
)

#: The fields that are the *result*. Everything an arm carries about how it was placed or how
#: long it took is metadata, and concurrency is allowed to change it.
RESULT_FIELDS = (
    "mode",
    "final_loss",
    "validation_loss",
    "quality",
    "mutation_count",
    "parameter_count",
    "active_parameter_count",
    "training_flops",
    "search_flops",
    "architecture_hash",
    "mutation_history",
)


def one_seed(*, seed: int, steps: int, batch_size: int, threads: int, out: Path) -> None:
    """Train one seed's null arm in this process and write its result.

    This is the child: the amendment's unit is one seed in one process, so the rehearsal has
    to cross a real process boundary for the comparison to mean anything.
    """
    import torch

    # Before anything is built: the two scenarios have to differ in concurrency and nothing
    # else, and a thread count difference moves a CPU reduction on its own.
    torch.set_num_threads(threads)

    from act.training.v0_session import V0SessionConfig, run_v0_session

    config = V0SessionConfig(
        seed=seed,
        arms=("null",),
        device="cpu",
        batch_size=batch_size,
        total_steps=steps,
    )
    result = run_v0_session(config, checkpoint_dir=out / f"seed-{seed}")
    payload = {
        "seed": seed,
        "arms": [
            {field: getattr(arm, field) for field in RESULT_FIELDS} for arm in result.arms
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"seed-{seed}.json").write_text(json.dumps(payload, indent=2) + "\n")


def compare(sequential: dict, concurrent: dict) -> list[str]:
    """Every field that moved between the two scenarios, as readable lines.

    Pure, and separated from the run so a test can check that the comparison is not vacuous:
    a check that cannot see a difference would report agreement for a broken amendment.
    """
    differences: list[str] = []
    if sequential.get("seed") != concurrent.get("seed"):
        differences.append(f"seed: {sequential.get('seed')} vs {concurrent.get('seed')}")
    for index, (left, right) in enumerate(
        zip(sequential.get("arms") or [], concurrent.get("arms") or [])
    ):
        for field in RESULT_FIELDS:
            if left.get(field) != right.get(field):
                differences.append(
                    f"seed {sequential.get('seed')} arm {index} {field}: "
                    f"{left.get(field)!r} vs {right.get(field)!r}"
                )
    if len(sequential.get("arms") or []) != len(concurrent.get("arms") or []):
        differences.append(
            f"arm count: {len(sequential.get('arms') or [])} vs "
            f"{len(concurrent.get('arms') or [])}"
        )
    return differences


def _child_env(threads: int) -> dict[str, str]:
    return dict(os.environ, OMP_NUM_THREADS=str(threads), PYTHONUNBUFFERED="1")


def _child_argv(seed: int, args: argparse.Namespace, out: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--one-seed",
        str(seed),
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--threads",
        str(args.threads),
        "--work",
        str(out),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="15024,15025", help="the two seeds to rehearse")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2, help="pinned per process")
    parser.add_argument("--work", type=Path, default=ROOT / "artifacts" / "null-pair-rehearsal" / "work")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--one-seed", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.one_seed is not None:
        one_seed(
            seed=args.one_seed,
            steps=args.steps,
            batch_size=args.batch_size,
            threads=args.threads,
            out=args.work,
        )
        return 0

    seeds = [int(seed) for seed in args.seeds.split(",")]
    if len(seeds) != 2:
        raise SystemExit("the amendment pairs exactly two seeds; pass two")
    env = _child_env(args.threads)
    # One directory per scenario. Sharing one would make the check vacuous twice over: the
    # comparison would re-read the first scenario's files, and `run_v0_session` *resumes* a
    # seed whose checkpoint root already exists, so the second scenario would not be a run at
    # all - it would be the first one's result, written again.
    sequential_dir = args.work / "sequential"
    concurrent_dir = args.work / "concurrent"
    for directory in (sequential_dir, concurrent_dir):
        if directory.exists():
            shutil.rmtree(directory)

    print(f"=== two-seed concurrency rehearsal: seeds {seeds} ===", flush=True)
    print(
        f"dev shape, {args.steps} steps, batch {args.batch_size}, "
        f"{args.threads} thread(s) per process, device cpu",
        flush=True,
    )

    started = time.perf_counter()
    for seed in seeds:  # sequentially: one process, then the other
        subprocess.run(_child_argv(seed, args, sequential_dir), env=env, check=True)
    sequential_seconds = time.perf_counter() - started

    started = time.perf_counter()
    running = [
        subprocess.Popen(_child_argv(seed, args, concurrent_dir), env=env) for seed in seeds
    ]
    codes = [process.wait() for process in running]
    concurrent_seconds = time.perf_counter() - started
    if any(code != 0 for code in codes):
        raise SystemExit(f"a concurrent child failed: exit codes {codes}")

    sequential = {
        seed: json.loads((sequential_dir / f"seed-{seed}.json").read_text())
        for seed in seeds
    }
    concurrent = {
        seed: json.loads((concurrent_dir / f"seed-{seed}.json").read_text())
        for seed in seeds
    }
    differences: list[str] = []
    for seed in seeds:
        differences += compare(sequential[seed], concurrent[seed])

    record = {
        "what_this_is": (
            "Rehearsal of the null queue's two-seeds-per-pair amendment: two seeds' null "
            "arms trained sequentially and concurrently in separate processes, compared on "
            "every field that is the result."
        ),
        "seeds": seeds,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "threads_per_process": args.threads,
        "device": "cpu",
        "result_fields_compared": list(RESULT_FIELDS),
        "excluded_as_metadata": list(METADATA_FIELDS),
        "sequential_seconds": round(sequential_seconds, 2),
        "concurrent_seconds": round(concurrent_seconds, 2),
        "identical": not differences,
        "differences": differences,
        "runs": {"sequential": sequential, "concurrent": concurrent},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    print(
        f"\nsequential {sequential_seconds:.1f} s | concurrent {concurrent_seconds:.1f} s"
    )
    for seed in seeds:
        arm = sequential[seed]["arms"][0]
        print(
            f"seed {seed}: validation_loss {arm['validation_loss']!r} "
            f"architecture_hash {arm['architecture_hash'][:12]} "
            f"training_flops {arm['training_flops']}"
        )
    if differences:
        print(f"\nDIFFERENT on {len(differences)} field(s):")
        for line in differences:
            print(f"  {line}")
        print("\nthe amendment's premise does not hold at this shape")
        return 1
    print("\nbit-identical on every result field: concurrency did not move either seed")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
