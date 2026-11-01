"""One seed of one width arm, at development fidelity, on the real token stream.

This is the pilot's unit of work, not a registered instrument: it runs the
session machinery at the dev shape (h320, 400 steps, events at 100/200/300) on
TinyStories, one process per (seed, arm), so a four-core Kaggle CPU session can
run four at once. The measured shape behind that choice is in
``docs/width-control-design.md`` section 10: twenty threads buy 3.2x over one,
so four single-thread processes are the cheap configuration.

The arm is a session arm in the ordinary sense (``V0SessionConfig.arms``):

    PYTHONPATH=src python scripts/width_pilot.py --seed 71000 --arm width
    PYTHONPATH=src python scripts/width_pilot.py --seed 71000 --arm width_random

``--data-dir`` must contain ``tokens.npy`` and ``val-tokens.npy``; on Kaggle
that is the mounted ``act-lab-tinystories-tokens`` dataset. A missing stream is
a refusal rather than a fallback to synthetic tokens, because a pilot that
silently trained on noise would make every loss it reports meaningless.

The record is one JSON file per job. It carries the session config, the arm
result (final held-out loss, parameter counts, the mutation history with each
graft's measured price), and the final genome's width - the last one is the
check on the resume path, since a checkpoint whose width moved is only
resumable if its genome moved with it.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from act.training import V0SessionConfig, run_v0_session
from act.training.v0_session import genome_from_checkpoint

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIRS = (Path("/kaggle/input"), ROOT / "artifacts" / "real-lm")


def _find_stream(start: Path) -> Path | None:
    """A directory holding ``tokens.npy`` beside ``val-tokens.npy``."""

    if not start.is_dir():
        return None
    for candidate in sorted(start.rglob("tokens.npy")):
        if (candidate.parent / "val-tokens.npy").is_file():
            return candidate.parent
    return None


def _data_dir(explicit: str | None) -> Path:
    if explicit is not None:
        directory = Path(explicit)
        if not (directory / "tokens.npy").is_file():
            raise SystemExit(f"no tokens.npy under {directory}")
        if not (directory / "val-tokens.npy").is_file():
            raise SystemExit(f"no val-tokens.npy beside {directory / 'tokens.npy'}")
        return directory
    for start in DEFAULT_DATA_DIRS:
        found = _find_stream(start)
        if found is not None:
            return found
    raise SystemExit(
        "no real token stream found; pass --data-dir (tokens.npy + val-tokens.npy). "
        "The pilot refuses synthetic tokens: a loss on noise is not a pilot result."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=71000)
    parser.add_argument("--arm", choices=("width", "width_random"), default="width")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--heads-added", type=int, default=2)
    parser.add_argument("--perturbation-scale", type=float, default=1e-3)
    parser.add_argument("--accept-margin", type=float, default=0.0)
    parser.add_argument("--random-probability", type=float, default=0.5)
    parser.add_argument(
        "--data-dir",
        default=None,
        help="directory with tokens.npy and val-tokens.npy (default: /kaggle/input)",
    )
    parser.add_argument("--checkpoint-dir", default="/tmp/width-pilot-checkpoints")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="host threads for this process (1: four processes per four-core session)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Before any model construction: the measured shape assumes one thread per
    # process, which is what makes four single-thread processes the cheap form.
    torch.set_num_threads(args.threads)
    data = _data_dir(args.data_dir)
    config = V0SessionConfig(
        seed=args.seed,
        arms=(args.arm,),
        device="cpu",
        dtype="fp32",
        total_steps=args.steps,
        mutation_interval=args.interval,
        mutation_budget=args.budget,
        width_heads_added=args.heads_added,
        width_perturbation_scale=args.perturbation_scale,
        width_accept_margin=args.accept_margin,
        width_random_probability=args.random_probability,
        data_path=str(data / "tokens.npy"),
        val_data_path=str(data / "val-tokens.npy"),
    )

    print(
        f"width pilot: seed {args.seed} arm {args.arm} "
        f"{config.training_tokens():,} tokens, events at "
        f"{[step for step in range(config.mutation_interval, args.steps + 1, config.mutation_interval)][: args.budget]}",
        flush=True,
    )
    started = time.time()
    result = run_v0_session(
        config,
        checkpoint_dir=Path(args.checkpoint_dir) / f"{args.arm}-{args.seed}",
        log=lambda line: print(f"  [{args.arm}] {line}", flush=True),
    )
    wall = time.time() - started
    arm = result.arms[0]

    checkpoint = (
        Path(args.checkpoint_dir)
        / f"{args.arm}-{args.seed}"
        / f"seed-{args.seed}"
        / f"arm-{args.arm}"
        / f"checkpoint-{args.steps:06d}.pt"
    )
    final_genome = (
        genome_from_checkpoint(checkpoint) if checkpoint.is_file() else None
    )
    record = {
        "kind": "width-pilot",
        "date": "2026-11-01",
        "seed": args.seed,
        "arm": args.arm,
        "fidelity": "development",
        "data": str(data),
        "config": result.config,
        "predicted_cost_ratio": result.predicted_cost_ratio,
        "wall_seconds": wall,
        "final_hidden_size": final_genome.hidden_size if final_genome else None,
        "final_heads": (
            final_genome.layers[0].heads if final_genome else None
        ),
        "arm_result": asdict(arm),
    }
    out = args.out
    if out is None:
        out = Path("/kaggle/working/width-pilot") / f"{args.arm}-seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        f"  final: loss {arm.validation_loss:.6f}  parameters {arm.parameter_count:,}  "
        f"mutations {arm.mutation_count}  {arm.tokens_per_second:,.0f} tok/s  "
        f"{wall / 60:.1f} min",
        flush=True,
    )
    print(f"  wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
