"""Fast development cycle: the dev-shape session on the real TinyStories stream.

This is the iteration loop, not protocol machinery. It runs the smallest
measured-fastest shape (``V0SessionConfig`` defaults, ~14.9M total / 7.6M
active) for a few hundred steps on the *real* token stream so a change can be
felt in minutes instead of a declared 285M run. Nothing here is a registered
measurement: the declared target is built by ``declared_session_config()`` and
only a declared run at declared fidelity may be recorded as section 5.1.

Config knobs the loop actually needs (each one is a local experiment, not a
protocol decision):

    PYTHONPATH=src python scripts/dev_cycle.py                  # 400 steps
    PYTHONPATH=src python scripts/dev_cycle.py --steps 40       # quick smoke
    PYTHONPATH=src python scripts/dev_cycle.py --device cpu
    PYTHONPATH=src python scripts/dev_cycle.py --dtype fp16
    PYTHONPATH=src python scripts/dev_cycle.py --synthetic      # no real text

Precision: keep the dev loop on fp32. The development card (GTX 1070, compute
capability 6.1, Pascal) is fastest in fp32 by a wide margin, and the reason is
NOT guard-scaler overhead - measured at the dev shape (batch 8 x context 512,
forward+backward+AdamW, steady state):

| dtype | guard scaler | ms/step | tokens/s |
|---|---|---|---|
| fp32 | - | 160 | 25,525 |
| fp16 | on | 389 | 10,519 |
| fp16 | off | 383 | 10,688 |
| bf16 | off | 199 | 20,574 |
| bf16 | on | 200 | 20,481 |

Two things fall out. The guard scaler costs fp16 ~6 ms/step (1.5%), so bf16 -
which needs no scaler - would not recover the fp16 penalty. And bf16 is only
emulated here (``torch.cuda.is_bf16_supported(including_emulation=False)`` is
False below capability 8.0), so it casts to fp32 and still lands 25% behind
plain fp32. FP16 is slow because Pascal's fp16 arithmetic runs at 1/64 the fp32
rate, which no dtype switch can fix.

This is a property of *this* card, not of the protocol. The dtype rule is a
throughput gate, so it stays as-is: fp16 is the right pick on the declared
Turing-class target (native fast fp16), and bf16 only pays from Ampere
(capability 8.0) onward - it is emulated on Turing too, so it does not help the
Kaggle target either.

The real stream is the TinyStories token stream built by
``scripts/train_real_lm.py`` (vocab 8192, matching the dev config's vocab, so no
embedding resize is needed). When it is absent the run falls back to the
synthetic uniform-token path and says so, because a dev cycle that silently
trained on noise would make every loss number here meaningless.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from act.model import estimated_parameter_count
from act.training import V0SessionConfig, run_v0_session
from act.training.runtime import resolve_device

ROOT = Path(__file__).resolve().parent.parent
REAL_TRAIN = ROOT / "artifacts" / "real-lm" / "tokens.npy"
REAL_VAL = ROOT / "artifacts" / "real-lm" / "val-tokens.npy"
CHECKPOINT_DIR = ROOT / "artifacts" / "dev-cycle" / "sessions"


def _real_streams() -> tuple[Path, Path] | None:
    """The TinyStories streams, or None when they have not been built."""
    if REAL_TRAIN.exists() and REAL_VAL.exists():
        return REAL_TRAIN, REAL_VAL
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400, help="training steps per arm")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    parser.add_argument("--dtype", default="fp32", help="fp32 | fp16")
    parser.add_argument("--seed", type=int, default=15000)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="train on uniform tokens instead of the real token stream",
    )
    args = parser.parse_args()

    streams = None if args.synthetic else _real_streams()
    overrides: dict = {
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "total_steps": args.steps,
    }
    if streams is not None:
        train, val = streams
        overrides.update(data_path=str(train), val_data_path=str(val))
    config = V0SessionConfig(**overrides)

    device = resolve_device(config.device)
    model_config = config.model_config()
    total, active = estimated_parameter_count(model_config)
    source = config.data_path if config.data_path else "synthetic uniform tokens"
    print(
        f"dev cycle: {total:,} total / {active:,} active parameters, "
        f"{config.training_tokens():,} tokens per arm\n"
        f"  device: {device}   dtype: {config.dtype}   steps: {config.total_steps}\n"
        f"  data:   {source}"
    )
    if config.data_path is None and not args.synthetic:
        print(
            "  NOTE: real stream not found at artifacts/real-lm/tokens.npy; "
            "run scripts/train_real_lm.py to build it"
        )

    # Model construction is part of what a dev cycle should time, but a warm-up
    # pass is not: the first step pays kernel autotuning that no later step does.
    started = time.perf_counter()
    result = run_v0_session(
        config,
        checkpoint_dir=CHECKPOINT_DIR,
        log=lambda line: print(f"  [session] {line}"),
    )
    wall = time.perf_counter() - started

    print("\n" + "-" * 68)
    print(f"{'arm':<10}{'final':>10}{'val':>10}{'mutations':>11}{'tok/s':>11}")
    for arm in result.arms:
        print(
            f"{arm.mode:<10}{arm.final_loss:>10.4f}{arm.validation_loss:>10.4f}"
            f"{arm.mutation_count:>11}{arm.tokens_per_second:>11,.0f}"
        )
    tokens = config.training_tokens() * len(result.arms)
    print("-" * 68)
    print(
        f"wall {wall:.1f}s for {tokens:,} tokens across {len(result.arms)} arm(s) "
        f"({tokens / wall:,.0f} tokens/s end to end)"
    )
    print(f"probe share of training FLOPs: {100 * result.predicted_cost_ratio:.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
