#!/usr/bin/env python3
"""Capacity-threshold sweep: test different mutation policies across model scales.

Measures whether local topology lookahead (adaptive probe argmin) beats
random churn, null, and directional controls at 20M/40M/60M/100M.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from act.model.v0_moe import V0MoEConfig, estimated_parameter_count
from act.training.v0_session import V0SessionConfig, run_v0_session

SCALES = {
    "20M": dict(
        vocab_size=8192, context_size=512, hidden_size=384, depth=6,
        attention_heads=6, expert_count=8, active_experts=2, expert_width=384,
    ),
    "40M": dict(
        vocab_size=8192, context_size=512, hidden_size=512, depth=6,
        attention_heads=8, expert_count=8, active_experts=2, expert_width=512,
    ),
    "60M": dict(
        vocab_size=8192, context_size=512, hidden_size=576, depth=8,
        attention_heads=9, expert_count=8, active_experts=2, expert_width=576,
    ),
    "100M": dict(
        vocab_size=8192, context_size=2048, hidden_size=768, depth=8,
        attention_heads=12, expert_count=8, active_experts=2, expert_width=768,
    ),
}

SEEDS = [42, 43, 44]
STEPS = 400
MUTATION_INTERVAL = 100
MUTATION_BUDGET = 3
BATCH_SIZE = 2
DATA_PATH = str(Path(__file__).resolve().parent.parent / "artifacts" / "real-lm" / "tokens.npy")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "capacity-sweep"

# Arms: (name, V0SessionConfig overrides)
ARM_CONFIGS = {
    "null": dict(arms=("null",)),
    "random": dict(arms=("random", "adaptive"), probe_selection="random"),
    "adaptive": dict(arms=("random", "adaptive"), probe_selection="argmin"),
    "anti": dict(arms=("random", "adaptive"), probe_selection="argmax"),
}


@dataclass
class SweepResult:
    scale: str
    seed: int
    arm: str
    total_params: int
    active_params: int
    loss: float
    mutations: int
    search_flops: float
    training_flops: float
    elapsed_s: float


def make_config(
    scale_name: str,
    scale_kwargs: dict,
    seed: int,
    arm_name: str,
    arm_overrides: dict,
    *,
    steps: int = STEPS,
) -> V0SessionConfig:
    is_null = arm_name == "null"
    return V0SessionConfig(
        seed=seed,
        arms=arm_overrides.get("arms", ("null",)),
        device="auto",
        vocab_size=scale_kwargs["vocab_size"],
        context_size=scale_kwargs["context_size"],
        hidden_size=scale_kwargs["hidden_size"],
        depth=scale_kwargs["depth"],
        attention_heads=scale_kwargs["attention_heads"],
        expert_count=scale_kwargs["expert_count"],
        active_experts=scale_kwargs["active_experts"],
        expert_width=scale_kwargs["expert_width"],
        batch_size=BATCH_SIZE,
        total_steps=steps,
        learning_rate=3e-4,
        mutation_interval=MUTATION_INTERVAL,
        mutation_budget=MUTATION_BUDGET,
        probe_candidate_cap=4,
        probe_candidate_policy="spread",
        probe_selection=arm_overrides.get("probe_selection", "argmin"),
        probe_batch_count=1,
        preserve_router_on_rebuild=True,
        new_expert_initialization="clone",
        dtype="fp32",
        declared=False,
        data_path=DATA_PATH,
    )


# Which session arm each sweep arm reports.
#
# ``run_v0_session`` returns its arms in ``config.arms`` order, and the three
# probe-selection arms above are all configured as ``("random", "adaptive")`` so that
# ``probe_selection`` - the only thing that distinguishes them - governs the
# ``adaptive`` arm. Reading ``arms[0]`` therefore recorded the plain random-churn arm
# for every sweep arm: the ``adaptive`` and ``anti`` rows carried the ``random`` arm's
# numbers, bit-identical to it, with the ``probe_selection`` that told them apart
# never read. The recorded arm is now selected by name rather than by position, and a
# mode the session did not return is an error rather than a silently wrong row.
#
# The plain random arm is still trained inside the adaptive and anti sessions and the
# sweep discards its result - 2x the compute for those rows. That is a study-design
# question rather than this defect, so it is left as it is.
ARM_MODE = {
    "null": "null",
    "random": "random",
    "adaptive": "adaptive",
    "anti": "adaptive",
}


def run_one(config: V0SessionConfig, scale_name: str, arm_name: str) -> SweepResult:
    cp_dir = f"artifacts/v0-sessions/{scale_name}-seed-{config.seed}-{arm_name}"
    t0 = time.monotonic()
    result = run_v0_session(config, checkpoint_dir=cp_dir)
    elapsed = time.monotonic() - t0

    total, active = estimated_parameter_count(config.model_config())
    wanted = ARM_MODE[arm_name]
    arm = next(
        (asdict(candidate) for candidate in result.arms if candidate.mode == wanted),
        None,
    )
    if arm is None:
        raise RuntimeError(
            f"sweep arm {arm_name!r} reports mode {wanted!r}, which the session did "
            f"not return (it returned {[a.mode for a in result.arms]})"
        )

    return SweepResult(
        scale=scale_name,
        seed=config.seed,
        arm=arm_name,
        total_params=total,
        active_params=active,
        loss=arm["final_loss"],
        mutations=arm["mutation_count"],
        search_flops=arm["search_flops"],
        training_flops=arm["training_flops"],
        elapsed_s=elapsed,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The sweep's axes, so an argument is honoured or rejected rather than ignored.

    Without a parser the script read no arguments at all: ``--help`` began a full
    sweep, and a misspelled flag was dropped while the default run went ahead. The
    selection also decides what gets reset, which matters because the reset has to
    name the directories the runs actually write.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scales",
        default=",".join(SCALES),
        help="comma-separated; one of " + ", ".join(SCALES),
    )
    parser.add_argument(
        "--arms",
        default=",".join(ARM_CONFIGS),
        help="comma-separated; one of " + ", ".join(ARM_CONFIGS),
    )
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    return parser.parse_args(argv)


def _selected(raw: str, known, flag: str) -> list[str]:
    """Names from a comma-separated flag, refused rather than dropped when unknown."""
    wanted = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise SystemExit(f"{flag}: unknown name(s) {unknown}; choose from {list(known)}")
    if not wanted:
        raise SystemExit(f"{flag}: no name given; choose from {list(known)}")
    return wanted


def main() -> None:
    import shutil
    args = parse_args()
    scales = _selected(args.scales, SCALES, "--scales")
    arms = _selected(args.arms, ARM_CONFIGS, "--arms")
    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    if not seeds:
        raise SystemExit("--seeds: no seed given; pass e.g. --seeds 42")
    output_dir: Path = args.out
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reset the directories the runs *write* - ``<scale>-seed-<seed>-<arm>``, which
    # ``run_one`` passes as ``checkpoint_dir`` - rather than ``<scale>-seed-<seed>``,
    # which no run ever creates. ``run_v0_session`` resumes from an existing
    # checkpoint directory, so the old glob reset nothing and every re-run continued
    # from whatever the previous one left behind, making the sweep's numbers depend
    # on it.
    for scale in scales:
        for seed in seeds:
            for arm in arms:
                shutil.rmtree(
                    f"artifacts/v0-sessions/{scale}-seed-{seed}-{arm}", ignore_errors=True
                )

    all_results: list[SweepResult] = []

    for scale_name in scales:
        scale_kwargs = SCALES[scale_name]
        print(f"\n{'='*60}")
        print(f"SCALE: {scale_name}")
        print(f"{'='*60}")

        for seed in seeds:
            for arm_name in arms:
                arm_overrides = ARM_CONFIGS[arm_name]
                print(f"  {scale_name} / seed={seed} / {arm_name} ... ", end="", flush=True)
                try:
                    config = make_config(
                        scale_name, scale_kwargs, seed, arm_name, arm_overrides,
                        steps=args.steps,
                    )
                    total, active = estimated_parameter_count(config.model_config())
                    result = run_one(config, scale_name, arm_name)
                    all_results.append(result)
                    print(f"loss={result.loss:.4f} mut={result.mutations} {result.elapsed_s:.1f}s")
                except Exception as e:
                    print(f"FAILED: {e}")

    # A run that fails is printed above and skipped, which is useful while a sweep is
    # running - but a file written from no results at all is not a record of anything.
    # The horizon arm already cost a session to this shape: every run in a stage was
    # lost and the stage still reported success, because the file was written either
    # way.
    if not all_results:
        raise SystemExit(
            "no run produced a result, so there is nothing to write; the failures "
            "above are the whole story"
        )

    results_path = output_dir / "threshold_sweep.json"
    with open(results_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nResults saved to {results_path}")

    print(f"\n{'='*90}")
    print("SUMMARY: Adaptive vs Random vs Null vs Anti (avg over seeds)")
    print(f"{'='*90}")
    print(f"{'Scale':<8} {'Null':>8} {'Random':>8} {'Adaptive':>8} {'Anti':>8}  {'A-R':>8} {'A-N':>8}")
    print("-" * 72)

    for scale_name in scales:
        by_arm = {}
        for r in all_results:
            if r.scale == scale_name:
                by_arm.setdefault(r.arm, []).append(r.loss)

        avg = {a: sum(v)/len(v) for a, v in by_arm.items()}
        a_r = avg.get("adaptive", 0) - avg.get("random", 0)
        a_n = avg.get("adaptive", 0) - avg.get("null", 0)
        print(f"{scale_name:<8} {avg.get('null',0):>8.4f} {avg.get('random',0):>8.4f} "
              f"{avg.get('adaptive',0):>8.4f} {avg.get('anti',0):>8.4f}  {a_r:>+8.4f} {a_n:>+8.4f}")

    print(f"\n{'='*90}")
    print("PAIR WINS (adaptive vs random, per seed)")
    print(f"{'='*90}")
    for scale_name in scales:
        wins = 0
        total = 0
        for seed in seeds:
            a = [r for r in all_results if r.scale == scale_name and r.seed == seed and r.arm == "adaptive"]
            b = [r for r in all_results if r.scale == scale_name and r.seed == seed and r.arm == "random"]
            if a and b:
                total += 1
                if a[0].loss < b[0].loss:
                    wins += 1
        print(f"  {scale_name}: adaptive wins {wins}/{total} pairs")


if __name__ == "__main__":
    main()
