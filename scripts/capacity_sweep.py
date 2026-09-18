#!/usr/bin/env python3
"""Capacity-threshold sweep: test different mutation policies across model scales.

Measures whether local topology lookahead (adaptive probe argmin) beats
random churn, null, and directional controls at 20M/40M/60M/100M.
"""
from __future__ import annotations

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


def make_config(scale_name: str, scale_kwargs: dict, seed: int, arm_name: str, arm_overrides: dict) -> V0SessionConfig:
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
        total_steps=STEPS,
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


def run_one(config: V0SessionConfig, scale_name: str, arm_name: str) -> SweepResult:
    cp_dir = f"artifacts/v0-sessions/{scale_name}-seed-{config.seed}-{arm_name}"
    t0 = time.monotonic()
    result = run_v0_session(config, checkpoint_dir=cp_dir)
    elapsed = time.monotonic() - t0

    total, active = estimated_parameter_count(config.model_config())
    arm = asdict(result.arms[0])

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


def main() -> None:
    import shutil
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for scale in SCALES:
        for s in SEEDS:
            shutil.rmtree(f"artifacts/v0-sessions/{scale}-seed-{s}", ignore_errors=True)

    all_results: list[SweepResult] = []

    for scale_name, scale_kwargs in SCALES.items():
        print(f"\n{'='*60}")
        print(f"SCALE: {scale_name}")
        print(f"{'='*60}")

        for seed in SEEDS:
            for arm_name, arm_overrides in ARM_CONFIGS.items():
                print(f"  {scale_name} / seed={seed} / {arm_name} ... ", end="", flush=True)
                try:
                    config = make_config(scale_name, scale_kwargs, seed, arm_name, arm_overrides)
                    total, active = estimated_parameter_count(config.model_config())
                    result = run_one(config, scale_name, arm_name)
                    all_results.append(result)
                    print(f"loss={result.loss:.4f} mut={result.mutations} {result.elapsed_s:.1f}s")
                except Exception as e:
                    print(f"FAILED: {e}")

    results_path = OUTPUT_DIR / "threshold_sweep.json"
    with open(results_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nResults saved to {results_path}")

    print(f"\n{'='*90}")
    print("SUMMARY: Adaptive vs Random vs Null vs Anti (avg over seeds)")
    print(f"{'='*90}")
    print(f"{'Scale':<8} {'Null':>8} {'Random':>8} {'Adaptive':>8} {'Anti':>8}  {'A-R':>8} {'A-N':>8}")
    print("-" * 72)

    for scale_name in SCALES:
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
    for scale_name in SCALES:
        wins = 0
        total = 0
        for seed in SEEDS:
            a = [r for r in all_results if r.scale == scale_name and r.seed == seed and r.arm == "adaptive"]
            b = [r for r in all_results if r.scale == scale_name and r.seed == seed and r.arm == "random"]
            if a and b:
                total += 1
                if a[0].loss < b[0].loss:
                    wins += 1
        print(f"  {scale_name}: adaptive wins {wins}/{total} pairs")


if __name__ == "__main__":
    main()
