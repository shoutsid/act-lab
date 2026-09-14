"""Section 5.9 study: does the toy adaptive edge clear the do-nothing bar?

Runs the registered study-3 config as a THREE-arm block per seed
(``arms="random,adaptive,null"``, ``independent_arm_rng=True``, ``cpu``) on
the fresh block 70000-70383, sharded for parallel execution. Because arm
order keeps indices 0/1 on random/adaptive and the null arm consumes no
global torch RNG, the adaptive-vs-random comparison is by construction
identical to the two-arm block (pinned by test).

Usage (one shard per worker, then the verdict):

    PYTHONPATH=src python scripts/null_toy_study.py run --shard 0 --shards 20
    ...
    PYTHONPATH=src python scripts/null_toy_study.py verdict --shards 20

The verdict computes both pre-registered primaries from the same fresh run:

1. null-vs-random  (churn harm at toy scale)      - one-tailed sign test
2. adaptive-vs-null (the new bar for Gate 3)      - one-tailed sign test

each against the Bonferroni alpha/2 = 0.0025 bar, using the exact binomial
survival from ``act.evolution.study``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from act.evolution.experiment import ExperimentConfig, run_experiment  # noqa: E402
from act.evolution.study import _binomial_survival, _gate_min_wins  # noqa: E402

SHARD_DIR = ROOT / "artifacts" / "toy-null-study"
SEEDS = tuple(range(70000, 70384))
REGISTERED = dict(
    hidden_size=24,
    vocab_size=32,
    depth=1,
    expert_count=4,
    active_experts=2,
    batch_size=2,
    sequence_length=12,
    steps=48,
    mutation_interval=12,
    mutation_budget=3,
    learning_rate=0.0003,
    validation_batches=4,
    adaptive_policy="probe_loss",
    adaptive_expert_metric="random",
    probe_candidate_cap=4,
    probe_batch_count=2,
    independent_arm_rng=True,
    device="cpu",
)
ALPHA = 0.05


def run_shard(shard: int, shards: int) -> Path:
    if shards < 1 or not 0 <= shard < shards:
        raise SystemExit("--shard must be in [0, --shards)")
    path = SHARD_DIR / f"shard{shard:02d}.json"
    if path.exists():
        print(f"shard {shard}: already complete ({path.name})")
        return path
    experiments = []
    for seed in SEEDS[shard::shards]:
        result = run_experiment(
            replace(
                ExperimentConfig(arms="random,adaptive,null", **REGISTERED),
                seed=seed,
            )
        )
        experiments.append(
            {
                "seed": seed,
                "losses": {arm.mode: arm.final_loss for arm in result.arms},
                "mutations": {arm.mode: arm.mutation_count for arm in result.arms},
                "search_flops": {arm.mode: arm.search_flops for arm in result.arms},
            }
        )
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"shard": shard, "shards": shards, "experiments": experiments}
    path.write_text(json.dumps(payload) + "\n")
    print(f"shard {shard}: wrote {len(experiments)} seeds -> {path.name}")
    return path


def load_shards(shards: int) -> list[dict]:
    experiments: list[dict] = []
    for shard in range(shards):
        path = SHARD_DIR / f"shard{shard:02d}.json"
        if not path.exists():
            raise SystemExit(f"missing shard: {path}")
        experiments.extend(json.loads(path.read_text())["experiments"])
    seen = {exp["seed"] for exp in experiments}
    if seen != set(SEEDS):
        raise SystemExit(f"shard set covers {len(seen)}/384 seeds; verdict withheld")
    return sorted(experiments, key=lambda exp: exp["seed"])


def verdict(shards: int) -> int:
    experiments = load_shards(shards)
    built_in_ok = all(
        exp["mutations"] == {"random": 3, "adaptive": 3, "null": 0}
        and exp["search_flops"]["null"] == 0
        for exp in experiments
    )
    if not built_in_ok:
        raise SystemExit("built-in check failed: unexpected mutation counts or null search FLOPs")
    print(f"built-in check: mutations 3/3/0 and zero null search FLOPs in {len(experiments)}/384 seeds: OK")

    null_deltas = [exp["losses"]["null"] - exp["losses"]["random"] for exp in experiments]
    bar_deltas = [exp["losses"]["adaptive"] - exp["losses"]["null"] for exp in experiments]
    rand_deltas = [exp["losses"]["adaptive"] - exp["losses"]["random"] for exp in experiments]

    def sign_test(deltas: list[float]) -> tuple[int, float, float]:
        wins = sum(delta < 0 for delta in deltas)
        p = _binomial_survival(wins, len(deltas))
        return wins, sum(deltas) / len(deltas), p

    n_wins, n_mean, n_p = sign_test(null_deltas)
    b_wins, b_mean, b_p = sign_test(bar_deltas)
    a_wins, a_mean, a_p = sign_test(rand_deltas)
    n_min = _gate_min_wins(len(null_deltas), ALPHA / 2)
    b_min = _gate_min_wins(len(bar_deltas), ALPHA / 2)

    print(f"\nprimary 1  null-vs-random:   wins {n_wins}/384, mean {n_mean:+.6f}, p {n_p:.2e} (bar {n_min})")
    print(f"primary 2  adaptive-vs-null: wins {b_wins}/384, mean {b_mean:+.6f}, p {b_p:.2e} (bar {b_min})")
    print(f"secondary  adaptive-vs-random: wins {a_wins}/384, mean {a_mean:+.6f}, p {a_p:.2e}")
    print(f"per-arm mean losses: random {sum(e['losses']['random'] for e in experiments)/384:.6f}, "
          f"adaptive {sum(e['losses']['adaptive'] for e in experiments)/384:.6f}, "
          f"null {sum(e['losses']['null'] for e in experiments)/384:.6f}")

    p1 = n_wins >= n_min
    p2 = b_wins >= b_min
    if p1 and p2:
        outcome = "1+2 PASS: the toy edge is real beyond the pool effect AND churn suppression"
    elif p1:
        outcome = "1 PASS / 2 FAIL: toy Gate 3 passes are substantially churn suppression"
    elif p2:
        outcome = "1 FAIL / 2 PASS: no toy churn harm; random-churn comparisons remain meaningful"
    else:
        outcome = "both FAIL: the adaptive arm does not clear the do-nothing bar"
    print(f"\nOUTCOME (alpha={ALPHA}, Bonferroni alpha/2={ALPHA / 2}): {outcome}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run one shard")
    run_p.add_argument("--shard", type=int, required=True)
    run_p.add_argument("--shards", type=int, required=True)
    verdict_p = sub.add_parser("verdict", help="aggregate shards and score the contract")
    verdict_p.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run_shard(args.shard, args.shards)
        return 0
    return verdict(args.shards)


if __name__ == "__main__":
    sys.exit(main())
