"""Section 5.32 study: the prune-target question, re-registered with power.

Pre-registered in ``RESEARCH_REPORT.md`` section 5.32. Section 5.31 asked whether
*which* expert a prune removes changes the prune cost, registered a paired **sign
test**, and failed both primaries - while the means ordered exactly as predicted
(`Q < R < B`) at both depths, with win rates of 53-54%. At that win rate the sign
test needs ~1396 seeds to clear the Bonferroni bar; the block had 384. So section
5.31's branch (d) is a power failure rather than evidence of absence, and the
question is re-opened on a properly powered block with a mean-based statistic
declared in advance.

Design is section 5.31's exactly - three policies, `forced_operation="prune"`,
`preserve_router_on_rebuild=True`, one initial model and data schedule per seed -
with two declared changes:

1. **A mean-based statistic.** The primary is a one-sided paired **sign-flip
   permutation test on the mean of the per-seed deltas**, 20,000 permutations
   seeded from the study hash, against ``alpha/4 = 0.0125``. Permutation-based
   rather than a t-test because the deltas are heavy-tailed.
2. **A block ~4x larger** (1536 seeds per depth), sized from section 5.31's
   observed effect and spread rather than by tradition.

Every comparison stores the mean, its standard error, the permutation p and the
sign-test p together, so a failure can be read as "no effect" or "no resolution"
rather than guessed at.

Labels: ``R`` random target (historical), ``Q`` quietest, ``B`` busiest, ``F`` the
unmutated fixed arm, ``*_own`` each fused run's own final genome retrained.

Usage:

    PYTHONPATH=src python scripts/prune_target_powered_study.py run --shard 0 --shards 20 --profile pow2
    PYTHONPATH=src python scripts/prune_target_powered_study.py verdict --profile pow2
    PYTHONPATH=src python scripts/prune_target_powered_study.py summary --profile pow2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from act.evolution.experiment import ExperimentConfig, run_experiment  # noqa: E402
from act.evolution.registry import _stable_hash  # noqa: E402
from act.evolution.reuse import transport_genome  # noqa: E402
from act.evolution.study import _binomial_survival, _gate_min_wins  # noqa: E402
from act.topology import ArchitectureGenome, genome_hash  # noqa: E402

SHARD_DIR = ROOT / "artifacts" / "toy-prune-target-powered-study"
PROFILES: dict[str, dict] = {
    "pow2": {
        "seeds": tuple(range(115000, 116536)),
        "overrides": {"depth": 2},
        "shards": 20,
        "report_section": "5.32",
        "summary": "artifacts/study-summaries/prune-target-powered-depth2-1536.json",
    },
    "pow3": {
        "seeds": tuple(range(117000, 118536)),
        "overrides": {"depth": 3},
        # Depth 3 costs ~2x depth 2 per seed, and a 20-shard split puts 77 seeds in
        # one process (~15 min) - too coarse to run in one sitting. More shards,
        # same seed block.
        "shards": 120,
        "report_section": "5.32",
        "summary": "artifacts/study-summaries/prune-target-powered-depth3-1536.json",
    },
}
REGISTERED = dict(
    hidden_size=24,
    vocab_size=32,
    depth=4,
    expert_count=5,
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
    probe_candidate_policy="spread",
    independent_arm_rng=True,
    device="cpu",
    forced_operation="prune",
    preserve_router_on_rebuild=True,
)
POLICIES = (("R", "random"), ("Q", "quietest"), ("B", "busiest"))
ALPHA = 0.05
DECLARED_N = 1536
PER_PRIMARY_ALPHA = ALPHA / 4
N_PERMUTATIONS = 20000
_PERMUTATION_CHUNK = 2000


def _permutation_p(deltas: list[float], seed: int) -> float:
    """One-sided sign-flip permutation p for the mean being negative.

    Under the null the paired deltas are symmetric about zero, so flipping each
    delta's sign at random gives the null distribution of the mean. Vectorised in
    torch and chunked to bound memory; seeded so the p-value is reproducible.
    """
    values = torch.tensor(deltas, dtype=torch.float64)
    observed = float(values.mean())
    generator = torch.Generator().manual_seed(seed)
    at_least_as_extreme = 0
    for start in range(0, N_PERMUTATIONS, _PERMUTATION_CHUNK):
        size = min(_PERMUTATION_CHUNK, N_PERMUTATIONS - start)
        signs = (
            torch.randint(
                0, 2, (size, values.numel()), generator=generator, dtype=torch.float64
            )
            * 2
            - 1
        )
        means = (signs @ values) / values.numel()
        at_least_as_extreme += int((means <= observed).sum())
    return (at_least_as_extreme + 1) / (N_PERMUTATIONS + 1)


def _comparison(deltas: list[float]) -> dict:
    """Both statistics side by side, so a failure is readable.

    ``passes`` is the REGISTERED decision (the permutation p) and the sign test is
    reported next to it rather than instead of it - the whole point of section
    5.32 is that these two can disagree, and section 5.31 shows what it costs to
    store only one.
    """
    n = len(deltas)
    mean = sum(deltas) / n
    wins = sum(delta < 0 for delta in deltas)
    variance = (
        sum((delta - mean) ** 2 for delta in deltas) / (n - 1) if n > 1 else 0.0
    )
    standard_error = math.sqrt(variance / n) if n else 0.0
    permutation_p = _permutation_p(deltas, seed=0xC0FFEE + n)
    return {
        "n": n,
        "wins": wins,
        "sign_test_p": _binomial_survival(wins, n),
        "mean_delta": mean,
        "standard_error": standard_error,
        "t": mean / standard_error if standard_error else 0.0,
        "permutation_p": permutation_p,
        "passes": permutation_p <= PER_PRIMARY_ALPHA,
        "sign_test_passes": _binomial_survival(wins, n) <= PER_PRIMARY_ALPHA,
        "the_two_statistics_disagree": (
            (permutation_p <= PER_PRIMARY_ALPHA)
            != (_binomial_survival(wins, n) <= PER_PRIMARY_ALPHA)
        ),
        "min_wins_for_the_sign_test": _gate_min_wins(n, PER_PRIMARY_ALPHA),
    }


def _run_seed(seed: int, profile: str) -> dict:
    values = {**REGISTERED, **PROFILES[profile]["overrides"]}
    outcome: dict[str, dict] = {}
    for label, policy in POLICIES:
        config = ExperimentConfig(
            **{
                **values,
                "seed": seed,
                "arms": "fixed,random",
                "prune_expert_policy": policy,
            }
        )
        arms = {arm.mode: arm for arm in run_experiment(config).arms}
        fused = arms["random"]
        outcome[label] = {
            "fixed_loss": arms["fixed"].final_loss,
            "fused_loss": fused.final_loss,
            "fused_genome": fused.genome,
            "fused_mutations": fused.mutation_count,
            "own_transport_loss": transport_genome(
                ArchitectureGenome.from_dict(fused.genome),
                config,
                initialization="rewound",
            ).final_loss,
            "experts_per_layer": [
                len(layer["experts"]) for layer in fused.genome["layers"]
            ],
        }
    losses = {"F": outcome["R"]["fixed_loss"]}
    for label, _ in POLICIES:
        losses[label] = outcome[label]["fused_loss"]
        losses[f"{label}_own"] = outcome[label]["own_transport_loss"]
    return {
        "seed": seed,
        "losses": losses,
        "penalties": {
            label: losses[f"{label}_own"] - losses[label] for label, _ in POLICIES
        },
        "fixed_arm_identical_across_policies": len(
            {outcome[label]["fixed_loss"] for label, _ in POLICIES}
        )
        == 1,
        "final_expert_counts_agree": len(
            {
                tuple(outcome[label]["experts_per_layer"])
                for label, _ in POLICIES
            }
        )
        == 1,
        "distinct_policy_genomes": len(
            {
                genome_hash(ArchitectureGenome.from_dict(outcome[label]["fused_genome"]))
                for label, _ in POLICIES
            }
        ),
        "mutations": {label: outcome[label]["fused_mutations"] for label, _ in POLICIES},
        "experts_per_layer": outcome["R"]["experts_per_layer"],
    }


def run_shard(shard: int, shards: int, profile: str) -> Path:
    if shards < 1 or not 0 <= shard < shards:
        raise SystemExit("--shard must be in [0, --shards)")
    if profile not in PROFILES:
        raise SystemExit(f"--profile must be one of {sorted(PROFILES)}")
    directory = SHARD_DIR / profile
    path = directory / f"shard{shard:02d}.json"
    if path.exists():
        print(f"shard {shard}: already complete ({path.name})")
        return path
    experiments = [
        _run_seed(seed, profile) for seed in PROFILES[profile]["seeds"][shard::shards]
    ]
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "shard": shard,
                "shards": shards,
                "profile": profile,
                "experiments": experiments,
            }
        )
        + "\n"
    )
    print(f"shard {shard}: wrote {len(experiments)} seeds -> {path.name}")
    return path


def _shards_for(profile: str) -> int:
    return PROFILES[profile]["shards"]


def load_shards(shards: int, profile: str) -> list[dict]:
    experiments: list[dict] = []
    for shard in range(shards):
        path = SHARD_DIR / profile / f"shard{shard:02d}.json"
        if not path.exists():
            raise SystemExit(f"missing shard: {path}")
        experiments.extend(json.loads(path.read_text())["experiments"])
    seen = {exp["seed"] for exp in experiments}
    expected = set(PROFILES[profile]["seeds"])
    if seen != expected:
        raise SystemExit(
            f"shard set covers {len(seen)}/{len(expected)} seeds; verdict withheld"
        )
    return sorted(experiments, key=lambda exp: exp["seed"])


def _built_in_checks(experiments: list[dict], profile: str) -> dict:
    spec = PROFILES[profile]
    distinct_three = sum(
        1 for exp in experiments if exp["distinct_policy_genomes"] == 3
    )
    return {
        "block_matches_declared_n": len(spec["seeds"]) == DECLARED_N,
        "the_fixed_arm_is_identical_across_all_three_policies": all(
            exp["fixed_arm_identical_across_policies"] for exp in experiments
        ),
        "all_three_policies_used_the_full_budget": all(
            all(count == 3 for count in exp["mutations"].values())
            for exp in experiments
        ),
        "the_policies_end_at_the_same_expert_counts": all(
            exp["final_expert_counts_agree"] for exp in experiments
        ),
        "no_layer_went_below_two_experts": all(
            min(exp["experts_per_layer"]) >= 2 for exp in experiments
        ),
        # Same reconciliation as section 5.31.1: a chance collision between two
        # policies is legitimate, so this is a rate, not an all-seeds requirement.
        "the_policies_are_live": distinct_three > 0,
        "collisions_are_a_minority": distinct_three >= len(experiments) // 2,
    }


def summarize(profile: str) -> dict:
    spec = PROFILES[profile]
    experiments = load_shards(_shards_for(profile), profile)
    n = len(experiments)
    checks = _built_in_checks(experiments, profile)
    failed = {name: value for name, value in checks.items() if not value}
    if failed:
        raise SystemExit(f"built-in checks failed: {failed}")

    def losses(label: str) -> list[float]:
        return [exp["losses"][label] for exp in experiments]

    def penalties(label: str) -> list[float]:
        return [exp["penalties"][label] for exp in experiments]

    def delta(first: list[float], second: list[float]) -> list[float]:
        return [x - y for x, y in zip(first, second)]

    comparisons = {
        # Registered primaries, on the FUSED loss.
        "fused_Q_vs_R": _comparison(delta(losses("Q"), losses("R"))),
        "fused_R_vs_B": _comparison(delta(losses("R"), losses("B"))),
        # Mechanism secondaries on the penalty.
        "penalty_quietest_less_than_random": _comparison(
            delta(penalties("Q"), penalties("R"))
        ),
        "penalty_busiest_worse_than_random": _comparison(
            delta(penalties("B"), penalties("R"))
        ),
        # Context.
        "fused_Q_vs_F": _comparison(delta(losses("Q"), losses("F"))),
        "fused_R_vs_F": _comparison(delta(losses("R"), losses("F"))),
    }
    p1 = comparisons["fused_Q_vs_R"]["passes"]
    p2 = comparisons["fused_R_vs_B"]["passes"]
    if p1 and p2:
        outcome = (
            "branch (a) the prune target is a policy lever, monotone in load, and "
            "section 5.30's deficit is partly addressable by target selection"
        )
    elif p1 and not p2:
        outcome = (
            "branch (b) the lever is one-sided: the quietest is a good target but "
            "the busiest is not especially bad"
        )
    elif p2 and not p1:
        outcome = (
            "branch (c) the busiest is a bad target but the quietest is no better "
            "than random"
        )
    else:
        outcome = (
            "branch (d) with ~4x the power of section 5.31, the prune cost does not "
            "depend on the target - it is intrinsic to removing an expert at all"
        )
    payload = {
        "record_type": "prune_target_powered_study",
        "protocol": "random-vs-quietest-vs-busiest-prune-target-permutation-powered",
        "profile": profile,
        "report_section": spec["report_section"],
        "alpha": ALPHA,
        "per_primary_alpha": PER_PRIMARY_ALPHA,
        "declared_n": DECLARED_N,
        "n_permutations": N_PERMUTATIONS,
        "depth": spec["overrides"]["depth"],
        "seeds": list(spec["seeds"]),
        "config": {**REGISTERED, **spec["overrides"]},
        "per_seed": experiments,
        "comparisons": comparisons,
        "loss_means": {
            label: sum(losses(label)) / n
            for label in ("F", "R", "Q", "B", "R_own", "Q_own", "B_own")
        },
        "penalty_means": {label: sum(penalties(label)) / n for label, _ in POLICIES},
        "seeds_with_three_distinct_genomes": sum(
            1 for exp in experiments if exp["distinct_policy_genomes"] == 3
        ),
        "outcome": outcome,
        "built_in_checks": checks,
        "summary_note": (
            "Section 5.32. Section 5.31's block, statistic and question, with the "
            "statistic declared as a one-sided paired sign-flip permutation test on "
            "the mean and the block sized ~4x from section 5.31's observed effect "
            "and spread. Both primaries are on the FUSED loss, signed so a win is "
            "the primary's prediction: fused(Q) - fused(R) < 0 and fused(R) - "
            "fused(B) < 0. Every comparison stores the mean, its standard error, the "
            "permutation p and the sign-test p together, so a failure can be read as "
            "'no effect' or 'no resolution' rather than guessed at."
        ),
    }
    payload["study_hash"] = _stable_hash(
        {
            "config": payload["config"],
            "seeds": payload["seeds"],
            "per_seed": payload["per_seed"],
            "alpha": PER_PRIMARY_ALPHA,
        }
    )
    payload["study_id"] = f"study_{payload['study_hash']}"
    payload["config_hash"] = _stable_hash(payload["config"])
    return payload


def verdict(profile: str) -> int:
    payload = summarize(profile)
    n = len(payload["per_seed"])
    print(f"loss means: {payload['loss_means']}")
    print(f"penalty means: {payload['penalty_means']}")
    print(
        f"seeds with 3 distinct surviving sets: "
        f"{payload['seeds_with_three_distinct_genomes']}/{n}"
    )
    for name, comparison in payload["comparisons"].items():
        print(
            f"\n{name}: n {comparison['n']}, mean {comparison['mean_delta']:+.5f} "
            f"(se {comparison['standard_error']:.5f}, t {comparison['t']:+.2f})\n"
            f"    permutation p {comparison['permutation_p']:.2e} "
            f"-> {'PASS' if comparison['passes'] else 'FAIL'}\n"
            f"    sign test {comparison['wins']}/{comparison['n']} p "
            f"{comparison['sign_test_p']:.2e} "
            f"(bar {comparison['min_wins_for_the_sign_test']}) "
            f"-> {'PASS' if comparison['sign_test_passes'] else 'FAIL'}"
            f"{'  <-- THE TWO DISAGREE' if comparison['the_two_statistics_disagree'] else ''}"
        )
    print(f"\nOUTCOME: {payload['outcome']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run one shard")
    run_p.add_argument("--shard", type=int, required=True)
    run_p.add_argument("--shards", type=int, required=True)
    run_p.add_argument("--profile", default="pow2", choices=sorted(PROFILES))
    verdict_p = sub.add_parser("verdict", help="aggregate shards")
    verdict_p.add_argument("--profile", default="pow2", choices=sorted(PROFILES))
    sum_p = sub.add_parser("summary", help="write the committed study summary")
    sum_p.add_argument("--profile", default="pow2", choices=sorted(PROFILES))
    args = parser.parse_args()
    if args.command == "run":
        run_shard(args.shard, args.shards, args.profile)
        return 0
    if args.command == "summary":
        payload = summarize(args.profile)
        out = ROOT / PROFILES[args.profile]["summary"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"summary written: {out} ({payload['study_id']})")
        return 0
    return verdict(args.profile)


if __name__ == "__main__":
    raise SystemExit(main())
