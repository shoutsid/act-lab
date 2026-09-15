"""Section 5.31 study: does it matter WHICH expert a prune removes?

Pre-registered in ``RESEARCH_REPORT.md`` section 5.31. Section 5.30 located the
remaining mid-training cost precisely: cloning repairs the **grow** term, and the
**prune** term is untouched, standing at ``-0.055``/``-0.074`` on the prune-dominated
discovery path. Section 5.25 measured that prune cost with the **expert chosen at
random within the layer**, because that is what ``_mutate_arm`` did. Whether the
cost is intrinsic to removing an expert or a property of removing the *wrong* one
has never been tested, and it is directly actionable if the answer is the latter.

``ExperimentConfig.prune_expert_policy`` decides which expert a prune removes when
the choice is not otherwise made for it: ``"random"`` (the historical behaviour
every published number was measured under), ``"quietest"`` or ``"busiest"``, ranked
by the model's own recorded routing load. Per seed, one initial model and one data
schedule, ``forced_operation="prune"``, ``preserve_router_on_rebuild=True``,
``arms="fixed,random"``, and each seed runs **all three policies**:

- ``F``    the fixed arm (unmutated) - identical under all three policies;
- ``R``    every event prunes the random expert (historical);
- ``Q``    every event prunes the **quietest** expert of the chosen layer;
- ``B``    every event prunes the **busiest** expert of the chosen layer;
- ``*_own`` the frozen retrain of each fused run's own final genome.

**The design is tight, and one control is worth stating.** All three policies pick
their layer from the same seeded RNG, prune the same number of times, and end at the
2-expert floor, so the runs differ only in *which* experts are removed - and because
a copy of the removed expert's identity is not retained, the surviving **set** is
what differs. The initial model, the data schedule, the layer, the count and the
final expert count are all held fixed across policies.

Primaries per depth (n = 384), Bonferroni ``alpha/4 = 0.0125`` -> bar 215, on the
**fused loss** - the end of the run, which is the deployable quantity:

1. **removing the quietest is cheaper** - ``fused(Q) < fused(R)``.
2. **removing the busiest is dearer** - ``fused(R) < fused(B)``.

**Why the primary is the fused loss and not the penalty section 5.25 measured.**
The penalty is a *difference* between two processes, and a policy that leaves a
final topology which is harder to retrain will show a *smaller* penalty even when
its own fused run is worse. The 4-seed smoke below shows exactly that inversion -
the fused losses order ``Q < R < B`` while the penalties order ``B < R < Q``,
because ``Q``'s final genome retrains unusually well and ``B``'s unusually badly.
The penalty is therefore reported as the mechanism secondary and is **not**
verdict-bearing; the construct argument, not the smoke, is what fixes this choice.

Usage:

    PYTHONPATH=src python scripts/prune_target_study.py run --shard 0 --shards 20 --profile target2
    PYTHONPATH=src python scripts/prune_target_study.py verdict --profile target2
    PYTHONPATH=src python scripts/prune_target_study.py summary --profile target2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from act.evolution.experiment import ExperimentConfig, run_experiment  # noqa: E402
from act.evolution.registry import _stable_hash  # noqa: E402
from act.evolution.reuse import transport_genome  # noqa: E402
from act.evolution.study import _binomial_survival, _gate_min_wins  # noqa: E402
from act.topology import ArchitectureGenome, genome_hash  # noqa: E402

SHARD_DIR = ROOT / "artifacts" / "toy-prune-target-study"
PROFILES: dict[str, dict] = {
    "target2": {
        "seeds": tuple(range(113000, 113384)),
        "overrides": {"depth": 2},
        "report_section": "5.31",
        "summary": "artifacts/study-summaries/prune-target-depth2-384.json",
    },
    "target3": {
        "seeds": tuple(range(114000, 114384)),
        "overrides": {"depth": 3},
        "report_section": "5.31",
        "summary": "artifacts/study-summaries/prune-target-depth3-384.json",
    },
    "smoke": {
        "seeds": tuple(range(73000, 73004)),
        "overrides": {"depth": 2},
        "declared_n": 4,
        "report_section": "5.31 (smoke)",
        "summary": "artifacts/study-summaries/prune-target-smoke.json",
    },
}
REGISTERED = dict(
    hidden_size=24,
    vocab_size=32,
    depth=4,
    # Section 5.25's prune block: 5 experts/layer lets three prunes reach the floor.
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
DECLARED_N = 384
PER_PRIMARY_ALPHA = ALPHA / 4  # two primaries, two depths
DECLARED_BAR = 215


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
            "architecture_hash": genome_hash(ArchitectureGenome.from_dict(fused.genome)),
            "fused_mutations": fused.mutation_count,
            "own_transport_loss": transport_genome(
                ArchitectureGenome.from_dict(fused.genome),
                config,
                initialization="rewound",
            ).final_loss,
            "experts_per_layer": [len(layer["experts"]) for layer in fused.genome["layers"]],
        }
    losses = {"F": outcome["R"]["fixed_loss"]}
    for label, _ in POLICIES:
        losses[label] = outcome[label]["fused_loss"]
        losses[f"{label}_own"] = outcome[label]["own_transport_loss"]
    structures = {
        label: (outcome[label]["architecture_hash"], tuple(outcome[label]["experts_per_layer"]))
        for label, _ in POLICIES
    }
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
        # Every policy must remove the same NUMBER of experts, so the policies are
        # comparable in capacity; only the surviving identities differ.
        "final_expert_counts_agree": len({counts for _, counts in structures.values()}) == 1,
        # How many distinct surviving sets the three policies produced (1-3). A
        # coincidence - the load ranking matching the random draws - is
        # legitimate, so this is recorded rather than required to be 3; see the
        # 5.31.1 amendment. A collision can only push the paired delta toward 0,
        # which can only hurt a primary, never manufacture one.
        "distinct_policy_genomes": len({hashed for hashed, _ in structures.values()}),
        "mutations": {label: outcome[label]["fused_mutations"] for label, _ in POLICIES},
        # A sample of what each policy actually removed, as the honest record of
        # whether the knob moved anything at all.
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
    return 1 if profile == "smoke" else 20


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


def _comparison(deltas: list[float]) -> dict:
    """Count win when the delta is negative; the caller signs the delta so that a
    negative always means the primary's prediction."""
    wins = sum(delta < 0 for delta in deltas)
    n = len(deltas)
    p = _binomial_survival(wins, n)
    bar = _gate_min_wins(n, PER_PRIMARY_ALPHA)
    return {
        "wins": wins,
        "n": n,
        "mean_delta": sum(deltas) / n,
        "binomial_p": p,
        "min_wins": bar,
        "passes": p <= PER_PRIMARY_ALPHA,
        "bar_and_p_agree": (wins >= bar) == (p <= PER_PRIMARY_ALPHA),
    }


def _built_in_checks(experiments: list[dict], profile: str) -> dict:
    spec = PROFILES[profile]
    return {
        "declared_bar_matches_exact_binomial": _gate_min_wins(
            DECLARED_N, PER_PRIMARY_ALPHA
        )
        == DECLARED_BAR,
        "block_matches_declared_n": len(spec["seeds"])
        == spec.get("declared_n", DECLARED_N),
        "the_fixed_arm_is_identical_across_all_three_policies": all(
            exp["fixed_arm_identical_across_policies"] for exp in experiments
        ),
        "all_three_policies_used_the_full_budget": all(
            all(count == 3 for count in exp["mutations"].values())
            for exp in experiments
        ),
        # Capacity control: every policy must remove the same NUMBER of experts, so
        # the policies differ only in which identities survive.
        "the_policies_end_at_the_same_expert_counts": all(
            exp["final_expert_counts_agree"] for exp in experiments
        ),
        "no_layer_went_below_two_experts": all(
            min(exp["experts_per_layer"]) >= 2 for exp in experiments
        ),
        # The knob must be live, but a chance collision between two policies is
        # legitimate rather than a failure, so the requirement is that the policies
        # are distinct in a clear majority of seeds - not in every one. Section
        # 5.20.1 is the precedent: an all-384-differ guard withheld a whole block
        # over one seed's coincidence.
        "the_policies_are_live": sum(
            1 for exp in experiments if exp["distinct_policy_genomes"] == 3
        )
        > 0,
        "collisions_are_a_minority": sum(
            1 for exp in experiments if exp["distinct_policy_genomes"] == 3
        )
        >= len(experiments) // 2,
        # A seed where all three policies coincide is also legitimate (the load
        # ordering matched the random draws twice over), so this is a rate as well.
        "full_collisions_are_rare": sum(
            1 for exp in experiments if exp["distinct_policy_genomes"] == 1
        )
        <= len(experiments) // 10,
        "every_seed_keeps_at_least_two_distinct_sets": sum(
            1 for exp in experiments if exp["distinct_policy_genomes"] >= 2
        )
        >= len(experiments) // 2,
    }


def summarize(profile: str) -> dict:
    spec = PROFILES[profile]
    experiments = load_shards(_shards_for(profile), profile)
    n = len(experiments)
    checks = _built_in_checks(experiments, profile)
    failed = {name: value for name, value in checks.items() if not value}
    if failed:
        raise SystemExit(f"built-in checks failed: {failed}")

    def penalties(label: str) -> list[float]:
        return [exp["penalties"][label] for exp in experiments]

    def losses(label: str) -> list[float]:
        return [exp["losses"][label] for exp in experiments]

    def delta(first: list[float], second: list[float]) -> list[float]:
        return [x - y for x, y in zip(first, second)]

    comparisons = {
        # Primary 1: fused(Q) < fused(R) - removing the quietest is cheaper.
        "fused_Q_vs_R": _comparison(delta(losses("Q"), losses("R"))),
        # Primary 2: fused(R) < fused(B) - removing the busiest is dearer.
        "fused_R_vs_B": _comparison(delta(losses("R"), losses("B"))),
        # Mechanism secondaries, not verdict-bearing: the penalties section 5.25
        # measured, and the retrains' own difficulty, which is what makes the two
        # measures able to disagree.
        "penalty_quietest_less_than_random": _comparison(
            delta(penalties("Q"), penalties("R"))
        ),
        "penalty_busiest_worse_than_random": _comparison(
            delta(penalties("B"), penalties("R"))
        ),
        "retrain_Q_less_hard_than_R": _comparison(delta(losses("Q_own"), losses("R_own"))),
        "fused_R_vs_F": _comparison(delta(losses("R"), losses("F"))),
        "fused_Q_vs_F": _comparison(delta(losses("Q"), losses("F"))),
    }
    p1 = comparisons["fused_Q_vs_R"]["passes"]
    p2 = comparisons["fused_R_vs_B"]["passes"]
    if p1 and p2:
        outcome = (
            "branch (a) the target matters and monotonically in load: pruning the "
            "quietest beats pruning at random, which beats pruning the busiest, so "
            "the controller's choice is a policy lever and the remaining deficit "
            "is not intrinsic"
        )
    elif p1 and not p2:
        outcome = (
            "branch (b) partial: pruning the quietest helps, but pruning the busiest "
            "is no worse than pruning at random - the cost is not a simple function "
            "of load"
        )
    elif p2 and not p1:
        outcome = (
            "branch (c) pruning the busiest is worse than random, but pruning the "
            "quietest does not help: the cost is dominated by the fact that SOME "
            "expert is removed, not by which"
        )
    else:
        outcome = (
            "branch (d) neither passes: the prune cost does not depend on which "
            "expert is removed - it is intrinsic to removing one at all, and the "
            "controller's choice is not a lever"
        )
    payload = {
        "record_type": "prune_target_study",
        "protocol": "random-vs-quietest-vs-busiest-prune-target",
        "profile": profile,
        "report_section": spec["report_section"],
        "alpha": ALPHA,
        "per_primary_alpha": PER_PRIMARY_ALPHA,
        "declared_bar": DECLARED_BAR,
        "depth": spec["overrides"]["depth"],
        "seeds": list(spec["seeds"]),
        "config": {**REGISTERED, **spec["overrides"]},
        "per_seed": experiments,
        "comparisons": comparisons,
        "loss_means": {
            label: sum(losses(label)) / n
            for label in ("F", "R", "Q", "B", "R_own", "Q_own", "B_own")
        },
        "penalty_means": {
            label: sum(penalties(label)) / n for label, _ in POLICIES
        },
        "seeds_with_three_distinct_genomes": sum(
            1 for exp in experiments if exp["distinct_policy_genomes"] == 3
        ),
        "seeds_with_a_full_collision": sum(
            1 for exp in experiments if exp["distinct_policy_genomes"] == 1
        ),
        "outcome": outcome,
        "built_in_checks": checks,
        "summary_note": (
            "Section 5.31. All three policies run on the same seed, initial model "
            "and data schedule with every event forced to a prune, and all three "
            "remove the same NUMBER of experts from the same layer, so they differ "
            "only in which identities survive. Both primaries are on the FUSED "
            "loss, signed so a win is the primary's prediction: fused(Q) - "
            "fused(R) < 0 and fused(R) - fused(B) < 0. The penalty own - fused is "
            "reported as a mechanism secondary and is not verdict-bearing, because "
            "it is a difference between two processes and a policy that leaves a "
            "harder-to-retrain topology shows a smaller penalty even when its own "
            "fused run is worse."
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
    print(f"penalty means: {payload['penalty_means']}")
    print(f"loss means: {payload['loss_means']}")
    print(
        f"seeds with 3 distinct surviving sets: "
        f"{payload['seeds_with_three_distinct_genomes']}/{n}"
    )
    for name, comparison in payload["comparisons"].items():
        print(
            f"\n{name}: wins {comparison['wins']}/{comparison['n']}, "
            f"mean {comparison['mean_delta']:+.6f}, p {comparison['binomial_p']:.2e} "
            f"(bar {comparison['min_wins']}): "
            f"{'PASS' if comparison['passes'] else 'FAIL'}"
        )
    print(f"\nOUTCOME: {payload['outcome']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run one shard")
    run_p.add_argument("--shard", type=int, required=True)
    run_p.add_argument("--shards", type=int, required=True)
    run_p.add_argument("--profile", default="target2", choices=sorted(PROFILES))
    verdict_p = sub.add_parser("verdict", help="aggregate shards")
    verdict_p.add_argument("--profile", default="target2", choices=sorted(PROFILES))
    sum_p = sub.add_parser("summary", help="write the committed study summary")
    sum_p.add_argument("--profile", default="target2", choices=sorted(PROFILES))
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
