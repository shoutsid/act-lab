"""Section 5.30 study: the stratified re-test of cloning on the discovery path.

Pre-registered in ``RESEARCH_REPORT.md`` section 5.30. Section 5.29's registered
verdicts (P1 fail, P2 pass) are artifacts of its block: the probe-guided adaptive
arm grows in only **18%** of events, so **53% of seeds contain no grow at all** and
cloning is then a literal no-op. A full-block sign test scores such a seed as a
non-win for P1, and it contributes the full historical penalty to P2. Stratifying
on whether a grow occurred reverses both readings, but that was post-hoc.

This study re-tests it on **fresh blocks with the stratification declared in
advance**. Design is otherwise identical to section 5.29 - ``arms="fixed,adaptive"``,
drawn operations, ``preserve_router_on_rebuild=True``, the same probe configuration:

- ``F``      the fixed arm (unmutated) - identical under both settings;
- ``R``      adaptive discovery, new experts freshly drawn (historical);
- ``R_own``  ``R``'s final genome retrained from its own initialization;
- ``C``      adaptive discovery, new experts cloned from their layer's busiest
  survivor;
- ``C_own``  ``C``'s final genome retrained from its own initialization.

**The stratum is a pre-treatment covariate, deliberately.** A seed enters the grow
stratum when the **historical (`random` initialization) run** drew at least one
grow - not when each run did. The clone setting changes the weights, not the
candidate enumeration or the probe's scoring, so it cannot decide which seeds enter
the stratum. The cost of using the historical run's mix is dilution on the few
seeds where the two mixes differ; the benefit is that the stratum cannot be
outcome-dependent.

**Decision rule.** The primary is the exact one-sided binomial p on the grow
stratum against ``alpha/4 = 0.0125``. The min-wins bar is reported for the realized
stratum size (`_gate_min_wins(n, 0.0125)`: 105 at n = 179) rather than fixed in
advance, because the stratum size is itself a random draw.

Primaries (two, two depths):
1. **cloning helps where it can act** - ``C`` < ``R`` on the grow stratum.
2. **no residual penalty in the grow stratum** - ``C_own`` < ``C`` there, i.e. the
   predicted FAILURE of P2.

Usage:

    PYTHONPATH=src python scripts/adaptive_clone_stratified_study.py run --shard 0 --shards 20 --profile strat2
    PYTHONPATH=src python scripts/adaptive_clone_stratified_study.py verdict --profile strat2
    PYTHONPATH=src python scripts/adaptive_clone_stratified_study.py summary --profile strat2
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

SHARD_DIR = ROOT / "artifacts" / "toy-adaptive-clone-stratified-study"
PROFILES: dict[str, dict] = {
    "strat2": {
        "seeds": tuple(range(111000, 111384)),
        "overrides": {"depth": 2},
        "report_section": "5.30",
        "summary": "artifacts/study-summaries/adaptive-clone-stratified-depth2-384.json",
    },
    "strat3": {
        "seeds": tuple(range(112000, 112384)),
        "overrides": {"depth": 3},
        "report_section": "5.30",
        "summary": "artifacts/study-summaries/adaptive-clone-stratified-depth3-384.json",
    },
    "smoke": {
        "seeds": tuple(range(73000, 73004)),
        "overrides": {"depth": 2},
        "declared_n": 4,
        "report_section": "5.30 (smoke)",
        "summary": "artifacts/study-summaries/adaptive-clone-stratified-smoke.json",
    },
}
REGISTERED = dict(
    hidden_size=24,
    vocab_size=32,
    depth=4,
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
    probe_candidate_policy="spread",
    independent_arm_rng=True,
    device="cpu",
    forced_operation=None,
    preserve_router_on_rebuild=True,
)
ALPHA = 0.05
DECLARED_N = 384
PER_PRIMARY_ALPHA = ALPHA / 4
EXPECTED_STRATUM_N = 179
EXPECTED_BAR = 105


def _run_seed(seed: int, profile: str) -> dict:
    values = {**REGISTERED, **PROFILES[profile]["overrides"]}
    outcome: dict[str, dict] = {}
    for label, initialization in (("R", "random"), ("C", "clone")):
        config = ExperimentConfig(
            **{
                **values,
                "seed": seed,
                "arms": "fixed,adaptive",
                "new_expert_initialization": initialization,
            }
        )
        arms = {arm.mode: arm for arm in run_experiment(config).arms}
        discovered = arms["adaptive"]
        outcome[label] = {
            "fixed_loss": arms["fixed"].final_loss,
            "fused_loss": discovered.final_loss,
            "fused_genome": discovered.genome,
            "architecture_hash": genome_hash(
                ArchitectureGenome.from_dict(discovered.genome)
            ),
            "fused_mutations": discovered.mutation_count,
            "own_transport_loss": transport_genome(
                ArchitectureGenome.from_dict(discovered.genome),
                config,
                initialization="rewound",
            ).final_loss,
            "operations": [event["operation"] for event in discovered.mutation_history],
            "experts_per_layer": [
                len(layer["experts"]) for layer in discovered.genome["layers"]
            ],
        }
    historical_ops = outcome["R"]["operations"]
    grows = sum(1 for op in historical_ops if op == "grow")
    return {
        "seed": seed,
        "losses": {
            "F": outcome["R"]["fixed_loss"],
            "R": outcome["R"]["fused_loss"],
            "R_own": outcome["R"]["own_transport_loss"],
            "C": outcome["C"]["fused_loss"],
            "C_own": outcome["C"]["own_transport_loss"],
        },
        # The pre-registered stratification variable: the HISTORICAL run's drawn mix.
        "stratum": "grow" if grows >= 1 else "no_grow",
        "historical_grows": grows,
        "historical_prunes": sum(1 for op in historical_ops if op == "prune"),
        "historical_operations": historical_ops,
        "clone_operations": outcome["C"]["operations"],
        "clone_grows": sum(1 for op in outcome["C"]["operations"] if op == "grow"),
        "fixed_arm_identical_across_settings": outcome["R"]["fixed_loss"]
        == outcome["C"]["fixed_loss"],
        "same_final_genome": outcome["R"]["architecture_hash"]
        == outcome["C"]["architecture_hash"],
        "same_operation_mix": historical_ops == outcome["C"]["operations"],
        "mutations": {
            "R": outcome["R"]["fused_mutations"],
            "C": outcome["C"]["fused_mutations"],
        },
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
    wins = sum(delta < 0 for delta in deltas)
    n = len(deltas)
    p = _binomial_survival(wins, n) if n else 1.0
    bar = _gate_min_wins(n, PER_PRIMARY_ALPHA) if n else 0
    return {
        "wins": wins,
        "n": n,
        "mean_delta": sum(deltas) / n if n else 0.0,
        "binomial_p": p,
        "min_wins": bar,
        "passes": p <= PER_PRIMARY_ALPHA,
        "bar_and_p_agree": (wins >= bar) == (p <= PER_PRIMARY_ALPHA),
    }


def _built_in_checks(experiments: list[dict], profile: str) -> dict:
    spec = PROFILES[profile]
    grow_stratum = [exp for exp in experiments if exp["stratum"] == "grow"]
    no_grow_stratum = [exp for exp in experiments if exp["stratum"] == "no_grow"]
    return {
        "block_matches_declared_n": len(spec["seeds"])
        == spec.get("declared_n", DECLARED_N),
        "every_seed_is_stratified": len(grow_stratum) + len(no_grow_stratum)
        == len(experiments),
        "the_stratum_is_the_historical_runs_mix": all(
            (exp["historical_grows"] >= 1) == (exp["stratum"] == "grow")
            for exp in experiments
        ),
        "the_fixed_arm_is_identical_across_both_settings": all(
            exp["fixed_arm_identical_across_settings"] for exp in experiments
        ),
        "both_discovery_runs_used_the_full_budget": all(
            exp["mutations"]["R"] == 3 and exp["mutations"]["C"] == 3
            for exp in experiments
        ),
        # Positive control: where the historical run drew NO grow, the clone setting
        # must produce an identical run. If it ever differs, the stratum boundary is
        # not the no-op boundary and the stratification is unsound.
        "the_no_grow_stratum_is_a_true_no_op": all(
            exp["losses"]["C"] == exp["losses"]["R"] for exp in no_grow_stratum
        ),
        "the_block_is_prune_dominated_like_section_5_29": sum(
            exp["historical_grows"] for exp in experiments
        )
        < 3 * len(experiments) // 2,
        "no_layer_went_below_two_experts": all(
            min(exp["experts_per_layer"]) >= 2 for exp in experiments
        ),
    }


def summarize(profile: str) -> dict:
    spec = PROFILES[profile]
    experiments = load_shards(_shards_for(profile), profile)
    n = len(experiments)
    checks = _built_in_checks(experiments, profile)
    failed = {name: value for name, value in checks.items() if not value}
    if failed:
        raise SystemExit(f"built-in checks failed: {failed}")

    grow_stratum = [exp for exp in experiments if exp["stratum"] == "grow"]
    no_grow_stratum = [exp for exp in experiments if exp["stratum"] == "no_grow"]

    def delta_for(subset: list[dict], first: str, second: str) -> list[float]:
        return [exp["losses"][first] - exp["losses"][second] for exp in subset]

    comparisons = {
        # Registered primaries, on the grow stratum only.
        "C_vs_R_grow_stratum": _comparison(delta_for(grow_stratum, "C", "R")),
        "C_own_vs_C_grow_stratum": _comparison(delta_for(grow_stratum, "C_own", "C")),
        # The rest of the block, reported but not verdict-bearing.
        "C_vs_R_full_block": _comparison(delta_for(experiments, "C", "R")),
        "C_own_vs_C_no_grow_stratum": _comparison(
            delta_for(no_grow_stratum, "C_own", "C")
        ),
        "R_own_vs_R_no_grow_stratum": _comparison(
            delta_for(no_grow_stratum, "R_own", "R")
        ),
        "C_vs_F_full_block": _comparison(delta_for(experiments, "C", "F")),
    }
    p1 = comparisons["C_vs_R_grow_stratum"]["passes"]
    p2 = comparisons["C_own_vs_C_grow_stratum"]["passes"]
    if p1 and not p2:
        outcome = (
            "branch (a) the stratification replicates: cloning helps discovery "
            "wherever a grow occurs and no penalty remains there, so section 5.29's "
            "branch (c) is a statistic artefact rather than a mechanism result"
        )
    elif p1 and p2:
        outcome = (
            "branch (b) partial: cloning helps discovery where it binds but a "
            "residual penalty survives in the grow stratum, so the discovery path "
            "keeps a real mid-flight cost the random arm does not have"
        )
    elif p2:
        outcome = (
            "branch (c) the stratified reading does not replicate: section 5.29's "
            "branch (c) stands on its merits"
        )
    else:
        outcome = (
            "branch (d) neither passes: inconclusive, recorded without a reading"
        )
    payload = {
        "record_type": "adaptive_clone_stratified_study",
        "protocol": "pre-registered-grow-stratum-on-probe-guided-discovery",
        "profile": profile,
        "report_section": spec["report_section"],
        "alpha": ALPHA,
        "per_primary_alpha": PER_PRIMARY_ALPHA,
        "expected_stratum_n": EXPECTED_STRATUM_N,
        "expected_bar": EXPECTED_BAR,
        "depth": spec["overrides"]["depth"],
        "seeds": list(spec["seeds"]),
        "config": {**REGISTERED, **spec["overrides"]},
        "per_seed": experiments,
        "stratum_sizes": {
            "grow": len(grow_stratum),
            "no_grow": len(no_grow_stratum),
        },
        "comparisons": comparisons,
        "loss_means": {
            label: sum(exp["losses"][label] for exp in experiments) / n
            for label in ("F", "R", "R_own", "C", "C_own")
        },
        "clone_gain_grow_stratum": comparisons["C_vs_R_grow_stratum"]["mean_delta"],
        "penalty_cloned_grow_stratum": comparisons["C_own_vs_C_grow_stratum"][
            "mean_delta"
        ],
        "prune_domain_penalty": comparisons["C_own_vs_C_no_grow_stratum"][
            "mean_delta"
        ],
        "seeds_with_identical_final_genome": sum(
            1 for exp in experiments if exp["same_final_genome"]
        ),
        "historical_operation_mix": {
            "grow": sum(exp["historical_grows"] for exp in experiments),
            "prune": sum(exp["historical_prunes"] for exp in experiments),
        },
        "outcome": outcome,
        "built_in_checks": checks,
        "summary_note": (
            "Section 5.30. Pre-registered stratified re-test of section 5.29: the "
            "primary is evaluated only on seeds where the HISTORICAL (random "
            "initialization) run drew at least one grow, so the stratum cannot be "
            "outcome-dependent. The decision is the exact one-sided binomial p "
            "against alpha/4 = 0.0125 on that stratum; min_wins is reported for the "
            "realized stratum size rather than fixed in advance, because the stratum "
            "size is itself a random draw. The no-grow stratum is a positive control "
            "(the clone setting must be a bit-identical no-op there) and is reported "
            "as the prune-domain penalty, not as a verdict. Negative means the "
            "first-named run won."
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
    print(
        f"strata: grow {payload['stratum_sizes']['grow']}/{n}, "
        f"no-grow {payload['stratum_sizes']['no_grow']}/{n}   "
        f"historical mix: {payload['historical_operation_mix']}   "
        f"identical genomes {payload['seeds_with_identical_final_genome']}/{n}"
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
    run_p.add_argument("--profile", default="strat2", choices=sorted(PROFILES))
    verdict_p = sub.add_parser("verdict", help="aggregate shards")
    verdict_p.add_argument("--profile", default="strat2", choices=sorted(PROFILES))
    sum_p = sub.add_parser("summary", help="write the committed study summary")
    sum_p.add_argument("--profile", default="strat2", choices=sorted(PROFILES))
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
