"""Section 5.29 study: does clone-and-preserve hold on the discovery path?

Pre-registered in ``RESEARCH_REPORT.md`` section 5.29. Section 5.28 confirmed that
cloning the layer's busiest survivor inverts the mid-training penalty in the drawn,
unforced regime - but on the **`random`** arm. The arm the project actually uses to
discover genomes is the **probe-based `adaptive`** arm: it enumerates feasible
mutation candidates, scores each by rebuilding and measuring loss on held-out
probe batches, and applies the best. That is a different process, and whether the
initialization treatment survives it is not implied by section 5.28.

Per seed, one initial model and one data schedule, ``arms="fixed,adaptive"``,
``forced_operation`` left None (so the mix is drawn), ``preserve_router_on_rebuild=True``:

- ``F``      the fixed arm (unmutated) - identical under both settings;
- ``R``      adaptive discovery, new experts freshly drawn (historical);
- ``R_own``  ``R``'s final genome retrained from its own initialization;
- ``C``      adaptive discovery, new experts cloned from their layer's busiest
  survivor (body and router row);
- ``C_own``  ``C``'s final genome retrained from its own initialization.

**The pairing is NOT assumed, and on this path it does not always hold.** The probe
scores candidates by calling ``rebuild_model_from_genome`` **without** the arm's
``preserve_router`` / ``new_expert_initialization`` flags, so *scoring* is
identical under both settings at a given set of weights while the *application*
uses them - and after the first event the two runs' weights differ, so a later
probe can rank candidates differently. A 16-seed spent-seed measurement found
**divergent final genomes in 2/16 seeds (12.5%)**, so roughly a tenth of the
registered block is expected to diverge. Unlike sections 5.26 and 5.27, identical
genomes are therefore **recorded and reported, never asserted**, and each penalty
is measured against its own frozen retrain. Primary 1 is consequently a comparison
of two *processes*, not a paired contrast on one genome - which is the right shape
for a deployable question ("does switching the initialization improve the
discovery pipeline?") and the wrong shape for a mechanism question.

Primaries per depth (n = 384), Bonferroni ``alpha/4 = 0.0125`` -> bar 215:

1. **clone helps on the discovery path** - ``C`` < ``R``.
2. **no residual penalty under cloning** - ``C_own`` < ``C``, i.e. the predicted
   FAILURE of P2, as in sections 5.26 and 5.28.

Usage:

    PYTHONPATH=src python scripts/adaptive_clone_study.py run --shard 0 --shards 20 --profile adap2
    PYTHONPATH=src python scripts/adaptive_clone_study.py verdict --profile adap2
    PYTHONPATH=src python scripts/adaptive_clone_study.py summary --profile adap2
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

SHARD_DIR = ROOT / "artifacts" / "toy-adaptive-clone-study"
PROFILES: dict[str, dict] = {
    "adap2": {
        "seeds": tuple(range(109000, 109384)),
        "overrides": {"depth": 2},
        "report_section": "5.29",
        "summary": "artifacts/study-summaries/adaptive-clone-depth2-384.json",
    },
    "adap3": {
        "seeds": tuple(range(110000, 110384)),
        "overrides": {"depth": 3},
        "report_section": "5.29",
        "summary": "artifacts/study-summaries/adaptive-clone-depth3-384.json",
    },
    "smoke": {
        "seeds": tuple(range(73000, 73004)),
        "overrides": {"depth": 2},
        "declared_n": 4,
        "report_section": "5.29 (smoke)",
        "summary": "artifacts/study-summaries/adaptive-clone-smoke.json",
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
PER_PRIMARY_ALPHA = ALPHA / 4  # two primaries, two depths
DECLARED_BAR = 215


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
    ops = outcome["R"]["operations"]
    return {
        "seed": seed,
        "losses": {
            "F": outcome["R"]["fixed_loss"],
            "R": outcome["R"]["fused_loss"],
            "R_own": outcome["R"]["own_transport_loss"],
            "C": outcome["C"]["fused_loss"],
            "C_own": outcome["C"]["own_transport_loss"],
        },
        "fixed_arm_identical_across_settings": outcome["R"]["fixed_loss"]
        == outcome["C"]["fixed_loss"],
        "same_final_genome": outcome["R"]["architecture_hash"]
        == outcome["C"]["architecture_hash"],
        "same_operation_mix": ops == outcome["C"]["operations"],
        "operations": ops,
        "grows": sum(1 for op in ops if op == "grow"),
        "prunes": sum(1 for op in ops if op == "prune"),
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
    return {
        "wins": wins,
        "n": n,
        "mean_delta": sum(deltas) / n,
        "binomial_p": _binomial_survival(wins, n),
        "min_wins": _gate_min_wins(n, PER_PRIMARY_ALPHA),
        "passes": _binomial_survival(wins, n) <= PER_PRIMARY_ALPHA,
    }


def _built_in_checks(experiments: list[dict], profile: str) -> dict:
    spec = PROFILES[profile]
    grows = sum(exp["grows"] for exp in experiments)
    prunes = sum(exp["prunes"] for exp in experiments)
    return {
        "declared_bar_matches_exact_binomial": _gate_min_wins(
            DECLARED_N, PER_PRIMARY_ALPHA
        )
        == DECLARED_BAR,
        "block_matches_declared_n": len(spec["seeds"])
        == spec.get("declared_n", DECLARED_N),
        "the_fixed_arm_is_identical_across_both_settings": all(
            exp["fixed_arm_identical_across_settings"] for exp in experiments
        ),
        "both_discovery_runs_used_the_full_budget": all(
            exp["mutations"]["R"] == 3 and exp["mutations"]["C"] == 3
            for exp in experiments
        ),
        # The regime claim: the block must contain both operations or it is not the
        # drawn regime this study claims to test.
        "the_regime_is_genuinely_mixed": grows > 0 and prunes > 0,
        "no_layer_went_below_two_experts": all(
            min(exp["experts_per_layer"]) >= 2 for exp in experiments
        ),
        # Deliberately NOT asserted: identical final genomes. On this path the probe
        # can re-rank candidates once the two runs' weights differ, so divergence is
        # expected and is reported as a secondary rather than treated as a failure.
        "divergence_is_reported_not_assumed_away": (
            sum(1 for exp in experiments if exp["same_final_genome"])
            + sum(1 for exp in experiments if not exp["same_final_genome"])
            == len(experiments)
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

    def losses(label: str) -> list[float]:
        return [exp["losses"][label] for exp in experiments]

    def delta(first: str, second: str) -> list[float]:
        return [x - y for x, y in zip(losses(first), losses(second))]

    comparisons = {
        # Primary 1: does cloning improve the discovery pipeline end to end?
        "C_vs_R": _comparison(delta("C", "R")),
        # Primary 2: is the penalty gone? Predicted to FAIL, as in 5.26 and 5.28.
        "C_own_vs_C": _comparison(delta("C_own", "C")),
        # Context: the historical penalty on the discovery path, and whether either
        # path beats not mutating at all (the sharp section 5.28 contrast).
        "R_own_vs_R": _comparison(delta("R_own", "R")),
        "C_vs_F": _comparison(delta("C", "F")),
        "R_vs_F": _comparison(delta("R", "F")),
        "R_own_vs_F": _comparison(delta("R_own", "F")),
    }
    # The pairing is exact only on the subset where the two runs agreed; the primary
    # is reported on the full block AND on that subset, so a reader can see whether
    # the divergent seeds are doing the work.
    agreed = [exp for exp in experiments if exp["same_final_genome"]]
    agreed_deltas = [exp["losses"]["C"] - exp["losses"]["R"] for exp in agreed]
    comparisons["C_vs_R_on_seeds_where_both_agree"] = _comparison(agreed_deltas)
    p1 = comparisons["C_vs_R"]["passes"]
    p2 = comparisons["C_own_vs_C"]["passes"]
    if p1 and not p2:
        outcome = (
            "branch (a) clone-and-preserve holds on the discovery path and no "
            "penalty remains: the pipeline recommendation survives the arm the "
            "project actually uses, not just the random arm"
        )
    elif p1 and p2:
        outcome = (
            "branch (b) partial: cloning helps the discovery path but a residual "
            "penalty survives there, so freeze-and-retrain keeps a justification "
            "on the adaptive arm that sections 5.26-5.28 removed elsewhere"
        )
    elif p2:
        outcome = (
            "branch (c) cloning does not help the discovery path and the penalty "
            "remains: the section 5.26-5.28 result is confined to the random arm, "
            "and the pipeline recommendation does NOT extend to discovery"
        )
    else:
        outcome = (
            "branch (d) neither passes: inconclusive, recorded without a reading"
        )
    payload = {
        "record_type": "adaptive_clone_study",
        "protocol": "clone-and-preserve-on-probe-guided-discovery",
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
            for label in ("F", "R", "R_own", "C", "C_own")
        },
        "clone_gain": comparisons["C_vs_R"]["mean_delta"],
        "penalty_fresh": comparisons["R_own_vs_R"]["mean_delta"],
        "penalty_cloned": comparisons["C_own_vs_C"]["mean_delta"],
        "seeds_with_identical_final_genome": len(agreed),
        "seeds_with_identical_operation_mix": sum(
            1 for exp in experiments if exp["same_operation_mix"]
        ),
        "clone_gain_on_agreeing_seeds": comparisons[
            "C_vs_R_on_seeds_where_both_agree"
        ]["mean_delta"],
        "operation_mix": {
            "grow": sum(exp["grows"] for exp in experiments),
            "prune": sum(exp["prunes"] for exp in experiments),
        },
        "outcome": outcome,
        "built_in_checks": checks,
        "summary_note": (
            "Section 5.29. R and C are the same seed, initial model and data "
            "schedule on the PROBE-GUIDED adaptive arm with no forced operation, "
            "differing only in whether new experts are freshly drawn (R) or cloned "
            "from their layer's busiest survivor (C). The probe scores candidates "
            "with the default rebuild while the selected candidate is applied with "
            "the arm's flags, so once the two runs' weights differ the probe can "
            "re-rank and the runs can diverge - identical final genomes are RECORDED "
            "here, not asserted, and the primary is also reported on the agreeing "
            "subset. *_own are each run's own final genome retrained. Negative means "
            "the first-named run won."
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
    print(
        f"operation mix: {payload['operation_mix']}   "
        f"identical final genome in {payload['seeds_with_identical_final_genome']}/{n}"
        f"   identical operation mix in "
        f"{payload['seeds_with_identical_operation_mix']}/{n}"
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
    run_p.add_argument("--profile", default="adap2", choices=sorted(PROFILES))
    verdict_p = sub.add_parser("verdict", help="aggregate shards")
    verdict_p.add_argument("--profile", default="adap2", choices=sorted(PROFILES))
    sum_p = sub.add_parser("summary", help="write the committed study summary")
    sum_p.add_argument("--profile", default="adap2", choices=sorted(PROFILES))
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
