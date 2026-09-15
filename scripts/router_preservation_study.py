"""Section 5.24 study: is the mid-training penalty the discarded routing head?

Pre-registered in ``RESEARCH_REPORT.md`` section 5.24. Section 5.23 exonerated
optimizer history, leaving the architecture change itself as the mechanism. This
study attacks the sharpest candidate inside it, which is a **defect, not a
design choice**: the router's weight is ``hidden x experts``, so it changes shape
whenever a layer's expert count changes. ``rebuild_model_from_genome`` copies
same-shape keys, so the mutated layer's *entire* learned routing function is
re-drawn at random at every event - while the experts' own weights survive by id.
(The unmutated layers' routers survive, which is how the loss is visible at all.)

``preserve_router=True`` copies the surviving experts' routing rows by id and
leaves only a grown expert's row fresh. Per seed, on one initial model and one
data schedule, with ``arms="fixed,random"``:

- ``F``      the fixed arm (unmutated) - identical under both settings;
- ``R``      the fused random arm, routing head discarded on rebuild (historical);
- ``R_own``  ``R``'s final genome retrained from its own initialization;
- ``Rp``     the same fused arm with the routing head preserved;
- ``Rp_own`` ``Rp``'s final genome retrained from its own initialization.

Primaries per depth (n = 384), Bonferroni ``alpha/4 = 0.0125`` -> bar 215:

1. **preservation helps** - ``Rp`` < ``R``.
2. **residual penalty** - ``Rp_own`` < ``Rp``: does a penalty remain once the
   routing head survives the mutation?

Usage:

    PYTHONPATH=src python scripts/router_preservation_study.py run --shard 0 --shards 20 --profile router2
    PYTHONPATH=src python scripts/router_preservation_study.py verdict --profile router2
    PYTHONPATH=src python scripts/router_preservation_study.py summary --profile router2
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
from act.topology import ArchitectureGenome  # noqa: E402

SHARD_DIR = ROOT / "artifacts" / "toy-router-study"
PROFILES: dict[str, dict] = {
    "router2": {
        "seeds": tuple(range(96000, 96384)),
        "overrides": {"depth": 2},
        "report_section": "5.24",
        "summary": "artifacts/study-summaries/router-preservation-depth2-384.json",
    },
    "router3": {
        "seeds": tuple(range(97000, 97384)),
        "overrides": {"depth": 3},
        "report_section": "5.24",
        "summary": "artifacts/study-summaries/router-preservation-depth3-384.json",
    },
    "smoke": {
        "seeds": tuple(range(73000, 73004)),
        "overrides": {"depth": 2},
        "declared_n": 4,
        "report_section": "5.24 (smoke)",
        "summary": "artifacts/study-summaries/router-preservation-smoke.json",
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
)
ALPHA = 0.05
DECLARED_N = 384
PER_PRIMARY_ALPHA = ALPHA / 4
DECLARED_BAR = 215


def _run_seed(seed: int, profile: str) -> dict:
    values = {**REGISTERED, **PROFILES[profile]["overrides"]}
    outcome: dict[str, dict] = {}
    for label, preserve in (("discarded", False), ("preserved", True)):
        config = ExperimentConfig(
            **{
                **values,
                "seed": seed,
                "arms": "fixed,random",
                "preserve_router_on_rebuild": preserve,
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
            # Validity: transporting the UNMUTATED genome must reproduce the
            # fixed arm's own run exactly, under this setting as under any other.
            "fixed_transport_loss": transport_genome(
                ArchitectureGenome.from_dict(arms["fixed"].genome),
                config,
                initialization="rewound",
            ).final_loss,
        }
    return {
        "seed": seed,
        "losses": {
            "F": outcome["discarded"]["fixed_loss"],
            "R": outcome["discarded"]["fused_loss"],
            "R_own": outcome["discarded"]["own_transport_loss"],
            "Rp": outcome["preserved"]["fused_loss"],
            "Rp_own": outcome["preserved"]["own_transport_loss"],
        },
        "fixed_check": {
            "discarded_reproduces": outcome["discarded"]["fixed_transport_loss"]
            == outcome["discarded"]["fixed_loss"],
            "preserved_reproduces": outcome["preserved"]["fixed_transport_loss"]
            == outcome["preserved"]["fixed_loss"],
            "preservation_leaves_the_fixed_arm_alone": outcome["discarded"][
                "fixed_loss"
            ]
            == outcome["preserved"]["fixed_loss"],
        },
        "mutations": {
            "R": outcome["discarded"]["fused_mutations"],
            "Rp": outcome["preserved"]["fused_mutations"],
        },
        "routers_differ": _stable_hash(outcome["discarded"]["fused_genome"])
        != _stable_hash(outcome["preserved"]["fused_genome"]),
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
    mean = sum(deltas) / n
    p = _binomial_survival(wins, n)
    bar = _gate_min_wins(n, PER_PRIMARY_ALPHA)
    return {
        "wins": wins,
        "n": n,
        "mean_delta": mean,
        "binomial_p": p,
        "min_wins": bar,
        "passes": p <= PER_PRIMARY_ALPHA,
        "bar_and_p_agree": (wins >= bar) == (p <= PER_PRIMARY_ALPHA),
    }


def _built_in_checks(experiments: list[dict], profile: str) -> dict:
    return {
        "declared_bar_matches_exact_binomial": _gate_min_wins(
            DECLARED_N, PER_PRIMARY_ALPHA
        )
        == DECLARED_BAR,
        "block_matches_declared_n": len(PROFILES[profile]["seeds"])
        == PROFILES[profile].get("declared_n", DECLARED_N),
        "fixed_transport_reproduces_the_fixed_arm_under_both_settings": all(
            exp["fixed_check"]["discarded_reproduces"]
            and exp["fixed_check"]["preserved_reproduces"]
            for exp in experiments
        ),
        # The flag must not perturb a run that never mutates, or P1 and P2 would
        # be comparing two different baselines rather than one.
        "preservation_leaves_the_fixed_arm_alone": all(
            exp["fixed_check"]["preservation_leaves_the_fixed_arm_alone"]
            for exp in experiments
        ),
        "both_fused_runs_used_the_full_budget": all(
            exp["mutations"]["R"] == 3 and exp["mutations"]["Rp"] == 3
            for exp in experiments
        ),
    }


def summarize(profile: str) -> dict:
    spec = PROFILES[profile]
    experiments = load_shards(_shards_for(profile), profile)
    n = len(experiments)
    checks = _built_in_checks(experiments, profile)
    if not all(checks.values()):
        raise SystemExit(f"built-in checks failed: {checks}")

    def losses(label: str) -> list[float]:
        return [exp["losses"][label] for exp in experiments]

    def delta(first: str, second: str) -> list[float]:
        return [x - y for x, y in zip(losses(first), losses(second))]

    comparisons = {
        # Primary 1: does keeping the learned routing head help the fused run?
        "Rp_vs_R": _comparison(delta("Rp", "R")),
        # Primary 2: does a penalty remain once the routing head survives?
        "Rp_own_vs_Rp": _comparison(delta("Rp_own", "Rp")),
        # Context: the historical penalty on this block, for continuity.
        "R_own_vs_R": _comparison(delta("R_own", "R")),
    }
    p1 = comparisons["Rp_vs_R"]["passes"]
    p2 = comparisons["Rp_own_vs_Rp"]["passes"]
    if p1 and not p2:
        outcome = (
            "branch (a) preservation helps and no penalty remains: the discarded "
            "routing head IS the mechanism - cheap fix available, section 5.22 "
            "qualified"
        )
    elif p1 and p2:
        outcome = (
            "branch (b) partial: preservation helps but a residual penalty "
            "survives, so the mechanism is more than the routing head"
        )
    elif p2:
        outcome = (
            "branch (c) preservation does not help and the penalty remains: the "
            "discarded routing head is NOT the mechanism - the cost of changing "
            "the architecture mid-run is intrinsic, and section 5.22 stands"
        )
    else:
        outcome = (
            "branch (d) preservation neither helps nor leaves a penalty: "
            "inconclusive, recorded without either reading"
        )
    payload = {
        "record_type": "router_preservation_study",
        "protocol": "discarded-routing-head-vs-preserved",
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
        "loss_means": {label: sum(losses(label)) / n for label in ("F", "R", "R_own", "Rp", "Rp_own")},
        "penalty_discarded": comparisons["R_own_vs_R"]["mean_delta"],
        "penalty_preserved": comparisons["Rp_own_vs_Rp"]["mean_delta"],
        "preservation_gain": comparisons["Rp_vs_R"]["mean_delta"],
        "outcome": outcome,
        "built_in_checks": checks,
        "summary_note": (
            "Section 5.24. R and Rp are the same seed, initial model and data "
            "schedule, differing only in whether a mutated layer's learned router "
            "survives the rebuild. R_own and Rp_own are the frozen retrains of each "
            "fused run's own final genome, so R_own-R and Rp_own-Rp are the two "
            "penalties. Negative means the first-named run won."
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
    print(f"loss means: {payload['loss_means']}")
    for name, comparison in payload["comparisons"].items():
        print(
            f"\n{name}: wins {comparison['wins']}/{comparison['n']}, "
            f"mean {comparison['mean_delta']:+.6f}, p {comparison['binomial_p']:.2e} "
            f"(bar {comparison['min_wins']}): "
            f"{'PASS' if comparison['passes'] else 'FAIL'}"
        )
    print(
        f"\npenalty (router discarded): {payload['penalty_discarded']:+.4f}   "
        f"penalty (router preserved): {payload['penalty_preserved']:+.4f}   "
        f"gain from preserving: {payload['preservation_gain']:+.4f}"
    )
    print(f"\nOUTCOME: {payload['outcome']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run one shard")
    run_p.add_argument("--shard", type=int, required=True)
    run_p.add_argument("--shards", type=int, required=True)
    run_p.add_argument("--profile", default="router2", choices=sorted(PROFILES))
    verdict_p = sub.add_parser("verdict", help="aggregate shards")
    verdict_p.add_argument("--profile", default="router2", choices=sorted(PROFILES))
    sum_p = sub.add_parser("summary", help="write the committed study summary")
    sum_p.add_argument("--profile", default="router2", choices=sorted(PROFILES))
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
