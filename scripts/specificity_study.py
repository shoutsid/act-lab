"""Section 5.20 study: is the shallow-depth topology value selection-specific?

Pre-registered in ``RESEARCH_REPORT.md`` section 5.20. Section 5.19 found the
adaptive arm's final topology winning under matched initialization at depths 2
and 3 (`260/384`, `270/384`) - where the *while-training* comparison has the same
arm losing to doing nothing. The alternative explanation is that nothing about
selection is being measured: the arm prunes, its topologies are smaller, and
perhaps any churned topology retrains better.

Section 5.17's depth-4 block separated these (random-churn topology `205/384`
mean `-0.0024` versus the adaptive topology `255/384`). This study measures the
same separation at depths 2 and 3, with three rewound transports per seed, all
on the seed's own initial model:

- ``A_own`` adaptive genome, rewound to the own initial model
- ``R_own`` random-churn genome, rewound to the own initial model
- ``F_own`` unmutated genome, rewound to the own initial model (reproduces the
          fixed arm's own run exactly - the validity check)

Primaries per depth (n = 384), Bonferroni alpha/4 = 0.0125 -> exact bar 215:

1. selection-specific value - ``A_own`` < ``R_own``
2. matched-init value replicates - ``A_own`` < ``F_own``

Usage:

    PYTHONPATH=src python scripts/specificity_study.py run --shard 0 --shards 20 --profile depth2
    PYTHONPATH=src python scripts/specificity_study.py verdict --profile depth2
    PYTHONPATH=src python scripts/specificity_study.py summary --profile depth2
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

SHARD_DIR = ROOT / "artifacts" / "toy-specificity-study"
PROFILES: dict[str, dict] = {
    "depth2": {
        "seeds": tuple(range(89000, 89384)),
        "overrides": {"depth": 2},
        "report_section": "5.20",
        "summary": "artifacts/study-summaries/specificity-depth2-384.json",
    },
    "depth3": {
        "seeds": tuple(range(91000, 91384)),
        "overrides": {"depth": 3},
        "report_section": "5.20",
        "summary": "artifacts/study-summaries/specificity-depth3-384.json",
    },
    # Section 5.22: the mid-training process penalty, on the design of section
    # 5.20 exactly. Same three transports; the declared primaries are the
    # fused-versus-frozen contrasts rather than the topology contrasts.
    "confirm2": {
        "seeds": tuple(range(92000, 92384)),
        "overrides": {"depth": 2},
        "report_section": "5.22",
        "summary": "artifacts/study-summaries/frozen-delivery-depth2-384.json",
    },
    "confirm3": {
        "seeds": tuple(range(93000, 93384)),
        "overrides": {"depth": 3},
        "report_section": "5.22",
        "summary": "artifacts/study-summaries/frozen-delivery-depth3-384.json",
    },
    "smoke": {
        "seeds": tuple(range(73000, 73004)),
        "overrides": {"depth": 2},
        "declared_n": 4,
        "report_section": "5.20 (smoke)",
        "summary": "artifacts/study-summaries/specificity-smoke.json",
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
ARMS = "random,adaptive,fixed"
#: label -> genome arm
TRANSPORTS = {"A_own": "adaptive", "R_own": "random", "F_own": "fixed"}
ALPHA = 0.05
DECLARED_N = 384
PER_PRIMARY_ALPHA = ALPHA / 4  # two depths x two primaries
#: The bar the pre-registration states, asserted against the exact computation
#: rather than restated (the section-5.16 declaration got this wrong).
DECLARED_BAR = 215


def _config(seed: int, profile: str) -> ExperimentConfig:
    return ExperimentConfig(
        **{**REGISTERED, **PROFILES[profile]["overrides"]}, seed=seed, arms=ARMS
    )


def _run_seed(seed: int, profile: str) -> dict:
    config = _config(seed, profile)
    arms = {arm.mode: arm for arm in run_experiment(config).arms}
    transport: dict[str, dict] = {}
    for label, mode in TRANSPORTS.items():
        result = transport_genome(
            ArchitectureGenome.from_dict(arms[mode].genome),
            config,
            initialization="rewound",
        )
        transport[label] = {
            "loss": result.final_loss,
            "parameter_count": result.parameter_count,
            "active_parameter_count": result.active_parameter_count,
            "training_flops": result.training_flops,
            "architecture_hash": result.architecture_hash,
            "initialization": dict(result.initialization),
        }
    return {
        "seed": seed,
        "discovery": {
            "losses": {mode: arms[mode].final_loss for mode in ("random", "adaptive", "fixed")},
            "mutations": {
                mode: arms[mode].mutation_count for mode in ("random", "adaptive", "fixed")
            },
            "training_flops": {
                mode: arms[mode].training_flops for mode in ("random", "adaptive", "fixed")
            },
        },
        "transport": transport,
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
    payload = {
        "shard": shard,
        "shards": shards,
        "profile": profile,
        "experiments": experiments,
    }
    path.write_text(json.dumps(payload) + "\n")
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
    seeds = PROFILES[profile]["seeds"]
    seen = {exp["seed"] for exp in experiments}
    if seen != set(seeds):
        raise SystemExit(f"shard set covers {len(seen)}/{len(seeds)} seeds; verdict withheld")
    return sorted(experiments, key=lambda exp: exp["seed"])


def _comparison(deltas: list[float]) -> dict:
    wins = sum(delta < 0 for delta in deltas)
    n = len(deltas)
    mean = sum(deltas) / n
    stddev = (
        (sum((value - mean) ** 2 for value in deltas) / n) ** 0.5 if n > 1 else 0.0
    )
    p = _binomial_survival(wins, n)
    bar = _gate_min_wins(n, PER_PRIMARY_ALPHA)
    return {
        "wins": wins,
        "n": n,
        "mean_delta": mean,
        "delta_stddev": stddev,
        "binomial_p": p,
        "min_wins": bar,
        "passes": p <= PER_PRIMARY_ALPHA,
        "bar_and_p_agree": (wins >= bar) == (p <= PER_PRIMARY_ALPHA),
    }


#: Section 5.22's profiles run the same design for a different question, so the
#: shipped defaults must stay section 5.20's.
CONFIRM_PROFILES = ("confirm2", "confirm3")


def _frozen_delivery(experiments: list[dict]) -> dict:
    """Section 5.22's primaries: does the fused run lose to its own frozen retrain?

    Negative deltas mean the frozen retrain won. Both rows are directional - the
    frozen path must WIN - so neither can be satisfied by the discounted sign.
    """
    return {
        "A_own_vs_fused_adaptive": _comparison(
            [
                exp["transport"]["A_own"]["loss"]
                - exp["discovery"]["losses"]["adaptive"]
                for exp in experiments
            ]
        ),
        "R_own_vs_fused_random": _comparison(
            [
                exp["transport"]["R_own"]["loss"]
                - exp["discovery"]["losses"]["random"]
                for exp in experiments
            ]
        ),
    }


def _frozen_delivery_outcome(comparisons: dict) -> str:
    adaptive = comparisons["A_own_vs_fused_adaptive"]["passes"]
    random_arm = comparisons["R_own_vs_fused_random"]["passes"]
    if adaptive and random_arm:
        return (
            "branch (a) both PASS: the penalty is a PROCESS property, not a "
            "topology one - the fused run is the wrong instrument for selection "
            "value and two-stage delivery (discover, freeze, retrain) is adopted"
        )
    if adaptive:
        return (
            "branch (b) adaptive PASS / random FAIL: the penalty is "
            "TOPOLOGY-specific after all - section 5.21's arm-generic reading is "
            "withdrawn and the fused comparison stands as the right instrument"
        )
    return (
        "branch (c) neither PASSES: REPLICATION FAILURE - section 5.17.2's "
        "observation and section 5.21's extension are both unconfirmed and no "
        "delivery-path change is adopted"
    )


def _coincident_arm_seeds(experiments: list[dict]) -> int:
    """Seeds whose adaptive and random arms discovered the same genome."""
    return sum(
        1
        for exp in experiments
        if exp["transport"]["A_own"]["architecture_hash"]
        == exp["transport"]["R_own"]["architecture_hash"]
    )


def _built_in_checks(experiments: list[dict], profile: str) -> dict:
    return {
        "declared_bar_matches_exact_binomial": _gate_min_wins(
            DECLARED_N, PER_PRIMARY_ALPHA
        )
        == DECLARED_BAR,
        "block_matches_declared_n": len(PROFILES[profile]["seeds"])
        == PROFILES[profile].get("declared_n", DECLARED_N),
        "own_rewind_reproduces_fixed_arm_exactly": all(
            exp["transport"]["F_own"]["loss"] == exp["discovery"]["losses"]["fixed"]
            and exp["transport"]["F_own"]["training_flops"]
            == exp["discovery"]["training_flops"]["fixed"]
            for exp in experiments
        ),
        "rewound_transports_transfer_parameters": all(
            exp["transport"][label]["initialization"]["copied_parameters"] > 0
            for exp in experiments
            for label in TRANSPORTS
        ),
        # The two arms draw independently, so a chance coincidence is possible
        # where the feasible topology space is small - depth 2 is 2 layers x 4
        # experts over 3 mutations, and one seed in 384 lands on the same
        # genome. A coincidence is harmless only if the transports are then the
        # SAME architecture on the SAME initialization, so the seed's delta is
        # exactly zero and it can never be counted as a win; and only if such
        # seeds are too rare to make the comparison architecture-to-itself.
        # Section 5.20.1 records why this replaced an all-384-differ assertion.
        "coincident_arm_topologies_are_exact_ties": all(
            exp["transport"]["A_own"]["loss"] == exp["transport"]["R_own"]["loss"]
            for exp in experiments
            if exp["transport"]["A_own"]["architecture_hash"]
            == exp["transport"]["R_own"]["architecture_hash"]
        ),
        "coincident_arm_topologies_are_rare": (
            _coincident_arm_seeds(experiments) <= max(1, 0.01 * len(experiments))
        ),
        "adaptive_mutated_3": all(
            exp["discovery"]["mutations"]["adaptive"] == 3 for exp in experiments
        ),
        "random_mutated_3": all(
            exp["discovery"]["mutations"]["random"] == 3 for exp in experiments
        ),
        "fixed_never_mutated": all(
            exp["discovery"]["mutations"]["fixed"] == 0 for exp in experiments
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
        return [exp["transport"][label]["loss"] for exp in experiments]

    def delta(a: str, b: str) -> list[float]:
        return [x - y for x, y in zip(losses(a), losses(b))]

    comparisons = {
        "A_own_vs_R_own": _comparison(delta("A_own", "R_own")),
        "A_own_vs_F_own": _comparison(delta("A_own", "F_own")),
        # The control that defines what "value" means: does churn alone produce a
        # better-retraining topology? If this passes, primary 1 inverts.
        "R_own_vs_F_own": _comparison(delta("R_own", "F_own")),
    }
    active = {
        label: sum(exp["transport"][label]["active_parameter_count"] for exp in experiments)
        / n
        for label in TRANSPORTS
    }
    payload = {
        "record_type": "specificity_study",
        "protocol": "selection-specificity-under-matched-initialization",
        "profile": profile,
        "report_section": spec["report_section"],
        "alpha": ALPHA,
        "per_primary_alpha": PER_PRIMARY_ALPHA,
        "declared_bar": DECLARED_BAR,
        "depth": PROFILES[profile]["overrides"]["depth"],
        "arms": ["random", "adaptive", "fixed"],
        "transports": dict(TRANSPORTS),
        "independent_arm_rng": True,
        "seeds": list(spec["seeds"]),
        "config": {**REGISTERED, **spec["overrides"], "arms": ARMS},
        "per_seed": [
            {
                "seed": exp["seed"],
                "discovery_losses": exp["discovery"]["losses"],
                "transport_losses": {
                    label: exp["transport"][label]["loss"] for label in TRANSPORTS
                },
                "transport_active_parameter_count": {
                    label: exp["transport"][label]["active_parameter_count"]
                    for label in TRANSPORTS
                },
                # Carried so the coincidence guard of section 5.20.1 is
                # recomputable from the artifact rather than taken on trust.
                "transport_architecture_hashes": {
                    label: exp["transport"][label]["architecture_hash"]
                    for label in TRANSPORTS
                },
            }
            for exp in experiments
        ],
        "comparisons": comparisons,
        "transport_means": {label: sum(losses(label)) / n for label in TRANSPORTS},
        "discovery_means": {
            arm: sum(exp["discovery"]["losses"][arm] for exp in experiments) / n
            for arm in ("random", "adaptive", "fixed")
        },
        "mean_active_parameters": active,
        # Seeds where the two arms independently landed on the same genome. They
        # are exact ties and can only bias AGAINST primary 1, so the primary is
        # also computed without them (``*_differ_only``) in section 5.20.1.
        "coincident_arm_topology_seeds": _coincident_arm_seeds(experiments),
        "adaptive_smaller_than_random_seeds": sum(
            1
            for exp in experiments
            if exp["transport"]["A_own"]["active_parameter_count"]
            < exp["transport"]["R_own"]["active_parameter_count"]
        ),
        "random_matches_initial_size_seeds": sum(
            1
            for exp in experiments
            if exp["transport"]["R_own"]["active_parameter_count"]
            == exp["transport"]["F_own"]["active_parameter_count"]
        ),
        "built_in_checks": checks,
        "summary_note": (
            "Selection specificity at shallow depth (report section 5.20): three "
            "rewound transports per seed on the seed's own initial model, so "
            "initialization is matched on every side and only the topology "
            "differs. F_own is the unmutated genome and must reproduce the fixed "
            "arm's own run exactly. R_own is the control that decides what the "
            "advantage means: if churn alone beats the initial topology, "
            "A_own_vs_R_own is not evidence of selection. Statistics recompute "
            "exactly from per_seed; p-values are exact one-tailed binomial."
        ),
    }
    if profile in CONFIRM_PROFILES:
        # Section 5.22's own block, emitted only for its own profiles so each
        # record carries exactly the study it belongs to (the section-5.20
        # profiles' committed payloads stay as they were written).
        frozen = _frozen_delivery(experiments)
        payload["frozen_delivery"] = frozen
        payload["frozen_delivery_outcome"] = _frozen_delivery_outcome(frozen)
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
    n = len(payload["seeds"])
    print(
        f"built-in check: bar {payload['declared_bar']} at alpha={payload['per_primary_alpha']}; "
        f"own rewind reproduces the fixed arm exactly: "
        f"{payload['built_in_checks']['own_rewind_reproduces_fixed_arm_exactly']}"
    )
    print(f"transport means: {payload['transport_means']}")
    print(f"discovery means: {payload['discovery_means']}")
    print(f"mean active parameters: {payload['mean_active_parameters']}")
    for name, comparison in payload["comparisons"].items():
        print(
            f"\n{name}: wins {comparison['wins']}/{comparison['n']}, "
            f"mean {comparison['mean_delta']:+.6f}, p {comparison['binomial_p']:.2e} "
            f"(bar {comparison['min_wins']}): "
            f"{'PASS' if comparison['passes'] else 'FAIL'}"
        )
    print(
        f"\ncapacity: adaptive smaller than random-churn topology in "
        f"{payload['adaptive_smaller_than_random_seeds']}/{n} seeds; random-churn "
        f"topology the same size as the initial one in "
        f"{payload['random_matches_initial_size_seeds']}/{n}"
    )
    if "frozen_delivery" not in payload:
        return 0
    print("\n--- section 5.22: fused run versus its own frozen retrain ---")
    for name, comparison in payload["frozen_delivery"].items():
        print(
            f"{name}: wins {comparison['wins']}/{comparison['n']}, "
            f"mean {comparison['mean_delta']:+.6f}, p {comparison['binomial_p']:.2e} "
            f"(bar {comparison['min_wins']}): "
            f"{'PASS' if comparison['passes'] else 'FAIL'}"
        )
    print(f"OUTCOME: {payload['frozen_delivery_outcome']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run one shard")
    run_p.add_argument("--shard", type=int, required=True)
    run_p.add_argument("--shards", type=int, required=True)
    run_p.add_argument("--profile", default="depth2", choices=sorted(PROFILES))
    verdict_p = sub.add_parser("verdict", help="aggregate shards")
    verdict_p.add_argument("--profile", default="depth2", choices=sorted(PROFILES))
    sum_p = sub.add_parser("summary", help="write the committed study summary")
    sum_p.add_argument("--profile", default="depth2", choices=sorted(PROFILES))
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
