"""Section 5.17 study: is initialization or topology the important artifact?

Pre-registered in ``RESEARCH_REPORT.md`` section 5.17 (plan RQ6 / section 55).
Per seed, one discovery block on the section-5.10/5.16 registered config
(``arms="random,adaptive,fixed"``, depth 4, ``probe_candidate_policy="spread"``,
cap-4/2-batch probes, 48 steps, interval 12, budget 3, ``independent_arm_rng``,
cpu), then the discovered genomes are transported under both plan-section-55
initialization treatments:

- ``B`` rewound - rebuilt, then every component that existed at initialization
  restored to its initial weights by component identity (MoE expert id); grown
  experts keep the fresh draw. On an unmutated genome this is identical to
  ``C`` by construction.
- ``C`` fresh - rebuilt from the block's own draw, nothing transferred.

Five transports per seed: B/C for the adaptive and random genomes, and B for the
fixed (unmutated) genome, which must reproduce the fixed arm's own discovery run
exactly. ``A`` (evolved weights) is the discovery arm's own run; ``D`` (teacher)
is out of scope at toy fidelity and is not faked.

Primaries (exact one-tailed sign test, n = 384, per-primary threshold
alpha/2 = 0.025 - the section 5.9-5.15 convention - which at n = 384 is the
exact bar 212 wins; the declared bar is asserted against ``_gate_min_wins`` at
run time, so the section-5.16 inconsistency cannot recur silently):

1. initialization carries value - B_adaptive < C_adaptive.
2. topology alone, initialization level - B_adaptive < B_fixed.

Usage:

    PYTHONPATH=src python scripts/initialization_study.py run --shard 0 --shards 20
    PYTHONPATH=src python scripts/initialization_study.py verdict --shards 20
    PYTHONPATH=src python scripts/initialization_study.py summary
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
from act.topology import ArchitectureGenome, create_genome, genome_hash  # noqa: E402

SHARD_DIR = ROOT / "artifacts" / "toy-initialization-study"
PROFILES: dict[str, dict] = {
    # The registered block (fresh: used by no other study).
    "depth4": {
        "seeds": tuple(range(82000, 82384)),
        "report_section": "5.17",
        "summary": "artifacts/study-summaries/initialization-depth4-384.json",
    },
    # Machinery smoke test only, on a spent block.
    "smoke": {
        "seeds": tuple(range(73000, 73004)),
        "report_section": "5.17 (smoke)",
        "summary": "artifacts/study-summaries/initialization-smoke.json",
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
TREATMENTS = {
    "B_adaptive": ("adaptive", "rewound"),
    "C_adaptive": ("adaptive", "fresh"),
    "B_random": ("random", "rewound"),
    "C_random": ("random", "fresh"),
    "B_fixed": ("fixed", "rewound"),
}
ALPHA = 0.05
# The declared per-primary threshold and the exact bar it implies, which the
# script ASSERTS rather than restating: section 5.16's declaration stated a
# threshold and a bar that matched neither each other nor any bar at n = 384.
DECLARED_ALPHA = ALPHA / 2  # 0.025, the section 5.9-5.15 convention
DECLARED_N = 384  # the declared study size (block 82000-82383)
DECLARED_BAR = 212  # exact one-tailed binomial bar at n = 384, alpha = 0.025


def _initial_genome(seed: int) -> ArchitectureGenome:
    return create_genome(
        seed,
        hidden_size=REGISTERED["hidden_size"],
        vocab_size=REGISTERED["vocab_size"],
        depth=REGISTERED["depth"],
        ffn="moe",
        expert_count=REGISTERED["expert_count"],
        active_experts=REGISTERED["active_experts"],
    )


def _grown_experts(genome: ArchitectureGenome, initial: ArchitectureGenome) -> int:
    """How many experts exist that did not exist at initialization."""
    grown = 0
    for layer, initial_layer in zip(genome.layers, initial.layers):
        initial_ids = {expert.id for expert in initial_layer.experts}
        grown += sum(1 for expert in layer.experts if expert.id not in initial_ids)
    return grown


def _run_seed(seed: int) -> dict:
    config = ExperimentConfig(**REGISTERED, seed=seed, arms=ARMS)
    discovery = run_experiment(config)
    arms = {arm.mode: arm for arm in discovery.arms}
    initial = _initial_genome(seed)
    initial_hash = genome_hash(initial)
    transport: dict[str, dict] = {}
    grown: dict[str, int] = {}
    unchanged: dict[str, bool] = {}
    for label, (mode, initialization) in TREATMENTS.items():
        genome = ArchitectureGenome.from_dict(arms[mode].genome)
        result = transport_genome(genome, config, initialization=initialization)
        transport[label] = {
            "loss": result.final_loss,
            "parameter_count": result.parameter_count,
            "active_parameter_count": result.active_parameter_count,
            "training_flops": result.training_flops,
            "architecture_hash": result.architecture_hash,
            "initialization": dict(result.initialization),
        }
        grown[label] = _grown_experts(genome, initial)
        unchanged[label] = result.architecture_hash == initial_hash
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
            "architecture_hash": {
                mode: arms[mode].architecture_hash for mode in ("random", "adaptive", "fixed")
            },
        },
        "transport": transport,
        "grown_experts": grown,
        "topology_unchanged": unchanged,
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
    experiments = [_run_seed(seed) for seed in PROFILES[profile]["seeds"][shard::shards]]
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
    return 20 if profile == "depth4" else 1


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


def _comparison(deltas: list[float], alpha: float = DECLARED_ALPHA) -> dict:
    """One pre-registered primary: sign-test statistics at the declared bar."""
    wins = sum(delta < 0 for delta in deltas)
    n = len(deltas)
    mean = sum(deltas) / n
    stddev = (
        (sum((value - mean) ** 2 for value in deltas) / n) ** 0.5 if n > 1 else 0.0
    )
    p = _binomial_survival(wins, n)
    bar = _gate_min_wins(n, alpha)
    return {
        "wins": wins,
        "n": n,
        "mean_delta": mean,
        "delta_stddev": stddev,
        "binomial_p": p,
        "min_wins": bar,
        "passes": p <= alpha,
        # Both statements of the same test, recorded so a reader can see the
        # bar and the p-threshold agree (section 5.16's declaration did not).
        "bar_and_p_agree": (wins >= bar) == (p <= alpha),
    }


def _built_in_checks(experiments: list[dict], n: int) -> dict:
    """Pre-conditions that must hold before the primaries can be read.

    The rewind identity is the SUBTLE one, and section 5.17 declares it wrongly
    at first (see report section 5.17.1): B and C coincide only when the
    *architecture* is unchanged, because the fresh draw depends on the shapes
    being drawn. A genome that only pruned experts is still a different
    architecture, so its fresh draw differs and rewinding genuinely restores
    the survivors' initial weights - which is what makes B vs C a comparison
    there rather than a tautology. "No grown experts" is therefore NOT the
    condition, and the pre-run smoke test caught the claim that it was.
    """
    return {
        "declared_bar_matches_exact_binomial": _gate_min_wins(DECLARED_N, DECLARED_ALPHA)
        == DECLARED_BAR,
        "registered_block_matches_declared_n": len(PROFILES["depth4"]["seeds"])
        == DECLARED_N,
        "rewound_transfers_parameters": all(
            exp["transport"][label]["initialization"]["copied_parameters"] > 0
            for exp in experiments
            for label in ("B_adaptive", "B_random", "B_fixed")
        ),
        "unchanged_topologies_have_b_equal_to_c": all(
            exp["transport"][f"B_{arm}"]["loss"] == exp["transport"][f"C_{arm}"]["loss"]
            for exp in experiments
            for arm in ("adaptive", "random")
            if exp["topology_unchanged"][f"B_{arm}"]
        ),
        "mutated_topologies_have_b_different_from_c": all(
            exp["transport"][f"B_{arm}"]["loss"] != exp["transport"][f"C_{arm}"]["loss"]
            for exp in experiments
            for arm in ("adaptive", "random")
            if not exp["topology_unchanged"][f"B_{arm}"]
        ),
        "rewound_reproduces_fixed_arm_exactly": all(
            exp["transport"]["B_fixed"]["loss"] == exp["discovery"]["losses"]["fixed"]
            and exp["transport"]["B_fixed"]["training_flops"]
            == exp["discovery"]["training_flops"]["fixed"]
            for exp in experiments
        ),
        "fixed_never_mutated": all(
            exp["discovery"]["mutations"]["fixed"] == 0 for exp in experiments
        ),
        "adaptive_mutated_3": all(
            exp["discovery"]["mutations"]["adaptive"] == 3 for exp in experiments
        ),
        "random_mutated_3": all(
            exp["discovery"]["mutations"]["random"] == 3 for exp in experiments
        ),
        "transport_modes_recorded": all(
            exp["transport"]["B_adaptive"]["initialization"]["mode"] == "rewound"
            and exp["transport"]["C_adaptive"]["initialization"]["mode"] == "fresh"
            and exp["transport"]["B_random"]["initialization"]["mode"] == "rewound"
            and exp["transport"]["C_random"]["initialization"]["mode"] == "fresh"
            and exp["transport"]["B_fixed"]["initialization"]["mode"] == "rewound"
            for exp in experiments
        ),
    }


def _capacity_view(experiments: list[dict], other: str) -> dict:
    matched = [
        (exp["transport"]["B_adaptive"]["loss"], exp["transport"][other]["loss"])
        for exp in experiments
        if exp["transport"]["B_adaptive"]["active_parameter_count"]
        == exp["transport"][other]["active_parameter_count"]
    ]
    return {
        "matched_seeds": len(matched),
        "matched": _comparison([a - b for a, b in matched]) if matched else None,
        "adaptive_active_params_higher_seeds": sum(
            exp["transport"]["B_adaptive"]["active_parameter_count"]
            > exp["transport"][other]["active_parameter_count"]
            for exp in experiments
        ),
        "adaptive_active_params_lower_seeds": sum(
            exp["transport"]["B_adaptive"]["active_parameter_count"]
            < exp["transport"][other]["active_parameter_count"]
            for exp in experiments
        ),
        "n": len(experiments),
    }


def summarize(profile: str) -> dict:
    """Build the committed study summary from this profile's shard payloads."""
    spec = PROFILES[profile]
    experiments = load_shards(_shards_for(profile), profile)
    n = len(experiments)
    checks = _built_in_checks(experiments, n)
    if not all(checks.values()):
        raise SystemExit(f"built-in checks failed: {checks}")
    def delta(a: str, b: str) -> list[float]:
        return [
            exp["transport"][a]["loss"] - exp["transport"][b]["loss"]
            for exp in experiments
        ]

    comparisons = {
        "B_adaptive_vs_C_adaptive": _comparison(delta("B_adaptive", "C_adaptive")),
        "B_adaptive_vs_B_fixed": _comparison(delta("B_adaptive", "B_fixed")),
        "B_random_vs_C_random": _comparison(delta("B_random", "C_random")),
        "C_adaptive_vs_B_fixed": _comparison(delta("C_adaptive", "B_fixed")),
    }
    trained_weight_deltas = [
        exp["discovery"]["losses"]["adaptive"] - exp["transport"]["B_adaptive"]["loss"]
        for exp in experiments
    ]
    trained_weights = _comparison(trained_weight_deltas)
    transport_means = {
        label: sum(exp["transport"][label]["loss"] for exp in experiments) / n
        for label in TREATMENTS
    }
    p1 = comparisons["B_adaptive_vs_C_adaptive"]["passes"]
    p2 = comparisons["B_adaptive_vs_B_fixed"]["passes"]
    if p1 and p2:
        outcome = (
            "branch (d) both PASS: both the structure and its initialization "
            "carry value"
        )
    elif p1:
        outcome = (
            "branch (b) 1 PASS / 2 FAIL: reusing the original initialization "
            "preserves the advantage while the structure alone does not - a "
            "lottery-ticket-like reading"
        )
    elif p2:
        outcome = (
            "branch (c) 1 FAIL / 2 PASS: the discovered topology IS a standalone "
            "artifact once initialization is held level; section 5.16's verdict "
            "is revised"
        )
    else:
        outcome = (
            "branch (a) both FAIL: neither the topology nor its original "
            "initialization is a standalone artifact at toy fidelity - the "
            "advantage is the trained weights"
        )
    payload = {
        "record_type": "initialization_study",
        "protocol": "discover-then-fresh-and-rewound-init-transport",
        "profile": profile,
        "report_section": spec["report_section"],
        "alpha": ALPHA,
        "per_primary_alpha": DECLARED_ALPHA,
        "declared_bar": DECLARED_BAR,
        "arms": ["random", "adaptive", "fixed"],
        "treatments": {label: list(pair) for label, pair in TREATMENTS.items()},
        "independent_arm_rng": True,
        "seeds": list(spec["seeds"]),
        "config": {**REGISTERED, "arms": ARMS},
        "per_seed": [
            {
                "seed": exp["seed"],
                "discovery_losses": exp["discovery"]["losses"],
                "transport_losses": {
                    label: exp["transport"][label]["loss"] for label in TREATMENTS
                },
                "transport_active_parameter_count": {
                    label: exp["transport"][label]["active_parameter_count"]
                    for label in TREATMENTS
                },
                "grown_experts": exp["grown_experts"],
            }
            for exp in experiments
        ],
        "comparisons": comparisons,
        "trained_weight_value": trained_weights,
        "capacity": {
            "B_adaptive_vs_B_fixed": _capacity_view(experiments, "B_fixed"),
            "B_adaptive_vs_C_adaptive": _capacity_view(experiments, "C_adaptive"),
        },
        "transport_means": transport_means,
        "section_55_inventory": {
            "n": n,
            "pure_prune_seeds": sum(
                1 for exp in experiments if exp["grown_experts"]["B_adaptive"] == 0
            ),
            "grown_expert_seeds": sum(
                1 for exp in experiments if exp["grown_experts"]["B_adaptive"] > 0
            ),
            "unchanged_adaptive_topologies": sum(
                1 for exp in experiments if exp["topology_unchanged"]["B_adaptive"]
            ),
            "grown_experts_total": sum(
                exp["grown_experts"]["B_adaptive"] for exp in experiments
            ),
        },
        "built_in_checks": checks,
        "outcome": outcome,
        "summary_note": (
            "Initialization vs topology (report section 5.17, plan RQ6 / section "
            "55): each discovered genome is transported under BOTH section-55 "
            "initialization treatments - 'rewound' restores every component that "
            "existed at initialization to its initial weights by MoE expert id "
            "(grown experts keep the fresh draw), 'fresh' transfers nothing. "
            "B_fixed reproduces the fixed arm's own run exactly on every seed, "
            "which is the per-seed validity check; the teacher treatment is out "
            "of scope at toy fidelity. Statistics recompute exactly from "
            "per_seed; p-values are exact one-tailed binomial."
        ),
    }
    payload["study_hash"] = _stable_hash(
        {
            "config": payload["config"],
            "seeds": payload["seeds"],
            "per_seed": payload["per_seed"],
            "alpha": DECLARED_ALPHA,
        }
    )
    payload["study_id"] = f"study_{payload['study_hash']}"
    payload["config_hash"] = _stable_hash(payload["config"])
    return payload


def verdict(profile: str) -> int:
    payload = summarize(profile)
    n = len(payload["seeds"])
    print(
        f"built-in check: declared bar {DECLARED_BAR} equals the exact binomial bar at "
        f"n={DECLARED_N}, alpha={DECLARED_ALPHA}: "
        f"{payload['built_in_checks']['declared_bar_matches_exact_binomial']}"
    )
    print(
        "built-in check: rewound transport of the unmutated genome reproduces the "
        f"fixed arm exactly: {payload['built_in_checks']['rewound_reproduces_fixed_arm_exactly']}"
    )
    inventory = payload["section_55_inventory"]
    print(
        f"built-in check: B == C exactly where the topology is unchanged and differs "
        f"where it is not ("
        f"{inventory['unchanged_adaptive_topologies']}/{n} adaptive topologies unchanged, "
        f"{inventory['pure_prune_seeds']}/{n} seeds had no grown experts, "
        f"{inventory['grown_experts_total']} grown experts total): "
        f"{payload['built_in_checks']['unchanged_topologies_have_b_equal_to_c'] and payload['built_in_checks']['mutated_topologies_have_b_different_from_c']}"
    )
    print(f"\ntransport means: {payload['transport_means']}")
    print("discovery means:")
    for arm in ("random", "adaptive", "fixed"):
        mean = sum(exp["discovery_losses"][arm] for exp in payload["per_seed"]) / n
        print(f"    {arm:9s} {mean:.6f}")
    for name, comparison in payload["comparisons"].items():
        print(
            f"\n{name}: wins {comparison['wins']}/{comparison['n']}, "
            f"mean {comparison['mean_delta']:+.6f}, p {comparison['binomial_p']:.2e} "
            f"(bar {comparison['min_wins']}): "
            f"{'PASS' if comparison['passes'] else 'FAIL'}"
        )
    tw = payload["trained_weight_value"]
    print(
        f"\nsecondary A_adaptive - B_adaptive (what only the trained weights hold): "
        f"wins {tw['wins']}/{tw['n']}, mean {tw['mean_delta']:+.6f}, "
        f"p {tw['binomial_p']:.2e}"
    )
    for name, view in payload["capacity"].items():
        matched = view["matched"]
        detail = (
            f", matched subset {matched['wins']}/{matched['n']} (bar {matched['min_wins']})"
            if matched
            else ", no capacity-matched seeds"
        )
        print(
            f"capacity {name}: adaptive active params higher in "
            f"{view['adaptive_active_params_higher_seeds']}/{view['n']}, lower in "
            f"{view['adaptive_active_params_lower_seeds']}{detail}"
        )
    print(f"\nOUTCOME: {payload['outcome']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run one shard")
    run_p.add_argument("--shard", type=int, required=True)
    run_p.add_argument("--shards", type=int, required=True)
    run_p.add_argument("--profile", default="depth4", choices=sorted(PROFILES))
    verdict_p = sub.add_parser("verdict", help="aggregate shards and score the contract")
    verdict_p.add_argument("--profile", default="depth4", choices=sorted(PROFILES))
    sum_p = sub.add_parser("summary", help="write the committed study summary")
    sum_p.add_argument("--profile", default="depth4", choices=sorted(PROFILES))
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
