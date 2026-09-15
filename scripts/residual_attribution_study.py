"""Section 5.25 study: is the residual penalty a pruned expert or a grown one?

Pre-registered in ``RESEARCH_REPORT.md`` section 5.25. Section 5.24 removed the
discarded routing head, which accounted for just over half the mid-training
penalty, and left a residual of ``~0.044``/``~0.058``. Two candidates remain and a
subgroup split cannot separate them, because the operation mix was itself chosen
by the same coin flip the subgroups would be defined by. So the operation is
**forced** instead: every event is a prune in one profile and a grow in the other,
each compared against a frozen retrain of its own final genome.

Profiles (each its own fresh block, ``preserve_router_on_rebuild=True``):

- ``prune``: 5 experts/layer, all three events prunes -> the lost-capacity
  candidate. 5 -> 2 is the worst case for a single lane and stays at the floor.
- ``grow``:  4 experts/layer, all three events grows -> the fresh-initialization
  candidate. The 8-expert ceiling leaves room for three grows on one lane.

Per seed, ``arms="fixed,random"``: ``F`` (validity), ``Rp`` (the forced fused run)
and ``Rp_own`` (the frozen retrain of ``Rp``'s final genome).

Primaries (n = 384 each, Bonferroni ``alpha/2 = 0.025`` -> bar 212):

1. **prune residual** - ``Rp_own`` < ``Rp`` on the prune block.
2. **grow residual** - ``Rp_own`` < ``Rp`` on the grow block.

Usage:

    PYTHONPATH=src python scripts/residual_attribution_study.py run --shard 0 --shards 20 --profile prune
    PYTHONPATH=src python scripts/residual_attribution_study.py verdict --profile prune
    PYTHONPATH=src python scripts/residual_attribution_study.py summary --profile prune
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

SHARD_DIR = ROOT / "artifacts" / "toy-residual-study"
PROFILES: dict[str, dict] = {
    "prune": {
        "seeds": tuple(range(98000, 98384)),
        "overrides": {
            "depth": 2,
            "expert_count": 5,
            "forced_operation": "prune",
            "preserve_router_on_rebuild": True,
        },
        "report_section": "5.25",
        "summary": "artifacts/study-summaries/residual-prune-384.json",
    },
    "grow": {
        "seeds": tuple(range(99000, 99384)),
        "overrides": {
            "depth": 2,
            "expert_count": 4,
            "forced_operation": "grow",
            "preserve_router_on_rebuild": True,
        },
        "report_section": "5.25",
        "summary": "artifacts/study-summaries/residual-grow-384.json",
    },
    "smoke": {
        "seeds": tuple(range(73000, 73004)),
        "overrides": {
            "depth": 2,
            "expert_count": 5,
            "forced_operation": "prune",
            "preserve_router_on_rebuild": True,
        },
        "declared_n": 4,
        "report_section": "5.25 (smoke)",
        "summary": "artifacts/study-summaries/residual-smoke.json",
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
PER_PRIMARY_ALPHA = ALPHA / 2  # one primary per block, two blocks
DECLARED_BAR = 212


def _run_seed(seed: int, profile: str) -> dict:
    values = {**REGISTERED, **PROFILES[profile]["overrides"]}
    config = ExperimentConfig(**{**values, "seed": seed, "arms": "fixed,random"})
    arms = {arm.mode: arm for arm in run_experiment(config).arms}
    fused = arms["random"]
    fixed = arms["fixed"]

    def experts(genome: dict) -> list[int]:
        return [len(layer["experts"]) for layer in genome["layers"]]

    return {
        "seed": seed,
        "losses": {
            "F": fixed.final_loss,
            "Rp": fused.final_loss,
            "Rp_own": transport_genome(
                ArchitectureGenome.from_dict(fused.genome),
                config,
                initialization="rewound",
            ).final_loss,
        },
        "fixed_transport_loss": transport_genome(
            ArchitectureGenome.from_dict(fixed.genome),
            config,
            initialization="rewound",
        ).final_loss,
        "mutations": fused.mutation_count,
        "experts": {
            "initial_per_layer": values["expert_count"],
            "final": experts(fused.genome),
        },
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
    start_count = spec["overrides"]["expert_count"]
    operation = spec["overrides"]["forced_operation"]
    sign = -1 if operation == "prune" else 1
    return {
        "declared_bar_matches_exact_binomial": _gate_min_wins(
            DECLARED_N, PER_PRIMARY_ALPHA
        )
        == DECLARED_BAR,
        "block_matches_declared_n": len(spec["seeds"])
        == spec.get("declared_n", DECLARED_N),
        "fixed_transport_reproduces_the_fixed_arm": all(
            exp["fixed_transport_loss"] == exp["losses"]["F"] for exp in experiments
        ),
        # The whole point of forcing: every event must be the forced operation,
        # so the total expert count must move by exactly the budget in one
        # direction. A fallback substitution would break this.
        "every_event_was_the_forced_operation": all(
            sum(exp["experts"]["final"])
            == start_count * spec["overrides"]["depth"] + sign * 3
            for exp in experiments
        ),
        "every_seed_used_the_full_budget": all(
            exp["mutations"] == 3 for exp in experiments
        ),
        "no_layer_went_below_two_experts": all(
            min(exp["experts"]["final"]) >= 2 for exp in experiments
        ),
    }


def summarize(profile: str) -> dict:
    spec = PROFILES[profile]
    experiments = load_shards(_shards_for(profile), profile)
    n = len(experiments)
    checks = _built_in_checks(experiments, profile)
    if not all(checks.values()):
        raise SystemExit(f"built-in checks failed: {checks}")

    losses = {
        label: [exp["losses"][label] for exp in experiments]
        for label in ("F", "Rp", "Rp_own")
    }
    comparison = _comparison(
        [y - z for y, z in zip(losses["Rp_own"], losses["Rp"])]
    )
    payload = {
        "record_type": "residual_attribution_study",
        "protocol": "forced-operation-residual-attribution",
        "profile": profile,
        "operation": spec["overrides"]["forced_operation"],
        "report_section": spec["report_section"],
        "alpha": ALPHA,
        "per_primary_alpha": PER_PRIMARY_ALPHA,
        "declared_bar": DECLARED_BAR,
        "seeds": list(spec["seeds"]),
        "config": {**REGISTERED, **spec["overrides"]},
        "per_seed": experiments,
        "comparisons": {"Rp_own_vs_Rp": comparison},
        "loss_means": {label: sum(values) / n for label, values in losses.items()},
        "residual_penalty": comparison["mean_delta"],
        "built_in_checks": checks,
        "summary_note": (
            "Section 5.25. Every mutation event is forced to one operation, so the "
            "residual penalty of section 5.24 is attributed to a pruned expert's "
            "lost capacity (prune block) or a grown expert's fresh initialization "
            "(grow block) rather than to their mix. Rp_own - Rp is the residual; "
            "negative means the frozen retrain won."
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
    comparison = payload["comparisons"]["Rp_own_vs_Rp"]
    print(f"loss means: {payload['loss_means']}")
    print(
        f"\nRp_own_vs_Rp: wins {comparison['wins']}/{comparison['n']}, "
        f"mean {comparison['mean_delta']:+.6f}, p {comparison['binomial_p']:.2e} "
        f"(bar {comparison['min_wins']}): "
        f"{'PASS' if comparison['passes'] else 'FAIL'}"
    )
    print(
        f"residual penalty on the {payload['operation']} block: "
        f"{payload['residual_penalty']:+.4f}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run one shard")
    run_p.add_argument("--shard", type=int, required=True)
    run_p.add_argument("--shards", type=int, required=True)
    run_p.add_argument("--profile", default="prune", choices=sorted(PROFILES))
    verdict_p = sub.add_parser("verdict", help="aggregate shards")
    verdict_p.add_argument("--profile", default="prune", choices=sorted(PROFILES))
    sum_p = sub.add_parser("summary", help="write the committed study summary")
    sum_p.add_argument("--profile", default="prune", choices=sorted(PROFILES))
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
