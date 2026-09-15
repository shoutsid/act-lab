"""Section 5.26 study: is the residual penalty the fresh random expert?

Pre-registered in ``RESEARCH_REPORT.md`` section 5.26. Section 5.25 forced every
event to a grow and showed a grow costs 1.6x a prune, so the residual mid-training
penalty of ``~0.044``/``~0.058`` that survived section 5.24 is attributed to the
one thing a grow alone introduces: a **freshly drawn expert**. Its body and its
router row are random noise, so the model must first spend gradient learning to
route around it (and the router must learn to send it tokens at all).

``new_expert_initialization="clone"`` gives a grown expert a working twin instead:
a copy of the busiest surviving expert in the same layer, by the model's own
recorded routing load, with that expert's router row copied too. On one initial
model and one data schedule, with ``preserve_router_on_rebuild=True`` and every
event forced to a grow, per seed:

- ``F``      the fixed arm (unmutated) - identical under both settings;
- ``R``      the fused grow run, new expert freshly drawn (historical);
- ``R_own``  ``R``'s final genome retrained from its own initialization;
- ``C``      the fused grow run, new expert cloned from the busiest survivor;
- ``C_own``  ``C``'s final genome retrained from its own initialization.

The forced operation makes the two settings choose the **same layer** (the
chooser consumes its own seeded RNG, not the weights) and cloning consumes no
randomness, so the two runs end on the same architecture: P1 is a paired contrast
on one genome, and ``C_own == R_own`` exactly.

Primaries per depth (n = 384), Bonferroni ``alpha/4 = 0.0125`` -> bar 215:

1. **cloning helps** - ``C`` < ``R``: does a grown expert that starts as a twin
   beat one that starts as noise, on the same seed and the same final genome?
2. **residual penalty** - ``C_own`` < ``C``: does a penalty remain once the new
   expert is cloned, or does cloning remove the mid-training penalty entirely?

Usage:

    PYTHONPATH=src python scripts/new_expert_init_study.py run --shard 0 --shards 20 --profile init2
    PYTHONPATH=src python scripts/new_expert_init_study.py verdict --profile init2
    PYTHONPATH=src python scripts/new_expert_init_study.py summary --profile init2
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

SHARD_DIR = ROOT / "artifacts" / "toy-new-expert-study"
PROFILES: dict[str, dict] = {
    "init2": {
        "seeds": tuple(range(103000, 103384)),
        "overrides": {"depth": 2},
        "report_section": "5.26",
        "summary": "artifacts/study-summaries/new-expert-init-depth2-384.json",
    },
    "init3": {
        "seeds": tuple(range(104000, 104384)),
        "overrides": {"depth": 3},
        "report_section": "5.26",
        "summary": "artifacts/study-summaries/new-expert-init-depth3-384.json",
    },
    "smoke": {
        "seeds": tuple(range(73000, 73004)),
        "overrides": {"depth": 2},
        "declared_n": 4,
        "report_section": "5.26 (smoke)",
        "summary": "artifacts/study-summaries/new-expert-init-smoke.json",
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
    # CPU on purpose: at this shape the card is launch-bound and slower (measured
    # 2.6x in section 5.27's config), and every committed number in sections
    # 5.16-5.26 was measured on CPU, so a device switch would stop the record
    # reproducing. The real-data legs use the GPU.
    device="cpu",
    # Every event is a grow: section 5.25 attributed the residual to the grow,
    # and a grow is the only operation that introduces a new expert at all.
    forced_operation="grow",
    preserve_router_on_rebuild=True,
)
ALPHA = 0.05
DECLARED_N = 384
PER_PRIMARY_ALPHA = ALPHA / 4  # two primaries, two depths
DECLARED_BAR = 215


def _run_seed(seed: int, profile: str) -> dict:
    values = {**REGISTERED, **PROFILES[profile]["overrides"]}
    outcome: dict[str, dict] = {}
    for label, initialization in (("random", "random"), ("clone", "clone")):
        config = ExperimentConfig(
            **{
                **values,
                "seed": seed,
                "arms": "fixed,random",
                "new_expert_initialization": initialization,
            }
        )
        arms = {arm.mode: arm for arm in run_experiment(config).arms}
        fused = arms["random"]
        outcome[label] = {
            "fixed_loss": arms["fixed"].final_loss,
            "fixed_genome": arms["fixed"].genome,
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
            "F": outcome["random"]["fixed_loss"],
            "R": outcome["random"]["fused_loss"],
            "R_own": outcome["random"]["own_transport_loss"],
            "C": outcome["clone"]["fused_loss"],
            "C_own": outcome["clone"]["own_transport_loss"],
        },
        "fixed_check": {
            "random_reproduces": outcome["random"]["fixed_transport_loss"]
            == outcome["random"]["fixed_loss"],
            "clone_reproduces": outcome["clone"]["fixed_transport_loss"]
            == outcome["clone"]["fixed_loss"],
            "cloning_leaves_the_fixed_arm_alone": outcome["random"]["fixed_loss"]
            == outcome["clone"]["fixed_loss"],
        },
        # The pairing claim: the two settings must make the SAME architectural
        # choices, so the contrast is initialization on one genome, not two.
        "same_final_genome": _stable_hash(outcome["random"]["fused_genome"])
        == _stable_hash(outcome["clone"]["fused_genome"]),
        "same_own_reference": outcome["random"]["own_transport_loss"]
        == outcome["clone"]["own_transport_loss"],
        "mutations": {
            "R": outcome["random"]["fused_mutations"],
            "C": outcome["clone"]["fused_mutations"],
        },
        "start_experts": _expert_total(outcome["random"]["fixed_genome"]),
        "end_experts": _expert_total(outcome["clone"]["fused_genome"]),
    }


def _expert_total(genome: dict) -> int:
    return sum(len(layer["experts"]) for layer in genome["layers"])


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
    spec = PROFILES[profile]
    start = spec["overrides"].get("expert_count", REGISTERED["expert_count"])
    depth = spec["overrides"]["depth"]
    return {
        "declared_bar_matches_exact_binomial": _gate_min_wins(
            DECLARED_N, PER_PRIMARY_ALPHA
        )
        == DECLARED_BAR,
        "block_matches_declared_n": len(spec["seeds"])
        == spec.get("declared_n", DECLARED_N),
        "fixed_transport_reproduces_the_fixed_arm_under_both_settings": all(
            exp["fixed_check"]["random_reproduces"]
            and exp["fixed_check"]["clone_reproduces"]
            for exp in experiments
        ),
        # The knob must not perturb a run that never mutates, or P1 and P2 would
        # be comparing two different baselines rather than one.
        "cloning_leaves_the_fixed_arm_alone": all(
            exp["fixed_check"]["cloning_leaves_the_fixed_arm_alone"]
            for exp in experiments
        ),
        "both_fused_runs_used_the_full_budget": all(
            exp["mutations"]["R"] == 3 and exp["mutations"]["C"] == 3
            for exp in experiments
        ),
        # Every event was a grow: the total expert count must rise by the budget.
        "every_event_was_a_grow": all(
            exp["end_experts"] - exp["start_experts"] == 3 for exp in experiments
        ),
        "block_started_at_the_declared_expert_total": all(
            exp["start_experts"] == start * depth for exp in experiments
        ),
        # The pairing claim, checked rather than assumed: if the two settings
        # diverged, P1 would be comparing two genomes and the frozen references
        # would differ, so the penalty framing would be meaningless.
        "both_settings_choose_the_same_final_genome": all(
            exp["same_final_genome"] for exp in experiments
        ),
        "the_frozen_reference_is_identical_under_both_settings": all(
            exp["same_own_reference"] for exp in experiments
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
        # Primary 1: does a cloned new expert beat a freshly drawn one?
        "C_vs_R": _comparison(delta("C", "R")),
        # Primary 2: does a penalty remain once the new expert is cloned?
        "C_own_vs_C": _comparison(delta("C_own", "C")),
        # Context: the random-init penalty on this block, for continuity with 5.25.
        "R_own_vs_R": _comparison(delta("R_own", "R")),
    }
    p1 = comparisons["C_vs_R"]["passes"]
    p2 = comparisons["C_own_vs_C"]["passes"]
    if p1 and not p2:
        outcome = (
            "branch (a) cloning helps and no penalty remains: the freshly drawn "
            "expert IS the residual mechanism - a grown expert should start as a "
            "twin, and section 5.22's two-stage delivery is qualified again"
        )
    elif p1 and p2:
        outcome = (
            "branch (b) partial: cloning helps but a residual penalty survives, "
            "so the mechanism is more than the fresh draw"
        )
    elif p2:
        outcome = (
            "branch (c) cloning does not help and the penalty remains: the fresh "
            "draw is NOT the mechanism - the cost of growing mid-run is intrinsic"
        )
    else:
        outcome = (
            "branch (d) cloning neither helps nor leaves a penalty: inconclusive, "
            "recorded without either reading"
        )
    payload = {
        "record_type": "new_expert_init_study",
        "protocol": "cloned-vs-fresh-grown-expert",
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
        "penalty_fresh": comparisons["R_own_vs_R"]["mean_delta"],
        "penalty_cloned": comparisons["C_own_vs_C"]["mean_delta"],
        "clone_gain": comparisons["C_vs_R"]["mean_delta"],
        "outcome": outcome,
        "built_in_checks": checks,
        "summary_note": (
            "Section 5.26. R and C are the same seed, initial model and data "
            "schedule, every event forced to a grow, differing only in whether a "
            "grown expert starts freshly drawn or as a clone of the layer's busiest "
            "survivor (body and router row). R_own and C_own are the frozen retrains "
            "of each fused run's own final genome, so R_own-R and C_own-C are the "
            "two penalties. The forced operation makes both settings choose the "
            "same layer, so the final genomes are identical and C_own == R_own "
            "exactly - both checked in built_in_checks, not assumed. Negative means "
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
    print(f"loss means: {payload['loss_means']}")
    for name, comparison in payload["comparisons"].items():
        print(
            f"\n{name}: wins {comparison['wins']}/{comparison['n']}, "
            f"mean {comparison['mean_delta']:+.6f}, p {comparison['binomial_p']:.2e} "
            f"(bar {comparison['min_wins']}): "
            f"{'PASS' if comparison['passes'] else 'FAIL'}"
        )
    print(
        f"\npenalty (fresh expert): {payload['penalty_fresh']:+.4f}   "
        f"penalty (cloned expert): {payload['penalty_cloned']:+.4f}   "
        f"gain from cloning: {payload['clone_gain']:+.4f}"
    )
    print(f"\nOUTCOME: {payload['outcome']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run one shard")
    run_p.add_argument("--shard", type=int, required=True)
    run_p.add_argument("--shards", type=int, required=True)
    run_p.add_argument("--profile", default="init2", choices=sorted(PROFILES))
    verdict_p = sub.add_parser("verdict", help="aggregate shards")
    verdict_p.add_argument("--profile", default="init2", choices=sorted(PROFILES))
    sum_p = sub.add_parser("summary", help="write the committed study summary")
    sum_p.add_argument("--profile", default="init2", choices=sorted(PROFILES))
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
