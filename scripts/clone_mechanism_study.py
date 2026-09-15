"""Section 5.27 study: is a cloned expert's benefit its body or its routing row?

Pre-registered in ``RESEARCH_REPORT.md`` section 5.27. Section 5.26 showed that
cloning the layer's busiest survivor into a grown expert does not merely remove
the mid-training penalty but **inverts** it - the clone-grow fused run beat the
unmutated arm, the frozen retrain of its own final genome and the historical
random-grow run. It also declared the mechanism unresolved, because cloning does
two things at once: it gives the new expert a trained **body**, and it copies the
survivor's **router row**, so the router already sends the twin pair the same
tokens.

``new_expert_initialization="clone_body"`` is identical to ``"clone"`` in every
respect except that the new expert keeps its freshly drawn router row. Comparing
the three settings on one seed, one initial model, one data schedule, every event
forced to a grow, with ``preserve_router_on_rebuild=True``:

- ``F``      the fixed arm (unmutated) - identical under all three settings;
- ``R``      fused grow run, new expert freshly drawn (historical);
- ``B``      fused grow run, body cloned, **fresh router row**;
- ``C``      fused grow run, body cloned, **router row cloned** (= section 5.26);
- ``*_own``  the frozen retrain of each fused run's own final genome.

The forced grow picks its layer from the seed's own RNG and cloning consumes no
randomness, so all three settings choose the **same final genome** and all three
frozen references are bit-identical - asserted, not assumed.

Primaries per depth (n = 384), Bonferroni ``alpha/4 = 0.0125`` -> bar 215:

1. **the body helps** - ``B`` < ``R``: does a trained body alone, with no routing
   affinity conferred, beat a fresh draw?
2. **routing affinity adds** - ``C`` < ``B``: does copying the router row on top
   of that body add anything?

Usage:

    PYTHONPATH=src python scripts/clone_mechanism_study.py run --shard 0 --shards 20 --profile mech2
    PYTHONPATH=src python scripts/clone_mechanism_study.py verdict --profile mech2
    PYTHONPATH=src python scripts/clone_mechanism_study.py summary --profile mech2
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

SHARD_DIR = ROOT / "artifacts" / "toy-clone-mechanism-study"
PROFILES: dict[str, dict] = {
    "mech2": {
        "seeds": tuple(range(105000, 105384)),
        "overrides": {"depth": 2},
        "report_section": "5.27",
        "summary": "artifacts/study-summaries/clone-mechanism-depth2-384.json",
    },
    "mech3": {
        "seeds": tuple(range(106000, 106384)),
        "overrides": {"depth": 3},
        "report_section": "5.27",
        "summary": "artifacts/study-summaries/clone-mechanism-depth3-384.json",
    },
    "smoke": {
        "seeds": tuple(range(73000, 73004)),
        "overrides": {"depth": 2},
        "declared_n": 4,
        "report_section": "5.27 (smoke)",
        "summary": "artifacts/study-summaries/clone-mechanism-smoke.json",
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
    # CPU on purpose, and measured rather than assumed. This substrate is far too
    # small for the card to help: six training runs per seed of a ~50k-parameter
    # model, each 48 steps, so every op is a kernel launch with the GPU idle in
    # between. Timed on this machine at the mech2 shape, CUDA is 2.6x SLOWER than
    # CPU (5.26 vs 2.04 s/seed). CPU is also what every committed number in
    # sections 5.16-5.26 was measured on, and CUDA's reduction order and RNG
    # streams differ, so moving a registered block to the GPU would silently stop
    # the record reproducing. The real-data legs (scripts/train_real_lm.py,
    # 20.9M params) do use the GPU, where it is worth ~18k tokens/s.
    device="cpu",
    forced_operation="grow",
    preserve_router_on_rebuild=True,
)
# label -> new_expert_initialization
SETTINGS = (("R", "random"), ("B", "clone_body"), ("C", "clone"))
ALPHA = 0.05
DECLARED_N = 384
PER_PRIMARY_ALPHA = ALPHA / 4  # two primaries, two depths
DECLARED_BAR = 215


def _run_seed(seed: int, profile: str) -> dict:
    values = {**REGISTERED, **PROFILES[profile]["overrides"]}
    outcome: dict[str, dict] = {}
    for label, initialization in SETTINGS:
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
            "fused_genome": fused.genome,
            "fused_loss": fused.final_loss,
            "fused_mutations": fused.mutation_count,
            "own_transport_loss": transport_genome(
                ArchitectureGenome.from_dict(fused.genome),
                config,
                initialization="rewound",
            ).final_loss,
        }
    losses = {"F": outcome["R"]["fixed_loss"]}
    for label, _ in SETTINGS:
        losses[label] = outcome[label]["fused_loss"]
        losses[f"{label}_own"] = outcome[label]["own_transport_loss"]
    genomes = {label: _stable_hash(outcome[label]["fused_genome"]) for label, _ in SETTINGS}
    owns = {label: outcome[label]["own_transport_loss"] for label, _ in SETTINGS}
    return {
        "seed": seed,
        "losses": losses,
        "fixed_arm_identical_across_settings": len(
            {outcome[label]["fixed_loss"] for label, _ in SETTINGS}
        )
        == 1,
        "same_final_genome": len(set(genomes.values())) == 1,
        "same_own_reference": len(set(owns.values())) == 1,
        "mutations": {label: outcome[label]["fused_mutations"] for label, _ in SETTINGS},
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
    return {
        "declared_bar_matches_exact_binomial": _gate_min_wins(
            DECLARED_N, PER_PRIMARY_ALPHA
        )
        == DECLARED_BAR,
        "block_matches_declared_n": len(spec["seeds"])
        == spec.get("declared_n", DECLARED_N),
        "the_fixed_arm_is_identical_across_all_three_settings": all(
            exp["fixed_arm_identical_across_settings"] for exp in experiments
        ),
        "all_three_settings_used_the_full_budget": all(
            all(count == 3 for count in exp["mutations"].values())
            for exp in experiments
        ),
        # The pairing claim, checked rather than assumed: if the three settings
        # diverged, P1 and P2 would be comparing different genomes.
        "all_three_settings_choose_the_same_final_genome": all(
            exp["same_final_genome"] for exp in experiments
        ),
        "the_frozen_reference_is_identical_under_all_three_settings": all(
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
        # Primary 1: does a trained body alone beat a fresh draw?
        "B_vs_R": _comparison(delta("B", "R")),
        # Primary 2: does copying the router row on top of that body add anything?
        "C_vs_B": _comparison(delta("C", "B")),
        # Context: section 5.26's whole-clone gain on this block, and the fresh
        # draw's penalty, for continuity.
        "C_vs_R": _comparison(delta("C", "R")),
        "R_own_vs_R": _comparison(delta("R_own", "R")),
    }
    p1 = comparisons["B_vs_R"]["passes"]
    p2 = comparisons["C_vs_B"]["passes"]
    if p1 and p2:
        outcome = (
            "branch (a) both contribute: a trained body and the copied routing "
            "affinity each buy something, so cloning is two effects and neither "
            "is the whole of section 5.26"
        )
    elif p1 and not p2:
        outcome = (
            "branch (b) the trained body is the whole effect: copying the router "
            "row adds nothing, so the minimal change is body cloning alone"
        )
    elif p2 and not p1:
        outcome = (
            "branch (c) routing affinity is the whole effect: a cloned body with "
            "a fresh row does not help, so the row must be copied for cloning to "
            "work"
        )
    else:
        outcome = (
            "branch (d) neither passes: the three-way split is inconclusive, "
            "recorded without a reading"
        )
    payload = {
        "record_type": "clone_mechanism_study",
        "protocol": "cloned-body-vs-cloned-router-row",
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
            for label in ("F", "R", "B", "C", "R_own", "B_own", "C_own")
        },
        "body_gain": comparisons["B_vs_R"]["mean_delta"],
        "row_gain": comparisons["C_vs_B"]["mean_delta"],
        "clone_gain": comparisons["C_vs_R"]["mean_delta"],
        "penalty_fresh": comparisons["R_own_vs_R"]["mean_delta"],
        "outcome": outcome,
        "built_in_checks": checks,
        "summary_note": (
            "Section 5.27. R, B and C are the same seed, initial model and data "
            "schedule, every event forced to a grow, differing only in whether a "
            "grown expert is freshly drawn (R), a clone of the layer's busiest "
            "survivor's BODY with a fresh router row (B), or that body with the "
            "survivor's router row copied too (C). The three settings choose the "
            "same final genome, so B-R is the trained body's contribution and "
            "C-B is the routing affinity's. Negative means the first-named run won."
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
        f"\nbody gain (B-R): {payload['body_gain']:+.4f}   "
        f"row gain (C-B): {payload['row_gain']:+.4f}   "
        f"total clone gain (C-R): {payload['clone_gain']:+.4f}"
    )
    print(f"\nOUTCOME: {payload['outcome']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run one shard")
    run_p.add_argument("--shard", type=int, required=True)
    run_p.add_argument("--shards", type=int, required=True)
    run_p.add_argument("--profile", default="mech2", choices=sorted(PROFILES))
    verdict_p = sub.add_parser("verdict", help="aggregate shards")
    verdict_p.add_argument("--profile", default="mech2", choices=sorted(PROFILES))
    sum_p = sub.add_parser("summary", help="write the committed study summary")
    sum_p.add_argument("--profile", default="mech2", choices=sorted(PROFILES))
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
