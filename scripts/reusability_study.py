"""Section 5.16 study: is the discovered topology a reusable artifact?

Pre-registered in ``RESEARCH_REPORT.md`` section 5.16. Per seed, one three-arm
discovery block (``arms="random,adaptive,fixed"``, the section 5.6/5.10
registered config: depth 4, ``probe_candidate_policy="spread"``, cap-4/2-batch
probes, 48 steps, interval 12, budget 3, ``independent_arm_rng=True``, cpu),
then each arm's final genome is transported: rebuilt with fresh random weights
(``initialize_retraining(mode="fresh")``) and retrained for the same 48 steps
on the same per-seed data schedule with mutation disabled. The topology is
frozen, so the genome - not the controller - is what is under test.

The arm order keeps ``random``/``adaptive`` at indices 0/1 exactly as the
section 5.10 block did, so their discovery runs are the registered ones; the
``fixed`` arm (which never mutates) is appended last and consumes no global
torch RNG, so it cannot perturb them (pinned by test).

Primaries (exact one-tailed sign test, Bonferroni alpha/2 = 0.0025, n = 384):

1. reusability    - transported adaptive topology beats the transported random
                    topology on final held-out loss.
2. over-doing-nothing - transported adaptive topology beats the transported
                    fixed (initial, never-mutated) topology.

Usage (one shard per worker, then the verdict):

    PYTHONPATH=src python scripts/reusability_study.py run --shard 0 --shards 20
    ...
    PYTHONPATH=src python scripts/reusability_study.py verdict --shards 20
    PYTHONPATH=src python scripts/reusability_study.py summary
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

SHARD_DIR = ROOT / "artifacts" / "toy-reusability-study"
PROFILES: dict[str, dict] = {
    # The registered block. Fresh: used by no other study.
    "depth4": {
        "seeds": tuple(range(81000, 81384)),
        "report_section": "5.16",
        "summary": "artifacts/study-summaries/reusability-depth4-384.json",
    },
    # Machinery smoke test only: a spent block (section 5.12's depth4-prefix
    # cells) so a dry run cannot contaminate the registered verdict block.
    "smoke": {
        "seeds": tuple(range(73000, 73004)),
        "report_section": "5.16 (smoke)",
        "summary": "artifacts/study-summaries/reusability-smoke.json",
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
TRANSPORTED = ("random", "adaptive", "fixed")
ALPHA = 0.05
# The pre-registered per-primary threshold, taken VERBATIM from section 5.16:
# "Bonferroni alpha/2 = 0.0025 ... bar 214/384". That parenthetical is
# internally inconsistent - the exact binomial bar at 0.0025 is 220, at the
# sibling studies' 0.025 it is 212, and 214 matches no threshold at all (it
# sits at alpha 0.0140) - and both numbers are in the STRICT direction. The
# declaration is what binds, so the registered reading is the declared
# threshold 0.0025 (bar 220), and the sibling convention (0.025, bar 212) is
# recorded alongside it rather than substituted for it, because swapping in
# the looser bar after seeing 212 wins would be exactly the post-hoc
# flexibility pre-registration exists to prevent.
PRIMARY_ALPHA = 0.0025
SIBLING_ALPHA = ALPHA / 2  # 0.025: what sections 5.9/5.10/5.12/5.15 used


def _run_seed(seed: int) -> dict:
    """One discovery block plus one fresh-init transport per discovered genome."""
    config = ExperimentConfig(**REGISTERED, seed=seed, arms=ARMS)
    discovery = run_experiment(config)
    arms = {arm.mode: arm for arm in discovery.arms}
    transport = {
        mode: transport_genome(ArchitectureGenome.from_dict(arms[mode].genome), config)
        for mode in TRANSPORTED
    }
    return {
        "seed": seed,
        "discovery": {
            "losses": {mode: arms[mode].final_loss for mode in TRANSPORTED},
            "mutations": {mode: arms[mode].mutation_count for mode in TRANSPORTED},
            "parameter_count": {mode: arms[mode].parameter_count for mode in TRANSPORTED},
            "active_parameter_count": {
                mode: arms[mode].active_parameter_count for mode in TRANSPORTED
            },
            "architecture_hash": {
                mode: arms[mode].architecture_hash for mode in TRANSPORTED
            },
            "training_flops": {mode: arms[mode].training_flops for mode in TRANSPORTED},
        },
        "transport": {
            "losses": {mode: transport[mode].final_loss for mode in TRANSPORTED},
            "parameter_count": {
                mode: transport[mode].parameter_count for mode in TRANSPORTED
            },
            "active_parameter_count": {
                mode: transport[mode].active_parameter_count for mode in TRANSPORTED
            },
            "training_flops": {
                mode: transport[mode].training_flops for mode in TRANSPORTED
            },
            "initialization_modes": {
                mode: transport[mode].initialization.get("mode") for mode in TRANSPORTED
            },
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
    seeds = PROFILES[profile]["seeds"][shard::shards]
    experiments = [_run_seed(seed) for seed in seeds]
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


def _comparison(deltas: list[float], alpha: float = PRIMARY_ALPHA) -> dict:
    """One pre-registered primary: sign-test statistics at the declared bar.

    ``alpha`` is the per-primary threshold (the section 5.16 declaration); the
    sibling studies' looser bar is reported beside it, never instead of it.
    """
    wins = sum(delta < 0 for delta in deltas)
    n = len(deltas)
    mean = sum(deltas) / n
    stddev = (
        (sum((value - mean) ** 2 for value in deltas) / n) ** 0.5 if n > 1 else 0.0
    )
    p = _binomial_survival(wins, n)
    return {
        "wins": wins,
        "n": n,
        "mean_delta": mean,
        "delta_stddev": stddev,
        "binomial_p": p,
        "min_wins_bonferroni": _gate_min_wins(n, alpha),
        "passes_bonferroni": p <= alpha,
        "min_wins_sibling_bonferroni": _gate_min_wins(n, SIBLING_ALPHA),
        "passes_sibling_bonferroni": p <= SIBLING_ALPHA,
    }


VALIDITY_TOLERANCE = 1e-6


def _transport_validity(experiments: list[dict]) -> dict:
    """How exactly does the transport reproduce a known arm?

    The transported *fixed* genome must reproduce that arm's own discovery run:
    the fixed arm never mutates, so its genome is the block's initial
    architecture and transporting it retrains the same architecture on the same
    schedule with the same seed. Its error is the noise floor for every delta
    in this study.
    """
    deltas = [
        abs(exp["transport"]["losses"]["fixed"] - exp["discovery"]["losses"]["fixed"])
        for exp in experiments
    ]
    return {
        "max_abs_delta": max(deltas),
        "exact_reproductions": sum(1 for delta in deltas if delta == 0.0),
        "mean_abs_delta": sum(deltas) / len(deltas),
        "n": len(deltas),
    }


def _built_in_checks(experiments: list[dict]) -> dict:
    """Pre-conditions that must hold before the primaries can be read.

    A failure is a machinery failure: verdict withheld. Transport validity is
    the load-bearing one - without it a delta cannot be attributed to the
    topology - and zero mutations during transport is what makes the genome,
    rather than the controller, the thing under test.
    """
    validity = _transport_validity(experiments)
    return {
        "transport_validity_within_tolerance": (
            validity["max_abs_delta"] <= VALIDITY_TOLERANCE
        ),
        "transport_training_flops_match_fixed": all(
            exp["transport"]["training_flops"]["fixed"]
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
        "transport_uses_fresh_initialization": all(
            mode == "fresh"
            for exp in experiments
            for mode in exp["transport"]["initialization_modes"].values()
        ),
    }


def _capacity_view(experiments: list[dict], other: str) -> dict:
    """The primary restricted to seeds where the two topologies match in size.

    Section 0's hypothesis is at equivalent capacity, so a structural win at a
    higher parameter count is capacity-confounded. This re-runs the paired
    comparison on the matched subset and reports how the sizes actually differ.
    """
    matched = [
        (exp["transport"]["losses"]["adaptive"], exp["transport"]["losses"][other])
        for exp in experiments
        if exp["transport"]["active_parameter_count"]["adaptive"]
        == exp["transport"]["active_parameter_count"][other]
    ]
    deltas = [a - b for a, b in matched]
    higher = sum(
        exp["transport"]["active_parameter_count"]["adaptive"]
        > exp["transport"]["active_parameter_count"][other]
        for exp in experiments
    )
    lower = sum(
        exp["transport"]["active_parameter_count"]["adaptive"]
        < exp["transport"]["active_parameter_count"][other]
        for exp in experiments
    )
    return {
        "matched_seeds": len(deltas),
        "matched": _comparison(deltas) if deltas else None,
        "adaptive_active_params_higher_seeds": higher,
        "adaptive_active_params_lower_seeds": lower,
        "n": len(experiments),
    }


def summarize(profile: str, alpha: float = PRIMARY_ALPHA) -> dict:
    """Build the committed study summary from this profile's shard payloads.

    Self-contained: the per-seed discovery and transport losses let every
    quoted statistic be recomputed exactly, so the report numbers stay
    machine-checked against ``artifacts/study-summaries/`` without committing
    the (gitignored) shard payloads.
    """
    spec = PROFILES[profile]
    experiments = load_shards(_shards_for(profile), profile)
    n = len(experiments)
    checks = _built_in_checks(experiments)
    if not all(checks.values()):
        raise SystemExit(f"built-in checks failed: {checks}")

    comparisons = {
        "transport_adaptive_vs_transport_random": _comparison(
            [
                exp["transport"]["losses"]["adaptive"]
                - exp["transport"]["losses"]["random"]
                for exp in experiments
            ],
            alpha,
        ),
        "transport_adaptive_vs_transport_fixed": _comparison(
            [
                exp["transport"]["losses"]["adaptive"]
                - exp["transport"]["losses"]["fixed"]
                for exp in experiments
            ],
            alpha,
        ),
    }
    discovery_means = {
        mode: sum(exp["discovery"]["losses"][mode] for exp in experiments) / n
        for mode in TRANSPORTED
    }
    transport_means = {
        mode: sum(exp["transport"]["losses"][mode] for exp in experiments) / n
        for mode in TRANSPORTED
    }
    validity = _transport_validity(experiments)
    noise_floor = validity["max_abs_delta"]
    inconclusive = [
        name
        for name, comparison in comparisons.items()
        if abs(comparison["mean_delta"]) <= noise_floor
    ]
    p1 = comparisons["transport_adaptive_vs_transport_random"]["passes_bonferroni"]
    p2 = comparisons["transport_adaptive_vs_transport_fixed"]["passes_bonferroni"]
    q1 = comparisons["transport_adaptive_vs_transport_random"][
        "passes_sibling_bonferroni"
    ]
    q2 = comparisons["transport_adaptive_vs_transport_fixed"]["passes_sibling_bonferroni"]
    if q1 and q2:
        outcome_under_sibling_bar = "branch (a) both pass under the sibling bar"
    elif q1:
        outcome_under_sibling_bar = (
            "branch (b) primary 1 passes / primary 2 fails under the sibling bar"
        )
    else:
        outcome_under_sibling_bar = "both fail under the sibling bar too"
    if inconclusive:
        outcome = (
            "INCONCLUSIVE: primary margin inside the transport noise floor "
            f"({inconclusive})"
        )
    elif p1 and p2:
        outcome = (
            "branch (a) both PASS: the discovered topology is a reusable "
            "artifact at toy fidelity"
        )
    elif p1:
        outcome = (
            "branch (b) 1 PASS / 2 FAIL: the adaptive topology beats random "
            "churn's structure but not the never-mutated one - it preserves "
            "rather than creates"
        )
    elif p2:
        outcome = (
            "branch 1 FAIL / 2 PASS: beats the initial topology but not "
            "random churn's structure"
        )
    else:
        outcome = (
            "branch (c) both FAIL: the adaptive advantage lives in the weights "
            "and/or initialization, not in the structure it chose"
        )
    payload = {
        "record_type": "reusability_study",
        "protocol": "discover-then-fresh-init-transport",
        "profile": profile,
        "report_section": spec["report_section"],
        "alpha": alpha,
        "bonferroni_alpha": PRIMARY_ALPHA,
        "sibling_bonferroni_alpha": SIBLING_ALPHA,
        "arms": ["random", "adaptive", "fixed"],
        "transported_arms": list(TRANSPORTED),
        "independent_arm_rng": True,
        "seeds": list(spec["seeds"]),
        "config": {**REGISTERED, "arms": ARMS},
        "per_seed": [
            {
                "seed": exp["seed"],
                "discovery_losses": exp["discovery"]["losses"],
                "transport_losses": exp["transport"]["losses"],
                "transport_active_parameter_count": exp["transport"][
                    "active_parameter_count"
                ],
                "transport_parameter_count": exp["transport"]["parameter_count"],
            }
            for exp in experiments
        ],
        "comparisons": comparisons,
        "capacity": {
            "adaptive_vs_random": _capacity_view(experiments, "random"),
            "adaptive_vs_fixed": _capacity_view(experiments, "fixed"),
        },
        "discovery_arm_means": discovery_means,
        "transport_arm_means": transport_means,
        "identity_deltas": {
            "transport_fixed_minus_discovery_fixed_mean": sum(
                exp["transport"]["losses"]["fixed"]
                - exp["discovery"]["losses"]["fixed"]
                for exp in experiments
            )
            / n,
            "transport_random_minus_discovery_random_mean": sum(
                exp["transport"]["losses"]["random"]
                - exp["discovery"]["losses"]["random"]
                for exp in experiments
            )
            / n,
            "transport_adaptive_minus_discovery_adaptive_mean": sum(
                exp["transport"]["losses"]["adaptive"]
                - exp["discovery"]["losses"]["adaptive"]
                for exp in experiments
            )
            / n,
        },
        "built_in_checks": checks,
        "transport_validity": validity,
        "outcome": outcome,
        "outcome_under_sibling_bar": outcome_under_sibling_bar,
        "summary_note": (
            "Fresh-initialization transport (report section 5.16, plan sections "
            "52/55): each arm's discovered genome is rebuilt with fresh random "
            "weights via initialize_retraining(mode='fresh') and retrained for "
            "the same 48 steps on the same per-seed data schedule with mutation "
            "disabled. Transport validity is the fixed genome's exact "
            "reproduction of its own discovery run, which is why the name "
            "identity delta is also reported. Statistics recompute exactly from "
            "per_seed; p-values are exact one-tailed binomial."
        ),
    }
    payload["study_hash"] = _stable_hash(
        {
            "config": payload["config"],
            "seeds": payload["seeds"],
            "per_seed": payload["per_seed"],
            "alpha": alpha,
        }
    )
    payload["study_id"] = f"study_{payload['study_hash']}"
    payload["config_hash"] = _stable_hash(payload["config"])
    return payload


def verdict(profile: str) -> int:
    payload = summarize(profile)
    checks = payload["built_in_checks"]
    n = len(payload["seeds"])
    validity = payload["transport_validity"]
    print(
        f"built-in check: transported fixed genome reproduces its discovery run "
        f"exactly in {validity['exact_reproductions']}/{n} seeds, "
        f"max |delta| {validity['max_abs_delta']:.2e}"
    )
    print("built-in check: transport uses fresh initialization, zero mutations: OK")
    print(f"\ndiscovery means: {payload['discovery_arm_means']}")
    print(f"transport means: {payload['transport_arm_means']}")
    for name, comparison in payload["comparisons"].items():
        print(
            f"\n{name}: wins {comparison['wins']}/{comparison['n']}, "
            f"mean {comparison['mean_delta']:+.6f}, p {comparison['binomial_p']:.2e}"
        )
        print(
            f"    registered bar {comparison['min_wins_bonferroni']} "
            f"(p <= {PRIMARY_ALPHA}): "
            f"{'PASS' if comparison['passes_bonferroni'] else 'FAIL'}"
            f"   | sibling studies' bar {comparison['min_wins_sibling_bonferroni']} "
            f"(p <= {SIBLING_ALPHA}): "
            f"{'pass' if comparison['passes_sibling_bonferroni'] else 'fail'}"
        )
    for name, view in payload["capacity"].items():
        matched = view["matched"]
        detail = (
            f", matched subset {matched['wins']}/{matched['n']} (bar "
            f"{matched['min_wins_bonferroni']})"
            if matched
            else ", no capacity-matched seeds"
        )
        print(
            f"capacity {name}: adaptive active params higher in "
            f"{view['adaptive_active_params_higher_seeds']}/{view['n']}, lower in "
            f"{view['adaptive_active_params_lower_seeds']}{detail}"
        )
    print(
        f"\nOUTCOME (registered per-primary threshold {PRIMARY_ALPHA}, "
        f"declared n={n}): {payload['outcome']}"
        f"\n  under the sibling studies' bar instead: "
        f"{payload['outcome_under_sibling_bar']}"
    )
    if payload["outcome"].startswith("INCONCLUSIVE"):
        return 2
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
