"""Section 5.23 study: is the mid-training penalty optimizer history or architecture?

Pre-registered in ``RESEARCH_REPORT.md`` section 5.23. Section 5.22 established
that a fused run loses ~0.1 nats to a frozen retrain of its own final genome, and
section 5.22.1 ruled the size of the final topology out as the explanation. The
mutation path re-creates the AdamW optimizer whenever it rebuilds the model
(``_mutate_by_probe`` and the random path both assign a fresh ``AdamW``), so every
event discards all moment estimates - that is the one confound left.

Per seed, on the same initial model and the same data schedule:

- ``F``  the ``fixed`` arm's own run - unmutated, no optimizer resets;
- ``U``  the SAME unmutated genome with optimizer resets at the mutation steps
         (``ExperimentConfig.optimizer_reset_steps``), so ``U`` differs from ``F``
         by optimizer history alone;
- ``R``  the fused ``random`` arm (mutations applied in place, resets included);
- ``A``  the fused ``adaptive`` arm, recorded for continuity.

Primaries per depth (n = 384), Bonferroni ``alpha/4 = 0.0125`` -> bar 215:

1. **reset cost** - ``F`` < ``U``: does discarding Adam's moments, with the
   architecture held fixed, cost anything?
2. **architecture cost beyond resets** - ``U`` < ``R``: is the fused run worse
   than a run that only resets? If yes the mid-flight rebuild itself costs.

Usage:

    PYTHONPATH=src python scripts/optimizer_reset_study.py run --shard 0 --shards 20 --profile reset2
    PYTHONPATH=src python scripts/optimizer_reset_study.py verdict --profile reset2
    PYTHONPATH=src python scripts/optimizer_reset_study.py summary --profile reset2
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
from act.evolution.study import _binomial_survival, _gate_min_wins  # noqa: E402

SHARD_DIR = ROOT / "artifacts" / "toy-reset-study"
PROFILES: dict[str, dict] = {
    "reset2": {
        "seeds": tuple(range(94000, 94384)),
        "overrides": {"depth": 2},
        "report_section": "5.23",
        "summary": "artifacts/study-summaries/optimizer-reset-depth2-384.json",
    },
    "reset3": {
        "seeds": tuple(range(95000, 95384)),
        "overrides": {"depth": 3},
        "report_section": "5.23",
        "summary": "artifacts/study-summaries/optimizer-reset-depth3-384.json",
    },
    "smoke": {
        "seeds": tuple(range(73000, 73004)),
        "overrides": {"depth": 4},
        "declared_n": 4,
        "report_section": "5.23 (smoke)",
        "summary": "artifacts/study-summaries/optimizer-reset-smoke.json",
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


def _reset_steps(profile: str) -> tuple[int, ...]:
    """The exact steps the fused arms reset at: interval x 1..budget."""
    config = {**REGISTERED, **PROFILES[profile]["overrides"]}
    return tuple(
        config["mutation_interval"] * (index + 1)
        for index in range(config["mutation_budget"])
    )


def _run_seed(seed: int, profile: str) -> dict:
    values = {**REGISTERED, **PROFILES[profile]["overrides"]}
    fused = {
        arm.mode: arm
        for arm in run_experiment(
            ExperimentConfig(**{**values, "seed": seed, "arms": "fixed,random,adaptive"})
        ).arms
    }
    reset = run_experiment(
        ExperimentConfig(
            **values,
            seed=seed,
            arms="fixed",
            optimizer_reset_steps=_reset_steps(profile),
        )
    ).arms[0]
    if reset.mode != "fixed":
        raise RuntimeError(f"reset twin came back as {reset.mode!r}")
    random_arm = fused["random"]
    return {
        "seed": seed,
        "reset_steps": list(_reset_steps(profile)),
        "losses": {
            "F": fused["fixed"].final_loss,
            "U": reset.final_loss,
            "R": random_arm.final_loss,
            "A": fused["adaptive"].final_loss,
        },
        "mutation_count": {
            "F": fused["fixed"].mutation_count,
            "U": reset.mutation_count,
            "R": random_arm.mutation_count,
            "A": fused["adaptive"].mutation_count,
        },
        "training_flops": {
            "F": fused["fixed"].training_flops,
            "U": reset.training_flops,
            "R": random_arm.training_flops,
            "A": fused["adaptive"].training_flops,
        },
        # ``_stable_hash`` over the reported genome dicts: the twin's must equal
        # the fixed arm's exactly, which is what makes it a clean control.
        "genome_hash": {
            "F": _stable_hash(fused["fixed"].genome),
            "U": _stable_hash(reset.genome),
            "R": _stable_hash(random_arm.genome),
            "A": _stable_hash(fused["adaptive"].genome),
        },
        "active_parameter_count": {
            "F": fused["fixed"].active_parameter_count,
            "U": reset.active_parameter_count,
            "R": random_arm.active_parameter_count,
            "A": fused["adaptive"].active_parameter_count,
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
    mean = sum(deltas) / n
    stddev = (sum((value - mean) ** 2 for value in deltas) / n) ** 0.5 if n > 1 else 0.0
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


def _built_in_checks(experiments: list[dict], profile: str) -> dict:
    return {
        "declared_bar_matches_exact_binomial": _gate_min_wins(
            DECLARED_N, PER_PRIMARY_ALPHA
        )
        == DECLARED_BAR,
        "block_matches_declared_n": len(PROFILES[profile]["seeds"])
        == PROFILES[profile].get("declared_n", DECLARED_N),
        # The twin is a clean control only if the architecture never moved.
        "reset_twin_genome_is_the_fixed_genome": all(
            exp["genome_hash"]["U"] == exp["genome_hash"]["F"] for exp in experiments
        ),
        "reset_twin_never_mutated": all(
            exp["mutation_count"]["U"] == 0 for exp in experiments
        ),
        "fixed_never_mutated": all(
            exp["mutation_count"]["F"] == 0 for exp in experiments
        ),
        # A reset is not a training event, so the booked work must be identical.
        "reset_twin_books_the_same_training_flops": all(
            exp["training_flops"]["U"] == exp["training_flops"]["F"]
            for exp in experiments
        ),
        "fused_arms_used_the_full_budget": all(
            exp["mutation_count"]["R"] == len(exp["reset_steps"])
            and exp["mutation_count"]["A"] == len(exp["reset_steps"])
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
        # Primary 1: does optimizer history alone cost anything?
        "F_vs_U": _comparison(delta("F", "U")),
        # Primary 2: is the fused run worse than a resets-only run?
        "U_vs_R": _comparison(delta("U", "R")),
        # Reported context: the whole penalty on this block, per section 5.22.
        "U_vs_A": _comparison(delta("U", "A")),
    }
    reset_cost = comparisons["F_vs_U"]["mean_delta"]
    total = comparisons["U_vs_R"]["mean_delta"]
    payload = {
        "record_type": "optimizer_reset_study",
        "protocol": "optimizer-history-vs-architecture-decomposition",
        "profile": profile,
        "report_section": spec["report_section"],
        "alpha": ALPHA,
        "per_primary_alpha": PER_PRIMARY_ALPHA,
        "declared_bar": DECLARED_BAR,
        "depth": spec["overrides"]["depth"],
        "reset_steps": list(_reset_steps(profile)),
        "seeds": list(spec["seeds"]),
        "config": {**REGISTERED, **spec["overrides"]},
        "per_seed": experiments,
        "comparisons": comparisons,
        "loss_means": {label: sum(losses(label)) / n for label in ("F", "U", "R", "A")},
        "reset_cost": -reset_cost,
        "architecture_cost_beyond_resets": -total,
        "built_in_checks": checks,
        "summary_note": (
            "Section 5.23 decomposition. F and U differ by optimizer resets alone "
            "(same seed, same initial model, same data schedule, same genome - "
            "pinned by a built-in check), so F_vs_U measures the cost of "
            "discarding Adam's moments. U and R differ by whether the architecture "
            "changed mid-flight, so U_vs_R measures the cost of the rebuild itself. "
            "Negative means the first-named run won."
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
    print(f"reset steps: {payload['reset_steps']}")
    for name, comparison in payload["comparisons"].items():
        print(
            f"\n{name}: wins {comparison['wins']}/{comparison['n']}, "
            f"mean {comparison['mean_delta']:+.6f}, p {comparison['binomial_p']:.2e} "
            f"(bar {comparison['min_wins']}): "
            f"{'PASS' if comparison['passes'] else 'FAIL'}"
        )
    p1 = payload["comparisons"]["F_vs_U"]["passes"]
    p2 = payload["comparisons"]["U_vs_R"]["passes"]
    if not p1:
        branch = (
            "branch (c) resets cost nothing: the penalty is the ARCHITECTURE "
            "change, two-stage delivery stands unqualified"
        )
    elif p2:
        branch = (
            "branch (a) both components: resets AND the rebuild cost - "
            "optimizer-preserving mutation is a partial fix worth its own study"
        )
    else:
        branch = (
            "branch (b) the penalty is essentially all optimizer reset - "
            "the cheap fix is available and 5.22's change is qualified"
        )
    print(f"\nOUTCOME: {branch}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run one shard")
    run_p.add_argument("--shard", type=int, required=True)
    run_p.add_argument("--shards", type=int, required=True)
    run_p.add_argument("--profile", default="reset2", choices=sorted(PROFILES))
    verdict_p = sub.add_parser("verdict", help="aggregate shards")
    verdict_p.add_argument("--profile", default="reset2", choices=sorted(PROFILES))
    sum_p = sub.add_parser("summary", help="write the committed study summary")
    sum_p.add_argument("--profile", default="reset2", choices=sorted(PROFILES))
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
