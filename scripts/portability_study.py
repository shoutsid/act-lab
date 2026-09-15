"""Section 5.18 study: is the genome portable, or paired with its own init?

Pre-registered in ``RESEARCH_REPORT.md`` section 5.18. Section 5.16 found the
discovered topology worthless once its weights were discarded; section 5.17
found it valuable when its survivors kept their *own* original initialization,
and worthless when they got a fresh draw. This study varies the one thing those
two never varied - whose initialization the survivors hold, with the topology
fixed - to separate "irreducibly paired" from "portable".

Per seed: one discovery block on the registered 5.10/5.16/5.17 config, then
three transports plus one validity transport:

- ``O``  adaptive final genome, rewound to its **own** initial model
- ``D``  adaptive final genome, rewound to a **donor** initial model (seed+1000)
- ``F``  fixed (unmutated) genome, rewound to the **donor** initial model -
         which copies every tensor, so ``F`` *is* the donor's model trained on
         this seed's schedule. ``D`` and ``F`` therefore share one
         initialization and differ only in topology.
- ``O_fixed`` fixed genome rewound to its own initial model - the validity
         transport, which must reproduce the fixed arm's own discovery run.

Primaries (exact one-tailed sign test, n = 384, per-primary threshold
alpha/2 = 0.025 -> exact bar 212; the declared bar is asserted at run time):

1. the pairing matters - ``O`` < ``D``
2. the genome is portable - ``D`` < ``F``

Usage:

    PYTHONPATH=src python scripts/portability_study.py run --shard 0 --shards 20
    PYTHONPATH=src python scripts/portability_study.py verdict --shards 20
    PYTHONPATH=src python scripts/portability_study.py summary
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from act.evolution.experiment import ExperimentConfig, run_experiment  # noqa: E402
from act.evolution.registry import _stable_hash  # noqa: E402
from act.evolution.reuse import transport_genome  # noqa: E402
from act.evolution.study import _binomial_survival, _gate_min_wins  # noqa: E402
from act.topology import ArchitectureGenome  # noqa: E402

SHARD_DIR = ROOT / "artifacts" / "toy-portability-study"
#: Declared donor offset: seed ``s``'s foreign initialization is seed ``s+1000``'s
#: initial model, so the donor block is 84000-84383 for the registered block.
DONOR_OFFSET = 1000
PROFILES: dict[str, dict] = {
    "depth4": {
        "seeds": tuple(range(83000, 83384)),
        "report_section": "5.18",
        "summary": "artifacts/study-summaries/portability-depth4-384.json",
    },
    # Machinery smoke test only, on spent blocks.
    "smoke": {
        "seeds": tuple(range(73000, 73004)),
        "report_section": "5.18 (smoke)",
        "summary": "artifacts/study-summaries/portability-smoke.json",
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
TRANSPORTS = {
    "O": (("adaptive", "rewound"), None),
    "D": (("adaptive", "rewound"), "donor"),
    "F": (("fixed", "rewound"), "donor"),
    "O_fixed": (("fixed", "rewound"), None),
}
ALPHA = 0.05
DECLARED_ALPHA = ALPHA / 2  # 0.025, the section 5.9-5.17 convention
DECLARED_N = 384
DECLARED_BAR = 212  # exact one-tailed binomial bar at n = 384, alpha = 0.025


def _configs(seed: int) -> tuple[ExperimentConfig, ExperimentConfig]:
    host = ExperimentConfig(**REGISTERED, seed=seed, arms=ARMS)
    donor = replace(host, seed=seed + DONOR_OFFSET)
    return host, donor


def _run_seed(seed: int) -> dict:
    host, donor = _configs(seed)
    arms = {arm.mode: arm for arm in run_experiment(host).arms}
    transport: dict[str, dict] = {}
    for label, ((mode, initialization), source) in TRANSPORTS.items():
        result = transport_genome(
            ArchitectureGenome.from_dict(arms[mode].genome),
            host,
            initialization=initialization,
            rewind_source=donor if source == "donor" else None,
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
        "donor_seed": seed + DONOR_OFFSET,
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
        "bar_and_p_agree": (wins >= bar) == (p <= alpha),
    }


def _built_in_checks(experiments: list[dict]) -> dict:
    """Pre-conditions that must hold before the primaries can be read.

    The load-bearing ones: the own-initialization rewind must reproduce a known
    arm exactly (or the rewind path is not trustworthy), the donor rewinds must
    actually transfer, and the unmutated genome rewound to the donor must copy
    every tensor - that last one is what makes ``D`` vs ``F`` a comparison with
    initialization held identical.
    """
    return {
        "declared_bar_matches_exact_binomial": _gate_min_wins(DECLARED_N, DECLARED_ALPHA)
        == DECLARED_BAR,
        "registered_block_matches_declared_n": len(PROFILES["depth4"]["seeds"])
        == DECLARED_N,
        "own_rewind_reproduces_fixed_arm_exactly": all(
            exp["transport"]["O_fixed"]["loss"] == exp["discovery"]["losses"]["fixed"]
            and exp["transport"]["O_fixed"]["training_flops"]
            == exp["discovery"]["training_flops"]["fixed"]
            for exp in experiments
        ),
        "donor_rewind_transfers_parameters": all(
            exp["transport"][label]["initialization"]["copied_parameters"] > 0
            for exp in experiments
            for label in ("D", "F")
        ),
        "donor_model_replaces_every_tensor": all(
            exp["transport"]["F"]["initialization"]["skipped_parameters"] == 0
            for exp in experiments
        ),
        "transport_modes_recorded": all(
            exp["transport"][label]["initialization"]["mode"] == "rewound"
            for exp in experiments
            for label in ("O", "D", "F", "O_fixed")
        ),
        "fixed_never_mutated": all(
            exp["discovery"]["mutations"]["fixed"] == 0 for exp in experiments
        ),
        "adaptive_mutated_3": all(
            exp["discovery"]["mutations"]["adaptive"] == 3 for exp in experiments
        ),
    }


def summarize(profile: str) -> dict:
    """Build the committed study summary from this profile's shard payloads."""
    spec = PROFILES[profile]
    experiments = load_shards(_shards_for(profile), profile)
    n = len(experiments)
    checks = _built_in_checks(experiments)
    if not all(checks.values()):
        raise SystemExit(f"built-in checks failed: {checks}")

    def losses(label: str) -> list[float]:
        return [exp["transport"][label]["loss"] for exp in experiments]

    def delta(a: str, b: str) -> list[float]:
        return [x - y for x, y in zip(losses(a), losses(b))]

    # Primary 1 is the difference-in-differences (report section 5.18.1, a
    # pre-run correction): `O - D` is contaminated by this seed's own-versus-
    # donor initialization-quality difference, which the smoke block measured at
    # ~5x the effect being chased (stdev 0.86 vs a topology effect near 0.1).
    # Subtracting the same quantity on the unmutated topology removes it. The
    # raw `O - D` stays reported as a secondary.
    pairing_deltas = [
        (exp["transport"]["O"]["loss"] - exp["transport"]["D"]["loss"])
        - (exp["discovery"]["losses"]["fixed"] - exp["transport"]["F"]["loss"])
        for exp in experiments
    ]
    comparisons = {
        "pairing_difference_in_differences": _comparison(pairing_deltas),
        "D_vs_F": _comparison(delta("D", "F")),
        "O_vs_D": _comparison(delta("O", "D")),
        # Both sides on their own initialization - the section-5.17 P2 contrast
        # replicated on this block, and the row the synthesis table quotes.
        "O_vs_O_fixed": _comparison(delta("O", "O_fixed")),
        # Secondary controls: is the donor initialization intrinsically better or
        # worse for the unmutated topology, and does the fresh/foreign pair
        # behave alike?
        "F_vs_own_fixed": _comparison(
            [
                exp["transport"]["F"]["loss"] - exp["discovery"]["losses"]["fixed"]
                for exp in experiments
            ]
        ),
        "D_vs_O_fixed": _comparison(delta("D", "O_fixed")),
    }
    p1 = comparisons["pairing_difference_in_differences"]["passes"]
    p2 = comparisons["D_vs_F"]["passes"]
    if p1 and p2:
        outcome = (
            "branch (d) both PASS: the pairing and portability both hold - the "
            "genome is portable AND its own initialization adds value"
        )
    elif p1:
        outcome = (
            "branch (b) 1 PASS / 2 FAIL: the initialization is paired with the "
            "topology and the genome is not portable"
        )
    elif p2:
        outcome = (
            "branch (c) 1 FAIL / 2 PASS: the pairing is not what 5.17 measured "
            "and the topology is portable under any initialization - 5.17's P1 "
            "reading is revised and the section-52 scale-up is triggered"
        )
    else:
        outcome = (
            "branch (a) both FAIL: the artifact is irreducibly paired - neither "
            "the topology alone nor a same-distribution reinitialization carries "
            "it; plan section 102's reusability claim is unsupported at toy "
            "fidelity in its own terms"
        )
    payload = {
        "record_type": "portability_study",
        "primary_1_statistic": "difference_in_differences",
        "protocol": "donor-initialization-transport",
        "profile": profile,
        "report_section": spec["report_section"],
        "alpha": ALPHA,
        "per_primary_alpha": DECLARED_ALPHA,
        "declared_bar": DECLARED_BAR,
        "donor_offset": DONOR_OFFSET,
        "arms": ["random", "adaptive", "fixed"],
        "transports": {label: [list(pair), source] for label, (pair, source) in TRANSPORTS.items()},
        "independent_arm_rng": True,
        "seeds": list(spec["seeds"]),
        "donor_seeds": [seed + DONOR_OFFSET for seed in spec["seeds"]],
        "config": {**REGISTERED, "arms": ARMS},
        "per_seed": [
            {
                "seed": exp["seed"],
                "donor_seed": exp["donor_seed"],
                "discovery_losses": exp["discovery"]["losses"],
                "transport_losses": {
                    label: exp["transport"][label]["loss"] for label in TRANSPORTS
                },
                "transport_active_parameter_count": {
                    label: exp["transport"][label]["active_parameter_count"]
                    for label in TRANSPORTS
                },
            }
            for exp in experiments
        ],
        "comparisons": comparisons,
        "transport_means": {
            label: sum(losses(label)) / n for label in TRANSPORTS
        },
        "discovery_means": {
            arm: sum(exp["discovery"]["losses"][arm] for exp in experiments) / n
            for arm in ("random", "adaptive", "fixed")
        },
        "adaptive_active_params_lower_than_initial_seeds": sum(
            1
            for exp in experiments
            if exp["transport"]["D"]["active_parameter_count"]
            < exp["transport"]["F"]["active_parameter_count"]
        ),
        "built_in_checks": checks,
        "outcome": outcome,
        "summary_note": (
            "Genome portability (report section 5.18): three transports of the "
            "discovered genomes per seed. O = adaptive genome rewound to its own "
            "initial model; D = adaptive genome rewound to a donor's initial "
            "model (seed+1000); F = unmutated genome rewound to the donor's "
            "initial model, which copies every tensor and so IS the donor model "
            "trained on this seed's schedule. D and F therefore share one "
            "initialization and differ only in topology, which is what makes "
            "D_vs_F a test of the topology's standalone value. Statistics "
            "recompute exactly from per_seed; p-values are exact one-tailed "
            "binomial."
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
        "built-in check: own-initialization rewind reproduces the fixed arm exactly: "
        f"{payload['built_in_checks']['own_rewind_reproduces_fixed_arm_exactly']}"
    )
    print(
        "built-in check: donor rewind replaces every tensor of the unmutated genome "
        f"(so F is the donor model): "
        f"{payload['built_in_checks']['donor_model_replaces_every_tensor']}"
    )
    print(f"\ntransport means: {payload['transport_means']}")
    print(f"discovery means: {payload['discovery_means']}")
    print(
        "primary 1 is the difference-in-differences; the raw own-minus-donor "
        "comparison is reported beside it as a secondary"
    )
    for name, comparison in payload["comparisons"].items():
        print(
            f"\n{name}: wins {comparison['wins']}/{comparison['n']}, "
            f"mean {comparison['mean_delta']:+.6f}, p {comparison['binomial_p']:.2e} "
            f"(bar {comparison['min_wins']}): "
            f"{'PASS' if comparison['passes'] else 'FAIL'}"
        )
    print(
        f"\ncapacity: adaptive active params below the initial topology in "
        f"{payload['adaptive_active_params_lower_than_initial_seeds']}/{n} seeds"
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
