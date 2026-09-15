"""Section 5.33 study: is the probe's own prune target better than a load rule?

Pre-registered in ``RESEARCH_REPORT.md`` section 5.33. Two findings set this up:

1. Section 5.32 showed that on the **non-probe** paths, removing the least-loaded
   expert is worth 27.6-41.6% of the historical prune penalty - a real lever.
2. The same work turned up a structural fact, now pinned by
   ``tests/test_prune_target_policy.py``: ``prune_expert_policy`` is **inert** on the
   probe-guided discovery path, because ``_mutate_by_probe`` enumerates every expert
   through ``feasible_expert_mutations`` and returns before the policy branch is
   reached. The discovery controller already owns the prune-target lever, and
   exercises it by measured probe loss.

So the open question is not whether the target matters - it is whether the probe's
*own* way of choosing it beats the cheap load rule. That needs the load rule on the
probe path, which is what ``probe_selection="quietest"`` adds (both probe paths, so
they cannot drift). The rule isolates the target: it fires only when the probe's own
pick is a prune, and then takes the least-loaded expert in the probed pool instead.

Arms, all on the section 5.30 registered configuration with the section 5.28
recommended settings (clone new experts, preserve the router), so this is the
configuration the delivery pipeline would actually run:

- ``A``  probe selection ``argmin`` (the registered default, and the historical run);
- ``Q``  probe selection ``quietest`` - the load rule on the prune target;
- ``M``  probe selection ``random`` - the pool-matched loss-blind control;
- ``G``  probe selection ``argmax`` - the inverted control;
- ``F``  the fixed arm (unmutated), identical under all four;
- ``A_own`` / ``Q_own``  each fused run's final genome retrained from its own
  initialization.

Primaries per depth, on the FUSED loss (the deployed number under section 5.28's
keep-the-fused-run recommendation):

1. **the probe's selection carries information here** - ``A`` < ``M``. Pool-matched,
   so the candidates and the probe cost are identical and only the rule differs.
2. **the probe beats the cheap load rule** - ``A`` < ``Q``. This is the headline:
   if it fails, a two-line load rule replaces the probe's prune decision.

Statistic, declared in advance: a one-sided paired **sign-flip permutation test on
the mean** of the per-seed deltas, 20,000 permutations seeded from the block size,
against ``alpha/4 = 0.0125`` - the statistic section 5.32 adopted after section 5.31
showed the paired sign test needs ~4x the block for the same effect. Both statistics
are stored side by side for every comparison.

Usage:

    PYTHONPATH=src python scripts/probe_selection_study.py run --shard 0 --shards 60 --profile sel2
    PYTHONPATH=src python scripts/probe_selection_study.py verdict --profile sel2
    PYTHONPATH=src python scripts/probe_selection_study.py summary --profile sel2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from act.evolution.experiment import ExperimentConfig, run_experiment  # noqa: E402
from act.evolution.registry import _stable_hash  # noqa: E402
from act.evolution.reuse import transport_genome  # noqa: E402
from act.evolution.study import _binomial_survival, _gate_min_wins  # noqa: E402
from act.topology import ArchitectureGenome, genome_hash  # noqa: E402

SHARD_DIR = ROOT / "artifacts" / "toy-probe-selection-study"
PROFILES: dict[str, dict] = {
    "sel2": {
        "seeds": tuple(range(119000, 120536)),
        "overrides": {"depth": 2},
        "shards": 60,
        "report_section": "5.33",
        "summary": "artifacts/study-summaries/probe-selection-depth2-1536.json",
    },
    "sel3": {
        "seeds": tuple(range(121000, 122536)),
        "overrides": {"depth": 3},
        "shards": 60,
        "report_section": "5.33",
        "summary": "artifacts/study-summaries/probe-selection-depth3-1536.json",
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
    new_expert_initialization="clone",
)
SELECTIONS = (("A", "argmin"), ("Q", "quietest"), ("M", "random"), ("G", "argmax"))
ALPHA = 0.05
DECLARED_N = 1536
PER_PRIMARY_ALPHA = ALPHA / 4
N_PERMUTATIONS = 20000
_PERMUTATION_CHUNK = 2000


def _permutation_p(deltas: list[float], seed: int) -> float:
    """One-sided sign-flip permutation p for the mean being negative."""
    values = torch.tensor(deltas, dtype=torch.float64)
    observed = float(values.mean())
    generator = torch.Generator().manual_seed(seed)
    at_least_as_extreme = 0
    for start in range(0, N_PERMUTATIONS, _PERMUTATION_CHUNK):
        size = min(_PERMUTATION_CHUNK, N_PERMUTATIONS - start)
        signs = (
            torch.randint(
                0, 2, (size, values.numel()), generator=generator, dtype=torch.float64
            )
            * 2
            - 1
        )
        means = (signs @ values) / values.numel()
        at_least_as_extreme += int((means <= observed).sum())
    return (at_least_as_extreme + 1) / (N_PERMUTATIONS + 1)


def _comparison(deltas: list[float]) -> dict:
    """Both statistics side by side; ``passes`` is the registered one."""
    n = len(deltas)
    mean = sum(deltas) / n
    wins = sum(delta < 0 for delta in deltas)
    variance = (
        sum((delta - mean) ** 2 for delta in deltas) / (n - 1) if n > 1 else 0.0
    )
    standard_error = math.sqrt(variance / n) if n else 0.0
    permutation_p = _permutation_p(deltas, seed=0xC0FFEE + n)
    return {
        "n": n,
        "wins": wins,
        "sign_test_p": _binomial_survival(wins, n),
        "mean_delta": mean,
        "standard_error": standard_error,
        "t": mean / standard_error if standard_error else 0.0,
        "permutation_p": permutation_p,
        "passes": permutation_p <= PER_PRIMARY_ALPHA,
        "sign_test_passes": _binomial_survival(wins, n) <= PER_PRIMARY_ALPHA,
        "the_two_statistics_disagree": (
            (permutation_p <= PER_PRIMARY_ALPHA)
            != (_binomial_survival(wins, n) <= PER_PRIMARY_ALPHA)
        ),
        "min_wins_for_the_sign_test": _gate_min_wins(n, PER_PRIMARY_ALPHA),
    }


def _run_seed(seed: int, profile: str) -> dict:
    values = {**REGISTERED, **PROFILES[profile]["overrides"]}
    outcome: dict[str, dict] = {}
    fixed_loss: float | None = None
    for label, selection in SELECTIONS:
        config = ExperimentConfig(
            **{
                **values,
                "seed": seed,
                "arms": "fixed,adaptive",
                "probe_selection": selection,
            }
        )
        arms = {arm.mode: arm for arm in run_experiment(config).arms}
        fixed = arms["fixed"].final_loss
        if fixed_loss is None:
            fixed_loss = fixed
        discovered = arms["adaptive"]
        history = discovered.mutation_history
        prunes = sum(1 for event in history if event["operation"] == "prune")
        outcome[label] = {
            "fixed_loss": fixed,
            "fused_loss": discovered.final_loss,
            "fused_genome": discovered.genome,
            "mutations": discovered.mutation_count,
            "prunes": prunes,
            "search_flops": discovered.search_flops,
        }
    # Only the two selectors on trial get a frozen retrain: the control arms are
    # there to bound the comparison, not to be delivered.
    for label in ("A", "Q"):
        config = ExperimentConfig(
            **{
                **values,
                "seed": seed,
                "arms": "fixed,adaptive",
                "probe_selection": dict(SELECTIONS)[label],
            }
        )
        outcome[label]["own_transport_loss"] = transport_genome(
            ArchitectureGenome.from_dict(outcome[label]["fused_genome"]),
            config,
            initialization="rewound",
        ).final_loss
    losses = {"F": fixed_loss}
    for label, _ in SELECTIONS:
        losses[label] = outcome[label]["fused_loss"]
    for label in ("A", "Q"):
        losses[f"{label}_own"] = outcome[label]["own_transport_loss"]
    return {
        "seed": seed,
        "losses": losses,
        "penalties": {label: losses[f"{label}_own"] - losses[label] for label in ("A", "Q")},
        "fixed_arm_identical_across_selections": len(
            {outcome[label]["fixed_loss"] for label, _ in SELECTIONS}
        )
        == 1,
        "mutations": {label: outcome[label]["mutations"] for label, _ in SELECTIONS},
        "prunes": {label: outcome[label]["prunes"] for label, _ in SELECTIONS},
        "final_genomes_distinct": len(
            {
                genome_hash(ArchitectureGenome.from_dict(outcome[label]["fused_genome"]))
                for label, _ in SELECTIONS
            }
        ),
        "search_flops": {label: outcome[label]["search_flops"] for label, _ in SELECTIONS},
    }


def run_shard(shard: int, shards: int, profile: str) -> Path:
    if shards < 1 or not 0 <= shard < shards:
        raise SystemExit("--shard must be in [0, --shards)")
    if profile not in PROFILES:
        raise SystemExit(f"--profile must be one of {sorted(PROFILES)}")
    directory = SHARD_DIR / profile
    path = directory / f"shard{shard:03d}.json"
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
    return PROFILES[profile]["shards"]


def load_shards(shards: int, profile: str) -> list[dict]:
    experiments: list[dict] = []
    for shard in range(shards):
        path = SHARD_DIR / profile / f"shard{shard:03d}.json"
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


def _built_in_checks(experiments: list[dict], profile: str) -> dict:
    """Validity checks, all of them RATES rather than all-seeds requirements.

    The first version of this demanded a prune in every selection of every seed, and
    failed on the 8 seeds where a selection happened to grow at all three of its
    events. That is a legitimate outcome, not a defect - the same over-strict-guard
    mistake section 5.31.1 had to reconcile - so the binding check is now a fraction,
    with the realized counts recorded in the summary so a reader can see them.
    """
    spec = PROFILES[profile]
    all_four = sum(1 for exp in experiments if exp["final_genomes_distinct"] == 4)
    pairs = len(experiments) * len(SELECTIONS)
    with_prune = sum(
        1 for exp in experiments for count in exp["prunes"].values() if count >= 1
    )
    return {
        "block_matches_declared_n": len(spec["seeds"]) == DECLARED_N,
        "the_fixed_arm_is_identical_across_all_four_selections": all(
            exp["fixed_arm_identical_across_selections"] for exp in experiments
        ),
        "all_selections_used_the_full_budget": all(
            all(count == 3 for count in exp["mutations"].values())
            for exp in experiments
        ),
        # The treatment has to bind: if a selection never pruned, it could not
        # respond to a prune-target rule at all and its seeds would score as
        # non-wins for a reason the study is not about (section 5.30's lesson).
        "the_treatment_binds_in_at_least_95pct_of_seed_selection_pairs": (
            with_prune / pairs >= 0.95
        ),
        # A chance collision between two selection rules is legitimate, so this is
        # a rate rather than an all-seeds requirement (section 5.31.1's lesson).
        "the_selections_are_mostly_live": all_four >= len(experiments) // 2,
        "the_selections_are_live": all_four > 0,
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

    def delta(first: list[float], second: list[float]) -> list[float]:
        return [x - y for x, y in zip(first, second)]

    comparisons = {
        "fused_A_vs_M": _comparison(delta(losses("A"), losses("M"))),
        "fused_A_vs_Q": _comparison(delta(losses("A"), losses("Q"))),
        "fused_Q_vs_M": _comparison(delta(losses("Q"), losses("M"))),
        "fused_A_vs_G": _comparison(delta(losses("A"), losses("G"))),
        "fused_A_vs_F": _comparison(delta(losses("A"), losses("F"))),
        "fused_Q_vs_F": _comparison(delta(losses("Q"), losses("F"))),
        "own_A_vs_Q": _comparison(delta(losses("A_own"), losses("Q_own"))),
        "penalty_Q_less_than_A": _comparison(
            delta(
                [exp["penalties"]["Q"] for exp in experiments],
                [exp["penalties"]["A"] for exp in experiments],
            )
        ),
    }
    p1 = comparisons["fused_A_vs_M"]["passes"]
    p2 = comparisons["fused_A_vs_Q"]["passes"]
    if p1 and p2:
        outcome = (
            "branch (a) the probe's selection carries information here AND beats the "
            "cheap load rule: section 5.32's lever is not reachable more cheaply, and "
            "the discovery residual is not a prune-target problem"
        )
    elif p2 and not p1:
        outcome = (
            "branch (b) the load rule beats the probe while the probe is no better "
            "than its own pool-matched control - an indictment of the selection rule "
            "on this configuration"
        )
    elif p1 and not p2:
        outcome = (
            "branch (c) the probe's selection is informative but the load rule is no "
            "worse on the prune target: the probe's prune choice can be replaced by a "
            "two-line rule, and the search gets cheaper"
        )
    else:
        outcome = (
            "branch (d) neither passes: on this configuration the prune-target choice "
            "carries no measurable information under either rule"
        )
    payload = {
        "record_type": "probe_selection_study",
        "protocol": "probe-argmin-vs-quietest-load-rule-vs-pool-matched-control",
        "profile": profile,
        "report_section": spec["report_section"],
        "alpha": ALPHA,
        "per_primary_alpha": PER_PRIMARY_ALPHA,
        "declared_n": DECLARED_N,
        "n_permutations": N_PERMUTATIONS,
        "depth": spec["overrides"]["depth"],
        "seeds": list(spec["seeds"]),
        "config": {**REGISTERED, **spec["overrides"]},
        "per_seed": experiments,
        "comparisons": comparisons,
        "loss_means": {
            label: sum(losses(label)) / n
            for label in ("F", "A", "Q", "M", "G", "A_own", "Q_own")
        },
        "prune_counts": {
            label: sum(exp["prunes"][label] for exp in experiments)
            for label, _ in SELECTIONS
        },
        "seed_selection_pairs_with_a_prune": sum(
            1
            for exp in experiments
            for count in exp["prunes"].values()
            if count >= 1
        ),
        "seed_selection_pairs": len(experiments) * len(SELECTIONS),
        "seeds_with_four_distinct_genomes": sum(
            1 for exp in experiments if exp["final_genomes_distinct"] == 4
        ),
        "outcome": outcome,
        "built_in_checks": checks,
        "summary_note": (
            "Section 5.33. Does the probe's own prune-target selection beat the cheap "
            "load rule that section 5.32 validated on the non-probe paths? Both "
            "primaries are on the FUSED loss, signed so a win is the primary's "
            "prediction: fused(A) - fused(M) < 0 and fused(A) - fused(Q) < 0. The "
            "quietest rule isolates the TARGET - it fires only when the probe's own "
            "pick is a prune, and then takes the least-loaded expert in the probed "
            "pool - so the A-vs-Q contrast differs in the prune target alone. All "
            "four selections share the fixed arm and the probe pool."
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
    print(f"prune counts: {payload['prune_counts']}")
    print(
        f"seeds with 4 distinct final genomes: "
        f"{payload['seeds_with_four_distinct_genomes']}/{n}"
    )
    for name, comparison in payload["comparisons"].items():
        print(
            f"\n{name}: n {comparison['n']}, mean {comparison['mean_delta']:+.5f} "
            f"(se {comparison['standard_error']:.5f}, t {comparison['t']:+.2f})\n"
            f"    permutation p {comparison['permutation_p']:.2e} "
            f"-> {'PASS' if comparison['passes'] else 'FAIL'}\n"
            f"    sign test {comparison['wins']}/{comparison['n']} p "
            f"{comparison['sign_test_p']:.2e} "
            f"(bar {comparison['min_wins_for_the_sign_test']}) "
            f"-> {'PASS' if comparison['sign_test_passes'] else 'FAIL'}"
            f"{'  <-- THE TWO DISAGREE' if comparison['the_two_statistics_disagree'] else ''}"
        )
    print(f"\nOUTCOME: {payload['outcome']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run one shard")
    run_p.add_argument("--shard", type=int, required=True)
    run_p.add_argument("--shards", type=int, required=True)
    run_p.add_argument("--profile", default="sel2", choices=sorted(PROFILES))
    verdict_p = sub.add_parser("verdict", help="aggregate shards")
    verdict_p.add_argument("--profile", default="sel2", choices=sorted(PROFILES))
    sum_p = sub.add_parser("summary", help="write the committed study summary")
    sum_p.add_argument("--profile", default="sel2", choices=sorted(PROFILES))
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
