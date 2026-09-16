"""Section 5.34 study: does the probe score candidates under the rebuild it applies?

Pre-registered in ``RESEARCH_REPORT.md`` section 5.34. Section 5.24 found that the
historical rebuild discards a mutated layer's learned router whenever its expert
count changes, and sections 5.26-5.33 measured everything under the fix
(``preserve_router_on_rebuild=True``, ``new_expert_initialization="clone"``).

The fix reached the **applied** rebuild only. ``_mutate_by_probe`` still scores every
candidate through ``rebuild_model_from_genome``'s historical defaults, so a prune
candidate is scored after its layer's routing head has been re-drawn at random - the
section 5.24 perturbation itself - and a grow candidate is scored with a random new
expert rather than the cloned one sections 5.26/5.27 measured as better by ~0.10
nats. The new ``probe_scoring`` config field selects which rebuild the candidate is
*scored* in: ``"default"`` (historical, and every committed number's setting) or
``"applied"`` (the arm's own settings).

Arms per seed, all on the section 5.30/5.33 registered configuration:

- ``F``  the fixed arm (unmutated), identical under both settings;
- ``D``  the adaptive arm under ``probe_scoring="default"``;
- ``S``  the adaptive arm under ``probe_scoring="applied"``;
- ``D_own`` / ``S_own``  each fused run's final genome retrained from its own
  initialization.

Primaries per depth:

1. ``fused(S) < fused(D)`` - the deployed number under section 5.28's
   keep-the-fused-run recommendation.
2. ``own(S) < own(D)`` - the freeze-and-retrain delivered number. Declared
   *uncertain* in advance: section 5.32's lever moved the fused column while
   slightly worsening this one.

Statistic, declared in advance: a one-sided paired sign-flip permutation test on the
mean of the per-seed deltas, 20,000 permutations, against ``alpha/4 = 0.0125``, with
the paired sign test stored beside it.

The mechanism block forces the operation (section 5.25's control) and declares a
prediction: the gain is concentrated on the **prune** block, because that is where the
default scoring's distortion is the section 5.24 router discard itself, while the grow
block's distortion is only fresh-vs-cloned expert initialisation.

The integrity check that makes the comparison clean: with ``probe_candidate_cap=1`` the
probed pool holds exactly one candidate, so the score cannot influence which candidate
is applied - only the ranking could ever do that - and the two scoring rules MUST give
bit-identical runs. That is checked over a 64-seed block at each depth, and pinned as a
unit test in ``tests/test_probe_scoring.py``. (A ``probe_selection="random"`` block is
NOT an identity: the seeded index indexes the list AFTER it is sorted by probe loss, so
the chosen candidate is score-dependent even under the loss-blind rule.)

Usage:

    PYTHONPATH=src python scripts/probe_scoring_study.py run --shard 0 --shards 60 --profile score2
    PYTHONPATH=src python scripts/probe_scoring_study.py verdict --profile score2
    PYTHONPATH=src python scripts/probe_scoring_study.py summary --profile score2
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

SHARD_DIR = ROOT / "artifacts" / "toy-probe-scoring-study"
PROFILES: dict[str, dict] = {
    "score2": {
        "seeds": tuple(range(123000, 124536)),
        "overrides": {"depth": 2},
        "shards": 60,
        "report_section": "5.34",
        "kind": "main",
        "summary": "artifacts/study-summaries/probe-scoring-depth2-1536.json",
    },
    "score3": {
        "seeds": tuple(range(125000, 126536)),
        "overrides": {"depth": 3},
        "shards": 60,
        "report_section": "5.34",
        "kind": "main",
        "summary": "artifacts/study-summaries/probe-scoring-depth3-1536.json",
    },
    "grow2": {
        "seeds": tuple(range(127000, 127536)),
        "overrides": {"depth": 2, "forced_operation": "grow"},
        "shards": 30,
        "report_section": "5.34",
        "kind": "mechanism",
        "summary": "artifacts/study-summaries/probe-scoring-forced-grow-depth2-536.json",
    },
    "prune2": {
        "seeds": tuple(range(127000, 127536)),
        "overrides": {"depth": 2, "forced_operation": "prune"},
        "shards": 30,
        "report_section": "5.34",
        "kind": "mechanism",
        "summary": "artifacts/study-summaries/probe-scoring-forced-prune-depth2-536.json",
    },
    "id2": {
        "seeds": tuple(range(128000, 128064)),
        "overrides": {"depth": 2, "probe_candidate_cap": 1},
        "shards": 8,
        "report_section": "5.34",
        "kind": "identity",
        "summary": "artifacts/study-summaries/probe-scoring-identity-depth2-64.json",
    },
    "id3": {
        "seeds": tuple(range(128064, 128128)),
        "overrides": {"depth": 3, "probe_candidate_cap": 1},
        "shards": 8,
        "report_section": "5.34",
        "kind": "identity",
        "summary": "artifacts/study-summaries/probe-scoring-identity-depth3-64.json",
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
    probe_selection="argmin",
)
SCORINGS = (("D", "default"), ("S", "applied"))
ALPHA = 0.05
DECLARED_N = 1536
PER_PRIMARY_ALPHA = ALPHA / 4
N_PERMUTATIONS = 20000
_PERMUTATION_CHUNK = 2000
MIN_BINDING_RATE = 0.10
# The probe books the SCORED model's active-parameter count, which depends on the
# model's own routing, so the two scoring rules book slightly different search cost
# (measured; section 5.34 records it rather than asserting exact equality). This is
# a guard against the knob having changed something other than how candidates rank.
MAX_MEAN_SEARCH_COST_DELTA = 0.02


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


def _history_signature(arm) -> list[tuple[str, str, int | None]]:
    return [
        (str(event["operation"]), str(event["layer_id"]), event.get("expert_id"))
        for event in arm.mutation_history or ()
    ]


def _run_seed(seed: int, profile: str) -> dict:
    spec = PROFILES[profile]
    values = {**REGISTERED, **spec["overrides"]}
    kind = spec["kind"]
    outcome: dict[str, dict] = {}
    for label, scoring in SCORINGS:
        config = ExperimentConfig(
            **{**values, "seed": seed, "arms": "fixed,adaptive", "probe_scoring": scoring}
        )
        arms = {arm.mode: arm for arm in run_experiment(config).arms}
        discovered = arms["adaptive"]
        history = discovered.mutation_history or ()
        outcome[label] = {
            "fixed_loss": arms["fixed"].final_loss,
            "fused_loss": discovered.final_loss,
            "mutations": discovered.mutation_count,
            "grows": sum(1 for event in history if event["operation"] == "grow"),
            "prunes": sum(1 for event in history if event["operation"] == "prune"),
            "search_flops": discovered.search_flops,
            "signature": _history_signature(discovered),
            "genome": discovered.genome,
            "probe_losses": [float(event["probe_loss"]) for event in history],
        }
    record: dict = {"seed": seed}
    if kind == "identity":
        # cap 1: the probed pool holds exactly one candidate, so the score cannot
        # decide which candidate is applied and the two scoring rules must be
        # bit-identical. Any difference falsifies the claim that this knob changes
        # only the score used to rank a pool whose membership it does not touch.
        record["identical_fused"] = outcome["D"]["fused_loss"] == outcome["S"]["fused_loss"]
        record["identical_fixed"] = outcome["D"]["fixed_loss"] == outcome["S"]["fixed_loss"]
        record["identical_selection"] = outcome["D"]["signature"] == outcome["S"]["signature"]
        record["identical_flops"] = outcome["D"]["search_flops"] == outcome["S"]["search_flops"]
        record["fused_delta"] = outcome["S"]["fused_loss"] - outcome["D"]["fused_loss"]
        return record

    for label, _ in SCORINGS:
        config = ExperimentConfig(
            **{
                **values,
                "seed": seed,
                "arms": "fixed,adaptive",
                "probe_scoring": dict(SCORINGS)[label],
            }
        )
        outcome[label]["own_transport_loss"] = transport_genome(
            ArchitectureGenome.from_dict(outcome[label]["genome"]),
            config,
            initialization="rewound",
        ).final_loss
    events_default = outcome["D"]["signature"]
    events_applied = outcome["S"]["signature"]
    record["losses"] = {
        "F": outcome["D"]["fixed_loss"],
        "D": outcome["D"]["fused_loss"],
        "S": outcome["S"]["fused_loss"],
        "D_own": outcome["D"]["own_transport_loss"],
        "S_own": outcome["S"]["own_transport_loss"],
    }
    record["penalties"] = {
        "D": record["losses"]["D_own"] - record["losses"]["D"],
        "S": record["losses"]["S_own"] - record["losses"]["S"],
    }
    record["fixed_arm_identical_across_settings"] = (
        outcome["D"]["fixed_loss"] == outcome["S"]["fixed_loss"]
    )
    record["search_flops"] = {
        label: outcome[label]["search_flops"] for label, _ in SCORINGS
    }
    record["search_flops_relative_delta"] = (
        abs(outcome["S"]["search_flops"] - outcome["D"]["search_flops"])
        / min(outcome["S"]["search_flops"], outcome["D"]["search_flops"])
        if min(outcome["S"]["search_flops"], outcome["D"]["search_flops"])
        else 0.0
    )
    record["probe_losses"] = {
        label: outcome[label]["probe_losses"] for label, _ in SCORINGS
    }
    record["mutations_match"] = outcome["D"]["mutations"] == outcome["S"]["mutations"]
    record["selection_changes"] = sum(
        1 for first, second in zip(events_default, events_applied) if first != second
    )
    record["events"] = len(events_applied)
    record["operation_changes"] = sum(
        1
        for first, second in zip(events_default, events_applied)
        if first[0] != second[0]
    )
    record["grows"] = {"D": outcome["D"]["grows"], "S": outcome["S"]["grows"]}
    record["prunes"] = {"D": outcome["D"]["prunes"], "S": outcome["S"]["prunes"]}
    record["final_genomes_distinct"] = len(
        {
            genome_hash(ArchitectureGenome.from_dict(outcome[label]["genome"]))
            for label, _ in SCORINGS
        }
    )
    record["losses_identical"] = record["losses"]["D"] == record["losses"]["S"]
    return record


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


def _mechanism_checks(experiments: list[dict], profile: str) -> dict:
    """The forced-operation blocks must actually have been forced.

    Section 5.25's control: a forced operation the genome cannot perform is left
    unperformed rather than replaced by the opposite one, so the check is that the
    blocks contain no event of the other operation at all.
    """
    operation = PROFILES[profile]["overrides"]["forced_operation"]
    other = "prune" if operation == "grow" else "grow"
    wrong = sum(
        1 for exp in experiments if exp["prunes" if other == "prune" else "grows"]["D"]
    )
    return {
        "no_event_of_the_other_operation": wrong == 0,
        "the_fixed_arm_is_identical_across_settings": all(
            exp["fixed_arm_identical_across_settings"] for exp in experiments
        ),
        "both_settings_used_the_full_budget": all(
            exp["mutations_match"] for exp in experiments
        ),
    }


def _main_checks(experiments: list[dict], profile: str) -> dict:
    """Validity checks, expressed as rates rather than all-seeds requirements.

    Section 5.31.1's lesson: an all-seeds requirement on a chance-collision-prone
    event turns a legitimate outcome into a spurious failure. The binding check is
    the fraction of events whose selection the two scoring rules disagree on.
    """
    spec = PROFILES[profile]
    events = sum(exp["events"] for exp in experiments)
    changes = sum(exp["selection_changes"] for exp in experiments)
    return {
        "block_matches_declared_n": len(spec["seeds"]) == DECLARED_N,
        "the_fixed_arm_is_identical_across_settings": all(
            exp["fixed_arm_identical_across_settings"] for exp in experiments
        ),
        "the_search_cost_is_close_on_average": (
            sum(exp["search_flops_relative_delta"] for exp in experiments)
            / len(experiments)
            <= MAX_MEAN_SEARCH_COST_DELTA
        ),
        "both_settings_used_the_full_budget": all(
            exp["mutations_match"] for exp in experiments
        ),
        "the_treatment_binds": changes / events >= MIN_BINDING_RATE,
        "the_two_settings_are_not_redundant": any(
            not exp["losses_identical"] for exp in experiments
        ),
    }


def _identity_checks(experiments: list[dict], profile: str) -> dict:
    return {
        "every_seed_is_bit_identical_on_the_fused_loss": all(
            exp["identical_fused"] for exp in experiments
        ),
        "every_seed_is_bit_identical_on_the_fixed_arm": all(
            exp["identical_fixed"] for exp in experiments
        ),
        "every_seed_selected_the_same_candidate": all(
            exp["identical_selection"] for exp in experiments
        ),
    }


def summarize(profile: str) -> dict:
    spec = PROFILES[profile]
    experiments = load_shards(_shards_for(profile), profile)
    n = len(experiments)
    kind = spec["kind"]
    if kind == "main":
        checks = _main_checks(experiments, profile)
    elif kind == "mechanism":
        checks = _mechanism_checks(experiments, profile)
    else:
        checks = _identity_checks(experiments, profile)
    failed = {name: value for name, value in checks.items() if not value}
    if failed:
        raise SystemExit(f"built-in checks failed: {failed}")

    def losses(label: str) -> list[float]:
        return [exp["losses"][label] for exp in experiments]

    def delta(first: list[float], second: list[float]) -> list[float]:
        return [x - y for x, y in zip(first, second)]

    events = sum(exp.get("events", 0) for exp in experiments)
    changes = sum(exp.get("selection_changes", 0) for exp in experiments)
    payload: dict = {
        "record_type": "probe_scoring_study",
        "protocol": "probe-scored-in-the-default-rebuild-vs-the-applied-rebuild",
        "profile": profile,
        "kind": kind,
        "report_section": spec["report_section"],
        "alpha": ALPHA,
        "per_primary_alpha": PER_PRIMARY_ALPHA,
        "declared_n": DECLARED_N,
        "n_permutations": N_PERMUTATIONS,
        "depth": spec["overrides"]["depth"],
        "seeds": list(spec["seeds"]),
        "config": {**REGISTERED, **spec["overrides"]},
        "per_seed": experiments,
        "built_in_checks": checks,
    }
    if kind == "identity":
        payload["summary_note"] = (
            "Section 5.34's integrity check, run at probe_candidate_cap=1 so the probed "
            "pool holds exactly one candidate and the score cannot decide what gets "
            "applied. probe_scoring must then change nothing: the two settings must be "
            "bit-identical on the fixed arm, the fused loss, the selected candidates and "
            "the booked search FLOPs. A non-identical seed would falsify the claim that "
            "the knob changes only the score used to rank a pool whose membership it "
            "does not touch - and would mean this study is measuring something else."
        )
        payload["all_identical"] = all(
            exp["identical_fused"]
            and exp["identical_fixed"]
            and exp["identical_selection"]
            for exp in experiments
        )
        payload["search_flops_identical_fraction"] = sum(
            1 for exp in experiments if exp["identical_flops"]
        ) / len(experiments)
    else:
        comparisons = {
            "fused_S_vs_D": _comparison(delta(losses("S"), losses("D"))),
            "own_S_vs_D": _comparison(delta(losses("S_own"), losses("D_own"))),
            "fused_D_vs_F": _comparison(delta(losses("D"), losses("F"))),
            "fused_S_vs_F": _comparison(delta(losses("S"), losses("F"))),
            "penalty_S_less_than_D": _comparison(
                delta(
                    [exp["penalties"]["S"] for exp in experiments],
                    [exp["penalties"]["D"] for exp in experiments],
                )
            ),
        }
        payload["comparisons"] = comparisons
        payload["loss_means"] = {
            label: sum(losses(label)) / n
            for label in ("F", "D", "S", "D_own", "S_own")
        }
        payload["events"] = events
        payload["selection_changes"] = changes
        payload["selection_change_rate"] = changes / events if events else 0.0
        payload["operation_changes"] = sum(
            exp.get("operation_changes", 0) for exp in experiments
        )
        payload["grow_totals"] = {
            label: sum(exp["grows"][label] for exp in experiments)
            for label, _ in SCORINGS
        }
        payload["prune_totals"] = {
            label: sum(exp["prunes"][label] for exp in experiments)
            for label, _ in SCORINGS
        }
        payload["seeds_with_two_distinct_final_genomes"] = sum(
            1 for exp in experiments if exp["final_genomes_distinct"] == 2
        )
        payload["search_flops_relative_delta_mean"] = sum(
            exp["search_flops_relative_delta"] for exp in experiments
        ) / len(experiments)
        payload["search_flops_relative_delta_max"] = max(
            exp["search_flops_relative_delta"] for exp in experiments
        )
        payload["summary_note"] = (
            "Section 5.34. The probe's candidates are scored in one rebuild "
            "(probe_scoring=\"default\", the historical defaults) and applied in another "
            "(the arm's own preserve_router/new_expert_initialization settings). This "
            "block scores them under the applied rebuild instead, changing nothing else: "
            "same pool, same probe batches, same budget, same applied weights for the same "
            "chosen candidate (pinned by the identity block and by tests/test_probe_scoring.py)."
        )
        if kind == "main":
            p1 = comparisons["fused_S_vs_D"]["passes"]
            p2 = comparisons["own_S_vs_D"]["passes"]
            if p1 and p2:
                outcome = (
                    "branch (a) scoring under the applied rebuild improves BOTH the fused "
                    "run and the delivered artifact: the selector's objective must match "
                    "the rebuild it applies"
                )
            elif p1 and not p2:
                outcome = (
                    "branch (b) the fix improves the fused run but not the delivered "
                    "artifact: the pipeline question stays with section 5.28"
                )
            elif p2 and not p1:
                outcome = (
                    "branch (c) the fix improves the delivered artifact but not the fused "
                    "run - the two columns disagree in the opposite direction to branch (b)"
                )
            else:
                outcome = (
                    "branch (d) neither column improves: the mismatched scoring rebuild "
                    "is a documentation defect rather than a behavioural one"
                )
            payload["outcome"] = outcome
        else:
            payload["prediction_note"] = (
                "Declared in advance: the gain is concentrated on the PRUNE block, where "
                "the default scoring's distortion is section 5.24's router discard itself, "
                "rather than on the grow block, where it is only fresh-vs-cloned expert "
                "initialisation. Compare this block's fused_S_vs_D against the sister "
                "forced block at the same depth."
            )
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
    kind = payload["kind"]
    if kind == "identity":
        print(f"identity block: all {len(payload['per_seed'])} seeds bit-identical: "
              f"{payload['all_identical']}")
        return 0
    n = len(payload["per_seed"])
    print(f"loss means: {payload['loss_means']}")
    print(
        f"selection changed on {payload['selection_changes']}/{payload['events']} events "
        f"({100 * payload['selection_change_rate']:.1f}%), operation changed on "
        f"{payload['operation_changes']}"
    )
    print(f"grows D/S: {payload['grow_totals']}  prunes D/S: {payload['prune_totals']}")
    print(
        f"seeds with two distinct final genomes: "
        f"{payload['seeds_with_two_distinct_final_genomes']}/{n}"
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
    if "outcome" in payload:
        print(f"\nOUTCOME: {payload['outcome']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run one shard")
    run_p.add_argument("--shard", type=int, required=True)
    run_p.add_argument("--shards", type=int, required=True)
    run_p.add_argument("--profile", default="score2", choices=sorted(PROFILES))
    verdict_p = sub.add_parser("verdict", help="aggregate shards")
    verdict_p.add_argument("--profile", default="score2", choices=sorted(PROFILES))
    sum_p = sub.add_parser("summary", help="write the committed study summary")
    sum_p.add_argument("--profile", default="score2", choices=sorted(PROFILES))
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
