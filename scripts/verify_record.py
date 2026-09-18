"""Re-check the phase record's numbers against the artifacts they came from.

Every claim in ``artifacts/ARCH_SEARCH_FINDINGS.md`` and ``README.md`` was computed
from a ledger, and a ledger can be re-read. This script recomputes the load-bearing
ones and fails when the record and the artifacts disagree.

It exists because three of those numbers were wrong when checked by hand, and none of
them were catchable any other way:

* §10's cost total matched neither of the columns above it (191 against rows summing
  to 196), and two arms' cost files were pooled analyses sitting in the wrong
  directory.
* the screening threshold was pinned at 0.018 nats, a value its cited measurement does
  not produce under any definition (the set's 400-step std is 0.0286 and its
  single-mean standard error 0.0165).
* a budget ratio was quoted from one draw seed as if it were typical of draws.

The test suite cannot catch these: it tests what the code does, not whether the record
describes what the code did. So this is the record's own test - run it after any
analysis is re-run, and after editing the findings.

Usage::

    PYTHONPATH=src python scripts/verify_record.py
    PYTHONPATH=src python scripts/verify_record.py --quick   # skip the slow sampling
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from act.search.architecture_eval import load_results  # noqa: E402
from act.search.architecture_report import (  # noqa: E402
    crossed_contrast,
    paired_design_contrast,
    rank,
    replication_decision,
    search_cost,
    spread_by_length,
    with_structure,
)
from arch_zoo_screen import (  # noqa: E402
    SCREEN_SEED_STD_BY_STEPS,
    comparison_noise_nats,
)
from space_typicality import budget_matching_draw, set_total_params  # noqa: E402

#: Documented values are quoted from the record to the precision it states them at, so
#: comparisons are relative rather than exact: 3.28 in a table against a computed
#: 3.2774 is agreement, not a discrepancy.
RTOL = 5e-3


@dataclass
class Claim:
    """One documented number, the recomputed value, and where each lives."""

    what: str
    documented: object
    computed: object
    source: str
    detail: str = ""
    tolerance: float = RTOL

    @property
    def ok(self) -> bool:
        expected, actual = self.documented, self.computed
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if isinstance(expected, bool) or isinstance(actual, bool):
                return bool(expected) == bool(actual)
            # An exact count must match exactly; a quoted statistic need not.
            if isinstance(expected, int) and not isinstance(actual, float):
                return expected == actual
            return abs(actual - expected) <= self.tolerance * max(abs(expected), 1e-12)
        return expected == actual


@dataclass
class Audit:
    claims: list[Claim] = field(default_factory=list)

    def check(self, *args, **kwargs) -> None:
        self.claims.append(Claim(*args, **kwargs))

    @property
    def failures(self) -> list[Claim]:
        return [claim for claim in self.claims if not claim.ok]


def arm_results(root: Path, name: str):
    """Every ledger row in an arm's directory, joined to its structural genes."""
    directory = root / name
    return with_structure(
        load_results(sorted(directory.glob("*_seed*_results.jsonl"))),
        directory / "zoo_definition.json",
    )


def own_cost(root: Path, name: str) -> dict:
    """The arm's own share of its cost file, which a pooled write puts in its total."""
    cost = json.loads((root / name / "search_cost.json").read_text())
    return cost.get("by_experiment", {}).get(name, cost)


def check_costs(root: Path, audit: Audit) -> None:
    """§10's table, row by row, against each arm's own ledgers."""
    documented = {
        "arch-zoo-100m-v1": (47, 3.598),
        "arch-zoo-100m-v2": (24, 3.201),
        "arch-zoo-100m-v2-horizon": (48, 3.254),
        "arch-zoo-100m-v3-crossed": (24, 3.277),
        "arch-zoo-100m-v3-crossed-confirm": (24, 3.256),
        "arch-zoo-100m-v4-head-paired": (24, 3.499),
    }
    runs_total = 0
    hours_total = 0.0
    for name, (runs, hours) in documented.items():
        own = own_cost(root, name)
        runs_total += own["runs_completed"]
        hours_total += own["wall_clock_hours"]
        audit.check(
            f"{name}: runs", runs, own["runs_completed"], "§10 table"
        )
        audit.check(
            f"{name}: GPU-hours", hours, own["wall_clock_hours"], "§10 table"
        )
        # The documented figure is the arm's own, so a pooled file must say so - the
        # failure this guards is a pooled total read as the primary arm's.
        cost = json.loads((root / name / "search_cost.json").read_text())
        if "by_experiment" in cost:
            audit.check(
                f"{name}: pooled total equals its split",
                cost["runs_completed"],
                sum(row["runs_completed"] for row in cost["by_experiment"].values()),
                "search_cost.json invariant",
            )

    mixer = json.loads((root / "mixer-benchmark" / "mixer_throughput.json").read_text())
    measured = [row for row in mixer["records"] if "total_step_s" in row]
    step_hours = sum(row["total_step_s"] for row in measured) / 3600
    audit.check(
        "mixer: patterns measured", 5, len(measured), "§10 table", "9 attempted, 4 OOM"
    )
    # Quoted to the precision the table states ("~0.04"): the exact sum of step times
    # is 0.0364 h, and the record deliberately rounds it up rather than claiming a
    # figure the setup time was never recorded for.
    audit.check("mixer: GPU-hours of measured step time", 0.0364, step_hours, "§10 table")
    runs_total += len(measured)
    hours_total += step_hours

    audit.check("§10 total: runs", 196, runs_total, "§10 table")
    audit.check("§10 total: GPU-hours completed", 20.121, hours_total, "§10 table")
    # The cap is part of the record's pre-registered budget, so the total landing just
    # past it is a claim rather than a rounding detail: v4 was declared "the last arm
    # that fits" against a projection of 3.3 h and cost 3.499, which is the overrun.
    audit.check(
        "§10 total: past the declared 20 h cap",
        True,
        hours_total > 20.0,
        "§10",
        "16.622 completed + v4's measured 3.499",
    )


def check_horizon_table(root: Path, audit: Audit) -> None:
    """§4's spread-and-noise table on the one matched set that ran three lengths."""
    results = arm_results(root, "arch-zoo-100m-v2") + arm_results(
        root, "arch-zoo-100m-v2-horizon"
    )
    by_length = spread_by_length(results)
    documented = {
        400: (0.5457, 0.0286, 19.09),
        800: (0.3244, 0.0165, 19.71),
        1200: (0.2462, 0.0148, 16.59),
    }
    for steps, (spread, std, sigmas) in documented.items():
        row = by_length[str(steps)]
        audit.check(f"§4 {steps}: spread", spread, row["spread"], "§4 table")
        audit.check(f"§4 {steps}: seed std", std, row["within_seed_std"], "§4 table")
        audit.check(f"§4 {steps}: spread in noise units", sigmas, row["sigmas"], "§4 table")
    decay = by_length["decay"]
    audit.check("§4 decay: spread shrink 400->1200", 2.22, decay["spread_shrink"], "§4")
    audit.check("§4 decay: per doubling", 1.65, decay["per_doubling"], "§4")
    audit.check(
        "§4 decay: same winner at every length",
        True,
        decay["best_at_every_length"],
        "§4",
    )
    # The measured stds behind the promotion threshold, which are the same numbers.
    for steps, (_, std, _) in documented.items():
        if steps in SCREEN_SEED_STD_BY_STEPS:
            audit.check(
                f"threshold input {steps}: std",
                std,
                SCREEN_SEED_STD_BY_STEPS[steps],
                "arch_zoo_screen.SCREEN_SEED_STD_BY_STEPS",
            )
    audit.check(
        "promotion threshold at 2 seeds / 400 steps",
        0.0286,
        comparison_noise_nats(2, 400),
        "§4",
        "std * sqrt(2 / seeds)",
    )
    audit.check(
        "promotion threshold at 3 seeds / 400 steps",
        0.02335,
        comparison_noise_nats(3, 400),
        "§4",
    )


def check_v1_screening(root: Path, audit: Audit) -> None:
    """§4a's screening claims: the decay and the wrong winner, both from v1."""
    results = arm_results(root, "arch-zoo-100m-v1")
    by_length = spread_by_length(results)
    audit.check("§4a v1: spread shrink 400->1200", 3.7115, by_length["decay"]["spread_shrink"], "§4a")

    def mean_at(steps: int) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for entry in results:
            if entry.steps == steps:
                grouped.setdefault(entry.name, []).append(entry.validation_loss)
        return {name: statistics.fmean(values) for name, values in grouped.items()}

    short, long = mean_at(400), mean_at(1200)
    common = sorted(set(short) & set(long))
    if len(common) >= 2:
        short_spread = max(short[n] for n in common) - min(short[n] for n in common)
        long_spread = max(long[n] for n in common) - min(long[n] for n in common)
        audit.check("§4a v1: 400->1200 spread ratio", 3.71, short_spread / long_spread, "§4a")

    ranked_short = rank([r for r in results if r.steps == 400], 400)
    ranked_long = rank([r for r in results if r.steps == 1200], 1200)
    if ranked_short and ranked_long:
        winner = ranked_long[0][0]
        order = [name for name, _, _ in ranked_short]
        position = order.index(winner) + 1 if winner in order else None
        gap = dict((n, m) for n, m, _ in ranked_short)[winner] - ranked_short[0][1]
        audit.check(
            "§4a: the eventual winner's place at 400 steps", 2, position, "§4a"
        )
        # The record states this at one significant figure ("by 0.009 nats"), so the
        # tolerance has to absorb that rounding - 5% of 0.009 - rather than pretend the
        # documented value carries three digits.
        audit.check(
            "§4a: its gap to the 400-step winner",
            0.009,
            gap,
            "§4a",
            tolerance=0.05,
        )
        # The claim's basis: a quarter of that screen's own seed std.
        stds = [
            statistics.stdev(values)
            for values in (
                [r.validation_loss for r in results if r.steps == 400 and r.name == name]
                for name in set(short)
            )
            if len(values) >= 2
        ]
        audit.check(
            "§4a: that screen's per-architecture seed std",
            0.0390,
            statistics.fmean(stds),
            "§4",
        )


def pooled_confirmation_results(root: Path):
    """The six-seed design: the crossed arm's ledgers plus its confirmation arm's.

    Pooling is a concatenation of ledgers, not a seed list handed to one arm: passing
    the six seeds while reading one directory silently reports the three-seed result
    under a six-seed label, which is how this audit first disagreed with its own
    record and is exactly the class of mistake it exists to catch.
    """
    return arm_results(root, "arch-zoo-100m-v3-crossed") + arm_results(
        root, "arch-zoo-100m-v3-crossed-confirm"
    )


def check_attribution(root: Path, audit: Audit, *, quick: bool) -> None:
    """§2c: the factorial's effects on three seeds, six, and the fresh three."""

    def crossed_from(results, seeds=None) -> dict:
        return crossed_contrast(
            results, 1200, seeds=seeds or sorted({r.seed for r in results})
        )

    def crossed(name: str, seeds=None) -> dict:
        return crossed_from(arm_results(root, name), seeds)

    def main_effect(contrast: dict, factor: str) -> dict:
        return contrast["main"][factor]

    def interaction(contrast: dict) -> dict:
        # Reported per factor and identical for both, since a two-level interaction has
        # one size; read off the first factor so the audit does not assume which.
        return contrast["main"][contrast["factor_a"]]

    three = crossed("arch-zoo-100m-v3-crossed")
    audit.check("§2c v3: hidden_size", -0.1836, main_effect(three, "hidden_size")["nats"], "§2c")
    audit.check("§2c v3: depth", 0.0783, main_effect(three, "depth")["nats"], "§2c")
    audit.check(
        "§2c v3: exact p at the 46656-assignment floor",
        4.287e-05,
        main_effect(three, "hidden_size")["p"],
        "§2c",
        "2 / 46656",
    )
    audit.check(
        "§2c v3: main effects at the floor",
        True,
        main_effect(three, "hidden_size")["at_floor"],
        "§2c",
    )
    audit.check(
        "§2c v3: interaction at its 3-seed floor",
        0.25,
        interaction(three)["interaction_p_seed_blocked"],
        "§2c",
        "2 / 2^3, which is why three seeds could not resolve it",
    )

    fresh = crossed("arch-zoo-100m-v3-crossed-confirm")
    audit.check("§2c replication: hidden_size", -0.1851, main_effect(fresh, "hidden_size")["nats"], "§2c")
    audit.check("§2c replication: depth", 0.0865, main_effect(fresh, "depth")["nats"], "§2c")

    pooled_results = pooled_confirmation_results(root)
    pooled = crossed_from(pooled_results)
    audit.check("§2c pooled: hidden_size", -0.1843, main_effect(pooled, "hidden_size")["nats"], "§2c")
    audit.check("§2c pooled: depth", 0.0824, main_effect(pooled, "depth")["nats"], "§2c")
    pooled_interaction = interaction(pooled)
    audit.check("§2c pooled: interaction size", 0.0827, pooled_interaction["interaction_nats"], "§2c")
    audit.check(
        "§2c pooled: interaction at its 6-seed floor",
        0.03125,
        pooled_interaction["interaction_p_seed_blocked"],
        "§2c",
        "2 / 2^6",
    )
    contrasts = pooled_interaction["interaction_seed_contrasts"]
    audit.check(
        "§2c pooled: all six seed contrasts agree in sign",
        True,
        len(contrasts) == 6 and len({value > 0 for value in contrasts}) == 1,
        "§2c",
        f"{len(contrasts)} contrasts",
    )
    audit.check(
        "§2c pooled: runs (the seed count the rule was declared for)",
        48,
        search_cost(pooled_results, 8)["runs_completed"],
        "§2c",
    )


def check_structure_and_budget(root: Path, audit: Audit, *, quick: bool) -> None:
    """README's structural claims, and §11's budget-matched draw."""
    report = json.loads((root / "arch-zoo-100m-v2" / "matching_report.json").read_text())
    rows = report["candidates"]
    active = [row["active_params"] for row in rows]
    audit.check(
        "README: active-parameter range under a matched budget",
        4.51,
        max(active) / min(active),
        "README",
        "the ×4.5 range the active-parameter hypothesis fails across",
    )
    audit.check(
        "README: matched total-parameter band (v2)",
        1.08,
        report["param_matched"]["spread"]["ratio"],
        "§10/README",
    )
    ranked = rank(
        load_results(sorted((root / "arch-zoo-100m-v2").glob("*_seed*_results.jsonl"))),
        1200,
    )
    order = [name for name, _, _ in ranked]
    dense = [
        row["name"] for row in rows if row["active_params"] == row["total_params"]
    ]
    if dense:
        audit.check(
            "README: the fully dense design's rank",
            7,
            order.index(dense[0]) + 1,
            "README",
            dense[0],
        )

    # §11: a band-matched draw is not an equal-compute draw, and the matched seed is.
    curated = set_total_params(root / "arch-zoo-100m-v1" / "zoo_definition.json")
    if quick:
        match = budget_matching_draw(
            target_total=curated, size=13, seeds=range(48), target_params=100_000_000,
            tolerance=0.30,
        )
        audit.check(
            "§11: a draw seed can be budget-matched",
            True,
            abs(match["ratio"] - 1.0) < 0.05,
            "§9a",
            f"best of 48 draws: seed {match['seed']} at {match['ratio']:.4f}x",
        )
        return
    match = budget_matching_draw(
        target_total=curated, size=13, seeds=range(200), target_params=100_000_000,
        tolerance=0.30,
    )
    audit.check(
        "§9a: the budget-matched draw seed", 151, match["seed"], "§9a", f"{match['ratio']:.4f}x"
    )
    audit.check("§9a: its budget ratio", 0.9996, match["ratio"], "§9a")
    dist = match["distribution"]
    audit.check("§9a: median band-matched draw vs the curated total", 1.1208, dist["median"], "§9a")
    audit.check("§9a: p10", 1.0523, dist["p10"], "§9a")
    audit.check("§9a: p90", 1.1840, dist["p90"], "§9a")


def check_head_paired(root: Path, audit: Audit) -> None:
    """§2c's head-count confound, as the arm that ran to break it measured it.

    The crossed arm could not separate residual-stream width from head count, because
    the family fixes ``heads = hidden / 64``. This arm varies head count at a fixed
    hidden size with both cost axes bit-identical inside a pair, so the numbers below
    are the ones the phase's headline recommendation now rests on.
    """
    results = arm_results(root, "arch-zoo-100m-v4-head-paired")
    paired = paired_design_contrast(results, "attention_heads")
    crossed = crossed_contrast(results, 1200)
    hidden = crossed["main"]["hidden_size"]
    heads = crossed["main"]["attention_heads"]

    audit.check(
        "§2c v4: the paired design's factor",
        "attention_heads",
        paired["gene"],
        "§2c",
        "depth fixed, everything else held fixed",
    )
    audit.check("§2c v4: paired units", 12, paired["pairs"], "§2c", "4 pairs x 3 seeds")
    audit.check("§2c v4: head contrast", -0.0177, paired["mean_difference"], "§2c")
    audit.check("§2c v4: head contrast p", 0.1396, paired["p"], "§2c", "exact, 4096 sign vectors")
    audit.check(
        "§2c v4: paired procedure floor",
        4.88e-04,
        paired["min_achievable_p"],
        "§2c",
        "2 / 2^12",
    )
    audit.check(
        "§2c v4: within-pair signs are inconsistent",
        True,
        not paired["sign_consistent"],
        "§2c",
        f"{paired['negative']} negative, {paired['positive']} positive",
    )
    audit.check("§2c v4: negative pairs", 8, paired["negative"], "§2c")
    audit.check("§2c v4: positive pairs", 4, paired["positive"], "§2c")

    audit.check("§2c v4: width contrast", -0.1562, hidden["nats"], "§2c")
    audit.check(
        "§2c v4: width contrast at the 46656-assignment floor",
        True,
        hidden["at_floor"],
        "§2c",
        "4.287e-05",
    )
    audit.check("§2c v4: head contrast in the factorial view", -0.0177, heads["nats"], "§2c")
    audit.check(
        "§2c v4: head contrast non-significant",
        True,
        heads["p"] > 0.05,
        "§2c",
        f"p = {heads['p']:.4f}",
    )
    audit.check(
        "§2c v4: width holds at both head counts",
        True,
        all(row["nats"] < 0 for row in hidden["by_other_level"].values()),
        "§2c",
        f"8: {hidden['by_other_level']['8']['nats']:.4f}, "
        f"16: {hidden['by_other_level']['16']['nats']:.4f}",
    )

    # The lead the arm produces rather than the result it was bought for: the head
    # effect is level-dependent, and at H=896 every one of the six within-pair
    # differences is negative. Both are recorded as a lead because the level split was
    # chosen after seeing the interaction and the seed-blocked test is floored at 0.25.
    by_level = heads["by_other_level"]
    audit.check(
        "§2c v4: head effect at H=512", 0.0109, by_level["512"]["nats"], "§2c"
    )
    audit.check(
        "§2c v4: head effect at H=896", -0.0462, by_level["896"]["nats"], "§2c"
    )
    wide = [
        pair["difference"]
        for pair in paired["per_pair"]
        if pair["base"]["hidden_size"] == 896
    ]
    audit.check(
        "§2c v4: at H=896 every pair favours more heads",
        len(wide),
        sum(1 for value in wide if value < 0),
        "§2c",
        "6 units, so 2 / 2^6 = 0.031 is their exact floor",
    )
    audit.check(
        "§2c v4: at H=896 the mean head effect", -0.0462, statistics.fmean(wide), "§2c"
    )
    audit.check("§2c v4: interaction size", 0.0571, heads["interaction_nats"], "§2c")
    audit.check(
        "§2c v4: interaction at its 3-seed floor",
        0.25,
        heads["interaction_p_seed_blocked"],
        "§2c",
        "2 / 2^3 - why this arm cannot decide it",
    )
    contrasts = heads["interaction_seed_contrasts"]
    audit.check(
        "§2c v4: all three seed contrasts agree in sign",
        True,
        len(contrasts) == 3 and len({value > 0 for value in contrasts}) == 1,
        "§2c",
        f"{len(contrasts)} contrasts",
    )
    # The pre-registered branch is decided by two numbers, and the rest of its clauses
    # are judgements the findings record in prose: this asserts the numeric half.
    audit.check(
        "§2c v4: lands on the pre-registered 'width is the lever' condition",
        True,
        abs(paired["mean_difference"]) < 0.05 and heads["p"] > 0.05,
        "§2c",
        "|head| < 0.05 and non-significant, width at floor",
    )

    ranked = rank(results, 1200)
    audit.check(
        "§2c v4: best architecture",
        "hd2_D10_H896_E16_A6_W128_Hd16",
        ranked[0][0],
        "§2c",
    )
    audit.check("§2c v4: spread", 0.2126, ranked[-1][1] - ranked[0][1], "§2c")
    by_length = spread_by_length(results)["1200"]
    audit.check("§2c v4: seed std", 0.0137, by_length["within_seed_std"], "§2c")
    audit.check("§2c v4: spread in seed-noise units", 15.53, by_length["sigmas"], "§2c")


def check_replication_rule(root: Path, audit: Audit) -> None:
    """The pre-registered rule's branch, computed from the same outcome it renders from."""
    from act.search.architecture_report import best_available_summary

    pooled = pooled_confirmation_results(root)
    outcome = best_available_summary(pooled, sorted({r.seed for r in pooled}))
    rule = replication_decision(outcome)
    audit.check(
        "§2c: pre-registered branch",
        "REPLICATES, INTERACTION RESOLVES",
        rule["label"],
        "§2c",
    )
    audit.check(
        "§2c: the rule's own seed-sign condition is met",
        True,
        all(row["met"] for row in rule["conditions"] if "same sign" in row["condition"]),
        "§2c",
    )
    audit.check("§2c pooled: best architecture", "xlohi_D5_H896_E16_A2_W384", outcome["best"], "§2c")
    audit.check("§2c pooled: spread", 0.3018, outcome["architecture_spread"], "§2c")
    audit.check(
        "§2c pooled: spread in seed-noise units", 15.28, outcome["sigmas"], "§2c", "README headline"
    )


def run_audit(root: Path, *, quick: bool = False) -> Audit:
    audit = Audit()
    check_costs(root, audit)
    check_horizon_table(root, audit)
    check_v1_screening(root, audit)
    check_attribution(root, audit, quick=quick)
    check_structure_and_budget(root, audit, quick=quick)
    check_replication_rule(root, audit)
    check_head_paired(root, audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the phase record against artifacts")
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="skip the 200-draw budget scan, which is the only slow part",
    )
    args = parser.parse_args()

    audit = run_audit(Path(args.artifacts_root), quick=args.quick)
    for claim in audit.claims:
        mark = "ok  " if claim.ok else "FAIL"
        detail = f"  ({claim.detail})" if claim.detail else ""
        print(
            f"  {mark} {claim.what:<58} documented {claim.documented!r:<14} "
            f"computed {claim.computed!r:<14} [{claim.source}]{detail}"
        )
    failures = audit.failures
    print(
        f"\n{len(audit.claims) - len(failures)}/{len(audit.claims)} claims agree with the "
        "artifacts"
    )
    if failures:
        print("\nDisagreements (the record or the artifact is wrong, and which must be decided):")
        for claim in failures:
            print(f"  - {claim.what}: documented {claim.documented!r}, computed {claim.computed!r}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
