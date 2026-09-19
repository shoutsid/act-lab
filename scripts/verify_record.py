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
from dataclasses import dataclass, field, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from act.search.architecture_eval import load_results  # noqa: E402
from act.search.architecture_report import (  # noqa: E402
    FINALIST_ORDER_FRESH_SEEDS,
    HEAD_PAIRING_RULE,
    FINALIST_ORDER_STEPS,
    best_available_summary,
    blocked_permutation_test,
    crossed_contrast,
    finalist_order_decision,
    horizon_decision,
    paired_design_contrast,
    precision_offset,
    rank,
    rank_stability,
    replication_decision,
    search_cost,
    spread_by_length,
    verdict,
    with_structure,
)
from arch_zoo_screen import (  # noqa: E402
    SCREEN_SEED_STD_BY_STEPS,
    comparison_noise_nats,
    seed_std_for_steps,
    seed_std_measured_at_steps,
)
from set_comparison_power import candidate_population, power_at  # noqa: E402
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
        "arch-zoo-100m-v4-head-paired-confirm": (24, 3.511),
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

    audit.check("§10 total: runs", 220, runs_total, "§10 table")
    audit.check("§10 total: GPU-hours completed", 23.632, hours_total, "§10 table")
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


def check_head_paired_confirm(root: Path, audit: Audit) -> None:
    """The head-paired confirmation arm's structural record.

    The arm's contrast numbers land when its sweep does; the checks below are true at
    any point mid-sweep and guard the property the whole arm rests on rather than any
    loss: the candidate set is byte-identical to the parent's, so pooling the two arms
    measures seeds and nothing else. NOTES.md records this was verified by dry run
    before the hours were committed - this makes it a standing check instead of a
    note-time assertion.
    """
    parent_dir = root / "arch-zoo-100m-v4-head-paired"
    confirm_dir = root / "arch-zoo-100m-v4-head-paired-confirm"
    if not confirm_dir.is_dir():
        return

    parent_zoo = json.loads((parent_dir / "zoo_definition.json").read_text())
    confirm_zoo = json.loads((confirm_dir / "zoo_definition.json").read_text())
    audit.check(
        "§2c v4-confirm: candidate set byte-identical to the parent's",
        True,
        parent_zoo["candidates"] == confirm_zoo["candidates"],
        "NOTES.md design integrity",
        "8 candidates; the two definitions differ in experiment_id alone",
    )

    rows = arm_results(root, confirm_dir.name)
    seeds = {51, 52, 53}
    audit.check(
        "§2c v4-confirm: every ledger row is on a fresh seed",
        True,
        all(row.seed in seeds for row in rows),
        "NOTES.md",
        "42-50 were already used to design, screen or score something in this phase",
    )
    audit.check(
        "§2c v4-confirm: the plan is flat at 1200 steps",
        True,
        all(row.steps == 1200 for row in rows),
        "NOTES.md",
        f"{len(rows)} of 24 rows written so far",
    )
    audit.check(
        "§2c v4-confirm: the structural join reaches the paired genes",
        True,
        all(
            row.hidden_size in (512, 896) and row.attention_heads in (8, 16)
            for row in rows
        ),
        "NOTES.md design",
        "hidden_size x attention_heads is the crossing the pooled rule reads",
    )
    if len(rows) < 24:
        return

    # The six-seed record, as the pooled analysis wrote it into the parent's
    # decision.md, and the branch as HEAD_PAIRING_RULE decides it - recomputed here
    # from the two arms' ledgers concatenated, for the same reason as
    # pooled_confirmation_results: a seed list handed to one directory would
    # report the three-seed result under a six-seed label.
    from act.search.architecture_report import (
        best_available_summary,
        head_pairing_decision,
        paired_level_split,
    )

    pooled = arm_results(root, parent_dir.name) + rows
    outcome = best_available_summary(pooled, sorted({r.seed for r in pooled}))
    rule = head_pairing_decision(outcome)
    audit.check(
        "§2c v4 pooled: pre-registered branch",
        "HEAD EFFECT IS REAL AT WIDE WIDTH",
        rule["label"],
        "§2c",
        "HEAD_PAIRING_RULE, all five conditions",
    )
    audit.check(
        "§2c v4 pooled: every rule condition met",
        True,
        all(row["met"] for row in rule["conditions"]) and len(rule["conditions"]) == 5,
        "§2c",
    )

    paired = paired_design_contrast(pooled, "attention_heads")
    audit.check("§2c v4 pooled: paired units", 24, paired["pairs"], "§2c", "4 pairs x 6 seeds")
    audit.check("§2c v4 pooled: head contrast", -0.0234, paired["mean_difference"], "§2c")
    audit.check(
        "§2c v4 pooled: head contrast p",
        0.0115,
        paired["p"],
        "§2c",
        "Monte-Carlo, 2^24 sign vectors past the enumeration cap",
        tolerance=0.05,
    )
    audit.check(
        "§2c v4 pooled: within-pair signs still inconsistent",
        True,
        not paired["sign_consistent"],
        "§2c",
        f"{paired['negative']} negative, {paired['positive']} positive",
    )
    audit.check("§2c v4 pooled: negative pairs", 17, paired["negative"], "§2c")

    split = paired_level_split(pooled, "attention_heads", "hidden_size")["levels"]
    audit.check("§2c v4 pooled: head effect at H=512", 0.0084, split["512"]["mean_difference"], "§2c")
    audit.check(
        "§2c v4 pooled: H=512 head effect non-significant",
        True,
        split["512"]["p"] > 0.05,
        "§2c",
        f"p = {split['512']['p']:.3f}, 12 units",
    )
    audit.check("§2c v4 pooled: head effect at H=896", -0.0553, split["896"]["mean_difference"], "§2c")
    audit.check(
        "§2c v4 pooled: at H=896 every pair favours more heads",
        12,
        split["896"]["negative"],
        "§2c",
        "12 units, so the exact p is at its 2 / 2^12 floor",
    )
    audit.check(
        "§2c v4 pooled: H=896 head effect at the exact floor",
        4.88e-04,
        split["896"]["p"],
        "§2c",
    )

    crossed = crossed_contrast(pooled, 1200)
    hidden = crossed["main"]["hidden_size"]
    heads = crossed["main"]["attention_heads"]
    audit.check("§2c v4 pooled: width contrast", -0.1617, hidden["nats"], "§2c")
    audit.check(
        "§2c v4 pooled: width at the Monte-Carlo resolution",
        True,
        hidden["at_floor"],
        "§2c",
        "5e-06 at 200,000 draws",
    )
    audit.check("§2c v4 pooled: interaction size", 0.0636, heads["interaction_nats"], "§2c")
    audit.check(
        "§2c v4 pooled: interaction at its 6-seed floor",
        0.03125,
        heads["interaction_p_seed_blocked"],
        "§2c",
        "2 / 2^6 - below the declared 0.05, which three seeds could not reach",
    )
    contrasts = heads["interaction_seed_contrasts"]
    audit.check(
        "§2c v4 pooled: all six seed contrasts agree in sign",
        True,
        len(contrasts) == 6 and all(value > 0 for value in contrasts),
        "§2c",
        f"{len(contrasts)} contrasts",
    )
    audit.check("§2c v4 pooled: best architecture", "hd2_D10_H896_E16_A6_W128_Hd16", outcome["best"], "§2c")
    audit.check("§2c v4 pooled: spread", 0.2221, outcome["architecture_spread"], "§2c")
    audit.check("§2c v4 pooled: spread in seed-noise units", 13.56, outcome["sigmas"], "§2c")

    # The robustness check v2's, the crossed arm's and the horizon arm's all carry, run
    # here for the first time - and on this arm it runs the other way, because the arm's
    # recorded headline is a *negative*: "head count is free". The like-for-like pair is
    # held-out loss against final training loss, since both are loss at the end of
    # training; `mean_train_loss` averages over the whole trajectory, so it moves with
    # convergence speed and answers a different question. The rule reads signs and exact
    # re-split statistics rather than the sampled ones, so the calls below run at 20,000
    # draws - verified to return each branch unchanged from the record's 200,000, and
    # 4x cheaper than paying the record's resolution to re-derive a label.
    from act.search.architecture_report import paired_level_split

    pooled_seeds = sorted({row.seed for row in pooled})
    for field, label in (("final_train_loss", "final"), ("mean_train_loss", "mean")):
        rows = [replace(entry, validation_loss=getattr(entry, field)) for entry in pooled]
        crossed_reading = crossed_contrast(rows, 1200, samples=20_000)["main"]
        hidden_reading = crossed_reading["hidden_size"]
        level = paired_level_split(rows, "attention_heads", "hidden_size")["levels"]["896"]
        audit.check(
            f"§2c v4 pooled: width attribution holds on {label} train loss",
            True,
            hidden_reading["nats"] < 0 and hidden_reading["at_floor"],
            "§2c",
            f"{hidden_reading['nats']:+.4f} nats at floor - the lever the plan spends on",
        )
        audit.check(
            f"§2c v4 pooled: H=896 head effect holds on {label} train loss",
            12,
            level["negative"],
            "§2c",
            f"{level['mean_difference']:+.4f} nats, every one of the 12 units favouring more heads",
        )
        reading = head_pairing_decision(
            best_available_summary(rows, pooled_seeds, samples=20_000)
        )
        audit.check(
            f"§2c v4 pooled: the rule's branch on {label} train loss",
            "HEAD EFFECT IS REAL AT WIDE WIDTH"
            if field == "final_train_loss"
            else "INTERACTION ABSENT - THE THREE-SEED AGREEMENT WAS SEED LUCK",
            reading["label"],
            "§2c",
            "stable on the like-for-like pair"
            if field == "final_train_loss"
            else "one per-seed interaction contrast flips sign here, which the record names",
        )

    audit.check(
        "§2c v4 pooled: the head contrast sits inside 2x the rule's own threshold",
        True,
        HEAD_PAIRING_RULE["main_p"] < paired["p"] < 2 * HEAD_PAIRING_RULE["main_p"],
        "§2c",
        f"p = {paired['p']:.5f} against a declared {HEAD_PAIRING_RULE['main_p']}, a margin of "
        f"{paired['p'] / HEAD_PAIRING_RULE['main_p']:.2f}x - not the 0.05 a reader applies",
    )


def check_finalist_order(root: Path, audit: Audit) -> None:
    """§9's open item, as the arm that closes it measures it.

    ``arch-zoo-100m-v1`` closed on "NOT ESTABLISHED - this experiment cannot yet
    tell": a 3-block exact test whose floor is 1/36, on finalists that were promoted
    *on those same seeds*, so it is mildly anti-conservative in the winner's favour.
    v1 named its own fix, and this arm is that fix and nothing else - no new
    architectures, no new length, no retuned knob. The checks below split the way the
    head-paired confirmation arm's do: the design integrity that has to hold at any
    point mid-run, then the pooled six-block record once its nine rows are there.
    """
    parent_dir = root / "arch-zoo-100m-v1"
    arm_dir = root / "arch-zoo-100m-v1-finalists-6seed"
    if not arm_dir.is_dir():
        return

    parent_zoo = json.loads((parent_dir / "zoo_definition.json").read_text())
    arm_zoo = json.loads((arm_dir / "zoo_definition.json").read_text())
    parent_by_name = {c["name"]: c for c in parent_zoo["candidates"]}
    arm_candidates = {c["name"]: c for c in arm_zoo["candidates"]}

    # The finalists are read off v1's own ledgers rather than restated here, so this
    # check cannot drift from the promotion it is claiming to extend.
    parent_rows = arm_results(root, parent_dir.name)
    finalists = sorted({row.name for row in parent_rows if row.steps == FINALIST_ORDER_STEPS})
    audit.check(
        "§9 finalists: the arm runs exactly v1's promoted finalists",
        finalists,
        sorted(arm_candidates),
        "NOTES.md design",
        f"{len(finalists)} architectures, the ones v1 promoted at {FINALIST_ORDER_STEPS} steps",
    )
    audit.check(
        "§9 finalists: candidate genomes byte-identical to the record's",
        True,
        bool(arm_candidates)
        and all(parent_by_name.get(name) == c for name, c in arm_candidates.items()),
        "NOTES.md design integrity",
        "the two zoo definitions differ in experiment_id alone, so pooling the arms "
        "measures seeds and nothing else",
    )

    rows = arm_results(root, arm_dir.name)
    fresh = set(FINALIST_ORDER_FRESH_SEEDS)
    audit.check(
        "§9 finalists: every ledger row is on a screening-free seed",
        True,
        all(row.seed in fresh for row in rows),
        "NOTES.md",
        "42-44 screened and promoted these finalists, so a fresh block is the point",
    )
    audit.check(
        "§9 finalists: the plan is flat at the rule's declared length",
        True,
        all(row.steps == FINALIST_ORDER_STEPS for row in rows),
        "NOTES.md",
        f"{len(rows)} of 9 rows written so far",
    )
    audit.check(
        "§9 finalists: the validation stream is the record's, not a re-split one",
        True,
        bool(rows) and all(row.val_source == parent_rows[0].val_source for row in rows),
        "NOTES.md",
        "val-tokens.npy is shorter than this schedule needs, so both arms hold the "
        f"same 2M tokens out of the same stream: {parent_rows[0].val_source}",
    )

    # The arm is only poolable because a row without run_config.mixed_precision is
    # fp32 by construction - the field arrived with autocast in 07fc36a and v1
    # predates it. Reading that absence as "unknown" refused the pool this arm exists
    # to make, which is how the guard came to be corrected; the check keeps the two
    # arms at one precision rather than trusting that they are.
    def _precisions(directory: Path) -> set[str]:
        values: set[str] = set()
        for ledger in sorted(directory.glob("*_seed*_results.jsonl")):
            for line in ledger.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    recorded = (row.get("run_config") or {}).get("mixed_precision")
                    values.add(str(recorded) if recorded else "off")
        return values

    audit.check(
        "§9 finalists: both arms' rows read as a single precision",
        {"off"},
        _precisions(parent_dir) | _precisions(arm_dir),
        "NOTES.md",
        "v1's rows carry no mixed_precision key at all (they predate 07fc36a) and are "
        "fp32 by construction; the rule's two readings are only comparable if the "
        "fresh blocks were trained the same way",
    )

    if len(rows) < 9:
        return

    # --- the measured record -------------------------------------------------
    # Recomputed with the same functions the decision renders from, so the branch
    # below cannot be transcribed by hand: the documented value is the one the
    # arm's decision.md carries. The six-block enumeration is exact (6^6 = 46,656
    # under the cap), so these are reproducible rather than sampled.
    pooled_rows = parent_rows + rows
    pooled_test = blocked_permutation_test(pooled_rows, FINALIST_ORDER_STEPS)
    outcome = {
        "steps": FINALIST_ORDER_STEPS,
        "lengths_present": [FINALIST_ORDER_STEPS],
        "blocked_test": pooled_test,
    }
    rule = finalist_order_decision(outcome, pooled_rows)
    audit.check(
        "§9 finalists: the pre-registered branch",
        "order-coupled",
        rule["branch"],
        "v1 decision.md",
        "the pooled test clears the threshold while the screening-free blocks "
        "neither separate the finalists nor agree on a winner, so the significance "
        "is the blocks that selected them",
    )
    audit.check("§9 finalists: pooled six-block p", 0.001286, rule["p_all"], "v1 decision.md")
    audit.check(
        "§9 finalists: pooled six-block floor", 0.0001286, rule["pooled_floor"],
        "v1 decision.md", "1/7776, against v1's 1/36",
    )
    audit.check("§9 finalists: screening-free p", 0.1111, rule["p_fresh"], "v1 decision.md")
    audit.check(
        "§9 finalists: screening-free floor", 0.02778, rule["fresh_floor"],
        "v1 decision.md", "1/36 over the three uncoupled blocks",
    )
    audit.check(
        "§9 finalists: condition results, in the declared order",
        [True, False, False],
        [row["met"] for row in rule["conditions"]],
        "v1 decision.md",
        "pooled clears, screening-free does not, and the top does not replicate",
    )
    audit.check(
        "§9 finalists: architectures finishing first somewhere in the six blocks",
        2,
        len(set(rule["block_tops"])),
        "v1 decision.md",
        f"block tops: {sorted(set(rule['block_tops']))}",
    )

    # The split check the rest of the record carries, on the rule that closed §9's open
    # item - the branch has to survive the loss it is read on, or the instruction to
    # carry the effect size but not the ranking does not. It does, on all three readings.
    # One of the three conditions the rule transcribes does not, and that is exactly the
    # clause the branch's explanation leans on: whether the screening-free blocks alone
    # clear the threshold. Both facts are checked, because recording only the first
    # would leave the reasoning behind the branch unguarded.
    for field, label, expected_p_all, expected_p_fresh in (
        ("final_train_loss", "final train loss", 82 / 7776, 2 / 36),
        ("mean_train_loss", "the trajectory average", 2 / 7776, 1 / 36),
    ):
        on_loss = [
            replace(entry, validation_loss=getattr(entry, field)) for entry in pooled_rows
        ]
        reading = finalist_order_decision(
            {
                "steps": FINALIST_ORDER_STEPS,
                "lengths_present": [FINALIST_ORDER_STEPS],
                "blocked_test": blocked_permutation_test(on_loss, FINALIST_ORDER_STEPS),
            },
            on_loss,
        )
        audit.check(
            f"§9 finalists: the branch holds on {label}",
            "order-coupled",
            reading["branch"],
            "v1 decision.md",
            "the third condition decides it in every reading - two winners top the six blocks",
        )
        audit.check(
            f"§9 finalists: pooled six-block p on {label}",
            expected_p_all,
            reading["p_all"],
            "v1 decision.md",
        )
        audit.check(
            f"§9 finalists: screening-free p on {label}",
            expected_p_fresh,
            reading["p_fresh"],
            "v1 decision.md",
            "above the declared 0.05 on the like-for-like pair, the three-block floor beyond it",
        )

    # The verdict must not read the rule's *presence* as its *result*. It did: the
    # pooled p - the very number the rule reads as the screening seeds - produced
    # "the finalists having already been confirmed on seeds never used in
    # screening" at the top of the artifact, directly above a section saying no
    # ranking is recommended. These three claims are what keep that from returning.
    rendered = best_available_summary(
        pooled_rows, sorted({row.seed for row in pooled_rows}), samples=2_000
    )
    rendered["finalist_order_decision"] = finalist_order_decision(rendered, pooled_rows)
    headline, action = verdict(rendered)
    audit.check(
        "§9 finalists: the headline names which blocks established the effect",
        True,
        "SCREENING-FREE" in headline,
        "v1 decision.md headline",
        "the pooled label alone would repeat the over-claim the arm exists to rule out",
    )
    audit.check(
        "§9 finalists: the action does not claim a confirmed ordering",
        True,
        "already been confirmed" not in action,
        "v1 decision.md",
    )
    audit.check(
        "§9 finalists: the action carries no ranking into the next phase",
        True,
        "not a ranking" in action,
        "v1 decision.md",
    )


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


def check_power_study(root: Path, audit: Audit, *, quick: bool) -> None:
    """§9a's size condition: what the affordable §11 comparison can and cannot see.

    The record's fourth precondition is a power figure, not a measured loss, so it is
    recomputed from the same population and noise the study reads out of the ledgers.
    The power *rates* are checked as orderings rather than as decimals: they are
    deterministic given the seeds, but the record quotes them to two places and a
    relative tolerance on a 0.04 rate is not a meaningful test.
    """
    population, noise = candidate_population(root, "arch-zoo-100m-v1", 400)
    audit.check(
        "§9a: the candidate spread the power is computed from",
        0.7411,
        max(population) - min(population),
        "§9a",
    )
    audit.check("§9a: the per-run seed std it is computed against", 0.0390, noise, "§9a")

    # Every power figure the check reads, computed up front and in parallel. Each
    # call is an independent Monte-Carlo run seeded inside ``power_at`` (seed=0), so
    # the numbers are identical to calling them one by one - only the wall clock
    # changes: this check was 27 s of the audit's full mode, bounded by nothing
    # but the serial loop.
    specs = [(0.0, 13, 2), (0.30, 13, 2), (0.20, 13, 2)]
    if not quick:
        specs += [(0.15, 26, 2), (0.10, 26, 2), (0.10, 13, 2), (0.10, 13, 3), (0.10, 13, 4)]
    powers = _powers_in_parallel(population, noise, specs)

    def power(advantage: float, *, set_size: int = 13, blocks: int = 2) -> float:
        return powers[(advantage, set_size, blocks)]

    audit.check(
        "§9a: 13 per set rejects a true null only at the nominal rate",
        True,
        power(0.0) < 0.08,
        "§9a",
        "the power table's own null row",
    )
    audit.check(
        "§9a: 80% power is reached at 0.30 nats, and not at 0.20",
        True,
        power(0.30) >= 0.8 and power(0.20) < 0.8,
        "§9a",
    )
    if quick:
        return
    audit.check(
        "§9a: 26 per set reaches 80% power at 0.155 nats",
        True,
        power(0.15, set_size=26) > 0.7 and power(0.10, set_size=26) < 0.5,
        "§9a",
    )
    audit.check(
        "§9a: more seeds are not the lever",
        True,
        all(abs(power(0.10, blocks=blocks) - power(0.10)) < 0.08 for blocks in (3, 4)),
        "§9a",
        "2, 3 and 4 seeds agree at a 0.10-nat advantage",
    )


def _power_for(args: tuple) -> tuple[tuple, float]:
    population, noise, (advantage, set_size, blocks) = args
    return (advantage, set_size, blocks), power_at(
        population,
        advantage=advantage,
        model="shift-all",
        statistic="mean",
        set_size=set_size,
        blocks=blocks,
        noise=noise,
        replicates=200,
        samples=2_000,
    )


def _powers_in_parallel(
    population: list[float], noise: float, specs: list[tuple]
) -> dict[tuple, float]:
    import concurrent.futures

    jobs = [(population, noise, spec) for spec in specs]
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        return dict(pool.map(_power_for, jobs))


def check_horizon_10x(root: Path, audit: Audit) -> None:
    """Does the crossed attribution survive 10x the training?

    The arm re-trains the crossed factorial's *byte-identical* candidate set at 12,000
    steps instead of 1,200 on two seeds, so this is the one place in the record where
    an effect is measured at a length a real training budget would occupy. It is read
    by ``horizon_decision``, and every number below is recomputed from its own ledgers
    rather than quoted from the decision it produced.

    The preconditions come first: the rule is declared on the 12,000-step rows while a
    1,200-step anchor sits in the same directory, so an arm that lost its horizon and
    kept its anchor would otherwise have every claim below silently re-read at the
    wrong length - which is what the arm's own first attempt did.
    """
    results = arm_results(root, "arch-zoo-100m-v3-crossed-horizon10x")
    audit.check("horizon: runs (8 candidates x 2 lengths x 2 seeds)", 32, len(results), "arm ledgers")
    audit.check("horizon: seeds", [42, 43], sorted({r.seed for r in results}), "arm ledgers")
    audit.check("horizon: lengths", [1200, 12000], sorted({r.steps for r in results}), "arm ledgers")

    by_length = spread_by_length(results)
    documented = {
        1200: (0.3085, 0.0118, 26.16),
        12000: (0.1059, 0.0056, 18.88),
    }
    for steps, (spread, std, sigmas) in documented.items():
        row = by_length[str(steps)]
        audit.check(f"§3 horizon {steps}: spread", spread, row["spread"], "decision.md")
        audit.check(f"§3 horizon {steps}: seed std", std, row["within_seed_std"], "decision.md")
        audit.check(f"§3 horizon {steps}: spread in noise units", sigmas, row["sigmas"], "decision.md")

    # §3's decay is fitted over 400->1200. This is the same quantity at 10x the
    # horizon, and the reading is that it decays *slower* than §3 extrapolates - so
    # the extrapolation is an over-estimate of how fast an effect fades, which is the
    # direction that matters for anyone sizing a budget from it.
    decay = by_length["decay"]
    audit.check("§3 horizon: spread shrink 1200->12000", 2.91, decay["spread_shrink"], "decision.md")
    audit.check("§3 horizon: spread shrink per doubling", 1.38, decay["per_doubling"], "decision.md")
    audit.check(
        "§3 horizon: decays slower than §3's 1.65 per doubling",
        True,
        decay["per_doubling"] < 1.65,
        "§3",
        "the 1200-step decay rate extrapolates to more decay than 10x training shows",
    )
    # The phase-relevant reading is not the spread in nats but its size against seed
    # noise: at 1200 steps this arm's spread is 26x its own seed std and at 12,000 it
    # is still 18.9x, so the architecture effect does not fade into noise.
    audit.check(
        "§3 horizon: still far above seed noise, so it does not wash out",
        True,
        decay["sigmas_at_last"] > 5.0,
        "decision.md",
        f"{decay['sigmas_at_first']:.2f} -> {decay['sigmas_at_last']:.2f} seed-noise units",
    )

    seeds = sorted({r.seed for r in results})
    decision = horizon_decision(best_available_summary(results, seeds))
    audit.check("§2c horizon: the rule reaches a branch", True, bool(decision.get("available")), "HORIZON_RULE")
    audit.check("§2c horizon: branch", "shallow-persists", decision.get("branch"), "HORIZON_RULE")

    contrast = crossed_contrast(results, 12000, seeds=seeds)
    depth = contrast["main"]["depth"]
    width = contrast["main"]["hidden_size"]
    audit.check("§2c horizon: depth main effect", 0.0209, depth["nats"], "decision.md")
    audit.check("§2c horizon: depth p (exact over 1296 assignments)", 0.003086, depth["p"], "decision.md")
    audit.check(
        "§2c horizon: depth keeps its sign (shallow still wins)",
        True,
        depth["nats"] > 0,
        "decision.md",
        "the finding the arm exists to test",
    )
    audit.check("§2c horizon: hidden_size main effect", -0.0645, width["nats"], "decision.md")
    audit.check("§2c horizon: hidden_size p (at the 1296-assignment floor)", 0.001543, width["p"], "decision.md")
    audit.check("§2c horizon: hidden_size at the design floor", True, width["at_floor"], "decision.md")
    audit.check("§2c horizon: interaction size", -0.0090, depth["interaction_nats"], "decision.md")
    audit.check(
        "§2c horizon: interaction at its 2-seed floor",
        0.5,
        depth["interaction_p_seed_blocked"],
        "decision.md",
        "2 / 2^2, so two seeds can neither confirm nor exclude it",
    )
    audit.check(
        "§2c horizon: every seed agrees on the interaction sign",
        True,
        len({value > 0 for value in depth["interaction_seed_contrasts"]}) == 1,
        "decision.md",
    )

    # The robustness story the crossed arm states has to hold at 10x the horizon too,
    # and the split check matters *more* here: a longer schedule is where a difference
    # in how much each architecture overfits would widen. Both main effects keep their
    # signs on training loss and stay significant, so the attribution is about the
    # optimisation rather than about where the validation tokens were cut.
    as_train = crossed_contrast(
        [replace(entry, validation_loss=entry.final_train_loss) for entry in results],
        12000,
    )["main"]
    for gene, held_out in (("depth", depth), ("hidden_size", width)):
        audit.check(
            f"§2c horizon: {gene} not a holdout-split artifact (training loss)",
            True,
            as_train[gene]["p"] < 0.01 and (as_train[gene]["nats"] > 0) == (held_out["nats"] > 0),
            "decision.md",
            f"on final train loss {as_train[gene]['nats']:+.4f} nats at p={as_train[gene]['p']:.4g}",
        )

    # One further check the horizon arm can carry and one it cannot, and the record
    # states both: the spread is not one candidate's doing, while the crossed
    # attribution is not checkable this way at 2 candidates per cell. The refusal is a
    # supported outcome, so it is checked as one rather than left unmentioned.
    names = sorted({entry.name for entry in results})
    leave_one_out = [
        [entry for entry in results if entry.name != dropped] for dropped in names
    ]
    worst = max(
        blocked_permutation_test(reduced, 12000, samples=20_000)["p"]
        for reduced in leave_one_out
    )
    audit.check(
        "§2c horizon: spread survives dropping any one architecture",
        True,
        worst < 0.01,
        "§8",
        f"worst leave-one-out p={worst:.4g} over 20k draws, against the 0.05 threshold",
    )
    audit.check(
        "§2c horizon: the crossed attribution refuses leave-one-out on every candidate",
        len(names),
        sum(
            1
            for reduced in leave_one_out
            if not crossed_contrast(reduced, 12000)["available"]
        ),
        "decision.md",
        "7 rows cannot fill four cells balanced on both seeds",
    )

    # Two seeds is arithmetic here rather than caution: the re-split test is
    # two-sided, so one seed's 36 assignments floor its p at 2/36 = 0.0556 - above the
    # 0.05 threshold. A single-seed record of this arm could not have stated a verdict.
    single = crossed_contrast(results, 12000, seeds=[seeds[0]])
    audit.check(
        "§2c horizon: one seed's re-split assignments",
        36,
        single["main"]["depth"]["assignments"],
        "HORIZON_RULE",
    )
    audit.check(
        "§2c horizon: one seed's p floor is above the 0.05 threshold",
        2 / 36,
        single["main"]["depth"]["min_achievable_p"],
        "HORIZON_RULE",
        "so a single seed cannot reach significance in this design at all",
    )

    ordering = rank_stability(results)[0]
    # The record states this to two places ("0.90"), so the tolerance absorbs that
    # rounding - half a unit in the last place, 0.006 against 0.9 - rather than
    # pretend the documented value carries a fourth significant figure.
    audit.check(
        "§3 horizon: Spearman 1200->12000",
        0.90,
        ordering["spearman"],
        "decision.md",
        tolerance=0.006,
    )
    audit.check(
        "§3 horizon: the winner is NOT preserved across 10x",
        False,
        bool(ordering["winner_preserved"]),
        "decision.md",
        f"{ordering['best_at_from']} -> {ordering['best_at_to']}",
    )

    # The precision offset is what licenses reading fp16 rows as evidence about
    # architecture at all, so it is checked against the size of the effect it carries
    # rather than only against its own value.
    offset = precision_offset(
        results,
        arm_results(root, "arch-zoo-100m-v3-crossed"),
        steps=1200,
        reference_label="arch-zoo-100m-v3-crossed",
    )
    audit.check("§3 horizon: fp16 vs fp32 offset, mean", -0.0030, offset["mean_delta"], "decision.md")
    audit.check(
        "§3 horizon: the offset is smaller than the effect it carries",
        True,
        abs(offset["mean_delta"]) < abs(depth["nats"]),
        "decision.md",
        "otherwise fp16 would be measuring its own arithmetic",
    )

    # The search program confirms finalists at the horizon it trains for, and this arm
    # is where that horizon's noise was measured - so the promotion threshold a 12,000-
    # step confirmation would use is checkable here. It borrows the 1,200-step std
    # (the table of measured stds ends there), which over-estimates this arm's measured
    # noise at that length by a factor the record quotes. The factor is computed from
    # the arm's own rows rather than from a printed decimal, so the claim cannot drift
    # away from the measurement it is about.
    audit.check(
        "§9 horizon: a 12,000-step threshold borrows the 1,200-step std",
        1200,
        seed_std_measured_at_steps(12000),
        "§9",
        "the measured-std table ends at 1,200 steps",
    )
    audit.check(
        "§9 horizon: that borrow over-estimates the measured noise by",
        2.64,
        seed_std_for_steps(12000) / by_length["12000"]["within_seed_std"],
        "§9",
        "safe for a promotion decision, costly in seeds",
    )

    # A phase of its own: the arm's NOTES opens a horizon phase with its own cap, on
    # Kaggle rather than the local card, so this is deliberately *not* folded into
    # §10's closed 23.632 total - the overrun it did not extend.
    cost = search_cost(results, 8)
    audit.check("horizon phase: runs", 32, cost["runs_completed"], "arm NOTES")
    audit.check("horizon phase: GPU-hours", 9.22, cost["wall_clock_hours"], "arm NOTES")
    audit.check(
        "horizon phase: inside its own 15 GPU-h cap",
        True,
        cost["wall_clock_hours"] < 15.0,
        "arm NOTES",
        "a separate cap, so it does not extend §10's overrun by drift",
    )

    # The search program's last step, priced. The loop §10 quotes confirms its
    # finalists at 1,200 steps - the length this phase measured at - while §9
    # specifies the horizon the search actually trains for, which no arm had ever run
    # and so had never been priced. Both lengths here are measured on one design, and
    # the per-run cost is linear in steps between them, which is what licenses the
    # local extrapolation rather than assuming it.
    stages = cost["by_stage"]
    per_run_1200 = stages["1200"]["elapsed_hours"] / stages["1200"]["runs"]
    per_run_12000 = stages["12000"]["elapsed_hours"] / stages["12000"]["runs"]
    audit.check("§10 horizon: a 1,200-step run costs (Kaggle fp16)", 0.0523, per_run_1200, "arm ledgers")
    audit.check("§10 horizon: a 12,000-step run costs (Kaggle fp16)", 0.524, per_run_12000, "arm ledgers")
    audit.check(
        "§10 horizon: cost is linear in steps over 1,200->12,000",
        10.016,
        per_run_12000 / per_run_1200,
        "arm ledgers",
        "10x the steps for 10.016x the time, which is what licenses the local extrapolation",
    )
    finalists = 8 * 3
    local_per_run_12000 = 0.133 * 10
    audit.check(
        "§10 loop: confirming 8 finalists x 3 seeds at 12,000, Kaggle fp16",
        12.57,
        finalists * per_run_12000,
        "§10",
        "the step the loop's quoted 17 GPU-hours leaves out",
    )
    audit.check(
        "§10 loop: the same confirmation on the local card (extrapolated)",
        31.92,
        finalists * local_per_run_12000,
        "§10",
    )
    audit.check(
        "§10 loop: screening plus a horizon confirmation, on Kaggle",
        29.56,
        100 * 3 * 0.046 + 8 * 3 * 0.133 + finalists * per_run_12000,
        "§10",
    )
    audit.check(
        "§10 loop: the platform ratio on the step that dominates it",
        2.54,
        local_per_run_12000 / per_run_12000,
        "§10",
    )


#: The audit's checks in record order. Each is a pure function of the artifacts root
#: that appends to the audit it is handed, so they can run in separate processes and
#: have their claims concatenated in this order - the output is identical to running
#: them one after another, only the wall clock changes (measured: 63 s -> ~28 s in
#: full mode, 27 s -> ~8 s in quick mode, bounded by the slowest check).
CHECKS: tuple[str, ...] = (
    "check_costs",
    "check_horizon_table",
    "check_v1_screening",
    "check_attribution",
    "check_structure_and_budget",
    "check_power_study",
    "check_replication_rule",
    "check_head_paired",
    "check_head_paired_confirm",
    "check_finalist_order",
    "check_horizon_10x",
)


def _run_check(name: str, root: str, quick: bool) -> list[Claim]:
    """One check in its own audit, for a worker process."""
    import inspect

    check = globals()[name]
    audit = Audit()
    kwargs = {"quick": quick} if "quick" in inspect.signature(check).parameters else {}
    check(Path(root), audit, **kwargs)
    return audit.claims


def run_audit(root: Path, *, quick: bool = False, parallel: bool = True) -> Audit:
    """Every check, claims in record order.

    ``parallel`` runs the checks in a process pool; it is what makes the audit
    cheap enough to sit in the test suite. Set it off to debug a single check's
    traceback in-process.
    """
    audit = Audit()
    if not parallel:
        for name in CHECKS:
            audit.claims.extend(_run_check(name, str(root), quick))
        return audit
    import concurrent.futures

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(CHECKS)) as pool:
        futures = [pool.submit(_run_check, name, str(root), quick) for name in CHECKS]
        for future in futures:
            audit.claims.extend(future.result())
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the phase record against artifacts")
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="skip the 200-draw budget scan and the power study's sampling",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="run the checks one after another in this process (for debugging)",
    )
    args = parser.parse_args()

    audit = run_audit(Path(args.artifacts_root), quick=args.quick, parallel=not args.serial)
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
