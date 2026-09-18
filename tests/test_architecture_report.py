"""Tests for architecture-search reporting and the decision rule.

The decision rule is the part that tells the project whether to spend more GPU
hours, so it is tested for the ways it could over-claim. In particular: with one
seed per architecture there is no noise estimate, and the spread must not be
reported as an effect.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import scipy.stats

from act.search.architecture_eval import EvalResult, load_results
from act.search.architecture_report import (
    HEAD_PAIRING_RULE,
    _compact,
    _spearman_exact_p,
    axis_association,
    best_available_summary,
    blocked_permutation_test,
    crossed_contrast,
    head_pairing_decision,
    joint_structural_fit,
    paired_contrast_test,
    paired_design_contrast,
    paired_level_split,
    replication_decision,
    _paired_contrasts_at,
    _paired_level_splits_at,
    effect_in_compute_units,
    effect_summary,
    rank,
    rank_stability,
    search_cost,
    spread_by_length,
    structural_signal_by_length,
    two_set_blocked_test,
    variance_decomposition,
    verdict,
    with_structure,
    write_decision,
)

STEPS = 400
TOKENS = STEPS * 2 * 1024


def result(name: str, validation_loss: float, seed: int = 42, steps: int = STEPS) -> EvalResult:
    return EvalResult(
        name=name,
        family="test",
        genome_hash=f"hash-{name}",
        seed=seed,
        total_params=100_000_000,
        active_params=40_000_000,
        contracted_flops_per_token=2.5e8,
        dense_runtime_flops_per_token=4.0e8,
        training_flops=2.5e8 * TOKENS,
        validation_flops=2.5e8 * 16 * 2 * 1024,
        final_train_loss=validation_loss + 0.2,
        mean_train_loss=validation_loss + 0.25,
        validation_loss=validation_loss,
        steps=steps,
        train_tokens=TOKENS,
        val_tokens=16 * 2 * 1024,
        elapsed_s=170.0,
        measured_step_s=0.44,
        measured_tokens_per_s=4700.0,
        peak_memory_mb=3400.0,
        device="cuda",
        description="",
        val_source="holdout:2000000",
    )


#: A designed set holds most genes constant, and the ledger records that as a
#: value rather than a zero. Zero means "not recorded" - a ledger from before
#: the field existed - which changes what the report is entitled to conclude, so
#: the two cases must not be spelled the same way in a fixture.
HELD_GENES = {"depth": 8, "attention_heads": 8, "expert_count": 8, "active_experts": 2}


def designed(name: str, validation_loss: float, seed: int = 42, **genes) -> EvalResult:
    """A run from a designed set: unvaried structural genes recorded as held."""
    return replace(result(name, validation_loss, seed=seed), **(HELD_GENES | genes))


# --- loading --------------------------------------------------------------


def test_load_results_round_trips_jsonl(tmp_path):
    path = tmp_path / "stage1_seed42_results.jsonl"
    rows = [result("a", 3.5), result("b", 3.6)]
    path.write_text("\n".join(json.dumps(asdict(r)) for r in rows) + "\n")
    loaded = load_results(path)
    assert [r.name for r in loaded] == ["a", "b"]
    assert loaded[0].validation_loss == pytest.approx(3.5)
    # the fields needed to audit a run without rerunning it
    assert loaded[1].val_source == "holdout:2000000"
    assert loaded[1].seed == 42
    assert loaded[1].measured_tokens_per_s == pytest.approx(4700.0)


def test_load_results_accepts_several_ledgers(tmp_path):
    first = tmp_path / "one.jsonl"
    second = tmp_path / "two.jsonl"
    first.write_text(json.dumps(asdict(result("a", 3.5))) + "\n")
    second.write_text(json.dumps(asdict(result("b", 3.6))) + "\n")
    assert len(load_results([first, second])) == 2


def test_load_results_skips_blank_lines(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text(json.dumps(asdict(result("a", 3.5))) + "\n\n")
    assert len(load_results(path)) == 1


# --- ranking and effect size ---------------------------------------------


def test_rank_averages_across_seeds():
    results = [result("a", 3.5, seed=42), result("a", 3.7, seed=43), result("b", 3.6, seed=42)]
    ranked = rank(results, STEPS)
    assert ranked[0][0] == "a"
    assert ranked[0][1] == pytest.approx(3.6)
    assert ranked[0][2] == 2


def test_rank_ignores_other_training_lengths():
    results = [result("a", 3.5, steps=400), result("b", 9.9, steps=200)]
    assert [name for name, _, _ in rank(results, 400)] == ["a"]


def test_single_seed_gives_no_sigma_estimate():
    """Regression: a zero noise floor must not read as overwhelming evidence."""
    results = [result("a", 3.5), result("b", 4.1)]
    summary = effect_summary(results, STEPS, [42])
    assert summary["available"]
    assert summary["multi_seed"] is False
    assert summary["sigmas"] is None
    assert summary["within_seed_std"] == 0.0
    assert summary["architecture_spread"] == pytest.approx(0.6)


def test_two_seeds_produce_a_finite_sigma_estimate():
    results = [
        result("a", 3.50, seed=42),
        result("a", 3.52, seed=43),
        result("b", 3.90, seed=42),
        result("b", 3.92, seed=43),
    ]
    summary = effect_summary(results, STEPS, [42, 43])
    assert summary["multi_seed"] is True
    assert summary["sigmas"] is not None
    assert summary["sigmas"] > 10
    assert summary["best"] == "a"


def test_paired_contrast_uses_only_shared_seeds():
    results = [
        result("a", 3.50, seed=42),
        result("a", 3.52, seed=43),
        result("b", 3.70, seed=42),
        result("c", 3.90, seed=43),
    ]
    summary = effect_summary(results, STEPS, [42, 43])
    paired = {row["candidate"]: row for row in summary["paired_vs_best"]}
    assert paired["b"]["paired_seeds"] == [42]
    assert paired["c"]["paired_seeds"] == [43]
    assert "a" not in paired


def test_best_available_summary_falls_back_to_shorter_stages():
    results = [result("a", 3.5, seed=42, steps=400), result("b", 3.6, seed=42, steps=400)]
    results += [result("a", 9.0, seed=42, steps=200), result("b", 9.1, seed=42, steps=200)]
    summary = best_available_summary(results, [42])
    assert summary["steps"] == 400


# --- the decision rule ----------------------------------------------------


def decision_text(tmp_path, results, steps=STEPS, seeds=(42,), samples=200_000) -> str:
    path = tmp_path / "decision.md"
    write_decision(path, effect_summary(results, steps, list(seeds), samples=samples))
    return path.read_text()


def test_single_seed_is_provisional_not_an_effect(tmp_path):
    text = decision_text(tmp_path, [result("a", 3.5), result("b", 4.1)])
    assert "PROVISIONAL" in text
    assert "EFFECT PRESENT" not in text


def test_two_seeds_cannot_establish_an_effect_however_large(tmp_path):
    """Two seeds is a hard power floor, not a threshold the data can clear.

    With k architectures on 2 seeds the exact blocked test has only k!^2
    assignments, and the most extreme statistic is attained by k! consistent
    rewrites of them, so the smallest attainable p is 1/k!. For k=2 that is 0.5.
    Reporting a large but under-powered difference as established would be
    exactly the error this rule exists to prevent.
    """
    results = [
        result("a", 3.50, seed=42),
        result("a", 3.52, seed=43),
        result("b", 3.95, seed=42),
        result("b", 3.97, seed=43),
    ]
    text = decision_text(tmp_path, results, seeds=(42, 43))
    assert "NOT ESTABLISHED" in text
    assert "EFFECT ESTABLISHED" not in text
    assert "p=0.5" in text
    blocked = blocked_permutation_test(results, STEPS)
    assert blocked["min_achievable_p"] == pytest.approx(0.5)


def test_headline_never_rounds_a_p_value_to_zero():
    """A sampled null distribution resolves p only so finely.

    The headline used to format with 4 decimal places, so the v2 matched design's
    Monte-Carlo p printed as "p=0.0000" in the bolded line - reading as exact
    zero, and claiming more than 200k samples of a null can support, right above
    an evidence line giving the procedure's own resolution as 5e-06.
    """
    # 8 architectures on 3 seeds: too many within-block assignments to enumerate
    # (8!^3), so this goes through the Monte-Carlo path that the defect appeared
    # in. 7 non-tied blocks is enough that .4f would render the result as zero.
    results = [
        result(name, 3.60 + index * 0.05, seed=seed)
        for index, name in enumerate("abcdefgh")
        for seed in (42, 43, 44)
    ]
    summary = effect_summary(results, STEPS, [42, 43, 44])
    blocked = summary["blocked_test"]
    assert blocked["exact"] is False
    p_value = blocked["p"]
    assert p_value < 1e-4  # small enough that .4f would have shown zero
    headline = verdict(summary)[0]
    assert "p=0.0000" not in headline
    # Either it is stated to significant digits, or it is labelled as the bound
    # that the procedure can actually resolve - and here it is the latter, since
    # the estimate sits exactly on the Monte-Carlo resolution.
    assert f"p={p_value:.4g}" in headline or "p<=" in headline
    assert "p<=5e-06" in headline


def test_consistent_ordering_on_three_seeds_is_established(tmp_path):
    """The rule must be able to say yes, or it is not a test."""
    results = [
        result("a", 3.50, seed=42),
        result("a", 3.51, seed=43),
        result("a", 3.52, seed=44),
        result("b", 3.70, seed=42),
        result("b", 3.71, seed=43),
        result("b", 3.72, seed=44),
        result("c", 3.90, seed=42),
        result("c", 3.91, seed=43),
        result("c", 3.92, seed=44),
    ]
    text = decision_text(tmp_path, results, seeds=(42, 43, 44))
    assert "ARCHITECTURE EFFECT ESTABLISHED" in text
    blocked = blocked_permutation_test(results, STEPS)
    # Perfect consistency over 3 architectures on 3 seeds is exactly 1/36.
    assert blocked["p"] == pytest.approx(1 / 36)
    assert blocked["min_achievable_p"] == pytest.approx(1 / 36)


def test_effect_within_seed_noise_is_not_reported(tmp_path):
    results = [
        result("a", 3.500, seed=42),
        result("a", 3.400, seed=43),
        result("b", 3.505, seed=42),
        result("b", 3.395, seed=43),
    ]
    text = decision_text(tmp_path, results, seeds=(42, 43))
    assert "NO MATERIAL SIGNAL" in text
    assert "EFFECT ESTABLISHED" not in text
    # The recommendation must name the genes that were *measured* to run, and say
    # so about module type rather than recommending the one axis that cannot be
    # run at this scale (4-7x slower, OOM at 6 of 8 layers).
    assert "genes that actually run" in text
    assert "Module type is not available at this scale" in text


def test_spread_above_floor_but_underpowered_is_not_established(tmp_path):
    """The intermediate branch: big enough to matter, too noisy to claim."""
    results = [
        result("a", 3.50, seed=42),
        result("a", 3.40, seed=43),
        result("b", 3.40, seed=42),
        result("b", 3.30, seed=43),
    ]
    text = decision_text(tmp_path, results, seeds=(42, 43))
    assert "NOT ESTABLISHED" in text
    assert "not evidence that architecture does not matter" in text
    summary = effect_summary(results, STEPS, [42, 43])
    assert summary["architecture_spread"] > 0.01


def test_no_results_is_insufficient_data(tmp_path):
    path = tmp_path / "decision.md"
    write_decision(path, {"available": False})
    assert "INSUFFICIENT DATA" in path.read_text()


def test_recorded_v1_zoo_still_reaches_the_recorded_verdict():
    """Pins the phase's headline result against the real ledgers.

    The decision *is* the deliverable here, so a later change to the statistics
    must not be able to flip it silently. Skipped when the research artifacts are
    not present, e.g. a fresh checkout of the source alone.
    """
    ledgers = sorted(Path("artifacts/arch-zoo-100m-v1").glob("*_seed*_results.jsonl"))
    if not ledgers:
        pytest.skip("arch-zoo-100m-v1 ledgers not present")

    results = load_results(ledgers)
    summary = best_available_summary(results, [42, 43, 44])
    assert summary["steps"] == 1200
    assert summary["architecture_spread"] == pytest.approx(0.0404, abs=0.002)

    blocked = summary["blocked_test"]
    assert blocked["exact"] is True
    assert blocked["blocks"] == 3 and blocked["architectures"] == 3
    assert blocked["p"] > 0.05
    assert blocked["min_achievable_p"] == pytest.approx(1 / 36)
    # The effect is small but real in the sense that it is not zero; the verdict
    # is about whether this design can separate it, and it cannot.
    assert summary["compute_units"]["equivalent_steps"] == pytest.approx(68, abs=8)
    assert "NOT ESTABLISHED" in verdict(summary)[0]


def test_recorded_v2_matched_zoo_establishes_an_architecture_effect():
    """The matched-total design's verdict, pinned against the real ledgers.

    v1 varied total and active parameters together, so it could not attribute its
    spread to either. v2 holds total parameters to a 1.08x band while spreading
    active parameters 4.51x, and reaches a verdict v1 could not. Assertions are
    banded rather than exact because the run adds seeds over time, which is the
    intent: the *claims* must survive, the precise numbers may move.
    """
    directory = Path("artifacts/arch-zoo-100m-v2")
    ledgers = sorted(directory.glob("*_seed*_results.jsonl"))
    if not ledgers:
        pytest.skip("arch-zoo-100m-v2 ledgers not present")

    # The join is mandatory, not optional: without it every structural axis reads
    # as "not recorded" and the leading axis is chosen from the cost axes alone.
    results = with_structure(load_results(ledgers), directory / "zoo_definition.json")
    seeds = sorted({r.seed for r in results})
    summary = best_available_summary(results, seeds)
    assert summary["steps"] == 1200
    assert summary["architecture_spread"] > 0.2
    assert summary["blocked_test"]["architectures"] == 8
    assert summary["blocked_test"]["p"] < 0.01
    assert "ESTABLISHED" in verdict(summary)[0]

    # Total parameters are held, which is what makes the spread attributable.
    association = summary["axis_association"]
    axes = {row["axis"]: row for row in association["axes"]}
    assert axes["total parameters"]["spread_ratio"] < 1.10
    assert axes["active parameters"]["held"] is False
    assert axes["active parameters"]["spread_ratio"] > 4.0

    # The refutation this design exists to produce: across a 4.5x range, active
    # parameters show no association with loss. If this ever becomes significant
    # the recorded conclusion changes and the test must be revisited.
    assert axes["active parameters"]["p"] > 0.05

    # And the axis that does track loss is not separable from attention heads in
    # this set, which the decision must say rather than implying identification.
    collinearity = association["collinearity"]
    assert collinearity["leading"] == "hidden size"
    assert "attention heads" in {row["axis"] for row in collinearity["entangled"]}


def test_recorded_v2_effect_survives_dropping_any_one_architecture():
    """The established verdict must not rest on a single architecture.

    A spread that collapses when one candidate is removed is a statement about
    that candidate, not about architecture in general. Every leave-one-out set
    stays significant here, which is what makes the headline claim a general one.
    """
    directory = Path("artifacts/arch-zoo-100m-v2")
    ledgers = sorted(directory.glob("*_seed*_results.jsonl"))
    if not ledgers:
        pytest.skip("arch-zoo-100m-v2 ledgers not present")

    results = with_structure(load_results(ledgers), directory / "zoo_definition.json")
    names = sorted({r.name for r in results})
    assert names  # the design must exist before it can be dismantled

    worst_p = 0.0
    for dropped in names:
        reduced = [r for r in results if r.name != dropped]
        # 20k draws resolve to 5e-05, far below the 0.01 this asserts; the default
        # 200k is the record's resolution and costs 8x here for no extra decision.
        blocked = blocked_permutation_test(reduced, 1200, samples=20_000)
        assert blocked["available"], f"dropping {dropped} left no testable design"
        assert blocked["architectures"] == len(names) - 1
        worst_p = max(worst_p, blocked["p"])
    assert worst_p < 0.01, f"the effect is fragile: worst leave-one-out p = {worst_p}"


def test_recorded_v2_effect_is_not_an_artifact_of_the_holdout_split():
    """Architectures could differ in how much they overfit rather than in how
    well they model, so the same test is run on training loss. If held-out loss
    alone showed the effect it would be a split artifact."""
    directory = Path("artifacts/arch-zoo-100m-v2")
    ledgers = sorted(directory.glob("*_seed*_results.jsonl"))
    if not ledgers:
        pytest.skip("arch-zoo-100m-v2 ledgers not present")

    results = with_structure(load_results(ledgers), directory / "zoo_definition.json")
    as_train = [
        replace(entry, validation_loss=entry.final_train_loss) for entry in results
    ]
    blocked = blocked_permutation_test(as_train, 1200)
    assert blocked["p"] < 0.01


def test_established_verdict_names_the_horizon_limit_only_when_unmeasured():
    """An established effect at one training length is not a licence to search.

    The same project measured the spread falling 3.7x between 400 and 1200 steps,
    so the next-step text has to carry that limit - but repeating it once the
    design has actually measured several lengths would be noise.
    """
    single = [
        result(name, loss, seed=seed)
        for name, loss in (("a", 3.60), ("b", 3.70), ("c", 3.80))
        for seed in (42, 43, 44)
    ]
    summary = effect_summary(single, STEPS, [42, 43, 44])
    assert summary["blocked_test"]["p"] < 0.05
    assert "training length" in verdict(summary)[1]

    # Same effect, but the design ran a shorter length too, so the decay is
    # already reported in the spread-by-length table.
    multi = single + [
        result(name, loss + 0.5, seed=seed, steps=200)
        for name, loss in (("a", 3.60), ("b", 3.70), ("c", 3.80))
        for seed in (42, 43, 44)
    ]
    summary = effect_summary(multi, STEPS, [42, 43, 44])
    assert len(summary["spread_by_length"]) >= 2
    assert "training length" not in verdict(summary)[1]


def test_decision_lists_ranked_architectures_and_seeds(tmp_path):
    results = [
        result("a", 3.50, seed=42),
        result("a", 3.52, seed=43),
        result("b", 3.95, seed=42),
        result("b", 3.97, seed=43),
    ]
    text = decision_text(tmp_path, results, seeds=(42, 43))
    assert "`a`" in text and "`b`" in text
    assert "3.51" in text


# --- cost accounting ------------------------------------------------------


def test_search_cost_counts_training_and_evaluation(tmp_path):
    results = [result("a", 3.5), result("b", 3.6)]
    cost = search_cost(results, candidates=13)
    assert cost["runs_completed"] == 2
    assert cost["candidates_generated"] == 13
    assert cost["candidate_training_flops"] == pytest.approx(2 * 2.5e8 * TOKENS)
    assert cost["total_search_flops"] == pytest.approx(
        cost["candidate_training_flops"] + cost["candidate_evaluation_flops"]
    )
    assert cost["wall_clock_hours"] == pytest.approx(2 * 170.0 / 3600.0)
    assert str(STEPS) in cost["by_stage"]


# --- the statistics -------------------------------------------------------


def test_unpaired_anova_matches_a_direct_sum_of_squares():
    results = [
        result("a", 3.50, seed=42),
        result("a", 3.60, seed=43),
        result("a", 3.70, seed=44),
        result("b", 4.00, seed=42),
        result("b", 4.10, seed=43),
        result("b", 4.20, seed=44),
    ]
    anova = variance_decomposition(results, STEPS)
    assert anova["available"]
    assert anova["df_between"] == 1 and anova["df_within"] == 4
    # Hand-computed: ms_between = 3*0.25^2/1, ms_within = 0.01, so F = 37.5.
    # Asserted exactly, and cross-checked against scipy, so a mistake in the
    # sum-of-squares bookkeeping cannot pass silently as "roughly large".
    assert anova["f"] == pytest.approx(37.5)
    assert anova["p"] < 0.005
    f_scipy, p_scipy = scipy.stats.f_oneway([3.50, 3.60, 3.70], [4.00, 4.10, 4.20])
    assert anova["f"] == pytest.approx(f_scipy)
    assert anova["p"] == pytest.approx(p_scipy)


def test_anova_is_unavailable_without_replication():
    results = [result("a", 3.5, seed=42), result("b", 3.6, seed=42)]
    assert variance_decomposition(results, STEPS)["available"] is False


def test_blocked_test_enumerates_the_full_assignment_space():
    results = [
        result(name, loss, seed=seed)
        for name, loss in (("a", 3.60), ("b", 3.70), ("c", 3.80))
        for seed in (42, 43, 44)
    ]
    blocked = blocked_permutation_test(results, STEPS)
    assert blocked["exact"] is True
    assert blocked["architectures"] == 3 and blocked["blocks"] == 3
    assert blocked["assignments"] == 6**3


def test_blocked_test_requires_a_shared_seed_in_every_block():
    """An unbalanced design must be restricted, not silently tested as balanced."""
    results = [
        result("a", 3.50, seed=42),
        result("a", 3.52, seed=43),
        result("b", 3.70, seed=42),
        result("b", 3.72, seed=43),
        result("c", 3.90, seed=44),
    ]
    blocked = blocked_permutation_test(results, STEPS)
    assert blocked["available"]
    assert blocked["seeds"] == [42, 43]
    assert blocked["names"] == ["a", "b"]
    # Restricting is only honest if what was restricted *out* is named: the block
    # count printed beside the p is the survivors, not the design.
    assert blocked["seeds_dropped"] == [44]
    assert blocked["architectures_dropped"] == ["c"]
    # Seed 44 carries fewer runs than the fullest seed, so this is an unfinished
    # sweep rather than a zoo defect.
    assert blocked["unfinished_seeds"] == [44]
    assert blocked["unbalanced_seeds"] == []


def test_a_seed_that_ran_everything_but_cannot_intersect_is_a_defect():
    """A dropped seed that is *not* short is reported differently, because it is.

    Both failures shrink the rectangle identically, so only the per-seed row count
    separates them: a seed still training has fewer rows, while a seed that has run
    as many rows as any other but cannot be intersected is an unbalanced design.
    Reading the second as the first would let a broken zoo pass as an arm that is
    merely unfinished.
    """
    results = [
        result("a", 3.50, seed=42), result("a", 3.51, seed=43), result("a", 3.52, seed=44),
        result("b", 3.70, seed=42), result("b", 3.71, seed=43), result("b", 3.72, seed=44),
        result("c", 3.90, seed=42), result("c", 3.91, seed=43),
        result("d", 4.10, seed=44),
    ]
    blocked = blocked_permutation_test(results, STEPS)
    assert blocked["seeds"] == [42, 43] and blocked["blocks"] == 2
    assert blocked["seeds_dropped"] == [44]
    # Seed 44 wrote three rows, as many as any other seed, so it is not short.
    assert blocked["unfinished_seeds"] == []
    assert blocked["unbalanced_seeds"] == [44]
    assert blocked["architectures_dropped"] == ["d"]


def test_the_decision_carries_each_promotion_verdict_and_its_derivation(tmp_path):
    """A promotion verdict is part of the arm's reading, not only its terminal log.

    The threshold depends on the stage's seed count and on the measured seed std at
    its length, so both are printed beside the margin: a bare nats figure cannot be
    checked, and a verdict that exists only in stdout is lost with the terminal.
    """
    results = [
        result(name, loss, seed=seed)
        for name, loss in (("a", 3.50), ("b", 3.56), ("c", 3.90), ("d", 4.10))
        for seed in (42, 43)
    ]
    outcome = best_available_summary(results, [42, 43])
    outcome["promotions"] = [
        {
            "label": "Screen",
            "steps": 400,
            "seeds": 2,
            "std": 0.0286,
            "threshold": 0.0286,
            "threshold_overridden": False,
            "cut_margin": 0.06,
            "resolvable": True,
            "cut_pair": [{"name": "b", "loss": 3.56}, {"name": "c", "loss": 3.90}],
            "borderline": [],
        },
        {
            "label": "Stage 2",
            "steps": 800,
            "seeds": 2,
            "std": 0.0165,
            "threshold": 0.0165,
            "threshold_overridden": False,
            "cut_margin": 0.004,
            "resolvable": False,
            "cut_pair": [{"name": "a", "loss": 3.50}, {"name": "b", "loss": 3.504}],
            "borderline": [{"name": "c", "loss": 3.51, "gap_to_cut": 0.006}],
        },
    ]
    path = tmp_path / "decision.md"
    write_decision(path, outcome)
    text = path.read_text()

    assert "## Promotion at each screening cut" in text
    assert "sqrt(2 / seeds)" in text
    # Both stages are costed with their own std and seed count, so a two-seed 400-step
    # cut and a two-seed 800-step one do not share a threshold.
    assert "| Screen | 400 | 2 | 0.0286 | 0.0286 | 0.0600 | decided by the data |" in text
    assert "| Stage 2 | 800 | 2 | 0.0165 | 0.0165 | 0.0040 | **not resolved** |" in text
    # The unresolved cut names the pair it could not separate and the other candidate
    # inside the threshold, because that is the set a tie-break would be choosing from.
    assert "selection rather than a decision" in text
    assert "`a` (3.5000) and `b` (3.5040)" in text
    assert "`c` (0.0060)" in text


def test_a_decision_without_promotion_reports_says_nothing_about_them(tmp_path):
    results = [
        result(name, loss, seed=seed)
        for name, loss in (("a", 3.50), ("b", 3.90))
        for seed in (42, 43)
    ]
    text = decision_text(tmp_path, results, seeds=(42, 43))
    assert "Promotion at each screening cut" not in text


def test_a_complete_design_discloses_nothing(tmp_path):
    results = [
        result(name, loss, seed=seed)
        for name, loss in (("a", 3.60), ("b", 3.70), ("c", 3.80))
        for seed in (42, 43, 44)
    ]
    blocked = blocked_permutation_test(results, STEPS)
    assert blocked["seeds_dropped"] == []
    assert blocked["architectures_dropped"] == []
    text = decision_text(tmp_path, results, seeds=(42, 43, 44))
    assert "excluded from that test" not in text
    assert "provisional" not in text.lower()


def test_decision_names_the_seed_the_blocked_test_could_not_use(tmp_path):
    """The p-value is over the rectangle; the decision has to say so.

    Without this line a reader sees '2 blocks x 2 architectures' and a floor of
    1/2, and has to infer from the seed list that a third seed was written. The
    line states the cause as well, because 'still training' and 'unbalanced' call
    for different responses.
    """
    results = [
        result("a", 3.50, seed=42), result("a", 3.52, seed=43),
        result("b", 3.70, seed=42), result("b", 3.72, seed=43),
        result("c", 3.90, seed=44),
    ]
    text = decision_text(tmp_path, results, seeds=(42, 43, 44))
    assert "Seeds excluded from that test: [44]" in text
    assert "still training" in text
    assert "Architectures excluded from that test: ['c']" in text


def test_blocked_test_orders_architectures_best_first():
    results = [
        result(name, loss, seed=seed)
        for name, loss in (("zzz", 3.60), ("aaa", 3.90))
        for seed in (42, 43)
    ]
    blocked = blocked_permutation_test(results, STEPS)
    # Alphabetical would put 'aaa' first; the ordering must be by loss.
    assert blocked["names"] == ["zzz", "aaa"]


def test_blocked_rank_and_value_tests_are_both_reported():
    results = [
        result(name, loss, seed=seed)
        for name, loss in (("a", 3.60), ("b", 3.70), ("c", 3.80))
        for seed in (42, 43, 44)
    ]
    blocked = blocked_permutation_test(results, STEPS)
    assert blocked["p"] is not None
    assert blocked["p_ranks"] is not None
    assert blocked["p_paired_t_reference"] is not None


def test_large_designs_switch_to_monte_carlo_rather_than_a_chi_square():
    """8!^3 is 6.6e13 assignments, so enumeration cannot run on a matched design.

    The alternative must not be the chi-square approximation: at 3 seeds that
    approximation is exactly where the primary test would quietly lose its
    validity. Sampling the same null keeps the test meaningful.
    """
    results = [
        result(f"arch{index}", 3.0 + index * 0.02, seed=seed)
        for index in range(8)
        for seed in (42, 43, 44)
    ]
    blocked = blocked_permutation_test(results, STEPS, samples=2_000)
    assert blocked["method"] == "monte-carlo"
    assert blocked["exact"] is False
    assert blocked["assignments"] == math.factorial(8) ** 3
    assert blocked["p_resolution"] == pytest.approx(1 / 2001)
    # A perfectly consistent 8-way ordering must still come out significant.
    assert blocked["p"] < 0.01


def test_monte_carlo_does_not_fire_on_pure_noise():
    """Calibration: a test that always rejects is not a test."""
    generator = random.Random(4)
    results = [
        result(f"arch{index}", 3.0 + generator.gauss(0, 0.05), seed=seed)
        for index in range(8)
        for seed in (42, 43, 44)
    ]
    blocked = blocked_permutation_test(results, STEPS, samples=4_000)
    assert blocked["method"] == "monte-carlo"
    assert blocked["p"] > 0.05


def test_exact_path_reports_its_own_resolution():
    results = [
        result(name, loss, seed=seed)
        for name, loss in (("a", 3.50), ("b", 3.70), ("c", 3.90))
        for seed in (42, 43, 44)
    ]
    blocked = blocked_permutation_test(results, STEPS)
    assert blocked["method"] == "exact"
    assert blocked["p_resolution"] == pytest.approx(1 / 216)


def test_no_design_reports_a_p_value_below_its_power_floor():
    """The decision text quotes a floor, so the floor has to be a real bound.

    ``min_achievable_p`` is what the report tells a reader the design can and
    cannot resolve. If it over- or under-stated the true minimum, the stated
    reason for a non-significant verdict would be wrong.
    """
    generator = random.Random(0)
    for _ in range(40):
        results = [
            result(name, 3.0 + generator.random(), seed=seed)
            for name in ("a", "b", "c")
            for seed in (42, 43, 44)
        ]
        blocked = blocked_permutation_test(results, STEPS)
        assert blocked["p"] >= blocked["min_achievable_p"] - 1e-12
        assert blocked["min_achievable_p"] == pytest.approx(1 / 36)


def test_power_floor_follows_the_factorial_of_the_design():
    """Floor is 1/k!^(blocks-1): it is a property of the design, not the data."""
    for k, blocks in ((3, 2), (3, 3), (4, 2), (5, 2)):
        results = [
            result(f"arch{index}", 3.0 + index * 0.01, seed=100 + block)
            for index in range(k)
            for block in range(blocks)
        ]
        blocked = blocked_permutation_test(results, STEPS)
        assert blocked["min_achievable_p"] == pytest.approx(
            1 / math.factorial(k) ** (blocks - 1)
        ), f"k={k} blocks={blocks}"


def test_compute_units_converts_a_gap_into_training_steps():
    results = []
    for name, offset in (("a", 0.0), ("b", 0.10)):
        for seed in (42, 43):
            results.append(result(name, 4.00 + offset, seed=seed, steps=400))
            results.append(result(name, 3.00 + offset, seed=seed, steps=1200))
    units = effect_in_compute_units(results, 1200)
    assert units["available"]
    # Loss falls exactly 1.0 nat over 800 steps, so 0.10 nats is 80 steps.
    assert units["slope_per_step"] == pytest.approx(1.0 / 800)
    assert units["equivalent_steps"] == pytest.approx(80.0)
    assert units["fraction_of_budget"] == pytest.approx(80 / 1200)


def test_compute_units_refuses_a_rising_loss_slope():
    """A non-decreasing loss gives no denominator, and inventing one is worse."""
    results = []
    for name in ("a", "b"):
        for seed in (42, 43):
            results.append(result(name, 3.00, seed=seed, steps=400))
            results.append(result(name, 4.00, seed=seed, steps=1200))
    units = effect_in_compute_units(results, 1200)
    assert units["available"] is False
    assert "not decreasing" in units["reason"]


def test_rank_stability_flags_a_winner_that_does_not_survive():
    """The failure that matters: short screening selecting the wrong finalist."""
    results = []
    # 'early' leads at 400 steps, 'late' overtakes by 1200, 'static' stays put.
    for seed in (42, 43, 44):
        results.append(result("early", 3.40, seed=seed, steps=400))
        results.append(result("late", 3.50, seed=seed, steps=400))
        results.append(result("static", 3.60, seed=seed, steps=400))
        results.append(result("early", 3.10, seed=seed, steps=1200))
        results.append(result("late", 3.00, seed=seed, steps=1200))
        results.append(result("static", 3.20, seed=seed, steps=1200))
    rows = rank_stability(results)
    assert len(rows) == 1
    assert rows[0]["winner_preserved"] is False
    assert rows[0]["best_at_from"] == "early"
    assert rows[0]["best_at_to"] == "late"


def test_spread_by_length_uses_only_the_common_architecture_set():
    results = [
        result("a", 3.00, seed=42, steps=400),
        result("b", 3.50, seed=42, steps=400),
        result("c", 9.00, seed=42, steps=400),  # not run at 1200, must be excluded
        result("a", 2.00, seed=42, steps=1200),
        result("b", 2.10, seed=42, steps=1200),
    ]
    summary = spread_by_length(results)
    assert summary["400"]["architectures"] == 2
    assert summary["400"]["spread"] == pytest.approx(0.50)
    assert summary["1200"]["spread"] == pytest.approx(0.10)


def test_effect_summary_carries_the_evidence_bundle():
    results = [
        result(name, loss, seed=seed)
        for name, loss in (("a", 3.60), ("b", 3.70), ("c", 3.80))
        for seed in (42, 43, 44)
    ]
    summary = effect_summary(results, STEPS, [42, 43, 44])
    for key in (
        "anova",
        "blocked_test",
        "rank_stability",
        "compute_units",
        "spread_by_length",
        "axis_association",
    ):
        assert key in summary


def test_selection_caveat_appears_only_when_selection_happened():
    """A flat design promotes nothing, so claiming the caveat understates it."""
    flat = [
        result(name, loss, seed=seed)
        for name, loss in (("a", 3.60), ("b", 3.70), ("c", 3.80))
        for seed in (42, 43)
    ]
    assert effect_summary(flat, STEPS, [42, 43])["selection_caveat"] == ""
    assert effect_summary(flat, STEPS, [42, 43])["screened_at_shorter_lengths"] == 0

    # Same architectures, but now six were screened at a shorter length first.
    halved = flat + [
        result(f"candidate{index}", 3.5 + index * 0.05, seed=seed, steps=200)
        for index in range(6)
        for seed in (42, 43)
    ]
    summary = effect_summary(halved, STEPS, [42, 43])
    assert "anti-conservative" in summary["selection_caveat"]
    assert summary["screened_at_shorter_lengths"] == 6


def test_axis_association_marks_a_held_axis_as_held():
    """A matched design deliberately holds an axis; correlating against a
    constant is meaningless, and reporting a p-value for it would be noise."""
    results = []
    for name, active in (("a", 40_000_000), ("b", 60_000_000), ("c", 80_000_000)):
        for seed in (42, 43):
            results.append(
                replace(
                    result(name, 3.60 - active / 1e9, seed=seed),
                    total_params=100_000_000,
                    active_params=active,
                )
            )
    association = axis_association(results, STEPS)
    assert association["available"]
    rows = {row["axis"]: row for row in association["axes"]}
    assert rows["total parameters"]["held"] is True
    assert "spearman" not in rows["total parameters"]
    assert rows["active parameters"]["held"] is False
    # More active parameters was constructed to mean lower loss, so the
    # correlation with loss is perfectly negative.
    assert rows["active parameters"]["spearman"] == pytest.approx(-1.0)
    assert rows["active parameters"]["spread_ratio"] == pytest.approx(2.0)


def test_axis_association_needs_three_architectures():
    results = [result("a", 3.5, seed=42), result("a", 3.4, seed=43)]
    assert axis_association(results, STEPS)["available"] is False


def definition_file(tmp_path, name: str = "arch") -> Path:
    path = tmp_path / "zoo_definition.json"
    path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "name": name,
                        "genome": {
                            "hidden_size": 896,
                            "layers": [
                                {
                                    "heads": 14,
                                    "active_experts": 2,
                                    "experts": [{"width": 128}, {"width": 128}],
                                }
                            ],
                        },
                    }
                ]
            }
        )
    )
    return path


def test_with_structure_fills_genes_a_ledger_predates(tmp_path):
    """Regression: the join used to live in the reporting script only.

    Any other caller then read six structural axes as "not recorded" and got a
    leading axis chosen from the cost axes alone - a silently inverted
    conclusion, which is what happened to this very test before the move.
    """
    results = [result("arch", 3.5), result("arch", 3.4, seed=43)]
    assert results[0].hidden_size == 0

    joined = with_structure(results, definition_file(tmp_path))
    assert joined[0].hidden_size == 896
    assert joined[0].depth == 1
    assert joined[0].attention_heads == 14
    assert joined[0].expert_count == 2
    assert joined[0].active_experts == 2
    assert joined[0].expert_width == 128

    # And the axis table can now see them at all: without the join the row reads
    # "not recorded" instead of carrying a value.
    unjoined = results + [result("other", 3.7), result("third", 3.8)]
    without = axis_association(unjoined, STEPS)
    assert {row["axis"] for row in without["axes"] if row.get("unrecorded")} == {
        "depth",
        "hidden size",
        "attention heads",
        "expert count",
        "active experts",
        "expert width",
    }


def test_with_structure_does_not_overwrite_a_recorded_gene(tmp_path):
    """A ledger written after the fields existed is the better source."""
    recorded = replace(result("arch", 3.5), hidden_size=512)
    joined = with_structure([recorded], definition_file(tmp_path))
    assert joined[0].hidden_size == 512
    assert joined[0].attention_heads == 14


def test_with_structure_without_a_definition_is_a_noop(tmp_path):
    results = [result("arch", 3.5)]
    assert with_structure(results, tmp_path / "absent.json") == results


def test_axis_association_flags_an_entangled_axis():
    """Holding total parameters fixed while widening the residual stream forces
    experts narrower, so the two genes move together and the set cannot say
    which one the loss followed. Reporting only the leading axis would over-read.
    """
    layouts = ((512, 512, 3.60), (640, 384, 3.55), (896, 256, 3.50))
    results = [
        designed(f"arch{index}", loss, seed=seed, hidden_size=hidden, expert_width=width)
        for index, (hidden, width, loss) in enumerate(layouts)
        for seed in (42, 43)
    ]
    collinearity = axis_association(results, STEPS)["collinearity"]
    assert collinearity["available"] is True
    assert collinearity["leading"] == "hidden size"
    entangled = {row["axis"]: row["spearman"] for row in collinearity["entangled"]}
    assert entangled["expert width"] == pytest.approx(-1.0)
    assert "cannot separate" in collinearity["caveat"]


def test_axis_association_does_not_flag_independent_axes():
    """The mirror case: when the leading axis is genuinely uncorrelated with the
    other spread axes, the caveat must say so rather than staying silent, so a
    reader can tell "not checked" apart from "checked and clean".
    """
    layouts = ((512, 512, 3.60), (640, 256, 3.55), (896, 384, 3.50))
    results = [
        designed(f"arch{index}", loss, seed=seed, hidden_size=hidden, expert_width=width)
        for index, (hidden, width, loss) in enumerate(layouts)
        for seed in (42, 43)
    ]
    collinearity = axis_association(results, STEPS)["collinearity"]
    assert collinearity["leading"] == "hidden size"
    assert collinearity["entangled"] == []
    assert collinearity["structural_axes_unrecorded"] == []
    assert "no co-varying gene in this set explains the association" in collinearity["caveat"]


def test_axis_association_says_when_the_structural_join_is_missing():
    """The mirror-image error reads as a *finding*: with the structural genes
    never joined they are all zeros, they drop out of the spread, and the lead is
    chosen from the cost axes alone - after which the caveat would claim no
    structural gene explains the association, when none was examined. A report
    that cannot tell "checked and clean" from "not checked" is the failure this
    guards. arch-zoo-100m-v1's ledgers predate these fields, so it is the real
    case, not a hypothetical one.
    """
    results = [
        replace(
            result(f"arch{index}", 3.60 - index * 0.05, seed=seed),
            total_params=100_000_000 + index * 10_000_000,
            active_params=40_000_000 + index * 5_000_000,
        )
        for index in range(4)
        for seed in (42, 43)
    ]
    collinearity = axis_association(results, STEPS)["collinearity"]
    assert collinearity["available"] is True
    assert len(collinearity["structural_axes_unrecorded"]) == 6
    assert "structural axes are unrecorded" in collinearity["caveat"]
    assert "no structural attribution can be made" in collinearity["caveat"]
    assert "not explained" not in collinearity["caveat"]


def test_axis_association_needs_two_spread_axes_for_collinearity():
    """One spread axis has nothing to be confounded with."""
    results = [
        replace(result(f"arch{index}", 3.60 - index * 0.05, seed=seed), hidden_size=hidden)
        for index, hidden in enumerate((512, 640, 896))
        for seed in (42, 43)
    ]
    collinearity = axis_association(results, STEPS)["collinearity"]
    assert collinearity["available"] is False
    assert "2+ spread axes" in collinearity["reason"]


def test_exact_spearman_p_is_not_zero_at_perfect_correlation():
    """Regression: scipy's t approximation reports p=0 at |rho|=1.

    For 3 architectures a perfect rank correlation is 2 of the 6 arrangements,
    so the honest p is 1/3. Reading "p = 0.0000" off three points would claim
    far more than three points can support.
    """
    p_value, exact = _spearman_exact_p([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    assert exact is True
    assert p_value == pytest.approx(1 / 3)
    # And the value scipy would have reported, to document why this exists.
    _, approximate = scipy.stats.spearmanr([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    assert approximate == pytest.approx(0.0)


def test_exact_spearman_p_falls_back_when_enumeration_is_too_large():
    values = [float(index) for index in range(12)]
    p_value, exact = _spearman_exact_p(values, values, cap=1000)
    assert exact is False
    assert p_value is None


def test_compact_formats_large_counts_readably():
    assert _compact(77_171_584) == "77.2M"
    assert _compact(3498.0) == "3.5k"
    assert _compact(0.4428) == "0.4428"


def test_decision_reports_the_axis_section(tmp_path):
    results = []
    for name, active in (("a", 40_000_000), ("b", 60_000_000), ("c", 80_000_000)):
        for seed in (42, 43):
            results.append(
                replace(
                    result(name, 3.60 - active / 1e9, seed=seed),
                    active_params=active,
                )
            )
    text = decision_text(tmp_path, results, seeds=(42, 43))
    assert "Which axis moves with loss" in text
    assert "held constant" in text
    assert "hypothesis, not a result" in text
    # Every run in this design shares one hidden size, so nothing is confounded;
    # the block is still printed, because silence would be unreadable as "clean".
    assert "Confounding structure" in text


def test_verdict_is_unavailable_without_results():
    headline, action = verdict({"available": False})
    assert "INSUFFICIENT DATA" in headline


def test_search_cost_reports_tokens_per_stage():
    results = [result("a", 3.5, steps=400), result("b", 3.6, steps=400), result("c", 3.7, steps=200)]
    cost = search_cost(results, candidates=3)
    assert cost["by_stage"]["400"]["runs"] == 2
    assert cost["by_stage"]["200"]["runs"] == 1


def crossed_result(
    name: str,
    loss: float,
    *,
    depth: int,
    hidden: int,
    seed: int,
    steps: int = STEPS,
) -> EvalResult:
    return replace(
        result(name, loss, seed=seed, steps=steps),
        depth=depth,
        hidden_size=hidden,
    )


def crossed_design(depth_effect: float, hidden_effect: float, noise: float = 0.0):
    """Two depths x two hidden sizes, two architectures per cell, three seeds.

    ``depth_effect`` is added at the deep level and ``hidden_effect`` at the
    narrow level, so both are losses that make an architecture worse.
    """
    results = []
    for depth in (5, 14):
        for hidden in (512, 896):
            for seed in (42, 43, 44):
                for rep in (0, 1):
                    jitter = noise * ((rep * 2 - 1) + (seed - 43) / 3)
                    loss = (
                        3.0
                        + (depth_effect if depth == 14 else 0.0)
                        + (hidden_effect if hidden == 512 else 0.0)
                        + jitter
                    )
                    results.append(
                        crossed_result(
                            f"D{depth}_H{hidden}_r{rep}",
                            loss,
                            depth=depth,
                            hidden=hidden,
                            seed=seed,
                        )
                    )
    return results


def test_crossed_contrast_recovers_both_main_effects():
    design = crossed_design(depth_effect=0.08, hidden_effect=0.20)
    contrast = crossed_contrast(design, STEPS)
    assert contrast["available"]
    assert {contrast["factor_a"], contrast["factor_b"]} == {"depth", "hidden_size"}
    # Deep is worse by construction, so its main effect is positive.
    assert contrast["main"]["depth"]["nats"] == pytest.approx(0.08, abs=1e-9)
    # Narrow is worse, and the effect is signed as high level minus low level,
    # i.e. 896 minus 512, which is negative when 896 is better.
    assert contrast["main"]["hidden_size"]["nats"] == pytest.approx(-0.20, abs=1e-9)
    assert contrast["main"]["depth"]["p"] is not None
    assert contrast["main"]["depth"]["p"] < 0.05


def test_crossed_contrast_separates_a_gene_a_matched_zoo_cannot():
    """The whole point: two genes that always move together are separable here.

    A matched design whose zoos trade hidden size against depth reports both axes
    as equally associated with loss. Crossing them makes each main effect an
    average over the other's levels, so a real effect on one gene is recovered
    even when the other is held irrelevant.
    """
    design = crossed_design(depth_effect=0.0, hidden_effect=0.25)
    contrast = crossed_contrast(design, STEPS)
    assert contrast["main"]["hidden_size"]["nats"] == pytest.approx(-0.25, abs=1e-9)
    assert contrast["main"]["depth"]["nats"] == pytest.approx(0.0, abs=1e-9)
    # Depth has no effect, so its p-value must not clear the bar.
    assert contrast["main"]["depth"]["p"] > 0.05


def test_crossed_contrast_is_unavailable_without_a_crossed_pair():
    flat = [result(f"a{i}", 3.5 + i / 100) for i in range(3)]
    contrast = crossed_contrast(flat, STEPS)
    assert not contrast["available"]
    assert "binary structural genes" in contrast["reason"]


def test_crossed_contrast_reports_the_interaction():
    """An interaction is what a marginal association cannot express at all."""
    depth_effect_at_narrow = 0.20
    depth_effect_at_wide = 0.02
    results = []
    for depth in (5, 14):
        for hidden in (512, 896):
            cost = (
                depth_effect_at_narrow
                if (depth == 14 and hidden == 512)
                else depth_effect_at_wide
                if depth == 14
                else 0.0
            )
            for seed in (42, 43, 44):
                for rep in (0, 1):
                    results.append(
                        crossed_result(
                            f"D{depth}_H{hidden}_r{rep}",
                            3.0 + cost,
                            depth=depth,
                            hidden=hidden,
                            seed=seed,
                        )
                    )
    contrast = crossed_contrast(results, STEPS)
    # The depth penalty is larger at the narrow hidden size, so the per-level
    # contrasts must differ by exactly that gap.
    assert contrast["main"]["depth"]["interaction_nats"] == pytest.approx(
        depth_effect_at_narrow - depth_effect_at_wide, abs=1e-9
    )
    assert contrast["main"]["hidden_size"]["interaction_nats"] == pytest.approx(
        depth_effect_at_narrow - depth_effect_at_wide, abs=1e-9
    )


def test_spread_by_length_reports_decay_and_noise_units():
    results = []
    for name, base in (("best", 3.0), ("worst", 3.6)):
        for steps in (400, 800):
            for seed in (42, 43):
                # The gap halves when training doubles, and the seed scatter keeps
                # its size, so the effect weakens in seed-noise units.
                loss = 3.0 + (0.6 if name == "worst" else 0.0) * (400 / steps)
                results.append(result(name, loss + (0.01 if seed == 43 else 0.0), seed=seed, steps=steps))
    summary = spread_by_length(results)
    assert summary["400"]["spread"] == pytest.approx(0.6, abs=1e-9)
    assert summary["800"]["spread"] == pytest.approx(0.3, abs=1e-9)
    decay = summary["decay"]
    assert decay["doublings"] == pytest.approx(1.0)
    assert decay["per_doubling"] == pytest.approx(2.0, abs=1e-9)
    assert decay["best_at_every_length"] is True


def test_decision_states_the_horizon_decay(tmp_path):
    results = []
    for name, base in (("best", 3.0), ("worst", 3.6)):
        for steps in (400, 800):
            for seed in (42, 43, 44):
                loss = 3.0 + (0.6 if name == "worst" else 0.0) * (400 / steps)
                results.append(result(name, loss, seed=seed, steps=steps))
    text = decision_text(tmp_path, results, steps=800, seeds=(42, 43, 44))
    assert "Spread against training length" in text
    assert "per doubling of training" in text
    assert "same architecture is best at every length" in text


def test_verdict_names_the_attributed_lever_when_crossed():
    outcome = effect_summary(crossed_design(depth_effect=0.08, hidden_effect=0.20), STEPS, [42, 43, 44])
    headline, action = verdict(outcome)
    assert "ESTABLISHED" in headline
    # The larger resolved main effect is named, and the smaller one is not dropped.
    assert "`hidden_size`" in action
    assert "`depth`" in action
    assert "spend the search budget there" in action


def test_verdict_omits_attribution_without_a_crossing():
    results = [
        replace(result(f"a{i}", 3.0 + i / 10, seed=seed), hidden_size=512, depth=8)
        for i in range(4)
        for seed in (42, 43, 44)
    ]
    headline, action = verdict(effect_summary(results, STEPS, [42, 43, 44]))
    assert "spend the search budget there" not in action


def _horizon_results():
    directory = Path("artifacts/arch-zoo-100m-v2-horizon")
    ledgers = sorted(directory.glob("*_seed*_results.jsonl"))
    if not ledgers:
        pytest.skip("arch-zoo-100m-v2-horizon ledgers not present")
    return with_structure(load_results(ledgers), directory / "zoo_definition.json")


def test_recorded_horizon_decay_is_a_variance_decay_not_a_signal_decay():
    """The phase's horizon claim, pinned against the real ledgers.

    Sub-clauses that matter and would change the conclusion if they flipped:
    the spread decays monotonically, the winner is the same architecture at every
    length, and seed noise decays enough that the spread stays a large multiple
    of it rather than washing out. Assertions are banded, because the arms add
    seeds and lengths over time and the *claims* are what must survive.
    """
    horizon = _horizon_results()
    v2_dir = Path("artifacts/arch-zoo-100m-v2")
    v2_ledgers = sorted(v2_dir.glob("*_seed*_results.jsonl"))
    if not v2_ledgers:
        pytest.skip("arch-zoo-100m-v2 ledgers not present")
    pooled = horizon + with_structure(
        load_results(v2_ledgers), v2_dir / "zoo_definition.json"
    )

    summary = spread_by_length(pooled)
    lengths = [key for key in summary if key != "decay"]
    assert len(lengths) >= 3, "the decay claim needs at least three lengths"

    spreads = [summary[length]["spread"] for length in lengths]
    assert spreads == sorted(spreads, reverse=True), f"spread is not monotone: {spreads}"

    decay = summary["decay"]
    # v1's independent estimate was 1.4-1.6x per doubling; the matched design must
    # land in the same regime for the replication claim to hold.
    assert 1.2 < decay["per_doubling"] < 2.5, decay["per_doubling"]
    # The whole reading: the same architecture wins at every length.
    assert decay["best_at_every_length"] is True

    # Noise decays with the spread, so detectability holds. If this multiple
    # collapsed the "variance not signal" reading would be wrong.
    for length in lengths:
        assert summary[length]["sigmas"] > 5.0, (length, summary[length]["sigmas"])
    assert decay["sigmas_at_last"] > 0.5 * decay["sigmas_at_first"]

    # Screening ranks a short run like a long one on this set.
    stability = rank_stability(pooled)
    assert stability
    assert min(row["spearman"] for row in stability) > 0.8, stability


def test_recorded_crossed_factorial_attributes_the_effect_to_hidden_size():
    """The phase's attribution claim, pinned against the real ledgers.

    v2 could not separate hidden size from expert width, so the attribution was
    open. The crossed arm's whole purpose is to make both main effects
    identifiable, and the claims that must survive are: both factors are
    significant, hidden size is the larger lever, and depth's sign is negative in
    the sense that shallow wins.
    """
    directory = Path("artifacts/arch-zoo-100m-v3-crossed")
    ledgers = sorted(directory.glob("*_seed*_results.jsonl"))
    if not ledgers:
        pytest.skip("arch-zoo-100m-v3-crossed ledgers not present")

    results = with_structure(load_results(ledgers), directory / "zoo_definition.json")
    contrast = crossed_contrast(results, 1200, samples=200_000)
    assert contrast["available"]
    assert {contrast["factor_a"], contrast["factor_b"]} == {"depth", "hidden_size"}

    depth = contrast["main"]["depth"]
    hidden = contrast["main"]["hidden_size"]

    # Both levers are real. A non-significant depth here would change the reading
    # from "both genes matter" to "only the residual stream does".
    assert depth["p"] < 0.01 and hidden["p"] < 0.01
    # Deeper is worse at matched total parameters and matched hidden size.
    assert depth["nats"] > 0
    # Wider is better: the effect is signed high-minus-low level, i.e. 896 minus
    # 512, so a bigger hidden size shows up as a negative number.
    assert hidden["nats"] < 0
    # Residual-stream width is the bigger lever, which is the attribution.
    assert abs(hidden["nats"]) > 1.5 * abs(depth["nats"])

    # Every cell is populated, which is what makes the crossing interpretable
    # rather than a confound with its own missing data.
    assert all(cell["runs"] >= 6 for cell in contrast["cells"].values())

    # The interaction is explicitly *not* claimed; if it ever resolves, the
    # decision text should be revisited, so it is asserted as unresolved here.
    assert depth["interaction_p"] is None or depth["interaction_p"] > 0.05


def test_crossed_contrast_reports_interaction_against_three_nulls():
    """The interaction needs all three numbers to be read honestly.

    The re-split test is exact but assumes no main effect, so it is conservative.
    The seed-blocked sign test is better calibrated but floored at 2/2^seeds.
    Reporting only the first would understate a real interaction; reporting only
    the third would overstate it.
    """
    # Two things must be true for all three nulls to be defined, and both are easy
    # to get wrong when hand-building a design: the interaction must be nonzero (or
    # the seed contrasts are all exactly zero), and it must *vary by seed* (or their
    # variance is zero and the paired-t is undefined). Within-block replicate jitter
    # does not do it, because the two arms of a block share it and it cancels out of
    # the contrast, so the variation is put on the interaction itself.
    design = []
    for depth in (5, 14):
        for hidden in (512, 896):
            for seed in (42, 43, 44):
                # The seed term rides on one arm only. Adding it at both hidden
                # levels would cancel out of the interaction and leave the seed
                # contrasts identical again, which is the trap this test walks into
                # on purpose: a design can look noisy and still have no interaction
                # variance at all.
                cost = (
                    (0.18 + 0.004 * (seed - 43)) if hidden == 512 else 0.04
                ) if depth == 14 else 0.0
                for rep in (0, 1):
                    design.append(
                        crossed_result(
                            f"D{depth}_H{hidden}_r{rep}",
                            3.0 + cost + 0.01 * (rep * 2 - 1),
                            depth=depth,
                            hidden=hidden,
                            seed=seed,
                        )
                    )
    contrast = crossed_contrast(design, STEPS)
    main = contrast["main"]["depth"]

    assert main["interaction_p"] is not None
    assert main["interaction_p_seed_blocked"] is not None
    assert main["interaction_p_paired_t"] is not None
    # Three seeds is the design here, so the sign test's floor is 2/2^3.
    assert main["interaction_p_seed_blocked"] >= 2 / 2 ** 3 - 1e-12
    assert main["interaction_p_seed_blocked_floor"] == pytest.approx(2 / 2 ** 3)
    assert len(main["interaction_seed_contrasts"]) == 3
    assert contrast["main"]["hidden_size"]["interaction_nats"] == pytest.approx(
        main["interaction_nats"], abs=1e-9
    )


def test_crossed_contrast_handles_many_seeds_without_a_mismatched_index():
    """Regression: `picks` holds one index *per block*, not per block slot.

    The observed assignment was passed as ``range(len(attainable))``, which reads
    element i of block i. That only survives while every block has at least as
    many attainable splits as there are blocks, so it silently produced a wrong
    observed statistic at three seeds and crashed outright at twelve. A design
    with enough seeds to trip it is the cheap guard.
    """
    base = crossed_design(depth_effect=0.08, hidden_effect=0.2, noise=0.01)
    # Twelve seeds rather than three: five blocks per other-level is enough for
    # the block count to exceed each block's attainable splits.
    design = base + [replace(entry, seed=entry.seed + offset) for entry in base for offset in (10, 20, 30)]
    contrast = crossed_contrast(design, STEPS)
    assert contrast["available"]
    for main in contrast["main"].values():
        assert main["p"] is not None
        assert 0.0 < main["p"] <= 1.0
        # A real, uniform effect must not come back as the least extreme value.
        assert main["p"] < 0.05


def test_seed_blocked_interaction_resolves_once_seeds_are_plentiful():
    """With enough seeds the well-calibrated test can clear the re-split floor.

    A consistent interaction is invisible to the conservative test at few seeds
    but becomes significant to the seed-blocked test as seeds accumulate, which is
    the whole reason the confirmation arm adds seeds rather than architectures.
    """
    interaction = 0.07
    results = []
    for depth in (5, 14):
        for hidden in (512, 896):
            for seed in range(42, 54):  # twelve seeds
                for rep in (0, 1):
                    cost = (
                        interaction
                        if (depth == 14 and hidden == 512)
                        else 0.0
                    )
                    results.append(
                        crossed_result(
                            f"D{depth}_H{hidden}_r{rep}",
                            3.0 + cost,
                            depth=depth,
                            hidden=hidden,
                            seed=seed,
                        )
                    )
    main = crossed_contrast(results, STEPS)["main"]["depth"]
    assert len(main["interaction_seed_contrasts"]) == 12
    # Every seed agrees, so the sign test is at its floor: 2/2^12.
    assert main["interaction_p_seed_blocked"] == pytest.approx(2 / 2 ** 12, abs=1e-12)
    assert main["interaction_p_seed_blocked_floor"] == pytest.approx(2 / 2 ** 12)
    assert main["interaction_nats"] == pytest.approx(interaction, abs=1e-9)


def test_decision_says_when_the_interaction_test_is_at_its_floor(tmp_path):
    """`p = 0.25` with three seeds is the most extreme result that test can
    express, and reading it as "probably no interaction" is an error this project
    has already made once on a different floor-bound statistic. The decision has
    to say so, and it has to say what would resolve it, rather than leaving a
    floor value sitting in a table for the reader to interpret.
    """
    design = []
    for depth in (5, 14):
        for hidden in (512, 896):
            for seed in (42, 43, 44):
                for rep in (0, 1):
                    # One arm carries a consistent interaction with seed jitter, so
                    # all three seed contrasts agree in sign and none dominates.
                    cost = (
                        (0.08 + 0.004 * (seed - 43)) if (depth == 14 and hidden == 512) else 0.0
                    )
                    design.append(
                        crossed_result(
                            f"D{depth}_H{hidden}_r{rep}",
                            3.0 + cost + 0.01 * (rep * 2 - 1),
                            depth=depth,
                            hidden=hidden,
                            seed=seed,
                        )
                    )

    text = decision_text(tmp_path, design, steps=STEPS, seeds=(42, 43, 44))
    assert "at its floor" in text
    assert "most extreme p 3 seeds can express" in text
    assert "more seeds, not a different test" in text
    assert "procedure floor 0.25" in text


def test_paired_contrast_detects_a_consistent_effect_at_its_floor():
    """The reason the head-paired design is worth its hours.

    Twelve paired differences give an exact floor of 2/2^12. The same data
    collapsed to arm means over three seeds would floor at 2/2^3 = 0.25, which is
    the floor that left the crossed design's interaction unresolved. Pairing is
    what buys the resolution, so it is asserted rather than assumed.
    """
    differences = [0.08, 0.09, 0.07, 0.10, 0.06, 0.08, 0.09, 0.11, 0.05, 0.07, 0.08, 0.10]
    result = paired_contrast_test([(3.0, 3.0 + d) for d in differences])
    assert result["available"]
    assert result["pairs"] == 12
    assert result["mean_difference"] == pytest.approx(sum(differences) / 12)
    assert result["method"] == "exact"
    assert result["assignments"] == 2 ** 12
    assert result["p"] == pytest.approx(result["min_achievable_p"])
    assert result["sign_consistent"]


def test_paired_contrast_does_not_find_an_effect_in_noise():
    rng = random.Random(0)
    pairs = [(3.0 + rng.gauss(0, 0.02), 3.0 + rng.gauss(0, 0.02)) for _ in range(12)]
    result = paired_contrast_test(pairs)
    assert result["p"] > 0.05
    assert not result["sign_consistent"]


def test_paired_contrast_reports_direction_not_just_magnitude():
    """A significant negative effect must not read as a positive one."""
    negative = [(3.0, 3.0 - 0.08 - 0.01 * i) for i in range(12)]
    result = paired_contrast_test(negative)
    assert result["mean_difference"] < 0
    assert result["negative"] == 12
    assert result["positive"] == 0
    assert result["p"] < 0.01


def test_paired_contrast_is_unavailable_without_pairs():
    result = paired_contrast_test([])
    assert not result["available"]
    assert result["pairs"] == 0


def paired_design_results(
    head_effect: float = 0.09,
    width_effect: float = -0.10,
    seeds: tuple[int, ...] = (48, 49, 50),
    pairs: int = 4,
    head_effect_by_hidden: dict[int, float] | None = None,
    noise: float = 0.0,
):
    """A synthetic head-paired design built from the real zoo generator.

    Loss is additive in the two factors by construction, so the paired analysis
    has a known answer to recover: the head contrast should return exactly
    ``head_effect`` regardless of the width contrast, because each pair holds the
    width fixed.

    ``head_effect_by_hidden`` plants a *level-dependent* head effect instead - the
    shape the confirmation arm exists to decide - and ``noise`` adds a per-run
    jitter, which is what makes the per-seed interaction contrasts disagree in sign
    and is therefore how the "absent" branch gets exercised.
    """
    from act.search.architecture_zoo import generate_head_paired_zoo

    zoo = generate_head_paired_zoo(
        head_counts=(8, 16), pairs=pairs, hidden_sizes=(512, 896), depths=(10,)
    )
    results = []
    for candidate in zoo:
        genome = candidate.genome
        for seed in seeds:
            head = (
                head_effect
                if head_effect_by_hidden is None
                else head_effect_by_hidden[genome.hidden_size]
            )
            jitter = (
                0.0
                if noise == 0.0
                else random.Random(f"{candidate.name}:{seed}").gauss(0.0, noise)
            )
            loss = (
                3.0
                + (head if genome.layers[0].heads == 16 else 0.0)
                + (width_effect if genome.hidden_size == 896 else 0.0)
                + jitter
            )
            results.append(
                replace(
                    result(candidate.name, loss, seed=seed, steps=STEPS),
                    family=candidate.family,
                    description=candidate.description,
                    total_params=candidate.total_params,
                    active_params=candidate.active_params,
                    depth=genome.depth,
                    hidden_size=genome.hidden_size,
                    attention_heads=genome.layers[0].heads,
                    expert_count=len(genome.layers[0].experts),
                    active_experts=genome.layers[0].active_experts,
                    expert_width=genome.layers[0].experts[0].width,
                )
            )
    return results


def test_paired_design_contrast_recovers_the_held_fixed_gene():
    """Each pair holds every other gene fixed, so the contrast is that gene alone."""
    contrast = paired_design_contrast(paired_design_results(head_effect=0.09), "attention_heads")
    assert contrast["available"]
    assert contrast["pairs"] == 12  # 4 pairs x 3 seeds
    assert contrast["mean_difference"] == pytest.approx(0.09, abs=1e-9)
    assert contrast["p"] == pytest.approx(contrast["min_achievable_p"])
    assert contrast["sign_consistent"]


def test_paired_design_contrast_is_unaffected_by_the_other_factor():
    """The width contrast must not leak into the head contrast.

    This is the property that makes pairing worth more than band matching: in a
    band-matched design the two factors correlate, so each carries some of the
    other. Here the width effect is varied threefold across runs and the head
    estimate must not move.
    """
    for width in (-0.05, -0.10, -0.30):
        contrast = paired_design_contrast(
            paired_design_results(head_effect=0.09, width_effect=width),
            "attention_heads",
        )
        assert contrast["mean_difference"] == pytest.approx(0.09, abs=1e-9)


def test_paired_design_contrast_is_unavailable_without_a_paired_structure():
    results = [
        replace(result(f"a{i}", 3.0 + i / 10, seed=seed), hidden_size=512, depth=8)
        for i in range(4)
        for seed in (42, 43, 44)
    ]
    contrast = paired_design_contrast(results, "attention_heads")
    assert not contrast["available"]


def test_paired_contrasts_are_detected_from_the_data_only():
    """Auto-detection must see a paired design and reject a band-matched one.

    A report that claimed a paired contrast the ledger does not support would
    overstate the matching, so this checks both directions: the synthetic paired
    design is found, and the real band-matched arms are not.
    """
    found = _paired_contrasts_at(paired_design_results(), STEPS)
    assert set(found) == {"attention_heads"}
    assert found["attention_heads"]["pairs"] == 12

    # A band-matched set: candidates differ in several genes, so no gene forms
    # complete pairs with everything else held fixed.
    banded = []
    for index, (depth, hidden, heads) in enumerate(
        [(5, 512, 8), (14, 896, 14), (8, 832, 13), (10, 512, 8)]
    ):
        for seed in (42, 43, 44):
            banded.append(
                replace(
                    result(f"b{index}", 3.0 + index / 10, seed=seed),
                    depth=depth,
                    hidden_size=hidden,
                    attention_heads=heads,
                    expert_count=8,
                    active_experts=4,
                    expert_width=512,
                )
            )
    assert not _paired_contrasts_at(banded, STEPS)


def partial_head_paired_design():
    """v4's design with its last seed cut off after the 16-head half of every pair.

    That is the shape an interrupted sweep leaves: the seed carries fewer rows
    than the others, and every one of its groups is missing the same level.
    """
    design = paired_design_results(head_effect=0.09, seeds=(48, 49, 50))
    return [
        entry for entry in design if entry.seed != 50 or entry.attention_heads == 16
    ]


def test_paired_contrast_survives_an_unfinished_seed():
    """v4's headline test must not be deleted by a half-written last seed.

    Any incomplete group used to discard the whole *gene*, so an analysis run
    while the sweep was still training reported no paired contrast at all - an
    absence a reader would take for "this design is not paired". The contrast is
    computed from the complete pairs and the drop is named instead.
    """
    complete = _paired_contrasts_at(paired_design_results(seeds=(48, 49, 50)), STEPS)
    assert complete["attention_heads"]["pairs"] == 12
    assert complete["attention_heads"]["seeds"] == [48, 49, 50]
    assert "dropped_groups" not in complete["attention_heads"]

    found = _paired_contrasts_at(partial_head_paired_design(), STEPS)
    assert "attention_heads" in found
    contrast = found["attention_heads"]
    assert contrast["pairs"] == 8  # two whole seeds
    assert contrast["seeds"] == [48, 49]
    assert contrast["seeds_dropped"] == [50]
    assert len(contrast["dropped_groups"]) == 4
    # Fewer units means a higher floor, and the floor is printed beside the p, so
    # it has to move with the drop rather than stay at the design's 2/2^12.
    assert contrast["min_achievable_p"] == pytest.approx(2 / 2 ** 8)
    assert contrast["mean_difference"] == pytest.approx(0.09, abs=1e-9)


def test_a_seed_missing_whole_pairs_is_disclosed_not_silent():
    """A short seed can leave no *half* pair, and then nothing above notices.

    If an unfinished sweep stops between pairs, every group it produced is
    complete and every group check passes - the unit count simply drops and the
    floor beside the p rises with no reason given. The pair count per seed is the
    only place that shows, so it is reported.
    """
    design = paired_design_results(head_effect=0.09, seeds=(48, 49, 50))

    def pair_key(entry):
        return (
            entry.depth,
            entry.hidden_size,
            entry.expert_count,
            entry.active_experts,
            entry.expert_width,
        )

    last_two = sorted({pair_key(entry) for entry in design})[-2:]
    partial = [
        entry
        for entry in design
        if not (entry.seed == 50 and pair_key(entry) in last_two)
    ]
    contrast = _paired_contrasts_at(partial, STEPS)["attention_heads"]
    assert contrast["pairs"] == 10  # 4 + 4 whole pairs, plus 2 from seed 50
    assert "dropped_groups" not in contrast
    assert contrast["seeds_partial"] == [{"seed": 50, "pairs": 2, "of": 4}]
    assert contrast["min_achievable_p"] == pytest.approx(2 / 2 ** 10)


def test_a_pair_incomplete_on_a_full_seed_still_refuses():
    """A duplicated member is not an unfinished run, and must not be dropped.

    The seed here holds every other seed's row count, so nothing is *unwritten*:
    the pairing is malformed. Reporting the pairs that happen to be complete
    would present a broken zoo as a matched design.
    """
    design = paired_design_results(seeds=(48, 49))
    donor = [entry for entry in design if entry.seed == 48 and entry.attention_heads == 16]
    malformed = [entry for entry in design if entry.seed != 48 or entry.attention_heads != 8]
    malformed += [replace(entry, name=entry.name + "_dup") for entry in donor]
    rows = [sum(1 for entry in group if entry.seed == seed) for group in (design, malformed) for seed in (48,)]
    assert rows[0] == rows[1], rows  # same row count, different structure
    assert "attention_heads" not in _paired_contrasts_at(malformed, STEPS)


def test_decision_discloses_dropped_pairs(tmp_path):
    text = decision_text(tmp_path, partial_head_paired_design(), seeds=(48, 49, 50))
    assert "Paired contrast" in text
    assert "**Groups dropped: 4**" in text
    assert "seed(s) [50]" in text
    clean = decision_text(
        tmp_path, paired_design_results(seeds=(48, 49, 50)), seeds=(48, 49, 50)
    )
    assert "Groups dropped" not in clean


def test_decision_states_the_paired_contrast(tmp_path):
    results = paired_design_results(head_effect=0.09)
    text = decision_text(tmp_path, results, steps=STEPS, seeds=(48, 49, 50))
    assert "Paired contrast" in text
    assert "`attention_heads`" in text
    assert "Sign consistency" in text
    # The floor must be stated, because it is what the p-value is bounded by.
    assert "Procedure floor" in text


# --- the level split: a pooled contrast can cancel two levels ----------------


def level_dependent_head_design(seeds=(48, 49, 50), noise: float = 0.0):
    """A head effect that helps at wide width and does nothing at narrow.

    This is the shape v4 measured and could not decide: +0.0109 nats at hidden 512
    against -0.0462 at 896, which pools to -0.0177 and reads as "no head effect"
    whether or not the level split is real.
    """
    return paired_design_results(
        head_effect_by_hidden={512: 0.01, 896: -0.05}, seeds=seeds, noise=noise
    )


def test_paired_level_split_recovers_each_levels_own_effect():
    design = level_dependent_head_design()
    pooled = paired_design_contrast(design, "attention_heads")
    split = paired_level_split(design, "attention_heads", "hidden_size")
    assert split["available"]
    assert split["other_gene"] == "hidden_size"
    assert sorted(split["levels"]) == ["512", "896"]
    assert split["levels"]["512"]["mean_difference"] == pytest.approx(0.01, abs=1e-9)
    wide = split["levels"]["896"]
    assert wide["mean_difference"] == pytest.approx(-0.05, abs=1e-9)
    assert wide["pairs"] == 6  # two base configurations x three seeds
    # Consistent at the wide level alone, and the split says so rather than
    # letting the pooled average hide it.
    assert wide["sign_consistent"]
    assert wide["p"] == pytest.approx(wide["min_achievable_p"])
    assert not split["sign_agreement"]
    # The pooled number is exactly the average of the two levels, which is the
    # reason the split exists: neither level's number can be read off it.
    assert pooled["mean_difference"] == pytest.approx((0.01 - 0.05) / 2, abs=1e-9)


def test_paired_level_split_reports_agreement_without_calling_it_a_test():
    split = paired_level_split(
        paired_design_results(head_effect=0.09), "attention_heads", "hidden_size"
    )
    assert split["available"] and split["sign_agreement"]
    assert "not evidence that the levels differ" in split["reading"]


def test_paired_level_split_refuses_rather_than_reporting_a_subset():
    """A split missing a level is a different question, not a smaller answer."""
    design = paired_design_results()
    broken = [
        entry
        for entry in design
        if not (entry.hidden_size == 896 and entry.attention_heads == 16)
    ]
    split = paired_level_split(broken, "attention_heads", "hidden_size")
    assert not split["available"]
    assert "no complete pairs" in split["reason"]
    assert split["levels_present"] == ["512"]
    one_level = paired_level_split(
        [entry for entry in design if entry.hidden_size == 512],
        "attention_heads",
        "hidden_size",
    )
    assert not one_level["available"]
    assert "does not vary" in one_level["reason"]


def test_level_splits_are_reported_for_a_two_level_partner_only():
    """A partner with two levels is a crossed factor, so the split is the question.

    Splitting on a multi-level gene would add descriptive slices and no single
    interaction to read, and the design that needs this rule crossed exactly two.
    """
    found = _paired_level_splits_at(paired_design_results(), STEPS)
    assert "attention_heads" in found
    assert set(found["attention_heads"]) == {"hidden_size"}


# --- the head-pairing rule, branch by branch ---------------------------------

SIX = (48, 49, 50, 51, 52, 53)


def head_rule_outcome(results, seeds):
    """The outcome the rule reads, as the CLI builds it from a summary."""
    return effect_summary(results, STEPS, list(seeds), samples=2_000)


def test_head_rule_reads_the_level_split_rather_than_the_pooled_contrast():
    outcome = head_rule_outcome(level_dependent_head_design(seeds=SIX), SIX)
    rule = head_pairing_decision(outcome)
    assert rule["branch"] == "heads-free-at-wide"
    assert rule["level_split"]["wide_level"] == "896"
    assert rule["level_split"]["wide_nats"] == pytest.approx(-0.05, abs=1e-9)
    # The pooled contrast is a fifth of the wide level's effect and would read as
    # nothing on its own; the branch comes from the level, not the average.
    assert rule["pooled_head_contrast"]["nats"] == pytest.approx(-0.02, abs=1e-9)
    assert all(row["met"] for row in rule["conditions"])


def test_head_rule_says_open_when_the_floor_is_above_the_threshold():
    """Three seeds cannot resolve this interaction however consistent it is."""
    seeds = (48, 49, 50)
    outcome = head_rule_outcome(level_dependent_head_design(seeds=seeds), seeds)
    rule = head_pairing_decision(outcome)
    assert rule["branch"] == "open-floor-limited"
    assert rule["interaction"]["p_seed_blocked"] == pytest.approx(0.25)
    assert "above the declared" in rule["note"]
    # Every seed agreed, which is exactly what a floor-limited arm looks like.
    conditions = {row["condition"]: row["met"] for row in rule["conditions"]}
    assert conditions["every seed contrast the same sign"]
    assert not conditions["seed-blocked interaction p < 0.05"]


def test_head_rule_closes_the_axis_when_the_seed_agreement_was_luck():
    outcome = head_rule_outcome(
        paired_design_results(head_effect=0.0, noise=0.05, seeds=SIX), SIX
    )
    rule = head_pairing_decision(outcome)
    assert rule["branch"] == "head-axis-closed"
    contrasts = rule["interaction"]["contrasts"]
    assert len({value > 0 for value in contrasts}) == 2, contrasts
    assert "cheaper" in rule["reading"]


def test_head_rule_gates_on_the_width_attribution_first():
    """A head reading from a design whose width contrast failed is not a reading."""
    outcome = head_rule_outcome(
        paired_design_results(
            head_effect_by_hidden={512: 0.01, 896: -0.05},
            width_effect=+0.05,
            seeds=SIX,
        ),
        SIX,
    )
    rule = head_pairing_decision(outcome)
    assert rule["branch"] == "width-fails"
    conditions = {row["condition"]: row["met"] for row in rule["conditions"]}
    assert not conditions["hidden_size significant, negative (p < 0.01)"]


def test_head_rule_is_unavailable_without_a_paired_head_design():
    outcome = head_rule_outcome(crossed_design(depth_effect=0.08, hidden_effect=0.2), (42, 43, 44))
    rule = head_pairing_decision(outcome)
    assert not rule["available"]
    assert rule["branch"] == "unavailable"
    assert rule["conditions"] == []


def test_head_rule_has_a_branch_for_a_real_but_uniform_head_effect():
    """A head effect identical at both widths is not "worth nothing at both".

    This case is missing from the arm's table as first written, and the test suite
    found it rather than the arm: the interaction is zero, so every
    interaction-based branch is wrong about it, and only the pooled contrast can
    say whether the head axis is a real uniform lever or closed at zero.
    """
    results = paired_design_results(head_effect=-0.09, seeds=SIX)
    outcome = head_rule_outcome(results, SIX)
    rule = head_pairing_decision(outcome)
    assert rule["branch"] == "head-uniform"
    assert "add heads *globally*" in rule["reading"]
    assert "uniform across" in rule["note"]
    # The same design with no head effect at all closes the axis instead, so the
    # branch is decided by the pooled contrast rather than by the interaction being
    # zero in both cases.
    closed = head_pairing_decision(
        head_rule_outcome(paired_design_results(head_effect=0.0, seeds=SIX), SIX)
    )
    assert closed["branch"] == "head-axis-closed"
    assert closed["pooled_head_contrast"]["nats"] == pytest.approx(0.0, abs=1e-9)


def test_decision_renders_the_head_rule_and_the_level_split(tmp_path):
    results = level_dependent_head_design(seeds=SIX)
    outcome = head_rule_outcome(results, SIX)
    outcome["head_pairing_decision"] = head_pairing_decision(outcome)
    path = tmp_path / "decision.md"
    write_decision(path, outcome)
    text = path.read_text()
    assert "Paired level split" in text
    assert "Pre-registered rule: is the head effect level-dependent?" in text
    assert "HEAD EFFECT IS REAL AT WIDE WIDTH" in text
    assert "hidden_size=896" in text
    assert "| Condition | Value | Met |" in text


# --- joint structural fit --------------------------------------------------


#: Depth and hidden deliberately crossed rather than tied: making one an affine
#: function of the other makes them exact duplicates, which the fit drops on
#: purpose, so a fixture that ties them never exercises the fit at all.
UNCORRELATED_LAYOUTS = (
    (6, 512), (7, 896), (8, 576), (9, 640), (10, 832), (11, 704), (12, 768),
    (13, 960), (6, 704), (8, 512), (10, 896), (12, 576), (7, 960), (11, 640),
)


def test_joint_fit_recovers_known_coefficients():
    """A fit that cannot recover a planted relationship is not a fit. The two
    genes carry opposite signs, so a sign or axis mix-up inside the solver would
    show up rather than cancel."""
    rng = random.Random(7)
    results = []
    for index, (depth, hidden) in enumerate(UNCORRELATED_LAYOUTS):
        loss = 3.2 + 0.08 * (depth - 10) - 0.0018 * (hidden - 700) + rng.gauss(0, 0.004)
        for seed in (42, 43):
            results.append(
                replace(
                    result(f"arch{index}", loss, seed=seed),
                    depth=depth,
                    hidden_size=hidden,
                )
            )
    fit = joint_structural_fit(results, STEPS)
    assert fit["available"]
    genes = {row["gene"]: row for row in fit["genes"]}
    assert genes["depth"]["beta"] > 0.4
    assert genes["hidden_size"]["beta"] < -0.5
    assert fit["r_squared"] > 0.9
    assert fit["dropped"] == []


def test_joint_fit_drops_an_exactly_redundant_gene():
    """The family fixes heads = hidden/64, so those two columns are identical
    after standardisation. Keeping both makes the whole fit rank-deficient, and
    answering that with zero coefficients would read as "this gene does nothing"
    when the truth is that it is perfectly redundant. A third, genuinely
    independent gene keeps the remaining fit identifiable, which is what makes
    the difference between the two failure modes visible."""
    layouts = ((512, 4), (576, 16), (640, 8), (704, 12), (768, 6), (832, 10), (896, 14), (960, 8))
    results = []
    for index, (hidden, experts) in enumerate(layouts):
        loss = 3.6 - 0.0012 * (hidden - 700) + 0.004 * (experts - 10)
        for seed in (42, 43):
            results.append(
                replace(
                    result(f"arch{index}", loss, seed=seed),
                    hidden_size=hidden,
                    attention_heads=hidden // 64,
                    expert_count=experts,
                )
            )
    fit = joint_structural_fit(results, STEPS)
    assert fit["available"]
    dropped = {row["gene"]: row for row in fit["dropped"]}
    assert dropped["attention_heads"]["duplicate_of"] == "hidden_size"
    assert abs(dropped["attention_heads"]["spearman"]) == pytest.approx(1.0)
    # The surviving fit must be a real one, not the degenerate all-zero answer.
    assert fit["r_squared"] > 0.5
    assert {row["gene"] for row in fit["genes"]} == {"hidden_size", "expert_count"}


def test_joint_fit_refuses_rather_than_reporting_zeros_without_structure():
    """Unrecorded structural genes are the failure `structural_fields` documents.
    Returning beta=0 here would turn "not joined" into "no effect"."""
    results = [result(f"arch{index}", 3.5 + index * 0.01) for index in range(5)]
    fit = joint_structural_fit(results, STEPS)
    assert fit["available"] is False
    assert "structural" in fit["reason"]
    assert fit["structural_axes_unrecorded"] is True


def test_joint_fit_flags_jointly_predictive_but_unseparable_genes():
    """Two genes that move together at rho = -0.8 can drive the fit without
    either being individually attributable. Reporting the model R^2 without that
    split would repeat the error the per-axis table exists to avoid."""
    rng = random.Random(11)
    # Mostly-anti-correlated, with two inversions so the pair is entangled rather
    # than perfectly redundant: exactly -1 would be dropped as a duplicate.
    layouts = ((6, 896), (7, 832), (8, 768), (9, 800), (10, 704), (11, 640), (12, 672), (13, 576))
    results = []
    for index, (depth, hidden) in enumerate(layouts):
        loss = 3.2 + 0.03 * (depth - 10) - 0.0012 * (hidden - 700) + rng.gauss(0, 0.006)
        for seed in (42, 43):
            results.append(
                replace(
                    result(f"arch{index}", loss, seed=seed),
                    depth=depth,
                    hidden_size=hidden,
                )
            )
    fit = joint_structural_fit(results, STEPS)
    assert fit["available"]
    assert fit["r_squared"] > 0.8
    pairs = {frozenset((row["a"], row["b"])) for row in fit["entangled"]}
    assert frozenset(("depth", "hidden_size")) in pairs
    assert "cannot be told apart" in fit["caveat"]


def test_recorded_v1_screening_spread_is_structural_not_cost():
    """Pinned against v1's real ledgers, and a correction to what this project
    first concluded from them.

    `ARCH_SEARCH_FINDINGS.md` section 5 read v1's cost-axis table as "nothing
    structural explains the screening spread". It was the missing structural join:
    v1's rows record zeros for those six genes, so they drop out of the spread and
    the lead is drawn from the cost axes alone. Joining the genome recovers them,
    and the same 13 architectures then say the opposite - the spread is ~92%
    explained by residual width and depth, jointly, while every *cost* axis stays
    flat. That is v2's attribution replicating on an independent design, which is
    the strongest evidence this phase has, so it must not be able to rot silently.
    """
    directory = Path("artifacts/arch-zoo-100m-v1")
    ledgers = sorted(directory.glob("stage1_seed*_results.jsonl"))
    if not ledgers:
        pytest.skip("arch-zoo-100m-v1 screening ledgers not present")

    raw = load_results(ledgers)
    joined = with_structure(raw, directory / "zoo_definition.json")

    # The ledgers really do predate the structural fields, which is why the join
    # is the whole difference between section 5's claim and this one.
    assert all(row.depth == 0 for row in raw)
    assert {row.depth for row in joined} != {0}

    axes = {row["axis"]: row for row in axis_association(joined, STEPS)["axes"]}
    assert axes["hidden size"]["spearman"] < -0.9
    assert axes["hidden size"]["p"] < 0.001
    assert axes["depth"]["spearman"] > 0.8
    # No cost axis explains it - this half of section 5 does hold.
    for label in ("total parameters", "active parameters", "contracted FLOPs/token"):
        assert axes[label]["p"] > 0.05, label

    fit = joint_structural_fit(joined, STEPS)
    assert fit["available"]
    assert fit["architectures"] == 13
    assert fit["r_squared"] > 0.85
    # The model is real; the individual coefficients are not, because the two
    # genes move together at rho ~ -0.8. Both facts must survive together.
    assert fit["p_model"] < 0.01
    assert frozenset(("hidden_size", "depth")) in {
        frozenset((row["a"], row["b"])) for row in fit["entangled"]
    }
    # heads = hidden/64 makes those two columns identical, so one is dropped.
    assert {row["gene"] for row in fit["dropped"]} == {"attention_heads"}

    # Without the join the report must say it cannot attribute, not stay silent.
    unjoined = axis_association(raw, STEPS)["collinearity"]
    assert len(unjoined["structural_axes_unrecorded"]) == 6
    assert "no structural attribution can be made" in unjoined["caveat"]


# --- structural signal against training length -----------------------------


def test_structural_signal_by_length_separates_a_short_from_a_long_signal():
    """The claim this function exists to make is a *comparison* across lengths,
    so it has to be sensitive to a gene that only becomes readable with training.
    Here the planted width effect is one third as large at 400 steps, and the
    function must report a weaker fit there rather than an average of the two.
    """
    rng = random.Random(3)
    results = []
    for index, (depth, hidden) in enumerate(UNCORRELATED_LAYOUTS):
        base = 3.2 + 0.05 * (depth - 10) - 0.0012 * (hidden - 700)
        for steps, width_scale in ((400, 0.35), (1200, 1.0)):
            loss = base - width_scale * 0.0015 * (hidden - 700) + rng.gauss(0, 0.004)
            for seed in (42, 43):
                results.append(
                    replace(
                        result(f"arch{index}", loss, seed=seed, steps=steps),
                        depth=depth,
                        hidden_size=hidden,
                    )
                )

    signal = structural_signal_by_length(results)
    assert set(signal) == {"400", "1200", "agreement"}
    short, long = signal["400"], signal["1200"]
    assert short["available"] and long["available"]
    assert short["architectures"] == long["architectures"] == len(UNCORRELATED_LAYOUTS)
    # Both lengths see the same leading gene and sign; only the strength differs.
    assert signal["agreement"]["same_leading_gene"] is True
    assert signal["agreement"]["same_leading_sign"] is True
    assert long["r_squared"] > short["r_squared"]
    assert long["genes"][0]["beta"] != pytest.approx(short["genes"][0]["beta"])


def test_structural_signal_by_length_is_empty_without_common_architectures():
    """Comparing a length's fit against a *different* architecture set would
    measure the set, not the length, so a set with no overlap must report
    nothing rather than two fits over different candidates."""
    disjoint = [
        result("only-400", 3.5, seed=42, steps=400),
        result("only-400b", 3.6, seed=43, steps=400),
        result("only-400c", 3.7, seed=44, steps=400),
        result("only-1200", 3.5, seed=42, steps=1200),
        result("only-1200b", 3.6, seed=43, steps=1200),
        result("only-1200c", 3.7, seed=44, steps=1200),
    ]
    assert structural_signal_by_length(disjoint) == {}
    # A single length has nothing to compare against.
    assert structural_signal_by_length([r for r in disjoint if r.steps == 400]) == {}


def test_recorded_structural_signal_is_present_at_a_third_of_the_cost():
    """A search-budget claim, pinned against the real ledgers.

    The GPU budget for this phase is binding, and the question that relieves it
    is how little training a screen needs. On v2's eight matched candidates the
    *same* joint fit is significant at all three lengths with essentially the
    same R^2, so a 400-step screen (~1.1 GPU-hours) recovers what a 1200-step run
    (~3.2) establishes. Without this pin a later change to the fit or the
    `--also` pooling could quietly turn "screening suffices" into a claim this
    project no longer has evidence for.
    """
    v2_dir = Path("artifacts/arch-zoo-100m-v2")
    v2_ledgers = sorted(v2_dir.glob("*_seed*_results.jsonl"))
    if not v2_ledgers:
        pytest.skip("arch-zoo-100m-v2 ledgers not present")
    pooled = _horizon_results() + with_structure(
        load_results(v2_ledgers), v2_dir / "zoo_definition.json"
    )

    signal = structural_signal_by_length(pooled)
    lengths = [key for key in signal if key != "agreement"]
    assert len(lengths) >= 3, lengths
    assert signal["agreement"]["same_leading_gene"] is True
    assert signal["agreement"]["same_leading_sign"] is True

    for length in lengths:
        row = signal[length]
        assert row["available"], length
        assert row["architectures"] == 8, length
        # Jointly predictive at every length, including the shortest.
        assert row["p_model"] < 0.05, (length, row["p_model"])
        assert row["r_squared"] > 0.9, (length, row["r_squared"])
    # And the shortest length is not merely "also significant" - it is within the
    # same explanatory band as the longest, which is the whole claim.
    shortest, longest = signal[lengths[0]], signal[lengths[-1]]
    assert abs(shortest["r_squared"] - longest["r_squared"]) < 0.06


# --- procedure floors on the crossed contrasts -----------------------------


def test_crossed_main_effect_reports_its_exact_floor_signal(tmp_path):
    """The main effects are the phase's headline numbers and they land on the
    enumeration's floor. A record that printed only `p = 4.3e-05` would invite the
    reader to treat it as "this is how significant the effect is"; what it actually
    says is that the observed re-splitting was matched by nothing but its own
    mirror. The distinction has to be machine-checkable, not just prose.
    """
    design = crossed_design(depth_effect=0.09, hidden_effect=0.22, noise=0.005)
    contrast = crossed_contrast(design, STEPS)
    assert contrast["available"]
    for gene, main in contrast["main"].items():
        assert main["method"] == "exact", gene
        assert main["assignments"] > 1, gene
        assert main["min_achievable_p"] == pytest.approx(2.0 / main["assignments"])
        # A large, clean effect must land on the floor rather than merely below 0.05.
        assert main["at_floor"] is True, gene
        assert main["p"] == pytest.approx(main["min_achievable_p"])

    text = decision_text(tmp_path, design, steps=STEPS, seeds=(42, 43, 44))
    assert "Procedure floor" in text
    assert "at floor" in text
    assert "largest the design can express" in text


def test_crossed_main_effect_is_not_labelled_at_floor_when_it_is_not(tmp_path):
    """The floor label is a claim about the design, so it must not be applied to a
    weak or noisy effect. Same design, one arm's difference buried in noise: the
    p must rise off the floor and the label must clear."""
    design = crossed_design(depth_effect=0.02, hidden_effect=0.03, noise=0.05)
    contrast = crossed_contrast(design, STEPS)
    main = contrast["main"]["hidden_size"]
    assert main["p"] > main["min_achievable_p"]
    assert main["at_floor"] is False
    text = decision_text(tmp_path, design, steps=STEPS, seeds=(42, 43, 44))
    assert "largest the design can express" not in text


def test_monte_carlo_main_effect_still_reports_a_floor(tmp_path):
    """Past the enumeration cap the exact test becomes Monte-Carlo, whose floor is
    1/(samples+1). Leaving it unstated would let a sampling resolution pass for a
    strong result - the same failure mode as quoting a floor-bound p as though it
    were the evidence."""
    base = crossed_design(depth_effect=0.09, hidden_effect=0.22, noise=0.005)
    # Six seeds takes the assignment count over the default 1e6 cap.
    six = base + [
        replace(entry, seed=entry.seed + offset)
        for entry in base
        for offset in (10, 20, 30)
    ]
    contrast = crossed_contrast(six, STEPS, samples=2_000)
    assert contrast["available"]
    for gene, main in contrast["main"].items():
        assert main["method"] == "monte-carlo", gene
        assert main["assignments"] > 1_000_000, gene
        assert main["min_achievable_p"] == pytest.approx(1 / 2_001)
        assert main["at_floor"] is True, gene

    text = decision_text(
        tmp_path, six, steps=STEPS, seeds=tuple(sorted({e.seed for e in six})), samples=2_000
    )
    assert "Procedure floor" in text
    assert "raise `--samples`" in text
    # And the floor value itself must be visible, not just the label.
    assert "5e-06" in text or "4.999975e-06" in text


def test_recorded_crossed_main_effects_are_at_their_enumeration_floor():
    """Pinned against v3's real ledgers, because this is a statement about the
    phase's headline evidence. Both main effects were reported at p = 4.287e-05
    over 46656 assignments, which is exactly 2/46656 - the smallest an exact
    two-sided test can report. If a later change made the test re-splittable in
    more ways the label would silently stop being true, and the record would
    overstate the p-value's meaning."""
    directory = Path("artifacts/arch-zoo-100m-v3-crossed")
    ledgers = sorted(directory.glob("*_seed*_results.jsonl"))
    if not ledgers:
        pytest.skip("arch-zoo-100m-v3-crossed ledgers not present")
    results = with_structure(
        load_results(ledgers), directory / "zoo_definition.json"
    )
    contrast = crossed_contrast(results, 1200, samples=20_000)
    assert contrast["available"]
    main = contrast["main"]["hidden_size"]
    assert main["method"] == "exact"
    assert main["assignments"] == 46656
    assert main["p"] == pytest.approx(2 / 46656)
    assert main["min_achievable_p"] == pytest.approx(2 / 46656)
    assert main["at_floor"] is True
    # The effect size, which is the quantity that is *not* floor-bound.
    assert main["nats"] < -0.15


# --- crossed-design robustness ---------------------------------------------


def test_crossed_contrast_refuses_an_unbalanced_design_without_crashing():
    """Regression: a lopsided factorial used to raise ZeroDivisionError.

    Dropping one candidate from v3 leaves its D14/H896 cell with one run on each
    depth instead of two. The old code checked only that cells were *populated*,
    then derived the per-level block count from the seed list, so it sliced an
    empty level and divided by zero - a crash inside `write_decision`, which runs
    at the end of a sweep. The honest answer is a refusal, because a level mean
    over fewer blocks than the other is not a contrast.
    """
    directory = Path("artifacts/arch-zoo-100m-v3-crossed")
    ledgers = sorted(directory.glob("*_seed*_results.jsonl"))
    if not ledgers:
        pytest.skip("arch-zoo-100m-v3-crossed ledgers not present")
    results = with_structure(load_results(ledgers), directory / "zoo_definition.json")
    assert crossed_contrast(results, 1200, samples=2_000)["available"] is True

    names = sorted({entry.name for entry in results})
    for dropped in names:
        reduced = [entry for entry in results if entry.name != dropped]
        contrast = crossed_contrast(reduced, 1200, samples=2_000)
        # A design with 2 candidates per cell cannot lose one and stay balanced.
        assert contrast["available"] is False, dropped
        assert "balanced" in contrast["reason"], dropped

    # And a single candidate per cell is a perfectly valid crossed design.
    single = [
        entry
        for entry in results
        if entry.name in {
            "xhihi_D14_H896_E4_A2_W288",
            "xhilo_D14_H512_E4_A4_W1024",
            "xlohi_D5_H896_E4_A4_W1536",
            "xlolo_D5_H512_E12_A2_W1152",
        }
    ]
    balanced = crossed_contrast(single, 1200, samples=2_000)
    assert balanced["available"] is True
    assert set(balanced["main"]) == {"depth", "hidden_size"}


def test_crossed_contrast_reports_the_seeds_that_finished():
    """An unfinished seed must cost resolution, not the whole contrast.

    The confirmation arm's last seed is still being written when its analysis is
    first run, and the cells it has not reached yet make the design incomplete.
    Refusing the contrast there deletes the phase's headline test from a report
    over a run that is merely not finished - which is what happened: the section
    was absent from stdout with nothing saying why. The seeds that finished are
    reported, and the ones dropped are named.
    """
    design = crossed_design(depth_effect=0.08, hidden_effect=0.20)
    complete = crossed_contrast(design, STEPS, samples=2_000)
    assert complete["available"]
    assert complete["seeds_dropped"] == []

    # Seeds 42 and 43 whole; seed 44 stops after its narrow half.
    partial = [entry for entry in design if entry.seed != 44 or entry.hidden_size == 512]
    contrast = crossed_contrast(partial, STEPS, samples=2_000)
    assert contrast["available"]
    assert contrast["seeds"] == [42, 43]
    assert contrast["seeds_dropped"] == [44]
    # The dropped seed carries no effect of its own here, so the estimates must
    # be unchanged rather than merely re-based.
    for gene in ("depth", "hidden_size"):
        assert contrast["main"][gene]["nats"] == pytest.approx(
            complete["main"][gene]["nats"]
        ), gene


def test_crossed_contrast_honours_the_seeds_it_is_given():
    """A caller can read the arm over the seeds that finished, without an edit.

    This is what makes an in-flight confirmation arm analysable: naming its
    complete seeds must give exactly those, not the ledger's full seed set.
    """
    design = crossed_design(depth_effect=0.08, hidden_effect=0.20)
    contrast = crossed_contrast(design, STEPS, samples=2_000, seeds=[42, 43])
    assert contrast["available"]
    assert contrast["seeds"] == [42, 43]
    assert contrast["seeds_dropped"] == []
    # A seed the ledger does not have is not a seed this design can use.
    assert crossed_contrast(design, STEPS, samples=2_000, seeds=[99])["available"] is False


def test_an_unbalanced_seed_is_not_droppable_unless_it_is_unfinished():
    """Dropping is for rows not written yet, so a defect cannot pass as one.

    Both cases look the same to a balance check; the row count per seed is what
    separates them. A seed holding every other seed's row count and still failing
    the balance check is a broken zoo, and a contrast built from a subset of it
    would misreport a design that never was crossed.
    """
    design = crossed_design(depth_effect=0.08, hidden_effect=0.20)
    # Move one of seed 43's D5/H896 runs into D5/H512: the seed still holds eight
    # runs, but that cell now has three on one arm and the other has one.
    skewed = [
        replace(entry, hidden_size=512)
        if entry.seed == 43 and entry.name == "D5_H896_r1"
        else entry
        for entry in design
    ]
    counts: dict[int, int] = {}
    for entry in skewed:
        counts[entry.seed] = counts.get(entry.seed, 0) + 1
    assert counts[43] == counts[42], counts  # nothing was left unwritten
    contrast = crossed_contrast(skewed, STEPS, samples=2_000)
    assert contrast["available"] is False
    assert "balanced" in contrast["reason"]


def test_decision_discloses_a_dropped_seed(tmp_path):
    """The drop has to reach the decision, or a 5-seed mean reads as a 6-seed one."""
    design = crossed_design(depth_effect=0.08, hidden_effect=0.20)
    partial = [entry for entry in design if entry.seed != 44 or entry.hidden_size == 512]
    text = decision_text(tmp_path, partial, seeds=(42, 43, 44))
    assert "Crossed factorial" in text
    assert "**Seeds dropped: [44].**" in text
    assert "over 2 seeds, not 3" in text
    # A complete design says nothing about drops, because there were none.
    clean = decision_text(tmp_path, design, seeds=(42, 43, 44))
    assert "Seeds dropped" not in clean
    assert "every one of its 3 seeds" in clean


def crossed_with_interaction(
    *,
    depth_effect: float,
    hidden_effect: float,
    interaction,
    seeds: tuple[int, ...],
):
    """A crossed design with a planted interaction, over any seed count.

    ``interaction`` is the extra depth cost at the *narrow* level (per seed, if a
    callable is given). That value *is* the 2x2 interaction - each main effect
    picks up half of it - so the design has a known answer for the rule to
    recover, rather than a right answer supplied by the test's assertion.
    """
    results = []
    for depth in (5, 14):
        for hidden in (512, 896):
            for seed in seeds:
                loss = 3.0 + (depth_effect if depth == 14 else 0.0)
                loss += hidden_effect if hidden == 512 else 0.0
                if depth == 14 and hidden == 512:
                    loss += interaction(seed) if callable(interaction) else interaction
                for rep in (0, 1):
                    results.append(
                        crossed_result(
                            f"D{depth}_H{hidden}_r{rep}",
                            loss,
                            depth=depth,
                            hidden=hidden,
                            seed=seed,
                        )
                    )
    return results


def replication_outcome(results, seeds):
    """The pooled outcome the rule reads: a full summary, as the CLI builds it."""
    return effect_summary(results, STEPS, list(seeds), samples=2_000)


SIX_SEEDS = (42, 43, 44, 45, 46, 47)


def test_replication_rule_resolves_a_consistent_interaction():
    # `hidden_effect` is a loss added at the *narrow* level, as in every other
    # design helper here, so a positive value is what reproduces the original
    # attribution's negative `hidden_size` main effect.
    design = crossed_with_interaction(
        depth_effect=0.08, hidden_effect=0.30, interaction=0.10, seeds=SIX_SEEDS
    )
    decision = replication_decision(replication_outcome(design, SIX_SEEDS))
    assert decision["available"]
    assert decision["branch"] == "replicates-and-resolves"
    assert all(row["met"] for row in decision["conditions"]), decision["conditions"]
    # Six agreeing contrasts floor the sign test at 2/2^6, which is what makes the
    # declared 0.05 reachable at all - the whole reason the arm buys a third pair
    # of seeds rather than another pair of architectures.
    assert decision["interaction"]["floor"] == pytest.approx(2 / 2 ** 6)
    assert decision["interaction"]["p_seed_blocked"] < 0.05


def test_replication_rule_reads_mixed_seed_contrasts_as_absent():
    """The *absent* branch is a result, not a failure: separable genes search cheaper."""
    design = crossed_with_interaction(
        depth_effect=0.08,
        hidden_effect=0.30,
        interaction=lambda seed: 0.12 if seed % 2 == 0 else -0.12,
        seeds=SIX_SEEDS,
    )
    decision = replication_decision(replication_outcome(design, SIX_SEEDS))
    assert decision["branch"] == "replicates-no-interaction"
    assert decision["conditions"][3]["met"] is False  # the interaction p cannot
    assert decision["conditions"][4]["met"] is False  # and the signs disagree
    # The main effects still have to hold, or the branch would mean something else.
    assert all(row["met"] for row in decision["conditions"][:3])


def test_replication_rule_calls_a_floor_bound_interaction_inconclusive():
    """A p that cannot clear the threshold says nothing about the interaction.

    At five seeds the sign test floors at 2/2^5 = 0.0625, above the declared 0.05,
    so an arm bought with one seed too few lands in the inconclusive branch even
    with every contrast agreeing. Reading that as an *absent* interaction is the
    same mistake the horizon arm made on a floor-bound statistic, so the branch is
    reported with the floor that bound it.
    """
    seeds = SIX_SEEDS[:-1]
    design = crossed_with_interaction(
        depth_effect=0.08, hidden_effect=0.30, interaction=0.10, seeds=seeds
    )
    decision = replication_decision(replication_outcome(design, seeds))
    assert decision["branch"] == "inconclusive"
    assert decision["conditions"][4]["met"] is True
    assert decision["interaction"]["floor"] == pytest.approx(2 / 2 ** 5)
    assert "above the declared 0.05" in decision["note"]


def test_replication_rule_fails_when_the_lever_is_weaker_than_declared():
    """The 1.5x clause is part of the replication, not decoration."""
    design = crossed_with_interaction(
        depth_effect=0.20, hidden_effect=0.20, interaction=0.0, seeds=SIX_SEEDS
    )
    decision = replication_decision(replication_outcome(design, SIX_SEEDS))
    assert decision["branch"] == "fails-to-replicate"
    assert decision["conditions"][2]["met"] is False
    assert decision["conditions"][0]["met"] is True  # signed and significant...
    assert decision["conditions"][1]["met"] is True  # ...but not 1.5x the other


def test_replication_rule_fails_on_a_flipped_sign():
    """A sign flip is the branch that makes this a replication rather than a re-fit."""
    design = crossed_with_interaction(
        depth_effect=0.08, hidden_effect=-0.30, interaction=0.10, seeds=SIX_SEEDS
    )
    decision = replication_decision(replication_outcome(design, SIX_SEEDS))
    assert decision["branch"] == "fails-to-replicate"
    assert decision["conditions"][0]["met"] is False
    assert decision["main"]["hidden_size"]["nats"] > 0


def test_replication_rule_is_unavailable_without_a_crossed_design():
    decision = replication_decision(
        {"crossed_contrast": {"available": False, "reason": "fewer than two binary structural genes"}}
    )
    assert decision["available"] is False
    assert decision["branch"] == "unavailable"
    assert "fewer than two binary" in decision["reason"]


def test_decision_renders_the_replication_rule(tmp_path):
    design = crossed_with_interaction(
        depth_effect=0.08, hidden_effect=0.30, interaction=0.10, seeds=SIX_SEEDS
    )
    outcome = replication_outcome(design, SIX_SEEDS)
    outcome["replication_decision"] = replication_decision(outcome)
    path = tmp_path / "decision.md"
    write_decision(path, outcome)
    text = path.read_text()
    assert "Pre-registered rule: does the attribution replicate?" in text
    assert "**REPLICATES, INTERACTION RESOLVES**" in text
    assert "every seed contrast the same sign" in text

    # Without the rule the decision is unchanged: it is opt-in, because the rule
    # was declared for one arm's seed count rather than for the analysis at large.
    plain = tmp_path / "plain.md"
    write_decision(plain, replication_outcome(design, SIX_SEEDS))
    assert "Pre-registered rule" not in plain.read_text()


def paired_crossed_results():
    """The crossed arm and its fresh-seed confirmation, pooled as the record does.

    Structural genes come from the *primary* directory's definition, which is the
    byte-identical candidate set the confirmation re-used: loading the two arms
    from different definitions would silently re-derive the genes and make the
    pooling meaningless.
    """
    primary = Path("artifacts/arch-zoo-100m-v3-crossed")
    confirm = Path("artifacts/arch-zoo-100m-v3-crossed-confirm")
    ledgers = sorted(primary.glob("*_seed*_results.jsonl")) + sorted(
        confirm.glob("*_seed*_results.jsonl")
    )
    if len(ledgers) < 6:
        pytest.skip("crossed arm and its confirmation ledgers are not both present")
    return with_structure(load_results(ledgers), primary / "zoo_definition.json")


def test_recorded_confirmation_replicates_the_attribution():
    """The headline claim: the attribution is a property of the design, not the seeds.

    The phase's conclusion rests on the three fresh seeds reproducing what seeds
    42/43/44 measured. Pinned against the real ledgers with the tolerances the
    claim actually makes: the same signs, a magnitude within a few hundredths of a
    nat, and the lever still more than 1.5x the depth effect.
    """
    primary = Path("artifacts/arch-zoo-100m-v3-crossed")
    confirm = Path("artifacts/arch-zoo-100m-v3-crossed-confirm")
    if not (confirm / "zoo_definition.json").exists():
        pytest.skip("confirmation arm ledgers not present")
    original = crossed_contrast(
        with_structure(
            load_results(sorted(primary.glob("*_seed*_results.jsonl"))),
            primary / "zoo_definition.json",
        ),
        1200,
        samples=20_000,
    )
    fresh = crossed_contrast(
        with_structure(
            load_results(sorted(confirm.glob("*_seed*_results.jsonl"))),
            confirm / "zoo_definition.json",
        ),
        1200,
        samples=20_000,
    )
    assert original["available"] and fresh["available"]
    assert fresh["seeds"] == [45, 46, 47]
    for gene, sign in (("hidden_size", -1), ("depth", +1)):
        # Same sign, and the magnitude survives to within 0.02 nats.
        assert fresh["main"][gene]["nats"] * sign > 0, gene
        assert fresh["main"][gene]["nats"] == pytest.approx(
            original["main"][gene]["nats"], abs=0.02
        ), gene
    assert abs(fresh["main"]["hidden_size"]["nats"]) >= 1.5 * abs(
        fresh["main"]["depth"]["nats"]
    )


def test_recorded_fresh_seeds_alone_cannot_resolve_the_interaction():
    """Three seeds floor the calibrated test at 0.25, so the arm *must* be pooled.

    Pinned because it is the arm's whole design argument, and because the failure
    mode is someone reading the fresh seeds' consistent signs as a resolution: at
    three seeds the test cannot report below 0.25 whatever the data say.
    """
    confirm = Path("artifacts/arch-zoo-100m-v3-crossed-confirm")
    if not (confirm / "zoo_definition.json").exists():
        pytest.skip("confirmation arm ledgers not present")
    results = with_structure(
        load_results(sorted(confirm.glob("*_seed*_results.jsonl"))),
        confirm / "zoo_definition.json",
    )
    outcome = effect_summary(results, 1200, [45, 46, 47], samples=20_000)
    decision = replication_decision(outcome)
    assert decision["branch"] == "inconclusive"
    assert decision["interaction"]["floor"] == pytest.approx(2 / 2 ** 3)
    assert decision["conditions"][4]["met"] is True  # the signs do agree...
    assert decision["conditions"][3]["met"] is False  # ...and cannot be decisive


def test_recorded_pooled_six_seed_record_hits_the_declared_branch():
    """The pooled record lands on *replicates, interaction resolves*.

    This is the phase's second headline and it is a pre-registered branch, so it is
    pinned as the branch rather than as a p-value: the seeded-blocked test tracks a
    sign test whose floor is 2/2^seeds, and the two Monte-Carlo main effects are
    bounded by the draw count (the record's own run used 200,000 draws; this test
    uses fewer, which only makes the bound weaker and the branch harder to reach).
    """
    results = paired_crossed_results()
    outcome = effect_summary(
        results, 1200, [42, 43, 44, 45, 46, 47], samples=20_000
    )
    decision = replication_decision(outcome)
    assert decision["available"]
    assert decision["branch"] == "replicates-and-resolves"
    assert all(row["met"] for row in decision["conditions"]), decision["conditions"]

    crossed = outcome["crossed_contrast"]
    assert crossed["seeds"] == [42, 43, 44, 45, 46, 47]
    anchor = crossed["main"][crossed["factor_a"]]
    # The floor is the arm's arithmetic: at six seeds the sign test cannot report
    # below 2/2^6, and it is *at* that floor with every contrast agreeing.
    assert anchor["interaction_p_seed_blocked"] == pytest.approx(2 / 2 ** 6)
    assert anchor["interaction_p_seed_blocked_floor"] == pytest.approx(2 / 2 ** 6)
    contrasts = anchor["interaction_seed_contrasts"]
    assert len(contrasts) == 6
    assert len({value > 0 for value in contrasts}) == 1
    # The interaction is claimed to be *level-dependent*, so the two per-level
    # depth contrasts must differ in that direction rather than merely in sign.
    by_level = crossed["main"]["depth"]["by_other_level"]
    narrow = by_level[min(by_level, key=lambda level: float(level))]["nats"]
    wide = by_level[max(by_level, key=lambda level: float(level))]["nats"]
    assert narrow > wide > 0


def test_the_pooled_record_on_disk_carries_the_branch():
    """The committed decision must state the branch the code computes.

    A record that reported the interaction as open while the analysis returns the
    resolved branch would be the worst of both: the code would look like it decided
    and the document would look like it had not.
    """
    path = Path("artifacts/arch-zoo-100m-v3-crossed/decision.md")
    if not path.exists():
        pytest.skip("pooled crossed decision not written yet")
    text = path.read_text()
    assert "## Pre-registered rule: does the attribution replicate?" in text
    assert "**REPLICATES, INTERACTION RESOLVES**" in text
    assert "every seed contrast the same sign" in text
    # All six contrasts, and the floor they are read against: a record showing five
    # would mean the pooled run had silently dropped a seed, which is precisely the
    # failure this session's disclosure work exists to make visible.
    assert "0.03125" in text
    for contrast in ("+0.0819", "+0.0840", "+0.0443", "+0.0970", "+0.0691", "+0.1200"):
        assert contrast in text


def test_recorded_crossed_attribution_is_not_an_artifact_of_the_holdout_split():
    """The phase's attribution has to survive the same split check v2's does.

    Architectures could differ in how much they overfit rather than in how well
    they model, in which case a held-out-loss contrast would be measuring the
    split. Re-running the identical factorial on training loss keeps both main
    effects, at the same signs and on the same floor, so the attribution is about
    the optimisation and not about where the validation tokens were cut.
    """
    directory = Path("artifacts/arch-zoo-100m-v3-crossed")
    ledgers = sorted(directory.glob("*_seed*_results.jsonl"))
    if not ledgers:
        pytest.skip("arch-zoo-100m-v3-crossed ledgers not present")
    results = with_structure(load_results(ledgers), directory / "zoo_definition.json")

    held_out = crossed_contrast(results, 1200, samples=20_000)["main"]
    as_train = crossed_contrast(
        [replace(entry, validation_loss=entry.final_train_loss) for entry in results],
        1200,
        samples=20_000,
    )["main"]

    for gene in ("depth", "hidden_size"):
        assert as_train[gene]["p"] < 0.01, gene
        # Same sign, comparable size: the effect is not a split artifact.
        assert (as_train[gene]["nats"] > 0) == (held_out[gene]["nats"] > 0), gene
        assert as_train[gene]["nats"] == pytest.approx(
            held_out[gene]["nats"], rel=0.35
        ), gene


# --- pre-specified joint models --------------------------------------------


def test_joint_fit_honours_a_pre_specified_model():
    """A named model must be fitted as named, not re-selected.

    Selection picks a different model in each subset, so a stable R2 across
    subsets can be a different hypothesis each time. Asking for a specific model
    is the only way to ask whether *that* model holds, which is the question a
    designed factorial licenses - so the order and membership have to survive.
    """
    rng = random.Random(5)
    results = []
    for index, (depth, hidden) in enumerate(UNCORRELATED_LAYOUTS):
        loss = 3.2 + 0.05 * (depth - 10) - 0.0012 * (hidden - 700) + rng.gauss(0, 0.004)
        for seed in (42, 43):
            results.append(
                replace(
                    result(f"arch{index}", loss, seed=seed),
                    depth=depth,
                    hidden_size=hidden,
                )
            )

    fit = joint_structural_fit(results, STEPS, genes=("depth", "hidden_size"))
    assert fit["available"]
    # Order is the caller's, not the |rho| ordering the selector would apply.
    assert [row["gene"] for row in fit["genes"]] == ["depth", "hidden_size"]
    assert fit["dropped"] == []

    # A model naming an unrecorded gene must refuse, not silently fit a subset.
    missing = joint_structural_fit(results, STEPS, genes=("depth", "expert_width"))
    assert missing["available"] is False
    assert "no variance" in missing["reason"]

    unknown = joint_structural_fit(results, STEPS, genes=("depth", "nonsense"))
    assert unknown["available"] is False
    assert "not structural genes" in unknown["reason"]


def test_joint_fit_refuses_a_rank_deficient_pre_specified_model():
    """`heads = hidden/64` makes those two columns identical. The selector drops
    one and carries on, which is right when the caller asked for "the best fit".
    When the caller named a model, dropping a predictor answers a different
    question, so the honest reply is a refusal."""
    layouts = ((512, 4), (576, 16), (640, 8), (704, 12), (768, 6), (832, 10), (896, 14), (960, 8))
    results = []
    for index, (hidden, experts) in enumerate(layouts):
        loss = 3.6 - 0.0012 * (hidden - 700) + 0.004 * (experts - 10)
        for seed in (42, 43):
            results.append(
                replace(
                    result(f"arch{index}", loss, seed=seed),
                    hidden_size=hidden,
                    attention_heads=hidden // 64,
                    expert_count=experts,
                )
            )

    selected = joint_structural_fit(results, STEPS)
    assert selected["available"] is True
    assert {row["gene"] for row in selected["dropped"]} == {"attention_heads"}

    named = joint_structural_fit(results, STEPS, genes=("hidden_size", "attention_heads"))
    assert named["available"] is False
    assert "rank-deficient" in named["reason"]
    assert {row["gene"] for row in named["dropped"]} == {"attention_heads"}


def test_signal_by_length_reports_the_pre_specified_model(tmp_path):
    design = []
    for index, (depth, hidden) in enumerate(UNCORRELATED_LAYOUTS):
        base = 3.2 + 0.05 * (depth - 10) - 0.0012 * (hidden - 700)
        for steps in (400, 1200):
            for seed in (42, 43):
                design.append(
                    replace(
                        result(f"arch{index}", base, seed=seed, steps=steps),
                        depth=depth,
                        hidden_size=hidden,
                    )
                )

    signal = structural_signal_by_length(design, genes=("depth", "hidden_size"))
    assert signal["model"]["pre_specified"] == ["depth", "hidden_size"]
    assert signal["400"]["available"] and signal["1200"]["available"]
    for length in ("400", "1200"):
        assert [row["gene"] for row in signal[length]["genes"]] == [
            "depth",
            "hidden_size",
        ]


def test_recorded_prespecified_screening_model_holds_at_every_length():
    """Pinned against the real ledgers, and the strongest form of §3's screening
    claim: the model the crossed factorial identified is fitted unchanged at each
    length, so no per-length selection is involved. `hidden_size` is individually
    significant even at 400 steps, which is what licenses screening there."""
    v2_dir = Path("artifacts/arch-zoo-100m-v2")
    v2_ledgers = sorted(v2_dir.glob("*_seed*_results.jsonl"))
    if not v2_ledgers:
        pytest.skip("arch-zoo-100m-v2 ledgers not present")
    pooled = _horizon_results() + with_structure(
        load_results(v2_ledgers), v2_dir / "zoo_definition.json"
    )

    signal = structural_signal_by_length(pooled, genes=("hidden_size", "depth"))
    lengths = [key for key in signal if key not in ("agreement", "model")]
    assert len(lengths) >= 3, lengths

    for length in lengths:
        row = signal[length]
        assert row["available"], length
        assert row["p_model"] < 0.01, (length, row["p_model"])
        assert row["r_squared"] > 0.9, (length, row["r_squared"])
        genes = {entry["gene"]: entry for entry in row["genes"]}
        # The same genes, in the order requested, at every length.
        assert list(genes) == ["hidden_size", "depth"], length
        # And the identified lever is significant at the cheapest length too.
        assert genes["hidden_size"]["beta"] < 0, length
        assert genes["hidden_size"]["p"] < 0.05, (length, genes["hidden_size"]["p"])


# --- section 11: curated set versus random set ------------------------------


def set_members(label: str, size: int) -> tuple[str, ...]:
    return tuple(f"{label}_{index}" for index in range(size))


def set_rows(
    qualities: dict[str, list[float]],
    generator: random.Random,
    *,
    noise: float,
    seeds: tuple[int, ...] = (42, 43),
) -> list[EvalResult]:
    """Ledger rows for two named sets, where each member carries one quality.

    The quality is the member's own - it is the same on every seed, which is what
    the real arm's rows do and what an assignment in this design has to respect.
    """
    return [
        result(f"{label}_{index}", quality + generator.gauss(0.0, noise), seed=seed)
        for label, values in qualities.items()
        for index, quality in enumerate(values)
        for seed in seeds
    ]


def test_set_test_counts_member_splits_not_seed_splits():
    """The null's unit is the architecture, so seeds do not multiply the labellings.

    A set label is a property of a member and holds on every seed that member ran,
    so one split is drawn and read through all the seeds. Counting a split per seed
    instead would give C(12, 6) ** 3 = 7.9e8 labellings, and every one of them is a
    design the sweep cannot produce - a candidate cannot be in the curated set on
    one seed and outside it on another. It is also not harmless: an assignment that
    varies by seed splits each member's rows across the two sides, which removes the
    member-level spread from the null's differences while the observed difference
    still carries it, so the null comes out ~1/sqrt(2) as wide as the value it is
    testing and the test rejects on a real spread alone.
    """
    generator = random.Random(4)
    qualities = {
        label: [generator.uniform(3.0, 4.0) for _ in range(6)] for label in ("random", "manual")
    }
    rows = set_rows(qualities, generator, noise=0.01, seeds=(42, 43, 44))
    sets = (("random", set_members("random", 6)), ("manual", set_members("manual", 6)))

    outcome = two_set_blocked_test(rows, STEPS, sets)

    assert outcome["available"], outcome
    assert outcome["method"] == "exact"
    assert outcome["assignments"] == math.comb(12, 6) == 924
    assert outcome["min_achievable_p"] == pytest.approx(2 / 924)
    assert outcome["seeds"] == [42, 43, 44]
    # What the per-seed count would have been, so the two cannot be confused again.
    assert math.comb(12, 6) ** 3 == 788_889_024


def test_set_test_blocks_out_a_seed_wide_offset():
    """A seed that trained everything worse is not evidence about the two sets."""
    generator = random.Random(5)
    qualities = {
        label: [3.4 + 0.05 * index for index in range(6)] for label in ("random", "manual")
    }
    seeds = (42, 43, 44)
    rows = [
        result(
            f"{label}_{index}",
            quality + (5.0 if seed == 43 else 0.0) + generator.gauss(0.0, 0.01),
            seed=seed,
        )
        for seed in seeds
        for label, values in qualities.items()
        for index, quality in enumerate(values)
    ]
    sets = (("random", set_members("random", 6)), ("manual", set_members("manual", 6)))

    outcome = two_set_blocked_test(rows, STEPS, sets)

    assert outcome["difference_nats"] == pytest.approx(0.0, abs=0.02)
    assert outcome["p"] > 0.2
    # Each seed's own difference is free of the offset, which is the mechanism.
    assert all(abs(row["difference"]) < 0.02 for row in outcome["per_block"])


def test_set_test_answers_each_statistic_with_one_sign_convention():
    """``mean``, ``best`` and ``worst`` ask different questions; the sign is shared.

    A negative difference favours the first set on all three, so a reader does not
    have to remember which statistic inverted the comparison.
    """
    generator = random.Random(6)
    size = 6
    sets = (("random", set_members("random", size)), ("manual", set_members("manual", size)))
    rows = set_rows(
        {
            "random": [3.6 + 0.02 * index for index in range(size)],
            # Every manual member is 0.2 nats better, so all three statistics must
            # say the second set is ahead.
            "manual": [3.4 + 0.02 * index for index in range(size)],
        },
        generator,
        noise=0.01,
    )

    for statistic in ("mean", "best", "worst"):
        outcome = two_set_blocked_test(rows, STEPS, sets, statistic=statistic)
        assert outcome["available"], (statistic, outcome)
        assert outcome["statistic"] == statistic
        assert outcome["difference_nats"] == pytest.approx(0.2, abs=0.05), statistic
        assert outcome["per_set"]["random"]["mean"] > outcome["per_set"]["manual"]["mean"]
        assert outcome["p"] < 0.05, statistic
        assert outcome["sign_consistency"] is True


def test_set_test_refuses_a_comparison_it_cannot_make():
    """Unequal sets, shared members and one seed are refused with the reason."""
    generator = random.Random(7)
    size = 4
    sets = (("random", set_members("random", size)), ("manual", set_members("manual", size)))
    rows = set_rows(
        {label: [3.5 + 0.01 * index for index in range(size)] for label in ("random", "manual")},
        generator,
        noise=0.01,
    )

    unbalanced = two_set_blocked_test(
        rows,
        STEPS,
        (("random", set_members("random", size)), ("manual", set_members("manual", size + 1))),
    )
    assert not unbalanced["available"]
    assert "unequal size" in unbalanced["reason"]

    overlapping = two_set_blocked_test(
        rows,
        STEPS,
        (("random", set_members("random", size)), ("manual", set_members("random", size))),
    )
    assert not overlapping["available"]
    assert "share architectures" in overlapping["reason"]

    one_seed = two_set_blocked_test(
        set_rows(
            {label: [3.5 + 0.01 * index for index in range(size)] for label in ("random", "manual")},
            generator,
            noise=0.01,
            seeds=(42,),
        ),
        STEPS,
        sets,
    )
    assert not one_seed["available"]
    assert "2+ seeds" in one_seed["reason"]

    missing = two_set_blocked_test(rows, STEPS + 100, sets)
    assert not missing["available"]
    assert "no 500-step rows" in missing["reason"]


def test_set_test_is_calibrated_when_members_carry_spread_across_seeds():
    """Regression, at the size the section 11 comparison actually runs.

    The member-level spread here is ~19x the per-run noise, which is the regime the
    real arm is in (a candidate's quality spread dwarfs its seed noise). Under a
    null that is properly built the rejection rate stays in the nominal region no
    matter how large that spread is - the split is the same quantity on both sides.
    Under a per-seed split it does not: an assignment that breaks a member's rows
    apart cancels the spread in the null while the observed difference still carries
    it, and on this data the same 300 replicates rejected a true null 22.3% of the
    time (17.5% at 13 candidates per set, where this comparison actually runs).
    Left in, that would have entered the record as a more sensitive comparison than
    the phase can buy.
    """
    generator = random.Random(11)
    size = 6
    sets = (("random", set_members("random", size)), ("manual", set_members("manual", size)))
    replicates = 300
    hits = 0
    for _ in range(replicates):
        qualities = {
            label: [generator.uniform(3.0, 4.0) for _ in range(size)]
            for label in ("random", "manual")
        }
        rows = set_rows(qualities, generator, noise=0.01)
        # Forced onto the sampled path, which is the one an affordable comparison
        # uses at 13 candidates per set (C(26, 13) = 10.4M labellings).
        outcome = two_set_blocked_test(
            rows, STEPS, sets, samples=400, max_assignments=100
        )
        assert outcome["method"] == "monte-carlo", outcome
        assert outcome["assignments"] == math.comb(2 * size, size)
        hits += outcome["p"] < 0.05

    rate = hits / replicates
    # 300 replicates put the sampling error on a 5% rate at ~1.3 points, so this
    # separates a calibrated test from the 17% one with room to spare.
    assert rate < 0.12, rate
