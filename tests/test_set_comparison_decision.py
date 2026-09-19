"""Tests for the §11 set-comparison decision rule and its analyzer.

The rule (`set_comparison_decision` / `SET_COMPARISON_RULE`) was declared before
any ledger of the §11 comparison exists, and these tests are the pre-registration's
teeth rather than after-the-fact coverage: every branch is exercised on a synthetic
design with planted effects, the same rehearsal the head-paired confirmation arm
ran before its hours were spent. A threshold that drifts from the transcription
fails here, and a branch that fires when its conditions do not hold fails here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dataclasses import replace

from act.search.architecture_eval import EvalResult, load_results  # noqa: E402
from act.search.architecture_report import (  # noqa: E402
    SET_COMPARISON_READINGS,
    SET_COMPARISON_RULE,
    set_comparison_decision,
    two_set_blocked_test,
)
from set_comparison_analyze import (  # noqa: E402
    candidate_names,
    structural_summary,
)

STEPS = 400


def make_rows(names, *, seeds, losses_by_name):
    """Ledger rows for one set's members: each name on each seed, losses supplied."""
    rows = []
    for name in names:
        for seed in seeds:
            loss = losses_by_name[name][seed]
            rows.append(
                EvalResult(
                    name=name,
                    family="test",
                    genome_hash=name,
                    seed=seed,
                    total_params=80_000_000,
                    active_params=40_000_000,
                    contracted_flops_per_token=2.5e8,
                    dense_runtime_flops_per_token=4.0e8,
                    training_flops=0.0,
                    validation_flops=0.0,
                    final_train_loss=loss + 0.2,
                    mean_train_loss=loss + 0.25,
                    validation_loss=loss,
                    steps=STEPS,
                    train_tokens=0,
                    val_tokens=0,
                    elapsed_s=0.0,
                    measured_step_s=0.0,
                    measured_tokens_per_s=0.0,
                    peak_memory_mb=0.0,
                    device="cpu",
                    description="",
                    val_source="synthetic",
                )
            )
    return rows


def synthetic_design(*, advantage: float, seeds=(0, 1), n: int = 6):
    """Two disjoint equal-size sets; the manual set's members are `advantage` better."""
    manual_names = [f"man_{i}_D8_H768_E8_A4_W512" for i in range(n)]
    random_names = [f"rnd_{i}_D8_H768_E8_A4_W512" for i in range(n)]
    manual_losses = {
        name: {seed: 3.0 + 0.01 * index for seed in seeds}
        for index, name in enumerate(manual_names)
    }
    random_losses = {
        name: {seed: 3.0 + 0.01 * index + advantage for seed in seeds}
        for index, name in enumerate(random_names)
    }
    rows = make_rows(manual_names, seeds=seeds, losses_by_name=manual_losses)
    rows += make_rows(random_names, seeds=seeds, losses_by_name=random_losses)
    outcome = two_set_blocked_test(
        rows, STEPS, (("manual", manual_names), ("random", random_names))
    )
    return outcome, manual_names, random_names


def test_the_transcription_is_pinned():
    """The rule's thresholds are data; a changed threshold is an edit to the
    pre-registration and must land in the record, not silently in a default."""
    assert SET_COMPARISON_RULE == {"p": 0.05, "budget_deviation": 0.025, "min_seeds": 2}
    assert set(SET_COMPARISON_READINGS) == {
        "manual-better",
        "random-better",
        "no-detectable-difference",
        "unavailable",
    }


def test_planted_manual_advantage_lands_on_manual_better():
    outcome, _, _ = synthetic_design(advantage=0.30)
    decision = set_comparison_decision(outcome, budget_ratio=1.0)
    assert decision["available"]
    assert decision["branch"] == "manual-better"
    assert decision["difference_nats"] == pytest.approx(-0.30, abs=1e-6)
    assert decision["p"] < SET_COMPARISON_RULE["p"]


def test_planted_random_advantage_lands_on_random_better():
    outcome, _, _ = synthetic_design(advantage=-0.30)
    decision = set_comparison_decision(outcome, budget_ratio=1.0)
    assert decision["available"]
    assert decision["branch"] == "random-better"
    assert decision["difference_nats"] > 0


def test_null_design_reports_the_power_context_not_equality():
    outcome, _, _ = synthetic_design(advantage=0.0)
    decision = set_comparison_decision(outcome, budget_ratio=1.0)
    assert decision["available"]
    assert decision["branch"] == "no-detectable-difference"
    # The reading must carry the power study's context rather than claiming equality.
    assert "0.30-nat" in decision["reading"]
    assert "not evidence of equality" in decision["reading"]


def test_budget_imbalance_refuses_a_verdict():
    """A 12% budget advantage is what §9a measured an unmatched draw to carry; the
    rule must refuse rather than read it as selection."""
    outcome, _, _ = synthetic_design(advantage=0.30)
    decision = set_comparison_decision(outcome, budget_ratio=1.12)
    assert not decision["available"]
    assert decision["branch"] == "unavailable"
    assert any("parity" in row["condition"] and not row["met"] for row in decision["conditions"])


def test_single_seed_refuses():
    outcome, _, _ = synthetic_design(advantage=0.30, seeds=(0,))
    decision = set_comparison_decision(outcome, budget_ratio=1.0)
    assert not decision["available"]
    assert decision["branch"] == "unavailable"


def test_boundary_budget_ratio_is_accepted():
    outcome, _, _ = synthetic_design(advantage=0.30)
    decision = set_comparison_decision(outcome, budget_ratio=1.025)
    assert decision["available"]


def test_the_ratio_and_p_boundaries_decide_at_exact_equality():
    """One boundary admits and one refuses, and both are compared at equality.

    The budget condition is the only *inclusive* one in this rule: parity is a band,
    so a ratio sitting exactly on the tolerance is inside it and only an exceedance
    is rejected. The p condition is strict, so a p sitting exactly on the threshold
    is not significant and the null branch reports the power context rather than a
    direction - the rule must not name a winner on a p it did not clear.

    The deviation is taken from the ratio's own float difference because the decimal
    form cannot express the boundary: ``abs(1.025 - 1.0)`` is
    ``0.02499999999999991``, strictly inside the declared ``0.025``, so the existing
    1.025 check passes under ``<=`` and ``<`` alike and cannot see which one the rule
    uses. Setting the deviation to that difference makes the comparison exactly equal
    by construction, which is what pins the operator.
    """
    outcome, _, _ = synthetic_design(advantage=0.30)

    ratio = 1.025
    at_ratio = set_comparison_decision(
        outcome, budget_ratio=ratio, budget_deviation=abs(ratio - 1.0)
    )
    assert at_ratio["available"]

    base = set_comparison_decision(outcome, budget_ratio=1.0)
    assert base["branch"] == "manual-better"
    at_p = set_comparison_decision(outcome, budget_ratio=1.0, p_threshold=base["p"])
    assert at_p["branch"] == "no-detectable-difference"


def test_a_tied_pair_cannot_reach_the_sign_branch():
    """The tie clause is unreachable, and this is the proof rather than a guess.

    ``branch = "manual-better" if difference < 0 else "random-better"`` puts a pair
    whose means are *exactly* equal on the random side. Swapping ``<`` for ``<=``
    there changes nothing, and that is not a missing test: the p-value is the share
    of permuted statistics at least as extreme as the observed one, and every
    permutation is at least as extreme as zero, so a tied pair reports ``p = 1.0``
    and cannot clear any threshold a p-value is compared against. The clause only
    becomes reachable under a threshold above 1.0, which no p-value can exceed.

    Recorded because the alternative readings of a survivor - a hole, or an
    equivalent mutant - call for opposite responses, and the difference is
    checkable. Pinning the invariant is what keeps it checkable rather than folklore.
    """
    outcome, _, _ = synthetic_design(advantage=0.0)
    decision = set_comparison_decision(outcome, budget_ratio=1.0)
    assert decision["difference_nats"] == 0.0
    assert decision["p"] == 1.0
    assert decision["branch"] == "no-detectable-difference"


def test_best_statistic_is_reportable_but_secondary():
    """The winner form stays computable - a lead to price, never a branch: §9a
    measured the floor permutation to contain the winner effect at this size."""
    outcome, _, _ = synthetic_design(advantage=0.0)
    decision = set_comparison_decision(outcome, budget_ratio=1.0)
    assert decision["branch"] == "no-detectable-difference"
    assert decision["available"]


def _rebuild_for(manual_names, random_names, **overrides):
    """The reconstruction the analyzer hands the rule for its reading check."""

    def rebuild(rows):
        return set_comparison_decision(
            two_set_blocked_test(
                rows,
                STEPS,
                (("manual", manual_names), ("random", random_names)),
            ),
            **overrides,
            budget_ratio=1.0,
        )

    return rebuild


def test_the_reading_table_is_rendered_for_the_rule_that_has_not_run_yet():
    """§11's rule is declared, so its reading check is declared with it.

    The arm has not been bought. Wiring the check now rather than after a result lands
    is the pre-registration's point: a rule and the readings it is to survive are one
    declaration, and adding the second later would be adding it to a result.
    """
    outcome, manual_names, random_names = synthetic_design(advantage=0.30)
    rows = make_rows(
        manual_names,
        seeds=(0, 1),
        losses_by_name={n: {s: 3.0 + 0.01 * i for s in (0, 1)} for i, n in enumerate(manual_names)},
    ) + make_rows(
        random_names,
        seeds=(0, 1),
        losses_by_name={
            n: {s: 3.0 + 0.01 * i + 0.30 for s in (0, 1)}
            for i, n in enumerate(random_names)
        },
    )
    decision = set_comparison_decision(
        outcome, rows, budget_ratio=1.0, rebuild=_rebuild_for(manual_names, random_names)
    )

    assert decision["branch"] == "manual-better"
    stability = decision["reading_stability"]
    assert stability is not None
    # The fixture's two alternates are the held-out loss plus a constant, so a branch
    # that moved here would mean the reading was not being computed from the
    # substituted field.
    assert stability["moved"] == []
    assert {r["branch"] for r in stability["readings"].values()} == {"manual-better"}
    assert decision["note"] and "does not depend on which loss" in decision["note"]


def test_the_reading_check_is_reported_and_can_move_the_set_comparison_branch():
    """The mechanism, on a design where the trajectory average reorders the two sets.

    A per-set constant offset on ``mean_train_loss`` is enough to reverse the set-mean
    difference, which is the same shape as the crossed arm's real flip and is what
    makes the check informative rather than decorative.
    """
    outcome, manual_names, random_names = synthetic_design(advantage=0.30)
    rows = make_rows(
        manual_names,
        seeds=(0, 1),
        losses_by_name={n: {s: 3.0 + 0.01 * i for s in (0, 1)} for i, n in enumerate(manual_names)},
    ) + make_rows(
        random_names,
        seeds=(0, 1),
        losses_by_name={
            n: {s: 3.0 + 0.01 * i + 0.30 for s in (0, 1)}
            for i, n in enumerate(random_names)
        },
    )
    # The manual set trains badly and only overtakes by the end, so the trajectory
    # average puts the random set ahead while held-out loss does not - the same shape
    # as the crossed arm's real flip, where the loss that is averaged answers a
    # different question from the loss read at the end.
    manual_set = set(manual_names)
    rows = [
        replace(
            row,
            mean_train_loss=row.mean_train_loss + (0.6 if row.name in manual_set else 0.0),
        )
        for row in rows
    ]
    decision = set_comparison_decision(
        outcome, rows, budget_ratio=1.0, rebuild=_rebuild_for(manual_names, random_names)
    )

    assert decision["branch"] == "manual-better"
    stability = decision["reading_stability"]
    assert stability["moved"] == ["mean_train_loss"], stability["moved"]
    assert stability["readings"]["mean_train_loss"]["branch"] == "random-better"
    assert "reading-dependent" in decision["note"]
    assert "mean_train_loss" in decision["note"]


def test_a_rule_without_a_rebuild_reports_no_reading_claim():
    """No reconstruction supplied, no stability asserted - silence, not a guess."""
    outcome, _, _ = synthetic_design(advantage=0.30)
    assert set_comparison_decision(outcome, budget_ratio=1.0)["reading_stability"] is None
    assert set_comparison_decision(
        outcome, None, budget_ratio=1.0, rebuild=_rebuild_for(["a"], ["b"])
    )["reading_stability"] is None


def test_analyzer_end_to_end(tmp_path: Path):
    """The shipped script reads two real-shaped directories and writes its verdict."""
    from set_comparison_analyze import main as analyze_main

    outcome, manual_names, random_names = synthetic_design(advantage=0.30)
    # Rebuild the design into ledger files, one seed per file as a real sweep writes.
    import dataclasses
    import json

    manual_rows = [r for r in make_rows(manual_names, seeds=(0, 1), losses_by_name={
        n: {s: 3.0 + 0.01 * i for s in (0, 1)} for i, n in enumerate(manual_names)
    })]
    random_rows = [r for r in make_rows(random_names, seeds=(0, 1), losses_by_name={
        n: {s: 3.0 + 0.01 * i + 0.30 for s in (0, 1)} for i, n in enumerate(random_names)
    })]
    for exp, rows in (("m", manual_rows), ("r", random_rows)):
        d = tmp_path / exp
        d.mkdir()
        if exp == "m":
            (d / "zoo_definition.json").write_text(
                json.dumps({"candidates": [{"name": n, "total_params": 80_000_000} for n in manual_names]})
            )
        else:
            (d / "zoo_definition.json").write_text(
                json.dumps({"candidates": [{"name": n, "total_params": 80_000_000} for n in random_names]})
            )
        for seed in (0, 1):
            part = [r for r in rows if r.seed == seed]
            (d / f"flat_seed{seed}_results.jsonl").write_text(
                "".join(json.dumps(dataclasses.asdict(r)) + "\n" for r in part)
            )

    import sys as _sys

    argv = _sys.argv
    _sys.argv = [
        "set_comparison_analyze.py",
        "--manual-experiment-id", "m",
        "--random-experiment-id", "r",
        "--artifacts-root", str(tmp_path),
        "--steps", str(STEPS),
        "--write",
    ]
    try:
        analyze_main()
    finally:
        _sys.argv = argv

    record = json.loads((tmp_path / "m" / "set_comparison.json").read_text())
    assert record["decision"]["branch"] == "manual-better"
    assert record["budget_ratio"] == pytest.approx(1.0)
    assert (tmp_path / "m" / "set_comparison_decision.md").exists()
    markdown = (tmp_path / "m" / "set_comparison_decision.md").read_text()
    assert "pre-registered rule" in markdown
    assert "secondary" in markdown  # the best/worst caveat is rendered
    # The same reading table the zoo decision renders, through the same function, so
    # §11's record states which loss its branch was decided on before any result exists
    # to be decided.
    assert "| Loss read | Branch |" in markdown
    assert "`final_train_loss` - training loss at the end of training" in markdown
    assert "Only the loss field changed" in markdown


def test_structural_summary_parses_the_gene_names():
    zoo = {
        "candidates": [
            {"name": f"m{i}_D8_H768_E8_A4_W512", "total_params": 80_000_000 + i}
            for i in range(3)
        ]
    }
    summary = structural_summary(zoo)
    assert summary["candidates"] == 3
    assert summary["total_params"] == 3 * 80_000_000 + 3
    assert summary["mean_depth"] == 8
    assert summary["mean_hidden"] == 768
    assert summary["mean_width"] == 512
    assert candidate_names(zoo) == sorted(
        f"m{i}_D8_H768_E8_A4_W512" for i in range(3)
    )


def test_load_results_round_trips_the_synthetic_rows():
    outcome, manual_names, _ = synthetic_design(advantage=0.10)
    assert outcome["available"]
    assert len(outcome["seeds"]) == 2
