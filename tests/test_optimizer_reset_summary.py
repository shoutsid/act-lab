"""The committed section 5.23 summaries stay machine-checked.

Section 5.23 decomposes the mid-training penalty: two runs that differ by
optimizer resets alone (``F`` vs ``U``) and two that differ by whether the
architecture changed (``U`` vs ``R``). The registered verdict is branch (c) -
resets cost nothing, the rebuild is the whole penalty - and the reading has a
quantitative half that matters as much as the pass/fail half: the rebuild's cost
alone has to account for the *whole* penalty section 5.22 measured on independent
blocks. Both halves are pinned here, so a later block that quietly changes either
fails the suite rather than the reader.
"""

from __future__ import annotations

import json

import pytest

from act.evolution.figures_data import summaries_dir
from act.evolution.study import _binomial_survival, _gate_min_wins

PER_PRIMARY_ALPHA = 0.0125
DECLARED_BAR = 215
#: file -> (study id, seed block, {comparison: (wins, mean, p, passes)})
QUOTED = {
    "optimizer-reset-depth2-384.json": (
        "study_e0f1f2629785dc17",
        (94000, 94384),
        {
            "F_vs_U": (157, 0.004907, 1.00, False),
            "U_vs_R": (299, -0.106882, 2.48e-29, True),
            "U_vs_A": (235, -0.040173, 6.66e-06, True),
        },
    ),
    "optimizer-reset-depth3-384.json": (
        "study_09573ffbfb828091",
        (95000, 95384),
        {
            "F_vs_U": (105, 0.016403, 1.00, False),
            "U_vs_R": (311, -0.127563, 2.22e-36, True),
            "U_vs_A": (223, -0.035418, 9.08e-04, True),
        },
    ),
}
#: Section 5.22's total penalty against the fused random arm, per depth, for the
#: cross-study accounting: the rebuild's cost alone should account for all of it.
SECTION_5_22_PENALTY = {"optimizer-reset-depth2-384.json": -0.099580,
                        "optimizer-reset-depth3-384.json": -0.121876}


def _load(name: str) -> dict:
    path = summaries_dir() / name
    if not path.exists():
        pytest.fail(f"missing committed study summary: {path}")
    return json.loads(path.read_text())


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_section_5_23_numbers_recompute_from_per_seed(name: str) -> None:
    study_id, block, quoted = QUOTED[name]
    summary = _load(name)
    assert summary["record_type"] == "optimizer_reset_study"
    assert summary["protocol"] == "optimizer-history-vs-architecture-decomposition"
    assert summary["study_id"] == study_id
    assert summary["per_primary_alpha"] == PER_PRIMARY_ALPHA
    assert summary["declared_bar"] == DECLARED_BAR
    assert summary["reset_steps"] == [12, 24, 36]
    start, stop = block
    assert summary["seeds"] == list(range(start, stop))
    assert summary["seeds"] == [row["seed"] for row in summary["per_seed"]]

    for comparison_name, (wins, mean, p, passes) in quoted.items():
        first, second = comparison_name.split("_vs_")
        deltas = [
            row["losses"][first] - row["losses"][second] for row in summary["per_seed"]
        ]
        assert sum(delta < 0 for delta in deltas) == wins, comparison_name
        assert sum(deltas) / len(deltas) == pytest.approx(mean, abs=5e-6)
        assert _binomial_survival(wins, len(deltas)) == pytest.approx(p, abs=1e-3)
        assert summary["comparisons"][comparison_name]["passes"] is passes
        assert summary["comparisons"][comparison_name]["min_wins"] == _gate_min_wins(
            len(deltas), PER_PRIMARY_ALPHA
        )


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_section_5_23_control_is_clean(name: str) -> None:
    """F and U must differ by optimizer history alone, or P1 measures nothing."""
    checks = _load(name)["built_in_checks"]
    assert all(checks.values()), checks
    for key in (
        "reset_twin_genome_is_the_fixed_genome",
        "reset_twin_never_mutated",
        "reset_twin_books_the_same_training_flops",
    ):
        assert checks[key] is True, key


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_section_5_23_branch_is_c_and_the_accounting_closes(name: str) -> None:
    """Branch (c) plus the quantitative claim that make it actionable.

    Branch (c) says the rebuild is the whole penalty. That is only meaningful if
    the rebuild's measured cost is actually the size of the penalty section 5.22
    found - so the two are compared here rather than left to a reader.
    """
    summary = _load(name)
    comparisons = summary["comparisons"]
    assert comparisons["F_vs_U"]["passes"] is False, "P1 must fail for branch (c)"
    assert comparisons["U_vs_R"]["passes"] is True, "P2 must pass for branch (c)"

    rebuild_cost = comparisons["U_vs_R"]["mean_delta"]
    total_penalty = SECTION_5_22_PENALTY[name]
    # Same sign, and the rebuild alone accounts for essentially all of it.
    assert rebuild_cost < 0 and total_penalty < 0
    assert 0.8 <= rebuild_cost / total_penalty <= 1.2, (rebuild_cost, total_penalty)
    # And the exonerating direction of P1: the reset twin was not harmed.
    assert summary["reset_cost"] <= 0
