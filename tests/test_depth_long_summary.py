"""Section 5.54's recorded result: the fourth branch, and every number behind it.

The arm registered four branches and landed on "neither": 4x training moves
nothing below the onset, so the compute covariate is closed too. These pins are
read from the committed summary, so a regenerated record that changes a win
count, a mean or the branch label fails here rather than silently rewriting the
report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import null_toy_study as study  # noqa: E402

SUMMARIES = ROOT / "artifacts" / "study-summaries"
PROFILE = study.LONG_D3_PROFILE
CELL = "depth3-long-576.json"
SHORT = "null-control-depth3-384.json"
ABOVE_SHORT = "null-control-depth4-prefix-384.json"
ABOVE_LONG = "width-d4-long-w24-576.json"


def _record(name: str) -> dict:
    return json.loads((SUMMARIES / name).read_text())


def _mean(name: str, left: str, right: str) -> float:
    rows = _record(name)["per_seed"]
    return sum(row["losses"][left] - row["losses"][right] for row in rows) / len(rows)


def test_the_summary_is_the_frozen_revision_and_the_registered_arm() -> None:
    summary = _record(CELL)
    assert summary["profile"] == PROFILE
    assert summary["report_section"] == "5.54"
    assert summary["code_revision"] == study.PROFILES[PROFILE]["run_revision"] == "64046da"
    assert len(summary["seeds"]) == 576
    assert summary["seeds"][0] == 90000
    assert summary["seeds"][-1] == 90575
    # Two primaries, so the level is alpha/2 - the arm's own declaration, not a
    # CLI default (the defect class 5.50's near-miss came from).
    assert summary["bonferroni_alpha"] == pytest.approx(study.ALPHA / 2)
    assert summary["comparisons"]["adaptive_vs_null"]["min_wins_bonferroni"] == 313


def test_the_recorded_cell_statistics() -> None:
    """P1's four numbers, as the report quotes them."""
    cell = _record(CELL)["comparisons"]["adaptive_vs_null"]
    assert cell["wins"] == 281
    assert cell["n"] == 576
    assert cell["mean_delta"] == pytest.approx(0.03900495181894965)
    assert cell["binomial_p"] == pytest.approx(0.7339963132961969)
    assert cell["passes_bonferroni"] is False
    # The conjunction fails on both halves: short of the bar AND non-negative.
    assert cell["wins"] < cell["min_wins_bonferroni"]
    assert cell["mean_delta"] > 0


def test_the_arm_means_make_doing_nothing_the_best_arm() -> None:
    arms = _record(CELL)["arm_means"]
    assert arms["adaptive"] == pytest.approx(6.3452956675448355)
    assert arms["null"] == pytest.approx(6.306290715725885)
    assert arms["random"] == pytest.approx(6.434678504760894)
    assert arms["null"] < arms["adaptive"] < arms["random"]
    # 5.53's damage cells read the same way, which is what the outcome claims.
    assert arms["null"] == min(arms.values())


def test_the_outcome_names_the_fourth_branch() -> None:
    summary = _record(CELL)
    assert summary["outcome"] == "cell does not create value (5.54's per-cell primary: the conjunction fails)"
    assert "branch " not in summary["outcome"]


def test_the_regime_primary_did_not_move() -> None:
    """P2's recorded failure, with the rerun of the statistic it reports."""
    short = [
        row["losses"]["null"] - row["losses"]["random"] for row in _record(SHORT)["per_seed"]
    ]
    long_ = [
        row["losses"]["null"] - row["losses"]["random"] for row in _record(CELL)["per_seed"]
    ]
    u, p = study._rank_sum_one_sided(short, long_)
    assert u == pytest.approx(112872.0)
    assert p == pytest.approx(0.7060403994225366)
    assert p > study.ALPHA / 2, "P2 was registered to pass at alpha/2; it did not"
    # The reversed direction is a dead null, so the reading is an absence and
    # not a ruler pointing the wrong way.
    _, reverse = study._rank_sum_one_sided(long_, short)
    assert reverse == pytest.approx(0.2940, abs=1e-3)


def test_the_same_ruler_sees_the_depth_four_flip() -> None:
    """The non-detection above is informative because this one is not a non-detection.

    Both cells are the arm's own treatment (steps x4, interval x4) and both are
    compared the same way; only the depth differs.
    """
    short = [
        row["losses"]["null"] - row["losses"]["random"]
        for row in _record(ABOVE_SHORT)["per_seed"]
    ]
    long_ = [
        row["losses"]["null"] - row["losses"]["random"] for row in _record(ABOVE_LONG)["per_seed"]
    ]
    u, p = study._rank_sum_one_sided(short, long_)
    assert u == pytest.approx(28795.0)
    assert p < 1e-80
    assert _mean(ABOVE_SHORT, "null", "random") < 0 < _mean(ABOVE_LONG, "null", "random")


def test_the_budget_moved_the_value_reading_at_depth_four_and_not_at_depth_three() -> None:
    """The 2x2's two halves, as the outcome table states them."""
    short_v = [row["losses"]["adaptive"] - row["losses"]["null"] for row in _record(SHORT)["per_seed"]]
    long_v = [row["losses"]["adaptive"] - row["losses"]["null"] for row in _record(CELL)["per_seed"]]
    _, p = study._rank_sum_one_sided(short_v, long_v)
    assert p == pytest.approx(0.0819, abs=1e-3), "the below-onset value move is not significant"
    assert p > study.ALPHA / 2

    above_short = [
        row["losses"]["adaptive"] - row["losses"]["null"]
        for row in _record(ABOVE_SHORT)["per_seed"]
    ]
    above_long = [
        row["losses"]["adaptive"] - row["losses"]["null"] for row in _record(ABOVE_LONG)["per_seed"]
    ]
    _, above_p = study._rank_sum_one_sided(above_long, above_short)
    assert above_p < 1e-50, "the above-onset value move is the one that happened"
    assert _mean(ABOVE_SHORT, "adaptive", "null") > _mean(ABOVE_LONG, "adaptive", "null")


def test_the_controller_edge_survives_at_both_budgets() -> None:
    """The search still selects; it just cannot beat doing nothing."""
    edge = _record(CELL)["comparisons"]["adaptive_vs_random"]
    assert edge["wins"] == 443
    assert edge["mean_delta"] == pytest.approx(-0.08938283721605937)
    assert edge["passes_bonferroni"] is True
    collapse = _record(CELL)["comparisons"]["null_vs_random"]
    assert collapse["wins"] == 400
    assert collapse["mean_delta"] == pytest.approx(-0.12838778903500903)
    assert collapse["passes_bonferroni"] is True
    # Negative collapse means the do-nothing arm is the better one - the same
    # sign convention 5.52's table states ("positive = the do-nothing arm is the
    # worse one"), and the reason the regime has not moved.
    assert collapse["mean_delta"] < 0


def test_the_report_quotes_the_recorded_cell() -> None:
    """The prose's own numbers, normalized so wrapping cannot hide a drift."""
    report = " ".join((ROOT / "docs" / "RESEARCH_REPORT.md").read_text().split())
    for fragment in (
        "**Outcome (run 2026-09-21, revision `64046da`, 20 shards; committed summary",
        "`depth3-long-576.json`). Registered branch 4: both FAIL.**",
        "`281/576` wins against the bar of `313` (mean `+0.039005`, p `0.734`)",
        "`-0.106794` -> `-0.128388`",
        "rank-sum U `112872.0`, one-sided p `0.706`",
        "`6.3063` < `6.3453` < `6.4347`",
        "p `3.4e-40`",
    ):
        assert fragment in report, fragment
    # And the arm is in the study table for its own report section.
    assert study.PROFILES[PROFILE]["summary"].endswith(CELL)
