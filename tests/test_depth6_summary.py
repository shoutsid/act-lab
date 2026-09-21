"""Section 5.56's recorded result: the third rung, the constant slope, and the flip.

The arm registered four branches and landed on P1+P2 PASS - the slope continues.
The interesting record is the ladder it completes and the decomposition of what
carries the gain, so both are pinned here from the committed summaries rather
than quoted from the prose.
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
PROFILE = study.DEPTH6_PROFILE
CELL = "depth6-dose-576.json"
LADDER = (
    ("depth2", "null-control-depth2-384.json"),
    ("depth3", "null-control-depth3-384.json"),
    ("depth4-prefix", "null-control-depth4-prefix-384.json"),
    ("depth5-dose", "depth5-dose-576.json"),
    (PROFILE, CELL),
)


def _record(name: str) -> dict:
    return json.loads((SUMMARIES / name).read_text())


def _col(name: str, left: str, right: str) -> list[float]:
    return [
        row["losses"][left] - row["losses"][right] for row in _record(name)["per_seed"]
    ]


def _mean(name: str, left: str, right: str) -> float:
    values = _col(name, left, right)
    return sum(values) / len(values)


def test_the_summary_is_the_frozen_revision_and_the_registered_arm() -> None:
    summary = _record(CELL)
    assert summary["profile"] == PROFILE
    assert summary["report_section"] == "5.56"
    assert summary["code_revision"] == study.PROFILES[PROFILE]["run_revision"] == "9f31495"
    assert len(summary["seeds"]) == 576
    assert summary["seeds"][0] == 92384
    assert summary["seeds"][-1] == 92959
    assert summary["bonferroni_alpha"] == pytest.approx(study.ALPHA / 2)
    assert summary["comparisons"]["adaptive_vs_null"]["min_wins_bonferroni"] == 313


def test_the_recorded_cell_statistics() -> None:
    """Both primaries' numbers, as the report quotes them."""
    cell = _record(CELL)["comparisons"]["adaptive_vs_null"]
    assert cell["wins"] == 442
    assert cell["n"] == 576
    assert cell["mean_delta"] == pytest.approx(-0.117494, abs=1e-6)
    assert cell["binomial_p"] == pytest.approx(1.13e-39, rel=1e-2)
    assert cell["passes_bonferroni"] is True

    five = _col("depth5-dose-576.json", "adaptive", "null")
    six = _col(CELL, "adaptive", "null")
    u, p = study._rank_sum_one_sided(six, five)
    assert u == pytest.approx(142035.0)
    assert p == pytest.approx(1.20e-05, rel=1e-2)
    assert p <= study.ALPHA / 2


def test_the_outcome_names_the_first_branch() -> None:
    summary = _record(CELL)
    assert summary["outcome"].startswith("cell creates value (5.56's per-cell primary")
    assert "branch " not in summary["outcome"]


def test_the_ladder_is_monotone_and_its_slope_does_not_decay() -> None:
    """Five rungs, one direction, and a gradient that stays near 0.04 per layer."""
    values = [_mean(name, "adaptive", "null") for _, name in LADDER]
    assert values[0] == pytest.approx(0.020811, abs=1e-6)
    assert values[1] == pytest.approx(0.004643, abs=1e-6)
    assert values[2] == pytest.approx(-0.037147, abs=1e-6)
    assert values[3] == pytest.approx(-0.075335, abs=1e-6)
    assert values[4] == pytest.approx(-0.117494, abs=1e-6)
    assert values == sorted(values, reverse=True), "the ladder is not monotone"
    increments = [b - a for a, b in zip(values, values[1:])]
    assert increments[0] == pytest.approx(-0.0162, abs=1e-3)
    for step in increments[1:]:
        # Past the first step every increment is the same size to within 10%.
        assert step == pytest.approx(-0.0407, abs=4e-3), increments
    # The newest step is the largest since the gate itself.
    assert min(increments[1:]) == increments[3]


def test_the_arm_levels_flip_the_churn_regime_at_depth_six() -> None:
    """The collapse crosses zero: doing nothing stops being the better arm."""
    names = [name for _, name in LADDER]
    collapses = [_mean(name, "null", "random") for name in names]
    assert collapses[:4] == pytest.approx(
        [-0.099659, -0.106794, -0.070072, -0.029226], abs=1e-6
    )
    assert collapses[4] == pytest.approx(0.028013, abs=1e-6)
    assert all(c < 0 for c in collapses[:4]) and collapses[4] > 0
    # And the arm orderings, which say the same thing in levels.
    for index in (0, 1):
        arms = _record(names[index])["arm_means"]
        assert arms["null"] == min(arms.values()), names[index]
    for index in (2, 3, 4):
        arms = _record(names[index])["arm_means"]
        assert arms["adaptive"] == min(arms.values()), names[index]
    last = _record(names[4])["arm_means"]
    assert last["adaptive"] < last["random"] < last["null"]


def test_the_gain_is_the_churn_term_and_not_the_controller() -> None:
    """The decomposition over the whole upper ladder, recomputed."""
    three, six = LADDER[1][1], CELL
    edges = (_mean(three, "adaptive", "random"), _mean(six, "adaptive", "random"))
    collapses = (_mean(three, "null", "random"), _mean(six, "null", "random"))
    values = (_mean(three, "adaptive", "null"), _mean(six, "adaptive", "null"))
    assert edges[0] == pytest.approx(-0.102151, abs=1e-6)
    assert edges[1] == pytest.approx(-0.089482, abs=1e-6)
    u, p = study._rank_sum_one_sided(
        _col(three, "adaptive", "random"), _col(six, "adaptive", "random")
    )
    assert u == pytest.approx(107963.0)
    assert p == pytest.approx(0.266, abs=1e-3), "the controller's edge is not flat over the ladder"
    _, reverse = study._rank_sum_one_sided(
        _col(six, "adaptive", "random"), _col(three, "adaptive", "random")
    )
    assert reverse == pytest.approx(0.734, abs=1e-3)

    u_c, p_c = study._rank_sum_one_sided(
        _col(three, "null", "random"), _col(six, "null", "random")
    )
    assert u_c == pytest.approx(63764.0)
    assert p_c < 1e-25
    # The split of the improvement, and the identity it rests on.
    improvement = values[1] - values[0]
    edge_part = edges[1] - edges[0]
    churn_part = -(collapses[1] - collapses[0])
    assert improvement == pytest.approx(-0.122137, abs=1e-6)
    assert edge_part == pytest.approx(0.012670, abs=1e-6)
    assert churn_part == pytest.approx(-0.134807, abs=1e-6)
    assert edge_part + churn_part == pytest.approx(improvement)
    assert edge_part > 0, "the controller's contribution is not a gain at all"


def test_the_report_quotes_the_recorded_cell() -> None:
    """The prose's own numbers, normalized so wrapping cannot hide a drift."""
    report = " ".join((ROOT / "docs" / "RESEARCH_REPORT.md").read_text().split())
    for fragment in (
        "revision `9f31495`, 20 shards; committed summary `depth6-dose-576.json`)",
        "Registered branch 1: P1+P2 PASS.**",
        "`442/576` against `313`, mean `-0.117494`",
        "rank-sum U `142035.0`, one-sided p `1.20e-05`",
        "`-0.0162` (2 -> 3), `-0.0418` (3 -> 4), `-0.0382` (4 -> 5) and `-0.0422` (5 -> 6)",
        "`+0.028013`",
        "U `107963.0`, p `0.266`",
        "U `63764.0`, p `4.7e-29`",
        "`+0.012670`",
        "`-0.134807`",
    ):
        assert fragment in report, fragment
    assert study.PROFILES[PROFILE]["summary"].endswith(CELL)
