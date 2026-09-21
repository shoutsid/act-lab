"""Section 5.55's recorded result: the first branch, and the ladder behind it.

The arm registered four branches and landed on P1+P2 PASS - the gate is open
above the onset and the axis pays. These pins are read from the committed
summary, so a regenerated record that changes a win count, a mean or the branch
label fails here rather than silently rewriting the report.
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
PROFILE = study.DEPTH_DOSE_PROFILE
CELL = "depth5-dose-576.json"
REFERENCE = "null-control-depth4-prefix-384.json"
# The rest of the ladder, in depth order.
LADDER = (
    ("depth2", "null-control-depth2-384.json"),
    ("depth3", "null-control-depth3-384.json"),
    ("depth4-prefix", REFERENCE),
    (PROFILE, CELL),
)


def _record(name: str) -> dict:
    return json.loads((SUMMARIES / name).read_text())


def _col(profile: str, left: str, right: str) -> list[float]:
    name = Path(study.PROFILES[profile]["summary"]).name
    return [
        row["losses"][left] - row["losses"][right] for row in _record(name)["per_seed"]
    ]


def _mean(profile: str, left: str, right: str) -> float:
    values = _col(profile, left, right)
    return sum(values) / len(values)


def test_the_summary_is_the_frozen_revision_and_the_registered_arm() -> None:
    summary = _record(CELL)
    assert summary["profile"] == PROFILE
    assert summary["report_section"] == "5.55"
    assert summary["code_revision"] == study.PROFILES[PROFILE]["run_revision"] == "5fb7bd3"
    assert len(summary["seeds"]) == 576
    assert summary["seeds"][0] == 91384
    assert summary["seeds"][-1] == 91959
    assert summary["bonferroni_alpha"] == pytest.approx(study.ALPHA / 2)
    assert summary["comparisons"]["adaptive_vs_null"]["min_wins_bonferroni"] == 313


def test_the_recorded_cell_statistics() -> None:
    """P1's four numbers, as the report quotes them."""
    cell = _record(CELL)["comparisons"]["adaptive_vs_null"]
    assert cell["wins"] == 375
    assert cell["n"] == 576
    assert cell["mean_delta"] == pytest.approx(-0.07533454253441757)
    assert cell["binomial_p"] == pytest.approx(1.889e-13, rel=1e-3)
    assert cell["passes_bonferroni"] is True
    assert cell["wins"] > cell["min_wins_bonferroni"]
    assert cell["mean_delta"] < 0


def test_the_arm_means_keep_the_above_onset_ordering() -> None:
    arms = _record(CELL)["arm_means"]
    assert arms["adaptive"] == pytest.approx(11.4870, abs=5e-4)
    assert arms["null"] == pytest.approx(11.5624, abs=5e-4)
    assert arms["random"] == pytest.approx(11.5916, abs=5e-4)
    # Same ordering as the committed above-onset cell, and the opposite of the
    # two below-onset rungs - which is what makes the arm ordering a second
    # reading of the ladder.
    assert arms["adaptive"] < arms["null"] < arms["random"]
    assert _record(REFERENCE)["arm_means"]["adaptive"] < _record(REFERENCE)["arm_means"]["null"]
    for below in ("depth2", "depth3"):
        below_arms = _record(Path(study.PROFILES[below]["summary"]).name)["arm_means"]
        assert below_arms["null"] == min(below_arms.values()), below


def test_the_outcome_names_the_first_branch() -> None:
    summary = _record(CELL)
    assert summary["outcome"].startswith("cell creates value (5.55's per-cell primary")
    assert "branch " not in summary["outcome"]


def test_the_ladder_is_monotone_in_depth() -> None:
    """Four rungs, one direction: `+0.0208`, `+0.0046`, `-0.0371`, `-0.0753`."""
    values = {profile: _mean(profile, "adaptive", "null") for profile, _ in LADDER}
    assert values["depth2"] == pytest.approx(0.020811, abs=1e-6)
    assert values["depth3"] == pytest.approx(0.004643, abs=1e-6)
    assert values["depth4-prefix"] == pytest.approx(-0.037147, abs=1e-6)
    assert values[PROFILE] == pytest.approx(-0.075335, abs=1e-6)
    ordered = [values[profile] for profile, _ in LADDER]
    assert ordered == sorted(ordered, reverse=True), "the ladder is not monotone"
    # The increments, as the outcome states them.
    increments = [b - a for a, b in zip(ordered, ordered[1:])]
    assert increments[0] == pytest.approx(-0.0162, abs=1e-3)
    assert increments[1] == pytest.approx(-0.0417, abs=1e-3)
    assert increments[2] == pytest.approx(-0.0382, abs=1e-3)
    # The largest single step is the one that crosses the gate.
    assert min(increments) == increments[1]


def test_the_improvement_above_the_onset_is_not_the_controller() -> None:
    """The decomposition: the edge is flat and the churn term moves."""
    edges = {p: _mean(p, "adaptive", "random") for p, _ in LADDER}
    collapses = {p: _mean(p, "null", "random") for p, _ in LADDER}
    assert edges["depth2"] == pytest.approx(-0.078848, abs=1e-6)
    assert edges["depth3"] == pytest.approx(-0.102151, abs=1e-6)
    assert edges["depth4-prefix"] == pytest.approx(-0.107220, abs=1e-6)
    assert edges[PROFILE] == pytest.approx(-0.104561, abs=1e-6)
    assert collapses["depth4-prefix"] == pytest.approx(-0.070072, abs=1e-6)
    assert collapses[PROFILE] == pytest.approx(-0.029226, abs=1e-6)
    # Recomputing both rank-sums the report quotes.
    short, long_ = _col("depth4-prefix", "adaptive", "random"), _col(PROFILE, "adaptive", "random")
    u, p = study._rank_sum_one_sided(short, long_)
    assert u == pytest.approx(109725.0)
    assert p == pytest.approx(0.4184, abs=1e-3), "the controller's edge is not flat"
    _, reverse = study._rank_sum_one_sided(long_, short)
    assert reverse == pytest.approx(0.5817, abs=1e-3)
    short_c = _col("depth4-prefix", "null", "random")
    long_c = _col(PROFILE, "null", "random")
    u_c, p_c = study._rank_sum_one_sided(short_c, long_c)
    assert u_c == pytest.approx(95041.0)
    assert p_c == pytest.approx(1.100e-04, rel=1e-2)
    # And the identity the decomposition rests on, at every rung.
    for profile, _ in LADDER:
        assert _mean(profile, "adaptive", "random") - _mean(profile, "null", "random") == (
            pytest.approx(_mean(profile, "adaptive", "null"))
        )


def test_the_report_quotes_the_recorded_cell() -> None:
    """The prose's own numbers, normalized so wrapping cannot hide a drift."""
    report = " ".join((ROOT / "docs" / "RESEARCH_REPORT.md").read_text().split())
    for fragment in (
        "**Outcome (run 2026-09-21, revision `5fb7bd3`, 20 shards; committed summary",
        "`depth5-dose-576.json`). Registered branch 1: P1+P2 PASS.**",
        "`375/576` against `313`, mean `-0.075335`, p `1.89e-13`",
        "rank-sum U `97584.0`, one-sided p `9.99e-04`",
        "`+0.0208`, `+0.0046`, `-0.0371`, `-0.0753`",
        "U `109725.0`, p `0.418`",
        "U `95041.0`, p `1.1e-04`",
    ):
        assert fragment in report, fragment
    # And the arm is in the study table for its own report section.
    assert study.PROFILES[PROFILE]["summary"].endswith(CELL)
