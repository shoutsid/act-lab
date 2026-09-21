"""Section 5.57's recorded result: the fourth rung, and a forecast that half missed.

The arm was registered against the constant-gradient model's *own numbers* - an
increment in `(-0.047, -0.033)` and a controller's edge within `0.020` of the
depth-6 reading - so the record to pin is not only the branch it landed on, which
is P1+P2 PASS, but the two forecasts evaluated against it. One hit and one missed,
and the miss is the finding: the fourth increment is 1.5x all three that produced
the model, so the axis is super-linear past depth 6 rather than constant.

Everything here is recomputed from the committed summaries and the declared bands.
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
PROFILE = study.DEPTH7_PROFILE
CELL = "depth7-dose-576.json"
REFERENCE = "depth6-dose-576.json"
LADDER = (
    ("depth2", "null-control-depth2-384.json"),
    ("depth3", "null-control-depth3-384.json"),
    ("depth4-prefix", "null-control-depth4-prefix-384.json"),
    ("depth5-dose", "depth5-dose-576.json"),
    ("depth6-dose", REFERENCE),
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
    assert summary["report_section"] == "5.57"
    assert summary["code_revision"] == study.PROFILES[PROFILE]["run_revision"] == "18ae507"
    assert len(summary["seeds"]) == 576
    assert summary["seeds"][0] == 93384
    assert summary["seeds"][-1] == 93959
    assert summary["bonferroni_alpha"] == pytest.approx(study.ALPHA / 2)
    assert summary["comparisons"]["adaptive_vs_null"]["min_wins_bonferroni"] == 313


def test_the_recorded_cell_statistics() -> None:
    """Both primaries' numbers, as the report quotes them."""
    cell = _record(CELL)["comparisons"]["adaptive_vs_null"]
    assert cell["wins"] == 503
    assert cell["n"] == 576
    assert cell["mean_delta"] == pytest.approx(-0.181042, abs=1e-6)
    assert cell["binomial_p"] == pytest.approx(2.92e-80, rel=1e-2)
    assert cell["passes_bonferroni"] is True

    six = _col(REFERENCE, "adaptive", "null")
    seven = _col(CELL, "adaptive", "null")
    u, p = study._rank_sum_one_sided(seven, six)
    assert u == pytest.approx(128927.0)
    assert p == pytest.approx(2.95e-11, rel=1e-2)
    assert p <= study.ALPHA / 2


def test_the_outcome_names_the_first_branch() -> None:
    summary = _record(CELL)
    assert summary["outcome"].startswith("cell creates value (5.57's per-cell primary")
    assert "branch " not in summary["outcome"]


def test_the_ladder_is_monotone_and_the_gradient_steepens_at_the_fourth_rung() -> None:
    """Six rungs, one direction - and a fourth step that breaks the flat series.

    5.56 read three increments near `0.04` as a constant gradient. This arm was
    registered against exactly that reading, and the reading is what did not hold:
    the last three steps are flat to within 10% for two of them and then `0.0635`.
    Pinned both ways, because a report that goes back to calling the series
    constant is claiming a model this record refutes.
    """
    values = [_mean(name, "adaptive", "null") for _, name in LADDER]
    assert values[0] == pytest.approx(0.020811, abs=1e-6)
    assert values[1] == pytest.approx(0.004643, abs=1e-6)
    assert values[2] == pytest.approx(-0.037147, abs=1e-6)
    assert values[3] == pytest.approx(-0.075335, abs=1e-6)
    assert values[4] == pytest.approx(-0.117494, abs=1e-6)
    assert values[5] == pytest.approx(-0.181042, abs=1e-6)
    assert values == sorted(values, reverse=True), "the ladder is not monotone"

    increments = [b - a for a, b in zip(values, values[1:])]
    assert increments[0] == pytest.approx(-0.0162, abs=1e-3)
    for step in increments[1:4]:
        assert step == pytest.approx(-0.0407, abs=4e-3), increments
    assert increments[4] == pytest.approx(-0.0635, abs=1e-3), increments
    # The claim, stated as the arithmetic that makes it true: the newest step is
    # at least 1.4x every other step above the gate, and the largest of all five.
    assert abs(increments[4]) > 1.4 * max(abs(s) for s in increments[1:4])
    assert abs(increments[4]) == max(abs(s) for s in increments)


def test_the_prediction_missed_the_increment_and_hit_the_edge() -> None:
    """The forecast was declared as two numbers, so both are read against it."""
    six, seven = _mean(REFERENCE, "adaptive", "null"), _mean(CELL, "adaptive", "null")
    increment = seven - six
    low, high = study.DEPTH7_GRADIENT_BAND
    assert not (low <= increment <= high), (
        "the declared gradient band contains the measurement, so this record no "
        "longer shows the miss it reports"
    )
    assert increment == pytest.approx(-0.063548, abs=1e-6)

    edge_six = _mean(REFERENCE, "adaptive", "random")
    edge_seven = _mean(CELL, "adaptive", "random")
    assert abs(edge_six - edge_seven) <= study.DEPTH7_EDGE_BAND
    assert edge_six == pytest.approx(-0.089482, abs=1e-6)
    assert edge_seven == pytest.approx(-0.086622, abs=1e-6)


def test_the_arm_levels_hold_the_churn_regime_depth_six_opened() -> None:
    """The collapse keeps growing, and the arm order is the one 5.56 established."""
    names = [name for _, name in LADDER]
    collapses = [_mean(name, "null", "random") for name in names]
    assert collapses[:4] == pytest.approx(
        [-0.099659, -0.106794, -0.070072, -0.029226], abs=1e-6
    )
    assert collapses[4] == pytest.approx(0.028013, abs=1e-6)
    assert collapses[5] == pytest.approx(0.094421, abs=1e-6)
    assert all(c < 0 for c in collapses[:4]) and all(c > 0 for c in collapses[4:])
    # The newest step grows the do-nothing arm's disadvantage by more than 3x.
    assert collapses[5] / collapses[4] > 3.0
    last = _record(CELL)["arm_means"]
    assert last["adaptive"] < last["random"] < last["null"]
    assert last["null"] - last["adaptive"] == pytest.approx(0.181042, abs=1e-5)


def test_the_newest_gain_is_again_the_churn_term_and_not_the_controller() -> None:
    """The 6 -> 7 step decomposed, and the ladder it extends to 3 -> 7."""
    edges = (
        _mean(REFERENCE, "adaptive", "random"),
        _mean(CELL, "adaptive", "random"),
    )
    collapses = (_mean(REFERENCE, "null", "random"), _mean(CELL, "null", "random"))
    values = (_mean(REFERENCE, "adaptive", "null"), _mean(CELL, "adaptive", "null"))

    u, p = study._rank_sum_one_sided(
        _col(CELL, "adaptive", "random"), _col(REFERENCE, "adaptive", "random")
    )
    assert u == pytest.approx(168265.0)
    assert p == pytest.approx(0.663, abs=1e-2), "the controller's edge is not flat here"
    _, reverse = study._rank_sum_one_sided(
        _col(REFERENCE, "adaptive", "random"), _col(CELL, "adaptive", "random")
    )
    assert reverse == pytest.approx(0.337, abs=1e-2)

    u_c, p_c = study._rank_sum_one_sided(
        _col(REFERENCE, "null", "random"), _col(CELL, "null", "random")
    )
    assert u_c == pytest.approx(128811.0)
    assert p_c == pytest.approx(2.6e-11, rel=1e-2)

    edge_part = edges[1] - edges[0]
    churn_part = -(collapses[1] - collapses[0])
    improvement = values[1] - values[0]
    assert improvement == pytest.approx(-0.063548, abs=1e-6)
    assert edge_part == pytest.approx(0.002860, abs=1e-6)
    assert churn_part == pytest.approx(-0.066408, abs=1e-6)
    assert edge_part + churn_part == pytest.approx(improvement)
    assert edge_part > 0, "the controller's contribution is not a gain at all"

    # And the whole upper ladder, which now runs to depth 7.
    three = "null-control-depth3-384.json"
    # Same argument order the depth-6 record uses (shallower first, so a small p
    # reads "the shallower cell's value is the smaller one"), so the two ladders'
    # rank-sums are comparable rather than mirrored.
    u_e, p_e = study._rank_sum_one_sided(
        _col(three, "adaptive", "random"), _col(CELL, "adaptive", "random")
    )
    assert u_e == pytest.approx(106031.5)
    assert p_e == pytest.approx(0.139, abs=1e-3)
    _, reverse_e = study._rank_sum_one_sided(
        _col(CELL, "adaptive", "random"), _col(three, "adaptive", "random")
    )
    assert reverse_e == pytest.approx(0.861, abs=1e-3)
    u_l, p_l = study._rank_sum_one_sided(
        _col(three, "null", "random"), _col(CELL, "null", "random")
    )
    assert u_l == pytest.approx(40968.0)
    assert p_l == pytest.approx(9.0e-62, rel=1e-2)
    assert _mean(CELL, "adaptive", "null") - _mean(three, "adaptive", "null") == (
        pytest.approx(-0.185686, abs=1e-6)
    )
    assert (
        _mean(CELL, "adaptive", "random") - _mean(three, "adaptive", "random")
    ) == pytest.approx(0.015529, abs=1e-6)
    assert -(_mean(CELL, "null", "random") - _mean(three, "null", "random")) == (
        pytest.approx(-0.201215, abs=1e-6)
    )


def test_the_reports_ladder_row_is_the_recorded_cell() -> None:
    """The outcome table's own row, not only the fragments quoted around it."""
    rows = [
        line
        for line in (ROOT / "docs" / "RESEARCH_REPORT.md").read_text().splitlines()
        if line.startswith("| **`depth7-dose`** | **7** | **576**")
    ]
    assert len(rows) == 1, f"the outcome ladder row is missing or duplicated: {rows}"
    assert rows[0] == (
        "| **`depth7-dose`** | **7** | **576** | **`-0.181042`** | `-0.086622` | "
        "**`+0.094421`** | **adaptive** | **`9.7065 / 9.8876 / 9.7932`** |"
    ), rows[0]


def test_the_report_quotes_the_recorded_cell() -> None:
    """The prose's own numbers, normalized so wrapping cannot hide a drift."""
    report = " ".join((ROOT / "docs" / "RESEARCH_REPORT.md").read_text().split())
    for fragment in (
        "revision `18ae507`, 20 shards; committed summary `depth7-dose-576.json`)",
        "Registered branch 1: P1+P2 PASS - and the declared increment missed",
        "`503/576` against `313`, mean `-0.181042`",
        "rank-sum U `128927.0`, one-sided p `2.95e-11`",
        "`-0.0418` (3 -> 4), `-0.0382` (4 -> 5), `-0.0422` (5 -> 6) and `-0.0635` (6 -> 7)",
        "a **miss, and 1.5x every previous step of the upper ladder**",
        "the controller's own edge is flat (rank-sum p `0.663`)",
        "contributes `-0.066408` and the edge contributes `+0.002860`",
        "U `106031.5`, p `0.139`",
        "U `40968.0`, p `9.0e-62`",
        "`-0.185686`",
        "`-0.201215`",
        "`+0.094421`",
    ):
        assert fragment in report, fragment


def test_the_declared_bands_are_the_registered_ones() -> None:
    """The forecast is only a forecast if its band was fixed before the run."""
    assert study.DEPTH7_GRADIENT_BAND == (-0.047, -0.033)
    assert study.DEPTH7_EDGE_BAND == 0.020
    assert study.DEPTH7_REFERENCE == study.DEPTH6_PROFILE
    assert study.PROFILES[PROFILE]["summary"].endswith(CELL)
