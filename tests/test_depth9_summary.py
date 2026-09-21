"""Section 5.59's recorded result: the sixth rung, and a third forecast that missed.

Two forecasts had missed from opposite sides of the same series when this arm was
registered, which left three readings rather than two - flat, the mild rise the
last two increments showed, and something steeper than the fourth. This arm
declared the middle one. The measurement came back **in the 0.002-wide seam
between the two bands it was bracketed by**: `-0.048724`, above the mild-rise
band's shallow edge by `0.000276` and below the flat band on the same side, so
none of the three readings is what the axis is doing.

What is left standing is the shape three points make: increments `0.0635`,
`0.0541`, `0.0487` declining monotonically with the gap above the flat level
(`0.0407`) shrinking by a factor of ~0.59 each rung - an excursion decaying back
toward the flat regime rather than a steeper axis.

The second declared prediction hit for a fourth consecutive rung, and this is
also where 5.58's own marginal reading died: the controller's edge decline it
reported at p `0.0244` over 3 -> 8 reads p `0.879` over 3 -> 9.

Everything here is recomputed from the committed summaries and the declared
bands, including both predictions read against their own measurements.
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
PROFILE = study.DEPTH9_PROFILE
CELL = "depth9-dose-576.json"
REFERENCE = "depth8-dose-576.json"
DEEPEST_SHALLOW = "null-control-depth3-384.json"
FLAT_LEVEL = -0.040713
LADDER = (
    ("depth2", "null-control-depth2-384.json"),
    ("depth3", "null-control-depth3-384.json"),
    ("depth4-prefix", "null-control-depth4-prefix-384.json"),
    ("depth5-dose", "depth5-dose-576.json"),
    ("depth6-dose", "depth6-dose-576.json"),
    ("depth7-dose", "depth7-dose-576.json"),
    (REFERENCE, REFERENCE),
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


def _increments() -> list[float]:
    values = [_mean(name, "adaptive", "null") for _, name in LADDER]
    return [b - a for a, b in zip(values, values[1:])]


def test_the_summary_is_the_frozen_revision_and_the_registered_arm() -> None:
    summary = _record(CELL)
    assert summary["profile"] == PROFILE
    assert summary["report_section"] == "5.59"
    assert summary["code_revision"] == study.PROFILES[PROFILE]["run_revision"] == "1816e34"
    assert len(summary["seeds"]) == 576
    assert summary["seeds"][0] == 95384
    assert summary["seeds"][-1] == 95959
    assert summary["bonferroni_alpha"] == pytest.approx(study.ALPHA / 2)
    assert summary["comparisons"]["adaptive_vs_null"]["min_wins_bonferroni"] == 313


def test_the_recorded_cell_statistics() -> None:
    """Both primaries' numbers, as the report quotes them."""
    cell = _record(CELL)["comparisons"]["adaptive_vs_null"]
    assert cell["wins"] == 559
    assert cell["n"] == 576
    assert cell["mean_delta"] == pytest.approx(-0.283891, abs=1e-6)
    assert cell["binomial_p"] == pytest.approx(7.81e-142, rel=1e-2)
    assert cell["passes_bonferroni"] is True

    eight = _col(REFERENCE, "adaptive", "null")
    nine = _col(CELL, "adaptive", "null")
    u, p = study._rank_sum_one_sided(nine, eight)
    assert u == pytest.approx(135354.0)
    assert p == pytest.approx(3.19e-08, rel=1e-2)
    assert p <= study.ALPHA / 2


def test_the_outcome_names_the_first_branch() -> None:
    summary = _record(CELL)
    assert summary["outcome"].startswith("cell creates value (5.59's per-cell primary")
    assert "branch " not in summary["outcome"]


def test_the_ladder_is_monotone_and_the_excursion_is_decaying() -> None:
    """Eight rungs, one direction - and a series that is neither flat nor rising.

    The three readings that have been registered and missed are refuted by
    arithmetic, not by taste: the last three steps all sit *above* the flat
    level, so the flat model is wrong; they decline monotonically, so the
    accelerating model is wrong; and the decline is geometric at ~0.59, so the
    mild rise is an excursion on its way back rather than a new gradient.
    """
    values = [_mean(name, "adaptive", "null") for _, name in LADDER]
    assert values[0] == pytest.approx(0.020811, abs=1e-6)
    assert values[1] == pytest.approx(0.004643, abs=1e-6)
    assert values[2] == pytest.approx(-0.037147, abs=1e-6)
    assert values[3] == pytest.approx(-0.075335, abs=1e-6)
    assert values[4] == pytest.approx(-0.117494, abs=1e-6)
    assert values[5] == pytest.approx(-0.181042, abs=1e-6)
    assert values[6] == pytest.approx(-0.235166, abs=1e-6)
    assert values[7] == pytest.approx(-0.283891, abs=1e-6)
    assert values == sorted(values, reverse=True), "the ladder is not monotone"

    increments = _increments()
    assert increments[0] == pytest.approx(-0.0162, abs=1e-3)
    for step in increments[1:4]:
        assert step == pytest.approx(FLAT_LEVEL, abs=4e-3), increments
    assert increments[4] == pytest.approx(-0.0635, abs=1e-3), increments
    assert increments[5] == pytest.approx(-0.0541, abs=1e-3), increments
    assert increments[6] == pytest.approx(-0.0487, abs=1e-3), increments

    # Above the flat level: the constant-gradient reading is refuted by the
    # fourth rung and has not come back.
    excursion = [abs(step) - abs(FLAT_LEVEL) for step in increments[4:]]
    assert all(gap > 0 for gap in excursion), excursion
    # Monotone decline, and geometrically so: what is decaying is the excursion,
    # which is what makes the next increment predictable rather than open.
    assert excursion == sorted(excursion, reverse=True), excursion
    ratios = [b / a for a, b in zip(excursion, excursion[1:])]
    assert all(0.5 < ratio < 0.7 for ratio in ratios), ratios


def test_the_miss_landed_in_the_seam_between_the_two_bands() -> None:
    """Not in this arm's band, not in the flat band, and in neither's company.

    This is the reading the record has to keep: three registered bands have now
    been missed, and the third one landed in the gap the first two left. Naming
    it "flat" because it is the shallower side would be the report claiming more
    than the number says, which is why the verdict prints the seam instead.
    """
    increment = _increments()[6]
    assert increment == pytest.approx(-0.048724, abs=1e-6)
    low, high = study.DEPTH9_GRADIENT_BAND
    flat_low, flat_high = study.DEPTH7_GRADIENT_BAND
    assert not (low <= increment <= high), "5.59's own band contains the measurement"
    assert not (flat_low <= increment <= flat_high), "the flat band contains it too"
    # Between them, not below both: the seam is the interval the two bands left.
    assert high < increment < flat_low, (increment, high, flat_low)
    assert flat_low - high == pytest.approx(0.002), "the seam is 0.002 wide"
    assert increment - high == pytest.approx(0.000276, abs=1e-6)


def test_the_edge_prediction_hit_for_a_fourth_rung_and_5_58s_reading_died() -> None:
    """The one reading that had moved at all does not survive the next rung."""
    edge_eight = _mean(REFERENCE, "adaptive", "random")
    edge_nine = _mean(CELL, "adaptive", "random")
    assert edge_eight == pytest.approx(-0.070113, abs=1e-6)
    assert edge_nine == pytest.approx(-0.082532, abs=1e-6)
    assert abs(edge_nine - edge_eight) <= study.DEPTH9_EDGE_BAND

    # 5.58 reported this decline as descriptive because it was marginal. One more
    # rung sends it back: flat at p 0.879, which is the whole reason the report
    # refuses to carry a marginal reading into the synthesis.
    u_e, p_e = study._rank_sum_one_sided(
        _col(DEEPEST_SHALLOW, "adaptive", "random"), _col(CELL, "adaptive", "random")
    )
    assert u_e == pytest.approx(105663.0)
    assert p_e == pytest.approx(0.121, abs=1e-3)
    _, reverse_e = study._rank_sum_one_sided(
        _col(CELL, "adaptive", "random"), _col(DEEPEST_SHALLOW, "adaptive", "random")
    )
    assert reverse_e == pytest.approx(0.879, abs=1e-3)
    assert _mean(CELL, "adaptive", "random") - _mean(
        DEEPEST_SHALLOW, "adaptive", "random"
    ) == pytest.approx(0.019619, abs=1e-6)


def test_the_arm_levels_keep_extending_the_churn_regime() -> None:
    """The collapse keeps growing and the arm order is still the one depth 6 opened."""
    names = [name for _, name in LADDER]
    collapses = [_mean(name, "null", "random") for name in names]
    assert collapses[:4] == pytest.approx(
        [-0.099659, -0.106794, -0.070072, -0.029226], abs=1e-6
    )
    assert collapses[4:] == pytest.approx(
        [0.028013, 0.094421, 0.165053, 0.201358], abs=1e-6
    )
    assert all(c < 0 for c in collapses[:4]) and all(c > 0 for c in collapses[4:])
    last = _record(CELL)["arm_means"]
    assert last["adaptive"] < last["random"] < last["null"]
    assert last["null"] - last["adaptive"] == pytest.approx(0.283891, abs=1e-5)


def test_the_newest_gain_is_again_the_churn_term_and_not_the_controller() -> None:
    """The 8 -> 9 step decomposed, and the six-rung ladder it extends to 3 -> 9."""
    edges = (_mean(REFERENCE, "adaptive", "random"), _mean(CELL, "adaptive", "random"))
    collapses = (_mean(REFERENCE, "null", "random"), _mean(CELL, "null", "random"))
    values = (_mean(REFERENCE, "adaptive", "null"), _mean(CELL, "adaptive", "null"))

    u, p = study._rank_sum_one_sided(
        _col(REFERENCE, "adaptive", "random"), _col(CELL, "adaptive", "random")
    )
    assert p == pytest.approx(0.831, abs=1e-2), "the controller's edge is not flat here"

    u_c, p_c = study._rank_sum_one_sided(
        _col(REFERENCE, "null", "random"), _col(CELL, "null", "random")
    )
    assert p_c == pytest.approx(3.2e-06, rel=1e-2)

    edge_part = edges[1] - edges[0]
    churn_part = -(collapses[1] - collapses[0])
    improvement = values[1] - values[0]
    assert improvement == pytest.approx(-0.048724, abs=1e-6)
    assert edge_part == pytest.approx(-0.012419, abs=1e-6)
    assert churn_part == pytest.approx(-0.036305, abs=1e-6)
    assert edge_part + churn_part == pytest.approx(improvement)
    assert abs(churn_part) > 2 * abs(edge_part), "churn is no longer the majority term"

    # The whole upper ladder, now six rungs deep: the edge is flat across it and
    # the churn term carries more than the whole improvement.
    u_l, p_l = study._rank_sum_one_sided(
        _col(DEEPEST_SHALLOW, "null", "random"), _col(CELL, "null", "random")
    )
    assert u_l == pytest.approx(19510.0)
    assert p_l == pytest.approx(3.7e-104, rel=1e-2)
    assert _mean(CELL, "adaptive", "null") - _mean(
        DEEPEST_SHALLOW, "adaptive", "null"
    ) == pytest.approx(-0.288534, abs=1e-6)
    assert _mean(CELL, "adaptive", "random") - _mean(
        DEEPEST_SHALLOW, "adaptive", "random"
    ) == pytest.approx(0.019619, abs=1e-6)
    assert -(_mean(CELL, "null", "random") - _mean(
        DEEPEST_SHALLOW, "null", "random"
    )) == pytest.approx(-0.308153, abs=1e-6)


def test_the_reports_ladder_row_is_the_recorded_cell() -> None:
    """The outcome table's own row, not only the fragments quoted around it."""
    rows = [
        line
        for line in (ROOT / "docs" / "RESEARCH_REPORT.md").read_text().splitlines()
        if line.startswith("| **`depth9-dose`** | **9** | **576**")
    ]
    assert len(rows) == 1, f"the outcome ladder row is missing or duplicated: {rows}"
    assert rows[0] == (
        "| **`depth9-dose`** | **9** | **576** | **`-0.283891`** | **`-0.082532`** | "
        "**`+0.201358`** | **adaptive** | **`8.5254 / 8.8093 / 8.6079`** |"
    ), rows[0]


def test_the_report_quotes_the_recorded_cell() -> None:
    """The prose's own numbers, normalized so wrapping cannot hide a drift."""
    report = " ".join((ROOT / "docs" / "RESEARCH_REPORT.md").read_text().split())
    for fragment in (
        "revision `1816e34`, 20 shards; committed summary `depth9-dose-576.json`)",
        "P1+P2 PASS - and the increment missed for the third consecutive rung, into the "
        "seam between the two bands that preceded it",
        "`559/576` against `313`, mean `-0.283891`",
        "rank-sum U `135354.0`, one-sided p `3.19e-08`",
        "`-0.0418`, `-0.0382`, `-0.0422`, `-0.0635`, `-0.0541`, `-0.0487`",
        "`0.0228 -> 0.0134 -> 0.0080`, a factor of `0.59` and `0.60`",
        "the increment ending at depth 10 falls inside the flat band",
        "U `105663.0`, p `0.121` (reversed `0.879`)",
        "U `19510.0`, p `3.7e-104`",
        "`-0.036305`",
        "`-0.012419`",
        "`+0.201358`",
    ):
        assert fragment in report, fragment
    # The seam has to be described as a seam: a reading that called itself either
    # the flat or the mild-rise side would be claiming more than the number says.
    assert (
        "this one missed into the `0.002`-wide gap *between* the two bands its "
        "predecessors declared" in report
    )


def test_the_declared_bands_are_the_registered_ones() -> None:
    """The forecast is only a forecast if its band was fixed before the run."""
    assert study.DEPTH9_GRADIENT_BAND == (-0.069, -0.049)
    assert study.DEPTH9_EDGE_BAND == 0.020
    assert study.DEPTH9_REFERENCE == study.DEPTH8_PROFILE
    assert study.PROFILES[PROFILE]["summary"].endswith(CELL)
