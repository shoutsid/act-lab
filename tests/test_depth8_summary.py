"""Section 5.58's recorded result: the fifth rung, and a second forecast that missed.

5.57 asked the constant-gradient model for a number and the number came back 1.5x
too small, so this arm was registered against the *fourth increment itself* - a
choice between two readings rather than a fresh guess. The measurement came back
between them: the increment is `-0.054124`, inside neither band, so the flat model
is refuted by the fourth rung and the accelerating model by the fifth, and what the
ladder supports is the mild rise.

Everything here is recomputed from the committed summaries and the declared bands,
including the two predictions read against their own measurements.
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
PROFILE = study.DEPTH8_PROFILE
CELL = "depth8-dose-576.json"
REFERENCE = "depth7-dose-576.json"
LADDER = (
    ("depth2", "null-control-depth2-384.json"),
    ("depth3", "null-control-depth3-384.json"),
    ("depth4-prefix", "null-control-depth4-prefix-384.json"),
    ("depth5-dose", "depth5-dose-576.json"),
    ("depth6-dose", "depth6-dose-576.json"),
    ("depth7-dose", REFERENCE),
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
    assert summary["report_section"] == "5.58"
    assert summary["code_revision"] == study.PROFILES[PROFILE]["run_revision"] == "a21c4a9"
    assert len(summary["seeds"]) == 576
    assert summary["seeds"][0] == 94384
    assert summary["seeds"][-1] == 94959
    assert summary["bonferroni_alpha"] == pytest.approx(study.ALPHA / 2)
    assert summary["comparisons"]["adaptive_vs_null"]["min_wins_bonferroni"] == 313


def test_the_recorded_cell_statistics() -> None:
    """Both primaries' numbers, as the report quotes them."""
    cell = _record(CELL)["comparisons"]["adaptive_vs_null"]
    assert cell["wins"] == 544
    assert cell["n"] == 576
    assert cell["mean_delta"] == pytest.approx(-0.235166, abs=1e-6)
    assert cell["binomial_p"] == pytest.approx(1.46e-121, rel=1e-2)
    assert cell["passes_bonferroni"] is True

    seven = _col(REFERENCE, "adaptive", "null")
    eight = _col(CELL, "adaptive", "null")
    u, p = study._rank_sum_one_sided(eight, seven)
    assert u == pytest.approx(131833.0)
    assert p == pytest.approx(8.12e-10, rel=1e-2)
    assert p <= study.ALPHA / 2


def test_the_outcome_names_the_first_branch() -> None:
    summary = _record(CELL)
    assert summary["outcome"].startswith("cell creates value (5.58's per-cell primary")
    assert "branch " not in summary["outcome"]


def test_the_ladder_is_monotone_and_the_gradient_is_mildly_rising() -> None:
    """Seven rungs, one direction - and a series that refutes both forecasts.

    The flat reading needs the fourth increment to be ~`0.04`; the accelerating
    reading needs the fifth to stay at the fourth's size. Neither holds, and the
    two assertions below are the arithmetic that says so: the last two steps are
    above the flat level, and the newest is *below* the one before it.
    """
    values = [_mean(name, "adaptive", "null") for _, name in LADDER]
    assert values[0] == pytest.approx(0.020811, abs=1e-6)
    assert values[1] == pytest.approx(0.004643, abs=1e-6)
    assert values[2] == pytest.approx(-0.037147, abs=1e-6)
    assert values[3] == pytest.approx(-0.075335, abs=1e-6)
    assert values[4] == pytest.approx(-0.117494, abs=1e-6)
    assert values[5] == pytest.approx(-0.181042, abs=1e-6)
    assert values[6] == pytest.approx(-0.235166, abs=1e-6)
    assert values == sorted(values, reverse=True), "the ladder is not monotone"

    increments = [b - a for a, b in zip(values, values[1:])]
    assert increments[0] == pytest.approx(-0.0162, abs=1e-3)
    for step in increments[1:4]:
        assert step == pytest.approx(-0.0407, abs=4e-3), increments
    assert increments[4] == pytest.approx(-0.0635, abs=1e-3), increments
    assert increments[5] == pytest.approx(-0.0541, abs=1e-3), increments
    # The crossing: the newest step is above the flat level and below the fourth,
    # which is what makes the series neither constant nor accelerating.
    assert abs(increments[5]) > 1.2 * max(abs(s) for s in increments[1:4])
    assert abs(increments[5]) < abs(increments[4])


def test_both_predictions_missed_and_the_edge_prediction_hit_again() -> None:
    """The band this arm declared, and the one 5.57 declared, both excluded."""
    increment = _mean(CELL, "adaptive", "null") - _mean(REFERENCE, "adaptive", "null")
    assert increment == pytest.approx(-0.054124, abs=1e-6)
    accelerating_low, accelerating_high = study.DEPTH8_GRADIENT_BAND
    flat_low, flat_high = study.DEPTH7_GRADIENT_BAND
    assert not (accelerating_low <= increment <= accelerating_high), (
        "the accelerating band contains the measurement, so this record no longer "
        "shows the miss it reports"
    )
    assert not (flat_low <= increment <= flat_high), (
        "the flat band contains the measurement too, so neither forecast was refuted"
    )
    # And the two misses are on opposite sides: a low miss here, a high miss there.
    assert increment > accelerating_high
    assert increments_minus_fourth(REFERENCE) < flat_low

    edge_seven = _mean(REFERENCE, "adaptive", "random")
    edge_eight = _mean(CELL, "adaptive", "random")
    assert abs(edge_seven - edge_eight) <= study.DEPTH8_EDGE_BAND
    assert edge_seven == pytest.approx(-0.086622, abs=1e-6)
    assert edge_eight == pytest.approx(-0.070113, abs=1e-6)


def increments_minus_fourth(reference: str) -> float:
    """The 6 -> 7 increment: what 5.57 measured and what 5.58 was declared against."""
    six = _mean("depth6-dose-576.json", "adaptive", "null")
    return _mean(reference, "adaptive", "null") - six


def test_the_arm_levels_keep_extending_the_churn_regime() -> None:
    """The collapse keeps growing and the arm order is the one depth 6 opened."""
    names = [name for _, name in LADDER]
    collapses = [_mean(name, "null", "random") for name in names]
    assert collapses[:4] == pytest.approx(
        [-0.099659, -0.106794, -0.070072, -0.029226], abs=1e-6
    )
    assert collapses[4:] == pytest.approx([0.028013, 0.094421, 0.165053], abs=1e-6)
    assert all(c < 0 for c in collapses[:4]) and all(c > 0 for c in collapses[4:])
    # The newest step grows the do-nothing arm's disadvantage by more than 1.7x.
    assert collapses[6] / collapses[5] > 1.7
    last = _record(CELL)["arm_means"]
    assert last["adaptive"] < last["random"] < last["null"]
    assert last["null"] - last["adaptive"] == pytest.approx(0.235166, abs=1e-5)


def test_the_newest_gain_is_again_the_churn_term_and_not_the_controller() -> None:
    """The 7 -> 8 step decomposed, and the five-rung ladder it extends to 3 -> 8."""
    edges = (_mean(REFERENCE, "adaptive", "random"), _mean(CELL, "adaptive", "random"))
    collapses = (_mean(REFERENCE, "null", "random"), _mean(CELL, "null", "random"))
    values = (_mean(REFERENCE, "adaptive", "null"), _mean(CELL, "adaptive", "null"))

    u, p = study._rank_sum_one_sided(
        _col(REFERENCE, "adaptive", "random"), _col(CELL, "adaptive", "random")
    )
    assert u == pytest.approx(160319.0)
    assert p == pytest.approx(0.162, abs=1e-2), "the controller's edge is not flat here"

    u_c, p_c = study._rank_sum_one_sided(
        _col(REFERENCE, "null", "random"), _col(CELL, "null", "random")
    )
    assert u_c == pytest.approx(123866.0)
    assert p_c == pytest.approx(4.9e-14, rel=1e-2)

    edge_part = edges[1] - edges[0]
    churn_part = -(collapses[1] - collapses[0])
    improvement = values[1] - values[0]
    assert improvement == pytest.approx(-0.054124, abs=1e-6)
    assert edge_part == pytest.approx(0.016509, abs=1e-6)
    assert churn_part == pytest.approx(-0.070633, abs=1e-6)
    assert edge_part + churn_part == pytest.approx(improvement)
    assert edge_part > 0, "the controller's contribution is not a gain at all"

    # The whole upper ladder, now five rungs deep. The edge's own decline reaches
    # p 0.024 here - a descriptive, unregistered rank-sum, and the first reading in
    # which it reaches any significance level, which is why the report says so.
    three = "null-control-depth3-384.json"
    u_e, p_e = study._rank_sum_one_sided(
        _col(three, "adaptive", "random"), _col(CELL, "adaptive", "random")
    )
    assert u_e == pytest.approx(102296.0)
    assert p_e == pytest.approx(0.0244, abs=1e-3)
    _, reverse_e = study._rank_sum_one_sided(
        _col(CELL, "adaptive", "random"), _col(three, "adaptive", "random")
    )
    assert reverse_e == pytest.approx(0.976, abs=1e-3)
    u_l, p_l = study._rank_sum_one_sided(
        _col(three, "null", "random"), _col(CELL, "null", "random")
    )
    assert u_l == pytest.approx(25327.0)
    assert p_l == pytest.approx(1.5e-91, rel=1e-2)
    assert _mean(CELL, "adaptive", "null") - _mean(three, "adaptive", "null") == (
        pytest.approx(-0.239810, abs=1e-6)
    )
    assert (
        _mean(CELL, "adaptive", "random") - _mean(three, "adaptive", "random")
    ) == pytest.approx(0.032038, abs=1e-6)
    assert -(_mean(CELL, "null", "random") - _mean(three, "null", "random")) == (
        pytest.approx(-0.271848, abs=1e-6)
    )


def test_the_reports_ladder_row_is_the_recorded_cell() -> None:
    """The outcome table's own row, not only the fragments quoted around it."""
    rows = [
        line
        for line in (ROOT / "docs" / "RESEARCH_REPORT.md").read_text().splitlines()
        if line.startswith("| **`depth8-dose`** | **8** | **576**")
    ]
    assert len(rows) == 1, f"the outcome ladder row is missing or duplicated: {rows}"
    assert rows[0] == (
        "| **`depth8-dose`** | **8** | **576** | **`-0.235166`** | `-0.070113` | "
        "**`+0.165053`** | **adaptive** | **`9.0747 / 9.3099 / 9.1448`** |"
    ), rows[0]


def test_the_report_quotes_the_recorded_cell() -> None:
    """The prose's own numbers, normalized so wrapping cannot hide a drift."""
    report = " ".join((ROOT / "docs" / "RESEARCH_REPORT.md").read_text().split())
    for fragment in (
        "revision `a21c4a9`, 20 shards; committed summary `depth8-dose-576.json`)",
        "P1+P2 PASS - and the second forecast missed too",
        "`544/576` against `313`, mean `-0.235166`",
        "rank-sum U `131833.0`, one-sided p `8.12e-10`",
        "`-0.0162`, `-0.0418`, `-0.0382`, `-0.0422`, `-0.0635`, `-0.0541`",
        "U `102296.0`, p `0.0244` (reversed `0.976`)",
        "U `25327.0`, p `1.5e-91`",
        "`+0.016509`",
        "`-0.070633`",
        "`+0.165053`",
    ):
        assert fragment in report, fragment


def test_the_declared_bands_are_the_registered_ones() -> None:
    """The forecast is only a forecast if its band was fixed before the run."""
    assert study.DEPTH8_GRADIENT_BAND == (-0.069, -0.059)
    assert study.DEPTH8_EDGE_BAND == 0.020
    assert study.DEPTH8_REFERENCE == study.DEPTH7_PROFILE
    assert study.PROFILES[PROFILE]["summary"].endswith(CELL)
