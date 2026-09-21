"""Section 5.61's recorded result: the second bisection, where both brackets hit.

The registration declared the gate still shut at 4.01x and two brackets - the
depth-3 edge between its 2.87x and 5.66x readings, and the pool term positive but
below its 2.87x peak. **All three obtained.** The gate misses the bar by five
seeds (`310/576`, mean `-0.028562`, p `0.0365`), so the width price narrows to
(123,006, 173,672] parameters and the ratio to the depth axis's bracket becomes
**(4.01, 7.46]**.

The brackets are the interesting half: both hold, and the *honest* form is that
neither local slope inside them is sharp - the pool term is not yet significantly
below its peak (p `0.275`) and the edge is only marginally above the 5.66x reading
(p `0.051`). What the dose locates is the two turning points: the pool peaks at
2.87x, the edge at 5.66x, and the gate opens in the window between them.

Also pinned: the mean crossed while the win rate did not - the near miss is a
statement about which statistic moved, not a clean non-detection.
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
BELOW = "width-d3-dose4-576.json"
ABOVE = "width-d4-dose4-576.json"
BELOW_LOWER = "width-d3-dose-576.json"  # 2.87x neighbour
BELOW_UPPER = "width-d3-dose3-576.json"  # 5.66x neighbour
ABOVE_LOWER = "width-d4-dose-576.json"  # 3.80x neighbour
ABOVE_UPPER = "width-d4-dose3-576.json"  # 7.52x neighbour


def _record(name: str) -> dict:
    return json.loads((SUMMARIES / name).read_text())


def _col(name: str, left: str, right: str) -> list[float]:
    return [row["losses"][left] - row["losses"][right] for row in _record(name)["per_seed"]]


def _mean(name: str, left: str, right: str) -> float:
    values = _col(name, left, right)
    return sum(values) / len(values)


def test_both_summaries_are_the_frozen_revision_and_their_blocks() -> None:
    for name, first, last in ((BELOW, 98384, 98959), (ABOVE, 99384, 99959)):
        summary = _record(name)
        assert summary["code_revision"] == "04dc4b3"
        assert summary["report_section"] == "5.61"
        assert len(summary["seeds"]) == 576
        assert summary["seeds"][0] == first and summary["seeds"][-1] == last
        assert summary["bonferroni_alpha"] == pytest.approx(study.ALPHA / 3)
        assert summary["comparisons"]["adaptive_vs_null"]["min_wins_bonferroni"] == 315


def test_the_near_miss_is_recorded_as_a_near_miss() -> None:
    """Five seeds short of the bar, with the mean already crossed.

    This is the pin that keeps the reading honest. "The gate is still shut at
    4.01x" is true at the declared level and it is *not* the same statement as
    "this dose is inert": the mean delta has flipped sign since 2.87x while the
    win rate has not. A rewrite that reports either half alone fails here.
    """
    below = _record(BELOW)["comparisons"]["adaptive_vs_null"]
    assert below["wins"] == 310
    assert below["min_wins_bonferroni"] == 315
    assert below["wins"] == 315 - 5, "the miss is no longer the recorded five seeds"
    assert below["passes_bonferroni"] is False
    assert below["mean_delta"] == pytest.approx(-0.028562, abs=1e-6)
    assert below["binomial_p"] == pytest.approx(0.0365, rel=1e-2)
    # The mean has crossed: it was positive one dose lower.
    assert _mean(BELOW_LOWER, "adaptive", "null") == pytest.approx(+0.061466, abs=1e-6)
    assert below["mean_delta"] < 0 < _mean(BELOW_LOWER, "adaptive", "null")

    above = _record(ABOVE)["comparisons"]["adaptive_vs_null"]
    assert above["wins"] == 551
    assert above["mean_delta"] == pytest.approx(-0.418214, abs=1e-6)
    assert above["binomial_p"] == pytest.approx(1.65e-130, rel=1e-2)
    assert above["passes_bonferroni"] is True

    u, p3 = study._rank_sum_one_sided(
        _col(ABOVE, "adaptive", "null"), _col(ABOVE_LOWER, "adaptive", "null")
    )
    assert u == pytest.approx(88052.0)
    assert p3 == pytest.approx(1.55e-43, rel=1e-2)


def test_both_declared_brackets_hold_with_their_qualifications() -> None:
    """The brackets are not free: their bounds are the neighbours' own readings.

    Both obtained, and both are *gentle* inside - which is the part that has to
    survive. The pool term is not significantly below its peak, and the edge is
    only marginally above its upper neighbour's reading.
    """
    edge_low, edge_high = study.DOSE4_EDGE_BRACKET
    edge = _mean(BELOW, "adaptive", "random")
    assert edge_low == pytest.approx(_mean(BELOW_UPPER, "adaptive", "random"), abs=1e-6)
    assert edge_high == pytest.approx(_mean(BELOW_LOWER, "adaptive", "random"), abs=1e-6)
    assert edge == pytest.approx(-0.326291, abs=1e-6)
    assert edge_low < edge < edge_high, "the edge is outside its declared bracket"

    pool_low, pool_high = study.DOSE4_POOL_BRACKET
    pool = _mean(BELOW, "random", "null")
    assert pool_low == 0.0
    assert pool_high == pytest.approx(_mean(BELOW_LOWER, "random", "null"), abs=1e-6)
    assert pool == pytest.approx(+0.297729, abs=1e-6)
    assert pool_low < pool < pool_high, "the pool term is outside its declared bracket"

    # The qualifications, in the direction the report states them.
    _, edge_over_lower = study._rank_sum_one_sided(
        _col(BELOW, "adaptive", "random"), _col(BELOW_LOWER, "adaptive", "random")
    )
    assert edge_over_lower == pytest.approx(7.7e-05, rel=1e-2)
    _, edge_under_upper = study._rank_sum_one_sided(
        _col(BELOW_UPPER, "adaptive", "random"), _col(BELOW, "adaptive", "random")
    )
    assert edge_under_upper == pytest.approx(0.051, abs=1e-3), "the edge's rise is no longer marginal"
    _, pool_under_peak = study._rank_sum_one_sided(
        _col(BELOW, "random", "null"), _col(BELOW_LOWER, "random", "null")
    )
    assert pool_under_peak == pytest.approx(0.275, abs=1e-2), "the pool's decline is no longer gentle"
    # The pool term is *larger* at 4.01x than at 5.66x - so the reading is taken
    # in the reversed direction, and the forward one is asserted too so neither
    # can be quoted as the other.
    _, pool_over_upper = study._rank_sum_one_sided(
        _col(BELOW_UPPER, "random", "null"), _col(BELOW, "random", "null")
    )
    assert pool_over_upper == pytest.approx(2.54e-03, rel=1e-2)
    _, pool_forward = study._rank_sum_one_sided(
        _col(BELOW, "random", "null"), _col(BELOW_UPPER, "random", "null")
    )
    assert pool_forward == pytest.approx(0.997, abs=1e-2)


def test_the_turning_points_are_where_the_report_says_they_are() -> None:
    """The pool peaks at 2.87x and the edge at 5.66x - read off the committed cells."""
    ladder = (
        "null-control-depth3-384.json",
        "width-d3-equal-576.json",
        BELOW_LOWER,
        BELOW,
        BELOW_UPPER,
        "width-d3-dose2-576.json",
    )
    edges = [_mean(name, "adaptive", "random") for name in ladder]
    pools = [_mean(name, "random", "null") for name in ladder]
    assert edges == pytest.approx(
        [-0.102151, -0.130727, -0.243801, -0.326291, -0.359770, -0.310646], abs=1e-6
    )
    assert pools == pytest.approx(
        [0.106794, 0.148365, 0.305267, 0.297729, 0.231778, -0.585988], abs=1e-6
    )
    assert edges[4] == min(edges), "the edge's maximum is not at 5.66x"
    assert pools[2] == max(pools), "the pool term's peak is not at 2.87x"
    # The gate opens strictly between the two turning points - read off the bar,
    # not off the mean's sign, since the mean at 4.01x has already crossed
    # (`-0.028562`) while the win rate has not.
    assert _record(BELOW)["comparisons"]["adaptive_vs_null"]["passes_bonferroni"] is False
    assert _record(BELOW_UPPER)["comparisons"]["adaptive_vs_null"]["passes_bonferroni"] is True
    assert _mean(BELOW, "adaptive", "null") < 0, (
        "the mean no longer shows the crossing that makes this dose a near miss"
    )
    assert pools[3] > 0, "the pool term is negative at the dose where the gate is still shut"


def test_the_above_gate_asymmetry_is_four_doses_deep() -> None:
    """Every above-gate step: the edge flat, the pool moving. No exceptions."""
    steps = ((ABOVE_LOWER, BELOW if False else ABOVE), (ABOVE, ABOVE_UPPER))
    for lower, upper in steps:
        _, edge_p = study._rank_sum_one_sided(
            _col(upper, "adaptive", "random"), _col(lower, "adaptive", "random")
        )
        _, pool_p = study._rank_sum_one_sided(
            _col(upper, "random", "null"), _col(lower, "random", "null")
        )
        assert edge_p > study.ALPHA / 3, (lower, upper, edge_p)
        assert pool_p < 1e-20, (lower, upper, pool_p)
    # The three committed steps, with the numbers the report quotes.
    assert _mean(ABOVE, "adaptive", "random") - _mean(
        ABOVE_LOWER, "adaptive", "random"
    ) == pytest.approx(-0.016327, abs=1e-6)
    assert _mean(ABOVE, "random", "null") - _mean(
        ABOVE_LOWER, "random", "null"
    ) == pytest.approx(-0.192214, abs=1e-6)


def test_the_tightened_price_ratio_is_the_brackets_quotient() -> None:
    """(4.01, 7.46] - the width window over the depth window, both ends bounded."""
    depth_low, depth_high = 23_280, 30_672
    width_low, width_high = 123_006, 173_672
    assert width_low / depth_high == pytest.approx(4.01, abs=0.01)
    assert width_high / depth_low == pytest.approx(7.46, abs=0.01)
    # And the window really is the 1.41x interval 5.61 left, not 5.60's 1.97x.
    assert width_high / width_low == pytest.approx(1.41, abs=0.01)
    assert 173_672 / 88_032 == pytest.approx(1.97, abs=0.01)


def test_the_report_quotes_the_recorded_cells_and_the_tightened_bound() -> None:
    report = " ".join((ROOT / "docs" / "RESEARCH_REPORT.md").read_text().split())
    for fragment in (
        "revision `04dc4b3`, 20 shards per cell",
        "the window narrows to (123,006, 173,672]",
        "330/576" if False else "310/576",
        "**by five seeds**",
        "`-0.028562`",
        "`0.0365`",
        "`-0.326291`",
        "`+0.297729`",
        "the pool term peaks at 2.87x and the controller's edge peaks at 5.66x",
        "**(4.01, 7.46]**",
        "`U = 88052.0`, p `1.55e-43`",
        # The qualifications are claims too: a bracket that obtained *gently* is
        # not the same statement as one that turned sharply, and the report's
        # synthesis is judged on the honest form.
        "not yet significantly below its 2.87x peak",
        "only marginally above its 5.66x reading (p `0.051`)",
    ):
        assert fragment in report, fragment
    rows = [
        line
        for line in (ROOT / "docs" / "RESEARCH_REPORT.md").read_text().splitlines()
        if line.startswith("| depth 3, width 57 (4.01x) |")
        or line.startswith("| depth 4, width 57 (5.32x) |")
    ]
    assert rows == [
        "| depth 3, width 57 (4.01x) | 123,006 | 310/576 | `-0.028562` | `0.0365` | **no** (bar `315`) |",
        "| depth 4, width 57 (5.32x) | 163,134 | 551/576 | `-0.418214` | `1.65e-130` | **yes** |",
    ], rows
    # 5.60's own interval is kept, scoped, rather than silently rewritten.
    assert "section 5.60's own outcome keeps the `2.9x-7.5x` interval it" in report
