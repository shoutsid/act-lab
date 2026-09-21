"""Section 5.60's recorded result: the width bisection, and the mechanism it replaced.

The registration bisected the window 5.49 and 5.50 left open (the width axis's
gate between 2.87x and 11.15x) at its geometric midpoint in parameter space.
**The gate opened at 5.66x**: 389/576, mean `-0.127992`, so capacity does buy the
gate at this dose and the window narrows to (2.87x, 5.66x) = 88,032-173,672
parameters.

The declared prediction missed, and the miss is the finding. It declared the pool
term would still be rising toward its flip; instead it has already **peaked at
2.87x** (`+0.305267` -> `+0.231778`, p `1.35e-04`) and the gate opened anyway. The
bisection shows why two cells could not: the width axis's gate opens in **two
stages** - at 5.66x it is the controller's own edge that crosses it, at its
ladder maximum, while the pool term is still positive; the unmutated collapse
takes over later, and above the onset it is the only mechanism that moves.

Everything here is recomputed from the committed summaries.
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
BELOW = "width-d3-dose3-576.json"
ABOVE = "width-d4-dose3-576.json"
BELOW_REFERENCE = "width-d3-dose-576.json"
ABOVE_REFERENCE = "width-d4-dose-576.json"

# The window, as parameter counts at depth 3 and the depth axis's own bracket.
WINDOW_LOW = 88_032  # 2.87x: closed
WINDOW_HIGH = 173_672  # 5.66x: open
DEPTH_OPEN = 30_672  # depth 4: the shallowest cell with the gate
DEPTH_CLOSED = 23_280  # depth 3: the deepest cell without it


def _record(name: str) -> dict:
    return json.loads((SUMMARIES / name).read_text())


def _col(name: str, left: str, right: str) -> list[float]:
    return [row["losses"][left] - row["losses"][right] for row in _record(name)["per_seed"]]


def _mean(name: str, left: str, right: str) -> float:
    values = _col(name, left, right)
    return sum(values) / len(values)


def test_both_summaries_are_the_frozen_revision_and_their_blocks() -> None:
    for name, first, last in (
        (BELOW, 96384, 96959),
        (ABOVE, 97384, 97959),
    ):
        summary = _record(name)
        assert summary["code_revision"] == "698a866"
        assert summary["report_section"] == "5.60"
        assert len(summary["seeds"]) == 576
        assert summary["seeds"][0] == first and summary["seeds"][-1] == last
        assert summary["bonferroni_alpha"] == pytest.approx(study.ALPHA / 3)
        assert summary["comparisons"]["adaptive_vs_null"]["min_wins_bonferroni"] == 315


def test_the_recorded_cell_statistics_and_the_third_primary() -> None:
    below = _record(BELOW)["comparisons"]["adaptive_vs_null"]
    assert below["wins"] == 389
    assert below["mean_delta"] == pytest.approx(-0.127992, abs=1e-6)
    assert below["binomial_p"] == pytest.approx(1.31e-17, rel=1e-2)
    assert below["passes_bonferroni"] is True

    above = _record(ABOVE)["comparisons"]["adaptive_vs_null"]
    assert above["wins"] == 575
    assert above["mean_delta"] == pytest.approx(-0.725298, abs=1e-6)
    assert above["binomial_p"] == pytest.approx(2.33e-171, rel=1e-2)

    u, p3 = study._rank_sum_one_sided(
        _col(ABOVE, "adaptive", "null"), _col(ABOVE_REFERENCE, "adaptive", "null")
    )
    assert u == pytest.approx(20954.0)
    assert p3 == pytest.approx(1.27e-145, rel=1e-2)
    assert p3 <= study.ALPHA / 3


def test_the_window_narrowed_and_the_price_ratio_is_bounded() -> None:
    """The registration's own branch 1, as arithmetic on the two closed windows.

    The gate now has a price on both axes at this scale, so their ratio is
    bounded: even the most capacity-favourable pairing of the two windows puts
    the width price at ~2.9x the depth price, and the least favourable at ~7.5x.
    A report that quotes a single ratio, or claims the two prices are comparable,
    fails here.
    """
    assert _mean(BELOW, "adaptive", "null") < 0, "the bisection cell does not open the gate"
    assert _mean(BELOW_REFERENCE, "adaptive", "null") > 0, "2.87x is no longer below the gate"
    assert WINDOW_LOW < WINDOW_HIGH and DEPTH_CLOSED < DEPTH_OPEN
    low_ratio = WINDOW_LOW / DEPTH_OPEN
    high_ratio = WINDOW_HIGH / DEPTH_CLOSED
    assert low_ratio == pytest.approx(2.87, abs=0.01)
    assert high_ratio == pytest.approx(7.46, abs=0.01)
    # The sharpest form needs no window: the deepest measured cell has the gate
    # open at 23% *fewer* parameters than the width dose that does not open it.
    assert 67_632 / WINDOW_LOW == pytest.approx(0.768, abs=0.001)
    assert _mean("depth9-dose-576.json", "adaptive", "null") < 0


def test_the_declared_prediction_missed_and_the_state_it_landed_in() -> None:
    """The pool term peaked at 2.87x; the gate opened anyway.

    This is the pin that matters most, because the honest record is that the
    registration's mechanism was refuted - not that it was confirmed by the gate
    happening to open. A rewrite that reverses the declared direction, or that
    quietly drops the prediction because the branch passed, fails here.
    """
    new = _mean(BELOW, "random", "null")
    reference = _mean(BELOW_REFERENCE, "random", "null")
    assert new == pytest.approx(0.231778, abs=1e-6)
    assert reference == pytest.approx(0.305267, abs=1e-6)
    assert new > 0, "the pool term is no longer positive at the bisection"
    assert new < reference, "the pool term is not smaller, so the miss is not the recorded one"
    _, smaller_p = study._rank_sum_one_sided(
        _col(BELOW, "random", "null"), _col(BELOW_REFERENCE, "random", "null")
    )
    assert smaller_p == pytest.approx(1.35e-04, rel=1e-2)
    # The declared direction is still on the record as declared.
    assert study.DOSE3_POOL_GROWS is True


def test_the_gate_opens_in_two_stages() -> None:
    """Below the onset the edge crosses first; above it, only the pool term moves."""
    ladder = (
        "null-control-depth3-384.json",
        "width-d3-equal-576.json",
        BELOW_REFERENCE,
        BELOW,
        "width-d3-dose2-576.json",
    )
    edges = [_mean(name, "adaptive", "random") for name in ladder]
    pools = [_mean(name, "random", "null") for name in ladder]
    assert edges == pytest.approx(
        [-0.102151, -0.130727, -0.243801, -0.359770, -0.310646], abs=1e-6
    )
    assert pools == pytest.approx(
        [0.106794, 0.148365, 0.305267, 0.231778, -0.585988], abs=1e-6
    )
    # The edge is at its maximum exactly at the rung where the gate opens, and
    # the pool term is not negative there.
    assert edges[3] == min(edges), "the bisection is not the edge's ladder maximum"
    assert pools[3] > 0 and pools[4] < 0, "the pool term does not cross after the gate opens"
    # Above the onset the roles are reversed: the edge is flat, the pool moves.
    # Read in the decline direction (the *deeper* dose smaller), which is how the
    # report states it, with the reverse asserted alongside so neither reading can
    # be quoted as if it were the other.
    _, edge_moves = study._rank_sum_one_sided(
        _col(ABOVE, "adaptive", "random"), _col(ABOVE_REFERENCE, "adaptive", "random")
    )
    _, edge_reversed = study._rank_sum_one_sided(
        _col(ABOVE_REFERENCE, "adaptive", "random"), _col(ABOVE, "adaptive", "random")
    )
    _, pool_moves = study._rank_sum_one_sided(
        _col(ABOVE, "random", "null"), _col(ABOVE_REFERENCE, "random", "null")
    )
    assert edge_moves == pytest.approx(0.199, abs=1e-2), "the above-gate edge is not flat"
    assert edge_reversed == pytest.approx(0.801, abs=1e-2)
    assert pool_moves == pytest.approx(4.6e-119, rel=1e-2)
    assert _mean(ABOVE, "adaptive", "random") - _mean(
        ABOVE_REFERENCE, "adaptive", "random"
    ) == pytest.approx(-0.004685, abs=1e-6)
    assert _mean(ABOVE, "random", "null") - _mean(
        ABOVE_REFERENCE, "random", "null"
    ) == pytest.approx(-0.510940, abs=1e-6)


def test_the_two_stage_reading_is_reported_as_descriptive() -> None:
    """The mechanism reading is not the registered prediction - the report says so."""
    report = " ".join((ROOT / "docs" / "RESEARCH_REPORT.md").read_text().split())
    assert "the *mechanism* reading above is descriptive, not a pre-registered test" in report
    assert "the registered prediction was the one that missed" in report


def test_the_reports_outcome_table_is_the_recorded_cells() -> None:
    rows = [
        line
        for line in (ROOT / "docs" / "RESEARCH_REPORT.md").read_text().splitlines()
        if line.startswith("| depth 3, width 68 (5.66x) |")
        or line.startswith("| depth 4, width 68 (7.52x) |")
    ]
    assert rows == [
        "| depth 3, width 68 (5.66x) | 173,672 | 389/576 | `-0.127992` | `1.31e-17` | **yes** |",
        "| depth 4, width 68 (7.52x) | 230,520 | 575/576 | `-0.725298` | `2.33e-171` | **yes** |",
    ], rows


def test_the_report_quotes_the_recorded_decomposition() -> None:
    report = " ".join((ROOT / "docs" / "RESEARCH_REPORT.md").read_text().split())
    for fragment in (
        "revision `698a866`, 20 shards per cell",
        "the window narrows to (2.87x, 5.66x)",
        "`U = 20954.0`, p `1.27e-145`",
        "`-0.073489`, rank-sum p `1.35e-04`",
        "**the same gate costs between 2.9x and 7.5x more",
        "**23% fewer**",
        "`-0.359770`",
        "`+0.231778`",
        "`-0.585988`",
        "p `4.6e-119`",
    ):
        assert fragment in report, fragment
    # And the section's own heading carries the RAN marker, so a reader cannot
    # mistake a run for a registration.
    assert "### 5.60 Pre-registration: the width bisection" in report
    assert "(RAN - summaries `width-d3-dose3-576.json`" in report
