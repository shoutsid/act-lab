"""Section 5.62's recorded result: the crossing under the other candidate policy.

One cell, one gene changed from 5.60's crossing dose. **P1 and both declared
predictions passed**: `387/576` against the bar of `313`, ratio `1.64`
(edge-dominated again with a positive pool), and the value `0.001510` from the
prefix reading - indistinguishable by rank-sum (p `0.472` / `0.528`).

The finding is that the *level* is policy-robust while the *mixture* is not: the
edge moves `+0.035826` and the pool term `-0.034316` across the policy change,
opposite directions and nearly equal magnitudes, cancelling in the contrast the
family reports. So the crossing is robust and the two-term split is a description
of the balance rather than an invariant - a caveat every prefix-policy
decomposition above inherits.

These pins keep both halves: the agreement, and the disagreement underneath it.
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
CELL = "width-d3-dose3-spread-576.json"
REFERENCE = "width-d3-dose3-576.json"


def _record(name: str) -> dict:
    return json.loads((SUMMARIES / name).read_text())


def _col(name: str, left: str, right: str) -> list[float]:
    return [row["losses"][left] - row["losses"][right] for row in _record(name)["per_seed"]]


def _mean(name: str, left: str, right: str) -> float:
    values = _col(name, left, right)
    return sum(values) / len(values)


def test_the_summary_is_the_frozen_revision_and_the_registered_arm() -> None:
    summary = _record(CELL)
    assert summary["profile"] == study.DOSE3_SPREAD_PROFILE
    assert summary["report_section"] == "5.62"
    assert summary["code_revision"] == "969a251"
    assert len(summary["seeds"]) == 576
    assert summary["seeds"][0] == 100000 and summary["seeds"][-1] == 100575
    assert summary["bonferroni_alpha"] == pytest.approx(study.ALPHA / 2)
    assert summary["comparisons"]["adaptive_vs_null"]["min_wins_bonferroni"] == 313


def test_the_primary_and_both_declared_predictions_passed() -> None:
    cell = _record(CELL)["comparisons"]["adaptive_vs_null"]
    assert cell["wins"] == 387
    assert cell["mean_delta"] == pytest.approx(-0.126482, abs=1e-6)
    assert cell["binomial_p"] == pytest.approx(5.63e-17, rel=1e-2)
    assert cell["passes_bonferroni"] is True

    edge = _mean(CELL, "adaptive", "random")
    pool = _mean(CELL, "random", "null")
    assert edge == pytest.approx(-0.323944, abs=1e-6)
    assert pool == pytest.approx(+0.197462, abs=1e-6)
    assert pool > 0 and abs(edge) / pool == pytest.approx(1.64, abs=0.01)
    assert abs(edge) / pool > study.DOSE3_SPREAD_RATIO_FLOOR

    reference = _mean(REFERENCE, "adaptive", "null")
    assert abs(cell["mean_delta"] - reference) <= study.DOSE3_SPREAD_VALUE_BAND
    assert abs(cell["mean_delta"] - reference) == pytest.approx(0.001510, abs=1e-6)


def test_the_value_agrees_and_the_components_do_not() -> None:
    """The pin that keeps the finding honest: indistinguishable totals, real parts.

    A rewrite that reports only the agreement makes the decomposition look
    invariant; one that reports only the moving parts makes the crossing look
    policy-specific. Both rank-sums are asserted, in both directions.
    """
    _, value_one_way = study._rank_sum_one_sided(
        _col(CELL, "adaptive", "null"), _col(REFERENCE, "adaptive", "null")
    )
    _, value_other_way = study._rank_sum_one_sided(
        _col(REFERENCE, "adaptive", "null"), _col(CELL, "adaptive", "null")
    )
    assert value_one_way == pytest.approx(0.472, abs=1e-2)
    assert value_other_way == pytest.approx(0.528, abs=1e-2)
    assert value_one_way > study.ALPHA / 2 and value_other_way > study.ALPHA / 2

    # The components move significantly, in opposite directions.
    _, edge_moves = study._rank_sum_one_sided(
        _col(REFERENCE, "adaptive", "random"), _col(CELL, "adaptive", "random")
    )
    _, pool_moves = study._rank_sum_one_sided(
        _col(CELL, "random", "null"), _col(REFERENCE, "random", "null")
    )
    assert edge_moves == pytest.approx(0.032, abs=1e-2), "the prefix edge is no longer larger"
    assert pool_moves == pytest.approx(0.041, abs=1e-2), "the spread pool is no longer smaller"
    edge_shift = _mean(CELL, "adaptive", "random") - _mean(REFERENCE, "adaptive", "random")
    pool_shift = _mean(CELL, "random", "null") - _mean(REFERENCE, "random", "null")
    assert edge_shift == pytest.approx(+0.035826, abs=1e-6)
    assert pool_shift == pytest.approx(-0.034316, abs=1e-6)
    # They cancel: the value shift is two orders smaller than either part.
    assert abs(edge_shift + pool_shift) == pytest.approx(0.001510, abs=1e-6)
    assert abs(edge_shift + pool_shift) < 0.05 * min(abs(edge_shift), abs(pool_shift))


def test_the_absolute_levels_move_far_more_than_the_contrast() -> None:
    """The policy is worth ~0.35 nats in absolute loss and ~0.0015 in the contrast."""
    spread = _record(CELL)["arm_means"]
    prefix = _record(REFERENCE)["arm_means"]
    assert spread["adaptive"] < spread["null"] < spread["random"], spread
    assert prefix["adaptive"] < prefix["null"] < prefix["random"], prefix
    shifts = {arm: spread[arm] - prefix[arm] for arm in ("adaptive", "null", "random")}
    # The policy costs this much absolute loss, per arm - 0.31 to 0.35, so the
    # report quotes it as a range rather than a single number.
    assert all(0.30 < shift < 0.36 for shift in shifts.values()), shifts
    assert min(shifts.values()) == pytest.approx(0.3098, abs=1e-3)
    assert max(shifts.values()) == pytest.approx(0.3457, abs=1e-3)
    # The right comparison is shift against shift: the policy moves the arms by
    # ~0.33 nats while moving the contrast the family reports by 0.0015.
    value_shift = abs(_mean(CELL, "adaptive", "null") - _mean(REFERENCE, "adaptive", "null"))
    assert value_shift < 0.01 * min(shifts.values()), (value_shift, shifts)


def test_the_report_quotes_the_recorded_cell_and_the_mixture() -> None:
    report = " ".join((ROOT / "docs" / "RESEARCH_REPORT.md").read_text().split())
    for fragment in (
        "revision `969a251`, 20 shards; committed summary `width-d3-dose3-spread-576.json`",
        "the crossing is policy-robust",
        "`387/576` against the bar of `313`, mean `-0.126482`",
        "the ratio is **1.64**",
        "`0.001510` from the prefix cell's reading",
        "prefix larger at p `0.032`",
        "spread smaller at p `0.041`",
        "a description of the balance rather than an\ninvariant of the mechanism" if False else "a description of the balance rather than an invariant of the mechanism",
        "`0.31`-`0.35` nats in absolute loss (per arm) and `0.0015` nats in what the gate measures",
    ):
        assert fragment in report, fragment
    assert "(RAN - summary `width-d3-dose3-spread-576.json`)" in report
