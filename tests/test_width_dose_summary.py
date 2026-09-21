"""Section 5.49's committed dose summaries stay machine-checked.

The registered outcome (branch 3, P2 only) and the two *descriptive* readings
the report labels as such - the below-onset average getting worse as width
grows, and the churn harm vanishing above the gate at 3.80x width - all
recompute here from the committed summaries, so the labels cannot drift apart
from the numbers. The design pins live in `test_width_arm.py`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from act.evolution.figures_data import summaries_dir
from act.evolution.study import _gate_min_wins

CELLS = {
    "width-d3-dose": {
        "summary": "width-d3-dose-576.json",
        "block": (78000, 78576),
        "depth": 3,
        "hidden_size": 48,
        "adaptive_vs_null": (235, +0.061466, 1.0),
        "adaptive_vs_random": (430, -0.243801, 9.631e-34),
        "null_vs_random": (494, -0.305267, 5.389e-73),
        "creates_value": False,
        "label": "cell does not create value (5.49's per-cell primary",
        "arm_means": {"adaptive": 20.506183, "null": 20.444717, "random": 20.749984},
    },
    "width-d4-dose": {
        "summary": "width-d4-dose-576.json",
        "block": (79000, 79576),
        "depth": 4,
        "hidden_size": 48,
        "adaptive_vs_null": (477, -0.209673, 1.358e-60),
        "adaptive_vs_random": (432, -0.194518, 1.085e-34),
        "null_vs_random": (266, +0.015155, 9.697e-01),
        "creates_value": True,
        "label": "cell creates value (5.49's per-cell primary",
        "arm_means": {"adaptive": 16.56971, "null": 16.779383, "random": 16.764228},
    },
}
# The committed width levels at each depth, in this arm's units - the axis the
# report's descriptive readings are stated on.
DEPTH3 = {
    "null-control-depth3-384.json": (188, +0.004643),
    "width-d3-equal-576.json": (252, +0.017638),
    "width-d3-dose-576.json": (235, +0.061466),
}
DEPTH4 = {
    "null-control-depth4-prefix-384.json": (225, -0.037147),
    "width-d4-equal-576.json": (316, -0.032649),
    "width-d4-dose-576.json": (477, -0.209673),
}


def _load(name: str) -> dict:
    path = summaries_dir() / name
    if not path.exists():
        pytest.fail(f"missing committed study summary: {path}")
    return json.loads(path.read_text())


@pytest.mark.parametrize("profile", sorted(CELLS))
def test_the_quoted_statistics_recompute_from_the_per_seed_triples(profile: str) -> None:
    cell = CELLS[profile]
    payload = _load(cell["summary"])
    assert payload["code_revision"] == "e5dcbf6"  # the dose pre-registration
    assert payload["report_section"] == "5.49"
    assert (payload["seeds"][0], payload["seeds"][-1] + 1) == cell["block"]
    assert len(payload["seeds"]) == 576
    assert payload["config"]["depth"] == cell["depth"]
    assert payload["config"]["hidden_size"] == cell["hidden_size"]

    pairs = {
        "adaptive_vs_null": ("adaptive", "null"),
        "adaptive_vs_random": ("adaptive", "random"),
        "null_vs_random": ("null", "random"),
    }
    for name, (left, right) in pairs.items():
        deltas = [
            row["losses"][left] - row["losses"][right] for row in payload["per_seed"]
        ]
        wins = sum(delta < 0 for delta in deltas)
        mean = sum(deltas) / len(deltas)
        quoted_wins, quoted_mean, quoted_p = cell[name]
        assert wins == quoted_wins
        assert mean == pytest.approx(quoted_mean, abs=5e-7)
        assert payload["comparisons"][name]["binomial_p"] == pytest.approx(
            quoted_p, rel=2e-3
        )

    assert payload["built_in_checks"] == {
        "mutations_random_3": True,
        "mutations_adaptive_3": True,
        "mutations_null_0": True,
        "null_search_flops_zero": True,
    }
    for arm, expected in cell["arm_means"].items():
        assert payload["arm_means"][arm] == pytest.approx(expected, abs=5e-7)
    # The 5.49 cells carry the arm's own per-cell label, not the 5.9 vocabulary
    # (whose pair of primaries is a different pair). Post-run correction
    # 2026-09-21; no measured number moved.
    assert payload["outcome"].startswith(cell["label"])
    assert payload["bonferroni_alpha"] == pytest.approx(0.025)


def test_the_registered_branch_is_p2_only() -> None:
    """The registered outcome, recomputed from both cells' comparisons."""
    creates = {}
    for profile, cell in CELLS.items():
        comparison = _load(cell["summary"])["comparisons"]["adaptive_vs_null"]
        assert comparison["min_wins_bonferroni"] == 313
        creates[profile] = (
            comparison["wins"] >= 313 and comparison["mean_delta"] < 0
        )
    assert creates == {"width-d3-dose": False, "width-d4-dose": True}
    branch = "depth stands even at 2.87x width"
    report = (Path(__file__).resolve().parent.parent / "docs" / "RESEARCH_REPORT.md").read_text()
    assert branch in report


def _means_by_width(summaries: dict) -> list[float]:
    means = []
    for name, (wins, mean) in summaries.items():
        recorded = _load(name)["comparisons"]["adaptive_vs_null"]
        assert recorded["wins"] == wins
        assert recorded["mean_delta"] == pytest.approx(mean, abs=5e-7)
        means.append(mean)
    return means


def test_the_below_onset_trend_is_monotone_against_width() -> None:
    """Below the gate the arm's average outcome weakens at every width step.

    Descriptive, not registered - but the report's "gets worse as width grows"
    sentence is exactly this, and it holds at all three levels.
    """
    means = _means_by_width(DEPTH3)
    assert all(b > a for a, b in zip(means, means[1:])), means
    assert means[-1] - means[0] == pytest.approx(0.0568, abs=1e-3)


def test_the_above_onset_dose_step_is_where_the_amplifier_lives() -> None:
    """Above the gate the +1.7% step is a wash and the 3.8x dose is not.

    The report says capacity is an amplifier; the numbers say the amplifier is
    the *dose*: `-0.0371 -> -0.0326` (a +0.0045 wash, within these cells'
    noise) and then `-0.0326 -> -0.2097` (a `-0.1770` step). A monotone-in-width
    claim would be false at this depth, which is why it is not made.
    """
    means = _means_by_width(DEPTH4)
    assert means[1] - means[0] == pytest.approx(+0.0045, abs=2e-3)
    assert means[2] - means[1] == pytest.approx(-0.1770, abs=2e-3)
    assert means[2] < means[0] - 0.1


def test_the_churn_harm_is_absent_only_at_the_dose() -> None:
    """The first cell in the program where churning does not pay.

    null-vs-random at depth 4 x width 48 is 266/576 (mean +0.0152) - the null
    arm fails its own bar - while both narrower cells in the same regime pass
    it with ~0.67-0.69 win rates. The report labels this as a width-dependent
    churn harm; this is the number that makes the label checkable.
    """
    dose = _load("width-d4-dose-576.json")["comparisons"]["null_vs_random"]
    assert dose["wins"] == 266
    assert dose["mean_delta"] > 0
    assert not dose["passes_bonferroni"]
    for narrower in ("null-control-depth4-prefix-384.json", "width-d4-equal-576.json"):
        comparison = _load(narrower)["comparisons"]["null_vs_random"]
        assert comparison["passes_bonferroni"]
        assert comparison["mean_delta"] < 0
        assert comparison["wins"] / comparison["n"] > dose["wins"] / dose["n"]


def test_the_dose_blocks_are_fresh_against_every_committed_summary() -> None:
    used: set[int] = set()
    names = {cell["summary"] for cell in CELLS.values()}
    for path in sorted(summaries_dir().glob("*.json")):
        if path.name in names:
            continue
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            continue
        seeds = payload.get("seeds")
        if isinstance(seeds, list):
            used |= {seed for seed in seeds if isinstance(seed, int)}
    for cell in CELLS.values():
        block = set(range(*cell["block"]))
        assert not (block & used), f"overlap: {sorted(block & used)[:5]}"


def test_the_dose_bar_is_the_same_one_the_other_arms_used() -> None:
    assert _gate_min_wins(576, 0.025) == 313
    assert math.isclose(313 / 576, 0.5434, abs_tol=5e-5)
