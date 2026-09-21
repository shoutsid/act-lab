"""Section 5.53's recorded result: the fourth branch, and every number behind it.

The design pins live in `test_experts_arm.py`; these read the two committed
summaries and the report against each other, so a quoted figure cannot drift
from the artifact it came from, and the five-cell table the branch rests on is
recomputed from the committed per-seed triples rather than trusted as prose.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import null_toy_study as study  # noqa: E402

from act.evolution.figures_data import summaries_dir  # noqa: E402

REPORT = ROOT / "docs" / "RESEARCH_REPORT.md"
CELLS = {
    "experts-d3-dose": {
        "summary": "experts-d3-dose-576.json",
        "block": (88176, 88752),
        "depth": 3,
        "adaptive_vs_null": (211, +0.059544, 1.0),
        "adaptive_vs_random": (391, -0.077625, 2.938e-18),
        "null_vs_random": (465, -0.137169, 9.030e-53),
        "arm_means": {
            "adaptive": 13.447387288428015,
            "null": 13.387843482610252,
            "random": 13.525012504309416,
        },
    },
    "experts-d4-dose": {
        "summary": "experts-d4-dose-576.json",
        "block": (89384, 89960),
        "depth": 4,
        "adaptive_vs_null": (260, +0.019507, 9.913e-01),
        "adaptive_vs_random": (372, -0.075803, 1.212e-12),
        "null_vs_random": (412, -0.095310, 6.641e-26),
        "arm_means": {
            "adaptive": 12.557427870730558,
            "null": 12.537920881890589,
            "random": 12.633231261538135,
        },
    },
}
# The committed cells the two new ones are read against, in the report's table
# order, with the arm means the "best arm" column names.
TABLE = {
    "null-control-depth3-384.json": {
        "wins": 188,
        "mean": +0.004643,
        "best": "null",
        "means": {"adaptive": 13.4549, "null": 13.4503, "random": 13.5571},
    },
    "width-d3-equal-576.json": {
        "wins": 252,
        "mean": +0.017638,
        "best": "null",
        "means": {"adaptive": 14.9771, "null": 14.9594, "random": 15.1078},
    },
    "null-control-depth4-prefix-384.json": {
        "wins": 225,
        "mean": -0.037147,
        "best": "adaptive",
        "means": {"adaptive": 12.4413, "null": 12.4785, "random": 12.5485},
    },
}


def _load(name: str) -> dict:
    path = summaries_dir() / name
    if not path.exists():
        pytest.fail(f"missing committed study summary: {path}")
    return json.loads(path.read_text())


def _flat(text: str) -> str:
    """The report with runs of whitespace collapsed, so wrapped prose can be quoted."""
    return " ".join(text.split())


@pytest.fixture(scope="module")
def report() -> str:
    return _flat(REPORT.read_text())


def _deltas(summary: dict, left: str, right: str) -> list[float]:
    return [row["losses"][left] - row["losses"][right] for row in summary["per_seed"]]


@pytest.mark.parametrize("profile", sorted(CELLS))
def test_the_new_cells_are_the_pre_registered_ones(profile: str) -> None:
    cell = CELLS[profile]
    summary = _load(cell["summary"])
    assert summary["profile"] == profile
    assert summary["report_section"] == "5.53"
    # Frozen before the first shard ran: the registration commit.
    assert summary["code_revision"] == study.PROFILES[profile]["run_revision"] == "76d2134"
    assert (summary["seeds"][0], summary["seeds"][-1] + 1) == cell["block"]
    assert len(summary["seeds"]) == 576
    assert summary["config"]["depth"] == cell["depth"]
    assert summary["config"]["expert_count"] == 6
    assert summary["config"]["hidden_size"] == 24
    assert summary["config"]["active_experts"] == 2
    assert summary["bonferroni_alpha"] == pytest.approx(0.025)
    assert summary["built_in_checks"] == {
        "mutations_adaptive_3": True,
        "mutations_null_0": True,
        "mutations_random_3": True,
        "null_search_flops_zero": True,
    }


@pytest.mark.parametrize("profile", sorted(CELLS))
def test_the_quoted_statistics_recompute_from_the_per_seed_triples(profile: str) -> None:
    cell = CELLS[profile]
    summary = _load(cell["summary"])
    pairs = {
        "adaptive_vs_null": ("adaptive", "null"),
        "adaptive_vs_random": ("adaptive", "random"),
        "null_vs_random": ("null", "random"),
    }
    for name, (left, right) in pairs.items():
        deltas = _deltas(summary, left, right)
        wins = sum(delta < 0 for delta in deltas)
        mean = sum(deltas) / len(deltas)
        quoted_wins, quoted_mean, quoted_p = cell[name]
        assert wins == quoted_wins
        assert mean == pytest.approx(quoted_mean, abs=5e-7)
        assert summary["comparisons"][name]["wins"] == quoted_wins
        assert summary["comparisons"][name]["binomial_p"] == pytest.approx(
            quoted_p, rel=2e-3
        )
    for arm, expected in cell["arm_means"].items():
        assert summary["arm_means"][arm] == pytest.approx(expected, abs=5e-7)


@pytest.mark.parametrize("profile", sorted(CELLS))
def test_the_registered_branch_is_the_fourth_one(profile: str, report: str) -> None:
    """Both cells fail the conjunction, and the summary says so."""
    summary = _load(CELLS[profile]["summary"])
    comparison = summary["comparisons"]["adaptive_vs_null"]
    assert comparison["min_wins_bonferroni"] == 313
    creates = comparison["wins"] >= 313 and comparison["mean_delta"] < 0
    assert creates is False
    assert comparison["passes_bonferroni"] is False
    assert summary["outcome"].startswith("cell does not create value (5.53's per-cell primary")
    assert "Registered branch 4: both FAIL." in report
    assert "expert capacity is harmful at both depths" in report


def test_the_five_cell_table_recomputes_and_the_ordering_inverts(report: str) -> None:
    """The table and the inversion the branch rests on, from the records.

    At depth 4 the E=4 cell's best arm is the adaptive one and the E=6 cell's is
    the do-nothing one - the claim the report states in words and the branch
    turns on.
    """
    for name, expected in TABLE.items():
        summary = _load(name)
        deltas = _deltas(summary, "adaptive", "null")
        assert sum(delta < 0 for delta in deltas) == expected["wins"]
        assert sum(deltas) / len(deltas) == pytest.approx(expected["mean"], abs=5e-7)
        # The report's table quotes the two new cells' means to six places.
        assert f"`{expected['mean']:+.6f}`" in report
        assert {
            arm: round(value, 4) for arm, value in summary["arm_means"].items()
        } == expected["means"]
        best = min(summary["arm_means"], key=lambda arm: summary["arm_means"][arm])
        assert best == expected["best"]
        for value in (f"`{expected['mean']:+.6f}`",):
            assert value in report, f"the report does not quote {value}"
    # And the new cells' ordering: the do-nothing arm is best at both.
    for cell in CELLS.values():
        means = _load(cell["summary"])["arm_means"]
        assert min(means, key=lambda arm: means[arm]) == "null"
    assert "the ordering inverts - the do-nothing arm becomes the best one" in report
    assert "12.5379" in report and "12.5574" in report and "12.6332" in report
    assert "12.4413" in report and "12.5485" in report


def test_the_dose_rank_sums_are_recomputed_and_run_the_right_way(report: str) -> None:
    """The E=4 cell's deltas are stochastically *more negative* than the E=6 cell's.

    Both quoted p-values are recomputed here, and the direction is asserted: a
    dose that removed value creation must make the smaller-expert cell the more
    negative one.
    """
    for depth, treat, ref, quoted in (
        (3, "experts-d3-dose-576.json", "null-control-depth3-384.json", "8.0e-09"),
        (4, "experts-d4-dose-576.json", "null-control-depth4-prefix-384.json", "9.1e-09"),
    ):
        a = _deltas(_load(treat), "adaptive", "null")
        b = _deltas(_load(ref), "adaptive", "null")
        _, p = study._rank_sum_one_sided(b, a)
        assert p < 1e-8
        assert p == pytest.approx(float(quoted), rel=0.1)
        assert sum(b) / len(b) < sum(a) / len(a), f"depth {depth}: the direction moved"
    assert "stochastically *more negative*" in report


def test_the_report_scopes_the_reading_it_does_not_make(report: str) -> None:
    """The substrate-only scope is stated, so the arm is not read as a verdict."""
    assert "the expert count is a property of the *substrate* in this arm" in report
    assert "does not measure whether a search that grows its own experts does better" in report
