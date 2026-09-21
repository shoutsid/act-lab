"""Section 5.48's committed width-arm summaries stay machine-checked.

The two cells' statistics must recompute exactly from the stored per-seed
triples, the built-in checks must hold, the registered branch must keep its
label, and the blocks must be fresh - the report calls them fresh, which a
reused block would make false without changing any statistic. The design pins
(equal-parameter width, the bar, the branches) live in `test_width_arm.py`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from act.evolution.figures_data import summaries_dir
from act.evolution.study import _binomial_survival, _gate_min_wins

CELLS = {
    "width-d3-equal": {
        "summary": "width-d3-equal-576.json",
        "block": (76000, 76576),
        "depth": 3,
        "hidden_size": 28,
        "adaptive_vs_null": (252, +0.017638, 9.99e-01),
        "adaptive_vs_random": (410, -0.130727, 4.15e-25),
        "null_vs_random": (453, -0.148365, 1.21e-45),
        "creates_value": False,
        "arm_means": {"adaptive": 14.97705, "null": 14.959412, "random": 15.107777},
    },
    "width-d4-equal": {
        "summary": "width-d4-equal-576.json",
        "block": (77000, 77576),
        "depth": 4,
        "hidden_size": 28,
        "adaptive_vs_null": (316, -0.032649, 1.0922e-02),
        "adaptive_vs_random": (393, -0.120658, 6.40e-19),
        "null_vs_random": (387, -0.088010, 5.63e-17),
        "creates_value": True,
        "arm_means": {"adaptive": 13.650294, "null": 13.682942, "random": 13.770952},
    },
}
BRANCH = "P2 only: the onset is depth-like at this width"


def _load(name: str) -> dict:
    path = summaries_dir() / name
    if not path.exists():
        pytest.fail(f"missing committed study summary: {path}")
    return json.loads(path.read_text())


@pytest.mark.parametrize("profile", sorted(CELLS))
def test_the_quoted_statistics_recompute_from_the_per_seed_triples(profile: str) -> None:
    cell = CELLS[profile]
    payload = _load(cell["summary"])
    assert payload["code_revision"] == "38f3921"  # the pre-registration revision
    assert payload["report_section"] == "5.48"
    seeds = payload["seeds"]
    assert (seeds[0], seeds[-1] + 1) == cell["block"]
    assert len(seeds) == 576
    assert payload["config"]["depth"] == cell["depth"]
    assert payload["config"]["hidden_size"] == cell["hidden_size"]
    assert payload["config"]["arms"] == "random,adaptive,null"
    assert payload["independent_arm_rng"] is True

    per_seed = payload["per_seed"]
    assert len(per_seed) == 576
    deltas_by_arm = {
        "adaptive_vs_null": ("adaptive", "null"),
        "adaptive_vs_random": ("adaptive", "random"),
        "null_vs_random": ("null", "random"),
    }
    for name, (left, right) in deltas_by_arm.items():
        deltas = [row["losses"][left] - row["losses"][right] for row in per_seed]
        wins = sum(delta < 0 for delta in deltas)
        mean = sum(deltas) / len(deltas)
        quoted_wins, quoted_mean, quoted_p = cell[name]
        assert wins == quoted_wins
        assert mean == pytest.approx(quoted_mean, abs=5e-7)
        assert _binomial_survival(wins, len(deltas)) == pytest.approx(quoted_p, rel=1e-3)
        recorded = payload["comparisons"][name]
        assert recorded["wins"] == quoted_wins
        assert recorded["binomial_p"] == pytest.approx(quoted_p, rel=1e-3)

    for arm, expected in cell["arm_means"].items():
        assert payload["arm_means"][arm] == pytest.approx(expected, abs=5e-7)


@pytest.mark.parametrize("profile", sorted(CELLS))
def test_the_registered_branch_and_its_conjunction(profile: str) -> None:
    """A cell creates value only if wins clear the bar AND the mean is negative."""
    cell = CELLS[profile]
    payload = _load(cell["summary"])
    comparison = payload["comparisons"]["adaptive_vs_null"]
    bar = _gate_min_wins(576, 0.025)
    assert bar == 313
    assert comparison["min_wins_bonferroni"] == bar
    creates_value = comparison["wins"] >= bar and comparison["mean_delta"] < 0
    assert creates_value is cell["creates_value"]
    assert comparison["passes_bonferroni"] is cell["creates_value"]
    # The outcome names *this arm's* per-cell primary, and must keep its two
    # distinct values. It used to carry the 5.9 vocabulary instead, whose two
    # primaries are null-vs-random and adaptive-vs-null *of one cell*, so it
    # labelled this arm's below-onset cell "branch 2 (1 PASS / 2 FAIL)" - a pass
    # - where the registration records a failure. (Post-run correction,
    # 2026-09-21: the labels were rewritten at the source so the arm's own
    # declaration decides them; no measured number moved.)
    expected = (
        "cell creates value (5.48's per-cell primary"
        if cell["creates_value"]
        else "cell does not create value (5.48's per-cell primary"
    )
    assert payload["outcome"].startswith(expected)
    assert payload["bonferroni_alpha"] == pytest.approx(0.025)


def test_the_registered_outcome_is_the_one_the_verdict_prints() -> None:
    """The two cells' committed numbers map to the registered branch 3.

    Recomputes the branch from the committed summaries rather than trusting the
    prose: P2 only is what the arm registered for these numbers.
    """
    cells = {profile: _load(CELLS[profile]["summary"]) for profile in CELLS}
    creates = {}
    for profile, payload in cells.items():
        comparison = payload["comparisons"]["adaptive_vs_null"]
        creates[profile] = (
            comparison["wins"] >= comparison["min_wins_bonferroni"]
            and comparison["mean_delta"] < 0
        )
    assert creates == {"width-d3-equal": False, "width-d4-equal": True}
    branches = {
        (True, True): "the onset is a capacity onset",
        (True, False): "widening creates value below the onset and destroys it above",
        (False, True): BRANCH,
        (False, False): "width is harmful at both depths here",
    }
    branch = branches[(creates["width-d3-equal"], creates["width-d4-equal"])]
    assert branch == BRANCH
    report = (Path(__file__).resolve().parent.parent / "docs" / "RESEARCH_REPORT.md").read_text()
    assert branch.split(":")[0] in report, "the report no longer states the registered branch"


def test_the_built_in_checks_hold_in_every_seed() -> None:
    for profile, cell in CELLS.items():
        payload = _load(cell["summary"])
        checks = payload["built_in_checks"]
        assert checks == {
            "mutations_random_3": True,
            "mutations_adaptive_3": True,
            "mutations_null_0": True,
            "null_search_flops_zero": True,
        }
        assert all(row["losses"]["null"] > 0 for row in payload["per_seed"])


def test_the_width_blocks_are_fresh_against_every_committed_summary() -> None:
    """The report calls 76000-76575 and 77000-77575 fresh blocks; check it."""
    used: set[int] = set()
    for path in sorted(summaries_dir().glob("*.json")):
        if path.name in {cell["summary"] for cell in CELLS.values()}:
            continue
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):  # some files are panels, not records
            continue
        seeds = payload.get("seeds")
        if isinstance(seeds, list):
            used |= {seed for seed in seeds if isinstance(seed, int)}
    for cell in CELLS.values():
        block = set(range(*cell["block"]))
        assert not (block & used), f"block overlaps an earlier study: {sorted(block & used)[:5]}"
