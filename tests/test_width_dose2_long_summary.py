"""Section 5.51's committed long-budget summary stays machine-checked.

The registered outcome (branch 1: the collapse relieved by the budget and value
creation surviving it) and the two *descriptive* readings the report labels as
such - the collapse halving rather than vanishing, and the controller's own
edge shrinking to nearly nothing with the budget - all recompute here from the
committed summaries, so the labels cannot drift apart from the numbers. The
design pins live in `test_width_dose2_long.py`.
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

CELL = {
    "summary": "width-d4-dose2-long-576.json",
    "block": (87000, 87576),
    "study_id": "study_055561a11315cf41",
    "adaptive_vs_null": (576, -0.714872, 4.043e-174),
    "adaptive_vs_random": (335, -0.006596, 5.160e-05),
    "null_vs_random": (0, +0.708276, 1.0),
    "arm_means": {"adaptive": 3.833337, "null": 4.548209, "random": 3.839933},
}
REFERENCE = "width-d4-dose2-576.json"


def _load(name: str) -> dict:
    path = summaries_dir() / name
    if not path.exists():
        pytest.fail(f"missing committed study summary: {path}")
    return json.loads(path.read_text())


def _deltas(name: str, left: str, right: str) -> list[float]:
    return [
        row["losses"][left] - row["losses"][right] for row in _load(name)["per_seed"]
    ]


def _mean_delta(name: str, left: str, right: str) -> float:
    deltas = _deltas(name, left, right)
    return sum(deltas) / len(deltas)


def test_the_quoted_statistics_recompute_from_the_per_seed_triples() -> None:
    payload = _load(CELL["summary"])
    assert payload["code_revision"] == "a590a7f"  # the long-budget pre-registration
    assert payload["report_section"] == "5.51"
    assert payload["study_id"] == CELL["study_id"]
    assert (payload["seeds"][0], payload["seeds"][-1] + 1) == CELL["block"]
    assert len(payload["seeds"]) == 576
    assert payload["config"]["steps"] == 192
    assert payload["config"]["mutation_interval"] == 48

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
        quoted_wins, quoted_mean, quoted_p = CELL[name]
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
    for arm, expected in CELL["arm_means"].items():
        assert payload["arm_means"][arm] == pytest.approx(expected, abs=5e-7)
    assert payload["outcome"].startswith("cell creates value (5.51's per-cell primary")
    assert payload["alpha"] == 0.05
    assert payload["bonferroni_alpha"] == pytest.approx(0.025, abs=1e-12)
    for comparison in payload["comparisons"].values():
        assert comparison["min_wins_bonferroni"] == 313
        assert comparison["passes_bonferroni"] == (
            comparison["binomial_p"] <= payload["bonferroni_alpha"]
        )


def test_the_registered_branch_is_p1_and_p2_pass() -> None:
    """The branch the registration fixed in advance, recomputed from both cells.

    P1 is the non-paired rank-sum against the committed 48-step cell - the same
    test 5.50 introduced - and P2 is 5.50's own per-cell conjunction. Only the
    conjunction is a per-cell statement; P1 is a claim about two cells and can
    only be read through the registered test.
    """
    payload = _load(CELL["summary"])
    creates = (
        payload["comparisons"]["adaptive_vs_null"]["wins"] >= 313
        and payload["comparisons"]["adaptive_vs_null"]["mean_delta"] < 0
    )
    collapsed_long = _deltas(CELL["summary"], "null", "random")
    collapsed_reference = _deltas(REFERENCE, "null", "random")
    u, p = study._rank_sum_one_sided(collapsed_long, collapsed_reference)
    assert u == pytest.approx(2925.0)
    assert p == pytest.approx(1.741e-183, rel=1e-2)
    assert p <= 0.025
    assert creates
    # The report is wrapped prose, so the branch sentence can straddle lines:
    # normalise whitespace before looking for it, or the pin would pass or fail
    # on line width rather than on the recorded reading.
    branch = "5.50's crossing is not a fixed-budget artifact"
    report = " ".join((ROOT / "docs" / "RESEARCH_REPORT.md").read_text().split())
    assert branch in report


def test_the_collapse_is_relieved_by_half_not_removed() -> None:
    """The arm says 'the collapse was the budget's' - and the numbers bound it.

    The pool effect falls from `+1.574` to `+0.708` - to 45% of its 48-step
    value, not to zero - and the unmutated arm is *still* the worst arm in the
    longer cell (`null 4.548` against `adaptive 3.833` / `random 3.840`).
    Reading branch 1 as 'the budget explains the collapse' would overclaim;
    reading it as 'the budget takes more than half of it and the substrate
    keeps the same shape' is what the numbers say.
    """
    longer = _mean_delta(CELL["summary"], "null", "random")
    reference = _mean_delta(REFERENCE, "null", "random")
    assert longer == pytest.approx(+0.708276, abs=5e-7)
    assert reference == pytest.approx(+1.573907, abs=5e-7)
    assert longer < 0.5 * reference  # relieved by more than half, not removed
    means = _load(CELL["summary"])["arm_means"]
    assert means["null"] > means["adaptive"]
    assert means["null"] > means["random"]


def test_the_controller_edge_shrinks_with_the_budget() -> None:
    """The descriptive reading that sharpens 5.50's decomposition.

    adaptive-vs-random is `-0.159` at 48 steps and `-0.0066` at 192 - a 24x
    shrink, with the reverse rank-sum at p `2.1e-45`: the edge still exists
    (335/576 against the bar, p `5.2e-5`) but it is no longer what the value
    creation rests on. Above the gate, at both budgets, what the do-nothing
    comparison measures is the mutational pool - and the controller's share of
    it is budget-dependent in the *opposite* direction from the collapse's.
    """
    longer = _mean_delta(CELL["summary"], "adaptive", "random")
    reference = _mean_delta(REFERENCE, "adaptive", "random")
    assert longer == pytest.approx(-0.006596, abs=5e-7)
    assert reference == pytest.approx(-0.159059, abs=5e-7)
    assert abs(reference) > 20 * abs(longer)
    _, p_shrunk = study._rank_sum_one_sided(
        _deltas(REFERENCE, "adaptive", "random"),
        _deltas(CELL["summary"], "adaptive", "random"),
    )
    assert p_shrunk == pytest.approx(2.137e-45, rel=1e-2)


def test_the_long_budget_arm_has_exactly_one_cell() -> None:
    """The absolute comparison is budget-internal and stays that way.

    4x training moves the same width-96 arms from `15.1`-`16.9` to
    `3.8`-`4.5`, but there is no 192-step width-24 control anywhere, so no
    cross-width absolute claim can be made at this budget - and adding a second
    long-budget cell would silently change what the arm's branch table means.
    """
    long_cells = sorted(p for p in study.PROFILES if p.endswith("-long"))
    assert long_cells == ["width-d4-dose2-long"]
    longer = _load(CELL["summary"])["arm_means"]
    reference = _load(REFERENCE)["arm_means"]
    for arm in ("adaptive", "null", "random"):
        assert longer[arm] < reference[arm] - 5.0
