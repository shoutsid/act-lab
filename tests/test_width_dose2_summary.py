"""Section 5.50's committed second-dose summaries stay machine-checked.

The registered outcome (branch 1, P1+P2 PASS, and the third primary passing)
and the three *descriptive* readings the report labels as such - the
controller's edge flat above the gate while the pool effect explodes, the
below-gate edge growing while its pool effect flips sign, and the absolute
levels degrading - all recompute here from the committed summaries, so the
labels cannot drift apart from the numbers. The design pins live in
`test_width_dose2.py`.
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
from act.evolution.study import _gate_min_wins  # noqa: E402

CELLS = {
    "width-d3-dose2": {
        "summary": "width-d3-dose2-576.json",
        "block": (80000, 80576),
        "depth": 3,
        "hidden_size": 96,
        "study_id": "study_f8aef48c2ff1f023",
        "adaptive_vs_null": (573, -0.896634245, 1.288e-166),
        "adaptive_vs_random": (455, -0.310646023, 8.742e-47),
        "null_vs_random": (49, +0.585988223, 1.0),
        "arm_means": {"adaptive": 20.22363, "null": 21.120264, "random": 20.534276},
    },
    "width-d4-dose2": {
        "summary": "width-d4-dose2-576.json",
        "block": (84000, 84576),
        "depth": 4,
        "hidden_size": 96,
        "study_id": "study_ae8289de6f8d34d6",
        "adaptive_vs_null": (576, -1.732966621, 4.043e-174),
        "adaptive_vs_random": (431, -0.159059297, 3.247e-34),
        "null_vs_random": (0, +1.573907324, 1.0),
        "arm_means": {"adaptive": 15.127319, "null": 16.860286, "random": 15.286378},
    },
}
# The two ladders the report's decomposition table is stated on, as
# (adaptive-vs-null, adaptive-vs-random, random-vs-null) means.
DEPTH4_LADDER = {
    "null-control-depth4-prefix-384.json": (-0.037147240, -0.107219576, +0.070072336),
    "width-d4-equal-576.json": (-0.032648873, -0.120658487, +0.088009614),
    "width-d4-dose-576.json": (-0.209673296, -0.194518461, -0.015154835),
    "width-d4-dose2-576.json": (-1.732966621, -0.159059297, -1.573907324),
}
DEPTH3_LADDER = {
    "null-control-depth3-384.json": (+0.004643268, -0.102151196, +0.106794464),
    "width-d3-equal-576.json": (+0.017638496, -0.130726605, +0.148365101),
    "width-d3-dose-576.json": (+0.061465862, -0.243800676, +0.305266538),
    "width-d3-dose2-576.json": (-0.896634245, -0.310646023, -0.585988223),
}


def _load(name: str) -> dict:
    path = summaries_dir() / name
    if not path.exists():
        pytest.fail(f"missing committed study summary: {path}")
    return json.loads(path.read_text())


def _deltas(name: str, left: str, right: str) -> list[float]:
    return [
        row["losses"][left] - row["losses"][right]
        for row in _load(name)["per_seed"]
    ]


def _mean_delta(name: str, left: str, right: str) -> float:
    deltas = _deltas(name, left, right)
    return sum(deltas) / len(deltas)


def _stored(name: str, left: str, right: str) -> float:
    """The committed comparison's mean, which is always stored as left - right.

    The report's decomposition table quotes random-vs-null, the summary stores
    null-vs-random: the same number, opposite sign. The two conventions have
    already cost this program two incidents (5.39.1, 5.40), so this test states
    which one each assertion is in rather than relying on the reader.
    """
    return _load(name)["comparisons"][f"{left}_vs_{right}"]["mean_delta"]


@pytest.mark.parametrize("profile", sorted(CELLS))
def test_the_quoted_statistics_recompute_from_the_per_seed_triples(profile: str) -> None:
    cell = CELLS[profile]
    payload = _load(cell["summary"])
    assert payload["code_revision"] == "05efa7b"  # the second-dose pre-registration
    assert payload["report_section"] == "5.50"
    assert payload["study_id"] == cell["study_id"]
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
    # The 5.50 cells carry the arm's own per-cell label, not the 5.9 vocabulary
    # (whose pair of primaries is a different pair).
    assert payload["outcome"].startswith("cell creates value (5.50's per-cell primary")


def test_the_summaries_level_is_the_three_primary_one() -> None:
    """Three primaries: the bars and the recorded level are alpha/3's, not alpha/2's.

    The first verdict run of this arm applied alpha/6 here and printed bar 318
    where the registration said 315 - stricter than declared, and still the
    defect 5.16.1 adjudicates. The level is stored once and recomputed here.
    """
    assert _gate_min_wins(576, 0.05 / 3) == 315
    for cell in CELLS.values():
        payload = _load(cell["summary"])
        assert payload["alpha"] == 0.05
        assert payload["bonferroni_alpha"] == pytest.approx(0.05 / 3, abs=1e-12)
        for comparison in payload["comparisons"].values():
            assert comparison["min_wins_bonferroni"] == 315
            assert comparison["passes_bonferroni"] == (
                comparison["binomial_p"] <= payload["bonferroni_alpha"]
            )


def test_the_registered_branch_is_p1_and_p2_pass() -> None:
    """Both cells create value: the branch the registration fixed in advance."""
    creates = {}
    for profile, cell in CELLS.items():
        comparison = _load(cell["summary"])["comparisons"]["adaptive_vs_null"]
        creates[profile] = (
            comparison["wins"] >= 315 and comparison["mean_delta"] < 0
        )
    assert creates == {"width-d3-dose2": True, "width-d4-dose2": True}
    branch = "capacity-driven above a threshold between 3.8x and 14.8x"
    report = (ROOT / "docs" / "RESEARCH_REPORT.md").read_text()
    assert branch in report


def test_the_third_primary_is_a_perfect_separation() -> None:
    """P3 as registered: the 14.8x cell's deltas all sit below the 3.8x rung's.

    U = 0 is the strongest form the test can take - every one of the 576 wider
    seeds' adaptive-vs-null deltas is smaller than every one of the 576 rung's.
    """
    wider = _deltas("width-d4-dose2-576.json", "adaptive", "null")
    rung = _deltas("width-d4-dose-576.json", "adaptive", "null")
    u, p = study._rank_sum_one_sided(wider, rung)
    assert u == 0.0
    assert p == pytest.approx(4.797e-190, rel=1e-2)
    assert p <= 0.05 / 3
    assert max(wider) < min(rung)


def test_the_controller_edge_is_flat_above_the_gate() -> None:
    """The descriptive reading that qualifies 5.49's amplifier language.

    adaptive-vs-random across the four depth-4 rungs: -0.107, -0.121, -0.195,
    -0.159. The w96-vs-w48 rank-sum cannot see an increase (p 0.98 in that
    direction) and the reverse direction reads p 0.016 - so if anything the
    edge is marginally *smaller* at 14.8x. The contrast the primary measures
    grew by an order of magnitude; the controller's own edge did not grow at
    all.
    """
    means = [recorded[1] for recorded in DEPTH4_LADDER.values()]
    for name, expected in DEPTH4_LADDER.items():
        assert _stored(name, "adaptive", "random") == pytest.approx(expected[1], abs=5e-7)
        assert _mean_delta(name, "adaptive", "random") == pytest.approx(
            expected[1], abs=5e-7
        )
    assert means[2] - means[1] == pytest.approx(-0.0739, abs=2e-3)
    assert means[3] - means[2] == pytest.approx(+0.0355, abs=2e-3)
    wider = _deltas("width-d4-dose2-576.json", "adaptive", "random")
    rung = _deltas("width-d4-dose-576.json", "adaptive", "random")
    _, p_bigger = study._rank_sum_one_sided(wider, rung)
    _, p_smaller = study._rank_sum_one_sided(rung, wider)
    assert p_bigger == pytest.approx(0.984, abs=5e-3)
    assert p_smaller == pytest.approx(0.0159, abs=5e-3)


def test_the_pool_effect_is_what_scales_above_the_gate() -> None:
    """random-vs-null: +0.070, +0.088, -0.015, -1.574 - the do-nothing collapse."""
    for name, expected in DEPTH4_LADDER.items():
        assert _mean_delta(name, "random", "null") == pytest.approx(expected[2], abs=5e-7)
        assert _stored(name, "null", "random") == pytest.approx(-expected[2], abs=5e-7)
    wider = _deltas("width-d4-dose2-576.json", "random", "null")
    rung = _deltas("width-d4-dose-576.json", "random", "null")
    u, p = study._rank_sum_one_sided(wider, rung)
    assert u == pytest.approx(52.0)
    assert p == pytest.approx(6.289e-190, rel=1e-2)
    # The absolute levels the report calls the arm's own limit: the wide cells
    # are far worse than the committed rung, so the contrast is not a quality
    # gain.
    wide = _load("width-d4-dose2-576.json")["arm_means"]
    rung_means = _load("null-control-depth4-prefix-384.json")["arm_means"]
    assert wide["adaptive"] > rung_means["adaptive"] + 2.0
    assert wide["null"] > rung_means["null"] + 4.0


def test_below_the_gate_the_edge_grows_while_the_pool_flips_sign() -> None:
    """The depth-3 ladder: -0.102, -0.131, -0.244, -0.311 and a pool sign flip."""
    edge = [recorded[1] for recorded in DEPTH3_LADDER.values()]
    pool = [recorded[2] for recorded in DEPTH3_LADDER.values()]
    for name, expected in DEPTH3_LADDER.items():
        assert _mean_delta(name, "adaptive", "random") == pytest.approx(
            expected[1], abs=5e-7
        )
        assert _mean_delta(name, "random", "null") == pytest.approx(
            expected[2], abs=5e-7
        )
    assert all(b < a for a, b in zip(edge, edge[1:])), edge
    assert pool[0] > 0 and pool[1] > 0 and pool[2] > 0 and pool[3] < 0
    wider = _deltas("width-d3-dose2-576.json", "adaptive", "random")
    rung = _deltas("width-d3-dose-576.json", "adaptive", "random")
    _, p = study._rank_sum_one_sided(wider, rung)
    assert p == pytest.approx(1.374e-3, rel=2e-2)
