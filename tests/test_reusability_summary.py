"""The committed section 5.16 summary stays machine-checked.

Pins the versioned record produced by ``scripts/reusability_study.py summary``:
the quoted statistics must recompute exactly from the stored per-seed losses,
the built-in checks must hold, and the registered verdict must keep its label -
including the threshold resolution of section 5.16.1, so a later reader cannot
silently swap the declared bar for the looser sibling one.
"""

from __future__ import annotations

import json
import math

import pytest

from act.evolution.figures_data import summaries_dir
from act.evolution.study import _binomial_survival, _gate_min_wins

SUMMARY = "reusability-depth4-384.json"
# The numbers quoted in the report's section 5.16.1 table:
# comparison -> (wins, mean, p, registered pass, sibling pass)
QUOTED = {
    "transport_adaptive_vs_transport_random": (212, -0.101588, 2.32e-02, False, True),
    "transport_adaptive_vs_transport_fixed": (211, -0.086552, 2.94e-02, False, False),
}
DISCOVERY_QUOTED = {
    ("adaptive", "random"): (259, -0.1036, 3.53e-12),
    ("adaptive", "fixed"): (216, -0.0174, 8.18e-03),
}
DECLARED_PER_PRIMARY_ALPHA = 0.0025
SIBLING_PER_PRIMARY_ALPHA = 0.025


def _load() -> dict:
    path = summaries_dir() / SUMMARY
    if not path.exists():
        pytest.fail(f"missing committed study summary: {path}")
    return json.loads(path.read_text())


def test_reusability_summary_recomputes_from_per_seed() -> None:
    summary = _load()
    assert summary["record_type"] == "reusability_study"
    assert summary["protocol"] == "discover-then-fresh-init-transport"
    assert summary["seeds"] == [row["seed"] for row in summary["per_seed"]]
    assert summary["seeds"] == list(range(81000, 81384))
    for name, quoted in QUOTED.items():
        treatment, control = name.split("_vs_")
        treatment = treatment.removeprefix("transport_")
        control = control.removeprefix("transport_")
        comparison = summary["comparisons"][name]
        deltas = [
            row["transport_losses"][treatment] - row["transport_losses"][control]
            for row in summary["per_seed"]
        ]
        wins = sum(delta < 0 for delta in deltas)
        mean = sum(deltas) / len(deltas)
        p = _binomial_survival(wins, len(deltas))
        assert comparison["wins"] == wins == quoted[0]
        assert math.isclose(comparison["mean_delta"], mean, rel_tol=0, abs_tol=1e-12)
        assert math.isclose(mean, quoted[1], rel_tol=0, abs_tol=5e-7)
        assert math.isclose(comparison["binomial_p"], p, rel_tol=1e-6, abs_tol=0)
        assert math.isclose(p, quoted[2], rel_tol=5e-3)
        assert comparison["passes_bonferroni"] is quoted[3]
        assert comparison["passes_sibling_bonferroni"] is quoted[4]


def test_reusability_summary_thresholds_are_the_declared_ones() -> None:
    """Section 5.16.1: the registered bar is the declared 0.0025, not 0.025."""
    summary = _load()
    assert summary["bonferroni_alpha"] == DECLARED_PER_PRIMARY_ALPHA
    assert summary["sibling_bonferroni_alpha"] == SIBLING_PER_PRIMARY_ALPHA
    for comparison in summary["comparisons"].values():
        assert comparison["min_wins_bonferroni"] == _gate_min_wins(
            comparison["n"], DECLARED_PER_PRIMARY_ALPHA
        )
        assert comparison["min_wins_sibling_bonferroni"] == _gate_min_wins(
            comparison["n"], SIBLING_PER_PRIMARY_ALPHA
        )
        # The bar and the p-threshold are the same test, stated two ways.
        assert (comparison["wins"] >= comparison["min_wins_bonferroni"]) == (
            comparison["binomial_p"] <= DECLARED_PER_PRIMARY_ALPHA
        )
        assert comparison["passes_bonferroni"] == (
            comparison["wins"] >= comparison["min_wins_bonferroni"]
        )
    # The knife edge that made the resolution load-bearing: primary 1 sits
    # exactly on the sibling bar and below the declared one.
    primary_1 = summary["comparisons"]["transport_adaptive_vs_transport_random"]
    assert primary_1["wins"] == primary_1["min_wins_sibling_bonferroni"] == 212
    assert primary_1["min_wins_bonferroni"] == 220


def test_reusability_summary_built_in_checks_and_labels() -> None:
    summary = _load()
    checks = summary["built_in_checks"]
    assert all(checks.values())
    assert all(isinstance(value, bool) for value in checks.values())
    validity = summary["transport_validity"]
    assert validity["exact_reproductions"] == validity["n"] == 384
    assert validity["max_abs_delta"] == validity["mean_abs_delta"] == 0.0
    assert summary["identity_deltas"]["transport_fixed_minus_discovery_fixed_mean"] == 0.0
    assert summary["outcome"].startswith("branch (c) both FAIL")
    assert "branch (b)" in summary["outcome_under_sibling_bar"]
    # Both primaries fail at the *declared* bar in the registered reading.
    assert all(
        not comparison["passes_bonferroni"]
        for comparison in summary["comparisons"].values()
    )
    seeds = set(summary["seeds"])
    assert len(seeds) == 384
    for other in ("null-control-depth4-384.json", "gate3-probe-loss-384.json"):
        assert seeds.isdisjoint(
            set(json.loads((summaries_dir() / other).read_text())["seeds"])
        )


def test_reusability_summary_arm_means_and_replicated_ordering() -> None:
    summary = _load()
    n = len(summary["per_seed"])
    for arm, mean in summary["transport_arm_means"].items():
        recomputed = sum(row["transport_losses"][arm] for row in summary["per_seed"]) / n
        assert math.isclose(mean, recomputed, rel_tol=0, abs_tol=1e-12)
    for arm, mean in summary["discovery_arm_means"].items():
        recomputed = sum(row["discovery_losses"][arm] for row in summary["per_seed"]) / n
        assert math.isclose(mean, recomputed, rel_tol=0, abs_tol=1e-12)
    # The discovery half replicates section 5.10's depth-4 ordering out of
    # sample; the transport keeps the means but not the per-seed consistency.
    assert (
        summary["discovery_arm_means"]["adaptive"]
        < summary["discovery_arm_means"]["fixed"]
        < summary["discovery_arm_means"]["random"]
    )
    assert (
        summary["transport_arm_means"]["adaptive"]
        < summary["transport_arm_means"]["fixed"]
        < summary["transport_arm_means"]["random"]
    )
    for (treatment, control), quoted in DISCOVERY_QUOTED.items():
        deltas = [
            row["discovery_losses"][treatment] - row["discovery_losses"][control]
            for row in summary["per_seed"]
        ]
        wins = sum(delta < 0 for delta in deltas)
        mean = sum(deltas) / len(deltas)
        p = _binomial_survival(wins, len(deltas))
        assert wins == quoted[0]
        assert math.isclose(mean, quoted[1], rel_tol=0, abs_tol=5e-5)
        assert math.isclose(p, quoted[2], rel_tol=5e-3)
        assert wins >= _gate_min_wins(len(deltas), SIBLING_PER_PRIMARY_ALPHA)
