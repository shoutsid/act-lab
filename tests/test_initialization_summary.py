"""The committed section 5.17 summary stays machine-checked.

Pins the versioned record produced by ``scripts/initialization_study.py
summary``: the quoted statistics must recompute exactly from the stored
per-seed losses, the built-in checks must hold (including the run-time check
that the DECLARED bar equals the exact binomial bar - the defect section
5.16.1 had to repair after the fact), and the registered verdict must keep its
label, so the branch cannot be re-read after the fact.
"""

from __future__ import annotations

import json
import math

import pytest

from act.evolution.figures_data import summaries_dir
from act.evolution.study import _binomial_survival, _gate_min_wins

SUMMARY = "initialization-depth4-384.json"
DECLARED_ALPHA = 0.025
DECLARED_BAR = 212
# Report section 5.17.2 table: comparison -> (wins, mean, p, passes)
QUOTED = {
    "B_adaptive_vs_C_adaptive": (215, -0.080576, 1.08e-02, True),
    "B_adaptive_vs_B_fixed": (274, -0.112303, 1.46e-17, True),
    "B_random_vs_C_random": (189, 0.042417, 6.40e-01, False),
    "C_adaptive_vs_B_fixed": (198, -0.031727, 2.87e-01, False),
}
SUBGROUP_QUOTED = {
    # (subgroup, comparison) -> (wins, n, mean)
    ("pure_prune", "B_adaptive_vs_C_adaptive"): (123, 213, -0.1018),
    ("grown", "B_adaptive_vs_C_adaptive"): (92, 171, -0.0542),
    ("pure_prune", "B_adaptive_vs_B_fixed"): (163, 213, -0.1425),
    ("grown", "B_adaptive_vs_B_fixed"): (111, 171, -0.0746),
}


def _load() -> dict:
    path = summaries_dir() / SUMMARY
    if not path.exists():
        pytest.fail(f"missing committed study summary: {path}")
    return json.loads(path.read_text())


def test_initialization_summary_recomputes_from_per_seed() -> None:
    summary = _load()
    assert summary["record_type"] == "initialization_study"
    assert summary["protocol"] == "discover-then-fresh-and-rewound-init-transport"
    assert summary["seeds"] == [row["seed"] for row in summary["per_seed"]]
    assert summary["seeds"] == list(range(82000, 82384))
    for name, quoted in QUOTED.items():
        treatment, control = name.split("_vs_")
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
        assert math.isclose(mean, quoted[1], rel_tol=0, abs_tol=5e-6)
        assert math.isclose(comparison["binomial_p"], p, rel_tol=1e-6, abs_tol=0)
        assert math.isclose(p, quoted[2], rel_tol=5e-3)
        assert comparison["passes"] is quoted[3]
        assert comparison["min_wins"] == _gate_min_wins(len(deltas), DECLARED_ALPHA)
        assert comparison["bar_and_p_agree"] is True
        assert comparison["passes"] == (wins >= DECLARED_BAR)


def test_initialization_summary_declared_bar_is_the_exact_one() -> None:
    """Section 5.17 asserts the declaration instead of restating it."""
    summary = _load()
    assert summary["per_primary_alpha"] == DECLARED_ALPHA
    assert summary["declared_bar"] == DECLARED_BAR == _gate_min_wins(384, DECLARED_ALPHA)
    assert summary["built_in_checks"]["declared_bar_matches_exact_binomial"] is True
    assert summary["built_in_checks"]["registered_block_matches_declared_n"] is True


def test_initialization_summary_built_in_checks_and_branch() -> None:
    summary = _load()
    assert all(summary["built_in_checks"].values())
    assert all(isinstance(value, bool) for value in summary["built_in_checks"].values())
    assert summary["outcome"].startswith("branch (d) both PASS")
    # The two primaries are the pass column of the table; the control and the
    # fresh-init replication of section 5.16 are the fail column.
    assert summary["comparisons"]["B_adaptive_vs_C_adaptive"]["passes"] is True
    assert summary["comparisons"]["B_adaptive_vs_B_fixed"]["passes"] is True
    assert summary["comparisons"]["B_random_vs_C_random"]["passes"] is False
    assert summary["comparisons"]["C_adaptive_vs_B_fixed"]["passes"] is False
    seeds = set(summary["seeds"])
    assert len(seeds) == 384
    for other in (
        "null-control-depth4-384.json",
        "reusability-depth4-384.json",
        "gate3-probe-loss-384.json",
    ):
        assert seeds.isdisjoint(
            set(json.loads((summaries_dir() / other).read_text())["seeds"])
        )


def test_initialization_summary_subgroups_pin_the_mechanism() -> None:
    """P1 concentrates where rewinding is total; P2 holds in both subgroups."""
    summary = _load()
    pure = [
        row for row in summary["per_seed"] if row["grown_experts"]["B_adaptive"] == 0
    ]
    grown = [
        row for row in summary["per_seed"] if row["grown_experts"]["B_adaptive"] > 0
    ]
    assert (len(pure), len(grown)) == (213, 171)
    for (subgroup, name), quoted in SUBGROUP_QUOTED.items():
        rows = pure if subgroup == "pure_prune" else grown
        treatment, control = name.split("_vs_")
        deltas = [
            row["transport_losses"][treatment] - row["transport_losses"][control]
            for row in rows
        ]
        wins = sum(delta < 0 for delta in deltas)
        mean = sum(deltas) / len(deltas)
        assert (wins, len(deltas)) == (quoted[0], quoted[1])
        assert math.isclose(mean, quoted[2], rel_tol=0, abs_tol=5e-5)
    # P1 passes on the pure-prune subset and fails where experts were grown;
    # P2 passes on both.
    assert 123 >= _gate_min_wins(213, DECLARED_ALPHA) == 122
    assert 92 < _gate_min_wins(171, DECLARED_ALPHA)
    assert 163 >= _gate_min_wins(213, DECLARED_ALPHA)
    assert 111 >= _gate_min_wins(171, DECLARED_ALPHA)
    inventory = summary["section_55_inventory"]
    assert inventory["unchanged_adaptive_topologies"] == 0
    assert inventory["grown_experts_total"] == 201


def test_initialization_summary_unregistered_observation_is_labelled() -> None:
    """The trained-weights secondary is recorded, not silently promoted."""
    summary = _load()
    trained = summary["trained_weight_value"]
    assert trained["wins"] == 113  # the clean retrain wins the other 271
    assert math.isclose(trained["mean_delta"], 0.096090, rel_tol=0, abs_tol=5e-6)
    assert trained["mean_delta"] > 0  # the discovery run is the *worse* one
