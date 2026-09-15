"""The committed real-text study summaries stay machine-checked.

These tests pin the versioned records produced by
``scripts/real_text_summaries.py build`` (report sections 5.11 and 5.13):
the quoted statistics must recompute exactly from the stored per-seed
payloads, the built-in checks must hold, and the stored verdicts must keep
their labels. Like the toy records, the files are self-contained (per-seed
losses embedded), so the gitignored run outputs are not load-bearing for
the report's numbers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from act.evolution.figures_data import summaries_dir
from act.evolution.registry import _stable_hash
from act.evolution.study import _binomial_survival, _gate_min_wins

RECORDS = {
    "theta-sweep-9008-9017.json": "real_text_theta_sweep",
    "heuristic-9008-9017.json": "real_text_heuristic_study",
}
# Quoted in the report's section 5.11 / 5.13 outcome paragraphs:
# comparison -> (wins, mean, p).
QUOTED = {
    "theta-sweep-9008-9017.json": {
        "t002": (0, +0.002675, 1.0),
        "t05": (0, +0.023611, 1.0),
        "t01": (0, +0.071282, 1.0),
    },
    "heuristic-9008-9017.json": {
        "heuristic_vs_null": (0, +0.121796, 1.0),
    },
}
# The monotone dose-response is the scientific residue of section 5.11.
DOSE_RESPONSE = (("t002", "t05"), ("t05", "t01"))


def _load(name: str) -> dict:
    path = summaries_dir() / name
    if not path.exists():
        pytest.fail(f"missing committed study summary: {path}")
    return json.loads(path.read_text())


@pytest.mark.parametrize("filename,record_type", sorted(RECORDS.items()))
def test_real_text_summary_schema_and_hashes(filename: str, record_type: str) -> None:
    summary = _load(filename)
    assert summary["record_type"] == record_type
    assert summary["seeds"] == list(range(9008, 9018))
    assert summary["seeds"] == [row["seed"] for row in summary["per_seed"]]
    assert summary["report_section"] in {"5.11", "5.13"}
    assert summary["code_revision"] in {"9bd9323", "0ef29de"}
    # The hash/id fields recompute from the stored payload.
    expected_hash = _stable_hash(
        {
            "config": summary["config"],
            "seeds": summary["seeds"],
            "per_seed": summary["per_seed"],
            "alpha": summary["alpha"],
        }
    )
    assert summary["study_hash"] == expected_hash
    assert summary["study_id"] == f"study_{expected_hash}"
    assert summary["config_hash"] == _stable_hash(summary["config"])
    # The block is the theta sweep's fresh block: disjoint from every toy
    # null-control block and from the spent 9000-9011 diagnostic block.
    for other in ("null-control-depth1-384.json", "null-control-depth4-384.json"):
        toy = json.loads((summaries_dir() / other).read_text())
        assert set(summary["seeds"]).isdisjoint(set(toy["seeds"]))


@pytest.mark.parametrize("filename,record_type", sorted(RECORDS.items()))
def test_real_text_summary_recomputes_from_per_seed(
    filename: str, record_type: str
) -> None:
    summary = _load(filename)
    alpha = summary["bonferroni_alpha"]
    for name, quoted in QUOTED[filename].items():
        treatment, control = name.split("_vs_") if "_vs_" in name else (name, "null")
        comparison = summary["comparisons"][name]
        deltas = [
            row["losses"][treatment] - row["losses"][control]
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
        assert comparison["min_wins_bonferroni"] == _gate_min_wins(len(deltas), alpha)
        # Nothing cleared: the falsification labels must stay falsified.
        assert comparison["passes_bonferroni"] is False
    # Arm means recompute too.
    for arm, mean in summary["arm_means"].items():
        recomputed = sum(row["losses"][arm] for row in summary["per_seed"]) / len(
            summary["per_seed"]
        )
        assert math.isclose(mean, recomputed, rel_tol=0, abs_tol=1e-12)


def test_theta_sweep_dose_response_and_acceptance_rate() -> None:
    """Mean harm grows monotonically with the acceptance rate (section 5.11)."""
    summary = _load("theta-sweep-9008-9017.json")
    comparisons = summary["comparisons"]
    rates = {
        name: comparison["accept_rate"]
        for name, comparison in comparisons.items()
    }
    harms = {
        name: comparison["mean_delta"] for name, comparison in comparisons.items()
    }
    for low, high in DOSE_RESPONSE:
        assert rates[low] < rates[high]
        assert harms[low] < harms[high]
    # Accept counts recompute from the per-seed accept tallies.
    for name, comparison in comparisons.items():
        accepts = sum(row["accepted"][name] for row in summary["per_seed"])
        assert comparison["accepts"] == accepts
        assert comparison["accept_rate"] == pytest.approx(
            accepts / (len(summary["per_seed"]) * 3)
        )
    assert summary["family_verdict"].startswith("no theta clears")


def test_heuristic_summary_targeting_footprint() -> None:
    """The grow-layer histogram matches the report's section 5.13 record."""
    summary = _load("heuristic-9008-9017.json")
    assert summary["grow_layer_histogram"] == {
        "L0": 4, "L3": 2, "L4": 5, "L5": 10, "L6": 3, "L7": 6,
    }
    # Every event is a heuristic grow; every seed used its full budget.
    for row in summary["per_seed"]:
        assert row["mutations"] == 3
        assert len(row["targeted_layers"]) == 3
        assert all(row["targeted_layers"])
    assert summary["outcome"].startswith("falsified (0/10 wins)")


@pytest.mark.parametrize("filename,record_type", sorted(RECORDS.items()))
def test_real_text_summary_built_in_checks_and_null_arm(
    filename: str, record_type: str
) -> None:
    summary = _load(filename)
    assert all(summary["built_in_checks"].values())
    # The null arm is the shared do-nothing baseline of both studies: zero
    # mutations, zero search FLOPs, identical values across the two records.
    for row in summary["per_seed"]:
        assert row["losses"]["null"] > 0
    assert "null" in summary["arms"]
