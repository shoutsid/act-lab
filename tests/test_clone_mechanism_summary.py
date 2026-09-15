"""Pins for the section 5.27 stored records (cloned body vs copied router row).

Section 5.26 inverted the mid-training penalty by cloning the layer's busiest
survivor, and declared its mechanism unresolved: cloning gives a new expert a
trained *body* AND copies the survivor's *router row*. Section 5.27 separated them
with a third setting, ``clone_body``, which is identical except the grown expert
keeps a freshly drawn row. Verdict branch (a): **both contribute, the body
~3/4** - the body alone already beats the unmutated arm, and the row adds a real
second term on top.

Pinned here, because the study text leans on all three and none is re-derived
anywhere else:

1. **All three settings pair exactly.** Same final genome, bit-identical frozen
   reference, and an identical fixed arm in every seed - which is what makes
   `B - R` the body's contribution and `C - B` the row's, on one genome.
2. **The branch (a) split**, including that each contribution is significant on
   its own. A test asserting only "cloning helps" would let the decomposition rot.
"""

from __future__ import annotations

import json

import pytest

from act.evolution.figures_data import summaries_dir
from act.evolution.study import _binomial_survival, _gate_min_wins

PER_PRIMARY_ALPHA = 0.05 / 4
DECLARED_BAR = 215

QUOTED: dict[str, dict] = {
    "clone-mechanism-depth2-384.json": {
        "study_id": "study_3bcc9e86b575b7a6",
        "seeds": (105000, 105384),
        "comparisons": {
            # name -> (wins, mean, p, passes)
            "B_vs_R": (295, -0.074389, 3.40e-27, True),
            "C_vs_B": (242, -0.026445, 1.89e-07, True),
            "C_vs_R": (320, -0.100834, 2.38e-42, True),
            "R_own_vs_R": (224, -0.026199, 6.38e-04, True),
        },
        "means": {
            "C": 14.192764208341638,
            "B": 14.219208889330426,
            "F": 14.239759052793184,
            "R_own": 14.267398788904151,
            "R": 14.293597950289646,
        },
        "body_gain": -0.074389,
        "row_gain": -0.026445,
        "clone_gain": -0.100834,
    },
    "clone-mechanism-depth3-384.json": {
        "study_id": "study_872d33bdd99496a8",
        "seeds": (106000, 106384),
        "comparisons": {
            "B_vs_R": (300, -0.084398, 7.00e-30, True),
            "C_vs_B": (238, -0.021955, 1.54e-06, True),
            "C_vs_R": (312, -0.106353, 5.17e-37, True),
            "R_own_vs_R": (230, -0.038128, 6.18e-05, True),
        },
        "means": {
            "C": 13.347045584892234,
            "B": 13.369000274688005,
            "F": 13.394676009193063,
            "R_own": 13.415270478154222,
            "R": 13.453398437549671,
        },
        "body_gain": -0.084398,
        "row_gain": -0.021955,
        "clone_gain": -0.106353,
    },
}


def _load(name: str) -> dict:
    path = summaries_dir() / name
    if not path.exists():
        pytest.fail(f"missing committed study summary: {path}")
    return json.loads(path.read_text())


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_section_5_27_numbers_recompute_from_per_seed(name: str) -> None:
    quoted = QUOTED[name]
    summary = _load(name)
    assert summary["record_type"] == "clone_mechanism_study"
    assert summary["study_id"] == quoted["study_id"]
    assert summary["per_primary_alpha"] == PER_PRIMARY_ALPHA
    assert summary["declared_bar"] == DECLARED_BAR
    start, stop = quoted["seeds"]
    assert summary["seeds"] == list(range(start, stop))
    assert summary["seeds"] == [row["seed"] for row in summary["per_seed"]]
    assert all(summary["built_in_checks"].values()), summary["built_in_checks"]

    for comparison_name, (wins, mean, p, passes) in quoted["comparisons"].items():
        first, second = comparison_name.split("_vs_")
        deltas = [
            row["losses"][first] - row["losses"][second] for row in summary["per_seed"]
        ]
        stored = summary["comparisons"][comparison_name]
        assert sum(delta < 0 for delta in deltas) == wins, comparison_name
        assert sum(deltas) / len(deltas) == pytest.approx(mean, abs=5e-6)
        assert _binomial_survival(wins, len(deltas)) == pytest.approx(p, rel=1e-2)
        # The stored block must agree with its own per_seed rows.
        assert stored["wins"] == wins, comparison_name
        assert stored["mean_delta"] == pytest.approx(mean, abs=5e-6)
        assert stored["n"] == len(deltas)
        assert stored["passes"] is passes
        assert stored["min_wins"] == _gate_min_wins(len(deltas), PER_PRIMARY_ALPHA)

    for label, value in quoted["means"].items():
        assert summary["loss_means"][label] == pytest.approx(value, abs=1e-6)
    for key in ("body_gain", "row_gain", "clone_gain"):
        assert summary[key] == pytest.approx(quoted[key], abs=5e-6), key


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_section_5_27_all_three_settings_pair_exactly(name: str) -> None:
    summary = _load(name)
    rows = summary["per_seed"]
    assert all(
        all(count == 3 for count in row["mutations"].values()) for row in rows
    ), "every arm must use the full mutation budget"
    assert all(row["same_final_genome"] for row in rows), (
        "the three settings ended on different final genomes; B-R and C-B would no "
        "longer be contributions measured on one genome"
    )
    assert all(row["same_own_reference"] for row in rows)
    assert all(row["fixed_arm_identical_across_settings"] for row in rows)
    for label in ("R", "B", "C"):
        assert all(
            row["losses"][f"{label}_own"] == row["losses"]["R_own"] for row in rows
        ), f"{label}_own must be the same frozen reference"


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_section_5_27_branch_a_both_terms_significant(name: str) -> None:
    """Branch (a): the body AND the routing row each buy something."""
    summary = _load(name)
    comparisons = summary["comparisons"]
    assert comparisons["B_vs_R"]["passes"] is True, "P1 must pass for branch (a)"
    assert comparisons["C_vs_B"]["passes"] is True, "P2 must pass for branch (a)"
    assert summary["body_gain"] < 0, "the trained body must help"
    assert summary["row_gain"] < 0, "the copied router row must help"


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_section_5_27_the_body_is_the_larger_term(name: str) -> None:
    """The report's headline split: the body is ~3/4 of the total clone gain."""
    summary = _load(name)
    share = summary["body_gain"] / summary["clone_gain"]
    assert 0.70 < share < 0.82, (
        "the quoted 'body is about three quarters' split; a change here means the "
        "report's consequence paragraph is stale"
    )
    assert summary["body_gain"] < summary["row_gain"], "the body dominates the row"
    # The two terms partition the total exactly: C-R = (B-R) + (C-B).
    assert summary["clone_gain"] == pytest.approx(
        summary["body_gain"] + summary["row_gain"], abs=1e-9
    )


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_section_5_27_both_terms_each_beat_the_unmutated_arm(name: str) -> None:
    """The four-deep ordering the report prints: C < B < F < own < R."""
    means = _load(name)["loss_means"]
    assert means["C"] < means["B"], "adding the row beats the body clone alone"
    assert means["B"] < means["F"], "the body clone alone beats the unmutated arm"
    assert means["F"] < means["R_own"], "the frozen retrain beats the fresh draw"
    assert means["R_own"] < means["R"]
