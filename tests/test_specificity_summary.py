"""The committed section 5.20 summaries stay machine-checked.

Pins the versioned records produced by ``scripts/specificity_study.py summary``:
the quoted statistics must recompute exactly from the stored per-seed losses,
the built-in checks must hold, and the registered verdict must keep its branch
label - here branch (a), which directs the research programme toward controller
work and away from the section-52 scale-up. Re-reading that label after the fact
is what this test exists to prevent.

It also pins section 5.20.1's amendment: the seeds whose two arms coincide must
be *exact ties* (so the ambiguity can only bias against primary 1), and the
primary must pass under the declared reading with the tie included.
"""

from __future__ import annotations

import json

import pytest

from act.evolution.figures_data import summaries_dir
from act.evolution.study import _binomial_survival, _gate_min_wins

PER_PRIMARY_ALPHA = 0.0125
DECLARED_BAR = 215
# Report section 5.20 table: profile -> comparison -> (wins, mean, p, passes)
QUOTED = {
    "specificity-depth2-384.json": {
        "A_own_vs_R_own": (243, -0.051537, 1.09e-07, True),
        "A_own_vs_F_own": (264, -0.072870, 7.64e-14, True),
        "R_own_vs_F_own": (213, -0.021333, 1.81e-02, False),
    },
    "specificity-depth3-384.json": {
        "A_own_vs_R_own": (248, -0.081657, 5.86e-09, True),
        "A_own_vs_F_own": (265, -0.082889, 3.43e-14, True),
        "R_own_vs_F_own": (197, -0.001231, 3.23e-01, False),
    },
}
SEED_BLOCKS = {
    "specificity-depth2-384.json": (89000, 89384),
    "specificity-depth3-384.json": (91000, 91384),
}
STUDY_IDS = {
    "specificity-depth2-384.json": "study_f10d1b1cd3812bc1",
    "specificity-depth3-384.json": "study_b6455e54f8801fd8",
}


def _load(name: str) -> dict:
    path = summaries_dir() / name
    if not path.exists():
        pytest.fail(f"missing committed study summary: {path}")
    return json.loads(path.read_text())


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_specificity_summary_recomputes_from_per_seed(name: str) -> None:
    summary = _load(name)
    assert summary["record_type"] == "specificity_study"
    assert summary["protocol"] == "selection-specificity-under-matched-initialization"
    assert summary["study_id"] == STUDY_IDS[name]
    assert summary["per_primary_alpha"] == PER_PRIMARY_ALPHA
    assert summary["declared_bar"] == DECLARED_BAR
    start, stop = SEED_BLOCKS[name]
    assert summary["seeds"] == list(range(start, stop))
    assert summary["seeds"] == [row["seed"] for row in summary["per_seed"]]

    for comparison_name, quoted in QUOTED[name].items():
        wins, mean, p, passes = quoted
        treatment, control = comparison_name.split("_vs_")
        deltas = [
            row["transport_losses"][treatment] - row["transport_losses"][control]
            for row in summary["per_seed"]
        ]
        assert sum(delta < 0 for delta in deltas) == wins
        assert sum(deltas) / len(deltas) == pytest.approx(mean, abs=5e-6)
        assert _binomial_survival(wins, len(deltas)) == pytest.approx(p, rel=1e-2)
        assert summary["comparisons"][comparison_name]["passes"] is passes
        # The declared bar is the exact binomial bar, not a restated constant.
        assert _gate_min_wins(len(deltas), PER_PRIMARY_ALPHA) == summary["comparisons"][
            comparison_name
        ]["min_wins"]


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_specificity_built_in_checks_hold(name: str) -> None:
    summary = _load(name)
    assert all(summary["built_in_checks"].values()), summary["built_in_checks"]


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_coincident_arms_are_exact_ties_and_rare(name: str) -> None:
    """Section 5.20.1: a chance coincidence may only ever be a non-win.

    Depth 2's topology space is small enough that two independent draws can land
    on the same genome; the amendment to the guard is legitimate only because
    such a seed is an exact tie (and so cannot inflate primary 1), and because
    the primary passes with the tie counted as a non-win.
    """
    summary = _load(name)
    coincident = [
        row
        for row in summary["per_seed"]
        if row["transport_architecture_hashes"]["A_own"]
        == row["transport_architecture_hashes"]["R_own"]
    ]
    assert summary["coincident_arm_topology_seeds"] == len(coincident)
    assert len(coincident) <= max(1, 0.01 * len(summary["per_seed"]))
    # A coincident seed must be an exact tie: same genome on the same own
    # initialization can only produce an identical loss, never a win.
    for row in coincident:
        assert row["transport_losses"]["A_own"] == row["transport_losses"]["R_own"]


def test_specificity_registered_branch_is_a_selection_specific() -> None:
    """Branch (a): primary 1 and primary 2 pass at BOTH depths.

    That is the registered reading, and it is the one whose consequence (redirect
    to controller research, deprioritise the section-52 scale-up) is quoted. This
    test fails if either depth stops passing, i.e. if the recorded branch no
    longer matches the numbers it was read from.
    """
    for name in QUOTED:
        comparisons = _load(name)["comparisons"]
        assert comparisons["A_own_vs_R_own"]["passes"] is True, name
        assert comparisons["A_own_vs_F_own"]["passes"] is True, name
