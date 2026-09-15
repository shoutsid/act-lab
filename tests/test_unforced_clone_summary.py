"""Pins for the section 5.28 stored records (clone-and-preserve, drawn mutations).

Sections 5.26 and 5.27 established that cloning the busiest survivor inverts the
mid-training penalty, but only on **forced** operations. Section 5.28 tested the
regime the pipeline was actually adopted from - ``forced_operation`` left None, so
the grow/prune mix is drawn - and the result is sharper than the forced blocks
could show: on this substrate the historical path leaves the run **worse than never
mutating** (significantly, at both depths), while cloning makes mutating **pay**.

Three things are pinned, none of them re-derived anywhere else:

1. **The regime really is drawn.** A block that quietly contained only grows would
   be a forced block in disguise and would not answer the question asked, so both
   operations are required to be present and their counts are checked.
2. **The pairing, which this design did not assume.** Unlike sections 5.26/5.27 a
   drawn prune can fall back to a load-based grow, so identical genomes could not
   be asserted - only recorded. It held in 384/384 both times, and that is checked.
3. **The sign flip.** `F < R` (mutating the historical way costs versus not
   mutating) and `C < F` (cloning makes mutating pay), each significant. A test
   asserting only "clone helps" would let the actionable half of the result rot.
"""

from __future__ import annotations

import json

import pytest

from act.evolution.figures_data import summaries_dir
from act.evolution.study import _binomial_survival, _gate_min_wins

PER_PRIMARY_ALPHA = 0.05 / 4
DECLARED_BAR = 215

QUOTED: dict[str, dict] = {
    "unforced-clone-depth2-384.json": {
        "study_id": "study_5ece35fb3fbc8724",
        "seeds": (107000, 107384),
        "comparisons": {
            # name -> (wins, mean, p, passes)
            "C_vs_R": (276, -0.071978, 2.26e-18, True),
            "C_own_vs_C": (170, +0.024255, 9.89e-01, False),
            "R_own_vs_R": (255, -0.047724, 6.13e-11, True),
            "C_own_vs_R_own": (0, 0.0, 1.0, False),
        },
        "means": {
            "C": 14.26301389808456,
            "C_own": 14.28726851195097,
            "R_own": 14.28726851195097,
            "F": 14.290546896557013,
            "R": 14.33499219827354,
        },
        "clone_gain": -0.071978,
        "penalty_fresh": -0.047724,
        "penalty_cloned": +0.024255,
        "operation_mix": {"grow": 622, "prune": 530},
        # secondary contrasts the report quotes
        "C_vs_F": (238, -0.0275),
        "F_vs_R": (251, -0.0444),
        "R_own_vs_F": (186, -0.0033),
    },
    "unforced-clone-depth3-384.json": {
        "study_id": "study_f8bc02ca0f048a79",
        "seeds": (108000, 108384),
        "comparisons": {
            "C_vs_R": (252, -0.065137, 4.63e-10, True),
            "C_own_vs_C": (178, +0.011344, 9.31e-01, False),
            "R_own_vs_R": (256, -0.053793, 3.06e-11, True),
            "C_own_vs_R_own": (0, 0.0, 1.0, False),
        },
        "means": {
            "C": 13.386352291951576,
            "C_own": 13.397696658348044,
            "R_own": 13.397696658348044,
            "F": 13.4111617654562,
            "R": 13.451489645366868,
        },
        "clone_gain": -0.065137,
        "penalty_fresh": -0.053793,
        "penalty_cloned": +0.011344,
        "operation_mix": {"grow": 548, "prune": 604},
        "C_vs_F": (237, -0.0248),
        "F_vs_R": (236, -0.0403),
        "R_own_vs_F": (207, -0.0135),
    },
}


def _load(name: str) -> dict:
    path = summaries_dir() / name
    if not path.exists():
        pytest.fail(f"missing committed study summary: {path}")
    return json.loads(path.read_text())


def _deltas(summary: dict, first: str, second: str) -> list[float]:
    return [
        row["losses"][first] - row["losses"][second] for row in summary["per_seed"]
    ]


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_section_5_28_numbers_recompute_from_per_seed(name: str) -> None:
    quoted = QUOTED[name]
    summary = _load(name)
    assert summary["record_type"] == "unforced_clone_study"
    assert summary["study_id"] == quoted["study_id"]
    assert summary["per_primary_alpha"] == PER_PRIMARY_ALPHA
    assert summary["declared_bar"] == DECLARED_BAR
    start, stop = quoted["seeds"]
    assert summary["seeds"] == list(range(start, stop))
    assert summary["seeds"] == [row["seed"] for row in summary["per_seed"]]
    assert all(summary["built_in_checks"].values()), summary["built_in_checks"]

    for comparison_name, (wins, mean, p, passes) in quoted["comparisons"].items():
        first, second = comparison_name.split("_vs_")
        deltas = _deltas(summary, first, second)
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
    for key in ("clone_gain", "penalty_fresh", "penalty_cloned"):
        assert summary[key] == pytest.approx(quoted[key], abs=5e-6), key
    assert summary["operation_mix"] == quoted["operation_mix"]


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_section_5_28_the_regime_is_genuinely_drawn(name: str) -> None:
    """Both operations must occur, or this is a forced block in disguise."""
    summary = _load(name)
    grows = sum(row["grows"] for row in summary["per_seed"])
    prunes = sum(row["prunes"] for row in summary["per_seed"])
    assert grows > 0 and prunes > 0, "the block must contain both operations"
    assert grows + prunes == 3 * len(summary["per_seed"])
    assert summary["operation_mix"] == {"grow": grows, "prune": prunes}
    assert all(row["mutations"]["R"] == 3 for row in summary["per_seed"])


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_section_5_28_the_pairing_held_without_being_assumed(name: str) -> None:
    """Identical genomes were recorded rather than asserted, because a drawn prune
    can fall back to a load-based grow which reads the weights cloning changes.
    It held anyway - so the pairing is exact even here."""
    summary = _load(name)
    rows = summary["per_seed"]
    n = len(rows)
    assert summary["seeds_with_identical_final_genome"] == n
    assert summary["seeds_with_identical_operation_mix"] == n
    assert all(row["same_final_genome"] for row in rows)
    assert all(row["same_operation_mix"] for row in rows)
    assert all(row["fixed_arm_identical_across_settings"] for row in rows)
    assert all(row["losses"]["C_own"] == row["losses"]["R_own"] for row in rows)


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_section_5_28_branch_a_and_the_reversal(name: str) -> None:
    """Branch (a): cloning helps and no penalty remains."""
    summary = _load(name)
    comparisons = summary["comparisons"]
    assert comparisons["C_vs_R"]["passes"] is True, "P1 must pass for branch (a)"
    assert comparisons["C_own_vs_C"]["passes"] is False, "P2 must fail for branch (a)"
    assert summary["penalty_fresh"] < 0, "the historical path must cost something"
    assert summary["penalty_cloned"] > 0, "cloning must invert the penalty"


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_section_5_28_mutating_the_historical_way_costs_versus_not_mutating(
    name: str,
) -> None:
    """The actionable half: F < R significantly, and cloning flips the sign."""
    quoted = QUOTED[name]
    summary = _load(name)

    # These three secondary contrasts are quoted to 4dp in the report, so the
    # tolerance is 1e-4 rather than the 5e-6 used for the registered primaries.
    wins, mean = quoted["F_vs_R"]
    deltas = _deltas(summary, "F", "R")
    assert sum(delta < 0 for delta in deltas) == wins, "F < R win count"
    assert sum(deltas) / len(deltas) == pytest.approx(mean, abs=1e-4)
    assert _binomial_survival(wins, len(deltas)) < 0.0125, (
        "the historical path must be significantly worse than not mutating; this "
        "is what makes the pipeline recommendation a requirement rather than a "
        "preference"
    )

    wins, mean = quoted["C_vs_F"]
    deltas = _deltas(summary, "C", "F")
    assert sum(delta < 0 for delta in deltas) == wins, "C < F win count"
    assert sum(deltas) / len(deltas) == pytest.approx(mean, abs=1e-4)
    assert _binomial_survival(wins, len(deltas)) < 0.0125, (
        "cloning must make mutating pay versus not mutating"
    )

    wins, _ = quoted["R_own_vs_F"]
    deltas = _deltas(summary, "R_own", "F")
    assert sum(delta < 0 for delta in deltas) == wins, "R_own < F win count"
    assert summary["loss_means"]["C"] < min(
        summary["loss_means"]["F"], summary["loss_means"]["R_own"]
    ), "the clone arm must be the best of the three by mean"
