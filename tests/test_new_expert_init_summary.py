"""Pins for the section 5.26 stored records (cloned vs freshly drawn expert).

Section 5.26 asked whether the residual mid-training penalty section 5.25
attributed to the grow is the **freshly drawn expert**. It answered branch (a):
cloning the layer's busiest survivor removes the penalty and inverts it, and the
clone-grow fused run beats every reference including the unmutated arm.

Two things are pinned that the study text leans on hardest, and neither is
reasoned about anywhere else:

1. **The pairing is exact.** Both settings must choose the same final genome, or
   the "penalty" contrast is between two substrates and the framing is
   meaningless. Checked as `same_final_genome` and `C_own == R_own` in every seed.
2. **The sign reversal.** The cloned arm's penalty is *negative* - the fused run
   beats the frozen retrain. A test that only asserted "P1 passes" would let the
   headline claim rot silently if the sign ever flipped back.
"""

from __future__ import annotations

import json

import pytest

from act.evolution.figures_data import summaries_dir
from act.evolution.study import _binomial_survival, _gate_min_wins

PER_PRIMARY_ALPHA = 0.05 / 4
DECLARED_BAR = 215

# name -> (study_id, (seed_start, seed_stop), quoted comparisons, quoted means)
QUOTED: dict[str, dict] = {
    "new-expert-init-depth2-384.json": {
        "study_id": "study_99a12c16f910e9c0",
        "seeds": (103000, 103384),
        "comparisons": {
            # name -> (wins, mean, p, passes)
            "C_vs_R": (315, -0.100535, 5.88e-39, True),
            "C_own_vs_C": (126, +0.059960, 1.0, False),
            "R_own_vs_R": (229, -0.040575, 9.36e-05, True),
        },
        "means": {
            "C": 14.241404,
            "F": 14.288690,
            "C_own": 14.301364,
            "R": 14.341939,
        },
        "penalty_fresh": -0.040575,
        "penalty_cloned": +0.059960,
        "clone_gain": -0.100535,
    },
    "new-expert-init-depth3-384.json": {
        "study_id": "study_ee833b6b284195f9",
        "seeds": (104000, 104384),
        "comparisons": {
            "C_vs_R": (307, -0.103641, 6.40e-34, True),
            "C_own_vs_C": (131, +0.074491, 1.0, False),
            "R_own_vs_R": (221, -0.029151, 1.79e-03, True),
        },
        "means": {
            "C": 13.381863,
            "F": 13.436168,
            "C_own": 13.456354,
            "R": 13.485505,
        },
        "penalty_fresh": -0.029151,
        "penalty_cloned": +0.074491,
        "clone_gain": -0.103641,
    },
}


def _load(name: str) -> dict:
    path = summaries_dir() / name
    if not path.exists():
        pytest.fail(f"missing committed study summary: {path}")
    return json.loads(path.read_text())


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_section_5_26_numbers_recompute_from_per_seed(name: str) -> None:
    quoted = QUOTED[name]
    summary = _load(name)
    assert summary["record_type"] == "new_expert_init_study"
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
        # The stored comparison block must agree with its own per_seed rows, or a
        # reader quoting the headline number would be quoting a field nobody
        # recomputes.
        assert stored["wins"] == wins, comparison_name
        assert stored["mean_delta"] == pytest.approx(mean, abs=5e-6)
        assert stored["n"] == len(deltas)
        assert stored["passes"] is passes
        assert stored["min_wins"] == _gate_min_wins(len(deltas), PER_PRIMARY_ALPHA)

    for label, value in quoted["means"].items():
        assert summary["loss_means"][label] == pytest.approx(value, abs=1e-6)
    assert summary["penalty_fresh"] == pytest.approx(quoted["penalty_fresh"], abs=5e-6)
    assert summary["penalty_cloned"] == pytest.approx(
        quoted["penalty_cloned"], abs=5e-6
    )
    assert summary["clone_gain"] == pytest.approx(quoted["clone_gain"], abs=5e-6)


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_section_5_26_pairing_is_exact(name: str) -> None:
    """Both settings must make the same architectural choices.

    This is what makes the penalty a within-seed paired contrast on one final
    genome rather than a comparison between two. The forced grow picks its layer
    from the seed's own RNG and cloning consumes no randomness, so it holds - and
    it is asserted here rather than argued from those two facts.
    """
    summary = _load(name)
    rows = summary["per_seed"]
    assert all(row["mutations"]["R"] == row["mutations"]["C"] == 3 for row in rows)
    assert all(row["same_final_genome"] is True for row in rows), (
        "the two settings ended on different final genomes; the penalty contrast "
        "would no longer be a paired comparison"
    )
    assert all(row["losses"]["C_own"] == row["losses"]["R_own"] for row in rows)
    assert all(row["same_own_reference"] is True for row in rows)


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_section_5_26_branch_a_and_the_sign_reversal(name: str) -> None:
    """Branch (a): cloning helps AND no penalty remains - the penalty reverses."""
    summary = _load(name)
    comparisons = summary["comparisons"]
    assert comparisons["C_vs_R"]["passes"] is True, "P1 must pass for branch (a)"
    assert comparisons["C_own_vs_C"]["passes"] is False, "P2 must fail for branch (a)"

    # P2 fails in the exonerating direction: the frozen retrain is WORSE than the
    # cloned fused run, so the "penalty" is negative rather than merely absent.
    assert summary["penalty_fresh"] < 0, "the fresh draw must cost something"
    assert summary["penalty_cloned"] > 0, (
        "cloning must invert the penalty, which is the branch (a) claim"
    )


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_section_5_26_the_clone_arm_beats_every_reference(name: str) -> None:
    """The report's headline ordering: C < F < C_own < R on the means."""
    means = _load(name)["loss_means"]
    assert means["C"] < means["F"], "the clone-grow run beats the unmutated arm"
    assert means["F"] < means["C_own"]
    assert means["C_own"] < means["R"], "the clone arm beats the historical arm"


def _seeds_of(payload: object) -> set[int]:
    """Collect every seed-ish integer from a summary, whatever its shape.

    Summaries are mostly dicts with a ``seeds`` list, but some records are plain
    lists of per-seed rows, so both are walked rather than assumed.
    """
    if isinstance(payload, dict):
        found = {int(seed) for seed in payload.get("seeds", []) or []}
        if not found and isinstance(payload.get("per_seed"), list):
            found = {
                int(row["seed"]) for row in payload["per_seed"] if "seed" in row
            }
        return found
    if isinstance(payload, list):
        found: set[int] = set()
        for item in payload:
            found |= _seeds_of(item)
        return found
    return set()


@pytest.mark.parametrize("name", sorted(QUOTED))
def test_section_5_26_used_fresh_seed_blocks(name: str) -> None:
    """The registered blocks must not overlap any earlier study's seeds.

    The report calls these "fresh blocks"; a reused block would make that claim
    false without changing any statistic, so it is checked.
    """
    used: set[int] = set()
    for path in sorted(summaries_dir().glob("*.json")):
        if path.name == name:
            continue
        used |= _seeds_of(json.loads(path.read_text()))
    overlap = used & set(_load(name)["seeds"])
    assert not overlap, f"seed block overlaps an earlier study: {sorted(overlap)[:5]}"
