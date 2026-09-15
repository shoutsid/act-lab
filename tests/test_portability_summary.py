"""The committed section 5.18 summary stays machine-checked.

Pins the versioned record produced by ``scripts/portability_study.py summary``,
including the two things that make the study readable rather than merely
positive:

- primary 1 is the **difference-in-differences**, not the raw own-minus-donor
  contrast (section 5.18.1's pre-run correction), so a later reader cannot swap
  the reported primary to the statistic that produces a different label;
- the summary's own secondary rows record that the *contaminated* contrast
  fails, which is the evidence that the correction matters.
"""

from __future__ import annotations

import json
import math

import pytest

from act.evolution.figures_data import summaries_dir
from act.evolution.study import _binomial_survival, _gate_min_wins

SUMMARY = "portability-depth4-384.json"
DECLARED_ALPHA = 0.025
DECLARED_BAR = 212
# Report section 5.18.2 table: comparison -> (wins, mean, p, passes)
QUOTED = {
    "pairing_difference_in_differences": (237, -0.083037, 2.54e-06, True),
    "D_vs_F": (234, -0.046511, 1.06e-05, True),
    "O_vs_O_fixed": (293, -0.129548, 3.65e-26, True),
    "O_vs_D": (199, -0.145159, 2.54e-01, False),
    "D_vs_O_fixed": (194, 0.015611, 4.39e-01, False),
    "F_vs_own_fixed": (190, 0.062122, 6.01e-01, False),
}


def _load() -> dict:
    path = summaries_dir() / SUMMARY
    if not path.exists():
        pytest.fail(f"missing committed study summary: {path}")
    return json.loads(path.read_text())


def _deltas(summary: dict, name: str) -> list[float]:
    rows = summary["per_seed"]
    if name == "pairing_difference_in_differences":
        return [
            (row["transport_losses"]["O"] - row["transport_losses"]["D"])
            - (row["discovery_losses"]["fixed"] - row["transport_losses"]["F"])
            for row in rows
        ]
    if name == "F_vs_own_fixed":
        return [
            row["transport_losses"]["F"] - row["discovery_losses"]["fixed"]
            for row in rows
        ]
    treatment, control = name.split("_vs_")
    return [
        row["transport_losses"][treatment] - row["transport_losses"][control]
        for row in rows
    ]


def test_portability_summary_recomputes_from_per_seed() -> None:
    summary = _load()
    assert summary["record_type"] == "portability_study"
    assert summary["protocol"] == "donor-initialization-transport"
    assert summary["primary_1_statistic"] == "difference_in_differences"
    assert summary["donor_offset"] == 1000
    assert summary["seeds"] == [row["seed"] for row in summary["per_seed"]]
    assert summary["seeds"] == list(range(83000, 83384))
    assert summary["donor_seeds"] == [seed + 1000 for seed in summary["seeds"]]
    for name, quoted in QUOTED.items():
        comparison = summary["comparisons"][name]
        deltas = _deltas(summary, name)
        wins = sum(delta < 0 for delta in deltas)
        mean = sum(deltas) / len(deltas)
        p = _binomial_survival(wins, len(deltas))
        assert comparison["wins"] == wins == quoted[0]
        assert math.isclose(comparison["mean_delta"], mean, rel_tol=0, abs_tol=1e-12)
        assert math.isclose(mean, quoted[1], rel_tol=0, abs_tol=5e-5)
        assert math.isclose(comparison["binomial_p"], p, rel_tol=1e-6, abs_tol=0)
        assert math.isclose(p, quoted[2], rel_tol=5e-2)
        assert comparison["passes"] is quoted[3]
        assert comparison["min_wins"] == _gate_min_wins(len(deltas), DECLARED_ALPHA)
        assert comparison["bar_and_p_agree"] is True


def test_portability_summary_primary_is_the_corrected_statistic() -> None:
    """Section 5.18.1: the run's label rests on the differenced contrast."""
    summary = _load()
    assert summary["per_primary_alpha"] == DECLARED_ALPHA
    assert summary["declared_bar"] == DECLARED_BAR == _gate_min_wins(384, DECLARED_ALPHA)
    primaries = (
        summary["comparisons"]["pairing_difference_in_differences"],
        summary["comparisons"]["D_vs_F"],
    )
    assert all(comparison["passes"] for comparison in primaries)
    # The raw own-minus-donor contrast FAILS, which is why the correction was
    # needed: it is dominated by the seed's initialization-quality difference.
    raw = summary["comparisons"]["O_vs_D"]
    assert raw["passes"] is False
    assert raw["wins"] < raw["min_wins"] < 237
    # The topology-versus-initial-topology rows split by whose init both sides
    # hold - every matched-init row wins, the mismatched ones fail. This is the
    # synthesis table of section 5.18.2, recomputed from per_seed.
    matched_own = summary["comparisons"]["O_vs_O_fixed"]
    assert (matched_own["wins"], matched_own["passes"]) == (293, True)
    matched_donor = summary["comparisons"]["D_vs_F"]
    assert (matched_donor["wins"], matched_donor["passes"]) == (234, True)
    for mismatched in ("O_vs_D", "D_vs_O_fixed"):
        row = summary["comparisons"][mismatched]
        assert row["wins"] < row["min_wins"]
        deltas = _deltas(summary, mismatched)
        assert sum(delta < 0 for delta in deltas) == row["wins"]


def test_portability_summary_built_in_checks_and_branch() -> None:
    summary = _load()
    checks = summary["built_in_checks"]
    assert all(checks.values())
    assert all(isinstance(value, bool) for value in checks.values())
    assert checks["own_rewind_reproduces_fixed_arm_exactly"] is True
    assert checks["donor_model_replaces_every_tensor"] is True
    assert summary["outcome"].startswith("branch (d) both PASS")
    seeds = set(summary["seeds"])
    assert len(seeds) == 384
    donors = set(summary["donor_seeds"])
    assert seeds.isdisjoint(donors)
    for other in (
        "null-control-depth4-384.json",
        "reusability-depth4-384.json",
        "initialization-depth4-384.json",
    ):
        other_seeds = set(json.loads((summaries_dir() / other).read_text())["seeds"])
        assert seeds.isdisjoint(other_seeds)
        assert donors.isdisjoint(other_seeds)


def test_portability_summary_means_and_discovery_ordering() -> None:
    summary = _load()
    n = len(summary["per_seed"])
    for label, mean in summary["transport_means"].items():
        recomputed = sum(
            row["transport_losses"][label] for row in summary["per_seed"]
        ) / n
        assert math.isclose(mean, recomputed, rel_tol=0, abs_tol=1e-12)
    for arm, mean in summary["discovery_means"].items():
        recomputed = sum(
            row["discovery_losses"][arm] for row in summary["per_seed"]
        ) / n
        assert math.isclose(mean, recomputed, rel_tol=0, abs_tol=1e-12)
    # Third fresh block, third replication of the depth-4 discovery ordering.
    means = summary["discovery_means"]
    assert means["adaptive"] < means["fixed"] < means["random"]
    transport = summary["transport_means"]
    assert transport["O"] < transport["O_fixed"] < transport["D"] < transport["F"]
    # The adaptive topology is smaller than the initial one, so no win here is
    # bought with active parameters.
    assert summary["adaptive_active_params_lower_than_initial_seeds"] > 300


# Section 5.19: the same four transports at depths 2 and 3, on the registered
# blocks 85000-85383 and 86000-86383 with donors +2000. Its registered branch
# table is SEPARATE from the module-level letters above - its second primary is
# O < O_fixed, not D < F - so its verdict is read from those two rows here.
# These summaries were regenerated from the surviving shards; see report 5.19.
SHALLOW = {
    "portability-depth2-384.json": {
        "study_id": "study_436e3cfb83e13fc0",
        "block": (85000, 85384),
        "quoted": {
            "pairing_difference_in_differences": (219, -0.044341, 3.38e-03, True),
            "O_vs_O_fixed": (260, -0.061844, 1.68e-12, True),
            "D_vs_F": (211, -0.017502, 2.94e-02, False),
            "O_vs_D": (177, 0.037571, 9.43e-01, False),
        },
    },
    "portability-depth3-384.json": {
        "study_id": "study_ea20b7b1ace9534b",
        "block": (86000, 86384),
        "quoted": {
            "pairing_difference_in_differences": (226, -0.064137, 3.05e-04, True),
            "O_vs_O_fixed": (270, -0.104181, 5.19e-16, True),
            "D_vs_F": (216, -0.040044, 8.18e-03, True),
            "O_vs_D": (202, -0.087132, 1.66e-01, False),
        },
    },
}
SHALLOW_ALPHA = 0.0125
SHALLOW_BAR = 215


def _load_named(name: str) -> dict:
    path = summaries_dir() / name
    if not path.exists():
        pytest.fail(f"missing committed study summary: {path}")
    return json.loads(path.read_text())


@pytest.mark.parametrize("name", sorted(SHALLOW))
def test_section_5_19_record_recomputes_and_keeps_its_registered_branch(name: str) -> None:
    spec = SHALLOW[name]
    summary = _load_named(name)
    assert summary["record_type"] == "portability_study"
    assert summary["study_id"] == spec["study_id"]
    assert summary["per_primary_alpha"] == SHALLOW_ALPHA
    assert summary["declared_bar"] == SHALLOW_BAR == _gate_min_wins(384, SHALLOW_ALPHA)
    assert summary["donor_offset"] == 2000
    start, stop = spec["block"]
    assert summary["seeds"] == list(range(start, stop))
    assert summary["donor_seeds"] == [seed + 2000 for seed in summary["seeds"]]
    assert all(summary["built_in_checks"].values()), summary["built_in_checks"]

    for comparison_name, quoted in spec["quoted"].items():
        comparison = summary["comparisons"][comparison_name]
        deltas = _deltas(summary, comparison_name)
        wins = sum(delta < 0 for delta in deltas)
        assert comparison["wins"] == wins == quoted[0]
        assert math.isclose(comparison["mean_delta"], quoted[1], rel_tol=0, abs_tol=5e-5)
        assert math.isclose(
            comparison["binomial_p"], quoted[2], rel_tol=5e-2
        )
        assert comparison["passes"] is quoted[3]
        assert comparison["min_wins"] == SHALLOW_BAR

    # Registered branch (c): BOTH primaries pass at BOTH depths, which is what
    # redirects the project to controller research.
    assert summary["comparisons"]["pairing_difference_in_differences"]["passes"] is True
    assert summary["comparisons"]["O_vs_O_fixed"]["passes"] is True


def test_section_5_19_auto_label_letters_index_section_5_18_not_5_19() -> None:
    """Pin the collision the report warns about, so it cannot drift silently.

    The script's automatic ``outcome`` string uses section 5.18's branch letters,
    so the depth-2 record says "branch (b)" while section 5.19's registered
    reading (both primaries pass) is branch (c). A reader must not quote the
    stored label as this section's verdict.
    """
    depth2 = _load_named("portability-depth2-384.json")
    depth3 = _load_named("portability-depth3-384.json")
    assert depth2["outcome"].startswith("branch (b)")
    assert depth3["outcome"].startswith("branch (d)")
    # ...whereas the section's own branch, read from its two primaries, is (c).
    for summary in (depth2, depth3):
        comparisons = summary["comparisons"]
        both_pass = (
            comparisons["pairing_difference_in_differences"]["passes"]
            and comparisons["O_vs_O_fixed"]["passes"]
        )
        assert both_pass is True
