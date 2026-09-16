"""Pin the section 5.38 real-path block summary (``real-path-clone-dev.json``).

The block's verdict is branch (c) - the branch the pre-registration fixed in advance as a
red flag, where the probe-driven arm improves while the clean seeded arm does not. That
verdict is only trustworthy if the facts under it are pinned: the clean arm's event
sequences really are setting-independent, the probe arm's really are not, and the two
primaries really do land on opposite sides of their bars. Every number here is read back
out of the committed summary and re-derived where it can be.
"""

from __future__ import annotations

import json
from pathlib import Path

SUMMARY = Path("artifacts/study-summaries/real-path-clone-dev.json")


def _summary() -> dict:
    return json.loads(SUMMARY.read_text())


def test_the_block_ran_its_declared_seeds_and_settings():
    payload = _summary()
    assert payload["report_section"] == "5.38"
    assert payload["declared_n"] == 6
    assert payload["seeds"] == [15210, 15211, 15212, 15213, 15214, 15215]
    # The section 5.36 smoke's seeds are disclosed as spent rather than reused.
    assert payload["spent_seeds_disclosed"] == [15201, 15202, 15203]
    assert set(payload["seeds"]).isdisjoint(payload["spent_seeds_disclosed"])
    assert payload["per_primary_alpha"] == 0.025
    assert all(payload["built_in_checks"].values()), payload["built_in_checks"]


def test_both_primaries_are_readable_and_land_opposite_ways():
    """The two statistics on the clean arm agree it misses; the probe arm clears its bar."""
    comparisons = _summary()["comparisons"]
    random_arm = comparisons["random_clone_vs_base"]
    adaptive_arm = comparisons["adaptive_clone_vs_base"]

    assert random_arm["n"] == adaptive_arm["n"] == 6
    assert random_arm["sign_test_min_wins"] == 6

    # The clean arm: helpful direction, not enough evidence.
    assert random_arm["mean_delta"] > 0
    assert random_arm["wins"] == 4
    assert random_arm["sign_test_passes"] is False
    assert random_arm["passes"] is False

    # The contaminated arm: decisive, and that is exactly the red flag.
    assert adaptive_arm["mean_delta"] > random_arm["mean_delta"]
    assert adaptive_arm["wins"] == 6
    assert adaptive_arm["sign_test_passes"] is True
    assert adaptive_arm["passes"] is True

    # Section 5.31.1's lesson: the two statistics disagreeing is itself a signal.
    assert adaptive_arm["the_two_statistics_disagree"] is False
    assert random_arm["the_two_statistics_disagree"] is False


def test_the_registered_branch_is_the_red_flag_one():
    outcome = _summary()["outcome"]
    assert outcome == {
        "branch_a_all_primaries_pass": False,
        "branch_b_random_only": False,
        "branch_c_adaptive_only": True,
        "branch_d_neither": False,
    }


def test_only_the_clean_arm_isolates_the_rebuild():
    """The fact the whole interpretation rests on: the random arm selected the same events."""
    payload = _summary()
    assert payload["events_changed_by_arm"] == {"adaptive": 6, "random": 0}
    assert payload["event_sequences_identical_on"] == {"adaptive": 0, "random": 6}
    assert payload["built_in_checks"]["the_random_arms_events_are_setting_independent"]
    # Preservation makes a grow look better to the probe, so the probe arm grows more:
    # the predicted direction, and the reason the two effects cannot be separated here.
    assert payload["grows_by_arm_and_setting"] == {
        "adaptive": {"base": 5, "clone": 6},
        "random": {"base": 9, "clone": 9},
    }


def test_the_per_seed_deltas_re_derive_the_published_means():
    payload = _summary()
    rows = payload["per_seed"]
    assert [row["seed"] for row in rows] == payload["seeds"]
    for arm, key in (("random", "random_clone_vs_base"), ("adaptive", "adaptive_clone_vs_base")):
        deltas = [row[f"{arm}_base"] - row[f"{arm}_clone"] for row in rows]
        stored = payload["comparisons"][key]["per_seed_deltas"]
        assert deltas == stored
        mean = sum(deltas) / len(deltas)
        assert abs(mean - payload["comparisons"][key]["mean_delta"]) < 1e-12
        wins = sum(1 for delta in deltas if delta > 0)
        assert wins == payload["comparisons"][key]["wins"]
    # The two reversing seeds the write-up names, checked by seed rather than by count.
    reversing = [
        row["seed"]
        for row in rows
        if row["random_base"] - row["random_clone"] < 0
    ]
    assert reversing == [15214, 15215]


def test_the_non_reproducibility_is_declared_for_the_gpu_block():
    payload = _summary()
    assert payload["bit_reproducible"] is False
    assert "GPU" in payload["reproduction_note"]


# --- section 5.38.1: the correctly sized clean-arm block ----------------------

N12 = Path("artifacts/study-summaries/real-path-clone-dev-n12.json")
POOLED = Path("artifacts/study-summaries/real-path-clone-dev-pooled.json")


def _n12() -> dict:
    return json.loads(N12.read_text())


def test_the_sized_block_ran_its_declared_seeds_and_passed_its_primary():
    payload = _n12()
    assert payload["report_section"] == "5.38.1"
    assert payload["declared_n"] == 12
    assert payload["seeds"] == list(range(15220, 15232))
    # Disjoint from section 5.38's block and from the spent pilot: no seed is reused.
    assert set(payload["seeds"]).isdisjoint({15210, 15211, 15212, 15213, 15214, 15215})
    assert set(payload["seeds"]).isdisjoint(payload["spent_seeds_disclosed"])
    # One primary here, so the per-primary bar is the full alpha.
    assert payload["per_primary_alpha"] == 0.05
    assert payload["config"]["arms"] == ["random"]
    assert all(payload["built_in_checks"].values()), payload["built_in_checks"]

    primary = payload["comparisons"]["random_clone_vs_base"]
    assert primary["n"] == 12
    assert primary["mean_delta"] > 0
    assert primary["permutation_p"] <= 0.05
    assert primary["passes"] is True
    assert payload["outcome"] == {"primary_passes": True}


def test_the_two_statistics_disagree_and_that_is_recorded_not_hidden():
    """The pass rides on the mean; the per-seed win rate is near chance.

    Section 5.31.1's signature, and the reason the write-up calls the effect small and
    right-skewed rather than claiming a reliable win.
    """
    primary = _n12()["comparisons"]["random_clone_vs_base"]
    assert primary["wins"] == 8
    assert primary["sign_test_min_wins"] == 10
    assert primary["sign_test_passes"] is False
    assert primary["the_two_statistics_disagree"] is True

    # The skew, named by seed in the report: one seed carries a quarter of the mean.
    deltas = primary["per_seed_deltas"]
    assert len(deltas) == 12
    assert max(deltas) > 0.18
    assert abs(sum(deltas) / len(deltas) - primary["mean_delta"]) < 1e-12
    without_peak = (sum(deltas) - max(deltas)) / (len(deltas) - 1)
    assert without_peak < 1.5 * primary["mean_delta"]


def test_the_clean_arm_still_isolated_the_rebuild_on_all_twelve_seeds():
    payload = _n12()
    assert payload["events_changed_by_arm"] == {"random": 0}
    assert payload["event_sequences_identical_on"] == {"random": 12}
    assert payload["grows_by_arm_and_setting"] == {
        "random": {"base": 21, "clone": 21}
    }


def test_the_pooled_secondary_is_marked_post_hoc_with_its_caveat():
    """Two blocks over the same prescription: reported together, never as a verdict."""
    payload = json.loads(POOLED.read_text())
    assert payload["post_hoc"] is True
    assert "not a verdict" in payload["caveat"]
    assert payload["blocks"] == {"5.38": 6, "5.38.1": 12}
    assert payload["n"] == 18
    assert len(payload["per_seed_deltas"]) == 18
    # The pooled mean test is decisive while the pooled win rate is not - the same
    # signature, and the reason the write-up separates expected loss from per-seed wins.
    assert payload["mean_delta"] > 0
    assert payload["permutation_p"] < 0.01
    assert payload["mean_test_passes"] is True
    assert payload["wins"] == 12
    assert payload["sign_test_min_wins"] == 13
    assert payload["sign_test_passes"] is False
    assert payload["the_two_statistics_disagree"] is True


def test_the_pooled_deltas_are_the_two_blocks_put_together():
    """Re-derive the pooled sample from the two block summaries, not from the pool."""
    pooled = json.loads(POOLED.read_text())["per_seed_deltas"]
    block_38 = [
        row["random_base"] - row["random_clone"] for row in _summary()["per_seed"]
    ]
    block_381 = [
        row["random_base"] - row["random_clone"] for row in _n12()["per_seed"]
    ]
    assert sorted(pooled) == sorted(block_38 + block_381)
    assert len(block_38) == 6 and len(block_381) == 12
