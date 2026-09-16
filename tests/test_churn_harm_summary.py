"""Pin the section 5.39 churn-harm summary (``real-path-churn-harm.json``).

The block's verdict is branch (b): doing nothing still beats churning with the repair
in place. That verdict is only trustworthy if the facts under it are pinned - the
per-seed losses really imply the win counts, the harm reduction really is the section
5.38.2 contrast, and the branch really follows from the two primaries rather than from
a sign convention. Every number here is read back out of the committed summary and
re-derived where it can be.

The direction convention is the whole point of section 5.39.1, so it is asserted
explicitly: deltas are ``null - churn``, and **negative** is the harmful direction.
If the runner ever flips the tail back, these tests fail rather than quietly
reporting the opposite branch.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

SUMMARY = Path("artifacts/study-summaries/real-path-churn-harm.json")
# Section 5.38.1's block (12 seeds) and section 5.38.2's declared post-hoc pool of
# both blocks (18 seeds). The reduction in harm is the 18-seed quantity.
CHURN_SUMMARY_N12 = Path("artifacts/study-summaries/real-path-clone-dev-n12.json")
CHURN_SUMMARY_POOLED = Path("artifacts/study-summaries/real-path-clone-dev-pooled.json")


def _summary() -> dict:
    return json.loads(SUMMARY.read_text())


def test_the_block_ran_its_declared_seeds_on_both_registered_queues():
    payload = _summary()
    assert payload["report_section"] == "5.39"
    assert payload["declared_n"] == 18
    assert payload["seeds"] == list(range(15210, 15216)) + list(range(15220, 15232))
    assert payload["seed_blocks"]["5.38"] == list(range(15210, 15216))
    assert payload["seed_blocks"]["5.38.1"] == list(range(15220, 15232))
    # Two primaries, so alpha/2 each, and the disclosed spent block is untouched.
    assert payload["per_primary_alpha"] == 0.025
    assert payload["spent_seeds_disclosed"] == [15201, 15202, 15203]
    assert set(payload["seeds"]).isdisjoint(payload["spent_seeds_disclosed"])
    assert all(payload["built_in_checks"].values()), payload["built_in_checks"]


def test_both_primaries_are_harmful_on_every_seed_under_both_rebuilds():
    comparisons = _summary()["comparisons"]
    for name in ("null_minus_random_base", "null_minus_random_clone"):
        row = comparisons[name]
        assert row["n"] == 18
        assert row["sign_test_min_wins"] == 14
        # The harmful direction is negative here, and every seed agrees with it.
        assert row["tail"] == "lower"
        assert row["mean_delta"] < 0
        assert row["wins_in_tested_direction"] == 18
        assert row["sign_test_p"] == 2.0**-18
        assert row["permutation_p"] <= 0.025
        assert row["established"] is True
        # The two statistics agree, so this is not section 5.31.1's small-effect case.
        assert row["the_two_statistics_disagree"] is False


def test_the_repair_recovers_about_a_third_of_the_harm_without_removing_it():
    comparisons = _summary()["comparisons"]
    untreated = comparisons["null_minus_random_base"]["mean_delta"]
    treated = comparisons["null_minus_random_clone"]["mean_delta"]
    assert untreated < treated < 0
    # Both means are negative, so the reduction in *harm* is the difference of the
    # magnitudes - and it is a third of the untreated harm, not more.
    reduction = abs(untreated) - abs(treated)
    assert reduction / abs(untreated) == 0.31427996249545165
    assert reduction > 0.05 and treated < 0


def test_the_reduction_reproduces_section_5_38_2_and_the_stored_rows():
    """The reduction is not a new quantity: it is 5.38's own contrast, recomputed."""
    payload = _summary()
    replication = payload["replication_check"]
    pooled = json.loads(CHURN_SUMMARY_POOLED.read_text())
    asserts_n12 = json.loads(CHURN_SUMMARY_N12.read_text())["comparisons"][
        "random_clone_vs_base"
    ]
    # Over all 18 seeds the reduction is section 5.38.2's pooled value exactly, and on
    # the 5.38.1 subset the stored rows reproduce that block's own committed mean.
    assert replication["mean_delta"] == pooled["mean_delta"]
    assert replication["wins_in_tested_direction"] == pooled["wins"]
    assert payload["pooled_5_38_1_replication_mean"] == asserts_n12["mean_delta"]
    assert payload["built_in_checks"]["stored_rows_reproduce_the_committed_5_38_1_mean"]
    # Per seed, (null - base) - (null - clone) == (clone - base), so the two primaries
    # and the replication cannot drift apart.
    for row in payload["per_seed"]:
        assert abs(
            row["null_minus_random_base"]
            - row["null_minus_random_clone"]
            - row["clone_minus_base"]
        ) < 1e-12


def test_every_delta_is_the_difference_of_the_losses_it_names():
    """Without this a tampered loss slips through while every aggregate still agrees."""
    for row in _summary()["per_seed"]:
        assert row["null_minus_random_base"] == (
            row["null_final_loss"] - row["random_base_final_loss"]
        )
        assert row["null_minus_random_clone"] == (
            row["null_final_loss"] - row["random_clone_final_loss"]
        )
        assert row["clone_minus_base"] == (
            row["random_clone_final_loss"] - row["random_base_final_loss"]
        )


def test_the_primaries_are_re_derived_from_the_per_seed_rows():
    payload = _summary()
    assert [row["seed"] for row in payload["per_seed"]] == payload["seeds"]
    for name in ("null_minus_random_base", "null_minus_random_clone"):
        row = payload["comparisons"][name]
        deltas = [entry[name] for entry in payload["per_seed"]]
        assert row["per_seed_deltas"] == deltas
        assert row["mean_delta"] == statistics.fmean(deltas)
        assert row["wins_in_tested_direction"] == sum(1 for delta in deltas if delta < 0)


def test_the_null_arm_did_nothing_and_the_churn_arms_spent_their_budget():
    payload = _summary()
    checks = payload["built_in_checks"]
    assert checks["no_null_arm_mutated"]
    assert checks["no_null_arm_booked_probe_flops"]
    assert checks["no_null_event_changed_the_architecture"]
    assert checks["null_events_are_all_recorded_no_ops"]
    assert checks["every_churn_arm_applied_exactly_its_three_mutations"]
    # The accounting detail section 5.39 reports: four cadence slots, three mutations.
    assert checks["every_null_arm_recorded_all_four_cadence_slots"]
    assert all(row["null_events"] == 4 for row in payload["per_seed"])


def test_the_branch_follows_from_the_primaries_and_is_the_registered_b():
    payload = _summary()
    comparisons = payload["comparisons"]
    p1 = comparisons["null_minus_random_base"]["established"]
    p2 = comparisons["null_minus_random_clone"]["established"]
    assert (p1, p2) == (True, True)
    assert payload["branch"] == "b_harm_survives_the_repair"
    # The registered (d) content did not occur: the repair made churn cheaper.
    assert payload["red_flag_treated_harm_larger"] is False


def test_the_summary_states_the_direction_it_was_measured_in():
    """Section 5.39.1's correction has to be legible from the record itself."""
    payload = _summary()
    assert "negative means churn is harmful" in payload["direction"]
    assert "5.39.1" in payload["direction"]
    for row in payload["comparisons"].values():
        assert row["tail"] == "lower"
