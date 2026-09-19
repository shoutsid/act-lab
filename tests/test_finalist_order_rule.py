"""The v1 finalists' pre-registered rule, exercised before its ledgers exist.

``arch-zoo-100m-v1-finalists-6seed/NOTES.md`` declares four branches plus the
floor-limited refusal, on the same exact seed-matched test read twice: over all six
blocks, and over the three that took no part in screening. Each branch is planted
here so the reading is fixed before any real number can be seen - and, in the case
of the clean reversal below, so the unsoundness of the rule's first draft is pinned
as a regression rather than left to the real data to expose.
"""

from __future__ import annotations

import math

import pytest

from act.search.architecture_eval import EvalResult
from act.search.architecture_report import (
    FINALIST_ORDER_FRESH_SEEDS,
    FINALIST_ORDER_RULE,
    FINALIST_ORDER_STEPS,
    blocked_permutation_test,
    finalist_order_decision,
)

SEEDS = (42, 43, 44, 45, 46, 47)
#: Loss by finishing position, so a planting is written as an ordering rather than
#: as numbers - the branch conditions are about which architecture comes first, not
#: about how far apart the losses are.
LOSS_BY_POSITION = (2.70, 2.75, 2.80)


def result(name: str, validation_loss: float, seed: int) -> EvalResult:
    return EvalResult(
        name=name,
        family="test",
        genome_hash=f"hash-{name}",
        seed=seed,
        total_params=100_000_000,
        active_params=40_000_000,
        contracted_flops_per_token=2.5e8,
        dense_runtime_flops_per_token=4.0e8,
        training_flops=2.5e8 * FINALIST_ORDER_STEPS * 2 * 1024,
        validation_flops=2.5e8 * 16 * 2 * 1024,
        final_train_loss=validation_loss + 0.2,
        mean_train_loss=validation_loss + 0.25,
        validation_loss=validation_loss,
        steps=FINALIST_ORDER_STEPS,
        train_tokens=FINALIST_ORDER_STEPS * 2 * 1024,
        val_tokens=16 * 2 * 1024,
        elapsed_s=170.0,
        measured_step_s=0.44,
        measured_tokens_per_s=4700.0,
        peak_memory_mb=3400.0,
        device="cuda",
        description="",
        val_source="holdout:2000000",
    )


#: Finishing order used throughout. ``a`` is the architecture v1 ranked first, so a
#: planting's first element is what the rule has to decide about. Seeds 42/43/44 are
#: v1's screening blocks; 45/46/47 are this arm's screening-free ones.
CONSISTENT = ("a", "b", "c")
REVERSED = ("b", "a", "c")


def plant(tops: list[tuple[str, ...]]) -> list[EvalResult]:
    """One result per architecture per seed, losses fixed by finishing position."""
    rows: list[EvalResult] = []
    for seed, order in zip(SEEDS, tops):
        for position, name in enumerate(order):
            rows.append(result(name, LOSS_BY_POSITION[position], seed))
    return rows


def decide(rows: list[EvalResult], **kwargs):
    pooled = blocked_permutation_test(rows, FINALIST_ORDER_STEPS)
    outcome = {
        "steps": FINALIST_ORDER_STEPS,
        "lengths_present": [FINALIST_ORDER_STEPS],
        "blocked_test": pooled,
    }
    return finalist_order_decision(outcome, rows, **kwargs), pooled


def test_the_declared_thresholds_are_the_ones_the_notes_transcribe():
    assert FINALIST_ORDER_RULE == {"pooled_p": 0.05, "fresh_p": 0.05}
    assert FINALIST_ORDER_STEPS == 1200
    assert FINALIST_ORDER_FRESH_SEEDS == (45, 46, 47)


def test_the_declared_six_block_floor_is_the_one_the_notes_state():
    """1/7776, and the enumeration stays exact at six blocks.

    v1's own decision quotes "design floor 0.02778" from 3 blocks; the arm's whole
    justification is that six blocks move that floor below the threshold, so the
    arithmetic is pinned rather than described.
    """
    floor = 1 / math.factorial(3) ** (len(SEEDS) - 1)
    assert floor == pytest.approx(1 / 7776)
    assert floor < FINALIST_ORDER_RULE["pooled_p"]
    # Under ``max_assignments`` (1,000,000) the test enumerates rather than samples,
    # which is what lets the report say "exact" at six blocks.
    assert 6**6 == 46_656 <= 1_000_000

    rule, pooled = decide(plant([CONSISTENT] * 6))
    assert pooled["method"] == "exact"
    assert pooled["min_achievable_p"] == pytest.approx(floor)


def test_order_established_when_every_block_ranks_the_same_architecture_first():
    rule, pooled = decide(plant([CONSISTENT] * 6))
    assert rule["available"] and rule["branch"] == "order-established"
    assert rule["label"].startswith("ORDER ESTABLISHED")
    assert [row["met"] for row in rule["conditions"]] == [True, True, True]
    assert rule["all_blocks_agree"] is True
    assert rule["block_tops"] == ["a"] * 6
    assert rule["pooled_winner"] == rule["fresh_winner"] == "a"
    assert rule["blocks_all"] == 6 and rule["blocks_fresh"] == 3
    assert rule["fresh_seeds"] == [45, 46, 47]
    assert rule["steps"] == FINALIST_ORDER_STEPS
    # Both readings sit at their floors, which is what full agreement means: the
    # pooled floor from six blocks and the fresh floor from three.
    assert pooled["p"] == pytest.approx(1 / 7776)


def test_a_clean_reversal_is_coupled_not_established():
    """The regression the rule's first draft failed.

    Two consistent halves that disagree about the top: the screening blocks rank
    ``a`` first in every seed, the fresh blocks rank ``b`` first in every seed. The
    omnibus test is significant on the pool - its statistic rises with any
    consistent separation - so a rule that only compared "the pooled winner" with
    "the fresh winner" read this as a replicated ordering. The top must replicate
    across every block for the branch to mean anything.
    """
    rule, pooled = decide(plant([CONSISTENT] * 3 + [REVERSED] * 3))
    assert pooled["p"] < FINALIST_ORDER_RULE["pooled_p"]
    assert rule["available"] and rule["branch"] == "order-coupled"
    assert rule["all_blocks_agree"] is False
    assert len(set(rule["block_tops"])) == 2
    assert [row["met"] for row in rule["conditions"]] == [True, True, False]
    assert "2 different winners across 6 of 6 blocks" in rule["conditions"][2]["value"]


def test_order_coupled_when_one_fresh_block_overturns_the_top():
    """The milder shape of the same thing, with the pool still significant."""
    rule, pooled = decide(plant([CONSISTENT] * 5 + [REVERSED]))
    assert pooled["p"] < FINALIST_ORDER_RULE["pooled_p"]
    assert rule["branch"] == "order-coupled"
    assert rule["conditions"][2]["met"] is False
    assert rule["p_fresh"] >= FINALIST_ORDER_RULE["fresh_p"]


def test_order_coupled_when_the_screening_seeds_carry_the_significance():
    """Screening blocks disagree among themselves; the fresh blocks are unanimous.

    The pooled p clears on the strength of the blocks that selected the finalists,
    which is the anti-conservative coupling v1's caveat named, so the branch must
    not read as established even though the fresh three are perfectly consistent.
    """
    rule, pooled = decide(plant([REVERSED, CONSISTENT, REVERSED] + [CONSISTENT] * 3))
    assert pooled["p"] < FINALIST_ORDER_RULE["pooled_p"]
    assert rule["branch"] == "order-coupled"
    assert rule["conditions"][1]["met"] is True
    assert rule["conditions"][2]["met"] is False


def test_still_not_established_when_six_blocks_cannot_separate():
    """A rotation, not an alternation.

    Alternating two orderings is still a separated design - whichever architecture
    finishes last in *both* is separated in every block, and that alone carries the
    omnibus statistic to p=0.008. A rotation that gives each architecture each
    finishing position exactly twice leaves no column separated at all, which is
    what an undecidable design actually looks like.
    """
    alternating = [
        tuple(CONSISTENT[(position + index) % 3] for position in range(3))
        for index in range(len(SEEDS))
    ]
    for name in CONSISTENT:
        finishes = [order.index(name) for order in alternating]
        assert finishes.count(0) == finishes.count(1) == finishes.count(2) == 2
    rule, pooled = decide(plant(alternating))
    assert pooled["p"] >= FINALIST_ORDER_RULE["pooled_p"]
    assert rule["branch"] == "still-not-established"
    assert rule["label"].startswith("STILL NOT ESTABLISHED")
    assert rule["conditions"][0]["met"] is False
    assert rule["all_blocks_agree"] is False


def test_floor_limited_when_the_declared_threshold_sits_below_the_floor():
    """A threshold under the exact floor is undecidable by analysis, not by data."""
    rule, _ = decide(plant([CONSISTENT] * 6), pooled_p=1e-5)
    assert rule["branch"] == "floor-limited"
    assert rule["label"].startswith("OPEN")


def test_unavailable_when_the_outcome_is_not_at_the_declared_length():
    """The HORIZON_STEPS lesson: the length is declared, never inferred.

    ``best_available_summary`` reports the longest length that *has rows*, so an arm
    that lost its declared length would otherwise be read at whatever survived.
    """
    rows = plant([CONSISTENT] * 6)
    pooled = blocked_permutation_test(rows, FINALIST_ORDER_STEPS)
    outcome = {
        "steps": 800,
        "lengths_present": [800, 1200],
        "blocked_test": pooled,
    }
    rule = finalist_order_decision(outcome, rows)
    assert rule["available"] is False and rule["branch"] == "unavailable"
    assert "declared" in rule["reason"]


def test_unavailable_when_the_screening_free_blocks_are_incomplete():
    """A partial arm is refused rather than labelled.

    Two of the three fresh seeds landed. The fresh reading would then run over two
    blocks - floor 1/6, far above the threshold - and its seed set is not the
    declared one, so the rule declines instead of reporting a branch from a design
    it did not pre-register.
    """
    rows = [row for row in plant([CONSISTENT] * 6) if row.seed != 47]
    pooled = blocked_permutation_test(rows, FINALIST_ORDER_STEPS)
    outcome = {
        "steps": FINALIST_ORDER_STEPS,
        "lengths_present": [FINALIST_ORDER_STEPS],
        "blocked_test": pooled,
    }
    rule = finalist_order_decision(outcome, rows)
    assert rule["available"] is False and rule["branch"] == "unavailable"
    assert "45, 46" in rule["reason"]
    assert rule["reading"]


def test_unavailable_when_there_is_no_seed_matched_test_at_the_length():
    rule = finalist_order_decision({"steps": FINALIST_ORDER_STEPS, "lengths_present": []}, [])
    assert rule["available"] is False and rule["branch"] == "unavailable"


def test_every_declared_branch_has_a_reading_and_a_label():
    from act.search.architecture_report import FINALIST_ORDER_READINGS

    for branch in (
        "order-established",
        "order-coupled",
        "still-not-established",
        "floor-limited",
    ):
        assert branch in FINALIST_ORDER_READINGS
        assert FINALIST_ORDER_READINGS[branch]
    assert "unavailable" in FINALIST_ORDER_READINGS


def test_the_fresh_reading_is_worth_running_and_two_seeds_would_not_be():
    """Three fresh blocks bind at 1/36, below the threshold; two would bind at 1/6.

    This is why the arm buys three fresh seeds rather than two: at two the fresh
    reading could never clear the declared threshold however the ordering fell, and
    the branch gated on it would be unreachable rather than negative.
    """
    all_rows = plant([CONSISTENT] * 6)
    fresh = [row for row in all_rows if row.seed in FINALIST_ORDER_FRESH_SEEDS]
    three = blocked_permutation_test(fresh, FINALIST_ORDER_STEPS)
    two = blocked_permutation_test(
        [row for row in fresh if row.seed in FINALIST_ORDER_FRESH_SEEDS[:2]],
        FINALIST_ORDER_STEPS,
    )
    assert three["min_achievable_p"] == pytest.approx(1 / 36)
    assert three["min_achievable_p"] < FINALIST_ORDER_RULE["fresh_p"]
    assert two["min_achievable_p"] == pytest.approx(1 / 6)
    assert two["min_achievable_p"] >= FINALIST_ORDER_RULE["fresh_p"]
