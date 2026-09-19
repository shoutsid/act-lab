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
    best_available_summary,
    blocked_permutation_test,
    finalist_order_decision,
    verdict,
    write_decision,
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


def test_every_boundary_in_the_rule_decides_at_exact_equality():
    """All three thresholds are compared at exact equality, and each refuses.

    Each is a value the enumeration returns exactly - the pooled test enumerates at
    six blocks, so its p lands on ``1/7776``-spaced values and on the declared 0.05
    itself whenever the design produces it - so none is a boundary the data is
    guaranteed to avoid. Each is pinned because reading one generously asserts a
    replicated ordering the design did not establish.
    """
    # (a) A pooled p sitting exactly on ``pooled_p`` is not established. The blocks
    # disagree here, so the branch that follows is *coupled*, and the planting's p is
    # above the floor, so the floor clause is not what decides it.
    rows = plant([CONSISTENT] * 5 + [REVERSED])
    _, pooled = decide(rows)
    assert pooled["p"] > pooled["min_achievable_p"]
    at = decide(rows, pooled_p=pooled["p"])[0]
    assert at["branch"] == "still-not-established"

    # (b) A fresh p sitting exactly on ``fresh_p`` does not clear it, so an
    # established pooled ordering whose blocks all agree is *coupled*, not
    # established - the difference between the arm's verdict and a qualifier on it.
    rows = plant([CONSISTENT] * 6)
    clear = decide(rows, pooled_p=0.05)[0]
    assert clear["branch"] == "order-established"
    at = decide(rows, pooled_p=0.05, fresh_p=clear["p_fresh"])[0]
    assert at["branch"] == "order-coupled"

    # (c) A pooled floor *equal* to the threshold still refuses: the floor cannot
    # produce a smaller p, so no branch is reachable by more analysis.
    rows = plant([CONSISTENT] * 6)
    _, pooled = decide(rows)
    at = decide(rows, pooled_p=pooled["min_achievable_p"])[0]
    assert at["branch"] == "floor-limited"


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


#: A planting with a material spread but an ordering that does not reproduce, which
#: is the shape v1's own ledgers have: the three architectures separate by 0.025 nats
#: on the mean - above the materiality floor, so the verdict is the *statistical*
#: one rather than NO MATERIAL SIGNAL - while the top of the per-seed ordering
#: changes twice in six blocks. The arm therefore closes on NOT ESTABLISHED even
#: after the re-run, and the action text is what has to change in that case.
WEAKLY_SEPARATED = [
    ("b", "a", "c"),
    ("a", "c", "b"),
    ("b", "c", "a"),
    ("c", "b", "a"),
    ("a", "b", "c"),
    ("a", "b", "c"),
]


def _outcome(rows: list[EvalResult]) -> dict:
    return best_available_summary(rows, list(SEEDS), samples=2_000)


def test_decision_renders_the_finalist_rule_and_its_two_readings(tmp_path):
    """The branch is the arm's deliverable, so it has to reach decision.md."""
    rows = plant([CONSISTENT] * 6)
    outcome = _outcome(rows)
    outcome["finalist_order_decision"] = finalist_order_decision(outcome, rows)
    path = tmp_path / "decision.md"
    write_decision(path, outcome)
    text = path.read_text()
    assert "Pre-registered rule: was v1's finalist ordering underpowered?" in text
    assert "**ORDER ESTABLISHED" in text
    assert "| Condition | Value | Met |" in text
    # Both readings are named in the conditions, so a reader can see that the
    # screening-free blocks were consulted rather than take the pooled p on trust.
    assert "screening-free blocks [45, 46, 47]" in text
    assert "all 6 blocks" in text


def test_the_standing_instruction_survives_until_the_follow_up_is_read(tmp_path):
    """Without the rule in the outcome, both clues stay as they always were."""
    rows = plant([CONSISTENT] * 6)
    path = tmp_path / "decision.md"
    write_decision(path, _outcome(rows))
    text = path.read_text()
    assert "confirm the finalists on seeds never used in screening first" in text
    assert "already been confirmed" not in text
    assert "finalist-order rule below" not in text
    assert "Pre-registered rule: was v1's finalist ordering" not in text


def test_the_not_established_instruction_stops_asking_for_what_was_done(tmp_path):
    """v1's own case: the re-run was bought and it changed nothing.

    The closing instruction used to tell every reader to re-run the finalists on
    seeds never used in screening - which is the arm that produced this very
    outcome. Repeating it sends the next reader to buy compute that is already in
    the file, so once the rule is in the outcome the action reports what it bought
    instead of asking for it again.
    """
    rows = plant(WEAKLY_SEPARATED)
    outcome = _outcome(rows)
    assert outcome["architecture_spread"] > 0.01
    assert verdict(outcome)[0].startswith("NOT ESTABLISHED")

    before = tmp_path / "before.md"
    write_decision(before, outcome)
    assert "Resolve the effect size first: re-run the finalists" in before.read_text()

    outcome["finalist_order_decision"] = finalist_order_decision(outcome, rows)
    assert outcome["finalist_order_decision"]["branch"] == "still-not-established"
    after = tmp_path / "after.md"
    write_decision(after, outcome)
    text = after.read_text()
    assert "The finalists have been re-run on seeds never used in screening" in text
    assert "Resolve the effect size first" not in text
    assert "**STILL NOT ESTABLISHED" in text
    # The rest of the standing instruction is still the right advice, so it stays.
    assert "matched-budget structural design" in text
    assert "GDN costs 4-7x attention" in text


def test_the_established_verdict_does_not_read_a_coupled_branch_as_confirmed(tmp_path):
    """Buying the follow-up is not the ordering having replicated.

    This is the arm's actual outcome. ``verdict()`` took the finalist-order rule's
    mere presence in the outcome as its *result*, so a pooled p - the very number
    the rule reads as the screening seeds - produced "the finalists having already
    been confirmed on seeds never used in screening". That sentence is what an
    auditor carries away, and the rule's own section underneath it says no ranking
    is recommended. Both the headline and the action now follow the branch.
    """
    rows = plant([CONSISTENT] * 3 + [REVERSED] * 3)
    outcome = best_available_summary(rows, list(SEEDS), samples=2_000)
    outcome["finalist_order_decision"] = finalist_order_decision(outcome, rows)
    assert outcome["finalist_order_decision"]["branch"] == "order-coupled"

    headline, action = verdict(outcome)
    # The pooled test did clear the threshold, so the effect label stands - but it
    # says on which blocks, because the screening-free ones did not.
    assert headline.startswith("ARCHITECTURE EFFECT ESTABLISHED ON THE POOLED BLOCKS")
    assert "SCREENING-FREE" in headline
    assert "already been confirmed" not in action
    assert "carrying the effect size but not a ranking" in action

    path = tmp_path / "decision.md"
    write_decision(path, outcome)
    text = path.read_text()
    assert "already been confirmed" not in text
    assert "SCREENING-FREE" in text
    assert "**ORDER COUPLED" in text

    # The confirming branch keeps its original wording: only the coupled case moved,
    # so nothing that already reported a replicated ordering was reworded.
    established_rows = plant([CONSISTENT] * 6)
    established = best_available_summary(established_rows, list(SEEDS), samples=2_000)
    established["finalist_order_decision"] = finalist_order_decision(
        established, established_rows
    )
    assert established["finalist_order_decision"]["branch"] == "order-established"
    assert "already been confirmed" in verdict(established)[1]
    assert "ARCHITECTURE EFFECT ESTABLISHED - spread" in verdict(established)[0]


def test_the_selection_caveat_stops_asking_for_the_confirmation_it_has(tmp_path):
    """The caveat's closing sentence is an instruction, and it has been carried out.

    v1's Evidence section ends by telling the reader to confirm on seeds never used
    in screening. Left verbatim in the pooled record that sends the next reader to
    buy the nine runs which produced it - and it keeps the caveat reading as though
    the test were merely soft, when the confirmation has since said the pooled
    significance *is* the coupling the caveat warns about. Only the remedy goes: the
    diagnosis above it stays true either way.
    """
    from act.search.architecture_report import (
        CONFIRM_REMEDY,
        _caveat_after_the_confirmation,
    )

    diagnosis = (
        "13 architectures were screened at shorter lengths than these 3, and "
        "promotion used those runs, so this test is mildly anti-conservative. "
    )
    caveat = diagnosis + CONFIRM_REMEDY
    rows = plant([CONSISTENT] * 3 + [REVERSED] * 3)
    rule, _ = decide(rows)
    assert rule["branch"] == "order-coupled"

    resolved = _caveat_after_the_confirmation(caveat, rule)
    assert CONFIRM_REMEDY not in resolved
    assert "did not replicate" in resolved
    assert diagnosis in resolved

    # An arm that has not run the confirmation still has to be told to.
    assert _caveat_after_the_confirmation(caveat, {}) == caveat

    # And the replacement reaches the artifact, which is the whole point: the record
    # is what an auditor reads, not the function.
    outcome = best_available_summary(rows, list(SEEDS), samples=2_000)
    outcome["selection_caveat"] = caveat
    outcome["finalist_order_decision"] = finalist_order_decision(outcome, rows)
    path = tmp_path / "decision.md"
    write_decision(path, outcome)
    text = path.read_text()
    assert CONFIRM_REMEDY not in text
    assert "did not replicate" in text


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
