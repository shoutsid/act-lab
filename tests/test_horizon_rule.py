"""The horizon arm's pre-registered rule, exercised before its ledgers exist.

``arch-zoo-100m-v3-crossed-horizon10x/NOTES.md`` declares three branches on one
quantity - the crossed factorial's depth main effect at the arm's longest length -
plus the floor-limited refusal every rule in this phase carries. Each is planted
here so the reading is fixed before any real number can be seen, which is the same
rehearsal-before-the-hours standard the head-paired confirmation arm held.
"""

from __future__ import annotations

import pytest

from act.search.architecture_report import HORIZON_RULE, HORIZON_STEPS, horizon_decision


def _outcome(
    depth_nats: float,
    depth_p: float | None,
    floor: float,
    width_nats: float = -0.04,
    steps: int = 12000,
    lengths=(12000, 1200),
):
    return {
        "steps": steps,
        "lengths_present": list(lengths),
        "crossed_contrast": {
            "available": True,
            "factor_a": "depth",
            "factor_b": "hidden_size",
            "main": {
                "depth": {"nats": depth_nats, "p": depth_p, "min_achievable_p": floor},
                "hidden_size": {"nats": width_nats, "p": 0.0015, "min_achievable_p": floor},
            },
        },
    }


TWO_SEED_FLOOR = 2 / 1296  # C(4,2)^4 assignments over 2 seeds x 2 partner levels
ONE_SEED_FLOOR = 2 / 36


def test_the_declared_threshold_is_the_one_the_notes_transcribe():
    assert HORIZON_RULE == {"main_p": 0.05}
    # The length is declared, not inferred: the pre-registration evaluates the rule
    # "on the 12,000-step rows" and the anchor sits in the same directory.
    assert HORIZON_STEPS == 12000


def test_shallow_wins_persists_when_deep_minus_shallow_is_positive_and_significant():
    rule = horizon_decision(_outcome(+0.016, 0.0015, TWO_SEED_FLOOR))
    assert rule["available"] and rule["branch"] == "shallow-persists"
    assert rule["label"].startswith("SHALLOW-WINS PERSISTS")
    assert [row["met"] for row in rule["conditions"]] == [True, True, True]
    assert rule["steps"] == 12000


def test_depth_reverses_when_deep_now_wins_significantly():
    rule = horizon_decision(_outcome(-0.02, 0.003, TWO_SEED_FLOOR))
    assert rule["branch"] == "depth-reverses"
    assert "EARLY-TRAINING" in rule["label"]
    assert rule["conditions"][1]["met"] is False


def test_depth_washes_out_when_the_test_cannot_separate_the_depths():
    for nats in (+0.004, -0.004):
        rule = horizon_decision(_outcome(nats, 0.31, TWO_SEED_FLOOR))
        assert rule["branch"] == "depth-washes-out", nats


def test_the_threshold_is_exclusive_and_the_floor_refusal_is_inclusive():
    """Both boundaries decide at exact equality, and both decide against the arm.

    The declared rule is ``p < main_p``, so a p landing *on* 0.05 is not evidence of
    a depth difference. The exact re-split test returns values on a discrete grid
    (``2/1296`` here), so the threshold is a number it can actually return - not a
    boundary the data is guaranteed to avoid - and reading it as significant would
    report a direction from an undecidable result.

    The floor refusal mirrors that: a floor *equal* to the threshold cannot produce
    a p below it, so the design cannot reach any branch and the refusal applies at
    equality too.
    """
    # p exactly at the threshold: not significant, so no direction is recommended.
    rule = horizon_decision(_outcome(+0.03, HORIZON_RULE["main_p"], TWO_SEED_FLOOR))
    assert rule["conditions"][0]["met"] is False
    assert rule["branch"] == "depth-washes-out"

    # floor exactly at the threshold: no branch is reachable, so the refusal holds.
    rule = horizon_decision(_outcome(+0.004, 0.30, HORIZON_RULE["main_p"]))
    assert rule["conditions"][0]["met"] is False
    assert rule["branch"] == "open-floor-limited"


def test_one_seed_cannot_reach_a_branch_however_the_contrast_falls():
    """2/36 = 0.056 sits above 0.05, so a single seed returns the refusal, not a verdict."""
    rule = horizon_decision(_outcome(+0.03, ONE_SEED_FLOOR, ONE_SEED_FLOOR))
    assert rule["branch"] == "open-floor-limited"
    assert rule["conditions"][0]["met"] is False
    # A significant result is never floor-limited: the floor only refuses a null.
    rule = horizon_decision(_outcome(+0.03, 0.001, 0.0015))
    assert rule["branch"] == "shallow-persists"


def test_width_is_reported_as_a_secondary_and_does_not_gate_the_branch():
    """The arm's question is depth; a width reversal is recorded, not decided."""
    rule = horizon_decision(_outcome(+0.016, 0.0015, TWO_SEED_FLOOR, width_nats=+0.02))
    assert rule["branch"] == "shallow-persists"
    assert rule["conditions"][2]["met"] is False
    assert "does not gate" in rule["conditions"][2]["condition"]


def test_the_anchor_cannot_be_read_as_the_horizon_result_even_alone():
    """The hole this arm's first attempt fell through.

    ``lengths_present`` only carries lengths that *have rows*, so an arm that lost
    its 12,000-step runs but kept the anchor has 1200 as its longest length present
    and would otherwise look like a design that only ever asked for 1200 - which,
    with the depth effect positive and significant, reads as "shallow wins
    persists at this horizon". The declared length is named, so nothing but an
    explicit override reads the anchor.
    """
    anchor_only = _outcome(+0.08, 0.0015, TWO_SEED_FLOOR, steps=1200, lengths=(1200,))
    rule = horizon_decision(anchor_only)
    assert rule["available"] is False and rule["branch"] == "unavailable"
    assert "declared on 12000" in rule["reason"]
    assert "lengths with rows: [1200]" in rule["reason"]

    # Reading a design that genuinely declares 1200 is possible, but has to be said.
    assert horizon_decision(anchor_only, required_steps=1200)["branch"] == "shallow-persists"

    # And the fallback case: the anchor reported while the horizon has rows too.
    fell_back = horizon_decision(_outcome(+0.08, 0.0015, TWO_SEED_FLOOR, steps=1200))
    assert fell_back["available"] is False


def test_the_rule_reads_real_ledgers_and_refuses_one_seed():
    """Against the crossed arm's committed rows: three seeds reach a branch and
    record the length; one seed sits at the 2/36 floor and is refused."""
    from pathlib import Path

    from act.search.architecture_eval import load_results
    from act.search.architecture_report import best_available_summary, with_structure

    directory = Path("artifacts/arch-zoo-100m-v3-crossed")
    ledgers = sorted(directory.glob("*_seed*_results.jsonl"))
    if not ledgers:
        pytest.skip("arch-zoo-100m-v3-crossed ledgers not present")
    results = with_structure(load_results(ledgers), directory / "zoo_definition.json")
    three_seeds = best_available_summary(results, [42, 43, 44], samples=2_000)
    # Stated as a 1200-step design, which is what this arm is, the rule reaches a
    # branch on real ledgers.
    three = horizon_decision(three_seeds, required_steps=1200)
    assert three["branch"] == "shallow-persists" and three["steps"] == 1200
    # Under the horizon arm's own declaration these same rows are the *anchor*, so
    # the rule refuses instead of reporting a 1200-step branch as the horizon.
    assert horizon_decision(three_seeds)["branch"] == "unavailable"
    one = horizon_decision(
        best_available_summary([r for r in results if r.seed == 42], [42], samples=2_000),
        required_steps=1200,
    )
    assert one["branch"] == "open-floor-limited"
    assert one["depth_floor"] == pytest.approx(2 / 36)


def test_a_design_without_depth_as_a_factor_is_refused():
    outcome = {
        "steps": 1200,
        "crossed_contrast": {
            "available": True,
            "main": {"hidden_size": {"nats": -0.16, "p": 5e-6}, "attention_heads": {"nats": -0.02, "p": 0.02}},
        },
    }
    rule = horizon_decision(outcome, required_steps=1200)
    assert rule["available"] is False and rule["branch"] == "unavailable"
    assert "no crossed contrast with depth" in rule["reason"]
    assert horizon_decision({})["available"] is False


@pytest.mark.parametrize("p", [None])
def test_an_uncomputable_p_is_not_read_as_significant(p):
    rule = horizon_decision(_outcome(+0.05, p, TWO_SEED_FLOOR))
    assert rule["branch"] == "depth-washes-out"
    assert rule["conditions"][0]["value"] == "p not computable"
