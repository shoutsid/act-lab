"""Pin the section 5.40 deployed-configuration block (``real-path-deployed-vs-null.json``).

The block asked whether adaptive selection beats doing nothing on real text with the
*deployed* treatment (``preserve_router_on_rebuild`` + ``clone``) on twelve fresh
seeds, and it is the only real-text test of the value-creation claim that the project
actually ships. Its verdict is branch (b): the null arm has the lowest loss on 12/12
seeds, so section 5.13's closure is re-established on the deployed configuration and
the value-creation claim gets no real-text support.

Two things therefore have to be pinned rather than trusted.

The **convention**, because this program has now inverted it twice. Deltas are
``null - arm``, so negative means the *mutating arm* is worse - the null arm, doing
nothing, has the lower loss. That is asserted here convention-free, from the per-seed
losses, so a flipped tail in the runner cannot hide: the argmin of the three arms is
the null on every seed.

The **derivation**, because the first run of the block tested P1 on the churn-harm
tail (``lower``) as well, which made ``established`` mean "the null is significantly
better than adaptive" - the exact opposite of P1 passing - and printed branch (a),
the real-text leg reopening. The numbers in the artifact were right; only the label
was wrong. The tail of each primary is asserted, and the branch is re-derived from
the stored comparisons, so that run's artifact would fail here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from churn_harm_study import compare  # noqa: E402
from deployed_vs_null_study import (  # noqa: E402
    ALPHA,
    PER_PRIMARY_ALPHA,
    SEEDS,
    branch_from_primaries,
)

SUMMARY = ROOT / "artifacts" / "study-summaries" / "real-path-deployed-vs-null.json"


def _summary() -> dict:
    return json.loads(SUMMARY.read_text())


def test_the_block_ran_its_declared_seeds_on_both_registered_sessions() -> None:
    payload = _summary()
    assert payload["report_section"] == "5.40"
    assert payload["declared_n"] == 12
    assert payload["seeds"] == list(range(15240, 15252))
    assert payload["seeds"] == list(SEEDS)
    assert payload["per_primary_alpha"] == 0.025
    assert payload["alpha"] == ALPHA
    assert all(payload["built_in_checks"].values()), payload["built_in_checks"]
    # The registration's own post-hoc table, on the spent seeds, still recomputes.
    assert all(payload["mechanism_diagnostic"]["matches_registration"].values())


def test_the_null_arm_has_the_lowest_loss_on_every_fresh_seed() -> None:
    """The convention, read off the losses rather than off any delta's sign."""
    payload = _summary()
    rows = payload["per_seed"]
    assert len(rows) == 12
    for row in rows:
        losses = {
            "null": row["null_final_loss"],
            "random": row["random_final_loss"],
            "adaptive": row["adaptive_final_loss"],
        }
        assert min(losses, key=losses.__getitem__) == "null"
        # So every delta is negative for both mutating arms, and the null's win is
        # the same fact as the sign of the stored comparison.
        assert row["null_minus_adaptive"] < 0
        assert row["null_minus_random"] < 0
    assert all(row["null_events"] == 4 for row in rows)
    assert all(row["adaptive_mutations"] == 3 for row in rows)
    assert all(row["null_search_flops"] == 0 for row in rows)


def test_each_primary_is_tested_on_its_own_tail() -> None:
    """P1 and P2 are opposite claims about one delta, so one tail cannot serve both."""
    comparisons = _summary()["comparisons"]
    assert comparisons["null_minus_adaptive"]["tail"] == "upper"
    assert comparisons["null_minus_random"]["tail"] == "lower"
    assert comparisons["adaptive_minus_random"]["tail"] == "upper"
    tails = _summary()["primary_tails"]
    assert tails["null_minus_adaptive"].startswith("upper")
    assert tails["null_minus_random"].startswith("lower")


def test_the_verdict_is_the_registered_branch_and_its_prediction_was_met() -> None:
    payload = _summary()
    p1 = payload["comparisons"]["null_minus_adaptive"]
    p2 = payload["comparisons"]["null_minus_random"]
    red_flag = payload["comparisons"]["adaptive_minus_random"]

    # P1 fails: adaptive does not beat doing nothing (0/12 seeds in that direction).
    assert (p1["mean_delta"], p1["wins_in_tested_direction"]) == (
        pytest.approx(-0.13554, abs=5e-6),
        0,
    )
    assert p1["permutation_p"] == 1.0
    assert not p1["established"]
    # P2 passes: churn harm reproduces on fresh seeds, 12/12, at the block's own bar.
    assert p2["mean_delta"] == pytest.approx(-0.14566, abs=5e-6)
    assert p2["wins_in_tested_direction"] == 12
    assert p2["permutation_p"] <= PER_PRIMARY_ALPHA
    assert p2["established"]
    # No red flag: the two mutating arms are tied (6/12, p = 0.61), as the
    # diagnostic on spent seeds saw them.
    assert not red_flag["established"]

    assert branch_from_primaries(False, True, False) == payload["branch"]
    assert payload["branch"] == "b_closure_holds_under_the_deployed_configuration"
    assert payload["prediction_met"] == {
        "null_minus_adaptive": "fails",
        "null_minus_random": "passes (negative at the bar)",
    }
    assert payload["registered_prediction"]["null_minus_adaptive"] == "fails"


def test_the_run_that_reported_the_reopening_leg_would_fail_here() -> None:
    """The inversion, constructed: P1 on the churn-harm tail reads as established.

    The first run of this block compared ``null - adaptive`` on ``lower``, so the
    same twelve losses produced ``established = True`` for P1 and branch (a) - a
    headline claim of real-text value creation from a block where the null arm won
    every seed. Recomputing that tail from the stored per-seed losses is the
    constructed failure: it is established, it is the opposite reading, and every
    assertion above about the stored artifact is what distinguishes them.
    """
    payload = _summary()
    rows = payload["per_seed"]
    null = [row["null_final_loss"] for row in rows]
    adaptive = [row["adaptive_final_loss"] for row in rows]
    wrong_tail = compare("null_minus_adaptive", null, adaptive, PER_PRIMARY_ALPHA, "lower")
    assert wrong_tail["established"]
    assert wrong_tail["mean_delta"] == pytest.approx(
        payload["comparisons"]["null_minus_adaptive"]["mean_delta"]
    )
    # With that flag, the derivation would have printed the reopening branch - which
    # is why the branch is a function of the tested tail and not of the losses alone.
    assert branch_from_primaries(True, True, False) == "a_real_text_leg_reopens"
    assert payload["branch"] != "a_real_text_leg_reopens"
