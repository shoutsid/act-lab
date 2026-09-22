"""Pin the pre-data reading of the section 5.14 do-nothing clause.

The clause decides whether Gate 4 is authorized, and how much it can decide is a property
of the rule and the noise rather than of the run - so it was measured before the declared
block's null queue exists, and the reading is pinned here so the verdict cannot be
re-narrated after it lands.

Two numbers carry it. Under **no effect** the clause fires with probability 79/4096 =
1.93%: a rule that withholds on >=10 of 12 adverse seeds almost never withholds on noise,
so a cleared clause is not evidence of value (it clears 98.07% of no-effect blocks). At
the **dev-fidelity effect** section 5.40 measured - the only real-text effect this program
has with the treatment in place, z = 1.20 - it fires with probability 0.8483, so a
withheld Gate 4 is strong evidence rather than a coin flip.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import v0_verdict  # noqa: E402
from null_clause_power import (  # noqa: E402
    DECLARED_SEEDS,
    Z_GRID,
    build_report,
    clause_reading,
    dev_fidelity_anchor,
)

ARTIFACT = ROOT / "artifacts" / "null-clause-power.json"


def test_the_clause_is_read_from_the_verdicts_own_constants() -> None:
    """The analysis must describe the rule the verdict applies, not a copy of it."""
    assert DECLARED_SEEDS == len(v0_verdict.DECLARED_SEEDS) == 12
    assert v0_verdict.NULL_WIN_LIMIT == 10
    # Every withhold clause is the same 10-of-12 sign test on the adaptive arm, differing
    # only in the comparison arm - which is why this one script can price all three, and
    # why a future amendment to any one limit has to break this analysis.
    assert v0_verdict.HARM_GUARD_LOSS_LIMIT == v0_verdict.NULL_WIN_LIMIT
    assert v0_verdict.POOL_CONTROL_WIN_LIMIT == v0_verdict.NULL_WIN_LIMIT


def test_all_three_withhold_clauses_are_the_same_rule_on_different_arms() -> None:
    payload = json.loads(ARTIFACT.read_text())
    assert payload["clauses_share_one_limit"] is True
    assert set(payload["clauses"]) == {
        "harm_guard",
        "pool_matched_control",
        "null_clause",
    }
    for row in payload["clauses"].values():
        assert row["fires_when_arm_beats_adaptive_on_at_least"] == 10
        assert row["level_under_no_effect"] == 79 / 4096
    # Union bound, since the three are positively dependent rather than independent.
    assert payload["union_bound_level"] == 3 * 79 / 4096
    assert math.isclose(payload["union_bound_level"], 0.057861328125, abs_tol=1e-12)


def test_the_clauses_are_not_redundant_on_the_one_block_that_exists() -> None:
    """Section 5.40's dev-fidelity seeds, read clause by clause.

    The null clause fires there (the null arm beats adaptive on 12/12) while the harm
    guard does not (the random arm beats adaptive on 6/12): a verdict can withhold on one
    clause and clear another, so the clauses are three conditions rather than one. The
    third arm that block lacks is measured on the section 5.66 block, and its reading is
    attached from that record rather than restated here.
    """
    read = json.loads(ARTIFACT.read_text())["dev_fidelity_clause_read"]
    assert read["declared_block"] is False
    assert (read["null_beats_adaptive"], read["null_clause_fires"]) == (12, True)
    assert (read["random_beats_adaptive"], read["harm_guard_fires"]) == (6, False)
    assert read["pool_matched_control"].startswith("not present in this block")
    # And the counts agree with section 5.40's own comparison, which counts the seeds
    # favouring random at 6/12.
    comparison = json.loads(
        (ROOT / "artifacts/study-summaries/real-path-deployed-vs-null.json").read_text()
    )["comparisons"]["adaptive_minus_random"]
    assert comparison["wins_in_tested_direction"] == 6


def test_every_withhold_clause_has_its_own_dev_fidelity_anchor() -> None:
    """The gap this artifact used to state: two of three clauses had no anchor.

    The clause table was anchored on the null clause's contrast alone. Section 5.66
    measured all three at the same fidelity on fresh seeds under the declared policies,
    and each anchor here must be that measurement - recomputed from the study's record,
    with the firing rate computed through this module's own ``clause_reading``.
    """
    payload = json.loads(ARTIFACT.read_text())
    anchors = payload["dev_fidelity_anchors_by_clause"]
    assert set(anchors) == {"harm_guard", "pool_matched_control", "null_clause"}
    study = json.loads(
        (ROOT / "artifacts/study-summaries/dev-clause-anchor.json").read_text()
    )
    blocks = {
        "harm_guard": "harm_guard_spread",
        "pool_matched_control": "pool_matched_control",
        "null_clause": "null_clause_spread",
    }
    for clause, block in blocks.items():
        row = study["clauses"][block]
        anchor = anchors[clause]
        assert anchor["comparison_wins"] == row["comparison_wins"]
        assert anchor["limit"] == row["limit"] == v0_verdict.NULL_WIN_LIMIT
        assert anchor["n"] == 12
        assert anchor["fires_at_dev_fidelity"] == (
            row["comparison_wins"] >= row["limit"]
        )
        assert anchor["mean_delta_comparison_minus_adaptive"] == pytest.approx(
            row["mean_delta_comparison_minus_adaptive"]
        )
        assert anchor["z"] == pytest.approx(row["anchor_z"])
        # The rate is this module's arithmetic on the study's measurement, not a copy.
        assert anchor["p_clause_fires_at_z"] == clause_reading(anchor["z"])["p_clause_fires"]
        assert anchor["level_under_no_effect"] == 79 / 4096
        assert "dev-clause-anchor.json" in anchor["source"]
        assert "5.66" in anchor["source"]
    # The anchors are contrasts, not verdicts: nothing in them authorizes Gate 4.
    assert payload["dev_fidelity_anchors_by_clause"]
    assert "three measured analogues" in payload["readings"]["anchors_by_clause"]


def test_the_clause_fires_on_79_of_4096_no_effect_blocks() -> None:
    """The level, in closed form: a withhold rule this lenient decides little on noise."""
    no_effect = clause_reading(0.0)
    assert no_effect["p_seed_favours_null"] == 0.5
    assert math.isclose(no_effect["p_clause_fires"], 79 / 4096, abs_tol=1e-12)
    assert math.isclose(no_effect["p_clause_fires"], 0.019287109375, abs_tol=1e-12)
    assert math.isclose(1.0 - no_effect["p_clause_fires"], 0.980712890625, abs_tol=1e-12)
    # Equivalently: clearing the clause needs only 3 of 12 seeds favouring adaptive.
    assert DECLARED_SEEDS - v0_verdict.NULL_WIN_LIMIT + 1 == 3


def test_the_dev_fidelity_effect_fires_the_clause_about_85_percent_of_the_time() -> None:
    anchor = dev_fidelity_anchor()
    # Section 5.40's own rows: the null arm ahead on 12/12 fresh seeds.
    assert anchor["null_wins_observed"] == 12
    assert anchor["effect_nats_null_minus_adaptive"] == -0.1355433166027069
    assert anchor["sample_sd"] == 0.11287968831893397
    assert math.isclose(anchor["z"], 1.2007768503022296, abs_tol=1e-12)
    reading = clause_reading(anchor["z"])
    assert math.isclose(reading["p_clause_fires"], 0.8483, abs_tol=5e-5)
    assert math.isclose(1.0 - reading["p_clause_fires"], 0.1517, abs_tol=5e-5)
    # The power curve is monotone, and the planning bar is cleared before z = 1.5.
    fires = [clause_reading(z)["p_clause_fires"] for z in Z_GRID]
    assert fires == sorted(fires)
    assert clause_reading(1.5)["p_clause_fires"] > 0.8
    assert clause_reading(0.5)["p_clause_fires"] < 0.5


def test_the_committed_artifact_is_what_this_script_produces_now() -> None:
    """Whole-artifact equality, so a hand-restated field cannot hide.

    Per-field assertions against literals would pass with a stale artifact while the
    code that wrote it computed something else - the pass bar was already restated by
    hand once in this file's construction and survived the field-by-field version.
    """
    payload = json.loads(ARTIFACT.read_text())
    assert payload == build_report()
    assert payload["report_section"] == "5.14"
    assert payload["clause"]["null_win_limit"] == v0_verdict.NULL_WIN_LIMIT
    assert payload["clause"]["adaptive_wins_required_to_clear"] == 3
    assert payload["clause"]["source"].endswith("(imported, not restated)")
    assert payload["level_under_no_effect"]["p_clause_fires"] == 79 / 4096
    assert "3 of 12" in payload["readings"]["what_a_pass_is_worth"]
    assert "0.8483" in payload["readings"]["modal_verdict"]
