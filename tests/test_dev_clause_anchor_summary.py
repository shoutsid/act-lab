"""Pin the section 5.66 dev-fidelity clause anchors (``dev-clause-anchor.json``).

The declared Gate-4 verdict withholds on three clauses, and the pre-verdict reading
(``null_clause_power.py``) had an anchor for only one of them. This block measures the
other two, and re-measures the first under the deployed policies, at dev fidelity on
fresh seeds - so the numbers here are read as *anchors* for the clause table, not as
verdicts. Nothing in this file may read them as a Gate 4 result.

Three things are pinned rather than trusted.

The **statistics**, because they are the verdict's own: each clause's win count and
limit must recompute from the stored per-seed losses against the limits imported from
``scripts/v0_verdict.py``. A drifted limit or a win count edited by hand fails here.

The **anchors**, because the firing rate is only meaningful if it comes from the same
function that produced the existing null-clause reading: z is recomputed from the
stored mean and sd, and the rate must equal ``clause_reading(z)`` exactly.

The **gate**, because an anchor read from a block that does not reproduce section
5.40's ordering is not an anchor. The replication count is recomputed from the
per-seed losses, and the built-in check that gates the anchors must be exactly that
boolean.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from dev_clause_anchor_study import (  # noqa: E402
    ANCHOR_CLAUSES,
    CLAUSES,
    KINDS,
    REPLICATION_BAR,
    SEEDS,
    TREATMENT,
    clause_row,
)
from null_clause_power import clause_reading  # noqa: E402
from v0_verdict import (  # noqa: E402
    HARM_GUARD_LOSS_LIMIT,
    NULL_WIN_LIMIT,
    POOL_CONTROL_WIN_LIMIT,
)

SUMMARY = ROOT / "artifacts" / "study-summaries" / "dev-clause-anchor.json"


def _summary() -> dict:
    return json.loads(SUMMARY.read_text())


#: Which per-seed fields a clause compares, so the statistics can be recomputed from
#: the committed rows rather than read back from the clause block.
CLAUSE_FIELDS = {
    "harm_guard_spread": ("spread_adaptive_final_loss", "spread_random_final_loss"),
    "harm_guard_prefix": ("prefix_adaptive_final_loss", "prefix_random_final_loss"),
    "pool_matched_control": ("control_loss_blind_final_loss", "control_random_final_loss"),
    "null_clause_spread": ("spread_adaptive_final_loss", "null_final_loss"),
    "null_clause_prefix": ("prefix_adaptive_final_loss", "null_final_loss"),
}


def test_the_block_ran_its_declared_seeds_and_all_four_sessions() -> None:
    payload = _summary()
    assert payload["report_section"] == "5.66"
    assert payload["kind"] == "dev-clause-anchor"
    assert payload["declared_n"] == 12
    assert payload["seeds"] == list(SEEDS) == list(range(15260, 15272))
    assert all(payload["built_in_checks"].values()), payload["built_in_checks"]
    assert set(payload["config"]["sessions"]) == set(KINDS)
    # The deployment: treatment on every pair session, and the registered policies.
    for kind in ("pair_prefix", "pair_spread", "control_spread"):
        session = payload["config"]["sessions"][kind]
        assert session["preserve_router_on_rebuild"] is TREATMENT["preserve_router_on_rebuild"]
        assert session["new_expert_initialization"] == TREATMENT["new_expert_initialization"]
    assert payload["config"]["sessions"]["pair_prefix"]["probe_candidate_policy"] == "prefix"
    assert payload["config"]["sessions"]["pair_spread"]["probe_candidate_policy"] == "spread"
    assert payload["config"]["sessions"]["pair_spread"]["probe_selection"] == "argmin"
    assert payload["config"]["sessions"]["control_spread"]["probe_selection"] == "random"


def test_each_clause_statistic_recomputes_from_the_stored_rows() -> None:
    payload = _summary()
    rows = payload["per_seed"]
    for name, (adaptive_field, comparison_field) in CLAUSE_FIELDS.items():
        clause = payload["clauses"][name]
        wins = sum(
            1
            for row in rows
            if row[comparison_field] < row[adaptive_field]
        )
        # The comparison arm's win count and its limit, recomputed rather than read.
        assert clause["comparison_wins"] == wins
        assert clause["n"] == len(rows) == 12
        assert clause["fires_at_dev_fidelity"] == (wins >= clause["limit"])
        # Mean, sd and the paired statistics come from the shared comparison path.
        deltas = [row[comparison_field] - row[adaptive_field] for row in rows]
        mean = sum(deltas) / len(deltas)
        assert clause["mean_delta_comparison_minus_adaptive"] == pytest.approx(mean)
        assert clause["paired_statistics"]["mean_delta"] == pytest.approx(mean)
        assert clause["paired_statistics"]["tail"] == "lower"
        assert clause["paired_statistics"]["wins_in_tested_direction"] == wins
    # Exactly the registered five clauses, keyed as the script names them.
    assert set(payload["clauses"]) == set(CLAUSE_FIELDS)
    assert set(CLAUSE_FIELDS) == {name for name, _, _ in CLAUSES}


def test_the_limits_are_the_verdicts_own_constants() -> None:
    limits = _summary()["limits"]
    assert limits["harm_guard_spread"] == HARM_GUARD_LOSS_LIMIT
    assert limits["harm_guard_prefix"] == HARM_GUARD_LOSS_LIMIT
    assert limits["pool_matched_control"] == POOL_CONTROL_WIN_LIMIT
    assert limits["null_clause_spread"] == NULL_WIN_LIMIT
    assert limits["null_clause_prefix"] == NULL_WIN_LIMIT
    # Every withholding clause is the same 10-of-12 sign test; if a future amendment
    # gives one a different limit this study stops describing the verdict.
    assert len(set(limits.values())) == 1
    assert HARM_GUARD_LOSS_LIMIT == POOL_CONTROL_WIN_LIMIT == NULL_WIN_LIMIT


def test_the_anchor_z_and_its_firing_rate_recompute_through_the_shared_function() -> None:
    for clause in _summary()["clauses"].values():
        mean = clause["mean_delta_comparison_minus_adaptive"]
        sd = clause["sample_sd"]
        z = clause["anchor_z"]
        # z is signed so that positive means the comparison arm tends to win.
        assert z == pytest.approx(-mean / sd)
        # The rate is the level table's own arithmetic, not a second calculation.
        assert clause["firing_rate_at_anchor_z"] == clause_reading(z)["p_clause_fires"]
        assert clause["level_without_effect"] == clause_reading(0.0)["p_clause_fires"]
        if clause["comparison_wins"] > 6:
            assert z > 0
        elif clause["comparison_wins"] < 6:
            assert z < 0


def test_the_anchors_are_read_only_from_a_reproduced_ordering() -> None:
    payload = _summary()
    rows = payload["per_seed"]
    null_lowest = sum(
        1
        for row in rows
        if row["null_final_loss"]
        < min(row["prefix_random_final_loss"], row["prefix_adaptive_final_loss"])
    )
    assert null_lowest == payload["replication_check"]["seeds_with_null_lowest"]
    assert payload["replication_check"]["bar"] == REPLICATION_BAR
    assert payload["replication_check"]["reproduces_section_5_40_ordering"] == (
        null_lowest >= REPLICATION_BAR
    )
    # The ordering is the replication claim; the means are quoted beside it because a
    # fresh-seed replication is not bit-equality.
    assert payload["replication_check"]["this_block"][
        "null_minus_adaptive_mean"
    ] == pytest.approx(
        sum(row["null_final_loss"] - row["prefix_adaptive_final_loss"] for row in rows)
        / len(rows)
    )
    # The anchors are gated on it, and that gate is one of the built-in checks.
    assert payload["built_in_checks"][
        "the_anchors_are_read_only_from_a_reproduced_ordering"
    ] == payload["replication_check"]["reproduces_section_5_40_ordering"]


def test_every_anchor_clause_is_read_from_the_declared_primary_policy() -> None:
    """Every anchor is read under ``spread``, the policy the declared primary runs.

    The root is the *session* the contrast is read from, and the two anchors do not share
    one: the harm guard and the do-nothing clause are read from ``pair_spread``, while the
    pool-matched control is read from ``control_spread`` - the loss-blind arm is the
    adaptive arm under random selection, and only that session carries it. So the property
    that holds for all three is the *policy*, not the session name.
    """
    payload = _summary()
    assert payload["anchor_clauses"] == list(ANCHOR_CLAUSES)
    sessions = payload["config"]["sessions"]
    for name in payload["anchor_clauses"]:
        root = payload["clauses"][name]["root"]
        assert root != "null", "the null session has no probe policy to anchor from"
        assert sessions[root]["probe_candidate_policy"] == "spread"
    assert payload["clauses"]["harm_guard_spread"]["root"] == "pair_spread"
    assert payload["clauses"]["null_clause_spread"]["root"] == "pair_spread"
    # The pool-matched root is the one session whose selection rule is the loss-blind one.
    pool_root = payload["clauses"]["pool_matched_control"]["root"]
    assert pool_root == "control_spread"
    assert sessions[pool_root]["probe_selection"] == "random"
    # The prefix roots are the policy check, not the anchor.
    assert set(payload["clauses"]) - set(payload["anchor_clauses"]) == {
        "harm_guard_prefix",
        "null_clause_prefix",
    }
    # A5: the two policies' means are stored and their sign agreement recomputes. The
    # label is the delta's own direction - `null_minus_adaptive` is `null - adaptive` -
    # so the field order here is the label's order, not the reverse. (Both this mapping
    # and the root assertion above were inverted in the first version of this file, and
    # neither could fire until the block produced a summary to read: a pre-registration's
    # own tests run for the first time against the outcome they were written for.)
    policy = payload["policy_check"]
    rows = payload["per_seed"]
    contrasts = {
        "null_minus_adaptive": ("null_final_loss", "prefix_adaptive_final_loss"),
        "null_minus_random": ("null_final_loss", "prefix_random_final_loss"),
        "adaptive_minus_random": ("prefix_adaptive_final_loss", "prefix_random_final_loss"),
    }
    for label, (left_field, right_field) in contrasts.items():
        prefix_mean = sum(row[left_field] - row[right_field] for row in rows) / len(rows)
        assert policy[label]["prefix_mean"] == pytest.approx(prefix_mean)
        assert policy[label]["signs_agree"] == (
            (policy[label]["prefix_mean"] >= 0) == (policy[label]["spread_mean"] >= 0)
        )
    # The three contrasts are what they say: the null arm's delta is negative on both
    # comparisons (it has the *lower* loss) and the treatment's edge over churn is signed
    # the way the A1 rows report it.
    assert policy["null_minus_adaptive"]["prefix_mean"] < 0
    assert policy["null_minus_random"]["prefix_mean"] < 0
    assert policy["adaptive_minus_random"]["signs_agree"] is True


def test_the_verdicts_own_clause_booleans_recompute_from_the_committed_rows() -> None:
    """The post-outcome reading, pinned from the rows rather than from the clause block.

    Which arm counts as a *win* differs between the two tables this study produces:
    ``clause_row`` records the comparison arm winning (churn for the harm guard and the
    pool-matched row, null for the do-nothing clause), while ``v0_verdict.py``'s section
    5.5 clause fires on the loss-blind arm winning. Both directions are recomputed here
    from the same committed losses, so a flip in either one fails rather than printing a
    table that reads backwards.
    """
    reading = _summary()["verdict_reading_at_dev_fidelity"]
    rows = _summary()["per_seed"]

    def wins(winner: str, loser: str) -> int:
        return sum(1 for row in rows if row[winner] < row[loser])

    adaptive_vs_churn = wins("spread_adaptive_final_loss", "spread_random_final_loss")
    loss_blind_vs_churn = wins("control_loss_blind_final_loss", "control_random_final_loss")
    adaptive_vs_null = wins("spread_adaptive_final_loss", "null_final_loss")

    assert reading["adaptive_wins_vs_churn"] == adaptive_vs_churn == 5
    assert reading["loss_blind_wins_vs_churn"] == loss_blind_vs_churn == 5
    assert reading["adaptive_wins_vs_null"] == adaptive_vs_null == 0

    # The same three counts in the other direction, from the clause block. This is the
    # crossing the summary exists to make explicit.
    clauses = _summary()["clauses"]
    assert clauses["harm_guard_spread"]["comparison_wins"] == wins(
        "spread_random_final_loss", "spread_adaptive_final_loss"
    )
    assert clauses["pool_matched_control"]["comparison_wins"] == wins(
        "control_random_final_loss", "control_loss_blind_final_loss"
    )
    assert clauses["null_clause_spread"]["comparison_wins"] == wins(
        "null_final_loss", "spread_adaptive_final_loss"
    )
    # With no ties the two directions partition the block, so neither can be flipped
    # without the other contradicting it.
    assert (
        adaptive_vs_churn
        + clauses["harm_guard_spread"]["comparison_wins"]
        == len(rows)
    )

    # The booleans and the label are the verdict's own arithmetic and its own precedence.
    assert reading["harm_guard_clear"] == (len(rows) - adaptive_vs_churn < HARM_GUARD_LOSS_LIMIT)
    assert reading["control_passes_gate"] == (loss_blind_vs_churn >= POOL_CONTROL_WIN_LIMIT)
    assert reading["null_clause_rejects"] == (
        len(rows) - adaptive_vs_null >= NULL_WIN_LIMIT
    )
    assert reading["indistinguishable"] == (
        not reading["control_passes_gate"]
        and adaptive_vs_churn < POOL_CONTROL_WIN_LIMIT
    )
    if not reading["harm_guard_clear"]:
        expected = "PASS_COST_ONLY_HARM_GUARD_REJECTS"
    elif reading["control_passes_gate"]:
        expected = "PASS_COST_ONLY_POOL_MATCHED_CONTROL_PASSES"
    elif reading["indistinguishable"]:
        expected = "PASS_COST_ONLY_NO_DEMONSTRATED_EDGE"
    elif reading["null_clause_rejects"]:
        expected = "PASS_COST_ONLY_NULL_CLAUSE_REJECTS"
    else:
        expected = "no quality-side clause fires"
    assert reading["label_if_the_declared_block_read_like_this"] == expected

    # The reading is recorded as a reading: the cost side is not evaluated here, and this
    # block adjudicates no branch of the declared verdict.
    assert "cost side is not evaluated" in reading["what_this_is"]
    assert _summary()["what_this_is_not"].startswith("Not a Gate 4 result")


def test_the_direction_of_the_pool_matched_clause_is_not_interchangeable() -> None:
    """The constructed failure: read the section 5.5 clause the way the row is written.

    Taking the pool-matched row's own firing direction (churn beating the loss-blind arm
    on >= 10) as the verdict's section 5.5 gate re-labels the block, because the two
    readings are complements of one another on a partition. A reader who used one table
    to check the other would therefore get the opposite branch, which is why both
    directions are committed together.
    """
    reading = _summary()["verdict_reading_at_dev_fidelity"]
    row_direction_fires = _summary()["clauses"]["pool_matched_control"][
        "fires_at_dev_fidelity"
    ]
    assert row_direction_fires is False
    assert reading["control_passes_gate"] is False
    # They are not the same statement: one says the loss-blind arm did not clear the bar,
    # the other says its churn arm did not either. Both are false here, and the block's
    # win counts are what separate them.
    assert reading["loss_blind_wins_vs_churn"] != _summary()["clauses"][
        "pool_matched_control"
    ]["comparison_wins"]
    assert (
        reading["loss_blind_wins_vs_churn"]
        + _summary()["clauses"]["pool_matched_control"]["comparison_wins"]
        == 12
    )


def test_the_loss_blind_arm_is_invariant_to_the_probe_policy_and_selection() -> None:
    """The pool-matched arm's defining property, measured rather than assumed.

    It never probes, so its trajectory must be independent of ``probe_candidate_policy``
    and ``probe_selection``. The three pair roots differ in exactly those two knobs, so
    the churn arm's losses are compared with *bit* equality - not ``approx`` - on every
    seed. If the loss-blind arm ever started reading the probe, the policies would move
    it and this fails.
    """
    payload = _summary()
    rows = payload["per_seed"]
    for row in rows:
        assert row["prefix_random_final_loss"] == row["spread_random_final_loss"]
        assert row["spread_random_final_loss"] == row["control_random_final_loss"]
    invariance = payload["loss_blind_invariance"]
    assert invariance["churn_identical_across_pair_roots"] is True
    assert invariance["seeds"] == len(rows) == 12
    # And the two pair roots really do differ in the knobs it is invariant to, so the
    # equality above is not two runs of one configuration.
    sessions = payload["config"]["sessions"]
    assert (
        sessions["pair_prefix"]["probe_candidate_policy"]
        != sessions["pair_spread"]["probe_candidate_policy"]
    )
    assert sessions["control_spread"]["probe_selection"] != sessions["pair_spread"][
        "probe_selection"
    ]


def test_the_instruments_power_crossings_are_what_the_report_quotes() -> None:
    """The 10-of-12 bar's power curve, pinned where section 5.66 quotes it.

    The section reads its anchor column by asking how large an effect the bar needs, and
    that is a property of the rule and the noise alone - so it is pinned rather than left
    as prose. The two crossings are what the section means by "a large-effect instrument";
    an amendment to the limit or to the sign test moves them and fails here.
    """

    def crossing(target: float) -> float:
        lo, hi = 0.0, 5.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if clause_reading(mid)["p_clause_fires"] < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    assert crossing(0.5) == pytest.approx(0.783, abs=5e-4)
    assert crossing(0.8) == pytest.approx(1.123, abs=5e-4)
    # The two points the anchor rows sit at, for the same reason: one clause's anchor is
    # taken where its effect is real and the other two where it is absent.
    assert clause_reading(1.489)["p_clause_fires"] == pytest.approx(0.9561, abs=5e-5)
    assert clause_reading(0.039)["p_clause_fires"] == pytest.approx(0.0249, abs=5e-5)


def test_each_committed_clause_block_is_exactly_its_own_recomputation() -> None:
    """The constructed failure: an anchor or a win count edited without its rows.

    ``clause_row`` produced this block from the committed per-seed losses, so the
    committed block is compared to that function's output whole. A hand-edited win
    count, a planted mean, a relaxed limit or a firing rate computed some other way
    all break the equality - the anchors cannot be adjusted after the fact, which is
    the reason the readings were registered before the first session ran.
    """
    payload = _summary()
    rows = payload["per_seed"]
    for name, root, comparison in CLAUSES:
        adaptive_field, comparison_field = CLAUSE_FIELDS[name]
        recomputed = clause_row(
            name,
            root,
            "null (do nothing)" if comparison == "null" else "churn (random)",
            [row[adaptive_field] for row in rows],
            [row[comparison_field] for row in rows],
            payload["limits"][name],
        )
        assert recomputed == payload["clauses"][name]
    # And a planted anchor does not survive even the cheap check above.
    clause = payload["clauses"]["pool_matched_control"]
    planted = dict(clause, mean_delta_comparison_minus_adaptive=0.0)
    assert planted != clause
    assert clause_reading(0.0)["p_clause_fires"] != clause_reading(
        clause["anchor_z"]
    )["p_clause_fires"]
