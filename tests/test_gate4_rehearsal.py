"""Pin the section 5.43 Gate-4 verdict rehearsal (``gate4-verdict-rehearsal.json``).

The rehearsal applies the real ``compute_verdict`` to synthetic declared-scale fixtures, so
that the whole decision path - every withholding clause, both absent-queue cases, both
short-queue cases, and the label precedence - runs in the configuration the declared run
will be in, without spending GPU time. These tests pin the resulting truth table and, just
as importantly, pin that the record is **labelled synthetic**: nothing here is evidence
about the hypothesis, and a record that could be read as evidence would be worse than no
record.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "artifacts" / "study-summaries" / "gate4-verdict-rehearsal.json"
sys.path.insert(0, str(ROOT / "scripts"))

import v0_verdict  # noqa: E402

# The registered rule, restated independently of the record so the record's own
# expectations cannot drift with it.
REGISTERED = {
    "all_clear": ("PASS", True),
    "cost_over_criterion": ("FAIL", False),
    "harm_guard_rejects": ("PASS_COST_ONLY_HARM_GUARD_REJECTS", False),
    "pool_control_passes": ("PASS_COST_ONLY_POOL_MATCHED_CONTROL_PASSES", False),
    "null_clause_rejects": ("PASS_COST_ONLY_NULL_REJECTS", False),
    "band_miss_at_declared_scale": ("PASS_WITH_QUALIFICATION", False),
    "control_absent": ("PASS_COST_ONLY_CONTROL_MISSING", False),
    "control_short": ("PASS_COST_ONLY_CONTROL_MISSING", False),
    "null_queue_absent": ("PASS_COST_ONLY_NULL_MISSING", False),
    "null_queue_short": ("PASS_COST_ONLY_NULL_MISSING", False),
    "primary_queue_short": ("PASS", False),
    # Section 5.44: this state used to authorize Gate 4; the completed clause 3 withholds
    # it. The tie case is the state the dev-fidelity evidence predicts for the real path.
    "control_ties_adaptive_nobody_demonstrates_an_edge": (
        "PASS_COST_ONLY_NO_DEMONSTRATED_EDGE",
        False,
    ),
    # Distinguishable: adaptive clears the bar, the loss-blind arm does not.
    "adaptive_clears_bar_loss_blind_does_not": ("PASS", True),
    # The proxy's boundary: adaptive 9 vs loss-blind 3 - clearly better, same side of the
    # bar - so it withholds. Recorded as a boundary rather than derived from the wording.
    "adaptive_just_short_of_bar_loss_blind_far_below": (
        "PASS_COST_ONLY_NO_DEMONSTRATED_EDGE",
        False,
    ),
    # Section 5.46's conformance shapes: all three authorize, because the conformance
    # block records a mismatch rather than gating on it. Making it gate would be a new
    # criterion and would need its own amendment.
    "untreated_rebuild_still_authorizes_but_is_flagged": ("PASS", True),
    "prefix_policy_is_flagged_not_authorized_away": ("PASS", True),
    "config_predates_the_fields": ("PASS", True),
    "cost_fail_beats_every_other_clause": ("FAIL", False),
    "harm_guard_beats_pool_control_and_null": (
        "PASS_COST_ONLY_HARM_GUARD_REJECTS",
        False,
    ),
    "pool_control_beats_null_clause": (
        "PASS_COST_ONLY_POOL_MATCHED_CONTROL_PASSES",
        False,
    ),
}


def _record() -> dict:
    return json.loads(RECORD.read_text())


def _by_name() -> dict[str, dict]:
    return {row["variant"]: row for row in _record()["variants"]}


def test_the_record_is_labelled_synthetic_and_validates_machinery_not_the_hypothesis():
    payload = _record()
    assert payload["synthetic"] is True
    assert "validates the decision path, not the hypothesis" in payload["what_this_is"]
    assert "No session ran" in payload["what_this_is"] or "no session ran" in payload[
        "what_this_is"
    ].lower()


def test_every_variant_matches_the_registered_rule():
    payload = _record()
    assert all(payload["built_in_checks"].values()), payload["built_in_checks"]
    rows = _by_name()
    assert set(rows) == set(REGISTERED)
    for name, (label, authorized) in REGISTERED.items():
        row = rows[name]
        assert row["expected_label"] == label
        assert row["observed_label"] == label
        assert row["observed_gate_4_authorized"] is authorized
        assert row["matches_registered_rule"] is True


def test_only_the_clear_distinguishable_and_conformance_cases_authorize_gate_4():
    """Authorization is exactly: every clause clear *and* the quality premise held.

    The tie case and the sub-bar case both withhold (section 5.44). The three section 5.46
    conformance variants authorize on purpose - they exist to show that a recorded
    configuration mismatch is flagged rather than gated - so they are listed here rather
    than filtered out, which is what keeps a future change from widening the set silently.
    """
    rows = _record()["variants"]
    authorizing = [row["variant"] for row in rows if row["observed_gate_4_authorized"]]
    assert authorizing == [
        "all_clear",
        "adaptive_clears_bar_loss_blind_does_not",
        "untreated_rebuild_still_authorizes_but_is_flagged",
        "prefix_policy_is_flagged_not_authorized_away",
        "config_predates_the_fields",
    ]
    assert _by_name()["all_clear"]["observed_exit_code"] == 0


def test_the_conformance_block_records_the_registered_configuration():
    """Section 5.46: what the run said it ran, against what was registered.

    Pinned in both directions - the registered fixtures conform, and the two shapes of a
    non-conforming run (the untreated rebuild, and the pre-5.5 prefix policy) are flagged
    with the fields that differ. The pre-field artifact reads as unrecorded instead, so no
    spent block is re-read as a mismatch.
    """
    payload = _record()
    rows = _by_name()

    untreated = rows["untreated_rebuild_still_authorizes_but_is_flagged"]
    assert untreated["config_conforms"] is False
    assert untreated["config_fields_differing"] == [
        "new_expert_initialization",
        "preserve_router_on_rebuild",
    ]
    assert untreated["observed_gate_4_authorized"] is True

    assert rows["prefix_policy_is_flagged_not_authorized_away"][
        "config_fields_differing"
    ] == ["probe_candidate_policy"]

    predates = rows["config_predates_the_fields"]
    assert predates["config_conforms"] is False
    assert predates["config_fields_differing"] == []
    # Every registered field plus the invocation's probe-batch-count is unrecorded.
    assert len(predates["config_fields_not_recorded"]) == (
        len(v0_verdict.REGISTERED_RUN_CONFIG) + 1
    )

    conforming = [
        row
        for row in payload["variants"]
        if not row["variant_knobs"].get("config_overrides")
        and not row["variant_knobs"].get("omit_config")
    ]
    assert conforming  # the check is not vacuous
    assert all(row["config_conforms"] is True for row in conforming)
    assert payload["built_in_checks"][
        "a_run_that_is_not_the_registered_configuration_is_flagged"
    ]
    assert payload["built_in_checks"]["the_conformance_block_changes_no_authorization"]


def test_the_clause_3_amendment_withholds_in_both_directions():
    """Pin the amendment so neither the code nor the report can drift back.

    Section 5.5 change 3 asks whether the pool-matched arm's quality is indistinguishable
    from the adaptive arm's. Only the 'both arms clear the bar' direction was shipped; the
    'neither clears it' direction authorized Gate 4, which is the state the dev-fidelity
    evidence predicts (section 5.13, 5.39). Section 5.44 completed the clause. Both
    directions are measured here: if either stops withholding, this test fails.
    """
    payload = _record()
    amendments = payload["registered_text_amendments"]
    assert len(amendments) == 1
    item = amendments[0]
    assert item["section"] == "5.5 change 3"
    assert "indistinguishable from the adaptive arm's" in item["registered_text"]
    assert "No adaptive-vs-control comparison existed" in item["shipped_rule_before"]
    assert "PASS, exit 0, Gate 4 AUTHORIZED" in item["demonstrated_before"]["observed_then"]
    assert "NEITHER arm" in item["amendment"]
    assert "strictly tightening" in item["direction"]
    # Each direction the amendment claims to close is measured, and each withholds.
    checked = 0
    for demo in item["demonstrated_after"]:
        row = _by_name()[demo["variant"]]
        assert row["observed_label"] == demo["label"]
        assert row["observed_gate_4_authorized"] is False
        assert row["observed_exit_code"] == 2
        checked += 1
    assert checked == 2
    # And the previous winner is no longer authorizing: the gap is closed, not covered.
    assert _by_name()[
        "control_ties_adaptive_nobody_demonstrates_an_edge"
    ]["observed_gate_4_authorized"] is False
    assert payload["built_in_checks"]["the_clause_3_amendment_withholds_in_both_directions"]


def test_label_precedence_is_verified_not_just_documented():
    """compute_verdict's comment states an order; only one pair of it was tested before."""
    rows = _by_name()
    # Cost failure outranks every other clause.
    assert rows["cost_fail_beats_every_other_clause"]["observed_label"] == "FAIL"
    # The harm guard outranks both the pool-matched control and the null clause.
    assert (
        rows["harm_guard_beats_pool_control_and_null"]["observed_label"]
        == "PASS_COST_ONLY_HARM_GUARD_REJECTS"
    )
    # The pool-matched clause outranks the null clause.
    assert (
        rows["pool_control_beats_null_clause"]["observed_label"]
        == "PASS_COST_ONLY_POOL_MATCHED_CONTROL_PASSES"
    )
    # In each combination the outranked clauses really were live, so the labels above are
    # precedence and not the absence of the other clauses.
    combo = rows["harm_guard_beats_pool_control_and_null"]
    assert combo["harm_guard_clear"] is False
    assert combo["pool_control_passes_gate"] is True
    assert combo["null_clause_rejects"] is True
    both = rows["pool_control_beats_null_clause"]
    assert both["pool_control_passes_gate"] is True
    assert both["null_clause_rejects"] is True


def test_short_queues_withhold_by_label_at_declared_scale():
    """The section 5.42 fix and the section 5.14 rule, at the scale the real run uses."""
    rows = _by_name()
    for name in ("control_absent", "control_short"):
        assert rows[name]["observed_label"] == "PASS_COST_ONLY_CONTROL_MISSING"
        assert rows[name]["pool_control_evaluated"] is False
    for name in ("null_queue_absent", "null_queue_short"):
        assert rows[name]["observed_label"] == "PASS_COST_ONLY_NULL_MISSING"
        assert rows[name]["null_evaluated"] is False


def test_an_incomplete_primary_queue_keeps_the_pass_label_but_withholds_authorization():
    """The completeness requirement lives in ``gate_4_authorized``, not in the label.

    Worth pinning explicitly: this variant exits 0 with an ordinary ``PASS`` and is still
    not authorized, so a reader who keys on the label alone would misread it.
    """
    row = _by_name()["primary_queue_short"]
    assert row["observed_label"] == "PASS"
    assert row["observed_exit_code"] == 0
    assert row["observed_gate_4_authorized"] is False


def test_the_band_and_the_cost_criterion_are_independent_at_declared_scale():
    rows = _by_name()
    # A band miss inside the 5% bar qualifies rather than fails, and stays unauthorized.
    miss = rows["band_miss_at_declared_scale"]
    assert miss["cost_percent"] <= 5.0
    assert miss["in_tolerance_band"] is False
    # A cost failure is a FAIL, not a qualification.
    fail = rows["cost_over_criterion"]
    assert fail["cost_percent"] > 5.0
    assert fail["observed_exit_code"] == 1


def test_the_clause_compares_the_two_arms_through_the_bar_not_through_raw_losses():
    """The shape of the completed clause, checked against the source.

    The two arms are compared by whether each clears the same pre-committed bar against
    its own churn arm - not by comparing per-seed losses across the two queues, which
    would be a different (and undeclared) statistic.
    """
    source = (ROOT / "scripts" / "v0_verdict.py").read_text()
    control = source[source.index("def _control_block") : source.index("def _null_block")]
    # The control block itself still only ever compares its own arm with its own churn arm.
    assert 'row["adaptive_wins"]' not in control
    assert '"adaptive_loss"' not in control
    # The comparison lives in compute_verdict, and it is a comparison of bar outcomes.
    verdict_body = source[source.index("def compute_verdict") :]
    assert "adaptive_has_edge = wins >= POOL_CONTROL_WIN_LIMIT" in verdict_body
    assert 'control_has_edge = bool(control["evaluated"])' in verdict_body
    assert (
        '"indistinguishable": bool(control["evaluated"])'
        in verdict_body
    )
    # No cross-queue per-seed loss comparison was introduced by the amendment.
    assert "control[" + chr(34) + "per_seed" + chr(34) + "]" not in verdict_body


def test_the_rehearsal_fixtures_really_are_at_the_declared_scale():
    """Otherwise the rehearsal would be exercising the section 5.41 defect instead."""
    payload = _record()
    assert all(row["run_is_at_declared_scale"] for row in payload["variants"])
    # At declared scale the run-scale reading must agree with the registered band.
    assert all(
        row["in_tolerance_band"] == row["in_tolerance_band_at_run_scale"]
        for row in payload["variants"]
    )
    declared = payload["declared_parameters"]
    assert declared["total"] == 285_336_576
    assert declared["active"] == 134_341_632
    assert declared["tokens_per_arm"] == 24_576_000


def test_check_mode_is_the_pre_flight_the_runbook_can_call():
    """``--check`` exits 0 on a matching record and non-zero on drift."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verdict_rehearsal.py"), "--check"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
