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


def test_exactly_one_variant_authorizes_gate_4():
    rows = _record()["variants"]
    authorizing = [row["variant"] for row in rows if row["observed_gate_4_authorized"]]
    assert authorizing == ["all_clear"]
    # And that one is an ordinary PASS at exit 0.
    assert _by_name()["all_clear"]["observed_exit_code"] == 0


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
