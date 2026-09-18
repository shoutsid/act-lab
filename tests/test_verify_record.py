"""Tests for the record audit: the phase's documented numbers against the artifacts.

The suite tests what the code does; this tests whether the record describes what the
code did, which is a different failure mode. It is worth a test rather than a habit
because three documented numbers were wrong when checked by hand - a cost total that
matched neither of its columns, a promotion threshold whose cited measurement does not
produce it, and a budget ratio quoted from one draw seed as though it were typical -
and none of those broke a single test.

The audit runs in quick mode here (skipping the 200-draw budget scan) so the suite
stays fast; the scan's own claims are checked by hand when the record changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from verify_record import Audit, Claim, run_audit  # noqa: E402

ARTIFACTS = ROOT / "artifacts"

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "arch-zoo-100m-v3-crossed").is_dir(),
    reason="needs the phase's recorded artifacts",
)


def test_every_documented_number_still_agrees_with_the_artifacts():
    audit = run_audit(ARTIFACTS, quick=True)
    assert audit.claims, "the audit found nothing to check"
    failures = [(claim.what, claim.documented, claim.computed) for claim in audit.failures]
    assert not failures, f"the record and the artifacts disagree: {failures}"


def test_the_audit_covers_the_headline_claims_rather_than_a_sample():
    """A verification script that quietly stopped checking half the record is worse
    than none, because it reports agreement."""
    audit = run_audit(ARTIFACTS, quick=True)
    checked = " | ".join(claim.what for claim in audit.claims)
    for expected in (
        "§2c pooled: depth",
        "§2c pooled: interaction at its 6-seed floor",
        "§2c v4: head contrast",
        "§2c v4: lands on the pre-registered 'width is the lever' condition",
        "§4 400: seed std",
        "§4a: the eventual winner's place at 400 steps",
        "§10 total: runs",
        "README: the fully dense design's rank",
        "promotion threshold at 2 seeds / 400 steps",
    ):
        assert expected in checked, expected
    assert len(audit.claims) >= 60


def test_an_exact_count_must_match_exactly():
    """This is the failure that hid in §10: a total that matched neither column.

    191 against 196 is 2.6% off, which any relative tolerance worth having would wave
    through, so counts are compared exactly and only quoted statistics get slack.
    """
    assert Claim("runs", 196, 196, "§10").ok
    assert not Claim("runs", 191, 196, "§10").ok
    # A statistic quoted to fewer digits than it is computed to is not a disagreement.
    assert Claim("spread", 0.2462, 0.24623161554336548, "§4").ok
    assert not Claim("spread", 0.2462, 0.3001, "§4").ok
    # Bools are not numbers: True == 1 would let a verdict comparison pass silently.
    assert Claim("signs agree", True, True, "§2c").ok
    assert not Claim("signs agree", False, True, "§2c").ok
    assert not Claim("best", "a", "b", "§2c").ok


def test_the_audit_reports_failures_rather_than_raising_them():
    audit = Audit()
    audit.check("a documented number", 1.0, 1.0, "§1")
    audit.check("a wrong one", 1.0, 2.0, "§1")
    assert len(audit.failures) == 1
    assert audit.failures[0].what == "a wrong one"
