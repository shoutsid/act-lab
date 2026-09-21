"""Tests for the record audit: the phase's documented numbers against the artifacts.

The suite tests what the code does; this tests whether the record describes what the
code did, which is a different failure mode. It is worth a test rather than a habit
because three documented numbers were wrong when checked by hand - a cost total that
matched neither of its columns, a promotion threshold whose cited measurement does not
produce it, and a budget ratio quoted from one draw seed as though it were typical -
and none of those broke a single test.

The audit runs in quick mode here (skipping the 200-draw budget scan) so the suite
stays fast; the scan's own claims are checked by hand when the record changes. One
test needs the full path rather than the quick one - the record states how many claims
the audit has, and the suite's own count is the quick path's - so the full audit is
cached per process and costs the ~2s it differs from the quick one by, once.
"""

from __future__ import annotations

import functools
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


@functools.lru_cache(maxsize=None)
def _audit_quick() -> Audit:
    return run_audit(ARTIFACTS, quick=True)


@functools.lru_cache(maxsize=None)
def _audit_full() -> Audit:
    return run_audit(ARTIFACTS, quick=False)


def test_every_documented_number_still_agrees_with_the_artifacts():
    audit = _audit_quick()
    assert audit.claims, "the audit found nothing to check"
    failures = [(claim.what, claim.documented, claim.computed) for claim in audit.failures]
    assert not failures, f"the record and the artifacts disagree: {failures}"


def test_the_audit_covers_the_headline_claims_rather_than_a_sample():
    """A verification script that quietly stopped checking half the record is worse
    than none, because it reports agreement."""
    audit = _audit_quick()
    checked = " | ".join(claim.what for claim in audit.claims)
    for expected in (
        "§2c pooled: depth",
        "§2c pooled: interaction at its 6-seed floor",
        "§2c v4: head contrast",
        "§2c v4: lands on the pre-registered 'width is the lever' condition",
        "§2c v4-confirm: candidate set byte-identical to the parent's",
        "§4 400: seed std",
        "§4a: the eventual winner's place at 400 steps",
        "§9a: 80% power is reached at 0.30 nats, and not at 0.20",
        "§9a: 13 per set rejects a true null only at the nominal rate",
        "§9a: a 0.60-nat better winner is invisible at 13 per set",
        "§9a: the same statistic sees a 0.30-nat set-wide shift",
        "§9a population: cells within +/-5% of 78M",
        "§9a v1: hidden spread percentile",
        "§9a v2: depth spread percentile",
        "§9a: its total parameters",
        "§9a: its mean candidate",
        "§9a: the random arm's plan is 39 runs",
        "§10 loop: the driver's plan for this loop is two stages",
        "§10 loop: the loop at a 12,000-step horizon, on Kaggle",
        "§10 total: runs",
        "README: the fully dense design's rank",
        "promotion threshold at 2 seeds / 400 steps",
    ):
        assert expected in checked, expected
    assert len(audit.claims) >= 60


def test_the_coverage_form_is_the_reason_the_mean_is_the_primary_statistic():
    """§9a pre-registers the set *mean* because the coverage form cannot be seen.

    If that claim is ever wrong, the pre-registration is wrong in a way no other
    test would notice: the arm would be measuring the statistic it can see rather
    than the one it declared. Both halves matter - that the winner form is inert,
    and that the same statistic is not simply a dead test - so a mutant that makes
    either one pass alone is caught.
    """
    audit = run_audit(ARTIFACTS, quick=True)
    claims = {claim.what: claim for claim in audit.claims}
    inert = claims["§9a: a 0.60-nat better winner is invisible at 13 per set"]
    contrast = claims["§9a: the same statistic sees a 0.30-nat set-wide shift"]
    assert inert.ok and contrast.ok
    assert "0.03" in inert.detail, inert.detail
    assert "0.9" in contrast.detail, contrast.detail


def test_the_findings_record_states_the_coverage_the_audit_actually_has() -> None:
    """The architecture record quoted **119** claims against the 274 it runs.

    A coverage number is a claim about the check itself, which makes it the one claim
    the check cannot disagree with — and this one had rotted by more than 2x, reading
    as confidence the audit does not have. Both numbers the record states are pinned:
    the full count, and the `--quick` count the suite's other tests run.
    """
    text = (ROOT / "artifacts" / "ARCH_SEARCH_FINDINGS.md").read_text()
    full = len(_audit_full().claims)
    quick = len(_audit_quick().claims)
    assert f"**{full} load-bearing claims**" in text, (
        f"the record no longer states the audit's full claim count ({full})"
    )
    assert f"**{quick}** of them on the `--quick` path" in text, (
        f"the record no longer states the quick path's claim count ({quick})"
    )


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
    # A percentile is a rank the tool renders to a whole percentage, so the audit
    # compares that rendering: 15.7 is p16, and a rank that moved within its own
    # rendering is not a drift the record could state.
    assert Claim("p16", 16, round(15.7), "§9a").ok
    assert not Claim("p16", 16, round(20.7), "§9a").ok
    # And a value quoted to one decimal rounds: the matched draw's depth mean is
    # 7.846 and the record's 7.8 is that rounded, so half a unit in the last quoted
    # place is the right tolerance and the default half a percent is too tight.
    assert not Claim("depth mean", 7.8, 7.846153846153846, "§9a").ok
    assert Claim("depth mean", 7.8, 7.846153846153846, "§9a", tolerance=0.05 / 7.8).ok
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
