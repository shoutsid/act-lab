"""The architecture record's *current-state* surfaces against the artifacts behind them.

`artifacts/ARCH_SEARCH_FINDINGS.md` is two documents in one. The registered arms'
bodies are history and stay verbatim — §2c's own text still calls the head axis a lead
its confirmation would decide, because that is what it read when it was written, and
§9's cost paragraph still quotes the projection as it stood. The header's summary, the
§8 deliverables table and the §10 cost table are the other half: those are *status*
surfaces, and nothing checked them.

By the time they were read, three had drifted:

- the header still listed the horizon question as open, "pre-registered in its own
  NOTES.md and runs on Kaggle T4 rather than the local card", nine days after that arm
  had run, landed and been recorded in §3 and §9 of the same document;
- the deliverables table still called the head-paired confirmation **running** after
  its `decision.md` was committed beside it;
- the audit's coverage was quoted at **119** claims against the 274 it recomputes
  (254 on the `--quick` path the suite runs).

A stale status row has no other symptom. The numbers around it are all checkable —
`verify_record.py` recomputes 274 of them — but "running" is a tense and "unresolved"
is a claim about a state, and neither is a number any audit can disagree with. Each is
therefore pinned here to the artifact that decides it, in both directions: the row must
say the arm is done once its record exists, and must not say so while it does not.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "artifacts" / "ARCH_SEARCH_FINDINGS.md"
ARTIFACTS = ROOT / "artifacts"

PARENT = ARTIFACTS / "arch-zoo-100m-v4-head-paired"
CONFIRM = ARTIFACTS / "arch-zoo-100m-v4-head-paired-confirm"
HORIZON = ARTIFACTS / "arch-zoo-100m-v3-crossed-horizon10x"

pytestmark = pytest.mark.skipif(
    not DOC.is_file(), reason="needs the architecture findings record"
)

SPREAD = re.compile(r"Architecture spread \(best to worst\): ([0-9.]+) nats")
DEPTH = re.compile(r"\|\s*`depth`\s*\|\s*\+([0-9.]+) nats")


def _flat(text: str) -> str:
    """Markdown wraps, so every assertion runs on whitespace-normalised text.

    A line-wise scan is defeated by a wrap, which is how two earlier versions of the
    report's own checks passed vacuously while the claim they guarded was false.
    """
    return " ".join(text.split())


def _doc() -> str:
    return _flat(DOC.read_text())


def _header() -> str:
    """The header only — the summary a reader meets before §1."""
    return _flat(DOC.read_text().split("## 1. ", 1)[0])


def _row(prefix: str) -> str:
    row = next(
        (line for line in DOC.read_text().splitlines() if line.startswith(prefix)),
        None,
    )
    assert row is not None, f"the deliverables table no longer has a {prefix!r} row"
    return _flat(row)


def test_the_header_does_not_call_a_measured_question_unresolved() -> None:
    """The header listed the horizon question as open after §3 had answered it.

    The pin reads the *arm*, not the prose: the depth effect the header quotes must be
    the one the arm's committed `decision.md` records. That is what makes this fail on
    the old text — which quoted no effect at all, only a plan.
    """
    header = _header()
    decision = HORIZON / "decision.md"
    if not decision.is_file():
        assert "+0.0209" not in header, "the header quotes a horizon result with no record"
        return
    effects = DEPTH.findall(decision.read_text())
    assert effects, "the horizon arm's decision no longer states its depth main effect"
    for effect in effects:
        assert f"+{effect}" in header, (
            f"the header does not state the horizon arm's measured depth effect +{effect}"
        )
    assert "runs on Kaggle T4 rather than the local card" not in header, (
        "the header states the horizon arm as a plan while its record is committed"
    )


def test_the_deliverables_row_follows_the_confirmation_arms_artifacts() -> None:
    """The row said **running** for an arm whose `decision.md` sits beside it."""
    row = _row("| Short architecture-zoo runs |")
    decision = CONFIRM / "decision.md"
    if not decision.is_file():
        assert "head-paired confirmation complete" not in row
        return
    assert "head-paired confirmation complete" in row, row
    assert "running" not in row, row
    # And the cost it quotes is the §10 table's figure, so the two tables cannot
    # drift apart — the failure mode that table's own prose warns about.
    cost = next(
        (
            line.split("|")[3].strip()
            for line in DOC.read_text().splitlines()
            if line.startswith("| head-paired confirmation, fresh seeds |")
        ),
        None,
    )
    assert cost is not None, "the §10 cost table no longer lists the confirmation arm"
    assert f"{cost} GPU-h" in row, f"the row does not quote the §10 cost {cost}"


def test_the_decision_row_quotes_the_spreads_the_arms_committed() -> None:
    """Two arms, two spreads, and the row quoted a third that no arm states.

    `verify_record.py` documents the parent's own three-seed spread (0.2126, its
    `§2c v4: spread` claim), while the parent's committed `decision.md` is the pooled
    six-seed render (0.2221) and the confirmation's is its own three (0.2412). A row
    that names one without saying which is how a reader ends up comparing two numbers
    from different designs.
    """
    row = _row("| Decision on whether differences are real |")
    for arm in (PARENT, CONFIRM):
        decision = arm / "decision.md"
        if not decision.is_file():
            continue
        match = SPREAD.search(decision.read_text())
        assert match, f"{arm.name}'s decision no longer states a spread"
        assert match.group(1) in row, (
            f"the decision row does not quote {arm.name}'s committed spread "
            f"{match.group(1)}"
        )
    assert "0.2126" in row, (
        "the decision row must name the parent's own three-seed spread too, since "
        "that is the figure the record audit recomputes"
    )
