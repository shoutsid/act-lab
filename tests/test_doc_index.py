"""The standing docs are indexed, and the index and cross-references resolve.

The project carries a plan, a status log, a research report, a runbook and a set of
per-arm notes - tens of thousands of lines across dozens of files, three of them
thousands of lines each - and until now nothing said which one owns which question.
A reader arriving at the front door saw one of them linked and had no way to tell
the prescriptive plan from the append-only log.

This module makes the map itself checkable rather than trusted. It asserts that
every standing document at the repository root is named in the README's
Documentation section, that every document that section names exists, and that
every cross-document section reference - ``RESEARCH_REPORT.md`` section 5.47, or
"plan section 100" - resolves to a heading that is actually there.

These are link checks, not statistics checks. The numbers behind the documents are
owned by ``scripts/verify_record.py`` and the per-study pinning tests.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
REPORT = ROOT / "RESEARCH_REPORT.md"
PLAN = ROOT / "complete_implementation_plan.md"

INDEX_HEADING = "## Documentation"

# Agent tooling rather than project documentation, and gitignored, so it is absent
# from a fresh clone. Excluded by name so the check means the same thing either way.
NOT_PROJECT_DOCS = {"AGENTS.md"}

# Directories that hold vendored or cache markdown, never project documentation.
SKIPPED_DIRS = {".git", ".venv", ".pytest_cache"}

BACKTICKED_MD = re.compile(r"`([^`\n]+\.md)`")

# The document kinds that live beside an arm or a benchmark. A new one of these
# appearing under artifacts/ has to be placed on the map, because these are the files
# a reader lands on by browsing rather than by being told where to look.
PER_ARM_DOCS = {
    "NOTES.md",
    "decision.md",
    "analysis.md",
    "FINDINGS.md",
    "PERFORMANCE_NOTES.md",
    "SCREENING_AUDIT.md",
}
# `RESEARCH_REPORT.md` §5.47 / RESEARCH_REPORT.md §5.47
REPORT_REF = re.compile(r"RESEARCH_REPORT\.md`?\s*§\s*([0-9]+(?:\.[0-9]+)*)")
# plan §100 / plan section 100 / plan sections 39 and 104 (the trailing number is
# the section; the leading one is carried by the same sentence's other match)
PLAN_REF = re.compile(r"\bplan\s+(?:§|sections?\s+)([0-9]+)")
# "## 3. Results", "### 5.47 Amendment", "### 5.20.1 Amendment" - leading number
# followed by the title, at any heading level.
NUMBERED_HEADING = re.compile(r"^#{1,6}\s*([0-9]+(?:\.[0-9]+)*)[\s.:]", re.MULTILINE)
# "# 98. Hypothesis Tracking" / "# 100. Final Research Deliverables"
PLAN_HEADING = re.compile(r"^#\s*([0-9]+)\.", re.MULTILINE)


def _index_section() -> str:
    """The README's Documentation section, up to the next level-2 heading."""

    text = README.read_text(encoding="utf-8")
    start = text.index(INDEX_HEADING)
    rest = text[start + len(INDEX_HEADING) :]
    end = rest.find("\n## ")
    return rest if end == -1 else rest[:end]


def _project_docs() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.md")
        if not any(part in SKIPPED_DIRS for part in path.parts)
    )


def test_every_standing_document_is_indexed() -> None:
    """A new root-level document cannot appear without being placed on the map."""

    section = _index_section()
    unindexed = [
        path.name
        for path in sorted(ROOT.glob("*.md"))
        if path.name not in NOT_PROJECT_DOCS
        and path.name != README.name
        and f"`{path.name}`" not in section
    ]

    assert unindexed == [], (
        "standing documents at the repository root are not named in the README's "
        f"Documentation section: {unindexed}"
    )


def test_every_document_the_index_names_exists() -> None:
    """The index cannot go stale by pointing at a document that was renamed away.

    A name carrying a ``/`` is a path and must resolve from the repository root. A bare
    filename is allowed to appear in prose without a directory - ``analysis.md`` names a
    kind of document as much as a file - so it only has to exist somewhere in the
    project rather than at the root. Either way the index cannot name a document that
    is nowhere.
    """
    present = {path.name for path in _project_docs()}
    missing = []
    for name in sorted(set(BACKTICKED_MD.findall(_index_section()))):
        # `artifacts/*/NOTES.md` and friends are deliberately patterns, describing a
        # class of per-arm file rather than one path.
        if "*" in name:
            continue
        if (ROOT / name).exists():
            continue
        if "/" not in name and name in present:
            continue
        missing.append(name)

    assert missing == [], (
        "the README's Documentation section names documents that do not exist: "
        f"{missing}"
    )


def test_every_per_arm_document_is_covered_by_a_pattern() -> None:
    """The per-arm notes are indexed by pattern, so the patterns must cover them.

    There is one set of these per arm, so the index names them as a class rather than
    one by one - this is the check that the class is actually covered, including a
    document added beside a new arm. Only the known kinds are required, so scratch
    markdown of another name does not fail the suite.
    """
    section = _index_section()
    tokens = BACKTICKED_MD.findall(section)
    literal = {token for token in tokens if "*" not in token}
    patterns = [token for token in tokens if "*" in token]

    uncovered = []
    for path in _project_docs():
        rel = path.relative_to(ROOT).as_posix()
        if not rel.startswith("artifacts/") or path.name not in PER_ARM_DOCS:
            continue
        if rel in literal or any(fnmatch.fnmatch(rel, pattern) for pattern in patterns):
            continue
        uncovered.append(rel)

    assert uncovered == [], (
        "these documents beside an arm are not covered by any pattern in the README's "
        f"Documentation section: {sorted(uncovered)}"
    )


def test_the_index_covers_the_documents_it_should() -> None:
    """Guard the guard: the section has to be non-trivial to be worth checking."""

    section = _index_section()
    named = {name for name in BACKTICKED_MD.findall(section) if "*" not in name}

    for expected in (
        "complete_implementation_plan.md",
        "RESEARCH_REPORT.md",
        "IMPLEMENTATION_STATUS.md",
        "T4_RUNBOOK.md",
    ):
        assert expected in named, f"{expected} is not in the Documentation index"


def test_report_section_references_resolve() -> None:
    """Every ``RESEARCH_REPORT.md`` section reference points at a real heading."""

    headings = set(NUMBERED_HEADING.findall(REPORT.read_text(encoding="utf-8")))
    dangling: dict[str, list[str]] = {}

    for path in _project_docs():
        for ref in REPORT_REF.findall(path.read_text(encoding="utf-8")):
            if ref not in headings:
                dangling.setdefault(ref, []).append(path.relative_to(ROOT).as_posix())

    assert dangling == {}, (
        "these references name a RESEARCH_REPORT.md section that does not exist: "
        f"{dangling}"
    )


def test_plan_section_references_resolve() -> None:
    """Every "plan section N" reference points at a section the plan actually has."""

    sections = set(PLAN_HEADING.findall(PLAN.read_text(encoding="utf-8")))
    dangling: dict[str, list[str]] = {}

    for path in _project_docs():
        for ref in PLAN_REF.findall(path.read_text(encoding="utf-8")):
            if ref not in sections:
                dangling.setdefault(ref, []).append(path.relative_to(ROOT).as_posix())

    assert dangling == {}, (
        "these references name a complete_implementation_plan.md section that does "
        f"not exist: {dangling}"
    )
