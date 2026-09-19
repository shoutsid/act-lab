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
    """The index cannot go stale by pointing at a document that was renamed away."""

    missing = [
        name
        for name in sorted(set(BACKTICKED_MD.findall(_index_section())))
        # `artifacts/*/NOTES.md` and friends are deliberately patterns, describing a
        # class of per-arm file rather than one path.
        if "*" not in name and not (ROOT / name).exists()
    ]

    assert missing == [], (
        "the README's Documentation section names documents that do not exist: "
        f"{missing}"
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
