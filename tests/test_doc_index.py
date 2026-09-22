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
DOCS = ROOT / "docs"
README = ROOT / "README.md"
REPORT = DOCS / "RESEARCH_REPORT.md"
PLAN = DOCS / "complete_implementation_plan.md"

# Only the front door and agent guidance belong at the repository root; every document
# a reader should find lives under ``docs/``.
ROOT_STRAY_ALLOWED = {"README.md", "AGENTS.md"}

INDEX_HEADING = "## Documentation"

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


def test_standing_documents_live_under_docs() -> None:
    """The layout is the point - a document drifting back to the root is the scatter.

    The root carried five documents of 700-6,700 lines each. They live under ``docs/``
    now, and this is what keeps that true rather than a one-off tidy-up.
    """
    stray = sorted(
        path.name
        for path in ROOT.glob("*.md")
        if path.name not in ROOT_STRAY_ALLOWED
    )

    assert stray == [], (
        f"standing documents belong under docs/, not at the repository root: {stray}"
    )


def test_every_standing_document_is_indexed() -> None:
    """A new document under ``docs/`` cannot appear without being placed on the map."""

    section = _index_section()
    unindexed = [
        path.name
        for path in sorted(DOCS.glob("*.md"))
        if f"`docs/{path.name}`" not in section
    ]

    assert unindexed == [], (
        "documents under docs/ are not named in the README's Documentation section: "
        f"{unindexed}"
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
        "docs/complete_implementation_plan.md",
        "docs/RESEARCH_REPORT.md",
        "docs/IMPLEMENTATION_STATUS.md",
        "docs/T4_RUNBOOK.md",
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


def _python_sources() -> list[Path]:
    """The implementation the front page's claims are about.

    ``src/`` and ``scripts/`` only, deliberately: this module spells the names it
    searches for (``extract_prior``, ``import jax``), so a search that included
    ``tests/`` would find its own literals and pass whatever the repository did - which
    is precisely how the first version of `test_the_front_page_advertises_no_tool...`
    failed on a repository with no such tool in it at all.
    """

    return [
        path
        for root in (ROOT / "src", ROOT / "scripts")
        for path in root.rglob("*.py")
    ]


def test_the_front_page_names_the_engine_the_repository_imports() -> None:
    """The README's architecture block is a *dependency* claim, so it can be checked.

    Measured 2026-09-22: that block read `TRAINING ENGINE | PyTorch + T4x2 / JAX + TPU
    v5e` while nothing in the repository imported jax (and the hardware the project has
    actually measured is a TPU **v3-8** under torch_xla, eight XLA devices, not a v5e).
    The invariant under test is **agreement**, not the absence of jax: add a JAX backend
    and this fails until the front page says so, which is the direction that matters.
    """

    engine = README.read_text(encoding="utf-8").split("TRAINING ENGINE")[1].splitlines()[1]
    assert "PyTorch" in engine

    jax_imported = any(
        re.search(r"^\s*(?:import jax|from jax)", path.read_text(encoding="utf-8"), re.M)
        for path in _python_sources()
    )
    assert ("JAX" in engine) == jax_imported, (
        f"the front page's TRAINING ENGINE line and the tree disagree about jax: "
        f"engine={engine.strip()!r}, src+scripts import jax={jax_imported}"
    )


#: A backticked snake_case identifier in the feature list - a symbol or a layout name,
#: as opposed to a path (``artifacts/mixer-benchmark/``) or a flag (``--device``), which
#: this deliberately does not try to resolve.
ADVERTISED_SYMBOL = re.compile(r"`([a-z][a-z0-9_]{3,})`")


def test_every_symbol_the_feature_list_names_exists_somewhere_in_the_tree() -> None:
    """The general form of the defect the prior-extraction bullet was: it named a thing.

    Measured 2026-09-22, the front page advertised a prior-extraction tool, JAX, a TPU v5e
    and attention-head mutations in as many words, and the repository had none of the
    four. This resolves each backticked identifier in the feature list against the
    implementation text - the same check that would have caught all four at once, and the
    one that catches the next invented name without anyone thinking to look for it.

    Deliberately *text*, not import: some of these are layout names (``gdn_3to1``) rather
    than Python symbols, and requiring importability would rule out naming a real one.
    """

    features = README.read_text(encoding="utf-8").split("## Features")[1].split("## ")[0]
    sources = {path: path.read_text(encoding="utf-8") for path in _python_sources()}

    missing = sorted(
        name
        for name in set(ADVERTISED_SYMBOL.findall(features))
        if not any(name in text for text in sources.values())
    )
    assert missing == [], (
        f"the front page's feature list names symbols no source file contains: {missing}"
    )


def test_the_front_page_advertises_no_tool_the_repository_lacks() -> None:
    """A bullet for a tool that does not exist is the same defect as a wrong number.

    `Qwen-Informed Prior: Prior extraction and structural analysis tool for hybrid
    architectures` sat in the README's feature list while nothing under `src/`,
    `scripts/` or `tests/` mentioned prior extraction at all - the only occurrence
    anywhere in the repository was the README's own generated egg-info mirror. Removed
    2026-09-22, and pinned as agreement in the same direction as the engine line: build
    the tool, and the front page has to name it.
    """

    readme = README.read_text(encoding="utf-8").lower()
    has_prior_tool = any(
        "extract_prior" in path.read_text(encoding="utf-8")
        or "prior_extraction" in path.read_text(encoding="utf-8")
        for path in _python_sources()
    )
    advertises_prior_tool = "qwen-informed prior" in readme
    assert advertises_prior_tool == has_prior_tool, (
        "the front page advertises a prior-extraction tool the repository does not "
        f"have, or has one it does not advertise (advertised={advertises_prior_tool}, "
        f"implemented={has_prior_tool})"
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
