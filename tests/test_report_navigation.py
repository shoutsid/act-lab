"""The report's header and index against its own contents.

The report's header said "Updated: 2026-09-13" while the body carried eight
further days of recorded results, and it described the document as a skeleton
for a milestone three sections out of date - because nothing checked. The index
has the same failure mode: a section added to the body while the Contents table
is left behind is invisible to every other test in this suite.

These tests close both, in the direction that fails loudly: the header's date
must be the newest date the body records, every top-level section must be named
in the Contents, and every section-5 subsection must fall inside exactly one
declared group.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "docs" / "RESEARCH_REPORT.md"

# The header sits before the Contents; the body is everything after it.
CONTENTS_HEADING = "## Contents"
DATE = re.compile(r"2026-\d{2}-\d{2}")
TOP_LEVEL = re.compile(r"^## (\d+)\. ", re.MULTILINE)
SUBSECTION = re.compile(r"^### 5\.(\d+)", re.MULTILINE)
UPDATED = re.compile(r"\*\*Updated: (\d{4}-\d{2}-\d{2})\*\*")


def _text() -> str:
    return REPORT.read_text()


def _split() -> tuple[str, str]:
    text = _text()
    marker = text.index(CONTENTS_HEADING)
    return text[:marker], text[marker:]


def _contents_index() -> str:
    """The Contents table only, so prose elsewhere cannot satisfy the checks."""
    _, contents = _split()
    lines = contents.splitlines()
    rows = [line for line in lines if line.startswith("|")]
    assert rows, "the Contents table is missing"
    return "\n".join(rows)


def test_the_header_date_is_the_newest_date_the_body_records() -> None:
    """A stale "Updated:" line is a defect with no other symptom."""
    header, body = _split()
    declared = UPDATED.search(header)
    assert declared, "the header no longer states an Updated date in the pinned form"
    newest = max(DATE.findall(body))
    assert declared.group(1) == newest, (
        f"the header says {declared.group(1)} but the body's newest date is {newest}"
    )


def test_every_top_level_section_is_in_the_contents() -> None:
    top = sorted({int(n) for n in TOP_LEVEL.findall(_text())})
    assert top, "no top-level sections found"
    index = _contents_index()

    # Anchored to the row so a subsection range like "5.1-5.3" cannot be read
    # as the top-level range "1-5".
    ranges = [
        (int(a), int(b))
        for a, b in re.findall(r"^\| \*\*(\d+)-(\d+)\*\*", index, re.MULTILINE)
    ]
    # Section 5 is the one top-level section indexed by its subsections instead
    # of by its own range, so a declared subsection group covers it too.
    covered = {
        int(a)
        for a in re.findall(r"^\| \*\*(\d+)\.\d+-\d+\.\d+\*\*", index, re.MULTILINE)
    }
    missing = [
        n for n in top if n not in covered and not any(a <= n <= b for a, b in ranges)
    ]
    assert not missing, f"top-level sections absent from the Contents: {missing}"


def test_every_section_5_subsection_falls_in_exactly_one_declared_group() -> None:
    """Adding a study without extending the index fails here."""
    numbers = sorted({int(n) for n in SUBSECTION.findall(_text())})
    assert numbers, "no section-5 subsections found"
    groups = [
        (int(start), int(end))
        for start, end in re.findall(
            r"^\| \*\*5\.(\d+)-5\.(\d+)\*\*", _contents_index(), re.MULTILINE
        )
    ]
    assert groups, "the Contents declares no section-5 groups"
    assert groups == sorted(groups), "the section-5 groups are not in order"
    for start, end in groups:
        assert start <= end
    for before, after in zip(groups, groups[1:]):
        assert after[0] == before[1] + 1, f"the groups leave a gap or overlap at {before}/{after}"
    covered = {n for start, end in groups for n in range(start, end + 1)}
    missing = [n for n in numbers if n not in covered]
    assert not missing, f"section-5 subsections outside every group: {missing}"


def test_every_registered_arm_is_inside_a_declared_group() -> None:
    """The index is read against the study table, not only against the body.

    The last group is the one that grows - an arm is registered roughly every
    time the report is - so pinning the table's own sections against the ranges
    is what catches an arm added without extending the index.
    """
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import null_toy_study as study  # noqa: E402

    def as_number(section: str) -> float:
        major, _, minor = section.partition(".")
        return float(major) * 1000 + float(minor or 0)

    groups = [
        (int(a), int(b))
        for a, b in re.findall(
            r"^\| \*\*5\.(\d+)-5\.(\d+)\*\*", _contents_index(), re.MULTILINE
        )
    ]
    covered = {n for a, b in groups for n in range(a, b + 1)}
    sections = {spec["report_section"] for spec in study.PROFILES.values()}
    outside = sorted((s for s in sections if int(s.split(".")[1]) not in covered), key=as_number)
    assert not outside, f"registered arms outside every Contents group: {outside}"
    newest = max(sections, key=as_number)
    last_end = groups[-1][1]
    assert int(newest.split(".")[1]) <= last_end < int(newest.split(".")[1]) + 10, (
        f"the last group ends at {last_end} but the newest arm is {newest}"
    )


def test_the_report_still_has_the_sections_the_header_promises() -> None:
    """The header names section 9 and 10 as the entry points; check they exist."""
    text = _text()
    for heading in (
        "## 9. Substrate audit",
        "## 10. Synthesis",
        "## 3. Results: the factor map",
    ):
        assert heading in text, heading


@pytest.mark.parametrize("name", ["IMPLEMENTATION_STATUS.md", "T4_RUNBOOK.md"])
def test_the_documents_the_header_points_at_exist(name: str) -> None:
    assert (ROOT / "docs" / name).is_file()
    assert (ROOT / "artifacts" / "ARCH_SEARCH_FINDINGS.md").is_file()


def test_the_deliverables_checklist_is_not_stale_about_the_handoff() -> None:
    """Items 3-5 said TODO/PARTIAL while the machinery shipped.

    The checklist claimed "no published artifacts" and "Adaptive checkpoints:
    TODO" while `src/act/training/v0_handoff.py` was already the declared path's
    publish-and-restore contract. Pin the two ends: the rows must claim DONE for
    the declared path, and the code the claim names must exist.
    """
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from act.training import v0_handoff  # noqa: E402

    assert v0_handoff.HANDOFF_DATASET == "act-checkpoints/v0-5-1"
    for name in ("publish_seed", "restore_seed", "published_seed_status", "seed_directory"):
        assert callable(getattr(v0_handoff, name)), name

    checklist = "\n".join(
        line for line in _text().splitlines() if re.match(r"^\| \d+\. ", line)
    )
    for item in ("| 3. Dataset manifests |", "| 4. Baseline checkpoints |", "| 5. Adaptive checkpoints |"):
        row = next((line for line in checklist.splitlines() if line.startswith(item)), None)
        assert row is not None, f"the checklist no longer has a row for {item}"
        assert "DONE for the declared" in row or "DONE as for item 4" in row, row


def test_the_real_text_series_the_checklist_claims_exists() -> None:
    """Item 9 points at the architecture track's own artifacts; keep it honest."""
    series = sorted((ROOT / "artifacts").glob("arch-zoo-100m-v*"))
    assert series, "the real-text series the checklist names is absent"
    assert (ROOT / "artifacts" / "ARCH_SEARCH_FINDINGS.md").is_file()
    row = next(
        (
            line
            for line in _text().splitlines()
            if line.startswith("| 9. Benchmark results |")
        ),
        None,
    )
    assert row is not None and "arch-zoo-100m-v1" in row, row
