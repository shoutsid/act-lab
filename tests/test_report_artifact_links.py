"""Every study the report cites has an artifact, and every artifact is cited.

Section 5.19 was run, quoted across the report, and used as another section's
motivation - while its committed summaries had never been written. A result in
active use with nothing behind it, found only when section 5.20 tried to cite it
and could not resolve the numbers. This module makes that defect class fail
loudly instead of silently: a study id named in the report must resolve to a
committed summary, and a committed summary must be referenced somewhere.

These are link checks, not statistics checks - the per-study pinning tests
(``test_initialization_summary``, ``test_portability_summary``,
``test_specificity_summary``, ...) own the numbers.
"""

from __future__ import annotations

import json
import re

from act.evolution.figures_data import summaries_dir

STUDY_ID = re.compile(r"study_[0-9a-f]{16}")
ROOT = summaries_dir().parents[1]  # .../<repo>/artifacts/study-summaries
REPORT = ROOT / "RESEARCH_REPORT.md"
STATUS = ROOT / "IMPLEMENTATION_STATUS.md"


def _records() -> dict[str, str]:
    """Committed study id -> filename, for the dict-shaped records only.

    A few files under the summaries directory are panels rather than single
    records (they hold a list); those carry no study id and are skipped.
    """
    records: dict[str, str] = {}
    for path in sorted(summaries_dir().glob("*.json")):
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and payload.get("study_id"):
            records[payload["study_id"]] = path.name
    return records


def test_every_cited_study_id_has_a_committed_summary() -> None:
    """A number quoted with an id must have an artifact under that id."""
    report = REPORT.read_text()
    cited = set(STUDY_ID.findall(report))
    assert cited, "report cites no study ids - the link check would be vacuous"
    have = _records()
    missing = sorted(cited - set(have))
    assert not missing, (
        "study id(s) cited in RESEARCH_REPORT.md with no committed summary: "
        f"{missing}. Either regenerate the summary from its shards or stop "
        "citing the id (report section 5.19 recorded this defect)."
    )


def test_every_committed_summary_is_referenced() -> None:
    """A committed artifact must be reachable from the written record."""
    text = REPORT.read_text() + STATUS.read_text()
    orphans = []
    for study_id, filename in sorted(_records().items()):
        if study_id in text or filename[: -len(".json")] in text:
            continue
        orphans.append(filename)
    assert not orphans, (
        "committed study summaries that neither the report nor the status doc "
        f"references: {orphans}. Cite them, or delete them if they are spent "
        "scratch rather than records."
    )
