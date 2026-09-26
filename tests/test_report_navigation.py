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

import json
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
    # And the other direction, which is what makes an over-claimed range fail:
    # the groups must cover exactly the subsections that exist. A group that
    # reaches past the newest section is claiming a section nobody wrote.
    extra = sorted(covered - set(numbers))
    assert not extra, f"the groups claim subsections that do not exist: {extra}"


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


def test_the_synthesis_depth_claims_match_the_committed_summaries() -> None:
    """§10.1's churn claim and its ladder must be true of the records, not of prose.

    §10.1 asserted "churn harm holds at every measured depth" and stopped the
    depth ladder at four rungs, both of which 5.55-5.56 falsified - and no claim
    audit or summary pin could see it, because a *universal* statement is not a
    number. The same reading caught 5.56's own successor: the section called
    the gradient "a constant, not a decaying one" on three increments, and the
    fourth (5.57) is 1.5x all of them. This reads the summaries directly: the
    sign of the collapse at every rung, the monotonicity of the value ladder,
    the depth the report says the churn regime flips at, and the increments the
    gradient claim rests on.

    The gradient claim itself has now moved a third time - 5.59's increment
    landed between the two bands its predecessors declared - so the assertion is
    no longer "which model is right" but the shape the three post-flat increments
    make: all above the flat level, declining monotonically, and declining by a
    roughly constant *factor*. A report that goes back to flatness, to
    acceleration, or to a decided mild rise fails here.
    """
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import null_toy_study as study  # noqa: E402

    rungs = (
        "depth2",
        "depth3",
        "depth4-prefix",
        "depth5-dose",
        "depth6-dose",
        "depth7-dose",
        "depth8-dose",
        "depth9-dose",
    )
    values, collapses = [], []
    for profile in rungs:
        name = Path(study.PROFILES[profile]["summary"]).name
        rows = json.loads((ROOT / "artifacts" / "study-summaries" / name).read_text())["per_seed"]
        values.append(
            sum(r["losses"]["adaptive"] - r["losses"]["null"] for r in rows) / len(rows)
        )
        collapses.append(
            sum(r["losses"]["null"] - r["losses"]["random"] for r in rows) / len(rows)
        )
    assert values == sorted(values, reverse=True), f"the ladder is not monotone: {values}"
    # The flip is at depth 6 and nowhere earlier - which is what §10.1 now says.
    assert all(c < 0 for c in collapses[:4]) and all(c > 0 for c in collapses[4:]), collapses
    text = " ".join(_text().split())
    assert "churn harm (null < random) holds at every measured depth **up to 5**" in text
    chain = (
        "mean delta +0.0508 > +0.0208 > +0.0046 > -0.0371 > -0.0753 > -0.1175 > -0.1810"
        " > -0.2352 > -0.2839"
    )
    assert chain in text
    # The gradient claim is read off the increments themselves, and it has now been
    # wrong three times: the flat reading (three steps at ~0.04) was refuted by the
    # fourth, the accelerating reading (the fourth repeated) by the fifth, and the
    # mild-rise reading by the sixth, which landed in the seam between their bands.
    # What is left is the shape: every post-flat increment above the flat level,
    # declining monotonically, and decaying by a roughly constant factor.
    steps = [values[i] - values[i + 1] for i in range(len(values) - 1)]
    flat = sum(steps[1:4]) / 3
    assert flat == pytest.approx(0.0407, abs=1e-3), steps
    excursion = [step - flat for step in steps[4:]]
    assert len(excursion) == 3 and all(gap > 0 for gap in excursion), excursion
    assert excursion == sorted(excursion, reverse=True), excursion
    ratios = [b / a for a, b in zip(excursion, excursion[1:])]
    assert all(0.5 < ratio < 0.7 for ratio in ratios), ratios
    assert (
        "The gradient above the gate is an excursion decaying back to a"
        " flat level, and three registered forecasts have now missed it"
    ) in text
    assert "-0.0162, -0.0418, -0.0382, -0.0422, -0.0635, -0.0541, -0.0487" in text
    assert "0.002-wide seam" in text
    assert "stops being so at depth 6" in text
    # The edge's marginal decline reading, which 5.58 reported as descriptive
    # only, is not carried as a finding - and the synthesis has to say why.
    assert "p `0.0244` over 3 -> 8, and p `0.121` over 3 -> 9" in text
    assert "does not survive the next rung" in text


def test_the_study_index_status_row_is_live() -> None:
    """The §5 index is a *status* table, so its cells are current-state surfaces.

    Its real-scale row captures current authorization and the last reported
    progress, without claiming the snapshot is a live Kaggle check. The registered
    `### 5.x` bodies remain history; only the status column changes.
    """
    row = next(
        (
            line
            for line in _text().splitlines()
            if line.startswith("| Real-scale cost measurement |")
        ),
        None,
    )
    assert row is not None, "the §5 index no longer has a real-scale row"
    assert "RESUMPTION AUTHORIZED" in row, row
    assert "IN FLIGHT" not in row, row
    assert "3 of 12" in row, row
    assert "docs/T4_RUNBOOK.md" in row, row
    runbook = (ROOT / "docs" / "T4_RUNBOOK.md").read_text()
    assert "RESUMPTION AUTHORIZED" in runbook
    assert "Do not start a second driver" in runbook
    assert "TPU probe v6" in runbook
    assert "14.68/30 GPU-hours remaining" in runbook
    assert "14.77" not in runbook
    assert "15,034.2 tokens/s" in runbook
    assert "tpu-probe-v6.json" in runbook
    assert "15024-15035" in row, row
    assert "primary kernel v9 was running" in row, row
    assert "14.68/30 GPU-hours remaining at the 19:12Z check" in row, row
    assert "act-v0-5-2" in row, row
    # The deliverables checklist is also an active status surface: keep its
    # seed denominator and artifact dataset aligned with the current queue.
    checklist = _text().split("## 7. Deliverables checklist", 1)[1].split("\n## ", 1)[0]
    assert "act-v0-5-2" in checklist
    assert "3 of the active block's 12 seeds (15024-15026)" in checklist
    assert "15000-15011" not in checklist
    # A live campaign's progress is a snapshot, never a pinned constant: the
    # row, the header and §4 must all say when they were true, so a stale count
    # reads as stale rather than as a wrong measurement. Checked on the
    # whitespace-normalized text, because the phrases wrap across lines and a
    # line-wise check silently skips them (the first version of this did).
    flat = " ".join(_text().split())
    progress = list(re.finditer(r"\d+ of \d+ (?:declared )?seeds", flat))
    assert progress, "the report no longer states any campaign progress"
    assert any(
        "last recorded" in flat[max(0, match.start() - 100) : match.end() + 120]
        for match in progress
    ), "the campaign progress must be labeled as a snapshot"


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


def test_the_crossing_rule_is_read_from_the_committed_ladders() -> None:
    """§10.1's ratio table: the gate's boundary in `|edge| / pool`, recomputed.

    The rule is a *description* of the committed cells, and it is the corrected
    form of the staging sections 5.60/5.61 read: those arms divided the mechanism
    by the side of the onset, and the onset cell itself is a counterexample. The
    assertions recompute every ratio and every gate verdict, so the table cannot
    drift from the summaries and the boundary has to stay bracketed the way the
    report says it is.
    """
    import json as _json

    summaries = ROOT / "artifacts" / "study-summaries"

    def terms(name: str) -> tuple[float, float, float]:
        rows = _json.loads((summaries / name).read_text())["per_seed"]

        def mean(left: str, right: str) -> float:
            values = [r["losses"][left] - r["losses"][right] for r in rows]
            return sum(values) / len(values)

        return mean("adaptive", "null"), mean("adaptive", "random"), mean("random", "null")

    def opened(name: str) -> bool:
        payload = _json.loads((summaries / name).read_text())
        return bool(payload["comparisons"]["adaptive_vs_null"]["passes_bonferroni"])

    ladder = (
        ("null-control-depth3-384.json", 0.96, False),
        ("width-d3-equal-576.json", 0.88, False),
        ("width-d3-dose-576.json", 0.80, False),
        ("width-d3-dose4-576.json", 1.10, False),
        ("width-d3-dose3-576.json", 1.55, True),
        ("null-control-depth4-prefix-384.json", 1.53, True),
        ("width-d4-equal-576.json", 1.37, True),
    )
    ratios: list[tuple[float, bool]] = []
    for name, expected_ratio, expected_open in ladder:
        value, edge, pool = terms(name)
        assert pool > 0, f"{name} no longer has a positive pool term"
        assert abs(edge) / pool == pytest.approx(expected_ratio, abs=0.01), name
        assert opened(name) is expected_open, name
        # The mean crosses before the bar does: every cell whose ratio clears 1
        # has a negative mean, including the one the bar still rejects.
        assert (value < 0) is (expected_ratio > 1.0), name
        ratios.append((abs(edge) / pool, expected_open))
    # The boundary the report quotes, stated as the bracket it is: no open cell
    # below the highest closed one, and the near miss is the top of the closed set.
    closed = [r for r, is_open in ratios if not is_open]
    open_ = [r for r, is_open in ratios if is_open]
    assert max(closed) == pytest.approx(1.0959, abs=0.001)
    assert min(open_) == pytest.approx(1.37, abs=0.01)
    assert max(closed) < min(open_), "the boundary is no longer bracketed"
    text = " ".join(_text().split())
    assert "bracketed by construction to `[1.10, 1.37]`" in text
    assert "corrects a reading sections 5.60 and 5.61 gave" in text
    assert "tracks the mechanism is the *dose*, not the side" in text
    # The rule's out-of-policy test, and the caveat it produced.
    assert "it has had its first out-of-policy test (5.62" in text
    assert "the crossing is **policy-robust**" in text
    assert "indistinguishable by rank-sum (p `0.472` / `0.528`)" in text
    assert (
        "The two-term split is therefore a description of the balance at a dose rather"
        " than an invariant of the mechanism"
    ) in text


def test_the_cross_axis_claims_the_bisection_added_are_live() -> None:
    """§10.1, §10.3, §10.5 read against 5.60: a price, a peak, and an exception.

    These are the claims that section 5.60 changed the *synthesis* to say, and
    each one is a statement no summary pin would catch on its own: a ratio
    between two windows, the location of a term's peak, and the one regime where
    the controller's edge crosses a threshold. A rewrite that reverts any of them
    to the pre-bisection picture fails here.
    """
    text = " ".join(_text().split())
    for fragment in (
        # The price of the gate, bounded on both axes.
        "2.9x to 7.5x more",
        "4.0x-7.5x more parameters along width at the below-onset depth",
        "**the ratio tightens to 4.0x-7.5x**",
        "the pool term peaks at 2.87x and the controller's edge at 5.66x",
        "23% *fewer* parameters than the width dose",
        # The mechanism, replaced: the pool term peaks before the gate opens.
        "The bisection then locates both flips",
        "it *peaks* at 2.87x (`+0.305267`), declines through the bisection",
        "So below the onset the flip happens *after* the gate opens",
        # The one place the controller's edge crosses a threshold, and its status.
        "the edge is flat (`-0.194518` -> `-0.199203`, p `0.199`)",
        "That is a hypothesis about where a transfer policy could eventually matter",
    ):
        assert fragment in text, fragment


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
