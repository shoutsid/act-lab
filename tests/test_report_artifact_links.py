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

The same defect class has a fresh-clone form: an artifact a test reads can be
present on the machine that generated it and absent from every other checkout,
because the blanket `artifacts/*` rule ignores it. `null-clause-power.json` and
`artifacts/gate3-384.jsonl` were both in exactly that position - the first found
and excepted, the second found by the same scan and left open until the record
that cites it was made verifiable. The last check here makes that class fail
loudly instead of silently, because it is invisible from inside a dirty tree.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

import pytest

from act.evolution.figures_data import summaries_dir

STUDY_ID = re.compile(r"study_[0-9a-f]{16}")
ROOT = summaries_dir().parents[1]  # .../<repo>/artifacts/study-summaries
REPORT = ROOT / "docs" / "RESEARCH_REPORT.md"
STATUS = ROOT / "docs" / "IMPLEMENTATION_STATUS.md"


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


SECTION_HEADER = re.compile(r"^### (5\.\d+(?:\.\d+)?) (.*)$", re.M)


def test_no_section_header_contradicts_its_own_summary() -> None:
    """A section labelled "not yet run" must not have a committed summary.

    The section list is how a reader enters this report, and several entries were
    labelled "designed, not yet run" long after their blocks had run and their
    summaries were committed - the same drift that let section 5.19 be cited with
    no artifact. ``report_section`` is machine-readable, so the contradiction is
    checkable: a summary that names a section proves that section ran.
    """
    report = REPORT.read_text()
    stale = []
    for match in SECTION_HEADER.finditer(report):
        section, title = match.group(1), match.group(2)
        if "not yet run" not in title:
            continue
        proving = [
            filename
            for study_id, filename in _records().items()
            if _section_of(filename) == section
        ]
        if proving:
            stale.append((section, proving))
    assert not stale, (
        "section header(s) still say 'not yet run' but a committed summary "
        f"proves the block ran: {stale}. Relabel the header."
    )


def _section_of(filename: str) -> str | None:
    payload = json.loads((summaries_dir() / filename).read_text())
    section = payload.get("report_section")
    return section if isinstance(section, str) else None


#: ``--out DIR`` / ``--output FILE`` / ``--compare-out FILE`` in a documented command.
OUTPUT_FLAG = re.compile(r"--[a-z-]*out[a-z-]*\s+(\S+)")


def test_a_documented_output_path_does_not_shadow_a_committed_summary() -> None:
    """A reproduce command must write the artifact it claims to reproduce.

    Section 9.10's block wrote `factor-map-recheck.json` to the artifacts root
    while the committed artifact is under `study-summaries/`, and `artifacts/*`
    is gitignored - so running it as documented produced an untracked file and
    left the committed one stale, with the reader believing they had reproduced
    it. The section named the correct path eleven lines further down, which is
    what makes this checkable rather than a matter of taste: a path that names a
    committed summary has to *be* that summary's path.
    """
    committed = {path.name for path in summaries_dir().glob("*.json")}
    assert committed, "no committed summaries - the check would be vacuous"
    # The trailing parts, not the whole path: a documented target is written relative
    # to the repository root and ``summaries_dir()`` is absolute.
    inside = summaries_dir().parts[-2:]
    for raw in OUTPUT_FLAG.findall(REPORT.read_text() + STATUS.read_text()):
        target = Path(raw.strip("'\"`,"))
        if target.name in committed:
            assert target.parent.parts[-2:] == inside, (
                f"{target} carries the name of a committed summary but is not "
                f"written into {summaries_dir()} - the command would leave the "
                "committed copy stale"
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


# --- artifacts a test reads must survive a fresh clone ------------------------

#: ``artifacts/...`` path literals as a test spells them, in either the
#: ``Path("artifacts/x.json")`` or the ``ROOT / "artifacts/x.json"`` style.
ARTIFACT_LITERAL = re.compile(r"artifacts/[A-Za-z0-9_./-]+")

#: A call that says an artifact may be *absent* rather than reading it.
SKIP_CALLS = frozenset({"skip", "skipif", "fail", "xfail"})


def _called_name(func: ast.expr) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _artifact_literals_in(source: str) -> tuple[set[str], set[str]]:
    """``(paths the module uses, paths it names only while skipping on their absence)``.

    Parsed rather than matched, because a test legitimately *names* a missing
    artifact in two places that are not claims about it: a comment (which never
    reaches the AST) and a ``pytest.skip``/``reason=`` message, which is a string
    constant like any other. Separating those from the paths a module *uses* is
    the difference between a check aimed at reads and one that fires on the first
    test that explains why it is skipping - which is the version this started as,
    and it failed on its own author's skip reason immediately.

    The second set is not discarded, it is the exemption: a test may read an
    artifact that only the author's machine has *provided it guards for the
    absence*, which is exactly the pattern the local-checkpoint and capacity-sweep
    tests use. Naming a path in a skip message and reading it in the same module
    is therefore a claim this check accepts, and reading it without one is not.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:  # unparsable is not this check's business
        return set(), set()
    named_to_skip: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        keywords = {keyword.arg for keyword in node.keywords}
        if _called_name(node.func) in SKIP_CALLS or "reason" in keywords:
            named_to_skip |= {
                id(inner) for inner in ast.walk(node)
            }
    used: set[str] = set()
    guarded: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if "artifacts/" not in node.value:
            continue
        found = set(ARTIFACT_LITERAL.findall(node.value))
        (guarded if id(node) in named_to_skip else used).update(found)
    return used, guarded


def test_every_artifact_a_test_reads_is_not_gitignored() -> None:
    """A test must not depend on a file that only its author's machine has.

    A committed test that reads a gitignored artifact passes where it was
    written and fails everywhere else - and the failure looks like a bug in the
    test rather than the missing commit, so it is worth detecting before a fresh
    clone does. Only paths that resolve to a file here are checked: a test may
    name a directory, a glob or an at-run-time path, and none of those is a
    claim about a committed file. The check is skipped, not passed, when git
    cannot answer - an installed copy of the package is not a checkout.
    """
    try:
        subprocess.run(
            ["git", "--version"], capture_output=True, check=True, cwd=ROOT
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git is not available to answer whether a path is ignored")

    checked: set[str] = set()
    offenders: list[tuple[str, str]] = []
    for test_file in sorted((ROOT / "tests").glob("*.py")):
        used, guarded = _artifact_literals_in(test_file.read_text())
        for name in sorted(used - guarded):
            if name in checked or not (ROOT / name).is_file():
                continue
            checked.add(name)
            result = subprocess.run(
                ["git", "check-ignore", "-q", name], capture_output=True, cwd=ROOT
            )
            if result.returncode == 128:
                pytest.skip("not a git checkout - the ignored-ness is unknowable")
            if result.returncode == 0:
                offenders.append((name, test_file.name))

    assert checked, "no test names a real artifact path - the check would be vacuous"
    assert not offenders, (
        "artifact(s) a test reads are gitignored, so the test can only pass on "
        f"the machine that generated them: {offenders}. Commit the artifact "
        "(with a `.gitignore` exception and a note saying why), or guard for its "
        "absence with a `pytest.skip` that names it so the skip is the honest "
        "report in a fresh clone."
    )
