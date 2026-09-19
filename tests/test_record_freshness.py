"""Tests for the record-freshness check.

The check exists because an analyzer defect left one arm's committed `decision.md`
unregenerable and nothing noticed: `verify_record.py` recomputes each claim from the
*ledgers*, so it never re-renders the artifact the claim is read from, and a record
that cannot be regenerated cannot be audited. These pin the two halves that can rot
without anyone seeing it - the runbook parser, and the requirement that every arm
document the command that writes its own record.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_record_freshness import (  # noqa: E402
    check_arm,
    commands_in,
    documented_commands,
    option,
    ordered_candidates,
    record_read_the_wrong_rows,
    script_commands,
    scratch_tree,
    source_names,
)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
WATCHERS = sorted((ROOT / "scripts").glob("run_*_when_*.sh"))
#: An analyzer invocation, capturing the flag that decides whether its stdout is
#: flushed as it is written or held in a buffer until exit.
ANALYZER_CALL = re.compile(r"python3( -u)?\s+scripts/arch_zoo_analyze\.py")


def test_commands_are_joined_across_line_continuations():
    """A wrapped invocation is one command, not its first line.

    Every runbook here wraps for width with a trailing backslash, so a parser that
    split on newlines first would read each invocation as a bare `--experiment-id`
    and report every arm as undocumented.
    """
    notes = (
        "```\n"
        "PYTHONPATH=src python scripts/arch_zoo_analyze.py \\\n"
        "  --experiment-id arm-a --also arm-b \\\n"
        "  --replication-rule --write\n"
        "```\n"
    )
    assert commands_in(notes) == [
        ["--experiment-id", "arm-a", "--also", "arm-b", "--replication-rule", "--write"]
    ]


def test_an_unlabelled_fence_is_still_a_runbook():
    """Most of these NOTES use bare fences.

    A pattern requiring a ```bash label finds nothing, and finding nothing reads as
    "this arm documents no command" - which would send a reader hunting for a missing
    runbook instead of at a parser that was too narrow.
    """
    notes = "```\nPYTHONPATH=src python scripts/arch_zoo_analyze.py --experiment-id arm-a --write\n```"
    assert commands_in(notes) == [["--experiment-id", "arm-a", "--write"]]


def test_several_commands_in_one_fence_are_separated():
    notes = (
        "```\n"
        "python scripts/arch_zoo_analyze.py --experiment-id arm-a --write\n"
        "\n"
        "python scripts/arch_zoo_analyze.py --experiment-id arm-b --write\n"
        "```\n"
    )
    assert [argv[1] for argv in commands_in(notes)] == ["arm-a", "arm-b"]


def test_a_command_is_matched_to_the_arm_whose_directory_it_writes(tmp_path):
    """A confirmation arm's NOTES may document its parent's pooled write.

    That is legitimate - the union of the two row sets is real, and a reader has to be
    able to run it - but it is not what this check looks for, which is the command
    that produces the file in *this* directory.
    """
    notes = tmp_path / "NOTES.md"
    notes.write_text(
        "```\n"
        "python scripts/arch_zoo_analyze.py --experiment-id parent --also child --write\n"
        "python scripts/arch_zoo_analyze.py --experiment-id child --write\n"
        "```\n"
    )
    assert documented_commands(notes, "child") == [["--experiment-id", "child", "--write"]]
    assert documented_commands(notes, "parent") == [
        ["--experiment-id", "parent", "--also", "child", "--write"]
    ]
    assert documented_commands(notes, "absent") == []


def test_source_names_reads_every_pooled_directory():
    assert source_names(["--experiment-id", "a", "--also", "b,c", "--write"]) == ["a", "b", "c"]
    assert source_names(["--experiment-id", "a", "--write"]) == ["a"]


def test_candidates_are_ordered_so_the_record_producing_one_goes_first(tmp_path):
    """Trying the wrong candidate costs a whole analysis.

    An arm can document both the three-seed read it opened with and the pooled
    six-seed read its confirmation bought; only one of them produced the file on disk.
    The seeds each would read are known from the ledgers it globs before anything
    runs, so the matching one goes first rather than being reached by elimination.
    """
    root = tmp_path / "root"
    for name, seeds in (("parent", (42, 43)), ("child", (45, 46))):
        directory = root / name
        directory.mkdir(parents=True)
        rows = "\n".join('{"seed": %d, "validation_loss": 3.0}' % seed for seed in seeds)
        (directory / f"flat1200_seed{seeds[0]}_results.jsonl").write_text(rows + "\n")

    arm = tmp_path / "arm"
    arm.mkdir()
    (arm / "ranked.json").write_text('{"seeds": [42, 43]}')

    standalone = ["--experiment-id", "parent", "--write"]
    pooled = ["--experiment-id", "parent", "--also", "child", "--write"]
    assert ordered_candidates([pooled, standalone], root, arm)[0] == standalone
    # And the other way round, from the same inputs: the record decides.
    (arm / "ranked.json").write_text('{"seeds": [42, 43, 45, 46]}')
    assert ordered_candidates([pooled, standalone], root, arm)[0] == pooled


def test_every_watcher_command_is_documented_by_the_arm_it_writes():
    """A watcher writes a record unattended, so its command belongs in the runbook.

    This is how the horizon arm was caught: its watcher ran `--offset-against` - the
    pre-registered precision-offset secondary - while its NOTES documented the same
    command without it. The record the watcher would write could not then have been
    reproduced from the documentation, and a reader following the runbook would never
    have seen the secondary at all. Neither the fence parser nor a textual diff can
    find that, because a watcher names its arm through `$EXP`.

    The invariant is one-directional. A watcher may not run anything the runbook does
    not document; the runbook may document more, because a person runs it by hand -
    the finalist watcher deliberately logs instead of writing, and its runbook carries
    the `--write` that produced the record.
    """
    root = Path(__file__).resolve().parent.parent
    watchers = sorted((root / "scripts").glob("run_*_when_*.sh"))
    assert watchers, "expected the post-run watchers"
    for watcher in watchers:
        commands = script_commands(watcher.read_text())
        assert commands, f"{watcher.name} runs no analysis"
        for argv in commands:
            experiment = option(argv, "--experiment-id")
            documented = documented_commands(
                root / "artifacts" / experiment / "NOTES.md", experiment
            )
            assert documented, f"{watcher.name} analyses {experiment}, which documents no command"
            # The documented invocation that shares the most with this one, so the
            # message names exactly what is missing rather than the whole flag set.
            best = max(documented, key=lambda entry: len(set(argv) & set(entry)))
            missing = sorted(set(argv) - set(best))
            assert not missing, (
                f"{watcher.name} runs {missing} for {experiment}, which its runbook does "
                "not document"
            )


def _arm_with_ledgers(root: Path, name: str, seeds: tuple[int, ...]) -> Path:
    """A scratch arm directory holding one ledger whose rows carry `seeds`."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    rows = "\n".join('{"seed": %d, "validation_loss": 3.0}' % seed for seed in seeds)
    (directory / f"flat1200_seed{seeds[0]}_results.jsonl").write_text(rows + "\n")
    return directory


def test_a_record_read_over_a_seed_set_its_runbook_does_not_read_is_named(tmp_path):
    """The horizon arm's landing, as a unit test.

    Its driver imported the kernel's own `decision.md` - whichever sweep finished
    last, and explicitly not the record - and the post-run analysis that *is* the
    record was killed inside the 4 seconds it runs for. So a single-seed file was
    left in place as the record of a two-seed arm, telling a reader to run two seeds
    per architecture about an arm that had already run two.

    The two disagree in a way that needs no regeneration to see: the record says one
    seed, the runbook reads two. That is the check, and it has to name the cause
    rather than the symptom, because a byte diff reports only "does not reproduce"
    after paying for a full analysis.
    """
    root = tmp_path / "artifacts"
    arm = _arm_with_ledgers(root, "arm", (42, 43))
    (arm / "ranked.json").write_text('{"seeds": [43]}')
    notes = arm / "NOTES.md"
    notes.write_text("```\npython scripts/arch_zoo_analyze.py --experiment-id arm --write\n```")

    message = record_read_the_wrong_rows(root, arm, documented_commands(notes, "arm"))
    assert message, "a single-seed record over two-seed ledgers must not pass"
    assert "[43]" in message and "42, 43" in message


def test_a_pooled_record_matches_the_pooling_command_it_was_written_by(tmp_path):
    """Equality per command, not containment against the union.

    v1's record is read over six seeds while its own directory holds three; the other
    three are the confirmation arm's, named by `--also` in the documented command. So
    requiring the record to equal the union of everything reachable would fail a
    legitimate arm - and requiring containment would have passed the horizon defect,
    whose record was a strict subset of its ledgers.
    """
    root = tmp_path / "artifacts"
    parent = _arm_with_ledgers(root, "parent", (42, 43, 44))
    _arm_with_ledgers(root, "child", (45, 46, 47))
    (parent / "ranked.json").write_text('{"seeds": [42, 43, 44, 45, 46, 47]}')
    notes = parent / "NOTES.md"
    notes.write_text(
        "```\n"
        "python scripts/arch_zoo_analyze.py --experiment-id parent --write\n"
        "python scripts/arch_zoo_analyze.py --experiment-id parent --also child --write\n"
        "```\n"
    )
    candidates = documented_commands(notes, "parent")
    # The standalone read over this arm's own three seeds is one of the candidates and
    # does not match, which is exactly why the comparison is per command.
    assert [option(argv, "--also") or "" for argv in candidates] == ["", "child"]
    assert record_read_the_wrong_rows(root, parent, candidates) is None


def test_check_arm_reports_a_wrong_row_set_before_paying_for_a_regeneration(tmp_path):
    """The diagnosis is cheap and specific, not a byte diff reached the expensive way.

    A full regeneration would also catch a record read over the wrong rows, but it
    reports the symptom - "decision.md does not reproduce" - after running an analysis
    that had no chance of agreeing. Deciding it from the ledgers first means the failure
    arrives in milliseconds and says which seed sets disagree, which is the whole
    difference between a reader knowing what happened and having to find out.
    """
    artifacts = tmp_path / "artifacts"
    arm = _arm_with_ledgers(artifacts, "arm", (42, 43))
    (arm / "ranked.json").write_text('{"seeds": [43]}')
    (arm / "decision.md").write_text("stale\n")
    (arm / "NOTES.md").write_text(
        "```\npython scripts/arch_zoo_analyze.py --experiment-id arm --write\n```"
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    scratch_tree(artifacts, scratch)

    result = check_arm(tmp_path, scratch, arm, strict_rank=True)
    assert not result.ok
    assert "wrong row set" in result.message
    assert any("[43]" in failure for failure in result.failures)


def test_every_watcher_runs_its_analysis_unbuffered():
    """The record of a killed run is the buffer that died with it.

    Every watcher redirects the analysis's stdout into its own log. That redirection
    makes stdout *block*-buffered, so a kill inside the analysis discards every line,
    including the exit line that would have reported the failure. Measured in this
    repository: the same print into the same redirection leaves 0 bytes when killed,
    and 20 bytes under `-u`, while stderr survives either way - which is why the
    horizon arm's watcher left nothing at all (no traceback, stdout lost) where the
    finalist watcher's log kept a full traceback of the analyzer defect it caught.

    The finalist watcher is why this is not cosmetic: it deliberately does not write a
    `decision.md`, so its log *is* the arm's record, and an unbuffered run is the only
    thing that keeps it.
    """
    assert WATCHERS, "expected the post-run watchers"
    for watcher in WATCHERS:
        calls = ANALYZER_CALL.findall(watcher.read_text())
        assert calls, f"{watcher.name} runs no analysis"
        assert all(flag == " -u" for flag in calls), (
            f"{watcher.name} runs the analyzer without -u, so a killed run leaves no "
            "record of having been attempted"
        )


def test_every_watcher_that_writes_a_record_verifies_the_landing():
    """An exit status is not a landing.

    A 0 says the analysis finished; it says nothing about whether it wrote anything.
    The failure that costs hours is a record that was never written, leaving an older
    artifact in place while the log reports success - which is precisely what the
    horizon arm's landing did, with the driver's imported kernel-side file standing as
    the record of a two-seed arm, and with a log that stopped mid-line rather than
    reporting a failure.

    So a watcher that writes a record has to confirm the landing, and the check it
    runs has to be able to fail it. This asserts the *repository's own* record check
    rather than any hand-rolled test of the file, for two reasons: that check is
    strictly stronger (it proves the record is byte-identical to what the documented
    command produces over the ledgers on disk, not merely that the file changed), and
    it is already covered by the rest of this module, so the watchers inherit a
    tested guard instead of re-implementing one in shell per arm.

    Writers are identified through the runbook parser rather than by substring,
    because the finalist watcher says "Deliberately *not* --write" in a comment - a
    substring match reads that prose as a flag and demands a verification from the one
    watcher that is right to have none.
    """
    writers = [
        watcher
        for watcher in WATCHERS
        if any("--write" in argv for argv in script_commands(watcher.read_text()))
    ]
    assert writers, "expected at least one watcher that writes a record"
    for watcher in writers:
        text = watcher.read_text()
        assert "check_record_freshness.py" in text, (
            f"{watcher.name} writes a record and never confirms it reproduces"
        )
        assert re.search(r"^\s*exit [1-9]", text, re.M), (
            f"{watcher.name} confirms its record but cannot fail on the result"
        )


def test_every_watcher_resolves_the_repository_root():
    """These commands name every path relative to the repository root.

    Launched from anywhere else a watcher finds no ledgers, so its completeness check
    reports a partial arm and it refuses - a safe failure, but a false one, and it is
    the kind a reader attributes to the arm rather than to how it was launched. One of
    the three did exactly this. Resolving the script's own directory is what makes the
    answer depend on where the script lives rather than on who called it.
    """
    assert WATCHERS, "expected the post-run watchers"
    for watcher in WATCHERS:
        assert 'cd "$(dirname "$0")/.."' in watcher.read_text(), (
            f"{watcher.name} does not resolve its own directory"
        )


def test_every_arm_with_a_record_documents_the_command_that_writes_it():
    """The standing guard, and cheap enough to run in the suite.

    This is the half of the check that needs no analysis: if an arm's NOTES give no
    invocation whose `--experiment-id` is that arm, then nobody can reproduce the
    record from the documentation, whatever the record contains. Both arms that failed
    this when it was written were documenting their *parent's* pooled write and
    nothing about themselves.
    """
    arms = sorted(
        path
        for path in ARTIFACTS.glob("arch-zoo-*")
        if path.is_dir()
        and (path / "decision.md").exists()
        and any(path.glob("*_seed*_results.jsonl"))
    )
    assert arms, "expected committed arms with ledgers"
    undocumented = [
        arm.name for arm in arms if not documented_commands(arm / "NOTES.md", arm.name)
    ]
    assert undocumented == []
