"""Tests for the record-freshness check.

The check exists because an analyzer defect left one arm's committed `decision.md`
unregenerable and nothing noticed: `verify_record.py` recomputes each claim from the
*ledgers*, so it never re-renders the artifact the claim is read from, and a record
that cannot be regenerated cannot be audited. These pin the two halves that can rot
without anyone seeing it - the runbook parser, and the requirement that every arm
document the command that writes its own record.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_record_freshness import (  # noqa: E402
    commands_in,
    documented_commands,
    ordered_candidates,
    source_names,
)

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


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
