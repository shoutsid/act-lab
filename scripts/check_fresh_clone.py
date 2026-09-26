"""Run the documented checks against a fresh clone of the committed tree.

The defect this exists for showed up the first time anybody asked the question.
Every check in `AGENTS.md` was green on the machine that wrote it, and the suite
had **33 failures** in a checkout that had never run the research: those tests
read evidence the blanket `artifacts/*` rule keeps as local scratch (the
`toy-null-study` shards, `real-lm/tokens.npy`), and from inside the tree a file
that is *present* is indistinguishable from a file that is *committed*. Nothing
was wrong with any individual check; the checks were being run somewhere no
reader can run them.

A clone rather than a filter on the working tree, deliberately: it is exactly
what a reader gets, the tracked/untracked line is git's to draw rather than this
script's, and no check has to know it is being run this way.

It clones **HEAD**, so run it after committing. Uncommitted edits are not part
of the check, which is the point - HEAD is what everyone else receives - and a
dirty tree is reported so the result is not read as being about the working
copy.

Usage::

    PYTHONPATH=src python scripts/check_fresh_clone.py
    PYTHONPATH=src python scripts/check_fresh_clone.py --only pytest
    PYTHONPATH=src python scripts/check_fresh_clone.py --keep     # inspect the clone

Exit status is 0 when every check passes, 1 when one fails, and 2 when the clone
itself could not be made (which is a problem with the environment, not a finding
about the tree).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Check:
    """One documented command, as the arguments after the interpreter."""

    name: str
    argv: tuple[str, ...]
    note: str


#: The checks `AGENTS.md` tells a reader to run, in the order it lists them.
CHECKS: tuple[Check, ...] = (
    Check("verify_record", ("scripts/verify_record.py",), "claims vs ledgers"),
    Check(
        "record_freshness",
        ("scripts/check_record_freshness.py", "--strict-rank"),
        "records regenerate",
    ),
    Check(
        "notebooks",
        ("scripts/build_notebooks.py", "--check"),
        "notebooks match the builder",
    ),
    Check("pytest", ("-m", "pytest", "-q"), "the suite"),
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True
    )


def head_commit(repo: Path) -> str:
    """The commit that would be cloned, or an empty string if this is not a repo."""
    completed = _git(repo, "rev-parse", "HEAD")
    return completed.stdout.strip() if completed.returncode == 0 else ""


def working_tree_is_dirty(repo: Path) -> bool | None:
    """Whether uncommitted work exists, or ``None`` when git cannot say."""
    completed = _git(repo, "status", "--porcelain")
    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())


def clone_head(repo: Path, destination: Path) -> None:
    """Clone the committed tree. Raises ``CalledProcessError`` on failure."""
    subprocess.run(
        ["git", "clone", "--quiet", str(repo), str(destination)],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def run_check(clone: Path, check: Check) -> tuple[bool, str]:
    """Run one check inside the clone, returning (passed, last non-empty line)."""
    completed = subprocess.run(
        [sys.executable, *check.argv],
        cwd=clone,
        # The documented commands are prefixed with `PYTHONPATH=src`; set it here
        # so the check runs the clone's own source rather than an installed copy.
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
    )
    output = (completed.stdout + "\n" + completed.stderr).splitlines()
    tail = [line for line in output if line.strip()]
    return completed.returncode == 0, tail[-1].strip() if tail else "(no output)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the documented checks against a fresh clone of HEAD"
    )
    parser.add_argument(
        "--source",
        default=str(REPO),
        help="the repository to clone (default: this script's repository)",
    )
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated check names to run (default: all of them)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the clone and print its path instead of deleting it",
    )
    args = parser.parse_args(argv)

    source = Path(args.source).resolve()
    wanted = {name.strip() for name in args.only.split(",") if name.strip()}
    unknown = wanted - {check.name for check in CHECKS}
    if unknown:
        raise SystemExit(
            f"unknown check(s): {sorted(unknown)}; "
            f"choose from {[check.name for check in CHECKS]}"
        )
    checks = [check for check in CHECKS if not wanted or check.name in wanted]

    commit = head_commit(source)
    if not commit:
        raise SystemExit(f"not a git checkout, so there is no committed tree: {source}")

    print(f"cloning HEAD {commit[:12]} from {source}")
    dirty = working_tree_is_dirty(source)
    if dirty:
        print(
            "  NOTE: the working tree has uncommitted changes. They are NOT part of "
            "this check - commit them and re-run to have them included."
        )

    temp_root = Path(tempfile.mkdtemp(prefix="fresh-clone-"))
    destination = temp_root / "clone"
    failures: list[str] = []
    try:
        try:
            clone_head(source, destination)
        except subprocess.CalledProcessError as error:
            print(f"could not clone {source}: {error.stderr or error}", file=sys.stderr)
            return 2

        for check in checks:
            passed, tail = run_check(destination, check)
            if not passed:
                failures.append(check.name)
            print(
                f"  {'ok  ' if passed else 'FAIL'} {check.name:<18} "
                f"{check.note:<28} {tail}"
            )

        if failures:
            print(
                f"\nFAILED {len(failures)}/{len(checks)}: {failures}. A check that "
                "passes only where the research ran is not a check a reader can run.",
                file=sys.stderr,
            )
            if not args.keep:
                print(
                    "  re-run with --keep to inspect the clone that failed",
                    file=sys.stderr,
                )
            return 1

        print(f"\n{len(checks)}/{len(checks)} checks pass in a fresh clone of {commit[:12]}")
        return 0
    finally:
        # Clean the parent even when `git clone` fails before it creates destination.
        # With --keep, only preserve a successfully-created clone.
        if args.keep and destination.exists():
            print(f"clone kept at {destination}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
