"""Check that every committed arm's decision record is still reproducible.

The defect this exists for is not hypothetical. ``joint_structural_fit`` emitted
``genes`` as coefficient rows on one path and as bare gene *names* on another, and a
consumer indexed that slot as rows, so ``arch_zoo_analyze.py`` raised ``TypeError``
for a whole arm's ledger data - including ``arch-zoo-100m-v1``, whose committed
``decision.md`` could no longer be regenerated at all. Nothing noticed, because the
only thing reading the record was ``verify_record.py``, and that recomputes each claim
from the **ledgers** rather than re-rendering the artifact the claim is read from. A
record that cannot be regenerated is a record that cannot be audited.

So this runs the *documented* command for each arm - parsed out of that arm's own
``NOTES.md``, so the runbook and the record cannot drift apart - against the arm's
ledgers in a scratch tree, and diffs the result against what is committed.

``decision.md`` must match byte for byte. It is the artifact the phase's claims are
read from, and it is deterministic: every committed arm reproduces exactly, including
the Monte-Carlo ones, whose draws are seeded.

``ranked.json`` is reported but not failed on. It carries a schema that legitimately
grows - newer runs record diagnostics the older files predate - so drift there is a
note about how old a file is, not a claim about the science. ``--strict-rank`` turns it
into a failure too, which is what makes "refresh this file" a check.

Two things make it fast enough to be run casually, and both are about *not* doing work
rather than about doing it differently - the analysis itself must stay bit-identical or
it could not check anything:

* the arms are independent, so they run concurrently. Serially this is ~1 minute of
  Monte-Carlo permutation; across a normal core count it is the slowest single arm.
* a candidate that cannot reproduce the record is not worth running, and the seeds a
  command would read are known from the ledgers it globs before any analysis starts.
  Without that, an arm whose record is the pooled six-seed write burns a full
  three-seed analysis on the way to the command that works.

Usage::

    PYTHONPATH=src python scripts/check_record_freshness.py            # every arm
    PYTHONPATH=src python scripts/check_record_freshness.py --only arch-zoo-100m-v1
    PYTHONPATH=src python scripts/check_record_freshness.py --serial    # for debugging
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ANALYZER = "arch_zoo_analyze.py"
#: Where the analyzer lives, relative to the repository root. The documented commands
#: are written from the repository root, so they are executed from there too.
SCRIPT = Path("scripts") / ANALYZER
#: Any fence language, including none: most of this repository's runbooks are in
#: unlabelled fences, and a labelled-only pattern silently finds nothing - which reads
#: as "the arm documents no command" rather than as a parser that was too narrow.
BASH_FENCE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.S)


@dataclass
class ArmResult:
    """What one arm's check found, kept separable so the report can be ordered."""

    name: str
    ok: bool
    message: str
    rank_detail: str = ""
    rank_drifted: bool = False
    failures: list[str] = field(default_factory=list)


def commands_in(notes: str) -> list[list[str]]:
    """Every analyzer invocation in a fenced block, as its option list.

    Blocks are wrapped for width with trailing backslashes, so the continuations are
    joined before the block is split into commands - otherwise every invocation in
    this repository would parse as its first line alone.
    """
    found: list[list[str]] = []
    for block in BASH_FENCE.findall(notes):
        for line in block.replace("\\\n", " ").splitlines():
            if ANALYZER not in line:
                continue
            argv = shlex.split(line.split(ANALYZER, 1)[1])
            if argv:
                found.append(argv)
    return found


def option(argv: list[str], name: str) -> str | None:
    """The value of a one-argument option, or ``None`` if it is absent."""
    if name not in argv:
        return None
    index = argv.index(name)
    return argv[index + 1] if index + 1 < len(argv) else None


def documented_commands(notes: Path, experiment_id: str) -> list[list[str]]:
    """Every invocation an arm's own NOTES gives that writes *that* arm's directory.

    Matched on ``--experiment-id`` rather than by position. A confirmation arm's NOTES
    legitimately documents the pooled write that lands in its *parent* - the union of
    the two row sets is real and a reader has to be able to run it - and an arm's own
    directory can hold a standalone record and a pooled one side by side, which is
    exactly the case for the v1 record. So all matches are returned and any one of
    them reproducing is enough; requiring a particular order would make the runbook's
    layout load-bearing.
    """
    return [
        argv
        for argv in commands_in(notes.read_text())
        if option(argv, "--experiment-id") == experiment_id
    ]


def source_names(argv: list[str]) -> list[str]:
    """The experiment directories a command reads, primary first."""
    names = [option(argv, "--experiment-id") or ""]
    names += [
        name.strip() for name in (option(argv, "--also") or "").split(",") if name.strip()
    ]
    return [name for name in names if name]


def expected_seeds(root: Path, argv: list[str]) -> set[int]:
    """The seeds a command would read, straight from the ledgers it globs.

    Read before the analysis rather than after it, which is the whole point: this is
    cheap enough to compute for a command that is never going to be run.
    """
    seeds: set[int] = set()
    for name in source_names(argv):
        for ledger in (root / name).glob("*_seed*_results.jsonl"):
            for line in ledger.read_text().splitlines():
                if line.strip():
                    seeds.add(int(json.loads(line)["seed"]))
    return seeds


def committed_seeds(arm: Path) -> set[int]:
    """The seeds the committed record was read over, from its own summary."""
    try:
        recorded = json.loads((arm / "ranked.json").read_text()).get("seeds") or []
    except (OSError, json.JSONDecodeError):
        return set()
    return {int(seed) for seed in recorded}


def ordered_candidates(candidates: list[list[str]], root: Path, arm: Path) -> list[list[str]]:
    """Candidates that can reproduce the committed record first, then the rest.

    An arm can document several invocations - the three-seed read it opened with and
    the pooled six-seed read its confirmation bought - and only one of them produced
    the file on disk. Trying the wrong one first costs a complete analysis, so the one
    whose seed set matches the record's goes first. A stable sort, so an arm whose
    candidates are indistinguishable keeps the order its runbook gives them.
    """
    wanted = committed_seeds(arm)
    return sorted(candidates, key=lambda argv: expected_seeds(root, argv) != wanted)


def _distinguishing(argv: list[str]) -> str:
    """The flags that tell two invocations for the same arm apart, for the report."""
    marks: list[str] = []
    other = option(argv, "--also")
    if other:
        marks.append(f"--also {other}")
    for flag in (
        "--signal-genes",
        "--replication-rule",
        "--head-rule",
        "--horizon-rule",
        "--finalist-order-rule",
    ):
        if flag in argv:
            marks.append(flag)
    return ", ".join(marks) if marks else "own ledgers, no rule"


def scratch_tree(artifacts: Path, root: Path) -> None:
    """Symlink every arm's *inputs* into the scratch root.

    Only the two inputs the analyzer reads, and only as symlinks: a copied or linked
    directory would let a stray ``--write`` overwrite the committed record this is
    meant to be comparing against.
    """
    for arm in sorted(path for path in artifacts.glob("arch-zoo-*") if path.is_dir()):
        target = root / arm.name
        target.mkdir(parents=True, exist_ok=True)
        for pattern in ("*_seed*_results.jsonl", "zoo_definition.json"):
            for path in arm.glob(pattern):
                (target / path.name).symlink_to(path.resolve())


def regenerate(repo: Path, root: Path, argv: list[str]) -> tuple[int, str]:
    """Run one documented command against the scratch tree. Always writes."""
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--artifacts-root", str(root), *argv, "--write"],
        cwd=repo,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
    )
    return completed.returncode, (completed.stderr or completed.stdout)[-800:]


def differing_keys(fresh: Path, committed: Path) -> list[str]:
    """Top-level keys whose values differ, so schema growth is named not guessed."""
    try:
        left = json.loads(fresh.read_text())
        right = json.loads(committed.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return [f"unreadable: {error}"]
    return sorted(
        key
        for key in set(left) | set(right)
        if left.get(key, "<absent>") != right.get(key, "<absent>")
    )


def check_arm(repo: Path, root: Path, arm: Path, *, strict_rank: bool) -> ArmResult:
    """Regenerate one arm from its documented command and diff it against the commit."""
    notes_path = arm / "NOTES.md"
    candidates = documented_commands(notes_path, arm.name) if notes_path.exists() else []
    if not candidates:
        # Not a rendering defect: an arm whose runbook documents only the pooled
        # write into its parent leaves a reader unable to reproduce the file in
        # front of them.
        return ArmResult(
            arm.name,
            False,
            "no self-targeting command in NOTES.md",
            failures=[f"{arm.name}: NOTES.md documents no command for this arm"],
        )

    fresh = root / arm.name
    last_error = "no documented command reproduced decision.md"
    for argv in ordered_candidates(candidates, root, arm):
        status, tail = regenerate(repo, root, argv)
        if status != 0 or not (fresh / "decision.md").exists():
            last = tail.strip().splitlines()[-1][:140] if tail.strip() else ""
            last_error = f"the documented command exited {status}: {last}"
            continue
        if (fresh / "decision.md").read_bytes() != (arm / "decision.md").read_bytes():
            last_error = "decision.md does not reproduce"
            continue
        drift = (
            differing_keys(fresh / "ranked.json", arm / "ranked.json")
            if (arm / "ranked.json").exists() and (fresh / "ranked.json").exists()
            else []
        )
        detail = (
            "ranked.json identical"
            if not drift
            else f"ranked.json differs in {len(drift)} key(s): {', '.join(drift[:4])}"
        )
        failures = [f"{arm.name}: ranked.json does not reproduce"] if drift and strict_rank else []
        return ArmResult(
            arm.name,
            not failures,
            "decision.md identical" if not failures else "ranked.json DIFFERS",
            rank_detail=f"{detail}  ({_distinguishing(argv)})",
            rank_drifted=bool(drift),
            failures=failures,
        )

    return ArmResult(arm.name, False, last_error, failures=[f"{arm.name}: {last_error}"])


def arms_with_records(artifacts: Path) -> list[Path]:
    return sorted(
        path
        for path in artifacts.glob("arch-zoo-*")
        if path.is_dir()
        and (path / "decision.md").exists()
        and any(path.glob("*_seed*_results.jsonl"))
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check each arm's committed decision.md against its ledgers"
    )
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--only", default="", help="comma-separated experiment ids")
    parser.add_argument(
        "--strict-rank",
        action="store_true",
        help="fail on ranked.json drift too, not only on decision.md drift",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="check the arms one after another in this process (for debugging)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the scratch tree and print its path instead of deleting it",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    artifacts = repo / args.artifacts_root
    wanted = {name.strip() for name in args.only.split(",") if name.strip()}
    arms = [arm for arm in arms_with_records(artifacts) if not wanted or arm.name in wanted]
    if not arms:
        raise SystemExit(f"no arm with a record and ledgers under {artifacts}")

    root = Path(tempfile.mkdtemp(prefix="record-freshness-"))
    results: list[ArmResult] = []
    failures: list[str] = []
    try:
        scratch_tree(artifacts, root)
        if args.serial:
            results = [
                check_arm(repo, root, arm, strict_rank=args.strict_rank) for arm in arms
            ]
        else:
            # Threads, not processes: the work is in a subprocess per arm, so the GIL
            # is not held while it runs. Each arm writes only its own scratch
            # directory, so concurrent arms never contend for a file.
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(arms)) as pool:
                futures = [
                    pool.submit(check_arm, repo, root, arm, strict_rank=args.strict_rank)
                    for arm in arms
                ]
                results = [future.result() for future in futures]

        drifting: list[str] = []
        for result in results:
            failures.extend(result.failures)
            if result.rank_drifted and not args.strict_rank:
                drifting.append(f"{result.name}: {result.rank_detail}")
            print(
                f"  {'ok  ' if result.ok else 'FAIL'} {result.name:<40} {result.message}"
                + (f"\n       {'':<40} {result.rank_detail}" if result.ok else "")
            )

        print(
            f"\n{sum(1 for r in results if r.ok)}/{len(results)} committed records reproduce "
            "from their own documented command"
        )
        if drifting:
            print(
                "\nranked.json is an older schema in these arms - additive keys only, no claim "
                "changes. Re-run the documented command with --write in the arm's directory to "
                "refresh it:"
            )
            for line in drifting:
                print(f"  - {line}")
        if failures:
            print("\nNot reproducible (the runbook or the artifact is wrong, and which must be decided):")
            for line in failures:
                print(f"  - {line}")
    finally:
        if args.keep:
            print(f"\nscratch tree kept at {root}")
        else:
            subprocess.run(["rm", "-rf", str(root)], check=False)

    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
