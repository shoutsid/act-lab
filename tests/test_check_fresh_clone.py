"""Tests for the fresh-clone check (scripts/check_fresh_clone.py).

The script's whole claim is that it sees something the working tree cannot: a
test that passes *here* because the file it reads is on this disk. These tests
pin that claim in both directions, on a throwaway repository - a committed test
reading an ignored, untracked artifact is shown to pass in that repository's own
tree and to fail in the clone the check makes - and pin the parts that decide
what the check is about (HEAD rather than the working copy) and what it does
with the clone it made.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import check_fresh_clone as check  # noqa: E402


@pytest.fixture(autouse=True)
def _require_git() -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not available, so there is no clone to make")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _commit(repo: Path, message: str) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)


def _make_repo(tmp_path: Path) -> Path:
    """A repository whose test suite passes here and only here.

    `artifacts/*` is ignored by its `.gitignore` and never committed, exactly as
    this project's own rule treats a run's output - so a test that reads it is
    green in this working tree and cannot pass in a checkout.
    """
    repo = tmp_path / "source"
    (repo / "tests").mkdir(parents=True)
    (repo / "artifacts").mkdir()
    (repo / ".gitignore").write_text("artifacts/*\n")
    (repo / "artifacts" / "local.txt").write_text("present here, never committed\n")
    (repo / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    _commit(repo, "initial")
    return repo


def _run(repo: Path, *extra: str) -> int:
    return check.main(["--source", str(repo), "--only", "pytest", *extra])


def _pytest_in(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=repo,
        capture_output=True,
        text=True,
    )


def test_a_repository_whose_tests_need_nothing_passes(tmp_path: Path) -> None:
    assert _run(_make_repo(tmp_path)) == 0


def test_it_catches_a_test_that_reads_an_uncommitted_artifact(tmp_path: Path) -> None:
    """The defect this exists for, reproduced: green here, red in a checkout.

    Both directions are asserted, because only the contrast is the finding - on
    its own, a failing clone run is indistinguishable from a genuinely broken
    test.
    """
    repo = _make_repo(tmp_path)
    (repo / "tests" / "test_ok.py").write_text(
        "from pathlib import Path\n"
        "\n"
        "def test_reads_the_local_artifact():\n"
        "    assert Path('artifacts/local.txt').read_text().strip()\n"
    )
    _commit(repo, "read the local artifact")

    # It passes in the tree that has the file...
    assert _pytest_in(repo).returncode == 0
    # ...and the check, which only has the committed files, says so.
    assert _run(repo) == 1


def test_it_clones_head_rather_than_the_working_copy(tmp_path: Path) -> None:
    """A fix that has not been committed is not part of what a reader receives."""
    repo = _make_repo(tmp_path)
    (repo / "tests" / "test_ok.py").write_text("def test_ok():\n    assert False\n")
    _commit(repo, "a committed failure")
    # Fixed here, uncommitted: the check must still see the committed version.
    (repo / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n")

    assert _pytest_in(repo).returncode == 0
    assert _run(repo) == 1


def test_a_source_that_is_not_a_checkout_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as info:
        check.main(["--source", str(tmp_path), "--only", "pytest"])
    assert "not a git checkout" in str(info.value)


def test_an_unknown_check_name_is_refused(tmp_path: Path) -> None:
    """A typo must not silently narrow the check to the names that parsed."""
    repo = _make_repo(tmp_path)
    with pytest.raises(SystemExit) as info:
        check.main(["--source", str(repo), "--only", "pytest,not_a_check"])
    assert "unknown check" in str(info.value)


def test_the_clone_is_removed_unless_it_is_kept(tmp_path: Path, capsys) -> None:
    repo = _make_repo(tmp_path)
    before = set(Path(tempfile.gettempdir()).glob("fresh-clone-*"))

    assert _run(repo) == 0
    assert set(Path(tempfile.gettempdir()).glob("fresh-clone-*")) == before

    assert _run(repo, "--keep") == 0
    kept = Path(capsys.readouterr().out.split("clone kept at ", 1)[1].strip())
    try:
        assert kept.is_dir(), kept
    finally:
        shutil.rmtree(kept.parent, ignore_errors=True)


def test_the_check_list_agrees_with_the_document() -> None:
    """The script exists to run the checks the documents list, so drift between
    the two is the defect class it was written for and the list is pinned to
    them rather than to itself.

    `AGENTS.md` is the copy that names the commands as a set, and it is
    **gitignored on purpose** - it is the local agent contract, not project
    content - so this is one of the checks that can only run where it is. It
    skips by name there, which is the honest report: a fresh clone has nothing
    to disagree with.
    """
    agents = ROOT / "AGENTS.md"
    if not agents.is_file():
        pytest.skip("AGENTS.md is local-only (gitignored), so there is nothing to pin against")
    text = agents.read_text()
    for name, command in (
        ("verify_record", "scripts/verify_record.py"),
        ("record_freshness", "scripts/check_record_freshness.py"),
        ("notebooks", "scripts/build_notebooks.py"),
        ("pytest", "-m pytest"),
    ):
        assert command in text, f"{command} is no longer documented in AGENTS.md"
        declared = next(item for item in check.CHECKS if item.name == name)
        assert command in " ".join(declared.argv), (name, declared.argv)
