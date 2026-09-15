"""Tests for the Kaggle queue driver (``scripts/kaggle_queue.py``).

The driver decides when a multi-session queue is finished, so the two things
worth pinning are the completion signal (which must be cumulative, and must
not count an interrupted pair) and the status parser, which is fed the CLI's
own prose rather than a machine-readable field.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import kaggle_queue  # noqa: E402


@pytest.fixture(autouse=True)
def _default_mount_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ACT_T4_OWNER", raising=False)
    monkeypatch.delenv("ACT_T4_SLUG", raising=False)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ('sid123123234/act-t4-session has status "KernelWorkerStatus.RUNNING"', "RUNNING"),
        ('sid123123234/act-t4-session has status "KernelWorkerStatus.QUEUED"', "QUEUED"),
        ('sid123123234/act-t4-session has status "KernelWorkerStatus.COMPLETE"', "COMPLETE"),
        ('sid123123234/act-t4-session has status "KernelWorkerStatus.ERROR"', "ERROR"),
        ("something unexpected", ""),
    ],
)
def test_parse_status_reads_the_cli_prose(line: str, expected: str) -> None:
    assert kaggle_queue.parse_status(line) == expected


def test_only_stopped_states_are_terminal() -> None:
    assert kaggle_queue.is_terminal("COMPLETE")
    assert kaggle_queue.is_terminal("ERROR")
    assert kaggle_queue.is_terminal("CANCEL_ACKNOWLEDGED")
    # Not started yet, or still working: keep polling.
    assert not kaggle_queue.is_terminal("QUEUED")
    assert not kaggle_queue.is_terminal("RUNNING")
    assert not kaggle_queue.is_terminal("NEW_SCRIPT")
    assert not kaggle_queue.is_terminal("")


def test_progress_counts_skips_and_completions_but_not_interruptions() -> None:
    session_log = {
        "declared_seed_queue": list(range(15000, 15012)),
        "pairs": [
            {"seed": 15000, "action": "skipped_complete"},
            {"seed": 15001, "action": "completed"},
            {"seed": 15002, "action": "interrupted"},
        ],
    }
    # Cumulative by construction: the skipped seed was published by an earlier
    # session, so the count does not depend on which session reports it.
    assert kaggle_queue.queue_progress(session_log) == (2, 12)


def test_progress_of_a_first_session_is_the_latest_published_seed() -> None:
    session_log = {
        "declared_seed_queue": list(range(15000, 15012)),
        "pairs": [{"seed": seed, "action": "completed"} for seed in range(15000, 15002)],
    }
    assert kaggle_queue.queue_progress(session_log) == (2, 12)


def test_progress_handles_an_empty_log() -> None:
    assert kaggle_queue.queue_progress({}) == (0, 0)


def test_stage_writes_a_self_contained_mode_pinned_kernel(tmp_path: Path) -> None:
    folder = kaggle_queue.stage(tmp_path / "push", "control")

    metadata = json.loads((folder / "kernel-metadata.json").read_text())
    assert metadata["id"] == "sid123123234/act-t4-control"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    entry = folder / metadata["code_file"]
    assert entry.is_file()
    # A script kernel receives only code_file, so the mode must live inside it.
    source = entry.read_text()
    assert 'os.environ.setdefault(MODE_ENV, \'control\')' in source
    assert "raise SystemExit(main())" in source
    # And it must actually be runnable Python once assembled.
    compile(source, str(entry), "exec")


def test_staged_kernel_is_pinned_per_queue(tmp_path: Path) -> None:
    for mode in ("session", "null", "verdict"):
        folder = kaggle_queue.stage(tmp_path / mode, mode)
        metadata = json.loads((folder / "kernel-metadata.json").read_text())
        assert f"MODE_ENV, '{mode}'" in (folder / metadata["code_file"]).read_text()


def test_stage_fails_loudly_when_the_guard_moves(tmp_path: Path, monkeypatch) -> None:
    # If the driver's __main__ guard is refactored, staging must refuse rather
    # than push a kernel that exits on the mode check without doing any work.
    driver = tmp_path / "kaggle_t4.py"
    driver.write_text("print('no guard here')\n")
    monkeypatch.setattr(kaggle_queue, "__file__", str(tmp_path / "kaggle_queue.py"))

    with pytest.raises(SystemExit, match="guard drifted"):
        kaggle_queue.stage(tmp_path / "push", "session")


def test_find_session_log_prefers_the_output_root_copy(tmp_path: Path) -> None:
    (tmp_path / "session-log.json").write_text("{}")
    nested = tmp_path / "act-lab" / "artifacts" / "v0-sessions"
    nested.mkdir(parents=True)
    (nested / "session-log.json").write_text("{}")

    assert kaggle_queue.find_session_log(tmp_path) == tmp_path / "session-log.json"


def test_find_session_log_returns_none_when_absent(tmp_path: Path) -> None:
    assert kaggle_queue.find_session_log(tmp_path) is None


def test_load_session_log_round_trips(tmp_path: Path) -> None:
    payload = {"declared_seed_queue": [15000], "pairs": []}
    (tmp_path / "session-log.json").write_text(json.dumps(payload))

    assert kaggle_queue.load_session_log(tmp_path) == payload


def test_unknown_queue_is_rejected() -> None:
    with pytest.raises(SystemExit):
        kaggle_queue.main(["--queue", "not-a-queue"])
