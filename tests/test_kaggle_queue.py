"""Tests for the Kaggle queue driver (``scripts/kaggle_queue.py``).

The driver decides when a multi-session queue is finished, so the two things
worth pinning are the completion signal (which must be cumulative, and must
not count an interrupted pair) and the status parser, which is fed the CLI's
own prose rather than a machine-readable field.
"""

from __future__ import annotations

import json
import subprocess
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


def test_a_live_session_is_adopted_rather_than_duplicated() -> None:
    """Pushing a version does not stop the running one.

    Measured 2026-09-15: a second push left the first session running and the
    GPU burn doubled (2.00x against 1.00x). Two sessions of one queue would
    also race for the same dataset and the same handoff roots.
    """
    assert kaggle_queue.should_adopt("RUNNING")
    assert kaggle_queue.should_adopt("QUEUED")
    assert kaggle_queue.should_adopt("NEW_SCRIPT")
    # Finished, cancelled, or never pushed: the driver should push.
    assert not kaggle_queue.should_adopt("COMPLETE")
    assert not kaggle_queue.should_adopt("ERROR")
    assert not kaggle_queue.should_adopt("CANCEL_ACKNOWLEDGED")
    assert not kaggle_queue.should_adopt("")


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


def test_staged_kernel_pins_the_declared_window(tmp_path: Path) -> None:
    """``--hours`` used to be a no-op: the kernel reads it from its own env.

    Exporting ``ACT_T4_HOURS`` on the pushing side only sets it there, so the
    kernel fell back to its default; the quota gate compares this same number,
    so a silent fallback would make the gate unsound.
    """
    folder = kaggle_queue.stage(tmp_path / "push", "session", 6.5)
    metadata = json.loads((folder / "kernel-metadata.json").read_text())
    source = (folder / metadata["code_file"]).read_text()

    assert source.endswith(
        'if __name__ == "__main__":\n'
        "    os.environ.setdefault(MODE_ENV, 'session')\n"
        "    os.environ.setdefault(HOURS_ENV, '6.5')\n"
        "    raise SystemExit(main())\n"
    )
    compile(source, str(folder / metadata["code_file"]), "exec")


def test_staged_kernel_without_a_window_keeps_the_kernel_default(tmp_path: Path) -> None:
    folder = kaggle_queue.stage(tmp_path / "push", "session")
    metadata = json.loads((folder / "kernel-metadata.json").read_text())
    assert "setdefault(HOURS_ENV" not in (folder / metadata["code_file"]).read_text()


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


QUOTA_PAYLOAD = json.dumps(
    [
        {
            "resource": "GPU",
            "used": "1.24h",
            "remaining": "28.76h",
            "total": "30.00h",
            "refreshAt": "2026-09-19T00:00:00",
        },
        {
            "resource": "TPU",
            "used": "1.83h",
            "remaining": "18.17h",
            "total": "20.00h",
            "refreshAt": "2026-09-19T00:00:00",
        },
    ]
)


def test_parse_quota_reads_the_resource_line() -> None:
    quota = kaggle_queue.parse_quota(QUOTA_PAYLOAD)

    assert quota == {
        "remaining_hours": 28.76,
        "refresh_at": "2026-09-19T00:00:00",
    }
    # The TPU line is a different accelerator and must not be read as the GPU.
    assert kaggle_queue.parse_quota(QUOTA_PAYLOAD, "TPU")["remaining_hours"] == 18.17


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        "{}",
        "[]",
        json.dumps([{"resource": "CPU", "remaining": "1h"}]),
        json.dumps([{"resource": "GPU", "remaining": "unknown"}]),
        json.dumps([{"resource": "GPU"}]),
    ],
)
def test_parse_quota_returns_none_when_it_cannot_tell(payload: str) -> None:
    assert kaggle_queue.parse_quota(payload) is None


def test_quota_shortfall_blocks_only_when_the_week_cannot_cover_it() -> None:
    quota = {"remaining_hours": 28.76, "refresh_at": "2026-09-19T00:00:00"}
    assert kaggle_queue.quota_shortfall(quota, 9.0) is None
    # Exactly the declared window still counts as coverable.
    assert kaggle_queue.quota_shortfall({"remaining_hours": 9.0}, 9.0) is None

    short = kaggle_queue.quota_shortfall(quota | {"remaining_hours": 2.0}, 9.0)
    assert short is not None
    assert "2 GPU-h" in short and "9 h" in short
    assert "2026-09-19" in short


def test_an_unreadable_quota_line_does_not_block_a_session() -> None:
    """Cannot-tell must not become a stall: the gate only ever spends less."""
    assert kaggle_queue.quota_shortfall(None, 9.0) is None


def test_drive_refuses_to_push_when_quota_cannot_cover_the_session(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        if argv[1] == "quota":
            return subprocess.CompletedProcess(
                argv,
                0,
                json.dumps(
                    [
                        {
                            "resource": "GPU",
                            "used": "28h",
                            "remaining": "2h",
                            "total": "30h",
                            "refreshAt": "2026-09-19T00:00:00",
                        }
                    ]
                ),
                "",
            )
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(kaggle_queue, "_run", fake_run)

    code = kaggle_queue.drive("session", hours=9.0, poll=0.0, timeout=1.0, max_sessions=1)

    assert code == kaggle_queue.EXIT_QUOTA_INSUFFICIENT
    assert not [argv for argv in calls if argv[1:3] == ["kernels", "push"]]


def test_drive_still_pushes_when_the_week_can_cover_the_session(monkeypatch) -> None:
    """Positive control: the gate must not become a reason not to run."""
    calls: list[list[str]] = []

    def fake_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        if argv[1] == "quota":
            return subprocess.CompletedProcess(argv, 0, QUOTA_PAYLOAD, "")
        if argv[1:3] == ["kernels", "status"]:
            return subprocess.CompletedProcess(
                argv, 0, 'has status "KernelWorkerStatus.COMPLETE"', ""
            )
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(kaggle_queue, "_run", fake_run)

    code = kaggle_queue.drive("session", hours=9.0, poll=0.0, timeout=1.0, max_sessions=1)

    assert [argv for argv in calls if argv[1:3] == ["kernels", "push"]]
    # The push happened, then it stopped on the missing session log.
    assert code == kaggle_queue.EXIT_SESSION_FAILED
