"""Tests for the Kaggle queue driver (``scripts/kaggle_queue.py``).

The driver decides when a multi-session queue is finished, so the two things
worth pinning are the completion signal (which must be cumulative, and must
not count an interrupted pair) and the status parser, which is fed the CLI's
own prose rather than a machine-readable field.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import kaggle_queue  # noqa: E402
import kaggle_t4  # noqa: E402


@pytest.fixture(autouse=True)
def _default_mount_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("ACT_T4_OWNER", raising=False)
    monkeypatch.delenv("ACT_T4_SLUG", raising=False)
    # The driver now takes an exclusive per-queue lock, so two workers calling
    # ``main(["--queue", "session"])`` look exactly like two drivers - which is the thing
    # the lock prevents, and also what would otherwise serialise this file or refuse its
    # second test. Each test gets its own lock directory; the lock's own behaviour is
    # pinned separately by the contention test below.
    monkeypatch.setenv("ACT_QUEUE_LOCK_DIR", str(tmp_path / "queue-locks"))


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


def _run_smoke_session(root: Path, output: str) -> dict:
    """Run the registered launch path at reduced fidelity, on CPU.

    ``--tokens-per-second`` above the fp32 floor because the measured artifact
    decides fp16, and ``_autocast_context`` refuses fp16 on CPU.
    """
    session_dir = root / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "act.training.cli",
            "--mode",
            "pair-session",
            "--profile",
            "--device",
            "cpu",
            "--seeds",
            "15000-15001",
            "--session-dir",
            str(session_dir),
            "--dataset-root",
            str(root / "datasets"),
            "--hours",
            "1",
            "--tokens-per-second",
            "1000000",
            "--output",
            str(session_dir / output),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads((session_dir / output).read_text())


def test_a_smoke_session_round_trips_through_the_progress_reader(tmp_path: Path) -> None:
    """The driver's completion signal is a contract with a real session log.

    Every unit test above feeds ``queue_progress`` a hand-built log; this one
    runs the registered launch command and feeds it the log that command
    actually writes, then re-runs it against the published version. A resumed
    session must still read as complete - if skipped seeds did not count, the
    driver would re-push a session that had nothing left to do.
    """
    fresh = _run_smoke_session(tmp_path, "session-log.json")
    assert [pair["action"] for pair in fresh["pairs"]] == ["completed", "completed"]
    assert kaggle_queue.queue_progress(fresh) == (2, len(fresh["declared_seed_queue"]))

    resumed = _run_smoke_session(tmp_path, "session-log-2.json")
    assert [pair["action"] for pair in resumed["pairs"]] == [
        "skipped_complete",
        "skipped_complete",
    ]
    assert kaggle_queue.queue_progress(resumed) == (2, len(resumed["declared_seed_queue"]))
    # The published version is what makes the resume possible, and it is the
    # same directory a Kaggle dataset version would carry.
    published = tmp_path / "datasets" / "act-checkpoints" / "v0-5-1"
    assert (published / "seed-15000" / "manifests" / "manifest.json").is_file()


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
        "    os.environ.setdefault(OWNER_ENV, 'sid123123234')\n"
        "    os.environ.setdefault(SLUG_ENV, 'act-v0-5-1')\n"
        "    os.environ.setdefault(HOURS_ENV, '6.5')\n"
        "    raise SystemExit(main())\n"
    )
    compile(source, str(folder / metadata["code_file"]), "exec")


def test_staged_kernel_pins_the_dataset_it_will_publish_to(tmp_path: Path, monkeypatch) -> None:
    """``Queue.dataset`` is resolved inside the kernel, from its own env.

    An ``ACT_T4_OWNER``/``ACT_T4_SLUG`` override set here would otherwise be
    invisible there, and the session would publish to the default dataset
    while the driver believed it was using the override.
    """
    monkeypatch.setenv("ACT_T4_OWNER", "someone-else")
    monkeypatch.setenv("ACT_T4_SLUG", "other-dataset")
    folder = kaggle_queue.stage(tmp_path / "push", "session", 9.0)
    metadata = json.loads((folder / "kernel-metadata.json").read_text())
    source = (folder / metadata["code_file"]).read_text()

    assert "os.environ.setdefault(OWNER_ENV, 'someone-else')" in source
    assert "os.environ.setdefault(SLUG_ENV, 'other-dataset')" in source
    # The staged metadata must agree with what the kernel will resolve.
    assert metadata["dataset_sources"] == ["someone-else/act-lab-code"]


def test_staged_kernel_without_a_window_keeps_the_kernel_default(tmp_path: Path) -> None:
    folder = kaggle_queue.stage(tmp_path / "push", "session")
    metadata = json.loads((folder / "kernel-metadata.json").read_text())
    assert "setdefault(HOURS_ENV" not in (folder / metadata["code_file"]).read_text()


def test_stage_fails_loudly_when_the_guard_moves(tmp_path: Path, monkeypatch) -> None:
    # If the driver's __main__ guard is refactored, staging must refuse rather
    # than push a kernel that exits on the mode check without doing any work.
    # The body comes from `kernel_body`, so that is the seam this pins - the old
    # version moved `__file__`, which no longer decides which tree is read.
    monkeypatch.setattr(
        kaggle_queue, "kernel_body", lambda repo: "print('no guard here')\n"
    )

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
        # Context, not a gate: the campaign's health report shows how much of the week is
        # already spent, because "6 h left" means something different at 1/30 and 24/30.
        "used_hours": 1.24,
        "total_hours": 30.0,
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
    monkeypatch.setattr(kaggle_queue, "ensure_code_dataset", lambda *a, **k: "")

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
    monkeypatch.setattr(kaggle_queue, "ensure_code_dataset", lambda *a, **k: "")

    code = kaggle_queue.drive("session", hours=9.0, poll=0.0, timeout=1.0, max_sessions=1)

    assert [argv for argv in calls if argv[1:3] == ["kernels", "push"]]
    # The push happened, then it stopped on the missing session log.
    assert code == kaggle_queue.EXIT_SESSION_FAILED


def test_the_quota_wait_is_measured_from_the_apis_own_refresh_time() -> None:
    """The wait is a number read off the payload, not a guess."""
    quota = {"remaining_hours": 2.0, "refresh_at": "2026-09-26T00:00:00"}
    noon_before = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    assert kaggle_queue.seconds_until_refresh(quota, now=noon_before) == pytest.approx(
        12 * 3600 + kaggle_queue.QUOTA_REFRESH_MARGIN_SECONDS
    )
    # A refresh time that has already passed still returns the margin rather than
    # zero, so a stale payload cannot turn the wait into a spin.
    day_after = datetime(2026, 9, 27, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    assert kaggle_queue.seconds_until_refresh(
        quota, now=day_after
    ) == pytest.approx(kaggle_queue.QUOTA_REFRESH_MARGIN_SECONDS)
    # And "cannot tell" is never a number to sleep on.
    assert kaggle_queue.seconds_until_refresh(None) is None
    assert kaggle_queue.seconds_until_refresh({}) is None
    assert kaggle_queue.seconds_until_refresh({"refresh_at": ""}) is None
    assert kaggle_queue.seconds_until_refresh({"refresh_at": "soon"}) is None


def test_waiting_for_the_refresh_refuses_to_sleep_blind() -> None:
    """An unreadable refresh time is a reason to stop, not to sleep.

    A blind sleep of unknown length is how an unattended campaign stalls for
    days without anyone learning that it had.
    """
    slept: list[float] = []
    lines: list[str] = []

    def collect(message: str, **_kwargs: object) -> None:
        lines.append(message)

    assert kaggle_queue.wait_for_refresh(None, log=collect, sleep=slept.append) is False
    assert slept == []
    assert any("unreadable" in line for line in lines)

    quota = {"remaining_hours": 1.0, "refresh_at": "2026-09-26T00:00:00"}
    assert kaggle_queue.wait_for_refresh(quota, log=collect, sleep=slept.append) is True
    assert len(slept) == 1 and slept[0] > 0
    assert any("refresh" in line for line in lines)


def test_wait_for_quota_sleeps_and_retries_instead_of_exiting(monkeypatch) -> None:
    """The flag turns exit 3 into "wait for the week, then try again"."""
    codes = [kaggle_queue.EXIT_QUOTA_INSUFFICIENT, kaggle_queue.EXIT_COMPLETE]
    monkeypatch.setattr(kaggle_queue, "drive", lambda *a, **k: codes.pop(0))
    waits: list[object] = []
    monkeypatch.setattr(
        kaggle_queue,
        "wait_for_refresh",
        lambda quota, **k: waits.append(quota) or True,
    )
    quota = {"remaining_hours": 1.0, "refresh_at": "2026-09-26T00:00:00"}
    monkeypatch.setattr(kaggle_queue, "active_quota", lambda *a, **k: quota)

    assert kaggle_queue.main(["--queue", "session", "--wait-for-quota"]) == 0
    assert waits == [quota]


def test_without_the_flag_a_short_week_still_exits_three(monkeypatch) -> None:
    """The default stays the previous behaviour: stop, and say why."""
    monkeypatch.setattr(
        kaggle_queue, "drive", lambda *a, **k: kaggle_queue.EXIT_QUOTA_INSUFFICIENT
    )
    monkeypatch.setattr(
        kaggle_queue,
        "wait_for_refresh",
        lambda *a, **k: pytest.fail("waited without --wait-for-quota"),
    )
    assert (
        kaggle_queue.main(["--queue", "session"])
        == kaggle_queue.EXIT_QUOTA_INSUFFICIENT
    )


def test_the_wait_is_bounded_by_max_waits(monkeypatch) -> None:
    """A queue that never fits must not sleep forever."""
    attempts = {"count": 0}

    def fake_drive(*a, **k):
        attempts["count"] += 1
        return kaggle_queue.EXIT_QUOTA_INSUFFICIENT

    waits: list[float] = []
    monkeypatch.setattr(kaggle_queue, "drive", fake_drive)
    monkeypatch.setattr(kaggle_queue, "wait_for_refresh", lambda *a, **k: waits.append(1.0) or True)

    code = kaggle_queue.main(
        ["--queue", "session", "--wait-for-quota", "--max-waits", "1"]
    )
    assert code == kaggle_queue.EXIT_QUOTA_INSUFFICIENT
    assert attempts["count"] == 2  # one attempt, one wait, one more attempt
    assert len(waits) == 1


def test_a_failed_session_reports_the_loss_instead_of_going_silent(
    monkeypatch, capsys
) -> None:
    """A killed session must say so, and say what survived.

    Measured 2026-09-20: session 1 was pushed into a 9 h window and killed with
    ERROR 8 h later. The driver printed "kernel log follows" and then nothing at
    all, because `kaggle kernels output` had returned no files - so an operator
    could not tell an empty log from a missing one, on a session that had burned
    8.05 GPU-h. The elapsed time, the quota burned and the emptiness are now all
    named, and the (un)survival of the queue is stated rather than implied.
    """
    calls: list[list[str]] = []

    def fake_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        if argv[1] == "quota":
            return subprocess.CompletedProcess(argv, 0, QUOTA_PAYLOAD, "")
        if argv[1:3] == ["kernels", "status"]:
            # Terminal before the push, so the driver adopts nothing and pushes;
            # then the session itself ends in ERROR.
            return subprocess.CompletedProcess(
                argv, 0, 'has status "KernelWorkerStatus.ERROR"', ""
            )
        if argv[1:3] == ["kernels", "output"]:
            target = Path(argv[argv.index("-p") + 1])
            target.mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(kaggle_queue, "_run", fake_run)
    monkeypatch.setattr(kaggle_queue, "ensure_code_dataset", lambda *a, **k: "")

    code = kaggle_queue.drive("session", hours=9.0, poll=0.0, timeout=1.0, max_sessions=1)

    assert code == kaggle_queue.EXIT_SESSION_FAILED
    assert [argv for argv in calls if argv[1:3] == ["kernels", "push"]]
    out = capsys.readouterr().out
    assert "session ended with ERROR after 0.00 h" in out
    assert "no log came back from any channel" in out
    assert "carries no files at all" in out
    assert "restarts from its last published seed" in out
    # And it must not claim a log it does not have: the banner that promised one
    # and then printed nothing is the defect this pins.
    assert "kernel log follows" not in out
    # The log is *requested*, not assumed absent. `kaggle kernels output` returns
    # files and has never returned a log, so a driver that ran only that command
    # reported "the platform returned no kernel log" for a log it never asked
    # for - which is what this pins. Measured 2026-09-21: three ERROR endings of
    # one queue were each explained without the session's own account.
    assert ["kaggle", "kernels", "logs", "sid123123234/act-t4-session"] in calls


def test_a_failed_session_names_what_the_platform_did_return(
    monkeypatch, capsys
) -> None:
    """A partial survival is reported as partial, not as a total loss.

    When the platform keeps some files but no log, the session got far enough to
    write its audit trail, and an operator reading the driver's output should be
    able to tell that from the session that left nothing at all.
    """

    def fake_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        if argv[1] == "quota":
            return subprocess.CompletedProcess(argv, 0, QUOTA_PAYLOAD, "")
        if argv[1:3] == ["kernels", "status"]:
            return subprocess.CompletedProcess(
                argv, 0, 'has status "KernelWorkerStatus.ERROR"', ""
            )
        if argv[1:3] == ["kernels", "output"]:
            target = Path(argv[argv.index("-p") + 1])
            target.mkdir(parents=True, exist_ok=True)
            (target / "session-log.json").write_text("{}")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(kaggle_queue, "_run", fake_run)
    monkeypatch.setattr(kaggle_queue, "ensure_code_dataset", lambda *a, **k: "")

    code = kaggle_queue.drive("session", hours=9.0, poll=0.0, timeout=1.0, max_sessions=1)

    assert code == kaggle_queue.EXIT_SESSION_FAILED
    out = capsys.readouterr().out
    assert "no log came back from the live stream" in out
    assert "what the output did return: session-log.json" in out
    assert "no output files at all" not in out


def test_a_failed_session_reports_the_log_it_captured_while_it_ran(
    monkeypatch, capsys
) -> None:
    """The live capture is preferred, and named as the channel it came from.

    Measured 2026-09-21: the platform kept no log at all for the session it
    killed (``kaggle kernels logs`` returns nothing for a failed run, and full
    logs for a completed one), so the streamed copy is the only account of that
    failure that can exist. Nothing else in the driver can produce it.
    """

    @contextlib.contextmanager
    def fake_capture(kernel: str, path: Path):
        Path(path).write_text(
            "=== ACT lab: session session ===\n"
            "restored 4 seed version(s) from act-v0-5-1: [15000, 15001]\n"
        )
        yield

    def fake_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        if argv[1] == "quota":
            return subprocess.CompletedProcess(argv, 0, QUOTA_PAYLOAD, "")
        if argv[1:3] == ["kernels", "status"]:
            return subprocess.CompletedProcess(
                argv, 0, 'has status "KernelWorkerStatus.ERROR"', ""
            )
        if argv[1:3] == ["kernels", "output"]:
            Path(argv[argv.index("-p") + 1]).mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(kaggle_queue, "capture_live_log", fake_capture)
    monkeypatch.setattr(kaggle_queue, "_run", fake_run)
    monkeypatch.setattr(kaggle_queue, "ensure_code_dataset", lambda *a, **k: "")

    code = kaggle_queue.drive("session", hours=9.0, poll=0.0, timeout=1.0, max_sessions=1)

    assert code == kaggle_queue.EXIT_SESSION_FAILED
    out = capsys.readouterr().out
    assert "--- session log (live stream captured while it ran) ---" in out
    assert "restored 4 seed version(s) from act-v0-5-1" in out


def test_the_execution_log_is_fetched_when_the_output_carries_none() -> None:
    """``kaggle kernels logs`` is the channel that has the log; use it."""
    calls: list[list[str]] = []

    def fake_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        if argv[1:3] == ["kernels", "logs"]:
            return subprocess.CompletedProcess(argv, 0, "RuntimeError: boom", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    original = kaggle_queue._run
    kaggle_queue._run = fake_run
    try:
        text, source = kaggle_queue.session_failure_log(Path("/nonexistent"), "o/k")
    finally:
        kaggle_queue._run = original

    assert calls == [["kaggle", "kernels", "logs", "o/k"]]
    assert text == "RuntimeError: boom"
    assert "kaggle kernels logs" in source


def test_the_run_clock_is_measured_from_the_first_running_poll() -> None:
    """Queue time is not training time, and must not be read as either.

    A session that sits QUEUED for an hour and then dies at startup looks
    identical to an hour of training if the clock runs from the push - the same
    class of error as reporting a 0.29 h launch failure as the platform's 8 h
    window, which is what the elapsed time was used for.
    """
    statuses = ["QUEUED", "QUEUED", "RUNNING", "ERROR"]

    def fake_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        status = statuses.pop(0) if statuses else "ERROR"
        return subprocess.CompletedProcess(
            argv, 0, f'has status "KernelWorkerStatus.{status}"', ""
        )

    original = kaggle_queue._run
    kaggle_queue._run = fake_run
    try:
        status, running_since = kaggle_queue._wait_for_terminal_detailed(
            "o/k", poll=0.0, timeout=60.0
        )
    finally:
        kaggle_queue._run = original
    assert status == "ERROR"
    assert running_since is not None

    # Never RUNNING: there is no run to measure, so the caller falls back to the
    # wall clock rather than being handed a fabricated run length.
    statuses[:] = ["QUEUED", "ERROR"]
    kaggle_queue._run = fake_run
    try:
        _, running_since = kaggle_queue._wait_for_terminal_detailed(
            "o/k", poll=0.0, timeout=60.0
        )
    finally:
        kaggle_queue._run = original
    assert running_since is None
    assert kaggle_queue.ran_note(None) == ""
    assert "after the first RUNNING poll" in kaggle_queue.ran_note(1.5)


def _fake_kaggle(calls: list[list[str]], session_logs: list[dict], publish_logs=None):
    """A ``_run`` that reports COMPLETE and writes the i-th session log.

    ``publish_logs`` optionally supplies the exit-publish audit trail the
    kernel also keeps, which the driver reports as the session's survival.
    """
    written = {"count": 0}

    def fake_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        if argv[1] == "quota":
            return subprocess.CompletedProcess(argv, 0, QUOTA_PAYLOAD, "")
        if argv[1:3] == ["kernels", "status"]:
            return subprocess.CompletedProcess(
                argv, 0, 'has status "KernelWorkerStatus.COMPLETE"', ""
            )
        if argv[1:3] == ["kernels", "output"]:
            target = Path(argv[argv.index("-p") + 1])
            target.mkdir(parents=True, exist_ok=True)
            index = min(written["count"], len(session_logs) - 1)
            (target / "session-log.json").write_text(json.dumps(session_logs[index]))
            if publish_logs is not None:
                publish = publish_logs[min(written["count"], len(publish_logs) - 1)]
                (target / kaggle_queue.PUBLISH_LOG_NAME).write_text(json.dumps(publish))
            written["count"] += 1
        return subprocess.CompletedProcess(argv, 0, "", "")

    return fake_run


def _drive_no_progress(monkeypatch, session_logs: list[dict], max_sessions: int) -> tuple[int, list[list[str]]]:
    calls: list[list[str]] = []
    monkeypatch.setattr(kaggle_queue, "_run", _fake_kaggle(calls, session_logs))
    monkeypatch.setattr(kaggle_queue, "ensure_code_dataset", lambda *a, **k: "")
    code = kaggle_queue.drive(
        "session", hours=9.0, poll=0.0, timeout=1.0, max_sessions=max_sessions
    )
    return code, [argv for argv in calls if argv[1:3] == ["kernels", "push"]]


DECLARED_QUEUE = list(range(15000, 15012))


def test_drive_stops_when_a_session_advances_nothing(monkeypatch) -> None:
    """A success that made no progress must not be repeated.

    An idled session (the clock never fit a pair) reports success and later
    sessions would do exactly the same, so re-pushing spends quota on
    nothing - the driver's own version of the burn it exists to prevent.
    """
    code, pushes = _drive_no_progress(
        monkeypatch,
        [{"declared_seed_queue": DECLARED_QUEUE, "pairs": []}],
        max_sessions=5,
    )

    assert code == kaggle_queue.EXIT_NO_PROGRESS
    assert len(pushes) == 1


def test_drive_keeps_going_while_sessions_advance(monkeypatch) -> None:
    """Positive control: progress is what continues the loop."""
    code, pushes = _drive_no_progress(
        monkeypatch,
        [
            {"declared_seed_queue": DECLARED_QUEUE, "pairs": [{"action": "completed"}]},
            {
                "declared_seed_queue": DECLARED_QUEUE,
                "pairs": [{"action": "completed"}, {"action": "completed"}],
            },
        ],
        max_sessions=2,
    )

    # Two advancing sessions run; the budget then runs out, not the guard.
    assert len(pushes) == 2
    assert code == kaggle_queue.EXIT_BUDGET_EXHAUSTED


def test_drive_re_stages_when_the_code_dataset_moves_between_sessions(
    monkeypatch,
) -> None:
    """A body staged at 09:00 and pushed at 18:00 must run the code it reports.

    The kernel body is embedded from the tree, but ``src/`` is imported from
    the mounted dataset - so if the dataset moves between staging and a push
    (another queue's drive, or a fix landed while this one waited), the kernel
    would silently run different training code than the revision its session
    records.
    """
    calls: list[list[str]] = []
    staged: list[str] = []
    # Start-up check, then the check before each of the two pushes: the second
    # push is the one that finds the dataset has moved.
    revisions = iter(["aaaa", "aaaa", "bbbb"])

    def fake_stage(directory, queue_mode, hours=None):
        staged.append(str(directory))
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "kernel-metadata.json").write_text("{}\n")
        return folder

    logs = [
        {"declared_seed_queue": DECLARED_QUEUE, "pairs": [{"action": "completed"}]},
        {
            "declared_seed_queue": DECLARED_QUEUE,
            "pairs": [{"action": "completed"}, {"action": "completed"}],
        },
    ]
    monkeypatch.setattr(kaggle_queue, "_run", _fake_kaggle(calls, logs))
    monkeypatch.setattr(kaggle_queue, "stage", fake_stage)
    monkeypatch.setattr(
        kaggle_queue, "ensure_code_dataset", lambda *a, **k: next(revisions)
    )

    kaggle_queue.drive("session", hours=9.0, poll=0.0, timeout=1.0, max_sessions=2)

    pushes = [argv for argv in calls if argv[1:3] == ["kernels", "push"]]
    assert len(pushes) == 2
    # Staged once at start, then re-staged for the push whose dataset moved.
    assert len(staged) == 2
    assert pushes[0][pushes[0].index("-p") + 1] != pushes[1][pushes[1].index("-p") + 1]


def test_publish_log_name_matches_the_remote_wrapper() -> None:
    """The driver reads the file the wrapper writes; a drift reads as silence."""
    from act.ops.kaggle_remote import PUBLISH_LOG_NAME as WRAPPER_NAME

    assert kaggle_queue.PUBLISH_LOG_NAME == WRAPPER_NAME


def test_drive_reports_the_exit_publish_outcome(monkeypatch, capsys) -> None:
    """Whether a session's pairs survived is the number that matters."""
    calls: list[list[str]] = []
    monkeypatch.setattr(
        kaggle_queue,
        "_run",
        _fake_kaggle(
            calls,
            [
                {
                    "declared_seed_queue": DECLARED_QUEUE,
                    "pairs": [{"action": "completed"}],
                },
                {
                    "declared_seed_queue": DECLARED_QUEUE,
                    "pairs": [{"action": "completed"}, {"action": "completed"}],
                },
            ],
            publish_logs=[
                {"status": "published", "seeds_published": 6},
                {"status": "published", "seeds_published": 12},
            ],
        ),
    )
    monkeypatch.setattr(kaggle_queue, "ensure_code_dataset", lambda *a, **k: "rev")

    kaggle_queue.drive("session", hours=9.0, poll=0.0, timeout=1.0, max_sessions=2)

    printed = capsys.readouterr().out
    assert "exit publish: published (6 seed version(s))" in printed
    assert "exit publish: published (12 seed version(s))" in printed


def test_drive_stops_when_the_exit_publish_failed(monkeypatch, capsys) -> None:
    """A session whose push failed leaves nothing behind to carry forward.

    The handoff roots are in the container overlay, so re-training those pairs
    costs a whole window to re-attempt the same push - stop and let the
    recorded error be read first.
    """
    calls: list[list[str]] = []
    monkeypatch.setattr(
        kaggle_queue,
        "_run",
        _fake_kaggle(
            calls,
            [
                {
                    "declared_seed_queue": DECLARED_QUEUE,
                    "pairs": [{"action": "completed"}],
                }
            ],
            publish_logs=[{"status": "failed", "seeds_published": 6, "error": "RuntimeError: 500"}],
        ),
    )
    monkeypatch.setattr(kaggle_queue, "ensure_code_dataset", lambda *a, **k: "rev")

    code = kaggle_queue.drive(
        "session", hours=9.0, poll=0.0, timeout=1.0, max_sessions=3
    )

    assert code == kaggle_queue.EXIT_SESSION_FAILED
    pushes = [argv for argv in calls if argv[1:3] == ["kernels", "push"]]
    assert len(pushes) == 1
    assert "pairs are lost with the container" in capsys.readouterr().out


def test_drive_stops_when_a_session_failed_a_pair(monkeypatch) -> None:
    """A pair that raised must not be followed by a session that repeats it.

    The session stops its own queue so its completed seeds reach the dataset,
    and its kernel still exits 0 - so the failure is visible only in the log,
    while the pairs it *did* complete look like ordinary progress. Without
    this check the loop would push again and pay for the same failure.
    """
    code, pushes = _drive_no_progress(
        monkeypatch,
        [
            {
                "declared_seed_queue": DECLARED_QUEUE,
                "pairs": [
                    {"seed": 15000, "action": "completed"},
                    {
                        "seed": 15001,
                        "action": "training_failed",
                        "error": "RuntimeError: CUDA out of memory",
                    },
                ],
            }
        ],
        max_sessions=3,
    )

    assert code == kaggle_queue.EXIT_SESSION_FAILED
    assert len(pushes) == 1


def test_report_verdict_mirrors_the_registered_code(tmp_path: Path) -> None:
    """The kernel exits 0 whatever the decision, so the artifact carries it."""
    nested = tmp_path / "act-lab" / "artifacts"
    nested.mkdir(parents=True)
    (nested / kaggle_queue.VERDICT_NAME).write_text(
        json.dumps({"verdict": "FAIL", "exit_code": 1})
    )

    assert kaggle_queue.report_verdict(tmp_path) == 1


def test_report_verdict_refuses_a_provisional_pass_on_an_incomplete_block(
    tmp_path: Path, capsys
) -> None:
    """The script's provisional verdict must not become the run's exit code.

    On an incomplete block the verdict script deliberately still reports a
    provisional PASS with Gate 4 withheld and `queue_complete: false`. The
    driver's exit code is about the run, so reading it as PASS would let a
    scripted caller treat a partial block as the registered decision.
    """
    (tmp_path / kaggle_queue.VERDICT_NAME).write_text(
        json.dumps(
            {
                "verdict": "PASS",
                "exit_code": 0,
                "queue_complete": False,
                "seeds_missing": [15011],
                "gate_4_authorized": False,
            }
        )
    )

    assert kaggle_queue.report_verdict(tmp_path) == kaggle_queue.EXIT_NO_VERDICT
    out = capsys.readouterr().out
    assert "provisional" in out and "keep driving the queues" in out


def test_report_verdict_mirrors_a_complete_blocks_code(tmp_path: Path) -> None:
    """Positive control: a complete block still returns its registered code."""
    (tmp_path / kaggle_queue.VERDICT_NAME).write_text(
        json.dumps(
            {"verdict": "PASS", "exit_code": 0, "queue_complete": True}
        )
    )

    assert kaggle_queue.report_verdict(tmp_path) == 0


def test_report_verdict_reports_no_verdict_distinctly(tmp_path: Path) -> None:
    """"No verdict produced" must not read as a cost FAIL."""
    assert kaggle_queue.report_verdict(tmp_path) == kaggle_queue.EXIT_NO_VERDICT
    assert kaggle_queue.EXIT_NO_VERDICT != 1

    (tmp_path / kaggle_queue.VERDICT_NAME).write_text(json.dumps({"verdict": "PASS"}))
    assert kaggle_queue.report_verdict(tmp_path) == kaggle_queue.EXIT_NO_VERDICT


def test_run_once_pushes_the_verdict_and_returns_its_decision(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        if argv[1:3] == ["kernels", "status"]:
            return subprocess.CompletedProcess(
                argv, 0, 'has status "KernelWorkerStatus.COMPLETE"', ""
            )
        if argv[1:3] == ["kernels", "output"]:
            target = Path(argv[argv.index("-p") + 1])
            target.mkdir(parents=True, exist_ok=True)
            (target / kaggle_queue.VERDICT_NAME).write_text(
                json.dumps({"verdict": "PASS_WITH_QUALIFICATION", "exit_code": 2})
            )
        return subprocess.CompletedProcess(argv, 0, "", "")

    ensured: list[int] = []
    monkeypatch.setattr(kaggle_queue, "_run", fake_run)
    monkeypatch.setattr(
        kaggle_queue, "ensure_code_dataset", lambda *a, **k: ensured.append(1) or ""
    )

    code = kaggle_queue.run_once("verdict", poll=0.0, timeout=1.0)

    assert code == 2
    assert len([argv for argv in calls if argv[1:3] == ["kernels", "push"]]) == 1
    # The verdict reads the runner's summaries, so it must run the same code.
    assert ensured == [1]


def test_run_once_reports_a_diagnostic_kernels_log(monkeypatch, capsys) -> None:
    """A diagnostic's result is its log - the device count and the placement.

    Reporting only "the kernel succeeded" would hide the measurement an
    operator ran the job for, and fetching it by hand afterwards is exactly
    the step this tool exists to remove.
    """

    def fake_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        if argv[1:3] == ["kernels", "status"]:
            return subprocess.CompletedProcess(
                argv, 0, 'has status "KernelWorkerStatus.COMPLETE"', ""
            )
        if argv[1:3] == ["kernels", "output"]:
            target = Path(argv[argv.index("-p") + 1])
            target.mkdir(parents=True, exist_ok=True)
            (target / "act-t4-selfcheck.log").write_text(
                "[5/5] per-arm processes ...\n    one arm per device: cuda:0, cuda:1\n"
                "v0 session self-check: all checks passed\n"
            )
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(kaggle_queue, "_run", fake_run)
    monkeypatch.setattr(kaggle_queue, "ensure_code_dataset", lambda *a, **k: "")

    assert kaggle_queue.run_once("selfcheck", poll=0.0, timeout=1.0) == 0
    printed = capsys.readouterr().out
    assert "one arm per device: cuda:0, cuda:1" in printed
    assert "all checks passed" in printed


def test_a_restore_probe_that_measured_then_died_keeps_its_number(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    """The measurement is the result, so a failure on the way out must not lose it.

    A session pays this download before it trains, and the kernel log carrying it
    is exactly what the platform discards for a killed run, so the record is
    imported before the status is judged - the same ordering ``archzoo`` uses for
    ledger rows that cost GPU-hours to regenerate.
    """

    def fake_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        if argv[1:3] == ["kernels", "status"]:
            return subprocess.CompletedProcess(
                argv, 0, 'has status "KernelWorkerStatus.ERROR"', ""
            )
        if argv[1:3] == ["kernels", "output"]:
            target = Path(argv[argv.index("-p") + 1])
            target.mkdir(parents=True, exist_ok=True)
            (target / kaggle_t4.RESTORE_PROBE_NAME).write_text(
                json.dumps(
                    {
                        "dataset": "o/ds",
                        "seeds_restored": [15000, 15001, 15002, 15003],
                        "minutes": 41.5,
                        "gib_written": 22.7,
                        "accelerator": "cpu",
                    }
                )
                + "\n"
            )
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(kaggle_queue, "_run", fake_run)
    monkeypatch.setattr(kaggle_queue, "ensure_code_dataset", lambda *a, **k: "")
    # Staging is real by default and reads the kernel body from REPO_ROOT, so it
    # is redirected here along with the import target: what is under test is the
    # ordering of the import against the status check, not the staging.
    monkeypatch.setattr(kaggle_queue, "stage", lambda d, mode, hours=None: Path(d))
    repo = tmp_path / "repo"
    monkeypatch.setattr(kaggle_queue, "REPO_ROOT", repo)

    code = kaggle_queue.run_once("restoreprobe", poll=0.0, timeout=1.0)

    assert code == kaggle_queue.EXIT_SESSION_FAILED
    transferred = repo / "artifacts" / kaggle_t4.RESTORE_PROBE_NAME
    assert transferred.is_file()
    assert "41.5 min" in capsys.readouterr().out


def test_a_missing_restore_probe_record_is_reported_not_invented(
    tmp_path: Path, capsys
) -> None:
    """A probe that died before measuring must not read as a measurement."""
    assert kaggle_queue.import_restore_probe(tmp_path, tmp_path / "repo") is None
    assert f"no {kaggle_t4.RESTORE_PROBE_NAME}" in capsys.readouterr().out


def test_the_documented_verdict_command_is_accepted(monkeypatch) -> None:
    """The runbook tells operators to run this; it must not be a usage error."""
    seen: list[str] = []

    def fake_once(mode: str, *, poll: float, timeout: float) -> int:
        seen.append(mode)
        return 0

    monkeypatch.setattr(kaggle_queue, "run_once", fake_once)

    assert kaggle_queue.main(["--queue", "verdict"]) == 0
    assert seen == ["verdict"]
    assert kaggle_queue.main(["--queue", "preflight"]) == 0
    assert seen == ["verdict", "preflight"]
    assert kaggle_queue.main(["--queue", "dtypeprobe"]) == 0
    assert seen == ["verdict", "preflight", "dtypeprobe"]


def test_every_one_shot_mode_has_metadata_to_push() -> None:
    """A mode the driver accepts but cannot stage would push nothing."""
    for mode in kaggle_queue.ONE_SHOT_MODES:
        assert kaggle_t4.metadata_for(mode)["id"], mode


def test_burned_note_reports_quota_spent_on_nothing() -> None:
    before = {"remaining_hours": 28.0, "refresh_at": "x"}
    after = {"remaining_hours": 25.5, "refresh_at": "x"}
    assert kaggle_queue.burned_note(before, after) == " (it used 2.50 GPU-h of the quota)"
    # Unreadable, or no measurable burn: say nothing rather than guess.
    assert kaggle_queue.burned_note(None, after) == ""
    assert kaggle_queue.burned_note(before, None) == ""
    assert kaggle_queue.burned_note(before, before) == ""


def test_code_files_ship_the_package_and_not_the_artifacts(monkeypatch) -> None:
    """The kernel runs this code, so what ships is load-bearing.

    ``artifacts/`` must not ship: a session has to start with no ``artifacts/``
    directory so the launch command cannot read a benchmark artifact that this
    session did not measure itself.
    """
    monkeypatch.delenv("ACT_T4_OWNER", raising=False)
    files = kaggle_queue.code_files(ROOT)

    assert "pyproject.toml" in files
    assert "src/act/training/runtime.py" in files
    assert "scripts/kaggle_t4.py" in files
    assert not [name for name in files if name.split("/")[0] == "artifacts"]


def test_code_revision_reports_a_dirty_tree_instead_of_refusing(monkeypatch) -> None:
    """A dirty tree is reported, not fatal - the payload is HEAD's either way.

    The refusal this replaced was measured to cost a campaign: the driver calls
    this before every push, so one uncommitted edit during a poll ended a
    two-week unattended run *after* a session had trained and published. What
    the refusal protected against - publishing uncommitted edits under a
    committed revision's name - is now impossible, because the archive is built
    from HEAD (`test_the_staged_archive_is_head_bytes_even_when_dirty`).
    """
    def clean(argv: list[str]) -> subprocess.CompletedProcess[str]:
        if "--porcelain" in argv:
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 0, "deadbeef\n", "")

    monkeypatch.setattr(kaggle_queue, "_run", clean)
    assert kaggle_queue.code_revision(ROOT) == "deadbeef"
    assert kaggle_queue.dirty_paths(ROOT) == []

    def dirty(argv: list[str]) -> subprocess.CompletedProcess[str]:
        if "--porcelain" in argv:
            return subprocess.CompletedProcess(argv, 0, " M src/act/x.py\n?? scratch.py\n", "")
        return subprocess.CompletedProcess(argv, 0, "deadbeef\n", "")

    monkeypatch.setattr(kaggle_queue, "_run", dirty)
    # The revision is still the commit - and the dirt is *named*, not hidden.
    assert kaggle_queue.code_revision(ROOT) == "deadbeef"
    assert kaggle_queue.dirty_paths(ROOT) == ["src/act/x.py", "scratch.py"]

    monkeypatch.setattr(
        kaggle_queue,
        "_run",
        lambda argv: subprocess.CompletedProcess(argv, 128, "", "not a git repository"),
    )
    with pytest.raises(SystemExit, match="not a git checkout"):
        kaggle_queue.code_revision(ROOT)
    # An unresolvable repository cannot report dirt either; it must not raise out
    # of the reporter, which runs on the driver's hot path.
    assert kaggle_queue.dirty_paths(ROOT) == []


def _scratch_repo(tmp_path: Path) -> Path:
    """A one-file git repo, committed, for the publish-source tests."""
    repo = tmp_path / "repo"
    (repo / "src" / "act").mkdir(parents=True)
    (repo / "src" / "act" / "__init__.py").write_text("VERSION = 'committed'\n")
    (repo / "pyproject.toml").write_text("[project]\nname = 'scratch'\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True)
    _scratch_commit(repo, "scratch")
    return repo


def _scratch_commit(repo: Path, message: str) -> None:
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=t",
            "commit",
            "-qm",
            message,
        ],
        check=True,
        capture_output=True,
    )


def test_the_staged_archive_is_head_bytes_even_when_dirty(tmp_path: Path) -> None:
    """The marker names a commit, so the payload must be that commit's bytes.

    The old path read the working tree, which is how an uncommitted edit could
    have shipped under a clean commit's name - the reason for the refusal that
    killed a live campaign. Sourcing from HEAD makes it impossible rather than
    detected, and this pins it on a real repository: the file is edited after
    the commit, and the archive still carries the committed text.
    """
    repo = _scratch_repo(tmp_path)
    (repo / "src" / "act" / "__init__.py").write_text("VERSION = 'UNCOMMITTED'\n")

    folder = kaggle_queue.stage_code_dataset(repo, tmp_path / "code", "deadbeef")

    with tarfile.open(folder / kaggle_queue.CODE_ARCHIVE_NAME) as archive:
        member = archive.extractfile("src/act/__init__.py")
        assert member is not None
        assert member.read().decode() == "VERSION = 'committed'\n"
        assert "UNCOMMITTED" not in archive.extractfile(
            "src/act/__init__.py"
        ).read().decode()
    # And the revision the marker claims is the one HEAD resolves to.
    assert kaggle_queue.code_revision(repo) == subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_the_kernel_body_is_the_committed_copy_when_it_is_edited(tmp_path: Path) -> None:
    """A locally edited kernel driver cannot ship against HEAD's package.

    A clean tree keeps a local edit testable (the two texts are the same); a
    dirty one ships the commit, so the body and the mounted ``src/`` are always
    one revision - which is the mismatch the old refusal existed to prevent.
    """
    repo = _scratch_repo(tmp_path)
    body_path = repo / kaggle_queue.KERNEL_BODY_NAME
    body_path.parent.mkdir(parents=True, exist_ok=True)
    body_path.write_text("print('committed body')\n")
    _scratch_commit(repo, "body")
    # Clean: the working copy is used, so an edit under test is the text tested.
    assert kaggle_queue.kernel_body(repo) == "print('committed body')\n"
    body_path.write_text("print('edited')\n")
    assert kaggle_queue.kernel_body(repo) == "print('committed body')\n"


def test_stage_code_dataset_archives_the_package(tmp_path: Path, monkeypatch) -> None:
    """The tree must travel as an archive, not as directories.

    Kaggle's uploader with its default ``dir_mode`` silently drops every
    subdirectory, so a folder-shaped push produced a code dataset with no
    ``src/`` - every kernel then died at bootstrap on "no code dataset under
    /kaggle/input" - while the loose ``revision.txt`` survived and made it look
    current. The archive's internal paths are what the service restores.
    """
    monkeypatch.setattr(
        kaggle_queue, "code_files", lambda repo: ["pyproject.toml", "src/act/__init__.py"]
    )
    folder = kaggle_queue.stage_code_dataset(ROOT, tmp_path / "code", "cafe1234")

    assert (folder / kaggle_queue.REVISION_MARKER).read_text().strip() == "cafe1234"
    metadata = json.loads((folder / "dataset-metadata.json").read_text())
    assert metadata["id"].endswith("/act-lab-code")
    # The package lives in the archive; nothing is pushed as a bare directory.
    assert not (folder / "src").exists()
    with tarfile.open(folder / kaggle_queue.CODE_ARCHIVE_NAME) as archive:
        names = set(archive.getnames())
    assert {"pyproject.toml", "src/act/__init__.py", kaggle_queue.REVISION_MARKER} <= names


def test_code_payload_defect_catches_the_subdirectory_drop(tmp_path: Path) -> None:
    """A marker alone is not evidence the code is there.

    This is the published state that slipped through: ``revision.txt`` and the
    top-level files present, the package tree gone.
    """
    (tmp_path / kaggle_queue.REVISION_MARKER).write_text("cafe1234\n")
    assert kaggle_queue.code_payload_defect(tmp_path) is not None

    (tmp_path / "pyproject.toml").write_text("[project]\n")
    assert kaggle_queue.code_payload_defect(tmp_path) is not None

    (tmp_path / "src" / "act").mkdir(parents=True)
    assert kaggle_queue.code_payload_defect(tmp_path) is None

    # The service wraps an uploaded archive in a directory named after it, so
    # a nested tree is fine - the kernel searches for it either way.
    wrapped = tmp_path / "wrapped" / "code" / "nested"
    (wrapped / "src" / "act").mkdir(parents=True)
    (wrapped / "pyproject.toml").write_text("[project]\n")
    assert kaggle_queue.code_payload_defect(tmp_path / "wrapped") is None


def test_ensure_code_dataset_republishes_a_marker_current_but_empty_dataset(
    tmp_path: Path, monkeypatch
) -> None:
    """Revision match plus a missing tree must still count as stale."""
    calls: list[list[str]] = []

    def fake_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        if argv[1:3] == ["datasets", "version"]:
            return subprocess.CompletedProcess(argv, 0, "version pushed", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    served = {"count": 0}

    def fake_download(dataset: str, *, work) -> Path:
        here = Path(work)
        here.mkdir(parents=True, exist_ok=True)
        (here / kaggle_queue.REVISION_MARKER).write_text("cafe1234\n")
        # Defective on the pre-publish probe (the marker is there, the tree is
        # not), healthy afterwards, so the post-publish read-back passes.
        if served["count"]:
            (here / "code").mkdir(parents=True, exist_ok=True)
            (here / "code" / "pyproject.toml").write_text("[project]\n")
            (here / "code" / "src" / "act").mkdir(parents=True, exist_ok=True)
        served["count"] += 1
        return here

    monkeypatch.setattr(kaggle_queue, "_run", fake_run)
    monkeypatch.setattr(kaggle_queue, "download_code_dataset", fake_download)
    monkeypatch.setattr(kaggle_queue, "code_revision", lambda repo: "cafe1234")
    monkeypatch.setattr(
        kaggle_queue, "stage_code_dataset", lambda repo, directory, revision: Path(directory)
    )

    revision = kaggle_queue.ensure_code_dataset(tmp_path)

    assert revision == "cafe1234"
    # The marker matched, and the dataset was still republished.
    assert [argv for argv in calls if argv[1:3] == ["datasets", "version"]]


def test_stage_code_dataset_fails_loudly_without_the_package(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(kaggle_queue, "code_files", lambda repo: ["pyproject.toml"])
    with pytest.raises(SystemExit, match="pyproject.toml"):
        kaggle_queue.stage_code_dataset(ROOT, tmp_path / "code", "cafe1234")


def test_published_code_revision_is_none_when_the_marker_is_absent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        kaggle_queue,
        "_run",
        lambda argv: subprocess.CompletedProcess(argv, 0, "", ""),
    )
    assert kaggle_queue.published_code_revision("owner/ds", work=tmp_path) is None

    (tmp_path / kaggle_queue.REVISION_MARKER).write_text("cafe1234\n")
    assert kaggle_queue.published_code_revision("owner/ds", work=tmp_path) == "cafe1234"


def test_stage_code_dataset_leaves_the_source_tree_untouched(
    tmp_path: Path, monkeypatch
) -> None:
    """Staging must not write into the repo it stages.

    The marker used to be written to the *source* tree and archived from
    there, which left ``revision.txt`` in the repo. Untracked, that makes the
    next ``code_revision`` refuse the tree as dirty - one publish blocking
    every publish after it.
    """
    monkeypatch.setattr(
        kaggle_queue,
        "code_files",
        lambda repo: ["pyproject.toml", "src/act/__init__.py"],
    )
    marker_in_repo = ROOT / kaggle_queue.REVISION_MARKER
    marker_in_repo.unlink(missing_ok=True)

    folder = kaggle_queue.stage_code_dataset(ROOT, tmp_path / "code", "cafe1234")

    assert not marker_in_repo.exists()
    assert (folder / kaggle_queue.REVISION_MARKER).read_text().strip() == "cafe1234"
    with tarfile.open(folder / kaggle_queue.CODE_ARCHIVE_NAME) as archive:
        assert kaggle_queue.REVISION_MARKER in set(archive.getnames())


def test_ensure_code_dataset_retries_a_read_back_that_lags_the_push(
    tmp_path: Path, monkeypatch
) -> None:
    """A download right after a version push can serve the previous version.

    Measured 2026-09-15: the read-back saw the old revision immediately after a
    successful publish, which is indistinguishable from a rejected push. The
    single-read check raised SystemExit there even though the push had landed.
    """
    calls: list[list[str]] = []

    def fake_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "version pushed", "")

    served = {"count": 0}

    def fake_download(dataset: str, *, work) -> Path:
        here = Path(work)
        here.mkdir(parents=True, exist_ok=True)
        (here / "src" / "act").mkdir(parents=True, exist_ok=True)
        (here / "pyproject.toml").write_text("[project]\n")
        # Read 0 is the pre-publish state; read 1 is the race, still showing the
        # old revision; read 2 finally sees the pushed one.
        stale = served["count"] < 2
        (here / kaggle_queue.REVISION_MARKER).write_text(
            "old9999\n" if stale else "cafe1234\n"
        )
        served["count"] += 1
        return here

    monkeypatch.setattr(kaggle_queue, "_run", fake_run)
    monkeypatch.setattr(kaggle_queue, "download_code_dataset", fake_download)
    monkeypatch.setattr(kaggle_queue, "code_revision", lambda repo: "cafe1234")
    monkeypatch.setattr(
        kaggle_queue, "stage_code_dataset", lambda repo, directory, revision: Path(directory)
    )
    monkeypatch.setattr(kaggle_queue, "CODE_VERIFY_POLL_SECONDS", 0.0)

    assert kaggle_queue.ensure_code_dataset(tmp_path) == "cafe1234"
    assert served["count"] >= 3
    assert [argv for argv in calls if argv[1:3] == ["datasets", "version"]]


def test_an_expired_oauth_token_is_refreshed_once_and_the_call_retried(monkeypatch) -> None:
    """The CLI refuses instead of refreshing; print-access-token refreshes."""
    calls: list[list[str]] = []

    def fake(argv, capture_output, text):
        calls.append(list(argv))
        if argv[:3] == ["kaggle", "auth", "print-access-token"]:
            return subprocess.CompletedProcess(argv, 0, "tok\n", "")
        first_time = sum(1 for c in calls if c == argv) == 1
        if first_time:
            return subprocess.CompletedProcess(argv, 1, "", kaggle_queue.AUTH_REQUIRED + "\n...")
        return subprocess.CompletedProcess(argv, 0, "ok\n", "")

    monkeypatch.setattr(subprocess, "run", fake)
    result = kaggle_queue._run(["kaggle", "kernels", "output", "k", "-p", "/x"])
    assert result.returncode == 0 and result.stdout == "ok\n"
    assert [c[:3] for c in calls] == [
        ["kaggle", "kernels", "output"],
        ["kaggle", "auth", "print-access-token"],
        ["kaggle", "kernels", "output"],
    ]
    # A non-kaggle command, or a different failure, is never retried.
    calls.clear()
    kaggle_queue._run(["git", "status"])
    assert len(calls) == 1


def test_archzoo_is_a_one_shot_and_its_output_is_imported_whole(tmp_path: Path) -> None:
    """The ledgers cost GPU-hours; the import must keep every file and read the status."""
    assert "archzoo" in kaggle_queue.ONE_SHOT_MODES
    output = tmp_path / "output" / kaggle_t4.ARCHZOO_EXPERIMENT
    output.mkdir(parents=True)
    (output / "flat12000_seed42_results.jsonl").write_text("{}\n")
    (output / "zoo_definition.json").write_text("{}")
    (output / kaggle_t4.ARCHZOO_STATUS_NAME).write_text(
        json.dumps({"complete": True, "stopped_at_deadline": False, "hours": 5.5,
                    "ledger_rows": {"42": {"12000": 1}}})
    )
    repo = tmp_path / "repo"
    lines: list[str] = []
    status = kaggle_queue.import_archzoo_output(
        tmp_path / "output", repo, log=lambda m, **_: lines.append(m)
    )
    target = repo / "artifacts" / kaggle_t4.ARCHZOO_EXPERIMENT
    assert sorted(p.name for p in target.iterdir()) == [
        kaggle_t4.ARCHZOO_STATUS_NAME,
        "flat12000_seed42_results.jsonl",
        "zoo_definition.json",
    ]
    assert status["complete"] is True
    assert any("complete=True" in line for line in lines)
    # An output with no experiment directory is reported, not crashed on.
    assert kaggle_queue.import_archzoo_output(tmp_path / "empty", repo, log=lambda m, **_: None) is None


def test_a_second_driver_is_refused_rather_than_racing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The 2026-09-21 loss: two drivers, one kernel, the second push replacing the first.

    The adopt-instead-of-push rule cannot close this - both drivers check the status, both
    find nothing live, and both push. Only mutual exclusion can, so this pins that the
    second driver exits 6 instead of pushing, and that the lock is released afterwards so
    the campaign can continue.
    """
    monkeypatch.setenv("ACT_QUEUE_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setattr(kaggle_queue, "drive", lambda *a, **k: kaggle_queue.EXIT_COMPLETE)
    kernel = str(kaggle_t4.metadata_for("session")["id"])

    with kaggle_queue.queue_lock(kernel):
        assert kaggle_queue.main(["--queue", "session"]) == kaggle_queue.EXIT_ALREADY_RUNNING

    # Released: the next driver runs rather than being locked out for good.
    assert kaggle_queue.main(["--queue", "session"]) == kaggle_queue.EXIT_COMPLETE


def test_the_lock_says_who_holds_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A refusal that names the holder is actionable; "locked" alone is not."""
    monkeypatch.setenv("ACT_QUEUE_LOCK_DIR", str(tmp_path / "locks"))
    kernel = "owner/act-t4-session"
    with kaggle_queue.queue_lock(kernel):
        path = tmp_path / "locks" / "act-t4-session.lock"
        holder = json.loads(path.read_text())
        assert holder["pid"] > 0
        assert holder["kernel"] == kernel
        assert holder["since"]
    # The record is advisory and the exclusion is the flock: a refusal reads this file only
    # while somebody holds the lock, so a leftover record cannot mislead anyone, and the
    # next acquire overwrites it. What matters - that the lock is genuinely released - is
    # pinned by the second-driver test above.


def test_a_fast_logless_failure_is_not_reported_as_the_platform_window() -> None:
    """A 0.29 h launch failure must not be explained as an 8 h platform kill.

    The old text did exactly that, printing "8.05 GPU-h burned" for a run that spent
    0.27 - to an operator deciding whether to spend more.
    """
    assert kaggle_queue.PLATFORM_KILL_HOURS == 7.5
    fast = kaggle_queue.logless_failure_note(0.29)
    assert "startup" in fast
    assert "0.27" not in fast and "0.29" in fast
    assert "8.05 GPU-h" not in fast
    assert "session window" not in fast

    long = kaggle_queue.logless_failure_note(8.0)
    assert "session window" in long
    assert "8 h 00 m" in long
