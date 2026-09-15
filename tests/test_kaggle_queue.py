"""Tests for the Kaggle queue driver (``scripts/kaggle_queue.py``).

The driver decides when a multi-session queue is finished, so the two things
worth pinning are the completion signal (which must be cumulative, and must
not count an interrupted pair) and the status parser, which is fed the CLI's
own prose rather than a machine-readable field.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
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


def _fake_kaggle(calls: list[list[str]], session_logs: list[dict]):
    """A ``_run`` that reports COMPLETE and writes the i-th session log."""
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


def test_report_verdict_mirrors_the_registered_code(tmp_path: Path) -> None:
    """The kernel exits 0 whatever the decision, so the artifact carries it."""
    nested = tmp_path / "act-lab" / "artifacts"
    nested.mkdir(parents=True)
    (nested / kaggle_queue.VERDICT_NAME).write_text(
        json.dumps({"verdict": "FAIL", "exit_code": 1})
    )

    assert kaggle_queue.report_verdict(tmp_path) == 1


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

    monkeypatch.setattr(kaggle_queue, "_run", fake_run)

    code = kaggle_queue.run_once("verdict", poll=0.0, timeout=1.0)

    assert code == 2
    assert len([argv for argv in calls if argv[1:3] == ["kernels", "push"]]) == 1


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
    assert not [name for name in files if name.split("/")[0] in {"artifacts", "public", "server"}]
    assert "vite.config.ts" not in files


def test_code_revision_refuses_an_unpublishable_tree(monkeypatch) -> None:
    """A dirty tree has no reproducible revision, and the handoff records one."""
    def clean(argv: list[str]) -> subprocess.CompletedProcess[str]:
        if "--porcelain" in argv:
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 0, "deadbeef\n", "")

    monkeypatch.setattr(kaggle_queue, "_run", clean)
    assert kaggle_queue.code_revision(ROOT) == "deadbeef"

    def dirty(argv: list[str]) -> subprocess.CompletedProcess[str]:
        if "--porcelain" in argv:
            return subprocess.CompletedProcess(argv, 0, " M src/act/x.py\n", "")
        return subprocess.CompletedProcess(argv, 0, "deadbeef\n", "")

    monkeypatch.setattr(kaggle_queue, "_run", dirty)
    with pytest.raises(SystemExit, match="uncommitted"):
        kaggle_queue.code_revision(ROOT)

    monkeypatch.setattr(
        kaggle_queue,
        "_run",
        lambda argv: subprocess.CompletedProcess(argv, 128, "", "not a git repository"),
    )
    with pytest.raises(SystemExit, match="not a git checkout"):
        kaggle_queue.code_revision(ROOT)


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
