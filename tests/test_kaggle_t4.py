"""Tests for the Kaggle T4 driver (``scripts/kaggle_t4.py``).

The driver is what actually gets pushed for the section 5.1 run, so the two
things worth pinning are (a) that the mount-layout search survives every
Kaggle attachment layout - the first run died on the assumption that it did
not - and (b) that each queue's launch command still says exactly what the
runbook says, since a drifted flag is a protocol change, not a typo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import kaggle_t4  # noqa: E402


@pytest.fixture(autouse=True)
def _default_mount_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assertions below quote the default owner/slug, so pin them."""
    monkeypatch.delenv(kaggle_t4.OWNER_ENV, raising=False)
    monkeypatch.delenv(kaggle_t4.SLUG_ENV, raising=False)


def _make_repo(root: Path, relative: str) -> Path:
    code = root / relative
    (code / "src" / "act").mkdir(parents=True)
    (code / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    return code


@pytest.mark.parametrize(
    "layout",
    [
        "act-lab-code",
        "sid123123234/act-lab-code",
        "datasets/sid123123234/act-lab-code",
    ],
)
def test_find_code_root_handles_every_attachment_layout(
    tmp_path: Path, layout: str
) -> None:
    root = tmp_path / "input"
    code = _make_repo(root, layout)
    assert kaggle_t4.find_code_root(root) == code


def test_find_code_root_ignores_a_decoy_and_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "input"
    (root / "unrelated-dataset").mkdir(parents=True)
    (root / "unrelated-dataset" / "pyproject.toml").write_text("[project]\n")
    assert kaggle_t4.find_code_root(root) is None
    assert kaggle_t4.find_code_root(tmp_path / "does-not-exist") is None


def test_bootstrap_makes_the_copied_tree_importable_in_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The returned env dict only ever configured *subprocesses*.

    The pre-flight, the dtype probe and the verdict import ``act`` in this very
    process straight after bootstrapping, so returning ``PYTHONPATH`` was never
    enough - measured 2026-09-15, the dtype probe died on
    ``ModuleNotFoundError: No module named 'act'`` eight seconds in, and the
    session and the verdict call the same bootstrap.
    """
    source = _make_repo(tmp_path / "input", "sid123123234/act-lab-code")
    # A name nothing else imports, so the check cannot be satisfied by a module
    # that was already in ``sys.modules``.
    (source / "src" / "actlab_mount_probe.py").write_text(
        "VALUE = 'from the mount'\n", encoding="utf-8"
    )
    work = tmp_path / "work"
    monkeypatch.setattr(kaggle_t4, "find_code_root", lambda *a, **k: source)
    monkeypatch.setattr(kaggle_t4, "WORK_ROOT", work)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))

    root, env = kaggle_t4._bootstrap()

    assert root == work
    assert env["PYTHONPATH"] == str(work / "src")
    assert sys.path[0] == str(work / "src")

    import actlab_mount_probe

    assert actlab_mount_probe.VALUE == "from the mount"


def test_bootstrap_fails_closed_without_a_code_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(kaggle_t4, "find_code_root", lambda *a, **k: None)
    monkeypatch.setattr(kaggle_t4, "WORK_ROOT", tmp_path / "work")
    with pytest.raises(SystemExit, match="no code dataset"):
        kaggle_t4._bootstrap()


def test_primary_launch_command_is_the_registered_one() -> None:
    queue = kaggle_t4.QUEUES["session"]
    assert kaggle_t4.session_argv(queue, 9.0) == [
        sys.executable,
        "-m",
        "act.training.cli",
        "--mode",
        "pair-session",
        "--device",
        "cuda",
        "--hours",
        "9",
        "--seeds",
        "15000-15011",
        "--probe-candidate-policy",
        "spread",
        "--dataset-root",
        "artifacts/datasets",
        "--session-dir",
        "artifacts/v0-sessions",
        "--kaggle-dataset",
        "sid123123234/act-v0-5-1",
    ]


def test_control_queue_isolates_the_decision_rule() -> None:
    argv = kaggle_t4.session_argv(kaggle_t4.QUEUES["control"])
    assert argv[-1] == "sid123123234/act-v0-5-1-control"
    # Same candidate pool as the primary, so the only difference is the rule.
    assert argv[argv.index("--probe-candidate-policy") + 1] == "spread"
    assert argv[argv.index("--probe-selection") + 1] == "random"
    assert argv[argv.index("--dataset-root") + 1] == "artifacts/datasets-control"
    assert argv[argv.index("--session-dir") + 1] == "artifacts/v0-sessions-control"
    assert "--session-arms" not in argv


def test_null_queue_is_policy_free_and_never_shares_roots() -> None:
    argv = kaggle_t4.session_argv(kaggle_t4.QUEUES["null"])
    assert argv[argv.index("--session-arms") + 1] == "null"
    assert "--probe-candidate-policy" not in argv
    assert "--probe-selection" not in argv
    assert argv[argv.index("--dataset-root") + 1] == "artifacts/datasets-null"
    assert argv[-1] == "sid123123234/act-v0-5-1-null"


def test_handoff_roots_are_repo_relative_outside_a_kernel() -> None:
    # The runbook prints repo-relative roots, and a local proof run uses them.
    queue = kaggle_t4.QUEUES["control"]
    assert queue.dataset_root == "artifacts/datasets-control"
    assert queue.session_dir == "artifacts/v0-sessions-control"


def test_reclaim_deletes_the_root_after_a_publish_that_did_not_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "act-lab-session-deadbeef"
    (root / "artifacts" / "datasets").mkdir(parents=True)
    (root / "artifacts" / "datasets" / "seed-15000.tar").write_bytes(b"x" * 32)
    monkeypatch.setattr(kaggle_t4, "workspace_root", lambda mode="": root)

    kaggle_t4._reclaim()

    assert not root.exists()


def test_reclaim_keeps_the_root_when_the_publish_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A failed publish means the overlay copy is the only copy of this
    # session's completions; it must survive for inspection.
    root = tmp_path / "act-lab-session-deadbeef"
    root.mkdir(parents=True)
    log = root / "remote-publish-log.json"
    log.write_text(json.dumps({"status": "failed"}))
    monkeypatch.setattr(kaggle_t4, "workspace_root", lambda mode="": root)

    kaggle_t4._reclaim(log)

    assert root.exists()


def test_reclaim_never_touches_the_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    # Outside a kernel the workspace root is the repo itself; a stray rmtree
    # here would be catastrophic, so it must be a no-op.
    monkeypatch.setattr(kaggle_t4, "workspace_root", lambda mode="": Path("."))

    kaggle_t4._reclaim()

    assert Path(".").exists()
    assert Path("scripts/kaggle_t4.py").is_file()


def test_run_token_is_stable_per_container_and_empty_outside_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(kaggle_t4.CONTAINER_NAME_ENV, raising=False)
    assert kaggle_t4.run_token() == ""

    monkeypatch.setenv(
        kaggle_t4.CONTAINER_NAME_ENV, "kaggle_abc123-350086509-webtier"
    )
    token = kaggle_t4.run_token()
    assert token == kaggle_t4.run_token()  # deterministic within a run
    monkeypatch.setenv(
        kaggle_t4.CONTAINER_NAME_ENV, "kaggle_zzz999-350086509-webtier"
    )
    # Two kernels of the same queue must not share an overlay directory.
    assert kaggle_t4.run_token() != token


def test_handoff_roots_move_off_the_working_mount_in_a_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # /kaggle/working is a 20 GB loopback mount; a completed seed publishes
    # ~6.4 GB of checkpoint, so 12 seeds cannot possibly fit there.
    monkeypatch.setattr(
        kaggle_t4, "workspace_root", lambda mode="": Path("/tmp") / f"act-lab-{mode}"
    )
    queue = kaggle_t4.QUEUES["session"]
    assert queue.dataset_root == "/tmp/act-lab-session/artifacts/datasets"
    assert queue.session_dir == "/tmp/act-lab-session/artifacts/v0-sessions"


def test_concurrent_modes_get_separate_overlay_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The overlay is host-shared, so two kernels that overlap in time must not
    # land in the same directory.
    monkeypatch.setattr(
        kaggle_t4, "workspace_root", lambda mode="": Path("/tmp") / f"act-lab-{mode}"
    )
    roots = {mode: q.dataset_root for mode, q in kaggle_t4.QUEUES.items()}
    assert len(set(roots.values())) == len(roots)
    assert all(root.startswith("/tmp/act-lab-") for root in roots.values())


def test_session_argv_carries_the_absolute_handoff_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        kaggle_t4, "workspace_root", lambda mode="": Path("/tmp") / f"act-lab-{mode}"
    )
    argv = kaggle_t4.session_argv(kaggle_t4.QUEUES["null"])
    expected = "/tmp/act-lab-null/artifacts"
    assert argv[argv.index("--dataset-root") + 1] == f"{expected}/datasets-null"
    assert argv[argv.index("--session-dir") + 1] == f"{expected}/v0-sessions-null"


def test_queues_own_disjoint_datasets_and_roots() -> None:
    queues = list(kaggle_t4.QUEUES.values())
    assert len({q.dataset for q in queues}) == len(queues)
    assert len({q.dataset_root for q in queues}) == len(queues)
    assert len({q.session_dir for q in queues}) == len(queues)


def test_kernel_metadata_pins_the_t4_and_keeps_internet_on() -> None:
    metadata = kaggle_t4.metadata_for("session")
    # ``gpu_type`` is silently ignored by the CLI; only machine_shape counts.
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert "gpu_type" not in metadata
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is True
    assert metadata["kernel_type"] == "script"
    assert metadata["is_private"] is True
    assert metadata["dataset_sources"] == ["sid123123234/act-lab-code"]


def test_every_queue_mode_has_push_metadata() -> None:
    for mode in ("session", "control", "null"):
        metadata = kaggle_t4.metadata_for(mode)
        assert metadata["id"] == kaggle_t4.QUEUES[mode].kernel_id
        assert metadata["enable_gpu"] is True


def test_verdict_metadata_asks_for_no_accelerator() -> None:
    # The verdict only downloads datasets and does arithmetic, so it must not
    # queue behind a GPU or spend accelerator quota.
    metadata = kaggle_t4.metadata_for("verdict")
    assert metadata["id"] == "sid123123234/act-t4-verdict"
    assert metadata["enable_gpu"] is False
    assert metadata["machine_shape"] == ""
    assert metadata["enable_internet"] is True


def test_unknown_metadata_mode_fails_closed() -> None:
    with pytest.raises(KeyError):
        kaggle_t4.metadata_for("not-a-mode")


def test_selfcheck_metadata_wants_a_gpu_and_is_pushed() -> None:
    """The placement check is about devices, so it needs the accelerator."""
    metadata = kaggle_t4.metadata_for("selfcheck")
    assert metadata["id"] == "sid123123234/act-t4-selfcheck"
    assert metadata["enable_gpu"] is True
    assert "selfcheck" in kaggle_t4.PUSH_MODES


def test_selfcheck_mode_runs_the_battery_on_cuda(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """It must run the shipped battery, not a copy of it, on the GPU device.

    A battery that drifted from the one in the tree would pass while checking
    nothing - and the per-arm placement check only means something on a host
    that really has two T4s, which is what this mode exists to reach.
    """
    calls: list[list[str]] = []
    monkeypatch.setattr(kaggle_t4, "_bootstrap", lambda: (tmp_path, {}))
    monkeypatch.setattr(kaggle_t4, "_run", lambda argv, env, **_: calls.append(argv) or 0)
    monkeypatch.setenv(kaggle_t4.MODE_ENV, "selfcheck")
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Tesla T4")

    assert kaggle_t4.main() == 0
    assert len(calls) == 1
    battery = calls[0]
    assert battery[1] == "scripts/v0_session_selfcheck.py"
    assert (ROOT / battery[1]).is_file()
    assert battery[-2:] == ["--device", "cuda"]


def test_a_failed_selfcheck_fails_the_kernel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A battery that fails must not be reported as a passing check."""

    def failing_run(argv: list[str], env: dict[str, str], **_: object) -> int:
        raise SystemExit("command failed with exit 1: ['selfcheck']")

    monkeypatch.setattr(kaggle_t4, "_bootstrap", lambda: (tmp_path, {}))
    monkeypatch.setattr(kaggle_t4, "_run", failing_run)
    monkeypatch.setenv(kaggle_t4.MODE_ENV, "selfcheck")
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Tesla T4")

    with pytest.raises(SystemExit):
        kaggle_t4.main()


def test_verdict_command_covers_all_three_queues() -> None:
    assert kaggle_t4.verdict_argv(
        "python",
        primary="artifacts/datasets",
        control="artifacts/datasets-control",
        null="artifacts/datasets-null",
    ) == [
        "python",
        "scripts/v0_verdict.py",
        "--dataset-root",
        "artifacts/datasets",
        "--control-root",
        "artifacts/datasets-control",
        "--null-root",
        "artifacts/datasets-null",
        "--probe-batch-count",
        "1",
    ]


def test_verdict_exit_codes_are_outcomes_not_failures() -> None:
    # 1 (cost FAIL) and 2 (a withholding clause fired) are pre-registered
    # verdicts, so the driver must not treat them as command failures.
    assert set(kaggle_t4.VERDICT_EXIT_CODES) == {0, 1, 2}
    assert kaggle_t4.VERDICT_EXIT_CODES[1] == "FAIL"
    # 3 (no verdict produced) is NOT an outcome: the driver must fail loudly,
    # because a malformed invocation recorded as a verdict is exactly the
    # failure mode the distinct code exists to prevent.
    assert 3 not in kaggle_t4.VERDICT_EXIT_CODES


def test_benchmark_argv_is_the_registered_step_count() -> None:
    assert kaggle_t4.benchmark_argv("python") == [
        "python",
        "scripts/v0_throughput_benchmark.py",
        "--steps",
        "50",
    ]


def test_unknown_mode_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(kaggle_t4.MODE_ENV, "not-a-mode")
    with pytest.raises(SystemExit):
        kaggle_t4.main()
