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


def test_dtypeprobe_prices_the_placement_the_runner_will_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe's plan must be the benchmark's, priced at this host.

    The benchmark prices the launch input with ``arm_slots(...)``, but the probe
    used the dataclass default (one device), so on Kaggle's two-T4 kernels the
    plan a human reads to decide - pairs per session, sessions for the primary
    queue - was twice the run the launch would execute. Measured on the same
    host 2026-09-20: the self-check reports 2 devices, one arm per device.
    """
    import torch

    from act.training.session_plan import PairSchedule

    rate = 4_926.4
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    placed = kaggle_t4.probe_pair_schedule(rate, 2, torch.device("cuda"))
    assert placed == PairSchedule(tokens_per_second=rate, arms=2, arm_slots=2)
    assert (placed.full_pairs_per_session, placed.sessions_for_primary) == (6, 2)
    # And the line the probe logs is the one priced here, not a second spelling -
    # carrying both the registered section 5.2 arithmetic and the capacity the
    # runner's own gate honours (one pair fewer per session here, so one session
    # more for the primary queue).
    assert (
        "pair 1.486 h, 5 pairs/session by rule 1 (6 registered), "
        "3 sessions for the primary queue (2 registered); "
        "three queues 9 sessions / 53.5 h training floor"
    ) in kaggle_t4.dtypeprobe_report({"fp16": rate}, 2, torch.device("cuda"))

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    sequential = kaggle_t4.probe_pair_schedule(rate, 2, torch.device("cuda"))
    assert (sequential.full_pairs_per_session, sequential.sessions_for_primary) == (
        3,
        4,
    )
    assert (
        "2 pairs/session by rule 1 (3 registered), "
        "6 sessions for the primary queue (4 registered); "
        # The null queue is one arm and therefore one wave: on a one-device host
        # it keeps the device-per-arm capacity while the pair queues halve, so
        # the campaign total is not 3x the primary's session count.
        "three queues 15 sessions / 86.7 h training floor"
    ) in kaggle_t4.dtypeprobe_report({"fp16": rate}, 2, torch.device("cuda"))
    # The same rate and the same overhead, in one wave instead of two - which is
    # the whole difference the fix turns on: the two-wave reading bills the
    # arm-time twice, so it halves the pairs a session can fit.
    assert placed.pair_time_hours == pytest.approx(
        (sequential.pair_time_hours + 0.1) / 2
    )


def test_seed_summaries_are_kept_under_unique_names(tmp_path: Path) -> None:
    """Publish insurance: the verdict's per-seed inputs survive a failed dataset push.

    Every published seed writes ``seed-summary.json``, so keeping them under their own
    names would keep exactly one - the helper must name them by seed, and this test
    fails if that regresses.
    """
    root = tmp_path / "datasets"
    for seed in (15000, 15001):
        directory = root / "act-checkpoints" / "v0-5-1" / f"seed-{seed}"
        directory.mkdir(parents=True)
        payload = json.dumps({"seed": seed, "arms": {"adaptive": {"final_loss": 1.0}}})
        (directory / "seed-summary.json").write_text(payload, encoding="utf-8")
        # The publisher writes the same payload under reports/ as well - the layout
        # the real dataset has, and the reason the target name must come from the
        # seed directory rather than from whatever the walk happens to find.
        (directory / "reports").mkdir()
        (directory / "reports" / "seed-summary.json").write_text(payload, encoding="utf-8")
    output = tmp_path / "out"
    output.mkdir()

    assert kaggle_t4._keep_seed_summaries(root, output) == 2
    kept = sorted(p.name for p in output.iterdir())
    assert kept == ["seed-15000-summary.json", "seed-15001-summary.json"]
    payload = json.loads((output / "seed-15001-summary.json").read_text())
    assert payload["seed"] == 15001


def test_keeping_seed_summaries_is_a_no_op_before_any_pair_completes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets"
    root.mkdir()
    output = tmp_path / "out"
    output.mkdir()
    assert kaggle_t4._keep_seed_summaries(root, output) == 0
    assert list(output.iterdir()) == []


def test_the_session_mode_calls_the_summary_insurance() -> None:
    """The helper is useless unwired: pin the call site, not just the function.

    ``mode_session`` writes to ``/kaggle/working`` through ``_keep``, so this asserts at
    the bytecode level (the names the function references) rather than running it - a
    source-text match would pass on a commented-out call, and this fails when the call
    is dropped, which is the regression that matters on the exit path.
    """
    assert "_keep_seed_summaries" in kaggle_t4.mode_session.__code__.co_names


def test_unknown_mode_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(kaggle_t4.MODE_ENV, "not-a-mode")
    with pytest.raises(SystemExit):
        kaggle_t4.main()


# --- the architecture-search horizon arm -----------------------------------


def test_archzoo_argv_restates_the_crossed_design_and_pins_the_arm() -> None:
    """The design half must be the crossed arm's; the arm half its NOTES.md.

    A drifted design flag would train a different candidate set under the
    crossed arm's name, and a drifted arm flag would change what the arm
    measures - both are protocol changes rather than typos, so every flag the
    ledgers record is asserted rather than the list being compared loosely.
    """
    argv = kaggle_t4.archzoo_argv(
        "python", 43, data_dir="/kaggle/input/tokens", artifacts_root="/work/artifacts"
    )
    flags = dict(zip(argv[2::2], argv[3::2]))
    assert argv[:2] == ["python", "scripts/arch_zoo_screen.py"]
    # The crossed arm's design, flag for flag.
    assert flags["--design"] == "crossed"
    assert flags["--cross-factors"] == "depth,hidden_size"
    assert flags["--cross-replicates"] == "2"
    assert flags["--match-target"] == "78000000"
    assert flags["--match-tolerance"] == "0.05"
    assert flags["--zoo-seed"] == "42"
    assert flags["--context-size"] == "1024"
    assert flags["--batch-size"] == "2"
    # The arm as pre-registered: flat at the anchor and the 10x length, fp16,
    # exactly one seed per process, on the attached stream.
    assert flags["--experiment-id"] == "arch-zoo-100m-v3-crossed-horizon10x"
    assert flags["--plan"] == "flat"
    assert flags["--flat-steps"] == "1200,12000"
    assert flags["--finalist-seeds"] == "43"
    assert flags["--mixed-precision"] == "fp16"
    assert flags["--device"] == "cuda"
    assert flags["--data-path"] == "/kaggle/input/tokens/tokens.npy"
    assert flags["--val-data-path"] == "/kaggle/input/tokens/val-tokens.npy"
    assert flags["--artifacts-root"] == "/work/artifacts"
    # The token guard must not bind - the wall clock is the guard - so the
    # budget at the assumed rate has to exceed the plan's own tokens.
    tokens = 8 * (1200 + 12000) * 2 * 1024
    assert float(flags["--budget-hours"]) * 3600 * float(flags["--assumed-tokens-per-s"]) > tokens
    assert kaggle_t4.ARCHZOO_DEADLINE_HOURS < 12.0


def test_archzoo_seeds_are_the_crossed_arms_own_first_two() -> None:
    """Every 12k-step row needs a same-seed 1200-step fp32 row in the record."""
    assert kaggle_t4.ARCHZOO_SEEDS == (42, 43)


def test_archzoo_metadata_attaches_the_token_stream_and_wants_the_t4() -> None:
    metadata = kaggle_t4.metadata_for("archzoo")
    assert metadata["id"] == "sid123123234/act-t4-archzoo"
    assert metadata["enable_gpu"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == [
        "sid123123234/act-lab-code",
        "sid123123234/act-lab-tinystories-tokens",
    ]
    assert "archzoo" in kaggle_t4.PUSH_MODES
    # The other modes must not have grown a second dataset by accident.
    assert kaggle_t4.metadata_for("session")["dataset_sources"] == ["sid123123234/act-lab-code"]


@pytest.mark.parametrize(
    "layout",
    ["act-lab-tinystories-tokens", "datasets/sid123123234/act-lab-tinystories-tokens"],
)
def test_find_tokens_root_needs_both_files_at_any_depth(tmp_path: Path, layout: str) -> None:
    mount = tmp_path / layout
    mount.mkdir(parents=True)
    (mount / "tokens.npy").write_bytes(b"x")
    # A stream without its held-out file is a different measurement: refused.
    assert kaggle_t4.find_tokens_root(tmp_path) is None
    (mount / "val-tokens.npy").write_bytes(b"x")
    assert kaggle_t4.find_tokens_root(tmp_path) == mount
    assert kaggle_t4.find_tokens_root(tmp_path / "absent") is None


def test_ledger_rows_are_counted_per_length_for_one_seed(tmp_path: Path) -> None:
    (tmp_path / "flat1200_seed42_results.jsonl").write_text("{}\n{}\n\n{}\n")
    (tmp_path / "flat12000_seed42_results.jsonl").write_text("{}\n")
    (tmp_path / "flat12000_seed43_results.jsonl").write_text("{}\n{}\n")
    assert kaggle_t4._ledger_rows(tmp_path, 42) == {"1200": 3, "12000": 1}
    assert kaggle_t4._ledger_rows(tmp_path, 43) == {"12000": 2}
    assert kaggle_t4._ledger_rows(tmp_path, 44) == {}


def _drive_archzoo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    written: str,
    exit_43: int,
) -> tuple[dict, list[tuple[str, str]]]:
    """Run ``mode_archzoo`` over a fake sweep that writes ``written``'s ledgers.

    Returns the kernel's own status record and the (seed, device) placements, so a
    test can state what the arm owed and what it produced independently.
    """
    import subprocess
    import types

    code = _make_repo(tmp_path / "input", "datasets/sid123123234/act-lab-code")
    (code / "scripts").mkdir()
    tokens = tmp_path / "input" / "datasets" / "sid123123234" / "act-lab-tinystories-tokens"
    tokens.mkdir(parents=True)
    (tokens / "tokens.npy").write_bytes(b"x")
    (tokens / "val-tokens.npy").write_bytes(b"x")
    work = tmp_path / "work"
    output = tmp_path / "output"
    output.mkdir()

    # Both mount searches bind INPUT_ROOT as a default at definition time, so
    # they are pointed at the fixture directly, as the bootstrap test does.
    monkeypatch.setattr(kaggle_t4, "find_code_root", lambda *a, **k: code)
    monkeypatch.setattr(kaggle_t4, "find_tokens_root", lambda *a, **k: tokens)
    monkeypatch.setattr(kaggle_t4, "WORK_ROOT", work)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(kaggle_t4, "ARCHZOO_POLL_SECONDS", 0.01)
    monkeypatch.setattr(kaggle_t4, "report_resources", lambda: None)
    monkeypatch.setattr(
        kaggle_t4,
        "_keep_tree",
        lambda path: __import__("shutil").copytree(path, output / path.name, dirs_exist_ok=True),
    )
    fake_torch = types.SimpleNamespace(
        __version__="test",
        cuda=types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 2),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    runtime = types.ModuleType("act.training.runtime")
    runtime.gpu_inventory = lambda: ["T4", "T4"]
    monkeypatch.setitem(sys.modules, "act.training.runtime", runtime)

    launched: list[tuple[str, str]] = []

    class FakeProc:
        def __init__(self, argv, env, stdout, stderr):
            seed = argv[argv.index("--finalist-seeds") + 1]
            launched.append((seed, env["CUDA_VISIBLE_DEVICES"]))
            exp = Path(argv[argv.index("--artifacts-root") + 1]) / kaggle_t4.ARCHZOO_EXPERIMENT
            exp.mkdir(parents=True, exist_ok=True)
            # The design the sweeps write, so the candidate count is readable and
            # the completeness check compares against it rather than "some rows".
            (exp / "zoo_definition.json").write_text(
                json.dumps({"candidates": [{"name": f"c{i}"} for i in range(8)]})
            )
            for length in written.split(","):
                (exp / f"flat{length}_seed{seed}_results.jsonl").write_text("{}\n" * 8)
            self.returncode = 0 if seed == "42" else exit_43

        def poll(self):
            return self.returncode

    monkeypatch.setattr(subprocess, "Popen", FakeProc)

    kaggle_t4.mode_archzoo()

    status = json.loads(
        (output / kaggle_t4.ARCHZOO_EXPERIMENT / kaggle_t4.ARCHZOO_STATUS_NAME).read_text()
    )
    return status, launched


def test_archzoo_mode_places_one_seed_per_device_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The kernel must exit 0 whatever the sweeps did, with the outcome as data."""
    status, launched = _drive_archzoo(
        monkeypatch, tmp_path, written="1200,12000", exit_43=3
    )
    assert sorted(launched) == [("42", "0"), ("43", "1")]
    assert status["exit_codes"] == {"42": 0, "43": 3}
    assert status["complete"] is False  # a failed seed is not a complete arm
    assert status["stopped_at_deadline"] is False
    assert status["ledger_rows"]["42"] == {"1200": 8, "12000": 8}
    assert status["mixed_precision"] == "fp16"


def test_an_arm_that_trained_none_of_its_horizon_is_not_complete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The production failure, pinned: clean exits over an empty design.

    Every 12,000-step run raised at data load, the harness caught each one and kept
    going, and both sweeps exited 0 - so `complete` was True on an arm that had
    trained none of the horizon it was launched for, and the driver, reading only
    that flag, reported success. The kernel's status artifact is the only thing that
    can say otherwise before the analysis runs, so completeness is read off the
    requested (length, seed) grid.
    """
    status, _ = _drive_archzoo(monkeypatch, tmp_path, written="1200", exit_43=0)
    assert status["exit_codes"] == {"42": 0, "43": 0}
    assert status["complete"] is False
    assert status["missing"] == [f"seed {seed} at 12000 steps" for seed in (42, 43)]
    assert status["expected_runs_per_length"] == 8


def test_a_served_grid_is_complete(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    status, _ = _drive_archzoo(monkeypatch, tmp_path, written="1200,12000", exit_43=0)
    assert status["complete"] is True
    assert status["missing"] == []


def test_a_partly_served_length_counts_as_missing(tmp_path: Path) -> None:
    """A length is served when *every* candidate has a row, not when one does."""
    rows = {"42": {"1200": 8, "12000": 3}, "43": {"1200": 8, "12000": 8}}
    assert kaggle_t4._missing_archzoo_runs(
        rows, seeds=(42, 43), requested=[1200, 12000], expected=8
    ) == ["seed 42 at 12000 steps"]
    # An unreadable design falls back to "at least one row", never to "served".
    assert kaggle_t4._missing_archzoo_runs(
        {}, seeds=(42,), requested=[1200], expected=0
    ) == ["seed 42 at 1200 steps"]
    assert kaggle_t4._expected_candidates(tmp_path / "no-arm") == 0
