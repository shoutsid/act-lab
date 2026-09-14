"""Tests for the section 5.2 per-seed pair runner.

Tiny non-declared configs keep the whole file fast while exercising the real
loop: deadline gating, cooperative interrupt with an in-progress snapshot,
restart-from-scratch semantics, skip-completed idempotence, and the chained
session story (session 1 interrupted -> session 2 finishes and resumes).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from act.training import V0SessionConfig
from act.training.v0_handoff import (
    load_seed_summary,
    published_seed_status,
    seed_directory,
)
from act.training.v0_runner import DECLARED_SEEDS, run_pair_session
from act.training.v0_session import V0SessionInterrupted


def _config(**overrides) -> V0SessionConfig:
    values = dict(
        declared=False,
        device="cpu",
        batch_size=2,
        vocab_size=64,
        context_size=256,
        hidden_size=64,
        depth=2,
        attention_heads=4,
        expert_count=4,
        expert_width=32,
        total_steps=12,
        mutation_interval=6,
        mutation_budget=2,
        probe_candidate_cap=4,
        probe_batch_count=1,
        validation_batches=2,
        checkpoint_interval=4,
        max_checkpoints=2,
    )
    values.update(overrides)
    return V0SessionConfig(**values)


def test_declared_seed_block_matches_the_adopted_revision() -> None:
    # Re-reserved from 9000-9011 by the adopted section 5.5 change 4: the
    # original block was spent by the section 5.3 diagnostic, so the harm
    # guard was no longer blind. See RESEARCH_REPORT.md section 5.5 point 3.
    assert DECLARED_SEEDS == tuple(range(15000, 15012))
    assert 9000 not in DECLARED_SEEDS


def test_single_arm_null_session_publishes_one_arm(tmp_path: Path) -> None:
    """The do-nothing control publishes a single-arm seed summary.

    The null arm books zero search FLOPs (no probes, no rebuilds) and the
    pair summary carries its loss under ``null_final_loss`` the way gated
    sessions publish ``gated_final_loss``.
    """
    summary = run_pair_session(
        config=_config(arms=("null",)),
        session_dir=tmp_path / "sessions",
        dataset_root=tmp_path / "dataset",
        code_revision="rev",
        remaining_hours=8.0,
        measured_tokens_per_second=1e9,  # instantaneous tiny pair
        seeds=(9000,),
        output=tmp_path / "summary.json",
    )
    assert [entry["action"] for entry in summary["pairs"]] == ["completed"]
    entry = summary["pairs"][0]
    assert entry["null_final_loss"] is not None
    assert entry["random_final_loss"] is None
    assert entry["search_flops_by_arm"] == {"null": 0}
    seed_summary = load_seed_summary(tmp_path / "dataset", 9000)
    assert seed_summary["status"] == "complete"
    arm = seed_summary["arms"]["null"]
    assert arm["mutation_count"] == 0
    assert arm["search_flops"] == 0


def test_rejects_invalid_clock_and_throughput(tmp_path: Path) -> None:
    config = _config()
    with pytest.raises(ValueError, match="measured_tokens_per_second"):
        run_pair_session(
            config=config, session_dir=tmp_path / "s", dataset_root=tmp_path / "d",
            code_revision="rev", remaining_hours=8.0, measured_tokens_per_second=0.0,
        )
    with pytest.raises(ValueError, match="remaining_hours"):
        run_pair_session(
            config=config, session_dir=tmp_path / "s", dataset_root=tmp_path / "d",
            code_revision="rev", remaining_hours=0.0, measured_tokens_per_second=1e5,
        )


def test_publish_only_clock_runs_no_pairs(tmp_path: Path) -> None:
    summary = run_pair_session(
        config=_config(),
        session_dir=tmp_path / "sessions",
        dataset_root=tmp_path / "dataset",
        code_revision="rev",
        remaining_hours=0.2,  # under the 15-minute publish-only floor
        measured_tokens_per_second=50_000.0,
        output=tmp_path / "summary.json",
    )
    assert summary["pairs"] == []
    assert summary["pairs_completed"] == 0
    assert (tmp_path / "summary.json").is_file()


def test_boundary_stop_clock_starts_no_pairs(tmp_path: Path) -> None:
    # A 0.5 h clock cannot hold pair time + the 45-min publish reserve.
    summary = run_pair_session(
        config=_config(),
        session_dir=tmp_path / "sessions",
        dataset_root=tmp_path / "dataset",
        code_revision="rev",
        remaining_hours=0.5,
        measured_tokens_per_second=2_000.0,  # pair time ~1.39 h > 0.5 h
        output=tmp_path / "summary.json",
    )
    assert summary["pairs"] == []
    assert summary["pair_time_hours"] + 0.75 > 0.5  # why rule 1 refused


def test_first_session_interrupts_and_snapshots(tmp_path: Path) -> None:
    """A clock that dies mid-pair publishes in-progress and completes nothing."""
    consumed = {"seconds": 0.0}

    def clock() -> float:
        # Fake wall clock in SECONDS (the monotonic contract): 180 s = 0.05 h
        # consumed per call. Entry at 1.0 h passes rule 1 for the tiny pair
        # (0.10 h + 0.75 h reserve fits 0.95 h), then the burn-down crosses
        # the 0.75 h intra-pair floor first at the step-10 checkpoint
        # (checks at 0.90/0.85/0.80 h pass, 0.70 h < 0.75 h stops).
        consumed["seconds"] += 180.0
        return consumed["seconds"]

    summary = run_pair_session(
        config=_config(total_steps=24, checkpoint_interval=2),
        session_dir=tmp_path / "sessions",
        dataset_root=tmp_path / "dataset",
        code_revision="rev",
        remaining_hours=1.0,
        measured_tokens_per_second=1e9,  # instantaneous tiny pair
        clock=clock,
        seeds=(9000,),
        output=tmp_path / "summary.json",
    )
    actions = [entry["action"] for entry in summary["pairs"]]
    assert actions == ["interrupted"]
    interrupted = summary["pairs"][0]
    assert 0 < interrupted["interrupted_step"] < 24
    assert interrupted["interrupted_step"] == 10
    status = published_seed_status(tmp_path / "dataset", 9000)
    assert status == "in-progress"
    summary_data = load_seed_summary(tmp_path / "dataset", 9000)
    assert summary_data["status"] == "in-progress"
    assert summary_data["interrupted_step"] == interrupted["interrupted_step"]


def test_second_session_completes_and_next_seed_starts_fresh(tmp_path: Path) -> None:
    """Session chain: interrupted seed is restarted (not resumed) and finishes."""
    dataset_root = tmp_path / "dataset"
    session_dir = tmp_path / "sessions"
    config = _config(total_steps=24, checkpoint_interval=2)

    # Session 1: die mid-pair with an in-progress snapshot. Same fake-clock
    # shape as the single-session interrupt test (seconds, 180 s per call).
    consumed = {"seconds": 0.0}

    def clock() -> float:
        consumed["seconds"] += 180.0
        return consumed["seconds"]

    run_pair_session(
        config=config,
        session_dir=session_dir,
        dataset_root=dataset_root,
        code_revision="rev",
        remaining_hours=1.0,
        measured_tokens_per_second=1e9,
        clock=clock,
        seeds=(9000,),
    )
    assert published_seed_status(dataset_root, 9000) == "in-progress"

    # Session 2: plenty of clock, two queued seeds. The stale snapshot must
    # be replaced by a complete version from an uninterrupted pair, and the
    # runner must move on to the next queued seed.
    summary = run_pair_session(
        config=config,
        session_dir=session_dir,
        dataset_root=dataset_root,
        code_revision="rev-2",
        remaining_hours=8.0,
        measured_tokens_per_second=50_000.0,
        seeds=(9000, 9001),
        output=tmp_path / "summary2.json",
    )
    actions = [entry["action"] for entry in summary["pairs"]]
    assert actions == ["completed", "completed"]
    assert published_seed_status(dataset_root, 9000) == "complete"
    assert published_seed_status(dataset_root, 9001) == "complete"
    assert summary["latest_published_seed"] == 9001


def test_completed_seeds_are_skipped_idempotently(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    session_dir = tmp_path / "sessions"
    config = _config()
    first = run_pair_session(
        config=config,
        session_dir=session_dir,
        dataset_root=dataset_root,
        code_revision="rev",
        remaining_hours=8.0,
        measured_tokens_per_second=50_000.0,
        seeds=(9000,),
    )
    assert first["pairs"][0]["action"] == "completed"

    # Re-running reports the completed seed skipped and consumes no clock:
    # the same command is literally the "schedule further sessions" rule.
    second = run_pair_session(
        config=config,
        session_dir=session_dir,
        dataset_root=dataset_root,
        code_revision="rev",
        remaining_hours=0.55,  # too small for a new pair, fine for skips
        measured_tokens_per_second=50_000.0,
        seeds=(9000,),
    )
    assert second["pairs"] == [{"seed": 9000, "action": "skipped_complete"}]


def test_session_file_records_the_story(tmp_path: Path) -> None:
    output = tmp_path / "artifacts" / "session-1.json"
    summary = run_pair_session(
        config=_config(),
        session_dir=tmp_path / "sessions",
        dataset_root=tmp_path / "dataset",
        code_revision="rev",
        remaining_hours=8.0,
        measured_tokens_per_second=50_000.0,
        seeds=(9000,),
        output=output,
    )
    stored = json.loads(output.read_text())
    assert stored["protocol"] == "v0-compute-efficiency-5.2"
    assert stored["seeds_requested"] == [9000]
    assert stored["declared_seed_queue"] == list(DECLARED_SEEDS)
    assert stored["pairs_completed"] == summary["pairs_completed"]
    assert stored["remaining_hours_at_entry"] == 8.0
    assert stored["latest_published_seed"] == 9000


def test_interrupted_exception_carries_step() -> None:
    error = V0SessionInterrupted("stopped", 120)
    assert error.step == 120
    assert "stopped" in str(error)


# --- CLI wrapper (--mode pair-session) --------------------------------------


def _write_benchmark(path: Path, tokens_per_second: float, dtype: str) -> None:
    """A minimal-but-honest benchmark artifact as the benchmark script writes it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "benchmark": "v0-throughput-5.2",
                "measured_tokens_per_second": tokens_per_second,
                "dtype_decision": dtype,
            }
        )
    )


def _cli_argv(extra: list[str], benchmark: Path | None = None, profile: bool = True) -> list[str]:
    argv = ["act.training.cli", "--mode", "pair-session", "--hours", "8"]
    if profile:
        argv.append("--profile")
    if benchmark is not None:
        argv += ["--benchmark", str(benchmark)]
    return argv + extra


def test_cli_pair_session_runs_the_loop(tmp_path: Path, monkeypatch) -> None:
    import sys

    from act.training import cli

    output = tmp_path / "log.json"
    monkeypatch.setattr(
        sys, "argv",
        _cli_argv(
            [
                "--tokens-per-second", "50000",
                "--seeds", "9000",
                "--session-dir", str(tmp_path / "sessions"),
                "--dataset-root", str(tmp_path / "dataset"),
                "--output", str(output),
            ]
        ),
    )
    cli.main()
    summary = json.loads(output.read_text())
    assert summary["protocol"] == "v0-compute-efficiency-5.2"
    assert summary["pairs_completed"] == 1
    assert published_seed_status(tmp_path / "dataset", 9000) == "complete"


def test_cli_reads_benchmark_artifact_and_dtype(tmp_path: Path, monkeypatch) -> None:
    import sys

    from act.training import cli

    benchmark = tmp_path / "v0-benchmark.json"
    _write_benchmark(benchmark, 2_000.0, "fp16")  # below the fp32 floor
    output = tmp_path / "log.json"
    monkeypatch.setattr(
        sys, "argv",
        _cli_argv(
            [
                "--benchmark", str(benchmark),
                "--seeds", "9000",
                "--session-dir", str(tmp_path / "sessions"),
                "--dataset-root", str(tmp_path / "dataset"),
                "--output", str(output),
            ]
        ),
    )
    cli.main()
    summary = json.loads(output.read_text())
    assert summary["measured_tokens_per_second"] == 2_000.0
    # The pair ran with the measured dtype decision applied to the session.
    payload = json.loads(benchmark.read_text())
    assert payload["dtype_decision"] == "fp16"


def test_cli_rejects_stale_benchmark_artifact(tmp_path: Path, monkeypatch) -> None:
    """A recorded dtype that contradicts the measured number must not launch."""
    import sys

    import pytest as pytest_module

    from act.training import cli

    benchmark = tmp_path / "v0-benchmark.json"
    _write_benchmark(benchmark, 2_000.0, "fp32")  # recorded != derived (fp16)
    monkeypatch.setattr(
        sys, "argv",
        _cli_argv(
            [
                "--benchmark", str(benchmark),
                "--seeds", "9000",
                "--session-dir", str(tmp_path / "sessions"),
                "--dataset-root", str(tmp_path / "dataset"),
            ]
        ),
    )
    with pytest_module.raises(SystemExit, match="stale or edited"):
        cli.main()
    assert not (tmp_path / "sessions").exists()


def test_cli_rejects_dtype_override_contradicting_the_rule(
    tmp_path: Path, monkeypatch
) -> None:
    import sys

    import pytest as pytest_module

    from act.training import cli

    monkeypatch.setattr(
        sys, "argv",
        _cli_argv(
            [
                "--tokens-per-second", "50000",  # measures fp32
                "--dtype", "fp16",  # but the user forces fp16
                "--seeds", "9000",
                "--session-dir", str(tmp_path / "sessions"),
                "--dataset-root", str(tmp_path / "dataset"),
            ]
        ),
    )
    with pytest_module.raises(SystemExit, match="contradicts"):
        cli.main()


def test_cli_requires_benchmark_or_throughput(tmp_path: Path, monkeypatch) -> None:
    import sys

    import pytest as pytest_module

    from act.training import cli

    monkeypatch.setattr(
        sys, "argv",
        _cli_argv(
            [
                "--session-dir", str(tmp_path / "sessions"),
                "--dataset-root", str(tmp_path / "dataset"),
                "--benchmark", str(tmp_path / "missing.json"),
            ]
        ),
    )
    with pytest_module.raises(SystemExit, match="not found"):
        cli.main()


def test_cli_default_queue_is_the_declared_block(tmp_path: Path, monkeypatch) -> None:
    """Without --seeds the launch command queues exactly the declared block."""
    import sys

    captured: dict = {}

    from act.training import cli

    def fake_run_pair_session(**kwargs):
        captured.update(kwargs)
        return {
            "protocol": "v0-compute-efficiency-5.2",
            "seeds_requested": list(kwargs["seeds"]),
            "declared_seed_queue": [9000, 9001],
            "remaining_hours_at_entry": kwargs["remaining_hours"],
            "measured_tokens_per_second": kwargs["measured_tokens_per_second"],
            "pair_time_hours": 0.1,
            "pairs_completed": 0,
            "latest_published_seed": None,
            "pairs": [],
        }

    monkeypatch.setattr(cli, "run_pair_session", fake_run_pair_session)
    output = tmp_path / "log.json"
    monkeypatch.setattr(
        sys, "argv",
        _cli_argv(
            [
                "--tokens-per-second", "50000",
                "--session-dir", str(tmp_path / "sessions"),
                "--dataset-root", str(tmp_path / "dataset"),
                "--output", str(output),
            ],
            profile=False,  # the real launch command runs the declared config
        )
    )
    cli.main()
    assert captured["seeds"] == tuple(range(15000, 15012))
    assert captured["code_revision"]  # captured from git, not 'unknown'
    assert captured["config"].dtype == "fp32"
    assert captured["config"].max_checkpoints == 2  # registered pin
    assert captured["config"].declared is True  # no --profile model overrides


def test_cli_defaults_to_the_registered_probe_knobs(tmp_path: Path, monkeypatch) -> None:
    """An unflagged launch still reproduces the original registration."""
    import sys

    captured: dict = {}

    from act.training import cli

    def fake_run_pair_session(**kwargs):
        captured.update(kwargs)
        return {
            "protocol": "v0-compute-efficiency-5.2",
            "seeds_requested": list(kwargs["seeds"]),
            "declared_seed_queue": [],
            "remaining_hours_at_entry": kwargs["remaining_hours"],
            "measured_tokens_per_second": kwargs["measured_tokens_per_second"],
            "pair_time_hours": 0.1,
            "pairs_completed": 0,
            "latest_published_seed": None,
            "pairs": [],
        }

    monkeypatch.setattr(cli, "run_pair_session", fake_run_pair_session)
    monkeypatch.setattr(
        sys, "argv",
        _cli_argv(
            [
                "--tokens-per-second", "50000",
                "--seeds", "15000",
                "--session-dir", str(tmp_path / "sessions"),
                "--dataset-root", str(tmp_path / "dataset"),
                "--output", str(tmp_path / "log.json"),
            ],
            profile=False,
        )
    )
    cli.main()

    assert captured["config"].probe_candidate_policy == "prefix"
    assert captured["config"].probe_selection == "argmin"


def test_cli_passes_the_adopted_probe_knobs_into_the_session(
    tmp_path: Path, monkeypatch
) -> None:
    """Adopted section 5.5: the launch command can select the primary and control.

    The primary run passes `--probe-candidate-policy spread`; the pool-matched
    control is the same command over the same seeds with
    `--probe-selection random` and its own dataset root, so the regime is
    recorded in every seed summary rather than left implicit.
    """
    import sys

    captured: dict = {}

    from act.training import cli

    def fake_run_pair_session(**kwargs):
        captured.update(kwargs)
        return {
            "protocol": "v0-compute-efficiency-5.2",
            "seeds_requested": list(kwargs["seeds"]),
            "declared_seed_queue": [],
            "remaining_hours_at_entry": kwargs["remaining_hours"],
            "measured_tokens_per_second": kwargs["measured_tokens_per_second"],
            "pair_time_hours": 0.1,
            "pairs_completed": 0,
            "latest_published_seed": None,
            "pairs": [],
        }

    monkeypatch.setattr(cli, "run_pair_session", fake_run_pair_session)
    monkeypatch.setattr(
        sys, "argv",
        _cli_argv(
            [
                "--tokens-per-second", "50000",
                "--seeds", "15000,15001",
                "--probe-candidate-policy", "spread",
                "--probe-selection", "random",
                "--session-dir", str(tmp_path / "sessions"),
                "--dataset-root", str(tmp_path / "dataset-control"),
                "--output", str(tmp_path / "log.json"),
            ],
            profile=False,
        )
    )
    cli.main()

    assert captured["config"].probe_candidate_policy == "spread"
    assert captured["config"].probe_selection == "random"
    assert captured["seeds"] == (15000, 15001)


# --- Remote dataset publish-on-exit (--kaggle-dataset) -----------------------


def _fake_remote(monkeypatch, calls: dict, *, fail: bool = False) -> None:
    """Stub the lazy remote imports inside cli._publish_remote/restore."""
    import act.ops.kaggle_remote as remote

    class StubClient:
        def create_version(self, folder, version_notes):
            if fail:
                raise RuntimeError("409 conflict")
            calls["versions"] = calls.get("versions", 0) + 1
            return version_notes

        def download_files(self, dataset, target):
            calls["downloads"] = calls.get("downloads", 0) + 1

    def fake_default_client():
        calls["client_built"] = calls.get("client_built", 0) + 1
        return StubClient()

    monkeypatch.setattr(remote, "default_client", fake_default_client)


def test_cli_restore_failure_blocks_the_session_before_training(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Fail-fast: no remote state means no training and a clean error."""
    import sys

    from act.training import cli

    calls: dict = {}
    _fake_remote(monkeypatch, calls)

    class Blocked(Exception):
        pass

    def failing_restore(*args, **kwargs):
        raise RuntimeError("403 forbidden")

    import act.ops.kaggle_remote as remote

    monkeypatch.setattr(remote, "restore_latest", failing_restore)
    monkeypatch.setattr(
        sys, "argv",
        _cli_argv(
            [
                "--tokens-per-second", "50000",
                "--session-dir", str(tmp_path / "sessions"),
                "--dataset-root", str(tmp_path / "dataset"),
                "--kaggle-dataset", "owner/act-checkpoints-v0-5-1",
            ]
        ),
    )
    with pytest.raises(SystemExit, match="session not started"):
        cli.main()
    assert not (tmp_path / "sessions").exists()  # nothing ran


def test_cli_without_kaggle_flag_touches_no_remote(tmp_path: Path, monkeypatch) -> None:
    """The default command must not build a client or publish remotely."""
    import sys

    from act.training import cli

    calls: dict = {}
    _fake_remote(monkeypatch, calls)

    def fake_run_pair_session(**kwargs):
        return {
            "protocol": "v0-compute-efficiency-5.2",
            "seeds_requested": [9000],
            "declared_seed_queue": [9000],
            "remaining_hours_at_entry": 8.0,
            "measured_tokens_per_second": 50000.0,
            "pair_time_hours": 0.1,
            "pairs_completed": 0,
            "latest_published_seed": None,
            "pairs": [],
        }

    monkeypatch.setattr(cli, "run_pair_session", fake_run_pair_session)
    monkeypatch.setattr(
        sys, "argv",
        _cli_argv(
            [
                "--tokens-per-second", "50000",
                "--session-dir", str(tmp_path / "sessions"),
                "--dataset-root", str(tmp_path / "dataset"),
                "--output", str(tmp_path / "log.json"),
            ]
        ),
    )
    cli.main()
    assert calls == {}  # no client built, no download, no version push


def test_cli_publishes_on_exit_with_kaggle_flag(tmp_path: Path, monkeypatch) -> None:
    """--kaggle-dataset restores before the loop and publishes after it."""
    import sys

    from act.training import cli

    calls: dict = {}
    _fake_remote(monkeypatch, calls)

    def fake_run_pair_session(**kwargs):
        # The restore must already have happened when the loop starts.
        assert calls.get("downloads") == 1
        # Simulate a completed pair's local handoff publish.
        seed_dir = Path(kwargs["dataset_root"]) / "act-checkpoints" / "v0-5-1" / "seed-9000"
        (seed_dir / "manifests").mkdir(parents=True, exist_ok=True)
        (seed_dir / "manifests" / "manifest.json").write_text("{}\n")
        return {
            "protocol": "v0-compute-efficiency-5.2",
            "seeds_requested": [9000],
            "declared_seed_queue": [9000],
            "remaining_hours_at_entry": 8.0,
            "measured_tokens_per_second": 50000.0,
            "pair_time_hours": 0.1,
            "pairs_completed": 1,
            "latest_published_seed": 9000,
            "pairs": [],
        }

    monkeypatch.setattr(cli, "run_pair_session", fake_run_pair_session)
    monkeypatch.setattr(
        sys, "argv",
        _cli_argv(
            [
                "--tokens-per-second", "50000",
                "--session-dir", str(tmp_path / "sessions"),
                "--dataset-root", str(tmp_path / "dataset"),
                "--output", str(tmp_path / "log.json"),
                "--kaggle-dataset", "owner/act-checkpoints-v0-5-1",
            ]
        ),
    )
    cli.main()
    assert calls.get("downloads") == 1  # session-start restore
    assert calls.get("versions") == 1  # exit-path publish


def test_cli_exit_publish_failure_does_not_crash_the_session(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A failed remote push records the outcome and exits zero-ish."""
    import sys

    from act.training import cli

    calls: dict = {}
    _fake_remote(monkeypatch, calls, fail=True)

    def fake_run_pair_session(**kwargs):
        # Simulate a completed pair's local handoff publish so the exit
        # push has content and actually reaches the (failing) client.
        seed_dir = Path(kwargs["dataset_root"]) / "act-checkpoints" / "v0-5-1" / "seed-9000"
        (seed_dir / "manifests").mkdir(parents=True, exist_ok=True)
        (seed_dir / "manifests" / "manifest.json").write_text("{}\n")
        return {
            "protocol": "v0-compute-efficiency-5.2",
            "seeds_requested": [9000],
            "declared_seed_queue": [9000],
            "remaining_hours_at_entry": 8.0,
            "measured_tokens_per_second": 50000.0,
            "pair_time_hours": 0.1,
            "pairs_completed": 1,
            "latest_published_seed": 9000,
            "pairs": [],
        }

    monkeypatch.setattr(cli, "run_pair_session", fake_run_pair_session)
    monkeypatch.setattr(
        sys, "argv",
        _cli_argv(
            [
                "--tokens-per-second", "50000",
                "--session-dir", str(tmp_path / "sessions"),
                "--dataset-root", str(tmp_path / "dataset"),
                "--output", str(tmp_path / "log.json"),
                "--kaggle-dataset", "owner/act-checkpoints-v0-5-1",
            ]
        ),
    )
    cli.main()  # must not raise
    stderrish = capsys.readouterr().out
    assert '"status": "failed"' in stderrish  # outcome recorded, not raised
