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
from act.training.v0_runner import (
    DECLARED_SEEDS,
    BlockTreatmentMismatch,
    check_block_treatment_uniform,
    run_pair_session,
)
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
    # Re-reserved twice, for the same reason both times: the block must be blind and it
    # must be the registered configuration. 9000-9011 -> 15000-15011 (section 5.5 change
    # 4: the original block was spent by the section 5.3 diagnostic, so the harm guard was
    # no longer blind); 15000-15011 -> 15024-15035 (section 5.65: its first four seeds ran
    # before the launch path carried the 5.47 rebuild treatment, and their harm numbers
    # had been read). See RESEARCH_REPORT.md sections 5.5, 5.47, 5.65 and 9.9.
    assert DECLARED_SEEDS == tuple(range(15024, 15036))
    assert 9000 not in DECLARED_SEEDS
    assert 15000 not in DECLARED_SEEDS  # the superseded block keeps its own artifact
    assert 15012 not in DECLARED_SEEDS  # the benchmark's reserved data seed


def test_failed_local_publish_stops_the_queue_without_losing_earlier_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed handoff publish must not discard work already published.

    The exit publish never raises by design so a broken push cannot destroy a
    session; the per-pair local publish needs the mirror-image guarantee. It
    must not be recorded as ``completed`` (the seed is not complete-
    restorable), and it must stop the queue rather than spend more GPU on a
    substrate that cannot publish - while leaving the first seed's version in
    place for the exit publish to push.
    """
    from act.training import v0_runner

    real_publish = v0_runner.publish_seed
    calls: list[int] = []

    def flaky_publish(dataset_root, result, **kwargs):
        calls.append(int(result.config["seed"]))
        if len(calls) == 2:
            raise OSError(28, "No space left on device")
        return real_publish(dataset_root, result, **kwargs)

    monkeypatch.setattr(v0_runner, "publish_seed", flaky_publish)
    summary = run_pair_session(
        config=_config(),
        session_dir=tmp_path / "sessions",
        dataset_root=tmp_path / "dataset",
        code_revision="rev",
        remaining_hours=8.0,
        measured_tokens_per_second=1e9,
        seeds=(9000, 9001, 9002),
        output=tmp_path / "summary.json",
    )

    actions = [entry["action"] for entry in summary["pairs"]]
    assert actions == ["completed", "publish_failed"]
    assert summary["pairs_completed"] == 1
    # The third seed was never attempted, and the first seed stays published.
    assert 9002 not in calls
    assert published_seed_status(tmp_path / "dataset", 9000) == "complete"
    assert published_seed_status(tmp_path / "dataset", 9001) != "complete"


def test_failed_pair_stops_the_queue_without_losing_earlier_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pair that raises must not cost the session the seeds it published.

    The durable copy is the exit publish, which a crash never reaches, so a
    training failure (an OOM, a killed arm process) has to stop the queue and
    let the session exit with what it completed - while never being recorded
    as ``completed``, so the queue accounting and the driver stay honest.
    """
    from act.training import v0_runner

    real_pair = v0_runner._run_one_pair
    calls: list[int] = []

    def flaky_pair(*, seed: int, **kwargs):
        calls.append(seed)
        if seed == 9001:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return real_pair(seed=seed, **kwargs)

    monkeypatch.setattr(v0_runner, "_run_one_pair", flaky_pair)
    summary = run_pair_session(
        config=_config(),
        session_dir=tmp_path / "sessions",
        dataset_root=tmp_path / "dataset",
        code_revision="rev",
        remaining_hours=8.0,
        measured_tokens_per_second=1e9,
        seeds=(9000, 9001, 9002),
        output=tmp_path / "summary.json",
    )

    actions = [entry["action"] for entry in summary["pairs"]]
    assert actions == ["completed", "training_failed"]
    assert calls == [9000, 9001]  # 9002 was never started
    assert "out of memory" in summary["pairs"][1]["error"]
    assert summary["pairs_completed"] == 1
    # The seed that finished stays published, for the exit publish to push.
    assert published_seed_status(tmp_path / "dataset", 9000) == "complete"
    assert published_seed_status(tmp_path / "dataset", 9001) != "complete"


def test_every_completed_pair_is_mirrored_to_the_dataset(tmp_path: Path) -> None:
    """Each pair reaches the versioned dataset as it completes, not at exit.

    Measured 2026-10-05: four sessions in a row ended ERROR inside the single
    exit upload, and each lost the three pairs it had trained. The hook is what
    makes a pair durable while the session is still running, so it must be
    called once per completed pair, after that pair's own local publish (pushing
    earlier would upload a folder without this seed in it), and its outcome must
    land in the session log beside the seed.
    """
    mirrored: list[int] = []
    published_at: list[str | None] = []

    def mirror(seed: int) -> dict[str, object]:
        mirrored.append(seed)
        published_at.append(published_seed_status(tmp_path / "dataset", seed))
        return {"status": "published", "seeds_published": 1}

    summary = run_pair_session(
        config=_config(),
        session_dir=tmp_path / "sessions",
        dataset_root=tmp_path / "dataset",
        code_revision="rev",
        remaining_hours=8.0,
        measured_tokens_per_second=1e9,
        seeds=(9000, 9001),
        output=tmp_path / "summary.json",
        remote_publish=mirror,
    )

    assert mirrored == [9000, 9001]
    assert published_at == ["complete", "complete"]
    assert [entry["remote_publish"] for entry in summary["pairs"]] == [
        {"status": "published", "seeds_published": 1},
        {"status": "published", "seeds_published": 1},
    ]

    # Re-running the same command is how the queue advances, and every seed is
    # now skipped - which must not mirror anything: re-pushing an unchanged
    # handoff is the whole ~38 GiB upload this change exists to avoid.
    mirrored.clear()
    rerun = run_pair_session(
        config=_config(),
        session_dir=tmp_path / "sessions",
        dataset_root=tmp_path / "dataset",
        code_revision="rev",
        remaining_hours=8.0,
        measured_tokens_per_second=1e9,
        seeds=(9000, 9001),
        output=tmp_path / "summary.json",
        remote_publish=mirror,
    )
    assert [entry["action"] for entry in rerun["pairs"]] == [
        "skipped_complete",
        "skipped_complete",
    ]
    assert mirrored == []


def test_an_interrupted_pair_is_not_mirrored(tmp_path: Path) -> None:
    """A deadline-stop snapshot is not progress, so it is not published.

    The snapshot exists for crash forensics and is deliberately re-trained from
    scratch by the next session (pairs are never resumed across sessions), so
    pushing it would spend an upload on a state the queue does not count and
    will not keep.
    """
    consumed = {"seconds": 0.0}

    def clock() -> float:
        consumed["seconds"] += 180.0
        return consumed["seconds"]

    mirrored: list[int] = []

    def mirror(seed: int) -> dict[str, object]:
        mirrored.append(seed)
        return {"status": "published", "seeds_published": 1}

    summary = run_pair_session(
        config=_config(total_steps=24, checkpoint_interval=2),
        session_dir=tmp_path / "sessions",
        dataset_root=tmp_path / "dataset",
        code_revision="rev",
        remaining_hours=1.0,
        measured_tokens_per_second=1e9,
        clock=clock,
        seeds=(9000,),
        output=tmp_path / "summary.json",
        remote_publish=mirror,
    )

    assert [entry["action"] for entry in summary["pairs"]] == ["interrupted"]
    assert mirrored == []
    assert "remote_publish" not in summary["pairs"][0]
    assert published_seed_status(tmp_path / "dataset", 9000) == "in-progress"


def test_a_failing_remote_mirror_is_recorded_and_does_not_stop_the_queue(
    tmp_path: Path,
) -> None:
    """A push that raises, or reports a failure, must not cost the session.

    The exit publish already never raises so a broken push cannot destroy a
    finished session; the per-pair mirror needs the same guarantee, because it
    now runs inside the loop, and it must not stop the queue the way a failed
    *local* publish does - the local handoff is intact, the next pair's push
    carries the whole folder, and a persistent failure is reported by the exit
    publish the driver reads.
    """

    def mirror(seed: int) -> dict[str, object]:
        if seed == 9000:
            raise RuntimeError("kaggle api unavailable")
        return {"status": "failed", "error": "HTTPError: 503"}

    summary = run_pair_session(
        config=_config(),
        session_dir=tmp_path / "sessions",
        dataset_root=tmp_path / "dataset",
        code_revision="rev",
        remaining_hours=8.0,
        measured_tokens_per_second=1e9,
        seeds=(9000, 9001),
        output=tmp_path / "summary.json",
        remote_publish=mirror,
    )

    assert [entry["action"] for entry in summary["pairs"]] == [
        "completed",
        "completed",
    ]
    assert summary["pairs_completed"] == 2
    raised = summary["pairs"][0]["remote_publish"]
    assert raised["status"] == "raised"
    assert "kaggle api unavailable" in raised["error"]
    assert summary["pairs"][1]["remote_publish"] == {
        "status": "failed",
        "error": "HTTPError: 503",
    }
    # Both seeds stay complete locally: a failed push is the exit publish's
    # problem, not a reason to throw away trained work.
    assert published_seed_status(tmp_path / "dataset", 9000) == "complete"
    assert published_seed_status(tmp_path / "dataset", 9001) == "complete"


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


def test_single_arm_heuristic_session_publishes_one_arm(tmp_path: Path) -> None:
    """The routing-statistics heuristic publishes like the other single arms.

    The pair summary carries its loss under ``heuristic_final_loss`` and the
    published arm books rebuild-only search FLOPs (no probe bookings).
    """
    summary = run_pair_session(
        config=_config(arms=("heuristic",), mutation_budget=2),
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
    assert entry["heuristic_final_loss"] is not None
    assert entry["random_final_loss"] is None
    assert sorted(entry["search_flops_by_arm"]) == ["heuristic"]
    assert entry["search_flops_by_arm"]["heuristic"] > 0
    seed_summary = load_seed_summary(tmp_path / "dataset", 9000)
    assert seed_summary["status"] == "complete"
    arm = seed_summary["arms"]["heuristic"]
    assert arm["mutation_count"] == 2
    assert all(
        event["operation"] == "grow" for event in arm["mutation_history"]
    )


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


def _write_benchmark(
    path: Path,
    tokens_per_second: float,
    dtype: str,
    schedule_tokens_per_second: float | None = None,
) -> None:
    """A minimal-but-honest benchmark artifact as the benchmark script writes it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "v0-throughput-5.2",
        "measured_tokens_per_second": tokens_per_second,
        "dtype_decision": dtype,
    }
    if schedule_tokens_per_second is not None:
        payload["schedule_tokens_per_second"] = schedule_tokens_per_second
    path.write_text(json.dumps(payload))


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
    assert captured["seeds"] == tuple(range(15024, 15036))
    assert captured["code_revision"]  # captured from git, not 'unknown'
    assert captured["config"].dtype == "fp32"
    assert captured["config"].max_checkpoints == 2  # registered pin
    assert captured["config"].declared is True  # no --profile model overrides


def test_cli_schedules_from_the_trained_dtype_rate(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Pair time must come from the rate the session will actually run at.

    The pre-committed rule reads the fp32 number, but the session trains the
    decided dtype, and on the T4 fp16 does 2.27x the tokens/s (4,695 vs 2,066).
    Scheduling from the fp32 number made every session stop with usable clock
    left and bought twice the sessions it needed.
    """
    import sys

    from act.training import cli

    benchmark = tmp_path / "v0-benchmark.json"
    _write_benchmark(benchmark, 2_000.0, "fp16", schedule_tokens_per_second=4_500.0)
    captured: dict = {}

    def fake_run_pair_session(**kwargs):
        captured.update(kwargs)
        summary = {
            "protocol": "v0-compute-efficiency-5.2",
            "seeds_requested": list(kwargs["seeds"]),
            "declared_seed_queue": [9000],
            "remaining_hours_at_entry": kwargs["remaining_hours"],
            "measured_tokens_per_second": kwargs["measured_tokens_per_second"],
            "pair_time_hours": 0.1,
            "pairs_completed": 0,
            "latest_published_seed": None,
            "pairs": [],
        }
        summary.update(kwargs.get("summary_extra") or {})  # as the real runner does
        return summary

    monkeypatch.setattr(cli, "run_pair_session", fake_run_pair_session)
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
    cli.main()

    assert captured["config"].dtype == "fp16"
    assert captured["measured_tokens_per_second"] == 4_500.0
    # The runner records both rates in the durable session log it writes.
    assert captured["summary_extra"] == {
        "dtype": "fp16",
        "dtype_rule_tokens_per_second": 2_000.0,
        "schedule_tokens_per_second": 4_500.0,
    }
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["dtype_rule_tokens_per_second"] == 2_000.0
    assert printed["schedule_tokens_per_second"] == 4_500.0
    assert printed["dtype"] == "fp16"


def test_cli_falls_back_for_artifacts_without_a_schedule_rate(
    tmp_path: Path, monkeypatch
) -> None:
    """An artifact written before the field existed schedules from its own rate."""
    import sys

    from act.training import cli

    benchmark = tmp_path / "v0-benchmark.json"
    _write_benchmark(benchmark, 2_000.0, "fp16")  # no schedule_tokens_per_second
    captured: dict = {}

    def fake_run_pair_session(**kwargs):
        captured.update(kwargs)
        summary = {
            "protocol": "v0-compute-efficiency-5.2",
            "seeds_requested": list(kwargs["seeds"]),
            "declared_seed_queue": [9000],
            "remaining_hours_at_entry": kwargs["remaining_hours"],
            "measured_tokens_per_second": kwargs["measured_tokens_per_second"],
            "pair_time_hours": 0.1,
            "pairs_completed": 0,
            "latest_published_seed": None,
            "pairs": [],
        }
        summary.update(kwargs.get("summary_extra") or {})
        return summary

    monkeypatch.setattr(cli, "run_pair_session", fake_run_pair_session)
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
    cli.main()

    assert captured["measured_tokens_per_second"] == 2_000.0


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


# --- The section 5.28 rebuild treatment on the declared launch path ----------


def _fake_pair_session_result(kwargs: dict) -> dict:
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


def _run_cli_capturing_config(tmp_path: Path, monkeypatch, extra: list[str]) -> dict:
    """Drive pair-session mode with a stubbed runner and return the config it got."""
    import sys

    captured: dict = {}

    from act.training import cli

    def fake_run_pair_session(**kwargs):
        captured.update(kwargs)
        return _fake_pair_session_result(kwargs)

    monkeypatch.setattr(cli, "run_pair_session", fake_run_pair_session)
    monkeypatch.setattr(
        sys,
        "argv",
        _cli_argv(
            [
                "--tokens-per-second", "50000",
                "--seeds", "15000",
                *extra,
                "--session-dir", str(tmp_path / "sessions"),
                "--dataset-root", str(tmp_path / "dataset"),
                "--output", str(tmp_path / "log.json"),
            ],
            profile=False,
        ),
    )
    cli.main()
    return captured["config"]


def test_cli_keeps_the_historical_rebuild_unless_the_treatment_is_asked_for(
    tmp_path: Path, monkeypatch
) -> None:
    """The unflagged launch is the historical rebuild, and that is pinned.

    Every committed number from sections 5.16-5.23 was measured with
    `preserve_router_on_rebuild=False` / `new_expert_initialization="random"`
    (section 5.16 records that the default was deliberately kept for exactly
    that reason), so flipping it here would silently change what the declared
    launch command means. The flags below are how a run opts in instead.
    """
    config = _run_cli_capturing_config(tmp_path, monkeypatch, [])

    assert config.preserve_router_on_rebuild is False
    assert config.new_expert_initialization == "random"


def test_cli_can_specify_the_rebuild_treatment_on_the_declared_path(
    tmp_path: Path, monkeypatch
) -> None:
    """Section 5.38.3 says a real-scale protocol should carry the treatment.

    Before these flags the declared launch command could not express that at
    all: `_run_pair_session` built its config from `declared_session_config` and
    passed no rebuild field, so the treatment was unreachable from the runbook.
    """
    config = _run_cli_capturing_config(
        tmp_path,
        monkeypatch,
        ["--preserve-router-on-rebuild", "--new-expert", "clone"],
    )

    assert config.preserve_router_on_rebuild is True
    assert config.new_expert_initialization == "clone"
    # The treatment must not move the geometry the declared prediction reads.
    assert config.total_steps == 6000
    assert config.probe_batch_count == 1


def test_cli_refuses_clone_without_router_preservation(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A half-specified treatment is an operator error, not a silent downgrade."""
    import pytest

    with pytest.raises(SystemExit) as excinfo:
        _run_cli_capturing_config(tmp_path, monkeypatch, ["--new-expert", "clone"])

    assert "--preserve-router-on-rebuild" in str(excinfo.value)


# --- Remote dataset publish-on-exit (--kaggle-dataset) -----------------------


def _fake_remote(monkeypatch, calls: dict, *, fail: bool = False) -> None:
    """Stub the lazy remote imports inside cli._publish_remote/restore."""
    import act.ops.kaggle_remote as remote

    class StubClient:
        def create_version(self, folder, version_notes, *, seeds=None):
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
            "pairs": [{"seed": 9000, "action": "completed"}],
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


def test_cli_exit_publish_skips_seeds_restored_as_complete(
    tmp_path: Path, monkeypatch
) -> None:
    """A seed the restore found complete is not re-pushed at exit (2026-10-19)."""
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
            "latest_published_seed": 9000,
            "pairs": [{"seed": 9000, "action": "skipped_complete"}],
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
    assert calls.get("downloads") == 1
    assert calls.get("versions") is None  # nothing trained: no exit push


def test_cli_exit_publish_failure_does_not_crash_the_session(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A failed remote push records the outcome and exits zero-ish."""
    import sys

    from act.training import cli

    calls: dict = {}
    _fake_remote(monkeypatch, calls, fail=True)
    # This test exercises the exit path of a failed push, not the retry schedule: the
    # three attempts still happen, but their backoff is zeroed so the suite does not
    # spend 15 s (5.0 x 1 + 5.0 x 2) in wall-clock sleep. The schedule itself is pinned
    # where it belongs, in tests/test_kaggle_remote.py, which injects a recording sleep.
    import act.ops.kaggle_remote as remote_module

    monkeypatch.setattr(remote_module, "PUBLISH_RETRY_SECONDS", 0.0)

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
            "pairs": [{"seed": 9000, "action": "completed"}],
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


# --- the section 5.14 do-nothing queue launch (--session-arms) --------------


def test_cli_session_arms_default_is_the_registered_pair(
    tmp_path: Path, monkeypatch
) -> None:
    """An unflagged launch still runs the registered random/adaptive pair."""
    import sys

    from act.training import cli

    captured: dict = {}

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

    assert captured["config"].arms == ("random", "adaptive")


def test_cli_session_arms_null_builds_a_declared_null_session(
    tmp_path: Path, monkeypatch
) -> None:
    """The 5.14 do-nothing queue launch: declared fidelity, single null arm."""
    import sys

    from act.training import cli

    captured: dict = {}

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
                "--seeds", "15024-15035",
                "--session-arms", "null",
                "--session-dir", str(tmp_path / "sessions"),
                "--dataset-root", str(tmp_path / "dataset"),
                "--output", str(tmp_path / "log.json"),
            ],
            profile=False,  # the real launch command runs the declared config
        )
    )
    cli.main()

    config = captured["config"]
    assert config.arms == ("null",)
    assert config.declared is True  # declared 285M fidelity, no overrides
    assert config.total_steps == 6_000
    assert config.probe_batch_count == 1
    # The runbook's documented range form must parse, not crash.
    assert captured["seeds"] == tuple(range(15024, 15036))


def test_cli_seeds_accepts_the_runbook_range_form(tmp_path: Path, monkeypatch) -> None:
    """The runbook documents '--seeds 15024-15035'; the CLI must honor it.

    Regression pin for a latent parser bug the 5.14 null-queue tests exposed:
    the pair-session seed parser only handled comma lists, so the documented
    launch command crashed with ``invalid literal for int()``.
    """
    import sys

    from act.training import cli

    captured: dict = {}

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
                "--seeds", "15024-15035",
                "--session-dir", str(tmp_path / "sessions"),
                "--dataset-root", str(tmp_path / "dataset"),
                "--output", str(tmp_path / "log.json"),
            ],
            profile=False,
        )
    )
    cli.main()

    assert captured["seeds"] == tuple(range(15024, 15036))


def test_cli_session_arms_rejects_unknown_and_empty(tmp_path: Path, monkeypatch) -> None:
    import sys

    from act.training import cli

    monkeypatch.setattr(
        sys, "argv",
        _cli_argv(
            [
                "--tokens-per-second", "50000",
                "--session-arms", "bogus",
                "--session-dir", str(tmp_path / "sessions"),
                "--dataset-root", str(tmp_path / "dataset"),
            ],
            profile=False,
        )
    )
    with pytest.raises(ValueError, match="unsupported V0 session arms"):
        cli.main()  # unknown arm is refused by the config validation

    monkeypatch.setattr(
        sys, "argv",
        _cli_argv(
            [
                "--tokens-per-second", "50000",
                "--session-arms", " , ",
                "--session-dir", str(tmp_path / "sessions"),
                "--dataset-root", str(tmp_path / "dataset"),
            ],
            profile=False,
        )
    )
    with pytest.raises(SystemExit, match="--session-arms resolved to an empty tuple"):
        cli.main()


# --- Single-treatment blocks (section 5.47) -----------------------------------
#
# A block is what the verdict judges, and its seeds are compared *within* the block,
# so a block whose seeds disagree on the rebuild treatment is two experiments sharing
# a seed range. The treated and untreated declared runs differ on exactly two fields,
# both valid, neither a gate input - so nothing downstream would have refused the mix.


def _publish_tiny_seed(
    dataset_root: Path,
    tmp_path: Path,
    seed: int,
    log=None,
    **overrides,
):
    """Publish one complete tiny seed, recording `overrides` in its summary config."""
    return run_pair_session(
        config=_config(**overrides),
        session_dir=tmp_path / "sessions",
        dataset_root=dataset_root,
        code_revision="rev",
        remaining_hours=8.0,
        measured_tokens_per_second=1e9,
        seeds=(seed,),
        output=tmp_path / f"summary-{seed}.json",
        log=log,
    )


def test_a_session_refuses_to_append_a_seed_under_a_different_treatment(
    tmp_path: Path,
) -> None:
    """The declared block's own failure, in miniature.

    Seeds 15000-15003 were published under the historical rebuild because the Kaggle
    launch builder never passed the two flags the runbook prints, so the block is
    untreated and the amended command would have appended treated seeds to it.
    """
    dataset_root = tmp_path / "dataset"
    _publish_tiny_seed(dataset_root, tmp_path, 9000)  # the CLI default: untreated
    log: list[str] = []

    with pytest.raises(BlockTreatmentMismatch) as excinfo:
        _publish_tiny_seed(
            dataset_root,
            tmp_path,
            9001,
            log=log.append,
            preserve_router_on_rebuild=True,
            new_expert_initialization="clone",
        )

    message = str(excinfo.value)
    # Both fields, both values, and which published seeds hold the other value.
    assert "preserve_router_on_rebuild: this session True, published False" in message
    assert "new_expert_initialization: this session 'clone', published 'random'" in message
    assert "seed(s) 9000" in message
    # The refusal is before the pair, not after it: nothing trained, nothing published.
    assert published_seed_status(dataset_root, 9001) is None
    assert not any("starting pair" in line for line in log)


def test_a_narrower_queue_still_sees_the_seeds_it_would_be_appended_to(
    tmp_path: Path,
) -> None:
    """The first version of this guard read only the seeds it was handed, and missed it.

    A resumed campaign can be launched ``--seeds 15004-15011`` - the queue is a flag
    precisely so a block can be advanced in pieces - and that session appends to the same
    block. Reading the requested seeds instead of the published versions made the check
    silent in exactly the case it exists for, because the seeds that disagree are the ones
    already finished.
    """
    dataset_root = tmp_path / "dataset"
    _publish_tiny_seed(dataset_root, tmp_path, 9000)  # untreated, as published

    with pytest.raises(BlockTreatmentMismatch, match="9000"):
        _publish_tiny_seed(
            dataset_root,
            tmp_path,
            9001,
            preserve_router_on_rebuild=True,
            new_expert_initialization="clone",
        )


def test_a_matching_block_treatment_is_stated_in_the_log_and_the_pair_still_runs(
    tmp_path: Path,
) -> None:
    """The guard is not only a refusal - it is how the log says which run this is."""
    dataset_root = tmp_path / "dataset"
    _publish_tiny_seed(dataset_root, tmp_path, 9000)
    log: list[str] = []

    summary = _publish_tiny_seed(dataset_root, tmp_path, 9001, log=log.append)

    assert [entry["action"] for entry in summary["pairs"]] == ["completed"]
    assert any("block treatment matches the published seed(s)" in line for line in log)
    assert any("preserve_router_on_rebuild=False" in line for line in log)


def test_the_first_session_states_the_treatment_it_sets(tmp_path: Path) -> None:
    """The session that opens a block defines its treatment, and says so."""
    log: list[str] = []
    _publish_tiny_seed(tmp_path / "dataset", tmp_path, 9000, log=log.append)

    assert any(
        "no published seed records a rebuild treatment yet" in line for line in log
    )


def test_a_summary_without_a_recorded_config_is_not_read_as_a_conflict(
    tmp_path: Path,
) -> None:
    """Artifacts from before the fields existed carry no record, not a disagreement.

    The verdict's conformance block already reads that case as ``not_recorded`` so no
    spent block is re-read as non-conforming; the runner's guard has to agree, or it
    would refuse to extend a block that recorded nothing to disagree with.
    """
    dataset_root = tmp_path / "dataset"
    reports = seed_directory(dataset_root, 9000) / "reports"
    reports.mkdir(parents=True)
    (reports / "seed-summary.json").write_text(
        json.dumps({"seed": 9000, "status": "complete"})
    )

    check_block_treatment_uniform(
        dataset_root=dataset_root,
        config=_config(
            preserve_router_on_rebuild=True, new_expert_initialization="clone"
        ),
    )


# --- the revision a Kaggle session records on every published seed ------------


def test_the_handoff_revision_comes_from_the_mounted_marker_without_git(
    tmp_path: Path,
) -> None:
    """Measured 2026-09-27: every published manifest of the campaign said ``unknown``.

    A Kaggle session imports ``src/`` from a mounted dataset that has no ``.git``, so
    ``git rev-parse`` is not a source of provenance there - and the field every seed's
    manifest carries was the literal string ``unknown`` for all six published primary
    seeds. The mounted tree does carry the commit the driver archived, in ``revision.txt``
    beside ``pyproject.toml``, which is what this reads.
    """
    from act.training import cli

    commit = "0123456789abcdef0123456789abcdef01234567"
    (tmp_path / cli.REVISION_MARKER).write_text(commit + "\n", encoding="utf-8")

    assert cli._revision_from_marker(tmp_path) == commit
    # An explicit root is answered from that root alone: no git call, so this is the
    # in-kernel path exercised for real.
    assert cli._git_revision(tmp_path) == commit
    # Neither source present is still `unknown` rather than an empty string.
    assert cli._revision_from_marker(tmp_path / "elsewhere") is None
    assert cli._git_revision(tmp_path / "elsewhere") == "unknown"
    (tmp_path / cli.REVISION_MARKER).write_text("\n", encoding="utf-8")
    assert cli._git_revision(tmp_path) == "unknown"


def test_the_revision_marker_name_is_the_one_the_code_dataset_writes() -> None:
    """The two sides have to agree on the file, or the fallback reads nothing forever.

    ``kaggle_queue.stage_code_dataset`` writes the marker; this reader looks for it. They
    are separate modules with a shared string, so the agreement is pinned rather than
    assumed - the same failure shape as the doubled upload layout that made every kernel
    die at bootstrap while the marker alone looked healthy.
    """
    import sys

    from act.training import cli

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "scripts"))
    import kaggle_queue  # noqa: PLC0415 - imported here so the pin is about the constant

    assert cli.REVISION_MARKER == kaggle_queue.REVISION_MARKER


def test_a_workstation_still_records_gits_own_answer(tmp_path: Path) -> None:
    """The marker is a fallback, not a replacement: git answers first where it can."""
    import subprocess

    from act.training import cli

    expected = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=Path(__file__).resolve().parent.parent,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()

    assert cli._git_revision() == expected
