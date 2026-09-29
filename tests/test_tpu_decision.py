"""Pin the pre-registered TPU port decision, and that it says nothing it cannot support.

The decision rule was declared **before** the probe was pushed (see `DECISION_RULE` in
`scripts/tpu_probe.py`), because the one thing a go/no-go must not be is fitted to the number
that arrives. These tests pin the two thresholds, the arithmetic between them, and - the case
that matters most - that an incomplete record is `INCONCLUSIVE` rather than a pass. A probe
that fails to produce an eight-worker measurement is exactly the outcome where "no problem
found" would be the wrong reading.

They also pin the properties that decide what a *partial* record contains, because both were
learned by losing a queued run: the measurement order runs the rule's own dtype first, and a
phase that cannot run is contained (recorded, and the later measurements still happen). An
uncontained failure does not just lose its number - it ends the kernel ERROR after hours in a
queue, which is what the 2026-09-22 run did inside a dtype the decision never reads.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import threading
import time
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import tpu_probe  # noqa: E402
from act.model.factory import v0_moe_config_from_genome  # noqa: E402
from act.topology.genome import (  # noqa: E402
    ArchitectureGenome,
    create_genome,
    validate_genome,
)


def _record(*, aggregate: float, workers: int, first: float, steady: float) -> dict:
    """A probe record with only the fields the decision reads."""
    return {
        "bf16": {
            "rebuild": {
                "first_step_seconds_including_compile": first,
                "steady_step_seconds_mean": steady,
            }
        },
        "independent_seeds": {
            "workers": workers,
            "aggregate_tokens_per_second": aggregate,
        },
    }


#: The T4 pair the port would have to beat, from the launch artifact's derived schedule.
T4 = tpu_probe.T4_PAIR_TOKENS_PER_SECOND


def test_the_thresholds_are_the_declared_ones() -> None:
    assert tpu_probe.PORT_THRESHOLD_MULTIPLE == 2.0
    assert tpu_probe.RECOMPILE_OVERHEAD_LIMIT == 0.10
    assert T4 == 5_083.4


def test_the_rebuild_count_is_the_declared_cadence() -> None:
    """3 events per arm, each with a mutation plus 4 probed candidates."""
    assert tpu_probe.MUTATION_EVENTS_PER_ARM == 3
    assert tpu_probe.PROBED_CANDIDATES_PER_EVENT == 4
    assert tpu_probe.REBUILDS_PER_ARM == 15


def test_the_week_pivot_is_the_second_queue_not_the_pass_mark() -> None:
    """The rule's bar moves the verdict; the campaign's wall clock moves higher up.

    Both readings come from the one measurement, so the arithmetic that separates them is
    pinned here: at the declared 2.0x the TPU's week carries one pair queue (16.1 h) and the
    GPU side is 34.6 h, which is still two weeks against a 30 h week. It is the *second*
    queue that takes the block to one week, and the cheapest two are the null queue and one
    pair queue.
    """
    at_bar = tpu_probe.week_capacity(tpu_probe.PORT_THRESHOLD_MULTIPLE)
    assert at_bar["pair_queue_hours"] == pytest.approx(16.12, abs=0.01)
    assert at_bar["null_queue_hours"] == pytest.approx(8.06, abs=0.01)
    assert at_bar["carries_two_queues"] is False
    assert at_bar["gpu_hours_left"] == pytest.approx(
        2 * tpu_probe.GPU_HOURS_PER_QUEUE, abs=0.02
    )
    assert at_bar["gpu_weeks_left"] == 2

    pivot = tpu_probe.ONE_GPU_WEEK_MULTIPLE
    assert pivot == pytest.approx(2.417, abs=0.002)
    assert pivot > tpu_probe.PORT_THRESHOLD_MULTIPLE, (
        "a pivot at or below the bar would make the two readings the same number"
    )
    at_pivot = tpu_probe.week_capacity(pivot)
    assert at_pivot["null_and_pair_queue_hours"] == pytest.approx(
        tpu_probe.TPU_QUOTA_HOURS, abs=0.05
    )
    assert at_pivot["carries_two_queues"] is True
    assert at_pivot["gpu_hours_left"] == pytest.approx(
        tpu_probe.GPU_HOURS_PER_QUEUE, abs=0.02
    )
    assert at_pivot["gpu_weeks_left"] == 1


def test_the_null_amendment_is_the_other_route_to_one_week() -> None:
    """The two recorded options compose, and that is what lowers the pivot to the bar.

    Two queues on the TPU reach one week only at 2.417. **One** queue reaches it at the
    rule's own 2.0x if the null queue carries two seeds per pair - the amendment the journal
    already prices - because that queue is the only one-arm one, so halving its GPU hours is
    what closes the 34.63 -> 25.97 gap. Recording the composition here is what stops it being
    discovered after a verdict lands between the two numbers.
    """
    at_bar = tpu_probe.week_capacity(tpu_probe.PORT_THRESHOLD_MULTIPLE)
    amended_null = (
        tpu_probe.GPU_HOURS_PER_QUEUE / tpu_probe.NULL_SEEDS_PER_PAIR_IF_AMENDED
    )
    assert at_bar["carries_two_queues"] is False
    assert at_bar["one_week_with_one_queue_and_the_amendment"] is True
    assert at_bar["gpu_null_queue_hours_if_amended"] == pytest.approx(amended_null, abs=0.01)
    assert at_bar["gpu_hours_left_with_one_queue_and_the_amendment"] == pytest.approx(
        tpu_probe.GPU_HOURS_PER_QUEUE + amended_null, abs=0.02
    )
    assert (
        at_bar["gpu_hours_left_with_one_queue_and_the_amendment"]
        <= tpu_probe.GPU_QUOTA_HOURS
    )

    # A rate that cannot carry even one queue on the TPU gets neither route, amendment or
    # not: the fit needs the pair queue inside the TPU's own week as well.
    slow = tpu_probe.week_capacity(1.5)
    assert slow["pair_queue_hours"] > tpu_probe.TPU_QUOTA_HOURS
    assert slow["one_week_with_one_queue_and_the_amendment"] is False


def test_the_capacity_reading_rides_with_the_verdict() -> None:
    """A PORT between the bar and the pivot must not be read as "the block fits a week"."""
    decision = tpu_probe.evaluate(
        _record(aggregate=2.0 * T4, workers=8, first=31.0, steady=1.0)
    )
    assert decision["verdict"] == "PORT"
    assert decision["capacity"]["carries_two_queues"] is False
    assert decision["capacity"]["gpu_weeks_left"] == 2

    # And a record the rule cannot decide carries no capacity reading at all, rather than
    # one computed from a rate nothing measured.
    assert tpu_probe.evaluate({})["capacity"] is None


def test_the_probe_and_the_session_plan_agree_on_the_pair_overhead() -> None:
    """One number in two files: the probe's GPU clock has to be the campaign's own."""
    from act.training.session_plan import PAIR_OVERHEAD_HOURS, PairSchedule

    assert tpu_probe.PAIR_OVERHEAD_HOURS == PAIR_OVERHEAD_HOURS
    schedule = PairSchedule(
        tokens_per_second=tpu_probe.T4_PAIR_TOKENS_PER_SECOND, arms=2, arm_slots=2
    )
    assert tpu_probe.PAIR_HOURS_ON_T4 == pytest.approx(schedule.pair_time_hours)


def test_fast_and_cheap_to_rebuild_ports() -> None:
    decision = tpu_probe.evaluate(
        _record(aggregate=3 * T4, workers=8, first=31.0, steady=1.0)
    )
    assert decision["verdict"] == "PORT"
    assert decision["throughput_ok"] and decision["recompile_overhead_ok"]
    assert decision["measured"]["multiple_of_t4_pair"] == pytest.approx(3.0, rel=0.01)
    assert decision["measured"]["rebuild_compile_seconds"] == 30.0


def test_a_fast_accelerator_does_not_save_an_expensive_rebuild() -> None:
    """Great tokens/s, compiles that eat the session: the session path is out.

    The fixed-graph branch is *not* a consolation invented afterwards - it is declared
    with the rule, because the recompile cost is a property of the session path
    (a mutation event and each probed candidate rebuild the model) and not of an arm that
    trains one sampled architecture to a horizon.
    """
    decision = tpu_probe.evaluate(
        _record(aggregate=6 * T4, workers=8, first=601.0, steady=1.0)
    )
    assert decision["verdict"] == "PORT_FIXED_GRAPH_ONLY"
    assert decision["throughput_ok"] is True
    assert decision["recompile_overhead_ok"] is False
    assert "the session path is out" in decision["why"]


def test_slow_throughput_does_not_port_even_with_trivial_compiles() -> None:
    decision = tpu_probe.evaluate(
        _record(aggregate=1.5 * T4, workers=8, first=1.5, steady=1.0)
    )
    assert decision["verdict"] == "DO-NOT-PORT"
    assert decision["throughput_ok"] is False
    assert decision["recompile_overhead_ok"] is True
    assert "below the declared" in decision["why"]


def test_slow_and_expensive_ports_nothing() -> None:
    """Neither branch: the throughput bar is the one condition both uses share."""
    decision = tpu_probe.evaluate(
        _record(aggregate=1.5 * T4, workers=8, first=601.0, steady=1.0)
    )
    assert decision["verdict"] == "DO-NOT-PORT"
    assert decision["throughput_ok"] is False
    assert decision["recompile_overhead_ok"] is False
    assert "below the declared" in decision["why"] and "recompiles" in decision["why"]


def test_the_boundary_is_inclusive_on_both_conditions() -> None:
    """Exactly at the declared threshold ports; the rule is a bar, not an aspiration."""
    # Choose a compile time that lands the overhead exactly on the limit at this rate.
    aggregate = tpu_probe.PORT_THRESHOLD_MULTIPLE * T4
    decision = tpu_probe.evaluate(
        _record(aggregate=aggregate, workers=8, first=1.0, steady=1.0)
    )
    assert decision["verdict"] == "PORT"
    assert decision["recompile_overhead_ok"] is True


@pytest.mark.parametrize(
    "record",
    [
        {},
        {"independent_seeds": {"workers": 8, "aggregate_tokens_per_second": 40_000.0}},
        {"bf16": {"rebuild": {"first_step_seconds_including_compile": 5.0,
                              "steady_step_seconds_mean": 1.0}}},
        _record(aggregate=3 * T4, workers=0, first=31.0, steady=1.0),
        _record(aggregate=3 * T4, workers=8, first=31.0, steady=0),
    ],
)
def test_an_incomplete_record_is_never_a_pass(record: dict) -> None:
    """The reading that would be wrong if it defaulted the other way."""
    decision = tpu_probe.evaluate(record)
    assert decision["verdict"] == "INCONCLUSIVE"
    assert "missing" in decision["why"]
    assert "throughput_ok" not in decision


def test_a_record_without_the_parallel_measurement_is_recorded_as_such(tmp_path: Path) -> None:
    """A probe that ran but skipped its eight-worker pass cannot decide, and says so."""
    path = tmp_path / "partial.json"
    path.write_text(
        json.dumps(
            {
                "device": {"xla_devices": 8},
                "fp32": {},
                "bf16": {
                    "rebuild": {
                        "first_step_seconds_including_compile": 31.0,
                        "steady_step_seconds_mean": 1.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    decision = tpu_probe.evaluate(json.loads(path.read_text()))
    assert decision["verdict"] == "INCONCLUSIVE"
    assert "independent_seeds" in decision["why"]


def _small_genome() -> ArchitectureGenome:
    """The declared shape in miniature: uniform expert counts, uniform active count."""
    return create_genome(
        15_000,
        hidden_size=64,
        vocab_size=128,
        depth=3,
        ffn="moe",
        expert_count=4,
        active_experts=2,
    )


def test_the_rebuild_grows_one_expert_with_the_projects_own_operator() -> None:
    genome = _small_genome()
    before = [len(layer.experts) for layer in genome.layers]

    rebuilt, expert_id, layer_id = tpu_probe._rebuild_genome(genome)

    assert layer_id == rebuilt.layers[-1].id
    assert expert_id in {expert.id for expert in rebuilt.layers[-1].experts}
    assert [len(layer.experts) for layer in rebuilt.layers] == [
        *before[:-1],
        before[-1] + 1,
    ]
    # The caller's genome is the declared one and is reused for the other dtype.
    assert [len(layer.experts) for layer in genome.layers] == before


def test_the_rebuild_is_buildable_because_active_counts_stay_uniform() -> None:
    """The invariant whose loss killed the probe's first run.

    Measured 2026-09-21: the probe raised ``ValueError: V0 MoE reconstruction requires
    uniform active expert counts`` 148 s in and wrote no record, so 6.5 h of Kaggle queue
    produced no number. The rebuild the project's own event performs keeps that uniform.
    """
    rebuilt, _, _ = tpu_probe._rebuild_genome(_small_genome())

    active = {layer.active_experts for layer in rebuilt.layers if layer.enabled}
    assert len(active) == 1
    config = v0_moe_config_from_genome(rebuilt)  # must not raise
    assert config.experts_per_layer[-1] == 5
    assert config.active_experts == 2


def test_the_declared_genome_rebuilds_at_the_pre_registered_shape() -> None:
    """The exact call that raised on Kaggle, at the shape the probe really uses.

    Measured locally 2026-09-21: the declared genome is 12 layers of 8 experts with 2
    active, and the rebuild is 12 layers of 8 with the last holding 9 - config and model
    both build, at 287,434,752 parameters. The small-genome test pins the invariant; this
    pins it where the probe runs, because a shape-dependent guard would only appear at
    declared scale - which is the scale the probe dies at, and the queue is 6.5 h.
    """
    from act.training.v0_session import _genome_from_config, declared_session_config

    genome = _genome_from_config(declared_session_config())
    rebuilt, _, layer_id = tpu_probe._rebuild_genome(genome)

    assert layer_id == genome.layers[-1].id
    config = v0_moe_config_from_genome(rebuilt)  # must not raise
    assert config.depth == len(genome.layers)
    assert config.experts_per_layer[-1] == config.experts_per_layer[0] + 1
    assert config.active_experts == 2


def test_the_mutation_that_broke_the_probe_validates_but_cannot_be_built() -> None:
    """Why the rebuild uses the operator instead of editing the genome's dict itself.

    Pins the gap rather than a preference: a per-layer ``active_experts`` is valid to the
    genome validator, and the V0 MoE config holds one scalar for every layer, so a
    hand-rolled expert *and* active-expert bump is a genome the project can hold and
    cannot build. If this ever stops raising, the probe's justification is worth revisiting.
    """
    mutated = _small_genome().to_dict()
    last = mutated["layers"][-1]
    last["experts"].append({"id": "L2-probe", "width": 64, "birth_step": 0})
    last["active_experts"] += 1
    rebuilt = ArchitectureGenome.from_dict(mutated)

    assert validate_genome(rebuilt).valid
    with pytest.raises(ValueError, match="uniform active expert counts"):
        v0_moe_config_from_genome(rebuilt)


def test_a_partial_record_is_written_where_it_stands(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "tpu-probe.json"
    tpu_probe._save({"completed": ["fp32.throughput"]}, target)
    assert json.loads(target.read_text())["completed"] == ["fp32.throughput"]


def test_the_measurement_order_is_the_decisions() -> None:
    """bf16 and the aggregate first, the fp32 diagnostic last.

    ``evaluate`` reads ``bf16`` and ``independent_seeds`` (pinned against this order by the
    test below, so the two cannot drift apart). A probe is queued for hours and runs once,
    so the order is what a *partial* record still contains - and on 2026-09-22 the run
    spent itself on fp32 and delivered neither of the measurements the rule needs.
    """
    assert tpu_probe.phase_names(("bf16", "fp32"), workers=8) == (
        "bf16.throughput",
        "bf16.rebuild",
        "independent_seeds",
        "fp32.throughput",
        "fp32.rebuild",
    )
    # Without the parallel pass there is nothing to put between the dtypes.
    assert tpu_probe.phase_names(("bf16", "fp32"), workers=0) == (
        "bf16.throughput",
        "bf16.rebuild",
        "fp32.throughput",
        "fp32.rebuild",
    )


def test_the_dtype_the_decision_reads_is_the_one_measured_first() -> None:
    """The coupling itself, stated as one fact rather than two literals that agree.

    ``evaluate`` names the fields a port depends on - a record missing one is
    INCONCLUSIVE, never a pass - and the measurement order runs that dtype first. They were
    *not* coupled on 2026-09-22: the order measured fp32 first, the rule needed bf16, and
    the one queued run produced neither.
    """
    why = tpu_probe.evaluate({})["why"]
    first = tpu_probe.MEASUREMENT_ORDER[0]
    assert first == "bf16"
    assert f"{first}.rebuild" in why
    assert tpu_probe.phase_names(tpu_probe.MEASUREMENT_ORDER, workers=8)[0] == (
        f"{first}.throughput"
    )


def _rank_row(
    rank: int,
    *,
    started: float,
    finished: float,
    tokens_per_step: int = 4096,
    steady_steps: int = 11,
    steps: int = 12,
    device: str | None = None,
) -> dict:
    """One rank's report, with only the fields the aggregate reads."""
    named = device or f"xla:{rank}"
    return {
        "rank": rank,
        "seed": 15_000 + rank,
        "device": named,
        # A bound rank: its thread is on the device it was given (the parent checks these two
        # against each other), and the process default the binding replaced is recorded.
        "thread_device_before": "xla:0",
        "thread_device_after": named,
        "steps": steps,
        "steady_steps": steady_steps,
        "tokens_per_step": tokens_per_step,
        "steady_started": started,
        "steady_finished": finished,
    }


def _fake_ranks(rows: dict[int, dict], recorded: list[dict]):
    """A ``run_rank`` that writes each rank's report instead of training a seed.

    It takes ``torch_xla`` and ignores it: this stand-in writes a report in the parent's own
    process, so it never steps and has no rank thread to bind. The binding itself is pinned
    by ``test_binding_a_thread_names_its_chip...`` and by the real-``_seed_rank`` test below.
    """

    def run_rank(
        *, rank, device, config, steps, dtype, directory, ready, torch, xm, torch_xla=None
    ):
        recorded.append(
            {
                "rank": rank,
                "device": str(device),
                "steps": steps,
                "dtype": dtype,
                "directory": str(directory),
                "seed": config.seed,
                "barrier": ready,
            }
        )
        # A rank with no row writes nothing: the reports a failed rank leaves behind are the
        # case the parent refuses rather than the case this fake builds.
        if rank in rows:
            (Path(directory) / f"rank-{rank}.json").write_text(
                json.dumps(rows[rank], sort_keys=True), encoding="utf-8"
            )

    return run_rank


#: A shape a CPU test can afford: the declared session's fields, none of its size.
_TINY_CONFIG = None


def _tiny_config():
    """Built directly rather than through ``declared_session_config``, which validates.

    The declared constructor refuses anything but the pre-registered shape (the protocol's
    tolerance band is derived from its exact parameter count), so a reduced-fidelity test
    config is constructed from the dataclass - which is what the model's own tests do.
    """
    global _TINY_CONFIG
    if _TINY_CONFIG is None:
        from act.training.v0_session import V0SessionConfig

        _TINY_CONFIG = V0SessionConfig(
            seed=7,
            vocab_size=128,
            context_size=16,
            hidden_size=64,
            depth=2,
            attention_heads=1,
            expert_count=4,
            active_experts=2,
            expert_width=64,
            batch_size=1,
            total_steps=3,
            learning_rate=1e-3,
        )
    return _TINY_CONFIG


def test_the_aggregate_is_the_window_all_eight_ranks_stepped_in(tmp_path: Path) -> None:
    """One independent seed per device, in one process, read off the window they share.

    The aggregate is not the sum of eight rates each rank measured for itself: a step
    marker may sync more than the rank that called it, and a sum would then count the same
    device time once per rank. The window all ranks were stepping in is the quantity that
    survives that, and it is what the campaign's aggregate is.
    """
    recorded: list[dict] = []
    rows = {
        rank: _rank_row(rank, started=1000.0 + rank * 0.001, finished=1050.0 + rank * 0.001)
        for rank in range(8)
    }
    block = tpu_probe.measure_independent_seeds(
        torch=torch, xm=object(), devices=8, workers=8, steps=12, dtype="bf16",
        directory=tmp_path, run_rank=_fake_ranks(rows, recorded),
    )

    # The window: the earliest steady start to the latest steady finish.
    window = (1050.0 + 7 * 0.001) - 1000.0
    assert block["workers"] == 8 and block["completed"] == 8
    assert block["steady_window_seconds"] == pytest.approx(window, abs=1e-3)
    assert block["steady_steps_per_rank"] == 11
    assert block["steady_tokens"] == 8 * 11 * 4096
    assert block["aggregate_tokens_per_second"] == pytest.approx(
        8 * 11 * 4096 / window, rel=1e-3
    )
    assert block["ratio_to_t4_pair"] == pytest.approx(
        block["aggregate_tokens_per_second"] / T4, rel=1e-2
    )
    # The mechanism, named beside the number: this is not the process-based aggregate, and a
    # reader has to be able to tell them apart.
    assert block["taken_in"] == "single-process-thread-per-device"
    assert "bound to its own device" in block["rank_threads"]
    assert "one thread per device" in block["aggregate_definition"]
    assert "process RNG is shared" in block["seed_note"]
    # Every rank was told its own ordinal and its own device: the rank *is* the device, which
    # is what makes the distinct-device check below a real one rather than a formality.
    assert [entry["rank"] for entry in recorded] == list(range(8))
    assert [entry["device"] for entry in recorded] == [f"xla:{r}" for r in range(8)]
    assert [entry["seed"] for entry in recorded] == [15_000 + r for r in range(8)]
    assert {entry["dtype"] for entry in recorded} == {"bf16"}
    assert {entry["steps"] for entry in recorded} == {12}
    # One barrier for all eight: the ranks meet at the same object, or they do not meet.
    assert all(entry["barrier"] is recorded[0]["barrier"] for entry in recorded)


def test_the_barrier_sits_between_the_compile_step_and_the_steady_steps() -> None:
    """``after_first`` runs exactly once, after the first step - the aggregate's window.

    A hook that ran before the first step, or on every step, would put a compile inside the
    measured window or close it early. The compile is the one cost the aggregate means to
    exclude, so where the hook fires is the measurement.
    """
    calls: list[str] = []

    class _Model:
        def loss(self, inputs):  # noqa: ANN001
            return torch.zeros((), requires_grad=True)

    class _Xm:
        def mark_step(self):  # noqa: ANN201
            calls.append("step")

    measured = tpu_probe._measure(
        torch,
        _Xm(),
        _Model(),
        torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.0),
        _tiny_config(),
        steps=3,
        device=torch.device("cpu"),
        after_first=lambda: calls.append("hook"),
    )

    assert calls == ["step", "hook", "step", "step"]
    assert measured["steps"] == 3
    assert measured["tokens_per_step"] == 16


def test_a_rank_that_waits_out_the_barrier_names_the_wait(monkeypatch) -> None:
    """A bare ``BrokenBarrierError`` names neither the rank, the phase, nor the budget.

    Four attempts at this measurement were each diagnosed from a line that did not name
    itself, so the wait is wrapped where it happens: a rank that failed before compiling is
    the usual cause, and it is usually visible in the other ranks' reports.
    """
    monkeypatch.setenv("TPU_PROBE_SEEDS_TIMEOUT", "0.05")
    # The type, not just the message: it is what the parent separates a *cause* from ranks
    # that merely stopped waiting for it (see the test below).
    with pytest.raises(tpu_probe.RankWaitTimeout, match="rank 1 waited"):
        tpu_probe._wait_for_ranks(threading.Barrier(2), rank=1)


def test_one_dead_rank_costs_seconds_not_the_whole_barrier_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rank that dies must not leave the others waiting out an hour of billed TPU time.

    The barrier's budget is the phase's clock (an hour by default), and the ranks are exactly
    the case where waiting in order hides the cause: rank 1 fails, ranks 0 and 2 sit at the
    barrier until the budget runs out, and the phase then reports a broken barrier instead of
    rank 1's error. So the parent awaits with ``FIRST_EXCEPTION`` and aborts the barrier as
    soon as one rank raises, and the failure it raises names the cause and the collateral
    separately. The budget is shortened here so a regression fails this test rather than
    stalling the suite.
    """
    monkeypatch.setenv("TPU_PROBE_SEEDS_TIMEOUT", "60")

    def run_rank(
        *, rank, device, config, steps, dtype, directory, ready, torch, xm, torch_xla=None
    ):
        if rank == 1:
            raise RuntimeError("RESOURCE_EXHAUSTED: no memory for this rank's model")
        tpu_probe._wait_for_ranks(ready, rank=rank)

    started = time.perf_counter()
    with pytest.raises(RuntimeError) as raised:
        tpu_probe.measure_independent_seeds(
            torch=torch, xm=object(), devices=None, workers=3, steps=3, dtype="fp32",
            directory=tmp_path,
            config_for=lambda rank: _tiny_config(),
            device_for=lambda rank: torch.device("cpu", rank),
            run_rank=run_rank,
        )
    elapsed = time.perf_counter() - started

    assert elapsed < 10, "the budget was waited out instead of the rank's own error"
    message = str(raised.value)
    assert "the aggregate did not complete" in message
    assert "rank 1 raised RuntimeError: RESOURCE_EXHAUSTED" in message
    assert "rank(s) [0, 2] stopped at the barrier" in message


@pytest.mark.parametrize("workers", [0, -1])
def test_the_plan_refuses_an_aggregate_of_no_ranks(workers: int) -> None:
    """``--workers 0`` means "skip the parallel pass" in ``main``; as a plan it is nothing."""
    with pytest.raises(ValueError, match="at least one rank"):
        tpu_probe.parallel_plan(workers, 12, "bf16", "/tmp/x")


def test_a_host_with_four_devices_cannot_measure_eight_ranks(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="4 XLA device"):
        tpu_probe.measure_independent_seeds(
            torch=torch, xm=object(), devices=4, workers=8, steps=12, dtype="bf16",
            directory=tmp_path,
            run_rank=_fake_ranks({}, []),
        )


def test_a_rank_that_wrote_no_report_is_an_error_not_an_average(tmp_path: Path) -> None:
    rows = {rank: _rank_row(rank, started=1.0, finished=2.0) for rank in range(7)}
    with pytest.raises(RuntimeError, match="rank 7 of 8 wrote no report"):
        tpu_probe.measure_independent_seeds(
            torch=torch, xm=object(), devices=None, workers=8, steps=12, dtype="bf16",
            directory=tmp_path, run_rank=_fake_ranks(rows, []),
        )


def test_the_ranks_have_to_be_on_distinct_devices(tmp_path: Path) -> None:
    """The claim the aggregate carries: one independent seed per *device*.

    Eight reports that all name ``xla:0`` are one device measured eight times, and the
    arithmetic would happily return eight times its rate. So the reports are checked - a
    property the process-based version could not state at all, because its ranks' devices
    were whatever the runtime's default was and nothing recorded them.
    """
    rows = {
        rank: _rank_row(rank, started=1.0, finished=2.0, device="xla:0")
        for rank in range(8)
    }
    with pytest.raises(RuntimeError, match="named 1 device"):
        tpu_probe.measure_independent_seeds(
            torch=torch, xm=object(), devices=None, workers=8, steps=12, dtype="bf16",
            directory=tmp_path, run_rank=_fake_ranks(rows, []),
        )


def test_binding_a_thread_names_its_chip_and_reports_the_default_it_replaced() -> None:
    """The property the one-process aggregate rests on, and that v6 is the first to have.

    ``xm.mark_step()`` is ``torch_xla.sync()`` in torch_xla 2.8, and that marks the step on
    the *calling thread's* current device - for a fresh thread, the process default, which
    every earlier phase of the probe leaves at ``xla:0``. A rank that never binds therefore
    steps chip 0 while its own tensors sit on another chip. Both values are returned so the
    parent can check the second against the device the rank was given.
    """

    class _TorchXla:
        def __init__(self) -> None:
            self.calls: list[int | None] = []

        def device(self, index=None):  # noqa: ANN001, ANN201
            self.calls.append(index)
            return torch.device("xla", 0 if index is None else index)

    xla = _TorchXla()
    assert tpu_probe.bind_thread_device(xla, torch.device("xla", 3)) == ("xla:0", "xla:3")
    assert xla.calls == [None, 3]


def test_binding_refuses_a_device_without_an_ordinal() -> None:
    """An ordinal-less device would make the binding a read of the process default."""

    class _TorchXla:
        def device(self, index=None):  # noqa: ANN001, ANN201
            return torch.device("xla")

    with pytest.raises(RuntimeError, match="names no device ordinal"):
        tpu_probe.bind_thread_device(_TorchXla(), torch.device("xla"))


def test_a_rank_whose_thread_is_not_bound_is_not_averaged_in(tmp_path: Path) -> None:
    """A rank that stepped the wrong chip has a fast, wrong window - so it is refused.

    This is the failure mode the binding exists for, and it does not look like a failure:
    seven ranks that mark chip 0's steps compile their graphs and return quickly, and the
    aggregate would read as a *high* number for the wrong hardware.
    """
    rows = {rank: _rank_row(rank, started=1.0, finished=2.0) for rank in range(8)}
    rows[5]["thread_device_after"] = "xla:0"
    with pytest.raises(RuntimeError, match="rank 5 was given xla:5 but runs with xla:0"):
        tpu_probe.measure_independent_seeds(
            torch=torch, xm=object(), devices=None, workers=8, steps=12, dtype="bf16",
            directory=tmp_path, run_rank=_fake_ranks(rows, []),
        )


def test_the_default_rank_path_refuses_without_the_module_that_binds_it(
    tmp_path: Path,
) -> None:
    """The module is required because the binding is not optional on the default path."""
    with pytest.raises(RuntimeError, match="needs the torch_xla module"):
        tpu_probe.measure_independent_seeds(
            torch=torch, xm=object(), devices=None, workers=8, steps=12, dtype="bf16",
            directory=tmp_path,
        )


def test_ranks_that_disagree_on_the_shape_are_not_one_measurement(tmp_path: Path) -> None:
    """A rank on a different shape makes the aggregate the sum of two measurements."""
    rows = {rank: _rank_row(rank, started=1.0, finished=2.0) for rank in range(8)}
    rows[3] = _rank_row(3, started=1.0, finished=2.0, tokens_per_step=2048)
    with pytest.raises(RuntimeError, match="did not measure one shape"):
        tpu_probe.measure_independent_seeds(
            torch=torch, xm=object(), devices=None, workers=8, steps=12, dtype="bf16",
            directory=tmp_path, run_rank=_fake_ranks(rows, []),
        )
    rows = {rank: _rank_row(rank, started=1.0, finished=2.0) for rank in range(8)}
    rows[5] = _rank_row(5, started=1.0, finished=2.0, steady_steps=4)
    with pytest.raises(RuntimeError, match="did not measure one shape"):
        tpu_probe.measure_independent_seeds(
            torch=torch, xm=object(), devices=None, workers=8, steps=12, dtype="bf16",
            directory=tmp_path, run_rank=_fake_ranks(rows, []),
        )


def test_a_window_with_no_steady_step_measures_the_compile(tmp_path: Path) -> None:
    """``--steps 1`` would leave the window holding nothing but the compile."""
    rows = {
        rank: _rank_row(rank, started=1.0, finished=2.0, steady_steps=0)
        for rank in range(8)
    }
    with pytest.raises(RuntimeError, match="no steady step"):
        tpu_probe.measure_independent_seeds(
            torch=torch, xm=object(), devices=None, workers=8, steps=1, dtype="bf16",
            directory=tmp_path, run_rank=_fake_ranks(rows, []),
        )


def test_a_rank_records_the_device_it_ran_on_and_the_window_it_stepped_in(
    tmp_path: Path,
) -> None:
    """The worker's own contract, run for real on CPU at a reduced-fidelity shape.

    The report is what the aggregate reads, so its fields are pinned where they are written
    rather than where they are read: the device, the seed, the two window stamps and the
    steady-step count that says how many steps the window holds. ``xm`` is a stand-in - a
    CPU device and a step marker that records - because torch_xla is not installed here;
    everything else is the probe's own path, model included.
    """
    steps: list[float] = []

    class _Xm:
        def xla_device(self):  # noqa: ANN201
            return torch.device("cpu:0")

        def mark_step(self):  # noqa: ANN201
            steps.append(time.perf_counter())

    class _TorchXla:
        """``torch_xla.device`` for this thread: it reports the current one, then binds."""

        def __init__(self) -> None:
            self.current = "cpu:0"
            self.calls: list[int | None] = []

        def device(self, index=None):  # noqa: ANN001, ANN201
            self.calls.append(index)
            if index is not None:
                self.current = f"cpu:{index}"
            return torch.device(self.current)

    xla = _TorchXla()
    config = _tiny_config()
    directory = tmp_path / "ranks"
    directory.mkdir()
    tpu_probe._seed_rank(
        rank=0,
        device=torch.device("cpu", 0),
        config=config,
        steps=3,
        dtype="fp32",
        directory=directory,
        ready=threading.Barrier(1),
        torch=torch,
        torch_xla=xla,
        xm=_Xm(),
    )

    row = json.loads((directory / "rank-0.json").read_text(encoding="utf-8"))
    assert row["rank"] == 0
    assert row["seed"] == config.seed == 7
    assert row["device"] == "cpu:0"
    # The binding, as the thread performed it: read the current device, then name the
    # ordinal it was given - and the report carries both, which is what the parent checks.
    assert xla.calls == [None, 0]
    assert row["thread_device_before"] == "cpu:0"
    assert row["thread_device_after"] == row["device"]
    assert row["steps"] == 3 and row["steady_steps"] == 2
    assert row["steady_finished"] > row["steady_started"] > 0
    assert row["first_step_seconds_including_compile"] > 0
    assert row["steady_step_seconds_mean"] > 0
    assert row["tokens_per_step"] == config.batch_size * config.context_size
    assert len(steps) == 3  # one step marker per step, the hook included


def test_the_aggregate_runs_for_real_with_eight_rank_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The concurrency the arithmetic tests stand in for: a pool, a barrier, one window.

    ``run_rank`` fakes pin the arithmetic and every refusal, and by construction they cannot
    pin what runs them - eight threads, one barrier all eight must reach, reports written
    from inside them. torch_xla is not installed here either, so the module that binds each
    thread is a stand-in; what matters is that the *wiring* is the real one, and each rank's
    report carries the device its thread ended up on, so a binding that was skipped or given
    the wrong ordinal shows up here rather than only on a TPU. The barrier's budget is
    brought down from its hour so a broken path fails the suite instead of stalling it.
    """
    monkeypatch.setenv("TPU_PROBE_SEEDS_TIMEOUT", "60")

    class _TorchXla:
        def device(self, index=None):  # noqa: ANN001, ANN201
            return torch.device("cpu", 0 if index is None else index)

    class _Xm:
        def xla_device(self):  # noqa: ANN201
            return torch.device("cpu", 0)

        def mark_step(self):  # noqa: ANN201
            pass

    tiny = _tiny_config()
    block = tpu_probe.measure_independent_seeds(
        torch=torch,
        xm=_Xm(),
        torch_xla=_TorchXla(),
        devices=8,
        workers=8,
        steps=3,
        dtype="fp32",
        directory=tmp_path,
        config_for=lambda rank: dataclasses.replace(tiny, seed=15_000 + rank),
        device_for=lambda rank: torch.device("cpu", rank),
    )

    assert block["completed"] == 8
    assert block["steady_steps_per_rank"] == 2
    assert block["aggregate_tokens_per_second"] > 0
    assert {row["device"] for row in block["per_rank"]} == {f"cpu:{r}" for r in range(8)}
    for row in block["per_rank"]:
        # The binding on the real path, per rank: the thread was on the process default
        # first, and the parent's check is what makes "its own device" a fact of the record.
        assert row["thread_device_before"] == "cpu:0"
        assert row["thread_device_after"] == row["device"]


class _FakeXm:
    def get_xla_supported_devices(self):  # noqa: ANN201
        return [f"xla:{rank}" for rank in range(8)]

    def xla_device(self):  # noqa: ANN201
        return "xla:0"


class _FakeXr:
    def memory_info(self, device):  # noqa: ANN001, ANN201
        return {"bytes_used": None, "bytes_limit": 0}


def test_the_hardware_report_carries_the_platform_topology_environment(monkeypatch) -> None:
    """The values the v5 failure was a disagreement about, in the record this time.

    ``tpu.configure_topology`` writes its process grid and its worker-address list with
    ``os.environ.setdefault``, so a value the platform set first wins over the one torch_xla
    computes - and the record that failed to spawn said only "Expected 8 worker addresses,
    got 1". These are the keys that decide it, recorded with ``None`` for unset so a reader
    can tell a missing value from a missing key.
    """
    monkeypatch.setenv("TPU_PROCESS_BOUNDS", "2,2,2")
    monkeypatch.setenv("TPU_PROCESS_ADDRESSES", "localhost:8476")

    reported = tpu_probe.environment_report()
    assert reported["TPU_PROCESS_BOUNDS"] == "2,2,2"
    assert reported["TPU_PROCESS_ADDRESSES"] == "localhost:8476"
    assert reported["TPU_WORKER_HOSTNAMES"] is None  # collected, and unset
    assert {"PJRT_DEVICE", "TPU_VISIBLE_CHIPS"} <= set(tpu_probe.TOPOLOGY_ENV_KEYS)

    report = tpu_probe._device_report(
        types.SimpleNamespace(__version__="2.8.0+test"), _FakeXm(), _FakeXr()
    )
    assert report["pytorch"] == "2.8.0+test"
    assert report["xla_devices"] == 8
    assert report["device_kind"] == "xla:0"
    assert report["environment"]["TPU_PROCESS_BOUNDS"] == "2,2,2"
    assert report["memory_bytes_limit"] is None


def _fake_torch_xla(monkeypatch, *, chips, metadata) -> None:
    """A synthetic ``torch_xla`` whose TPU internals answer without a TPU."""
    fake_tpu = types.ModuleType("torch_xla._internal.tpu")
    fake_tpu.num_available_chips = chips
    fake_tpu.get_tpu_env = metadata
    internal = types.ModuleType("torch_xla._internal")
    internal.tpu = fake_tpu
    root = types.ModuleType("torch_xla")
    root._internal = internal
    monkeypatch.setitem(sys.modules, "torch_xla", root)
    monkeypatch.setitem(sys.modules, "torch_xla._internal", internal)
    monkeypatch.setitem(sys.modules, "torch_xla._internal.tpu", fake_tpu)


def test_the_tpu_report_records_the_chip_count_and_the_metadata(monkeypatch) -> None:
    """The two inputs ``num_local_processes`` and ``configure_topology`` read.

    Neither was in any record the process-based attempts wrote, which is why the v5 failure
    had to be reconstructed from a kernel log. Recorded here, the next reader gets them.
    """
    _fake_torch_xla(
        monkeypatch,
        chips=lambda: 8,
        metadata=lambda: {"ACCELERATOR_TYPE": "v3-8", "WORKER_ID": 0},
    )
    assert tpu_probe._tpu_report() == {
        "pci_chips": 8,
        "metadata": {"ACCELERATOR_TYPE": "v3-8", "WORKER_ID": "0"},
    }


def test_a_tpu_report_that_cannot_read_the_platform_says_so(monkeypatch) -> None:
    """Diagnostics for a decision: an unreadable metadata server must not stop a probe.

    The failure is recorded under its own name rather than raised, because the measurement
    this report rides along with is the one that decides - and a record that says
    ``EnvironmentError`` is a record a reader can act on.
    """

    def _unreachable():
        raise EnvironmentError("Failed to get TPU metadata")

    _fake_torch_xla(monkeypatch, chips=lambda: 0, metadata=_unreachable)
    report = tpu_probe._tpu_report()
    assert report["pci_chips"] == 0
    # Recorded under its own name (``EnvironmentError`` is ``OSError`` in Python 3, so the
    # name is read off the exception rather than assumed), with the message attached.
    assert "Failed to get TPU metadata" in str(report["metadata"])
    assert str(report["metadata"]).split(":")[0].endswith("Error")


def test_a_phase_that_cannot_run_is_recorded_and_the_rest_still_run(tmp_path: Path) -> None:
    """The containment the 2026-09-22 run needed and did not have.

    That probe measured fp32's throughput, then died inside fp32's rebuild and took bf16,
    the eight-worker aggregate and its own exit status with it. A failure is now a row in
    the record: it stays out of ``completed``, names itself, and the measurements after it
    still happen - which matters because a failure *is* often a memory verdict, and the
    numbers after this one are the ones the decision reads.
    """
    path = tmp_path / "tpu-probe.json"
    record: dict = {"completed": [], "errors": []}
    ran: list[str] = []

    def exhausted() -> None:
        raise RuntimeError(
            "Bad StatusOr access: RESOURCE_EXHAUSTED: Attempting to reserve 7.96G"
        )

    def later() -> None:
        ran.append("independent_seeds")

    assert tpu_probe._run_phase(record, "fp32.rebuild", exhausted, path) is False
    assert tpu_probe._run_phase(record, "independent_seeds", later, path) is True

    assert record["completed"] == ["independent_seeds"]
    assert len(record["errors"]) == 1
    failure = record["errors"][0]
    assert failure["phase"] == "fp32.rebuild"
    assert failure["error"] == (
        "RuntimeError: Bad StatusOr access: RESOURCE_EXHAUSTED: "
        "Attempting to reserve 7.96G"
    )
    # The message is the symptom; the frames are the site. Pinned because the message
    # alone was ambiguous enough to cost a day on 2026-09-29, and because it is the one
    # thing a failure has to carry when the run that produced it is queued once.
    assert "RuntimeError: Bad StatusOr access" in failure["traceback"]
    assert "exhausted" in failure["traceback"]
    assert ran == ["independent_seeds"]


def test_the_record_reaches_disk_before_the_next_measurement_runs(tmp_path: Path) -> None:
    """The first run wrote its record once, at the end, and died 148 s in.

    So 6.5 h of queue produced no number at all - not even the throughput it had already
    measured. This pins the *timing* rather than the existence of a call - the following
    phase reads the file and sees what the previous one left - and it pins it for a phase
    that **failed** as well as one that completed, because a dying run is the case this
    exists for. A version that saved only on success passed an earlier version of this
    test, which is why the failing branch is asserted here.
    """
    path = tmp_path / "tpu-probe.json"
    record: dict = {"completed": [], "errors": []}
    on_disk: list[object] = []

    def read_it_back() -> None:  # stands in for the next measurement
        on_disk.append(json.loads(path.read_text()))

    def exhausted() -> None:
        raise RuntimeError("RESOURCE_EXHAUSTED")

    tpu_probe._run_phase(record, "bf16.throughput", lambda: None, path)
    tpu_probe._run_phase(record, "bf16.rebuild", read_it_back, path)
    tpu_probe._run_phase(record, "fp32.throughput", exhausted, path)

    assert on_disk == [{"completed": ["bf16.throughput"], "errors": []}]
    failure = json.loads(path.read_text())["errors"][0]
    assert failure["phase"] == "fp32.throughput"
    assert failure["error"] == "RuntimeError: RESOURCE_EXHAUSTED"
    assert "RuntimeError: RESOURCE_EXHAUSTED" in failure["traceback"]


def test_each_phase_is_bound_to_its_own_dtype_and_state() -> None:
    """The failure mode a loop-built table hides: every phase running as the last dtype.

    A closure reading ``dtype`` or ``state`` at call time writes the same measurements
    under two names, and the record then **reads as complete and is wrong** - which is
    worse than a crash, because nothing downstream can tell. So the binding is asserted
    rather than eyeballed, and so is the *sharing* it depends on: a rebuild is buildable
    only because its own dtype's throughput pass left a genome in that dtype's state, so a
    table that gave each phase a fresh dict would turn every rebuild into a recorded error.
    """
    calls: list[tuple[str, int]] = []

    def build_and_time(dtype: str, state: dict) -> None:
        calls.append((f"build:{dtype}", id(state)))
        state["genome"] = f"{dtype}-genome"

    def rebuild_of(dtype: str, state: dict) -> None:
        calls.append((f"rebuild:{dtype}", id(state)))
        # Raises for real if the throughput pass did not run against this same dict.
        calls.append((f"from:{state['genome']}", id(state)))

    states, work = tpu_probe.phase_table(("bf16", "fp32"), build_and_time, rebuild_of)

    assert set(work) == {"bf16.throughput", "bf16.rebuild", "fp32.throughput", "fp32.rebuild"}
    assert states["bf16"] is not states["fp32"]
    for name in ("bf16.throughput", "bf16.rebuild", "fp32.throughput", "fp32.rebuild"):
        work[name]()

    assert [name for name, _ in calls] == [
        "build:bf16", "rebuild:bf16", "from:bf16-genome",
        "build:fp32", "rebuild:fp32", "from:fp32-genome",
    ]
    # Each dtype's two phases share one dict; the two dtypes share nothing.
    assert {state for _, state in calls[:3]} == {id(states["bf16"])}
    assert {state for _, state in calls[3:]} == {id(states["fp32"])}


def test_the_fetch_script_decides_where_the_record_lands() -> None:
    """A probe is queued for hours, runs once, and exists for its verdict.

    So the verdict is printed by the fetch that delivers the record, not left for a second
    hand-run command that can be forgotten. Both terminal branches apply it, which is the
    point of the containment fix: a run that ends ERROR still writes a record, and a record
    that is incomplete is `INCONCLUSIVE` with exit 3 - a legitimate answer to print rather
    than a reason to say nothing.

    (This is why the script is `fetch_*` rather than `run_*`: the `run_*_when_*` watchers
    run the arch-zoo analyzer and their logs are their arms' records, and
    `test_record_freshness.py` requires an analyzer call of every one of them.)
    """
    text = (ROOT / "scripts" / "fetch_tpu_probe_when_done.sh").read_text(encoding="utf-8")
    assert "scripts/tpu_probe.py --evaluate" in text
    complete_branch, error_branch = text.split("*COMPLETE*)")[1].split(
        "*ERROR* | *CANCELLED*)"
    )
    assert "evaluate_record" in complete_branch
    assert "evaluate_record" in error_branch
    # Best-effort by construction: no interpreter must not undo a fetch already made.
    assert ") || true" in text


def test_the_push_waits_for_a_gap_with_the_guards_own_answer() -> None:
    """The v6 push was refused by the probe's own guard, so the wait is an instrument.

    Measured 2026-09-26: ``push_tpu_probe.py`` refused to republish the code dataset because
    ``act-t4-session`` was RUNNING - correctly, since a session's body and its mounted
    ``src/`` have to be one revision - which leaves the push waiting for a gap nobody is
    watching for. Two properties keep the waiter from being a way to break the campaign. **It
    asks the guard's own function** whether a session is live: two copies of that answer is
    how a publish lands inside a live mount, which is what cost the 2026-09-21 session. And
    **it publishes nothing itself** - it runs the script that re-checks the campaign at its
    own push time, because the gap it waited for can close in the seconds between a poll and
    the publish.
    """
    text = (ROOT / "scripts" / "push_tpu_probe_when_idle.sh").read_text(encoding="utf-8")
    assert "push_tpu_probe.live_campaign_kernel" in text
    assert "scripts/push_tpu_probe.py" in text
    # The dataset is published by that script and by nothing else here.
    assert "ensure_code_dataset" not in text
    # And the measurement is collected without a second hand-run: the fetch watcher.
    assert "fetch_tpu_probe_when_done.sh" in text
    # Bounded, because a queue can hold the driver's lock for hours.
    assert "WAIT_POLLS" in text and "gave up" in text
    # And a failed push says *which* failure it was: the log is the only account of why the
    # probe did not go out, so it must not report a platform failure as a resumed campaign.
    assert "the push exited" in text and "5 is the guard" in text


def test_the_declared_model_is_released_before_any_rebuild_is_measured() -> None:
    """The memory shape that killed the 2026-09-22 run, pinned where it can be seen.

    That run died with 7.96G wanted and 5.33G free, holding two fp32 models plus two AdamW
    states at once - because it measured the rebuild before releasing the declared model.
    A session's own event peak is narrower: ``v0_session._adaptive_event`` holds the
    previous model while it rebuilds the winner, but only one optimizer's state is ever
    stepped, so what an event costs on top of one session is a model's parameters and
    grads. Releasing the pair after its throughput is timed keeps that true, and bf16 is
    what the decision reads either way - so nothing this probe is read for depends on the
    two coexisting.

    Pinned as source order rather than behaviour: releasing device memory has no CPU
    stand-in, and the cost of a silent revert is another queued run that dies where this
    one did.
    """
    source = (ROOT / "scripts" / "tpu_probe.py").read_text(encoding="utf-8")
    body = source.split("def build_and_time", 1)[1].split("def rebuild_of", 1)[0]
    assert "del model, optimizer" in body
    assert body.index("del model, optimizer") > body.index("_measure(")
