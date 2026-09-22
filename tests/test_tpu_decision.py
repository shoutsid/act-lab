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

import json
import sys
from pathlib import Path

import pytest

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
    assert record["errors"] == [
        {
            "phase": "fp32.rebuild",
            "error": "RuntimeError: Bad StatusOr access: RESOURCE_EXHAUSTED: "
            "Attempting to reserve 7.96G",
        }
    ]
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
    assert json.loads(path.read_text())["errors"] == [
        {"phase": "fp32.throughput", "error": "RuntimeError: RESOURCE_EXHAUSTED"}
    ]


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
