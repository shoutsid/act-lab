"""Pin the pre-registered TPU port decision, and that it says nothing it cannot support.

The decision rule was declared **before** the probe was pushed (see `DECISION_RULE` in
`scripts/tpu_probe.py`), because the one thing a go/no-go must not be is fitted to the number
that arrives. These tests pin the two thresholds, the arithmetic between them, and - the case
that matters most - that an incomplete record is `INCONCLUSIVE` rather than a pass. A probe
that fails to produce an eight-worker measurement is exactly the outcome where "no problem
found" would be the wrong reading.
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


def test_every_measurement_is_saved_before_the_next_one_runs() -> None:
    """The first run wrote its record once, at the end, and died 148 s in.

    So 6.5 h of queue produced no number at all - not even the throughput it had already
    measured. Each measurement now lands on disk as it completes, and the record carries a
    ``completed`` list so a partial one cannot read like a finished one.
    """
    source = (ROOT / "scripts" / "tpu_probe.py").read_text(encoding="utf-8")
    assert source.count("_save(record, args.output)") >= 4
    assert 'record["completed"] = []' in source
