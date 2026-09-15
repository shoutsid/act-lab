"""The genome-extraction and fresh-initialization transport path (report 5.16).

Section 5.16 asks whether a topology discovered by the adaptive arm is still
advantageous when it is rebuilt from scratch, so three things must hold before
any delta can be read as an effect of the *structure*:

1. the discovered topology survives extraction (``ArmResult.genome`` round
   trips to an ``ArchitectureGenome`` with the same hash),
2. the transport procedure is faithful - transporting the unmutated ``fixed``
   genome reproduces that arm's own discovery run exactly, since it is the
   same architecture retrained on the same schedule,
3. the transport really is fresh initialization (no weights transferred) and
   the topology really is frozen (zero mutations during transport).

Seeds here are from already-spent blocks only, so no machinery test observes a
seed in the registered verdict block (81000-81383).
"""

from __future__ import annotations

import json
import math

import pytest

from act.evolution.experiment import ExperimentConfig, run_experiment
from act.evolution.figures_data import summaries_dir
from act.evolution.reuse import transport_genome
from act.topology import ArchitectureGenome, create_genome, genome_hash

SPENT_SEEDS = (73000, 73001)
BASE = dict(
    hidden_size=24,
    vocab_size=32,
    depth=4,
    expert_count=4,
    active_experts=2,
    batch_size=2,
    sequence_length=12,
    steps=24,
    mutation_interval=8,
    mutation_budget=2,
    learning_rate=0.0003,
    validation_batches=4,
    adaptive_policy="probe_loss",
    adaptive_expert_metric="random",
    probe_candidate_cap=4,
    probe_batch_count=2,
    probe_candidate_policy="spread",
    independent_arm_rng=True,
    device="cpu",
)
ARMS = "random,adaptive,fixed"


def _discover(seed: int, **overrides) -> tuple[dict, ExperimentConfig]:
    config = ExperimentConfig(**{**BASE, **overrides}, seed=seed, arms=ARMS)
    result = run_experiment(config)
    return {arm.mode: arm for arm in result.arms}, config


def test_extracted_genome_round_trips_through_json() -> None:
    """Plan section 11 separation: the genome outlives the run, as data."""
    arms, _ = _discover(SPENT_SEEDS[0])
    for mode, arm in arms.items():
        payload = json.loads(json.dumps(arm.genome))
        restored = ArchitectureGenome.from_dict(payload)
        assert genome_hash(restored) == arm.architecture_hash, mode
        assert restored.layers == ArchitectureGenome.from_dict(arm.genome).layers


def test_fixed_arm_genome_is_the_block_initial_genome() -> None:
    """The unmutated arm *is* the no-mutation topology: no separate null needed."""
    arms, config = _discover(SPENT_SEEDS[0])
    initial = create_genome(
        config.seed,
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        depth=config.depth,
        ffn="moe",
        expert_count=config.expert_count,
        active_experts=config.active_experts,
    )
    assert arms["fixed"].mutation_count == 0
    assert arms["fixed"].genome == initial.to_dict()
    assert arms["random"].genome != initial.to_dict()
    assert arms["adaptive"].genome != initial.to_dict()


@pytest.mark.parametrize("seed", SPENT_SEEDS)
def test_transport_reproduces_the_unmutated_arm_exactly(seed: int) -> None:
    """Transport validity: a known arm must come back before deltas mean anything."""
    arms, config = _discover(seed)
    transported = transport_genome(
        ArchitectureGenome.from_dict(arms["fixed"].genome), config
    )
    assert transported.final_loss == arms["fixed"].final_loss
    assert transported.training_flops == arms["fixed"].training_flops
    assert transported.parameter_count == arms["fixed"].parameter_count
    assert transported.active_parameter_count == arms["fixed"].active_parameter_count
    assert transported.architecture_hash == arms["fixed"].architecture_hash


@pytest.mark.parametrize("seed", SPENT_SEEDS)
def test_transport_is_fresh_weights_with_a_frozen_topology(seed: int) -> None:
    """The transport rebuilds the genome; it does not replay the arm's weights."""
    arms, config = _discover(seed)
    for mode in ("random", "adaptive"):
        genome = ArchitectureGenome.from_dict(arms[mode].genome)
        transported = transport_genome(genome, config)
        # Fresh initialization: nothing copied from the run that found it.
        assert transported.initialization["mode"] == "fresh"
        assert transported.initialization["copied_parameters"] == 0
        assert transported.initialization["skipped_parameters"] == 0
        # Frozen topology: same architecture, no mutations.
        assert transported.mutation_count == 0
        assert transported.architecture_hash == genome_hash(genome)
        # The transport books the frozen genome's cost at every step, so its
        # budget is fully predictable. The discovery arm's accumulate is a
        # different quantity - it charges each step the architecture that arm
        # happened to have then, so pre-mutation (e.g. larger) shapes are in
        # it too. Equal only when the arm never mutated; this is why the
        # study compares transported topologies on *paired* transport runs
        # rather than against the discovery arm's own FLOP total.
        assert transported.training_flops == (
            transported.parameter_count
            * config.steps
            * config.batch_size
            * config.sequence_length
            * 2
        )
        # ...and with fresh weights the loss is a fresh draw, not a replay.
        assert transported.final_loss != arms[mode].final_loss


def test_transport_accepts_the_serialized_genome() -> None:
    """The study transports what the discovery run recorded, not a rebuilt copy."""
    arms, config = _discover(SPENT_SEEDS[0])
    from_dict = transport_genome(ArchitectureGenome.from_dict(arms["random"].genome), config)
    from_payload = transport_genome(json.loads(json.dumps(arms["random"].genome)), config)
    assert from_dict.final_loss == from_payload.final_loss


def test_transport_rejects_a_mixed_family_genome() -> None:
    arms, config = _discover(SPENT_SEEDS[0])
    dense = create_genome(config.seed, hidden_size=24, vocab_size=32, depth=2, ffn="dense")
    with pytest.raises(TypeError):
        transport_genome(dense, config)


def test_registered_discovery_config_reproduces_its_own_summary() -> None:
    """The 5.16 discovery block is the registered 5.10 config, still live.

    The committed per-seed payloads are gitignored, so the only free-standing
    guard that the *code* still produces section 5.10's registered numbers is
    to re-run one of its seeds and compare against the committed summary. The
    tolerance is float-scale: AdamW over 48 steps accumulates ~1e-6 of host
    dependent drift (measured 6.7e-7 for this seed), while any behavioural
    change moves these numbers by far more.
    """
    summary = json.loads(
        (summaries_dir() / "null-control-depth4-384.json").read_text()
    )
    recorded = next(row["losses"] for row in summary["per_seed"] if row["seed"] == 71000)
    registered = {
        key: value
        for key, value in summary["config"].items()
        if key in ExperimentConfig.__dataclass_fields__ and key not in {"seed", "arms"}
    }
    result = run_experiment(ExperimentConfig(**registered, seed=71000, arms="random,adaptive,null"))
    for arm in result.arms:
        assert math.isclose(arm.final_loss, recorded[arm.mode], rel_tol=0, abs_tol=1e-4), (
            arm.mode,
            arm.final_loss,
            recorded[arm.mode],
        )
