"""The forced-operation control used to attribute the residual penalty.

Section 5.24 left a residual mid-training penalty of ~0.05 once a mutated layer's
learned router survives the rebuild. Its two candidates are a **pruned** expert's
lost capacity and a **grown** expert's fresh initialization, and a subgroup split
would confound them with the coin flip that chose them.
``ExperimentConfig.forced_operation`` makes every event one operation, so each
candidate gets its own run.

The important property, and the reason this is tested rather than assumed: a
forced operation that the genome cannot perform must be left UNPERFORMED, not
replaced by the opposite operation. A silent substitution would make the
prune-only profile quietly contain grows and measure the wrong thing.
"""

from __future__ import annotations

import pytest

from act.evolution import ExperimentConfig, run_experiment

BASE = dict(
    hidden_size=24,
    vocab_size=32,
    depth=2,
    expert_count=5,
    active_experts=2,
    batch_size=2,
    sequence_length=12,
    steps=36,
    mutation_interval=12,
    mutation_budget=3,
    validation_batches=2,
    device="cpu",
    arms="random",
)


def _total_experts(genome: dict) -> int:
    return sum(len(layer["experts"]) for layer in genome["layers"])


def test_forced_operation_validation() -> None:
    for bad in ("GROW", "keep", ""):
        with pytest.raises(ValueError, match="forced_operation"):
            run_experiment(ExperimentConfig(**BASE, forced_operation=bad))
    run_experiment(ExperimentConfig(**BASE))
    run_experiment(ExperimentConfig(**BASE, forced_operation="grow"))
    run_experiment(ExperimentConfig(**BASE, forced_operation="prune"))


def test_forced_prune_prunes_every_event() -> None:
    arm = run_experiment(ExperimentConfig(**BASE, forced_operation="prune")).arms[0]
    assert arm.mutation_count == 3
    # Experts are per layer, so the starting total is expert_count x depth.
    assert _total_experts(arm.genome) == 5 * BASE["depth"] - 3


def test_forced_grow_grows_every_event() -> None:
    arm = run_experiment(ExperimentConfig(**BASE, forced_operation="grow")).arms[0]
    assert arm.mutation_count == 3
    assert _total_experts(arm.genome) == 5 * BASE["depth"] + 3


def test_forced_operation_that_cannot_be_performed_is_not_substituted() -> None:
    """Starting at the 2-expert floor, a forced prune must leave the arm alone.

    If it fell back to the load-based grow, this profile would be measuring grows
    while claiming to measure prunes - the exact silent substitution the knob's
    docstring promises not to make.
    """
    floor = dict(BASE, expert_count=4)
    starting = 4 * BASE["depth"]
    inspected = 0
    for seed in (1, 2, 3, 4, 5):
        arm = run_experiment(
            ExperimentConfig(**floor, seed=seed, forced_operation="prune")
        ).arms[0]
        inspected += 1
        counts = [len(layer["experts"]) for layer in arm.genome["layers"]]
        assert min(counts) >= 2, "a forced prune went below the 2-expert floor"
        # Every booked mutation removed exactly one expert. If a failed forced
        # prune had been replaced by the load-based GROW fallback, the count
        # would rise while the mutation count also rose.
        assert _total_experts(arm.genome) == starting - arm.mutation_count
    assert inspected == 5


def test_forced_operation_is_live_by_default_it_is_a_coin_flip() -> None:
    forced = run_experiment(ExperimentConfig(**BASE, forced_operation="grow")).arms[0]
    coin = run_experiment(ExperimentConfig(**BASE)).arms[0]
    assert _total_experts(forced.genome) != _total_experts(coin.genome)
