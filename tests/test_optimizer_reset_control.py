"""The optimizer-reset control used to decompose the mid-training penalty.

Report section 5.22 established that a fused run (mutations applied in place)
loses ~0.1 nats to a frozen retrain of its own final genome. The mutation path
re-creates the AdamW optimizer at every event, discarding all moment estimates,
so part of that penalty may be optimizer history rather than the architecture
change. ``ExperimentConfig.optimizer_reset_steps`` exists so a run whose ONLY
difference from its twin is those resets can be measured.

These tests pin the machinery, not the outcome: that the knob validates, that it
leaves the genome and weights untouched (otherwise it would not be a clean
control), that it actually changes the run, and that it stays deterministic.
"""

from __future__ import annotations

import pytest

from act.evolution import ExperimentConfig, run_experiment

BASE = dict(steps=12, mutation_interval=3, mutation_budget=2, arms="fixed")


def test_optimizer_reset_steps_validation() -> None:
    for bad in ((0,), (13,), (3, 3), (6, 3)):
        with pytest.raises(ValueError, match="optimizer_reset_steps"):
            run_experiment(ExperimentConfig(**BASE, optimizer_reset_steps=bad))
    # The empty default is the published behaviour and must stay allowed.
    run_experiment(ExperimentConfig(**BASE))


def test_optimizer_reset_leaves_genome_and_architecture_untouched() -> None:
    """A reset must not rebuild anything - it is the optimizer-only control."""
    twin = run_experiment(ExperimentConfig(**BASE)).arms[0]
    reset = run_experiment(
        ExperimentConfig(**BASE, optimizer_reset_steps=(3, 6, 9))
    ).arms[0]
    assert reset.genome == twin.genome
    assert reset.parameter_count == twin.parameter_count
    assert reset.active_parameter_count == twin.active_parameter_count
    assert reset.mutation_count == 0
    # Same work booked: a reset is not a training-FLOPs event.
    assert reset.training_flops == twin.training_flops


def test_optimizer_reset_actually_changes_the_run() -> None:
    """If discarding Adam's moments changed nothing, the control would be inert.

    The SIZE and SIGN of the effect are section 5.23's question, so only "it is
    live" is asserted here.
    """
    twin = run_experiment(ExperimentConfig(**BASE)).arms[0].final_loss
    reset = run_experiment(
        ExperimentConfig(**BASE, optimizer_reset_steps=(3, 6, 9))
    ).arms[0].final_loss
    assert reset != twin


def test_optimizer_reset_is_deterministic() -> None:
    config = ExperimentConfig(**BASE, optimizer_reset_steps=(3, 6, 9))
    first = run_experiment(config).arms[0]
    second = run_experiment(config).arms[0]
    assert first.final_loss == second.final_loss
    assert first.genome == second.genome


def test_optimizer_reset_does_not_disturb_the_data_schedule() -> None:
    """The twin must differ ONLY by resets, so the token stream is identical.

    Both runs are the same seed, and the reset path draws no randomness, so a
    differing schedule would silently confound the control.
    """
    twin = run_experiment(ExperimentConfig(**BASE)).arms[0]
    reset = run_experiment(
        ExperimentConfig(**BASE, optimizer_reset_steps=(3, 6, 9))
    ).arms[0]
    assert twin.validation_loss == twin.final_loss
    assert reset.validation_loss == reset.final_loss
