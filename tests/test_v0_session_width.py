"""The width arm: a graft at a structural event, and the schedule read against it.

These run the real session loop at a toy shape - six steps, two events - because
what is pinned is not arithmetic but the wiring: that the event grafts, that a
declined graft leaves the model and optimizer alone, that both arms pay the same
probe, and that the checkpoint carries the widened genome the resume path
rebuilds from. A test that only called ``graft_widened_model`` would not see the
event loop at all, which is where an arm can be wired into one branch and
forgotten in another.
"""

from __future__ import annotations

import pytest

from act.training import V0SessionConfig, run_v0_session
from act.training.v0_session import genome_from_checkpoint

TINY = dict(
    vocab_size=64,
    context_size=32,
    hidden_size=128,
    depth=2,
    attention_heads=2,
    expert_count=2,
    active_experts=2,
    expert_width=128,
)


def _config(**overrides: object) -> V0SessionConfig:
    fields: dict[str, object] = dict(
        seed=7,
        arms=("width",),
        device="cpu",
        total_steps=6,
        batch_size=2,
        mutation_interval=3,
        mutation_budget=2,
        validation_batches=1,
        checkpoint_interval=100,
        # Large enough that the toy graft cannot lose by this much, so the
        # accept path is exercised without depending on a noise draw.
        width_accept_margin=1e9,
        **TINY,
    )
    fields.update(overrides)
    return V0SessionConfig(**fields)  # type: ignore[arg-type]


def _arm(config: V0SessionConfig, tmp_path):
    return run_v0_session(config, checkpoint_dir=tmp_path).arms[0]


def test_a_width_event_grows_the_model_and_records_the_graft(tmp_path) -> None:
    """Two events at +128: 128 -> 256 -> 384, with the graft row each time."""

    arm = _arm(_config(), tmp_path)

    assert arm.mutation_count == 2
    accepted = [row for row in arm.mutation_history if row["gate_decision"] == "accept"]
    assert len(accepted) == 2
    assert (accepted[0]["from_hidden"], accepted[0]["to_hidden"]) == (128, 256)
    assert (accepted[1]["from_hidden"], accepted[1]["to_hidden"]) == (256, 384)
    assert accepted[0]["heads_added"] == 2
    assert accepted[0]["perturbation_coordinates"] > 0
    assert accepted[0]["total_after"] > accepted[0]["total_before"]
    # Both probe evaluations are booked, so the cost identity is not a guess.
    assert arm.search_flops > 0

    # The resume path rebuilds from the checkpoint's genome, so a checkpoint
    # whose width moved is only resumable if the genome moved with it.
    genome = genome_from_checkpoint(
        tmp_path / "seed-7" / "arm-width" / "checkpoint-000006.pt"
    )
    assert genome.hidden_size == 384
    assert genome.layers[0].heads == 6


def test_the_adaptive_rule_declines_a_graft_that_loses(tmp_path) -> None:
    """A rejected event changes nothing but still records what it would have cost."""

    arm = _arm(_config(width_accept_margin=-1e9), tmp_path)

    assert arm.mutation_count == 0
    assert [row["gate_decision"] for row in arm.mutation_history] == ["reject", "reject"]
    assert all(row["draw_probe_loss"] is not None for row in arm.mutation_history)
    assert all(row["mutation_count"] == 0 for row in arm.mutation_history)

    genome = genome_from_checkpoint(
        tmp_path / "seed-7" / "arm-width" / "checkpoint-000006.pt"
    )
    assert genome.hidden_size == 128


def test_the_random_schedule_is_a_control_at_the_same_probe_cost(tmp_path) -> None:
    """At probability 0 it never widens, and it still pays to look.

    That is the design's point: the control differs from the adaptive arm in the
    decision rule, not in the information it buys or the events it consumes.
    """

    arm = _arm(
        _config(arms=("width_random",), width_random_probability=0.0), tmp_path
    )

    assert arm.mutation_count == 0
    assert {row["gate_decision"] for row in arm.mutation_history} == {"reject_random"}
    assert all(row["draw_probe_loss"] is not None for row in arm.mutation_history)
    assert arm.search_flops > 0


def test_declines_do_not_buy_extra_width_events(tmp_path) -> None:
    """Both width arms fire the registered slots, not one per accepted graft.

    The registration fixes three decision events before the run (the pilot's
    100/200/300 in a 400-step horizon). This is that schedule at toy scale:
    interval 3, budget 3, 12 steps, so the slots are 3/6/9 and step 12 - after
    the last optimizer step and before validation - is not an event. A coin
    arm that never accepts and an adaptive arm that always declines must both
    stop at the third slot; budgeting accepted mutations instead fired a
    fourth event at step 12 in every fetched ``width_random`` record.
    """

    for arm_name, knobs in (
        ("width", {"width_accept_margin": -1e9}),
        ("width_random", {"width_random_probability": 0.0}),
    ):
        arm = _arm(
            _config(
                arms=(arm_name,),
                total_steps=12,
                mutation_interval=3,
                mutation_budget=3,
                **knobs,
            ),
            tmp_path / arm_name,
        )
        assert [row["step"] for row in arm.mutation_history] == [3, 6, 9]
        assert arm.mutation_count == 0


def test_the_width_arm_is_single_and_its_knobs_are_validated() -> None:
    with pytest.raises(ValueError):
        _config(arms=("width", "random"))
    with pytest.raises(ValueError):
        _config(arms=("width_random", "random"))
    with pytest.raises(ValueError):
        _config(width_heads_added=0)
    with pytest.raises(ValueError):
        _config(width_perturbation_scale=-1.0)
    with pytest.raises(ValueError):
        _config(width_random_probability=1.5)
