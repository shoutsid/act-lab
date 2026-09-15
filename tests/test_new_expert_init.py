"""The cloned-expert initialization, and the pairing property it enables.

Section 5.25 attributed the residual mid-training penalty to the grow, the only
operation that introduces a **freshly drawn** expert. ``new_expert_initialization``
lets a grown expert start as a twin of the layer's busiest survivor instead - body
and router row - so section 5.26 can ask whether the fresh draw is the mechanism.

Two properties are pinned here because section 5.26's whole design rests on them:

1. ``clone`` copies the busiest survivor's body AND its router row. A cloned body
   with a freshly drawn row would start with no routing affinity at all, so the
   row is what makes it a twin rather than a coincidence of weights.
2. ``clone`` requires ``preserve_router_on_rebuild=True``: it is defined as copying
   a router row, so with the router being discarded it has no meaning. It raises
   rather than silently cloning nothing, which would let a study read a clone arm
   that never cloned.
"""

from __future__ import annotations

import pytest
import torch

from act.evolution import ExperimentConfig, run_experiment
from act.model import model_from_genome, rebuild_model_from_genome
from act.topology import ArchitectureGenome, create_genome, grow_expert, prune_expert

BODY_SUFFIXES = ("0.weight", "0.bias", "2.weight", "2.bias")


def _genome(expert_count: int = 4):
    return create_genome(
        7,
        hidden_size=24,
        vocab_size=32,
        depth=2,
        ffn="moe",
        expert_count=expert_count,
        active_experts=2,
    )


def _grown_target(expert_count: int = 4):
    target = _genome(expert_count)
    new_id = grow_expert(target, target.layers[0].id, step=1)
    assert new_id is not None
    return target, new_id


def test_clone_copies_the_busiest_survivors_body_and_router_row() -> None:
    source_genome = _genome()
    model = model_from_genome(source_genome, context_size=32, num_heads=4)
    with torch.no_grad():
        # Distinct weights per expert, and a routing load that makes index 3 the
        # busiest. The clone must pick 3, so nothing here depends on a tie.
        for index, expert in enumerate(model.blocks[0].experts):
            expert[0].weight.fill_(float(index) + 1.0)
        model.blocks[0].last_load = torch.tensor([0.1, 0.4, 0.05, 0.45])
        model.blocks[0].router.weight.fill_(0.25)

    target, new_id = _grown_target()
    rebuilt = rebuild_model_from_genome(
        model,
        target,
        context_size=32,
        num_heads=4,
        preserve_router=True,
        new_expert_initialization="clone",
    )
    source_state = model.state_dict()
    rebuilt_state = rebuilt.state_dict()
    new_index = [expert.id for expert in target.layers[0].experts].index(new_id)

    for suffix in BODY_SUFFIXES:
        assert torch.equal(
            rebuilt_state[f"blocks.0.experts.{new_index}.{suffix}"],
            source_state[f"blocks.0.experts.3.{suffix}"],
        ), f"grown expert body {suffix} was not cloned from the busiest survivor"
    assert torch.equal(
        rebuilt_state["blocks.0.router.weight"][new_index],
        source_state["blocks.0.router.weight"][3],
    ), "the cloned expert's router row must be copied too"


def test_random_initialization_leaves_the_grown_expert_fresh() -> None:
    source_genome = _genome()
    model = model_from_genome(source_genome, context_size=32, num_heads=4)
    with torch.no_grad():
        for index, expert in enumerate(model.blocks[0].experts):
            expert[0].weight.fill_(float(index) + 1.0)
        model.blocks[0].last_load = torch.tensor([0.1, 0.4, 0.05, 0.45])

    target, new_id = _grown_target()
    rebuilt = rebuild_model_from_genome(
        model,
        target,
        context_size=32,
        num_heads=4,
        preserve_router=True,
        new_expert_initialization="random",
    )
    rebuilt_state = rebuilt.state_dict()
    new_index = [expert.id for expert in target.layers[0].experts].index(new_id)
    clone_weights = model.state_dict()["blocks.0.experts.3.0.weight"]
    assert not torch.equal(
        rebuilt_state[f"blocks.0.experts.{new_index}.0.weight"], clone_weights
    ), "the historical default must still draw a fresh expert"


def test_clone_moves_nothing_when_no_expert_is_grown() -> None:
    """A prune-only rebuild has no new expert, so cloning must be a no-op."""
    source_genome = _genome()
    model = model_from_genome(source_genome, context_size=32, num_heads=4)
    with torch.no_grad():
        for index, expert in enumerate(model.blocks[0].experts):
            expert[0].weight.fill_(float(index) + 1.0)
        model.blocks[0].last_load = torch.tensor([0.1, 0.4, 0.05, 0.45])
        for block in model.blocks:
            block.router.weight.fill_(0.25)

    target = _genome()
    prune_expert(target, target.layers[0].id, target.layers[0].experts[-1].id)
    rebuilt = rebuild_model_from_genome(
        model,
        target,
        context_size=32,
        num_heads=4,
        preserve_router=True,
        new_expert_initialization="clone",
    )
    assert bool((rebuilt.blocks[0].router.weight == 0.25).all())
    assert bool((rebuilt.blocks[1].router.weight == 0.25).all())


_BASE = dict(
    hidden_size=24,
    vocab_size=32,
    depth=2,
    expert_count=4,
    active_experts=2,
    batch_size=2,
    sequence_length=12,
    steps=12,
    mutation_interval=6,
    mutation_budget=1,
    validation_batches=2,
    device="cpu",
    arms="random",
    forced_operation="grow",
)


def test_clone_requires_the_router_to_survive() -> None:
    """Cloning is copying a router row, so the combination that clones nothing
    must be refused rather than silently accepted."""
    with pytest.raises(ValueError, match="clone"):
        run_experiment(
            ExperimentConfig(
                **_BASE,
                new_expert_initialization="clone",
                preserve_router_on_rebuild=False,
            )
        )
    # Both legitimate settings run.
    run_experiment(
        ExperimentConfig(
            **_BASE,
            new_expert_initialization="clone",
            preserve_router_on_rebuild=True,
        )
    )
    run_experiment(ExperimentConfig(**_BASE, new_expert_initialization="random"))


def test_new_expert_initialization_validation() -> None:
    for bad in ("Clone", "teacher", ""):
        with pytest.raises(ValueError, match="new_expert_initialization"):
            run_experiment(
                ExperimentConfig(
                    **_BASE,
                    new_expert_initialization=bad,
                    preserve_router_on_rebuild=True,
                )
            )


def test_the_two_settings_build_the_same_architecture() -> None:
    """Section 5.26's pairing: the forced grow picks its layer from the seed's own
    RNG and cloning consumes no randomness, so both settings must end on one
    genome. If this ever fails, the contrast is between two substrates."""
    from act.topology import genome_hash

    common = {
        **_BASE,
        "steps": 36,
        "mutation_budget": 3,
        "preserve_router_on_rebuild": True,
        "seed": 4242,
    }
    random_arm = run_experiment(
        ExperimentConfig(**common, new_expert_initialization="random")
    ).arms[0]
    clone_arm = run_experiment(
        ExperimentConfig(**common, new_expert_initialization="clone")
    ).arms[0]
    assert random_arm.mutation_count == clone_arm.mutation_count == 3
    assert genome_hash(
        ArchitectureGenome.from_dict(random_arm.genome)
    ) == genome_hash(ArchitectureGenome.from_dict(clone_arm.genome))
