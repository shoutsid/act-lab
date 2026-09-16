"""The declared-path rebuild's router treatment (report sections 5.24 / 5.34).

``rebuild_v0_moe_from_genome`` copies expert bodies by id, but the router is
``hidden_size x experts_in_layer`` and its shape moves with the expert count, so the
same-shape copy cannot carry it: by default the whole routing head is re-drawn at
random on every event. These tests pin the fix on this path - which the toy path got
in section 5.24 and which the declared/real path still had - and pin that the
defaults are untouched, because every committed number was measured under them.
"""

from __future__ import annotations

import torch

from act.model import v0_moe_model_from_genome
from act.model.factory import rebuild_v0_moe_from_genome
from act.training import V0SessionConfig
from act.topology import create_genome, grow_expert, prune_expert

ROUTER = "moe.router.weight"


def _genome(seed: int = 11, expert_count: int = 4, depth: int = 2):
    return create_genome(
        seed,
        hidden_size=16,
        vocab_size=64,
        depth=depth,
        ffn="moe",
        expert_count=expert_count,
        active_experts=2,
    )


def _filled(model, value: float):
    with torch.no_grad():
        for block in model.blocks:
            block.moe.router.weight.fill_(value)
            block.moe.experts[0][0].weight.fill_(value)
    return model


def test_the_default_still_discards_the_mutated_layers_router() -> None:
    """The defect itself, pinned: experts survive by id, the routing head does not."""
    model = _filled(v0_moe_model_from_genome(_genome()), 0.25)
    target = _genome()
    assert prune_expert(target, target.layers[0].id, target.layers[0].experts[-1].id)
    rebuilt = rebuild_v0_moe_from_genome(model, target)
    assert bool((rebuilt.blocks[0].moe.experts[0][0].weight == 0.25).all())
    assert not bool((rebuilt.blocks[0].moe.router.weight == 0.25).all())
    # The unmutated layer's router is carried, which shows the copy loop works and
    # the loss is specific to the mutated layer.
    assert bool((rebuilt.blocks[1].moe.router.weight == 0.25).all())


def test_preserve_router_carries_exactly_the_surviving_rows() -> None:
    model = _filled(v0_moe_model_from_genome(_genome()), 0.25)
    target = _genome()
    assert prune_expert(target, target.layers[0].id, target.layers[0].experts[-1].id)
    rebuilt = rebuild_v0_moe_from_genome(model, target, preserve_router=True)
    rows = rebuilt.blocks[0].moe.router.weight
    assert rows.shape[0] == 3
    assert bool((rows == 0.25).all())


def test_clone_initialises_a_grown_expert_from_the_busiest_survivor() -> None:
    """The new expert must be a working twin, body and routing row both."""
    model = _filled(v0_moe_model_from_genome(_genome()), 0.25)
    with torch.no_grad():
        # Give the survivors distinct values and loads so the source is knowable.
        for index, expert in enumerate(model.blocks[0].moe.experts):
            expert[0].weight.fill_(float(index + 1))
        model.blocks[0].moe.last_load = torch.tensor([1.0, 9.0, 2.0, 3.0])
        model.blocks[0].moe.router.weight.copy_(
            torch.arange(4, dtype=torch.float32).reshape(4, 1).expand(4, 16)
        )
    target = _genome()
    assert grow_expert(target, target.layers[0].id, step=100) is not None
    rebuilt = rebuild_v0_moe_from_genome(
        model, target, preserve_router=True, new_expert_initialization="clone"
    )
    new_index = len(model.blocks[0].moe.experts)  # the appended slot
    busiest = 1  # last_load above
    assert bool(
        (
            rebuilt.blocks[0].moe.experts[new_index][0].weight
            == model.blocks[0].moe.experts[busiest][0].weight
        ).all()
    )
    assert bool(
        (
            rebuilt.blocks[0].moe.router.weight[new_index]
            == model.blocks[0].moe.router.weight[busiest]
        ).all()
    )


def test_clone_body_leaves_the_new_row_fresh() -> None:
    """The control setting: same body, row drawn - so the row must NOT be copied."""
    model = _filled(v0_moe_model_from_genome(_genome()), 0.25)
    with torch.no_grad():
        model.blocks[0].moe.router.weight.fill_(0.75)
    target = _genome()
    assert grow_expert(target, target.layers[0].id, step=100) is not None
    torch.manual_seed(5)
    rebuilt = rebuild_v0_moe_from_genome(
        model, target, preserve_router=True, new_expert_initialization="clone_body"
    )
    new_index = len(model.blocks[0].moe.experts)
    assert bool(
        (
            rebuilt.blocks[0].moe.experts[new_index][0].weight
            == model.blocks[0].moe.experts[0][0].weight
        ).all()
    )
    assert not bool((rebuilt.blocks[0].moe.router.weight[new_index] == 0.75).all())


def test_the_session_config_rejects_cloning_without_preservation() -> None:
    import pytest

    with pytest.raises(ValueError):
        V0SessionConfig(new_expert_initialization="clone")
    with pytest.raises(ValueError):
        V0SessionConfig(new_expert_initialization="whatever")
    config = V0SessionConfig(
        preserve_router_on_rebuild=True, new_expert_initialization="clone"
    )
    assert config.preserve_router_on_rebuild is True


def test_the_dev_defaults_are_unchanged() -> None:
    """Every committed real-path number was measured with these two values."""
    config = V0SessionConfig()
    assert config.preserve_router_on_rebuild is False
    assert config.new_expert_initialization == "random"
