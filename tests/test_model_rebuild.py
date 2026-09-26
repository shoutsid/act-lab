import pytest
import torch

from act.model import model_from_genome, rebuild_model_from_genome
from act.model.factory import rebuild_v0_moe_from_genome
from act.model import v0_moe_model_from_genome
from act.topology import create_genome, prune_expert


@pytest.mark.parametrize("setting", ["clone", "clone_body"])
def test_toy_rebuild_requires_router_preservation_for_clone_modes(setting: str) -> None:
    genome = create_genome(8, hidden_size=24, vocab_size=32, depth=1, ffn="moe")
    model = model_from_genome(genome, context_size=12)

    with pytest.raises(ValueError, match="requires preserve_router=True"):
        rebuild_model_from_genome(
            model,
            genome,
            context_size=12,
            preserve_router=False,
            new_expert_initialization=setting,
        )


@pytest.mark.parametrize(
    "rebuild",
    [rebuild_model_from_genome, rebuild_v0_moe_from_genome],
)
def test_rebuild_entry_points_reject_unknown_new_expert_initialization(rebuild) -> None:
    genome = create_genome(8, hidden_size=24, vocab_size=32, depth=1, ffn="moe")
    model = (
        model_from_genome(genome, context_size=12)
        if rebuild is rebuild_model_from_genome
        else v0_moe_model_from_genome(
            create_genome(8, hidden_size=24, vocab_size=32, depth=1, ffn="moe")
        )
    )

    with pytest.raises(ValueError, match="new_expert_initialization"):
        rebuild(
            model,
            genome,
            preserve_router=True,
            new_expert_initialization="cloen",
        )


def test_moe_rebuild_preserves_expert_weights_by_id() -> None:
    torch.manual_seed(41)
    genome = create_genome(8, hidden_size=24, vocab_size=32, depth=1, ffn="moe")
    original = model_from_genome(genome, context_size=12)
    source_weight = original.blocks[0].experts[1][0].weight.detach().clone()

    assert prune_expert(genome, "L0", "L0E0")
    rebuilt = rebuild_model_from_genome(original, genome, context_size=12)

    assert rebuilt.blocks[0].expert_ids == ("L0E1", "L0E2", "L0E3")
    assert torch.equal(rebuilt.blocks[0].experts[0][0].weight, source_weight)