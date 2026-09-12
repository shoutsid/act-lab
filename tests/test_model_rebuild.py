import torch

from act.model import model_from_genome, rebuild_model_from_genome
from act.topology import create_genome, prune_expert


def test_moe_rebuild_preserves_expert_weights_by_id() -> None:
    torch.manual_seed(41)
    genome = create_genome(8, hidden_size=24, vocab_size=32, depth=1, ffn="moe")
    original = model_from_genome(genome, context_size=12)
    source_weight = original.blocks[0].experts[1][0].weight.detach().clone()

    assert prune_expert(genome, "L0", "L0E0")
    rebuilt = rebuild_model_from_genome(original, genome, context_size=12)

    assert rebuilt.blocks[0].expert_ids == ("L0E1", "L0E2", "L0E3")
    assert torch.equal(rebuilt.blocks[0].experts[0][0].weight, source_weight)