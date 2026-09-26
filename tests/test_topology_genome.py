from copy import deepcopy

import pytest

from act.topology import clone_expert, create_genome, genome_hash, grow_expert, prune_expert, validate_genome
from act.topology.genome import ArchitectureGenome


def test_genome_round_trip_identity_and_hash() -> None:
    genome = create_genome(5, ffn="moe")
    restored = create_genome(5, ffn="moe")
    assert genome.to_json() == restored.to_json()
    assert genome_hash(genome) == genome_hash(restored)


def test_prune_and_grow_preserve_safety() -> None:
    genome = create_genome(9, ffn="moe")
    original = deepcopy(genome)
    assert prune_expert(genome, "L0", "L0E0")
    assert validate_genome(genome).valid
    assert grow_expert(genome, "L0", step=12) is not None
    assert validate_genome(genome).valid
    assert genome.layers[0].experts != original.layers[0].experts


def test_invalid_stale_index_is_rejected() -> None:
    genome = create_genome(3)
    genome.layers[0].index = 4
    result = validate_genome(genome)
    assert not result.valid
    assert any("stale index" in error for error in result.errors)


@pytest.mark.parametrize(
    ("hidden_size", "heads"),
    [(0, 4), (64, 0)],
)
def test_invalid_zero_width_or_head_count_is_reported_without_arithmetic_error(
    hidden_size: int, heads: int
) -> None:
    genome = create_genome(3)
    genome.hidden_size = hidden_size
    genome.layers[0].heads = heads

    validation = validate_genome(genome)

    assert not validation.valid
    assert validation.errors
    with pytest.raises(ValueError, match="invalid genome"):
        ArchitectureGenome.from_dict(genome.to_dict())


def test_prune_with_stale_index_does_not_leave_a_partial_mutation() -> None:
    genome = create_genome(10, depth=2, ffn="moe")
    assert prune_expert(genome, "L0", "L0E0")  # layer 0 now has three experts
    target = genome.layers[1]
    target.index = 0  # targets layer 0's expert count in the old implementation
    original_ids = [expert.id for expert in target.experts]

    assert not prune_expert(genome, target.id, original_ids[0])
    assert [expert.id for expert in genome.layers[1].experts] == original_ids


def test_create_genome_rejects_zero_model_dimensions() -> None:
    with pytest.raises(ValueError, match="positive"):
        create_genome(4, hidden_size=0)
    with pytest.raises(ValueError, match="positive"):
        create_genome(4, vocab_size=0)


def test_validation_rejects_cycles_in_layer_input_graph() -> None:
    genome = create_genome(8, depth=2)
    genome.layers[0].inputs = ["L1"]
    genome.layers[1].inputs = ["L0"]

    validation = validate_genome(genome)

    assert not validation.valid
    assert "layer inputs contain a cycle" in validation.errors


def test_rewire_rejects_a_cycle_and_restores_the_original_edges() -> None:
    from act.topology import rewire_layer

    genome = create_genome(9, depth=3)
    assert rewire_layer(genome, "L1", "L0")
    assert rewire_layer(genome, "L2", "L1")
    before = genome_hash(genome)

    assert not rewire_layer(genome, "L0", "L2")

    assert genome_hash(genome) == before
    assert validate_genome(genome).valid


def test_clone_creates_a_new_expert_identity() -> None:
    genome = create_genome(12, ffn="moe")
    clone_id = clone_expert(genome, "L0", "L0E1", step=4)
    assert clone_id == "L0E4"
    assert validate_genome(genome).valid