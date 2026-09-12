from copy import deepcopy

from act.topology import create_genome, genome_hash, grow_expert, prune_expert, validate_genome


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