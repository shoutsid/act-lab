from act.topology import apply_transaction, create_genome, genome_hash


def test_failed_transaction_restores_every_genome_field() -> None:
    genome = create_genome(3, ffn="moe")
    before = genome_hash(genome)

    result = apply_transaction(
        genome,
        "invalid",
        lambda value: (value.layers[0].experts.clear() or True),
    )

    assert not result.applied
    assert result.before_hash == result.after_hash == before
    assert genome_hash(genome) == before
    assert len(genome.layers[0].experts) == 4


def test_successful_transaction_records_hash_transition() -> None:
    genome = create_genome(4, ffn="moe")
    result = apply_transaction(
        genome,
        "disable-layer",
        lambda value: setattr(value.layers[-1], "enabled", False) or True,
    )
    assert result.applied
    assert result.before_hash != result.after_hash
