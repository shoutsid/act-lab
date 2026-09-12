from act.topology import change_module, create_genome, toggle_layer, validate_genome


def test_module_mutation_preserves_valid_genome() -> None:
    genome = create_genome(2, hidden_size=16, depth=2, ffn="dense")
    assert change_module(genome, "L1", "hybrid")
    assert genome.layers[1].module == "hybrid"
    assert validate_genome(genome).valid


def test_depth_mutation_never_disables_all_layers() -> None:
    genome = create_genome(3, hidden_size=16, depth=2, ffn="dense")
    assert toggle_layer(genome, "L1", False)
    assert validate_genome(genome).valid
    assert not toggle_layer(genome, "L0", False)
    assert genome.layers[0].enabled
