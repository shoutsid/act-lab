from act.evolution import enumerate_module_depth_variants
from act.topology import create_genome


def test_module_depth_variants_are_stable_and_valid() -> None:
    genome = create_genome(5, hidden_size=16, depth=2, ffn="dense")
    variants = enumerate_module_depth_variants(genome)
    assert variants
    assert all(variant.enabled_layers >= 1 for _, variant in variants)
    assert len({variant.architecture_hash for _, variant in variants}) == len(variants)
    assert variants == enumerate_module_depth_variants(genome)
