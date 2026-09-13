"""Serializable architecture genomes and safe topology mutations."""

from .genome import ArchitectureGenome, create_genome, genome_hash, validate_genome
from .mutations import (
    feasible_expert_mutations,
    grow_expert,
    clone_expert,
    merge_experts,
    prune_expert,
    rewire_layer,
    split_expert,
    MutationResult,
    apply_transaction,
    change_module,
    toggle_layer,
)

__all__ = [
    "ArchitectureGenome",
    "create_genome",
    "genome_hash",
    "feasible_expert_mutations",
    "grow_expert",
    "clone_expert",
    "MutationResult",
    "apply_transaction",
    "merge_experts",
    "prune_expert",
    "rewire_layer",
    "split_expert",
    "validate_genome",
    "change_module",
    "toggle_layer",
]