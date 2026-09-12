"""Deterministic module and effective-depth variant generation."""

from dataclasses import dataclass
from copy import deepcopy

from act.topology import ArchitectureGenome, change_module, genome_hash, toggle_layer


@dataclass(frozen=True)
class ModuleDepthVariant:
    operation: str
    target: str
    module: str | None
    enabled_layers: int
    architecture_hash: str


def enumerate_module_depth_variants(
    genome: ArchitectureGenome,
) -> tuple[tuple[ArchitectureGenome, ModuleDepthVariant], ...]:
    """Generate valid one-step module/depth variants in stable genome order."""
    variants: list[tuple[ArchitectureGenome, ModuleDepthVariant]] = []
    for layer in genome.layers:
        for module in ("attention", "gdn", "hybrid"):
            candidate = deepcopy(genome)
            if change_module(candidate, layer.id, module):
                variants.append(
                    (
                        candidate,
                        ModuleDepthVariant(
                            "change_module",
                            layer.id,
                            module,
                            sum(item.enabled for item in candidate.layers),
                            genome_hash(candidate),
                        ),
                    )
                )
        if layer.enabled:
            candidate = deepcopy(genome)
            if toggle_layer(candidate, layer.id, False):
                variants.append(
                    (
                        candidate,
                        ModuleDepthVariant(
                            "disable_layer",
                            layer.id,
                            None,
                            sum(item.enabled for item in candidate.layers),
                            genome_hash(candidate),
                        ),
                    )
                )
    return tuple(variants)
