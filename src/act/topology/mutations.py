"""Safe, in-place topology operators for the Python genome."""

from copy import deepcopy

from .genome import ArchitectureGenome, ExpertGene, validate_genome


def prune_expert(genome: ArchitectureGenome, layer_id: str, expert_id: str) -> bool:
    layer = next((layer for layer in genome.layers if layer.id == layer_id), None)
    if layer is None or layer.ffn != "moe" or len(layer.experts) <= 2:
        return False
    original = deepcopy(genome)
    layer.experts = [expert for expert in layer.experts if expert.id != expert_id]
    if len(layer.experts) == len(original.layers[layer.index].experts):
        return False
    layer.active_experts = min(layer.active_experts, len(layer.experts))
    if not validate_genome(genome).valid:
        genome.layers = original.layers
        return False
    return True


def grow_expert(genome: ArchitectureGenome, layer_id: str, step: int) -> str | None:
    layer = next((layer for layer in genome.layers if layer.id == layer_id), None)
    if layer is None or layer.ffn != "moe" or len(layer.experts) >= 16:
        return None
    original = deepcopy(genome)
    used_ids = {expert.id for expert in layer.experts}
    suffix = len(layer.experts)
    expert_id = f"{layer.id}E{suffix}"
    while expert_id in used_ids:
        suffix += 1
        expert_id = f"{layer.id}E{suffix}"
    layer.experts.append(ExpertGene(id=expert_id, width=layer.experts[0].width, birth_step=step))
    if not validate_genome(genome).valid:
        genome.layers = original.layers
        return None
    return expert_id