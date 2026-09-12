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


def merge_experts(
    genome: ArchitectureGenome,
    layer_id: str,
    keep_expert_id: str,
    remove_expert_id: str,
) -> bool:
    """Remove a redundant expert while retaining the selected expert identity."""
    layer = next((item for item in genome.layers if item.id == layer_id), None)
    if layer is None or layer.ffn != "moe" or keep_expert_id == remove_expert_id:
        return False
    if len(layer.experts) <= 2:
        return False
    ids = {expert.id for expert in layer.experts}
    if keep_expert_id not in ids or remove_expert_id not in ids:
        return False
    original = deepcopy(genome)
    layer.experts = [expert for expert in layer.experts if expert.id != remove_expert_id]
    layer.active_experts = min(layer.active_experts, len(layer.experts))
    if not validate_genome(genome).valid:
        genome.layers = original.layers
        return False
    return True


def split_expert(genome: ArchitectureGenome, layer_id: str, expert_id: str, step: int) -> str | None:
    """Create a new expert with the source expert's shape and a fresh identity."""
    layer = next((item for item in genome.layers if item.id == layer_id), None)
    if layer is None or layer.ffn != "moe" or len(layer.experts) >= 16:
        return None
    source = next((expert for expert in layer.experts if expert.id == expert_id), None)
    if source is None:
        return None
    original = deepcopy(genome)
    suffix = len(layer.experts)
    new_id = f"{layer.id}E{suffix}"
    used_ids = {expert.id for expert in layer.experts}
    while new_id in used_ids:
        suffix += 1
        new_id = f"{layer.id}E{suffix}"
    layer.experts.append(ExpertGene(id=new_id, width=source.width, birth_step=step))
    if not validate_genome(genome).valid:
        genome.layers = original.layers
        return None
    return new_id


def rewire_layer(genome: ArchitectureGenome, layer_id: str, input_layer_id: str) -> bool:
    """Replace a layer's residual input edge, rejecting cycles and unknown nodes."""
    layer = next((item for item in genome.layers if item.id == layer_id), None)
    if layer is None or input_layer_id == layer_id:
        return False
    if not any(item.id == input_layer_id for item in genome.layers):
        return False
    original = deepcopy(genome)
    layer.inputs = [input_layer_id]
    if not validate_genome(genome).valid:
        genome.layers = original.layers
        return False
    return True