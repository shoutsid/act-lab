"""Safe, in-place topology operators for the Python genome."""

from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Callable, Mapping, Sequence

from .genome import ArchitectureGenome, ExpertGene, validate_genome


@dataclass(frozen=True)
class MutationResult:
    applied: bool
    operation: str
    before_hash: str
    after_hash: str
    errors: tuple[str, ...] = ()


def apply_transaction(
    genome: ArchitectureGenome,
    operation: str,
    change: Callable[[ArchitectureGenome], bool],
) -> MutationResult:
    """Apply a topology change transactionally, restoring the prior genome on failure."""
    from .genome import genome_hash

    before = genome_hash(genome)
    original = deepcopy(genome)

    def restore() -> None:
        for field in fields(genome):
            setattr(genome, field.name, deepcopy(getattr(original, field.name)))

    try:
        changed = bool(change(genome))
        validation = validate_genome(genome) if changed else None
    except Exception as error:
        restore()
        return MutationResult(False, operation, before, before, (str(error),))

    if not changed or validation is None or not validation.valid:
        restore()
        errors = (
            tuple(validation.errors)
            if validation is not None and not validation.valid
            else ("operation made no change",)
        )
        return MutationResult(False, operation, before, before, errors)
    return MutationResult(True, operation, before, genome_hash(genome))


def feasible_expert_mutations(
    genome: ArchitectureGenome,
) -> tuple[tuple[str, str, str | None], ...]:
    """Enumerate every feasible expert grow/prune candidate for a genome.

    Yields ``("grow", layer_id, None)`` for every MoE layer below the 16-expert
    ceiling and ``("prune", layer_id, expert_id)`` for every expert above the
    two-expert floor, in layer-then-expert order. Shared by the toy evolution
    experiments and the V0-scale session so candidate semantics cannot drift.
    """
    candidates: list[tuple[str, str, str | None]] = []
    for layer in genome.layers:
        if layer.ffn != "moe":
            continue
        if len(layer.experts) < 16:
            candidates.append(("grow", layer.id, None))
        if len(layer.experts) > 2:
            for expert in layer.experts:
                candidates.append(("prune", layer.id, expert.id))
    return tuple(candidates)


def quietest_prune_index(
    candidates: Sequence[tuple[str, str, str | None]],
    layer_experts: Mapping[str, Sequence[str]],
    loads: Mapping[str, Sequence[float]],
    fallback: int = 0,
) -> int:
    """Index of the prune candidate whose expert carries the least recorded load.

    Shared by both probe paths so the rule cannot drift between them. The probe
    enumerates *every* feasible candidate (``feasible_expert_mutations``) and scores
    them by probe loss; this helper answers the different question of which expert a
    cheap load rule would remove instead, so the two selectors can be compared.

    ``candidates`` is the pool **in the order it was decided on** (for the probe
    paths that is probe-loss order) and the return value indexes that same sequence.
    Non-prune candidates are ignored, and an expert whose load was never recorded is
    skipped rather than guessed at. ``fallback`` is returned when the pool holds no
    scoreable prune, so the caller's existing rule decides in that case rather than
    an arbitrary pick. Ties resolve to the lowest index, deterministically.
    """
    best_load: float | None = None
    best_index = fallback
    for index, (operation, layer_id, expert_id) in enumerate(candidates):
        if operation != "prune" or expert_id is None:
            continue
        experts = layer_experts.get(layer_id)
        recorded = loads.get(layer_id)
        if not experts or not recorded:
            continue
        if expert_id not in experts:
            continue
        position = experts.index(expert_id)
        if position >= len(recorded):
            continue
        load = float(recorded[position])
        if best_load is None or load < best_load:
            best_load = load
            best_index = index
    return best_index


def prune_expert(genome: ArchitectureGenome, layer_id: str, expert_id: str) -> bool:
    layer = next((layer for layer in genome.layers if layer.id == layer_id), None)
    if layer is None or layer.ffn != "moe" or len(layer.experts) <= 2:
        return False
    original = deepcopy(genome)
    original_expert_count = len(layer.experts)
    layer.experts = [expert for expert in layer.experts if expert.id != expert_id]
    if len(layer.experts) == original_expert_count:
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


def clone_expert(genome: ArchitectureGenome, layer_id: str, expert_id: str, step: int) -> str | None:
    """Clone an expert gene with a fresh identity for weight-level transfer."""
    return split_expert(genome, layer_id, expert_id, step)


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


def change_module(genome: ArchitectureGenome, layer_id: str, module: str) -> bool:
    """Change a layer's V0 module family while preserving its topology identity."""
    if module not in {"attention", "gdn", "hybrid"}:
        return False
    layer = next((item for item in genome.layers if item.id == layer_id), None)
    if layer is None or layer.module == module:
        return False
    original = deepcopy(genome)
    layer.module = module
    if not validate_genome(genome).valid:
        genome.layers = original.layers
        return False
    return True


def toggle_layer(genome: ArchitectureGenome, layer_id: str, enabled: bool) -> bool:
    """Enable or disable a layer, rejecting genomes with no executable layers."""
    layer = next((item for item in genome.layers if item.id == layer_id), None)
    if layer is None or layer.enabled == enabled:
        return False
    original = deepcopy(genome)
    layer.enabled = enabled
    if not validate_genome(genome).valid:
        genome.layers = original.layers
        return False
    return True