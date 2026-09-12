"""Build baseline models from the shared Python topology genome."""

from .dense import DenseBaselineConfig, DenseCausalLM
from .moe import StaticMoECausalLM, StaticMoEConfig
from act.topology.genome import ArchitectureGenome, validate_genome


def model_from_genome(
    genome: ArchitectureGenome,
    *,
    context_size: int = 32,
    num_heads: int | None = None,
) -> DenseCausalLM | StaticMoECausalLM:
    validation = validate_genome(genome)
    if not validation.valid:
        raise ValueError("cannot build model from invalid genome: " + "; ".join(validation.errors))
    enabled = [layer for layer in genome.layers if layer.enabled]
    kinds = {layer.ffn for layer in enabled}
    if len(kinds) != 1:
        raise ValueError("mixed dense/MoE genomes are not supported by the baseline factory")
    heads = num_heads or (4 if genome.hidden_size % 4 == 0 else 1)
    if genome.hidden_size % heads != 0:
        raise ValueError("num_heads must divide hidden_size")
    if "dense" in kinds:
        return DenseCausalLM(
            DenseBaselineConfig(
                vocab_size=genome.vocab_size,
                context_size=context_size,
                hidden_size=genome.hidden_size,
                num_layers=len(enabled),
                num_heads=heads,
                ffn_size=genome.hidden_size * 4,
            )
        )
    first = enabled[0]
    expert_widths = {expert.width for layer in enabled for expert in layer.experts}
    active_counts = {layer.active_experts for layer in enabled}
    if len(expert_widths) != 1 or len(active_counts) != 1:
        raise ValueError("baseline MoE factory requires uniform expert widths and active counts")
    return StaticMoECausalLM(
        StaticMoEConfig(
            vocab_size=genome.vocab_size,
            context_size=context_size,
            hidden_size=genome.hidden_size,
            num_layers=len(enabled),
            num_heads=heads,
            ffn_size=next(iter(expert_widths)),
            num_experts=len(first.experts),
            active_experts=next(iter(active_counts)),
            expert_ids=tuple(
                tuple(expert.id for expert in layer.experts) for layer in enabled
            ),
        )
    )


def rebuild_model_from_genome(
    previous: DenseCausalLM | StaticMoECausalLM,
    genome: ArchitectureGenome,
    *,
    context_size: int = 32,
    num_heads: int | None = None,
) -> DenseCausalLM | StaticMoECausalLM:
    """Build a mutated model while retaining every compatible parameter."""
    rebuilt = model_from_genome(genome, context_size=context_size, num_heads=num_heads)
    source_state = previous.state_dict()
    target_state = rebuilt.state_dict()

    expert_prefixes: set[str] = set()
    if isinstance(previous, StaticMoECausalLM) and isinstance(rebuilt, StaticMoECausalLM):
        for layer_index, target_block in enumerate(rebuilt.blocks):
            source_block = previous.blocks[layer_index]
            source_ids = {expert_id: index for index, expert_id in enumerate(source_block.expert_ids)}
            for target_index, expert_id in enumerate(target_block.expert_ids):
                source_index = source_ids.get(expert_id)
                if source_index is None:
                    continue
                target_prefix = f"blocks.{layer_index}.experts.{target_index}."
                source_prefix = f"blocks.{layer_index}.experts.{source_index}."
                expert_prefixes.add(target_prefix)
                for suffix in ("0.weight", "0.bias", "2.weight", "2.bias"):
                    target_key = target_prefix + suffix
                    source_key = source_prefix + suffix
                    target_state[target_key].copy_(source_state[source_key])

    for key, value in target_state.items():
        if any(key.startswith(prefix) for prefix in expert_prefixes):
            continue
        if key in source_state and source_state[key].shape == value.shape:
            value.copy_(source_state[key])
    rebuilt.load_state_dict(target_state)
    return rebuilt