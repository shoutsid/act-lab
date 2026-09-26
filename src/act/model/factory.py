"""Build baseline models from the shared Python topology genome."""

from .dense import DenseBaselineConfig, DenseCausalLM
from .moe import StaticMoECausalLM, StaticMoEConfig
from .v0 import V0CausalLM, V0ModelConfig
from .v0_moe import V0MoECausalLM, V0MoEConfig
from act.topology.genome import ArchitectureGenome, validate_genome
from numbers import Integral


_NEW_EXPERT_INITIALIZATIONS = {"random", "clone", "clone_body"}


def _validate_rebuild_options(
    preserve_router: bool, new_expert_initialization: str
) -> None:
    if new_expert_initialization not in _NEW_EXPERT_INITIALIZATIONS:
        raise ValueError(
            "new_expert_initialization must be random, clone, or clone_body"
        )
    if new_expert_initialization in {"clone", "clone_body"} and not preserve_router:
        raise ValueError(
            f"new_expert_initialization={new_expert_initialization!r} requires preserve_router=True"
        )


def model_from_genome(
    genome: ArchitectureGenome,
    *,
    context_size: int = 32,
    num_heads: int | None = None,
) -> DenseCausalLM | StaticMoECausalLM:
    validation = validate_genome(genome)
    if not validation.valid:
        raise ValueError("cannot build model from invalid genome: " + "; ".join(validation.errors))
    if not isinstance(context_size, Integral) or isinstance(context_size, bool) or context_size < 2:
        raise ValueError("context_size must be an integer of at least 2")
    if num_heads is not None and (
        not isinstance(num_heads, Integral) or isinstance(num_heads, bool) or num_heads < 1
    ):
        raise ValueError("num_heads must be a positive integer")
    enabled = [layer for layer in genome.layers if layer.enabled]
    kinds = {layer.ffn for layer in enabled}
    if len(kinds) != 1:
        raise ValueError("mixed dense/MoE genomes are not supported by the baseline factory")
    heads = num_heads if num_heads is not None else (4 if genome.hidden_size % 4 == 0 else 1)
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
    preserve_router: bool = False,
    new_expert_initialization: str = "random",
) -> DenseCausalLM | StaticMoECausalLM:
    """Build a mutated model while retaining every compatible parameter.

    ``preserve_router`` controls what happens to a mutated layer's router (the
    bias-free ``hidden x experts`` routing head). Its weight shape depends on the
    expert count, so the generic same-shape copy below CANNOT carry it across a
    mutation: by default the whole routing head is therefore re-drawn at random
    whenever that layer's expert count changes, discarding the learned routing
    function while the experts' own weights survive by id. That default is kept
    because it is the behaviour every published study measured. With
    ``preserve_router`` the surviving experts' rows are copied by id and only
    genuinely new experts (a grow) keep the fresh draw - the treatment used to
    test whether the mid-training penalty is this discard (report section 5.24).

    ``new_expert_initialization`` decides what a genuinely NEW expert (a grow)
    starts as:

    - ``"random"``: the rebuild's own draw - the historical behaviour, and the
      only one any published number was measured under.
    - ``"clone"``: a copy of the busiest surviving expert in the same layer, by
      the model's own recorded routing load, with that expert's router row copied
      too. The new expert therefore starts as a working twin of an expert the
      router already sends tokens to, instead of as noise the model must first
      learn to route around (report section 5.26). Cloning the router row is what
      makes it a twin: a cloned expert body with a freshly drawn row would start
      with no routing affinity at all.
    - ``"clone_body"``: the same copy of the busiest survivor's body, but the new
      expert keeps its **freshly drawn router row**. Identical to ``"clone"`` in
      every other respect, so comparing the two separates a trained *body* from
      the *routing affinity* that copying the row confers - the mechanism section
      5.26 declared unresolved (report section 5.27).
    """
    _validate_rebuild_options(preserve_router, new_expert_initialization)
    rebuilt = model_from_genome(genome, context_size=context_size, num_heads=num_heads)
    source_state = previous.state_dict()
    target_state = rebuilt.state_dict()

    expert_prefixes: set[str] = set()
    handled: set[str] = set()
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
            if not preserve_router:
                continue
            router_key = f"blocks.{layer_index}.router.weight"
            if router_key not in source_state or router_key not in target_state:
                continue
            handled.add(router_key)
            # Row i of the router is expert i's routing vector, so a surviving
            # expert keeps its learned row; a grown expert's row stays as drawn.
            for target_index, expert_id in enumerate(target_block.expert_ids):
                source_index = source_ids.get(expert_id)
                if source_index is None:
                    continue
                target_state[router_key][target_index].copy_(
                    source_state[router_key][source_index]
                )
            if new_expert_initialization not in {"clone", "clone_body"}:
                continue
            new_indices = [
                target_index
                for target_index, expert_id in enumerate(target_block.expert_ids)
                if expert_id not in source_ids
            ]
            survivors = sorted(
                source_index
                for expert_id, source_index in source_ids.items()
                if expert_id in set(target_block.expert_ids)
            )
            if not new_indices or not survivors:
                continue
            loads = getattr(previous, "routing_load", None)
            layer_loads = (
                loads[layer_index]
                if loads is not None and layer_index < len(loads)
                else None
            )
            # The busiest survivor is the one whose function is most in demand,
            # so duplicating it is the cheapest place to add capacity. Falls back
            # to the lowest-index survivor when no load has been recorded (a
            # model that has not trained yet).
            source_index = (
                max(survivors, key=lambda index: float(layer_loads[index]))
                if layer_loads is not None
                else survivors[0]
            )
            for target_index in new_indices:
                for suffix in ("0.weight", "0.bias", "2.weight", "2.bias"):
                    target_key = (
                        f"blocks.{layer_index}.experts.{target_index}.{suffix}"
                    )
                    source_key = f"blocks.{layer_index}.experts.{source_index}.{suffix}"
                    if target_key in target_state and source_key in source_state:
                        target_state[target_key].copy_(source_state[source_key])
                if (
                    new_expert_initialization == "clone"
                    and router_key in target_state
                    and router_key in source_state
                ):
                    handled.add(router_key)
                    target_state[router_key][target_index].copy_(
                        source_state[router_key][source_index]
                    )

    for key, value in target_state.items():
        if key in handled:
            continue
        if any(key.startswith(prefix) for prefix in expert_prefixes):
            continue
        if key in source_state and source_state[key].shape == value.shape:
            value.copy_(source_state[key])
    rebuilt.load_state_dict(target_state)
    return rebuilt


def v0_moe_config_from_genome(
    genome: ArchitectureGenome, *, honor_module_types: bool = False
) -> V0MoEConfig:
    """Derive the declared-scale MoE config from a V0 MoE genome.

    ``honor_module_types`` is opt-in. Historically ``layer.module`` was metadata
    on this path and the model always built attention, so honouring it by
    default would silently change the reconstruction of every already-recorded
    genome whose module metadata had been edited by a topology mutation. Static
    architecture search opts in; legacy and churn paths keep the old meaning.
    """
    validation = validate_genome(genome)
    if not validation.valid:
        raise ValueError("cannot build V0 MoE model from invalid genome: " + "; ".join(validation.errors))
    enabled = [layer for layer in genome.layers if layer.enabled]
    if not enabled:
        raise ValueError("V0 MoE genome has no enabled layers")
    if any(layer.ffn != "moe" for layer in enabled):
        raise ValueError("V0 MoE reconstruction requires MoE feed-forward layers")
    widths = {expert.width for layer in enabled for expert in layer.experts}
    active_counts = {layer.active_experts for layer in enabled}
    if len(widths) != 1:
        raise ValueError("V0 MoE reconstruction requires uniform expert widths")
    if len(active_counts) != 1:
        raise ValueError("V0 MoE reconstruction requires uniform active expert counts")
    return V0MoEConfig(
        vocab_size=genome.vocab_size,
        context_size=genome.context_size,
        hidden_size=genome.hidden_size,
        depth=len(enabled),
        attention_heads=enabled[0].heads,
        experts_per_layer=tuple(len(layer.experts) for layer in enabled),
        active_experts=next(iter(active_counts)),
        expert_width=next(iter(widths)),
        expert_ids=tuple(
            tuple(expert.id for expert in layer.experts) for layer in enabled
        ),
        module_types=tuple(layer.module for layer in enabled) if honor_module_types else (),
    )


def v0_moe_model_from_genome(
    genome: ArchitectureGenome,
    *,
    honor_module_types: bool = False,
) -> V0MoECausalLM:
    """Construct a fresh V0 sparse-MoE model matching a genome exactly."""
    config = v0_moe_config_from_genome(genome, honor_module_types=honor_module_types)
    return V0MoECausalLM(config)


def rebuild_v0_moe_from_genome(
    previous: V0MoECausalLM,
    genome: ArchitectureGenome,
    *,
    preserve_router: bool = False,
    new_expert_initialization: str = "random",
) -> V0MoECausalLM:
    """Rebuild a mutated V0 MoE model, copying every retained expert by id.

    Expert parameters are transferred by expert identity (genome ids), not
    slot order, so prune and grow events never shuffle trained weights between
    different experts. Every other parameter copies when shape-compatible.

    ``preserve_router`` and ``new_expert_initialization`` are the declared-path
    counterparts of the same two arguments on ``rebuild_model_from_genome``, and they
    default to the historical behaviour so every committed number keeps its meaning:

    - the router is ``hidden_size x experts_in_layer`` (``_MoEFeedForward.router``),
      so its shape moves with the expert count and the same-shape copy below CANNOT
      carry it. By default the whole routing head is therefore re-drawn at random
      whenever that layer's expert count changes - the section 5.24 defect, which
      the toy path fixed there and which this path still had. With ``preserve_router``
      the surviving experts' rows are copied by id and only genuinely new experts
      (a grow) keep the fresh draw.
    - ``new_expert_initialization`` decides what a genuinely new expert starts as:
      ``"random"`` (the historical draw), ``"clone"`` (a copy of the busiest
      surviving expert in the same layer by ``routing_load``, router row included,
      so the new expert starts as a working twin) or ``"clone_body"`` (the same body
      with a freshly drawn row - the control that separates a trained body from the
      routing affinity the copied row confers). Report sections 5.26/5.27.
    """
    _validate_rebuild_options(preserve_router, new_expert_initialization)
    rebuilt = v0_moe_model_from_genome(genome)
    source_state = previous.state_dict()
    target_state = rebuilt.state_dict()
    expert_prefixes: set[str] = set()
    handled: set[str] = set()
    for layer_index, target_block in enumerate(rebuilt.blocks):
        source_block = previous.blocks[layer_index]
        source_ids = {
            expert_id: index for index, expert_id in enumerate(source_block.moe.expert_ids)
        }
        for target_index, expert_id in enumerate(target_block.moe.expert_ids):
            source_index = source_ids.get(expert_id)
            if source_index is None:
                continue
            target_prefix = f"blocks.{layer_index}.moe.experts.{target_index}."
            source_prefix = f"blocks.{layer_index}.moe.experts.{source_index}."
            expert_prefixes.add(target_prefix)
            for suffix in ("0.weight", "2.weight"):
                target_state[target_prefix + suffix].copy_(
                    source_state[source_prefix + suffix]
                )
        if not preserve_router:
            continue
        router_key = f"blocks.{layer_index}.moe.router.weight"
        if router_key not in source_state or router_key not in target_state:
            continue
        handled.add(router_key)
        # Row i of the router is expert i's routing vector, so a surviving expert
        # keeps its learned row; a grown expert's row stays as drawn until the
        # cloning branch below decides it.
        for target_index, expert_id in enumerate(target_block.moe.expert_ids):
            source_index = source_ids.get(expert_id)
            if source_index is None:
                continue
            target_state[router_key][target_index].copy_(
                source_state[router_key][source_index]
            )
        if new_expert_initialization not in {"clone", "clone_body"}:
            continue
        new_indices = [
            target_index
            for target_index, expert_id in enumerate(target_block.moe.expert_ids)
            if expert_id not in source_ids
        ]
        survivors = sorted(
            source_index
            for expert_id, source_index in source_ids.items()
            if expert_id in set(target_block.moe.expert_ids)
        )
        if not new_indices or not survivors:
            continue
        loads = getattr(previous, "routing_load", None)
        layer_loads = (
            loads[layer_index] if loads is not None and layer_index < len(loads) else None
        )
        # The busiest survivor is the one whose function is most in demand, so
        # duplicating it is the cheapest place to add capacity. Falls back to the
        # lowest-index survivor when no load has been recorded.
        source_index = (
            max(survivors, key=lambda index: float(layer_loads[index]))
            if layer_loads is not None
            else survivors[0]
        )
        for target_index in new_indices:
            for suffix in ("0.weight", "2.weight"):
                target_key = f"blocks.{layer_index}.moe.experts.{target_index}.{suffix}"
                source_key = f"blocks.{layer_index}.moe.experts.{source_index}.{suffix}"
                if target_key in target_state and source_key in source_state:
                    target_state[target_key].copy_(source_state[source_key])
            if new_expert_initialization == "clone":
                target_state[router_key][target_index].copy_(
                    source_state[router_key][source_index]
                )
    for key, value in target_state.items():
        if key in handled:
            continue
        if any(key.startswith(prefix) for prefix in expert_prefixes):
            continue
        if key in source_state and source_state[key].shape == value.shape:
            value.copy_(source_state[key])
    rebuilt.load_state_dict(target_state)
    return rebuilt


def v0_model_from_genome(genome: ArchitectureGenome) -> V0CausalLM:
    """Reconstruct a V0 dense/hybrid model from its complete genome."""
    validation = validate_genome(genome)
    if not validation.valid:
        raise ValueError("cannot build V0 model from invalid genome: " + "; ".join(validation.errors))
    if any(layer.ffn != "dense" for layer in genome.layers if layer.enabled):
        raise ValueError("V0 reconstruction currently requires dense feed-forward layers")
    enabled = [layer for layer in genome.layers if layer.enabled]
    if not enabled:
        raise ValueError("V0 genome has no enabled layers")
    if len({layer.ffn_width for layer in enabled}) != 1:
        raise ValueError("V0 reconstruction requires uniform feed-forward widths")
    module_types = tuple(layer.module for layer in enabled)
    return V0CausalLM(
        V0ModelConfig(
            vocab_size=genome.vocab_size,
            context_size=genome.context_size,
            hidden_size=genome.hidden_size,
            depth=len(enabled),
            attention_heads=enabled[0].heads,
            ffn_width=enabled[0].ffn_width,
            module_types=module_types,
        )
    )