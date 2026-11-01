"""The width mutation: a trained model widened in place, and the perturbation it owes.

:mod:`act.model.widen` builds a *function-preserving* graft. That is the mechanism, not yet a
search operator: a zero-padded graft is a saddle, because every new coordinate starts at exactly
zero gradient (measured in ``tests/test_width_widen.py``). This module supplies the two missing
halves and keeps them separate so either can be tested alone:

* :func:`widen_genome` is the mutation at the genome level. ``hidden_size`` grows by a whole
  number of heads, every attention layer gains those heads, ``head_dim`` is held fixed, and the
  genome's parameter ceilings are re-checked. It touches no tensors and needs no model.
* :func:`perturb_new_coordinates` breaks the saddle. It writes small noise into exactly the
  coordinates the graft created - the new embedding columns, head slots, input columns, output
  rows and norm entries - and nothing else, so the old function is preserved while the new units
  can start receiving gradient.
* :func:`graft_widened_model` is the operator a run would call: widen the genome, widen the
  model, perturb, and return the record a mutation history needs.

The operator is deliberately **grow-only** and **MHA-only**. A shrink is pruning (a different
operator with a different information loss), and a grouped-query or recurrent layer changes
geometry when heads are added, so both are refused rather than approximated - the same choices
:func:`act.model.widen.widen_config` makes, restated at the genome level where the search would
otherwise be free to propose them.
"""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import Tensor

from act.topology.genome import ArchitectureGenome, genome_hash, validate_genome

from .factory import v0_moe_config_from_genome
from .v0_moe import V0MoECausalLM, estimated_parameter_count
from .widen import widen_config, widen_model, widening_report


def _rms(tensor: Tensor) -> float:
    """Root-mean-square of a tensor, used so a norm's constant ones still set a scale."""
    return float(tensor.detach().float().pow(2).mean().sqrt())


def _generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def widen_genome(
    genome: ArchitectureGenome,
    *,
    heads_added: int = 2,
    max_params: int | None = None,
    max_active_params: int | None = None,
) -> ArchitectureGenome | None:
    """The genome side of one width step, or ``None`` when the step is infeasible.

    ``heads_added`` heads are added at the fixed ``head_dim`` the genome already has, so
    ``hidden_size`` grows by ``heads_added * head_dim`` and every attention layer gains the same
    heads. The requirements are the ones the model's construction imposes: every layer attention,
    every layer MHA (``heads == kv_heads``), and one head count across the enabled layers, because
    ``v0_moe_config_from_genome`` carries a single ``attention_heads`` for the whole model. A
    genome that does not meet them gets ``None`` rather than a step the run would later fail on.

    Expert widths are untouched: they are the experts' inner widths, not the residual stream, and
    a width mutation changes the stream the experts read from and write to. The returned genome is
    a copy; the argument is never modified.
    """

    if heads_added < 1:
        raise ValueError("a width step adds at least one head")
    widened = deepcopy(genome)
    enabled = [layer for layer in widened.layers if layer.enabled]
    if not enabled:
        return None
    if any(layer.module != "attention" for layer in widened.layers):
        return None
    if any(layer.heads != layer.kv_heads for layer in widened.layers):
        return None
    if len({layer.heads for layer in enabled}) != 1:
        return None
    head_dim = widened.hidden_size // enabled[0].heads
    if head_dim * enabled[0].heads != widened.hidden_size:
        return None

    widened.hidden_size += heads_added * head_dim
    for layer in widened.layers:
        layer.heads += heads_added
        layer.kv_heads += heads_added
    if not validate_genome(widened).valid:
        return None

    try:
        config = v0_moe_config_from_genome(widened)
    except ValueError:
        return None
    total, active = estimated_parameter_count(config)
    total_cap = widened.max_params if max_params is None else max_params
    active_cap = widened.max_active_params if max_active_params is None else max_active_params
    if total > total_cap or active > active_cap:
        return None
    return widened


def _noise(
    shape: tuple[int, ...],
    *,
    sigma: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    return torch.randn(shape, generator=generator, device=device, dtype=dtype) * sigma


def perturb_new_coordinates(
    old: V0MoECausalLM,
    widened: V0MoECausalLM,
    *,
    scale: float = 1e-3,
    seed: int = 0,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Write small noise into the coordinates the graft created, and only those.

    The scale is relative: each tensor's sigma is ``scale`` times the source tensor's
    root-mean-square, so a norm's constant ones set a scale where a linear weight's spread would
    not. Every coordinate the graft copied is left exactly as the graft left it (norms included,
    which the graft rescaled by ``sqrt(h/h')``); the new ones are drawn from a local
    generator, so a seeded run's global RNG stream is not disturbed (the same discipline
    :func:`act.model.widen.widen_model` follows when it constructs the wider model).

    Returns ``{"coordinates", "max_abs", "scale"}`` - the number of new entries written, the
    largest perturbation, and the relative scale - which is what the graft record quotes.
    """

    if scale < 0:
        raise ValueError("perturbation scale must be non-negative")
    old_hidden = old.config.hidden_size
    new_hidden = widened.config.hidden_size
    old_heads = old.config.attention_heads
    new_heads = widened.config.attention_heads
    if new_hidden <= old_hidden or new_heads <= old_heads:
        raise ValueError("perturb_new_coordinates expects a wider model")
    head_dim = new_hidden // new_heads
    if head_dim != old_hidden // old_heads:
        raise ValueError("the graft must hold head_dim fixed")

    sources = dict(old.named_parameters())
    coordinates = 0
    largest = 0.0
    handled: set[str] = set()
    with torch.no_grad():
        for name, parameter in widened.named_parameters():
            source = sources.get(name)
            if source is None:
                raise ValueError(f"the widened model has a parameter the source lacks: {name}")
            sigma = _rms(source) * scale
            draw = _generator(parameter.device, seed) if generator is None else generator

            def write(block: Tensor) -> None:
                nonlocal coordinates, largest
                coordinates += block.numel()
                largest = max(largest, float(block.abs().max()) if block.numel() else 0.0)

            if name == "token_embedding.weight":
                block = _noise(
                    (parameter.shape[0], new_hidden - old_hidden),
                    sigma=sigma,
                    generator=draw,
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                parameter[:, old_hidden:] = block
                write(block)
            elif name == "final_norm.weight" or name.endswith(
                ("norm_attn.weight", "norm_moe.weight")
            ):
                block = _noise(
                    (new_hidden - old_hidden,),
                    sigma=sigma,
                    generator=draw,
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                parameter[old_hidden:] = block
                write(block)
            elif name.endswith("attention.qkv.weight"):
                view = parameter.view(3, new_heads, head_dim, new_hidden)
                heads_block = _noise(
                    (3, new_heads - old_heads, head_dim, new_hidden),
                    sigma=sigma,
                    generator=draw,
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                columns_block = _noise(
                    (3, old_heads, head_dim, new_hidden - old_hidden),
                    sigma=sigma,
                    generator=draw,
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                view[:, old_heads:] = heads_block
                view[:, :old_heads, :, old_hidden:] = columns_block
                write(heads_block)
                write(columns_block)
            elif name.endswith("attention.out.weight"):
                rows = _noise(
                    (new_hidden - old_hidden, new_hidden),
                    sigma=sigma,
                    generator=draw,
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                columns = _noise(
                    (old_hidden, new_hidden - old_hidden),
                    sigma=sigma,
                    generator=draw,
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                parameter[old_hidden:] = rows
                parameter[:old_hidden, old_hidden:] = columns
                write(rows)
                write(columns)
            elif name.endswith("moe.router.weight"):
                block = _noise(
                    (parameter.shape[0], new_hidden - old_hidden),
                    sigma=sigma,
                    generator=draw,
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                parameter[:, old_hidden:] = block
                write(block)
            elif ".moe.experts." in name and name.endswith(".0.weight"):
                block = _noise(
                    (parameter.shape[0], new_hidden - old_hidden),
                    sigma=sigma,
                    generator=draw,
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                parameter[:, old_hidden:] = block
                write(block)
            elif ".moe.experts." in name and name.endswith(".2.weight"):
                block = _noise(
                    (new_hidden - old_hidden, parameter.shape[1]),
                    sigma=sigma,
                    generator=draw,
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                parameter[old_hidden:] = block
                write(block)
            elif source.shape != parameter.shape:
                raise ValueError(
                    f"{name} changed shape but is not a coordinate this operator knows; "
                    "the graft layout and the perturbation have drifted apart"
                )
            else:
                continue
            handled.add(name)

    # Anything whose shape grew must have been written above. A silent miss would leave part of
    # the new stream at exactly zero gradient - the saddle this function exists to break.
    for name, parameter in widened.named_parameters():
        source = sources[name]
        if source.shape != parameter.shape and name not in handled:
            raise ValueError(f"new coordinates left unperturbed: {name}")
    return {"coordinates": float(coordinates), "max_abs": largest, "scale": scale}


def graft_record(
    old: V0MoECausalLM,
    widened: V0MoECausalLM,
    before: ArchitectureGenome,
    after: ArchitectureGenome,
    perturbation: dict[str, float],
    *,
    heads_added: int,
    seed: int,
) -> dict[str, object]:
    """The mutation-history row for one width step, priced in the units a gate needs."""
    report = widening_report(old, widened)
    return {
        "kind": "widen",
        "operation": "widen",
        "layer_id": None,
        "expert_id": None,
        "heads_added": heads_added,
        "from_hidden": report["from_hidden"],
        "to_hidden": report["to_hidden"],
        "from_heads": report["from_heads"],
        "to_heads": report["to_heads"],
        "head_dim": report["head_dim"],
        "total_before": report["total_before"],
        "total_after": report["total_after"],
        "active_before": report["active_before"],
        "active_after": report["active_after"],
        "total_ratio": report["total_ratio"],
        "active_ratio": report["active_ratio"],
        "new_parameters": report["new_parameters"],
        "new_parameters_share": report["new_parameters_share"],
        "norm_epsilon_before": report["norm_epsilon_before"],
        "norm_epsilon_after": report["norm_epsilon_after"],
        "perturbation_scale": perturbation["scale"],
        "perturbation_coordinates": int(perturbation["coordinates"]),
        "perturbation_max_abs": perturbation["max_abs"],
        "seed": seed,
        "architecture_hash_before": genome_hash(before),
        "architecture_hash_after": genome_hash(after),
    }


def graft_widened_model(
    model: V0MoECausalLM,
    genome: ArchitectureGenome,
    *,
    heads_added: int = 2,
    perturbation_scale: float = 1e-3,
    seed: int = 0,
    generator: torch.Generator | None = None,
) -> tuple[V0MoECausalLM, ArchitectureGenome, dict[str, object]]:
    """Widen a trained model and its genome in one step.

    The returned tuple is ``(model, genome, record)``. The old model is not modified: the graft
    builds a new one, the perturbation writes only into it, and the source genome is copied by
    :func:`widen_genome`. The model and the genome must already agree on width and head count -
    a mismatch would widen the genome one way and the tensors another, and the run's record would
    describe a model it does not have.
    """

    if genome.hidden_size != model.config.hidden_size:
        raise ValueError(
            "the genome and the model disagree on hidden_size: "
            f"{genome.hidden_size} against {model.config.hidden_size}"
        )
    enabled = [layer for layer in genome.layers if layer.enabled]
    if not enabled or enabled[0].heads != model.config.attention_heads:
        raise ValueError(
            "the genome and the model disagree on attention_heads: "
            f"{enabled[0].heads if enabled else None} against {model.config.attention_heads}"
        )
    if len(enabled) != model.config.depth:
        raise ValueError("the genome and the model disagree on depth")

    widened_genome = widen_genome(genome, heads_added=heads_added)
    if widened_genome is None:
        raise ValueError("no feasible width step for this genome")
    config = widen_config(
        model.config,
        hidden_size=widened_genome.hidden_size,
        attention_heads=widened_genome.layers[0].heads,
    )
    expected = v0_moe_config_from_genome(widened_genome)
    if (
        config.hidden_size != expected.hidden_size
        or config.attention_heads != expected.attention_heads
        or config.depth != expected.depth
        or config.experts_per_layer != expected.experts_per_layer
        or config.active_experts != expected.active_experts
        or config.expert_width != expected.expert_width
    ):
        raise ValueError("the widened genome and the widened model disagree on architecture")

    widened = widen_model(model, config)
    perturbation = perturb_new_coordinates(
        model,
        widened,
        scale=perturbation_scale,
        seed=seed,
        generator=generator,
    )
    record = graft_record(
        model,
        widened,
        genome,
        widened_genome,
        perturbation,
        heads_added=heads_added,
        seed=seed,
    )
    return widened, widened_genome, record
