"""Function-preserving width growth for the V0 decoder, and the contract it is held to.

The architecture search's one replicated finding is about the residual stream: width pays
(``-0.1836`` nats) and depth costs (``+0.0783``, shallow beating deep), both at an enumeration
floor. The adaptive controller cannot reach either gene - it can only move expert capacity, which
two studies measure as inert or worse - and the reason is in ``docs/width-control-design.md``:
**100% of the parameters change shape with ``hidden_size``**, so a width mutation applied to a
partly trained model destroys it rather than redirecting it.

This module is the mechanism that removes that obstacle. :func:`widen_model` returns a model of
the requested width that computes *the same function* as the one handed in: the new coordinates
are zero, the new heads contribute nothing, and the one term that cannot be zero - RMSNorm's
scale, which divides by the root-mean-square over the **new** width - is compensated exactly by
one scalar per norm. The measurements behind that claim, and the two places where "same function"
is exact only to rounding, are in ``tests/test_width_widen.py``.

Three things this deliberately is not:

* **not a pruning operator.** ``hidden_size`` may only grow here; shrinking a trained model onto
  new coordinates is a different operation with a different loss of information, and slipping
  both under one name is how a mutation budget gets spent on the wrong one.
* **not a mixed-architecture operator.** A ``gdn`` or ``hybrid`` layer's recurrence has a
  ``head_dim x head_dim`` state, and with ``head_dim`` held fixed a width step adds heads, so the
  recurrence's geometry changes and zero-padding does not preserve it. Such a model is refused
  rather than widened approximately - the same choice the KV-cache work made, and for the same
  reason.
* **not a search operator on its own.** It produces one exact graft. Whether a *partly trained*
  model survives the round trip is a separate measurement, and the gate's pre-registration is
  where that is decided - see the design note's section 5.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from act.model.v0_moe import V0MoEConfig, V0MoECausalLM, estimated_parameter_count


def widen_config(
    config: V0MoEConfig,
    *,
    hidden_size: int,
    attention_heads: int | None = None,
) -> V0MoEConfig:
    """The wider configuration, with ``head_dim`` held fixed unless asked otherwise.

    ``attention_heads`` defaults to the count that keeps ``head_dim`` unchanged, which is the only
    case this module can preserve exactly: the old heads map onto the new tensor's first head
    slots, so their output columns are reused verbatim. Growing the residual stream *without*
    adding heads means growing ``head_dim`` instead, which changes the rotary geometry and the
    attention's arithmetic - a different attention, not a wider one - so it is refused.
    """
    if hidden_size <= config.hidden_size:
        raise ValueError(
            "widen_config grows a residual stream; "
            f"asked for {hidden_size} from {config.hidden_size} (pruning is a different operator)"
        )
    if any(module != "attention" for module in (config.module_types or ("attention",) * config.depth)):
        raise ValueError(
            "a recurrent (gdn/hybrid) layer's state is sized by head_dim, so a width step cannot "
            "be made function-preserving for it; this operator refuses rather than approximates"
        )
    head_dim = config.hidden_size // config.attention_heads
    heads = attention_heads if attention_heads is not None else hidden_size // head_dim
    if hidden_size % heads:
        raise ValueError(f"hidden_size {hidden_size} is not divisible by {heads} heads")
    if hidden_size // heads != head_dim:
        raise ValueError(
            f"head_dim would change ({head_dim} -> {hidden_size // heads}); holding it fixed is "
            "what lets the old heads be copied rather than re-derived"
        )
    # **The epsilon is part of the compensation, and this is the only field here that is not a
    # shape.** _RMSNorm computes ``x * rsqrt(mean(x^2) + eps) * w``, and for ``x' = [x, 0]`` the
    # wider mean is ``mean(x^2) * h/h'`` - so scaling the weight alone leaves
    # ``rsqrt(a/h' + eps)`` against ``rsqrt(a/h + eps)``, which differ by a factor that depends on
    # the token's own ``a = ||x||^2``. Scaling the epsilon by the same ``h/h'`` makes the two
    # radicands proportional for **every** input, which is what turns a per-batch agreement into
    # an identity. Measured before it was compensated: the logits moved by 5.8e-4, i.e. a norm
    # guard of 1e-6 is not negligible against a fresh model's activation scale.
    return V0MoEConfig(
        vocab_size=config.vocab_size,
        context_size=config.context_size,
        hidden_size=hidden_size,
        depth=config.depth,
        attention_heads=heads,
        experts_per_layer=config.experts_per_layer,
        active_experts=config.active_experts,
        expert_width=config.expert_width,
        rotary_base=config.rotary_base,
        norm_epsilon=config.norm_epsilon * (config.hidden_size / hidden_size),
        expert_ids=config.expert_ids,
        module_types=config.module_types,
        gdn_chunk_size=config.gdn_chunk_size,
    )


def _widen_norm(old: Tensor, new: Tensor) -> Tensor:
    """Copy an RMSNorm weight, then compensate for the wider root-mean-square.

    ``mean(x'^2)`` for ``x' = [x, 0]`` is ``mean(x^2) * h/h'``, so the normalised coordinates come
    out scaled by ``sqrt(h'/h)`` and scaling the copied weight by ``sqrt(h/h')`` cancels it. The
    new coordinates stay zero because their weights are zero. This is one half of the
    compensation - the other half is the epsilon, which :func:`widen_config` scales and which is
    what makes the pair an identity for every input rather than for one batch.
    """
    old_hidden = old.shape[0]
    new_hidden = new.shape[0]
    widened = torch.zeros_like(new)
    widened[:old_hidden] = old * math.sqrt(old_hidden / new_hidden)
    return widened


def _widen_linear(old: Tensor, new: Tensor) -> Tensor:
    """A bias-free ``Linear``'s weight placed in the new tensor's top-left block.

    ``weight`` is ``(out, in)``, so this copies every old output unit and every old input
    coordinate into the coordinates they keep, and leaves all new rows and columns zero: a new
    input coordinate contributes nothing to any old output, and a new output unit emits nothing.
    That covers the input projection, the output projection, the router and both expert matrices,
    because each of them only ever grows on one side.
    """
    widened = torch.zeros_like(new)
    widened[: old.shape[0], : old.shape[1]] = old
    return widened


def _widen_heads(old_qkv: Tensor, new_qkv: Tensor, heads: int, head_dim: int) -> Tensor:
    """Copy a fused qkv projection with the old *head slots* placed, not the old rows.

    ``qkv`` reshapes to ``(batch, seq, 3, heads, head_dim)``, so its rows are grouped by q/k/v and
    then by head. Widening the stream adds heads, which means the old ``q`` block of ``heads *
    head_dim`` rows lands in the new tensor's first ``heads`` slots of the ``q`` group - not in its
    first ``heads * head_dim`` rows. Copying rows directly would silently permute q into k.
    """
    old_hidden = old_qkv.shape[0] // 3
    old_heads = old_hidden // head_dim
    widened = torch.zeros_like(new_qkv)
    old = old_qkv.reshape(3, old_heads, head_dim, -1)
    target = widened.reshape(3, heads, head_dim, -1)
    target[:, :old_heads, :, : old.shape[-1]] = old
    return widened


def widen_model(model: V0MoECausalLM, new_config: V0MoEConfig) -> V0MoECausalLM:
    """A wider model that computes the same function as ``model``.

    Every copied parameter is placed so that the old computation is reproduced exactly and every
    new coordinate is a no-op: zero input columns, zero output rows, zero new head slots, and the
    one scalar-per-norm compensation :func:`_widen_norm` describes. Nothing is initialised from
    noise, which also means the new parameters start at exactly zero gradient - measured in
    ``tests/test_width_widen.py``, and the reason a search operator built on this owes the new
    coordinates a perturbation rather than a training step.
    """
    old_config = model.config
    if (
        new_config.vocab_size != old_config.vocab_size
        or new_config.context_size != old_config.context_size
        or new_config.depth != old_config.depth
        or new_config.experts_per_layer != old_config.experts_per_layer
        or new_config.active_experts != old_config.active_experts
        or new_config.expert_width != old_config.expert_width
    ):
        raise ValueError("widen_model changes hidden width and head count only")
    if new_config.hidden_size <= old_config.hidden_size:
        raise ValueError("widen_model grows a residual stream; it does not shrink one")
    if new_config.attention_heads <= old_config.attention_heads:
        raise ValueError("a width step here must add heads: head_dim is held fixed")

    expected_config = widen_config(
        old_config,
        hidden_size=new_config.hidden_size,
        attention_heads=new_config.attention_heads,
    )
    if new_config != expected_config:
        raise ValueError("widen_model changes hidden width and head count only")

    head_dim = new_config.hidden_size // new_config.attention_heads
    reference_parameter = next(model.parameters())
    # Construction initializes every weight from a seeded distribution, and every one of them is
    # then written below - so the init is irrelevant to the result and must not consume the
    # caller's RNG stream, which a search that widens inside a seeded run would notice. Match the
    # source model's dtype and device before copying: otherwise widening a float64/CUDA model
    # silently returns a float32/CPU model, changing both the function and whether it can run.
    with torch.random.fork_rng():
        wider = V0MoECausalLM(new_config).to(
            device=reference_parameter.device, dtype=reference_parameter.dtype
        )

    with torch.no_grad():
        # Tied input and output embedding: the new hidden coordinates are zero, which is what
        # makes the widened residual stream a no-op for the logits.
        wider.token_embedding.weight.zero_()
        wider.token_embedding.weight[:, : old_config.hidden_size] = model.token_embedding.weight

        wider.final_norm.weight.copy_(_widen_norm(model.final_norm.weight, wider.final_norm.weight))

        for old_block, new_block in zip(model.blocks, wider.blocks):
            new_block.norm_attn.weight.copy_(
                _widen_norm(old_block.norm_attn.weight, new_block.norm_attn.weight)
            )
            new_block.norm_moe.weight.copy_(
                _widen_norm(old_block.norm_moe.weight, new_block.norm_moe.weight)
            )
            new_block.attention.qkv.weight.copy_(
                _widen_heads(
                    old_block.attention.qkv.weight,
                    new_block.attention.qkv.weight,
                    new_config.attention_heads,
                    head_dim,
                )
            )
            new_block.attention.out.weight.copy_(
                _widen_linear(old_block.attention.out.weight, new_block.attention.out.weight)
            )
            new_block.moe.router.weight.copy_(
                _widen_linear(old_block.moe.router.weight, new_block.moe.router.weight)
            )
            for old_expert, new_expert in zip(old_block.moe.experts, new_block.moe.experts):
                new_expert[0].weight.copy_(
                    _widen_linear(old_expert[0].weight, new_expert[0].weight)
                )
                new_expert[2].weight.copy_(
                    _widen_linear(old_expert[2].weight, new_expert[2].weight)
                )
            new_block.moe.last_load = old_block.moe.last_load.clone()

    wider.activation_checkpointing = model.activation_checkpointing
    wider.train(model.training)
    return wider


def widening_report(model: V0MoECausalLM, wider: V0MoECausalLM) -> dict:
    """The cost of one graft, in the units the pre-registration has to carry."""
    old_total, old_active = estimated_parameter_count(model.config)
    new_total, new_active = estimated_parameter_count(wider.config)
    return {
        "from_hidden": model.config.hidden_size,
        "to_hidden": wider.config.hidden_size,
        "from_heads": model.config.attention_heads,
        "to_heads": wider.config.attention_heads,
        "head_dim": wider.config.hidden_size // wider.config.attention_heads,
        "norm_epsilon_before": model.config.norm_epsilon,
        "norm_epsilon_after": wider.config.norm_epsilon,
        "total_before": old_total,
        "total_after": new_total,
        "active_before": old_active,
        "active_after": new_active,
        "total_ratio": new_total / old_total,
        "active_ratio": new_active / old_active,
        "new_parameters": new_total - old_total,
        "new_parameters_share": (new_total - old_total) / new_total,
    }
