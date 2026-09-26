"""Alternative expert-dispatch formulations for the sparse MoE block, and in-place switches.

``_MoEFeedForward.forward`` is a *dense* expert evaluation: one batched matmul applies every
expert to every token, and the top-k rows are selected afterwards. The routing contract books
only the experts each token activates, so the dense form executes ``num_experts / active_experts``
times the arithmetic - **1.934x** at the code-LM student's shape, which is 48.3% of every forward
pass, because the experts are 68.0% of the parameters. That measurement and its consequences are
in ``docs/training-throughput.md``.

Two things are worth separating here, because they have different status:

**The formulation is settled.** :func:`grouped_forward` routes once, orders the ``(token, slot)``
rows by expert with a single stable sort, runs one contiguous matmul pair per expert, and
scatters the gate-weighted rows back with one ``index_add_``. It is the *same function* as the
dense form: measured at the code-LM shape in float64, they agree to **5.1e-16** with a
bit-identical routing load, which is the tolerance ``tests/test_v0_session.py`` already pins the
dense/``eager_forward`` pair at. ``tests/test_moe_dispatch.py`` holds that, including at the
whole-model level: the logits of a model with every block switched to this dispatcher match the
unswitched model's, and a seeded generation produces the same token ids. Skipped experts also
receive explicit zero gradients without being evaluated, preserving the dense path's optimizer
semantics for weight decay and existing moments.

**Which one to run is not settled, and this module does not settle it.** The only measurement
arguing the dense form's way is a dev-shape figure that implies more than the card's physical
FLOP peak (see the same note), and the crossover between the two forms has to be re-measured at
the shape that matters - that is what ``scripts/moe_dispatch_benchmark.py`` is for. So the
dispatcher is available, exact, and **not on any path by default**: nothing in this package
switches a model over, and a caller that wants it says so with :func:`use_grouped`. That is also
why the switch is in-place and reversible rather than a config field: it does not change what a
model *is*, so it must not appear in a checkpoint, a config, or a report that describes an
architecture - only in the record of how a run was executed.
"""

from __future__ import annotations

import types

import torch
from torch import Tensor, nn
from torch.autograd import Function

from act.model.v0_moe import _MoEFeedForward


class _AttachZeroGrad(Function):
    """Keep skipped parameters' optimizer semantics without evaluating their experts."""

    @staticmethod
    def forward(ctx, output: Tensor, *parameters: Tensor) -> Tensor:
        ctx.parameter_specs = tuple(
            (parameter.shape, parameter.device, parameter.dtype, parameter.requires_grad)
            for parameter in parameters
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        zero_grads = tuple(
            torch.zeros(shape, device=device, dtype=dtype) if requires_grad else None
            for shape, device, dtype, requires_grad in ctx.parameter_specs
        )
        return (grad_output, *zero_grads)


def grouped_forward(moe: _MoEFeedForward, hidden: Tensor) -> Tensor:
    """Top-k routing with only the routed experts evaluated.

    Rows of the flattened ``(token, slot)`` space are ordered by expert, so each expert's work is
    one contiguous slice and one matmul pair; experts that received no rows are skipped rather
    than run on an empty batch. Routing is delegated to ``_route`` so the contract - including
    the recorded ``last_load`` bookkeeping - is shared with the dense and eager forms instead of
    being reimplemented here.
    """
    flat = hidden.reshape(-1, hidden.shape[-1])
    weights, indices, _ = moe._route(flat)
    tokens, k = indices.shape

    rows_of_expert = indices.reshape(-1)
    # Row ``r`` belongs to token ``r // k`` and carries gate weight ``weights[r // k, r % k]``,
    # which is ``weights.reshape(-1)[r]`` because the top-k axis is the minor one.
    row_gate = weights.reshape(-1)
    order = torch.argsort(rows_of_expert, stable=True)
    counts = torch.bincount(rows_of_expert, minlength=len(moe.experts))

    output = torch.zeros_like(flat)
    unused_experts = []
    start = 0
    for expert_index, expert in enumerate(moe.experts):
        count = int(counts[expert_index])
        if count == 0:
            unused_experts.append(expert)
            continue
        rows = order[start : start + count]
        start += count
        expert_output = expert(flat[rows // k])
        output.index_add_(0, rows // k, expert_output * row_gate[rows].unsqueeze(-1))

    # Dense dispatch evaluates every expert, so even an unused expert's parameters receive zero
    # gradients. Optimizers such as AdamW distinguish zero gradients from ``None`` (weight decay
    # and existing moments still update the former). Attach exact zero gradients after all output
    # scatters, without reading or evaluating the skipped expert weights.
    if torch.is_grad_enabled():
        for expert in unused_experts:
            output = _AttachZeroGrad.apply(output, *tuple(expert.parameters()))
    return output.reshape_as(hidden)


def _moe_blocks(model: nn.Module) -> list[_MoEFeedForward]:
    blocks = getattr(model, "blocks", None)
    if not blocks:
        raise ValueError("model has no blocks to switch: expected a V0MoECausalLM-like module")
    moes = [getattr(block, "moe", None) for block in blocks]
    if any(moe is None for moe in moes):
        raise ValueError("a block has no 'moe' module, so its expert dispatch cannot be switched")
    return moes  # type: ignore[return-value]


def use_grouped(model: nn.Module) -> int:
    """Switch every block's expert dispatch to :func:`grouped_forward`, in place.

    Returns the number of blocks switched. Autograd-compatible, so this is usable for training
    as well as inference - but see the module docstring: the choice is a measurement's to make,
    not a default's, and ``scripts/moe_dispatch_benchmark.py`` is where that measurement lives.
    """
    moes = _moe_blocks(model)
    for moe in moes:
        moe.forward = types.MethodType(grouped_forward, moe)
    return len(moes)


def use_dense(model: nn.Module) -> int:
    """Restore the class's dense expert evaluation, undoing :func:`use_grouped`."""
    moes = _moe_blocks(model)
    for moe in moes:
        if "forward" in vars(moe):
            del moe.forward
    return len(moes)


def dispatch_of(model: nn.Module) -> str:
    """Which expert dispatch a model is currently wired for, as a string to record.

    Reported per block rather than per model on purpose: a partially switched model would
    otherwise read as one thing, and the only honest partial answer is to say so.
    """
    moes = _moe_blocks(model)
    switched = sum(1 for moe in moes if "forward" in vars(moe))
    if switched == 0:
        return "dense"
    if switched == len(moes):
        return "grouped"
    return f"mixed:{switched}/{len(moes)}"
