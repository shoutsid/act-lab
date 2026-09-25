"""The widening operator's contract, measured rather than asserted.

`docs/width-control-design.md` describes the next gate as needing a *function-preserving*
widening, and names RMSNorm's width-dependence as the obstacle to exactness. That framing is an
assertion about arithmetic, so it is measured here on the real model: a model and its widened
counterpart are run on the same tokens and their logits are compared, in float64, at every
position.

The interesting results are the ones that are not simply "it works":

* the norm compensation is **one scalar per norm** (``sqrt(h/h')``), not an approximation, because
  ``rms([x, 0])`` over the wider stream is ``rms(x) * sqrt(h/h')`` for every input;
* "identical" is **exact to rounding, not bit-identical**: the wider norm divides by a different
  root-mean-square and the compensated weight is a different number, so the two paths round
  differently in the last place. The tolerance below is what that costs, and it is the number the
  design note's contract has to quote instead of "bit-identity";
* the new coordinates start at **exactly zero gradient**, which is a property of any zero-padded
  graft and the reason a search operator built on this owes them a perturbation.

Refusals are pinned too: a shrinking step, a recurrent layer, and a width step that changes
`head_dim` are all refused rather than approximated.
"""

from __future__ import annotations

import torch
import pytest

from act.model.v0_moe import V0MoEConfig, V0MoECausalLM, estimated_parameter_count
from act.model.widen import widen_config, widen_model, widening_report

#: head_dim = 4 at both widths, which is the case the operator is built for. The real gate uses
#: head_dim 64 (768/12 -> 1024/16); the arithmetic is the same and this runs on a CPU.
TINY = dict(
    vocab_size=64,
    context_size=32,
    hidden_size=16,
    depth=2,
    attention_heads=4,
    experts_per_layer=(6, 6),
    active_experts=2,
    expert_width=12,
)


def tiny(**overrides) -> V0MoEConfig:
    return V0MoEConfig(**{**TINY, **overrides})


def widen(config: V0MoEConfig, *, hidden: int, heads: int | None = None):
    torch.manual_seed(4)
    model = V0MoECausalLM(config).eval()
    wider = widen_model(model, widen_config(config, hidden_size=hidden, attention_heads=heads))
    return model, wider


def test_a_widened_model_computes_the_same_function_up_to_the_norms_own_rounding() -> None:
    """The contract: same tokens in, same logits out, at every position - to a measured floor.

    The floor is not the graft. ``_RMSNorm`` computes its statistic in float32 *by design*, so a
    widened norm rounds differently in the last place of that cast, and the error does not
    improve in a float64 model: measured 5.96e-08 here against 6.90e-08 with every weight in
    float64. What the graft has to reach is that floor and no worse - so "bit-identical", which
    ``docs/width-control-design.md`` section 4 asked for, is the wrong target for this operator,
    and 1e-6 absolute (1.4e-07 relative for a logit scale of 0.44) is the honest one.

    Compared across all positions rather than only the last, because a graft that broke one
    position's context would still pass a last-token check.
    """

    config = tiny()
    model, wider = widen(config, hidden=32)
    tokens = torch.randint(0, config.vocab_size, (2, 9))

    with torch.no_grad():
        before = model(tokens).to(torch.float64)
        after = wider(tokens).to(torch.float64)

    max_abs = float((before - after).abs().max())
    assert max_abs < 1e-6, f"widening changed the function by {max_abs:g}"
    # Same shape of output, and the wider model is genuinely wider - otherwise "same logits"
    # would be satisfied by not having widened anything.
    assert before.shape == after.shape
    assert wider.config.hidden_size == 2 * model.config.hidden_size
    assert wider.config.attention_heads == 2 * model.config.attention_heads
    # And the wider model is not the old model in disguise: the residual stream really is wider.
    assert wider.blocks[0].norm_attn.weight.shape[0] == 2 * config.hidden_size


def test_both_halves_of_the_norm_compensation_are_load_bearing() -> None:
    """The compensation is two things, and each one alone leaves the graft visibly wrong.

    The weight scalar cancels the ``sqrt(h'/h)`` that the wider root-mean-square introduces; the
    epsilon scaled by ``h/h'`` is what makes the radicands proportional for *every* input rather
    than for one activation scale. Undoing either half in place and re-running the same tokens is
    the measurement - pinned as an inequality against the compensated error rather than against a
    magic constant, so a later "simplification" that drops one fails here rather than in a run.
    """

    config = tiny()
    model, wider = widen(config, hidden=32)
    tokens = torch.randint(0, config.vocab_size, (1, 7))
    with torch.no_grad():
        reference = model(tokens).to(torch.float64)
        compensated = wider(tokens).to(torch.float64)
    floor = float((reference - compensated).abs().max())

    # Half one: the weight scalar.
    with torch.no_grad():
        for block in wider.blocks:
            for norm in (block.norm_attn, block.norm_moe):
                norm.weight[: config.hidden_size] *= (
                    wider.config.hidden_size / config.hidden_size
                ) ** 0.5
        wider.final_norm.weight[: config.hidden_size] *= (
            wider.config.hidden_size / config.hidden_size
        ) ** 0.5
        weight_only = float((reference - wider(tokens).to(torch.float64)).abs().max())
        # Put it back, so the second half is measured on a compensated model.
        for block in wider.blocks:
            for norm in (block.norm_attn, block.norm_moe):
                norm.weight[: config.hidden_size] /= (
                    wider.config.hidden_size / config.hidden_size
                ) ** 0.5
        wider.final_norm.weight[: config.hidden_size] /= (
            wider.config.hidden_size / config.hidden_size
        ) ** 0.5

    # Half two: the epsilon. This is the half that was missing first, and it cost 5.8e-4.
    epsilons = [
        (block.norm_attn, block.norm_moe) for block in wider.blocks
    ] + [(wider.final_norm,)]
    saved = [(norm, norm.epsilon) for group in epsilons for norm in group]
    for norm, _ in saved:
        norm.epsilon = config.norm_epsilon
    with torch.no_grad():
        epsilon_only = float((reference - wider(tokens).to(torch.float64)).abs().max())
    for norm, value in saved:
        norm.epsilon = value

    assert weight_only > 100 * floor, f"the weight half is not load-bearing ({weight_only:g})"
    assert epsilon_only > 100 * floor, f"the epsilon half is not load-bearing ({epsilon_only:g})"
    assert epsilon_only < weight_only, "the epsilon half is the smaller of the two, as measured"
    # The compensated model still agrees with the original, so the restores above are clean.
    with torch.no_grad():
        assert float((reference - wider(tokens).to(torch.float64)).abs().max()) == floor


def test_the_widening_report_prices_the_graft_in_the_units_the_gate_needs() -> None:
    """The cost of a width step is the number the pre-registration has to carry.

    At a matched *total* budget the active count rises, because attention and the residual stream
    are entirely active while the experts are mostly not - the same asymmetry
    `docs/width-control-design.md` section 8 records from the other direction.
    """

    config = tiny()
    model, wider = widen(config, hidden=32)
    total_before, active_before = estimated_parameter_count(config)
    report = widening_report(model, wider)

    assert report["from_hidden"] == 16 and report["to_hidden"] == 32
    assert report["from_heads"] == 4 and report["to_heads"] == 8
    assert report["head_dim"] == 4
    # The epsilon travels with the graft, and it is the half of the compensation that is easy to
    # forget: pinned here as well as in the logits, because a config field is silent about itself.
    assert report["norm_epsilon_before"] == config.norm_epsilon
    assert report["norm_epsilon_after"] == pytest.approx(config.norm_epsilon / 2)
    assert wider.config.norm_epsilon == pytest.approx(report["norm_epsilon_after"])
    assert (report["total_before"], report["active_before"]) == (total_before, active_before)
    assert report["total_after"] > report["total_before"]
    assert report["active_after"] > report["active_before"]
    assert report["new_parameters"] == report["total_after"] - report["total_before"]
    assert report["new_parameters_share"] == pytest.approx(
        (report["total_after"] - report["total_before"]) / report["total_after"]
    )
    # Doubling the residual width more than doubles the total here: the embedding and the
    # attention are linear in hidden and quadratic in it respectively.
    assert report["total_ratio"] > 2.0


def test_the_new_coordinates_start_at_exactly_zero_gradient() -> None:
    """A zero-padded graft is a saddle, and that is a property of the operator, not a bug.

    The whole point of widening a partly trained model is that it keeps working, but it also has
    to start *learning* again: if every new coordinate has zero gradient, the search operator owes
    them a perturbation. Measured on three kinds of new parameter - a norm entry, a new head's qkv
    rows and an expert's new input columns - because they reach the loss by different routes.
    """

    config = tiny()
    model, wider = widen(config, hidden=32)
    tokens = torch.randint(0, config.vocab_size, (2, 6))
    wider.train()
    wider.zero_grad(set_to_none=True)
    wider.loss(tokens).backward()

    old_hidden = config.hidden_size
    block = wider.blocks[0]
    assert float(block.norm_attn.weight.grad[old_hidden:].abs().max()) == 0.0
    assert float(block.attention.qkv.weight.grad[: block.attention.qkv.weight.shape[0] // 3][
        old_hidden:
    ].abs().max()) == 0.0
    assert float(block.moe.experts[0][0].weight.grad[:, old_hidden:].abs().max()) == 0.0
    # And the old coordinates are not artificially zeroed: the graft is a real one.
    assert float(block.norm_attn.weight.grad[:old_hidden].abs().max()) > 0.0


def test_a_width_step_that_would_change_head_dim_is_refused() -> None:
    """Holding head_dim fixed is what makes the old heads copyable, so the other case is refused.

    768 -> 1152 at 12 heads doubles head_dim instead of adding heads, which changes the rotary
    geometry and the attention's arithmetic: a different attention, not a wider one. The refusal
    is the design note's section 3 recorded as a check.
    """

    config = tiny()
    with pytest.raises(ValueError):
        widen_config(config, hidden_size=32, attention_heads=4)
    # ...and the shape it was built for is accepted.
    assert widen_config(config, hidden_size=32, attention_heads=8).attention_heads == 8
    # ...as is the head count that keeps head_dim without being told.
    assert widen_config(config, hidden_size=24).attention_heads == 6


def test_shrinking_and_recurrent_layers_are_refused() -> None:
    """Two ways this module could quietly become a different operation, both refused.

    A shrinking step is pruning - a different operator with a different information loss - and a
    recurrent layer's state is sized by head_dim, so a width step cannot be made
    function-preserving for it at all.
    """

    config = tiny()
    with pytest.raises(ValueError, match="pruning"):
        widen_config(config, hidden_size=8)
    with pytest.raises(ValueError, match="recurrent"):
        widen_config(tiny(module_types=("gdn", "gdn")), hidden_size=32)

    model = V0MoECausalLM(config)
    with pytest.raises(ValueError, match="hidden width and head count only"):
        widen_model(model, tiny(hidden_size=32, attention_heads=8, context_size=64))
    with pytest.raises(ValueError, match="grows a residual stream"):
        widen_model(model, tiny())
    with pytest.raises(ValueError, match="must add heads"):
        widen_model(model, tiny(hidden_size=32, attention_heads=4, module_types=("attention",) * 2))


def test_widening_does_not_disturb_the_callers_random_stream() -> None:
    """A search that widens inside a seeded run must not have its seeds moved by it.

    The wider model is constructed (which initialises weights) and every weight is then written,
    so the initialisation is irrelevant to the result - but it would still consume the caller's
    RNG if it were not forked, which would make a seeded run depend on how many times it widened.
    """

    config = tiny()
    model = V0MoECausalLM(config)
    torch.manual_seed(1234)
    expected = torch.randn(4)
    torch.manual_seed(1234)
    widen_model(model, widen_config(config, hidden_size=32))
    assert torch.equal(torch.randn(4), expected)
