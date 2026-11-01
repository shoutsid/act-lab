"""The width mutation's contract: the genome step, the perturbation, and the record.

`docs/width-control-design.md` names the remaining work as "a mutation that widens a partly
trained model, records the graft in the run's own signature, and perturbs the new coordinates,
which start at exactly zero gradient". Each of those is a separate claim, so each is measured
here on a tiny CPU config:

* the genome step holds ``head_dim`` fixed, raises every layer's heads together, and refuses the
  shapes the model cannot widen exactly (recurrent layers, grouped-query attention, a step that
  would cross the genome's own parameter ceilings);
* the perturbation writes only into the coordinates the graft created - every old coordinate is
  bit-identical - and it makes the new ones receive gradient, which the unperturbed graft does
  not;
* the record carries the price in the units the pre-registration has to quote.

The real-shape graft (768 -> 1024) is the same arithmetic; these tests are CPU-only by design so
nothing here needs a GPU or a Kaggle session.
"""

from __future__ import annotations

import pytest
import torch

from act.model.factory import v0_moe_config_from_genome, v0_moe_model_from_genome
from act.model.v0_moe import estimated_parameter_count
from act.model.width_growth import (
    graft_widened_model,
    perturb_new_coordinates,
    widen_genome,
)
from act.topology.genome import create_genome, validate_genome


def genome(**overrides):
    """A tiny MoE genome: hidden 16, four heads, head_dim 4, twelve experts in two layers."""
    return create_genome(
        11,
        hidden_size=16,
        depth=2,
        ffn="moe",
        expert_count=6,
        active_experts=2,
        **overrides,
    )


def graft(*, scale: float = 1e-3):
    source = genome()
    model = v0_moe_model_from_genome(source).eval()
    tokens = torch.randint(0, source.vocab_size, (2, 7))
    wider, widened, record = graft_widened_model(
        model, source, perturbation_scale=scale, seed=3
    )
    return source, model, tokens, wider, widened, record


def test_the_genome_step_holds_head_dim_and_updates_every_layer() -> None:
    """Hidden grows by whole heads, not by a free scalar; every layer must agree."""
    source = genome()
    widened = widen_genome(source, heads_added=2)

    assert widened is not None
    assert widened.hidden_size == 24
    assert {layer.heads for layer in widened.layers} == {6}
    assert {layer.kv_heads for layer in widened.layers} == {6}
    # head_dim is the quantity the graft's exactness rests on.
    assert widened.hidden_size // widened.layers[0].heads == 16 // 4
    assert validate_genome(widened).valid
    # Expert widths are the experts' inner widths, not the residual stream.
    assert {expert.width for layer in widened.layers for expert in layer.experts} == {16}

    config = v0_moe_config_from_genome(widened)
    assert (config.hidden_size, config.attention_heads, config.depth) == (24, 6, 2)

    # A copy, never an in-place edit: the source is what the run's current model was built from.
    assert source.hidden_size == 16
    assert {layer.heads for layer in source.layers} == {4}


def test_the_genome_step_refuses_what_the_model_cannot_widen_exactly() -> None:
    """Every refusal here is a shape the operator would otherwise have to approximate."""
    recurrent = genome()
    recurrent.layers[0].module = "gdn"
    assert widen_genome(recurrent) is None

    grouped = genome()
    grouped.layers[0].kv_heads = 2  # 4 % 2 == 0, so the genome itself is legal
    assert widen_genome(grouped) is None

    with pytest.raises(ValueError):
        widen_genome(genome(), heads_added=0)


def test_a_step_that_would_cross_the_genomes_ceilings_is_refused() -> None:
    """The ceilings are the genome's own; a step over either one is not a candidate."""
    source = genome()
    total, active = estimated_parameter_count(v0_moe_config_from_genome(source))
    source.max_params = total
    source.max_active_params = active
    assert widen_genome(source, heads_added=2) is None
    # ...and the untightened genome passes, so the refusal is the ceiling and not the shape.
    assert widen_genome(genome(), heads_added=2) is not None


def test_the_graft_with_no_perturbation_still_computes_the_source_function() -> None:
    """`graft_widened_model` is the wrapper around `widen_model`; the wrapper adds no error."""
    _, model, tokens, wider, _, _ = graft(scale=0.0)
    with torch.no_grad():
        before = model(tokens).to(torch.float64)
        after = wider(tokens).to(torch.float64)
    assert float((before - after).abs().max()) < 1e-6


def test_the_perturbation_touches_only_the_coordinates_the_graft_created() -> None:
    """Copied coordinates stay as the graft left them; only the new ones are written.

    The comparison is against the *unperturbed graft*, not the source model: the graft itself
    rescales every copied norm weight by ``sqrt(h/h')``, and that is function preservation, not
    the perturbation. What this test pins is the second operation's blast radius.
    """
    source = genome()
    model = v0_moe_model_from_genome(source).eval()
    snapshot = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    unperturbed, _, _ = graft_widened_model(model, source, perturbation_scale=0.0, seed=3)
    perturbed, widened, record = graft_widened_model(
        model, source, perturbation_scale=1e-3, seed=3
    )
    baseline = dict(unperturbed.named_parameters())
    old_hidden, new_hidden = 16, 24
    old_heads, new_heads = 4, 6
    head_dim = 4

    with torch.no_grad():
        for name, parameter in perturbed.named_parameters():
            before = baseline[name]
            if name == "token_embedding.weight":
                assert torch.equal(parameter[:, :old_hidden], before[:, :old_hidden]), name
                assert float(before[:, old_hidden:].abs().max()) == 0, name
                assert float(parameter[:, old_hidden:].abs().max()) > 0, name
            elif name == "final_norm.weight" or name.endswith(
                ("norm_attn.weight", "norm_moe.weight")
            ):
                assert torch.equal(parameter[:old_hidden], before[:old_hidden]), name
                assert float(before[old_hidden:].abs().max()) == 0, name
                assert float(parameter[old_hidden:].abs().max()) > 0, name
            elif name.endswith("attention.qkv.weight"):
                new_view = parameter.view(3, new_heads, head_dim, new_hidden)
                old_view = before.view(3, new_heads, head_dim, new_hidden)
                assert torch.equal(
                    new_view[:, :old_heads, :, :old_hidden],
                    old_view[:, :old_heads, :, :old_hidden],
                ), name
                assert float(old_view[:, old_heads:].abs().max()) == 0, name
                assert float(old_view[:, :old_heads, :, old_hidden:].abs().max()) == 0, name
                assert float(new_view[:, old_heads:].abs().max()) > 0, name
                assert float(new_view[:, :old_heads, :, old_hidden:].abs().max()) > 0, name
            elif name.endswith("attention.out.weight"):
                assert torch.equal(
                    parameter[:old_hidden, :old_hidden],
                    before[:old_hidden, :old_hidden],
                ), name
                assert float(before[old_hidden:].abs().max()) == 0, name
                assert float(before[:old_hidden, old_hidden:].abs().max()) == 0, name
                assert float(parameter[old_hidden:].abs().max()) > 0, name
                assert float(parameter[:old_hidden, old_hidden:].abs().max()) > 0, name
            elif name.endswith("moe.router.weight") or (
                ".moe.experts." in name and name.endswith(".0.weight")
            ):
                assert torch.equal(parameter[:, :old_hidden], before[:, :old_hidden]), name
                assert float(before[:, old_hidden:].abs().max()) == 0, name
                assert float(parameter[:, old_hidden:].abs().max()) > 0, name
            elif ".moe.experts." in name and name.endswith(".2.weight"):
                assert torch.equal(parameter[:old_hidden], before[:old_hidden]), name
                assert float(before[old_hidden:].abs().max()) == 0, name
                assert float(parameter[old_hidden:].abs().max()) > 0, name
            else:
                raise AssertionError(f"unclassified parameter {name}")

    # The source model is what the run keeps if the mutation is rejected; it must be untouched.
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, snapshot[name]), name
    assert source.hidden_size == 16 and widened.hidden_size == 24
    assert record["perturbation_coordinates"] > 0
    assert record["perturbation_max_abs"] > 0


def test_the_perturbation_gives_the_new_coordinates_a_gradient() -> None:
    """The saddle is the reason this operator exists; after the write, the new units learn."""
    _, _, tokens, wider, _, _ = graft(scale=1e-3)
    wider.train()
    wider.zero_grad(set_to_none=True)
    wider.loss(tokens).backward()

    block = wider.blocks[0]
    assert float(block.norm_attn.weight.grad[16:].abs().max()) > 0
    qkv_grad = block.attention.qkv.weight.grad.view(3, 6, 4, 24)
    assert float(qkv_grad[:, 4:].abs().max()) > 0
    assert float(qkv_grad[:, :4, :, 16:].abs().max()) > 0
    assert float(block.moe.experts[0][0].weight.grad[:, 16:].abs().max()) > 0
    assert float(block.moe.experts[0][2].weight.grad[16:].abs().max()) > 0


def test_the_record_carries_the_price_and_the_graft_identity() -> None:
    """The mutation-history row has to be enough to price and reproduce the step."""
    source, model, _, wider, widened, record = graft(scale=1e-3)

    assert record["kind"] == "widen"
    assert (record["from_hidden"], record["to_hidden"]) == (16, 24)
    assert (record["from_heads"], record["to_heads"]) == (4, 6)
    assert record["head_dim"] == 4
    assert record["new_parameters"] == record["total_after"] - record["total_before"]
    assert record["total_ratio"] == pytest.approx(
        record["total_after"] / record["total_before"]
    )
    assert record["active_ratio"] == pytest.approx(
        record["active_after"] / record["active_before"]
    )
    # The epsilon is part of the graft, not a config detail: it halves when the width doubles.
    assert record["norm_epsilon_after"] == pytest.approx(
        source.hidden_size / widened.hidden_size * record["norm_epsilon_before"]
    )
    assert record["architecture_hash_before"] != record["architecture_hash_after"]
    assert record["seed"] == 3
    assert record["perturbation_scale"] == 1e-3
    assert record["perturbation_coordinates"] > 0
    # The record describes the model that was actually produced.
    assert record["total_after"] == estimated_parameter_count(wider.config)[0]


def test_a_genome_that_disagrees_with_its_model_is_refused() -> None:
    """A silent mismatch would widen the genome one way and the tensors another."""
    model = v0_moe_model_from_genome(genome())
    other = genome()
    other.hidden_size = 32
    with pytest.raises(ValueError, match="hidden_size"):
        graft_widened_model(model, other)
