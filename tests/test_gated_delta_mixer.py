"""Tests for the gated delta mixer and per-layer module-type plumbing.

A wrong mixer would corrupt an architecture search for the same reason a wrong
seed did, so the recurrence is checked against properties that follow from its
definition rather than only against a re-implementation of itself:

- with write strength 1 the erase term makes the state reproduce the written
  value for the written key exactly (the defining property of a delta rule);
- with write strength 0 only the decay acts, so the readout scales as a power of
  the gate;
- the readout is causal.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from act.model.v0_moe import (
    DECLARED_V0_CONFIG,
    V0MoEConfig,
    V0MoECausalLM,
    _GatedDeltaMixer,
    estimated_parameter_count,
    resolve_module_types,
)

BATCH, SEQ, HEADS, DIM = 2, 6, 2, 3
HIDDEN = HEADS * DIM


def mixer_config(**overrides) -> V0MoEConfig:
    params = dict(
        vocab_size=64,
        context_size=SEQ,
        hidden_size=HIDDEN,
        depth=1,
        attention_heads=HEADS,
        experts_per_layer=(2,),
        active_experts=2,
        expert_width=HIDDEN,
        module_types=("gdn",),
    )
    params.update(overrides)
    return V0MoEConfig(**params)


def neutral_mixer() -> _GatedDeltaMixer:
    """A mixer whose output projection is the identity, so readouts compare directly."""
    mixer = _GatedDeltaMixer(mixer_config())
    with torch.no_grad():
        mixer.out.weight.copy_(torch.eye(HIDDEN))
    return mixer


def normalized(batch: int, seq: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    raw = torch.randn(batch, seq, HEADS, DIM, generator=generator)
    return F.normalize(raw, dim=-1)


def test_unit_write_strength_reproduces_the_written_value():
    """The delta rule's defining property: S^T k == v after writing (k, v) with beta=1."""
    keys = normalized(BATCH, SEQ)
    values = torch.randn(BATCH, SEQ, HEADS, DIM)
    beta = torch.ones(BATCH, SEQ, HEADS)
    alpha = torch.ones(BATCH, SEQ, HEADS)

    out = neutral_mixer().mix(keys, keys, values, alpha, beta)
    assert torch.allclose(out, values.reshape(BATCH, SEQ, HIDDEN), atol=1e-5)


def constant_keys() -> torch.Tensor:
    """One repeated L2-normalized key, so ``k_s . k_t == 1`` for every pair.

    Probing with a fixed key removes the ``k_s . k_t`` factor from the readout and
    leaves the decay as the only term under test.
    """
    single = normalized(BATCH, 1)
    return single.expand(BATCH, SEQ, HEADS, DIM).contiguous()


# --- segmented scan (brief section 14: GDN as a searchable gene) ----------
#
# The explicit scan retains a (batch, heads, key_dim, value_dim) write and state
# per timestep, which is ~0.8 GB per GDN layer at the zoo's geometry and makes
# all-GDN exceed an 8 GB card. Segmentation stores only chunk boundaries and
# recomputes within a chunk. That is only usable as a search variable if it is
# numerically free, so these tests pin equivalence rather than approximate
# agreement.


def test_chunked_scan_reproduces_the_full_scan_exactly():
    mixer = _GatedDeltaMixer(mixer_config())
    generator = torch.Generator().manual_seed(7)
    q = F.normalize(torch.randn(BATCH, SEQ, HEADS, DIM, generator=generator), dim=-1)
    k = F.normalize(torch.randn(BATCH, SEQ, HEADS, DIM, generator=generator), dim=-1)
    v = torch.randn(BATCH, SEQ, HEADS, DIM, generator=generator)
    alpha = torch.rand(BATCH, SEQ, HEADS, generator=generator)
    beta = torch.rand(BATCH, SEQ, HEADS, generator=generator)

    reference = mixer.mix(q, k, v, alpha, beta)
    # 1, 2 and 5 exercise a boundary at every step, evenly, and not at all;
    # 4 deliberately does not divide SEQ=6, so the final segment is short.
    for chunk in (1, 2, 4, 5, SEQ, SEQ + 1):
        chunked = mixer.mix(q, k, v, alpha, beta, chunk_size=chunk)
        assert torch.allclose(reference, chunked, atol=1e-6, rtol=0), f"chunk={chunk}"


def test_chunked_scan_gradients_match_the_full_scan():
    """Checkpointing changes *how* the backward runs, so it must be checked."""
    mixer = _GatedDeltaMixer(mixer_config())
    generator = torch.Generator().manual_seed(11)
    base = [
        F.normalize(torch.randn(BATCH, SEQ, HEADS, DIM, generator=generator), dim=-1),
        F.normalize(torch.randn(BATCH, SEQ, HEADS, DIM, generator=generator), dim=-1),
        torch.randn(BATCH, SEQ, HEADS, DIM, generator=generator),
        torch.rand(BATCH, SEQ, HEADS, generator=generator),
        torch.rand(BATCH, SEQ, HEADS, generator=generator),
    ]

    collected: list[list[torch.Tensor]] = []
    for chunk in (0, 2, 4):
        inputs = [tensor.clone().requires_grad_(True) for tensor in base]
        out = mixer.mix(*inputs, chunk_size=chunk)
        out.pow(2).sum().backward()
        collected.append([tensor.grad for tensor in inputs])

    for index, other in enumerate(collected[1:], start=1):
        for a, b in zip(collected[0], other):
            assert torch.allclose(a, b, atol=1e-6, rtol=0), f"gradient mismatch at chunk set {index}"


def test_state_survives_a_chunk_boundary():
    """The failure a segmented scan invites: silently resetting state per chunk.

    With a constant key, a single unit write at t=0 and no decay, every later
    readout must still return the written value. If the state were reset at a
    boundary, the segments after it would read out zeros instead.
    """
    mixer = neutral_mixer()
    keys = constant_keys()
    values = torch.randn(BATCH, SEQ, HEADS, DIM)
    alpha = torch.ones(BATCH, SEQ, HEADS)
    beta = torch.zeros(BATCH, SEQ, HEADS)
    beta[:, 0] = 1.0
    # Nothing after t=0 writes and nothing decays, so the state is frozen at
    # ``k v_0^T`` and every readout - including those in later chunks - returns
    # v_0. Segments that reset their state would read out zeros instead.
    written = values[:, 0].reshape(BATCH, 1, HIDDEN).expand(BATCH, SEQ, HIDDEN)

    for chunk in (1, 2, 3, 5):
        out = mixer.mix(keys, keys, values, alpha, beta, chunk_size=chunk)
        assert torch.allclose(out, written, atol=1e-5), f"state lost at chunk={chunk}"


def test_chunk_size_defaults_to_off_and_is_validated():
    assert V0MoEConfig().gdn_chunk_size == 0
    assert _GatedDeltaMixer(mixer_config()).chunk_size == 0
    assert _GatedDeltaMixer(mixer_config(gdn_chunk_size=128)).chunk_size == 128
    import pytest

    with pytest.raises(ValueError):
        V0MoEConfig(
            context_size=SEQ,
            hidden_size=HIDDEN,
            depth=1,
            attention_heads=HEADS,
            experts_per_layer=(2,),
            gdn_chunk_size=-1,
        )


def test_zero_write_strength_leaves_only_decay():
    """With beta=0 after one write, the readout decays as alpha^t."""
    keys = constant_keys()
    values = torch.randn(BATCH, SEQ, HEADS, DIM)
    decay = 0.5
    alpha = torch.full((BATCH, SEQ, HEADS), decay)
    beta = torch.zeros(BATCH, SEQ, HEADS)
    beta[:, 0] = 1.0  # one write at t=0, then pure decay

    out = neutral_mixer().mix(keys, keys, values, alpha, beta)
    written = values[:, 0].reshape(BATCH, HIDDEN)
    for step in range(SEQ):
        expected = written * (decay**step)
        assert torch.allclose(out[:, step], expected, atol=1e-5), f"step {step}"


def test_decay_is_applied_to_the_retained_state_only():
    """A weaker gate must leave proportionally less of the earlier write behind."""
    keys = constant_keys()
    values = torch.randn(BATCH, SEQ, HEADS, DIM)
    beta = torch.zeros(BATCH, SEQ, HEADS)
    beta[:, 0] = 1.0
    mixer = neutral_mixer()

    slow = mixer.mix(keys, keys, values, torch.full((BATCH, SEQ, HEADS), 0.9), beta)
    fast = mixer.mix(keys, keys, values, torch.full((BATCH, SEQ, HEADS), 0.1), beta)
    written = values[:, 0].reshape(BATCH, HIDDEN)

    assert torch.allclose(slow[:, -1], written * 0.9 ** (SEQ - 1), atol=1e-5)
    assert torch.allclose(fast[:, -1], written * 0.1 ** (SEQ - 1), atol=1e-5)


def test_readout_is_causal():
    """Changing a future token must not change any earlier output."""
    keys = normalized(BATCH, SEQ)
    values = torch.randn(BATCH, SEQ, HEADS, DIM)
    alpha = torch.full((BATCH, SEQ, HEADS), 0.7)
    beta = torch.full((BATCH, SEQ, HEADS), 0.5)
    mixer = neutral_mixer()

    baseline = mixer.mix(keys, keys, values, alpha, beta)

    perturbed_values = values.clone()
    perturbed_values[:, -1] += 5.0
    perturbed = mixer.mix(keys, keys, perturbed_values, alpha, beta)

    assert torch.allclose(baseline[:, :-1], perturbed[:, :-1], atol=1e-6)


def test_mix_matches_an_explicit_scan():
    """Indexing and decay-axis consistency against a plain loop over the equation."""
    keys = normalized(BATCH, SEQ)
    queries = normalized(BATCH, SEQ)
    values = torch.randn(BATCH, SEQ, HEADS, DIM)
    alpha = torch.rand(BATCH, SEQ, HEADS)
    beta = torch.rand(BATCH, SEQ, HEADS)

    got = neutral_mixer().mix(queries, keys, values, alpha, beta)

    state = torch.zeros(BATCH, HEADS, DIM, DIM)
    outputs = []
    for step in range(SEQ):
        key, value = keys[:, step], values[:, step]
        prediction = torch.einsum("bhij,bhi->bhj", state, key)
        error = value - prediction
        state = state * alpha[:, step].unsqueeze(-1).unsqueeze(-1) + beta[
            :, step
        ].unsqueeze(-1).unsqueeze(-1) * torch.einsum("bhi,bhj->bhij", key, error)
        outputs.append(torch.einsum("bhij,bhi->bhj", state, queries[:, step]))
    expected = torch.stack(outputs, dim=1).reshape(BATCH, SEQ, HIDDEN)

    assert torch.allclose(got, expected, atol=1e-6)


def test_gates_are_bounded():
    """alpha must be a contraction and beta a [0,1] write strength."""
    mixer = _GatedDeltaMixer(mixer_config())
    hidden = torch.randn(BATCH, SEQ, HIDDEN)
    _, _, _, alpha, beta = mixer.project(hidden)
    assert (alpha > 0).all() and (alpha <= 1).all()
    assert (beta >= 0).all() and (beta <= 1).all()


def test_projections_are_key_value_dimension_matched():
    mixer = _GatedDeltaMixer(mixer_config())
    hidden = torch.randn(BATCH, SEQ, HIDDEN)
    q, k, v, _, _ = mixer.project(hidden)
    assert q.shape == k.shape == v.shape == (BATCH, SEQ, HEADS, DIM)
    assert torch.allclose(k.norm(dim=-1), torch.ones(BATCH, SEQ, HEADS), atol=1e-5)


# --- module-type plumbing -------------------------------------------------


def test_module_types_change_parameter_count():
    attention = estimated_parameter_count(mixer_config(module_types=("attention",)))
    gdn = estimated_parameter_count(mixer_config(module_types=("gdn",)))
    hybrid = estimated_parameter_count(mixer_config(module_types=("hybrid",)))
    assert gdn[0] > attention[0]
    assert hybrid[0] > gdn[0]


def test_all_attention_pattern_matches_legacy_count():
    """An explicit attention pattern must reproduce the pre-existing accounting."""
    explicit = estimated_parameter_count(mixer_config(module_types=("attention",) * 1))
    implicit = estimated_parameter_count(mixer_config(module_types=()))
    assert explicit == implicit


def test_declared_config_parameter_counts_are_unchanged():
    """The pre-registered protocol identity must not move."""
    assert estimated_parameter_count(DECLARED_V0_CONFIG) == (285_336_576, 134_341_632)


def test_resolve_module_types_defaults_to_attention():
    assert resolve_module_types(mixer_config(module_types=())) == ("attention",)
    assert resolve_module_types(mixer_config(module_types=("gdn",))) == ("gdn",)


def test_estimated_counts_match_the_built_model_for_every_pattern():
    """Cost matching is meaningless if the analytic count disagrees with reality."""
    for pattern in (
        ("attention",) * 3,
        ("gdn",) * 3,
        ("hybrid",) * 3,
        ("attention", "gdn", "hybrid"),
    ):
        config = V0MoEConfig(
            vocab_size=64,
            context_size=SEQ,
            hidden_size=24,
            depth=3,
            attention_heads=3,
            experts_per_layer=(4,) * 3,
            active_experts=2,
            expert_width=24,
            module_types=pattern,
        )
        estimated = estimated_parameter_count(config)
        model = V0MoECausalLM(config)
        assert estimated == (model.parameter_count, model.active_parameter_count), pattern


def test_module_pattern_zoo_isolates_the_mixer():
    """The pattern zoo must vary only ``layer.module``."""
    from act.search.architecture_zoo import generate_module_pattern_zoo

    zoo = generate_module_pattern_zoo(depth=8, hidden_size=768, expert_count=8, active_experts=2)
    assert len(zoo) >= 8
    geometries = {(c.genome.depth, c.genome.hidden_size) for c in zoo}
    assert len(geometries) == 1
    patterns = {tuple(layer.module for layer in c.genome.layers) for c in zoo}
    assert len(patterns) == len(zoo)
    # the brief's 3:1 schedule and its inverse must both be present as controls
    names = {c.name for c in zoo}
    assert any("gdn_3to1" in name for name in names)
    assert any("att_3to1" in name for name in names)


def test_mixed_module_pattern_builds_and_runs():
    config = V0MoEConfig(
        vocab_size=64,
        context_size=SEQ,
        hidden_size=HIDDEN,
        depth=4,
        attention_heads=HEADS,
        experts_per_layer=(2,) * 4,
        active_experts=2,
        expert_width=HIDDEN,
        module_types=("attention", "gdn", "gdn", "gdn"),
    )
    model = V0MoECausalLM(config)
    tokens = torch.randint(0, 64, (BATCH, SEQ))
    assert model(tokens).shape == (BATCH, SEQ, 64)
    assert [block.module_type for block in model.blocks] == ["attention", "gdn", "gdn", "gdn"]
    assert model.blocks[0].mixer is None
    assert model.blocks[1].attention is None


def test_invalid_module_pattern_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        V0MoEConfig(
            vocab_size=64,
            context_size=SEQ,
            hidden_size=HIDDEN,
            depth=4,
            attention_heads=HEADS,
            experts_per_layer=(2,) * 4,
            active_experts=2,
            expert_width=HIDDEN,
            module_types=("attention", "gdn"),
        )
    with pytest.raises(ValueError):
        V0MoEConfig(
            vocab_size=64,
            context_size=SEQ,
            hidden_size=HIDDEN,
            depth=2,
            attention_heads=HEADS,
            experts_per_layer=(2,) * 2,
            active_experts=2,
            expert_width=HIDDEN,
            module_types=("attention", "diffusion"),
        )


def test_gdn_model_trains_a_step():
    """Gradients must reach the mixer projections."""
    config = mixer_config()
    model = V0MoECausalLM(config)
    tokens = torch.randint(0, 64, (BATCH, SEQ))
    loss = model.loss(tokens)
    loss.backward()
    assert model.blocks[0].mixer.gate_proj.weight.grad is not None
    assert model.blocks[0].mixer.gate_proj.weight.grad.abs().sum() > 0
