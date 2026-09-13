"""Fast unit tests for the V0-scale training path (model, rebuild, protocol).

Deliberately NO session runs here: training sessions are integration checks
and live in ``scripts/v0_session_selfcheck.py`` (run on demand before the
declared protocol). Every test below is pure arithmetic, model construction,
or a mocked CLI invocation, so the whole file stays in about a second.
"""

import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from act.model import (
    DECLARED_V0_CONFIG,
    V0MoECausalLM,
    V0MoEConfig,
    estimated_parameter_count,
    rebuild_v0_moe_from_genome,
    validate_declared_v0_config,
)
from act.topology import (
    create_genome,
    feasible_expert_mutations,
    grow_expert,
    prune_expert,
)
from act.training import (
    V0SessionConfig,
    V0SessionResult,
    aggregate_arm_cost_ratio,
    harm_guard_rejects,
    predicted_cost_ratio,
    run_v0_session,
    sign_pairs,
)
from act.training.v0_session import V0ArmResult


TINY = dict(
    vocab_size=64,
    context_size=256,
    hidden_size=64,
    depth=2,
    attention_heads=4,
    experts_per_layer=(4, 4),
    active_experts=2,
    expert_width=32,
)


def test_declared_v0_config_reproduces_protocol_parameters() -> None:
    validate_declared_v0_config(DECLARED_V0_CONFIG)
    total, active = estimated_parameter_count(DECLARED_V0_CONFIG)
    assert total == 285_336_576
    assert active == 134_341_632
    assert 256_000_000 <= total <= 350_000_000  # plan section 8 V0 target


def test_v0_moe_materialized_counts_match_estimator() -> None:
    config = V0MoEConfig(**TINY)
    model = V0MoECausalLM(config)
    total, active = estimated_parameter_count(config)
    assert model.parameter_count == total
    assert model.active_parameter_count == active
    tokens = torch.randint(0, TINY["vocab_size"], (2, 16))
    assert torch.isfinite(model.loss(tokens))


def test_v0_session_predicts_protocol_cost_ratios() -> None:
    primary = 100.0 * predicted_cost_ratio(V0SessionConfig())
    secondary = 100.0 * predicted_cost_ratio(V0SessionConfig(probe_batch_count=2))
    assert primary == pytest.approx(0.0942, abs=5e-4)
    assert secondary == pytest.approx(0.1883, abs=5e-3)


def test_v0_attention_matches_multihead_attention_under_parameter_parity() -> None:
    """Bias-free SDPA attention is exactly causal MHA with copied weights."""
    from act.model.v0_moe import _SelfAttention

    torch.manual_seed(3)
    hidden, heads, seq, batch = 64, 4, 32, 2
    mha = nn.MultiheadAttention(hidden, heads, dropout=0.0, batch_first=True, bias=False)
    attention = _SelfAttention(V0MoEConfig(**TINY))
    with torch.no_grad():
        attention.qkv.weight.copy_(mha.in_proj_weight)
        attention.out.weight.copy_(mha.out_proj.weight)
    attention.eval()
    mha.eval()
    x = torch.randn(batch, seq, hidden)
    mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    expected, _ = mha(x, x, x, attn_mask=mask, need_weights=False)
    # Position 0 makes the rotary transform the identity, so the parity check
    # isolates the attention kernel from the position model.
    actual = attention(x, torch.zeros(seq, dtype=torch.long))
    assert torch.allclose(actual, expected, atol=1e-6)


def test_v0_rebuild_transfers_expert_weights_by_identity() -> None:
    torch.manual_seed(0)
    genome = create_genome(
        7, hidden_size=64, vocab_size=64, depth=3, ffn="moe",
        expert_count=4, active_experts=2,
    )
    genome.context_size = 256
    for layer in genome.layers:
        layer.heads = 4
        for expert in layer.experts:
            expert.width = 32
    model = V0MoECausalLM(V0MoEConfig(
        vocab_size=64, context_size=256, hidden_size=64, depth=3,
        attention_heads=4, experts_per_layer=(4, 4, 4), active_experts=2,
        expert_width=32,
    ))
    assert grow_expert(genome, "L0", 3)
    assert grow_expert(genome, "L1", 3)
    assert prune_expert(genome, "L2", "L2E0")
    rebuilt = rebuild_v0_moe_from_genome(model, genome)
    assert rebuilt.parameter_count == estimated_parameter_count(rebuilt.config)[0]
    # Shared (non-expert) parameters survive unchanged.
    assert torch.equal(rebuilt.blocks[0].attention.qkv.weight, model.blocks[0].attention.qkv.weight)
    # Retained experts keep their trained weights under their genome id.
    before = model.blocks[1].moe.experts[2][0].weight
    index = rebuilt.blocks[1].moe.expert_ids.index("L1E2")
    assert torch.equal(rebuilt.blocks[1].moe.experts[index][0].weight, before)


def test_feasible_mutations_shared_with_toy_path() -> None:
    genome = create_genome(
        5, hidden_size=24, vocab_size=32, depth=1, ffn="moe",
        expert_count=4, active_experts=2,
    )
    candidates = feasible_expert_mutations(genome)
    assert candidates[0] == ("grow", "L0", None)
    prunes = [candidate for candidate in candidates if candidate[0] == "prune"]
    assert {candidate[2] for candidate in prunes} == {"L0E0", "L0E1", "L0E2", "L0E3"}


def test_v0_moe_config_validates_structure() -> None:
    with pytest.raises(ValueError, match="experts_per_layer"):
        V0MoEConfig(experts_per_layer=(4, 4))  # depth defaults to 12
    with pytest.raises(ValueError, match="at least 2"):
        V0MoEConfig(experts_per_layer=(1,) * 4, depth=4)
    with pytest.raises(ValueError, match="16 experts"):
        V0MoEConfig(experts_per_layer=(17,) * 4, depth=4)
    with pytest.raises(ValueError, match="active_experts"):
        V0MoEConfig(experts_per_layer=(4,) * 4, depth=4, active_experts=5)
    with pytest.raises(ValueError, match="expert_ids"):
        V0MoEConfig(
            experts_per_layer=(4,) * 4, depth=4, expert_ids=(("L0E0", "L0E1"),) * 4
        )


def test_declared_guard_rejects_undeclared_model(tmp_path: Path) -> None:
    # Raises during validation, before any session work, so this stays fast.
    with pytest.raises(ValueError, match="declared"):
        run_v0_session(
            V0SessionConfig(
                seed=11, declared=True, device="cpu", vocab_size=64,
                context_size=256, hidden_size=64, depth=2, attention_heads=4,
                expert_count=4, expert_width=32,
            ),
            checkpoint_dir=tmp_path / "bad",
        )


def test_protocol_aggregation_and_harm_guard() -> None:
    def _result(adaptive_loss: float, random_loss: float) -> V0SessionResult:
        arm_values = {
            "adaptive": (adaptive_loss, 101.0, 11.0),
            "random": (random_loss, 100.0, 0.0),
        }
        arms = tuple(
            V0ArmResult(
                mode=mode,
                final_loss=value[0],
                validation_loss=value[0],
                quality=-value[0],
                mutation_count=3,
                parameter_count=value[1],
                active_parameter_count=value[1] // 2,
                training_flops=10_000,
                search_flops=int(value[2] * 10),
                architecture_hash=f"hash-{mode}",
                mutation_history=(),
                wall_time_seconds=1.0,
                tokens_per_second=1000.0,
            )
            for mode, value in arm_values.items()
        )
        return V0SessionResult(config={}, predicted_cost_ratio=0.001, arms=arms)

    results = [_result(1.0, 1.1), _result(0.9, 1.0)]
    # Aggregate cost: (110 + 110) / (10000 + 10000)
    assert aggregate_arm_cost_ratio(results) == pytest.approx(0.011)
    pairs = sign_pairs(results)
    assert pairs == ((1.0, 1.1), (0.9, 1.0))
    assert not harm_guard_rejects(pairs)
    losing = [_result(1.1, 1.0) for _ in range(10)]
    assert harm_guard_rejects(sign_pairs(losing))
    borderline = [_result(1.1, 1.0) for _ in range(9)]
    assert not harm_guard_rejects(sign_pairs(borderline))


def test_cli_v0_mode_writes_summary(tmp_path: Path, monkeypatch) -> None:
    from act.training import cli

    def fake_run(config, checkpoint_dir="artifacts/v0-sessions", log=None):
        arms = (
            V0ArmResult(
                mode="random",
                final_loss=1.10,
                validation_loss=1.10,
                quality=-1.10,
                mutation_count=3,
                parameter_count=100,
                active_parameter_count=50,
                training_flops=10_000,
                search_flops=0,
                architecture_hash="r",
                mutation_history=(),
                wall_time_seconds=1.0,
                tokens_per_second=1.0,
            ),
            V0ArmResult(
                mode="adaptive",
                final_loss=1.05,
                validation_loss=1.05,
                quality=-1.05,
                mutation_count=3,
                parameter_count=101,
                active_parameter_count=51,
                training_flops=10_000,
                search_flops=110,
                architecture_hash="a",
                mutation_history=(),
                wall_time_seconds=2.0,
                tokens_per_second=2.0,
            ),
        )
        return V0SessionResult(
            config={"seed": config.seed}, predicted_cost_ratio=0.000942, arms=arms
        )

    monkeypatch.setattr(cli, "run_v0_session", fake_run)
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "act.training.cli",
            "--mode", "v0",
            "--seeds", "9000,9001",
            "--checkpoint-dir", str(tmp_path / "sessions"),
            "--output", str(output),
        ],
    )
    cli.main()
    summary = json.loads(output.read_text())
    assert summary["protocol"] == "v0-compute-efficiency-5.1"
    assert summary["seeds"] == [9000, 9001]
    assert summary["adaptive_wins"] == 2
    assert not summary["harm_guard_rejects"]
    assert summary["measured_cost_percent"] == pytest.approx(1.1)
    assert len(summary["per_seed"]) == 2
