"""Fast unit tests for the V0-scale training path (model, rebuild, protocol).

Deliberately NO session runs here: training sessions are integration checks
and live in ``scripts/v0_session_selfcheck.py`` (run on demand before the
declared protocol). Every test below is pure arithmetic, model construction,
or a mocked CLI invocation, so the whole file stays in about a second.
"""

import json
import math
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


def test_v0_session_config_rejects_unknown_dtype() -> None:
    with pytest.raises(ValueError, match="dtype"):
        V0SessionConfig(dtype="bf16")


def test_fp16_session_trains_and_saves_scaler_state(tmp_path: Path) -> None:
    """The fp16 path runs on CPU and checkpoints carry the scaler state."""
    config = V0SessionConfig(
        seed=21,
        arms=("random", "adaptive"),
        declared=False,
        device="cpu",
        vocab_size=64,
        context_size=256,
        hidden_size=64,
        depth=2,
        attention_heads=4,
        expert_count=4,
        expert_width=32,
        total_steps=4,
        mutation_interval=2,
        mutation_budget=1,
        probe_batch_count=1,
        validation_batches=2,
        checkpoint_interval=2,
        max_checkpoints=2,
        dtype="fp16",
    )
    result = run_v0_session(config, checkpoint_dir=tmp_path / "fp16")
    assert all(arm.mutation_count == 1 for arm in result.arms)
    payload = torch.load(
        tmp_path / "fp16" / "seed-21" / "arm-random" / "checkpoint-000004.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert payload["version"] == 5
    scaler_state = payload["scaler"]
    assert isinstance(scaler_state, dict) and "scale" in scaler_state
    assert float(scaler_state["scale"]) > 0


def test_fp16_resume_is_bit_identical(tmp_path: Path) -> None:
    """An fp16 session resumed from mid-run reproduces the fp16 reference."""
    config = V0SessionConfig(
        seed=22,
        arms=("random", "adaptive"),
        declared=False,
        device="cpu",
        vocab_size=64,
        context_size=256,
        hidden_size=64,
        depth=2,
        attention_heads=4,
        expert_count=4,
        expert_width=32,
        total_steps=8,
        mutation_interval=4,
        mutation_budget=2,
        probe_batch_count=1,
        validation_batches=2,
        checkpoint_interval=2,
        max_checkpoints=2,
        dtype="fp16",
    )
    reference = run_v0_session(config, checkpoint_dir=tmp_path / "ref")
    arm_dir = tmp_path / "resume" / "seed-22" / "arm-random"
    run_v0_session(config, checkpoint_dir=tmp_path / "resume")
    newest = max(arm_dir.glob("checkpoint-*.pt"), key=lambda p: int(p.stem.split("-")[-1]))
    assert newest.stem.endswith("0008")
    newest.unlink()  # force a mid-run resume from the step-6 checkpoint
    resumed = run_v0_session(config, checkpoint_dir=tmp_path / "resume")
    for before, after in zip(reference.arms, resumed.arms):
        assert after.final_loss == before.final_loss
        assert after.architecture_hash == before.architecture_hash
        assert after.training_flops == before.training_flops
        assert after.search_flops == before.search_flops
        assert after.mutation_count == before.mutation_count


def test_version4_checkpoint_still_loads(tmp_path: Path) -> None:
    """A v4 payload (no scaler entry) resumes in an fp32 session."""
    config = V0SessionConfig(
        seed=23,
        arms=("random",),
        declared=False,
        device="cpu",
        vocab_size=64,
        context_size=256,
        hidden_size=64,
        depth=2,
        attention_heads=4,
        expert_count=4,
        expert_width=32,
        total_steps=2,
        mutation_interval=2,
        mutation_budget=1,
        probe_batch_count=1,
        validation_batches=2,
        checkpoint_interval=2,
        max_checkpoints=2,
    )
    run_v0_session(config, checkpoint_dir=tmp_path / "v5")
    checkpoint = tmp_path / "v5" / "seed-23" / "arm-random" / "checkpoint-000002.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["version"] = 4
    payload.pop("scaler", None)  # emulate a pre-fp16 version-4 checkpoint
    torch.save(payload, checkpoint)
    resumed = run_v0_session(config, checkpoint_dir=tmp_path / "v5")
    assert resumed.arms[0].final_loss == resumed.arms[0].final_loss  # finite
    assert resumed.arms[0].parameter_count > 0


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


def test_v0_session_config_rejects_unknown_probe_selection() -> None:
    with pytest.raises(ValueError, match="probe_selection"):
        V0SessionConfig(probe_selection="median")


def test_argmax_selection_takes_the_probe_worst_candidate(tmp_path: Path) -> None:
    """The argmax control probes identically but decides inverted.

    Same seed, same candidate enumeration, same probe batches and booked
    probe FLOPs as the argmin controller - only the decision rule differs -
    so at the first event both arms rank the SAME candidate set and the
    argmax pick's recorded probe loss must be >= the argmin pick's (equal
    only under an all-candidate tie; strictly greater only if the choices
    differ, since one candidate cannot be both strict min and strict max).
    """
    common = dict(
        seed=31,
        arms=("adaptive",),
        declared=False,
        device="cpu",
        vocab_size=64,
        context_size=256,
        hidden_size=64,
        depth=2,
        attention_heads=4,
        expert_count=4,
        expert_width=32,
        total_steps=6,
        mutation_interval=2,
        mutation_budget=2,
        probe_batch_count=1,
        validation_batches=2,
        checkpoint_interval=2,
        max_checkpoints=2,
    )
    reference = run_v0_session(
        V0SessionConfig(**common), checkpoint_dir=tmp_path / "argmin"
    )
    control = run_v0_session(
        V0SessionConfig(**common, probe_selection="argmax"),
        checkpoint_dir=tmp_path / "argmax",
    )
    ref_events = reference.arms[0].mutation_history
    max_events = control.arms[0].mutation_history
    assert len(ref_events) == len(max_events) == 2
    assert ref_events[0]["policy"] == "adaptive_probe_argmin"
    assert max_events[0]["policy"] == "adaptive_probe_argmax"
    assert control.arms[0].search_flops > 0
    assert max_events[0]["probe_loss"] >= ref_events[0]["probe_loss"]
    if max_events[0]["probe_loss"] > ref_events[0]["probe_loss"]:
        choice = ("operation", "layer_id", "expert_id")
        assert tuple(max_events[0][k] for k in choice) != tuple(
            ref_events[0][k] for k in choice
        )


def test_probe_landscape_records_ranked_candidates(tmp_path: Path) -> None:
    """Landscape instrumentation logs every probed candidate, best-first.

    Mirrors the toy path's ``probe_landscape`` contract: one entry per
    evaluated candidate sorted ascending by probe loss (index k = k-th best),
    with ``probe_selected_index`` pointing into it - so an argmin event
    selects index 0. Bookkeeping only: the winner's probe loss, candidate
    count, and search FLOPs are unchanged from the un-instrumented run.
    """
    common = dict(
        seed=41,
        arms=("adaptive",),
        declared=False,
        device="cpu",
        vocab_size=64,
        context_size=256,
        hidden_size=64,
        depth=2,
        attention_heads=4,
        expert_count=4,
        expert_width=32,
        total_steps=6,
        mutation_interval=2,
        mutation_budget=2,
        probe_batch_count=1,
        validation_batches=2,
        checkpoint_interval=2,
        max_checkpoints=2,
    )
    plain = run_v0_session(
        V0SessionConfig(**common), checkpoint_dir=tmp_path / "plain"
    )
    instrumented = run_v0_session(
        V0SessionConfig(**common, probe_landscape=True),
        checkpoint_dir=tmp_path / "landscape",
    )
    plain_arm = plain.arms[0]
    landscape_arm = instrumented.arms[0]
    assert landscape_arm.final_loss == plain_arm.final_loss
    assert landscape_arm.search_flops == plain_arm.search_flops
    for event in landscape_arm.mutation_history:
        landscape = event["probe_landscape"]
        assert len(landscape) == event["probe_candidates"]
        losses = [item["probe_loss"] for item in landscape]
        assert losses == sorted(losses)
        assert all(math.isfinite(loss) for loss in losses)
        assert all(item["operation"] in {"grow", "prune"} for item in landscape)
        assert all("L" in item["layer_id"] for item in landscape)
        selected = event["probe_selected_index"]
        assert 0 <= selected < len(landscape)
        assert event["probe_loss"] == landscape[selected]["probe_loss"]
        assert selected == 0  # argmin rule picks the probe-best candidate


def test_probe_landscape_marks_argmax_tail_selection(tmp_path: Path) -> None:
    """Under argmax the recorded selected index is the probe-worst rank."""
    result = run_v0_session(
        V0SessionConfig(
            seed=42,
            arms=("adaptive",),
            declared=False,
            device="cpu",
            vocab_size=64,
            context_size=256,
            hidden_size=64,
            depth=2,
            attention_heads=4,
            expert_count=4,
            expert_width=32,
            total_steps=4,
            mutation_interval=2,
            mutation_budget=1,
            probe_batch_count=1,
            validation_batches=2,
            checkpoint_interval=2,
            max_checkpoints=2,
            probe_selection="argmax",
            probe_landscape=True,
        ),
        checkpoint_dir=tmp_path / "argmax-landscape",
    )
    for event in result.arms[0].mutation_history:
        landscape = event["probe_landscape"]
        selected = event["probe_selected_index"]
        assert selected == len(landscape) - 1
        losses = [item["probe_loss"] for item in landscape]
        assert event["probe_loss"] == max(losses)


def test_probe_landscape_off_by_default(tmp_path: Path) -> None:
    """Without the flag, events carry no landscape fields (contract parity)."""
    result = run_v0_session(
        V0SessionConfig(
            seed=43,
            arms=("adaptive",),
            declared=False,
            device="cpu",
            vocab_size=64,
            context_size=256,
            hidden_size=64,
            depth=2,
            attention_heads=4,
            expert_count=4,
            expert_width=32,
            total_steps=4,
            mutation_interval=2,
            mutation_budget=1,
            probe_batch_count=1,
            validation_batches=2,
            checkpoint_interval=2,
            max_checkpoints=2,
        ),
        checkpoint_dir=tmp_path / "off",
    )
    for event in result.arms[0].mutation_history:
        assert "probe_landscape" not in event
        assert "probe_selected_index" not in event


# --- long-horizon probe (section 5.3 contingency fix candidate) -------------


def test_v0_session_config_rejects_negative_probe_finetune_steps() -> None:
    with pytest.raises(ValueError, match="probe_finetune_steps"):
        V0SessionConfig(probe_finetune_steps=-1)


def test_predicted_cost_ratio_books_lookahead_steps() -> None:
    """R > 0 books R full-model training steps per probed candidate."""
    total, active = estimated_parameter_count(DECLARED_V0_CONFIG)
    config = V0SessionConfig(probe_finetune_steps=1)
    probe_tokens = config.batch_size * config.context_size * config.probe_batch_count
    expected = (
        3
        * 4
        * (probe_tokens * 2 * active + 1 * probe_tokens * 2 * total)
        / (2 * total * config.training_tokens())
    )
    assert predicted_cost_ratio(config) == pytest.approx(expected)
    assert predicted_cost_ratio(config) > predicted_cost_ratio(V0SessionConfig())
    # Criterion guardrail at the proof configuration: each lookahead step
    # adds exactly one full-model training step per candidate (3x the probe
    # forward), so R=1 lands at ~3% (clean) and R=2 sits exactly at the 5%
    # criterion with no margin - measured ratios run slightly above predicted
    # on real data, so R=2 breaches in practice and R=1 is the only clean
    # setting.
    proof = dict(
        vocab_size=8_000,
        context_size=512,
        hidden_size=256,
        depth=8,
        attention_heads=8,
        expert_count=8,
        active_experts=2,
        expert_width=256,
        batch_size=8,
        total_steps=600,
        mutation_interval=200,
        declared=False,
    )
    assert 0.02 < predicted_cost_ratio(V0SessionConfig(**proof, probe_finetune_steps=1)) < 0.05
    assert predicted_cost_ratio(
        V0SessionConfig(**proof, probe_finetune_steps=2)
    ) == pytest.approx(0.0499773, abs=1e-4)


def test_lookahead_probe_records_steps_books_cost_and_reranks(tmp_path: Path) -> None:
    """The long-horizon probe adapts each candidate before scoring it.

    Both runs probe the same initial model at the first event, so bit-equal
    first-event landscapes would mean the fine-tune step did nothing. The
    lookahead run books strictly more search FLOPs (the adaptation steps on
    the full probed model) and records R in every event.
    """
    common = dict(
        seed=51,
        arms=("adaptive",),
        declared=False,
        device="cpu",
        vocab_size=64,
        context_size=256,
        hidden_size=64,
        depth=2,
        attention_heads=4,
        expert_count=4,
        expert_width=32,
        total_steps=6,
        mutation_interval=2,
        mutation_budget=2,
        probe_batch_count=1,
        validation_batches=2,
        checkpoint_interval=2,
        max_checkpoints=2,
        probe_landscape=True,
    )
    snapshot = run_v0_session(
        V0SessionConfig(**common), checkpoint_dir=tmp_path / "snapshot"
    )
    lookahead = run_v0_session(
        V0SessionConfig(**common, probe_finetune_steps=1),
        checkpoint_dir=tmp_path / "lookahead",
    )
    snap_arm = snapshot.arms[0]
    look_arm = lookahead.arms[0]
    assert all(event["probe_finetune_steps"] == 0 for event in snap_arm.mutation_history)
    assert all(event["probe_finetune_steps"] == 1 for event in look_arm.mutation_history)
    assert look_arm.mutation_count == snap_arm.mutation_count == 2
    assert look_arm.search_flops > snap_arm.search_flops
    # The ranking is recomputed after adaptation, and the landscape contract
    # (sorted ascending, selected index consistent) holds under lookahead.
    assert (
        look_arm.mutation_history[0]["probe_landscape"]
        != snap_arm.mutation_history[0]["probe_landscape"]
    )
    for event in look_arm.mutation_history:
        landscape = event["probe_landscape"]
        losses = [item["probe_loss"] for item in landscape]
        assert losses == sorted(losses)
        selected = event["probe_selected_index"]
        assert 0 <= selected < len(landscape)
        assert event["probe_loss"] == landscape[selected]["probe_loss"]


def test_lookahead_probe_resume_is_bit_identical(tmp_path: Path) -> None:
    """Throwaway fine-tune batches re-derive exactly on mid-run resume."""
    config = V0SessionConfig(
        seed=52,
        arms=("random", "adaptive"),
        declared=False,
        device="cpu",
        vocab_size=64,
        context_size=256,
        hidden_size=64,
        depth=2,
        attention_heads=4,
        expert_count=4,
        expert_width=32,
        total_steps=8,
        mutation_interval=4,
        mutation_budget=2,
        probe_batch_count=1,
        validation_batches=2,
        checkpoint_interval=2,
        max_checkpoints=2,
        probe_finetune_steps=1,
    )
    reference = run_v0_session(config, checkpoint_dir=tmp_path / "ref")
    arm_dir = tmp_path / "resume" / "seed-52" / "arm-random"
    run_v0_session(config, checkpoint_dir=tmp_path / "resume")
    newest = max(arm_dir.glob("checkpoint-*.pt"), key=lambda p: int(p.stem.split("-")[-1]))
    assert newest.stem.endswith("0008")
    newest.unlink()  # force a mid-run resume from the step-6 checkpoint
    resumed = run_v0_session(config, checkpoint_dir=tmp_path / "resume")
    for before, after in zip(reference.arms, resumed.arms):
        assert after.final_loss == before.final_loss
        assert after.architecture_hash == before.architecture_hash
        assert (after.training_flops, after.search_flops) == (
            before.training_flops,
            before.search_flops,
        )
        assert json.dumps(after.mutation_history, sort_keys=True) == json.dumps(
            before.mutation_history, sort_keys=True
        )
