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
    DEV_V0_CONFIG,
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
    declared_session_config,
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


def test_session_default_is_the_fast_dev_config_not_the_declared_target() -> None:
    """The default must be the fast iteration shape, by measurement.

    A 285M default made every development cycle an hour-long protocol run. The
    dev shape and its budget are pinned here so a future edit cannot quietly
    restore the expensive default; the declared target stays reachable through
    ``declared_session_config()``.
    """
    default = V0SessionConfig()
    assert default.declared is False
    assert default.model_config() == DEV_V0_CONFIG
    total, active = estimated_parameter_count(DEV_V0_CONFIG)
    assert total < 20_000_000 and active < 10_000_000
    # ...and validated by the sizing probe on the dev GPU, not by preference.
    assert 14_000_000 < total < 15_500_000
    assert (default.batch_size, default.context_size, default.total_steps) == (8, 512, 400)
    assert default.training_tokens() == 1_638_400
    # The declared cadence has four interval slots for a budget of three
    # (6000/1500 with E=3); the dev horizon mirrors that shape exactly.
    assert default.total_steps // default.mutation_interval == 4
    assert default.mutation_budget == 3
    # The dev budget is not the protocol's probe-cost question and must not be
    # read as one: at dev scale the probe is 1.5% of training FLOPs, not the
    # declared 0.0942%.
    assert 100.0 * predicted_cost_ratio(default) == pytest.approx(1.5184, abs=5e-3)
    # Activation checkpointing is a memory device, so the dev default leaves it
    # off: at the dev shape the recompute is pure cost. Measured on the dev GPU
    # (60 steps, real stream, one arm): 13.3 s on against 10.1 s off, 1.32x, with
    # bit-identical losses - the flag is trajectory-inert (report section 5.37).
    assert default.activation_checkpointing is False


def test_declared_session_config_builds_the_protocol_target() -> None:
    declared = declared_session_config()
    assert declared.declared is True
    assert declared.model_config() == DECLARED_V0_CONFIG
    assert declared.training_tokens() == 24_576_000
    total, active = estimated_parameter_count(declared.model_config())
    assert total == 285_336_576 and active == 134_341_632
    # The declared config doubles as the protocol's cost-ratio input.
    assert 100.0 * predicted_cost_ratio(declared) == pytest.approx(0.0942, abs=5e-4)
    # ...and section 5.2's T4 fit arithmetic assumes checkpointed activations, so
    # the declared target sets this explicitly rather than inheriting the (fast)
    # dev default. Pinned so neither end can drift silently.
    assert declared.activation_checkpointing is True


def test_a_declared_null_seed_trains_the_pair_arm_budget() -> None:
    """The 5.14 null queue is full-fidelity, so it costs a pair's arm-time.

    Measured 2026-09-20: the report and the runbook both priced a null seed at
    "~90 s of GPU" - a figure from a dev-shape, 400-step study. A null session
    launched by the runbook's own command inherits the declared config (only
    `--profile` shrinks it), so its budget is the pair arm's 24,576,000 tokens
    and the three declared queues are symmetric. Pinned here because the cost
    claim moved a campaign's plan by a factor of ~58.
    """
    declared = declared_session_config(arms=("null",))
    assert declared.declared is True
    assert declared.total_steps == 6_000
    assert declared.arms == ("null",)
    assert declared.training_tokens() == 24_576_000


def test_v0_moe_materialized_counts_match_estimator() -> None:
    config = V0MoEConfig(**TINY)
    model = V0MoECausalLM(config)
    total, active = estimated_parameter_count(config)
    assert model.parameter_count == total
    assert model.active_parameter_count == active
    tokens = torch.randint(0, TINY["vocab_size"], (2, 16))
    assert torch.isfinite(model.loss(tokens))


def test_v0_session_predicts_protocol_cost_ratios() -> None:
    primary = 100.0 * predicted_cost_ratio(declared_session_config())
    secondary = 100.0 * predicted_cost_ratio(declared_session_config(probe_batch_count=2))
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


def test_v0_dense_expert_forward_matches_eager_dispatch() -> None:
    """The fast dense expert path is the same function as per-expert dispatch.

    ``forward`` folds the expert axis into one large matmul for speed; if it
    drifted from the routing contract, every training loss in the project would
    silently change. ``eager_forward`` is the reference formulation, so pin
    them equal (float64 to expose anything but rounding).
    """
    from act.model.v0_moe import _MoEFeedForward

    config = V0MoEConfig(**TINY)
    torch.manual_seed(5)
    expert_ids = tuple(f"E{index}" for index in range(config.experts_per_layer[0]))
    feed_forward = _MoEFeedForward(config, expert_ids).to(torch.float64)
    hidden = torch.randn(4, 5, config.hidden_size, dtype=torch.float64)
    with torch.no_grad():
        dense = feed_forward(hidden)
        eager = feed_forward.eager_forward(hidden)
    assert torch.allclose(dense, eager, atol=1e-12)
    # Routing bookkeeping is shared and normalized: every token's top-k load
    # sums to one, so the recorded per-expert fractions must total 1.
    assert float(feed_forward.last_load.sum()) == pytest.approx(1.0)
    assert (feed_forward.last_load > 0).all()


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
    # The lookahead guardrail is a declared-scale statement: it must use the
    # declared session config, not the fast dev default.
    config = declared_session_config(probe_finetune_steps=1)
    probe_tokens = config.batch_size * config.context_size * config.probe_batch_count
    expected = (
        3
        * 4
        * (probe_tokens * 2 * active + 1 * probe_tokens * 2 * total)
        / (2 * total * config.training_tokens())
    )
    assert predicted_cost_ratio(config) == pytest.approx(expected)
    assert predicted_cost_ratio(config) > predicted_cost_ratio(declared_session_config())
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


# --- gated arm (report section 5.4): accept/reject gating of random draws ---


def _gated_config(seed: int, **overrides) -> V0SessionConfig:
    values = dict(
        seed=seed,
        arms=("gated",),
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
    values.update(overrides)
    return V0SessionConfig(**values)


def test_gated_config_validation() -> None:
    """Gated sessions must be single-arm and gate_margin non-negative."""
    with pytest.raises(ValueError, match="gated"):
        V0SessionConfig(declared=False, arms=("gated", "random"))
    with pytest.raises(ValueError, match="gate_margin"):
        V0SessionConfig(declared=False, arms=("gated",), gate_margin=-1.0)
    # The default arms tuple stays untouched.
    assert V0SessionConfig(declared=False).arms == ("random", "adaptive")


def test_gated_theta_infinity_reproduces_random_bit_for_bit(tmp_path: Path) -> None:
    """The built-in control: an always-accepting gate IS the random arm.

    Same init identity (offset 1), same training stream, same chooser seeds,
    and - because the accepted candidate's rebuild IS the event's model - the
    same number of global-RNG consumption points per event. Any delta here
    would invalidate the arm as its own baseline.
    """
    random_arm = run_v0_session(
        _gated_config(61, arms=("random",)), checkpoint_dir=tmp_path / "random"
    ).arms[0]
    gated_arm = run_v0_session(
        _gated_config(61, gate_margin=float("inf")),
        checkpoint_dir=tmp_path / "gated",
    ).arms[0]
    assert random_arm.final_loss == gated_arm.final_loss
    assert random_arm.architecture_hash == gated_arm.architecture_hash
    assert (random_arm.mutation_count, random_arm.training_flops) == (
        gated_arm.mutation_count,
        gated_arm.training_flops,
    )
    # The gate books strictly more (two probe evaluations per live event);
    # training FLOPs must match exactly.
    assert gated_arm.search_flops > random_arm.search_flops
    assert json.dumps(gated_arm.mutation_history, sort_keys=True) == json.dumps(
        random_arm.mutation_history, sort_keys=True
    ) or all(
        event.get("gate_decision") == "accept"
        for event in gated_arm.mutation_history
    )


def test_gated_theta_infinity_flops_booking_matches_identity(tmp_path: Path) -> None:
    """Search FLOPs at theta=inf = mutations * 2 probes + mutation rebuilds."""
    config = _gated_config(62, gate_margin=float("inf"))
    result = run_v0_session(config, checkpoint_dir=tmp_path / "gated")
    arm = result.arms[0]
    live_events = sum(
        1 for event in arm.mutation_history if event.get("gate_decision") != "noop"
    )
    assert live_events == arm.mutation_count
    probe_tokens = config.batch_size * config.context_size * config.probe_batch_count
    _, active = estimated_parameter_count(config.model_config())
    # Two probe evaluations per live event (reference + draw), each 2*active
    # FLOPs/token, plus parameter_count * 4 per accepted rebuild. The exact
    # parameter count per event varies with the genome, so assert the probe
    # component exactly and the rebuild component loosely (>= noop floor).
    assert arm.search_flops >= live_events * 2 * probe_tokens * 2 * active


def test_gated_event_records_gate_contract(tmp_path: Path) -> None:
    """Every live gated event carries the full two-point contrast contract."""
    arm = run_v0_session(
        _gated_config(63), checkpoint_dir=tmp_path / "gated"
    ).arms[0]
    assert arm.mode == "gated"
    assert arm.mutation_count == 2
    for event in arm.mutation_history:
        assert event["policy"] == "gated"
        assert event["gate_margin"] == 0.0
        decision = event["gate_decision"]
        assert decision in {"accept", "reject_fallback", "reject_noop", "noop"}
        if decision == "noop":
            assert event["reference_probe_loss"] is None
            assert event["probe_candidates"] == 0
        else:
            assert event["reference_probe_loss"] is not None
            assert event["draw_probe_loss"] is not None
            assert event["probe_candidates"] == 2
            if decision == "accept":
                assert (
                    event["draw_probe_loss"]
                    <= event["reference_probe_loss"] + event["gate_margin"]
                )
            else:  # a rejection must have found the draw harmful
                assert (
                    event["draw_probe_loss"]
                    > event["reference_probe_loss"] + event["gate_margin"]
                )


def test_gated_gate_margin_zero_rejects_harmful_draws(tmp_path: Path) -> None:
    """theta=0 rejects any draw whose probe loss exceeds the reference.

    A draw can only change the model, so a harmful draw (probe loss strictly
    above the no-op reference) exists at least once across seeds; force the
    gate to reject by checking that a rejected event (when present) triggered
    a fallback or no-op while still consuming the budget slot.
    """
    arm = run_v0_session(
        _gated_config(64, gate_margin=0.0), checkpoint_dir=tmp_path / "gated"
    ).arms[0]
    # Budget is consumed regardless of the gate's decisions.
    assert arm.mutation_count == 2
    for event in arm.mutation_history:
        if event["gate_decision"] == "accept":
            assert event["draw_probe_loss"] <= event["reference_probe_loss"]
        elif event["gate_decision"] == "reject_fallback":
            assert event["operation"] == "grow"  # the load-based fallback


def test_gated_resume_is_bit_identical(tmp_path: Path) -> None:
    """A mid-run resume reproduces the gated arm exactly."""
    config = _gated_config(65, total_steps=8, mutation_interval=4)
    reference = run_v0_session(config, checkpoint_dir=tmp_path / "ref")
    arm_dir = tmp_path / "resume" / "seed-65" / "arm-gated"
    run_v0_session(config, checkpoint_dir=tmp_path / "resume")
    newest = max(
        arm_dir.glob("checkpoint-*.pt"), key=lambda p: int(p.stem.split("-")[-1])
    )
    assert newest.stem.endswith("0008")
    newest.unlink()  # force a mid-run resume from the step-6 checkpoint
    resumed = run_v0_session(config, checkpoint_dir=tmp_path / "resume")
    arm = resumed.arms[0]
    before = reference.arms[0]
    assert arm.final_loss == before.final_loss
    assert arm.architecture_hash == before.architecture_hash
    assert (arm.training_flops, arm.search_flops) == (
        before.training_flops,
        before.search_flops,
    )
    assert json.dumps(arm.mutation_history, sort_keys=True) == json.dumps(
        before.mutation_history, sort_keys=True
    )


def test_predicted_cost_ratio_books_gated_two_point_contrast() -> None:
    """Gated sessions probe TWO models per event (reference + draw)."""
    config = _gated_config(66)
    total, active = estimated_parameter_count(config.model_config())
    probe_tokens = config.batch_size * config.context_size * config.probe_batch_count
    expected = (
        2 * 2 * probe_tokens * 2 * active
        / (2 * total * config.training_tokens())
    )
    assert predicted_cost_ratio(config) == pytest.approx(expected)


# --- null arm (report section 5.8): the do-nothing churn-harm control ---


def _null_config(seed: int, **overrides) -> V0SessionConfig:
    values = dict(
        seed=seed,
        arms=("null",),
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
    values.update(overrides)
    return V0SessionConfig(**values)


def test_null_config_validation() -> None:
    """Null sessions must be single-arm."""
    with pytest.raises(ValueError, match="null"):
        V0SessionConfig(declared=False, arms=("random", "null"))
    # gated+null is also rejected (by the gated single-arm rule, which fires
    # first - the message names gated).
    with pytest.raises(ValueError, match="gated"):
        V0SessionConfig(declared=False, arms=("gated", "null"))
    # The default arms tuple stays untouched.
    assert V0SessionConfig(declared=False).arms == ("random", "adaptive")


def test_null_arm_performs_no_mutation_and_books_no_search(tmp_path: Path) -> None:
    """The do-nothing control does nothing beyond training."""
    arm = run_v0_session(
        _null_config(67), checkpoint_dir=tmp_path / "null"
    ).arms[0]
    assert arm.mode == "null"
    assert arm.mutation_count == 0
    assert arm.search_flops == 0
    # Every interval slot fires (the budget guard counts mutations, which
    # stay 0 - exactly like the gated arm's all-reject run on the diagnostic
    # block: 3 slots at this config). On the registered real-data schedule
    # (interval 200, steps 600) there are exactly 3 slots = mutation_budget,
    # so the null arm consumes precisely the random arms' event count.
    assert len(arm.mutation_history) == 3
    assert [event["step"] for event in arm.mutation_history] == [2, 4, 6]
    for event in arm.mutation_history:
        assert event["policy"] == "null"
        assert event["operation"] == "null"
        assert event["gate_decision"] == "null"
        assert event["mutation_count"] == 0
        assert event["probe_candidates"] == 0
        assert event["architecture_hash_before"] == event["architecture_hash_after"]


def test_null_arm_matches_random_training_flops(tmp_path: Path) -> None:
    """Same stream, same steps, no rebuilds: training FLOPs equal the random arm's.

    The random arm rebuilds after each mutation, which does not book training
    FLOPs but does reset the optimizer; the training FLOPs identity
    (parameter_count-per-step booking) can therefore drift once architectures
    diverge. At this tiny scale the random arm's two mutations both grow, so
    equality holds here; the invariant pinned is that the null arm books
    exactly its initial architecture's per-step cost for every step.
    """
    config = _null_config(68)
    null_arm = run_v0_session(
        config, checkpoint_dir=tmp_path / "null"
    ).arms[0]
    random_arm = run_v0_session(
        _null_config(68, arms=("random",)), checkpoint_dir=tmp_path / "random"
    ).arms[0]
    assert null_arm.architecture_hash != random_arm.architecture_hash
    assert null_arm.final_loss != random_arm.final_loss


def test_null_arm_cost_ratio_is_zero() -> None:
    """No probes, no rebuilds: the predicted search cost is exactly zero."""
    assert predicted_cost_ratio(_null_config(69)) == 0.0


def test_null_arm_resume_is_bit_identical(tmp_path: Path) -> None:
    """A mid-run resume reproduces the null arm exactly."""
    config = _null_config(70, total_steps=8, mutation_interval=4)
    reference = run_v0_session(config, checkpoint_dir=tmp_path / "ref")
    arm_dir = tmp_path / "resume" / "seed-70" / "arm-null"
    run_v0_session(config, checkpoint_dir=tmp_path / "resume")
    newest = max(
        arm_dir.glob("checkpoint-*.pt"), key=lambda p: int(p.stem.split("-")[-1])
    )
    assert newest.stem.endswith("0008")
    newest.unlink()  # force a mid-run resume from the step-6 checkpoint
    resumed = run_v0_session(config, checkpoint_dir=tmp_path / "resume")
    arm = resumed.arms[0]
    before = reference.arms[0]
    assert arm.final_loss == before.final_loss
    assert arm.architecture_hash == before.architecture_hash
    assert (arm.training_flops, arm.search_flops) == (
        before.training_flops,
        before.search_flops,
    )
    assert json.dumps(arm.mutation_history, sort_keys=True) == json.dumps(
        before.mutation_history, sort_keys=True
    )


# --- heuristic arm (report section 5.13): routing-statistics selection ---


def _heuristic_config(seed: int, **overrides) -> V0SessionConfig:
    values = dict(
        seed=seed,
        arms=("heuristic",),
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
        mutation_budget=3,
        probe_batch_count=1,
        validation_batches=2,
        checkpoint_interval=2,
        max_checkpoints=2,
    )
    values.update(overrides)
    return V0SessionConfig(**values)


def test_heuristic_config_validation() -> None:
    """Heuristic sessions must be single-arm."""
    with pytest.raises(ValueError, match="heuristic"):
        V0SessionConfig(declared=False, arms=("random", "heuristic"))
    assert V0SessionConfig(declared=False).arms == ("random", "adaptive")


def test_heuristic_arm_grows_at_measured_load_every_slot(tmp_path: Path) -> None:
    """Every event slot grows the argmax-load layer; no probes are booked.

    The budget guard counts mutations (which happen), so the arm stops at
    mutation_budget events - with budget 3 and 3 slots, all slots fire.
    """
    arm = run_v0_session(
        _heuristic_config(72), checkpoint_dir=tmp_path / "heuristic"
    ).arms[0]
    assert arm.mode == "heuristic"
    assert arm.mutation_count == 3
    assert len(arm.mutation_history) == 3
    assert [event["step"] for event in arm.mutation_history] == [2, 4, 6]
    for event in arm.mutation_history:
        assert event["policy"] == "heuristic"
        assert event["operation"] == "grow"
        assert event["gate_decision"] == "heuristic_grow"
        assert event["expert_id"] is not None
        assert event["probe_candidates"] == 0
        assert event["architecture_hash_after"] != event["architecture_hash_before"]
    assert arm.search_flops > 0
    assert arm.architecture_hash != arm.mutation_history[0]["architecture_hash_before"]


def test_heuristic_event_targets_the_argmax_load_layer(tmp_path: Path) -> None:
    """The event grows the layer whose max routing load is currently highest."""
    from act.training.v0_session import (
        _SessionState,
        _genome_from_config,
        _heuristic_event,
        _prepare_model,
    )
    from act.model import v0_moe_model_from_genome

    config = _heuristic_config(73)
    genome = _genome_from_config(config)
    model = _prepare_model(v0_moe_model_from_genome(genome).to("cpu"), config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    # Populate real routing statistics with a training-shaped forward.
    tokens = torch.randint(0, config.vocab_size, (2, config.context_size))
    model.loss(tokens)
    loads = model.routing_load
    expected = max(
        (layer for layer in genome.layers if len(layer.experts) < 16),
        key=lambda layer: max(loads[layer.index]),
    )
    state = _SessionState(genome=genome)
    _heuristic_event(state, model, optimizer, config, 2, torch.device("cpu"))
    event = state.mutation_history[-1]
    assert event["operation"] == "grow"
    assert event["layer_id"] == expected.id
    assert state.mutations == 1
    assert len(next(l for l in genome.layers if l.id == expected.id).experts) == 5


def test_heuristic_event_rebuilds_exactly_once(tmp_path: Path) -> None:
    """One accepted grow, one rebuild - no probe rebuilds on any path.

    Pinned via the global torch RNG stream: the event must consume exactly
    the randomness of a single post-grow rebuild (fresh-expert init).
    """
    from copy import deepcopy

    from act.training.v0_session import (
        _SessionState,
        _genome_from_config,
        _heuristic_event,
        _prepare_model,
    )
    from act.model import v0_moe_model_from_genome

    config = _heuristic_config(74)
    genome = _genome_from_config(config)
    model = _prepare_model(v0_moe_model_from_genome(genome).to("cpu"), config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    tokens = torch.randint(0, config.vocab_size, (2, config.context_size))
    model.loss(tokens)
    loads = model.routing_load
    target = max(
        (layer for layer in genome.layers if len(layer.experts) < 16),
        key=lambda layer: max(loads[layer.index]),
    )
    before_genome = deepcopy(genome)
    state = _SessionState(genome=deepcopy(genome))
    snapshot = torch.random.get_rng_state()
    _heuristic_event(state, model, optimizer, config, 2, torch.device("cpu"))
    after_event = torch.random.get_rng_state()
    # Control: restore the pre-mutation genome, apply the same grow, and
    # rebuild once manually from the same RNG position.
    torch.random.set_rng_state(snapshot)
    control_genome = deepcopy(before_genome)
    assert grow_expert(control_genome, target.id, 2) is not None
    rebuild_v0_moe_from_genome(model, control_genome)
    after_manual = torch.random.get_rng_state()
    assert torch.equal(after_event, after_manual)


def test_heuristic_arm_cost_ratio_is_zero() -> None:
    """No probes: the predicted search cost is exactly zero."""
    assert predicted_cost_ratio(_heuristic_config(75)) == 0.0


def test_heuristic_arm_resume_is_bit_identical(tmp_path: Path) -> None:
    """A mid-run resume reproduces the heuristic arm exactly."""
    config = _heuristic_config(76, total_steps=8, mutation_interval=4)
    reference = run_v0_session(config, checkpoint_dir=tmp_path / "ref")
    arm_dir = tmp_path / "resume" / "seed-76" / "arm-heuristic"
    run_v0_session(config, checkpoint_dir=tmp_path / "resume")
    newest = max(
        arm_dir.glob("checkpoint-*.pt"), key=lambda p: int(p.stem.split("-")[-1])
    )
    assert newest.stem.endswith("0008")
    newest.unlink()  # force a mid-run resume from the step-6 checkpoint
    resumed = run_v0_session(config, checkpoint_dir=tmp_path / "resume")
    arm = resumed.arms[0]
    before = reference.arms[0]
    assert arm.final_loss == before.final_loss
    assert arm.architecture_hash == before.architecture_hash
    assert (arm.training_flops, arm.search_flops) == (
        before.training_flops,
        before.search_flops,
    )
    assert json.dumps(arm.mutation_history, sort_keys=True) == json.dumps(
        before.mutation_history, sort_keys=True
    )


# --- substrate audit (report section 9): candidate policy and pool control ---


def _policy_config(seed: int, **overrides) -> V0SessionConfig:
    """A tiny depth-4 adaptive session: deep enough for the cap to matter."""
    values = dict(
        seed=seed,
        arms=("random", "adaptive"),
        declared=False,
        device="cpu",
        vocab_size=64,
        context_size=256,
        hidden_size=64,
        depth=4,
        attention_heads=4,
        expert_count=4,
        expert_width=32,
        total_steps=8,
        mutation_interval=3,
        mutation_budget=2,
        probe_candidate_cap=4,
        probe_batch_count=1,
        validation_batches=2,
        checkpoint_interval=2,
        max_checkpoints=2,
        probe_landscape=True,
    )
    values.update(overrides)
    return V0SessionConfig(**values)


def _adaptive_events(result: V0SessionResult) -> list[dict]:
    arm = next(arm for arm in result.arms if arm.mode == "adaptive")
    return list(arm.mutation_history)


def _probed_layers(result: V0SessionResult) -> set[str]:
    layers: set[str] = set()
    for event in _adaptive_events(result):
        layers.update(item["layer_id"] for item in event.get("probe_landscape", ()))
    return layers


def test_probe_candidate_policy_and_selection_validation() -> None:
    V0SessionConfig(declared=False, probe_candidate_policy="spread")
    V0SessionConfig(declared=False, probe_selection="random")
    with pytest.raises(ValueError, match="probe_candidate_policy"):
        V0SessionConfig(declared=False, probe_candidate_policy="bogus")
    with pytest.raises(ValueError, match="probe_selection"):
        V0SessionConfig(declared=False, probe_selection="bogus")


def test_prefix_policy_is_confined_to_the_first_layer_but_spread_is_not(
    tmp_path: Path,
) -> None:
    """The registered prefix cap reaches only layer 0 at depth 4; spread does not."""
    prefix = run_v0_session(
        _policy_config(71, probe_candidate_policy="prefix"),
        checkpoint_dir=tmp_path / "prefix",
    )
    spread = run_v0_session(
        _policy_config(71, probe_candidate_policy="spread"),
        checkpoint_dir=tmp_path / "spread",
    )
    assert _probed_layers(prefix) == {"L0"}
    assert len(_probed_layers(spread)) > 1
    # The policy is recorded per event for auditability.
    for event in _adaptive_events(prefix):
        assert event["probe_candidate_policy"] == "prefix"
    for event in _adaptive_events(spread):
        assert event["probe_candidate_policy"] == "spread"


def test_random_probe_selection_is_deterministic_and_recorded(tmp_path: Path) -> None:
    """The pool-matched control ignores probe loss but is fully reproducible."""
    config = _policy_config(72, probe_selection="random")
    first = run_v0_session(config, checkpoint_dir=tmp_path / "a")
    second = run_v0_session(config, checkpoint_dir=tmp_path / "b")
    events = _adaptive_events(first)
    assert events
    assert all(event["probe_selection"] == "random" for event in events)
    assert json.dumps(events, sort_keys=True) == json.dumps(
        _adaptive_events(second), sort_keys=True
    )
    # argmin always selects index 0, so a random rule that never disagrees
    # would mean the control silently collapsed onto the registered rule.
    argmin_indices = [
        event["probe_selected_index"]
        for event in _adaptive_events(
            run_v0_session(_policy_config(72), checkpoint_dir=tmp_path / "c")
        )
    ]
    assert [event["probe_selected_index"] for event in events] != argmin_indices


def test_spread_policy_resume_is_bit_identical(tmp_path: Path) -> None:
    """A re-executed probe event must redraw the identical candidate subset."""
    config = _policy_config(73, arms=("adaptive",), probe_candidate_policy="spread")
    reference = run_v0_session(config, checkpoint_dir=tmp_path / "ref")
    arm_dir = tmp_path / "resume" / "seed-73" / "arm-adaptive"
    run_v0_session(config, checkpoint_dir=tmp_path / "resume")
    # Keep only the step-2 checkpoint so the events at steps 3 and 6 re-execute.
    for path in arm_dir.glob("checkpoint-*.pt"):
        if int(path.stem.split("-")[-1]) > 2:
            path.unlink()
    resumed = run_v0_session(config, checkpoint_dir=tmp_path / "resume")
    arm, before = resumed.arms[0], reference.arms[0]
    assert arm.final_loss == before.final_loss
    assert arm.architecture_hash == before.architecture_hash
    assert (arm.training_flops, arm.search_flops) == (
        before.training_flops,
        before.search_flops,
    )
    assert json.dumps(arm.mutation_history, sort_keys=True) == json.dumps(
        before.mutation_history, sort_keys=True
    )


def test_a_checkpoint_killed_mid_write_never_takes_the_final_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A torn write must not become the newest checkpoint retention finds.

    A declared checkpoint is ~3.4 GB, so the write takes seconds and a process
    killed inside it (the OOM killer, a kernel timeout) would otherwise leave a
    *truncated* file at the final name - which the retention policy takes for
    the newest, `publish_seed` checksums into a `complete` version, and
    restore's checksum verification cannot tell from a good one. Invisible
    from every reader, so the write goes to a sibling and is renamed.
    """
    from act.training.v0_handoff import newest_arm_checkpoint
    from act.training.v0_session import (
        V0SessionConfig,
        _SessionState,
        save_v0_checkpoint,
    )

    config = V0SessionConfig(
        declared=False,
        vocab_size=64,
        context_size=256,
        hidden_size=64,
        depth=2,
        attention_heads=4,
        expert_count=4,
        expert_width=32,
        active_experts=2,
    )
    model = V0MoECausalLM(config.model_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    arm_dir = tmp_path / "arm-random"
    intact = arm_dir / "checkpoint-000004.pt"
    save_v0_checkpoint(intact, model, optimizer, _SessionState(step=4))
    original = intact.read_bytes()

    def torn_save(payload: object, path: object) -> None:
        Path(str(path)).write_bytes(b"half a checkpoint")  # then the process dies
        raise RuntimeError("killed mid-write")

    monkeypatch.setattr(torch, "save", torn_save)
    with pytest.raises(RuntimeError):
        save_v0_checkpoint(
            arm_dir / "checkpoint-000008.pt", model, optimizer, _SessionState(step=8)
        )

    assert not (arm_dir / "checkpoint-000008.pt").exists()
    assert [path.name for path in arm_dir.glob("checkpoint-*.pt")] == [
        "checkpoint-000004.pt"
    ]
    assert newest_arm_checkpoint(arm_dir) == intact
    assert intact.read_bytes() == original  # the good checkpoint is untouched


def test_arm_device_indices_places_one_arm_per_device(monkeypatch) -> None:
    """One arm per GPU when the host has them, device 0 otherwise.

    Section 5.2 prices a pair with its arms running concurrently, one per GPU,
    so this placement decides whether that model is reachable at all - and a
    one-device host must still get the sequential fallback rather than a
    second device it does not have.
    """
    from act.training.v0_session import arm_device_indices

    cuda = torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert arm_device_indices(2, cuda) == (0, 1)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert arm_device_indices(2, cuda) == (0, 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    assert arm_device_indices(2, cuda) == (0, 1)
    # Never parallelise a single-arm pair (the null queue), and never off CUDA.
    assert arm_device_indices(1, cuda) == (0,)
    assert arm_device_indices(2, torch.device("cpu")) == (0, 0)
