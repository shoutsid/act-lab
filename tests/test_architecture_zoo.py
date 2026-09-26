"""Tests for the architecture zoo and the static evaluation harness.

These pin the properties that made the first screening run uninterpretable:
the seed has to reach the initialization, loss has to come from a held-out
stream, and FLOP accounting has to distinguish architectures that differ in
active parameters.
"""

from __future__ import annotations

import numpy as np
import pytest

from act.search.architecture_eval import (
    RunConfig,
    estimate_dense_runtime_flops_per_token,
    estimate_flops_per_token,
    learning_rate_at,
    train_architecture,
)
from act.search.architecture_zoo import (
    DEFAULT_ACTIVE_EXPERTS,
    DEFAULT_DEPTHS,
    DEFAULT_EXPERT_COUNTS,
    DEFAULT_EXPERT_WIDTHS,
    DEFAULT_HIDDEN_SIZES,
    architecture_distance,
    cell_training_flops_per_token,
    crossing_balance,
    gene_value,
    generate_architecture_zoo,
    generate_crossed_zoo,
    generate_head_paired_zoo,
    generate_matched_zoo,
    head_pair_matching,
    make_moe_genome,
    matching_quality,
    min_pairwise_distance,
)
from act.search.architecture_zoo import _enumerate_cells, _stratify

TINY = dict(hidden_size=32, expert_count=4, active_experts=2, expert_width=32)
VOCAB = 256
CONTEXT = 16


def tiny_genome(**overrides):
    params = dict(
        seed=0,
        depth=2,
        hidden_size=32,
        expert_count=4,
        active_experts=2,
        expert_width=32,
        vocab_size=VOCAB,
        context_size=CONTEXT,
    )
    params.update(overrides)
    return make_moe_genome(**params)


def tiny_run(**overrides) -> RunConfig:
    params = dict(
        steps=5,
        batch_size=2,
        context_size=CONTEXT,
        warmup_steps=2,
        val_batches=2,
        val_holdout_tokens=256,
        seed=42,
    )
    params.update(overrides)
    return RunConfig(**params)


# --- zoo generation -------------------------------------------------------


def test_zoo_is_within_tolerance_and_spans_multiple_families():
    zoo = generate_architecture_zoo(target_params=100_000_000, tolerance=0.3, seed=42)
    assert 8 <= len(zoo) <= 16
    for candidate in zoo:
        assert abs(candidate.total_params - 100_000_000) / 100_000_000 <= 0.3
        assert candidate.active_params <= candidate.total_params
    assert len({candidate.family for candidate in zoo}) >= 4


def test_zoo_keeps_candidates_apart_structurally():
    zoo = generate_architecture_zoo(target_params=100_000_000, seed=42)
    assert min_pairwise_distance(zoo) >= 0.05


def test_zoo_genome_hash_is_deterministic():
    from act.topology.genome import genome_hash

    first = generate_architecture_zoo(target_params=100_000_000, seed=42)
    second = generate_architecture_zoo(target_params=100_000_000, seed=42)
    assert [c.name for c in first] == [c.name for c in second]
    assert [genome_hash(c.genome) for c in first] == [genome_hash(c.genome) for c in second]


def test_zoo_spans_active_parameter_range():
    """The point of the zoo is structural variation, not seed variation."""
    zoo = generate_architecture_zoo(target_params=100_000_000, seed=42)
    actives = [candidate.active_params for candidate in zoo]
    assert max(actives) / min(actives) > 2.0


# --- distance metric ------------------------------------------------------


# --- matched designs (brief section 21) ----------------------------------


def test_param_matched_zoo_holds_total_and_spreads_active():
    zoo = generate_matched_zoo(target=100_000_000, match_on="total", tolerance=0.03, limit=8)
    assert len(zoo) >= 6
    totals = [c.total_params for c in zoo]
    actives = [c.active_params for c in zoo]
    assert max(totals) / min(totals) <= 1.08
    assert max(actives) / min(actives) > 1.3


def test_flop_matched_zoo_holds_flops_and_spreads_total():
    zoo = generate_matched_zoo(target=2.6e8, match_on="flops", tolerance=0.03, limit=8)
    flops = [
        cell_training_flops_per_token(
            (c.total_params, c.active_params, c.genome.depth, c.genome.hidden_size), 1024
        )
        for c in zoo
    ]
    totals = [c.total_params for c in zoo]
    assert max(flops) / min(flops) <= 1.08
    assert max(totals) / min(totals) > 1.2


def test_matching_on_params_does_not_match_flops():
    """The reason section 15 asks for a measured cost axis: these are different."""
    zoo = generate_matched_zoo(target=100_000_000, match_on="total", tolerance=0.03, limit=8)
    flops = [
        cell_training_flops_per_token(
            (c.total_params, c.active_params, c.genome.depth, c.genome.hidden_size), 1024
        )
        for c in zoo
    ]
    assert max(flops) / min(flops) > 1.3


def test_matched_zoo_respects_the_memory_ceiling():
    """A matched design must not hand back a model the GPU cannot hold.

    Matching on the per-token budget leaves total parameters free to grow, so
    this is where the ceiling actually bites.
    """
    zoo = generate_matched_zoo(
        target=35_000_000, match_on="active", limit=8, max_total_params=46_000_000
    )
    assert zoo
    assert all(c.total_params <= 46_000_000 for c in zoo)


def test_matching_quality_reports_the_matched_axis():
    zoo = generate_matched_zoo(target=100_000_000, match_on="total", limit=8)
    quality = matching_quality(zoo, "total")
    assert quality["matched_spread_over_mean"] <= 0.10
    assert quality["varied_spread_ratio"] > 1.2
    assert quality["matched_min"] <= quality["matched_max"]


def test_matched_design_covers_the_varied_axis_it_claims_to_spread():
    """Regression: selecting end-first silently halved the range covered.

    Filling the slots by walking the matched band from one end does not work,
    because the diversity floor rejects most neighbours and every accepted cell
    then comes from the same end. At 78M +/-5% the band holds 1,059 cells
    spanning 18.0M..81.8M active parameters, and the end-first sweep returned
    18.0M..37.8M - so a design whose entire purpose is "matched total parameters,
    active parameters spread" would never once have tested a dense model.

    The assertion is coverage of the available band, not merely that the spread
    exceeds some ratio: the broken version already cleared 2.1x.
    """
    target, tolerance = 78_000_000, 0.05
    zoo = generate_matched_zoo(
        target=target, match_on="total", tolerance=tolerance, limit=8, min_distance=0.05
    )
    band = [
        cell
        for cell in _enumerate_cells(
            context_size=1024,
            vocab_size=8192,
            depths=DEFAULT_DEPTHS,
            hidden_sizes=DEFAULT_HIDDEN_SIZES,
            expert_counts=DEFAULT_EXPERT_COUNTS,
            active_experts_options=DEFAULT_ACTIVE_EXPERTS,
            expert_widths=DEFAULT_EXPERT_WIDTHS,
            max_total_params=150_000_000,
        )
        if abs(cell[0] - target) / target <= tolerance
    ]
    band_low = min(cell[1] for cell in band)
    band_high = max(cell[1] for cell in band)
    covered_low = min(c.active_params for c in zoo)
    covered_high = max(c.active_params for c in zoo)
    assert (covered_high - covered_low) / (band_high - band_low) > 0.8
    # The extremes must be represented, not merely approached: the band's
    # densest cell is fully dense (active == total), which is the interesting
    # opposite end from the sparsest expert-routed design.
    assert covered_low < band_low + 0.05 * (band_high - band_low)
    assert covered_high > band_high - 0.05 * (band_high - band_low)


def test_stratify_gives_each_bin_at_least_one_cell():
    chunks = _stratify(list(range(10)), 4)
    assert len(chunks) == 4
    assert [len(c) for c in chunks] == [2, 3, 2, 3]
    assert sorted(x for chunk in chunks for x in chunk) == list(range(10))


def test_stratify_keeps_the_extremes_reachable():
    chunks = _stratify(["low", "mid", "high"], 3)
    assert chunks[0] == ["low"]
    assert chunks[-1] == ["high"]


def test_stratify_degenerates_gracefully():
    assert _stratify([], 4) == [[]]
    assert _stratify([1, 2, 3], 1) == [[1, 2, 3]]
    # More bins than cells must not drop a cell or emit an empty bin.
    chunks = _stratify([1, 2], 5)
    assert sorted(x for chunk in chunks for x in chunk) == [1, 2]


def test_matched_design_selection_is_deterministic():
    """Same arguments must give the same design, or a run cannot be audited.

    Pinned because the selection is now stratified rather than a greedy sweep,
    and a non-deterministic tie-break in the binning would silently change which
    architectures an experiment actually ran.
    """
    kwargs = dict(target=78_000_000, match_on="total", tolerance=0.05, limit=8, min_distance=0.05)
    first = generate_matched_zoo(**kwargs)
    second = generate_matched_zoo(**kwargs)
    assert [(c.name, c.total_params, c.active_params) for c in first] == [
        (c.name, c.total_params, c.active_params) for c in second
    ]


def test_matched_design_has_no_duplicate_architectures():
    """`_stratify` can emit overlapping chunks when bins exceed the cell count."""
    from act.topology.genome import genome_hash

    zoo = generate_matched_zoo(
        target=78_000_000, match_on="total", tolerance=0.02, limit=8, min_distance=0.0
    )
    hashes = [genome_hash(candidate.genome) for candidate in zoo]
    assert len(hashes) == len(set(hashes)), "the same architecture was selected twice"


def test_matched_zoo_rejects_unknown_axis():
    import pytest

    with pytest.raises(ValueError):
        generate_matched_zoo(target=100_000_000, match_on="vibes")


def test_matched_zoo_errors_when_nothing_matches():
    import pytest

    with pytest.raises(ValueError):
        generate_matched_zoo(
            target=10_000_000_000, match_on="total", tolerance=0.001, limit=4
        )


def test_matched_zoo_candidates_are_valid_genomes():
    from act.topology.genome import genome_hash, validate_genome

    zoo = generate_matched_zoo(target=100_000_000, match_on="total", limit=6)
    hashes = set()
    for candidate in zoo:
        assert validate_genome(candidate.genome).valid
        hashes.add(genome_hash(candidate.genome))
    assert len(hashes) == len(zoo)


def test_crossed_zoo_is_balanced_on_both_factors():
    """The crossing's entire claim is that its two genes are independent.

    A matched design that spreads one axis does not control the others: at 78M
    the matched-total zoo has hidden size correlated with expert width at
    rho = -0.74, so no amount of seeding can tell them apart. Crossing makes the
    two named genes orthogonal by construction, and that property is what the
    attribution rests on, so it is asserted rather than assumed.
    """
    zoo = generate_crossed_zoo(
        target_params=78_000_000,
        tolerance=0.05,
        factors=("depth", "hidden_size"),
        replicates=2,
    )
    assert len(zoo) == 8
    balance = crossing_balance(zoo, ("depth", "hidden_size"))
    assert balance["available"]
    assert balance["balanced"] is True
    assert balance["spearman"] == pytest.approx(0.0)
    # Two levels each, four candidates each: 2 x 2 with two replicates.
    assert balance["levels"]["depth"]["5"] == 4
    assert balance["levels"]["depth"]["14"] == 4
    assert balance["levels"]["hidden_size"]["512"] == 4
    assert balance["levels"]["hidden_size"]["896"] == 4


def test_crossed_zoo_spreads_both_factors_widely():
    """Maximizing one factor's spread alone picks near-adjacent levels of the
    other: on this crossing it selects depth 5 against 6, the two most similar
    depths in the grid, because they leave the most room for expert width.
    Spread on both genes is the point, so the objective must contain both."""
    zoo = generate_crossed_zoo(
        target_params=78_000_000, tolerance=0.05, factors=("depth", "hidden_size")
    )
    depths = sorted({gene_value(candidate, "depth") for candidate in zoo})
    hidden = sorted({gene_value(candidate, "hidden_size") for candidate in zoo})
    assert depths[0] == min(DEFAULT_DEPTHS)
    assert depths[-1] == max(DEFAULT_DEPTHS)
    assert hidden[0] == min(DEFAULT_HIDDEN_SIZES)
    assert hidden[-1] == max(DEFAULT_HIDDEN_SIZES)


def test_crossed_zoo_holds_the_matched_budget():
    zoo = generate_crossed_zoo(
        target_params=78_000_000,
        tolerance=0.05,
        factors=("depth", "hidden_size"),
        replicates=2,
    )
    totals = [candidate.total_params for candidate in zoo]
    assert min(totals) >= 78_000_000 * 0.95
    assert max(totals) <= 78_000_000 * 1.05
    # The budget is paid for by the genes left free, so the crossed pair must
    # not be the thing that moved to keep it matched - that is the design.
    assert max(totals) / min(totals) < 1.10


def test_crossed_zoo_has_no_duplicate_architectures():
    from act.topology.genome import genome_hash, validate_genome

    zoo = generate_crossed_zoo(
        target_params=78_000_000,
        tolerance=0.05,
        factors=("depth", "hidden_size"),
        replicates=2,
    )
    hashes = [genome_hash(candidate.genome) for candidate in zoo]
    assert len(hashes) == len(set(hashes)), "the same architecture was selected twice"
    for candidate in zoo:
        assert validate_genome(candidate.genome).valid


def test_crossed_zoo_is_deterministic():
    first = generate_crossed_zoo(target_params=78_000_000, tolerance=0.05)
    second = generate_crossed_zoo(target_params=78_000_000, tolerance=0.05)
    assert [(c.name, c.total_params) for c in first] == [
        (c.name, c.total_params) for c in second
    ]


def test_crossed_zoo_rejects_a_bad_factor_pair():
    with pytest.raises(ValueError):
        generate_crossed_zoo(factors=("depth", "depth"))
    with pytest.raises(ValueError):
        generate_crossed_zoo(factors=("depth", "vibes"))


def test_crossed_zoo_errors_when_no_crossing_fits():
    """A budget with too few cells per level combination must fail loudly.

    Silently dropping to one replicate would make a "replicated" design that
    cannot separate a gene effect from within-cell variation, while still
    reporting itself as crossed.
    """
    with pytest.raises(ValueError):
        generate_crossed_zoo(
            target_params=78_000_000, tolerance=0.001, replicates=8
        )


def test_gene_value_reads_structure_off_the_genome():
    """Read from the genome, not parsed from the name, so a design's balance is
    checked against what the model will actually be built from."""
    zoo = generate_crossed_zoo(target_params=78_000_000, tolerance=0.05)
    candidate = zoo[0]
    assert gene_value(candidate, "depth") == candidate.genome.depth == len(
        candidate.genome.layers
    )
    assert gene_value(candidate, "hidden_size") == candidate.genome.hidden_size
    assert gene_value(candidate, "expert_count") == len(
        candidate.genome.layers[0].experts
    )
    assert gene_value(candidate, "active_experts") == (
        candidate.genome.layers[0].active_experts
    )
    assert gene_value(candidate, "expert_width") == max(
        expert.width for expert in candidate.genome.layers[0].experts
    )
    with pytest.raises(ValueError):
        gene_value(candidate, "vibes")


def test_crossing_balance_detects_a_leaky_design():
    """The balance check must fail when the design is not actually balanced.

    A detector that only ever agrees with the design it is checking is worth
    nothing, so this unbalances a crossing by hand and requires the report to
    say so.
    """
    zoo = generate_crossed_zoo(
        target_params=78_000_000, tolerance=0.05, factors=("depth", "hidden_size")
    )
    # Drop one level combination, leaving depth 14 at only two candidates.
    leaky = [
        candidate
        for candidate in zoo
        if not (candidate.genome.depth == 14 and candidate.genome.hidden_size == 512)
    ]
    balance = crossing_balance(leaky, ("depth", "hidden_size"))
    assert balance["available"]
    assert balance["balanced"] is False
    assert balance["spearman"] == pytest.approx(0.5)
    assert balance["levels"]["depth"]["14"] == 2


def test_crossing_balance_needs_two_factors():
    zoo = generate_crossed_zoo(target_params=78_000_000, tolerance=0.05)
    with pytest.raises(ValueError):
        crossing_balance(zoo, ("depth",))


def test_distance_is_zero_for_identical_genomes():
    genome = tiny_genome()
    assert architecture_distance(genome, genome) == 0.0


def test_distance_is_symmetric():
    a, b = tiny_genome(), tiny_genome(depth=4, hidden_size=48)
    assert architecture_distance(a, b) == pytest.approx(architecture_distance(b, a))


def test_distance_registers_module_type_change():
    """The previous metric hardcoded this term to zero."""
    a = tiny_genome()
    b = tiny_genome()
    b.layers[1].module = "gdn"
    assert architecture_distance(a, b) > 0.0


def test_distance_registers_expert_count_and_active_changes():
    base = tiny_genome()
    more_experts = tiny_genome(expert_count=8)
    more_active = tiny_genome(active_experts=4)
    assert architecture_distance(base, more_experts) > 0.0
    assert architecture_distance(base, more_active) > 0.0


def test_distance_registers_depth_and_width():
    base = tiny_genome()
    assert architecture_distance(base, tiny_genome(depth=6)) > 0.0
    assert architecture_distance(base, tiny_genome(hidden_size=64)) > 0.0


# --- FLOP accounting -----------------------------------------------------


def test_contracted_flops_scale_with_active_parameters():
    """Two candidates with equal total parameters must not report equal FLOPs."""
    from act.model.factory import v0_moe_config_from_genome

    sparse = tiny_genome(expert_count=8, active_experts=2)
    dense = tiny_genome(expert_count=8, active_experts=8)
    sparse_flops = estimate_flops_per_token(
        v0_moe_config_from_genome(sparse), context_size=CONTEXT
    )
    dense_flops = estimate_flops_per_token(
        v0_moe_config_from_genome(dense), context_size=CONTEXT
    )
    assert dense_flops > sparse_flops


def test_dense_runtime_cost_exceeds_contracted_cost_for_sparse_moe():
    """The dense-expert implementation bills all experts even when few are active."""
    from act.model.factory import v0_moe_config_from_genome

    config = v0_moe_config_from_genome(tiny_genome(expert_count=8, active_experts=2))
    contracted = estimate_flops_per_token(config, context_size=CONTEXT)
    runtime = estimate_dense_runtime_flops_per_token(config, context_size=CONTEXT)
    assert runtime > contracted


def test_fully_active_moe_has_equal_contract_and_runtime():
    from act.model.factory import v0_moe_config_from_genome

    config = v0_moe_config_from_genome(tiny_genome(expert_count=4, active_experts=4))
    assert estimate_flops_per_token(
        config, context_size=CONTEXT
    ) == pytest.approx(estimate_dense_runtime_flops_per_token(config, context_size=CONTEXT))


def test_longer_context_costs_more_attention_flops():
    from act.model.factory import v0_moe_config_from_genome

    config = v0_moe_config_from_genome(tiny_genome())
    short = estimate_flops_per_token(config, context_size=16)
    long = estimate_flops_per_token(config, context_size=64)
    assert long > short


# --- learning-rate schedule ----------------------------------------------


def test_lr_schedule_warms_up_then_decays():
    run = tiny_run(steps=10, warmup_steps=4, learning_rate=1e-3, min_lr_ratio=0.1)
    assert learning_rate_at(0, run) < run.learning_rate
    assert learning_rate_at(3, run) == pytest.approx(run.learning_rate)
    assert learning_rate_at(9, run) < learning_rate_at(4, run)
    assert learning_rate_at(9, run) >= run.learning_rate * run.min_lr_ratio - 1e-12


# --- harness correctness -------------------------------------------------


@pytest.mark.parametrize(
    "override, message",
    [
        ({"steps": 0}, "steps, batch_size"),
        ({"batch_size": 0}, "steps, batch_size"),
        ({"context_size": 0}, "steps, batch_size"),
        ({"val_batches": 0}, "steps, batch_size"),
        ({"warmup_steps": -1}, "warmup_steps"),
        ({"learning_rate": 0}, "learning_rate"),
        ({"learning_rate": float("inf")}, "learning_rate"),
        ({"min_lr_ratio": -0.1}, "min_lr_ratio"),
        ({"min_lr_ratio": 1.1}, "min_lr_ratio"),
        ({"weight_decay": -0.1}, "weight_decay"),
        ({"grad_clip": float("nan")}, "grad_clip"),
        ({"val_holdout_tokens": -1}, "val_holdout_tokens"),
        ({"gdn_chunk_size": -1}, "gdn_chunk_size"),
        ({"steps": 1.5}, "must be integers"),
    ],
)
def test_run_config_rejects_invalid_training_settings(override, message):
    with pytest.raises(ValueError, match=message):
        tiny_run(**override)


def test_run_config_allows_disabled_gradient_clipping_and_chunking():
    run = tiny_run(grad_clip=0.0, gdn_chunk_size=0)
    assert run.grad_clip == 0.0
    assert run.gdn_chunk_size == 0


def test_run_config_counts_tokens():
    run = tiny_run(steps=5, batch_size=2, context_size=16)
    assert run.train_tokens == 160
    assert run.val_tokens == 2 * 2 * 16


def test_same_seed_reproduces_validation_loss():
    """The recorded seed has to determine the initialization run over run."""
    genome = tiny_genome()
    first = train_architecture(genome, run=tiny_run(seed=7), device="cpu")
    second = train_architecture(genome, run=tiny_run(seed=7), device="cpu")
    assert first.validation_loss == pytest.approx(second.validation_loss, rel=0, abs=1e-12)


def test_different_seeds_give_different_initializations():
    genome = tiny_genome()
    first = train_architecture(genome, run=tiny_run(seed=7), device="cpu")
    second = train_architecture(genome, run=tiny_run(seed=8), device="cpu")
    assert first.validation_loss != second.validation_loss


def test_validation_loss_is_reported_separately_from_training_loss():
    result = train_architecture(tiny_genome(), run=tiny_run(), device="cpu")
    assert result.validation_loss > 0
    assert result.final_train_loss > 0
    assert result.mean_train_loss > 0
    assert result.steps == 5


def test_mixed_precision_is_recorded_with_the_result():
    """A loss is only comparable to another loss measured in the same precision."""
    result = train_architecture(tiny_genome(), run=tiny_run(mixed_precision="bf16"), device="cpu")
    assert result.run_config["mixed_precision"] == "bf16"


def test_invalid_mixed_precision_is_rejected():
    with pytest.raises(ValueError):
        train_architecture(
            tiny_genome(), run=tiny_run(mixed_precision="float8"), device="cpu"
        )


def test_fp16_requires_cuda():
    """The repository's own autocast helper refuses fp16 on CPU; that is upheld."""
    with pytest.raises(ValueError):
        train_architecture(
            tiny_genome(), run=tiny_run(mixed_precision="fp16"), device="cpu"
        )


def test_activation_checkpointing_is_numerically_transparent():
    """Checkpointing trades compute for memory, so it must not move the loss.

    If it did, a candidate's footprint would change its score and every
    memory-motivated setting would become a confound.
    """
    plain = train_architecture(
        tiny_genome(), run=tiny_run(seed=11), device="cpu"
    )
    checkpointed = train_architecture(
        tiny_genome(), run=tiny_run(seed=11, activation_checkpointing=True), device="cpu"
    )
    assert checkpointed.validation_loss == pytest.approx(plain.validation_loss, abs=1e-6)
    assert checkpointed.final_train_loss == pytest.approx(plain.final_train_loss, abs=1e-6)


def test_result_records_the_full_run_configuration():
    """A result has to be auditable without rerunning it."""
    result = train_architecture(tiny_genome(), run=tiny_run(), device="cpu")
    for key in ("steps", "batch_size", "context_size", "learning_rate", "seed", "val_batches"):
        assert key in result.run_config
    assert result.val_source


def test_holdout_validation_is_disjoint_from_training(tmp_path):
    from act.search.architecture_eval import _prepare_streams

    stream = np.arange(5000, dtype=np.int32) % VOCAB
    path = tmp_path / "tokens.npy"
    np.save(path, stream)

    run = RunConfig(
        steps=5,
        batch_size=2,
        context_size=16,
        val_batches=2,
        val_holdout_tokens=1000,
        seed=1,
        train_data_path=str(path),
    )
    train, val, source = _prepare_streams(VOCAB, run)

    assert source.startswith("holdout:")
    assert train.numel() + val.numel() <= stream.size
    assert train.tolist() == stream[:-1000].tolist()
    assert val.tolist() == stream[-1000:].tolist()


# --- capability probes on finalists (brief section 19) --------------------


def probe_batches():
    import torch

    generator = torch.Generator().manual_seed(5)
    return {
        "story": [torch.randint(0, VOCAB, (2, CONTEXT), generator=generator) for _ in range(2)],
        "ood": [torch.randint(0, VOCAB, (2, CONTEXT), generator=generator)],
    }


def test_capability_probes_are_recorded_when_supplied():
    result = train_architecture(
        tiny_genome(),
        name="probed",
        run=RunConfig(steps=3, batch_size=2, context_size=CONTEXT, val_batches=2, seed=1),
        device="cpu",
        probes=probe_batches(),
    )
    assert sorted(result.capability) == ["ood", "story"]
    entry = result.capability["story"]
    assert entry["name"] == "story"
    # Accuracy and loss are counted over next-token positions, so a sequence of
    # length T contributes T-1: 2 batches x 2 sequences x 15.
    assert entry["tokens"] == 2 * 2 * (CONTEXT - 1)
    assert entry["loss"] > 0
    assert 0.0 <= entry["accuracy"] <= 1.0


def test_capability_probes_are_absent_by_default():
    """Probing during screening would add a second selection objective."""
    result = train_architecture(
        tiny_genome(),
        name="unprobed",
        run=RunConfig(steps=3, batch_size=2, context_size=CONTEXT, val_batches=2, seed=1),
        device="cpu",
    )
    assert result.capability == {}


def test_ledgers_written_before_capability_existed_still_load(tmp_path):
    """Backward compatibility: every v1/v2 ledger predates this field."""
    from dataclasses import asdict

    import json as json_module

    from act.search.architecture_eval import load_results

    result = train_architecture(
        tiny_genome(),
        name="legacy",
        run=RunConfig(steps=3, batch_size=2, context_size=CONTEXT, val_batches=2, seed=1),
        device="cpu",
    )
    row = asdict(result)
    row.pop("capability")
    path = tmp_path / "legacy.jsonl"
    path.write_text(json_module.dumps(row) + "\n")
    assert load_results(path)[0].capability == {}


@pytest.mark.parametrize(
    "tokens, message",
    [
        (np.array([0.0, 1.5, 2.0], dtype=np.float32), "integer token ids"),
        (np.array([0, -1, 2], dtype=np.int32), "outside"),
        (np.array([0, VOCAB, 2], dtype=np.int64), "outside"),
    ],
)
def test_token_loader_rejects_invalid_ids_before_training(tmp_path, tokens, message):
    from act.search.architecture_eval import _load_tokens

    path = tmp_path / "invalid-tokens.npy"
    np.save(path, tokens)

    with pytest.raises(ValueError, match=message):
        _load_tokens(str(path), VOCAB, length=1, seed=3)


def test_a_stream_that_serves_the_schedule_once_is_admitted(tmp_path):
    """The admission bar is the schedule, not twice it, and no window repeats.

    The bar used to be ``train_tokens * 2``, which refused the horizon arm's
    12,000-step design on a 40.0M-token stream even though it consumes 24.6M and
    reuses nothing - sixteen runs raised at data load and the arm trained none of
    what it was launched for. These are the arithmetic facts that made the
    mismatch invisible until hours into a kernel session.
    """
    from act.search.architecture_eval import _prepare_streams, required_stream_tokens

    run = RunConfig(
        steps=200,
        batch_size=2,
        context_size=16,
        val_batches=2,
        val_holdout_tokens=500,
        seed=1,
        train_data_path=str(tmp_path / "tokens.npy"),
    )
    needed = required_stream_tokens(run, val_tokens_available=None)
    np.save(run.train_data_path, np.arange(needed, dtype=np.int32) % VOCAB)

    train, val, source = _prepare_streams(VOCAB, run)

    assert source.startswith("holdout:")
    # Every window the schedule asks for exists, and none is reused.
    assert train.numel() // (run.context_size + 1) == run.steps * run.batch_size
    assert train.numel() + val.numel() <= needed


def test_a_stream_too_short_for_the_schedule_is_still_refused(tmp_path):
    """The bar moved, it did not disappear: a single token short is refused."""
    from act.search.architecture_eval import _prepare_streams, required_stream_tokens

    run = RunConfig(
        steps=200,
        batch_size=2,
        context_size=16,
        val_batches=2,
        val_holdout_tokens=500,
        seed=1,
        train_data_path=str(tmp_path / "short.npy"),
    )
    np.save(
        run.train_data_path,
        np.arange(required_stream_tokens(run, val_tokens_available=None) - 1, dtype=np.int32)
        % VOCAB,
    )
    with pytest.raises(ValueError, match="need"):
        _prepare_streams(VOCAB, run)


def test_the_holdout_is_only_taken_when_validation_cannot_stand_alone(tmp_path):
    """A big enough val file costs the training stream nothing; a small one does.

    The horizon arm's val-tokens.npy holds 7,190 tokens and cannot support a
    16-batch evaluation at context 1024, so its runs hold out from the training
    stream - which is why its 12,000-step requirement is 26.6M rather than 24.6M.
    """
    from act.search.architecture_eval import (
        required_holdout_tokens,
        required_stream_tokens,
        required_train_tokens,
    )

    run = RunConfig(steps=12000, batch_size=2, context_size=1024, val_batches=16)
    assert required_train_tokens(run) == 24_600_000
    assert required_holdout_tokens(run, val_tokens_available=7_190) == 2_000_000
    assert required_holdout_tokens(run, val_tokens_available=40_000) == 0
    assert required_stream_tokens(run, val_tokens_available=7_190) == 26_600_000
    assert required_stream_tokens(run, val_tokens_available=40_000) == 24_600_000


def test_undersized_validation_file_falls_back_to_holdout(tmp_path):
    """The repository's val-tokens.npy is far too small to evaluate at 1024."""
    from act.search.architecture_eval import _prepare_streams

    train_path = tmp_path / "train.npy"
    val_path = tmp_path / "val.npy"
    np.save(train_path, np.arange(5000, dtype=np.int32) % VOCAB)
    np.save(val_path, np.arange(20, dtype=np.int32) % VOCAB)

    run = RunConfig(
        steps=5,
        batch_size=2,
        context_size=16,
        val_batches=2,
        val_holdout_tokens=1000,
        seed=1,
        train_data_path=str(train_path),
        val_data_path=str(val_path),
    )
    _, _, source = _prepare_streams(VOCAB, run)
    assert source.startswith("holdout:")


# --- head-paired design: the one contrast that can be matched exactly ---------
#
# The model family fixes heads = hidden_size / 64, so every earlier design in this
# project confounds residual-stream width with head count, and the two readings
# imply opposite advice (one costs parameters, one is free). Head count does not
# cost parameters, so a paired design can hold cost *identical* within a pair
# rather than landing candidates in a band. These tests pin that property, because
# it is the entire reason the design exists.


def head_paired():
    return generate_head_paired_zoo(
        head_counts=(8, 16),
        pairs=4,
        hidden_sizes=(512, 896),
        depths=(10,),
    )


def test_head_paired_holds_cost_identical_within_a_pair():
    zoo = head_paired()
    matching = head_pair_matching(zoo, (8, 16))
    assert matching["all_complete"]
    assert matching["all_exactly_matched"]
    for entry in matching["pairs"]:
        assert entry["head_counts"] == [8, 16]
        assert entry["total_exactly_matched"]
        assert entry["active_exactly_matched"]


def test_head_paired_matches_total_params_more_tightly_than_any_band():
    """The matching guarantee is exactness, not a tighter tolerance."""
    zoo = head_paired()
    matching = head_pair_matching(zoo, (8, 16))
    # Every other design this project ran matched within 5-10%; this is under 5%.
    assert matching["total_spread_ratio"] < 1.05


def test_head_paired_holds_depth_constant():
    """Depth is a real lever, so a width contrast must not be confounded with it."""
    zoo = head_paired()
    assert {gene_value(candidate, "depth") for candidate in zoo} == {10}


def test_head_paired_is_balanced_on_both_factors():
    zoo = head_paired()
    balance = crossing_balance(zoo, ("hidden_size", "attention_heads"))
    assert balance["balanced"]
    # Levels are string-keyed so the report can go straight into JSON.
    assert balance["levels"]["hidden_size"] == {"512": 4, "896": 4}
    assert balance["levels"]["attention_heads"] == {"8": 4, "16": 4}


def test_head_paired_names_encode_heads_because_names_would_collide():
    """Without the head suffix the two members of a pair share a name.

    Ledger rows are keyed by candidate name, so a collision would silently merge a
    pair into one architecture and destroy the design.
    """
    zoo = head_paired()
    names = [candidate.name for candidate in zoo]
    assert len(set(names)) == len(names)
    assert all("_Hd" in name for name in names)
    # The base part of the name is shared within a pair, which is the point.
    pairs = {}
    for candidate in zoo:
        pairs.setdefault(candidate.name.rsplit("_Hd", 1)[0], []).append(candidate)
    assert len(pairs) == 4
    assert all(len(members) == 2 for members in pairs.values())


def test_head_paired_refuses_to_silently_drop_a_hidden_size():
    """A narrowing must be loud, not a quiet fallback.

    7 divides 896 but not 512. Filtering would leave the design with 896 only -
    still a balanced-looking factorial, but with no width contrast in it, and
    reporting success. That is the failure this raises on instead.
    """
    with pytest.raises(ValueError, match="admit no head count"):
        generate_head_paired_zoo(
            head_counts=(7, 14), pairs=2, hidden_sizes=(512, 896), depths=(10,)
        )


def test_head_paired_raises_when_no_head_count_fits_the_band():
    with pytest.raises(ValueError, match="admits every head count"):
        generate_head_paired_zoo(
            head_counts=(3, 5), pairs=2, hidden_sizes=(512, 896), depths=(10,)
        )


def test_head_paired_needs_at_least_two_head_counts():
    with pytest.raises(ValueError, match="at least two head counts"):
        generate_head_paired_zoo(head_counts=(8,), pairs=2, depths=(10,))


def test_head_count_is_free_so_a_pair_costs_the_same_either_way():
    """The fact the design is built on, asserted directly rather than assumed."""
    from act.search.architecture_zoo import _estimate_params

    totals = {
        _estimate_params(
            depth=10, hidden_size=896, expert_count=8, active_experts=4,
            expert_width=512, heads=heads,
        )
        for heads in (4, 8, 14, 16, 28)
    }
    assert len(totals) == 1


def test_default_crossing_keeps_its_original_names():
    """Adding a head suffix must not rename existing designs' candidates.

    Ledgers on disk are keyed by these names, so a suffix leaking into a design
    that does not vary head count would orphan every result already recorded.
    """
    zoo = generate_crossed_zoo(factors=("depth", "hidden_size"), replicates=2)
    assert zoo
    assert not any("_Hd" in candidate.name for candidate in zoo)


def test_head_count_can_be_crossed_when_a_design_asks_for_it():
    """The gene is reachable as a crossed factor, which is how a width-vs-heads
    attribution becomes expressible at all."""
    zoo = generate_crossed_zoo(
        factors=("hidden_size", "attention_heads"),
        replicates=2,
        head_counts=(8, 16),
        hidden_sizes=(512, 896),
        depths=(8,),
    )
    balance = crossing_balance(zoo, ("hidden_size", "attention_heads"))
    assert balance["balanced"]
    assert {gene_value(candidate, "attention_heads") for candidate in zoo} == {8, 16}
    assert {gene_value(candidate, "depth") for candidate in zoo} == {8}


def test_head_count_is_free_in_flops_too_not_only_parameters():
    """The claim the paired design rests on, checked across every cost convention.

    If head count moved parameters or either FLOP accounting, the within-pair
    matching would be a band like everywhere else and the design's stronger
    guarantee would be an illusion. Measured throughput is the one quantity that
    can legitimately differ (more heads, more kernel launches), and the ledger
    records it per run rather than this test assuming it away.
    """
    from act.search.architecture_eval import (
        estimate_dense_runtime_flops_per_token,
        estimate_flops_per_token,
        v0_moe_config_from_genome,
    )

    zoo = generate_head_paired_zoo(
        head_counts=(8, 16), pairs=4, hidden_sizes=(512, 896), depths=(10,)
    )
    by_base: dict[str, list] = {}
    for candidate in zoo:
        by_base.setdefault(candidate.name.rsplit("_Hd", 1)[0], []).append(candidate)
    assert len(by_base) == 4

    for base, members in by_base.items():
        assert len(members) == 2
        conventions = set()
        for member in members:
            config = v0_moe_config_from_genome(member.genome)
            conventions.add(
                (
                    member.total_params,
                    member.active_params,
                    round(estimate_flops_per_token(config), 6),
                    round(estimate_dense_runtime_flops_per_token(config), 6),
                )
            )
        assert len(conventions) == 1, (base, conventions)
