import math

from act.evolution import ExperimentConfig, run_experiment, run_gate_study
from act.evolution.experiment import _mutation_candidates
from act.topology import create_genome


def _initial_candidate_count() -> int:
    genome = create_genome(
        3,
        hidden_size=24,
        vocab_size=32,
        depth=1,
        ffn="moe",
        expert_count=4,
        active_experts=2,
    )
    return len(_mutation_candidates(genome))


def test_adaptive_probe_policy_is_bounded_and_matched() -> None:
    result = run_experiment(
        ExperimentConfig(
            seed=3,
            steps=12,
            mutation_interval=3,
            mutation_budget=2,
            arms="random,adaptive",
        )
    )
    arms = {arm.mode: arm for arm in result.arms}
    assert arms["random"].mutation_count == arms["adaptive"].mutation_count == 2
    initial_candidates = _initial_candidate_count()
    for event in arms["adaptive"].mutation_history:
        assert event["policy"] == "adaptive_probe_loss"
        assert event["probe_candidates"] == min(4, initial_candidates)
        assert event["probe_batch_count"] == 2
        assert math.isfinite(float(event["probe_loss"]))
    assert arms["adaptive"].search_flops > arms["random"].search_flops
    assert math.isfinite(arms["adaptive"].final_loss)


def test_adaptive_signal_policy_stays_available() -> None:
    config = ExperimentConfig(
        seed=11,
        steps=6,
        mutation_interval=3,
        mutation_budget=1,
        adaptive_policy="signal",
        arms="random,adaptive",
    )
    result = run_experiment(config)
    arms = {arm.mode: arm for arm in result.arms}
    assert arms["random"].mutation_count == arms["adaptive"].mutation_count == 1
    assert all("policy" not in event for event in arms["adaptive"].mutation_history)
    assert run_experiment(config).to_dict() == result.to_dict()


def test_adaptive_probe_policy_is_deterministic() -> None:
    config = ExperimentConfig(
        seed=13,
        steps=6,
        mutation_interval=3,
        mutation_budget=1,
        arms="random,adaptive",
    )
    assert run_experiment(config).to_dict() == run_experiment(config).to_dict()


def test_study_reports_search_flops_by_arm() -> None:
    config = ExperimentConfig(
        steps=6,
        mutation_interval=3,
        mutation_budget=1,
        arms="random,adaptive",
    )
    study = run_gate_study(config, seeds=(21, 22))
    totals = study.search_flops_by_arm
    assert set(totals) == {"random", "adaptive"}
    assert totals["adaptive"] > totals["random"]
    assert study.treatment_search_overhead_ratio > 1.0
    summary = study.to_dict()
    assert summary["search_flops_by_arm"] == totals
    assert summary["treatment_search_overhead_ratio"] == (
        study.treatment_search_overhead_ratio
    )


def _probed_layers(experiment) -> list[str]:
    layers: list[str] = []
    for arm in experiment.arms:
        if arm.mode != "adaptive":
            continue
        for event in arm.mutation_history:
            layers.extend(item["layer_id"] for item in event.get("probe_landscape", ()))
    return layers


# The candidate cap used to truncate a layer-major list, so at depth > 1 a
# bounded probe could never reach past the first layer. The spread policy
# samples the whole feasible set instead.
_SPREAD_BASE = dict(
    seed=5,
    steps=12,
    mutation_interval=3,
    mutation_budget=3,
    depth=4,
    hidden_size=24,
    expert_count=4,
    probe_candidate_cap=2,
    probe_batch_count=1,
    probe_landscape=True,
    arms="random,adaptive",
)


def test_prefix_candidate_policy_is_confined_to_the_first_layer() -> None:
    prefix_layers = _probed_layers(run_experiment(ExperimentConfig(**_SPREAD_BASE)))
    assert len(set(prefix_layers)) == 1


def test_spread_candidate_policy_reaches_beyond_the_first_layer() -> None:
    spread = ExperimentConfig(**_SPREAD_BASE, probe_candidate_policy="spread")
    spread_layers = _probed_layers(run_experiment(spread))
    assert len(set(spread_layers)) > 1
    # Per-event subsets are seeded from (config.seed + step): reproducible.
    assert _probed_layers(run_experiment(spread)) == spread_layers


def test_spread_policy_keeps_budgets_matched_and_is_recorded() -> None:
    spread = ExperimentConfig(**_SPREAD_BASE, probe_candidate_policy="spread")
    arms = {arm.mode: arm for arm in run_experiment(spread).arms}
    assert arms["random"].mutation_count == arms["adaptive"].mutation_count
    for event in arms["adaptive"].mutation_history:
        assert event["probe_candidate_policy"] == "spread"


def test_probe_selection_rules_differ_and_are_deterministic() -> None:
    # Same candidate pool, same probe cost; only the decision rule differs.
    base = dict(
        seed=7,
        steps=12,
        mutation_interval=3,
        mutation_budget=3,
        depth=4,
        hidden_size=24,
        expert_count=4,
        probe_candidate_cap=4,
        probe_batch_count=1,
        probe_landscape=True,
        arms="random,adaptive",
    )

    def selections(policy: str):
        experiment = run_experiment(ExperimentConfig(**base, probe_selection=policy))
        arm = next(arm for arm in experiment.arms if arm.mode == "adaptive")
        return tuple(event["probe_selected_index"] for event in arm.mutation_history)

    argmin_selections = selections("argmin")
    assert argmin_selections and all(index == 0 for index in argmin_selections)
    # Random draws are reproducible across runs (seeded per event).
    assert selections("random") == selections("random")
    assert selections("argmax") != argmin_selections


def test_probe_selection_is_recorded_in_history() -> None:
    experiment = run_experiment(
        ExperimentConfig(**_SPREAD_BASE, probe_selection="random")
    )
    arm = next(arm for arm in experiment.arms if arm.mode == "adaptive")
    assert all(
        event["probe_selection"] == "random" for event in arm.mutation_history
    )


def test_config_rejects_invalid_probe_budgets() -> None:
    import pytest

    with pytest.raises(ValueError):
        run_experiment(ExperimentConfig(steps=1, probe_batch_count=0))
    with pytest.raises(ValueError):
        run_experiment(ExperimentConfig(steps=1, probe_candidate_cap=0))
    with pytest.raises(ValueError):
        run_experiment(ExperimentConfig(steps=1, adaptive_policy="bogus"))
    with pytest.raises(ValueError):
        run_experiment(ExperimentConfig(steps=1, probe_candidate_cap=0, validation_batches=1))
    with pytest.raises(ValueError):
        run_experiment(ExperimentConfig(steps=1, probe_candidate_policy="bogus"))
    with pytest.raises(ValueError):
        run_experiment(ExperimentConfig(steps=1, probe_selection="bogus"))


def test_probe_batch_count_clamps_to_validation_batches() -> None:
    result = run_experiment(
        ExperimentConfig(
            seed=31,
            steps=3,
            mutation_interval=2,
            mutation_budget=1,
            validation_batches=1,
            arms="random,adaptive",
        )
    )
    arms = {arm.mode: arm for arm in result.arms}
    assert arms["random"].mutation_count == arms["adaptive"].mutation_count == 1
    assert arms["adaptive"].mutation_history[0]["probe_batch_count"] == 1
