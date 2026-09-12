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
