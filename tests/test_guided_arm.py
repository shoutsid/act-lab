import math

import pytest

from act.evolution import ExperimentConfig, run_experiment, run_gate_study
from act.evolution.experiment import _mutation_candidates, _parse_arms
from act.topology import create_genome


def test_arms_spec_is_validated() -> None:
    assert _parse_arms("fixed,random") == ("fixed", "random")
    with pytest.raises(ValueError):
        _parse_arms("")
    with pytest.raises(ValueError):
        _parse_arms("fixed,fixed")
    with pytest.raises(ValueError):
        _parse_arms("bogus")
    with pytest.raises(ValueError):
        run_experiment(
            ExperimentConfig(steps=3, mutation_interval=2, arms="fixed,bogus")
        )


def test_guided_arm_matches_random_budget_and_records_probe() -> None:
    result = run_experiment(
        ExperimentConfig(
            seed=5,
            steps=12,
            mutation_interval=3,
            mutation_budget=2,
            arms="random,guided",
        )
    )
    arms = {arm.mode: arm for arm in result.arms}
    assert set(arms) == {"random", "guided"}
    assert arms["random"].mutation_count == arms["guided"].mutation_count == 2
    assert arms["guided"].search_flops > arms["random"].search_flops
    assert math.isfinite(arms["guided"].final_loss)
    initial = create_genome(
        5,
        hidden_size=24,
        vocab_size=32,
        depth=1,
        ffn="moe",
        expert_count=4,
        active_experts=2,
    )
    for index, event in enumerate(arms["guided"].mutation_history):
        assert event["policy"] == "guided"
        assert math.isfinite(float(event["probe_loss"]))
        assert event["probe_candidates"] > 0
        assert event["probe_batch_count"] == 4
    assert arms["guided"].mutation_history[0]["probe_candidates"] == len(
        _mutation_candidates(initial)
    )


def test_guided_arm_is_deterministic() -> None:
    config = ExperimentConfig(
        seed=7,
        steps=6,
        mutation_interval=3,
        mutation_budget=1,
        arms="random,guided",
    )
    assert run_experiment(config).to_dict() == run_experiment(config).to_dict()


def test_guided_gate_study_scores_guided_against_random() -> None:
    config = ExperimentConfig(
        steps=6, mutation_interval=3, mutation_budget=1, arms="random,guided"
    )
    study = run_gate_study(config, seeds=(1, 2), treatment="guided")
    assert study.treatment == "guided"
    assert study.matched_mutation_budget
    summary = study.to_dict()
    assert summary["treatment"] == "guided"
    assert summary == run_gate_study(config, seeds=(1, 2), treatment="guided").to_dict()
