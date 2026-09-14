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


# --- null arm (report section 5.9): the do-nothing churn-harm control -------


def test_null_arm_is_a_budget_matched_no_op() -> None:
    """The null arm consumes every event slot and mutates nothing."""
    result = run_experiment(
        ExperimentConfig(
            seed=8,
            steps=12,
            mutation_interval=3,
            mutation_budget=2,
            arms="random,null",
        )
    )
    arms = {arm.mode: arm for arm in result.arms}
    assert set(arms) == {"random", "null"}
    assert arms["random"].mutation_count == 2
    assert arms["null"].mutation_count == 0
    # Two event slots consumed and recorded as no-ops.
    assert len(arms["null"].mutation_history) == 2
    for event in arms["null"].mutation_history:
        assert event["policy"] == "null"
        assert event["operation"] == "null"
        assert event["architecture_hash_before"] == event["architecture_hash_after"]
    assert arms["null"].search_flops == 0
    assert arms["null"].architecture_hash != arms["random"].architecture_hash


def test_null_arm_reproduces_fixed_trajectory_loss() -> None:
    """Same init, same stream, no mutations: null must equal fixed on loss.

    (They can differ in recorded bookkeeping - fixed never even enters the
    event branch - but the trained weights are identical.)
    """
    result = run_experiment(
        ExperimentConfig(
            seed=9,
            steps=12,
            mutation_interval=3,
            mutation_budget=2,
            arms="fixed,null",
        )
    )
    arms = {arm.mode: arm for arm in result.arms}
    assert arms["fixed"].final_loss == arms["null"].final_loss
    assert arms["fixed"].architecture_hash == arms["null"].architecture_hash
    assert arms["fixed"].training_flops == arms["null"].training_flops
    assert arms["null"].mutation_history  # the null arm records its no-ops
    assert not arms["fixed"].mutation_history


def test_null_study_scores_null_against_random() -> None:
    config = ExperimentConfig(
        steps=6, mutation_interval=3, mutation_budget=1, arms="random,null"
    )
    study = run_gate_study(config, seeds=(1, 2), treatment="null")
    assert study.treatment == "null"
    assert study.matched_mutation_budget
    summary = study.to_dict()
    assert summary["treatment"] == "null"
    assert summary == run_gate_study(config, seeds=(1, 2), treatment="null").to_dict()


def test_null_and_adaptive_share_one_experiment_block() -> None:
    """A three-arm block pairs adaptive-vs-random and null-vs-random per seed."""
    config = ExperimentConfig(
        steps=6,
        mutation_interval=3,
        mutation_budget=1,
        arms="random,adaptive,null",
        independent_arm_rng=True,
    )
    study = run_gate_study(config, seeds=(3,), treatment="null")
    arms = {arm.mode: arm for arm in study.experiments[0].arms}
    assert set(arms) == {"random", "adaptive", "null"}
    assert arms["random"].mutation_count == arms["adaptive"].mutation_count == 1
    assert arms["null"].mutation_count == 0
    assert len(arms["null"].mutation_history) == 1
    # The budget check that would abort a treatment-vs-random study is the
    # null arm's slot-count check instead.
    assert study.matched_mutation_budget


def test_three_arm_block_replicates_the_committed_decoupled_block() -> None:
    """Adding the null arm must not perturb adaptive/random (report 9.8/5.9).

    The section-5.9 study runs the registered section-5.2 config as a 3-arm
    block. Rebuild init consumes only per-(arm, event) re-seeds of the global
    RNG, the null arm consumes none, and arm order keeps indices 0/1 on
    random/adaptive - so the adaptive-vs-random comparison must reproduce the
    committed decoupled recheck of study 3 bit-for-bit on anchor seed 3000.
    """
    registered = dict(
        seed=3000,
        hidden_size=24,
        vocab_size=32,
        depth=1,
        expert_count=4,
        active_experts=2,
        batch_size=2,
        sequence_length=12,
        steps=48,
        mutation_interval=12,
        mutation_budget=3,
        learning_rate=0.0003,
        validation_batches=4,
        adaptive_policy="probe_loss",
        adaptive_expert_metric="random",
        probe_candidate_cap=4,
        probe_batch_count=2,
        independent_arm_rng=True,
        device="cpu",
    )
    two_arm = {
        arm.mode: arm.final_loss
        for arm in run_experiment(ExperimentConfig(arms="random,adaptive", **registered)).arms
    }
    result = run_experiment(ExperimentConfig(arms="random,adaptive,null", **registered))
    arms = {arm.mode: arm for arm in result.arms}
    # Adding the null arm perturbs the registered comparison not at all -
    # the environment-independent form of the replication claim (absolute
    # constants are environment-sensitive at the 1e-6 level, report 9.11).
    assert arms["adaptive"].final_loss == two_arm["adaptive"]
    assert arms["random"].final_loss == two_arm["random"]
    assert arms["null"].mutation_count == 0
    assert len(arms["null"].mutation_history) == 3
    assert arms["null"].search_flops == 0
