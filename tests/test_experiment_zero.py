import math

from act.evolution import ExperimentConfig, run_experiment, run_gate_study


def test_experiment_zero_uses_matched_mutation_budget() -> None:
    result = run_experiment(
        ExperimentConfig(steps=12, mutation_interval=3, mutation_budget=2)
    )
    arms = {arm.mode: arm for arm in result.arms}
    assert set(arms) == {"fixed", "random", "adaptive"}
    assert arms["fixed"].mutation_count == 0
    assert arms["random"].mutation_count == arms["adaptive"].mutation_count == 2
    assert all(math.isfinite(arm.final_loss) for arm in arms.values())
    assert all(arm.training_flops > 0 for arm in arms.values())
    assert all(arm.validation_loss == arm.final_loss for arm in arms.values())
    assert all(
        arm.validation_flops
        == arms[arm.mode].active_parameter_count * 2 * 12 * 2 * 4
        for arm in arms.values()
    )
    assert all(arm.active_parameter_count > 0 for arm in arms.values())
    assert all(arm.validation_flops > 0 for arm in arms.values())
    assert arms["fixed"].architecture_distance == 0
    assert all(
        arm.architecture_distance >= 0
        for arm in (arms["random"], arms["adaptive"])
    )
    assert arms["random"].search_flops > 0
    assert arms["adaptive"].search_flops > 0
    assert len(arms["random"].mutation_history) == 2
    assert len(arms["adaptive"].mutation_history) == 2
    assert all(
        event["architecture_hash_before"] != event["architecture_hash_after"]
        or event["operation"] in {"grow", "prune"}
        for event in arms["adaptive"].mutation_history
    )
    for mode in ("random", "adaptive"):
        records = arms[mode].importance_records
        assert records
        assert all(record["architecture_hash"] for record in records)
        assert all(
            math.isfinite(float(record[field]))
            for record in records
            for field in (
                "routing_load",
                "gradient_norm",
                "parameter_norm",
                "fisher_ema",
                "temporal_stability",
            )
        )


def test_experiment_zero_is_deterministic() -> None:
    config = ExperimentConfig(seed=17, steps=12, mutation_interval=6, mutation_budget=2)
    first = run_experiment(config).to_dict()
    second = run_experiment(config).to_dict()
    assert first == second


def test_experiment_zero_adaptive_gate_probe() -> None:
    wins = 0
    for seed in (1, 2, 3, 42):
        arms = {
            arm.mode: arm
            for arm in run_experiment(ExperimentConfig(seed=seed)).arms
        }
        assert arms["random"].mutation_count == arms["adaptive"].mutation_count == 3
        wins += arms["adaptive"].final_loss < arms["random"].final_loss
    assert wins >= 3


def test_gate_study_serializes_reproducible_summary() -> None:
    study = run_gate_study(
        ExperimentConfig(steps=12, mutation_interval=6, mutation_budget=2),
        seeds=(1, 2),
    )
    assert study.seeds == (1, 2)
    assert study.adaptive_wins >= 0
    assert len(study.experiments) == 2
    assert study.matched_mutation_budget
    assert study.delta_stddev >= 0.0
    assert study.to_dict()["win_rate"] == study.win_rate
    assert study.to_dict() == run_gate_study(
        ExperimentConfig(steps=12, mutation_interval=6, mutation_budget=2),
        seeds=(1, 2),
    ).to_dict()


def test_experiment_zero_can_run_on_cuda_when_available() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    result = run_experiment(
        ExperimentConfig(
            device="cuda",
            steps=2,
            mutation_interval=2,
            mutation_budget=1,
            validation_batches=1,
        )
    )
    assert all(math.isfinite(arm.final_loss) for arm in result.arms)