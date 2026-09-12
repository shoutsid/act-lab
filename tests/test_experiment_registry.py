import json

from act.evolution import (
    ExperimentConfig,
    append_result,
    append_study,
    experiment_hash,
    run_experiment,
    run_gate_study,
    study_hash,
)


def test_registry_is_reproducible_and_append_only(tmp_path) -> None:
    result = run_experiment(ExperimentConfig(seed=12, steps=3, mutation_interval=2, mutation_budget=1))
    path = tmp_path / "experiments.jsonl"
    first = append_result(path, result)
    second = append_result(path, result)

    assert first["experiment_hash"] == second["experiment_hash"] == experiment_hash(result)
    assert first["config_hash"]
    assert first["dataset_hash"]
    assert first["tokenizer_hash"]
    assert first["architecture_hashes"]["adaptive"]
    assert first["tokens"] == 72
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 2
    assert records[0] == records[1]


def test_registry_persists_reproducible_gate_studies(tmp_path) -> None:
    study = run_gate_study(
        ExperimentConfig(steps=3, mutation_interval=2, mutation_budget=1),
        seeds=(1, 2),
    )
    path = tmp_path / "experiments.jsonl"
    first = append_study(path, study)
    second = append_study(path, study)

    assert first["record_type"] == "gate_study"
    assert first["study_hash"] == second["study_hash"] == study_hash(study)
    assert first["seeds"] == [1, 2]
    assert first["adaptive_wins"] == study.adaptive_wins
    assert first["matched_mutation_budget"] is True
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0] == records[1]