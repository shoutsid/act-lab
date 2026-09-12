import json

from act.evolution import ExperimentConfig, append_result, experiment_hash, run_experiment


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