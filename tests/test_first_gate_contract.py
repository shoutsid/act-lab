from act.evaluation import summarize_experiment
from act.evolution import ExperimentConfig, run_experiment
from act.topology import (
    create_genome,
    merge_experts,
    rewire_layer,
    split_expert,
)


def test_genome_json_round_trip_and_structural_mutations() -> None:
    genome = create_genome(4, depth=2, ffn="moe")
    restored = type(genome).from_dict(genome.to_dict())
    assert restored.to_json() == genome.to_json()
    assert split_expert(restored, "L0", "L0E0", step=2) == "L0E4"
    assert merge_experts(restored, "L0", "L0E0", "L0E4")
    assert rewire_layer(restored, "L1", "L0")


def test_rewire_rejects_self_and_unknown_edges() -> None:
    genome = create_genome(5, depth=2, ffn="moe")
    assert not rewire_layer(genome, "L0", "L0")
    assert not rewire_layer(genome, "L1", "missing")


def test_evaluation_summary_contains_cost_and_delta() -> None:
    result = run_experiment(ExperimentConfig(seed=8, steps=3, mutation_interval=2, mutation_budget=1))
    summary = summarize_experiment(result)
    assert summary["arms"]["adaptive"]["validation_flops"] > 0
    assert isinstance(summary["adaptive_minus_random"], float)
