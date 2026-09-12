import pytest

from act.evolution import (
    PopulationCandidate,
    dominates,
    pareto_frontier,
    select_population,
)
from act.topology import create_genome


def _candidate(seed: int, quality: float, active: int, flops: int) -> PopulationCandidate:
    return PopulationCandidate(create_genome(seed, hidden_size=8, vocab_size=16), quality, active, flops)


def test_pareto_frontier_keeps_quality_efficiency_tradeoffs() -> None:
    efficient = _candidate(1, 0.8, 10, 10)
    quality = _candidate(2, 1.0, 20, 20)
    dominated = _candidate(3, 0.7, 20, 20)
    frontier = pareto_frontier([efficient, quality, dominated])
    assert {item.architecture_hash for item in frontier} == {
        efficient.architecture_hash,
        quality.architecture_hash,
    }
    assert dominates(quality, dominated)


def test_population_selection_is_deterministic_and_bounded() -> None:
    candidates = [_candidate(seed, 0.5 + seed / 10, 10 + seed, 100 - seed) for seed in range(4)]
    first = select_population(candidates, 2)
    second = select_population(list(reversed(candidates)), 2)
    assert first == second
    assert len(first) == 2


def test_population_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="population size"):
        select_population([], 0)
