"""Deterministic Pareto population selection for topology search."""

from dataclasses import dataclass

from act.topology import ArchitectureGenome, genome_hash


@dataclass(frozen=True)
class PopulationCandidate:
    genome: ArchitectureGenome
    quality: float
    active_params: int
    search_flops: int

    @property
    def architecture_hash(self) -> str:
        return genome_hash(self.genome)


def dominates(left: PopulationCandidate, right: PopulationCandidate) -> bool:
    """Return whether left is no worse on every objective and better on one."""
    left_values = (left.quality, -left.active_params, -left.search_flops)
    right_values = (right.quality, -right.active_params, -right.search_flops)
    return all(a >= b for a, b in zip(left_values, right_values)) and any(
        a > b for a, b in zip(left_values, right_values)
    )


def pareto_frontier(
    candidates: tuple[PopulationCandidate, ...] | list[PopulationCandidate],
) -> tuple[PopulationCandidate, ...]:
    """Return a stable, hash-deduplicated non-dominated frontier."""
    unique: dict[str, PopulationCandidate] = {}
    for candidate in candidates:
        current = unique.get(candidate.architecture_hash)
        if current is None or _sort_key(candidate) < _sort_key(current):
            unique[candidate.architecture_hash] = candidate
    values = tuple(unique.values())
    return tuple(
        sorted(
            (candidate for candidate in values if not any(
                dominates(other, candidate) for other in values if other is not candidate
            )),
            key=_sort_key,
        )
    )


def select_population(
    candidates: tuple[PopulationCandidate, ...] | list[PopulationCandidate],
    size: int,
) -> tuple[PopulationCandidate, ...]:
    """Select at most size candidates, retaining frontier diversity first."""
    if size < 1:
        raise ValueError("population size must be positive")
    frontier = pareto_frontier(candidates)
    if len(frontier) >= size:
        return frontier[:size]
    selected = list(frontier)
    remaining = sorted(
        (candidate for candidate in candidates if candidate not in selected),
        key=_sort_key,
    )
    return tuple((selected + remaining)[:size])


def _sort_key(candidate: PopulationCandidate) -> tuple[float, int, int, str]:
    return (
        -candidate.quality,
        candidate.active_params,
        candidate.search_flops,
        candidate.architecture_hash,
    )
