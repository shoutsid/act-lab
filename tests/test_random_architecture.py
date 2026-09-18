"""Tests for the random architecture-search baseline (brief section 11).

The baseline has to be genuinely unoptimised and genuinely from the same space
as the curated zoo, or the random-versus-manual comparison means nothing.
"""

from __future__ import annotations

import pytest

from act.search.architecture_zoo import (
    architecture_distance,
    enumerate_search_space,
    generate_architecture_zoo,
)
from act.search.random_architecture import RandomArchitectureSearch
from act.topology.genome import genome_hash, validate_genome


def sampler(**overrides) -> RandomArchitectureSearch:
    params = dict(
        target_params=100_000_000,
        tolerance=0.15,
        size=8,
        seed=0,
        max_total_params=150_000_000,
    )
    params.update(overrides)
    return RandomArchitectureSearch(**params)


def test_sample_returns_the_requested_count():
    assert len(sampler(size=8).sample()) == 8
    assert len(sampler(size=3).sample()) == 3


def test_sample_is_reproducible_from_the_seed():
    first = [c.name for c in sampler().sample()]
    second = [c.name for c in sampler().sample()]
    assert first == second


def test_different_seeds_draw_different_architectures():
    assert [c.name for c in sampler().sample()] != [c.name for c in sampler().sample(seed=99)]


def test_sample_without_replacement():
    zoo = sampler(size=12).sample()
    names = [c.name for c in zoo]
    assert len(set(names)) == len(names)


def test_drawn_candidates_are_valid_and_distinct_genomes():
    zoo = sampler().sample()
    hashes = set()
    for candidate in zoo:
        assert validate_genome(candidate.genome).valid
        hashes.add(genome_hash(candidate.genome))
    assert len(hashes) == len(zoo)


def test_draw_respects_the_parameter_budget_band():
    search = sampler(tolerance=0.10)
    for candidate in search.sample():
        offset = abs(candidate.total_params - search.target_params) / search.target_params
        assert offset <= 0.10


def test_space_is_the_declared_search_space():
    """The baseline must not see a wider or narrower space than the curated zoo.

    Both restrict the same enumerated grid to a budget band, so a cell inside one
    band and outside the other is a difference of tolerance, not of space.
    """
    search = sampler(tolerance=1.0)
    declared = enumerate_search_space(
        context_size=search.context_size, vocab_size=search.vocab_size,
        depths=search.depths, hidden_sizes=search.hidden_sizes,
        expert_counts=search.expert_counts,
        active_experts_options=search.active_experts_options,
        expert_widths=search.expert_widths,
        max_total_params=search.max_total_params,
    )
    assert len(search.space()) == len(declared)


def test_curated_zoo_cells_live_in_the_same_space():
    """Every curated candidate must be drawable by the baseline in principle."""
    search = sampler(tolerance=1.0)
    space = {
        (c[2], c[3], c[4], c[5], c[6])
        for c in search.space()
    }
    zoo = generate_architecture_zoo(target_params=100_000_000, tolerance=0.3, limit=14)
    for candidate in zoo:
        layer = candidate.genome.layers[0]
        key = (
            candidate.genome.depth,
            candidate.genome.hidden_size,
            len(layer.experts),
            layer.active_experts,
            layer.experts[0].width,
        )
        assert key in space, candidate.name


def test_random_draw_spans_compute_not_just_parameters():
    """A uniform draw is a real baseline only if it actually varies cost."""
    zoo = sampler(size=12, tolerance=1.0).sample()
    actives = [c.active_params for c in zoo]
    assert max(actives) / min(actives) > 2.0


def test_describe_space_reports_the_band():
    description = sampler().describe_space()
    assert description["cells"] > 0
    assert description["total_spread_ratio"] > 1.0
    assert description["target_params"] == 100_000_000


def test_empty_space_is_an_error_not_an_empty_draw():
    with pytest.raises(ValueError):
        sampler(target_params=10**10, tolerance=0.001).sample()


def test_baseline_applies_no_evolutionary_operator():
    """No growth, pruning, cloning, merging, splitting or rewiring: genomes are
    sampled once and never changed."""
    zoo = sampler().sample()
    frozen = [genome_hash(candidate.genome) for candidate in zoo]
    for candidate, digest in zip(zoo, frozen):
        assert genome_hash(candidate.genome) == digest


def test_sample_materializes_real_genomes_not_placeholder_objects():
    zoo = sampler().sample()
    assert all(candidate.genome.depth == len(candidate.genome.layers) for candidate in zoo)
    assert all(candidate.active_params <= candidate.total_params for candidate in zoo)
    assert all(
        architecture_distance(candidate.genome, candidate.genome) == 0.0
        for candidate in zoo
    )
