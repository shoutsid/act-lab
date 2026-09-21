"""Tests for the space-typicality analysis.

The script decides whether a chosen architecture set occupies a distinguishable
part of the search space, and its output is used to argue that a random-vs-manual
comparison is or is not confounded by hindsight. So the parts tested are the ones
that could make it say the wrong thing: the percentile convention at the extremes,
the spread statistic's insensitivity to a single outlier, and - for the section 11
budget check - whether the matched draw is really the closest one on offer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# Imported by module name rather than as ``scripts.space_typicality``: ``scripts``
# is not a package and the repository root is not on ``sys.path`` under a plain
# ``pytest`` run, so the dotted form made this whole module uncollectable - the
# suite reported an error it could not attribute to a test, and none of the tests
# below had run at all.
from space_typicality import (  # noqa: E402
    DRIVER_RANDOM_TOLERANCE,
    random_arm_command,
    GENES,
    budget_matching_draw,
    population_structure,
    sample_statistics,
    set_total_params,
    structure_of_set,
    typicality,
)


def test_sample_statistics_reports_mean_and_pairwise_spread_per_gene():
    sample = [(4, 100), (8, 200), (12, 300)]
    stats = sample_statistics(sample)
    assert stats["depth_mean"] == pytest.approx(8.0)
    # Mean pairwise |difference|: (4+8+4)/3 for depths, (100+200+100)/3 for widths.
    assert stats["depth_spread"] == pytest.approx(16 / 3)
    assert stats["hidden_size_mean"] == pytest.approx(200.0)
    assert stats["hidden_size_spread"] == pytest.approx(400 / 3)


def test_spread_responds_to_an_outlier_less_than_a_range_does():
    """A range is set entirely by the extreme pair; spread weighs every pair.

    Not a claim that spread is outlier-immune - one point among five moves it a
    lot - but that its *relative* response is smaller, which is why the report
    uses it to describe coverage of a set rather than its extremes.
    """
    clustered = [(8, 500), (9, 500), (8, 520), (9, 520)]
    with_outlier = clustered + [(60, 5000)]

    def depth_range(sample):
        values = [depth for depth, _ in sample]
        return max(values) - min(values)

    spread_growth = (
        sample_statistics(with_outlier)["depth_spread"]
        / sample_statistics(clustered)["depth_spread"]
    )
    range_growth = depth_range(with_outlier) / depth_range(clustered)
    assert range_growth > spread_growth
    # The spread stays far below the range it is drawn from, because most pairs
    # in the set are still close together.
    assert sample_statistics(with_outlier)["depth_spread"] < depth_range(with_outlier) / 2


def test_typicality_puts_a_population_typical_set_near_fifty():
    """A set whose mean *is* the population mean must read near the middle.

    This is the calibration the whole report rests on: if a set with no location
    signal came back at a percentile extreme, every real finding from this script
    would be an artifact of the convention.
    """
    population = [(depth, 512) for depth in range(4, 21) for _ in range(4)]
    population_mean = sum(depth for depth, _ in population) / len(population)
    typical = [(int(population_mean), 512)] * 3
    result = typicality(typical, population, draws=2000, seed=0)
    assert 20 < result["depth_mean"]["percentile"] < 80


def test_typicality_flags_a_set_pinned_to_one_extreme():
    population = [(depth, 512) for depth in range(4, 21) for _ in range(4)]
    lowest = sorted(population)[:3]
    result = typicality(lowest, population, draws=2000, seed=0)
    assert result["depth_mean"]["percentile"] < 5
    assert result["depth_spread"]["percentile"] < 5


def test_typicality_is_extreme_in_both_directions_not_just_low():
    """A set at the top of the space must read near 100, not near 0.

    The percentile convention is `<=`, so an off-by-one in the comparison would
    make every unusual set look extreme in the same direction and turn a real
    'unusually diverse' finding into a spurious 'unusually clustered' one.
    """
    population = [(depth, 512) for depth in range(4, 21) for _ in range(4)]
    widest = population[-12:]
    result = typicality(widest, population, draws=2000, seed=0)
    assert result["depth_mean"]["percentile"] > 95


def test_typicality_rejects_a_sample_larger_than_the_population():
    with pytest.raises(ValueError):
        typicality([(1, 1), (2, 2)], [(1, 1)], draws=10)


def test_typicality_rejects_an_empty_sample():
    with pytest.raises(ValueError):
        typicality([], [(1, 1)], draws=10)


def test_structure_of_set_reads_a_real_zoo_definition():
    from pathlib import Path

    definition = Path("artifacts/arch-zoo-100m-v3-crossed/zoo_definition.json")
    if not definition.exists():
        pytest.skip("crossed zoo definition not present")
    sample = structure_of_set(definition)
    assert len(sample) == 8
    # The crossed design is a balanced 2x2, so both levels of both genes appear
    # exactly four times; that is what makes it usable as an attribution design.
    for index, gene in enumerate(GENES):
        levels = {}
        for entry in sample:
            levels[entry[index]] = levels.get(entry[index], 0) + 1
        assert sorted(levels.values()) == [4, 4], (gene, levels)


def test_set_total_params_sums_the_definition():
    from pathlib import Path

    definition = Path("artifacts/arch-zoo-100m-v1/zoo_definition.json")
    if not definition.exists():
        pytest.skip("v1 zoo definition not present")
    total = set_total_params(definition)
    # The curated set's mean is ~86M, so the total must be near n x mean, not n x
    # the band centre - the whole point of matching on the sum.
    assert 1.0e9 < total < 1.25e9, total


def test_budget_matching_draw_finds_the_closest_draw():
    """The reported seed has to be the best of the scanned ones, not just a good one."""
    target = 1_121_409_984  # v1's curated total, which a band-matched draw overshoots
    match = budget_matching_draw(
        target_total=target, size=13, seeds=range(24), target_params=100_000_000,
        tolerance=0.30,
    )
    assert match["scanned"] == 24
    assert match["seed"] in range(24)
    # Brute force the same scan independently: the choice must be the minimiser.
    best = min(
        (
            row for row in _scan(target, 13, range(24))
        ),
        key=lambda row: (abs(row["ratio"] - 1.0), row["seed"]),
    )
    assert match["seed"] == best["seed"]
    assert match["total_params"] == best["total_params"]
    assert match["ranked"][0]["seed"] == match["seed"]
    assert match["ranked"] == sorted(
        match["ranked"], key=lambda row: (abs(row["ratio"] - 1.0), row["seed"])
    )


def test_the_launch_command_carries_every_draw_affecting_flag():
    """A seed alone does not reproduce a draw; the band it was scanned at does.

    The same seed draws different architectures at the driver's default band, and
    measured against v1's curated total that is 1.1435x against the matched
    0.9996x - a 14% compute difference introduced by an omission. So the command is
    built rather than transcribed, and the band is a required argument.
    """
    command = random_arm_command(
        seed=151, size=13, target_params=100_000_000, tolerance=0.30
    )
    assert "--design random" in command
    assert "--zoo-seed 151" in command
    assert "--random-tolerance 0.3" in command
    assert "--candidates 13" in command
    assert "--target-params 100000000" in command
    assert "--max-total-params 150000000" in command
    assert f"--random-tolerance {DRIVER_RANDOM_TOLERANCE}" not in command
    with pytest.raises(TypeError):
        random_arm_command(seed=1, size=1, target_params=1)


def test_the_distribution_is_reported_over_every_scanned_draw():
    """The closest draw is not the typical one, and the report must not imply it is.

    The band constrains each candidate, not the set total, so a scan that printed
    only its closest few would make a 12% median overshoot look like a tail case.
    The distribution is therefore computed over *all* scanned draws - including the
    ones ranked too far from the target to be listed - which is the difference
    between 'the draw we can pick' and 'what a draw costs by default'. The label
    used to print the fifth-closest draw as the worst scanned one, which understated
    the spread by an order of magnitude.
    """
    target = 1_121_409_984
    match = budget_matching_draw(
        target_total=target, size=13, seeds=range(24), target_params=100_000_000,
        tolerance=0.30,
    )
    rows = _scan(target, 13, range(24))
    ratios = sorted(row["ratio"] for row in rows)
    dist = match["distribution"]
    assert set(dist) == {"mean", "median", "p10", "p90", "min", "max"}
    assert dist["min"] == pytest.approx(ratios[0])
    # The maximum is over every scanned draw, not over the five that are listed.
    assert dist["max"] == pytest.approx(ratios[-1])
    assert dist["max"] >= match["ranked"][-1]["ratio"]
    assert dist["median"] == pytest.approx(sum(ratios) / len(ratios), rel=0.2)
    assert dist["p10"] <= dist["median"] <= dist["p90"]
    # The closest draw is at or below the distribution's own centre, and for this
    # band it is well below: matching is the exception, not what the band gives.
    assert match["ratio"] <= dist["median"]


def _scan(target_total, size, seeds):
    """Every scanned draw, via the same public entry point one seed at a time."""
    rows = []
    for seed in seeds:
        single = budget_matching_draw(
            target_total=target_total, size=size, seeds=[seed],
            target_params=100_000_000, tolerance=0.30,
        )
        rows.append(
            {
                "seed": single["seed"],
                "total_params": single["total_params"],
                "ratio": single["ratio"],
            }
        )
    return rows


def test_budget_matching_draw_returns_the_structure_it_matched():
    """Matching on budget can move coverage, so the drawn set is returned.

    Section 11's hypothesised mechanism for a designed set is coverage, so a
    budget-matched draw that also happened to be unusually diverse would need to be
    reported rather than assumed neutral - which is only possible if the structure
    comes back with the seed.
    """
    match = budget_matching_draw(
        target_total=1_121_409_984, size=13, seeds=range(4), target_params=100_000_000,
        tolerance=0.30,
    )
    assert len(match["structure"]) == 13
    statistics = sample_statistics(match["structure"])
    assert set(statistics) == {f"{gene}_{stat}" for gene in GENES for stat in ("mean", "spread")}


def test_budget_matching_draw_is_deterministic_and_validates_its_input():
    first = budget_matching_draw(
        target_total=1_121_409_984, size=13, seeds=range(8), target_params=100_000_000,
        tolerance=0.30,
    )
    second = budget_matching_draw(
        target_total=1_121_409_984, size=13, seeds=range(8), target_params=100_000_000,
        tolerance=0.30,
    )
    assert first == second
    with pytest.raises(ValueError):
        budget_matching_draw(
            target_total=1, size=0, seeds=range(2), target_params=100_000_000
        )
    with pytest.raises(ValueError):
        budget_matching_draw(
            target_total=1, size=2, seeds=[], target_params=100_000_000
        )


def test_population_structure_is_non_empty_and_within_the_band():
    population = population_structure(target_params=78_000_000, tolerance=0.05)
    assert population
    depths = {depth for depth, _ in population}
    hidden = {size for _, size in population}
    # The band is wide enough to contain more than one shape, or nothing about
    # typicality would be measurable inside it.
    assert len(depths) > 1 and len(hidden) > 1
