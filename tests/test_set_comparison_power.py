"""Tests for the section 11 power study, whose whole output is a spend decision.

The study answers "how large a curation advantage can a set-versus-set comparison
detect at the size this phase can afford", and the answer is what licenses or
forbids spending GPU hours. Its cheapest row is the one that has to be right: the
null row. If the comparison rejects on data with no curation effect at all, then
every power figure above it is inflated by that amount, and the phase buys a
comparison less sensitive than the table promises.

So the generator is checked for being centred on the real population it claims to
draw from, and the null row is checked to sit at the nominal rate on the assembled
test rather than at whatever the test happens to produce.
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from act.search.architecture_report import two_set_blocked_test  # noqa: E402
from set_comparison_power import (  # noqa: E402
    candidate_population,
    power_at,
    synthetic_results,
)

ARM = "arch-zoo-100m-v1"
STEPS = 400
SET_SIZE = 13
MEMBERS = (
    tuple(f"random_{index}" for index in range(SET_SIZE)),
    tuple(f"manual_{index}" for index in range(SET_SIZE)),
)
SETS = (("random", MEMBERS[0]), ("manual", MEMBERS[1]))


def measured() -> tuple[list[float], float]:
    directory = Path("artifacts") / ARM
    if not sorted(directory.glob("*_seed*_results.jsonl")):
        pytest.skip(f"{ARM} ledgers not present")
    return candidate_population(Path("artifacts"), ARM, STEPS)


def one_replicate(population, noise, generator, advantage: float, model: str) -> list:
    return synthetic_results(
        population,
        set_size=SET_SIZE,
        blocks=2,
        noise=noise,
        advantage=advantage,
        model=model,
        generator=generator,
    )


def test_population_and_noise_come_from_the_real_ledgers_at_the_named_length():
    """The table's inputs are measurements, not assumptions - so pin the measurements."""
    population, noise = measured()
    assert len(population) == SET_SIZE
    # The curated set's size is the design's size, and the spread is the real one:
    # 0.7411 nats across the 13 candidates at 400 steps.
    assert max(population) - min(population) == pytest.approx(0.7411, abs=0.01)
    assert 0.02 < noise < 0.06, noise


def test_a_represents_nothing_set_is_centred_on_the_population():
    """No planted effect must mean no difference, or the whole table reads high.

    Each set is an independent draw of the same size from the same measured
    population, and the member's row on each seed is that member's quality plus its
    own noise, which is the structure the real arm has.
    """
    population, noise = measured()
    generator = random.Random(3)
    differences = [
        two_set_blocked_test(
            one_replicate(population, noise, generator, 0.0, "shift-all"),
            STEPS,
            SETS,
            samples=1,
        )["difference_nats"]
        for _ in range(200)
    ]
    error = statistics.stdev(differences) / len(differences) ** 0.5
    assert abs(statistics.mean(differences)) < 4 * error, statistics.mean(differences)


def test_both_models_improve_the_curated_set_the_way_they_claim():
    """``shift-all`` improves every member; ``shift-best`` only the winner.

    The distinction is the study's reason to exist - section 9a predicted curation's
    advantage sits in coverage rather than location - so the two models must not be
    the same experiment in disguise. On the set *mean* they differ by 13x, by
    construction: 13 members moving 0.10 against one moving 0.10.
    """
    population, noise = measured()

    # One stream for every call, so both sets' draws are identical and the only
    # thing that differs between calls is the planted effect.
    def difference(advantage: float, model: str, statistic: str = "mean") -> float:
        return two_set_blocked_test(
            one_replicate(population, noise, random.Random(8), advantage, model),
            STEPS,
            SETS,
            statistic=statistic,
            samples=1,
        )["difference_nats"]

    baseline = difference(0.0, "shift-all")
    # A negative difference favours the first (random) set, so improving the curated
    # set reads as the difference rising.
    assert difference(0.10, "shift-all") - baseline == pytest.approx(0.10, abs=1e-6)
    assert difference(0.10, "shift-best") - baseline == pytest.approx(
        0.10 / SET_SIZE, abs=1e-6
    )

    # And on the statistic the coverage hypothesis is about, the winner improves by
    # the whole advantage rather than by the gap to the runner-up: improving the best
    # member cannot be truncated, because it stays the best member. The 0.02 slack is
    # that gap - the two best of 13 sit 0.039 nats apart on average, and the realized
    # shift is 0.0925 here because one seed's floor was set by the runner-up before
    # the improvement. Applied as a penalty on the winner the same 0.10 would reach
    # the statistic as roughly that gap, ~0.009 at the median, and the whole
    # ``shift-best`` row of the table would read as dead.
    best_baseline = difference(0.0, "shift-best", "best")
    assert difference(0.10, "shift-best", "best") - best_baseline == pytest.approx(
        0.10, abs=0.02
    )


def test_a_better_winner_alone_is_invisible_to_the_sets_floor():
    """The coverage hypothesis cannot be bought with GPU hours, and this is why.

    A permutation null shuffling the two sets already contains the improved winner:
    half of its assignments put that member on either side. Whichever side holds it
    takes the low floor and the other side's floor sits one winner-gap above it, so
    the *magnitude* of the floor difference barely changes with the labelling - the
    observed one is not in the tail of that null, it is the middle of it. A 0.20-nat
    better winner therefore reads as nothing, which is a statement about the design
    rather than about the effect, and it is the reason the table's measurable row is
    ``shift-all``.
    """
    population, noise = measured()
    generator = random.Random(0)
    probabilities = [
        two_set_blocked_test(
            one_replicate(population, noise, generator, 0.20, "shift-best"),
            STEPS,
            SETS,
            statistic="best",
            samples=1000,
        )["p"]
        for _ in range(20)
    ]
    # The null is as extreme as the data in most draws, so p sits against its upper
    # bound rather than near zero.
    assert statistics.median(probabilities) > 0.5, probabilities
    assert sum(p < 0.05 for p in probabilities) <= 2, probabilities


def test_the_null_row_of_the_table_stays_at_the_nominal_rate():
    """Regression: an anti-conservative comparison makes the study read optimistic.

    Up to a 24-point error on this row is available from a null that splits the two
    sets per seed instead of per member: 0.175 measured on this population against
    the 0.043-0.058 a properly built null gives, which is exactly the kind of
    inflation that turns "not affordable" into "affordable".
    """
    population, noise = measured()
    rate = power_at(
        population,
        advantage=0.0,
        model="shift-all",
        statistic="mean",
        set_size=SET_SIZE,
        blocks=2,
        noise=noise,
        replicates=200,
        samples=500,
    )
    assert rate < 0.12, rate


def test_a_real_advantage_is_detected_so_the_table_is_not_all_floor():
    """The other side of calibration: a test that never rejects is equally useless."""
    population, noise = measured()
    rate = power_at(
        population,
        advantage=0.20,
        model="shift-all",
        statistic="mean",
        set_size=SET_SIZE,
        blocks=2,
        noise=noise,
        replicates=100,
        samples=500,
    )
    assert rate > 0.4, rate
