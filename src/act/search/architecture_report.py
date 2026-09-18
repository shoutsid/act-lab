"""Reporting and decision logic for architecture-search experiments.

Kept separate from the driver so a sweep can be reported on at any point, not
only when all three stages finish. A four-hour run that dies at 90% should still
produce a ranked table and a decision, and the analysis should be executable
without re-training anything.

The central job is separating an *architecture* effect from initialization
noise, and that is a statistical question rather than a threshold. Three
quantities are computed, because they disagree and the disagreement matters:

``variance_decomposition``
    One-way ANOVA. The conventional unpaired test. It treats every run as
    independent, throwing away the fact that each architecture ran on the *same*
    seeds, so between-seed variance lands in the error term.
``blocked_permutation_test``
    The seed-matched test the brief's section 20 exists to enable. Each seed is a
    block, so the shared seed effect cancels and the null distribution over rank
    orderings is small enough to enumerate *exactly*.
``effect_in_compute_units``
    An effect size in the currency the brief actually budgets in. A nat
    difference is meaningless on its own; "worth 63 extra training steps out of
    1200" is a decision.

A verdict of "not established" from these tests is a statement about evidence,
not about the world, and the module says so explicitly at n = 3 seeds.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from itertools import combinations, permutations, product
import random
from typing import Sequence
from pathlib import Path

import scipy.stats

from act.search.architecture_eval import EvalResult

#: Spread below this is not worth funding a search program over, whatever its
#: p-value. Deliberately lax: its job is to stop a statistically real but
#: practically irrelevant difference from launching a large search, and to be
#: overridden by a confident large effect rather than to gate one.
MATERIALITY_FLOOR_NATS = 0.01


def rank(results: list[EvalResult], steps: int) -> list[tuple[str, float, int]]:
    """Mean held-out validation loss per architecture for one training length."""
    grouped = _losses_by_architecture(results, steps)
    ranked = [
        (name, sum(by_seed.values()) / len(by_seed), len(by_seed))
        for name, by_seed in grouped.items()
    ]
    ranked.sort(key=lambda item: item[1])
    return ranked


def _losses_by_architecture(
    results: list[EvalResult], steps: int
) -> dict[str, dict[int, float]]:
    """Validation loss per architecture keyed by seed, at one training length."""
    grouped: dict[str, dict[int, float]] = {}
    for result in results:
        if result.steps == steps:
            grouped.setdefault(result.name, {})[result.seed] = result.validation_loss
    return grouped


def _complete_design(
    results: list[EvalResult], steps: int
) -> tuple[list[str], list[int], dict[str, dict[int, float]]]:
    """Architectures and seeds forming a rectangular design at one length.

    A blocked test needs every architecture measured on every seed. Rather than
    silently testing an unbalanced subset, this intersects to the largest
    rectangular block and reports what it had to drop.
    """
    grouped = _losses_by_architecture(results, steps)
    eligible = {name: by_seed for name, by_seed in grouped.items() if len(by_seed) >= 2}
    if len(eligible) < 2:
        return [], [], grouped
    common = set.intersection(*(set(by_seed) for by_seed in eligible.values()))
    seeds = sorted(common)
    if not seeds:
        return [], [], grouped
    names = sorted(eligible)
    return names, seeds, grouped


def _design_drops(
    results: Sequence[EvalResult],
    steps: int,
    names: Sequence[str],
    seeds: Sequence[int],
    grouped: dict[str, dict[int, float]],
) -> dict:
    """What the rectangular intersection removed, and which failure that is.

    A blocked test needs a complete rectangle, so a partial sweep shrinks the
    design silently unless the shrink is reported. The block count printed beside
    the p-value is the *remaining* seeds, and a reader who does not know a seed was
    dropped reads a 2-block result as the whole design. Two different failures hide
    behind the same drop, so they are separated here the same way the contrasts
    separate them: a seed carrying fewer rows than the fullest is an unfinished
    sweep, and a seed that ran every candidate but cannot be intersected is an
    unbalanced design - which is a zoo defect, not a missing row.
    """
    kept = set(seeds)
    all_seeds = sorted({result.seed for result in results if result.steps == steps})
    dropped = [seed for seed in all_seeds if seed not in kept]
    counts: dict[int, int] = {}
    for result in results:
        if result.steps == steps:
            counts[result.seed] = counts.get(result.seed, 0) + 1
    fullest = max(counts.values()) if counts else 0
    kept_names = set(names)
    return {
        "seeds": dropped,
        "architectures": [
            name for name in sorted(grouped) if name not in kept_names
        ],
        "unfinished_seeds": [
            seed for seed in dropped if counts.get(seed, 0) < fullest
        ],
        "unbalanced_seeds": [
            seed for seed in dropped if counts.get(seed, 0) >= fullest
        ],
    }


def _short_seeds(results: Sequence[EvalResult], seeds: Sequence[int]) -> list[int]:
    """Seeds carrying fewer runs than the fullest seed: an unfinished sweep.

    This exists to keep two different failures apart. A design analysed while its
    last seed is still training is incomplete *there* and only there, so the arm's
    question is still answerable from the seeds that finished. A design whose
    cells are unbalanced on a seed that ran everything is *wrong*, and quietly
    testing a subset of it would hide that. Both look identical to a balance
    check, so the row count per seed is what separates them.
    """
    counts: dict[int, int] = {}
    for entry in results:
        counts[entry.seed] = counts.get(entry.seed, 0) + 1
    if not counts:
        return []
    fullest = max(counts.values())
    return [seed for seed in seeds if counts.get(seed, 0) < fullest]


def variance_decomposition(results: list[EvalResult], steps: int) -> dict:
    """One-way ANOVA: is between-architecture variance larger than within?

    The unpaired contrast to ``blocked_permutation_test``. Both are reported
    because they routinely disagree, and the disagreement is the evidence that
    seed matching was worth doing.
    """
    grouped = _losses_by_architecture(results, steps)
    samples = [sorted(by_seed.values()) for by_seed in grouped.values() if len(by_seed) >= 2]
    if len(samples) < 2:
        return {"available": False, "reason": "fewer than two architectures on 2+ seeds"}

    total_n = sum(len(s) for s in samples)
    grand = sum(sum(s) for s in samples) / total_n
    ss_between = sum(len(s) * (sum(s) / len(s) - grand) ** 2 for s in samples)
    ss_within = sum(
        sum((value - sum(s) / len(s)) ** 2 for value in s) for s in samples
    )
    df_between = len(samples) - 1
    df_within = total_n - len(samples)
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within if df_within > 0 else 0.0
    if ms_within > 0:
        f_stat = ms_between / ms_within
        p_value = float(scipy.stats.f.sf(f_stat, df_between, df_within))
    else:
        f_stat = float("inf")
        p_value = 0.0 if ms_between > 0 else 1.0

    return {
        "available": True,
        "test": "one-way ANOVA (unpaired)",
        "architectures": len(samples),
        "seeds_per_architecture": sorted({len(s) for s in samples}),
        "ss_between": ss_between,
        "ss_within": ss_within,
        "df_between": df_between,
        "df_within": df_within,
        "f": f_stat,
        "p": p_value,
    }


def _blocked_permutation_p(
    matrix: list[list[float]],
    statistic,
    *,
    cap: int,
    samples: int = 200_000,
    seed: int = 0,
) -> tuple[float | None, int, str]:
    """p-value over within-block assignments, exactly or by Monte-Carlo.

    Exhaustive enumeration is ``(k!)^blocks`` assignments, which is trivial for a
    3-candidate screen (216) and infeasible for a matched design of 8 candidates
    on 3 seeds (8!^3 ~ 6.6e13). When the enumeration exceeds ``cap`` this samples
    uniformly from the same null rather than falling back to the chi-square
    approximation - which is what a designed experiment would otherwise be
    silently downgraded to, and which is known to be poor at 3 seeds.

    The statistic depends only on the per-architecture column sums, so both paths
    accumulate those sums rather than materialising assignments.

    Returns ``(p, assignments, method)`` where method is ``"exact"``,
    ``"monte-carlo"`` or ``"unavailable"``.
    """
    blocks = len(matrix)
    k = len(matrix[0])
    per_block = list(permutations(range(k)))
    assignments = len(per_block) ** blocks
    observed = statistic(tuple(sum(row[j] for row in matrix) for j in range(k)))

    if assignments <= cap:
        sums: list[tuple[float, ...]] = [tuple([0.0] * k)]
        for row in matrix:
            sums = [
                tuple(acc + row[index] for acc, index in zip(total, perm))
                for total in sums
                for perm in per_block
            ]
        at_least = sum(1 for total in sums if statistic(total) >= observed - 1e-12)
        return at_least / len(sums), assignments, "exact"

    if samples <= 0:
        return None, assignments, "unavailable"

    generator = random.Random(seed)
    order = list(range(k))
    at_least = 0
    for _ in range(samples):
        totals = [0.0] * k
        for row in matrix:
            generator.shuffle(order)
            for column in range(k):
                totals[column] += row[order[column]]
        if statistic(tuple(totals)) >= observed - 1e-12:
            at_least += 1
    # (r + 1) / (n + 1) rather than r / n: a Monte-Carlo p of exactly 0 would
    # claim more certainty than the sample can support.
    return (at_least + 1) / (samples + 1), assignments, "monte-carlo"


def blocked_permutation_test(
    results: list[EvalResult],
    steps: int,
    *,
    max_assignments: int = 1_000_000,
    samples: int = 200_000,
) -> dict:
    """Exact seed-matched test: does the architecture effect survive the seed noise?

    Every architecture runs on the same seeds, so a seed is a block. Under the
    null of no architecture effect, which value goes to which architecture inside
    a block is exchangeable, so the null distribution is uniform over all
    ``(k!)^blocks`` assignments and can be enumerated *exactly* rather than
    approximated - which matters, because the usual chi-square approximation to
    Friedman is poor at n = 3.

    ``values`` is the primary test: it permutes the actual losses, so it uses the
    magnitude of each difference, not just the ordering. ``ranks`` runs the same
    enumeration on within-block ranks as an outlier-insensitive robustness check.
    The two disagreeing is informative - rank consistency alone can be weak
    evidence when one architecture wins by a hair.

    Blocking removes between-seed variance from the error term, so this is more
    powerful than the unpaired test. The flip side is a floor: with 3
    architectures on 3 seeds the *smallest* attainable p-value is
    ``1/36 = 0.028``, and it requires the two non-tied architectures to order
    consistently in every seed. A non-significant result here is therefore "not
    yet established", not "disproved".
    """
    names, seeds, grouped = _complete_design(results, steps)
    if len(names) < 2 or len(seeds) < 2:
        return {
            "available": False,
            "reason": "needs 2+ architectures measured on 2+ shared seeds",
        }
    drops = _design_drops(results, steps, names, seeds, grouped)

    # Order architectures best-to-worst by mean loss so column 0 is the winner
    # and column -1 the loser, which is what the paired reference test contrasts.
    names = sorted(
        names, key=lambda name: sum(grouped[name][seed] for seed in seeds) / len(seeds)
    )
    k = len(names)
    blocks = len(seeds)
    raw = [[grouped[name][seed] for name in names] for seed in seeds]
    # Block-centering makes the statistic invariant to a per-seed offset, which is
    # exactly the seed effect the blocking is meant to remove.
    centered = [
        [value - sum(row) / len(row) for value in row] for row in raw
    ]
    value_stat = lambda sums: sum(s * s for s in sums) / blocks  # noqa: E731
    rank_stat = lambda sums: (  # noqa: E731
        12.0 / (blocks * k * (k + 1)) * sum(s * s for s in sums) - 3.0 * blocks * (k + 1)
    )

    value_p, value_assignments, value_method = _blocked_permutation_p(
        centered, value_stat, cap=max_assignments, samples=samples
    )
    ranked = [list(scipy.stats.rankdata(row)) for row in raw]
    rank_p, rank_assignments, rank_method = _blocked_permutation_p(
        ranked, rank_stat, cap=max_assignments, samples=samples
    )

    # Reference only: the paired test most readers will reach for, contrasting the
    # best architecture against the worst seed-by-seed. Not primary because it
    # assumes normality of the per-seed differences, which 3 seeds cannot support.
    differences = [row[0] - row[-1] for row in raw]
    try:
        _, paired_t_p = scipy.stats.ttest_1samp(differences, 0.0)
    except (ValueError, ZeroDivisionError):
        paired_t_p = None

    method = value_method if value_method == rank_method else f"{value_method}/{rank_method}"
    exact = value_method == "exact" and rank_method == "exact"
    if value_p is None:
        try:
            _, value_p = scipy.stats.friedmanchisquare(
                *[[grouped[name][seed] for name in names] for seed in seeds]
            )
        except ValueError:
            value_p = None

    return {
        "available": True,
        "test": "blocked permutation (" + method + ")",
        "method": method,
        "architectures": k,
        "blocks": blocks,
        "names": names,
        "seeds": seeds,
        # The seeds the design started with and could not be tested over, kept apart
        # by cause so the report can tell an unfinished sweep from a broken zoo.
        "seeds_dropped": drops["seeds"],
        "architectures_dropped": drops["architectures"],
        "unfinished_seeds": drops["unfinished_seeds"],
        "unbalanced_seeds": drops["unbalanced_seeds"],
        "p": value_p,
        "p_ranks": rank_p,
        "p_paired_t_reference": None if paired_t_p is None else float(paired_t_p),
        "statistic": value_stat(tuple(sum(row[j] for row in centered) for j in range(k))),
        "exact": exact,
        "assignments": value_assignments,
        "rank_assignments": rank_assignments,
        # Two different floors, reported separately because they bind in
        # different places: the *design* floor is what the data can distinguish,
        # and the method floor is the smallest p the procedure can output.
        "min_achievable_p": 1.0 / (math.factorial(k) ** max(1, blocks - 1)),
        "p_resolution": (1.0 / value_assignments) if exact else (1.0 / (samples + 1)),
        # Which architecture came first in each seed, so a reader can see whether
        # the result is 3/3 or 2/3 agreeing rather than inferring it from p.
        "orderings": [
            [names[index] for index in scipy.stats.rankdata(row).argsort()] for row in raw
        ],
    }


def two_set_blocked_test(
    results: list[EvalResult],
    steps: int,
    sets: tuple[tuple[str, Sequence[str]], tuple[str, Sequence[str]]],
    *,
    statistic: str = "mean",
    samples: int = 200_000,
    max_assignments: int = 1_000_000,
    seed: int = 0,
) -> dict:
    """Seed-blocked comparison of two *sets* of architectures under equal compute.

    Section 11 asks a different question from every other test in this module: not
    "does this architecture beat that one" but "does a curated *set* of 13 beat a
    random draw of 13 trained with the same budget".    The unit of exchange under the null is therefore the set *label*, and a label
    belongs to an architecture rather than to one run: with no curation effect the
    two sets' members are interchangeable, so the null draws **one** split of the
    ``2n`` members into two groups of ``n`` and reads every seed through that same
    split. A seed blocks the comparison, it does not add a level of exchange - a
    fresh split per seed would take each member's shared quality out of the null's
    differences while the observed difference still carries it, and the test would
    then reject on a real member-level spread alone. That is also what separates
    this null from ``blocked_permutation_test``'s: an architecture null sets the
    member effect to zero, and this one does not, so the two cannot be permuted the
    same way. The assignment count is ``C(2n, n)``, not ``C(2n, n)**seeds``.

    Equal compute is a precondition rather than an argument. Unequal set sizes
    would compare two budgets rather than two selections, and a shared member would
    make the two sets overlap, so both are refused with the reason rather than
    computed over.

    ``statistic`` picks the quantity the sets are compared on, and the choice is
    substantive enough to name rather than average away:

    * ``mean`` - is the curated set's *average* candidate better? This is the
      question "does curation carry signal", and it averages the candidate-quality
      spread down, so it is the better-powered of the three.
    * ``best`` - does the curated set produce a better *winner*? This is what a
      search budget actually buys, and it is dominated by the top of the quality
      distribution, so it is noisier: with 13 candidates a set's best is one draw
      from its upper tail.
    * ``worst`` - is its floor higher (fewer duds)? The mirror of ``best``, and the
      quantity a search that samples widely cares about when a bad candidate costs
      the same as a good one.

    The difference is signed as *first set minus second set*, so a negative value
    means the first is better - the same convention every other contrast here uses.
    """
    functions = {"mean": lambda values: sum(values) / len(values), "best": min, "worst": max}
    if statistic not in functions:
        raise ValueError(f"unknown statistic {statistic!r}; choose from {sorted(functions)}")
    measure = functions[statistic]

    labels = (str(sets[0][0]), str(sets[1][0]))
    members = (list(sets[0][1]), list(sets[1][1]))
    if not members[0] or not members[1]:
        return {"available": False, "reason": "both sets must name at least one architecture"}
    if len(members[0]) != len(members[1]):
        return {
            "available": False,
            "reason": (
                f"sets of unequal size compare two budgets, not two selections: "
                f"{len(members[0])} against {len(members[1])}"
            ),
        }
    overlap = sorted(set(members[0]) & set(members[1]))
    if overlap:
        return {
            "available": False,
            "reason": "the sets share architectures: " + ", ".join(overlap),
        }

    every = [*members[0], *members[1]]
    per_name: dict[str, dict[int, float]] = {}
    for entry in results:
        if entry.steps == steps and entry.name in set(every):
            per_name.setdefault(entry.name, {})[entry.seed] = entry.validation_loss
    absent = [name for name in every if not per_name.get(name)]
    if absent:
        return {
            "available": False,
            "reason": f"no {steps}-step rows for: " + ", ".join(sorted(absent)),
        }
    shared: set[int] | None = None
    for name in every:
        available = set(per_name[name])
        shared = available if shared is None else (shared & available)
    seeds = sorted(shared or [])
    if len(seeds) < 2:
        return {
            "available": False,
            "reason": (
                f"a seed-blocked test needs 2+ seeds carrying both sets; {len(seeds)} do"
            ),
        }

    n = len(members[0])
    k = 2 * n
    # The first set occupies the first n columns of every block, which is the
    # observed labelling the null is measured against.
    blocks = [
        [per_name[name][block] for name in members[0]]
        + [per_name[name][block] for name in members[1]]
        for block in seeds
    ]

    def per_block_differences(selection: Sequence[int]) -> list[float]:
        """One *member-level* split, applied to every block.

        A label belongs to an architecture, not to one seed's run: the same 13
        candidates sit on the same side in every seed, so a split that varies by
        seed is not a labelling this design can produce. That is not pedantry. The
        quantity the set null leaves in place is the *member* effect - one
        candidate really is better than another, and that quality is shared by its
        rows in every seed - so an assignment that breaks a member's rows apart
        across sides cancels the member variance out of the null's differences
        while the observed difference still carries it. The null would then be
        narrower than the quantity it tests and a real member-level spread alone
        would show up as a small p. Drawing one split and applying it to every
        block keeps the two on the same scale.
        """
        chosen = set(selection)
        differences = []
        for values in blocks:
            first = [value for index, value in enumerate(values) if index in chosen]
            second = [value for index, value in enumerate(values) if index not in chosen]
            differences.append(measure(first) - measure(second))
        return differences

    observed_selection = tuple(range(n))
    observed_differences = per_block_differences(observed_selection)
    observed = sum(observed_differences) / len(observed_differences)

    # The assignment count is computed rather than materialised: at 13+13
    # candidates a single split is already C(26,13) = 10.4M labellings, and
    # building that list before checking the cap is what makes this test hang
    # *because* it is about to sample instead of enumerate. The list is only ever
    # built inside the branch that enumerates, where the cap bounds its size.
    # The count is over member-level splits: seeds multiply the number of *values*
    # the statistic averages, not the number of labellings the design can take.
    assignments = math.comb(k, n)
    if assignments <= max_assignments:
        at_least = 0
        for selection in combinations(range(k), n):
            drawn = per_block_differences(selection)
            if abs(sum(drawn) / len(drawn)) >= abs(observed) - 1e-12:
                at_least += 1
        p_value, method = at_least / assignments, "exact"
    else:
        generator = random.Random(seed)
        at_least = 0
        for _ in range(samples):
            drawn = per_block_differences(tuple(generator.sample(range(k), n)))
            if abs(sum(drawn) / len(drawn)) >= abs(observed) - 1e-12:
                at_least += 1
        # (r + 1) / (n + 1): a Monte-Carlo p of exactly 0 would claim more
        # certainty than the draw count can support, and this test's design floor
        # is far below what 200,000 draws can resolve.
        p_value, method = (at_least + 1) / (samples + 1), "monte-carlo"

    def describe(names: Sequence[str]) -> dict:
        values = [per_name[name][block] for name in names for block in seeds]
        return {
            "runs": len(values),
            "mean": sum(values) / len(values),
            "best": min(values),
            "worst": max(values),
        }

    return {
        "available": True,
        "steps": steps,
        "statistic": statistic,
        "labels": list(labels),
        "per_block": [
            {
                "seed": block,
                "difference": difference,
                labels[0]: measure(blocks[index][:n]),
                labels[1]: measure(blocks[index][n:]),
            }
            for index, (block, difference) in enumerate(zip(seeds, observed_differences))
        ],
        "difference_nats": observed,
        "per_set": {labels[0]: describe(members[0]), labels[1]: describe(members[1])},
        "p": p_value,
        "method": method,
        "assignments": assignments,
        # The design's own floor: two label assignments attain the extreme, since
        # swapping the two sets gives the mirror of the same statistic.
        "min_achievable_p": 2 / assignments,
        "p_resolution": (1 / (samples + 1)) if method == "monte-carlo" else None,
        "sign_consistency": all(value < 0 for value in observed_differences)
        or all(value > 0 for value in observed_differences),
        "seeds": seeds,
        "note": (
            "Difference is the first set minus the second, averaged over seeds, so a "
            "negative value favours the first. Effect sizes are horizon-dependent "
            "(section 3): quote this with its training length."
        ),
    }


def paired_contrast_test(pairs: Sequence[tuple[float, float]]) -> dict:
    """Exact sign-permutation test on paired differences.

    For the head-paired design, the two members of a pair are the *same* base
    configuration at different attention-head counts, so their costs are identical
    and the only thing separating them is the gene under test. That makes each pair
    its own control, and the right unit of analysis is the within-pair difference,
    not the difference of arm means.

    Averaging to arm means instead would discard the pairing and, with three seeds,
    leave a floor of 2/2^3 = 0.25 - the same floor that left the crossed design's
    interaction unresolved. Pairing over four pairs across three seeds gives twelve
    differences and an exact floor of 2/2^12 = 4.9e-04, which is the whole reason
    the design holds a base configuration fixed.

    The null is that each difference is as likely negative as positive under no
    effect, so the p-value enumerates every sign vector when there are few enough,
    and samples otherwise. A paired t is reported alongside as a reference.
    """
    differences = [treatment - baseline for baseline, treatment in pairs]
    if not differences:
        return {"available": False, "pairs": 0}
    mean = sum(differences) / len(differences)
    resolutions = 2 ** len(differences)

    if resolutions <= 1_000_000:
        at_least = 0
        for signs in product((1, -1), repeat=len(differences)):
            flipped = sum(s * d for s, d in zip(signs, differences)) / len(differences)
            if abs(flipped) >= abs(mean) - 1e-12:
                at_least += 1
        p_value, method = at_least / resolutions, "exact"
    else:
        generator = random.Random(0)
        at_least = 0
        draws = 200_000
        for _ in range(draws):
            flipped = sum(
                (1 if generator.random() < 0.5 else -1) * d for d in differences
            ) / len(differences)
            if abs(flipped) >= abs(mean) - 1e-12:
                at_least += 1
        p_value, method = (at_least + 1) / (draws + 1), "monte-carlo"

    variance = (
        sum((d - mean) ** 2 for d in differences) / (len(differences) - 1)
        if len(differences) > 1
        else 0.0
    )
    paired_t_p = None
    if variance > 0:
        t_value = mean / math.sqrt(variance / len(differences))
        paired_t_p = float(
            2 * scipy.stats.t.sf(abs(t_value), df=len(differences) - 1)
        )

    negatives = sum(1 for d in differences if d < 0)
    return {
        "available": True,
        "pairs": len(differences),
        "mean_difference": mean,
        "p": p_value,
        "method": method,
        "assignments": resolutions,
        "min_achievable_p": 2 / resolutions,
        "sign_consistent": negatives in (0, len(differences)),
        "negative": negatives,
        "positive": len(differences) - negatives,
        "differences": differences,
        "p_paired_t": paired_t_p,
    }


def _paired_contrasts_at(results: list[EvalResult], steps: int) -> dict[str, dict]:
    """Paired-gene contrasts for every structural gene that forms complete pairs.

    A design is 'paired' in a gene when holding the other structural genes fixed
    leaves exactly two levels of it, on every seed. That is detected from the data
    rather than declared by the caller, so a report cannot claim a paired contrast
    the ledger does not support - and a design that only looks paired (because a
    band happened to match a pair) yields nothing rather than a false positive.

    A group can be missing a level for two very different reasons: the sweep has
    not finished that seed yet, or the pairing never had that member. Only the
    first is a reason to report a smaller design, so a gene whose incomplete
    groups all sit on an unfinished seed is reported with ``dropped_groups`` -
    the contrast is over the complete pairs, and the drop is stated. Anything else
    still refuses, because a subset of a malformed pairing is not a paired design.
    """
    at_length = [entry for entry in results if entry.steps == steps]
    short = set(_short_seeds(at_length, sorted({entry.seed for entry in at_length})))
    found: dict[str, dict] = {}
    for gene in STRUCTURAL_GENES:
        contrast = paired_design_contrast(at_length, gene)
        if not contrast.get("available"):
            continue
        incomplete = contrast.get("incomplete_groups") or []
        dropped = [row for row in incomplete if row["seed"] in short]
        if len(dropped) != len(incomplete):
            continue
        # A paired contrast is only meaningful if every pair carried two levels and
        # nothing else varies inside a pair, which is what a complete grouping and
        # a single pair size demonstrate.
        sizes = {entry["runs"] for entry in contrast["per_pair"]}
        if len(sizes) != 1:
            continue
        if dropped:
            contrast["dropped_groups"] = dropped
            contrast["seeds_dropped"] = sorted({row["seed"] for row in dropped})
        # A seed can also be short without any *half* pair: an unfinished sweep can
        # simply not have reached a pair's two members yet. Nothing is incomplete
        # then, so the group checks above say nothing, and the unit count quietly
        # drops while the floor printed beside the p rises. The pair count per seed
        # is what shows it, and it is reported whether or not groups were dropped.
        counts: dict[int, int] = {}
        for row in contrast["per_pair"]:
            seed = row["base"]["seed"]
            counts[seed] = counts.get(seed, 0) + 1
        if counts:
            fullest = max(counts.values())
            partial = [
                {"seed": seed, "pairs": number, "of": fullest}
                for seed, number in sorted(counts.items())
                if number < fullest
            ]
            if partial:
                contrast["seeds_partial"] = partial
        found[gene] = contrast
    return found


def _paired_level_splits_at(results: list[EvalResult], steps: int) -> dict[str, dict]:
    """Level splits for every paired gene against every two-level partner.

    Only two-level partners are split on, and that is a property of the question
    rather than a shortcut: a two-level partner is a crossed factor, so the split
    *is* the interaction question asked in paired units - one number per level, each
    with the pair as its own control. A three- or four-level gene has no single
    interaction to read, so splitting on it would add descriptive slices to the
    decision without answering anything the design crossed.
    """
    at_length = [entry for entry in results if entry.steps == steps]
    paired_genes = sorted(_paired_contrasts_at(at_length, steps))
    if not paired_genes:
        return {}
    two_level = [
        name
        for name in STRUCTURAL_GENES
        if len({getattr(entry, name, 0) for entry in at_length}) == 2
    ]
    found: dict[str, dict] = {}
    for gene in paired_genes:
        splits = {
            other: paired_level_split(at_length, gene, other)
            for other in two_level
            if other != gene
        }
        available = {name: split for name, split in splits.items() if split.get("available")}
        if available:
            found[gene] = available
    return found


def paired_design_contrast(
    results: list[EvalResult],
    gene: str,
    *,
    subgroup: tuple[str, float] | None = None,
) -> dict:
    """Contrast one gene across candidates that differ in nothing else.

    A paired design holds every other gene fixed and varies ``gene``, so each pair
    is its own control and the within-pair difference carries none of the between-
    candidate variation that a band-matched contrast has to average over. Two
    things follow: the contrast is exactly matched, and the unit of analysis is the
    pair rather than the arm mean.

    This exists because collapsing a paired design to arm means throws the pairing
    away. With 4 pairs on 3 seeds the paired test has twelve units and an exact
    floor of 2/2^12, while arm means would have three and floor at 2/2^3 - the same
    floor that left the crossed design's interaction unresolved. Which test is used
    is therefore not a detail; it decides whether the arm can answer its question.

    Pairs are formed by grouping on every structural gene *except* ``gene`` within
    a seed, so a group of the wrong size is a design error rather than something
    silently averaged away.

    ``subgroup`` restricts the contrast to rows holding ``(gene, value)`` fixed
    first, which is how a level split is computed by :func:`paired_level_split`: a
    pooled paired contrast is an average over the levels of every gene it holds
    fixed, and two levels with opposite effects average to nothing. The restriction
    is carried in the returned bundle, so a level's number cannot be read as the
    pooled one.
    """
    if gene not in STRUCTURAL_GENES:
        raise ValueError(f"unknown gene {gene!r}; choose from {STRUCTURAL_GENES}")
    at_length = [entry for entry in results if entry.validation_loss is not None]
    if subgroup is not None:
        other_gene, value = subgroup
        at_length = [
            entry for entry in at_length if getattr(entry, other_gene, None) == value
        ]
    if not at_length:
        return {"available": False, "reason": "no results"}

    keys = [name for name in STRUCTURAL_GENES if name != gene]
    groups: dict[tuple, dict[int, list[float]]] = {}
    for entry in at_length:
        base = tuple(getattr(entry, name, 0) for name in keys) + (entry.seed,)
        groups.setdefault(base, {}).setdefault(getattr(entry, gene, 0), []).append(
            entry.validation_loss
        )

    levels = sorted({getattr(entry, gene, 0) for entry in at_length})
    if len(levels) < 2:
        return {"available": False, "reason": f"{gene} does not vary"}
    low, high = levels[0], levels[-1]

    def average(values: list[float]) -> float:
        return sum(values) / len(values)

    incomplete: list[dict] = []
    pairs: list[tuple[float, float]] = []
    per_pair: list[dict] = []
    for base, by_level in sorted(groups.items(), key=lambda item: str(item[0])):
        if low not in by_level or high not in by_level:
            # Reported with the seed, because that is what tells an unfinished
            # sweep apart from a malformed pairing: see `_paired_contrasts_at`.
            incomplete.append(
                {
                    "base": dict(zip([*keys, "seed"], base)),
                    "seed": base[-1],
                    "levels_present": sorted(by_level),
                }
            )
            continue
        baseline = average(by_level[low])
        treatment = average(by_level[high])
        pairs.append((baseline, treatment))
        per_pair.append(
            {
                "base": dict(zip([*keys, "seed"], base)),
                "levels": [low, high],
                "baseline": baseline,
                "treatment": treatment,
                "difference": treatment - baseline,
                "runs": len(by_level[low]) + len(by_level[high]),
            }
        )

    contrast = paired_contrast_test(pairs)
    if not contrast.get("available"):
        return {
            "available": False,
            "reason": "no group carried both levels",
            "gene": gene,
            "levels": [low, high],
            "incomplete_groups": incomplete,
        }
    contrast.update(
        {
            "gene": gene,
            "levels": [low, high],
            "incomplete_groups": incomplete,
            "per_pair": per_pair,
            "pair_keys": keys,
            # The seeds the pairs came from, so a contrast over two seeds cannot
            # be read as one over three.
            "seeds": sorted({row["base"]["seed"] for row in per_pair}),
        }
    )
    return contrast


def paired_level_split(
    results: list[EvalResult], gene: str, other_gene: str
) -> dict:
    """The paired contrast at each level of a second gene.

    A pooled paired contrast is an average over the levels of everything it holds
    fixed, so one level's effect can carry it, or two opposite effects can cancel
    into a null. This is the arm that says which: the head-paired confirmation arm
    found the head effect at +0.0109 nats on a narrow residual stream against
    -0.0462 on a wide one, which pools to -0.0177 and would read as "no head
    effect" without this split.

    The split is descriptive and is reported beside its own tests rather than
    instead of them: agreement in sign across levels says the pooled number is not
    hiding a reversal, and the number that decides whether a level *split* is real
    is the interaction test on the crossed factorial, computed separately.
    """
    at_length = [entry for entry in results if entry.validation_loss is not None]
    levels = sorted({getattr(entry, other_gene, 0) for entry in at_length})
    if len(levels) < 2:
        return {
            "available": False,
            "gene": gene,
            "other_gene": other_gene,
            "reason": f"{other_gene} does not vary, so there is no level to split on",
        }
    split: dict[str, dict] = {}
    missing: list[float] = []
    for level in levels:
        contrast = paired_design_contrast(
            at_length, gene, subgroup=(other_gene, level)
        )
        if not contrast.get("available"):
            missing.append(level)
            continue
        split[str(level)] = contrast
    if missing:
        # Refused rather than reported over the levels that happened to work: a
        # split that silently omits a level is a different question, and the reader
        # has no way to tell it from the design's answer.
        return {
            "available": False,
            "gene": gene,
            "other_gene": other_gene,
            "levels_present": sorted(split),
            "reason": (
                f"no complete pairs at {other_gene} level(s) "
                f"{', '.join(f'{level:g}' for level in missing)}"
            ),
        }
    means = [row["mean_difference"] for row in split.values()]
    agree = all(value < 0 for value in means) or all(value > 0 for value in means)
    reading = (
        "every level agrees in sign, so the pooled contrast is not averaging two "
        "opposite effects away - but agreement is not evidence that the levels "
        "differ, which is the interaction test's question"
        if agree
        else "the levels disagree in sign, so the pooled contrast is an average of "
        "opposite effects and neither level's number can be read off it"
    )
    return {
        "available": True,
        "gene": gene,
        "other_gene": other_gene,
        "levels": split,
        "sign_agreement": agree,
        "reading": reading,
    }


def rank_stability(results: list[EvalResult]) -> list[dict]:
    """Does a short screening run rank architectures the way a long run does?

    This is what decides whether successive halving is a valid search strategy or
    an expensive way to select on initialization speed. If the short-run ranking
    does not survive, then promotion decisions are noise and the screening stages
    are buying nothing.
    """
    lengths = sorted({result.steps for result in results})
    comparisons: list[dict] = []
    for short, long in zip(lengths, lengths[1:]):
        short_means = {name: sum(v.values()) / len(v) for name, v in _losses_by_architecture(results, short).items()}
        long_means = {name: sum(v.values()) / len(v) for name, v in _losses_by_architecture(results, long).items()}
        shared = sorted(set(short_means) & set(long_means))
        if len(shared) < 3:
            continue
        short_values = [short_means[name] for name in shared]
        long_values = [long_means[name] for name in shared]
        rho, _ = scipy.stats.spearmanr(short_values, long_values)
        short_best = min(shared, key=lambda name: short_means[name])
        long_best = min(shared, key=lambda name: long_means[name])
        comparisons.append(
            {
                "from_steps": short,
                "to_steps": long,
                "architectures": len(shared),
                "spearman": None if math.isnan(rho) else float(rho),
                "best_at_from": short_best,
                "best_at_to": long_best,
                "winner_preserved": short_best == long_best,
                "gap_at_from": max(short_values) - min(short_values),
                "gap_at_to": max(long_values) - min(long_values),
            }
        )
    return comparisons


def effect_in_compute_units(
    results: list[EvalResult], steps: int, *, vs_steps: int | None = None
) -> dict:
    """Express the architecture spread as equivalent training steps.

    A 0.04 nat spread means nothing without a scale. The observed slope of loss
    against training length gives one: the spread is ``spread / slope`` extra
    steps of training on the losing architecture. That is directly comparable to
    the search budget the brief caps, which a nat difference is not.
    """
    grouped = _losses_by_architecture(results, steps)
    means = {name: sum(v.values()) / len(v) for name, v in grouped.items()}
    if len(means) < 2:
        return {"available": False}

    spread = max(means.values()) - min(means.values())

    lengths = sorted({result.steps for result in results if result.steps < steps}, reverse=True)
    if vs_steps is None:
        vs_steps = lengths[0] if lengths and lengths[0] < steps else None
    if vs_steps is None:
        return {"available": False, "spread": spread, "reason": "no shorter length to fit a slope"}

    shorter = _losses_by_architecture(results, vs_steps)
    shared = sorted(set(shorter) & set(grouped))
    slopes: list[float] = []
    for name in shared:
        short_values = shorter[name]
        long_values = grouped[name]
        common_seeds = sorted(set(short_values) & set(long_values))
        if not common_seeds:
            continue
        for seed in common_seeds:
            slopes.append((short_values[seed] - long_values[seed]) / (steps - vs_steps))
    if not slopes:
        return {"available": False, "spread": spread, "reason": "no architecture on both lengths"}

    slope = sum(slopes) / len(slopes)
    if slope <= 0:
        # Loss not decreasing over this interval means the slope cannot be used
        # to convert a gap into steps, and pretending otherwise would invent a
        # denominator.
        return {"available": False, "spread": spread, "reason": "loss not decreasing", "slope_per_step": slope}

    equivalent_steps = spread / slope
    return {
        "available": True,
        "spread": spread,
        "slope_per_step": slope,
        "fitted_between": [vs_steps, steps],
        "slope_pairs": len(slopes),
        "equivalent_steps": equivalent_steps,
        "fraction_of_budget": equivalent_steps / steps,
    }


#: Structural axes tested against loss, with human labels. Every one is recorded
#: on every result, so an association is computable without knowing which zoo the
#: runs came from.
COST_AXES: tuple[tuple[str, str], ...] = (
    ("total_params", "total parameters"),
    ("active_params", "active parameters"),
    ("active_fraction", "active / total parameters"),
    ("contracted_flops_per_token", "contracted FLOPs/token"),
    ("dense_runtime_flops_per_token", "dense-runtime FLOPs/token"),
    ("measured_tokens_per_s", "measured throughput"),
    ("depth", "depth"),
    ("hidden_size", "hidden size"),
    ("attention_heads", "attention heads"),
    ("expert_count", "expert count"),
    ("active_experts", "active experts"),
    ("expert_width", "expert width"),
)


def _compact(value: float) -> str:
    """Format a parameter or FLOP count as ``77.2M`` rather than ``7.717e+07``."""
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= scale:
            return f"{value / scale:.3g}{suffix}"
    return f"{value:.4g}"


def _spearman_exact_p(
    values: Sequence[float], losses: Sequence[float], *, cap: int = 60_000
) -> tuple[float | None, bool]:
    """Exact two-sided permutation p-value for a Spearman correlation.

    ``scipy.stats.spearmanr`` derives its p-value from a t approximation that
    returns exactly 0 whenever the correlation reaches +/-1 - which is precisely
    what a designed set with a handful of architectures tends to produce, and
    reading ``p = 0.0000`` off three points would be badly misleading. With n!
    small enough to enumerate, the exact permutation distribution is used: for
    n = 3 a perfect rank correlation is 2 of the 6 arrangements, so p = 1/3.
    """
    n = len(values)
    total = math.factorial(n)
    if total > cap:
        return None, False
    a = list(scipy.stats.rankdata(values))
    b = list(scipy.stats.rankdata(losses))

    def rho(x: Sequence[float], y: Sequence[float]) -> float:
        squared = sum((xi - yi) ** 2 for xi, yi in zip(x, y))
        return 1.0 - 6.0 * squared / (n * (n * n - 1))

    observed = abs(rho(a, b))
    at_least = sum(1 for perm in permutations(b) if abs(rho(a, perm)) >= observed - 1e-12)
    return at_least / total, True


#: Structural genes that live in a zoo *definition* rather than on a result.
#: They are per-architecture constants, so they are joined onto results before any
#: association is computed.
STRUCTURAL_GENES: tuple[str, ...] = (
    "depth",
    "hidden_size",
    "attention_heads",
    "expert_count",
    "active_experts",
    "expert_width",
)


def structural_fields(definition: Path | str) -> dict[str, dict[str, int]]:
    """Structural genes per candidate name, read from a zoo definition.

    Ledgers written before ``EvalResult`` carried these fields record zeros for
    them, and a matched design cannot say which structural gene drives its spread
    without them - which is the whole question a matched design exists to answer.
    The definition holds each candidate's full genome, so the analysis joins on it
    rather than re-deriving structure by parsing candidate names.

    This lives in the library, not beside the reporting script, because
    ``axis_association`` reads the structural genes off the results: a caller that
    forgets the join gets six axes reported as "not recorded" and a leading axis
    chosen from the cost axes alone. That failure is silent and inverts the
    conclusion, so the join is reachable from wherever the analysis is.
    """
    path = Path(definition)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    fields: dict[str, dict[str, int]] = {}
    for candidate in payload.get("candidates", []):
        genome = candidate.get("genome") or {}
        layers = genome.get("layers") or []
        if not layers:
            continue
        widths = [
            expert["width"]
            for layer in layers
            for expert in (layer.get("experts") or [])
        ]
        fields[candidate["name"]] = {
            "depth": len(layers),
            "hidden_size": int(genome.get("hidden_size") or 0),
            "attention_heads": int(layers[0].get("heads") or 0),
            "expert_count": max(len(layer.get("experts") or []) for layer in layers),
            "active_experts": int(layers[0].get("active_experts") or 0),
            "expert_width": max(widths) if widths else 0,
        }
    return fields


def with_structure(
    results: Sequence[EvalResult], definition: Path | str
) -> list[EvalResult]:
    """Fill unrecorded structural genes on results from a zoo definition.

    A result that already carries a gene keeps it, so a ledger written after the
    fields existed is never overwritten by the definition.
    """
    structure = structural_fields(definition)
    if not structure:
        return list(results)
    return [
        replace(
            entry,
            **{
                key: value
                for key, value in structure.get(entry.name, {}).items()
                if getattr(entry, key, 0) == 0
            },
        )
        for entry in results
    ]


def axis_association(results: list[EvalResult], steps: int) -> dict:
    """Which structural axis moves with loss, for a *designed* set of architectures.

    This is meaningful only for a matched design. Among v1's 13 curated
    candidates total and active parameters move together, so no single axis is
    identifiable there and every association is confounded. In a matched design
    one axis is deliberately held and another deliberately spread, which is what
    makes a per-axis correlation readable at all - and is most of the reason to
    build one.

    Exploratory and uncorrected for multiple comparisons. Six axes over a handful
    of architectures will produce a marginal p-value by chance, so a result here
    is a hypothesis for the next experiment, never the conclusion. The
    pre-specified comparison remains the effect summary and its seed-matched
    test.
    """
    grouped = _losses_by_architecture(results, steps)
    if len(grouped) < 3:
        return {"available": False, "reason": "needs 3+ architectures"}

    names = sorted(grouped)
    losses = [sum(grouped[name].values()) / len(grouped[name]) for name in names]
    by_name = {}
    for result in results:
        if result.steps == steps:
            by_name.setdefault(result.name, result)

    spread: list[tuple[str, list[float]]] = []
    axes: list[dict] = []
    for field_name, label in COST_AXES:
        if field_name == "active_fraction":
            values = [
                by_name[name].active_params / by_name[name].total_params
                if by_name[name].total_params
                else 0.0
                for name in names
            ]
        else:
            values = [float(getattr(by_name[name], field_name)) for name in names]
        if len(set(values)) < 2:
            # A held axis has no variance, which is the point of holding it:
            # reporting a correlation against a constant is meaningless. An axis
            # that is zero everywhere means the field was not recorded (a ledger
            # from before it existed) and is reported separately, not as held.
            axes.append(
                {
                    "axis": label,
                    "held": True,
                    "unrecorded": all(value == 0 for value in values),
                    "values": values,
                }
            )
            continue
        rho, approximate_p = scipy.stats.spearmanr(values, losses)
        exact_p, exact = _spearman_exact_p(values, losses)
        spread.append((label, values))
        axes.append(
            {
                "axis": label,
                "held": False,
                "spearman": None if math.isnan(rho) else float(rho),
                "p": exact_p if exact else (None if math.isnan(approximate_p) else float(approximate_p)),
                "p_exact": exact,
                "p_t_approximation": None if math.isnan(approximate_p) else float(approximate_p),
                "min": min(values),
                "max": max(values),
                "spread_ratio": (max(values) / min(values)) if min(values) > 0 else None,
            }
        )

    return {
        "available": True,
        "steps": steps,
        "architectures": len(names),
        "axes": axes,
        "collinearity": _axis_collinearity(spread, axes),
        "multiple_comparisons": (
            f"{len(COST_AXES)} axes tested uncorrected; treat any single marginal "
            "p-value as a hypothesis, not a result. Exact p-values are enumerated "
            "over n! arrangements and are not comparable to a chi-square test."
        ),
    }


#: Two axes correlated at least this strongly cannot be told apart at this
#: sample size, so attributing the spread to one of them is not identified.
ENTANGLED_AT: float = 0.7


def _axis_collinearity(
    spread: Sequence[tuple[str, Sequence[float]]], axes: Sequence[dict]
) -> dict:
    """Which axes are confounded with the one that tracks loss best.

    A designed set spreads some axes and holds others, and among the spread ones
    the generator's own constraint usually ties two together - holding total
    parameters fixed while widening the residual stream forces experts narrower,
    so "wide hidden" and "narrow experts" are one finding seen from both ends.
    Reporting the strongest association without this would invite exactly the
    over-reading the axis table is already cautious about.

    The mirror-image error is worse because it reads as a finding: if the
    structural genes were never joined, they are all zeros, they drop out of
    ``spread``, and the lead is chosen from the cost axes alone. The caveat then
    tells the reader the association is "not explained by a co-varying
    structural gene in this set" when no structural gene was examined at all.
    That is the failure ``structural_fields`` documents, so it is detected and
    stated here instead of being left to every caller to remember.
    """
    if len(spread) < 2:
        return {"available": False, "reason": "needs 2+ spread axes"}
    scored = [
        (row["axis"], row["spearman"])
        for row in axes
        if not row.get("held") and row.get("spearman") is not None
    ]
    if not scored:
        return {"available": False, "reason": "no axis has a computable association"}
    lead_label, lead_rho = max(scored, key=lambda item: abs(item[1]))
    lead_values = dict(spread)[lead_label]
    entangled = []
    for label, values in spread:
        if label == lead_label:
            continue
        rho, _ = scipy.stats.spearmanr(lead_values, values)
        if not math.isnan(rho) and abs(float(rho)) >= ENTANGLED_AT:
            entangled.append({"axis": label, "spearman": float(rho)})
    entangled.sort(key=lambda item: -abs(item["spearman"]))
    missing_structural = [row["axis"] for row in axes if row.get("unrecorded")]
    if missing_structural:
        caveat = (
            f"Among {len(spread)} spread axes, `{lead_label}` tracks loss most "
            f"strongly, but {len(missing_structural)} structural axes are "
            "unrecorded ("
            + ", ".join(f"`{name}`" for name in missing_structural)
            + "), so this lead was chosen from the cost axes alone and no "
            "structural attribution can be made from this report."
        )
    elif entangled:
        caveat = (
            f"Among {len(spread)} spread axes, `{lead_label}` tracks loss most "
            "strongly, but it moves with "
            + ", ".join(
                f"`{item['axis']}` ({item['spearman']:+.2f})" for item in entangled
            )
            + f" at |rho| >= {ENTANGLED_AT}, so this set cannot separate them."
        )
    else:
        caveat = (
            f"Among {len(spread)} spread axes, `{lead_label}` tracks loss most "
            f"strongly, and no other *examined* axis is correlated with it at "
            f"|rho| >= {ENTANGLED_AT}, so no co-varying gene in this set "
            "explains the association."
        )
    return {
        "available": True,
        "leading": lead_label,
        "leading_spearman": lead_rho,
        "entangled": entangled,
        "threshold": ENTANGLED_AT,
        "spread_axes": len(spread),
        "structural_axes_unrecorded": missing_structural,
        "caveat": caveat,
    }


def _ols(
    y: Sequence[float], columns: Sequence[Sequence[float]]
) -> tuple[list[float] | None, float | None]:
    """Least-squares coefficients and R^2, via normal equations.

    Pure Python to match the rest of the module: the designs here have a handful
    of predictors on a dozen architectures, where a linear-algebra dependency
    would buy nothing.

    Returns ``(None, None)`` when the predictors are rank-deficient. That case
    must not be answered with zero coefficients: a caller reading
    ``beta=0, R2=0`` concludes "this gene does nothing", when the truth is that
    the gene is perfectly redundant and its effect is unidentifiable. The
    distinction is the entire point of the fit.
    """
    k = len(columns)
    n = len(y)
    gram = [
        [sum(columns[i][r] * columns[j][r] for r in range(n)) for j in range(k)]
        for i in range(k)
    ]
    rhs = [sum(columns[i][r] * y[r] for r in range(n)) for i in range(k)]
    for col in range(k):
        pivot = max(range(col, k), key=lambda row: abs(gram[row][col]))
        if abs(gram[pivot][col]) < 1e-12:
            return None, None
        gram[col], gram[pivot] = gram[pivot], gram[col]
        rhs[col], rhs[pivot] = rhs[pivot], rhs[col]
        for row in range(col + 1, k):
            factor = gram[row][col] / gram[col][col]
            for inner in range(col, k):
                gram[row][inner] -= factor * gram[col][inner]
            rhs[row] -= factor * rhs[col]
    beta = [0.0] * k
    for row in range(k - 1, -1, -1):
        remaining = rhs[row] - sum(gram[row][c] * beta[c] for c in range(row + 1, k))
        beta[row] = remaining / gram[row][row]
    residual = sum(
        (y[r] - sum(beta[i] * columns[i][r] for i in range(k))) ** 2 for r in range(n)
    )
    total = sum((value - sum(y) / n) ** 2 for value in y)
    return beta, (1 - residual / total) if total else 0.0


def _zscore(values: Sequence[float]) -> list[float]:
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    return [(value - mean) / sd for value in values] if sd else [0.0] * len(values)


def joint_structural_fit(
    results: list[EvalResult],
    steps: int,
    *,
    max_predictors: int = 3,
    samples: int = 20_000,
    seed: int = 0,
    genes: Sequence[str] | None = None,
) -> dict:
    """Fit loss on the structural genes *together*, for an observational set.

    ``axis_association`` tests each gene on its own, which is the right test for a
    design that spread one axis and held the rest. On a set the generator did not
    orthogonalise - v1's curated families trade depth against residual width, at
    rho = -0.80 - a gene can look strong alone and add nothing once its partner
    is in the model, or the reverse. Reading the per-axis table as a ranking of
    levers is the error this function exists to catch.

    Coefficients are standardised, so each is "nats per SD of this gene, holding
    the others fixed". p-values permute the losses across architectures, which is
    the same null the per-axis Spearman test uses: architectures are exchangeable
    under "architecture does not matter".

    Pass ``genes`` to fit a **pre-specified** model rather than selecting the
    strongest predictors first. The distinction matters when the same fit is
    compared across subsets: selection picks a different model in each one, so an
    apparently stable R^2 can be a different hypothesis fitting equally well. A
    pre-specified model answers "does *this* model hold here", which is the
    question a designed factorial has already licensed.
    """
    grouped = _losses_by_architecture(results, steps)
    names = sorted(grouped)
    if len(names) < 3:
        return {"available": False, "reason": "needs 3+ architectures"}
    losses = [sum(grouped[name].values()) / len(grouped[name]) for name in names]
    by_name: dict[str, EvalResult] = {}
    for result in results:
        if result.steps == steps:
            by_name.setdefault(result.name, result)

    requested = list(genes) if genes else None
    if requested:
        unknown = [gene for gene in requested if gene not in STRUCTURAL_GENES]
        if unknown:
            return {
                "available": False,
                "reason": f"not structural genes: {', '.join(unknown)}",
            }
    candidates = []
    for gene in (requested if requested else STRUCTURAL_GENES):
        values = [float(getattr(by_name[name], gene)) for name in names]
        if len(set(values)) < 2 or all(value == 0 for value in values):
            # Held (by design) or unrecorded (pre-structural ledger), and neither
            # can carry a coefficient.
            if requested:
                return {
                    "available": False,
                    "reason": (
                        f"`{gene}` was requested but has no variance in this set "
                        "(held by design, or not recorded)"
                    ),
                }
            continue
        rho = scipy.stats.spearmanr(values, losses).statistic
        candidates.append((gene, values, abs(0.0 if math.isnan(rho) else float(rho))))
    if not candidates:
        return {
            "available": False,
            "reason": "no structural gene was recorded with variance",
            "structural_axes_unrecorded": True,
        }
    # A pre-specified model is kept in the order it was requested and never
    # truncated: reordering or trimming it would quietly turn it back into a
    # selected model, and the coefficients are only comparable to each other if
    # the model is the one that was asked for.
    if not requested:
        candidates.sort(key=lambda item: -item[2])
    # Drop genes that are *exactly* determined by one already taken. The family
    # fixes `attention_heads = hidden_size / 64`, so those two columns are
    # identical after standardisation and a fit over both is rank-deficient.
    # Near-collinear genes are kept - separating them is what the caveat below
    # is for - but an exact duplicate carries no information at all, and keeping
    # it would make the whole fit unidentifiable rather than just that pair.
    chosen: list[tuple[str, list[float], float]] = []
    dropped: list[dict] = []
    for candidate in candidates:
        if len(chosen) >= max_predictors:
            break
        duplicate = None
        for accepted in chosen:
            rho = float(scipy.stats.spearmanr(candidate[1], accepted[1]).statistic)
            if abs(rho) >= 1 - 1e-9:
                duplicate = (accepted[0], rho)
                break
        if duplicate:
            if requested:
                # Dropping it would silently answer a different question: the
                # caller asked whether *this* model holds, and a model with one of
                # its predictors removed is not that model.
                return {
                    "available": False,
                    "reason": (
                        f"requested model is rank-deficient: `{candidate[0]}` is "
                        f"identical to `{duplicate[0]}` (rho={duplicate[1]:+.4f}) at "
                        "this set, so neither coefficient is identifiable"
                    ),
                    "dropped": [
                        {
                            "gene": candidate[0],
                            "duplicate_of": duplicate[0],
                            "spearman": duplicate[1],
                        }
                    ],
                }
            dropped.append(
                {
                    "gene": candidate[0],
                    "duplicate_of": duplicate[0],
                    "spearman": duplicate[1],
                }
            )
            continue
        chosen.append(candidate)
    if len(chosen) < 2:
        return {
            "available": False,
            "reason": "needs 2+ independent spread structural genes to fit jointly",
            "dropped": dropped,
        }

    columns = [_zscore(item[1]) for item in chosen]
    response = _zscore(losses)
    beta, r2 = _ols(response, columns)
    if beta is None or r2 is None:
        return {
            "available": False,
            "reason": "predictors are rank-deficient at this sample size",
            "genes": [item[0] for item in chosen],
            "dropped": dropped,
        }

    rng = random.Random(seed)
    generator_null_betas = [[] for _ in chosen]
    null_r2 = []
    for _ in range(samples):
        shuffled = response[:]
        rng.shuffle(shuffled)
        null_beta, null_fit = _ols(shuffled, columns)
        for index, value in enumerate(null_beta):
            generator_null_betas[index].append(value)
        null_r2.append(null_fit)

    def tail(observed: float, null: Sequence[float]) -> float:
        extreme = sum(1 for value in null if abs(value) >= abs(observed) - 1e-12)
        return (extreme + 1) / (len(null) + 1)

    genes = []
    for index, (gene, values, _) in enumerate(chosen):
        genes.append(
            {
                "gene": gene,
                "beta": beta[index],
                "p": tail(beta[index], generator_null_betas[index]),
                "min": min(values),
                "max": max(values),
            }
        )
    collinear = [
        (chosen[i][0], chosen[j][0], float(scipy.stats.spearmanr(chosen[i][1], chosen[j][1]).statistic))
        for i in range(len(chosen))
        for j in range(i + 1, len(chosen))
        if abs(float(scipy.stats.spearmanr(chosen[i][1], chosen[j][1]).statistic)) >= ENTANGLED_AT
    ]
    return {
        "available": True,
        "steps": steps,
        "architectures": len(names),
        "genes": genes,
        "r_squared": r2,
        "p_model": tail(r2, null_r2),
        "entangled": [
            {"a": a, "b": b, "spearman": rho} for a, b, rho in collinear
        ],
        "dropped": dropped,
        "samples": samples,
        "caveat": (
            "Coefficients are standardised and each p-value is two-sided against "
            "a permutation null, but predictors are collinear by construction "
            "here, so a coefficient surviving the model does not mean the "
            "underlying lever is identified. Use a designed set to separate them."
        )
        + (
            "; "
            + "; ".join(
                f"`{a}` and `{b}` move together at rho={rho:+.2f}"
                for a, b, rho in collinear
            )
            + " and cannot be told apart at this size, so their split in the fit "
            "is arbitrary."
            if collinear
            else ""
        ),
    }


def spread_by_length(results: list[EvalResult]) -> dict[str, dict]:
    """Spread at each training length, on architectures common to all lengths.

    Comparing the spread at 400 steps against the spread at 1200 steps across
    *different* architecture sets confounds the effect with the set, so this
    restricts to the intersection.
    """
    lengths = sorted({result.steps for result in results})
    per_length = {length: _losses_by_architecture(results, length) for length in lengths}
    if not lengths:
        return {}
    common = set.intersection(*(set(per_length[length]) for length in lengths))
    if not common:
        return {}
    summary: dict[str, dict] = {}
    for length in lengths:
        means = {
            name: sum(per_length[length][name].values()) / len(per_length[length][name])
            for name in sorted(common)
        }
        # Seed noise is carried alongside the spread because they decay together:
        # in this project the spread falls with training but so does the
        # architecture-to-architecture seed scatter, so a shrinking spread is not
        # by itself evidence that the effect washes out. The ratio is the number
        # that says whether a longer run would still separate them.
        stds = []
        for name in sorted(common):
            values = list(per_length[length][name].values())
            if len(values) >= 2:
                mean = sum(values) / len(values)
                stds.append(
                    math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
                )
        noise = sum(stds) / len(stds) if stds else None
        spread = max(means.values()) - min(means.values())
        summary[str(length)] = {
            "architectures": len(means),
            "spread": spread,
            "within_seed_std": noise,
            "sigmas": (spread / noise) if noise else None,
            "best": min(means, key=lambda name: means[name]),
        }

    ordered = [summary[str(length)] for length in lengths]
    first, last = ordered[0], ordered[-1]
    if first["spread"] > 0 and last["spread"] > 0 and lengths[-1] > lengths[0]:
        doublings = math.log2(lengths[-1] / lengths[0])
        summary["decay"] = {
            "from_steps": lengths[0],
            "to_steps": lengths[-1],
            "doublings": doublings,
            "spread_shrink": first["spread"] / last["spread"],
            "per_doubling": (first["spread"] / last["spread"]) ** (1 / doublings),
            "sigmas_at_first": first["sigmas"],
            "sigmas_at_last": last["sigmas"],
            "best_at_every_length": len({row["best"] for row in ordered}) == 1,
        }
    return summary


def structural_signal_by_length(
    results: list[EvalResult],
    *,
    max_predictors: int = 3,
    samples: int = 20_000,
    genes: Sequence[str] | None = None,
) -> dict[str, dict]:
    """Joint structural fit at each training length, on common architectures.

    ``spread_by_length`` says the spread shrinks with training. It does not say
    whether the *reason* for the spread is still readable at the shorter length,
    which is the question that decides how much a screen has to be paid for: if
    the same model over the same genes is significant at 400 steps, a search can
    screen there and confirm the finalists at full length rather than screening
    at full length.

    Same restriction as ``spread_by_length``: only architectures run at every
    length, so a length's fit is never compared against a different set.

    Pass ``genes`` to hold the model fixed across lengths. Without it each length
    selects its own strongest predictors, and "the same model holds at every
    length" then means only that *some* model with the same R^2 does. With it, the
    question is the sharper one a designed factorial licenses: does this
    specific model hold at each length, with the same coefficient.
    """
    lengths = sorted({result.steps for result in results})
    if len(lengths) < 2:
        return {}
    per_length = {length: _losses_by_architecture(results, length) for length in lengths}
    common = set.intersection(*(set(per_length[length]) for length in lengths))
    if len(common) < 3:
        return {}
    kept = [result for result in results if result.name in common]
    summary: dict[str, dict] = {}
    for length in lengths:
        fit = joint_structural_fit(
            kept,
            length,
            max_predictors=max_predictors,
            samples=samples,
            genes=genes,
        )
        summary[str(length)] = {
            "available": fit.get("available", False),
            "reason": fit.get("reason"),
            "architectures": fit.get("architectures", len(common)),
            "r_squared": fit.get("r_squared"),
            "p_model": fit.get("p_model"),
            "genes": [
                {"gene": row["gene"], "beta": row["beta"], "p": row["p"]}
                for row in fit.get("genes", [])
            ],
            "dropped": fit.get("dropped", []),
        }
    if genes:
        summary["model"] = {"pre_specified": list(genes)}
    fitted = [
        row
        for key, row in summary.items()
        if key not in ("agreement", "model") and row.get("available")
    ]
    if len(fitted) >= 2:
        short, long = fitted[0], fitted[-1]
        summary["agreement"] = {
            "leading_genes": [
                row["gene"] for row in (short["genes"][:1] + long["genes"][:1])
            ],
            "same_leading_gene": (
                bool(short["genes"]) and bool(long["genes"])
                and short["genes"][0]["gene"] == long["genes"][0]["gene"]
            ),
            "same_leading_sign": (
                bool(short["genes"]) and bool(long["genes"])
                and (short["genes"][0]["beta"] > 0) == (long["genes"][0]["beta"] > 0)
            ),
        }
    return summary


def crossed_contrast(
    results: list[EvalResult],
    steps: int,
    *,
    samples: int = 200_000,
    max_assignments: int = 1_000_000,
    seeds: Sequence[int] | None = None,
) -> dict:
    """2x2 factorial contrast over two binary structural genes.

    A matched design answers *whether* architecture matters but not *which gene*,
    and its marginal rank correlations cannot separate two genes that move
    together: a zoo that trades residual-stream width against expert width
    reports both with the same association and the axis table cannot say which is
    doing the work. That was the v2 experiment's open confound - hidden size and
    expert width correlated at -0.74, so "hidden size matters" and "expert width
    does not" were one finding seen from two ends.

    A design that *crosses* two genes at two levels each can separate them,
    because each gene's main effect is then an average over both levels of the
    other rather than a contrast confounded with it. The same data also gives the
    interaction, which is the thing a marginal association cannot express at all
    - whether a gene's cost depends on the level of another.

    Genes are read off the results (see ``with_structure``), and the design
    qualifies only if exactly two carry exactly two levels each with all four
    cells populated on every seed.

    "Every seed" is the target, not always what a sweep has. An interrupted run
    leaves its last seed with fewer rows than the others, and refusing the
    contrast there would drop the phase's headline test over a run that has not
    finished - the same failure the ledger globbing was already fixed for once. So
    the seeds that did not finish are dropped and *returned* in ``seeds_dropped``,
    and the main effects are then averages over the seeds named in ``seeds``
    rather than over every seed the ledger mentions. Only seeds short of the
    fullest one's row count may be dropped: a seed that ran everything and still
    fails a balance check is a defect in the zoo, and that still refuses.

    The null is the design's own blocking. For a main effect of factor A the
    a-levels are exchangeable *within* each (b-level, seed) block, so each
    block's pooled runs are re-split between the two a-arms; seeds block for the
    same reason the seed-matched test blocks on them, so the shared seed effect
    cancels. Enumerated exactly when feasible, Monte-Carlo from the same null
    otherwise.
    """
    at_length = [entry for entry in results if entry.steps == steps]
    binary: dict[str, list[int]] = {}
    for gene in STRUCTURAL_GENES:
        values = sorted({getattr(entry, gene, 0) for entry in at_length})
        if len(values) == 2 and values[0]:
            binary[gene] = values
    if len(binary) < 2:
        return {"available": False, "reason": "fewer than two binary structural genes"}

    discovered = sorted({entry.seed for entry in at_length})
    considered = sorted(set(seeds) & set(discovered)) if seeds else discovered
    if not considered:
        return {
            "available": False,
            "reason": "none of the requested seeds have runs at this length",
            "seeds": [],
            "seeds_dropped": [],
        }

    def cells(
        factor_a: str, factor_b: str, block: Sequence[int]
    ) -> dict[tuple[int, int], dict[int, list[float]]] | None:
        out: dict[tuple[int, int], dict[int, list[float]]] = {}
        for a_value in binary[factor_a]:
            for b_value in binary[factor_b]:
                for seed in block:
                    found = [
                        entry.validation_loss
                        for entry in at_length
                        if getattr(entry, factor_a) == a_value
                        and getattr(entry, factor_b) == b_value
                        and entry.seed == seed
                    ]
                    if not found:
                        return None
                    out[(a_value, b_value)] = {**out.get((a_value, b_value), {}), seed: found}
        # Populated is not enough: the arms must be *balanced* too. A cell with two
        # runs on one level and one on the other cannot contribute a level
        # contrast, so `block_contributions` has to skip it - and then one level's
        # mean is taken over fewer blocks than the other's, which makes the
        # interaction a difference of incomparable means. That happens for real:
        # dropping one candidate from v3 leaves the D14/H896 cell lopsided. Better
        # to call the design out of balance here than to report a contrast built
        # from an unbalanced subset.
        for b_value in binary[factor_b]:
            for seed in block:
                counts = {
                    len(out[(a_value, b_value)][seed]) for a_value in binary[factor_a]
                }
                if len(counts) != 1:
                    return None
        for a_value in binary[factor_a]:
            for seed in block:
                counts = {
                    len(out[(a_value, b_value)][seed]) for b_value in binary[factor_b]
                }
                if len(counts) != 1:
                    return None
        return out

    def usable_seeds(factor_a: str, factor_b: str) -> list[int]:
        """Seeds on which this crossing is complete and balanced on their own."""
        return [
            seed for seed in considered if cells(factor_a, factor_b, [seed]) is not None
        ]

    # Seeds still running carry fewer rows than the fullest seed, and only those
    # may be dropped. A seed that ran everything and still fails the balance check
    # is a zoo defect, and the refusal below is for that case.
    short = set(_short_seeds(at_length, considered))
    chosen: tuple[str, str] | None = None
    retained: list[int] = considered
    for factor_a, factor_b in combinations(binary, 2):
        if len(usable_seeds(factor_a, factor_b)) == len(considered):
            chosen = (factor_a, factor_b)
            break
    if chosen is None and short:
        best: tuple[tuple[str, str], list[int]] | None = None
        required = len(considered) - len(short)
        for factor_a, factor_b in combinations(binary, 2):
            kept = [
                seed for seed in usable_seeds(factor_a, factor_b) if seed not in short
            ]
            # Every seed that is *not* still running has to be usable, or this is
            # not the design's crossing and the drop would be hiding a defect.
            if kept and len(kept) == required:
                if best is None or len(kept) > len(best[1]):
                    best = ((factor_a, factor_b), kept)
        if best is not None:
            chosen, retained = best
    if chosen is None:
        return {
            "available": False,
            "reason": (
                "no crossed pair populates all four cells with balanced arms on "
                "every seed"
            ),
            "seeds": considered,
            "seeds_dropped": [],
        }

    factor_a, factor_b = chosen
    seeds = retained
    dropped = [seed for seed in considered if seed not in seeds]
    a_levels, b_levels = binary[factor_a], binary[factor_b]
    grid = cells(factor_a, factor_b, seeds)
    assert grid is not None

    def block_contributions(
        factor: str,
        levels: tuple[int, int],
        other_levels: list[int],
        other: str,
    ) -> tuple[list[float], list[list[float]]]:
        """Per-block (level-high minus level-low) contrasts, and their null set.

        Returns the observed contribution per block plus, for every block, all
        contributions attainable by re-splitting that block's pooled runs between
        the two arms. Enumerating from the splits rather than re-deriving the
        model keeps the null the design's actual randomization: with 2 runs per
        arm a block admits C(4, 2) = 6 splits.
        """
        observed: list[float] = []
        attainable: list[list[float]] = []
        level_counts: list[int] = []
        for other_value in other_levels:
            count = 0
            for seed in seeds:
                low: list[float] = []
                high: list[float] = []
                for entry in at_length:
                    if entry.seed != seed or getattr(entry, other) != other_value:
                        continue
                    if getattr(entry, factor) == levels[0]:
                        low.append(entry.validation_loss)
                    elif getattr(entry, factor) == levels[1]:
                        high.append(entry.validation_loss)
                if not low or len(low) != len(high):
                    # A cell with one arm missing cannot contribute a contrast, and
                    # contributing it as zero would drag the level mean. Skipping is
                    # right; what matters is that the *count* is returned so a caller
                    # can tell whether the two levels still have equal numbers of
                    # blocks. They have to, because the interaction is a difference
                    # of level means, and a mean over 5 blocks is not comparable to
                    # one over 6.
                    continue
                observed.append(sum(high) / len(high) - sum(low) / len(low))
                count += 1
                pooled = low + high
                size = len(low)
                # `rest - pick` rather than the other way round because `pooled` is
                # `low + high`: the first split picks the low runs, so this ordering
                # makes attainable[0] the observed contrast itself. That invariant is
                # what lets the observed statistic be looked up as index 0 of every
                # block instead of being special-cased, and it is asserted by test.
                attainable.append(
                    [
                        sum(rest) / size - sum(pick) / size
                        for split in combinations(range(len(pooled)), size)
                        for pick, rest in [
                            (
                                [pooled[i] for i in split],
                                [pooled[i] for i in range(len(pooled)) if i not in split],
                            )
                        ]
                    ]
                )
            level_counts.append(count)
        return observed, attainable, level_counts

    def main_effect(factor: str, levels: list[int], other: str) -> dict:
        other_values = [value for value in binary[other]]
        observed_blocks, attainable, level_counts = block_contributions(
            factor, (levels[0], levels[1]), other_values, other
        )
        observed = sum(observed_blocks) / len(observed_blocks)
        assignments = 1
        for block in attainable:
            assignments *= len(block)

        def statistic(picks: Sequence[int]) -> float:
            return sum(block[index] for block, index in zip(attainable, picks)) / len(attainable)

        # The interaction is the difference between the per-other-level contrasts.
        # It is tested against the same re-split null, which is the *joint* null of
        # no main effect and no interaction: a valid but conservative test for the
        # interaction alone, and labeled as such rather than left implicit.
        #
        # The split uses the *actual* per-level block counts rather than the seed
        # count. Those agree for a complete design and diverge the moment a run is
        # missing - a leave-one-out set, or an interrupted sweep - and splitting on
        # the seed count there silently mixes blocks from the two levels, or slices
        # an empty level and divides by zero.
        balanced_levels = len(set(level_counts)) == 1 and len(level_counts) == 2
        per_level_blocks = level_counts[0] if level_counts else 0

        def interaction_statistic(picks: Sequence[int]) -> float:
            values = [attainable[i][picks[i]] for i in range(len(attainable))]
            if not balanced_levels or per_level_blocks == 0:
                return 0.0
            first = values[:per_level_blocks]
            second = values[per_level_blocks:]
            return sum(first) / len(first) - sum(second) / len(second)

        # `picks` holds one index *per block*, so the observed assignment is all
        # zeros: index 0 of each block is that block's observed contrast (see
        # `block_contributions`). Using `range(len(attainable))` here indexes past
        # the end of every block but the longest, which is a crash rather than a
        # wrong number, and it was.
        observed_interaction = interaction_statistic([0] * len(attainable))

        if assignments <= max_assignments:
            at_least = 0
            at_least_interaction = 0
            total = 0
            for picks in product(*[range(len(block)) for block in attainable]):
                total += 1
                if abs(statistic(picks)) >= abs(observed) - 1e-12:
                    at_least += 1
                if abs(interaction_statistic(picks)) >= abs(observed_interaction) - 1e-12:
                    at_least_interaction += 1
            p_value, method = at_least / total, "exact"
            p_interaction = at_least_interaction / total
        elif samples > 0:
            generator = random.Random(0)
            at_least = 0
            at_least_interaction = 0
            for _ in range(samples):
                picks = [generator.randrange(len(block)) for block in attainable]
                if abs(statistic(picks)) >= abs(observed) - 1e-12:
                    at_least += 1
                if abs(interaction_statistic(picks)) >= abs(observed_interaction) - 1e-12:
                    at_least_interaction += 1
            p_value = (at_least + 1) / (samples + 1)
            p_interaction = (at_least_interaction + 1) / (samples + 1)
            method = "monte-carlo"
        else:
            p_value, method, p_interaction = None, "unavailable", None

        # The observed contributions are appended in (other level, seed) order,
        # so they group back into one contrast per level of the crossed factor.
        # The difference between those two contrasts is the interaction: it asks
        # whether this gene's cost depends on the other gene's level, which a
        # marginal association cannot express at all.
        # Grouped by the counts the contributions actually came back in, for the
        # same reason the split above uses them: `seeds` is the design's *intended*
        # block count, and a design missing a run has fewer.
        grouped: dict[int, list[float]] = {level: [] for level in other_values}
        index = 0
        for position, other_value in enumerate(other_values):
            for _ in range(level_counts[position] if position < len(level_counts) else 0):
                grouped[other_value].append(observed_blocks[index])
                index += 1
        level_means = {
            level: (sum(values) / len(values) if values else None)
            for level, values in grouped.items()
        }
        interaction = (
            level_means[other_values[0]] - level_means[other_values[1]]
            if len(other_values) == 2
            and balanced_levels
            and None not in (level_means[other_values[0]], level_means[other_values[1]])
            else None
        )
        # The re-split null above assumes no main effect either, so it is
        # conservative for the interaction: a real main effect inflates the null's
        # spread and hides a real interaction. There is a better-calibrated test,
        # because the same architectures run on every seed, so the seed-level
        # interaction contrast is a *paired* observation and the seeds are blocks.
        #
        # It is reported alongside rather than instead, because it rests on a
        # weaker guarantee: the re-split test is exact under the design's own
        # randomization, while a sign test on the seed contrasts assumes those
        # contrasts are symmetric about zero under the null. Reported together, a
        # reader can see the interaction is not being waved through on an
        # assumption, and the pair is more informative than either alone.
        seed_blocked_p = None
        seed_blocked_floor = None
        paired_t_p = None
        if len(other_values) == 2 and interaction is not None:
            # One contrast per seed, on the arm it belongs to, is (other level 0)
            # minus (other level 1); this is exactly `grouped` differenced per seed.
            per_seed = [
                grouped[other_values[0]][index] - grouped[other_values[1]][index]
                for index in range(min(len(values) for values in grouped.values()))
            ]
            observed_seed_contrast = sum(per_seed) / len(per_seed)
            if per_seed:
                at_least_seed = 0
                for signs in product((1, -1), repeat=len(per_seed)):
                    flipped = sum(s * v for s, v in zip(signs, per_seed)) / len(per_seed)
                    if abs(flipped) >= abs(observed_seed_contrast) - 1e-12:
                        at_least_seed += 1
                seed_blocked_p = at_least_seed / (2 ** len(per_seed))
                # Two-sided, so the smallest this can report is 2/2^n: one sign
                # vector matching the observed signs and its mirror. With three
                # seeds that is 0.25, which is *not* weak evidence - it is the
                # most extreme result the test can express. Reporting the floor
                # next to the p is what stops "p = 0.25, all seeds agreeing"
                # being read as "the interaction is probably absent", which is the
                # same reading error the spread-decay rule made on a floor-bound
                # statistic.
                seed_blocked_floor = 2 / (2 ** len(per_seed))
            if len(per_seed) >= 2:
                mean = observed_seed_contrast
                variance = (
                    sum((v - mean) ** 2 for v in per_seed) / (len(per_seed) - 1)
                )
                if variance > 0:
                    t_value = mean / math.sqrt(variance / len(per_seed))
                    paired_t_p = float(
                        2 * scipy.stats.t.sf(abs(t_value), df=len(per_seed) - 1)
                    )
        # The smallest p an exact two-sided re-split test can report is
        # 2/assignments: the observed assignment always counts itself, and its
        # mirror counts with it. Reported because these main effects land on it -
        # a 46656-assignment enumeration cannot distinguish "the observed
        # separation is the largest this design can express" from "this effect
        # would be this significant at any scale", and a reader quoting
        # p = 4.3e-05 deserves to know which one it is.
        # A Monte-Carlo fallback has its own floor, 1/(samples+1), because the
        # estimator adds one to the count. Reporting ``None`` there would leave
        # the floor column looking as though no bound applied, which is how a
        # sampling resolution gets mistaken for a strong result.
        if method == "exact":
            floor = 2.0 / assignments
        elif method == "monte-carlo":
            floor = 1.0 / (samples + 1)
        else:
            floor = None
        return {
            "nats": observed,
            "p": p_value,
            "min_achievable_p": floor,
            "method": method,
            "assignments": assignments,
            "at_floor": (
                floor is not None and p_value is not None and p_value <= floor + 1e-15
            ),
            "by_other_level": {
                str(level): {"nats": level_means[level], "blocks": len(values)}
                for level, values in grouped.items()
            },
            "interaction_nats": interaction,
            "interaction_p": p_interaction,
            "interaction_p_seed_blocked": seed_blocked_p,
            "interaction_p_seed_blocked_floor": seed_blocked_floor,
            "interaction_p_paired_t": paired_t_p,
            "interaction_seed_contrasts": [
                grouped[other_values[0]][index] - grouped[other_values[1]][index]
                for index in range(min(len(values) for values in grouped.values()))
            ]
            if len(other_values) == 2
            else [],
        }

    return {
        "available": True,
        "steps": steps,
        "factor_a": factor_a,
        "factor_b": factor_b,
        # The seeds the contrast is actually averaged over, and the ones that could
        # not be. Without both, a report over five of six seeds is
        # indistinguishable from one over six.
        "seeds": seeds,
        "seeds_dropped": dropped,
        "levels": {factor_a: a_levels, factor_b: b_levels},
        "cells": {
            f"{factor_a}={a_value}|{factor_b}={b_value}": {
                "runs": sum(len(v) for v in grid[(a_value, b_value)].values()),
                "mean_val_loss": sum(
                    loss for v in grid[(a_value, b_value)].values() for loss in v
                )
                / sum(len(v) for v in grid[(a_value, b_value)].values()),
            }
            for a_value in a_levels
            for b_value in b_levels
        },
        "main": {
            factor_a: main_effect(factor_a, a_levels, factor_b),
            factor_b: main_effect(factor_b, b_levels, factor_a),
        },
    }


def effect_summary(
    results: list[EvalResult],
    steps: int,
    seeds: list[int],
    *,
    samples: int = 200_000,
    signal_genes: Sequence[str] | None = None,
) -> dict:
    """Separate the architecture effect from initialization variance.

    Paired deltas use only seeds where two architectures were *both* run, which
    is what makes the comparison seed-matched rather than merely
    same-seed-count (brief section 20).

    ``samples`` bounds the resolution of the seed-matched test when the
    assignment space is too large to enumerate, and so bounds the smallest
    p-value the report can state. It is a parameter rather than a constant
    because a design with an enormous spread reports ``p <= 1/(samples+1)``,
    which is a property of the procedure and not of the world; raising it is how
    a reader closes the gap between that bound and the design's own floor.
    """
    ranked = rank(results, steps)
    if not ranked:
        return {"available": False, "steps": steps, "seeds": seeds}

    grouped = _losses_by_architecture(results, steps)

    per_arch_std: list[float] = []
    for by_seed in grouped.values():
        if len(by_seed) < 2:
            continue
        values = list(by_seed.values())
        mean = sum(values) / len(values)
        per_arch_std.append(math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)))

    best_name, best_loss, best_runs = ranked[0]
    paired = []
    for name, _, _ in ranked[1:]:
        shared = sorted(set(grouped.get(name, {})) & set(grouped.get(best_name, {})))
        if not shared:
            continue
        deltas = [grouped[name][seed] - grouped[best_name][seed] for seed in shared]
        paired.append(
            {
                "candidate": name,
                "mean_delta_vs_best": sum(deltas) / len(deltas),
                "paired_seeds": shared,
                "delta_per_seed": deltas,
            }
        )

    spread = ranked[-1][1] - best_loss
    noise = sum(per_arch_std) / len(per_arch_std) if per_arch_std else 0.0
    multi_seed = bool(per_arch_std)

    # Successive halving promotes architectures using shorter runs on a
    # correlated measurement, which biases a later test of those same
    # architectures toward looking different. Whether that happened is a fact
    # about the data, not about the plan: it happened iff more architectures were
    # run at some shorter length than were run here. A flat design ran every
    # candidate at this length, so asserting the caveat would understate its
    # evidence.
    shorter_counts = [
        len(_losses_by_architecture(results, length))
        for length in {result.steps for result in results}
        if length < steps
    ]
    selected_from = max(shorter_counts) if shorter_counts else 0
    selection_caveat = (
        f"{selected_from} architectures were screened at shorter lengths than "
        f"these {len(grouped)}, and promotion used those runs, so this test is "
        "mildly anti-conservative. Confirming on seeds never used in screening "
        "removes it."
        if selected_from > len(grouped)
        else ""
    )
    return {
        "available": True,
        "steps": steps,
        "ranked": [{"name": n, "mean_val_loss": m, "runs": c} for n, m, c in ranked],
        "best": best_name,
        "best_mean_val_loss": best_loss,
        "best_runs": best_runs,
        "architecture_spread": spread,
        "within_seed_std": noise,
        # Descriptive only. It compares a range that grows with the number of
        # architectures against a per-architecture standard deviation, so it is
        # not a test; the p-values below are. Kept because it is the number the
        # earlier phase's decision rule used, and the comparison is the point.
        "sigmas": (spread / noise) if multi_seed and noise > 0 else None,
        "multi_seed": multi_seed,
        "paired_vs_best": paired,
        "seeds": seeds,
        "seeds_per_architecture": {name: sorted(by_seed) for name, by_seed in grouped.items()},
        "anova": variance_decomposition(results, steps),
        "blocked_test": blocked_permutation_test(results, steps, samples=samples),
        "rank_stability": rank_stability(results),
        "spread_by_length": spread_by_length(results),
        "structural_signal_by_length": structural_signal_by_length(
            results, genes=signal_genes
        ),
        "axis_association": axis_association(results, steps),
        "joint_structural_fit": joint_structural_fit(results, steps),
        # `seeds` is threaded through, not left to the ledger, so a caller who
        # names the seeds an arm is to be read over gets exactly those: a sweep
        # whose last seed is still training can be reported on the seeds that
        # finished without an edit to the analysis.
        "crossed_contrast": crossed_contrast(
            results, steps, samples=samples, seeds=seeds
        ),
        "paired_contrasts": _paired_contrasts_at(results, steps),
        "paired_level_splits": _paired_level_splits_at(results, steps),
        "compute_units": effect_in_compute_units(results, steps),
        "selection_caveat": selection_caveat,
        "screened_at_shorter_lengths": selected_from,
    }


def best_available_summary(
    results: list[EvalResult],
    seeds: list[int],
    *,
    samples: int = 200_000,
    signal_genes: Sequence[str] | None = None,
) -> dict:
    """Effect summary at the longest training length that actually completed.

    A sweep reports what it has. Confirmation stages are strictly longer than
    screening, so the longest completed length is the most informative evidence
    available, and falling back keeps a partial run analysable.
    """
    lengths = sorted({result.steps for result in results}, reverse=True)
    for length in lengths:
        summary = effect_summary(
            results, length, seeds, samples=samples, signal_genes=signal_genes
        )
        if summary.get("available"):
            return summary
    return {"available": False}


def search_cost(results: list[EvalResult], candidates: int) -> dict:
    """Total cost of *discovering* an architecture, not just of training the winner."""
    training_flops = sum(r.training_flops for r in results)
    evaluation_flops = sum(r.validation_flops for r in results)
    seconds = sum(r.elapsed_s for r in results)
    return {
        "candidates_generated": candidates,
        "runs_completed": len(results),
        "candidate_training_flops": training_flops,
        "candidate_evaluation_flops": evaluation_flops,
        "search_overhead_flops": 0.0,
        "total_search_flops": training_flops + evaluation_flops,
        "tokens_trained": sum(r.train_tokens for r in results),
        "wall_clock_hours": seconds / 3600.0,
        "by_stage": {
            str(steps): {
                "runs": sum(1 for r in results if r.steps == steps),
                "tokens": sum(r.train_tokens for r in results if r.steps == steps),
                "elapsed_hours": sum(r.elapsed_s for r in results if r.steps == steps) / 3600.0,
            }
            for steps in sorted({r.steps for r in results})
        },
    }


def verdict(outcome: dict) -> tuple[str, str]:
    """Turn the evidence bundle into a decision and the next action.

    The rule reads the seed-matched p-value first and the effect size second. A
    difference that is not established is *not* evidence of absence at these
    sample sizes, and the wording says so rather than implying architecture
    does not matter.
    """
    if not outcome.get("available"):
        return (
            "INSUFFICIENT DATA - no stage produced results.",
            "Run at least one stage to completion before drawing a conclusion.",
        )
    if not outcome.get("multi_seed"):
        return (
            "PROVISIONAL - single-seed evidence, so the spread is not yet separable "
            f"from initialization noise (spread {outcome['architecture_spread']:.4f} nats).",
            "Run at least two seeds per architecture before deciding. One seed "
            "per architecture leaves no noise estimate at all, so no spread is "
            "interpretable however large it looks.",
        )

    spread = outcome["architecture_spread"]
    blocked = outcome.get("blocked_test") or {}
    anova = outcome.get("anova") or {}
    p_value = blocked.get("p")
    test_name = blocked.get("test", "blocked test")
    if p_value is None:
        p_value = anova.get("p")
        test_name = anova.get("test", "unpaired test")

    units = outcome.get("compute_units") or {}
    size = (
        f", worth {units['equivalent_steps']:.0f} training steps "
        f"({units['fraction_of_budget'] * 100:.1f}% of the {outcome['steps']}-step budget)"
        if units.get("available")
        else ""
    )

    if spread <= MATERIALITY_FLOOR_NATS:
        return (
            f"NO MATERIAL SIGNAL - spread {spread:.4f} nats is below the "
            f"{MATERIALITY_FLOOR_NATS} nat floor{size}.",
            "Do not build evolutionary search. Widen the search over the genes that "
            "actually run - depth, width, expert count, active experts, expert width - "
            "then revisit model scale. Module type is not available at this scale: "
            "measured at 100M/context 1024, GDN costs 4-7x attention and exceeds an "
            "8 GB card at 6 of 8 layers.",
        )
    if p_value is None:
        return (
            f"PROVISIONAL - spread {spread:.4f} nats but no test was computable{size}.",
            "Collect at least two seeds per architecture so a test exists.",
        )
    if p_value < 0.05:
        # A spread is a statement about one training length. This project measured
        # it falling 3.7x between 400 and 1200 steps on the v1 families, so an
        # established effect at a single length is a licence to search only for as
        # long as the effect survives training. Say so, unless the design already
        # ran several lengths and the decay is therefore already in the table.
        measured = [
            key
            for key in (outcome.get("spread_by_length") or {})
            if key != "decay"
        ]
        horizon = (
            " This design measured one training length. The spread does fall with "
            "training - 2.22x over 400 -> 1200 steps on a matched design, 1.65x per "
            "doubling - but seed noise falls with it, so the ordering and the "
            "signal-to-noise did not move over that range. Quote any effect size "
            "with its length, and confirm the ordering holds at the horizon the "
            "search actually trains for."
            if len(measured) < 2
            else ""
        )
        # When the design crossed two genes, say which one the effect belongs to
        # rather than only that it exists: a search needs to know where to spend
        # its budget, and that is the question a matched design cannot answer.
        crossed = outcome.get("crossed_contrast") or {}
        attribution = ""
        if crossed.get("available"):
            resolved = [
                (gene, row)
                for gene, row in crossed["main"].items()
                if row.get("p") is not None and row["p"] < 0.05
            ]
            if resolved:
                gene, row = max(resolved, key=lambda item: abs(item[1]["nats"]))
                smaller = sorted(other for other, _ in resolved if other != gene)
                attribution = (
                    f" The crossed factorial attributes it to `{gene}` "
                    f"({row['nats']:+.4f} nats, p={row['p']:.4g})"
                    + (
                        f", with `{', '.join(smaller)}` real but smaller"
                        if smaller
                        else ""
                    )
                    + "; spend the search budget there."
                )
        # 4 significant digits, not 4 decimal places: a Monte-Carlo p of 5e-06
        # formats as "p=0.0000" at .4f, which reads as exact zero and claims more
        # than a sampled null distribution can support. When the estimate lands at
        # the procedure's own resolution it is a bound, so it is labelled as one.
        resolution = blocked.get("p_resolution")
        floored = resolution is not None and p_value <= resolution
        shown = (
            f"p<={resolution:.2g}" if floored else f"p={p_value:.4g}"
        )
        return (
            f"ARCHITECTURE EFFECT ESTABLISHED - spread {spread:.4f} nats, {test_name} "
            f"{shown}{size}.",
            "Proceed to evolutionary architecture search (arch-evo-100m-v1), and "
            "confirm the finalists on seeds never used in screening first so the "
            "selection coupling is removed."
            + attribution
            + horizon,
        )
    return (
        f"NOT ESTABLISHED - spread {spread:.4f} nats is above the materiality floor "
        f"but {test_name} gives p={p_value:.4g}, so it is not separable from "
        f"initialization noise at {blocked.get('blocks', 'these')} seeds{size}. "
        "This is not evidence that architecture does not matter; it is evidence "
        "that this experiment cannot yet tell.",
        "Do not build the full evolutionary search. Resolve the effect size first: "
        "re-run the finalists on seeds never used in screening, which removes the "
        "selection coupling and raises power, and test a matched-budget structural "
        "design so any spread is attributable to one axis. Note that widening with "
        "module-type genes is not an option here - measured at 100M/context 1024, "
        "GDN costs 4-7x attention and exceeds an 8 GB card at 6 of 8 layers.",
    )


#: Defaults of the crossed confirmation arm's pre-registered rule, declared in
#: ``artifacts/arch-zoo-100m-v3-crossed-confirm/NOTES.md`` before its numbers
#: existed. Held as data rather than prose because a rule whose thresholds live
#: only in a document gets reapplied by eye once the numbers are in, and the eye
#: that reads a p-value already knows which branch it wants.
REPLICATION_RULE: dict[str, float] = {
    "main_p": 0.01,
    "ratio": 1.5,
    "interaction_p": 0.05,
}

#: What each branch means for the search, quoted from the pre-registration so the
#: consequence is fixed before the result rather than chosen after it.
REPLICATION_READINGS: dict[str, str] = {
    "replicates-and-resolves": (
        "The attribution replicates on seeds never used to design or screen "
        "anything, and the depth penalty is level-dependent. A search must mutate "
        "depth and width *jointly*: a mutation that is good at one width is not "
        "good at the other, so the space is not separable in these two genes and "
        "the search has to cover their joint space."
    ),
    "replicates-no-interaction": (
        "The attribution replicates and the interaction does not: the two genes "
        "are additively separable at this scale. This is the better outcome for a "
        "search than an interaction, because depth and width can then be "
        "optimised one at a time instead of over their joint space - a large "
        "reduction in the search problem."
    ),
    "fails-to-replicate": (
        "The attribution does not survive fresh seeds, so it is a property of the "
        "original seeds rather than of the design. \u00a72c's attribution is a "
        "correction to be written, not a finding to build a search on."
    ),
    "inconclusive": (
        "The main effects replicate but the interaction is still open. Before "
        "buying more seeds, note whether the threshold was reachable at all: if "
        "the seed-blocked test's floor is above the declared 0.05, the arm could "
        "not have resolved it however consistent the seed contrasts were, and "
        "that is a statement about the design's size rather than about the world."
    ),
    "unavailable": (
        "The rule needs a crossed factorial, and this outcome has none - either "
        "the design is not crossed or the crossed section could not be computed, "
        "which is reported above with its reason."
    ),
}


def replication_decision(
    outcome: dict,
    *,
    main_p: float = REPLICATION_RULE["main_p"],
    ratio: float = REPLICATION_RULE["ratio"],
    interaction_p: float = REPLICATION_RULE["interaction_p"],
) -> dict:
    """Apply the confirmation arm's pre-registered rule to a pooled outcome.

    The rule, as declared before the arm's training data existed:

    | outcome | condition |
    |---|---|
    | attribution replicates, interaction resolves | both main effects ``p < 0.01`` with the *same signs* as the original attribution and ``|hidden| >= 1.5 x |depth|``; seed-blocked interaction ``p < 0.05`` with every seed contrast the same sign |
    | attribution replicates, interaction absent | main effects as above; the seed contrasts have mixed signs |
    | attribution fails to replicate | either main effect non-significant, or a sign flips, or ``|hidden| < 1.5 x |depth|`` |
    | inconclusive | main effects replicate, the interaction is non-significant *and* not sign-consistent, while the contrasts straddle zero without agreeing |

    The signs are not a free parameter: the original attribution measured
    ``hidden_size`` negative and ``depth`` positive, and the branch about sign
    flipping is what makes this a *replication* test rather than a re-run of the
    same regression.

    Two things are returned beside the branch and matter more than it: the
    **evidence for each condition** (so the branch can be checked rather than
    trusted) and the **floor** of the seed-blocked test. A p that cannot go below
    the declared threshold makes the inconclusive branch a statement about the
    arm's size, and the reading says so instead of leaving it looking like an
    absent interaction.
    """
    crossed = outcome.get("crossed_contrast") or {}
    main = crossed.get("main") or {}
    hidden, depth = main.get("hidden_size"), main.get("depth")
    if not crossed.get("available") or hidden is None or depth is None:
        return {
            "available": False,
            "branch": "unavailable",
            "label": "UNAVAILABLE - the rule needs both declared factors",
            "reading": REPLICATION_READINGS["unavailable"],
            "conditions": [],
            "reason": crossed.get("reason", "no crossed contrast in this outcome"),
        }

    def significant(row: dict) -> bool:
        return row.get("p") is not None and row["p"] < main_p

    hidden_ok = significant(hidden) and hidden["nats"] < 0
    depth_ok = significant(depth) and depth["nats"] > 0
    magnitude_ok = abs(hidden["nats"]) >= ratio * abs(depth["nats"])
    replicated = hidden_ok and depth_ok and magnitude_ok

    # The interaction is reported on each main effect and is the same number from
    # either side of a 2x2, so it is read once rather than averaged.
    anchor = crossed["main"][crossed["factor_a"]]
    contrasts = list(anchor.get("interaction_seed_contrasts") or [])
    positives = [value for value in contrasts if value > 0]
    negatives = [value for value in contrasts if value < 0]
    consistent = bool(positives or negatives) and not (positives and negatives)
    mixed = bool(positives and negatives)
    p_interaction = anchor.get("interaction_p_seed_blocked")
    floor = anchor.get("interaction_p_seed_blocked_floor")
    resolved = (
        p_interaction is not None and p_interaction < interaction_p and consistent
    )

    if not replicated:
        branch = "fails-to-replicate"
    elif resolved:
        branch = "replicates-and-resolves"
    elif mixed:
        # The pre-registration's *absent* branch, and its condition is exactly this:
        # the seed contrasts disagree, so no level dependence is being measured.
        branch = "replicates-no-interaction"
    else:
        # Signs agree (or there is no contrast to disagree with) and the test still
        # did not clear the declared threshold. The pre-registered table does not
        # cover this cell - *absent* requires mixed signs, and the inconclusive row
        # assumes the contrasts disagree - so it is read as inconclusive rather than
        # as an absent interaction, because a consistent interaction that the test
        # could not resolve is precisely a result the arm's size failed to reach.
        # `note` below says which of the two reasons applies.
        branch = "inconclusive"

    conditions = [
        {
            "condition": "hidden_size significant, negative",
            "value": f"{hidden['nats']:+.4f} nats, p={hidden['p']:.4g}",
            "met": hidden_ok,
        },
        {
            "condition": "depth significant, positive (shallow wins)",
            "value": f"{depth['nats']:+.4f} nats, p={depth['p']:.4g}",
            "met": depth_ok,
        },
        {
            # Escaped pipes: this label is rendered into a Markdown table cell, and
            # a bare `|` there splits the row into extra columns.
            "condition": f"\\|hidden\\| >= {ratio:g} x \\|depth\\|",
            "value": f"{abs(hidden['nats']):.4f} vs {abs(depth['nats']):.4f} nats",
            "met": magnitude_ok,
        },
        {
            "condition": f"seed-blocked interaction p < {interaction_p:g}",
            "value": (
                "not computable"
                if p_interaction is None
                else f"p={p_interaction:.4g}"
                + ("" if floor is None else f" (floor {floor:.4g})")
            ),
            "met": bool(p_interaction is not None and p_interaction < interaction_p),
        },
        {
            "condition": "every seed contrast the same sign",
            "value": (
                "no seed contrasts recorded"
                if not contrasts
                else ", ".join(f"{value:+.4f}" for value in contrasts)
            ),
            "met": consistent,
        },
    ]

    label = {
        "replicates-and-resolves": "REPLICATES, INTERACTION RESOLVES",
        "replicates-no-interaction": "REPLICATES, INTERACTION ABSENT",
        "fails-to-replicate": "ATTRIBUTION FAILS TO REPLICATE",
        "inconclusive": "INCONCLUSIVE - the interaction is still open",
    }[branch]

    note = None
    if branch == "inconclusive":
        if floor is not None and floor >= interaction_p:
            # The load-bearing detail: a consistent interaction that cannot clear the
            # threshold because the test's floor is above it says nothing about the
            # interaction, only about how many seeds were bought.
            note = (
                f"The seed-blocked test's floor at {len(contrasts)} seeds is "
                f"{floor:.4g}, which is above the declared {interaction_p:g} - so this "
                "arm could not have resolved the interaction however consistent the "
                "seed contrasts were. More seeds raise the floor's denominator; "
                "nothing else does."
            )
        elif p_interaction is None or not contrasts:
            note = (
                "The seed-blocked sign test could not be computed from these "
                "contrasts, so neither the resolved nor the absent branch is "
                "available and the question stays open."
            )
        else:
            note = (
                f"A consistent set of contrasts did not clear the declared "
                f"{interaction_p:g} (p={p_interaction:.4g}, floor {floor:.4g}). The "
                "interaction is therefore neither resolved nor shown absent: the "
                "pre-registered *absent* branch needs the signs to disagree, and "
                "these agree."
            )

    return {
        "available": True,
        "branch": branch,
        "label": label,
        "reading": REPLICATION_READINGS[branch],
        "note": note,
        "thresholds": {
            "main_p": main_p,
            "ratio": ratio,
            "interaction_p": interaction_p,
        },
        "conditions": conditions,
        "seeds": crossed.get("seeds"),
        "seeds_dropped": crossed.get("seeds_dropped") or [],
        "interaction": {
            "nats": anchor.get("interaction_nats"),
            "p_seed_blocked": p_interaction,
            "floor": floor,
            "contrasts": contrasts,
        },
        "main": {
            "hidden_size": {"nats": hidden["nats"], "p": hidden["p"]},
            "depth": {"nats": depth["nats"], "p": depth["p"]},
        },
    }


#: Defaults of the head-paired confirmation arm's pre-registered rule, declared in
#: ``artifacts/arch-zoo-100m-v4-head-paired-confirm/NOTES.md`` before its numbers
#: existed. Held as data for the same reason ``REPLICATION_RULE`` is: a threshold
#: that lives only in a document gets reapplied by eye once the numbers arrive, and
#: the eye reading a p-value already knows which branch it wants.
HEAD_PAIRING_RULE: dict[str, float] = {
    "main_p": 0.01,
    "interaction_p": 0.05,
}

#: What each branch means for the search, quoted from the pre-registration so the
#: consequence is fixed before the result rather than chosen after it.
HEAD_PAIRING_READINGS: dict[str, str] = {
    "heads-free-at-wide": (
        "The level dependence is real: more attention heads help at wide "
        "residual-stream width and do not at narrow. A search over this family must "
        "mutate `attention_heads` *jointly* with `hidden_size` - the free lever is "
        "only free at wide widths, so \"add heads\" is a rule with a precondition "
        "rather than a global one."
    ),
    "heads-not-at-wide": (
        "The interaction resolves and does not point at a free head gain at wide "
        "width: the wide level's paired head effect is not negative, or the "
        "interaction runs the other way. Width stays the lever, the head axis is "
        "informative only as a warning against a global \"more heads\" rule, and no "
        "head budget is warranted."
    ),
    "head-uniform": (
        "There is no level dependence, but the head effect on the pooled design is "
        "real and the same at both widths: heads are a lever, just not a "
        "level-dependent one. The parent arm's three-seed exclusion of a head "
        "effect is corrected, and the search should add heads *globally* rather "
        "than jointly with width - the opposite mutation shape from the "
        "level-dependent branch."
    ),
    "head-axis-closed": (
        "Head count is worth about nothing at both levels, so the three-seed sign "
        "agreement was seed luck. This is the *cheaper* world for a search: "
        "`attention_heads` need not be mutated jointly with `hidden_size`, which "
        "returns that axis to a free knob rather than a search dimension."
    ),
    "width-fails": (
        "The phase's width attribution does not survive on the design with the "
        "tightest matching in it, so \u00a72c's headline is a correction to be written "
        "rather than a finding to build a search on. Investigate before trusting "
        "either arm."
    ),
    "open-floor-limited": (
        "The seed contrasts agree but the seed-blocked test's floor is above the "
        "declared threshold, so this arm's *size* stopped it: the interaction is "
        "neither resolved nor shown absent. More seeds raise the floor's "
        "denominator; nothing else does."
    ),
    "unavailable": (
        "The rule needs a paired head design crossed with a two-level second gene, "
        "and this outcome has none - either the design is not paired or the crossed "
        "section could not be computed, which is reported with its reason."
    ),
}


def head_pairing_decision(
    outcome: dict,
    *,
    main_p: float = HEAD_PAIRING_RULE["main_p"],
    interaction_p: float = HEAD_PAIRING_RULE["interaction_p"],
) -> dict:
    """Apply the head-paired confirmation arm's pre-registered rule.

    The rule, as declared before the arm's training data existed:

    | outcome | condition |
    |---|---|
    | head effect is real at wide hidden | seed-blocked interaction ``p < 0.05`` with every seed contrast the same sign, interaction ``> 0``, and the wide level's paired head effect ``< 0`` |
    | interaction resolves, against heads | the same, but the wide level's paired head effect ``>= 0`` |
    | head effect is real but uniform | the interaction is not resolved *and* the pooled paired head contrast is significant and sign-consistent |
    | interaction absent | the seed contrasts have mixed signs, and the pooled contrast is not significant |
    | the width attribution fails to replicate | the pooled ``hidden_size`` main effect flips sign or is non-significant at the declared 0.01 |

    The uniform branch is not in the arm's pre-registered table as first written, and
    the test suite is what found the gap rather than the arm: a head effect that is
    real and identical at both widths has no interaction to resolve *and* is not
    "worth nothing at both levels", so the absent branch's reading would have been
    false about it. The threshold it is decided by is the same declared
    ``interaction_p`` applied to the pooled contrast, so no new number entered the
    rule after the data - and the arm's own notes carry the row, added while its
    runs were still training.

    Every condition is a sign or a threshold, unlike the arm's parent, whose own
    five-outcome table had to declare two of its clauses person-judgements. That is
    what makes this rule mechanical: the reading is chosen from the transcription in
    ``HEAD_PAIRING_RULE`` rather than by eye once the numbers land.

    The order of the tests is the pre-registration's, and it matters: the width
    check gates everything, because if the phase's one established attribution does
    not hold on its tightest-matched design then neither contrast in this rule is
    worth reading.
    """
    crossed = outcome.get("crossed_contrast") or {}
    main = crossed.get("main") or {}
    hidden = main.get("hidden_size")
    heads = main.get("attention_heads")
    pooled = (outcome.get("paired_contrasts") or {}).get("attention_heads") or {}
    if (
        not crossed.get("available")
        or hidden is None
        or heads is None
        or not pooled.get("available")
    ):
        return {
            "available": False,
            "branch": "unavailable",
            "label": "UNAVAILABLE - THE RULE NEEDS A HEAD-PAIRED CROSSED DESIGN",
            "reading": HEAD_PAIRING_READINGS["unavailable"],
            "conditions": [],
            "reason": (
                crossed.get("reason")
                or "no crossed contrast on hidden_size x attention_heads"
            ),
        }

    width_ok = hidden.get("p") is not None and hidden["p"] < main_p and hidden["nats"] < 0

    contrasts = list(heads.get("interaction_seed_contrasts") or [])
    positives = [value for value in contrasts if value > 0]
    negatives = [value for value in contrasts if value < 0]
    consistent = bool(positives or negatives) and not (positives and negatives)
    p_interaction = heads.get("interaction_p_seed_blocked")
    floor = heads.get("interaction_p_seed_blocked_floor")
    resolved = (
        p_interaction is not None and p_interaction < interaction_p and consistent
    )
    # A consistent sign set that the test could not resolve, with a floor above the
    # threshold, is a statement about the arm's size rather than about the head
    # effect - the reading error the horizon arm already taught this phase.
    floor_limited = (
        consistent
        and not resolved
        and floor is not None
        and floor >= interaction_p
    )

    # The wide level is the top level of the partner factor, detected rather than
    # hardcoded, and the split's number is the one the rule reads: the pooled
    # contrast is an average over both levels and cannot decide a direction.
    splits = (outcome.get("paired_level_splits") or {}).get("attention_heads") or {}
    partner = crossed.get("factor_a")
    if partner == "attention_heads":
        partner = crossed.get("factor_b")
    split = splits.get(partner) or (next(iter(splits.values())) if len(splits) == 1 else {})
    levels = sorted(split.get("levels") or {}, key=lambda key: float(key))
    wide_level = levels[-1] if levels else None
    wide = (split.get("levels") or {}).get(wide_level) if wide_level else None
    wide_nats = None if wide is None else wide["mean_difference"]
    wide_ok = wide_nats is not None and wide_nats < 0

    interaction_nats = heads.get("interaction_nats")
    pooled_p = pooled.get("p")
    # A real head effect with no level dependence is neither of the pre-registered
    # branches about the *interaction*, so it is its own: the pooled test decides it,
    # at the threshold already declared for this rule.
    pooled_significant = bool(
        pooled_p is not None
        and pooled_p < interaction_p
        and pooled.get("sign_consistent")
    )
    if not width_ok:
        branch = "width-fails"
    elif resolved and interaction_nats is not None and interaction_nats > 0 and wide_ok:
        branch = "heads-free-at-wide"
    elif resolved:
        branch = "heads-not-at-wide"
    elif floor_limited:
        branch = "open-floor-limited"
    elif pooled_significant:
        branch = "head-uniform"
    else:
        branch = "head-axis-closed"

    level_text = (
        "no level split available"
        if wide is None
        else (
            f"{partner}={float(wide_level):g}: {wide_nats:+.4f} nats, "
            f"p={wide['p']:.4g} (floor {wide['min_achievable_p']:.4g}, "
            f"{wide['pairs']} units)"
        )
    )
    conditions = [
        {
            "condition": f"hidden_size significant, negative (p < {main_p:g})",
            "value": f"{hidden['nats']:+.4f} nats, p={hidden['p']:.4g}",
            "met": bool(width_ok),
        },
        {
            "condition": f"seed-blocked interaction p < {interaction_p:g}",
            "value": (
                "not computable"
                if p_interaction is None
                else f"p={p_interaction:.4g}"
                + ("" if floor is None else f" (floor {floor:.4g})")
            ),
            "met": bool(p_interaction is not None and p_interaction < interaction_p),
        },
        {
            "condition": "every seed contrast the same sign",
            "value": (
                "no seed contrasts recorded"
                if not contrasts
                else ", ".join(f"{value:+.4f}" for value in contrasts)
            ),
            "met": consistent,
        },
        {
            "condition": "interaction nats > 0 (heads help more at wide width)",
            "value": (
                "not computable"
                if interaction_nats is None
                else f"{interaction_nats:+.4f} nats"
            ),
            "met": bool(interaction_nats is not None and interaction_nats > 0),
        },
        {
            "condition": "the wide level's paired head effect < 0",
            "value": level_text,
            "met": bool(wide_ok),
        },
    ]

    label = {
        "heads-free-at-wide": "HEAD EFFECT IS REAL AT WIDE WIDTH",
        "heads-not-at-wide": "INTERACTION RESOLVES, AGAINST HEADS AT WIDE WIDTH",
        "head-uniform": "HEAD EFFECT IS REAL AND UNIFORM - NO LEVEL DEPENDENCE",
        "head-axis-closed": (
            "INTERACTION ABSENT - THE THREE-SEED AGREEMENT WAS SEED LUCK"
        ),
        "width-fails": "\u00a72C'S WIDTH ATTRIBUTION FAILS TO REPLICATE",
        "open-floor-limited": (
            "OPEN - THE SEED-BLOCKED FLOOR IS ABOVE THE DECLARED THRESHOLD"
        ),
    }[branch]

    note = None
    if branch == "width-fails":
        note = (
            "The rule is read in this order deliberately: the width check gates the "
            "rest, so a head reading is not reported from a design whose width "
            "contrast has already failed to reproduce."
        )
    elif branch == "open-floor-limited":
        note = (
            f"The seed-blocked test's floor at {len(contrasts)} seeds is {floor:.4g}, "
            f"above the declared {interaction_p:g} - so this arm could not have "
            "resolved the interaction however consistent the seed contrasts were."
        )
    elif branch == "head-axis-closed" and float(p_interaction or 0) < interaction_p:
        # Defensive: reachable only if the sign-consistency test and the p-value ever
        # disagree, which the floor arithmetic makes impossible on a complete sweep.
        note = (
            "The test cleared the threshold while the contrasts disagreed in sign, "
            "so no single sign is being measured; treat the branch as open rather "
            "than as an absent interaction."
        )
    if pooled_significant:
        note = (note + " " if note else "") + (
            "The pooled head contrast is significant on its own, which the parent arm's "
            "three-seed result excluded. "
            + (
                "With no level dependence behind it, the effect is uniform across "
                "widths, so the search should add heads globally rather than jointly."
                if branch == "head-uniform"
                else "That is a correction to that exclusion rather than a fifth branch "
                "of this rule: read the level split above to see which level it comes "
                "from."
            )
        )
    elif pooled_p is not None and pooled_p < interaction_p:
        note = (note + " " if note else "") + (
            "The pooled head contrast clears the threshold on a sign-inconsistent set "
            "of pairs, so it is a systematic average of disagreeing units rather than "
            "a uniform head effect - the level split above is what says which levels "
            "produce it, and it is the reason that split is reported rather than "
            "inferred from this average."
        )

    return {
        "available": True,
        "branch": branch,
        "label": label,
        "reading": HEAD_PAIRING_READINGS[branch],
        "note": note,
        "thresholds": {"main_p": main_p, "interaction_p": interaction_p},
        "conditions": conditions,
        "seeds": crossed.get("seeds"),
        "seeds_dropped": crossed.get("seeds_dropped") or [],
        "interaction": {
            "nats": interaction_nats,
            "p_seed_blocked": p_interaction,
            "floor": floor,
            "contrasts": contrasts,
        },
        "main": {
            "hidden_size": {"nats": hidden["nats"], "p": hidden["p"]},
            "attention_heads": {"nats": heads["nats"], "p": heads["p"]},
        },
        "pooled_head_contrast": {
            "nats": pooled.get("mean_difference"),
            "p": pooled_p,
            "floor": pooled.get("min_achievable_p"),
            "units": pooled.get("pairs"),
            "sign_consistent": pooled.get("sign_consistent"),
        },
        "level_split": {
            "partner": partner,
            "wide_level": wide_level,
            "wide_nats": wide_nats,
            "sign_agreement": split.get("sign_agreement"),
        },
    }


def write_decision(path: Path, outcome: dict) -> None:
    """Record whether an architectural signal exists, and the resulting next step."""
    headline, action = verdict(outcome)

    lines = [
        "# Architecture Zoo Decision",
        "",
        f"**{headline}**",
        "",
        f"Next step: {action}",
        "",
    ]
    if outcome.get("ranked"):
        lines += [
            f"## Ranked at {outcome['steps']} steps (mean held-out validation loss)",
            "",
            "| Rank | Architecture | Mean val loss | Runs |",
            "|---|---|---|---|",
        ]
        for index, row in enumerate(outcome["ranked"], start=1):
            lines.append(f"| {index} | `{row['name']}` | {row['mean_val_loss']:.4f} | {row['runs']} |")

        blocked = outcome.get("blocked_test") or {}
        anova = outcome.get("anova") or {}
        units = outcome.get("compute_units") or {}
        lines += [
            "",
            "## Evidence",
            "",
            f"- Architecture spread (best to worst): {outcome['architecture_spread']:.4f} nats",
            f"- Mean within-architecture seed std: {outcome['within_seed_std']:.4f} nats "
            "(descriptive; a range over architectures is not a test statistic)",
            f"- Spread in seed-noise units: "
            + ("undefined (single-seed)" if outcome.get("sigmas") is None else f"{outcome['sigmas']:.2f}"),
        ]
        if blocked.get("available"):
            resolution = blocked.get("p_resolution")
            lines.append(
                f"- Seed-matched test ({blocked['test']}): p={blocked['p']:.4g} over "
                f"{blocked['blocks']} blocks x {blocked['architectures']} architectures; "
                f"design floor {blocked['min_achievable_p']:.4g}"
                + (f", procedure resolution {resolution:.4g}" if resolution else "")
            )
            # The block count above is the seeds that survived the rectangle, so a
            # dropped seed has to be named or a 2-block result reads as the design.
            if blocked.get("unfinished_seeds"):
                lines.append(
                    f"- Seeds excluded from that test: {blocked['unfinished_seeds']} "
                    "(fewer runs written at this length than the fullest seed, i.e. "
                    "still training). The block count above is the seeds that "
                    "finished; re-running the analysis completes it."
                )
            if blocked.get("architectures_dropped"):
                lines.append(
                    f"- Architectures excluded from that test: "
                    f"{blocked['architectures_dropped']} (not measured on every "
                    "seed, so the seed-matched test cannot include them)."
                )
            if blocked.get("unbalanced_seeds"):
                lines.append(
                    f"- **Seeds excluded that had written runs for every "
                    f"candidate: {blocked['unbalanced_seeds']}** - those rows exist "
                    "but do not intersect, which is a zoo defect rather than an "
                    "unfinished sweep. Treat the test above as provisional."
                )
        if anova.get("available"):
            lines.append(
                f"- Unpaired {anova['test']}: F={anova['f']:.2f} on "
                f"({anova['df_between']}, {anova['df_within']}) df, p={anova['p']:.4f}"
            )
        if units.get("available"):
            lines.append(
                f"- Effect size: {units['spread']:.4f} nats = {units['equivalent_steps']:.0f} "
                f"training steps ({units['fraction_of_budget'] * 100:.1f}% of budget), "
                f"fitted over {units['fitted_between']}"
            )
        if outcome.get("selection_caveat"):
            lines += ["", f"Caveat: {outcome['selection_caveat']}"]
        lines.append("")

    by_length = outcome.get("spread_by_length") or {}
    if by_length:
        lines += [
            "## Spread against training length (common architecture set)",
            "",
            "| Steps | Architectures | Spread | Seed std | Spread in seed-noise units | Best |",
            "|---|---|---|---|---|---|",
        ]
        for length, row in by_length.items():
            if length == "decay":
                continue
            noise = row.get("within_seed_std")
            sigmas = row.get("sigmas")
            lines.append(
                f"| {length} | {row['architectures']} | {row['spread']:.4f} | "
                + ("n/a" if noise is None else f"{noise:.4f}")
                + " | "
                + ("n/a" if sigmas is None else f"{sigmas:.2f}")
                + f" | `{row['best']}` |"
            )
        lines.append("")
        decay = by_length.get("decay") or {}
        if decay:
            lines += [
                f"Spread falls {decay['spread_shrink']:.2f}x over "
                f"{decay['from_steps']} -> {decay['to_steps']} steps "
                f"({decay['per_doubling']:.2f}x per doubling of training). "
                + (
                    f"Seed noise falls with it ({decay['sigmas_at_first']:.2f} -> "
                    f"{decay['sigmas_at_last']:.2f} in seed-noise units), so the effect "
                    "does not wash out relative to noise at these lengths."
                    if None not in (decay["sigmas_at_first"], decay["sigmas_at_last"])
                    and decay["sigmas_at_last"] >= 0.5 * decay["sigmas_at_first"]
                    else "Seed noise shrinks more slowly, so the effect weakens relative to "
                    "noise as training lengthens."
                ),
                "",
                "The same architecture is best at every length: "
                f"**{'yes' if decay['best_at_every_length'] else 'NO'}**.",
                "",
            ]

    # Spread shrinking says the *size* of the effect falls. It does not say
    # whether the reason for it is still readable at the shorter length, which is
    # what decides how much of a screen has to be paid for.
    signal = outcome.get("structural_signal_by_length") or {}
    fitted = {
        k: v
        for k, v in signal.items()
        if k not in ("agreement", "model") and v.get("available")
    }
    pre_specified = (signal.get("model") or {}).get("pre_specified")
    if len(fitted) >= 2:
        lines += [
            "## Structural signal against training length (common architecture set)",
            "",
            (
                "The same joint fit at every length, on a **pre-specified** model "
                f"({', '.join('`' + g + '`' for g in pre_specified)}). The model is "
                "fixed across lengths, so agreement here is a statement about the "
                "model rather than about which predictors each length happened to "
                "pick."
                if pre_specified
                else "The same joint fit at each length. If the model is already "
                "significant at the shortest length, a search can screen there and "
                "confirm finalists at full length."
            ),
            "",
            "| Steps | Architectures | R2 | Model p | "
            + ("First gene" if pre_specified else "Leading gene")
            + " | beta | p |",
            "|---|---|---|---|---|---|---|",
        ]
        for length, row in fitted.items():
            lead = row["genes"][0] if row["genes"] else None
            lines.append(
                f"| {length} | {row['architectures']} | {row['r_squared']:.3f} | "
                f"{row['p_model']:.4g} | "
                + ("n/a | n/a | n/a |" if lead is None else
                   f"`{lead['gene']}` | {lead['beta']:+.3f} | {lead['p']:.4g} |")
            )
        agreement = signal.get("agreement") or {}
        lines += [""]
        if agreement:
            if pre_specified:
                lines += [
                    "The model is the same at every length by construction, so the "
                    "question this table answers is whether *it* holds at each "
                    "one — and the coefficient p-values below the model do not "
                    "depend on a per-length selection at all.",
                    "",
                ]
            else:
                lines += [
                    "Same leading gene at every fitted length: "
                    f"**{'yes' if agreement['same_leading_gene'] else 'NO'}**"
                    + (
                        f" (`{agreement['leading_genes'][0]}`), with the same sign: "
                        f"**{'yes' if agreement['same_leading_sign'] else 'NO'}**."
                        if agreement["leading_genes"]
                        else "."
                    ),
                    "",
                    "Caveat: the leading gene is *selected* from the spread axes at "
                    "each length, so its own p-value is uncorrected for that "
                    "selection and is not comparable to the designed factorial's "
                    "exact test. It is a consistency check, not a replacement for "
                    "it. Pass `--signal-genes` to hold the model fixed instead.",
                    "",
                ]

    if outcome.get("rank_stability"):
        lines += [
            "## Screening validity: does a short run rank like a long one?",
            "",
            "| From | To | n | Spearman | Best at short | Best at long | Winner kept |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in outcome["rank_stability"]:
            rho = "n/a" if row["spearman"] is None else f"{row['spearman']:.2f}"
            lines.append(
                f"| {row['from_steps']} | {row['to_steps']} | {row['architectures']} | {rho} | "
                f"`{row['best_at_from']}` | `{row['best_at_to']}` | "
                f"{'yes' if row['winner_preserved'] else 'NO'} |"
            )
        lines.append("")

    association = outcome.get("axis_association") or {}
    if association.get("available"):
        lines += [
            "## Which axis moves with loss (exploratory)",
            "",
            "| Axis | Spearman | p | Range | Ratio |",
            "|---|---|---|---|---|",
        ]
        for row in association["axes"]:
            if row.get("held"):
                note = "not recorded" if row.get("unrecorded") else "held constant"
                lines.append(f"| {row['axis']} | — | — | {note} | — |")
                continue
            ratio = "n/a" if row["spread_ratio"] is None else f"{row['spread_ratio']:.2f}x"
            # 4 significant digits: a ratio like 0.597..0.623 is the whole point of
            # the row, and rounding it to "0 .. 1" would misstate the design.
            mark = "" if row.get("p_exact") else "~"
            lines.append(
                f"| {row['axis']} | {row['spearman']:+.3f} | {mark}{row['p']:.4f} | "
                f"{_compact(row['min'])} .. {_compact(row['max'])} | {ratio} |"
            )
        lines += ["", f"*{association['multiple_comparisons']}*", ""]
        collinearity = association.get("collinearity") or {}
        if collinearity.get("available"):
            lines += ["**Confounding structure.** " + collinearity["caveat"], ""]

    # The per-axis table above tests each gene alone, which overstates a gene's
    # standing whenever its partners vary too. This is the same set with all of
    # them in the model at once.
    fit = outcome.get("joint_structural_fit") or {}
    if fit.get("available"):
        lines += [
            "## Structural genes fitted together",
            "",
            f"Standardised least squares over {fit['architectures']} architectures: "
            f"R2 = {fit['r_squared']:.3f}, model p = {fit['p_model']:.4g} against a "
            f"permutation null ({fit['samples']} draws). Coefficients are nats per "
            "standard deviation of that gene, holding the others fixed.",
            "",
            "| Gene | beta | p | Range |",
            "|---|---|---|---|",
        ]
        for row in fit["genes"]:
            lines.append(
                f"| {row['gene']} | {row['beta']:+.3f} | {row['p']:.4g} | "
                f"{row['min']:g} .. {row['max']:g} |"
            )
        for row in fit.get("dropped", []):
            lines.append(
                f"| {row['gene']} | — | — | identical to `{row['duplicate_of']}` "
                f"(rho={row['spearman']:+.4f}), so it carries no separate information |"
            )
        lines += [
            "",
            f"*{fit['caveat']}*",
            "",
            "A gene can be significant alone and not here, or the reverse. Neither "
            "table is a lever ranking: the only design that ranks levers is one "
            "that varies them independently by construction.",
            "",
        ]

    paired = outcome.get("paired_contrasts") or {}
    for gene, contrast in sorted(paired.items()):
        p_text = "n/a" if contrast["p"] is None else f"{contrast['p']:.4g}"
        lines += [
            "## Paired contrast: " + f"`{gene}` varied with everything else held fixed",
            "",
            f"{contrast['pairs']} pairs at levels {contrast['levels']}, so the "
            "within-pair difference is a contrast between candidates that differ in "
            "nothing else. That is a stronger guarantee than a matched band: the "
            "pair is its own control.",
            "",
            "| Quantity | Value |",
            "|---|---|",
            f"| Mean within-pair difference | {contrast['mean_difference']:+.4f} nats |",
            f"| Exact sign-permutation p | {p_text} ({contrast['method']}, "
            f"{contrast['assignments']} sign vectors) |",
            f"| Procedure floor | {contrast['min_achievable_p']:.4g} |",
            f"| Paired t (reference) | "
            + ("n/a" if contrast.get("p_paired_t") is None else f"{contrast['p_paired_t']:.4g}")
            + " |",
            f"| Sign consistency | {contrast['negative']} negative, "
            f"{contrast['positive']} positive |",
            "",
            "Per-pair differences: "
            + ", ".join(f"{d:+.4f}" for d in contrast["differences"][:12])
            + (
                f" (+{len(contrast['differences']) - 12} more)"
                if len(contrast["differences"]) > 12
                else ""
            ),
            "",
            "A significant contrast with an *inconsistent* sign across pairs means "
            "one or two pairs carry the result, which the per-pair list above shows "
            "rather than hides.",
            "",
        ]
        for row in contrast.get("seeds_partial") or []:
            lines += [
                f"**Seed {row['seed']} contributes {row['pairs']} of {row['of']} "
                "pairs**, with no half-written pair among them: those runs have not "
                "been written yet. The pairs that exist are complete, so the "
                "contrast is valid over them, and the unit count - and therefore "
                "the floor printed beside the p - follows the shorter seed.",
                "",
            ]
        dropped_pairs = contrast.get("dropped_groups") or []
        if dropped_pairs:
            lines += [
                f"**Groups dropped: {len(dropped_pairs)}**, all on seed(s) "
                f"{contrast.get('seeds_dropped')}, rows the sweep has not written "
                "yet. The contrast above is over the pairs that are complete, so "
                "it is the answer with less resolution rather than a different "
                "answer - and the floor printed beside it already reflects the "
                "smaller unit count. Re-running the analysis when the sweep "
                "finishes tightens it.",
                "",
            ]

    # A pooled paired contrast is an average over the levels of everything it holds
    # fixed, so the split is what says whether the pooled number describes either
    # level. Written for every paired gene with a two-level partner, because that is
    # where a level difference is an interaction rather than a slice.
    for gene, splits in sorted((outcome.get("paired_level_splits") or {}).items()):
        for partner, split in sorted(splits.items()):
            lines += [
                f"## Paired level split: `{gene}` at each level of `{partner}`",
                "",
                "The pooled paired contrast above is an average over these levels, so "
                "it can carry one level or cancel two. "
                + split["reading"].capitalize()
                + ".",
                "",
                "| Level | Mean within-pair difference | p | Units | Floor | Signs |",
                "|---|---|---|---|---|---|",
            ]
            for level, contrast in sorted(
                split["levels"].items(), key=lambda item: float(item[0])
            ):
                p_text = "n/a" if contrast["p"] is None else f"{contrast['p']:.4g}"
                lines.append(
                    f"| {partner}={level} | {contrast['mean_difference']:+.4f} nats "
                    f"| {p_text} | {contrast['pairs']} "
                    f"| {contrast['min_achievable_p']:.4g} "
                    f"| {contrast['negative']} negative, "
                    f"{contrast['positive']} positive |"
                )
            lines += [
                "",
                "Agreement in sign across levels is a description, not a test: it says "
                "the pooled number is not hiding a reversal. Whether the levels "
                "actually differ is the interaction test on the crossed factorial "
                "above, and each level's own p is bounded by its own floor.",
                "",
            ]

    promotions = outcome.get("promotions") or []
    if promotions:
        lines += [
            "## Promotion at each screening cut",
            "",
            "A screen ranks candidates but does not adjudicate a near tie, and the "
            "candidate a cut drops is never trained again. Each cut is tested against "
            "the standard error of the difference between two candidates' screen "
            "means - the measured per-architecture seed std at that length times "
            "`sqrt(2 / seeds)` - so a margin inside that threshold means the two runs "
            "did not separate, whatever the ranking says.",
            "",
            "| stage | steps | seeds | seed std | threshold | cut margin | verdict |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in promotions:
            threshold = f"{row['threshold']:.4f}"
            if row.get("threshold_overridden"):
                threshold += " (overridden)"
            lines.append(
                f"| {row['label']} | {row['steps']} | {row['seeds']} | "
                f"{row['std']:.4f} | {threshold} | {row['cut_margin']:.4f} | "
                + ("decided by the data" if row["resolvable"] else "**not resolved**")
                + " |"
            )
        lines.append("")
        for row in promotions:
            if row["resolvable"]:
                continue
            low, high = row["cut_pair"]
            lines.append(
                f"`{row['label']}`: the cut lands between `{low['name']}` "
                f"({low['loss']:.4f}) and `{high['name']}` ({high['loss']:.4f}), "
                f"{row['cut_margin']:.4f} nats apart against a "
                f"{row['threshold']:.4f}-nat threshold, so the promotion it makes is "
                "selection rather than a decision."
            )
            borderline = row.get("borderline") or []
            if borderline:
                lines.append(
                    "Within the threshold of the cut, so a tie-break would be choosing "
                    "between these as well: "
                    + ", ".join(
                        f"`{entry['name']}` ({entry['gap_to_cut']:.4f})"
                        for entry in borderline
                    )
                    + "."
                )
            lines.append("")

    crossed = outcome.get("crossed_contrast") or {}
    if crossed.get("available"):
        a, b = crossed["factor_a"], crossed["factor_b"]
        used_seeds = crossed.get("seeds") or []
        dropped_seeds = crossed.get("seeds_dropped") or []
        populated = (
            f"every cell populated on every one of its {len(used_seeds)} seeds"
            if not dropped_seeds
            else f"every cell populated on the {len(used_seeds)} seeds that finished"
        )
        lines += [
            "## Crossed factorial: which gene is doing the work",
            "",
            f"`{a}` x `{b}`, both at two levels, {populated}. Unlike the "
            "marginal associations above, each main effect here is an average "
            "over both levels of the other gene, so the two are separable even "
            "when the zoo correlates them.",
            "",
        ]
        if dropped_seeds:
            lines += [
                f"**Seeds dropped: {dropped_seeds}.** They carry fewer runs than "
                "the fullest seed, so the design is incomplete *there* rather "
                "than wrong, and the contrast is reported over the seeds that "
                "finished rather than withheld. Every number below is an average "
                f"over {len(used_seeds)} seeds, not "
                f"{len(used_seeds) + len(dropped_seeds)}, and the procedure "
                "floors scale with the seed count - so confirm them once the "
                "sweep has finished.",
                "",
            ]
        lines += [
            f"| {a} \\ {b} | " + " | ".join(str(v) for v in crossed["levels"][b]) + " |",
            "|---|" + "---|" * len(crossed["levels"][b]),
        ]
        for a_value in crossed["levels"][a]:
            row = []
            for b_value in crossed["levels"][b]:
                cell = crossed["cells"].get(f"{a}={a_value}|{b}={b_value}", {})
                row.append(f"{cell.get('mean_val_loss', float('nan')):.4f}")
            lines.append(f"| **{a}={a_value}** | " + " | ".join(row) + " |")
        lines += [
            "",
            "| Gene | Main effect | p | Method | Assignments | Procedure floor |",
            "|---|---|---|---|---|---|",
        ]
        for gene in (a, b):
            main = crossed["main"][gene]
            p_text = "unavailable" if main["p"] is None else f"{main['p']:.4g}"
            floor = main.get("min_achievable_p")
            if floor is None:
                floor_text = "n/a"
            else:
                floor_text = f"{floor:.4g}"
                if main["method"] == "monte-carlo":
                    floor_text += " (Monte-Carlo, raise `--samples`)"
                if main.get("at_floor"):
                    floor_text = "**at floor** — " + floor_text
            lines.append(
                f"| `{gene}` | {main['nats']:+.4f} nats | {p_text} | {main['method']} | "
                f"{main['assignments']} | {floor_text} |"
            )
        lines.append("")
        floored = [gene for gene in (a, b) if crossed["main"][gene].get("at_floor")]
        if floored:
            sample = crossed["main"][floored[0]]
            if sample["method"] == "exact":
                bound = (
                    "the observed re-splitting was matched only by its own "
                    f"mirror, so the exact test over this design's "
                    f"{sample['assignments']}-assignment null cannot report "
                    "anything smaller"
                )
            else:
                bound = (
                    "the assignment space is past what can be enumerated, so the "
                    "test fell to Monte-Carlo and its p is bounded by the draw "
                    "count rather than by the evidence — raise `--samples` before "
                    "quoting it"
                )
            lines += [
                (
                    "Both main effects sit on their procedure floor"
                    if len(floored) == 2
                    else f"`{floored[0]}` sits on its procedure floor"
                )
                + f" — {bound}. Such a p-value says *this separation is the "
                "largest the design can express*, not how significant the effect "
                "would be at a larger size.",
                "",
                "The effect *size* is the interpretable quantity here "
                "(`main effect` in nats, and its step-equivalent below); the "
                "p-value is bounded by the design rather than by the evidence.",
                "",
            ]
        contrast = crossed["main"][a]
        for gene in (a, b):
            main = crossed["main"][gene]
            pieces = ", ".join(
                f"{level}: {row['nats']:+.4f}" for level, row in main["by_other_level"].items()
            )
            lines.append(f"- `{gene}` main effect by level of the other gene: {pieces}")
        if contrast.get("interaction_nats") is not None:
            # Three numbers, because they answer different questions and reporting
            # only the strictest would misrepresent the evidence. The re-split test
            # is exact but assumes no main effect either; the seed-blocked sign
            # test is the well-calibrated one but assumes symmetry; the paired t is
            # a reference. A reader should see all three rather than whichever one
            # the code happened to keep.
            re_split = contrast.get("interaction_p")
            seed_blocked = contrast.get("interaction_p_seed_blocked")
            seed_floor = contrast.get("interaction_p_seed_blocked_floor")
            paired_t = contrast.get("interaction_p_paired_t")
            contrasts = contrast.get("interaction_seed_contrasts") or []

            def _fmt(value):
                return "n/a" if value is None else f"{value:.4g}"

            seeds_consistent = bool(contrasts) and len({v > 0 for v in contrasts}) == 1
            floor_note = ""
            if seed_floor is not None and seed_blocked is not None:
                floor_note = f" (procedure floor {seed_floor:.4g})"
            lines += [
                "",
                f"Interaction: {contrast['interaction_nats']:+.4f} nats.",
                "",
                "| test | null | p |",
                "|---|---|---|",
                f"| re-split within blocks | no main effect *and* no interaction "
                f"(conservative) | {_fmt(re_split)} |",
                f"| seed-blocked sign | seeds exchangeable, symmetric contrasts "
                f"| {_fmt(seed_blocked)}{floor_note} |",
                f"| paired t (reference) | normal contrasts | {_fmt(paired_t)} |",
                "",
            ]
            if (
                seed_floor is not None
                and seed_blocked is not None
                and seed_blocked <= seed_floor + 1e-12
                and len(contrasts) >= 2
                and seeds_consistent
            ):
                lines += [
                    f"The seed-blocked test is **at its floor**: {seed_floor:.4g} is "
                    f"the most extreme p {len(contrasts)} seeds can express, so "
                    "every seed agreeing is the strongest that test can say. "
                    "Resolving the interaction further needs more seeds, not a "
                    "different test; each extra seed costs ~1.1 GPU-hours.",
                    "",
                ]
            if contrasts:
                lines.append(
                    "Per-seed interaction: "
                    + ", ".join(f"{v:+.4f}" for v in contrasts)
                    + (
                        " — every seed agrees on the sign, which is what the "
                        "seed-blocked test measures."
                        if seeds_consistent
                        else " — the sign is **not** consistent across seeds."
                    )
                )
                lines.append("")
            lines.append(
                "The re-split test assumes no main effect either, so it is "
                "conservative for the interaction alone and a large value there "
                "does not rule an interaction out; the seed-blocked test is the "
                "better-calibrated one and its floor is 2/2^seeds. Treat a "
                "consistent sign across every seed with a large re-split p as a "
                "lead that more seeds would settle, not as evidence against."
            )
        lines.append("")

    # Only present when a caller asked for it, because each rule was declared for a
    # *pooled* arm with a known seed count: applying one to a different design would
    # label a question that arm never pre-registered. Both are rendered by the same
    # code because a rule's branch, its evidence and its floors should not be
    # presented differently depending on which arm declared it.
    for key, title in (
        ("replication_decision", "Pre-registered rule: does the attribution replicate?"),
        (
            "head_pairing_decision",
            "Pre-registered rule: is the head effect level-dependent?",
        ),
    ):
        rule = outcome.get(key) or {}
        if not rule:
            continue
        lines += [
            f"## {title}",
            "",
            f"**{rule['label']}**",
            "",
            rule["reading"],
            "",
        ]
        if not rule.get("available"):
            lines += [f"*Not applied: {rule.get('reason')}*", ""]
            continue
        lines += [
            "| Condition | Value | Met |",
            "|---|---|---|",
        ]
        for row in rule["conditions"]:
            lines.append(
                f"| {row['condition']} | {row['value']} | "
                f"{'yes' if row['met'] else 'NO'} |"
            )
        lines.append("")
        if rule.get("note"):
            lines += [rule["note"], ""]
        if rule.get("seeds_dropped"):
            lines += [
                f"Applied over seeds {rule.get('seeds')}; "
                f"{rule['seeds_dropped']} could not be read "
                "(reason in the crossed section above).",
                "",
            ]

    if outcome.get("paired_vs_best"):
        lines += [
            "## Seed-matched contrast against the best architecture",
            "",
            "| Candidate | Mean delta | Paired seeds |",
            "|---|---|---|",
        ]
        for row in outcome["paired_vs_best"]:
            lines.append(
                f"| `{row['candidate']}` | {row['mean_delta_vs_best']:+.4f} | {row['paired_seeds']} |"
            )
        lines.append("")

    if outcome.get("seeds_per_architecture"):
        lines += [
            "## Seeds per architecture",
            "",
            "```json",
            json.dumps(outcome["seeds_per_architecture"], indent=2),
            "```",
            "",
        ]

    path.write_text("\n".join(lines))
