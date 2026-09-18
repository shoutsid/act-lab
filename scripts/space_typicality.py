"""How typical is a chosen architecture set, compared to random draws?

The brief's section 11 asks for the first scientific comparison of *uniformly
sampled* versus *manually selected* architectures under equal compute. That
comparison has two confounds worth settling before any GPU hours are spent, and
this script settles both from the zoo definitions alone, with no training.

1. **Hindsight.** If the manual set already sits in a structurally unusual part of
   the search space, then "manual beats random" would measure hindsight rather
   than judgement. That is the typicality table below.
2. **Budget.** "Equal compute" is not bought by drawing from the same *band*: the
   sum of total parameters over a set is the arm's training budget, and a band is
   a per-candidate constraint whose set total moves with the draw. Measured here,
   a 13-candidate uniform draw at 100M +/-30% can carry **1.19x** the curated
   set's total parameters, handing the random arm that much more training against
   effects this phase measures at 0.18 nats. `--budget-set` scans draw seeds and
   reports the one whose total matches, so the equal-compute claim is a checkable
   number rather than an assumption about the band.

For each set the typicality table reports two things against the population of all
cells inside the parameter band:

``mean``
    where the set sits in depth and hidden size. A set concentrated in one region
    is a set that was picked for a reason, and if that region is one the project
    has since measured to be good, the manual arm carries hindsight.
``spread``
    the mean pairwise structural distance inside the set: coverage rather than
    location. Under equal compute a *diverse* set explores more of the space, so
    coverage is the mechanism by which a designed zoo can beat a random draw even
    with an unremarkable mean.

Percentiles are computed by Monte-Carlo over uniform draws of the same size from
the same grid, because the statistics are not distribution-free and the grid is
not uniform over shapes. ``--draws`` controls the resolution: a percentile is
reported to about ``100/draws``, so the default of 4000 resolves to ~0.03%.

Usage::

    PYTHONPATH=src python scripts/space_typicality.py \
        --definition artifacts/arch-zoo-100m-v2/zoo_definition.json --size 8 \
        --target-params 78000000 --tolerance 0.05

    # and the section 11 budget check, against the curated set it will face:
    PYTHONPATH=src python scripts/space_typicality.py \
        --definition artifacts/arch-zoo-100m-v1/zoo_definition.json \
        --budget-set artifacts/arch-zoo-100m-v1/zoo_definition.json \
        --budget-candidates 13 --target-params 100000000 --tolerance 0.3
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as stats
from itertools import combinations
from pathlib import Path
from typing import Sequence

from act.search.architecture_report import structural_fields
from act.search.architecture_zoo import candidate_from_cell
from act.search.random_architecture import RandomArchitectureSearch

#: The driver's own default for ``--random-tolerance``. It is *not* the band this
#: scan uses, and it changes which architectures a seed draws: seed 151 gives
#: 0.9996x the curated total at 0.30 and 1.1435x at 0.15, so the flag belongs in the
#: launch line rather than being left to the driver's default.
DRIVER_RANDOM_TOLERANCE = 0.15


#: Structural genes compared. Deliberately just the two the crossed factorial
#: identified as real levers - a set that is typical in expert width but unusual
#: in depth is still unusual in the gene that matters.
GENES = ("depth", "hidden_size")


def random_arm_command(
    *,
    seed: int,
    size: int,
    target_params: int,
    tolerance: float,
    context_size: int = 1024,
    max_total_params: int = 150_000_000,
) -> str:
    """The exact driver command that trains the draw this scan matched.

    Returned rather than printed inline so the scan can hand over a launch line that
    is complete: a line naming only the seed leaves every draw-affecting flag to the
    driver's defaults, and one of those defaults is not the band that was scanned -
    so the arm gets trained on a different set of architectures with nothing in the
    record saying the budget match was lost.
    """
    return (
        "PYTHONPATH=src python scripts/arch_zoo_screen.py --design random"
        f" --candidates {size} --zoo-seed {seed}"
        f" --target-params {target_params}"
        f" --random-tolerance {tolerance}"
        f" --context-size {context_size}"
        f" --max-total-params {max_total_params}"
    )


def structure_of_set(definition: Path | str) -> list[tuple[int, int]]:
    """``(depth, hidden_size)`` per candidate in a zoo definition."""
    fields = structural_fields(definition)
    return [(entry["depth"], entry["hidden_size"]) for entry in fields.values()]


def population_structure(
    *,
    target_params: int,
    tolerance: float,
    context_size: int = 1024,
    vocab_size: int = 8192,
) -> list[tuple[int, int]]:
    """Structure of every cell in the budgeted search space.

    Enumerated rather than sampled: the population is the null, and using a sample
    of it for the null would add sampling error to the thing being measured.
    """
    sampler = RandomArchitectureSearch(
        target_params=target_params,
        tolerance=tolerance,
        size=1,
        context_size=context_size,
        vocab_size=vocab_size,
        max_total_params=400_000_000,
    )
    structure: list[tuple[int, int]] = []
    for cell in sampler.space():
        genome = candidate_from_cell(
            cell,
            seed=0,
            context_size=context_size,
            vocab_size=vocab_size,
            name_prefix="pop_",
            family="population",
        ).genome
        structure.append((genome.depth, genome.hidden_size))
    return structure


def set_total_params(definition: Path | str) -> int:
    """Sum of total parameters over a zoo definition's candidates.

    This is the quantity an arm's training budget is proportional to, which is why
    section 11's "equal compute" is a statement about the *set total* and not about
    the band its members were drawn from.
    """
    payload = json.loads(Path(definition).read_text())
    candidates = payload.get("candidates") or []
    if not candidates:
        raise ValueError(f"no candidates recorded in {definition}")
    return sum(int(candidate["total_params"]) for candidate in candidates)


def budget_matching_draw(
    *,
    target_total: int,
    size: int,
    seeds: Sequence[int],
    target_params: int = 100_000_000,
    tolerance: float = 0.30,
    context_size: int = 1024,
    vocab_size: int = 8192,
    max_total_params: int = 150_000_000,
) -> dict:
    """The uniform draw whose *set total* is closest to a curated set's.

    Selecting a draw seed is a design decision made before any training, from the
    architecture definitions alone, which is what makes it legitimate to
    pre-register: nothing about a candidate's loss enters it. What it does buy is
    coverage, and coverage is section 11's hypothesised mechanism - so the drawn
    set's structure is returned beside it, to be reported rather than assumed
    neutral.
    """
    if size < 1:
        raise ValueError("a draw needs at least one candidate")
    if not seeds:
        raise ValueError("no draw seeds to scan")

    ranked: list[dict] = []
    for seed in seeds:
        sampler = RandomArchitectureSearch(
            target_params=target_params,
            tolerance=tolerance,
            size=size,
            seed=seed,
            context_size=context_size,
            vocab_size=vocab_size,
            max_total_params=max_total_params,
        )
        zoo = sampler.sample()
        total = sum(candidate.total_params for candidate in zoo)
        ranked.append(
            {
                "seed": int(seed),
                "total_params": total,
                "ratio": total / target_total,
                "structure": [
                    (candidate.genome.depth, candidate.genome.hidden_size)
                    for candidate in zoo
                ],
            }
        )
    # Closest first, then by seed so the choice is reproducible when two draws
    # round to the same distance.
    ranked.sort(key=lambda row: (abs(row["ratio"] - 1.0), row["seed"]))
    best = ranked[0]
    # The band constrains each candidate, not the set total, so how far a typical
    # draw sits from the curated total is the number that decides whether the
    # comparison needs a matched seed at all - and a scan that only reported its
    # closest few would make the gap look like a tail case rather than the norm.
    ratios = sorted(row["ratio"] for row in ranked)
    return {
        "seed": best["seed"],
        "total_params": best["total_params"],
        "ratio": best["ratio"],
        "target_total": target_total,
        "scanned": len(ranked),
        "ranked": ranked[:5],
        "distribution": {
            "mean": stats.fmean(ratios),
            "median": stats.median(ratios),
            "p10": ratios[max(0, round(0.10 * (len(ratios) - 1)))],
            "p90": ratios[min(len(ratios) - 1, round(0.90 * (len(ratios) - 1)))],
            "min": ratios[0],
            "max": ratios[-1],
        },
        "structure": best["structure"],
    }


def sample_statistics(sample: list[tuple[int, int]]) -> dict[str, float]:
    """Mean location and mean pairwise spread, per gene.

    ``spread`` is the mean pairwise absolute difference rather than the range: a
    range is fixed by two architectural outliers and says nothing about the rest
    of the set, which is exactly the failure mode being guarded against.
    """
    out: dict[str, float] = {}
    for index, gene in enumerate(GENES):
        values = [entry[index] for entry in sample]
        out[f"{gene}_mean"] = stats.mean(values)
        out[f"{gene}_spread"] = (
            stats.mean([abs(a - b) for a, b in combinations(values, 2)])
            if len(values) > 1
            else 0.0
        )
    return out


def typicality(
    sample: list[tuple[int, int]],
    population: list[tuple[int, int]],
    *,
    draws: int = 4000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Percentile of each statistic against uniform draws of the same size.

    A percentile near 50 means the set is indistinguishable from a random draw of
    its own size; near 0 or 100 means it occupies a distinguishable part of the
    space on that statistic.
    """
    if not sample:
        raise ValueError("cannot assess typicality of an empty set")
    if len(sample) > len(population):
        raise ValueError("sample is larger than the population it is compared to")

    observed = sample_statistics(sample)
    generator = random.Random(seed)
    null: dict[str, list[float]] = {key: [] for key in observed}
    for _ in range(draws):
        draw = [population[index] for index in generator.sample(range(len(population)), len(sample))]
        for key, value in sample_statistics(draw).items():
            null[key].append(value)

    return {
        key: {
            "value": value,
            # <= rather than <, so a set that is at the extreme still gets a
            # percentile rather than 0 and so the two-sided reading is symmetric.
            "percentile": 100.0
            * sum(1 for entry in null[key] if entry <= value)
            / len(null[key]),
        }
        for key, value in observed.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--definition",
        action="append",
        required=True,
        help="zoo_definition.json to assess; repeatable",
    )
    parser.add_argument("--target-params", type=int, default=78_000_000)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--context-size", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument(
        "--draws",
        type=int,
        default=4000,
        help="Monte-Carlo draws; a percentile resolves to about 100/draws",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--budget-set",
        default="",
        help=(
            "a zoo_definition.json whose *total* parameters a random draw should "
            "match, for section 11's equal-compute condition. Reports the draw "
            "seed to launch the random arm with, so the two arms train the same "
            "number of parameters rather than merely drawing from the same band."
        ),
    )
    parser.add_argument(
        "--budget-candidates",
        type=int,
        default=0,
        help="draw size for --budget-set; defaults to that set's candidate count",
    )
    parser.add_argument(
        "--budget-seeds",
        type=int,
        default=200,
        help="how many draw seeds to scan (0..N-1); more seeds mean a closer match",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    population = population_structure(
        target_params=args.target_params,
        tolerance=args.tolerance,
        context_size=args.context_size,
        vocab_size=args.vocab_size,
    )
    print(
        f"Population: {len(population)} cells within +/-{args.tolerance:.0%} of "
        f"{args.target_params:,} params"
    )
    base = sample_statistics(population)
    print(
        "  {a}: mean {b:.2f}, mean pairwise spread {c:.2f}".format(
            a=GENES[0],
            b=base[f"{GENES[0]}_mean"],
            c=base[f"{GENES[0]}_spread"],
        )
    )
    print(
        "  {a}: mean {b:.0f}, mean pairwise spread {c:.0f}".format(
            a=GENES[1],
            b=base[f"{GENES[1]}_mean"],
            c=base[f"{GENES[1]}_spread"],
        )
    )
    print()

    header = "  ".join(
        f"{gene} {stat}".rjust(22) for gene in GENES for stat in ("mean", "spread")
    )
    print(f"{'set':<30}  n  {header}")
    for definition in args.definition:
        sample = structure_of_set(definition)
        if not sample:
            print(f"{definition:<30}  --  (no structural genes recorded)")
            continue
        result = typicality(sample, population, draws=args.draws, seed=args.seed)
        cells = "  ".join(
            f"{result[f'{gene}_{stat}']['value']:>10.2f} (p{result[f'{gene}_{stat}']['percentile']:>4.0f})"
            for gene in GENES
            for stat in ("mean", "spread")
        )
        print(f"{Path(definition).parent.name:<30} {len(sample):>3}  {cells}")

    print()
    print(
        "pXX is the percentile among uniform draws of the same size. Near 50 means "
        "indistinguishable from random on that statistic."
    )
    print(
        "A 'spread' percentile near 100 means the set is unusually *diverse* for "
        "its size, which is a coverage advantage rather than a location one."
    )

    if not args.budget_set:
        return

    definition = args.budget_set
    target_total = set_total_params(definition)
    size = args.budget_candidates or len(structure_of_set(definition))
    print()
    print("Budget matching for section 11's equal-compute condition")
    print(
        f"  {Path(definition).parent.name} holds {size} candidates totalling "
        f"{target_total:,} params"
    )
    match = budget_matching_draw(
        target_total=target_total,
        size=size,
        seeds=range(args.budget_seeds),
        target_params=args.target_params,
        tolerance=args.tolerance,
        context_size=args.context_size,
        vocab_size=args.vocab_size,
    )
    print(f"  closest of {match['scanned']} draws of {size} candidates:")
    for row in match["ranked"]:
        print(
            f"    seed {row['seed']:>4}: {row['total_params']:>15,} params "
            f"({row['ratio']:.4f}x)"
        )
    dist = match["distribution"]
    print(
        f"  but the band fixes each candidate, not the set total: across all "
        f"{match['scanned']} draws the median carries {dist['median']:.4f}x the "
        f"curated set (mean {dist['mean']:.4f}, p10 {dist['p10']:.4f}, "
        f"p90 {dist['p90']:.4f}, range {dist['min']:.4f}-{dist['max']:.4f})."
    )
    if dist["median"] > 1.0 + 1e-4:
        print(
            "  So a band-matched draw is not an equal-compute draw: matching the "
            "curated total is what makes the comparison fair, and the seed below "
            "is chosen for that rather than assumed neutral."
        )
    drawn = sample_statistics(match["structure"])
    print(
        "  the matched draw's own structure: "
        + ", ".join(
            f"{gene} mean {drawn[f'{gene}_mean']:.1f} spread "
            f"{drawn[f'{gene}_spread']:.1f}"
            for gene in GENES
        )
    )
    print(
        "  Report that structure beside the result: matching on budget can move "
        "coverage, and coverage is the mechanism section 11 hypothesises for a "
        "designed set."
    )
    print("  Launch the random arm with exactly this command, so the arm is trained")
    print("  on the draw whose budget was matched rather than on a nearby one:")
    print(
        "    "
        + random_arm_command(
            seed=match["seed"],
            size=size,
            target_params=args.target_params,
            tolerance=args.tolerance,
            context_size=args.context_size,
        )
    )
    print(
        f"    that draw is {match['ratio']:.4f}x the curated set (worst scanned "
        f"draw {dist['max']:.4f}x, median {dist['median']:.4f}x)."
    )
    if abs(args.tolerance - DRIVER_RANDOM_TOLERANCE) > 1e-9:
        print(
            f"  Note: the band is required, not optional. The driver defaults to "
            f"--random-tolerance {DRIVER_RANDOM_TOLERANCE}, and the same seed draws a "
            "different set of architectures there, so the budget match above would "
            "be lost with nothing in the record saying so."
        )


if __name__ == "__main__":
    main()
