#!/usr/bin/env python3
"""What the section 11 comparison can detect at the size this phase can afford.

Section 11 asks for the first scientific comparison of randomly sampled against
manually selected architectures under equal compute. The comparison has been
*ready* since section 9a - the curated sets carry no hindsight advantage and a
budget-matched draw exists (sampler seed 151, 0.9996x the curated total) - and
what it has never had is an answer to the question that decides whether to spend
the hours: **how large a curation advantage can a set-versus-set comparison
detect at the size this phase can run?**

That question is answerable with no GPU time at all, from measurements the phase
already owns:

* the **population** of candidate qualities is the per-candidate mean losses the
  arm actually recorded at the length being compared - the real spread of a
  curated set rather than an assumed one;
* the **noise** is the arm's measured within-architecture seed std at that
  length, which is what the design has to see through;
* the **design** is the only part that is chosen, and it is chosen from what is
  affordable: 13 candidates per set (the curated set's size), a seeded number of
  blocks, and the two alternative models of what curation buys.

Two alternative models, because they are different hypotheses about how a manual
search wins, and they have very different costs to detect:

``shift-all``
    Curation lifts every candidate in the set - human judgement is right about the
    architecture family in general.
``shift-best``
    Curation only finds a better winner - the mechanism section 9a predicted when
    it found the curated sets' advantage to be in coverage rather than location.
    This model turns out to be **undetectable at the affordable design size**, and
    for a reason no amount of GPU time fixes: a permutation null on the sets' floors
    already contains the improved winner, because half its assignments put that
    member on either side. Whichever side holds it takes the low floor and the other
    side's floor sits one winner-gap above it, so the magnitude of the difference is
    near the observed one whatever the labelling - the test can only see a shift of
    the whole set. What is measurable about section 11's question is the
    ``shift-all`` row.

Usage::

    PYTHONPATH=src python scripts/set_comparison_power.py
    PYTHONPATH=src python scripts/set_comparison_power.py \
        --arm arch-zoo-100m-v2 --steps 1200 --set-size 8 --blocks 3
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from act.search.architecture_eval import EvalResult, load_results  # noqa: E402
from act.search.architecture_report import (  # noqa: E402
    two_set_blocked_test,
    with_structure,
)

#: Losses here are far from convergence and the effect size depends on the
#: horizon, so the population and the noise are always read at one length and the
#: table says which. A 400-step screen's candidate spread is 3.7x the 1200-step
#: one (section 4a), which makes a short-screen comparison *more* sensitive to a
#: real difference and less transferable to the horizon a search trains for.
DETECTABLE_POWER = 0.8


def candidate_population(root: Path, arm: str, steps: int) -> tuple[list[float], float]:
    """Per-candidate mean losses at one length, and the measured seed std there."""
    directory = root / arm
    results = with_structure(
        load_results(sorted(directory.glob("*_seed*_results.jsonl"))),
        directory / "zoo_definition.json",
    )
    grouped: dict[str, dict[int, float]] = {}
    for entry in results:
        if entry.steps == steps:
            grouped.setdefault(entry.name, {})[entry.seed] = entry.validation_loss
    if not grouped:
        raise SystemExit(f"no {steps}-step rows in {directory}")
    means = [sum(by_seed.values()) / len(by_seed) for by_seed in grouped.values()]
    stds = [
        (sum((value - mean) ** 2 for value in by_seed.values()) / (len(by_seed) - 1)) ** 0.5
        for by_seed in grouped.values()
        if len(by_seed) > 1
        for mean in [sum(by_seed.values()) / len(by_seed)]
    ]
    if not stds:
        raise SystemExit(f"no candidate in {directory} ran on 2+ seeds at {steps} steps")
    return means, sum(stds) / len(stds)


def synthetic_results(
    population: list[float],
    *,
    set_size: int,
    blocks: int,
    noise: float,
    advantage: float,
    model: str,
    generator: random.Random,
) -> list[EvalResult]:
    """One replicate of the comparison, as ledger rows the real test can read.

    The random set is drawn from the measured population and the curated set is the
    same draw made ``advantage`` nats better, per the model - on every member
    (``shift-all``) or on its best member only (``shift-best``), where the shift
    moves the set's own best draw rather than inventing a candidate better than any
    the phase has trained.

    The shift is applied as an *improvement* (a lower loss), and the direction is
    not cosmetic. On the winner it decides whether the planted effect reaches the
    statistic at all: the two best members of a 13-candidate draw sit 0.039 nats
    apart on average and 0.009 at the median, so a 0.20-nat penalty on the best draw
    pushes it above the runner-up 99.9% of the time and the set's floor moves by
    that gap instead of by the advantage - a nominal 0.20-nat effect that arrives
    as ~0.01, which is not distinguishable from a manipulation that never happened.
    Improving the winner cannot be truncated, because it stays the winner - and a
    curated *winner* is the hypothesis section 9a named.

    Each run carries **one** noise draw at the measured per-run seed std, and no
    shared per-seed offset: a per-seed offset cancels in a within-block difference,
    which is exactly what blocking is for, so adding one would only inflate the
    variance and understate power. Two draws would do the same by a factor of
    sqrt(2), and a power table that understates power is the wrong side to be
    wrong on when the question is whether to spend GPU hours.
    """
    rows: list[EvalResult] = []
    draw = lambda: [generator.choice(population) for _ in range(set_size)]  # noqa: E731
    for label in ("random", "manual"):
        qualities = draw()
        if label == "manual":
            if model == "shift-all":
                qualities = [value - advantage for value in qualities]
            else:
                best = min(range(len(qualities)), key=lambda index: qualities[index])
                qualities[best] -= advantage
        for index, quality in enumerate(qualities):
            for block in range(blocks):
                loss = quality + generator.gauss(0.0, noise)
                rows.append(
                    EvalResult(
                        name=f"{label}_{index}",
                        family="power",
                        genome_hash=f"{label}-{index}",
                        seed=block,
                        total_params=100_000_000,
                        active_params=40_000_000,
                        contracted_flops_per_token=2.5e8,
                        dense_runtime_flops_per_token=4.0e8,
                        training_flops=0.0,
                        validation_flops=0.0,
                        final_train_loss=loss + 0.2,
                        mean_train_loss=loss + 0.25,
                        validation_loss=loss,
                        steps=400,
                        train_tokens=0,
                        val_tokens=0,
                        elapsed_s=0.0,
                        measured_step_s=0.0,
                        measured_tokens_per_s=0.0,
                        peak_memory_mb=0.0,
                        device="cpu",
                        description="",
                        val_source="synthetic",
                    )
                )
    return rows


def power_at(
    population: list[float],
    *,
    advantage: float,
    model: str,
    statistic: str,
    set_size: int,
    blocks: int,
    noise: float,
    replicates: int,
    samples: int,
    seed: int = 0,
) -> float:
    """Fraction of replicates whose seed-blocked test separates the two sets."""
    generator = random.Random(seed)
    random_members = tuple(f"random_{index}" for index in range(set_size))
    manual_members = tuple(f"manual_{index}" for index in range(set_size))
    hits = 0
    for _ in range(replicates):
        rows = synthetic_results(
            population,
            set_size=set_size,
            blocks=blocks,
            noise=noise,
            advantage=advantage,
            model=model,
            generator=generator,
        )
        outcome = two_set_blocked_test(
            rows,
            400,
            (("random", random_members), ("manual", manual_members)),
            statistic=statistic,
            samples=samples,
            seed=seed,
        )
        if outcome.get("available") and outcome["p"] < 0.05:
            hits += 1
    return hits / replicates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="arch-zoo-100m-v1")
    parser.add_argument(
        "--steps",
        type=int,
        default=400,
        help=(
            "length the population and the noise are read at. 400 steps is the "
            "affordable option: the curated arm's 13 candidates already have "
            "400-step rows on two seeds, so only the random arm has to be bought"
        ),
    )
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--set-size", type=int, default=13)
    parser.add_argument("--blocks", type=int, default=2, help="shared seeds")
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument(
        "--samples",
        type=int,
        default=2_000,
        help="Monte-Carlo draws per replicate; the design's own floor is reported too",
    )
    parser.add_argument(
        "--advantages",
        default="0,0.01,0.02,0.03,0.05,0.075,0.10,0.15,0.20,0.30,0.40,0.60",
        help="curation advantages in nats to sweep",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.artifacts_root)
    population, noise = candidate_population(root, args.arm, args.steps)
    advantages = [float(value) for value in args.advantages.split(",")]

    print(f"Population: {args.arm} at {args.steps} steps")
    print(
        f"  {len(population)} candidates, means "
        f"{min(population):.4f} .. {max(population):.4f} "
        f"(spread {max(population) - min(population):.4f} nats)"
    )
    print(f"  measured per-run seed std: {noise:.4f} nats")
    print(
        f"Design: {args.set_size} candidates per set x {args.blocks} seeds = "
        f"{2 * args.set_size * args.blocks} runs, "
        f"{args.replicates} replicates x {args.samples} permutation draws\n"
    )

    for model in ("shift-all", "shift-best"):
        print(f"Alternative model: {model}")
        print(
            f"  {'advantage':>10} | {'mean':>6} | {'best':>6} | {'worst':>6}"
        )
        print(f"  {'-' * 10}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 6}")
        detectable: dict[str, float | None] = {"mean": None, "best": None, "worst": None}
        for advantage in advantages:
            powers = {}
            for statistic in ("mean", "best", "worst"):
                powers[statistic] = power_at(
                    population,
                    advantage=advantage,
                    model=model,
                    statistic=statistic,
                    set_size=args.set_size,
                    blocks=args.blocks,
                    noise=noise,
                    replicates=args.replicates,
                    samples=args.samples,
                    seed=args.seed,
                )
                if detectable[statistic] is None and powers[statistic] >= DETECTABLE_POWER:
                    detectable[statistic] = advantage
            print(
                f"  {advantage:>10.3f} | {powers['mean']:>6.2f} | "
                f"{powers['best']:>6.2f} | {powers['worst']:>6.2f}"
            )
        for statistic, value in detectable.items():
            answer = "not reached in this sweep" if value is None else f"{value:.3f} nats"
            print(f"  -> 80% power on the {statistic:<5} statistic at {answer}")
        print()

    # The cost is the part that makes this a decision rather than a curiosity.
    rate = 0.046 if args.steps == 400 else 0.133
    runs = args.set_size * args.blocks
    print(
        f"Affordable form: the curated arm's {args.set_size} candidates already hold "
        f"{args.steps}-step rows on {args.blocks} seeds, so only the random arm is bought:"
        f" {runs} runs x ~{rate:.3f} GPU-hours = ~{runs * rate:.2f} GPU-hours."
    )
    print(
        "Caveat that travels with every number above: effect sizes are "
        "horizon-dependent (1.65x per doubling), and section 4a measured a "
        "400-step screening stage to report a spread 3.7x the one that survives at "
        "1200. So this answers \"is the difference visible at this length\", and any "
        "quoted size is a 400-step size."
    )


if __name__ == "__main__":
    main()
