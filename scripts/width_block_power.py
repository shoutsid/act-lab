#!/usr/bin/env python3
"""What a smaller width block decides, and what the hours are actually spent on.

The width arm's price at declared fidelity is not a plumbing cost: it is the
protocol's token budget. One arm trains 24,576,000 tokens at the measured
4,454.4 tok/s, which is 1.53 h of arithmetic; a seed runs two arms and pays a
fixed event overhead, and a widening schedule makes each segment 1.358x as
expensive because the model grows at three events. That is 2.59 h per paired
seed all-in, and 12 seeds - the block the sign test was written for - is 31.1 h.

So the question this script answers is the one the design note deferred: **which
of those hours buy evidence, and which buy resolution nobody needs.** It is
computed, not asserted:

* the exact one-sided sign-test p-value of every bar the block could use, so
  "fewer seeds" is read against what it actually decides rather than against a
  habit;
* the two-stage rule (six seeds first, extend only when the first six are
  ambiguous) with its level under no effect and its expected cost under a range
  of per-seed win probabilities;
* the decomposition of the 2.59 h, so it is visible that only ~0.5 h of each
  seed is overhead an engineer could remove and the rest is the declared token
  budget.

Nothing here is a measurement of the arm - the arm has not run. The rates are
the committed ones from ``docs/width-control-design.md`` section 9 (the
2026-10-31 session: 1.63 h per 24,576,000-token pair at h768, 6.03 GPU-h for
three seed sessions).

Usage::

    PYTHONPATH=src python scripts/width_block_power.py
"""

from __future__ import annotations

import argparse
import math

# Committed rates (docs/width-control-design.md section 9), not re-derived here.
TOKENS_PER_ARM = 24_576_000
MEASURED_TOKENS_PER_SECOND = 4_454.4
PAIR_EVENT_OVERHEAD_HOURS = 0.1
WIDTH_SCHEDULE_MULTIPLIER = 1.358
SESSION_OVERHEAD_HOURS = 0.38

#: The registered quality bar: adaptive must win 10 of 12 paired seeds.
REGISTERED_WINS = 10
REGISTERED_SEEDS = 12
#: The smaller bar this script prices: a perfect sweep, which is a *stricter*
#: test per seed than 10 of 12 (1/64 against 79/4096).
SWEEP_SEEDS = 6


def per_seed_hours() -> dict[str, float]:
    """The 2.59 h, decomposed into the parts an engineer can and cannot remove."""
    training = TOKENS_PER_ARM / MEASURED_TOKENS_PER_SECOND / 3600.0
    pair = training + PAIR_EVENT_OVERHEAD_HOURS
    widened = pair * WIDTH_SCHEDULE_MULTIPLIER
    return {
        "one_arm_training": training,
        "pair_training_and_events": pair,
        "pair_after_width_schedule": widened,
        "all_in": widened + SESSION_OVERHEAD_HOURS,
    }


def binomial_tail(wins: int, trials: int, probability: float) -> float:
    """P(X >= wins) for X ~ Binomial(trials, probability), exactly."""
    return sum(
        math.comb(trials, k) * probability**k * (1.0 - probability) ** (trials - k)
        for k in range(wins, trials + 1)
    )


def bar_level(wins: int, trials: int) -> float:
    """One-sided level of a k-of-n all-wins bar under no effect."""
    return binomial_tail(wins, trials, 0.5)


def two_stage(p: float, *, first: int, total: int, bar: int) -> dict[str, float]:
    """Level and expected seeds of "look at ``first``, extend only if ambiguous".

    The rule: after ``first`` seeds, a perfect sweep is a positive result and
    ``first - 3`` or fewer wins is futility (the full bar has become
    unreachable); anything in between extends to ``total`` seeds and is read
    against ``bar``. Returns the probability of a positive result and the
    expected number of seeds, both under a per-seed win probability ``p``.
    """

    def pmf(trials: int, wins: int) -> float:
        return math.comb(trials, wins) * p**wins * (1.0 - p) ** (trials - wins)

    positive = 0.0
    expected = 0.0
    rest = total - first
    for wins in range(first + 1):
        chance = pmf(first, wins)
        if wins == first:
            positive += chance
            expected += chance * first
        elif wins <= first - 3:
            expected += chance * first
        else:
            reached = binomial_tail(bar - wins, rest, p)
            positive += chance * reached
            expected += chance * total
    return {"positive_probability": positive, "expected_seeds": expected}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probabilities",
        default="0.5,0.65,0.8,0.9",
        help="per-seed adaptive-win probabilities to price the two-stage rule at",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hours = per_seed_hours()
    all_in = hours["all_in"]

    print("Per paired seed, from the committed rates")
    print(f"  24,576,000 tokens / {MEASURED_TOKENS_PER_SECOND:,.1f} tok/s = "
          f"{hours['one_arm_training']:.4f} h per arm (both arms overlap on two T4s)")
    print(f"  + {PAIR_EVENT_OVERHEAD_HOURS:.2f} h events/rebuild            "
          f"= {hours['pair_training_and_events']:.4f} h per pair")
    print(f"  x {WIDTH_SCHEDULE_MULTIPLIER:.3f} (the model grows at three events) "
          f"= {hours['pair_after_width_schedule']:.4f} h")
    print(f"  + {SESSION_OVERHEAD_HOURS:.2f} h startup/restore/publish   "
          f"= {all_in:.4f} h all-in")
    share = 1.0 - (SESSION_OVERHEAD_HOURS + PAIR_EVENT_OVERHEAD_HOURS) / all_in
    print(f"  -> {100 * share:.1f}% of every seed's hours is the declared token budget\n")

    print("Exact one-sided sign test: what each bar decides")
    print(f"  {'bar':>8}  {'level under no effect':>22}  {'block cost':>10}")
    for wins, seeds in (
        (SWEEP_SEEDS, SWEEP_SEEDS),
        (7, 7),
        (8, 8),
        (REGISTERED_WINS, REGISTERED_SEEDS),
        (11, REGISTERED_SEEDS),
        (12, REGISTERED_SEEDS),
    ):
        level = bar_level(wins, seeds)
        print(f"  {wins:>2}/{seeds:<5}  {level:>22.6f}  {seeds * all_in:>8.1f} h")

    print("\nThe same seed count, read as a price")
    for seeds in (6, 8, 10, 12):
        print(f"  {seeds:>2} seeds  {seeds * all_in:>6.1f} h")

    probabilities = [float(value) for value in args.probabilities.split(",")]
    print("\nTwo-stage rule: six seeds first, extend to twelve only if ambiguous")
    print("  (6/6 is positive; 0-2 wins of 6 is futility; 3-5 wins extends to 10/12)")
    print(f"  {'p(win)':>7}  {'P(positive)':>12}  {'E[seeds]':>9}  {'E[hours]':>9}")
    for p in probabilities:
        reading = two_stage(p, first=SWEEP_SEEDS, total=REGISTERED_SEEDS, bar=REGISTERED_WINS)
        print(
            f"  {p:>7.2f}  {reading['positive_probability']:>12.4f}  "
            f"{reading['expected_seeds']:>9.2f}  "
            f"{reading['expected_seeds'] * all_in:>9.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
