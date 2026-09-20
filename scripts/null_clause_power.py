"""What the section 5.14 do-nothing clause can decide, measured before its data exists.

Section 5.14's clause is a sign test over the declared 12-seed block with a fixed limit:
Gate 4 is withheld when the adaptive arm "loses-or-ties" against the null arm on
``>= NULL_WIN_LIMIT = 10`` of 12 seeds. The mirror of the registered harm guard, and
deliberately so - but it is a *rule*, and how much a rule decides is a property of the
rule and the noise, not of the run. This script computes that property, and it is
deliberately run **before** block 15000-15011's null queue exists, so the reading of
whichever verdict lands is fixed in advance rather than chosen afterwards.

What it reports:

* the clause's **level** - how often it fires when there is no effect at all (ties count
  against the adaptive arm, so this is the conservative reading);
* its **firing rate** as a function of the per-seed effect, parameterized by ``z`` =
  effect / sd, because the declared-fidelity noise is not known until the block runs;
* the **dev-fidelity anchor** - section 5.40 measured the only real-text effect this
  program has with the treatment in place (the null arm ahead by 0.13554 nats at sd
  0.1129), so ``z = 1.20`` is the measured analogue, not a guess;
* and the asymmetry that matters for the write-up: **a pass requires only 3 of 12 seeds**,
  so a verdict whose null clause clears is not evidence that adaptive beats doing nothing
  - it is evidence that the clause did not fire, and the value claim then rests on the
  registered cost/band/control rules exactly as registered.

Usage:

    PYTHONPATH=src python scripts/null_clause_power.py
    PYTHONPATH=src python scripts/null_clause_power.py --output artifacts/null-clause-power.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The clause's own constants, imported rather than restated: a drifted limit or block
# length here would silently describe a different rule than the verdict applies.
from v0_verdict import DECLARED_SEEDS as DECLARED_BLOCK  # noqa: E402
from v0_verdict import (  # noqa: E402
    HARM_GUARD_LOSS_LIMIT,
    NULL_WIN_LIMIT,
    POOL_CONTROL_WIN_LIMIT,
)
from deployed_vs_null_study import SUMMARY_PATH as DEPLOYED_SUMMARY  # noqa: E402

DECLARED_SEEDS = len(DECLARED_BLOCK)

# Every withhold clause in the section 5.1 verdict is the *same* rule - a 10-of-12 sign
# test on the adaptive arm's losses - differing only in which arm it is measured against.
# Named here as one table so the unification is checked rather than assumed: if a future
# amendment gives one clause a different limit, the assertion below fails and this analysis
# stops describing the verdict.
WITHHOLD_CLAUSES: dict[str, tuple[int, str]] = {
    "harm_guard": (HARM_GUARD_LOSS_LIMIT, "the pool-matched random arm"),
    "pool_matched_control": (POOL_CONTROL_WIN_LIMIT, "the pool-matched control arm"),
    "null_clause": (NULL_WIN_LIMIT, "the null arm (do nothing)"),
}
assert len({limit for limit, _ in WITHHOLD_CLAUSES.values()}) == 1, (
    "the withhold clauses no longer share one limit; this analysis assumes they do"
)

# Effect sizes to tabulate, in units of the per-seed sd.
Z_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.203, 1.5, 2.0)


def _phi(value: float) -> float:
    """Standard normal CDF, via erfc so the tails keep their precision."""
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def _binomial_tail(wins: int, trials: int, probability: float) -> float:
    """P(X >= wins) for X ~ Binomial(trials, probability), computed exactly."""
    return sum(
        math.comb(trials, k) * probability**k * (1.0 - probability) ** (trials - k)
        for k in range(wins, trials + 1)
    )


def clause_reading(z: float) -> dict[str, float]:
    """The clause's error rates when a seed favours the null with probability ``p``.

    ``p = Phi(z)`` is the per-seed probability that the null arm's loss is lower, which
    is all the sign test sees: the clause fires iff ``NULL_WIN_LIMIT`` or more seeds
    favour the null, i.e. iff the adaptive arm wins at most
    ``DECLARED_SEEDS - NULL_WIN_LIMIT`` of them. Ties count against the adaptive arm, so
    a real tie distribution makes the clause fire slightly *more* often than this - the
    conservative direction for the arm seeking authorization.
    """
    p_null = _phi(z)
    fires = _binomial_tail(NULL_WIN_LIMIT, DECLARED_SEEDS, p_null)
    return {
        "z": z,
        "p_seed_favours_null": p_null,
        "p_clause_fires": fires,
        "p_clause_passes": 1.0 - fires,
    }


def clause_table() -> dict[str, dict[str, float]]:
    """Every withhold clause's level and firing rate on the same z grid.

    Each clause fires when its comparison arm beats adaptive on at least its limit of the
    declared seeds, so each is a one-sided sign test with the same level. They are *not*
    independent - they are computed on the same seeds and the same adaptive arm - and a
    block hard enough to beat adaptive usually beats it in every clause, which makes the
    dependence positive and the union rate below the sum of the marginals. The sum is
    reported as the conservative bound.
    """
    table: dict[str, dict[str, float]] = {}
    for name, (limit, arm) in WITHHOLD_CLAUSES.items():
        rows = {
            f"z={z}": _binomial_tail(limit, DECLARED_SEEDS, _phi(z)) for z in Z_GRID
        }
        table[name] = {
            "comparison_arm": arm,
            "fires_when_arm_beats_adaptive_on_at_least": limit,
            "level_under_no_effect": _binomial_tail(limit, DECLARED_SEEDS, 0.5),
            **rows,
        }
    return table


def dev_fidelity_clause_read() -> dict[str, object]:
    """Which clauses would fire on the one real-text block that exists (section 5.40).

    Dev fidelity, not the declared block, and two of the three arms are present there: the
    null arm and the random arm, both paired against the same adaptive arm. It is recorded
    because it shows the clauses are not redundant - at this effect size one fires and the
    other does not - and because the pool-matched control arm has never been run beside
    them, which the reading states rather than assuming away.
    """
    payload = json.loads(DEPLOYED_SUMMARY.read_text())
    rows = payload["per_seed"]
    null_beats = sum(1 for row in rows if row["null_final_loss"] < row["adaptive_final_loss"])
    random_beats = sum(
        1 for row in rows if row["random_final_loss"] < row["adaptive_final_loss"]
    )
    limit = NULL_WIN_LIMIT
    return {
        "fidelity": "dev (section 5.40's fresh block, seeds 15240-15251)",
        "declared_block": False,
        "null_beats_adaptive": null_beats,
        "null_clause_fires": null_beats >= limit,
        "random_beats_adaptive": random_beats,
        "harm_guard_fires": random_beats >= limit,
        "pool_matched_control": (
            "not evaluated: section 5.40's block carries no pool-matched control arm, and "
            "the clauses are not redundant - on these seeds the null clause fires and the "
            "harm guard does not"
        ),
    }


def dev_fidelity_anchor() -> dict[str, object]:
    """The only measured real-text effect with the treatment in place, from section 5.40.

    Read from the committed summary rather than restated, so the anchor moves with the
    block it comes from: the paired ``null - adaptive`` mean and sd over those 12 seeds.
    """
    payload = json.loads(DEPLOYED_SUMMARY.read_text())
    comparison = payload["comparisons"]["null_minus_adaptive"]
    mean = float(comparison["mean_delta"])
    sd = float(comparison["sample_sd"])
    return {
        "source": "artifacts/study-summaries/real-path-deployed-vs-null.json (section 5.40)",
        "fidelity": "dev (14.9M total / 7.6M active), 400 steps, depth 6, treatment on",
        "effect_nats_null_minus_adaptive": mean,
        "sample_sd": sd,
        "z": abs(mean) / sd,
        "null_wins_observed": sum(
            1 for row in payload["per_seed"] if row["null_final_loss"] < row["adaptive_final_loss"]
        ),
    }


def build_report() -> dict:
    """The whole reading, as the committed artifact holds it.

    Assembled here rather than inline in ``main`` so a test can compare the committed
    artifact against what this script produces *now*: a field restated by hand (the pass
    bar, say) would otherwise sit in the artifact looking correct while the code that
    wrote it no longer computes it.
    """
    anchor = dev_fidelity_anchor()
    rows = [clause_reading(z) for z in Z_GRID]
    level = clause_reading(0.0)["p_clause_fires"]
    passes_required = DECLARED_SEEDS - NULL_WIN_LIMIT + 1
    report = {
        "record_type": "study-summary",
        "kind": "null-clause-power",
        "report_section": "5.14",
        "question": (
            "how much does the registered do-nothing clause decide, on the declared "
            "12-seed block, before its own data exists"
        ),
        "clause": {
            "statistic": "exact sign test on the paired per-seed null-vs-adaptive losses",
            "declared_seeds": DECLARED_SEEDS,
            "null_win_limit": NULL_WIN_LIMIT,
            "fires_when_adaptive_wins_at_most": DECLARED_SEEDS - NULL_WIN_LIMIT,
            "adaptive_wins_required_to_clear": passes_required,
            "ties_count_against_the_adaptive_arm": True,
            "source": "scripts/v0_verdict.py::_null_block (imported, not restated)",
        },
        "level_under_no_effect": {
            "p_clause_fires": level,
            "p_clause_passes": 1.0 - level,
            "note": (
                "a rule that withholds on >=10 of 12 adverse seeds almost never withholds "
                "on noise: f = 79/4096 = 0.0193"
            ),
        },
        "by_effect_size": rows,
        "clauses": clause_table(),
        "clauses_share_one_limit": True,
        "union_bound_level": (
            len(WITHHOLD_CLAUSES)
            * _binomial_tail(NULL_WIN_LIMIT, DECLARED_SEEDS, 0.5)
        ),
        "dev_fidelity_anchor": anchor,
        "anchor_reading": clause_reading(abs(anchor["z"])),
        "dev_fidelity_clause_read": dev_fidelity_clause_read(),
        "thresholds": {"planning": 0.80, "adequate": 0.50},
        "readings": {
            "one_rule_three_arms": (
                "all three withhold clauses are the same 10-of-12 sign test on the adaptive "
                "arm, differing only in the comparison arm (random, pool-matched control, "
                "null); each has level 0.0193 and the union is at most "
                f"{len(WITHHOLD_CLAUSES) * _binomial_tail(NULL_WIN_LIMIT, DECLARED_SEEDS, 0.5):.4f} "
                "by the union bound, less in fact because a hard block fires them together"
            ),
            "clauses_are_not_redundant": (
                "on section 5.40's dev-fidelity seeds the null clause fires (the null beats "
                "adaptive on 12/12) while the harm guard does not (the random arm beats it on "
                "7/12), so a verdict can withhold on one clause and clear another - and the "
                "pool-matched control arm has never been run beside them"
            ),
            "modal_verdict": (
                "with the dev-fidelity effect (z = 1.20) the clause fires with probability "
                f"{clause_reading(abs(anchor['z']))['p_clause_fires']:.4f}; since section 5.40 "
                "measured the null arm ahead on 12/12 fresh seeds, a withheld Gate 4 is the "
                "modal outcome and it is strong evidence, not a coin flip"
            ),
            "what_a_pass_is_worth": (
                f"a pass needs only {passes_required} of {DECLARED_SEEDS} seeds favouring the "
                "adaptive arm, and under no effect it happens with probability "
                f"{1.0 - level:.4f} - so a cleared null clause is not evidence that adaptive "
                "beats doing nothing, and the verdict's value claim then rests on the "
                "registered cost/band/control rules exactly as registered"
            ),
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = build_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
