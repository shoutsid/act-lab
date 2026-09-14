"""Toy policy ablation: does probe selection beat a loss-blind pool control?

Reproduces the substrate audit reported in ``RESEARCH_REPORT.md`` sections 9.4,
9.5 and 9.7. Three comparisons per candidate policy, all on the SAME seeds and
the SAME substrate:

- ``argmin vs uniform``   - the registered Gate 3 claim: probe-loss argmin over
  a capped candidate set against the random arm's uniform layer/operation churn.
- ``pool-random vs uniform`` - a loss-blind choice from the *same* probed
  candidates at the *same* probe cost, against uniform churn. Isolates the
  candidate pool's contribution.
- ``argmin vs pool-random`` (paired) - the controlled test of the mechanism:
  same candidates, same cost, only the decision rule differs.

The registered controller compares against uniform churn, which conflates the
candidate pool with the selection rule. At depth > 1 under the default
``prefix`` policy the capped pool is confined to the earliest layer(s), so the
first two rows can pass on the pool alone; the paired row is what separates
them.

Usage:
  PYTHONPATH=src python scripts/toy_policy_ablation.py
  PYTHONPATH=src python scripts/toy_policy_ablation.py --seeds 40000-40031
  PYTHONPATH=src python scripts/toy_policy_ablation.py --steps 48 --policies spread
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from act.evolution.experiment import ExperimentConfig, run_experiment  # noqa: E402
from act.evolution.study import _binomial_survival  # noqa: E402

# The audit's block. Disjoint from 1000-8000 (the depth-1 factor map), 9000-9011
# (the section 5.1 queue), 20000-20127 (the depth-4 policy matrix), 30000-30011
# (the reduced-dim V0 check) and 5000-6000 (cadence/variance studies).
DEFAULT_SEEDS = (40000, 40127)


def _parse_seeds(spec: str) -> tuple[int, ...]:
    """Accept ``A-B`` (inclusive range) or a comma-separated list."""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        start, end = spec.split("-", 1)
        return tuple(range(int(start), int(end) + 1))
    return tuple(int(value.strip()) for value in spec.split(",") if value.strip())


def _collect(
    seeds: tuple[int, ...],
    rule: str,
    base: dict,
) -> dict[int, tuple[float, float]]:
    """Per-seed (adaptive, random) final losses for one decision rule.

    One experiment trains both arms interleaved in a single step loop, so the
    uniform baseline comes free with each rule.

    NOTE (substrate audit, report section 9.8): the two arms share the global
    torch RNG, and a grow mutation draws fresh expert weights while a prune
    does not. Two runs whose treatments choose differently therefore consume
    different numbers of global draws, which perturbs the random arm's later
    rebuilds. The random arm is consequently only *approximately* the same
    baseline across rules, and the caller reports how often it differs rather
    than assuming independence.
    """
    losses: dict[int, tuple[float, float]] = {}
    for seed in seeds:
        result = run_experiment(
            ExperimentConfig(seed=seed, probe_selection=rule, **base)
        )
        arms = {arm.mode: arm.final_loss for arm in result.arms}
        losses[seed] = (arms["adaptive"], arms["random"])
    return losses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=f"{DEFAULT_SEEDS[0]}-{DEFAULT_SEEDS[1]}")
    parser.add_argument("--policies", default="prefix,spread")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--expert-count", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=24)
    parser.add_argument("--cap", type=int, default=4)
    parser.add_argument("--batch-count", type=int, default=1)
    parser.add_argument("--mutation-interval", type=int, default=8)
    parser.add_argument("--mutation-budget", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--independent-arm-rng",
        action="store_true",
        help="seed rebuild init per (arm, event) so the random baseline is the "
        "same control under every rule (report section 9.8; changes the "
        "historical shared-RNG numbers)",
    )
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    base = dict(
        depth=args.depth,
        hidden_size=args.hidden_size,
        expert_count=args.expert_count,
        steps=args.steps,
        mutation_interval=args.mutation_interval,
        mutation_budget=args.mutation_budget,
        probe_candidate_cap=args.cap,
        probe_batch_count=args.batch_count,
        device=args.device,
        arms="random,adaptive",
        independent_arm_rng=args.independent_arm_rng,
    )
    print(
        f"toy policy ablation: depth {args.depth}, hidden {args.hidden_size}, "
        f"{args.expert_count} experts/layer, cap {args.cap}/"
        f"{args.batch_count}-batch, steps {args.steps}, {len(seeds)} seeds"
    )
    for policy in (name.strip() for name in args.policies.split(",") if name.strip()):
        started = time.time()
        policy_base = {**base, "probe_candidate_policy": policy}
        argmin = _collect(seeds, "argmin", policy_base)
        pooled = _collect(seeds, "random", policy_base)
        # The random arm must be bit-identical across rules, or the paired
        # comparison would be measuring the shared baseline, not the rule.
        # How strongly the shared-RNG coupling perturbs the baseline: 0 would
        # mean the arms are independent and one uniform baseline serves every
        # rule exactly. Any nonzero count means the random arm is a slightly
        # different control in the two runs.
        decoupled = sum(argmin[s][1] == pooled[s][1] for s in seeds)
        trials = len(seeds)

        print(f"\npolicy={policy}  ({time.time() - started:.0f}s)")
        for label, losses in (
            ("argmin vs uniform", {s: argmin[s][0] for s in seeds}),
            ("pool-random vs uniform", {s: pooled[s][0] for s in seeds}),
        ):
            wins = sum(losses[s] < argmin[s][1] for s in seeds)
            mean_delta = statistics.mean(losses[s] - argmin[s][1] for s in seeds)
            print(
                f"  {label:22s} wins {wins:4d}/{trials}  mean_d {mean_delta:+.4f}  "
                f"p {_binomial_survival(wins, trials):.4g}"
            )

        pair = [argmin[s][0] - pooled[s][0] for s in seeds]
        pair_wins = sum(delta < 0 for delta in pair)
        print(
            f"  {'argmin vs pool-random':22s} wins {pair_wins:4d}/{trials}  "
            f"mean_d {statistics.mean(pair):+.4f}  "
            f"p {_binomial_survival(pair_wins, trials):.4g}  "
            f"(identical picks {sum(delta == 0.0 for delta in pair)}/{trials})"
        )
        print(
            f"  [coupling] random arm bit-identical across rules in "
            f"{decoupled}/{trials} seeds (shared global RNG)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
