"""Decide the §11 random-vs-manual comparison from two ledgers, mechanically.

The comparison this serves is §9a's: two *sets* of architectures under equal
compute — the phase's curated (manual) set re-run flat, against a budget-matched
random draw of the same size — and the question is whether curation carries
set-mean signal at this scale. The statistic is `two_set_blocked_test` (the unit
of exchange is the set label, not one run), and the decision is
`set_comparison_decision`, whose thresholds live in `SET_COMPARISON_RULE` and
were declared in the findings record **before** any ledger of this comparison
existed — the same discipline the head-paired confirmation arm held itself to.

The ``best`` and ``worst`` statistics are computed and reported beside the
decision, because §9a measured that the winner form is invisible at every size
this phase can afford (a permutation null on the floors already contains it) and
they are therefore secondaries: a significant ``best`` result would be a lead to
price, not a branch of the rule.

Usage::

    PYTHONPATH=src python scripts/set_comparison_analyze.py \
        --manual-experiment-id arch-zoo-100m-v11-manual \
        --random-experiment-id arch-zoo-100m-v11-random \
        --steps 1200 --write
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from act.search.architecture_eval import load_results
from act.search.architecture_report import (
    reading_stability_lines,
    SET_COMPARISON_RULE,
    set_comparison_decision,
    two_set_blocked_test,
)

_GENES = re.compile(
    r"_D(?P<depth>\d+)_H(?P<hidden>\d+)_E(?P<experts>\d+)_A(?P<active>\d+)_W(?P<width>\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manual-experiment-id", required=True)
    parser.add_argument("--random-experiment-id", required=True)
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument(
        "--steps",
        type=int,
        default=1200,
        help="training length the comparison reads at; effect sizes are horizon-dependent",
    )
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument(
        "--statistic",
        default="mean",
        choices=["mean", "best", "worst"],
        help=(
            "which statistic the rule reads; mean is the pre-registered primary "
            "because §9a measured the other two forms to be invisible at this size"
        ),
    )
    parser.add_argument(
        "--budget-ratio",
        type=float,
        default=None,
        help=(
            "manual set total over random set total; default computes it from the "
            "two zoo definitions, which is the §9a-matched quantity"
        ),
    )
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def load_zoo(root: Path, experiment_id: str) -> dict:
    return json.loads((root / experiment_id / "zoo_definition.json").read_text())


def candidate_names(zoo: dict) -> list[str]:
    return sorted(candidate["name"] for candidate in zoo["candidates"])


def structural_summary(zoo: dict) -> dict:
    """Descriptive per-set gene means, parsed from the candidate names.

    A descriptive companion to the decision, not a test: §9a's predicted mechanism
    is coverage, and this is what lets a reader see which genes the two sets
    covered differently without re-deriving it.
    """
    genes = {"depth": [], "hidden": [], "experts": [], "active": [], "width": []}
    for candidate in zoo["candidates"]:
        match = _GENES.search(candidate["name"])
        if match:
            for key in genes:
                genes[key].append(int(match.group(key)))
    totals = [candidate["total_params"] for candidate in zoo["candidates"]]
    summary = {
        "candidates": len(totals),
        "total_params": sum(totals),
        "mean_total_params": sum(totals) / len(totals),
    }
    for key, values in genes.items():
        if values:
            summary[f"mean_{key}"] = sum(values) / len(values)
    return summary


def render_markdown(
    *,
    manual_id: str,
    random_id: str,
    steps: int,
    primary: dict,
    decision: dict,
    outcomes: dict[str, dict],
    manual_summary: dict,
    random_summary: dict,
    budget_ratio: float,
) -> str:
    lines = [
        "# §11: random vs manual, decided by the pre-registered rule",
        "",
        f"Manual set: `{manual_id}`. Random draw: `{random_id}`. "
        f"Read at {steps} steps; quote every effect size with that length.",
        "",
        f"Budget ratio (manual/random): **{budget_ratio:.4f}** — "
        f"{manual_summary['mean_total_params'] / 1e6:.2f}M against "
        f"{random_summary['mean_total_params'] / 1e6:.2f}M per candidate. "
        f"The rule requires parity within "
        f"{SET_COMPARISON_RULE['budget_deviation']:.1%}.",
        "",
        "## Per-set structure (descriptive — the coverage mechanism §9a predicted)",
        "",
        "| set | candidates | mean total | mean depth | mean hidden | mean width |",
        "|---|---|---|---|---|---|",
        (
            f"| manual | {manual_summary['candidates']} "
            f"| {manual_summary.get('mean_total_params', 0) / 1e6:.1f}M "
            f"| {manual_summary.get('mean_depth', float('nan')):.1f} "
            f"| {manual_summary.get('mean_hidden', float('nan')):.0f} "
            f"| {manual_summary.get('mean_width', float('nan')):.0f} |"
        ),
        (
            f"| random | {random_summary['candidates']} "
            f"| {random_summary.get('mean_total_params', 0) / 1e6:.1f}M "
            f"| {random_summary.get('mean_depth', float('nan')):.1f} "
            f"| {random_summary.get('mean_hidden', float('nan')):.0f} "
            f"| {random_summary.get('mean_width', float('nan')):.0f} |"
        ),
        "",
        f"## Primary ({primary['statistic']})",
        "",
        f"**{decision['label']}**",
        "",
        decision["reading"],
        "",
        "| Condition | Value | Met |",
        "|---|---|---|",
    ]
    for row in decision["conditions"]:
        lines.append(f"| {row['condition']} | {row['value']} | {'yes' if row['met'] else 'NO'} |")
    lines.append("")
    if decision.get("note"):
        lines += [decision["note"], ""]
    # The same table the zoo decision renders, from the same function: the §11
    # comparison's branch is a threshold on a loss like every other rule's, and a second
    # spelling of the table is how two arms end up describing one check differently.
    if decision.get("reading_stability"):
        lines += reading_stability_lines(
            decision["reading_stability"], decision["label"]
        )
    lines += ["## All three statistics", ""]
    for name in ("mean", "best", "worst"):
        outcome = outcomes[name]
        if not outcome.get("available"):
            lines.append(f"- **{name}**: unavailable — {outcome.get('reason', '')}")
            continue
        caveat = (
            ""
            if name == "mean"
            else " (secondary: invisible at this size per §9a — a lead to price, not a branch)"
        )
        lines.append(
            f"- **{name}**: {outcome['difference_nats']:+.4f} nats, "
            f"p = {outcome['p']:.4g} ({outcome['method']}), "
            f"sign-consistent {outcome['sign_consistency']}, "
            f"{len(outcome['seeds'])} seeds{caveat}"
        )
    lines += [
        "",
        outcome_note(primary),
        "",
    ]
    return "\n".join(lines)


def outcome_note(outcome: dict) -> str:
    if not outcome.get("available"):
        return ""
    return (
        f"Procedure: {outcome['method']} over {outcome['assignments']:,} member-level "
        f"assignments; the design's own floor is {outcome['min_achievable_p']:.3g} "
        + (
            f", the Monte-Carlo resolution is {outcome['p_resolution']:.3g}."
            if outcome.get("p_resolution")
            else "."
        )
    )


def main() -> None:
    args = parse_args()
    root = Path(args.artifacts_root)
    manual_zoo = load_zoo(root, args.manual_experiment_id)
    random_zoo = load_zoo(root, args.random_experiment_id)
    manual_names = candidate_names(manual_zoo)
    random_names = candidate_names(random_zoo)

    manual_total = sum(c["total_params"] for c in manual_zoo["candidates"])
    random_total = sum(c["total_params"] for c in random_zoo["candidates"])
    budget_ratio = (
        args.budget_ratio if args.budget_ratio is not None else manual_total / random_total
    )

    results = load_results(
        sorted((root / args.manual_experiment_id).glob("*_seed*_results.jsonl"))
    ) + load_results(sorted((root / args.random_experiment_id).glob("*_seed*_results.jsonl")))

    outcomes = {
        statistic: two_set_blocked_test(
            results,
            args.steps,
            (("manual", manual_names), ("random", random_names)),
            statistic=statistic,
            samples=args.samples,
        )
        for statistic in ("mean", "best", "worst")
    }
    def rebuild(rows) -> dict:
        """The same rule over the same rows with a different loss substituted.

        This comparison's evidence is a two-set test built from both arms' rows rather
        than one of the zoo outcome's own blocks, so it hands the reconstruction to
        ``set_comparison_decision`` instead of asking it to rebuild a block it does not
        have. Everything else about the reading - the fields substituted, the thresholds,
        the branch logic - is the shared path the other four rules use.
        """
        return set_comparison_decision(
            two_set_blocked_test(
                rows,
                args.steps,
                (("manual", manual_names), ("random", random_names)),
                statistic=args.statistic,
                samples=args.samples,
            ),
            budget_ratio=budget_ratio,
        )

    decision = set_comparison_decision(
        outcomes[args.statistic],
        results,
        rebuild=rebuild,
        samples=args.samples,
        budget_ratio=budget_ratio,
    )

    manual_summary = structural_summary(manual_zoo)
    random_summary = structural_summary(random_zoo)
    markdown = render_markdown(
        manual_id=args.manual_experiment_id,
        random_id=args.random_experiment_id,
        steps=args.steps,
        primary=outcomes[args.statistic],
        decision=decision,
        outcomes=outcomes,
        manual_summary=manual_summary,
        random_summary=random_summary,
        budget_ratio=budget_ratio,
    )

    print(markdown)
    if args.write:
        target = root / args.manual_experiment_id
        (target / "set_comparison.json").write_text(
            json.dumps(
                {
                    "manual_experiment": args.manual_experiment_id,
                    "random_experiment": args.random_experiment_id,
                    "steps": args.steps,
                    "budget_ratio": budget_ratio,
                    "primary_statistic": args.statistic,
                    "decision": decision,
                    "outcomes": outcomes,
                    "manual_structure": manual_summary,
                    "random_structure": random_summary,
                    "rule": SET_COMPARISON_RULE,
                },
                indent=2,
            )
            + "\n"
        )
        (target / "set_comparison_decision.md").write_text(markdown)
        print(f"\nWrote set_comparison.json, set_comparison_decision.md to {target}")


if __name__ == "__main__":
    main()
