"""Report on an architecture-search experiment from its ledger, at any point.

Reads ``stage*_seed*_results.jsonl`` from an experiment directory and produces
the ranked table, effect size, seed-matched contrasts, search-cost accounting and
decision. No training happens, so this works on a partial sweep, on a sweep that
was interrupted, and on a finished one.

Usage::

    PYTHONPATH=src python scripts/arch_zoo_analyze.py --experiment-id arch-zoo-100m-v1
    PYTHONPATH=src python scripts/arch_zoo_analyze.py --experiment-id probe --seeds 42,43,44
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from act.search.architecture_eval import load_results, print_results_table
from act.search.architecture_report import (
    HEAD_PAIRING_RULE,
    HORIZON_RULE,
    REPLICATION_RULE,
    best_available_summary,
    head_pairing_decision,
    horizon_decision,
    replication_decision,
    search_cost,
    with_structure,
    write_decision,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse an architecture-search experiment")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--seeds", default=None, help="comma-separated; default: discovered")
    parser.add_argument(
        "--also",
        default="",
        help=(
            "comma-separated additional experiment ids whose ledgers are pooled "
            "with this one. Used for a confirmation that re-runs the same "
            "architectures on fresh seeds: pooling them raises the seed count, "
            "and therefore the design's power floor, without the confirmation "
            "run overwriting this experiment's zoo definition and matching "
            "reports."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200_000,
        help=(
            "Monte-Carlo draws for the seed-matched test when the assignment space "
            "is too large to enumerate. The smallest p the procedure can report is "
            "1/(samples+1), so a design whose spread is enormous reports "
            "p<=5e-06 at the default while its own floor may be orders of "
            "magnitude lower. Raise this when the headline bound is the number "
            "being quoted."
        ),
    )
    parser.add_argument(
        "--signal-genes",
        default="",
        help=(
            "comma-separated structural genes to hold the by-length joint fit to. "
            "Default is to let each length select its own predictors, which can "
            "make a stable R2 look like a stable model when it is a different "
            "hypothesis each time. Name the genes a designed factorial already "
            "identified to ask the sharper question."
        ),
    )
    parser.add_argument(
        "--replication-rule",
        action="store_true",
        help=(
            "apply the crossed confirmation arm's pre-registered rule to the "
            "outcome and record which of its four branches the result lands on. "
            "Opt-in because the rule was declared for that arm's pooled seed "
            "count: run it on another design and it would label a question that "
            "design never pre-registered. Thresholds are in REPLICATION_RULE."
        ),
    )
    parser.add_argument(
        "--head-rule",
        action="store_true",
        help=(
            "apply the head-paired confirmation arm's pre-registered rule to the "
            "outcome and record which of its branches the result lands on. Opt-in "
            "for the same reason as --replication-rule: the rule was declared for "
            "that arm's pooled seed count, and applying it to another design would "
            "label a question the design never pre-registered. Thresholds are in "
            "HEAD_PAIRING_RULE."
        ),
    )
    parser.add_argument(
        "--horizon-rule",
        action="store_true",
        help=(
            "apply the horizon arm's pre-registered rule (does the depth sign survive "
            "the training length?) to the crossed contrast at the longest completed "
            "length and record its branch. Opt-in like the other rules; thresholds "
            "are in HORIZON_RULE."
        ),
    )
    parser.add_argument("--write", action="store_true", help="write ranked.json and decision.md")
    return parser.parse_args()


def cost_by_experiment(
    directories: Sequence[Path], candidates: int
) -> dict[str, dict]:
    """Runs, tokens and GPU-hours per source directory of a pooled analysis.

    Reported per experiment rather than in aggregate because a pooled analysis writes
    into the primary arm's directory: the totals there cover arms whose ledgers are
    elsewhere, and the sum is the one number a reader cannot decompose by looking at
    the directory it sits in.
    """
    split: dict[str, dict] = {}
    for source in directories:
        ledgers = sorted(source.glob("*_seed*_results.jsonl"))
        if not ledgers:
            continue
        per = search_cost(load_results(ledgers), candidates)
        # Keys deliberately identical to the cost dict's own, so a reader can compare
        # a share against the total without translating names - two spellings of the
        # same quantity in one file is how an audit misreads it.
        split[source.name] = {
            "runs_completed": per["runs_completed"],
            "tokens_trained": per["tokens_trained"],
            "wall_clock_hours": per["wall_clock_hours"],
        }
    return split


def ledger_precisions(source: Path) -> set[str]:
    """The autocast precisions one directory's ledgers recorded.

    ``run_config.mixed_precision`` travels with every row for exactly the reason
    this function exists: a loss measured under one precision is not comparable to
    a loss measured under another. A row written before the field existed records
    nothing, which is "unrecorded" rather than "the same as anything else".
    """
    values: set[str] = set()
    for ledger in sorted(source.glob("*_seed*_results.jsonl")):
        for line in ledger.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            values.add(
                str((row.get("run_config") or {}).get("mixed_precision") or "unrecorded")
            )
    return values


def main() -> None:
    args = parse_args()
    directory = Path(args.artifacts_root) / args.experiment_id
    # Matches both plan shapes: `stage1_seed42_results.jsonl` from successive
    # halving and `flat_seed42_results.jsonl` / `flat400_seed42_results.jsonl`
    # from the flat plan. Globbing `stage*` alone silently reported "no ledgers"
    # for a flat sweep - after all of its training had already been paid for.
    directories = [directory] + [
        Path(args.artifacts_root) / name.strip()
        for name in args.also.split(",")
        if name.strip()
    ]
    ledgers = sorted(
        ledger for source in directories for ledger in source.glob("*_seed*_results.jsonl")
    )
    if not ledgers:
        raise SystemExit(
            f"no result ledgers found in {directory} "
            f"(looked for *_seed*_results.jsonl)"
        )
    if len(directories) > 1:
        print(f"Pooling {len(directories)} experiment directories: "
              + ", ".join(str(source) for source in directories))
        # Pooling is only meaningful between ledgers measured the same way. The
        # horizon arm is fp16 and the record it is compared against is fp32, so
        # pooling them would produce an effect size that is part precision offset -
        # which is the comparison the arm's own anchor exists to make explicitly,
        # and the reason RunConfig records precision with every row.
        precisions = {source.name: ledger_precisions(source) for source in directories}
        measured = {value for values in precisions.values() for value in values}
        if len(measured) > 1:
            per_source = ", ".join(
                f"{name} {'/'.join(sorted(values)) if values else 'no rows'}"
                for name, values in sorted(precisions.items())
            )
            raise SystemExit(
                "refusing to pool ledgers measured in different precisions: "
                f"{per_source}. A loss measured in one precision is not comparable "
                "to a loss measured in another, and the pooled effect size would be "
                "part precision offset. Analyse each precision on its own, and "
                "compare them on the arm's own same-precision anchor."
            )

    # Structural genes live in the definition, not the ledgers, and every
    # structural axis is reported as "not recorded" without this join - which
    # silently chooses the leading axis from the cost axes alone.
    results = with_structure(
        load_results(ledgers), directory / "zoo_definition.json"
    )
    print(f"Experiment: {args.experiment_id}")
    print(f"Ledgers: {len(ledgers)}  runs: {len(results)}")

    seeds = (
        [int(s) for s in args.seeds.split(",")]
        if args.seeds
        else sorted({result.seed for result in results})
    )
    print(f"Seeds: {seeds}")

    lengths = sorted({result.steps for result in results})
    print(f"Training lengths completed: {lengths}\n")

    for length in lengths:
        print(f"--- {length} steps ---")
        print_results_table([r for r in results if r.steps == length])

    signal_genes = [
        gene.strip() for gene in args.signal_genes.split(",") if gene.strip()
    ] or None
    outcome = best_available_summary(
        results, seeds, samples=args.samples, signal_genes=signal_genes
    )
    if args.replication_rule:
        # Computed from the same outcome the decision renders, so the branch and
        # the evidence it was read from cannot drift apart.
        outcome["replication_decision"] = replication_decision(outcome)
    if args.head_rule:
        outcome["head_pairing_decision"] = head_pairing_decision(outcome)
    if args.horizon_rule:
        outcome["horizon_decision"] = horizon_decision(outcome)
    candidates = 0
    definition = directory / "zoo_definition.json"
    if definition.exists():
        candidates = len(json.loads(definition.read_text()).get("candidates", []))

    cost = search_cost(results, candidates)
    print(f"{'=' * 78}")
    print(
        f"Search cost: {cost['total_search_flops']:.3e} FLOPs over "
        f"{cost['tokens_trained'] / 1e6:.1f}M tokens, {cost['wall_clock_hours']:.2f} GPU-hours"
    )
    print(f"  by stage: {cost['by_stage']}")
    print(f"{'=' * 78}")

    if outcome.get("available"):
        print(f"Best at {outcome['steps']} steps: {outcome['best']} ({outcome['best_mean_val_loss']:.4f})")
        units = outcome.get("sigmas")
        units_text = "undefined (single-seed)" if units is None else f"{units:.2f}"
        print(
            f"Spread: {outcome['architecture_spread']:.4f} nats = {units_text} x "
            f"within-seed std ({outcome['within_seed_std']:.4f})"
        )
        print(f"Runs per architecture: {sorted({r['runs'] for r in outcome['ranked']})}")
        for row in outcome.get("paired_vs_best", []):
            print(
                f"  {row['candidate']:<42} {row['mean_delta_vs_best']:+.4f} "
                f"over seeds {row['paired_seeds']}"
            )
        # Spread shrinking says the effect gets smaller with training; it does not
        # say whether its *cause* is still readable at the shorter length. That is
        # the number a search budget depends on, because screening at 400 steps
        # costs about a third of confirming at 1200.
        raw_signal = outcome.get("structural_signal_by_length") or {}
        signal = {
            k: v
            for k, v in raw_signal.items()
            if k not in ("agreement", "model") and v.get("available")
        }
        if len(signal) >= 2:
            model_genes = (raw_signal.get("model") or {}).get("pre_specified")
            heading = "Structural signal by training length (common architecture set)"
            if model_genes:
                heading += f", pre-specified model {', '.join(model_genes)}"
            print(f"\n{heading}:")
            for length, row in signal.items():
                lead = row["genes"][0] if row["genes"] else None
                lead_text = (
                    "no gene fitted" if lead is None
                    else f"lead {lead['gene']} beta={lead['beta']:+.3f} p={lead['p']:.4g}"
                )
                print(
                    f"  {length:>5} steps  R2={row['r_squared']:.3f}  "
                    f"p={row['p_model']:.4g}  {lead_text}"
                )
            agreement = raw_signal.get("agreement") or {}
            if agreement:
                if model_genes:
                    print(
                        "  the model is fixed across lengths, so these are the "
                        "same coefficients at every length rather than the same "
                        "R2 from a different model."
                    )
                else:
                    print(
                        "  same leading gene at every length: "
                        f"{'yes' if agreement['same_leading_gene'] else 'NO'}; "
                        "the leading gene is selected per length, so treat its own "
                        "p as a consistency check, not a designed test. Pass "
                        "--signal-genes to hold the model fixed."
                    )

        # The two designs that answer *which gene* rather than *whether*, printed
        # to stdout as well as into decision.md: a reader running the command should
        # not have to open a file to see the analysis the design was built for.
        crossed = outcome.get("crossed_contrast") or {}
        if crossed.get("available"):
            print(f"\nCrossed design ({crossed['factor_a']} x {crossed['factor_b']}):")
            dropped = crossed.get("seeds_dropped") or []
            if dropped:
                # Stated here as well as in decision.md: it changes what the main
                # effects are averages over, and the floors scale with the count.
                print(
                    f"  over seeds {crossed['seeds']}; dropped {dropped} "
                    "(incomplete, reported rather than withheld)"
                )
            for gene, main in crossed["main"].items():
                p_text = "n/a" if main["p"] is None else f"{main['p']:.4g}"
                floor_text = " (at the procedure floor)" if main.get("at_floor") else ""
                print(
                    f"  {gene:<16} {main['nats']:+.4f} nats  p={p_text} "
                    f"({main['method']}, {main['assignments']} assignments)"
                    f"{floor_text}"
                )
                levels = ", ".join(
                    f"{level}: {row['nats']:+.4f}"
                    for level, row in main["by_other_level"].items()
                )
                print(f"    by level of the other gene: {levels}")
                if main.get("interaction_nats") is not None:
                    ip = main.get("interaction_p_seed_blocked")
                    floor = main.get("interaction_p_seed_blocked_floor")
                    floor_text = "" if floor is None else f" (floor {floor:.4g})"
                    print(
                        f"    interaction {main['interaction_nats']:+.4f} nats, "
                        f"seed-blocked p="
                        + ("n/a" if ip is None else f"{ip:.4g}")
                        + floor_text
                    )
            print(
                "  Note: the re-split interaction test assumes no main effect too, "
                "so it is conservative; the seed-blocked test is the calibrated one."
            )

        # The joint fit answers the question the per-axis table cannot: on a set
        # the generator did not orthogonalise, does a gene still carry signal once
        # its partners are in the model? For an observational set like v1's
        # curated families this is the difference between "hidden size tracks
        # loss" and "hidden size and depth jointly explain 92% of it".
        fit = outcome.get("joint_structural_fit") or {}
        if fit.get("available"):
            print(
                f"\nJoint structural fit (n={fit['architectures']}, "
                f"R2={fit['r_squared']:.3f}, p={fit['p_model']:.4g}):"
            )
            for row in fit["genes"]:
                print(
                    f"  {row['gene']:<18} beta={row['beta']:+.3f}  p={row['p']:.4g}  "
                    f"range {row['min']:g}..{row['max']:g}"
                )
            for row in fit.get("dropped", []):
                print(
                    f"  {row['gene']:<18} dropped: identical to {row['duplicate_of']} "
                    f"(rho={row['spearman']:+.4f})"
                )
            if fit.get("entangled"):
                print(f"  entangled: {fit['caveat']}")

        # The level split, because a pooled paired contrast is an average over the
        # levels of everything it holds fixed and two opposite effects cancel.
        for gene, splits in sorted((outcome.get("paired_level_splits") or {}).items()):
            for partner, split in sorted(splits.items()):
                print(f"\nPaired level split ({gene} at each level of {partner}):")
                for level, contrast in sorted(
                    split["levels"].items(), key=lambda item: float(item[0])
                ):
                    p_text = "n/a" if contrast["p"] is None else f"{contrast['p']:.4g}"
                    print(
                        f"  {partner}={level}: {contrast['mean_difference']:+.4f} nats "
                        f"p={p_text} floor {contrast['min_achievable_p']:.4g} "
                        f"({contrast['pairs']} units, {contrast['negative']} negative / "
                        f"{contrast['positive']} positive)"
                    )
                print(f"  {split['reading']}")

        for gene, contrast in sorted((outcome.get("paired_contrasts") or {}).items()):
            p_text = "n/a" if contrast["p"] is None else f"{contrast['p']:.4g}"
            print(f"\nPaired contrast ({gene}, everything else held fixed):")
            print(
                f"  {contrast['pairs']} pairs at levels {contrast['levels']}: "
                f"{contrast['mean_difference']:+.4f} nats"
            )
            print(
                f"  p={p_text} ({contrast['method']}, {contrast['assignments']} sign "
                f"vectors), floor {contrast['min_achievable_p']:.4g}, "
                f"{contrast['negative']} negative / {contrast['positive']} positive"
            )
            if contrast.get("dropped_groups"):
                print(
                    f"  groups dropped: {len(contrast['dropped_groups'])} on seed(s) "
                    f"{contrast['seeds_dropped']} (rows the sweep has not written)"
                )
            for row in contrast.get("seeds_partial") or []:
                print(
                    f"  seed {row['seed']} contributes {row['pairs']} of "
                    f"{row['of']} pairs (rows the sweep has not written)"
                )
    else:
        print("No completed stage carried more than one run yet.")

    for key, heading, thresholds in (
        ("replication_decision", "replication", REPLICATION_RULE),
        ("head_pairing_decision", "head-pairing", HEAD_PAIRING_RULE),
        ("horizon_decision", "horizon", HORIZON_RULE),
    ):
        rule = outcome.get(key) or {}
        if not rule:
            continue
        print(f"\nPre-registered {heading} rule (thresholds {thresholds}):")
        print(f"  {rule['label']}")
        if rule.get("available"):
            for row in rule["conditions"]:
                print(
                    f"    {'ok ' if row['met'] else 'NO '} {row['condition']}: "
                    f"{row['value']}"
                )
            if rule.get("note"):
                print(f"  {rule['note']}")
        else:
            print(f"  not applied: {rule.get('reason')}")

    if args.write:
        if len(directories) > 1:
            # A pooled write lands in the primary experiment's directory while
            # covering every pooled one, so its totals are larger than the arm they
            # sit beside: the crossed arm's cost file reads 6.53 GPU-hours next to
            # ledgers holding 3.28 of them. Without the split, an auditor reading
            # that file attributes the confirmation arm's hours to the crossed arm -
            # which is exactly the double-count this field exists to prevent.
            split = cost_by_experiment(directories, candidates)
            cost = {
                **cost,
                "pooled_experiments": [source.name for source in directories],
                "by_experiment": split,
                "pooled_note": (
                    "Pooled analysis: the totals above span every experiment in "
                    "pooled_experiments and are written into the primary one's "
                    "directory. Read by_experiment for each arm's own share."
                ),
            }
        (directory / "ranked.json").write_text(json.dumps(outcome, indent=2))
        (directory / "search_cost.json").write_text(json.dumps(cost, indent=2))
        write_decision(directory / "decision.md", outcome)
        print(f"\nWrote ranked.json, search_cost.json, decision.md to {directory}")
        if len(directories) > 1:
            print("Cost by experiment (the pooled total above is their sum):")
            for name, row in cost["by_experiment"].items():
                print(f"  {name:<40} {row['runs_completed']:>3} runs  "
                      f"{row['wall_clock_hours']:>6.2f} GPU-hours")


if __name__ == "__main__":
    main()
