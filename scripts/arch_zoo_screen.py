"""Architecture zoo screening: successive halving over matched-budget genomes.

Answers one question cheaply: do structurally different architectures with
matched budgets produce reproducible differences in learning efficiency at
~100M parameters, on real text at context 1024?

Pipeline::

    generate zoo
        -> matching + distance report
        -> stage 1 (all candidates, short)
        -> keep best half
        -> stage 2 (longer)
        -> keep best 3
        -> stage 3 (multi-seed confirmation)

Usage::

    PYTHONPATH=src python scripts/arch_zoo_screen.py --experiment-id arch-zoo-100m-v1
    PYTHONPATH=src python scripts/arch_zoo_screen.py --dry-run      # report only, no GPU

Every result is appended to ``results.jsonl`` as it finishes, so an interrupted
sweep keeps the runs that already completed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from itertools import product
from pathlib import Path

from act.search.architecture_eval import (
    EvalResult,
    RunConfig,
    evaluate_batch,
    print_results_table,
)
from act.search.architecture_report import (
    best_available_summary,
    rank,
    search_cost,
    verdict,
    write_decision,
)
from act.search.architecture_zoo import (
    ArchitectureCandidate,
    compute_distance_matrix,
    crossing_balance,
    generate_architecture_zoo,
    generate_crossed_zoo,
    generate_head_paired_zoo,
    generate_matched_zoo,
    head_pair_matching,
    matching_quality,
    min_pairwise_distance,
    print_zoo_summary,
)
from act.search.random_architecture import RandomArchitectureSearch


#: Measured per-architecture seed std in mean held-out loss, by screening length:
#: 0.0286 nats at 400 steps and 0.0165 at 800, from arch-zoo-100m-v2-horizon's
#: matched eight-candidate set (8 candidates x 3 seeds), and 0.0148 at 1200 from the
#: same candidates' seeds pooled with v2. These are the `seed std` column of findings
#: section 4 and the values `spread_by_length` reports.
#:
#: They are *stds*. A promotion decision does not need the std, it needs the standard
#: error of the difference between two candidates' screen means, which is
#: `std * sqrt(2 / seeds)` - see `comparison_noise_nats`. The phase's screening
#: threshold was pinned at 0.018 nats, a number this set does not produce under any
#: definition: its 400-step std is 0.0286 and its single-mean standard error 0.0165,
#: so 0.018 is 63% of what a two-seed screen has to clear.
SCREEN_SEED_STD_BY_STEPS: dict[int, float] = {400: 0.0286, 800: 0.0165, 1200: 0.0148}
SCREEN_SEED_STD_NATS = SCREEN_SEED_STD_BY_STEPS[400]


def seed_std_for_steps(steps: int) -> float:
    """The measured seed std at a screening length, or the nearest measured one.

    An unmeasured length takes the nearest measured length *at or below* it, which is
    the conservative direction: the std falls with training, so borrowing a longer
    stage's smaller std would understate the noise at a shorter stage.
    """
    if steps in SCREEN_SEED_STD_BY_STEPS:
        return SCREEN_SEED_STD_BY_STEPS[steps]
    below = [known for known in SCREEN_SEED_STD_BY_STEPS if known <= steps]
    return SCREEN_SEED_STD_BY_STEPS[max(below)] if below else SCREEN_SEED_STD_NATS


def comparison_noise_nats(
    seeds: int, steps: int = 400, *, std: float | None = None
) -> float:
    """The gap two candidates' screen means must clear to count as separated.

    The margin at a promotion cut is a difference of two means, each over `seeds`
    runs, so its standard error is `std * sqrt(2 / seeds)`: identical to the std for a
    two-seed screen, and 0.82x the std for three. Multiplying the *measured* std by
    the seed count is what keeps the threshold tied to the design that produced the
    number, instead of to a constant whose provenance could not be reproduced.
    """
    if seeds < 1:
        raise ValueError("a promotion comparison needs at least one seed per candidate")
    measured = seed_std_for_steps(steps) if std is None else std
    return measured * math.sqrt(2.0 / seeds)


def screen_noise(args: argparse.Namespace, seeds: int, steps: int) -> float:
    """The promotion threshold in use: the flag if given, else the derived one."""
    if args.screen_noise_nats is not None:
        return float(args.screen_noise_nats)
    return comparison_noise_nats(seeds, steps)


def promotion_margins(
    ranked: list[tuple[str, float, int]], keep: int, noise: float
) -> dict:
    """Which promotions the screening stage cannot resolve, and the gap at the cut.

    ``ranked`` is one stage's ranking, best validation loss first. The margin that
    matters is the one *at the cut* - the gap between the last candidate the plan
    keeps and the first it drops - because that is the pair whose separation is
    being asserted. If the gap is inside the seed noise the two runs did not
    separate, and promoting one while discarding the other is elimination rather
    than adjudication. That distinction needs saying out loud: the dropped
    candidate is never trained again, so a tie-break at the cut is a decision the
    search cannot revisit.
    """
    if keep <= 0 or keep >= len(ranked):
        return {
            "available": False,
            "reason": "the cut falls at an end of the ranking, so nothing is decided",
            "borderline": [],
        }
    cut = ranked[keep][1]
    margin = cut - ranked[keep - 1][1]
    return {
        "available": True,
        "kept": [name for name, _, _ in ranked[:keep]],
        "dropped": [name for name, _, _ in ranked[keep:]],
        # The two candidates the cut *decides between*: the last one kept and the
        # first one dropped. Their gap is the margin being tested.
        "cut_pair": [
            {"name": ranked[keep - 1][0], "loss": ranked[keep - 1][1]},
            {"name": ranked[keep][0], "loss": cut},
        ],
        "cut_margin": margin,
        "resolvable": margin >= noise,
        "noise": noise,
        # Anything else inside the noise of the cut loss could equally have sat in
        # the cut pair, so a tie-break is choosing between these too. The candidate
        # *at* the cut is excluded because its own gap is zero by construction and
        # would otherwise appear in every report.
        "borderline": [
            {"name": name, "loss": loss, "gap_to_cut": abs(loss - cut)}
            for index, (name, loss, _) in enumerate(ranked)
            if index != keep and abs(loss - cut) < noise
        ],
    }


def report_promotion(
    label: str,
    ranked: list[tuple[str, float, int]],
    keep: int,
    *,
    steps: int,
    noise: float,
    seeds: int | None = None,
) -> dict:
    """Print whether a stage's cut is resolvable, and what sits inside the noise.

    The seed count is printed with the threshold because the two are not separable:
    the threshold is the standard error of a difference between two means, so the same
    0.0286-nat std gives a 0.0286-nat threshold at two seeds and 0.02335 at three. A
    reader who sees only the nats cannot tell which design produced them.
    """
    margins = promotion_margins(ranked, keep, noise)
    over = "" if seeds is None else f" over {seeds} seeds each"
    if not margins["available"]:
        return margins
    if margins["resolvable"]:
        print(
            f"{label} promotion: cut margin {margins['cut_margin']:.4f} nats clears "
            f"the {noise:.4f}-nat threshold{over} at the {steps}-step screen."
        )
        return margins
    low, high = margins["cut_pair"]
    print(
        f"{label} promotion: cut margin {margins['cut_margin']:.4f} nats is INSIDE "
        f"the {noise:.4f}-nat threshold{over} at the {steps}-step screen, so these runs "
        f"did not separate `{low['name']}` ({low['loss']:.4f}) from "
        f"`{high['name']}` ({high['loss']:.4f}) - screening eliminates, it does "
        "not adjudicate."
    )
    if margins["borderline"]:
        print(
            "  within the noise of the cut: "
            + ", ".join(
                f"`{row['name']}` ({row['gap_to_cut']:.4f})"
                for row in margins["borderline"]
            )
        )
        print(
            "  A dropped candidate here is never trained again, so if the search "
            "needs this tie decided, promote the whole borderline set rather than "
            "letting a cut decide it."
        )
    return margins


def promoted_names(
    ranked: list[tuple[str, float, int]],
    keep: int,
    *,
    noise: float,
    promote_ties: bool = False,
) -> dict:
    """Which candidates a screening stage advances, in ranking order.

    Normally that is simply the top `keep`. With `promote_ties` it is instead the
    top `keep` *plus* every candidate inside the noise of the cut loss, which is
    the concrete form of "screening eliminates, it does not adjudicate": when the
    cut sits inside the seed noise the two runs did not separate, so the tie is
    handed to the confirmation stage rather than resolved by a margin the data
    cannot support. It only ever *adds* candidates, and only when the cut is
    unresolvable, so it cannot turn a clean cut into a different one.

    The candidate *at* the cut is always added, even though the report omits it
    from the borderline list (its own gap is zero, so it would appear in every
    report). For promotion it is the one name that cannot be left out: the cut
    lands on it, so advancing every candidate near the cut but not that one would
    still drop the candidate the unresolvable margin was about.
    """
    margins = promotion_margins(ranked, keep, noise)
    chosen = {name for name, _, _ in ranked[:keep]}
    promoted_ties: list[str] = []
    if promote_ties and margins.get("available") and not margins["resolvable"]:
        tied = {row["name"] for row in margins["borderline"]} | {ranked[keep][0]}
        promoted_ties = [name for name, _, _ in ranked if name in tied and name not in chosen]
        chosen |= tied
    return {
        "names": [name for name, _, _ in ranked if name in chosen],
        "promoted_ties": promoted_ties,
        "margins": margins,
    }


def _is_sweep_invocation(cmdline: str) -> bool:
    """True for a python process running the zoo driver, but not for its wrappers.

    A queued launch is usually wrapped in a shell (`bash -c '... arch_zoo_screen.py
    ...'`) or an external waiter (`while pgrep -f arch_zoo_screen.py`), and those
    cmdlines name the script without being a sweep. Requiring the interpreter to
    be the process is what keeps the warning from naming the queue as competition.
    """
    parts = cmdline.split()
    if not parts:
        return False
    executable = parts[0].rsplit("/", 1)[-1]
    return executable.startswith("python") and any(
        part.endswith("arch_zoo_screen.py") for part in parts[1:]
    )


def competing_sweeps(
    listing: "list[tuple[int, str]] | None" = None, *, exclude: int | None = None
) -> list[int]:
    """Other zoo sweeps running right now, so a shared GPU is visible at launch.

    A sweep that shares the GPU is invisible from inside it: candidates simply take
    proportionally longer, and the budget guard cannot see that because it counts
    tokens rather than wall clock. Today's head-paired arm spent three hours on a
    single candidate for exactly this reason, which reads as a hang from the log.

    Linux only. Elsewhere this returns nothing rather than guessing, and `listing`
    can be passed directly so the matching is testable without any processes.
    """
    if listing is None:
        proc = Path("/proc")
        if not proc.is_dir():
            return []
        listing = []
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                raw = (entry / "cmdline").read_bytes()
            except OSError:  # the process exited between listing and reading
                continue
            listing.append((int(entry.name), raw.replace(b"\0", b" ").decode("utf-8", "replace")))
    me = os.getpid() if exclude is None else exclude
    return [
        pid for pid, cmdline in listing if pid != me and _is_sweep_invocation(cmdline)
    ]


def warn_if_gpu_shared(competing: "list[int]") -> bool:
    """Print the shared-GPU warning, and say whether there was one to print."""
    if not competing:
        return False
    print(
        f"WARNING: {len(competing)} other zoo sweep(s) already running "
        f"(pid {', '.join(str(pid) for pid in competing)}). They share this GPU, "
        "so every candidate below costs proportionally more wall clock, and the "
        "token budget guard will not notice because it counts tokens rather than "
        "hours. Expect the candidates-per-hour rate in the log, not the phase's "
        "published rate."
    )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Architecture zoo screening")
    parser.add_argument("--experiment-id", default="arch-zoo-100m-v1")
    parser.add_argument(
        "--design",
        default="families",
        choices=[
            "families",
            "matched-total",
            "matched-active",
            "matched-flops",
            "crossed",
            "head-paired",
            "random",
        ],
        help=(
            "families: wide structural zoo around a parameter budget. "
            "matched-*: hold one cost axis fixed and spread the other "
            "(brief section 21) - total parameters, active parameters, or "
            "per-token FLOPs. "
            "crossed: hold total parameters fixed and cross two structural "
            "genes in a balanced factorial, so the pair is unconfounded and "
            "an attribution to either is identifiable. "
            "random: uniformly sampled baseline from the same space "
            "(brief section 11)."
        ),
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=14,
        help="how many architectures one draw returns",
    )
    parser.add_argument(
        "--random-tolerance",
        type=float,
        default=0.15,
        help="parameter budget band for the random baseline",
    )
    parser.add_argument(
        "--match-target",
        type=float,
        default=None,
        help="value of the matched axis; defaults per design",
    )
    parser.add_argument(
        "--match-tolerance",
        type=float,
        default=0.03,
        help="tight matching band for a matched design",
    )
    parser.add_argument(
        "--max-total-params",
        type=int,
        default=150_000_000,
        help="memory ceiling for a matched design, so it stays runnable",
    )
    parser.add_argument(
        "--only",
        default="",
        help=(
            "comma-separated exact zoo names to run, e.g. for confirming the "
            "finalists of an earlier sweep on fresh seeds without re-running the "
            "architectures that were already screened out. Unknown names are an "
            "error rather than a silent no-op, so a typo cannot quietly shrink an "
            "experiment."
        ),
    )
    parser.add_argument("--target-params", type=int, default=100_000_000)
    parser.add_argument("--tolerance", type=float, default=0.3)
    parser.add_argument("--context-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--mixed-precision",
        default="off",
        choices=["off", "fp16", "bf16"],
        help=(
            "autocast precision for every run in the sweep, recorded in each ledger "
            "row's run_config. The phase's record is fp32 (`off`); a loss measured in "
            "one precision is not comparable to one measured in another, so an arm "
            "that trains fp16 has to carry its own same-precision anchor at any "
            "length it is compared against. fp16 is native on Turing (T4) tensor "
            "cores and measured 2.56x fp32 there; on Pascal it is slower than fp32."
        ),
    )
    parser.add_argument("--stage1-steps", type=int, default=400)
    parser.add_argument("--stage2-steps", type=int, default=800)
    parser.add_argument("--stage3-steps", type=int, default=1200)
    parser.add_argument("--screen-seeds", default="42,43")
    parser.add_argument("--finalist-seeds", default="42,43,44")
    parser.add_argument("--keep-stage2", type=int, default=6)
    parser.add_argument("--keep-stage3", type=int, default=3)
    parser.add_argument(
        "--promote-ties",
        action="store_true",
        help=(
            "for --plan screen: when a stage's cut margin is inside the seed "
            "noise, advance the whole borderline set rather than letting an "
            "unresolvable gap decide which candidate is never trained again. "
            "Widens the confirmation stage only when the screen could not "
            "separate the pair it is cutting between."
        ),
    )
    parser.add_argument(
        "--flat-steps",
        default="",
        help=(
            "comma-separated training lengths for --plan flat, e.g. 400,1200. "
            "Defaults to --stage3-steps. Running several lengths on the same "
            "candidates is what lets the report measure how the spread decays "
            "with training, which arch-zoo-100m-v1 showed is the binding "
            "property: its spread fell 3.7x between 400 and 1200 steps."
        ),
    )
    parser.add_argument(
        "--cross-factors",
        default="depth,hidden_size",
        help=(
            "comma-separated pair of structural genes for --design crossed, "
            "chosen from depth, hidden_size, expert_count, active_experts and "
            "expert_width. They are crossed at two levels each in a balanced "
            "factorial, so neither is confounded with the other."
        ),
    )
    parser.add_argument(
        "--cross-replicates",
        type=int,
        default=2,
        help=(
            "candidates per level combination for --design crossed, so the zoo "
            "holds 4 x replicates candidates. Two gives one replication per cell "
            "of the crossing, which is what separates a gene effect from "
            "within-cell variation."
        ),
    )
    parser.add_argument(
        "--head-counts",
        default="8,16",
        help=(
            "comma-separated attention-head counts for --design head-paired. "
            "Attention parameters are 4 * hidden^2 however that width is divided, "
            "so head count is free at a fixed hidden size and two counts can be "
            "matched exactly rather than only landed in a band. Each count must "
            "divide every hidden size in the design."
        ),
    )
    parser.add_argument(
        "--head-pairs",
        type=int,
        default=4,
        help=(
            "base configurations for --design head-paired, so the zoo holds "
            "pairs x len(head-counts) candidates. Heads vary *within* a pair at "
            "identical total and active parameters."
        ),
    )
    parser.add_argument(
        "--head-depths",
        default="10",
        help=(
            "comma-separated depths to draw --design head-paired base "
            "configurations from. Fixing one depth is what keeps the width "
            "contrast from being confounded with depth."
        ),
    )
    parser.add_argument(
        "--plan",
        default="halving",
        choices=["halving", "flat", "screen"],
        help=(
            "halving: screen short, promote survivors, confirm on more seeds. "
            "screen: the two-stage loop the findings recommend - screen every "
            "candidate at --stage1-steps, promote the top --keep-stage3, confirm "
            "those at --stage3-steps. It drops halving's middle stage because "
            "section 4a measured an 800-step stage to be a second short-run proxy, "
            "so its budget buys a promotion decided on a proxy rather than a "
            "finalist measured at the horizon. Pair with --promote-ties when the "
            "cut sits inside the seed noise. "
            "flat: every candidate at the final length on every finalist seed. "
            "Flat costs more per candidate but does not select on a short-run "
            "proxy, which arch-zoo-100m-v1 measured to be biased (its 400-step "
            "screen picked the wrong winner). Use flat when the quantity of "
            "interest is the spread at the final length."
        ),
    )
    parser.add_argument("--data-path", default="artifacts/real-lm/tokens.npy")
    parser.add_argument("--val-data-path", default="artifacts/real-lm/val-tokens.npy")
    parser.add_argument("--val-batches", type=int, default=16)
    parser.add_argument("--zoo-seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--budget-hours", type=float, default=20.0)
    parser.add_argument(
        "--assumed-tokens-per-s",
        type=float,
        default=4500.0,
        help=(
            "throughput used to project the plan and the budget guard. The phase's "
            "own runs have measured 4,900-6,100 tok/s at 100M parameters on the "
            "8 GB card, so this default is deliberately conservative: a projected "
            "cost stated at a rate nobody reached is the wrong side to be wrong on "
            "when the question is whether an arm fits (rescale by the measured "
            "rate from any completed ledger, which prints in the run rows)"
        ),
    )
    parser.add_argument(
        "--screen-noise-nats",
        type=float,
        default=None,
        help=(
            "override the promotion threshold, which by default is derived from the "
            "measured seed std at the stage's length and the number of screen seeds: "
            "std * sqrt(2 / seeds) nats, i.e. 0.0286 nats for a 2-seed 400-step screen "
            "and 0.0165 for a 2-seed 800-step one. Used only to report whether a "
            "promotion margin is large enough to be a decision, so setting it changes "
            "no run: reporting a cut as unresolved is not the same as overriding it."
        ),
    )
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--dry-run", action="store_true", help="emit reports, train nothing")
    return parser.parse_args()


def make_run(args: argparse.Namespace, steps: int, seed: int) -> RunConfig:
    return RunConfig(
        steps=steps,
        batch_size=args.batch_size,
        context_size=args.context_size,
        learning_rate=args.learning_rate,
        seed=seed,
        train_data_path=args.data_path,
        val_data_path=args.val_data_path,
        val_batches=args.val_batches,
        mixed_precision=args.mixed_precision,
    )


def _seed_list(spec: str) -> list[int]:
    """Seeds from a comma-separated flag, ignoring blanks."""
    return [int(part) for part in spec.split(",") if part.strip()]


def planned_stages(args: argparse.Namespace, zoo_size: int) -> list[dict]:
    """The batches each plan will ask for, in the order the driver asks for them.

    Pure arithmetic over the plan flags, mirroring the execution path: a stage loops
    seeds on the outside and candidates on the inside, so the budget guard - which
    breaks out of that loop - can only stop at a batch boundary. Stating the plan
    this way is what lets a dry run cost an arm before any GPU time, and lets the
    guard's stopping point be predicted rather than discovered hours into a run.
    """
    screen_seeds = _seed_list(args.screen_seeds)
    finalist_seeds = _seed_list(args.finalist_seeds)
    flat_lengths = [int(s) for s in args.flat_steps.split(",") if s.strip()]
    if args.plan == "flat":
        return [
            {
                "label": f"flat {length}",
                "steps": length,
                "seeds": finalist_seeds,
                "candidates": zoo_size,
            }
            for length in (flat_lengths or [args.stage3_steps])
        ]
    if args.plan == "screen":
        return [
            {
                "label": "screen",
                "steps": args.stage1_steps,
                "seeds": screen_seeds,
                "candidates": zoo_size,
            },
            {
                "label": "confirmation",
                "steps": args.stage3_steps,
                "seeds": finalist_seeds,
                "candidates": args.keep_stage3,
                # --promote-ties advances the whole borderline set, so the count is a
                # floor rather than a number: an estimate that printed it as exact
                # would understate the arm exactly when the screen was inconclusive.
                "may_grow": bool(args.promote_ties),
            },
        ]
    return [
        {
            "label": "stage 1",
            "steps": args.stage1_steps,
            "seeds": screen_seeds,
            "candidates": zoo_size,
        },
        {
            "label": "stage 2",
            "steps": args.stage2_steps,
            "seeds": screen_seeds,
            "candidates": args.keep_stage2,
        },
        {
            "label": "stage 3",
            "steps": args.stage3_steps,
            "seeds": finalist_seeds,
            "candidates": args.keep_stage3,
        },
    ]


def plan_cost(args: argparse.Namespace, stages: list[dict]) -> dict:
    """Tokens and GPU-hours the plan asks for, and where the budget guard stops.

    The guard spends a *token* budget, so a plan's cost is computable before any
    training - which matters because the phase's cap is stated in GPU-hours and the
    guard cannot see wall clock at all. Both units are reported beside the batch at
    which the guard would break, because that batch is otherwise only discovered by
    watching an arm stop part way through its design.
    """
    per_step: dict[int, int] = {}

    def tokens_for(steps: int) -> int:
        if steps not in per_step:
            per_step[steps] = make_run(args, steps, 0).train_tokens
        return per_step[steps]

    budget_tokens = args.budget_hours * 3600 * args.assumed_tokens_per_s
    planned_tokens = sum(
        tokens_for(stage["steps"]) * stage["candidates"] * len(stage["seeds"])
        for stage in stages
    )
    # Counted in individual training runs, not batches: the phase's cost table is
    # kept in runs (24 for a flat arm of 8 candidates on 3 seeds), and a batch count
    # would silently divide every arm by its zoo size.
    planned_runs = sum(
        stage["candidates"] * len(stage["seeds"]) for stage in stages
    )

    spent = 0
    covered_batches = 0
    covered_runs = 0
    stops_at: dict | None = None
    for stage in stages:
        for seed in stage["seeds"]:
            cost = tokens_for(stage["steps"]) * stage["candidates"]
            if spent + cost > budget_tokens:
                stops_at = {
                    "label": stage["label"],
                    "seed": seed,
                    "steps": stage["steps"],
                }
                break
            spent += cost
            covered_batches += 1
            covered_runs += stage["candidates"]
        if stops_at:
            break
    rate = args.assumed_tokens_per_s
    return {
        "budget_hours": args.budget_hours,
        "budget_tokens": budget_tokens,
        "assumed_tokens_per_s": rate,
        "planned_runs": planned_runs,
        "planned_batches": sum(len(stage["seeds"]) for stage in stages),
        "planned_tokens": planned_tokens,
        "planned_hours": planned_tokens / rate / 3600,
        "covered_runs": covered_runs,
        "covered_batches": covered_batches,
        "covered_hours": spent / rate / 3600,
        "stops_at": stops_at,
        "fits": stops_at is None,
        "stages": stages,
    }


def print_plan_cost(cost: dict) -> None:
    """State the plan's projected cost, and where the guard would stop it."""
    print(
        f"Plan: {cost['planned_runs']} runs, "
        f"{cost['planned_tokens'] / 1e6:.1f}M tokens, "
        f"~{cost['planned_hours']:.2f} GPU-hours at "
        f"{cost['assumed_tokens_per_s']:,.0f} tok/s"
    )
    for stage in cost["stages"]:
        growth = " (grows with --promote-ties ties)" if stage.get("may_grow") else ""
        print(
            f"  {stage['label']:<14} {stage['candidates']:>3} candidates x "
            f"{len(stage['seeds'])} seeds x {stage['steps']} steps{growth}"
        )
    if cost["fits"]:
        print(f"  within the {cost['budget_hours']}h budget guard")
        return
    stop = cost["stops_at"]
    print(
        f"  WARNING: the {cost['budget_hours']}h budget guard stops this plan at "
        f"{stop['label']} seed {stop['seed']}, after {cost['covered_runs']} of "
        f"{cost['planned_runs']} runs (~{cost['covered_hours']:.2f} GPU-hours), so the "
        "arm would report on a partial design. Raise --budget-hours or narrow the "
        "plan before spending the hours."
    )


def matching_report(
    candidates: list[ArchitectureCandidate],
    context_size: int,
    target_params: int,
    match_on: str | None = None,
) -> dict:
    """Parameter, FLOP and runtime-cost spread across the zoo.

    Reports both FLOP axes because they disagree: the contracted top-k FLOPs
    make two candidates comparable on paper, while the dense-expert runtime this
    repository executes scales with total expert count. A candidate with many
    narrow experts can carry identical contracted FLOPs and cost several times
    more wall clock, so wall-clock matching is reported separately.
    """
    from act.search.architecture_eval import (
        estimate_dense_runtime_flops_per_token,
        estimate_flops_per_token,
    )
    from act.model.factory import v0_moe_config_from_genome

    rows = []
    for candidate in candidates:
        config = v0_moe_config_from_genome(candidate.genome)
        contracted = estimate_flops_per_token(config, context_size=context_size)
        runtime = estimate_dense_runtime_flops_per_token(config, context_size=context_size)
        rows.append(
            {
                "name": candidate.name,
                "family": candidate.family,
                "total_params": candidate.total_params,
                "active_params": candidate.active_params,
                "expert_width": candidate.genome.layers[0].experts[0].width,
                "contracted_flops_per_token": contracted,
                "dense_runtime_flops_per_token": runtime,
                "runtime_over_contract": runtime / contracted,
            }
        )

    def spread(key: str) -> dict:
        values = [row[key] for row in rows]
        low, high = min(values), max(values)
        return {
            "min": low,
            "max": high,
            "ratio": high / low if low else float("inf"),
        }

    result = {
        "candidates": rows,
        # The arm's training budget is the *sum* over the set, not the band each
        # candidate sits in, and section 11's equal-compute condition is written in
        # terms of that sum. It is reported here so the random arm's budget can be
        # compared against a curated arm's without recomputing it from the ledger.
        "set_total_params": sum(row["total_params"] for row in rows),
        "param_matched": {
            "spread": spread("total_params"),
            "within_30pct_of_target": sum(
                1
                for row in rows
                if abs(row["total_params"] - target_params) / target_params <= 0.3
            ),
        },
        "contracted_flop_matched": {"spread": spread("contracted_flops_per_token")},
        "runtime_cost": {"spread": spread("dense_runtime_flops_per_token")},
        "note": (
            "Matching on contracted FLOPs does not match wall clock: the dense-expert "
            "runtime evaluates every expert on every token, so runtime cost tracks "
            "total expert count."
        ),
    }
    if match_on:
        result["matched_design"] = matching_quality(candidates, match_on, context_size)
    return result


#: Parameter budget per matched design, used when ``--match-target`` is omitted.
#: Each design matches on a different axis, so the default that makes it a sensible
#: zoo differs; crossed and head-paired both match total parameters and default to
#: the budget the phase's arms have been run at.
MATCHED_DESIGN_TARGETS = {
    "matched-total": 100_000_000.0,
    "matched-active": 35_000_000.0,
    "matched-flops": 2.6e8,
    "crossed": 78_000_000.0,
    "head-paired": 78_000_000.0,
}


def effective_target(args: argparse.Namespace) -> float:
    """The parameter budget the zoo will actually be built around.

    A matched design is built around ``--match-target`` and falls back to its own
    default when that is omitted, so the header must print the *resolved* value: the
    alternative is naming a budget the zoo never used, and formatting the raw flag
    crashes outright on the documented way to take the default.
    """
    if args.design in MATCHED_DESIGN_TARGETS:
        return float(args.match_target or MATCHED_DESIGN_TARGETS[args.design])
    return float(args.target_params)


def main() -> None:
    args = parse_args()
    screen_seeds = _seed_list(args.screen_seeds)
    finalist_seeds = _seed_list(args.finalist_seeds)

    output_dir = Path(args.artifacts_root) / args.experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    band = {
        "families": args.tolerance,
        "random": args.random_tolerance,
    }.get(args.design, args.match_tolerance)
    # A matched design is built around --match-target, not --target-params. Printing
    # the latter reads as though the zoo aimed at a budget it never used.
    target = effective_target(args)
    print(f"Experiment: {args.experiment_id}")
    print(
        f"Design {args.design}, target {target:,.0f} params (+/-{band:.0%}), "
        f"context {args.context_size}, batch {args.batch_size}, budget {args.budget_hours}h, "
        f"precision {args.mixed_precision}"
    )

    match_on: str | None = None
    if args.design == "families":
        zoo = generate_architecture_zoo(
            target_params=args.target_params,
            tolerance=args.tolerance,
            seed=args.zoo_seed,
            context_size=args.context_size,
            limit=args.candidates,
        )
    elif args.design == "crossed":
        factors = tuple(
            name.strip() for name in args.cross_factors.split(",") if name.strip()
        )
        if len(factors) != 2:
            raise SystemExit(
                f"--cross-factors needs exactly two gene names, got {args.cross_factors!r}"
            )
        zoo = generate_crossed_zoo(
            target_params=int(target),
            tolerance=args.match_tolerance,
            factors=factors,  # type: ignore[arg-type]
            replicates=args.cross_replicates,
            seed=args.zoo_seed,
            context_size=args.context_size,
            max_total_params=args.max_total_params,
        )
    elif args.design == "head-paired":
        zoo = generate_head_paired_zoo(
            target_params=int(target),
            tolerance=args.match_tolerance,
            head_counts=(
                int(value) for value in args.head_counts.split(",") if value.strip()
            ),
            pairs=args.head_pairs,
            seed=args.zoo_seed,
            context_size=args.context_size,
            max_total_params=args.max_total_params,
            depths=(int(value) for value in args.head_depths.split(",") if value.strip()),
        )
    elif args.design == "random":
        sampler = RandomArchitectureSearch(
            target_params=args.target_params,
            tolerance=args.random_tolerance,
            size=args.candidates,
            seed=args.zoo_seed,
            context_size=args.context_size,
            max_total_params=args.max_total_params,
        )
        space = sampler.describe_space()
        print(
            f"Random baseline space: {space['cells']} cells, "
            f"total params x{space['total_spread_ratio']:.2f}, "
            f"active params x{space['active_spread_ratio']:.2f}"
        )
        (output_dir / "search_space.json").write_text(json.dumps(space, indent=2))
        zoo = sampler.sample()
    else:
        match_on = args.design.removeprefix("matched-")
        zoo = generate_matched_zoo(
            target=target,
            match_on=match_on,
            tolerance=args.match_tolerance,
            seed=args.zoo_seed,
            context_size=args.context_size,
            max_total_params=args.max_total_params,
            limit=args.candidates,
        )
    if args.only:
        wanted = [name.strip() for name in args.only.split(",") if name.strip()]
        available = {candidate.name for candidate in zoo}
        missing = [name for name in wanted if name not in available]
        if missing:
            raise SystemExit(
                f"--only names not present in the zoo: {missing}\n"
                f"available: {sorted(available)}"
            )
        keep = set(wanted)
        zoo = [candidate for candidate in zoo if candidate.name in keep]
        print(f"\n--only: restricted to {len(zoo)} of the zoo's architectures")

    print_zoo_summary(zoo)

    distances = compute_distance_matrix(zoo)
    matching = matching_report(zoo, args.context_size, args.target_params, match_on)

    if "matched_design" in matching:
        quality = matching["matched_design"]
        print(
            f"Matched design ({quality['match_on']}): matched axis holds within "
            f"{quality['matched_spread_over_mean']:.2%}, varied axis spans "
            f"x{quality['varied_spread_ratio']:.2f}"
        )

    print("Matching report")
    print(f"  set total params  : {matching['set_total_params']:,} "
          f"(mean {matching['set_total_params'] / len(zoo):,.0f}/candidate)")
    print(f"  total params      : {matching['param_matched']['spread']['min']:,} .. "
          f"{matching['param_matched']['spread']['max']:,} "
          f"(x{matching['param_matched']['spread']['ratio']:.2f})")
    print(f"  active params     : {min(c.active_params for c in zoo):,} .. "
          f"{max(c.active_params for c in zoo):,}")
    print(f"  contracted FLOPs  : x{matching['contracted_flop_matched']['spread']['ratio']:.2f}")
    print(f"  runtime cost      : x{matching['runtime_cost']['spread']['ratio']:.2f}")
    print(f"  min zoo distance  : {min_pairwise_distance(zoo):.3f}\n")

    if args.design == "crossed":
        cross_factors = tuple(
            name.strip() for name in args.cross_factors.split(",") if name.strip()
        )
        balance = crossing_balance(zoo, cross_factors)
        if balance.get("available"):
            # The crossing's whole claim is that its two genes are independent,
            # so report the evidence for that instead of asserting it.
            print(
                f"Crossed design ({cross_factors[0]} x {cross_factors[1]}): "
                f"rho = {balance['spearman']:+.3f}"
                + (" (balanced)" if balance["balanced"] else " (NOT balanced)")
            )
            for factor in balance["factors"]:
                levels = balance["levels"][factor]
                print(f"  {factor:<16}: " + ", ".join(
                    f"{value} x{count}" for value, count in levels.items()
                ))
            print()
            matching["crossed_design"] = balance

    if args.design == "head-paired":
        # The claim here is stronger than the crossing's, so it gets its own
        # report: within a pair the candidates should be *identical* in total and
        # active parameters, because only head count differs and it is free.
        paired = head_pair_matching(
            zoo, [int(v) for v in args.head_counts.split(",") if v.strip()]
        )
        print(
            "Head-paired design: every pair complete = "
            + ("yes" if paired["all_complete"] else "NO")
            + ", exactly matched total and active params = "
            + ("yes" if paired["all_exactly_matched"] else "NO")
        )
        for entry in paired["pairs"]:
            base = entry["base"]
            print(
                f"  pair D{base['depth']}_H{base['hidden_size']}_E{base['expert_count']}"
                f"_A{base['active_experts']}_W{base['expert_width']}: "
                f"heads {entry['head_counts']} at {entry['total_params']:,} total "
                f"params"
            )
        print(
            f"  total-parameter spread across pairs: "
            f"x{paired['total_spread_ratio']:.3f}"
        )
        print()
        matching["head_paired_design"] = paired

    (output_dir / "zoo_definition.json").write_text(
        json.dumps(
            {
                "experiment_id": args.experiment_id,
                "context_size": args.context_size,
                "batch_size": args.batch_size,
                "candidates": [
                    {
                        "name": c.name,
                        "family": c.family,
                        "genome": c.genome.to_dict(),
                        "total_params": c.total_params,
                        "active_params": c.active_params,
                        "description": c.description,
                    }
                    for c in zoo
                ],
            },
            indent=2,
        )
    )
    (output_dir / "distance_matrix.json").write_text(
        json.dumps({"names": [c.name for c in zoo], "matrix": distances}, indent=2)
    )
    (output_dir / "matching_report.json").write_text(json.dumps(matching, indent=2))
    print(f"Reports written to {output_dir}")

    print_plan_cost(plan_cost(args, planned_stages(args, len(zoo))))

    if args.dry_run:
        print("\n--dry-run: stopping before any training.")
        return

    warn_if_gpu_shared(competing_sweeps())

    # Restart exactly the ledgers this invocation is about to rewrite, rather
    # than every ledger in the directory. The old blanket glob matched only
    # `stage*`, so a second flat run in the same directory silently accumulated
    # duplicate rows instead of restarting, and a horizon arm could not keep both
    # training lengths side by side in one experiment.
    flat_lengths = [int(s) for s in args.flat_steps.split(",") if s.strip()]
    if args.plan == "flat":
        planned_tags = [
            f"flat{length if args.flat_steps else ''}_seed{seed}"
            for length, seed in product(
                flat_lengths or [args.stage3_steps], finalist_seeds
            )
        ]
    elif args.plan == "screen":
        planned_tags = [
            f"{stage}_seed{seed}"
            for stage, seeds in (("stage1", screen_seeds), ("stage3", finalist_seeds))
            for seed in seeds
        ]
    else:
        planned_tags = [
            f"{stage}_seed{seed}"
            for stage, seeds in (
                ("stage1", screen_seeds),
                ("stage2", screen_seeds),
                ("stage3", finalist_seeds),
            )
            for seed in seeds
        ]
    for tag in planned_tags:
        (output_dir / f"{tag}_results.jsonl").unlink(missing_ok=True)

    all_results: list[EvalResult] = []
    budget_tokens = args.budget_hours * 3600 * args.assumed_tokens_per_s
    spent_tokens = 0
    promotions: list[dict] = []

    def report_cut(label: str, ranked: list, keep: int, steps: int, seeds: int) -> dict:
        """Print a stage's cut verdict and keep it for the decision file.

        The verdict used to exist only in stdout, which is where an arm's promotion
        reasoning went to die: the decision file is the durable record, so the same
        margin, threshold and seed count are carried into it.
        """
        noise = screen_noise(args, seeds, steps)
        margins = report_promotion(
            label, ranked, keep, steps=steps, noise=noise, seeds=seeds
        )
        if margins.get("available"):
            promotions.append(
                {
                    "label": label,
                    "steps": steps,
                    "seeds": seeds,
                    "std": seed_std_for_steps(steps),
                    "threshold": noise,
                    "threshold_overridden": args.screen_noise_nats is not None,
                    "cut_margin": margins["cut_margin"],
                    "resolvable": margins["resolvable"],
                    "cut_pair": margins["cut_pair"],
                    "borderline": margins["borderline"],
                }
            )
        return margins

    def run_stage(
        label: str,
        candidates: list[ArchitectureCandidate],
        steps: int,
        seeds: list[int],
        tag: str,
    ) -> list[EvalResult]:
        """Train every candidate in `candidates` at `steps` on each seed."""
        nonlocal spent_tokens
        produced: list[EvalResult] = []
        for seed in seeds:
            run = make_run(args, steps, seed)
            if spent_tokens + run.train_tokens * len(candidates) > budget_tokens:
                print(f"budget reached before {label} completed")
                break
            print(f"\n=== {label}: {len(candidates)} candidates x {steps} steps, seed {seed} ===")
            batch = evaluate_batch(
                candidates,
                run=run,
                device=args.device,
                output_dir=str(output_dir),
                tag=f"{tag}_seed{seed}",
            )
            produced += batch
            spent_tokens += run.train_tokens * len(batch)
        return produced

    def write_reports(results: list[EvalResult]) -> None:
        """Emit ranked table, search cost and decision for whatever completed."""
        outcome = best_available_summary(
            results, sorted(set(screen_seeds + finalist_seeds))
        )
        # Promotion verdicts are part of the arm's reading, not just its log: which
        # cuts were decided by the data is what says whether the promotion was
        # selection or coin-flipping.
        outcome["promotions"] = promotions
        cost = search_cost(results, len(zoo))
        (output_dir / "ranked.json").write_text(json.dumps(outcome, indent=2))
        (output_dir / "search_cost.json").write_text(json.dumps(cost, indent=2))
        write_decision(output_dir / "decision.md", outcome)

        print(f"\n{'=' * 78}")
        print(f"Search cost: {cost['total_search_flops']:.3e} FLOPs over "
              f"{cost['tokens_trained'] / 1e6:.1f}M tokens, {cost['wall_clock_hours']:.2f} GPU-hours")
        print(f"{'=' * 78}")
        if outcome.get("available"):
            print(f"Best at {outcome['steps']} steps: "
                  f"{outcome['best']} ({outcome['best_mean_val_loss']:.4f})")
            print(f"Spread: {outcome['architecture_spread']:.4f} nats "
                  f"(within-seed std {outcome['within_seed_std']:.4f})")
            blocked = outcome.get("blocked_test") or {}
            anova = outcome.get("anova") or {}
            units = outcome.get("compute_units") or {}
            if blocked.get("available"):
                ranks = blocked.get("p_ranks")
                print(f"Seed-matched exact test: p={blocked['p']:.4f} over "
                      f"{blocked['blocks']} blocks x {blocked['architectures']} architectures "
                      f"(min attainable {blocked['min_achievable_p']:.4f})"
                      + (f", rank-based p={ranks:.4f}" if ranks is not None else ""))
                # The test can only run over a complete rectangle, so anything the
                # rectangle left out is said here rather than left to the block count.
                if blocked.get("unfinished_seeds"):
                    print(f"  seeds excluded (still training): "
                          f"{blocked['unfinished_seeds']} - the block count above is "
                          "the seeds that finished")
                if blocked.get("architectures_dropped"):
                    print(f"  architectures excluded (not on every seed): "
                          f"{blocked['architectures_dropped']}")
                if blocked.get("unbalanced_seeds"):
                    print("  WARNING: seeds excluded that had written every candidate: "
                          f"{blocked['unbalanced_seeds']} - a zoo defect, so the "
                          "test above is provisional")
            if anova.get("available"):
                print(f"Unpaired ANOVA: F={anova['f']:.2f}, p={anova['p']:.4f}")
            if units.get("available"):
                print(f"Effect size: {units['spread']:.4f} nats = "
                      f"{units['equivalent_steps']:.0f} training steps "
                      f"({units['fraction_of_budget']:.1%} of budget)")
            print(f"Verdict: {verdict(outcome)[0]}")
        print(f"\nDecision written to {output_dir / 'decision.md'}")

    if args.plan == "flat":
        # Every candidate at the final length on every finalist seed. Halving is
        # cheaper but promotes on a short-run proxy, and arch-zoo-100m-v1 showed
        # that proxy is biased: its 400-step screen put the eventual winner in
        # second place, by 0.009 nats against that screen's own 0.0390-nat
        # per-architecture seed std, and the spread fell 3.7x between 400 and 1200
        # steps. A design whose purpose is to estimate the final spread must not
        # select candidates on it.
        for length in flat_lengths or [args.stage3_steps]:
            tag = f"flat{length if args.flat_steps else ''}"
            flat = run_stage(f"Flat {length}", list(zoo), length, finalist_seeds, tag)
            print_results_table(flat)
            all_results += flat
        write_reports(all_results)
        return

    if args.plan == "screen":
        # ---- Screen: all candidates, short, then confirm on more seeds ----
        stage1 = run_stage("Screen", list(zoo), args.stage1_steps, screen_seeds, "stage1")
        all_results += stage1
        print_results_table(stage1)

        stage1_rank = rank(stage1, args.stage1_steps)
        if not stage1_rank:
            print("screen produced no results; stopping")
            write_reports(all_results)
            return
        report_cut(
            "Screen", stage1_rank, args.keep_stage3, args.stage1_steps,
            len(screen_seeds),
        )
        promotion = promoted_names(
            stage1_rank, args.keep_stage3,
            noise=screen_noise(args, len(screen_seeds), args.stage1_steps),
            promote_ties=args.promote_ties,
        )
        if promotion["promoted_ties"]:
            print(
                "Promoting the borderline set as well (the cut could not separate "
                f"it): {promotion['promoted_ties']}"
            )
        finalists = [c for c in zoo if c.name in promotion["names"]]
        print(
            f"Confirmation finalists ({len(finalists)} of {len(zoo)}): "
            f"{[c.name for c in finalists]}"
        )

        stage3 = run_stage(
            "Confirmation", finalists, args.stage3_steps, finalist_seeds, "stage3"
        )
        all_results += stage3
        print_results_table(stage3)

        write_reports(all_results)
        return

    # ---- Stage 1: all candidates, short --------------------------------
    stage1 = run_stage("Stage 1", list(zoo), args.stage1_steps, screen_seeds, "stage1")
    all_results += stage1
    print_results_table(stage1)

    stage1_rank = rank(stage1, args.stage1_steps)
    if not stage1_rank:
        print("stage 1 produced no results; stopping")
        write_reports(all_results)
        return
    report_cut("Stage 1", stage1_rank, args.keep_stage2, args.stage1_steps, len(screen_seeds))
    keep = [name for name, _, _ in stage1_rank[: args.keep_stage2]]
    survivors = [c for c in zoo if c.name in keep]
    print(f"Stage 1 survivors: {keep}")

    # ---- Stage 2: longer, survivors only -------------------------------
    stage2 = run_stage("Stage 2", survivors, args.stage2_steps, screen_seeds, "stage2")
    all_results += stage2
    print_results_table(stage2)

    stage2_rank = rank(stage2, args.stage2_steps)
    if not stage2_rank:
        print("stage 2 produced no results; stopping")
        write_reports(all_results)
        return
    report_cut("Stage 2", stage2_rank, args.keep_stage3, args.stage2_steps, len(screen_seeds))
    finalist_names = [name for name, _, _ in stage2_rank[: args.keep_stage3]]
    finalists = [c for c in zoo if c.name in finalist_names]
    print(f"Stage 3 finalists: {finalist_names}")

    # ---- Stage 3: multi-seed confirmation -------------------------------
    stage3 = run_stage("Stage 3", finalists, args.stage3_steps, finalist_seeds, "stage3")
    all_results += stage3
    print_results_table(stage3)

    write_reports(all_results)


if __name__ == "__main__":
    main()
