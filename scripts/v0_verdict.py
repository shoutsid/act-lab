"""Compute the pre-registered section 5.1 verdict from a finished V0 dataset.

Reads every published seed summary under ``<dataset-root>/act-checkpoints/
v0-5-1/`` and applies the decision rules fixed in RESEARCH_REPORT.md section
5.1 - without re-tuning anything:

- **Primary (cost)**: C = aggregate adaptive search FLOPs / aggregate adaptive
  training FLOPs, the same accounting as every toy study. Criterion: C <= 5%.
- **Tolerance band**: C_measured must land in [0.5, 2] x C_expected for the
  run's probe budget (cap-4/1-batch: 0.0942%, cap-4/2-batch: 0.1883%).
- **Harm guard**: adaptive must not lose in >= 10 of the 12 pairs (exact
  one-tailed sign test p = 0.0193); a rejection withholds the Gate 4
  recommendation even when the cost criteria pass.
- **Pool-matched control clause** (adopted section 5.5, change 3): the protocol
  runs a second session over the same pairs with `probe_selection="random"` -
  the same candidates at the same probe cost, chosen without reference to
  probe loss. If that loss-blind arm also clears the same 10-of-12 bar against
  its own random arm, the pool/location effect alone is sufficient to pass, so
  the adaptive edge does not demonstrate selection value and the Gate 4
  recommendation is withheld (the cost PASS stands). A missing control is
  equally disqualifying for Gate 4, since the clause cannot be evaluated.
- **Do-nothing clause** (pre-registered section 5.14): the protocol runs a
  null-session queue at declared fidelity over the SAME block (single-arm
  `arms=("null",)` sessions, own roots). If the adaptive arm fails to beat
  the null arm on >= 10 of 12 seeds, the adaptive edge does not demonstrate
  value over doing nothing and Gate 4 is withheld (the cost PASS stands). A
  missing or incomplete null queue is equally disqualifying: running the
  do-nothing control is part of the protocol, and its absence must never
  read as "no null effect found". The registered rules (cost, band, harm
  guard, pool clause) are untouched.

Usage:

    PYTHONPATH=src python scripts/v0_verdict.py \
        --dataset-root artifacts/datasets \
        --control-root artifacts/datasets-control \
        --null-root artifacts/datasets-null

Exit code 0 = PASS (cost + band + harm guard + control + null clauses clear),
1 = FAIL (primary cost miss), 2 = PASS with qualification (cost passes but the
measured C lands outside the tolerance band, the harm guard rejects, the
pool-matched control also passes, the null clause rejects, or a required
control queue was not supplied), 3 = no verdict (the invocation was malformed:
missing handoff, incomplete seed, unmatched budgets, a null root that is not a
null queue). Code 3 exists so an operator error can never be read as the
pre-registered FAIL, which over a multi-session run is the difference between
"the amortization hypothesis is false" and "the command pointed somewhere
else". The JSON verdict is printed and written beside the dataset root for the
artifact trail, and is only ever written when a verdict was produced.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from act.training.session_plan import (
    DECLARED_ACTIVE_PARAMETERS,
    DECLARED_TOTAL_PARAMETERS,
    SEED_TOKENS_PER_ARM,
    expected_probe_cost_percent,
    tolerance_band,
)
from act.training.v0_handoff import HANDOFF_DATASET, published_seed_status

# Re-reserved from 9000-9011 by the adopted section 5.5 revision (change 4);
# see RESEARCH_REPORT.md sections 5.5 and 9.9.
DECLARED_SEEDS = tuple(range(15000, 15012))
PRIMARY_COST_PERCENT = 5.0
HARM_GUARD_LOSS_LIMIT = 10  # adaptive worse in >= 10 of 12 pairs rejects
# The loss-blind pool-matched arm winning >= 10 of 12 pairs means a
# pool/location effect alone clears the gate, so the adaptive edge does not
# demonstrate selection value and Gate 4 is withheld (section 5.5 change 3).
POOL_CONTROL_WIN_LIMIT = 10
# The do-nothing clause (pre-registered section 5.14): adaptive failing to
# beat the null arm on >= 10 of 12 seeds means the edge does not demonstrate
# value over doing nothing; Gate 4 is withheld and the cost PASS stands.
NULL_WIN_LIMIT = 10

# No verdict was produced (malformed invocation). Distinct from 1 = FAIL so
# the two are never confused by an unattended operator or a wrapper script.
INPUT_ERROR_EXIT_CODE = 3


class InputError(SystemExit):
    """Malformed or incomplete inputs: no verdict was produced (exit 3).

    ``pytest.raises(SystemExit, match=...)`` still sees the message, because
    ``__str__`` returns it; the process exit code is the distinct 3.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
        self.code = INPUT_ERROR_EXIT_CODE

    def __str__(self) -> str:
        return self.message


def parse_seeds(spec: str) -> tuple[int, ...]:
    """Parse ``A-B`` (inclusive range) or a comma-separated seed list.

    The registered queue is ``DECLARED_SEEDS``; this exists so a re-reserved
    block (section 5.5 change 4) is a flag rather than a code edit, keeping
    the revision executable without touching registered defaults.
    """
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        start, end = spec.split("-", 1)
        return tuple(range(int(start), int(end) + 1))
    seeds = tuple(int(value.strip()) for value in spec.split(",") if value.strip())
    if not seeds:
        raise InputError("empty seed specification")
    if len(set(seeds)) != len(seeds):
        raise InputError("seed specification contains duplicates")
    return seeds


def _seed_numbers(dataset_root: Path) -> list[int]:
    handoff = dataset_root / HANDOFF_DATASET
    if not handoff.is_dir():
        raise InputError(f"no handoff dataset under {handoff}")
    seeds = []
    for path in handoff.glob("seed-*"):
        stem = path.stem.split("-")[-1]
        if stem.isdigit() and published_seed_status(dataset_root, int(stem)) == "complete":
            seeds.append(int(stem))
    return sorted(seeds)


def _load_seed(dataset_root: Path, seed: int) -> dict:
    path = dataset_root / HANDOFF_DATASET / f"seed-{seed:04d}" / "reports" / "seed-summary.json"
    if not path.is_file():
        raise InputError(f"missing seed summary: {path}")
    summary = json.loads(path.read_text())
    if summary.get("status") != "complete":
        raise InputError(f"seed {seed} is not a complete version")
    return summary


def _arm_pairs(dataset_root: Path, seeds: list[int]) -> list[dict[str, object]]:
    """Per-seed adaptive/random final losses for a published seed block."""
    pairs: list[dict[str, object]] = []
    for seed in seeds:
        arms = _load_seed(dataset_root, seed)["arms"]
        if "random" not in arms or "adaptive" not in arms:
            raise InputError(f"seed {seed} is missing the random/adaptive pair")
        pairs.append(
            {
                "seed": seed,
                "adaptive_loss": float(arms["adaptive"]["final_loss"]),
                "random_loss": float(arms["random"]["final_loss"]),
                "adaptive_wins": float(arms["adaptive"]["final_loss"])
                < float(arms["random"]["final_loss"]),
            }
        )
    return pairs


def _control_block(dataset_root: Path, declared_seeds: tuple[int, ...]) -> dict:
    """Pool-matched loss-blind control block over the same seed pairs.

    The control's clause is ``wins >= POOL_CONTROL_WIN_LIMIT``, so a control that
    completed fewer than the limit's worth of pairs **cannot fire it**: the arm would
    have to win 10 of the 4 seeds it ran. ``evaluated`` therefore requires the control
    to be complete on the declared block, exactly as ``_null_block`` requires of the
    do-nothing queue - anything less withholds Gate 4 with the reason recorded rather
    than leaving the clause unfireable and reporting a clean PASS. Measured before the
    fix: a control run on 4 seeds winning all 4 authorized Gate 4 (``verdict: PASS``,
    ``gate_4_authorized: True``, exit 0) because 4 < 10.
    """
    complete = _seed_numbers(dataset_root)
    seeds = [seed for seed in complete if seed in declared_seeds]
    missing = sorted(set(declared_seeds) - set(seeds))
    if missing:
        return {
            "evaluated": False,
            "dataset_root": str(dataset_root),
            "pairs": len(seeds),
            "seeds_missing": missing,
            "win_limit": POOL_CONTROL_WIN_LIMIT,
            "reason": (
                f"pool-matched control incomplete on the declared block: missing "
                f"seed(s) {missing} ({len(seeds)} of {len(declared_seeds)} pairs); the "
                f"section 5.5 change-3 clause needs {POOL_CONTROL_WIN_LIMIT} pairs "
                "before it can fire at all, so an incomplete control cannot be "
                "evaluated and Gate 4 is withheld"
            ),
        }
    pairs = _arm_pairs(dataset_root, seeds)
    wins = sum(1 for pair in pairs if pair["adaptive_wins"])
    # The clause bites when the loss-blind arm clears the same bar the harm
    # guard uses: a pool/location effect alone would then pass the gate.
    return {
        "evaluated": True,
        "dataset_root": str(dataset_root),
        "pairs": len(pairs),
        "seeds_missing": missing,
        "adaptive_wins": wins,
        "win_limit": POOL_CONTROL_WIN_LIMIT,
        "control_passes_gate": wins >= POOL_CONTROL_WIN_LIMIT,
        "per_seed": pairs,
    }


def _null_block(
    null_root: Path | None,
    declared_seeds: tuple[int, ...],
    primary_summaries: dict[int, dict],
) -> dict[str, object]:
    """The section 5.14 do-nothing queue: per-seed null vs adaptive pairing.

    The null queue lives under its own dataset root (the section 5.5
    control-root pattern): single-arm ``arms=("null",)`` sessions at declared
    fidelity over the SAME declared block. ``evaluated`` requires the queue
    to exist AND be complete on the declared block; anything less withholds
    Gate 4 rather than reading as a null result.
    """
    if null_root is None:
        return {
            "evaluated": False,
            "reason": (
                "no --null-root supplied; the pre-registered section 5.14 "
                "do-nothing clause requires the null-session queue before "
                "Gate 4 can be authorized"
            ),
        }
    complete = _seed_numbers(null_root)
    if not complete:
        return {
            "evaluated": False,
            "reason": (
                f"no complete seed versions under {null_root}; the section 5.14 "
                "do-nothing queue has not been run"
            ),
        }
    seeds = [seed for seed in complete if seed in declared_seeds]
    missing = sorted(set(declared_seeds) - set(seeds))
    if missing:
        return {
            "evaluated": False,
            "reason": (
                f"null queue incomplete on the declared block: missing seed(s) "
                f"{missing}; the section 5.14 clause cannot be evaluated and "
                "Gate 4 is withheld"
            ),
        }
    # Pairing is over the intersection of complete seeds on both sides: a
    # seed missing on either side is already reported by that side's own
    # queue-completeness accounting, and Gate 4 additionally requires both
    # queues complete.
    pairs: list[dict[str, object]] = []
    adaptive_losses = 0
    for seed in [seed for seed in seeds if seed in primary_summaries]:
        arms = _load_seed(null_root, seed)["arms"]
        if "null" not in arms:
            raise InputError(f"null queue seed {seed} is missing the null arm")
        if "adaptive" in arms or "random" in arms:
            raise InputError(
                f"null queue seed {seed} must be a single-arm null session "
                "(adaptive/random arms found; is --null-root pointed at a "
                "pair-session dataset?)"
            )
        # The pairing is cross-queue by design: the adaptive loss comes from
        # the PRIMARY run's seed summary, the null loss from the do-nothing
        # queue - the null sessions carry no adaptive arm of their own.
        primary_arms = primary_summaries[seed]["arms"]
        if "adaptive" not in primary_arms:
            raise InputError(
                f"seed {seed}: the primary dataset has no adaptive arm to pair "
                "against the null queue"
            )
        adaptive_loss = float(primary_arms["adaptive"]["final_loss"])
        null_loss = float(arms["null"]["final_loss"])
        # "Beat" is strict: ties count against the arm seeking Gate 4, the
        # same conservative reading the harm guard applies to losses.
        won = adaptive_loss < null_loss
        if not won:
            adaptive_losses += 1
        pairs.append(
            {
                "seed": seed,
                "adaptive_loss": adaptive_loss,
                "null_loss": null_loss,
                "null_search_flops": int(arms["null"]["search_flops"]),
                "null_mutation_count": int(arms["null"]["mutation_count"]),
                "adaptive_beats_null": won,
            }
        )
    for pair in pairs:
        if pair["null_search_flops"] != 0 or pair["null_mutation_count"] != 0:
            raise InputError(
                f"null queue seed {pair['seed']}: the null arm must book zero "
                "search FLOPs and zero mutations"
            )
    wins = len(pairs) - adaptive_losses
    return {
        "evaluated": True,
        "null_root": str(null_root),
        "pairs": len(pairs),
        "seeds_missing": [],
        "adaptive_wins": wins,
        "adaptive_losses_or_ties": adaptive_losses,
        "win_limit": NULL_WIN_LIMIT,
        "clause_rejects": adaptive_losses >= NULL_WIN_LIMIT,
        "per_seed": pairs,
    }


def _run_scale_reading(
    cost_percent: float,
    predicted_percent: float,
    pairs: list[dict[str, object]],
    training: int,
) -> dict:
    """The pre-registered band re-expressed at the scale the run executed at.

    The registered band is ``[0.5, 2] x`` the **declared-scale** prediction, and the
    registration's own rationale for that slack is that "every architectural deviation
    between toy and V0 ... enters the identity as a bounded multiplicative factor on
    ``N_active`` or ``D``". This reads the same identity with those two factors taken
    from the run instead of from the declaration: the search side is proportional to the
    active width, and the training side is the run's own booked training FLOPs, so

        C(run) = C_declared x (declared_total / measured_total) x (N_run / N_declared)

    This **reduces to the registered comparison exactly** when the run is at declared
    scale, where both factors are 1 - it is a generalisation of the registered check,
    not a second criterion, and it is deliberately descriptive: ``in_tolerance_band``
    and the verdict label are still computed from the declared-scale prediction and do
    not move. It exists because otherwise the registered band can only be passed by a
    run at the declared model size and the declared token budget, so applying it to any
    reduced-fidelity or partially-completed block reports a miss that says more about
    the arithmetic than about the hypothesis.

    Assumptions, stated because they are the identity's own: the declared search shape
    (``DECLARED_EVENTS`` events, ``DECLARED_CANDIDATES`` candidates, one probe batch of
    ``PROBE_TOKENS_PER_BATCH`` tokens per event). A run that also changed those is not
    covered by this reading and its ratio should be read as an upper bound.
    """
    declared_total = 2 * DECLARED_TOTAL_PARAMETERS * SEED_TOKENS_PER_ARM * len(pairs)
    horizon_factor = declared_total / training if training else math.inf
    widths = [int(pair["adaptive_active_parameters"]) for pair in pairs]
    mean_active = sum(widths) / len(widths) if widths else 0.0
    shape_factor = mean_active / DECLARED_ACTIVE_PARAMETERS if mean_active else math.inf
    scale = horizon_factor * shape_factor
    predicted_at_run_scale = predicted_percent * scale
    ratio = cost_percent / predicted_at_run_scale if predicted_at_run_scale else math.inf
    at_declared_scale = abs(scale - 1.0) <= 0.02
    return {
        "descriptive_only": True,
        "measured_cost_percent": cost_percent,
        "predicted_cost_percent_at_run_scale": predicted_at_run_scale,
        "measured_over_predicted_at_run_scale": ratio,
        "in_tolerance_band_at_run_scale": 0.5 <= ratio <= 2.0,
        "scale_factor_measured_to_declared": scale,
        "horizon_factor": horizon_factor,
        "mean_adaptive_active_parameters": mean_active,
        "declared_total_training_flops": declared_total,
        "measured_total_training_flops": training,
        "run_is_at_declared_scale": at_declared_scale,
        "assumptions": (
            "the declared search shape: 3 events, 4 candidates, one probe batch of "
            "4096 tokens per event"
        ),
        "reading": (
            "declared scale: the registered band applies as written"
            if at_declared_scale
            else (
                f"run executed at {scale:.4g}x the declared scale, so the registered "
                f"band cannot be met by construction; against the identity at the "
                f"run's own scale the measured cost is {ratio:.4f}x the prediction "
                f"({'inside' if 0.5 <= ratio <= 2.0 else 'outside'} the [0.5, 2] band)"
            )
        ),
    }


def compute_verdict(
    dataset_root: Path,
    *,
    probe_batch_count: int,
    declared_seeds: tuple[int, ...] = DECLARED_SEEDS,
    control_root: Path | None = None,
    null_root: Path | None = None,
) -> dict:
    """Assemble the machine-checked section 5.1 verdict from published seeds.

    ``declared_seeds`` defaults to the registered queue; overriding it does
    not loosen any rule (the 5% criterion, the tolerance band, the harm guard
    and the queue-completeness requirement all apply to whatever block is
    declared). ``control_root`` is the pool-matched loss-blind arm's dataset;
    the adopted section 5.5 protocol requires it, so its absence withholds the
    Gate 4 recommendation rather than silently skipping the clause.
    ``null_root`` is the section 5.14 do-nothing queue's dataset (single-arm
    null sessions over the same block); same rule: required for Gate 4.
    """
    if not declared_seeds:
        raise InputError("declared seed block must not be empty")
    complete = _seed_numbers(dataset_root)
    if not complete:
        raise InputError("no complete seed versions found; nothing to judge")
    # The pre-registered analysis is over the declared queue exactly; stray
    # complete seeds (e.g. smoke tests) must not pollute the aggregate.
    strays = sorted(set(complete) - set(declared_seeds))
    seeds = [seed for seed in complete if seed in declared_seeds]
    missing = sorted(set(declared_seeds) - set(seeds))
    summaries = {seed: _load_seed(dataset_root, seed) for seed in seeds}

    # Aggregate per-arm FLOPs across all complete seeds (the pre-registered
    # primary metric's accounting; same convention as the toy studies).
    search = 0
    training = 0
    pairs: list[dict[str, object]] = []
    for seed in seeds:
        arms = summaries[seed]["arms"]
        if "random" not in arms or "adaptive" not in arms:
            raise InputError(f"seed {seed} is missing the random/adaptive pair")
        if int(arms["adaptive"]["mutation_count"]) != int(arms["random"]["mutation_count"]):
            raise InputError(f"seed {seed}: unmatched mutation budgets")
        search += int(arms["adaptive"]["search_flops"])
        training += int(arms["adaptive"]["training_flops"])
        pairs.append(
            {
                "seed": seed,
                "adaptive_loss": float(arms["adaptive"]["final_loss"]),
                "random_loss": float(arms["random"]["final_loss"]),
                "adaptive_wins": float(arms["adaptive"]["final_loss"])
                < float(arms["random"]["final_loss"]),
                # Carried for the run-scale reading below: the identity's search
                # side is proportional to the active width, which every event
                # moves, so the run's own number is the one to use.
                "adaptive_active_parameters": int(arms["adaptive"]["active_parameter_count"]),
            }
        )

    cost_ratio = search / training if training else math.inf
    cost_percent = 100.0 * cost_ratio
    predicted_percent = expected_probe_cost_percent(probe_batch_count)
    band_low, band_high = tolerance_band(probe_batch_count)
    losses = sum(1 for pair in pairs if not pair["adaptive_wins"])
    wins = len(pairs) - losses

    cost_pass = cost_percent <= PRIMARY_COST_PERCENT
    in_band = band_low <= cost_percent <= band_high
    run_scale = _run_scale_reading(cost_percent, predicted_percent, pairs, training)
    harm_guard_clear = losses < HARM_GUARD_LOSS_LIMIT
    if control_root is None:
        control: dict = {
            "evaluated": False,
            "reason": (
                "no --control-root supplied; the adopted section 5.5 protocol "
                "requires the pool-matched loss-blind arm before Gate 4 can "
                "be authorized"
            ),
        }
    else:
        control = _control_block(control_root, declared_seeds)
    null = _null_block(null_root, declared_seeds, summaries)

    # Section 5.5 change 3 asks about one thing: is the pool-matched arm's quality
    # *indistinguishable* from the adaptive arm's? As shipped, the clause implemented the
    # direction where both arms clear the bar (the control clears POOL_CONTROL_WIN_LIMIT,
    # so a pool/location effect alone would have passed the gate). The other direction was
    # missing: **neither** arm clears the bar - the pool-matched arm is then exactly as
    # good as the adaptive one, both having shown no edge over their own churn arm, and
    # there is no quality premise to recommend Gate 4 on. The harm guard cannot catch it
    # (it only rejects adaptive losing >= HARM_GUARD_LOSS_LIMIT pairs) and neither can the
    # section 5.14 clause (which compares churn against do-nothing, not adaptive against
    # the loss-blind arm). Measured before this amendment: adaptive 6/12, control 6/12, a
    # clean cost PASS authorized Gate 4 at exit 0. See section 5.44.
    adaptive_has_edge = wins >= POOL_CONTROL_WIN_LIMIT
    control_has_edge = bool(control["evaluated"]) and bool(control["control_passes_gate"])
    quality_premise = {
        "adaptive_wins": wins,
        "adaptive_clears_bar": adaptive_has_edge,
        "control_clears_bar": control_has_edge,
        "win_limit": POOL_CONTROL_WIN_LIMIT,
        "indistinguishable": bool(control["evaluated"])
        and not adaptive_has_edge
        and not control_has_edge,
        "reading": (
            "both arms clear the bar (pool/location effect alone would pass the gate)"
            if adaptive_has_edge and control_has_edge
            else "neither arm clears the bar (no edge to recommend Gate 4 on)"
            if not adaptive_has_edge
            else "adaptive clears the bar and the loss-blind arm does not"
        ),
    }

    # Label precedence among the withholding clauses: anti-adaptivity (harm
    # guard), then the two pool-matched readings (control clears the bar, then
    # neither arm does), then the section 5.14 do-nothing clause, then a missing
    # control, then a missing null queue, then the band miss. All of them exit 2
    # and all withhold Gate 4; the verdict string must surface which one fired.
    if not cost_pass:
        verdict = "FAIL"
        exit_code = 1
    elif not harm_guard_clear:
        verdict = "PASS_COST_ONLY_HARM_GUARD_REJECTS"
        exit_code = 2
    elif control["evaluated"] and control["control_passes_gate"]:
        verdict = "PASS_COST_ONLY_POOL_MATCHED_CONTROL_PASSES"
        exit_code = 2
    elif quality_premise["indistinguishable"]:
        verdict = "PASS_COST_ONLY_NO_DEMONSTRATED_EDGE"
        exit_code = 2
    elif null["evaluated"] and null["clause_rejects"]:
        verdict = "PASS_COST_ONLY_NULL_REJECTS"
        exit_code = 2
    elif not control["evaluated"]:
        verdict = "PASS_COST_ONLY_CONTROL_MISSING"
        exit_code = 2
    elif not null["evaluated"]:
        verdict = "PASS_COST_ONLY_NULL_MISSING"
        exit_code = 2
    elif not in_band:
        verdict = "PASS_WITH_QUALIFICATION"
        exit_code = 2
    else:
        verdict = "PASS"
        exit_code = 0

    return {
        "protocol": "v0-compute-efficiency-5.1",
        "probe_batch_count": probe_batch_count,
        "declared_seeds": list(declared_seeds),
        "seeds_complete": seeds,
        "seeds_missing": missing,
        "seeds_stray": strays,
        "queue_complete": not missing,
        "measured_cost_percent": cost_percent,
        "primary_criterion_percent": PRIMARY_COST_PERCENT,
        "primary_pass": cost_pass,
        "predicted_cost_percent": predicted_percent,
        "tolerance_band_percent": [band_low, band_high],
        "in_tolerance_band": in_band,
        "band_reading": run_scale["reading"],
        "run_scale": run_scale,
        "harm_guard": {
            "adaptive_wins": wins,
            "adaptive_losses": losses,
            "pairs": len(pairs),
            "loss_limit": HARM_GUARD_LOSS_LIMIT,
            "clear": harm_guard_clear,
        },
        "pool_matched_control": control,
        "quality_premise": quality_premise,
        "null_control": null,
        "per_seed": pairs,
        "verdict": verdict,
        # The registered code, carried in the artifact as well as returned: the
        # Kaggle-side runner exits 0 whatever the decision (the decision *is*
        # the artifact), so anything that reads the verdict back needs the code
        # to travel with it rather than re-deriving it from the string.
        "exit_code": exit_code,
        "gate_4_authorized": verdict == "PASS" and not missing,
        "notes": (
            f"Gate 4 authorization requires all {len(declared_seeds)} declared "
            "seeds complete, cost <= 5%, measured C inside the [0.5, 2]x band, "
            "the harm guard clear, the adopted section 5.5 pool-matched "
            "control clause satisfied (the loss-blind arm must NOT clear the "
            "gate, the control must have been run, and the two arms must not "
            "be indistinguishable with neither clearing the bar - section "
            "5.44), the quality premise recorded, and the section 5.14 "
            "do-nothing clause satisfied (adaptive must beat the null arm on "
            "the declared queue, and the null queue must have been run). A "
            "band miss forces re-derivation before Gate 4 (no knob re-tuning "
            "against the measured C)."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", default="artifacts/datasets",
        help="root holding act-checkpoints/v0-5-1 (default artifacts/datasets)",
    )
    parser.add_argument(
        "--probe-batch-count", type=int, default=1,
        help="the run's pre-registered probe budget (1 primary, 2 secondary)",
    )
    parser.add_argument(
        "--seeds", default=None,
        help="declared seed block as 'A-B' or a comma-separated list "
        "(default: the registered queue)",
    )
    parser.add_argument(
        "--control-root", type=Path, default=None,
        help="dataset root of the pool-matched loss-blind control arm (the "
        "same seed pairs run with probe_selection=random); required for Gate 4 "
        "authorization under the adopted section 5.5 protocol",
    )
    parser.add_argument(
        "--null-root", type=Path, default=None,
        help="dataset root of the section 5.14 do-nothing queue (single-arm "
        "null sessions at declared fidelity over the same seeds); required for "
        "Gate 4 authorization under the pre-registered section 5.14 protocol",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="verdict JSON path (default <dataset-root>/v0-5-1-verdict.json)",
    )
    args = parser.parse_args()
    try:
        declared_seeds = parse_seeds(args.seeds) if args.seeds else DECLARED_SEEDS
        verdict = compute_verdict(
            Path(args.dataset_root),
            probe_batch_count=args.probe_batch_count,
            declared_seeds=declared_seeds,
            control_root=args.control_root,
            null_root=args.null_root,
        )
    except InputError as error:
        # No verdict was produced, so nothing is written and the exit code is
        # the distinct 3 - never 1, which means a real cost FAIL.
        print(f"no verdict produced: {error}", file=sys.stderr)
        return INPUT_ERROR_EXIT_CODE
    output = args.output or Path(args.dataset_root) / "v0-5-1-verdict.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    printable = {key: value for key, value in verdict.items() if key != "per_seed"}
    print(json.dumps(printable, indent=2, sort_keys=True))
    if not verdict["queue_complete"]:
        print(
            f"WARNING: {len(verdict['seeds_missing'])} declared seed(s) missing: "
            f"{verdict['seeds_missing']}"
        )
    if verdict["seeds_stray"]:
        print(
            f"WARNING: {len(verdict['seeds_stray'])} complete seed(s) outside the "
            f"declared queue are excluded from the verdict: {verdict['seeds_stray']}"
        )
    control = verdict["pool_matched_control"]
    if not control["evaluated"]:
        print(
            "WARNING: no pool-matched control supplied; the adopted section 5.5 "
            "clause cannot be evaluated and Gate 4 is withheld"
        )
    else:
        if control["seeds_missing"]:
            print(
                f"WARNING: control arm is missing declared seed(s): "
                f"{control['seeds_missing']}"
            )
        if control["control_passes_gate"]:
            print(
                f"WARNING: the loss-blind pool-matched arm won "
                f"{control['adaptive_wins']}/{control['pairs']} of its pairs "
                f"(limit {control['win_limit']}); the adaptive edge does not "
                f"demonstrate selection value and Gate 4 is withheld"
            )
    null = verdict["null_control"]
    if not null["evaluated"]:
        print(
            "WARNING: the section 5.14 do-nothing queue was not supplied (or is "
            "incomplete on the declared block); the do-nothing clause cannot be "
            "evaluated and Gate 4 is withheld"
        )
    elif null["clause_rejects"]:
        print(
            f"WARNING: the adaptive arm failed to beat the null arm on "
            f"{null['adaptive_losses_or_ties']}/{null['pairs']} seeds (limit "
            f"{null['win_limit']}); the adaptive edge does not demonstrate "
            f"value over doing nothing and Gate 4 is withheld"
        )
    return {
        "FAIL": 1,
        "PASS_WITH_QUALIFICATION": 2,
        "PASS_COST_ONLY_HARM_GUARD_REJECTS": 2,
        "PASS_COST_ONLY_POOL_MATCHED_CONTROL_PASSES": 2,
        "PASS_COST_ONLY_NULL_REJECTS": 2,
        "PASS_COST_ONLY_CONTROL_MISSING": 2,
        "PASS_COST_ONLY_NULL_MISSING": 2,
    }.get(verdict["verdict"], 0)


if __name__ == "__main__":
    raise SystemExit(main())
