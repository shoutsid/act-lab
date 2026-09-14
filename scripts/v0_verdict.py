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

Usage:

    PYTHONPATH=src python scripts/v0_verdict.py \
        --dataset-root artifacts/datasets \
        --control-root artifacts/datasets-control

Exit code 0 = PASS (cost + band + harm guard + control clause clear), 1 = FAIL
(primary cost miss), 2 = PASS with qualification (cost passes but the measured
C lands outside the tolerance band, the harm guard rejects, the pool-matched
control also passes, or no control was supplied). The JSON verdict is printed
and written beside the dataset root for the artifact trail.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from act.training.session_plan import expected_probe_cost_percent, tolerance_band
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
        raise SystemExit("empty seed specification")
    if len(set(seeds)) != len(seeds):
        raise SystemExit("seed specification contains duplicates")
    return seeds


def _seed_numbers(dataset_root: Path) -> list[int]:
    handoff = dataset_root / HANDOFF_DATASET
    if not handoff.is_dir():
        raise SystemExit(f"no handoff dataset under {handoff}")
    seeds = []
    for path in handoff.glob("seed-*"):
        stem = path.stem.split("-")[-1]
        if stem.isdigit() and published_seed_status(dataset_root, int(stem)) == "complete":
            seeds.append(int(stem))
    return sorted(seeds)


def _load_seed(dataset_root: Path, seed: int) -> dict:
    path = dataset_root / HANDOFF_DATASET / f"seed-{seed:04d}" / "reports" / "seed-summary.json"
    if not path.is_file():
        raise SystemExit(f"missing seed summary: {path}")
    summary = json.loads(path.read_text())
    if summary.get("status") != "complete":
        raise SystemExit(f"seed {seed} is not a complete version")
    return summary


def _arm_pairs(dataset_root: Path, seeds: list[int]) -> list[dict[str, object]]:
    """Per-seed adaptive/random final losses for a published seed block."""
    pairs: list[dict[str, object]] = []
    for seed in seeds:
        arms = _load_seed(dataset_root, seed)["arms"]
        if "random" not in arms or "adaptive" not in arms:
            raise SystemExit(f"seed {seed} is missing the random/adaptive pair")
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
    """Pool-matched loss-blind control block over the same seed pairs."""
    complete = _seed_numbers(dataset_root)
    seeds = [seed for seed in complete if seed in declared_seeds]
    missing = sorted(set(declared_seeds) - set(seeds))
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


def compute_verdict(
    dataset_root: Path,
    *,
    probe_batch_count: int,
    declared_seeds: tuple[int, ...] = DECLARED_SEEDS,
    control_root: Path | None = None,
) -> dict:
    """Assemble the machine-checked section 5.1 verdict from published seeds.

    ``declared_seeds`` defaults to the registered queue; overriding it does
    not loosen any rule (the 5% criterion, the tolerance band, the harm guard
    and the queue-completeness requirement all apply to whatever block is
    declared). ``control_root`` is the pool-matched loss-blind arm's dataset;
    the adopted section 5.5 protocol requires it, so its absence withholds the
    Gate 4 recommendation rather than silently skipping the clause.
    """
    if not declared_seeds:
        raise SystemExit("declared seed block must not be empty")
    complete = _seed_numbers(dataset_root)
    if not complete:
        raise SystemExit("no complete seed versions found; nothing to judge")
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
            raise SystemExit(f"seed {seed} is missing the random/adaptive pair")
        if int(arms["adaptive"]["mutation_count"]) != int(arms["random"]["mutation_count"]):
            raise SystemExit(f"seed {seed}: unmatched mutation budgets")
        search += int(arms["adaptive"]["search_flops"])
        training += int(arms["adaptive"]["training_flops"])
        pairs.append(
            {
                "seed": seed,
                "adaptive_loss": float(arms["adaptive"]["final_loss"]),
                "random_loss": float(arms["random"]["final_loss"]),
                "adaptive_wins": float(arms["adaptive"]["final_loss"])
                < float(arms["random"]["final_loss"]),
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
    # Label precedence among the withholding clauses: anti-adaptivity (harm
    # guard), then the pool-matched clause, then the control being absent, then
    # the band miss. All of them exit 2 and all withhold Gate 4; the verdict
    # string must surface which one fired.
    if not cost_pass:
        verdict = "FAIL"
        exit_code = 1
    elif not harm_guard_clear:
        verdict = "PASS_COST_ONLY_HARM_GUARD_REJECTS"
        exit_code = 2
    elif control["evaluated"] and control["control_passes_gate"]:
        verdict = "PASS_COST_ONLY_POOL_MATCHED_CONTROL_PASSES"
        exit_code = 2
    elif not control["evaluated"]:
        verdict = "PASS_COST_ONLY_CONTROL_MISSING"
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
        "harm_guard": {
            "adaptive_wins": wins,
            "adaptive_losses": losses,
            "pairs": len(pairs),
            "loss_limit": HARM_GUARD_LOSS_LIMIT,
            "clear": harm_guard_clear,
        },
        "pool_matched_control": control,
        "per_seed": pairs,
        "verdict": verdict,
        "gate_4_authorized": verdict == "PASS" and not missing,
        "notes": (
            f"Gate 4 authorization requires all {len(declared_seeds)} declared "
            "seeds complete, cost <= 5%, measured C inside the [0.5, 2]x band, "
            "the harm guard clear, and the adopted section 5.5 pool-matched "
            "control clause satisfied (the loss-blind arm must NOT clear the "
            "gate, and the control must have been run). A band miss forces "
            "re-derivation before Gate 4 (no knob re-tuning against the "
            "measured C)."
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
        "--output", type=Path, default=None,
        help="verdict JSON path (default <dataset-root>/v0-5-1-verdict.json)",
    )
    args = parser.parse_args()
    declared_seeds = parse_seeds(args.seeds) if args.seeds else DECLARED_SEEDS
    verdict = compute_verdict(
        Path(args.dataset_root),
        probe_batch_count=args.probe_batch_count,
        declared_seeds=declared_seeds,
        control_root=args.control_root,
    )
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
    return {
        "FAIL": 1,
        "PASS_WITH_QUALIFICATION": 2,
        "PASS_COST_ONLY_HARM_GUARD_REJECTS": 2,
        "PASS_COST_ONLY_POOL_MATCHED_CONTROL_PASSES": 2,
        "PASS_COST_ONLY_CONTROL_MISSING": 2,
    }.get(verdict["verdict"], 0)


if __name__ == "__main__":
    raise SystemExit(main())
