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

Usage:

    PYTHONPATH=src python scripts/v0_verdict.py --dataset-root artifacts/datasets

Exit code 0 = PASS (cost + band + harm guard clear), 1 = FAIL (primary cost
miss), 2 = PASS with qualification (cost passes but the measured C lands
outside the tolerance band, or the harm guard rejects). The JSON verdict is
printed and written beside the dataset root for the artifact trail.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from act.training.session_plan import expected_probe_cost_percent, tolerance_band
from act.training.v0_handoff import HANDOFF_DATASET, published_seed_status

DECLARED_SEEDS = tuple(range(9000, 9012))
PRIMARY_COST_PERCENT = 5.0
HARM_GUARD_LOSS_LIMIT = 10  # adaptive worse in >= 10 of 12 pairs rejects


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


def compute_verdict(
    dataset_root: Path,
    *,
    probe_batch_count: int,
    declared_seeds: tuple[int, ...] = DECLARED_SEEDS,
) -> dict:
    """Assemble the machine-checked section 5.1 verdict from published seeds.

    ``declared_seeds`` defaults to the registered queue; overriding it does
    not loosen any rule (the 5% criterion, the tolerance band, the harm guard
    and the queue-completeness requirement all apply to whatever block is
    declared).
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
    if not cost_pass:
        verdict = "FAIL"
        exit_code = 1
    elif not harm_guard_clear:
        # The harm guard is the stronger blocker (anti-adaptivity, not
        # bookkeeping), so it takes label precedence over a band miss: both
        # exit 2, but the verdict string must surface the rejection.
        verdict = "PASS_COST_ONLY_HARM_GUARD_REJECTS"
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
        "per_seed": pairs,
        "verdict": verdict,
        "gate_4_authorized": verdict == "PASS" and not missing,
        "notes": (
            f"Gate 4 authorization requires all {len(declared_seeds)} declared "
            "seeds complete, cost <= 5%, measured C inside the [0.5, 2]x band, "
            "and the harm guard clear. A band miss forces re-derivation before "
            "Gate 4 (no knob re-tuning against the measured C)."
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
        "--output", type=Path, default=None,
        help="verdict JSON path (default <dataset-root>/v0-5-1-verdict.json)",
    )
    args = parser.parse_args()
    declared_seeds = parse_seeds(args.seeds) if args.seeds else DECLARED_SEEDS
    verdict = compute_verdict(
        Path(args.dataset_root),
        probe_batch_count=args.probe_batch_count,
        declared_seeds=declared_seeds,
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
    return {"FAIL": 1, "PASS_WITH_QUALIFICATION": 2, "PASS_COST_ONLY_HARM_GUARD_REJECTS": 2}.get(
        verdict["verdict"], 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
