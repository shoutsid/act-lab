"""Section 5.8 verdict: the do-nothing control arm vs the baseline random arms.

Machine-checks the pre-registered contract from RESEARCH_REPORT.md 5.8:
  - built-in checks: zero search FLOPs / zero mutations, and the replication
    check |null - gated| per seed (the gated arm accepted nothing, so the
    null run should reproduce its losses);
  - outcome (a): null beats the paired random arms (mean delta < 0 or
    wins >= 5/8) -> churn harm is real on this block;
  - outcome (b): |mean delta| within +/-0.1 nats -> the gated -0.12 was seed
    luck; the gate program stays closed;
  - outcome (c): null loses to random (mean delta > 0) -> churn is net
    helpful at this schedule; the churn-harm reading of 5.7 is retracted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEEDS = tuple(range(9000, 9008))
CHECKPOINTS = "act-checkpoints/v0-5-1"


def _loss(root: Path, seed: int, arm: str) -> float | None:
    path = root / CHECKPOINTS / f"seed-{seed:04d}" / "reports" / "seed-summary.json"
    if not path.is_file():
        return None
    summary = json.loads(path.read_text())
    if summary.get("status") != "complete":
        return None
    arm_data = summary["arms"].get(arm)
    return None if arm_data is None else float(arm_data["final_loss"])


def main() -> int:
    diag = ROOT / "artifacts" / "real-diag"
    null_root = diag / "null" / "datasets"
    gated_root = diag / "gated" / "datasets"
    baseline_root = ROOT / "artifacts" / "real-proof" / "datasets"

    rows: list[tuple[int, float, float, float]] = []
    for seed in SEEDS:
        null_loss = _loss(null_root, seed, "null")
        baseline = _loss(baseline_root, seed, "random")
        gated_loss = _loss(gated_root, seed, "gated")
        if null_loss is None or baseline is None:
            missing = [n for n, v in (("null", null_loss), ("random", baseline)) if v is None]
            print(f"seed {seed}: incomplete ({', '.join(missing)}); verdict withheld")
            return 1
        rows.append((seed, null_loss, baseline, gated_loss if gated_loss is not None else float("nan")))

    built_in_zero = True
    for seed in SEEDS:
        path = null_root / CHECKPOINTS / f"seed-{seed:04d}" / "reports" / "seed-summary.json"
        summary = json.loads(path.read_text())
        arm = summary["arms"]["null"]
        if arm["mutation_count"] != 0 or arm["search_flops"] != 0:
            print(f"seed {seed}: BUILT-IN CHECK FAILED (mutations={arm['mutation_count']}, search_flops={arm['search_flops']})")
            built_in_zero = False
    if built_in_zero:
        print("built-in check: zero mutations and zero search FLOPs in 8/8 seeds: OK")

    gaps = [abs(n - g) for _, n, _, g in rows]
    max_gap = max(gaps)
    print(f"replication check vs gated run: max |null - gated| = {max_gap:.6f}")
    if max_gap <= 1e-4:
        print("  <= 1e-4: the gated run REPLICATES (per section 5.8, at most an environment note)")
    else:
        print("  > 1e-4: environment-level divergence (the section 9.11 portability caveat)")

    deltas = [n - r for _, n, r, _ in rows]
    wins = sum(1 for d in deltas if d < 0)
    mean_delta = sum(deltas) / len(deltas)
    per_seed = ", ".join(
        f"{seed}: {n:.4f} vs {r:.4f} ({n - r:+.4f})" for seed, n, r, _ in rows
    )
    print(f"\nper-seed (null vs baseline random): {per_seed}")
    print(f"wins {wins}/8, mean delta {mean_delta:+.4f}")

    if mean_delta < 0 or wins >= 5:
        outcome = "(a) churn harm is REAL on this block"
        consequence = (
            "every real-data variant's loss is re-read as partly churn harm; the gate"
            " follow-up becomes whether ACCEPTED draws beat doing nothing"
        )
    elif mean_delta <= 0.1:
        outcome = "(b) the gated -0.12 was seed luck"
        consequence = "the gate program stays closed; the churn-harm reading is retracted"
    else:
        outcome = "(c) churn is net helpful at this schedule"
        consequence = "the churn-harm reading of section 5.7 is retracted"
    print(f"\nOUTCOME {outcome}")
    print(f"consequence: {consequence}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
