"""Real-data selection diagnostic: what fixes adaptive selection on real text?

The real-data proof run (``local_proof_run.py --real-data``) found the
cap-4/1-batch controller losing 11 of 12 pairs on TinyStories - the
anti-adaptivity harm guard engaged. This diagnostic isolates *why* with two
variants on the SAME paired seed block (9000-9007), same stream, same
horizon, same protocol machinery:

- ``var``  : cap-4 / 4 probe batches - quadruple probe quality at fixed
  richness (the study-6 variance hypothesis, now on real text).
- ``rich`` : cap-8 / 1 probe batch - double the candidate sample at fixed
  probe quality (the study-8/9 richness axis, now on real text).
- ``argmax`` : cap-4 / 1 probe batch with the decision rule inverted -
  deliberately select the probe-WORST candidate. The inverted-signal
  discriminator for the diagnostic's stably-misaligned finding: argmax
  beating random means the probe ranking is informative but inverted
  (fix: negate the criterion); argmax also losing means the probe is
  uninformative on real text (fix: probe horizon/tokens, not the rule).
  Same candidates, same probe batches, same search FLOPs as the baseline
  adaptive arm - only the decision rule differs.
- ``heldout`` : cap-4 / 1 probe batch / argmin, but probe windows are drawn
  from the held-out validation stream instead of the training stream. The
  section-5.3 post-argmax fix candidate: the argmax verdict ruled the
  single-batch probe ranking uninformative on real text and licensed only
  probe horizon/tokens as the fix axis. The "more probe batches" reading is
  already falsified (``var``: 0/8), so ``heldout`` tests the remaining
  "different probe data" reading: argmin may be anti-predictive because
  probe windows sit in-corpus (memorization-adjacent signal that does not
  transfer to the arm's future). Only the probe windows' source stream
  differs from the baseline adaptive arm - same candidates, same probe
  batch count, same booked probe FLOPs.

Pre-registered outcome contract (fixed before running): under ``heldout``,
the probe-rank instrumentation reports the mean selected rank (0 = probe-
best) and the paired losses decide the arm. Heldout-argmin winning its
pairs would mean in-corpus probe windows are the mechanism and probe-data
provenance is the fix; heldout-argmin losing like train-argmin (mean delta
~ +0.29) falsifies the provenance hypothesis and leaves the ranking
uninformative under both sources - at which point the probe's information
content, not its data, is the binding constraint (longer probe horizon is
the remaining licensed lever, at 4x cost). Either outcome is decisive:
mechanism claims are directly measurable via ``probe_landscape`` ranks.

Per-event probe cost scales as candidates x batches, so ``var`` quadruples
the baseline cost (~1% -> ~4% of training FLOPs, measured 3.99%) and
``rich`` doubles it (measured 1.98%); ``argmax`` matches the baseline's
~1%. All are under the 5% criterion, and this is a diagnostic, not the
pre-registered cost protocol. Comparisons are paired: each variant runs its
own random arm per seed, and final losses are also compared against the
baseline adaptive arm's published summaries for the same seeds.

Every variant runs with ``probe_landscape=True``, so each adaptive event's
mutation history carries the full per-candidate probe ranking (sorted best-
first) with the selected candidate's index. ``analyze()`` aggregates a
per-variant ``probe_rank`` block (mean selected rank, rank histogram, near-
top rate), making selection-mechanism claims directly measurable. Existing
summaries pre-dating the instrumentation count zero landscape events; rerun
with ``--force`` (previous results are moved aside with a timestamp suffix).

Usage:
  PYTHONPATH=src python scripts/real_data_diagnostic.py --dry-run
  PYTHONPATH=src python scripts/real_data_diagnostic.py               # all variants
  PYTHONPATH=src python scripts/real_data_diagnostic.py --variants argmax
  PYTHONPATH=src python scripts/real_data_diagnostic.py --variants heldout --force
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from act.training import V0SessionConfig  # noqa: E402
from act.training.v0_runner import run_pair_session  # noqa: E402

DIAG_ROOT = ROOT / "artifacts" / "real-diag"
BASELINE_ROOT = ROOT / "artifacts" / "real-proof" / "datasets"
DEFAULT_SEEDS = tuple(range(9000, 9008))

VARIANTS: dict[str, dict[str, object]] = {
    "var": {"probe_candidate_cap": 4, "probe_batch_count": 4},
    "rich": {"probe_candidate_cap": 8, "probe_batch_count": 1},
    "argmax": {"probe_selection": "argmax"},
    "heldout": {"probe_data_source": "heldout"},
}


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _run_variant(
    name: str,
    overrides: dict[str, int],
    seeds: tuple[int, ...],
    steps: int,
    tokens_per_second: float,
) -> Path:
    from local_proof_run import PROOF_CONFIG, REAL_VOCAB_SIZE, _ensure_real_streams

    data_path, val_path = _ensure_real_streams()
    config = V0SessionConfig(
        **{
            **PROOF_CONFIG.__dict__,
            "total_steps": steps,
            "vocab_size": REAL_VOCAB_SIZE,
            "data_path": str(data_path),
            "val_data_path": str(val_path),
            # Per-candidate probe ranks land in each adaptive event's history
            # (probe_landscape / probe_selected_index), so mechanism claims are
            # directly measurable from the published seed summaries.
            "probe_landscape": True,
            **overrides,
        }
    )
    session_dir = DIAG_ROOT / name / "sessions"
    dataset_root = DIAG_ROOT / name / "datasets"
    session_dir.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    _log(
        f"variant {name}: cap={config.probe_candidate_cap} "
        f"batches={config.probe_batch_count} "
        f"selection={config.probe_selection} "
        f"probe_data={config.probe_data_source} seeds={list(seeds)}"
    )
    run_pair_session(
        seeds=seeds,
        config=config,
        session_dir=session_dir,
        dataset_root=dataset_root,
        code_revision=_git_revision(),
        remaining_hours=12.0,
        measured_tokens_per_second=tokens_per_second,
        output=dataset_root / "local-proof-session.json",
        log=_log,
    )
    return dataset_root


def _seed_losses(dataset_root: Path, seeds: tuple[int, ...]) -> dict[int, dict[str, float]]:
    losses: dict[int, dict[str, float]] = {}
    for seed in seeds:
        path = (
            dataset_root / "act-checkpoints" / "v0-5-1" / f"seed-{seed:04d}"
            / "reports" / "seed-summary.json"
        )
        if not path.is_file():
            continue
        summary = json.loads(path.read_text())
        if summary.get("status") != "complete":
            continue
        arms = summary["arms"]
        losses[seed] = {
            "random": float(arms["random"]["final_loss"]),
            "adaptive": float(arms["adaptive"]["final_loss"]),
        }
    return losses


def _landscape_stats(dataset_root: Path, seeds: tuple[int, ...]) -> dict:
    """Aggregate per-event selected probe ranks from published seed summaries.

    Landscape instrumentation (``probe_landscape: True``) records each
    event's full candidate ranking with the selected candidate's index, so
    ``selected_rank`` is directly measurable: rank 0 = the probe-LOWEST-loss
    candidate, rank k-1 = the probe-highest. ``near_top_rate`` is the share of
    selections within one candidate of the probe-best pick.
    """
    ranks: list[int] = []
    candidate_counts: list[int] = []
    for seed in seeds:
        path = (
            dataset_root / "act-checkpoints" / "v0-5-1" / f"seed-{seed:04d}"
            / "reports" / "seed-summary.json"
        )
        if not path.is_file():
            continue
        summary = json.loads(path.read_text())
        if summary.get("status") != "complete":
            continue
        for arm in summary["arms"].values():
            for event in arm.get("mutation_history", []):
                landscape = event.get("probe_landscape")
                if not landscape:
                    continue
                ranks.append(int(event["probe_selected_index"]))
                candidate_counts.append(len(landscape))
    if not ranks:
        return {
            "events_with_landscape": 0,
            "mean_selected_rank": None,
            "selected_rank_counts": {},
            "near_top_rate": None,
        }
    rank_counts: dict[str, int] = {}
    for rank in ranks:
        key = str(rank)
        rank_counts[key] = rank_counts.get(key, 0) + 1
    n = len(ranks)
    return {
        "events_with_landscape": n,
        "mean_selected_rank": sum(ranks) / n,
        "selected_rank_counts": dict(sorted(rank_counts.items(), key=lambda kv: int(kv[0]))),
        "near_top_rate": sum(1 for rank in ranks if rank <= 1) / n,
    }


def analyze(seeds: tuple[int, ...], steps: int) -> dict:
    baseline = _seed_losses(BASELINE_ROOT, seeds)
    report: dict[str, object] = {
        "protocol": "real-data-selection-diagnostic",
        "seeds": list(seeds),
        "steps": steps,
        "baseline": "cap-4/1-batch (artifacts/real-proof)",
        "variants": {},
    }
    for name, overrides in VARIANTS.items():
        root = DIAG_ROOT / name / "datasets"
        losses = _seed_losses(root, seeds)
        paired = []
        for seed in sorted(set(losses) & set(baseline)):
            paired.append(
                {
                    "seed": seed,
                    "adaptive": losses[seed]["adaptive"],
                    "random": losses[seed]["random"],
                    "adaptive_wins": losses[seed]["adaptive"] < losses[seed]["random"],
                    "delta_vs_baseline_adaptive": losses[seed]["adaptive"]
                    - baseline[seed]["adaptive"],
                    "baseline_random": baseline[seed]["random"],
                    "baseline_adaptive": baseline[seed]["adaptive"],
                }
            )
        wins = sum(1 for p in paired if p["adaptive_wins"])
        mean_delta = (
            sum(p["adaptive"] - p["random"] for p in paired) / len(paired) if paired else None
        )
        mean_vs_baseline = (
            sum(p["delta_vs_baseline_adaptive"] for p in paired) / len(paired)
            if paired
            else None
        )
        report["variants"][name] = {
            "config": overrides,
            "pairs": paired,
            "adaptive_wins": wins,
            "pairs_scored": len(paired),
            "mean_delta_vs_own_random": mean_delta,
            "mean_adaptive_vs_baseline_adaptive": mean_vs_baseline,
            "random_replicates_baseline": all(
                abs(p["random"] - p["baseline_random"]) < 1e-4 for p in paired
            ),
            "probe_rank": _landscape_stats(root, seeds),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        default="var,rich,argmax,heldout",
        help="comma list from: var,rich,argmax,heldout",
    )
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--benchmark-steps", type=int, default=20)
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun even if complete results exist (previous results are "
        "moved aside with a timestamp suffix; the runner skips completed "
        "pairs unless their state is removed first)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}; choose from {sorted(VARIANTS)}")

    from local_proof_run import _measure_tokens_per_second

    from local_proof_run import PROOF_CONFIG, REAL_VOCAB_SIZE, _ensure_real_streams

    data_path, val_path = _ensure_real_streams()
    probe_config = V0SessionConfig(
        **{
            **PROOF_CONFIG.__dict__,
            "total_steps": args.steps,
            "vocab_size": REAL_VOCAB_SIZE,
            "data_path": str(data_path),
            "val_data_path": str(val_path),
        }
    )
    tokens_per_second = _measure_tokens_per_second(probe_config, steps=args.benchmark_steps)
    tokens_per_arm = probe_config.training_tokens()
    pair_hours = tokens_per_arm / tokens_per_second / 3600.0 + 0.1
    _log(
        f"plan: {len(names)} variant(s) x {len(seeds)} seeds, ~{pair_hours:.2f} h/pair, "
        f"~{len(names) * len(seeds) * pair_hours:.1f} h total"
    )
    for name in names:
        _log(f"  {name}: {VARIANTS[name]}")
    if args.dry_run:
        return 0

    for name in names:
        if args.force:
            has_results = any((DIAG_ROOT / name).iterdir()) if (DIAG_ROOT / name).is_dir() else False
            if has_results:
                stamp = time.strftime("%Y%m%d-%H%M%S")
                aside = DIAG_ROOT / f"{name}-pre-landscape-{stamp}"
                (DIAG_ROOT / name).rename(aside)
                _log(f"{name}: previous results moved aside -> {aside.name}")
        _run_variant(name, VARIANTS[name], seeds, args.steps, tokens_per_second)

    report = analyze(seeds, args.steps)
    out = DIAG_ROOT / "diagnostic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _log(f"analysis written: {out}")
    for name in names:
        entry = report["variants"][name]
        ranks = entry["probe_rank"]
        _log(
            f"{name}: adaptive wins {entry['adaptive_wins']}/{entry['pairs_scored']}, "
            f"mean delta vs own random {entry['mean_delta_vs_own_random']:.4f}, "
            f"mean adaptive vs baseline adaptive {entry['mean_adaptive_vs_baseline_adaptive']:.4f}, "
            f"random replicates baseline: {entry['random_replicates_baseline']}, "
            f"mean selected probe rank {ranks['mean_selected_rank']}, "
            f"ranks {ranks['selected_rank_counts']}"
        )
    return 0


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
