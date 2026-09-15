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
- ``long`` : cap-4 / 1 probe batch / argmin with ``probe_finetune_steps=1``
  - the long-horizon probe as a one-step lookahead. Each probed candidate is
  fine-tuned for 1 AdamW step on a freshly drawn training batch before its
  probe loss is measured on the same probe batches, so selection scores
  post-mutation trainability instead of the instantaneous loss dip (a fresh
  expert looks bad before any adaptation, which plausibly biases argmin
  against grows - the baseline's observed 4/20 prune bias). Cost adds ~2%
  of training FLOPs at the proof configuration (total ~3.0%, under the 5%
  criterion; R=2 would sit exactly at the 5% boundary with no margin, so
  R=1 is the only criterion-clean setting); the decision rule, candidate
  cap, and probe-batch yardstick are unchanged from the baseline adaptive
  arm.

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

OUTCOME (run 2026-09-14, revision ``eaa1c45``): heldout 0/8 wins, mean
delta +0.2840 (baseline argmin +0.2858) - provenance FALSIFIED with the
manipulation verified real (selected probe losses moved 4.4195 -> 4.2848,
9/24 picks changed).

Pre-registered outcome contract for ``long`` (fixed before running): the
mechanism claim under test is that the snapshot probe's anti-predictive
ranking is an adaptation-blindness artifact (scoring instantaneous loss
instead of post-mutation trainability), visible as argmin's prune bias.
(a) ``long`` beating its random arm on the paired block would mean the
snapshot probe's blindness is the mechanism and lookahead selection is a
fix candidate worth pre-registering on a fresh seed block. (b) ``long``
losing like the baseline (mean delta ~ +0.29) would mean one adaptation
step does not restore ranking information - the probe's blindness is not
(or not only) a missing-adaptation artifact, and the remaining licensed
lever is probe token count at fixed fidelity. Secondary observable: the
selected operation balance (``probe_landscape`` makes it native) - a
long-run argmin that still picks ~5:1 prune:grow while winning would be
evidence the prune bias itself was not the failure mode. Cost guardrail:
each lookahead step adds ~2% of training FLOPs, so R=1 lands ~3.0% while
R=2 sits exactly at the 5% criterion with no margin (measured ratios run
slightly above predicted on real data) - the variant is run at R=1 or not
at all.

OUTCOME (run 2026-09-14, revision ``eaa1c45``): long 0/8 wins, mean delta
+0.3106 - adaptation blindness FALSIFIED. The lookahead re-ranked as
designed (grows 5/24; seed 9000 top-ranked the grow at 2/3 events - the
snapshot probe's prune bias is a ranking artifact) but not toward what
helps; across seeds mean selected probe loss correlates positively with
final harm (long +0.51, heldout +0.89; 8 points, indicative). Unified
verdict and next direction live in report section 5.3.

- ``gated`` : NOT a ranking variant at all - the section-5.4 pre-registered
  fix candidate after all five ordering variants lost. A single-arm session
  that draws one random mutation per event (the random arm's exact chooser
  semantics, drawn first so its load-fallback sees post-training routing
  statistics), scores the two-point contrast - the no-op reference (current
  model) vs the drawn mutation on the same probe batches - and accepts the
  draw unless ``draw_probe_loss > reference_probe_loss + gate_margin``
  (theta, default 0); a rejected draw rolls back and applies the load-based
  grow fallback so the mutation budget stays matched. Uses only the probe's
  ABSOLUTE signal, abandoning the anti-predictive ordering entirely. The
  arm shares the random arm's init identity and streams, so theta=inf is
  bit-identical to random (the built-in control; pinned by test). Cost: two
  probe evaluations per live event, ~2% of training FLOPs. Paired against
  the baseline block's stored random arms for the same seeds.

Pre-registered outcome contract for ``gated`` (fixed before running):
(a) beating the paired random arms (mean delta < 0 or wins >= 5/8) means
absolute-harm gating carries real signal the ranking lacks - pre-register
an intermediate-theta sweep on a fresh block. (b) losing/tying like the
baseline argmin (mean delta in the +0.28..+0.31 band) means the absolute
signal is as empty as the ordering, concluding the probe-on-real-text
program as falsified in both usable forms (ordering and threshold) - the
next direction leaves probe-derived selection entirely (e.g.
routing-statistics heuristics). Secondary observables: accept rate (a
near-1 rate makes the arm an expensive random replica and falsifies the
gate's discriminativeness) and the mean draw-minus-reference contrast.

OUTCOME (run 2026-09-14, revision ``ef4bacd``): outcome (a)'s letter fired,
its content never exercised - gated 8/8 wins, mean delta ``-0.1217`` (the
first negative real-text delta of any treatment), BUT accept rate 0/24: all
draws probed harmful (mean draw-minus-reference ``+0.1065``), every
fallback dead at ``expert_count=8`` (``reject_noop`` x24), so the arm
counted 0 mutations in 8/8 seeds and won as the NULL arm - matched-budget
churn vs no churn, not gate discrimination. Draws were bit-identical to the
random arms' own operations (24/24; the built-in control held). First
direct measurement of real-text churn harm; partial explanation of every
earlier variant's loss (all spent the full budget). Follow-ups (own
pre-registrations): a do-nothing control arm on a fresh block; an
intermediate-theta sweep (theta 0.05/0.1 would have accepted 11/24 and
21/24 of the observed draws). Full write-up: report section 5.7.

- ``null`` : the section-5.8 do-nothing control - a single-arm session
  (``V0SessionConfig(arms=("null",))``) that shares the random arm's init
  identity (offset 1), training stream, and event cadence, but performs no
  mutation at any structural event: each event appends one recorded no-op
  (operation "null"), books no probe FLOPs, and consumes the budget slot,
  so the arm is budget-matched with zero churn. Paired against the
  baseline block's stored random arms, it measures pure churn harm at
  matched budget on this block.

Pre-registered outcome contract for ``null`` (fixed before running):
(a) the null arm beating the paired random arms (mean delta < 0 or wins
>= 5/8) establishes churn harm as a real effect on this block and
re-labels every real-data variant's loss as partly churn harm; the gated
follow-up question then becomes whether ACCEPTED draws beat doing
nothing, not whether gating beats random. (b) a null-vs-random delta
inside +/-0.1 nats would mean the gated run's -0.12 was seed luck, and
the gate program stays closed. Built-in checks: the null arm must book
zero search FLOPs and zero mutations, and its final losses should
reproduce the gated run bit-for-bit on the same seeds (the gated arm
accepted nothing - this is the replication check of the section-5.4 run
at essentially zero marginal cost).

OUTCOME (run 2026-09-14, revision ``9e5e448``): outcome (a) - churn harm
is REAL. null 8/8 wins, mean delta ``-0.1217`` (per-seed -0.08 to -0.20,
all negative); zero mutations / zero search FLOPs in 8/8 seeds; and the
replication check is EXACT - max |null - gated| = 0.000000 on all eight
seeds, the null arm reproducing the gated run bit-for-bit as the section
5.7 degeneracy reading predicted. Every section-5.3 variant's loss is
partly churn harm; the theta sweep's control should be the null arm; the
honest real-text ladder for Gate 4 is null < random < adaptive.Verdict machine-checked by ``scripts/null_verdict.py``; write-up in report
section 5.8.

- ``heuristic`` : the section-5.13 routing-statistics heuristic - the
  section-5.11 contract's named non-probe direction. A single-arm session
  (``V0SessionConfig(arms=("heuristic",))``): deterministic, grow-only,
  growing one expert at the argmax-routing-load layer (the same per-layer
  load key the random arm's fallback reads) every event, no chooser draw,
  no probes, zero search FLOPs beyond the post-grow rebuild. Paired
  against the theta sweep's published null arms on the same seeds, it
  asks the last open real-text question: can DIRECTED churn beat doing
  nothing when probe-ordered and gated churn both lost to it?

Pre-registered outcome contract for ``heuristic`` (fixed before running):
(a) heuristic beats the paired null arms (sign test, 9/10 on the fresh
block 9008-9017, Bonferroni alpha/2 = 0.025): directed churn carries
real signal on real text - the non-probe selection program OPENS, and a
powered follow-up gets pre-registered. (b) heuristic loses (or ties):
the real-text ladder closes at the top too - no topology mutation policy
beats doing nothing at this fidelity, and the next direction is no
real-data topology mutation at this scale (resources move to Gate 4's
compute-efficiency protocol). Built-in checks: zero probe FLOPs, all
mutations are grows at the measured argmax-load layer, and every
heuristic loss must sit within the block's observed loss range (a far
outlier means the stream identity broke). Should the null block ever be
reproduced elsewhere, expect bit-identical null values - the heuristic
arm does not perturb them (it consumes no chooser RNG and shares the
null arm's init identity and stream).

OUTCOME (2026-09-15, revision ``0ef29de``): interpretation branch (b) -
the heuristic LOSES to the null arm on every seed (0/10 wins, mean
+0.1218, per-seed +0.049 to +0.410, p 1.0 vs the 9/10 bar). Directed
churn is still churn: growing an untrained expert at the argmax-load
layer is the maximally disruptive single mutation. Built-in checks
held (30/30 grows at the measured load-argmax layer, rebuild-only
search FLOPs). The real-text topology program is CLOSED at this
fidelity; the one remaining open gate is Gate 4.

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
  PYTHONPATH=src python scripts/real_data_diagnostic.py --variants long --force
  PYTHONPATH=src python scripts/real_data_diagnostic.py --variants gated
  PYTHONPATH=src python scripts/real_data_diagnostic.py --variants null
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
    "long": {"probe_finetune_steps": 1},
    # Section 5.4's fix candidate: accept/reject gating of random draws by
    # the probe's ABSOLUTE signal (draw vs no-op reference, margin theta).
    # A single-arm session - the baseline comparison comes from the same
    # seeds' published baseline random arms, and the arm is its own control
    # at theta=inf (it reproduces the random arm exactly).
    "gated": {"arms": ("gated",), "gate_margin": 0.0},
    # Section 5.8's do-nothing control (the gated outcome's follow-up):
    # matched-budget churn vs NO churn. A single-arm session that shares the
    # random arm's init identity and training stream but performs no
    # mutation at any structural event (each event is one recorded no-op;
    # zero probe FLOPs). Paired against the baseline block's stored random
    # arms; its losses should reproduce the gated run bit-for-bit on the
    # same seeds, because the gated arm accepted nothing.
    "null": {"arms": ("null",)},
    # Section 5.13: the routing-statistics heuristic (the section-5.11
    # contract's named non-probe direction): deterministic grow at the
    # argmax-routing-load layer, every event, no probes. Single-arm, paired
    # against the theta sweep's published null arms on the same seeds.
    "heuristic": {"arms": ("heuristic",)},
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
        f"probe_data={config.probe_data_source} "
        f"lookahead={config.probe_finetune_steps} "
        f"gate_margin={config.gate_margin} arms={config.arms} seeds={list(seeds)}"
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


def _seed_losses(
    dataset_root: Path,
    seeds: tuple[int, ...],
    arms: tuple[str, ...] = ("random", "adaptive"),
) -> dict[int, dict[str, float]]:
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
        summary_arms = summary["arms"]
        losses[seed] = {name: float(summary_arms[name]["final_loss"]) for name in arms}
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


def _gate_stats(dataset_root: Path, seeds: tuple[int, ...]) -> dict:
    """Aggregate gate decisions from a gated variant's published summaries.

    The gated arm's mechanism observables: how often the absolute-harm gate
    accepted the draw, rejected it onto the load-based grow fallback, or hit
    a no-op, plus the mean draw-vs-reference probe contrast on decided
    events (negative = the accepted draws were on average probe-better than
    the no-op).
    """
    decisions: dict[str, int] = {}
    contrasts: list[float] = []
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
        for event in summary["arms"].get("gated", {}).get("mutation_history", []):
            decision = event.get("gate_decision")
            if decision is None:
                continue
            decisions[decision] = decisions.get(decision, 0) + 1
            if event.get("draw_probe_loss") is not None and event.get(
                "reference_probe_loss"
            ) is not None:
                contrasts.append(
                    float(event["draw_probe_loss"])
                    - float(event["reference_probe_loss"])
                )
    total = sum(decisions.values())
    return {
        "events_with_decision": total,
        "decision_counts": dict(sorted(decisions.items())),
        "accept_rate": (
            decisions.get("accept", 0) / total if total else None
        ),
        "mean_draw_minus_reference": (
            sum(contrasts) / len(contrasts) if contrasts else None
        ),
    }


def _heuristic_stats(dataset_root: Path, seeds: tuple[int, ...]) -> dict:
    """Aggregate the heuristic arm's bookkeeping from published summaries.

    Reports the mutation count (all grows, one per event at this config),
    the per-layer grow histogram (the load-argmax targeting's footprint),
    and the total search FLOPs (the post-grow rebuilds only - zero probe
    bookings is the built-in check).
    """
    mutations = 0
    grows: dict[str, int] = {}
    search_flops = 0
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
        arm = summary["arms"].get("heuristic", {})
        mutations += int(arm.get("mutation_count", 0))
        search_flops += int(arm.get("search_flops", 0))
        for event in arm.get("mutation_history", []):
            if event.get("operation") == "grow":
                grows[str(event.get("layer_id"))] = grows.get(str(event.get("layer_id")), 0) + 1
    return {
        "mutations": mutations,
        "grow_layers": dict(sorted(grows.items())),
        "search_flops": search_flops,
    }


def _null_stats(dataset_root: Path, seeds: tuple[int, ...]) -> dict:
    """Aggregate the null arm's bookkeeping (zero mutations by construction).

    Any nonzero mutation count here means the arm is not the do-nothing
    control it claims to be; the count is reported so the check is
    machine-readable from the published diagnostic.
    """
    events = 0
    mutations = 0
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
        arm = summary["arms"].get("null", {})
        mutations += int(arm.get("mutation_count", 0))
        events += len(arm.get("mutation_history", []))
    return {"events_recorded": events, "mutations": mutations}


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
        single_arm = overrides.get("arms")
        arm_names = (
            single_arm
            if single_arm in (("gated",), ("null",), ("heuristic",))
            else ("random", "adaptive")
        )
        losses = _seed_losses(root, seeds, arm_names)
        treatment = arm_names[0] if len(arm_names) == 1 else "adaptive"
        paired = []
        for seed in sorted(set(losses) & set(baseline)):
            paired.append(
                {
                    "seed": seed,
                    "treatment": treatment,
                    treatment: losses[seed][treatment],
                    "random": losses[seed].get(
                        "random", baseline[seed]["random"]
                    ),
                    "adaptive_wins": losses[seed][treatment]
                    < losses[seed].get("random", baseline[seed]["random"]),
                    "delta_vs_baseline_adaptive": losses[seed][treatment]
                    - baseline[seed]["adaptive"],
                    "baseline_random": baseline[seed]["random"],
                    "baseline_adaptive": baseline[seed]["adaptive"],
                }
            )
        wins = sum(1 for p in paired if p["adaptive_wins"])
        mean_delta = (
            sum(p[treatment] - p["random"] for p in paired) / len(paired) if paired else None
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
        if treatment == "gated":
            report["variants"][name]["gate"] = _gate_stats(root, seeds)
        if treatment == "null":
            report["variants"][name]["null"] = _null_stats(root, seeds)
        if treatment == "heuristic":
            report["variants"][name]["heuristic"] = _heuristic_stats(root, seeds)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        default="var,rich,argmax,heldout,long,gated",
        help="comma list from: var,rich,argmax,heldout,long,gated",
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
        gate = entry.get("gate")
        gate_text = (
            f", gate accept {gate['accept_rate']}, decisions {gate['decision_counts']}, "
            f"mean draw-minus-reference {gate['mean_draw_minus_reference']}"
            if gate
            else ""
        )
        _log(
            f"{name}: adaptive wins {entry['adaptive_wins']}/{entry['pairs_scored']}, "
            f"mean delta vs own random {entry['mean_delta_vs_own_random']:.4f}, "
            f"mean adaptive vs baseline adaptive {entry['mean_adaptive_vs_baseline_adaptive']:.4f}, "
            f"random replicates baseline: {entry['random_replicates_baseline']}, "
            f"mean selected probe rank {ranks['mean_selected_rank']}, "
            f"ranks {ranks['selected_rank_counts']}"
            f"{gate_text}"
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
