# arch-zoo-100m-v3-crossed-horizon10x — Does the Attribution Survive 10× the Training?

**Design:** `crossed`, factors `depth × hidden_size`, 2 levels each, 2 replicates
(8 candidates), total parameters matched to 78M ±5%, zoo seed 42 — **the
byte-identical candidate set of `arch-zoo-100m-v3-crossed`** (verified by
`--dry-run` before launch: the two `zoo_definition.json` files differ in
`experiment_id` alone). Plan `flat` at **1200 and 12,000 steps**, seeds **42/43**,
context 1024, batch 2, **fp16 autocast**, on **Kaggle T4×2** — one sweep per
device, one seed each — via `scripts/kaggle_queue.py --queue archzoo`.

## Why this arm exists

Every effect the phase has measured is at 1200 steps (2.4M tokens), and §11 of the
findings says the one thing the record cannot claim: the decay in §3 is measured
over 400→1200 only, "nothing here shows it holds at 10× the horizon, which is the
regime a real training budget occupies." The literature makes that a live risk
rather than a formality: at convergence prior work favours *deep-narrow* over
shallow-wide at fixed parameters (Tay et al. 2021, "Scale Efficiently",
https://arxiv.org/abs/2109.10686; Petty et al. 2023,
https://arxiv.org/abs/2310.19956), while the crossed arm found **shallow beats deep
by +0.08 nats** at matched total parameters. Width winning is consistent with prior
work (Levine et al. 2020, https://arxiv.org/abs/2006.12467), and the head effect
appearing only at wide hidden is consistent with the low-rank head bottleneck
(Bhojanapalli et al. 2020, https://arxiv.org/abs/2002.05202). So the finding most
likely to be an early-training artefact is the **depth sign**, and that is the
primary quantity here.

The design is the crossed factorial rather than the matched zoo because the
factorial is what identifies depth separately from width; the matched set could
only report a ranking.

## What the record says on these seeds, for reference

Computed from `arch-zoo-100m-v3-crossed/flat1200_seed42/43_results.jsonl` (fp32):

| cell (depth, hidden) | mean val loss, seeds 42/43 |
|---|---|
| D5, H512 | 2.9097 |
| D5, H896 | 2.7584 |
| D14, H512 | 3.0353 |
| D14, H896 | 2.8011 |

Depth main effect (D14 − D5) **+0.0842**, width main effect (H896 − H512)
**−0.1928**, on these two seeds. Under §3's measured decay (1.65× per doubling of
training; 1200→12,000 is 3.32 doublings, ÷5.3) the *point predictions* at 12,000
steps are depth ≈ +0.016 and width ≈ −0.036 nats, with the seed std decaying from
0.0148 to roughly 0.004. Those are what "the decay extrapolates" looks like; the
rule below does not depend on them.

## Pre-registered decision rule (all numeric)

Evaluated on the **12,000-step** rows, pooled over seeds 42 and 43, by
`crossed_contrast` — the exact re-split test over the design's blocks (2 seeds ×
2 levels of the other gene = 4 blocks of 4 candidates, C(4,2)⁴ = **1,296
assignments, floor 2/1,296 = 0.0015**). Thresholds live in `HORIZON_RULE`
(`horizon_decision`, `--horizon-rule`) so the branch is read mechanically once the
ledgers land.

| outcome | condition | reading |
|---|---|---|
| **shallow-wins persists** | depth main effect `> 0` and `p < 0.05` | the depth sign is horizon-stable to 10×; §2c's "spend at low depth" stands at this scale, quoted with its length |
| **depth reverses** | depth main effect `< 0` and `p < 0.05` | the 1200-step depth finding was early-training dynamics, as the literature predicts; §2c's depth recommendation is **withdrawn** for realistic horizons and the record is amended, not overwritten |
| **depth washes out** | `p >= 0.05`, either sign | the depth lever is horizon-limited: no direction is recommended, and the search should treat depth as a cost knob only |

Secondary, reported whatever the primary says:

- **Width main effect at 12,000** (sign, nats, p) — expected to hold negative; a
  sign flip here would reopen the phase's headline and is reported as such.
- **Interaction** — with two seeds its seed-blocked floor is 2/2² = 0.5, so it is
  descriptive only; the re-split p is printed beside it.
- **Ordering:** Spearman between the 8 candidates' 12,000-step means and their
  1200-step fp32 means from the record on the same seeds. §3 found the ordering
  horizon-stable over 3×; this is the 10× check.
- **Decay, measured inside one precision:** spread and seed std at 12,000 against
  the arm's own 1200-step fp16 anchor, as a per-doubling rate to set beside §3's
  1.65×.
- **Precision offset:** per candidate, anchor (fp16, 1200) minus record (fp32,
  1200), same seed. This is the number that says how far fp16 losses sit from the
  record's, and it is measured rather than assumed.

Preconditions: both seeds complete at both lengths (16 rows per seed). A partial
arm is analysed over the seeds that finished with the gap **named** in the
decision (the in-flight-seed disclosure the analyzer already makes); with a single
seed the exact test's floor is 2/36 = 0.056 > 0.05, so no branch can be reached
and the rule returns `open-floor-limited` rather than a verdict.

## Precision, and why the anchor

fp16 is chosen for speed on Turing tensor cores (the T4 dtype probe measured
2.56× fp32). The record is fp32, and `RunConfig` records precision with every row
because a loss in one precision is not comparable to a loss in another. Two
consequences are designed around rather than caveated:

1. The **primary** compares 12,000-step rows to each other under one precision,
   so it needs no cross-precision comparison at all.
2. The **decay** is measured 1200→12,000 *within fp16* via the anchor, and the
   anchor-vs-record gap measures the precision offset directly. The anchor costs
   ~8 × 3 min per seed.

## Cost

32 runs, 216.3M tokens per seed (8 × 13,200 steps × 2,048 tokens), 432.6M total.
At the driver's deliberately conservative 9,000 tok/s (fp16 on a T4 at ~78M) that
is **~6.7 GPU-hours per card, run in parallel in one ~6.7 h session**, inside the
12 h Kaggle cap with the kernel's own 11 h deadline as the binding guard. The
per-run pace is read off the first finished candidate in the kernel log rather
than assumed. Kaggle bills a T4×2 session by the hour, so the quota cost is ~7 h of
the 30 GPU-h week.

**Budget declaration.** The architecture-search phase closed at 23.632 GPU-hours
against its 10–20 cap (§10). This arm opens a **horizon phase** with its own cap of
**15 GPU-hours**, on Kaggle rather than the local card, so it does not extend an
overrun by drift.

## Operational notes

- Token stream: Kaggle dataset `sid123123234/act-lab-tinystories-tokens`
  (`tokens.npy` 40.0M tokens, `val-tokens.npy`), attached beside the code dataset.
  A 12,000-step run consumes 24.6M tokens, under one epoch of the stream.
- The two sweeps write disjoint ledgers (`flat<steps>_seed<seed>_results.jsonl`)
  and identical design files. `ranked.json`/`decision.md` in the kernel output are
  whichever sweep finished last and are **not** the record; the analysis is re-run
  locally over both seeds' ledgers after import.
- The flat plan restarts its own seed's ledgers on relaunch (no resume), so a
  deadline-stopped session keeps its completed rows in the kernel output but a
  second session would retrain that seed from the start.
- This Kaggle CLI (2.2.4) needs `KAGGLE_API_TOKEN=$(kaggle auth print-access-token)`
  for write calls once the hourly OAuth token has expired.

## How to analyse it

```
PYTHONPATH=src python scripts/kaggle_queue.py --queue archzoo      # push, wait, import
PYTHONPATH=src python scripts/arch_zoo_analyze.py \
  --experiment-id arch-zoo-100m-v3-crossed-horizon10x --write --horizon-rule
```

Then extend `verify_record.py` with the measured rows, update §3/§9/§11 of
`ARCH_SEARCH_FINDINGS.md` and the README, and run the whole suite.

## Status

**Pre-registered, not yet run** (2026-09-19). Nothing in this file may change after
the first ledger row exists except the Status section.
