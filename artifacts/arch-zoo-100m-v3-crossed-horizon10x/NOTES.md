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
  --experiment-id arch-zoo-100m-v3-crossed-horizon10x --write --horizon-rule \
  --offset-against arch-zoo-100m-v3-crossed --offset-steps 1200
```

`--offset-against` is the fifth secondary, mechanized: this arm's fp16 anchor at 1200
steps is batch-identical to the record's fp32 rows, so the pair measures the precision
offset directly, at the one length where the two arms share a design. It is read
*beside* the record and never pooled with it - the two were measured under different
precisions, and that difference is the quantity. This is exactly what
`scripts/run_horizon_analysis_when_kernel_exits.sh` runs once all 32 rows exist, so it
is also the command that reproduces `decision.md` in this directory.

Then extend `verify_record.py` with the measured rows, update §3/§9/§11 of
`ARCH_SEARCH_FINDINGS.md` and the README, and run the whole suite.

## Status

**Pre-registered 2026-09-19; first attempt failed on a harness guard, was re-run after
correcting it, and landed the same day on `shallow-wins persists` (all 2026-09-19).**
Nothing above this section has changed since the first ledger row existed.

### First attempt: the horizon never ran

The kernel `act-t4-archzoo` ran 0.50 h and exited `COMPLETE`. The 1200-step anchor
trained; **all sixteen 12,000-step runs failed before a step**, each with

```
ValueError: .../act-lab-tinystories-tokens/tokens.npy holds 40043724 tokens, need 51153024
```

The 51.15M was the harness's admission check in `architecture_eval._prepare_streams`,
which asked for **twice** the schedule plus the 2M validation holdout. This file's own
cost section says the design is feasible - 12,000 steps consume 24.6M tokens and use
24,000 of the stream's 37,115 windows, so no window repeats, and the arm is under one
epoch as intended. The check was the thing that was wrong, not the design: it is now
`required_train_tokens`, the schedule itself (24.6M here) rather than an undocumented
2x multiple, with a test pinning that a one-epoch stream is admitted and a
single-token-short one still refused. **The design, the candidate set, the seeds, the
two lengths and the precision are unchanged by that correction** - a relaxation of an
admission test cannot alter a run that already passed it, and `verify_record.py`
reproduces all 125 recorded claims after the change.

Two reporting defects let a 0-for-16 arm read as a success, both now fixed:
`kaggle_t4.mode_archzoo` derived `complete` from `not stopped_at_deadline` and the
process exit codes, and the screen exits 0 when every candidate in a stage fails; it
now compares the requested (length, seed) grid against the ledgers. And the screen
had no data-feasibility check, so a plan that could not be served was discovered
inside the kernel rather than at launch. The screen now reads the stream's header
per length at launch and refuses before any training; `--dry-run` reports the same
check without training, so the numbers can be read without the hours.

### What the failed attempt did produce: the fp16 anchor, at 1200 steps

Both seeds' 1200-step fp16 ledgers completed (16 rows) and **are reusable as the
anchor**: the stream and holdout are the same size as the record's, so the window
count, the seeded permutation and therefore the batch schedule are identical to the
fp32 record's runs. On the same two seeds, the crossed main effects are

| quantity | fp16 (this arm, 1200) | fp32 (record, 1200) | offset |
|---|---|---|---|
| depth (D14 - D5) | +0.0742 | +0.0842 | -0.010 |
| hidden (H896 - H512) | -0.1918 | -0.1928 | +0.001 |

per-cell differences 0.001-0.018 nats. So the precision offset is small against
effects of 0.08-0.19 nats, which is the arm's own check on its choice of fp16, and it
is now measured rather than assumed.

### The re-run

The design is re-launched unchanged: the same byte-identical crossed candidate set,
seeds 42/43, flat at 1200 and 12,000 steps, fp16, one sweep per T4. The 1200-step
anchor is retrained rather than carried over, because a flat plan restarts its own
seed's ledgers on relaunch and special-casing one length inside the arm would be a
change to the design. Cost is ~4 GPU-hours at the first attempt's measured 17.2k
tokens/s, against the 15 GPU-h horizon cap; the first attempt spent 0.5.

A watcher runs alongside the driver (`scripts/run_horizon_analysis_when_kernel_exits.sh`,
detached, logging to `analysis_watcher.log`): it waits for the driver to exit, then
**refuses to write** unless all 32 rows exist (8 candidates x {1200, 12000} x seeds
42/43), and only then runs the pre-registered command above with `--write`. The
refusal is the second half of the same fix: this file's rule is declared on the
12,000-step rows while the 1200-step anchor sits in the same directory, so `HORIZON_STEPS`
now names the length instead of inferring "the longest length that has rows" - which
on the first attempt's ledgers would have resolved to 1200 and reported the anchor as
the horizon result. The rule returns `unavailable` at any other length, and the
analysis refuses outright when the horizon rows are absent.

### How the secondaries above are now read

They were listed in the pre-registration without a mechanical reader, so they were
made mechanical before the ledgers landed rather than read by eye afterwards:

- **Ordering** and **decay** are computed from this arm's own two lengths by
  `rank_stability` and `spread_by_length`, which the analyzer already reports. The
  reference is the arm's own 1200-step fp16 anchor rather than the record's fp32
  rows, which is strictly better here: the anchor's batch schedule is identical to
  the record's, so the comparison is not confounded by a data difference, and it is
  measured inside one precision.
- **Precision offset** is `precision_offset`, reached with
  `--offset-against arch-zoo-100m-v3-crossed --offset-steps 1200` and rendered into
  `decision.md` as a per-candidate table. It reads the record's ledgers *beside*
  this arm rather than pooling them - and pooling is now refused outright when two
  directories recorded different precisions, which is the guard that makes the
  distinction structural rather than a matter of remembering.
- **Width and interaction** are already carried by the crossed contrast the rule
  reports as its secondary, at whatever length the rule read.

### Landed: 4.83 h on Kaggle T4×2, and the branch the rule returns

The re-run completed `complete=True` with **32/32 rows** (8 candidates × {1200, 12000}
× seeds 42/43) — the grid the first attempt was missing entirely — in 4.83 h of wall
clock, ~9.22 GPU-hours token-derived, inside the 15 GPU-h cap. The pre-registered rule
lands on **shallow-wins persists**:

| condition | value | met |
|---|---|---|
| depth main effect significant (exact re-split p < 0.05) | p = 0.003086 (floor 0.001543) | yes |
| depth main effect > 0 (deep − shallow; shallow wins) | +0.0209 nats at 12,000 steps | yes |
| secondary: width main effect < 0 (reported, not gating) | −0.0645 nats, p = 0.001543 (at floor) | yes |

So the depth sign is horizon-stable to 10× and §2c's "spend at low depth" stands at
this scale. Against this arm's own same-precision 1,200-step anchor the spread falls
0.3085 → 0.1059 nats (2.91×, **1.38× per doubling** against §3's 1.65× — the rate
**over-predicts** the decay, which is the safe direction for a budget that reads it)
and seed std 0.0118 → 0.0056, leaving the effect at **18.88× its own seed noise** at
12,000 steps: it does not wash out. The ordering is the qualified half — Spearman 0.90
with the **winner changing** (`xlohi_D5_H896_E16_A2_W384` → `xlohi_D5_H896_E4_A4_W1536`),
which is precisely why the arm's pre-registered rule rather than the screen decides the
branch. The precision offset against the fp32 record at 1,200 steps is **−0.0030 nats**
mean (−0.0257 to +0.0104), smaller than the depth effect it carries, which is the
arm's own check on having chosen fp16. The interaction is −0.0090 nats with the
seed-blocked test at its 2-seed floor of 0.5, so the two seeds can neither confirm nor
exclude it, as the pre-registration said in advance.

**The branch's robustness was measured at this length, not inherited from 1,200.**
Re-running the identical factorial on *training* loss keeps both main effects at the
same signs and significant — depth **+0.0249** (p = 0.0077, against +0.0209 held-out)
and width **−0.0667** at the same 1296-assignment floor — so the horizon attribution is
not an artifact of where the validation tokens were cut. Dropping any one of the 8
candidates leaves the spread significant (worst blocked p = **0.0018**). The *crossed
attribution* itself cannot be checked that way and refuses rather than reporting a
subset: at 2 candidates per cell, removing one unbalances the factorial, so it returns
unavailable for all eight — a 3-replicate design would support the check, as v2's
8-distinct-configuration arm did.

**And two seeds is arithmetic rather than caution.** The re-split test is two-sided, so
the smallest p it can report is `2/assignments`: 1296 assignments on both seeds give
0.001543, but one seed has 36 and a floor of **2/36 = 0.0556** — above the 0.05 the rule
reads against. **A single seed cannot reach this branch at all**, which is why the
stale one-seed file the landing left behind had no verdict to state as well as no
verdict to state it with. (What does replicate per seed is the effect *size*: depth
+0.0211/+0.0207 and width −0.0618/−0.0673.) All six checks are recomputed from these
ledgers by `scripts/verify_record.py`.

### The landing had to be verified, because the first one was not

The driver imports the kernel's own `decision.md`/`ranked.json`/`search_cost.json`,
which are whichever sweep finished last and are **not** this arm's record. The local
re-analysis that *is* the record launched and was killed inside the 4 seconds it runs
for, so `decision.md` was left **older than every ledger it claims to describe**
(18:32:47 against 18:32:49) — the artifact-level proof that it was not derived from
them — and the file it left in place was the kernel's single-seed "PROVISIONAL" write,
under-reading a complete two-seed arm by half. Three things were closed rather than
re-run by hand:

- the post-run analysis runs **unbuffered** (`python3 -u`): a stdout redirect to a file
  is block-buffered, so a kill inside that window loses the output *and* any traceback,
  which is why the watcher log stopped mid-line with no record of the attempt;
- the watcher **no longer reports on an exit status alone** — which says the process
  finished, not that it wrote. It re-renders `decision.md` from this directory's own
  documented command and requires it back byte-identical, via the repository's own
  `scripts/check_record_freshness.py --only`. That is strictly stronger than a
  timestamp comparison (it proves the record *is* what the runbook produces over the
  ledgers on disk) and it names the cause when it fails;
- `scripts/check_record_freshness.py` itself now fails any arm whose committed record
  was read over a **seed set that no documented command reads**. That is the
  content-based form of the same check (it survives a fresh clone, where mtimes do
  not) and it names the cause before paying for a regeneration;

To reproduce this directory's record by hand:

```
PYTHONPATH=src python scripts/arch_zoo_analyze.py --write --horizon-rule \
  --experiment-id arch-zoo-100m-v3-crossed-horizon10x \
  --offset-against arch-zoo-100m-v3-crossed --offset-steps 1200
```
