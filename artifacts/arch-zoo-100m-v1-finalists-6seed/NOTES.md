# arch-zoo-100m-v1-finalists-6seed — resolving v1's `NOT ESTABLISHED`

This arm exists to close one item that `arch-zoo-100m-v1` listed against itself.

v1's verdict is **NOT ESTABLISHED**: the 3 finalists' spread is 0.0404 nats, but the
seed-matched exact blocked permutation test returns `p = 0.1389` over 3 blocks x 3
architectures. Its decision text is explicit that this is not a negative result —
"this is evidence that this experiment cannot yet tell" — and its own `Next step`
names the fix:

> re-run the finalists on seeds never used in screening, which removes the selection
> coupling and raises power

and its caveat names why the existing test is soft:

> 13 architectures were screened at shorter lengths than these 3, and promotion used
> those runs, so this test is mildly anti-conservative. Confirming on seeds never used
> in screening removes it.

Signed off in `artifacts/ARCH_SEARCH_FINDINGS.md` section 9 as an open item at
~1.3 GPU-h. This arm is that item and nothing else: **no new architectures, no new
lengths, no new hyper-parameters.**

## What it trains

The three v1 finalists at 1200 steps — byte-identical genomes to v1's stage 3, since
the zoo is regenerated from the same `--design families --candidates 13
--target-params 100000000 --tolerance 0.3 --zoo-seed 42` and then filtered by name
with `--only`:

| architecture | v1 mean val loss (seeds 42/43/44) |
|---|---|
| `narrow_experts_D8_H896_E12_A4_W256` | 2.7492 |
| `shallow_wide_D6_H896_E8_A2_W896` | 2.7724 |
| `narrow_experts_D10_H768_E12_A4_W288` | 2.7896 |

Seeds **45, 46, 47** — never used in v1's screening (42/43/44) or anywhere else in
this phase. 3 architectures x 3 fresh seeds x 1200 steps = **9 runs, 22.1M tokens**.

## Cost, and why it runs in fp32

~1.3 GPU-h at v1's measured 4,873 tok/s on the local GTX 1070.

It runs in **fp32**, not fp16, and that is a constraint rather than a preference:

- the pooled test requires one precision, and v1's rows are fp32;
- the gaps to be resolved are **0.0232 and 0.0172 nats**, while the measured fp16
  differential reaches **0.018 nats** (`ARCH_SEARCH_FINDINGS.md`, the 16 anchor rows
  of the horizon arm's first attempt). A perturbation of the same size as the effect
  would leave the measured ordering partly a numerics artefact.

Kaggle was considered and rejected on this ground alone: its ~3x throughput advantage
over the local card is an *fp16* advantage (13,930 tok/s vs 4,873), and on fp32 the
T4 is only ~1.25x the GTX 1070, which does not pay for a publish, an import and a
concurrency slot beside the running horizon kernel. The local card is idle, so this
runs in parallel with that kernel rather than competing with it.

## Pre-registered rule — `FINALIST_ORDER_RULE`

Declared before any row of this arm exists. The quantity is the seed-matched exact
blocked permutation test (`blocked_permutation_test`) over the 3 finalists at 1200
steps, read twice:

- **`p_all`** — all **6** seed blocks: v1's {42,43,44} pooled with this arm's
  {45,46,47}. Declared design floor `1/(3!)^(6-1) = 1/7776 = 1.29e-4`. The
  enumeration is `6^6 = 46,656` assignments, under the 1,000,000 cap, so the test
  stays **exact** and is not downgraded to Monte-Carlo.
- **`p_fresh`** — the 3 fresh blocks {45,46,47} **alone**. These seeds took no part in
  screening or promotion, so this test is free of the selection coupling v1's caveat
  names. Declared design floor `1/36 = 2.78e-2`.

Threshold **0.05** on both. Branches, fixed now:

| branch | conditions | reading |
|---|---|---|
| `order-established` | `p_all < 0.05` **and** `p_fresh < 0.05` **and** the same architecture ranks first in **every** block of both readings (6 of 6) | v1's 3-seed null was an underpowered design, not evidence of no ordering. The ranking is real and the anti-conservative caveat is discharged on uncoupled seeds. |
| `order-coupled` | `p_all < 0.05` **but** the blocks do not agree on which architecture comes first | the pooled significance rests on agreement among blocks that disagree about the top, so nothing has been replicated and no ranking is recommended — this is the branch that would falsify "v1 was merely underpowered". |
| `still-not-established` | `p_all >= 0.05` with the 6-block floor below threshold | at 6 seeds the finalists remain inseparable. This *upgrades* v1's "cannot yet tell" to "can now tell, and there is no detectable ordering at this design's power". |
| `floor-limited` | the 6-block floor `>= 0.05` | no branch is reachable however the contrasts fall; more seeds decide it, more analysis cannot. |
| `unavailable` | the pooled 1200-step design is not complete (3 architectures on 6 shared seeds) | the rule refuses rather than labelling a partial arm. |

`order-established` is deliberately gated on the fresh blocks as well as the pool. A
pooled test alone still contains the selection it is meant to check — the finalists
were *chosen* because they looked good on seeds 42/43/44, and that favours the winner
there most, so pooling raises power at the cost of re-admitting a mild bias. Requiring
the uncoupled blocks to agree is what makes the branch mean "real" rather than "real
among the seeds that picked it".

**Amendment, before any row of this arm exists.** The branch conditions above were
first drafted as "the fresh blocks' first-ranked architecture is the 6-block
first-ranked architecture". Planting synthetic designs to rehearse every branch
(`tests/test_finalist_order_rule.py`) showed that reading to be unsound, not merely
loose: `blocked_permutation_test` is an *omnibus* test, so its statistic rises with
any consistent column separation, and a design whose three screening blocks rank A
first while its three fresh blocks rank B first is significant on the pool while
having replicated nothing — it passed the drafted condition. Every branch condition
was therefore tightened before training started, in the stricter direction, and the
six-block agreement requirement is what closes it. No threshold moved, and no row of
this arm had been written when the amendment was made.

**Pre-declared expectation, falsifiable and not gating.** v1's per-seed ordering is
**2-of-3 consistent** — seeds 42 and 43 give `W256 < W896 < W288` exactly, seed 44
swaps the top two — and the gaps to preserve are 0.0232 (1st vs 2nd) and 0.0172 (2nd vs
3rd). The test resolves the ordering only if the fresh blocks reproduce that sign at
both gaps. Nothing in this document is adjusted after the rows land.

## Status

Pre-registered; sweep running on the local GTX 1070 (idle, so it runs beside the
Kaggle horizon arm rather than competing with it). Nothing here re-tunes a protocol
knob, and this arm does not touch any recorded number of v1 — it is read *alongside*
v1 by `arch_zoo_analyze.py --also`, and only after its 9 rows exist.

The run is watched by `scripts/run_finalist_analysis_when_sweep_exits.sh`, which
waits for the sweep and refuses to analyse unless all three fresh-seed ledgers hold
3 rows *and* v1's three screening ledgers are present — a partial arm is detectable
because the sweep writes a ledger row per completed run, which is exactly how the
horizon arm's first attempt was found to be 0-for-16 while reporting success. The
watcher logs the full pre-registered analysis rather than writing it, because
`--write` supersedes v1's committed `decision.md` and that is a deliberate step, not
an unattended one. Verdict lands in `analysis_watcher.log`.

Runbook, when the sweep exits:

```bash
PYTHONPATH=src python scripts/arch_zoo_analyze.py \
  --experiment-id arch-zoo-100m-v1 \
  --also arch-zoo-100m-v1-finalists-6seed \
  --finalist-order-rule
```

Add `--write` to supersede v1's own `decision.md` once the branch above has been
read; `verdict()` then drops its stale instruction to run exactly this arm.
