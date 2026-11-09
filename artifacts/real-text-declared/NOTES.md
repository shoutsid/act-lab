# Section 5.70 — the declared-shape real-text block (registered before any seed)

## Why this block exists

The declared Gate-4 block finished on 2026-10-31 with
`PASS_COST_ONLY_NO_DEMONSTRATED_EDGE` (exit 2): the cost criterion passed and the harm
guard cleared, but adaptive won **6/12** and the loss-blind control **7/12** against their
own churn arms, so no quality edge was demonstrated and nothing downstream was authorized.
That block trained on **synthetic tokens**, and its losses sat on the uniform-vocabulary
floor where the arm difference is `~1e-4` — §10.2 says so explicitly.

The only real-text evidence is at **development fidelity**: §5.40 (`null - adaptive`
`-0.1355` nats, 12/12) and §5.66 (`-0.1262`, 12/12) both favor doing nothing, on the
deployed rebuild treatment. §10.3 names the gap that leaves: *declared-scale real-text
behavior*. Nobody has measured the adaptive-versus-random edge at the declared
285M total / 134M active shape on the real TinyStories stream. That is the one question
the free GPU week can buy, and this block buys it.

## The question

At the declared shape, on real text, does the adaptive controller beat matched random
churn on final held-out loss? This is Gate 4's withheld **quality premise**, moved from
the synthetic substrate to the real one at the same shape and token budget.

## Design

| | |
|---|---|
| **Seeds** | **15036-15047** (twelve, the next block after 15024-15035), both arms in one session per seed |
| **Arms** | `adaptive` (`probe_loss` policy, the campaign's `PRIMARY_POLICY`) and `random`; paired within seed, same initialization identity, same data schedule, matched three-event budget at steps 1500/3000/4500 |
| **Shape** | declared: 285M total / 134M active parameters, batch 2 x context 2048, 6000 steps = **24,576,000 tokens per arm**, fp16 on one T4 x2 |
| **Stream** | the published real stream `sid123123234/act-lab-tinystories-tokens` (`tokens.npy` + `val-tokens.npy`); validation loss on the held-out stream |
| **Treatment** | §5.47 rebuild pair on every arm: `preserve_router_on_rebuild=True`, `new_expert_initialization="clone"` |
| **Venue** | Kaggle GPU, three waves of four seeds (`act-realtext-a/b/c`), ~6.5 h each against the platform's 12 h session limit |
| **Primary** | paired sign test, `adaptive < random` on final loss, **10 of 12** bar; exact one-sided p `79/4096 = 0.019287` |
| **Secondary** | mean `adaptive - random` (negative favors adaptive) and the paired SD, reported beside the primary |

## Declared predictions and branches

**Prediction.** The negative replicates: **adaptive wins at most 3 of 12**. It follows from
§5.40 and §5.66 (dev-fidelity real text favored null 12/12 twice) and from the synthetic
declared block's 6/12, which is the same direction at the same shape.

| branch | reading | consequence |
|---|---|---|
| **A: adaptive wins >= 10/12** | the edge survives at declared scale on real text | Gate 4's quality premise is supported at this fidelity; the next registered step is the pool term (a declared real-text null arm), not Gate 5 |
| **B: adaptive wins <= 2/12** | the dev-fidelity negative replicates at declared scale | the real-text declared arm closes; no Gate 5, no further spend on this axis |
| **C: 3-9 wins** | not demonstrated | reported as inconclusive, same withholding as the synthetic block; no downstream authorization |

Branch C is the one the synthetic block already landed in (6/12), so a second C would be
consistent rather than surprising; it would not be evidence for the controller.

## The vocabulary mismatch, recorded before the first pair (2026-11-08)

The declared config's vocabulary is **32,768**, while the published stream's ids span
**0-8,191** (the stream was built with the 8,192-token BPE the real-LM leg trained).
`WindowSampler` accepts the stream - every id is inside the vocabulary - but the upper
24,576 embedding rows never receive a gradient. That is a capacity the data cannot use,
and it is identical for both arms, so it cannot produce the paired contrast; what it
does mean is that the absolute losses here are **not** comparable to the real-LM
report's 8,192-vocab model, and the block is the declared *shape* on real tokens rather
than the real-LM model's shape. Rebuilding the stream at 32,768 is impossible with this
tokenizer, and shrinking the model's vocabulary would change the declared parameter
count and void the registration. It is run as declared and read for the sign and the
paired delta, not for its absolute loss.

## The caveat this block carries

The committed stream is **~40M tokens** against **24.576M per arm**, so twelve seeds share
it: each seed draws its own random windows, but the seeds are not twelve independent text
samples. This is the declared **shape** on real text, not a text-generalisation claim, and
the record says so. Building twelve disjoint 24.576M-token shards from the 1.8 GB corpus
would remove the caveat and is the next block's data work if branch A fires; it was not
worth leaving this week's GPU hours idle for.

## Cost

Twelve pairs at the campaign's measured **1.63 h/pair** is **19.6 h**, plus three kernel
starts and model builds (**<= 0.75 h**): **~20.3 GPU-hours**, inside the 30 h week. Zero
TPU-hours and zero local experiment compute; the local steps are pytest, staging and the
statistics over the fetched rows.

**Why not TPU.** The port path's declared-shape container still dies on
`Authentication required to call the Kaggle API` (2026-10-05, zero seeds published), and
its registered dtype is fp32, which cannot be paired with this block's fp16 arms without
making dtype a confound. The TPU week stays unused until that defect is fixed for a study
that needs it.

## Run provenance: wave c exists as two versions (2026-11-09)

Wave c was pushed at 03:37Z when a slot freed and **again at 03:48Z** when the
watcher was restarted; the restart had no memory of the accepted version. Kaggle
accepted version 2 while version 1 was `RUNNING`, so the platform may have run the
same four seeds twice. The CLI exposes only the latest version and has no cancel, so
the record takes the fetched rows from version 2 and prices the duplicate at up to
**~6.3 GPU-hours** (worst case, taking the block to ~25-27 h of the 30 h week). The
watcher now refuses to push when a c version already exists. The duplicate cannot
change the paired contrast: same seeds, same revision, same data.

## What this does not do

It does not measure the pool term (`random - null`) at declared scale: that needs its own
twelve null seeds, a second GPU week. It does not reopen the closed width programme. It
does not settle text beyond TinyStories. It does not authorize Gate 5.

## Evidence this block will leave

- per-seed rows: `artifacts/real-text-declared/kernel/<wave>/seed-<n>.json`
  (`final_loss`, `validation_loss`, FLOPs, mutations, architecture hash, wall time);
- the fetched summary and log in the same tree;
- the reading: `artifacts/study-summaries/real-text-declared-12.json`, written only after
  all twelve pairs are fetched, with the sign test and the mean computed from the rows;
- the report section 5.70 and the status entry, added with the result.

Runner/pusher: `scripts/push_real_text_declared.py`. Registration pins:
`tests/test_real_text_declared.py`.
