# arch-zoo-100m-v2-horizon — Does the Matched Effect Survive Training?

**Design:** `matched-total`, same 8 candidates as `arch-zoo-100m-v2`, plan `flat`,
lengths **400 and 800** steps, seeds 42/43/44, context 1024, batch 2. At 1200
steps the same candidates already have 3 seeds in `arch-zoo-100m-v2`, so the three
arms together form one decay curve on one architecture set.

## Why this is the gating experiment

`arch-zoo-100m-v2` established that architecture choice matters: 0.2416 nats
spread against 0.0164 nats of seed noise, seed-matched **p = 0.00023**, at a
matched total-parameter budget.

But the phase's other measurement, from v1, says the spread **decays with
training** — 1.6× per doubling from 400→800 steps and 1.4× over 800→1200, both
monotone, across two architecture sets. That evidence comes from v1's *unmatched*
families, whose total parameters span 70.8M–123.3M while active parameters span
25.3M–77.1M. So the decay rate is currently a property of a confounded set, and it
is the number that decides how much a search is worth:

- If the spread keeps decaying, a 0.24-nat spread at 1200 steps is much smaller at
  10× the horizon, and the value of searching *structure* falls with the training
  budget — which points at scale and duration rather than at architecture.
- If it does not decay, v1's decay was an artifact of its unmatched families, and
  architecture search is worth considerably more than §3 implies.

Either answer is a result. This arm is ~3.6 GPU-hours against a 10–20 GPU-hour
cap, and it is the cheapest way to close the question.

## Design integrity

The dry run was checked before spending the hours: this arm builds the **same 8
candidates** as `arch-zoo-100m-v2`, byte-identically —

```
Design matched-total, target 78,000,000.0 params (+/-5%)
  total params      : 75,027,264 .. 81,149,760 (x1.08)
  active params     : 18,003,456 .. 81,149,760
  min zoo distance  : 0.256
```

A decay comparison across *different* architecture sets measures the set, not the
length, which is the flaw in v1's version of this measurement. The candidate list
is therefore a precondition, not a detail.

Flat at each length, so nothing is promoted on a short-run proxy (§4: the 400-step
screen picks the wrong winner by 0.009 nats, a quarter of the 0.0390-nat
per-architecture seed std on that same set, so the two runs did not separate). The
selection caveat that applies to v1 does not apply here.

## Pre-registered decision rule

Written before the numbers exist, because a decay rate invites a post-hoc fit.

| outcome | condition | reading |
|---|---|---|
| **decay confirmed** | shrink per doubling within 1.34–1.66× of v1's 1.4–1.6× band, monotone across 400→800→1200 | the effect is horizon-dependent; report every architecture effect size with its training length, and put the search on hold pending a scale/horizon arm |
| **no decay** | ratio < 1.15 per doubling somewhere in the curve | v1's decay was a property of its confounded set, not of training; the matched 0.24-nat spread is the effect size to plan around |
| **partial** | something in between, or non-monotone | report the fitted per-doubling rate with its interval; the search decision moves to the attribution question, which cannot be settled by more seeds |

Two secondary readings, both already computed by the report and both reported
whatever the primary says:

- `spread_by_length` — spread at each length over the architecture set common to
  all three, as the primary quantity.
- `rank_stability` — Spearman and winner-preserved between successive lengths. A
  spread that persists while the *ordering* scrambles means the effect is real but
  not searchable by short-run screening.

## Cost

8 candidates × 3 seeds × (400 + 800) steps = 48 runs. At v2's measured 0.31–0.51
s/step for 1200-step runs, the short runs are ~1.6 and ~3.2 minutes each, so the
arm projects to **~3.6 GPU-hours** (the driver's own guard, at its default 4,500
tok/s assumption, agrees). Running text for the phase after this: v1 3.60 + v2
~3.2 + this ~3.6 + mixer ~0.2 ≈ **10.6 GPU-hours**, inside the cap.

## How to analyse it

The horizon arm writes its own reduced `decision.md` from its two lengths. The
authoritative pooled analysis pools with v2, which is where the 1200-step seeds
live:

```
PYTHONPATH=src python scripts/arch_zoo_analyze.py \
  --experiment-id arch-zoo-100m-v2 --also arch-zoo-100m-v2-horizon \
  --signal-genes hidden_size,depth --write
```

`--signal-genes` names the model used for the by-length fit, so each length
estimates the *same* coefficients rather than selecting its own predictors. The
committed `decision.md` was generated with it. Re-running without the flag
produces the secondary selected-model table (identical reading, weaker claim),
and is safe to do — but the two tables are not interchangeable and the record
should say which one it holds.

That writes to the **primary** directory, so v2's committed `decision.md` becomes
the three-length record. Pooling is safe here because the candidate names are
identical and `with_structure` joins each ledger's structural genes on name; the
definition in the primary directory supplies them.

## Result

**The pre-registered rule fired: decay confirmed.** 48 runs, 3.25 GPU-hours (the
estimate was 3.6). Pooled with v2's 1200-step seeds, one architecture set, three
lengths, all flat:

| Steps | Spread | Seed std | Spread in seed-noise units | Best |
|---|---|---|---|---|
| 400 | 0.5457 | 0.0286 | 19.09 | `mtot_D6_H896_E16_A6_W288` |
| 800 | 0.3244 | 0.0165 | 19.71 | `mtot_D6_H896_E16_A6_W288` |
| 1200 | 0.2462 | 0.0148 | 16.59 | `mtot_D6_H896_E16_A6_W288` |

Shrink per doubling of training: **1.65x**, monotone across the curve. v1's
estimate from its unrelated, confounded families was 1.4–1.6x per doubling, and
this matched design lands inside the pre-registered 1.34–1.66x band. So v1's decay
rate was **not** an artifact of its unmatched families: the decay is a property of
training length, and it replicates on a controlled set.

It is also worth recording cost, because this arm is where the phase's
information-per-GPU-hour question gets a number: the same 0.5457-nat ordering is
visible at 400 steps for ~1.1 GPU-hours, and resolving it to 1200 steps costs 3.2.

## The rule fired, but not for the reason it anticipated

This is the part the pre-registration got wrong, and the correction is the more
useful result.

The rule only looked at the spread, and the spread does fall as predicted. But
seed noise falls *with* it — 19.09 → 19.71 → 16.59 in seed-noise units. The
between-architecture range and the within-architecture scatter are both shrinking
because a longer run estimates the same loss with less variance, so the effect is
**no more and no less detectable at 400 steps than at 1200**. A spread that decays
in lockstep with its own noise is a variance decay, not evidence that architecture
stops mattering as training lengthens.

Two secondary readings, both pre-registered as reportables, say the same thing:

- **The winner is identical at every length.** The same architecture is best at
  400, 800 and 1200 steps, on every seed.
- **Rank stability is 0.95 (400→800) and 0.98 (800→1200)**, winner preserved in
  both. Short-run screening ranks like a long run here.

So the pre-registered action — "put the search on hold pending a scale/horizon
arm" — is not what the combined evidence supports. What the combined evidence
supports is: the *magnitude* of the effect is horizon-dependent and every effect
size should be quoted with its training length, while the *ordering* and the
*signal-to-noise* are stable over a 3x horizon. The action taken instead is to
proceed with the search, quoting the 1200-step spread, because the ordering a
search consumes is the part that did not move.

This is recorded as a correction to the decision rule, not as an override of it:
a rule that thresholds a spread without normalising by the noise it is measured
against will fire on variance decay and will read it as signal decay. Reporting
the seed-noise units alongside the spread is what closes that gap, and
`spread_by_length` now carries it.

## The *explanation* survives the screen too, not just the ordering

The decay question was answered on the spread and the ranking. Both of those are
about *ordering* — a search is handed the ordering, but a research decision is
typically made on the *reason*. This arm can speak to that as well, because it
leaves the same eight candidates at 400 and 800 steps and v2 supplies 1200: fit
the structural genes jointly at each length and ask whether the same model is
readable at a third of the training.

`structural_signal_by_length` (same eight candidates at every length). The primary
table holds the model **fixed** to the two genes the crossed factorial identified,
so the same coefficients are estimated at every length and no per-length selection
is involved:

| Steps | R² | Model p | `hidden_size` beta | p | `depth` beta | p |
|---|---|---|---|---|---|---|
| 400 | 0.927 | 0.0024 | −0.863 | **0.0145** | +0.528 | 0.175 |
| 800 | 0.936 | 0.0019 | −0.944 | **0.00075** | +0.335 | 0.407 |
| 1200 | 0.953 | 0.0011 | −0.959 | **0.00035** | +0.310 | 0.444 |

The model is significant at every length, `hidden_size` is *individually*
significant at every length including the cheapest one, and its coefficient
tightens monotonically with training. **What training length changes is the size of
the effect, not the identifiability of what drives it** — the counterpart, at the
level of the explanation, of the `sigma`-ratio argument above.

Letting each length select its own predictors gives the same reading (R²
0.946 / 0.960 / 0.955, `hidden_size` leading with a negative sign at all three),
but that version cannot distinguish "the same model holds" from "a different model
holds equally well", so it is the secondary check rather than the primary one.

Two honest limits, both carried in `decision.md`:

- Even a pre-specified coefficient p is not the designed factorial's exact test in
  `arch-zoo-100m-v3-crossed`, where `hidden_size` and `depth` were crossed by
  construction. This is a consistency check; the factorial is the identification.
- At 400 steps the per-axis table tests 12 axes uncorrected, so a marginal per-axis
  p (0.038 for hidden size there) is a hypothesis. The model p-values above do not
  depend on that table, because the model was named in advance.

## Status

**Complete.** Authoritative pooled three-length record: `arch-zoo-100m-v2/decision.md`
(regenerated with `--also arch-zoo-100m-v2-horizon`). This directory's own
`decision.md` covers its two measured lengths.
