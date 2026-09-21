# Architecture Search Phase — Findings

Answers the question the phase was set: **is there an architectural signal worth
searching for at ~100M?**

**Answer: yes at a matched parameter budget, and the lever is residual-stream
width.** On v1's unmatched families at 3 seeds the finalist contrast was 0.0404
nats and *not* significant (p = 0.139, power floor 0.028). On v2's matched-total
design — total parameters held to ×1.08 while active parameters spread ×4.51 — the
spread is **0.2462 nats** at 3 seeds and the seed-matched test gives
**p <= 5e-06**. A crossed factorial then attributes it: hidden size is worth
**−0.1836 nats (~274 training steps)** and depth **+0.0783 nats (~117), with
shallow beating deep**, both at the exact enumeration's floor p = 4.3e-05 — i.e.
the smallest p the 46,656-assignment test can report, so the nats are the quantity
that carries the claim and the p says only that the design could not have
distinguished a larger effect (§2c).

Three things that were open have been measured, not deferred:

| question | answer | where |
|---|---|---|
| Does the effect survive longer training? | The spread decays (1.65× per doubling, against v1's pairwise 1.59×/1.44×) but **noise decays with it** — ordering and detectability are horizon-stable; the winner is identical at 400, 800 and 1200 steps | §3 |
| Which gene carries it? | **Residual-stream width, not head count** — head count measures −0.018 nats pooled (p = 0.14, signs 8/4) against a width effect of −0.1562 nats on the tightest matching in the phase | §2c |
| Is it one architecture or the set? | Every leave-one-out of the 8 stays significant; the effect is stronger on training loss than held-out | §2a |

The attribution then **replicated on seeds never used to design or screen
anything**, and the interaction resolved with it — the fourth question, and the
one the arm was bought for:

| pooled, 6 seeds (48 runs) | value |
|---|---|
| `hidden_size` | **−0.1843 nats** (p = 5e-06) |
| `depth` | **+0.0824 nats** (p = 5e-06) |
| interaction | **+0.0827 nats**, seed-blocked p = **0.031** — its floor at 6 seeds — every one of the six seed contrasts agreeing |
| spread | 0.3018 nats, 15.3× the seed noise |
| branch | **replicates, interaction resolves** (§2c) |

The head axis then resolved the same way, on the same arithmetic: the head-paired
arm broke the width/head confound *in width's favour* (head count −0.0177 nats
pooled at p = 0.14; width −0.1562 at the factorial's floor) and left a lead its
three seeds could not decide. Its confirmation arm (`arch-zoo-100m-v4-head-paired-confirm`,
fresh seeds 51–53, byte-identical candidates) pooled to six seeds and landed on the
pre-registered **head effect is real at wide width** branch, all five conditions met
mechanically by `HEAD_PAIRING_RULE`:

| pooled, 6 seeds (48 runs) | value |
|---|---|
| `hidden_size` | **−0.1617 nats** (Monte-Carlo p = 5e-06, at resolution) |
| head effect at H=512 | **+0.0084 nats**, p = 0.37 (12 units, 5/7 signs) — nothing |
| head effect at H=896 | **−0.0553 nats**, p = 4.9e-04 **at the 2/2¹² floor**, 12/12 pairs favour more heads |
| interaction | **+0.0636 nats**, seed-blocked p = **0.031** — its six-seed floor — all six seed contrasts positive |
| pooled head contrast | −0.0234 nats, p = 0.012, signs 17/7 — an average of two disagreeing levels, not a uniform effect |

So "add heads" is a rule with a precondition: at a wide residual stream it is worth
~0.055 nats for zero parameters, at a narrow one nothing. A search over this family
must mutate `attention_heads` **jointly with `hidden_size`**, as it must mutate
`depth` jointly with it.

What remains genuinely unresolved:

- **The *shape* of the decay past 12,000 steps, not whether the ordering holds.** §3
  fits its rate over 400→1200 steps, and the arm built to test a longer length — the
  crossed factorial at 12,000 steps, `arch-zoo-100m-v3-crossed-horizon10x`, on Kaggle
  T4×2 under its own pre-registered 15 GPU-hour cap — **has run and landed**: the
  depth finding (shallow beats deep) survives 10× the horizon at **+0.0209 nats**,
  exact p = 0.003086 against a floor of 0.001543, which is the reverse of what prior
  work predicted. Two things it did not settle are stated in §3 beside it: the
  rate measured over 1200→12,000 is **1.38× per doubling** against the 1.65× fitted
  over 400→1200, so two horizons fix a rate and not a curve; and the **winner changes**
  over 10× even though the ranking holds at Spearman 0.90, so a screen promotes a
  subset rather than choosing.
- **Expert width as shock absorber**, so "wider residual stream" and "less expert
  capacity" are two readings of one trade, and expert count and width are one knob
  (`E x W` enters the parameter count as a product).

All 220 runs and their records are reproducible from the ledgers in `artifacts/`
without retraining.

---

## 1. The prior verdict was not supported by its evidence

The inherited conclusion — "architecture effect present, proceed to
evolutionary search" — rested on a harness with five independent defects, any one
of which invalidates it. Full audit in `arch-zoo/SCREENING_AUDIT.md`.

| # | Defect | Consequence |
|---|---|---|
| 1 | `evaluate_architecture` accepted `seed=` but never called `torch.manual_seed` | the recorded seed never reached the init; §20's seed-matched control did not exist |
| 2 | Headline table was uniform random tokens at context 512 | values 9.0496–9.0873 straddle the entropy floor `ln(8192) = 9.0109`; it ranked initialisation luck |
| 3 | `flops_per_token = 2.0 * total_params` | identical for a 44.1M-active and a 53.5M-active model; attention cost absent, so context never entered |
| 4 | No validation loss; `val-tokens.npy` holds 7,190 tokens | ~7 windows at context 1024 |
| 5 | Results written only at the end | the real-data sweep left a zoo definition and no results |

## 2. The corrected measurement

`arch-zoo-100m-v1`: 13 candidates, 8 structural families, 47 runs, 3.60 GPU-hours.

Ranked at 1200 steps × 3 seeds (mean held-out validation loss):

| rank | architecture | total | active | val loss |
|---|---|---|---|---|
| 1 | `narrow_experts_D8_H896_E12_A4_W256` | 77.2M | 47.8M | 2.7492 |
| 2 | `shallow_wide_D6_H896_E8_A2_W896` | 103.7M | 45.9M | 2.7724 |
| 3 | `narrow_experts_D10_H768_E12_A4_W288` | 83.1M | 47.7M | 2.7896 |

Spread 0.0404 nats, mean within-architecture seed std 0.0179.

**Tests, all reported because they disagree:**

| test | p | note |
|---|---|---|
| exact seed-matched permutation, **values** (primary) | **0.139** | the design §20 asked for; 216 assignments enumerated exactly |
| same, on within-block ranks (robustness) | 0.194 | rank consistency alone is weak here — one architecture wins by a hair |
| unpaired one-way ANOVA | 0.135 | throws away the blocking, for contrast |
| paired t, best vs worst, seed by seed | 0.093 | reference only; 3 seeds cannot support its normality assumption |

**Effect size:** 0.0404 nats ≈ **68 training steps**, 5.6% of the 1200-step
budget, fitted on the observed 800→1200 slope.

**Verdict: NOT ESTABLISHED.** Not evidence of absence — with 3 architectures on
3 seeds the *smallest attainable* p is 1/36 = 0.028, so this design cannot
resolve a 2-of-3-consistent ordering however it falls. That floor is a property
of the design, not the world, and it is the reason the recommendation is "run
more seeds", not "architecture does not matter".

## 2a. The matched-budget result: the effect is real

`arch-zoo-100m-v2` holds total parameters to **×1.08** (75.0M–81.1M) while
deliberately spreading active parameters **×4.51** (18.0M–81.1M), depth 5–14,
experts 4–16, expert width 128–1536. It runs flat — no promotion — so nothing is
selected on a short-run proxy, and every candidate is a cell of the same
15,224-cell grid the random baseline samples.

24 runs (8 architectures × seeds 42/43/44), 1200 steps, ~3.2 GPU-hours:

| rank | architecture | total | active | val loss |
|---|---|---|---|---|
| 1 | `mtot_D6_H896_E16_A6_W288` | 76.3M | 45.3M | 2.7446 |
| 2 | `mtot_D8_H832_E6_A2_W576` | 75.0M | 44.4M | 2.7859 |
| 3 | `mtot_D14_H896_E8_A5_W128` | 78.1M | 68.5M | 2.7932 |
| … | | | | |
| 8 | `mtot_D10_H512_E8_A6_W768` | 77.6M | 61.9M | 2.9908 |

| test | value |
|---|---|
| spread | 0.2462 nats = **16.6×** the within-architecture seed std (0.0148) |
| seed-matched permutation (3 blocks × 8 architectures) | **p <= 5e-06**, the Monte-Carlo procedure's resolution (design floor 6.2e-10) |
| unpaired one-way ANOVA | F = 83.2 on (7, 16) df |

(At the intermediate 2-seed stage this read 0.2416 nats and p = 0.00023; the third
seed moved neither the verdict nor the winner.)

**Verdict, unlike v1's: ESTABLISHED.** The same statistic that returned p = 0.139
on v1's 3-candidate finalist set returns p <= 5e-06 here, and the difference is
the design rather than the training: v2's power floor is 1/8!³ rather than 1/3!²,
because there are eight architectures to order instead of three and three blocks
to order them within.

**The active-parameters hypothesis is refuted.** Active parameters span ×4.51 at
a matched total budget and show no association with loss (ρ = +0.214, p = 0.62).
The fully **dense** design places 7th of 8, and the **sparsest** (23% active)
beats it by 0.052 nats. Spending every parameter on every token is not better
than spending a quarter of them.

**What the effect is carried by is identified by the next arm, not this one.**
Here the strongest association is hidden size (ρ = −0.861, p = 0.013), but
(i) 12 axes are tested uncorrected and 0.013 × 12 does not survive, (ii) hidden
size and attention heads are perfectly collinear in this design
(`heads = hidden / 64`), and (iii) cross-correlating the design's own axes shows
hidden size is nearly independent of active parameters (+0.11) but strongly
entangled with expert width (−0.74) — holding total parameters fixed while
widening the residual stream forces experts narrower, so "wide hidden" and "narrow
experts" are one finding seen from both ends. `decision.md` prints this confound
next to the axis table, and §2c resolves it with a design that can separate them.

## 2b. What a matched budget can and cannot identify

v2 establishes *that* architecture matters but not *which* gene carries it, and
that is a property of the search space rather than a shortage of seeds. Working it
out is worth a section, because every future matched design inherits the same
constraint.

**At a fixed total-parameter budget the free genes are not independent.** Total
parameters are

```
embedding + depth x (4 x hidden^2 + 2 x experts x width x hidden)
```

so with the budget pinned, raising hidden size must be paid for by depth, expert
count, or expert width. Any matched design is therefore a *slice* through a
constraint surface, and each slice confounds whichever genes it does not name.
v2 slices on hidden-vs-width and thereby confounds depth, expert count and width:
its hidden size correlates with expert width at rho = -0.74.

**Two genes are not even separable in principle.** Within a fixed
``(hidden, depth)``, total parameters depend on `experts x width` as a product, so
expert *count* and expert *width* are one knob — total expert capacity — and no
number of runs can attribute an effect to one rather than the other. Reporting
them as two axes, as v2's table does, invites an attribution that cannot exist.

**Crossing `hidden x width` fails, and the failure is checkable.** The obvious
fix is a 2x2 factorial on the two genes that v2 confounded. Enumerating the four
corners of `hidden in {512, 896} x width in {256, 1536}` at 78M shows each corner
is reachable at exactly **one depth** — 14, 10, 8 and 5 respectively:

| corner | cells in band | depth |
|---|---|---|
| H=512, W=256 | 5 | 14 only |
| H=512, W=1536 | 3 | 10 only |
| H=896, W=256 | 18 | 8 and 10 |
| H=896, W=1536 | 3 | 5 only |

So that crossing trades a hidden-width confound for a perfect depth confound. It
would have looked like a cleaner design than v2 while being strictly less
informative — the cost of the constraint is paid either way, and the only question
is which gene you choose to pay it with.

**Crossing `depth x hidden` works, because expert width can absorb without limit.**
At a fixed depth, hidden size spans 512-896 at *every* depth from 5 to 14, with
expert width absorbing the difference (at D=8, H=512 needs `E x W` of 6x1024 while
H=896 needs 4x704). `arch-zoo-100m-v3-crossed` uses this:

| property | v2 | v3 crossed |
|---|---|---|
| rho(depth, hidden size) | uncontrolled (corner cells force depth) | **0.000**, balanced 2x2 |
| rho(hidden size, expert width) | -0.74 | **-0.22** |
| total parameters matched | x1.08 | x1.05 |
| active parameters spread | x4.51 | x3.69 |

That design can separate depth from residual-stream width, which v2 could not,
and it reports its own balance at launch (`rho = +0.000 (balanced)`) so the claim
is checked rather than asserted. It still cannot separate expert width from expert
count, which is the irreducibility above.

The practical rule for the search that follows: **name the genes you intend to
compare, and verify the crossing before spending the GPU hours.** The check costs
seconds of enumeration and one dry run; discovering it afterwards costs the arm.

## 2c. The attribution, measured: residual-stream width, not head count

`arch-zoo-100m-v3-crossed` ran the `depth x hidden_size` design above — 24 runs,
3.28 GPU-hours, flat at 1200 steps, seeds 42/43/44. It landed on the pre-registered
fourth outcome, the one the design's own notes called "the most informative":
both levers are real and they are separable.

Cell means (held-out validation loss, 6 runs each):

| depth \ hidden | 512 | 896 |
|---|---|---|
| **D5** | 2.9096 | 2.7611 |
| **D14** | 3.0229 | 2.8043 |

| gene | main effect | p | equivalent training steps |
|---|---|---|---|
| `hidden_size` (896 − 512) | **−0.1836 nats** | 4.29e-05 (exact, 46,656 assignments) | ~274 (22.8% of budget) |
| `depth` (D14 − D5) | **+0.0783 nats** | 4.29e-05 (exact) | ~117 (9.7% of budget) |
| interaction | +0.0701 nats | 0.24 at 3 seeds — **resolved to +0.0827 at 6** (§2c) | ~104 (8.7%) |

`crossed_contrast` computes this: each main effect is an average over both levels
of the other gene, tested against the design's own blocking (the factor's labels
are exchangeable within each (other-level, seed) block, so seeds still block).

**Both main effects sit on the enumeration's floor**, and that has to be read
carefully. The exact test re-splits each block, so it has 46,656 assignments and
cannot report a two-sided p smaller than 2/46,656 = 4.29e-05 — which is exactly
what both genes report: the observed re-splitting was matched by nothing but its
own mirror. The p-value therefore says *this separation is the largest the design
can express*, not how significant the effect would be at a larger size. The floor
shrinks as blocks are added — another seed multiplies the assignment count — so
the limit here is this arm's size, not the test.

**The effect size is the interpretable quantity**, and it is not floor-bound:
−0.1836 and +0.0783 nats, and the step-equivalents in the table. Every claim in
this section rests on those, not on the p-values.

**What this settles, and it is the phase's second headline:** v2's confound is
resolved rather than merely acknowledged. Wide residual stream is worth
**0.184 nats, ~274 training steps**, and it is not a depth artifact.

The attribution passes the same robustness checks v2's does, where they apply.
**Not a holdout-split artifact:** re-running the identical factorial on *training*
loss keeps both main effects at the same signs and the same floor (depth +0.0827,
hidden_size −0.1807). **Not one candidate's doing:** this arm cannot be checked by
leave-one-out, and that is worth stating plainly rather than leaving unmentioned —
with exactly 2 candidates per cell, removing one unbalances the factorial, so the
contrast refuses to report instead of computing over an unbalanced subset. A
3-replicate design would support the check; v2's 8-distinct-configuration arm did.
Both behaviours are pinned by tests: the split check as a claim, the refusal as a
guard against the crash it used to be.

*With one confound of its own, recorded below and since measured rather than left
open:* this model family fixes ``heads = hidden_size / 64``, so "wider residual
stream" and "more attention heads" are the same knob in this design and the
attribution cannot separate them. Since head count is **parameter-free** at fixed
hidden size, the two readings have opposite consequences for where to spend a
search budget — so the confound was worth an arm of its own, and
`arch-zoo-100m-v4-head-paired` (the subsection at the end of §2c) varied head count
at a fixed hidden size with both cost axes identical inside a pair. It puts the head
contrast at **−0.0177 nats (p = 0.14)** and keeps this row's width effect at
**−0.1562 nats at the same exact floor**. Read the `hidden_size` row below as
**residual-stream width**.

**Depth is real but smaller, and its sign is the unintuitive part: shallow wins.**
D5 beats D14 at *both* hidden levels with total parameters matched to ±5% and
hidden size pinned. A search assuming "deeper is better" would spend budget in the
wrong direction.

**v2's refutation of active parameters survives a much wider spread.** Active
parameters span 21.3M–78.5M (×3.69) inside a design whose best architecture is
30.4M active while its *dense* 78.5M counterpart places second. The axes that do
not track loss are active parameters and dense-runtime FLOPs.

**The attribution replicated, and the interaction with it.** The depth penalty was a
2.6× per-level gap at three seeds (+0.1133 at H=512 against +0.0433 at H=896) that
the tests could only call a lead: p = 0.24 conservatively, and the better-calibrated
seed-blocked test was *floored* at 2/2³ = 0.25, so it could not have reported the
interaction however consistent the data were.

`arch-zoo-100m-v3-crossed-confirm` bought three fresh seeds (45/46/47) on a
byte-identical candidate set — the replication, on seeds never used to design or
screen anything — and pooling them gives 6 seeds and 48 runs:

| pooled 6 seeds | main effect | p |
|---|---|---|
| `hidden_size` (896 − 512) | **−0.1843 nats** | 5e-06 (Monte-Carlo, 200k draws) |
| `depth` (D14 − D5) | **+0.0824 nats** | 5e-06 |
| interaction | **+0.0827 nats** | seed-blocked **0.031**; re-split 0.015 |

Against the original arm's −0.1836 / +0.0783, the magnitudes survive fresh seeds
to within 0.005 nats, the signs are unchanged, `|hidden| = 2.24 × |depth|` clears
the pre-registered 1.5× ratio, and the ranking is unchanged too (the same
`xlohi_D5_H896_E16_A2_W384` wins both arms). On its own the fresh-seed arm is
floor-limited exactly as pre-registered — three seeds cannot resolve an interaction
— and every one of its contrasts agrees in sign (+0.0970, +0.0691, +0.1200).

**The pooled interaction resolves and is level-dependent**: +0.0827 nats, and at six
seeds the seed-blocked floor falls to 2/2⁶ = 0.031, below the pre-registered 0.05,
so a consistent sign across all six contrasts (+0.0819, +0.0840, +0.0443, +0.0970,
+0.0691, +0.1200) now *decides* it rather than pointing at it. The depth penalty at
H=512 is +0.1238 nats against +0.0410 at H=896 — a 3.0× gap, wider than the 2.6×
the three-seed arm showed. The conservative re-split test agrees (p = 0.0149),
which it did not before.

**What that changes is the *shape of the search*, not the ranking of the levers.**
A mutation that is good at one width is not good at the other, so the space is not
separable in `depth × hidden_size` and a search must cover their joint space —
mutating them together — rather than optimising one and then the other. The
separable case would have been the cheaper search, and this arm is what rules it
out.

The reading was **mechanical rather than by eye**: `replication_decision` applies
that arm's pre-registered table (thresholds in `REPLICATION_RULE`, transcribed
before its fresh seeds were read) and states which of the four branches a pooled
outcome lands in, condition by condition. It returns **replicates, interaction
resolves** on the pooled set, and returned the *inconclusive* branch on the
three-seed arm and on any five-seed subset — the verdict the arm's own arithmetic
predicts, since five seeds floor the test at 2/2⁵ = 0.0625, above the declared
0.05, and six at 0.031.

**That branch is also reading-dependent, and the condition that moves is the magnitude
ratio rather than any test.** Re-read on the other losses the same ledgers carry, held-out
loss and final training loss both land on *replicates, interaction resolves*, and the
trajectory average lands on **attribution fails to replicate**. The mechanism is worth
naming, because the failure branch's wording covers three different things: on
`mean_train_loss` both main effects stay significant with the same signs at the design's
floor (−0.2255 width, +0.1814 depth), the seed-blocked interaction stays at its 2/2⁶
floor with every seed contrast agreeing — and only `|hidden| ≥ 1.5 × |depth|` stops
holding, at 0.2255 against 0.1814, a ratio of 1.24 against the declared 1.5. So nothing
about the *attribution* moved; what moved is how large depth's penalty looks relative to
width's when the loss is averaged over the trajectory rather than read at the end.

That matters because the ratio is a statement about end-of-training magnitudes, and the
trajectory average weights the early phase — where depth's penalty is evidently larger —
more heavily (depth +0.0824 → +0.1814, a factor of 2.20, against width's 1.22). It is
recorded as a sensitivity rather than a second answer, and it is a caution about reading
magnitude ratios off a loss the ratio was not declared on, not about the width finding:
the arm's `decision.md` states it beside the branch, with the condition named.

**Still not separable, and the reason is the budget rather than the search:**
expert width is the shock absorber for any hidden-size change at fixed depth, so
"wider residual stream" and "less expert capacity" remain two readings of one
trade. `hidden x width` cannot be crossed either — §2b shows every corner exists
at exactly one depth. Expert count and expert width are one knob (`E x W` is the
product that enters the parameter count) and no design at this scale separates
them.

### A confound in the attribution itself: head count rides along with hidden size

There is one more limitation, and unlike the others it applies to the *crossed*
design rather than being escaped by it.

The model family fixes attention geometry as ``heads = hidden_size / 64``
(``_heads_for``), so **hidden size and attention-head count are the same knob** in
every design this phase has run — including the crossed factorial that produced
the attribution. Measured on the crossed arm's own ledgers:

| hidden_size | attention_heads | runs |
|---|---|---|
| 512 | 8 | 12 |
| 896 | 14 | 12 |

One head count per level, deterministically. So "−0.1836 nats for hidden size"
is equally "−0.1836 nats for going from 8 to 14 attention heads", and nothing in
this phase distinguishes them.

That matters because of a fact about the parameter count, which is checkable in
one line and is checked here:

| hidden_size | heads | total params |
|---|---|---|
| 896 | 4 | 91,822,976 |
| 896 | 8 | 91,822,976 |
| 896 | 14 | 91,822,976 |
| 896 | 16 | 91,822,976 |
| 896 | 28 | 91,822,976 |

**Head count is free.** Attention parameters are ``4 * hidden^2`` however that
width is divided, so at fixed hidden size the number of heads costs nothing at
all. The two readings of the attribution therefore have very different
consequences:

- If the lever is **residual-stream width**, budget must be spent on it — the
  phase's stated recommendation.
- If the lever is **head count**, it is bought at *zero* parameter cost, and the
  recommendation is wrong in a way that leaves free performance on the table.

This is testable and the machinery mostly exists: `make_moe_genome` already takes
an explicit ``heads`` override, so only the *design* is missing. Crossing
``hidden_size x attention_heads`` is the clean way to do it — at fixed hidden size
the two head levels are **exactly** matched in both total and active parameters,
which is a stronger matching guarantee than any other cross this project has run.

### The confound, measured: the lever is width, not head count

`arch-zoo-100m-v4-head-paired` ran that crossing — 24 runs, 3.499 GPU-hours, seeds
48/49/50, flat at 1200 steps. Depth is fixed at 10, `hidden_size` 512 × 896 is
crossed with `attention_heads` 8 × 16 over four base configurations, and because
head count is parameter-free the two members of each pair are **identical in total
parameters, active parameters, contracted FLOPs and dense-runtime FLOPs** — matched
exactly rather than landed in the 5–10% band every earlier design settled for
(×1.019 across pairs, against v3's ×1.05).

| view | quantity | value |
|---|---|---|
| paired, 12 units | head effect, 16 − 8 heads | **−0.0177 nats** |
| paired, 12 units | exact sign-permutation p | 0.1396 (floor 4.9e-04) |
| paired, 12 units | sign consistency | 8 negative, 4 positive |
| factorial | `hidden_size` | **−0.1562 nats**, p = 4.287e-05, **at the 46,656-assignment floor** |
| factorial | `attention_heads` | −0.0177 nats, p = 0.1217 |
| factorial | interaction | +0.0571 nats |

**This lands on the arm's pre-registered "width is the lever" branch**, decided by
the two numeric conditions it declared before its runs existed: the head contrast is
non-significant with |effect| = 0.018 < 0.05, while the width contrast holds. So §2c
stands as written and the confound is resolved *in its favour*: the phase's
recommendation — spend budget on the residual stream — is unchanged, and "add heads,
which costs nothing" is not an alternative to it.

**The head clause of that branch is the one place in the phase where the holdout-split
check moves a decision, so it is worth stating rather than smoothing over.** The width
clause is solid on every loss reading (−0.1562 held-out, −0.1555 on final training loss,
−0.1846 on mean training loss, all at the design's floor). The head clause is not: on
held-out loss the head contrast is non-significant (paired p = 0.140, factorial
p = 0.122) and the branch is taken, but read instead on **final training loss** the same
contrast is **significant at 0.044** (paired 0.04395, factorial 0.04381) and the
branch's non-significance condition no longer holds. Read on the trajectory average
(`mean_train_loss`, which moves with how fast an architecture descends rather than with
how well it fits) it is non-significant again at 0.630.

That dependence is informative rather than fatal. An effect that is present on training
loss and absent on held-out loss is an effect that did not generalise — which is the
reading that leaves the recommendation intact, so this arm's own numbers argue *for*
not privileging head count. What it does rule out is quoting the head contrast as a null:
the honest sentence is that it is **small (~0.02 nats, a third of the width effect at
most) and reading-dependent in significance**, which is why what the search program
spends on is the level split below and not this contrast's p-value.

Two things the arm adds to the width finding rather than merely repeating it:

- **The width effect replicates on a different design at tighter matching.** −0.1562
  nats here against the crossed arm's −0.1836, with depth held constant this time and
  both factors balanced 4 × 4 — and it is negative at *both* head counts (−0.1277 at
  8 heads, −0.1847 at 16), so it is not a head-count artifact either. That it comes
  in 15% smaller under tighter matching is the direction the confound would predict
  if part of the crossed arm's number had been head count.
- **Head count is not nothing; it is level-dependent, and that is a lead rather than
  a result.** The head effect is **+0.0109 nats at hidden 512** and **−0.0462 at
  896** — an interaction of +0.0571 nats with all three seed contrasts agreeing in
  sign (+0.0383, +0.0730, +0.0598) — and at H=896 every one of the six within-pair
  differences favours more heads, which is the most extreme six-unit outcome there is
  (2/2⁶ = 0.031). None of that is decidable from this arm: the seed-blocked
  interaction test is floored at **2/2³ = 0.25** on three seeds, so no analysis of
  these ledgers can report the interaction however consistent its contrasts are, and
  the level split was chosen after seeing the interaction rather than pre-registered.
  This is the same *shape* of claim the depth penalty was at three seeds, and it needs
  the same remedy — another seed block, ~3.5 GPU-hours — which the phase's budget no
  longer had. It was recorded as the next arm's question rather than as a finding,
  and that arm was bought anyway with its overrun declared first.

### The lead, decided: the head effect is real at wide width and absent at narrow

`arch-zoo-100m-v4-head-paired-confirm` — 24 runs, 3.511 GPU-hours, fresh seeds 51/52/53
(never used to design, screen or score anything in the phase), candidate set verified
byte-identical to the parent's before launch and asserted by `verify_record.py` since.
Pooling it with the parent gives six seeds and 48 runs, and the branch is read by
`HEAD_PAIRING_RULE` (`--head-rule`), whose five conditions were all exercised on
planted synthetic designs before the hours were spent:

| pooled, 6 seeds | quantity | value |
|---|---|---|
| factorial | `hidden_size` | **−0.1617 nats**, Monte-Carlo p = 5e-06 (at the 200,000-draw resolution; the 2.2e9-assignment space is past enumeration) |
| factorial | `attention_heads` | −0.0234 nats, p = 0.015 |
| factorial | interaction | **+0.0636 nats**, seed-blocked p = **0.03125 = 2/2⁶, at floor**, per-seed +0.0383 / +0.0730 / +0.0598 / +0.0754 / +0.0399 / +0.0953 |
| paired, 12 units | head effect at **H=512** | **+0.0084 nats**, exact p = 0.373, signs 5 negative / 7 positive |
| paired, 12 units | head effect at **H=896** | **−0.0553 nats**, exact p = 4.88e-04 **at the 2/2¹² floor**, signs **12 negative / 0 positive** |
| paired, 24 units | pooled head contrast | −0.0234 nats, Monte-Carlo p = 0.0115, signs 17/7 |

**Branch: HEAD EFFECT IS REAL AT WIDE WIDTH** — width significant and negative,
interaction below 0.05 with every seed contrast agreeing, interaction positive, and
the wide level's paired head effect negative. The reading the rule prints beside it
is the one the rehearsal was built to force: the pooled contrast clears its
threshold on a *sign-inconsistent* set of pairs, so it is a systematic average of
two disagreeing levels — not a uniform head effect — and only the level split says
which level produces it.

What it changes: the recommendation gains a clause rather than reversing. Width is
still the lever (−0.1617 here, third replication at the tightest matching in the
phase, ×1.019 across pairs). Head count is a *free* ~0.055 nats — about a third of
the width effect — but only at the wide level, so it is not a global "more heads"
rule and a search must mutate `attention_heads` jointly with `hidden_size`. That is
now the second joint constraint on the search space (with `depth × hidden_size`,
above), and both were found the same way: a three-seed arm whose contrasts all
agreed but whose calibrated test was floored above the declared threshold, decided
by the seeds that lowered the floor and by nothing else.

**The branch survives the loss readings; the pooled contrast it is *read* against is
thin, and one reading reverses the branch.** Held-out loss and final training loss are
the like-for-like pair — both are loss at the end of training — and the rule lands on
the same branch on both, five conditions of five, with width at floor in each read. The
fragile quantity is the pooled head contrast: **p = 0.0115 against the rule's own
declared 0.01**, a margin of 1.15× or about six steps of that contrast's 1/4096 sampling
resolution, and the arm's two halves straddle the looser 0.05 that a reader would
naturally apply from opposite sides — the parent is non-significant held-out and
significant on final train loss, and its confirmation is the reverse. On a third reading
the branch changes: `mean_train_loss` averages over the whole trajectory, and on it one
of the six per-seed interaction contrasts is −0.0050 (the others +0.0306 to +0.0652),
the seed-blocked interaction leaves its 2/2⁶ floor for 0.0625, and the rule reports
**INTERACTION ABSENT — THE THREE-SEED AGREEMENT WAS SEED LUCK** instead.

What does *not* move is what the search program spends on. Under all three readings the
width contrast is negative and at the design's floor (−0.1617 / −0.1549 / −0.1865), and
the H=896 level effect is negative with **all 12 units favouring more heads** at the
exact 2/2¹² floor (−0.0553 / −0.0582 / −0.0344), while H=512 straddles zero and stays
non-significant. So the pricing rests on the level split, which replicates on every
reading, rather than on the pooled contrast, which does not — and the arm's headline
should be read with that margin named instead of as a null. Every number in this
paragraph is recomputed from the two arms' ledgers by `verify_record.py`.

The arm's own `decision.md` prints this rather than relying on a reader of the
findings: the branch is a threshold on a loss, so the artifact re-reads the rule over
the other losses those ledgers carry — same rows, only the loss field substituted —
and prints the branch each one lands on, with the disagreeing reading in bold. Running
it costs ~3.6 s against the 200,000-draw analysis it belongs to (6.2 s), and it is not
gated on the margin looking thin, because a silent skip would be indistinguishable from
a branch that held. A rule applied to an outcome without its rows reports no stability
claim at all rather than an unverified one.

## 3. The effect decays with training — but the decay is variance, not signal

**The v1 estimate, from overlapping subsets of its confounded families:**

The architecture spread shrank monotonically as training lengthened, on two
**overlapping** subsets of v1's families:

| comparison | n | spread at short → long | shrink | Spearman |
|---|---|---|---|---|
| 400 → 800 steps | 6 | 0.1796 → 0.1126 | 1.6× | 0.886 |
| 800 → 1200 steps | 3 | 0.0584 → 0.0404 | 1.4× | 1.000 |
| 400 → 1200 steps | 3 | 0.1500 → 0.0404 | 3.7× | — |

The overlapping subsets matter: the 3 architectures in the second row are a
subset of the 6 in the first, so this is *consistency within one experiment*, not
independent replication, and the two sets were also selected by the halving
schedule (§4a). It is the best evidence the phase had, and it is not strong enough
to plan a search around.

**The measurement, on one matched architecture set with no promotion:**

`arch-zoo-100m-v2-horizon` ran the *same 8 candidates* as v2 — verified
byte-identical before spending the hours — at 400 and 800 steps, then pooled with
v2's 1200-step seeds. 48 runs, 3.25 GPU-hours, one architecture set, one design,
three lengths, flat at each:

| steps | spread | seed std | spread in seed-noise units | best |
|---|---|---|---|---|
| 400 | 0.5457 | 0.0286 | 19.09 | `mtot_D6_H896_E16_A6_W288` |
| 800 | 0.3244 | 0.0165 | 19.71 | `mtot_D6_H896_E16_A6_W288` |
| 1200 | 0.2462 | 0.0148 | 16.59 | `mtot_D6_H896_E16_A6_W288` |

**The decay is confirmed and it replicates.** 2.22× over 400→1200, i.e. **1.65× per
doubling**, monotone — against v1's own per-doubling shrinks of **1.59×** (400→800,
6 architectures) and **1.44×** (800→1200, 3), measured on unrelated, unmatched
families. So the matched design lands within about 10% of v1's rate, just above the
band rather than inside it, and v1's decay was not an artifact of its confounds. This
is the one v1 finding the matched design confirmed rather than corrected.

**But the decay is in the variance, not in the signal, and that changes the
conclusion.** Seed noise falls in lockstep (19.09 → 19.71 → 16.59 seed-noise
units) because a longer run estimates the same loss with less variance. On top of
that the **winner is identical at every length on every seed**, and screening rank
correlation is 0.95 (400→800) and 0.98 (800→1200) with the winner preserved.

A spread that shrinks at nearly the same rate as the noise it is measured against
is not evidence that architecture stops mattering; it is evidence that the
*variance of the loss estimate* shrinks. The effect stays **16–20× its own seed
noise at every length** (19.09 → 19.71 → 16.59), so it does not become harder to
detect as training lengthens, and the ordering — the thing a search actually
consumes — did not move over a 3× change in horizon. The ratio is not *exactly*
flat (it softens ~15% from 400 to 1200), so the honest form of the claim is
"stable in seed-noise units", not "unchanged".

There is a third denominator, and it is the one the brief asks for because it is
denominated in compute rather than in nats. Converting each length's spread into
equivalent training steps uses the observed loss slope, and that slope flattens as
training proceeds — 1.50e-03 nats/step over 400→800 against 6.71e-04 over
800→1200, a 2.2× flattening. Three denominators, three answers:

| denominator | 800 → 1200 | reading it invites |
|---|---|---|
| raw nats | 0.3244 → 0.2462 (**−24%**) | "the effect shrinks with training" |
| seed-noise units | 19.71 → 16.59 (**−16%**) | "it shrinks more slowly than that" |
| share of the training budget (217 → 367 equivalent steps of 800 / 1200) | 27.1% → 30.6% (**+13%**) | "relative to the compute spent, it does not shrink at all" |

The budget share is the most stable of the three and the one denominated in the
currency a search spends. None is exactly flat, so the honest claims are "stable
in seed-noise units" and "stable in budget-fraction terms", not "unchanged" — and
the table exists so a reader can see which denominator any quoted number used.

This is a correction to a decision *rule*, and worth stating as one: the rule that
gated this experiment thresholded the spread without normalising by the noise it
was measured against, so it fired on variance decay and would have read that as
signal decay. `spread_by_length` now reports the seed-noise units alongside the
spread for exactly this reason.

**The decay, measured again at 10× the horizon.**

The rate above is fitted over 400→1200 steps. `arch-zoo-100m-v3-crossed-horizon10x`
re-ran the crossed factorial's **byte-identical** candidate set at **12,000 steps on
two seeds, fp16**, with a same-precision 1200-step anchor in the same directory so
the decay is measured inside one precision. 32 runs, 9.22 GPU-hours, Kaggle T4×2:

| steps | spread | seed std | spread in seed-noise units | best |
|---|---|---|---|---|
| 1200 (fp16 anchor) | 0.3085 | 0.0118 | 26.16 | `xlohi_D5_H896_E16_A2_W384` |
| 12000 | 0.1059 | 0.0056 | 18.88 | `xlohi_D5_H896_E4_A4_W1536` |

**The decay is real at 10×, and slower than the rate above extrapolates.** 2.91× over
1200→12000 is **1.38× per doubling**, against the 1.65× fitted over 400→1200 and the
5.3× that rate would predict across these 3.32 doublings. So extending §3's rate an
order of magnitude **over-predicts** how fast an effect fades, which is the safe
direction for anyone sizing a budget from it. The pre-registered point predictions are
mixed by comparison: they put depth at +0.016 and width at −0.036 nats, and the
measurement lands depth at **+0.0209** — the sign and size as predicted — while width
decays materially *less* than predicted (−0.0645).

**Seed noise falls with it, and the effect stays far above the noise:** 26.16 → 18.88
seed-noise units, still ~19× at 10× the horizon. So the reading this section already
reached holds at this length unchanged — what shrinks is the variance of the loss
estimate, not the detectability of the architecture effect.

**The ordering is where 10× differs from 3×.** Spearman between the candidates'
1200-step and 12,000-step means is **0.90**, but the **winner changes**: `W384` leads at
1200 and `W1536` at 12,000. Over 400→1200 this section could say the winner was
identical at every length on every seed; at 10× that is false. The screen still ranks,
so promoting a *subset* from a short run remains sound, and the screen's winner must
not be carried as the winner — which is the pre-registered reason this arm's own rule,
and not the screen, decides the depth branch (§9).

The measured fp16 offset against the fp32 record at 1200 steps is **−0.0030 nats**
mean per candidate (−0.0257 to +0.0104), smaller than the depth effect it carries, so
reading fp16 rows as evidence about architecture is licensed by measurement rather
than by assumption.

The consequences that constrain every future claim:

- **Effect sizes are horizon-dependent; rank order is not — but the winner is.**
  Quote every effect size with its training length (0.55 nats at 400, 0.25 at 1200,
  0.11 at 12,000 for these sets). The ranking correlation survives 10× (Spearman
  0.90), and the *identity of the best candidate* does not: `W384` leads at 1200 and
  `W1536` at 12,000. A screen is therefore a filter, not a chooser, however good its
  correlation.
- **Successive halving overstates the effect it is hunting** — v1's screening
  stage measured a spread 3.7× larger than the one that survived. Unchanged by
  the above; it is a statement about the *proxy*, not the horizon.
- **§15's matched budget needs a horizon term.** "Matched FLOPs" without a
  training length does not fix the quantity being compared.
- **Short-run screening is usable here.** A 400-step run costs ~1.1 GPU-hours and
  ranks the set at Spearman 0.95 against 1200 steps with the winner intact, so
  screening at 400 is a legitimate way to spend §16's information-per-GPU-hour
  budget — provided the *spread* it reports is never quoted as the effect size.
  The 10× check bounds how far that licence extends: the ranking holds at 0.90 to
  12,000 steps, and the winner changes, so screening **eliminates** at every range
  this phase validated and **chooses** at none of them. Finalists still have to be
  confirmed at the horizon the search actually trains for.
- **It is not only the ordering that survives the screen — the *explanation*
  does.** Fitting the genes the crossed factorial identified (`hidden_size`,
  `depth`) as a **fixed** model at each length, on the same eight candidates
  throughout, gives R² = 0.927 / 0.936 / 0.953 with model p = 0.0024 / 0.0019 /
  0.0011 at 400 / 800 / 1200 steps — and `hidden_size` is individually
  significant at every one of them (−0.863, p = 0.0145 at 400). Holding the model
  fixed is what makes this a statement about the model rather than about whichever
  predictors each length happened to select (the selected-model version is the
  same story: R² 0.946 / 0.960 / 0.955, leading on `hidden_size` every time). So
  the question §3 leaves open — is the effect's *cause* still readable at a third
  of the training — answers yes: what shrinks with training is the effect size,
  not the identifiability of what drives it. The budget consequence is in §10, and
  it is the largest single saving available here.

  *Caveat that applies either way, and is reported in `decision.md`:* the leading
  gene is selected from the spread axes when no model is named, and even a
  pre-specified model's coefficient p is not the designed factorial's exact test.
  This is a consistency check, not a second designed experiment. `--signal-genes`
  names the model; the `hidden × heads` crossing is what identifies it.

## 4. The screen: what it can and cannot resolve

| from | to | n | Spearman | best at short | best at long | winner kept |
|---|---|---|---|---|---|---|
| 400 | 800 | 6 | 0.886 | `shallow_wide_D6` | `narrow_experts_D8` | **NO** |
| 800 | 1200 | 3 | 1.000 | `narrow_experts_D8` | `narrow_experts_D8` | yes |

At 400 steps the ranking is broadly right — Spearman 0.886 — but the *winner*
flips to `narrow_experts_D8` at 800 steps.

**This is often summarised as "the screen does not predict the winner", and the
matched design now shows that is too strong.** The top two at 400 steps were
separated by 0.009 nats, against a per-architecture seed std of 0.0390 nats on that
same 400-step set — a quarter of a std, so those two runs did not separate; a screen
cannot resolve a difference smaller than its own noise, which is a statement about
the *margin* rather than about short-run screening. When the margin exceeds the noise the
screen holds: on v2's matched set the 400-step ranking correlates 0.95 with 1200
steps and keeps the winner, and 800→1200 correlates 0.98 (§3).

The two readings are consistent once separated, and the distinction is worth
keeping because it decides how screening is used:

- **Screening to *rank* is sound**, and at 400 steps costs ~1.1 GPU-hours against
  3.2 for 1200 — the cheapest information per GPU-hour in the phase. The joint
  structural signal is equally significant at 400 as at 1200 (§3), so a screen
  does not merely order the candidates the same way, it recovers the same
  *reason* — at a third of the price. The one caveat is selection: the leading
  gene is picked *from* the spread axes at each length, so its own p-value is
  uncorrected for that choice and is a consistency check rather than a designed
  test. The designed test remains §2c's crossed factorial.
- **Screening to *promote* is not**, because promotion discards the candidates a
  noisy proxy mis-ranked, and the discarded ones are never seen again at a longer
  horizon. A promotion decision needs the margin to exceed the seed noise, which
  at this scale means ≥800 steps. The threshold is not a pinned constant: it is the
  standard error of the *difference* between two screen means, `std * sqrt(2 / seeds)`
  off the measured stds in §4 (0.0286 nats at 400 steps, 0.0165 at 800, 0.0148 at
  1200), so a two-seed 400-step screen needs a 0.0286-nat margin to claim a cut is a
  decision. The earlier pin of 0.018 nats was not a quantity this design produces —
  it is 63% of that threshold, i.e. it would have called ties resolved that were not.
- **Screening to *estimate an effect size* is worst of all**, because §4a shows
  the screening stage reports a spread 3.7× larger than the one that survives.

**The corrected threshold is stricter and it overturns nothing already measured.**
Replayed over the phase's own screens, every historical cut resolves under either
threshold, because the margins there are 0.079–0.088 nats against thresholds of
0.012–0.029: v1's 400-step cut (0.0884 nats, 2 seeds each), its 800-step cut
(0.0422), the matched set's 400-step cut (0.0835, 3 seeds) and its 1200-step cut
(0.0798). The correction bites where a search's cut lands on a near tie — which is
what §9's loop does by construction — and one is visible even in a long run: at 1200
steps the third-to-fourth cut has a candidate within 0.012 nats of the cut loss,
inside that stage's 0.0121-nat threshold, so a tie-break there would be choosing
between two runs that did not separate.

Each verdict is now written into the arm's `decision.md`, not only printed: the row
carries the stage, its length, its seed count, the measured std, the derived
threshold and the margin, so which cuts were decided by data is checkable from the
record rather than from a terminal that has since closed.

This is why v2 ran flat rather than halving: a design measuring the final spread
must not select on a proxy.

## 4a. What successive halving actually bought

§16 asks for the cost of *discovering* an architecture, and §29 for information
per GPU-hour. The v1 records answer that directly, because a flat design at the
same length is a computable counterfactual:

| plan | runs | GPU-hours | architectures tested at 1200 steps | design power floor |
|---|---|---|---|---|
| successive halving (as run) | 47 | 3.60 | 3 | 1/3!² = 0.028 |
| flat at 1200 steps | 39 | 5.44 | 13 | 1/13!² ≈ 2.6e-20 |

Halving saved **34%** of the compute. In exchange it reduced the tested set from
13 architectures to the 3 that a biased short-run proxy happened to favour — and
that proxy put the eventual winner in second place. So the saving was real but it
bought a comparison with a p-value floor of 0.028 that could not resolve the
effect it was looking for, on a subset chosen by a criterion this same experiment
shows does not predict the outcome.

At this scale the cheaper plan produced a strictly less informative experiment for
a third of the cost. That is worth knowing before choosing a plan for a larger
search: screening is usually justified by the assumption that short runs rank
architectures the way long runs do, and here they rank them at Spearman 0.886
while getting the winner wrong.

## 5. Nothing structural explains the screening spread

Across v1's stage 1, no structural variable predicts loss at 400 steps:

| variable | Spearman ρ | p |
|---|---|---|
| total parameters | −0.261 | 0.388 |
| active parameters | −0.137 | 0.655 |
| contracted FLOPs/token | −0.077 | 0.803 |
| dense-runtime FLOPs/token | −0.327 | 0.275 |
| measured throughput | +0.038 | 0.901 |

So the 0.72-nat screen spread is not a parameter-count effect — and with one seed
per candidate it also cannot be separated from noise. v1's own design could not
attribute it; that is what v2's matched-total design exists to fix.

## 6. Two matching bugs found and fixed

**The zoo was drawn from a wider grid than the random baseline.** Writing the
containment test showed the curated candidates used depth 14, 5–6 active experts
and expert widths the declared search space could not produce. The random
baseline would have sampled a strictly *narrower* space than the manual picks it
is compared against, flattering the curated zoo in exactly the §11 comparison.
There is now one authoritative grid (`enumerate_search_space`, 15,224 cells)
that every method draws from, with a test asserting containment.

**Matched designs covered half the axis they claimed to spread.** Filling slots by
walking the matched band from one end means the structural-diversity floor
rejects most neighbours, so every accepted cell comes from the same end. At 78M
±5% the band holds 1,059 cells spanning 18.0M–81.8M active parameters; the
end-first sweep returned 18.0M–37.8M, so a design whose entire purpose is
"matched total, active spread" would never have tested a dense model. Stratifying
the axis into one bin per slot fixes it:

| design | matched axis | varied axis (before → after) |
|---|---|---|
| `matched-total` @78M | 7.9% | 2.10× → **4.51×** |
| `matched-flops` @2.6e8 | 7.6% | 1.56× → **3.33×** |
| `matched-active` @35M | 8.7% | 1.41× → **3.89×** |

The existing test passed the buggy version, because it asserted
`varied_spread_ratio > 1.2` and the broken design reached 2.1×. The replacement
asserts *coverage of the available band*, which is the property that matters.

Three further defects were in the analysis path rather than the science, and all
three would have been silent:

- **The analyser could not see a flat sweep at all.** `arch_zoo_analyze.py`
  globbed `stage*_seed*_results.jsonl`, which matches successive halving and
  nothing else. Since v2 runs flat, its report would have come back
  "no result ledgers found" *after* all of its training had been paid for. The
  glob is now `*_seed*_results.jsonl` and the error names the pattern it looked
  for.
- **Stale ledgers were only partly cleaned.** The driver deleted `stage*` files
  before a run, so a second flat run in the same directory accumulated duplicate
  rows instead of restarting. It now deletes exactly the tags it is about to
  write, which also lets a horizon arm keep two training lengths side by side in
  one experiment.
- **The selection caveat was asserted rather than derived.** `effect_summary`
  hardcoded "finalists were promoted using shorter runs", which is true of v1 and
  false of v2's flat design, so the report would have understated v2's evidence.
  Whether promotion happened is a fact about the data - more architectures were
  run at a shorter length than were run here - and it is now computed that way.
- **The structural genes were joined in the wrong place.** Six of the twelve
  design axes (depth, hidden size, attention heads, expert count, active experts,
  expert width) live in the zoo *definition*, not in the ledger, because ledgers
  predating those fields record zeros. The join existed, but only inside the
  reporting script — so any other caller read all six as "not recorded" and had
  its leading axis chosen from the cost axes alone. It reported
  `total parameters` as the strongest association where the joined analysis
  reports `hidden size`, which is a silently inverted conclusion. The join is now
  `with_structure` in the report library, used by the script and by the test that
  pins the v2 verdict.

- **The crossed design's level selection maximized the wrong thing.** The first
  version chose factor levels by widening one factor as far as possible, which on
  a `depth x hidden_size` crossing selects **depth 5 against 6** — the two most
  similar depths in the grid — because adjacent depths leave the most room for
  expert width. The zoo would have reported itself as a balanced crossing with
  `rho = 0.000` while holding depth nearly constant, and the depth axis would
  have looked null for the trivial reason that it barely varied. The objective is
  now the *product* of the two spreads, and a test asserts both factors reach the
  extremes of their domains. This was caught by printing the design before
  spending hours on it, which is the second time this phase that a dry run paid
  for itself.

A fifth change is statistical rather than a bug fix. A matched design of 8
candidates on 3 seeds has 8!³ ≈ 6.6e13 within-block assignments, so the exact
enumeration cannot run and the primary test was silently falling back to the
chi-square approximation - which is exactly where a 3-seed test loses its
validity. It now samples the same null instead: a real 0.30-nat gradient over
8×3 gives p = 5e-5, pure noise gives p = 0.55, and the v1 3-candidate shape still
enumerates exactly. The decision text reports the method and the procedure's
resolution alongside the design's own power floor, because those are different
limits and only one of them is a property of the design.

## 7. §14 — module type is not available at this scale

`layer.module` was a live gene that the MoE path ignored entirely; GDN is now a
real mixer (`_GatedDeltaMixer`, a compact member of the Gated DeltaNet family) so
the axis is at least representable. Measured at depth 8, hidden 768, context
1024, batch 2:

| pattern | GDN layers | tok/s | vs attention | peak MB |
|---|---|---|---|---|
| `all_attention` | 0/8 | 4,793 | 1.0× | 3,498 |
| `att_3to1` | 2/8 | 1,209 | 4.0× | 5,048 |
| `alternating` | 4/8 | 700 | 6.8× | 6,605 |
| `gdn_3to1` / `all_gdn` / `all_hybrid` | 6–8/8 | — | **OOM** | > 8,192 |

The memory cost is understood rather than merely observed: the explicit scan
retains a `(B, H, d, d)` write and state per timestep, ≈0.8 GB per GDN layer, and
the prediction matches the measurement at 2 layers (5.0 GB), 4 layers (6.6 GB)
and 8 (OOM). Segmented checkpointing is implemented (`gdn_chunk_size`) and
verified to be numerically free — outputs *and gradients* identical, state
verified to survive a chunk boundary — which removes the memory blocker but not
the time one: at 4 GDN layers the forward runs 4,096 sequential Python steps and
2.9 s/step is launch-bound.

So module type cannot be a compute-matched gene here without a chunkwise parallel
kernel. Per §30 that is deferred, not written speculatively. **This is the one
item on the §28 checklist that is blocked, and it is blocked by hardware rather
than by the science.**

The wired-through state is: `layer.module` selects the mixer,
`RunConfig.gdn_chunk_size` reaches the config, and both are recorded per run, so
the axis is one kernel away from being searchable rather than one refactor away.

## 8. Deliverables against §28

| §28 item | status |
|---|---|
| 8–16 valid genomes | 13 (v1) + 8 matched-total (v2) + 8 crossed (v3), all cells of one declared space |
| Parameter/FLOP matching report | `matching_report.json`, three cost axes; runtime-vs-contract FLOP ratio spans 1.00–3.18 |
| Short architecture-zoo runs | v1 complete (47 runs, 3.60 GPU-h); v2 complete (24 runs, 3.20 GPU-h, flat 1200 steps); horizon complete (48 runs, 3.25 GPU-h, flat 400 + 800); crossed complete (24 runs, 3.28 GPU-h, flat 1200); crossed confirmation complete (24 runs, 3.26 GPU-h, fresh seeds 45–47, pooled into the crossed arm's decision); head-paired complete (24 runs, 3.50 GPU-h, flat 1200, seeds 48–50 — **width confirmed, head count excluded**); head-paired confirmation complete (24 runs, 3.511 GPU-h, fresh seeds 51–53 — pre-registered with its budget overrun declared in that directory, and pooled into the head-paired record: **head count priced, free only on a wide residual stream**) |
| Ranked candidate table | `ranked.json` |
| Crossed (factorial) design | `--design crossed`: balanced `depth x hidden_size` factorial with a reported balance check; added because v2's matched set confounds hidden size with expert width at rho = -0.74, so its attribution is unidentifiable without one. **Run and analysed** — `crossed_contrast` reports both main effects at exact p = 4.3e-05 (§2c) |
| Head-count gene made crossable | `attention_heads` added to the crossable genes and `_head_counts` to the cell enumeration, with a regression test that existing designs keep their original candidate names. `generate_head_paired_zoo` (`--design head-paired`) goes further: because head count is parameter-free, a pair can be matched on total *and* active parameters **exactly** rather than landed in a 5–10% band, and `paired_contrast_test` gives the paired contrast an exact floor of 2/2^12. **Run and analysed**: the head contrast lands at −0.0177 nats (p = 0.14, signs 8/4) against a width contrast of −0.1562 nats at the factorial's floor, which is the pre-registered "width is the lever" branch |
| Architecture-distance matrix | `distance_matrix.json`; real module-type edit distance, min pairwise 0.060 |
| Search-cost accounting | `search_cost.json`; includes the cost of *discovering* the architecture, not only training the winner. A pooled write is labelled: `pooled_experiments` plus a `by_experiment` split, because the file lands in the primary arm's directory while covering both |
| The record itself, checked | `scripts/verify_record.py` recomputes **283 load-bearing claims** from the ledgers (**263** of them on the `--quick` path the test suite runs) — every §10 cost row, the §4 spread/noise table, §4a's decay and wrong winner, §2c's three-seed, six-seed and replication effects with their floors, the head-paired arm's paired and factorial contrasts and the branch its numeric conditions select, the confirmation arm's design-integrity claims (candidate set byte-identical to the parent's, fresh seeds, flat plan, the paired genes reachable through the structural join), §9a's budget match and its new size condition (the power figure is recomputed from the same population and noise, not read from the table), §5.63's transfer rehearsal (the fresh path replaces every trained value while the restore path reproduces them exactly), and the README headline's structural claims — and fails when the record and the artifacts disagree. Pinned by `tests/test_verify_record.py`, because three documented numbers were wrong after the test suite was green |
| Capability probes on finalists (§19) | plumbed: `train_architecture(..., probes=...)` records per-probe loss/accuracy/token count in `EvalResult.capability`. Empty by default, and tested to stay empty, because probing every candidate would silently add a second selection objective. No probe *set* exists in this repository yet, so none is applied |
| Seed-matched results | §20 implemented and used; exact blocked permutation test, extended with a crossed factorial (`crossed_contrast`) whose main effects are exact and whose interaction is reported against three nulls; the confirmation arm's four-outcome rule is evaluated by `replication_decision`/`--replication-rule` with the thresholds in `REPLICATION_RULE`, so a pooled branch is read mechanically rather than by eye |
| Robustness checks on the headline claims | Held-out-loss effects re-run on training loss (v2, the crossed factorial and the horizon arm at 12,000 steps), so none is a holdout-split artifact; leave-one-out over candidates where the design allows it (v2's 8 distinct configurations, and the horizon arm's spread, whose worst drop leaves p = 0.0018) and an explicit *refusal* where it does not (the crossed and horizon arms' 2-per-cell design becomes unbalanced, which used to crash rather than report); the horizon arm's two-seed requirement is checked as *forced* rather than cautious, since a single seed's 2/36 floor sits above the 0.05 threshold; every significance headline in the phase has now been re-read on training loss, and the two that move are both *negatives*: the head-paired arms' branch holds on the like-for-like pair (held-out and final training loss) and reverses on a trajectory-averaged loss, and the screening-free finalists' "not established" holds on the pair while reaching its three-block floor beyond it — while the positive claims (v2's and the crossed arm's attribution, the horizon arm's depth sign, and both spread-only arms at 3× and 10×) hold on every reading; the pooled finalist-order *branch* also holds on all three, and only the clause explaining it moves; the head-paired pricing the recommendation rests on holds on all three, at the exact floor with all 12 wide-level units on one side; spread-not-worse and dropped-candidate guards; an **in-flight seed is dropped from the crossed and paired contrasts and named**, so a sweep analysed while its last seed is still training reports the contrast over the seeds that finished rather than reporting nothing — the drop is disclosed in the decision and the procedure floor moves with it, and only a seed short of the fullest one's row count is droppable, so an unbalanced-but-complete seed still refuses; a seed that stopped *between* pairs leaves nothing incomplete and is named by its pair count instead, because the unit count is the only place that shows and the floor printed beside the p moves with it; the seed-matched permutation test runs only over a complete rectangle, so the seeds and architectures it had to intersect away are named too — the block count beside its p is the survivors, not the design, and a dropped seed that wrote as many rows as any other is called a zoo defect rather than an unfinished sweep. All pinned by tests against the real ledgers |
| Procedure floors reported beside every p | `min_achievable_p` on the blocked permutation test, the crossed main effects (2/46,656 = 4.29e-05, both **at floor**), the seed-blocked interaction (2/2^seeds) and the paired contrast (2/2^12). Added after the horizon arm showed what a floor-bound statistic read as evidence costs: a p that is the most extreme its test can express is a statement about the design, not about the effect |
| §11 random-vs-manual readiness | `RandomArchitectureSearch` baseline plus `scripts/space_typicality.py`, which checks whether the curated sets occupy a distinguishable region of the space before any equal-compute comparison is run. They do not (§9a), so the comparison is not circular — and the *budget* check is done too, which is the one condition a band does not buy: a band-matched 13-candidate draw carries a **median 1.12×** the curated set's total parameters (p10–p90 1.05–1.18×, range 1.00–1.25× over 200 draws, median 1.15× at a ±15% band), because the curated set sits at 86M against the nominal 100M — so a budget-matched sampler seed must be pre-registered (§9a). The *size* condition is now measured too, before the hours: `scripts/set_comparison_power.py` puts 80% power at a **0.30-nat set-mean advantage** for 13 candidates per set over 2 seeds (26 runs, ~1.20 GPU-hours), and shows the lever is candidates per set rather than seeds — 26 per set reaches 0.155 nats at 2.39 GPU-hours, while 2/3/4 seeds are indistinguishable. §11 as designed is therefore affordable but not decidable at the curated set's size, and its coverage form is invisible at any size measured — 0.03 power at a 0.60-nat better winner, against 0.94 for the same statistic on a set-wide shift (§9a) |
| Decision on whether differences are real | `decision.md` per experiment — v1 NOT ESTABLISHED (power floor 1/36); v2 **ESTABLISHED** at p <= 5e-06 on 3 seeds (v1's figure was 0.00023 on 2); crossed **ESTABLISHED** at p <= 5e-06 with both main effects at the exact floor; the six-seed pooled record (crossed + confirmation) **ESTABLISHED** at p <= 5e-06 on 6 seeds, spread 0.3018 nats; head-paired **ESTABLISHED** as well (spread 0.2221 nats on the committed six-seed record, p <= 5e-06, against 0.2126 on the parent's own three), and its own question answered the way the confirmation resolved it rather than the way its first read suggested: the pooled head contrast is −0.0234 nats (p = 0.0115), which is an average of two *disagreeing* levels — −0.0553 nats at H=896 with 12/12 pairs negative, against +0.0084 nats at H=512 (p = 0.37) — so head count is real at wide width and nothing at narrow; its confirmation arm **ESTABLISHED** on fresh seeds (spread 0.2412 nats, p <= 5e-06); each design's power floor and procedure resolution stated |
| No 285M run | none |

## 9. Decision, per §23

Differences are **established at a matched budget**, and both questions that gated
*how much* it is worth have been measured rather than deferred:

1. **Horizon — measured at 3× and at 10×, and it does not block the search.**
~3.6 GPU-hours, then ~9.2 more on a cap of its own. The decay is real and replicates
v1 (2.22× over 400→1200, 1.65× per doubling, against v1's pairwise 1.59× and 1.44×)
*and* the noise decays with it, so detectability is horizon-stable. At **10× the
horizon** — the crossed factorial's byte-identical candidate set at 12,000 steps on
two seeds — the pre-registered rule lands on **shallow-wins persists**: depth
**+0.0209 nats, exact p = 0.003086** (floor 0.001543) and width −0.0645 nats at its
own floor, so the depth recommendation survives the length prior work predicted would
reverse it. Two qualifications come with it: the decay is *slower* than §3's rate
extrapolates (1.38× per doubling against 1.65×), and the **winner changes over 10×**
even though the ranking holds at Spearman 0.90 — a screen promotes, it does not
choose.

   **Its robustness was checked at that length rather than inherited from 1,200.**
   Re-run on *training* loss the two main effects keep their signs and stay
   significant (depth **+0.0249** at p = 0.0077, width **−0.0667** at the same
   1296-assignment floor), so the horizon attribution is not an artifact of where the
   validation tokens were cut — a check that matters *more* at a longer schedule,
   which is where a difference in how much each architecture overfits would widen.
   Dropping any one of the 8 candidates still leaves the spread significant (worst
   blocked p = **0.0018**). The *crossed* attribution cannot be checked that way and
   says so: at 2 candidates per cell, removing one unbalances the factorial, so it
   **refuses on all eight** rather than computing over a subset no design supports.
   And two seeds here is arithmetic rather than caution — the re-split test is
   two-sided, so one seed's 36 assignments floor its p at **2/36 = 0.0556**, above the
   0.05 threshold, which is why the arm's stale single-seed record had no verdict to
   state as well as no verdict to state it with. **No single seed can reach
   significance in this design at all.**

   Reading: the effect is a lever, every effect size must be quoted with its
training length, and the depth sign is now horizon-stable to 10×. Full record in §3.

2. **Attribution — measured, replicated, with its last confound now closed.** ~10.0
GPU-hours for the crossed arm, its confirmation and the head-paired arm.
Residual-stream width is the lever at **−0.1836 nats (~274 steps)**, depth is real
but smaller at **+0.0783 nats (~117 steps)**, both at the exact enumeration floor
p = 4.3e-05, and depth's sign is the unintuitive one (**shallow beats deep** at
matched total parameters and matched hidden size). Everything in that paragraph
**replicates on seeds never used to design or screen anything**: −0.1843 and
+0.0824 pooled over six seeds, same signs, same ranking, and the 2.6× level
dependence of the depth penalty resolved to +0.0827 nats at the six-seed floor.
The one reading that could have overturned it — that "hidden size" was really
attention-head count, which is free — was tested by the head-paired arm at
*identical* within-pair cost and came back **−0.0177 nats (p = 0.14)** against a
width contrast of **−0.1562 nats at the factorial's floor**. Width stands and the
recommendation does not reverse. Full record in §2c.

3. **A search program, aimed at the identified levers — with the degree of
   specificity that survives the confound.** This is the phase's recommendation,
   and the crossed arm is what makes it specific: spend budget on the residual
   stream and at *low* depth, against total-parameter matching, rather than in
   expert capacity. Both levers are now unconditional: depth's effect was measured
   and unconfounded from the start, and the width recommendation no longer waits on
   the head-paired arm, which found head count worth −0.0177 nats (p = 0.14). The
   head axis is now *priced* rather than excluded: its confirmation arm resolved the
   interaction at the six-seed floor, and heads are worth **−0.0553 nats at wide
   hidden sizes and nothing at narrow** — free parameters, so a search should spend
   the head gene *only* where the residual stream is wide, and mutate the two
   jointly (§2c). The pricing rests on that level split rather than on the pooled
   head contrast, and deliberately: the pooled contrast clears the pooled rule's
   declared threshold by only 1.15× (p = 0.0115 against 0.01) and its significance
   is loss-reading-dependent, while the level split holds on every reading with all
   12 wide-level units favouring more heads at the exact floor (§2c).

   The shape of the loop follows from §3 and §4 rather than from taste: a
   **screen at 400 steps, promote only on margins that clear the seed noise, and
   confirm finalists at the horizon the search actually trains for.** Screening at
   400 costs 0.046 GPU-hours per run against 0.133 at 1200, and §3 shows it
   recovers the explanation and not only the ordering. Its *price*, though, is set
   by its last step, and §10 now prices that: **17.0 GPU-hours while the confirmation
   stays at 1,200, but 26.4 with it at a 12,000-step horizon on Kaggle and 45.7 on
   this card** — because at that length the confirmation stops being the small term
   it is at 1,200: 12.57 GPU-h against 13.80 of screening on Kaggle, and 31.92
   against 13.80 on this card. (Those two totals were quoted as 29.6 and ~49 until
   they were re-derived from their own arithmetic: each added the horizon
   confirmation to the *complete* 1,200-step loop, paying twice to confirm the same
   24 finalists. §10 states the composition and records the correction.) The
   obligation that does *not* relax: quote every effect size with
   its training length, because that is the one quantity the screen gets wrong by
   3.7×.

   A confirmation longer than 1,200 steps carries one measured caveat, and it is in
   the safe direction. The promotion threshold needs a seed std *at the confirmation
   length*, the table of measured stds ends at 1,200, and `seed_std_for_steps` takes
   the nearest measured length at or below the stage. For the 12,000-step horizon this
   phase measured, that borrow is a **2.6× over-estimate**: the arm's 12,000-step rows
   give a per-architecture std of **0.0056 nats** against the borrowed 0.0148, so the
   margin demanded is 2.6× wider than the data requires. It costs seeds, not
   correctness — a wider margin promotes fewer candidates on noise — and the measured
   figure is deliberately **not** substituted, because it is not the same measurement:
   different candidate set (the crossed factorial against the matched v2 set),
   different precision (fp16 against fp32), and two seeds against three. The arm's own
   1,200-step anchor reads 0.0118 where that table reads 0.0148 at the same length,
   which is how far apart the two designs' noise levels actually are. The std is now
   printed with the length it was measured at whenever it is borrowed, so the
   provenance travels with the number instead of being inferred from it.

Running or done, and the reason each was worth the hours:

- **The depth×width interaction** — **done**, 24 runs, 3.26 GPU-hours, as
  `arch-zoo-100m-v3-crossed-confirm`. It bought three fresh seeds on a
  byte-identical candidate set with a pre-registered four-outcome rule, and the
  pooled six-seed analysis lands on **replicates, interaction resolves**:
  interaction +0.0827 nats with the seed-blocked p at its 2/2⁶ = 0.031 floor and
  all six seed contrasts agreeing, main effects −0.1843 / +0.0824 as against the
  original −0.1836 / +0.0783. The arm's arithmetic is why it needed *seeds* rather
  than architectures: three seeds floored the test at 0.25 and five at 0.0625,
  both above the declared 0.05, so only the sixth could decide it. Consequence for
  the search: `depth` and `width` are **not** separable and must be mutated
  jointly (§2c).
- **Width vs head count** — **done**, 24 runs, 3.499 GPU-hours, as
  `arch-zoo-100m-v4-head-paired`. The one open question that could have *changed
  the recommendation* rather than refined it: head count is free, and §2c's
  attribution could not tell it apart from residual-stream width. Depths fixed, both
  factors balanced, and within-pair costs **identical** rather than merely close, so
  the head contrast is the most precisely matched measurement this project can make —
  and it came back **−0.0177 nats, p = 0.14, signs 8/4**, against a width contrast of
  −0.1562 nats at the exact floor. That is the pre-registered "width is the lever"
  branch, so the phase's recommendation stands. The arm's one new lead — a
  level-dependent head effect, +0.0571 nats of interaction with all three seed
  contrasts and all six H=896 pairs agreeing, against a seed-blocked floor of 0.25
  that no re-analysis can beat — was priced at one more seed block (~3.5 GPU-hours).
- **The head interaction** — **done**, 24 runs, 3.511 GPU-hours, as
  `arch-zoo-100m-v4-head-paired-confirm` on fresh seeds 51–53 with the candidate set
  verified byte-identical and the budget overrun declared in its pre-registration
  before its first run. Pooled to six seeds it lands on **head effect is real at wide
  width**: interaction +0.0636 nats at the 2/2⁶ = 0.031 floor with all six seed
  contrasts agreeing, head effect −0.0553 nats at H=896 (12/12 pairs, at the 2/2¹²
  floor) against +0.0084 at H=512 (p = 0.37). Consequence for the search:
  `attention_heads` and `hidden_size` are **not** separable either and must be
  mutated jointly, with the head gene spent only at wide widths (§2c).
- **More seeds on v1's finalists** — **done**, 9 runs, 1.236 GPU-hours, as
  `arch-zoo-100m-v1-finalists-6seed`. Fresh seeds 45–47, the candidate set verified
  byte-identical to v1's stage 3, and deliberately **fp32**: v1's rows predate
  `mixed_precision`, the pooled test needs one precision, and the measured fp16
  differential (0.018 nats) is the size of the gaps to be resolved (0.0232 and
  0.0172). It therefore ran on the idle local card in parallel with the horizon
  kernel rather than on Kaggle, whose ~3× throughput advantage is an *fp16*
  advantage and only ~1.25× at fp32. Pooled to six seeded blocks it lands on
  **order coupled**: the pooled exact test clears at p=0.001286 (floor 1/7776) while
  the screening-free blocks alone neither clear the threshold (p=0.1111, floor 1/36)
  nor agree on a winner — two different architectures finish first across the six
  blocks. The pooled significance **is** the blocks that selected the finalists, the
  anti-conservative coupling v1's caveat named, so **no ranking is carried into the
  search** and v1's `Next step` no longer points at this arm. Read on the other two
  losses the branch is stable: all three clear the pooled threshold (10/7776 held-out,
  82/7776 on final training loss, 2/7776 on the trajectory average, against a 1/7776
  floor) and two different winners top the six blocks in every reading, so the third
  condition decides the branch whichever loss it is read on. The reading-dependent
  clause is the *second* one — "the screening-free blocks alone clear the threshold"
  is 4/36 held-out and 2/36 on final training loss (both above 0.05, which is what
  licenses this paragraph) but **1/36, the three-block floor, on the trajectory
  average**, where the uncoupled blocks do separate the finalists. So the verdict holds
  on the pair the split check is designed around, and the sentence explaining *why* is
  the part that moves. Consequence: v1's
  *ordering* is not evidence to build on; the effect size is quoted at 1200 steps as
  always, and the search's direction comes from §2c's designed factorial, which
  never rested on a ranking.
- **Does the depth attribution survive 10× the training?** — **done**, 32 runs,
  9.22 GPU-hours, as `arch-zoo-100m-v3-crossed-horizon10x`, on Kaggle T4×2 under its
own pre-registered 15 GPU-hour cap. The one reading most likely to be an
early-training artefact: at convergence prior work favours deep-narrow at fixed
parameters, while the crossed arm found shallow beating deep by +0.08 nats at 1200
steps. Byte-identical candidate set, flat at 1200 and 12,000 steps, two seeds, fp16
with a same-precision anchor so the decay needs no cross-precision comparison, and a
four-outcome rule declared before the first row existed. It lands on **shallow-wins
persists** (depth +0.0209 nats, p = 0.003086; width −0.0645 at floor), so §2c's
"spend at low depth" stands at this scale — the lever list does not need amending.
The arm's own two qualifications are in §3: the decay is slower than §3's rate
extrapolates, and the winner changes across 10× even though the ranking holds. The
consequence for the search is that dismissing the depth axis was wrong *and* that
the horizon at which a finalist is confirmed is load-bearing rather than a formality.
- **The §11 random-vs-manual comparison** (~a v1-scale sweep), which §9a shows is
  now unblocked *and* de-confounded.

Module-type widening (§14) is **not** in that list because it is not runnable at
this scale on this card, which is a finding rather than an omission.

## 9a. The §11 random-vs-manual comparison: ready, de-confounded, and out of reach at its own size

§11 asks for the first scientific comparison of randomly sampled vs manually
selected architectures under equal compute. The precondition this document
previously set — an architecture effect distinguishable from seed noise — is met
by §2a: 0.2462 nats of spread against 0.0148 nats of seed noise, at p <= 5e-06,
with the lever identified as residual-stream width (§2c), so the comparison has a
declared direction rather than only a null to reject. The section now closes on a
fourth condition that no GPU time fixes: at the curated set's own 13 candidates the
comparison cannot see the effect it is asking about (§9a, last condition).

### The confound, checked before the hours were spent

An equal-compute comparison of manual vs random is only informative if the manual
set is a *fair* manual selection. If the hand-curated zoos already sat in the
region §2c later measured to be good, then "manual beats random" would measure
hindsight rather than judgement, and the conclusion would be circular.

That is checkable from the zoo definitions alone, with no training, and the
answer is that **the curated sets carry no such advantage** — measured by
`scripts/space_typicality.py` against every cell within ±5% of 78M (1059 cells),
with each statistic's percentile taken among uniform draws of the same size and
quoted as the tool renders it, to the whole percentage:

| set | n | depth mean | depth spread | hidden mean | hidden spread |
|---|---|---|---|---|---|
| population (uniform grid) | 1059 | 8.97 | 3.35 | 718 | 143 |
| `v1` manual families | 13 | 9.69 (p83) | 2.82 (**p16**) | 684 (**p19**) | 151 (p64) |
| `v2` manual matched-total | 8 | 9.38 (p68) | 4.39 (**p96**) | 680 (**p22**) | 194 (**p98**) |
| `v3` *designed* factorial | 8 | 9.50 (p72) | 5.14 (**p100**) | 704 (p42) | 219 (**p100**) |

`v3` is a balanced experimental design rather than a manual selection and is
included only as a contrast: its 100th-percentile spread is what a design looks
like when diversity is the explicit goal.

**Two readings, both useful:**

- **No location advantage, and if anything a penalty.** The manual sets' hidden
  size sits at the 19th and 22nd percentile — they are *narrower* than a random
  draw, which is the direction §2c measured to be worse. So a manual win in §11
  cannot be hindsight: the curated sets were not picked toward wide residual
  streams. If anything the manual arm starts disadvantaged on the one gene known
  to matter, which makes any manual win *more* interesting, not less.

  Note what these percentiles do and do not say. A percentile is a position, not
  a p-value: p19 means 19% of random draws of that size land even narrower, so the
  lean toward narrow hidden sizes is a *direction*, not a significant bias, and
  with 8–13 candidates it should stay a direction. The claim being supported is
  the null one — no detectable location advantage — and that is what the
  comparison needs to be non-circular.
- **A coverage advantage, concentrated in the designed set.** `v2`'s spread is at
  the 96th/98th percentile and `v3`'s at the 100th, while `v1`'s hand-authored
  families sit at the 16th percentile for depth spread — 13 candidates crowded
  onto depths 8–12. Under equal compute a diverse set covers more of the space,
  so **§11's likely mechanism is coverage, not per-candidate quality.** That is
testable and it predicts the shape of the result: if manual wins, expect it to
win by its set's spread rather than by its best member, and design the analysis to
separate those.

### Conditions that still apply

Four were learned in this phase, and the last two were measured rather than
assumed — the third while checking the draw, the fourth before any hours were
spent on it:

- **Equal compute must mean equal *total* parameters, not equal FLOPs.** §2a shows
  a ×4.51 spread in active parameters under a matched total budget has no
  association with loss, while `matching_report.json` records runtime-vs-contract
  FLOP ratios spanning 1.00–3.18 for v1. "Matched FLOPs" is three different claims
  depending on whether it means the routing contract, the arithmetic executed, or
  the clock, and the recorded wall-clock spread is the smallest of them.
- **Neither side may be screened on short runs.** A promotion-based halving
  schedule would report whichever set the short horizon favoured (§4), and would
  additionally *report* a spread 3.7× larger than the one that survives (§4a).
- **The two sets must be matched on budget, and the default draw is not.**
  "Equal compute" is not bought by drawing from the same ±30% band: the *sum* of
total parameters over a set is what the arm's training budget is, and a band is
  a per-candidate constraint whose set total varies with the draw. Measured with
  no training at all — `v1`'s 13 curated families hold 70.8M–123.3M total
  parameters (×1.74, **sum 1.121e9**, mean 86.3M), while a 13-candidate uniform
  draw at the same nominal 100M ±30% (seed 42) holds ×1.79 and **sum 1.331e9** —
  **1.187× the manual arm's budget**, and 1.150× even at ±15%. That seed is one
  draw rather than the rule, so the distribution is measured too: over 200  draw seeds the median is **1.12×** and the p10–p90 span is 1.05–1.18× (mean 1.12,
  full range 1.00–1.25×) at ±30%, and the median is **1.15×** at ±15% (span
  1.07–1.24×). The reason is structural rather than random — the
  curated set's mean candidate sits at 86M against the nominal 100M, so *any* draw
  centred on 100M carries about 12–15% more budget. Against effects this phase
  measures at 0.18 nats, that is a comparison run with 12–19% more training than
  the manual arm. Both distributions are printed by `space_typicality.py` beside
  the closest draw, so the quoted spans are reproducible from the tool rather than
  from one seed. So the draw has to be budget-matched
  (choose the sampler seed whose total lands within a stated tolerance of the
  curated set's total, and pre-register that seed) rather than merely
  band-matched, and the matching statistic is reported before the hours are spent.
  That is now measured rather than cautioned: scanned over 200 draw seeds, the
  closest 13-candidate draw to `v1`'s total is **seed 151**, at 1,120,992,384
  params — **0.9996×** the curated set, against 1.1872× for the seed-42 draw — so
the launch line is `--design random --candidates 13 --zoo-seed 151
--target-params 100000000 --random-tolerance 0.3 --context-size 1024
--max-total-params 150000000`. Pre-flighted through the driver itself
(`--dry-run`, 2026-09-18): the emitted zoo reproduces the matched draw exactly —
13 candidates, set total 1,120,992,384 (0.9996×), mean 86.23M/candidate — and the
driver's plan projection for the flat arm is 39 runs ≈ 5.92 GPU-hours at its
conservative 4,500 tok/s, beside the ~5.2 the measured card rate implies for the
same 95.8M tokens. The band is part of the line rather than a default
  because it decides the draw: the *same* seed at the driver's default 0.15 band
  produces a set totalling 1,282,387,008, **1.1435×** the curated budget, so seeding
  without the band silently drops the very match it was chosen for (checked against
  the driver's own zoo output, not the sampler's). The matched
  draw is *not* a diversity advantage in disguise either: its structure (depth
  mean 7.8 spread 3.2; hidden mean 718.8 spread 137.8) sits near the population's
  (9.49 / 3.50 and 716 / 144), which is what makes it usable as the comparison's
  random arm rather than a second designed one.
- **The design has to be able to see the effect, and at 13 candidates per set it
  cannot.** Measured with no GPU time at all — `scripts/set_comparison_power.py`
  draws both sets from the curated arm's own measured candidate spread (0.7411
  nats across its 13 candidates at 400 steps) through its measured per-run seed
  std (0.0390) and runs the shipped seed-blocked test on the result: at 13
  candidates per set over 2 shared seeds — 26 runs to buy, **~1.20 GPU-hours** —
  **80% power needs a 0.30-nat set-mean advantage.** The null rows of that table
  read 0.04/0.04/0.03 against the nominal 0.05 for the `mean`/`best`/`worst`
  statistics, so the power figures above them are the test's own and not a
  calibration artifact.

  What more GPU time can and cannot buy is the reason this is a condition rather
  than a caution, and both rows are measured at 200 replicates x 2000 draws:

  | candidates per set | runs to buy | GPU-hours | 80% power at | power at 0.10 nats |
  |---|---|---|---|---|
  | 13 (the curated set's size) | 26 | 1.20 | **0.30 nats** | 0.23 |
  | 26 | 52 | 2.39 | **0.155 nats** | 0.41 |
  | 52 | 104 | 4.78 | **0.11 nats** | 0.69 |

  **Seeds are not the lever.** The same design at 2, 3 and 4 shared seeds gives
  0.23 / 0.21 / 0.22 power at a 0.10-nat advantage and 0.64 / 0.65 / 0.66 at 0.20,
  i.e. identical to within the table's resolution, because the uncertainty is
  *which* candidates each set drew and a candidate is on the same side on every
  seed — the member-level split is drawn once, so no number of seeds averages it
  away. Candidates per set is the lever that works, and the curated arm holds 13,
  which is the size that needs a 0.30-nat effect. The coverage form of the
  hypothesis — curation finds a better *winner*, the mechanism §9a itself
  predicted — is invisible at every size measured, across a winner made better by
  up to 0.60 nats: a permutation null on the sets' floors already contains that
  winner, since half its assignments put the member on either side, so the floor
  difference keeps its magnitude whatever the labelling. Measured through the
  shipped test on the phase's own population, a 0.60-nat better winner — twice the
  effect the mean form is priced at, and four fifths of the entire 0.74-nat
  candidate spread — leaves the `best` statistic at **0.03 power** at 13 candidates
  per set, **0.02** at 26 and **0.04** at 52, every one of them at or below the
  nominal 0.05 rate, while the *same statistic* sees a 0.30-nat set-wide shift at
  **0.94**. Doubling the better winner's advantage from 0.30 to 0.60 leaves the rate
  where it was, **0.03 at both**, so the coverage form is not merely underpriced at
  this size: the test is *insensitive* to it, which is the structural argument above
  read off the shipped test rather than asserted. The best statistic is therefore not
  a test that cannot see anything — it is the one that sees a set-wide shift — it is
  specifically blind to a single improved member, and the lever that buys the mean
  form does not buy the coverage form at all. So the statistic §11 is pre-registered
  on is not a preference between two measurable questions: it is the only one of the
  two the affordable design can see. What §11 can measure at this
  scale is the set-mean form, and that is not affordable at the curated set's size.
`RandomArchitectureSearch` is implemented and tested against the same grid the
curated sets draw from, so a *band-matched* draw is one command; a
*budget-matched* one is the condition above, and it is checkable with no GPU. What
was missing was the typicality check, and that is now done.

**The comparison's decision machinery is now declared too, before any of its data
exist.** `SET_COMPARISON_RULE` fixes the thresholds (set-mean p < 0.05; budget
parity within 2.5%; 2+ shared seeds) and `set_comparison_decision` reads the
branch mechanically — manual better, random better, or no detectable difference,
with preconditions that *refuse* a verdict over an unequal budget or an incomplete
block rather than compute one. The set mean is the pre-registered primary because
§9a measured the winner and coverage forms to be invisible at every size this
phase can afford; `best` and `worst` are computed and reported as secondaries with
that caveat attached. `scripts/set_comparison_analyze.py` applies the rule to two
real experiment directories (budget ratio computed from the two zoo definitions,
per-set structural summaries rendered beside the verdict), and every branch is
exercised on synthetic planted-effect designs by `tests/test_set_comparison_decision.py`
— the same rehearsal-before-the-hours standard the head-paired confirmation arm
held. When the comparison's hours are bought, its verdict is a command, not an
afternoon of ad-hoc statistics.

**Its reading check is declared with it rather than added later.** This branch is a
threshold on a loss like every other pre-registered rule in the phase, and by the time
§11 runs the other four will have shown what that costs: two of the four move when the
same rows are read on a different loss, and one of those is §2c's own attribution. So
`set_comparison_decision` re-reads itself over the same alternate losses through the
shared path the other rules use — held-out, final training loss, and the trajectory
average — and its record renders the same table. Wiring a check in *after* a result is
how a declared threshold becomes a chosen one, which is the thing §11's pre-registration
exists to prevent; the check costs the arm one extra two-set test per alternate and is
part of the declaration above, not a response to anything.

## 10. Session cost

| arm | runs | GPU-hours |
|---|---|---|
| v1 sweep | 47 | 3.60 |
| mixer benchmark | 5 measured of 9 patterns (4 OOM) | ~0.04 (step time; setup unrecorded) |
| v2 flat, 1200 steps | 24 | 3.20 |
| horizon arm, 400 + 800 steps | 48 | 3.25 |
| crossed arm v3, 1200 steps | 24 | 3.28 |
| crossed confirmation, fresh seeds | 24 | 3.26 |
| head-paired v4, 1200 steps | 24 | 3.499 |
| head-paired confirmation, fresh seeds | 24 | 3.511 |
| **total** | **220** | **23.632** — against a 10–20 GPU-hour cap, the overrun declared before the last arm ran |

Two things about this table are worth stating because they are easy to mis-sum. The
mixer row is a set of 12-step throughput benchmarks rather than training runs, and it
is a partial set: 4 of the 9 patterns failed with `out_of_memory` and are recorded as
such, so the count is 5 measured out of 9 attempted — an earlier version of this
table took the 9 as measured and then summed its rows to a total that matched neither. And an arm's `search_cost.json` is not
necessarily that arm's own when an analysis pooled two arms with `--also`: the pooled
write lands in the primary arm's directory, so the crossed arm's file held 48 runs /
6.53 h while its own ledgers hold 24 / 3.28. Those files now carry
`pooled_experiments` and a `by_experiment` split (3.28 + 3.26 for that pair), which is
what makes this table checkable against the artifacts instead of trusted.

These are the token-derived figures, which is the right unit for comparing arms
and the wrong one for predicting when a run finishes. v4 was launched at 09:28
behind the confirmation arm on the single 8 GB card, and completed no candidate for
three hours while that sweep held it - visible only as wall clock, since the cost
record counts tokens and its budget guard counts tokens too. So a sweep now warns
at launch when another one already owns the card, and the per-run pace is read off
a finished candidate rather than assumed.

That judgement is computable before the hours are spent: a sweep prints its plan at
launch (runs, tokens, projected GPU-hours) and names the stage and seed at which the
token budget guard would break, which is its effective size when the cap binds;
`--dry-run` prints the same with no GPU. The projection uses a deliberately
conservative rate — 4,500 tok/s against the 4,900-6,100 measured on this card — so a
completed ledger's own tok/s rescales it to an estimate of wall clock, and the two
units are reported side by side because the guard counts tokens and cannot see
hours at all (the shared-card episode above).

The cap binds now, and the phase has closed past it deliberately rather than by
drift. v4 was declared the last arm that fits, on a projection of 3.3 GPU-hours,
and it cost 3.499 — so completed work lands at **20.12 against the 10–20 cap**, at
its ceiling rather than inside it. The one arm run after it was the head
interaction's confirmation, at 3.511 GPU-hours, which took the phase to **23.632, an
overrun of ~3.6 declared in that arm's pre-registration before its first run** —
because the interaction was undecidable from data already paid for (a 0.25 floor
against a 0.05 threshold, §2c) and carrying it into the search program as an unknown
owed either a free lever or a wasted axis. It decided the lever (§2c), and the phase
is closed at that figure: the horizon arm that followed ran under its own
pre-registered budget on Kaggle hardware, not against this cap.

**The horizon phase, on its own cap.** `arch-zoo-100m-v3-crossed-horizon10x` ran
after this phase closed, under a **separate 15 GPU-hour cap pre-registered in its own
NOTES** so that it could not extend the overrun by drift:

| arm | runs | GPU-hours | cap |
|---|---|---|---|
| horizon arm, crossed factorial at 12,000 steps × 2 seeds | 32 | 9.22 | 15 (its own) |

At 4.83 h of wall clock on Kaggle T4×2 — two sweeps in parallel, one seed each — it is
the largest single arm here and the only one on remote hardware. It is deliberately
**not** summed into the 23.632 above: that is the closed phase's own total against its
own 10–20 cap, and adding them would invent a figure neither declaration authorised.
The 9.22 is token-derived (the sum of both sweeps' per-run times) against 4.83 h of
one two-card session, which is the same 2× the rest of this table's unit carries.

The rest is a budget decision rather than an analysis one, and the phase's own
numbers price it: the §11 comparison needs *both* arms flat and budget-matched —
v1's manual numbers came off a promotion schedule, so the random arm alone is 39 runs
at 1200 steps ≈ 5.2 GPU-hours before anything is re-run (§9a) — and the cheapest
lever available is screening at 400 steps (1.1 GPU-hours against 3.2 for the same
ordering at Spearman 0.95 *and* the same joint structural fit, §3).

Concretely, at the measured per-run costs (~0.046 GPU-hours for 400 steps, ~0.133
for 1200) a `screen→confirm` search over 100 candidates × 3 seeds costs
100×3×0.046 + 8×3×0.133 ≈ **17 GPU-hours** against **40** for running all 100 at
full length — a 2.4× saving, and the screening half of it buys the same
attribution §2c needed a designed factorial to get.

**That 17 GPU-hours confirms the finalists at 1,200 steps — the length this phase
measured at — while §9 specifies the horizon the search actually trains for. The
horizon arm prices that step, at both lengths and on one design:**

| step | runs measured | h/run | 24 runs (8 finalists × 3 seeds) |
|---|---|---|---|
| 1,200 steps, Kaggle T4 fp16 | 16 | 0.0523 | 1.26 GPU-h |
| 12,000 steps, Kaggle T4 fp16 | 16 | **0.524** | **12.57 GPU-h** |
| 12,000 steps, local card fp32 (extrapolated ×10 from 0.133) | — | 1.33 | 31.92 GPU-h |

The two Kaggle rows are both *measured*, and cost is linear in steps between them —
10.016× the time for 10× the steps — which is what licenses the local extrapolation
rather than merely assuming it; a sublinear rate would have made the confirmation
cheaper than this. What the confirmation's *length* decides is a substitution rather
than an addition, because the loop screens and then confirms: **17.0 GPU-h with the
confirmation at 1,200** (13.80 screening + 3.19), **26.4 GPU-h with it at the
12,000-step horizon on Kaggle** (13.80 + 12.57), and **45.7 GPU-h with it on this
card** (13.80 + 31.92). Each horizon total is the screening half plus the
confirmation *alone*; an earlier version of this paragraph quoted 29.6 and ~49,
which added the horizon confirmation to the complete 1,200-step loop and so paid
twice for confirming the same 24 finalists. The error is recorded rather than
silently corrected because the loop's total is a budget number, and it is the one
figure in this section that a plan would be sized against.

What the substitution changes is which half dominates. At 1,200 steps the
confirmation is 3.19 against 13.80 of screening, so the loop is a screening loop
with a confirmation attached; at the horizon it is 12.57 (91% of the screening half)
on Kaggle and 31.92 (2.3×) on this card, so the confirmation sets the price. That is
also the strongest argument for the horizon phase having run on Kaggle: the same 24
confirmations cost **2.54× less per run** there, and it is the step where the
platform choice matters most.

It is the same reason the baseline in the sentence above is quoted at 1,200 and not
at the horizon. "All 100 at full length" is 100×3×0.133 ≈ 40 GPU-h at 1,200, so the
two figures are like for like at the length the phase measured. At a 12,000-step
horizon that baseline is ~399 GPU-h, and the loop's saving is *larger* in ratio while
costing far more in absolute terms — both of which a budget has to see, and neither
of which the single "~17 GPU-hours" figure showed. Nothing here is a defect in that
figure; it is a step the phase had never run, and therefore never priced. The one thing it must not be
asked to do is settle a near tie: promotion needs the margin to clear the seed
noise, which at 400 steps is 0.0286 nats of per-architecture std and therefore a
0.0286-nat margin for a two-seed screen (§4). Screening eliminates; it does not
adjudicate.

Every arm's projection landed within 10% of its cost (horizon 3.6 → 3.25, crossed
3.2 → 3.28, head-paired 3.3 → 3.499, the largest miss at 6%), and the driver's own
budget guard agreed with all of them — the s/step model in §4a is calibrated well
enough to plan with.

Information per GPU-hour, measured rather than assumed: a 400-step screen ranks
the matched set at Spearman 0.95 against 1200 steps **and** fits the same
**pre-specified** joint model (R² 0.927 vs 0.953, and `hidden_size` individually
significant at both: p = 0.0145 vs 0.00035) for **1.1 GPU-hours** against 3.2,
which makes screening the cheapest decision-relevant measurement in the phase —
provided its spread is never quoted as the effect size (§4).

## 11. Standing caveats

- **Selection coupling.** v1's finalists were promoted using shorter runs on the
  same seeds, so a test of those finalists is mildly anti-conservative.
  `decision.md` says so; v2 avoids it by construction and fresh seeds would remove
  it entirely.
- **The GDN mixer is not Qwen3-Next's block** (shared key/value dimension, no
  short convolution) and nothing here speaks to the 3:1 schedule's optimality. It
  is benchmarked as one layout among nine, next to its inverse as a control.
- **Losses here are ~1.6–4.2 nats**, far from convergence. At 1200 steps (2.4M
  tokens) these models are heavily undertrained; the horizon arm trains 12,000 steps
  (24.6M tokens) and reaches 1.62–1.72, which is the closest this project has come to
a trained model and is still an order of magnitude from converged. The decay in §3
  is the clearest evidence that short-horizon comparisons measure early optimisation
  dynamics as much as architecture — but note that the decay is in *variance*, so
  what is horizon-limited is the effect *size*, and the *ordering* is only stable in
  rank rather than in which candidate wins (§3).
- **The attribution is one trade, not two independent levers.** Expert width is
  the shock absorber for any hidden-size change at fixed depth, so §2c's
  "residual-stream width matters" and "expert capacity does not" are two readings
  of a single budget trade. Expert count and expert width are not separable in
  principle at all (`E x W` enters the parameter count as a product).
- **"Hidden size" is residual-stream width; the head-count reading is excluded.**
  This family fixes `heads = hidden/64` in *every* design, including the crossed
  factorial that produced the attribution, so the confound was real for the whole
  phase — the head-paired arm is what closed it, at identical within-pair cost, and
  it put head count at −0.0177 nats (p = 0.14) against a width effect of −0.1562 nats
  at the factorial's floor (§2c). The recommendation does not reverse. What remains
  is a clause on the *head* axis rather than a caveat on the headline: the
  confirmation arm decided the level dependence at six seeds (−0.0553 nats at hidden
  896, 12/12 pairs; +0.0084 at 512; interaction at the 2/2⁶ floor), so heads are a
  free lever *only* on a wide residual stream, and the level split it rests on was
  pre-registered in that arm rather than chosen after the fact.
- **The interaction was resolved by adding seeds, and that is the only thing that
  resolved it.** +0.0701 nats at p = 0.24 on three seeds became +0.0827 with the
  seed-blocked p at its six-seed floor. The lesson worth keeping is about
  *floors* rather than about this gene: three seeds floored the calibrated test at
  0.25 and five at 0.0625, so no analysis of those arms could have reported the
  interaction however consistent the contrasts were — the same reading error the
  horizon arm made on a floor-bound statistic. A pre-registered rule whose
  threshold is below the test's floor is not decidable by more analysis, only by
  more data.
- **The decay rate is measured at two horizons, and they disagree in the safe
  direction.** 1.65× per doubling fits 400→1200; the measured rate over 1200→12,000
  is **1.38×**, so extending §3's rate an order of magnitude **over-predicts** the
  decay. Read it as an upper bound on how fast effects fade, not a law: two points
  fix a rate and not a curve, so the shape past 12,000 steps is still unmeasured.
  What the 10× check did settle is the ordering: rank correlation survives (Spearman
  0.90) while the **winner changes**, so a screen promotes a subset and never names
  the winner across a horizon change.
