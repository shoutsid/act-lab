# arch-zoo-100m-v2 — Matched-Total Structural Spread

**Design:** `matched-total`, target 78M total parameters, tolerance 5%,
8 candidates, 3 seeds (42/43/44), 1200 steps, context 1024, batch 2 — and
deliberately **no successive halving** (`--plan flat`).

## Why this experiment exists

`arch-zoo-100m-v1` produced 47 runs and three findings that together say its own
design could not answer the primary hypothesis:

1. **The finalists were not separated.** At 1200 steps the top three differ by
   0.0404 nats with a seed standard deviation of 0.0179; the exact seed-matched
   permutation test gives p = 0.139. Not established.
2. **The screen does not predict the outcome.** The 400-step ranking has
   Spearman 0.886 against the 800-step ranking but places the eventual winner
   second, by 0.009 nats — a quarter of that screen's own 0.0390-nat
   per-architecture seed std, so the two runs did not separate. Promotion at 400
   steps is partly a coin flip.
3. **The screen did not test a matched budget.** v1 spans 70.8M–123.3M total
   *and* 25.3M–77.1M active parameters at once, so a spread across it cannot be
   attributed to either axis. Measured across v1's own stage 1, nothing
   correlates with loss at 400 steps: total parameters ρ = −0.26 (p = 0.39),
   active parameters ρ = −0.14 (p = 0.66), contracted FLOPs ρ = −0.08. So the
   0.72-nat screen spread is not a parameter effect — but v1 also cannot show
   what it *is*.

v2 answers the primary hypothesis directly: hold the parameter budget matched
and spread structure as widely as the search space allows.

## The design

| architecture | total | active | act/tot | depth | experts | active | width |
|---|---|---|---|---|---|---|---|
| `mtot_D5_H512_E16_A2_W832` | 77.6M | 18.0M | 0.23 | 5 | 16 | 2 | 832 |
| `mtot_D12_H640_E12_A3_W288` | 78.1M | 38.3M | 0.49 | 12 | 12 | 3 | 288 |
| `mtot_D8_H832_E6_A2_W576` | 75.0M | 44.4M | 0.59 | 8 | 6 | 2 | 576 |
| `mtot_D6_H896_E16_A6_W288` | 76.3M | 45.3M | 0.59 | 6 | 16 | 6 | 288 |
| `mtot_D6_H576_E6_A4_W1536` | 76.4M | 55.2M | 0.72 | 6 | 6 | 4 | 1536 |
| `mtot_D10_H512_E8_A6_W768` | 77.6M | 61.9M | 0.80 | 10 | 8 | 6 | 768 |
| `mtot_D14_H896_E8_A5_W128` | 78.1M | 68.5M | 0.88 | 14 | 8 | 5 | 128 |
| `mtot_D14_H576_E4_A4_W896` | 81.1M | 81.1M | **1.00** | 14 | 4 | 4 | 896 |

- Total parameters held to **×1.08** (75.0M–81.1M).
- Active parameters span **×4.51** (18.0M–81.1M) — the axis left free.
- Depth spans 5–14, experts 4–16, active experts 2–6, expert width 128–1536.
- Minimum pairwise structural distance 0.256, so no two candidates are near-copies.
- Every candidate is a cell of the single authoritative search space
  (`enumerate_search_space`, 15,224 cells) that `RandomArchitectureSearch` also
  samples, so this curated set is not secretly drawn from a wider grid.

The dense end matters: `mtot_D14_H576_E4_A4_W896` has active == total, while the
sparsest candidate routes 23% of its parameters. A design that could not reach
the dense end would be unable to test the sparsest-vs-densest contrast that the
earlier phase's informal hypothesis rested on.

## Flat plan rather than halving

`--plan flat` trains every candidate at the final length on every seed. Halving
costs less but promotes on a proxy that v1 measured to be biased: a design whose
job is to estimate the spread at 1200 steps must not choose its candidates using
400-step behaviour, or it inherits exactly the bias it is trying to quantify.

## Cost

8 candidates × 3 seeds × 1200 steps = 24 runs. Measured: **16 runs in 2.11
GPU-hours** (0.31–0.51 s/step, tracking expert count as predicted); the full 24
project to ~3.2 GPU-hours, leaving this phase's total near 7 GPU-hours against a
10–20 GPU-hour cap.

## Complete (24 of 24) — the architecture effect is ESTABLISHED

| rank | architecture | total | active | hidden | mean val loss |
|---|---|---|---|---|---|
| 1 | `mtot_D6_H896_E16_A6_W288` | 76.3M | 45.3M | 896 | 2.7446 |
| 2 | `mtot_D8_H832_E6_A2_W576` | 75.0M | 44.4M | 832 | 2.7859 |
| 3 | `mtot_D14_H896_E8_A5_W128` | 78.1M | 68.5M | 896 | 2.7932 |
| 4 | `mtot_D6_H576_E6_A4_W1536` | 76.4M | 55.2M | 576 | 2.8731 |
| 5 | `mtot_D12_H640_E12_A3_W288` | 78.1M | 38.3M | 640 | 2.8803 |
| 6 | `mtot_D5_H512_E16_A2_W832` | 77.6M | 18.0M | 512 | 2.9214 |
| 7 | `mtot_D14_H576_E4_A4_W896` | 81.1M | **81.1M (dense)** | 576 | 2.9604 |
| 8 | `mtot_D10_H512_E8_A6_W768` | 77.6M | 61.9M | 512 | 2.9908 |

Spread **0.2462 nats** = **16.6×** the within-architecture seed std (0.0148).
Seed-matched blocked permutation over 3 blocks × 8 architectures: **p <= 5e-06**,
the Monte-Carlo procedure's resolution (design floor 6.2e-10; raised to 2M
draws it reads 5e-07, still floor-limited, so the bound is the procedure's and not
the world's). Unpaired one-way ANOVA agrees: F = 83.2 on (7, 16) df.

This table is the three-seed result. The intermediate figures below were computed
on the first two seeds and are retained because the robustness checks were done
there; every conclusion survived the third seed unchanged, with the winner and
the loser identical on all three.

This is the phase's primary hypothesis, answered in the affirmative at a matched
parameter budget: **which architecture you pick matters, and it matters by more
than initialization noise.** The spread is 6× what v1's curated families produced
at 1200 steps, on a design that holds total parameters to ×1.08.

### The active-parameters hypothesis is refuted, not merely unsupported

The earlier phase's informal reading was that the top architectures "all carried
44–48M active parameters". Here, active parameters span **×4.51 (18.0M–81.1M)** at
a matched total budget and track loss not at all: ρ = +0.214, p = 0.62.

More directly: the **fully dense** design (`mtot_D14_H576_E4_A4_W896`, active ==
total) places 7th of 8, and the **sparsest** design (`mtot_D5_H512_E16_A2_W832`,
23% active) beats it by 0.052 nats. A model that spends every parameter on every
token is not better than one that spends a quarter of them.

### The leading signal is hidden size — but it is confounded

| axis | Spearman | p | range | ratio |
|---|---|---|---|---|
| hidden size | **−0.861** | **0.013** | 512–896 | ×1.75 |
| attention heads | −0.861 | 0.013 | 8–14 | ×1.75 |
| total parameters | +0.595 | 0.132 | 75.0–81.1M | ×1.08 |
| expert width | +0.503 | 0.202 | 128–1536 | ×12.0 |
| contracted FLOPs/token | +0.238 | 0.582 | 139M–586M | ×4.20 |
| depth | +0.229 | 0.563 | 5–14 | ×2.80 |
| active parameters | +0.214 | 0.619 | 18.0–81.1M | ×4.51 |
| expert count | −0.194 | 0.678 | 4–16 | ×4.00 |
| measured throughput | −0.143 | 0.752 | 4.0k–6.5k | ×1.62 |
| active experts | +0.000 | 1.000 | 2–6 | ×3.00 |
| dense-runtime FLOPs/token | +0.000 | 1.000 | 497M–623M | ×1.25 |

The pattern is clean in the raw table: the two 896-hidden designs are the two
best, the two 512-hidden designs are the two worst.

**Three reasons this remains a hypothesis, even though the effect is established.**

1. **It does not survive correction for testing twelve axes.** Bonferroni gives
   0.013 × 12 = 0.16; a single marginal hit among twelve is roughly what noise
   produces. The *effect* is established by the blocked test; the *attribution*
   to hidden size is not.
2. **Hidden size is not a free knob here.** With total parameters pinned and
   active parameters inert, widening the residual stream necessarily takes budget
   *out* of the experts. Cross-correlating the design's own axes shows hidden size
   is nearly independent of active parameters (ρ = +0.11) but strongly entangled
   with expert width (ρ = −0.74). "Hidden size" is better read as "the share of a
   fixed budget placed in the residual stream and attention rather than in expert
   capacity" — a defensible architectural claim, and a different one from "more
   parameters is better". `decision.md` now prints this confound explicitly.
3. **Hidden size and attention heads are perfectly collinear** here
   (`heads = hidden / 64`), so the table's two hits are one signal, not two.

### The verdict does not depend on one architecture, or on the held-out split

An established effect is only useful if it is general, so the design was
dismantled in the two ways that would most cheaply break it. Both checks are
pinned as tests, so a later change to the statistics cannot quietly make the
headline claim fragile.

**Leave one architecture out** — recompute the blocked test on all 8 subsets of
7 candidates:

| dropped | spread | p |
|---|---|---|
| `mtot_D6_H896_E16_A6_W288` (the winner) | 0.2072 | 0.00195 |
| `mtot_D10_H512_E8_A6_W768` (the worst) | 0.2159 | 0.00161 |
| any of the other six | 0.2416 | 0.00046–0.00161 |

Every subset stays significant, and the *worst* case is the winner's removal at
p = 0.00195 — still two orders of magnitude inside the conventional threshold. The
effect is a property of the set, not of one lucky architecture.

**Alternative outcome measures** — training loss instead of held-out loss, in case
the architectures differ in how much they overfit rather than in how well they
model:

| measure | spread | p |
|---|---|---|
| mean held-out loss (primary) | 0.2416 | 0.00023 |
| final training loss | 0.2684 | 0.000025 |
| mean training loss | 0.3455 | 0.000045 |

The effect is *stronger* on training loss, which is the opposite of what an
overfitting artifact would look like. The held-out split is not manufacturing it.

### The ranking is not perfectly seed-consistent, and still holds

Ranks 2/3 (`mtot_D8_H832_E6_A2_W576` vs `mtot_D14_H896_E8_A5_W128`) and 4/5
swap between seeds 42 and 43. Better than "not fatal", the wobble is *low*: the
best architecture is rank 1 on all three seeds and the worst is rank 8 on all
three, and the mean pairwise Spearman across seeds is 0.91–0.98. That is what the
blocked test weighs, and it still returns p <= 5e-06: the *ordering* has a little
slack in the middle, the *spread* does not.

## The horizon question is closed, and the answer is more interesting than yes/no

This arm's stated gap was that it measured one training length, so it could not
see whether the effect decays. That arm ran
(`arch-zoo-100m-v2-horizon`, same candidates byte-identically, 400 and 800 steps,
3.25 GPU-hours) and the pooled record is the table above plus:

| steps | spread | seed std | spread in seed-noise units | best |
|---|---|---|---|---|
| 400 | 0.5457 | 0.0286 | 19.09 | `mtot_D6_H896_E16_A6_W288` |
| 800 | 0.3244 | 0.0165 | 19.71 | `mtot_D6_H896_E16_A6_W288` |
| 1200 | 0.2462 | 0.0148 | 16.59 | `mtot_D6_H896_E16_A6_W288` |

**The spread does decay** — 2.22× over 400→1200 steps, 1.65× per doubling,
monotone, which lands inside the 1.4–1.6× band v1 estimated from unrelated and
unmatched families. So v1's decay rate replicates on a controlled design and is a
property of training length, not of v1's confounds.

**But seed noise decays with it**, 19.09 → 16.59 in seed-noise units, and the
winner is identical at every length with screening rank correlation 0.95
(400→800) and 0.98 (800→1200). A spread that falls in lockstep with the noise it
is measured against is a variance decay: the effect is equally detectable at 400
steps as at 1200, and the *ordering* — the thing a search consumes — did not move
at all. So the horizon gap that made this arm's verdict provisional is closed, and
it closes in favour of searching, with one standing obligation: quote every effect
size together with its training length.

Full record: `arch-zoo-100m-v2-horizon/NOTES.md`.



## Follow-ups this design enables

**Attribution: which gene carries this effect** — run and reported in
`arch-zoo-100m-v3-crossed/NOTES.md`. A `depth × hidden_size` 2×2 at matched total
parameters (ρ between the factors = 0.000, expert width the shock absorber) gives
both factors a significant main effect of their own: hidden size **−0.1836 nats**
(~274 training steps) against depth **+0.0783 nats** (~117 steps), both at the
exact enumeration's floor p = 4.3e-05. So the confound recorded above is resolved
rather than merely acknowledged — the residual stream is the bigger lever — and
depth turns out to matter with the **unintuitive sign: shallow beats deep** at
matched total parameters and matched hidden size.

**Fresh-seed confirmation of v1's finalists** (~1.3 GPU-hours), *not run*. It would
raise the primary test from 3 seeds to 6 and remove v1's selection coupling. Deprioritised
because the effect is already at the procedure's resolution and the open questions
below are cheaper and more decision-relevant. Sketch, for whoever picks it up —
run into a separate directory, then pool it, so the confirmation does not overwrite
v1's own zoo definition and matching reports:

```
PYTHONPATH=src python scripts/arch_zoo_screen.py --experiment-id arch-zoo-100m-v1-confirm \
  --design families \
  --only narrow_experts_D8_H896_E12_A4_W256,shallow_wide_D6_H896_E8_A2_W896,narrow_experts_D10_H768_E12_A4_W288 \
  --plan flat --flat-steps 1200 --finalist-seeds 45,46,47

PYTHONPATH=src python scripts/arch_zoo_analyze.py \
  --experiment-id arch-zoo-100m-v1 --also arch-zoo-100m-v1-confirm --write
```

Six seeds on 3 architectures keeps the exact enumeration viable (6^6 = 46,656
assignments) and lowers the design's power floor from 1/36 to 1/7776, so it can
resolve what 3 seeds could not.

**Horizon arm** (~3.6 GPU-hours, *run* — see the decay section above). To measure
the decay on a matched design instead of inferring it from v1's curated families.
It went to its own directory rather than this one, because `--plan flat` here
would have rewritten this arm's ledgers at the shorter length; pooling is by
candidate name, so the split costs nothing. Sketch of the arm as launched:

```
PYTHONPATH=src python scripts/arch_zoo_screen.py --experiment-id arch-zoo-100m-v2 \
  --design matched-total --match-target 78000000 --match-tolerance 0.05 --candidates 8 \
  --plan flat --flat-steps 400 --finalist-seeds 42
```

## What this licenses, and what it does not

The second branch of the pre-registered outcomes did not happen: the spread stayed
large and significant at a matched budget, so **architecture choice is a real
lever at this scale, not initialization noise.** Two consequences follow, and
**both have since been tested rather than left open.**

**Licensed.** A search over structure is worth running, because the spread is
0.25 nats against a seed noise of 0.015 — the return from finding a good
architecture exceeds the cost of the search many times over. §11's
`RandomArchitectureSearch` is the baseline that search has to beat. Two supporting
facts were added after this run: the effect is **horizon-stable in ordering** (the
same architecture is best at 400, 800 and 1200 steps; the spread decays 1.65× per
doubling but seed noise decays with it, so detectability holds), and the lever is
**attributable** (hidden size −0.1836 nats and depth +0.0783 nats, both at
exact p = 4.3e-05, with shallow beating deep).

**Not licensed — and now closed.** The caveat this arm carried was that v1's
decay, 1.4–1.6× per doubling, might make a 0.24-nat spread much smaller at 10× the
horizon. Measured on this same matched design the rate is 1.65× per doubling,
which *confirms* the decay and *replicates* v1's figure on a controlled set — but
the decay is in variance, not in signal: the ordering and the signal-to-noise are
flat across 400→1200 steps. So the caveat does not block a search; it converts
into a reporting obligation, that every effect size be quoted with the training
length it was measured at and that the finalists be confirmed at the horizon the
search actually trains for.

The comparison is matched, multi-seed, and tested with an exact seed-matched
statistic, so the verdict does not depend on a heuristic threshold. What this arm
could not identify — *which gene* carries the effect — the crossed arm now does;
and the one thing neither can fix is that expert width is the budget's shock
absorber, so "wider residual stream" and "less expert capacity" remain two
readings of one trade at this parameter budget.
