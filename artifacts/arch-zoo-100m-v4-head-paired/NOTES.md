# arch-zoo-100m-v4-head-paired — Is the Lever Width, or Head Count?

**Design:** `head-paired`, depth **fixed at 10**, `hidden_size` 512 × 896,
`attention_heads` 8 × 16, 4 base configurations (8 candidates), total parameters
matched to 78M ±5%, plan `flat` at 1200 steps, seeds 48/49/50, context 1024,
batch 2.

## The confound this exists to break

`arch-zoo-100m-v3-crossed` attributed the phase's architecture effect to
**hidden size: −0.1836 nats** (≈274 training steps), with depth second at
+0.0783. That attribution carries a confound the project had not noticed, and it
is not escaped by crossing depth against hidden size, because the model family
fixes attention geometry:

```
heads = _heads_for(hidden_size) = hidden_size / 64
```

Measured on the crossed arm's own ledgers:

| hidden_size | attention_heads | runs |
|---|---|---|
| 512 | 8 | 12 |
| 896 | 14 | 12 |

One head count per level, deterministically. So "−0.1836 nats for hidden size" is
equally "−0.1836 nats for going from 8 to 14 attention heads", and nothing in
this phase separates them.

**Why that matters enough to spend hours on:** head count is *free*.

| hidden_size | heads | total params |
|---|---|---|
| 896 | 4 | 91,822,976 |
| 896 | 8 | 91,822,976 |
| 896 | 14 | 91,822,976 |
| 896 | 16 | 91,822,976 |
| 896 | 28 | 91,822,976 |

Attention parameters are ``4 * hidden^2`` however that width is divided, so at a
fixed hidden size the number of heads costs nothing at all. The two readings of
the attribution therefore imply opposite advice:

- **Residual-stream width** → spend budget on it. The phase's stated
  recommendation.
- **Head count** → buy it for free, and the recommendation is wrong in a way that
  leaves performance on the table.

They are distinguishable only by a design that varies head count at a held-fixed
hidden size. That is what this arm is.

## The design, and why no band is involved

Every previous design matched candidates by landing them in a parameter band,
which leaves 5–10% of spread in the matched axis. Head count does not need a band,
because it costs nothing: at a fixed `(depth, hidden, experts, width)` two head
counts are **identical in both total and active parameters**. So the candidates
come in pairs, and the pair is its own control.

| pair | base configuration | heads | total params |
|---|---|---|---|
| 1 | `D10_H512_E4_A2_W1536` | 8, 16 | 77,625,856 |
| 2 | `D10_H512_E12_A6_W512` | 8, 16 | 77,666,816 |
| 3 | `D10_H896_E4_A3_W512` | 8, 16 | 76,207,488 |
| 4 | `D10_H896_E16_A6_W128` | 8, 16 | 76,315,008 |

| property | value | comparison |
|---|---|---|
| depth | fixed at 10 | the crossed arm could not control it |
| rho(hidden_size, attention_heads) | **0.000** (balanced 4 × 4 each) | previously the same knob |
| total parameters, within a pair | **identical** | 5–10% spread in every earlier design |
| contracted FLOPs, within a pair | **identical** | — |
| dense-runtime FLOPs, within a pair | **identical** | — |
| total parameters, across pairs | ×1.019 | v2 ×1.08, v3 ×1.05 |
| min pairwise structural distance | 0.050 | 0.132 in v3 |

"Free" is checked across all three cost conventions the project records, not
assumed from the parameter count. Within each pair the two head counts are
bit-identical on total parameters, contracted FLOPs/token and dense-runtime
FLOPs/token, because the cost model's only head-dependent term is absent: forward
cost is ``2 * params + 4 * depth * seq * hidden``, and head count appears nowhere
in it. The one quantity that could still differ is *measured wall-clock
throughput* (more heads means more kernel launches), which every ledger row records
as ``measured_tokens_per_s``; the analysis is pre-registered to report it rather
than assume it away.

Two things that look like flaws and are not:

- **Min pairwise distance is low (0.050)** because the two members of a pair are
  deliberately near-identical — they differ only in head count. The project uses
  min distance as a design-quality signal, and here it is measuring the design
  working as intended. Distinctness is bought *across* pairs, not within them.
- **`rho = 0.000` is exact, not approximate**, because both factors are balanced
  2 × 2 with 2 replicates; a balanced factorial is orthogonal by construction.

## Pre-registered decision rule

Written before the numbers exist. The head contrast is analysed as a *paired*
design (`paired_contrast_test`): per (pair, seed) it takes
`loss(16 heads) − loss(8 heads)`, giving 4 pairs × 3 seeds = **12 paired
differences** and an exact floor of **2/2¹² = 4.9e-04**. Collapsing to arm means
instead would floor at 2/2³ = 0.25 — the same floor that left the crossed
interaction unresolved — so pairing is the point, not a convenience.

| outcome | condition | reading |
|---|---|---|
| **head count is the lever** | head contrast significant with \|effect\| ≥ 0.10 nats and the width contrast collapsing toward 0 on the pooled 3-seed set | §2c's attribution is **corrected**: the lever is a free gene, and the search recommendation changes from "spend budget on the residual stream" to "add heads, which costs nothing". This is the outcome with the largest consequence |
| **width is the lever** | head contrast non-significant (\|effect\| < 0.05) while the width contrast holds near −0.18 | §2c stands as written and the confound is resolved *in its favour*. The caveat is then closed rather than merely recorded |
| **both contribute** | both contrasts significant, additive or interacting | report the decomposition; the search should tune both, and the head axis is preferred where budget is tight |
| **neither replicates** | both contrasts non-significant | surprising and consequential: the crossed attribution would be called into question, since this design has the strongest matching in the project. Investigate before trusting either arm |
| **head count matters *negatively*** | significant positive head effect (more heads worse) | plausible if head width shrinks past a useful size (16 heads is 32-wide at 512, 56-wide at 896); report as a width-per-head effect, not a count effect |

Which conditions are decidable, stated here so the reading is not mistaken for
something the record computes. Two of the five are numeric and are reported by the
analysis directly: the \|effect\| ≥ 0.10 and \|effect\| < 0.05 head thresholds, and
both contrasts' p-values against their floors. The rest are judgements the
decision record cannot encode — "collapsing toward 0", "holds near −0.18", and
whether the two contrasts are additive or interacting — and they are deliberately
*left* as judgements rather than assigned cutoffs here: inventing a number now
would look like pre-registration while being nothing of the sort. So this arm's
branch is read by a person from the printed contrasts and floors, and that is worth
knowing before the numbers arrive rather than after. A later arm whose branches are
all numeric can be evaluated mechanically, as the crossed confirmation arm's rule
is (`--replication-rule`).

Two secondary readings, reported whatever the primary says:

- **Per-pair effects**, so a single pair driving the whole contrast is visible
  rather than averaged away.
- **The width contrast on this design**, as a replication of §2c at a *tighter*
  match than the original: −0.1836 nats is what the crossed arm measured at ×1.05;
  this arm measures it at ×1.019 with depth fixed.

## Cost

8 candidates × 3 seeds × 1200 steps = 24 runs, **~3.3 GPU-hours** at v3's measured
rate.

Running total for the phase: 3.60 (v1) + ~3.2 (v2) + 0.2 (mixer) + 3.25 (horizon)
+ 3.28 (v3) + ~2.6 (v3-confirm, running) + ~3.3 (this) ≈ **19.5 GPU-hours** against
the 10–20 cap. This arm is the last one that fits, which is stated here rather than
discovered when the budget is gone.

## How to analyse it

```
PYTHONPATH=src python scripts/arch_zoo_analyze.py \
  --experiment-id arch-zoo-100m-v4-head-paired \
  --also arch-zoo-100m-v4-head-paired-confirm \
  --samples 200000 --head-rule --write
```

`--also` pools this arm with `arch-zoo-100m-v4-head-paired-confirm`, which is what the
committed `decision.md` in this directory records - the confirmation re-ran the same
candidates on fresh seeds, so the two row sets pool to six seed blocks - and
`--head-rule` renders the pre-registered branch. Without both, the command reproduces
the three-seed record this arm closed on before its confirmation ran.

The analysis was wired and validated **before** the arm was launched, so the
numbers land into a path that is already tested rather than being analysed ad hoc:

- **`paired_contrasts` is detected from the ledger, not declared by the caller.**
  A gene counts as paired only when holding every other structural gene fixed
  leaves exactly two levels of it, with one pair size, and every pair complete.
  A design that merely *looks* paired because a band happened to match a pair
  yields nothing, and that is tested in both directions — the synthetic paired
  design is found, the real band-matched arms are not.
- **An unfinished seed costs resolution, not the contrast.** A group can be
  missing a level because the sweep has not reached it yet or because the pairing
  never had that member, and the row count per seed is what tells them apart: a
  seed short of the fullest one is still running, so its groups are dropped and
  named (`dropped_groups`, `seeds_dropped`) and the contrast is reported over the
  complete pairs. Anything else still yields nothing, so a malformed pairing
  cannot pass as a matched design. This was tightened after the crossed
  confirmation arm exposed the failure mode: analysing a sweep mid-flight
  previously deleted the gene's whole contrast, which reads as *"this design is
  not paired"*. The unit count, and therefore the floor printed beside the p,
  follows the drop — 2/2¹² on the complete design, higher while a seed is in
  flight — so a number read early cannot be mistaken for the finished one. A seed
  that stopped *between* pairs is the case none of that catches, because every
  group it wrote is complete: it is reported by its pair count (`seeds_partial`)
  instead, since the unit count is the only place a whole missing pair shows.
- **The paired contrast** groups the candidates on their base configuration and
  calls `paired_contrast_test`, giving 12 within-pair differences and the exact
  floor of 4.9e-04. It is printed to stdout and written into the decision, together
  with the per-pair differences, so one pair driving the result is visible rather
  than averaged away.
- **The factorial view** comes from `crossed_contrast` on
  `hidden_size x attention_heads`, which works because both factors are balanced
  with two replicates per cell — an exact enumeration over 46,656 assignments.

The two reports answer different questions from the same data, which is a feature
and a check: the factorial gives the width contrast and the interaction, the paired
contrast gives the head contrast at the tightest matching the project can achieve.
They should agree on the head effect, and a disagreement would mean the pair
structure is not what the zoo claims.**Validated end to end on the real zoo with known effects.** The check runs the
driver's `--dry-run` to emit this design's actual `zoo_definition.json`, builds a
ledger from those exact candidate genomes, plants +0.09 nats for 16 heads and
−0.10 for H896, and runs both analyses on the result. Everything recovered:

| view | recovered | true | p | floor |
|---|---|---|---|---|
| paired `attention_heads` | +0.0900 | +0.0900 | 4.88e-04 | 4.88e-04 (2/4096), **at floor** |
| factorial `hidden_size` | −0.1000 | −0.1000 | 4.29e-05 | 4.29e-05 (2/46656), **at floor** |
| factorial `attention_heads` | +0.0900 | +0.0900 | 4.29e-05 | 4.29e-05, **at floor** |
| interaction | 0.0000 | 0.0000 | — | — |

The paired test returned all 12 within-pair differences positive, so it sits on
its floor, which is the outcome the design was built for: had it needed more
resolution than 2/2¹² the arm would not be worth its hours.

One thing this confirmed rather than assumed: with depth held at 10 the factorial
view selects `hidden_size × attention_heads` — the genes actually crossed here —
not `depth × hidden_size`, because depth has no variance in this design to
contrast. The two views therefore answer the head question from different
constructions of the same data, which is the intended cross-check.

So the path is known to work before any GPU time is spent on it, and a failure
when the real numbers arrive would be a result about the design rather than about
the tooling.

## Status

**Complete: 24 of 24 runs, 3.499 GPU-hours.** The outcome against the
pre-registered rule, and the branch it landed on, are at the end of this file — the
short version is that the head contrast is −0.0177 nats (p = 0.14), the width
contrast holds at −0.1562 nats, and the arm therefore lands on "width is the lever".
At 525 s per run against a 499 s projection it is the phase's largest cost miss
(6%), which carries the phase total to 20.12 GPU-hours against the 10–20 cap. The
pre-registration above was written before any of this arm's runs existed, as was the
queued launch: the process was started at 09:28 behind
`arch-zoo-100m-v3-crossed-confirm` on the one 8 GB card, and nothing enforced the
queue. Over the next three hours it completed **no candidate at all** while
consuming about seven minutes of CPU time; its zoo reports were written at 12:31:55,
68 s after that sweep's last ledger, and its first 1200-step candidate finished at
12:40:17 having reported 499 s of training. So the first candidate began at ~12:32,
just after the card freed: those three hours are scheduling, not training, and the
arm was live and idle throughout them.

Measured from the candidates that have finished since: **499 s per 1200-step run at
4,924 tok/s**, so 24 runs (8 candidates x seeds 48-50) is ~3.3 h and the arm lands
around 16:00. That is the same per-run cost the confirmation arm's flat runs
showed, so the card has not been shared since and the numbers below should be read
as uncontended.

The stall is invisible in this arm's own record: `search_cost.json` counts GPU-hours
from tokens trained, which is the right unit for comparing arms and says nothing
about wall clock. So the wall-clock facts are recorded here, and the driver now
warns at launch when another sweep already holds the card — its budget guard counts
tokens rather than hours, so it is structurally unable to notice on its own.

No candidate of this arm ran jointly with the earlier sweep, so no row is
contaminated by contention; the ledger's per-row `measured_step_s` and
`measured_tokens_per_s` are enough to confirm that against any later run.

## Complete (24 of 24): the lever is width, and the head effect is level-dependent

Seeds 48/49/50, all 8 candidates, flat at 1200 steps, **3.499 GPU-hours** at a mean
4,718 tok/s (4,176–5,503, the spread tracking expert width and hidden size as
predicted), 447–603 s per run against the 499 s the pre-registration projected from
the first finished candidate. Contention did not creep back in: every row's
`measured_tokens_per_s` sits inside the range the confirmation arm's uncontended flat
runs showed.

| view | quantity | value |
|---|---|---|
| paired, 12 units | head effect, 16 − 8 heads | **−0.0177 nats** |
| paired, 12 units | exact sign-permutation p | 0.1396 (floor 4.9e-04) |
| paired, 12 units | sign consistency | 8 negative, 4 positive |
| factorial | `hidden_size` | −0.1562 nats, p = 4.287e-05, **at the 46,656-assignment floor** |
| factorial | `attention_heads` | −0.0177 nats, p = 0.1217 |
| factorial | interaction | +0.0571 nats |

**Branch read against the pre-registered table: "width is the lever."** It is
decided by the two conditions that table makes numeric — the head contrast is
non-significant and |effect| = 0.0177 < 0.05, while the width contrast holds. The
branch's remaining clauses ("collapsing toward 0", "holds near −0.18") were declared
person-judgements in the pre-registration rather than cutoffs, and are read that way
here rather than by inventing thresholds now: −0.0177 is 11% of −0.1562, and the
width contrast replicates at 85% of the crossed arm's −0.1836 under *tighter*
matching, on a design that holds depth fixed and is balanced 4 × 4.

Both secondary readings the pre-registration asked for whatever the primary said:

- **Per-pair effects**, so one pair cannot carry the number: −0.0334, +0.0628,
  −0.0008, +0.0233, +0.0117, +0.0015, −0.0530, −0.0501, −0.0705, −0.0336, −0.0215,
  −0.0484. The six H=512 pairs (first six) straddle zero — 2 negative, 4 positive,
  mean +0.0109 — while the six H=896 pairs are all negative, mean −0.0462.
- **The width contrast at this design's matching** (×1.019 against the crossed arm's
  ×1.05): −0.1562 nats, and negative at *both* head counts (−0.1277 at 8 heads,
  −0.1847 at 16), so the width finding is not a head-count artifact either.

### What this arm does not settle, and what it would cost

The interaction is a **lead, not a result**: +0.0571 nats with all three seed
contrasts agreeing in sign (+0.0383, +0.0730, +0.0598) and the paired-t reference at
0.030 — against a seed-blocked test sitting **at its 2/2³ = 0.25 floor**, so it is
this arm's structure rather than its data that stops it. That is exactly the
situation the crossed confirmation arm was bought for, and the arithmetic is the
same: the floor falls to 2/2⁶ = 0.031 with a second seed block, so a fresh-seed
replication of this design (24 runs, ~3.5 GPU-hours) would decide it. The phase's
budget sits at 20.12 of a 10–20 GPU-hour cap, so that is a decision rather than a
queued run.

The level split carries a second caveat, stated because it is the kind of thing
this project has been burned by: it was **chosen after seeing the interaction**, not
pre-registered, so the per-level numbers above are an observation to be confirmed
rather than the arm's answer. A subgroup picked post hoc at a floor-bound test is
how a spread that was really variance decay got read as signal in §3 of the phase
record.

## The loss reading moves this arm's head clause, and leaves the width one alone

The holdout-split check the other three arms carry was run on this arm too, and this
is the one place in the phase where it moves a decision. Width is unaffected under
every reading (−0.1562 held-out, −0.1555 on final training loss, −0.1846 on mean
training loss, each at the design's floor). The head clause is not: held-out, the head
contrast is non-significant (paired p = 0.140, factorial p = 0.122) and the
pre-registered branch is taken; read on **final training loss** the same contrast is
significant at **0.044** and the branch's non-significance condition no longer holds;
read on the trajectory average (`mean_train_loss`, which tracks how fast an
architecture descends rather than how well it fits) it is non-significant again at
0.630.

That is informative rather than fatal — an effect present on training loss and absent
on held-out loss is an effect that did not generalise, which is the reading that argues
*for* not privileging head count — but it does rule out quoting this contrast as a
null. The honest sentence is that it is **small (~0.02 nats, at most a third of the
width effect) and reading-dependent in significance**, which is why the recommendation
rests on the level split rather than on this p-value.

## The record states that reading, so it is not left to a reader to hit

The section above was written into the phase record first, which left the artifact a
reader actually quotes — this arm's pooled `decision.md` — stating the branch with no
mention of the loss it was read on. The rule now re-reads itself over the other losses
these ledgers carry and the decision prints the table: held-out and final training loss
both land on `HEAD EFFECT IS REAL AT WIDE WIDTH`, the trajectory average lands on
`INTERACTION ABSENT - THE THREE-SEED AGREEMENT WAS SEED LUCK` and is bolded as the
reading that moves. Substituting the loss is the only change — same rows, same seeds,
same declared thresholds — so it is a second reading of one design rather than a second
experiment, and the branch to quote is still the pre-registered one.

Running it costs ~3.6 s on top of the 6.2 s analysis, and it is deliberately *not* gated
on the margin looking thin: a check that silently skips itself is indistinguishable
from a branch that held, which is the failure mode this whole section is about.
