# Putting width under adaptive control — a design note, before the hours are spent

**Status: design, not a record. Nothing here is measured except where it says so.** This
note exists because the two tracks of this project have never been read against each other in
the *search space* (the report says so in §10.4), and doing so changes what the next gate
should be. It is written before any implementation because on this subject the design is most
of the work, and because the last thing to decide *after* the hours are spent is what the
hours were for.

## 1. The gap this is aimed at

The architecture-search phase measured one large, replicated effect and it is not a
mutation:

| gene | architecture phase (real text, ~100M, 1,200 steps) | the V0 session's search space |
|---|---|---|
| residual-stream width | **the lever**: −0.1562 nats at the factorial's exact 46,656-assignment floor | unreachable |
| head count | −0.0553 nats *at wide width only*, parameter-free at fixed width | unreachable |
| depth | real but smaller, and **shallow wins**: +0.0209 nats at 10× the horizon | unreachable |
| expert capacity | **uncorrelated with loss**: ×4.51 active parameters at rho +0.214 (p = 0.62); fully dense places 7th of 8 | **the only reachable axis** |

`feasible_expert_mutations` yields expert grow/prune and nothing else, and the reachable gene is
the one a track measures *against* selection value when it is raised (5.53: at the onset,
depth 3 with six experts does not open the gate at 211/576, and above it the same dose removes
the value creation the four-expert cell has). So the question this note is about is not "is
topology useful" — it is **whether the machinery the project spent two phases building has
anything to select between** once it is pointed at a gene that moves. That is a different
question from Gate 4, and it is the only form of the original hypothesis that the measured
evidence does not already answer.

**What would make it worth running.** The topology track's own two-term decomposition
(`adaptive − null` = `(adaptive − random)` + `(random − null)`) says the honest primary is the
**controller's own edge over a random schedule**, not the total gain: across the depth ladder
that edge is flat to slightly worse (p = 0.879 from depth 3 to 9) and the pool term carries
essentially all of the value. A width arm that reports a total gain without that decomposition
would repeat the mistake the churn programme had to unlearn.

## 2. What a width mutation costs, measured on the model in the tree

Run against `act.model.V0MoECausalLM` at the code-LM leg's `match` shape, from the model's own
parameter list:

| | |
|---|---|
| parameters whose shape depends on `hidden_size` | **166,423,296 of 166,423,296 — 100.0%, across all 254 tensors** |
| the largest single width-dependent tensor | `token_embedding`, 24,772,608 (14.9% of the model) |
| what else is in the same boat | every attention projection, every router, every expert's three matrices, every RMSNorm gain, and the **optimizer's AdamW moments** (2× the parameters) |

This is the finding that changes the design: `grow_expert` adds 3hW to one layer out of twelve,
which is why the project could measure its penalty, its clone-inversion and its prune-dominance
locally. **A width mutation is not that kind of perturbation — it is a reparameterisation of
every weight in the model at once**, and the project has already measured what a mid-training
architecture change costs when it *is* local: the penalty is the architecture change itself
(5.23), a freshly drawn expert is most of the residual (5.26), and cloning an existing expert
with its routing row inverts it (5.26-5.28). None of that machinery extends to 254 tensors.

## 3. The constraints the model's own declarations impose

The declared-config validation requires `hidden_size % 128 == 0` and
`(hidden_size // attention_heads) % 64 == 0`. Measured consequences, from constructing the
model at each width:

| hidden | heads | head_dim | parameters | vs 768 | legal |
|---|---|---|---|---|---|
| 768 | 12 | 64 | 166,423,296 | 1.000x | yes |
| 768 | 6 | 128 | 166,423,296 | **1.000x** | yes |
| 832 | 13 | 64 | 182,847,808 | 1.099x | yes |
| 896 | 14 | 64 | 199,665,536 | 1.200x | yes |
| 1024 | 16 | 64 | 234,480,640 | 1.409x | yes |
| 1536 | 12 | 128 | 389,469,696 | 2.340x | yes |

Two things follow, and both are load-bearing rather than incidental.

1. **Heads are free at fixed width** — `768/12` and `768/6` are the same parameter count, which
   is the arch phase's "head count is parameter-free" statement read on this model.
2. **Width cannot be moved alone.** At `head_dim = 64` every +128 of width is +2 heads; the only
   way to hold `attention_heads` fixed while widening is to double `head_dim` (768 → 1536 with
   12 heads), and rotary position encoding and the GDN recurrence are both functions of
   `head_dim` — so that is a **different attention geometry, not a wider one**. A width arm is
   therefore, by the model's own construction, an arm about a *joint* width × heads mutation.
   The arch phase priced the head half of that: −0.0553 nats at wide width, +0.0084 at narrow.
   Any upside a width controller can claim is bounded by the sum of a width effect and an
   already-priced head effect, and the attribution between them is entangled *by construction*
   rather than by oversight.

## 4. The mechanism, and the contract it has to meet

The mutation shape with a chance of being informative is a **function-preserving widening**, and
as of 2026-09-25 it is implemented and measured: `src/act/model/widen.py` returns a model of a
requested width that computes the same function as the one handed in, checked in
`tests/test_width_widen.py`.

It is **not** the Net2Net duplication this section first proposed, and the difference is the whole
result. Duplicating units and splitting the weights that touch them needs a per-unit split factor
and a matching compensation at every consumer. What works instead is **zero extension**: the new
coordinates are zero, the new heads emit zero, and every new input column and output row is zero,
so nothing is duplicated and there is no split to get wrong. The project already reasons in this
shape of contract — `activation_checkpointing` is a *trajectory-inert* lever verified
bit-identical (sections 5.37/5.37.1) — and the KV-cache work on the code-LM student was stopped by
exactly such a measurement.

**The named obstacle was RMSNorm, and it is exactly compensable — as two scalars, not one.** The
widened stream is `x' = [x, 0]`, so `mean(x'^2) = mean(x^2) · h/h'` and the normalised coordinates
come out scaled by `sqrt(h'/h)`; scaling the copied weight by `sqrt(h/h')` cancels that. But the
epsilon sits *inside* that root-mean-square, so what remains is `rsqrt(a/h' + eps)` against
`rsqrt(a/h + eps)` for the token's own `a = \|x\|^2` — which no constant weight can fix, because the
discrepancy depends on the activation. Scaling the epsilon by `h/h'` makes the two radicands
proportional for **every** input, and the pair is an identity. Measured, each half is load-bearing:

| state of the compensation | logits against the un-widened model |
|---|---|
| neither half | **5.79e-4** |
| weight scalar only, epsilon left alone | >100x the compensated error |
| epsilon scaled only, weight left alone | >100x the compensated error |
| both halves, at the student's shape (768 -> 1024, 12 -> 16 heads) | **2.50e-06 absolute, 9.7e-07 relative** |

So the contract this section demanded is met, but **not in the form it asked for**: bit-identity is
the wrong target for this operator. `_RMSNorm` computes its statistic in float32 *by design*, so a
widened norm rounds in the last place of that cast, and the residue does not improve with a float64
model (6.90e-08 there against 5.96e-08 on a toy) — which is what identifies the floor as the
model's own normalisation rather than as the graft. The honest contract is "the same function to
within the model's own normalisation rounding", and that is the sentence the test carries.

**Cost of one graft at the student's shape** (`widening_report`; vocabulary 32,256, context 512,
depth 12, 8 experts, 2 active, width 768):

| step | total parameters | active parameters |
|---|---|---|
| h768 -> h896 (12 -> 14 heads) | 166,423,296 -> 199,665,536 (**1.200x**) | 81,488,640 -> 100,575,104 (**1.234x**) |
| h768 -> h1024 (12 -> 16 heads) | 166,423,296 -> 234,480,640 (**1.409x**) | 81,488,640 -> 121,234,432 (**1.488x**) |
| h768 -> h1152 (12 -> 18 heads) | 166,423,296 -> 270,868,608 (**1.628x**) | 81,488,640 -> 143,466,624 (**1.761x**) |

The active cost runs ahead of the parameter cost at every step (1.488x against 1.409x), which is the
asymmetry section 8 records from the other direction, and it is why a width schedule's win is
ambiguous between *capacity* and *compute* unless section 5's controls are run.

**And the new coordinates start at exactly zero gradient** — the graft is a saddle. That is a
property of any zero extension rather than a defect, and it means a search operator built on this
owes the new coordinates a perturbation rather than a training step; otherwise the controller widens
a model and nothing happens. Measured for three routes to the loss (a norm entry, a new head's qkv
rows, an expert's new input columns).

Two things are **refused rather than approximated**, both pinned by test: a *shrinking* step — that
is pruning, a different operator with a different information loss, and slipping both under one name
is how a mutation budget gets spent on the wrong one — and a **recurrent (`gdn`/`hybrid`) layer**,
whose `head_dim x head_dim` state changes geometry when a width step adds heads, so no zero extension
preserves it. If exactness had *not* been reachable, the fallback would not have been a weaker
contract: it would have been the observation the project already made about cloning, that **the value
came from copy-and-preserve rather than from adding capacity** — a widening that re-instantiates a
unit from scratch is a *grow*, which is the more expensive half of the measured mutation penalty
(5.25) and the half cloning inverted (5.26).

## 5. The design, if the contract holds

| element | declaration |
|---|---|
| **Primary** | `adaptive − random`, the controller's own edge, on final held-out loss, seed-paired, exact sign/re-split test with its floor printed beside its p |
| **Secondary** | `adaptive − fixed-narrow-at-matched-final-parameters` (the pool term), reported separately and *not* as the headline |
| **Controls** | (i) a random width schedule at the same cadence and the same number of events; (ii) a fixed-wide model trained for the same tokens (matched final parameters, unmatched FLOPs — stated); (iii) a fixed-narrow model trained for the same FLOPs (matched compute, unmatched parameters — also stated) |
| **Confound, declared up front** | a schedule that widens late spends fewer FLOPs for the same tokens, so a win is ambiguous between *capacity* and *compute*; the two fixed controls above are what separates them, and the report's own 5.54 (4× training is inert below the onset, not inert above it) is the prior on which direction that cut goes |
| **Cadence** | the same three events the topology arms use, so the event count is not a new variable |
| **Admission bar** | the arm does not run until (a) the preservation test exists and passes, (b) the arm is costed from committed per-run figures, (c) its rule and its four outcomes are in this file, and (d) the card or the quota is free — a shared card is what made a 3.3 GPU-hour arm take 3 hours to complete one candidate (§10 of the findings) |
| **Outcomes, four, declared** | (1) edge positive and clear → the original hypothesis has a form that survives on a gene that moves, and the search should be rebuilt around width; (2) pool positive, edge flat → **the same result the churn programme found**, now on the lever that matters: mutation helps, selection does not, and the adaptive machinery is not the contribution; (3) edge negative → the controller actively harms a lever that pays, which is a sharper version of the churn finding; (4) undecidable at the seeds affordable → the honest exit, and it is why the bar above matters |

**Prior, stated before it is tested.** On everything measured so far, outcome (2) is the most
likely: the value in this family has always been the substrate's need for perturbation rather
than the controller's skill at choosing it, and the one regime where selection crossed a
threshold (wide and shallow, below the onset) is recorded in the report as "a hypothesis ...
measured nowhere else". This arm is the cheapest way to find out whether that survives the move
to the gene that pays — and if it lands on (2), then the project's contribution is the
architecture finding plus a well-established negative about learned topology, which is a
publishable pair. It should be run knowing that this is the likely answer.

## 6. What this note does *not* recommend

It does not recommend re-opening Gate 4 on expert capacity first. That gate is registered, its
null clause is the operative bar, its most likely outcome is written down as the null arm
winning, and the gene it can actually move is the one two studies measure as inert or worse. It
is not worthless — it is the registered test of the hypothesis as declared, and a null there is
a real result at declared fidelity — but it is the *third* thing to spend a GPU week on, behind
settling the genome question above and behind using the week to train something whose numbers a
reader can compare against an outside model (which is what the code-LM leg is for).

## 7. Sequenced actions

1. **The mechanism is built, and it landed because it is a new file rather than an edit.**
   `src/act/model/widen.py` and `tests/test_width_widen.py` do not touch `v0_moe.py` or
   `train_code_lm.py`, which is what let them land while the campaign's two arms are still in
   flight: the arms' provenance is that no file they import changed, and a new module satisfies
   that where an edit to the model would not. The freeze this section recorded is still what
   governs item 4 below — nothing the arms read has moved. What remains is the *operator*: a
   mutation that widens a partly trained model, records the graft in the run's own signature, and
   perturbs the new coordinates, which start at exactly zero gradient.
2. **The preservation test exists and passes**, so the admission bar's item (a) is met: a fixed
   batch, logits before and after, across every position, with both halves of the norm
   compensation shown to be load-bearing and a refusal for the layer type the widening cannot make
   exact. One thing the test changed: the bar was written as **bit-identity**, and the measurement
   says the achievable contract is the model's own normalisation rounding (2.50e-06 absolute,
   9.7e-07 relative at the student's shape), because `_RMSNorm` computes its statistic in float32
   by design. That is a weaker contract than the bar as drafted, and it is adopted with its reason
   rather than by lowering the assertion until it passed.
3. **The cost, from committed figures, before the arm is pushed.** The measured rates are in
   `artifacts/arch-zoo/PERFORMANCE_NOTES.md` and in the code-LM leg's reports; a width arm at
   this model's shape must be priced at ~1.1-1.4× the `match` student's per-token cost for the
   wider arms.
4. **Then a decision, deliberately, with the quota in hand** — which is what the report says
   should happen with a width operation rather than adding it to a running queue.

## 8. The first arm's shape, measured at a matched budget

The recipe this note argues for has to be run *as a shape*, and the shape is not free: at a
matched **total** parameter budget a wide-shallow student executes **more** active arithmetic
than the deep one, because the residual stream and attention are entirely active while the
experts that a depth cut removes are mostly inactive. So an arm has to say which budget it
holds. Measured with `estimated_parameter_count` on the real model, vocabulary 32,256 (the
student's) and context 512:

| config | total | vs `match` | active | vs `match` | active/total |
|---|---|---|---|---|---|
| `match` h768 d12 H12 E8 A2 W768 | 166,423,296 | 1.000x | 81,488,640 | 1.000x | 0.490 |
| wide h1024 d6 H16 E8 A2 W1100 | 166,392,832 | **1.000x** | 85,292,032 | 1.047x | 0.513 |
| wide h1024 d6 H16 E8 A2 W1024 | 158,921,728 | 0.955x | 83,424,256 | 1.024x | 0.525 |
| wide h896 d6 H14 E7 A2 W1568 | 166,232,192 | 0.999x | 81,936,512 | **1.005x** | 0.493 |
| wide h1024 d5 H16 E8 A2 W1376 | 166,775,808 | 1.002x | 82,234,368 | 1.009x | 0.493 |
| deep h896 d8 H14 E8 A2 W896 | 157,424,512 | 0.946x | 80,354,176 | 0.986x | 0.510 |

That table's shape is not a needle in a haystack, and the search behind it is over the
model's **own** legal grid rather than a convenient one: hidden multiple of 128 from 896 to
1280 with `heads = hidden/64`, depth 4 to 13, expert count 2 to 16 with `active_experts = 2`
(the class floor, which rejects 1), and expert width 256 to 2048 step 32. On that grid, **46**
shapes hold both budgets within +/-2% (241 within +/-5%), so the choice below is a choice
rather than the only option. The arm this note
recommends is the second row, because it is the one that holds the routing regime
(`E8 A2 W=h`) and the head geometry fixed while halving depth and raising width by a third —
the two genes the crossed factorial actually priced. It buys that by spending 4.7% more
active FLOPs per token than `match` at the same total parameters, which is a confound **in the
wide arm's favour** and therefore has to be declared in the pre-registration rather than
quietly absorbed. Rows four and five are the fallbacks if it is the *active*
budget that has to be held: row four is the tightest joint fit the grid contains (`0.999x`
total against `1.005x` active) and pays for it with the expert count as well, while row five
keeps `E8` and moves the expert width instead. Either way the arm prices a joint change rather
than the width gene alone, which is the trade to state rather than hide.

Both recommended shapes satisfy the project's own kernel-alignment rule for a declared config
(`hidden_size % 128 == 0`, `head_dim = 64`, `context_size % 128 == 0`), and neither is
constructible any other way: at `head_dim = 64` every +128 of width is +2 heads (768/12 ->
1024/16), which is the joint constraint section 3 records.

The entry to add when the tree is safe to edit:

```python
"wide": dict(hidden_size=1024, depth=6, attention_heads=16, expert_count=8, active_experts=2, expert_width=1100),
```

**This arm is trained from scratch, and it is not the adaptive gate of section 5.** The two are
different experiments that share a gene, and conflating them would spend the arm on the wrong
question: this one asks whether the architecture search's width finding survives at a scale where
the model can do something (it changes width *and* depth, at a matched total budget, from a random
init), while the gate asks whether a controller can *choose* to widen a model that has already
trained. The mechanism in `src/act/model/widen.py` serves the second: it grows the residual stream
at **fixed depth**, so the shape it can reach from the code-LM student is h768 d12 -> h1024 d12
(234,480,640 parameters), not this section's h1024 d6. Reaching an arm's shape by pruning depth is
a different operator with a different information loss, and is not what widening does.

It is two arms, not one: `ours-wide-distill` against `ours-match-distill`, and `ours-wide-ce`
against `ours-match-ce`, so the width effect is read within each objective and the objective
effect is read within each shape. The declared headline is `wide - match` at matched total
parameters, reported with the 4.7% active-FLOP gap beside it, and the paired per-window test
in `scripts/code_lm_table.py` is the estimator — the arms differ in exactly one declared
thing, so its refusals do not fire.
