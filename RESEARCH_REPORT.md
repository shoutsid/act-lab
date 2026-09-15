# Adaptive Computational Topology — Research Report

Status: skeleton covering the narrowed first scientific milestone (Gate 3,
low-fidelity Experiment 0). Updated: 2026-09-13 (deliverable 2's reproducible
notebooks built and verified; see section 7).

This report is the narrative companion to `IMPLEMENTATION_STATUS.md` (the
implementation tracker) and the append-only study records under
`artifacts/study-summaries/`.

Reproducibility at notebook fidelity (plan sections 39 and 104) now ships with
the report: all ten notebooks (`00_environment` through `09_compare`) live
under `notebooks/`, are emitted byte-identically by
`scripts/build_notebooks.py` (`--check` verifies the committed files), and
were executed end-to-end during verification; each ends by writing a summary
record under `artifacts/notebook-records/` per plan section 40's session
contract. The gate-study notebooks (03, 09) run deliberately 4-seed demos of
the machinery - the powered 384-seed verdicts remain the versioned studies of
section 3. Every number below is machine-checked against
those records; `study_id` values are reproducible via
`PYTHONPATH=src python -m act.evolution.cli ...` from the stated config and
seeds.

## 1. Hypothesis (plan §98 format)

- **Hypothesis**: adaptive topology improves capability per FLOP.
- **Prediction (Gate 3)**: adaptive topology selection beats matched random
  churn on final held-out loss.
- **Control**: identical training schedule and identical mutation count and
  cadence. The original claim was that "only the *selection policy* of
  mutations differs"; section 9.5 shows this is **not accurate** as implemented -
  the arms also differ in their candidate distribution, and a loss-blind
  choice from the same candidate pool reproduces most of the effect. The
  corrected control is pool-matched. Sections 5.8-5.10 add a third control:
  the **do-nothing arm** (matched budget, zero mutations), which beat random
  churn on real text (5.8) and exposed study 3's depth-1 toy edge as churn
  suppression (5.9), while the depth-4 pass survives it (5.10). The claim
  "adaptive improves capability per FLOP" is therefore now supported only
  where a controller clears the null bar: depth 4 with the spread policy.
- **Decision if false**: stop and simplify the architecture controller; do not
  scale until a controller passes.

## 2. Method

Matched-budget Experiment 0 in the Python training path: a tiny static-MoE
causal LM, three structural mutation events unless stated otherwise, and two
arms compared per seed — **random** (uniform mutation choice) versus a
**treatment** (cheap-signal controller, bounded probe-loss controller, or the
probe-guided diagnostic upper bound). Final loss is averaged over four
deterministic held-out validation batches; mutation budgets must match exactly
or the study aborts.

**Substrate and control caveats (section 9.5).** Every study in section 3 ran
at `depth=1` with 4 experts, i.e. a 5-candidate mutation space on a single
layer — the toy path could not build depth > 1 until the fix recorded in
section 9.1. The random control draws uniformly over layers and operations
while a bounded treatment draws from a capped candidate list, so the two arms
differ in candidate distribution as well as selection; the study below is
interpreted with that in mind, and the pool-matched control (section 9.5)
should be the baseline for any re-run. **Sections 5.9-5.10 sharpen these
caveats into a verdict: on this depth-1 substrate the study-3 pass is churn
suppression (null < adaptive < random), so the depth-1 rows must be read
with the do-nothing bar in mind and only the depth-4 family (sections 5.6,
5.10) supports the improvement claim as stated.**

Gate-study verdicts are objective and machine-computed
(`src/act/evolution/study.py`):

- **Primary**: exact one-tailed binomial sign test on per-seed wins at
  alpha = 0.05 (a pass needs a minimum win count reported with every result:
  209 of 384 seeds, 207 of 380, 57 of 96).
- **Secondary**: directed one-tailed paired t (helpful treatment = negative
  loss delta; p = P(T <= t)).
- **Cost**: probe search FLOPs are booked per arm and reported as
  `search_flops_by_arm` and as probe cost relative to training FLOPs.

Every study was **pre-registered in the working notes before its seed block was
examined**: config, seeds, thresholds, and the decision rule were fixed first;
each study used a disjoint fresh seed block (or same-seed pairing where the
design requires it), and no study was re-run on its own seeds after seeing the
result.

## 3. Results: the factor map — nine studies, three passes

**(2026-09-15 note on scope.** This section is the ORIGINAL depth-1 factor
map, preserved as run. Sections 5.8-5.15 later added the null control and
re-ran the decisive comparisons against it: at depth 1 the celebrated
passes are churn suppression (the adaptive arm loses to doing nothing),
the depth-4 pass survives the null bar, and the dose-response places the
suppression-to-creation onset between depth 3 and 4. The synthesis in
section 10 consolidates all of it; the table below is the historical
record.*)

Nine pre-registered gate-style quality tests now cover the controller design
space at toy fidelity. One configuration family - a rich candidate set - has
passed three times, twice decisively; on the decoupled arm-RNG path of section
9.11 the same three passes hold with stronger statistics and a fourth study
(variance reduction) reaches a threshold-level pass that fails family-wise
correction and is therefore reported as a null.

| # | Study | Selection policy | Events x probe budget | Horizon | Seeds | Wins | Mean delta | Binomial p | Paired-t p | Probe cost | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Legacy controller | routing/Fisher cheap signals | 3 x n/a | 48 | 384 (1000-1383) | 195/384 | +0.006766 | 0.3993 | 0.8437 | ~0 | FAIL |
| 2 | Guided upper bound (diagnostic) | probe-loss, all candidates, all 4 batches | 3 x 4b all-cand | 48 | 96 (2000-2095) | 59/96 | -0.041325 | 0.0158 | 0.00025 | high | PASS |
| 3 | Probe-loss controller | argmin of probed loss | 3 x cap4, 2 batches | 48 | 384 (3000-3383) | 213/384 | -0.025955 | 0.0181 | 7.4e-05 | 41.9% | **PASS (quality)** |
| 4 | Cheap probe | argmin of probed loss | 3 x cap2, 1 batch | 144 | 380 (4004-4383) | 181/380 | +0.011747 | 0.8351 | 0.8857 | 4.13% | FAIL |
| 5 | Cadence | as (4), 6 events | 6 x cap2, 1 batch | 144 | 384 (5000-5383) | 137/384 | +0.047014 | 1.0 | ~1.0 | 8.41% | FAIL (worse than random, z = -5.61) |
| 6 | Variance reduction | as (4), 4 probe batches | 3 x cap2, 4 batches | 144 | 384 (6000-6383) | 179/384 | -0.003907 | 0.9159 | 0.3453 | 16.0% | FAIL |
| 7 | Horizon replication | as (3) | 3 x cap4, 2 batches | 144 | 384 (8000-8383) | 231/384 | -0.052402 | 4.04e-05 | 1.07e-08 | 14.0% | **PASS (decisive)** |
| 8 | Richness A | cap-4 candidates, 1 probe batch | 3 x cap4, 1 batch | 144 | 384 (7000-7383) | 228/384 | -0.036982 | 1.4e-04 | 5.0e-05 | 7.3% | **PASS (decisive)** |
| 9 | Richness B (same seeds as 8) | cap-2 candidates, 2 probe batches | 3 x cap2, 2 batches | 144 | 384 (7000-7383) | 193/384 | +0.007156 | 0.47965 | 0.7758 | 8.1% | FAIL |

A same-seed **placement** study (study 4's config with mutations moved from
steps 12/24/36 to 40/80/120) is a seventh, mechanistic comparison rather than a
gate test: late placement changed 39% of probe decisions but did not
significantly change outcomes (201/380 pairs favored late placement, paired
p = 0.0788), exonerating mutation placement as the explanation of study 4's
failure.

**Every number in the table above is from the coupled arm-RNG path** (section
9.8). The factor map was re-run with the coupling fixed, on the same configs and
seed blocks, in the declared reproduction check of section 9.10; the decoupled
table is section 9.11. Headline: **9 of 10 verdict labels reproduce**, every
effect size moves toward the treatment (all ten win shifts positive), the
number of studies clearing family-wise correction rises from 2 to 4, and the one
label flip is the variance study - a threshold pass (209/384, p 0.046) that
fails alpha/9 and is therefore a null in both directions rather than a new
positive. Read the rest of this section with those corrections applied.

### 3.1 Narrative

The legacy controller (study 1) was the state the project started in: cheap
signals — routing load, gradient norms, Fisher-style estimates — with no probe
cost at all. Across 384 seeds it is a coin flip with a slightly negative mean:
whatever those signals measure at this fidelity, they do not select better
mutations than chance. That negative result is what justified replacing the
controller.

Before replacing it, the **guided arm** (study 2) established that the
hypothesis is testable at this fidelity at all: probing *every* feasible
mutation on held-out batches and taking the best beat random churn
significantly (p = 0.0158). Selection can win; the information exists.

The **bounded probe-loss controller** (study 3) then passed Gate 3 on quality:
capping the candidate set at 4 and the probe budget at 2 batches preserved a
real, if modest, advantage (213/384 wins, threshold 209; paired-t p = 7.4e-05).
Its pre-registered cost criterion failed — probe search consumed 41.9% of the
training FLOPs at this 48-step horizon — so compute-efficiency was left as an
explicit open question, not a footnote. The pre-registered **horizon
replication** (study 7) then re-ran that exact configuration on a 144-step
horizon with fresh seeds and passed decisively: the win rate *rose* to 60.2%,
the mean delta *doubled* to -0.0524, both statistics cleared the family-wise
(alpha/7) bar, and probe cost landed at 14.0% — exactly the 41.9%/3 predicted
by fixed-per-event probe cost amortizing over a longer horizon. The pass is not horizon-specific, and the contrast with study 4 (identical horizon, cheaper
probe: p = 0.835) pins the active axis on candidate richness plus probe
quality, not on the horizon itself. The pre-registered **richness study**
(studies 8 and 9) then isolated that axis at fixed nominal probe tokens
(192/event for both arms): a rich candidate set with even a single-batch probe
passed decisively (228/384, p = 1.4e-04), while a narrow candidate set with
doubled probe quality was a coin flip (193/384, p = 0.48) on the same seeds.
The paired A-vs-B comparison confirms richness directly: A better in 225/384
pairs (sign z = 3.37, two-sided p = 8e-04; mean paired delta -0.0441).

Reducing probe cost (study 4: fewer candidates, 1 batch, longer 144-step
horizon to amortize) hit the 5% cost target (4.13%) but lost the quality edge
entirely. The **cadence** study (study 5) tested whether accumulating the weak
signal over six events instead of three would help; it did the opposite —
137/384 wins is significantly *worse* than random (z = -5.61, p ~ 2e-08).
On the decoupled path the same study is 148/384, still significantly worse
(P(X<=148) = 4.1e-06) but less severely: the harm claim holds, its magnitude
was inflated by the baseline (section 9.11).
The **variance** study (study 6) tested the complementary winner's-curse
hypothesis — that single-batch probes mislead because their argmin partly
selects for probe-batch noise — by quadrupling probe batches at fixed
candidates; the mean moved in the predicted direction (+0.0117 to -0.0039) but
not significantly, so variance reduction alone does not restore the edge.
This is the study whose label flips on the decoupled path (179/384 -> 209/384,
p 0.046), where 209/384 is exactly the alpha=0.05 threshold; it fails alpha/9,
so the corrected reading is "no probe-batch effect in either direction" rather
than a pass (section 9.11).

### 3.2 Mechanism: what the probe signal is worth

A pre-registered counterfactual analysis
(`artifacts/study-summaries/probe-validity-analysis.json`; the committed
generator is `scripts/probe_validity_analysis.py`, which reproduces its stored
results) forced each of the 4 candidates at the first mutation event (branch 0
verified to reproduce the greedy run, and cross-checked seed-for-seed against
the study summary's adaptive arm) and compared probe ranks to realized final
ranks across 384 seeds. Both RNG regimes, since section 9.11 re-ran it:

| Statistic | Coupled (published) | Decoupled (section 9.11) |
|---|---|---|
| mean per-seed Spearman rho | 0.1057 (t(383) = 3.64) | **0.0776** (t = 2.58) |
| one-tailed p | 1.4e-04 | **4.98e-03** |
| rho > 0 in | 53.1% of seeds | 51.3% of seeds |
| pooled within-seed z-scored Pearson r | 0.1041 | 0.0809 |
| probe-argmin = realized best (chance 25%) | 28.1% | 28.4% |
| mean worst-branch excess avoided | 0.0860 | 0.0839 |
| mean loss left on the table | 0.0601 | 0.0592 |

- Probe loss **does** predict realized loss, but more weakly than published:
  rho = 0.0776 on the corrected path (p = 4.98e-03), still clearly positive
  and, with the coupled path, the only place the signal is measured directly
  rather than through a win count.
- The **avoidance** reading is confirmed and if anything sharper: the
  probe-argmin matches the realized best in only 28.4% of seeds (chance 25%)
  while avoiding a mean worst-branch excess of 0.0839 and leaving only 0.0592
  on the table. The probe's value is in not picking badly, not in picking the
  best - and that survives the RNG fix with essentially unchanged magnitude,
  whereas the rank correlation weakens. That is consistent with section 9.11's
  finding that the treatment's *edge* persists while its *ranking* signal is
  weaker than the coupled path suggested.

**Wherever the rest of this report says "rho ~ 0.11", read it as the coupled
amplitude (0.1057); the corrected value is `0.078`.** The qualitative
statements are unchanged, the numbers are not.

The coherent story across all nine studies: **selection value comes from having
candidates worth choosing between**. Probe loss is a real but weak predictor
(rho ~ 0.078 on the corrected path, 0.11 coupled) whose value is concentrated
in avoiding bad mutations; a rich
candidate set (cap 4) supplies that avoidance opportunity and passes Gate 3 at
every horizon and probe-batch count tried, while every narrow-candidate
variant is a coin flip regardless of probe quality, and repeated greedy
argmin over more events (study 5) amplifies probe noise into a significant
loss. Probe-batch count is secondary within the rich regime (study 8's
cap-4/1-batch delta -0.037 vs the replication's cap-4/2-batch -0.052;
two-sample z ~ 1.1, not significant). The passing family's probe cost is
fixed per event, so it amortizes with horizon (41.9% -> 14.0% -> 7.3%); at
toy scale it is still not negligible, and whether it becomes negligible at
research scale is precisely the plan's deferred compute-efficiency milestone.

### 3.3 Statistical caution

- Nine gate-style quality tests have been run. On the coupled path study 3's
  sign-test pass (p = 0.0181) was marginal under Bonferroni while studies 7 and
  8 cleared the family-wise bar (alpha/9 ~ 0.0056) on *both* statistics
  (4.0e-05/1.1e-08 and 1.4e-04/5.0e-05). On the **decoupled path (section
  9.11) four studies clear alpha/9** - horizon (1.7e-09), richness A (9.4e-05),
  probe-loss (0.00128, which was the marginal one and is no longer marginal),
  and the guided diagnostic (0.00279) - so the family-level evidence no longer
  depends on counting only two clear tests. The guided study (2) is a
  diagnostic, not a gate claim.
- Studies 1, 3, 5, 6, 7, 8 are mutually independent (disjoint seed blocks).
  Studies 4, 9, and the placement study share seeds by design (4 and placement
  as same-seed paired comparisons; 9 paired against 8). Study 6's descriptive
  mean comparison against study 4 is a two-sample approximation
  (z ~ -1.13), not a paired test.
- All raw records are bit-for-bit reproducible; `study_id` values are stable
  across re-runs and re-emissions.

## 4. What Gate 3 currently means

**Gate 3 is established on a substrate that can express topology search.** The
project's gate evidence is now the pre-registered section 5.6 depth-4
confirmation (section 9.9): committed before the run and executed once on the
never-observed block 60000-60383, it passes the registered comparison
(255/384, p `6.1e-11`) *and* the controlled selection increment against a
loss-blind choice from the identical candidate pool (236/384, p `4.1e-06`),
while the inverted-signal control does not pass and the loss-blind pool control
clears only alpha (`0.046`). This matters because it is the first gate result
measured on a pool (20 feasible candidates over 4 layers) that is genuinely
larger than the search cap, and with the arm-RNG coupling fixed (`independent_
arm_rng=True`) and out-of-sample - whereas the depth-1 passes below were
measured on a degenerate 5-candidate single-layer space, on the coupled RNG
path, and needed the window > 1 mutation event to begin differentiating.

Everything in the remainder of this section predates that confirmation and
describes the depth-1 factor map; read it as a weaker, degenerate-space
instance of the same result rather than as the project's gate evidence.

Gate 3 ("adaptive > random; if failed, stop and simplify") had been passed
three times by one controller family - bounded probe-loss selection with a rich
candidate set (cap 4) - across two horizons and two probe-batch counts, with
two of the three passes decisive under family-wise correction. The mechanism
is localized: richness of the candidate set is the active axis (paired
A-vs-B p = 8e-04), probe-batch count is secondary, and cadence beyond three
events on a weak probe is actively harmful. **Mechanism attribution revised
by section 5.12:** the richness-vs-batch reading above was measured at depth 1
only; the 2x2 (depth x candidate policy, all cells against a null control)
shows DEPTH, not the candidate policy, is the axis that separates churn
suppression from value creation - at depth 1 the adaptive arm loses to doing
nothing under BOTH policies, and at depth 4 it beats doing nothing under BOTH.
What remains open is
**compute-efficiency**: the validated cost-saving variant (cap-4/1-batch)
cuts probe search to 7.3% of toy-scale training FLOPs and amortizes with
horizon exactly as the fixed-per-event model predicts, but scaling stays
withheld until a real-scale measurement shows the cost becomes negligible -
which is the plan's deferred compute-efficiency milestone - after which Gate 4
(adaptive > static MoE) is the next universal gate. That measurement is no
longer a vague milestone: it is pre-registered as a protocol with declared
config, predicted cost, and decision rules (section 5.1), now amended by the
do-nothing clause of section 5.14. The controller
default remains cap-4/2-batch; cap-4/1-batch is the validated cheaper option.

## 5. Pre-registered next studies

| Study | Factor | Config | Seeds | Primary | Decision rule | Status |
|---|---|---|---|---|---|---|
| Horizon replication | study 3 config at steps=144 | cap4/2-batch, 3 events, steps=144 | fresh 8000-8383 | binomial sign test, alpha=0.05 | PASS -> pass generalizes; FAIL -> pass was horizon-specific | **DONE: PASS** (231/384, p 4.04e-05, t p 1.07e-08, cost 14.0%) |
| Richness | candidate set size at fixed probe tokens | cap4/1-batch (A) vs cap2/2-batch (B), 3 events, steps=144 | fresh 7000-7383 | binomial sign test, alpha=0.05; factor claim: paired A-vs-B, one-tailed "rich better" | PASS & A>B -> richness is the active axis; B passes & B>=A -> probe quality is; neither -> both jointly necessary | **DONE: PASS for richness** (A 228/384, p 1.4e-04, t p 5.0e-05, cost 7.3%; B 193/384, p 0.48, cost 8.1%; paired A better 225/384, z = 3.37, p 8e-04) |
| Real-scale cost measurement | probe-cost amortization | cap4/1-batch (primary) + cap4/2-batch (secondary), 3 events, 24.6M tokens/arm, declared V0 285M config, **`probe_candidate_policy=spread`** (see 5.1/5.5) | **15000-15011** (adopted 5.5 change 4; the declared 9000-9011 was spent by the 5.3 diagnostic) | C = adaptive search/training FLOPs per arm, <= 5% (pre-registration 1's rule) + tolerance band [0.047%, 0.189%] + **pool-matched control clause** | cost + band + harm guard clear **and the loss-blind control arm must not clear its own gate** -> Gate 4 authorized; control passes, control missing, band miss or harm guard -> Gate 4 withheld at exit 2 | **REGISTERED as revised (section 5.5, adopted); execution planned (section 5.2); not yet run** |
| Depth-4 Gate 3 confirmation | substrate expressiveness + selection value, with the arm-RNG fix | cap4/2-batch, `spread`, `independent_arm_rng`, 3 events, steps=48, depth 4 | fresh 60000-60383 (384 pairs) | argmin vs uniform churn, alpha/3 = 0.01667 (>= 214/384); paired argmin vs pool-matched control | PASS -> Gate 3 established on a non-degenerate substrate, 5.5 adopted; POOL-DRIVEN -> withdraw the selection claim; NOT REPRODUCED -> Gate 3 open, 5.1 must not launch | **DONE: PASS** (argmin vs uniform 255/384, p `6.1e-11`; paired increment 236/384, p `4.1e-06`; pool-matched control 209/384, p `0.046`; validity control does not pass). Section 9.9 |

### 5.1 Protocol: real-scale compute-efficiency measurement (pre-registered; gates Gate 4)

**Role.** Plan section 56 (Experiment 13, "Compute-Efficiency Scaling") requires
search FLOPs to be included in the adaptive total; this protocol pre-registers
the minimal decisive version of that experiment. Gate 4 (adaptive > static
MoE) is not entered until it passes. It is a measurement-plus-evidence
protocol, not a quality gate.

**Hypothesis.** Probe-selection cost is fixed per mutation event while training
cost scales with tokens, so the cost ratio collapses as scale and horizon grow.
The toy anchors are on-model: 41.91% at 48 steps (study 3), 14.03% at 144
steps with 3x the training (study 7), 7.29% for the 1-batch variant (study 8).

**Prediction (declared before measurement).** With the code's accounting
convention (forward-equivalent FLOPs on both sides: `parameter_count x tokens
x 2` per training step; probe forwards `active_parameter_count x
probe_tokens x 2` booked into search; no backward term on either side),

```text
C_expected = S / T
S = E * k * Bp * 2 * N_active      (E events, k candidates, Bp probe tokens)
T = 2 * P * D                      (D = training tokens per arm)
```

Declared V0 configuration (plan section 8): L=12, d_model=1024, heads=16,
context=2048, vocab=32768, E=8 experts/layer, k=2 active, ffn=1024 per expert
(active FFN width 2048 = the plan's ~4096 dense-equivalent), AdamW,
checkpointed activations. This gives P = 285,336,576 total parameters (inside
the plan's 256-350M target) and N_active = 134,341,632. Declared budget: D =
24,576,000 tokens per arm (batch 2 x context 2048 x 6,000 steps); probe batch
= one training-batch shape (2 x 2048 = 4,096 tokens, semantics-preserving as
at toy scale); E=3 events at steps 1500/3000/4500.

```text
cap4/1-batch (primary):   S = 1.32e13, T = 1.40e16, C = 0.0942%
cap4/2-batch (secondary): S = 2.64e13, T = 1.40e16, C = 0.1883%
```

- **Tolerance band (pre-committed)**: C_measured in [0.5, 2] x C_expected =
  [0.047%, 0.189%] for the primary. Rationale: every architectural deviation
  between toy and V0 (per-layer expert share, parameter growth, batch shape)
  enters the identity as a bounded multiplicative factor on N_active or D; a
  factor-2 miss means the trained model is not the declared one, and the
  config, not the hypothesis, is at fault.
- **Margin**: the prediction sits ~53x (primary) and ~27x (secondary) under
  the 5% bar, and ~11x/5x under a 1% milestone bar.

**Primary criterion (cost).** Pre-registration 1's rule, adopted unchanged:
total probe search FLOPs <= 5% of the adaptive arm's training FLOPs, measured
by the same accounting that produced every toy number (aggregate per-arm
search FLOPs over aggregate per-arm training FLOPs). The 5% bar is retained
even though the prediction is far under it: loosening a criterion after
seeing the measurement is the failure mode this project's protocol exists to
prevent. If the measurement lands in the band, the pass is decisive at any
bar >= 0.2%.

**Secondary criteria.**

1. *Harm guard (not a powered quality claim)*: on the same 12 seed pairs the
   adaptive-vs-random sign direction must not be significantly anti-adaptive:
   if adaptive loses in >= 10 of 12 pairs (exact one-tailed sign test p =
   0.0193), cost-efficiency is recorded as passing but the Gate 4
   recommendation is withheld - cheap-and-harmful is worse than
   expensive-and-helpful (study 5's lesson).
2. *Throughput log (descriptive)*: tokens/s per arm per session and wall-clock
   overhead of probe rebuilds, reported per plan section 57's training metric
   list, so scaling behavior is measured, not assumed.

**Design.** 12 seed pairs, seeds 9000-9011 (fresh block), arms `random,adaptive`,
  <!-- SUPERSEDED (2026-09-14): this block is spent - see section 5.5 point 3.
  The registered text is left as written for the audit trail; the ADOPTED
  section 5.5 revision moves the queue to 15000-15011. -->
primary config cap-4/1-batch, secondary run cap-4/2-batch on the same seeds,
budget matched at E=3, steps=6000, batch 2, context 2048. Every quantity in
plan section 57's Model and Training metric lists is reported per arm
(parameters, active parameters, FLOPs/token, tokens, steps, wall time,
GPU-hours, training/search/evaluation/total FLOPs). Checkpoints persist
optimizer state, genome, and mutation history at every event (existing
checkpoint contract), so runs resume across the 12-hour Kaggle session limit.
Feasibility (declared): per-arm training ~= 0.7 h at 10k tokens/s on T4;
12 pairs x 2 arms x ~1.5 h ~= 36 h ~= two Kaggle sessions; fp32 AdamW ~=
4.6 GB plus activations within 16 GB.

**Decision rules.**

- **PASS (cost milestone)**: C_measured <= 5%, inside the tolerance band, and
  the harm guard clear -> Gate 4 work (adaptive > static MoE) is authorized on
  top of the measured configuration.
- **PASS with qualification**: C_measured <= 5% but outside the band -> record
  the discrepancy, identify which multiplicative assumption failed, and
  re-derive the prediction before Gate 4. No knob may be re-tuned against the
  measured C to pull it into the band (study 3's cost-criterion discipline).
- **FAIL**: C_measured > 5% -> the amortization hypothesis is false at V0
  scale; the factor map's cost axis gets a V0 row and Gate 4 stays blocked.
- A hardware or session failure counts as infrastructure, not evidence;
  resumed runs continue from checkpoints on the same seeds.

**Honesty constraints.** Seeds and thresholds fixed before launch; no re-runs
on measured seeds; probe budgets may not change between this pre-registration
and the measurement; all criteria (cost, band, harm guard) are reported
whatever the outcome.

**Execution.** The protocol runs from `T4_RUNBOOK.md` (pre-flight, session-1
throughput benchmark, the one-line launch command, queue rules, contingencies).
The verdict is machine-checked by `scripts/v0_verdict.py`, which reads the
published seed summaries and applies exactly the rules above - aggregate
adaptive cost vs the 5% criterion, the [0.5, 2]x tolerance band from
`expected_probe_cost_percent`, and the 10-of-12 harm guard - over the declared
queue precisely (stray complete seeds are excluded with a warning, not
aggregated); no verdict number is hand-computed. As adopted, the declared queue
is 15000-15011 and the control clause below also applies (section 5.5). Every branch of that
decision contract is unit-pinned in `tests/test_v0_verdict.py` (exit codes 0,
1, and 2, the harm-guard boundary, queue-completeness gating of
`gate_4_authorized`, stray-seed exclusion, and the abort conditions), so the
registered rules cannot drift before the T4 run.

### 5.2 GPU execution plan: session budgeting and checkpoint handoff (12-hour Kaggle limit)

Operational companion to section 5.1, pre-committed before launch. Kaggle
facts per plan sections 5 and 40: 12-hour GPU sessions, 20 GB auto-saved
`/kaggle/working`, additional scratch that does not survive the session,
notebook outputs publishable as versioned datasets, and the rule that a
session must never contain irreplaceable state.

**Fit and footprint (validated arithmetic, fp32).** Per GPU: parameters
1.06 GB + gradients 1.06 GB + AdamW moments 2.13 GB = 4.25 GB resident;
activation checkpointing keeps activations near 0.2 GB, so ~4.5 GB of 16 GB
per T4 with 11+ GB headroom - one arm per GPU, no sharding, and all tensors
persist on their own GPU between events (probe rebuilds run on the arm's
GPU, never crossing devices). Checkpoint payload = parameters + AdamW
moments = 3.19 GB; gradients are not saved, and RNG/genome/counters are
negligible.

**Throughput gate (pre-committed dtype rule).** 10k tokens/s needs 8.1
TFLOPS at the ~6xN_active physical fwd+bwd estimate - 100% of the T4's 8.1
TFLOPS fp32 peak, unrealistic in eager mode. Session 1 therefore opens with
a 50-step benchmark of the declared model: if measured fp32 throughput is
below 2,500 tokens/s, the primary config runs in fp16 with GradScaler (T4
has no bf16; the 65 TFLOPS tensor-core peak makes 10k tokens/s comfortable).
Dtype is an execution detail touching no pre-registered criterion (FLOPs
counters book operations, not time), the tolerance band is the safety net,
and the measured dtype is recorded in the published artifact.

**Working-directory ledger (20 GB).** A checkpoint payload is parameters +
AdamW moments = 3.19 GB per arm. Sessions run `max_checkpoints=2`, so an
in-flight pair holds 2 checkpoints x 2 arms x 3.19 GB = 12.76 GB of resume
state, plus the finished pair's final states 2 x 1.06 = 2.13 GB until
publish: steady state ~14.9 GB of 20 GB (5+ GB headroom). `max_checkpoints=3`
would put the pair's resume set at 19.1 GB and overflow with finals (21.3
GB) - pinned at 2. Final states and metrics are wiped from the working
directory after each publish.

**Schedule (arms run in parallel, one per GPU; parameterized by measurement).**
Pair time = 24,576,000 tokens / measured per-arm throughput + ~0.1 h of
event/rebuild overhead, and a session offers ~9.0 h of usable work after
boot, restore, and the publish reserve. A pair is atomic: it is never
started unless it fits entirely, and pairs are never split across sessions
(see deadline rules). Full pairs per session = floor(9.0 / pair time):

| Measured per-arm throughput | Pair time | Pairs/session | Primary sessions | Total incl. secondary |
|---|---|---|---|---|
| 10,000 tokens/s (declared optimistic) | 0.78 h | 11 | 2 | 4 |
| 5,000 tokens/s | 1.47 h | 6 | 2 | 4 |
| 2,500 tokens/s (fp32 floor) | 2.83 h | 3 | 4 | 8 |

The fp16 dtype rule exists to keep the measured point in the upper rows: at
the T4's 65 TFLOPS fp16 tensor peak, 10k tokens/s is ~12% utilization. There
is no fixed reserve session - the deadline rules guarantee every session
ends in a publishable state, further sessions are scheduled until seed 9011
completes, and the table bounds the worst case at 8 working sessions.

**Deadline rules.** (1) Never start a new pair unless the *measured* pair
time + 45 min publish reserve fits the remaining session clock. (2) Remaining
time < 60 min mid-pair: stop at the next step boundary, save both arms,
publish immediately. (3) Remaining time < 15 min: publish only - event
checkpoints are already saved. A pair split across a session boundary leaves
only event checkpoints behind and is restarted from scratch (never resumed)
in the next session, so each seed's 24.576M training tokens come from one
uninterrupted execution per arm.

**Checkpoint handoff (working dir -> versioned dataset).** After each pair
or deadline stop: save each arm's checkpoint at the latest event boundary,
write per-arm metrics and the seed summary JSON, then publish one version
per completed seed to `act-checkpoints/v0-5-1/seed-NNNN/` through the
existing manifest contract (`publish_artifacts` checksums every payload).
Publish is all-or-nothing per seed - the manifest is written last, after all
checksums, so a partial publish is never mistaken for a complete one; per
plan section 38 a restarted session is equivalent to an uninterrupted run,
which the self-check verifies bit-identically on the resume path.

**Dataset ledger and crash bounds.** Versioned datasets accumulate history:
final states for 12 seeds x 2 arms = 25.5 GB, plus per-session safety
snapshots of the latest in-progress event checkpoints (4 payloads = 2 per
arm x 3.19 GB = 12.76 GB per interrupted session; <= 3 expected) -> ~64 GB
worst case, reducible to the 25.5 GB of finals by pruning snapshots of
completed pairs. Restore per plan section 40: locate the latest checkpoint dataset
for the seed, validate checksums, restore, continue. Any crash loses at
most one in-progress pair; an unpublished event interval costs <= 10
minutes of retraining (1500 steps at 10k tokens/s); a crash before a
session's first publish costs at most one pair. A hardware or session
failure counts as infrastructure, not evidence; a lost seed is re-run on
the same seed, never replaced.

**Pre-launch checklist.** (1) `PYTHONPATH=src python
scripts/v0_session_selfcheck.py` passes from the committed revision on the
Kaggle machine. (2) The session 1 benchmark completes with measured
tokens/s recorded and the dtype decision logged in the artifact. (3)
`max_checkpoints=2` and `checkpoint_interval=500` confirmed in the session
config. (4) Seed block untouched (SUPERSEDED: adopted section 5.5 change 4
moves the declared block from 9000-9011 to 15000-15011, with the primary
controller on `spread` and a pool-matched control session); arms
`random,adaptive`; probe budgets cap-4/1-batch (primary) and cap-4/2-batch
(secondary) exactly as pre-registered. (5) Per-session `--output` summary JSONs retained in the
published artifact.

**Pre-launch tooling (built).** Items (2) and (3)'s arithmetic are now code,
so the session runner executes the registered numbers rather than re-deriving
them on the day:

- `scripts/v0_throughput_benchmark.py` - the session-1 benchmark. It times
  the declared model's real step loop (loss forward + backward + AdamW, the
  same op mix and FLOPs accounting as the session runner, activation
  checkpointing on as declared), applies the pre-committed dtype rule, and
  writes the artifact (`artifacts/v0-benchmark.json`) with the measured
  tokens/s, the dtype decision, and the derived schedule. `--profile` runs a
  tiny smoke config for local checks. Verified locally: the declared model
  reproduces 285,336,576 / 134,341,632 parameters and measures
  ~1,736 tokens/s on a GTX 1070 (below the 2,500 tokens/s fp32 floor, so the
  dtype rule correctly selects fp16 there); the T4 decision comes from the
  T4 run.
- `act.training.session_plan` (stdlib-only, unit-tested) - the registered
  schedule arithmetic: the dtype rule, pair time (tokens / throughput +
  0.1 h event overhead), full pairs per session, primary/secondary session
  counts, the 20 GB working-dir ledger (max_checkpoints=2 pinned, steady
  state ~14.9 GB, headroom ~5.1 GB), the deadline decision ladder
  (45 min publish reserve, 60 min boundary stop, 15 min publish-only), and
  the 5.1 predicted probe cost with its [0.5, 2]x tolerance band. The tests
  pin the section 5.2 table rows (10k -> 0.78 h / 11 pairs / 2+4 sessions;
  5k -> 1.47 h / 6; 2.5k -> 2.83 h / 3 / 4+8), so any drift from the
  registered plan fails before launch.
- The declared session config now trains with checkpointed activations
  (`activation_checkpointing=True`, section 5.1), which the 5.2 fit
  arithmetic assumes; the self-check re-verified bit-identical mid-run
  resume with it enabled (blocks carry no dropout, so recompute is
  deterministic).
- `act.training.v0_handoff` - the versioned-dataset checkpoint handoff.
  `publish_seed` stages one completed seed's newest per-arm checkpoints,
  arm metrics, and seed summary under a temporary directory, writes the
  checksummed manifest last, and only then swaps the seed directory into
  `act-checkpoints/v0-5-1/seed-NNNN/` - a partial publish is never
  mistaken for a complete one. `restore_seed` verifies every checksum
  before re-materializing the state into a session directory with the
  original step-named checkpoint filenames, so `run_v0_session` resumes
  exactly where the publish left off; `latest_published_seed` locates the
  newest complete version (a directory without a manifest is not a
  version), and `clean_working_dir=True` reclaims the working copy after a
  publish per the 20 GB ledger rule. Tests cover the atomic re-publish,
  checksum-corruption rejection, and the bit-identical resume of an
  interrupted run through working-dir -> dataset -> working-dir.
- `act.training.v0_runner.run_pair_session` - the session loop that ties the
  three pieces together over the declared seed queue 9000-9011. Registered
  semantics it enforces: a pair is **atomic** (never started unless pair
  time + the 45-min reserve fits the remaining clock - rule 1 is the only
  start gate - and never resumed across sessions; a pair split by the
  boundary restarts from scratch, and any stale working-dir state is
  discarded before a fresh start); the session runner's cooperative stop
  raises at the next checkpoint boundary once the clock passes the reserve,
  and the loop publishes the snapshot as an `in-progress` dataset version
  (forensic only - the next session ignores it and restarts the pair);
  completed pairs publish `complete` versions and reclaim the working dir;
  seeds with a `complete` version are skipped before any gating, so
  "further sessions are scheduled until seed 9011 completes" is literally
  re-running the same  command. The session summary JSON records the queue, the measured
  throughput, per-pair actions (completed / interrupted / skipped), and
  FLOPs counters for the artifact trail.
- The throughput-gated dtype rule is fully executable: the session runner
  trains fp16 sessions under autocast with a GradScaler over fp32 master
  weights (`V0SessionConfig(dtype="fp16")`), and the checkpoint contract
  bumped to version 5 to carry the scaler state so fp16 resume is
  bit-identical (verified by the self-check and unit tests; version-4
  payloads still load in fp32 sessions, and are rejected only under fp16,
  where the missing scaler state would silently change the schedule).
  Dtype books operations, not time: fp16 and fp32 sessions produce
  identical FLOPs counters, so no pre-registered criterion is touched.
- The launch command is one line. `python -m act.training.cli --mode
  pair-session` reads the session-1 benchmark artifact (default
  `artifacts/v0-benchmark.json`), re-derives the dtype decision from the
  measured throughput and refuses a stale or hand-edited artifact (the
  recorded decision must equal the derived one), applies the registered
  `max_checkpoints=2` pin, runs `run_pair_session` over the declared queue
  with the dataset handoff, and writes the session log
  (`artifacts/v0-sessions/session-log.json`). `--tokens-per-second`
  replaces the artifact, `--hours` sets the remaining clock, `--profile`
  smoke-tests the whole path on a tiny config. Adding
  `--kaggle-dataset owner/slug` completes the remote leg of the section
  5.2 handoff (`act.ops.kaggle_remote`), live-verified with real
  credentials: publish -> download -> bit-identical seed round trip
  through the real API (manifest checksums re-verified on return), and
  the registered launch command runs restore -> skip-complete -> exit
  publish with exit 0. Live probes fixed the upload layout (per-seed
  tars with `seed-NNNN/...` internal paths, pushed flat - the service
  auto-extracts archives server-side, and the package's default
  directory handling either drops subdirectories or collides seeds).
  Validated end-to-end: benchmark artifact -> CLI -> 12/12 declared
  pairs published.

### 5.3 Real-data selection diagnostic: neither toy axis survives real text (harm-guard follow-up)

The section 5.1 machinery executed on the real TinyStories token stream
(reduced fidelity, seeds 9000-9011; recorded in `IMPLEMENTATION_STATUS.md`)
ended in `PASS_COST_ONLY_HARM_GUARD_REJECTS`: the cost identity held again
(measured 1.0077% vs 0.9977% predicted), but the adaptive arm lost 11 of 12
pairs. The pre-registered next step (study 5's winner's-curse reading) was
to test the two toy-winning axes on real text. That diagnostic
(`scripts/real_data_diagnostic.py`) ran both variants over the SAME paired
seed block 9000-9007 at steps=600 through the production `run_pair_session`,
with each variant running its own random arm per seed; every variant's random
arm reproduced its baseline counterpart's final loss exactly, so within each
pair only the selection policy differed. There is no significance machinery
here - 8 seeds is a diagnostic sample, not a gate - the numbers below are
paired descriptions.

| Variant | Factor doubled | Config | Measured probe cost | Adaptive wins | Mean delta vs own random | Mean adaptive vs baseline adaptive |
|---|---|---|---|---|---|---|
| var | probe quality (batches) | cap-4 / 4-batch | 3.99% | 0/8 | +0.2858 | +0.0103 |
| rich | candidate richness | cap-8 / 1-batch | 1.98% | 1/8 | +0.1661 | -0.1094 |

Findings:

- **Variance is excluded as the mechanism - and not by the win counts.** In 5
  of 8 seeds the 4-batch argmin chose bit-identically to the baseline
  1-batch argmin (final-loss delta exactly 0.0); where it did differ
  (seeds 9002, 9005, 9006) outcomes were mixed. Quadrupling probe batches
  almost never changed the decision, so the probe ranking is already stable
  at baseline quality: on real text it is stably misaligned, not stochastically
  wrong. Study 6's winner's-curse mechanism (argmin amplifying probe-batch
  noise) cannot explain the real-data failure.
- **Richness operates in absolute terms but not relative to random.** The
  `rich` adaptive arm beat the baseline adaptive arm by a mean 0.109 in final
  loss - the toy richness axis itself still transfers - yet it still lost to
  its own random arm by 0.166. Random probe-argmin candidates are
  systematically better on real text than the probe's best picks.
- **Revised reading of the original harm-guard run.** The study-4-era
  conclusion "selection value on real text needs richer probes or longer
  horizons" is narrowed by this result: neither axis of the toy factor map
  restores the edge at this fidelity. The coherent account is that single-
  batch probe-loss ranking, already weak on synthetic data (rho ~ 0.11 coupled,
  section 9.11 gives 0.078,
  section 3.2), carries an anti-predictive component on real language
  structure that more probe batches do not wash out (stability) and more
  candidates do not offset (relative loss persists).

**Pre-registered next discriminator - RUN: the probe is uninformative, not
inverted.** The argmax control ran on the same paired block through the same
machinery (`V0SessionConfig.probe_selection="argmax"`; only the decision rule
differs - same candidates, same probe batches, same booked probe FLOPs,
measured 1.01% of training FLOPs vs the baseline's ~1.0%). Outcome:
**0/8 adaptive wins, mean delta vs own random +0.3048** - argmax also loses,
and by almost exactly as much as argmin (+0.2858); its adaptive arm sits
+0.0293 from the baseline argmin adaptive arm. Two checks confirm the control
was non-degenerate: on seed 9000 it picked 0/3 identical (operation, layer,
expert) triples to argmin, and the operation balance inverted (argmin chose
20 prune / 4 grow across all events; argmax 11 prune / 13 grow). By the
discriminator contract fixed before the run: both tails of the probe ranking
losing to random by the same amount means the single-batch probe-loss ranking
carries **no candidate-ordering information on real text at this fidelity** -
the signal is uninformative, not inverted, so negating the criterion cannot
help and the fix must come from probe horizon/tokens (more or later probe
data), not from the decision rule. The diagnostic remains outside the gate
framework until a passing configuration is identified and pre-registered on
a fresh seed block.

Raw evidence: `artifacts/real-diag/diagnostic.json` (per-seed paired losses,
all three variants; run log alongside); seed summaries under
`artifacts/real-diag/{var,rich,argmax}/datasets/act-checkpoints/v0-5-1/`.
The `argmax` variant is unit-pinned (`tests/test_v0_session.py`: config
validation and the inverted-decision path), and the mutation-history policy
string (`adaptive_probe_argmin` / `adaptive_probe_argmax`) audits which rule
produced each recorded event.

**Per-candidate rank logging.** The mechanism claims above (pick identity,
operation balance, selected rank) were reconstructed post-hoc from raw
histories. The diagnostic now records them natively: `V0SessionConfig`
gained a `probe_landscape` flag (bookkeeping only; off by default, mirroring
the toy path's instrumentation) that writes every probed candidate's probe
loss into each adaptive event's history, sorted best-first, plus
`probe_selected_index` pointing into that ordering (index 0 = probe-best,
last = probe-worst, so an argmin event selects 0 and an argmax event selects
the tail). The diagnostic enables it for every variant and aggregates a
per-variant `probe_rank` block into `diagnostic.json` (event count, mean
selected rank, rank histogram, near-top rate). Any future mechanism claim -
including the probe-horizon/tokens fix this section motivates - is thus
directly measurable from the published seed summaries rather than inferred.
Pre-instrumentation artifacts report `events_with_landscape: 0` until a
variant is re-run (`--force` moves previous results aside, since the runner
skips completed pairs).

**Fix candidate run - probe-data provenance (`heldout`): pre-registered
hypothesis FALSIFIED (contract outcome b).** The argmax verdict licensed exactly one fix axis: probe
horizon/tokens ("more or later probe data"). Its "more" reading is already
falsified - `var` quadrupled same-distribution probe batches and went 0/8 -
so the next single-axis candidate tests the "different probe data" reading:
the probe windows currently come from the training stream, so the argmin
ranking may be anti-predictive because it scores candidates on text the arm
is about to train on (in-corpus, memorization-adjacent signal that does not
transfer to the arm's future). The `heldout` variant (`V0SessionConfig
.probe_data_source="heldout"`; cap-4/1-batch/argmin, probe batches drawn
from the held-out validation stream via the existing `WindowSampler`
discipline) changes only the probe windows' source stream - same
candidates, same probe batch count, same booked probe FLOPs (baseline-
matched, ~1%), same paired seed block 9000-9007, and both arms' resume
determinism is pinned by test. Pre-registered outcome contract: (a)
heldout-argmin beating its random arm would mean in-corpus probe windows
are the mechanism and probe-data provenance is the fix; (b) heldout-argmin
losing like train-argmin (mean delta ~ +0.29) falsifies provenance as the
mechanism and leaves the probe ranking uninformative under both sources -
at which point the probe's information content, not its data, is the
binding constraint and the remaining licensed lever is a longer probe
horizon (more/later probe tokens, at ~4x probe cost). Either outcome is
decisive through the `probe_rank` instrumentation (mean selected rank and
histogram per variant), so mechanism claims stay directly measurable.
Standard caveats: diagnostic, not a pre-registered gate; 8 seeds
unpowered; TinyStories validation is held-out *stories* from the same
corpus, not a distribution shift - which is exactly the point (probe on
text the arm never trains on). Unit-pinned in `tests/test_v0_real_data.py`
(config validation, real stream-switch at the first event, cost parity,
bit-identical resume).

Heldout outcome (run 2026-09-14, revision `eaa1c45`): **0/8 adaptive wins,
mean delta vs own random `+0.2840`** - indistinguishable from the baseline
argmin's `+0.2858`, with the adaptive arm just `+0.0085` from the baseline
adaptive arm, at the designed 1.01% probe cost; every random arm replicated
its baseline counterpart. The manipulation was real: mean selected probe
loss moved from `4.4195` (train windows) to `4.2848` (held-out windows),
and 9/24 picks changed vs the baseline argmin - more than `var`'s extra
same-distribution batches moved (7/24). Yet the outcomes did not move.
Conclusion: the argmin ranking is a function of the current model and the
candidate set, not of which iid TinyStories windows score it; probe-data
provenance is excluded as the mechanism.

**Fix candidate run - the long-horizon probe as a one-step lookahead
(`long`): pre-registered adaptation-blindness hypothesis FALSIFIED (contract
outcome b).** Every snapshot-probe axis is by
now exhausted: same-distribution probe batches (`var`, 0/8), candidate
richness (`rich`, 1/8), decision rule (`argmax`, 0/8), and probe provenance
(`heldout`, designed). They share one suspect: the probe measures the
*instantaneous* post-mutation loss, while the arm's outcome depends on
*post-mutation trainability* - a freshly initialized expert looks bad
before any adaptation, which structurally biases argmin against grows (the
observed 4/20 prune bias). The one probe-horizon reading that attacks this
directly is extending the probe in time: `V0SessionConfig
.probe_finetune_steps = R` fine-tunes each probed candidate for R AdamW
steps (session learning rate, fresh optimizer, no momentum carried in) on
freshly drawn training batches before re-scoring the SAME probe batches,
so the ranking becomes a trainability comparison. The diagnostic runs it
at R=1: each lookahead step adds ~2% of training FLOPs (R full-model
steps per candidate under the section-5.1 identity), so R=1 lands ~3.0%
while R=2 sits exactly at the 5% criterion with no margin - the variant
runs at R=1 or not at all. Resume determinism holds: fine-tune batches
come from a dedicated throwaway generator seeded per event
(`seed ^ 0xF1A7 ^ step * 0x9E3779B9`), redrawn exactly whenever an event
re-executes, never checkpointed. Pre-registered outcome contract: (a)
`long` beating its random arm on the paired block would mean the snapshot
probe's adaptation blindness is the mechanism and lookahead selection is a
fix candidate worth pre-registering on a fresh seed block; (b) `long`
losing like the baseline (mean delta ~ +0.29) would mean one adaptation
step does not restore ranking information, leaving probe token count at
fixed fidelity as the remaining licensed lever. Secondary observable: the
selected operation balance - a `long` argmin that still picks ~5:1
prune:grow while winning would show the prune bias itself was not the
failure mode. Unit-pinned in `tests/test_v0_session.py` (config
validation, lookahead cost identity, adaptation actually re-ranking the
landscape, bit-identical resume).

Long outcome (run 2026-09-14, revision `eaa1c45`): **0/8 adaptive wins,
mean delta vs own random `+0.3106`** (baseline argmin: `+0.2858`), the
adaptive arm `+0.0350` from the baseline adaptive arm, at the designed
3.00% probe cost. The lookahead did exactly what it was designed to do at
the ranking level: every landscape re-ranked after adaptation, the
operation balance relaxed from 4/24 to 5/24 grows, and on seed 9000 the
probe top-ranked the grow candidate at 2 of 3 events (the snapshot probe
ranks grows last - the prune bias is a ranking artifact of the snapshot
probe). But re-ranking did not re-rank *toward what helps*: per-seed
deltas stayed inside the historical harm band, and across seeds the
correlation between mean selected probe loss and final harm is POSITIVE
(`+0.51` for `long`, `+0.89` for `heldout`; 8 points, indicative not
established) - seeds whose argmin picked higher-probe-loss candidates were
harmed more. One adaptation step restores ranking dynamics, not ranking
validity.

**Unified verdict after five diagnostic variants.** The probe ranking is
real, reproducible, and adaptive - stably bit-identical under extra
batches (`var`), coherent under a different window source (`heldout`),
re-ranking sensibly under lookahead (`long`) - but its ordering is
unrelated or inverse to downstream value: argmin loses at every fidelity
and source (+0.28 to +0.31), argmax loses (+0.30), and the only variant
with any win (`rich`, cap-8, whose 8/24 grow share matches the random
arms') is also the one that most dilutes the ranking's influence. Every
horizon/token reading of the section-5.3 licensed fix axis - more batches,
different source, lookahead in time - is now exhausted, and the one
untested reading (more probe tokens at fixed fidelity) would sharpen a
signal these runs show to be anti-correlated with outcome; it is not
recommended without a new mechanism hypothesis. The observationally
motivated next direction is a different axis - changing how the ranking
is *used* (coarser granularity such as layer-level selection, or
accept/reject gating of random draws rather than argmin over candidates) -
which sits outside the section-5.3 contract and requires its own
pre-registration. All five variants' paired results, probe-rank blocks,
and per-event landscapes are in `artifacts/real-diag/diagnostic.json`
(run log `run-heldout-long.log` alongside); 8-seed power caveats apply
throughout.

### 5.4 Pre-registration: accept/reject gating of random draws (RAN - outcome in section 5.7)

**Motivation.** Section 5.3's unified verdict locates the failure in the
probe's candidate ORDERING, not in its measurements: the same probe
readings that are anti-predictive when ranked are still real, reproducible
measurements of post-mutation loss on the actual training stream. The
`rich` variant adds the sharper hint - the only configuration that ever
beat its random arm is the one that most dilutes the ranking's influence.
So section 5.4 keeps the random arm's draw distribution (which is fine:
random arms never lose to random) and adds exactly one binary decision per
event: does the drawn mutation hurt the CURRENT model, measured by the
probe's ABSOLUTE signal?

**Semantics (implemented in `V0SessionConfig.arms=("gated",)` and
`_gated_event`).** Each event: (1) draw one random mutation with the
random arm's exact chooser semantics - first, so its internal load-fallback
reads the same post-training routing statistics the random arm sees; (2) if
the draw changed nothing, record a no-op (no probes booked, budget slot
consumed, identical to the random arm's wall); (3) otherwise score the
two-point contrast on the same probe batches - the no-op reference (current
model) and the drawn mutation (one rebuild, which is also the accepted
winner, so the global-RNG stream stays in lockstep with the random arm's
one-rebuild-per-accepted-event pattern); (4) accept the draw unless
`draw_probe_loss > reference_probe_loss + gate_margin` (theta), in which
case roll back and apply the draw's own load-based grow fallback so the
mutation budget stays matched (publish refuses unmatched budgets); a
dead-end fallback is a recorded no-op. Gated sessions are single-arm by
definition and publish alongside the baseline block's stored random arms.
Every event records `gate_decision` (`accept` / `reject_fallback` /
`reject_noop` / `noop`), both probe losses, and theta, so accept rate and
draw-vs-reference contrast are directly measurable.

**Built-in control.** The gated arm shares the random arm's init identity
(offset 1), training stream, and chooser seeds; at `gate_margin = inf`
every draw is accepted and the arm reproduces the random arm BIT-FOR-BIT
(pinned by test, including FLOPs: two probe evaluations booked per live
event, nothing on no-ops). Any finite theta therefore differs from random
only where the gate intervenes - the cleanest possible budget-matched
baseline.

**Cost.** Two probe evaluations per live event at the baseline's ~1%
per-event booking lands the gated arm at roughly 2% of training FLOPs -
half the adaptive arm's cost, well under the 5% criterion.

**theta justification.** `theta = 0` (accept only draws whose probe loss
does not exceed the reference's) is the maximally conservative point of
the axis and the cheapest falsification test: if even rejecting only
strictly-harmful draws cannot beat random, the absolute signal is empty
and the axis is dead; if it wins, intermediate theta (accepting small
probe regressions for architecture gains the probe cannot see) is a
follow-up sweep, pre-registered separately.

**Pre-registered outcome contract (8 paired seeds 9000-9007, revision
eaa1c45 protocol, ~1.2 h on the local GPU).**
- **(a) `gated` (theta=0) beats its paired random arms** - mean delta < 0
  or wins >= 5/8: absolute-harm gating carries real signal the ranking
  lacks; pre-register an intermediate-theta sweep and a power block on a
  fresh seed range.
- **(b) `gated` loses or ties like the baseline argmin** (mean delta in
  the +0.28..+0.31 band): the absolute signal is as empty as the ordering,
  and the probe-on-real-text program is concluded as falsified in both of
  its usable forms; the next direction moves away from probe-derived
  selection entirely (e.g. routing-statistics heuristics).
- Secondary observables: accept rate (a near-1 rate would make the arm an
  expensive random replica and would itself falsify the gate's
  discriminativeness), mean draw-minus-reference contrast, and operation
  balance under rejection (rejections should skew prune-accepts and
  grow-fallbacks if the snapshot bias shows up in absolute form).

**Outcome (run 2026-09-14, revision `ef4bacd`): outcome (a)'s letter fired
but its content was never exercised - full write-up in section 5.7.**

### 5.5 Revision to the 5.1 protocol (ADOPTED - the operative protocol)

The substrate audit (section 9) invalidated a premise of section 5.1, so the
protocol could not launch as written. The original text above is left intact
for the audit trail; **this subsection is the adopted revision and supersedes
it**: section 5.1's cost criteria, budget, model and decision rules stand
exactly as registered, and only the four changes below apply. Where the two
readings of section 5.1 differ, this one governs.

**Why a revision is needed.** Section 5.1 is a *cost* protocol whose value
depends on the controller under measurement being the validated one. Two
findings break that:

1. The registered primary controller uses `probe_candidate_policy="prefix"`
   over `feasible_expert_mutations`, which is layer-major. At the declared
   depth (12 layers x 8 experts = 108 candidates) a cap of 4 reaches only
   layer 0 and its first three prunes, so the declared search would explore
   one layer out of twelve. The V0 session now exposes `spread` for exactly
   this reason.
2. The quality evidence is a toy-substrate result carrying a real pool
   component. The registered depth-4 confirmation (section 9.9) gives the
   registered comparison 255/384 against uniform churn while a loss-blind
   same-pool control gets only 209/384 (p `0.046`, failing alpha/3), so the pool
   accounts for ~32% of the mean gain, and the probe ordering adds a decisive
   increment on top (236/384, p `4.1e-06`). The mechanism is real; the effect
   size measured against uniform churn is nonetheless inflated by the pool, so
   the protocol should price the controller against a pool-matched control.
3. **The declared seed block is spent, so the protocol can no longer be a blind
   measurement.** Section 5.1 reserves "fresh seed pairs (9000-9011)". They are
   not fresh: the section 5.3 real-data diagnostic executed the section 5.1
   machinery over the *entire* queue at reduced fidelity (steps=600, TinyStories)
   through the same `run_pair_session`, and a section 5.1 verdict has already
   been computed over those seeds - `artifacts/real-proof/datasets/
   v0-5-1-verdict.json` reads `PASS_COST_ONLY_HARM_GUARD_REJECTS`, measured cost
   `1.0077%`, `gate_4_authorized: false`, with the adaptive arm losing 11 of 12
   pairs. Seed 9000 is additionally the single-seed default of `act.training.cli
   --mode v0`, so it is the smoke seed. The consequence is specific and
   disqualifying: the harm guard is one of the three branches that decide the
   verdict, and its outcome on this block is already known. Re-running the same
   declared queue at full fidelity after observing a harm-guard rejection is
   exactly the "measure again until it passes" pattern the rest of this
   protocol is built to prevent, and the seed-level pair difficulties are also
   no longer novel. The prescribed remedy is not to reinterpret the old run: it
   is to move the declared block to genuinely unused seeds and keep the reduced-
   fidelity diagnostic labelled as a diagnostic (which it already is).

**Proposed changes (design only; every registered threshold is unchanged).**

1. Primary controller uses `probe_candidate_policy="spread"` (cap 4 drawn
   from the whole feasible set). This changes no cost term: the section 5.1
   identity `S = E * k * Bp * 2 * N_active` still holds with the same E, k and
   Bp, so the declared prediction `C = 0.094%` and the `[0.5, 2]x` band
   `[0.047%, 0.189%]` carry over unchanged. The 5% criterion, the band, and
   the 10-of-12 harm guard are all untouched, as is the no-re-tuning rule.
2. Add a pool-matched control arm (the same candidates, same probe cost, and
   a loss-blind uniform choice - `probe_selection="random"`) on the same 12
   seed pairs. Its per-seed quality is reported descriptively.
3. Add one interpretation clause, in the same spirit as the harm guard: if
   the pool-matched arm's quality is indistinguishable from the adaptive
   arm's, record the cost PASS but withhold the Gate 4 recommendation, since
   the quality premise would then be unsupported. This is a reporting rule,
   not a new threshold.
4. **Re-reserve the declared seed block to 15000-15011** (12 pairs, verified
   unused across `src/`, `scripts/`, `tests/` and every documented block:
   1000-8000 factor map, 5000-6000 cadence/variance, 9000-9011 spent above,
   20000-20127, 30000-30011, 40000-40127, 50000-50383, 60000-60383). Nothing
   else about the queue changes: 12 pairs, the same per-pair protocol, the same
   5% criterion, band, harm guard and no-re-tuning rule, all now genuinely
   blind. `scripts/v0_verdict.py` gained a `--seeds "A-B"` flag for exactly
   this case (`compute_verdict`'s `declared_seeds`, default the registered
   queue, so the registered behaviour is untouched until the flag is used and
   no criterion is loosened by it - unit-pinned in `tests/test_v0_verdict.py`).

**Explicitly unchanged.** The declared V0 model
(285,336,576 / 134,341,632), the budget D = 24,576,000 tokens per arm, E = 3
events, the planned/measured test sizes, the 5% criterion, the tolerance band,
the harm guard, and the rule that no knob is re-tuned against the measured C.
The one registered value that does move is the seed block, for the reason in
point 3 above.

**Status: ADOPTED** (signed off 2026-09-14) and implemented, so it now requires
execution rather than a decision.

Adoption turned out to need a small code change after all, all of it additive:

- `v0_runner.DECLARED_SEEDS` and `v0_verdict.DECLARED_SEEDS` moved to
  **15000-15011** (change 4), taking the benchmark's reserved data seed to
  15012 and the lone-session default seed with them.
- `act.training.cli --mode pair-session` gained `--probe-candidate-policy` and
  `--probe-selection` (changes 1 and 2), both defaulting to the registered
  values, so an unflagged launch still reproduces the original registration.
- `scripts/v0_verdict.py` gained the pool-matched control block and the
  interpretation clause (change 3), plus `--control-root` and the `--seeds`
  override. It also gained `POOL_CONTROL_WIN_LIMIT = 10`: the clause fires when
  the loss-blind arm clears the **same** 10-of-12 bar the harm guard uses, so
  "indistinguishable" is a pre-committed rule rather than a judgement call, and
the failure modes are distinguished by label (`PASS_COST_ONLY_POOL_MATCHED_
CONTROL_PASSES` vs `PASS_COST_ONLY_CONTROL_MISSING`, both withholding Gate 4 at
exit 2 while the cost PASS stands). A missing control is disqualifying rather
than silently skipped.

Its two evidentiary premises are now registered results: the section 5.6
depth-4 confirmation passed out-of-sample including the pool-matched control
(section 9.9), and point 3's spent block is a fact about the repository rather
than an inference.

**Launch (adopted).** The primary run is
`--probe-candidate-policy spread --seeds 15000-15011`; the pool-matched control
is the same command with `--probe-selection random` over the same seeds into its
own dataset root; the verdict is `scripts/v0_verdict.py --dataset-root <primary>
--control-root <control>`. `T4_RUNBOOK.md` carries the exact commands. The registered defaults remain
`prefix`/`argmin`, so the original protocol stays runnable; the revised one is
reached by setting `probe_candidate_policy="spread"`. The pool-matched control
needs no new arm: a session's decision rule is session-level config, so the
control runs as a second session over the same seed pairs with
`probe_selection="random"`, paired against the shared baseline random arms -
exactly how `scripts/real_data_diagnostic.py` already pairs its variants. The
only genuinely new reporting surface is the interpretation clause in change
(3), which is text, not machinery.

### 5.6 Pre-registration: depth-4 Gate 3 confirmation on a fresh seed block

**Role.** Section 9.7 is the project's strongest substrate result - the
registered comparison and the controlled selection increment both pass on a
depth-4 pool with the arm coupling fixed - but it is *exploratory and
unregistered*, measured on a block that was itself chosen after the audit. This
study converts it into registered evidence by freezing the design here and
running it once on a **fresh, never-observed seed block**. It is the entry
condition for both section 5.5 (whose revision is justified by the same result)
and any Gate 4 / compute spend.

**Hypothesis.** On a substrate deep enough to express a topology search, (a)
probe-loss selection beats matched-budget uniform topology churn, and (b) the
probe ordering carries a value increment over a loss-blind choice from the
*identical* candidate pool at the *identical* probe cost.

**Substrate (frozen).** Toy generator, `depth=4`, `hidden_size=24`,
`vocab_size=32`, `expert_count=4`, `active_experts=2`, `sequence_length=12`,
`batch_size=2`, `steps=48`, `mutation_interval=12`, `mutation_budget=3`,
`probe_candidate_policy="spread"`, `probe_candidate_cap=4`,
`probe_batch_count=2`, `independent_arm_rng=True`, arms `random,adaptive`.
Feasible candidate set at the initial genome is 20 (4 layers x {2 grows, 3
prunes}), so a cap of 4 is a real budget over the space rather than a layer-0
prefix - the property section 5.5 depends on.

**Seeds.** Fresh block **60000-60383** (384 pairs), disjoint from every
previously used block: 1000-8000 (factor map), 5000-6000 (cadence/variance),
9000-9011 (section 5.1 queue), 20000-20127 (depth-4 policy matrix),
30000-30011 (reduced-dim V0 check), 40000-40127 (policy ablation),
50000-50383 (section 9.7 powered block). No seed in this block has been run in
any configuration before, so the result is out-of-sample for every prior number.

**Comparisons.** All three are computed on the same 384 seeds; every decision
rule is a run of the same substrate with the rule as the only difference.

| # | Comparison | Role | Test |
|---|---|---|---|
| 1 | `argmin` vs uniform churn | primary (the registered Gate 3 claim) | one-tailed exact binomial |
| 2 | `random`-in-pool vs uniform churn | pool/location contribution | one-tailed exact binomial |
| 3 | `argmin` vs `random`-in-pool (paired, same pool & cost) | the selection increment | one-tailed exact binomial on paired deltas |

**Thresholds (declared before the run).** At n=384 the minimum passing win
counts are **209/384** at alpha=0.05, and **214/384** at the Bonferroni level
alpha/3 = 0.01667 used for the three comparisons. (For reference, the depth-1
study 3 result of 213/384 clears alpha=0.05 but misses alpha/3 by one seed.)

**Decision rules (frozen).**

- **PASS** - comparisons 1 *and* 3 clear alpha/3, and comparison 2 is either
  significant or not (its value only bounds the pool contribution): Gate 3 is
  established on a non-degenerate substrate, section 5.5's justification is
  confirmed, and the Gate 4 path opens.
- **POOL-DRIVEN (withdraw the selection claim)** - comparison 1 clears alpha/3
  but comparison 3 does not: most of the effect is candidate placement, the
  probe-ordering mechanism is not supported, and the plan's "stop and
  simplify" rule applies to the search design.
- **NOT REPRODUCED** - comparison 1 fails alpha/3: the depth-1 passes are
  recording-space artifacts of a degenerate 5-candidate space, Gate 3 is open,
  section 5.5 is **not** adopted, and the section 5.1 T4 run must not launch
  (its cost measurement would price an unvalidated controller).
- **VOID (validity control)** - the inverted-signal control (`argmax` vs
  uniform churn) passes alpha/3 on the same seeds: the probe signal carries no
  ordering information at this scale and the study says nothing either way.

**Secondary observables (reported, not gating).** Random-arm invariance must be
384/384 (it is the decoupling guarantee, and any other value means the run is
invalid); mean adaptive-minus-random and paired mean deltas with paired-t
p-values; per-arm search FLOPs and the search/training cost ratio; and the
count of seeds on which `argmin` and `random`-in-pool choose the identical
candidate.

**Reproduce (one line, after this file is committed).**

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m act.evolution.cli --steps 48 \
  --depth 4 --expert-count 4 --probe-candidate-policy spread \
  --probe-candidate-cap 4 --probe-batch-count 2 --independent-arm-rng \
  --probe-selection argmin --arms random,adaptive \
  --seeds $(seq -s, 60000 60383) --output artifacts/depth4-confirm.jsonl
```

and the same command with `--probe-selection random` for comparison 2, and with
`--probe-selection argmax` for the validity control. Reproduction note: the
candidate policy and the pool-random draw are seeded from `(seed, step)` and the
arm RNG is per `(arm, event)`, so these runs are deterministic on CPU and
re-runnable bit-identically.

**Status.** Registered here (committed as `b876ac1`, *before* the run, per the
project's pre-registration rule); executed once on the declared fresh block.
**Result: PASS** - comparisons 1 (255/384, p `6.1e-11`) and 3 (236/384,
p `4.1e-06`) both clear alpha/3, the pool-matched control clears only alpha
(209/384, p `0.046`), and the validity control does not pass. Full write-up in
section 9.9.

### 5.7 Gated outcome: the first real-text win, won by the null arm (2026-09-14)

The section-5.4 gated arm (theta = 0) ran once on the paired block (seeds
9000-9007, steps 600, revision `ef4bacd`; run log
`artifacts/real-diag/run-gated.log`, aggregates in
`artifacts/real-diag/diagnostic.json`). The headline: **8/8 wins, mean delta
vs own random `-0.1217`** (per-seed deltas `-0.23` to `-0.70`), the adaptive
arm `-0.3972` from the baseline adaptive arm - the first negative real-text
delta of any treatment - at 0.50% probe cost. Every secondary observable,
however, says the mechanism is degenerate: **accept rate 0/24** (100%
reject), mean draw-minus-reference `+0.1065` (min `+0.0097`, max `+0.9154`,
harmful in 24/24), and `mutation_count = 0` in 8/8 seeds with a constant
architecture hash. The gate vetoed every draw, and at the real-data config's
`expert_count = 8` the load-based fallback's `< 8` guard cannot fire, so all
24 decisions are `reject_noop`: the gated arm trained its initial
architecture for 600 steps while the random arm performed its three
mutations.

The comparison is therefore budget-matched churn versus NO churn, and the
built-in control proves it: the draws were bit-identical to the random
arms' own operations (24/24 - on seed 9000 the gate vetoed exactly grow L7 /
prune L2 / prune L7), so the only treatment difference was suppression.
The honest reading has three parts. (i) This is the first direct measurement
of churn harm on real text: at this fidelity, doing nothing beats random
churn by `-0.12` nats. (ii) It retroactively explains part of every earlier
variant's loss - argmin, argmax, `var`, `rich`, `heldout`, and `long` all
spent the full mutation budget by design, while a controller that mutates
less (including "never") beats the random arms. (iii) It does NOT validate
the gate: nothing was ever accepted, so the accept path was never exercised
by the data, and a controller that always rejects would have produced the
same 8/8. Outcome (a)'s pre-registered letter ("mean delta < 0 or wins
>= 5/8") fired; its content ("absolute-harm gating carries real signal the
ranking lacks") did not.

Follow-ups, each requiring its own pre-registration: (1) a do-nothing
control arm (zero mutations, budget matched as no-ops) on a fresh seed
block, to confirm the `-0.12` is pure churn harm rather than seed luck -
now pre-registered as section 5.8; (2) an intermediate-theta sweep - the recorded contrasts make the acceptance
windows concrete (theta `0.05` would have accepted 11/24 of the observed
draws, theta `0.1` would have accepted 21/24) - whose real question is not
"gated beats random" (the null arm already does that) but "accepted draws
beat doing nothing"; with draw-minus-reference >= 0 in 24/24, large theta
approaches the random arm and the informative window is narrow. Eight-seed
power caveats apply throughout.

### 5.8 Pre-registration: the do-nothing control arm (IMPLEMENTED as the `null` arm; results in sections 5.9, 5.10 and 5.15)

**Motivation.** Section 5.7's follow-up (1). The gated outcome's 8/8 win
was produced by an arm that performed zero mutations - either churn harm
is real on real text, or that block's pairing got lucky. The question is
separable from the gate entirely: run the arm the gated arm degenerated
into, with the gate (and its probe cost) removed.

**Semantics (implemented in `V0SessionConfig.arms=("null",)`).** The null
arm is a single-arm session that shares the random arm's init identity
(offset 1), training stream, and event cadence, but performs NO mutation
at any structural event: each event appends one recorded no-op
(operation `"null"`), books no probe FLOPs (`search_flops` stays exactly
0), and consumes the budget slot. At the diagnostic schedule (steps 600,
interval 200) there are exactly 3 event slots, matching the baseline
random arms' 3 mutations. There are no tunable parameters.

**Block choice.** The study runs on the already-published paired block
9000-9007 by design: the comparison pairs the null arm against those same
seeds' stored baseline random arms, and a do-nothing arm has no knobs to
retune, so reusing the block cannot reproduce the "measure again until it
passes" pattern the pre-registration rule exists to prevent.

**Built-in checks (must hold, machine-checked from the published
summaries).** Zero search FLOPs and zero mutations in every seed; and the
replication check of the section-5.4 run - because the gated arm accepted
nothing, the null arm's final losses should reproduce the gated run's
final losses on the same seeds (in principle bit-for-bit: same init seed,
same training batches, identical parameters throughout, and the gated
arm's rejected rebuilds touched no persistent state). The verdict script
reports the max |null - gated| per-seed gap; a gap above 1e-4 is a
finding about the environment, not the hypothesis (section 9.11's
portability caveat).

**Pre-registered outcome contract (8 paired seeds, ~0.14 h/pair).**
- **(a) the null arm beats the paired random arms** (mean delta < 0 or
  wins >= 5/8): churn harm is real on this block. Every real-data
  variant's loss is then re-read as partly churn harm (all spent the full
  budget by design), and the gate follow-up question becomes whether
  ACCEPTED draws beat doing nothing - not whether gating beats random.
- **(b) the null-vs-random delta sits within +/-0.1 nats** (the gated
  delta's magnitude, unresolved at 8 seeds): the gated run's `-0.12` was
  seed luck; the gate program stays closed and the churn-harm reading is
  retracted.
- **(c) the null arm LOSES to the random arms** (mean delta > 0): churn
  is net helpful at this schedule - this would falsify the churn-harm
  reading of section 5.7 and demand a re-examination of the gated result.

**Cost.** Zero probe FLOPs; one arm per session, so pair time matches the
gated run's (~2.5 min on the local GPU).

**Outcome (run 2026-09-14, revision `9e5e448`): outcome (a) - churn harm is
REAL on this block.** 8/8 wins, mean delta `-0.1217` (per-seed deltas
`-0.0804` to `-0.2003`, every seed negative; verdict machine-checked by
`scripts/null_verdict.py`). Both built-in checks held: zero mutations and
zero search FLOPs in 8/8 seeds, and the replication check is exact - max
|null - gated| = `0.000000` across all eight seeds, the null arm
reproducing the gated run bit-for-bit as the degeneracy reading of section
5.7 predicted (the gated arm's vetoed rebuilds genuinely left no residue,
and the determinism contract survives a fresh process).

Consequences. (1) The `-0.12` is not seed luck: at this schedule on real
text, matched-budget random churn costs ~0.12 nats against doing nothing.
(2) Every real-data variant's loss in section 5.3 is re-read as partly
churn harm - argmin, argmax, `var`, and `heldout` spent the full budget
for deltas of `+0.17` to `+0.31`, i.e. roughly the churn cost plus a
directly harmful selection, while `rich`'s `+0.166` sits near the churn
line. (3) The gate follow-up question is now sharp: not "does gating beat
random" (doing nothing already does) but "do ACCEPTED draws beat doing
nothing" - the intermediate-theta sweep's control arm should be the null
arm, not random. (4) For the eventual Gate 4 protocol, the random arm is a
weak baseline on real text: a controller must now clear the null arm, and
the honest ladder is null < random < adaptive-at-theta. Eight-seed power
still applies; a powered confirmation on a fresh block (or at the toy
scale where 384 seeds are cheap) is the next pre-registration if this axis
is pursued.

### 5.9 Pre-registration: does the toy adaptive edge clear the do-nothing bar? (RAN - suite `null_toy_study.py`, summary `null-control-depth1-384.json`)

**Motivation.** Sections 5.7-5.8 established real-text churn harm and showed
a do-nothing arm beating random churn there. Every toy Gate-style pass -
including the section 5.6 depth-4 confirmation - was scored against random
churn, and the toy path never had a null arm, so "adaptive > random" has
never been tested against "adaptive > nothing". If churn is also harmful on
the synthetic substrate, part of every toy pass is churn suppression and the
pool-matched control of section 9.5 needs a third control.

**Design.** Study 3's registered config (hidden 24, depth 1, expert_count 4,
active 2, steps 48, interval 12, budget 3, cap 4 / 2 probe batches,
`adaptive_policy=probe_loss`), run as a THREE-arm block per seed -
`arms="random,adaptive,null"`, `independent_arm_rng=True` (pinned explicitly
per the section 9.8 convention), `device=cpu` - on the never-observed fresh
block **70000-70383** (384 seeds). Arm order keeps indices 0/1 on
random/adaptive, and the null arm consumes no global torch RNG, so the
adaptive-vs-random comparison is by construction identical to the two-arm
block (pinned by test: 3-arm == 2-arm bit-for-bit on anchor seed 3000; the
committed decoupled recheck's absolute constants are environment-sensitive
at 1e-6 per section 9.11, so the invariance is pinned within-environment).
Both comparisons below come from the same fresh run - no committed numbers
are mixed with fresh ones.

**Pre-registered outcome contract (alpha = 0.05, two primaries, Bonferroni
alpha/2 = 0.0025; verdict machine-checked, exact binomial via
`act.evolution.study`).**
1. **Null-vs-random** (churn-harm replication at toy scale): one-tailed
   sign test on per-seed null-minus-random deltas (null < random = churn
   harm). Pass = clears alpha/2.
2. **Adaptive-vs-null** (the new bar for Gate 3 claims): one-tailed sign
   test on per-seed adaptive-minus-null deltas (adaptive < null = the
   controller beats doing nothing). Pass = clears alpha/2.
- Secondary (reported, not gating): adaptive-vs-random on the same run
  (expected to replicate the decoupled direction); mean deltas of all three
  pairings; per-arm search FLOPs (null must book exactly 0) and mutation
  counts (null exactly 0, random/adaptive exactly the budget).

**Interpretation grid (pre-registered).**
- 1 and 2 both pass: the toy edge is real beyond both the pool effect and
  churn suppression; the factor map stands with an added control.
- 1 passes, 2 fails: toy churn harm is as large as the adaptive edge - the
  Gate 3 passes are substantially churn suppression and the factor map must
  be re-derived against the null arm.
- 1 fails (no toy churn harm): the toy substrate differs qualitatively from
  real text on this axis; comparisons against random churn remain meaningful
  at toy scale, and the null control stays a real-data requirement.
- 2 fails with adaptive LOSING to null: the strongest form of the section-1
  "stop and simplify" rule applies.

**Cost.** ~1.5-2 s/seed at depth 1 -> 384 seeds in ~13 min single-core,
sharded 20-way well under 2 minutes wall (CPU per the section 5.6 hardware
note).

**Outcome (run 2026-09-14, revision `78186e4`): interpretation branch 2 -
1 PASS / 2 FAIL.** Verdict machine-checked (`scripts/null_toy_study.py
verdict`; committed record
`artifacts/study-summaries/null-control-depth1-384.json`, recomputable
from its per-seed triples and pinned by test):
built-in checks held in 384/384 seeds (mutations 3/3/0, zero null search
FLOPs). **Primary 1 PASS**: null-vs-random 280/384, mean `-0.0795`, p
`4.69e-20` - churn harm replicates on the toy substrate. **Primary 2 FAIL
decisively**: adaptive-vs-null 117/384, mean `+0.0508`, p `1.0` - on
study 3's config the controller LOSES to doing nothing. Secondary:
adaptive-vs-random 225/384, mean `-0.0288`, p `4.43e-04`, replicating the
decoupled study-3 direction on a fresh block (222/384 on the committed
block). Per-arm mean losses: null `15.0745` < adaptive `15.1253` < random
`15.1540` - the ordering is null < adaptive < random. Reading: the
study-3 "adaptive edge" is directional and replicable, but it is churn
SUPPRESSION - a controller that mutates more sensibly than random still
ends up worse than never mutating at all. Study 3's factor-map row must
now be quoted as "adaptive < random, both > null". The registered
section-5.2 config is depth 1, the degenerate 5-candidate space of
section 9.5; the open question is whether the section-5.6 depth-4 pass
(spread policy, a 20-candidate pool genuinely larger than the cap)
clears the do-nothing bar - pre-registered next as section 5.10.

### 5.10 Pre-registration: the depth-4 confirmation against the null arm (RAN - summary `null-control-depth4-384.json`)

**Motivation.** Section 5.9's branch 2 leaves the project's current Gate 3
evidence - the section-5.6 depth-4 confirmation (spread policy, cap 4 over
a 20-candidate pool, `independent_arm_rng=True`) - as the last pass not yet
faced with a do-nothing control. Its registered comparisons beat random
churn and a loss-blind same-pool choice, but both controls still churn; if
the null arm beats them all, the pool story collapses too.

**Design.** The section-5.6 config (study-3 defaults with depth 4 and
`probe_candidate_policy="spread"`), run as the same three-arm block per
seed (`arms="random,adaptive,null"`, `independent_arm_rng=True`,
`device=cpu`) on the never-observed fresh block **71000-71383** (384
seeds). Same machinery as section 5.9 (the `depth4` profile of
`scripts/null_toy_study.py`); same pre-registered primaries and
Bonferroni alpha/2 bar, plus the section-5.6 secondaries as secondary
observables (loss-blind pool comparison is NOT re-run here - it is
scored in section 5.6 and the pool story is only re-opened if primary 2
fails).

**Interpretation grid.**
- Primary 2 passes (adaptive < null clears alpha/2): the depth-4 edge is
  genuine value creation beyond both the pool effect and churn suppression;
  Gate 3 stands on the depth-4 substrate.
- Primary 2 fails: the project's Gate 3 evidence is churn suppression at
  every tested fidelity - the honest reading becomes "no topology controller
  tested so far creates value; several mutate less harmfully than random"
  and the section-1 "stop and simplify" rule governs the next step.
- Primary 1 fails (no depth-4 churn harm): informative either way - it
  would mean churn harm is fidelity/config-dependent, weakening the
  real-text transfer of the toy null result.

**Cost.** ~1.5-2 s/seed at depth 4 on CPU (measured section 9.3), sharded
20-way ~1 min wall.

**Outcome (run 2026-09-14, revision `a9632a6`): interpretation branch 1 -
both primaries PASS.** Verdict machine-checked (the `depth4` profile;
committed record
`artifacts/study-summaries/null-control-depth4-384.json`, pinned by test
like the depth-1 record); built-in checks held in 384/384 seeds. **Primary 1 PASS**: null-vs-random 254/384,
mean `-0.0833`, p `1.22e-10` - churn harm exists at depth 4 as well.
**Primary 2 PASS**: adaptive-vs-null 218/384, mean `-0.0297`, p
`4.58e-03` - the depth-4 controller beats doing nothing, out-of-sample,
clearing Bonferroni alpha/2. Secondary: adaptive-vs-random 263/384, mean
`-0.1130`, p `1.68e-13`, replicating the section-5.6 registered direction
on a fresh block (255/384 on the committed one). Per-arm mean losses:
adaptive `12.4289` < null `12.4586` < random `12.5419` - the ordering
FLIPS relative to depth 1. Reading: of the raw adaptive-vs-random edge at
depth 4 (`0.1130`), about `0.083` is churn suppression and pool effects,
and about `0.030` (~26%) is genuine value creation that survives the
do-nothing bar. Combined with section 5.9, the factor map's summary is now
sharper: at depth 1 the probe-loss controller suppresses churn but destroys
no value (null < adaptive < random); at depth 4 with the spread policy it
creates real value on top of not churning harmfully (adaptive < null <
random). The richness axis separates suppression from creation - coherent
with sections 5.3-5.6 - and Gate 3's depth-4 evidence stands, now with a
third control under it. (Revised by section 5.12: the depth x policy 2x2
shows DEPTH, not the candidate policy, is the separating factor - the
attribution above is confounded and superseded.)

### 5.11 Pre-registration: the intermediate-theta sweep against the null arm (RAN - summary `theta-sweep-9008-9017.json`; `IMPLEMENTATION_STATUS.md`)

**Motivation.** Section 5.7's follow-up (2), now correctly aimed: the
question is not "does gating beat random" (doing nothing already does,
section 5.8) but "do ACCEPTED draws beat doing nothing". The recorded
section-5.4 contrasts make the acceptance windows concrete - theta `0.02`
would have accepted 4/24 observed draws, theta `0.05` accepted 11/24,
theta `0.1` accepted 21/24 - so the sweep samples the conservative,
middle, and near-random points of the axis.

**Design.** Single-arm sessions on the V0 diagnostic path (steps 600,
interval 200, budget 3, cap 4 / 1 probe batch) on the fresh block
**9008-9017** (10 seeds). Seeds 9008-9011 have published baseline
random/adaptive arms from the section-5.1 proof run, but NO null or gated
outcome has ever been observed at any seed 9008-9017; the primary
comparison involves only fresh outcomes. Arms run: the null control
(once) and `gated` at theta in {`0.02`, `0.05`, `0.1`} - the family is
declared here because the theta values were chosen from the already-
observed 5.4 contrasts (posthoc family, pre-declared membership); the
family-wise correction buys back the posthoc selection. Power: n = 10 is
the smallest block whose sign-test bar (9/10, p `0.0098`) clears the
Bonferroni alpha/3 = `0.0167` bar - 8 seeds cannot pass at all (7/8 gives
p `0.0353`).

**Primary.** For each theta: one-tailed sign test on per-seed
gated-minus-null final-loss deltas (gated < null = accepted draws beat
doing nothing), each against alpha/3; the family verdict is "at least one
theta clears".

**Pre-registered interpretation.**
- Some theta clears alpha/3: the gate axis reopens - accepted draws carry
  value the null arm misses; pre-register a powered block (>= 32 seeds) on
  a fresh range around the winning theta.
- No theta clears: the probe-on-real-text program is concluded as
  falsified in ALL of its usable forms at this fidelity - ordering
  (5.3-5.4), absolute threshold at theta=0 (5.4), and intermediate
  thresholds (this study) - and the next direction moves off probe-derived
  selection entirely (routing-statistics heuristics, or no real-data
  topology mutation at this scale).

**Secondary (reported, not gating).** Accept rate and accepted-operation
balance per theta (mechanism: conservative theta should accept mostly
benign prunes; near-random theta approaches the random arm); the
same-seed comparison against the published random arms at 9008-9011
(secondary, 4 pairs only); per-arm search FLOPs (gated ~2x the baseline
adaptive booking, null exactly 0).

**Cost.** ~10 null sessions + 30 gated sessions at ~2.5 min each, ~1.7 h
on the local GPU; idempotent (skip-completed), resumable.

**Outcome (run 2026-09-14, revision `9bd9323`): NO theta clears - the
pre-registered falsification branch fires, and the probe-on-real-text
program is concluded as falsified in all of its usable forms.** Verdict
machine-checked (committed study summary
`artifacts/study-summaries/theta-sweep-9008-9017.json`, built by
`scripts/real_text_summaries.py` from the run payloads; the working
`artifacts/real-diag/theta-sweep/verdict.json` is written by
`scripts/theta_sweep.py`). All four sessions completed 10/10 seeds.
Per-theta primaries (sign test vs the null arm, bar 9/10, Bonferroni
alpha/3 = `0.0167`):

- theta `0.02` (accept 2/30, 6.7%): wins **0/10**, mean delta `+0.0027`,
  p `1.0000` - FAIL;
- theta `0.05` (accept 13/30, 43.3%): wins **0/10**, mean delta
  `+0.0236`, p `1.0000` - FAIL;
- theta `0.1` (accept 23/30, 76.7%): wins **0/10**, mean delta
  `+0.0713`, p `1.0000` - FAIL.

**Mechanism (verified from the published seed summaries).** Every seed
where a theta arm accepted nothing is BIT-IDENTICAL to the null arm
(8/10 seeds at theta `0.02`, 2/10 at `0.05`); where the accept/reject
sequences of two thetas coincide (9008, 9015, 9017), the two arms'
losses are bit-identical. The sweep therefore measures exactly what it
was designed to measure: the delta is the cost of accepted churn, and it
is monotonically harmful in acceptance rate - a clean dose-response of
churn harm (`+0.0027` at 6.7% accepted, `+0.0236` at 43%, `+0.0713` at
77%). The two probe evaluations per event never selected a draw whose
net effect on final loss was positive for the gate: probed-harmless
draws are harmful in reality (the section-5.4 finding, now confirmed
with accepts in the loop), and the threshold family cannot separate them.
Secondary notes: per-arm probe FLOPs ran as predicted (~2x baseline
adaptive booking); the same-seed comparison against the published random
arms at 9008-9011 is consistent, with one informative outlier - the
random arm at 9008 (loss `4.1279`, vs block-mates `3.2-3.5`) is itself a
churn catastrophe of exactly the kind section 5.8 established.

**Consequences, as pre-registered.** The gate axis is CLOSED on real
text at this fidelity: ordering (5.3-5.4), absolute threshold at
theta=0 (5.4), and intermediate thresholds (this study) all falsified.
The next real-data direction moves off probe-derived selection entirely
(routing-statistics heuristics, or no real-data topology mutation at
this scale). The honest real-text ladder is null < everything: at this
fidelity the only topology policy that helps is none.

### 5.12 Pre-registration: the depth x candidate-policy 2x2 null-control study (RAN - summaries `null-control-depth1-spread-384.json`, `null-control-depth4-prefix-384.json`)

**Motivation.** Sections 5.9 and 5.10 differ in TWO registered fields at
once: 5.9 is depth 1 + the prefix candidate policy, 5.10 is depth 4 +
the spread policy. The headline reading - "the richness axis separates
churn suppression (depth 1) from value creation (depth 4)" - rests on a
two-cell comparison that confounds model depth with the candidate-pool
policy. This study runs the missing two cells so the factor becomes a
proper 2x2: (depth 1 + spread) and (depth 4 + prefix), each with the
same three arms (random, adaptive, null) and the same primary contract
as sections 5.9/5.10.

**Design.** The study-3 registered toy config (hidden 24, cap 4 / 2
batches, 48 steps, interval 12, budget 3, `adaptive_policy=probe_loss`,
`adaptive_expert_metric=random`, `independent_arm_rng=True`), three arms
per experiment, 384 fresh seeds per cell, both cells novel
configurations on never-observed blocks: depth 1 + spread on block
**72000-72383**, depth 4 + prefix on block **73000-73383**. Sharded
20-way on CPU as in 5.9/5.10; verdict machine-checked by
`scripts/null_toy_study.py` (new profiles `depth1-spread` and
`depth4-prefix`), summaries committed to `artifacts/study-summaries/`
and pinned by test like the depth-1/depth-4 records.

**Primaries (per cell, identical contract to 5.9/5.10; the two cells
are one declared family and each cell's primaries are held to
Bonferroni alpha/2, exactly as in 5.9/5.10; no cross-cell claim is
gated).** (1) null-vs-random sign test - is churn harm present under
this policy? (2) adaptive-vs-null sign test - does the controller beat
doing nothing under this policy?

**Pre-registered interpretation (per cell).**
- (depth 4 + prefix) primary 2 PASSES: depth-4 value creation is robust
  to the candidate policy - the 5.9-vs-5.10 contrast is carried by
  depth, and the axis reading is revised to "depth separates
  suppression from creation".
- (depth 4 + prefix) primary 2 FAILS while 5.10 passed: the depth-4
  edge REQUIRES the spread policy - value creation narrows to the
  conjunction "depth 4 + spread", and Gate 3's configuration dependence
  must be quoted wherever its evidence is cited.
- (depth 1 + spread) primary 2 PASSES: spread alone creates value even
  at depth 1 - the candidate-policy axis, not depth, separates
  suppression from creation, and study 3's depth-1 prefix pass was
  suppression because of the conjunction, not the depth.
- (depth 1 + spread) primary 2 FAILS (expected): consistent with 5.9;
  the contrast with a passing depth-4 cell localizes the effect.
- Primary 1 fails in a cell (no churn harm there): that cell's primary
  2 remains interpretable but carries no churn-suppression component;
  quote that cell's ordering directly.

**Cost.** ~1.5-2 s/seed at depth 4 on CPU (measured section 9.3),
sharded 20-way ~1 min wall per cell.

**Outcome (run 2026-09-14/15, revision `142ad44`): DEPTH, not the
candidate policy, separates churn suppression from value creation.**
Built-in checks held in 384/384 seeds per cell; verdicts
machine-checked (`scripts/null_toy_study.py`), committed summaries
`artifacts/study-summaries/null-control-depth1-spread-384.json` and
`null-control-depth4-prefix-384.json`, pinned by test like the 5.9/5.10
records. The registered 2x2, with the two new cells completing it:

- depth 1 + prefix (5.9): primary 2 FAIL (117/384, mean `+0.0508`);
- depth 1 + spread (72000-72383): primary 1 PASS (null-vs-random
  289/384, mean `-0.0761`, p `3.5e-24`), primary 2 FAIL
  (adaptive-vs-null 131/384, mean `+0.0475`, p `1.0`) - ordering null
  `15.0884` < adaptive `15.1359` < random `15.1645`: branch 2,
  replicating 5.9 under the spread policy;
- depth 4 + prefix (73000-73383): primary 1 PASS (266/384, mean
  `-0.0701`, p `1.5e-14`), **primary 2 PASS (225/384, mean `-0.0371`,
  p `4.4e-04`)** - ordering adaptive `12.4413` < null `12.4785` <
  random `12.5485`: branch 1, replicating 5.10 under the prefix policy;
- depth 4 + spread (5.10): primary 2 PASS (218/384, mean `-0.0297`).

Reading, as pre-registered in the first interpretation branch: the
depth-4 edge does NOT require the spread policy (the prefix margin,
`-0.0371`, is if anything larger than 5.10's spread margin, `-0.0297`),
and the spread policy alone does not create value at depth 1 (`+0.0475`
versus 5.9's prefix `+0.0508` - the two depth-1 cells are
indistinguishable). Churn harm passes primary 1 in all four cells. The
section-5.10 reading ("the richness axis separates suppression from
creation") is REVISED: depth is the separating factor, and the
candidate policy is exchangeable within the registered cap-4 / 2-batch
budget. Gate 3's value-creation claim should be quoted as depth-driven
at this scale; the depth-1 rows (all four arms' comparisons at depth 1)
remain churn-suppression phenomena.

### 5.13 Pre-registration: the routing-statistics heuristic against the null arm (RAN - summary `heuristic-9008-9017.json`)

**Motivation.** Section 5.11's falsification branch named the next
real-data direction: off probe-derived selection entirely, starting
with routing-statistics heuristics. The minimal such policy is also the
one the training loop already computes for free: grow one expert at the
argmax-routing-load layer (the same per-layer load key the random arm's
fallback reads), every event, deterministically - no chooser draw, no
probes, zero search FLOPs beyond the post-grow rebuild. After
probe-ORDERED churn (5.3) and probe-GATED churn (5.4/5.11) both lost to
doing nothing, this asks the last open real-text question: can DIRECTED
churn beat doing nothing?

**Design.** Single-arm sessions on the V0 diagnostic path (steps 600,
interval 200, budget 3, cap 4 / 1 probe batch) on the fresh block
**9008-9017** (10 seeds) - the theta sweep's block, whose published null
arms are the control. The heuristic arm shares the random arm's init
identity and training stream, consumes no chooser RNG, and grows at the
measured argmax-load layer each event (operator cap: layers grow to 16
experts, so the policy is structurally alive at expert_count=8).
Primary: one-tailed sign test on per-seed heuristic-minus-null final-loss
deltas (heuristic < null = directed churn beats doing nothing) against
Bonferroni alpha/2 = `0.025` (bar 9/10; n = 10 as in 5.11).

**Built-in checks.** Zero probe FLOPs; every mutation is a grow at the
layer that was argmax-load at the event (recorded per event); mutation
count 3/3 seeds (the budget guard counts mutations, which happen). The
null values on this block are already published, so the comparison
involves no new control run.

**Pre-registered interpretation.**
- Heuristic clears the bar: directed churn carries real signal on real
  text - the non-probe selection program OPENS; pre-register a powered
  block (>= 32 seeds) plus the load-threshold family (grow only when the
  argmax load exceeds a margin) as the follow-up.
- Heuristic loses or ties: the real-text ladder closes at the top too -
  no topology mutation policy beats doing nothing at this fidelity - and
  the next direction is NO real-data topology mutation at this scale;
  resources move to Gate 4's compute-efficiency protocol, the one
  remaining open gate.

**Secondary (reported, not gating).** The grow-layer histogram (targeting
footprint); per-seed deltas against the published random arms at
9008-9011 (4 pairs, descriptive only); cost (predicted 0 probe ratio;
rebuild bookings only).

**Cost.** ~10 single-arm sessions at ~2.5 min each, ~25 min on the local
GPU; idempotent (skip-completed), resumable.

**Outcome (run 2026-09-15, revision `0ef29de`): interpretation branch
(b) - the heuristic loses to doing nothing on EVERY seed, and the
real-text ladder closes at the top.** Wins 0/10 vs the published null
arms, mean delta `+0.1218` (per-seed `+0.049` to `+0.410`, all
positive), p `1.0000` against the 9/10 Bonferroni bar. Committed study
summary `artifacts/study-summaries/heuristic-9008-9017.json` (built by
`scripts/real_text_summaries.py` from the run payloads; per-seed losses,
targeted layers, and the grow-layer histogram embedded, statistics
recomputed by pinning tests). Built-in checks
held: 30/30 mutations all grows at the measured argmax-load layer
(histogram L5 x10, L7 x6, L4 x5, L0 x4, L6 x3, L3 x2 - L1/L2 never
targeted), and search FLOPs carry rebuild bookings only (a single probe
evaluation would exceed the arm's whole search-FLOPs total). Reading:
growing an untrained expert exactly where routing mass concentrates is
the maximally disruptive single mutation - directed churn is STILL
harmful, and the direction did not save it. Secondary: against the
published random arms at 9008-9011 the heuristic averages `-0.109`
(1/4 wins; the mean is dominated by the 9008 random churn catastrophe,
`4.128`). The complete real-text picture at this fidelity:

- probe-ORDERED churn loses to doing nothing (5.3, all five variants);
- probe-GATED churn loses to doing nothing at every threshold (5.4,
  5.11);
- load-DIRECTED churn loses to doing nothing on every seed (this
  section);
- the null arm is the only policy that wins, and it wins against
  everything.

**Consequence, as pre-registered.** No topology mutation policy beats
doing nothing on real text at this scale; the real-data topology
program is CLOSED, and the project's one remaining open gate is Gate 4
(the compute-efficiency protocol of section 5.5, block 15000-15011,
unspent). The toy factor map (5.9-5.12) stands on its own: depth-driven
value creation exists at depth 4 there, and closing the real-text leg
removes the transfer assumption, not the toy result.

### 5.14 Pre-registration: the do-nothing clause for Gate 4 (designed, not yet run)

**Motivation.** The section 5.1/5.5 verdict structure predates the null
arm. Its harm guard (adaptive worse in >= 10 of 12 pairs vs RANDOM) and
its pool-matched clause both measure against churn arms; sections 5.8,
5.11 and 5.13 then established that on real text at diagnostic fidelity
churn is itself the harm (null < random, < gated at every threshold,
< heuristic, all at p < 0.001). If that ordering holds at declared
fidelity, the registered gate could pass while adaptive still loses to
doing nothing - authorizing Gate 4 on a comparison that no longer
carries the selection claim. The 2x2 (5.12) shows the depth factor cuts
both ways: the declared config is depth 12, deeper than any toy cell
where creation won, so no toy result predicts the declared-fidelity
outcome. Only the declared-scale measurement can.

**Design.** A **null-session queue** at DECLARED fidelity over the SAME
block 15000-15011, run as its own single-arm sessions (`arms=("null",)`)
into its own dataset/session roots (as the 5.5 pool-matched control uses
its own roots), publishing per-seed `null_final_loss` under the existing
runner contract (5.8). Expected cost ~90 s/seed of GPU (~18 min over 12
sessions) plus restore/publish overhead; no probes, no search FLOPs,
mutations always zero. A null session trains the SAME init and data
stream as the pair sessions, so its losses are paired with the primary
run's adaptive/random arms per seed. No probe-budget flag is needed; the
null arm is policy-free by construction.

**Adoption rule.** The T4 runbook (T4_RUNBOOK.md) gains a step 2c: run
the null queue with `--session-arms null` and separate roots
(`artifacts/datasets-null`, `artifacts/v0-sessions-null`, own Kaggle
dataset slug). The verdict (scripts/v0_verdict.py) gains a
`--null-root` argument; with it, the per-seed null losses are loaded and
two amendments apply:

1. **Do-nothing clause (the 5.14 primary).** Gate 4 is additionally
   withheld if the adaptive arm fails to beat the null arm on the
   declared queue: adaptive-minus-null losses are counted and Gate 4 is
   withheld when adaptive loses-or-ties >= `NULL_WIN_LIMIT = 10` of 12
   seeds (the mirror of the registered harm-guard limit). The verdict
   label is `PASS_COST_ONLY_NULL_REJECTS` (exit 2): the cost PASS and
   the registered rules all stand; the adaptive edge does not
   demonstrate value over doing nothing.
2. **Mandatory-execution clause.** A missing or incomplete null queue
   withholds Gate 4 (`PASS_COST_ONLY_NULL_MISSING`, exit 2) - running
   the do-nothing control is part of the protocol; its absence must
   never read as "no null effect found".

Clause precedence: cost FAIL (exit 1) > harm guard > pool-matched
control > **null clause (5.14)** > control missing > null missing >
band miss. All withholding clauses exit 2 and leave the cost PASS
standing; the label names which fired.

**Pre-registered interpretation (fixed before running).**
- (a) Adaptive clears the do-nothing bar at declared fidelity (null
  rejects clause does not fire): the selection claim survives the
  strictest control - the gate then stands or falls on the registered
  cost/band/control rules exactly as registered.
- (b) The null clause fires (adaptive loses-or-ties >= 10/12 vs null):
  the real-text null ordering REPLICATES at declared fidelity, Gate 4 is
  withheld, and the 5.1 program's selection premise is falsified at its
  own measurement scale. The cost/band/control machinery stands as a
  protocol demonstration; no re-tuning, no new block, no knob changes
  (the 5.5 discipline). The project's open frontier is then the toy
  factor map (5.12: depth-driven value creation) and Gate 4 is closed
  with the real-text program.
- (c) Mixed (5-9/12 adaptive losses vs null): treat exactly like the
  registered harm-guard gray zone - cost PASS stands, Gate 4 is
  authorized only if the clause does NOT fire (losses < 10); the gray
  zone itself is recorded but authorizes nothing.

**Built-in checks (machine-checked at verdict time).** Null arms must
book zero search FLOPs and zero mutations at declared fidelity too
(rebuild bookings still land in search FLOPs for mutating arms only;
the null arm has none); null queue completeness on the same block;
per-seed pairing (each null loss must come from the same seed as the
pair it is compared against).

**Cost.** ~90 s/seed on the T4 (24.576M tokens at the measured rate,
no probe/event overhead beyond zero mutations) - ~7 min of GPU across
the queue, plus per-session restore/publish overhead. It runs as
regular sessions in the same 12-hour Kaggle windows, after the primary
queue completes (the runbook order).

### 5.15 Pre-registration: the depth dose-response of value creation (RAN - summaries `null-control-depth2-384.json`, `null-control-depth3-384.json`; `IMPLEMENTATION_STATUS.md`)

**Motivation.** The 2x2 (5.12) established that DEPTH - not the candidate
policy - separates churn suppression (depth 1: adaptive loses to the
null arm at 131/384 and 117/384) from value creation (depth 4: adaptive
beats the null arm at 225/384 and 218/384). But two points do not locate
an ONSET: the transition could sit anywhere in (1, 4]. This study places
depths 2 and 3 on the same axis, with the prefix policy (the registered
default) held fixed, completing a depth-1/2/3/4 series at otherwise
identical registered configs.

**Design.** Two new profiles of the 5.9-5.12 machinery (three arms
`random,adaptive,null` per experiment, `independent_arm_rng`, cap-4/
2-batch probes, 48 steps, interval 12, budget 3): depth 2 on fresh block
**74000-74383**, depth 3 on fresh block **75000-75383** - both disjoint
from every spent block (70000-70383, 71000-71383, 72000-72384,
73000-73384) and from all real-text blocks. 384 experiments per cell,
20-way CPU sharding as in 5.9-5.12.

**Primaries (per cell, Bonferroni alpha/2 = `0.025`, n = 384, bar
214/384):** (1) null < random (churn harm present at this depth);
(2) adaptive < null (value creation beyond doing nothing at this depth).

**Pre-registered interpretation (fixed before running).** Let w(d) be
primary-2 wins at depth d (w(1) = 131, w(4) = 225 from 5.12):
- (a) w(2) and w(3) both clear: value creation onset begins at or below
  depth 2, and the 5.9 depth-1 failure is a narrowly shallow-substrate
  effect - the depth axis sharpens to a threshold between 1 and 2.
- (b) w(2) and w(3) both fail: onset sits between depth 3 and 4 - a
  sharp late threshold consistent with depth-4-specific expressiveness.
- (c) w(2) fails, w(3) clears: onset is between 2 and 3; the monotone
  reading of the depth axis is confirmed with the threshold located.
- (d) w(2) clears, w(3) fails: a NON-monotone response - value creation
  is not a pure function of depth; the 5.12 attribution would need a
  third factor, and the depth axis reading is suspended until a
  follow-up isolates it (no re-tuning of existing cells).
Primary 1 is expected to clear at every depth (churn harm is
substrate-general per 5.8-5.13); a primary-1 failure at any depth is
recorded as a revision of the churn-harm generality claim, not a
protocol deviation.

**Built-in checks.** Identical to 5.9-5.12 (zero null mutations and
search FLOPs; matched budgets; arm-RNG isolation via the 2-arm ==
3-arm invariant).

**Cost.** ~2 s/seed at depth 1 scaled by depth (deeper models are
cheaper per seed than the depth-4 cell's ~2.4 s), so roughly 13-16 min
per cell across 20 shards; ~30 min total.

**Outcome (run 2026-09-15, revision `bbad439`): interpretation branch
(b) - the value-creation onset sits between depth 3 and depth 4.**
Built-in checks held in 384/384 seeds per cell; committed study summaries
`artifacts/study-summaries/null-control-depth2-384.json` and
`null-control-depth3-384.json`. Per-depth primaries (n = 384, bar 212/384,
Bonferroni alpha/2 = `0.025`):

| depth | P1 null < random | P2 adaptive < null | mean adaptive-null | ordering |
|---|---|---|---|---|
| 1 (5.9) | 280/384, p `4.7e-20` | 117/384 | `+0.0508` | null < adaptive < random |
| 2 (5.15) | 290/384, p `1.2e-24` | 161/384 | `+0.0208` | null < adaptive < random |
| 3 (5.15) | 286/384, p `9.5e-23` | 188/384 | `+0.0046` | null < adaptive < random |
| 4 (5.12) | 266/384, p `1.5e-14` | **225/384**, p `4.4e-04` | `-0.0371` | adaptive < null < random |

The dose-response is monotone in wins (117 < 161 < 188 < 225) and in mean
harm (`+0.0508` > `+0.0208` > `+0.0046` > `-0.0371`), with the sign flip
just above depth 3. Branch (d) non-monotonicity is excluded. Churn harm
(primary 1) clears at every depth, extending its generality across the
whole measured substrate range. The depth-4 prefix margin
(`-0.0371`) slightly exceeds the depth-4-spread one (`-0.0297`), so the
series brackets the onset to (3, 4]: the sharp question for any future
study is depth 3 vs 4 with a finer covariate, not a re-tune of the
existing cells. Consolidated depth-axis reading: **value creation by
probe-ordered topology mutation requires substrate depth sufficient to
support it (onset between depth 3 and 4); below that, every arm that
mutates loses to doing nothing, and only the richness of choice masks
it against random churn.**

### 5.16 Pre-registration: is the discovered topology a reusable artifact? (plan section 52 / section 55, toy fidelity)

**Motivation.** Every Gate 3 comparison so far scores a topology *while it is
trained*: an arm's final loss contains whatever its own weights happened to
learn, so no existing result separates the structure from the run that produced
it. The plan's stronger claim (section 102) is that the topology is the
extractable artifact - *"can the discovered topology be extracted as a reusable
computational genome that remains advantageous when freshly initialized and
trained independently? That is the result worth pursuing."* - and plan
sections 52/55 register the experiments that test it. The initialization
policies exist (`act.training.retraining`, copied/skipped counts reported) but
no study has used them: report deliverable 10 ("Fresh-initialization results")
is TODO, and `IMPLEMENTATION_STATUS` records it as "architecture-equivalence
studies remain". This is the cheapest decisive version of that experiment, and
it is the first one that can falsify the project's core framing rather than add
evidence to the while-training comparisons.

**Hypothesis.** The depth-4 adaptive arm's advantage is carried *structurally*:
its discovered topology, rebuilt with fresh random weights and retrained with
the topology frozen, still beats the random-churn arm's discovered topology
treated identically.

**Prediction (declared before measurement).** Section 5.10 decomposed the raw
depth-4 edge: of `0.1130` adaptive-minus-random, only `0.0297` (~26%) survived
the do-nothing bar *with weights included*. Removing the weights can only
shrink the structural residual, so: **primary 1 (structure beats random churn's
structure) is predicted to pass weakly or fail; primary 2 (structure beats the
no-mutation topology) is predicted to fail.** A pass on primary 2 would be the
project's first structural-value evidence and would materially strengthen the
computational-genome claim; the prediction is recorded so that a weak pass is
not later read as a strong one.

**Design (one block per seed, the section-5.6/5.10 registered config: depth 4,
`probe_candidate_policy="spread"`, cap-4/2-batch probes, 48 steps, interval 12,
budget 3, `independent_arm_rng=True`, cpu), fresh block 81000-81383 (384
seeds), 20-way sharding as in 5.9-5.15.** Per seed:

1. **Discovery.** One three-arm block (`arms="fixed,random,adaptive"`). Each
   arm's final genome is recorded - the toy `ArmResult` now carries the
   serialized genome, which is plan section 11's weight/topology separation made
   usable rather than merely present. The `fixed` arm never mutates, so its
   genome *is* the initial genome; a test pins that the `fixed` genome equals
   the discovery-block initial genome, so "the no-mutation topology" needs no
   separate null arm.
2. **Transport.** For each of the three final genomes: build a fresh model from
   the genome (`model_from_genome`), train it for the same 48 steps on the same
   per-seed data schedule with **mutation disabled** (the topology is frozen -
   the genome, not the controller, is under test), and score the same four
   deterministic held-out validation batches. A test pins that a transported
   *fixed* genome reproduces the fixed arm's own training and validation
   batches, so the comparison is paired within seed.

**Primaries** (exact one-tailed sign test, Bonferroni alpha/2 = `0.0025`, n =
384, bar 214/384):
1. **Reusability.** transported adaptive topology beats the transported random
   topology on final held-out loss, in >= bar seeds.
2. **Over doing nothing.** transported adaptive topology beats the transported
   fixed (initial, never-mutated) topology, in >= bar seeds.

**Secondaries (reported, not verdict-bearing).**
- **Transport validity / noise floor.** the transported fixed topology's loss
  against the fixed arm's own discovery loss, per seed: the transport procedure
  must reproduce a known arm before its deltas can be read as effects. The
  absolute spread of this difference is recorded as the noise floor, and any
  primary margin inside it is recorded as inconclusive rather than a pass.
- **Capacity.** transported `parameter_count` and `active_parameter_count` per
  arm. Section 0's hypothesis is at *equivalent capacity* and section 21 caps
  the budget, so a structural win at higher parameter count is recorded as
  capacity-confounded, not as topology value.
- **Discovery-time arm losses** for context, expected to reproduce section
  5.10's ordering (adaptive < fixed ~ null < random).

**Pre-registered interpretation (fixed before running).**
- (a) Both primaries pass: the discovered topology is a reusable artifact at
  toy fidelity - the first *structural* evidence for the plan's headline claim,
  and the trigger for the section-52 scaling version (300M -> 1B) as the next
  rung rather than more while-training studies.
- (b) Primary 1 passes, primary 2 fails: the adaptive topology beats random
  churn's *structure* but not the never-mutated one - it preserves rather than
  creates, consistent with section 5.10's decomposition and with the depth-1
  reading; the genome is a stable object but not yet a superior one.
- (c) Primary 1 fails: the adaptive advantage lives in the weights and/or
  initialization rather than in the structure it chose. The "reusable
  computational genome" framing is not supported at toy fidelity, and the next
  registered step is section 55's initialization-mode comparison
  (evolved/rewound/fresh/teacher on one genome), not scaling.
- (d) Transport validity fails: the procedure itself is not reproducible, the
  study is recorded as inconclusive, and no other reading is taken from it.

**Scope limits (pre-declared).** This is toy fidelity: it tests the reusability
*mechanism*, not section 52's scaling claim (a genome found at 300M transferring
to 600M/1B), which needs the declared path and is not attempted here. It also
does not test whether a *fresh controller* re-discovers the same topology -
the topology is frozen at transport precisely so that only the genome is under
test. Machinery smoke-tests run on already-spent blocks only (73000-73383);
the verdict uses 81000-81383 and nothing else.

**Cost.** 3 discovery runs + 3 transports per seed, the discovery runs the same
size as section 5.10's (measured ~1.5-2 s/seed/cell at depth 4 on CPU);
sharded 20-way, minutes of wall clock.

### 5.16.1 Amendment: the declared threshold was internally inconsistent

Recorded after the run, before any primary was read, because the declaration
above cannot be applied as written. It states the per-primary threshold two
ways that disagree: "Bonferroni alpha/2 = `0.0025`" and "bar 214/384" at
n = 384. Neither matches the other, and neither matches what every sibling
study in this report applied. The exact one-tailed binomial bars at n = 384
are **220 at 0.0025** and **212 at 0.025**; 214 corresponds to no threshold at
all (it sits at alpha `0.0140`). Section 5.15's declaration carries the same
"bar 214/384" wording - so "214" is a slip inherited from that section, whose
own outcome paragraph records the applied bar as 212 at alpha/2 = `0.025`.

The `0.0025` is unique to section 5.16 and is a factor-10 slip for `0.025`.
Both slips happen to point the STRICT way, which makes the resolution
load-bearing: the measured primary-1 win count is **212**, which is exactly
the sibling bar and far below the declared one. Making this choice after
seeing 212 would be precisely the post-hoc flexibility pre-registration
exists to prevent, so the rule applied is: **a self-inconsistent declaration
is read as declared, i.e. at the stricter `0.0025`/bar 220, and the looser
reading is reported beside it rather than substituted for it.** Section 5.16's
registered verdict is therefore scored at bar 220; the sibling-convention
verdict is also reported, and section 5.16.2 labels both.

**Outcome (run 2026-09-15, block 81000-81383, shards from revision `e62a732`;
aggregation amended per 5.16.1): registered branch (c) - both primaries fail.**

Built-in checks held in 384/384 seeds: the transported unmutated genome
reproduces its own discovery run **exactly** (max |delta| `0.0`; 384/384 exact
matches), every transport used `initialize_retraining(mode="fresh")` with
zero copied parameters, and zero mutations occurred during transport. So the
procedure is faithful and the deltas below are attributable to the topology.

| comparison (n = 384) | wins | mean delta | p | bar 220 (registered) | bar 212 (sibling) |
|---|---|---|---|---|---|
| P1 transport adaptive < transport random | 212/384 | `-0.1016` | `2.32e-02` | **FAIL** | pass (exactly at bar) |
| P2 transport adaptive < transport fixed | 211/384 | `-0.0866` | `2.94e-02` | **FAIL** | fail |
| (discovery) adaptive < random | 259/384 | `-0.1036` | `3.53e-12` | pass | pass |
| (discovery) adaptive < fixed | 216/384 | `-0.0174` | `8.18e-03` | pass | pass |

Arm means: discovery `adaptive 12.3580 < fixed 12.3754 < random 12.4617`;
transport `adaptive 12.2889 < fixed 12.3754 < random 12.3905`.

Three things this licenses beyond the branch label:

1. **The registered depth-4 ordering replicates out of sample.** The
discovery half of this block is the section 5.10 config (the arm order keeps
`random`/`adaptive` at indices 0/1), run on a fresh 384-seed block: adaptive
beats random `259/384` (5.10: `263/384`) and fixed `216/384`. The while-training
result is not a property of 5.10's seed block.
2. **Discarding the weights costs the edge its consistency, not its mean.**
adaptive-vs-random mean delta barely moves (`-0.1036` -> `-0.1016`) while the
win rate collapses `259/384 -> 212/384`, and a seed's discovery ranking
predicts its transport ranking only `201/384` times (52%, chance). The
topology carries the *average* effect and none of the reliability.
3. **The transport is not capacity-confounded in the flattering direction.**
The adaptive topologies are usually *smaller* (lower active parameter count in
264/384 seeds, higher in 30). On the 90 seeds where the two are
capacity-matched, adaptive beats random `40/90` (bar 59) - i.e. not even
at the loose bar. A structural win here would not have been bought with
parameters, and there is no win to explain.

> **Superseded in part - read section 5.18.2 before quoting this.** The
> comparison this section fails is *initialization-mismatched*: the adaptive
> side carries a redrawn initialization while the control carries its original
> one. Section 5.18 re-ran it with initialization matched on both sides (donor
> on both: `234/384`; own on both: `293/384`) and the topology wins both times.
> The registered verdict below stands as what this section's design could
> measure, not as the project's finding on whether the topology is a standalone
> artifact.

**Reading.** Under either bar, primary 2 fails: a topology discovered by the
probing arm is *not* a better standalone architecture than the initial,
never-mutated one. Primary 1 is the knife edge - directionally favourable
(mean `-0.10`) and exactly at the sibling bar, but failing the bar this
section declared. The substantive finding is branch (c): **at toy fidelity the
adaptive advantage lives in the trained weights and/or initialization rather
than in the structure the controller chose.** That is the pre-registered
prediction written in 5.16 ("primary 1 ... predicted to pass weakly or fail;
primary 2 ... predicted to fail"), so no prediction is being repaired here -
or is a pass on primary 2, which 5.16 declared would be the project's first
*structural* evidence for the computational-genome claim. There is none.

**Consequence for the plan.** Branch (c)'s registered next step is plan
section 55's initialization-mode comparison on a single genome
(evolved/rewound/fresh/teacher), not the section-52 scale-up: the scale-up
asks whether a genome found at 300M transfers to 600M/1B, and this study's
result gives no reason to believe the genome is doing work that would survive
the transfer. The extraction machinery (genome in the record, fresh-init
transport, transport validity) is now measured and reusable for that study.
Committed summary: `artifacts/study-summaries/reusability-depth4-384.json`
(`study_f0334893263366c2`).

### 5.17 Pre-registration: is initialization or topology the important artifact? (plan RQ6 / section 55, toy fidelity)

**Motivation.** Section 5.16 discarded the trained weights and the adaptive
arm lost its edge, which is exactly the question plan RQ6 asks in the other
direction: given a discovered topology, does the *initialization* matter? Plan
section 55 ("one of the project's headline experiments") registers four
treatments on one genome - evolved weights, rewound initial weights, fresh
random initialization, teacher initialization - and plan section 11 requires
the machinery for it to be unambiguous ("we need to be able to do *final
topology + original initialization*" so the Lottery Ticket experiments are
interpretable). Section 5.16 built the fresh half; this study builds the
rewound half and compares them on the same topology.

**What "rewound" means here, declared precisely.** Build the final genome with
the block's fresh draw, then restore every component that *existed in the
initial genome* to its initial weights, identified by component identity (MoE
expert id), not by tensor position. Components that did not exist at
initialization - grown experts - keep the fresh draw, because they have no
original initialization to be rewound to. On an unmutated genome this makes
rewound and fresh **identical by construction**, which is the study's first
built-in check; on a pruned genome the two differ only in whose random weights
the survivors hold.

**Design.** One discovery block per seed on the section-5.16/5.10 registered
config (`arms="random,adaptive,fixed"`, depth 4, `probe_candidate_policy="spread"`,
cap-4/2-batch probes, 48 steps, interval 12, budget 3, `independent_arm_rng=True`,
cpu), fresh block **82000-82383** (384 seeds), 20-way sharding as in 5.9-5.16.
Per seed, five transports of three discovered genomes:

| label | genome | initialization |
|---|---|---|
| `B_adaptive` | adaptive final | rewound |
| `C_adaptive` | adaptive final | fresh |
| `B_random` | random final | rewound |
| `C_random` | random final | fresh |
| `B_fixed` | fixed (initial) | rewound = fresh = the block's initial model |

All transports train the same 48 steps on the same per-seed data schedule with
mutation disabled, and are scored on the same four held-out batches. `A` (plan
section 55's evolved-weights treatment) is the discovery arm's own run, already
measured by the block; `D` (teacher) is **out of scope at toy fidelity** and is
not faked - the plan's teacher is Qwen-derived (section 51) and
`IMPLEMENTATION_STATUS` records that no external Qwen artifacts are bundled or
claimed.

**Primaries** (exact one-tailed sign test, n = 384, per-primary threshold
`alpha/2 = 0.025` - the section 5.9-5.15 convention - which at n = 384 is the
exact bar **212 wins**; the script asserts the declared bar equals
`_gate_min_wins(n, alpha/2)` at run time so the section-5.16 inconsistency
cannot recur silently):
1. **Initialization carries value.** `B_adaptive` beats `C_adaptive` on final
   held-out loss, in >= 212 seeds: with the topology held fixed, the original
   initialization of its surviving components is worth something.
2. **The topology alone, with initialization held level.** `B_adaptive` beats
   `B_fixed`, in >= 212 seeds. Both arms carry their original initialization
   (`B_fixed` *is* the initial model), so this is the cleanest available test
   of whether the discovered structure has standalone value: it removes the
   section-5.16 confound where the adaptive side got wholly new weights while
   the fixed side kept its own.

**Predictions (declared before measurement).** Section 5.16 found the
while-training edge survives fresh initialization in the mean (`-0.1016`) but
not in per-seed consistency (`212/384`, and a seed's discovery ranking predicts
its transported ranking only `201/384` times). Rewinding cannot restore a
*grown* expert's lucky draw, and the toy's prune decisions are made on probe
loss after the prune rather than on survivor luck, so: **primary 1 is predicted
to fail** (a pass would mean surviving components' particular random draws are
load-bearing, i.e. a lottery-ticket-like effect at this scale); **primary 2 is
predicted to fail on wins while staying directionally favourable** (mean < 0),
matching section 5.16's P2 shape (`211/384`, mean `-0.0866`).

**Secondaries (reported, not verdict-bearing).**
- **Transport validity.** `B_fixed` must reproduce the fixed arm's own
  discovery run *exactly*, and `B_adaptive`/`C_adaptive` must be identical to
  each other whenever the adaptive topology has no grown experts (pure-prune
  genomes). Both are checked per seed, not assumed.
- **How much the trained weights are worth.** `A_adaptive - B_adaptive` per
  seed: the part of the arm's advantage that only exists in the weights it
  trained (section 5.16's branch-(c) claim, quantified).
- **Procedure control.** `B_random` vs `C_random` - does rewinding help random
  churn's topology too? A pass there means rewinding is a generic procedure
effect rather than something specific to the adaptive arm's survivors.
- **Capacity.** transported parameter/active-parameter counts, and primary 2
  re-run on the subset where the adaptive and initial topologies are
  capacity-matched (section 0's equivalent-capacity requirement).

**Pre-registered interpretation (fixed before running).**
- (a) Both primaries fail: neither the topology nor its original initialization
  is a standalone artifact at toy fidelity - the adaptive arm's advantage is
  the trained weights, full stop, and the plan's "computational genome" framing
  has no support at this fidelity. Next registered step becomes measuring how
  far that weight-borne advantage survives *within* a run (the plan's
  compute-efficiency protocol, Gate 4), not further extraction studies.
- (b) Primary 1 passes, primary 2 fails: reusing the original initialization is
  what preserves the advantage while the structure alone does not - a
  lottery-ticket-like reading, and the trigger for a reward-signal study (does
  the probe select components that were lucky at init?).
- (c) Primary 1 fails, primary 2 passes: the discovered topology is a real
  standalone artifact once initialization is held level. This would REVISE
  section 5.16's branch-(c) verdict - its P2 failure would be attributable to
  the fresh draw rather than to the structure - and it triggers the section-52
  scaling version.
- (d) Both primaries pass: both the structure and its initialization carry
  value; the strongest reading, and the section-52 scale-up is triggered with
  the initialization policy fixed as part of the artifact.

**Scope limits (pre-declared).** Toy fidelity, one seed block, one substrate.
The teacher treatment (D) is untested, not tested-and-failed. This study does
not test whether a fresh controller re-discovers the topology. Machinery
smoke-tests run on already-spent blocks (73000-73383) only; the verdict uses
82000-82383 and nothing else.

**Cost.** 1 discovery block (3 arms) + 5 transports per seed, ~6 arm-runs of
the section-5.16 size; measured ~5 s/seed single-process at depth 4, so
~30 min of CPU across 20 shards.

### 5.17.1 Pre-run correction: the rewind identity is not what 5.17 declares

Caught by the machinery smoke test on spent block 73000-73003, **before the
registered block was run and before any primary was read**. Section 5.17's
transport-validity clause claims that `B_adaptive` and `C_adaptive` "must be
identical to each other whenever the adaptive topology has no grown experts
(pure-prune genomes)". That is false, and the smoke test failed on it, which is
how it was found.

The fresh draw depends on the *shapes being drawn*: a genome that only pruned
experts is still a different architecture, so rebuilding it consumes the RNG
differently and no surviving component gets the value the initial model's draw
gave it. ``B == C`` therefore holds exactly when the **architecture is
unchanged** - which only the unmutated `fixed` genome satisfies - and not when
"nothing grew". Measured on the smoke block: 3/4 seeds had no grown adaptive
experts, yet `B != C` on all of them.

The correction runs *against* the study's interest in the sense that it removes
a free identity, and *for* it in the sense that it is what makes primary 1 a
real comparison: rewinding changes the adaptive topology's starting point even
when no expert was ever grown, because the survivors keep their own original
initialization instead of a redrawn one. The primaries, the threshold
(`alpha/2 = 0.025`, bar 212 at n = 384) and the four interpretation branches are
**unchanged**; only the built-in pre-conditions are restated:

1. the declared bar equals `_gate_min_wins(384, 0.025)` - asserted at runtime,
   so section 5.16's inconsistency cannot recur silently;
2. the registered block's seed count equals the declared n = 384;
3. `B_fixed` reproduces the `fixed` arm's own discovery run **exactly** (loss and
   training FLOPs) on every seed - the load-bearing validity check;
4. `B == C` exactly where the topology is unchanged, and `B != C` everywhere it
   is not;
5. every rewound transport copied at least one parameter, and every transport's
   recorded initialization mode is the one it was asked for.

### 5.17.2 Outcome: registered branch (d) - both the structure and its initialization carry value

Run 2026-09-15, block 82000-82383 (fresh), 20 shards, revision `e62a732`.
Built-in checks held on every seed: the declared bar equals the exact binomial
bar, `B_fixed` reproduces the `fixed` arm's own discovery run **exactly** (loss
and training FLOPs, 384/384), `B == C` exactly where the topology is unchanged
and never where it is not, and every rewound transport transferred parameters.
Substrate inventory: 213/384 seeds had no grown experts (pure-prune
topologies), 171/384 grew at least one (201 grown experts total), and no
adaptive topology was unchanged. **Transport validity is exact, which is what
makes the deltas below attributable to the initializations rather than to the
procedure - and this time the arm that must be reproduced is the one whose
architecture is unchanged, which the pre-run correction (5.17.1) fixed before
the block was run.**

| comparison (n = 384, bar 212) | wins | mean delta | p | verdict |
|---|---|---|---|---|
| **P1** `B_adaptive` < `C_adaptive` (rewound < fresh) | 215/384 | `-0.0806` | `1.08e-02` | **PASS** |
| **P2** `B_adaptive` < `B_fixed` (topology, init level) | 274/384 | `-0.1123` | `1.46e-17` | **PASS** |
| control `B_random` < `C_random` (does rewinding help any topology?) | 189/384 | `+0.0424` | `6.40e-01` | fail |
| replication `C_adaptive` < `B_fixed` (section 5.16's P2, fresh init) | 198/384 | `-0.0317` | `2.87e-01` | fail |
| (discovery) adaptive < random | 255/384 | `-0.0941` | `6.13e-11` | pass |
| (discovery) adaptive < fixed | 205/384 | `-0.0162` | `1.01e-01` | fail |

Transport arm means: `B_adaptive 12.3101 < C_random 12.3776 < C_adaptive 12.3907
< B_random 12.4200 < B_fixed 12.4224`. Discovery means: `adaptive 12.4062 <
fixed 12.4224 < random 12.5003`.

> **Extended by section 5.18.** This section's own-initialization rows are the
> *matched* half of the picture and are confirmed there (`O` vs `O_fixed`,
> `293/384`); its fresh-init rows (`198/384`) are the *mismatched* half and are
> explained there as an unmatched control rather than as a weak topology. Read
> 5.18.2 for the synthesis.

**What this says.** With the initialization held as level as the topology
allows - both sides carry their original initialization, one as the unmutated
initial model and the other as its survivors rewound to their initial weights -
the discovered topology beats the never-mutated one `274/384` with mean
`-0.1123` (P2). The same comparison with a *fresh* draw on the adaptive side
fails `198/384` (mean `-0.0317`), reproducing section 5.16's P2 on an
independent block (`211/384`, mean `-0.0866`). The difference between those two
rows is one variable: whose random weights the surviving experts hold. So
section 5.16's branch (c) is **revised, and narrowed**: the topology is a
standalone artifact, but only as the *pair* (discovered structure + the
initialization its components actually had). Plan section 11's insistence on
`final topology + original initialization` "without ambiguity" is empirically
load-bearing rather than a bookkeeping preference, and plan section 102's
"computational genome" is supported only in that paired form. That is this
run's registered branch (d), and it also carries branch (c)'s consequence: the
section-52 scale-up is triggered, with the initialization policy declared as
part of the artifact.

Three checks keep the reading honest, and each was pre-registered:

1. **The control rejects the generic explanation.** Rewinding the *random*
churn arm's topology does nothing (`189/384`, mean `+0.0424`; on the pure-prune
subset `109/213`). If rewinding were a general "restore good weights"
procedure, this is where it would show; it does not. The effect is specific to
the topology the adaptive controller chose.
2. **P1 is concentrated exactly where rewinding is total.** Splitting by growth:
pure-prune seeds `123/213` (p `1.4e-02`, bar 122) versus seeds with grown
experts `92/171` (p `1.8e-01`). A grown expert has no original initialization to
be rewound to, so the treatment is partial there - the effect is strongest
where the mechanism can actually act, which is the shape a real mechanism has
and not the shape a confound has. P2, by contrast, holds in both subgroups
(`163/213` and `111/171`, p `1.8e-15` and `5.9e-05`): the topology's value does
not depend on how it was reached.
3. **Not a capacity effect.** The adaptive topology is *smaller* than the initial
one in `352/384` seeds (higher in 32), so P2 is a win with fewer active
parameters - the opposite of the confound section 0 warns about. P1 also clears
at the bar on the subset whose active-parameter counts match (`194/348`,
bar 193).

**One unregistered observation, flagged as such.** The discovery arm's own run
is *worse* than retraining the same topology cleanly from the original
initialization: `A_adaptive - B_adaptive` is positive (mean `+0.0961`; the
clean retrain wins `271/384`). The training run that discovers a topology is not
the best use of that topology - it pays for three mid-flight rebuilds and three
optimizer resets on the way. This was not pre-registered, it is a `p = 1.0`
one-tailed artifact of the sign, and it is recorded as an observation for a
future registered study rather than as a finding.

**Consequence for the plan.** Both section-55 treatments that can be built at
toy fidelity pass, so the plan's RQ6 has a toy answer: **the topology and its
initialization are jointly the artifact; neither alone is.** The teacher
treatment (D) remains untested - it needs the Qwen-derived artifacts section 51
describes, which this project does not bundle. The committed summary is
`artifacts/study-summaries/initialization-depth4-384.json`
(`study_aef9c55774ab2c04`).

### 5.18 Pre-registration: is the genome portable, or irreducibly paired with its own initialization?

**Motivation.** Section 5.16 found the discovered topology worthless once its
weights were discarded; section 5.17 found it *valuable* when its surviving
components kept their original initialization, and found that replacing that
initialization with a fresh draw removes the value (`C_adaptive_vs_B_fixed`
fails `198/384`, reproducing 5.16). So the artifact is at least a *pair*. Two
readings survive that result and they have opposite consequences for the plan:

- **Paired**: these specific initial values are what the topology is worth. The
  genome is not a portable object, and plan section 102's "reusable
  computational genome" claim - which explicitly requires the topology to
  remain advantageous "when freshly initialized and trained independently" - is
  unsupported at toy fidelity in its own terms.
- **Portable**: the topology carries value under *any* initialization drawn
  like the original, and the fresh-draw failure of 5.16 has some other cause
  (the draw is a *different* sample, not a *foreign* one). Then the genome is
  transferable, and section 52's scale-up is the right next rung.

Distinguishing them needs one variable that 5.16 and 5.17 never varied: **whose
initialization the survivors hold, holding the topology fixed.**

**Design.** Same discovery block and registered config as 5.16/5.17
(`arms="random,adaptive,fixed"`, depth 4, `probe_candidate_policy="spread"`,
cap-4/2-batch probes, 48 steps, interval 12, budget 3, `independent_arm_rng`,
cpu), fresh block **83000-83383**, 20-way sharding. A declared **donor block
84000-84383** supplies foreign initializations (used for nothing else): seed
`s`'s donor is `s + 1000`, so the donor's initial model is a same-shape,
same-distribution draw that is foreign to the topology. Three transports per
seed:

| label | genome | initialization |
|---|---|---|
| `O` | adaptive final | **own** initial model - the section-5.17 `B_adaptive` treatment |
| `D` | adaptive final | **donor** initial model (id-aware rewind to the donor's initial weights) |
| `F` | fixed (initial) | **donor** initial model |

`F` is the control that makes `D` readable: rewinding the *unmutated* genome to
a donor's initial model copies every tensor, so `F` is exactly *the donor's
initial model itself*, trained and scored on this seed's data schedule. `D` and
`F` therefore have **identical initialization** - the donor's - and differ only
in topology (plus the adaptive genome's grown experts, which have no donor
counterpart and take the fresh draw in both). That is the cleanest available
test of the topology's standalone value, and unlike section 5.16's it is not
confounded by the two sides holding different kinds of weights.

**Primaries** (exact one-tailed sign test, n = 384, per-primary threshold
`alpha/2 = 0.025`, the section 5.9-5.17 convention, which at n = 384 is the
exact bar **212 wins**; the declared bar is asserted against `_gate_min_wins` at
run time):
1. **The pairing matters.** `O` beats `D`, in >= 212 seeds: the topology does
   better with the initialization it was found with than with a foreign one of
   the same shape and distribution.
2. **The genome is portable.** `D` beats `F`, in >= 212 seeds: under an
   initialization that is equally foreign to both sides, the discovered
   topology beats the never-mutated one.

**Predictions (declared before measurement).** Primary 1 is predicted to
**pass weakly or marginally** - it is the section-5.17 P1 comparison
(`215/384`, p `1.08e-02`) with a different non-own initialization on the far
side, and a donor draw is exchangeable with the fresh draw, so it should land
near that. Primary 2 is predicted to **fail**: two independent blocks have
already failed this comparison under own-init-and-fresh (`211/384`, `198/384`),
and nothing about a foreign initialization suggests it would rescue the
topology. A pass on primary 2 would be the first support for plan section 102's
reusability claim at any fidelity, so the prediction is recorded to make sure a
pass is not later read as a fluke or vice versa.

**Secondaries (reported, not verdict-bearing).**
- **Transport validity.** `F` must equal the *donor's* initial model trained on
  this seed's schedule; the script asserts `O` on the unmutated genome
  reproduces the `fixed` arm's own run exactly (the 5.16/5.17 identity) and that
  the donor-initialized transports copied parameters.
- **Donor exchangeability.** `D` vs the section-5.17 `C_adaptive` treatment is
  not re-run here, but `D` vs `O` and `F` vs the fixed arm's own run are both
  recorded so a reader can see whether a foreign init behaves like a fresh one.
- **Capacity.** transported parameter/active-parameter counts; adaptive
  topologies are smaller than the initial one, so a win is not bought with
  parameters.

**Pre-registered interpretation (fixed before running).**
- (a) Both primaries fail: the artifact is **irreducibly paired** - neither the
  topology alone nor a same-distribution reinitialization carries it. Plan
  section 102's reusability claim is unsupported at toy fidelity in its own
  terms ("freshly initialized and trained independently"), and the section-52
  scale-up is NOT triggered; the next registered step becomes measuring how
  much of the paired advantage is the discovery *procedure* rather than the
  artifact (the unregistered observation in 5.17.2).
- (b) P1 passes, P2 fails: the initialization is paired with the topology, and
  the genome is not portable. Same consequence as (a) with a stronger
  mechanism - the pairing is measurable, not inferred.
- (c) P1 fails, P2 passes: the pairing is not what 5.17 was measuring, and the
  topology is portable under any initialization. This would REVISE 5.17's P1
  reading (its `215/384` would be a marginal artifact) and trigger the
  section-52 scale-up.
- (d) Both pass: both the pairing and portability hold - the strongest reading,
  and section 52's scale-up is triggered with both properties declared.

**Scope limits (pre-declared).** Toy fidelity, one substrate, one donor map
(`s + 1000`, declared here). The teacher treatment remains untested for want of
Qwen artifacts. Machinery smoke-tests run on already-spent blocks
(73000-73383) only; the verdict uses 83000-83383 and nothing else.

**Cost.** 1 discovery block (3 arms) + 3 transports per seed, ~4 arm-runs of the
5.16 size; measured ~4 s/seed single-process at depth 4, so ~25 min of CPU
across 20 shards.

### 5.18.1 Pre-run correction: primary 1 must be a difference-in-differences

Recorded before the registered block was run and before any primary was read,
from the machinery smoke test on spent block 73000-73003.

As declared, primary 1 compares `O` against `D`, which differ in initialization
source but share the adaptive topology. That contrast is dominated by something
else: this seed's own-versus-donor initialization *quality*, which has nothing
to do with the topology and is large. On the smoke seeds:

| seed | `O - D` (adaptive topology) | own(initial) `- F` (unmutated topology) | difference |
|---|---|---|---|
| 73000 | `-0.0546` | `+0.0674` | `-0.1220` |
| 73001 | `+0.2216` | `-0.0247` | `+0.2463` |
| 73002 | `+1.5949` | `+1.6336` | `-0.0387` |
| 73003 | `-0.2080` | `-0.2288` | `+0.0208` |
| mean | `+0.3885` | `+0.3619` | `+0.0266` |

Columns 2 and 3 are nearly identical - both are mostly measuring *the same* draw
quality difference, with a standard deviation of `0.86`, roughly five times the
`~0.1` topology effect this study is chasing. Primary 1 as declared would
largely test which seed got the luckier pair of initializations.

The correction: primary 1 becomes the **difference-in-differences**, the
quantity columns 2 and 3 differ by. It subtracts the same seed-level
initialization-quality term on the never-mutated topology, so what remains is
the *extra* own-initialization advantage the discovered topology earns - the
pairing effect itself. `D_vs_F` (primary 2) already holds initialization
identical on both sides and needs no correction. The raw `O - D` is still
reported, as a secondary.

Two things about this correction are worth stating plainly, because it changes a
pre-registered primary. First, it was chosen from a *spent* block, for variance
reasons, before the registered block ran - and the smoke block's own
DiD mean, `+0.0266`, points **against** the pairing hypothesis, so this is not a
choice in the study's favour; a design picked to flatter the hypothesis would
have kept the raw column. Second, the prediction for the amended primary 1 is
weaker than for the original, and is recorded as such:

> Section 5.17's `215/384` was itself a raw own-versus-non-own contrast of the
> contaminated kind, so removing the seed-level initialization term may leave a
> smaller residual. Amended primary 1 is predicted to **fail**; a pass means the
> pairing survives the subtraction, i.e. the discovered topology earns
> materially more from its own initialization than the unmutated one does.

Primary 2 (`D` < `F`) is unchanged, and its prediction is unchanged: **fail**.
The four interpretation branches are restated over the amended primary 1
(pairing) and primary 2 (portability); nothing else in section 5.18 changes.

### 5.18.2 Outcome: registered branch (d) - the genome IS portable, and its own initialization adds value

Run 2026-09-15, block 83000-83383 (fresh), donor block 84000-84383, 20 shards,
revision `137f6d0`. All built-in checks held on every seed: own-initialization
rewinding reproduces the `fixed` arm's own run **exactly** (loss and FLOPs,
384/384), the donor rewind replaces **every** tensor of the unmutated genome
(`skipped_parameters == 0`, so `F` really is the donor's model), and every
depicted transport transferred parameters.

| comparison (n = 384, bar 212) | wins | mean delta | p | prediction | verdict |
|---|---|---|---|---|---|
| **P1** pairing, difference-in-differences | 237/384 | `-0.0830` | `2.54e-06` | fail | **PASS** |
| **P2** `D` < `F` (topology portable) | 234/384 | `-0.0465` | `1.06e-05` | fail | **PASS** |
| raw `O` < `D` (contaminated, secondary) | 199/384 | `-0.1452` | `2.54e-01` | n/a | fail |
| topologies matched on *own* init: `O` < `O_fixed` | 293/384 | `-0.1295` | `3.65e-26` | n/a | pass |
| donor init on *both* sides: `D` < `O_fixed` | 194/384 | `+0.0156` | `4.39e-01` | n/a | fail |
| was the donor init better for the initial topology? `F` vs own | 190/384 | `+0.0621` | `6.01e-01` | n/a | fail (exchangeable) |
| (discovery) adaptive < random | 264/384 | `-0.1084` | `7.64e-14` | | pass |
| (discovery) adaptive < fixed | 218/384 | `-0.0306` | `4.58e-03` | | pass |

Transport means: `O 12.2707 < O_fixed 12.4003 < D 12.4159 < F 12.4624`.

**Both pre-registered predictions were wrong, and the correction in 5.18.1 is
what made the measurement able to show it.** The amended primary 1 was
predicted to fail; it passes at p `2.5e-06` - the discovered topology earns
materially *more* from its own initialization than the never-mutated topology
does, which is the pairing effect measured cleanly. Primary 2 was predicted to
fail on the strength of two earlier blocks; it passes at p `1.1e-05` - and that
is the first support this project has for plan section 102's reusability claim
at any fidelity.

**The synthesis across 5.16-5.18, which is the point of running all three.** Put
the topology-versus-initial-topology comparisons side by side by what the two
sides are initialized with:

| initialization | comparison | wins | reading |
|---|---|---|---|
| both own (matched) | 5.17 `B_adaptive` vs `B_fixed`; 5.18 `O` vs `O_fixed` | 274/384, 293/384 | topology wins |
| both donor (matched) | 5.18 `D` vs `F` | 234/384 | topology wins |
| adaptive fresh, control own (mismatched) | 5.16 P2; 5.17 `C_adaptive` vs `B_fixed` | 211/384, 198/384 | fails |

Every **matched-initialization** comparison wins and both **mismatched** ones
fail. So sections 5.16's branch (c) and 5.17's failure rows were not measuring a
weak topology: they were measuring a **control whose initialization was not
matched to the treatment's** - the adaptive side carried a redrawn
initialization while the control carried its original one, and the seed-level
quality difference between those two draws (standard deviation `0.86` on the
smoke block) swamps the topology effect (~`0.1`). That is exactly the term the
difference-in-differences removes, and with it removed the pairing effect is
`p = 2.5e-06` rather than invisible.

**What the artifact is, then, at toy fidelity.** The discovered topology has
standalone value - it beats the never-mutated topology when both are given the
*same* initialization, whether that initialization is their own (293/384) or a
foreign donor's (234/384). It also extracts *additional* value from its own
initialization, beyond what the never-mutated topology extracts from it
(237/384 on the differenced contrast). So the plan's two claims are both
supported, and they are not competitors: **section 102's reusable genome is real
when the comparison is initialization-matched, and section 11's "final topology
+ original initialization" is real as a further gain on top.**

**Limits, recorded rather than glossed.** (i) Primary 1's prediction was wrong
in the direction of *understating* the effect, so the amended statistic should
not be read as having been chosen because it worked - 5.18.1 records that it was
chosen before the run, on a spent block, and that the smoke block's own DiD mean
pointed against the hypothesis. (ii) In `D` the adaptive topology's *grown*
experts (and its resized routers) still hold fresh draws, because no donor
component corresponds to them, so "donor-initialized" is exact for components
that existed at initialization and fresh for the rest; the grown-expert subgroup
split is therefore not reported for this block. (iii) The donor map (`s+1000`)
is one declared choice; a different donor distribution is untested. (iv) The
teacher treatment remains untested for want of the Qwen artifacts section 51
describes.

The committed summary is `artifacts/study-summaries/portability-depth4-384.json`
(`study_757e78b90de8a510`).

### 5.19 Pre-registration: does the matched-initialization topology value share the depth-4 onset?

**Motivation.** Section 5.18 leaves the project's newest positive result attached
to a single depth. That depth is not arbitrary: sections 5.12 and 5.15 located
the *while-training* adaptive value-creation onset between depth 3 and depth 4
(adaptive-vs-null wins `117/384` at depth 1, `161/384` at depth 2, `188/384` at
3, `225/384` at 4 - monotone, with the sign of the mean flipping just above
depth 3). If the topology's matched-initialization value comes from the same
mechanism, it should onset at the same place. If it does not, the onset the
depth dose-response found is a property of the **controller** (probe-based
selection needs a substrate rich enough to discriminate) rather than of the
substrate - which would REDIRECT the project toward the controller instead of
toward depth.

**Design.** Section 5.18's design unchanged except for depth: the same registered
config, the same four transports (`O`, `D`, `F`, `O_fixed`), the same donor
mechanism. Two fresh blocks, 20-way sharding each:

| block | seeds | depth | donor offset | donor block |
|---|---|---|---|---|
| depth 2 | 85000-85383 | 2 | +2000 | 87000-87383 |
| depth 3 | 86000-86383 | 3 | +2000 | 88000-88383 |

Declared difference from the earlier shallow-depth cells: sections 5.12's and
5.15's depth-1/2/3 cells used the `prefix` candidate policy, while these use
`spread` - matching 5.18 exactly so the depths are comparable to each other.
Any disagreement with 5.15's cells therefore has the policy as an alternative
explanation and is recorded as such rather than attributed to depth.

**Primaries (per depth, n = 384):**
1. **matched topology value** - `O` < `O_fixed` (both sides on their own
   initialization, so only the topology differs).
2. **pairing** - the difference-in-differences of section 5.18.1.

Two depths x two primaries = four tests, so the declared per-primary threshold
is **Bonferroni `alpha/4 = 0.0125`**, which at n = 384 is the exact bar **215**
(not 212 - the script computes and asserts its own declared bar at run time).
The section-5.18 depth-4 rows (`O_vs_O_fixed` `293/384`; DiD `237/384`) are
reported as context but come from a different block and a looser threshold
(`0.025`), so they are not part of this study's family.

**Predictions (declared before measurement).** Both primaries are predicted to
**fail at both depths**. The topology's matched-init value at depth 4
(`O` wins `293/384`, mean `-0.1295`) is the largest effect this project has
measured, and if it comes from the same expressiveness the controller's onset
found, there is nothing at depth 2 (where every mutating arm loses to doing
nothing) for it to exploit. A pass at either depth is the more informative
outcome and means the shallow-depth failure belongs to the selection rule, not
to the substrate.

**Secondaries (reported, not verdict-bearing).** Portability per depth (`D` <
`F`); the raw contaminated contrast `O` vs `D`; the donor-init control (`F` vs
the fixed arm's own run); capacity (active-parameter counts, which the depth-4
block showed favour the initial topology); and the discovery-time ordering, for
continuity with section 5.15's cells.

**Pre-registered interpretation (fixed before running).**
- (a) Both depths fail both primaries: **co-located onsets** - the matched-init
topology value is depth-4-specific, sharing the threshold the controller's
while-training value has (§5.15's branch (b)). The section-52 scale-up must not
assume the effect exists at smaller widths/depths, and "depth buys
expressiveness that mutations can exploit" becomes the joint reading.
- (b) Depth 2 fails, depth 3 passes: the topology onset sits between 2 and 3,
one step below the controller's - partial co-location, and the controller's
depth-3 failure becomes a selection problem at a depth where the substrate
already supports value.
- (c) Both depths pass: the topology mechanism is **not** depth-limited at these
sizes, so §5.15's onset is a property of probe-based selection. The project
redirects to controller research (why does the probe fail to find structure
that is there?), and the section-52 scale-up is deprioritised.
- (d) Depth 2 passes, depth 3 fails: non-monotone - the depth axis reading is
suspended pending a follow-up, and no other reading is taken.

**Scope limits (pre-declared).** Toy fidelity, one substrate, `spread` policy at
all depths, one donor map (+2000). The teacher treatment remains untested.
Machinery smoke-tests run on spent blocks only; the verdicts use 85000-85383 and
86000-86383 and nothing else.

**Cost.** 2 blocks x (1 discovery + 4 transports) per seed; shallow depths are
cheaper per seed than depth 4, so ~45 min of CPU total across 20 shards per
block.

### 5.19 Outcome: both depths pass - the matched-init value is not depth-limited (branch (c))

The registered blocks 85000-85383 (depth 2) and 86000-86383 (depth 3), donors
+2000, 384 seeds each. **A bookkeeping gap is recorded here:** this run's
committed summaries were never written - section 5.20's motivation cites its
numbers (`260/384`, `270/384`) with no artifact behind them. The shards survived
under `artifacts/toy-portability-study/{depth2,depth3}` and the summaries are now
generated from them: `study_436e3cfb83e13fc0` and `study_ea20b7b1ace9534b`. Both
built-in validity checks hold in both blocks (own rewind reproduces the fixed
arm exactly; donor rewind replaces every tensor of the unmutated genome).

| | depth 2 (n = 384) | depth 3 (n = 384) |
|---|---|---|
| **P1** pairing, difference-in-differences | **219/384**, mean −0.0443, p 3.38e-03 | **226/384**, mean −0.0641, p 3.05e-04 |
| **P2** `O` < `O_fixed` (matched topology value) | **260/384**, mean −0.0618, p 1.68e-12 | **270/384**, mean −0.1042, p 5.19e-16 |
| secondary `D` < `F` (portability) | 211/384, mean −0.0175, p 2.94e-02 | 216/384, mean −0.0400, p 8.18e-03 |
| secondary `O` < `D` (raw, contaminated) | 177/384, mean +0.0376, p 9.43e-01 | 202/384, mean −0.0871, p 1.66e-01 |

**Registered verdict: branch (c).** Both depths pass both primaries, so the
topology mechanism is **not** depth-limited at these sizes and section 5.15's
depth-4 onset is a property of **probe-based selection**, not of the substrate.
The pre-registered consequence is a redirect to controller research - why does
the probe fail to find structure that is demonstrably there? - and a
deprioritisation of the section-52 scale-up. The raw own-minus-donor secondary
fails at both depths, exactly as section 5.18.1's correction predicted.

**Prediction wrong again, in the same direction.** Section 5.19 declared that
*both* primaries would fail at *both* depths; all four passed. That is the fourth
of my last six predictions to be wrong and the fourth to understate an effect.

**One trap worth naming.** `scripts/portability_study.py` prints an automatic
`outcome` label whose letters index **section 5.18's** four-branch contract
(its primaries are the DiD and `D` < `F`), so the depth-2 record carries the
string "branch (b) ... the genome is not portable". That label is *not* section
5.19's registered branch, which is (c) as read from the two primaries above. The
report is authoritative; the script's label is retained for continuity with the
committed depth-4 record rather than silently relabelled.

### 5.20 Pre-registration: is the shallow-depth topology value *selection*-specific?

**Motivation.** Section 5.19 measured, at depths 2 and 3, a large
matched-initialization advantage for the topology the adaptive arm ends up with
(`260/384` and `270/384`, p `1.7e-12` and `5.2e-16`) - larger than anything the
*while-training* comparison shows at those depths, where the adaptive arm loses
to doing nothing (`161/384`, `188/384`). The obvious alternative explanation is
that the comparison is not measuring adaptive *selection* at all:

- **selection-specific**: the controller chose structure that trains better when
  restarted, so the value is in the choice.
- **shrinkage / any-churn**: the arm prunes experts, the adaptive topologies are
  smaller than the initial one (`346/384` and `353/384` seeds), and perhaps *any*
  churned - or any smaller - topology retrains better on this substrate. Then
  section 5.19's result is a capacity/regularization effect and the controller
  is incidental.

Section 5.17's block already separates these at depth 4, and separates them
cleanly: under the *same* rewound treatment, the **random-churn** topology is
indistinguishable from the initial topology (`B_random` vs `B_fixed`:
`205/384`, mean `-0.0024`, p `0.101`, i.e. essentially zero mean) while the
adaptive topology beats both it (`255/384`, mean `-0.1099`, p `6.13e-11`) and
the initial topology (`274/384`, p `1.46e-17`). Nothing like that measurement
exists at depths 2 and 3, and that is the gap this study closes.

**Design.** Three rewound transports per seed, all on the seed's **own** initial
model so initialization is matched on every side: `A_own` (adaptive genome),
`R_own` (random-churn genome), `F_own` (unmutated genome, which reproduces the
fixed arm's own run exactly and is the validity check). Fresh blocks, 20-way
sharding:

| block | seeds | depth |
|---|---|---|
| depth 2 | 89000-89383 | 2 |
| depth 3 | 91000-91383 | 3 |

Same registered config and `spread` policy as sections 5.18-5.19.

**Primaries (per depth, n = 384), threshold Bonferroni `alpha/4 = 0.0125`
(exact bar **215**, computed and asserted at run time):**
1. **selection-specific value** - `A_own` < `R_own`.
2. **matched-init value replicates** - `A_own` < `F_own`, which is section
   5.19's primary 1 measured again on an unspent block.

The `R_own` < `F_own` row is reported as the **control that defines what "value"
means here**: if it passes, churn alone produces better-retraining topologies and
primary 1's interpretation inverts.

**Predictions (declared before measurement).** Two of my last four predictions
have been wrong and both times in the direction of understating an effect, so
this is stated carefully. Primary 1 is predicted to **fail at depth 2 and pass
at depth 3**: the controller's selection signal is weakest where section 5.15
found its while-training value weakest (depth 2), while the shrinkage
alternative should not depend on depth. Primary 2 is predicted to **pass at both
depths** (replication of 5.19's `260/384` and `270/384`). A depth-2 primary-1
pass would mean the selection signal is real even where the while-training
comparison cannot see it, which is the reading section 5.19 could not
distinguish.

**Secondaries (reported, not verdict-bearing).** Capacity: the active-parameter
counts of all three topologies per seed, so "is the adaptive topology just
smaller?" is answerable directly and not by inference - if `R_own`'s topologies
are pruned to a similar size, size cannot explain primary 1, and if they are not,
primary 1's interpretation is recorded as capacity-confounded. Also the
random-topology control (`R_own` vs `F_own`), the discovery-time ordering for
continuity with section 5.15, and the raw transported losses.

**Pre-registered interpretation (fixed before running).**
- (a) Primary 1 passes at both depths and primary 2 replicates: the shallow-depth
topology value is **selection-specific**, so section 5.15's depth-4 onset is a
*selection* onset and section 5.19's branch (c) stands - the controller, not the
substrate, is what fails at shallow depth. Redirects to controller research and
deprioritises the section-52 scale-up.
- (b) Primary 1 fails at both depths: the shallow-depth advantage is **not**
selection-specific. Section 5.19's result is re-read as shrinkage/any-churn, the
"reusable genome" claim is weakened to "smaller topologies retrain better at
shallow depth", and the section-52 scale-up must carry this control.
- (c) Primary 1 fails at depth 2 / passes at depth 3: specificity needs depth -
consistent with a selection signal that scales with the substrate, and the
depth-2 result is recorded as a shrinkage effect specifically.
- (d) Primary 2 fails at either depth: section 5.19's own result does not
replicate on a fresh block. That is recorded as a replication failure, the
depth-2/3 matched-init result is withdrawn pending a follow-up, and no other
reading is taken from this study.

**Scope limits (pre-declared).** Toy fidelity, one substrate, `spread` policy,
own-initialization only (the donor machinery of 5.18 is not re-run here). The
teacher treatment remains untested. Smoke tests on spent blocks only; the
verdicts use 89000-89383 and 91000-91383 and nothing else.

**Cost.** 2 blocks x (1 discovery + 3 transports) per seed, ~3.5 s/seed at these
depths; ~45 min of CPU across 20 shards per block.

### 5.20 Result: selection-specific at both depths (branch (a))

Run on the registered blocks only (89000-89383 depth 2, 91000-91383 depth 3),
`study_f10d1b1cd3812bc1` and `study_b6455e54f8801fd8`. Transport validity held:
the unmutated genome rewound to the seed's own initial model reproduced the fixed
arm's own discovery run exactly in **384/384 seeds at both depths**, so every
delta below is a topology effect with initialization matched on all three sides.

| | depth 2 (n = 384) | depth 3 (n = 384) |
|---|---|---|
| **P1** `A_own` < `R_own` (selection-specific) | **243/384**, mean −0.0515, p 1.09e-07 | **248/384**, mean −0.0817, p 5.86e-09 |
| **P2** `A_own` < `F_own` (replicates §5.19) | **264/384**, mean −0.0729, p 7.64e-14 | **265/384**, mean −0.0829, p 3.43e-14 |
| control `R_own` < `F_own` (churn alone) | 213/384, mean −0.0213, p 1.81e-02 | 197/384, mean −0.0012, p 3.23e-01 |

Bar 215 at the pre-registered `alpha/4 = 0.0125`; P1 and P2 pass at both depths.

**Registered verdict: branch (a).** The shallow-depth topology value is
**selection-specific**, so section 5.15's depth-4 onset is a *selection* onset,
section 5.19's branch (c) stands, and the controller - not the substrate - is
what fails at shallow depth. The pre-registered consequence is a redirect to
controller research and a **deprioritisation of the section-52 scale-up**.

**My prediction was wrong, in the same direction as before.** I declared P1
"fail at depth 2, pass at depth 3", reasoning that the selection signal should be
weakest where section 5.15 found its while-training value weakest. P1 passed at
depth 2 by a wide margin. This is the third of my last five predictions to be
wrong, and all three understated an effect; the pattern is recorded rather than
explained away.

**The control is clean at depth 3, and not decisively zero at depth 2.** At
depth 3 the random-churn topology is indistinguishable from the initial one
(mean −0.0012, 1.5% of the total). At depth 2 churn contributes a real fraction -
mean −0.0213, about 29% of the total `A_own`-vs-`F_own` effect, nominally
significant (p 0.018) though failing the declared Bonferroni bar. So the depth-2
result is *largely* selection-specific rather than purely so, and the direct
P1 contrast (−0.0515, p 1.09e-07) is the number to quote, not the total.

**Capacity does not explain it.** The adaptive topology is smaller than the
random-churn one in 257/384 (depth 2) and 250/384 (depth 3) seeds - an advantage,
not a dominant one - and the random-churn topology is the same size as the
initial one in **0/384** seeds at both depths. Churn therefore did prune, in
every seed, and still produced no retraining advantage: shrinking is not the
mechanism.

### 5.20.1 Amendment (made after the block ran): the coincidence guard

The first verdict attempt **withheld the depth-2 block**, because a built-in
check asserted that the adaptive and random arms end on different genomes in
*every* seed and one seed (89193, of 384) does not. Depth 2's feasible topology
space is 2 layers x 4 experts over 3 mutations, so two independent draws landing
on the same genome is expected, not degenerate. The check was replaced by what
it needs to assert:

- coincident arms must be **exact ties** - same genome *and* same
  initialization, so the seed's delta is exactly 0 and it can never be counted a
  win; and
- coincident arms must be **rare** (<= 1% of the block).

The amendment cannot manufacture the result, and this is the point worth
checking: it changes no statistic. The *declared* reading - all 384 seeds, the
tie counted as a non-win - already passes P1 (243/384, p 1.09e-07) and P2
(264/384, p 7.64e-14). Dropping the tie seed moves nothing: P1 243/383,
p 7.92e-08; P2 263/383, p 1.05e-13. A tie can only be a non-win, so this
ambiguity is conservative in the direction of the primary. The per-seed data is
committed in both summaries, so the differ-only statistics recompute from the
artifact. The depth-3 block was unaffected (0/384 coincident).

### 5.21 Observation (unregistered, cross-study): the fused run is dominated by its own final topology, at every depth and for *both* mutating arms

**This is not a pre-registered result.** It is a reanalysis of three blocks that
were run for other reasons (depth 2 and 3 for section 5.20, depth 4 for section
5.17), found while checking section 5.17.2's unregistered observation for
recurrence. Section 5.17.2 already reported the depth-4 case and did not check
the random arm; the new content here is that the effect **replicates at depths 2
and 3 and holds for the random arm as well**, which changes the reading.

**The comparison.** Every one of these blocks measured, per seed, the arm's own
fused run (48 steps, three mutations applied in place) and a *frozen retrain* of
that arm's final genome from the same initialization on the same data schedule.
The frozen path differs in exactly the ways "freeze it" implies: no mid-flight
architecture changes, no optimizer resets, and the final topology present from
step 0. Positive deltas below mean the fused run lost.

| depth | arm | frozen retrain wins | mean delta (fused − frozen) | p |
|---|---|---|---|---|
| 2 | adaptive | **314/384** | +0.1027 | 2.66e-38 |
| 2 | random | **292/384** | +0.1050 | 1.17e-25 |
| 3 | adaptive | **288/384** | +0.0913 | 1.07e-23 |
| 3 | random | **293/384** | +0.1248 | 3.65e-26 |
| 4 | adaptive | **271/384** | +0.0961 | 2.16e-16 |
| 4 | random | **258/384** | +0.0803 | 7.33e-12 |

Validity: at all three depths the *unmutated* genome transported to its own
initialization reproduces that arm's own run with max |delta| **exactly 0.0**,
which is the identity the transport is built to satisfy and the reason a
topology effect can be separated from a transport artifact at all.

**Why this matters more than the depth-4 original did.** Section 5.17.2's
version could be read as a fact about the *discovered* topology. It is not: the
**random** arm's frozen retrain beats its own fused run by as much or more
(depth 3: +0.1248 vs the adaptive arm's +0.0913). The penalty is therefore a
property of the **process** - mutating a live training run in place - not of
which topology the controller picked. Two consequences follow.

1. **The ~0.10 nat process penalty is roughly twice the topology effect**
   sections 5.17-5.20 spend their power measuring (means of −0.05 to −0.13). The
   fused while-training comparison is not a sensitive instrument for selection
   value: it measures the topology signal *through* a process penalty of
   comparable size, which is a sufficient explanation of why that comparison is
   weak-to-negative at shallow depth while section 5.20 shows the topology value
   is real and selection-specific there.
2. **Two-stage delivery is not an analysis convenience, it is a requirement.**
   Plan section 102's "extract the genome" and section 11's weight/topology
   separation, and the `07_extract` -> `08_retrain` notebook pair, describe the
   pipeline that this measurement says is *better*, not merely tidier.

**Caveats, stated plainly.** Equal step budget and identical data schedule, but
neither equal FLOPs nor equal optimizer history - the frozen path trains a
pruned topology (the random-churn topology is never the initial size in 0/384
seeds at every depth), so it likely does *less* per-step work, which if anything
strengthens the direction. The frozen path also starts a fresh optimizer rather
than carrying state across mutations, and that difference is inside "the
process" as defined here, so it is named as a mechanism candidate rather than
controlled out. Toy fidelity, synthetic tokens, one substrate throughout. The
sign of this effect was visible in section 5.17.2 before any of today's blocks
ran, so this is a replication of an existing observation, not a discovery.

### 5.22 Pre-registration: is the mid-training process penalty real and arm-generic?

**Motivation.** Section 5.21 is an unregistered reanalysis - three blocks, chosen
after the fact, on which the desired contrast happens to be computable. Its
claim (the fused run is dominated by its own frozen final topology, and the
penalty is arm-generic rather than topology-specific) needs a block that was
collected for *this* question, with the threshold declared before the run.

**Design.** Section 5.20's registered design exactly - the registered toy config
(`spread` policy, cap-4/2-batch probes, 48 steps, interval 12, budget 3,
`independent_arm_rng`, arms `random,adaptive,fixed`), three rewound transports
per seed on the seed's own initial model. No new machinery: the frozen retrain of
each arm's final genome is already computed as `A_own` / `R_own` / `F_own`, and
`F_own` reproducing the fixed arm's own run exactly remains the validity check.
Fresh depth-2 and depth-3 blocks, 20-way sharding:

| block | seeds | depth |
|---|---|---|
| depth 2 | 92000-92383 | 2 |
| depth 3 | 93000-93383 | 3 |

**Primaries (per depth, n = 384), Bonferroni `alpha/4 = 0.0125` -> bar 215.**
1. **process penalty, adaptive** - fused adaptive run *loses* to `A_own`.
2. **process penalty, random** - fused random run *loses* to `R_own`.

Both primaries are directional (the frozen retrain must win), so neither can be
satisfied by the sign that section 5.21 discounts.

**Predictions (declared before measurement).** Both pass at both depths. This is
a *replication* pre-registration and is declared as such: the effect was visible
at 384-seed scale on three separate blocks with p below 1e-11 everywhere, so a
failure here would be more informative than a pass, and the diagnostic value is
in the asymmetry between the two primaries rather than in either alone.

**Pre-registered interpretation (fixed before running).**
- (a) Both primaries pass at both depths: the penalty is a **process** property.
  The fused run is recorded as the wrong instrument for selection value, the
two-stage pipeline (discover -> freeze -> retrain) is adopted as the project's
delivery path, and section 5.15's depth dose-response is re-read as a statement
about the *fused* process rather than about where value becomes discoverable.
- (b) Only the adaptive row passes: the penalty is **topology-specific** after
  all, section 5.21's arm-generic reading is withdrawn, and the adaptive arm's
  mutations are the disruptive ones - a materially different conclusion that
  would restore the fused comparison as the right instrument.
- (c) Neither passes at a depth where section 5.21 found it: **replication
  failure**. Section 5.17.2's observation and section 5.21's extension are both
  marked unconfirmed, and no delivery-path change is adopted.

**Secondaries (reported, not verdict-bearing).** The same contrast at depth 4 on
this block family is *not* run (depth 4's evidence is section 5.17's block); the
fused-to-frozen delta split by whether the frozen topology is pure-prune or has
grown experts; the active-parameter ratio between fused and frozen, so the
unequal-FLOPs caveat is quantified rather than asserted; and the frozen-vs-fixed
rows (`A_own` vs `F_own`), which section 5.20 already measured, as the check that
this block family behaves like its predecessor.

**Scope limits (pre-declared).** Toy fidelity, one substrate, synthetic tokens,
`spread` policy. Equal step budget, unequal FLOPs and unequal optimizer history
are properties of the contrast, not controlled out - if the effect survives here
the *mechanism* still needs its own study (fresh-optimizer control, matched-FLOPs
control). The teacher treatment remains untested.

**Cost.** Identical to section 5.20: 2 blocks x (1 discovery + 3 transports) per
seed, ~3.5 s/seed, ~45 min CPU across 20 shards per block.

### 5.22 Outcome: the penalty is a process property (branch (a), both depths)

Run on the registered fresh blocks 92000-92383 (depth 2) and 93000-93383 (depth
3), `study_bc1a4fdfe99fde63` and `study_f23d93bf49e18a57`. The design is section
5.20's exactly, so the validity check carries over: the unmutated genome rewound
to the seed's own initial model reproduced the fixed arm's own run exactly in
384/384 seeds at both depths.

| | depth 2 (n = 384) | depth 3 (n = 384) |
|---|---|---|
| **P1** frozen adaptive retrain beats fused adaptive run | **296/384**, mean −0.0795, p 1.02e-27 | **291/384**, mean −0.0936, p 3.69e-25 |
| **P2** frozen random retrain beats fused random run | **286/384**, mean −0.0996, p 9.45e-23 | **300/384**, mean −0.1219, p 7.00e-30 |

Bar 215 at `alpha/4 = 0.0125`; both primaries pass at both depths.

**Registered verdict: branch (a) - the penalty is a process property, not a
topology one.** The random arm's frozen retrain beats its own fused run by *more*
than the adaptive arm's does at both depths, exactly as section 5.21 predicted
from the reanalysis. Section 5.21 is therefore confirmed, and section 5.17.2's
narrower reading (an observation about the discovered topology) is superseded.

**The pre-registered consequence is adopted:** the fused run is the wrong
instrument for selection value, and the two-stage delivery path (discover ->
freeze the genome -> retrain from the original initialization) becomes the
project's pipeline. Plan section 102's genome extraction and section 11's
weight/topology separation are thereby a performance requirement, not bookkeeping,
and section 5.15's depth dose-response is re-read as a statement about the
**fused** process rather than about where value becomes discoverable.

> **Superseded in part - read this with sections 5.23-5.28.** The consequence
> adopted above was earned on the *then-current* mutation path, and sections 5.23
> to 5.28 decompose precisely that path. Optimizer history accounts for none of
> the penalty (5.23); a discarded router accounted for over half and is a
> **defect**, not a design property (5.24); and the remainder is the freshly drawn
> new expert, which cloning **inverts** rather than merely removes (5.26, 5.27 -
> the body is ~3/4 of it, the copied router row the rest). Section 5.28 then ran
> the drawn-mutation regime this section was *adopted from* and found that with
> `preserve_router_on_rebuild=True` and `new_expert_initialization="clone"` the
> historical path is worse than never mutating at all. **The pipeline recommended
> here - freeze the genome and retrain - is therefore a regression on this
> substrate**, and the recommended pipeline is now clone-and-preserve, keeping the
> fused run. The historical text above is left as written because it is the record
> of what this block licensed at the time; it is not current guidance.

**Both of this block family's other readings replicated**, which is what makes it
a usable family. Section 5.20's primaries: `A_own` < `R_own` at `256/384` and
`255/384`, `A_own` < `F_own` at `257/384` and `272/384` - a fourth and fifth
independent block agreeing with section 5.20's branch (a). And the random-churn
control came back neutral again at depth 2 (`195/384`, mean −0.0042, p 0.399) and
near-neutral at depth 3 (`214/384`, mean −0.0204, p 0.014, failing the
Bonferroni bar), matching section 5.20's control rows rather than contradicting
them.

**What this does not settle.** The mechanism inside "the process" is unseparated:
a frozen retrain starts a fresh optimizer and trains a pruned topology, so
neither optimizer history nor per-step FLOPs are matched, and the pre-registration
declared both as uncontrolled. Two candidate mechanisms - mid-flight optimizer
resets, and capacity loss from pruning - remain live, and the frozen path is
*different work* rather than strictly less. The teacher treatment is still
untested.

### 5.22.1 Post-hoc: the penalty is not a capacity artifact

Section 5.22's adopted consequence - two-stage delivery - is only earned if the
penalty is a **process** effect. The competing reading is that a frozen retrain
simply trains a *smaller* model (mutating arms prune), and that smaller models
train better at this scale, in which case the operational lesson would be "prune
early", not "freeze and retrain". Two independent checks over all five blocks
that carry these contrasts rule that out.

**1. The penalty does not track how much the topology changed size.** Per seed,
correlating the fused-run penalty against the frozen topology's size change:

| block | corr(penalty, size change) | seeds where the topology shrank |
|---|---|---|
| depth 2 (5.20) | −0.039 | 52% |
| depth 3 (5.20) | −0.051 | 56% |
| depth 4 (5.17) | −0.002 | 54% |
| depth 2 (5.22) | +0.013 | 46% |
| depth 3 (5.22) | +0.046 | 48% |

Every correlation is within ±0.05 of zero, and four of five are negative - seeds
whose topology shrank *more* paid a slightly *smaller* penalty. The last column
matters on its own: random churn **grows** experts in roughly half the seeds, so
"the run got pruned" is not even the typical case the penalty is charged in.

**2. Size buys nothing when the process is removed.** Comparing the frozen
retrain of the churned topology against the frozen retrain of the *unpruned*
initial topology - same process on both sides, so this is a pure capacity
comparison - the churned topology is not better in any block: `213/384`, `197/384`,
`205/384`, `195/384`, `214/384` (means −0.021 to −0.001). A pruned topology,
retrained, is indistinguishable from the unpruned one. "Smaller models train
better at this scale" is therefore false at this fidelity.

Together these say the fused run loses ~0.1 nats to something the *process* does,
not to the size of the model it ends up with. The frozen path also does *less*
per-step work when the topology shrank, which pushes the same direction. **The
one live confound left is optimizer history**: a frozen retrain starts a fresh
optimizer while the fused run rebuilds one at each of its three mutations, and
that is not separable from these blocks. It is named here rather than controlled,
and is the next registered axis.

### 5.23 Pre-registration: is the mid-training penalty optimizer history or the architecture change?

**Motivation.** Section 5.22 established the penalty and section 5.22.1 ruled
out the size of the resulting topology. One confound is left, and it is a
mechanical one rather than a statistical one: the mutation path re-creates the
optimizer whenever it rebuilds the model - `_mutate_by_probe` and the random path
both assign a fresh `torch.optim.AdamW` after `rebuild_model_from_genome` - so
every event discards all Adam moment estimates. A fused run therefore pays three
full Adam restarts that a frozen retrain never pays. If that is the whole story,
the ~0.10-nat penalty is **not** evidence that mutating a live run is
architecturally harmful, and section 5.22's adopted consequence would be
qualified: the cheap fix would be to preserve optimizer state across mutations
rather than to abandon the fused pipeline.

**Design.** New opt-in knob `ExperimentConfig.optimizer_reset_steps`: at the named
steps every arm's optimizer is re-created from scratch with the genome and the
weights untouched. It is empty by default, so no published number moves. Per seed,
on one initial model and one data schedule:

| run | what it is | differs from | by |
|---|---|---|---|
| `F` | the `fixed` arm's own run | - | - |
| `U` | the same unmutated genome, optimizer reset at 12/24/36 | `F` | optimizer history **alone** |
| `R` | the fused `random` arm | `U` | the mid-flight rebuild |
| `A` | the fused `adaptive` arm | `U` | the mid-flight rebuild |

`F` and `U` share a seed, an initial model, a data schedule and a genome (the last
is a built-in check), so a difference between them cannot be anything but the
optimizer. Fresh blocks, 20-way sharding: **depth 2 = 94000-94383**, **depth 3 =
95000-95383**.

**Primaries (per depth, n = 384), Bonferroni `alpha/4 = 0.0125` -> bar 215.**
1. **reset cost** - `F` < `U`. Does optimizer history alone move the loss?
2. **architecture cost beyond resets** - `U` < `R`. Is the fused run worse than a
   run that only resets?

**Prediction (declared before the registered blocks, and informed by a spent-block
smoke that is disclosed rather than hidden).** The machinery smoke ran on seeds
73000-73002 (spent): the reset-only twin was near-neutral and flipped sign across
seeds (per-seed `+0.025`, `+0.050`, `-0.024`, mean `+0.017`), against a penalty of
`~0.10`. That is a factor of six smaller and not consistently signed, so **P1 is
predicted to FAIL and P2 to PASS: branch (c).** Stating this is the point of the
smoke convention used elsewhere in this report - a prediction chosen from a spent
block is declared as such, and here it points *against* the mechanism the design
was built to detect. Three seeds at depth 4 cannot settle it, which is why the
registered block exists.

**Pre-registered interpretation (fixed before running).**
- (a) P1 and P2 both pass: **two components** - Adam restarts cost something and
  the rebuild costs something on top. Two-stage delivery stands, and
  optimizer-preserving mutation becomes a partial fix worth its own study.
- (b) P1 passes and P2 fails: the penalty is **essentially all optimizer reset**.
  Section 5.22's delivery change is qualified (the fused pipeline is viable if
  Adam state survives), and the recommended fix is optimizer preservation.
- (c) P1 fails: **resets are not the cost** - the architecture change is.
  Section 5.22's adopted consequence stands unqualified and no cheaper fix
  exists on this axis. This is the branch the disclosed smoke predicts.

**Secondaries (reported, not verdict-bearing).** `U` vs `A` (the adaptive arm's
penalty against the same reset-only reference); the reset step count and their
sizing against the fused schedule; and the reset-only loss means beside the fused
means, so the magnitude comparison is readable without recomputing it.

**Scope limits (pre-declared).** Toy fidelity, synthetic tokens, one substrate,
`spread` policy. The reset twin keeps the *initial* architecture, so P2 compares
two different final topologies rather than one topology in two processes -
`U` vs `R` is a legitimate directional test of "does the rebuild cost beyond
resets" but it is not a same-topology contrast, and it is recorded as such. The
other half of the mechanism, a fresh expert's random initialization after a grow,
is not isolated by this design at all.

**Cost.** 2 runs per seed instead of 1 discovery plus 3 retrains, so cheaper than
section 5.22: ~2.5 s/seed at these depths, well under 10 minutes of CPU across 20
shards per block.

### 5.23 Outcome: resets cost nothing - the architecture change is the whole penalty (branch (c))

Run on the registered fresh blocks 94000-94383 (depth 2) and 95000-95383 (depth
3), `study_e0f1f2629785dc17` and `study_09573ffbfb828091`. Every built-in check
held in both blocks, including the one that makes the control clean: the
reset-only twin's genome is byte-identical to the fixed arm's, it never mutated,
and it books exactly the same training FLOPs - so `F` and `U` differ by optimizer
history and nothing else.

| | depth 2 (n = 384) | depth 3 (n = 384) |
|---|---|---|
| **P1** reset cost: `F` < `U` | 157/384, mean +0.0049, p 1.00 | 105/384, mean +0.0164, p 1.00 |
| **P2** rebuild cost beyond resets: `U` < `R` | **299/384**, mean −0.1069, p 2.48e-29 | **311/384**, mean −0.1276, p 2.22e-36 |
| context: `U` < `A` | 235/384, mean −0.0402, p 6.66e-06 | 223/384, mean −0.0354, p 9.08e-04 |

Bar 215 at `alpha/4 = 0.0125`.

**Registered verdict: branch (c) - resets are not the cost, and the architecture
change is.** P1 fails at both depths *and in the exonerating direction*: the
reset-only twin was not merely unharmed but slightly **better** than its no-reset
twin in the majority of seeds (`F` wins only 157/384 and 105/384; the twin's mean
advantage is 0.005 and 0.016 nats). Three full Adam restarts cost nothing
measurable on this substrate.

**The accounting closes.** The rebuild's cost alone (`U` vs `R`: −0.1069, −0.1276)
is within a few percent of the *entire* penalty section 5.22 measured against the
fused random arm on independent blocks (−0.0996, −0.1219). Optimizer history
accounts for **zero** of the penalty; the mid-flight architecture change accounts
for all of it. Section 5.22's adopted consequence therefore stands unqualified,
and the cheaper alternative is closed off: preserving Adam state across mutations
would buy nothing.

**Prediction correct, and it was the smoke that got it right.** Section 5.23
declared P1 fail / P2 pass before the registered blocks, on the strength of three
spent-block seeds whose reset twin was near-neutral and sign-flipping. That is the
first correct prediction in this sequence after four that understated an effect,
and the difference is that this one was grounded in a measurement rather than in
a prior about which axis *should* matter.

**What remains unseparated.** The reset twin keeps the *initial* architecture, so
`U` vs `R` is a between-topology comparison: it shows the fused run is worse than
a resets-only run, not that one topology is worse in two processes. And the
component that a grow introduces - a **fresh random initialization for the new
expert** - is not isolated by this design at all; it is now the leading candidate
for what remains inside "the architecture change".

### 5.24 Pre-registration: is the penalty the discarded routing head?

**Motivation, and a defect rather than a hypothesis.** Section 5.23 left "the
architecture change" as the whole penalty without saying what inside it costs.
Inspecting the rebuild path found a concrete candidate: a layer's router weight is
`hidden x experts`, so **its shape changes whenever that layer's expert count
changes**, and `rebuild_model_from_genome` carries weights across a mutation only
when shapes match. The consequence, verified directly rather than inferred: on a
prune, the surviving experts' weights are copied by id while the mutated layer's
**entire learned routing head is re-drawn at random**; on a grow, every surviving
expert keeps its weights and every surviving router *row* is still discarded. The
unmutated layers' routers survive, which is what makes the loss visible.

So every mutation event in every fused run to date has thrown away a layer's
learned routing function. `rebuild_model_from_genome(preserve_router=True)` copies
the surviving experts' routing rows by id and leaves only a grown expert's row
fresh; the default is unchanged, so no published number moves.

**Design.** Per seed, one initial model and one data schedule, `arms="fixed,random"`:

| run | what it is |
|---|---|
| `F` | the fixed arm - must be identical under both settings |
| `R` | the fused random arm, routing head discarded on rebuild (historical) |
| `R_own` | `R`'s final genome retrained from its own initialization |
| `Rp` | the same fused arm with the routing head preserved |
| `Rp_own` | `Rp`'s final genome retrained from its own initialization |

Fresh blocks, 20-way sharding: **depth 2 = 96000-96383**, **depth 3 =
97000-97383**.

**Primaries (per depth, n = 384), Bonferroni `alpha/4 = 0.0125` -> bar 215.**
1. **preservation helps** - `Rp` < `R`.
2. **residual penalty** - `Rp_own` < `Rp`: does a penalty remain once the routing
   head survives the mutation?

**Prediction (declared before the registered blocks, and the smokes disagree with
each other - which is stated rather than resolved in the prediction's favour).**
Two spent-block checks were run, and they point in opposite directions. A 5-seed
direct check (seeds 73000-73004, penalty measured as fused minus its own frozen
retrain) showed preservation *slightly worsening* the mean penalty (`+0.048` ->
`+0.071`, per-seed spread `0.03` to `0.18`). A 4-seed check in this study's own
format showed the opposite (`-0.125` -> `-0.069`: the penalty roughly halved,
with a residual still present). Both are far too small to separate a `~0.07`
effect from a per-seed spread several times larger. The declared prediction is
**P1 passes and P2 passes: branch (b)** - preservation helps partially and a
residual penalty survives - chosen because it is the reading the second smoke's
direction *and* magnitude both support, and flagged here as weakly held.

**Pre-registered interpretation (fixed before running).**
- (a) P1 passes and P2 fails: the discarded routing head **is** the mechanism and
  is sufficient, so a one-line fix removes the penalty, section 5.22's delivery
  change is qualified, and the fused pipeline becomes viable again.
- (b) P1 passes and P2 passes: **partial** - the routing head is part of the cost
  and the rest is intrinsic to changing the architecture mid-run.
- (c) P1 fails and P2 passes: the routing head is **not** the mechanism; the cost
  is intrinsic. Section 5.22 stands unqualified, and the remaining candidates are
  the pruned expert's lost capacity and a grown expert's fresh initialization.
- (d) P1 fails and P2 fails: **inconclusive** - preserved and discarded runs
  differ without either helping or leaving a penalty. Recorded with no reading.

**Secondaries (reported, not verdict-bearing).** The historical penalty `R_own`
vs `R` on this block, for continuity with section 5.22; the two penalty
magnitudes side by side; and whether the two settings' final genomes differ at all.

**Scope limits (pre-declared).** Toy fidelity, synthetic tokens, one substrate,
`spread` policy, the `random` arm only (the adaptive arm is not re-run here so
that the two settings differ in exactly one variable). Preservation carries rows
for surviving experts; a **grown** expert's row remains freshly drawn, so this
design tests the discard of learned routing and not the cost of a fresh expert's
initialization - which stays the leading contender if P1 fails.

**Cost.** 2 fused runs plus 2 frozen retrains per seed, ~2 s/seed at these depths.

### 5.24 Outcome: a real defect carried over half the penalty (branch (b))

Run on the registered fresh blocks 96000-96383 (depth 2) and 97000-97383 (depth
3), `study_18c56d8bf539e2f3` and `study_65103c230c610378`. Every built-in check
held in both blocks, including that the flag leaves the fixed arm bit-identical
and that transporting the unmutated genome still reproduces it exactly.

| | depth 2 (n = 384) | depth 3 (n = 384) |
|---|---|---|
| **P1** preservation helps: `Rp` < `R` | **264/384**, mean −0.0542, p 7.64e-14 | **270/384**, mean −0.0694, p 5.19e-16 |
| **P2** residual penalty: `Rp_own` < `Rp` | **254/384**, mean −0.0441, p 1.22e-10 | **258/384**, mean −0.0583, p 7.33e-12 |
| context: historical penalty `R_own` < `R` | 304/384, mean −0.0983, p 3.78e-32 | 301/384, mean −0.1277, p 1.94e-30 |

| penalty | depth 2 | depth 3 |
|---|---|---|
| routing head discarded (historical) | −0.0983 | −0.1277 |
| routing head preserved | −0.0441 | −0.0583 |
| **removed by keeping the router** | **55.1%** | **54.3%** |

**Registered verdict: branch (b) - partial.** Keeping a mutated layer's learned
routing head removes **just over half** of the mid-training penalty at both
depths, and a residual penalty of `~0.044`/`~0.058` survives. So a genuine
implementation defect - not a design property of mutating a live run - was
carrying the majority of the effect that section 5.22's delivery-path change was
adopted for. The change remains warranted (a residual cost is real and
replicated), but it is now clear that the dominant term was avoidable rather
than intrinsic.

**This study is unusually cleanly paired, which is worth stating because it was
checked rather than assumed.** The random arm's mutation choice comes from its own
seeded `random.Random(seed + step)` and preservation consumes no randomness, so
the two settings make the *same* architectural choices: the final genomes are
identical in **384/384 seeds at both depths**, and the two frozen retrains are
bit-identical in **384/384 seeds** (`R_own == Rp_own` exactly, at both depths).
The residual penalty is therefore measured against the same frozen reference on
the same final topology - it cannot be a topology difference, and it is a
within-seed paired contrast by construction.

**Prediction correct, second in a row, and from a disclosed conflict.** The
prediction was branch (b), declared while stating plainly that the two spent-block
smokes *disagreed with each other* (a 5-seed direct check showed no gain, a 4-seed
check in the study's own format showed the penalty halving) and that the
per-seed spread was several times the effect. The prediction was right on
direction and magnitude; the honest reason it was right is the second smoke, and
the reason it cannot be credited to judgement is that the first said the opposite.

**Consequence for the code, and a deliberate non-change.** The default
`preserve_router=False` is retained even though the discard is evidently
unintended, because it is the behaviour every number in sections 5.16-5.23 was
measured under; flipping the default would silently invalidate the committed
record and break the reproducibility of every one of those studies. New work
should set `preserve_router_on_rebuild=True`, and this paragraph is the reason a
reader will find the default looking wrong.

**What is left.** The residual `~0.044`/`~0.058` is now the only unexplained part,
and this design cannot separate its two candidates: a **pruned** expert's lost
capacity, and a **grown** expert's fresh random initialization (preservation
carries surviving rows and leaves a new expert's row freshly drawn). Section
5.23's between-topology caveat also still applies.

### 5.25 Pre-registration: is the residual a pruned expert or a grown one?

**Motivation.** Section 5.24 removed the discarded routing head and left a
residual of `~0.044` (depth 2) and `~0.058` (depth 3). Two candidates remain - a
**pruned** expert's lost capacity, and a **grown** expert's fresh random
initialization - and they cannot be separated by subgroup split, because the mix
of operations was chosen by the same seeded coin flip that the subgroups would be
defined by. So the operation is forced instead.

**Design.** `ExperimentConfig.forced_operation` makes every event by a
non-adaptive arm the named operation; a forced operation the genome cannot perform
is left **unperformed** rather than replaced by the opposite one, so a profile
that quietly stops mutating fails its budget check instead of measuring the wrong
thing. Each profile is its own fresh block, with
`preserve_router_on_rebuild=True` (section 5.24's treatment) and
`arms="fixed,random"`:

| block | seeds | operation | experts/layer | ends at |
|---|---|---|---|---|
| prune | 98000-98383 | all 3 events prune | 5 | 2 on the worst lane, still legal |
| grow | 99000-99383 | all 3 events grow | 4 | 7 on the worst lane, under the 8 ceiling |

The two blocks have different expert counts by necessity, but each primary is a
*within-block* paired contrast against that block's own frozen retrain, so the
substrate difference is not a confound for either one.

**Primaries (n = 384 per block), Bonferroni `alpha/2 = 0.025` -> bar 212.**
1. **prune residual** - `Rp_own` < `Rp` on the prune block: does losing an expert's
   learned function cost anything once the routing head survives?
2. **grow residual** - `Rp_own` < `Rp` on the grow block: does adding a freshly
   initialized expert cost anything under the same treatment?

**Prediction (declared before the registered blocks; the spent-block smoke is
disclosed and it covered only one profile).** A 4-seed machinery smoke ran the
prune profile on spent seeds 73000-73003 and showed residual `-0.0583`, close to
section 5.24's mixed residual - so **P1 is predicted to pass**. **P2 is predicted
to fail**, giving **branch (b): the residual is the pruned expert's lost
capacity.** The reasoning is that a prune destroys a trained function
irreversibly, while a grow *adds* capacity: the surviving experts keep their
weights, the routing head now keeps its rows for them (section 5.24), and the
new expert's only cost is a transient while the model learns to use it. Section
5.17's subgroup split pointed the same way - the matched-init effect was
concentrated in pure-prune seeds - and that is a *different* measurement on
different blocks, so it is offered as agreement rather than as the basis.

**Pre-registered interpretation (fixed before running).**
- (a) both pass: the residual is **both** operations - losing capacity and adding
  an untrained one both cost, so it is intrinsic to changing the architecture.
- (b) P1 passes, P2 fails: the residual is the **pruned expert's lost capacity**.
  The cheapest attack is then a mutation policy that prunes less, or trains the
  survivor to compensate, rather than anything about initialization.
- (c) P1 fails, P2 passes: the residual is the **grown expert's fresh
  initialization**, and the fix is an initialization for new experts (section 51's
  teacher treatment becomes directly relevant).
- (d) neither passes: no residual in either *pure* profile. Since section 5.24
  measured one on mixed runs, the residual would then require the **mix** itself
  rather than either operation - a different mechanism from both candidates, and
  the attribution is recorded as failed.

**Secondaries (reported, not verdict-bearing).** The two residual magnitudes side
by side with section 5.24's mixed figure, so the additive reading ("the mixed
residual should be about half the pure one") is checkable; the fixed arm's own
loss per block; and the final per-lane expert counts, which the built-in check
requires to have moved by exactly the budget in one direction.

**Scope limits (pre-declared).** Toy fidelity, synthetic tokens, one substrate,
depth 2 only, the `random` arm only, and the operation is forced rather than
drawn - so what is attributed is the *cost of a forced operation*, which is the
right quantity for attribution but is not itself a deployable policy. A grow's
new expert is freshly initialized by the existing path and no alternative
initialization is tested here.

**Cost.** One fused run plus one frozen retrain per seed, ~1.4 s/seed.

### 5.25 Outcome: both operations cost, and the grow costs more (branch (a))

Run on the registered fresh blocks 98000-98383 (prune) and 99000-99383 (grow),
`study_68591d55f3a56db7` and `study_c7ec09294da043fc`. Every built-in check held
in both blocks, including the one the forced operation exists for: **every event
was the forced operation**, so the total expert count moved by exactly the budget
in one direction, no lane dipped below two experts, and nothing was ever
substituted by the fallback grow.

| | prune block (n = 384) | grow block (n = 384) |
|---|---|---|
| **P1/P2** residual: `Rp_own` < `Rp` | **241/384**, mean −0.0297, p 3.25e-07 | **241/384**, mean −0.0476, p 3.25e-07 |
| per forced event | 0.0099 | 0.0159 |

Both primaries pass at the Bonferroni bar of 212.

**Registered verdict: branch (a) - the residual is both operations.** Losing a
trained expert's function costs, *and* adding a freshly initialized one costs, so
the residual is intrinsic to changing the architecture in either direction rather
than to one operation. The two blocks landed on the same win count (241/384) and
different magnitudes, which is a coincidence of the two effects' noise, not a
relationship.

**The prediction was wrong, and the way it was wrong is the useful part.** I
declared P1 pass / P2 fail - branch (b), the pruned expert's lost capacity -
reasoning that a prune destroys a trained function irreversibly while a grow only
*adds* capacity a preserved routing head can learn to use. In fact the **grow
costs about 1.6x the prune** (0.0159 vs 0.0099 per event). Making a new expert
and teaching the model to route to it is the more expensive operation, not the
cheaper one. This is the fifth prediction in the sequence to understate an effect,
and the first where the miss was in the *comparison* rather than the existence of
an effect.

**An additive consistency check, which is a real test and it roughly holds.**
Section 5.24's mixed-run residual (depth 2, routing head preserved) was `0.0441`.
If per-event costs add, a run with `1.5` prunes and `1.5` grows should cost
`1.5 x 0.0099 + 1.5 x 0.0159 = 0.0386` - within 13% of the measured figure, on
different blocks, with a different substrate (5 and 4 experts/layer here versus
4 in section 5.24) and an operation mix that was assumed rather than recorded.
It is offered as agreement, not as a derivation.

**Consequences.** With section 5.24, the mid-training penalty now decomposes as:
just over half from a discarded routing head (a defect, fixed by
`preserve_router=True`), and the remainder split between pruning and growing at
about 1:1.6. The remaining lever is therefore not "mutate less" but the
**initialization of a new expert**, which is precisely the axis plan section 51's
teacher treatment addresses - and which this project has never tested, because it
needs the Qwen-derived artifacts the plan describes and this repository does not
bundle. That is now the best-motivated untested experiment in the programme.

**Caveats, as declared.** Depth 2 only; the `random` arm only; operations forced
rather than drawn, so what is attributed is the cost of a forced operation, which
is the right quantity for attribution but is not itself a policy. The two blocks
necessarily start at different expert counts (5 and 4 per layer), so their
magnitudes are comparable in shape but not substrate-identical, and the
per-operation split above inherits that.

### 5.26 Pre-registration: is the residual penalty the freshly drawn expert?

**Motivation.** Section 5.25 forced every event to a grow and found a grow costs
about `1.6x` a prune (`0.0159` vs `0.0099` per event), so the residual mid-training
penalty that survived section 5.24 is concentrated in the operation that *adds* an
expert. A grow introduces exactly one thing a prune cannot: a **freshly drawn
expert** - a random body and a random router row - which the model must spend
gradient learning to route around, and which the router must learn to send tokens
to. Section 5.25 closed by naming an initialization for new experts as the
best-motivated untested experiment available, and noted it was blocked on the
Qwen-derived teacher artifacts this repository does not bundle. This study tests
the mechanism with something the model already contains: its own busiest
surviving expert.

**Design.** `ExperimentConfig.new_expert_initialization` decides what a grown
expert starts as. `"random"` is the rebuild's own draw - the historical
behaviour, and the only one every published number was measured under.
`"clone"` copies the busiest surviving expert of the same layer, chosen by the
model's own recorded routing load, **together with that expert's router row** - a
cloned body with a fresh row would start with no routing affinity at all, so
cloning the row is what makes it a twin rather than a coincidence of weights.
(Cloning therefore requires `preserve_router_on_rebuild=True`, and the config
refuses the combination that would silently clone nothing.)

Each block is one fresh seed set, `forced_operation="grow"`,
`preserve_router_on_rebuild=True`, `arms="fixed,random"`, and each seed runs
**both** settings so the contrast is paired:

| label | run |
|---|---|
| `F` | the fixed arm (unmutated) - identical under both settings |
| `R` | fused grow run, new expert freshly drawn (historical) |
| `R_own` | `R`'s final genome retrained from its own initialization |
| `C` | fused grow run, new expert cloned from the busiest survivor |
| `C_own` | `C`'s final genome retrained from its own initialization |

**Why this is a paired contrast on one genome.** The forced operation makes the
non-adaptive arm choose its layer from `random.Random(seed)`, which consumes the
seed's own RNG and not the weights, and cloning consumes no randomness - so both
settings make the **same architectural choices** and end on the same final genome.
That is asserted as a built-in check (`same_final_genome`, and `C_own == R_own`
exactly), not assumed: were it false, the penalties would be measured against two
different substrates and the framing would be meaningless.

**Primaries per depth (n = 384), Bonferroni `alpha/4 = 0.0125` -> bar 215.**
1. **cloning helps** - `C` < `R`: does a grown expert that starts as a working twin
   beat one that starts as noise, on the same seed and the same final genome?
2. **residual penalty** - `C_own` < `C`: does a penalty remain once the new expert
   is cloned, or does cloning remove the mid-training penalty entirely?

**Prediction (declared before the registered blocks; a spent-block smoke is
disclosed and it covered one depth).** A 4-seed machinery smoke (depths 2,
spent seeds 73000-73003) showed clone gain `-0.0402`, fresh penalty `-0.0615`,
cloned penalty `-0.0213`, with `C_own == R_own` bit-identical - so **P1 is
predicted to pass and P2 to pass**, giving **branch (b): cloning helps but a
residual penalty survives.** The reasoning is that a twin removes the transient
cold-start cost (the new expert starts already doing useful work on tokens the
router already sends its source), but the run still pays for a mid-flight
architecture change: the expert count rises, the softmax denominator over routing
weights moves, and the model renormalizes around a new lane in a fixed step
budget. Cloning removes one term, not the physics of the change.

**Pre-registered interpretation (fixed before running, in section 5.24's letter
order: (a) means *helps, and no penalty remains*).**
- (a) P1 passes, P2 fails: cloning helps and no penalty remains - a grown expert
  should start as a twin, and section 5.22's two-stage delivery is qualified
  again (any remaining cost would be the freeze/retrain overhead, not an
  intrinsic process cost).
- (b) both pass: cloning helps but a residual penalty survives, so the mechanism
  is more than the fresh draw.
- (c) P2 passes, P1 fails: cloning does not help and the penalty survives, so the
  fresh draw is not the mechanism and section 5.25's residual is intrinsic.
- (d) neither passes: inconclusive, recorded without either reading.

*(The branch list above was written in the opposite letter order to section
5.24's convention and to this study's script, which maps P1-pass/P2-fail to
(a). The letters are reconciled to the script - see the 5.26.1 amendment - and
the substance of each branch is unchanged by the reconciliation.)*

**Secondaries (reported, not verdict-bearing).** `R_own - R` and `C_own - C` side
by side (the two penalties); the fraction of the random-init penalty that cloning
removes; and `C - R` per depth for continuity with the smoke.

**Scope limits (pre-declared).** Toy fidelity, synthetic tokens, one substrate,
`random` arm only, the operation forced rather than drawn, and cloning from the
layer's busiest survivor rather than from a trained teacher - so what is tested is
*self*-cloning, not plan section 51's teacher treatment. A clone that works here
would bound how much the teacher treatment can buy, not replace it.

**Cost.** Two fused runs plus two frozen retrains per seed, ~1.4 s/seed.

### 5.26 Outcome: cloning does not merely remove the penalty, it inverts it (branch (a))

Run on the registered fresh blocks 103000-103383 (depth 2) and 104000-104383
(depth 3), `study_99a12c16f910e9c0` and `study_ee833b6b284195f9`. Every built-in
check held in both blocks, including the two the design's validity rests on: the
fixed transport reproduced the fixed arm under **both** settings, and the pairing
claim held exactly - **the two settings chose the same final genome in 384/384
seeds at both depths**, so `C_own == R_own` bit-identically in 384/384. The
contrast is therefore initialization on one genome, not two substrates.

| | depth 2 (n = 384) | depth 3 (n = 384) |
|---|---|---|
| **P1** cloning helps: `C` < `R` | **315/384**, mean −0.1005, p 5.9e-39 | **307/384**, mean −0.1036, p 6.4e-34 |
| **P2** residual penalty: `C_own` < `C` | 126/384, mean **+0.0600**, p 1.00 - FAIL | 131/384, mean **+0.0745**, p 1.00 - FAIL |

| penalty | depth 2 | depth 3 |
|---|---|---|
| fresh expert (`R_own − R`) | −0.0406 | −0.0292 |
| cloned expert (`C_own − C`) | **+0.0600** | **+0.0745** |
| gain from cloning (`C − R`) | −0.1005 | −0.1036 |

**Registered verdict: branch (a) - the freshly drawn expert IS the residual
mechanism.** The prediction (P1 pass, P2 pass, branch (b)) was half wrong: P1
passed by a wide margin, but P2 did not merely fail - it **reversed**. With a
cloned new expert the fused run is *better* than a clean frozen retrain of its own
final genome, by `+0.060` and `+0.074`, where with a freshly drawn one it was
worse by `−0.041` and `−0.029`. The mid-training penalty is not merely reduced by
cloning; it becomes a mid-training **advantage**. This is the sixth prediction in
the sequence to understate an effect, and the second where the miss was in the
comparison rather than its existence.

**The full ordering, on the means, is the same at both depths**
(`C < F < C_own < R`):

| | depth 2 mean | depth 3 mean |
|---|---|---|
| `C` clone-grow fused | **14.2414** | **13.3819** |
| `F` unmutated fixed arm | 14.2887 | 13.4362 |
| `C_own` frozen retrain of the final genome | 14.3014 | 13.4564 |
| `R` random-grow fused (historical) | 14.3419 | 13.4855 |

The clone-grow run beats **all three** references, including the unmutated arm
(`C` < `F`: **296/384**, mean −0.0473, p 1.0e-27; **291/384**, mean −0.0543,
p 3.7e-25). Section 5.25's residual is not merely attributable to the fresh draw:
the fresh draw was the *whole* grow penalty and more.

**Why this is the coherent reading rather than a surprise.** A frozen retrain of
the final genome starts from scratch on seven experts in the same 48 steps; a
clone-grow run carries four experts it has already trained and adds three that
start as working twins of the busiest survivor. What the two-stage delivery of
section 5.22 was doing was *throwing away trained function to avoid the cost of
changing architecture mid-flight* - and section 5.24 and this result together say
that cost was avoidable, not intrinsic. Once the routing head survives and a new
expert starts as a twin, continuing the run dominates restarting it.

**Consequences.** Section 5.22 adopted `discover -> freeze genome -> retrain from
the original initialization`. On this substrate that pipeline is now a
**regression**, not a safeguard: with `preserve_router_on_rebuild=True` and
`new_expert_initialization="clone"`, the fused run beats the retrain in 258/384
(depth 2, p 7.3e-12) and 253/384 (depth 3, p 2.4e-10) seeds. The recommendation
for new work is therefore the reverse: **keep the fused run**, with the router
preserved and new experts cloned - and plan section 102's genome extraction is not
needed for performance. The best-motivated untested experiment named by section
5.25 is thus answered by something the model already contains, which also means
plan section 51's teacher treatment now has a *bound*: it can only pay off to the
extent an external teacher beats the model's own busiest expert as a source for a
new expert.

**The mechanism inside "clone" is not settled, and the two candidates are worth
naming.** Cloning does two things at once: it gives the new expert a *trained
body* instead of noise, and it copies the survivor's *router row*, so the router
already sends tokens to the twin and the twin pair receives the same tokens. The
second could be doing the work - duplicating a function the data wants, in a way
that needs no new routing decision - and this design does not separate them. A
clone with a fresh router row, and a "copy a *random* survivor's body" arm, would
between them separate body from routing affinity. Declared here as unresolved
rather than attributed.

**Caveats, as declared.** Toy fidelity, synthetic tokens, one substrate, depth 2
and 3, the `random` arm only, the operation forced rather than drawn (so this is
the cost of a forced grow, not a deployable policy), and cloning from the layer's
busiest survivor rather than a trained teacher. The frozen retrains are of *each*
fused run's final genome; they happen to be the same genome here, which is checked
rather than assumed.

### 5.26.1 Amendment (made after the block ran): reconciled branch letters

The pre-registered branch list was written in the opposite letter order to
section 5.24's convention and to this study's own script, which maps
P1-pass/P2-fail to **(a)**. The letters are reconciled to the script. This
changes no statistic, no threshold and no reading: the observed outcome is
P1 pass / P2 fail, whose substance ("cloning helps and no penalty remains") is
identical under either labelling, and P2 failed in the *exonerating* direction -
`C_own > C` - so the penalised reading was never available. The same class of
labelling trap as section 5.19's auto-label, recorded rather than quietly fixed.

**A second lapse, of the same kind and worth separating.** This pre-registration
text was written and its smoke disclosed *before* the registered blocks were run,
as it claims - but it was **committed in the same commit as the result**, not
beforehand. Every earlier study in this series (5.20, 5.22, 5.24, 5.25) has the
pre-registration in a prior commit, so its order is demonstrable from history;
here it is only asserted. The text itself is not reconstructed - the decline of
that claim would be visible in the disclosed smoke, whose numbers (clone gain
`-0.0402`, penalties `-0.0615`/`-0.0213`) and predicted branch (b) are unchanged
from before the run, and the run falsified the prediction rather than confirming
it. Recorded so that a reader who checks the history finds the lapse stated
rather than having to notice it.

### 5.27 Pre-registration: is a cloned expert's benefit its body or its routing row?

**Motivation.** Section 5.26 showed that cloning the layer's busiest survivor
into a grown expert does not merely remove the mid-training penalty but inverts
it: the clone-grow fused run beat the unmutated arm, the frozen retrain of its own
final genome and the historical random-grow run, at both depths. It also declared
the mechanism **unresolved**, because cloning does two things at once - it gives
the new expert a trained **body**, and it copies the survivor's **router row**, so
the router already sends the twin pair the same tokens. Either could be doing the
work, and the two imply different minimal changes.

**Design.** `new_expert_initialization="clone_body"` is identical to `"clone"` in
every respect except that the grown expert keeps its **freshly drawn router row**.
Per seed, one initial model, one data schedule, every event forced to a grow,
`preserve_router_on_rebuild=True`, `arms="fixed,random"`:

| label | run |
|---|---|
| `F` | the fixed arm (unmutated) - identical under all three settings |
| `R` | fused grow run, new expert freshly drawn (historical) |
| `B` | fused grow run, body cloned from the busiest survivor, **fresh router row** |
| `C` | fused grow run, body cloned **and** the survivor's router row copied |
| `*_own` | the frozen retrain of each fused run's own final genome |

All three settings choose the same layer from the seed's own RNG and none of them
consumes randomness, so they end on the **same final genome** and all three frozen
references are bit-identical - both asserted as built-in checks, as in section
5.26. `B - R` is therefore the trained body's contribution and `C - B` is the
routing affinity's, on one genome.

**Primaries per depth (n = 384), Bonferroni `alpha/4 = 0.0125` -> bar 215.**
1. **the body helps** - `B` < `R`: does a trained body alone, conferring no
   routing affinity, beat a fresh draw?
2. **routing affinity adds** - `C` < `B`: does copying the router row on top of
   that body add anything?

**Prediction (declared before the registered blocks; a spent-block smoke is
disclosed and it covered one depth).** A 4-seed machinery smoke (depth 2, spent
seeds 73000-73003) showed body gain `-0.0231`, row gain `-0.0172` and total clone
gain `-0.0403`, with all three frozen references identical - nearly additive, and
body-weighted. So **both primaries are predicted to pass**, giving **branch (a):
both contribute, the body more than the row.** The reasoning is that a trained
body removes the cold-start transient (the twin is already doing useful work from
step 0), while a copied row removes the transient in the *routing* too (the router
need not learn to send the new lane tokens before the body can help). They attack
different halves of the same cost, which is why near-additivity is predicted and a
clean one-or-the-other is not. I record the obvious caution: the smoke's per-seed
spread is larger than the effect it measures, so it is a machinery check with a
direction, not evidence.

**Pre-registered interpretation (fixed before running, in section 5.24's letter
order: (a) means both primaries pass).**
- (a) both pass: cloning is two effects. The minimal change is the full clone;
  neither body cloning nor row copying alone reproduces section 5.26.
- (b) P1 passes, P2 fails: the **trained body is the whole effect** and the copied
  row adds nothing - the minimal change is body cloning alone, and section 5.26's
  "twin" framing (which leans on the row) is wrong about why it works.
- (c) P2 passes, P1 fails: **routing affinity is the whole effect**; a cloned body
  with a fresh row does not help, so the row must be copied.
- (d) neither passes: inconclusive, recorded without a reading.

**Secondaries (reported, not verdict-bearing).** `B - R`, `C - B` and `C - R` side
by side per depth, so the additivity of the two contributions is checkable; the
fixed arm's own loss; and each arm's penalty against its (identical) frozen
reference.

**Scope limits (pre-declared).** Toy fidelity, synthetic tokens, one substrate,
depth 2 and 3, `random` arm only, the operation forced rather than drawn, and
cloning from the layer's busiest survivor rather than a trained teacher. This
separates *body* from *routing affinity* only; it does not test whether a **random**
survivor would clone as well as the busiest one, which is a third arm this design
does not carry.

**Cost.** Three fused runs plus three frozen retrains per seed, ~3.7 s/seed.

### 5.27 Outcome: cloning is two effects, and the body is ~3/4 of it (branch (a))

Run on the registered fresh blocks 105000-105383 (depth 2) and 106000-106383
(depth 3), `study_3bcc9e86b575b7a6` and `study_872d33bdd99496a8`. Every built-in
check held in both blocks, and the three-way pairing is exact: **all three
settings chose the same final genome in 384/384 seeds, all three frozen
references are bit-identical in 384/384, and the fixed arm reproduced exactly
under all three settings in 384/384** - so `B - R` is the trained body's
contribution and `C - B` the routing affinity's, on one genome.

| | depth 2 (n = 384) | depth 3 (n = 384) |
|---|---|---|
| **P1** the body helps: `B` < `R` | **295/384**, mean −0.0744, p 3.4e-27 | **300/384**, mean −0.0844, p 7.0e-30 |
| **P2** routing affinity adds: `C` < `B` | **242/384**, mean −0.0264, p 1.9e-07 | **238/384**, mean −0.0220, p 1.5e-06 |
| total clone gain: `C` < `R` | **320/384**, mean −0.1008, p 2.4e-42 | **312/384**, mean −0.1064, p 5.2e-37 |

| contribution | depth 2 | depth 3 | share of the total |
|---|---|---|---|
| trained body (`B - R`) | −0.0744 | −0.0844 | **73.8%** / **79.4%** |
| routing affinity (`C - B`) | −0.0264 | −0.0220 | 26.2% / 20.6% |

**Registered verdict: branch (a) - both contribute, the body more than the row.**
The prediction was right, and it was right for a reason worth separating from the
six that preceded it: it came from the disclosed 4-seed smoke (body `-0.0231`, row
`-0.0172`) rather than from a prior about which axis *should* matter. The caveat I
attached to that smoke - its per-seed spread exceeds the effect it measures - did
not bite, but the registered block was what settled it.

**The ordering is the same at both depths, and it is now four-deep**
(`C < B < F < own < R`):

| | depth 2 mean | depth 3 mean |
|---|---|---|
| `C` body + router row cloned | **14.1928** | **13.3470** |
| `B` body cloned, fresh row | 14.2192 | 13.3690 |
| `F` unmutated fixed arm | 14.2398 | 13.3947 |
| `own` frozen retrain of the final genome | 14.2674 | 13.4153 |
| `R` fresh draw (historical) | 14.2936 | 13.4534 |

So the body clone alone already beats the unmutated arm and the frozen retrain,
and copying the router row on top beats the body clone alone. Two separable
mechanisms, both real, each beating the historical behaviour by itself.

**Consequences.** Section 5.26's "twin" framing was right but imprecise: the
trained body is the larger half, and it is the part that does *not* depend on
routing affinity, so it would be the minimal change if only one were affordable.
The row is a real second term (20-26%) and is cheap - it is a row copy - so the
recommendation stands as the full clone. This also sharpens section 5.25's open
question about plan section 51's teacher treatment: a teacher would have to beat
the model's own busiest expert **as a body source** for the larger half, and
separately solve the routing-row problem the copy solves for free.

**What this does not settle, and one thing it newly raises.**
- The third arm is missing: this separates *body* from *routing affinity*, but not
  "clone the busiest survivor" from "clone *any* survivor". A `clone_random` arm
  (a random survivor's body and row) would test whether busyness matters at all,
  or whether any trained expert is as good a source. Declared as not carried here.
- Every clone in this study clones the *same* expert body that survives, so the
  layer ends with an identical twin pair whose rows tie. Whether that degeneracy
  costs anything at larger active-expert counts is untested - the toy substrate
  has `active_experts=2` and, unusually, the degeneracy helped.

**Caveats, as declared.** Toy fidelity, synthetic tokens, one substrate, depths 2
and 3, `random` arm only, the operation forced rather than drawn (so this is the
cost of a forced grow, not a deployable policy), and cloning from the layer's
busiest survivor rather than a trained teacher.

### 5.28 Pre-registration: does the cloning result survive the deployed regime?

**Motivation.** Sections 5.24, 5.26 and 5.27 all measured their effects on
**forced** operations - every event a grow, or every event a prune - because that
is what makes a contribution attributable. But the delivery decision those
sections changed was adopted from the **mixed, unforced** regime: section 5.22
concluded `discover -> freeze the genome -> retrain from the original
initialization` from blocks where the operation and its layer were drawn from the
seeded coin flip. Sections 5.26 and 5.27 then found that the penalty that
justified that pipeline is invertible, on forced blocks. Whether an invertible
penalty on forced blocks says anything about the drawn regime the pipeline was
adopted from is exactly the gap this study closes. (That the process penalty is
**arm**-generic was already established by section 5.22, which is why this tests
the regime on the `random` arm rather than re-testing the arms.)

**Design.** `forced_operation` is **left None**, so the operation mix is whatever
the coin flip produces, and both settings run on the same seed, initial model and
data schedule with `preserve_router_on_rebuild=True` and `arms="fixed,random"`:

| label | run |
|---|---|
| `F` | the fixed arm (unmutated) - identical under both settings |
| `R` | fused mixed run, new experts freshly drawn (historical) |
| `R_own` | `R`'s final genome retrained from its own initialization |
| `C` | fused mixed run, new experts cloned from their layer's busiest survivor |
| `C_own` | `C`'s final genome retrained from its own initialization |

**The pairing claim is weak here, and the design does not assume it.** The mutation
*choices* are still drawn from the seed's own RNG, so they should be identical
under both settings - but a drawn prune the genome cannot perform falls back to a
**load-based** grow, and that path reads the weights, which cloning changes. The
two settings can therefore diverge after the first fallback. So unlike sections
5.26 and 5.27, identical final genomes are **not asserted**: the divergence rate is
recorded and reported, and each arm's penalty is measured against its **own**
frozen retrain, which is well defined either way.

**Primaries per depth (n = 384), Bonferroni `alpha/4 = 0.0125` -> bar 215.**
1. **clone helps in the deployed regime** - `C` < `R`.
2. **no residual penalty under cloning** - `C_own` < `C`, i.e. the predicted
   **failure** of P2, as in section 5.26.

**Prediction (declared before the registered blocks; a spent-block smoke is
disclosed).** A 4-seed machinery smoke (depth 2, spent seeds 73000-73003) drew a
genuinely mixed block (6 grows, 6 prunes) with **4/4 identical final genomes**,
and showed clone gain `-0.0661`, fresh penalty `-0.0691`, cloned penalty
`-0.0030` - so **P1 is predicted to pass and P2 to fail**, giving **branch (a):
the inversion survives the deployed regime and no penalty remains.** The reasoning
is that nothing in section 5.26's mechanism mentions *forcing*: a cloned expert
starts as a working twin whether the grow was chosen or imposed, and the operation
that costs is still a grow either way. The declared caution is that the smoke's
per-seed spread exceeds the effect, as in section 5.27.

**Pre-registered interpretation (fixed before running, in section 5.24's letter
order: (a) means helps, and no penalty remains).**
- (a) P1 passes, P2 fails: the inversion survives the drawn regime. Clone-and-
  preserve should then be the pipeline, and section 5.22's freeze-and-retrain is a
  **regression outside the forced blocks too**, not a safeguard specific to them.
- (b) both pass: cloning helps in the deployed regime but a residual penalty
  survives there, so freeze-and-retrain keeps a justification the forced blocks
  did not show - and the regime, not the operation, is what differs.
- (c) P2 passes, P1 fails: the section 5.26/5.27 result is **confined to forced
  operations**, and section 5.22's pipeline stands.
- (d) neither passes: inconclusive, recorded without a reading.

**Secondaries (reported, not verdict-bearing).** The operation mix and the number
of seeds where the two settings' architectures or operation mixes diverge (the
fallback channel); `R_own - R` and `C_own - C` side by side, plus
`C_own - R_own`; and the fixed arm's own loss.

**Scope limits (pre-declared).** Toy fidelity, synthetic tokens, one substrate,
depths 2 and 3, the `random` arm only (arm-genericity of the process penalty is
cited from section 5.22, not re-measured), and a 3-event budget. This tests the
**drawn-operation regime**, not the probe-based adaptive selection path, which is
the next axis and is not carried here.

**Cost.** Two fused runs plus two frozen retrains per seed, ~2.1 s/seed.

### 5.28 Outcome: the inversion survives the deployed regime (branch (a))

Run on the registered fresh blocks 107000-107383 (depth 2) and 108000-108383
(depth 3), `study_5ece35fb3fbc8724` and `study_f8bc02ca0f048a79`. Every built-in
check held, and the regime claim was met: the blocks are genuinely drawn mixes
(**622 grows / 530 prunes** at depth 2, **548 / 604** at depth 3), not forced
blocks in disguise.

| | depth 2 (n = 384) | depth 3 (n = 384) |
|---|---|---|
| **P1** clone helps: `C` < `R` | **276/384**, mean −0.0720, p 2.3e-18 | **252/384**, mean −0.0651, p 4.6e-10 |
| **P2** residual penalty: `C_own` < `C` | 170/384, mean **+0.0243** — FAIL | 178/384, mean **+0.0113** — FAIL |
| historical penalty: `R_own` < `R` | 255/384, mean −0.0477, p 6.1e-11 | 256/384, mean −0.0538, p 3.1e-11 |

**Registered verdict: branch (a).** Cloning helps in the drawn regime and no
penalty remains - P2 fails in the exonerating direction again, so the fused run
with a cloned new expert beats a clean retrain of its own final genome. The
prediction was right, third in a row.

**The pre-declared divergence channel did not fire.** Identical final genome in
**384/384** and identical operation mix in **384/384**, at both depths, with
`C_own == R_own` bit-identically - so the load-based fallback never took the two
settings down different paths, even though a layer sits at the 2-expert floor in
69 (depth 2) and 72 (depth 3) seeds. The pairing held without being assumed; the
design was built to survive it failing and it did not fail.

**The result is sharper than "cloning helps", in a way the forced blocks could not
show.** In this regime the historical path is not merely worse than cloning - it
is worse than **doing nothing**:

| | depth 2 | depth 3 |
|---|---|---|
| `F` < `R` (mutating the historical way costs vs not mutating) | **251/384**, −0.0444, p 8.9e-10 | **236/384**, −0.0403, p 4.1e-06 |
| `C` < `F` (cloning makes mutating *pay*) | **238/384**, −0.0275, p 1.5e-06 | **237/384**, −0.0248, p 2.5e-06 |
| `R_own` < `F` (the retrain vs not mutating) | 186/384, −0.0033, p 0.75 | 207/384, −0.0135, p 0.069 |

The sign of the whole pipeline flips. Under the historical initialisation, running
the mutation process leaves you **worse off than never mutating** - significantly,
at both depths. Under cloning, the same process leaves you significantly better
off. The frozen retrain, meanwhile, is statistically indistinguishable from the
unmutated arm (p 0.75 and 0.069), which is what a mixed draw should do: its net
topology change is a wash. Section 5.22 recommended freeze-and-retrain because the
fused run was worse than the retrain; here the retrain buys nothing over never
mutating, and cloning beats both.

| | depth 2 mean | depth 3 mean |
|---|---|---|
| `C` clone-grow fused | **14.2630** | **13.3864** |
| `own` frozen retrain of the final genome | 14.2873 | 13.3977 |
| `F` unmutated fixed arm | 14.2905 | 13.4112 |
| `R` fresh draw (historical) | 14.3350 | 13.4515 |

Note the ordering differs from section 5.26's: there `F < own` held significantly,
here `own` and `F` tie. That is the regime difference showing up honestly - a
run of forced grows ends at a topology worse than the initial one, a mixed draw
ends at one that is a wash, and either way cloning is the term that decides the
sign.

**Consequences.** Section 5.22's `discover -> freeze genome -> retrain` is now
contradicted in the regime it was adopted from, not only in the forced blocks of
sections 5.26-5.27. The pipeline recommendation is **clone-and-preserve, keeping
the fused run**: it beats the historical path in 276/384 (depth 2) and 252/384
(depth 3) seeds, beats the retrain in 214/384 and 206/384, and beats not mutating
at all in 238/384 and 237/384. That is a stronger claim than section 5.26 could
make, because "mutating pays" is a property a delivery pipeline can be built on,
and "mutating costs less than it used to" is not.

**Caveats, as declared.** Toy fidelity, synthetic tokens, one substrate, depths 2
and 3, the `random` arm only - arm-genericity of the process penalty is **cited**
from section 5.22, not re-measured here - a 3-event budget, and cloning from the
layer's busiest survivor rather than a trained teacher. The probe-based **adaptive**
selection path, which is what the project actually uses to discover genomes, is not
tested by this study and is the next axis; nothing here speaks to it.

### 5.29 Pre-registration: does clone-and-preserve hold on the discovery path?

**Motivation.** Section 5.28 confirmed the clone inversion in the drawn, unforced
regime - but on the **`random`** arm, whose mutation choices come from the seed's
own RNG. The arm the project actually uses to discover genomes is the **probe-guided
`adaptive`** arm: it enumerates feasible mutation candidates, scores each by
rebuilding the model and measuring held-out probe loss, and applies the best. That
is a different process, and the pre-registered consequence of section 5.28 is only
usable if it survives it.

**Design.** One seed set per depth, `arms="fixed,adaptive"`, `forced_operation`
left None so the mix is drawn, `preserve_router_on_rebuild=True`:

| label | run |
|---|---|
| `F` | the fixed arm (unmutated) - identical under both settings |
| `R` | adaptive discovery, new experts freshly drawn (historical) |
| `R_own` | `R`'s final genome retrained from its own initialization |
| `C` | adaptive discovery, new experts cloned from their layer's busiest survivor |
| `C_own` | `C`'s final genome retrained from its own initialization |

**The pairing is NOT assumed, and on this path it does not always hold - measured
before the design was fixed.** Reading the probe path shows that `_mutate_by_probe`
scores candidates with `rebuild_model_from_genome` called **without** the arm's
`preserve_router` / `new_expert_initialization` flags, while the *selected* candidate
is applied with them. So scoring is init-independent at a given set of weights,
but once the two runs' weights differ a later probe can rank candidates
differently. A 16-seed spent-seed measurement (seeds 73000-73015) found **divergent
final genomes in 2/16 seeds (12.5%)**, with clone gain `-0.0682` and 9/16 wins.
Roughly a tenth of the registered block is therefore expected to diverge. Unlike
sections 5.26 and 5.27, identical genomes are **recorded and reported, never
asserted**, each penalty uses its own frozen retrain, and the primary is also
reported on the subset where the two runs agreed so a reader can see whether the
divergent seeds carry it. Primary 1 is consequently a comparison of two
**processes**, not a paired contrast on one genome - the right shape for a
deployable question and the wrong shape for a mechanism question.

**Primaries per depth (n = 384), Bonferroni `alpha/4 = 0.0125` -> bar 215.**
1. **clone helps on the discovery path** - `C` < `R`: does switching the new-expert
   initialization improve the discovery pipeline end to end?
2. **no residual penalty under cloning** - `C_own` < `C`, i.e. the predicted
   **failure** of P2, as in sections 5.26 and 5.28.

**Prediction (declared before the registered blocks; the measurements above are
disclosed).** A 4-seed machinery smoke (depth 2, spent seeds 73000-73003) showed
clone gain `-0.0721`, cloned penalty `+0.0255` and `C` < `F` by `-0.0357`, with
**1/4** divergent genomes. So **P1 is predicted to pass and P2 to fail**, giving
**branch (a): the recommendation survives the discovery path.** The reasoning is
that the *application* of a mutation is what section 5.26's mechanism acts on - a
cloned expert starts as a working twin whether a probe or a coin flip chose the
grow - while the probe's *scoring* is unchanged by the setting by construction, so
cloning adds its benefit without disturbing which candidates get chosen. The
disclosed caution is that the smoke's per-seed spread again exceeds the effect, and
that the 12.5% divergence rate means a tenth of the block compares two different
architectures.

**Pre-registered interpretation (fixed before running, in section 5.24's letter
order: (a) means helps, and no penalty remains).**
- (a) P1 passes, P2 fails: the pipeline recommendation extends to the arm the
  project actually uses, so clone-and-preserve is the pipeline for real discovery
  and not only for the random arm.
- (b) both pass: cloning helps discovery but a residual penalty survives under
  adaptive selection, so freeze-and-retrain keeps a justification there.
- (c) P2 passes, P1 fails: the section 5.26-5.28 result is **confined to the random
  arm**, and the recommendation does **not** extend to discovery.
- (d) neither passes: inconclusive, recorded without a reading.

**Secondaries (reported, not verdict-bearing).** The divergence rate in final
genomes and in operation mix; the primary recomputed on the agreeing subset; the
operation mix; `R_own - R`, `C_own - C`, and both paths against `F` (the sharp
section 5.28 contrast, so it can be checked on this arm); and the fixed arm's loss.

**Scope limits (pre-declared).** Toy fidelity, synthetic tokens, one substrate,
depths 2 and 3, a 3-event budget, `probe_candidate_cap=4` with `spread` policy and
`argmin` selection (the project's registered discovery configuration), and cloning
from the layer's busiest survivor rather than a trained teacher. Because the two
settings can diverge, this measures the **end-to-end** effect of the setting on
discovery; it does not separate "better mutations applied" from "better candidates
chosen", and a design that matched the probe's scoring to the application would be
needed to do that.

**Cost.** Two discovery runs plus two frozen retrains per seed, ~2.4 s/seed.

### 5.29 Outcome: branch (c) as registered - but both primaries are an artifact of prune-dominance

Run on the registered fresh blocks 109000-109383 (depth 2) and 110000-110383
(depth 3), `study_59e969bbd7767f2d` and `study_c28ad3ce4784f3b9`. Every built-in
check held and the pairing was recorded rather than assumed: identical final
genome in **339/384** and **344/384** - so the pre-registered 12.5% divergence rate
was real, at 11.7% and 10.4%, and the design's decision not to assume the pairing
was the right one.

| | depth 2 (n = 384) | depth 3 (n = 384) |
|---|---|---|
| **P1** clone helps: `C` < `R` | 134/384, mean −0.0281, p 1.00 — **FAIL** | 131/384, mean −0.0280, p 1.00 — **FAIL** |
| **P2** residual penalty: `C_own` < `C` | **221/384**, mean −0.0164, p 1.8e-03 — **PASS** | **232/384**, mean −0.0334, p 2.6e-05 — **PASS** |
| historical penalty: `R_own` < `R` | **259/384**, mean −0.0431, p 3.5e-12 | **266/384**, mean −0.0646, p 1.5e-14 |
| primary on the agreeing subset | 103/339, mean −0.0250 — FAIL | 103/344, mean −0.0236 — FAIL |

**Registered verdict: branch (c)** - P1 fails and P2 passes, so as registered the
clone result does not extend to the discovery path. The prediction (branch (a)) was
wrong: this is the fourth miss in a row after three correct ones.

> **Superseded for interpretation by section 5.30 - do not act on the branch (c)
> reading below.** That study re-ran this design on fresh blocks with the
> stratification declared in advance and found the grow-stratum effect at 73% of
the stratum in both blocks (p ~1e-11) *while the full-block statistic failed again
> (142/384, 130/384)*, so the following readings are properties of the statistic
> rather than of the mechanism. The numbers in this section stand as measured.

**But branch (c)'s pre-registered interpretation is not what the data says, and the
reason is mechanical.** The adaptive arm is not a 50/50 mix: it drew only **211 of
1152 events as grows (18.3%)** at depth 2 and 213 at depth 3, because
probe-loss selection overwhelmingly prefers pruning. So in **205/384** seeds there
is **no grow at all**, and cloning is then a literal no-op - `C - R` is exactly
`+0.0000` with **0/205 wins**, which is the cleanest possible positive control that
the setting does nothing when nothing is grown.

Two consequences follow, and both registered verdicts fall out of them:

1. **P1 failed because 53% of the block could not respond to the treatment.** A
   sign test counts a no-op seed as a non-win, and a structural tie can only ever
   be a non-win - the same asymmetry section 5.20.1 had to reason about, here at
   three times the scale.
2. **P2 passed for the same reason.** On a no-grow seed the two runs are identical,
   so `C_own - C` **is** the historical penalty; those 205 seeds carried the
   registered pass. The clone-specific quantity was never measured by P2.

**Stratifying on whether a grow actually occurred reverses both readings.** The
stratification variable is the drawn operation mix, which the clone setting does
not determine:

| | depth 2 | depth 3 |
|---|---|---|
| seeds with **≥1 grow** | 179 | 177 |
| `C` < `R` on those seeds | **134/179 (75%)**, mean −0.0602, p 9.1e-12 | **131/177 (74%)**, mean −0.0608, p 6.0e-11 |
| `C_own` < `C` on those seeds | 84/179, mean **+0.0126**, p 0.82 | 84/177, mean **+0.0016**, p 0.77 |
| seeds with **0 grows** | 205 | 207 |
| `C` < `R` on those seeds | 0/205, mean +0.0000 | 0/207, mean +0.0000 |
| `C_own` < `C` on those seeds | 137/205, mean −0.0416, p 8.2e-07 | 148/207, mean −0.0633, p 2.6e-10 |

**Where cloning can act at all, it behaves exactly as in sections 5.26 and 5.28:**
it helps by `−0.060` (three quarters of grow-containing seeds) and the residual
penalty is **gone** (P2 fails in the exonerating direction, `+0.0126`/`+0.0016`).
Not one of the 205 no-grow seeds contributed to that. So the substantive answer is
the opposite of the registered branch (c) reading: the effect **does** extend to
discovery, and the registered failure is a property of the statistic, not of the
mechanism.

**The full ordering is a different shape on this arm, and one part of section 5.28
is now qualified.** Means are `own < C < R < F` at both depths:

| | depth 2 mean | depth 3 mean |
|---|---|---|
| `C_own` / `R_own` (frozen retrain) | 14.1102 / 14.1115 | 13.3191 / 13.3223 |
| `C` clone-fused | 14.1266 | 13.3557 |
| `R` historical-fused | 14.1546 | 13.3837 |
| `F` unmutated | 14.1756 | 13.4237 |

Both fused runs beat not mutating (`C` < `F` at **252/384**, p 4.6e-10 and
**279/384**, p 1.3e-19), so mutating the discovery way pays - but **the frozen
retrain still beats both**, and it does so because of the prune term, not the grow
term: on no-grow seeds the fused run loses to its own retrain by `−0.0416`/`−0.0633`,
which is section 5.25's prune cost (about `−0.0099` per forced prune, times three)
with the grow term removed. Section 5.28's "keep the fused run" therefore holds
cleanly for the `random` arm, whose mix is half grows, and **does not hold as
stated for discovery, which prunes 82% of the time**: cloning repairs the grow term
and leaves the prune term untouched.

| | depth 2 | depth 3 |
|---|---|---|
| `C` < `F` | **252/384**, −0.0491, p 4.6e-10 | **279/384**, −0.0681, p 1.3e-19 |
| `R` < `F` | 210/384, −0.0210, p 0.037 (misses the bar) | **244/384**, −0.0400, p 6.2e-08 |
| `R_own` < `F` | **261/384**, −0.0641, p 7.9e-13 | **284/384**, −0.1046, p 7.9e-22 |

### 5.29.1 Amendment (made after the block ran): the primaries are diluted, not the mechanism

Both registered primaries are functions of a statistic that a 53%-no-op block
destroys: a full-block sign test, where a seed that cannot respond is scored as a
non-win for P1 and contributes the historical penalty to P2. The stratified numbers
above are **post-hoc** and are recorded as such. Two things bound how much they can
be trusted:

- **The stratification variable is not outcome-dependent** in the way a subgroup
  split usually is. Whether a seed contains a grow is decided by the drawn
  operation mix, which the `new_expert_initialization` setting does not determine -
  the clone setting changes the *weights*, not the enumeration or the probe's
  scoring. The `C - R == 0` exactly on all 205 no-grow seeds is direct evidence the
  setting had no channel through which to move the mix. On the 19/13 seeds whose
  operation mixes did differ, however, the assignment is not perfectly
  pre-treatment, which is a real if small impurity.
- **It was not pre-registered**, and the fix is a pre-registered test on fresh
  blocks rather than a post-hoc reading. That is section 5.30, registered below.

**The lesson worth keeping.** This is the project's first statistic whose *validity*
was destroyed by a structural no-op rather than by noise, and it produced two
confident, opposite-to-the-truth verdicts from a mechanism that is working. A
future block that tests a treatment should record how often the treatment could
bind before choosing a majority test over all seeds.

### 5.30 Pre-registration: the stratified test, registered before it runs

**Motivation.** Section 5.29's registered verdicts are artifacts of a block in
which 53% of seeds cannot respond to the treatment. The stratified reading is that
cloning helps discovery wherever a grow occurs, and that the residual penalty is
gone there - the same result as sections 5.26 and 5.28. That reading is post-hoc,
so it is tested on fresh blocks with the statistic fixed in advance.

**Design.** Identical to section 5.29 - `arms="fixed,adaptive"`, drawn operations,
`preserve_router_on_rebuild=True`, same probe configuration - plus the
stratification declared up front:

- the **grow stratum** is the seeds where the **historical (`random`
  initialization) run** drew at least one grow. Using the historical run's mix, not
each run's own, keeps the stratum a pre-treatment covariate: the clone setting
  cannot influence which seeds enter it, at the cost of diluting the stratum on the
  few seeds where the mixes differ.
- the **no-grow stratum** is the rest, and is reported as the prune-domain penalty.

**Primaries (Bonferroni `alpha/4 = 0.0125`; two depths), evaluated on the grow
stratum.**
1. **cloning helps where it can act** - `C` < `R` on the grow stratum.
2. **no residual penalty in the grow stratum** - `C_own` < `C` there, i.e. the
   predicted **failure** of P2.

**Prediction.** Both blocks are expected to land at roughly 179/384 grow-stratum
seeds with `C` < `R` at about 75% and mean `~-0.060`, and P2 to fail in the
exonerating direction - **branch (a)**: section 5.29's stratification is confirmed
and its registered branch (c) is explained as a statistic artefact.

**Pre-registered interpretation.**
- (a) P1 passes, P2 fails on the grow stratum: the clone result extends to
discovery where the treatment binds, and section 5.29's branch (c) is recorded as
superseded for interpretation (its registered numbers stand as measured).
- (b) both pass: cloning helps discovery but a residual penalty survives in the grow
  stratum, so the discovery path keeps a real mid-flight cost the `random` arm does
  not have.
- (c) P2 passes, P1 fails: the stratified reading does not replicate and section
  5.29's branch (c) stands on its merits.
- (d) neither passes: inconclusive.

**Decision rule (declared, because the stratum size is itself a random draw).** The
primary is the **exact one-sided binomial p on the grow stratum** against
`alpha/4 = 0.0125`. A fixed min-wins bar cannot be pre-committed the way section
5.29's bar of 215 was, because the number of seeds that enter the stratum is
determined by the drawn operation mix. The bar is therefore reported **for the
realized stratum size** as a convenience (`_gate_min_wins(n_grow, 0.0125)`):
**105** wins at the expected `n = 179`, **104** at `n = 177`, and the decision is
made on `p <= 0.0125` in every case. `bar_and_p_agree` is asserted for the realized
sizes so the two can never silently diverge. Section 5.29's observed grow strata
were 179 and 177, so 104-105 is the expected region and the expected effect size
(75% of the stratum, p ~1e-11) clears it by a wide margin.

**Scope limits.** As section 5.29, plus: the no-grow stratum is reported but is not
verdict-bearing, and the grow stratum is expected to be under half the block, so
both primaries are evaluated on roughly 177-179 seeds rather than 384.

**Cost.** Two discovery runs plus two frozen retrains per seed, ~2.4 s/seed.

### 5.30 Outcome: the stratification replicates, and so does the artefact (branch (a))

Run on the registered fresh blocks 111000-111383 (depth 2) and 112000-112383
(depth 3), `study_54cff7dc8d8b6f11` and `study_da528e81c3204901`. All built-in
checks held, including the positive control the stratification lives on: **in the
no-grow stratum the clone setting is a bit-identical no-op** - `C == R` exactly in
all 189 and 205 seeds, so `C_own - C` **is** `R_own - R` to the last digit.

| grow stratum | depth 2 (n = 195) | depth 3 (n = 179) |
|---|---|---|
| **P1** clone helps where it can act: `C` < `R` | **142/195 (73%)**, mean −0.0542, p 7.1e-11, bar 114 — **PASS** | **130/179 (73%)**, mean −0.0466, p 5.9e-10, bar 105 — **PASS** |
| **P2** residual penalty there: `C_own` < `C` | 98/195, mean −0.0064, p 0.50 — FAIL | 77/179, mean **+0.0144**, p 0.97 — FAIL |

| the rest of the block (reported, not verdict-bearing) | depth 2 | depth 3 |
|---|---|---|
| **full-block** `C` < `R` — the section 5.29 statistic | 142/384, mean −0.0275 — **FAIL** | 130/384, mean −0.0217 — **FAIL** |
| no-grow stratum: `C_own` < `C` (= `R_own` < `R`) | 134/189, mean −0.0553, p 4.3e-09 | 151/205, mean −0.0741, p 4.1e-12 |
| full-block `C` < `F` | 263/384, mean −0.0508, p 1.7e-13 | 260/384, mean −0.0575, p 1.7e-12 |
| operation mix drawn | 218 grow / 934 prune | 208 grow / 944 prune |

**Registered verdict: branch (a).** The stratification replicates at both depths
at **73% of the grow stratum in both blocks** - against 75% and 74% post-hoc in
section 5.29 - and the residual penalty is gone there. The prediction was correct,
and it is worth saying plainly that this is a **confirmation, not a surprise**: the
post-hoc section 5.29 numbers pointed this way, which is precisely why the fresh
blocks were run rather than the post-hoc reading being taken. What they add is
out-of-sample replication with the statistic fixed in advance.

**The sharper finding is that the artefact replicated too.** The full-block
sign test - the exact statistic section 5.29 registered - fails again here, at
**142/384** and **130/384**, *while* the pre-registered stratified test on the same
runs passes at p ~1e-11. Same block design, same seeds-worth of evidence, opposite
verdicts, and the difference is only whether the 189 and 205 seeds that could not
respond are counted. That is a direct demonstration on independent blocks that
section 5.29's branch (c) was a property of the statistic and not of the mechanism.

**Section 5.29's branch (c) is therefore superseded for interpretation** - its
registered numbers stand exactly as measured, and its *reading* ("the clone result
is confined to the random arm and does not extend to discovery") is wrong. The
same pointer added to section 5.22 applies here in reverse: the registered verdict
of 5.29 should not be acted on.

**The mechanism is now located precisely, and it is only half the discovery path.**

| | depth 2 | depth 3 |
|---|---|---|
| grow stratum (195, 179 seeds): cloning buys | −0.0542 | −0.0466 |
| no-grow stratum (189, 205 seeds): the historical penalty stands | −0.0553 | −0.0741 |

The discovery path prunes **82%** of its events, so cloning gets to act in only
about half the seeds, and where it cannot act the mid-flight penalty is untouched
and large. Means are `own < C < R < F` at both depths -
`C` 14.2688 / 13.3529, `R` 14.2963 / 13.3746, `F` 14.3196 / 13.4103, `own`
14.2368-14.2383 / 13.3177-13.3200 - so on this arm the frozen retrain still wins
overall, on the strength of the prune term alone. **Section 5.28's "keep the fused
run" holds for the `random` arm, whose mix is half grows, and does not hold as
stated for discovery**, where the remaining lever is the prune cost of section 5.25
rather than anything about initialization. That is the honest state of the
delivery recommendation: cloning is necessary but not sufficient, and the
deficit that remains on prune-dominated discovery runs is a separate, open
problem.

**Caveats, as declared.** Toy fidelity, synthetic tokens, one substrate, depths 2
and 3, a 3-event budget, the project's registered probe configuration, and cloning
from the layer's busiest survivor rather than a trained teacher. The grow stratum
is a little over half the block, so both primaries rest on ~180-195 seeds rather
than 384; the decision was made on the exact binomial p at `alpha/4`, and the
realized bars (114 and 105) matched the pre-declared expectation (105 at n = 179).
The stratum is defined by the historical run's drawn mix, which keeps it
pre-treatment; on the 65 and 48 seeds where the two runs' genomes diverged, the
clone run's own mix is not guaranteed to contain a grow, which dilutes the stratum
toward the null rather than inflating it.

### 5.30.1 Methodological conclusion, recorded because it generalises

A treatment that binds in only part of a block cannot be tested with a majority
statistic over the whole block. Here it produced two confident verdicts opposite
to the truth from a mechanism that was working, and then reproduced both the
truth and the artefact on independent blocks. The two guardrails are now in the
``adaptive_clone_stratified_study.py`` built-in checks and are the pattern to copy:

1. **Record how often the treatment can bind before choosing the statistic**, and
   report the stratum sizes next to the primary rather than the block size.
2. **Stratify on a pre-treatment variable** - here the *historical* run's drawn
   operation mix, so the setting under test cannot decide who enters the stratum -
   and include a positive control that the treatment is a bit-identical no-op
   where it should be one. That control is what makes the stratum boundary
   credible rather than convenient.

## 6. Reproducibility

- Verdict machinery and experiment code: `src/act/evolution/`
  (commits `d972d4c`, `d5ee330`, `838f619` and later).
- Study commands, seed blocks, and `study_id` values are listed per record in
  `artifacts/study-summaries/*.json`; raw JSONL dumps reproduce from config +
  seeds via `act.evolution.cli` and are regenerable deterministically on CPU.
- Registry records for single experiments append to `artifacts/experiments.jsonl`
  (gitignored); the curated summaries are versioned.

## 7. Deliverables checklist (plan §100)

| Plan deliverable | Status |
|---|---|
| 1. Source repository | DONE (this repository) |
| 2. Reproducible Kaggle notebooks | DONE for the plan section 39 set (`00_environment` + section 104's `01`-`03` and `04`-`09` per section 39's role list; deterministic builder `scripts/build_notebooks.py`, byte-identical re-runs, all ten notebooks' cells executed and verified) |
| 3. Dataset manifests | PARTIAL (synthetic-tokenizer manifests only) |
| 4. Baseline checkpoints | PARTIAL (local checkpoint format; no published artifacts) |
| 5. Adaptive checkpoints | TODO |
| 6. Architecture genomes | DONE for the gate model (serialization, hashing, fossils) |
| 7. Topology evolution logs | PARTIAL (mutation histories and fossils in records) |
| 8. Experiment registry | DONE for the gate (append-only JSONL + study summaries) |
| 9. Benchmark results | PARTIAL (this report's factor map; no real-corpus benchmarks) |
| 10. Fresh-initialization results | DONE for the toy-fidelity mechanism (sections 5.16-5.18.2: genome extraction, fresh/rewound/donor transport with 384/384 exact transport validity; the topology has standalone value under a *matched* initialization - own 293/384, donor 234/384 - and gains extra from its own initialization, 237/384 on the differenced contrast); the section-52 scaling version (300M -> 1B) is TODO and is triggered |
| 11. Qwen architecture analysis | TODO (deferred) |
| 12. Paper-quality figures | DONE for the factor map, in **two sets** (section 8): the registered coupled-path figures, and the corrected decoupled set (`--set decoupled`) plotted from summaries emitted by `scripts/factor_map_recheck.py --summaries` and re-checked by `check_summary` at plot time. Both regenerate byte-identically. Both sets carry all nine rows (section 9.12 recovered study 1's predated config field), and figure 4's counterfactual analysis is re-run separately by `scripts/probe_validity_analysis.py`. |

## 8. Figures

The four pre-registered figures are generated from the versioned study
summaries by `scripts/generate_figures.py` (`PYTHONPATH=src python
scripts/generate_figures.py`; output under `artifacts/study-summaries/figures/`).
Before plotting, the script recomputes every stored verdict from the raw
per-seed records - binomial p-value, pass decision, mean paired delta, the
probe-cost identity, and the argmin concordance - and refuses to plot on any
mismatch, so a figure can never silently drift from the machine-checked
records. Re-runs are byte-identical. The data assembly lives in
`act.evolution.figures_data` (standard library only), unit-tested in
`tests/test_report_figures.py` against scipy reference values.

**Two sets, because the factor map has two records.** The figures above are the
**registered coupled-path** record. Section 9.11 corrected every one of those
numbers, so the corrected figures are generated separately:

```bash
PYTHONPATH=src python scripts/generate_figures.py                # coupled (above)
PYTHONPATH=src python scripts/generate_figures.py --set decoupled  # corrected
```

The decoupled set is written to `artifacts/study-summaries/decoupled/figures/`
and plotted from summaries in the committed schema produced by
`scripts/factor_map_recheck.py --compare ... --summaries` - each one validated by
`check_summary` before it is written and again at plot time, so the corrected
figures carry the same machine check as the registered ones. The two sets plot
the same factor map - both carry all nine rows, and section 9.12 recovered the
field that once left study 1 out of the corrected one - so the difference is in
what the arms' RNG regime does to the numbers, not in which studies appear. Two
points about the corrected set are worth stating:

- **Figure 3 shows studies 1, 3, 4, 5 and 6** - the per-seed delta
  distributions - in both sets.
- **Figure 4 is set-specific.** The counterfactual probe-rank analysis behind
  it is not an arm comparison, so it is re-run separately
  (`scripts/probe_validity_analysis.py`) and the decoupled set plots its own
  analysis rather than sharing the coupled one: concordance 0.284 vs 0.25 and
  rho 0.0776 (sections 3.2, 9.11). With that, **no figure in the corrected set
  is plotted from coupled-path data.** Because the two analyses differ only
  slightly in the plotted cells, figure 4's title now names its RNG regime - a
  title-only edit that also regenerated the registered coupled figure, which is
  the one intentional change to a published figure in this revision.

Section 3's table, and the corrected win counts it quotes, come from the
decoupled set. Both sets regenerate byte-identically, so neither can drift from
the summaries it is plotted from.

1. **`fig1-winrate-forest.png` - win-rate forest plot**: one row per study
   (#1-#9), point estimate with the exact binomial (Clopper-Pearson) 95%
   interval, threshold line at the 384-seed pass boundary (209/384).
2. **`fig2-quality-vs-cost.png` - quality vs probe cost scatter**: mean delta
   (x) against probe cost as % of treatment training FLOPs (y, log scale),
   marker size = seed count, color = verdict. The passing cap-4 family buys
   quality with probe FLOPs; every cheap variant is a coin flip.
3. **`fig3-delta-distributions.png` - per-seed paired delta distributions**
   for studies 1, 3, 4, 5, 6 (boxes colored by verdict, zero line = no
   adaptive benefit).
4. **`fig4-probe-rank-heatmap.png` - probe-rank vs final-rank heatmap** from
   the 384 counterfactual branches: rows are probe ranks, columns realized
   final ranks, each row the conditional distribution of final rank given
   probe rank; cell (probe 1, final 1) is the argmin concordance - 0.281
   coupled, 0.284 decoupled (chance 0.250 in both).

## 9. Substrate audit: a per-layer bug, a probe-policy axis, and a pool confound

**Status: exploratory and UNREGISTERED.** These runs are not part of the gate
family and were not pre-registered; they use the fresh block 20000-20127
(disjoint from every prior block) and no rule was tuned against an outcome.
They are reported because section 9.5 revises a claim section 2 makes.

### 9.1 The toy path could not represent more than one layer

`StaticMoEConfig` carried a single `num_experts` for every block and
`StaticMoEBlock.__init__` required `len(expert_ids) == config.num_experts`.
Any genome with unequal per-layer expert counts - i.e. any real topology
mutation at depth > 1 - therefore raised `expert_ids must match num_experts`
on rebuild. Every published gate study ran at `depth=1` (confirmed against
`artifacts/study-summaries/*.json` and the `ExperimentConfig` defaults), so the
entire factor map - nine studies, three passes, the richness axis, the probe
validity analysis - was produced on a **single-layer, 4-expert, 5-candidate
mutation space**. Each block now derives its expert count from its own ids, so
heterogeneous layers build and evolve.

### 9.2 The candidate cap truncated a layer-major list

`feasible_expert_mutations` enumerates layer-major (per layer: `grow`, then
that layer's prunes). The bounded probe took `candidates[:cap]`, so at depth
greater than 1 a cap of 4 reached only the first layer's grow plus three of its
prunes - the cap was a *layer* constraint, not a search budget.
`ExperimentConfig.probe_candidate_policy` now selects `"prefix"` (the
historical default, so published studies still reproduce) or `"spread"` (a
deterministic per-event subset of the whole feasible set). At depth 4 / cap 4,
prefix probes only layer 0; spread spans layers.

### 9.3 Iteration cost (why the audit was cheap)

Depth-4 toy studies run at ~1.5 s/seed on CPU; a 128-seed study takes ~85 s.
GPU is *slower* for this substrate (measured 2.21 vs 1.52 s/seed at hidden 24 -
kernel-launch overhead dominates). The historical depth-1 studies ran at
~1.4 s/seed.

### 9.4 Does the effect survive a deeper substrate?

64 seeds 20000-20063, steps=24, cap 4/1-batch, uniform-random baseline:

| Substrate / policy | Wins | Mean delta | Binomial p |
|---|---|---|---|
| depth 1, prefix (historical) | 38/64 | -0.0245 | 0.084 |
| depth 4, prefix (cap reaches layer 0 only) | 47/64 | -0.1000 | 1.1e-04 |
| depth 4, spread (cap spans the space) | 48/64 | -0.1133 | 3.9e-05 |

So the pass is not an artifact of the depth-1 space, and the spread fix does
not weaken it. A probe-rank gradient on the same pool and cost (64 seeds)
confirms the ordering carries signal: probe-best 47/64, forced middle 37/64,
probe-worst 35/64.

### 9.5 The pool confound: "only selection differs" is not true

The adaptive and random arms differ in **two** ways, not one: which candidates
can be mutated (a capped pool) and how one is chosen from it. Section 2's
method claim that "only the *selection policy* of mutations differs" is
incorrect. `probe_selection="random"` (mirroring the V0 diagnostic's rule)
holds pool and probe cost fixed and chooses uniformly, isolating the decision
rule. 128 seeds 20000-20127, depth 4, steps=24, cap 4/1-batch, same uniform
baseline (threshold 74/128):

| Adaptive decision rule | Wins | Mean delta | Binomial p |
|---|---|---|---|
| argmin (registered) | 88/128 | -0.0813 | 1.3e-05 |
| **random within the same pool** | **80/128** | **-0.0340** | **0.0030** |
| argmax (inverted) | 64/128 | +0.0108 | 0.535 |

The ordering is real - a monotone gradient across the full range of the rule
(88 > 80 > 64, argmax at chance) - but a control that ignores probe loss
entirely and picks randomly from the same pool **also passes decisively**.
Roughly 40% of the argmin edge is therefore attributable to the candidate
pool (at depth 4, to concentrating mutations on layer 0), not to selection.
(These runs are all on the shared-RNG path; the clean, powered re-measurement
in section 9.7 puts the pool's share of the mean gain at ~29% - lower, and with
selection carrying the larger, significant part.)

**Consequences.** (1) The factor map's passes overstate probe selection:
part of the effect is a location policy shared by the pool, and the effect
size, not the mechanism, is what needs correcting. (2) Any re-run must use a
pool-matched random baseline, so the section 5.1 cost protocol should not
proceed as though the quality edge were established at V0 scale. (3) The
section 5.1/5.2 registered controller truncates the same layer-major list
(108 candidates at the declared depth, so a cap of 4 reaches layer 0 only),
and the V0 session now carries the same `probe_candidate_policy` and
`probe_selection` knobs - but the REGISTERED config is unchanged at the
`prefix`/`argmin` defaults, so the pre-registered protocol still carries the
confound until the policy is deliberately switched and re-registered. See
`IMPLEMENTATION_STATUS.md`'s substrate-audit section for the code status.

**Caveats.** Exploratory and unregistered; 128 seeds at steps=24 (shorter than
the registered 48); one fresh block; no significance claim beyond the binomial
reported. The formal re-registration on a fresh block, with the pool-matched
control as the baseline, is the planned next step.

### 9.6 The same structure in the declared V0 path - and no edge at reduced scale

`V0SessionConfig` now exposes the same `probe_candidate_policy` and
`probe_selection` knobs (registered defaults unchanged at `prefix`/`argmin`).
A reduced-dim session run (depth 6 x 4 experts = 30 candidates, cap 4,
steps=120, 12 seeds 30000-30011, both arms per seed, no real data) confirms the
structural difference in the declared code path: prefix probes only layers
L0-L1 (the candidate list spills into L1 once L0's prune slots empty), while
spread probes all six.

It does **not** quantify the confound, because at this fidelity there is no
adaptive edge to confound:

| Rule | Wins | Mean delta |
|---|---|---|
| prefix / argmin (registered) | 3/12 | +0.0005 |
| spread / argmin | 5/12 | +0.0009 |
| prefix / random (pool-matched) | 5/12 | -0.0003 |

All three sit at noise (mean deltas ~1e-03, no rule distinguishable from
random). This is a 12-seed unregistered check at reduced dimensions: it
establishes nothing about V0 behaviour, and it is consistent with the
real-data harm-guard rejection (section 5.3) - the adaptive edge has never
been observed outside the tiny toy substrate.

**Consequence for the section 5.1 protocol.** The T4 run should not be launched
on the current pre-registration as though the quality edge were established:
(a) the registered prefix controller cannot reach past the first one or two
layers at the declared depth, and (b) the only positive evidence for selection
value comes from a substrate with a demonstrated pool confound. Deciding the
candidate policy and adding a pool-matched control arm (same candidates, same
probe cost, loss-blind choice) is a prerequisite, and that decision belongs in
the pre-registration, not in the analysis.

### 9.7 The depth-4 mechanism, measured cleanly: the gate and the increment both pass

Under the revised design (`spread`, so a cap of 4 is a real budget over 20
candidates rather than a layer-0 prefix). Four blocks; the differences are RNG
regime (section 9.8), sample size and horizon:

| Comparison | Wins | Mean delta | Binomial p |
|---|---|---|---|
| **steps=24, 128 seeds, SHARED RNG** (historical default), block 40000-40127 | | | |
| argmin vs uniform churn (the registered claim) | 65/128 | -0.0143 | 0.465 |
| loss-blind pool vs uniform churn | 63/128 | +0.0100 | 0.604 |
| argmin vs loss-blind pool (paired, same pool/cost) | 76/128 | -0.0243 | 0.021 |
| **steps=24, 128 seeds, INDEPENDENT arm RNG**, block 40000-40127 | | | |
| argmin vs uniform churn | 80/128 | -0.0609 | 0.0030 |
| loss-blind pool vs uniform churn | 68/128 | -0.0214 | 0.268 |
| argmin vs loss-blind pool (paired) | 72/128 | -0.0395 | 0.092 |
| **steps=48, 64 seeds, INDEPENDENT arm RNG** (underpowered; threshold 39/64) | | | |
| argmin vs uniform churn | 33/64 | -0.0345 | 0.450 |
| loss-blind pool vs uniform churn | 31/64 | +0.0005 | 0.646 |
| argmin vs loss-blind pool (paired) | 39/64 | -0.0350 | 0.052 |
| **steps=48, 384 seeds, INDEPENDENT arm RNG - POWERED**, block 50000-50383 | | | |
| argmin vs uniform churn (the registered claim) | **246/384** | **-0.0745** | **1.9e-08** |
| loss-blind pool vs uniform churn | 217/384 | -0.0215 | 0.0062 |
| argmin vs loss-blind pool (paired) | **222/384** | **-0.0529** | **0.0013** |

**Withdrawn claim.** An earlier draft of this section concluded, from the
shared-RNG rows, that "the registered Gate 3 comparison does not survive a
non-degenerate pool". That reading is **withdrawn**: the shared-RNG path is not
a neutral default, and fixing it moves the registered comparison from 65/128
(p = 0.465) to 80/128 (p = 0.0030) on the same seeds.

**Final reading of the clean, powered block (the one that decides).** At 384
seeds, steps=48, independent arm RNG, on a fresh block:

- The **registered comparison passes decisively**: 246/384 (64.1%), mean
  `-0.0745`, p `1.9e-08`.
- A **loss-blind choice from the identical pool also beats uniform churn**
  (217/384, p `0.0062`), so the candidate pool is part of the effect - the
  location contribution is about `0.0215 / 0.0745 ~= 29%` of the mean gain, not
  most of it as the coupled `prefix` runs of 9.5 suggested.
- **Probe ordering adds a decisive increment over the identical pool at
  identical cost**: 222/384 (57.8%), mean `-0.0529`, p `0.0013`.
- All three comparisons clear Bonferroni alpha/3 = 0.0167.
- The underpowered steps=48/64-seed block is consistent with this rate (at
  57.8%, 37 expected of 64, observed 33), so its null result is a power effect,
  not a different mechanism.

The random-arm invariance check is exact in this regime (384/384 seeds), which
is what the decoupling guarantees.

**Implication for Gate 3.** Once the arm coupling is fixed, the gate passes on a
substrate that can express topology (depth 4 with a non-degenerate candidate
pool), on both the registered comparison and the controlled increment test. The
depth-1 passes of section 3 should be regarded as a weaker, degenerate-space
instance of the same result rather than the project's only evidence - but the
depth-4 result is still toy-scale and needs a pre-registered re-run before it
is cited as the project's gate.

**Caveats.** Unregistered, one fresh block (50000-50383), synthetic tokens, a
tiny depth-4 model (24 hidden), and a claim only about the toy substrate - the
V0 declared path has not been measured in this regime. Reproduce with
`PYTHONPATH=src python scripts/toy_policy_ablation.py --policies spread
--independent-arm-rng`.

### 9.8 The toy arms are coupled through the global torch RNG

`run_experiment` trains the arms interleaved in one step loop against a shared
global torch RNG. A `grow` mutation constructs a new expert and draws its
weights from that stream; a `prune` draws nothing. Two runs whose treatments
choose a different mix of grow and prune therefore consume different numbers of
global draws, which perturbs the *random* arm's later rebuilds. The random arm
is consequently only approximately the same baseline across treatments.

Measured on the section 9.7 block (128 seeds, spread policy): the random arm's
final loss is bit-identical between the argmin run and the pool-random run in
only **70/128** seeds - in 58 of 128 seeds the "uniform baseline" is a
different control in the two runs. `scripts/toy_policy_ablation.py` prints this
as a `[coupling]` line rather than asserting independence.

Consequences:

- It is not a nuisance term but a first-order effect at low power: section 9.7
  measures the registered comparison at 65/128 (p = 0.465) under the shared RNG
  and 80/128 (p = 0.0030) under independent arm RNG on the same seeds, and the
  size of the swing shrinks as the sample grows. Any toy arm comparison made on
  the shared path - which is **every published gate study**, all nine of them
  plus the placement and probe-validity runs - carries a baseline that drifts
  with the treatment, so the reported effect sizes are not exactly controlled.
  The effect on the conclusions is smaller than the summary above first
  implied: re-running study 3's own block (3000-3383) reproduces it exactly on
  the shared path (213/384, mean `-0.025955`, p `0.01814` - bit-for-bit the
  committed artifact) and gives a slightly **stronger** result with the fix
  (222/384, mean `-0.034406`, p `0.00128`). The depth-1 pass therefore survives;
  the fix should still be the default for new studies.
- The V0 path does not share it: arms are trained sequentially with per-arm
  seeding, which is why the real-data diagnostic checked and found every random
  arm replicating its baseline counterpart exactly.

**The fix is implemented as an opt-in flag.** `ExperimentConfig
.independent_arm_rng` (default `False`) re-seeds the global RNG per (arm, event)
before each mutation, so rebuild initialisation cannot leak between arms; the
shared init identity that the pairing relies on is untouched. The default stays
`False` so existing numbers reproduce unchanged, which is also why the historical
studies still need an explicit re-run rather than being silently reinterpreted.

**Convention for new studies (decided here, rather than by flipping a
default).** Every new study must pin the flag explicitly and appears in the
record with it set, as section 5.6 does. The defaults are deliberately *not*
flipped: the CLI and `ExperimentConfig` together are the reproduction surface
for history, and the pre-audit study summaries do not carry the field at all,
so a flipped default would silently change the numbers a reader gets when
reproducing a committed artifact - the exact failure this report avoids
everywhere else. Pinning per study makes the regime visible in each record and
leaves old reproductions honest.

### 9.9 Result: the depth-4 confirmation PASSES out-of-sample

The section 5.6 pre-registration was committed as `b876ac1`, *before* the run,
and then executed once on the never-observed block 60000-60383 (384 seeds,
depth 4, `spread`, `independent_arm_rng=True`, cap 4/2-batch, steps=48). Every
number below is out-of-sample for the entire project.

| # | Comparison | Wins | Mean delta | Exact binomial p | Threshold |
|---|---|---|---|---|---|
| 1 | `argmin` vs uniform churn (registered claim) | **255/384** (66.4%) | **-0.1004** | **6.1e-11** | PASS (needs 214 at alpha/3) |
| 2 | pool-random vs uniform churn | 209/384 (54.4%) | -0.0324 | 0.046 | fails alpha/3; marginal at alpha |
| 3 | `argmin` vs pool-random, paired (mechanism) | **236/384** (61.5%) | **-0.0680** | **4.1e-06** | PASS (needs 214 at alpha/3) |
| V | `argmax` vs uniform churn (validity control) | 173/384 (45.1%) | +0.0355 | 0.977 | does not pass - study is not void |

Random-arm invariance is **384/384** across all three decision rules, which is
exactly what the decoupling guarantees and confirms the run is valid.

**Verdict: PASS.** Comparisons 1 and 3 clear Bonferroni alpha/3 = 0.01667 by a
wide margin (both p < 1e-05), and the inverted-signal control does not pass, so
the signal carries real ordering information. Under section 5.6's frozen rules:
Gate 3 is established on a substrate that can express a topology search, and
section 5.5's justification is confirmed.

**What it changes, and what it does not.**

- The registered Gate 3 comparison now passes **out-of-sample** on a
  non-degenerate pool (20 feasible candidates over 4 layers), not just on the
  exploratory block that motivated the design. The rate is slightly stronger
  than the exploratory run (66.4% vs 64.1% for comparison 1; 61.5% vs 57.8%
  for the paired increment), so the effect is not a block artifact.
- The **pool is a minority of the effect**: a loss-blind choice from the
  identical pool clears only alpha = 0.05 (209/384, p = 0.046) and fails
  alpha/3, contributing ~`0.0324 / 0.1004 ~= 32%` of the mean gain. The
  selection increment is the larger share (~68%) and clears alpha/3 easily.
  This resolves the section 9.5 question in the mechanism's favour on a
  substrate where the pool is genuinely larger than the cap.
- It does **not** extend the claim beyond the toy substrate. The model is
  hidden-24 depth-4 on synthetic tokens; the V0 declared path has still not
  been measured in this regime (section 9.6 found no edge there at reduced
  dimension, and section 5.3 found none on real text). Gate 3 is established
  for the toy gate, which is what the plan's gate 3 asks for; the real-scale
  question is the section 5.1 cost protocol.
- The absolute search cost is unchanged in character: probe search is ~93x the
  random arm's search FLOPs here (the mechanism's price at toy scale), which is
  precisely why section 5.1 measures the *ratio to training* rather than the
  ratio to the churn baseline.

**Reproduce.** The section 5.6 one-liners (three `act.evolution.cli` runs, one
per decision rule, over `60000-60383`), aggregated as above; on CPU the whole
study is ~2.5 min at 16 single-threaded shards. Hardware note worth keeping:
`--device` must be pinned to `cpu` for this substrate. The default `auto`
resolves to CUDA, and 16 processes contending for one GPU made the same study
take >13 min before it was killed - the hidden-24 model is CPU-faster (section
9.3), and a GPU-contended run is strictly worse than a CPU-sharded one.

### 9.10 Declared re-run of the published factor map on the decoupled path

**Why.** Section 9.8 establishes that all nine published gate studies - and the
guided upper bound - ran with the arms coupled through the global torch RNG, so
their "matched random churn" baseline drifts with the treatment. The depth-1
factor map is the evidence behind section 3's narrative ("nine studies, three
passes", "richness is the active axis", "cadence beyond three events is
actively harmful"). Only one of those studies has been re-run since: study 3's
own block came back 222/384 with the fix (still PASS) and saved. The rest are
still quoted from the coupled path.

**Declaration (fixed before the run).**

- **Scope.** Every committed study summary that reconstructs the published
experiment exactly from its recorded config: the eight probe-loss gate studies
(`probe-loss`, `horizon`, `rich`, `narrow`, `cheap-probe`, `variance`,
`cadence`, `late-placement`) plus `guided-upper-bound-96`. Each is re-run with
its **published config, published seed block and treatment arm**, with
`independent_arm_rng=True` as the only change, at the same probe budgets.
- **Reconstruction: `gate3-legacy-signal-384`.** Its recorded config predates
  the `adaptive_policy` field, so running it as recorded would silently default
  to `probe_loss` and measure a different experiment. The recorded value is
  recoverable (`"signal"` - see section 9.12), so the study is re-run on its
  completed config rather than excluded.
- **Decision rule.** A study **reproduces** if its verdict label at alpha=0.05
  is unchanged (`gate_passed` recomputed on the same n); it **flips** otherwise.
  A flip revises the published claim; it is not a new result to be preferred,
  and the coupled number is not "wrong" - it is the number that path produces.
  A reproducing study gains no strength from the recheck: it is still a
  depth-1 toy gate.
- **Reported either way** (so a "reproduces" label cannot hide a material
effect-size move): win counts, exact binomial p, mean adaptive-minus-random
delta, and the win-count delta against the committed summary.
- **Device.** The summaries record `device: auto`; the recheck pins `cpu`
  (section 9.3) on the strength of the section 9.9 result that this substrate
  reproduces committed CUDA-produced numbers exactly on CPU.

**Outcome contract.** No outcome is pre-judged. A clean sweep means section 3's
narrative survives the control fix and only the effect sizes need restating; any
flip is written into section 3 and the milestone table as a revision, with the
coupled number kept beside it for the audit trail.

**Reproduce.**

```bash
PYTHONPATH=src python scripts/factor_map_recheck.py --list
# 16 single-threaded shards per study, then aggregate:
PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/factor_map_recheck.py \
  --study gate3-probe-loss-384 --shard 0 --shards 16 --out /tmp/recheck
PYTHONPATH=src python scripts/factor_map_recheck.py \
  --compare '/tmp/recheck/*.json' --compare-out artifacts/factor-map-recheck.json
```

`--coupled` re-runs the historical shared-RNG path for a control comparison, so
the coupled/decoupled difference is measurable per study rather than inferred.
**Result recorded in section 9.11.**

### 9.11 Result: nine of ten labels reproduce, and every effect moves toward the treatment

Ran as declared (144 shards over 9 studies, ~6 min wall on 16 single-threaded
cores; comparison artifact
`artifacts/study-summaries/factor-map-recheck.json`). Study 1 was added
separately by section 9.12, which recovered the field its record predates.

| # | Study | Published | Recheck (decoupled) | Win shift | Verdict |
|---|---|---|---|---|---|
| 1 | Legacy controller | 195/384 FAIL (p 0.399) | 201/384 FAIL (p 0.193) | +6 | reproduces |
| 2 | Guided upper bound | 59/96 PASS (p 0.0158) | 62/96 PASS (p 0.00279) | +3 | reproduces |
| 3 | Probe-loss controller | 213/384 PASS (p 0.0181) | 222/384 PASS (p 0.00128) | +9 | reproduces |
| 4 | Cheap probe | 181/380 FAIL (p 0.835) | 188/380 FAIL (p 0.601) | +7 | reproduces |
| 5 | Cadence | 137/384 FAIL (p ~ 1.0) | 148/384 FAIL (p ~ 1.0) | +11 | reproduces |
| 6 | Variance reduction | 179/384 FAIL (p 0.916) | 209/384 **PASS** (p 0.046) | +30 | **FLIPS** |
| 7 | Horizon replication | 231/384 PASS (p 4.04e-05) | 250/384 PASS (p 1.68e-09) | +19 | reproduces |
| 8 | Richness A | 228/384 PASS (p 1.4e-04) | 229/384 PASS (p 9.36e-05) | +1 | reproduces |
| 9 | Richness B | 193/384 FAIL (p 0.480) | 198/384 FAIL (p 0.287) | +5 | reproduces |

**What this changes.**

- **The section 3 narrative survives.** Nine of ten labels reproduce, so
  "richness is the active axis", "probe quality is secondary", "the pass is not
  horizon-specific" and "the legacy signal controller is a coin flip" are all
  unchanged in verdict.
- **But every single study moved toward the treatment** (win shifts +1 to +30,
  mean +9.2 wins ~ +2.6 points, all ten positive). The coupling was therefore
  a *directional* bias, not symmetric noise: on the shared-RNG path every
  published number - passes and nulls alike - understated the adaptive arm.
  That is the strongest statement this recheck licenses, and it is a statement
  about the previous measurements rather than about the controller.
- **The family-level evidence strengthens.** Clearing Bonferroni alpha/9 goes
  from **2 of 9 studies** (horizon, richness A) to **4 of 9**: probe-loss now
  clears it (p 0.00128) where the coupled number was marginal (0.0181), and the
  guided diagnostic does too (0.00279). Horizon also moves from 4.0e-05 to
  1.7e-09.
- **The one flip is a null, not a new positive.** Variance reduction moves from
  a clear-looking FAIL (179/384) to a threshold PASS (209/384, exactly the
  alpha=0.05 minimum). At p 0.046 it does **not** clear alpha/9, so the honest
  reading of that study is "no probe-batch effect in either direction" - the
  coupled baseline made a null look like a marked failure. It must not be
  quoted as evidence that quadrupling probe batches helps.
- **The one "actively harmful" claim survives, weakened.** Cadence stays
  significantly worse than random on both paths, but less severely: 137/384
  (P(X<=137) = 1.1e-08) becomes 148/384 (P(X<=148) = 4.1e-06). The claim is
  real; its published magnitude was inflated by the baseline.
- **Cross-validation.** The probe-loss recheck returned 222/384, bit-identical
  to the independent running made in section 9.8 through a different entry
  point (the CLI over the same block), which is the check that the recheck
  harness measures the same thing.
- **The mechanism analysis was re-run as well** - the counterfactual probe-rank
  sweep behind section 3.2 and figure 4, which until now had **no committed
  generator** and was therefore reproducible only in principle. It has one now
  (`scripts/probe_validity_analysis.py`), and the first thing that script did
  was reproduce the committed coupled results field for field from a fresh
  4-branch x 384-seed sweep, which is what makes the decoupled re-run a
  controlled change rather than a second opinion. Decoupled: mean per-seed
  Spearman rho **0.1057 -> 0.0776** (one-tailed p 1.4e-04 -> 4.98e-03) and
  pooled within-seed Pearson r 0.1041 -> 0.0809, while the avoidance statistics
  barely move (concordance 0.281 -> 0.284 vs chance 0.25, worst-branch excess
  avoided 0.0860 -> 0.0839, left on table 0.0601 -> 0.0592). The ranking signal
  is weaker than published; the "value is in not choosing badly" reading is
  confirmed and does not depend on the RNG regime. **Honesty note:** unlike
  section 9.10 this re-run was not pre-registered. What stands in its place is
  that the analysis has a frozen `design` string, the seed block and every
  statistic are unchanged, the coupled path was reproduced field for field
  first, and the only difference is the RNG regime - so it is a controlled
  re-measurement rather than a fresh hypothesis. It should still be treated as
  a reproduction check, not as new pre-registered evidence.

**Caveats.** Nine studies at depth 1 on a five-candidate space, so this
strengthens or corrects a degenerate-space factor map, not the project's gate
(section 9.9 is the gate). The variance flip sits exactly on a threshold, so
it is reported as a null rather than as a pass. The shift's magnitude tracks
neither probe cost nor event count cleanly (richness A moves +1 while variance
moves +30), so the coupling's dependence on the treatment's choices is
characterised but not explained; the natural hypothesis - more probe draws give
sharper picks, and grows draw weights where prunes do not, so sharper picking
consumes more of the shared stream - is untested here. The
legacy controller was initially excluded as not reconstructible; section 9.12
recovered the predated field and its row is the first in the table above, so no
study in the factor map now rests on the coupled path alone.

**Consequence for the project.** No published *verdict* in the factor map needs
revising except variance, and no headline claim is overturned. Every published
*effect size* does need the recheck number beside it, and the depth-1 factor map
should from here on be quoted from the decoupled column.

### 9.12 Declared: the legacy signal study is reconstructible, not unknowable

Section 9.10 excluded `gate3-legacy-signal-384` on the grounds that its
recorded config predates the probe fields, so `ExperimentConfig(**recorded)`
would fill them with today's defaults and "silently reproduce a different
experiment". Inspecting the record rather than the summary shows that half of
this is right and half is a mistake.

The record does predate exactly one decision-bearing field: `adaptive_policy`
(added by `d972d4c`, which replaced the cheap-signal controller with bounded
probe-loss selection). Defaulting it yields `probe_loss` - a different
experiment. But the value is *recoverable*, on three independent grounds:

1. `d972d4c`'s diff inserts the probe branch **above** the legacy
   `random.Random(seed)` chooser and does not touch the fallback body, so
   `adaptive_policy="signal"` *is* the pre-transition controller rather than a
   reconstruction of its behaviour.
2. The raw record `artifacts/gate3-384.jsonl` corroborates it directly: its
   per-arm `mutation_history` entries carry no probe fields at all, so the
   probe controller never ran in it.
3. It reproduces the published coupled record. One published seed
   (1000) lands within **4.8e-7** under `signal`, against **1.1e-1** under the
   defaulted `probe_loss` - float drift versus a different experiment. This is
   pinned as a test (`tests/test_factor_map_recheck.py`), not left as prose.

**Declared before running.** The legacy study joins the section 9.10 recheck
set with `adaptive_policy="signal"` supplied (`RECONSTRUCTION` in
`scripts/factor_map_recheck.py`), and its decoupled re-run is then judged by
section 9.10's *unchanged* rule: the label reproduces when it is unchanged at
alpha=0.05 on the same n, and flips otherwise. The validation criterion above
is deliberately outcome-independent - it compares against the **published**
coupled record, so the reconstruction cannot be accepted or rejected on the
strength of what the decoupled run turns out to say. What this does *not* fix:
the record is still missing `adaptive_policy` as a stored field, so a future
reader who does not apply the reconstruction will still get the probe_loss
experiment; the field is supplied at run time, not written back into the
artifact.

## 10. Synthesis: the state of the program after sections 5.8-5.15

**(2026-09-15.** This section consolidates the pre-registered studies of
sections 5.8-5.15 with the earlier factor map into the current picture.
Every number is machine-checked against the committed study summaries in
`artifacts/study-summaries/` by pinning tests.*)

### 10.1 The two-axis picture

The program now has two cleanly separated empirical axes:

1. **The churn axis (substrate-general).** On real text at diagnostic
   fidelity, EVERY topology mutation policy loses to doing nothing:
   probe-ordered churn (5.3, all five variants), probe-gated churn at
   every threshold (5.4, 5.11 - with a monotone dose-response of harm vs
   acceptance rate, +0.0027 / +0.0236 / +0.0713), load-directed churn
   (5.13, 0/10 vs null), and random churn by construction (5.8, 8/8 for
   the null). The null arm never lost a registered comparison on real
   text. On the toy path, churn harm (null < random) holds at every
   measured depth (5.9-5.15, all p < 1e-9).

2. **The depth axis (toy path).** Value creation by probe-ordered
   selection is a function of substrate depth: adaptive-vs-null wins go
   117 (d1) < 161 (d2) < 188 (d3) < 225 (d4) of 384, mean delta
   +0.0508 > +0.0208 > +0.0046 > -0.0371 - a monotone dose-response with
   the sign flip between depth 3 and 4 (5.15), robust to the candidate
   policy (5.12 2x2: prefix and spread agree in every cell).

Together: **selection value exists, but only where the substrate is deep
enough to support it.** Below the onset, every mutating arm loses to doing
nothing; the historical depth-1 passes were churn suppression measured
against random churn (the only control they had), not value creation.
Above it, about a quarter of the raw edge survives the strictest control
(0.0297-0.0371 of 0.1130).

### 10.2 What this means for the gates

- **Gate 3 stands, narrowed.** The depth-4 confirmation (5.6) passes with
  the null control under it (5.10): adaptive beats doing nothing
  out-of-sample, 218/384 with spread and 225/384 with prefix (5.12b).
  The depth-1 instances of Gate 3 are re-quoted as churn suppression.
- **Gate 4 is prepared, amended, and unspent.** The 5.1/5.5
  compute-efficiency protocol (block 15000-15011) now carries the 5.14
  do-nothing clause: a null queue at declared fidelity runs alongside the
  primary and pool-matched control queues, and Gate 4 is withheld unless
  the adaptive arm beats doing nothing on the declared block. The
  three-queue launch path is smoke-tested; the run needs a T4.
- **The real-text topology program is closed** (5.13). Its residue is
  the churn-harm mechanism and the replication-grade tooling (null arm,
  bit-identity checks, pre-registration discipline) the toy program now
  inherits.

### 10.3 What would change this picture

By pre-registration, the named open questions are: (i) the Gate 4 run -
a pass authorizes the next gate; a null-clause rejection replicates the
real-text ordering at declared fidelity and closes the 5.1 program's
selection premise at its own scale; (ii) the depth-3/4 boundary - the
onset is bracketed to (3, 4], and a finer covariate (parameter count,
layer width, expert count) would separate depth per se from capacity;
(iii) a real-text scale run above the diagnostic fidelity (the closed
program says nothing about longer horizons or larger models). Nothing in
the current picture depends on re-tuning an existing cell, and no knob
may be moved against a measured result (the 5.5 discipline).
