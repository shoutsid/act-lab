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
  corrected control is pool-matched.
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
should be the baseline for any re-run.

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
events on a weak probe is actively harmful. What remains open is
**compute-efficiency**: the validated cost-saving variant (cap-4/1-batch)
cuts probe search to 7.3% of toy-scale training FLOPs and amortizes with
horizon exactly as the fixed-per-event model predicts, but scaling stays
withheld until a real-scale measurement shows the cost becomes negligible -
which is the plan's deferred compute-efficiency milestone - after which Gate 4
(adaptive > static MoE) is the next universal gate. That measurement is no
longer a vague milestone: it is pre-registered as a protocol with declared
config, predicted cost, and decision rules (section 5.1). The controller
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

### 5.4 Pre-registration: accept/reject gating of random draws (designed, not yet run)

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

### 5.8 Pre-registration: the do-nothing control arm (designed, not yet run)

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

### 5.9 Pre-registration: does the toy adaptive edge clear the do-nothing bar? (designed, not yet run)

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
verdict`; shards under `artifacts/toy-null-study/`, not committed):
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

### 5.10 Pre-registration: the depth-4 confirmation against the null arm (designed, not yet run)

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
shards under `artifacts/toy-null-study/depth4/`, not committed); built-in
checks held in 384/384 seeds. **Primary 1 PASS**: null-vs-random 254/384,
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
third control under it.

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
| 10. Fresh-initialization results | TODO (policies exist; studies deferred) |
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
