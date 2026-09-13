# Adaptive Computational Topology — Research Report

Status: skeleton covering the narrowed first scientific milestone (Gate 3,
low-fidelity Experiment 0). Updated: 2026-09-13.

This report is the narrative companion to `IMPLEMENTATION_STATUS.md` (the
implementation tracker) and the append-only study records under
`artifacts/study-summaries/`. Every number below is machine-checked against
those records; `study_id` values are reproducible via
`PYTHONPATH=src python -m act.evolution.cli ...` from the stated config and
seeds.

## 1. Hypothesis (plan §98 format)

- **Hypothesis**: adaptive topology improves capability per FLOP.
- **Prediction (Gate 3)**: adaptive topology selection beats matched random
  churn on final held-out loss.
- **Control**: identical training schedule, identical mutation count and
  cadence, only the *selection policy* of mutations differs.
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
passed three times, twice decisively.

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
The **variance** study (study 6) tested the complementary winner's-curse
hypothesis — that single-batch probes mislead because their argmin partly
selects for probe-batch noise — by quadrupling probe batches at fixed
candidates; the mean moved in the predicted direction (+0.0117 to -0.0039) but
not significantly, so variance reduction alone does not restore the edge.

### 3.2 Mechanism: what the probe signal is worth

A pre-registered counterfactual analysis
(`artifacts/study-summaries/probe-validity-analysis.json`) forced each of the
4 candidates at the first mutation event (branch 0 verified trajectory-identical
to greedy selection) and compared probe ranks to realized final ranks across
384 seeds:

- Probe loss **does** predict realized loss: mean per-seed Spearman
  rho = 0.1057 (t(383) = 3.64, one-tailed p = 1.4e-04); pooled z-scored
  Pearson r = 0.1041 (z = 4.09).
- But weakly, and mostly as **avoidance**: the probe-argmin matches the
  realized best in only 28.1% of seeds (chance 25%), while it avoids a mean
  worst-branch excess of 0.0860 and leaves only 0.0601 on the table.

The coherent story across all nine studies: **selection value comes from having
candidates worth choosing between**. Probe loss is a real but weak predictor
(rho ~ 0.11) whose value is concentrated in avoiding bad mutations; a rich
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

- Nine gate-style quality tests have been run. Study 3's sign-test pass
  (p = 0.0181) is marginal under Bonferroni, but studies 7 and 8 clear the
  family-wise bar (alpha/9 ~ 0.0056) on *both* statistics (4.0e-05/1.1e-08
  and 1.4e-04/5.0e-05), so the family-level evidence for "probe-loss selection
  beats matched random churn on quality" no longer rests on a single marginal
  test. The guided study (2) was a diagnostic, not a gate claim.
- Studies 1, 3, 5, 6, 7, 8 are mutually independent (disjoint seed blocks).
  Studies 4, 9, and the placement study share seeds by design (4 and placement
  as same-seed paired comparisons; 9 paired against 8). Study 6's descriptive
  mean comparison against study 4 is a two-sample approximation
  (z ~ -1.13), not a paired test.
- All raw records are bit-for-bit reproducible; `study_id` values are stable
  across re-runs and re-emissions.

## 4. What Gate 3 currently means

Gate 3 ("adaptive > random; if failed, stop and simplify") has now been passed
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
| Real-scale cost measurement | probe-cost amortization | cap4/1-batch (primary) + cap4/2-batch (secondary), 3 events, 24.6M tokens/arm, declared V0 285M config (see 5.1) | fresh 9000-9011 (12 pairs) | C = adaptive search/training FLOPs per arm, <= 5% (pre-registration 1's rule) + tolerance band [0.047%, 0.189%] | PASS + in-band + harm guard clear -> Gate 4 authorized; outside band -> re-derive before Gate 4; FAIL -> Gate 4 stays blocked | **PRE-REGISTERED (section 5.1), not yet run** |

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
| 2. Reproducible Kaggle notebooks | TODO (deferred until a gate passes) |
| 3. Dataset manifests | PARTIAL (synthetic-tokenizer manifests only) |
| 4. Baseline checkpoints | PARTIAL (local checkpoint format; no published artifacts) |
| 5. Adaptive checkpoints | TODO |
| 6. Architecture genomes | DONE for the gate model (serialization, hashing, fossils) |
| 7. Topology evolution logs | PARTIAL (mutation histories and fossils in records) |
| 8. Experiment registry | DONE for the gate (append-only JSONL + study summaries) |
| 9. Benchmark results | PARTIAL (this report's factor map; no real-corpus benchmarks) |
| 10. Fresh-initialization results | TODO (policies exist; studies deferred) |
| 11. Qwen architecture analysis | TODO (deferred) |
| 12. Paper-quality figures | TODO (suggested: win-rate forest plot across studies; quality-vs-probe-cost scatter; per-seed delta distributions; probe-rank vs final-rank heatmap) |

## 8. Figures (to generate)

1. **Win-rate forest plot**: one row per study (#1-#9), point estimate with
   exact binomial 95% interval, threshold line at the pass boundary.
2. **Quality vs probe cost scatter**: mean delta (x) against probe cost as % of
   training FLOPs (y), marker size = seed count, color = verdict.
3. **Per-seed paired delta distributions** for studies 1, 3, 4, 5, 6.
4. **Probe-rank vs final-rank heatmap** from the counterfactual branches.
