# Adaptive Computational Topology — Research Report

Status: skeleton covering the narrowed first scientific milestone (Gate 3,
low-fidelity Experiment 0). Updated: 2026-09-13 (factor-map figures generated;
see section 8).

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
| Real-scale cost measurement | probe-cost amortization | cap4/1-batch (primary) + cap4/2-batch (secondary), 3 events, 24.6M tokens/arm, declared V0 285M config (see 5.1) | fresh 9000-9011 (12 pairs) | C = adaptive search/training FLOPs per arm, <= 5% (pre-registration 1's rule) + tolerance band [0.047%, 0.189%] | PASS + in-band + harm guard clear -> Gate 4 authorized; outside band -> re-derive before Gate 4; FAIL -> Gate 4 stays blocked | **PRE-REGISTERED (section 5.1), execution planned (section 5.2), not yet run** |

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
config. (4) Seed block 9000-9011 untouched; arms `random,adaptive`; probe
budgets cap-4/1-batch (primary) and cap-4/2-batch (secondary) exactly as
pre-registered. (5) Per-session `--output` summary JSONs retained in the
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
| 12. Paper-quality figures | DONE for the factor map (four figures generated by `scripts/generate_figures.py` from the versioned summaries; see section 8) |

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
   probe rank; cell (probe 1, final 1) is the argmin concordance 0.281
   (chance 0.250).
