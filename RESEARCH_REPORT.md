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

**Execution.** The protocol runs from `T4_RUNBOOK.md` (pre-flight, session-1
throughput benchmark, the one-line launch command, queue rules, contingencies).
The verdict is machine-checked by `scripts/v0_verdict.py`, which reads the
published seed summaries and applies exactly the rules above - aggregate
adaptive cost vs the 5% criterion, the [0.5, 2]x tolerance band from
`expected_probe_cost_percent`, and the 10-of-12 harm guard - over the declared
queue 9000-9011 precisely (stray complete seeds are excluded with a warning,
not aggregated); no verdict number is hand-computed. Every branch of that
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
  batch probe-loss ranking, already weak on synthetic data (rho ~ 0.11,
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

### 5.5 Proposed revision to the 5.1 protocol (pre-launch, REQUIRES SIGN-OFF)

The substrate audit (section 9) invalidated a premise of section 5.1, so the
protocol should not launch as written. The original text above is left intact
for the audit trail; this subsection is the proposed revision, and it takes
effect only if adopted before the T4 run.

**Why a revision is needed.** Section 5.1 is a *cost* protocol whose value
depends on the controller under measurement being the validated one. Two
findings break that:

1. The registered primary controller uses `probe_candidate_policy="prefix"`
   over `feasible_expert_mutations`, which is layer-major. At the declared
   depth (12 layers x 8 experts = 108 candidates) a cap of 4 reaches only
   layer 0 and its first three prunes, so the declared search would explore
   one layer out of twelve. The V0 session now exposes `spread` for exactly
   this reason.
2. The positive quality evidence for probe selection came from a substrate
   with a demonstrated pool confound (section 9.5): at depth 4 the registered
   rule wins 88/128 while a loss-blind choice from the same candidate pool
   wins 80/128, so roughly 40% of the effect is attributable to the pool, not
   to selection. Measuring the *cost* of that controller would price a quality
   claim that is not established at any real fidelity.

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

**Explicitly unchanged.** The seed block 9000-9011, the declared V0 model
(285,336,576 / 134,341,632), the budget D = 24,576,000 tokens per arm, E = 3
events, the planned/measured test sizes, the 5% criterion, the tolerance band,
the harm guard, and the rule that no knob is re-tuned against the measured C.

**Status.** Proposed, not adopted; adopting it is a pre-registration and config
decision, **not a code change**. The registered defaults remain
`prefix`/`argmin`, so the original protocol stays runnable; the revised one is
reached by setting `probe_candidate_policy="spread"`. The pool-matched control
needs no new arm: a session's decision rule is session-level config, so the
control runs as a second session over the same seed pairs with
`probe_selection="random"`, paired against the shared baseline random arms -
exactly how `scripts/real_data_diagnostic.py` already pairs its variants. The
only genuinely new reporting surface is the interpretation clause in change
(3), which is text, not machinery.

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

### 9.7 The registered Gate 3 comparison does not survive a non-degenerate pool

The audit's two threads meet in one test: under the **revised** design
(`probe_candidate_policy="spread"`, so a cap of 4 is a real budget over 20
candidates rather than a layer-0 prefix), on a fresh block 40000-40127
(depth 4, steps=24, cap 4/1-batch, 128 seeds, ~143 s):

| Comparison | Wins | Mean delta | Binomial p |
|---|---|---|---|
| argmin vs uniform churn (**the registered Gate 3 claim**) | 65/128 | -0.0143 | 0.465 |
| loss-blind pool vs uniform churn | 63/128 | +0.0100 | 0.604 |
| **argmin vs loss-blind pool** (paired, same pool, same cost) | **76/128** | **-0.0243** | **0.0208** |

Read together with section 9.5, the picture is coherent and sharper than 9.5
alone:

- The **registered comparison fails** once the candidate pool is not
  concentrated on layer 0: argmin does not beat uniform churn (65/128,
  p = 0.465). The large depth-4 effects seen under the `prefix` policy in 9.5
  (88/128) were therefore driven by the layer-0 pool, not by probe selection -
  `prefix`/`argmin` and `prefix`/`random` both passed there (88 and 80), while
  `spread`/`argmin` does not.
- The **probe ordering itself is real**: against a loss-blind choice from the
  identical pool at identical cost, argmin wins the paired comparison 76/128
  (mean -0.0243, p = 0.021). This is the one positive, controlled result for
  the mechanism anywhere in the project.
- Neither arm beats uniform churn, and the pool's loss-blind choice is if
  anything slightly worse than uniform (+0.010, n.s.), so the benefit is
  specifically the probe's ordering, not the pool.

**Implication for Gate 3.** Gate 3 as registered ("adaptive > random churn",
per plan section 99's stop-and-simplify rule) is not reproduced on a substrate
that can express topology: the depth-1 passes (section 3) are best explained as
properties of a 5-candidate, single-layer space plus a layer-concentration
pool effect. The defensible claim is now narrower: **probe-loss ordering beats
a pool-matched loss-blind control by ~0.02 nats** (p = 0.02, one block, steps
24). Whether that increment is worth its probe cost, and whether it survives a
powered pre-registered study at a research-relevant scale, is the decision the
project now faces - not how to amortize the cost of a controller whose
registered comparison fails.

**Caveats.** Unregistered, one block, 128 seeds at steps=24, tiny depth-4
(model 24 hidden), no family-wise correction, and a claim about the *toy*
substrate - the V0 path has not been measured this way. It is reported because
it is the first controlled measurement of the mechanism and it inverts the
project's headline claim; it should be re-run pre-registered before it drives a
decision. Reproduce with `PYTHONPATH=src python scripts/toy_policy_ablation.py
--policies spread`, which recomputes these three rows from the committed code.

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

- The "identical training schedule, matched random churn" baseline (sections 1
  and 2) is only *approximately* fixed across treatments in the toy path. This
  is a second reason the toy arm comparison is weaker than the report claimed,
  independent of the pool confound. The V0 path does not share it: arms are
  trained sequentially with per-arm seeding, which is why the real-data
  diagnostic checked and found every random arm replicating its baseline
  counterpart exactly.
- The treatment-vs-treatment comparison (argmin vs pool-random, section 9.7)
  is unaffected, because both sides are treatments trained the same way.
- The `vs uniform` rows in 9.4, 9.5 and 9.7 are approximate for this reason and
  should not be the basis of a claim; the paired rows should.

Fixing the coupling would mean giving each arm its own generator for rebuild
initialisation (as the V0 session does), which changes every historical toy
number and would need to be re-registered rather than slipped in.
