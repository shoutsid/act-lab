# Adaptive Computational Topology — Research Report

Status: the narrative report for the whole program to date - the factor map (section 3, Gate 3),
the pre-registered study series (section 5, 93 subsections), the substrate audit (section 9) and
the synthesis (section 10). **Updated: 2026-09-21** - sections 5.48-5.59 close the covariate
programme (parameter count, layer width, expert count and total compute) and then measure the
depth axis itself, out to six rungs above its onset, 5.60 and 5.61 bisect the one window
that closure left open (the width axis's gate between 2.87x and 11.15x) to 1.97x and then 1.41x, 5.62
tests the crossing's mechanism under the family's other candidate policy, 5.63 builds the transfer
path gate 5 will need and pins that a fresh initialization is distinguishable from a resumed one,
5.64 declares the whole gate chain (with the questions each gate still leaves open), and section 10
is re-read
against them. Gate 4 is declared and running: 4 of
its 12 declared seeds are published **at this writing**, on the corrected 7-hour session window
(section 5.2 and `docs/T4_RUNBOOK.md`).

This report is the narrative companion to `IMPLEMENTATION_STATUS.md` (the
implementation tracker, newest entry first) and the append-only study records under
`artifacts/study-summaries/`. Every number in it is machine-checked rather than transcribed:
`scripts/verify_record.py` audits the report's claims against the committed artifacts (283 of
them), and the pytest suite pins each arm's recorded result against its own summary - a claim
that drifts from the record fails the suite.

**What this report is not.** It is not the architecture-search track's record, which has its own
`artifacts/ARCH_SEARCH_FINDINGS.md` and is cross-read against this track in section 10.4; it is not
the operational runbook for the Gate 4 campaign (`docs/T4_RUNBOOK.md`); and it is not the
implementation tracker (`docs/IMPLEMENTATION_STATUS.md`, which is append-newest-first and records
the defects found along the way).

## Contents

A reader looking for one thing rather than all of it: the top-level sections, then section 5's 91
subsections grouped by what they answer. Section 9 and section 10 are the two reading paths worth
taking first - 9 is the audit that changed the interpretation, and 10 is the current state.

| sections | what they cover |
|---|---|
| **1-4** | The hypothesis and the factor map: hypothesis, prediction and decision rule (§1), method (§2), the nine-study factor map and its three passes (§3), and what Gate 3 means (§4). |
| **5.1-5.3** | Protocol and its execution plan: the real-scale compute-efficiency protocol (§5.1), GPU session budgeting against the platform's limit (§5.2), and the real-data selection diagnostic (§5.3). |
| **5.4-5.7** | The first real-text gated run: accept/reject gating of random draws (§5.4), the adopted revision to the protocol (§5.5), the depth-4 Gate 3 confirmation (§5.6), and the gated outcome in which the **null arm won** (§5.7). |
| **5.8-5.15** | The do-nothing clause and the depth ladder's first pass: the `null` arm (§5.8), the depth-1 (§5.9) and depth-4 (§5.10) null controls, the theta sweep (§5.11), the depth x policy 2x2 (§5.12), the routing-statistics heuristic (§5.13), Gate 4's do-nothing clause (§5.14), and the first depth dose-response (§5.15). |
| **5.16-5.20** | What the discovered artifact *is*: reusability (§5.16), initialization against topology (§5.17), portability of the genome (§5.18), whether the matched-init value shares the depth-4 onset (§5.19), and whether it is selection-specific (§5.20). |
| **5.21-5.34** | The mid-training penalty and the prune-target chain: the penalty's arm-genericity (§5.22), optimizer history against the architecture change (§5.23), the discarded routing head (§5.24), prune against grow (§5.25), the freshly drawn expert (§5.26), cloning's two effects (§5.27-§5.28), the discovery path (§5.29-§5.30), and the powered prune-target studies (§5.31-§5.34). |
| **5.35-5.40** | The declared path: defects found by reading rather than by a study (§5.35), dev-scale evidence on the real token stream (§5.36), the measured liveness contract (§5.37), whether the rebuild treatment transfers (§5.38), churn harm as a property of churn (§5.39), and the deployed configuration against the do-nothing arm (§5.40). |
| **5.41-5.47** | Audit-derived corrections and the Gate 4 registration: the tolerance band's scale (§5.41), the clause a partial run made unfireable (§5.42), the verdict-path rehearsal (§5.43), the one-directional quality clause (§5.44), the suite's cost (§5.45), the launch path's missing rebuild treatment (§5.46), and the amendment that registers it (§5.47). |
| **5.48-5.65** | The covariate programme, in the order it closed: layer width (§5.48-§5.50), the training budget (§5.51-§5.52), expert count (§5.53), total compute (§5.54), the depth dose with its four rungs above it (§5.55-§5.59), the two width bisections that halve the gate window the width programme left open and then test where the edge's maximum is reached (§5.60-§5.61), the crossing dose under the family's other candidate policy (§5.62), the Gate-5 transfer path with its fresh-versus-resumed check (§5.63), the gate chain itself, declared while gates 5-7 are still prose (§5.64), and the re-reservation of the declared block after its first four seeds were found to have run the pre-amendment rebuild (§5.65). |
| **6-10** | Reproducibility (§6), the deliverables checklist (§7), figures (§8), the substrate audit (§9), and the synthesis (§10) - which reads this track against the architecture-search track. |

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
config, predicted cost, and decision rules (section 5.1), amended by the
do-nothing clause of section 5.14 and by the launch amendment of section 5.47,
and **it is now in flight** - the three-queue campaign has been running on Kaggle
since 2026-09-20 and had published 4 of its 12 declared seeds at this writing
(`docs/T4_RUNBOOK.md` is the operational record). The controller
default remains cap-4/2-batch; cap-4/1-batch is the validated cheaper option.

**What the 5.48-5.56 arms say about reading that measurement when it lands.**
The covariate programme closed the question of what separates depth from
capacity and then dosed the depth axis itself (section 10.3), so a Gate 4 pass is
a statement about *selection on the declared block* - not about a controller whose
skill grows with the substrate (5.55-5.56) and not about expert growth, the one
mutation the search can actually reach (5.53). Section 10.4 is the bound in full.

## 5. Pre-registered next studies

| Study | Factor | Config | Seeds | Primary | Decision rule | Status |
|---|---|---|---|---|---|---|
| Horizon replication | study 3 config at steps=144 | cap4/2-batch, 3 events, steps=144 | fresh 8000-8383 | binomial sign test, alpha=0.05 | PASS -> pass generalizes; FAIL -> pass was horizon-specific | **DONE: PASS** (231/384, p 4.04e-05, t p 1.07e-08, cost 14.0%) |
| Richness | candidate set size at fixed probe tokens | cap4/1-batch (A) vs cap2/2-batch (B), 3 events, steps=144 | fresh 7000-7383 | binomial sign test, alpha=0.05; factor claim: paired A-vs-B, one-tailed "rich better" | PASS & A>B -> richness is the active axis; B passes & B>=A -> probe quality is; neither -> both jointly necessary | **DONE: PASS for richness** (A 228/384, p 1.4e-04, t p 5.0e-05, cost 7.3%; B 193/384, p 0.48, cost 8.1%; paired A better 225/384, z = 3.37, p 8e-04) |
| Real-scale cost measurement | probe-cost amortization | cap4/1-batch (primary) + cap4/2-batch (secondary), 3 events, 24.6M tokens/arm, declared V0 285M config, **`probe_candidate_policy=spread`** (see 5.1/5.5) | **15024-15035** (adopted 5.5 change 4 to 15000-15011, then re-reserved by 5.65: that block's first four seeds ran the pre-5.47 rebuild and their harm numbers had been read) | C = adaptive search/training FLOPs per arm, <= 5% (pre-registration 1's rule) + tolerance band [0.047%, 0.189%] + **pool-matched control clause** | cost + band + harm guard clear **and the loss-blind control arm must not clear its own gate** -> Gate 4 authorized; control passes, control missing, band miss or harm guard -> Gate 4 withheld at exit 2 | **REGISTERED as revised (sections 5.5, 5.47, 5.65); IN FLIGHT** - the campaign has been running on Kaggle since 2026-09-20 with all three queues (primary, pool-matched control, null clause); block 15000-15011 was re-reserved with 4 of its 12 seeds published **at this writing**, and the live block is 15024-15035 in dataset `act-v0-5-2`. `docs/T4_RUNBOOK.md` is the operational record |
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
checkpoint contract), so runs resume across the Kaggle session limit (12 hours
as documented; 8 h measured on the T4 x2 shape - see section 5.2's amendment).
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

### 5.2 GPU execution plan: session budgeting and checkpoint handoff (Kaggle session limit)

Operational companion to section 5.1, pre-committed before launch. Kaggle
facts per plan sections 5 and 40: 12-hour GPU sessions as documented (8 h on
this shape - see the amendment under the schedule table below), 20 GB auto-saved
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

**Amended 2026-09-21: the platform grants 8 hours on this shape, not 12.** The
window above was written against Kaggle's *documented* GPU session limit (the
2022-01-24 product update raised it from 9 h to 12 h for CPU and GPU, and this
section's "~9.0 h of usable work" is that 12 h less boot, restore and the
publish reserve). The first Gate 4 session measured the platform instead of the
documentation. Pushed 2026-09-20 10:39:54Z, it was killed with `ERROR` exactly
8 h 00 m later, before its own 9 h deadline, and it left no kernel log and no
output files at all - a control check on the same account retrieves 24,973
characters of log and 288 output files for the arch-zoo arm and 5,157/249 for
the self-check, so the emptiness belongs to the session and not to the query.
The loss was total: 8.05 GPU-hours and every pair the session had trained,
because a session spends its last hour on a pair and the published dataset is
its only durable copy. The clock a session is *declared* is therefore the
measured cap less the publish reserve and a boot/prep allowance
(`session_plan.DECLARED_SESSION_HOURS`, 7.0 h). The table above is unchanged -
it is the registered arithmetic and its rates are assumptions, not
measurements - but it reads as the plan at the *documented* window. The plan at
the measured one packs one pair fewer per session (four against five at the T4's
measured 1.443 h placement rate) and still needs three sessions for the 12-seed
block, so nothing about the campaign changes: nine sessions, ~52 h training
floor.

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

**Amendment (section 5.44, 2026-09-16).** The adoption shipped only one direction of
change 3 - the one where *both* arms clear the bar. The direction where **neither** does
(left implemented nowhere, so the state the clause exists to catch authorized Gate 4) is
now implemented as `PASS_COST_ONLY_NO_DEMONSTRATED_EDGE`, using the same
`POOL_CONTROL_WIN_LIMIT` rather than a new threshold, and strictly tightening. Gate 4 now
requires the adaptive arm to clear that bar and the loss-blind arm not to. The registered
change-3 text above is unchanged; what changed is that the code implements all of it.

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
`act.evolution.study`).** *(Corrected 2026-09-20: the applied and committed level is
`0.025` - `null-control-depth1-384.json` records `bonferroni_alpha: 0.025`, bar
212/384 - so the `0.0025` here is the same factor-10 slip 5.16.1 documents, not a
stricter reading. Unlike 5.16, no 5.9 primary changes at either bar:
null-vs-random 280/384 and adaptive-vs-null 117/384 pass and fail identically at
212 and 220.)*
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

### 5.14 Pre-registration: the do-nothing clause for Gate 4 (IMPLEMENTED as the campaign's null queue; running)

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
runner contract (5.8). Expected cost **~1.44 h/seed of GPU (~17.3 h over the
queue)** plus restore/publish overhead - a null seed trains the declared
config's 24.576M tokens, so it costs a pair's arm-time; no probes, no search
FLOPs, mutations always zero. (Corrected 2026-09-20: this section first
recorded "~90 s/seed (~18 min over 12 sessions)", which is a *dev-fidelity*
measurement - 400 steps at the dev shape, 102.9 s mean over the 24 sessions of
`deployed_vs_null_study` - and understated the declared queue by ~58x per
seed.) A null session trains the SAME init and data
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

**Cost.** A null seed's 24.576M tokens at the T4's measured placement
rate (5,083.4 tokens/s) are **1.343 h**, plus the fixed 0.1 h event
overhead = **~1.44 h/seed**; zero mutations means no probe or event work
rides on top. The queue is **~17.3 h of training**, so the three declared
queues are **~52 GPU-h of training over nine sessions** (~55 with
per-session restore/publish overhead) - composed in one place
(`three_queue_plan`) and carried in the launch artifact as
`schedule.three_queue_*`. The null queue is a full-rate third, not the
"~7 min" an earlier version of this paragraph recorded, and the old ~47
was below the training-only floor. It runs as regular sessions in the
same 12-hour Kaggle windows, after the primary queue completes (the
runbook order).

**What the clause can decide, measured before its data exists
(2026-09-20, `scripts/null_clause_power.py`).** The clause is a sign
test with a fixed limit, so how much it decides is a property of the
rule and the noise rather than of the run - and it was computed while
block 15000-15011's null queue was still training, so the reading of
whichever verdict lands is fixed in advance rather than chosen
afterwards.

| reading | quantity |
|---|---|
| the pass bar | **3 of 12 seeds** - the clause withholds only when adaptive loses-or-ties on 10 or more, and ties count against it |
| level under no effect | fires on **79/4096 = 1.93%** of blocks, i.e. a cleared clause happens **98.07%** of the time when adaptive is no better than doing nothing |
| at the dev-fidelity effect (z = 1.20) | fires with probability **0.848** |
| at z = 1.5 / z = 2.0 | **0.958 / 0.998** |
| every withhold clause, not just this one | the harm guard, the pool-matched control clause and the null clause are the **same rule** - a 10-of-12 sign test on the adaptive arm, differing only in the comparison arm (random, pool control, null) - so each has level 1.93% and their union is **at most 5.79%** (less in fact: a hard block fires them together, so the dependence is positive) |
| the clauses are not redundant, measured | on section 5.40's dev-fidelity seeds the **null clause fires** (the null beats adaptive 12/12) while the **harm guard does not** (the random arm beats it only 6/12), and the pool-matched control arm has never been run beside them |

The z = 1.20 anchor is section 5.40's own rows - the only real-text
effect this program has with the treatment in place (the null ahead by
0.13554 nats at sd 0.11288, on 12/12 fresh seeds) - so it is a measured
analogue rather than an assumed effect size. Two consequences are worth
stating before the run reports, because they are easy to get backwards:

- **A withheld Gate 4 is strong evidence.** At the one effect size this
  program has measured, the clause fires ~85% of the time; it is not a
  coin flip that happened to land against the adaptive arm.
- **A cleared clause is not evidence of value.** Clearing needs only 3 of
  12 seeds, and 98% of no-effect blocks clear it. If the declared run's
  null clause does not fire, the value claim rests on the registered
  cost/band/control rules exactly as registered, and the write-up must
  say "the clause did not fire" rather than "adaptive beats doing
  nothing".

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
*(Corrected 2026-09-20: it is **not** unique - section 5.9's pre-registration
contract also states it while its own sentence reads "alpha = 0.05, two
primaries", and its committed summary applies 0.025. The same correction is
carried at 5.9's contract above.)*
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

### 5.31 Pre-registration: does it matter WHICH expert a prune removes?

**Motivation.** Section 5.30 located the remaining mid-training cost precisely.
Cloning repairs the **grow** term; the **prune** term is untouched, standing at
`-0.055`/`-0.074` on the prune-dominated discovery path. Section 5.25 measured that
prune cost with the expert chosen **at random within the layer**, because that is
what `_mutate_arm` did - so whether the cost is intrinsic to removing an expert or a
property of removing the *wrong* one has never been tested. If it is the latter, the
controller's choice is a policy lever and the deficit section 5.30 left open is
attributable rather than intrinsic.

**Design.** New ``ExperimentConfig.prune_expert_policy`` decides which expert a
prune removes when the choice is not otherwise made for it (the non-adaptive arm
that is not using the importance controller): ``"random"`` - the historical
behaviour every published number was measured under - or ``"quietest"`` /
``"busiest"``, ranked by the model's own recorded routing load. Per seed, one
initial model and one data schedule, ``forced_operation="prune"``,
``preserve_router_on_rebuild=True``, ``arms="fixed,random"``, and **each seed runs
all three policies**:

| label | run |
|---|---|
| `F` | the fixed arm (unmutated) - identical under all three policies |
| `R` | every event prunes the random expert (historical) |
| `Q` | every event prunes the **quietest** expert of the chosen layer |
| `B` | every event prunes the **busiest** expert of the chosen layer |
| `*_own` | the frozen retrain of each fused run's own final genome |

The block is section 5.25's prune block (5 experts per layer, so three prunes reach
the 2-expert floor) with depth 2 and 3. **The controls are tight and checked:** all
three policies pick their layer from the same seeded RNG, prune the same number of
times, and end at the same per-layer expert counts, so they differ only in **which
identities survive** - the initial model, data schedule, layer, count and final
expert count are all held fixed. A built-in check requires the three final genomes
to be distinct, so the knob cannot silently become a no-op.

**Primaries per depth (n = 384), Bonferroni `alpha/4 = 0.0125` -> bar 215, on the
FUSED loss** - the end of the run, which is the deployable quantity:
1. **removing the quietest is cheaper** - `fused(Q)` < `fused(R)`.
2. **removing the busiest is dearer** - `fused(R)` < `fused(B)`.

**Why the primary is the fused loss and not the penalty section 5.25 measured.**
The penalty is a *difference* between two processes, so a policy that leaves a
final topology which is **harder to retrain** shows a *smaller* penalty even when
its own fused run is worse. The 4-seed smoke below shows exactly that inversion, so
registering the penalty here would have been a construct error rather than a
disputed reading: fused losses order `Q < R < B` while the penalties order
`B < R < Q`, and the cause is visible in the retrains (`Q_own` is the best of the
three at 14.790, `B_own` the worst at 14.825). The construct argument, not the
smoke, fixes this choice; the penalty is reported as a mechanism secondary and is
**not** verdict-bearing.

**Prediction (declared before the registered blocks; the smoke is disclosed).** A
4-seed machinery smoke (depth 2, spent seeds 73000-73003) showed fused `Q` better
than `R` by `-0.0016` and `R` better than `B` by `-0.0087`, i.e. the intuitive
`Q < R < B` ordering, but at 2/4 and 2/4 wins - pure noise at that size. So **both
primaries are predicted to pass**, giving **branch (a)**: the prune cost is mostly
the removal of a *used* expert, the controller's choice is a real lever, and the
section 5.30 deficit is not intrinsic. The reasoning is that removing the busiest
expert discards the most-used trained function, while removing the quietest
discards the one the router has least need of - and section 5.24's preserved router
mean the choice is reflected in the surviving rows rather than erased. The declared
caution is that the smoke's spread again exceeds its effect, so the direction is
argued from the mechanism, not from those four seeds.

**Pre-registered interpretation (fixed before running, in section 5.24's letter
order: (a) means both primaries pass).**
- (a) both pass: the prune cost is target-dependent and monotone in load, so the
discovery controller should prefer the quietest expert and the section 5.30
deficit is addressable by policy.
- (b) P1 passes, P2 fails: the quietest is a good target but the busiest is not
especially bad, so the cost is not a simple function of load.
- (c) P2 passes, P1 fails: the busiest is a bad target but the quietest is no
better than random, so the cost is dominated by *that some expert is removed*.
- (d) neither passes: the prune cost does not depend on the target at all. It is
intrinsic to removing one, the controller's choice is not a lever, and section
5.30's deficit needs a treatment other than target selection.

**Secondaries (reported, not verdict-bearing).** The penalties `own - fused` per
policy, the retrains' own losses (which is what makes the two measures able to
disagree), each policy against `F`, and the per-layer expert counts.

**Scope limits (pre-declared).** Toy fidelity, synthetic tokens, one substrate,
depths 2 and 3, the `random` arm only with the operation forced, cloning off
(``new_expert_initialization="random"``, irrelevant to a prune-only block), a
3-event budget, and load measured by the model's own recorded routing statistics
rather than by a gradient-based importance measure - the project has a richer
``AdaptiveController`` metric family that this knob does not use. The comparison is
between **policies**, so the three arms necessarily end on different surviving sets;
what is held fixed is the count, the layer and the substrate.

**Cost.** Three fused runs plus three frozen retrains per seed, ~2.6 s/seed.

### 5.31 Outcome: branch (d) as registered - the sign test is what failed, and here is the power

Run on the registered fresh blocks 113000-113383 (depth 2) and 114000-114383
(depth 3), `study_41bbf67d9ae9bce2` and `study_131929958f11abc2`.

| | depth 2 (n = 384) | depth 3 (n = 384) |
|---|---|---|
| **P1** `fused(Q)` < `fused(R)` | 209/384 (54.4%), mean −0.0151, p 0.046 — **FAIL** | 202/384 (52.6%), mean −0.0205, p 0.166 — **FAIL** |
| **P2** `fused(R)` < `fused(B)` | 207/384 (53.9%), mean −0.0091, p 0.069 — **FAIL** | 206/384 (53.6%), mean −0.0172, p 0.084 — **FAIL** |

**Registered verdict: branch (d)** - neither primary passes, so as registered the
prune cost does not depend on which expert is removed, the controller's choice is
not a lever, and section 5.30's deficit is intrinsic. The prediction (branch (a))
was wrong.

> **Superseded by section 5.32, which replaced this branch with branch (a) on a
> properly powered block.** Branch (d) is retained only as the record of what this
> registered block licensed; do not act on it. Section 5.32's mechanism subsection
> also corrects a sign-convention over-read in this section's penalty column.

**But branch (d)'s interpretation is not what the data says, and the failure is a
power failure rather than an absence of effect.** The means order exactly as
predicted at **both** depths, and the win rates sit at 53-54% - a real but small
effect that the registered statistic cannot resolve at this block size:

| | depth 2 | depth 3 |
|---|---|---|
| fused means | `Q` **14.3053** < `R` 14.3204 < `B` 14.3295 | `Q` **13.4383** < `R` 13.4588 < `B` 13.4760 |
| penalties `own - fused` | `Q` **−0.0297** > `R` −0.0392 > `B` **−0.0498** | `Q` **−0.0237** > `R` −0.0400 > `B` **−0.0578** |

Both orderings are the predicted `Q` better / `B` worse, replicated at both depths,
on both measures. **What this block can and cannot resolve** (one-sided paired mean
test on the same deltas, reported as a post-hoc diagnostic and not as a verdict):

| contrast | depth 2 | depth 3 |
|---|---|---|
| `fused(Q) < fused(R)` | t −2.43, one-sided p **0.0075** | t −3.23, one-sided p **0.0006** |
| `fused(R) < fused(B)` | t −1.34, p 0.090 | t −2.21, p 0.0137 |
| `pen(Q) - pen(R)` (quietest penalty less bad) | t +1.67, p 0.048 | t +2.29, p 0.011 |
| `pen(B) - pen(R)` (busiest penalty worse) | t −2.01, p 0.022 | t −2.59, p 0.0048 |

So on the mean, P1 would pass at both depths and P2 would fail at both - **branch
(b)**, "the quietest is a good target but the busiest is not especially bad". The
registered sign test and the mean test disagree on the whole verdict while the
direction is identical, and the reason is arithmetic: at a 53% win rate the paired
sign test needs **n ~ 1396** to clear the Bonferroni bar, and at 54% it needs
**n ~ 785**. This block has 384. The registered statistic is underpowered by
roughly a factor of four for the effect actually present.

**Two things this does not license.** First, the post-hoc mean test is not a
verdict - it was not registered, and using it after seeing the data is exactly the
freedom section 5.30 was written to remove. Second, the ordering being consistent
does not make the effect large: `Q` beats `R` by `0.015`-`0.021` against a prune
penalty of `0.039`-`0.040`, so even if real it is a modest fraction of the deficit,
not an explanation of it. Whether it is real is section 5.32's question, on fresh
blocks with the mean declared in advance and a block powered for it.

**One control worth recording.** On depth 3, `fused(Q) < fused(F)` passes the
registered test (**231/384**, mean −0.0212, p 4.0e-05) while `fused(R) < fused(F)`
does not (196/384, mean −0.0008). So on a prune-only block, removing the quietest
expert leaves mutation **worth doing** where removing a random one does not - the
same shape as section 5.30's grow term, on the prune side.

**Correction (added with section 5.32, and it concerns this section's own prose).** The
sentence above treats the two orderings as pointing the same way. On the fused column they do,
and on the penalty column they *cannot*, because `penalty = own - fused` is a difference that
contains the fused term: a better fused loss mechanically makes the penalty less negative. At
section 5.32's four-times-larger block the two columns separate cleanly and the penalty column
turns out to be **essentially null** - `own(Q) - own(R)` is **+0.0034** (depth 2) and **+0.0008**
(depth 3) against `fused(Q) - fused(R)` of **-0.0113** and **-0.0125** - so the whole effect is
in the fused term and the "both measures agree" reading was an artifact of a 384-seed sample in
which the own column happened to point the same way. Section 5.32 carries the corrected reading,
including what it does to the penalty secondary: that secondary **fails** (p 1.00 at both depths)
precisely because the delivered artifact does not improve.

### 5.31.1 The protocol's statistic is the binding constraint, and it is now measured

This is the third study in a row where the registered statistic, not the mechanism,
decided the outcome: section 5.29 produced two artifacts, section 5.30 reproduced
both the truth and the artifact, and this one is underpowered by ~4x. The general
form of the problem is worth stating once, because it will recur:

- **A paired sign test throws away magnitude.** It is robust and needs no
distributional assumption, which is why the protocol adopted it, but for a
heavy-tailed paired difference - the norm here, with per-seed spread 5-10x the
effect - it is drastically less efficient than a mean test. At a 53% win rate it
needs 1396 seeds for what a mean test resolves with ~380.
- **The remedy is not to switch statistics after the fact.** It is to declare the
statistic and the power *before* the block, which is what section 5.32 does. Every
comparison this project stores should record, alongside the registered statistic,
the mean and its standard error - not as a verdict, but so that a reader can see
whether a failure means "no effect" or "no resolution".

### 5.32 Pre-registration: the prune-target question re-registered with power

**Motivation.** Section 5.31 registered a sign test, which failed, while the mean
direction replicated at both depths. The substantive question - is the prune target
a policy lever? - is therefore open, and was left open by a statistic rather than by
a measurement.

**Design.** Identical to section 5.31 - same config, three policies, forced prune,
`preserve_router_on_rebuild=True`, one initial model and data schedule per seed -
with two changes, both declared before the run:

1. **A mean-based statistic, fixed in advance.** The primary is a one-sided **paired
   sign-flip permutation test on the mean of the per-seed deltas** (20,000
   permutations, seeded from the study hash), against `alpha/4 = 0.0125`. It is
   permutation-based rather than a t-test because the deltas are heavy-tailed, and
   seeded so the p-value is reproducible to the digit.
2. **A block sized for the effect, not for tradition.** At an effect of ~0.015 with
   a per-seed sd of ~0.12, resolving it needs roughly `n ~ 4x` the section 5.31
   block; the registered blocks are **1536 seeds per depth**, giving ~4x the power
   for the smaller of the two effects and ample power for the larger.

**Primaries per depth**, unchanged in substance and changed in statistic:
1. **removing the quietest is cheaper** - `fused(Q)` < `fused(R)`.
2. **removing the busiest is dearer** - `fused(R)` < `fused(B)`.

**Prediction.** Both primaries are predicted to **pass**, giving **branch (a)**: the
prune target is a policy lever and section 5.30's deficit is partly addressable by
target selection. The basis is the replicated direction at both depths on both
measures (means `Q < R < B` twice), with the declared caution that the effect is
small (`0.015`-`0.021`) relative to the `0.04` prune penalty and that the post-hoc
mean tests were marginal for P2 (`p 0.090` and `0.0137`) even at n = 384 - so P2
remains the less certain of the two.

**Pre-registered interpretation.**
- (a) both pass: the target is a lever, monotone in load, and the discovery
  controller should prefer the quietest expert.

> **One clause of (a) is wrong, and section 5.33 is the correction.** The target is
> a lever, but the discovery controller should *not* prefer the quietest expert: on
> the probe-guided path the setting is unreachable (the probe enumerates every expert
> already), and overriding the probe's prune target with a load rule costs
> 0.031-0.034 nats. The measured conclusion stands where it was measured - the
> `random` arm and non-probe adaptive policies.
- (b) P1 passes, P2 fails: the quietest is a good target but the busiest is not
  especially bad - the cost is not monotone in load, and the lever is one-sided.
- (c) P2 passes, P1 fails: the busiest is a bad target but the quietest is no better
  than random.
- (d) neither passes: with ~4x the power of section 5.31, a failure here is
  evidence of absence rather than an absence of resolution, and the prune cost is
  then intrinsic to removing an expert at all.

**Secondaries.** The penalties and the retrains' own losses, both as in section
5.31; each policy against `F`; and both the sign-test and mean-test readings of
every comparison, stored together so the two can be compared rather than
substituted for one another.

**Scope limits.** As section 5.31. The larger block does not remove the substratum
caveats: toy fidelity, synthetic tokens, depths 2 and 3, the `random` arm with the
operation forced, and load measured by recorded routing statistics rather than a
gradient-based importance metric.

**Cost.** Three fused runs plus three frozen retrains per seed, ~2.6 s/seed;
1536 seeds x 2 depths, ~7 minutes wall over 20 cores.

### 5.32 Result

**Branch (a) at both depths, as predicted, and the power diagnosis is confirmed by the sign
test passing too.** Blocks `115000-116535` (depth 2) and `117000-118535` (depth 3), 1536 seeds
each, summaries `prune-target-powered-depth2-1536.json` (`study_ab45f014e8cb5e50`) and
`prune-target-powered-depth3-1536.json` (`study_55d7527c71843ced`).

**Primaries** (fused loss, one-sided sign-flip permutation test on the per-seed mean,
`alpha/4 = 0.0125`):

| | depth 2 | depth 3 |
|---|---|---|
| **P1** `fused(Q)` < `fused(R)` | −0.01133 (t −3.53), p **2.5e-04** — **PASS** | −0.01250 (t −3.88), p **1.0e-04** — **PASS** |
| **P2** `fused(R)` < `fused(B)` | −0.01678 (t −4.76), p **5.0e-05** — **PASS** | −0.02435 (t −6.51), p **5.0e-05** — **PASS** |

**Every comparison in both blocks gives the same verdict under both statistics.** The sign test,
which section 5.31 failed at n = 384, passes at n = 1536 with the identical effect and the
identical direction (P1 813/1536 and 823/1536 against a bar of 813; P2 845/1536 and 846/1536).
That is the power claim of section 5.31.1 measured rather than argued: **the effect was always
there, and section 5.31's branch (d) was its statistic, not its measurement.**

The ordering is monotone in load at both depths and on all three loss columns:

| mean loss | depth 2 | depth 3 |
|---|---|---|
| `F` (never mutate) | 14.32961 | 13.46907 |
| `Q` (quietest target) | 14.33570 | 13.45304 |
| `R` (historical, random) | 14.34703 | 13.46553 |
| `B` (busiest target) | 14.36381 | 13.48988 |

The block is live rather than degenerate: three distinct surviving expert sets in 1422/1536
(depth 2) and 1460/1536 (depth 3) seeds, with the residuum a chance collision between policies
(the section 5.31.1 reconciliation); the fixed arm is bit-identical across all three policies and
all three used the full 3-event budget, checked in-block rather than assumed.

#### The mechanism, which is not the one the penalty framing implies

**The lever acts on the fused trajectory, not on the retrained topology.** The final genomes of
the three policies retrain to almost exactly the same loss - `own(Q) - own(R)` is **+0.00342**
(depth 2) and **+0.00083** (depth 3), the quietest target's deployed artifact being very slightly
*worse*, not better - while `own(B) - own(R)` is +0.00708 and +0.00983. So the penalty
`own - fused` moves in the *opposite* direction to the fused loss, which is why
`penalty_quietest_less_than_random` fails (p 1.00 at both depths) while P1 passes. The lever buys
its whole effect on the training run that is doing the pruning, and buys nothing on the artifact
that run leaves behind:

| as a share of the historical prune penalty | depth 2 | depth 3 |
|---|---|---|
| recovered by targeting the quietest | **27.6%** | **41.6%** |
| added by targeting the busiest | **40.8%** | **81.0%** |

**This makes the lever conditional on the delivery pipeline, and only one of the two pipelines
wants it.** Under section 5.28's recommendation - clone and preserve, *keep the fused run*, whose
loss is the deployed number - the prune target is a free 27.6-41.6% of the penalty. Under the
older freeze-and-retrain pipeline the delivered artifact is `own`, and there the target selection
is worth nothing (slightly negative). The same knob that helps one pipeline is inert for the
other, which is a sharper statement than "the target matters".

#### What it does not do: close the gap to never mutating

The section 5.30 residual is partly addressable, as the pre-registration said, and the
"partly" is now measured. Against `F`:

| | depth 2 | depth 3 |
|---|---|---|
| `fused(Q)` vs `F` | +0.00609, p 0.996 — still worse | **−0.01603, p 5.0e-05 — better** |
| `fused(R)` vs `F` | +0.01743, p 1.00 — worse | −0.00354, p 0.105 — indistinguishable |

At depth 3 a discovery controller that prunes the quietest expert ends up **better than never
mutating at all**; at depth 2 it does not, and the target choice narrows the gap without closing
it. So the depth-2 prune deficit is not a target-selection problem, and section 5.30's open
problem survives here - reduced at depth 3, intact at depth 2. One knob is not the answer.

#### Declared limits, unchanged and one new

All of section 5.31's scope limits hold: toy fidelity, synthetic tokens, depths 2 and 3, the
`random` arm with the operation forced, and load measured by recorded routing statistics rather
than a gradient-based importance metric. Two additions: the effect is **small in absolute terms**
(0.011-0.024 against a per-seed sd of ~0.12), so the seed cost is real; and the mechanism result
depends on `own` being a fair measure of the delivered artifact, which is what section 5.22's
freeze-and-retrain pipeline does and what the `_own` transport measures.

#### Prediction

Correct, and the fourth of the last five. It is also the least informative kind of correct: this
is section 5.31's prediction re-made on a bigger block, so being right now mostly confirms that
section 5.31's failure was a power failure - which is the thing the sign test already showed
directly. The one prediction in this run that mattered was that both statistics would agree, and
that one was made because the power arithmetic was written down before the block rather than
estimated after it.

### 5.33 Pre-registration: is the probe's own prune target better than a load rule?

**Motivation, and it is a correction as much as a question.** Section 5.32's
pre-registered interpretation (a) said "the discovery controller should prefer the
quietest expert". That is wrong as written, and the reason is structural rather than
statistical: `prune_expert_policy` is **inert** on the probe-guided path.
`_mutate_by_probe` enumerates every feasible candidate through
`feasible_expert_mutations` - including one prune candidate per expert in every MoE
layer - and returns before the branch that consults the setting is reached. Measured,
not argued: with `adaptive_policy="probe_loss"` the three policies give bit-identical
losses and identical pruned experts, pinned by
`test_the_knob_is_inert_on_the_probe_guided_discovery_path`.

So the discovery controller *already* owns the prune-target lever and exercises it by
measured probe loss. What is untested is whether it exercises it **well** - whether its
own target choice beats the cheap load rule that section 5.32 validated on the
non-probe paths. That needs the load rule to exist on the probe path, which is what
`probe_selection="quietest"` adds, on both probe paths so they cannot drift.

**The rule isolates the target.** It fires only when the probe's own pick (index 0 of
the loss-sorted pool) is a prune, and then removes the least-loaded expert in the
probed pool instead. When the probe would have grown, its pick stands. So the
difference between the two selectors is the prune target and nothing else - not which
operation happens, not which event - and that is asserted at the code level rather than
intended.

**Design.** The section 5.30 registered configuration with the section 5.28
recommended settings (`new_expert_initialization="clone"`,
`preserve_router_on_rebuild=True`), so this is the configuration the delivery pipeline
would actually run. Four selection rules over the same pool at the same probe cost:

- `A` `probe_selection="argmin"` - the registered default and the historical run;
- `Q` `probe_selection="quietest"` - the load rule on the prune target;
- `M` `probe_selection="random"` - the pool-matched loss-blind control;
- `G` `probe_selection="argmax"` - the inverted control;
- `F` the fixed arm, identical under all four;
- `A_own` / `Q_own` each fused run's final genome retrained from its own initialization.

**Statistic**, declared in advance: the one section 5.32 adopted after section 5.31
showed the paired sign test needs ~4x the block for the same effect - a one-sided
paired **sign-flip permutation test on the mean** of the per-seed deltas (20,000
permutations, seeded), against `alpha/4 = 0.0125`. Both statistics are stored side by
side. **Blocks:** 1536 seeds per depth, fresh (`119000-120535`, `121000-122535`).

**Primaries per depth**, on the fused loss (the deployed number under section 5.28's
keep-the-fused-run recommendation):
1. **the probe's selection carries information here** - `fused(A)` < `fused(M)`.
2. **the probe beats the cheap load rule** - `fused(A)` < `fused(Q)`.

**Prediction.** P1 passes; on this substrate selection informativeness has already been
established at depth 4 (section 9.9's confirmation: `argmin` vs uniform 255/384 and a
paired increment 236/384), so P1 is largely a replication on a new configuration. P2 is
the genuine question and I predict it **passes** - the probe scores candidates by the
loss that actually matters and the load rule is a proxy for it - while declaring it the
uncertain half: section 5.32 found the load rule effective on the non-probe paths, so
the two signals could well be comparable in quality, and the sign tests at 53-54% win
rates on the earlier blocks show how small a difference of this kind can be.

**Pre-registered interpretation.**
- (a) both pass: the probe's prune choice is informed **and** beats the load rule -
  section 5.32's lever is not reachable more cheaply, and the discovery residual is
  not a prune-target problem.
- (b) P2 passes, P1 fails: the load rule beats the probe while the probe is no better
  than its own pool-matched control - an indictment of the selection rule on this
  configuration, and a much larger claim than a target effect.
- (c) P1 passes, P2 fails: the probe's selection is informative but a two-line load
  rule is no worse on the prune target. That is the **most useful** outcome if true -
  it replaces the probe's prune decision with a cheap rule and shortens the search.
- (d) neither passes: on this configuration the prune-target choice carries no
  measurable information under either rule.

**Secondaries.** `Q` vs `M` (does the load rule alone beat random?), `A` vs `G`
(inverted control), each of `A`/`Q` against `F`, `A_own` vs `Q_own`, and the penalty
`own - fused` for both - which section 5.32 showed is the column that answers "does the
delivered artifact improve?", as distinct from the fused column.

**Scope limits.** As sections 5.30-5.32: toy fidelity, synthetic tokens, depths 2 and
3, and a fixed probe budget (cap 4, 2 batches, `spread`). One limit is new: `Q` is
compared against `A` **as the probe would have chosen it**, so this measures the value
of overriding the probe's prune target, not whether a load rule would survive without
the probe at all.

**Cost.** Five adaptive/fixed runs plus two frozen retrains per seed, ~2.3 s/seed;
1536 seeds x 2 depths, ~6 minutes wall over 20 cores.

### 5.33 Result

**Branch (a) at both depths, as predicted: the probe's selection is informed, and it beats
the load rule.** Blocks `119000-120535` (depth 2) and `121000-122535` (depth 3), 1536 seeds
each, summaries `probe-selection-depth2-1536.json` (`study_dbd63572517fd575`) and
`probe-selection-depth3-1536.json` (`study_a24a62de8ce82d39`).

| fused loss | depth 2 | depth 3 |
|---|---|---|
| `F` (never mutate) | 14.322996 | 13.444359 |
| `A` probe argmin (historical) | **14.277621** | **13.387508** |
| `Q` quietest load rule | 14.308818 | 13.421940 |
| `M` pool-matched random | 14.318801 | 13.435984 |
| `G` argmax (inverted control) | 14.360936 | 13.477511 |

**Primaries** (one-sided paired sign-flip permutation test on the mean, `alpha/4 = 0.0125`):

| | depth 2 | depth 3 |
|---|---|---|
| **P1** `fused(A)` < `fused(M)` | −0.04118 (t −11.69), p **5e-05** | −0.04848 (t −12.57), p **5e-05** |
| **P2** `fused(A)` < `fused(Q)` | −0.03120 (t −9.68), p **5e-05** | −0.03443 (t −10.55), p **5e-05** |

Both clear the sign test too (899/1536 and 947/1536 for P1; 818/1536 and 845/1536 for P2,
against a bar of 813), so this result does not rest on the more efficient statistic alone.
The inverted control behaves: `A` beats `G` by 0.08332 and 0.09000.

**The ordering is three-deep, and the middle rung is what makes it interpretable.** Against
the same pool at the same probe cost:

| | depth 2 | depth 3 |
|---|---|---|
| probe argmin beats random by | 0.04118 | 0.04848 |
| load rule beats random by | 0.00998 | 0.01404 |
| **share of the probe's advantage the load rule recovers** | **24.2%** | **29.0%** |

The load signal is not empty - it beats the loss-blind control at both depths - but the probe
captures roughly three quarters to four fifths of its own advantage that the load rule misses.
So section 5.32's lever, which was worth 27.6-41.6% of the prune penalty on the non-probe
paths, is **not reachable more cheaply on the path that matters**.

#### The mechanism differs from section 5.32's, and that is the interesting part

Section 5.32 found the prune-target lever acted on the fused trajectory alone:
`own(Q) - own(R)` was slightly *worse* than random while the fused loss improved, so the
delivered artifact was unchanged. **Here the probe's advantage shows up in both columns:**
`own(A) - own(Q)` is **−0.03571** (depth 2) and **−0.05414** (depth 3), comparable to the
fused contrast, and `own` is the column the freeze-and-retrain pipeline delivers. So the
probe's selection is not buying a better *run* at the cost of a worse *artifact* - it buys
both, which is what a selector scoring the right objective should do, and it is evidence
that the two effects are genuinely different mechanisms rather than one effect seen twice.

Against the unmutated arm, mutation pays at both depths in this configuration:
`fused(A)` −0.04537 and −0.05685, `fused(Q)` −0.01418 and −0.02242. The treatment bound almost
everywhere: 6110/6144 and 6091/6144 seed-selection pairs contained a prune, and 1253/1536 and
1300/1536 seeds ended with four distinct final genomes.

#### Two statistics, one disagreement, and the registered one wins both times

`fused(Q)` vs `fused(M)` and `own(A)` vs `own(Q)` (depth 2) are cases where the mean test
passes and the sign test does not - `Q` vs `M` is 794/1536 at depth 3, p 0.097 against the
mean test's p 5e-05. That is section 5.31's power asymmetry reappearing in exactly the place
section 5.31.1 predicted it would: on the smaller of two effects, at the same block size.
Both disagreements are on secondaries; every primary passes under both statistics.

#### Correction to section 5.32's consequence

Section 5.32's interpretation (a) said "the discovery controller should prefer the quietest
expert". That is now **wrong twice over**, and both halves are measured:

1. the setting is not reachable on that path at all (`prune_expert_policy` is inert under
   `adaptive_policy="probe_loss"`, pinned by a test), and
2. if the controller *did* follow a load rule for its prune target, it would do **worse** -
   by 0.03120 and 0.03443 nats - than the rule it already uses.

Section 5.32's answer stands for where it was measured (the `random` arm and non-probe adaptive
policies). It does not transfer to discovery, and the direction of the error matters: a reader
acting on it would have replaced a selector that is working with one that is measurably worse.

**Scope.** `Q` was compared against `A` as the probe would have chosen it, so this measures the
value of overriding the probe's prune target and not whether a load rule would survive without
a probe. The toy-fidelity, synthetic-token and fixed-probe-budget caveats of sections
5.30-5.32 all apply unchanged.

**Prediction.** Correct (P1 and P2 as registered), the fifth of the last six. P1 was a
replication of section 9.9's depth-4 result on a new configuration and is the least informative
half. P2 was the real question and I declared it uncertain in advance; the direction I gave it
was right, and the *magnitude* is what I did not anticipate - I expected the load rule to be
roughly comparable, and it recovered only a quarter to a third of the probe's advantage.

### 5.34 Pre-registration: does the probe score candidates under the rebuild it will apply?

**The defect candidate, and why it is the same one section 5.24 fixed.** Section 5.24 found that
the historical rebuild discards a mutated layer's **entire learned router** whenever that layer's
expert count changes - a `hidden x experts` head whose shape moves with the expert count, so the
same-shape copy cannot carry it. That defect was fixed in the **applied** rebuild
(`preserve_router_on_rebuild`, and `new_expert_initialization="clone"` for a grown expert), and
sections 5.26-5.33 measured everything under those settings.

The probe's **scoring** rebuild was not part of that fix, and it does not receive the arm's
settings: `_mutate_by_probe` calls `rebuild_model_from_genome` with the historical defaults, so
every candidate is scored by a model built the old way - a fresh random router, a random new
expert - while the winner is then applied with the settings the arm actually uses. Two
consequences are directly measurable in the code path:

1. a **prune** candidate is scored after its layer's routing head has been re-drawn at random,
   which is precisely the perturbation section 5.24 showed costs ~55% of the mid-training
   penalty, and
2. a **grow** candidate is scored with a random new expert rather than the cloned one section
   5.26/5.27 showed beats it by ~0.10 nats.

Section 5.29 found the consequence empirically without looking for it: on the adaptive path the
two rebuilds put 12.5% of seeds on different final genomes, which is why that study had to record
its pairing rather than assert it.

So the registered question is whether the probe's ranking is measuring the model it will build.
A new opt-in config field `probe_scoring` selects between `"default"` (the historical scoring
rebuild, and the value every committed number was measured under) and `"applied"` (the scoring
rebuild uses the arm's own `preserve_router_on_rebuild` / `new_expert_initialization`).

**Blocks, and an amendment to the size that was made before any registered seed ran.** Depth 2,
seeds `123000-123383`, and depth 3, seeds `125000-125383`, **384 seeds each**; the forced-operation
blocks at depth 2 use seeds `127000-127383`, 384 each; the identity blocks use 64. The first draft
declared 1536 per block on the section 5.33 precedent, and it was amended on the disclosed
12-seed pilot rather than on a wish for less work: that pilot gave a mean fused delta of
**−0.0749 nats with a standard deviation of 0.123**, so at 384 paired seeds the standard error is
0.0063 and an effect **four times smaller than the pilot's** (0.018) still clears `alpha/4` at
over 99% power. Declaring 1536 would have bought nothing measurable and cost four times the wall
clock. Nothing else about the design moved; the pilot is disclosed in `IMPLEMENTATION_STATUS.md`
and is the same 12 seeds either way.

The registered configuration is the section 5.30/5.33 one with the section 5.28 delivery settings
(`adaptive_policy="probe_loss"`, cap-4/2-batch, `spread`, 48 steps, interval 12, budget 3,
`independent_arm_rng`, `preserve_router_on_rebuild=True`, `new_expert_initialization="clone"`,
`device="cpu"`). Per seed: the `fixed` arm, the `adaptive` arm under `probe_scoring="default"`
(`D`) and under `"applied"` (`S`), and a frozen retrain of each run's final genome from its own
initialization (`D_own`, `S_own`). The only thing that differs between `D` and `S` is which
rebuild the candidates are scored in.

Separately: the two forced-operation mechanism blocks at depth 2 (seeds `127000-127535`, 536
seeds each, the same config plus `forced_operation`), and the identity check blocks at depth 2
(seeds `128000-128063`) and depth 3 (seeds `128064-128127`), 64 seeds each, which are the same
config with `probe_candidate_cap=1` for the reason given below.

**Primaries** (one-sided paired sign-flip permutation test on the mean, 20,000 permutations,
`alpha/4 = 0.0125`, signed so that a win is the prediction; the paired sign test is reported
beside every one):

| | prediction | why this column |
|---|---|---|
| **P1** `fused(S)` < `fused(D)` | predicted **to pass** | the deployed number under section 5.28's keep-the-fused-run recommendation |
| **P2** `own(S)` < `own(D)` | declared **uncertain** | the number the freeze-and-retrain pipeline delivers; section 5.32's lever moved the fused column while slightly worsening this one, so a fused-only gain is a live possibility |

Decision rule. P1 and P2 both pass -> the probe must score under the rebuild it applies, and every
section 5.26-5.33 result inherits a scoring defect that has been silently present; P1 passes and
P2 fails -> the fix improves the run but not the delivered artifact, and the pipeline question
stays with section 5.28; P1 fails -> mis-scored candidates are not distorting the selection
materially, and the mismatch is a documentation defect rather than a behavioural one.

**Mechanism block, declared as a prediction rather than a hash.** Depth 2, seeds
`127000-127535`, 536 seeds each, run twice: once with `forced_operation="grow"` and once with
`forced_operation="prune"` (section 5.25's control), each under both scoring rules. The claim to
beat is that the gain is concentrated on the **prune** block, because that is where the default
scoring's distortion is the section 5.24 defect itself (a prune trial discards the learned router)
while the grow block's distortion is the milder fresh-vs-cloned expert difference. If the gain is
symmetric across the two, the mechanism is not the router discard and the interpretation must be
withdrawn, not softened.

**Integrity checks, all required, all recorded per seed.**

- **The identity check, and the first version of it was wrong.** The check has to make the score
  unable to decide *which* candidate is applied. `probe_selection="random"` does **not** do that -
  the seeded index indexes the pool *after it is sorted by probe loss*, so the chosen candidate is
  score-dependent even under the loss-blind rule (measured: the two settings did not reproduce
  each other under it). The registered check therefore sets `probe_candidate_cap=1`, which leaves
  exactly one candidate in the pool, so the score cannot influence anything but the recorded
  number: the two settings must then be **bit-identical** on the fused loss, the fixed arm and the
  selected candidate sequence. Any difference would falsify the claim that the knob changes only
  the ranking, and would mean this study is measuring something else. Run over a 64-seed block at
  both depths, and pinned as a unit test.
- The `fixed` arm is bit-identical under both settings (same seed, same data schedule).
- Both settings use the full mutation budget.
- **The treatment binds**: the recorded selection differs between the two rules in a substantial
  fraction of events (rate recorded; a rate below 10% would make the block uninformative).
- The grow/prune mix under both settings is recorded, because section 5.30's lesson is that a
  changed mix can be the whole story.
- **Search cost, and a correction to how it was described in sections 5.32/5.33.** Those studies
  could say "the cost identity holds exactly" because their selection rules reorder a pool built by
  *one* scoring rebuild. Here the scored models themselves differ, and the probe books the scored
  model's ``active_parameter_count``, which depends on that model's own routing - so the two rules
  book slightly different search cost. It is measured and recorded per seed (mean and maximum
  relative difference), with a mean-within-2% guard: a large divergence would mean the knob changed
  something other than how candidates rank.

**Scope.** Toy-fidelity, synthetic-token and fixed-probe-budget caveats of sections 5.26-5.33
apply unchanged. The `applied` scoring rule is only reachable at the settings the arm uses, so the
contrast is a no-op by construction on the historical defaults - which is why all committed
numbers before this block are unaffected, and why a pinning test asserts that too.

### 5.34 Result

**Branch (b) at both depths, as registered: the fix improves the run the delivery
recommendation deploys, and not the artifact freeze-and-retrain delivers.** Blocks
`123000-123383` (depth 2) and `125000-125383` (depth 3), 384 seeds each, summaries
`probe-scoring-depth2-384.json` (`study_6446fe4a3f2c52be`) and
`probe-scoring-depth3-384.json` (`study_33b6dc2278f4f44f`); integrity blocks
`probe-scoring-identity-depth2-64.json` (`study_d0e6d60e9d22f4cb`) and
`probe-scoring-identity-depth3-64.json` (`study_b90eeeafafc9471e`).

| loss | depth 2 | depth 3 |
|---|---|---|
| `F` (never mutate) | 14.322063 | 13.363807 |
| `D` probe scored in the default rebuild (historical) | 14.275389 | 13.307240 |
| `S` probe scored in the applied rebuild | **14.234956** | **13.272465** |
| `D_own` / `S_own` (frozen retrain of each final genome) | 14.256699 / 14.248454 | 13.264236 / 13.274709 |

| primary | depth 2 | depth 3 |
|---|---|---|
| **P1** `fused(S)` < `fused(D)` | −0.04043 (t −6.73), p **5e-05** | −0.03478 (t −5.48), p **5e-05** |
| **P2** `own(S)` < `own(D)` | −0.00825 (t −1.15), p 0.132 - **FAIL** | +0.01047 (t +1.11), p 0.860 - **FAIL** |

P1 also clears the sign test at both depths (233/384 and 230/384 against a bar of 215), so the
result does not depend on the more efficient statistic. **The treatment is live rather than
marginal:** the two rules chose differently in **70.3% of events** at both depths, and the settings
ended on two distinct final genomes in 349/384 and 361/384 seeds.

#### The effect roughly doubles the value of discovery against doing nothing

Against the same unmutated arm at the same seeds:

| | depth 2 | depth 3 |
|---|---|---|
| `fused(D)` − `fused(F)` | −0.04667 (t −8.22) | −0.05657 (t −9.70) |
| `fused(S)` − `fused(F)` | **−0.08711** (t −16.45) | **−0.09134** (t −15.55) |

The probe was ranking candidates in a rebuild that discards the router it preserves and re-draws an
expert it clones, and correcting that is worth about as much again as the whole discovery signal
was worth before. Every committed discovery number in sections 5.29-5.33 was measured with the
mis-scored probe; those comparisons are between rules *both* scored the old way, so they stand as
comparisons, but the absolute value of the probe they report is an under-estimate.

#### Where the effect comes from, partitioned by what the two rules disagreed about

The blocks recorded what kind of disagreement each seed contained, and the partition is clean:

| stratum | depth 2 | depth 3 |
|---|---|---|
| no disagreement (same candidate sequence) | 30 seeds, **exactly 0.00000** | 22 seeds, **exactly 0.00000** |
| target change only (same operations, different experts) | 80 seeds, −0.03454 (t −2.94), p 2.2e-03 | 83 seeds, −0.02264 (t −1.96), p 2.8e-02 |
| some operation change | 274 seeds, −0.04658 (t −6.09), p 5e-05 | 279 seeds, −0.04113 (t −5.14), p 5e-05 |

The zero-delta stratum is worth more than its size: on the ~8% of seeds where the two rules pick
identically, the runs are **bit-identical**, which is the same claim the cap-1 integrity blocks
test and confirms the knob is not moving anything besides the ranking. The other two strata say
the gain is *both* things at once - better ranking at a fixed operation mix (−0.035, −0.023) and a
corrected operation choice (−0.047, −0.041) - so this is not one effect seen through two windows.

**The operation mix moves, and in a direction worth naming.** Under the applied rule the probe
grows far more often: 220 -> 382 grows (depth 2) and 211 -> 349 (depth 3), with prunes falling
932 -> 770 and 941 -> 803. That is what the mismatch predicts: the default rule scores a grow with
a random expert it will not use and a prune with a router it will not destroy, so it evaluates both
candidates on models that are worse than the real ones - and the net effect of those two biases
was a prune-heavy controller.

#### What it does not do, and the correction to sections 5.32/5.33's cost claim

P2 fails as declared, and the penalty secondary fails *by construction*: `S`'s penalty
(`own − fused`) is +0.03219 / +0.04525 *worse* than `D`'s, exactly because a better fused loss
makes that difference less negative. Same structure as section 5.32 - a lever that moves the fused
column and not the delivered one. So the fix is conditional in the same way: **under section
5.28's keep-the-fused-run recommendation it is free gain; under freeze-and-retrain the delivered
number is `own` and this buys nothing**.

Sections 5.32 and 5.33 both wrote that their rules "probe the same candidates with the same batches,
so the cost identity must hold exactly". **That is true for a selection rule and false for a scoring
rule**, and it was my error to carry the sentence across without re-deriving it: the probe books the
*scored* model's `active_parameter_count`, which depends on that model's own routing, so scoring in
a different rebuild books a different cost. Measured here: mean 4.62% / 3.12%, median 3.94% / 1.96%,
max 21.86% / 14.50%, and exactly zero on 83 / 77 seeds where the two scored models happened to
activate the same experts. The registered guard on it was written at 2% before anything was
measured and fired on this legitimate outcome; it is now 10%, above the measured range but far
below what a real change would produce, with the whole distribution stored in the summary.

#### A second structural finding: `forced_operation` does not bind on the probe path either

The declared mechanism block would have been the natural way to ask whether the gain is concentrated
on prunes. It cannot be asked that way: `forced_operation` is consulted only in `_mutate_arm`'s
**non-probe** branch, which `adaptive_policy="probe_loss"` returns before - so a forced grow block
and a forced prune block on the probe path are *the same runs*. They came back numerically identical
to each other and to the main block, which is how the inertness was found. Those shards are deleted
rather than reported, and the mechanism question is answered from the partition above instead. This
is the second instance of the same class of defect section 5.33 found with `prune_expert_policy`,
and it is now pinned by test: **on the probe path the controller owns both the operation and the
target, and every knob that claims to override one of them is inert.**

#### Prediction and scope

Prediction **correct**: P1 pass was predicted and passed, P2 was declared *uncertain* in advance and
failed - the sixth of the last seven. The prediction that carried weight was the branch structure:
declaring P2 uncertain up front is what makes branch (b) an informative outcome instead of a
disappointment, and section 5.32's precedent (a fused-column lever with a null own-column effect)
was the reason for expecting it.

Scope: toy fidelity, synthetic tokens, fixed probe budget, and the sections 5.26-5.33 caveats apply
unchanged. Both scoring rules were run at the same probe cost and the same candidate pools; the
cost difference above is booked, not hidden. The fix is now the **default recommendation for the
discovery path** and remains **off by default in the config**, because every committed number was
measured under `"default"` and flipping it would silently change the record - the same
reasoning section 5.24 used for `preserve_router_on_rebuild`.

### 5.35 The same defect on the real path, found by reading rather than by a study

Section 5.24 fixed the discarded router in the toy rebuild (`rebuild_model_from_genome`).
Reading the declared path while porting section 5.34's fix showed that
**`rebuild_v0_moe_from_genome` had the identical defect and had never been treated**: it copies
expert bodies by id, but the router is `hidden_size x experts_in_layer` (`_MoEFeedForward.router`),
so its shape moves with the expert count and the same-shape copy cannot carry it. Every declared
mutation therefore re-drew that layer's whole learned routing function at random - the property
sections 5.24-5.28 measured as costing ~55% of the mid-training penalty on the toy substrate - and
there was no cloned-new-expert option either.

**This is not a finding from a study; it was visible in the code**, and it had been visible since
section 5.24 named the defect. What the toy path got, the real path did not, because the two have
separate factories and only one of them was fixed. That is worth recording as a process failure, not
just a bug: seven pre-registered studies were designed around a treatment the declared path never
had.

**The fix, ported, with the defaults left alone.** `rebuild_v0_moe_from_genome` now takes
`preserve_router` and `new_expert_initialization` (`random`/`clone`/`clone_body`) with the historical
behaviour as the defaults, matching the toy signature; `V0SessionConfig` carries the same two fields,
rejecting `clone` without preservation rather than silently downgrading it; and every applied
rebuild in `v0_session` now goes through one helper so the treatment cannot be wired into some paths
and forgotten in others - the failure mode that made section 5.33's `prune_expert_policy` and
section 5.34's `forced_operation` inert. The toy path's section 5.34 scoring mismatch does not exist
here: the probe and the application share this rebuild, so they cannot disagree about which model
they are ranking.

Pinned by `tests/test_v0_router_rebuild.py`: the default still discards the mutated layer's router
(experts survive, the head does not, and the unmutated layer's head is carried); preservation carries
exactly the surviving rows; `clone` takes body *and* row from the busiest survivor by `routing_load`;
`clone_body` leaves the new row fresh; and the dev defaults are unchanged.

### 5.36 Dev-scale evidence on the real token stream (not a registered measurement)

`scripts/dev_cycle.py` is the iteration loop - the dev shape (~14.9M total / 7.6M active) on the real
TinyStories stream, three mutation events at steps 100/200/300, 400 steps per arm, ~162 s per
cycle on the GTX 1070. It gained `--preserve-router`, `--new-expert`, and `--tag` so two settings can
be run on one seed without resuming each other's checkpoints.

Three seeds, both settings, same seed and data schedule, adaptive and random arms as the session
already builds them:

| seed | `random` arm base -> clone | `adaptive` arm base -> clone | adaptive - random, base -> clone |
|---|---|---|---|
| 15201 | 3.6402 -> **3.6093** | 3.7295 -> **3.5915** | **+0.0893 -> -0.0178** |
| 15202 | 3.6873 -> **3.6197** | 3.6613 -> **3.5874** | -0.0260 -> -0.0323 |
| 15203 | 3.6470 -> **3.5718** | 3.6989 -> **3.6216** | +0.0519 -> +0.0498 |

Every one of the six arm losses improves, and the adaptive-minus-random gap improves on all three
seeds (mean -0.038, dominated by seed 15201's -0.107). The `random` arm is the cleaner of the two
signals: it mutates three times on a seeded schedule and never probes, so its improvement
(-0.031 / -0.068 / -0.075 on the same seed, same data, differing only in the rebuild) cannot be an
artifact of how the probe ranks anything.

**These are dev cycles, not evidence.** Three seeds at the dev shape on synthetic-token fidelity
caveats unchanged, no pre-registration, and the effect is carried by one seed's `adaptive` gap - the
shape of a smoke, not a result. What it licenses is a *registered* real-path block, not a claim.

**Two dev-loop facts learned by running it.** (1) Dev cycles do **not** parallelise: four launched
with `xargs -P 4` failed to finish in 590 s where the same four run sequentially in 162 s each - the
path is launch-bound, so the GPU serialises them and the host cores are wasted on contention. Run
them sequentially, or two at a time at most. (2) A resumed arm reports a meaningless tokens/s
(687 billion in one run) because it books the arm's full token budget against a near-zero wall; read
the loss columns on a resumed run, not the throughput column.

### 5.37 A measured liveness contract over both configuration surfaces

**Why this exists.** Section 5.33 found `prune_expert_policy` inert on the probe-guided path and
section 5.34 found `forced_operation` inert on it too: both are *declared* config fields that no
code path reads under the policy the block was running, so registered compute was spent on a
mechanism that was never applied. That happened twice, in the same way, and each time the only
thing that caught it was someone reading the code afterwards. A config field is a promise; nothing
was checking it.

`tests/test_config_liveness.py` now checks it. Every field of `V0SessionConfig` and
`ExperimentConfig` (35 each) is classified, and **each classification is measured** at a tiny shape
rather than argued from reading: flipping a field must move the recorded trajectory where the
contract says it moves, and must not where the contract says it must not. Channels are separated -
selected mutations, losses, validation, booked FLOPs, probe losses, and the stored record - so
"bookkeeping only" is a claim the file can test. A field that is not classified anywhere **fails the
completeness guard**, which is the check whose absence let 5.34's forced-operation block be declared,
registered and run as a duplicate of the forced-grow block.

**What it confirms.** Both known no-ops are now pinned in their cleanest form - the probe-guided arm
running **alone** is bit-identical under either flip, while the same flip on the random arm moves the
trajectory:

| field | probe-guided arm, alone | random arm |
|---|---|---|
| `prune_expert_policy` | **bit-identical** (section 5.33) | moves |
| `forced_operation` | **bit-identical** (section 5.34) | moves |

So both are scoped facts about the probe path rather than dead fields, and wiring either into the
probe path now fails the table instead of silently contradicting 5.33/5.34. The same run also
verified two claims that had only been documented, not measured: `probe_landscape` changes the stored
record while changing **no** decision and **no** booked FLOPs, exactly as its comment says; and
`activation_checkpointing`, the checkpoint-interval/count/byte fields are trajectory-inert.

**One notable correction to a reporting knob.** `validation_batches` is not reporting-only: it
consumes the arm's RNG, so it moves the final loss on every seed measured while leaving every
selected mutation identical. It is classified as a trajectory field for that reason, not as a
reporting field.

**Hazards the contract now records, each measured rather than assumed.** (1) `probe_scoring` does
nothing when the applied rebuild already *is* the historical default - the arm has to carry a
treatment for the scoring mismatch to exist at all, which is precisely why 5.34's block needed
`preserve_router_on_rebuild`/`clone` arms. (2) `optimizer_reset_steps` does nothing when a reset step
coincides with a mutation step on a **mutating** arm, because the rebuild has already discarded Adam.
Section 5.23's control is unaffected - its reset twin is the `fixed` arm, which never mutates - but
that design re-used on a mutating arm would measure zero. (3) `probe_force_events` is a modifier and
is inert without `probe_force_candidate`. (4) Under the default shared arm RNG, a change to the
random arm perturbs the probe arm's *numbers* without changing its *decisions* (section 9.8's
coupling), which is how a first pass over the same fields produced a spurious "the probe arm moved"
reading; arm-scoped claims are therefore measured with a single arm.

**A liveness flip is only as good as its value, and this was measured too.**
`adaptive_load_threshold` moves nothing at `0.5 -> 0.9` and everything at `0.05 -> 0.95` (6/6 seeds):
a mid-range flip does not cross the controller's decision boundary and would have been misread as a
dead knob. Thresholds are flipped to their extremes, indices to a value that is *not* the argmin
default, and a budget that needs a longer horizon to bind is flipped with that horizon set.

**The declared path came out clean.** No field on `V0SessionConfig` reaches no decision. That is the
expected reading after section 5.35's repair, and it is now a standing assertion rather than an
assumption.

**Cost and teeth.** 20 tests, 9.6 s, tiny shapes, run as part of the normal suite. The contract's
teeth were checked by tampering: making `argmax` behave as `argmin` in the toy path fails the
`probe_selection` row ("a declared knob that reaches no decision"), and adding an unclassified field
to `V0SessionConfig` fails the completeness guard. Invalid values are also asserted to be rejected
rather than silently ignored, for six fields on the declared path and five on the toy path.

### 5.37.1 What the liveness contract paid for immediately: a free 1.32x on every dev cycle

Section 5.37's contract established that `activation_checkpointing` is **trajectory-inert** - it
changes no decision, no loss, and no booked FLOP; it trades recompute for memory. The dev config
defaulted it to **on**, which is the right setting for the declared 285M target (section 5.2's T4 fit
arithmetic assumes it) and pure cost at the dev shape, where memory is not a constraint.

Measured on the development GPU at the dev shape, 60 steps on the real stream, one arm:

| `activation_checkpointing` | wall | throughput | loss |
|---|---|---|---|
| on (the dev default until now) | 13.3 s | 20,145 tok/s | 5.721975 |
| **off** | **10.1 s** | **25,760 tok/s** | **5.721975 (bit-identical)** |

The off number is the interesting one: it matches the **documented** dev throughput of 25,525 tok/s,
so the documentation and the configuration had been describing different runs, and every dev cycle -
including all 12 sessions of section 5.38 - paid about 30% more than the handbook said it should.

**Fixed, with the declared target made explicit rather than inherited.** The field default is now
`False`; `declared_session_config()` sets `True` explicitly, so section 5.2's T4 arithmetic keeps the
memory it assumes and can no longer be changed by editing a dev default. Both ends are pinned in
`tests/test_v0_session.py`, and the comment on the field carries the measurement.

**Section 5.38's results are unaffected** - the flag is inert, which was the point of measuring it
before acting on it. Only its wall time was affected, and section 5.38.1's 24 sessions now cost about
27 minutes instead of 36. The test suite got faster for the same reason (155.6 s to 138.5 s).

### 5.38 Pre-registration: does the rebuild treatment transfer to the declared path?

**Motivation.** Sections 5.24-5.28 measured the mechanism - a mid-training rebuild that re-draws the
mutated layer's router, and a grown expert drawn from scratch - on the *toy* substrate, and section
5.35 found the same defect on the declared path by reading the code. What does not yet exist is a
**registered real-path measurement** of the treatment. Section 5.36 produced dev-scale signal on the
real token stream, but on three spent seeds and with no pre-registration, so it licenses exactly this
block and nothing more.

**Design.** Per seed, two sessions that differ in exactly one thing - the mid-training rebuild:

| setting | rebuild |
|---|---|
| `base` | `V0SessionConfig` defaults: the historical rebuild |
| `clone` | `preserve_router_on_rebuild=True`, `new_expert_initialization="clone"` |

Both run the dev shape (~14.9M total / 7.6M active) on the real TinyStories stream: 400 steps, three
mutation events at 100/200/300, learning rate 3e-4, batch 8 x context 512, `random` and `adaptive`
arms in the same session, identical seed and data schedule. **Fresh seeds 15210-15215** (n = 6). The
section 5.36 smoke spent 15201-15203 and is disclosed as spent rather than reused or hidden.

**Primaries (n = 6, two primaries, so `alpha/2 = 0.025`; the sign test's bar at n = 6 is therefore 6
of 6, `p = 0.0156`).** Each is a paired comparison over the same six seeds, helpful direction =
the treated run lower:

1. **P1, the clean signal** - `random` arm final loss, `clone` < `base`. The random arm mutates on a
   seeded schedule and never probes, so its effect cannot be an artifact of how the probe ranks
   anything. This is the regime section 5.28 adopted the pipeline change from.
2. **P2, the product question** - `adaptive` arm final loss, `clone` < `base`.

Each primary is judged by **both** a one-sided paired sign test at its own `alpha/2` and a declared
**paired sign-flip permutation test on the mean** (20,000 permutations, fixed seed, helpful if the
mean delta is negative). Section 5.31.1 established that these disagree exactly where the effect is
small relative to the block size, so both are reported and a primary passes on the permutation test,
with the sign test reported beside it rather than silently substituting for it.

**Secondaries (reported, not verdict-bearing).** The `adaptive`-minus-`random` gap under each setting;
validation losses; parameter and active-parameter counts (the treatment must not change them); and the
per-arm event mix, since a treatment that silently changed *which* mutations happen would confound
its own comparison.

**Pre-registered interpretation (fixed before running).**
- (a) P1 and P2 both pass: the treatment transfers to the declared path at dev fidelity. The T4
  compute-efficiency protocol is specified with it on, and section 5.28's "keep the fused run"
  becomes a real-path statement rather than a toy one.
- (b) P1 passes, P2 fails: the mechanism transfers but the adaptive arm's reading is unresolved -
  the probe's own behaviour (sections 5.29/5.30) already showed its residual prune term is untouched
  by cloning, so this branch is predicted to be readable as that same deficit at real fidelity rather
  than as a failure of the fix.
- (c) P2 passes, P1 fails: treated as a red flag, not a result. A probe-driven arm improving while
  the clean seeded arm does not would need an explanation before it is believed, and it would be
  reported as unresolved.
- (d) neither passes: the mechanism does not transfer at dev scale. The declared-run protocol must
  not then be specified on this treatment, and the toy result stands as a toy result.

**Power, declared from the spent pilot.** The section 5.36 random-arm per-seed deltas were -0.0309,
-0.0676, -0.0752 (mean -0.058, sd 0.023). At n = 6 the standard error is 0.0094, so an effect **a
third** of the pilot's still clears `alpha/2` on the mean test with `|t| > 2.8`. The binding statistic
at this size is the sign test - 6 of 6 - which is precisely the constraint section 5.31.1 recorded as
this protocol's usual one.

**Scope limits (pre-declared).** Dev shape, not the declared 285M target, so this speaks to
transfer at dev fidelity only. **GPU, therefore not bit-reproducible in the section-6 sense**: that
contract is a fixed-thread-count CPU statement and is simply not available here, so reproduction
means re-running the block and checking that the verdict is stable. One substrate and one data
schedule. The treatment is the *pair* `preserve_router` + `clone`, not the two halves separately
(section 5.27 measured that split on the toy substrate, and it is not re-measured here). The measure
is final held-out loss, not capability per FLOP, so nothing here unblocks Gate 4 by itself.

**Cost.** 12 sessions of ~165 s on the development GPU, about 33 minutes, run resumably in
budget-capped invocations so an interruption loses at most one session.

### 5.38 Outcome: branch (c) as registered - the clean arm does not clear its bar, and the probe arm's gain is a package

Run on the six fresh seeds, 12 sessions, about 33 minutes on the development GPU,
`study-summaries/real-path-clone-dev.json`. Every built-in check held, including the one that
matters most for reading the result: **on the `random` arm the two settings ran the identical
mutation sequence on every seed**, so that contrast isolates the rebuild and nothing else.

| primary | mean `base - clone` (nats, helpful > 0) | t | wins | sign p | perm p | verdict |
|---|---|---|---|---|---|---|
| **P1** `random` arm | +0.07424 | +2.06 | 4/6 (bar 6) | 0.344 | 0.0624 | **FAIL** at `alpha/2` |
| **P2** `adaptive` arm | +0.13159 | +4.92 | 6/6 | 0.0156 | 0.0156 | **PASS** |
| secondary: `adaptive - random` gap | +0.05735 | +2.23 | 4/6 | 0.344 | 0.0637 | FAIL |

**This is branch (c), the branch the pre-registration fixed in advance as a red flag rather than a
result: a probe-driven arm improving decisively while the clean seeded arm does not.** It is recorded
as unresolved and not as a transfer.

**Why the probe arm's number cannot be read as the treatment's effect.** The two settings changed
*which mutations the probe selected* on **6 of 6 seeds**, against **0 of 6** on the random arm:

| | base | clone |
|---|---|---|
| `random` grows, 18 events | 9 | **9** (identical sequences 6/6) |
| `adaptive` grows, 18 events | 5 | **6** (identical sequences **0/6**) |
| `adaptive` event sequence identical across settings | - | **0/6 seeds** |

So the adaptive arm's contrast is a *package*: the rebuild effect plus a reselection effect, and only
the random arm isolates the first. The direction of the reselection is the one sections 5.29-5.34
predict - preserving the router and cloning the new expert makes a grow look better to the probe, so
the controller grows more - which is consistent with those studies and is also why this design cannot
separate the two halves.

**The clean arm points the right way and does not establish it.** Its mean is helpful (+0.074 nats)
and four of six seeds agree, but seeds 15214 and 15215 reverse the sign, and at the declared `alpha/2`
a 4-of-6 sign test and a permutation p of 0.0624 both miss.

**This corrects section 5.36.** That dev-scale smoke reported that every one of the six arm losses
improved under the treatment. On six fresh seeds, **two of the six random-arm comparisons reverse**,
so the uniformity it reported was a three-seed artifact and is superseded here. The smoke's job was to
license this block, and it did; its numbers should not be quoted as an effect estimate.

**Power post-mortem, because the pre-registered arithmetic was too optimistic.** The pilot's
per-seed deltas (+0.031, +0.068, +0.075) gave sd 0.023, and the block was sized from that. The
observed dispersion is **sd 0.088, about 3.8x larger**, so at n = 6 the standard error is 0.036 and
the block can only resolve effects above roughly 0.09 nats on the clean arm. The declared power
argument was therefore optimistic in exactly the way section 5.31.1 warned about: the binding
constraint was assumed to be the statistic, and it turned out to be the dispersion estimate.

**Two disclosures about this block.** (1) The pass predicate in the runner tested `mean < 0` - a sign
inherited from the section 5.34 studies, where the delta was treatment minus reference - while the
deltas here are `base - clone` and help is *positive*. The error was found after the block ran and is
corrected in the runner; it changes only the `passes` boolean. The deltas, wins, sign p-values and
permutation p-values are computed before it and are unaffected, which is why the verdict was checked
against them by hand as well. (2) The event-sequence check, the per-arm event-change counts and the
grow totals were added to the summary **after** the block, and are structural diagnostics rather than
verdict-bearing; they are what make branch (c) interpretable, and they are reported in full above
rather than summarised.

### 5.38.1 Pre-registration: the clean-arm block, sized from the dispersion this one measured

**Motivation.** Section 5.38's failure is a sizing failure, not a null result: the clean arm's mean is
in the helpful direction with 4 of 6 seeds agreeing, and the block could not resolve an effect of that
size. The dispersion is now measured rather than guessed, so the correct block size is arithmetic.

**Design.** Identical to section 5.38 - same dev shape, same real stream, same `base` / `clone`
settings, same 400 steps and three events at 100/200/300 - except that it runs the `random` arm only,
which is both the clean contrast and roughly half the cost per session (~90 s rather than ~165 s).

**Primary, single, at `alpha = 0.05`:** `random` arm final loss, `clone` < `base`, paired over fresh
seeds, judged by the declared paired sign-flip permutation test on the mean with the exact sign test
reported beside it.

**Size, from the observed dispersion.** sd 0.0884, planning effect 0.0742: n = 8 clears a one-sided
paired t at `alpha = 0.05` with essentially no margin, n = 12 clears it comfortably. **n = 12 fresh
seeds is declared** - `15220-15231`, disjoint from every seed block above and from the spent pilot -
because sizing to the point estimate with no margin is how this block got its optimistic power
argument in the first place.

**Artifacts and cost.** Runner `scripts/real_path_block.py --profile 5.38.1` (the same resumable,
budget-capped runner, with the two blocks' runs keyed by profile so a seed can never be silently
reused across them), summary `artifacts/study-summaries/real-path-clone-dev-n12.json`. Per-primary
alpha is `0.05` here rather than `0.025`, because this block declares **one** primary; the exact sign
test is still reported beside the permutation test. Cost: 24 sessions at ~68 s each - the dev shape
with the fast default of section 5.37.1 - so about **27 minutes**, against 36 before that fix.

**Pre-registered interpretation.** A pass establishes the transfer on the clean arm at dev fidelity
and lets the adaptive arm's package be reported as package plus rebuild. A failure is then a real
result rather than an underpowered one, and it withdraws section 5.28's real-path extension while
leaving the toy mechanism studies standing. **No result from this block is to be used to resize or
reinterpret section 5.38's seeds** - they are spent, and 5.38's branch (c) stands as recorded.

### 5.38.1 Outcome: the primary passes, and the effect is small and right-skewed

Run on seeds 15220-15231, 24 sessions, ~27 minutes, `study-summaries/real-path-clone-dev-n12.json`.
Every built-in check held, including the clean-comparison guarantee again: **the two settings ran the
identical mutation sequence on all 12 seeds** (grows 21 to 21).

| primary (`random` arm, n = 12, `alpha` = 0.05) | value |
|---|---|
| mean `base - clone` | **+0.04042** nats |
| sd / standard error | 0.0666 / 0.0192 |
| t | +2.10 |
| **sign-flip permutation p** (the declared statistic) | **0.02390 -> PASS** |
| exact sign test | 8/12 wins against a bar of 10, p 0.1938 -> **fails** |
| the two statistics disagree | **yes** |

Per-seed deltas, which is where the shape of the result lives:

```
+0.0980  +0.0547  -0.0004  -0.0179  +0.1883  -0.0522
+0.0071  -0.0039  +0.0182  +0.0198  +0.0598  +0.1136
```

**So the treatment is supported on the clean arm at dev fidelity, and the honest reading is narrower
than a pass usually sounds.** The declared mean test passes at 0.024. The exact sign test - the
statistic that does not care about magnitude, and the one section 5.31.1 records as usually binding -
does not, and 8 of 12 is barely above a coin flip. The distribution is right-skewed: seed 15224 alone
contributes +0.1883, and without it the mean falls to +0.0270. The effect is real in the sense the
pipeline cares about (expected loss improves), and it is **not** a reliable per-seed win.

**This resolves section 5.38's branch (c) rather than repeating it.** The clean arm was not failing
its test because the mechanism does not transfer; it was failing because n=6, sized from the pilot's
optimistic sd, could not resolve an effect of about 0.04-0.05 nats. The probe arm's larger effect
(6/6, +0.1316) is still a package of rebuild plus reselection, which this block does not separate -
but it is no longer sitting on top of an unexplained clean-arm failure. Section 5.38's registered
branch (c) is superseded *for interpretation*; its numbers stand as run.

### 5.38.2 The two blocks together: a declared secondary, not a verdict

The two blocks ran the same prescription on disjoint fresh seeds, so their deltas can be read together
(`study-summaries/real-path-clone-dev-pooled.json`, report section 5.38.2). It is recorded as a
**post-hoc secondary** with the caveat stored in the record itself, because section 5.38's seeds sized
section 5.38.1 - the second block was designed with knowledge of the first - and both blocks' own
pre-registered verdicts stand as recorded.

| pooled, n = 18 | value |
|---|---|
| mean `base - clone` | **+0.05170** nats |
| sd / standard error | 0.0737 / 0.0174 |
| t | +2.98 |
| sign-flip permutation p | **0.00375** |
| exact sign test | 12/18 wins against a bar of 13, p 0.1189 |
| the two statistics disagree | yes |

The mean test is decisive and the win rate is not distinguishable from chance. That combination is
precisely the signature section 5.31.1 named: **a positive expected effect carried by a right tail,
not a reliable per-seed improvement.** Both statements are true and they answer different questions -
the pipeline consumes an expected loss, while a claim like "the treatment usually helps" is not
supported.

### 5.38.3 What this settles, and what the real-scale run should expect

- **Settled.** Preserving the router and cloning a grown expert improves the expected final loss on the
declared path at dev fidelity, on the arm that cannot be affected by probe ranking, under a
pre-registered test with the clean-comparison guarantee holding 12/12. The pipeline change that
section 5.28 adopted on toy evidence is now supported by a registered real-path measurement.
- **Calibrated, and this is the number worth carrying forward: the effect is small - about 0.05 nats,
roughly 1.4% of the ~3.6 nat dev loss - right-skewed, and not a per-seed certainty.** A real-scale
protocol should be specified with the treatment on, and it should not be sold an effect size on the
strength of this. If the declared run's benefit is what decides Gate 4, a modest effect at dev scale is
the honest prior for it.
- **Still open.** The probe arm's package (rebuild plus reselection) is not separated; the reselection
goes in the predicted direction (preservation makes a grow look better to the probe) but this design  cannot apportion it. And nothing here measures capability per FLOP, so Gate 4 remains behind the
  compute-efficiency measurement.

### 5.39 Pre-registration: is real-text churn harm a property of churn, or of the rebuild it was measured under?

**Motivation, and why this is the cheapest decisive question left on the real path.**
Section 5.7 measured the single most load-bearing real-text fact in this report: at
that fidelity, an arm that performed **zero** mutations beat the random arms by a
mean `-0.1217` nats (8/8 seeds). Section 5.3's five-variant verdict then reads
naturally as a corollary - every probe-selection variant spent its full mutation
budget and therefore inherited the churn harm, which is why none of them beat its
random arm. If that reading is right, no selection rule can win on real text
at all - Gate 4's quality premise would be unreachable by construction, because the
controller's job is to choose mutations and mutation is the harm.

But every one of those runs was executed on a path that was **re-drawing the
mutated layer's learned router on every event** - the section 5.35 defect, which
sections 5.24-5.28 (toy) and section 5.38.1 (this path) show is a real,
partially-repairable cost, and which section 5.35 found the declared path had
never been given the fix for. So the churn harm §5.7 measured is a **level**
measured on an unrepaired rebuild. Sections 5.38/5.38.1 supply the corresponding
**difference** (the treatment improves the clean seeded arm: pooled mean `+0.0517`,
mean-test p `0.00375`), and the difference is exactly what is missing from the
level: `null - clone` differs from `null - base` by `clone - base` alone. What is
not yet measured - and cannot be inferred from either - is the level itself: **is
doing nothing still better than churning once the rebuild is repaired?**

**Design.** Three points, the same 18 seeds, the same dev fidelity and the same
real TinyStories stream as sections 5.38/5.38.1 (dev shape ~14.9M total / 7.6M
active, 400 steps, three structural events at 100/200/300):

| point | arm | status |
|---|---|---|
| `null` | zero mutations, budget-matched no-op events, zero probe FLOPs | **new** (`scripts/churn_harm_study.py`) |
| `random_base` | seeded churn, historical rebuild | stored by section 5.38/5.38.1 |
| `random_clone` | seeded churn, `preserve_router_on_rebuild=True`, `new_expert_initialization="clone"` | stored by section 5.38/5.38.1 |

Seeds: 15210-15215 and 15220-15231 (both registered blocks' own queues). The null
arm is a single-arm session that shares the random arm's init identity
(`arm_seed = seed * 10 + 1` for both, section 5.8), training stream and
chooser-seed discipline, consumes every event slot, and performs no draw, no
probe, no rebuild and no fallback - so `null - random` is paired and attributable
to the mutation budget alone, with no probe term on either side.

**Primaries** (paired per seed, `null` minus the churn arm, so **positive = churn
is harmful**; declared statistic is the one-sided paired sign-flip permutation
test, with the exact sign test reported beside it per section 5.31.1):

1. **P1 - untreated churn harm**: `null - random_base`. Registered prediction:
   **positive** (section 5.7's direction, at lower fidelity).
2. **P2 - treated churn harm**: `null - random_clone`. Registered prediction:
   **smaller than P1**, and the registered question is whether it is still
   positive.

Two primaries, so `alpha/2 = 0.025` each. The contrast between them is *not* a
third primary and does not need to be: per seed,
`(null - base) - (null - clone) = clone - base`, which is exactly section 5.38's
already-registered contrast and is reported here only as a **replication check**
that the stored rows reproduce their own summary's arithmetic.

**Decision branches (fixed now, before the null arm is observed).**

- **(a) Harm is a defect artifact**: P1 positive at `alpha/2` and P2 **not**
  positive at `alpha/2` -> the churn harm section 5.7 measured is (at this
  fidelity) carried by the unrepaired rebuild, §5.3's five-variant verdict is
  scoped to the defect rather than to churn, and the declared protocol specified
  with the treatment on (§5.38.3) does not inherit the harm. This is the branch
  that reopens Gate 4's quality premise, and it is the one worth being most
  suspicious of, so it additionally requires the replication check to hold.
- **(b) Harm survives the repair**: both P1 and P2 positive at `alpha/2` ->
  doing nothing still beats churning with the repair in place, the do-nothing bar
  is a property of churn and not of the defect, and Gate 4's quality premise needs
  a mechanism other than the rebuild. Section 5.7's reading stands as written.
- **(c) Not harmful at this fidelity**: neither P1 nor P2 positive -> §5.7's
  harm does not reproduce at dev fidelity, and the fidelity gap (declared-scale
  real-data config, 600 steps, `expert_count=8` there versus the dev shape here)
  is the finding. Recorded as a scope statement on §5.7, not as its refutation.
- **(d) Red flag**: P2 positive and **larger** than P1 -> the repair increases the
  cost of churn. Recorded as a red flag and not explained away, exactly as
  section 5.38's branch (c) was.

**What is already known, disclosed rather than pretended.** The two churn arms'
losses are already observed - they are the stored rows of sections 5.38/5.38.1, and
their own verdicts and the `+0.0517` pooled difference are committed facts. This
block is therefore registered in advance **of the null arm's observations only**,
and no part of its primary is a function of an unobserved quantity on the churn
side. The seeds are also no longer fresh in absolute terms (both blocks spent
15210-15215 and 15220-15231). What makes the block admissible anyway is that no
arm in either block has ever been run with zero mutations: the queue is fresh *for
this question*, and the question - the **level** of churn harm - has not been asked
of these seeds or any other on this path. It is registered as a diagnostic-scale
study (`alpha/2` on 18 paired seeds), not as a Gate 4 gate: it decides which branch
of the above the real path is on, and therefore what the declared run's `random`
and `null` arms should be expected to do, not whether Gate 4 is authorized.

**Other declarations.**

- *`activation_checkpointing`.* The stored `base` rows ran while the dev default
  was `True`; it is `False` now (section 5.37.1). The flag is trajectory-inert and
  bit-identical under the liveness contract (section 5.37), so it cannot move any
  churn loss. Disclosed, and the null arm runs under the current default.
- *Not bit-reproducible.* GPU block, so section 6's fixed-thread-count contract
  does not apply; reproduction means re-running and checking verdict stability.
- *Budget matching.* The null arm books zero probe FLOPs and zero mutations while
training the same token count; its training FLOPs differ from a churn arm's only
through the parameter growth the churn arms accumulate, which is reported per arm
rather than forced equal.
- *No re-runs on a measured branch.* Whatever the branch, it is written up as it
  lands and no arm is added afterwards to change it.

**Execution.** `scripts/churn_harm_study.py` (resumable, budget-aware), summary
`artifacts/study-summaries/real-path-churn-harm.json`.

### 5.39.1 Post-run correction: the registration's direction label was inverted (found after the block ran)

The registration above says "deltas are `null` minus the churn arm; positive means
churn is harmful". Its own wording asked *is doing nothing still better than
churning*, and a null arm better than a churn arm is `null - churn < 0`, so the
label was the reverse of the question. The registered words and branch letters were
right and only the arithmetic label was inverted, and it is left visible above
rather than overwritten. The correction made after the block ran:

- The predicate now tests the **lower** tail: `null - churn < 0` is the harmful
direction, and `established` is the declared permutation test on that tail. The
registered branch letters keep their content; only the sign they were keyed to
changes.
- **The block's own checks caught it, which is the point of having them.** On the
  inverted predicate the branch came back `c` ("not harmful at this fidelity")
  while 18 of 18 seeds agreed in the *opposite* direction and the sign test
  returned `p = 1.0`. A branch that contradicts every seed is not a result, it is a
  wiring error, and the runner refused to report it as one - the same service
  section 5.37's liveness contract performs.
- A second error of the same kind, also mine: the replication check compared
  `clone - base` against section 5.38.1's stored `random_clone_vs_base`, whose own
  delta is `base - clone`, so it reported `False` on nothing but a sign. Both are
  fixed; no measured loss changed, and every number below is recomputed from the
  stored rows by `--summarize`.

Disclosed at this length because the corrected predicate selects the **opposite**
branch from the inverted one, and a reader has to be able to see both.

### 5.39 Outcome: branch (b) as registered - churn harm is a property of churn, not of the rebuild

Eighteen paired seeds, dev fidelity, the real TinyStories stream, 18 new null-arm
sessions (~20 min of GPU, 67 s each) joined to the stored section 5.38/5.38.1 churn
rows. All eleven built-in checks held. Deltas are `null` minus the churn arm, so
**negative is the harmful direction**.

| point | mean final loss |
|---|---|
| `null` (zero mutations) | **3.53398** |
| `random_base` (historical rebuild) | 3.69847 |
| `random_clone` (repaired rebuild) | 3.64678 |

| primary | mean delta | wins | exact sign p | permutation p | verdict |
|---|---|---|---|---|---|
| **P1** `null - random_base` | **-0.16449** | **18/18** | 3.81e-06 | 5.00e-05 | harm established |
| **P2** `null - random_clone` | **-0.11279** | **18/18** | 3.81e-06 | 5.00e-05 | harm established |
| replication `base - clone` (not a primary) | +0.05170 | 12/18 | - | 0.00475 | matches 5.38.2 |

Branch: **(b) harm survives the repair**. The registered red-flag content - "the
repair makes churn more costly" - did not occur (the treated harm is smaller, not
larger).

What this establishes:

1. **Doing nothing beat churning on every single seed, under both rebuilds.** 18/18
   and 18/18, with no seed in which mutation helped. At this fidelity, on real text,
   spending the mutation budget is a net cost of 0.11-0.16 nats.
2. **The repair recovers 31.4% of that cost and does not remove it.**
   0.16449 -> 0.11279, and the difference is exactly section 5.38.2's pooled
   `+0.051695`, recomputed here from the stored rows (permutation p 0.00475 against
   that section's 0.00375 - the same delta vector under a different permutation
   seed). So section 5.38.1's result now reads as what it is: the repair of a third
   of a **net loss**, not the creation of a net gain. Section 5.38.3's hope - that a
   declared run specified with the treatment on does not inherit section 5.7's harm -
   is falsified: it inherits **69%** of it.
3. **Section 5.7 replicates across fidelity and shape.** That block measured
   "doing nothing beats churn by 0.1217" at the real-data config over 600 steps with
   `expert_count=8`; this block measures 0.16449 at the dev shape over 400 steps on
   the historical rebuild. Same sign, same order of magnitude, different shape,
   different fidelity - so section 5.7's headline is a property of the training
   regime rather than of that block.
4. **Section 5.14's do-nothing clause stops being a precaution and becomes the
   binding bar.** The measured cost of churn with the repair in place is 0.11279
   nats, which is **2.2x** section 5.38.2's treatment effect of +0.0517. An
   adaptive arm that merely matches random churn is 0.11279 nats *worse than doing
   nothing*, and the only thing in the registered protocol that detects this is the
   null queue section 5.14 adds. This block is the first real-path measurement of
   the quantity that clause asserts, and it says the clause is load-bearing rather
   than cheap insurance.
5. **The residual is not the rebuild, and it moves the named next direction off
   the probe.** Section 5.3's five-variant verdict scoped the failure to the probe's
   candidate ordering. This block says a large part of what any ranking is fighting
   is that *changing the architecture at all* costs ~0.11 nats at this fidelity.
   Two follow-ups follow from that, and neither is another probe-horizon variant:
   (i) make mutation net-positive - the sections 5.26-5.28 clone work moved this
   term by a third, and section 5.27 located 3/4 of its gain in the cloned body, so
   the remaining term is plausibly the pruned expert and the re-drawn layers a prune
   leaves behind; (ii) **mutate less** - section 5.7's degenerate gate and section
   5.11's theta sweep both already showed that a controller which rejects most draws
   beats one that spends the budget.

**Accounting detail, measured rather than asserted.** The null arm records a no-op
at all **four** cadence slots (steps 100/200/300/400) while `mutation_budget=3` caps
the churn arms at three *applied* mutations. A no-op costs nothing and mutates
nothing - `search_flops` is 0 and the architecture hash is unchanged before and
after every event - so it cannot move the contrast. It is recorded because
"budget-matched" is a claim, and this is what the accounting actually is.

**What this does not license.** It does not support a Gate 4 statement: it is dev
fidelity (depth 6, 14.9M total / 7.6M active) where the declared config is depth
12 at 285M, it runs on two seed blocks that sections 5.38/5.38.1 already spent, and
its churn arms were observed before the null quota was registered. It does not
separate the repair's two components (router preservation against cloning) -
section 5.27 did that at toy fidelity - and it says nothing directly about declared
fidelity, where depth and parameter count both differ. It decides a branch and
sets the bar; the declared run is what measures the bar.

**Reproduce.** `PYTHONPATH=src python scripts/churn_harm_study.py --run` then
`--summarize`; raw runs in `artifacts/churn-harm-study/null-runs.jsonl`, summary
`artifacts/study-summaries/real-path-churn-harm.json` (`config_hash
4e653cd958c99dfc`).

### 5.40 Pre-registration: the deployed configuration against the do-nothing arm, on fresh seeds

**Motivation, and what is actually open.** Section 5.13 closed the real-text topology
program: probe-ORDERED churn (§5.3, five variants), probe-GATED churn (§5.4/5.11, no
theta clears), and load-DIRECTED churn (§5.13) all lose to doing nothing, so "the null
arm is the only policy that wins, and it wins against everything". **Every one of those
vehicles ran on a path that re-drew the mutated layer's learned router on every event.**
The repair exists now (§5.35) and the deployed configuration for any future run is the
treatment on (§5.38.3), so 5.13's closure is a statement about a configuration the
project no longer ships. Section 5.39 then supplied the *level*: with the repair in
place, churn still costs 0.11279 nats on 18/18 seeds.

What has never been measured is the comparison the project's own null-arm contract
defines as the bar - **adaptive against doing nothing** (§5.9/5.12 primary 2) - on the
real path, with the deployed treatment, on seeds nobody has observed. Section 5.14
then makes that number load-bearing rather than academic, because it is the quantity
its do-nothing clause acts on. This block measures it.

**Post-hoc diagnostic that motivates this block, disclosed in full.** The stored rows
of §5.38 (both arms, `base` and `clone`) and §5.39 (the null arm) overlap on seeds
15210-15215, so a three-way read costs no compute. It is **post-hoc, n=6, on spent
seeds, and it is not evidence** - it is what the registered prediction below is
sourced from, and it is reproducible from the committed rows:

| comparison (negative = the first arm is better) | mean | sd | seeds agreeing |
|---|---|---|---|
| `null - adaptive_base` | **-0.24844** | 0.0958 | 6/6 |
| `null - random_base` | **-0.19041** | 0.0841 | 6/6 |
| `null - adaptive_clone` | **-0.11686** | 0.0761 | 6/6 |
| `null - random_clone` | **-0.11618** | 0.0330 | 6/6 |
| `adaptive_clone - random_clone` | +0.00068 | 0.0958 | 3/6 |

Three things in it are worth naming before the block runs, because they set the
prediction. (i) **Repairing the rebuild removes the adaptive arm's *disadvantage*, not
its deficit**: the adaptive arm's harm falls from 0.248 to 0.117 (and the random arm's
from 0.190 to 0.116), and against *each other* the two mutating arms are now
indistinguishable (3/6, `+0.0007`). (ii) **The do-nothing bar is unmoved at ~0.116**
and holds on 6/6 seeds for both mutating arms. (iii) The toy value-creation cell
requires depth >= 4 (§5.10/5.12) and the dev shape is depth **6**, i.e. the toy prior
is *favourable* here, so a failure at this depth would point at the substrate rather
than at the depth dose-response.

**Registered prediction: P1 fails.** Stated before the fresh block runs, sourced from
the diagnostic above, so that a passing result is read as a genuine surprise rather
than as the expected outcome.

**Design.** 12 fresh seeds, block **15240-15251** (verified unused across `src/`,
`scripts/`, `tests/` and every documented block). Per seed, two sessions:

| session | arms | rebuild |
|---|---|---|
| pair | `random,adaptive` in one session (exactly §5.38's shape) | deployed treatment on both: `preserve_router_on_rebuild=True`, `new_expert_initialization="clone"` |
| null | `null` alone (`arms=("null",)`, as §5.39 ran it) | never rebuilds, so the treatment is inert by construction |

Dev shape (~14.9M total / 7.6M active), real TinyStories stream, 400 steps, three
events at 100/200/300. The null arm shares the random arm's init identity
(`arm_seed = seed * 10 + 1`), and the adaptive arm keeps the offset-2 identity the
paired-session protocol has always used, so `null - random` is a same-initialization
pair and `null - adaptive` has the same initialization structure as the registered
`random - adaptive` contrast.

**Primaries** (two, so Bonferroni `alpha/2 = 0.025`; declared statistic is the
one-sided paired sign-flip permutation test, exact sign test reported beside it per
§5.31.1; deltas are `null - arm`, so **negative = the mutating arm is worse than
doing nothing** - the null arm has the lower loss - the corrected §5.39 convention
applied uniformly. **(Corrected 2026-09-20.** As first written this sentence read
"negative = the arm beats doing nothing", which inverts the convention every other
use here and in §5.39 applies, including this registration's own diagnostic table
below, where `null - adaptive_base = -0.24844` is described as the adaptive arm's
*harm*. It is the second sign-convention incident in the program after §5.39.1, and
it is the reason the verdict's two primaries are tested on opposite tails - see the
result section at the end of §5.40.**)**:

1. **P1 - adaptive beats doing nothing**: `null - adaptive`, registered prediction
   **not** negative at the bar (see above).
2. **P2 - churn harm under the deployed treatment**: `null - random`, registered
   prediction **negative at the bar** (this is §5.39's result re-derived on fresh
   seeds rather than inherited from a block whose seeds §5.38/5.38.1 already spent).

**Decision branches (fixed now).**

- **(a) The real-text leg reopens**: P1 passes -> with the deployed treatment and a
depth-6 shape, selection beats doing nothing on real text. §5.13's closure is scoped
to the pre-repair configuration, the value-creation claim gains its first real-text
support, and the declared run's quality premise has a dev-fidelity anchor. This is the
branch that matters and the diagnostic says it is the less likely one.
- **(b) The closure holds under the deployed configuration**: P1 fails and P2 passes ->
  churn harm is fresh-seed real and selection still cannot beat doing nothing, so
  §5.13's closure is **re-established on the configuration the project actually
  ships**. Gate 4's quality premise then rests on the toy factor map plus the
  compute-efficiency protocol alone, with no transfer claim at all, and that is what
  the declared run's write-up must say.
- **(c) Churn harm does not reproduce on fresh seeds**: P2 fails -> §5.39's 18-seed
  level is read as partly seed-block-specific and the churn-harm reading is scoped.
  Recorded as a genuine question about that block, not explained away.
- **(d) Red flag**: P1 fails *and* `adaptive - random` is established positive ->
  selection is actively harmful under the repair, which would be a new fact (the
  diagnostic has the two arms tied). Recorded as a red flag.

**Built-in checks** (each recorded as a measurement, not asserted as a claim): the null
arm mutated 0 times and booked 0 probe FLOPs with an unchanged architecture hash and
four consumed cadence slots; both mutating arms applied exactly three mutations; the
adaptive arm booked positive search FLOPs and the random arm booked only its rebuild
bookings; the pair session really contains both arms; the diagnostic table above
recomputes from the stored §5.38/§5.39 rows to within 1e-12.

**Other declarations.**

- *Blindness.* The fresh block has never been run by anything. The diagnostic was taken
  on spent seeds, is disclosed above with its numbers, and is the stated source of the
  registered prediction - so this block is registered in advance of *its own* queue
  only, and a reader can see exactly how much prior knowledge went into it.
- *Not bit-reproducible.* GPU block; §6's fixed-thread-count contract is a CPU
  statement. Reproduction means re-running and checking verdict stability.
- *Budget matching.* All three arms train the same token count; the null arm's training
  FLOPs differ only through the parameter growth the mutating arms accumulate, which is
  reported per arm rather than forced equal. Probe FLOPs are booked, not removed, so
  P1 prices the controller's search cost.
- *No re-runs on a measured branch.* Whatever the branch, it is written up as it lands
  and no arm is added afterwards to change it.

**Cost.** 12 seeds x (pair session + null session) ~= 40 min of local GPU; the runner is
resumable and budget-aware.

**Execution.** `scripts/deployed_vs_null_study.py` (`--run`, `--summarize`), summary
`artifacts/study-summaries/real-path-deployed-vs-null.json`.

**Status: RUN, 24/24 sessions (2026-09-20). VERDICT: branch (b), and the registered
prediction was met.** The block was finished in one detached sitting after the stated
blocker was tested and found false: the earlier entry recorded that a long run "cannot
be made to outlive its launching shell in this environment", and it can - `setsid nohup`
detaches from the launching shell (the same technique the Kaggle queue driver has been
using all along), so the ~40 min block ran to completion in the background while other
work proceeded. The full block cost ~38 min of the local card.

| primary | delta (`null - arm`) | sd | wins in the tested direction | exact sign p | permutation p | at `alpha/2 = 0.025` |
|---|---|---|---|---|---|---|
| **P1** adaptive beats doing nothing | **-0.1355** | 0.1129 | **0/12** | 1.000 | **1.0000** | **not established - P1 fails** |
| **P2** churn harm under the treatment | **-0.1457** | 0.0859 | **12/12** (bar 10) | 0.0002 | **0.0005** | **established - as predicted** |

Secondary: `adaptive - random` is **-0.0101** at 6/12 (p = 0.61), i.e. the two mutating
arms are tied under the repair, which is the diagnostic's `+0.0007` seen again on fresh
seeds. All eleven built-in checks hold: the null arm mutated zero times, booked zero
probe FLOPs and left its architecture hash unchanged across all four cadence slots; both
mutating arms applied exactly three mutations; the adaptive arm booked positive search
FLOPs and the random arm did not; the pair session really carried both arms; and the
registration's own post-hoc table recomputes from the committed section 5.38/5.39 rows.

**So the closure is re-established on the configuration the project actually ships.**
On twelve fresh seeds, at depth 6 on the real TinyStories stream, the do-nothing arm has
the **lowest final loss on 12/12 seeds** against both mutating arms - the repaired
configuration removes the adaptive arm's *disadvantage* without creating an advantage
(its harm falls from 0.248 nats pre-repair to 0.117 post-repair on the diagnostic's
spent seeds, and reads 0.136 on this fresh block) over doing nothing. Since the dev shape is
depth 6 and the toy value-creation cell requires depth >= 4, the prior was *favourable*
here, so this failure points at the substrate rather than at the depth dose-response -
exactly the reading the registration pre-declared for a P1 failure. Gate 4's quality
premise therefore rests on the toy factor map plus the compute-efficiency protocol
alone, with **no transfer claim at all**, and that is what the declared run's write-up
must say.

**Two corrections the run surfaced, both recorded because they are the interesting part.**

1. *The registration sentence that states the sign convention is inverted.* It says
   "deltas are `null - arm`, so **negative = the arm beats doing nothing**". Every other
   use of the convention in this program - section 5.39's summary and its test, and
   section 5.40's own diagnostic table ("the adaptive arm's harm falls from 0.248 to
   0.117", for `null - adaptive <= 0`) - reads the same delta as **negative = the
   mutating arm is worse**, i.e. the null has the lower loss. The registration's
   sentence is corrected here rather than left standing, because a reader checking the
   verdict against it would conclude the opposite of what the numbers say. This is the
   second sign-convention incident in the program (section 5.39.1 was the first), which
   is why the quantity itself is now asserted convention-free in a test.
2. *The first summary printed the inverted branch.* P1 and P2 are opposite claims about
   the same delta, so they need **opposite tails** (P1: `null - adaptive` significantly
   positive; P2: `null - random` significantly negative). The first run tested both on
   the churn-harm tail, so `established` for P1 meant "the null is significantly better
   than adaptive" - P1 *failing* - and the derivation printed **branch (a), the real-text
   leg reopening**, on a block where the null arm won every seed. The numbers were right
   and only the label was wrong, which is why nothing else caught it: the artifact's own
   `per_seed` rows contradict its branch. `tests/test_deployed_vs_null_summary.py` now
   pins the tail of each primary, the branch re-derived from the stored comparisons, and
   the constructed failure (recomputing P1 on the wrong tail reads as established) - so
   re-running the original bug fails three tests rather than printing a headline claim.

The runner is resumable and budget-aware, and `--summarize` recomputes all of the above,
including the registration's table as a built-in check.

### 5.41 The tolerance band could only be passed at the declared scale (found by reading the committed verdicts)

**What was seen.** Both committed real-path verdicts - `artifacts/real-proof/verdict.json`
(real TinyStories, seeds 9000-9011, section 5.3) and `artifacts/local-proof/verdict.json`
(synthetic tokens, same seeds) - record `in_tolerance_band: false`. The registered
decision rules make a band miss a `PASS_WITH_QUALIFICATION` that "forces re-derivation
before Gate 4", so the two artifacts have read as a band miss since section 5.3.

**Why they cannot have done anything else.** `compute_verdict` takes the measured
`cost_percent` from the run's own booked FLOPs, but derives the band from
`expected_probe_cost_percent`, which is pinned to the **declared** V0 shape
(285,336,576 total / 134,341,632 active, 24,576,000 tokens per arm). The section 5.3
block executed 600 steps on a ~12.2M-parameter model - **10.6983x** away from the
declared scale. No accounting behaviour, correct or incorrect, could have landed inside
that band: the check was measuring the distance between the run and the declaration
rather than between the measurement and the identity.

**The fix is the registration's own rationale, applied.** Section 5.1 justifies the
band's slack by saying "every architectural deviation between toy and V0 (per-layer
expert share, parameter growth, batch shape) enters the identity as a bounded
multiplicative factor on `N_active` or `D`". So the factor is applied explicitly, from
the run's own numbers:

```text
C(run) = C_declared x (T_declared / T_measured) x (N_run / N_declared)
```

which reduces to the registered comparison exactly at declared scale, where both factors
are 1. It is a generalisation of the registered check, not a second criterion.

| block | scale, measured/declared | measured / predicted **at the run's own scale** | registered `in_tolerance_band` |
|---|---|---|---|
| `real-proof` (real TinyStories) | 10.6983x | **1.000286** | false |
| `local-proof` (synthetic tokens) | 10.6631x | **1.000278** | false |

**So the only real-path execution of the compute-efficiency protocol confirms its own
cost identity to 3 parts in 10,000** - with a small systematic +0.03% overshoot on both
blocks - while both artifacts report a miss. The identity was never in doubt here; the
check was applied at the wrong scale.

**What moved, and what did not.** Additive only: a `run_scale` block and a
`band_reading` string. `in_tolerance_band`, `predicted_cost_percent`,
`tolerance_band_percent` and the verdict string are unchanged, verified by re-running the
verdict over both committed blocks under the code immediately before and after this
change: every field this change touched is identical, and the labels do not move
(`real-proof` `PASS_COST_ONLY_HARM_GUARD_REJECTS` on both sides; `local-proof`
`PASS_COST_ONLY_CONTROL_MISSING` on both sides). **Correction (section 5.44):** the first
version of this paragraph said the re-run returned `PASS_COST_ONLY_HARM_GUARD_REJECTS`
"in each, exactly as committed", which is wrong for `local-proof` on two counts - that
artifact's committed label is `PASS_WITH_QUALIFICATION`, and the current code returns
`PASS_COST_ONLY_CONTROL_MISSING` for it because the artifact predates the section 5.5
control clause (it has no `pool_matched_control` key at all). The additive conclusion
stands and is now stated from the before/after comparison rather than asserted. No threshold, criterion or rule moves, the new reading is labelled
`descriptive_only`, and the report does not spend it as a criterion - the band's own
qualification still fires. What it buys is legibility on the run that decides the gate:
a declared queue that deadline-stops, resumes, or is pre-flighted at reduced fidelity now
says *which* reading applies instead of reporting a miss that no accounting error could
have avoided. The declared numbers were also inlined inside
`expected_probe_cost_percent`; they are now named constants
(`DECLARED_TOTAL_PARAMETERS`, `DECLARED_ACTIVE_PARAMETERS`, `PROBE_TOKENS_PER_BATCH`,
`DECLARED_EVENTS`, `DECLARED_CANDIDATES`) so the re-derivation reads the same numbers the
prediction does - prediction unchanged to the last digit, `0.0941636252059042`.

Pinned by three tests in `tests/test_v0_verdict.py`: the exact reduction to the
registered band at declared scale, the additive contract on a reduced-scale fixture, and
the committed section 5.3 block's own numbers.

**Recorded as a pattern.** This is the third instance of the same family as section
5.33's `forced_operation` no-op and section 5.37's unclassified field: **a check that
cannot be passed on the input it is applied to.** This one is worth the write-up because
it sat inside the artifact that decides Gate 4, and because its output was a
qualification that reads as a finding about the hypothesis.

### 5.42 A partially-run pool-matched control made its own clause unfireable (found by comparing two sibling clauses)

**What was seen.** `v0_verdict.py` has two clause evaluators that should behave alike,
and they disagreed about incompleteness. `_null_block` states it in its docstring:
"``evaluated`` requires the queue to exist AND be complete on the declared block; anything
less withholds Gate 4 rather than reading as a null result." `_control_block` returned
``evaluated: True`` with a `seeds_missing` list and no such guard.

**Why that is exploitable.** The pool-matched clause is
`wins >= POOL_CONTROL_WIN_LIMIT` (10). On fewer than ten pairs it cannot fire, however
the loss-blind arm performs - the arm would have to win ten of the four seeds it ran.
`compute_verdict` gates the clause as `control["evaluated"] and
control["control_passes_gate"]`, so an incomplete control reads as *"clause evaluated,
did not fire"*, and `gate_4_authorized` consults only the **primary** queue's
missingness. Built from the verdict's own fixtures, before the fix:

| control pairs | loss-blind arm | clause | verdict | `gate_4_authorized` | exit |
|---|---|---|---|---|---|
| 4 of 12 | wins 4/4 | **unfireable** (4 < 10) | `PASS` | **True** | 0 |

So the clause whose entire purpose is to stop a pool/location effect from authorizing
Gate 4 could be defeated by **not finishing the control queue** - which is exactly the
state a deadline-stopped Kaggle session produces, and nothing warned.

**Fix.** Mirror `_null_block`: `evaluated` now requires the control to be complete on the
declared block, with the missing seeds and the reason recorded. The caller's existing
`PASS_COST_ONLY_CONTROL_MISSING` label (exit 2, Gate 4 withheld) picks it up, so no new
label appears and no other registered behaviour moves. A complete control behaves exactly
as registered, including a 9-of-12 win count staying authorized. After the fix, the same
fixture reads `PASS_COST_ONLY_CONTROL_MISSING`, `gate_4_authorized: False`, exit 2.

**The boundary is pinned too.** A control at 11 of 12 where the loss-blind arm *loses*
every pair is withheld as well: eleven pairs cannot evaluate a clause that needs twelve,
and withholding is what the missing control, the incomplete null queue and the incomplete
primary queue all already do. Two tests, teeth verified by reverting the guard (both
fail; restore passes 38/38).

**Fourth instance, and the shape is now nameable.** Section 5.33's `forced_operation`
no-op, section 5.37's unclassified field, section 5.41's band scale, and this one:
**a check whose result can be decided by something other than the quantity it claims to
measure** - a knob that never binds, a field nothing reads, a band applied at the wrong
scale, a threshold its own input is too short to reach. All four were found by reading,
none by running a study, and all four sat in machinery the project was about to rely on.

### 5.43 Rehearsing the declared Gate-4 verdict path before spending the GPU time

**Why this exists.** The verdict is applied exactly once, at the end of a multi-hour
Kaggle spend. Every clause in it had a unit test at *some* scale, but the whole path had
never run together, at declared scale, in the configuration the real run will be in - and
sections 5.41 and 5.42 both found defects in exactly that machinery, while reading it.
Leaving the first end-to-end run to be the real one means either defect would have
produced a confidently wrong answer at the end of the spend, with nothing to compare it
to.

**What it does.** `scripts/verdict_rehearsal.py` builds synthetic declared-scale fixtures
(285,336,576 total / 134,341,632 active, 24,576,000 tokens per arm, all three queues) and
applies the real `compute_verdict` to **fourteen** variants: every withholding clause in
isolation, both absent-queue cases, both short-queue cases, and the three combinations
whose label precedence is documented in `compute_verdict`'s own comment and, before this,
only one pair of which had ever been tested. The truth table is written to
`artifacts/study-summaries/gate4-verdict-rehearsal.json` and pinned by
`tests/test_gate4_rehearsal.py`.

```text
ok  all_clear                              PASS                                         exit 0 AUTHORIZES
ok  cost_over_criterion                    FAIL                                         exit 1
ok  harm_guard_rejects                     PASS_COST_ONLY_HARM_GUARD_REJECTS            exit 2
ok  pool_control_passes                    PASS_COST_ONLY_POOL_MATCHED_CONTROL_PASSES   exit 2
ok  null_clause_rejects                    PASS_COST_ONLY_NULL_REJECTS                  exit 2
ok  band_miss_at_declared_scale            PASS_WITH_QUALIFICATION                      exit 2
ok  control_absent                         PASS_COST_ONLY_CONTROL_MISSING               exit 2
ok  control_short                          PASS_COST_ONLY_CONTROL_MISSING               exit 2
ok  null_queue_absent                      PASS_COST_ONLY_NULL_MISSING                  exit 2
ok  null_queue_short                       PASS_COST_ONLY_NULL_MISSING                  exit 2
ok  primary_queue_short                    PASS                                         exit 0
ok  cost_fail_beats_every_other_clause     FAIL                                         exit 1
ok  harm_guard_beats_pool_control_and_null PASS_COST_ONLY_HARM_GUARD_REJECTS            exit 2
ok  pool_control_beats_null_clause         PASS_COST_ONLY_POOL_MATCHED_CONTROL_PASSES   exit 2
```

**Three things the rehearsal settled that no earlier test did.**

1. **The label precedence is real, not just a comment.** With the harm guard, the
   pool-matched clause and the null clause *all live at once*, the label is the harm
   guard's; with the pool-matched and null clauses live, it is the pool-matched one. The
   outranked clauses' own flags are recorded alongside, so the labels are precedence
   rather than the absence of the other clauses. Cost failure outranks all of them and is
   the only exit 1.
2. **`primary_queue_short` is the subtle row.** One seed missing from the primary queue
   yields an ordinary `PASS` at exit 0 **and** `gate_4_authorized: False`: the completeness
   requirement lives in the authorization flag, not in the label, so a reader keying on
   the label alone misreads it. Now pinned explicitly.
3. **The section 5.41 and 5.42 fixes hold at declared scale, through the real path.**
   Every fixture reports `run_is_at_declared_scale` and the two band readings agree, and
   both short-queue variants come back `..._MISSING` - the 5.42 guard firing end to end
   rather than only in its own unit test.

**It also caught my own unit error, which is the best evidence that it does something.**
The first pass expressed both cost variants as "6x the prediction", but the prediction is
itself a percentage (0.0942%), so 6x is 0.565% - comfortably *inside* the 5% criterion.
The machinery returned `PASS_WITH_QUALIFICATION` and `HARM_GUARD_REJECTS` where the
rehearsal had written `FAIL`, and its own check reported the drift. Crossing the 5%
criterion takes a multiple above 53.1. A built-in check now ties each variant's measured
cost to its knob, so the unit slip cannot recur silently. That is the same family as
section 5.41 - a quantity compared against a threshold in the wrong units - committed by
the author of section 5.41, one section later.

**What it is not.** No session ran; every loss is fabricated. The record carries
``synthetic: true`` and states that it validates the decision path rather than the
hypothesis, and a test pins both, because a record that could be read as evidence would be
worse than no record. Its use is as a pre-flight:
`PYTHONPATH=src python scripts/verdict_rehearsal.py --check` exits non-zero if any
variant's label, exit code or authorization stops matching the registered rule, and
`T4_RUNBOOK.md` step 0 now calls it before session 1.

### 5.44 The clause that protects the quality premise was shipped in one direction only (found by reading the adopted registration)

**What was seen.** Section 5.5 change 3 was **ADOPTED** on 2026-09-14 and reads:

> if the pool-matched arm's quality is indistinguishable from the adaptive arm's, record
> the cost PASS but withhold the Gate 4 recommendation, since the quality premise would
> then be unsupported

It says *indistinguishable*, which has two directions. The shipped rule in
`v0_verdict.py` implemented one of them: the clause fires when the loss-blind arm clears
the 10-of-12 bar against its own churn arm
(`PASS_COST_ONLY_POOL_MATCHED_CONTROL_PASSES`), i.e. when **both** arms clear it and a
pool/location effect alone would have passed the gate. The other direction - **neither**
arm clears it, so the two are exactly as indistinguishable as in the first case - was not
implemented anywhere. There is no adaptive-vs-control comparison in the verdict at all.

**Why that is not academic.** Nothing else in the gate looks at the adaptive arm's
quality. The harm guard is a floor, not a quality claim: it rejects only when adaptive
loses to its own churn arm in **10 or more of 12** pairs, so 3 wins already clears it. And
the section 5.14 clause compares churn against do-nothing, not adaptive against the
loss-blind arm. Measured on the verdict's own fixtures, before this amendment:

| adaptive vs churn | loss-blind arm | before | after |
|---|---|---|---|
| 6/12 (a tie) | 6/12 | `PASS`, exit 0, **Gate 4 AUTHORIZED** | `PASS_COST_ONLY_NO_DEMONSTRATED_EDGE`, exit 2, withheld |
| 9/12 | 3/12 | `PASS`, exit 0, **AUTHORIZED** | `PASS_COST_ONLY_NO_DEMONSTRATED_EDGE`, exit 2, withheld |
| **3/12** (one win above rejection) | 0/12 | `PASS`, exit 0, **AUTHORIZED** | `PASS_COST_ONLY_NO_DEMONSTRATED_EDGE`, exit 2, withheld |
| 2/12 | 0/12 | `PASS_COST_ONLY_HARM_GUARD_REJECTS` | unchanged |
| 10/12 | 3/12 | `PASS`, authorized | unchanged, authorized |
| 12/12 | 0/12 | `PASS`, authorized | unchanged, authorized |

So the gate could authorize on a run in which the controller was beaten by uniform churn
in **9 of 12 seed pairs**, and it did so at the harm guard's own boundary - one win away
from rejection. The tie row is the state the dev-fidelity evidence predicts for the real
path (section 5.13 closed the real-text topology program with every vehicle carrying the
router defect; section 5.39 showed the real path's remaining churn cost). That is the
worst case for the missing direction: the exact state the clause exists to catch was the
state the shipped rule authorized.

**The amendment.** The clause is completed in the missing direction, using the same
pre-committed constant rather than a new threshold: it fires when neither arm clears
`POOL_CONTROL_WIN_LIMIT` against its own churn arm, and reads **strictly tightening** - it
can only turn an authorization into a withholding. The record carries a `quality_premise`
block (`adaptive_wins`, `adaptive_clears_bar`, `control_clears_bar`, `win_limit`,
`indistinguishable`, `reading`) so the reason is stored rather than inferred from the
label. Consequence, stated plainly: **authorizing Gate 4 now requires the adaptive arm to
clear the pre-committed bar and the loss-blind arm not to.**

**The boundary, flagged rather than derived.** `indistinguishable` is operationalised as
*both arms on the same side of the bar*, not as a statistical test of equality. That makes
the clause withhold in the 9-vs-3 row above, where the adaptive arm is clearly better than
the loss-blind arm but neither clears the bar. That is the reading a reader may want to
contest, and it is recorded as a boundary of the proxy in
`registered_text_amendments[0].boundary_reading` rather than presented as the wording's
only possible meaning.

**No spent block is reinterpreted - verified, not argued.** The two committed real-path
verdicts were re-computed under the pre-amendment and post-amendment code: identical
labels (`real-proof` `PASS_COST_ONLY_HARM_GUARD_REJECTS` both sides; `local-proof`
`PASS_COST_ONLY_CONTROL_MISSING` both sides). `real-proof`'s committed label is the same
string, so nothing about it moves; `local-proof`'s committed label is
`PASS_WITH_QUALIFICATION` and differs from *both* re-runs because that artifact predates
the section 5.5 control clause - a pre-existing drift, now stated in the 5.41 correction
above.

**How it was found, and what caught my own error.** By reading the adopted registration
against the code, then by adding a rehearsal variant for the predicted real-path state.
The variant's first expectation was **wrong** - I wrote `PASS` for the 9-vs-3 row,
believing the clause should only fire on a near-tie - and the rehearsal's own
`every_variant_matches_the_registered_rule` check reported the drift against the real
`compute_verdict` output rather than letting my expectation become the record. Two of the
verdict's own unit tests also failed on the amendment, at the harm-guard boundary and on a
re-declared block; both were rewritten to pin the *interaction* (guard clear, quality
clause withholding) instead of being relaxed.

Teeth-checked by removing the withholding branch: the two faulted unit rows fail, the
rehearsal's own `--check` exits 1 with three checks false, and restoring it returns 52/52.
The rehearsal now runs 16 variants with 9 checks.

### 5.45 The test suite was slow for two measurable reasons, one of them in production code

**Why this was worth doing.** The suite is the feedback loop for every change here, and it
had crept to ~140 s. It was measured rather than guessed: `pytest --durations=35` attributed
~90 s of 140 s to 25 tests, and 41 s of that to four.

**The first 15.0 s was production retry backoff paid by one error-path test.**
`test_v0_runner.py` drove a publish failure through `cli.main()` and paid the real
`time.sleep` backoff inside `publish_on_exit` - exactly 15.0 s, 11% of the whole suite in
wall-clock padding. `publish_on_exit` already accepts an injectable `sleep` and the
dedicated remote tests already inject a no-op; only the CLI-level test paid the production
delay. It now injects one too.

**The second was an O(n²) big-integer search in the shared statistics library.**
`act/evolution/study.py::_gate_min_wins` walks wins downward and re-sums `math.comb` from
scratch at each step through `_binomial_survival`, so a 1536-seed block does ~1.06M
`math.comb` calls: **16.15 s**, from the one test that exercises that block size. It is now
done with the exact recurrence - accumulate the binomial tail once, top-down, dividing by
the exact factor - in 0.4 ms, a **~400x** improvement. Because the number the function
returns is a *pre-registered* analysis threshold, equality with the old implementation was
verified rather than assumed: **bit-identical on 199 checks** across 12 block sizes
(including 1536), 8 alphas, and the tail-sum identity.

| | before | after |
|---|---|---|
| `_gate_min_wins(1536, 1/60)` | ~16 s | 0.4 ms |
| `_gate_min_wins(384, 1/60)` | 20.6 ms | 0.05 ms |
| full suite, `OMP_NUM_THREADS=1` (the convention every committed number uses) | ~140 s | **98.5 s** |
| full suite, threads unpinned | - | 88.3 s |

The two fixes are 31.2 s of the ~42 s reduction; the remaining time is real work -729
tests whose slowest entries are now genuine fp16 session round-trips (9.1 s and 4.4 s),
not padding. Both figures are given because the suite's wall time depends on the thread
count, so only the pinned row is comparable with the earlier ~140 s.

Nothing about any criterion changes: the fast path returns the same integer for every
(input, alpha) tested, and the tests that pin the pre-registered thresholds still pass.

### 5.46 The declared launch path cannot specify the rebuild treatment the report says it should carry (found by auditing the launch command against the adopted sections)

**What was seen.** Three places say what a real-scale V0 run should do about the
section 5.28 rebuild treatment, and the declared launch path can do none of them:

- Section 5.16: "**New work should set `preserve_router_on_rebuild=True`**, and this
  paragraph is the reason a reader will find the default looking wrong."
- Section 5.38.3: "A real-scale protocol **should be specified with the treatment on**, and
  it should not be sold an effect size on the strength of this."
- Section 5.39: on the real path, doing nothing beats churning by **0.16449** nats under
  the historical rebuild and by **0.11279** nats under the repair - the treatment is worth
  **1.46x** on the exact quantity the gate's quality clause reads.

But `cli._run_pair_session` builds its config as
`declared_session_config(arms=..., probe_batch_count=..., probe_candidate_policy=...,
probe_selection=..., device=..., dtype=..., max_checkpoints=...)` - no rebuild field - and
`declared_session_config` does not set one either, so the declared run inherits the field
defaults. There was **no CLI flag** for either `preserve_router_on_rebuild` or
`new_expert_initialization`, and no test or paragraph pinned the declared run's rebuild
policy. So the declared protocol was registered (section 5.5, 2026-09-14) before the
treatment existed (sections 5.16-5.28), and the plumbing to specify it was never added:
an unflagged launch always runs the historical rebuild.

**Why this is the highest-consequence member of this family so far.** The others made a
check unable to fail; this one makes the run unable to be what the report says it should
be. The verdict's clauses are read against the published run, so a declared run under the
historical rebuild would be judged on a configuration the project already decided not to
use - and the clause added in section 5.44 reads precisely that difference. Section 5.39's
stored rows say the gap is **0.0517 nats**, which is the same order as the treatment effect
itself (0.05170 pooled, section 5.38.1) and larger than the clean-arm effect measured from
scratch (0.04042, n=12). Under the historical rebuild the predicted real-path state is
"the adaptive arm does not clear the bar", which withholds Gate 4; under the repair it may
clear it. **The missing flag can decide the gate.**

**What was done now (additive, no default moved).** `act.training.cli` gained
`--preserve-router-on-rebuild` and `--new-expert {random,clone}`, threaded into
`declared_session_config` for both the declared and `--profile` shapes. Both default to the
historical behaviour, so an unflagged launch is byte-for-byte what every committed run
used; `--new-expert clone` without router preservation is refused with a one-line
`SystemExit` rather than a traceback. Three tests pin the default (so it cannot drift
silently), the threaded treatment, and the refusal - and pin that the treatment does not
move `total_steps` or `probe_batch_count`, the fields the declared cost prediction reads.
A second, useful fact fell out: the published seed summary already records the whole
config (`asdict`, 35 fields), so **which rebuild policy a run used is checkable from its
own artifacts** - the declared blocks' summaries predate the field, which is why this was
invisible.

**What was done about the command, and by whose decision.** The runbook's launch command
was not changed in this section, because registering the declared protocol on the treatment
is a sign-off decision. It is taken in **section 5.47** below, with the reasoning, and the
runbook now carries the two flags rather than a pending note.

**The second half, built in the same section.** The verdict never compared a published
run's recorded configuration against the registered protocol, so a run whose configuration
differed from the registration would have been judged as though it were the registration -
the same family as sections 5.41 and 5.44. The verdict now carries an additive
`config_conformance` block (`REGISTERED_RUN_CONFIG`, the machine-readable registration,
against every seed's recorded `config`): per field `matches` / `differs` with the recorded
values / `not_recorded`, plus `conforms` and a reading string. It is **descriptive only**,
on section 5.41's precedent - it changes no label, no exit code and no authorization, and
the rehearsal now pins that by building the untreated-rebuild and pre-5.5 `prefix` runs and
checking they keep the authorizations their clauses give them while being flagged. A
pre-field artifact reads as `not_recorded`, never as a mismatch, so no spent block is
re-read as non-conforming. A test drives the real launch path and asserts the registered
mapping cannot drift from what the command builds: a conformance block compared against a
stale constant would certify a run nothing produces.

**What the block says about the two committed real-path blocks, read and recorded.** Both
`artifacts/real-proof` and `artifacts/local-proof` now report `conforms: false` - differing
on nine shape/budget fields (they are 600-step depth-8 diagnostics at ~12M parameters, not
declared-scale runs) and unrecorded on the four fields their artifacts predate
(`probe_candidate_selection`, `probe_candidate_policy`, `preserve_router_on_rebuild`,
`new_expert_initialization`). That is the correct reading, not a defect: neither block was
ever a declared-configuration run, neither authorized Gate 4 (`PASS_COST_ONLY_HARM_GUARD_
REJECTS` and `PASS_COST_ONLY_CONTROL_MISSING`, unchanged by this addition). It is worth
recording because it is the first time the verdict has been able to say so.

### 5.47 Amendment: the declared run is registered with the rebuild treatment on

**Decision.** The declared V0 launch command passes `--preserve-router-on-rebuild
--new-expert clone`. Delegated to me rather than signed off by the project owner, so it is
recorded here with its reasoning and its reversibility.

**Grounds, in order of weight.**

1. **The untreated rebuild is a defect kept for backwards compatibility, not a choice.**
   Section 5.16: the router discard is "evidently unintended", and the default was retained
   only because flipping it would invalidate the committed record of sections 5.16-5.23.
   That reasoning binds *defaults*; it does not argue for running a declared protocol on the
   accidental behaviour.
2. **The registered rule that would have forbidden this has been superseded by its own
   correctly-sized block.** Section 5.38.1 passed the clean-arm primary (the arm probe
   ranking cannot affect) on 12 fresh seeds with the declared statistic (permutation
   p 0.0239) and the clean-comparison guarantee holding 12/12. Section 5.38's branch (c) -
   which says to treat a probe-arm-only improvement as a red flag - was diagnosed as a
   power failure there (n=6 sized from an sd 3.8x too small) and explicitly superseded for
   interpretation, with its numbers left standing.
3. **The gap it closes is larger than the effect being claimed.** Section 5.39: 1.46x on
   the churn cost (0.16449 untreated, 0.11279 repaired), against the 0.0517 pooled
   treatment effect - and it is the quantity section 5.44's quality clause reads.

**What this is not.** It is not a claim that the treatment decides Gate 4, and it must not
be reported as one. The effect is small (0.04042 on the clean arm, perm p 0.0239, but 8/12
against a 10-of-12 win bar, right-skewed), and section 5.39 showed the repair still leaves
**69%** of the churn harm - so section 5.7's "doing nothing beats churning" still holds in
the treated configuration and the section 5.14 do-nothing clause remains the binding
protection, not this amendment.

**What changes, exactly.** Two configuration fields, applied to **every** arm, so none of
the gate's arm-vs-arm comparisons is affected; no parameter count moves and the probe FLOPs
are identical, so the declared cost prediction and its band are untouched. **No criterion
changes**: the 5% criterion, the tolerance band, the harm guard, the pool-matched and
quality clauses of section 5.5, the do-nothing clause of section 5.14 and the queue
completeness requirement all stand as registered. Seeds (15000-15011), shape, tokens per
arm, event count and cadence, dtype and the adopted probe regime (`spread`, `argmin`,
cap 4, 1-batch) are all unchanged.

**Reversible, and honest about the alternative.** Before session 1, dropping the two flags
returns the protocol to its pre-amendment configuration; the conformance block would then
record the run as flagged - that is, an untreated declared run is now *visible in its own
artifact* rather than indistinguishable from a treated one. That legibility is the part of
this amendment that holds whichever way the configuration decision goes.

### 5.48 Pre-registration: the width covariate at the depth onset (RAN - summaries `width-d3-equal-576.json`, `width-d4-equal-576.json`)

Section 10.4 named residual width the largest matched-budget effect either track has
measured, and 10.3's open question (ii) asks whether the depth onset is *depth* or
*capacity*. The arch-track arm that **moves** width needs a transfer policy that does not
exist (10.5(B)); the arm that holds width **fixed** and asks whether selection value
follows it is answerable today on the toy path - which is where the onset lives and where
the controller's value is measurable at all (at dev fidelity on real text the two mutating
arms are tied: adaptive - random = -0.0101, mean 6/12, 5.40).

**Design.** Two fresh cells of the registered toy config (5.9: `random,adaptive,null`,
`independent_arm_rng=True`, `device=cpu`, cap 4 / 2 probe batches, 48 steps, interval 12,
budget 3, `adaptive_policy=probe_loss`), the same three-arm block per seed and the same
20-way sharding, with **one gene changed and nothing else**:

| cell | block | n | depth | width | total params | active params |
|---|---|---|---|---|---|---|
| `width-d3-equal` | 76000-76575 | 576 | 3 | **28** | 31,192 (+1.7%) | 11,704 (+2.0%) |
| `width-d4-equal` | 77000-77575 | 576 | 4 | **28** | 41,160 | 15,176 |

Width 28 is *solved*, not chosen: it is the closest integer width whose parameter count at
depth 3 equals the committed above-onset cell's - depth 4 x width 24 = **30,672 total /
11,472 active** (5.12's `depth4-prefix` cell). So `width-d3-equal` is an equal-parameter
partner of the cell that creates value, at a depth that does not; the match is pinned by a
test against the model's own parameter count, not restated.

**Primaries (per cell, one number, computed by the verdict's own function).** The
one-sided exact sign test on per-seed adaptive-vs-null deltas at `alpha = 0.05` with two
primaries, Bonferroni `alpha/2 = 0.025`; at n = 576 the bar is **313/576**
(`_gate_min_wins(576, 0.025)`, asserted by a test). A cell counts as *creating value* only
when the wins clear the bar **and** the mean delta is negative: the conjunction is
pre-registered because a win-rate crossing can be a tail effect - depth 1 is exactly that
(117/384 wins with the adaptive arm 0.0508 *worse* on average, 5.15) - and the six
committed cells never disagreed on the sign. The bar, the alpha and the code are one
statement (the 5.16.1 lesson), and the seed count was sized to the *committed* above-onset
rate (225/384 = 0.5859): exact power **0.982** at n = 576, 0.801 at 0.56, 0.641 at 0.55. A
smaller true effect reads as a null **by design**, which is stated here rather than
discovered later.

1. **P1 (capacity).** `width-d3-equal` creates value: a below-onset depth does create value
   once the substrate carries the above-onset cell's parameter count.
2. **P2 (persistence).** `width-d4-equal` creates value: the above-onset cell keeps its
   value at the wider level.

**Pre-registered interpretation (fixed before running).**

- **P1 and P2:** the onset is a *capacity* onset - depth and width trade off at matched
  parameters, so (ii)'s covariate is parameter count and 10.4's width lever is the same
  axis as the depth axis.
- **P1 only:** widening creates value below the onset and destroys it above - a depth x
  width interaction; the capacity reading needs the depth axis retained.
- **P2 only:** the onset is depth-like at this width; widening does not rescue a
  below-onset substrate.
- **Neither:** width is harmful at both depths here; the depth axis is not capacity, and
  the width covariate is closed for this search space.

No branch re-tunes anything, and a null result is recorded as a finding (the 5.5
discipline). The committed comparison cells are quoted as-is at the applied sibling bar
212/384: (depth 3, width 24) **188/384**, mean +0.0046 (5.15), below onset; (depth 4,
width 24) **225/384**, mean -0.0371 (5.12), above it. The two new cells are what decides
the question.

**Built-in checks.** Identical to 5.9-5.15: mutations 3/3/0, zero null search FLOPs,
matched budgets, arm-RNG isolation. The verdict is machine-checked (`width-verdict` reads
both cells and prints exactly one of the four branches above).

**Cost.** Measured at the new cells: ~1.2-1.4 s/seed (3 arms, 4 threads). Two cells x 576
seeds is ~25 min of single-thread CPU across 20 shards - minutes of wall clock, against no
GPU quota. Registered before the run; the revision that ran it is recorded in the
summaries' `code_revision`.

**Outcome (run 2026-09-20, revision `38f3921`, 20 shards per cell; committed
summaries `artifacts/study-summaries/width-d3-equal-576.json` and
`width-d4-equal-576.json`). Registered branch 3 - P2 only.** Built-in checks held in
576/576 seeds per cell (mutations 3/3/0, zero null search FLOPs).

| cell | params (total / active) | P2 adaptive-vs-null | mean delta | bar 313 | creates value |
|---|---|---|---|---|---|
| depth 3, width 24 (5.15) | 23,280 / 8,880 | 188/384 | `+0.0046` | fails at the sibling bar 212 | no |
| **depth 3, width 28** | **31,192 / 11,704** | **252/576** | **`+0.0176`** | fails (p `0.999`) | **no** |
| depth 4, width 24 (5.12) | 30,672 / 11,472 | 225/384 | `-0.0371` | passes 212 | yes |
| **depth 4, width 28** | **41,160 / 15,176** | **316/576** | **`-0.0326`** | **passes (p `0.011`)** | **yes** |

The equal-parameter contrast is the first and third rows' partner cells: at **the same
parameter count** (31,192 vs 30,672, within 1.7%), depth 3 does not create value and
depth 4 does. Widening does not buy what depth buys. The regime, not the level, is what
width leaves alone: at both widths depth 3 is a churn-suppression cell (arm means
`null 14.9594 < adaptive 14.9771 < random 15.1078` - adaptive beats random by `0.1307`,
p `4.2e-25`, and loses to doing nothing on average) and depth 4 is a value-creation cell
(`adaptive 13.6503 < null 13.6829 < random 13.7710`). The two width-28 cells are also
within noise of their width-24 counterparts as a *level* (descriptive two-proportion
`p = 0.11` and `p = 0.25`), so nothing here says width is harmful - it says width is not
the covariate the onset reads.

**What this closes, and what it does not.** It closes the matched-parameter form of open
question (ii): in this search space, capacity in the residual stream does not do what
depth does, at the parameter count where the above-onset cell's value exists. It does
**not** close "any width" - the tested contrast is one level (+1.7% parameters, width 28
vs 24) at two depths, and a dose (much wider below-onset substrates, no longer
parameter-matched) is a different registration. That registration was then written and run
the same day (**5.49**, below): the dose does not move the gate either, and it shows what
capacity *does* do above it. It also answers 10.5(A)'s question in the
form that could be asked today: at matched parameters, the controller's value does not
depend on the width it was given - its regime tracks depth - so moving width during a
search (10.5(B)) would be a question about a different search, exactly the consequence
10.5 registered for this outcome. No branch re-tunes anything.

### 5.49 Pre-registration: the width *dose* at both depths (RAN - summaries `width-d3-dose-576.json`, `width-d4-dose-576.json`)

Section 5.48 answered the matched-parameter form of open question (ii) - at equal parameter
count, depth 3 does not create value and depth 4 does - and registered explicitly what it
did **not** close: "any width". The tested contrast there was +1.7% parameters. This arm
spends width as a **dose**: width 48 at both depths, deliberately no longer
parameter-matched, to ask whether enough capacity can buy what the matched contrast could
not.

**Design.** The same registered toy config and three-arm block as 5.48 (and as 5.9-5.15)
with only the width gene changed - again nothing else:

| cell | block | n | depth | width | total params | active params | vs the above-onset cell |
|---|---|---|---|---|---|---|---|
| `width-d3-dose` | 78000-78575 | 576 | 3 | **48** | 88,032 | 31,584 | **2.87x** |
| `width-d4-dose` | 79000-79575 | 576 | 4 | **48** | 116,640 | 41,376 | 3.80x |

**Primaries.** Identical to 5.48, and stated the same way - one number, computed by the
verdict's own function: the one-sided exact sign test on per-seed adaptive-vs-null deltas
at `alpha = 0.05` with two primaries, Bonferroni `alpha/2 = 0.025`, bar **313/576**; a cell
creates value only when the wins clear the bar **and** the mean delta is negative. Power at
the committed above-onset rate (0.5859): **0.982** (0.801 at 0.56, 0.641 at 0.55) - the
same sizing as 5.48, restated because this arm's doses carry 2.87x the substrate that
produced that rate.

1. **P1 (the dose buys value below the onset).** `width-d3-dose` creates value.
2. **P2 (the above-onset cell survives the dose).** `width-d4-dose` creates value.

**Pre-registered interpretation (fixed before running).**

- **P1 and P2:** capacity in dose form buys value below the onset and preserves it above -
  the onset is capacity-driven above a threshold between +1.7% and +280% parameters, and
  5.48's null is a statement about *small* capacity shifts, not about capacity.
- **P1 only:** a dose buys value below the onset and destroys it above - a depth x width
  interaction in dose form.
- **P2 only:** depth stands even at 2.87x width; with 5.48, the width covariate is closed
  at this search space's reachable widths.
- **Neither:** width is harmful above the onset and useless below; the depth axis is not
  capacity in dose form.

No branch re-tunes anything (the 5.5 discipline), and both blocks are fresh - disjoint from
every committed study's seeds, checked by a test rather than asserted.

**Cost.** ~1.3-1.5 s/seed measured at the dose cells (3 arms, 4 threads); two cells x 576
seeds is ~30 min of single-thread CPU across 20 shards - minutes of wall clock, no GPU
quota - run while Gate 4 trains. Registered before the run; the running revision is
recorded in the summaries' `code_revision`.

**Outcome (run 2026-09-20, revision `e5dcbf6`, 20 shards per cell; committed summaries
`artifacts/study-summaries/width-d3-dose-576.json` and `width-d4-dose-576.json`).
Registered branch 3 - P2 only.** Built-in checks held in 576/576 seeds per cell.

| cell | params | P2 adaptive-vs-null | mean delta | creates value |
|---|---|---|---|---|
| depth 3, width 24 | 23,280 | 188/384 | `+0.0046` | no |
| depth 3, width 28 | 31,192 | 252/576 | `+0.0176` | no |
| **depth 3, width 48** | **88,032** | **235/576** | **`+0.0615`** | **no** |
| depth 4, width 24 | 30,672 | 225/384 | `-0.0371` | yes |
| depth 4, width 28 | 41,160 | 316/576 | `-0.0326` | yes |
| **depth 4, width 48** | **116,640** | **477/576** | **`-0.2097`** | **yes** |

So the pre-registered reading stands in its strongest form: **the gate is depth, and a
2.87x dose of capacity does not move it** - depth 3 fails at every width measured, and its
average outcome gets *worse* as width grows (`+0.0046` -> `+0.0176` -> `+0.0615`; win rate
`0.4896` -> `0.4375` -> `0.4080`, descriptive two-proportion `z = 2.49` for the 24-vs-48
ends, `p = 0.013`). Two cross-cell observations are **descriptive, not pre-registered**
(no cross-cell claim was registered in either arm):

1. **Above the gate, capacity is an amplifier.** The same dose that does nothing below it
takes the above-onset cell from `225/384` (mean `-0.0371`) to **`477/576` = 0.828** (mean
`-0.2097`) - a both-controls-clearing edge (vs random `432/576`, mean `-0.1945`, p
`1.1e-34`), and against width 24 the difference is `z = 8.2`. Value creation is depth-*gated*
and capacity-*scaled*.
2. **The churn harm itself is width-dependent above the gate - and it vanishes.** At depth 4
x width 48 the null arm does **not** beat random churn: `null-vs-random 266/576`, mean
`+0.0152`, p `0.97` - the first cell in the program where churning does not pay, against
`387/576` (mean `-0.0880`) at width 28 and `266/384` (mean `-0.0701`) at width 24 in the
same regime. Below the gate the harm is at its strongest (`494/576`, mean `-0.3053`), so
the two regimes separate on *both* readings: shallow-and-wide churns destructively without
creating value, deep-and-wide absorbs churn and creates value.

**What this changes.** Open question (ii) is answered at this space's reachable widths:
the onset is depth per se against the capacity covariate, and what capacity does is scale
the value above the gate and (at 3.80x) remove the churn harm. It makes 10.4's width lever
the same axis as this program's *amplifier* - the two tracks now share a measured
phenomenon (width scales selection value above a depth gate) - and it sharpens what
10.5(B) would have to be worth: a width *mutation* above the onset has a large measured
upside, while the transfer-policy problem (5.24, 5.40) is untouched by this arm, which
changes no model during training. Its 5.16.1-style discipline note: the branch was fixed
before the run, and the two descriptive readings above are labelled as such rather than
folded into the registered claim.

### 5.50 Pre-registration: the width amplifier's *second* dose (RAN - summaries `width-d3-dose2-576.json`, `width-d4-dose2-576.json`)

5.49 answered the dose form of open question (ii) at one dose: at 2.87x/3.80x the
substrate, capacity does not move the onset and does amplify the value above it
(`0.586` -> `0.828`, mean `-0.2097`), and it removed the churn harm. One dose cannot say
whether capacity keeps paying or plateaus - and that shape is exactly what 10.5(B)
prices: a width *mutation* above the onset is worth the marginal value of capacity, so a
plateau would say the transfer policy is not worth building. This arm spends a second,
larger dose, and it pre-registers the continuation claim as a test between two
independent cells - the family's first non-paired primary.

**Design.** The same registered toy config and three-arm block as 5.9-5.49, with only the
width gene changed:

| cell | block | n | depth | width | total params | active params | vs the above-onset cell | vs the 5.49 rung above it |
|---|---|---|---|---|---|---|---|---|
| `width-d3-dose2` | 80000-80575 | 576 | 3 | **96** | 341,952 | 118,464 | **11.15x** | 3.88x |
| `width-d4-dose2` | 84000-84575 | 576 | 4 | **96** | 454,464 | 156,480 | **14.82x** | 3.90x |

Both blocks are fresh against every other profile's registered block and every committed
summary - checked by a test, and the check earned its keep before the run: the depth-4
block's first draft (81000-81575) overlapped the section-5.17 reusability block
(81000-81383) and was moved to the first free window.

**Primaries.** Three this time, so the family's Bonferroni level is `alpha/3 = 0.01667`,
bar **315/576** - computed by the verdict's own `_gate_min_wins` (5.16.1's discipline),
not transcribed. The two cell primaries are the sibling cells' own test at the stricter
level; power at the committed above-onset rate (0.5859) is **0.97**.

1. **P1 (the largest dose buys no value below the onset).** `width-d3-dose2` creates
   value - wins clear the bar **and** the mean delta is negative.
2. **P2 (value creation survives the largest dose).** `width-d4-dose2` creates value,
   same conjunction.
3. **P3 (the amplifier keeps paying).** One-sided rank-sum of `width-d4-dose2`'s per-seed
   adaptive-vs-null deltas against the committed 3.80x rung's (`width-d4-dose`),
   stochastic superiority meaning the bigger dose's deltas tend more negative, at
   `alpha/3`. This is the family's **one asymptotic test** - average ranks for ties,
   tie-corrected variance, a continuity correction in the conservative direction -
   because P3 compares two independent fresh blocks, the one contrast no paired exact
   test can reach. It was rehearsed on committed data through the same function the
   verdict calls: on the measured dose step (3.80x versus the width-28 cell, a 2.84x
   parameter step) it reads `p = 4.5e-44`, and on the measured wash (+1.7%, width 28
   versus 24) `p = 0.50`. Detectability at n = 576 per cell: power ~0.8 at a probability
   of superiority of 0.55.

**Pre-registered interpretation (fixed before running).**

- **P1 and P2:** at 14.8x, capacity buys value at a below-onset depth - the onset is
  capacity-driven above a threshold between 3.8x and 14.8x, and 5.49's depth clause is a
  statement about the reachable doses up to 3.8x.
- **P1 only:** the largest dose buys value below the onset and destroys it above - the
  depth x width interaction inverts at high dose.
- **P2 only:** depth stands at every reachable dose; value creation survives 14.8x
  width. P3 then reads the dose form: **continues** if it clears `alpha/3`, otherwise
  "not distinguishable from a plateau at this level". The rider prints only on this
  branch, and no cell-level win rate - even one above the committed `0.828` - is allowed
  to carry it (pinned by a constructed test).
- **Neither:** at 14.8x nothing creates value - capacity above the gate reverses at the
  fixed training budget.

Secondaries, fixed in advance but outside the branch table: the churn reading
(null-vs-random sign test at the same `alpha/3`, labelled by that test alone, as in
5.49) and the descriptive cross-cell levels.

**Cost.** ~1.5 s/seed measured at the new cells (three-arm block, one thread per run;
width 96 at the same 48 steps) - two cells x 576 seeds is ~30 min of single-thread CPU
across 20 shards, minutes of wall clock, no GPU quota, run while Gate 4 trains.
Registered before the run; the running revision is recorded in the summaries'
`code_revision`.

**Outcome (run 2026-09-20, revision `05efa7b`, 20 shards per cell; committed summaries
`width-d3-dose2-576.json` and `width-d4-dose2-576.json`). Registered branch 1: P1+P2
PASS, with the third primary also passing.** Built-in checks held in 576/576 seeds per
cell, and six replays of committed seeds matched the shards exactly.

| cell | params | adaptive-vs-null | mean delta | p | creates value |
|---|---|---|---|---|---|
| depth 3, width 96 | 341,952 | 573/576 | `-0.8966` | `1.3e-166` | **yes** |
| depth 4, width 96 | 454,464 | 576/576 | `-1.7330` | `4.0e-174` | **yes** |

The registered reading stands in branch 1's form: at 11.15x/14.82x both cells clear the
do-nothing bar at `alpha/3` (bar `315/576`), so **the onset is capacity-driven above a
threshold between 3.8x and 14.8x** - 5.49's depth clause is a statement about the
reachable doses up to 3.8x. P3 also passes in its strongest possible form: the 14.8x
cell's adaptive-vs-null deltas are *completely* separated from the 3.8x rung's (`U = 0.0`,
p `4.8e-190`). Its dose-form rider was pre-registered for the P2-only branch, which did
not obtain, so the continuation is recorded here as a primary result rather than as that
branch's reading.

**The decomposition is where the interest is - and it qualifies 5.49's "amplifier"
language.** Every cell carries three paired contrasts; against the committed above-onset
rung and the earlier widths:

| cell (depth 4) | x the above-onset cell | adaptive-vs-null | adaptive-vs-random (the controller's edge) | random-vs-null (the pool effect) |
|---|---|---|---|---|
| width 24 (committed) | 1.00x | `-0.0371` | `-0.1072` | `+0.0701` |
| width 28 (5.48) | 1.34x | `-0.0326` | `-0.1207` | `+0.0880` |
| width 48 (5.49) | 3.80x | `-0.2097` | `-0.1945` | `-0.0152` |
| **width 96 (5.50)** | **14.82x** | **`-1.7330`** | **`-0.1591`** | **`-1.5739`** |

and below the gate, at depth 3 (same reading, the cells at 0.76x/1.02x/2.87x/11.15x):

| cell (depth 3) | adaptive-vs-null | adaptive-vs-random | random-vs-null |
|---|---|---|---|
| width 24 (committed) | `+0.0046` | `-0.1022` | `+0.1068` |
| width 28 (5.48) | `+0.0176` | `-0.1307` | `+0.1484` |
| width 48 (5.49) | `+0.0615` | `-0.2438` | `+0.3053` |
| **width 96 (5.50)** | **`-0.8966`** | **`-0.3106`** | **`-0.5860`** |

Three readings, all cross-cell and therefore **descriptive, not pre-registered** (the
family's convention, unchanged):

1. **The controller's edge does not scale above the gate; the pool effect does.**
   `adaptive`-vs-`random` reads `-0.107`, `-0.121`, `-0.195`, `-0.159` across the four
   depth-4 rungs: flat, and the w96-vs-w48 rank-sum on that contrast reads p `0.98` in
   the "larger at 14.8x" direction (the reverse direction reads p `0.016` - marginally
   *smaller*). What explodes is `random`-vs-`null`: `+0.070`, `+0.088`, `-0.015`,
   **`-1.574`** (rank-sum p `6.3e-190`). At 14.8x the unmutated arm is not slightly
   worse - it is the worst arm by a wide margin (depth-4 x width-96 arm means
   `adaptive 15.127 < random 15.286 < null 16.860`). **The dose buys escape from a
   collapse, not a sharper controller.**
2. **Below the gate the same decomposition explains the crossing.** The depth-3 edge
   grows with every rung (`-0.102`, `-0.131`, `-0.244`, `-0.311`; w96-vs-w48 p
   `1.4e-3`) and its pool effect changes sign (`+0.107` -> `+0.305` -> `-0.586`). At
   11.15x the below-onset cell creates value against the bar, and its controller edge
   (`-0.311`) is the largest in either ladder - so the crossing is the pool effect
   flipping, not the depth-3 controller changing regime.
3. **Absolute levels say the "value" is not a quality gain.** The width-96 arms
   (`15.1`-`16.9`) are far worse in absolute loss than the committed rung's
   (`12.4`-`12.5`): at a fixed 48-step budget, capacity is a liability that mutation
   partly repairs. The contrast is what this arm measured, and this confound - capacity
   versus trainability at fixed steps - is its own stated limit rather than a result.

**What this changes.** Open question (ii)'s boundary moves: the onset is depth-like up to
3.8x and capacity-driven at 14.8x, with the mechanism named. 10.1's churn axis flips sign
in width (harm at <=1.34x, absent at 3.8x, *pays* at 14.8x). And 10.5(B) sharpens rather
than resolves: the measured upside of a wide substrate is a package effect - mutation at
all - while the controller's own edge is flat, so the transfer-policy problem (5.24,
5.40) remains exactly where it was, untouched by any arm that changes no model shape
mid-run.

### 5.51 Pre-registration: the 14.8x collapse against the training budget (RAN - summary `width-d4-dose2-long-576.json`)

5.50 registered its own limit rather than only a result: at 14.8x the do-nothing arm collapses
(`random`-vs-`null` `-1.574`), and at a fixed 48-step budget "a wide substrate is unusable
unless mutated at all" cannot be separated from "it needs more training than the budget
gives it". The absolute levels say the same thing from the other side (the width-96 arms at
`15.1`-`16.9` against the committed rung's `12.4`-`12.5`). This arm asks which of the two
the collapse is, with one cell and two primaries.

**Design.** The identical depth-4 x width-96 config at **4x the steps (192) and 4x the
mutation interval (48)** - which keeps the mutational schedule at the same three events,
asserted by the built-in checks (3/3/0) - so the only changes are how much training happens
and when the events fall:

| cell | block | n | depth | width | steps | interval | events | total params |
|---|---|---|---|---|---|---|---|---|
| `width-d4-dose2-long` | 87000-87575 | 576 | 4 | 96 | **192** | **48** | 3 | 454,464 |

Block fresh against every other profile and every committed summary (checked by a test).
One thing is deliberately *not* changed: the learning rate stays at the registered value,
so this is not a tuned longer run - stated as a limit rather than tuned after the fact (the
5.5 discipline).

**Primaries.** Two, at the sibling arms' level: `alpha/2 = 0.025`, bar **313/576**.

1. **P1 (the collapse is the budget's).** The longer cell's collapse - its null-vs-random
   deltas, the quantity 5.50 measured at `+1.574` - is stochastically *smaller* than the
   48-step cell's: a one-sided rank-sum at `alpha/2`, the non-paired test 5.50 introduced
   and rehearsed on committed rungs (p `4.5e-44` on the measured dose step, `0.50` on the
   measured wash).
2. **P2 (value creation survives the budget).** adaptive-vs-null clears the bar **and** the
   mean delta is negative - 5.50's own per-cell conjunction.

**Pre-registered interpretation (fixed before running).**

- **P1 and P2:** the collapse was the budget and value creation survives it - 5.50's
  crossing is not a fixed-budget artifact, and the width story keeps its selection
  reading.
- **P1 only:** the collapse was the budget and the advantage goes with it - 5.50's crossing
  WAS the fixed-budget collapse, and the honest form of the width story is about
  trainability, not selection value.
- **P2 only:** the collapse is structural at width 96 - 4x training does not repair the
  unmutated arm - and value creation rides it.
- **Neither:** the longer budget neither relieves the collapse nor keeps the advantage.

Both branches were fixed before the run and the arm's timing pilot (two seeds, cost only,
seeds outside this block) ran with *both* outcomes on the table, so nothing here is
predicted in one direction. Descriptive secondaries: the controller's edge
(adaptive-vs-random, rank-sum against the 48-step cell) and the absolute levels.

**Cost.** ~4.7-5.3 s/seed measured at the new cell (three arms, one thread; 4x the steps of
the sibling cells): 576 seeds is ~48 min of single-thread CPU across 20 shards, minutes of
wall clock, no GPU quota. Registered before the run; the running revision is recorded in
the summary's `code_revision`.

**Outcome (run 2026-09-20, revision `a590a7f`, 20 shards; committed summary
`width-d4-dose2-long-576.json`). Registered branch 1: P1+P2 PASS: the collapse was the
budget and value creation survives it - 5.50's crossing is not a fixed-budget
artifact.** Built-in checks held in 576/576
seeds (the mutational schedule still 3/3/0), and three replays of committed seeds matched
the shards exactly.

| cell (depth 4, width 96) | steps | adaptive-vs-null | mean delta | collapse (null-vs-random) | controller edge (adaptive-vs-random) | arm means (adaptive/null/random) |
|---|---|---|---|---|---|---|
| 48-step (5.50) | 48 | 576/576 | `-1.7330` | `+1.5739` | `-0.1591` | `15.127 / 16.860 / 15.286` |
| **long (5.51)** | **192** | **576/576** | **`-0.7149`** | **`+0.7083`** | **`-0.0066`** | **`3.833 / 4.548 / 3.840`** |

- **P1 - the collapse was the budget's, by 55%.** The pool effect falls from `+1.574` to
  `+0.708` (rank-sum U `2925`, p `1.7e-183`) - relieved, but **not removed**: the
  unmutated arm is still the worst arm in the longer cell (`null 4.548` against
  `3.833`/`3.840`), so the substrate keeps the same shape at 4x the budget, just less
  extreme.
- **P2 - value creation survives, in its strongest form.** `576/576`, mean `-0.7149`, p
  `4.0e-174`: at a workable budget a 14.8x-wide substrate still clears the do-nothing bar
  at every seed.
- **Descriptive - the controller's edge is what the budget takes away.**
  `adaptive`-vs-`random` falls from `-0.159` to `-0.0066` (reverse rank-sum p `2.1e-45`):
  still detectable against zero (`335/576`, p `5.2e-5`), but 24x smaller. At *both*
  budgets the do-nothing comparison at 14.8x is carried by the mutational pool, and the
  controller's own share *shrinks* as the budget grows - so 5.50's "the dose buys escape
  from a collapse, not a sharper controller" holds at the longer budget in a stronger
  form, with the escape itself reduced.
- **The absolute-level limit, scoped.** 4x training moves the same arms from
  `15.1`-`16.9` to `3.8`-`4.5`, so 5.50's "capacity is a liability" is a *fixed-budget*
  statement and nothing more - but there is no 192-step width-24 control in this arm
  (pinned by a test), so no cross-width absolute claim can be made at the longer budget
  either. The learning rate was deliberately left at the registered value, so this is not
  a tuned longer run.
  **Closed 2026-09-21 by 5.52**, which registered and ran exactly that control: the
  collapse that survives 4x training is the wide substrate's, and the absolute-level
  ordering *inverts* with the budget (`15.13`/`15.29`/`16.86` at width 96 against
  `12.44`/`12.55`/`12.48` at width 24 at 48 steps, and `3.83`/`3.84`/`4.55` against
  `5.46`/`5.50`/`5.66` at 192). The learning-rate caveat stands at both widths.

**What this changes.** Open question (ii)'s answer keeps its mechanism and loses its
fixed-budget framing: the crossing at 14.8x is real at both budgets (5.50 and 5.51), the
collapse is largely a budget artifact, and the controller's share of the crossing is
small and *shrinking* with budget. 10.5(B)'s case that a width mutation is a package
effect therefore strengthens: at the longer budget the width story is entirely "mutate at
all".

### 5.52 Pre-registration: the long-budget width-24 control 5.51 named as absent (RAN - summary `width-d4-long-w24-576.json`)

5.51 registered its limit in the same breath as its result: at 14.8x the collapse is
*relieved by 55%* at 4x the budget (`+1.574` -> `+0.708`) and the arms move from `15.1`-`16.9`
to `3.8`-`4.5`, but "there is no 192-step width-24 control ... so no cross-width absolute
claim is made at the longer budget". This arm is that control, and it closes the gap on the
one axis such a claim needs: the same cell with only the width changed.

**Design.** One cell: 5.51's config with `hidden_size` 96 -> **24** and nothing else - depth
4, **192 steps**, **48**-step mutation interval (three events, asserted by the built-in checks
at 3/3/0), fresh block **87600-88175**. With the committed cells that gives the 2x2 the
collapse has to be read in:

| cell | section | block | n | depth | width | steps | interval | total params |
|---|---|---|---|---|---|---|---|---|
| `depth4-prefix` | 5.12 | 73000-73383 | 384 | 4 | 24 | 48 | 12 | 30,672 |
| `width-d4-dose2` | 5.50 | 84000-84575 | 576 | 4 | 96 | 48 | 12 | 454,464 |
| `width-d4-dose2-long` | 5.51 | 87000-87575 | 576 | 4 | 96 | 192 | 48 | 454,464 |
| `width-d4-long-w24` | **5.52** | **87600-88175** | 576 | 4 | 24 | **192** | **48** | 30,672 |

All four corners run the registered **prefix** candidate policy, which is what makes the two
widths comparable at both budgets: 5.10's depth-4 cell is the same width at the same budget
but runs the *spread* policy, so it is deliberately **not** the corner used here (pinned by a
test, because ranking against it would move width and policy at once). The block is fresh
against every other profile and every committed summary (checked by a test). The learning
rate and every other knob stay registered, so this is not a tuned longer run.

**Primaries.** Two, at the sibling arms' level: `alpha/2 = 0.025`, bar **313/576**.

1. **P1 (the residual collapse is the wide substrate's).** The width-24 cell's collapse - its
   null-vs-random deltas, the quantity 5.50 measured at `+1.574` and 5.51 at `+0.708` - is
   stochastically *smaller* than the width-96 long cell's: a one-sided rank-sum at `alpha/2`,
   the non-paired test 5.50 introduced and rehearsed on committed rungs.
2. **P2 (value creation survives at width 24).** adaptive-vs-null clears the bar **and** the
   mean delta is negative - the family's per-cell conjunction.

**Pre-registered interpretation (fixed before running).**

- **P1 and P2:** the collapse that survives 4x training is the wide substrate's and the
  narrow one creates value - so 5.51's relief is a *capacity* effect, and 5.50's dose ladder
  reads as a liability of capacity itself.
- **P1 only:** the collapse is the wide substrate's, but width 24 does not create value at the
  longer budget - the budget is not what the narrow substrate was short of.
- **P2 only:** the collapse is general across widths at 4x (both substrates churn away the
  unmutated arm) and the narrow cell still creates value.
- **Neither:** the longer budget neither relieves the narrow cell's collapse nor keeps its
  value.

**Descriptive secondaries.** The four corners' collapse means, the within-width relief (the
*difference in differences* - deliberately not a pre-registered test, because it is a
statistic on four independent cells), and the controller's own edge (adaptive-vs-random
against the width-96 long cell).

**Cost.** Measured at the new cell: ~4.6 s/seed (three arms, four threads; 4x the steps of the
sibling cells - the width-96 long cell measures 4.6 s/seed too, so it is the longer budget,
not the width, that this cost is). 576 seeds is ~44 min of single-thread CPU across 20
shards: minutes of wall clock, no GPU quota, run while the corrected Gate 4 session trains.
Registered before the run; the running revision is recorded in the summary's `code_revision`.

**Outcome (run 2026-09-21, revision `f636990`, 20 shards; committed summary
`width-d4-long-w24-576.json`). Registered branch 1: P1+P2 PASS.**

P1: the width-24 long cell's collapse is `+0.163809` against the width-96 long cell's
`+0.708276` - rank-sum U `34.0`, one-sided p `5.73e-190` - so **the collapse that survives 4x
training belongs to the wide substrate**, and 5.51's relief is a *capacity* effect rather
than a property of training. P2: the narrow cell creates value in the family's own
conjunction - `550/576` wins against the bar of `313`, mean `-0.204292`, p `3.50e-129`,
adaptive `5.4605` < random `5.5010` < null `5.6648` - so the longer budget is enough for the
*narrow* substrate to clear the bar too (it already did at 48 steps, `225/384`).

**The 2x2 is the finding, and its two halves point opposite ways.** Collapse is
null-vs-random (positive = the do-nothing arm is the worse one):

| cell | width | steps | collapse | adaptive-vs-null | own edge | arms (adaptive / random / null) |
|---|---|---|---|---|---|---|
| `depth4-prefix` | 24 | 48 | `-0.0701` | `225/384`, `-0.0371` | `-0.1072` | `12.4413 / 12.5485 / 12.4785` |
| `width-d4-dose2` | 96 | 48 | `+1.5739` | `576/576`, `-1.7330` | `-0.1591` | `15.1273 / 15.2864 / 16.8603` |
| `width-d4-long-w24` | 24 | **192** | `+0.1638` | `550/576`, `-0.2043` | `-0.0405` | `5.4605 / 5.5010 / 5.6648` |
| `width-d4-dose2-long` | 96 | **192** | `+0.7083` | `576/576`, `-0.7149` | `-0.0066` | `3.8333 / 3.8399 / 4.5482` |

4x training *relieves* the wide substrate's collapse by 55% (`+1.5739` -> `+0.7083`) and
*raises* the narrow one's (`-0.0701` -> `+0.1638`) - from a baseline where there was nothing
to relieve, because at width 24 the churn harm is already absent at 48 steps. So "more
training repairs the collapse" is not a property of training: it is the repair of a collapse
only capacity produces. (The within-width relief is the difference in differences,
`+0.2339` at width 24 against `-0.8656` at width 96 - descriptive, not a pre-registered test,
since it is a statistic on four independent cells.)

**And the absolute levels reverse with the budget** (descriptive, as every cross-cell level
reading in this family is): at 48 steps the wide substrate is the worse one (arms
`15.13`/`15.29`/`16.86` against `12.44`/`12.55`/`12.48`), and at 192 steps it is the better one
(`3.83`/`3.84`/`4.55` against `5.46`/`5.50`/`5.66`). That closes the limit 5.51 registered:
its "no cross-width absolute claim at the longer budget" now has the control it named, and
the claim the control supports is the strongest available form of 5.50's limit - *capacity is
a liability at the fixed 48-step budget, and given 4x the training the ordering inverts*.

**The controller's own edge is where the narrow substrate's value comes from** (descriptive).
At 192 steps the width-24 cell's adaptive-vs-random edge is `-0.0405` (`390/576`, p `6.2e-18`)
against the width-96 cell's `-0.0066` (`335/576`, p `5.2e-5`), a rank-sum at p `3.85e-16` that
the *narrow* cell's edge is the larger. So the value decomposes differently at each width: at
96 it is almost entirely the mutational package (the controller contributes ~1% of
`-0.7149`), at 24 the controller contributes ~20% of `-0.2043`. 5.51's "the value at 14.8x is
the mutational package" is a *wide-substrate* reading too.

**Scrutiny before write-up.** Built-in checks 576/576 (mutations 3/3/0, zero null search
FLOPs); three replays of committed seeds exact at `OMP_NUM_THREADS=1`; and the collapse, the
value and the edge were all computed before any of them appeared in prose. Scoped: the
learning rate is still the registered one at both widths, so this is not a tuned longer run,
and the two widths differ in parameter count by 14.8x - the control completes the 2x2 the
width line reads, it does not turn width into a matched-budget contrast (that is 5.48's
equal-parameter arm, at 48 steps).

### 5.52.1 Post-run correction: the committed labels named another arm's branches, and one statistic was never reproduced

Found 2026-09-21 by regenerating the committed summaries from their own shards instead of
reading them, in the same way 5.9's alpha slip was found. Three defects, one of them mine and
caught in the act. **No measured number moved:** every corrected field is a label or a level,
and every arm's statistics, branch and quoted figure recompute unchanged.

**1. Sections 5.48, 5.49 and 5.52 carried the 5.9 vocabulary's branch string.** That
vocabulary's two primaries are null-vs-random and adaptive-vs-null *of one cell*, so it can
describe the arms that registered that pair (5.9, 5.10, 5.12, 5.15) and no others. The width
arms registered a different pair - a per-cell conjunction on each of two cells (§5.48,
§5.49), a per-cell conjunction plus a rank-sum between cells (§5.50-§5.52) - and the labels
read backwards where it mattered most:

| summary | committed label | what that arm actually registered |
|---|---|---|
| `width-d3-equal` (5.48) | "branch 2 (1 PASS / 2 FAIL): the pass is substantially churn suppression" | **P2 only** - this cell does *not* create value (`252/576`, mean `+0.0176`) |
| `width-d3-dose` (5.49) | "branch 2 (1 PASS / 2 FAIL): ..." | **P2 only** - this cell does not create value either (`235/576`, mean `+0.0615`) |
| `width-d4-long-w24` (5.52) | "branch 3 (1 FAIL / 2 PASS): no churn harm on this substrate" | **branch 1: P1+P2 PASS** - and this cell's recorded finding is a *collapse* of `+0.163809`, not its absence |

The fix makes the style data rather than a section list: a profile declares
`per_cell_primary`, and `summarize` renders the arm's own primary from it (a fact the 5.50 fix
introduced for two arms only, while 5.48, 5.49 and 5.52 kept the old wording - the same "the
vocabulary describes a different pair of primaries" bug that fix was written to prevent).
Regenerating the five affected summaries moves exactly one line each; the 5.50 cells and 5.51
regenerate byte-identically.

**2. And the level was still reachable from the command line.** `summarize`'s CLI defaulted
the Bonferroni divisor to `2.0` for every profile, so regenerating one of the two 5.50 cells
wrote `bonferroni_alpha 0.025` and bar **313** where the registration states `alpha/3` and
bar **315** - silently, and *after* 5.50 had already recorded that exact class of
disagreement ("printing bar 318 where the registration said 315 ... a declaration whose bar
and alpha disagree cannot be applied as written"). This is not hypothetical: it is what the
first attempt at this very correction did. The divisor is now the profile's declaration
(`bonferroni_divisor`, required for all fourteen profiles) and a mismatched override is
refused before any shard is read, so the hole the 5.50 fix left is closed rather than
narrowed. With the declaration in place both 5.50 cells regenerate byte-identically at 315.

**3. Six committed summaries cannot be regenerated, and nothing was checking.** Recomputing
`delta_stddev` (the one comparison field no summary test recomputed) from the stored
`per_seed` triples reproduces **47 of 53** summaries bit-exactly - all eight width arms among
them - and misses exactly the six `null-control-*` records of sections 5.9-5.15, whose values
match `statistics.pstdev` instead: an accumulation the code has since replaced, and one that
is order-sensitive, so no single formula reproduces all six in the stored order either. The
drift is bounded at `2.8e-15` relative (~12 ulp), `delta_stddev` is quoted by no section and
read by no decision, and rewriting six committed records to make a statistic nobody reads
exact is not a repair - so it is **enumerated and bounded** by a test instead,
`tests/test_toy_summary_labels.py`, with the six files named so the vintage is a decision
rather than a discovered tolerance. The gap being closed is the missing recomputation, not
the last digits: before this, a change to any recorded statistic in that field would have
gone unnoticed.

**Pinned.** `test_the_label_style_is_the_arms_own_declaration`,
`test_every_committed_label_is_its_own_arms`,
`test_every_profile_declares_its_bonferroni_divisor` (declaration and committed bar agree),
`test_a_summary_cannot_be_written_at_a_level_its_registration_does_not_state`,
`test_the_committed_summary_regenerates` (byte-exact, with the six earlier records bounded to
`delta_stddev`), and `test_the_committed_stddevs_are_bounded_and_the_vintage_is_enumerated`.
Mutants: restoring the hard-coded section list fails 5 (exactly the five corrected summaries);
letting the divisor fall back to `2.0` fails 3, including the refusal itself.

### 5.53 Pre-registration: the *expert-count* covariate at the onset (RAN - summaries `experts-d3-dose-576.json`, `experts-d4-dose-576.json`)

Section 10.3(ii) named what the width line left open: the onset is bracketed to (3, 4], and the
covariates that could separate *depth* from *capacity* are "parameter count, layer width,
expert count". Parameter count and layer width are now measured - 5.48's equal-parameter width
partner does not move the gate, and neither do 5.49's or 5.50's doses until 14.8x - so **expert
count is the one that remains**, and it is not merely the last candidate: expert grow/prune is
the *only* operation the V0 search space can reach (section 10.4, pinned by
`tests/test_topology_transactions.py`), so whether expert count reads at the onset decides
whether the gene the search can actually move is the gene the ontology reads.

**Design.** Two fresh cells of the registered config with only `expert_count` changed, 4 -> 6,
holding `hidden_size` 24 and `active_experts` 2:

| cell | section | block | n | depth | width | E | total params | active params |
|---|---|---|---|---|---|---|---|---|
| `depth3` (committed, below onset) | 5.15 | 72000-73383 | 384 | 3 | 24 | 4 | 23,280 | 8,880 |
| `experts-d3-dose` | **5.53** | **88176-88751** | 576 | 3 | 24 | **6** | **30,624** | **9,024** |
| `depth4-prefix` (committed, above onset) | 5.12 | 73000-73383 | 384 | 4 | 24 | 4 | 30,672 | 11,472 |
| `experts-d4-dose` | **5.53** | **89384-89959** | 576 | 4 | 24 | **6** | **40,464** | **11,664** |

**The depth-3 cell is the family's tightest equal-parameter partner yet, and the first that buys
its parameters without buying any compute.** 30,624 total against the above-onset cell's 30,672
is **-0.156%**, where 5.48's width-28 partner was +1.695%. But a token still routes to two
experts, so its active (per-token) parameter count is **9,024 against 11,472: 21.3% lower** -
where the width partner matched *both* axes (+1.7% total, +2.0% active). The two arms therefore
ask different questions at the same parameter count: 5.48 asked whether **wider** is what depth
buys (matching everything), and this arm asks whether **stored capacity** is what the onset
reads while per-token compute goes *down*. No earlier cell in this program separates the two axes.

The depth-4 cell is the same dose above the gate (40,464 / 11,664, +31.9% / +1.7%), so the arm
reads the dose at both depths the way 5.49 did for width - and, as in 5.49, the below-gate cell
is the equal-parameter one and the above-gate cell is the continuation.

**The probe's candidate pool does not move with the cell** (pinned by a test). At
`probe_candidate_cap` 4 with the prefix policy the pool is
`[grow L0, prune L0E0, prune L0E1, prune L0E2]` at expert count 4 and at 6 alike, because the cap
binds at both counts (12 and 18 feasible candidates). So the treatment is the substrate's expert
count and not the candidate set the controller sees - without which the arm would move two
things at once.

**Primaries.** Two, at the sibling arms' level: `alpha/2 = 0.025`, bar **313/576**.

1. **P1 (the equal-parameter partner creates value).** The depth-3 depth-6 cell's
   adaptive-vs-null wins clear the bar **and** the mean delta is negative - the family's per-cell
   conjunction.
2. **P2 (the same dose above the gate creates value).** The depth-4 depth-6 cell's conjunction.

**Pre-registered interpretation (fixed before running).**

- **P1 and P2:** capacity is the covariate even when it is bought with experts, and it is
  *stored* capacity rather than per-token compute - the gate falls at an equal-parameter partner
  whose active parameter count is 21.3% **lower**, which no width cell could separate. Read with
  5.48, the matched-parameter form of open question (ii) is then answered in favour of capacity
  rather than depth per se, and depth's own contribution to the onset is what remains to be shown.
- **P1 only:** the expert dose buys value below the onset and destroys it above - a depth x
  expert-capacity interaction, with the reachable gene capacity-like only where there was no
  value to lose.
- **P2 only:** the gate does not move for expert count either, so at this scale the onset is
  depth per se against two measured covariates and the remaining candidate is total compute -
  recorded beside the fact that the search space's only reachable gene is one that does not read
  at the onset.
- **Neither:** expert capacity is harmful at both depths; the reachable axis is closed as a
  covariate too.

**Descriptive secondaries.** The four cells' arm levels and their churn readings
(null-vs-random, the axis the width line showed flipping sign: harm at <=1.34x, absent at 3.8x,
paying at 14.8x) and the controller's own edge (adaptive-vs-random) at each cell.

**Cost.** Measured at the new cells: ~1.95 s/seed (depth 3) and ~1.55 s/seed (depth 4) at four
threads, so two cells x 576 seeds is ~34 min of single-thread CPU across 40 shards - minutes of
wall clock, no GPU quota, run while the corrected Gate 4 session trains. The two cells' blocks are
the first free 576-windows at or after 88000 (5.52 holds 87600-88175), checked by a test against
every other profile and every committed summary. Registered before the run; the running revision
is recorded in each summary's `code_revision`.

**Outcome (run 2026-09-21, revision `76d2134`, 20 shards per cell; committed summaries
`experts-d3-dose-576.json`, `experts-d4-dose-576.json`). Registered branch 4: both FAIL.**

Neither cell clears the bar at `alpha/2` (bar 313/576), and the failure above the gate is the
finding:

| cell | total / active params | adaptive-vs-null | mean delta | collapse | controller's edge | best arm |
|---|---|---|---|---|---|---|
| `depth3` (5.15, below onset) | 23,280 / 8,880 | 188/384 | `+0.004643` | `-0.106794` | `-0.102151` | null |
| `width-d3-equal` (5.48) | 31,192 / 11,704 | 252/576 | `+0.017638` | `-0.148365` | `-0.130727` | null |
| **`experts-d3-dose`** | **30,624 / 9,024** | **211/576** | **`+0.059544`** | `-0.137169` | `-0.077625` | null |
| `depth4-prefix` (5.12, above onset) | 30,672 / 11,472 | 225/384 | `-0.037147` | `-0.070072` | `-0.107220` | **adaptive** |
| **`experts-d4-dose`** | **40,464 / 11,664** | **260/576** | **`+0.019507`** | `-0.095310` | `-0.075803` | **null** |

**P1 fails as the width arm failed.** At 30,624 total parameters - the family's tightest
equal-parameter partner, `-0.156%` against the above-onset cell's 30,672 - depth 3 does not
create value (`211/576`, mean `+0.059544`, p `1.000`), and it is *further* from the bar than the
committed below-onset cell already was (`188/384`). So the matched-parameter form of open
question (ii) now has two answers that agree: the gate does not move for layer width (5.48) and
it does not move for expert count. The remaining candidate is total compute.

**P2 fails in the opposite direction, and that is the result.** At E=4 the above-onset cell
creates value and the adaptive arm is its *best* arm (`12.4413` < `12.4785` < `12.5485`). At E=6
the same depth-4 cell stops creating value (`260/576`, mean `+0.019507`, p `0.991`) and the
ordering inverts - the do-nothing arm becomes the best one (`12.5379` < `12.5574` < `12.6332`).
The dose is a real, strong effect rather than noise: the E=4 cell's per-seed deltas are
stochastically *more negative* than the E=6 cell's at rank-sum p `9.1e-09` (depth 4) and
p `8.0e-09` (depth 3, against `depth3`), and at both depths in the direction that means the
extra experts *remove* value creation rather than add it.

**And it is not churn.** The collapse (null-vs-random) stays negative at every one of the five
cells - there is no churn harm anywhere in this covariate - and the controller's own edge over
the pool-matched random arm stays negative and significant at both new cells (`391/576`,
`-0.077625`, p `2.9e-18` at depth 3; `372/576`, `-0.075803`, p `1.2e-12` at depth 4). So the
controller still *selects*: what dies is its value relative to doing nothing. At E=6 the null
arm beats the mutating arms at both depths - the substrate's own extra capacity is worth more
than any of the three mutation policies layered on it.

**What this closes, and what it says about the search space.** Section 10.3(ii) named
"parameter count, layer width, expert count" as the covariates that could separate depth from
capacity, and all three are now measured at this scale: two do not move the gate, and the third
- expert count, the one that *is* a measured covariate here - moves it the wrong way above the
onset. That sharpens 10.4 rather than restating it. 10.4 observed that expert grow/prune is the
only operation the V0 search space can reach, and that the architecture track's measured lever
(residual width) is unreachable from it. This arm adds the complementary reading on the other
track's own terms: the reachable gene is not merely a different covariate from the one the
ontology reads, it is a covariate that reads *against* value creation when it is raised. A V0
search that grows experts is therefore not asking a question this program has measured to have
an answer - which is a bound on what a Gate 4 pass would license, not a verdict on Gate 4, whose
question is about *selection* against a null on a declared block and whose null clause is the
right control for exactly this kind of premise.

Scoped: the expert count is a property of the *substrate* in this arm, as it is in every 5.4x
cell - no run changed its expert count mid-flight, and the search's own grow/prune semantics are
untouched - so this bounds what a wider expert pool is worth at a fixed budget, and does not
measure whether a search that grows its own experts does better. The learning rate is the
registered one at every cell, and the 576-seed cells are read against the 384-seed committed
ones descriptively, as every cross-cell number in this family is.

**Scrutiny before write-up.** Built-in checks 576/576 per cell (mutations 3/3/0, zero null
search FLOPs); six replays of committed seeds (three per cell) exact at `OMP_NUM_THREADS=1`; and
the cell statistics, the five-cell table, the ordering inversion and the two dose rank-sums were
all computed before any of them entered prose.

### 5.54 Pre-registration: the *total-compute* covariate, in the 2x2's last empty corner (RAN - summary `depth3-long-576.json`)

10.3(ii) named three covariates that could separate depth from capacity - "parameter count, layer
width, expert count" - and then added a fourth thing the onset could be: **total compute**. Width
was closed by 5.48-5.52 (the matched-parameter partner does not move the gate; the doses pay only
at 14.8x) and expert count by 5.53 (the tightest equal-parameter partner does not move the gate
either, and the dose *removes* value above it). Budget is what remains, and it is the one
covariate that is not a property of the substrate at all.

This arm runs the **one corner of the depth x budget 2x2 at width 24 that is still empty.** Three
of its four corners are already committed:

| corner | steps | total / active params | adaptive-vs-null | collapse (null-vs-random) |
|---|---|---|---|---|
| `depth3` (5.15) | 48 | 23,280 / 8,880 | 188/384, `+0.004643` | `-0.106794` |
| **`depth3-long`** | **192** | **23,280 / 8,880** | *this arm* | *this arm* |
| `depth4-prefix` (5.12) | 48 | 30,672 / 11,472 | 225/384, `-0.037147` | `-0.070072` |
| `width-d4-long-w24` (5.52) | 192 | 30,672 / 11,472 | 550/576, `-0.2043` | `+0.1638` |

**Design.** One fresh cell of the registered config with only the budget changed: `depth` 3, `steps`
48 -> 192, `mutation_interval` 12 -> 48, holding `hidden_size` 24 and `expert_count` 4. That is
5.52's exact treatment (both multiplied by four) applied one depth lower, so the same three
mutational events run over four times the training. It is the family's **first arm that moves
neither a parameter nor the search space** - 5.48 moved width, 5.53 moved experts, and this moves
only the step count - which is what makes it the compute covariate rather than a second reading of
capacity.

**Why the regime primary, and not a third value primary.** 5.52 already measured what 4x training
does at depth 4: the collapse goes `-0.0701` -> `+0.1638`, a churn **harm** appearing where none
was, while the value reading improves (`-0.0371` -> `-0.2043`). So the informative question one
depth below the gate is not "does more training help" - it did, above the gate - but whether the
budget **reaches the regime** there: if 4x training is what makes a substrate churn, the
below-onset cell should start churning too even though it still creates no value.

**Primaries.** Two, at the sibling arms' level: `alpha/2 = 0.025`, bar **313/576**.

1. **P1 (value below the onset at 4x training).** `depth3-long`'s adaptive-vs-null wins clear the
   bar **and** the mean delta is negative - the family's per-cell conjunction.
2. **P2 (the budget moves the churn regime).** One-sided rank-sum of `depth3-long`'s null-vs-random
   deltas against the committed 48-step `depth3` cell's, asking whether the longer cell's deltas
   are **stochastically larger** (a churn harm). Deliberately a non-paired test between two
   independent 576-seed cells, and the family's first: the two cells share no seed, so there is no
   pairing to exploit.

**The rank-sum's sensitivity was rehearsed on the one measured instance before this arm ran.** At
depth 4 the same 48 -> 192 comparison gives U 28795, p 1.9e-84 on the measured flip. So the ruler
detects the regime change where the records already say one exists - P2 is only informative if it
is not blind to the thing it measures - and the pinned direction convention (short cell first)
means a reversed call reads as a dead null (p > 0.999), not as a second detection.

**Pre-registered interpretation (fixed before running).**

- **P1 and P2:** **total compute is the covariate** - 4x training creates value below the onset
  *and* brings the churn regime with it. That would make every below-onset cell's failure a
  fixed-budget statement, and the onset a compute threshold rather than depth per se. Read with
  5.52, the whole 2x2 would then be compute-driven at both depths.
- **P1 only:** the gate is compute and the regime change belongs to depth 4 - 4x training buys
  value below the onset without making that substrate churn, so the below-onset cell is short of
  *both* and more training supplies one.
- **P2 only:** the budget reaches the *regime* below the onset but not the *gate* - a churn harm
  appears where none was and no value is created, so the budget is not what the below-onset cell
  is short of even though it is enough to make the substrate churn.
- **Neither:** 4x training changes nothing below the onset - no churn harm and no value. 10.3(ii)'s
  last candidate is then closed too, and the onset is depth per se at every covariate this search
  space can measure or vary.

**Descriptive secondaries.** The full 2x2's four corners (value, collapse and the controller's own
edge at each), so the budget reading can be compared across depths; and the controller's own edge
(adaptive-vs-random) at the new cell, as at every cell in this family.

**Scope.** The 576-seed cell is read against the 384-seed committed corners **descriptively**, as
every cross-cell number in this family is; no cross-cell arithmetic enters either primary. The
learning rate is the registered one at both budgets and is not rescaled with the step count, so
this arm measures the effect of more steps at a fixed schedule, not the effect of a tuned long-budget
schedule - recorded here because a rescaled schedule is exactly the change that would be expected
to help further.

**Cost.** Measured on this cell's own config before registering: ~3.29 s/seed at four threads, so
576 seeds is ~30 min of single-thread CPU across 20 shards - minutes of wall clock, no GPU quota,
run while the corrected Gate 4 session trains. The block is 90000-90575, the first free 576-window
at or after 90000 (5.53 holds 88176-88751 and 89384-89959), checked by a test against every other
profile *and* every committed summary's recorded seeds. Registered before the run; the running
revision is recorded in the summary's `code_revision`.

**Outcome (run 2026-09-21, revision `64046da`, 20 shards; committed summary
`depth3-long-576.json`). Registered branch 4: both FAIL.**

Neither primary moves, and the *combined* 2x2 is the finding - because the same treatment that
moved two things at depth 4 moved nothing here:

| corner | steps | arms (adaptive / null / random) | value | collapse | own edge |
|---|---|---|---|---|---|
| `depth3` (5.15) | 48 | `13.4549 / 13.4503 / 13.5571` | `+0.004643` | `-0.106794` | `-0.102151` |
| **`depth3-long`** | **192** | **`6.3453 / 6.3063 / 6.4347`** | **`+0.039005`** | **`-0.128388`** | **`-0.089383`** |
| `depth4-prefix` (5.12) | 48 | `12.4413 / 12.4785 / 12.5485` | `-0.037147` | `-0.070072` | `-0.107220` |
| `width-d4-long-w24` (5.52) | 192 | `5.4605 / 5.6648 / 5.5010` | `-0.204292` | `+0.163809` | `-0.040483` |

**P1 fails, and it fails without moving at all.** At 192 steps depth 3 reaches `281/576` wins
against the bar of `313` (mean `+0.039005`, p `0.734`) - a win rate of `48.8%`, against the
committed 48-step cell's `49.0%` (`188/384`). Four times the training does not bring the
below-onset cell one step closer to the bar; the small move in the *mean* is in the direction
that would make it worse, and not significantly so (unpaired rank-sum of the value deltas, short
against long, p `0.082`).

**P2 fails, and it is the informative failure.** The collapse goes `-0.106794` -> `-0.128388` -
still negative, so the do-nothing arm is still the *better* one at both budgets, and the regime
has not moved (rank-sum U `112872.0`, one-sided p `0.706`). This is a non-detection that carries
weight precisely because the ruler is not blind: the identical comparison at depth 4 returns
U `28795.0`, p `1.9e-84`, and the reversed call at depth 3 reads as a dead null (p `0.294`), so
the test is measuring the thing it was registered to measure.

**The finding is that the budget's effect is itself gated by depth.** The same treatment - both
multiplied by four, the same three mutational events, the registered learning rate - does two
large things one depth up and neither of them here:

| 48 -> 192 steps | value (adaptive-vs-null rank-sum) | collapse (null-vs-random rank-sum) |
|---|---|---|
| **depth 4** | `-0.037147` -> `-0.204292`, p `2.6e-60` (better) | `-0.070072` -> `+0.163809`, p `1.9e-84` (regime moves) |
| **depth 3** | `+0.004643` -> `+0.039005`, p `0.082` (no move) | `-0.106794` -> `-0.128388`, p `0.706` (no move) |

So the budget is not a covariate that the below-onset cell is short of *or* one that damages it:
once there is a gate to cross, 4x training crosses further and brings a churn regime with it;
where there is no gate, 4x training is inert. Section 10.3(ii) named parameter count, layer
width, expert count and total compute, and 5.48-5.54 have now measured all four at this scale:
**three do not move the gate, one removes value above it, and the one that does move it is the
axis 10.1 identified in the first place.**

**Descriptive secondaries, and they read exactly as 5.53's damage cells did.** At 192 steps
depth 3's best arm is the do-nothing one (`6.3063` < `6.3453` < `6.4347`), the collapse is
negative with `null_vs_random` at `400/576` (p `2.4e-21`), and the controller's own edge over the
pool-matched random arm survives strongly (`443/576`, `-0.089383`, p `3.4e-40`). So the controller
still *selects* - this is not a broken search - what it cannot do is beat doing nothing, at either
budget, on this substrate.

**Scrutiny before write-up.** Built-in checks 576/576 (mutations 3/3/0, zero null search FLOPs);
three replays of committed seeds (90003, 90317, 90571) exact at `OMP_NUM_THREADS=1`; both
primaries, both rank-sums, the 2x2 levels and the arm orderings were computed before any of them
entered prose; and the P2 direction was rehearsed on the measured depth-4 flip *before* the arm
ran, which is what licenses reading p `0.706` as an absence rather than as a blind test.

**Scoped, and the dose caveat matters here.** The only dose that has ever crossed the gate is
14.8x of *width* (5.50); a 14.8x step increase is not measured, and the two doses are not
comparable, so the claim this arm licenses is the one it registered for - 4x training is inert
below the onset and not inert above it - and not a statement about the budget as such.

### 5.55 Pre-registration: the *depth dose* above the onset (RAN - summary `depth5-dose-576.json`)

The covariate programme closed with 5.54, and it closed on the axis it began from. Width was
dosed (24 -> 28 in 5.48, -> 96 in 5.49/5.50), expert count was dosed (4 -> 6 in 5.53), the
budget was dosed (48 -> 192 in 5.51/5.52/5.54). **Depth - the one covariate that actually moves
the gate - has only ever had its bracket measured**, and the bracket is one step wide:

| cell | depth | steps | total / active params | adaptive-vs-null | collapse |
|---|---|---|---|---|---|
| `depth2` (5.15) | 2 | 48 | 15,888 / 6,288 | 161/384, `+0.020811` | `-0.099659` |
| `depth3` (5.15) | 3 | 48 | 23,280 / 8,880 | 188/384, `+0.004643` | `-0.106794` |
| `depth4-prefix` (5.12) | 4 | 48 | 30,672 / 11,472 | 225/384, `-0.037147` | `-0.070072` |
| **`depth5-dose`** | **5** | **48** | **38,064 / 14,064** | *this arm* | *this arm* |

So the project's central statement - the gate is depth per se, at every covariate this search
space can measure or vary - currently rests on a comparison between depth 3 and depth 4 with
**nothing measured above the upper cell**. Above the onset at depth 4 there may be a slope (deeper
keeps paying, and a depth-4 pass is a point on an axis) or a cliff edge (depth 4 is the only depth
that works, and the pass is a knife-edge). Nothing in the program distinguishes the two.

**Design.** One fresh cell of the registered config with `depth` 4 -> 5 and **nothing else
changed** - same width 24, same expert count 4, same 48 steps, same 12-step interval, same cap,
same policy. This is the family's **cleanest dose**: 5.48 moved width, 5.53 moved experts, 5.54
moved the budget, and this moves one key. It is deliberately **not** a matched-parameter contrast,
and that is the right reading rather than a limitation: every rung of the ladder adds exactly
7,392 total / 2,592 active parameters, so a fifth layer carries its parameters irreducibly - and
5.48 and 5.53 already measured that buying parameters with width or with experts does not move the
gate. So this arm reads the depth axis **as a package**, which is the form the claim "the gate is
depth" takes on the substrate the search actually runs.

**Primaries.** Two, at the sibling arms' level: `alpha/2 = 0.025`, bar **313/576**.

1. **P1 (the gate stays open above the onset).** The depth-5 cell's adaptive-vs-null wins clear
   the bar **and** the mean delta is negative - the family's per-cell conjunction. Its failure
   would mean the gate is a narrow window at depth 4 rather than a threshold.
2. **P2 (a fifth layer creates *more* value than a fourth).** One-sided rank-sum of the depth-5
   cell's adaptive-vs-null deltas against the committed above-onset cell's, smaller meaning
   deeper is better. Non-paired, two independent cells, as 5.54's regime primary is.

**P2's sensitivity was rehearsed on a measured instance of the same comparison shape before this
arm ran.** The shape is "two independent cells of this family, same depth and width, one given a
strictly better substrate, and the test has to see that on the value deltas". The measured
instance is the budget dose at depth 4: `depth4-prefix` -> `width-d4-long-w24` moves the value
reading from `-0.037147` to `-0.204292`, and the comparison returns U `41829`, p `2.6e-60` in the
direction this primary registers. So P2 is a test known to detect a substrate that creates more
value where the records say one exists - without which its failure here would be uninformative -
and its direction is pinned (the reversed call reads as a dead null, p > 0.999).

**Pre-registered interpretation (fixed before running).**

- **P1 and P2:** **the gate stays open above the onset and the axis pays** - a fifth layer clears
  the bar *and* creates more value than a fourth. The onset is then a slope on both sides, and a
  Gate 4 pass at depth 4 is a point on a scaling axis rather than a knife-edge.
- **P1 only:** **the onset is a threshold at depth 4** - deeper keeps the gate open but buys
  nothing more. The step is real and the slope is not, which is the reading that matters most for
  Gate 4: the pass generalises in the sense of not being fragile, and there is nothing to gain by
  deepening either.
- **P2 only:** the layer creates more value than a fourth *and* still does not clear the bar on
  its own, so the upper edge of the gate is fragile - recorded as a fragile-upper-edge reading
  rather than as either of the two above.
- **Neither:** **the gate is not monotone in depth** - a fifth layer fails where a fourth passes,
  so the onset is a narrow window at depth 4. That is the strongest statement this family could
  make about Gate 4's fragility and it is recorded as such, with the consequence that nothing here
  licenses reading a depth-4 pass as a property of "more depth".

**Descriptive secondaries.** The full depth ladder (depths 2, 3, 4 and this cell) with value,
collapse and the controller's own edge at each, so the ladder can be read as a curve rather than
as a pair of cells.

**Cost.** Measured on this cell's own config before registering: ~1.51 s/seed at four threads
(against depth 4's ~1.23 s/seed on the same machine), so 576 seeds is ~14.5 min of single-thread
CPU across 20 shards - roughly four minutes of wall clock, no GPU quota, run while the corrected
Gate 4 session trains. The block is 91384-91959, the first free 576-window at or after 91000,
checked by a test against every other profile *and* every committed summary's recorded seeds.
Registered before the run; the running revision is recorded in the summary's `code_revision`.

**Outcome (run 2026-09-21, revision `5fb7bd3`, 20 shards; committed summary
`depth5-dose-576.json`). Registered branch 1: P1+P2 PASS.**

P1: the depth-5 cell clears the bar (`375/576` against `313`, mean `-0.075335`, p `1.89e-13`), so
**the gate is open above the onset** - a fifth layer creates value at the same width, expert
count, step count, interval and policy as the fourth. P2: its value deltas are stochastically
better than the committed above-onset cell's (rank-sum U `97584.0`, one-sided p `9.99e-04` against
the `0.025` bar), so **a fifth layer creates more value than a fourth**. The onset is therefore
**a slope on both sides**, and a depth-4 pass is a point on a scaling axis rather than a knife-edge.

**The depth ladder is monotone, and that is the finding.** Value is adaptive-vs-null (negative is
better), collapse is null-vs-random (negative = the do-nothing arm is the better one):

| cell | depth | n | value | controller's edge | collapse | best arm | arms (adaptive / null / random) |
|---|---|---|---|---|---|---|---|
| `depth2` (5.15) | 2 | 384 | `+0.020811` | `-0.078848` | `-0.099659` | null | `14.3953 / 14.3745 / 14.4742` |
| `depth3` (5.15) | 3 | 384 | `+0.004643` | `-0.102151` | `-0.106794` | null | `13.4549 / 13.4503 / 13.5571` |
| `depth4-prefix` (5.12) | 4 | 384 | `-0.037147` | `-0.107220` | `-0.070072` | **adaptive** | `12.4413 / 12.4785 / 12.5485` |
| **`depth5-dose`** | **5** | **576** | **`-0.075335`** | `-0.104561` | `-0.029226` | **adaptive** | **`11.4870 / 11.5624 / 11.5916`** |

Four rungs, one direction, no reversal: `+0.0208`, `+0.0046`, `-0.0371`, `-0.0753`. The increments
are `-0.0162` (2 -> 3), `-0.0417` (3 -> 4, the largest) and `-0.0382` (4 -> 5) - so the gate is a
steep step rather than a discontinuity in kind, and **the axis that moves the gate keeps paying
past it.** The arm ordering tells the same story from the other side: the do-nothing arm is the
best arm at depths 2 and 3 and the adaptive arm is the best at 4 and 5.

**And the improvement above the onset is not the controller.** Decomposing
`adaptive - null = (adaptive - random) - (null - random)` at the two cells the arm compares:

| 4 -> 5 layers | mean | rank-sum (4 -> 5) | reading |
|---|---|---|---|
| controller's edge (adaptive-vs-random) | `-0.107220` -> `-0.104561` | U `109725.0`, p `0.418` | **flat** - no move, in either direction (reversed p `0.582`) |
| collapse (null-vs-random) | `-0.070072` -> `-0.029226` | U `95041.0`, p `1.1e-04` | the churn term moves |

So the controller's own edge **saturates by depth 3** (`-0.0788`, `-0.1022`, `-0.1072`, `-0.1046`
across the ladder) and never moves again, and what a fifth layer buys is a *substrate* effect: the
unmutated form gets relatively worse against random churn (`-0.0701` -> `-0.0292`) while the
controller's advantage over that churn stays exactly where it was. That is the same decomposition
5.50 found at the width crossing - where the crossing was the pool effect and the controller's
edge stayed flat - now measured on the depth axis, and it means **the ladder's value is not
evidence that the controller gets better with depth.**

**What this settles about the project's central claim.** 5.48-5.54 closed every covariate
10.3(ii) named and left "the gate is depth per se". 5.55 shows that claim has a shape: the axis
is monotone in depth from 2 to 5, the largest single step is the one that crosses the gate, and
the step after it is nearly as large. A Gate 4 pass at depth 4 is therefore not a knife-edge
reading - the same controller on a deeper substrate of the same block would create *more* value,
not less - and equally it is not evidence of a controller whose skill grows: what grows is the
substrate's dependence on being mutated at all.

**Scrutiny before write-up.** Built-in checks 576/576 (mutations 3/3/0, zero null search FLOPs);
three replays of committed seeds (91385, 91650, 91958) exact at `OMP_NUM_THREADS=1`; both
primaries, both decomposition rank-sums, the ladder levels, the increments and the arm orderings
were computed before any of them entered prose; and P2's ruler was rehearsed on a measured
instance of the same comparison shape *before* the arm ran (U `41829`, p `2.6e-60`), which is what
licenses reading its result here as a detection rather than as an artifact of the ruler.

**Scoped.** The fifth layer carries 7,392 total / 2,592 active parameters irreducibly (every rung
of the ladder adds exactly those), so this is the depth axis *as a package* and not a
matched-parameter contrast - deliberately, since 5.48 and 5.53 measured that buying parameters
with width or experts does not move the gate. The ladder's rungs are 384-seed committed cells read
against a 576-seed cell descriptively; no cross-cell arithmetic enters either primary. The
learning rate, step count, interval and policy are the registered ones at every rung, and the
toy path's fidelity bounds are unchanged from the rest of the family.

### 5.56 Pre-registration: the confirmation rung above the depth dose (RAN - summary `depth6-dose-576.json`)

5.55 established that the depth axis keeps paying past the gate, and it did so with a **two-point
slope**: depth 4 creates value and depth 5 creates more. Two points fit a rising line and they also
fit a line that has already turned over between them, so the claim "the axis keeps paying" is
currently an extrapolation. This arm is the third point, and it is registered as a *confirmation*:
either outcome bounds the claim rather than only one of them doing so.

| cell | depth | steps | total / active params | adaptive-vs-null | collapse |
|---|---|---|---|---|---|
| `depth2` (5.15) | 2 | 48 | 15,888 / 6,288 | 161/384, `+0.020811` | `-0.099659` |
| `depth3` (5.15) | 3 | 48 | 23,280 / 8,880 | 188/384, `+0.004643` | `-0.106794` |
| `depth4-prefix` (5.12) | 4 | 48 | 30,672 / 11,472 | 225/384, `-0.037147` | `-0.070072` |
| `depth5-dose` (5.55) | 5 | 48 | 38,064 / 14,064 | 375/576, `-0.075335` | `-0.029226` |
| **`depth6-dose`** | **6** | **48** | **45,456 / 16,656** | *this arm* | *this arm* |

**Design.** One fresh cell of the registered config with `depth` 5 -> 6 and **nothing else
changed** - the same one-gene dose as 5.55, one layer further up. Every rung of the ladder adds
exactly 7,392 total / 2,592 active parameters, so this remains the depth axis *as a package*
rather than a matched-parameter contrast, deliberately, for the reasons 5.55 records.

**And the candidate pool does not move with depth, which is what keeps the dose single.** At
`probe_candidate_cap` 4 the pool is `[grow L0, prune L0E0, prune L0E1, prune L0E2]` at depths 3,
4, 5 and 6 alike, because the cap binds at every rung (15, 20, 25 and 30 feasible candidates).
Pinned by a test over the whole ladder, so the treatment is the substrate and not the set the
policy chooses from - the check 5.53 had to make for the expert dose, made here for depth.

**Primaries.** Two, at the sibling arms' level: `alpha/2 = 0.025`, bar **313/576**.

1. **P1 (the gate is still open).** The depth-6 cell's adaptive-vs-null conjunction. Its failure
   at the *third* rung above the onset is the strongest fragility reading this family can produce.
2. **P2 (a sixth layer creates more value than a fifth).** One-sided rank-sum of the depth-5
   cell's adaptive-vs-null deltas against this cell's, smaller meaning deeper is better. It asks
   whether the slope continues or flattens, and its ruler is 5.55's - rehearsed on a measured
   instance of the same comparison shape before that arm ran (U `41829`, p `2.6e-60`) - so a
   failure here is a real plateau rather than a ruler that cannot see a step.

**Pre-registered interpretation (fixed before running).**

- **P1 and P2:** **the slope continues** - the ladder is monotone across all three rungs above the
  onset (4, 5 and 6), so a depth-4 pass is a point on a rising curve rather than a first step that
  flattens, and the axis has shown no sign of saturation over 2 of its 3 steps.
- **P1 only:** **the axis saturates at depth 5** - a sixth layer keeps the gate open but buys
  nothing more than a fifth, so the rising part of the curve is the gate itself and one step past
  it. That bounds what deepening buys without weakening the Gate 4 reading.
- **P2 only:** a sixth layer creates more value than a fifth *and* still does not clear the bar on
  its own - recorded as a fragile upper reading rather than as either of the two above.
- **Neither:** **the ladder is not monotone past depth 5** - a sixth layer fails where a fifth
  passes. The monotone claim is then bounded to depths 2..5, and since 5.55's own result is the
  one being bounded, that is recorded against 5.55's scoping as well as here.

**Descriptive secondaries.** The ladder to depth 6 with value, collapse, the controller's own edge
and the best arm at each rung; and **which term moves** between the two top rungs, decomposed as
5.55 decomposed 4 -> 5 (`adaptive - null = (adaptive - random) - (null - random)`), where the
whole 4 -> 5 gain was the churn term (edge rank-sum p `0.418`, churn rank-sum p `1.1e-04`). The
absolute arm levels are quoted only as cross-cell descriptives, which is this family's convention.

**Scope.** The 384-seed rungs at depths 2-4 are read against the 576-seed rungs at 5 and 6
descriptively; no cross-cell arithmetic enters either primary. P2 compares two 576-seed cells and
is non-paired (different blocks), as 5.54's and 5.55's rank-sums are. The learning rate, step
count, interval, cap and policy are the registered ones at every rung.

**Cost.** Measured on this cell's own config before registering: ~1.77 s/seed at four threads
(against depth 5's ~1.51 s/seed on the same machine), so 576 seeds is ~17 min of single-thread CPU
across 20 shards - about five minutes of wall clock, no GPU quota, run while the corrected Gate 4
session trains. The block is 92384-92959, the first free 576-window at or after 92000, checked by a
test against every other profile *and* every committed summary's recorded seeds. Registered before
the run; the running revision is recorded in the summary's `code_revision`.

**Outcome (run 2026-09-21, revision `9f31495`, 20 shards; committed summary `depth6-dose-576.json`).
Registered branch 1: P1+P2 PASS.**

P1: the depth-6 cell clears the bar decisively (`442/576` against `313`, mean `-0.117494`,
p `1.13e-39`) - the gate is still open at the third rung above the onset. P2: its value deltas are
stochastically better than the depth-5 cell's (rank-sum U `142035.0`, one-sided p `1.20e-05` against
the `0.025` bar), so **the slope continues**. The ladder is monotone across every rung measured:

| cell | depth | n | value | controller's edge | collapse | best arm | arms (adaptive / null / random) |
|---|---|---|---|---|---|---|---|
| `depth2` (5.15) | 2 | 384 | `+0.020811` | `-0.078848` | `-0.099659` | null | `14.3953 / 14.3745 / 14.4742` |
| `depth3` (5.15) | 3 | 384 | `+0.004643` | `-0.102151` | `-0.106794` | null | `13.4549 / 13.4503 / 13.5571` |
| `depth4-prefix` (5.12) | 4 | 384 | `-0.037147` | `-0.107220` | `-0.070072` | **adaptive** | `12.4413 / 12.4785 / 12.5485` |
| `depth5-dose` (5.55) | 5 | 576 | `-0.075335` | `-0.104561` | `-0.029226` | **adaptive** | `11.4870 / 11.5624 / 11.5916` |
| **`depth6-dose`** | **6** | **576** | **`-0.117494`** | `-0.089482` | **`+0.028013`** | **adaptive** | **`10.5882 / 10.7057 / 10.6777`** |

**Above the onset the slope is a constant, not a decaying one.** The increments are `-0.0162`
(2 -> 3), `-0.0418` (3 -> 4), `-0.0382` (4 -> 5) and `-0.0422` (5 -> 6) - so past the first step
the axis buys **roughly `0.04` nats per added layer**, and the newest step is the largest since the
gate itself. 5.55 could only say the slope had not yet turned over between two points; three points
say it is not turning over, and the flatness of the increments is the strongest form of that.

**And the churn regime flips at depth 6.** The collapse (`null - random`, positive = the do-nothing
arm is the worse one) goes `-0.099659`, `-0.106794`, `-0.070072`, `-0.029226`, **`+0.028013`**: the
first rung of the ladder where doing nothing is *worse* than churning blind. The arm levels say it
plainly - at depth 6 the order is `10.5882` (adaptive) < `10.6777` (random) < `10.7057` (null),
where at every earlier rung the null arm was second or first. So the depth axis does not merely make
the controller better; it makes the **unmutated form** progressively less viable, and by depth 6
mutation of some kind is necessary rather than optional.

**The decomposition over the whole upper ladder is the result.** Value is
`adaptive - null = (adaptive - random) - (null - random)`, and across depth 3 -> 6:

| term | depth 3 | depth 6 | rank-sum | reading |
|---|---|---|---|---|
| controller's own edge | `-0.102151` | `-0.089482` | U `107963.0`, p `0.266` (reversed `0.734`) | **flat - no move** |
| churn term | `-0.106794` | `+0.028013` | U `63764.0`, p `4.7e-29` | moves, decisively |

The value improvement is `-0.122137`. Decomposed: the controller's edge contributes **`+0.012670`**
- i.e. it gets slightly *worse* - and the churn term contributes **`-0.134807`**, i.e. all of it and
more. The same holds for the 5 -> 6 step alone (edge p `0.153`, flat; churn p `5.5e-08`). So over
four layers of a substrate whose value improves by `0.12` nats, **the controller's own contribution
is flat to slightly negative and the entire gain is the substrate's growing dependence on being
mutated.** 5.50 found this shape at the width crossing and 5.55 found it at the first depth step;
with three points above the gate it is now a property of the axis rather than of one comparison,
and it is the sharpest form the program's central decomposition has taken.

**What this settles.** 5.54 established that the gate is depth per se, 5.55 that the axis keeps
paying past it, and this arm that the slope is constant across all three rungs above the onset while
the controller's contribution stays where it was. A Gate 4 pass at depth 4 is therefore a point on a
rising curve with a measured gradient - and the thing that rises is **not** selection skill. Read
with 5.53's finding that the one gene the search can move reads *against* value, the two tracks say
the same thing from opposite sides: what this program's value tracks is the substrate's need for
mutation, which the search does not create and cannot enlarge.

**Scrutiny before write-up.** Built-in checks 576/576 (mutations 3/3/0, zero null search FLOPs);
three replays of committed seeds (92400, 92700, 92958) exact at `OMP_NUM_THREADS=1`; both primaries,
both decomposition rank-sums (and the whole-ladder pair), the increments, the arm orderings and the
algebraic identity were all computed before any entered prose, and the identity itself was checked
rather than assumed. P2's ruler is 5.55's, rehearsed on a measured instance of the same comparison
shape before that arm ran.

**Scoped.** The sixth layer carries 7,392 total / 2,592 active parameters irreducibly, so this is
the depth axis *as a package*; every rung is otherwise the registered config, and the candidate pool
is identical at every depth (pinned by a test). The rungs mix 384-seed committed cells with 576-seed
cells and are read against each other **descriptively** - the `0.04`-per-layer gradient is a
descriptive statement about cross-cell levels, in this family's convention, while both primaries are
tests on their own cells. The increments are measured atthe registered 48 steps and the registered schedule; the ladder is not bounded from above by
anything measured here.

### 5.57 Pre-registration: the fourth rung, against the constant-gradient model (RAN - summary `depth7-dose-576.json`)

5.56 turned the depth axis from a direction into a model. Three rungs above the gate give
increments of `-0.0418`, `-0.0382` and `-0.0422` - about `0.04` per layer, with no decay - and the
controller's own edge is flat across all three. **A model that predicts a number is testable in a
way a direction is not**, so this arm is registered against the model's own forecast rather than
against one more "deeper is better".

| cell | depth | steps | total / active params | adaptive-vs-null | collapse |
|---|---|---|---|---|---|
| `depth2` (5.15) | 2 | 48 | 15,888 / 6,288 | 161/384, `+0.020811` | `-0.099659` |
| `depth3` (5.15) | 3 | 48 | 23,280 / 8,880 | 188/384, `+0.004643` | `-0.106794` |
| `depth4-prefix` (5.12) | 4 | 48 | 30,672 / 11,472 | 225/384, `-0.037147` | `-0.070072` |
| `depth5-dose` (5.55) | 5 | 48 | 38,064 / 14,064 | 375/576, `-0.075335` | `-0.029226` |
| `depth6-dose` (5.56) | 6 | 48 | 45,456 / 16,656 | 442/576, `-0.117494` | `+0.028013` |
| **`depth7-dose`** | **7** | **48** | **52,848 / 19,248** | *this arm* | *this arm* |

**Design.** One fresh cell of the registered config with `depth` 6 -> 7 and **nothing else changed** -
the same one-gene dose as 5.55/5.56, one rung further up. The candidate pool is identical
(`[grow L0, prune L0E0, prune L0E1, prune L0E2]` at depths 3 through 7, pinned by a test over the
whole ladder), so only the substrate moves, and every rung adds exactly 7,392 total / 2,592 active
parameters, so this remains the depth axis as a package.

**Primaries.** Two, at the sibling rungs' level: `alpha/2 = 0.025`, bar **313/576**.

1. **P1 (the gate is still open):** the family's per-cell conjunction.
2. **P2 (a seventh layer creates more value than a sixth):** one-sided rank-sum of the depth-6
   cell's adaptive-vs-null deltas against this cell's.

**The declared prediction, made before the run and reported either way.** Cross-cell levels are
*descriptive* in this family, so a numerical forecast is not dressed as a third primary - but it is
written down before the measurement, which is what makes it falsifiable:

- **the increment** in `adaptive - null` from depth 6 to depth 7 falls in `(-0.047, -0.033)`: the
  measured range of the three existing increments (`0.0382`-`0.0422`, i.e. about `0.040`) widened by
  a flat `0.005` at each end, so the band is the model's number with a stated margin rather than a
  tolerance fitted to the result; and
- **the controller's own edge** stays within `0.020` of the depth-6 reading (`-0.089482`), because
  the model says the gain is the churn term and the edge does not move.

A miss is as reportable as a hit. A forecast that is checked and fails tells this program more than
one that was never asked, and the second prediction is the sharper of the two: it is the model's
central claim (the axis pays through the substrate, not the controller) stated as a number.

**Pre-registered interpretation (fixed before running).**

- **P1 and P2:** **the slope continues** - the ladder is monotone across all four rungs above the
  onset, and a depth-4 pass sits on a curve with four measured points.
- **P1 only:** **the axis saturates at depth 6** - a seventh layer keeps the gate open but buys
  nothing more than a sixth, so the rising part of the curve is the gate and two steps past it.
- **P2 only:** a seventh layer creates more value than a sixth *and* still does not clear the bar on
  its own - a fragile upper reading.
- **Neither:** **the ladder is not monotone past depth 6** - a seventh layer fails where a sixth
  passes.

**Descriptive secondaries.** The ladder to depth 7 (value, collapse, the controller's edge and the
best arm at each rung), the decomposition of the 6 -> 7 step into the edge and the churn term as
5.55-5.56 decomposed theirs, and the two declared predictions above.

**Scope.** The rungs mix 384-seed committed cells with 576-seed cells and are read against each
other descriptively; both primaries are tests on their own cells, and the two predictions are
descriptive readings of cross-cell levels, stated as such. The learning rate, step count, interval,
cap and policy are the registered ones at every rung.

**Cost.** Measured on this cell's own config before registering: ~2.09 s/seed at four threads,
so 576 seeds is ~20 min of single-thread CPU across 20 shards - minutes of wall clock, no GPU
quota, run while the corrected Gate 4 session trains. The block is 93384-93959, the first free
576-window at or after 93000, checked by a test against every other profile *and* every committed
summary's recorded seeds. Registered before the run; the running revision is recorded in the
summary's `code_revision`.

**Outcome (run 2026-09-21, revision `18ae507`, 20 shards; committed summary `depth7-dose-576.json`).
Registered branch 1: P1+P2 PASS - and the declared increment missed, in the direction that makes the
axis steeper than the model forecast.**

P1: the depth-7 cell clears the bar decisively (`503/576` against `313`, mean `-0.181042`,
p `2.92e-80`). P2: its value deltas are stochastically better than the depth-6 cell's (rank-sum
U `128927.0`, one-sided p `2.95e-11`, means `-0.181042` vs `-0.117494`), so **the slope continues**
and the ladder is monotone across all four rungs above the onset:

| cell | depth | n | value | controller's edge | collapse | best arm | arms (adaptive / null / random) |
|---|---|---|---|---|---|---|---|
| `depth2` (5.15) | 2 | 384 | `+0.020811` | `-0.078848` | `-0.099659` | null | `14.3953 / 14.3745 / 14.4742` |
| `depth3` (5.15) | 3 | 384 | `+0.004643` | `-0.102151` | `-0.106794` | null | `13.4549 / 13.4503 / 13.5571` |
| `depth4-prefix` (5.12) | 4 | 384 | `-0.037147` | `-0.107220` | `-0.070072` | **adaptive** | `12.4413 / 12.4785 / 12.5485` |
| `depth5-dose` (5.55) | 5 | 576 | `-0.075335` | `-0.104561` | `-0.029226` | **adaptive** | `11.4870 / 11.5624 / 11.5916` |
| `depth6-dose` (5.56) | 6 | 576 | `-0.117494` | `-0.089482` | `+0.028013` | **adaptive** | `10.5882 / 10.7057 / 10.6777` |
| **`depth7-dose`** | **7** | **576** | **`-0.181042`** | `-0.086622` | **`+0.094421`** | **adaptive** | **`9.7065 / 9.8876 / 9.7932`** |

**The declared prediction was half right, and the half that missed is the informative one.** The
increment from depth 6 to depth 7 is `-0.063548` against a declared band of `(-0.047, -0.033)`: a
**miss, and 1.5x every previous step of the upper ladder**. The four increments are `-0.0418`
(3 -> 4), `-0.0382` (4 -> 5), `-0.0422` (5 -> 6) and `-0.0635` (6 -> 7). The second prediction hit:
the controller's own edge measured `-0.086622` against the declared band of `0.020` around
`-0.089482`, a move of `+0.0029`.

So the model's *mechanism* survived a second rung and its *magnitude* did not - and the mechanism is
the sharper claim. Across 6 -> 7 the controller's own edge is flat (rank-sum p `0.663`) while the
churn term moves decisively (p `2.6e-11`); of the `-0.063548` the newest layer buys, the churn term
contributes `-0.066408` and the edge contributes `+0.002860`, i.e. slightly *against*. Extended over
the whole upper ladder, 3 -> 7:

| term | depth 3 | depth 7 | rank-sum | reading |
|---|---|---|---|---|
| controller's own edge | `-0.102151` | `-0.086622` | U `106031.5`, p `0.139` (reversed `0.861`) | **flat - no move** |
| churn term | `-0.106794` | `+0.094421` | U `40968.0`, p `9.0e-62` | moves, decisively |

The value improvement over those four added layers is `-0.185686`: the churn term carries
`-0.201215` of it and the controller's edge `+0.015529` against, so 5.56's decomposition holds one
rung further up and with a wider margin rather than narrowing.

**What the miss corrects.** 5.56 read three flat increments as "the slope is a constant, not a
decaying one". A fourth point shows that flatness was not yet the *shape*: the axis buys a flat
`~0.04` per layer for the steps ending at depths 4, 5 and 6, and **`0.0635` for the step ending at
depth 7**, so the constant-gradient reading is bounded to the three rungs that produced it and the
axis is super-linear at the fourth. Nothing here weakens the ladder claim - all four rungs above the
onset pass and the newest step is the largest of the five - but it changes what a deeper substrate
is worth: the model understated the axis rather than overstating it.

**The collapse accelerates with it.** `null - random` runs `-0.099659`, `-0.106794`, `-0.070072`,
`-0.029226`, `+0.028013`, `+0.094421`, so the do-nothing arm's disadvantage grows **3.4x** across
the newest step alone. At depth 7 the arm order is `9.7065` (adaptive) < `9.7932` (random) <
`9.8876` (null) - the order depth 6 established, at 1.5x the separation.

**What this settles.** A Gate 4 pass at depth 4 is a point on a curve rising faster than the
previous rung suggested, and what rises is still the substrate's dependence on being mutated rather
than the controller's skill: over four added layers the edge is flat at p `0.861` while the churn
term moves at p `9.0e-62`. Read with 5.53, whose one movable gene reads *against* value, the two
readings agree - what this program's selection value tracks is something the search neither creates
nor enlarges.

**Scrutiny before write-up.** Built-in checks 576/576 (mutations 3/3/0, zero null search FLOPs);
three replays of committed seeds (93404, 93704, 93944) exact at `OMP_NUM_THREADS=1`; both primaries,
the increment, both decomposition rank-sums and the algebraic identity were computed before any of
them entered prose. The declared bands and both predictions were committed at `80a28a2` and
`f6f5a08`, before the first shard ran.

**Scoped.** The seventh layer carries 7,392 total / 2,592 active parameters irreducibly, so this is
the depth axis *as a package*; the candidate pool is identical at every rung (pinned by a test); and
the increments are cross-cell descriptives in this family's convention, while both primaries are
tests on their own cells. The ladder is not bounded from above by anything measured here.

### 5.58 Pre-registration: the fifth rung, against the fourth increment itself (RAN - summary `depth8-dose-576.json`)

5.57 asked the constant-gradient model for a number and the number came back **1.5x too small**:
`-0.063548` against a declared `(-0.047, -0.033)`. One point past a flat series cannot separate an
**acceleration** from a **wobble** - a single increment `0.02` above the flat three is as consistent
with a noisy fourth step as with an axis that steepens - so this arm asks the same question with the
fourth increment itself as the model. That makes its declaration a choice between two readings
rather than a fresh guess, and the reading it is declared against is a *miss one full band away*.

| cell | depth | steps | total / active params | adaptive-vs-null | collapse |
|---|---|---|---|---|---|
| `depth2` (5.15) | 2 | 48 | 15,888 / 6,288 | 161/384, `+0.020811` | `-0.099659` |
| `depth3` (5.15) | 3 | 48 | 23,280 / 8,880 | 188/384, `+0.004643` | `-0.106794` |
| `depth4-prefix` (5.12) | 4 | 48 | 30,672 / 11,472 | 225/384, `-0.037147` | `-0.070072` |
| `depth5-dose` (5.55) | 5 | 48 | 38,064 / 14,064 | 375/576, `-0.075335` | `-0.029226` |
| `depth6-dose` (5.56) | 6 | 48 | 45,456 / 16,656 | 442/576, `-0.117494` | `+0.028013` |
| `depth7-dose` (5.57) | 7 | 48 | 52,848 / 19,248 | 503/576, `-0.181042` | `+0.094421` |
| **`depth8-dose`** | **8** | **48** | **60,240 / 21,840** | *this arm* | *this arm* |

**Design.** One fresh cell of the registered config with `depth` 7 -> 8 and **nothing else changed** -
the same one-gene dose as 5.55-5.57, one rung further up. Every rung adds exactly 7,392 total /
2,592 active parameters. The candidate pool is identical across the whole ladder - 40 feasible
expert mutations at depth 8 truncated to the same four by `probe_candidate_cap` 4, pinned by a test
over depths 3 through 8 - so only the substrate moves.

**Primaries.** Two, at the sibling rungs' level: `alpha/2 = 0.025`, bar **313/576**.

1. **P1 (the gate is still open):** the family's per-cell conjunction.
2. **P2 (an eighth layer creates more value than a seventh):** one-sided rank-sum of the depth-7
   cell's adaptive-vs-null deltas against this cell's.

**The declared prediction, made before the run and reported either way.** Cross-cell levels are
*descriptive* in this family, so a numerical forecast is not dressed as a third primary - but it is
written down before the measurement:

- **the increment** in `adaptive - null` falls in `(-0.069, -0.059)`: **the fourth increment itself
  (`-0.0635`), widened by the same flat `0.005` at each end** that produced 5.57's band. The flat
  reading the first model predicted - the three increments `-0.0418`, `-0.0382`, `-0.0422` - is
  **excluded by the band**, so this is a test of acceleration and not a restatement of "deeper is
  better"; and
- **the controller's own edge** stays within `0.020` of the depth-7 reading (`-0.086622`), the
  model's central claim stated as a number for the third rung in a row.

**Pre-registered interpretation (fixed before running).**

- **P1 and P2:** **the slope continues** - the ladder is monotone across all five rungs above the
  onset, and the acceleration reading gains a second point.
- **P1 only:** **the axis saturates at depth 7** - an eighth layer keeps the gate open but buys
  nothing more than a seventh, which bounds the monotone claim to depths 2..7.
- **P2 only:** an eighth layer creates more value than a seventh *and* still does not clear the bar on
  its own - a fragile upper reading.
- **Neither:** **the ladder is not monotone past depth 7** - an eighth layer fails where a seventh
  passes, so 5.57's steepening was a wobble and the ladder's bound moves down to depth 7.

**Descriptive secondaries.** The ladder to depth 8 (value, collapse, the controller's edge and the
best arm at each rung), the decomposition of the 7 -> 8 step into the edge and the churn term, and
the two declared predictions above - read together with 5.57's, which is what turns one miss into a
statement about the axis rather than about one cell.

**Scope.** The rungs mix 384-seed committed cells with 576-seed cells and are read against each other
descriptively; both primaries are tests on their own cells, and the two predictions are descriptive
readings of cross-cell levels, stated as such. The learning rate, step count, interval, cap and
policy are the registered ones at every rung.

**Cost.** Measured on this cell's own config before registering: ~2.49 s/seed at four threads, so 576
seeds is ~24 min of single-thread CPU across 20 shards - about a minute of wall clock, no GPU quota,
run while the corrected Gate 4 session trains. The block is 94384-94959, the first free 576-window at
or after 94000, checked by a test against every other profile *and* every committed summary's
recorded seeds. Registered before the run; the running revision is recorded in the summary's
`code_revision`.

**Outcome (run 2026-09-21, revision `a21c4a9`, 20 shards; committed summary `depth8-dose-576.json`).
Registered branch 1: P1+P2 PASS - and the second forecast missed too, one band away in the other
direction.**

P1: the depth-8 cell clears the bar decisively (`544/576` against `313`, mean `-0.235166`,
p `1.46e-121`). P2: its value deltas are stochastically better than the depth-7 cell's (rank-sum
U `131833.0`, one-sided p `8.12e-10`, means `-0.235166` vs `-0.181042`), so **the slope continues**
and the ladder is monotone across all five rungs above the onset:

| cell | depth | n | value | controller's edge | collapse | best arm | arms (adaptive / null / random) |
|---|---|---|---|---|---|---|---|
| `depth2` (5.15) | 2 | 384 | `+0.020811` | `-0.078848` | `-0.099659` | null | `14.3953 / 14.3745 / 14.4742` |
| `depth3` (5.15) | 3 | 384 | `+0.004643` | `-0.102151` | `-0.106794` | null | `13.4549 / 13.4503 / 13.5571` |
| `depth4-prefix` (5.12) | 4 | 384 | `-0.037147` | `-0.107220` | `-0.070072` | **adaptive** | `12.4413 / 12.4785 / 12.5485` |
| `depth5-dose` (5.55) | 5 | 576 | `-0.075335` | `-0.104561` | `-0.029226` | **adaptive** | `11.4870 / 11.5624 / 11.5916` |
| `depth6-dose` (5.56) | 6 | 576 | `-0.117494` | `-0.089482` | `+0.028013` | **adaptive** | `10.5882 / 10.7057 / 10.6777` |
| `depth7-dose` (5.57) | 7 | 576 | `-0.181042` | `-0.086622` | `+0.094421` | **adaptive** | `9.7065 / 9.8876 / 9.7932` |
| **`depth8-dose`** | **8** | **576** | **`-0.235166`** | `-0.070113` | **`+0.165053`** | **adaptive** | **`9.0747 / 9.3099 / 9.1448`** |

**Both forecasts missed, and the two misses bracket the series.** The declared increment was
`(-0.069, -0.059)`; the measurement is `-0.054124`, a miss on the *low* side. 5.57's declared band
was `(-0.047, -0.033)` and the fourth increment missed it on the *high* side at `-0.0635`. The six
increments are now `-0.0162`, `-0.0418`, `-0.0382`, `-0.0422`, `-0.0635`, `-0.0541`: three steps at
the flat level, a fourth well above it and a fifth above it but below the fourth. So the axis is
**neither constant nor accelerating without bound** - it is mildly rising, `~0.04` for three steps
and then `0.054`-`0.064` for two, with the scatter that implies. The second prediction hit for the
third rung in a row: the controller's edge measured `-0.070113` against `-0.086622`, a move of
`+0.0165` inside the declared `0.020`.

**The decomposition is now five rungs deep.** Value is
`adaptive - null = (adaptive - random) - (null - random)`, and from depth 3 to depth 8:

| term | depth 3 | depth 8 | rank-sum | reading |
|---|---|---|---|---|
| controller's own edge | `-0.102151` | `-0.070113` | U `102296.0`, p `0.0244` (reversed `0.976`) | flat to *slightly worse*, and now marginally so |
| churn term | `-0.106794` | `+0.165053` | U `25327.0`, p `1.5e-91` | moves, decisively |

The value improvement over those five added layers is `-0.239810`: the churn term carries
`-0.271848` of it and the controller's own edge `+0.032038` **against**. The newest step alone
(`-0.054124`) decomposes the same way: the churn term contributes `-0.070633` (p `4.9e-14`) and the
edge contributes `+0.016509` (p `0.162`, flat).

One reading here is new and worth stating precisely: over 3 -> 7 the edge's decline was p `0.139`,
and over 3 -> 8 the same reading reaches **p `0.0244`** - so the edge is no longer merely flat, it is
weakly but measurably *shrinking* as the substrate deepens. That is a cross-cell rank-sum with no
multiplicity correction and a direction that was hypothesised rather than registered, so it is
reported as a descriptive reading and not as a test; but it is the first time this family's
`edge + churn = value` split has shown the edge moving in the *against* direction at any significance
level, and it makes the central point sharper rather than weaker.

**The collapse keeps accelerating.** `null - random` runs `-0.099659`, `-0.106794`, `-0.070072`,
`-0.029226`, `+0.028013`, `+0.094421`, `+0.165053`: the do-nothing arm's disadvantage grows
**1.75x** across the newest step, and at depth 8 the order is `9.0747` (adaptive) < `9.1448`
(random) < `9.3099` (null) - the null arm now 0.235 nats behind the adaptive one.

**What this settles.** Two registered forecasts, both missed, each from one side of the same series,
and the primaries passed both times: what the ladder supports is a **mildly rising gradient** that
neither a constant nor an accelerating model fits, with the laminar reading (flat `0.04`) refuted by
the fourth rung and the accelerating reading refuted by the fifth. What it does not support is any
bound on the axis: the ladder is monotone across five rungs above the onset and nothing here
measures where it stops, while the thing that rises - the substrate's need for mutation - is still
not the controller's skill.

**Scrutiny before write-up.** Built-in checks 576/576 (mutations 3/3/0, zero null search FLOPs);
three replays of committed seeds (94404, 94704, 94944) exact at `OMP_NUM_THREADS=1`; both primaries,
the increment, both decompositions and the algebraic identity computed before any entered prose.
The band and both predictions were committed at `a21c4a9`, before the first shard ran.

**Scoped.** The eighth layer carries 7,392 total / 2,592 active parameters irreducibly, so this is
the depth axis *as a package*; the candidate pool is identical at every rung (pinned by a test); and
the increments are cross-cell descriptives in this family's convention, while both primaries are
tests on their own cells.

### 5.59 Pre-registration: the sixth rung, against the regime the two misses bracketed (RAN - summary `depth9-dose-576.json`)

Two registered forecasts have now missed from **opposite sides of the same series**: the flat model
at the fourth increment (`-0.063548` against `(-0.047, -0.033)`) and the accelerating model at the
fifth (`-0.054124` against `(-0.069, -0.059)`). That is a stronger position than either prediction
alone: the two misses bracket the axis, and what they bracket has three live readings - flat, the
mild rise the last two increments show, and something steeper than the fourth. This arm declares the
middle one, and the band is constructed so that the two it excludes are a miss.

| cell | depth | steps | total / active params | adaptive-vs-null | collapse |
|---|---|---|---|---|---|
| `depth2` (5.15) | 2 | 48 | 15,888 / 6,288 | 161/384, `+0.020811` | `-0.099659` |
| `depth3` (5.15) | 3 | 48 | 23,280 / 8,880 | 188/384, `+0.004643` | `-0.106794` |
| `depth4-prefix` (5.12) | 4 | 48 | 30,672 / 11,472 | 225/384, `-0.037147` | `-0.070072` |
| `depth5-dose` (5.55) | 5 | 48 | 38,064 / 14,064 | 375/576, `-0.075335` | `-0.029226` |
| `depth6-dose` (5.56) | 6 | 48 | 45,456 / 16,656 | 442/576, `-0.117494` | `+0.028013` |
| `depth7-dose` (5.57) | 7 | 48 | 52,848 / 19,248 | 503/576, `-0.181042` | `+0.094421` |
| `depth8-dose` (5.58) | 8 | 48 | 60,240 / 21,840 | 544/576, `-0.235166` | `+0.165053` |
| **`depth9-dose`** | **9** | **48** | **67,632 / 24,432** | *this arm* | *this arm* |

**Design.** One fresh cell of the registered config with `depth` 8 -> 9 and **nothing else changed** -
the same one-gene dose as 5.55-5.58. Every rung adds exactly 7,392 total / 2,592 active parameters.
The candidate pool is identical across the whole ladder - 45 feasible expert mutations at depth 9
truncated to the same four by `probe_candidate_cap` 4, pinned by a test over depths 3 through 9 - so
only the substrate moves.

**Primaries.** Two, at the sibling rungs' level: `alpha/2 = 0.025`, bar **313/576**.

1. **P1 (the gate is still open):** the family's per-cell conjunction.
2. **P2 (a ninth layer creates more value than an eighth):** one-sided rank-sum of the depth-8
   cell's adaptive-vs-null deltas against this cell's.

**The declared prediction, made before the run and reported either way.**

- **the increment** in `adaptive - null` falls in `(-0.069, -0.049)`: **the last two increments
  (`-0.0635` and `-0.0541`) widened by the same flat `0.005` at each end** that produced the two
  bands before it. The flat reading sits *above* this band and a step steeper than the fourth by
  more than the declared margin sits *below* it, so a hit says the axis is in the mild-rise regime
  and a miss says which of the two refuted readings it is in - and `--depth9-verdict` prints that
  sentence on a miss rather than only disconfirming a number; and
- **the controller's own edge** stays within `0.020` of the depth-8 reading (`-0.070113`) - the same
  declared claim for a fourth rung in a row, and now the reading 5.58's own ladder first showed
  moving at all (its decline reaching p `0.0244` over 3 -> 8, descriptive and unregistered).

**Pre-registered interpretation (fixed before running).**

- **P1 and P2:** **the slope continues** - the ladder is monotone across all six rungs above the
  onset.
- **P1 only:** **the axis saturates at depth 8** - a ninth layer keeps the gate open but buys
  nothing more than an eighth, bounding the monotone claim to depths 2..8.
- **P2 only:** a ninth layer creates more value than an eighth *and* still does not clear the bar on
  its own - a fragile upper reading.
- **Neither:** **the ladder is not monotone past depth 8** - a ninth layer fails where an eighth
  passes.

**Descriptive secondaries.** The ladder to depth 9 (value, collapse, the controller's edge and the
best arm at each rung), the decomposition of the 8 -> 9 step into the edge and the churn term, and
the two declared predictions above - read together with 5.57's and 5.58's, so the ladder's gradient
is bracketed by three forecasts rather than one.

**Scope.** The rungs mix 384-seed committed cells with 576-seed cells and are read against each other
descriptively; both primaries are tests on their own cells, and the two predictions are descriptive
readings of cross-cell levels, stated as such. The learning rate, step count, interval, cap and
policy are the registered ones at every rung.

**Cost.** Measured on this cell's own config before registering: ~2.77 s/seed at four threads, so 576
seeds is ~27 min of single-thread CPU across 20 shards - about seven minutes of wall clock, no GPU
quota, run while the corrected Gate 4 session trains. The block is 95384-95959, the first free
576-window at or after 95000, checked by a test against every other profile *and* every committed
summary's recorded seeds. Registered before the run; the running revision is recorded in the
summary's `code_revision`.

**Outcome (run 2026-09-21, revision `1816e34`, 20 shards; committed summary `depth9-dose-576.json`).
Registered branch 1: P1+P2 PASS - and the increment missed for the third consecutive rung, into the
seam between the two bands that preceded it.**

P1: the depth-9 cell clears the bar decisively (`559/576` against `313`, mean `-0.283891`,
p `7.81e-142`). P2: its value deltas are stochastically better than the depth-8 cell's (rank-sum
U `135354.0`, one-sided p `3.19e-08`, means `-0.283891` vs `-0.235166`), so **the slope continues**
and the ladder is monotone across all six rungs above the onset:

| cell | depth | n | value | controller's edge | collapse | best arm | arms (adaptive / null / random) |
|---|---|---|---|---|---|---|---|
| `depth2` (5.15) | 2 | 384 | `+0.020811` | `-0.078848` | `-0.099659` | null | `14.3953 / 14.3745 / 14.4742` |
| `depth3` (5.15) | 3 | 384 | `+0.004643` | `-0.102151` | `-0.106794` | null | `13.4549 / 13.4503 / 13.5571` |
| `depth4-prefix` (5.12) | 4 | 384 | `-0.037147` | `-0.107220` | `-0.070072` | **adaptive** | `12.4413 / 12.4785 / 12.5485` |
| `depth5-dose` (5.55) | 5 | 576 | `-0.075335` | `-0.104561` | `-0.029226` | **adaptive** | `11.4870 / 11.5624 / 11.5916` |
| `depth6-dose` (5.56) | 6 | 576 | `-0.117494` | `-0.089482` | `+0.028013` | **adaptive** | `10.5882 / 10.7057 / 10.6777` |
| `depth7-dose` (5.57) | 7 | 576 | `-0.181042` | `-0.086622` | `+0.094421` | **adaptive** | `9.7065 / 9.8876 / 9.7932` |
| `depth8-dose` (5.58) | 8 | 576 | `-0.235166` | `-0.070113` | `+0.165053` | **adaptive** | `9.0747 / 9.3099 / 9.1448` |
| **`depth9-dose`** | **9** | **576** | **`-0.283891`** | **`-0.082532`** | **`+0.201358`** | **adaptive** | **`8.5254 / 8.8093 / 8.6079`** |

**The miss, and the small region it landed in.** The declared band was `(-0.069, -0.049)` and the
measured increment is **`-0.048724`** - above that band's shallow edge by `0.000276`, and outside the
flat model's own band (`-0.047, -0.033`) as well, on the same side. So the third forecast in a row
missed, and this one missed into the `0.002`-wide gap *between* the two bands its predecessors
declared: the reading is neither the mild rise it declared nor the flat level the first model
declared. The verdict prints that sentence on a miss rather than only reporting `False`, precisely so
that a third miss could be read as a location rather than as a disconfirmation.

**What three misses from three readings leave standing.** The increments ending at depths 4 through 9
run `-0.0418`, `-0.0382`, `-0.0422`, `-0.0635`, `-0.0541`, `-0.0487`. The first three are flat at
`0.0407`; the last three decline *monotonically* and the gap above that flat level shrinks
`0.0228 -> 0.0134 -> 0.0080`, a factor of `0.59` and `0.60` - so what the ladder shows above the
flat regime is an **excursion that is decaying back toward it geometrically**, not a steeper axis.
That is consistent with all three misses (it is above the closing of the flat band, below the
mild-rise band by the amount the decay had already erased) and it is the first reading in this
family that predicts the *near* future rather than describing the past: on the measured ratio the
increment ending at depth 10 falls inside the flat band. That is a prediction this section does not
test; it is named here so the next rung can be registered against it rather than against a fresh
guess.

**The edge band hit for a fourth consecutive rung, and 5.58's marginal reading did not survive the
fifth.** `|−0.082532 − (−0.070113)| = 0.012419` inside the declared `0.020`, so the controller's own
edge is still not moving with depth. That matters more than a fourth hit usually would, because the
one reading that had suggested otherwise is now gone: 5.58 reported the edge's decline reaching
p `0.0244` over 3 -> 8 and deliberately reported it as **descriptive and unregistered**, noting that a
single rank-sum with no multiplicity correction and a hypothesised direction should not be read as a
finding. Over 3 -> 9 the same contrast reads p `0.121` - not significance lost by a hair but a
marginal reading that one more rung simply does not reproduce - and the edge is `+0.019619`
*against* over the wider span. The discipline that labelled it descriptive rather than a finding is
what kept it out of the synthesis, and this rung is the payoff.

**The decomposition is now six rungs deep.** From depth 3 to depth 9:

| term | depth 3 | depth 9 | rank-sum | reading |
|---|---|---|---|---|
| controller's own edge | `-0.102151` | `-0.082532` | U `105663.0`, p `0.121` (reversed `0.879`) | **flat - no move** |
| churn term | `-0.106794` | `+0.201358` | U `19510.0`, p `3.7e-104` | moves, decisively |

The value improvement over those six added layers is `-0.288534` (p `6.4e-109`): the churn term
carries `-0.308153` of it and the controller's own edge `+0.019619` **against**, so essentially all
of it is churn. The newest step alone (`-0.048724`) decomposes the same way and with the same shares
as its predecessors: the churn term contributes `-0.036305` (p `3.2e-06`, 74.5%) and the edge
`-0.012419` (p `0.831`, flat).

**The collapse keeps growing.** `null - random` runs `-0.099659`, `-0.106794`, `-0.070072`,
`-0.029226`, `+0.028013`, `+0.094421`, `+0.165053`, `+0.201358`. At depth 9 the order is `8.5254`
(adaptive) < `8.6079` (random) < `8.8093` (null) - the order depth 6 established, with the null arm
now `0.284` nats behind the adaptive one.

**What this settles.** Three registered forecasts have now missed and six rungs have passed: the
gate is open and the value gap widens at every rung from depth 4 to depth 9, with no reversal, and
what widens is still *not* the controller's skill - the edge is flat across the whole span at
p `0.879` while the churn term moves at p `3.7e-104`. The refusal to claim more than that is not
new and is the point: nothing here bounds the axis from above, and the ladder's own gradient is an
excursion decaying back toward `0.04` per layer rather than a curve that steepens.

**Scrutiny before write-up.** Built-in checks held in 576/576 seeds (mutations 3/3/0, zero null
search FLOPs), re-derived from the shard files independently of the summary's own `built_in_checks`
field; three replays of committed seeds (95404, 95704, 95944) exact at `OMP_NUM_THREADS=1`; both
primaries, the increment, both decompositions and the algebraic identity computed before any of them
entered prose. The band and both predictions were committed at `1816e34`, before the first shard ran.

**Scoped.** The ninth layer carries 7,392 total / 2,592 active parameters irreducibly, so this is
the depth axis *as a package*; the candidate pool is identical at every rung (pinned by a test); and
the increments are cross-cell descriptives in this family's convention, while both primaries are
tests on their own cells.

### 5.60 Pre-registration: the width bisection that halves the gate window the covariate programme left open (RAN - summaries `width-d3-dose3-576.json`, `width-d4-dose3-576.json`)

Section 10.3 closed the covariate programme on the finding that **only the depth axis moves the
gate** - and that finding carries a number it never closed. The same programme measured the width
axis *doing* to the gate what depth does, at a larger dose: width 48 at depth 3 (2.87x the
above-onset cell, 88,032 parameters) does not create value (`+0.0615`, 5.49), and width 96 (11.15x,
341,952) does (`-0.8966`, 5.50). So the width axis's gate sits somewhere in a **3.9x window in
parameter space**, and that window is the widest unresolved quantity left in the programme: it is
the difference between "depth is the gate" and "depth is the cheapest route to the same capacity
threshold", which is a materially weaker claim and the one a reader of 10.3 cannot currently
distinguish.

This arm bisects it at the window's **geometric midpoint in parameter space** - the point that
halves the window in log terms whichever way the gate falls. Width 68 gives 173,672 parameters at
depth 3 (5.66x) and 230,520 at depth 4 (7.52x):

| cell | depth | width | total params | x the above-onset cell | measured |
|---|---|---|---|---|---|
| `depth3` (5.15) | 3 | 24 | 23,280 | 0.76x | `+0.004643` - no gate |
| `width-d3-equal` (5.48) | 3 | 28 | 31,192 | 1.02x | `+0.0176` - no gate |
| `width-d3-dose` (5.49) | 3 | 48 | 88,032 | **2.87x** | `+0.0615` - no gate |
| **`width-d3-dose3`** | **3** | **68** | **173,672** | **5.66x** | *this arm* |
| `width-d3-dose2` (5.50) | 3 | 96 | 341,952 | **11.15x** | `-0.8966` - **gate** |
| `depth4-prefix` (5.12) | 4 | 24 | 30,672 | 1.00x | `-0.037147` - gate |
| `width-d4-equal` (5.48) | 4 | 28 | 41,160 | 1.34x | `-0.0326` - gate |
| `width-d4-dose` (5.49) | 4 | 48 | 116,640 | **3.80x** | `-0.2097` - gate |
| **`width-d4-dose3`** | **4** | **68** | **230,520** | **7.52x** | *this arm* |
| `width-d4-dose2` (5.50) | 4 | 96 | 454,464 | **14.82x** | `-1.7330` - gate |

**Design.** Two fresh cells, each changing exactly one gene from the cell above it in its own
column - `hidden_size` 48 -> 68 - with the same 48 steps, interval, policy and cap. The depth-3 and
depth-4 columns are both measured, as every width dose in this programme has been, so the bisection
is read on the below-onset column (where the gate is the question) and the above-gate column (where
the dose's continuation is). No other knob moves, and the candidate pool is the registered one at
depth 3 and 4 (unchanged, and pinned since 5.55's ladder extended it).

**Primaries.** Three, at the 5.50 level - this arm carries the same third test - so
`alpha/3 = 0.0167`, bar **315/576**.

1. **P1 (the below-onset cell creates value):** the family's per-cell conjunction. This is the
gate's own reading, and it is the one the window's upper edge is decided by.
2. **P2 (the above-onset cell creates value):** the same conjunction one depth up.
3. **P3 (the dose keeps paying above the gate):** one-sided rank-sum of the depth-4 cell's
   adaptive-vs-null deltas against the 3.80x rung's.

**The declared prediction, made before the run and reported either way.**

- **the mechanism, not the outcome:** 5.50's account of why the gate opens is that the *pool term*
  (`random`-vs-`null`) flips - the unmutated arm stops being viable once the substrate is wide
  enough that training at a fixed 48-step budget requires mutation. On the committed cells that term
  runs `+0.1068` (1.00x), `+0.1484` (1.15x), `+0.3053` (2.87x) below the gate and `-0.5860`
  (11.15x) above it. **This arm declares the flip is above the bisection:** at 5.66x the depth-3
  pool term is still positive and larger than at 2.87x. Linked to it by that mechanism, and declared
  in the same breath: **the depth-3 cell does not create value** (P1 fails). A prediction about the
  mechanism is worth more than a prediction about the bar here, because it says *where* the window
  narrowed to rather than only which side of it the cell landed on; and
- **the above-gate continuation:** the depth-4 cell creates value (P2) and its deltas are
  stochastically better than the 3.80x rung's (P3), as they were at 14.82x against 3.80x.

**Pre-registered interpretation (fixed before running).**

- **not P1, P2, P3:** **the window narrows to (5.66x, 11.15x)** - capacity does not buy the gate at
  173,672 parameters at depth 3, against ~30,672 along depth, so the gate costs several times more
  along width than along depth and the depth reading survives in the form 10.3 gives it.
- **P1, P2, P3:** **the window narrows to (2.87x, 5.66x)** - capacity buys the gate at this dose, at
  a price between 88,032 and 173,672 parameters, and "only depth moves the gate" becomes "depth
  moves it cheapest".
- **P1, not P2:** width opens the gate below the onset and destroys value above it again at this
  dose - the 5.49 interaction in bisection form.
- **not P1, not P2:** width is harmful above the onset and useless below at this dose - the depth
  axis is not capacity in any reading.
- **not P3:** the above-gate dose stops paying between 3.80x and 7.52x, so the width ladder plateaus
  where the depth ladder does not, and the 5.50 reading that capacity scales above the gate is
  bounded to the doses under 3.80x.

**Descriptive secondaries.** The full width ladder at both depths (value, the controller's own edge
and the pool term), the bisection arithmetic (each cell's parameter count and multiplier), and the
declared mechanism prediction above - read together with 5.49's and 5.50's decompositions, so the
window is stated as an interval with a mechanism on either side of it rather than as a threshold
that happens to have been bracketed.

**Scope.** Both cells are doses on the width axis at a fixed 48-step budget, so the reading carries
5.50's own stated limit: at this budget a wide substrate is a different training problem, not only a
larger one, and what the dose buys may be the ability to train at all. The bisection is
cross-column descriptive in this family's convention (the committed reference cells have 384 and 576
seeds) while all three primaries are tests on their own cells.

**Cost.** Measured on each cell's own config before registering: ~1.74 s/seed at depth 3 and ~1.29
s/seed at depth 4 (four threads), so 576 seeds each is ~16.7 and ~12.4 min of single-thread CPU -
~29 min across 40 shards, no GPU quota, run while the Gate 4 campaign trains. The blocks are
96384-96959 and 97384-97959, the first free 576-windows at or after 96000 and 97000, checked by a
test against every other profile *and* every committed summary's recorded seeds. The cost probe ran
seed 96384 in each cell for timing only (the family's usual cost declaration); the declared
prediction is read off the committed 1.00x/1.15x/2.87x decomposition and not off that seed.
Registered before the run; the running revision is recorded in each summary's `code_revision`.

**Outcome (run 2026-09-21, revision `698a866`, 20 shards per cell; committed summaries
`width-d3-dose3-576.json` and `width-d4-dose3-576.json`). Registered branch 1 - all three primaries
pass: the window narrows to (2.87x, 5.66x). And the declared mechanism prediction missed, in the
direction that replaces the mechanism with a sharper one.**

| cell | params | adaptive-vs-null | mean delta | p | creates value |
|---|---|---|---|---|---|
| depth 3, width 68 (5.66x) | 173,672 | 389/576 | `-0.127992` | `1.31e-17` | **yes** |
| depth 4, width 68 (7.52x) | 230,520 | 575/576 | `-0.725298` | `2.33e-171` | **yes** |

The below-onset cell clears the bar at `alpha/3` (bar `315/576`), so **capacity does buy the gate at
this dose**, and the width axis's gate at depth 3 is now located between 88,032 and 173,672
parameters - a 1.97x window where 5.49/5.50 had left 3.9x. P3 passes in its strongest form: the
7.52x cell's deltas are separated from the 3.80x rung's (`U = 20954.0`, p `1.27e-145`).

**The declared prediction missed, and the miss is the result.** The registration declared the flip
in the pool term (`random`-vs-`null`) was *above* the bisection: at 5.66x the term would still be
positive **and larger** than at 2.87x. It is still positive (`+0.231778`) but it is significantly
**smaller** (`-0.073489`, rank-sum p `1.35e-04`) - so the term is not rising toward the flip, it has
already **peaked at 2.87x** and begun to decline. The gate opened anyway. That refutes the
mechanism as it was registered - as an equivalence, "the gate opens when the pool term flips" - and
the bisection is what replaces it, because two cells could not:

| width at depth 3 | x | value | controller's edge | pool term |
|---|---|---|---|---|
| 24 (5.15) | 0.76x | `+0.004643` | `-0.102151` | `+0.106794` |
| 28 (5.48) | 1.02x | `+0.017638` | `-0.130727` | `+0.148365` |
| 48 (5.49) | 2.87x | `+0.061466` | `-0.243801` | `+0.305267` |
| **68 (this arm)** | **5.66x** | **`-0.127992`** | **`-0.359770`** | **`+0.231778`** |
| 96 (5.50) | 11.15x | `-0.896634` | `-0.310646` | `-0.585988` |

**The width axis's gate opens in two stages, and the first one is the controller's.** At 5.66x the
gate is open *while the pool term is still positive* - what crosses it is the edge, at its ladder
maximum (`-0.359770`, having run `-0.102`, `-0.131`, `-0.244`), and 5.66x is the last rung at which
that is true: by 11.15x the edge has fallen back to `-0.310646` and the pool term has crossed to
`-0.585988`. One depth up the order is the reverse, which is what the two committed cells had shown:
from 3.80x to 7.52x the depth-4 edge is flat (`-0.194518` -> `-0.199203`, rank-sum p `0.199` in
the decline direction and `0.801` reversed) while the pool term moves at p `4.6e-119`
(`-0.015155` -> `-0.526095`). So capacity buys the gate below the onset by making selection beat
random by more, and only at larger doses by making the unmutated arm unviable; above the onset only
the second mechanism operates. 5.50's attribution of the crossing to the pool effect was read off
two cells (3.80x and 14.82x) that sit on the *above-onset* side of that division, which is why it is
right there and wrong as a general account.

**What the window now says, stated as a price.** With the bisection the gate's cost is bounded on
both axes at this search scale: along depth it opens between **23,280** and **30,672** parameters (a
1.32x step), and along width - at the below-onset depth - between **88,032** and **173,672**. The
ratio of those two prices is therefore bounded too: **the same gate costs between 2.9x and 7.5x more
along width than along depth**, the lower bound being the most capacity-favourable pairing of the
two windows. And the sharpest form of it needs no window at all: the depth-9 cell (5.59) has the gate
open at **67,632** parameters - **23% fewer** than the width-48 cell at depth 3 that does *not* open
it. "Only depth moves the gate" is now "depth reaches the gate at a fraction of the parameters,
width reaches it too", which is what 10.3 could not distinguish before this arm.

**The qualifier that survives unchanged.** The width-68 arms sit at `21.68`-`22.04` (depth 3) and
`16.53`-`17.25` (depth 4) in absolute loss, against `12.44`-`12.55` for the same-depth narrow cells,
so at a fixed 48-step budget this capacity is a liability that mutation repairs rather than a better
model - exactly 5.50's stated limit, now measured at a fourth dose. The value the gate reports here is
escape from that collapse, and one direction of it (the first stage) is the controller's.

**Scrutiny before write-up.** Built-in checks held in 576/576 seeds per cell (mutations 3/3/0, zero
null search FLOPs); both cells' summaries are at the frozen revision `698a866`; the three primaries,
the declared prediction, both decompositions and the algebraic identity were computed before any of
them entered prose. One defect of my own was caught before the run: the registration's window table
carried two invented parameter counts (the width-28 depth-3 cell written as 27,456 / 0.90x against
the committed 31,192 / 1.02x), and the column is now derived from the profiles by a test rather than
transcribed.

**Scoped.** Two doses on the width axis at the fixed 48-step budget, so the reading carries 5.50's
trainability limit above; the bisection is cross-column descriptive in this family's convention while
all three primaries are tests on their own cells; and the *mechanism* reading above is descriptive,
not a pre-registered test - the registered prediction was the one that missed.

### 5.61 Pre-registration: the second bisection, where the edge's maximum is reached (RAN - summaries `width-d3-dose4-576.json`, `width-d4-dose4-576.json`)

5.60 left two things: a narrower window, and a mechanism. The window is (2.87x, 5.66x) -
the gate's width price at the below-onset depth lies between 88,032 and 173,672 parameters, a 1.97x
interval. The mechanism is a **staging**: at 5.66x the gate is open while the pool term is still
positive, and what crosses it there is the controller's own edge at its ladder maximum
(`-0.359770`). One dose cannot say whether that maximum is approached gradually or reached
abruptly - the edge's committed readings below the onset are `-0.102151` (0.76x), `-0.130727` (1.02x),
`-0.243801` (2.87x), `-0.359770` (5.66x), so the rise from 2.87x to 5.66x is the largest step in the
series and nothing between them is measured.

This arm sits at the new window's **geometric midpoint**: width 57, 123,006 parameters at depth 3
(4.01x) and 163,134 at depth 4 (5.32x). It halves the price window again, and it is the first dose
placed specifically to test the staging rather than the bar:

| cell | depth | width | total params | x the above-onset cell | measured |
|---|---|---|---|---|---|
| `width-d3-dose` (5.49) | 3 | 48 | 88,032 | **2.87x** | `+0.061466` - no gate |
| **`width-d3-dose4`** | **3** | **57** | **123,006** | **4.01x** | *this arm* |
| `width-d3-dose3` (5.60) | 3 | 68 | 173,672 | **5.66x** | `-0.127992` - **gate** |
| `width-d4-dose` (5.49) | 4 | 48 | 116,640 | **3.80x** | `-0.209673` - gate |
| **`width-d4-dose4`** | **4** | **57** | **163,134** | **5.32x** | *this arm* |
| `width-d4-dose3` (5.60) | 4 | 68 | 230,520 | **7.52x** | `-0.725298` - gate |

**Design.** Two fresh cells, one gene each - `hidden_size` 48 -> 57 and 68 -> 57 respectively - with
every other registered knob unchanged. Each cell is a *step back* toward its own column's ladder:
the depth-3 cell sits between 5.49's and 5.60's doses, the depth-4 cell between 5.49's and 5.60's.

**Primaries.** Three at `alpha/3`, bar **315/576**, unchanged from 5.60 (the same three tests).

1. **P1 (the below-onset cell creates value):** the gate's own reading.
2. **P2 (the above-onset cell creates value):** the same conjunction one depth up.
3. **P3 (the dose keeps paying above the gate):** one-sided rank-sum against the 3.80x rung.

**The declared brackets, made before the run and reported either way.**

- **the depth-3 controller's edge lies strictly between its 2.87x and 5.66x readings**
  (`-0.359770 < edge < -0.243801`) - the edge is still rising at this dose and has *not* reached its
  maximum, which is the staging read as a gradient rather than a step; and
- **the depth-3 pool term is still positive and below its 2.87x peak**
  (`0 < pool < +0.305267`) - the decline 5.60 found has already begun here, so the peak at 2.87x is a
  real turning point and not an artefact of the two cells that bracketed it.

Linked to the first, and declared in the same breath: **the gate is still shut at 4.01x** (P1 is
declared to fail), which is what would place the width price in (123,006, 173,672] parameters -
a 1.41x window where 5.49/5.50 left 3.9x.

**Pre-registered interpretation (fixed before running).**

- **not P1, P2, P3:** **the window narrows to (123,006, 173,672]** - the gate crosses between 4.01x
  and 5.66x, and the ratio to the depth axis's own bracket (23,280-30,672) rises to 4.0x-7.5x.
- **P1, P2, P3:** **the window narrows to (88,032, 123,006]** - and the declared shut-gate prediction
  missed, which also places the crossing *below* the edge's ladder maximum, i.e. the staging boundary
  is further down than 5.60's single dose suggested.
- **P1, not P2:** width opens the gate below the onset and destroys value above it at this dose.
- **not P1, not P2:** width is harmful above the onset and useless below at this dose.
- **not P3:** the above-gate dose stops paying between 3.80x and 5.32x, bounding 5.50's "capacity
  scales above the gate" to the doses under 3.80x.

**Descriptive secondaries.** The full width ladders at both depths (value, the controller's edge and
the pool term), the bisection arithmetic, and both declared brackets above - read together with 5.60's
staging, so the mechanism is either confirmed as a gradient or located as a step.

**Scope.** The same fixed-48-step budget limit 5.50 stated and 5.60 re-measured: at this budget the
wide substrate is a worse model in absolute loss, and the value the gate reports is escape from that
collapse. The bisection is cross-column descriptive in this family's convention while all three
primaries are tests on their own cells; both brackets are descriptive, and the registered prediction
here is the gate's own.

**Cost.** Measured on each cell's own config before registering: ~1.58 s/seed at depth 3 and ~1.25
s/seed at depth 4, so 576 seeds each is ~15.1 and ~12.0 min of single-thread CPU across 40 shards -
no GPU quota, run while the Gate 4 campaign trains. The blocks are 98384-98959 and 99384-99959, the
first free 576-windows at or after those points, checked by a test against every other profile *and*
every committed summary's recorded seeds. As in 5.60 the cost probe ran one seed of each cell for
timing only, and both brackets are derived from the committed neighbours rather than from it.

**Outcome (run 2026-09-21, revision `04dc4b3`, 20 shards per cell; committed summaries
`width-d3-dose4-576.json` and `width-d4-dose4-576.json`). Registered branch 1 - P1 fails with P2+P3
passing: the window narrows to (123,006, 173,672]. Both declared brackets hit.**

| cell | params | adaptive-vs-null | mean delta | p | creates value |
|---|---|---|---|---|---|
| depth 3, width 57 (4.01x) | 123,006 | 310/576 | `-0.028562` | `0.0365` | **no** (bar `315`) |
| depth 4, width 57 (5.32x) | 163,134 | 551/576 | `-0.418214` | `1.65e-130` | **yes** |

The below-onset cell misses the bar **by five seeds** at `alpha/3 = 0.0167` (and by three at the
sibling arms' `alpha/2`), so the gate is still shut at 4.01x and the width axis's price is now
bounded to **(123,006, 173,672]** parameters - a 1.41x window where 5.49/5.50 left 3.9x. The miss is
worth stating precisely rather than as a clean non-detection: the cell's mean delta has *crossed*
(`-0.028562`, against `+0.061466` one dose lower) while its win rate has not, so at 4.01x the
advantage is large on some seeds and absent on the median one. P2 passes (`551/576`, p `1.65e-130`)
and P3 with it (`U = 88052.0`, p `1.55e-43`).

**Both declared brackets hit - so the staging is a gradient, not a step.**

| declared bracket | bound | measured | rank-sum against the bounds |
|---|---|---|---|
| controller's edge between its 2.87x and 5.66x readings | `-0.359770 < edge < -0.243801` | `-0.326291` | below the 2.87x reading at p `7.7e-05`; above the 5.66x reading at p `0.051` |
| pool term positive and below its 2.87x peak | `0 < pool < +0.305267` | `+0.297729` | above zero by construction of the ladder; below the 2.87x peak at p `0.275`; *larger* than the 5.66x reading at p `2.5e-03` |

The second row is the honest qualification and it matters: the pool term at 4.01x is **not yet
significantly below its 2.87x peak** (p `0.275`), and the edge is only marginally above its 5.66x
reading (p `0.051`). So the two brackets hold, but the local slopes inside them are gentle - what is
established is that both terms are between their neighbours' readings, not that either has turned
sharply. The turning points themselves are now located: **the pool term peaks at 2.87x and the
controller's edge peaks at 5.66x**, and the gate opens in the window between them, where the edge is
still rising and the pool has come off its peak but is still positive.

**Above the gate the asymmetry is now four doses deep.** From 3.80x to 5.32x the depth-4 edge is flat
(`-0.194518` -> `-0.210845`, p `0.180`) and the pool term carries everything (`-0.015155` ->
`-0.207369`, p `2.55e-28`); from 5.32x to 7.52x it is the same (`-0.210845` -> `-0.199203`, p `0.453`;
p `1.5e-63`). Four consecutive above-gate doses now show the controller's edge not moving while the
pool term moves decisively at every step - which is 5.50's decomposition, replicated rather than
restated, and the reason the two-stage account is read off the *decomposition* rather than off the bar.

**What the tightened window says.** The price ratio is now **(4.01, 7.46]** - the width axis's bracket
divided by the depth axis's own (23,280-30,672) - so even with the most capacity-favourable pairing
of the two windows, the same gate costs **four times** more along width than along depth. The report's
synthesis is updated to that bound; section 5.60's own outcome keeps the `2.9x-7.5x` interval it
measured, scoped to the window it left, and this section is what narrows it.

**Scrutiny before write-up.** Built-in checks held in 576/576 seeds per cell (mutations 3/3/0, zero
null search FLOPs); both summaries are at the frozen revision `04dc4b3`; the three primaries, both
declared brackets and the rank-sums against their bounds were computed before any entered prose. The
trainability limit is unchanged and re-measured: the width-57 arms sit at `21.39`-`21.72` (depth 3)
and `16.62`-`17.04` (depth 4) in absolute loss, against `12.44`-`12.55` for the narrow cells, so at
48 steps this capacity is still a liability that mutation repairs rather than a better model.

**Scoped.** Two doses at the fixed 48-step budget, with 5.50's trainability limit above; the bisection
is cross-column descriptive in this family's convention while all three primaries are tests on their
own cells; both brackets are descriptive, and the registered prediction that did the work here is the
gate's own. The window is now narrow enough that the next dose would be measuring a 1.4x interval -
what remains open is not the price but *why* 4.01x is a near miss: the mean has crossed and the win
rate has not.

### 5.62 Pre-registration: the crossing dose under the other candidate policy (RAN - summary `width-d3-dose3-spread-576.json`)

Every width dose and every rung of the depth ladder in this family runs the **prefix** candidate policy
(`probe_candidate_policy: prefix`, the registered default). That was invisible while the arms were
reporting prices and gates - a dose is a dose - and it becomes the load-bearing caveat the moment the
programme starts making claims about the *mechanism*: 5.60's finding that the gate is crossed by the
controller's edge at the crossing dose, and 10.1's ratio rule (`|adaptive - random| / (random - null)`,
open where it clears 1), both rest on one candidate-selection policy.

Section 5.12's 2x2 is the only evidence that the policy does not matter, and it is exactly the wrong
evidence for this question: it showed `prefix` and `spread` agree at **depths 1 and 4 at the
registered width** - the churn-suppression and already-open cells - and said nothing about the dose
where the gate crosses, which is the only place the mechanism has been claimed to differ. So this arm
runs *the crossing dose itself*, one gene changed:

| cell | depth | width | params | policy | measured |
|---|---|---|---|---|---|
| `width-d3-dose3` (5.60) | 3 | 68 | 173,672 | `prefix` | `389/576`, value `-0.127992`, edge `-0.359770`, pool `+0.231778` |
| **`width-d3-dose3-spread`** | **3** | **68** | **173,672** | **`spread`** | *this arm* |

**Design.** One fresh cell, identical to 5.60's crossing cell in every registered knob except
`probe_candidate_policy`. There is deliberately no depth-4 companion: the claim under test is about the
below-onset crossing, and above the onset the pool term dominates at every measured dose, so a
companion would test a mechanism the arm is not about.

**Primary.** One, at the single-cell level (`alpha/2 = 0.025`, bar **313/576**):

1. **P1 (the gate still opens under the other policy):** the family's per-cell conjunction.

**The declared predictions, made before the run and reported either way.**

- **the mechanism, as a number:** the ratio `|adaptive - random| / (random - null)` exceeds `1.0` -
  the crossing is again *edge-dominated*, with the pool term positive and smaller than the edge. This
  is the first arm in the family registered against the ratio rather than against a level, and it
  fails in a specific and informative way if the other policy reaches the gate through the pool term;
  and
- **the dose's identity:** the mean value lies within `0.10` of the prefix cell's reading
  (`-0.127992`), so the same dose under a different candidate policy is the same experiment within a
  stated margin rather than a different one.

**Pre-registered interpretation (fixed before running).**

- **P1 and the ratio:** **the crossing is policy-robust** - the edge-dominated stage is a property of
  the dose rather than of the prefix policy, and 10.1's rule has survived its first out-of-policy test.
- **P1 without the ratio:** the gate opens under both policies but through a different mixture, so the
  ratio rule is scoped to the prefix ladder and 10.1 has to say so.
- **the ratio without P1:** a second near miss with the same mechanism - the boundary between the
  mean's crossing and the bar's is about the distribution rather than about the policy.
- **neither:** **the crossing is prefix-specific** - this dose under the other policy votes with the
  closed set, and 5.60's edge-dominated stage is a prefix artefact.

**Descriptive secondaries.** The crossing dose under both policies, and the ratio table's neighbouring
doses (2.87x, 4.01x, 11.15x), so the policy comparison is read against the ladder it sits in rather
than in isolation.

**Scope.** One cell, one policy pair: this tests `spread` against `prefix` at one dose, not a policy
sweep, and the two policies' *agreement* at the other measured cells is 5.12's result rather than this
arm's. The reading carries the same fixed-48-step trainability limit as every width arm above
(5.50-5.61). Both declared predictions are descriptive in this family's convention; P1 is the test.

**Cost.** Measured on this cell's own config before registering: ~1.57 s/seed, so 576 seeds is ~15.1
min of single-thread CPU across 20 shards - no GPU quota, run while the Gate 4 campaign trains. The
block is 100000-100575, the first free 576-window at or after 100000, checked by a test against every
other profile *and* every committed summary's recorded seeds. The cost probe ran seed 100000 of that
block for timing only (the family's usual cost declaration); both declared predictions are anchored to
5.60's committed cell rather than to it.

**Outcome (run 2026-09-21, revision `969a251`, 20 shards; committed summary
`width-d3-dose3-spread-576.json`). Registered branch 1: P1 and both declared predictions pass - the
crossing is policy-robust. And the *mixture* underneath it is not, in a way worth stating.**

The cell creates value at the single-cell level (`387/576` against the bar of `313`, mean `-0.126482`,
p `5.63e-17`), the ratio is **1.64** (`edge -0.323944`, `pool +0.197462` - edge-dominated again, with
the pool term still positive), and the value lands `0.001510` from the prefix cell's reading, inside
the declared `0.10` band by a factor of sixty-six. On the level the two policies are not merely
close, they are indistinguishable: rank-sum on the per-seed value deltas reads p `0.472` in one
direction and p `0.528` in the other.

**The components do not agree, and the total does - which is the finding.** Where the two policies
differ is in the mixture, and the differences are real:

| term | prefix (5.60) | spread (this arm) | rank-sum |
|---|---|---|---|
| value | `-0.127992` | `-0.126482` | p `0.472` / `0.528` - indistinguishable |
| controller's edge | `-0.359770` | `-0.323944` | prefix larger at p `0.032` |
| pool term | `+0.231778` | `+0.197462` | spread smaller at p `0.041` |

The edge moves by `+0.035826` and the pool term by `-0.034316` across the policy change - opposite
directions, nearly equal magnitudes, cancelling to `+0.001510` in the contrast the family reports. So
the robust object is the **crossing**: which dose opens the gate, and that it is edge-dominated with a
positive pool, hold under both policies. The split between the two terms is *not* robust at the ~0.03
level, which means the two-term decomposition is a description of the balance rather than an
invariant of the mechanism - a caveat 5.50-5.61's readings inherit, since every one of them was read
off a prefix-policy cell.

**Absolute levels move far more than the contrast does.** The spread arms sit at `22.03`-`22.35`
against the prefix cell's `21.68`-`22.04`: the candidate policy is worth `0.31`-`0.35` nats in absolute
loss (per arm) and `0.0015` nats in what the gate measures. The arm order is unchanged (`adaptive < null <
random` in both), so this is not the policy changing who wins - it is the policy changing the level
the comparison is taken at, and the comparison surviving it.

**Scrutiny before write-up.** Built-in checks held in 576/576 seeds (mutations 3/3/0, zero null search
FLOPs); the summary is at the frozen revision `969a251`; both declared predictions, the three rank-sums
and the value difference were computed before any entered prose.

**Scoped.** One dose, one policy pair: `spread` against `prefix` at the crossing, not a policy sweep,
and the two policies' agreement at the *other* measured cells is still 5.12's result at depths 1 and 4
rather than this arm's. The reading carries the same fixed-48-step trainability limit as every width
arm above.

### 5.63 The Gate-5 transfer path, and the two paths a reader could not tell apart (RAN - summary `gate5-transfer-rehearsal.json`)

**Why this exists.** Section 99's gate 5 reads *fresh architecture remains competitive*: the claim
is that the topology a search discovered is worth carrying **without** the training that produced
it. Every checkpoint already carries both - the payload holds `training.genome` beside `model` -
but the only code that read a checkpoint's genome was the resume branch, which rebuilds a model
from that genome and then immediately restores `model`, `optimizer`, RNG and scaler state over it.
So an architecture could only ever travel *with* its training, and gate 5 had no path to take: Gate
4 passing would have left the next gate with nothing to run.

**What was added.** Two functions in `src/act/training/v0_session.py`.
`genome_from_checkpoint` reads the genome and nothing else, raising on a payload that is not a
version-4/5 session checkpoint and on one whose `training.genome` is absent or `None`.
`fresh_init_from_checkpoint` rebuilds a model from that genome, freshly initialized at the config
seed, leaving the optimizer and the token generator untouched. `scripts/gate5_rehearsal.py` runs
both against one real tiny session (seed 16000, depth 2, two steps, 20,768 parameters) and writes
`artifacts/study-summaries/gate5-transfer-rehearsal.json`, pinned by `tests/test_gate5_transfer.py`.

**The property the read depends on is that the two paths are distinguishable**, and the rehearsal
measures it on the published checkpoint:

| path | parameter names | shapes | trained values |
|---|---|---|---|
| fresh init from the genome | same | same | **all 20 differ** (largest `0.125166`) |
| restore `payload["model"]` | same | same | identical, every tensor |

Topology transfers; training does not. If those two rows were the same object, a number produced by
the weights-carrying path could be read as a fresh-architecture number and nothing would notice -
the same shape as the stale-header and universal-claim defects section 10 records, one layer down.

**And one thing the first version of that check got wrong.** It asserted that a different config
seed moves *every* parameter. It does not: 15 of the 20 are seed-sensitive and the other five are
the normalization scales, which initialize to exactly `1.0` and so cannot be moved by any seed. The
check is now that the only parameters a seed cannot move are the normalization scales - a weight
matrix that stopped responding to the seed would be a real defect, and the assertion fails on it
rather than tolerating it.

**Scrutiny.** Ten built-in checks, all holding, and the negatives are among them: a payload whose
`training.genome` is `None` raises, a payload with an unknown version raises, two fresh inits at one
seed agree parameter-for-parameter, and a restore of the published weights is exact. The only
session the rehearsal spends is the two-step one that produces the checkpoint.

**Scoped.** This is not a Gate-5 result. The genome transferred is a two-step dev-shape one rather
than a discovered architecture, and nothing here says any architecture is competitive. What the
section establishes is that the read *exists*, that its guarantee is observable, and that violating
it fails a test rather than passing one - the gate's own bar (which comparison, against which
baselines, at what tolerance) is declared with the artifact it will be applied to. Gate 6 would need
the same primitive applied across seeds, which it inherits; gate 7 would additionally need the
genome's capacities carried at another scale, which this rehearsal does not touch.

### 5.64 The gate chain, declared while gates 5-7 are still prose (RAN - `src/act/training/gate_chain.py`)

**Why this exists.** Section 99 states its gates in one line each: *Adaptive > static MoE*,
*Fresh architecture remains competitive*, *Architecture transfers across seeds*,
*Architecture transfers across scale*, and then *only after Gate 7 should serious compute be
invested*. Gate 4 has a pre-registered verdict, a rehearsal of that verdict and knobs nobody may
move against a measured result - because it is the gate that was reached. The rest are sentences.
Defining them *after* their artifacts exist means defining them against the numbers that came back,
which is the same failure the study family spends pre-registration avoiding, one level up: a gate
whose measurement is chosen once the measurement is known is not a gate.

**What is declared.** `src/act/training/gate_chain.py` gives each gate its artifact, comparison,
bar, and the gate it requires:

| gate | the plan's line | artifact it reads | requires |
|---|---|---|---|
| 4 | Adaptive > static MoE. | the declared 12-seed block with its control and null queues, read by `scripts/v0_verdict.py` | - |
| 5 | Fresh architecture remains competitive. | the winning genome transferred **topology-only** (§5.63's primitive), trained at the declared protocol | 4 |
| 6 | Architecture transfers across seeds. | the same genome by `genome_hash` equality, freshly initialized at **held-out** seeds | 5 |
| 7 | Architecture transfers across scale. | a **declared scale map** plus a baseline parameter-matched at the new scale | 6 |

**The open questions are the deliverable, not a caveat on it.** Each unimplemented gate names what
its own line does not decide, and a test requires it to: whether gate 5's "competitive" means the
same win threshold Gate 4 applies to its harm guard or a non-inferiority margin against the
*searched* arm (different claims, and the line does not choose); whether gate 6's held-out block
must clear the bar seed by seed or in aggregate, and whether it should reuse the architecture
track's fresh-seed convention (the finalist re-ranking used 45-47); and - the one with the most
riding on it and the least written about it - whether gate 7's scale map is allowed to move
**depth**, because scaling width keeps a topology and scaling depth replaces it, so that choice
decides which claim the gate is testing. Declaring these now is what stops them being resolved
later, with the numbers in view.

**Scrutiny.** The plan's own text is read back out of `docs/complete_implementation_plan.md` and
each declaration must match it verbatim, so a plan change fails the suite instead of leaving this
module describing something else; the chain must be gapless, each gate requiring its predecessor;
exactly one gate is implemented; and every unimplemented gate must name its open questions.

**Scoped.** This declares **reads**; it does not make gates 5-7 runnable, and the module says so
itself - `declared_but_unimplemented()` returns `(5, 6, 7)` so that "the chain is declared" and
"the chain runs" cannot be read as the same statement. What exists today is gate 4's verdict and
gate 5's transfer path.

### 5.65 Amendment: the declared block is re-reserved to 15024-15035, because its first four seeds ran the pre-amendment rebuild

**Decision.** The declared block moves from **15000-15011** to **15024-15035**, and the
primary dataset root moves from `act-v0-5-1` to **`act-v0-5-2`**. All three queues
re-reserve together, because section 2c requires them to cover the same block.
Delegated and recorded here with its reasoning and its reversibility, like 5.47.

**Grounds, in order of weight.**

1. **The block was not the registered configuration, and it said so itself.** The launch
   path (`scripts/kaggle_t4.py::session_argv`) never passed the two flags the runbook
   prints, so the published summaries record `preserve_router_on_rebuild: false` and
   `new_expert_initialization: "random"` while the registration, the runbook and the
   verdict all name the treated one. The verdict's conformance block is descriptive and
   withholds nothing, so a Gate 4 verdict over that block would be a PASS whose own text
   reads *"the published run is not the registered configuration, so the verdict describes
   a different protocol than the one it names"*. For a pre-registered protocol the artifact
   has to be the protocol.
2. **The block was no longer blind.** Its first four seeds' harm numbers were read and
   reported while the campaign was stopped (adaptive 2 of 4; mean adaptive-minus-random
   +0.000041 nats). Section 5.5 change 4 re-reserved a block for exactly this reason - the
   harm guard is one of the three verdict branches and its power is a property of not
   having seen the data. Amending the *registration* to match what the buggy launch did,
   after those numbers existed, is the one move this project does not make.
3. **The alternative, priced.** Finishing 15000-15011 untreated costs ~6.6 GPU-h less (the
   four seeds are already trained) and produces a flagged verdict. Restarting the same
   seeds treated would cost the same as this amendment and would still reuse four seeds
   whose outcomes are known, so it loses on ground 2. The delta here is one block's worth
   of already-spent hours against the one artifact the whole protocol exists to produce.

**What changes.** The seed block, the dataset root, and the constants that name them
(`v0_runner.DECLARED_SEEDS`, `v0_verdict.DECLARED_SEEDS`, `kaggle_t4.SEED_BLOCK`,
`kaggle_t4.DEFAULT_SLUG`), plus the runbook. **Nothing else**: shape, tokens per arm, event
count and cadence, dtype, arms, the adopted probe regime (`spread`, `argmin`, cap 4,
1-batch), the 5% criterion, the tolerance band, the harm guard, the pool-matched and
quality clauses, the section 5.14 do-nothing clause and the queue-completeness requirement
all stand exactly as registered. The four untreated seeds are **not discarded**: they keep
their own versions in `act-v0-5-1` as a measured untreated block, and their numbers are in
the operational record.

**A fresh root, not only fresh seeds.** A session now refuses to train a pair whose
rebuild treatment differs from what the versions already in its dataset root record
(`v0_runner.check_block_treatment_uniform`), which is what stops two treatments sharing one
root after the fact. So one root is one block, and a re-reservation has to move the root
with the seeds. That guard is the part of this amendment that would have caught the whole
incident before the first seed: it fails closed, names both fields and both values, and
chooses neither remedy. The second half is a **doc/code agreement pin** - the runbook's
`pair-session` commands and `session_argv` must carry the same rebuild flags - because this
defect was invisible to a doc-only check (the doc was right) and to a code-only check (the
code, and the test named *"the registered one"*, agreed with each other).

**What this is not.** Not a change to section 5.47: the adopted protocol for the declared
run remains the treated configuration, and the launch path now implements it. Not a claim
that the treatment decides Gate 4 - 5.47's position (small, right-skewed, 69% of the churn
harm survives) stands unchanged. And not a finding that any *number* moves: no parameter
count, no probe FLOPs and no criterion depends on the two fields.

**15012-15023 is skipped deliberately.** 15012 is the benchmark's reserved data seed; the
reservation is one line in `v0_runner.DECLARED_SEEDS` and the skip is stated there so the
gap is not read as an oversight.

**Reversible.** Before the first session of the new block, returning to 15000-15011 and
finishing it untreated is a constant change plus the conformance block reporting what the
artifact already says. After the new block's first seed is published the re-reservation is
the less costly direction, for the reason in ground 3.

## 6. Reproducibility

- Verdict machinery and experiment code: `src/act/evolution/`
  (commits `d972d4c`, `d5ee330`, `838f619` and later).
- Study commands, seed blocks, and `study_id` values are listed per record in
  `artifacts/study-summaries/*.json`; raw JSONL dumps reproduce from config +
  seeds via `act.evolution.cli` and are regenerable deterministically on CPU.
- **Reproduction is bit-exact only at a fixed thread count, and this was measured
  rather than assumed.** Re-running section 5.25's committed prune block (seed
  98000) from the config stored in its own summary gives the identical losses at 1
  and at 4 torch threads, but **not at 2**, where the same config moves the last
  digits (`16.073335886001587` -> `16.073336362838745`, and `16.090655088424683` ->
  `16.090654611587524`). Reductions are thread-count sensitive, so "reproduces"
  is only well defined at a fixed thread count; the block runner and the study
  scripts set `OMP_NUM_THREADS=1` for this reason, and
  `tests/test_prune_target_policy.py` pins torch's thread count while it asserts
  the exact reproduction. All committed numbers in this report were produced with
  `OMP_NUM_THREADS=1`.
- Registry records for single experiments append to `artifacts/experiments.jsonl`
  (gitignored); the curated summaries are versioned.

## 7. Deliverables checklist (plan §100)

| Plan deliverable | Status |
|---|---|
| 1. Source repository | DONE (this repository) |
| 2. Reproducible Kaggle notebooks | DONE for the plan section 39 set (`00_environment` + section 104's `01`-`03` and `04`-`09` per section 39's role list; deterministic builder `scripts/build_notebooks.py`, byte-identical re-runs, all ten notebooks' cells executed and verified) |
| 3. Dataset manifests | DONE for the declared path (the manifest contract - `act.ops.publish_artifacts` / `restore_artifacts` - is what the V0 checkpoint handoff publishes through: staged under a temporary directory, manifest checksummed and written **last**, then the seed directory swapped in, so a partial publish is never mistaken for a complete one; `src/act/training/v0_handoff.py`); the synthetic-tokenizer manifests remain as before |
| 4. Baseline checkpoints | DONE for the declared block's published seeds: `act-checkpoints/v0-5-1/seed-NNNN/<arm>/` holds each arm's latest event checkpoint, its metrics JSON and the seed summary, and the non-adaptive arms are published through the same path as the adaptive one |
| 5. Adaptive checkpoints | DONE as for item 4 - the campaign's exit publish is what makes the session boundary survivable, and it has published 4 of the declared block's 12 seeds at this writing (section 5.2 for the handoff rule, `docs/T4_RUNBOOK.md` for the campaign) |
| 6. Architecture genomes | DONE for the gate model (serialization, hashing, fossils) |
| 7. Topology evolution logs | PARTIAL (mutation histories and fossils in records) |
| 8. Experiment registry | DONE for the gate (append-only JSONL + study summaries) |
| 9. Benchmark results | PARTIAL - this report's factor map, **plus the architecture-search track's real-text series** (`artifacts/arch-zoo-100m-v1` through `-v4-head-paired`, ~100M tokens, 1,200 steps at its registered horizon, with its own `ARCH_SEARCH_FINDINGS.md`), cross-read against this track in section 10.4; no real-corpus benchmark on *this* track |
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
  --compare '/tmp/recheck/*.json' \
  --compare-out artifacts/study-summaries/factor-map-recheck.json
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
   measured depth **up to 5** (5.9-5.15, all p < 1e-9) - and it does not
   hold at the next rung: at depth 6 the do-nothing arm is *worse* than
   churning blind, `+0.028013` (5.56), the first rung where that is so.
   **Width qualifies it (5.49,
   added 2026-09-20):** at depth 4 x width 48 the harm is *absent* -
   null-vs-random `266/576`, mean `+0.015`, p `0.97` - while at depth 3 it
   grows with width (mean `-0.305` at width 48). So the churn axis is
   substrate-general in depth but not in width: the same perturbation that
   destroys loss on a shallow substrate is absorbed, and value-relevant, on a
   deep-and-wide one. **The second dose reverses it (5.50, added 2026-09-20):** at
   width 96 the harm is not absent but negative - churning *pays* at both depths
   (depth 4: `576/576`, mean `-1.574`, p `4.0e-174`; depth 3: `527/576`, mean
   `-0.586`) - so width does not merely absorb the churn axis, it flips its sign once
   the substrate is wide enough that mutation is what makes it trainable at all -
   and 5.51 then halves the reversal at 4x the budget (`+0.708`, rank-sum p
   `1.7e-183`), so the sign flip is budget-sensitive too. **The bisection then locates both flips
(5.60, added 2026-09-21):** at depth 4 the churn term's sign changes between 1.34x and 3.80x
(`+0.088010` -> `-0.015155`), and at depth 3 it *peaks* at 2.87x (`+0.305267`), declines through the
bisection (`+0.231778` at 5.66x, still positive) and crosses only between 5.66x and 11.15x
(`-0.585988`). So below the onset the flip happens *after* the gate opens - which is why the
two-term decomposition, and not the bar, is what can say which mechanism crossed it.

   **This survives the one configuration the project ships (5.40, added
   2026-09-20).** The 5.13 closure was scoped to a pre-repair path, so the
   deployed treatment (`preserve_router_on_rebuild` + `clone` init) was
   tested against the null arm on twelve fresh seeds at depth 6: the null
   has the lowest final loss on **12/12** seeds, beating the adaptive arm by
   **0.1355 nats** (p = 1.0 against P1) and the random arm by **0.1457**
   (p = 0.0005 for churn harm). So the closure is *re-established on the
   deployed configuration*, and "the repair" removes the mutating arms'
   disadvantage without giving either an advantage - the two are tied
   (`adaptive - random` = -0.0101, p = 0.61).

2. **The depth axis (toy path).** Value creation by probe-ordered
   selection is a function of substrate depth, now measured across nine
   cells: adaptive-vs-null wins go 117 (d1) < 161 (d2) < 188 (d3) < 225
   (d4) of 384, then 375/576 (d5), 442/576 (d6), 503/576 (d7), 544/576 (d8)
   and 559/576 (d9), with mean delta
   +0.0508 > +0.0208 > +0.0046 > -0.0371 > -0.0753 > -0.1175 > -0.1810
   > -0.2352 > -0.2839
   - a monotone
   dose-response with the sign flip between depth 3 and 4 (5.15), robust
   to the candidate policy (5.12 2x2: prefix and spread agree in every
   cell). **The gradient above the gate is an excursion decaying back to a
   flat level, and three registered forecasts have now missed it**
   (increments -0.0162, -0.0418, -0.0382, -0.0422, -0.0635, -0.0541,
   -0.0487; 5.55-5.59): three steps at the flat ~0.04 level, then a fourth
   well above it, then two that decline monotonically back toward it with
   the gap shrinking by a factor of ~0.59 each rung. So the constant
   reading (5.56-5.57) *understated* the axis, the accelerating one (5.58)
   overstated it, and the mild-rise one (5.59) landed in the 0.002-wide seam
   between their bands - which is why this synthesis says what the series
   *is* (a decaying excursion) rather than picking the reading it most
   nearly matched. On that decay the next increment falls inside the flat
   band; that prediction is stated and untested, not asserted. The churn
   axis flips with it: the do-nothing arm is
   the better one at depths 2-5 and stops being so at depth 6, where the
   arm order becomes adaptive
   < random < null - which depths 7, 8 and 9 repeat, with the do-nothing
   arm 0.1175, 0.1811, 0.2352 and then 0.2839 nats behind the adaptive one.

Together: **selection value exists, but only where the substrate is deep
enough to support it - and the deeper the substrate, the less of that
value is the controller's.** Below the onset, every mutating arm loses to
doing nothing; the historical depth-1 passes were churn suppression
measured against random churn (the only control they had), not value
creation. At depth 4 about a quarter of the raw edge survives the strictest   control (0.0297-0.0371 of 0.1130); across the whole upper ladder the
   controller's own edge is **flat to slightly worse** (p `0.139` from depth 3
   to depth 7, p `0.0244` over 3 -> 8, and p `0.121` over 3 -> 9 - so the one
   marginal decline reading, reported as descriptive precisely because it was
   marginal, does not survive the next rung) while essentially all of the
   0.2885-nat improvement from depth 3 to depth 9 is the substrate's
   growing dependence on being mutated: the churn term carries `-0.308153` of
   it and the controller's own edge contributes `+0.019619` **against**
   (5.56-5.59).

**The crossing rule, read off both ladders (descriptive, added 2026-09-21).** Value decomposes
identically everywhere in this family: `adaptive - null` = (`adaptive - random`) + (`random - null`),
so the gate is open exactly when the *mean* of that sum is negative and the win rate clears the bar.
Written as a ratio of the two terms, the committed widths and depths fall into a clean order:

| cell | dose | controller's edge | pool term | \|edge\| / pool | gate |
|---|---|---|---|---|---|
| `depth3` (5.15) | 0.76x | `-0.1022` | `+0.1068` | 0.96 | closed |
| `width-d3-equal` (5.48) | 1.02x | `-0.1307` | `+0.1484` | 0.88 | closed |
| `width-d3-dose` (5.49) | 2.87x | `-0.2438` | `+0.3053` | 0.80 | closed |
| `width-d3-dose4` (5.61) | 4.01x | `-0.3263` | `+0.2977` | **1.10** | closed - mean crossed, bar not |
| `width-d3-dose3` (5.60) | 5.66x | `-0.3598` | `+0.2318` | **1.55** | **open** |
| `depth4-prefix` (5.12) | 1.00x (depth) | `-0.1072` | `+0.0701` | **1.53** | **open** |
| `width-d4-equal` (5.48) | 1.34x | `-0.1207` | `+0.0880` | **1.37** | **open** |

**No cell with a positive pool term and a ratio below 1 opens the gate** (0.96, 0.88, 0.80 - all
closed), **the single cell between the mean's crossing and the bar's is the 1.10 near miss, and every
cell at 1.37 or above is open** - with the pool term's own sign irrelevant once it turns negative
(the 3.80x, 5.32x, 7.52x and 14.82x doses, and every depth rung from 6 up, are open with a negative
one). So the crossing is located in the *ratio*, bracketed by construction to `[1.10, 1.37]`, and it
**corrects a reading sections 5.60 and 5.61 gave**: those arms read the staging as a division between
the sides of the onset ("below it the edge crosses, above it the pool does"), and the onset cell
itself is a counterexample - `depth4-prefix` is open at ratio 1.53 with a **positive** pool term. What
tracks the mechanism is the *dose*, not the side: on either ladder the small-dose end is
edge-dominated and the large-dose end is pool-dominated, and what makes the depth axis look
pool-only is that every rung at the registered width except the onset itself sits on the far side of
that transition. This is a descriptive reading of committed cells, made after both arms rather than
predicted by either, and the interval `[1.10, 1.37]` was bracketed by the doses rather than derived -
the test it invites is a dose placed to land inside it on a ladder that has not been swept.

**And it has had its first out-of-policy test (5.62, added 2026-09-21).** Every cell in the table
above is prefix-policy, so the crossing dose was re-run under `spread`: the gate opens again
(`387/576`), the ratio is again above 1 (`1.64`, pool positive), and the value is `0.001510` from the
prefix reading - indistinguishable by rank-sum (p `0.472` / `0.528`). So the crossing is
**policy-robust**. The *decomposition* underneath it is not, and the same arm is what shows it: the
edge and the pool term move `+0.035826` and `-0.034316` across the policy change - opposite
directions, nearly equal magnitudes, cancelling in the contrast. The two-term split is therefore a
description of the balance at a dose rather than an invariant of the mechanism, and every
decomposition above inherits that caveat, since all of them were read off prefix-policy cells.

### 10.2 What this means for the gates

- **Gate 3 stands, narrowed.** The depth-4 confirmation (5.6) passes with
  the null control under it (5.10): adaptive beats doing nothing
  out-of-sample, 218/384 with spread and 225/384 with prefix (5.12b).
  The depth-1 instances of Gate 3 are re-quoted as churn suppression.
- **Gate 4 is registered, amended, and running.** The 5.1/5.5
  compute-efficiency protocol (block 15000-15011) now carries the 5.14
  do-nothing clause: a null queue at declared fidelity runs alongside the
  primary and pool-matched control queues, and Gate 4 is withheld unless
  the adaptive arm beats doing nothing on the declared block. The
  campaign is live on Kaggle and had published 4 of the block's 12 seeds at
  this writing (sections 5.2 and 5.47 record the session window and the rebuild
  treatment; `docs/T4_RUNBOOK.md` is the operational record). The
  *read* of a pass, when it comes, is bounded by section 10.4: the block
  trains the one gene the search can move,  and 5.53-5.57 measure that gene
  and the substrate's need for mutation rather than controller skill. The
  three-queue launch path is smoke-tested; the run needs a T4. **In flight
  since 2026-09-20** (Kaggle T4, three queues, ~18.4 GPU-h each at the
  measured rate).

  **Its write-up now has less to lean on than when it was registered.**
  With 5.40 measured, the quality premise rests on the toy factor map plus
  the compute-efficiency protocol alone, with **no transfer claim**: at dev
  fidelity, at a depth the toy prior favours, and with the deployed
  treatment in place, doing nothing still wins every fresh seed. The
  declared run's null clause is therefore not a formality but the operative
  bar, and its most likely outcome is the one 5.40 saw at small scale - the
  null arm winning, which withholds Gate 4 at exit 2.
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
layer width, expert count) would separate depth per se from capacity.

**Where these stand at this writing: (ii) is closed and (i) is in flight.**
The covariate programme (5.48-5.53) measured every candidate (ii) named, and
sections 5.54-5.59 then dosed the depth axis itself; the answer is that only
depth moves the gate *cheaply* and that the axis keeps paying past it - with the
qualification section 5.60 adds below, that width reaches the same gate at
4.0x-7.5x the parameter price. (i) is the three-
queue campaign on Kaggle, 4 of its 12 declared seeds published at this writing. What follows is
that closing in the order it happened, kept as the audit trail of how (ii) was
settled rather than as a list of things still open.

Section 5.40 sharpens (ii) without settling it: at depth 6, where the toy
cell says selection should create value, real text still puts the null
0.136 nats ahead, so what (ii) is missing is not a deeper substrate but a
covariate that separates *substrate* from *depth* - and the one measured
candidate is residual width, which this search space cannot reach
(section 10.4). Section 5.48 then tested the one form the toy machinery can:
holding width fixed at the **equal-parameter partner** of the above-onset cell
(31,192 vs 30,672 parameters), the onset does not move - depth 3 x width 28
fails the do-nothing bar (252/576, mean `+0.0176`) where depth 4 x width 28
clears it (316/576, mean `-0.0326`) - so at this scale the onset is depth per se
against the measured covariate, and the remaining candidates are the unmeasured
ones (expert count, total compute). 5.49's dose form closes it further: a 2.87x
width dose does not move the gate either (depth 3 x width 48 fails, mean
`+0.0615`), while above the gate the same dose takes the win rate from `0.586` to
`0.828` and removes the churn harm (`266/576`, p `0.97`) - **depth gates,
capacity scales**. 5.50's second dose (width 96, 14.82x) then crosses the gate itself:
both depths clear the do-nothing bar at `alpha/3` (depth 3: `573/576`, mean `-0.8966`;
depth 4: `576/576`, mean `-1.7330`), so the onset is capacity-driven above a threshold
between 3.8x and 14.8x - with the decomposition that qualifies it: the *controller's*
edge over random is flat above the gate (`-0.159` at 14.8x against `-0.195` at 3.8x,
rank-sum p `0.98`) and what scales is the pool effect (`random`-vs-`null` `-0.0152` ->
`-1.5739`), because at the fixed 48-step budget a wide substrate is unusable unless
mutated at all; 5.51 (4x the steps, the same three mutational events) halves that
collapse (`+1.574` -> `+0.708`) while value creation survives at `576/576` (`-0.7149`),
and the controller's own edge - already flat above the gate - shrinks to `-0.0066` (from
`-0.159`), so the value at 14.8x is the mutational pool at both budgets. Section 5.53 then
measured the *third* covariate this list named - expert count - at the same
onset and at the family's tightest equal-parameter contrast yet (depth 3 with six experts,
30,624 vs 30,672 parameters): the gate does not move (`211/576`, mean `+0.059544`), and above
the onset the same dose takes value creation *away* (`260/576`, mean `+0.019507`, where the
depth-4 E=4 cell creates value at `225/384`, `-0.037147`, and reverses which arm is best).
**Two of the three named covariates do not move the gate and the third moves it the wrong way,
so the candidate that remains is total compute** - and the one the V0 search can actually move
is the one that reads against value creation. Section 5.54 then measured that last candidate,
in the 2x2's final empty corner (depth 3 at 192 steps, the family's only arm that moves neither a
parameter nor the search space): 4x training is **inert** below the onset, moving neither primary
(`281/576`, mean `+0.039005`, p `0.734`; collapse `-0.106794` -> `-0.128388`, rank-sum p `0.706`),
where the identical treatment one depth up moves both and strongly (`-0.037147` -> `-0.204292`,
p `2.6e-60`; collapse `-0.070072` -> `+0.163809`, p `1.9e-84`). So **all four named covariates are
now measured at this scale and only the depth axis moves the gate**, and what the budget buys
depends on being above that axis rather than being one - which is the sharpest form open question
(i) has taken. Section 5.55 then dosed that axis, the one covariate the family had never dosed:
depth 5 at the registered config clears the bar (`375/576`, mean `-0.075335`, p `1.89e-13`) and
creates *more* value than depth 4 (rank-sum p `9.99e-04`), so the ladder is monotone - `+0.0208`,
`+0.0046`, `-0.0371`, `-0.0753` across depths 2 to 5 - and **the gate is a steep step on a slope
rather than a discontinuity**, which is the strongest support the depth reading has had. The
decomposition is the qualification: the controller's own edge saturates by depth 3 (`-0.0788`,
`-0.1022`, `-0.1072`, `-0.1046`, with the 4 -> 5 move at p `0.418`) while the churn term is what
keeps moving (`-0.0701` -> `-0.0292`, p `1.1e-04`), so a deeper substrate pays because the
unmutated form gets relatively worse, not because the controller gets better. Section 5.56 then
closed the ladder at depth 6 and turned that into a property of the axis rather than of one
comparison: monotone across all five rungs, with a **constant** gradient above the gate
(`-0.0422` on the newest step, the largest since the gate itself), the churn regime flipped for the
first time (`+0.028013`, and the arm order becomes adaptive < random < null), and across the whole
upper ladder the controller's edge is flat to slightly worse (p `0.266`) while **all** of the
`0.122`-nat improvement is the churn term. So what the program's value tracks is the substrate's
need for mutation, which selection neither creates nor enlarges. Sections 5.57-5.59 then carried
that ladder to depth 9 and turned it into a model of its own gradient: the increments above the gate
were predicted three times - as flat, as accelerating, and as a mild rise - and **all three forecasts
missed**, the last landing in the 0.002-wide seam between the two bands its predecessors declared.
What the series actually shows is an excursion decaying back toward the flat `0.04` per layer by a
factor of `0.59` each rung, with the gate open at every one of the six rungs above the onset and the
controller's own edge still flat across the whole span (p `0.879` from depth 3 to depth 9) while the
churn term carries `-0.308153` of the `-0.288534` improvement. Section 5.60 then closed the one
number this paragraph left open - the width axis's gate, bracketed to a 3.9x window by 5.49 and
5.50 - by bisecting it at the parameter-space midpoint: **the gate opens at 5.66x** (`389/576`,
mean `-0.127992`), so capacity reaches this gate too, but the two axes' prices differ by a bounded
factor. Along depth the gate costs between 23,280 and 30,672 parameters; along width, at the
below-onset depth, between 88,032 and 173,672 - **2.9x to 7.5x more**, with the deepest cell (5.59,
67,632) opening it at 23% *fewer* parameters than the width dose that does not. Section 5.61 then
bisected again, to 123,006-173,672: the gate stays shut at 4.01x **by five seeds** (`310/576`,
though the mean has already crossed) so **the ratio tightens to 4.0x-7.5x**, and both of that arm's
declared brackets obtained - the depth-3 controller's edge lies between its 2.87x and 5.66x
readings (`-0.326291`, rising at p `7.7e-05` and still under the maximum at p `0.051`) and the pool
term is positive but off its peak (`+0.297729`, not yet significantly below it at p `0.275`). So the
two turning points are located: **the pool term peaks at 2.87x and the controller's edge at 5.66x,
and the gate opens in the window between them** - a gradient rather than a step, and above the gate
the asymmetry is now four consecutive doses deep (the edge flat at p `0.180` and `0.453` across two
more steps while the pool term moves at p `2.6e-28` and `1.5e-63`). And the bisection
refuted the mechanism 5.50 had attributed the crossing to, replacing it with a staged account: at
5.66x the gate is open while the pool term is still positive (`+0.231778`, having *peaked* at 2.87x)
- what crosses it is the controller's own edge, at its ladder maximum (`-0.359770`), and only at
larger doses does the unmutated collapse take over (`-0.585988` at 11.15x). Across the above-onset
dose *step* the order is reversed - the edge is flat (`-0.194518` -> `-0.199203`, p `0.199`) while
the pool term does everything (p `4.6e-119`) - and 10.1's ratio table is where that is generalised
and corrected: what tracks the mechanism is the dose rather than the side of the onset, and the
onset cell itself is edge-dominated. So "the gate is depth" becomes "depth is the cheapest substrate on
which mutation pays", and the one regime this program has found where *selection skill* is what
crosses a threshold is a wide-and-shallow one - a hypothesis about where a transfer policy might
eventually matter, named here as a hypothesis and measured nowhere else; and
(iii) a real-text scale run above the diagnostic fidelity (the closed
program says nothing about longer horizons or larger models). Nothing in
the current picture depends on re-tuning an existing cell, and no knob
may be moved against a measured result (the 5.5 discipline).

### 10.4 What Gate 4 buys, read against the other track

The architecture-search phase (its own `ARCH_SEARCH_FINDINGS.md`, sections 2c and 9)
and this topology program have never been read against each other. Read together, the
gap is the reachable operation set:

| gene | architecture phase (real text, ~100M, 1,200 steps) | the V0 session's search space |
|---|---|---|
| residual-stream width | **the lever**: -0.1562 nats at the factorial's exact 46,656-assignment floor | unreachable |
| head count | -0.0177 nats (p = 0.14), and parameter-free at fixed width | unreachable |
| depth | real but smaller, and **shallow wins**: +0.0209 nats at 10x the horizon, at its own floor | unreachable (layer count is fixed) |
| expert capacity | **uncorrelated with loss**: active parameters span x4.51 inside a matched design at rho = +0.214 (p = 0.62), and the fully dense design places 7th of 8 | **the only reachable axis** |

`feasible_expert_mutations` yields expert grow/prune and nothing else, and the only
assignment to `hidden_size` in `src/act/topology/` is the restore on a rejected
change. That is now pinned by `tests/test_topology_transactions.py`, so the claim fails
loudly if an operation is ever added instead of quietly going stale.

This does not demote Gate 4, whose question is about *selection* rather than about
which gene: does adaptive selection beat doing nothing on the declared block. Section
10.1's depth axis says selection value exists above an onset, and 10.2's null clause is
the right control for it. What it bounds is what a pass **licenses** - adaptive control
of expert capacity at block 15000-15011, which is not the same claim as adaptive
architecture search, and cannot move the recommendation of the track that measured
width, because the two search spaces share no gene.

**(Measured 2026-09-21, section 5.53 - and it sharpens the bound rather than only
restating it.)** The table's only reachable axis was given the same treatment as width, at the
family's tightest equal-parameter contrast: depth 3 with six experts reaches 30,624 parameters
against the above-onset cell's 30,672 (`-0.156%`), and it does *not* create value (`211/576`,
mean `+0.059544`). Above the onset the same dose does more than fail to help - it removes the
value creation the E=4 cell has (`260/576`, `+0.019507` against `225/384`, `-0.037147`) and
inverts the ordering so the do-nothing arm becomes the best one, at rank-sum p `9.1e-09` in the
direction that means the extra experts remove value. So the reachable gene is not merely a
different covariate from the one the ontology reads: it is a covariate that reads *against*
selection value when it is raised. A V0 search that grows experts is asking a question this
program has now measured a negative answer to - the bound 10.4 drew from the architecture track,
restated on the second track's own terms.

**(And the two tracks disagree about depth's sign, which is worth naming rather than
smoothing over.)** The table's depth row is the architecture phase's *loss* reading on real text:
at 10x the horizon, shallow wins (`+0.0209` nats, at its own floor). This track's depth
measurement is a *selection-value* reading on the toy substrate, and there it is monotone
increasing (5.55-5.56). Those are different quantities on different substrates, so they are not in
contradiction - but they are also not the same claim, and nothing in either program licenses
reading one as the other. What they jointly support is narrower and still useful: depth is the axis
on which *both* tracks find a real, non-zero effect, while the reachable gene is the one on which a
track finds none.

**And 5.54 bounds the other half of the same claim.** If the reachable gene were not the covariate,
loud, it would still be open that a pass is a *compute* statement rather than an architecture one -
the search buys more training either way. That is now measured too: 4x training is inert below the
onset and not inert above it (5.54), so what the block's 12 sessions train is not itself the
variable a pass would be reading. A Gate 4 pass would be a statement about *selection on this
block*, and neither the reachable gene nor the budget is an escape hatch from that reading.

**5.60 attaches a price to "the one axis that gates" - and one exception worth naming.** The width
axis reaches the same gate at 5.66x (5.60), so the honest form of the table's claim is that depth is
the *cheapest* axis on which mutation pays: the same gate costs 4.0x-7.5x more parameters along width
at the below-onset depth, and the deepest cell opens it at 23% fewer parameters than the width dose
that does not. The exception matters more than the price: every arm before this one found the
controller's own edge flat or irrelevant, and the bisection found the one place where it is the term
that crosses a threshold - below the onset at 5.66x, where the gate is open while the unmutated arm
is still viable and the edge is at its ladder maximum. That is a hypothesis about where a transfer
policy could eventually matter, not a measurement of it, and nothing else in this program has
reproduced it.

**5.55 sharpens the table below rather than adding a row to it.** The one axis that *gates* cheaply is
depth, and the newest measurement says it also **scales** past the gate (monotone from 2 to 5).
Layer count is fixed in the V0 session and unreachable from `feasible_expert_mutations`, so the
search cannot move the axis that both gates and scales - and the axis it *can* move reads against
value creation. Read with 5.55's decomposition, that is the sharpest form of the gap: what the
reachable mutation buys is a substrate effect (escape from a collapse), not a controller effect,
and no operation in the space reaches the axis whose effects are largest.

It also makes the priority of the *next* gate a live question rather than an
inference. The arch phase's width effect is the largest single effect either track has
measured at a matched budget, and it is the covariate open question (ii) above needs in
order to tell depth from capacity: the crossed arm is the program's only
matched-budget separation of the two (rho(depth, hidden) = 0.000 on a balanced 2x2),
and it answered *width*. A width operation is not free engineering - it reshapes every
weight in the stream, so it needs the treatment `grow_expert` already gives a new
expert, with re-initialisation and a parameter-budget check rather than a new genome
field - which is exactly why it is worth deciding on deliberately after Gate 4 rather
than adding to a running queue.

### 10.5 What putting width under test would take, read against the machinery that exists

Section 10.4 leaves one question open: what would the next gate actually have to build.
Read against the code, the answer separates into two claims that need different work -
and only one of them is a mutation problem.

**(A) Width as a gene the search is *given*.** The zoo machinery already crosses width at
a matched budget: `arch-zoo-100m-v3-crossed` is a balanced `depth x hidden_size` design
(24 runs, 3.28 GPU-h at 1,200 steps) whose `crossed_contrast` reports both main effects
at their exact floors, and `--design head-paired` went further by making the two halves
of a pair *identical* in parameters and FLOPs.So an arm that asks "does adaptive
expert-capacity control's value depend on the width it was given" needs a **design**, not a
mechanism: the same controller, run against substrates whose width is fixed at two
levels. It answers a question the program's own open question (ii) is pointed at (is the
depth onset depth, or capacity?), and it can be pre-registered and costed from the
committed per-run figures before any hours are spent.

**(Answered 2026-09-20, section 5.48 - registered and run while Gate 4 trained):** at
matched parameters the controller's value does **not** depend on the width it was given.
Depth 3 x width 28 (31,192 parameters, the equal-parameter partner of the committed
above-onset cell) fails the do-nothing bar - 252/576 wins, mean `+0.0176` - where depth 4
x width 24 (30,672 parameters) clears it, and depth 4 x width 28 keeps its value
(316/576, mean `-0.0326`). The consequence 10.5 registered for this outcome applies: a
search that *moves* width is asking about a different search than the one whose value was
measured, so (B)'s engineering is not what (A) was missing. What stays open is the
unmatched form - a dose of much wider, no-longer-equal-parameter below-onset substrates.
(That form was registered and run the same day - 5.50, width 96: the unmatched dose does
cross the gate, `573/576` and `576/576` against the do-nothing bar, and the same
decomposition shows the crossing is a pool effect rather than a controller effect.)

**(B) Width as a gene the search can *move*.** That is what section 10.4 shows is
missing, and the code says why it is the expensive half:

| what exists | what a width change does to it |
|---|---|
| `rebuild_v0_moe_from_genome` copies every retained expert by id and "every other parameter ... when shape-compatible" | a width change makes **every** parameter in the stream shape-incompatible, so nothing transfers and the model is re-drawn - a restart wearing a mutation's name |
| the genome already carries `hidden_size`, and `validate_genome` already enforces the parameter ceilings | nothing to add: the budget check a width mutation needs is the one already applied to grow/prune |
| the FLOP and parameter identities are computed from the genome (the matching report's three cost axes) | nothing to add, but every "matched budget" claim in a width arm would have to be re-derived from them rather than inherited |
| **no transfer policy exists anywhere in `src/act/model/`** (no slice, zero-pad or interpolation helper) | this is the whole of the missing work: a grown expert gets `new_expert_initialization`; a wider stream would need the same idea applied to every tensor, plus the optimizer state (shape-dependent) and the per-checkpoint payload (which scales with parameter count) |

(And the upside of reaching it is now measured, which this section's earlier form could
not say: 5.49 ran *the same controller* on a 3.8x wider substrate above the gate -
116,640 parameters against 30,672 - and took the do-nothing comparison from `225/384` to
`477/576` (mean `-0.2097`), with the churn harm absent. 5.50 then shows that at 14.8x
this upside is a *package* effect: the unmutated arm collapses (`random`-vs-`null`
`-1.574`) while the controller's edge over random stays flat (`-0.159`), so what a width
mutation would buy is escape from that collapse, not a better controller - and nothing
here measures a policy that widens *without* mutating. 5.51 sharpens it: at 4x the
budget the controller's edge above the gate is `-0.0066` (from `-0.159`), so the
measured value of a wide substrate is the mutational package, and the controller's price
for widening is whatever it costs to keep the package's events. That is a **substrate** effect,
not evidence that a width *mutation* is safe: nothing changed shape mid-run, so the
transfer-policy problem below is untouched, and a width mutation that re-draws the model
would pay 5.24's class of cost on every tensor at once.)

The project has already priced the *smallest* version of that failure. The one
shape-incompatible transfer it has measured is a single tensor - the router, whose shape
moves with the expert count - and section 5.24 exists because re-drawing it discarded
the learned routing function mid-training; the repair (`preserve_router_on_rebuild`) is
what section 5.40 then measured the *residual* cost of, and doing nothing still won every
fresh seed there by 0.136 nats. A width mutation with no transfer policy pays that class
of cost on every tensor at once, so the first thing such an arm would demonstrate is the
harm of the mutation rather than the value of the search.

**Neither may be added to a running queue** (the section 5.5 discipline), and both need
their own pre-registration before hours are spent. The order the machinery implies: (A)
first, because it is answerable today and it decides whether (B)'s engineering buys
anything - if the controller's value does not depend on the width it is given, moving
width itself is a question about a different search, not about this one.

**And (A) is now answered twice over, in the same direction.** Its registered form was
answered by 5.48 at matched parameters and by 5.49-5.50 in dose form (§10.5 above). The
depth axis then gave the same reading independently: across the whole upper ladder the
controller's own edge is **flat to slightly worse** (5.56, rank-sum p `0.266`) while the
value improves by `0.122` nats, and the entire improvement is the substrate's dependence
on mutation. So the answer to "does the controller's value depend on the substrate it is
given" is no, on both axes that have been dosed - which is the reading this section
registered as the one that would settle whether (B) buys anything. It does not buy a
better controller; what it would reach is the axis whose *substrate* effects are largest,
which is a different (and still open) question.
