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

## 3. Results: the factor map — six studies, one pass

Six pre-registered quality tests now cover the controller design space at toy
fidelity. Exactly one configuration has passed.

| # | Study | Selection policy | Events x probe budget | Horizon | Seeds | Wins | Mean delta | Binomial p | Paired-t p | Probe cost | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Legacy controller | routing/Fisher cheap signals | 3 x n/a | 48 | 384 (1000-1383) | 195/384 | +0.006766 | 0.3993 | 0.8437 | ~0 | FAIL |
| 2 | Guided upper bound (diagnostic) | probe-loss, all candidates, all 4 batches | 3 x 4b all-cand | 48 | 96 (2000-2095) | 59/96 | -0.041325 | 0.0158 | 0.00025 | high | PASS |
| 3 | Probe-loss controller | argmin of probed loss | 3 x cap4, 2 batches | 48 | 384 (3000-3383) | 213/384 | -0.025955 | 0.0181 | 7.4e-05 | 41.9% | **PASS (quality)** |
| 4 | Cheap probe | argmin of probed loss | 3 x cap2, 1 batch | 144 | 380 (4004-4383) | 181/380 | +0.011747 | 0.8351 | 0.8857 | 4.13% | FAIL |
| 5 | Cadence | as (4), 6 events | 6 x cap2, 1 batch | 144 | 384 (5000-5383) | 137/384 | +0.047014 | 1.0 | ~1.0 | 8.41% | FAIL (worse than random, z = -5.61) |
| 6 | Variance reduction | as (4), 4 probe batches | 3 x cap2, 4 batches | 144 | 384 (6000-6383) | 179/384 | -0.003907 | 0.9159 | 0.3453 | 16.0% | FAIL |

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
explicit open question, not a footnote.

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

The coherent story across all seven studies: probe-loss selection carries a
real but weak per-event signal whose value is concentrated in avoiding bad
mutations. Applied greedily over more events (study 5), the argmin amplifies
probe noise — a winner's curse. The single passing configuration (study 3)
paired a rich candidate set with a short horizon, at a probe cost that only
amortizes if training grows much longer than selection does; whether that
amortization holds at research scale is precisely the plan's deferred
compute-efficiency milestone.

### 3.3 Statistical caution

- Six gate-style quality tests have been run. Under a Bonferroni correction
  (alpha/3 ~ 0.0167 on the primary sign test), study 3's pass is marginal
  (p = 0.0181); its secondary paired-t (7.4e-05) is decisive. The guided study
  (2) was a diagnostic, not a gate claim.
- Studies 1, 3, 5, 6 are mutually independent (disjoint seed blocks). Study 4
  and the placement study share seeds by design (same-seed pairing). Study 6's
  descriptive mean comparison against study 4 is a two-sample approximation
  (z ~ -1.13), not a paired test.
- All raw records are bit-for-bit reproducible; `study_id` values are stable
  across re-runs and re-emissions.

## 4. What Gate 3 currently means

Gate 3 ("adaptive > random; if failed, stop and simplify") is in the state the
plan's protocol anticipated: the original controller failed and was simplified
into a probe-based one that can pass quality but not the toy-scale cost budget,
while every cheaper, higher-cadence, or variance-reduced variant fails quality —
the cadence variant significantly. No configuration yet satisfies quality and
compute-efficiency simultaneously at this fidelity, so **scaling remains
withheld**. The next pre-registered one-factor studies are (a) candidate
richness at fixed probe-token cost (cap-4/1-batch vs cap-2/2-batch) and (b)
replication of study 3's config on a 144-step horizon to test whether its pass
is horizon-specific. If neither passes, the documented negative results stand
and further selection work gates on real-scale measurement.

## 5. Pre-registered next studies

| Study | Factor | Config | Seeds | Primary | Decision rule |
|---|---|---|---|---|---|
| Richness | candidate set size at fixed probe tokens | cap4/1-batch vs cap2/2-batch, 3 events, steps=144 | fresh 7000-7383 | binomial sign test, alpha=0.05 | PASS -> richness is the active axis; FAIL -> proceed to horizon replication |
| Horizon replication | study 3 config at steps=144 | cap4/2-batch, 3 events, steps=144 | fresh 8000-8383 | binomial sign test, alpha=0.05 | PASS -> pass generalizes; FAIL -> pass was horizon-specific; selection work gates on real-scale measurement |

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

1. **Win-rate forest plot**: one row per study (#1-#6), point estimate with
   exact binomial 95% interval, threshold line at the pass boundary.
2. **Quality vs probe cost scatter**: mean delta (x) against probe cost as % of
   training FLOPs (y), marker size = seed count, color = verdict.
3. **Per-seed paired delta distributions** for studies 1, 3, 4, 5, 6.
4. **Probe-rank vs final-rank heatmap** from the counterfactual branches.
