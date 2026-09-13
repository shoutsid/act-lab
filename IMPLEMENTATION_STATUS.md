# ACT Implementation Status

Updated: 2026-09-13

This document compares the current workspace against
`Adaptive Computational Topology - Complete Kaggle Implementation Plan.md`.
It is the implementation tracker for the narrowed first scientific milestone.
The narrative companion covering the six-study factor map (six gate-style
quality tests, one pass) is `RESEARCH_REPORT.md`; all quoted statistics there
are machine-checked against `artifacts/study-summaries/`.

## Status Legend

- DONE: implemented and covered by a passing check.
- PARTIAL: a prototype or simulator exists, but the research contract is not complete.
- TODO: not implemented in the real Python training path.
- DEFERRED: intentionally blocked until the first scientific gate passes.

## Current Boundary

The project currently has two paths:

1. A feature-rich TypeScript laboratory simulator with fixed, random, and adaptive arms.
2. A small PyTorch foundation with dense and static-MoE models, deterministic training,
   checkpoint resume, topology genomes, safe prune/grow, and mutation-aware MoE rebuilds.
   It now also has versioned configuration schemas, deterministic checksum-addressed
   token shards, explicit CPU/CUDA device contracts, and reusable importance
   signal extraction for static-MoE experts.

The first real scientific result is now decided on three fronts. (1) The
pre-registered 384-seed Gate 3 study (seeds 1000-1383) FAILED for the legacy
cheap-signal controller: 195/384 wins (50.8%), mean delta `+0.006766`, binomial
p `0.3993` - small-sample noise, no benefit. (2) A probe-guided diagnostic upper
bound (`guided` arm) PASSED its 96-seed study (seeds 2000-2095): 59/96 wins,
mean delta `-0.041325`, binomial p `0.0158`, t p `0.00025` - proving topology
selection does express at this fidelity. (3) Making bounded probe-loss selection
the adaptive controller's default decision rule (cap 4 candidates, 2 probe
batches) PASSED the re-registered 384-seed Gate 3 re-run (seeds 3000-3383):
213/384 wins (55.5%, threshold 209), mean delta `-0.025955`, binomial p
`0.018138`, paired t p `7.4e-05`, matched budgets. The same pre-registration
included a search-cost criterion (probe FLOPs <= 5% of training FLOPs) which
FAILED at 41.9%: the quality gate is passed, compute-efficiency is not, and
amortization at realistic scale is the explicit next question rather than an
assumption. (4) A cheaper pre-registered variant (cap 2, 1 probe batch,
steps=144, seeds 4004-4383) satisfied the cost criterion (4.13% <= 5%,
confirming probe cost amortizes with horizon) but FAILED quality: 181/380 wins
(47.6%, threshold 207), mean delta `+0.011747`, binomial p `0.8351`, t p
`0.8857`.The current
map: probe-rich selection wins quality but misses toy-scale cost budgets;
probe-poor selection meets cost but loses the quality edge. (5) A same-seed
paired placement study (cheap probe, steps=144, mutations moved from steps
12/24/36 to 40/80/120; seeds 4004-4383) tested whether early placement explains
that failure: late placement was better in 201/380 pairs, mean paired delta
`-0.015666`, paired t p `0.0788` - directionally suggestive but NOT significant,
so placement does not statistically explain the horizon effect, even though it
changes 39% of probe decisions. The next study should vary probe cadence, not
placement. (6) A pre-registered probe-validity analysis (counterfactual branches
forcing candidate k in {1,2,3} at the first mutation event, seeds 3000-3383)
shows probe loss IS a statistically significant predictor of realized final loss
(mean per-seed Spearman rho `0.1057`, t(383) `3.638`, one-tailed p `1.374e-04`;
pooled z-scored Pearson r `0.1041`, z `4.09`) - but weakly: argmin concordance
`0.281` is near the 0.25 chance level, and the probe-best choice mainly avoids
the worst candidates (mean worst-branch excess avoided `0.0860` vs mean loss
left on the table `0.0601`). A weak per-event signal over only 3 mutation
events is consistent with the marginal Gate 3 pass and motivated a cadence
study. (7) That cadence study (cheap probe cap=2/1-batch, steps=144, interval=24,
budget=6 = six probe-loss events, fresh seeds 5000-5383) then FAILED HARD IN
REVERSE: 137/384 wins (35.7%, threshold 209), mean delta `+0.047014`, binomial
p `1.0`, and adaptive is significantly WORSE than random (worse-than-random
sign test z `-5.61`, two-sided p ~ `2e-08`). Accumulated greedy argmin on a
noisy single-batch probe is net harmful - consistent with a winner's-curse
mechanism: selecting the probe-argmin over many events amplifies probe-batch
noise instead of signal. (8) The pre-registered variance study (cap=2, FOUR
probe batches, 3 events, steps=144, fresh seeds 6000-6383) then FAILED its
quality gate: 179/384 wins (46.6%, threshold 209), mean delta `-0.003907`,
binomial p `0.9159`, t p `0.3453`. The mean shifted in the variance-hypothesis
direction (+0.0117 -> -0.0039) but not significantly (two-sample z ~ `-1.13`).
Per the pre-registered interpretation contract, probe variance reduction alone
does not restore the adaptive edge: candidate-set richness and/or the shorter
horizon of the single passing configuration remain the differentiating axes.
(9) The pre-registered horizon replication (the PASSING cap-4/2-batch config
re-run at steps=144, fresh seeds 8000-8383) PASSED and strengthened the
original: 231/384 wins (60.2%, threshold 209), mean delta `-0.052402`,
binomial p `4.04e-05`, paired-t p `1.07e-08`, matched budgets, probe cost
`14.0%` (amortization matches the fixed-per-event prediction of 41.9%/3).
The pass is therefore NOT horizon-specific; with seven gate-style tests the
study clears Bonferroni alpha/7 ~ 0.0071 on both statistics, and the contrast
with the cap-2/1-batch failure at the same horizon pins the active axis on
candidate richness + probe quality.

## Milestone Comparison

| Plan milestone | Status | Evidence / gap |
|---|---|---|
| 1. Foundation and baseline contract | DONE for the first gate | Python model, optimizer, training state, topology genome, hashing, model factory, and versioned checkpoint metadata are covered. Large-model and real-data contracts remain outside this gate. |
| 2. Dense baseline | PARTIAL | Dense causal LM trains on deterministic synthetic batches and resumes parameter-for-parameter. It is a tiny gate model, not the planned 256M-350M research model; validation loss, FLOPs, and real data metrics are not yet recorded. |
| 2. Static MoE baseline | DONE for the first gate | Static top-k MoE trains through the shared runtime and reports validation loss, active parameters, parameter count, and validation FLOPs. |
| 3. Topology graph and safety | PARTIAL | Python genome serialization, validation, hashing, prune/grow, merge/split/rewire guards, expert-ID weight preservation, and accepted mutation fossils exist. Full graph-aware model execution and post-mutation causal checks remain deferred. |
| 4. Random topology control | DONE for the first gate | Python Experiment 0 provides reproducible random churn with matched cadence, mutation count, training/search accounting, and report metrics. Real data and scale remain deferred. |
| 5. Importance instrumentation | DONE for the first gate | Python records routing/load, gradient norm, parameter norm, Fisher-style EMA, temporal stability, mutation count, and genome hash at structural windows. |
| 6. Adaptive controller | PARTIAL | Python exposes a deterministic, validated budget-aware `AdaptiveController` plus Pareto frontier/population selection over quality, active parameters, and search FLOPs. The validated default reaches 10/16 wins; broader multi-generation search remains. |
| 7. Experiment ledger | DONE for the first gate | Browser ledger and Python append-only JSONL registry store config, dataset/tokenizer hashes, code revision, hardware, tokens, FLOPs, arm metrics, mutation fossils, and architecture hashes. Deterministic evaluation summaries are available for report generation. |
| 8. Final gate: adaptive > random | PASSED twice (cap-4/2-batch at 48 and 144 steps, the second decisively); cheaper, higher-cadence, and variance-reduced variants FAILED | Factor map (all matched budgets, objective verdicts): cap-4/2-batch, 3 events, 48 steps -> PASS 213/384, p `0.0181`, at 41.9% search cost. Cap-2/1-batch, 3 events, 144 steps -> FAIL 181/380, p `0.8351`, at 4.13% cost. Placement (late, same seeds) -> no effect (p `0.0788`). Cap-2/1-batch, 6 events, 144 steps -> 137/384 (35.7%), significantly WORSE than random (z `-5.61`, p ~ `2e-08`). Cap-2/4-batch (variance reduction), 3 events, 144 steps -> 179/384, p `0.9159`. Horizon replication, cap-4/2-batch, 3 events, 144 steps -> PASS 231/384, p `4.04e-05`, t p `1.07e-08`, at 14.0% cost; clears Bonferroni alpha/7 on both statistics. Probe validity is real but weak (rho `0.11`); the active axis is candidate richness + probe quality, and probe cost amortizes with horizon as predicted. The remaining quality-vs-cost question is real-scale compute-efficiency (deferred milestone). |

## Plan Sections 3-6: Compute and Kaggle Operations

| Area | Status | Remaining work |
|---|---|---|
| T4x2 PyTorch/DDP | PARTIAL | Add device selection and DDP launch helpers now discover torchrun rank/world-size, initialize an appropriate process group, and wrap models only for multi-process runs; scheduler, accumulation, autocast, and activation checkpointing are available. T4x2 validation remains. |
| TPU v5e/JAX | DEFERRED | Port only after a stable architecture wins the GPU gate. |
| Kaggle artifact datasets | PARTIAL | Deterministic checkpoint, evaluation, report, and manifest layout contracts now exist; publishing automation remains. |
| Session resume protocol | PARTIAL | Local checkpoint resume is deterministic; Kaggle-style local publish/restore now verifies manifest checksums, while remote dataset restore and publish-on-exit remain. |
| Repository structure | PARTIAL | Core Python data, evaluation, evolution, importance, model, topology, training, and Kaggle-artifact implementations exist with focused tests; the planned notebooks and standalone research entry points remain. |

## Plan Sections 8-24: Model and Topology

| Area | Status | Remaining work |
|---|---|---|
| V0 256M-350M model | PARTIAL | Bounded V0 construction now runs finite forward/loss probes and reports parameter/logit shapes; target-scale memory and throughput measurements remain. |
| Attention/GDN/hybrid model core | PARTIAL | TypeScript simulator represents these module types; Python runtime currently supports dense and static MoE only. |
| Full genome contract | PARTIAL | Python genome now represents heads, modules, widths, residual paths, context, and compute ceilings and can reconstruct the V0 dense/hybrid model; MoE/shared-expert reconstruction remains. |
| Weight/topology separation | DONE for the first gate | Model and optimizer remain separate from the Python genome, while versioned checkpoints and Experiment 0 fossils package genome and mutation metadata together. |
| Importance levels and lifecycle | PARTIAL | Python exposes routing, gradient, parameter, Fisher-style EMA, magnitude, temporal-stability records, temporary ablation deltas, and append-only lifecycle fossils; population aggregation remains. |
| Expert ecology and similarity graph | PARTIAL | Device-safe expert signal extraction, functional sketches, and cosine similarity primitives exist; persistent graphs and lifecycle fossils remain. |

## Plan Sections 25-42: Data, Training, and Reproducibility

| Area | Status | Remaining work |
|---|---|---|
| Real tokenizer and shards | PARTIAL | Versioned deterministic binary token shards, bounded reads, checksums, manifests, and synthetic fixtures exist; real tokenizer/corpus adapters remain. |
| Real language-model validation | PARTIAL | Add real held-out batches; bounded perplexity and deterministic named capability probes now report weighted loss, perplexity, accuracy, and token counts, while real corpora remain. |
| Complete checkpoint format | PARTIAL | Versioned local checkpoints persist model/optimizer/training RNG state plus scheduler, genome, topology history, importance, dataset position, config, and revision metadata; compatibility validation and bounded session retention are implemented, while remote/session resume remains. |
| Experiment registry | DONE for the first gate | Append-only JSONL records include experiment/config/dataset/tokenizer hashes, hardware, tokens, FLOPs, architecture hashes, and arm results. |
| Reproducibility hashes | DONE for the first gate | Genome, config, dataset, tokenizer, and complete experiment hashes are recorded. |
| Research notebooks/scripts | PARTIAL | Dense and Experiment 0 CLIs now support baseline runs, append-only records, and multi-seed gate studies; extraction, retrain, and compare entry points remain. |

## Plan Sections 43-66: Scientific Experiments

| Experiment | Status |
|---|---|
| Experiment 0: fixed vs random vs adaptive | PARTIAL | Matched-budget Python runner, metrics, append-only single/study registry records, reproducible multi-seed CLI study, and a 6/8 probe exist; universal Gate 3 is still open. |
| Conventional dense/static-MoE frontier | PARTIAL |
| Dynamic sparse prune/grow | PARTIAL | Implemented in the Python Experiment 0 runner for safe expert prune/grow; richer topology operators remain. |
| Expert ecology clone/split/merge/death | TODO |
| Module-type evolution | DEFERRED |
| Adaptive depth | PARTIAL | Python topology mutations can safely toggle effective layer depth and change attention/GDN/hybrid module metadata; full multi-generation training comparisons remain. |
| Hierarchical search | PARTIAL | Deterministic layer/module candidate ranking now supports explicit depth penalties and inspectable priors. |
| Qwen-informed search | PARTIAL | Prior interface is available for imported architecture scores; no external Qwen artifacts are bundled or claimed. |
| Teacher initialization | PARTIAL | Explicit teacher and evolved shape-safe parameter transfer policies now exist; multi-seed retraining comparisons remain. |
| Architecture reusability and fresh initialization | PARTIAL | Fresh, evolved, and rewound initialization policies report copied/skipped tensors; architecture-equivalence studies remain. |
| Cross-seed stability and architecture equivalence | PARTIAL | Strict architecture-hash equivalence and cross-seed mean/stddev/min/max summaries now exist; full retraining studies remain. |
| Compute-efficiency scaling | DEFERRED |

## Plan Sections 67-105: Scale and Final Deliverables

These remain TODO or DEFERRED until the first gate passes:

- full metric and visualization report;
- topology fossils and trajectory plots;
- Multi-generation Pareto population search and richer diversity preservation;
- T4x2 scaling to 256M-350M;
- Qwen analyzer and architecture priors;
- fresh/rewound/teacher retraining;
- 500M-1B scaling;
- TPU port;
- paper-quality comparisons and final research report;
- versioned Kaggle datasets and reproducible notebooks.

## Immediate Execution Order

1. DONE: add Python metrics for validation loss, parameters, active parameters, training FLOPs,
   search FLOPs, validation FLOPs, and architecture distance.
2. DONE for the first gate: expand the Python checkpoint to include genome, mutation history,
   optimizer/scheduler, dataset position, and importance/controller state.
3. DONE for the first gate: implement a Python structural clock and random-churn controller
   using the existing prune/grow and rebuild APIs.
4. DONE for the first gate: implement cheap Python importance records: routing/load, gradient norm,
   parameter norm, Fisher-style running estimate, and temporal stability.
5. PARTIAL: use persisted routing records in the adaptive prune/grow controller while retaining
   the slower 48-step / 12-step structural clock and exact random budget. Signal sweeps did not
   yet improve the 6/8 default result.
6. DONE: add validation loss, active-parameter, and validation-FLOPs reporting to the runner.
7. RESOLVED (PASSED on quality, cost open): the cap-4/2-batch probe-loss
   controller passed the 384-seed Gate 3 re-run (213/384, p `0.018138`) at 41.9%
   search cost; a cheaper cap-2/1-batch variant at steps=144 then met the cost
   budget (4.13%) but failed quality (181/380, p `0.8351`) on fresh seeds
   4004-4383.
8. NEXT: Gate 3 quality now replicates across horizons (two passes, the 144-step
   one decisive under family-wise correction). The open question is no longer
   quality but compute-efficiency: probe cost 14.0% at toy scale amortizes with
   horizon and model size, which is exactly the plan's deferred
   compute-efficiency milestone. Recommended order: (a) real-scale cost
   measurement of the cap-4/2-batch controller, then (b) the pre-registered
   richness study (cap-4/1-batch vs cap-2/2-batch, fresh seeds 7000-7383) as a
   secondary factor analysis. Gate 4 (adaptive > static MoE) remains the next
   universal gate once compute-efficiency is justified.

## Verified Checks

- `PYTHONPATH=src python -m pytest -q` -> 86 passed, no failures.
- `PYTHONPATH=src python -m compileall -q src/act tests` -> passed.
- `npm run typecheck` -> passed.
- `npm run build` -> passed.
- `npm test -- --run` -> 185 passed, 0 failures (fixed grok-pwa-plugin and preview test port mismatches).
- Python Experiment 0 mutation-budget test -> passed.
- Python registry/CLI round trip -> passed.
- Initial Gate 3 trial -> adaptive lost to random on seeds 1, 2, 3, and 42.
- Four-seed gate test -> adaptive won 3/4 seeds with equal three-mutation budgets.
- Eight-seed generalization probe -> adaptive won 6/8 seeds; Gate 3 remained open pending a stronger pass.
- Sixteen-seed baseline -> adaptive won 10/16 seeds, with matched budgets and mean loss delta `-0.032691`.
- Expert-signal sweep -> random expert selection remained strongest at 6/8; routing, gradient,
  Fisher, and parameter-norm choices were retained as configurable probes but not selected by default.
- Gate-study CLI -> `PYTHONPATH=src python -m act.evolution.cli --seeds 1,2,3,42 --steps 48`
   emits a deterministic multi-seed JSON summary.
- Gate-study registry round trip -> the same CLI appends exactly one `gate_study` JSONL record.
- Gate-study budget invariant -> study execution and registry records enforce equal random/adaptive
   mutation counts for every seed.
- Browser ACT console and research invariants -> passed with no console errors in the latest checks.
- Fresh 16-seed Gate 3 study (`1,2,3,42,5,6,7,8,9,10,11,12,13,14,15,16`) -> 9/16 adaptive wins at
  48 steps, mean paired delta `-0.008127`, population standard deviation `0.162910`, and matched
  budgets; the scaling gate remains open.
- **384-seed pre-registered Gate 3 study (seeds 1000-1383, steps=48, budgets matched)**
  -> 195/384 adaptive wins (50.8%), mean paired delta `+0.006766`, stddev `0.131115`,
  exact binomial p `0.3993`, paired-t p `0.8437` (directed: treatment helps = negative
  delta); Gate 3 FAILED for the cheap-signal controller. Reproducible
  (`study_id` `48163d824f2d6852`, recorded in `artifacts/gate3-384.jsonl`).
- **96-seed guided-arm diagnostic (seeds 2000-2095, steps=48, budgets matched)**
  -> 59/96 guided wins (61.5%), mean delta `-0.041325`, binomial p `0.015774`,
  paired-t p `0.00025`; the probe-loss-greedy upper bound PASSED under the same
  verdict machinery (`study_id` `d2b76921ac40d39b`, `artifacts/guided-diagnostic-96.jsonl`).
- Gate-study verdict fields (`gate_alpha`, `gate_p_value`, `gate_min_wins`,
  `gate_passed`, `t_statistic`, `t_p_value_one_tailed`, `treatment`) added to
  `GateStudyResult` and gate-study registry records; `study_hash` now hashes the
  gate-decision inputs rather than float aggregates; the t p-value direction was
  corrected (helpful treatment = negative delta = P(T <= t)); covered by new unit tests.
- New opt-in `guided` arm (`--arms random,guided`, `--treatment guided`) with exact
  mutation-candidate enumeration, probe-loss selection, probe cost in `search_flops`,
  and probe metadata in mutation history.
- **Controller decision rule replaced**: the adaptive arm now defaults to bounded
  probe-loss selection (`adaptive_policy="probe_loss"`, `probe_candidate_cap=4`,
  `probe_batch_count=2`) via a shared probe-mutation engine; the legacy
  routing/gradient/Fisher rule remains available as `adaptive_policy="signal"`.
  Studies and registry records now include `search_flops_by_arm` and
  `treatment_search_overhead_ratio` for explicit search-cost accounting.
- **384-seed probe-loss Gate 3 re-run (seeds 3000-3383, steps=48, budgets matched)**
  -> 213/384 adaptive wins (55.5%; 209 required), mean delta `-0.025955`, binomial
  p `0.018138`, directed t p `7.4e-05`; quality PASSED, pre-registered <=5% search
  cost FAILED at 41.9% (`study_id` `27fe7e53d0533f29`,
  `artifacts/gate3-probe-loss-384.jsonl`).
- **380-seed cheap-probe study (seeds 4004-4383, steps=144, cap=2, 1 probe batch,
  budgets matched)** -> cost criterion PASSED (4.13% <= 5%; probe cost amortizes
  with horizon) but quality FAILED: 181/380 wins (47.6%), mean delta `+0.011747`,
  binomial p `0.8351`, t p `0.8857` (`study_id` `41f4b9f7cf6d82e0`,
  `artifacts/gate3-cheap-probe-380.jsonl`, summary in `artifacts/study-summaries/`).
  Confound noted: probe budget and horizon changed together vs the passing
  cap-4/2-batch/48-step study.
- **380-seed same-seed placement study (cheap probe, steps=144, interval=40 ->
  mutations at 40/80/120; seeds 4004-4383 identical to the early-placement run)**
  -> late placement better in 201/380 pairs, mean paired delta `-0.015666`,
  paired t p `0.0788`; late-study vs random alone: 198/380, binomial p `0.2208`.
  Placement changes 39% of probe decisions but does not statistically explain
  the horizon effect (`study_id` `34a9a5216a8aac36`,
  `artifacts/gate3-late-placement-380.jsonl`, summary in `artifacts/study-summaries/`).
- Added `--mutation-interval` and `--mutation-budget` to the experiment CLI
  (were config-only).
- **384-seed variance study (seeds 6000-6383, cap=2 candidates, 4 probe batches,
  3 events, steps=144, budgets matched)** -> quality FAILED: 179/384 wins (46.6%,
  threshold 209), mean delta `-0.003907`, binomial p `0.9159`, t p `0.3453`;
  probe cost 16.0% (descriptive). Probe variance reduction alone does not restore
  the adaptive edge (`study_id` `cdebaf2ed9f23f54`,
  `artifacts/gate3-variance-384.jsonl`, summary in `artifacts/study-summaries/`).
- **384-seed horizon replication (seeds 8000-8383, the passing cap-4/2-batch
  config at steps=144, budgets matched)** -> quality PASSED decisively: 231/384
  wins (60.2%, threshold 209), mean delta `-0.052402`, binomial p `4.04e-05`,
  t p `1.07e-08`; probe cost 14.0% (41.9%/3, matching the amortization model);
  clears Bonferroni alpha/7 ~ 0.0071 on both statistics
  (`study_id` `162362a15b1d6aee`, `artifacts/gate3-horizon-384.jsonl`, summary in
  `artifacts/study-summaries/`).
- **384-seed cadence study (seeds 5000-5383, steps=144, interval=24, budget=6,
  cheap probe cap=2/1-batch, budgets matched)** -> 137/384 adaptive wins (35.7%,
  threshold 209), mean delta `+0.047014`, binomial p `1.0`; adaptive significantly
  WORSE than random (z `-5.61`, two-sided p ~ `2e-08`); probe cost 8.41%
  (descriptive only - the 5% cap was not part of this pre-registration)
  (`study_id` `964e3f08e13a3710`, `artifacts/gate3-cadence-384.jsonl`, summary in
  `artifacts/study-summaries/`).
- **Probe-landscape instrumentation** (`--probe-landscape`, `--probe-force-candidate`,
  `--probe-force-events`) records every candidate's probe loss per mutation event
  and supports deterministic counterfactual branch runs; forced index 0 is verified
  trajectory-identical to greedy selection. Probe-validity counterfactual study
  (384 seeds x 4 candidate branches at event 1) recorded in
  `artifacts/study-summaries/probe-validity-analysis.json`.
- Gate studies now fail fast with a clear error when the `random` baseline arm is
  missing from `--arms`.
- Fixed: `injectGrokPwaHead` now prioritizes document title over site.json title for og:title.
- Fixed: preview test port assertions updated to match PREVIEW_PORT=4174.

## Environment Notes

- Git was initialized on 2026-09-12; public project baseline committed as `90416f3`.
- The app uses port `4173` because port `8080` is occupied by an unrelated local service.
- The installed PyTorch CUDA build does not support the local GTX 1070, so Python validation
  currently runs on CPU with a warning.
- `startup.sh` has a separate preview-process exit-code issue to clean up; it is not part of
  the scientific implementation and should not be solved by killing the unrelated 8080 service.
