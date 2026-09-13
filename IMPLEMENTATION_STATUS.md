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
candidate richness + probe quality. (10) The pre-registered richness study
then isolated that axis at fixed nominal probe tokens (192/event): arm A
cap-4/1-batch PASSED (228/384 wins, mean delta `-0.036982`, binomial p
`1.4e-04`, t p `5.0e-05`) while same-seed arm B cap-2/2-batch FAILED
(193/384, mean delta `+0.007156`, binomial p `0.47965`), and the paired
A-vs-B comparison confirms A better in 225/384 pairs (sign z `3.37`,
two-sided p `8e-04`). Candidate richness is the active axis; probe batches
matter little within the rich-cap regime (A's delta `-0.037` vs the
2-batch replication's `-0.052`, two-sample z ~ 1.1, n.s.). Gate-style quality
tests now number nine, with three passes, all from the cap-4 family.

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
| 8. Final gate: adaptive > random | PASSED three times (cap-4 family at 48 and 144 steps); richness identified as the active axis; cheaper/cadence/narrow variants FAILED | Factor map (all matched budgets, objective verdicts): cap-4/2-batch, 48 steps -> PASS 213/384, p `0.0181`, at 41.9% cost. Cap-2/1-batch, 144 steps -> FAIL 181/380, p `0.8351`, at 4.13% cost. Placement (late, same seeds) -> no effect (p `0.0788`). Cap-2/1-batch, 6 events -> 137/384, significantly WORSE than random (z `-5.61`). Cap-2/4-batch -> 179/384, p `0.9159`. Horizon replication cap-4/2-batch, 144 steps -> PASS 231/384, p `4.04e-05`, t p `1.07e-08`, cost 14.0%. Richness A cap-4/1-batch, 144 steps -> PASS 228/384, p `1.4e-04`, t p `5.0e-05`, cost 7.3%; same-seed B cap-2/2-batch -> FAIL 193/384, p `0.47965`; paired A>B 225/384 (z `3.37`). Probe validity real but weak (rho `0.11`); richness is the active axis, probe batches secondary. Nine gate-style tests, three passes; Bonferroni alpha/9 ~ 0.0056 is cleared by the 144-step replication and richness-A on both statistics. |

## Plan Sections 3-6: Compute and Kaggle Operations

| Area | Status | Remaining work |
|---|---|---|
| T4x2 PyTorch/DDP | PARTIAL | Add device selection and DDP launch helpers now discover torchrun rank/world-size, initialize an appropriate process group, and wrap models only for multi-process runs; scheduler, accumulation, autocast, and activation checkpointing are available. T4x2 validation remains. |
| TPU v5e/JAX | DEFERRED | Port only after a stable architecture wins the GPU gate. |
| Kaggle artifact datasets | DONE for the 5.2 handoff | Deterministic checkpoint, evaluation, report, and manifest layout contracts exist; the V0 5.2 per-seed handoff (`act.training.v0_handoff`: atomic publish, checksum-verified restore, bit-identical resume through the dataset round trip) is implemented, and the remote half is now code: `act.ops.kaggle_remote` wraps the optional kaggle package (injectable client, zero-copy push of the handoff directory with generated metadata, restore merges manifest-complete seed versions over matching locals and keeps local-only ones) with `--kaggle-dataset owner/slug` on the launch command - restore at session start (fail-fast before any training), publish-on-exit that never raises (outcome recorded in `remote-publish-log.json`). |
| Session resume protocol | DONE for the 5.2 handoff | Local checkpoint resume is deterministic for baselines and bit-identical for V0 sessions (verified mid-run resume: losses, FLOPs counters, architecture hashes, and mutation histories all reproduce exactly); the 5.2 versioned-dataset handoff round-trips a seed working-dir -> dataset -> working-dir bit-identically, and the remote leg now exists (`--kaggle-dataset` restore at session start, publish-on-exit; remote seed versions merge over matching locals, local-only seeds are kept, partial remote versions are ignored). |
| Repository structure | PARTIAL | Core Python data, evaluation, evolution, importance, model, topology, training, and Kaggle-artifact implementations exist with focused tests; all ten plan section 39 notebooks (`00`-`09`) now exist under `notebooks/` (deterministic builder, tested), while standalone research entry points remain. |

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
| Compute-efficiency scaling | PRE-REGISTERED (execution path built; pre-launch tooling done) | Real-scale measurement protocol pre-registered in `RESEARCH_REPORT.md` section 5.1: declared 285M V0 config, predicted probe cost 0.094% (band 0.047-0.189%), 5% criterion + tolerance band + harm guard on 12 fresh seed pairs (9000-9011); PASS authorizes Gate 4. The V0 training path is implemented: declared `V0MoECausalLM` reproduces 285,336,576 / 134,341,632 parameters exactly, per-layer expert genomes rebuild weight-preserving after grow/prune, and `act.training.cli --mode v0` runs resumable sessions with version-4 checkpoints (FLOPs counters, genome, generator state persisted). Session integration checks live in `scripts/v0_session_selfcheck.py` (run on demand, NOT in the pytest suite). GPU execution plan pre-committed in section 5.2: per-GPU fp32 fit ~4.5 GB of 16 GB, 3.19 GB checkpoint payloads, `max_checkpoints=2` (pair steady state ~14.9 GB of the 20 GB working dir), throughput-gated dtype rule (fp32 floor 2,500 tokens/s, fp16 fallback), parameterized session schedule with atomic pairs and deadline/publish rules, and the versioned-dataset checkpoint handoff (~45 GB ledger). The 5.2 pre-launch tooling is now code: `scripts/v0_throughput_benchmark.py` (declared-model step-loop benchmark emitting the dtype decision and schedule artifact) and `act.training.session_plan` (registered schedule/deadline/working-dir arithmetic, unit-tested against the 5.2 tables); declared sessions now train with checkpointed activations as specified, with resume re-verified bit-identical; `act.training.v0_runner.run_pair_session` ties benchmark numbers, deadline rules, session training, and the dataset handoff into the per-seed session loop (atomic pairs, in-progress snapshots on deadline stops, restart-from-scratch, skip-completed idempotence), and `--mode pair-session` exposes it as the single launch command with the dtype rule enforced against the benchmark artifact; the fp16 contingency trains under autocast + GradScaler with scaler state in the version-5 checkpoint contract. |

## Plan Sections 67-105: Scale and Final Deliverables

These remain TODO or DEFERRED until the first gate passes:

- full metric and visualization report (PARTIAL: the four pre-registered
  factor-map figures now exist under `artifacts/study-summaries/figures/`,
  generated reproducibly from the versioned summaries; trajectory plots and
  real-scale metric visualizations remain);
- topology fossils and trajectory plots;
- Multi-generation Pareto population search and richer diversity preservation;
- T4x2 scaling to 256M-350M;
- Qwen analyzer and architecture priors;
- fresh/rewound/teacher retraining;
- 500M-1B scaling;
- TPU port;
- paper-quality comparisons and final research report;
- versioned Kaggle datasets and reproducible notebooks (PARTIAL: all ten
  plan section 39 notebooks - `00_environment` plus `01`-`03` per section
  104 and `04_importance`, `05_adaptive`, `06_search`, `07_extract`,
  `08_retrain`, `09_compare` - now exist under `notebooks/`, emitted by the
  deterministic builder `scripts/build_notebooks.py` (byte-identical
  re-runs, unit-tested in `tests/test_notebooks.py`, every cell executed
  end-to-end); the remote dataset publishing is now code as well
  (`act.ops.kaggle_remote` + `--kaggle-dataset`, below) with the remaining
  piece being a real T4-credentialed round trip);

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
- `PYTHONPATH=src python -m act.training.cli --mode pair-session ...
  --kaggle-dataset owner/slug` -> the remote publish-on-exit wrapper
  (`act.ops.kaggle_remote`), **live-verified against the real Kaggle API
  with credentials**: publish -> download -> bit-identical seed handoff
  round trip (all manifest checksums re-verified on the returned copy),
  and the registered one-line launch command completes restore ->
  skip-complete -> exit publish with exit 0. The live probes established
  the upload layout the adapter implements: Kaggle datasets are flat, but
  the service auto-extracts uploaded tar/zip blobs preserving internal
  paths, and the package's default directory handling is unusable for
  nested trees (`skip` drops them silently; `tar` archives contents
  without a name prefix, so multiple seeds would collide) - the adapter
  therefore stages one tar per seed with paths rooted at `seed-NNNN/...`
  and pushes them flat. A nonexistent dataset (session 1) is the normal
  bootstrap state on restore (warn and train from scratch; atomic pairs
  make it safe) and falls back to `dataset_create_new` on publish; any
  other error is fail-fast at launch and never-raising at exit. The test
  dataset was deleted after the round trip; unit tests still inject a
  fake client, so the kaggle package is never needed by the suite.
8. NEXT: Gate 3 quality is established for the cap-4 controller family (three
   passes across horizons and probe-batch counts; richness is the active axis).
   The remaining blocker is compute-efficiency: cap-4/1-batch cuts probe cost to
   7.3% at toy scale and amortizes with horizon, but real-scale measurement is
   the plan's deferred compute-efficiency milestone and gates Gate 4 (adaptive >
   static MoE). Controller default stays cap-4/2-batch; cap-4/1-batch is the
   validated cost-saving option. The real-scale measurement is now
   **pre-registered** (`RESEARCH_REPORT.md` section 5.1): declared V0 config
   (285,336,576 total / 134,341,632 active parameters), predicted probe cost
   0.094% of training FLOPs under the measured per-event identity, decision
   rules fixed (5% criterion + factor-2 tolerance band + anti-adaptivity harm
   guard), seeds 9000-9011 reserved. Running it as specified - without retuning
   knobs against the result - is the entry condition for Gate 4. The section 5.2
   pre-launch tooling (throughput benchmark, session scheduler, deadline rules,
   working-dir ledger, versioned-dataset handoff) is now code and unit-tested,
   and the pieces are tied together by `act.training.v0_runner.run_pair_session`,
   the per-seed session loop over the declared queue 9000-9011 (atomic pairs,
   rule-1 gating, cooperative deadline stops with in-progress snapshots,
   restart-from-scratch, skip-completed idempotence). The launch command is
   `python -m act.training.cli --mode pair-session`: it reads the session-1
   benchmark artifact, re-derives and cross-checks the dtype decision,
   applies the registered max_checkpoints=2 pin, and runs the session loop
   with the dataset handoff. The registered fp16 contingency is executable:
   fp16 sessions train under autocast with a GradScaler (checkpoint contract
   v5 carries the scaler state; fp16 resume verified bit-identical, version-4
   payloads still load in fp32). What remains before launch is running the
   benchmark on the T4 to fix the measured dtype and schedule. The launch
   path is now reproducible-notebook-ready: all ten plan section 39
   notebooks (`00_environment` through `09_compare`) exist under
   `notebooks/` (deliverable 2), so a session can verify the compute
   contract before `--mode pair-session` and reproduce the baseline, churn,
   gate-study, importance, search, extraction, retraining, and comparison
   claims from the same committed revision.

## Verified Checks

- `PYTHONPATH=src python -m pytest -q` -> all unit tests pass (206, including the
  26 report-figure data-assembly tests, 14 session-plan tests pinning the
  section 5.2 schedule/dtype tables, 10 handoff tests covering atomic
  publish, checksum rejection, and bit-identical resume through the dataset
  round trip, 9 runner tests covering deadline gating, cooperative
  interrupt + in-progress snapshot, restart-from-scratch, and
  skip-completed idempotence, 22 notebook-builder tests pinning determinism,
  nbformat validity, and the section 104 claims, and fp16/CLI tests covering
  the fp16 session
  round trip, the version-4 compatibility path, and the pair-session launch
  command's artifact/dtype guards); session runs are deliberately excluded
  from the suite (they are integration checks, too slow for per-commit runs) and
  live in `scripts/v0_session_selfcheck.py`, which verifies matched budgets,
  FLOPs booking, bit-identical mid-run resume, and the version-4 checkpoint
  contract on a tiny config.
- `PYTHONPATH=src python scripts/build_notebooks.py` -> writes the ten
  notebooks (00-09) byte-identically; `--check` verifies the committed
  `.ipynb` files never drift from the builder output; every code cell of
  all ten notebooks was executed in order (cell-runner pattern, like the
  self-check): 00 reproduced the declared parameter counts, 01 proved
  train/resume/seed-reproducibility, 02 proved grow/rebuild/prune weight
  and hash round trips plus the registry append, 03 and 09 ran 4-seed
  gate studies to their recorded verdicts (demo-sized, not the powered
  384-seed studies), 04 extracted signals after a real backward pass and
  verified ablation restores parameters, 05 ran the three-arm matched-
  budget experiment with inspectable mutation histories, 06 exercised
  variants/prior ranking/frontier/Pareto selection, 07 froze and published
  the evolved genome through the checksummed manifest, and 08 compared
  fresh/rewound/teacher initializations with exact copied/skipped counts.
- `PYTHONPATH=src python -W error::UserWarning -c ...` ablation check ->
  the temporary-ablation helper now detaches losses before `float()`
  conversion (no tensor-scalar warning); covered by the notebook cell
  executions above.
- `PYTHONPATH=src python scripts/v0_session_selfcheck.py` -> re-passed after the
  declared sessions switched to checkpointed activations (resume stays
  bit-identical; recompute is deterministic because blocks carry no dropout),
  and after the fp16 path landed (step [3/4] verifies fp16 resume is
  bit-identical with scaler state restored, and that fp16 books the same
  FLOPs counters as fp32).
- `PYTHONPATH=src python -m act.training.cli --mode pair-session --profile ...`
  -> the launch command runs the whole chain end-to-end on the local GPU:
  it consumed a real benchmark artifact (smoke profile, 26,987 tokens/s,
  fp32 decision), gated and completed all 12 declared seed pairs, published
  each to the dataset handoff, and wrote the session log; the stale-artifact
  and dtype-override guards were exercised by unit tests.
- `PYTHONPATH=src python scripts/v0_throughput_benchmark.py` -> the declared-model
  benchmark runs end-to-end on the local GPU (285,336,576 / 134,341,632 exact
  parameters, ~1,736 tokens/s with activation checkpointing -> dtype rule
  selects fp16 below the 2,500 floor) and writes `artifacts/v0-benchmark.json`;
  `--profile` smoke-checks on tiny configs or CPU. The T4 decision comes from
  the T4 run.
- `PYTHONPATH=src python scripts/generate_figures.py` -> writes the four factor-map
  figures to `artifacts/study-summaries/figures/` (fig1 win-rate forest,
  fig2 quality-vs-cost, fig3 delta distributions, fig4 probe-rank heatmap),
  byte-identical across re-runs.
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
- **384-seed richness study (seeds 7000-7383, fixed nominal probe tokens of
  192/event, 3 events, steps=144, budgets matched)** -> arm A cap-4/1-batch
  PASSED: 228/384 wins (59.4%), mean delta `-0.036982`, binomial p `1.4e-04`,
  t p `5.0e-05`, cost 7.3% (`study_id` `994282a17955597e`); same-seed arm B
  cap-2/2-batch FAILED: 193/384, mean delta `+0.007156`, binomial p `0.47965`,
  cost 8.1% (`study_id` `de2f894d75eaedac`); paired A>B in 225/384 pairs
  (sign z `3.37`, two-sided p `8e-04`). Candidate richness is the active axis
  (`artifacts/gate3-rich-384.jsonl`, `artifacts/gate3-narrow-384.jsonl`,
  summaries in `artifacts/study-summaries/`).
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
- **Report figures generated** (deliverable 12): the four pre-registered figures
  (`artifacts/study-summaries/figures/`) are produced by `scripts/generate_figures.py`
  from the versioned summaries via `act.evolution.figures_data`; the script
  recomputes every verdict (binomial p, pass flag, mean delta, probe-cost
  identity, argmin concordance) from raw per-seed records before plotting and
  fails on any mismatch. Re-runs are byte-identical. Data assembly is unit-tested
  in `tests/test_report_figures.py` (26 tests; statistics helpers pinned to
  scipy reference values); run `PYTHONPATH=src python scripts/generate_figures.py`
  (requires matplotlib, not needed for the test suite).
- Gate studies now fail fast with a clear error when the `random` baseline arm is
  missing from `--arms`.
- Fixed: `injectGrokPwaHead` now prioritizes document title over site.json title for og:title.
- Fixed: preview test port assertions updated to match PREVIEW_PORT=4174.

## Environment Notes

- Git was initialized on 2026-09-12; public project baseline committed as `90416f3`.
- The app uses port `4173` because port `8080` is occupied by an unrelated local service.
- The installed PyTorch CUDA build does not support the local GTX 1070, so Python validation
  currently runs on CPU with a warning.
- Operational hazard (learned 2026-09-13): `V0SessionConfig` defaults are the DECLARED protocol
  dimensions, so a partial config silently instantiates the full 285M-parameter model on CPU
  (~1.1-3.4 GB checkpoints; a test run once accumulated 19 GB in /tmp before hitting the disk
  quota and hanging). Any reduced-fidelity session - tests, self-check, smoke - must set
  vocab/context/hidden/depth/expert dims explicitly; `scripts/v0_session_selfcheck.py` and the
  test suite both do. The `declared=True` guard rejects undeclared dimensions before any
  training work starts.
- `startup.sh` has a separate preview-process exit-code issue to clean up; it is not part of
  the scientific implementation and should not be solved by killing the unrelated 8080 service.
