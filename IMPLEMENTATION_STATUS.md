# ACT Implementation Status

Updated: 2026-09-12

This document compares the current workspace against
`Adaptive Computational Topology - Complete Kaggle Implementation Plan.md`.
It is the implementation tracker for the narrowed first scientific milestone.

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

The first real scientific result is partially demonstrated: Python adaptive topology
beats Python random churn in 6/8 low-fidelity seeds under identical tokens, mutation
schedule, and mutation budget, but the universal gate is intentionally still open.

## Milestone Comparison

| Plan milestone | Status | Evidence / gap |
|---|---|---|
| 1. Foundation and baseline contract | DONE for the first gate | Python model, optimizer, training state, topology genome, hashing, model factory, and versioned checkpoint metadata are covered. Large-model and real-data contracts remain outside this gate. |
| 2. Dense baseline | PARTIAL | Dense causal LM trains on deterministic synthetic batches and resumes parameter-for-parameter. It is a tiny gate model, not the planned 256M-350M research model; validation loss, FLOPs, and real data metrics are not yet recorded. |
| 2. Static MoE baseline | DONE for the first gate | Static top-k MoE trains through the shared runtime and reports validation loss, active parameters, parameter count, and validation FLOPs. |
| 3. Topology graph and safety | PARTIAL | Python genome serialization, validation, hashing, prune/grow, expert-ID weight preservation, and accepted mutation fossils exist. Merge, split, rewire, and post-mutation NaN gates remain deferred. |
| 4. Random topology control | DONE for the first gate | Python Experiment 0 provides reproducible random churn with matched cadence, mutation count, training/search accounting, and report metrics. Real data and scale remain deferred. |
| 5. Importance instrumentation | DONE for the first gate | Python records routing/load, gradient norm, parameter norm, Fisher-style EMA, temporal stability, mutation count, and genome hash at structural windows. |
| 6. Adaptive controller | PARTIAL | Python routing-led adaptive prune/grow mutates real static-MoE models and consumes persisted routing records. The validated default reaches 6/8 wins; routing, gradient, Fisher, and parameter-based expert pruning were probed and did not improve it. |
| 7. Experiment ledger | DONE for the first gate | Browser ledger and Python append-only JSONL registry store config, dataset/tokenizer hashes, code revision, hardware, tokens, FLOPs, arm metrics, mutation fossils, and architecture hashes. |
| 8. Final gate: adaptive > random | OPEN | The slower 48-step / 12-step schedule and simplified routing-led controller reproduce 3/4 adaptive wins across seeds 1, 2, 3, and 42. A universal or larger-seed pass is still required before scaling. |

## Plan Sections 3-6: Compute and Kaggle Operations

| Area | Status | Remaining work |
|---|---|---|
| T4x2 PyTorch/DDP | TODO | Add device selection, DDP launch, gradient accumulation, mixed precision, and activation checkpointing after the single-process gate. |
| TPU v5e/JAX | DEFERRED | Port only after a stable architecture wins the GPU gate. |
| Kaggle artifact datasets | TODO | Add checkpoint, experiment, evaluation, and final-model publishing layouts. |
| Session resume protocol | PARTIAL | Local checkpoint resume is deterministic. Kaggle dataset restore, checksums, shard position, and publish-on-exit are absent. |
| Repository structure | PARTIAL | Core Python directories now exist; data, evaluation, evolution, and importance implementations are still empty. |

## Plan Sections 8-24: Model and Topology

| Area | Status | Remaining work |
|---|---|---|
| V0 256M-350M model | TODO | Scale beyond the tiny deterministic test model only after the mechanism is proven. |
| Attention/GDN/hybrid model core | PARTIAL | TypeScript simulator represents these module types; Python runtime currently supports dense and static MoE only. |
| Full genome contract | PARTIAL | Python genome supports dense/MoE expert layouts; heads, modules, widths, residual paths, context, and compute ceilings need to be represented. |
| Weight/topology separation | DONE for the first gate | Model and optimizer remain separate from the Python genome, while versioned checkpoints and Experiment 0 fossils package genome and mutation metadata together. |
| Importance levels and lifecycle | PARTIAL | Simulator implementation and Python running statistics/temporal stability exist; broader lifecycle analysis remains. |
| Expert ecology and similarity graph | TODO | No Python routing metadata, functional sketches, graph, or causal ablation yet. |

## Plan Sections 25-42: Data, Training, and Reproducibility

| Area | Status | Remaining work |
|---|---|---|
| Real tokenizer and shards | TODO | Add fixed tokenizer, binary shards, checksums, deterministic ordering, and dataset manifests. |
| Real language-model validation | TODO | Add held-out batches, perplexity/validation loss, capability probes, and FLOPs accounting. |
| Complete checkpoint format | DONE for the first gate | Versioned local checkpoints persist model/optimizer/training RNG state plus scheduler, genome, topology history, importance, dataset position, config, and revision metadata. |
| Experiment registry | DONE for the first gate | Append-only JSONL records include experiment/config/dataset/tokenizer hashes, hardware, tokens, FLOPs, architecture hashes, and arm results. |
| Reproducibility hashes | DONE for the first gate | Genome, config, dataset, tokenizer, and complete experiment hashes are recorded. |
| Research notebooks/scripts | PARTIAL | Dense CLI exists; add baseline, random churn, measurement, adaptive, extraction, retrain, and compare entry points. |

## Plan Sections 43-66: Scientific Experiments

| Experiment | Status |
|---|---|
| Experiment 0: fixed vs random vs adaptive | PARTIAL | Matched-budget Python runner, metrics, registry, and a reproducible 3/4 adaptive four-seed probe exist; universal Gate 3 is still open. |
| Conventional dense/static-MoE frontier | PARTIAL |
| Dynamic sparse prune/grow | PARTIAL | Implemented in the Python Experiment 0 runner for safe expert prune/grow; richer topology operators remain. |
| Expert ecology clone/split/merge/death | TODO |
| Module-type evolution | DEFERRED |
| Adaptive depth | DEFERRED |
| Hierarchical search | DEFERRED |
| Qwen-informed search | DEFERRED |
| Teacher initialization | DEFERRED |
| Architecture reusability and fresh initialization | DEFERRED |
| Cross-seed stability and architecture equivalence | DEFERRED |
| Compute-efficiency scaling | DEFERRED |

## Plan Sections 67-105: Scale and Final Deliverables

These remain TODO or DEFERRED until the first gate passes:

- full metric and visualization report;
- topology fossils and trajectory plots;
- Pareto population search and diversity preservation;
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
7. IN PROGRESS: repeat Gate 3 across more seeds and longer low-fidelity runs.
8. Do not add merge, split, GDN, Qwen priors, or scale until Gate 3 passes.

## Verified Checks

- `PYTHONPATH=src python -m pytest -q` -> 13 passed after the current milestone edits.
- `PYTHONPATH=src python -m compileall -q src/act tests` -> passed.
- `npm run typecheck` -> passed.
- `npm run build` -> passed.
- Python Experiment 0 mutation-budget test -> passed.
- Python registry/CLI round trip -> passed.
- Initial Gate 3 trial -> adaptive lost to random on seeds 1, 2, 3, and 42.
- Four-seed gate test -> adaptive won 3/4 seeds with equal three-mutation budgets.
- Eight-seed generalization probe -> adaptive won 6/8 seeds; Gate 3 remains open pending a stronger pass.
- Expert-signal sweep -> random expert selection remained strongest at 6/8; routing, gradient,
  Fisher, and parameter-norm choices were retained as configurable probes but not selected by default.
- Browser ACT console and research invariants -> passed with no console errors in the latest checks.

## Environment Notes

- Git was initialized on 2026-09-12; no files were staged or committed.
- The app uses port `4173` because port `8080` is occupied by an unrelated local service.
- The installed PyTorch CUDA build does not support the local GTX 1070, so Python validation
  currently runs on CPU with a warning.
- `startup.sh` has a separate preview-process exit-code issue to clean up; it is not part of
  the scientific implementation and should not be solved by killing the unrelated 8080 service.