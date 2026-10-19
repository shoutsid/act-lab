# Adaptive Computational Topology (ACT)

ACT is a research codebase for one question: **can a language model's computational topology be learned during pretraining rather than fixed in advance?** The repository contains the model, the training and evaluation code, the topology operations that mutate it, and the full experimental record behind every claim.

The honest summary is that the original hypothesis is not established. Random topology churn did not reliably improve real-text training, and the probe-based adaptive selector did not beat random churn. The project pivoted to a narrower question that the evidence does answer: **does architecture materially affect training efficiency at these scales, and is a better architecture findable by search?** See [Findings](#findings) and [Status](#status).

## Contents

- [Findings](#findings)
- [Status](#status)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Reproducing the results](#reproducing-the-results)
- [Repository layout](#repository-layout)
- [Documentation](#documentation)
- [Limitations](#limitations)
- [License](#license)

## Findings

| Question | Result | Where it is recorded |
|---|---|---|
| Does architecture matter at matched parameter budget? | Yes. An 8-candidate design reaches a 0.2462-nat spread against 0.0148 nats of seed noise (exact seed-matched `p <= 5e-06`). | `artifacts/ARCH_SEARCH_FINDINGS.md` |
| Which lever? | Residual-stream width. Hidden size contributes **−0.1836 nats**; depth contributes **+0.0783 nats**, with shallow beating deep. Both reproduce on fresh seeds. | `artifacts/ARCH_SEARCH_FINDINGS.md` |
| Do the two interact? | Yes. The depth penalty depends on width (interaction **+0.0827 nats**, six-seed replication), so a search must mutate depth and width jointly. | `artifacts/ARCH_SEARCH_FINDINGS.md` |
| Do the findings survive longer training? | The shallow-beats-deep effect persists at 10× the training length (**+0.0209 nats**, exact `p = 0.003086`). The winner changes; the ranking holds (Spearman 0.90). | `artifacts/arch-zoo-100m-v3-crossed-horizon10x/` |
| Does active-parameter count explain loss? | No. Across a ×4.5 range under a matched budget it shows no association with loss. | `artifacts/ARCH_SEARCH_FINDINGS.md` |
| Can a small MoE learn real English? | Yes, as a proof of the training path. A 20.9M-parameter MoE trained on 98.3M tokens of TinyStories reaches **1.6228 nats** held-out loss (perplexity 5.07), against a uniform floor of 9.0109, a unigram model at 5.9681 and a bigram at 7.2348. | `artifacts/real-lm/report.json` |

Every number above is recomputed from the committed ledgers by `scripts/verify_record.py`, not copied from prose.

## Status

- **Research direction:** the architecture-search phase is the active line. Further scaling is gated on the width-control design in `docs/width-control-design.md`.
- **Gate 4 (Kaggle T4×2 campaign):** execution is in progress. The live state is recorded, with dates, in the newest entries of `docs/IMPLEMENTATION_STATUS.md`, which is the authoritative source for it. This README does not track live numbers.
- **Code-model distillation:** an experiment comparing distillation against plain cross-entropy for a small code model is scripted (`scripts/run_code_lm_campaign.sh`). Its headline table is written only when the campaign completes.

## Architecture

```text
┌──────────────────────────────────────────────┐
│ RESEARCH CONTROLLER                          │
│ evolutionary topology, experiments, cost     │
├──────────────────────────────────────────────┤
│ TOPOLOGY SYSTEM                              │
│ grow, prune, merge, split, rewire            │
├──────────────────────────────────────────────┤
│ MODEL CORE                                   │
│ attention, gated delta net, MoE, residual    │
├──────────────────────────────────────────────┤
│ TRAINING ENGINE                              │
│ PyTorch, CUDA 12.6 (T4x2 on Kaggle)          │
└──────────────────────────────────────────────┘
```

## Features

- **Topology mutation.** `prune_expert`, `grow_expert`, `clone_expert`, `merge_experts` and `split_expert` act on MoE experts; `rewire_layer` changes a layer's residual input edge; `change_module` swaps a layer's sequence mixer or feed-forward type; `toggle_layer` changes effective depth. Uniform active-expert invariants are enforced at build time.
- **Importance metrics.** Routing load, parameter and gradient norms, Fisher approximations (`fisher_ema`), functional redundancy, temporal stability, and causal ablation (`expert_ablation_delta`).
- **Prior-ranked module search.** A `SearchPrior` ranks a genome's layer and module candidates deterministically through `rank_module_candidates`, with a rationale for each candidate. No external model artifacts are bundled.
- **Named mixer layouts.** The `gdn_3to1` schedule (three gated-delta-net layers to one attention layer) among others, with a committed throughput benchmark in `artifacts/mixer-benchmark/`.
- **Deterministic token shards.** Checksum-addressed synthetic shards with bounded reads (`build_synthetic_shards`, `read_tokens`), the contract real binary shards will follow.
- **Kaggle-ready execution.** Stateless resume, dataset sharding with checksums, and safety budgets for T4x2 sessions.

## Installation

Requires Python 3.10 or newer. The project pins the CUDA 12.6 PyTorch wheel so Pascal GPUs such as the GTX 1070 (compute capability 6.1) receive `sm_61` kernels.

```bash
uv sync --extra dev
```

Use `--device cpu` on any command to force CPU execution, which is what the deterministic studies use.

## Quick start

Train a small topology model for a few steps:

```bash
PYTHONPATH=src uv run python -m act.training.cli --steps 8
```

Run the Experiment 0 topology gate, which compares fixed, random-churn and adaptive arms under identical batches and mutation budgets:

```bash
PYTHONPATH=src python -m act.evolution.cli --seeds 1,2,3,42 --steps 48
```

Train and sample the small language model on TinyStories:

```bash
PYTHONPATH=src python scripts/train_real_lm.py --max-steps 12000
PYTHONPATH=src python scripts/real_lm_infer.py --demo
```

The checkpoint is a run output and is not committed, so `real_lm_infer.py` needs the training command to have run first.

## Reproducing the results

Run the test suite:

```bash
PYTHONPATH=src python -m pytest -q
```

The suite runs in parallel by default (`pytest-xdist`); pass `-n 0` for a serial run.

Check that every documented claim still recomputes from its ledger, and that each committed record regenerates from its own documented command:

```bash
PYTHONPATH=src python scripts/verify_record.py
PYTHONPATH=src python scripts/check_record_freshness.py --strict-rank
```

To check the same things against a clean clone of `HEAD` (the view a reader of the repository gets), run:

```bash
PYTHONPATH=src python scripts/check_fresh_clone.py
```

Analysis of a sweep is run with `scripts/arch_zoo_analyze.py`; the screening and confirmation commands for each arm are in that arm's `NOTES.md`.

## Repository layout

```text
src/act/            the library
  config/           configuration schemas
  data/             deterministic token shards and readers
  evaluation/       metrics and held-out evaluation
  evolution/        the topology gate and its CLI
  importance/       importance metrics (routing load, Fisher, ablation)
  model/            the MoE causal language model
  ops/              Kaggle remote execution and handoff publishing
  search/           module ranking and architecture-search reporting
  topology/         mutation operations (prune, grow, clone, merge, split, rewire)
  training/         training engine, session runner and CLI
scripts/            research instruments and campaign drivers
tests/              the test suite
notebooks/          generated notebooks (rebuilt by scripts/build_notebooks.py)
docs/               plan, research report, status log, runbooks and design notes
artifacts/          committed experiment ledgers, records and analyses
```

## Documentation

The standing documents are under `docs/`. Each has one job:

- `docs/complete_implementation_plan.md` is the plan of record: the research questions, hardware strategy, model and search-space design, and gates. It is prescriptive.
- `docs/RESEARCH_REPORT.md` is the narrative report of results, with each study's pre-registration beside its outcome.
- `docs/IMPLEMENTATION_STATUS.md` is the implementation log. It is append-only and newest-first.
- `docs/T4_RUNBOOK.md` is the operational runbook for Kaggle sessions.
- `docs/handoff-sharding-design.md` describes how checkpoint handoffs are split across Kaggle dataset versions.
- `docs/width-control-design.md` is a design note for the next gate: width mutation and its pre-registered outcomes.
- `docs/training-throughput.md` is a measurement note on where a training run's wall-clock time goes.
- `docs/CONTRIBUTING.md` describes setup, the checks a change must pass, and the record conventions.

Per-arm material lives beside each experiment under `artifacts/`:

- `artifacts/ARCH_SEARCH_FINDINGS.md` synthesises the architecture-search evidence.
- `artifacts/*/NOTES.md` holds each arm's pre-registration and the command that reproduces its record.
- `artifacts/*/decision.md` is generated from the ledgers; regenerate it rather than editing it.
- `artifacts/*/analysis.md`, `artifacts/*/FINDINGS.md`, `artifacts/*/PERFORMANCE_NOTES.md` and `artifacts/*/SCREENING_AUDIT.md` are per-arm and per-benchmark notes. They are not machine-checked.

The index above is itself checked by `tests/test_doc_index.py`.

**Where to start.** Read `docs/RESEARCH_REPORT.md` for the findings, then the newest entries of `docs/IMPLEMENTATION_STATUS.md` for the current state. The plan in `docs/complete_implementation_plan.md` is the reference for what was intended, not what was built.

## Limitations

- **Scale.** The largest models here are about 100M parameters, trained for a fraction of the token budget of comparable published models. The findings are about relative architecture effects at these scales, not absolute capability.
- **Hardware.** The local development card is a GTX 1070 (fp32 in practice). The Kaggle T4 path runs fp16. Comparisons whose effect size is near the measured fp16 difference are kept in fp32.
- **Data.** The language-model proof uses TinyStories, a children's corpus. It shows that the training path works, not that the architecture hypothesis holds.
- **Negative results are recorded, not hidden.** The original random-churn and adaptive-selector hypotheses did not hold. Several effects stay open or are reported as not established.

## License

Copyright (c) 2026 James Summerill. All rights reserved. No part of this repository may be used, copied, modified, distributed or sold without prior written permission from the copyright holder. See `LICENSE`.
