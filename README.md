# Adaptive Computational Topology (ACT)

An experimental language-model pretraining framework where **computational topology is learned during pretraining**, rather than being fixed in advance.

## Core Hypothesis
A training process that jointly optimizes weights and computational topology can discover architectures with a superior capability/FLOP frontier compared to manually designed architectures of equivalent capacity.

## Architecture
```text
┌──────────────────────────────────────────────┐
│              RESEARCH CONTROLLER             │
│  evolutionary topology / experiments / cost │
├──────────────────────────────────────────────┤
│              TOPOLOGY SYSTEM                 │
│ grow / prune / merge / split / rewire       │
├──────────────────────────────────────────────┤
│                MODEL CORE                    │
│ attention / GDN / MoE / residual / depth    │
├──────────────────────────────────────────────┤
│             TRAINING ENGINE                  │
│ PyTorch + T4x2 / JAX + TPU v5e              │
└──────────────────────────────────────────────┘
```

## Features
- **Dynamic Topology Mutation**: Prune, Grow, Clone, Merge, Split, Rewire operations on layers, attention heads, MoE experts, and residual routes.
- **Multi-Level Importance Metrics**: Activation counters, gradient norms, Fisher approximations, functional redundancy (CKA / cosine), temporal stability, and causal ablation.
- **Evolutionary Search**: Pareto frontier optimization (Quality vs Compute/FLOPs), population diversity management, entropy tracking.
- **Kaggle-Optimized**: Designed for T4x2 environments with stateless resume, dataset sharding, checksums, and safety budgets.
- **Qwen-Informed Prior**: Prior extraction and structural analysis tool for hybrid architectures.

## Quick Start
```bash
pip install -e .
PYTHONPATH=src python -m act.training.cli --steps 8
```

For Pascal GPUs such as the GTX 1070 (compute capability 6.1), install the
CUDA 12.6 PyTorch build rather than a newer CUDA build that omits `sm_61`:

```bash
python -m pip install --force-reinstall torch==2.14.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

The browser research console uses the existing npm toolchain:

```bash
npm install
npm run dev
```

## Experiment 0 gate

The first executable topology gate compares fixed, random-churn, and adaptive
static-MoE arms under the same synthetic batches and mutation budget:

```bash
PYTHONPATH=src python -m act.evolution.cli --seeds 1,2,3,42 --steps 48
```

Experiment 0 defaults to CPU for comparable deterministic studies. Use
`--device cuda` to run a local study on a compatible NVIDIA GPU:

```bash
PYTHONPATH=src python -m act.evolution.cli --steps 48 --device cuda
```

The command prints and appends one `gate_study` record to
`artifacts/experiments.jsonl`. For a single append-only experiment record, omit
`--seeds` and choose an output path with `--output`.

The current low-fidelity study is reproducible and reports adaptive wins, loss
delta, architecture hashes, mutation fossils, importance records, and compute
accounting. The genome also supports guarded merge, split, and residual-input
rewire operations for topology-contract tests; the baseline model currently
executes the static MoE subset. The latest 16-seed run produced 9 adaptive wins,
and the stabilized eight-seed run averages four held-out batches and produced
5 adaptive wins with mean paired delta `-0.003346`. The broader Gate 3 remains
open and scaling is intentionally blocked.

The checkpoint resume contract is covered by:

```bash
PYTHONPATH=src python -m pytest tests/test_dense_baseline.py -q
```

The test requires interrupted training followed by checkpoint restore to match
continuous training parameter-for-parameter. This is the baseline contract
before adding a static MoE or topology mutation to the real training runtime.
