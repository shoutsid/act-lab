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
uv sync --extra dev
PYTHONPATH=src uv run python -m act.training.cli --steps 8
```

The project pins the CUDA 12.6 PyTorch wheel so Pascal GPUs such as the GTX
1070 (compute capability 6.1) receive `sm_61` kernels. Use `--device cpu` to
override the automatic GPU selection.

## Experiment 0 gate

The first executable topology gate compares fixed, random-churn, and adaptive
static-MoE arms under the same synthetic batches and mutation budget:

```bash
PYTHONPATH=src python -m act.evolution.cli --seeds 1,2,3,42 --steps 48
```

Experiment 0 automatically uses a compatible NVIDIA GPU when one is available.
Use `--device cpu` for comparable deterministic studies:

```bash
PYTHONPATH=src uv run python -m act.evolution.cli --steps 48
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

## Architecture search phase

Random topology churn did not reliably improve real-text training, and the
probe-based adaptive selector did not beat random churn, so the project pivoted
to the question underneath it: **does architecture materially affect training
efficiency at these scales, and is a better architecture findable by search?**

The answer, on the evidence in `artifacts/ARCH_SEARCH_FINDINGS.md`, is **yes**,
and the lever is the residual stream. At a matched total-parameter budget
(×1.02–1.08 across the phase's matched designs, ×1.08 on this one) an 8-candidate
design reaches a 0.2462-nat spread against 0.0148 nats of seed noise, exact
seed-matched `p <= 5e-06` (the Monte-Carlo resolution over 8 candidates x 3 seeds),
surviving both leave-one-architecture-out and a switch from held-out to training
loss. A crossed
factorial then attributes it — **hidden size −0.1836 nats, depth +0.0783 nats with
shallow beating deep**, both at the exact enumeration's floor `p = 4.3e-05` and
both surviving a switch to training loss. The active-parameters hypothesis that
motivated the earlier phase is refuted: across a ×4.5 range under a matched
budget, active parameters show no association with loss, and the fully dense
design places seventh of eight.

That attribution was then **replicated on fresh seeds and the interaction with it**:
three seeds never used to design or screen anything give −0.1851 / +0.0865, and
pooling all six (48 runs) leaves the main effects at −0.1843 / +0.0824 with the
same ranking, while the interaction the three-seed arm could only call a lead
resolves to **+0.0827 nats** with every one of the six seed contrasts agreeing and
the calibrated test at its six-seed floor. The reading is that the depth penalty
depends on the width — so a search has to mutate the two **jointly**, not one at a
time. How a branch like that is decided is pre-registered, and evaluated by
`--replication-rule` rather than by eye.

That `p = 4.3e-05` is worth reading carefully, and the decision record now labels
it: with 46,656 re-splittings the exact test cannot report less than 2/46,656, so
the main effects sit on the *procedure's* floor. The p says the observed
separation is the largest that design can express; the effect *size* in nats is
the quantity that carries the finding. Every test in the project reports its floor
beside its p for the same reason — the horizon arm had already shown what reading
a floor-bound statistic as evidence costs, and the confirmation arm what it costs
the other way round: a rule whose threshold sits *below* the test's floor cannot be
decided by more analysis of the same seeds, only by more seeds.

One confound was recorded against that attribution rather than hidden: this model
family fixes `heads = hidden_size / 64`, so "wider residual stream" and "more
attention heads" are the same knob in every design run, and head count is
**parameter-free** at fixed hidden size — so the two readings implied opposite
advice about where to spend a search budget. A head-paired arm then broke the
confound with two candidates differing in nothing but head count, matched to
*identical* total and active parameters, and the answer keeps the attribution as
given: **head count is worth −0.018 nats pooled (p = 0.14, signs 8/4) while width
holds at −0.1562 nats at the factorial's floor**, so the lever is residual-stream
width rather than a free gene wearing its name. The lead that arm left — a
level-dependent head effect its three seeds floored at 0.25 — was then bought a
second seed block on fresh seeds, and at six seeds the pre-registered rule lands on
**head effect is real at wide width**: −0.0553 nats at hidden 896 (12/12 pairs, at
the 2/2¹² floor) against +0.0084 at 512 (p = 0.37), interaction +0.0636 nats at its
2/2⁶ = 0.031 floor with every seed contrast agreeing. So more heads are a free
~0.055 nats *only* on a wide residual stream, and a search must mutate the two
jointly — the second joint constraint on the space after depth × width.

What none of that yet shows is whether it survives a realistic training horizon;
the depth finding in particular is the one prior work predicts should reverse with
training. That is the arm now running, pre-registered in
`artifacts/arch-zoo-100m-v3-crossed-horizon10x/NOTES.md`: the crossed factorial at
10× the training length, fp16 with a same-precision 1200-step anchor, on Kaggle
T4×2 via `scripts/kaggle_queue.py --queue archzoo`. Its rule is **declared on the
12,000-step rows and refuses any other length**, so the anchor cannot be read as the
horizon result. That is not a hypothetical: the arm's first attempt trained none of
its horizon runs and was reported complete, and the three defects that allowed it are
closed — an admission bar that asked for twice the schedule, a sweep that exits 0
after losing every run in a stage, and a driver that read process exit codes as
completeness rather than reading the ledgers.

The other open question is v1's own, and it is now an arm rather than a caveat. That
arm closed on "NOT ESTABLISHED — this experiment cannot yet tell": a 3-block exact
test whose floor is 1/36, run on finalists that were promoted *on those same seeds*,
so it is mildly anti-conservative in the winner's favour. Its own next step was to
re-run those finalists on seeds never used in screening, and
`artifacts/arch-zoo-100m-v1-finalists-6seed/NOTES.md` pre-registers exactly that —
the three finalists, byte-identical genomes, flat at 1200 steps on fresh seeds, read
*pooled* with v1 so the exact test moves to six blocks and a floor of 1/7776. Its
rule reads that test twice and gates the positive branch on **both**: the pooled
reading raises power, but a pooled test still contains the selection it is meant to
check, so the three screening-free blocks have to name the same architecture first
or the branch reports the ordering as coupled to the seeds that picked it. It runs
on the idle local card in fp32 — Kaggle's ~3× advantage is an fp16 advantage, and
fp16 is ruled out precisely here, because the gaps to resolve (0.017–0.023 nats) are
the size of the fp16 differential this phase measured (≤0.018).

The effect's *size* depends on the training horizon, and that was measured rather
than assumed: the spread decays 1.65× per doubling of training, replicating the
earlier estimate on a controlled design — but seed noise decays with it, so the
ordering is horizon-stable and the same architecture wins at 400, 800 and 1200
steps.

What survives a *short* run is not just the ordering but the explanation: fitting
the genes the factorial identified as a **fixed** model at 400 and at 1200 steps
gives R² 0.927 vs 0.953 with the identified lever individually significant at both
(p = 0.0145 vs 0.00035), on the same eight candidates. A 400-step screen therefore costs ~1.1 GPU-hours against 3.2 for the same
answer, which is the largest available saving in a phase whose GPU budget is now
the binding constraint. Screening eliminates; it does not adjudicate a near tie,
because promotion needs the margin to clear the seed noise — which the halving
plan now prints for each cut, naming the two candidates it could not separate and
any others inside the noise. The threshold is derived, not pinned: it is the
standard error of the difference between two screen means, `std * sqrt(2 / seeds)`,
from the measured seed stds of the matched horizon set (0.0286 nats at 400 steps,
0.0165 at 800, 0.0148 at 1200) — so a two-seed 400-step screen needs a 0.0286-nat
margin, and `--screen-noise-nats` overrides it. `--plan screen` is that loop as a runnable plan — screen every candidate
at `--stage1-steps`, confirm the survivors at `--stage3-steps` — and
`--promote-ties` acts on the print: when the cut sits inside the noise it advances
the whole borderline set, including the candidate the cut landed on, instead of
letting an unresolvable margin decide which candidate is never trained again.

What a matched budget can and cannot identify is derived in §2b of that document
— at fixed total parameters the free genes are not independent, expert count and
width are one knob rather than two, and the obvious factorial (hidden × width)
turns out to be fully confounded with depth, which is checkable by enumeration
before spending GPU hours.

```bash
# screen a matched or crossed zoo, then report on it without retraining
PYTHONPATH=src python scripts/arch_zoo_screen.py --design crossed \
  --cross-factors depth,hidden_size --match-target 78000000 \
  --plan screen --promote-ties --finalist-seeds 42,43,44
PYTHONPATH=src python scripts/arch_zoo_analyze.py --experiment-id <id> --write
```

Analysis runs on a partial sweep: a seed that is still training is dropped from
the crossed and paired contrasts — and from the seed-matched test, where the block
count printed beside the p is the survivors — and **named** in the decision, so an
interrupted arm reports over the seeds that finished instead of losing its
headline test. The same disclosure separates an unfinished seed (fewer rows than
the fullest seed) from an unbalanced one, which is a zoo defect and still refuses. A
confirmation arm's four-outcome rule is applied by `--replication-rule` rather
than read by eye, with its thresholds in `REPLICATION_RULE`. A sweep also says at
launch when another one is already running, because a shared GPU is otherwise
invisible from inside it: candidates just take proportionally longer, and the
budget guard counts tokens rather than hours, so the arm looks stalled rather than
contended. It also states its own plan before training — runs, tokens and projected
GPU-hours — and, when the token budget guard would stop it part way, the stage and
seed at which it would stop, so an arm that does not fit is visible on paper rather
than discovered hours in. `--dry-run` prints the same without touching the GPU.

Section 11's random arm has to match the curated set's *total* parameters, not
just its parameter band. The band is a per-candidate constraint and the arm's
budget is the set total, so they are different quantities: over 200 draws of the
same size, a band-matched draw carries a **median 1.12×** the curated set's total
(p10–p90 1.05×–1.18×, range 1.00×–1.25×, because the curated set sits at 86M
against the nominal 100M — and a median 1.15× even at a ±15% band, so tightening
the band does not buy equal compute). That is more training rather than a fair
comparison:

```bash
PYTHONPATH=src python scripts/space_typicality.py \
  --definition artifacts/arch-zoo-100m-v1/zoo_definition.json \
  --budget-set artifacts/arch-zoo-100m-v1/zoo_definition.json \
  --budget-candidates 13 --target-params 100000000 --tolerance 0.3
```

The record's own numbers are checked the same way as the code is. This recomputes
every load-bearing figure from the ledgers — cost rows, the spread-and-noise table,
the factorial's effects and their floors, the budget match, the structural claims —
and fails when the document and the artifacts disagree:

```bash
PYTHONPATH=src python scripts/verify_record.py
```

The scan prints the launch command for the arm it matched, with every flag that
decides the draw — the band included, because the driver's own default band draws a
different set from the same seed (1.14× the curated budget against the matched
0.9996×). Only then is the random arm trained on the architectures whose budget was
matched.

## Deterministic token shards

The shared data contract can generate checksum-addressed synthetic shards while
the real tokenizer and corpus adapters are implemented:

```python
from act.data import build_synthetic_shards, read_tokens

manifest = build_synthetic_shards("artifacts/synthetic-shards")
batch = read_tokens(manifest.shards[0], start=0, count=2048)
```

The same manifest and bounded-read contract will be used for real binary
training shards.
