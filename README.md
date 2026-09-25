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
│ PyTorch + T4 x2 (CUDA)                      │
└──────────────────────────────────────────────┘
```

## Features
- **Dynamic Topology Mutation**: `prune_expert`, `grow_expert`, `clone_expert`, `merge_experts` and `split_expert` on MoE experts; `rewire_layer` for a layer's residual input edge; `change_module` to change a layer's sequence mixer or feed-forward type; `toggle_layer` for effective depth. Uniform-active-expert invariants are enforced at build time.
- **Multi-Level Importance Metrics**: Routing load, parameter and gradient norms, Fisher approximations (`fisher_ema`), functional redundancy (cosine similarity and per-expert functional sketches), temporal stability, and causal ablation (`expert_ablation_delta`).
- **Evolutionary Search**: Pareto frontier optimization (Quality vs Compute/FLOPs) with frontier-diversity-first population selection.
- **Kaggle-Optimized**: Designed for T4x2 environments with stateless resume, dataset sharding, checksums, and safety budgets.
- **Prior-ranked module search**: a `SearchPrior` (imported module scores plus an explicit `depth_penalty`) ranks a genome's layer/module candidates deterministically through `rank_module_candidates`, with a rationale attached to each candidate. No external model artifacts are bundled or claimed.
- **Named mixer layouts**: the zoo's `gdn_3to1` Qwen3-Next-style 3:1 sequence schedule among others, with a committed throughput benchmark (`artifacts/mixer-benchmark/`).

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

## Documentation

The standing documents live under `docs/` and have distinct jobs, and it is worth
knowing which one owns which question before reading any of them. The plan is
**prescriptive**, the status log is **append-only**, the report is **narrative**, and a
load-bearing number is owned by the artifact behind it rather than by any prose.

They are cited across the project by *name* rather than by path - a script docstring
says "pre-registered in `RESEARCH_REPORT.md` section 5.30", and the plan is cited only
as "plan section N" - so a reference tells you which document, and this table tells you
where it is.

| Document | What it is | Authoritative for |
|---|---|---|
| `docs/complete_implementation_plan.md` | The plan of record, sections 0-105: research questions, hardware strategy, the model and search-space design, gates, deliverables. Prescriptive, so its section 7 repository layout is the intended one rather than the built one. Sections are cited across the project as "plan section N". | What the project set out to do |
| `docs/RESEARCH_REPORT.md` | The narrative report of results. Section 3 is the factor map, section 5 the pre-registered studies (each carrying its *Pre-registration* beside its *Outcome*), section 9 the substrate audit, section 10 the synthesis. | What was found, and how it reads |
| `docs/IMPLEMENTATION_STATUS.md` | The implementation tracker. Append-only and newest-first; each entry records what was found, what was fixed, and what finding it cost. Superseded entries stay as written. | What was done, and when |
| `docs/T4_RUNBOOK.md` | The operational runbook for Kaggle sessions: pre-flight checks, the session plan, the exact commands. | How to run a session |
| `artifacts/ARCH_SEARCH_FINDINGS.md` | The architecture-search synthesis behind the section below: matched-budget identifiability, the crossed factorial, the confirmation arms. | The search evidence in detail |
| `docs/width-control-design.md` | A **design note, not a record**: what a width mutation costs on this model (100% of the parameters, all 254 tensors), the constraints the model's own declared-config rule imposes on it, the function-preservation contract it would have to meet, and the four outcomes declared before it runs. Nothing in it is a measurement. | What the next gate should be, and what would make it worth its hours |
| `docs/training-throughput.md` | A **measurement note, not a record**: where a leg's wall-clock actually goes, recomputed from the project's own FLOP estimators - that the step loop already runs at 38-45% of the card's fp32 peak, that the dense expert form executes 1.934x the arithmetic the routing contract books, that a completion without a KV cache forwards 299x the tokens it needs to, and that neither closes a token budget 2,170x below the comparator's. | Why a run takes the hours it takes, and which speedup is real |

Per-arm material lives beside each arm under `artifacts/*/`:

| File | What it is |
|---|---|
| `artifacts/*/NOTES.md` | The arm's runbook: its pre-registration and the command that reproduces its record. The runbook is *executed* by `scripts/check_record_freshness.py`, so one that no longer reproduces its record fails rather than being discovered. |
| `artifacts/*/decision.md` | The arm's **generated** decision record, written by `scripts/arch_zoo_analyze.py --write`. Regenerate it; do not hand-edit it. |
| `artifacts/*/analysis.md`, `artifacts/*/FINDINGS.md`, `artifacts/*/PERFORMANCE_NOTES.md`, `artifacts/*/SCREENING_AUDIT.md` | Per-arm and per-benchmark notes. Not machine-checked. An `analysis.md` whose conclusion a sibling audit refuted says so at the top. |

Where a number in the report or in this README is load-bearing, the ledger under
the arm owns it: `scripts/verify_record.py` recomputes each claim from the rows and
fails when the document and the artifacts disagree. The index above is itself
checked - see `tests/test_doc_index.py`.

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

That branch is also re-read on the other losses the same ledgers carry, and it is the
clearest case for doing so: held-out loss and final training loss both land on
*replicates*, and the trajectory average lands on *fails to replicate* — because
`|hidden| ≥ 1.5 × |depth|` narrows from 2.24× to 1.24× while both main effects keep
their signs at the design's floor and the interaction stays at its. The clause that
moves is a **magnitude ratio**, which is a statement about end-of-training sizes, so
the honest reading is that the ratio is loss-dependent rather than the attribution.
The fresh three-seed arm narrows its own ratio the same way (1.21×) and the same
check is run on the phase's other two pre-registered branches.

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

That pricing rests on the level split rather than on the pooled head contrast, and
deliberately: the pooled contrast clears the rule's own declared threshold by only
**1.15×** (p = 0.0115 against 0.01) and its significance moves with which loss is
read — the parent arm is non-significant held-out and significant on final training
loss, its confirmation the reverse — while the level split holds on **every** loss
reading, with all 12 wide-level units favouring more heads at the exact floor and the
narrow level non-significant throughout. Read on a trajectory-averaged loss the
six-seed branch itself changes — and the arm's own `decision.md` prints that reading
table beside the branch, so the reversal is stated where the label is quoted rather
than only in this record. The width clause is solid on all three readings
(−0.156 / −0.155 / −0.185, each at the design's floor), so the recommendation does not
depend on the reading; the head contrast's null does.

That has now been measured, and the depth finding survives it. The crossed factorial
re-ran on its **byte-identical** candidate set at 10× the training length — 12,000
steps, two seeds, fp16 with a same-precision 1200-step anchor, on Kaggle T4×2, all
pre-registered in `artifacts/arch-zoo-100m-v3-crossed-horizon10x/NOTES.md` — and its
rule lands on **shallow-wins persists**: depth **+0.0209 nats, exact p = 0.003086**
(floor 0.001543), with width −0.0645 at its own floor. So the lever prior work
predicted would reverse with training does not: at a matched parameter budget,
shallow still beats deep at the length a real budget occupies. Two qualifications
come out of the same run and both matter to a search — the decay is **slower** than
the 1.65×-per-doubling extracted over 400→1200 (1.38× here), so that rate
over-predicts how fast an effect fades; and the **winner changes** across 10×, where
the ranking holds at Spearman 0.90 while `W384` gives way to `W1536`. A screen
promotes; it does not choose.

That branch's robustness was checked at the horizon rather than inherited from 1,200.
Re-run on training loss both effects keep their signs and stay significant (depth
**+0.0249**, width **−0.0667** at the same floor), so 10× the training does not make the
attribution a holdout-split artifact; dropping any one of the 8 candidates still leaves
the spread significant (worst p = **0.0018**); and the crossed attribution is one the
design **cannot** leave-one-out — at 2 candidates per cell, removing one unbalances the
factorial, so it refuses on all eight instead of reporting a subset. Two seeds is
likewise forced rather than cautious: the re-split test is two-sided, so one seed's
**2/36 = 0.0556** floor sits above the 0.05 it is read against — **no single seed can
reach this branch at all**, which is why the stale single-seed file the landing left
behind had no verdict to state rather than merely declining to state one.

Its rule is **declared on the 12,000-step rows and refuses any other length**, so the
anchor cannot be read as the horizon result. That is not a hypothetical: the arm's
first attempt trained none of its horizon runs and was reported complete, and the
three defects that allowed it are closed — an admission bar that asked for twice the
schedule, a sweep that exits 0 after losing every run in a stage, and a driver that
read process exit codes as completeness rather than reading the ledgers. A fourth was
closed on the landing itself. The post-run watcher was killed inside the 4-second
analysis it launched, so the file left in place as the record was the *kernel-side*
single-seed artifact the driver had imported — one that reported "run at least two
seeds per architecture" about a complete two-seed arm — and nothing noticed, because
a stdout redirect to a file is block-buffered and the watcher's log stopped mid-line.
The analysis now runs unbuffered, and the watcher no longer reports on an exit status
alone: it **re-renders the record from the arm's own documented command and requires it
to come back identical**, which is the same check the test suite runs and which names
the cause when it fails. `check_record_freshness.py` itself now fails any arm whose
committed record was read over a seed set no documented command reads — the
content-based form, which survives a fresh clone where mtimes do not.

The other open question was v1's own, and it is now answered. That arm closed on "NOT
ESTABLISHED — this experiment cannot yet tell": a 3-block exact test whose floor is
1/36, run on finalists that were promoted *on those same seeds*, so it is mildly
anti-conservative in the winner's favour. Its own next step was to re-run those
finalists on seeds never used in screening, and
`artifacts/arch-zoo-100m-v1-finalists-6seed/NOTES.md` pre-registered exactly that —
the three finalists, byte-identical genomes, flat at 1200 steps on fresh seeds, read
*pooled* with v1 so the exact test moves to six blocks and a floor of 1/7776. Its
rule reads that test twice and gates the positive branch on **both**, because a
pooled test still contains the selection it is meant to check. The arm ran on the
idle local card in fp32 — Kaggle's ~3× advantage is an fp16 advantage, and fp16 is
ruled out precisely here, because the gaps to resolve (0.017–0.023 nats) are the size
of the fp16 differential this phase measured (≤0.018).

It landed on **order coupled**: the pooled test clears at p=0.001286 (floor 1/7776),
but the screening-free blocks alone neither clear the threshold (p=0.1111, floor
1/36) nor agree on a winner. The significance is therefore the blocks that selected
the finalists, so v1's *ordering* is not evidence to build on and **no ranking is
carried into the search** — the direction comes from the crossed factorial, which
never rested on one. The reading sits in v1's `decision.md` beside v1's own numbers,
with the headline naming the blocks that established it rather than the pooled label
alone.

The effect's *size* depends on the training horizon, and that was measured rather
than assumed: over 400→1200 the spread decays 1.65× per doubling of training,
replicating the earlier estimate on a controlled design — but seed noise decays
with it, so detectability is horizon-stable and the same architecture wins at 400,
800 and 1200 steps. The 10× horizon qualifies both halves of that: the decay is
*slower* than 1.65× extrapolates (1.38× per doubling over 1200→12,000), and the
winner does not come through the change even though the ranking survives it.

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
margin, and `--screen-noise-nats` overrides it. A stage at any other length borrows
the nearest measured length at or below it, and the record names which one it came
from: at the 12,000-step horizon this phase measured, that borrow is a 2.6×
over-estimate (0.0148 against the measured 0.0056), which is safe for the promotion
decision and would be misleading as a citation. `--plan screen` is that loop as a runnable plan — screen every candidate
at `--stage1-steps`, confirm the survivors at `--stage3-steps` — and
`--promote-ties` acts on the print: when the cut sits inside the noise it advances
the whole borderline set, including the candidate the cut landed on, instead of
letting an unresolvable margin decide which candidate is never trained again. What
that loop *costs* depends entirely on where it confirms, and §10 prices both ends:
**17.0 GPU-hours** with the survivors confirmed at 1,200 steps — the length this phase
measured at — against **26.4** with them confirmed at the 12,000-step horizon the
loop is specified for, on Kaggle, and **45.7** on the local card. Each horizon total
is the screening half plus the confirmation alone (13.80 + 12.57, and 13.80 +
31.92), because the longer confirmation replaces the shorter one rather than joining
it — the totals were quoted as 29.6 and ~49 until that composition was re-derived.
At a realistic horizon the confirmation becomes the loop's largest term on the local
card and a near-equal one remotely, which is why the horizon phase itself ran there:
the same 24 confirmations cost 2.54× less per run there.

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

That check recomputes each claim from the **ledgers**, which is not the same thing as
being able to re-render the record the claim is read from — an analyzer defect once
left one arm's committed `decision.md` impossible to regenerate, and nothing noticed.
So the records are checked the other way round too. Each arm's own `NOTES.md` runbook
is parsed and *executed* against its ledgers in a scratch tree, and the result is
diffed against what is committed, so an arm whose runbook does not reproduce its
record is a failure rather than a discovery:

```bash
PYTHONPATH=src python scripts/check_record_freshness.py --strict-rank
```

The two checks above run in the working tree, which is the one place where the run
output under `artifacts/` is visible — so a record or a test can pass here and fail
for a reader, and nothing inside the tree can tell a file that is *present* from a
file that is *committed*. `scripts/check_fresh_clone.py` asks the reader's question
instead: it clones `HEAD` and runs the checks in the clone, so a test that reads run
output the ignore rule keeps local fails here rather than being discovered there:

```bash
PYTHONPATH=src python scripts/check_fresh_clone.py
```

The scan prints the launch command for the arm it matched, with every flag that
decides the draw — the band included, because the driver's own default band draws a
different set from the same seed (1.14× the curated budget against the matched
0.9996×). Only then is the random arm trained on the architectures whose budget was
matched.

## A small language model on real text

The topology machinery is what the project is *about*; this is the leg that shows
the model and the training path it is attached to work on English. A
**20,863,872-parameter** MoE (`act.model.V0MoECausalLM`, 6 layers of 8 experts with
2 active, hidden 384, vocab 8192, context 512, **10,247,040 active parameters**)
trains on real TinyStories text:

```bash
PYTHONPATH=src python scripts/train_real_lm.py --max-steps 12000
PYTHONPATH=src python scripts/real_lm_infer.py --demo
```

**98,304,000 tokens** later the held-out loss is **1.6228 nats (perplexity 5.07)**
against a uniform floor of **9.0109 nats (8192)** — the model explains **82% of the
entropy** of next-token prediction on text it never trained on. A perplexity with
nothing beside it proves nothing, so the same held-out stream carries three
reference models: uniform **9.0109**, unigram **5.9681**, bigram **7.2348**. The
model beats the bigram by **5.61 nats** even though the bigram was *fit on those
held-out tokens* and the model never saw them.

The qualitative half is in `artifacts/real-lm/report.json` beside the numbers: four
generations from fixed prompts, which `real_lm_infer.py --demo` prints from the
checkpoint. Both halves are checked rather than trusted —
`tests/test_real_lm_proof.py` recomputes the arithmetic, the learning curve and
every reference above from the committed record and the committed held-out stream,
and `scripts/verify_record.py` carries the same numbers as claims, so a README that
drifts from the artifact fails.

What this does **not** claim: it is a proof of the path, not of the hypothesis —
20M parameters on a children's corpus says the model trains and generates, not that
learned topology beats a fixed architecture, which is the question the phase above
is about. And the checkpoint is a 250 MB run output and is not committed, so a
reader can recompute every number in the record and read the generations, but
cannot re-run inference without running the training command first.

## Condensing a large code model into the project's own student

The English leg above shows the model trains on real text. This one is aimed at
the actual hypothesis: whether a large model's behaviour can be **condensed** into
the project's sparse, learned-topology student, and measured against a popular
model of that student's size class.

| role | model |
|---|---|
| teacher | `deepseek-ai/deepseek-coder-1.3b-base` - 1.3B dense, 2T code tokens, public, native to `transformers` |
| student | `act.model.V0MoECausalLM` at `match`: **166,423,296 total / 81,488,640 active** (h768 d12, 8 experts with 2 active, expert width 768) |
| reference | `bigcode/tiny_starcoder_py`, **164M dense** - the same size class, trained on 100B tokens on 32 A100s |

The metric is **bits per byte on held-out Python**, and it is not a stylistic
choice: perplexity is a function of the tokenizer, so a 32k teacher, a 32k student
and a 49k baseline cannot be put in one table with it. Bits per byte divides by the
bytes of the held-out *text*, which every model tokenizes differently and none of
them gets to choose. HumanEval pass@1 is reported beside it, executed rather than
approximated.

HumanEval is run at the decoding settings the published figure was measured under -
**512 generated tokens, temperature 0.2, top_p 0.95**, the
`bigcode-evaluation-harness` defaults - because a protocol of our own that is stricter
(256 tokens, sampling the full 32k tail) would read as the baseline underperforming its
own model card. One difference cannot be flagged away: the published 7.84% is the
harness's many-sample unbiased `pass@1` estimate and this scores one sample per problem,
which is noise rather than bias - about ±1.7 points at that rate over 164 problems - so
the published number is a reference for the scale, not a target. The table refuses rows
decoded differently, the same way it refuses rows scored on different held-out text.

The student cannot be windowed the way the two `hf:` rows are - at a 512-token context, a
300-token prompt plus 512 generated tokens does not fit - so it keeps the prompt's tail and
generates into a window whose positions restart at the window start. That choice also rules
out a key/value cache: rolling the window re-numbers every token in it and changes the
cached tokens' *activations*, not just their rotation, so a cache is a different model
rather than a faster one. The measurement is in `tests/test_code_bench.py`.

```bash
PYTHONPATH=src python scripts/train_code_lm.py --fetch-teacher   # once; 2.7 GB
bash scripts/run_code_lm_campaign.sh                            # score, both arms, benchmarks
PYTHONPATH=src python scripts/code_bench.py --model hf:bigcode/tiny_starcoder_py
```

The campaign scores the teacher over a fixed grid of **non-overlapping** 512-token
windows, then trains two arms on those same windows in the same order from the same
seed - one distilled against the teacher's distribution, one plain cross-entropy. The
only difference between them is the objective, so the difference between them is
attributable to distillation rather than to data. That claim is checked against the arms'
own reports and not against the command line: `scripts/check_code_lm_arms.py` compares the
two and refuses on any difference that is not the objective - because the claim was already
false once on disk, when a checkpoint from a 20-step smoke run sat where the control arm's
checkpoint goes and adopting it would have started that arm from another run's weights at
step 21 while still reporting the full budget. Non-overlapping windows are load
bearing: they make the teacher's context at a position *identical* to the student's,
where a sliding window would give the teacher more context and flatter the student.

The teacher's distribution is cached as a top-32 plus a residual bucket, and the KL is
computed over the whole vocabulary rather than over the 32 stored tokens. Truncating
and renormalising would leave the student's own tail unconstrained, and nothing in a
loss curve would show it: `tests/test_train_code_lm.py` checks the bucket in both
directions - a student that ignores the teacher's tail is charged for it, and one that
matches it scores exactly zero.

The two student arms share one tokenizer and are scored on the same 512-token windows
of the same held-out text, which makes their difference a *paired* measurement rather
than two independent ones: window 7 of one arm covers the same Python as window 7 of
the other, so the difficulty of the text cancels and only the objective is left. Every
row therefore records its per-window losses beside its total, and
`scripts/code_lm_table.py` runs a sign-flip permutation test over the paired
differences and refuses to render a table without one. The mean difference and its 95%
interval are the result; the p-value only says whether it needs a caveat, and it is
reported as a lower bound rather than as proof, because windows cut from one file are
not independent and nothing here can correct for that.

Measured on the development GTX 1070 (8 GiB): the teacher loads in 2.8 s and scores
**1,050 tokens/s**; the student trains at **~2,950 tokens/s**. That sets the campaign
budget at 15.36M teacher-labelled tokens (4.05 h) and 11,250 steps per arm (46.08M
student tokens, three passes, 4.34 h each).

What this does **not** claim, and the reason is arithmetic rather than modesty: the
baseline saw **100B tokens on 32 A100s in 18 hours**. One 1070 and three orders of
magnitude fewer tokens cannot close that gap, and a run that appeared to land near the
teacher's published 0.18 MultiPL HumanEval pass@1 would be evidence of a bug, not of
distillation working. The claim under test is narrower and testable - whether the
sparse architecture is a viable *container* for a large model's behaviour - and the
headline table is written from `artifacts/code-lm/bench/` when the campaign completes.

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
