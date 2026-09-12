# Adaptive Computational Topology
## Complete Research & Implementation Plan
### Kaggle-Free Compute Edition — September 2026

---

# 0. Executive Summary

## Objective

Build an experimental language-model training system in which **computational topology is learned during pretraining**, rather than being completely specified in advance.

The system should be able to discover:

- which layers are needed;
- which layer types are useful;
- how wide layers should be;
- how many experts are required;
- which experts specialize;
- which experts are redundant;
- which experts should merge or split;
- which attention heads/dimensions matter;
- how many experts each token should invoke;
- how much depth each token requires;
- which residual pathways are useful.

The central hypothesis is:

> **A training process that jointly optimizes weights and computational topology can discover architectures with a better capability/FLOP frontier than manually designed architectures of equivalent capacity.**

The project is explicitly **not** initially trying to train a frontier-scale model.

The first objective is to demonstrate the mechanism.

---

# 1. Research Questions

The project should answer these questions in order.

## RQ1 — Is topology evolution actually useful?

Does dynamically changing architecture during training beat:

- a fixed architecture;
- random topology changes;
- conventional pruning?

## RQ2 — Can useful topology be detected before convergence?

Does structural importance stabilize sufficiently early to guide topology decisions?

## RQ3 — Which importance signals work?

Compare:

- magnitude;
- activation frequency;
- router probability;
- gradient;
- Fisher;
- temporary causal ablation;
- functional redundancy;
- temporal stability.

## RQ4 — Does causal structural selection matter?

Does importance-guided evolution beat identical random topology churn?

## RQ5 — Are discovered structures reusable?

Can the final architecture be freshly initialized and independently trained?

## RQ6 — Is initialization or topology the important artifact?

Compare:

- evolved weights;
- rewound initialization;
- fresh random initialization;
- teacher initialization.

## RQ7 — Does architecture scale?

Does an architecture discovered at 300M transfer effectively to 500M/1B/3B?

## RQ8 — Can the system discover architectures competitive with manually designed Qwen-like hybrids?

Qwen3-Next is an important reference point because its released architecture already combines Gated DeltaNet, sparse MoE, shared experts and periodic full attention: 80B total, ~3B activated, 512 routed experts, 10 routed + 1 shared expert, and 48 layers with a 3:1 DeltaNet/full-attention pattern.

---

# 2. Research Scope

The project consists of four layers.

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
│ PyTorch + T4×2 / JAX + TPU v5e              │
└──────────────────────────────────────────────┘
```

Keep these layers independent.

The topology controller must not know how a GPU or TPU performs matrix multiplication.

---

# 3. Hardware Strategy

## Primary backend: Kaggle T4×2

Use T4×2 for the main research implementation.

Current Kaggle documentation identifies T4×2 as two NVIDIA T4 GPUs with 16 GB each and 29 GB host RAM; GPU notebook sessions can run up to 12 hours.

Use:

```text
PyTorch
torch.distributed
DDP
optional FSDP later
bf16/fp16
gradient accumulation
activation checkpointing
```

The initial system should fit completely within:

```text
GPU 0: ~16 GB
GPU 1: ~16 GB
Host:  ~29 GB
```

Do not make the architecture dependent on >2-way GPU parallelism.

---

# 4. Secondary Backend: TPU v5e-8

TPU should not be the first implementation.

Kaggle's v5e announcement specifies eight TPU chips with 16 GB HBM each. Kaggle's TPU documentation currently states 9-hour session limits and 20 TPU-hours/week.

TPU is attractive for:

- high-throughput dense pretraining;
- large batches;
- static architectures;
- candidate-model evaluation;
- final retraining.

TPU is less attractive for:

- dynamic graph mutation;
- arbitrary expert birth/death;
- frequent structural changes;
- Python-driven per-component experimentation.

Therefore:

### Phase A

Implement everything in PyTorch/T4.

### Phase B

Port the **stable final architecture** to JAX.

### Phase C

Use TPU for higher-throughput independent training of finalist architectures.

Do not build the topology controller twice initially.

---

# 5. Kaggle Operating Model

Kaggle sessions are ephemeral.

Current Kaggle documentation gives 20 GB auto-saved `/kaggle/working` storage and additional scratch space that does not survive the session. Notebook outputs can be turned into versioned datasets, making them suitable for checkpoint pipelines.

Therefore the system must assume:

```text
session starts
↓
download current experiment state
↓
run ≤ session budget
↓
checkpoint
↓
publish artifact
↓
session ends
```

Never depend on:

```text
global local filesystem
```

for state.

---

# 6. Kaggle Artifact Layout

Create separate versioned Kaggle datasets:

```text
act-code
act-tokenized-data
act-checkpoints
act-experiments
act-evaluations
act-final-models
```

Recommended logical structure:

```text
act-checkpoints/
    exp_000/
    exp_001/
    exp_002/
    ...

act-experiments/
    configs/
    metrics/
    architecture_genomes/
    mutation_logs/
    topology_snapshots/

act-evaluations/
    lm/
    capability/
    ablations/
    architecture_comparisons/
```

Do not put large temporary optimizer states into the primary research results dataset unless needed for reproducibility.

---

# 7. Repository Structure

The actual source should be a normal Git repository.

```text
adaptive-computational-topology/
│
├── pyproject.toml
├── README.md
├── LICENSE
│
├── configs/
│   ├── baseline_dense.yaml
│   ├── baseline_moe.yaml
│   ├── random_churn.yaml
│   ├── adaptive_v0.yaml
│   ├── adaptive_v1.yaml
│   └── qwen_prior.yaml
│
├── src/
│   └── act/
│       ├── model/
│       │   ├── transformer.py
│       │   ├── attention.py
│       │   ├── gdn.py
│       │   ├── moe.py
│       │   ├── expert.py
│       │   ├── residual.py
│       │   ├── depth.py
│       │   └── embedding.py
│       │
│       ├── topology/
│       │   ├── graph.py
│       │   ├── genome.py
│       │   ├── mutation.py
│       │   ├── grow.py
│       │   ├── prune.py
│       │   ├── merge.py
│       │   ├── split.py
│       │   └── rewire.py
│       │
│       ├── importance/
│       │   ├── activation.py
│       │   ├── gradient.py
│       │   ├── fisher.py
│       │   ├── ablation.py
│       │   ├── functional.py
│       │   └── stability.py
│       │
│       ├── evolution/
│       │   ├── controller.py
│       │   ├── population.py
│       │   ├── selection.py
│       │   ├── mutation.py
│       │   └── pareto.py
│       │
│       ├── training/
│       │   ├── loop.py
│       │   ├── checkpoint.py
│       │   ├── distributed.py
│       │   ├── scheduler.py
│       │   └── budget.py
│       │
│       ├── evaluation/
│       │   ├── perplexity.py
│       │   ├── ablation.py
│       │   ├── probes.py
│       │   ├── capability.py
│       │   └── efficiency.py
│       │
│       └── data/
│           ├── loader.py
│           ├── shards.py
│           └── tokenizer.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── evolve.py
│   ├── ablate.py
│   ├── extract.py
│   └── publish_kaggle.py
│
├── notebooks/
│   ├── 00_environment.ipynb
│   ├── 01_data_prepare.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_random_churn.ipynb
│   ├── 04_importance.ipynb
│   ├── 05_adaptive.ipynb
│   ├── 06_search.ipynb
│   ├── 07_extract.ipynb
│   ├── 08_retrain.ipynb
│   └── 09_compare.ipynb
│
└── tests/
    ├── test_model.py
    ├── test_topology.py
    ├── test_mutations.py
    ├── test_importance.py
    ├── test_checkpoint.py
    └── test_determinism.py
```

---

# 8. Model V0

Do not begin with 1B+.

Target:

## 256M–350M parameters

Possible configuration:

```text
24 layers
hidden size:       1024
attention heads:   16
FFN:               ~4096
context:           2048 initially
vocab:             32K–64K
```

Do not copy Qwen's 151k vocabulary at this scale.

A giant vocabulary would consume a disproportionate fraction of the model.

Use a 32K or 64K tokenizer for V0.

---

# 9. Model V0 Search Space

The model must support multiple computational variants.

Every layer can potentially select:

```text
ATTENTION
GDN
HYBRID
```

and FFN can be:

```text
DENSE
MOE
```

Initial expert pool:

```text
8–16 experts/layer
```

Active:

```text
1–4 experts/token
```

The number of experts should be configurable without rewriting the model.

---

# 10. V0 Architecture Genome

```python
ArchitectureGenome(
    depth=24,
    hidden_size=1024,

    layers=[
        LayerGene(
            module="attention",
            heads=16,
            kv_heads=4,
            ffn_width=4096,
            experts=0,
            active_experts=0,
            residual_paths=1,
        ),
        ...
    ],

    global={
        "max_compute": ...,
        "early_exit": ...,
    }
)
```

Every architecture must serialize completely into JSON.

Example:

```json
{
  "version": 1,
  "seed": 42,
  "depth": 24,
  "hidden_size": 1024,
  "layers": [
    {
      "type": "gdn",
      "experts": 16,
      "active_experts": 2,
      "ffn_width": 2048
    }
  ]
}
```

The genome must be sufficient to reconstruct the model.

---

# 11. Weight / Topology Separation

The architecture engine must maintain:

```text
ArchitectureGenome
+
Weights
+
OptimizerState
+
TrainingState
```

separately.

This is essential for the Lottery Ticket experiments.

We need to be able to do:

```text
final topology
+
original initialization
```

without ambiguity.

At initialization:

```text
initial_state.pt
```

is generated once.

For small experiments, retain it.

For larger experiments, only retain the parameters needed for candidate rewinding.

---

# 12. Training Loop

The primary training loop:

```text
for step in range(max_steps):

    batch = next(data)

    output = model(batch)

    loss = language_model_loss(output)

    loss.backward()

    optimizer.step()
    scheduler.step()

    update_statistics()

    if topology_evaluation_due:
        evaluate_structure()

    if topology_update_due:
        controller.step()

    if checkpoint_due:
        save_checkpoint()
```

The structural controller must run **outside** the gradient graph.

No DARTS-style architecture gradients in V0.

---

# 13. Why Discrete Search First

DARTS-style continuous architecture relaxation introduces well-known instability and collapse problems.

V0 therefore uses:

```text
discrete topology
+
slow structural updates
```

rather than:

```text
soft architecture probabilities
```

Evolutionary operations happen only at scheduled structural checkpoints.

---

# 14. Structural Clock

Do not mutate topology every training step.

Define:

```text
weight_step       = every step
statistics_step   = every 100 steps
probe_step        = every 1,000–5,000 steps
mutation_step     = every 5,000–20,000 steps
architecture_eval = every 25,000–100,000 steps
```

Exact intervals should be configurable.

The model needs time to adapt to each structural change.

---

# 15. Importance System

Implement the importance system in layers.

## Level 1 — Cheap

Every statistics interval:

```text
activation_count
routing_count
gradient_norm
parameter_norm
Fisher approximation
loss contribution
```

## Level 2 — Moderate

Periodically:

```text
functional similarity
routing co-occurrence
representation clustering
```

## Level 3 — Expensive

Only for uncertain candidates:

```text
temporary causal ablation
```

This prevents the system from executing thousands of unnecessary ablations.

---

# 16. Importance Record

Each component gets:

```python
ImportanceRecord(
    component_id,
    activation,
    routing,
    gradient,
    fisher,
    loss_delta,
    capability_delta,
    uniqueness,
    stability,
    uncertainty,
    age,
)
```

Persist this in compact form.

Do not save huge activation tensors.

Use running statistics.

---

# 17. Temporal Stability

For each component maintain an exponentially weighted history.

Example:

```python
stability = correlation(
    importance_history[-N:],
    previous_window
)
```

Track:

```text
rank stability
score stability
routing stability
functional stability
```

A component should not be permanently killed simply because it has one bad evaluation.

---

# 18. Component Lifecycle

```text
NEW
 ↓
EMERGING
 ↓
ACTIVE
 ↓
STABLE
 ↓
ESSENTIAL
```

Alternative:

```text
ACTIVE
 ↓
UNCERTAIN
 ↓
REDUNDANT
 ↓
DYING
 ↓
REMOVED
```

A component can recover.

Therefore "DYING" is not the same as deleted.

---

# 19. Random Topology Control — Experiment 0

This is mandatory.

Construct:

```text
CONTROL_RANDOM
```

that uses the exact same:

```text
mutation frequency
number of mutations
compute budget
component types
population size
training duration
```

as the adaptive model.

But chooses components randomly.

Compare:

```text
ADAPTIVE
RANDOM
```

under identical conditions.

This isolates:

```text
useful architecture discovery
```

from:

```text
topology churn as regularization
```

This experiment must pass before extensive causal-importance engineering.

---

# 20. Evolution Operators

## Prune

Remove:

```text
layer
head
expert
FFN group
residual path
```

according to confidence.

## Grow

Add:

```text
expert
FFN group
head
residual path
```

where the current capacity is demonstrably limiting performance.

## Clone

Copy an expert's parameters, then perturb slightly.

Use when:

```text
high demand
+
high specialization
+
routing saturation
```

## Split

Split a high-load expert into two.

Initial parameters can be:

```text
W_A = W + ε
W_B = W - ε
```

or partition dimensions.

## Merge

Merge functionally redundant experts.

Initially use:

```text
weighted parameter interpolation
```

then optionally implement output-space merging.

## Rewire

Change permissible routing or residual connections.

---

# 21. Capacity Budget

The optimizer must operate under an explicit compute budget.

For example:

```text
maximum parameters = 350M
maximum active parameters = 150M
maximum FLOPs/token = X
```

This prevents the evolutionary system from solving every problem by simply growing more compute.

A second experimental mode can allow variable budget.

---

# 22. Multi-Objective Evolution

Do not optimize one scalar too early.

Every architecture receives:

```text
quality
training_flops
inference_flops
parameters
active_parameters
memory
latency
```

Maintain a Pareto frontier.

Example:

```text
Architecture A
quality=0.81
FLOPs=1.0

Architecture B
quality=0.83
FLOPs=1.1

Architecture C
quality=0.84
FLOPs=1.8
```

None should automatically dominate solely by validation loss.

---

# 23. Expert Ecology

Each expert receives metadata:

```text
expert_id
birth_step
last_used_step
activation_rate
domain_distribution
importance
uniqueness
stability
parent
children
```

This permits studying:

```text
birth
specialization
competition
merging
death
```

The goal is not to anthropomorphize the model.

The metadata exists to test whether useful specialization actually emerges.

---

# 24. Expert Similarity

Periodically sample representative hidden states.

Run experts over the same input batch.

Store compressed representations rather than full tensors:

```text
mean output
variance
random projection
PCA sketch
```

Then calculate:

```text
cosine similarity
CKA-like similarity
correlation
```

Avoid storing entire expert activation datasets.

---

# 25. Expert Relationship Graph

Construct:

```text
G_expert = (V,E)
```

Nodes:

```text
experts
```

Edges:

```text
functional similarity
routing co-occurrence
synergy
interference
```

This graph becomes a source for:

```text
merge candidates
bridge detection
specialist detection
```

---

# 26. Temporary Causal Ablation

Never immediately destroy uncertain components.

Procedure:

```text
baseline batch/evaluation
        ↓
disable component
        ↓
run same batch/evaluation
        ↓
restore component
        ↓
compute delta
```

For the prototype:

```text
layer
expert
head
FFN group
```

Only.

Do not perform dimension-level ablations until the coarse-level system works.

---

# 27. Capability Probe Suite

Use a tiny, cheap probe battery during training.

The probe set should include:

```text
language
arithmetic
logic
coding
world knowledge
long-range dependency
cross-domain reasoning
```

These are not final benchmark scores.

They are **structural diagnostic tests**.

The full benchmark suite is run only for finalists.

---

# 28. Benchmark Separation

Separate:

### Training probes

Used by the topology controller.

### Development benchmarks

Used during research.

### Held-out benchmarks

Never exposed to the evolutionary controller.

This prevents architecture evolution from overfitting the benchmark suite.

---

# 29. Data

Use a two-tier dataset.

## Tier 1 — Architecture research

A high-quality, relatively small corpus sufficient to compare training dynamics.

Target:

```text
2–10B tokens
```

distributed over many Kaggle sessions if necessary.

## Tier 2 — Final candidate

Larger mixture:

```text
web
books
code
math
science
reference
```

Target:

```text
10–50B+ tokens
```

only for the strongest candidates.

Do not pretrain every experiment on massive data.

That would destroy the value of evolutionary search.

---

# 30. Tokenizer

Train a project-specific tokenizer.

Recommended:

```text
32,768 vocab
```

possibly:

```text
65,536
```

for the larger experiment.

Keep tokenizer fixed across all candidate models.

Tokenizer changes invalidate comparisons.

---

# 31. Dataset Sharding

Pre-tokenize into fixed-size shards:

```text
shard_00000.bin
shard_00001.bin
...
```

Use:

```text
uint16
```

where possible.

For vocab >65,535:

```text
uint32
```

or packed formats.

Each shard should contain:

```text
tokens
document boundaries
metadata
checksum
```

Create Kaggle Dataset versions from prepared output files. Kaggle explicitly supports turning Notebook output files into versioned datasets for reproducible pipelines.

---

# 32. Data Ordering

Use deterministic shard schedules.

Record:

```text
dataset_version
shuffle_seed
shard_order
epoch
offset
```

in the checkpoint.

This is essential for comparable multi-session experiments.

---

# 33. Optimizer

V0:

```text
AdamW
bf16/fp16
cosine decay
warmup
gradient clipping
```

Do not experiment with optimizer architecture until the topology mechanism works.

Qwen3-Next's optimizer choices are interesting later, but copying every architectural detail simultaneously would confound the experiment.

---

# 34. Mixed Precision

T4:

```text
fp16/bf16 depending on operator support
```

Keep master optimizer weights in fp32 where required.

TPU:

```text
bfloat16
```

Use bfloat16 naturally on TPU.

---

# 35. Memory Strategy for T4×2

Implement:

```text
gradient accumulation
activation checkpointing
DDP
optional FSDP
```

Do not rely on huge sequence lengths initially.

Start:

```text
context = 1024–2048
```

Move to:

```text
4096
8192
```

only after architecture mechanisms stabilize.

---

# 36. Model Parallelism

Avoid complex tensor parallelism in V0.

Use:

```text
DDP
```

with each GPU holding a full model replica.

For a 300M model this is practical.

The two GPUs primarily increase:

```text
effective batch
```

and training throughput.

Later, if the model exceeds one GPU:

```text
FSDP
```

can be introduced.

---

# 37. Checkpoint Format

Every checkpoint must contain:

```text
model_state
optimizer_state
scheduler_state
step
tokens_seen
rng_state
dataset_position
architecture_genome
topology_history
importance_state
evolution_state
experiment_config
git_commit
```

Use:

```text
checkpoint_latest
checkpoint_periodic
checkpoint_best
```

---

# 38. Checkpoint Resume

A Kaggle session restart must be equivalent to an uninterrupted training run.

Test:

```text
run 50k steps
save
reload
run 10k
```

against:

```text
run 60k continuously
```

within expected numerical tolerance.

This becomes a unit/integration test.

---

# 39. Kaggle Notebook Architecture

Create one notebook per role.

## `00_environment`

Verify:

```text
CUDA
GPU count
RAM
PyTorch
NCCL
disk
```

or:

```text
TPU mesh
JAX
device count
HBM
```

## `01_data_prepare`

Create/download tokenizer and shards.

## `02_baselines`

Train dense baseline.

## `03_random_churn`

Experiment 0.

## `04_measurement`

Importance-system validation.

## `05_adaptive`

Actual topology-evolving model.

## `06_population_search`

Generate/evaluate candidate genomes.

## `07_extract`

Freeze and extract architectures.

## `08_retrain`

Fresh-initialization training.

## `09_compare`

Final scientific analysis.

---

# 40. Kaggle Session Protocol

Every notebook begins by:

```text
1. determine experiment ID
2. load configuration
3. locate latest checkpoint dataset
4. validate checksum
5. restore state
6. continue
```

At end:

```text
1. save checkpoint
2. save metrics
3. save topology genome
4. save architecture history
5. save summary JSON
6. publish output
```

A session should never contain irreplaceable state.

---

# 41. Experiment Registry

Create:

```text
experiments.jsonl
```

Each record:

```json
{
  "experiment_id": "exp_0042",
  "parent": "exp_0039",
  "seed": 42,
  "architecture_hash": "...",
  "config_hash": "...",
  "git_commit": "...",
  "dataset_version": "...",
  "hardware": "T4x2",
  "tokens": 1200000000,
  "training_flops": 123456789,
  "search_flops": 123456,
  "result": {
    "val_loss": 2.31,
    "capability": 0.61
  }
}
```

This becomes the project's scientific ledger.

---

# 42. Reproducibility Hashes

Hash:

```text
architecture genome
training configuration
dataset manifest
tokenizer
code revision
random seed
```

Create:

```text
experiment_hash
```

No result is considered valid unless all six are recorded.

---

# 43. Experiment 0

## Random Topology Churn

Train:

```text
A — fixed baseline
B — random topology churn
C — adaptive topology
```

Keep identical:

```text
tokens
model budget
mutation count
mutation frequency
compute budget
seed distribution
```

Success:

```text
C > B
```

Failure:

```text
C ≈ B
```

If failure persists over multiple seeds:

**simplify the architecture controller.**

---

# 44. Experiment 1

## Conventional Architecture Baselines

Train:

```text
Dense Transformer
Static MoE
Static GDN/Attention hybrid
```

These establish the manually designed frontier.

---

# 45. Experiment 2

## Dynamic Sparse Training

Allow:

```text
prune
grow
```

but do not allow module-type changes.

Purpose:

Determine whether topology evolution at all is useful before increasing complexity.

---

# 46. Experiment 3

## Expert Ecology

Add:

```text
clone
split
merge
birth
death
```

Only.

Compare against Experiment 2.

---

# 47. Experiment 4

## Module-Type Evolution

Allow:

```text
attention
GDN
hybrid
```

to change.

This is the first experiment testing the deepest part of the hypothesis:

> Does the training process discover the computational primitive itself?

---

# 48. Experiment 5

## Adaptive Depth

Introduce token-level compute routing.

Mixture-of-Depths demonstrates that learned token-level compute allocation is already practical, so this should be integrated as an existing mechanism rather than reinvented.

Possible choices:

```text
2 layers
4 layers
8 layers
12 layers
full depth
```

Initially make this a separate branch of research.

Do not introduce it before the topology search is working.

---

# 49. Experiment 6

## Hierarchical Search

After the simpler experiments succeed, allow:

```text
layer
→ module
→ expert
→ group
→ dimension
```

to evolve.

This is expensive.

Use only on finalists.

---

# 50. Experiment 7

## Qwen-Informed Search

Extract structural priors from Qwen3-Next:

```text
3:1 GDN:attention schedule
512-ish expert population
shared expert
high expert sparsity
```

Then compare:

```text
uninformed search
vs
Qwen-seeded search
```

Important:

**Do not initialize the model with Qwen weights in this experiment.**

Only transfer architecture priors.

Qwen3-Next's released model card explicitly documents these architecture components, making it a reproducible reference rather than an informal inspiration.

---

# 51. Experiment 8

## Teacher Initialization

Now separately test:

```text
Qwen-derived initialization
```

against:

```text
random
rewound
```

This isolates transfer effects.

---

# 52. Experiment 9

## Architecture Reusability

Take an architecture discovered on:

```text
300M
```

Construct:

```text
300M
600M
1B
```

variants.

Train fresh.

Test whether performance remains superior.

This is the first test of the proposed **computational genome** concept.

---

# 53. Experiment 10

## Cross-Seed Stability

Run at least:

```text
seed 1
seed 2
seed 3
```

For the winning architecture.

A discovery that works only once is not a discovery.

Measure:

```text
architecture similarity
layer-type distribution
expert population
routing structure
capability
```

across seeds.

---

# 54. Experiment 11

## Architecture Equivalence

Train multiple independent populations.

Ask:

> Do different runs converge toward similar computational structures?

Possible outcomes:

```text
A ≈ B ≈ C
```

suggesting an architecture attractor.

Or:

```text
A != B != C
```

while capability is equal.

That would suggest multiple architectures occupy a common capability-equivalence class.

Both results are scientifically interesting.

---

# 55. Experiment 12

## Fresh Initialization Test

For the final discovered topology:

```text
Model A:
evolved weights

Model B:
rewound initial weights

Model C:
fresh random initialization

Model D:
teacher initialization
```

Train equally.

This should be one of the project's headline experiments.

---

# 56. Experiment 13

## Compute-Efficiency Scaling

Plot:

```text
quality
   │
   │                  ●
   │             ●
   │       ●
   │   ●
   └──────────────────────
          training FLOPs
```

For:

```text
Dense
MoE
Whittle
Adaptive
Random churn
```

Include:

```text
search FLOPs
```

in the Adaptive total.

---

# 57. Metrics

Every experiment must report:

## Model

```text
parameters
active parameters
layers
experts
average experts/token
FLOPs/token
```

## Training

```text
tokens
steps
wall time
GPU-hours
training FLOPs
search FLOPs
evaluation FLOPs
total FLOPs
```

## Quality

```text
validation loss
perplexity
capability probes
held-out benchmarks
```

## Topology

```text
prune count
grow count
merge count
split count
rewire count
mask stability
expert specialization
expert redundancy
architecture entropy
```

---

# 58. Important Derived Metric

Define:

## Capability per Compute

\[
CEC =
\frac{Capability}
{TrainingFLOPs + InferenceFLOPs}
\]

Also:

## Discovery-adjusted capability

\[
DAC =
\frac{Capability}
{DiscoveryFLOPs+TrainingFLOPs}
\]

The second is crucial.

A method that spends 10× more compute discovering a 5% better architecture is not necessarily useful.

---

# 59. Architecture Stability Metrics

For architecture snapshots \(G_t\):

\[
D_t = distance(G_t,G_{t-1})
\]

Track:

```text
layer-type edit distance
expert edit distance
head edit distance
routing distance
```

Define convergence when:

\[
D_t < \epsilon
\]

for several successive structural intervals.

---

# 60. Topology Entropy

For a population of architectures:

\[
H(G)=-\sum p(g)\log p(g)
\]

Monitor whether:

```text
high entropy → exploration
low entropy  → convergence
```

This gives an objective signal for changing from:

```text
exploration
```

to:

```text
exploitation
```

---

# 61. Evolution Controller

Implement V0 as a rule-based controller.

Do not use another LLM.

Pseudo-policy:

```python
if component.confidence_high and component.importance_low:
    prune(component)

elif component.high_load and component.confidence_high:
    grow(component)

elif component.high_similarity and both_are_redundant:
    merge(component_a, component_b)

elif component.high_load and multimodal:
    split(component)

elif layer.module_is_consistently_underperforming:
    mutate_module_type(layer)
```

Only after this works should we consider learning the controller itself.

---

# 62. Search Controller State

```python
ControllerState(
    step,
    population,
    architecture_frontier,
    mutation_history,
    importance_calibration,
    budget_remaining,
    exploration_temperature,
)
```

The controller itself must be checkpointable.

---

# 63. Exploration vs Exploitation

Early training:

```text
more exploration
```

Late training:

```text
more exploitation
```

For example:

```text
0–20% training:
high mutation

20–60%:
moderate mutation

60–85%:
low mutation

85–100%:
freeze/evaluate
```

But let structural stability override the fixed schedule.

---

# 64. Protect Against Catastrophic Topology Changes

Every structural modification should have:

```text
pre-change checkpoint
```

and a rollback mechanism.

If:

```text
loss spike > threshold
gradient instability
NaN
capability collapse
```

automatically:

```text
rollback
mark mutation harmful
reduce mutation magnitude
```

---

# 65. Safe Mutation Protocol

A topology change occurs as:

```text
snapshot
 ↓
mutate
 ↓
short adaptation window
 ↓
monitor loss
 ↓
accept / rollback
```

Do not evaluate a topology immediately after changing it and conclude that it is bad.

Allow the weights to adapt.

This is essential.

---

# 66. Prevent Architecture Overfitting

Never let the controller see the final benchmark suite.

Use:

```text
TRAINING SIGNALS
↓
architecture development probes
↓
candidate selection
↓
held-out evaluation
```

The benchmark suite should remain frozen.

---

# 67. Implementation Order

Do not implement the complete system at once.

## Milestone 1

Dense 250M LM trains and resumes correctly.

## Milestone 2

Static MoE trains correctly.

## Milestone 3

Topology graph serializes/deserializes.

## Milestone 4

Prune/grow works without changing model semantics unexpectedly.

## Milestone 5

Random topology churn works.

## Milestone 6

Importance instrumentation works.

## Milestone 7

Adaptive prune/grow beats random churn.

## Milestone 8

Merge/split works.

## Milestone 9

Module-type evolution works.

## Milestone 10

Fresh initialization experiment works.

## Milestone 11

Qwen-informed search works.

## Milestone 12

Scale to 500M–1B.

---

# 68. Unit Testing Requirements

Every topology operator must have tests.

### Prune

```text
output shape valid
parameter count decreases
serialization valid
```

### Grow

```text
parameter count increases as expected
existing weights preserved
new parameters initialized correctly
```

### Merge

```text
expert count decreases
routing indices repaired
forward pass valid
```

### Split

```text
expert count increases
parameter initialization deterministic
routing remains valid
```

### Rewire

```text
graph remains acyclic where required
forward pass valid
```

---

# 69. Numerical Tests

Every architecture mutation must pass:

```text
finite loss
finite gradients
no NaN
no Inf
stable optimizer
```

Run a tiny deterministic model in CI.

---

# 70. Scientific Test Harness

Create:

```text
pytest tests/scientific/
```

Tests:

```text
test_random_churn_has_same_budget()
test_adaptive_and_random_have_same_mutation_count()
test_search_cost_is_counted()
test_architecture_hash_reproducible()
test_resume_matches_continuous_training()
test_fresh_init_reproducible()
```

---

# 71. TPU Port

Once the architecture freezes:

```text
PyTorch checkpoint
      ↓
architecture genome
      ↓
JAX model constructor
      ↓
parameter conversion
      ↓
TPU training
```

Do not port dynamic mutations immediately.

The TPU experiment is:

> Does the discovered architecture remain advantageous when trained efficiently at larger scale?

This makes TPU a validation backend instead of making it a development dependency.

---

# 72. Optional TPU Evolution

Only after the project succeeds on GPU should we investigate JAX-native dynamic topology.

For TPU:

```text
host controller
      ↓
compile candidate
      ↓
train candidate
      ↓
return metrics
      ↓
controller mutates
```

The expensive part is recompilation.

Therefore structural changes should be much less frequent on TPU.

---

# 73. Low-Fidelity Search

Never give every architecture a full training run.

Use:

```text
Stage 1:
10M tokens

Stage 2:
50M tokens

Stage 3:
250M tokens

Stage 4:
1B+ tokens
```

Only architectures surviving each stage advance.

This is critical under Kaggle compute constraints.

---

# 74. Population Size

V0:

```text
8–16 candidates
```

V1:

```text
16–32
```

Do not start with hundreds.

The cost of evaluating candidates will dominate.

---

# 75. Candidate Selection

Use Pareto selection.

Keep candidates that are:

```text
high quality
low compute
structurally diverse
```

Avoid filling the population with tiny variations of one architecture.

---

# 76. Diversity Preservation

Two architectures with similar quality but dramatically different structure should both survive.

This prevents premature convergence.

Measure:

```text
genome distance
```

and maintain diversity.

---

# 77. Computational Genome

The final output of the discovery run should not merely be a checkpoint.

Produce:

```text
genome.json
```

containing:

```text
layer layout
module types
widths
heads
expert counts
expert grouping
routing
depth policy
residual topology
initialization metadata
scaling rules
```

This is the core research artifact.

---

# 78. Genome Scaling

Define transformations:

```text
scale_depth(genome, factor)
scale_width(genome, factor)
scale_experts(genome, factor)
```

Examples:

```text
300M genome
      ↓
600M
      ↓
1B
      ↓
3B
```

Do not assume every parameter scales linearly.

Preserve relative structural ratios where possible.

---

# 79. Final Architecture Comparison

The final paper-quality comparison should be:

```text
Dense Transformer
Static MoE
Qwen-inspired fixed hybrid
Whittle-derived subnet
Random topology evolution
Adaptive topology
```

At matched:

```text
parameters
active parameters
training FLOPs
tokens
context
```

with:

```text
3 seeds
```

for important comparisons.

---

# 80. Target Results

## Minimum success

Adaptive > Random topology churn.

## Strong success

Adaptive > Fixed MoE at equal total training FLOPs.

## Major success

Freshly initialized adaptive architecture > manually designed architecture.

## Exceptional result

Architecture transfers across model scales.

## Breakthrough result

Architecture discovered at small scale consistently produces a superior compute/capability frontier at substantially larger scales.

---

# 81. What We Are NOT Promising

This project should not claim:

> "We will train a Chinchilla-optimal frontier LLM for free."

Kaggle is not sufficient for that goal.

Instead:

> **Use free Kaggle to establish whether architecture discovery works at all.**

If the mechanism succeeds, the architecture can subsequently be scaled using rented, institutional, or personal compute.

The valuable artifact is the **architecture-discovery algorithm**, not a single free-trained checkpoint.

---

# 82. Kaggle Compute Allocation

Prioritize the free budget in this order:

```text
1. Experiment 0
2. Dense/MoE baselines
3. Adaptive prune/grow
4. Importance validation
5. Expert ecology
6. Module evolution
7. Fresh initialization
8. Qwen-informed search
9. 500M/1B finalists
10. TPU scaling
```

Do not reverse this order.

---

# 83. Expected Compute Discipline

Every experiment must have:

```text
maximum wall time
maximum tokens
maximum structural mutations
maximum candidate count
maximum evaluation budget
```

The controller should terminate itself before the Kaggle session limit.

For example:

```text
session budget = 11h
safety reserve = 45m
actual compute budget = 10h15m
```

This leaves room to save and publish checkpoints.

Kaggle currently permits up to 12-hour GPU notebook sessions and 9-hour TPU sessions.

---

# 84. Session Resume

A typical Kaggle GPU session:

```text
0:00
restore

0:05
validation

0:10–10:15
training/evolution

10:15
final evaluation

10:30
checkpoint

10:40
artifact packaging

11:00
stop
```

Do not attempt to consume the full session window.

---

# 85. First Implementation Sprint

Build only:

```text
Dense LM
Static MoE
Topology graph
Prune
Grow
Importance metrics
Random controller
Adaptive controller
Checkpoint/resume
Experiment registry
```

Do NOT initially implement:

```text
merge
split
GDN
adaptive depth
Qwen teacher
Fisher-heavy analysis
TPU backend
```

Get Experiment 0 operational first.

---

# 86. Second Implementation Sprint

Add:

```text
Fisher
functional similarity
temporary ablation
merge
split
expert ecology
```

Run:

```text
random vs adaptive
```

again.

---

# 87. Third Implementation Sprint

Add:

```text
GDN
module-type mutation
adaptive depth
```

Now test whether the architecture itself can change.

---

# 88. Fourth Implementation Sprint

Add:

```text
Qwen architecture priors
```

and compare:

```text
uninformed adaptive
Qwen-informed adaptive
```

Do not copy weights.

---

# 89. Fifth Implementation Sprint

Add:

```text
architecture extraction
fresh initialization
rewinding
teacher initialization
```

Run the four-way experiment.

---

# 90. Sixth Implementation Sprint

Scale:

```text
300M
→
500M
→
1B
```

only for the most promising architecture families.

---

# 91. Dependency Strategy

Prefer:

```text
torch
transformers
datasets
safetensors
numpy
scipy
networkx
pydantic
pyyaml
tqdm
```

Potentially:

```text
flash-attn
```

only where Kaggle compatibility makes it worthwhile.

Avoid building the project around complicated external frameworks until the core mechanism works.

Whittle can be installed from PyPI/source and is useful as a baseline/reference implementation, but our architecture engine should remain independent of it. Whittle explicitly supports supernetwork training, multi-objective search, structural pruning, distillation and efficiency estimation.

---

# 92. Baseline Integration With Whittle

Use Whittle as:

```text
baseline A
```

not as:

```text
our architecture framework
```

The experiment becomes:

```text
Whittle extracted subnet
vs
Adaptive discovered subnet
```

This is one of the most important comparisons.

---

# 93. External Reference Models

Use:

```text
Qwen3-Next-80B-A3B
```

primarily as:

```text
architecture reference
teacher
analysis target
```

not as something to train directly on Kaggle.

Its 80B total / ~3B active architecture is far beyond the practical V0 Kaggle training target. The released model uses 512 experts and native 262K context, while the instruct checkpoint is non-thinking; the separate Thinking variant exists and should be chosen deliberately when teacher reasoning behaviour is desired.

---

# 94. Qwen Analysis Tool

Build:

```text
tools/qwen_analyzer.py
```

Capabilities:

```text
inspect architecture
inspect router configuration
capture expert activations
calculate expert usage
cluster experts
estimate redundancy
export architecture priors
```

Output:

```text
qwen_prior.json
```

Example:

```json
{
  "attention_interval": 4,
  "expert_count": 512,
  "active_experts": 10,
  "shared_experts": 1,
  "candidate_ratios": [2, 3, 4, 5]
}
```

These are priors, not hard-coded answers.

---

# 95. Research Dashboard

Generate a machine-readable `report.json`.

Fields:

```text
experiment
architecture
tokens
loss
capability
training_flops
search_flops
total_flops
parameters
active_parameters
mutation_count
topology_stability
```

Then produce plots:

```text
quality vs FLOPs
quality vs parameters
quality vs active parameters
loss vs tokens
architecture distance vs training
expert population vs training
topology entropy vs training
```

---

# 96. Important Visualization

Plot topology through time.

Example:

```text
step →

Layer 0   ATT ATT ATT ATT ATT
Layer 1   GDN GDN GDN GDN ATT
Layer 2   MOE MOE MOE MOE MOE
Layer 3   ATT ATT GDN GDN GDN
...
```

This should allow us to visually observe:

```text
architecture emergence
```

and identify transition points.

---

# 97. Architecture Fossils

At every structural checkpoint save:

```text
genome_step_000.json
genome_step_001.json
...
```

This lets us reconstruct the evolutionary trajectory.

The trajectory itself is scientifically useful.

---

# 98. Hypothesis Tracking

Every experiment configuration should explicitly state:

```text
hypothesis
prediction
control
result
decision
```

Example:

```yaml
hypothesis:
  "adaptive topology improves capability per FLOP"

prediction:
  "adaptive > random churn"

control:
  "same mutation schedule, random components"

decision_if_false:
  "remove expensive importance subsystem"
```

This prevents the project from becoming uncontrolled architecture tinkering.

---

# 99. Go/No-Go Gates

## Gate 1

Dense baseline trains reliably.

## Gate 2

Random churn reproducible.

## Gate 3

Adaptive > random.

If failed:

**stop and simplify.**

## Gate 4

Adaptive > static MoE.

## Gate 5

Fresh architecture remains competitive.

## Gate 6

Architecture transfers across seeds.

## Gate 7

Architecture transfers across scale.

Only after Gate 7 should serious compute be invested.

---

# 100. Final Research Deliverables

The project should eventually produce:

```text
1. adaptive-topology source repository

2. reproducible Kaggle notebooks

3. dataset manifests

4. baseline checkpoints

5. adaptive checkpoints

6. architecture genomes

7. topology evolution logs

8. experiment registry

9. benchmark results

10. fresh-initialization results

11. Qwen architecture analysis

12. paper-quality figures

13. final research report
```

The **architecture genome and the topology controller** are the most valuable artifacts.

---

# 101. Final Project Definition

The complete system can be summarized as:

```text
                     TRAINING DATA
                          │
                          ▼
                ┌──────────────────┐
                │ Adaptive Model   │
                │ Supernetwork     │
                └────────┬─────────┘
                         │
                  FAST OPTIMIZER
                         │
                    weights
                         │
                         ▼
                ┌──────────────────┐
                │ Measurements     │
                ├──────────────────┤
                │ activation       │
                │ gradient         │
                │ Fisher           │
                │ routing          │
                │ ablation         │
                │ function         │
                │ stability        │
                └────────┬─────────┘
                         │
                  SLOW CONTROLLER
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
          PRUNE        GROW         MERGE
            │            │            │
            ▼            ▼            ▼
          SPLIT        CLONE        REWIRE
            │            │            │
            └────────────┼────────────┘
                         │
                   NEW TOPOLOGY
                         │
                         ▼
                    CONTINUE
                     TRAINING
                         │
                         ▼
                 TOPOLOGY FREEZES
                         │
                         ▼
                COMPUTATIONAL GENOME
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
           RANDOM      REWOUND     TEACHER
          RETRAIN      RETRAIN     RETRAIN
             │           │           │
             └───────────┼───────────┘
                         ▼
                   FINAL COMPARISON
                         │
                         ▼
                CAPABILITY / FLOPs
```

---

# 102. The Core Scientific Test

The entire project ultimately reduces to one experiment:

> **Given the same data, compute budget, parameter budget, and initialization conditions, can a topology discovered through training outperform a topology designed by humans?**

And the stronger version:

> **Can the discovered topology be extracted as a reusable computational genome that remains advantageous when freshly initialized and trained independently?**

That is the result worth pursuing.

---

# 103. Practical Target

### V0

```text
256–350M parameters
T4×2
1024–2048 context
2–10B research tokens
```

### V1

```text
500M–1B
T4×2 / TPU v5e
2048–4096 context
```

### V2

```text
1B–3B
TPU / external GPU required
```

### V3

```text
8B+
external compute
```

### Frontier objective

```text
10B–30B+
Chinchilla/overtraining experiments
```

The Kaggle phase exists to prove the **algorithm**, not to brute-force the final model.

---

# 104. First Three Kaggle Experiments

The first three actual notebooks should be:

```text
01_dense_baseline
```

Prove:

```text
model trains
checkpoint resumes
metrics reproducible
```

Then:

```text
02_random_topology_churn
```

Prove:

```text
topology mutation engine works
mutation cost measurable
random control reproducible
```

Then:

```text
03_adaptive_topology_v0
```

The sole question:

> **Does information-guided topology evolution beat exactly the same amount of random topology evolution?**

Everything else comes later.

---

# 105. Final Principle

Do not let the project become:

> "an extremely complicated way to prune a model."

The progression must remain:

```text
pruning
   ↓
dynamic sparsity
   ↓
topology evolution
   ↓
architecture discovery
   ↓
architecture reusability
   ↓
computational genome
```

The moment the evidence shows that the added complexity no longer improves the capability/FLOP frontier, stop adding mechanisms.

The goal is not maximum novelty.

The goal is to establish whether **learning the architecture itself** is a useful additional degree of freedom in LLM pretraining.