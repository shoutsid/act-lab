# Module-Type Cost at 100M / Context 1024

**Experiment:** `mixer-benchmark` (raw log: `bench.log`, data: `mixer_throughput.json`)
**Question (brief §14):** can `layer.type ∈ {attention, GDN, hybrid}` be a first-class
search gene at the first architecture-zoo scale?

**Answer: not on this hardware, in this implementation.** GDN costs 4–7× the wall
clock per layer and exceeds the card's memory at 6 of 8 layers.

## Geometry

Held fixed so the *only* variable is the sequence mixer: depth 8, hidden 768,
8 experts, 2 active, expert width 768, batch 2, context 1024, fp32. All nine
layouts are matched to the same block geometry and land within 0.03% of each
other on total parameters (100.7–100.9M), so this is a mixer comparison, not a
parameter comparison.

| pattern | GDN layers | params | tok/s | s/step | vs attention | peak MB |
|---|---|---|---|---|---|---|
| `all_attention` | 0/8 | 100.7M | **4,793** | 0.427 | 1.0× | 3,498 |
| `att_3to1` | 2/8 | 100.8M | 1,209 | 1.673 | **4.0×** | 5,048 |
| `alternating` | 4/8 | 100.8M | 700 | 2.918 | **6.8×** | 6,605 |
| `gdn_att_last` | 4/8 | 100.8M | 700 | 2.915 | **6.8×** | 6,597 |
| `gdn_att_first` | 4/8 | 100.8M | 693 | 2.943 | **6.9×** | 6,606 |
| `gdn_3to1` | 6/8 | 100.9M | — | — | **OOM** | > 8,192 |
| `all_gdn` | 8/8 | 100.7M | — | — | **OOM** | > 8,192 |
| `all_hybrid` | 8/8 | 100.9M | — | — | **OOM** | > 8,192 |
| `hybrid_quarter` | 6/8 | 100.8M | — | — | **OOM** | > 8,192 |

`all_attention` reproduces the independently measured `architecture_eval`
throughput for the same shape to within 1% (4,793 vs 4,733 tok/s), so the harness
is measuring what the training path measures.

## The memory cost is predictable, and was predicted before it was measured

The recurrence is written as an explicit scan
(`_GatedDeltaMixer.scan_segment`), so autograd retains a
`(batch, heads, key_dim, value_dim)` **write** and a **state** per timestep. At
batch 2, heads 12, head_dim 64 that is 2·12·64·64 = 98,304 floats = 393 KB each,
so ~0.8 MB retained per step, ×1024 steps ≈ **0.8 GB per GDN layer**.

| GDN layers | predicted | measured |
|---|---|---|
| 2 | ~5.0 GB | 5,048 MB |
| 4 | ~6.6 GB | 6,597–6,606 MB |
| 8 | ~10 GB | OOM (> 8,192 MB) |

The arithmetic closes to within a few percent, which means the constraint is
understood rather than merely observed.

## What this does and does not rule out

**Does not rule out GDN as an architecture.** This mixer is a compact member of
the Gated DeltaNet family — shared key/value dimension, no short convolution —
deliberately sized near attention so a matched-budget comparison is possible. It
is not Qwen3-Next's block and nothing measured here says anything about whether
the 3:1 schedule is a good design.

**Does rule out the comparison as currently posed at this scale.** §15 requires
*matched* budgets and §16 requires accounting for search cost. Spending 4–7× the
wall clock per module-type arm means a module-type search cannot be
compute-matched against an attention baseline within a 10–20 GPU-hour phase, and
three of the nine layouts cannot run at all.

## The fixable half: segmentation is implemented and verified

Chunked scanning (opt-in `gdn_chunk_size`) stores only segment boundaries and
recomputes within a segment, cutting retention by roughly an order of magnitude
at the cost of one extra forward per segment. It is numerically free — pinned by
`tests/test_gated_delta_mixer.py`:

- chunked output matches the full scan to 1e-6 for chunk sizes 1, 2, 4, 5, 6, 7,
  including sizes that do not divide the sequence evenly;
- **gradients** match the full scan, which is the part checkpointing actually
  changes;
- state is verified to survive a chunk boundary, since silently resetting it per
  segment would produce plausible-looking but wrong readouts.

Verification is by test rather than by argument because the earlier harness
defects in this phase were all of the form "the code did something reasonable
that was not what the record said".

## The unfixable half: the scan is launch-bound

Segmentation buys memory, not time. At depth 8 with 4 GDN layers the forward runs
4 × 1024 = 4,096 sequential Python-level steps, and the measured 2.9 s/step is
consistent with kernel-launch overhead rather than arithmetic. Recomputing during
backward makes the time cost of a *search* worse, not better.

Making module type a genuine matched-budget gene therefore needs a **chunkwise
parallel kernel** — a real implementation with a correctness risk, which per §30
should not be written until a simpler experiment earns it. The decision recorded
in `artifacts/arch-zoo-100m-v1/decision.md` does not depend on module type, so
this is deferred rather than blocking.

## Consequence for the search space

The module-type axis (§14) is **not available** at 100M/context 1024 on an 8 GB
GTX 1070. Any claim that the architecture range is "too narrow" must therefore be
made about the *structural* genes that do run — depth, width, expert count,
active experts, expert width — which is exactly what `arch-zoo-100m-v2`'s
matched-total design varies, at ×4.51 on active parameters within ×1.08 on total.
