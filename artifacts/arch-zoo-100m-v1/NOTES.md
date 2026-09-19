# arch-zoo-100m-v1 — experiment record

**Status: complete.** 47 runs, 3.60 GPU-hours. Verdict in `decision.md`:
**NOT ESTABLISHED**. Consolidated phase findings in
`artifacts/ARCH_SEARCH_FINDINGS.md`.

## What ran

`arch-zoo-100m-v1`: successive halving over the 13-candidate attention-MoE zoo,
context 1024, batch 2, on the real TinyStories stream.

- Stage 1: 13 candidates x 400 steps x 2 seeds
- Stage 2: best 6 x 800 steps x 2 seeds
- Stage 3: best 3 x 1200 steps x 3 seeds

Measured 150-200 s per run, roughly **3.7 GPU-hours** total. Results append to
`stage{1,2,3}_seed{N}_results.jsonl` as each run finishes, so an interrupted
sweep keeps every completed run.

### Finalists at 1200 steps x 3 seeds

| rank | architecture | total | active | val loss |
|---|---|---|---|---|
| 1 | `narrow_experts_D8_H896_E12_A4_W256` | 77.2M | 47.8M | 2.7492 |
| 2 | `shallow_wide_D6_H896_E8_A2_W896` | 103.7M | 45.9M | 2.7724 |
| 3 | `narrow_experts_D10_H768_E12_A4_W288` | 83.1M | 47.7M | 2.7896 |

Spread 0.0404 nats, mean within-architecture seed std 0.0179.

| test | p |
|---|---|
| exact seed-matched permutation on values (primary) | 0.139 |
| same on within-block ranks (robustness) | 0.194 |
| unpaired one-way ANOVA | 0.135 |
| paired t, best vs worst | 0.093 |

**NOT ESTABLISHED**, worth 68 training steps (5.6% of the budget). With 3
architectures on 3 seeds the smallest attainable p is 1/36 = 0.028, so this
design cannot resolve a 2-of-3-consistent ordering however it falls - a property
of the design, not of the world.

### Stage 1 ranked all 13 at 400 steps, and its spread did not survive

The stage-1 spread was 0.7173 nats, single seed. The top four all carried 44-48M
active parameters and the bottom two were the deepest designs - but depth, width
and parameter count move together here, and *none* of the recorded cost axes
predicts loss (total params ρ = −0.26, p = 0.39; active params ρ = −0.14,
p = 0.66). So 0.72 nats is not a parameter effect, and this design cannot say
what it is.

What it does show is that it does not persist. The spread shrinks as training
lengthens: 0.1500 → 0.0584 → 0.0404 across 400/800/1200 steps for the finalist
set, and 0.1796 → 0.1126 from 400 to 800 across the six stage-2 survivors. That
is a 1.4-1.6x shrink per doubling of training, which is why the 400-step screen
promoted `shallow_wide_D6` as the winner and 800 steps gave it to
`narrow_experts_D8` by 0.009 nats - inside the seed noise.

Watch progress, or report on a partial sweep at any point, with:

```
PYTHONPATH=src python scripts/arch_zoo_analyze.py --experiment-id arch-zoo-100m-v1 --write
```

The `decision.md` committed in this directory is the **pooled six-seed** record, not the
three-seed one the command above writes. Section 9's open item asked for these finalists
to be re-run on seeds that took no part in screening, and that arm is
`arch-zoo-100m-v1-finalists-6seed`; pooling the two lowers this design's power floor
from 1/36 to 1/7776, so the pooled write is the authoritative record here and is what
the branch quoted above was read from:

```
PYTHONPATH=src python scripts/arch_zoo_analyze.py \
  --experiment-id arch-zoo-100m-v1 \
  --also arch-zoo-100m-v1-finalists-6seed \
  --finalist-order-rule --write
```

Reproduce from scratch:

```
PYTHONPATH=src python scripts/arch_zoo_screen.py --experiment-id arch-zoo-100m-v1
```

## Throughput and precision

See `artifacts/arch-zoo/PERFORMANCE_NOTES.md`. Three findings that change how the
search should be run: batch size is worth more than precision on this card
(fp32 batch 8 is 1.53x batch 2); **fp16 is 2.3x slower than fp32 on a GTX 1070**,
so `mixed_precision` must stay `off` here despite being implemented and recorded;
and activation checkpointing is numerically transparent, so it cannot become a
confound. That document also records a 75x spread in the throughput figures
`v0_moe.py` states for one configuration (9.8k / 19.6k / 96k tokens/s), only the
lowest of which reproduces.

`RunConfig` now carries `mixed_precision` and `activation_checkpointing`, both
recorded with every result, because a loss measured in one precision is not
comparable to a loss measured in another.

## The one search space, and a fairness bug it exposed

There is now a single declared grid, `enumerate_search_space`, and every method
draws from it: the curated zoo selects cells, the matched designs pin one axis
within it, and the random baseline samples it uniformly. The grid is 15,224 cells
at context 1024 under a 150M parameter ceiling, spanning 12.1M-149.7M total and
10.8M-149.6M active parameters.

That unification fixed a real defect. The curated family recipes reached depth 14,
5-6 active experts and expert widths of 256-1152, none of which the previously
declared grid could produce. The random baseline was therefore sampling a
*strictly narrower space* than the manually selected candidates it exists to be
compared against - a bias that would have flattered the manual zoo in exactly the
comparison section 11 asks for. A containment test now asserts that every curated
candidate is a cell of the declared space, which is what caught it.

Expert widths are absolute rather than ratios of the hidden size, and the hidden
sizes are always included as possible widths, so `expert_width == hidden_size` is
always reachable.

## Cost matching (brief section 21)

Parameter count and per-token FLOPs are separate constraints, and the v1 families
zoo varies both at once - 70.8M-123.3M total *and* 25.3M-77.1M active - so it
cannot attribute an effect to either. `generate_matched_zoo` holds one axis and
spreads the other, and the driver reaches it with `--design`:

| design | held within | spread |
|---|---|---|
| `matched-total` @78M | total params 7.9% | active 18.0M-81.1M (**x4.51**) |
| `matched-flops` @2.6e8 | FLOPs/token 7.6% | total 27.8M-92.5M (**x3.33**) |
| `matched-active` @35M | active params 8.7% | total 33.5M-130.1M (**x3.89**) |

Those spreads were **half as wide before a bug was fixed in the selection**. The
original code filled slots by walking the matched band from one end, and because
the structural-diversity floor rejects most neighbours, every accepted cell came
from that end: at 78M ±5% the band holds 1,059 cells spanning 18.0M-81.8M active
parameters and the sweep returned 18.0M-37.8M. A design whose entire purpose is
"matched total, active spread" would never have tested a dense model. Stratifying
the axis into one bin per slot fixed it. The old test passed the broken version
because it asserted `varied_spread_ratio > 1.2` and the broken design reached
2.1x; the replacement asserts coverage of the available band.

These designs surface something the cost model has to be honest about. In
`matched-total`, contracted FLOPs/token still vary by **x1.62** while measured
wall-clock varies only **x1.08** - because the dense-expert runtime bills every
expert on every token, runtime tracks *total* expert count. So "matched FLOPs" is
three different claims depending on whether you mean the declared routing
contract, the arithmetic executed, or wall clock. All three are recorded per run.

## Random baseline (brief section 11)

`RandomArchitectureSearch` samples uniformly from the same grid, seeded and
without replacement, and applies no operator of any kind - a baseline that
quietly optimises is not a baseline. It is reachable as `--design random`. The
first comparison it supports is uniform sampling versus manual selection under
equal compute; `--candidates` sets the count for every design so the two are
comparable.

## Module type as a gene

`V0MoECausalLM` previously ignored `layer.module` entirely, so the zoo was
attention-only. Added `V0MoEConfig.module_types`, empty by default so every
recorded experiment, parameter count and reconstruction keeps its meaning
(`DECLARED_V0_CONFIG` still returns exactly `(285336576, 134341632)`), and
`_GatedDeltaMixer`, a gated delta-rule mixer. Per head, with L2-normalized `q`/`k`:

    alpha_t = exp(-softplus(gate_t))     decay, one value per key dimension
    beta_t  = sigmoid(write_t)           write strength, one per head
    S_t     = diag(alpha_t) (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
    o_t     = S_t^T q_t

**This is a compact member of the Gated DeltaNet family, not Qwen3-Next's block.**
It shares one key/value dimension and omits the short convolution, so it costs
`4h^2 + 2h*heads` against attention's `4h^2` - within a few percent, which is what
makes a matched-budget comparison possible. It is not a reproduction, and nothing
here implies the Qwen3-Next arrangement is optimal.

`tests/test_gated_delta_mixer.py` checks properties that follow from the
definition rather than a re-implementation of it: with `beta=1` the readout for
the written key reproduces the written value exactly (the defining property, and
the check that would catch a wrong decay axis); with `beta=0` after one write the
readout scales as `alpha^t`; perturbing the final token leaves earlier outputs
bit-identical; and the analytic parameter count equals the built model for every
mixer pattern.

`generate_module_pattern_zoo` holds geometry fixed and varies only the mixer, so a
difference between two candidates is attributable to the pattern:

```
aaaaaaaa all_attention   gggggggg all_gdn        hhhhhhhh all_hybrid
gagagaga alternating     aaagaaag att_3to1       gggaggga gdn_3to1
aaaagggg gdn_att_first   ggggaaaa gdn_att_last   hggghggg hybrid_quarter
```

The 3:1 schedule is one candidate among several, alongside its inverse, and is
nowhere assumed optimal. Prepared, not launched: module type is a different axis,
so it takes a new experiment ID (`arch-zoo-100m-v2`) rather than amending V1.

## Measured: the GDN mixer is not a usable search gene here

Taken once the sweep released the card. Full write-up in
`artifacts/mixer-benchmark/FINDINGS.md`; the short version, at depth 8, hidden
768, context 1024, batch 2, all nine layouts matched to within 0.03% on total
parameters:

| pattern | GDN layers | tok/s | vs attention | peak MB |
|---|---|---|---|---|
| `all_attention` | 0/8 | 4,793 | 1.0x | 3,498 |
| `att_3to1` | 2/8 | 1,209 | 4.0x | 5,048 |
| `alternating` | 4/8 | 700 | 6.8x | 6,605 |
| `gdn_3to1`, `all_gdn`, `all_hybrid` | 6-8/8 | — | **OOM** | > 8,192 |

`all_attention` reproduces this evaluator's own throughput for the same shape to
within 1%, so the benchmark measures what the training path measures.

The memory cost was predicted before it was measured: the explicit scan retains a
`(batch, heads, key_dim, value_dim)` write *and* state per timestep, about 0.8 GB
per GDN layer, and the prediction matches at 2 layers (5.0 GB), 4 (6.6 GB) and 8
(OOM). Segmented checkpointing (`gdn_chunk_size`) is implemented and verified
numerically free - outputs *and gradients* identical, state verified to survive a
chunk boundary - which removes the memory blocker but not the time one: 4 GDN
layers means 4,096 sequential Python steps per forward.

So module type is one chunkwise parallel kernel away from being searchable, not
one refactor away. Per §30 that kernel is deferred rather than written
speculatively.

## Standing caveats

- The v0 harness reported 573 tok/s where this one measures ~4,700 on the same
  shape. The gap was resolved: the v0 source was never committed, its bytecode
  survived in `__pycache__`, and disassembly shows a structurally equivalent loop,
  so the two figures cannot describe the same system. Those numbers should not be
  reused, and no conclusion in this phase depends on them. See
  `artifacts/arch-zoo/PERFORMANCE_NOTES.md`.
- Finalists were promoted using shorter runs on the same seeds, so the tests above
  are mildly anti-conservative. `decision.md` records this; v2 avoids it by
  construction and fresh seeds would remove it entirely.
- The GDN mixer is **not** Qwen3-Next's block (shared key/value dimension, no
  short convolution) and nothing in this experiment speaks to the 3:1 schedule's
  optimality.
