# Where the wall-clock goes, and which headroom is real

A **design and measurement note, not a record**. It answers three questions with
arithmetic and with the two rates the campaign has actually logged: why a leg of the
code-LM project takes the hours it takes, which of the available speedups are real, and
which of them change what the experiment *means*. Where a number is a live observation
rather than a stored artifact it says so and carries its timestamp; nothing here is a
ledger claim, and nothing here belongs in `scripts/verify_record.py`.

Written 2026-09-25 20:10 UTC, while the distillation leg's first arm was training.

## The card is already running at roughly half its arithmetic peak

The GTX 1070 has 1,920 fp32 lanes at 1.683 GHz boost, so its fp32 peak is
`1920 x 2 x 1.683e9` = **6.46 TFLOP/s**. Both of the project's *live* throughput
measurements land close to half of that, which is a normal figure for an eager
PyTorch step loop:

| run | config | tokens/s | dense FLOPs/token | implied rate | of peak |
|---|---|---|---|---|---|
| throughput benchmark, fp32 batch 8 | `DEV_V0_CONFIG` (h320 d6) | 8,747 | 101.4 MFLOP | 0.89 TFLOP/s | 13.7% |
| dev cycle end to end, 2026-09-19 | `DEV_V0_CONFIG` (h320 d6) | 24,299 | 101.4 MFLOP | 2.46 TFLOP/s | 38% |
| code-LM arm 1, 2026-09-25 20:10 | `match` (h768 d12) | 2,731 | 1,055.2 MFLOP | 2.88 TFLOP/s | 45% |

The two `DEV_V0_CONFIG` rows are different measurements of the same shape (the note that
records the second one says so, and flags that they disagree on peak memory), and the shape
they agree on is the smallest one the project runs: at 14.9M parameters it is launch-bound,
which is why it converts to 13.7% of peak while the code-LM shape - the one that matters for
a GPU week - converts to 45%.

Dense-runtime FLOPs come from `estimate_dense_runtime_flops_per_token`, which is the
estimator for the arithmetic this implementation *actually executes* rather than the
arithmetic the routing contract books (see the next section for why those differ).

Two consequences follow, and they point in opposite directions from the intuition that
"training is slow, so the training loop must be slow":

1. **There is little left in the step loop.** A perfect kernel-level rewrite of an eager
   fp32 loop that is already at 45% of peak is worth well under 2x, and the two live
   measurements agree on what the ceiling is.
2. **But the loop is executing more arithmetic than the contract requires**, and that
   gap is measured, not estimated. It is the one training-side lever that is not
   throughput tuning.

## Lever 1: the expert block executes 1.93x the arithmetic the contract books

`_MoEFeedForward.forward` is the *dense* form: every expert is applied to every token by
one batched matmul, and only then are the top-k rows selected and gate-weighted. The
routing contract books `active_parameter_count`, so the estimator pair separates the two:

| config | contracted | dense-runtime | executed/booked | if the expert block cost contracted |
|---|---|---|---|---|
| `match` (h768 d12 e8 a2) | 545.6 MFLOP/token | 1,055.2 MFLOP/token | **1.934x** | step 1.93x faster, 1.49 TFLOP/s |
| `DEV_V0_CONFIG` (h320 d6 e8 a2) | 57.1 MFLOP/token | 101.4 MFLOP/token | 1.774x | step 1.77x faster, 1.39 TFLOP/s |

The ratio is not a subtlety of this shape: `active/total` is 0.490 at `match` and 0.506
at the dev shape, so the dense form does about twice the expert arithmetic in both cases.
The module already carries the formulation that avoids it - `eager_forward`, true
per-expert dispatch - and it is **pinned equivalent to the dense `forward` by test**, so
the correctness harness for switching forms exists and is in use.

**The decision cannot be read off the existing measurement, and the existing measurement
is not usable as stated.** `_MoEFeedForward`'s docstring reports *51.8 ms/step eager vs
42.5 ms/step dense (79k -> 96k tokens/s, +18%)*, measured on this card for
`DEV_V0_CONFIG` at batch 8 x context 512, and that is the basis on which the dense form
was kept. At that config and batch, the dense form's arithmetic per step is
`101.4 MFLOP x 4096` = 415 GFLOP, so:

| quoted | implies | against the 6.46 TFLOP/s fp32 peak |
|---|---|---|
| 42.5 ms/step, dense | 9.77 TFLOP/s | **151% of peak** |
| 51.8 ms/step, eager | 8.02 TFLOP/s | **124% of peak** |

Neither pair can have been measured as described on this card at this config, and the
cross-check against the project's own measurement of the same shape says the same thing more
legibly: the throughput benchmark records **0.468 s/step** for fp32 batch 8 at this geometry,
while the docstring claims 0.0425 s/step - **11.0x faster than the same shape was measured at**,
and 151% of the card's physical peak. That is a stronger statement than
`artifacts/arch-zoo/PERFORMANCE_NOTES.md` makes when it records the docstring rates as mutually
inconsistent with each other and with the table: the *pair* may well have been measured on some
other shape, but it is not this one, so the +18% margin it was used to justify is unsupported in
the direction that matters (it is the figure that would *under*-provision a budget).

Why the crossover should be re-measured at the code-LM shape rather than extrapolated
from the dev shape, either way: `forward`'s waste is the same ~2x at both, but
`eager_forward`'s overhead is a per-expert Python pass - a boolean mask, a `nonzero`, a
gather and a `zeros_like` scatter, `2 x experts` times per layer - whose cost scales with
*tokens x layers*, while the arithmetic it saves scales with *tokens x parameters*. From
the dev shape to `match`, parameters rise 11.1x and layers rise 2x, so the overhead's
share of the step falls by roughly 5x. The measured 18% margin for dense at the small
shape is therefore not evidence about the large one, in either direction.

The right next step is the cheap one: time both forms at the `match` shape for a few
hundred steps, on the freed card, before changing any training code. Expected under the
arithmetic above is up to 1.93x with grouped dispatch (sort tokens by expert, one batched
gemm over the expert groups, one scatter), which is strictly better than either existing
form and still pinned equivalent by the same test. The measurement decides; this note
does not.

## Lever 2: generation re-forwards the whole prefix, once per token - ~300x to be had

This is where the difference between "training throughput" and "time to a result" lives,
and it is much larger than Lever 1. `train_code_lm.generate` and
`code_bench.sample_token` both sample one token from a **full forward over the whole
prefix**, with no KV cache and no `past_key_values`:

```python
logits = model(tokens[:, -CONTEXT_SIZE:])[:, -1, :]
```

Generating a 512-token completion from a ~185-token prompt therefore forwards

| | token-forwards | attention term (token-layers, squared) |
|---|---|---|
| no cache (today) | 208,516 | 91.0M |
| one prefill + cached steps | 697 | 128,945 |
| **ratio** | **299x** | **706x** |

At the campaign's measured 22 s/problem for a *toy* student (and 1.5-4 h estimated per
`match` arm for a 164-problem HumanEval pass), this is the difference between a benchmark
that fits in an evening and one that does not. It is also the single largest available
reduction in the gap between "training finished" and "we have a benchmark number".

It is **inference-only and cannot change what an arm learns**: a cache changes which
intermediate keys and values are recomputed, not which weights are updated or what the
logits are. That makes it safe for the *comparability* of checkpoints, but it is still a
change to `src/act/model/`, which every arm imports, so it must land between legs and
carry the numeric check that makes the claim testable - cached logits equal uncached
logits, token by token, at the project's usual tolerance.

## What is not a lever, and why

- **A smaller `--max-new-tokens`.** The published comparator (HumanEval pass@1 7.84% for
  `tiny_starcoder_py`) was decoded at 512 tokens, temperature 0.2, top-p 0.95;
  `code_lm_table` already refuses to tabulate rows decoded differently. Shortening it buys
  time by changing the protocol, which is the one trade this project has consistently
  refused.
- **A second GPU.** Both live measurements are single-device, and the step loop is
  single-device by construction. Two T4s would need the harness to become distributed,
  which is a project, not a setting - and on the evidence above the first 1.9x is
  available without it.
- **bf16/fp16 autocast on this card.** Pascal's fp16 rate is a fraction of its fp32 rate,
  which is why the project's own dev measurements are fp32. This is a T4 lever, not a
  1070 one.

## The honest size of the problem the speedups do not solve

The leg's student trains on `8 x 512 x 3 epochs` windows, i.e. **46,080,000 tokens**. At
166.4M parameters that is **0.28 tokens per parameter**. The comparator whose 7.84% this
leg is measured against was trained on 100B tokens - about **609 tokens per parameter**,
and **2,170x** the tokens this leg moves in total.

That is the actual answer to "why does it take so long to get a working model": at this
budget the leg is not slow, it is *small*, and a 1.93x step speedup moves 0.28 to 0.54
tokens per parameter on a road that is three orders of magnitude long. Speed matters
because it buys tokens at all, and because it decides whether the benchmark can be run
for every arm - not because the loop is where the model is being lost. Any plan that
expects a strong SWE model out of a leg-sized budget is mispriced regardless of how fast
the tokens move, and the honest framing of this leg stays what the README already says:
it tests whether *condensing a large teacher* transfers, not whether 46M tokens produce a
good model.

## Summary of the two levers

| lever | where | headroom | cost of landing it | changes what an arm learns? |
|---|---|---|---|---|
| grouped/sparse expert dispatch | `_MoEFeedForward` | up to **1.93x** on every training step | one function + reuse of the existing equivalence test; measure both shapes first | no - the two forms are already pinned equivalent |
| KV cache in generation | `src/act/model` + the two samplers | **~299x** on tokens forwarded per completion | a cache argument through the attention path, a cached-equals-uncached check, landing between legs | no - inference only |
