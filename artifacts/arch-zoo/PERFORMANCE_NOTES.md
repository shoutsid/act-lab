# Measured throughput, and a 75x spread in the numbers this repository states

Budget decisions for the architecture search rest on throughput, so throughput
was measured directly on the development GPU (GTX 1070, 8 GiB) rather than taken
from the existing documentation. The documentation does not agree with itself.

## Measured

DEV-scale geometry (depth 6, hidden 320, 8 experts, 2 active, expert width 320,
vocab 8192, context 512), 20 steps, real TinyStories tokens:

| precision | batch | context | tokens/s | s/step | peak MB | val loss |
|---|---|---|---|---|---|---|
| fp32 | 2 | 512 | 5,716 | 0.179 | 617 | 7.3981 |
| fp32 | 8 | 512 | **8,747** | 0.468 | 617 | 7.2507 |
| fp16 | 8 | 512 | 3,779 | 1.084 | 556 | 7.2505 |
| fp16 | 16 | 512 | 4,056 | 2.020 | 918 | 7.2337 |

100M-class geometry (depth 8, hidden 768, 8 experts, 2 active, width 768,
context 1024), batch 2, fp32:

| candidate | total | active | tokens/s | s/step | peak MB |
|---|---|---|---|---|---|
| `many_experts_D6_H704_E16_A2_W448` | 78.3M | 25.3M | 5,779 | 0.355 | 3,057 |
| `few_experts_broad_D10_H768_E4_A4_W768` | 77.1M | 77.1M | 5,010 | 0.409 | 3,022 |
| `balanced_D8_H768_E8_A2_W768` | 100.7M | 44.1M | 4,647 | 0.441 | 3,536 |
| `many_experts_D8_H640_E16_A2_W640` | 123.3M | 31.6M | 3,983 | 0.514 | 4,559 |

## Findings that change how the search should be run

**Batch size is worth more than precision on this card.** fp32 batch 8 is 1.53x
batch 2 (8,747 vs 5,716 tokens/s). Since the 100M zoo peaks at 4.6 GiB at batch 2,
batch is memory-limited rather than compute-limited here, and every candidate fits
comfortably - so the operating point should be the largest batch that fits, not
the smallest that runs.

**fp16 is 2.3x slower than fp32 on a GTX 1070, not faster.** Pascal has
1/64-rate fp16 and cuBLAS does not take a tensor-core path, so autocast adds cast
overhead and buys nothing. The `mixed_precision` option is implemented and
recorded anyway, because the hardware assumption behind it is exactly what a
recorded field protects against - but on this card it must be left `off`. Anyone
re-running on a T4 or newer should re-measure rather than assume.

**Activation checkpointing is numerically transparent.** With and without it, and
across fp32/bf16, the validation loss is bit-identical (5.5918) at test scale. So
it is a pure memory lever and cannot become a confound;
`test_activation_checkpointing_is_numerically_transparent` pins that.

## A 75x spread in the repository's own stated figures

For the same `DEV_V0_CONFIG` shape, `src/act/model/v0_moe.py` states three
mutually inconsistent throughputs:

| source | figure | tokens/s at batch 8 x context 512 |
|---|---|---|
| the measured-config table | 9.8k | 9.8k |
| prose: "folded its per-expert launches into batched matmuls" | "19.6k tokens/s (0.21 s/step)" | 19.6k |
| prose: `_MoEFeedForward` docstring | "42.5 ms/step dense (79k -> 96k tokens/s)" | 96k |

The table describes 9.8k as the value *before* the batched-matmul change and
19.6k as the value after. Measured here: **8.7k**. That matches the pre-change
figure and does not reproduce the claimed 2.2x improvement, let alone the 96k in
the `_MoEFeedForward` docstring (which cannot be reconciled with the table at all
- 96k tokens/s at 4096 tokens/step would be 42.5 ms/step, but the table implies
418 ms/step for the same shape).

This matters for planning: a budget computed from 19.6k would under-provision by
2.2x, and one from 96k by 11x. Every GPU-hour figure in this phase is therefore
computed from the numbers measured above, and the discrepancy is recorded rather
than smoothed over.

## The v0 architecture-zoo throughput figure

The v0 screening reports 573 tokens/s for a 100M model at batch 2, context 512
(1.785 s/step for 1024 tokens). Measured here for the same class of model: 4,647
tokens/s at batch 2, context 1024 (0.441 s/step for 2048 tokens).

The old harness's bytecode survived in `src/act/search/__pycache__/`
(`architecture_eval.cpython-314.pyc`, older than the rewrite), so its structure
was recovered and compared rather than guessed at: same loop, fp32, same model,
same `zero_grad`/`backward`/`step`, no `manual_seed`, no `cuda.synchronize`. The
two harnesses are equivalent in every respect that could affect speed.

Its code therefore cannot account for a 6-7x gap. Fitting `time = fixed +
rate x tokens` to the old and new figures returns a *negative* per-token rate,
which means they cannot describe the same system - so the old numbers were taken
under conditions the artifacts do not record (the v0 sweep ran between 23:06 and
01:06, alongside a `capacity-sweep` written at 22:13-22:41). They should not be
reused, and no conclusion in this phase depends on them.

## Re-measured end to end, 2026-09-19: 24,299 tokens/s wall

The tables above benchmark the model. A GPU-hour budget counts the tokens the
*full session* moves, so the same shape was re-measured through that path:

```bash
PYTHONPATH=src python scripts/dev_cycle.py --steps 400 --tag verify-doc-throughput
```

```text
dev cycle: 14,928,960 total / 7,556,160 active parameters, 1,638,400 tokens per arm
arm            final       val  mutations      tok/s
random        3.8938    3.8938          3     24,807
adaptive      3.8739    3.8739          3     24,386
wall 134.9s for 3,276,800 tokens across 2 arm(s) (24,299 tokens/s end to end)
```

fp32, batch 8 x context 512, the real TinyStories stream, 400 steps per arm, run in a
fresh checkpoint namespace so nothing resumed.

That is **2.8x** the 8.7k this file reports for the same nominal shape, and it is
2.25 minutes for two arms where the dev-loop claim was 2.8 minutes. It should not be
read as a refutation of the table above, because the two are probably not the same
measurement: this file records 617 MB peak where the model docstring records 0.90 GiB
for that shape, and a 20-step benchmark need not be at steady state. What is safe to
say is narrower and more useful: a budget for the dev shape on this card should be
computed from **~24.3k tokens/s end to end**. On that scale the docstring figures this
file flags split in two directions - 9.8k and 19.6k sit *below* the measurement, so a
budget drawn from them over-provisions and wastes GPU-hours, while the 96k in the
`_MoEFeedForward` docstring sits 4x *above* it and is the one still able to
under-provision, which is the failure this file was written about. The 8.7k here is
below the measurement too, so it errs the safe way.
