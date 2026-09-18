# Audit of the v0 architecture-zoo screening

**Status: superseded.** The conclusion recorded in `analysis.md` ("architecture
effect is small, do not proceed to evolutionary search") is not supported by the
evidence in this directory. The harness had five defects, each independently
sufficient to invalidate the comparison. This document is kept for historical
reproducibility; the v0 results should not be cited as a negative result about
architecture.

The superseding experiment is `arch-zoo-100m-v1`.

## Defect 1 — the recorded seed was never used

`evaluate_architecture(..., seed=seed)` stored the seed in the result but never
called `torch.manual_seed`. Initialization therefore depended on the global RNG
state, which advanced across the sequential sweep. Consequences:

- the brief's "critical scientific control" (seed-matched architecture
  comparison) was not implemented;
- architecture effects could not be separated from initialization order;
- every `"seed": 42` in `zoo_results.json` is decorative and the runs are not
  reproducible.

Fixed in `RunConfig`/`train_architecture`: the seed is drawn immediately before
model construction, and `test_same_seed_reproduces_validation_loss` pins it.

## Defect 2 — the headline evidence came from unlearnable data

The 11-candidate table in `analysis.md` was produced on **uniform random
tokens** at context 512. Uniform tokens carry no structure above the entropy
floor `ln(8192) = 9.0109`. Every recorded loss sits between 9.0496 and 9.0873 —
that is 0.4% to 0.8% above the floor.

The reported "range = 3.7x std dev" is therefore a ranking of initialization
luck, not of architecture. Two further points follow:

- the "effect does not transfer across data distributions" observation compares
  a noise ranking against a two-point sample; it carries no information;
- the brief already forbade this: section 17 asks for context 1024/2048, and the
  synthetic screening used 512.

The harness now warns loudly when no token file is supplied.

## Defect 3 — FLOPs/token did not depend on the architecture

`flops_per_token` was computed as `2.0 * total_params`. Since the zoo was built
at matched *total* parameters, every candidate received an almost identical
value: `D8_H768_E8_A2` (44.1M active) and `D10_H768_E6_A2` (53.5M active) are
both recorded as exactly `201451008.0`.

The attention term was also absent, so context length did not enter the cost
model at all. Every compute-matching claim in the v0 table is unsupported.

The harness now reports three separate axes:

| axis | what it measures |
|---|---|
| `contracted_flops_per_token` | the declared top-k routing contract |
| `dense_runtime_flops_per_token` | what this implementation actually executes |
| `measured_tokens_per_s` | wall clock |

These disagree substantially and the disagreement is a result in its own right:
`_MoEFeedForward.forward` evaluates **every** expert on **every** token and then
gathers the top-k rows, so runtime arithmetic tracks *total* expert count. Across
the measured zoo the runtime-to-contract ratio spans **1.00 to 3.18**. Two
candidates with identical contracted FLOPs/token can differ by 3x in wall clock.

## Defect 4 — no validation loss

`final_loss` was the loss of the last training step, the noisiest single number
available, and no held-out split was ever evaluated. The brief (section 19) asks
for validation loss during search.

Compounding this, the repository's held-out stream `artifacts/real-lm/val-tokens.npy`
holds **7,190 tokens** — about seven windows at context 1024, far too few for a
useful evaluation. The harness now carves a 2M-token holdout off the end of the
training stream and truncates training accordingly, so train and validation
never overlap.

## Defect 5 — the real-data sweep left no results

`artifacts/arch-zoo-real/` contains a `zoo_definition.json` and no results file.
Results were written only after the whole sweep finished, so the interruption
lost every completed run. Evaluation now appends each result to JSONL as it
completes.

## Consequence for the research plan

The screened space was also narrower than the brief specifies: all 11 candidates
were sparse MoE with `expert_width == hidden_size`, and the generator could not
emit a dense candidate at all. `architecture_distance` hardcoded its
module-type term to `0.0`.

The replacement zoo spans 8 families, 70.8M-123.3M total parameters and
**25.3M-77.1M active parameters** (a 3x spread at matched total budget), and
includes two effectively dense candidates (`active == total`). Structural
distance is real and enforced: minimum pairwise distance 0.060.

## Measured cost of the replacement experiment

Measured on the development GPU (GTX 1070, 8 GiB) at context 1024, batch 2:

| candidate | total | active | runtime/contract | tok/s | peak MB |
|---|---|---|---|---|---|
| `many_experts_D8_H640_E16_A2_W640` | 123.3M | 31.6M | 3.18 | 4,011 | 4,552 |
| `shallow_wide_D6_H896_E8_A2_W896` | 103.7M | 45.9M | 2.02 | 4,841 | 3,334 |
| `deep_narrow_D14_H512_E8_A2_W512` | 77.7M | 33.6M | 1.91 | 4,637 | 3,486 |
| `wide_experts_D12_H512_E4_A2_W1152` | 73.4M | 45.1M | 1.49 | 5,115 | 3,043 |
| `narrow_experts_D8_H896_E12_A4_W256` | 77.2M | 47.8M | 1.47 | 5,269 | 3,256 |

Every candidate fits in memory. The full three-stage successive-halving sweep
(13 candidates x 400 steps x 2 seeds, then 6 x 800 x 2 seeds, then 3 x 1200 x 3
seeds) is roughly **63M tokens ≈ 4 GPU-hours** at the measured rate, well inside
the 10-20 GPU-hour budget.

For reference, the v0 harness reported 573 tok/s where this one measures ~4,700
on the same model — about an 8x discrepancy that has not yet been explained and
should be understood before any prior wall-clock claim is reused.
