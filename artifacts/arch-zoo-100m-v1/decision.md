# Architecture Zoo Decision

**ARCHITECTURE EFFECT ESTABLISHED ON THE POOLED BLOCKS, NOT ON THE SCREENING-FREE ONES - spread 0.0434 nats, blocked permutation (exact) p=0.001286, worth 72 training steps (6.0% of the 1200-step budget). The pre-registered rule below reads that significance as the screening seeds.**

Next step: Proceed to evolutionary architecture search (arch-evo-100m-v1), carrying the effect size but not a ranking: the finalists' ordering did not replicate on the seeds that took no part in screening, so the pooled significance is the blocks that did (see the finalist-order rule below).

## Ranked at 1200 steps (mean held-out validation loss)

| Rank | Architecture | Mean val loss | Runs |
|---|---|---|---|
| 1 | `narrow_experts_D8_H896_E12_A4_W256` | 2.7518 | 6 |
| 2 | `shallow_wide_D6_H896_E8_A2_W896` | 2.7653 | 6 |
| 3 | `narrow_experts_D10_H768_E12_A4_W288` | 2.7952 | 6 |

## Evidence

- Architecture spread (best to worst): 0.0434 nats
- Mean within-architecture seed std: 0.0146 nats (descriptive; a range over architectures is not a test statistic)
- Spread in seed-noise units: 2.97
- Seed-matched test (blocked permutation (exact)): p=0.001286 over 6 blocks x 3 architectures; design floor 0.0001286, procedure resolution 2.143e-05
- Unpaired one-way ANOVA (unpaired): F=12.11 on (2, 15) df, p=0.0007
- Effect size: 0.0434 nats = 72 training steps (6.0% of budget), fitted over [800, 1200]

Caveat: 13 architectures were screened at shorter lengths than these 3, and promotion used those runs, so this test is mildly anti-conservative. Confirming on seeds never used in screening has since been run, and the ordering did not replicate there (see the pre-registered rule below), so the significance above rests on the coupled blocks this caveat warns about rather than on a property of the architectures.

## Spread against training length (common architecture set)

| Steps | Architectures | Spread | Seed std | Spread in seed-noise units | Best |
|---|---|---|---|---|---|
| 400 | 3 | 0.1500 | 0.0291 | 5.16 | `shallow_wide_D6_H896_E8_A2_W896` |
| 800 | 3 | 0.0584 | 0.0228 | 2.56 | `narrow_experts_D8_H896_E12_A4_W256` |
| 1200 | 3 | 0.0434 | 0.0146 | 2.97 | `narrow_experts_D8_H896_E12_A4_W256` |

Spread falls 3.46x over 400 -> 1200 steps (2.19x per doubling of training). Seed noise falls with it (5.16 -> 2.97 in seed-noise units), so the effect does not wash out relative to noise at these lengths.

The same architecture is best at every length: **NO**.

## Screening validity: does a short run rank like a long one?

| From | To | n | Spearman | Best at short | Best at long | Winner kept |
|---|---|---|---|---|---|---|
| 400 | 800 | 6 | 0.89 | `shallow_wide_D6_H896_E8_A2_W896` | `narrow_experts_D8_H896_E12_A4_W256` | NO |
| 800 | 1200 | 3 | 1.00 | `narrow_experts_D8_H896_E12_A4_W256` | `narrow_experts_D8_H896_E12_A4_W256` | yes |

## Which axis moves with loss (exploratory)

| Axis | Spearman | p | Range | Ratio |
|---|---|---|---|---|
| total parameters | +0.500 | 1.0000 | 77.2M .. 104M | 1.34x |
| active parameters | -0.500 | 1.0000 | 45.9M .. 47.8M | 1.04x |
| active / total parameters | -0.500 | 1.0000 | 0.4428 .. 0.6195 | 1.40x |
| contracted FLOPs/token | +0.500 | 1.0000 | 342M .. 380M | 1.11x |
| dense-runtime FLOPs/token | +0.500 | 1.0000 | 551M .. 688M | 1.25x |
| measured throughput | -1.000 | 0.3333 | 4.59k .. 5.24k | 1.14x |
| depth | +0.500 | 1.0000 | 6 .. 10 | 1.67x |
| hidden size | -0.866 | 0.6667 | 768 .. 896 | 1.17x |
| attention heads | -0.866 | 0.6667 | 12 .. 14 | 1.17x |
| expert count | +0.000 | 1.0000 | 8 .. 12 | 1.50x |
| active experts | +0.000 | 1.0000 | 2 .. 4 | 2.00x |
| expert width | +0.500 | 1.0000 | 256 .. 896 | 3.50x |

*12 axes tested uncorrected; treat any single marginal p-value as a hypothesis, not a result. Exact p-values are enumerated over n! arrangements and are not comparable to a chi-square test.*

**Confounding structure.** Among 12 spread axes, `measured throughput` tracks loss most strongly, but it moves with `hidden size` (+0.87), `attention heads` (+0.87) at |rho| >= 0.7, so this set cannot separate them.

## Pre-registered rule: was v1's finalist ordering underpowered?

**ORDER COUPLED - THE POOLED SIGNIFICANCE IS THE SCREENING SEEDS**

The pooled test clears the threshold, but the seed blocks do not agree on which architecture comes first. The significance then rests on agreement among blocks that disagree about the top - either because the screening seed that selected the finalists favour their own winner (the anti-conservative coupling v1's caveat named), or because the uncoupled seeds overturn it. Neither is a replicated ordering, so no ranking is recommended however the pooled p reads.

| Condition | Value | Met |
|---|---|---|
| all 6 blocks: p < 0.05 (v1's 3-block test raised power on the screening seeds) | p=0.001286 (floor 0.0001286) | yes |
| screening-free blocks [45, 46, 47] alone: p < 0.05 (no selection coupling) | p=0.1111 (floor 0.02778) | NO |
| every block ranks the same architecture first (both readings, so the top replicates on uncoupled seeds) | 2 different winners across 6 of 6 blocks | NO |

The branch does not depend on which loss it is read on: re-reading the same rule over `final_train_loss`, `mean_train_loss` lands on the same branch, with only the loss field substituted.

The branch is a threshold on a loss, so it was re-read on the other losses these ledgers carry - the same rows with only the loss field substituted, which is a second reading of one design rather than a second experiment:

| Loss read | Branch |
|---|---|
| `validation_loss` - held-out validation loss (the registered reading) | ORDER COUPLED - THE POOLED SIGNIFICANCE IS THE SCREENING SEEDS |
| `final_train_loss` - training loss at the end of training | ORDER COUPLED - THE POOLED SIGNIFICANCE IS THE SCREENING SEEDS |
| `mean_train_loss` - mean training loss over the trajectory | ORDER COUPLED - THE POOLED SIGNIFICANCE IS THE SCREENING SEEDS |

Only the loss field changed; the design, the seeds and the declared thresholds did not.

## Seed-matched contrast against the best architecture

| Candidate | Mean delta | Paired seeds |
|---|---|---|
| `shallow_wide_D6_H896_E8_A2_W896` | +0.0135 | [42, 43, 44, 45, 46, 47] |
| `narrow_experts_D10_H768_E12_A4_W288` | +0.0434 | [42, 43, 44, 45, 46, 47] |

## Seeds per architecture

```json
{
  "narrow_experts_D10_H768_E12_A4_W288": [
    42,
    43,
    44,
    45,
    46,
    47
  ],
  "narrow_experts_D8_H896_E12_A4_W256": [
    42,
    43,
    44,
    45,
    46,
    47
  ],
  "shallow_wide_D6_H896_E8_A2_W896": [
    42,
    43,
    44,
    45,
    46,
    47
  ]
}
```
