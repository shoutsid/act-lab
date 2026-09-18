# Architecture Zoo Decision

**ARCHITECTURE EFFECT ESTABLISHED - spread 0.3244 nats, blocked permutation (monte-carlo) p<=5e-06, worth 217 training steps (27.1% of the 800-step budget).**

Next step: Proceed to evolutionary architecture search (arch-evo-100m-v1), and confirm the finalists on seeds never used in screening first so the selection coupling is removed.

## Ranked at 800 steps (mean held-out validation loss)

| Rank | Architecture | Mean val loss | Runs |
|---|---|---|---|
| 1 | `mtot_D6_H896_E16_A6_W288` | 2.9830 | 3 |
| 2 | `mtot_D14_H896_E8_A5_W128` | 3.0288 | 3 |
| 3 | `mtot_D8_H832_E6_A2_W576` | 3.0295 | 3 |
| 4 | `mtot_D6_H576_E6_A4_W1536` | 3.1267 | 3 |
| 5 | `mtot_D12_H640_E12_A3_W288` | 3.1704 | 3 |
| 6 | `mtot_D5_H512_E16_A2_W832` | 3.1937 | 3 |
| 7 | `mtot_D14_H576_E4_A4_W896` | 3.2567 | 3 |
| 8 | `mtot_D10_H512_E8_A6_W768` | 3.3074 | 3 |

## Evidence

- Architecture spread (best to worst): 0.3244 nats
- Mean within-architecture seed std: 0.0165 nats (descriptive; a range over architectures is not a test statistic)
- Spread in seed-noise units: 19.71
- Seed-matched test (blocked permutation (monte-carlo)): p=5e-06 over 3 blocks x 8 architectures; design floor 6.151e-10, procedure resolution 5e-06
- Unpaired one-way ANOVA (unpaired): F=120.86 on (7, 16) df, p=0.0000
- Effect size: 0.3244 nats = 217 training steps (27.1% of budget), fitted over [400, 800]

## Spread against training length (common architecture set)

| Steps | Architectures | Spread | Seed std | Spread in seed-noise units | Best |
|---|---|---|---|---|---|
| 400 | 8 | 0.5457 | 0.0286 | 19.09 | `mtot_D6_H896_E16_A6_W288` |
| 800 | 8 | 0.3244 | 0.0165 | 19.71 | `mtot_D6_H896_E16_A6_W288` |

Spread falls 1.68x over 400 -> 800 steps (1.68x per doubling of training). Seed noise falls with it (19.09 -> 19.71 in seed-noise units), so the effect does not wash out relative to noise at these lengths.

The same architecture is best at every length: **yes**.

## Structural signal against training length (common architecture set)

The same joint fit at every length, on a **pre-specified** model (`hidden_size`, `depth`). The model is fixed across lengths, so agreement here is a statement about the model rather than about which predictors each length happened to pick.

| Steps | Architectures | R2 | Model p | First gene | beta | p |
|---|---|---|---|---|---|---|
| 400 | 8 | 0.927 | 0.0024 | `hidden_size` | -0.863 | 0.0145 |
| 800 | 8 | 0.936 | 0.00185 | `hidden_size` | -0.944 | 0.00075 |

The model is the same at every length by construction, so the question this table answers is whether *it* holds at each one — and the coefficient p-values below the model do not depend on a per-length selection at all.

## Screening validity: does a short run rank like a long one?

| From | To | n | Spearman | Best at short | Best at long | Winner kept |
|---|---|---|---|---|---|---|
| 400 | 800 | 8 | 0.95 | `mtot_D6_H896_E16_A6_W288` | `mtot_D6_H896_E16_A6_W288` | yes |

## Which axis moves with loss (exploratory)

| Axis | Spearman | p | Range | Ratio |
|---|---|---|---|---|
| total parameters | +0.452 | 0.2675 | 75M .. 81.1M | 1.08x |
| active parameters | +0.119 | 0.7930 | 18M .. 81.1M | 4.51x |
| active / total parameters | +0.119 | 0.7930 | 0.2319 .. 1 | 4.31x |
| contracted FLOPs/token | +0.167 | 0.7033 | 139M .. 586M | 4.20x |
| dense-runtime FLOPs/token | -0.071 | 0.8820 | 497M .. 623M | 1.25x |
| measured throughput | -0.048 | 0.9349 | 3.93k .. 6.39k | 1.63x |
| depth | +0.145 | 0.7142 | 5 .. 14 | 2.80x |
| hidden size | -0.897 | 0.0075 | 512 .. 896 | 1.75x |
| attention heads | -0.897 | 0.0075 | 8 .. 14 | 1.75x |
| expert count | -0.242 | 0.5942 | 4 .. 16 | 4.00x |
| active experts | -0.109 | 0.8595 | 2 .. 6 | 3.00x |
| expert width | +0.575 | 0.1383 | 128 .. 1.54k | 12.00x |

*12 axes tested uncorrected; treat any single marginal p-value as a hypothesis, not a result. Exact p-values are enumerated over n! arrangements and are not comparable to a chi-square test.*

**Confounding structure.** Among 12 spread axes, `hidden size` tracks loss most strongly, but it moves with `attention heads` (+1.00), `expert width` (-0.74) at |rho| >= 0.7, so this set cannot separate them.

## Structural genes fitted together

Standardised least squares over 8 architectures: R2 = 0.960, model p = 0.0044 against a permutation null (20000 draws). Coefficients are nats per standard deviation of that gene, holding the others fixed.

| Gene | beta | p | Range |
|---|---|---|---|
| hidden_size | -1.217 | 0.00755 | 512 .. 896 |
| expert_width | -0.505 | 0.4239 | 128 .. 1536 |
| expert_count | -0.309 | 0.4951 | 4 .. 16 |
| attention_heads | — | — | identical to `hidden_size` (rho=+1.0000), so it carries no separate information |

*Coefficients are standardised and each p-value is two-sided against a permutation null, but predictors are collinear by construction here, so a coefficient surviving the model does not mean the underlying lever is identified. Use a designed set to separate them.; `hidden_size` and `expert_width` move together at rho=-0.74 and cannot be told apart at this size, so their split in the fit is arbitrary.*

A gene can be significant alone and not here, or the reverse. Neither table is a lever ranking: the only design that ranks levers is one that varies them independently by construction.

## Seed-matched contrast against the best architecture

| Candidate | Mean delta | Paired seeds |
|---|---|---|
| `mtot_D14_H896_E8_A5_W128` | +0.0457 | [42, 43, 44] |
| `mtot_D8_H832_E6_A2_W576` | +0.0465 | [42, 43, 44] |
| `mtot_D6_H576_E6_A4_W1536` | +0.1437 | [42, 43, 44] |
| `mtot_D12_H640_E12_A3_W288` | +0.1873 | [42, 43, 44] |
| `mtot_D5_H512_E16_A2_W832` | +0.2107 | [42, 43, 44] |
| `mtot_D14_H576_E4_A4_W896` | +0.2736 | [42, 43, 44] |
| `mtot_D10_H512_E8_A6_W768` | +0.3244 | [42, 43, 44] |

## Seeds per architecture

```json
{
  "mtot_D5_H512_E16_A2_W832": [
    42,
    43,
    44
  ],
  "mtot_D12_H640_E12_A3_W288": [
    42,
    43,
    44
  ],
  "mtot_D8_H832_E6_A2_W576": [
    42,
    43,
    44
  ],
  "mtot_D6_H896_E16_A6_W288": [
    42,
    43,
    44
  ],
  "mtot_D6_H576_E6_A4_W1536": [
    42,
    43,
    44
  ],
  "mtot_D10_H512_E8_A6_W768": [
    42,
    43,
    44
  ],
  "mtot_D14_H896_E8_A5_W128": [
    42,
    43,
    44
  ],
  "mtot_D14_H576_E4_A4_W896": [
    42,
    43,
    44
  ]
}
```
