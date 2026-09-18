# Architecture Zoo Decision

**ARCHITECTURE EFFECT ESTABLISHED - spread 0.2462 nats, blocked permutation (monte-carlo) p<=5e-06, worth 367 training steps (30.6% of the 1200-step budget).**

Next step: Proceed to evolutionary architecture search (arch-evo-100m-v1), and confirm the finalists on seeds never used in screening first so the selection coupling is removed.

## Ranked at 1200 steps (mean held-out validation loss)

| Rank | Architecture | Mean val loss | Runs |
|---|---|---|---|
| 1 | `mtot_D6_H896_E16_A6_W288` | 2.7446 | 3 |
| 2 | `mtot_D8_H832_E6_A2_W576` | 2.7859 | 3 |
| 3 | `mtot_D14_H896_E8_A5_W128` | 2.7932 | 3 |
| 4 | `mtot_D6_H576_E6_A4_W1536` | 2.8731 | 3 |
| 5 | `mtot_D12_H640_E12_A3_W288` | 2.8803 | 3 |
| 6 | `mtot_D5_H512_E16_A2_W832` | 2.9214 | 3 |
| 7 | `mtot_D14_H576_E4_A4_W896` | 2.9604 | 3 |
| 8 | `mtot_D10_H512_E8_A6_W768` | 2.9908 | 3 |

## Evidence

- Architecture spread (best to worst): 0.2462 nats
- Mean within-architecture seed std: 0.0148 nats (descriptive; a range over architectures is not a test statistic)
- Spread in seed-noise units: 16.59
- Seed-matched test (blocked permutation (monte-carlo)): p=5e-06 over 3 blocks x 8 architectures; design floor 6.151e-10, procedure resolution 5e-06
- Unpaired one-way ANOVA (unpaired): F=83.19 on (7, 16) df, p=0.0000
- Effect size: 0.2462 nats = 367 training steps (30.6% of budget), fitted over [800, 1200]

## Spread against training length (common architecture set)

| Steps | Architectures | Spread | Seed std | Spread in seed-noise units | Best |
|---|---|---|---|---|---|
| 400 | 8 | 0.5457 | 0.0286 | 19.09 | `mtot_D6_H896_E16_A6_W288` |
| 800 | 8 | 0.3244 | 0.0165 | 19.71 | `mtot_D6_H896_E16_A6_W288` |
| 1200 | 8 | 0.2462 | 0.0148 | 16.59 | `mtot_D6_H896_E16_A6_W288` |

Spread falls 2.22x over 400 -> 1200 steps (1.65x per doubling of training). Seed noise falls with it (19.09 -> 16.59 in seed-noise units), so the effect does not wash out relative to noise at these lengths.

The same architecture is best at every length: **yes**.

## Structural signal against training length (common architecture set)

The same joint fit at every length, on a **pre-specified** model (`hidden_size`, `depth`). The model is fixed across lengths, so agreement here is a statement about the model rather than about which predictors each length happened to pick.

| Steps | Architectures | R2 | Model p | First gene | beta | p |
|---|---|---|---|---|---|---|
| 400 | 8 | 0.927 | 0.0024 | `hidden_size` | -0.863 | 0.0145 |
| 800 | 8 | 0.936 | 0.00185 | `hidden_size` | -0.944 | 0.00075 |
| 1200 | 8 | 0.953 | 0.0011 | `hidden_size` | -0.959 | 0.00035 |

The model is the same at every length by construction, so the question this table answers is whether *it* holds at each one — and the coefficient p-values below the model do not depend on a per-length selection at all.

## Screening validity: does a short run rank like a long one?

| From | To | n | Spearman | Best at short | Best at long | Winner kept |
|---|---|---|---|---|---|---|
| 400 | 800 | 8 | 0.95 | `mtot_D6_H896_E16_A6_W288` | `mtot_D6_H896_E16_A6_W288` | yes |
| 800 | 1200 | 8 | 0.98 | `mtot_D6_H896_E16_A6_W288` | `mtot_D6_H896_E16_A6_W288` | yes |

## Which axis moves with loss (exploratory)

| Axis | Spearman | p | Range | Ratio |
|---|---|---|---|---|
| total parameters | +0.595 | 0.1323 | 75M .. 81.1M | 1.08x |
| active parameters | +0.214 | 0.6191 | 18M .. 81.1M | 4.51x |
| active / total parameters | +0.214 | 0.6191 | 0.2319 .. 1 | 4.31x |
| contracted FLOPs/token | +0.238 | 0.5821 | 139M .. 586M | 4.20x |
| dense-runtime FLOPs/token | +0.000 | 1.0000 | 497M .. 623M | 1.25x |
| measured throughput | -0.143 | 0.7520 | 4.01k .. 6.5k | 1.62x |
| depth | +0.229 | 0.5631 | 5 .. 14 | 2.80x |
| hidden size | -0.861 | 0.0131 | 512 .. 896 | 1.75x |
| attention heads | -0.861 | 0.0131 | 8 .. 14 | 1.75x |
| expert count | -0.194 | 0.6782 | 4 .. 16 | 4.00x |
| active experts | +0.000 | 1.0000 | 2 .. 6 | 3.00x |
| expert width | +0.503 | 0.2022 | 128 .. 1.54k | 12.00x |

*12 axes tested uncorrected; treat any single marginal p-value as a hypothesis, not a result. Exact p-values are enumerated over n! arrangements and are not comparable to a chi-square test.*

**Confounding structure.** Among 12 spread axes, `hidden size` tracks loss most strongly, but it moves with `attention heads` (+1.00), `expert width` (-0.74) at |rho| >= 0.7, so this set cannot separate them.

## Structural genes fitted together

Standardised least squares over 8 architectures: R2 = 0.955, model p = 0.00465 against a permutation null (20000 draws). Coefficients are nats per standard deviation of that gene, holding the others fixed.

| Gene | beta | p | Range |
|---|---|---|---|
| hidden_size | -1.003 | 0.0586 | 512 .. 896 |
| expert_width | -0.068 | 0.9151 | 128 .. 1536 |
| depth | +0.287 | 0.5339 | 5 .. 14 |
| attention_heads | — | — | identical to `hidden_size` (rho=+1.0000), so it carries no separate information |

*Coefficients are standardised and each p-value is two-sided against a permutation null, but predictors are collinear by construction here, so a coefficient surviving the model does not mean the underlying lever is identified. Use a designed set to separate them.; `hidden_size` and `expert_width` move together at rho=-0.74 and cannot be told apart at this size, so their split in the fit is arbitrary.*

A gene can be significant alone and not here, or the reverse. Neither table is a lever ranking: the only design that ranks levers is one that varies them independently by construction.

## Seed-matched contrast against the best architecture

| Candidate | Mean delta | Paired seeds |
|---|---|---|
| `mtot_D8_H832_E6_A2_W576` | +0.0413 | [42, 43, 44] |
| `mtot_D14_H896_E8_A5_W128` | +0.0487 | [42, 43, 44] |
| `mtot_D6_H576_E6_A4_W1536` | +0.1285 | [42, 43, 44] |
| `mtot_D12_H640_E12_A3_W288` | +0.1357 | [42, 43, 44] |
| `mtot_D5_H512_E16_A2_W832` | +0.1768 | [42, 43, 44] |
| `mtot_D14_H576_E4_A4_W896` | +0.2158 | [42, 43, 44] |
| `mtot_D10_H512_E8_A6_W768` | +0.2462 | [42, 43, 44] |

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
