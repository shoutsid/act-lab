# Architecture Zoo Decision

**ARCHITECTURE EFFECT ESTABLISHED - spread 0.3018 nats, blocked permutation (monte-carlo) p<=5e-06.**

Next step: Proceed to evolutionary architecture search (arch-evo-100m-v1), and confirm the finalists on seeds never used in screening first so the selection coupling is removed. The crossed factorial attributes it to `hidden_size` (-0.1843 nats, p=5e-06), with `depth` real but smaller; spend the search budget there. This design measured one training length. The spread does fall with training - 2.22x over 400 -> 1200 steps on a matched design, 1.65x per doubling - but seed noise falls with it, so the ordering and the signal-to-noise did not move over that range. Quote any effect size with its length, and confirm the ordering holds at the horizon the search actually trains for.

## Ranked at 1200 steps (mean held-out validation loss)

| Rank | Architecture | Mean val loss | Runs |
|---|---|---|---|
| 1 | `xlohi_D5_H896_E16_A2_W384` | 2.7505 | 6 |
| 2 | `xlohi_D5_H896_E4_A4_W1536` | 2.7644 | 6 |
| 3 | `xhihi_D14_H896_E8_A6_W128` | 2.7893 | 6 |
| 4 | `xhihi_D14_H896_E4_A2_W288` | 2.8077 | 6 |
| 5 | `xlolo_D5_H512_E12_A2_W1152` | 2.8940 | 6 |
| 6 | `xlolo_D5_H512_E16_A6_W832` | 2.9069 | 6 |
| 7 | `xhilo_D14_H512_E4_A4_W1024` | 2.9961 | 6 |
| 8 | `xhilo_D14_H512_E16_A2_W256` | 3.0523 | 6 |

## Evidence

- Architecture spread (best to worst): 0.3018 nats
- Mean within-architecture seed std: 0.0197 nats (descriptive; a range over architectures is not a test statistic)
- Spread in seed-noise units: 15.28
- Seed-matched test (blocked permutation (monte-carlo)): p=5e-06 over 6 blocks x 8 architectures; design floor 9.384e-24, procedure resolution 5e-06
- Unpaired one-way ANOVA (unpaired): F=139.00 on (7, 40) df, p=0.0000

## Spread against training length (common architecture set)

| Steps | Architectures | Spread | Seed std | Spread in seed-noise units | Best |
|---|---|---|---|---|---|
| 1200 | 8 | 0.3018 | 0.0197 | 15.28 | `xlohi_D5_H896_E16_A2_W384` |

## Which axis moves with loss (exploratory)

| Axis | Spearman | p | Range | Ratio |
|---|---|---|---|---|
| total parameters | -0.571 | 0.1511 | 77.6M .. 81.3M | 1.05x |
| active parameters | -0.262 | 0.5364 | 21.3M .. 78.5M | 3.69x |
| active / total parameters | -0.204 | 0.6380 | 0.265 .. 1 | 3.77x |
| contracted FLOPs/token | -0.024 | 0.9768 | 159M .. 584M | 3.67x |
| dense-runtime FLOPs/token | +0.000 | 1.0000 | 497M .. 642M | 1.29x |
| measured throughput | +0.000 | 1.0000 | 3.81k .. 6.32k | 1.66x |
| depth | +0.436 | 0.2000 | 5 .. 14 | 2.80x |
| hidden size | -0.873 | 0.0714 | 512 .. 896 | 1.75x |
| attention heads | -0.873 | 0.0714 | 8 .. 14 | 1.75x |
| expert count | +0.150 | 0.6455 | 4 .. 16 | 4.00x |
| active experts | +0.000 | 0.8810 | 2 .. 6 | 3.00x |
| expert width | -0.095 | 0.8401 | 128 .. 1.54k | 12.00x |

*12 axes tested uncorrected; treat any single marginal p-value as a hypothesis, not a result. Exact p-values are enumerated over n! arrangements and are not comparable to a chi-square test.*

**Confounding structure.** Among 12 spread axes, `hidden size` tracks loss most strongly, but it moves with `attention heads` (+1.00) at |rho| >= 0.7, so this set cannot separate them.

## Structural genes fitted together

Standardised least squares over 8 architectures: R2 = 0.943, model p = 0.0039 against a permutation null (20000 draws). Coefficients are nats per standard deviation of that gene, holding the others fixed.

| Gene | beta | p | Range |
|---|---|---|---|
| hidden_size | -0.855 | 0.02745 | 512 .. 896 |
| depth | +0.425 | 0.3334 | 5 .. 14 |
| expert_count | +0.077 | 0.8733 | 4 .. 16 |
| attention_heads | — | — | identical to `hidden_size` (rho=+1.0000), so it carries no separate information |

*Coefficients are standardised and each p-value is two-sided against a permutation null, but predictors are collinear by construction here, so a coefficient surviving the model does not mean the underlying lever is identified. Use a designed set to separate them.*

A gene can be significant alone and not here, or the reverse. Neither table is a lever ranking: the only design that ranks levers is one that varies them independently by construction.

## Crossed factorial: which gene is doing the work

`depth` x `hidden_size`, both at two levels, every cell populated on every one of its 6 seeds. Unlike the marginal associations above, each main effect here is an average over both levels of the other gene, so the two are separable even when the zoo correlates them.

| depth \ hidden_size | 512 | 896 |
|---|---|---|
| **depth=5** | 2.9004 | 2.7575 |
| **depth=14** | 3.0242 | 2.7985 |

| Gene | Main effect | p | Method | Assignments | Procedure floor |
|---|---|---|---|---|---|
| `depth` | +0.0824 nats | 5e-06 | monte-carlo | 2176782336 | **at floor** — 5e-06 (Monte-Carlo, raise `--samples`) |
| `hidden_size` | -0.1843 nats | 5e-06 | monte-carlo | 2176782336 | **at floor** — 5e-06 (Monte-Carlo, raise `--samples`) |

Both main effects sit on their procedure floor — the assignment space is past what can be enumerated, so the test fell to Monte-Carlo and its p is bounded by the draw count rather than by the evidence — raise `--samples` before quoting it. Such a p-value says *this separation is the largest the design can express*, not how significant the effect would be at a larger size.

The effect *size* is the interpretable quantity here (`main effect` in nats, and its step-equivalent below); the p-value is bounded by the design rather than by the evidence.

- `depth` main effect by level of the other gene: 512: +0.1238, 896: +0.0410
- `hidden_size` main effect by level of the other gene: 5: -0.1430, 14: -0.2257

Interaction: +0.0827 nats.

| test | null | p |
|---|---|---|
| re-split within blocks | no main effect *and* no interaction (conservative) | 0.01494 |
| seed-blocked sign | seeds exchangeable, symmetric contrasts | 0.03125 (procedure floor 0.03125) |
| paired t (reference) | normal contrasts | 0.0005104 |

The seed-blocked test is **at its floor**: 0.03125 is the most extreme p 6 seeds can express, so every seed agreeing is the strongest that test can say. Resolving the interaction further needs more seeds, not a different test; each extra seed costs ~1.1 GPU-hours.

Per-seed interaction: +0.0819, +0.0840, +0.0443, +0.0970, +0.0691, +0.1200 — every seed agrees on the sign, which is what the seed-blocked test measures.

The re-split test assumes no main effect either, so it is conservative for the interaction alone and a large value there does not rule an interaction out; the seed-blocked test is the better-calibrated one and its floor is 2/2^seeds. Treat a consistent sign across every seed with a large re-split p as a lead that more seeds would settle, not as evidence against.

## Pre-registered rule: does the attribution replicate?

**REPLICATES, INTERACTION RESOLVES**

The attribution replicates on seeds never used to design or screen anything, and the depth penalty is level-dependent. A search must mutate depth and width *jointly*: a mutation that is good at one width is not good at the other, so the space is not separable in these two genes and the search has to cover their joint space.

| Condition | Value | Met |
|---|---|---|
| hidden_size significant, negative | -0.1843 nats, p=5e-06 | yes |
| depth significant, positive (shallow wins) | +0.0824 nats, p=5e-06 | yes |
| \|hidden\| >= 1.5 x \|depth\| | 0.1843 vs 0.0824 nats | yes |
| seed-blocked interaction p < 0.05 | p=0.03125 (floor 0.03125) | yes |
| every seed contrast the same sign | +0.0819, +0.0840, +0.0443, +0.0970, +0.0691, +0.1200 | yes |

## Seed-matched contrast against the best architecture

| Candidate | Mean delta | Paired seeds |
|---|---|---|
| `xlohi_D5_H896_E4_A4_W1536` | +0.0138 | [42, 43, 44, 45, 46, 47] |
| `xhihi_D14_H896_E8_A6_W128` | +0.0388 | [42, 43, 44, 45, 46, 47] |
| `xhihi_D14_H896_E4_A2_W288` | +0.0572 | [42, 43, 44, 45, 46, 47] |
| `xlolo_D5_H512_E12_A2_W1152` | +0.1435 | [42, 43, 44, 45, 46, 47] |
| `xlolo_D5_H512_E16_A6_W832` | +0.1563 | [42, 43, 44, 45, 46, 47] |
| `xhilo_D14_H512_E4_A4_W1024` | +0.2455 | [42, 43, 44, 45, 46, 47] |
| `xhilo_D14_H512_E16_A2_W256` | +0.3018 | [42, 43, 44, 45, 46, 47] |

## Seeds per architecture

```json
{
  "xlolo_D5_H512_E12_A2_W1152": [
    42,
    43,
    44,
    45,
    46,
    47
  ],
  "xlolo_D5_H512_E16_A6_W832": [
    42,
    43,
    44,
    45,
    46,
    47
  ],
  "xlohi_D5_H896_E16_A2_W384": [
    42,
    43,
    44,
    45,
    46,
    47
  ],
  "xlohi_D5_H896_E4_A4_W1536": [
    42,
    43,
    44,
    45,
    46,
    47
  ],
  "xhilo_D14_H512_E16_A2_W256": [
    42,
    43,
    44,
    45,
    46,
    47
  ],
  "xhilo_D14_H512_E4_A4_W1024": [
    42,
    43,
    44,
    45,
    46,
    47
  ],
  "xhihi_D14_H896_E4_A2_W288": [
    42,
    43,
    44,
    45,
    46,
    47
  ],
  "xhihi_D14_H896_E8_A6_W128": [
    42,
    43,
    44,
    45,
    46,
    47
  ]
}
```
