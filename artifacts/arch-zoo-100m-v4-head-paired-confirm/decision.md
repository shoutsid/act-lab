# Architecture Zoo Decision

**ARCHITECTURE EFFECT ESTABLISHED - spread 0.2412 nats, blocked permutation (monte-carlo) p<=5e-06.**

Next step: Proceed to evolutionary architecture search (arch-evo-100m-v1), and confirm the finalists on seeds never used in screening first so the selection coupling is removed. The crossed factorial attributes it to `hidden_size` (-0.1671 nats, p=4.287e-05); spend the search budget there. This design measured one training length. The spread does fall with training - 2.22x over 400 -> 1200 steps on a matched design, 1.65x per doubling - but seed noise falls with it, so the ordering and the signal-to-noise did not move over that range. Quote any effect size with its length, and confirm the ordering holds at the horizon the search actually trains for.

## Ranked at 1200 steps (mean held-out validation loss)

| Rank | Architecture | Mean val loss | Runs |
|---|---|---|---|
| 1 | `hd2_D10_H896_E16_A6_W128_Hd16` | 2.7615 | 3 |
| 2 | `hd4_D10_H896_E4_A3_W512_Hd16` | 2.7737 | 3 |
| 3 | `hd2_D10_H896_E16_A6_W128_Hd8` | 2.8244 | 3 |
| 4 | `hd4_D10_H896_E4_A3_W512_Hd8` | 2.8395 | 3 |
| 5 | `hd1_D10_H512_E4_A2_W1536_Hd8` | 2.9252 | 3 |
| 6 | `hd1_D10_H512_E4_A2_W1536_Hd16` | 2.9583 | 3 |
| 7 | `hd3_D10_H512_E12_A6_W512_Hd16` | 2.9814 | 3 |
| 8 | `hd3_D10_H512_E12_A6_W512_Hd8` | 3.0027 | 3 |

## Evidence

- Architecture spread (best to worst): 0.2412 nats
- Mean within-architecture seed std: 0.0151 nats (descriptive; a range over architectures is not a test statistic)
- Spread in seed-noise units: 16.01
- Seed-matched test (blocked permutation (monte-carlo)): p=5e-06 over 3 blocks x 8 architectures; design floor 6.151e-10, procedure resolution 5e-06
- Unpaired one-way ANOVA (unpaired): F=86.39 on (7, 16) df, p=0.0000

## Spread against training length (common architecture set)

| Steps | Architectures | Spread | Seed std | Spread in seed-noise units | Best |
|---|---|---|---|---|---|
| 1200 | 8 | 0.2412 | 0.0151 | 16.01 | `hd2_D10_H896_E16_A6_W128_Hd16` |

## Which axis moves with loss (exploratory)

| Axis | Spearman | p | Range | Ratio |
|---|---|---|---|---|
| total parameters | +0.830 | 0.0151 | 76.2M .. 77.7M | 1.02x |
| active parameters | -0.634 | 0.1317 | 46.2M .. 67M | 1.45x |
| active / total parameters | -0.634 | 0.1317 | 0.5948 .. 0.8796 | 1.48x |
| contracted FLOPs/token | -0.634 | 0.1317 | 340M .. 512M | 1.51x |
| dense-runtime FLOPs/token | -0.732 | 0.0687 | 529M .. 568M | 1.07x |
| measured throughput | +0.738 | 0.0458 | 4.16k .. 5.4k | 1.30x |
| depth | — | — | held constant | — |
| hidden size | -0.873 | 0.0714 | 512 .. 896 | 1.75x |
| attention heads | -0.218 | 0.9000 | 8 .. 16 | 2.00x |
| expert count | -0.180 | 0.8238 | 4 .. 16 | 4.00x |
| active experts | -0.026 | 0.9262 | 2 .. 6 | 3.00x |
| expert width | +0.540 | 0.1429 | 128 .. 1.54k | 12.00x |

*12 axes tested uncorrected; treat any single marginal p-value as a hypothesis, not a result. Exact p-values are enumerated over n! arrangements and are not comparable to a chi-square test.*

**Confounding structure.** Among 11 spread axes, `hidden size` tracks loss most strongly, but it moves with `total parameters` (-0.89), `active parameters` (+0.89), `active / total parameters` (+0.89), `contracted FLOPs/token` (+0.89), `dense-runtime FLOPs/token` (+0.89), `measured throughput` (-0.76), `expert width` (-0.71) at |rho| >= 0.7, so this set cannot separate them.

## Structural genes fitted together

Standardised least squares over 8 architectures: R2 = 0.935, model p = 0.01655 against a permutation null (20000 draws). Coefficients are nats per standard deviation of that gene, holding the others fixed.

| Gene | beta | p | Range |
|---|---|---|---|
| hidden_size | -1.091 | 0.0112 | 512 .. 896 |
| expert_width | -0.227 | 0.6776 | 128 .. 1536 |
| attention_heads | -0.164 | 0.6279 | 8 .. 16 |

*Coefficients are standardised and each p-value is two-sided against a permutation null, but predictors are collinear by construction here, so a coefficient surviving the model does not mean the underlying lever is identified. Use a designed set to separate them.; `hidden_size` and `expert_width` move together at rho=-0.71 and cannot be told apart at this size, so their split in the fit is arbitrary.*

A gene can be significant alone and not here, or the reverse. Neither table is a lever ranking: the only design that ranks levers is one that varies them independently by construction.

## Paired contrast: `attention_heads` varied with everything else held fixed

12 pairs at levels [8, 16], so the within-pair difference is a contrast between candidates that differ in nothing else. That is a stronger guarantee than a matched band: the pair is its own control.

| Quantity | Value |
|---|---|
| Mean within-pair difference | -0.0292 nats |
| Exact sign-permutation p | 0.04688 (exact, 4096 sign vectors) |
| Procedure floor | 0.0004883 |
| Paired t (reference) | 0.04294 |
| Sign consistency | 9 negative, 3 positive |

Per-pair differences: -0.0149, -0.0309, -0.0183, +0.0447, +0.0053, +0.0494, -0.0564, -0.0591, -0.0732, -0.0645, -0.0462, -0.0865

A significant contrast with an *inconsistent* sign across pairs means one or two pairs carry the result, which the per-pair list above shows rather than hides.

## Paired level split: `attention_heads` at each level of `hidden_size`

The pooled paired contrast above is an average over these levels, so it can carry one level or cancel two. The levels disagree in sign, so the pooled contrast is an average of opposite effects and neither level's number can be read off it.

| Level | Mean within-pair difference | p | Units | Floor | Signs |
|---|---|---|---|---|---|
| hidden_size=512 | +0.0059 nats | 0.6562 | 6 | 0.03125 | 3 negative, 3 positive |
| hidden_size=896 | -0.0643 nats | 0.03125 | 6 | 0.03125 | 6 negative, 0 positive |

Agreement in sign across levels is a description, not a test: it says the pooled number is not hiding a reversal. Whether the levels actually differ is the interaction test on the crossed factorial above, and each level's own p is bounded by its own floor.

## Crossed factorial: which gene is doing the work

`hidden_size` x `attention_heads`, both at two levels, every cell populated on every one of its 3 seeds. Unlike the marginal associations above, each main effect here is an average over both levels of the other gene, so the two are separable even when the zoo correlates them.

| hidden_size \ attention_heads | 8 | 16 |
|---|---|---|
| **hidden_size=512** | 2.9640 | 2.9698 |
| **hidden_size=896** | 2.8319 | 2.7676 |

| Gene | Main effect | p | Method | Assignments | Procedure floor |
|---|---|---|---|---|---|
| `hidden_size` | -0.1671 nats | 4.287e-05 | exact | 46656 | **at floor** — 4.287e-05 |
| `attention_heads` | -0.0292 nats | 0.06649 | exact | 46656 | 4.287e-05 |

`hidden_size` sits on its procedure floor — the observed re-splitting was matched only by its own mirror, so the exact test over this design's 46656-assignment null cannot report anything smaller. Such a p-value says *this separation is the largest the design can express*, not how significant the effect would be at a larger size.

The effect *size* is the interpretable quantity here (`main effect` in nats, and its step-equivalent below); the p-value is bounded by the design rather than by the evidence.

- `hidden_size` main effect by level of the other gene: 8: -0.1320, 16: -0.2022
- `attention_heads` main effect by level of the other gene: 512: +0.0059, 896: -0.0643

Interaction: +0.0702 nats.

| test | null | p |
|---|---|---|
| re-split within blocks | no main effect *and* no interaction (conservative) | 0.4046 |
| seed-blocked sign | seeds exchangeable, symmetric contrasts | 0.25 (procedure floor 0.25) |
| paired t (reference) | normal contrasts | 0.04951 |

The seed-blocked test is **at its floor**: 0.25 is the most extreme p 3 seeds can express, so every seed agreeing is the strongest that test can say. Resolving the interaction further needs more seeds, not a different test; each extra seed costs ~1.1 GPU-hours.

Per-seed interaction: +0.0754, +0.0399, +0.0953 — every seed agrees on the sign, which is what the seed-blocked test measures.

The re-split test assumes no main effect either, so it is conservative for the interaction alone and a large value there does not rule an interaction out; the seed-blocked test is the better-calibrated one and its floor is 2/2^seeds. Treat a consistent sign across every seed with a large re-split p as a lead that more seeds would settle, not as evidence against.

## Seed-matched contrast against the best architecture

| Candidate | Mean delta | Paired seeds |
|---|---|---|
| `hd4_D10_H896_E4_A3_W512_Hd16` | +0.0122 | [51, 52, 53] |
| `hd2_D10_H896_E16_A6_W128_Hd8` | +0.0629 | [51, 52, 53] |
| `hd4_D10_H896_E4_A3_W512_Hd8` | +0.0779 | [51, 52, 53] |
| `hd1_D10_H512_E4_A2_W1536_Hd8` | +0.1637 | [51, 52, 53] |
| `hd1_D10_H512_E4_A2_W1536_Hd16` | +0.1968 | [51, 52, 53] |
| `hd3_D10_H512_E12_A6_W512_Hd16` | +0.2199 | [51, 52, 53] |
| `hd3_D10_H512_E12_A6_W512_Hd8` | +0.2412 | [51, 52, 53] |

## Seeds per architecture

```json
{
  "hd1_D10_H512_E4_A2_W1536_Hd16": [
    51,
    52,
    53
  ],
  "hd1_D10_H512_E4_A2_W1536_Hd8": [
    51,
    52,
    53
  ],
  "hd2_D10_H896_E16_A6_W128_Hd16": [
    51,
    52,
    53
  ],
  "hd2_D10_H896_E16_A6_W128_Hd8": [
    51,
    52,
    53
  ],
  "hd3_D10_H512_E12_A6_W512_Hd16": [
    51,
    52,
    53
  ],
  "hd3_D10_H512_E12_A6_W512_Hd8": [
    51,
    52,
    53
  ],
  "hd4_D10_H896_E4_A3_W512_Hd16": [
    51,
    52,
    53
  ],
  "hd4_D10_H896_E4_A3_W512_Hd8": [
    51,
    52,
    53
  ]
}
```
