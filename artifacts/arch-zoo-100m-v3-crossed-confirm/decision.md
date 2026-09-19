# Architecture Zoo Decision

**ARCHITECTURE EFFECT ESTABLISHED - spread 0.2978 nats, blocked permutation (monte-carlo) p=1.5e-05.**

Next step: Proceed to evolutionary architecture search (arch-evo-100m-v1), and confirm the finalists on seeds never used in screening first so the selection coupling is removed. The crossed factorial attributes it to `hidden_size` (-0.1851 nats, p=4.287e-05), with `depth` real but smaller; spend the search budget there. This design measured one training length. The spread does fall with training - 2.22x over 400 -> 1200 steps on a matched design, 1.65x per doubling - but seed noise falls with it, so the ordering and the signal-to-noise did not move over that range. Quote any effect size with its length, and confirm the ordering holds at the horizon the search actually trains for.

## Ranked at 1200 steps (mean held-out validation loss)

| Rank | Architecture | Mean val loss | Runs |
|---|---|---|---|
| 1 | `xlohi_D5_H896_E16_A2_W384` | 2.7467 | 3 |
| 2 | `xlohi_D5_H896_E4_A4_W1536` | 2.7610 | 3 |
| 3 | `xhihi_D14_H896_E8_A6_W128` | 2.7807 | 3 |
| 4 | `xhihi_D14_H896_E4_A2_W288` | 2.8047 | 3 |
| 5 | `xlolo_D5_H512_E12_A2_W1152` | 2.8901 | 3 |
| 6 | `xlolo_D5_H512_E16_A6_W832` | 2.8925 | 3 |
| 7 | `xhilo_D14_H512_E4_A4_W1024` | 3.0065 | 3 |
| 8 | `xhilo_D14_H512_E16_A2_W256` | 3.0445 | 3 |

## Evidence

- Architecture spread (best to worst): 0.2978 nats
- Mean within-architecture seed std: 0.0229 nats (descriptive; a range over architectures is not a test statistic)
- Spread in seed-noise units: 13.01
- Seed-matched test (blocked permutation (monte-carlo)): p=1.5e-05 over 3 blocks x 8 architectures; design floor 6.151e-10, procedure resolution 5e-06
- Unpaired one-way ANOVA (unpaired): F=43.50 on (7, 16) df, p=0.0000

## Spread against training length (common architecture set)

| Steps | Architectures | Spread | Seed std | Spread in seed-noise units | Best |
|---|---|---|---|---|---|
| 1200 | 8 | 0.2978 | 0.0229 | 13.01 | `xlohi_D5_H896_E16_A2_W384` |

## Which axis moves with loss (exploratory)

| Axis | Spearman | p | Range | Ratio |
|---|---|---|---|---|
| total parameters | -0.571 | 0.1511 | 77.6M .. 81.3M | 1.05x |
| active parameters | -0.262 | 0.5364 | 21.3M .. 78.5M | 3.69x |
| active / total parameters | -0.204 | 0.6380 | 0.265 .. 1 | 3.77x |
| contracted FLOPs/token | -0.024 | 0.9768 | 159M .. 584M | 3.67x |
| dense-runtime FLOPs/token | +0.000 | 1.0000 | 497M .. 642M | 1.29x |
| measured throughput | +0.000 | 1.0000 | 3.86k .. 6.29k | 1.63x |
| depth | +0.436 | 0.2000 | 5 .. 14 | 2.80x |
| hidden size | -0.873 | 0.0714 | 512 .. 896 | 1.75x |
| attention heads | -0.873 | 0.0714 | 8 .. 14 | 1.75x |
| expert count | +0.150 | 0.6455 | 4 .. 16 | 4.00x |
| active experts | +0.000 | 0.8810 | 2 .. 6 | 3.00x |
| expert width | -0.095 | 0.8401 | 128 .. 1.54k | 12.00x |

*12 axes tested uncorrected; treat any single marginal p-value as a hypothesis, not a result. Exact p-values are enumerated over n! arrangements and are not comparable to a chi-square test.*

**Confounding structure.** Among 12 spread axes, `hidden size` tracks loss most strongly, but it moves with `attention heads` (+1.00) at |rho| >= 0.7, so this set cannot separate them.

## Structural genes fitted together

Standardised least squares over 8 architectures: R2 = 0.937, model p = 0.00405 against a permutation null (20000 draws). Coefficients are nats per standard deviation of that gene, holding the others fixed.

| Gene | beta | p | Range |
|---|---|---|---|
| hidden_size | -0.865 | 0.02485 | 512 .. 896 |
| depth | +0.421 | 0.3381 | 5 .. 14 |
| expert_count | +0.031 | 0.9443 | 4 .. 16 |
| attention_heads | — | — | identical to `hidden_size` (rho=+1.0000), so it carries no separate information |

*Coefficients are standardised and each p-value is two-sided against a permutation null, but predictors are collinear by construction here, so a coefficient surviving the model does not mean the underlying lever is identified. Use a designed set to separate them.*

A gene can be significant alone and not here, or the reverse. Neither table is a lever ranking: the only design that ranks levers is one that varies them independently by construction.

## Crossed factorial: which gene is doing the work

`depth` x `hidden_size`, both at two levels, every cell populated on every one of its 3 seeds. Unlike the marginal associations above, each main effect here is an average over both levels of the other gene, so the two are separable even when the zoo correlates them.

| depth \ hidden_size | 512 | 896 |
|---|---|---|
| **depth=5** | 2.8913 | 2.7539 |
| **depth=14** | 3.0255 | 2.7927 |

| Gene | Main effect | p | Method | Assignments | Procedure floor |
|---|---|---|---|---|---|
| `depth` | +0.0865 nats | 8.573e-05 | exact | 46656 | 4.287e-05 |
| `hidden_size` | -0.1851 nats | 4.287e-05 | exact | 46656 | **at floor** — 4.287e-05 |

`hidden_size` sits on its procedure floor — the observed re-splitting was matched only by its own mirror, so the exact test over this design's 46656-assignment null cannot report anything smaller. Such a p-value says *this separation is the largest the design can express*, not how significant the effect would be at a larger size.

The effect *size* is the interpretable quantity here (`main effect` in nats, and its step-equivalent below); the p-value is bounded by the design rather than by the evidence.

- `depth` main effect by level of the other gene: 512: +0.1342, 896: +0.0388
- `hidden_size` main effect by level of the other gene: 5: -0.1374, 14: -0.2328

Interaction: +0.0954 nats.

| test | null | p |
|---|---|---|
| re-split within blocks | no main effect *and* no interaction (conservative) | 0.06614 |
| seed-blocked sign | seeds exchangeable, symmetric contrasts | 0.25 (procedure floor 0.25) |
| paired t (reference) | normal contrasts | 0.02294 |

The seed-blocked test is **at its floor**: 0.25 is the most extreme p 3 seeds can express, so every seed agreeing is the strongest that test can say. Resolving the interaction further needs more seeds, not a different test; each extra seed costs ~1.1 GPU-hours.

Per-seed interaction: +0.0970, +0.0691, +0.1200 — every seed agrees on the sign, which is what the seed-blocked test measures.

The re-split test assumes no main effect either, so it is conservative for the interaction alone and a large value there does not rule an interaction out; the seed-blocked test is the better-calibrated one and its floor is 2/2^seeds. Treat a consistent sign across every seed with a large re-split p as a lead that more seeds would settle, not as evidence against.

## Pre-registered rule: does the attribution replicate?

**INCONCLUSIVE - the interaction is still open**

The main effects replicate but the interaction is still open. Before buying more seeds, note whether the threshold was reachable at all: if the seed-blocked test's floor is above the declared 0.05, the arm could not have resolved it however consistent the seed contrasts were, and that is a statement about the design's size rather than about the world.

| Condition | Value | Met |
|---|---|---|
| hidden_size significant, negative | -0.1851 nats, p=4.287e-05 | yes |
| depth significant, positive (shallow wins) | +0.0865 nats, p=8.573e-05 | yes |
| \|hidden\| >= 1.5 x \|depth\| | 0.1851 vs 0.0865 nats | yes |
| seed-blocked interaction p < 0.05 | p=0.25 (floor 0.25) | NO |
| every seed contrast the same sign | +0.0970, +0.0691, +0.1200 | yes |

The seed-blocked test's floor at 3 seeds is 0.25, which is above the declared 0.05 - so this arm could not have resolved the interaction however consistent the seed contrasts were. More seeds raise the floor's denominator; nothing else does. **The branch is reading-dependent**: on the same rows, with only the loss field substituted, the rule lands on `ATTRIBUTION FAILS TO REPLICATE` under `mean_train_loss`. Its condition `\|hidden\| >= 1.5 x \|depth\|` stops holding (0.2265 vs 0.1879 nats). The design, the seeds and the declared thresholds are unchanged, so the branch to quote is the pre-registered one - held-out validation loss - and the reading that moves it is named here rather than left to a reader who tries the other.

The branch is a threshold on a loss, so it was re-read on the other losses these ledgers carry - the same rows with only the loss field substituted, which is a second reading of one design rather than a second experiment:

| Loss read | Branch |
|---|---|
| `validation_loss` - held-out validation loss (the registered reading) | INCONCLUSIVE - the interaction is still open |
| `final_train_loss` - training loss at the end of training | INCONCLUSIVE - the interaction is still open |
| `mean_train_loss` - mean training loss over the trajectory | **ATTRIBUTION FAILS TO REPLICATE** |

Only the loss field changed; the design, the seeds and the declared thresholds did not.

## Seed-matched contrast against the best architecture

| Candidate | Mean delta | Paired seeds |
|---|---|---|
| `xlohi_D5_H896_E4_A4_W1536` | +0.0143 | [45, 46, 47] |
| `xhihi_D14_H896_E8_A6_W128` | +0.0340 | [45, 46, 47] |
| `xhihi_D14_H896_E4_A2_W288` | +0.0580 | [45, 46, 47] |
| `xlolo_D5_H512_E12_A2_W1152` | +0.1434 | [45, 46, 47] |
| `xlolo_D5_H512_E16_A6_W832` | +0.1458 | [45, 46, 47] |
| `xhilo_D14_H512_E4_A4_W1024` | +0.2597 | [45, 46, 47] |
| `xhilo_D14_H512_E16_A2_W256` | +0.2978 | [45, 46, 47] |

## Seeds per architecture

```json
{
  "xlolo_D5_H512_E12_A2_W1152": [
    45,
    46,
    47
  ],
  "xlolo_D5_H512_E16_A6_W832": [
    45,
    46,
    47
  ],
  "xlohi_D5_H896_E16_A2_W384": [
    45,
    46,
    47
  ],
  "xlohi_D5_H896_E4_A4_W1536": [
    45,
    46,
    47
  ],
  "xhilo_D14_H512_E16_A2_W256": [
    45,
    46,
    47
  ],
  "xhilo_D14_H512_E4_A4_W1024": [
    45,
    46,
    47
  ],
  "xhihi_D14_H896_E4_A2_W288": [
    45,
    46,
    47
  ],
  "xhihi_D14_H896_E8_A6_W128": [
    45,
    46,
    47
  ]
}
```
