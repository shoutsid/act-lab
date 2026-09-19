# Architecture Zoo Decision

**ARCHITECTURE EFFECT ESTABLISHED - spread 0.2221 nats, blocked permutation (monte-carlo) p<=5e-06.**

Next step: Proceed to evolutionary architecture search (arch-evo-100m-v1), and confirm the finalists on seeds never used in screening first so the selection coupling is removed. The crossed factorial attributes it to `hidden_size` (-0.1617 nats, p=5e-06), with `attention_heads` real but smaller; spend the search budget there. This design measured one training length. The spread does fall with training - 2.22x over 400 -> 1200 steps on a matched design, 1.65x per doubling - but seed noise falls with it, so the ordering and the signal-to-noise did not move over that range. Quote any effect size with its length, and confirm the ordering holds at the horizon the search actually trains for.

## Ranked at 1200 steps (mean held-out validation loss)

| Rank | Architecture | Mean val loss | Runs |
|---|---|---|---|
| 1 | `hd2_D10_H896_E16_A6_W128_Hd16` | 2.7600 | 6 |
| 2 | `hd4_D10_H896_E4_A3_W512_Hd16` | 2.7806 | 6 |
| 3 | `hd2_D10_H896_E16_A6_W128_Hd8` | 2.8204 | 6 |
| 4 | `hd4_D10_H896_E4_A3_W512_Hd8` | 2.8307 | 6 |
| 5 | `hd1_D10_H512_E4_A2_W1536_Hd8` | 2.9287 | 6 |
| 6 | `hd1_D10_H512_E4_A2_W1536_Hd16` | 2.9513 | 6 |
| 7 | `hd3_D10_H512_E12_A6_W512_Hd16` | 2.9762 | 6 |
| 8 | `hd3_D10_H512_E12_A6_W512_Hd8` | 2.9821 | 6 |

## Evidence

- Architecture spread (best to worst): 0.2221 nats
- Mean within-architecture seed std: 0.0164 nats (descriptive; a range over architectures is not a test statistic)
- Spread in seed-noise units: 13.56
- Seed-matched test (blocked permutation (monte-carlo)): p=5e-06 over 6 blocks x 8 architectures; design floor 9.384e-24, procedure resolution 5e-06
- Unpaired one-way ANOVA (unpaired): F=151.29 on (7, 40) df, p=0.0000

## Spread against training length (common architecture set)

| Steps | Architectures | Spread | Seed std | Spread in seed-noise units | Best |
|---|---|---|---|---|---|
| 1200 | 8 | 0.2221 | 0.0164 | 13.56 | `hd2_D10_H896_E16_A6_W128_Hd16` |

## Which axis moves with loss (exploratory)

| Axis | Spearman | p | Range | Ratio |
|---|---|---|---|---|
| total parameters | +0.830 | 0.0151 | 76.2M .. 77.7M | 1.02x |
| active parameters | -0.634 | 0.1317 | 46.2M .. 67M | 1.45x |
| active / total parameters | -0.634 | 0.1317 | 0.5948 .. 0.8796 | 1.48x |
| contracted FLOPs/token | -0.634 | 0.1317 | 340M .. 512M | 1.51x |
| dense-runtime FLOPs/token | -0.732 | 0.0687 | 529M .. 568M | 1.07x |
| measured throughput | +0.738 | 0.0458 | 4.18k .. 5.5k | 1.32x |
| depth | — | — | held constant | — |
| hidden size | -0.873 | 0.0714 | 512 .. 896 | 1.75x |
| attention heads | -0.218 | 0.9000 | 8 .. 16 | 2.00x |
| expert count | -0.180 | 0.8238 | 4 .. 16 | 4.00x |
| active experts | -0.026 | 0.9262 | 2 .. 6 | 3.00x |
| expert width | +0.540 | 0.1429 | 128 .. 1.54k | 12.00x |

*12 axes tested uncorrected; treat any single marginal p-value as a hypothesis, not a result. Exact p-values are enumerated over n! arrangements and are not comparable to a chi-square test.*

**Confounding structure.** Among 11 spread axes, `hidden size` tracks loss most strongly, but it moves with `total parameters` (-0.89), `active parameters` (+0.89), `active / total parameters` (+0.89), `contracted FLOPs/token` (+0.89), `dense-runtime FLOPs/token` (+0.89), `measured throughput` (-0.76), `expert width` (-0.71) at |rho| >= 0.7, so this set cannot separate them.

## Structural genes fitted together

Standardised least squares over 8 architectures: R2 = 0.947, model p = 0.0195 against a permutation null (20000 draws). Coefficients are nats per standard deviation of that gene, holding the others fixed.

| Gene | beta | p | Range |
|---|---|---|---|
| hidden_size | -1.073 | 0.0131 | 512 .. 896 |
| expert_width | -0.176 | 0.7379 | 128 .. 1536 |
| attention_heads | -0.138 | 0.6279 | 8 .. 16 |

*Coefficients are standardised and each p-value is two-sided against a permutation null, but predictors are collinear by construction here, so a coefficient surviving the model does not mean the underlying lever is identified. Use a designed set to separate them.; `hidden_size` and `expert_width` move together at rho=-0.71 and cannot be told apart at this size, so their split in the fit is arbitrary.*

A gene can be significant alone and not here, or the reverse. Neither table is a lever ranking: the only design that ranks levers is one that varies them independently by construction.

## Paired contrast: `attention_heads` varied with everything else held fixed

24 pairs at levels [8, 16], so the within-pair difference is a contrast between candidates that differ in nothing else. That is a stronger guarantee than a matched band: the pair is its own control.

| Quantity | Value |
|---|---|
| Mean within-pair difference | -0.0234 nats |
| Exact sign-permutation p | 0.01155 (monte-carlo, 16777216 sign vectors) |
| Procedure floor | 1.192e-07 |
| Paired t (reference) | 0.01004 |
| Sign consistency | 17 negative, 7 positive |

Per-pair differences: -0.0334, +0.0628, -0.0008, -0.0149, -0.0309, -0.0183, +0.0233, +0.0117, +0.0015, +0.0447, +0.0053, +0.0494 (+12 more)

A significant contrast with an *inconsistent* sign across pairs means one or two pairs carry the result, which the per-pair list above shows rather than hides.

## Paired level split: `attention_heads` at each level of `hidden_size`

The pooled paired contrast above is an average over these levels, so it can carry one level or cancel two. The levels disagree in sign, so the pooled contrast is an average of opposite effects and neither level's number can be read off it.

| Level | Mean within-pair difference | p | Units | Floor | Signs |
|---|---|---|---|---|---|
| hidden_size=512 | +0.0084 nats | 0.373 | 12 | 0.0004883 | 5 negative, 7 positive |
| hidden_size=896 | -0.0553 nats | 0.0004883 | 12 | 0.0004883 | 12 negative, 0 positive |

Agreement in sign across levels is a description, not a test: it says the pooled number is not hiding a reversal. Whether the levels actually differ is the interaction test on the crossed factorial above, and each level's own p is bounded by its own floor.

## Crossed factorial: which gene is doing the work

`hidden_size` x `attention_heads`, both at two levels, every cell populated on every one of its 6 seeds. Unlike the marginal associations above, each main effect here is an average over both levels of the other gene, so the two are separable even when the zoo correlates them.

| hidden_size \ attention_heads | 8 | 16 |
|---|---|---|
| **hidden_size=512** | 2.9554 | 2.9638 |
| **hidden_size=896** | 2.8255 | 2.7703 |

| Gene | Main effect | p | Method | Assignments | Procedure floor |
|---|---|---|---|---|---|
| `hidden_size` | -0.1617 nats | 5e-06 | monte-carlo | 2176782336 | **at floor** — 5e-06 (Monte-Carlo, raise `--samples`) |
| `attention_heads` | -0.0234 nats | 0.0154 | monte-carlo | 2176782336 | 5e-06 (Monte-Carlo, raise `--samples`) |

`hidden_size` sits on its procedure floor — the assignment space is past what can be enumerated, so the test fell to Monte-Carlo and its p is bounded by the draw count rather than by the evidence — raise `--samples` before quoting it. Such a p-value says *this separation is the largest the design can express*, not how significant the effect would be at a larger size.

The effect *size* is the interpretable quantity here (`main effect` in nats, and its step-equivalent below); the p-value is bounded by the design rather than by the evidence.

- `hidden_size` main effect by level of the other gene: 8: -0.1299, 16: -0.1935
- `attention_heads` main effect by level of the other gene: 512: +0.0084, 896: -0.0553

Interaction: +0.0636 nats.

| test | null | p |
|---|---|---|
| re-split within blocks | no main effect *and* no interaction (conservative) | 0.2615 |
| seed-blocked sign | seeds exchangeable, symmetric contrasts | 0.03125 (procedure floor 0.03125) |
| paired t (reference) | normal contrasts | 0.0008955 |

The seed-blocked test is **at its floor**: 0.03125 is the most extreme p 6 seeds can express, so every seed agreeing is the strongest that test can say. Resolving the interaction further needs more seeds, not a different test; each extra seed costs ~1.1 GPU-hours.

Per-seed interaction: +0.0383, +0.0730, +0.0598, +0.0754, +0.0399, +0.0953 — every seed agrees on the sign, which is what the seed-blocked test measures.

The re-split test assumes no main effect either, so it is conservative for the interaction alone and a large value there does not rule an interaction out; the seed-blocked test is the better-calibrated one and its floor is 2/2^seeds. Treat a consistent sign across every seed with a large re-split p as a lead that more seeds would settle, not as evidence against.

## Pre-registered rule: is the head effect level-dependent?

**HEAD EFFECT IS REAL AT WIDE WIDTH**

The level dependence is real: more attention heads help at wide residual-stream width and do not at narrow. A search over this family must mutate `attention_heads` *jointly* with `hidden_size` - the free lever is only free at wide widths, so "add heads" is a rule with a precondition rather than a global one.

| Condition | Value | Met |
|---|---|---|
| hidden_size significant, negative (p < 0.01) | -0.1617 nats, p=5e-06 | yes |
| seed-blocked interaction p < 0.05 | p=0.03125 (floor 0.03125) | yes |
| every seed contrast the same sign | +0.0383, +0.0730, +0.0598, +0.0754, +0.0399, +0.0953 | yes |
| interaction nats > 0 (heads help more at wide width) | +0.0636 nats | yes |
| the wide level's paired head effect < 0 | hidden_size=896: -0.0553 nats, p=0.0004883 (floor 0.0004883, 12 units) | yes |

The pooled head contrast clears the threshold on a sign-inconsistent set of pairs, so it is a systematic average of disagreeing units rather than a uniform head effect - the level split above is what says which levels produce it, and it is the reason that split is reported rather than inferred from this average. **The branch is reading-dependent**: on the same rows, with only the loss field substituted, the rule lands on `INTERACTION ABSENT - THE THREE-SEED AGREEMENT WAS SEED LUCK` under `mean_train_loss`. Its condition `seed-blocked interaction p < 0.05` stops holding (p=0.0625 (floor 0.03125)). Its condition `every seed contrast the same sign` stops holding (-0.0050, +0.0635, +0.0306, +0.0598, +0.0312, +0.0652). The design, the seeds and the declared thresholds are unchanged, so the branch to quote is the pre-registered one - held-out validation loss - and the reading that moves it is named here rather than left to a reader who tries the other.

The branch is a threshold on a loss, so it was re-read on the other losses these ledgers carry - the same rows with only the loss field substituted, which is a second reading of one design rather than a second experiment:

| Loss read | Branch |
|---|---|
| `validation_loss` - held-out validation loss (the registered reading) | HEAD EFFECT IS REAL AT WIDE WIDTH |
| `final_train_loss` - training loss at the end of training | HEAD EFFECT IS REAL AT WIDE WIDTH |
| `mean_train_loss` - mean training loss over the trajectory | **INTERACTION ABSENT - THE THREE-SEED AGREEMENT WAS SEED LUCK** |

Only the loss field changed; the design, the seeds and the declared thresholds did not.

## Seed-matched contrast against the best architecture

| Candidate | Mean delta | Paired seeds |
|---|---|---|
| `hd4_D10_H896_E4_A3_W512_Hd16` | +0.0206 | [48, 49, 50, 51, 52, 53] |
| `hd2_D10_H896_E16_A6_W128_Hd8` | +0.0604 | [48, 49, 50, 51, 52, 53] |
| `hd4_D10_H896_E4_A3_W512_Hd8` | +0.0707 | [48, 49, 50, 51, 52, 53] |
| `hd1_D10_H512_E4_A2_W1536_Hd8` | +0.1687 | [48, 49, 50, 51, 52, 53] |
| `hd1_D10_H512_E4_A2_W1536_Hd16` | +0.1913 | [48, 49, 50, 51, 52, 53] |
| `hd3_D10_H512_E12_A6_W512_Hd16` | +0.2162 | [48, 49, 50, 51, 52, 53] |
| `hd3_D10_H512_E12_A6_W512_Hd8` | +0.2221 | [48, 49, 50, 51, 52, 53] |

## Seeds per architecture

```json
{
  "hd1_D10_H512_E4_A2_W1536_Hd16": [
    48,
    49,
    50,
    51,
    52,
    53
  ],
  "hd1_D10_H512_E4_A2_W1536_Hd8": [
    48,
    49,
    50,
    51,
    52,
    53
  ],
  "hd2_D10_H896_E16_A6_W128_Hd16": [
    48,
    49,
    50,
    51,
    52,
    53
  ],
  "hd2_D10_H896_E16_A6_W128_Hd8": [
    48,
    49,
    50,
    51,
    52,
    53
  ],
  "hd3_D10_H512_E12_A6_W512_Hd16": [
    48,
    49,
    50,
    51,
    52,
    53
  ],
  "hd3_D10_H512_E12_A6_W512_Hd8": [
    48,
    49,
    50,
    51,
    52,
    53
  ],
  "hd4_D10_H896_E4_A3_W512_Hd16": [
    48,
    49,
    50,
    51,
    52,
    53
  ],
  "hd4_D10_H896_E4_A3_W512_Hd8": [
    48,
    49,
    50,
    51,
    52,
    53
  ]
}
```
