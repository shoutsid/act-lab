# Architecture Zoo Decision

**ARCHITECTURE EFFECT ESTABLISHED - spread 0.1059 nats, blocked permutation (monte-carlo) p=0.00025, worth 953 training steps (7.9% of the 12000-step budget).**

Next step: Proceed to evolutionary architecture search (arch-evo-100m-v1), and confirm the finalists on seeds never used in screening first so the selection coupling is removed. The crossed factorial attributes it to `hidden_size` (-0.0645 nats, p=0.001543), with `depth` real but smaller; spend the search budget there.

## Ranked at 12000 steps (mean held-out validation loss)

| Rank | Architecture | Mean val loss | Runs |
|---|---|---|---|
| 1 | `xlohi_D5_H896_E4_A4_W1536` | 1.6158 | 2 |
| 2 | `xlohi_D5_H896_E16_A2_W384` | 1.6416 | 2 |
| 3 | `xhihi_D14_H896_E4_A2_W288` | 1.6503 | 2 |
| 4 | `xhihi_D14_H896_E8_A6_W128` | 1.6579 | 2 |
| 5 | `xlolo_D5_H512_E16_A6_W832` | 1.6941 | 2 |
| 6 | `xlolo_D5_H512_E12_A2_W1152` | 1.7014 | 2 |
| 7 | `xhilo_D14_H512_E16_A2_W256` | 1.7067 | 2 |
| 8 | `xhilo_D14_H512_E4_A4_W1024` | 1.7216 | 2 |

## Evidence

- Architecture spread (best to worst): 0.1059 nats
- Mean within-architecture seed std: 0.0056 nats (descriptive; a range over architectures is not a test statistic)
- Spread in seed-noise units: 18.88
- Seed-matched test (blocked permutation (monte-carlo)): p=0.00025 over 2 blocks x 8 architectures; design floor 2.48e-05, procedure resolution 5e-06
- Unpaired one-way ANOVA (unpaired): F=52.67 on (7, 8) df, p=0.0000
- Effect size: 0.1059 nats = 953 training steps (7.9% of budget), fitted over [1200, 12000]

## Spread against training length (common architecture set)

| Steps | Architectures | Spread | Seed std | Spread in seed-noise units | Best |
|---|---|---|---|---|---|
| 1200 | 8 | 0.3085 | 0.0118 | 26.16 | `xlohi_D5_H896_E16_A2_W384` |
| 12000 | 8 | 0.1059 | 0.0056 | 18.88 | `xlohi_D5_H896_E4_A4_W1536` |

Spread falls 2.91x over 1200 -> 12000 steps (1.38x per doubling of training). Seed noise falls with it (26.16 -> 18.88 in seed-noise units), so the effect does not wash out relative to noise at these lengths.

The same architecture is best at every length: **NO**.

## Precision offset against another experiment's ledgers

Per-candidate validation loss at 1200 steps, seed-matched, against `arch-zoo-100m-v3-crossed`: **-0.0030 nats** mean (-0.0257 to +0.0104) over 8 architectures. A gap this size is the precision's own contribution to every loss above, not an architecture effect.

| Architecture | Seeds | This arm | Reference | Delta |
|---|---|---|---|---|
| `xhilo_D14_H512_E16_A2_W256` | [42, 43] | 3.0506 | 3.0763 | -0.0257 |
| `xlohi_D5_H896_E16_A2_W384` | [42, 43] | 2.7421 | 2.7491 | -0.0070 |
| `xhihi_D14_H896_E4_A2_W288` | [42, 43] | 2.8056 | 2.8114 | -0.0059 |
| `xlolo_D5_H512_E16_A6_W832` | [42, 43] | 2.9164 | 2.9215 | -0.0051 |
| `xhihi_D14_H896_E8_A6_W128` | [42, 43] | 2.7882 | 2.7907 | -0.0025 |
| `xlohi_D5_H896_E4_A4_W1536` | [42, 43] | 2.7718 | 2.7677 | +0.0041 |
| `xhilo_D14_H512_E4_A4_W1024` | [42, 43] | 3.0021 | 2.9943 | +0.0078 |
| `xlolo_D5_H512_E12_A2_W1152` | [42, 43] | 2.9082 | 2.8978 | +0.0104 |

## Structural signal against training length (common architecture set)

The same joint fit at each length. If the model is already significant at the shortest length, a search can screen there and confirm finalists at full length.

| Steps | Architectures | R2 | Model p | Leading gene | beta | p |
|---|---|---|---|---|---|---|
| 1200 | 8 | 0.950 | 0.0057 | `hidden_size` | -0.894 | 0.0196 |
| 12000 | 8 | 0.947 | 0.01335 | `hidden_size` | -0.901 | 0.0193 |

Same leading gene at every fitted length: **yes** (`hidden_size`), with the same sign: **yes**.

Caveat: the leading gene is *selected* from the spread axes at each length, so its own p-value is uncorrected for that selection and is not comparable to the designed factorial's exact test. It is a consistency check, not a replacement for it. Pass `--signal-genes` to hold the model fixed instead.

## Screening validity: does a short run rank like a long one?

| From | To | n | Spearman | Best at short | Best at long | Winner kept |
|---|---|---|---|---|---|---|
| 1200 | 12000 | 8 | 0.90 | `xlohi_D5_H896_E16_A2_W384` | `xlohi_D5_H896_E4_A4_W1536` | NO |

## Which axis moves with loss (exploratory)

| Axis | Spearman | p | Range | Ratio |
|---|---|---|---|---|
| total parameters | -0.571 | 0.1511 | 77.6M .. 81.3M | 1.05x |
| active parameters | -0.310 | 0.4618 | 21.3M .. 78.5M | 3.69x |
| active / total parameters | -0.228 | 0.5962 | 0.265 .. 1 | 3.77x |
| contracted FLOPs/token | -0.071 | 0.8820 | 159M .. 584M | 3.67x |
| dense-runtime FLOPs/token | +0.000 | 1.0000 | 497M .. 642M | 1.29x |
| measured throughput | +0.000 | 1.0000 | 9.77k .. 17.1k | 1.75x |
| depth | +0.436 | 0.2000 | 5 .. 14 | 2.80x |
| hidden size | -0.873 | 0.0714 | 512 .. 896 | 1.75x |
| attention heads | -0.873 | 0.0714 | 8 .. 14 | 1.75x |
| expert count | +0.150 | 0.6455 | 4 .. 16 | 4.00x |
| active experts | +0.000 | 0.8810 | 2 .. 6 | 3.00x |
| expert width | -0.095 | 0.8401 | 128 .. 1.54k | 12.00x |

*12 axes tested uncorrected; treat any single marginal p-value as a hypothesis, not a result. Exact p-values are enumerated over n! arrangements and are not comparable to a chi-square test.*

**Confounding structure.** Among 12 spread axes, `hidden size` tracks loss most strongly, but it moves with `attention heads` (+1.00) at |rho| >= 0.7, so this set cannot separate them.

## Structural genes fitted together

Standardised least squares over 8 architectures: R2 = 0.947, model p = 0.01335 against a permutation null (20000 draws). Coefficients are nats per standard deviation of that gene, holding the others fixed.

| Gene | beta | p | Range |
|---|---|---|---|
| hidden_size | -0.901 | 0.0193 | 512 .. 896 |
| depth | +0.323 | 0.4657 | 5 .. 14 |
| expert_count | +0.063 | 0.8979 | 4 .. 16 |
| attention_heads | — | — | identical to `hidden_size` (rho=+1.0000), so it carries no separate information |

*Coefficients are standardised and each p-value is two-sided against a permutation null, but predictors are collinear by construction here, so a coefficient surviving the model does not mean the underlying lever is identified. Use a designed set to separate them.*

A gene can be significant alone and not here, or the reverse. Neither table is a lever ranking: the only design that ranks levers is one that varies them independently by construction.

## Crossed factorial: which gene is doing the work

`depth` x `hidden_size`, both at two levels, every cell populated on every one of its 2 seeds. Unlike the marginal associations above, each main effect here is an average over both levels of the other gene, so the two are separable even when the zoo correlates them.

| depth \ hidden_size | 512 | 896 |
|---|---|---|
| **depth=5** | 1.6977 | 1.6287 |
| **depth=14** | 1.7141 | 1.6541 |

| Gene | Main effect | p | Method | Assignments | Procedure floor |
|---|---|---|---|---|---|
| `depth` | +0.0209 nats | 0.003086 | exact | 1296 | 0.001543 |
| `hidden_size` | -0.0645 nats | 0.001543 | exact | 1296 | **at floor** — 0.001543 |

`hidden_size` sits on its procedure floor — the observed re-splitting was matched only by its own mirror, so the exact test over this design's 1296-assignment null cannot report anything smaller. Such a p-value says *this separation is the largest the design can express*, not how significant the effect would be at a larger size.

The effect *size* is the interpretable quantity here (`main effect` in nats, and its step-equivalent below); the p-value is bounded by the design rather than by the evidence.

- `depth` main effect by level of the other gene: 512: +0.0164, 896: +0.0254
- `hidden_size` main effect by level of the other gene: 5: -0.0691, 14: -0.0600

Interaction: -0.0090 nats.

| test | null | p |
|---|---|---|
| re-split within blocks | no main effect *and* no interaction (conservative) | 0.5972 |
| seed-blocked sign | seeds exchangeable, symmetric contrasts | 0.5 (procedure floor 0.5) |
| paired t (reference) | normal contrasts | 0.3514 |

The seed-blocked test is **at its floor**: 0.5 is the most extreme p 2 seeds can express, so every seed agreeing is the strongest that test can say. Resolving the interaction further needs more seeds, not a different test; each extra seed costs ~1.1 GPU-hours.

Per-seed interaction: -0.0146, -0.0035 — every seed agrees on the sign, which is what the seed-blocked test measures.

The re-split test assumes no main effect either, so it is conservative for the interaction alone and a large value there does not rule an interaction out; the seed-blocked test is the better-calibrated one and its floor is 2/2^seeds. Treat a consistent sign across every seed with a large re-split p as a lead that more seeds would settle, not as evidence against.

## Pre-registered rule: does the depth sign survive the horizon?

**SHALLOW-WINS PERSISTS AT THIS HORIZON**

The depth sign is horizon-stable to this length: shallow still beats deep at matched total parameters, so the record's 'spend at low depth' stands at this scale - quoted with its training length, as every effect size must be.

| Condition | Value | Met |
|---|---|---|
| depth main effect significant (exact re-split p < 0.05) | p=0.003086 (floor 0.001543) | yes |
| depth main effect > 0 (deep - shallow; shallow wins) | +0.0209 nats at 12000 steps | yes |
| secondary: width main effect < 0 (reported, does not gate) | -0.0645 nats, p=0.001543 | yes |

The branch does not depend on which loss it is read on: re-reading the same rule over `final_train_loss`, `mean_train_loss` lands on the same branch, with only the loss field substituted.

The branch is a threshold on a loss, so it was re-read on the other losses these ledgers carry - the same rows with only the loss field substituted, which is a second reading of one design rather than a second experiment:

| Loss read | Branch |
|---|---|
| `validation_loss` - held-out validation loss (the registered reading) | SHALLOW-WINS PERSISTS AT THIS HORIZON |
| `final_train_loss` - training loss at the end of training | SHALLOW-WINS PERSISTS AT THIS HORIZON |
| `mean_train_loss` - mean training loss over the trajectory | SHALLOW-WINS PERSISTS AT THIS HORIZON |

Only the loss field changed; the design, the seeds and the declared thresholds did not.

## Seed-matched contrast against the best architecture

| Candidate | Mean delta | Paired seeds |
|---|---|---|
| `xlohi_D5_H896_E16_A2_W384` | +0.0259 | [42, 43] |
| `xhihi_D14_H896_E4_A2_W288` | +0.0345 | [42, 43] |
| `xhihi_D14_H896_E8_A6_W128` | +0.0422 | [42, 43] |
| `xlolo_D5_H512_E16_A6_W832` | +0.0783 | [42, 43] |
| `xlolo_D5_H512_E12_A2_W1152` | +0.0856 | [42, 43] |
| `xhilo_D14_H512_E16_A2_W256` | +0.0909 | [42, 43] |
| `xhilo_D14_H512_E4_A4_W1024` | +0.1059 | [42, 43] |

## Seeds per architecture

```json
{
  "xlolo_D5_H512_E12_A2_W1152": [
    42,
    43
  ],
  "xlolo_D5_H512_E16_A6_W832": [
    42,
    43
  ],
  "xlohi_D5_H896_E16_A2_W384": [
    42,
    43
  ],
  "xlohi_D5_H896_E4_A4_W1536": [
    42,
    43
  ],
  "xhilo_D14_H512_E16_A2_W256": [
    42,
    43
  ],
  "xhilo_D14_H512_E4_A4_W1024": [
    42,
    43
  ],
  "xhihi_D14_H896_E4_A2_W288": [
    42,
    43
  ],
  "xhihi_D14_H896_E8_A6_W128": [
    42,
    43
  ]
}
```
