# Architecture Zoo Decision

**NOT ESTABLISHED - spread 0.0404 nats is above the materiality floor but blocked permutation (exact) gives p=0.139, so it is not separable from initialization noise at 3 seeds, worth 68 training steps (5.6% of the 1200-step budget). This is not evidence that architecture does not matter; it is evidence that this experiment cannot yet tell.**

Next step: Do not build the full evolutionary search. Resolve the effect size first: re-run the finalists on seeds never used in screening, which removes the selection coupling and raises power, and test a matched-budget structural design so any spread is attributable to one axis. Note that widening with module-type genes is not an option here - measured at 100M/context 1024, GDN costs 4-7x attention and exceeds an 8 GB card at 6 of 8 layers.

## Ranked at 1200 steps (mean held-out validation loss)

| Rank | Architecture | Mean val loss | Runs |
|---|---|---|---|
| 1 | `narrow_experts_D8_H896_E12_A4_W256` | 2.7492 | 3 |
| 2 | `shallow_wide_D6_H896_E8_A2_W896` | 2.7724 | 3 |
| 3 | `narrow_experts_D10_H768_E12_A4_W288` | 2.7896 | 3 |

## Evidence

- Architecture spread (best to worst): 0.0404 nats
- Mean within-architecture seed std: 0.0179 nats (descriptive; a range over architectures is not a test statistic)
- Spread in seed-noise units: 2.26
- Seed-matched test (blocked permutation (exact)): p=0.1389 over 3 blocks x 3 architectures; design floor 0.02778, procedure resolution 0.00463
- Unpaired one-way ANOVA (unpaired): F=2.84 on (2, 6) df, p=0.1353
- Effect size: 0.0404 nats = 68 training steps (5.6% of budget), fitted over [800, 1200]

Caveat: 13 architectures were screened at shorter lengths than these 3, and promotion used those runs, so this test is mildly anti-conservative. Confirming on seeds never used in screening removes it.

## Spread against training length (common architecture set)

| Steps | Architectures | Spread | Best |
|---|---|---|---|
| 400 | 3 | 0.1500 | `shallow_wide_D6_H896_E8_A2_W896` |
| 800 | 3 | 0.0584 | `narrow_experts_D8_H896_E12_A4_W256` |
| 1200 | 3 | 0.0404 | `narrow_experts_D8_H896_E12_A4_W256` |

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

## Seed-matched contrast against the best architecture

| Candidate | Mean delta | Paired seeds |
|---|---|---|
| `shallow_wide_D6_H896_E8_A2_W896` | +0.0232 | [42, 43, 44] |
| `narrow_experts_D10_H768_E12_A4_W288` | +0.0404 | [42, 43, 44] |

## Seeds per architecture

```json
{
  "narrow_experts_D10_H768_E12_A4_W288": [
    42,
    43,
    44
  ],
  "narrow_experts_D8_H896_E12_A4_W256": [
    42,
    43,
    44
  ],
  "shallow_wide_D6_H896_E8_A2_W896": [
    42,
    43,
    44
  ]
}
```
