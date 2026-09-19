# Architecture Zoo Decision

**NOT ESTABLISHED - spread 0.0463 nats is above the materiality floor but blocked permutation (exact) gives p=0.1111, so it is not separable from initialization noise at 3 seeds. This is not evidence that architecture does not matter; it is evidence that this experiment cannot yet tell.**

Next step: Do not build the full evolutionary search. Resolve the effect size first: re-run the finalists on seeds never used in screening, which removes the selection coupling and raises power, and test a matched-budget structural design so any spread is attributable to one axis. Note that widening with module-type genes is not an option here - measured at 100M/context 1024, GDN costs 4-7x attention and exceeds an 8 GB card at 6 of 8 layers.

## Ranked at 1200 steps (mean held-out validation loss)

| Rank | Architecture | Mean val loss | Runs |
|---|---|---|---|
| 1 | `narrow_experts_D8_H896_E12_A4_W256` | 2.7545 | 3 |
| 2 | `shallow_wide_D6_H896_E8_A2_W896` | 2.7582 | 3 |
| 3 | `narrow_experts_D10_H768_E12_A4_W288` | 2.8007 | 3 |

## Evidence

- Architecture spread (best to worst): 0.0463 nats
- Mean within-architecture seed std: 0.0082 nats (descriptive; a range over architectures is not a test statistic)
- Spread in seed-noise units: 5.64
- Seed-matched test (blocked permutation (exact)): p=0.1111 over 3 blocks x 3 architectures; design floor 0.02778, procedure resolution 0.00463
- Unpaired one-way ANOVA (unpaired): F=22.81 on (2, 6) df, p=0.0016

## Spread against training length (common architecture set)

| Steps | Architectures | Spread | Seed std | Spread in seed-noise units | Best |
|---|---|---|---|---|---|
| 1200 | 3 | 0.0463 | 0.0082 | 5.64 | `narrow_experts_D8_H896_E12_A4_W256` |

## Which axis moves with loss (exploratory)

| Axis | Spearman | p | Range | Ratio |
|---|---|---|---|---|
| total parameters | +0.500 | 1.0000 | 77.2M .. 104M | 1.34x |
| active parameters | -0.500 | 1.0000 | 45.9M .. 47.8M | 1.04x |
| active / total parameters | -0.500 | 1.0000 | 0.4428 .. 0.6195 | 1.40x |
| contracted FLOPs/token | +0.500 | 1.0000 | 342M .. 380M | 1.11x |
| dense-runtime FLOPs/token | +0.500 | 1.0000 | 551M .. 688M | 1.25x |
| measured throughput | -1.000 | 0.3333 | 4.63k .. 5.27k | 1.14x |
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
| `shallow_wide_D6_H896_E8_A2_W896` | +0.0038 | [45, 46, 47] |
| `narrow_experts_D10_H768_E12_A4_W288` | +0.0463 | [45, 46, 47] |

## Seeds per architecture

```json
{
  "narrow_experts_D10_H768_E12_A4_W288": [
    45,
    46,
    47
  ],
  "narrow_experts_D8_H896_E12_A4_W256": [
    45,
    46,
    47
  ],
  "shallow_wide_D6_H896_E8_A2_W896": [
    45,
    46,
    47
  ]
}
```
