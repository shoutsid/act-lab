# Architecture Zoo — Screening Results

> **Status: superseded.** The screens below and the decision they support are not
> supported by the evidence in this directory. `SCREENING_AUDIT.md` records the five
> defects, each independently sufficient to invalidate the comparison, and the reason
> the synthetic table ranks initialization luck rather than architecture: its 11
> candidates were trained on **uniform random tokens** at context 512, where every
> recorded loss sits between 0.43% and 0.85% above the entropy floor
> `ln(8192) = 9.0109` (the audit rounds that to "0.4% to 0.8%"). The brief asked for
> context 1024/2048.
>
> The numbers below reproduce `zoo_results.json` exactly - all 11 losses, the 0.0376
> range, the population std dev of 0.0102 and the floor. It is the *reading* of them
> that does not survive. The two real-data losses (3.5865, 3.6657) have no artifact
> behind them at all: that sweep wrote results only on completion and was
> interrupted.
>
> Kept for historical reproducibility. The superseding experiment is
> `arch-zoo-100m-v1`, and the audit's own summary is that the v0 results should not be
> cited as a negative result about architecture.

## Experiment Setup

- **Target**: 100M parameters (±30% tolerance)
- **Candidates**: 11 architectures varying depth, hidden size, expert count, active experts
- **Training**: 400 steps, batch=2, context=1024, learning_rate=3e-4
- **Seeds**: 1 (42) for screening
- **Data**: TinyStories (real text)

## Architecture Zoo

| Name | Total Params | Active Params | Depth | Hidden | Experts | Active |
|---|---|---|---|---|---|---|
| D8_H768_E8_A2 | 100.7M | 44.1M | 8 | 768 | 8 | 2 |
| D10_H768_E6_A2 | 100.7M | 53.5M | 10 | 768 | 6 | 2 |
| D8_H640_E12_A2 | 97.1M | 31.5M | 8 | 640 | 12 | 2 |
| D6_H768_E12_A2 | 105.4M | 34.7M | 6 | 768 | 12 | 2 |
| D10_H640_E8_A2 | 87.2M | 38.1M | 10 | 640 | 8 | 2 |
| D8_H768_E6_A3 | 81.8M | 53.5M | 8 | 768 | 6 | 3 |
| D10_H640_E12_A3 | 120.0M | 46.3M | 10 | 640 | 12 | 3 |
| D10_H512_E12_A2 | 77.7M | 25.2M | 10 | 512 | 12 | 2 |
| D6_H768_E8_A3 | 77.1M | 41.7M | 6 | 768 | 8 | 3 |
| D10_H768_E4_A3 | 77.1M | 65.3M | 10 | 768 | 4 | 3 |
| D8_H640_E8_A3 | 70.8M | 38.1M | 8 | 640 | 8 | 3 |

## Results

### Synthetic Random Data (200 steps, context=512)

| Rank | Name | Loss | Total Params | Active Params |
|---|---|---|---|---|
| 1 | D8_H768_E6_A3 | 9.0496 | 81.8M | 53.5M |
| 2 | D8_H640_E12_A2 | 9.0622 | 97.1M | 31.5M |
| 3 | D10_H768_E4_A3 | 9.0627 | 77.1M | 65.3M |
| 4 | D10_H512_E12_A2 | 9.0636 | 77.7M | 25.2M |
| 5 | D10_H768_E6_A2 | 9.0662 | 100.7M | 53.5M |
| ... | ... | ... | ... | ... |
| 11 | D8_H768_E8_A2 | 9.0873 | 100.7M | 44.1M |

**Range**: 0.0376 (0.4% of mean)
**Std dev**: 0.0102 (0.1% of mean)

### Real TinyStories Data (partial, 2 runs completed)

| Name | Loss | Total Params | Active Params |
|---|---|---|---|
| D8_H768_E8_A2 | **3.5865** | 100.7M | 44.1M |
| D10_H768_E6_A2 | 3.6657 | 100.7M | 53.5M |

**Note**: The real-data ordering is **OPPOSITE** to synthetic-data ordering. On synthetic data, D8_H768_E6_A3 won. On real data, D8_H768_E8_A2 won (with only 2 data points).

## Analysis

### Is There a Meaningful Architecture Effect?

**Synthetic data**: Yes, small but detectable (range = 3.7× std dev).

**Real data**: Inconclusive. Only 2 runs completed, and they show the opposite ordering from synthetic data.

### Key Observations

1. **Architecture effect is small**: Even on synthetic data, the range is only 0.4% of the mean loss.

2. **Effect does not transfer across data distributions**: The winner on synthetic data is the loser on real data (and vice versa).

3. **Active parameters may matter more than total**: On synthetic data, the best architecture (D8_H768_E6_A3) had fewer total params but more active params.

4. **GPU memory limits**: The largest candidates (120M params) would OOM on 8GB GPU, limiting practical search space.

## Decision

### Should We Proceed with Evolutionary Search?

**NO — not yet.**

Reasons:
1. The architecture effect is small (0.4%) and may not be statistically significant
2. The effect does not transfer across data distributions (synthetic vs real)
3. We only tested 11 candidates (out of hundreds of possible configurations)
4. We only ran 1 seed (no variance estimate)
5. The best architecture on real data has the most total params, not the most active params — contradicting the synthetic-data hypothesis

### Recommended Next Steps

1. **Run full zoo on real data with 2-3 seeds** (~10-15 GPU-hours on T4)
   - This will give us a definitive answer on whether architecture matters
   - If the effect is <0.1%, it's not worth searching for

2. **If effect is real (>0.1% with 2+ seeds)**:
   - Proceed to evolutionary search
   - Focus on the active-parameters hypothesis
   - Search over depth, expert count, and active-expert ratio

3. **If effect is negligible (<0.1%)**:
   - Architecture choice doesn't matter at this scale
   - Focus on other research directions (data, training duration, etc.)

## Files Generated

- `artifacts/arch-zoo/zoo_definition.json` — Zoo specification
- `artifacts/arch-zoo/zoo_results.json` — Synthetic data results
- `artifacts/arch-zoo-real/zoo_definition.json` — Real data zoo spec
- `src/act/search/architecture_zoo.py` — Zoo generator
- `src/act/search/architecture_eval.py` — Evaluator
- `scripts/arch_zoo_screen.py` — Screening script
