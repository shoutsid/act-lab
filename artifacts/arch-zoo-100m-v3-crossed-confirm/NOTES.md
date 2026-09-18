# arch-zoo-100m-v3-crossed-confirm — Replicating the Attribution on Fresh Seeds

**Design:** `crossed`, same factors `depth × hidden_size`, 2 replicates (8
candidates), total parameters matched to 78M ±5%, plan `flat` at 1200 steps,
seeds **45/46/47**, context 1024, batch 2.

## What it replicates, and why the seeds are the point

`arch-zoo-100m-v3-crossed` (seeds 42/43/44) established both main effects at the
exact enumeration's floor, `p = 4.29e-05`:

| gene | main effect | ~training steps |
|---|---|---|
| `hidden_size` (896 − 512) | −0.1836 nats | ~274 (22.8% of budget) |
| `depth` (D14 − D5) | +0.0783 nats | ~117 (9.7%) |

but left the **interaction unresolved**: +0.0701 nats, and the three per-seed
contrasts `+0.0819, +0.0840, +0.0443` all agree in sign.

That agreement is the interesting part. The reason the interaction reads as
non-significant is not that the evidence is weak — it is that **the test is
floor-limited**:

| test | 3 seeds | 6 seeds |
|---|---|---|
| seed-blocked sign test, floor | 2/2³ = 0.250 | 2/2⁶ = **0.031** |
| re-split within blocks (conservative) | 0.127 | — |

A sign test on the seed contrasts cannot report anything below 0.25 with three
seeds, so `p = 0.25` with every seed agreeing is the *most extreme possible*
result at this seed count. Adding seeds is the only way to resolve it, and it is
also the only way to get an independent replication of the attribution.

## Design integrity

The candidate list was verified **byte-identical** to `arch-zoo-100m-v3-crossed`
by dry run before the hours were committed (sha of the genome set:
`8a5c025a9e6c7d83`), and the crossing is re-reported as balanced
(`rho = +0.000`, depth 5 ×4 / 14 ×4, hidden 512 ×4 / 896 ×4). A replication across
a *different* architecture set would measure the set; this measures the seeds.

Fresh seeds are also the first time the crossed attribution is tested on seeds
never used to design or screen anything, so it doubles as the fresh-seed
confirmation the phase's decision rule asks for.

## Pre-registered decision rule

Written before the numbers exist. The re-split enumeration is
6!^(2×6) = 2.2e9 > the 1e6 cap, so it falls to Monte-Carlo at resolution 5e-06;
the seed-blocked test is enumerated exactly (2⁶ = 64 sign vectors).

Two consequences of pooling that are worth knowing before reading the output, both
verified on a synthetic six-seed copy of v3 so that nothing about the analysis
path is discovered after the hours are spent:

- **The main-effect test stops being exact.** Pooling takes the assignment count
  from 46,656 to 6¹² = 2.18e9, past the 1e6 cap, so both main effects fall to
  Monte-Carlo. Their p is then bounded by `--samples` (5e-06 at the default
  200,000) rather than by the design. That is *below* the pre-registered 0.01, so
  the rule is decidable either way, but it must be reported as a resolution rather
  than as the effect's significance.
- **The seed-blocked interaction floor drops to exactly 0.03125**, which is what
  the arm was bought for. The pre-registered interaction threshold is `< 0.05`, so
  a floor-limited result *can* now clear it — at three seeds it could not, which
  was the whole reason for the second arm.

| outcome | condition | reading |
|---|---|---|
| **attribution replicates, interaction resolves** | on the pooled 6-seed design: both main effects `p < 0.01` with the same signs and `\|hidden\| ≥ 1.5 × \|depth\|`; **and** seed-blocked interaction `p < 0.05` with all six seed contrasts the same sign | the depth penalty is level-dependent. A search must mutate depth and width **jointly**, because a mutation that is good at one width is not good at the other — the search space is not separable in these two genes |
| **attribution replicates, interaction absent** | main effects as above; interaction has **mixed signs across seeds** | the two genes are **additively separable** at this scale. This is a materially better outcome for a search than an interaction: independence means a search can optimise depth and width one at a time instead of needing to cover their joint space, which is a large reduction in the search problem |
| **attribution fails to replicate** | either main effect is non-significant, or a sign flips, or `\|hidden\| < 1.5 × \|depth\|` | a correction to §2c of the phase findings. The attribution would then be a property of seeds 42/43/44 rather than of the design, and the phase's headline reading would have to be withdrawn |
| **inconclusive** | main effects replicate but the interaction is non-significant **and** not sign-consistent, while the seed contrasts straddle zero without agreeing | report the interaction as still open; more seeds than six are needed, and the cost per seed is known (~1.1 GPU-hours) |

Note the asymmetry that makes the *absent* branch a real result rather than a
failure: for a search, separable genes are worth more than a detected
interaction. So both informative branches are actionable, in opposite directions.

## Cost

8 candidates × 3 seeds × 1200 steps = 24 runs. At v3's measured 3.28 GPU-hours
for the same 24 runs, this projects to **~3.3 GPU-hours**, and it is the last
compute this phase needs: the effect is established, the lever is identified, the
horizon is measured, and the only open statistical question is the interaction.

Running total after this arm: 3.60 (v1) + ~3.2 (v2) + 0.2 (mixer) + 3.25
(horizon) + 3.28 (v3) + ~3.3 ≈ **16.8 GPU-hours** against the 10–20 cap.

## How to analyse it

Independently, as a replication, and then pooled for power:

```
PYTHONPATH=src python scripts/arch_zoo_analyze.py \
  --experiment-id arch-zoo-100m-v3-crossed-confirm --samples 200000 --write

PYTHONPATH=src python scripts/arch_zoo_analyze.py \
  --experiment-id arch-zoo-100m-v3-crossed \
  --also arch-zoo-100m-v3-crossed-confirm --samples 200000 --write
```

The first is the replication — if the main effects hold there with the same signs,
the attribution is not a seed artifact. The second is the powered interaction
test, and it writes to the **primary** directory so v3's committed `decision.md`
becomes the six-seed record.

Both commands also take `--replication-rule`, which evaluates the rule table above
from the same outcome the decision renders and states which branch the result
lands in, with the evidence for each condition beside it. So the branch is read
from the pre-registered thresholds rather than by eye once the numbers exist —
which is the failure mode a table of four outcomes invites. Its arithmetic is the
reason this arm exists at all: the seed-blocked floor is `2/2ⁿ`, i.e. **0.25 at
three seeds, 0.0625 at five, 0.031 at six**, against a declared threshold of 0.05.
Five seeds could not have decided the interaction however consistent the contrasts
were; six can. The rule's thresholds live in `REPLICATION_RULE`, so a reader can
check the transcription against this file rather than trusting it.

## Result

**Complete.** 24 runs, 3.26 GPU-hours, and the pre-registered rule lands on the
decisive branch once pooled with v3:

| view | hidden_size | depth | interaction |
|---|---|---|---|
| fresh seeds 45–47 alone | −0.1851 | +0.0865 | sign-consistent, floor-limited at 0.25 |
| pooled six seeds (48 runs) | **−0.1843** | **+0.0824** | **+0.0827**, seed-blocked p = 0.031 (its floor) |

So the attribution is not a property of seeds 42/43/44 — the magnitudes survive to
within 0.005 nats and the ranking is unchanged (the same
`xlohi_D5_H896_E16_A2_W384` wins both arms, and the pooled spread is 0.3018 nats
at p <= 5e-06 over six blocks) — and the interaction **resolves**: all six contrasts
agree in sign (+0.0819, +0.0840, +0.0443, +0.0970, +0.0691, +0.1200), so at six
seeds the seed-blocked test's 2/2⁶ = 0.031 floor finally sits below the declared
0.05. The depth penalty is level-dependent: +0.1238 nats at H=512 against +0.0410
at H=896, a 3.0× gap against the 2.6× the three-seed arm showed.

The arm's own reading, applied by `--replication-rule` rather than by eye, is
**REPLICATES, INTERACTION RESOLVES**; run on the three-seed arm alone it correctly
returns *inconclusive* with the 0.25 floor as the reason, which is the branch
arithmetic that motivated the arm. The consequences for the search program are in
§2c and §9 of `ARCH_SEARCH_FINDINGS.md`: `depth` and `width` are not separable and
must be mutated **jointly**.

Records: this directory's `decision.md` is the fresh-seed replication; the pooled
six-seed record is `../arch-zoo-100m-v3-crossed/decision.md` and `ranked.json`, as
the pre-registration specified, with `analysis.log` holding both runs' output.

## Status

Complete. Pre-registered above before any training data existed; the candidate list
was verified byte-identical to the v3 arm before the hours were spent.
