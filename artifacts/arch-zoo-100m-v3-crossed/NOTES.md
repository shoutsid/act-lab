# arch-zoo-100m-v3-crossed — Separating Depth from Residual-Stream Width

**Design:** `crossed`, factors `depth × hidden_size`, 2 levels each, 2 replicates
(8 candidates), total parameters matched to 78M ±5%, plan `flat` at 1200 steps,
seeds 42/43/44, context 1024, batch 2.

## Why this experiment exists

`arch-zoo-100m-v2` **established** that architecture matters at a matched budget —
0.2416 nats of spread against 0.0164 nats of seed noise, *p = 0.00023* — and
**refuted** the active-parameters hypothesis while doing it (ρ = +0.214 over a
×4.51 range, p = 0.62).

What it could not do is say **which gene** carries the effect, and that is not
insufficient seeding — it is a property of the design:

- v2's strongest association is hidden size (ρ = −0.861), but hidden size and
  attention heads are perfectly collinear there (`heads = hidden / 64`).
- Cross-correlating v2's own axes shows hidden size is nearly independent of
  active parameters (+0.11) but **strongly entangled with expert width (−0.74)**:
  with the budget pinned, widening the residual stream forces experts narrower.
  So "wide hidden" and "narrow experts" are one finding seen from both ends.
- Depth is entangled too: total-parameter matching forces the corner cells, and
  at 78M the H=896 corners exist only at D5 and D8 while H=512 exists at D10 and
  D14. A naive 2×2 factorial there simply swaps one confound for another.

Both failed designs were checked rather than assumed: the four corners of
`hidden × width` at 78M have **fixed depths** (14 / 10 / 8 / 5), so crossing those
two genes confounds depth completely. Crossing `depth × hidden_size` is what the
space permits, because expert width and count absorb a hidden-size change at a
fixed depth without limit.

## The design

| candidate | total | active | depth | hidden | E | A | W |
|---|---|---|---|---|---|---|---|
| `xlolo_D5_H512_E12_A2_W1152` | 80.3M | 21.3M | 5 | 512 | 12 | 2 | 1152 |
| `xlolo_D5_H512_E16_A6_W832` | 77.6M | 35.0M | 5 | 512 | 16 | 6 | 832 |
| `xlohi_D5_H896_E16_A2_W384` | 78.5M | 30.4M | 5 | 896 | 16 | 2 | 384 |
| `xlohi_D5_H896_E4_A4_W1536` | 78.5M | **78.5M (dense)** | 5 | 896 | 4 | 4 | 1536 |
| `xhilo_D14_H512_E16_A2_W256` | 77.7M | 26.3M | 14 | 512 | 16 | 2 | 256 |
| `xhilo_D14_H512_E4_A4_W1024` | 77.6M | **77.6M (dense)** | 14 | 512 | 4 | 4 | 1024 |
| `xhihi_D14_H896_E4_A2_W288` | 81.3M | 66.8M | 14 | 896 | 4 | 2 | 288 |
| `xhihi_D14_H896_E8_A6_W128` | 78.1M | 71.7M | 14 | 896 | 8 | 6 | 128 |

| property | value | vs v2 |
|---|---|---|
| ρ(depth, hidden size) | **0.000** (balanced 2×2) | v2 could not control depth at all |
| ρ(hidden size, expert width) | **−0.22** | −0.74 in v2 |
| total parameters | 77.6M–81.3M (×1.05) | ×1.08 in v2 |
| active parameters | 21.3M–78.5M (×3.69) | ×4.51 in v2 |
| min pairwise structural distance | 0.132 | 0.256 in v2 |
| depth levels | 5 and 14, four candidates each | 5–14, uncontrolled |
| hidden levels | 512 and 896, four candidates each | 512–896, uncontrolled |

The crossing is reported by the driver at launch (`rho = +0.000 (balanced)`) and
computed by `crossing_balance`, which reads structure off the genomes rather than
parsing names. The report is a check on the design, not a restatement of it: a
test unbalances a crossing by hand and requires the report to say so.

**The genes that absorb are not independent, and this design cannot make them so.**
Expert width is the budget's shock absorber, so "hidden size matters here" still
implies "expert capacity does not", and the two remain one finding. What the
crossing buys is that the two genes it *names* are orthogonal, which is the
difference between an attribution that is confounded by depth and one that is not.

## Pre-registered outcomes

Written before the numbers exist.

| outcome | condition | reading |
|---|---|---|
| **residual-stream share is the lever** | hidden size's association survives with |ρ| ≥ 0.7 and the two hidden levels separate in the expected direction at *both* depths | v2's finding is attributable, not a depth artifact; the search should spend its budget in the residual stream rather than in expert capacity |
| **depth is the lever** | depth dominates and hidden size's association collapses toward 0 | v2's effect was depth expressed through hidden size — a correction to the phase's headline reading, and the search should spend budget on depth |
| **neither separates** | no axis reaches significance at 8 candidates × 3 seeds | the established spread is real but its cause is not identifiable at this scale, and per §23 the honest next step is scale, not a wider search |

Both a depth and a hidden-size effect being significant is a legitimate fourth
outcome and the most informative one: it says the two levers are real and
separable, which is what a search needs to trade them against each other.

## Cost

8 candidates × 3 seeds × 1200 steps = 24 runs. At v2's measured 0.31–0.51 s/step
this projects to **~3.2 GPU-hours**. Running total for the phase after this arm:
v1 3.60 + v2 ~3.2 + mixer ~0.2 + horizon ~3.6 + this ~3.2 ≈ **13.8 GPU-hours**,
inside the 10–20 cap.

## How to analyse it

```
PYTHONPATH=src python scripts/arch_zoo_analyze.py \
  --experiment-id arch-zoo-100m-v3-crossed \
  --also arch-zoo-100m-v3-crossed-confirm \
  --samples 200000 --replication-rule --write
```

`--also` pools this arm with `arch-zoo-100m-v3-crossed-confirm`, which is what the
committed `decision.md` in this directory records: the confirmation re-ran the same
byte-identical candidates on fresh seeds, so the two row sets pool to six seed blocks.
`--replication-rule` renders the pre-registered branch. Without both, the command
reproduces the three-seed record this arm closed on before its confirmation ran - the
runbook below this line predates the rule, and the record does not.

The axis table reports the associations; the confounding block reports what else
moves with whichever axis leads. Because the crossed pair is orthogonal, a leading
axis that is *not* flagged as entangled with its partner is the attribution
evidence — and the absences matter as much as the hits here.

## Result — the pre-registered fourth outcome

**Both levers are real, they are separable, and residual-stream width is the
bigger one.** 24 runs, 3.28 GPU-hours (the estimate was 3.2) — the pre-registered
"both significant" outcome, explicitly named as the most informative one.

Cell means (held-out validation loss, 6 runs per cell):

| depth \ hidden_size | 512 | 896 |
|---|---|---|
| **D5** | 2.9096 | 2.7611 |
| **D14** | 3.0229 | 2.8043 |

| Gene | Main effect | p | Equivalent training steps |
|---|---|---|---|
| `hidden_size` (896 − 512) | **−0.1836 nats** | 4.29e-05 (exact, 46,656 assignments) | ~274 (22.8% of budget) |
| `depth` (D14 − D5) | **+0.0783 nats** | 4.29e-05 (exact) | ~117 (9.7% of budget) |
| interaction | +0.0701 nats | 0.24 — **not resolved** | ~104 (8.7%) |

The step figures use the pooled curve's 6.71e-04 nats/step slope from
`effect_in_compute_units`. Both main effects sit on the enumeration's floor
(2/46,656 = 4.287e-05 exactly, the smallest an exact two-sided test over this
null can report), so the p-values here say *the observed separation is the
largest this design can express* rather than how significant the effect would be
at a larger size. The floor does shrink with more blocks — each extra seed
multiplies the assignment count — so the limit is this arm's size, not the test.
The `effect size` column is the interpretable one and it is not floor-bound.

`decision.md` reports the floor beside every p for this reason: a floor-bound
p-value read as an effect size is the same error the spread-decay rule made in the
horizon arm, where a statistic at its resolution floor was read as the absence of
a signal.

### What this settles

- **v2's confound is resolved.** v2 could not separate hidden size from expert
  width (−0.74 collinear), so "wide residual stream matters" and "expert capacity
  does not" were one finding seen from two ends. Crossing depth against hidden
  size gives each an orthogonal main effect: **wide is worth 0.184 nats, and it
  is not a depth artifact.**
- **Depth is a real but smaller lever, and it runs the unintuitive way: shallow
  wins.** D5 beats D14 at *both* hidden levels with total parameters matched to
  ±5% and hidden size fixed. A search that assumes "deeper is better" would spend
  budget in the wrong place.
- **v2's refutation of active parameters survives a much wider spread.** Active
  parameters span 21.3M–78.5M (×3.69) inside a design whose best architecture is
  30.4M active, and its *dense* 78.5M counterpart is second. Across the arm, the
  dense-runtime and active-parameter axes are the two that do **not** track loss.

### Robustness checks on the attribution

Two are available on this design, and both are run because an attribution claim
is only worth what its checks are:

- **Not a holdout-split artifact.** Re-running the identical factorial on
  *training* loss keeps both main effects, same signs, same floor: depth +0.0827,
  hidden_size −0.1807 against −0.1836. Architectures could in principle differ in
  how fast they overfit rather than in how well they model, and this rules that
  out.
- **Not attributable to one candidate — but only by construction, not by a
  check.** The design has exactly 2 candidates per cell, so removing any one
  leaves its cell lopsided and the factorial is no longer balanced. That is a
  real limit of a 2-replicate design, and it is reported as a refusal rather
  than worked around: `crossed_contrast` requires balanced arms on every seed and
  says so. v2's arm, with 8 candidates spread over 8 distinct configurations,
  could run leave-one-out; this one cannot. A 3-replicate design would.

Both behaviours are pinned by tests so they cannot silently change.

### What it does not settle

The interaction. The depth penalty looks level-dependent — +0.1133 nats at
H=512 against +0.0433 at H=896, a 2.6x difference — which would mean narrow
residual streams punish depth much harder. But p = 0.24 against a null that
assumes no main effect at all, so this is a **lead, not a result**. It is the
cheapest open question in the phase: 3 seeds × 4 architectures is already enough
to see a 0.07-nat interaction, and the arm that would resolve it is 24 runs.

Also unresolved by construction: expert width absorbs the budget at fixed depth,
so "hidden size matters" still implies "expert capacity does not." The crossed
pair is orthogonal; the *absorbers* are not, and no design at this scale can make
them so. That limitation is a property of the parameter budget, not of the search.

## Status

**Complete — and this directory's `decision.md` is now the six-seed pooled
record.** Three fresh seeds (45/46/47) were run on a byte-identical candidate set
by `arch-zoo-100m-v3-crossed-confirm`, exactly as its pre-registration specified,
and the pooled analysis was written *here* rather than into the confirmation
directory so the crossed arm's record carries all six seeds. The decision shows:

| pooled 6 seeds (48 runs) | value |
|---|---|
| `hidden_size` | −0.1843 nats, p = 5e-06 |
| `depth` | +0.0824 nats, p = 5e-06 |
| interaction | +0.0827 nats, seed-blocked p = 0.031 (its 2/2⁶ floor), all six contrasts agreeing |
| spread | 0.3018 nats, 15.3× the seed noise |

So the attribution on this page replicates on seeds never used to design or screen
anything, and the interaction this arm could only call a lead is **resolved**: the
depth penalty depends on the level of `hidden_size` (+0.1238 at H=512 against
+0.0410 at H=896), which makes the two genes non-separable for a search. The
fresh-seed replication alone is `../arch-zoo-100m-v3-crossed-confirm/decision.md`,
and both runs' output is in that directory's `analysis.log`.

Decision record: `decision.md` in this directory, which renders the crossed
factorial alongside the marginal associations — the marginal table alone shows
depth and hidden size with similar-looking ρ and cannot separate them, which is
exactly why the factorial section exists.

## The one clause of this attribution that moves when the loss changes

The pre-registered rule is a threshold comparison, and the quantity under the
threshold is a loss, so the record now re-reads the rule over the other losses the
same rows carry — same rows, same seeds, same declared thresholds, only the loss
field substituted. Held-out loss and final training loss both land on *replicates,
interaction resolves*. The trajectory average (`mean_train_loss`) lands on
**attribution fails to replicate**, and the interesting part is *which* condition
did it:

| condition | held-out | trajectory average |
|---|---|---|
| `hidden_size` significant, negative | −0.1843, p = 5e-06 | −0.2255, p = 5e-06 |
| `depth` significant, positive | +0.0824, p = 5e-06 | +0.1814, p = 5e-06 |
| `\|hidden\| ≥ 1.5 × \|depth\|` | 0.1843 vs 0.0824 → **holds** | 0.2255 vs 0.1814 → **1.24×, fails** |
| seed-blocked interaction p < 0.05 | 0.03125 (floor) | 0.03125 (floor) |
| every seed contrast the same sign | all six agree | all six agree |

So nothing about the attribution moved: both main effects keep their signs and sit at
the design's Monte-Carlo floor, and the interaction stays at its 2/2⁶ floor with every
seed contrast agreeing. What moved is the *magnitude ratio*, which is a statement
about end-of-training sizes — and the trajectory average weights the early phase, where
depth's penalty is evidently larger, more heavily (depth grows by 2.20×, width by
1.22×). The clause is a fair thing to have declared; reading it off a loss it was not
declared on is not.

This is not a six-seed accident: the fresh three-seed arm
(`../arch-zoo-100m-v3-crossed-confirm/`), which shares no seeds with this one, narrows
its ratio from 0.1851 vs 0.0865 to 0.2265 vs 0.1879 — **1.21×** against the same
declared 1.5 — and lands on the same alternate branch. Two independent seed sets reaching
the same reading means the effect is a property of the loss, which is why the honest
sentence is that the *ratio* is reading-dependent rather than the attribution.
