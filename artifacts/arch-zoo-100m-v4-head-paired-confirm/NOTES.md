# arch-zoo-100m-v4-head-paired-confirm — Is the Head Effect Level-Dependent, or Seed Luck?

**Design:** `head-paired`, depth **fixed at 10**, `hidden_size` 512 × 896,
`attention_heads` 8 × 16, 4 base configurations (8 candidates), total parameters
matched to 78M ±5%, plan `flat` at 1200 steps, seeds **51/52/53**, context 1024,
batch 2.

## What it replicates, and why the seeds are the point

`arch-zoo-100m-v4-head-paired` (seeds 48/49/50) did the job it was bought for: it
broke the head-count confound *in width's favour*.

| v4 | value |
|---|---|
| head contrast, 12 paired units | −0.0177 nats, exact p = 0.1396 (floor 4.9e-04), signs 8 negative / 4 positive |
| `hidden_size` in the factorial view | −0.1562 nats, p = 4.287e-05, at the 46,656-assignment floor |
| pre-registered branch | **width is the lever** |

What it *left* is a lead, and the reason it could not be more than a lead is this
arm's justification:

| test | 3 seeds | 6 seeds |
|---|---|---|
| seed-blocked interaction floor | 2/2³ = **0.250** | 2/2⁶ = **0.031** |

v4's interaction is +0.0571 nats with **all three seed contrasts agreeing in sign**
(+0.0383, +0.0730, +0.0598) and a paired-t reference at 0.030 — and the seed-blocked
test nevertheless reports exactly 0.25, because 0.25 is the most extreme value three
seeds can express. The three-seed agreement is not weak evidence that failed; it is
evidence that hit the ceiling of its own design. The level split it implies is large
relative to the pooled contrast: **+0.0109 nats at hidden 512 against −0.0462 at
896**, with all six H=896 within-pair differences negative — the most extreme
six-unit outcome there is (2/2⁶ = 0.031), which is only a floor because those six
units share three seeds.

No re-analysis of v4's ledgers can move that: the floor is a property of the seed
count, not of the test chosen. Six seeds put it at 0.031, below the declared 0.05
threshold, and **that is the whole reason this arm exists**. It is the same
arithmetic that bought `arch-zoo-100m-v3-crossed-confirm`, and the same lesson the
phase already wrote down twice: *a pre-registered rule whose threshold sits below
the test's floor cannot be decided by more analysis of the same seeds, only by more
seeds.*

Fresh seeds are also the first test of the head axis on seeds that were never used
to design, screen or score anything in this phase: 42/43/44 are v1/v2/v3, 45–47 the
crossed confirmation, 48–50 this arm's parent. 51/52/53 are new.

## Design integrity

The candidate list was verified **byte-identical** to
`arch-zoo-100m-v4-head-paired` by dry run into a scratch artifacts root before the
hours were committed: the same 8 candidates with the same names, depths, hidden
sizes, head counts, expert counts, active experts, expert widths and cost axes, a
genome-set sha of `8e75c92bdcbb164b` on both sides, and `zoo_definition.json`
differing only in `experiment_id`. The head-paired report re-states the design's own
guarantees on the emitted definition: every pair complete = yes, exactly matched
total *and* active parameters = yes, ×1.019 across pairs, min zoo distance 0.050 —
low on purpose, because the two members of a pair differ in nothing but head count.

So this arm measures the seeds, not the architecture set.

## Pre-registered decision rule

Written before the numbers exist, and all-numeric — unlike v4's own five-outcome
table, which declared two of its clauses (`collapsing toward 0`, `holds near −0.18`)
person-judgements rather than cutoffs. v4's notes said a later arm whose branches are
all numeric could be evaluated mechanically, and this is that arm; the thresholds
live in a declared table in the report library (`HEAD_PAIRING_RULE`), transcribed
here first and checked against it, so the branch is read from the transcription
rather than by eye once the numbers land.

Every quantity below is computed on the **pooled six-seed set** (v4's 24 ledger rows
plus this arm's 24, 48 runs). Three consequences of pooling, all verified on a
synthetic six-seed copy of this design before the hours are spent:

- **The factorial's main-effect test stops being exact.** Its assignment count on six
  blocks is past the 1e6 enumeration cap, so both main effects fall to Monte-Carlo at
  the `--samples` resolution (5e-06 at the default 200,000) — reported as a
  *resolution* rather than as significance, exactly as the crossed confirmation arm
  had to.
- **The seed-blocked interaction floor drops to exactly 0.03125**, below the declared
  0.05, so a floor-limited result can now clear it. At three seeds it could not.
- **The paired head contrast gains 12 units** (24 within-pair differences instead of
  12), so its exact floor falls from 2/2¹² to 2/2²⁴ — and the test stops being
  *exact* there, because 2²⁴ sign vectors are past the 1e6 enumeration cap: it falls
  to Monte-Carlo at the 200,000-draw resolution (5e-06), with the design floor of
  2/2²⁴ still reported beside its procedure resolution. Each level's own split stays
  at 12 units and is therefore still enumerated exactly, floor 2/2¹².

Every level condition below reads the H=896 (wide) number from the **paired level
split** — 12 within-pair differences, the pair as its own control, exact — and not
from the factorial's `by_other_level` nats, which are the design-level view of the
same quantity and are reported beside it. The rule reads the paired number because
pairing is the whole reason this design can measure the head effect at all.

| outcome | condition (all numeric) | reading |
|---|---|---|
| **head effect is real at wide hidden** | seed-blocked interaction `p < 0.05` with all six seed contrasts the same sign, interaction nats `> 0`, and the wide-level paired head effect `< 0` | the level dependence is real: more heads help at wide residual-stream width and not at narrow. A search over this family must mutate `attention_heads` **jointly with `hidden_size`**, and the free lever is only free at wide widths — so "add heads" is a rule with a precondition, not a global one. |
| **interaction resolves, against heads** | the same conditions, but the wide-level paired head effect `>= 0` | the level dependence is real and points the other way — heads are worth more at *narrow* width, or the wide level is where they cost. Either way no head budget is warranted: width stays the lever and the head axis is informative only as a warning against a global "more heads" rule. |
| **head effect is real but uniform** | the interaction is not resolved **and** the pooled paired head contrast is significant and sign-consistent at the declared 0.05 | there is no level dependence, but the head effect is real and the same at both widths. That corrects the parent arm's three-seed exclusion of a head effect, and the search should add heads **globally** rather than jointly. |
| **interaction absent — the three-seed agreement was seed luck** | the six seed contrasts have mixed signs **and** the pooled head contrast is not significant | head count is worth ~0 at both levels. The head axis closes, and this is the *cheaper* world for a search: `attention_heads` need not be mutated jointly with `hidden_size`, which returns that axis to a free knob rather than a search dimension. |
| **the width attribution fails to replicate** | the pooled `hidden_size` main effect flips sign or is non-significant at the declared 0.01 | a correction to §2c. §2c's width attribution is then questioned on the design with the tightest matching in the phase, and the phase's headline reading has to be reopened **before** any search spends budget on width. Investigate before trusting either arm. |

One row above is not in the table as originally written, and the reason is worth
recording because it is the process working rather than a flaw in it: a head effect
that is real and identical at both widths has **no interaction to resolve** *and* is
not "worth nothing at both levels", so the absent row's reading would have been
false about it. The gap was found by the test suite while this arm was still
training — writing the rule's branches as tests and running them on synthetic
designs with planted effects is what surfaced it — and the row is added here, before
any of this arm's numbers exist, at the *same* declared 0.05 threshold applied to the
pooled contrast. No threshold entered the rule after the data.

Secondary, reported whatever the primary says:

- **The per-level paired contrasts** (12 units each, exact, floor 2/2¹²): mean
  difference, p, floor, and sign consistency at H=512 and H=896, so a level's number
  is not read off a pooled average that two disagreeing levels could produce. These
  are written into `decision.md` for every paired gene with a two-level partner, not
  only this one.
- **The pooled head contrast** over 24 units, as the average effect with its own
  floor — the number that decides whether "add heads" is worth anything on average.
- **The width contrast on the pooled set**, as a third replication of §2c at ×1.019.

Which branch is *not* reachable is stated as part of the rule: there is no "heads are
the lever" branch, because v4's own pooled contrast already excludes it at three
seeds and this arm's job is the interaction, not a second chance at the main effect.
A pooled head contrast that turned significant on its own would be reported as a
correction to that exclusion rather than slid into a fifth branch after the fact.

## Cost

8 candidates × 3 seeds × 1200 steps = 24 runs. At v4's measured 525 s/run that is
**~3.5 GPU-hours**; the driver's conservative projection at 4,500 tok/s is 3.64
GPU-hours, and v4's own completed rate (4,718 tok/s) implies 3.50.

**This arm is over the phase's declared budget, and that is declared here rather than
discovered in the cost table afterwards.** The running total before it is **20.121
GPU-hours** — 16.622 for v1/mixer/v2/horizon/v3/confirmation, plus v4's 3.499 —
against a 10–20 GPU-hour cap, so the phase already closed at its ceiling and this
arm carries it to **~23.6**, an overrun of about 3.6 GPU-hours.

The case for spending it anyway, stated so a reader can disagree with it:

- The head interaction is the phase's only live lead, and it is **undecidable from
  data already paid for** — 0.25 floor against a 0.05 threshold.
- Carrying it into the search program as an unknown is not neutral: the search would
  have to either mutate a free gene on three seeds' evidence or leave a possibly free
  0.046 nats on the table, and §2c's own analysis shows what acting on a floor-bound
  statistic costs (the horizon arm) in both directions.
- The alternative use of these hours — the §11 random-vs-manual comparison — does not
  fit them: it needs *both* arms flat and budget-matched (v1's manual numbers came off
  a promotion schedule), which is ~10.4 GPU-hours for 26 candidates × 3 seeds at 1200
  steps, i.e. a new budget rather than a remainder.

## Validated before the hours were spent

The analysis path was rehearsed on a **synthetic six-seed copy of this design**
built from the arm's own ledger rows, with loss recomputed from a known model —
width −0.10 nats at hidden 896, head +0.01 at 512 and −0.05 at 896 — so every
quantity this arm promises has an answer before the real ones exist. Both arms were
written into a scratch artifacts root (the parent's 24 rows as seeds 48–50, the
confirmation's as 51–53) and the pre-registered command was run against it:

| quantity | planted | recovered |
|---|---|---|
| `hidden_size` main effect | −0.13 (the average of the two head-level effects) | **−0.13**, Monte-Carlo p = 5e-06, **at the procedure floor** |
| interaction | +0.06 | **+0.0600 nats**, seed-blocked p = **0.03125** — exactly its six-seed floor |
| the six per-seed interaction contrasts | all +0.06 | all +0.06, sign-consistent |
| wide-level paired head effect | −0.05 | **−0.0500 nats**, exact p = 4.88e-04 **at floor**, 12 units |
| narrow-level paired head effect | +0.01 | **+0.0100 nats**, exact p = 4.88e-04 **at floor**, 12 units |
| pooled head contrast | −0.02 over 24 units | **−0.0200 nats**, Monte-Carlo (2²⁴ past the enumeration cap), 12 negative / 12 positive |
| pre-registered branch | `heads-free-at-wide` | **HEAD EFFECT IS REAL AT WIDE WIDTH**, all five conditions met |
| pooled cost split | 24 + 24 runs, 3.50 h each | both arms reported separately under `by_experiment` |

The rehearsal also shows the failure it is guarding against, and this is the reason
the level split is reported rather than inferred: **the pooled head contrast is
*significant* at −0.02 nats (p = 0.0062) purely from averaging two opposite levels** —
12 pairs at +0.01 and 12 at −0.05 — while its within-pair signs disagree 12/12. A
reader given only the pooled number would have called a level-dependent effect a
uniform one; the split is what says otherwise, and the rule prints that reading
alongside the branch instead of leaving it to be noticed.

So a failure when the real numbers arrive is a result about the design rather than
about the tooling — the standard the parent arm held itself to.

## How to analyse it

Independently first, then pooled for power:

```
PYTHONPATH=src python scripts/arch_zoo_analyze.py \
  --experiment-id arch-zoo-100m-v4-head-paired-confirm --samples 200000 --write

PYTHONPATH=src python scripts/arch_zoo_analyze.py \
  --experiment-id arch-zoo-100m-v4-head-paired \
  --also arch-zoo-100m-v4-head-paired-confirm --samples 200000 --write
```

The first is the replication — if the width contrast holds at the same sign and the
head contrast stays small, the arm's parent is not a seed artifact. The second is the
powered interaction test, and it writes to the **parent** directory, so v4's
committed `decision.md` becomes the six-seed record, as the crossed confirmation arm
did to v3.

The branch is evaluated by `--head-rule`, which applies the table above from the same
outcome the decision renders, condition by condition, and states which branch the
result lands in. Its thresholds live in `HEAD_PAIRING_RULE` as data rather than in
prose, for the same reason the crossed confirmation arm's do: a threshold that lives
only in a document gets reapplied by eye once the numbers are in. Before the real
numbers existed, every branch was exercised against a synthetic design with planted
effects — a level-dependent head effect (the `heads-free-at-wide` branch), a uniform
one (`head-uniform`), a noise-only one (`head-axis-closed`), a flipped width
contrast (`width-fails`), and a three-seed copy of the level-dependent design, which
correctly returns `open-floor-limited` at the 0.25 floor rather than a verdict. So a
failure when the real numbers arrive is a result about the design rather than about
the tooling — the standard the parent arm held itself to.

## Status

**Running** — launched after this pre-registration was written, on a free card (the
driver warns when another sweep holds the GPU, which is the failure that made v4's
first three hours look like training when they were scheduling). Projected finish
~3.5 hours after launch; the per-run pace is read off the first finished candidate
rather than assumed.

One launch detail is worth recording because it is invisible in the ledger: the
**first attempt died silently** about six minutes in, killed when the shell that
started it exited, with no rows written and no candidate contaminated — the log
below the plan section shows the second attempt and the first candidate of it. It
was relaunched detached (`setsid`, unbuffered) and the aborted section is kept in
`run.log` rather than deleted, since "the arm produced nothing for six minutes" is
information a later reader of the wall clock would otherwise have to guess at.The driver's own guard cannot see this either: it counts tokens, and a killed process
trains none.

## Status update, mid-sweep (2026-09-18, seed 52 in progress)

Three things are now armed or checked rather than pending on memory:

- **The analysis is automated.** `scripts/run_confirm_analysis_when_sweep_exits.sh`
  runs detached (PID logged in `analysis_watcher.log`), polls the sweep process,
  verifies all three ledgers hold 8 rows before writing anything, then runs the two
  pre-registered commands above in order. It refuses to `--write` over an incomplete
  sweep, so a killed sweep produces a log line and no decision. The write path was
  rehearsed into an isolated *copy* of both arms' directories first - a first
  rehearsal through symlinked directories wrote its partial output into this
  directory, which was removed before any of its numbers could be mistaken for the
  arm's: a rehearsal that writes must not be able to reach the real record.
- **The design-integrity claim is a standing check.**
  `verify_record.py` now asserts this arm's candidate list is byte-identical to the
  parent's (`experiment_id` alone differs), every ledger row is on a fresh seed,
  the plan is flat at 1200, and the structural join reaches both paired genes.
  These are true at any point mid-sweep; the contrast rows land when the ledgers do.
- **Interim read, not a verdict** (5 seeds' ledgers at writing, floor 0.0625):
  all five seed interaction contrasts agree in sign (+0.0383 .. +0.0754), the wide
  level's paired split is 9/9 pairs favouring more heads, and the pooled head
  contrast is already clearing its threshold on that sign-inconsistent pair set -
  the exact shape the added uniform branch warns reads as a uniform effect when the
  levels disagree. The branch is decided only at six seeds, where the floor falls
  to 0.03125.

### When the sweep exits (runbook, so the handoff needs nothing but this file)

The watcher fires both pre-registered commands automatically and logs to
`analysis_watcher.log`. If it is not alive (check `pgrep -f
run_confirm_analysis_when_sweep_exits`), run them by hand:

```
PYTHONPATH=src .venv/bin/python3 scripts/arch_zoo_analyze.py \
  --experiment-id arch-zoo-100m-v4-head-paired-confirm --samples 200000 --write
PYTHONPATH=src .venv/bin/python3 scripts/arch_zoo_analyze.py \
  --experiment-id arch-zoo-100m-v4-head-paired \
  --also arch-zoo-100m-v4-head-paired-confirm --samples 200000 --write --head-rule
```

The second writes the six-seed `decision.md` into the parent directory. Then close
out the record:

1. Read the branch from the parent's `decision.md`; it is decided by
   `HEAD_PAIRING_RULE`, not by eye.
2. Extend `verify_record.py`'s `check_head_paired_confirm` with the measured
   contrast rows (pooled head, per-level splits, interaction at its floor, branch
   label) and the §10 cost row `(24, <measured hours>)` in `check_costs`.
3. Update `ARCH_SEARCH_FINDINGS.md`: §2c's lead paragraph, the §28 checklist row,
   §9's item list, and §10's cost table (220 runs, ~23.6 h, the pre-declared
   overrun) — then re-run `verify_record.py` and fix the count it prints.
4. `OMP_NUM_THREADS=1 .venv/bin/python3 -m pytest tests/ -q` — the whole suite,
   not just the verifier's tests.

Rehearsed before the hours: both commands on a complete synthetic six-seed copy
(relabelled parent rows, kept in /tmp and deleted — its branch firing was an
artifact of the duplicated rows, so it validates the machinery and forecasts
nothing).

## The branch survives the like-for-like readings; the pooled contrast is thin

The holdout-split check was run on the pooled six seeds. The rule lands on the same
branch on both like-for-like readings — held-out loss and final training loss, both
being loss at the end of training — with all five conditions met and width at floor in
each. Under a third reading it does not: `mean_train_loss` averages over the whole
trajectory, one of the six per-seed interaction contrasts is −0.0050 (the others
+0.0306 to +0.0652), the seed-blocked interaction leaves its 2/2⁶ floor for 0.0625, and
the rule reports **INTERACTION ABSENT — THE THREE-SEED AGREEMENT WAS SEED LUCK**.

What is thin is the pooled head contrast: **p = 0.0115 against the rule's declared
0.01**, a margin of 1.15× or about six steps of that contrast's 1/4096 sampling
resolution. The two halves straddle the looser 0.05 a reader would apply from opposite
sides — the parent is non-significant held-out and significant on final training loss,
this arm is the reverse — so the pooled p is not a stable summary of them.

What does not move is the level split, which is what the recommendation rests on: at
H=896 all 12 units favour more heads at the exact 2/2¹² floor under all three readings
(−0.0553 / −0.0582 / −0.0344), while H=512 stays non-significant throughout. Every
number here is recomputed from the two arms' ledgers by `scripts/verify_record.py`.

