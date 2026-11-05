# The width pilot - pre-registration, before any seed has run

**Registered 2026-11-01, before the first job.** This is not Gate 4 and it is not a declared
arm. It is the development-fidelity reading `docs/width-control-design.md` section 10 orders
before any GPU week is spent: does a controller that decides *whether to widen* beat a coin
that decides the same thing, on the gene the architecture phase found moves?

## The question and the two arms

The architecture phase's one large, replicated effect is residual width (-0.1562 nats at the
factorial floor). The topology track's own decomposition says the honest primary is not the
total gain but the controller's edge over a random schedule:

```
adaptive - null = (adaptive - random) + (random - null)
```

So the pilot runs two single-arm sessions per seed, differing only in the decision rule at
the same three events, with the same graft and the same two probe evaluations per event:

| arm | decision at each event |
|---|---|
| `width` | accept the graft when the widened model does not lose to the current one by more than `width_accept_margin` nats (registered: 0.0) |
| `width_random` | accept on a coin seeded from `(seed, step)` with probability 0.5 |

Both widen by **+2 heads** (`hidden 320 -> 448 -> 576 -> 704` at `head_dim` 64), perturb the
new coordinates at scale `1e-3`, and probe the current model and the graft on the same
batches. A declined event changes nothing but still records the loss the graft would have
paid. A third arm, `null`, is the unpriced baseline the pool term needs; it already exists in
the session machinery and is not re-registered here.

## Shape, seeds, and where it runs

| | |
|---|---|
| shape | the dev shape: h320, depth 6, 8 experts, 2 active, width 320, vocab 8192, context 512, batch 8 |
| budget | 400 steps = 1,638,400 tokens per arm, events at 100/200/300 |
| data | TinyStories, the published `act-lab-tinystories-tokens` stream (train and held-out) |
| seeds | 71000-71011 (12 paired seeds) |
| host | Kaggle CPU, four single-thread processes, **zero GPU-hours** |
| cost, measured | the pre-registration priced 108 min per seed and **5.4 h** for 12 seeds; the 2026-11-01 smoke then measured Kaggle's CPU at **300-355 tok/s per process** (against 1,010 on the workstation), so the block's envelope is **~8 h at four workers**, with the 12 h session cap as the stop |

The seeds are a fresh range, not the declared queue's 15000-15011: this is a dev-shape run and
must not be mistaken for a declared one in any later join.

**The smoke corrected the price.** Before the block ran, one seed and two arms at 20 steps
and two events (the harness check, not evidence) trained in **273 s** for the wider arm -
13.6 s/step, against 4.3 s/step for the same shape at one thread on the workstation. That is
3.2x slower per step, and it is the number the envelope above uses. The 5.4 h figure was a
workstation rate applied to Kaggle; it is kept here as the estimate the smoke replaced rather
than quietly overwritten.

## The reading, fixed before the data

For each seed, the paired difference is `width.validation_loss - width_random.validation_loss`
(negative favours the adaptive rule). The pilot reports:

1. the per-seed differences and their mean, with the paired standard deviation;
2. the exact one-sided sign test on the number of seeds the adaptive arm wins;
3. the final width each arm reached, and the number of accepted events, so a difference can
   be read against the schedule that produced it rather than against an assumed one.

**What counts as a positive.** Ten or more wins of twelve (exact one-sided level 79/4096 =
0.019287) is a positive edge worth pricing at declared fidelity. This is deliberately the
registered bar, not a looser pilot bar: the pilot is allowed to be underpowered, and it is not
allowed to lower the bar because it is small.

**What counts as flat or negative.** Nine wins or fewer. Under the stated prior - the edge is
flat, as it is across the depth ladder (p = 0.879 from depth 3 to 9) - this is the expected
reading, and it closes the width arm rather than opening the 31.2 h declared block.

## The four outcomes, declared

1. **Edge positive and clear (>= 10/12).** The controller's decision rule has value on a gene
   that moves. The 11-seed declared primary at 28.5 h is then justified, and the two fixed
   controls follow it.
2. **Pool positive, edge flat (the prior).** The graft helps and the decision rule does not.
   That is the churn programme's result on the lever that matters, and the project's
   contribution is the architecture finding plus a well-established negative about learned
   topology - not a rebuilt search.
3. **Both flat or negative.** Widening a partly trained dev model does not pay at this dose.
   The declared arm would spend 31.2 h to reproduce a known answer at a nicer fidelity, and
   should not be started.
4. **Undecidable (wide spread, or the run does not complete).** Reported as undecidable at
   this fidelity, with the interval, and no declared spend follows.

## What this does not authorize

- It does not authorize the declared width block. That decision reads this pilot's outcome
  and the quota, and it is a separate registration.
- It does not test the **fixed-wide** or **fixed-narrow** controls, so a positive result is
  still ambiguous between capacity and compute. Those are the declared block's job.
- It is not a scale model of the declared schedule. At the dev base +128 is +40% width and
  three events reach 3.06x the active parameters; at h768 the same step is +17% and 1.73x.
  A flat edge at the larger dose is the *stronger* negative; a positive one is the same sign,
  not the same size.
- It does not pool with any earlier dev-shape study: the arm and its probe cost are new.

## How to run and analyse it

```bash
# stage (writes artifacts/width-pilot/kernel/, runs nothing)
PYTHONPATH=src python scripts/push_width_pilot.py --seeds 71000-71011 --stage-only

# push the CPU kernel (publishes the code dataset, mounts the token dataset)
PYTHONPATH=src python scripts/push_width_pilot.py --seeds 71000-71011 --push

# fetch the records
PYTHONPATH=src python scripts/push_width_pilot.py --fetch --save artifacts/width-pilot/kaggle
```

A smoke run - one seed, 20 steps, two events - is the harness check and is not evidence.

## Cost and stop conditions

The pilot's 5.4 h is wall clock on Kaggle's four cores and bills no GPU quota. It is one CPU
session; if the session dies, the remaining seeds are relaunched rather than restarted, since
each job writes its own record and a record already fetched is not recomputed. The run stops
after the twelve seeds or at the session limit, whichever comes first.

## Launch (2026-11-04T16:36Z)

Kernel `sid123123234/act-width-pilot` version **4** was pushed and reached `RUNNING` with the
full registered plan: seeds 71000-71011, both arms, 400 steps, events at 100/200/300, four
single-thread workers on Kaggle CPU. The code dataset is at `8bb017a93dd7` (the working tree
differed only in report prose; HEAD is what ships). Records land under
`/kaggle/working/width-pilot/` and are read against the "reading, fixed before the data"
section above, not against the runs.

## First session: 15 of 24 jobs, cut by the platform clock (2026-11-05)

Version 4 ended with `CANCEL_ACKNOWLEDGED` after the log's last stream entry at 41,406 s
(11.5 h of logged work), during the `71007:width` job. The fetch recovered **15 records**:
complete pairs for seeds **71000-71006** (both arms, config-identical within each pair except
`arms`), plus `71007:width_random`. The nine missing jobs are `71007:width` and both arms of
**71008-71011**.

The session's measured per-job wall time is longer than the smoke's envelope: the wider arm
ran 2.65-3.30 h and the coin arm 2.04-3.04 h, because the graft is accepted at every event
on every completed wider run and the active model grows to 3.06x. The 24-job block was
priced at ~8 h from the 20-step smoke; that price was optimistic for the full 400-step
schedule, and the platform's 12 h window, not the 8 h envelope, was the binding stop.

## Resume (2026-11-05T15:55Z)

Kernel version **5** was pushed with a job-scoped plan rather than the seed block: the nine
missing `SEED:ARM` jobs only, so no fetched record is recomputed. The code dataset is at
`a067286` (the launcher gained `--jobs` for exactly this resume; HEAD is what ships). Shape,
budget, events, data and the two arms are unchanged from the registration; four single-thread
workers, zero GPU-hours. Expected wall time is ~9 h from the measured per-job times, inside
the 12 h session window. The records from both sessions are read together against the
"reading, fixed before the data" section above - the partial block is not read as a result.

## First attempt void: the scheduler counted accepted grafts (2026-11-05)

The 15 records fetched from versions 4 and 5 are **not evidence**. Read against the
registered three-event schedule, every `width_random` record carries four events at
100/200/300/**400**, and `width` seeds 71003 and 71006 carry a fourth as well because they
declined one. `_train_arm` gated the width event on `arm_state.mutations < mutation_budget`,
and `mutations` counts *accepted* widenings, so an arm that declined kept firing until it
reached three accepts or the horizon ended. The step-400 event fires after the last
optimizer step and before the final validation, so it changes the loss the block is read on.
The two arms therefore did not run "the same three events" this registration names, and no
paired reading exists.

The records and the reader's `protocol_deviation` output are preserved under
`artifacts/width-pilot/void-first-attempt/` (local, git-ignored). They are not pooled with
any later run and are not cited as a result. Kernel version 5 was deleted before it could
write more old-schedule records.

## Corrected rerun (2026-11-05)

The fix in this commit makes the width arms budget *events*: a width event fires when
`step <= mutation_interval * mutation_budget`, so the pilot's 400-step horizon has exactly
100/200/300 whether every graft is accepted or declined. The other arms keep their
successful-mutation budget unchanged. `test_declines_do_not_buy_extra_width_events` pins
the toy analogue - three slots at 3/6/9 in a 12-step horizon, no event at 12 - for both a
`width` arm that always declines and a `width_random` arm that never accepts.

The block is rerun **in full**, all 24 jobs, rather than resumed: the five width records
that happened to fire exactly three events are not reused, so every pair comes from the
corrected scheduler. Seeds 71000-71011, both arms, 400 steps, events at 100/200/300, four
single-thread workers, zero GPU-hours. The reading is still the one fixed before the data;
this section records a harness repair and a rerun, not a change to the rule.

Kernel `sid123123234/act-width-pilot` was recreated as version **1** and reached `RUNNING`
at **2026-11-05T16:13Z** with that full plan. The code dataset is at `44cfb1a` (the working
tree differed only in report prose; HEAD is what ships). At the measured per-job times the
24-job block does not fit one 12 h window, so a cut session is resumed with `--jobs` for
the missing `SEED:ARM` pairs under the same revision - the corrected records already
fetched are not recomputed, and the void attempt's records are never fetched into the
block's directory.
