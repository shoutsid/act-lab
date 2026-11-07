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

## Second session: 9 of 12 pairs, and the corrected schedule holds (2026-11-07)

Version **1** ended `CANCEL_ACKNOWLEDGED`, its log's last stream entry at **42,849.7 s**
(11.9 h of logged work), during the 71010 jobs. The fetch recovered **18 records**: complete
pairs for seeds **71000-71008**, config-identical within each pair except `arms`, and every
record carries exactly the registered events at **100/200/300**. The corrected scheduler
held, including on the arms that declined - 71003's wider arm declined step 100, and every
coin arm declined at least one.

The six missing jobs are both arms of **71009-71011**. Read against the registration they
are not a result: `scripts/analyze_width_pilot.py` exits **2** with `UNDECIDABLE - block
incomplete (9/12 pairs)`. Descriptively the adaptive arm is behind - **1 win of 9**
(p 0.998), mean delta **+0.054782 nats**, paired SD 0.049056 - because it accepted 25 of its
27 events and reached **704** width on seven seeds (576 on two), while the coin arm accepted
12 of 27 and reached 320-576. The reading is saved beside the records as `reading.json`
and is not cited as an outcome.

The per-job wall times explain the cut: wider **134.2-167.1 min** (mean 157.8), coin
**99.6-155.8** (mean 130.9) - **2,598 job-minutes** for the 18 jobs, so the 24-job block is
~57.8 job-hours, ~14.5 h at four workers against the platform's ~12 h. The ~8 h envelope
recorded in the registration stays the workstation-rate estimate it was written as, not a
measurement of the block.

## Resume, second attempt (2026-11-07)

Kernel version **2** was pushed with the six missing `SEED:ARM` jobs only, so no fetched
record is recomputed and the void attempt's directory is still never fetched. The code
dataset is at **`014195b`**; shape, budget, events, data and both arms are unchanged, four
single-thread workers, zero GPU-hours. Two waves at the measured per-job times is **~5.4 h**,
inside the session window. The block is read only when the reader exits 0; the partial
9/12 reading above is a description of where the session stopped.

## The reading: 12/12 pairs, 1 win, and the width arm closes (2026-11-07)

Kernel version **2** finished `COMPLETE` with **24 records, 12/12 complete pairs** (12,616 s of
logged work in the 12 h session). `scripts/analyze_width_pilot.py` exits **0** and prints
`STATUS: complete` / `OUTCOME: flat or negative edge (<= 9/12); close the width arm`. This is
the registered reading; the 9/12 description above is not.

| | |
|---|---|
| wins | **1 of 12** (losses 11, ties 0), exact one-sided p **0.999756**; bar 10/12 |
| mean delta | **+0.058707 nats** (`width - width_random`, negative favours the adaptive rule), paired SD **0.046951** |
| the one win | seed **71003**, `-0.026228`; the largest losses are **71001** `+0.129112` and **71010** `+0.124891` |
| schedule | wider arm accepted **33/36** events, ending at **704** on nine seeds and 576 on three; coin arm accepted **16/36**, ending at 448 on six, 576 on five, 320 on one |
| protocol | the registered events 100/200/300 on every record, no deviations |

The reading lands in outcome 3 of the four declared - *both flat or negative* - and its
registered consequence applies: the declared 31.2 h block is not started. Two limits are part
of the reading rather than of the prose around it. The pilot carried **no null arm**, so it
cannot separate outcome 2 (pool positive, edge flat) from outcome 3; what it measures is the
decision rule, which is the term the declared block was gated on, and the decomposition still
needs the null arm if anyone wants the pool term at this fidelity. And the dev dose is
**3.06x** active parameters against the declared schedule's **1.73x**, so a flat edge here is
a stronger negative at a larger dose, not a scale model of the declared one.

The block spent **zero GPU-hours**; its cost is the platform's wall clock, and the two cut
sessions are recorded above rather than folded into the result. The fetched records and this
reading are committed beside this note (`artifacts/width-pilot/kaggle/`, `reading.json`), and
the numbers above are the reader's output, not a hand computation.

## Amendment: the null arm, registered before its first job (2026-11-07)

The 12/12 reading above closes the *decision rule* and leaves the block's other term open. The
declared decomposition is

```
adaptive - null = (adaptive - random) + (random - null)
```

and only the first bracket has been measured. The second - the **pool term** - needs a do-nothing
arm, and outcome 2 of the four-outcome table (*pool positive, edge flat*) is therefore not yet
excluded. It is the outcome this note named as its **prior**, so leaving it unmeasured would report
the surprising branch as if it were the only one.

**The arm.** `null`, the session's do-nothing arm: it consumes each event slot and records a no-op
(`operation: "null"`, unchanged architecture hash, no probe and no search FLOPs). Shape, budget,
stream and events are the registered ones; the arm takes no graft, so its final width stays 320 and
its mutation count is 0.

**The block.** 12 null-arm jobs, seeds **71000-71011** - the same spent seeds, because the
comparison is paired against records already fetched, one process per job, four workers, Kaggle
CPU, **zero GPU-hours**.

**Pairing.** `random - null` is a same-initialization comparison, as §5.40 paired them
(`null` shares the random arm's `arm_seed = seed * 10 + 1` identity, so the two differ in the
mutation schedule and nothing else); the reader refuses a null record whose config differs from its
coin arm's in anything but `arms`, or that recorded anything other than a no-op event.

**The primary and the bar, fixed now.** `width_random - null` per seed, negative when the graft
pays; **10 or more of 12** at the same exact one-sided level (79/4096 = 0.019287). Two outcomes:

- **P1 (>= 10/12): pool positive.** With the edge flat at 1/12, this is outcome 2: the graft pays
  and the decision rule does not. The follow-up is the fixed-wide versus fixed-narrow controls,
  which separate capacity from compute - not another rule - and the width arm stays closed.
- **P2 (<= 9/12): pool flat or negative.** Outcome 3 stands on both terms, widening does not pay at
  3.06x active parameters, and the width programme closes at this dev dose.

**What it cannot do.** It cannot reopen the declared 31.2 h block on a positive pool term (that
needs its own registration and the two fixed controls), and it does not turn a dev-fidelity pool
term into a declared-scale one. The dose caveat from the design note applies unchanged: 3.06x
active parameters here against 1.73x at the declared shape.

**Cost, measured.** The coin arm's 12 jobs measured **99.6-155.8 min** (mean 130.9); a null arm does
no graft and no perturbation, so it is at or below that - **~5.5-6.5 h** at four workers, zero
GPU-hours, on the same kernel and the same stream.
