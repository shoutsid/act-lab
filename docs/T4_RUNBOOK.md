# T4 Pre-Launch Runbook — Section 5.1 Protocol Sessions

Operator runbook for executing the pre-registered compute-efficiency protocol
(`RESEARCH_REPORT.md` §5.1) on a Kaggle T4, using the session plan committed in
§5.2. Everything below is already implemented and tested; this document adds
only the exact commands, expected outputs, and decision points.

**Pre-registration discipline: no knob is tuned at any point in this runbook.**
The dtype rule, schedule constants, seed block, probe budget, criteria, and
tolerance band are fixed in code and in §5.1/§5.2, as amended by the **adopted
§5.5 revision** (signed off 2026-09-14). The verdict is machine-checked.

**What §5.5 changed, and only this:** the primary controller uses the `spread`
candidate policy; the declared queue is **15024-15035** (re-reserved twice -
§5.5 change 4 moved it off the spent 9000-9011, and §5.65 off 15000-15011, whose
first four seeds were published before the launch path carried the §5.47 rebuild
treatment); a **pool-matched loss-blind control
session** runs over the same seeds; and the verdict withholds Gate 4 if that
control clears its own gate, or if it was never run. The 5% criterion, the
tolerance band, the harm guard, the declared model, the budget and the
no-re-tuning rule are all untouched.

---

## 0. Pre-flight (before session 1)

1. **Code and revision.** Clone the repository and record the revision. Every
   session summary embeds the git revision (`code_revision`), so run untagged
   state only if you record it manually. Verify quickly:

   ```bash
   git rev-parse HEAD
   PYTHONPATH=src python -c "import act, torch; print(torch.__version__)"
   ```

   The Kaggle image ships `torch` and the `kaggle` package; nothing to install.

1b. **Rehearse the verdict path (no GPU, ~1 s).** The verdict in §4 is applied
   once, at the end of the whole spend. Two defects have already been found in
   that machinery by reading it (report sections 5.41, 5.42), so run its
   rehearsal before the spend rather than discovering a wiring error after it:

   ```bash
   PYTHONPATH=src python scripts/verdict_rehearsal.py --check
   ```

   It applies the real `compute_verdict` to synthetic declared-scale fixtures -
   every withholding clause, both absent-queue and both short-queue cases, and
   the label-precedence combinations - and exits non-zero if any variant's
   label, exit code or Gate-4 authorization stops matching the registered rule.
   It trains nothing and is not evidence about the hypothesis; it shows the
   decision path intact. Section 5.43 records the truth table.

2. **Working layout.** All commands run from the repository root with
   `PYTHONPATH=src`. The two persistent roots are:
   - `artifacts/datasets/` — the versioned checkpoint dataset (the Kaggle
     dataset content; ~45 GB ledger);
   - `artifacts/v0-sessions/` — the session checkpoint root (steady state
     ≈ 14.9 GB, `max_checkpoints=2` pin).

   **On Kaggle those roots are not under the working directory** — see the
   disk budget in §0.5. `scripts/kaggle_t4.py` passes absolute overlay roots
   for them, scoped by mode *and* by run
   (`/tmp/act-lab-<mode>-<run-token>/…`, the token derived from
   `KAGGLE_CONTAINER_NAME`), so two kernels can never share a directory;
   running the launch command by hand on Kaggle must pass the same kind of
   absolute root. **One session per queue at a time is the design** - the
   queues are sequential and the weekly GPU quota, not parallelism, is the
   constraint - and the run-scoped root exists so that an accidental overlap
   is harmless instead of corrupting a shared directory on the host overlay.
   The run's root is deleted at session end once the exit publish has reported
   anything other than `failed` (the durable copy is the published dataset).

3. **Remote dataset name.** Choose one `owner/slug` for the whole run (e.g.
   `<your-account>/act-v0-5-2`) and use it identically in every session. The
   first publish bootstraps dataset creation; you do not need to create it by
   hand. If you do create it, `kaggle datasets create -p <dir> -r private`.

4. **Environment check.** Run `notebooks/00_environment.ipynb` on the T4
   runtime. Expected, hard assertions inside:
   - CUDA available, T4-class device;
   - declared config reproduces **285,336,576 total / 134,341,632 active**
     parameters (the §5.1 tolerance band derives from these counts);
   - schedule/dtype constants printed for eyeball confirmation.

5. **Never launch on 9000–9011.** That registered block **is spent**: the §5.3
   real-data diagnostic ran the whole queue through the same `run_pair_session`
   at reduced fidelity and a §5.1 verdict already exists over it
   (`artifacts/real-proof/datasets/v0-5-1-verdict.json` — measured cost
   1.0077%, `PASS_COST_ONLY_HARM_GUARD_REJECTS`, `gate_4_authorized: false`,
   adaptive losing 11 of 12 pairs). The harm guard is one of the branches that
   decide this verdict, so a run on that queue would not be blind. The adopted
   §5.5 change 4 re-reserved the queue to the unused block **15000–15011**, and
   **§5.65 re-reserved it again to 15024-15035**: the first four seeds of
   15000-15011 were published before the launch path carried the §5.47 rebuild
   treatment **and their harm numbers were read and reported** while the campaign
   was stopped, so that block is neither the registered configuration nor blind
   over the seeds already seen. The live block is the code default
   (`v0_runner.DECLARED_SEEDS`), so the §2 command passes `--seeds 15024-15035`
   explicitly and never a spent block. A re-reservation needs a **fresh dataset
   root** as well as fresh seeds - one root is one block - which is why the slug
   above moved from `act-v0-5-1` to `act-v0-5-2`.
   Likewise, publish under **fresh** roots — never point a session at
   `artifacts/real-proof/datasets`, whose completed reduced-fidelity seed
   versions the skip-completed idempotence rule would treat as finished work.

---

## 0.5 Driving the run as Kaggle kernels

Sections 1-2c describe what a session *does*; this section is how a session is
started. The registered commands run as **script kernels** driven by
`scripts/kaggle_t4.py`, whose behaviour is selected by `ACT_T4_MODE`
(`preflight`, `session`, `control`, `null`, plus `probe`, `dtypeprobe`,
`verdict` and `metadata`). Every mode copies the attached code dataset into the writable
working dir, runs the registered command unchanged, and keeps the session log
and publish log in the kernel output.

Drive a queue with `scripts/kaggle_queue.py`, which stages the kernel, pushes
it, waits for the session, and re-pushes until the declared block is complete.
One push is one session, so `--max-sessions 1` means "run one session now",
and the default budget of eight covers the three sessions the primary queue
needs (4+4+4 pairs, by the runner's own start gate in the declared 7 h window -
see §3; the registered first pass is 6 pairs per session, which would be two),
with slack for a session that advances less than a full block:

```bash
python scripts/kaggle_queue.py --queue session
python scripts/kaggle_queue.py --queue control --max-sessions 8
python scripts/kaggle_queue.py --queue null --max-sessions 8    # §2c do-nothing baseline
python scripts/kaggle_queue.py --queue verdict          # after the three queues
python scripts/kaggle_queue.py --queue preflight        # one-shot diagnostics
python scripts/kaggle_queue.py --queue dtypeprobe       # fp32 vs fp16 rate (T4)
python scripts/kaggle_queue.py --queue selfcheck        # session battery + arm placement (T4)
python scripts/kaggle_queue.py --queue archzoo          # architecture-search horizon arm (one push)
```

`--wait-for-quota` is for the whole campaign rather than one queue: the three
queues need ~55 GPU-hours against a 30 GPU-hour week, so no single week can
finish them. Passed to a queue, a week too short to cover another session
**sleeps to the payload's own `refreshAt`** and re-runs instead of exiting 3 -
sleeping is free, and the alternative is a two-week run that stops every week
until an operator notices. It is bounded (`--max-waits`, default 2) so a queue
that never fits cannot sleep forever, and it refuses to sleep at all when the
refresh time is unreadable, because a blind sleep of unknown length is how an
unattended campaign stalls silently for days. Without the flag the behaviour is
unchanged: stop and say why.

`--queue` accepts eight values, and all eight are listed above: the three
queues (`session`, `control`, `null`) and the five one-shot jobs. `null` is the
§2c baseline that §1-2 describes the other two against, and `archzoo` is the
architecture-search horizon arm, whose own `NOTES.md` pre-registers its rule.

`preflight`, `dtypeprobe`, `selfcheck`, `verdict` and `archzoo` are one-shot jobs
rather than queues: they are pushed once and reported, with no re-push loop and
no quota gate (the verdict runs on CPU and bills nothing; the others are a few
minutes of GPU each, except `archzoo`, which is hours). All of them run the
mounted code dataset, so the driver publishes it if it lags the tree before
pushing - a one-shot that ran older code would report on a run it did not
measure.

`archzoo` is the one one-shot whose *output* is imported before its status is
checked, because a session that ended with an error after hours of training may
still have kept its ledgers, and those rows cost GPU-hours to regenerate. It is
also the one that reports **2** rather than 0 when the kernel exited cleanly but
the arm did not complete: the arm's own check reads the requested
(length, seed) grid off the ledgers, so a sweep that trained none of it says so. `--queue verdict` returns the registered
section 4 decision as its exit code - **0 PASS, 1 FAIL, 2 PASS with
qualification, 3 no verdict produced** - so the last step of the run is
scriptable, and it prints the verdict artifact. The kernel itself always exits
0 whatever the decision, so the code travels in `v0-5-1-verdict.json`
(`exit_code`) for the driver to read back. If that artifact says the declared
block is incomplete (`queue_complete: false`) the driver reports the verdict
but returns **3**, not the provisional PASS/FAIL: the script deliberately still
computes a provisional verdict on a partial block (seeds missing, Gate 4
withheld), but the driver's exit code is about the run, and a partial block has
not produced the registered decision.

The driver exits 0 when the queue is complete, 1 on a failed session, 2 when
the session budget ran out with seeds still missing, **3 when the weekly
quota cannot cover another session** - re-run it after `refreshAt` rather than
pushing again - and **4 when a session completed without advancing the
queue** (it reports how much quota that cost): a session that idles because
the clock never fit a pair reports success, so re-pushing repeats it verbatim,
which is the one loop that can spend a week's budget producing nothing. Fix
the cause - usually `--hours` - then re-run. It reads `kaggle quota --format json` before every push and
refuses when the remaining GPU hours are below the declared window, because a
session stopped part-way still bills every hour it used and produces nothing.
Over the ~52 GPU-hours of training the three queues need - nine sessions,
about 55 billed - against a 30 GPU-h/week quota (about two weeks of sessions)
that is the difference between waiting a day and losing the quota. The plan is
not prose: `three_queue_plan` composes it from the runner's own start gate and
the launch artifact carries it (`schedule.three_queue_*`). An unreadable quota
line never blocks (it can only ever spend less), and a session that is already
live is adopted instead of gated.

A script kernel receives **only** `code_file`: the CLI sends that file's text
and nothing else from the push folder, and the metadata has no field for
environment variables. The staged entry script is therefore generated with **every** knob the kernel
resolves from its own environment pinned inside it (`run-session.py`,
`run-verdict.py`, ...): the mode, the `--hours` window, and the owner/slug
that `Queue.dataset` - the dataset a session publishes to - is built from.
Each of those was a silent no-op when set on the pushing side, and each only
looked like it worked because the default happened to match; the owner/slug
one would have published to the default dataset while the driver believed
otherwise. Staging refuses if any of those constants is renamed in the body.
Copying
`kaggle_t4.py` into a folder and exporting `ACT_T4_MODE` on the pushing side
does **not** work - the kernel never sees that environment.

Three Kaggle-side facts that are not obvious, recorded because they already
cost one failed run:

1. **The code dataset mounts nested.** The current runtime attaches a dataset
   at `/kaggle/input/datasets/<owner>/<slug>/`, not `/kaggle/input/<slug>/`.
   The driver searches the mount tree for `pyproject.toml` + `src/act` instead
   of assuming a depth, so the flat and `<owner>/<slug>` layouts also resolve.
2. **`gpu_type` is not an accelerator pin.** `kaggle kernels push` honours
   `machine_shape` (`NvidiaTeslaT4`); an unknown key is silently dropped and
   the run lands on whatever Kaggle assigns. The generated metadata sets it.
3. **The dataset API needs no secrets.** `KaggleApi().authenticate()` succeeds
   inside a kernel from the session's own identity, with no environment
   variables and no `kaggle.json` (probed 2026-09-15; `ACT_T4_MODE=probe`
   re-checks). Internet access must stay enabled for restore and publish.
4. **`/kaggle/working` is a 20 GB mount and the handoff is tens of GB.**
   Measured on the T4 runtime (2026-09-15): `/dev/loop1` gives `/kaggle/working`
   19.5 GiB, while the root overlay (`/tmp`, `/kaggle/temp`) has 7.9 TB with
   1.1 TB free. A completed seed publishes its final checkpoint per arm
   *including the AdamW moments* — `model` + `optimizer` state is ~3.19 GiB per
   arm, so ~6.4 GiB per seed and ~77 GiB over the declared block, and the
   section 5.2 working-dir ledger (14.9 GiB steady state + 5.1 GiB headroom)
   accounts for the *session* alone, not the accumulating dataset root. Left in
   `/kaggle/working` the run dies on disk during its second pair — and because
   the per-pair local publish is not try/guarded, an `ENOSPC` there aborts the
   kernel *before* the exit publish, so the session publishes nothing. Hence
   the run-scoped overlay roots (`/tmp/act-lab-<mode>-<token>/artifacts/...`) in
   §0.2: nothing in those roots needs to survive the session, since the
   published dataset is restored at session start and pushed again at exit.
5. **A push does not supersede a running session.** `kaggle kernels push` adds a
   version and runs it *alongside* whatever is still going: measured
   2026-09-15, a second push left the first session running and the burn rate
   against the weekly quota line went from 1.00x to 2.00x - the one failure
   mode that costs quota without producing evidence, and `kernels status`
   reports only the latest version so the overlap is invisible from there.
   Sampling the quota line twice and differencing it is how it was caught. Run
   one session per queue: `scripts/kaggle_queue.py` refuses to add a second and
   adopts the live session instead. `kaggle kernels delete <owner>/<slug>
   --yes` *does* stop the runs (verified: the burn fell to 0.00x), but it stops
   every version including the one you want, so the only way back is
   delete-then-push-once. The CLI advertises
   `delete <owner>/<slug>/<version>` but raises in argument parsing before it
   calls anything, and the API has no version parameter, so stopping one
   version is not possible.
6. **The device count was never recorded.** Pre-flight logged `nvidia-smi` as
   `stdout[-800:]`, which is the *processes* table, so the GPU inventory table
   never reached the log; the benchmark recorded `device`/`device_name` but not
   `device_count`. Kaggle's shape string does not settle it either - the
   account's "T4 x2" notebook declares the same `machine_shape:
   NvidiaTeslaT4` as this run, and `machine_shape` is a free-form server-side
   string with no enum in the SDK. Meanwhile training reaches the GPU only
   through `resolve_device("auto")`, which resolves to plain `cuda` = **device
   0**, so a second device would sit idle for the whole run without appearing
   in any number. The benchmark now writes `gpu_inventory` (`count` + `names`)
   beside the rate and pre-flight logs it, so the next session settles it. If
   `count` is 2, the measured 1,958.7 tok/s is a one-device rate and the pairs
   - two independent arms - could train one arm per device for the same
   tokens, halving wall-clock and (since one session bills ~1.0x of the quota
   line regardless of devices, measured above) plausibly halving the plan;
   that is a *schedule* change, so it re-runs the section 5.2 benchmark and
   re-derives `full_pairs_per_session` rather than being assumed. **Settled
   2026-09-15: `count` is 2**, one arm now trains per device (see the schedule
   section above), and the three-queue plan is ~52 GPU-hours of training (nine
   sessions, ~55 billed) rather than the ~86 the fp32 sequential arithmetic
   implied.
7. **The kernel runs the mounted dataset, not this working tree.** The staged
   kernel body is embedded from the tree, but `src/`, `scripts/` and `tests/`
   are copied from the attached `act-lab-code` dataset, so a dataset that lags
   the tree runs different code than the run claims - and the handoff manifest
   records `code_revision`, so the provenance would be wrong too. Measured
   2026-09-15: the dataset was a commit behind, and every fix in between
   (including the guarded handoff publish) was absent from the session that was
   running. `kaggle_queue.ensure_code_dataset` runs before any quota is spent:
   it pins the dataset to `git rev-parse HEAD`, refuses a **non-git** tree (no
   commit to pin to), and republishes when the published copy is not at that revision
   **or does not carry the package tree**. The marker alone is not evidence -
   it is a loose top-level file, so it survives an upload that drops every
   subdirectory while the tree does not. So the tree travels as one archive
   (`code.tar`, whose internal paths the service restores), the staging marker
   is written into the staging folder and never the repo, and
   the read-back is polled, because a download issued immediately after a
   version push can serve the previous version and look like a rejected
   publish.

   **A dirty working tree is reported and does not stop the run** (2026-09-21).
   The payload is built from `HEAD` (`committed_tar`/`kernel_body`), never from
   the working tree, so an uncommitted edit cannot ship under a committed
   revision's name - which is what a dirty tree used to mean, and why
   `ensure_code_dataset` refused outright. That refusal was measured to cost a
   campaign: the driver checks the dataset before every push, so one
   uncommitted edit during a poll would end a two-week unattended run *after* a
   session had already trained and published. The failure is now removed rather
   than detected - the dataset is exactly a commit by construction - and the
   dirt is named in the log ("the working tree differs from `<rev>` in N
   path(s)") so the operator is not left assuming the tree is what ships. The
   kernel body follows the same rule: it is taken from the working tree only
   when that one file is unmodified, so a local edit stays testable without a
   commit while a body from one revision can never be embedded against a
   package from another.
   `artifacts/` is deliberately **not** shipped: a session must start with no
   `artifacts/` directory, so the launch command cannot read a benchmark
   artifact that this session did not measure itself.

**Session-1 inputs, measured on a Kaggle T4 (2026-09-15, `--steps 50`).** These
replace the section 5.2 feasibility assumptions; the launch command re-derives
the schedule decisions from the artifact rather than from the table. `1,958.7`
is a rate on the device training used (`device_name: Tesla T4`); the artifact's
`gpu_inventory` is what says how many devices the host offered next to it (item
6 - recorded from the next run onward, absent from this one).

| Field | Section 5.2 assumption | Measured on T4 |
|---|---|---|
| `measured_tokens_per_second` (fp32) | 10,000 | **2,124.2** |
| `dtype_decision` | fp32 | **fp16** (below the 2,500 floor - the registered contingency) |
| `gpu_inventory` | - | **2 x `Tesla T4`** (confirmed 2026-09-15) |

The pre-flight's `nvidia-smi` shows **two T4s**, both idle. The second one has
been idle for the whole run so far, and it is the reason the pair-time model's
parallelism is reachable at all - see below.

**The schedule is derived from the rate of the dtype the session trains.** The
fp32 number above is the pre-committed rule's *input*; it is not what the arms
run at. Measured on the T4 (`--queue dtypeprobe`, 2026-09-15):

| dtype | tokens/s | `pair_time_hours` | `full_pairs_per_session` | `sessions_for_primary` |
|---|---|---|---|---|
| fp32 | 2,124.2 | 6.53 | 1 | 12 |
| **fp16** (the decided dtype), arms sequential | **5,433.2** | **2.61** | **3** | **4** |
| **fp16, one arm per device** (what now runs) | **5,433.2** | **1.36** | **6** | **2** |

The last row is what the benchmark now reports on a two-T4 host: the placement
is part of the arithmetic, so the artifact describes the pair the session will
train rather than the sequential one it used to. The middle row is what the
same measurement produced before the placement was implemented, kept because
it is the evidence the `arms`-per-wave correction is worth 2x. **Both packing
columns are the *registered* fields** (`full_pairs_per_session`,
`sessions_for_primary`); the runner's own start gate adds the 45-minute publish
reserve to every start and packs one fewer pair, so plan from the artifact's
`pairs_per_session_rule1` / `sessions_for_primary_rule1` - at the 2026-09-20
rate (5,083.4 tokens/s, 1.443 h) that is 5 pairs and 3 sessions, not the 6 and
2 this table's form reads.

**Two corrections sit behind those rows, and they nearly cancel.**

1. *The schedule was built on the rule's input, not the runtime dtype.*
   Scheduling from the fp32 number while the arms train fp16 is 2.27x too
   pessimistic. The benchmark now measures the decided dtype too and records
   both (`schedule_tokens_per_second`, `tokens_per_second_by_dtype`), and the
   launch command schedules from the former while still enforcing the
   pre-committed rule against the latter.
2. *A pair's two arms are trained sequentially, not one per GPU.* The pair-time
   formula assumed the arms ran concurrently on separate devices; no runner
   implements that, and `run_v0_session` trains `config.arms` one after the
   other on one device (measured on a real pair, the arms' wall times added -
   0.531 s + 0.298 s - they did not overlap). The formula was missing the arm
   multiplicity, so every pair cost twice what the clock believed and the
   deadline rule would start pairs it could not finish - which section 5.2
   answers by restarting the pair from scratch, never resuming it.

Because the rate correction (2.56x) and the arm correction (2x) nearly cancel,
the *recorded* total was accidentally about right for the wrong reasons. The
corrected plan is: primary 12 pairs x 2.61 h = **~31 GPU-hours**, the 2b control
queue the same ~31, and 2c 12 pairs x 1.36 h (one arm) = **~16**, so about
**79 GPU-hours** against the free tier's 30/week. The recorded "2c ~0.3" does
not follow from the runner: an arm's token budget is `batch x context x steps`,
independent of how many arms run, so the null queue cannot be cheap.

**Superseded 2026-09-20.** The totals above are computed from the 2026-09-15
fp16 rate and the registered packing. The plan now comes from the measured
artifact and the runner's own gate, composed in one place (`three_queue_plan`):
at the T4's measured **5,083.4 tokens/s** placement rate a pair is 1.443 h, each
queue packs **5 pairs per session** (5+5+2 over three sessions), and the three
queues are **~52 GPU-hours of training over nine sessions** (~55 billed with
per-session overhead) - not ~47. The null queue in particular is a full-rate
third (~17.3 h), because a null seed trains the same 24.576M tokens as a pair
arm; a "~90 s/seed" figure that circulated for it came from a *dev-fidelity*
study (400 steps, dev shape) and understated the declared queue by ~58x per
seed. `schedule.three_queue_sessions` / `schedule.three_queue_training_hours`
are the artifact's own statement of this.

**And the second T4 is the point.** The pair-time model was written for one arm
per GPU; two T4s exist, so that model is reachable and the sequential runner is
what makes the plan larger than it needs to be. Running each arm on its own
device takes the pair to one arm-time - 24.576M / 5,433 / 3600 + 0.1 = **1.36 h,
6 pairs per session, 2 sessions for the primary queue**, and roughly **47
GPU-hours** for the whole run instead of 79. That is the registered design, not
a new one: it is the parallelism section 5.2 already describes.

**Implemented 2026-09-15** (`arm_worker.py` + `_train_arms`): a pair whose
host has a device per arm trains each arm in its own process, one per device,
and every arm records the device it used in its result, so the published
summary shows the placement. Nothing new is measured - the arms share nothing
but the seed, so the same numbers must come out however they are scheduled,
and `scripts/v0_session_selfcheck.py` checks exactly that (bit-identical arm
results, and the recorded placement, through both the process runner and the
session entry point).

**Verified on a real two-T4 kernel** (`--queue selfcheck`, 2026-09-15,
`sid123123234/act-t4-selfcheck`, ~2 min of GPU):

```
    host devices: 2
    one arm per device: cuda:0, cuda:1
v0 session self-check: all checks passed
```

That is the placement section 5.2 prices, confirmed on the hardware that has
it, before a session spent a window assuming it. Re-run it after any change to
the training loop or the placement code; on the project's one-GPU box it passes
with `host devices: 1`, which is the sequential fallback rather than evidence
of parallelism. If it ever fails, launch the queues with `--device cuda` and
remove the parallelism by training the arms sequentially - the pair-time
arithmetic follows the placement automatically (`arm_slots`), so the plan cannot
describe more concurrency than the run performs.

Pass `--hours 7` (the **declared window**: the measured 8 h platform cap less the
45-minute publish reserve and a 15-minute boot/prep allowance - see
`session_plan.PLATFORM_SESSION_CAP_HOURS`). At the fp16 rate the deadline rules
then place **two** completed pairs per session sequentially, or **four** with
one arm per device. Both are below the registered `floor(9 h / pair time)`
because rule 1 adds the publish reserve to every start; plan from the artifact's
`_rule1` fields (§3) rather than from the registered ones.

**Why 7 h and not the registered 9 h (amended 2026-09-21).** Section 5.2's
window was written against Kaggle's *documented* GPU limit - 12 h, per the
2022-01-24 product update that raised it from 9 h. The platform measured 8: Gate
4's session 1, pushed 2026-09-20 10:39:54Z, was killed with `ERROR` exactly 8 h
00 m later, before its own 9 h deadline. The kill was total - an empty kernel
log, no output files, nothing published - and it cost 8.05 GPU-h plus every pair
the session had trained, because a session spends its last hour on a pair and
the published dataset is its only durable copy. A 7 h declaration packs four
pairs and refuses a fifth, ending its last pair at ~5.8 h and publishing with
over an hour in hand, and it gives up nothing: 12 seeds need three sessions at
either packing, so the campaign is still nine sessions and a ~52 h training
floor.

A kernel that requests a GPU accelerator queues when no T4 is free (status
`QUEUED`, which consumes no quota); the queue clears on its own.

---

## 1. Session-1 throughput benchmark (fixes dtype + schedule)

```bash
PYTHONPATH=src python scripts/v0_throughput_benchmark.py --steps 50
```

Writes `artifacts/v0-benchmark.json` (the artifact the launch command reads).

Before launching the queues, run the session self-check on the **declared
pairing** - the seed queues train fp16 on CUDA, and every session after the
first resumes a CUDA fp16 checkpoint, so the resume-determinism claim is only
verified for the device/dtype it has actually been run on:

```bash
PYTHONPATH=src python scripts/v0_session_selfcheck.py --device cuda
```

Read from the printed JSON and record:

| Field | Meaning |
|---|---|
| `measured_tokens_per_second` | the **fp32** step-loop measurement (fwd+bwd+AdamW, activation checkpointing on) - the pre-committed rule's input |
| `dtype_decision` | `fp32` if ≥ 2,500 tokens/s, else `fp16` (pre-committed rule; not a choice) |
| `schedule_tokens_per_second` | the rate the schedule was derived from: the **decided dtype's**, measured in that dtype (equal to the fp32 rate when fp32 is decided, or on a device with no fp16 path) |
| `tokens_per_second_by_dtype` | both measured rates, so the two can never be confused after the fact |
| `schedule.pair_time_hours` | time per random+adaptive pair at `schedule_tokens_per_second` |
| `schedule.full_pairs_per_session` | section 5.2's registered arithmetic: pairs the *documented* 12-hour clock fits |
| `schedule.pairs_per_session_rule1` / `sessions_for_primary_rule1` | what the runner's start gate will actually do inside the **declared window** (4 pairs / 3 sessions at the measured rate) - plan from these, not from the registered row above |
| `schedule.declared_session_hours` / `platform_session_cap_hours` | the window the session is given (7 h) and the measured platform cap it is derived from (8 h) |
| `schedule.three_queue_sessions` / `three_queue_training_hours` | the whole campaign: nine sessions, a ~52 h training floor |
| `schedule.sessions_for_primary` / `sessions_total` | expected number of sessions for the 12-seed queue |
| `schedule.working_dir_headroom_gb` | must be positive (20 GB limit − steady state) |

Expected outcomes and responses:

- **OOM**: the script exits with a message pointing at the 16 GB T4 decision
  context. Do **not** shrink the config — run on a T4. `--profile` is for local
  smoke checks only and never produces launch input.
- **Unusable schedule**: the benchmark refuses to write an artifact when the
  measured pair time exceeds the usable window, because that yields
  `full_pairs_per_session: 0` — a session that bills its clock and publishes
  nothing, and the deadline rules never start a pair the clock cannot finish,
  so it fails silently. Measured cause (2026-09-15): a GPU whose *decided*
  dtype runs slower than fp32 — a Pascal card's fp16 path measured 411.6
  tokens/s against 1,733.1 in fp32, where the T4 does 4,694.9. The rule's
  decision context is a T4 (tensor cores: 65 vs 8.1 TFLOP/s); run there rather
  than recording a zero-session plan.
- **dtype = fp16**: expected on a T4 if throughput lands under the 2,500
  tokens/s floor. The sessions then train under autocast + GradScaler with
  fp32 master weights (checkpoint contract version 5 carries the scaler state;
  fp16 resume is bit-identical). Nothing to configure — the launch command
  re-derives the same decision from the artifact, and the benchmark measures
  that dtype too so the schedule is built on the rate the arms actually run at.

The benchmark is safe to re-run per session (it re-measures; the dtype rule
re-derives the same decision for the same rate class). Each session's artifact
is part of the trail; the launch command refuses a stale or hand-edited one.

---

## 2. The launch command (one line, registered as revised)

> The two rebuild flags below are the **section 5.47 amendment** (adopted 2026-09-16): the
> declared run carries the section 5.28 treatment, because the untreated rebuild is an
> unintended router discard kept as the default only to preserve the committed record of
> sections 5.16-5.23, and section 5.39 measured the difference it makes at **1.46x** on
> the churn cost the gate's quality clause reads. It is **not** a claim that the treatment
> decides Gate 4: the effect is small and right-skewed, and 69% of the churn harm remains,
> so the section 5.14 do-nothing clause is still the binding protection.

```bash
PYTHONPATH=src python -m act.training.cli --mode pair-session \
  --device cuda \
  --hours <remaining-clock-hours> \
  --seeds 15024-15035 \
  --probe-candidate-policy spread \
  --preserve-router-on-rebuild \
  --new-expert clone \
  --kaggle-dataset <owner>/<slug>
```

- `--hours`: the remaining session clock at entry — the platform's measured
  8-hour cap (not the 12 hours Kaggle documents) minus boot, restore and the
  45-min publish reserve, i.e. `--hours 7`. The deadline rules only start pairs
  that fit `pair_time + 45-min publish reserve`.
- `--seeds 15024-15035`, `--probe-candidate-policy spread` and the two build
  flags (`--preserve-router-on-rebuild`, `--new-expert clone`) are **not
  optional**: they are the adopted protocol (§5.5 and, for the rebuild, the
  §5.47 amendment). The code defaults are still the historical
  `prefix`/untreated pair, so every flag here is a deliberate restatement of
  the registered configuration rather than a change.
- Everything else defaults correctly: benchmark artifact
  `artifacts/v0-benchmark.json`, `--probe-batch-count 1` (the primary config),
  dataset root `artifacts/datasets`, session dir `artifacts/v0-sessions`.
- **The driver builds this same command.** `scripts/kaggle_t4.py::session_argv` is the
  launch path for all three queues and it carries the two rebuild flags from one constant
  (`REBUILD_TREATMENT_FLAGS`), so the command above and the one that actually runs are one
  protocol rather than two. That is now pinned in **both** directions by
  `test_the_runbook_and_the_launch_builder_carry_the_same_rebuild_flags`: every
  `pair-session` command printed here, and the builder, must carry the pair. Until
  2026-09-21 only this document carried them - the builder did not - and the first four
  published seeds therefore recorded the untreated configuration while the verdict, the
  registration (§5.47) and this runbook all named the treated one. Neither a doc-only nor
  a code-only check can see that: each side was individually consistent, and the test named
  "the registered one" was asserting the drifted argv.
- **A session will refuse to mix treatments, so do not fight the guard.** Before the first
  pair starts, the runner compares the treatment it is about to use against what the
  **published versions in the dataset root** record and raises `BlockTreatmentMismatch` if
  they differ, naming both fields, both values and the seeds that disagree. It chooses
  nothing - it prints the two remedies and stops. This matters because the seeds of a block
  are compared *within* the block: an untreated seed and a treated seed in one root are two
  experiments sharing a seed range, and no per-seed field partitions them afterwards.
  **Its one live incident is resolved by re-reservation.** Blocks `15000`-`15011`'s first
  four seeds are published under the historical untreated rebuild in `act-v0-5-1`, and the
  disposition recorded in `docs/IMPLEMENTATION_STATUS.md` is **§5.65's re-reservation**
  rather than finishing that block: its harm numbers had been read, so it could not be the
  blind registered block either way. Those four seeds keep their own artifact; the live
  block is `15024`-`15035` in `act-v0-5-2`, and the guard's job from here is to stop a
  *future* session from mixing treatments inside a root that is meant to hold one block.
- **Verify the run was the registered one.** Every seed summary records its whole
  config, and the verdict's `config_conformance` block compares it field by field against
  the registered protocol (`REGISTERED_RUN_CONFIG` in `scripts/v0_verdict.py`). It is
  descriptive - it changes no label and withholds nothing - so read it deliberately: if it
  reports `differs` on `preserve_router_on_rebuild`, the session ran without the §5.47
  treatment and its result is the untreated configuration, which must be said in the
  write-up rather than discovered later. **This is the check that reported the divergence
  after the fact, not one that prevents it**; the two bullets above are what prevent it, and
  the conformance block remains the audit that says the write-up was honest.
- Do **not** pass `--dtype` (it must equal the measured decision and is only
  there to catch operator error), `--tokens-per-second` (overrides the
  artifact), or `--probe-selection` (that is the control session in §2b).

**Session 1 expected behavior:**

1. Restore: warns that no remote dataset exists yet and continues — training
   without remote state is always safe (atomic pairs restart from scratch).
2. Trains pairs until the deadline rule stops at a checkpoint boundary;
   every completed pair publishes a `complete` dataset version and reclaims
   the working dir; a boundary-interrupted pair publishes an `in-progress`
   version (forensic only — the next session ignores it).
3. Exit publish pushes the handoff as a new dataset version and prints
   `remote_publish` with `ok=true`.
4. Final summary line: `pairs_completed`, `latest_published_seed`,
   `measured_tokens_per_second`, `pair_time_hours`.

**Every subsequent session is the identical command.** Verified end-to-end
behavior per session: restore latest → skip seeds with `complete` versions →
train the next incomplete pair from scratch → exit publish. Re-run the same
command until the queue completes; no scheduling decisions are made by hand.

**Failure modes at entry (fail-fast, by design):**

- Restore failure (bad credentials, wrong slug, network): clean one-line
  `SystemExit`, no training, nothing published. Fix the cause and relaunch.
- A `--dtype` contradiction with the artifact: refused before training.

---

## 2b. The pool-matched control session (adopted §5.5 change 2)

The same seeds, the same substrate, the same probe cost and the same `spread`
pool — the only difference is that the cap-bounded probe chooses *uniformly
among the candidates it evaluated* instead of taking the probe-best one. Run it
as a **second** queue with its own roots, so it can never be mistaken for the
primary measurement:

```bash
PYTHONPATH=src python -m act.training.cli --mode pair-session \
  --device cuda \
  --hours <remaining-clock-hours> \
  --seeds 15024-15035 \
  --probe-candidate-policy spread \
  --probe-selection random \
  --preserve-router-on-rebuild \
  --new-expert clone \
  --dataset-root artifacts/datasets-control \
  --session-dir artifacts/v0-sessions-control \
  --kaggle-dataset <owner>/<slug>-control
```

- `spread` is required here too. The clause isolates the *decision rule*; if
the control ran a different candidate pool, a null result would be
uninterpretable. **The two rebuild flags are required for the same reason**:
the control shares the primary's candidates, seeds and cost, so a different
rebuild policy would make the comparison a package of two changes instead of
the decision rule alone (§5.5 change 2).
- The control is descriptive evidence for the clause, not a second gate: its
own random arm is its paired baseline, exactly as in the primary run.
- Never point the control at `artifacts/datasets` (or the primary at the control
root): both queues hold the same seeds, and the two runs must not mix.

---

## 2c. The do-nothing null queue (pre-registered section 5.14)

The verdict's newest clause (§5.14) requires a third queue: single-arm null
sessions (`--session-arms null`) at the SAME declared fidelity over the SAME
block, into their own roots. A null session trains the identical init and
data stream but performs no probe and no mutation, so its per-seed loss is
the do-nothing baseline the adaptive arm must beat before Gate 4. Run it
AFTER the primary and §2b queues (it is **~1.44 h of GPU per seed** plus
session overhead - a null seed trains the same 24.576M tokens as a pair arm,
so the null queue is a full-rate third of the campaign, not an afterthought):

```bash
PYTHONPATH=src python -m act.training.cli --mode pair-session \
  --device cuda \
  --hours <remaining-clock-hours> \
  --seeds 15024-15035 \
  --session-arms null \
  --preserve-router-on-rebuild \
  --new-expert clone \
  --dataset-root artifacts/datasets-null \
  --session-dir artifacts/v0-sessions-null \
  --kaggle-dataset <owner>/<slug>-null
```

- No probe/policy flags are needed (the null arm is policy-free by
construction); `--seeds` and the roots must still be explicit, and the
null queue must never point at the primary or control roots.
- The two rebuild flags are inert on a null arm (it never rebuilds - zero
  mutations by construction, which the §5.14 checks enforce), and they are passed
  anyway so that all three queues are the same declaration. A config that
  differs between queues is a difference nobody would notice until it mattered.
- The published seed summaries carry `null_final_loss` and the zero-probe /
zero-mutation counters the §5.14 built-in checks read.

---

## 3. Running the queue to completion

The queue is 12 declared seeds (15024–15035), one atomic pair each. Schedule
expectations come from the benchmark artifact, and it now carries both readings:
`sessions_for_primary` / `sessions_total` are section 5.2's registered
arithmetic (`floor(usable hours / pair time)`), while
`sessions_for_primary_rule1` and `pairs_per_session_rule1` are what the runner's
own start gate will honour. **Plan from the `_rule1` fields.** The two agree
only while the 45-minute publish reserve does not bind; measured 2026-09-20 at
the fp16 placement rate (1.443 h per pair, one arm per device) the registered
field reads 2 sessions for the primary queue while the runner starts 5 pairs
per session and therefore needs 3 (5+5+2). The 20 GB working-dir ledger is part
of the plan's arithmetic; the publish reserve was not, which is exactly the gap
these fields close.

Rules the runbook operator must not work around (all enforced in code):

- Pairs are atomic: a pair split by the session boundary restarts from
  scratch next session. Never try to resume a split pair.
- `max_checkpoints=2` is pinned; the working-dir steady state (~14.9 GB) is a
  hard budget. Do not raise it.
- In-progress dataset versions are forensic only; do not restore from them.
- Stale working-dir state is discarded before a fresh pair start; do not hand
  curate `artifacts/v0-sessions`.
- Smoke tests (`--profile`) must use a **different** `--dataset-root` (or be
  cleaned up): the verdict step ignores complete seeds outside the declared
  block and warns about them, but the dataset stays cleanest when strays never
  land.

---

## 4. The verdict (machine-checked, pre-registered)

**Read this before the verdict lands, so it cannot be re-narrated afterwards
(2026-09-20).** Every withhold clause in the table below is the *same* rule: a 10-of-12
sign test on the adaptive arm, differing only in the comparison arm (random, pool
control, null). Three consequences, all computed in `scripts/null_clause_power.py`
while block 15000-15011 was still training:

- **A cleared clause is not evidence of value.** Clearing needs only **3 of 12** seeds,
  and under no effect it happens **98.07%** of the time (the clause fires on 79/4096
  blocks). If the verdict is a `PASS`, the claim it supports is "the clause did not
  fire", and the value inference rests on the registered cost/band/control rules.
- **A withheld Gate 4 is strong evidence.** At the dev-fidelity effect §5.40 measured
  (the null arm ahead by 0.136 nats at sd 0.113, z = 1.20) the null clause fires with
  probability **0.848**; at z = 1.5 it is 0.958. So `PASS_COST_ONLY_NULL_REJECTS` is not
  a coin flip that landed against the adaptive arm.
- **The clauses are not redundant.** On §5.40's fresh seeds the null clause fires
  (12/12) while the harm guard does not (6/12), so a verdict can withhold on one clause
  and clear another; the union bound on all three firing on noise is 5.79%.

When all 12 seeds publish `complete` versions:

```bash
PYTHONPATH=src python scripts/v0_verdict.py \
  --dataset-root artifacts/datasets \
  --control-root artifacts/datasets-control \
  --null-root artifacts/datasets-null \
  --probe-batch-count 1
```

After a Kaggle run these three roots are **empty locally**: the queues live in
the published datasets, and no local session ever held more than one session's
pairs. Each session publishes the whole handoff directory, so the newest
published version of each queue is cumulative - restores it and the queue is
reconstructed. `ACT_T4_MODE=verdict` does exactly that (restore all three,
then the command above, with the verdict JSON and its inputs kept in the kernel
output); it is a CPU kernel and needs no accelerator, so it never queues behind
a GPU. Running the command by hand locally is only correct if the roots were
actually populated (e.g. after a local proof run).

Exit codes and meanings (the two control rows are the adopted §5.5 clause;
the two null rows are the pre-registered §5.14 clause):

| Exit | Verdict | Pre-registered consequence |
|---|---|---|
| 0 | `PASS` | cost ≤ 5%, inside the [0.5, 2]× band, harm guard clear, **the loss-blind control did not clear its own gate, and the adaptive arm did clear it** (§5.44: the two arms must not be indistinguishable) → **Gate 4 authorized** (`gate_4_authorized: true` requires 12/12 seeds) |
| 1 | `FAIL` | primary cost > 5% → stop; §5.1 fails as registered |
| 2 | `PASS_WITH_QUALIFICATION` | cost ≤ 5% but outside the band → re-derive the predicted C **before** Gate 4; no knob re-tuning against the measured C |
| 2 | `PASS_COST_ONLY_HARM_GUARD_REJECTS` | harm guard hit (adaptive worse in ≥ 10 of 12 pairs) → do not proceed to Gate 4 |
| 2 | `PASS_COST_ONLY_POOL_MATCHED_CONTROL_PASSES` | the loss-blind arm won ≥ 10 of its 12 pairs, i.e. the pool alone clears the gate → the cost PASS stands but the quality inference does not; **never** treat this as "the controller wins" |
| 2 | `PASS_COST_ONLY_CONTROL_MISSING` | no `--control-root` given → the clause could not be evaluated; running the control is part of the protocol, so Gate 4 is withheld |
| 2 | `PASS_COST_ONLY_NO_DEMONSTRATED_EDGE` | **neither** arm cleared the pre-committed bar (adaptive < 10 of 12 wins against its own churn arm, and the loss-blind arm likewise) → the two are indistinguishable, so there is no quality premise to recommend Gate 4 on. The cost PASS stands. Added by §5.44; before it, this state authorized Gate 4 |
| 2 | `PASS_COST_ONLY_NULL_REJECTS` | adaptive failed to beat the null arm on ≥ 10 of 12 seeds → doing nothing is at least as good as the adaptive arm at declared fidelity; the cost PASS stands but the selection claim does not; **never** treat this as a tunable miss (no new block, no knob changes) |
| 2 | `PASS_COST_ONLY_NULL_MISSING` | no `--null-root` given (or the null queue is incomplete on the declared block) → running the do-nothing queue is part of the protocol, so Gate 4 is withheld |
| 3 | *no verdict* | the invocation was malformed (missing handoff, incomplete seed, unmatched budgets, a `--null-root` that is not a null queue). Nothing is written. This code is deliberately **not** 1: over an unattended multi-session run, an operator error must never be recordable as the pre-registered `FAIL` |

The verdict JSON is written to `<dataset-root>/v0-5-1-verdict.json` and prints
the aggregate cost, predicted value, band, harm-guard tally, the
`pool_matched_control` block, and per-seed win/loss pairs. Missing seeds are
listed (`queue_complete: false`) — keep running the launch command until the
queue completes before judging.

`--seeds "A-B"` (or a comma list) re-declares the block and loosens no
criterion; it is only for a future re-reservation, not for this run.

Secondary config (if pre-registered): same command with
`--probe-batch-count 2` (predicted 0.1883%, band 0.0942–0.3767%).

---

## 5. Contingencies

| Symptom | Response |
|---|---|
| OOM in benchmark or sessions on non-T4 hardware | The decision context is a 16 GB T4. Move the run; never shrink the declared config. |
| The session trains but the **exit publish fails** (driver reports a failed/incomplete queue) | The session's hours are not lost: `mode_session` keeps every published seed's summary in the kernel output as `seed-<n>-summary.json` (kilobytes), and those carry every number the verdict reads per seed (per-arm final loss, search FLOPs, mutation count). Fetch them with `kaggle kernels output <owner>/act-t4-session` and compare against the dataset; the *checkpoints* are dataset-only and cannot be recovered this way. |
| Exit publish fails (`remote_publish.status` = `failed`) | Never fatal to the *session*, and now retried 3 times with backoff before it is recorded - but treat it as a loss: the handoff roots live in the container overlay, so when the kernel ends there is no local copy for a later session to push, and every pair this session completed is gone. Check `remote-publish-log.json` (= `remote_publish` in the kernel log) for the error, then relaunch; the queue resumes from the last published version. |
| Two sessions of one queue running (burn at 2x) | They will not stop each other: `kaggle kernels delete <owner>/<slug> --yes`, then push once (§0.5 gotcha 5). Nothing is published by the abandoned version. |
| Local per-pair publish fails (`action: publish_failed`) | The pair is **not** recorded complete and the queue stops at that seed, so the session can still exit-publish the seeds that did complete (it would otherwise abort before publishing anything). Fix the substrate - disk space, permissions - and relaunch; the failed pair restarts from scratch. With the handoff roots on the overlay (§0.2) this should not fire. |
| Restore fails at session start | Fix credentials / `owner/slug` / connectivity; relaunch. Nothing ran, nothing published. |
| A pair raised while training (`training_failed` in the session log) | The queue stopped there and the completed seeds were exit-published; the driver exits 1. Read the recorded exception (usually an OOM or a killed arm process), fix the cause, relaunch - the failed pair restarts from scratch. Do not simply re-run it: the same push repeats the same failure. A `interrupted` action is *not* this: that is the deadline stop, which published an in-progress snapshot on purpose. |
| Session died mid-pair (reboot, timeout) | Relaunch the same command. The pair restarts from scratch; the stale `in-progress` version is ignored. **Everything that session completed is lost with it**: the durable copy is the exit publish (§6), which a killed session never reaches, so a session that died with five pairs done re-trains them. That is the price of one whole-handoff push per session; do not relaunch blindly if the same crash could repeat (check the kernel log for the cause first). |
| Kaggle auto-extracted archives | Expected and handled: the wrapper stages per-seed tars with `seed-NNNN/...` internal paths (live-verified bit-identical round trip). Restore also extracts blobs itself as defense-in-depth. |
| Verdict reports stray complete seeds | They are excluded from the aggregate automatically. For cleanliness, delete stray seed directories from the dataset root or use a dedicated root for smoke tests. |
| Benchmark artifact missing/stale | Re-run step 1. The launch command refuses a hand-edited or stale artifact by design. |
| Verdict says `PASS_COST_ONLY_CONTROL_MISSING` | The §2b control session has not run (or `--control-root` was omitted). Run §2b, then re-judge. Do not authorize Gate 4 on the strength of the primary run alone. |
| Verdict says `PASS_COST_ONLY_POOL_MATCHED_CONTROL_PASSES` | Record the cost PASS **and** the withheld Gate 4 as the result. This is a finding about the search design, not a failed run: do not tune the probe, the cap or the pool and re-run, and do not add a new seed block to try for a different answer. |
| Verdict says `PASS_COST_ONLY_NO_DEMONSTRATED_EDGE` | Record the cost PASS and the withheld Gate 4 as the result: the controller did not show an edge over its own churn arm at the pre-committed bar, so the quality premise this gate rests on is unsupported. Same no-retuning rule as the row above. Report it — with §5.13 and §5.39, this is a finding about the real path, not about this run. |

---

## 6. Artifact trail (what to keep per session)

- `artifacts/v0-benchmark.json` — the measured throughput and dtype decision.
- `artifacts/v0-sessions/session-log.json` — queue, per-pair actions
  (completed / interrupted / skipped / `training_failed`), FLOPs counters,
  `code_revision`. A `training_failed` entry carries the exception text and
  stops the queue: the seeds that completed are exit-published, and the driver
  exits 1 rather than pushing a session that would repeat the failure.
- `artifacts/datasets-control/` + `artifacts/v0-sessions-control/` — the §2b
  pool-matched arm (same seeds, `probe_selection=random`), and its own
  `session-log.json`. Keep both roots: the verdict's clause reads them, and
  every seed summary records its `probe_candidate_policy`/`probe_selection`, so
  which regime produced each number is auditable without this document.
- Dataset versions — every `complete` (and forensic `in-progress`) version of
  `<owner>/<slug>`, each with a checksummed manifest.
- `v0-5-1-verdict.json` — the final machine-checked §5.1 verdict.
- `remote-publish-log.json` — the exit-publish audit trail.

### Where the durable copy is, and what a crashed session costs

The published dataset version is the *only* thing that survives a session. The
handoff roots live in the container overlay (`/tmp`, ~1.1 TiB) rather than
`/kaggle/working` (a 20 GiB loopback) because a completed seed publishes its
final checkpoint per arm including the AdamW moments — ~6.4 GiB per seed,
~77 GiB over the 12-seed block — and the overlay is deleted at session end
once the publish has succeeded.

The exit publish is one `create_version` for the whole handoff, so it is
front-loaded with everything the session completed and its payload grows to the
full block by the last session. **That is why seeds are pushed at exit rather
than as they complete**: a per-seed push would re-upload the entire cumulative
handoff each time (tens of GiB, minutes of billed GPU per pair) to bound a loss
that only a crash can cause. The trade is deliberate: a session that dies
before its exit publish loses every pair it trained, section 5.2 restarts those
pairs from scratch, and the driver stops rather than re-pushing so the cause is
fixed first. Watch the `remote publish:` line and the push duration in the
exit log; if a push ever takes longer than the pairs it protects, revisit this
choice with that number rather than an assumption.

### Reading a session that died

`kaggle kernels output` returns a kernel's *files* and has never returned a
log, and the platform keeps **no log at all** for a session it kills - verified
2026-09-21: `kaggle kernels logs` returns a full log for a kernel that
completed and nothing whatsoever for the one that ended ERROR, so a failure has
to be captured while it is happening or not at all.

So the driver **streams and captures**: `kaggle kernels logs -f <kernel>` runs
for the whole wait, writing into `/tmp/act-queue-<queue>-*/live-<queue>.log`,
and the failure path reads four channels - that live capture, the child's stream
teed into the kernel output as `session-stream.log`, the log file in the kernel
output, and `kaggle kernels logs` asked afterwards - keeping the **longest** and
printing the tail **and naming which channel produced it**. A line saying no log
came back from any channel is a real statement about that run, and it is what the
earlier three failures could not distinguish from an empty log.

Two corrections came out of the 2026-09-21 3.50 h loss, and both are worth
knowing when reading a failure:

- **The capture can end long before the run does.** `kaggle kernels logs -f`
  retries a dropped stream five times and then gives up for good; that session's
  capture stopped at **4.1 min** and the session ran another 3.4 h. The capture
  now restarts while the kernel is live, skipping the CLI's replay of lines
already on disk.
- **The longest channel wins, not the first.** The truncated capture was genuine
  and was first in order, so it was returned and three longer channels were
  never read. An account that wins by order is worse than none, because it reads
  like the whole thing.

The launcher also **tees** the child's streamed output into
`/kaggle/working/session-stream.log` as it arrives, and keeps its audit trail -
`session-log.json`, `remote-publish-log.json`, and every seed's summary - in a
`finally`, so a session that dies still has them. Until 2026-09-21 those copies
sat after the child's exit, which is why three failures left an **empty** kernel
output: `kaggle kernels output` is exactly the command that still answers for a
run the platform killed. After any failed session, fetch it and read the last
few hundred lines before re-pushing.

The failure line also states the *run* duration beside the wall duration,
measured from the first `RUNNING` poll: wall time from the push contains the
queue wait, so a session queued for an hour that then died at startup read as an
hour of training.

Each session additionally **downloads the entire published dataset before it
trains** (`restore_latest`, since a queue republishes its whole handoff at
exit). That cost was priced nowhere; it is now logged as
`remote restore: <dataset> downloaded in N min (X GiB on disk)`. At four seeds it
is ~22.7 GiB and it grows ~6.4 GiB per seed, so a session's fixed startup cost
rises for as long as the queue runs - read that line before assuming a slow
start is a hang.

### Is the campaign alive? (one command, read from the durable side)

Both 2026-09 session losses were expensive partly because nothing said the
campaign had stopped. The driver runs on a workstation, so it dies with the
terminal, the shell or the machine - and a watchdog process dies with it, which
is why this is a report rather than a daemon:

```bash
PYTHONPATH=src python scripts/campaign_health.py           # human report
PYTHONPATH=src python scripts/campaign_health.py --json    # for a script or cron
PYTHONPATH=src python scripts/campaign_health.py --queue control
```

It reads the published dataset (the queue's real progress - which seeds are
complete at the *registered* step count, so a part-way snapshot does not read as
done), the kernel's status, the weekly quota, and whether a driver holds the
queue's lock. The verdict is the exit code:

| verdict | exit | what it means |
|---|---|---|
| `RUNNING` | 0 | a session is live; nothing to do |
| `STALLED` | 1 | no session live, quota available, **lock unheld**: relaunch the campaign |
| `WAITING_ON_QUOTA` | 2 | the week cannot fit a window; `--wait-for-quota` covers this, and hours left at the refresh are lost |
| `FAILED` | 3 | the last session ended in ERROR: read the cause before pushing again |
| `COMPLETE` | 4 | every declared seed is published; drive the next queue |

`STALLED` is the one that matters, and it is also the state a driver leaves
behind when it exits 1 (a session failed) or is killed with its terminal - it
stops there deliberately rather than re-pushing a session whose cause has not
been read, so something has to notice. `scripts/run_gate4_campaign.sh` prints
this report at every queue boundary for exactly that reason.
