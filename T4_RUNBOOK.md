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
candidate policy; the declared queue is **15000-15011** (the registered
9000-9011 was spent by the §5.3 diagnostic); a **pool-matched loss-blind control
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
   `<your-account>/act-v0-5-1`) and use it identically in every session. The
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
   §5.5 change 4 re-reserves the queue to the unused block **15000–15011**,
   which is now the code default (`v0_runner.DECLARED_SEEDS`), so the §2
   command passes `--seeds 15000-15011` explicitly and never the old block.
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
and the default budget covers the six sessions the primary queue needs:

```bash
python scripts/kaggle_queue.py --queue session
python scripts/kaggle_queue.py --queue control --max-sessions 8
python scripts/kaggle_queue.py --queue verdict          # after the three queues
python scripts/kaggle_queue.py --queue preflight        # one-shot diagnostics
python scripts/kaggle_queue.py --queue dtypeprobe       # fp32 vs fp16 rate (T4)
```

`preflight`, `dtypeprobe` and `verdict` are one-shot jobs rather than queues:
they are pushed once and reported, with no re-push loop and no quota gate (the
verdict runs on CPU and bills nothing). All three run the mounted code
dataset, so the driver publishes it if it lags the tree before pushing - a
one-shot that ran older code would report on a run it did not measure. `--queue verdict` returns the registered
section 4 decision as its exit code - **0 PASS, 1 FAIL, 2 PASS with
qualification, 3 no verdict produced** - so the last step of the run is
scriptable, and it prints the verdict artifact. The kernel itself always exits
0 whatever the decision, so the code travels in `v0-5-1-verdict.json`
(`exit_code`) for the driver to read back.

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
Over the ~12-session, ~86 GPU-hour plan against a 30 GPU-h/week quota that is
the difference between waiting a day and losing the quota. An unreadable quota
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
   line regardless of devices, measured above) plausibly halving the ~86
   GPU-hour plan; that is a *schedule* change, so it re-runs the section 5.2
   benchmark and re-derives `full_pairs_per_session` rather than being assumed.
7. **The kernel runs the mounted dataset, not this working tree.** The staged
   kernel body is embedded from the tree, but `src/`, `scripts/` and `tests/`
   are copied from the attached `act-lab-code` dataset, so a dataset that lags
   the tree runs different code than the run claims - and the handoff manifest
   records `code_revision`, so the provenance would be wrong too. Measured
   2026-09-15: the dataset was a commit behind, and every fix in between
   (including the guarded handoff publish) was absent from the session that was
   running. `kaggle_queue.ensure_code_dataset` runs before any quota is spent:
   it pins the dataset to `git rev-parse HEAD`, refuses outright on a dirty or
   non-git tree (a tree that is not a clean commit has no reproducible
   revision), and republishes when the published copy is not at that revision
   **or does not carry the package tree**. The marker alone is not evidence -
   it is a loose top-level file, so it survives an upload that drops every
   subdirectory while the tree does not. So the tree travels as one archive
   (`code.tar`, whose internal paths the service restores), the staging marker
   is written into the staging folder and never the repo (a marker left in the
   repo is untracked, which makes the next run refuse the tree as dirty), and
   the read-back is polled, because a download issued immediately after a
   version push can serve the previous version and look like a rejected
   publish.
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
| `measured_tokens_per_second` (fp32) | 10,000 | **1,958.7** |
| `dtype_decision` | fp32 | **fp16** (below the 2,500 floor - the registered contingency) |

**The schedule is derived from the rate of the dtype the session trains.** The
fp32 number above is the pre-committed rule's *input*; it is not what the arms
run at. Measured on the T4 (`--queue dtypeprobe`, 2026-09-15):

| dtype | tokens/s | `pair_time_hours` | `full_pairs_per_session` | `sessions_for_primary` |
|---|---|---|---|---|
| fp32 | 2,066.3 | 3.404 | 2 | 6 |
| **fp16** (the decided dtype) | **4,694.9** | **1.554** | **5** | **3** |

Scheduling from the fp32 rate while the sessions train fp16 is 2.27x too
pessimistic: each session's deadline rule stops with usable clock left and the
run buys about twice the sessions it needs. The consequence is a quota plan,
not a knob change - the same seeds, the same steps, the same probe cost, and no
pre-registered criterion or verdict, since section 5.2 makes dtype an execution
detail. At 1.554 h per pair the primary queue is ~19 GPU-hours, the 2b control
queue ~19 more, and 2c ~0.3 - about 38 GPU-hours against the free tier's 30
GPU-h/week, i.e. roughly two weeks of sessions instead of three.

Pass `--hours 9` (the registered usable window): at the fp16 rate the deadline
rules then place five completed pairs per session. The benchmark now measures
the decided dtype and records both numbers (`schedule_tokens_per_second`,
`tokens_per_second_by_dtype`), and the launch command schedules from the former
while still enforcing the pre-committed rule against the latter.

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
| `schedule.full_pairs_per_session` | pairs a 12-hour clock fits per the deadline rules |
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

```bash
PYTHONPATH=src python -m act.training.cli --mode pair-session \
  --device cuda \
  --hours <remaining-clock-hours> \
  --seeds 15000-15011 \
  --probe-candidate-policy spread \
  --kaggle-dataset <owner>/<slug>
```

- `--hours`: the remaining session clock at entry — Kaggle's 12-hour GPU limit
  minus boot and restore time. The deadline rules only start pairs that fit
  `pair_time + 45-min publish reserve`.
- `--seeds 15000-15011` and `--probe-candidate-policy spread` are **not
  optional**: they are the adopted §5.5 protocol. The code defaults remain the
  registered `15000-15011`/`prefix` pair, so the `--seeds` flag is a
  restatement rather than a change, and the policy flag is the adopted one.
- Everything else defaults correctly: benchmark artifact
  `artifacts/v0-benchmark.json`, `--probe-batch-count 1` (the primary config),
  dataset root `artifacts/datasets`, session dir `artifacts/v0-sessions`.
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
  --seeds 15000-15011 \
  --probe-candidate-policy spread \
  --probe-selection random \
  --dataset-root artifacts/datasets-control \
  --session-dir artifacts/v0-sessions-control \
  --kaggle-dataset <owner>/<slug>-control
```

- `spread` is required here too. The clause isolates the *decision rule*; if
the control ran a different candidate pool, a null result would be
uninterpretable.
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
AFTER the primary and §2b queues (it is ~90 s of GPU per seed plus session
overhead):

```bash
PYTHONPATH=src python -m act.training.cli --mode pair-session \
  --device cuda \
  --hours <remaining-clock-hours> \
  --seeds 15000-15011 \
  --session-arms null \
  --dataset-root artifacts/datasets-null \
  --session-dir artifacts/v0-sessions-null \
  --kaggle-dataset <owner>/<slug>-null
```

- No probe/policy flags are needed (the null arm is policy-free by
construction); `--seeds` and the roots must still be explicit, and the
null queue must never point at the primary or control roots.
- The published seed summaries carry `null_final_loss` and the zero-probe /
zero-mutation counters the §5.14 built-in checks read.

---

## 3. Running the queue to completion

The queue is 12 declared seeds (15000–15011), one atomic pair each. Schedule
expectations come from the benchmark artifact (`sessions_for_primary`,
`sessions_total`); the plan's own arithmetic already includes the 45-minute
publish reserve and the 20 GB working-dir ledger.

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
| 0 | `PASS` | cost ≤ 5%, inside the [0.5, 2]× band, harm guard clear, **and the loss-blind control did not clear its own gate** → **Gate 4 authorized** (`gate_4_authorized: true` requires 12/12 seeds) |
| 1 | `FAIL` | primary cost > 5% → stop; §5.1 fails as registered |
| 2 | `PASS_WITH_QUALIFICATION` | cost ≤ 5% but outside the band → re-derive the predicted C **before** Gate 4; no knob re-tuning against the measured C |
| 2 | `PASS_COST_ONLY_HARM_GUARD_REJECTS` | harm guard hit (adaptive worse in ≥ 10 of 12 pairs) → do not proceed to Gate 4 |
| 2 | `PASS_COST_ONLY_POOL_MATCHED_CONTROL_PASSES` | the loss-blind arm won ≥ 10 of its 12 pairs, i.e. the pool alone clears the gate → the cost PASS stands but the quality inference does not; **never** treat this as "the controller wins" |
| 2 | `PASS_COST_ONLY_CONTROL_MISSING` | no `--control-root` given → the clause could not be evaluated; running the control is part of the protocol, so Gate 4 is withheld |
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
| Exit publish fails (`remote_publish.ok=false`) | Never fatal by design. The local handoff under `artifacts/datasets/act-checkpoints/v0-5-1/` is intact; the next session's publish re-pushes it. Check `remote-publish-log.json` beside the dataset root. |
| Two sessions of one queue running (burn at 2x) | They will not stop each other: `kaggle kernels delete <owner>/<slug> --yes`, then push once (§0.5 gotcha 5). Nothing is published by the abandoned version. |
| Local per-pair publish fails (`action: publish_failed`) | The pair is **not** recorded complete and the queue stops at that seed, so the session can still exit-publish the seeds that did complete (it would otherwise abort before publishing anything). Fix the substrate - disk space, permissions - and relaunch; the failed pair restarts from scratch. With the handoff roots on the overlay (§0.2) this should not fire. |
| Restore fails at session start | Fix credentials / `owner/slug` / connectivity; relaunch. Nothing ran, nothing published. |
| Session died mid-pair (reboot, timeout) | Relaunch the same command. The pair restarts from scratch; the stale `in-progress` version is ignored. |
| Kaggle auto-extracted archives | Expected and handled: the wrapper stages per-seed tars with `seed-NNNN/...` internal paths (live-verified bit-identical round trip). Restore also extracts blobs itself as defense-in-depth. |
| Verdict reports stray complete seeds | They are excluded from the aggregate automatically. For cleanliness, delete stray seed directories from the dataset root or use a dedicated root for smoke tests. |
| Benchmark artifact missing/stale | Re-run step 1. The launch command refuses a hand-edited or stale artifact by design. |
| Verdict says `PASS_COST_ONLY_CONTROL_MISSING` | The §2b control session has not run (or `--control-root` was omitted). Run §2b, then re-judge. Do not authorize Gate 4 on the strength of the primary run alone. |
| Verdict says `PASS_COST_ONLY_POOL_MATCHED_CONTROL_PASSES` | Record the cost PASS **and** the withheld Gate 4 as the result. This is a finding about the search design, not a failed run: do not tune the probe, the cap or the pool and re-run, and do not add a new seed block to try for a different answer. |

---

## 6. Artifact trail (what to keep per session)

- `artifacts/v0-benchmark.json` — the measured throughput and dtype decision.
- `artifacts/v0-sessions/session-log.json` — queue, per-pair actions
  (completed / interrupted / skipped), FLOPs counters, `code_revision`.
- `artifacts/datasets-control/` + `artifacts/v0-sessions-control/` — the §2b
  pool-matched arm (same seeds, `probe_selection=random`), and its own
  `session-log.json`. Keep both roots: the verdict's clause reads them, and
  every seed summary records its `probe_candidate_policy`/`probe_selection`, so
  which regime produced each number is auditable without this document.
- Dataset versions — every `complete` (and forensic `in-progress`) version of
  `<owner>/<slug>`, each with a checksummed manifest.
- `v0-5-1-verdict.json` — the final machine-checked §5.1 verdict.
- `remote-publish-log.json` — the exit-publish audit trail.
