# TPU null queue — the fire, and what a reader should look for

## What is being run

Section 5.14's **option 2**, taken by the operator on 2026-09-29: the registered null queue's
**twelve declared seeds** (`15024-15035`, one null arm each, the declared 285M / 134M-active
config, 6,000 steps, checkpoint cadence 500, `max_checkpoints` 2, the section 5.47 rebuild
treatment), trained **one seed per chip** on a TPU and published into the null queue's own dataset
root `sid123123234/act-v0-5-2-null`. This is `kaggle_t4.session_argv(QUEUES["null"])` with two
changes, both named in the record the run writes: `--device xla`, and the placement — the session
CLI walks seeds in turn on one device, and at the probe's per-chip rate that is ~43 h against a
20 h week.

**The one field that is not the registration: `dtype: fp32`.** `V0SessionConfig` admits `fp32` and
`fp16` only (the amendment's `bf16` is not a session dtype at all), and fp16 on a TPU needs
`torch.autocast(device_type="xla", ...)` plus a `GradScaler`, which no record in this repository
shows working. fp32 is registered, needs no such machinery, and is the **slower** end of the two
dtypes `tpu_probe` v5 put within 6% of each other. Whether the verdict may *consume* this block as
the registered null clause is a separate registration decision and was **not** taken with the run.

## The fire, and why it is not instantaneous

| | |
|---|---|
| **Kernel** | `sid123123234/act-tpu-null-queue`, `enable_tpu` **on** / `enable_gpu` **off**, private, internet on. Staged entry `run-tpu-null-queue.py` sha256 **`76f8383d89ab3ca3aa8db9f1fc7ad8f99142dbfd0aa83e469945b7c397e2b827`**; `kernel-metadata.json` sha256 **`d63b5a1df7679da03576f5ac25de30e16fd42880b3343d42cba63ea806ded7f4`**. |
| **The command it fires** | `--device xla --seeds 15024-15035 --steps 6000 --workers 8 --fidelity declared --dataset sid123123234/act-v0-5-2-null --output /kaggle/working/tpu-null-queue.json` |
| **The code it runs** | Code dataset `sid123123234/act-lab-code`, downloaded back and checked at **`bf40d53`**: `code/scripts/tpu_null_queue.py` sha256 **`9e744d9c2edce37123782d29bd9b14b7ee1cf042c0ae9077e73ca12619c1e971`** (byte-identical to the tree the local XLA:CPU dry run ran) and `code/src/act/model/checkpointing.py` sha256 **`0bf0ee2e6de98a1c947aef8ade03611d816fafd504aa5f84d6de24a8f37917d2`**. **The dataset has since moved to `d5fdd7b`** (the GPU driver republishes it at session start, and it was relaunched at 11:56Z), and a queued kernel mounts the dataset's **latest version at run time** — so the record's `code_revision` will name whatever is latest when it starts, not the revision it was pushed from. What makes that safe is that the bytes are identical: `diff` between the two commits and a second download both give the **same two sha256s** above. **The runner's hash, not the revision string, is what identifies the code here** — and any edit to `scripts/tpu_null_queue.py` between the push and the start would move that hash silently. |
| **Why it has not started** | Kaggle allows **one TPU session at a time per account**, so the push was refused (`Maximum batch TPU session count of 1 reached`) while the price probe v2 held the slot. **And that refusal is invisible to an exit code:** `kaggle kernels push` prints that message and exits **0** — measured 2026-09-29, and it is what the first version of the slot waiter tripped over (it announced a fired queue for a kernel that did not exist; the account's own kernel list was the check that caught it). Success is now the CLI's own `successfully pushed` line, read by both the waiter and `push_tpu_null_queue.push_kernel` (exit 6). |
| **The other clock** | The GPU driver was **gone** (its lock unheld), so `scripts/run_gate4_campaign.sh` was relaunched at **11:56:14Z** (driver pid **2912294**, `artifacts/gate4-campaign.log`) and sleeps to the **2026-10-03** refresh under `--wait-for-quota`. Its next week runs the primary block's remaining six seeds, then the control queue, then the null queue — which this block makes nearly free, because the CLI labels an already-published seed `skipped_complete`. This run does not touch the GPU week. |
| **Who fires it** | `scripts/push_tpu_null_queue_when_free.sh`, polling every 120 s for up to 24 h, retrying the **staged** push (never re-running the pusher, which would republish the dataset at HEAD and let a later commit become the revision the kernel mounts *and the record names*). On success it arms `scripts/fetch_tpu_session_rehearsal_when_done.sh` with `KERNEL=/OUT_DIR=/PUSHER=` pointing here. |

The price probe is deliberately **not** cancelled to free the slot: it is a declared-shape smoke
test of the same path (one null seed, `declared_session_config`, fp32), and a declared shape that
still cannot run on a TPU is far cheaper to learn from a ~30 s run than from the first wave of a
twelve-seed block.

## What the record has to show

`/kaggle/working/tpu-null-queue.json`, fetched here with the log:

- `seeds_completed: 12` and `errors: []`;
- on **all twelve** arms `search_flops == 0` and `mutation_count == 0` — the null clause's own
  built-in check, so a policy arm cannot be read as this queue;
- `code_revision` matching the mounted revision above;
- `restore` naming whatever a previous attempt already published (empty on a first fire, and a
  seed listed there is one that was **not** re-trained);
- `publish.status: published`. A `failed` push with twelve summaries in the output directory is
  recoverable — `tpu_null_queue.keep_summaries` copies the kilobyte files flat, and they carry
  every number the verdict reads;
- `dtype: fp32`, `fidelity: declared`, `declared_seed: true`.

A partial block is not a lost block: the runner restores the published version and **skips** the
seeds it already carries, so a second fire finishes the block rather than redoing it.

## How it was verified before any TPU hour

Two development seeds on a real `xla` device (XLA:CPU, torch 2.8.0 / torch_xla 2.8.0, throwaway
`/tmp` env), one worker, 20 steps: `errors: []`, `seeds_completed: 2`, both arms
`search_flops 0 / mutation_count 0`, final loss mean **9.0736** (min 9.0733, max 9.0739), **1,041.5
tokens/s** per chip — and the artifact split visible in the tree: the record plus two
`seed-NNNN-summary.json` files (**12 KB**) under `--output`'s directory, the two published seed
trees (**343 MB**) under `--scratch`, with the session directories cleaned after publish. That dry
run is also what caught the earlier runner's `'V0ArmResult' object has no attribute 'tokens'`, which
would have failed every seed of the first wave *after* training it.

The declared shape cannot be rehearsed this way: on this host the session phase's resident set
climbed to **33.27 GB** in 195 s and the process was killed mid-phase, so the queued TPU run is the
declared shape's only instrumentation.

## State at the fire

- **TPU quota:** 16.77 of 20 h remaining, refreshing **2026-10-03**; expected cost ~5-6 h of wall
  clock (twelve seeds, eight chips, two waves — Kaggle bills wall clock). The price probe holds the
  slot.
- **GPU quota:** 1.65 of 30 h, same refresh. The primary block stands at **6 of 12** seeds. The
  driver is relaunched and holding the lock, waiting for the refresh (see the row above).

## What the first fire did, and the fix it forced (2026-09-30)

The kernel sat `QUEUED` from the fetch watcher's first poll, **2026-09-29T19:22Z**, and reached
`KernelWorkerStatus.RUNNING` by **2026-09-30T04:58:18Z**. It was `ERROR` at the watcher's next
poll, **05:00:19Z**, and that poll fetched it. The run itself is **59 s**: the runner printed its
record at `t = 59.06 s` and the kernel wrote `__results__.html` at `t = 81.6 s`. **No seed was
trained**, so the week cost the boot and nothing else.

| | |
|---|---|
| **The record** | `restore.error` = `ModuleNotFoundError: No module named 'kaggle.api'`; `errors: ["restore"]`, `seeds_completed: 0`, `seeds: {}`. Everything else is the registration: `seeds_requested` all twelve, `dtype: fp32`, `fidelity: declared`, `declared_seed: true`, `rebuild_treatment` `{preserve_router_on_rebuild: true, new_expert_initialization: clone}`, `code_revision` `d5fdd7b1b430b81fa09b36536467e13bd2102d53`. |
| **The site** | `act.ops.kaggle_remote.KaggleApiAdapter.__init__`, whose first act is `from kaggle.api.kaggle_api_extended import KaggleApi` - the restore's first call, so the run stopped where section 5.2 says it must (before a token is trained) and did not train over anything. |
| **Why nothing earlier caught it** | The three measurement kernels this project has fired (the probe, the rehearsal, the price probe) never touch the dataset API - they train and write a record - so the TPU image's client had never been imported by any of them. This is the **first TPU run with a remote half**, and its startup restore is the first line that asks for it. |
| **Why the guard missed it** | `KAGGLE_AVAILABLE = importlib.util.find_spec("kaggle") is not None` is **true** on that image: the *name* resolves while `kaggle.api` does not. The adapter's `if not KAGGLE_AVAILABLE` therefore passed, and the failure surfaced as a platform-shaped `ModuleNotFoundError` instead of the module's intended "the kaggle package is required" - a message that names no fix, which is why this took a fetched record to read. |
| **The fix, in the entry** | `push_tpu_probe.KAGGLE_BOOTSTRAP`, composed into this kernel's entry beside the XLA bootstrap and called **before** `runpy.run_path`. It probes the *import* and, when that fails, runs `pip install -q -U kaggle` - **upgraded**, not installed, because on an image that already carries a broken release a plain install reports "Requirement already satisfied" and changes nothing. The run has internet, which is what makes a bootstrap the right shape here rather than a pin. |
| **The fix, in the adapter** | `KaggleApiAdapter.__init__` now wraps that import and, on `ImportError`, raises a `RuntimeError` naming the release defect and `pip install -U kaggle`. The check cannot move to `KAGGLE_AVAILABLE` without importing the package, and the module is deliberately import-time lazy (a fresh-interpreter test pins that importing it leaves `kaggle` out of `sys.modules`), so the adapter's own import is the first place the answer is knowable. |
| **Pinned** | `tests/test_push_tpu_null_queue.py`: the entry carries the bootstrap and calls `ensure_kaggle()` before `runpy.run_path`, and the bootstrap probes the import rather than the name and upgrades rather than installs. `tests/test_kaggle_remote.py`: a present-but-broken `kaggle` is reported as the `RuntimeError` that names the fix, not a bare `ModuleNotFoundError`. |

v1's record and log are under `output/v1/`, because the fetcher writes fixed names into `output/`
and the re-fire overwrites them - the same split, and for the same reason, as
`artifacts/tpu-session-price/`. The re-fire pushes the **already-staged** folder, so the entry
that goes out is the fixed one; the mounted runner (`scripts/tpu_null_queue.py`) is unchanged by
the fix, so the record's `code_revision` naming a later commit would still name the same runner
bytes - which is the property this note's first section already rests on.

## The queue, read on 2026-10-01: still `QUEUED` at 25 h, and what that wait is

The re-fire was pushed **2026-09-30T12:28:44Z**. At **2026-10-01T13:46Z** - **25.3 h** later - the
kernel was still `KernelWorkerStatus.QUEUED`; it had not started. The fetch watcher the re-fire
armed had polled it 720 times, seen nothing but `QUEUED`, and given up at **12:37:06Z**: its bound
was **24 h**, an hour short of the wait, so the watcher that exists to collect the record had quit
before the run it was waiting for had begun.

What was checked, and what each check read:

| what was checked | what it read |
|---|---|
| **the queued version** | pulled back with `kaggle kernels pull`: `ensure_kaggle()` at line 56 and `runpy.run_path` at line 71, `enable_tpu: true`, `enable_gpu: false`. The version in the queue is the fixed one, so the wait cannot produce a second `restore` error. |
| **the account's other kernels** | the probe, the rehearsal and the price probe are all `COMPLETE`; nothing is `RUNNING`. Kaggle allows one TPU session at a time, so a stale session of ours holding the slot would show here - the queue entry is not being blocked by this account's own work. |
| **the queue itself** | `ApiGetKernelSessionStatusResponse` carries `status` and `failure_message` and nothing else: **there is no queue position to read**, and `kaggle kernels logs` against a queued kernel prints nothing (measured). The wait can be observed and dated, not estimated. |
| **the platform's record** | a Kaggle community thread of **2026-09-17**, "Extreamly long TPU queues" - "TPU queues started getting way worse", asking whether capacity had gone down - since deleted; and Kaggle's earlier note on the same class of event: "TPU capacity is temporarily reduced: Queue times for save versions are long ... we appreciate your patience". |
| **this account's own baseline** | the two TPU waits before this one were **~8.7 h** (price probe) and **9.6 h** (null-queue v1: 19:22:23Z -> 04:58:18Z). 25 h is about 2.6x the pool's own recent latency for this account, which is what a worse crunch reads as - not a wedged entry, and not a fault of the push. |

Two things were done about it:

- **The watcher was re-armed, not reset.** `fetch_tpu_session_rehearsal_when_done.sh` was relaunched
  against the same kernel with `POLL_SECONDS=180 MAX_POLLS=1920` (**96 h**), appended to
  `/tmp/tpu-null-watch.log` so the 24 h it had already recorded and the wait that follows stay in
  one file, with a marker line naming the re-arm. Nothing was re-pushed: the entry sits in a
  capacity-short queue, so a fresh version would trade a 25 h-old place for the back of the line -
  and the version already queued carries the fix.
- **The fetch bound is now its own, and longer.** `push_tpu_null_queue_when_free.sh` used to arm the
  watcher with a hardcoded `POLL_SECONDS=120 MAX_POLLS=720`, the slot waiter's own bound. It is now
  `FETCH_POLL_SECONDS`/`FETCH_MAX_POLLS`, defaulting to 180/1920 (**96 h**), because the two waits
  expect opposite things: the slot waiter waits for a slot that frees quickly, the fetch watcher
  waits on a queue that is a platform-capacity event. `tests/test_push_tpu_null_queue.py` pins the
  fetch bound as the longer of the two by parsing both off the shell rather than by pinning either
  to a number.

Still open: the re-fire itself is unobserved. When it starts, the runner needs ~6.19 h at the price
probe's rate and the week has 16.51 TPU-hours left before the **2026-10-03** refresh, so it fits;
the watcher fetches on either terminal state.

## The second fire (2026-10-01T18:06Z): the fix ran, and `pip` replaced nothing

The re-fire was pushed **2026-09-30T12:28:44Z** and left the queue **~29.5 h** later: the
re-armed watcher's poll 81 still read `QUEUED` at **17:51:22Z**, and its next poll,
**18:06:25Z**, found `KernelWorkerStatus.ERROR` and fetched. The run itself is **~85 s**
(`torch_xla: already-present` at t=55.35 s, the record at t=69.1 s, `__results__.html` at 84.6 s)
and its record is v1's exactly, down to `errors: ["restore"]`, `seeds_completed: 0`, `seeds: {}` -
the only difference is `scratch`, `/tmp/act-lab-null-queue-68154a5f47` against v1's
`fa378f39ab`. The `restore.error` is the same `ModuleNotFoundError: No module named 'kaggle.api'`.

So the entry *did* carry the first fix and the fix ran - and this is what it printed:

    pip installed kaggle but kaggle.api still will not import
    pip kaggle failed (0):
    WARNING: Running pip as the 'root' user can result in broken permissions ...
    [notice] A new release of pip is available: 25.0.1 -> 26.2.1
    [notice] To update, run: pip install --upgrade pip
    kaggle: unavailable

`pip` exited **0** after ~12 s having downloaded nothing and printed only its own warnings: on an
image whose metadata already names the current release, `-U` answers "Requirement already
satisfied" and replaces no file. Two mechanisms are consistent with that output and both are the
same fix's business: the in-place upgrade was a no-op, and the failed probe's `kaggle` was left in
`sys.modules`, where a cached module wins over anything a later install writes.

### The fix, and how it was read before it was fired

`KAGGLE_BOOTSTRAP` no longer upgrades in place. It installs a copy into a directory the entry text
owns (`/tmp/act-kaggle-client`) with `pip install -q --ignore-installed --target`, puts that
directory **first** on `sys.path` (a regular package there beats the image's broken copy, which the
upgrade could not replace), and drops every `kaggle*` entry from `sys.modules` plus
`importlib.invalidate_caches()` before each probe - including the retry, which would otherwise
import the copy the failed probe cached. A target copy that still will not import prints
`sys.path[:3]`, and a failed install prints what the name resolved to, so the next log names the
site rather than the symptom.

Read locally before firing, with the restaged entry's own `ensure_kaggle` extracted by `ast` and
run under the project venv (Python 3.11, no `kaggle` installed at all):

| scenario | what the bootstrap returned | what the import then resolved to |
|---|---|---|
| a broken `kaggle` package shadowing on `PYTHONPATH`, already imported | `installed:/tmp/act-kaggle-client` | the target's `kaggle/__init__.py` |
| a namespace `kaggle` (no `__init__.py`) shadowing, already imported | `installed:/tmp/act-kaggle-client` | the target's `kaggle/__init__.py` |
| a working copy already on `sys.path` | `already-present` (no install) | the working copy |

### The third fire

The entry was restaged (**sha256
`258d6fe430a62769127ef14810c8aec9bea813b197ca1b5bde32ee2389324e19`**; `kernel-metadata.json`
unchanged at `d63b5a1d...`) and pushed **2026-10-02T15:18:37Z** by the slot waiter's attempt 1 (the
price probe is `COMPLETE`), with the fetch watcher armed at the 96 h bound (pid **58331**). At
**15:18:38Z** the kernel read `QUEUED`; the two waits before this one were ~9.6 h and ~29.5 h, so
the bound has room, and the ~6.19 h the block needs at the price probe's rate still fits the
16.49 TPU-hours left before the **2026-10-03** refresh. The first re-arm's watcher (24 h bound,
which expired about five hours before the second fire started) has its full account preserved at
`/tmp/tpu-null-watch.log-20261001`.

v2's record and log were fetched into `output/` at **18:06:25Z** on 2026-10-01, so `output/v1/`
and `output/` are the two fires' evidence, now committed. The adapter half of the 2026-09-30 fix
(`KaggleApiAdapter.__init__` naming the defect) still needs a code-dataset republish to reach a
kernel; it is not on this queue's critical path, whose entry now makes the import work before the
runner asks - and the mounted dataset revision the record names is unchanged (`d5fdd7b`).

## The third fire (2026-10-02T15:18Z): the platform never ran the script

The entry was byte-identical to the fixed staged one (`258d6fe4`, pulled back and diffed), and
the run left the queue after **~29.5 h**: `QUEUED` at the watcher's poll 181 (**2026-10-03T00:22:48Z**,
the moment the account's GPU week refreshed), `RUNNING` from poll 206 (**01:38:09Z**), and `ERROR`
at **04:41:45Z** - **~3.0 h** of wall clock, which is exactly the **3.15 TPU-h** the quota moved
by (3.51 used on 2026-10-02T15:03Z -> 3.15 used on 2026-10-03, i.e. **-0.36 h**, a refund: the
refresh zeroed the counter and the run was billed before it).

What it produced: **nothing.** `kaggle kernels output` returned an **empty listing** and the fetch
kept a **0-byte** log; `kaggle kernels logs` prints nothing; and
`ApiGetKernelSessionStatusResponse.failure_message` is `''`. A run of this kernel that executes
even one line writes a record (`tpu_probe._save` runs before the restore) and its stdout carries
the bootstraps' own prints, so an empty output plus an empty failure message is the platform's
**provisioning** failure, not the runner's: the container never ran the script. Two earlier
measurements name the same class - the note in `restore_latest` records "a session ended ERROR
after 1.32 h with no log and no output" on 2026-09-21, and that is on a GPU kernel.

There is nothing to fix locally for this one. The queued version was verified byte-identical to
the staged entry, the mounted code dataset still carries the runner at sha256
`9e744d9c2edce3...`, and a re-fire costs a queue wait rather than a diagnosis. It was re-fired:
see below. The watcher that fetched it (24 h bound on the *fetch* side was 96 h; it fetched at
04:41:45Z) has its full account at `/tmp/tpu-null-watch.log-20261003`.

## The fourth fire (2026-10-03T16:27Z)

Pushed by `push_tpu_null_queue_when_free.sh` at **16:27:52Z** (attempt 1; the price probe is
`COMPLETE`), pushing the **already-staged** folder - the entry that goes out is the same
`258d6fe4` bytes - with a fresh fetch watcher (pid **384604**, log `/tmp/tpu-null-watch.log-v3`,
96 h bound). The kernel read `QUEUED` at the first poll.

**The TPU quota at this fire:** **16.85 h remaining** of 20 (the 2026-10-03 refresh has passed;
the next is 2026-10-10), and the block needs ~6.19 h at the price probe's rate. The two waits
before the third were ~9.6 h and ~29.5 h; this one is against a week that just refreshed, which
is the same shape as the second fire's wait.

**The evidence layout:** `output/v1/` is the first fire's record and log, `output/v2/` is the
second fire's (record + log, recovered from the commit because the third fire's fetch overwrote
the log with its empty one), `output/v3/` is the third fire's and `output/v4/` the fourth's -
each an empty log and no record, which is their whole content - and `output/` itself holds the
newest fetch's result.

### What the run left (2026-10-04)

The kernel held `QUEUED` through the watcher's poll 231 (**04:00:30Z**); polls 236 and 241
(**04:15:34Z**, **04:30:37Z**) were answered with a permission error that recovered on its own
(the watcher's log carries both); poll 246 (**04:45:40Z**) and every printed poll through 271
(**06:00:55Z**) read `RUNNING`; the check at **06:12:57Z** read `ERROR` and fetched. That is
**~13.8 h** in the queue from the push, about **1.5 h** of it `RUNNING`, for a week charged
**~1.84 TPU-h** and **nothing trained**.

The fetch kept **one file, the 0-byte log**: `kaggle kernels output` listed nothing else,
`kaggle kernels logs` prints nothing, and the status response's `failure_message` is `''`. A
run of this kernel writes its record *before* its restore and prints the bootstraps' own lines
to stdout, so an empty output beside an empty failure message is the third fire's
**provisioning** signature again - the container never ran the script - not a runner failure.
The queued version was the staged `258d6fe4` entry, so there was nothing to fix locally either;
the queue was re-fired, and this run's empty log is archived at `output/v4/` (its watcher's
full account: `/tmp/tpu-null-watch.log-v3`; push waiter: `/tmp/tpu-null-push-watch.log-v3`).

## The fifth fire (2026-10-04T12:41Z)

Pushed by `push_tpu_null_queue_when_free.sh` at **12:41:44Z** (attempt 1; the price probe is
`COMPLETE`), pushing the **already-staged** folder - the same `258d6fe4` entry bytes and the
same `d63b5a1d` metadata - with a fresh fetch watcher (pid **3799423**, log
`/tmp/tpu-null-watch.log-v4`, 96 h bound). The kernel read `QUEUED` at the first poll
(12:41:44Z).

**The TPU quota at this fire:** **~15.0 h remaining** of 20 (the 2026-10-03 refresh has passed;
the next is 2026-10-10), against the ~6.19 h the block needs at the price probe's rate. The
fourth fire's provisioning failure spent ~1.84 h of that week without training a seed; this
fire is the same re-fire, and its watcher fetches on either terminal state at the 96 h bound.

### What the run left (2026-10-05)

The watcher (pid 3799423) fetched when the kernel reached `ERROR`, and the harness read the log
as **no seeds**: `--- null queue / no seeds: the run left no per-seed entry to read`. That is
neither the third and fourth fires' provisioning signature nor a runner failure: the container
did run the script and did flush its own stdout in order, and this log is 46 events (40 stdout,
6 stderr) rather than an empty one. It names where it stopped, and the fix of `f7ee2eb` is
visible in it:

| | |
|---|---|
| `torch_xla: already-present` | 60.8 s |
| `kaggle: import failed; the name resolves to None` | 60.8 s |
| `kaggle: installed:/tmp/act-kaggle-client` | 82.1 s |
| `Authentication required to call the Kaggle API.` (+ the CLI's login / OAuth / `KAGGLE_API_TOKEN` / `~/.kaggle/access_token` help) | 82.1 s |
| `null queue runner: /kaggle/input/act-lab-code/code/scripts/tpu_null_queue.py` | 83.5 s |
| `Authentication required to call the Kaggle API.` (the same help text) | 86.2 s |
| `[NbConvertApp] Converting notebook __script__.ipynb to html` | 96.5 s |

So the install reached the container and its own report names the copy it owns - the import gap
that made the first two fires die at `restore` is closed - and the failure has moved one step
along the same path: the client the entry now owns refuses to **authenticate**, asking for a
credential in `~/.kaggle/access_token`, `KAGGLE_API_TOKEN`, or an interactive `kaggle auth login`,
none of which this container supplies. `tpu_null_queue.py` stops there rather than entering the
restore, so it writes no record and the harness can only report that there was nothing to read.
The next fix is therefore on the credential side - hand the container's own credentials to the
client it now owns - not another install. The whole run was **86 s** of wall clock and the week's
TPU line has moved 15.0 -> **14.99 h**; nothing was trained, and the null queue still stands at
**zero seeds published**. This run's log is what `output/act-tpu-null-queue.log` holds now (the
newest fetch's place, per the layout above).
