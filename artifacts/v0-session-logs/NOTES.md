# Campaign session logs - the evidence behind the exit-publish numbers

## The two fires that found the size wall (kernel v18, 2026-10-05 20:05Z; v19, 2026-10-06 10:45Z)

These are the first two sessions that publish **after every pair** (commit `e60b3b6`), and
they narrow the failure that cost the previous four sessions. Version 18 pushed the **7-seed
state and it landed** - the first per-pair push to land, and the reason the block moved
6 -> **7 of 12** - then died inside the push that would have made it **8**. Version 19
re-trained the same eighth seed and died inside the same push. Both kernel logs are empty
(`[]`, 3 B), so the only account is the driver's live capture, committed here beside them.

| file | what it is | sha256 |
|---|---|---|
| `session-session-20261006T001451Z-live.log` | v18's live capture, 122 lines, ending on the kaggle version warning after `seed 15031: pair complete` | `2d45a15278324a70f22ae8a7a0eb7f80359e06b3c9d6d9de7b5761540cfe1fec` |
| `session-session-20261006T001451Z-kernel-log.json` | v18's platform log: **zero events** | `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570` |
| `session-session-20261006T125450Z-live.log` | v19's live capture, 99 lines, same last line | `75466821abf41cbd308d6fd3956ab699061d2542dd97ed7fc301513157d7479a` |
| `session-session-20261006T125450Z-kernel-log.json` | v19's platform log: **zero events** | `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570` |

| | v18 | v19 |
|---|---|---|
| Window declared | 6.43 h (quota 6.91 h at entry) | **2.84 h** (quota 2.91 h left; the driver fitted the window to it) |
| Startup | restore **37.85 GiB in 4.8 min**, 6 seeds `[15024-15029]` | restore **44.13 GiB in 9.7 min**, 7 seeds `[15024-15030]` |
| Benchmark / pair | 1,248.1 tokens/s, pair **1.62 h** | 1,471.6 tokens/s, pair **1.40 h** |
| Code revision | `e60b3b6` (per-pair publish) | `81be2bf` |
| Pairs it trained | 15030 (`10.397789` / `10.397594`) and 15031 (`10.398133` / `10.398239`) | 15031 again (`10.398096` / `10.398072`) |
| What landed | `remote publish: ... version pushed (7 seed version(s)) in 16.9 min`, files created 2026-10-05T22:20:43-22:22:11Z | nothing |
| How it ended | `ERROR` at **00:14:53Z**; last line the kaggle 2.0.2 warning, i.e. inside the 8-seed push | `ERROR` at **12:54:51Z** after **2.15 h** and **1.99 GPU-h**; same last line |

The two captures are the same shape one seed apart, and that is the finding. The dataset's
last version is still the 7-seed one (`lastUpdated` **2026-10-05T22:17:32Z**, files minted
22:20-22:22Z); nothing from either 2026-10-06 fire exists as a version. So the kill happened
**before the upload reached finalize**, not after a version was created - which is why the
service itself has no failure record to read either. **The push that lands carries 7 seeds /
44.13 GiB, and the push that kills the container carries 8 / ~50.4 GiB.** The three earlier sessions (v15-v17) each pushed **9** seeds / ~56.7 GiB at exit
and died there. No Kaggle document in this tree names a per-version size limit, so what is
recorded is the bracket, not a cause: **a version at 44.13 GiB lands and one at ~50 GiB does
not.** Twelve seeds is ~75.6 GiB, so the registered block cannot be published as one dataset
version at any window length - the wall is structural, which is why the driver was right to
stop rather than re-fire.

The cost is now seed-specific. **15031 has been trained six times** - 2026-10-03T2236Z,
2026-10-04T1913Z, 2026-10-05T0351Z, 2026-10-05T1813Z, 2026-10-06T0014Z and
2026-10-06T1254Z - and every one of the six pairs completed and was discarded, because the
push that would make it durable is the one that dies. Its two arms sat on the uniform floor
in all six: random `10.397819`-`10.398514`, adaptive `10.397842`-`10.398252`, which is the
seeded block's expected scale, not a result. The published block is unchanged at
15024-15030.

## The session that finally advanced the queue (kernel v11, 2026-09-27)

The first session in four whose restore *worked*: `act-t4-session` version **11**, declared
window **6.5 h**, code dataset `80699f945648` (the restore fix), quota 6.57 h at entry and
1.65 h after - **4.92 GPU-h** spent for **three** published pairs (15027, 15028, 15029), the
queue moving **3 -> 6 of 12**. It ran unattended for its last hour: the workstation rebooted
at **15:46:08Z** while the driver was polling, and the session finished on Kaggle at ~15:54Z
and published at 15:57-15:59Z with nobody watching. The files here are the evidence fetched
afterwards (2026-09-27 17:0xZ, `kaggle kernels logs` + `kaggle kernels output`, taken while
version 11 was still the latest version - the next push replaces them).

| file | what it is | sha256 |
|---|---|---|
| `session-15027-29-log.json` | the platform's execution log for kernel version 11: **73 events** (6 on stderr), 11,406 bytes, `time` in seconds from kernel start | `ec4ab749c9ed30bac1290957abf8a33aab6cb851794c28062c2b98fba24501a0` |
| `session-15027-29-session.json` | the runner's own `session-log.json` (the queue loop's account: which seeds were skipped, which completed, per-pair losses and FLOPs) | `76b48c920333a5f134b81ce1c946654d295b10e3abcd4d5e44cae0ec8ac4d3cd` |
| `session-15027-29-publish.json` | the exit-publish receipt: `status=published`, `seeds_published=6`, `error=null` | `2f391ef9500a975d6af593827add2629ac19ef2c55a5305fa7efe33c2b3c0992` |
| `session-15027-29-benchmark.json` | the throughput benchmark that set the dtype (`fp16`) and the schedule (5,182 tokens/s, pair 1.4175 h) | `af4c10df1940aa0e17e2f2d5bc158ab27e9ef63ab6672fbeb1ada999b96f31fb` |
| `session-15027-29-child-stream.log` | the child's tee from the kernel output (`session-stream.log`): the restore line, all three pair completions, the stop and the publish | `6b34370f4a94835eb1d23abd7bf16e3c02b394cb5023e6c8e8844f898104dd17` |
| `session-15027-summary.json` | seed 15027's published record (re-trained this session; the window that produced it is the one the 403-era sessions had left `in-progress`) | `510bfa6f9de53ea8ba8fe482cda66bbae508ec92374b6a37177b95fa95eff861` |
| `session-15028-summary.json` | seed 15028's published record | `962bdcc11d200bece9fd4d7e51b7f009fc1c0e999832af095058b0cfef438b54` |
| `session-15029-summary.json` | seed 15029's published record | `d02d650b0ae3f23cd2761d5fe1658addcbba27379fbc484bbe1c05ffa0d55bb22` |

There is **no live capture** for this session, unlike v7's: the driver's tailer writes into
its own `mkdtemp` work directory, the reboot destroyed that directory, and the driver itself
died with it - so nothing called `keep_session_evidence` either. What survives is what the
platform kept: the log above and the version's output files. (The three recorded line types
that matter - `remote restore: ... downloaded in 8.7 min (25.23 GiB on disk)`, `restored 4
seed version(s) ... [15024, 15025, 15026, 15027]`, `remote publish: ... (6 seed version(s))` -
are in the tee.)

**What the session said, in its own lines.** The restore is the fix behaving: a 25.23 GiB
download in 8.7 min, then three seeds recognised as done and skipped
(`seed 15024/15025/15026: complete version exists, skipping`), the block treatment confirming
(`preserve_router_on_rebuild=True, new_expert_initialization='clone'`), then
`seed 15027: starting pair (remaining 6.23 h fits pair 1.42 h + 45 min reserve (rule 1))`,
pairs at 15027-15029, `seed 15030: remaining 1.82 h does not fit pair 1.42 h + 45 min reserve
(rule 1)`, `session done: 3 pair(s) completed, 1.82 h remaining`, and the publish. Compare
kernel v10, whose *whole* account was "no remote version ... available yet; training from
scratch": same command, same seeds, opposite outcome.

**What the three pairs measured.** Every arm sits on the uniform-vocabulary floor
`ln(32768) = 10.3972` nats, as the declared block's synthetic stream implies - adaptive and
random differ by 1e-4, and the sign changes seed to seed:

| seed | random | adaptive | `random - adaptive` |
|---|---|---|---|
| 15027 | 10.398536682 | 10.398457527 | `+7.9e-05` |
| 15028 | 10.398133039 | 10.398347378 | `-2.1e-04` |
| 15029 | 10.397966623 | 10.398276329 | `-3.1e-04` |

Across all six published seeds the record is **2 wins and 4 losses** for the adaptive arm, and
every seed's `predicted_cost_ratio` is identical to the benchmark's own
`predicted_probe_cost_percent.cap4_1batch` (`0.000941636252059042` = **0.0942%**, against the
registered 5% criterion and its `[0.047%, 0.189%]` band). The harm guard needs the adaptive
arm to lose **10 of 12** to reject, so at this floor - where a 1e-4 difference is noise - the
registered clauses have almost nothing to read; that is the recorded property of this block,
not a new finding.

**The provenance finding this fetch exposed.** The published records for **15024-15026 are
not the ones kernel v7 published**, and the dataset cannot say whose they are:

| | |
|---|---|
| **Measured** | The live dataset's `seed-15024/reports/seed-summary.json` checksums to `a07be977...` (its own manifest) and matches the copy in v11's output; the copy committed here from v7's output is `d27882c8...` and a **different training**: mutations at `L0E1`/`L11E4` against v7's `L10E1`/`L5E7`, architecture hash `64cd541996cd4609` against `291178c1d5309057`, adaptive loss `10.399189949` against `10.398051500`. Same for 15025 and 15026, whose manifests carry `8e15e277...` and `483394225...` rather than the committed copies' checksums. |
| **Why v11 did not write them** | Its own session log says `skipped_complete` for 15024-15026 and `completed` for 15027-15029, and the three records it *did* write match its output byte-for-byte. So the replacement happened before v11's restore. |
| **Who wrote them: v9 or v10, and that is as far as the artifact goes** | Only **v9** and **v10** ran between v7's publish and v11's restore, and both are documented as having restored nothing, retrained the block from its first seed and published over it. Which of the two wrote these three records is **not recoverable**: v9 ran the *same* revision as v7 (`2c0cf4ebfa61`), so if a declared-scale fp16 seed reproduces exactly then these are v10's - and v10 did run a **different tree**, which the driver's own log shows (`republishing from 1b0f210b7b7f`, naming `src/act/model/moe.py`), a revision whose `src/act/model/*` differs from both v7's and v11's. But exact reproduction at declared scale is **not established here**: the measurements are at development fidelity (the churn arm bit-identical across three pair roots on 12/12 seeds, and an arm's loss bit-identical with and without its co-running partner), and nothing in this tree pins cuDNN (`torch.backends.cudnn.deterministic` is never set), so a fp16 T4 run is not guaranteed bit-reproducible. So the honest statement is the pair of candidates and the reason they cannot be told apart - which is the same hole as the `unknown` below, seen from the other side. |
| **What is *not* recorded anywhere** | Every published manifest says `code_revision: "unknown"` (measured: all six downloaded), and the v0 checkpoint payload carries no revision and no timestamp either - it is `version 5` + `model`/`optimizer`/`rng`/`scaler`/`training`/`v0_session`, nothing dated. So the dataset itself cannot attribute a seed to the code that trained it - which is exactly why the row above can name the pair of candidate sessions and not one of them. |
| **Why it says unknown** | `act.training.cli._git_revision()` asks `git rev-parse`; a kernel imports `src/` from a **mounted dataset with no `.git`**, so the answer was always the literal `unknown`. The dataset does carry the commit - `code/revision.txt`, measured here as `80699f945648021b0111c9005c79b9b7048abb66` - beside the package tree, and nothing read it. It now does (the fallback added 2026-09-27), so the remaining primary seeds (15030-15035) and both later queues record the revision that trained them. |
| **What it changes about the block** | Nothing about its criteria - each pair is internally consistent and every seed ran the declared configuration (v11's log confirms the block treatment matches: `preserve_router_on_rebuild=True, new_expert_initialization='clone'`) - but the first three records cannot be shown to come from the same tree as the last three, and the control/null queues will run under the current revision, so the verdict's cross-queue pairings (adaptive from the primary record, null from the null queue) pair records whose revisions are, at best, only partly recorded. Nothing here measures a difference; what is measured is that 15024-15026 changed hands between v7 and v11. Re-training them under the current revision would make the block homogeneous for ~4.3 GPU-h of quota; it is **not** done, because re-running published seeds to taste is what the pre-registration forbids and the published block is the block. |

## Why there is no kernel-version-10 log in this directory (2026-09-27)

The session that stalled the queue - `act-t4-session` version 10, which completed
2026-09-27 and advanced the queue by nothing - left an account that was read, quoted into
`docs/IMPLEMENTATION_STATUS.md`, and then **destroyed by the next push**: `kaggle kernels
logs` serves the log of the *latest* version only, and fetching `…/act-t4-session/10` after
version 11 was queued returns an empty stream. The two lines that explain three sessions of
lost quota are therefore quoted in the status entry rather than committed here:

```
Dataset URL: https://www.kaggle.com/datasets/sid123123234/act-v0-5-2        (t = 223.0 s)
no remote version of sid123123234/act-v0-5-2 available yet; training from scratch (t = 223.1 s)
```

Two things follow, and both are now in the tooling rather than in a habit. First, a download
failure that *looks* like "nothing published yet" is classified against the owner's own
dataset listing instead of trusted (a missing dataset and an unserved one both answer 403),
retried, and then **stopping** the session rather than re-training published seeds
(`src/act/ops/kaggle_remote.py`; the runbook's own "fail-fast" rule for a restore). Second,
the driver keeps every session's own account at session end - the kernel log blob, the live
capture and the runner's `session-log.json` - in this directory
(`kaggle_queue.keep_session_evidence`, 2026-09-27), because a session that succeeds *and*
advances nothing is not a failure path and nothing else fetches its log. The next synthetic
session that ends this way therefore leaves `session-<queue>-<stamp>-kernel-log.json` here,
and this paragraph is the reason that file exists.

## What `session-15000-03-log.json` is

The platform's own log stream for `sid123123234/act-t4-session`, the Gate-4 primary
session that trained seeds **15000-15003** and **reached its exit publish**. Fetched
2026-09-22 through the driver (`kaggle kernels logs`), copied here byte-identical
(`sha256 8a6adf2dfef7209470f99a044defa1dd7d716d86074011d5c11fa7b067b528aa`, 11,637 bytes,
75 lines). It is a JSON array of `{stream_name, time, data}` rows, `time` in seconds from
the kernel's start.

Four of the campaign's sessions have died with nothing readable, so this log is the only
complete account of a session the campaign has, and the only measurement anywhere of what
the exit publish costs. Three documented numbers now rest on it.

## 1. The exit publish: 25.42 GiB in 12.0 minutes, i.e. 36.0 MiB/s

The runbook's section 6 asks for exactly this number, in writing, rather than an
assumption: *"Watch the `remote publish:` line and the push duration in the exit log; if a
push ever takes longer than the pairs it protects, revisit this choice with that number
rather than an assumption."* Here it is, from the log's own two facts:

| | |
|---|---|
| The child was launched | line 23, `t = 226.3 s`, with `--hours 7` |
| The child's own account of its window | line 42, `session done: 4 pair(s) completed, 1.12 h remaining` - so it consumed `7.00 - 1.12 = 5.88 h` and exited at `t = 226.3 + 5.88 x 3600 = 21,394.3 s` |
| The driver's account of the whole session | line 68, `=== session session finished after 6.14 h ===`, at `t = 22,116.4 s` |
| **Post-child work** | `22,116.4 - 21,394.3 = 722 s` = **12.0 min**, covering the remote publish, the `_keep` calls and the closing disk report |
| The payload | line 67, `handoff root ...: 25.42 GiB`; line 48, `remote publish: sid123123234/act-v0-5-1 version pushed (4 seed version(s))` |
| **Implied upload rate** | `25.42 GiB / 12.0 min` = **36.0 MiB/s** (288 Mbit/s) |

The 12.0 min is an **upper bound** on the publish: the `_keep` calls (lines 52-54) and the
disk report (lines 55-67) share it, and they write kilobytes. The earlier estimate in this
repository inferred ~51 MB/s from a *download* and was, on this measurement, optimistic by
about 1.4x.

## 2. What that does to the exit-publish trade - it stands, and now by 3.25x

The runbook's trade is that seeds are pushed **once at exit**, not as each pair completes,
because a per-seed push would re-upload the whole cumulative handoff every time. Priced on
the measured rate (6.36 GiB per seed, 4 seeds = 25.42 GiB):

| Scheme | Bytes over the declared block | Time |
|---|---|---|
| Exit publish, one per session (4, then 8, then 12 seeds) | **152.5 GiB** | **1.20 h** |
| Per-seed publish (1, 2, ... 12 seeds cumulative) | **495.7 GiB** | **3.91 h** |

**3.25x the bytes, 2.71 h apart.** So the trade is confirmed on a measurement rather than
on an argument, and the runbook's revisit criterion ("longer than the pairs it protects")
is met with room: this session's **12.0 min** push protected **339 min** of trained pairs.

The 45-minute publish reserve is also confirmed as correctly sized, and it is sized for the
**last** session, not this one: 12 seeds is 76.3 GiB, which is **36 min** at the measured
rate - under 45, over any smaller reserve.

## 3. Why four failure captures were empty: the child's stdout was block-buffered

This log settles a mechanism that was a hypothesis. Every one of the child's own opening
messages arrived **6.08 hours late**, in one batch, while the wrapper's own lines arrived
immediately:

| Rows | Producer | `time` | What it says |
|---|---|---|---|
| 17-21 | the wrapper | 226.3 s | `--- throughput benchmark ---`, `measured 1442.0 tokens/s`, `--- session queue ---` |
| 26-27 | **the child**, printed at its start | **22,116.3 s** | `Dataset URL: ...`, `no remote version of sid123123234/act-v0-5-1 available yet; training from scratch` |
| 28-42 | **the child** | **22,116.3 s** | every pair start, pair completion and the stop at seed 15004 |
| 48 | the child's publish path | 22,116.3 s | `remote publish: ... version pushed (4 seed version(s))` |

The child's restore messages are printed within seconds of its launch (`t = 226.3 s`) and
were delivered at `t = 22,116.3 s`: the stream was healthy and the wrapper's own lines were
live, so the delay is the **child's piped stdout**, which block-buffers - the child's whole
run is a few KB of text, under one 8 KiB stdio block, so nothing flushed until the process
exited. That is the exact
mechanism `kaggle_t4.py` now sets `PYTHONUNBUFFERED` for, and it is why a **killed** child
leaves no output at all while a session that exits normally leaves everything at the end.

Two consequences worth stating plainly. The live capture's `failed 5 times with no new
data; giving up` at 4.1 minutes was **inevitable** - there was nothing to read, from a
session that was running perfectly - so the fix belongs in the child's environment rather
than in the tailer's patience. And the four unexplained deaths are unexplained *because*
of this: a session killed mid-pair never flushes, which is why the platform's log for them
is empty rather than short.

## 3b. A message in this log that looks like a failure and is not

Lines 44-47 are four copies of `Error while trying to load upload info:
KaggleObject.from_dict() got an unexpected keyword argument 'token'`, and they sit
immediately **before** line 48's success (`version pushed (4 seed version(s))`). This log is
the reason that message can be called benign rather than dismissed: the container's own
`kaggle` 2.0.2 CLI rejects a field in a modern API response while *reading* upload info,
and the upload path continues. The runbook's symptom table now carries it, because a
reader diagnosing a lost publish would otherwise reasonably stop on it.

## 4. What the session did with its hours

| | |
|---|---|
| Window | `--hours 7`, finished at **6.14 h** of the platform's 8-hour cap |
| Pairs | **4** (15000-15003), consuming 5.88 h of the child's 7 h window |
| Pair time the schedule predicted | line 28, `pair time 1.41 h (2 arm(s) in 1 wave(s) of 24576000 tokens at 5194 tokens/s + 0.1 h event overhead)`, against `4 x 1.41 = 5.64 h` of a 5.88 h consumption - so the schedule is right to within the 0.24 h of startup the child pays too |
| Why it stopped at 4 | line 41, `seed 15004: remaining 1.12 h does not fit pair 1.41 h + 45 min reserve (rule 1)` |
| The unused tail | 1.12 h. Structural, not a defect: a fifth pair needs `1.41 + 0.75 = 2.16 h` of clock and the child had consumed 5.88 h by then, so the window would have to be **at least 8.04 h** - and the platform's cap is 8 h *total*, which must also cover boot (226 s) and the publish (12 min). A smaller reserve would not have helped either: 1.12 h is below the 1.41 h pair itself. |
| Pair time, against the schedule | The child consumed **5.88 h for 4 pairs** against `4 x 1.41 = 5.64 h` scheduled, so the schedule is right to within the **0.24 h** of startup the child also pays (the restore attempt and its `training from scratch` decision). The log carries **no per-pair durations** - every child line arrived in one batch at exit, which is the buffering finding above, not an omission here. |
| Per-seed payload | 25.42 GiB / 4 = **6.36 GiB**, matching the runbook's ~6.4 GiB per seed |
| The arms' own numbers | line 31-40: random vs adaptive final loss per seed, all within 1.2e-3 of `ln(32768) = 10.3972` (see `docs/IMPLEMENTATION_STATUS.md`, 2026-09-21, for why that is the expected scale on synthetic tokens) |

## What this log does not say

**Why the other sessions died.** This one ended normally. It is the *baseline* a healthy
session has, which is what makes the others' emptiness legible as buffering rather than as
a silent stall - but it is not a cause for them, and none is claimed here.


## What `session-15024-26-log.json` is

The platform's completed execution log for `sid123123234/act-t4-session`, Kaggle kernel
version **7**, fetched 2026-09-26 after the kernel reached `COMPLETE`. The raw file is the
Kaggle JSON array of `{stream_name, time, data}` events (`time` seconds from kernel start),
10,980 bytes, **69 events** (63 stdout and 6 stderr), sha256
`c56692a1b315d897c0cbcaf1c46003194ce10997fe2cda207de4e32b25a35a71`.
It records the result the earlier journal entry left pending.

| | |
|---|---|
| **Kernel state** | `COMPLETE`; the kernel log's final line reports `=== session session finished after 5.52 h ===`. This is a clean completion, not a crash. |
| **Benchmark and schedule** | Two Tesla T4s, torch 2.10.0+cu128. The 50-step benchmark selected **fp16** and measured **4,296.1 tokens/s** for the schedule (fp32 diagnostic 1,238.8); the planned pair time is **1.689 h**. |
| **Pairs** | Seeds **15024, 15025, 15026** completed on two cards in parallel; each record says `status=complete`, `total_steps=6000`, three mutations, `probe_candidate_policy=spread`, and the §5.47 rebuild pair (`preserve_router_on_rebuild=true`, `new_expert_initialization=clone`). Seed 15027 did not start: the remaining **1.58 h** did not fit a **1.689 h** pair plus the **45-minute publish reserve**. |
| **Published work** | `session-15024-26-publish.json` says `status=published`, `seeds_published=3`, `error=null`; Kaggle's dataset listing exposes each seed's complete manifest, report and both arm checkpoints. The dataset is `sid123123234/act-v0-5-2`. |
| **Capture reconciliation** | The saved `session-15024-26-child-stream.log` contains all three pair-completion lines, the queue stop and successful publish; it is an exact contiguous substring of the platform log's concatenated stdout. The recovered live capture is **not** byte- or line-for-line identical: it mixed restarts and wrapper output, reports **4.99 h** versus the platform's 5.52 h, and carries a different benchmark/schedule. It is retained only as forensic tailer evidence, not as the canonical run log. |
| **Queue state after this run** | The primary queue is **3/12**, not complete. Kaggle reports **19.48 GPU-h remaining of 30.00**, refreshing 2026-10-03. The attached code dataset is current. |
| **Other registered queues** | On 2026-09-26, the authenticated owner's `kaggle kernels list --mine` contained no `act-t4-control` or `act-t4-null`, and `datasets list --mine` contained no `act-v0-5-2-control` or `act-v0-5-2-null`. Direct queries for those exact configured IDs returned permission errors/403. Together, the account-wide listings establish that neither queue has been launched in this configured account; this is not just an unreadable dataset and not evidence of a zero-seed completed queue. The registration still requires both queues, and the campaign cannot produce its registered verdict until they are run. |

The official log and compact records live in `artifacts/v0-session-logs/`:
`session-15024-26-log.json`, `session-15024-26-child-stream.log`,
`session-15024-26-session.json`, `session-15024-26-publish.json`,
`session-15024-26-benchmark.json`, and `session-15024-summary.json` through
`session-15026-summary.json`. The complete Kaggle output also includes the attached code
tree; that duplicate source bundle was discarded rather than committed. Checkpoints remain
in the Kaggle dataset and were not downloaded (roughly 20 GB across both arm weights for
these three seeds).
