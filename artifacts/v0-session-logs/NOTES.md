# Campaign session logs - the evidence behind the exit-publish numbers

## What `session-15000-15003-log.json` is

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
were delivered at `t = 22,116.3 s`: the stream was healthy and the wrapper's lines were
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
