# ACT Implementation Status

## The control queue is 6/12, and its last week-hours went to the empty shard (2026-10-21)

Two control (§2b) sessions ran the declared `15024-15030` block, which is the whole seven-seed
shard `act-v0-5-2-control`. Both restored that shard (**18.99 GiB in 2.3 min**) and trained three
pairs; every publish landed and both exits were skipped as `already_published`. Peak local
footprint stayed near 19 GiB restored plus ~19 GiB staged - nowhere near the 2026-10-19 shape.

| | |
|---|---|
| **Block state** | `campaign_health --queue control`: **6 of 12** (15024-15029), quota **23.36 / 30 h**, refresh 2026-10-24. The remaining seeds split across two shards: **15030** in the seven-seed `act-v0-5-2-control` and **15031-15035** in `act-v0-5-2-p2-control`, which has **no version**. |
| **Readout (not a verdict)** | adaptive won **3 of 6**, mean `adaptive - random` **+0.000006 nats**, every arm within **0.001281 nats** of `ln(32768) = 10.3972`. Same uniform floor as the primary block's 12; the queue is incomplete, so the registered rule still withholds itself. |
| **Cost** | quota **16.71 -> 23.36 h** across the two sessions (~3.3 GPU-h each). |
| **What the driver then did** | It stopped at exit 3 rather than pushing a session the week cannot finish: **6.64 h** remained against the declared 7 h, and unused quota expires at the refresh. `campaign_health` called that state `STALLED` (no session live, quota available, lock unheld) - the state both 2026-09 losses sat in. |
| **The scheduling choice** | `15024-15030` would restore the ~38 GiB shard to train one seed and re-upload it as a **~44 GiB** version, with the same size in staging. `15031-15035` resolves to the **empty** `act-v0-5-2-p2-control`: **0 GiB restore**, and the publish writes a new small shard. Fired as kernel **v3**, `--hours 6.5 --seeds 15031-15035 --wait-for-quota` - the window is a scheduling knob (precedent 2026-09-27), the declaration is a disk budget, and the wait spends the 2026-10-24 refresh without an operator. `RUNNING` on a T4 as of 12:35Z. |
| **The seed that stays expensive** | **15030** is the seventh seed of its shard, so publishing it means restoring the six-seed shard and re-uploading all seven - the one remaining control seed that costs a full-shard push, at the 44 GiB size that has landed before. |

## The primary block is complete 12/12, and the scoping fix is verified in production (2026-10-20)

The session that finished the block declared only the seeds it still had to train (`--seeds
15034-15035`), so its startup restore resolved to **shard 1 alone**. That is the 2026-10-19
findings' fix doing the work it was written for: it restored **19.09 GiB in 19.2 min** where the
previous fire restored 44.13 + 19.09 GiB and then lost its writable layer at a 95 GiB footprint.

| | |
|---|---|
| **Block state** | `campaign_health --queue session`: **12 of 12** complete (15024-15035), session kernel `COMPLETE`, driver lock free. `latest_published_seed` **15035**, code revision **d12c3ca**. |
| **Restore** | shard 1 only, `19.09 GiB` in **19.2 min**, three versions `[15031, 15032, 15033]`; shard 0 was not in the declared block and was not downloaded. |
| **Pairs** | 15034 `random 10.397485` / `adaptive 10.397881` (**+3.95e-4**); 15035 `adaptive 10.397495` / `random 10.397844` (**-3.50e-4**). |
| **Publishes** | 4 seeds / **25.5 GiB** in 24.1 min, then 5 seeds / **31.8 GiB** in 30.3 min, both into `act-v0-5-2-p2`; the exit publish reused fingerprint `46069823f084` and uploaded nothing. |
| **Peak footprint** | ~**64 GiB** (19.09 restored + 2 x 6.4 trained + 31.8 staged) against the 95.0 GiB that went EROFS on 2026-10-19 - the number the fix was chosen for, now measured. |
| **Cost** | **4.79 GPU-h**; quota **16.71 / 30 h**, refresh 2026-10-24. |
| **Evidence** | The platform log survived and carries the child's stdout (21,905 B) - a clean exit, unlike every killed session's empty `[]`. Live capture, kernel log and session record are committed under `artifacts/v0-session-logs/` with their sha256s in its `NOTES.md`. |
| **Next, registered order** | The control queue (§2b) was fired as `--queue control --seeds 15024-15030`: **shard 0 only**, because a session spanning the shard boundary would have to restore shard 0 *and* stage both shards (44 + 32 GiB) - the footprint that has killed every container. Shard 1 (`--seeds 15031-15035`) follows under its own drive, then the null queue (§2c). The remaining ~16.7 h this week covers roughly two 7 h control sessions, so the block completes across the 2026-10-24 refresh. |
| **Verdict path** | `scripts/verdict_rehearsal.py --check` passes (exit 0) on the same tree that will apply the verdict, so the decision machinery is intact before the control and null spend. |

**The completed block's own readout, which is not the verdict.** `campaign_health --queue
session --readout` reads all twelve published summaries: **adaptive won 6 of 12**, mean
`adaptive - random` **+0.000063 nats**, and every arm sits within **0.001982 nats** of
`ln(32768) = 10.3972`, the loss a signal-free model reaches on this block's synthetic stream.
That is the floor, and it is the same reading the §5.40/§5.66 dev-fidelity studies gave
(null ahead, no adaptive edge). The pre-registered interpretation of 2026-09-20 anticipated
exactly this asymmetry: the harm guard fires only if the adaptive arm loses on 10 of 12 seeds,
and **6 of 12** is nowhere near it - so this block cannot withhold Gate 4 on the harm clause,
and equally cannot support a value claim. A cleared clause is "the clause did not fire", not
evidence of a win; the registered cost/band/control rules carry whatever the verdict says, and
the control and null queues are what make those readings interpretable.

## The wall is the container's disk, not the dataset version: the 2026-10-19 fire died on EROFS (2026-10-19/20)

Kernel **v21** restored both shards, skipped all ten published seeds, trained **15034**, and lost it. The pair-time push of **4 seeds / 25.5 GiB** into `act-v0-5-2-p2` ran for five minutes and then failed with `OSError: [Errno 30] Read-only file system` on the staging directory, three times, and the container was killed. Unlike every earlier fire this one flushed its own stream log, which `kaggle kernels output` still served: the cause is the child's `EROFS`, not a silent kill. Both the stream log and 15034's summary are committed in `artifacts/v0-session-logs/` (see its `NOTES.md`); the 6.4 GiB of checkpoints are gone, so the seed is trained again.

| | |
|---|---|
| **What died** | The **pair-time** push, not the exit push the 2026-10-19 fix addressed. It staged 25.5 GiB - the whole of shard 1 (3 restored seeds + 15034), because a dataset version replaces its content and the shard's older seeds must travel with the new one. |
| **Local bytes at that moment** | 44.13 (shard 0) + 19.09 (shard 1) + 6.4 (15034) + 25.5 (staging) = **95.0 GiB**. The restore itself peaked near 82 GiB (download plus copy) and survived. |
| **Why this is not the 44 -> 50 GiB version wall** | The payload that killed this container, 25.5 GiB, is **smaller** than the 44.13 GiB version that landed on 2026-10-05. What is common to every death is the *total* footprint: v18's landing push had ~44 GiB of handoff + 44.13 GiB of staging (88 GiB), its fatal 8-seed push needed ~101 GiB, the 04:10Z fire's fatal exit push needed ~107 GiB, and this one died at 95 GiB. The bracket recorded on 2026-10-18 was a real observation with the wrong cause; the writable layer is the variable that fits every fire. |
| **The waste that made it certain** | Shard 0 is 44.13 GiB this session could only read: its seven seeds were complete and the exit fix already guaranteed they would not be re-pushed. It was restored because the run declared `--seeds 15024-15035`. The CLI has always scoped the restore to the shards the declared seeds resolve to, so a narrower declaration restores 19.1 GiB instead of 63.2. |
| **Fix** | `kaggle_queue.py --seeds` (pinned into the pushed kernel as `ACT_T4_SEEDS`, read by `kaggle_t4.session_seeds()`); the default is still the declared block. It is a **training scope and a disk budget**: with 15034-15035 declared, only shard 1 is restored, and the peak is ~64 GiB even with both seeds trained and the 31.9 GiB shard staged. |
| **Pinned** | `tests/test_kaggle_t4.py::test_a_narrower_seed_block_scopes_the_restore`, `tests/test_kaggle_queue.py::test_staged_kernel_pins_a_narrowed_seed_block`, `::test_staged_kernel_without_a_seed_override_keeps_the_declared_block`. |
| **State at this entry** | `campaign_health`: **10 of 12** (15024-15033), session kernel `ERROR`, no driver live, quota **21.5 / 30 h** (refresh 2026-10-24). A session for 15034-15035 is fired with the narrowed block; the previous entry's "keep shards <= ~45 GiB" advice stands as a shard rule but is not what the ceiling is. |

## Next-step recommendations from the findings, not from outside literature (2026-10-19)

A web search on learned MoE topology returned only introductory material, nothing that bears on the open questions here, so these recommendations rest on the project's own record. None of them is a measurement.

| Recommendation | Why, from the record | What it would cost |
|---|---|---|
| **Search width and depth jointly, not one at a time** | The depth penalty depends on width (interaction +0.0827 nats, six-seed replication). A one-axis search can misattribute depth as width or the reverse. | The width-control gate in `docs/width-control-design.md`, already pre-registered. |
| **Treat head count as a joint gene with width** | Head count is worth nothing at narrow width (+0.0084, p=0.37) and -0.0553 nats at hidden 896 (12/12 pairs). | A head-by-width arm; reuses the existing head-paired design. |
| **Screen at 400 steps before confirming at 1,200** | A 400-step screen costs ~1.1 GPU-h against 3.2 for the same answer (R^2 0.927 vs 0.953, same lever significant at both). | Nothing new to build; the plan is in the zoo tooling already. |
| **Keep shards at or below ~45 GiB per version** | 44.13 GiB landed; ~50.4 GiB and up killed containers. The 2026-10-19 exit-push bug was the same ceiling hit by a re-upload. | Already enforced by the shard rule; the exit fix stops the re-upload. |
| **Do not start the control or null queues on this week's quota** | 23.8 of 30 GPU-h remain and the session queue still needs ~3.5 h for seeds 15034-15035. Control and null are large and their verdict depends on the sharded path first working end to end. | A decision for the operator, not a scheduling knob. |

## The exit push re-uploaded a restored 44 GiB shard and killed the session; fixed (2026-10-19)

The 2026-10-19 04:10Z session worked: it restored 15024-15030, trained **15031-15033**, and published them to the shard-1 dataset `act-v0-5-2-p2` (19.1 GiB, 10.0 min, `status: published`), which is the first live run of the sharded path at a real size. It then died inside the **exit** push of **7 seeds / 44.1 GiB** into shard 0, whose 15024-15030 it had only restored.

| | |
|---|---|
| **Cause** | `cli._publish_remote` was given every requested seed, and selected each shard that overlapped them. Restored seeds count as requested, so shard 0 was pushed again. The fingerprint reuse that should have skipped it reads the publish log, which lives in the container overlay and is empty in a fresh session. |
| **Fix** | The exit path pushes only seeds this session trained (`completed` or `publish_failed` pairs). When none were trained it skips the push and says so. Commit `6401c0d`. |
| **Pinned** | `tests/test_v0_runner.py::test_cli_exit_publish_skips_seeds_restored_as_complete` (new). Two existing CLI-exit fixtures now report their simulated pair as `completed`, which is what the runner emits. 112 tests pass across runner, kaggle_remote and campaign_health. |
| **State at this entry** | `campaign_health`: **10 of 12** (15024-15033). Quota **23.95 / 30 h**, refresh 2026-10-24. Driver launched on the session queue (`--max-sessions 2`, log `artifacts/session-queue-20261019.log`) against kernel v21, code at `3f59fe6`, to train 15034-15035. |

## The publish is not flaky: it is walled between 44 and 50 GiB, and the block cannot fit (2026-10-18, read from the 2026-10-05/06 fires)

The per-pair publish did what it was written to do, and then found the next wall. Kernel
**v18** (2026-10-05 20:05Z) pushed the **7-seed state and it landed** - the block moved
**6 -> 7 of 12**, the first progress since v11 - and then died inside the push that would
have made it **8**. Kernel **v19** (2026-10-06 10:45Z, a 2.84 h window fitted to the 2.91 h
of quota left) re-trained that same eighth seed and died inside the same push. Both kernel
logs are **empty** (`[]`, 3 B) and both live captures stop on the kaggle CLI's version
warning - the first line it prints - so the account is the driver's capture, committed under
`artifacts/v0-session-logs/` with its sha256s.

| | |
|---|---|
| **What the two captures bracket** | The dataset's last version is still the 7-seed one (`lastUpdated` **2026-10-05T22:17:32Z**); neither 2026-10-06 push left a version, so the kill landed **before finalize**, not after a version was created. The push that landed carried **7 seeds / 44.13 GiB** (35 files; restore reports the same 44.13 GiB) and took **16.9 min**. The push that killed v18 and v19 would have carried **8 seeds / ~50.4 GiB**. The three sessions before them (v15-v17) entered the exit push of **9 seeds / ~56.7 GiB** and died there. So: a version at 44.13 GiB lands; one at ~50 GiB does not. **No Kaggle document in this tree names that limit**, so the record keeps the bracket and not a cause. |
| **Why this is structural rather than another bad run** | The registered block is **12 seeds x ~6.3 GiB = ~75.6 GiB**, and the local handoff is pushed as one dataset version. At the measured wall the block can never be published as one version, at any window length - so firing another session cannot move seed 8, which is why the driver's own `a session failed - stopping so the cause can be read` was right. The cost is concentrated in one seed: **15031 has now been trained six times** (2026-10-03T2236Z through 2026-10-06T1254Z), and all six pairs completed and were discarded. |
| **What v18 did get right** | The per-pair publish bounded the loss in exactly the way `e60b3b6` argued: v15-v17 each lost the three pairs they trained, while v18 made 15030 durable (and 15024-15029 restored rather than re-trained) before the next push killed it. So the change is kept; what it exposed is the payload ceiling. |
| **The diagnostic gap this cost, now closed** | A push is a single blocking client call that prints nothing while it runs, so a kill at one minute and a kill at sixteen leave the same capture - and the two things that would settle the bracket (how many bytes were in flight, how long the push had been running) were not in it. `publish_handoff` now logs the payload **before** the client call (`remote publish: pushing N seed version(s), X GiB of handoff`), names each attempt with the same size, and beats every 60 s while the upload is in flight (`still pushing ... after M min`). Sizes are counted from the same file walk the restore prices itself with. The next fire therefore leaves a capture that says whether the 50 GiB push dies at the first byte, midway, or after the upload returns. |
| **What this entry does not establish** | The cause of the kill is **not known**. The two kernel logs are empty, no error line survived, and the observed correlation is size, not a quoted limit. The heartbeat is what makes the next observation falsifiable; it is not evidence for a particular threshold. |
| **Pinned** | `tests/test_kaggle_remote.py` (**+2**): the size and attempt lines are emitted **before** `create_version` is called and carry the seed count and GiB, and the heartbeat fires while a push is in flight and stops when it returns. The push's own contract is unchanged: a failed attempt is still retried, a failed push still never raises, and the reuse-by-fingerprint path still uploads nothing. |
| **The sharding, implemented from the pre-registered rule in `docs/handoff-sharding-design.md`** | Shard 0 is **the version that is already published** (15024-15030, 44.13 GiB), so nothing has to be re-pushed or moved; shard 1 is **15031-15035** (five seeds, ~31.5 GiB at the measured 6.30 GiB/seed - 62% of the 50.4 GiB push that kills the container), published as the derived id `<base>-p2`. Membership is positional over `DECLARED_SEEDS` and declared per queue (`shard_seeds=7` for the paired queues, `0` for the single-arm null queue), so a session that asks for a subset still pushes the shard the registered block puts those seeds in. A push stages only its own shard; restore downloads the queue's shards in order and stops at the first that does not exist; the publish log keeps every shard's fingerprint so one shard's success cannot make the next session re-upload the other's forty gigabytes. `campaign_health` reads the shards as one set and refuses a seed published in two (overlap is forbidden by the rule); a continuation that does not exist yet reads as empty rather than as an error, which under-reports at worst and cannot produce a false COMPLETE. The 6/6 alternative in the earlier draft of this row was rejected before implementation: it would need a published seed re-pushed into the second dataset, or left in both. |
| **What the first fire is now for** | The sharded path is unit-tested **and live-probed at kilobyte scale** (two new private datasets, one fake seed each: the filter uploaded only its own seed, a two-shard restore returned both, and a repeat of the first shard was `already_published` after the second pushed). The CLI itself also ran end to end against a throwaway live dataset - two development pairs, 1 then 2 seeds pushed to one shard, exit publish reused - and that dataset was deleted afterwards because it carried development records for declared seeds (`docs/handoff-sharding-design.md` section 9). What has never run is a **31.5 GiB** shard. The next session is that test: it restores shard 0 (44.13 GiB) and shard 1 if it exists, re-trains 15031 once, and publishes five seeds instead of the eight whose 50.4 GiB killed v18 and v19. If that push dies too, the size reading is wrong and the heartbeat says where; if it lands, the block is unblocked for 15032-15035. |
| **The state** | Primary block **7 of 12** (15024-15030) in `sid123123234/act-v0-5-2`; kernel `act-t4-session` v19 `ERROR`; **no driver is running** (it stopped at `2026-10-06T12:54:51Z`) and the last kernel run on the account is `2026-10-06T10:45:39Z`. The quota has since refreshed: measured by `campaign_health.py --queue session --json` on **2026-10-18**, **30.0 of 30 h remain, refresh 2026-10-24** - a full week, unspent. Nothing is relaunched by this entry: the next fire costs a session and is the experiment that tests the shard rule, which is the operator's to call. |

---

## The null queue is fired on the TPU: twelve declared seeds, one per chip (2026-09-29)

**Live update, 2026-09-29 11:55Z:** with the GPU week spent (**1.65 of 30 h**) and the TPU week at
**16.77 of 20 h** until the 2026-10-03 refresh, the operator reopened section 5.14's option 2 and
gave the go to spend it producing the block rather than pricing it. The two grounds the earlier
disposition rested on are **not** withdrawn - `bf16` is still not a session dtype, and the one
dtype the path can run, `fp32`, still differs from the registered fp16 - so what changed is not
the measurement but **whose decision it was**: the entry recorded the amendment as "the operator's
to sign, not a scheduling knob", and it has been signed as far as the run is concerned. What
follows is the runner that takes it, the pusher that fires it, and the reading that says whether
it landed.

| | |
|---|---|
| **What is being run, in the registration's own terms** | The registered null queue: the block's **twelve seeds** (`15024-15035`), one arm each (`--session-arms null`), the declared config (285M total / 134M active, batch 2 x context 2048, **6,000 steps**, checkpoint cadence 500, `max_checkpoints` 2 - the section 5.2 pin), the section 5.47 rebuild treatment on the arm, and `publish_seed` into the null queue's own dataset root (`act-v0-5-2-null`, i.e. `sid123123234/act-v0-5-2-null`). This is `kaggle_t4.session_argv(QUEUES["null"])`'s command with **two** differences, both named in the record: `--device xla`, and the placement - the CLI walks seeds **in turn on one device**, and at the probe's per-chip rate that is ~43 h against a 20 h week, while a v5e-8 carries eight at once. |
| **The dtype, which is the one field that is not the registration** | `DTYPE = "fp32"`: `V0SessionConfig` admits `fp32` and `fp16` and raises for anything else, and the declared `fp16` on a TPU needs `torch.autocast(device_type="xla", ...)` plus a `GradScaler`, which no record in this repository shows working. `fp32` is registered, needs no such machinery, and is the **slower** end of the two dtypes `tpu_probe` v5 put within 6% of each other - so the block errs toward slower, and the record carries `dtype` so twelve TPU-fp32 null seeds cannot be read as the registered fp16 block. **Whether the verdict may consume them as the registered null clause is a separate decision and is not taken here**; what is taken here is producing them, which is the part the idle week can do and six GPU seeds are not going to. |
| **The defect the first draft of the runner would have hit, found by reading the campaign's own kernel instead of by losing a wave** | The runner defaulted its **handoff roots to `--output`'s parent** - i.e. to `/kaggle/working`, which is a **20 GB loopback mount** (`kaggle_t4.workspace_root`'s own measurement: 19.5 GiB on `/dev/loop1`), while one completed seed publishes ~3.2 GiB of resume state *including the AdamW moments*. Twelve seeds is ~38 GiB: the mount fills and the file that fails to be written is the **record**, which is the one artifact a fetch reads. The T4 path already answers this - roots in the container overlay (`/tmp`, token-scoped because the overlay is host-shared), the working mount for the record and the published-you-don't-need-it summaries - so the runner now delegates to `kaggle_t4.workspace_root` on a kernel and keeps the output-relative placement only locally, where it stays inspectable. `--scratch` overrides it. |
| **And the second half of that defect: the exit path** | A queue is not a measurement, so the remote half matters: with `--dataset` given the run follows **section 5.2** the way a session does - **restore** the newest published version before training, **skip** the seeds that version already carries (a published seed is never re-trained over: measured 2026-09-26/27, the misreading of a 403 cost a session's quota and republished a seed over itself), and **publish** at exit through `kaggle_remote.publish_on_exit`, which retries and never raises because the container ends with the kernel. A restore that fails on a dataset the account owns stops the run **before** it trains (`exit 2`, `errors: ["restore"]`). The per-seed summaries are copied flat into the output directory as well, through the campaign's own `_keep_seed_summaries`, so a failed push still leaves every number the verdict reads. |
| **The reading the fetch prints, so the record cannot go unread** | `tpu_null_queue.verdict(record)`: how many of the twelve landed, in which dtype, how many arms **do nothing** (`search_flops 0` and `mutation_count 0` - the null clause's own built-in check, so a policy arm cannot be read as this queue), and the final losses' mean, min and max. It is printed by `--fetch` for the reason `artifacts/tpu-probe/NOTES.md` records: a result that needs a second hand-run command is a result that can be left unread. |
| **A third defect, in shared code, found staging this kernel** | `push_tpu_probe.XLA_BOOTSTRAP` - the fragment all four Torch-XLA entries paste in - calls `subprocess.run` but relies on its **entry** importing `subprocess` at module level. The price probe's entry names only `glob`, `runpy` and `sys`, so on the first image that actually needed the install the bootstrap would have died on a `NameError` *inside the code whose own prints are the diagnosis*. It never fired because the image already carries `torch_xla` (`already-present`), which is also why it was invisible. Fixed **in the fragment** rather than in one copy of it, with the reason recorded beside it. |
| **Verified on a real `xla` device before any TPU hour** | The whole changed path ran locally on **XLA:CPU** (`PJRT_DEVICE=CPU`, torch 2.8.0 / torch_xla 2.8.0 in a throwaway `/tmp` env): two development seeds, one worker, 20 steps, `errors: []`, `seeds_completed: 2`, final loss mean **9.0736** (min 9.0733, max 9.0739), **1,041.5 tokens/s** per chip, `search_flops 0` / `mutation_count 0` on both arms, and the new split visible in the artifact tree - the record plus two `seed-NNNN-summary.json` files under `--output`'s directory (**12 KB**), and the two published seed trees with their checkpoints under `--scratch` (**343 MB**), with the session directories cleaned after publish. Two seeds at dev fidelity is not a schedule input; it is the statement that the publish, the split and the reading work on this device. |
| **Pinned** | `tests/test_tpu_null_queue.py` (new, 22) and `tests/test_push_tpu_null_queue.py` (new, 10), plus one in `tests/test_push_tpu_probe.py` and assertions inside the existing bootstrap test in `tests/test_push_tpu_session_price_probe.py` (2091 -> **2124** in the suite). They hold the quiet ones: a seed spec that reads backwards or repeats is refused rather than silently training nothing; the declared seed is the registered config and **names its own dtype**; the development seed is the dev shape, so a dry run cannot be read as the declared one; on a kernel the roots are the campaign's answer and **not** `--output`'s parent; `main` writes the record to `--output` and the roots to `--scratch`; a restored seed is skipped and its losses read back from its own published summary; a failed restore trains nothing; the verdict counts a non-null arm against the clause; and the publish is the probe's own guard, whose refusal (exit 5) leaves no `kaggle kernels push` behind it. |
| **A fourth obstruction, and it is a platform rule: one TPU session at a time** | The push was **refused** - `Maximum batch TPU session count of 1 reached` - because the price probe's v2 is queued on the TPU and Kaggle counts it as the account's one session. That is not a reason to cancel the probe: it is a **declared-shape smoke test of the same path** (one null seed, `declared_session_config`, fp32), and if the declared shape still cannot run on a TPU it is far better to learn it from a ~30 s run than from the first wave of a twelve-seed block. So the wait is instrumented rather than hand-watched: `scripts/push_tpu_null_queue_when_free.sh` retries the **staged** push on a bounded poll (720 x 120 s), names the slot's holder in its log, and on success arms the fetch watcher. It does **not** re-run the pusher, and that is the point of it: the pusher republishes the code dataset before pushing, so re-running it later would let a later (docs-only) commit become the revision the kernel mounts *and the record names*. |
| **And the refusal is invisible to an exit code, which cost the first version of that waiter** | Measured 2026-09-29, in the foreground: `kaggle kernels push -p artifacts/tpu-null-queue` prints `Kernel push error: Maximum batch TPU session count of 1 reached.` and **exits 0**. The first version of the waiter tested `$?`, so it announced a fired queue, exited, and armed a fetch watcher against a kernel that did not exist - a consumer of this campaign would have read "pushed" and waited for a block that was never going to be trained. What caught it was the account's own kernel list (`kaggle kernels list --user ... -m` shows **no** `act-tpu-null-queue`), not the status call, which had 404'd from the start. Success is now the CLI's own line - `successfully pushed` - in **both** places: the waiter tests the output, and `push_tpu_null_queue.push_kernel` reads the same line so this pusher's exit status stops reporting "fired" for a push that never left (exit 6 instead). The lesson generalises to every pusher in this repository: none of them should be read off an exit code alone. |
| **The other idle clock, put back on the rails while this one waits** | The GPU driver was **gone** (see the correction above), and the state it left is exactly the one `campaign_health.py` calls `STALLED - relaunch the campaign`. So `scripts/run_gate4_campaign.sh` was relaunched at **11:56:14Z** (driver pid **2912294**, appending to `artifacts/gate4-campaign.log`); it printed the health block, saw `WAITING_ON_QUOTA`, and is sleeping to the **2026-10-03** refresh under `--wait-for-quota`, holding the lock so the same health report stops saying nothing is waiting. The next 30 GPU-h then go to the registered order - the primary block's remaining six seeds, then the control queue, then the null queue, which the TPU block below makes nearly free (the CLI walks the declared queue and labels an already-published seed `skipped_complete`). This is not part of the TPU decision; it is the campaign returning to the state its own instrument says it should be in. |
| **A provenance hazard the relaunch exposed, worth stating because it is invisible** | The driver's launch republished `act-lab-code` - it does that at session start - so the code dataset moved from **`bf40d53`** to **`d5fdd7b`**, and a queued Kaggle kernel mounts the dataset's **latest** version *at run time*. The record's `code_revision` therefore names whichever revision is latest when the kernel starts, not the one it was pushed from. What makes that safe rather than sloppy is that the bytes are identical: `scripts/tpu_null_queue.py` sha256 **`9e744d9c2edce37123782d29bd9b14b7ee1cf042c0ae9077e73ca12619c1e971`** and `src/act/model/checkpointing.py` sha256 **`0bf0ee2e6de98a1c947aef8ade03611d816fafd504aa5f84d6de24a8f37917d2`** in **both** revisions (checked by diffing the two commits and by downloading the dataset back twice). So the artifact that identifies the code is the **sha256 of the runner**, not the revision string - which is what `artifacts/tpu-null-queue/NOTES.md` records, and why it says so. Any future edit to the runner before the kernel starts would move that hash without moving anything a reader is looking at. |
| **The state** | Kernel **`sid123123234/act-tpu-null-queue` staged and waiting for the TPU slot** (the price probe holds it), from the code dataset carrying this run's runner (sha256 above, identical at `bf40d53` and `d5fdd7b`); the waiter above fires the push and then the fetch watcher. **TPU on / GPU off**, staging `--device xla --seeds 15024-15035 --steps 6000 --workers 8 --fidelity declared --dataset sid123123234/act-v0-5-2-null`, with the code dataset republished at this revision **through the probe's guard** and the mount downloaded back and checked to carry the runner. Expected wall: **~5-6 h** (twelve seeds, eight chips, two waves) against **16.77 h**, and Kaggle bills wall clock. The GPU week is untouched: the primary block stays at 6 of 12 and the next GPU work is the 2026-10-03 refresh. |
| **A correction to the two entries below: the driver's lock is **not** held (measured 2026-09-29)** | Both 2026-09-28/29 entries say "the driver still holds the lock and is sleeping to the 2026-10-03 refresh". It does not: pid **63681** is **gone**, and the last thing it wrote is the *pre-commitment* rather than the wait - `quota: 1.65 h remaining of 30.0 ... waiting rather than billing a partial session` at `2026-09-27T16:33:44Z`. The campaign's own health line says the same thing in the other direction (`driver lock: not held - nothing is waiting on this queue`), and it names the consequence: **nothing resumes at the refresh until the campaign is relaunched**, so hours left at 2026-10-03 are lost if nobody is watching. That is a fact about the *GPU* half of the campaign and does not touch this run; it is corrected here because two entries assert otherwise, and the conclusion each of them draws (no GPU queue moved) is unaffected. |

What this does not do is decide the amendment. A block of TPU-fp32 null seeds is evidence about
whether the *route* works - and it is the last piece of §5.14 option 2 that had never run once on
the hardware - but pairing it against the two-T4 fp16 primary is still two registration changes,
and this entry does not take them.

## The price probe bought the finding it was built for: the declared shape could not run on a TPU at all (2026-09-29)

**Live update, 2026-09-29 05:36Z (read 10:17Z):** `act-tpu-session-price` v1 came back **`ERROR`** -
**8.7 h** of queue, **28.09 s** of measured run - and it never reached a training step. What it
returned is the most useful reading the TPU week has produced: the registered session path at the
declared shape **cannot start on XLA**, for a reason no earlier record could have shown, and it
is now fixed. The evidence is committed under `artifacts/tpu-session-price/` (`NOTES.md` with the
three fetched files' sha256s). No campaign queue moved. *(**Corrected 2026-09-29:** this sentence
said "the driver still holds the lock and is sleeping to the 2026-10-03 refresh"; measured, the
driver process is **gone** and the lock is **not** held - see the leading entry. Nothing about the
price probe's reading changes.)*

| | |
|---|---|
| **What the record says** | `completed: ["environment"]`, and three errors, all **`AttributeError: module 'torch' has no attribute 'xla'`** - `session` at **4.5 s**, `resume` at **3.1 s**, `handoff` at **3.7 s**. The fetch's own line: *"no price: the session phase did not leave a throughput to read"*. The `environment` phase passed first (16.8 s, 8 bound devices, the tensor round-trip), so the hardware was fine and the failure was in the path - and it failed in *every* phase that trains, which is what says the defect is in the loop's construction rather than in one phase's setup. |
| **The site** | `torch/utils/checkpoint.py`'s `_get_device_module`: `if device == "meta": ... ; device_module = getattr(torch, device)` - reached from **line 1879**, inside `_checkpoint_without_reentrant_generator_impl`, which is the implementation this repository calls (`use_reentrant=False`) and which is where the RNG state is stashed per device. `_infer_device_type` returns `"xla"` for a checkpointed tensor on a TPU, so the lookup is `getattr(torch, "xla")` and XLA is a separate package, not an attribute of `torch`. Torch has more than a dozen lookups of that shape, which is why the message alone cost most of a day to place - see the diagnostic row below. |
| **Why the rehearsal could not have found it, and how it hid** | `activation_checkpointing` is the **declared-only** field: `declared_session_config` sets it **True**, explicitly, because "section 5.2's T4 fit arithmetic assumes checkpointed activations", while the development config the rehearsal trains has it **False** (the dev default, where memory is not the constraint). The rehearsal therefore ran the real session path, on XLA, and never entered a checkpointed block. Its own record says what it is - `declared_scale: false` - and this is the first thing that shape-level gap has actually cost: v1 of the price probe, fired at the declared shape, died before step one. |
| **Upstream's answer, which is the fix** | PyTorch/XLA ships `torch_xla/utils/checkpoint.py`, a **fork of that same file**, and its header says why: "PyTorch/XLA needs to add `optimization_barrier` before saving the input for the backward hence we slightly modify the upstream version". Its device lookups are `device_module = xm if device_type == "xla" else _get_device_module(device_type)` - i.e. the XLA device module where upstream assumes `torch` has one. It also documents its one constraint: `raise ValueError("XLA currently does not support use_reentrant==False")`. |
| **The fix** | A new `src/act/model/checkpointing.py` with one dispatch, `activation_checkpoint(device, fn, *args)`: the CPU and CUDA keep `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` **unchanged** - the call the GPU campaign's records were produced with - and `device.type == "xla"` routes to the fork, **without** the `use_reentrant` keyword, because the fork rejects `False` and defaults to the only implementation it has. All four call sites go through it (`dense`, `moe`, `v0_moe`'s block loop and its chunked scan), so the rule lives in one place - the same shape `runtime.step_marker` has for the lazy backend's step boundary, and for the same reason. Missing `torch_xla` is refused **by name** rather than by falling through to the upstream call. |
| **A second finding, and it is about the amendment itself** | Section 5.14's option 2 prices a **TPU/bf16** null arm against the two-T4 fp16 primary. `bf16` **is not a session dtype**: `V0SessionConfig` accepts `fp32` and `fp16` and raises for anything else. So that arm is not a flag on a push command - it is a **change to section 5.2's session model**, i.e. exactly the registration decision the amendment already was, now with a code change attached. And `fp16`, the declared dtype, is the one that needs `torch.autocast(device_type="xla", ...)` plus a `GradScaler` - two things no record in this repository shows running on a TPU. The runnable, registered option is **fp32**, so that is what v2 prices: the **slower** end of the two dtypes `tpu_probe` v5 measured within 6% of each other on this shape, which means the price errs **against** the queue rather than for it - the same direction the compile-inside-the-wall choice already errs. The record carries `dtype` and a `not_measured` line naming the gap, so the figure cannot be read as the amendment's own dtype. |
| **The diagnostic gap that made this expensive, now closed** | `tpu_probe._run_phase` recorded `f"{type(exc).__name__}: {exc}"` and nothing else, so a record from a queued run carried the symptom and not the site - and three phases' worth of `AttributeError: module 'torch' has no attribute 'xla'` is ambiguous across a dozen candidate lookups in torch alone. Every error entry now carries **`traceback`**, the frames as well as the message. That is one line of a record against a day of reading torch's source, and it is the *second* time a failure in this campaign was diagnosable only from a kernel log (v1's `Killed` was the first). |
| **Pinned** | `tests/test_activation_checkpointing.py` (**+6**) and `tests/test_tpu_session_rehearsal.py` (**+1**, 2091 in the suite): the rehearsal's `--checkpoint-activations` reaches the config the phases train and lands in the record, so the field that hid this cannot be silently off again; every non-XLA device keeps the upstream call with `use_reentrant=False`; the XLA call is the fork and is **never** passed `use_reentrant` (the `ValueError` the fork raises otherwise is the whole trap); an `xla` device without the package is refused by name; a checkpointed dense model routes **one call per block per step** through the dispatch; and so does a **V0 session** with the flag the declared config sets - one **null** arm, two blocks, two steps, four calls. That last one is the property whose absence let the defect reach a queue, and the liveness suite cannot pin it: a checkpointing call that silently did nothing would leave every loss identical and pass there. `tests/test_tpu_decision.py` now asserts the frames, not only the message, and `tests/test_tpu_session_price_probe.py` / `tests/test_push_tpu_session_price_probe.py` pin the dtype and pin that `bf16` is refused. |
| **The same defect, found and fixed locally the same day - and the route that makes that possible** | `torch_xla` runs XLA **on a CPU host** (`PJRT_DEVICE=CPU`), so the exact pair the Kaggle image runs (torch **2.8.0** / torch_xla **2.8.0**) can be rehearsed here in a throwaway `/tmp` env. Measured: the **pre-fix** call raises the recorded failure verbatim on a real `xla` tensor (`torch/utils/checkpoint.py` `_checkpoint_without_reentrant_generator` → `_get_device_module` → `getattr(torch, "xla")`), which turns the day of source reading into a two-line reproduction; the **fixed** path runs all four rehearsal phases with `activation_checkpointing: True` and `errors: []` - arm **754-776 tokens/s**, checkpoint a host payload (`["cpu"]`), resume **bit-identical** (9.06785535812378), handoff publishing a null arm at `search_flops 0, mutation_count 0`. What it cannot do is the declared shape: tried on the same host, the session phase's RSS climbed **0.5 → 33.27 GB in 195 s** and the process was killed mid-phase with no traceback and no further sample, so local runs are **development fidelity only** and the declared-shape price still needs the queued TPU run. (`dmesg` is not readable here, so the kill is read off the silence - the host copying a declared-size graph across the CPU client's boundary - rather than quoted from a kernel line.) |
| **The gap that let it reach a queue, closed at the source** | Both local dry runs before today were `--device cpu`, which cannot reach a `device.type == "xla"` branch - so they were never evidence about this path. `scripts/tpu_session_rehearsal.py` now takes **`--checkpoint-activations`**, which turns on the one field where the development and declared configs disagree in a way that matters (`declared_session_config` sets it True, the dev default is False), and the record carries **`activation_checkpointing`** so two rehearsal runs can be told apart. The runbook's new "Before firing at a TPU" section carries the recipe and both readings above. |
| **The state** | v1 cost **8.7 h** of queue and **~30 s** of TPU quota. Quota now: **TPU 16.77 of 20 h** (**3.23** used: the probe's 0.14, rehearsal v1's 3.02, rehearsal v2's ~0.04, this ~0.03) against **GPU 1.65 of 30 h**, both refreshing **2026-10-03**. The price probe is re-fixed and re-armed at **fp32** - code dataset republished at **`ecdde2491c03`**, kernel **version 2** pushed (`QUEUED`), the mount downloaded back and checked to carry the dispatch (so a *second* silent miss is not what is being waited on), and the watcher polling 24 h into `artifacts/tpu-session-price/output/`. §5.14's option 2 remains **unsigned** and now has two things to settle rather than one - the pairing (bf16 versus the primary's fp16) and the session model that would have to carry bf16 at all. |

The measurement that was meant to price the amendment instead priced its feasibility, which is what
"measure before committing 8.06 h" was for: 8.06 h of TPU time against a path that would have
died in its first phase, four seconds in, after the same queue.

**Disposition of section 5.14's option 2, taken on this evidence (2026-09-29): not adopted.** The
open question was never the price - it was whether the *pairing* the amendment needs is one the
registration can carry, and this run answered that it is not, on two measured grounds. (1) The arm
it names is **TPU/bf16**, and `bf16` is not a session dtype: `V0SessionConfig` admits `fp32` and
`fp16`, so the arm requires a change to section 5.2's session model *and* would then pair a
TPU-produced null loss against a two-T4 **fp16** primary loss - two registration changes, not one
flag. (2) The alternative the path can actually run is **fp32**, which is the *slower* end of the
two dtypes `tpu_probe` v5 put within 6% of each other - i.e. the amendment's only runnable form
carries a dtype difference from the primary as well. Against that, what the move buys is calendar:
the GPU week is spent (**1.65 of 30 h** remaining) until the **2026-10-03** refresh, so a queue
that leaves the GPU for TPU hours would matter to a block that is trying to fit one week - and it
is not worth two changes to the registered protocol. The TPU week is therefore left **unspent at
16.77 of 20 h**, and the re-armed price probe is informational from here: if it lands it bounds
the cost of a variant this entry does not adopt, and if it never runs nothing is blocked on it.
Neither the primary block (`15024-15035`, 6 of 12 complete) nor the driver's lock is affected;
the next real work in this campaign is the GPU refresh.

## The session path runs on the TPU, and its own record published an impossible rate (2026-09-28)

**Live update, 2026-09-28 19:36Z:** the rehearsal came back **`COMPLETE`** - all four phases in
`completed`, `errors` empty - after queueing **~10.8 h** and running **~2.5 min**. The whole
account is committed under `artifacts/tpu-session-rehearsal/` (`NOTES.md`, with the five fetched
files' sha256s). The campaign's own queues are untouched. *(**Corrected 2026-09-29:** "the driver
still holds the lock" was wrong at the time of writing - the process had already exited after
declining to bill a partial session, which is why nothing resumes at the refresh until the campaign
is relaunched. See the leading entry.)*

| | |
|---|---|
| **The four phases, measured** | `environment` **23.2 s**, `session` **45.9 s**, `resume` **1.6 s**, `handoff` **1.3 s**, off the platform log's own clock; the record is printed at 132.2 s of a 150.9 s kernel. |
| **The v1 defect, confirmed fixed where it was found** | v1's session phase was three hours of silence and then `Killed`; v2's is **45.9 s** and returns, and the same twenty development-fidelity steps take 24 s on a CPU - so the unmarked step was the whole difference, and this is the first measurement that `runtime.step_marker` does what it was written to do. The instrumentation that could not report v1 now reports v2: `watch.session` samples **7,045,312 kB** at 0 s and **11,986,972 kB** at 30 s of a phase that *finished*, which is what working looks like against v1's climb to the kill. `device_bytes_used` is `null` in every sample (and in every `tpu-probe` record back to v4), so this runtime does not expose it and the host's resident set is the growth signal that carries the watch - measured working, not assumed. |
| **The path's own contracts, each read off the artifact** | `resume.resumed_from_a_checkpoint` **true** with `same_loss_as_the_first_run` **true** and validation loss **9.066792488098145**, bit-identical to the fresh run - the contract a deadline stop depends on. `session.checkpoint_tensor_devices` is **`["cpu"]`**, so `_cpu_payload` is doing its job on a lazy backend and the checkpoint the TPU wrote is a host payload. `session.checkpoint_revision` and `manifest_code_revision` are both **`f3ed48f37ab1...`**, so a checkpoint separated from its manifest says what wrote it. And the §5.14 clause's own input came back off a real publish: `handoff.null_arm` is `{final_loss 9.066792488098145, mutation_count 0, search_flops 0}` with `summary_status: complete`. |
| **A second defect this record found, in the resume path** | The published `seed-summary.json` - the file §5.14's clause consumes - carries `tokens_per_second` **6871947673.6** over `wall_time_seconds` **1.1920928955078125e-05** for the resumed arm. `_state_from_payload` restores `training.tokens` **cumulatively** (81,920 = 20 x 8 x 512) while `wall_time` measured only this process's segment, and the rehearsal resumes from a checkpoint at step 20 of 20 - so the loop's `range(21, 21)` is empty and a token count is divided by a bare pair of clock reads. It is not only that case: *any* partial resume divides the whole token count by one segment's seconds, and a partial resume is what a deadline stop produces (`V0SessionInterrupted` at a checkpoint boundary, then the queue resumes), so the null queue this was rehearsing for would have published inflated rates on every seed that survived a stop. **Fixed:** `_SessionState.wall_seconds` is the arm's own training wall, stamped into `v0_session.wall_seconds` with every checkpoint before it is written, restored on resume and summed with the segment's - a resumed arm now reports the whole arm's rate (81920 / 40.70 = **2,013**, the number the same arm had just measured) and a zero-step resume reports that instead of an empty loop's. |
| **Pinned** | `tests/test_v0_session.py` (**+1**, 2071 in the suite): a fresh run's checkpoint carries a positive `wall_seconds` that grows with the arm, a resume with nothing left to run reproduces the arm's rate rather than an empty loop's, and a mid-run resume extends the wall instead of restarting it. Run against the pre-fix tree the test fails on the payload (`KeyError: 'wall_seconds'`), which is what makes it a pin rather than a description. |
| **What it does not price, and the amendment it unblocks** | It says nothing about the declared-scale rate on the session path: **2,013 tokens/s** here against the **15,034** the probe's step loop measured, a gap that is fidelity (20 steps pay a compile and a per-step sync that 6,000 amortize) and not a number this record can settle - so §5.14 option 2's **8.06 h at the 2.0x bar** is still the port rule's figure, not a measured session. What did change is that it was the *last* objection standing: the registered case against moving the null queue to the TPU was that the port verdict measured a step loop while a session adds activation checkpointing, the optimizer and scaler state, the per-arm RNG contract, a checkpoint saved **and resumed** across a stop, the publish and the placement - and each of those has now run once on XLA. *(**Corrected 2026-09-29:** that list was one item too long. Activation checkpointing had not run on XLA and could not: the rehearsal's development config carries it **off** and the declared config carries it **on**, and the declared-shape price probe died on exactly that field in every phase that trains - `torch.utils.checkpoint` resolves its device module as `torch.xla`, which does not exist. See the entry at the top of this file. The rehearsal's own labels said `declared_scale: false` and its record was read as saying so; the claim it did not support was this sentence's.)* The TPU week stands at **16.77 of 20 h remaining** (3.23 used: the probe's 0.14, v1's 3.02, v2's ~0.04, the price probe's ~0.03) against **1.65 of 30 GPU-h**, both refreshing **2026-10-03**, so the clock fits it. The option is still **not taken**, and for the reason recorded in the row below rather than for want of a rehearsal: it changes the do-nothing arm's hardware and dtype against a two-T4 fp16 primary, so it is a **registered amendment** - the operator's to sign, not a scheduling knob. |

The rehearsal remains what its own record says it is - `rehearsal: true`, `declared_scale: false`,
not a port and not a Gate-4 result - and it moved no queue: a defect that would have cost a GPU
session a night was found and a second one was fixed on a TPU week that was otherwise idle.

**Live update, 2026-09-28 20:54Z:** the one number §5.14's option 2 is priced on has now been
put to the path that would run it. The price - **8.06 h** - is the port rule's, measured by
`tpu_probe.py` v6 on a **hand-built step loop** at the declared shape, and the rehearsal's own
record declines to price anything: its `not_measured` says the development shape's tokens/s is
"a path measurement and not a schedule input". So
`scripts/tpu_session_price_probe.py` runs the real `run_v0_session` on
`declared_session_config()` - 285M total / 134M active, batch 2 x context 2048, fp16,
checkpointed activations - as a single **null** arm, and prices the result through
`tpu_probe.queue_tpu_hours`, the campaign's own arithmetic, so this figure and the 8.06 h bar
cannot be computed two different ways. It reports the multiple, the queue's hours at that rate,
the same hours if the eight chips do not scale, and whether it fits the week. Kernel
`act-tpu-session-price` **version 1** is `QUEUED` (TPU on, GPU off) from the code dataset
republished at **`98d5574`**; the mount was downloaded back and carries `revision.txt` at that
revision and the probe beside it, and
`scripts/fetch_tpu_session_rehearsal_when_done.sh` is polling for 24 h with `KERNEL`,
`OUT_DIR` and `PUSHER` overridden, so the record lands in `artifacts/tpu-session-price/output/`
on either terminal state.

Three things keep it from being read as more than it is. It is labelled - `declared_shape:true`
beside `declared_seed:false`, a scratch handoff root rather than any campaign dataset, and
`not_measured` naming what it does not establish. It trains **240** steps, not a seed's 6,000, so
its figure is a **floor**: the declared graph's compile sits inside the wall it divides by,
where a 6,000-step seed pays it once. And it costs minutes of the week's **16.80 h remaining**
beside a GPU week at **1.65 of 30**, both refreshing 2026-10-03 - so the measurement it buys
costs the campaign nothing it could otherwise spend before the refresh.

---

## The TPU rehearsal found the defect before it moved a queue (2026-09-28)

**Live update, 2026-09-28 05:00Z:** the rehearsal came back **`ERROR`** -
`sid123123234/act-tpu-session-rehearsal` v1, which queued for **8.9 h** and then ran **3.02 h**
before Kaggle killed it - and that is the reading, not a lost run. The campaign's own queues are
untouched: the driver still holds the lock (pid 63681, 16 h up) and is sleeping to the
2026-10-03 refresh with **1.65 of 30 GPU-h** left.

| | |
|---|---|
| **What came back, and what it did not** | The fetched record reads `completed: ["environment"]`, `errors: []`, and the fetched log carries `--- phase session` followed by **three hours of nothing** and then `Killed`. `errors` is empty because a phase's outcome is written when the phase *returns* - and this one never did - so the account is the phase boundary. No `session` entry survived (its entry and log were in memory), and the kernel's output holds no checkpoint: the loop never reached step 10 of 20. The fetch's `verdict` names exactly that state: `partial rehearsal: ran environment; session, resume, handoff did not run (no error recorded)`. |
| **What the environment phase did measure** | The hardware and the runtime are fine: `v5litepod-8`, eight bound devices (`xla:0`), `TPU_CHIPS_PER_PROCESS_BOUNDS 2,4,1`, torch **2.8.0+cpu** with torch_xla **2.8.0**, and a tensor round-trip to the bound device (`.sum() == 8.0`) in 14.1 s. The failure is in the **training loop**, which on this host is not the loop the port rule was priced on. |
| **The defect: the step was never marked** | XLA is lazy - the ops of a step are traced into a graph that does not execute until a step is marked - and `_train_arm` never marked one, so the loop traced the whole run and executed it as a single program. The probe's measured loop (15,034 tokens/s, the `PORT` verdict) marks after `optimizer.step()`; the runbook's own note is that without a marker the "steady step" being timed is graph construction. The same twenty development-fidelity steps take **24 s on the CPU**, which is what makes three hours a defect rather than slowness. `runtime.step_marker(device)` now returns `torch_xla.sync` on XLA (falling back to `xla_model.mark_step` on an image that predates the alias) and a no-op on CPU and CUDA, and the loop calls it after the optimizer's update - so the CPU and CUDA paths the campaign actually runs are numerically unchanged. |
| **And a rehearsal that could not report a hang** | Every phase could hang silently, for the reason above: the session's log lines live in the record's list, and the record is only written when a phase returns. The session's lines are now **printed as they arrive**, every phase runs under a `_Watch` that samples the host's resident set and the device's bytes in use every 30 s (printed into the kernel log *and* rewritten into a sidecar `*.watch.json` beside the record, through a rename so a killed process leaves a whole file), and a per-phase `faulthandler` deadline (1800 s for the session, 600-900 s for the rest, `TPU_SESSION_DEADLINE` overriding) dumps every thread's stack before the platform's own kill. A killed run now leaves a trace with a shape - flat is working, climbing is accumulating - and a stack that names the call it hung in, which is the artifact this run did not have. |
| **Pinned** | `tests/test_training_distributed.py` (**+3**): the marker is the same no-op object on CPU and CUDA, is `torch_xla.sync` on XLA, and falls back to `xla_model.mark_step`. `tests/test_v0_session.py` (**+1**): one marker per arm built from the arm's device, called **once per step** - so a refactor that hoists it out of the loop is caught here rather than on hardware nobody can attach a debugger to. `tests/test_tpu_session_rehearsal.py` (**+4**): the session log is printed and kept, the watch's samples and sidecar (the first sample is the baseline a reader compares against), the deadline and watch settings are part of the record, and every phase arms and cancels a deadline and lands its samples. |

This is a rehearsal of the *path*, not a port and not a Gate-4 result, and it moved no queue -
which is the point: a defect that costs a GPU session a night was found on a TPU week that stood
at **19.86 of 20 hours idle**. The next run is the same rehearsal, re-armed (**and returned above**: v2 ran all four phases):
kernel `act-tpu-session-rehearsal` **version 2** is `QUEUED` (commit **`f3ed48f`**), the code dataset
was republished at that revision and both the mount's `revision.txt` and the shipped
`runtime.py`, `v0_session.py` and `tpu_session_rehearsal.py` were downloaded back and checked to
carry the marker and the watch, and
`scripts/fetch_tpu_session_rehearsal_when_done.sh` is armed for 24 h of polls so the record
lands in `artifacts/tpu-session-rehearsal/output/` on either terminal state. The driver still
holds the lock and still sleeps to the 2026-10-03 refresh.

---

## The TPU holds a session path to rehearse, and a checkpoint now says what wrote it (2026-09-27)

**Live update, 2026-09-27 17:09Z:** the rehearsal is **pushed** -
`sid123123234/act-tpu-session-rehearsal`, version **1**, `QUEUED`, TPU on and GPU off, staged
from the code dataset republished at **`9105dc4`** - with
`scripts/fetch_tpu_session_rehearsal_when_done.sh` armed to keep whatever it lands in
`artifacts/tpu-session-rehearsal/output/`, on either terminal state. The campaign's own
queues are untouched: no session kernel is live (the guard checked before republishing), the
driver still holds the lock and is sleeping to the 2026-10-03 refresh.

The probe answered a different question than the one the amendment needs. `tpu_probe.py` v6
returned **PORT** on the *declared shape's step loop* (15,034.2 tokens/s, 2.96x the T4 pair,
recompiles 2.07% of a per-core training hour), and §5.14's option 2 - the **null queue**, twelve
single-arm seeds, **8.06 h of TPU time at the rule's own 2.0x bar** against a **19.86/20 TPU-hour
week** nobody else is using - is what that verdict was gating. But a session is not a step loop:
it adds activation checkpointing, the optimizer (and a GradScaler when the dtype is fp16), the
per-arm RNG contract, a checkpoint saved **and resumed** across a stop, the seed handoff publish,
and the two-arm placement. None of that has ever run on XLA. So the first move is the cheap one -
run the *path*, not assume it - and it is now staged to run.

| | |
|---|---|
| **`scripts/tpu_session_rehearsal.py`** | Runs the real `run_v0_session` at development fidelity (declared `False`, one **null** arm - the queue the amendment moves) in four contained phases: `environment` (torch/torch-xla versions, the platform's TPU topology block, device count, a bound device, and a tensor round-trip), `session` (the path, with a revision written into every checkpoint and the file then read back off the disk), `resume` (the same session from the checkpoint it just wrote - the contract a deadline stop depends on, and the first thing XLA can break), and `handoff` (`publish_seed` into a temporary dataset root, then the null clause's own fields read back out of the published summary). Every phase is contained (`tpu_probe._run_phase`), the record is saved after each one, and each phase files its own entry and log *before* it starts - so a run that dies inside the session still carries the environment reading rather than only an exception. |
| **It is labelled so it cannot be read as a result** | `rehearsal: true`, `fidelity: "development"`, `declared_scale: false`, and `not_measured` names the three things a reader could otherwise mistake it for: per-core parallelism (the probe's headline, measured there, not here), declared-fidelity throughput (the dev shape's tokens/s is a path measurement, not a schedule input), and the two-arm pair placement (the null queue is single-arm by construction). No queue moves on this record, and the kernel is pushed with `enable_tpu` **on** and `enable_gpu` off so it cannot answer on the wrong accelerator. |
| **A checkpoint is written from the host now** | `save_v0_checkpoint` routes its payload through `_cpu_payload`, which moves every tensor to the CPU before `torch.save`. On the CPU and CUDA paths that is the copy `torch.save` makes while serializing, and on the host it is a no-op (`Tensor.cpu()` returns `self`); on XLA it is a **fix**. `torch.save` writes XLA tensors as they are rather than copying them out - torch_xla ships `xm.save` for exactly that reason - and both ways that fails are unacceptable here: a payload of XLA-tagged tensors is unreadable on a host with a different device count (pytorch-lightning #2700; pytorch/xla #2364), and a lazy tensor whose storage was never materialized can serialize as the **wrong bytes**. A checkpoint that loads, checksums and publishes without being the training it names is the failure this project treats as worse than a crash, so the conversion is explicit and the `session` phase records `checkpoint_tensor_devices` off the file the device just wrote. |
| **`resolve_device` knows the device** | `xla` resolves to the *runtime's* default device (`torch_xla.device()`, which carries the ordinal `xla:0` rather than a bare type, so the resolution happens at startup and the arm records its placement), and it **refuses** with a named error when `torch_xla` is absent - the alternative is a name that resolves and then fails inside the first `.to()`, hours into a queued kernel. |
| **Pushed through the probe's own guard** | `scripts/push_tpu_session_rehearsal.py` republishes `act-lab-code` (a script kernel receives **only** its own `code_file`, so the rehearsal and the `src/` it imports both arrive there), and it does so through `push_tpu_probe.publish_code_dataset` - imported, not re-implemented: exit 5 and no push while a campaign session is mounted, which is the defect that cost the 2026-09-21 session. Its `--fetch --save` keeps the record *and* the log and prints `verdict(record)`, which holds "nothing ran" apart from "these phases ran and those did not". |
| **Pinned** | `tests/test_tpu_session_rehearsal.py` (new): the labels and the dev config, the phase order, the entry-and-log-filed-before-the-work property (a session that raises mid-step still lands its line), the null-clause guard recomputed from the published artifacts (six cases, including a mutating "null" arm), and one end-to-end CPU dry run whose record is the shape a fired run produces. `tests/test_push_tpu_session_rehearsal.py` (new): the accelerator flags, the three copies of the record name, the shared Torch-XLA bootstrap, the guard refusing before any push, the fetch keeping record + log + verdict, and the watcher fetching on **both** terminal states. `tests/test_v0_session.py` (**+2**): the save routes its payload through the host conversion, and the walk reaches every tensor in a nested payload while leaving device-free values alone. `tests/test_training_distributed.py` (**+1**): `xla` resolves, or the missing package is named. |

The block remains 15024-15035 in `act-v0-5-2` (primary) / `-control` / `-null`; nothing here
changes a seed, knob, dataset or criterion, and the null queue has **not** moved: the
rehearsal is what makes that decision decidable, and it is not itself a Gate-4 result.

---

## The first session whose restore worked advanced the queue, and nobody was watching (2026-09-27)

**Live update, 2026-09-27 17:30Z:** kernel v11 is `COMPLETE` and moved the primary block
**3 -> 6 of 12** - seeds **15027, 15028, 15029** published beside the restored 15024-15026 -
for **4.92 GPU-h** (quota 6.57 -> 1.65 of 30, refreshing 2026-10-03). It is the first session
in four that **restored** published state instead of re-training the block: the 403 misread is
fixed, and its own log shows the fix working - `remote restore: ... downloaded in 8.7 min
(25.23 GiB on disk)`, `restored 4 seed version(s) ... [15024, 15025, 15026, 15027]`, then three
`complete version exists, skipping` lines and three trained pairs. **No driver is running.**
The workstation rebooted at **15:46:08Z**, killing the waiting supervisor mid-poll; the session
it had already pushed finished on Kaggle at ~15:54Z and published at 15:57-15:59Z with nobody
watching. The campaign is relaunched with `--wait-for-quota` (registered 7 h window) so the
2026-10-03 refresh is spent automatically; do not start a second driver.

| | |
|---|---|
| **What v11 measured** | Pair time **1.4175 h** (benchmark 5,182 tokens/s, `fp16`, two T4s), 4.92 h wall for 3 pairs and the exit publish of **6 seed version(s)**; `seed 15030: remaining 1.82 h does not fit pair 1.42 h + 45 min reserve (rule 1)` is why it stopped. Every arm sits on the uniform floor `ln(32768) = 10.3972`: 15027 `+7.9e-05`, 15028 `-2.1e-04`, 15029 `-3.1e-04` (`random - adaptive`), so across all six published seeds the adaptive arm is **2 wins, 4 losses** and every seed's `predicted_cost_ratio` is the benchmark's own `0.000941636252059042` (**0.0942%** against the registered 5% criterion). The harm guard rejects at 10 losses of 12, so at this floor the registered clauses have almost nothing to read - the recorded property of a synthetic-token block, not a new finding. |
| **The evidence is kept, and it had to be fetched by hand** | `kaggle kernels logs` serves the *latest* version only, and nothing called `keep_session_evidence` because the driver died with the lock free. So the platform log, the runner's `session-log.json`, the publish receipt, the benchmark, the child's tee and the three seed records were fetched while v11 was still the latest version and committed under `artifacts/v0-session-logs/` with their sha256s (`NOTES.md`). There is no live capture for this run: the tailer's `mkdtemp` directory was destroyed by the reboot - named, because v7's capture exists and its absence here is a difference a reader would otherwise have to guess at. |
| **The report that made the death look healthy** | `campaign_health.py` read `WAITING_ON_QUOTA` - "the week cannot fit a 6.5 h session ...; the driver's `--wait-for-quota` covers this" - at a moment when **no process held the lock**: the lock appeared on the kernel line only when it was *held*, so "a driver is sleeping to the refresh" and "nothing will resume this campaign" printed the same way. Measured on the live state at 16:03Z before anything else: 6/12 complete, kernel `COMPLETE`, 1.65 h left, lock free. The report now always prints the lock (`driver lock     : not held - nothing is waiting on this queue`) and the quota verdict names which of the two it is (`a driver is waiting on it (pid N)` against `nothing holds the driver's lock, so nothing resumes at the refresh until the campaign is relaunched`). The verdict and its exit code are unchanged: the quota is short either way. |
| **And a provenance hole the same fetch exposed** | The published records for **15024-15026 are not kernel v7's**. Measured: the live dataset's `seed-15024/reports/seed-summary.json` checksums to `a07be977...` (its own manifest) and matches the copy in v11's output, while the copy committed from v7's output is `d27882c8...` and a **different training** - mutations at `L0E1`/`L11E4` against `L10E1`/`L5E7`, architecture hash `64cd541996cd4609` against `291178c1d5309057`, adaptive loss `10.399189949` against `10.398051500` (15025 and 15026 likewise). v11 did not write them - its own session log says `skipped_complete` for those three and its written records match its output byte-for-byte - so they came from the sessions in between, **v9 or v10**, the two documented as having restored nothing, retrained the block from its first seed and published over it. Which of the two is **not recoverable**: v9 ran the same revision as v7 (`2c0cf4ebfa61`), so these would be v10's if a declared-scale fp16 seed reproduces exactly - and v10 did run a **different tree** (the driver's log shows it republishing the code dataset from `1b0f210b7b7f` mid-campaign, naming `src/act/model/moe.py`). Exact reproduction is measured at development fidelity, not at declared scale, and nothing here pins cuDNN (`torch.backends.cudnn.deterministic` is never set). So the artifact supports "one of the two 403-era sessions", not which one - which is the same hole as the `unknown` in the next row. |
| **Why that was unmeasurable from the artifact** | Every published manifest says **`code_revision: "unknown"`** (all six downloaded and read), and the v0 checkpoint payload carries **no revision and no timestamp** either - `version`, `model`, `optimizer`, `torch_rng_state`, `scaler`, `training{step,tokens,generator_state,genome}`, `v0_session{flops,mutations,history}` and nothing dated - so the dataset cannot attribute a seed to the code that trained it. The cause is a one-line gap: `act.training.cli._git_revision()` asks `git rev-parse`, a kernel imports `src/` from a **mounted dataset with no `.git`**, and the answer was therefore always the literal `unknown`. The dataset *does* carry the commit - `code/revision.txt`, measured `80699f945648021b0111c9005c79b9b7048abb66` - beside the package tree; nothing read it. |
| **The fix** | `_git_revision()` falls back to that marker (git still answers first on a workstation, so nothing local changes), and the session prints `code revision: ...` at startup so the kernel log carries it too. The remaining primary seeds (15030-15035) and both later queues will therefore record the revision that trained them. It cannot retroactively fix 15024-15029 - their manifests stay `unknown`, which is the honest label for records whose revision is only deducible. |
| **What it changes about the block, stated rather than smoothed** | No criterion moves: each pair is internally consistent, every seed ran the declared configuration, and the block-treatment check that v11 logged (`preserve_router_on_rebuild=True, new_expert_initialization='clone'`) matches. But the first three records were plausibly written under a different revision than the last three, the primary block is therefore **not provably revision-homogeneous**, and the control/null queues will run under the current revision - so the verdict's cross-queue pairings (adaptive loss from the primary record, null loss from the null queue) compare records that may not come from the same tree. Nothing here measures a difference; what is measured is that the primary records are `unknown` and that 15024-15026 changed in between. Re-training 15024-15026 under the current revision would make the block homogeneous for **~4.3 GPU-h**; it is **not** done - re-running published seeds to taste is what the pre-registration forbids, and the published block is the block. |
| **The week's other clock is idle, and the null queue is where it could go** | Measured 2026-09-27: **TPU 19.86/20 h remaining** (0.14 h used, by the probe) against **GPU 1.65/30**, both refreshing 2026-10-03. The priced route is §5.14's option 2 - the **null queue** is 12 single-arm seeds and bills **16.97 h** of T4 wall clock, i.e. **8.06 h of TPU time at the port rule's own 2.0x bar** (the v6 step loop measured **2.96x** the T4-pair rate, which would put it near 5.7 h). Its own entry called it "gated on the port verdict", and that verdict now exists: v6 returned **PORT**. It is still **not taken**, for two reasons stated rather than assumed - moving the do-nothing arm to different hardware changes what it is paired against (the verdict pairs the null loss with the *primary* adaptive loss, so the clause would compare a bf16 TPU run with a fp16 T4 one: a **registered amendment**, which is the operator's to sign, not a scheduling knob like the session window), and the session path has never run on TPU at all: `tpu_probe.py` measured a step loop, while a session adds activation checkpointing, the optimizer/scaler state, the two-arm placement, the restore and the exit publish - and the probe's own record carries the caveat that its per-rank initializations shared the process RNG. The honest first step is therefore a **TPU rehearsal of the session path at development fidelity** (no GPU or TPU quota: the `gate5_rehearsal.py` shape, on one tiny session), which is what would make the amendment decidable. |
| **Pinned** | `tests/test_campaign_health.py` (**+2**, 28 in the file): a quota wait with no lock says so in words and in the lock line, and a lock whose holder cannot be read still reads as *held*. `tests/test_v0_runner.py` (**+3**): the marker beside the mounted tree answers when git cannot, an explicit root is answered from that root alone, the marker's own name is the one `kaggle_queue.stage_code_dataset` writes, and a workstation still records git's answer. **2035 tests pass.** |

The block remains 15024-15035 in `act-v0-5-2` (primary) / `-control` / `-null`; no seed, knob,
dataset or criterion changed, and there is still no verdict. Control and null are next in the
registered sequence.

---

## The health report counted a seed the queue had not finished (2026-09-27)

**Live update, 2026-09-27 11:05Z:** the primary session kernel is version **11**, `RUNNING`
since 10:59Z (queued from 10:12Z), mounting the code dataset published at the restore fix
(`80699f9`), with **6.57/30 GPU-h** left. `campaign_health.py` now reports the block as **3 of
12 complete (15024, 15025, 15026); stepped but not finished 15027 (in-progress)** and resumes
from seed **15027** - the seed the runner's own skip rule will start at. The platform's log
stream is answering **500** for this run so far (the driver's live tailer records the failures
in its capture), which is why the in-session restore line for v11 is not yet quoted here; the
session's own `session-log.json` reaches the repo through the kernel output either way
(`kaggle_queue.keep_session_evidence`).

**What is left to buy.** After this window the block needs **8 primary pairs**, then **12
control** and **12 null** pairs, each queue's 12 pairs at the measured **~1.7 h pair + 45 min
reserve**: about **54 GPU-h against a 30 h week**, so the campaign cannot finish before the
two refresh cycles at 2026-10-03 and 2026-10-10 and the driver *should* sit in
`WAITING_ON_QUOTA` between them. That is the registered scale, not a fault to push sessions
at - and the fitted window is what keeps the tail of a week from expiring unused.

| | |
|---|---|
| **What it said, and what it says now** | Both measured on the live dataset, minutes apart. As committed at `aa28172`: `4 of 12 complete [15024, 15025, 15026, 15027] next 15028 partial []`. Now: `3 of 12 complete [15024, 15025, 15026] next 15027 unconfirmed [15027] {'15027': 'in-progress'}`. The written surfaces were already right - the report's §5 index row says "three of 12 primary seeds (15024-15026)" - so the tool that generates them was the one that disagreed. |
| **Why the listing cannot answer it alone** | The listing carries file *names*, and a seed read as complete there when every declared arm's checkpoint name reached the declared step. A snapshot is written at each checkpoint while the evaluation runs after the last one, so a seed whose window closed in between has **both arms at `checkpoint-006000.pt`** and a record that still says `in-progress`. Measured on seed 15027: its `reports/seed-summary.json` is **1,676 B** with `status: in-progress`, `interrupted_step: 6000` and **no `final_loss` for either arm**, against seed 15024's **5,051 B**, `status: complete`, both arms' losses present. |
| **Why that matters** | The other two readers of this question already agreed with each other and disagreed with the report: the runner skips a seed only when that record says `complete` (`published_seed_status` -> `status == COMPLETE_STATUS`), and the verdict's `_seed_numbers` collects a seed only when the same field says `complete` (`v0_verdict.py`, which raises `seed N is not a complete version` otherwise). So the report named **15028** as the next seed while the next session will start at **15027**, and `verdict: COMPLETE` - the line that authorizes driving the *next* queue - would have fired on a block whose own record calls a seed unfinished. No quota was spent on this, because neither the runner nor the verdict ever read the listing; what was wrong was the surface an operator and these documents read. That is the state a queue reaches most often: the window closes right after a session's last checkpoint. |
| **The fix** | `stepped_seeds` names the seeds the listing can only *nominate*; `seed_statuses` fetches each candidate's own record (one `-f` download per candidate, **1.1-1.3 s** measured against the live dataset, never the ~24 GB session dataset); and `seed_progress(seeds, statuses=...)` now requires those statuses and splits three ways - `partial` (an arm short of the declared step), `unconfirmed` (stepped, record does not say `complete`; `unconfirmed_status` tells `in-progress` from an unreadable `None`), `complete`. `missing` and `next_seed` count both unfinished buckets, so the complete verdict needs every seed's own record. Only candidates are fetched: a seed already short of the declared step cannot be changed by its record, and the healthy case is the model default in the tests. |
| **Two smaller holes in the same report** | Measured on the **control** queue 2026-09-27, which has no kernel and no dataset yet: the status line printed `session kernel  : ` and nothing else, and the quota verdict said `cannot fit a 7.0 h session` while the campaign declares **6.5 h** windows (`HOURS`) - a verdict computed against a window no session uses, printed at the moment an operator decides whether the campaign is stuck. The status line now reads `unknown (not pushed yet, or the status query failed)` (the empty token is both states, and the driver reads it as *push*), and `run_gate4_campaign.sh` passes the window it fitted to the report (`--window "$HOURS"`), so the boundary line now reads `cannot fit a 6.5 h session`. |
| **A hazard found while fixing it** | The launcher change (`--window`, above) was committed while an instance of `run_gate4_campaign.sh` was running, which bash does not tolerate: it parses a script incrementally and keeps a **byte offset** into the file, and this instance's offset was **5317** - exactly the end of its outer `for` loop - so the inserted lines shifted everything the instance had not yet read. Measured on a byte-faithful copy of the script with the same insertion mid-run: the four queues all still ran and were sequenced correctly, and then the closing echo misparsed - `syntax error near unexpected token ')'`, exit **2**, no `campaign finished` line. The parsed loop is in memory, so the blast radius is the tail; the rule is now in the runbook banner: land launcher changes when no instance is live. |
| **Pinned** | `tests/test_campaign_health.py`: 15027's live shape (both arms at 6000, record `in-progress`) is not complete and is named in `missing`/`next_seed`; an unreadable record is `unconfirmed` with a `None` status rather than complete; the report line carries the seed and its own status; a stepped-but-unfinished tail seed keeps `verdict: COMPLETE` away; the wiring test asserts `seed_statuses` is asked for the candidates only; an unpushed kernel's empty status is named in the report rather than printed as a blank; and the launcher is pinned to pass the fitted window to the report. The two step-based tests now pass statuses explicitly, with the reason recorded in the new one. **2030 tests pass.** |

This is a reporting fix, not a verdict: no seed, knob, dataset or criterion changed, and the
block remains 15024-15035 in `act-v0-5-2` (primary) / `-control` / `-null`.

---

## Every session was re-training the block: a 403 that read as "nothing published yet" (2026-09-27)

**Live update, 2026-09-27 10:12Z:** the campaign driver is running again, this time with a
fitted **6.5 h** window (the week has **6.57/30 GPU-h** left, refreshing 2026-10-03) and
`--wait-for-quota`, so it sleeps to the refresh rather than letting the tail expire. The code
dataset was republished at the restore fix (`80699f9`) and session kernel version **11** is
`QUEUED`. The primary block is **3 complete + seed 15027 in-progress** as published; the
runner will re-train 15027 because a pair is atomic and an in-progress snapshot is forensic
only.

**What the stop was.** The previous driver ended at 2026-09-27T03:30:28Z with exit **4** - "a
session completed without advancing the queue (4 -> 3 of 12 seeds)", having spent 6.60 GPU-h.
That exit is correct; the reading it was based on was not.

| | |
|---|---|
| **The session's own account, verbatim** | v10's platform log: `Dataset URL: https://www.kaggle.com/datasets/sid123123234/act-v0-5-2` at `t = 223.0 s`, then `no remote version of sid123123234/act-v0-5-2 available yet; training from scratch (atomic pairs make this safe)` at `t = 223.1 s`. It then trained **15024, 15025, 15026** and published 15027 as an in-progress snapshot. The dataset had held complete versions of 15024-15026 for hours and of 15027 since v9. |
| **So the queue never advanced** | Every session restored nothing, so every session started at the first declared seed and re-trained seeds the queue already had - at ~1.5 h per pair, a whole window's quota per session - and *published over* them: v10 replaced v9's **complete** version of seed 15027 with an **in-progress** one. v9's own numbers say the same: 4 completed pairs and no `skipped_complete` entry, which is only possible if it saw no published state ~45 min after v7 had published it. |
| **Why the code read it as "missing"** | `_is_missing_dataset_error` accepts any `requests.HTTPError` whose response is **403 or 404**. Measured 2026-09-27 against the live API: downloading an id that does not exist answers **403** (`sid123123234/act-definitely-missing` and another account's dataset both do), so the error cannot distinguish "nothing published yet" from "will not serve it" - and the code treated both as the bootstrap state. |
| **It is not a permissions problem** | From the workstation the same account lists the dataset's files and mints a download URL for it; from inside a kernel the free CPU `restoreprobe` re-run at 2026-09-27 restored **4 seed version(s) [15024, 15025, 15026, 15027] in 12.4 min (25.24 GiB)** - the same call that failed in the sessions. So the path works; what the failing sessions hit is transient or version-state related, and **the error was never logged**, which is why it survived three sessions. |
| **A window in which that 403 is expected** | A pushed version's files appear in the listing **5-8 min after the push** (the current version, pushed 03:27:40Z, carries 03:32-03:34 file times). A restore issued in that window can answer 403 for a dataset that plainly exists. Stated as the mechanism the retry exists for; it is not yet measured directly. |
| **The fix** | `restore_latest` now classifies a download failure by asking the **owner's own listing** (`dataset_exists`: `dataset_list(mine=True)`, paged, ref-matched) instead of trusting the error code; "no such dataset" is still the bootstrap state and is not retried, an existing dataset is retried **4 times over 6 min**, and if it still will not download the session **raises** - the runbook's own rule ("a session that cannot get its requested remote state must stop here, not mid-pair"), which the 403 misread had been bypassing. Every failed attempt now logs its exception type and text. |
| **Pinned** | `tests/test_kaggle_remote.py`: the bootstrap case does not retry and does not back off; an owned dataset that will not download retries `RESTORE_ATTEMPTS` times and then raises with the error text; a transient 403 that clears restores normally; and `dataset_exists` pages past a first page that does not carry the ref. `dataset_exists` was additionally exercised against the live API with both the workstation's `kaggle` 2.2.4 and the kernel's 2.0.2 (existing -> `True`, missing -> `False`). 2021 tests pass. |

This is dry-run bookkeeping, not a verdict: no seed, knob, dataset or criterion changed, and the
block remains 15024-15035 in `act-v0-5-2` (primary) / `-control` / `-null`.

---

## Gate 4 resumption authorized; primary session v9 running (2026-09-26)

**Live update, 2026-09-26 19:12Z:** Kaggle reports the existing primary kernel v9 `RUNNING`;
`campaign_health.py` confirms the campaign driver lock (PID 3295862) and **14.68/30 GPU-hours**
remaining, resuming at seed 15027. The TPU probe is `COMPLETE`. The launch snapshot and probe
result follow below. No new Gate-4 kernel was launched for this update.

---

## TPU probe v6 completed; registered decision is PORT (2026-09-26)

Kernel v6 completed all five pre-registered phases on Kaggle's v5litepod-8. The versioned record
and platform execution log are preserved as `artifacts/tpu-probe/output/tpu-probe-v6.json` and
`artifacts/tpu-probe/output/act-tpu-probe-v6.log`; the mutable `tpu-probe.json` name was not
kept. The pre-registered rule returns **PORT**. This is a TPU hardware-port decision, not a
Gate-4 result, amendment, or authorization to move a campaign queue to TPU.

| | |
|---|---|
| **Run** | `sid123123234/act-tpu-probe` v6 reached `COMPLETE` on 2026-09-26. The log places the record at 478.75 s and notebook conversion at 496.35 s. Hardware: v5litepod-8 (`xla_devices=8`), PyTorch 2.8.0+cpu, torch_xla 2.8.0. |
| **Single-device diagnostics** | At 285,336,576 parameters: bf16 throughput **1,110.1 tokens/s**, rebuild **390.0**; fp32 throughput **1,121.9**, rebuild **372.0**. |
| **Decisive aggregate** | Eight ranks completed, one on each `xla:0`–`xla:7`, with each thread bound to its chip. Steady-window aggregate: **15,034.2 tokens/s**, **2.96×** the 5,083.4 T4-pair reference. Wall throughput including startup was 8,505.8 tokens/s; it is not the decision input. |
| **Applied rule** | `PORT`: the aggregate clears **10,166.8 tokens/s**; per-core equivalent **1,879.3 tokens/s**; recompile overhead **18.07 s**, or **2.07%** of projected per-core training time, below the registered 10% limit. |
| **Scope and caveat** | This is not a training result or Gate-4 pass, and it does not change the registered queues. The record says rank input generators were seeded but model initialization uses a shared process RNG; therefore those initializations are not reproducible from their per-rank seeds. |
| **Verification** | `scripts/tpu_probe.py --evaluate artifacts/tpu-probe/output/tpu-probe-v6.json` returns `PORT`. `tests/test_tpu_probe_evidence.py` recomputes that decision and checks the eight recorded device/thread bindings. |

---

## Gate-4 launch snapshot (2026-09-26)

This was the verified launch snapshot; the live update above supersedes only the status fields.


The user explicitly authorized resuming the registered campaign with the existing
configuration. Kaggle access was restored for API requests by passing the refreshed
`kaggle auth print-access-token` as `KAGGLE_API_TOKEN` (without printing or saving it).
Live checks at launch confirmed the prior primary queue at 3/12 (15024-15026), its kernel
`COMPLETE`, and quota 19.48/30 GPU-hours remaining until 2026-10-03. The owner's kernel
list showed no control/null kernels. The sequential campaign launcher then pushed exactly
one next primary session, `sid123123234/act-t4-session` version 9, which Kaggle initially
reported as `QUEUED`; the campaign driver held the queue lock and supervised it. The live
update above records the later transition to `RUNNING` and the remaining quota. Do not
launch a duplicate driver or kernel.

The registration and three-queue sequence remain unchanged. At the launch snapshot, the
published primary count was 3/12; kernel v9 was queued, not a completed seed or verdict.
The real-text development-fidelity studies (§§5.40 and 5.66) remain context, not a reason to
amend the declared-scale registration or infer a Gate-4 verdict.

| | |
|---|---|
| **Registered work to resume** | Preserve seeds 15024-15035, the declared 285M configuration, 6,000 steps / 24.576M tokens per arm, and the session → control → null → verdict sequence exactly as registered. |
| **Launch state** | Session kernel version 9 was `QUEUED`; the detached campaign driver holds its lock and supervises the existing job. Verified quota at launch was 19.48/30 GPU-hours, refreshing 2026-10-03. The live update above records the later `RUNNING` state. |
| **Decision state** | Resumption is authorized and one session has been pushed. The campaign is incomplete and has no Gate-4 verdict; the queued launch state was not completed training or a changed seed count. |
| **Evidence boundary** | The three primary seeds used synthetic tokens and finished near the uniform-vocabulary loss floor. The two real-TinyStories development-fidelity replications favored null on 12/12 seeds each, but do not establish declared-scale behavior and do not alter the preregistered queues. |

For operations, use `docs/T4_RUNBOOK.md` and let the existing `scripts/run_gate4_campaign.sh`
process supervise the active session; do not launch a second driver.

---

## Historical stop snapshot (2026-09-26)

The earlier stop disposition and the records it preserved are described below. It was
subsequently superseded by the explicit authorization to resume; the measurements and
artifacts remain valid historical evidence.

| | |
|---|---|
| **Evidence already in hand** | Two separate fresh 12-seed real-TinyStories blocks at development fidelity favored null over adaptive on **12/12** seeds: §5.40 (mean `null - adaptive` `-0.135543` nats; one-sided exact sign-test p `1/4096`) and §5.66 (mean `-0.126157` nats; p `1/4096`, under the `spread` policy). These are replications, not a pooled estimate. At this fidelity, the adaptive expert-capacity changes have not demonstrated value over doing nothing. |
| **Why Gate 4's remaining spend did not answer that question** | The registered primary queue was **3/12** complete (15024-15026); its three pairs trained on synthetic tokens and ended at the uniform-vocabulary loss floor. The registered pool-control and null queues had not been launched. No registered verdict existed; Gate 4 was neither passed nor failed. |
| **Historical disposition** | The stop was not an amendment to the registration or a Gate-4 verdict. The later resumption authorization keeps the same preregistered work and thresholds. |


## The live capture restarted a stream the CLI had given up on, on the first session that ran with nobody watching (2026-09-26)

The campaign restarted after the weekly quota refresh and pushed `act-t4-session` kernel
version 7. Its run exercised the log-restart path: the CLI's live stream gave up, the
supervisor reopened it, and the original capture's resume count included CLI connection
notices. That defect skipped kernel lines in the original capture and was fixed in
`scripts/kaggle_queue.py` by separating stdout from stderr and excluding notices from the
resume count. The recovered live capture was useful forensic evidence, but its final
benchmark and duration disagree with the platform's completed-run log, so it is **not** a
faithful record of version 7. The platform log and the child's tee were fetched after
completion; their verified outcome and limitations are recorded below and in
`artifacts/v0-session-logs/NOTES.md`.

**Closeout after the kernel finished.** Kaggle kernel version 7 is now `COMPLETE`. Its
platform log and child tee are saved in `artifacts/v0-session-logs/`; the compact
benchmark, queue log, publish receipt and three seed summaries are saved beside them.
The platform log is canonical: it says the run finished after 5.52 h, all three pairs
completed, and the dataset publish succeeded. The separately recovered live capture is
not a complete or exact copy: it contains a different benchmark schedule and reports a
4.99 h end marker, so it remains diagnostic evidence about the tailer, not the run. The
primary queue now has **3/12** seeds. At inspection, Kaggle showed **19.48/30 GPU-h**
remaining until 2026-10-03; no campaign driver was running. Kaggle's account listings
showed no control/null kernels or corresponding `act-v0-5-2` datasets in the
authenticated owner account; those queues have not been launched there and must not be
treated as complete. No further session was pushed.

| | |
|---|---|
| **What the driver did first** | `run_gate4_campaign.sh` started 2026-09-25T19:06:46Z; the health gate read the *previous* session as `ERROR` with **1.07 GPU-h** of the week left, and its own line says what it did about that: *"the week refreshes at 2026-09-26T00:00:00 (5.0 h away); waiting rather than billing a partial session"*. When the week turned, it republished the code dataset (at `c90cf749892e`), pushed `act-t4-session` **version 7**, and polled `QUEUED` -> `RUNNING`. That is the quota gate (`quota_shortfall`) doing in production what its tests pin. |
| **The give-up** | The session's own lines reached the capture normally (its header: `dataset=sid123123234/act-v0-5-2 hours=7 seeds=15024-15035`, `torch 2.10.0+cu128`, two `Tesla T4`, the free-disk report; then `seed 15024: starting pair (remaining 6.93 h fits pair 1.61 h + 45 min reserve (rule 1))` and `arms in parallel (random -> cuda:0, adaptive -> cuda:1)`). The next four lines are the CLI's: `Log stream connection lost, reconnecting...` three times (its source prints that from the second failure on), then `Log stream connection failed 5 times with no new data; giving up.` - written to the capture at **00:32:17Z**. |
| **The restart, and how it is datable from outside** | The CLI's `kernels_logs_cli` (read in the installed 2.2.4, `kaggle_api_extended.py`: `_LOG_STREAM_MAX_FAILURES = 5`, 1 s between retries) **returns** on that message, so the attempt ends with it. The supervisor then waits `LIVE_LOG_RESTART_SECONDS` (**15**) and opens another. The attempt running at 01:29:15Z is therefore a restart, and its predecessor ended 01:29:00Z - exactly 15 s earlier - so the loop was re-armed at ~00:32:32Z, not left dark. The interval between 00:32:17Z and 01:29:15Z is a **live-but-silent** attempt, which is what this session had to say: a pair takes 1.61 h. |
| **Why neither a give-up nor a restart can lose or duplicate a line** | Every attempt streams the log **from its first line** (measured 2026-09-21: 195 identical lines twice), and the supervisor appends only lines past the count already on disk - so what a dark window costs is latency, not lines, and lines produced while nothing was connected are read by the next attempt. When a replay does not match the captured line the supervisor writes `--- the replayed log did not match what was captured; lines may repeat from here ---`. That marker is a **symptom**, and this capture carries one at line 41 (written 02:06:46Z) - which is how the defect below was found, and why the sentence this row originally ended with ("no such marker, so its lines line up") is wrong. |
| **The marker fired, and it was not a repeat but a skip** | The count a restart resumes from has to count the kernel's log lines and nothing else. The tailer merged the CLI's stderr into stdout, so the notices about the connection (`Log stream connection lost, reconnecting...`, the give-up line) were written **and counted**: the original capture's 40 content lines were 33 kernel-log lines and **7 notices**, and the next replay was told to skip its first 40 lines - of which only 33 were kernel lines. Kernel lines 34-40 therefore fell inside the skipped prefix of **every** later attempt and could not be captured by *that* file, because `written` does not decrease. The mismatch at the boundary fired at 02:06:46Z (the capture's own mtime), near when the first pair was expected to finish. The marker's plain reading, "lines may repeat", under-states it: the arithmetic means lines **will be skipped**. The deficit **compounds**, because notices arriving after the kernel's current end are counted too, leaving the boundary at or past the end of the log - then the next replay skips the kernel's next lines as they arrive. Fixed in `scripts/kaggle_queue.py`: the streams are kept separate (`stdout=PIPE, stderr=PIPE`), a second thread writes notices **marked** with `--- kaggle cli: ` and **uncounted** (`_drain_notices`), both threads share the file lock so halves of a line cannot interleave, and each attempt joins the drain thread in its `finally`. Tests pin both the notice-burst/restart behavior and the `Popen` call itself. The original live capture used the old merged-stream behavior and remains forensic evidence; the final platform log and child tee, not that capture, are the canonical record of this session. |
| **The hole, observed an hour later** | At 02:53Z a **fresh** tailer with the fixed code replayed the available log and wrote 36 lines. At the time this was read as the missing pair completion plus the next pair start. The end-of-run comparison supersedes that reading: `/tmp/act-queue-session-4p3coav5/live-session-recovered.log` is not a complete record of kernel version 7. It contains a benchmark at **4,521 tokens/s** and a **1.61 h** pair schedule, whereas the platform's final log and saved `v0-benchmark.json` agree on **4,296 tokens/s** and **1.689 h**; its end marker says **4.99 h** versus the platform's **5.52 h**. It also contains the predicted completion text, but that resemblance is not a substitute for identity with the finished kernel's log. The recovered file is retained only as forensic evidence of what the tailer captured; the official `kaggle kernels logs` artifact and the child's `session-stream.log` are the run record. |
| **What the capture is not** | A line count is not a progress measure; silence in a live capture cannot distinguish a running pair from a stalled stream. The 02:53Z recovered file did show pair-completion text after a quiet interval, but its benchmark and final duration do not match kernel version 7, so that timing is not evidence about this run. The authoritative completion signal is the platform kernel status together with the session log, publish receipt and published dataset; those now show `COMPLETE`, three pairs published, and 3/12 queue seeds done. |
| **Why health could not show seeds yet** | Before the first exit publish, `kaggle datasets files sid123123234/act-v0-5-2` returned 403 and the slug was absent from `datasets list --mine`, which is the no-version state. After the kernel completed, the dataset listing returned seed versions 15024-15026 with complete manifests, reports and checkpoints; `campaign_health --queue session` now reads **3/12**. |
| **The check that completes it** | The final platform log was fetched after `COMPLETE` and saved byte-for-byte as `artifacts/v0-session-logs/session-15024-26-log.json` (69 events, 10,980 bytes, SHA-256 recorded in `artifacts/v0-session-logs/NOTES.md`). The child's tee is `session-15024-26-child-stream.log`; its bytes are an exact contiguous substring of the platform log's concatenated stdout and it includes all three pair completions, the queue stop and successful publish. The recovered live capture did **not** pass a line-for-line comparison: its benchmark and duration disagree with the final log, so it is explicitly not promoted as the canonical capture. The published summaries and exit-publish receipt independently agree with the official log. The queue health check reads the kernel state and published dataset; a separate process check confirmed no campaign driver remained. |
| **Pinned** | `tests/test_kaggle_queue.py` (**87** at the time of this run; two new tests for the notice fix) pins the behavior: restart after a dropped stream, replay-misalignment diagnosis, a tailer that cannot block the driver, and separate CLI notices that cannot advance the resume count. The current collected count may differ as tests evolve. |
| **Result** | The finished session's official record verifies the tailer fix's importance but does not validate the recovered capture: the capture's run details differ from the platform log. Kernel v7 completed cleanly in **5.52 h**, trained and published seeds **15024-15026**, and stopped before 15027 because 1.58 h did not fit the 1.689 h pair plus reserve. Dataset `sid123123234/act-v0-5-2` lists all three complete seed versions; the primary queue is **3/12**. At inspection, weekly GPU quota was **19.48 h remaining of 30.00**, refreshing 2026-10-03. No campaign process remained, and no next session was pushed. The authenticated owner listings contained no control/null kernel or dataset; those queues have **not been launched in this account** and remain required. The platform log and compact artifacts are preserved under `artifacts/v0-session-logs/`. |


Kernel **v5** of `act-tpu-probe` finished, its record was fetched and the pre-registered rule
was applied to it where it landed: **INCONCLUSIVE** (exit 3). The child process the previous
entry added to take the aggregate *ran* - so that fix did what it was for - and the failure
moved one layer down into libtpu's topology. The previous three attempts at this one measurement failed for three different reasons.
Kernel v6 later completed the aggregate without child processes, resolving the outstanding
hardware probe. The historical failures and their fixes are documented below.

| | |
|---|---|
| **The run** | `act-tpu-probe` v5, fetched 2026-09-26 00:26Z by the watcher, which kept the record and the kernel log and printed the verdict. It ran **452 s** of wall clock on the v3-8 (437 s to the record, then the kernel's own notebook conversion) and the platform reported it `COMPLETE` - the five-phase shape, four phases measured, the fifth contained as an error. The committed record is `artifacts/tpu-probe/output/tpu-probe-v5.json`, beside v4's. |
| **What v5 measured** | The same four phases as v4, on `xla:0` with `xla_devices: 8`, torch 2.8.0+cpu / torch_xla 2.8.0, 285,336,576 parameters either dtype: bf16 throughput **1,133.0 tokens/s** (first step including compile **65.10 s**, steady mean 3.62 s, min 0.99 s), bf16 rebuild **394.7** (first 28.77 s, steady 10.38 s), fp32 throughput **1,068.7** (first 71.89 s, steady 3.83 s, min 1.02 s), fp32 rebuild **358.2** (first 30.07 s, steady 11.43 s). |
| **What it did not** | `independent_seeds`: `RuntimeError: Bad StatusOr access: UNKNOWN: TPU initialization failed: Invalid --2a886c8_slice_builder_worker_addresses specified. Expected 8 worker addresses, got 1.` - raised inside the spawned ranks and carried out by the child-process failure path, with its stderr attached, so the record names the layer it died in. The rule needs `independent_seeds.aggregate_tokens_per_second` and `.workers`, so two records in a row now carry four phases and no decision. |
| **Why that is progress rather than a repeat** | The child process had touched nothing, so `xmp.spawn`'s precondition was met and the aggregate's own code ran for the first time. What failed is libtpu's *topology* check inside each spawned rank: the parent's single-process client initializes with `tpu_process_addresses="local"` (the kernel log's E0000 line at 63.5 s), while `tpu.configure_topology`'s `os.environ.setdefault` calls handed the ranks a process grid that wanted eight worker addresses with one set. The two inputs to that arithmetic - the platform's topology environment, and the PCI chip count `num_local_processes()` takes its minimum against - were in no record any probe had written, which is why v5 had to be diagnosed from the log. |
| **The inputs are now in the record** | `TOPOLOGY_ENV_KEYS`: 23 platform variables (`TPU_PROCESS_BOUNDS`, `TPU_PROCESS_ADDRESSES`, `TPU_VISIBLE_CHIPS`, `TPU_WORKER_HOSTNAMES`, the `PJRT_*` set, `LIBTPU_INIT_ARGS`, ...), recorded with `None` for unset so a reader can tell a missing value from a missing key; and `_tpu_report`, the PCI chip count plus the TPU metadata block, each failure recorded under its own exception name rather than raised. Both ride in the record's `device` block, so the next attempt (if there is one) is diagnosable from the artifact. |
| **The fix, and the reading behind it** | The aggregate rides the **single-process path**: one thread per device, in the process that is already running, each rank taking an explicit `xla:<rank>`, with a `threading.Barrier` that releases only once every rank has compiled. The number is the tokens the ranks pushed inside the window in which all of them were stepping - *not* a sum of per-rank rates, because a step marker may sync more than the rank that called it and a sum would then count the same device time once per rank. The record carries each rank's device, seed, window stamps and compile cost, so "one independent seed per chip" is a value the parent **checks** (all ranks present, one distinct device each, one shape, a window that holds a steady step) instead of a behaviour it trusts a spawn to have had. `xmp.spawn`, `--seeds-only`, `seeds_in_child`, `seeds_only_request` and `spawn_arguments` are deleted with it: the thread-per-device shape needs no untouched runtime, which is the only reason the child process existed. |
| **The binding a one-process aggregate needs, found while writing it** | A step marker is not a device argument: in torch_xla 2.8 `xm.mark_step()` is `torch_xla.sync()`, which calls `_XLAC._xla_step_marker(_xla_get_default_device(), [], ...)` and so acts on the **calling thread's** device. That value is a C++ `thread_local` (`g_current_device` in `aten_xla_bridge.cpp`) whose fallback, for a thread that has never set it, is the process default every earlier phase of this probe leaves at `xla:0`. The first version of the thread-per-device aggregate therefore had all eight ranks marking **chip 0's** steps: `GetLiveTensors(device)` collects only the named device's tensors, so seven ranks' graphs would never have executed and the "steady step" would have been graph construction - a fast, plausible, wrong aggregate, which is worse than the three crashes before it because nothing downstream could tell. Each rank now binds its own chip before it builds (`bind_thread_device`, i.e. `torch_xla.device(rank)`) - the same call torch_xla's own one-process launcher makes at the top of each thread (`_run_thread_per_device`) - and its report carries `thread_device_before`/`thread_device_after`, which the parent checks against the device the rank was given. So "one seed per chip" is a value the record states rather than a behaviour the runtime is trusted to have had. |
| **A dead rank refuses the aggregate in seconds, not in an hour** | The barrier's budget is the phase's clock, so the failure path the four attempts paid for had a second cost on top of the loss: a rank that died before compiling left the others at the barrier for up to an hour of **billed** TPU time, and what then reached the record was the barrier break rather than the error that caused it. The parent now awaits the ranks with `FIRST_EXCEPTION`, calls `Barrier.abort()` the moment one raises, and raises a summary that keeps cause and collateral apart - `rank 3 raised RuntimeError: RESOURCE_EXHAUSTED: ...; rank(s) [0, 1, 2, 4, 5, 6, 7] stopped at the barrier waiting for it` - which is what the barrier wait has its own exception type (`RankWaitTimeout`) for. Pinned by a test that shortens the budget to 60 s and asserts the refusal returns in under 10. |
| **Why not a fourth process attempt** | Its success depends on values this project cannot see from a record (the platform's environment) and on one it cannot even query without initializing the runtime the spawn needs untouched (the pre-spawn device count). The single-process path was subsequently measured on this hardware in v6, completing the aggregate on eight bound rank threads and establishing the registered `PORT` decision; the versioned record is documented above. |
| **What the two records say together** | The dtype ordering **flips**: v4 measured fp32 2% faster than bf16, v5 measures bf16 6% faster than fp32 - and v4 does not agree with itself, since its rebuild block puts bf16 ahead (379.4 against 370.2) while its throughput block puts fp32 ahead. So the previous entry's reading, "fp32 came out *faster* than bf16 by 2%, so the bf16 protocol amendment that the rule assumes buys nothing on this hardware", is **superseded** as it stands: what two runs on the same shape support is that the dtypes sit within a few percent of each other, not which one wins. Both records' per-core bf16 rates stay below the **1,270.8 tokens/s** eight cores would need to clear the rule's 2.0x bar - 1.73x at v4's rate, 1.78x at v5's. |
| **A caveat the records carry and the rule half-reads** | In both records a rebuild's *fastest* step is the same ~1.0 s as the throughput pass's while its mean is 10x that, so `steady_step_seconds_mean` over a four-step rebuild loop is not a steady step - and the rule's recompile-overhead arithmetic subtracts that mean from the first step, which under-states the compile cost, the *un*conservative direction for that condition. **Re-derived now, so the caution is a number rather than a worry:** at each record's own single-core rate a queue's per-core training time is **18.7 h** (v4) / **18.1 h** (v5), and its 45 rebuilds per core at the compile cost the rule computes (18.4 s either record) read **1.24%** and **1.27%** against the 10% limit - **1.89%** and **1.92%** under the conservative reading, which takes the fastest step instead of the mean. So the caveat moves the condition by ~1.5x and leaves it ~5x under its bar, and the condition would only bind at a rebuild compile of **~146-149 s** - ~14x the rebuild's own mean step, ~146-149x its fastest. It has also never been *reached* on a record, because both are INCONCLUSIVE before it. Nor is this a rule edit: the derivation reads the condition, and `tests/test_tpu_probe_evidence.py` recomputes all three numbers from each record, so the next record is measured against the derived bar rather than this one's rate. |
| **A guard that fired on prose, found by running the four checks** | `test_every_artifact_a_test_reads_is_not_gitignored` failed on `tests/test_train_code_lm.py` naming `artifacts/code-lm/checkpoints/match-ce.pt` in its *docstring* - not reading it - the moment the running campaign wrote that 2 GB checkpoint. A comment never reaches the AST and a docstring does, so `_artifact_literals_in` now exempts docstrings explicitly (a module-level `RECORD = "artifacts/..."` assignment is still a read and still checked), with both directions pinned. It is the third shape of "the search found itself" in this repository. |
| **And the push had to wait, so the wait is an instrument** | `push_tpu_probe.py` refused v6 with `sid123123234/act-t4-session` `RUNNING` (exit 5) - correctly: a session's body and its mounted `src/` have to be one revision, and the dataset is republished in place. A gap in the campaign is a platform event, not something to watch for by hand, so `scripts/push_tpu_probe_when_idle.sh` polls the **guard's own** `live_campaign_kernel()` (imported, not re-implemented - two copies of that answer is how the 2026-09-21 session was lost), publishes nothing itself, and on a gap runs the push and arms `fetch_tpu_probe_when_done.sh`. Bounded at 144 polls of 300 s; started 2026-09-26 00:52Z and **restarted 02:03Z** (poll 1/144 again, appended to the same log) after its retry line was changed to name the push's exit code - 5 is the guard seeing a session come back, anything else is the platform, and the old wording read as "the campaign resumed" either way. The restart is not tidiness: bash reads a script as it runs, so editing a live one's bytes under it is how a waiter turns into undefined behaviour. That guarded push produced completed kernel v6; its versioned record and `PORT` decision are preserved above and pinned in `tests/test_tpu_probe_evidence.py`. |
| **Pinned** | `tests/test_tpu_decision.py` (**50** collected, was 37): the window arithmetic, one barrier shared by all eight ranks, `taken_in`, the distinct-device / one-shape / missing-rank / empty-window refusals, the barrier's named timeout, the worker's report run for real on CPU at a reduced-fidelity shape, the environment and TPU reports including their failure branches, the new waiter's two load-bearing properties, and the rank-thread binding (both sides recorded, an ordinal-less device refused, a rank whose thread sits on another chip refused by name, the default rank path refusing without the module that binds it, a dead rank refusing in seconds with its own error named ahead of the ranks that only stopped waiting, and the whole aggregate run for real - eight rank threads, one barrier, one window - on a CPU host with torch_xla stood in for). `tests/test_tpu_probe_evidence.py` (**18**): v4/v5's four phases and failures, their `INCONCLUSIVE` decisions and historical forecasts, plus v6's five completed phases, eight bound device ranks and registered `PORT` decision. `tests/test_report_artifact_links.py` (**+1**, 6): a docstring mention is not a read. |
| **Result** | At the 2026-09-26 19:10Z snapshot, v6 was `COMPLETE` and Gate-4 kernel v9 was `RUNNING` with 14.80 GPU-hours remaining. The three-seed v7 session and the v6 probe record/log are preserved; see the updates above for v6's `PORT` result. |

## The real-text leg was unrunnable and unaudited; it is now both, and a 20.9M-parameter LM is proved to work (2026-09-25)

| | |
|---|---|
| **What state it was in** | `artifacts/real-lm/report.json` has been committed since **2026-09-14** — a 20,863,872-parameter MoE (10,247,040 active) trained on real TinyStories, val loss 1.7134 / perplexity 5.55, four generations — and **nothing recomputed a number in it**: no test read it, `verify_record.py` carried no claims on it, and the README did not mention it. The one leg that shows a real language model works was the one leg with no audit, and the ask was to train one and prove it. Trying to train it found the rest. |
| **Defect 1 - an undeclared dependency** | `scripts/train_real_lm.py` and `scripts/real_lm_infer.py` both `import tokenizers`, and it was **not in `pyproject.toml` and had 0 occurrences in `uv.lock`** — so the tokenizer stage and the whole inference path raised `ModuleNotFoundError` in a clean environment. Declared and installed (**tokenizers 0.23.2**, `uv add`). The repo's own rule is that a documented command must run; this one could not. |
| **Defect 2 - resume silently misaligned the optimizer** | `Optimizer.state_dict()` is **position-keyed** (`load_state_dict` maps its entries onto `param_groups[i]["params"]` *by index*) while `nn.Module.state_dict()` is **name-keyed**. A model whose parameter **order** changed while its names and shapes stayed the same therefore loads its weights strictly and attaches its Adam moments to the wrong tensors. That is what had happened between the two sessions, and the first step died inside `torch._foreach_lerp_` with `RuntimeError: output with shape [384] doesn't match the broadcast shape [1152, 384]` — a message naming no parameter, no cause and no remedy. **Measured on the real checkpoint: 36 of 128 parameters misaligned, a one-position shift.** The unrecoverable form of this is worse than the crash: had neighbouring shapes agreed, it would have trained on another parameter's momentum and said nothing. |
| **The fix, and why it is the shape it is** | Moments are stored and restored **by parameter name** (`checkpoint_payload` / `restore_optimizer`), each moment is checked against the parameter it claims to belong to, and a **positional payload is refused** with both remedies named rather than guessed at. The pre-fix checkpoint cannot be re-keyed (the names were never written down), so `--reset-optimizer` is the named remedy: resume the weights, start Adam fresh, and say so in the log — which is what this run did. |
| **The training run** | 6000 -> **12,000 steps**, 49.2M -> **98,304,000 tokens** of real TinyStories, **~17,400 tokens/s** on the local GTX 1070 (fp32, `sm_61`), ~46 min of wall clock, **zero Kaggle quota**. Held-out loss **1.7134 -> 1.6228**, perplexity **5.55 -> 5.07**. The checkpoint on disk is now in the name-keyed format: `optimizer: {groups, moments}`. |
| **And the schedule nuance, left in the curve rather than smoothed out** | `lr_at` is a function of the **total** step budget, so extending the budget restarted the cosine decay instead of finishing it: the run resumed at **lr 8.0e-05** and stepped back up to **4.02e-04**, which is why the held-out loss *rises* from 1.7144 at step 6100 to **1.7908 at step 6500** before falling at every evaluation after. A monotone-looking curve would have been the dishonest reading, so the check reads the shape and is calibrated on it: the restart is **0.076 nats** against a second rise of **0.001 nats**, which is the evaluation noise of a 7,190-token holdout. |
| **The proof, and the yardstick it needed** | A perplexity with nothing beside it proves nothing, so the record now carries reference models scored on the same held-out stream: uniform **9.0109** (= ln 8192), unigram **5.9681**, bigram **7.2348** (Laplace 1; both fit on the held-out tokens themselves). The model beats the **bigram by 5.61 nats** having never seen those tokens, and explains **82%** of the floor's entropy. `artifacts/real-lm/val-tokens.npy` (**14 KB**) is now committed so every one of those is recomputable by a reader — the corpus (1.9 GB) and the checkpoint (250 MB) stay local. |
| **Defect 3, found in the artifact itself** | The recorded samples were **raw byte-level output** — `ĠOnce Ġupon Ġa Ġtime` — while `real_lm_infer.py` had attached a byte-level decoder at load time all along, so the record's qualitative half was unreadable and the two scripts disagreed about what a sample *is*. The writer now attaches the decoder (`tokenizer_from_disk`). Re-rendering with the documented command reproduced **every non-sample field identically** — same loss, same baselines, same history — so fixing the samples moved no number, which is what makes the re-render a fix rather than a new result. |
| **Pinned** | `tests/test_train_real_lm_resume.py` (**7**) pins the checkpoint contract, including the shift itself (a neighbour's moment attached to the wrong name must be refused *by name*) and the live checkpoint when this machine has it; `tests/test_real_lm_proof.py` (**9**) recomputes the arithmetic, the curve and every baseline from the committed record and the committed held-out stream; `verify_record.py` gained **15 claims** (`check_real_lm`) read against the README's new section, and `ARCH_SEARCH_FINDINGS.md`'s stated coverage moved **314 -> 328** full / **294 -> 308** quick, which the suite checks. |
| **A note the earlier entries connect to** | Two of the new tests name an artifact only this machine has (the 250 MB checkpoint) and **guard** for its absence, so the artifact-read guard from this session had to learn the distinction it was missing: a path named in a `pytest.skip` is an exemption, the same path read without one is not. Without that, a correctly-guarded local test and the defect that guard exists for would look identical. |
| **What this does not claim** | The checkpoint is not committed, so a reader can recompute every number and read the generations but cannot re-run inference without running the training command first. The held-out set is **7,190 tokens**, so its evaluation noise is ~0.001 nats and small differences between near models are not resolvable here. And it is a proof of the **path** — 20M parameters on a children's corpus trains and generates — not of the project's hypothesis, which is the architecture question the phase above owns. |
| **Result** | **1690 tests** pass (1674 + 16); **328/328** claims agree with the artifacts; the real-text leg has a README entry, an audit and a reproducible command for the first time. Probe **v4 QUEUED**; TPU **19.79 h**, GPU **1.42 h** (both refreshing **2026-09-26**) — this work spent local GPU time only. |

## The fresh-clone gap is a check now, rather than something somebody has to remember to do (2026-09-25)

| | |
|---|---|
| **What was still missing** | The entry below fixes the 33 failures by hand, once. Nothing stopped the next test from reading local scratch, and nothing in the tree *can*: from inside the working tree a file that is present and a file that is committed are the same file, so all four documented checks can be green here and red for a reader. |
| **The instrument** | `scripts/check_fresh_clone.py`: it clones **HEAD** into a temp directory and runs `AGENTS.md`'s four checks there with the current interpreter (`PYTHONPATH=src`), one `ok`/`FAIL` line each carrying the check's own last line, exit **1** on any failure and **2** when the clone itself cannot be made. |
| **Why a clone and not a filter on the working tree** | A clone is what a reader gets; the tracked/untracked line is git's to draw rather than the script's inference; and no check has to know it is being run this way. The alternative - hiding untracked files under `artifacts/` for a subprocess - is the script reimplementing `git check-ignore`, which is the bug the scanner hit twice. |
| **The property it is honest about** | It checks **HEAD**, so it prints the sha it cloned and warns when the working tree is dirty (`NOTE: the working tree has uncommitted changes. They are NOT part of this check`). That is the intended semantics - HEAD is what everyone else receives - but it would be wrong to read a pass as a statement about uncommitted work, so the warning is not optional. |
| **Pinned on a throwaway repository, in both directions** | `tests/test_check_fresh_clone.py`, **7 tests**. The load-bearing one reproduces the defect: a committed test reading an ignored, untracked `artifacts/local.txt` **passes** in that repository's own tree and **fails** in the clone the check makes - the contrast is the finding, since a failing clone run alone is indistinguishable from a genuinely broken test. Also pinned: it sees HEAD rather than a fixed-but-uncommitted file; a source that is not a checkout is refused; an unknown check name is refused rather than silently narrowing the run; the clone is removed unless `--keep`; and the check list agrees with the document that names the commands as a set. |
| **The new test caught the same class in itself, immediately** | The first version pinned that list against `AGENTS.md` - which is **gitignored on purpose**: the ignore rule groups it with `.omo/`, `.freebuff/` and `.project_id` as this tree's agent-tool state rather than project content, so a fresh clone does not have it. The test would therefore have failed in exactly the checkout it exists to check. It now skips by name when the file is absent, and the check is documented for readers in `README.md`, which *is* tracked. Recorded because it is the third time in this session that a check written to catch machine-local state was itself machine-local - the class is easy to write and the guard is what finds it. |
| **Mutation-checked** | Pointing `run_check` at the source instead of the clone fails exactly **2** - the defect test and the HEAD test, i.e. precisely the two that pin what is being run. |
| **Measured on this tree** | `4/4 checks pass in a fresh clone of 352a1eb`: 314/314 claims, 9/9 records, 10 notebooks byte-identical, and **1633 passed / 34 skipped** - the fresh clone's numbers, which is the point. |
| **Result** | **1674 tests** pass locally (1667 + this script's 7); the fresh-clone check is documented in `README.md` beside the record checks, and in the local `AGENTS.md`; probe **v4 QUEUED**; TPU **19.79 h**, GPU **1.42 h** (both refreshing **2026-09-26**). |

## A fresh clone ran the suite for the first time, and 33 tests were reading evidence only this machine has (2026-09-25)

| | |
|---|---|
| **How it was found** | The entry below commits `artifacts/gate3-384.jsonl` so a *test* can run in a checkout that did not generate it. That raised the obvious follow-up - does the suite actually pass in a fresh clone? - and the answer was no. `git clone . /tmp/act-fresh` + the project's own `pytest -q`: **33 failed, 1633 passed, 1 skipped**. The class is invisible here by construction: the file a test reads is on this disk, so the failure exists only where the evidence does not. |
| **What was failing, in three groups** | **27** parametrisations of `test_toy_summary_labels.py::test_the_committed_summary_regenerates` and **1** `test_width_long_w24_summary.py::test_the_verdict_still_reports_this_branch`, all through `scripts/null_toy_study.py`, which recomputes a record from the run's raw rows under `artifacts/toy-null-study/<profile>/shard*.json`; and **5** in `test_capacity_sweep.py`, which build a `V0SessionConfig` naming `artifacts/real-lm/tokens.npy`. Every one of those paths is inside `artifacts/*`, i.e. local scratch by the blanket rule - the summaries those tests pin are committed and self-contained, the runs behind them are not. |
| **Why the failures read as defects rather than as skips** | Each died inside the code under test - `SystemExit: missing shard: ...` from `null_toy_study`, `ValueError: data stream not found: ...` from `V0SessionConfig`'s own validation - so the *verdict function* and the *sweep's arm mapping* looked broken when only their input was absent. That is the confusing half of the defect, and it is why the fix is an explicit skip naming what is missing rather than leaving the crash to be interpreted. |
| **The fix, in the repository's established form** | Three guards, each naming the absent evidence: `_shards_cover_the_block(profile)` in `test_toy_summary_labels.py` (the helper `_committed` already used for exactly this state), a shard-directory check in `test_width_long_w24_summary.py`, and a module-level `requires_real_corpus = pytest.mark.skipif(not Path(capacity_sweep.DATA_PATH).is_file(), ...)` on the five sweep tests. This follows the precedent already in the tree - `test_architecture_report.py`'s "Skipped when the research artifacts are not present, e.g. a fresh checkout of the source alone" - rather than inventing a convention. |
| **Nothing is weakened where the evidence exists** | Locally the corpus and the shards are present, so **no test skips**: the suite is **1667 passed, 0 skipped**. The guards fire only in a checkout that could not have run the check anyway, and every number the report quotes stays pinned by the summary-reading tests, which need no shards. |
| **Verified in the checkout that failed** | The fresh clone, with the guards, is **1633 passed, 34 skipped, 0 failed** (the 33 plus the one skip the clone already had). Same clone, same venv, no local artifacts copied in. |
| **The guard from the entry below had to be made precise, and its first version failed on its own author** | That check flagged `artifacts/real-lm/tokens.npy` the moment the new `test_capacity_sweep.py` *named* it in a skip reason and a comment - neither of which is a read. It is now AST-based: it parses each test module and looks only at string constants outside a `pytest.skip`/`skipif`/`fail` call or a `reason=` keyword, since a comment never reaches the AST and a skip message is a string like any other. Re-verified directly: a read literal is found, a comment and a skip reason are not. That the class tripped on a mention is worth recording - the repo has hit "the search found itself" twice before, and this is its third shape. |
| **Result** | Local: **1667 passed** (0 skipped), **314/314** claims agree with the artifacts. Fresh clone: **1633 passed, 34 skipped, 0 failed**. Probe **v4 QUEUED**; TPU quota **19.79 h**; GPU **1.42 h** (both refreshing **2026-09-26**). |

## The last fresh-clone defect is closed by committing the record, not by keeping it local (2026-09-25)

| | |
|---|---|
| **What was open** | The previous entry found `artifacts/gate3-384.jsonl` in the same position `null-clause-power.json` had been in: **cited by the report** (section 9.12, ground 2), **read directly by a test** (`tests/test_factor_map_recheck.py`), and gitignored by the blanket `artifacts/*` rule, so both the citation and the test were true only on the machine that generated the file. It was left open as a repository-structure call: commit 10 MB, or keep it local and let that test be understood as machine-local. |
| **The decision, and why** | **Committed.** The repository's own rule is "never cite what has no artifact", and the alternative - rewording the report and redirecting the test to the committed summary - cannot carry ground 2 at all: the summary drops `mutation_history` from every arm, so the probe-field absence is only checkable from the raw sweep. Keeping the record local would have left a published claim unverifiable in a fresh clone to save 10 MB, which is the wrong side of that trade in a repository whose deliverable *is* the record. The cost is stated rather than hidden: the file is **10,118,547 bytes**, so committed artifact bytes go **22.8 MB -> 32.9 MB** and this one record is **30.7%** of them. |
| **What the artifact is** | `artifacts/gate3-384.jsonl`, 3 lines (shards), written **2026-09-12**, `sha256 62875e36...`; **3,456 arm payloads** and **6,912 mutation events** across the depth-1 gate sweep. It is the raw sweep behind the committed summary `artifacts/study-summaries/gate3-legacy-signal-384.json`. |
| **The claim it carries is now recomputed, not quoted** | Ground 2 is prose no longer. `test_the_committed_raw_sweep_records_no_probe_controller` recomputes it from the committed record: the sweep's **event-key set is pinned exactly** (the six pre-probe keys) rather than scanned for a `probe` prefix, because a differently-named field would slip past a prefix test and the point is that this schema predates the probe; and **non-vacuity is asserted first** (over a thousand events), since a sweep with no events would satisfy "no probe fields" trivially - the same pass-for-the-wrong-reason shape the concurrency rehearsal closed. The report's ground 2 now names the test that pins it. |
| **The general defect, guarded rather than re-scanned** | The class was found twice by an ad-hoc scanner that was itself wrong twice. It is now a check with the same reach and no hand-running: `test_every_artifact_a_test_reads_is_not_gitignored` parses every `tests/*.py` for `artifacts/...` literals, keeps the ones that resolve to a file, and asks `git check-ignore` about each - so a test that reads a file only its author's machine has fails in the *author's* suite, naming both the artifact and the test. It is scoped to what tests read: a raw sweep cited in prose as provenance is not a committed-file claim, which is why the other nine gate3 sweeps are correctly left alone. It skips rather than passes when git cannot answer. |
| **Verified against the defect, both directions** | The scan finds **exactly one** offender in the tree - `artifacts/gate3-384.jsonl`, read by `test_factor_map_recheck.py`, the file this entry resolves. Removing the `.gitignore` exception makes the guard fail, and the failure names them: `[('artifacts/gate3-384.jsonl', 'test_factor_map_recheck.py')]`; restoring it passes. The new ground-2 test passes on the committed record over all three shards. |
| **Result** | **1667 tests** pass (1665 + the guard + the ground-2 pin); **314/314** claims still agree with the artifacts; probe **v4 QUEUED**; TPU quota **19.79 h**, GPU **1.42 h** (both refreshing **2026-09-26**, now tomorrow), so the campaign is unchanged and still stopped. |

## The dev-fidelity anchor column is filled, and dev fidelity lands in the branch the section 5.44 amendment added (2026-09-22)

| | |
|---|---|
| **The block** | Section 5.66, registered in `RESEARCH_REPORT.md` before its first session and run here in one detached sitting on the idle local card: 48/48 sessions, 15:00 to 16:32 (**~1.5 h** of wall clock, **zero** Kaggle GPU- or TPU-hours), 12 fresh seeds x 4 sessions at the dev shape on the real TinyStories stream with the deployed treatment on all four. |
| **The gate that had to hold first** | **A4 held: the null arm is the lowest of the three on 12/12 seeds** (bar 10), so section 5.40's ordering replicates on fresh seeds and the anchors are readable. `null - adaptive` reads **-0.1262** against section 5.40's **-0.1355**, and `null - random` **-0.1213** against **-0.1457** - quoted beside the ordering because a fresh-seed replication is not bit-equality. **A5 held**: all three contrasts agree in sign between the `prefix` and `spread` policies. |
| **The anchors, and why they differ in kind** | The do-nothing clause's anchor is taken where its effect is real: **z = +1.489**, firing rate **95.6%** - so that clause's *power* is now anchored. The harm guard's and the pool-matched control's are taken where their effect is **absent** (z = **+0.039** and **+0.128**, i.e. the treatment beats its own churn arm on 5/12 and the loss-blind arm on 5/12 - a coin flip), so their firing rates (2.49%, 4.28%) sit at the **false-alarm level (1.93%)**. Those two anchors calibrate the clauses' quiet cost, **not their power**, and that is recorded because the alternative reading ("the harm guard has been shown insensitive") is not what the block shows. |
| **The verdict's own arithmetic, read off the block** | adaptive beats churn **5/12**, the loss-blind arm beats churn **5/12**, adaptive beats null **0/12**. Through the verdict's own precedence: `harm_guard_clear` **True**, `control_passes_gate` **False**, `indistinguishable` **True**, `null_clause_rejects` **True** - and the label is **`PASS_COST_ONLY_NO_DEMONSTRATED_EDGE`**. So the clause the section 5.44 amendment added is exactly the clause that fires on the only block that has ever carried all three arms, and it fires **without the harm guard firing** (correctly: the guard asks whether the treatment is *worse* than churn, and at dev it is a hair better). |
| **The loss-blind arm, measured rather than assumed** | The churn arm is **bit-identical across all three pair roots on 12/12 seeds** - the same loss to the last digit under three different probe-policy/selection settings, which is what "loss-blind" means behaviourally. It carries a limit for the anchor table itself: the harm-guard and pool-matched rows are paired against the **same** churn trajectories, so they are not independent evidence of one another. |
| **Two errors of my own, both in the pre-registration's own test file** | The A5 block compared `prefix_adaptive - prefix_random` against a stored `adaptive - random` (inverted sign), and the anchor-root assertion required every anchor's root to be `pair_spread` when the pool-matched control's root is `control_spread`. **Neither could fire until the block produced a summary to read**: a pre-registration's tests run for the first time against the outcome they were written for, so they are code that ships unexecuted. Both fixed, and the second now pins the property that actually holds (every anchor is read under the `spread` policy) plus the pool root's own reason. |
| **One direction crossing, made explicit** | The A1 rows report each clause's **comparison arm** winning; `v0_verdict.py`'s section 5.5 clause fires on the **loss-blind arm** winning. The two tables state that clause's direction *oppositely*, so a reader checking one row against the other reads the clause backwards. Both directions are now committed together (`verdict_reading_at_dev_fidelity` carries `loss_blind_wins_vs_churn` beside the row's `comparison_wins`), the summary's own sentence for the pool row says which direction the clause fires in, and six mutants bite (including flipping either direction). |
| **A stale artifact, and the fresh-clone defect it led to** | The suite failed on `test_null_clause_power.py`: the last session's edit to `null_clause_power.py` was never followed by a regeneration of its artifact, and the half-finished wiring also referenced a key the function does not produce (`anchor_z` for `z`). Regenerated, key fixed. Fixing it surfaced a second thing: **`artifacts/null-clause-power.json` was gitignored by the blanket `artifacts/*` rule rather than by any decision**, so its test could only pass on the machine that generated it; it is now excepted and committed. |
| **And one open item, not decided unilaterally** | The same scan found **`artifacts/gate3-384.jsonl`** (3 lines, **10 MB**) in the identical position: cited by the report as the raw record corroborating a published claim, read by `tests/test_factor_map_recheck.py` - and absent from a fresh clone. It is ~45% of the currently committed artifact bytes (22.8 MB), which makes it a repository-structure call rather than a bug fix. **Left as an open item**, costed: commit it, or keep the 10 MB local and let that one test be understood as machine-local. The scanner that found it was itself wrong twice before it was right (it doubled the `artifacts/` prefix, and then reported clean), so the count is only trusted after the bug was fixed and re-run. |
| **The audit caught its own first error** | Section 5.66's numbers are now carried by `verify_record.py` (**314 claims**, up from 283) - and on its first run it failed one: the harm guard's firing rate was written **0.0246** where the `pair_spread` row's own z gives **0.0249**. The 0.0246 belongs to the `pair_prefix` policy-check row; the two anchors are 0.2% apart, so the copy was invisible to reading and only a recomputation finds it. Corrected in the report's table, its prose and the journal entry above. The instrument's own power curve (50% at **z = 0.783**, 80% at **z = 1.123**) is pinned in the study's test file for the same reason. |
| **Result** | **1665 tests** pass; report section 5.66 marked RAN with its outcome; **314/314** claims agree; probe **v4 QUEUED**; TPU quota **19.79 h**; GPU **1.42 h** (refreshing 2026-09-26). |

## The probe's third run measured everything except the one aggregate the decision reads, and the two defects behind that are fixed (2026-09-22)

| | |
|---|---|
| **The run** | `act-tpu-probe` v3 was pushed 2026-09-22 08:41Z and started in the afternoon after ~6.4 h in queue. It ran **405 s** and ended **COMPLETE** - a *record*, not a log, carrying all four dtype phases. |
| **What the v3 fixes bought, all three confirmed live** | The measurement order put **bf16 first** (the dtype the rule reads), the fp32 diagnostic last - and every phase's own `_save` landed: the record holds `bf16.throughput`, `bf16.rebuild`, `fp32.throughput`, `fp32.rebuild`. Containment worked exactly as designed: the phase that failed is a row in `errors` and the phases after it still ran. |
| **The numbers it produced** | **8 XLA devices** (`device_kind: xla:0`, torch 2.8.0+cpu, torch_xla 2.8.0), declared shape **285,336,576** parameters. **bf16 throughput 1,133.6 tokens/s on one core** (first step 65.84 s including compile, steady 3.6134 s per 4,096-token step); **bf16 rebuild 387.6 tok/s** (steady 10.5687 s, first step 28.57 s); **fp32 throughput 1,099.2**, fp32 rebuild 368.0. |
| **The fp32 anchor replicated** | The earlier measurement was **1,101.7** tok/s. This run's **1,099.2** is 0.2% away on a different day and a different kernel, so the per-core rate is a stable property of this shape, not a one-off. |
| **What it lost, and why** | `independent_seeds` raised before spawning anything: `ValueError: Unsupported nprocs (8). Please use nprocs=1 or None (default)`. torch_xla 2.8 accepts only ``1`` or ``None`` and takes ``None`` as "one process per available device". The rule divides the aggregate by the worker count, so a missing aggregate is not a smaller measurement - it is the one number the port headline is read from, and `evaluate` on the real record says **INCONCLUSIVE** (missing `independent_seeds.aggregate_tokens_per_second`, `.workers`), exit **3**, verified live. |
| **A second defect, in the same phase** | The aggregate was defined as total tokens over the **spawn's wall clock**, which contains eight process startups and eight XLA compilations - one-off costs a 24.6M-token arm pays once, and costs the rule already models separately from the rebuild measurement. It was therefore not comparable to the T4 pair's *steady* 5,083.4 tok/s it is divided into, nor to the per-core training hours derived from it. Fixed before the re-measurement rather than after: the aggregate is now the **sum of the ranks' steady rates**, with the wall-based figure recorded beside it so the difference is visible rather than assumed. |
| **What the measured numbers already bound, stated as a bound** | The throughput bar is `2 x 5,083.4 = 10,166.8` aggregate, i.e. 1,270.9 per core. bf16 measured **1,133.6**, so eight cores at the measured per-core rate sum to **9,068.8 = 1.784x** the T4 pair - **below the bar before any concurrency cost is measured**, since the aggregate is a sum of per-core rates and contention can only lower them. That is not a verdict (the rule reads the measured aggregate, which v4 will produce), and it is recorded here as a bound for the same reason the fp32 anchor was: so the reading is not discovered after the number lands. |
| **And the rebuild condition, priced** | bf16's rebuild compiles for ~18 s more than a steady step (28.57 - 10.57), against a per-core training time of ~9.0 h for a queue's share of tokens - so the recompile-overhead condition the rule applies at 10% is nowhere near binding. The blocker, if any, is throughput alone. |
| **Fixed, pinned, re-queued** | `spawn_arguments` (``nprocs=None``, as data) and `measure_independent_seeds` (device-count check, rank-completeness check, steady-sum aggregate) in `scripts/tpu_probe.py`; four mutants, four bites (nprocs back to 8, the wall-based aggregate, either check dropped). Probe **v4 pushed** at `1664194` - the fix commit, republished into the code dataset (`act-lab-code` at `1664194`), so the queued kernel mounts the fixed tree - and the watcher is alive to fetch the record and apply the rule where it lands. |
| **Result** | **1653 tests** pass (the section 5.66 block's own tests land with its own record); probe **v4 QUEUED**; TPU quota **19.8 h**; GPU **1.42 h** (refreshing 2026-09-26), so the campaign stays stopped. |

## The null queue's amendment is unblocked: two seeds trained concurrently are bit-identical to two trained in turn (2026-09-22)

| | |
|---|---|
| **What was missing** | The null-queue entry prices two options for its idle second T4, and the first was blocked on "a rehearsal proving two concurrently-running seeds are bit-identical to two sequential ones". That rehearsal was the whole blocker on the option priced at **8.49 GPU-h** - and, because the two options compose, on reaching a **one-week block at the rule's own 2.0x bar** instead of at 2.417. |
| **What was run** | `scripts/two_seed_concurrency_rehearsal.py`, seeds **15024/15025**, the **null arm** (the queue being amended), dev shape, **40 steps, batch 8**, **4 threads pinned per process**, device cpu: each seed's session trained in its **own process**, once one after the other and once at the same time, compared on all **11** fields that are the result. **CPU and free** - no GPU hour, no TPU hour. |
| **Result** | **Bit-identical on every field, both seeds**: validation loss `9.061052083969116` / `9.054726600646973`, architecture hashes `b9fc55adb199` / `a779f7f4f020`, training FLOPs `4,891,921,612,800`, and mutation count, parameter counts, search FLOPs and mutation history equal to the last digit. The scenarios' own wall times differ - **108.39 s sequential against 88.22 s concurrent**, so the pair really did overlap and contend - which is the difference concurrency is *allowed* to make and why the three arm-level timing/placement fields are excluded rather than compared. The rehearsal is also **deterministic across runs**: the same two losses came back from a second, independent execution. |
| **Why the comparison is not vacuous** | Four ways this check could have passed for the wrong reason, all closed and mutation-checked: a result **field compared by nothing** (the compared set is asserted to be exactly what `V0ArmResult` carries minus three *named* metadata fields, so silently dropping one fails, and moving a metadata field into the compared set fails too); a comparison that always **agrees** (parametrized over every field, each perturbation must be reported by name); and a **record whose flag was not computed from its payloads** (the committed record's verdict is recomputed from its own two payloads here, so an edited `identical` cannot survive). |
| **A defect in my own first draft, caught before it ran** | Both scenarios wrote to the **same paths**, which would have made the check vacuous twice over: the comparison would have re-read the first scenario's files, and `run_v0_session` **resumes** a seed whose checkpoint root already exists - so the second scenario would not have been a run at all. That is precisely the hazard `capacity_sweep.py` was found with on 2026-09-19 (a reset naming nothing while the session resumed). The scenarios now get distinct, **cleared** directories, and the record carries both payloads so the separation is visible in the artifact itself. |
| **What it establishes, and what it does not** | It establishes the mechanism the amendment rests on: **a seed's result does not depend on what trains beside it**, over the whole result rather than the validation loss alone, and in the amendment's own two-seed configuration rather than extrapolated from one seed's two arms. It is **CPU, where the two processes contend for the same cores**; the declared host is two T4s with one seed per device and no shared compute, so this is the conservative case. It is at the **dev shape**, not the declared one. What remains is the **registration** - a change to section 5.2's session model - which is a decision, not a measurement. |
| **Pinned** | `tests/test_two_seed_concurrency_rehearsal.py`, 14 tests. The record is committed (an `.gitignore` exception, with the two checkpoint trees left as local scratch) because it is the evidence for the option's price, and it carries both scenarios' full payloads so the comparison can be re-run from the artifact alone. |
| **Result** | **1651 tests pass**; probe v3 still **QUEUED**; GPU **1.42 h** and TPU **19.90 h**, both refreshing **2026-09-26**. |

## What a TPU verdict has to clear to put the block in one GPU week, declared before the number arrives (2026-09-22)

| | |
|---|---|
| **What was missing, and why now** | The probe that decides the port is **queued**, and the rule settles *whether* to port. Nothing recorded what a passing verdict does for the campaign's **calendar** - which is the other half of the question, since the block is ~52 GPU-h of training against a 30 GPU-h week. So the schedule reading of the same measurement is derived and fixed here, **while the number is still unknown**, for the reason the rule itself was declared early: a threshold that appears after the result reads as fitted to it. |
| **The billing unit, measured rather than assumed** | `kaggle quota --format json`, 2026-09-22: **TPU 0.10 h used of 20.00 h**, GPU **28.58 h used of 30.00 h**, both refreshing **2026-09-26**. The 0.10 h is the two probe runs (148 s and 180 s = **5.5 min**) with the **4.3 h of queue between them billed not at all** - so the TPU line bills the hour a v3-8 is *running*, which is exactly the unit the rule's own `per_core_training_hours` is already in. Without that, comparing TPU hours with the rule's arithmetic would have been an assumption; with it, the two are the same clock. |
| **The arithmetic, from the rule's own constants** | A pair queue is `12 x 2 x 24,576,000` = **589.8M tokens**; the null queue is exactly half (one arm per seed, `12 x 1 x 24,576,000`). Eight cores do a queue in `tokens / (5,083.4 x m) / 3600` hours, where `m` is the multiple `evaluate` already reports as `multiple_of_t4_pair`. A queue costs **17.32 GPU-h** (twelve pairs at the measured 1.4429 h) whichever queue it is. |
| **The table** | At the rule's own **2.0x** pass mark: pair queue **16.12 h**, null **8.06 h** - one queue fits the 20-hour week, and the GPU side is **34.63 h**, which is **still two weeks**. At **m = 2.4173**: pair **13.33 h**, null **6.67 h**, **null + one pair queue = 20.00 h** - two queues fit, leaving **17.32 GPU-h**, which is **one week**. At **m = 3.22**: two *pair* queues fit (20.02 h) for the same one-week outcome, which is why the pivot is the cheaper pair. |
| **The pivot, now a constant** | `ONE_GPU_WEEK_MULTIPLE = 2.417`, derived in `scripts/tpu_probe.py` from the two cheapest queues rather than written into a document. **Below it the port *alone* buys a two-week block however clean the verdict** - the only other route to one week is the null-queue amendment in the next row - so a result between the 2.0x bar and 2.417 is a PASS whose meaning is "worth porting, and the calendar does not move unless the amendment is adopted too": a different decision from "port", which must not be discovered after the number. |
| **The two recorded options compose, and that is what lowers the pivot to the bar** | The null queue is the campaign's only **one-arm** queue, so two seeds per pair halves its GPU hours - 12 pairs becomes 6 at the same pair time, **8.66 h** instead of 17.32. One pair queue on the TPU (**16.12 h** at the rule's own 2.0x) then leaves `17.32 + 8.66 = 25.97 GPU-h` against the 30-hour week: **one week at the bar rather than at 2.417**. Both routes are reported by `week_capacity` (`carries_two_queues`, `one_week_with_one_queue_and_the_amendment`), and **neither is adopted here** - the second needs the registration amendment the null-queue entry prices, which is a decision this entry deliberately does not take. |
| **The disposition, per band, fixed while the number is unknown** | **DO-NOT-PORT** (`m` below the rule's own 2.0x bar): no port, and the only lever is the amendment, which takes the GPU side from 51.95 h to **43.29 h** - still two weeks, so the block's calendar does not move. **PORT** (2.0 <= `m` < **2.4173**, the exact pivot - at the *rounded* 2.417 the two queues sum to 20.002 h and do not fit): the TPU carries one pair queue (**16.12 h** at the bar) and the GPU side is **34.63 h**, i.e. two weeks **unless** the amendment is adopted, which makes it **25.97 h** and **one week**. **PORT at `m` >= 2.4173**: two queues on the TPU, GPU **17.32 h**, one week with no amendment at all. **PORT_FIXED_GRAPH_ONLY**: only the null queue ports - it is the campaign's only fixed-graph queue - so it leaves the GPU for **8.06 h** of TPU time and both pair queues stay, keeping the GPU at **34.63 h and two weeks**; the **amendment is moot in this branch**, because the queue it would have paired has already left, and the TPU's remaining ~11.94 h cannot take a pair queue (they rebuild). |
| **What the amendment is still blocked on, stated precisely** | Not the mechanism. `run_v0_session` trains each arm in its **own subprocess** from the same config and seed, and the 2026-09-19 check measured an arm's validation loss **bit-identical** with and without its co-running partner (`9.031953811645508`). What is *not* measured is that configuration at the declared shape with two **seeds** rather than one seed's two arms - structurally the same thing, since an arm's numbers cannot depend on which partner co-runs when nothing is shared, but **asserted rather than measured**, which is the distinction this journal keeps. The rehearsal is CPU-free and **not run**. |
| **The fp32 anchor already in the record** | The diagnostic measured **m = 1.734** (1,101.7 tok/s per core, 8,814 aggregate). So the pass mark needs bf16 at **1.154x** the fp32 per-core rate, and the pivot at **1.394x**. The rule's bar and the campaign's clock are therefore two different multiples of the same unknown, and both are stated before it lands. |
| **Correcting this session's own earlier pricing** | The null-queue entry priced "the TPU's separate quota" at **16.97 h** - which is the *GPU* hours the move frees, not the TPU hours it consumes. Those are **8.06 h at the bar**, because a null seed is one arm: the two are different quantities and the entry's conclusion (it fits) survives only by luck of the null queue being the cheap one. Both are now stated in their own units. |
| **Pinned** | `tests/test_tpu_decision.py`: the pivot is the *second* queue and sits above the bar; the boundary is inclusive at the pivot (a float comparison was landing on the wrong side of its own definition, caught by the test); the capacity reading rides with the verdict and is `None` for a record the rule cannot decide; and `PAIR_OVERHEAD_HOURS` / `PAIR_HOURS_ON_T4` are asserted **equal to `act.training.session_plan`'s own**, so the probe and the campaign cannot drift to two clocks. Mutations checked: pivot from one queue fails 1, the fit test forced true fails 1, the GPU side inverted fails 1, the null queue priced as a pair queue fails 1, the capacity reading dropped from the verdict fails 1, the amendment halving nothing fails 1, the amended null queue priced as before fails 1, and the TPU-week half of the amendment's fit dropped fails 1. |
| **Result** | **1637 tests pass**; probe v3 still **QUEUED** on `714dfa3` (verified from the dataset's own `revision.txt`, and confirmed to carry all three fixes - decision order, phase containment, release-before-rebuild - with only a behaviour-preserving refactor since, so **no republish**, which would change the code under a run already queued); GPU **1.42 h**; TPU **19.90 h**; both refresh **2026-09-26**. |

## The three launch commands, re-derived after every change this session (2026-09-22)

| | |
|---|---|
| **What was checked, and why now** | The campaign restarts **2026-09-26** on block **15024-15035**, and this session changed the launch path (the section 5.47 treatment, the block guard, the dataset slugs, the child's stdout environment). So all three queues' commands were re-derived from the builder itself and read against what the runbook documents - a **zero-GPU** check, run *after* the changes rather than resting on the suite's doc/code agreement pin alone. |
| **Primary** | `--mode pair-session --device cuda --hours 7 --seeds 15024-15035 --probe-candidate-policy spread --preserve-router-on-rebuild --new-expert clone --dataset-root artifacts/datasets --session-dir artifacts/v0-sessions --kaggle-dataset sid123123234/act-v0-5-2` |
| **Control (section 2b)** | the same plus `--probe-selection random`, on `--dataset-root artifacts/datasets-control` / `--kaggle-dataset sid123123234/act-v0-5-2-control` |
| **Null (section 2c)** | the same with `--session-arms null` and **no probe flag at all** (the null arm is policy-free by construction), on `artifacts/datasets-null` / `sid123123234/act-v0-5-2-null` |
| **What the check settles** | All three carry the section 5.47 rebuild pair; all three name the **new** block, never the spent one; and the control and null dataset slugs **derive from the primary's** (`Queue.suffix`), so a re-reservation moves all three roots together instead of leaving two queues pointing at a block that is no longer the live one - which is the mismatch the run-time guard would otherwise have to refuse, days later, at the cost of a session. |
| **Result** | The next session's command is the registered one, and no GPU hour was spent establishing that. |

## The null queue bills a two-device session to train one arm, and that device is free (2026-09-22)

| | |
|---|---|
| **How it was found** | Reading `PairSchedule` for the campaign's arithmetic and noticing what the *null* queue's own numbers say: `pair_time_hours = ceil(arms / arm_slots) x tokens / rate + overhead`, and the null queue is `arms=1, arm_slots=1`. The primary is `arms=2, arm_slots=2`. Both are **1.4143 h** - the null queue pays a full pair for **half** the training, with the second T4 idle for the whole pair. |
| **Why it is structural rather than a bug** | The idle device is intrinsic: the null arm is single-arm by construction (section 5.14's do-nothing queue has no second arm to compare *within* a seed), the token budget is registered and identical, and a session is billed in wall clock, not device-hours. Nothing in the launch path is wrong - the *placement* is what leaves a device unused, and `PairSchedule` correctly refuses `arm_slots > arms`, so the arithmetic cannot even express the waste it is paying for. |
| **What it costs the verdict** | The null queue is 12 seeds x 1.4143 h = **16.97 h** of billed session time, i.e. **2.76 sessions** - a third of the three-queue campaign's sessions for a quarter of its arms. |
| **Option 1, priced: two seeds per pair** | A null pair with **two seeds, one arm each** (which is exactly `arms=2, arm_slots=2`) costs the same 1.4143 h and completes two seeds: **8.49 h / 1.38 sessions**, saving **8.49 h - about 1.4 sessions, 28% of a week's quota**. It needs a **registered amendment** (it changes the session's unit of work, so it changes section 5.2's session model) plus a rehearsal proving two concurrently-running seeds are bit-identical to two sequential ones - which this repository can do on CPU for free, and which is the same evidence standard the arms' own independence was held to ("the adaptive arm's validation loss is bit-identical with and without its pair"). **Not started, and deliberately not snuck in:** the launch path is the one thing whose changes have cost this campaign hours. *(Updated 2026-09-22: the rehearsal has now run and the two seeds are **bit-identical on every result field** - see the entry above. The amendment is left on the table as a registration decision, not a measurement.)* |
| **Option 2, priced: the TPU's separate quota** | The whole null queue is **16.97 h**, which fits inside the **20 TPU-h/week** quota that is entirely separate from the GPU's. That frees **16.97 GPU-h - over half a week** - for the primary and control queues, and it needs no new arithmetic at all. It is gated on the port verdict, not on anything in this entry. *(Corrected 2026-09-22: **16.97 h is the GPU time freed**, not the TPU time consumed - a null seed is one arm, so the queue is **8.06 h** at the port rule's bar. The conclusion holds; the two clocks are now stated separately. See the entry above.)* |
| **The pre-verdict observation that belongs on the record now** | The null queue is a **fixed-graph arm**: zero mutations and zero probes by construction (section 5.14's built-in checks enforce both), so it has no mid-run rebuild - which is what the port rule's `PORT_FIXED_GRAPH_ONLY` branch is about, and why the rule's own text names fixed-graph arms rather than the session path. It is also the pattern the probe already measures directly: `independent_seeds` spawns **one single-arm seed per core with no collectives**, which is what a null seed is. Written here, **before** the verdict exists, so that acting on it later cannot look like a consolation discovered after the numbers - which is the thing the rule was written to prevent. |
| **Result** | Two options recorded, neither implemented: one needs a registered amendment and a rehearsal, the other needs the probe's verdict. The campaign's quota arithmetic is otherwise unchanged (**1.42 h** left, refreshing **2026-09-26**). |

## The front page described an engine and a tool the repository does not have (2026-09-22)

| | |
|---|---|
| **What was found** | `README.md`'s architecture block read `TRAINING ENGINE | PyTorch + T4x2 / JAX + TPU v5e`, and its feature list advertised **"Qwen-Informed Prior: Prior extraction and structural analysis tool for hybrid architectures"** and **"entropy tracking"**. |
| **Checked rather than assumed, because a dependency claim is checkable** | `import jax` / `from jax`: **zero** occurrences under `src/`, `scripts/` and `tests/`, and no `jax` anywhere in `pyproject.toml` or `uv.lock`. The TPU this project has actually measured is a **v3-8** under **torch_xla** - `xla_devices: 8`, torch 2.8.0+cpu, torch_xla 2.8.0, from this session's own probe - not a v5e, and not JAX. |
| **The tool that does not exist** | `prior extraction`, `extract_prior`: the **only** occurrence anywhere in the repository was the README's own generated egg-info mirror (`src/adaptive_computational_topology.egg-info/PKG-INFO`, which is untracked and is a copy of an older README). Nothing under `src/`, `scripts/` or `tests/` mentions prior extraction at all. |
| **And one claim that was half true** | `entropy tracking`: no occurrence anywhere in `src/act/evolution` or `src/act/topology`. What does exist is `pareto_frontier`, `dominates` and a `select_population` that retains **frontier diversity** first - so the bullet keeps the Pareto and diversity halves and drops entropy. The importance bullet's `Activation counters` likewise became the signals the module computes (`routing_load`, `parameter_norm`); the rest of it is real (`gradient_norm`, `fisher_ema`, `cosine_similarity`, per-expert functional sketches, `temporal_stability`, `expert_ablation_delta`). |
| **What the Qwen bullet became** | **Two** bullets, both checked against the tree: **prior-ranked module search** - `SearchPrior` (`module_scores` plus an explicit `depth_penalty`) ranking a genome's layer/module candidates through `rank_module_candidates`, which is the real capability the plan's "prior interface" row refers to - and **named mixer layouts** (the zoo's `gdn_3to1` Qwen3-Next-style 3:1 schedule, with the committed mixer benchmark). Naming a lineage is honest; claiming a prior-*extraction* tool was not. |
| **My own first correction under-described the project, and that is the worse direction** | The first fix replaced the Qwen bullet with the mixer bullet alone, deleting a mention of a capability the repository **has**: `src/act/evolution/hierarchical.py` holds `SearchPrior`, `PriorRankedCandidate` and `rank_module_candidates`. A front page that claims what does not exist is one defect; one that hides what does exist is another, and I introduced it while fixing the first. Corrected to name both, and the new general check below is what makes that hard to repeat. |
| **A fourth claim, found while checking the third** | The same bullet claimed mutations "on layers, **attention heads**, MoE experts, and residual routes". There is **no head-level operator anywhere** in `src/act/topology/` (no `num_heads`, no head mutation), and the named operators do not span the four targets - `prune_expert`/`grow_expert`/`clone_expert`/`merge_experts`/`split_expert` act on experts, `rewire_layer` on a residual input edge, `change_module` on a layer's mixer/feed-forward type, `toggle_layer` on effective depth. Rewritten to say exactly that. |
| **Why it was worth doing and what was left alone** | The front page is what a reader believes the project is, and `docs/complete_implementation_plan.md` is a **frozen plan** that still reads "Kaggle T4x2 primary, TPU v5e-8 secondary, JAX" - that is the plan's own text and it is deliberately **not** amended, exactly as the report's historical registrations are not. The README describes the project; the plan describes the intention, and they should differ where the implementation moved. |
| **Pinned in both directions, and one check generalised** | `tests/test_doc_index.py` gained three **agreement** checks, not absence checks: the architecture block's framework names must agree with what `src/` and `scripts/` import; the feature list must advertise a prior-extraction tool exactly when one exists; and - the general form of the whole defect - **every backticked snake_case identifier in the feature list must appear in some source file**, which is the check that would have caught the invented prior tool, the JAX line and the attention-head claim at once, and catches the next invented name without anyone thinking to look. Mutations: the JAX/v5e line back fails 1, the Qwen bullet back fails 1, a plausible invented symbol (`causal_ablation_study`) fails 1, an invented operator (`grow_attention_head`) fails 1, and adding `import jax` to `src/` - the *other* direction - fails 1. |
| **A bug in my own first version, kept on the record** | The first `_python_sources()` searched `tests/` too, so the test found the very literals it searches for and **passed** on a repository with no such tool - the same self-reference shape as the earlier "the search found itself" failures. Scoped to `src/` and `scripts/`, with the reason in its docstring. |
| **Result** | **1632 tests pass**; README corrected in six places (`JAX`, `TPU v5e`, the prior tool, attention heads, activation counters, entropy tracking); the frozen plan untouched by design. |

## A capture held the whole log of the session that succeeded, and it settles the exit-publish trade with a measurement (2026-09-22)

| | |
|---|---|
| **How it was found** | Grepping the failure captures for `remote publish` - the runbook asks for that push's duration **by name** - and one capture turned out to hold the platform's complete kernel log for `act-t4-session`, the session that trained seeds 15000-15003 and reached its exit publish. Four other sessions have died leaving nothing readable, so this is the only full account of a session the campaign has. |
| **The number the runbook asked for** | **25.42 GiB pushed in 12.0 min = 36.0 MiB/s.** Derivation, from the log's own two facts: the child was launched at `t = 226.3 s` with `--hours 7` and its own line `session done: 4 pair(s) completed, 1.12 h remaining` fixes its exit at `t = 21,394.3 s`; the driver's `=== session session finished after 6.14 h ===` sits at `t = 22,116.4 s`; the 722 s between them covers the publish, the `_keep` calls and the closing disk report, so **12.0 min is an upper bound on the push**. The figure this repository used before was ~51 MB/s **inferred from a download**, and it was 1.4x optimistic. |
| **The trade, confirmed rather than assumed** | Both schemes are cumulative - the exit payload grows to the full block, and a per-seed push re-uploads the whole handoff each time - so the comparison is `152.5 GiB / 1.20 h` for one exit publish per session (4, 8, 12 seeds) against `495.7 GiB / 3.91 h` for per-seed, i.e. **3.25x the bytes, 2.71 h apart**. The revisit criterion is met with room: this session's 12.0 min push protected **339 min** of trained pairs. The 45-minute reserve also stands, and it is sized for the **last** session, not this one: 12 seeds is 76.3 GiB = **36 min** at the measured rate. |
| **The mechanism that made four failures unreadable, now proved rather than hypothesised** | Every one of the child's own opening messages arrived **6.08 h late**, in the same batch as its last line: `Dataset URL: ...` and `no remote version ... available yet; training from scratch` are printed seconds after its launch (`t = 226.3 s`) and were delivered at `t = 22,116.3 s`. The wrapper's own lines at `t = 226.3 s` arrived immediately, so the stream was healthy and the delay is the **child's piped stdout block-buffering** - its whole run is a few KB, under one 8 KiB block, so nothing flushed until the process exited. That is the exact mechanism `kaggle_t4.py` now sets `PYTHONUNBUFFERED` for. |
| **What that explains in hindsight** | The live capture's `failed 5 times with no new data; giving up` at 4.1 minutes was **inevitable**: there was nothing to read, from a session that was running perfectly. And the four unexplained deaths are unexplained *because* of this - a **killed** child never flushes at all, which is why their logs are empty rather than short. So the streaming fix's target is confirmed by evidence, though it has still never run on Kaggle: the next session is its first execution. |
| **A healthy session's baseline, for the first time** | Window `--hours 7`, finished at **6.14 h** of the platform's 8-hour cap; **4 pairs** consuming **5.88 h** of that window, against `4 x 1.41 = 5.64 h` from the schedule's own pair time (line 28) - so the schedule is right to within the 0.24 h of startup the child also pays. Arms in parallel, one per T4; payload 6.36 GiB per seed. It stopped at `seed 15004: remaining 1.12 h does not fit pair 1.41 h + 45 min reserve (rule 1)`, leaving 1.12 h of its window unused. The log carries **no per-pair durations**, because every child line arrived in one batch at exit - the buffering finding below, not an omission. |
| **The unused tail is structural, not a defect** | A fifth pair needs `1.41 + 0.75 = 2.16 h` of clock and the child had consumed 5.88 h by then, so it would have needed a window of **at least 8.04 h** - against a platform cap of **8 h total**, which must also cover boot (226 s) and the publish (12 min). So **4 pairs is the maximum this pair time allows on Kaggle**, and the stop is correct. A smaller reserve would not have helped either: the tail (1.12 h) is below the pair itself (1.41 h), so the rule never had a legal pair to start. Nothing to fix, and nothing invented to look like a fix. |
| **The trick worth *not* inventing** | Per-pair durability of the **numbers** looks like it wants a second, tiny dataset - and it does not: the log's line 54, `kept 4 seed summaries in the kernel output (publish insurance)`, is that mechanism already, at kilobytes, and the 2026-09-21 fixes made it survive a death. The **checkpoints** are the heavy part (6.36 GiB/seed) and no cheap scheme publishes them incrementally, which is precisely what the 3.25x measures. |
| **The evidence is committed, not quoted** | `artifacts/v0-session-logs/session-15000-15003-log.json` (byte-identical copy, `sha256 8a6adf2d...`, 75 lines) with `NOTES.md` carrying the derivations above, and a `.gitignore` allow-list following the arch-zoo convention so a future session log is committable without an edit. The log is the recomputation basis for every number in this entry. |
| **Result** | Runbook section 6 now states the measured trade instead of asking for it. Probe v3 still **QUEUED** (20 TPU-h available); GPU quota **1.42 h** against a 1.65 h pair, refreshing **2026-09-26**, so the campaign stays stopped. |

## The TPU probe finally measured real hardware, then died inside the one dtype the decision never reads (2026-09-22)

| | |
|---|---|
| **The run** | `act-tpu-probe` v2 was pushed 2026-09-21 21:04 and started **01:21** (about **4.3 h** in queue). It ran **180 s**, measured the declared 285M shape on real TPU hardware, and ended **ERROR**. |
| **The 2026-09-21 durability fix worked** | `kaggle kernels output` returned a **record** this time, not only a log, and it carries `completed: ["fp32.throughput"]`. The first run died 148 s in having written nothing at all, so this is precisely the difference the per-measurement `_save` bought: 4.3 h of queue no longer produces zero numbers. |
| **What it bought - the first real TPU numbers at this shape** | **8 XLA devices** (`device_kind: xla:0`; torch 2.8.0+cpu, torch_xla 2.8.0); declared shape **285,336,576 parameters**; fp32 throughput **1,101.7 tokens/s on one core**, first step **69.05 s including compile**, steady step **3.7178 s** for 4,096 tokens (min 1.0086 s). |
| **The failure, verbatim** | `RuntimeError: Bad StatusOr access: RESOURCE_EXHAUSTED: Error loading program: Attempting to reserve 7.96G at the bottom of memory. That was not possible. There are 5.33G free, 0B reserved, and 5.33G reservable`, raised through `_rebuild_cost` -> `_measure` -> `v0_moe.py::_route` (`self.last_load = torch.bincount(...)`). |
| **Why that cost the whole run** | The measurement order was `("fp32", "bf16")` while **`evaluate` reads `bf16` and `independent_seeds`** - fp32 is a diagnostic nothing decides on. So the run spent its one slot on the dtype that cannot decide, and the pre-registered rule over its own record is `INCONCLUSIVE`. Verified live rather than asserted: `tpu_probe.py --evaluate <that record>` prints the verdict, names the four missing fields, and exits **3**. |
| **Why the memory ran out** | The probe held the declared fp32 model **and its stepped AdamW state** while building the rebuilt model **and a second AdamW state** - two sessions' worth, in order to measure one rebuild. The session's own event peak is narrower: `v0_session._adaptive_event` does hold the previous model while it rebuilds the winner (`winner = _rebuild_mutated(model, ...)`), but only **one** optimizer's state is ever stepped, so what an event costs over a running session is a model's parameters and grads, not a second trained optimizer. |
| **Fix 1 - measure in the decision's order** | `MEASUREMENT_ORDER = ("bf16", "fp32")`, with the eight-worker aggregate placed after the first dtype's measurements and before the last dtype's. The order is a pure function (`phase_names`), so it is testable without a TPU - which matters because an order is exactly what a refactor flips silently. |
| **Fix 2 - contain a phase** | `_run_phase` records a failure as `{"phase": ..., "error": ...}` under `errors`, keeps it out of `completed`, saves the record, and lets the measurements after it run. A diagnostic's failure no longer ends the kernel ERROR and no longer takes the phases behind it down. Containment does not soften the rule: a missing measurement is still `INCONCLUSIVE`, never a pass. |
| **Fix 3 - release before rebuilding** | The declared model and its optimizer are dropped once their throughput is timed (the genome survives, being device-free), so the rebuild measures one graph rather than two sessions' state. |
| **Pinned** | `tests/test_tpu_decision.py`: the order itself; the **coupling** that the dtype the rule requires is the dtype measured first, asserted against `evaluate`'s own missing-field list rather than a second literal that happens to agree; each phase **bound to its own dtype and state**, so a loop-built table cannot run every phase as the last dtype and write one measurement under two names - a record that reads as complete and is wrong; a failing phase is recorded, returns `False`, and the next phase still runs; the record reaches disk **for a failing phase too**; and the release before any rebuild. Mutations checked: fp32 first fails 1, re-raising instead of recording fails 1, emitting the aggregate last fails 1, keeping the declared model alive fails 1, late-binding the phase closures fails 1, giving a rebuild a fresh state dict fails 1 - and **the "save only on success" mutant survived the first version of the durability test**, because that version never let a phase fail. Rewritten to assert the failing case, which is the case the test exists for. |
| **What the fp32 diagnostic already says - and what it does not** | The bar is `2 x 5,083.4 = 10,166.8` tokens/s aggregate, i.e. **1,270.9 per core**. fp32 measured **1,101.7**, so eight cores at the fp32 rate would be **1.73x** the T4 pair: a `DO-NOT-PORT` reading, on a dtype the session does not train in. bf16 needs only **1.154x** the fp32 per-core rate to clear the bar, and bf16 is what the rule reads. The two conditions also trade against each other, which is a further reason neither is decided here: a faster core **raises** the recompile fraction, since it is compile time over training time, so clearing throughput does not clear the rebuild condition. |
| **Result** | **1633 tests pass**; probe **v3 queued** on the fixed revision (`714dfa3`; the code dataset is republished at that hash, and nothing has republished since - which matters because a Kaggle kernel mounts the dataset's *latest* version when it starts, so a publish while it queues would change the code under a run already waiting). The record will now hold bf16 and the eight-worker aggregate **before** any fp32 diagnostic runs. |

## The declared block has been training the configuration the runbook does not describe (2026-09-21)

| | |
|---|---|
| **How it was found** | Reading the four published seeds' **own** summaries for their cost numbers - a free readout, one 5 KB file per seed - and noticing what else they record. `config.preserve_router_on_rebuild: false`, `config.new_expert_initialization: "random"`. |
| **The two facts that cannot both be true** | The runbook's section 2 prints `--preserve-router-on-rebuild --new-expert clone` and calls the pair **"not optional": they are the adopted protocol (section 5.5 and, for the rebuild, the section 5.47 amendment)"**. Section 5.47 registers those two fields as *the declared launch command's* configuration. The artifacts say the block ran neither. |
| **Why** | `session_argv` - the command the Kaggle driver actually builds for all three queues - **never passed the flags**. The runbook's command and the builder were two copies of one protocol, and only one of them was amended in section 5.47. `V0SessionConfig`'s defaults are the historical untreated pair (`False`, `"random"`) and the CLI deliberately keeps them that way to preserve the committed record of sections 5.16-5.23, so the flags are how a run opts in - and the launch path never opted in. |
| **Why nothing caught it, including a test named for the question** | `test_primary_launch_command_is_the_registered_one` asserted the *whole argv* as a literal list, without the two flags: the test named "the registered one" was pinning a command the runbook does not describe, so the code was self-consistent and green. The verdict would not have stopped it either - `config_conformance` **changes no criterion and gates nothing** (section 5.47: "the conformance block would then record the run as flagged"); it prints a `differs` row and still returns PASS. The runbook even anticipated the outcome in writing - "if it reports `differs` on `preserve_router_on_rebuild`, the session ran without the 5.47 treatment ... which must be said in the write-up rather than discovered later" - and nothing made it impossible. Both fields are valid, neither is a gate input, and no parameter count moves, so no assertion in the project fires on the difference. |
| **The consequence that had not happened yet** | With 4 of 12 seeds untreated, resuming on the amended command would have appended **treated** seeds 15004 onward to an **untreated** block. A block is what the verdict judges and its seeds are compared *within* it, so that is not one experiment with a mixed config - it is two experiments sharing a seed range, and **no per-seed field partitions them**, so the split could only have been reconstructed by hand. Nothing would have refused it: a session launched either way runs to completion and publishes. |
| **Fix 1 - the builder carries the registered protocol** | `session_argv` now appends `REBUILD_TREATMENT_FLAGS` (`--preserve-router-on-rebuild --new-expert clone`) unconditionally, which is what runbook section 2c asks for in as many words: they are inert on a null arm "and they are passed anyway so that all three queues are the same declaration. A config that differs between queues is a difference nobody would notice until it mattered." One constant, so the pair cannot be half-applied. |
| **Fix 2 - a session refuses to extend a block it would break** | `check_block_treatment_uniform` runs **immediately before the first pair starts** (never on a run whose queue is gated shut, which cannot mix anything) and compares the treatment this session would use against the treatment the **published versions** record. It reads the versions present in the root, not the seeds the invocation was handed - the first version read its own queue and was therefore silent in exactly the case it exists for, since a resumed campaign is launched `--seeds 15004-15011` and the disagreeing seeds are the ones already finished. On a conflict it raises `BlockTreatmentMismatch` naming both fields, both values and the published seeds, and **chooses nothing**: the two remedies are printed (finish the block under the configuration it records, or start a fresh block on an unspent range) and the disposition stays where it belongs. On agreement it states the treatment in the log - which is the legibility half, on the same principle as the conformance block. |
| **Fix 3 - the doc and the code are now pinned to each other** | `test_the_runbook_and_the_launch_builder_carry_the_same_rebuild_flags` extracts every `pair-session` command the runbook prints (sections 2, 2b, 2c) and asserts each carries the pair, then asserts the builder does. This is the only check that could have caught this class: a doc-only assertion passes (the doc was right) and a code-only assertion passes (the code was self-consistent). Agreement is the missing invariant, and it is now the one under test. |
| **Verified live, before any GPU hour** | Four steps on the real CLI path. (1) The null queue launched exactly as the builder now spells it - treated flags on a policy-free arm - completes and records `preserve_router_on_rebuild: True`, `new_expert_initialization: 'clone'`. (2) A resume of that block **without** the flags refuses: `preserve_router_on_rebuild: this session False, published True (seed(s) 9000)` beside the same line for the expert initialisation, and **trains and publishes nothing**. (3) A resume **with** them logs `block treatment matches the published seed(s)` and completes the pair. (4) The first session of a block states the treatment it is setting rather than staying silent. |
| **Pinned** | `tests/test_kaggle_t4.py`: the registered argv **with** the pair, and the runbook/builder agreement across all three sections. `tests/test_v0_runner.py`: the refusal (both fields, both values, the disagreeing seed, and that nothing trained or published), the narrower-queue case that the first version of the guard missed, the matching-block log line, the opening-session log line, and that a summary recording no config is read as silence rather than conflict - the same `not_recorded` convention the verdict uses, so no spent block is re-read as non-conforming. Mutations checked: dropping the flags from the builder fails **2** tests, reading no published seeds fails **2**, and removing the call site fails **3**. |
| **The disposition: re-reserve the block (section 5.65)** | Seeds **15024-15035**, dataset **`act-v0-5-2`**, all three queues together (section 2c requires the same block). Two grounds, in order. (1) **The artifact must be the protocol**: `15000-15011` ran the untreated rebuild, so a Gate 4 verdict over it would be a PASS whose own conformance text reads *"the published run is not the registered configuration, so the verdict describes a different protocol than the one it names"*. (2) **The block was no longer blind**: its first four seeds' harm numbers were read *and reported* while the campaign was stopped, which is precisely the condition section 5.5 change 4 re-reserved a block for - and amending the registration to match what the buggy launch did, after seeing those numbers, is the one move this project does not make. Finishing it untreated would have been **~6.6 GPU-h cheaper** and was rejected on those grounds, not on cost; restarting the same four seeds treated costs the same as this amendment and still reuses seeds whose outcomes are known. |
| **Cost, and what is kept** | The block is 12 seeds either way, so the delta is the four already-trained seeds: **~6.6 GPU-h**. The four are **not discarded** - they keep their versions in `act-v0-5-1` as a measured untreated block, and their numbers are already reported above. `15012-15023` is skipped deliberately: 15012 is the benchmark's reserved data seed, and the skip is stated in `DECLARED_SEEDS` so the gap is not read as an oversight. |
| **A cold start is required once** | Changing the seed block is not enough: a session refuses to train a pair whose treatment differs from the versions already in its **root**, so the block and the root move together - one root is one block. That is why `DEFAULT_SLUG` moved too, and why the four old seeds stay published in the old slug rather than being deleted (the runner never overwrites a complete version, and this amendment does not either). |
| **Result** | **1624 tests pass**; queue **4 of 12** on the superseded block, **0 of 12** on the live one; quota **1.42 h**; campaign stopped until the **2026-09-26** refresh. The next session trains **15024-15035** under the section 5.47 treatment and will refuse to mix. |

## What the four published seeds show, read for free while the GPU week is spent (2026-09-21)

| | |
|---|---|
| **The command** | `PYTHONPATH=src python scripts/campaign_health.py --readout`. The health report answered *is anything driving this?*; this answers *what does it show so far?* - from durable remote state, without a session, a GPU-hour, or the 24 GB dataset: each published seed's summary is a **5 KB file** and `kaggle datasets download -f` serves one path out of a dataset. Opt-in, because it costs a small download per seed and the health check is meant to be cheap. |
| **Why it is not a verdict, in the command itself** | The queue is incomplete and the registered rule withholds itself until it is, so the block reports the **inputs** - each seed's two arm losses, the difference, the win count - and says in its own output that no threshold is applied. A provisional verdict is exactly what pre-registration exists to prevent. |
| **The four seeds** | adaptive won **2 of 4** (seeds 15000, 15001); mean adaptive-minus-random **+0.000041 nats**. Per seed: 15000 `-0.000162`, 15001 `-0.000591`, 15002 `+0.000546`, 15003 `+0.000372`. |
| **The number that has to be read beside those** | Every arm sits within **0.001120 nats** of `ln(32768) = 10.3972`. The declared block trains on **synthetic tokens** - `config.data_path` is `null` in the published summary, read off the seed's own record rather than inferred - so `ln(vocab_size)` is the loss a model with nothing to learn reaches, and differences of 1e-4 nats are the differences between two arms that have both arrived there. |
| **The registered scope, which is what keeps that from being over-read** | Section 5.1 states its own scope and it is not quality: the protocol is "a measurement-plus-evidence protocol, not a quality gate", Gate 4 is "not entered until it passes", and the adaptive-vs-random comparison is introduced in as many words as **"Harm guard (not a powered quality claim)"** whose consequence is one-sided - "cheap-and-harmful is worse than expensive-and-helpful". Section 10.2's null clause has the same shape, and the project's own `null_clause_power.py` says a cleared clause is not evidence of value. So a floor-level 1e-4 difference is **not** a contradiction of the registered design; the design claims a **cost ratio** (search FLOPs <= 5% of training FLOPs, predicted at 0.0942% and far under the bar), which is data-independent. What the floor line is for is narrower and still worth having: it makes it visible that the block's arms are being compared at a scale where no learning signal exists, so that a later reader cannot narrate a 1e-4 win count as a quality result. |
| **The project has already priced this class of reading** | `SCREENING_AUDIT.md` invalidated the zoo's screening table for precisely this reason - "uniform random tokens ... the reason the synthetic table ranks initialization luck" - and `null_clause_power.py` computed, before this block's data existed, that a cleared clause needs only 3 of 12 seeds and fires on 79/4096 blocks under no effect. So the block's clauses were given their power in advance, and this readout adds no new claim to them; it makes the numbers visible while the hours are still being spent. |
| **Two measurements the seeds carry that nothing else did** | Per-arm `wall_time_seconds` is **1.434-1.463 h** (5,164-5,268 s), which confirms the benchmark's 1.6477 h pair prediction to ~7% with the arms in one wave - and bounds the 3.50 h loss at ~2.4 pairs, independent of the pair-time arithmetic reproducing itself. And per-arm `tokens_per_second` is **4,665-4,759** against the schedule's 4,410.88, so the schedule the deadline rules use is ~6% conservative. |
| **Result** | **1615 tests pass**; queue **4 of 12**; quota **1.42 h**; campaign stopped at exit 1. |

## A third session died unread, because the capture gave up at minute four and the child never spoke (2026-09-21)

| | |
|---|---|
| **The failure** | `act-t4-session` ended **ERROR after 3.50 h of wall clock, 3.48 h of it after the first RUNNING poll**, used **3.49 GPU-h**, and published **nothing**: the primary queue is unchanged at **4 of 12** (`15000`-`15003`). Weekly quota is now **1.42 h**, against a measured **1.65 h** pair. |
| **The capture ran, and died at 4.1 min** | The live capture is new since the last failure, and it worked - it holds the header, the disk report, the throughput benchmark and the child's command line. Then: `Log stream connection lost, reconnecting...` x3, `Log stream connection lost: failed 5 times with no new data; giving up`. The CLI gives up **permanently**, so 3.4 h of a 3.5 h run went unread by the one channel that exists for exactly this. |
| **Why it had no data to read** | `kaggle kernels logs -f` streams a kernel's *output*; a child whose output is buffered produces none. The session ran the code dataset at **`b1f614a`** (published 15:22:01, its push 15:21:46), and `Stream a step's output` (`8894e4c`) landed at **15:26:30** - **five minutes after the push**. So this session held every pair's output in memory with `capture_output=True` and released it only on exit, exactly as the fix's docstring describes, and the stream starved. Two halves of one failure, both now closed; the child-streaming fix has still never run on Kaggle. |
| **The TPU probe finally ran, and measured nothing** | After ~6.5 h in Kaggle's TPU queue the probe reached a v3-8 and raised **148 s in**: `ValueError: V0 MoE reconstruction requires uniform active expert counts`, from its own hand-written mutation - append an expert *and* raise that layer's `active_experts`. It wrote no record, so the pre-registered rule had nothing to read (`INCONCLUSIVE`) and 6.5 h of queue bought no number. |
| **The gap that made the probe's edit look legal** | A per-layer `active_experts` is **valid to the genome validator** and the V0 MoE config holds **one scalar for every layer**, so that edit is a genome the project can hold and cannot build. `grow_expert` - the operator a session actually applies - does not touch `active_experts`, and cannot produce it from the declared shape (8 experts, 2 active, prune clamped at the same count). So this was the probe's bug, not a session hazard, and it is now pinned as a trap: the hand-rolled version must keep raising. |
| **The buffering that was left, and it was the actual cause** | Streaming the child's output was necessary and **not sufficient**: the child's stdout is a *pipe* here, so Python block-buffers it (8 KiB), and the runner logs through `log=print`, which does not flush. A session that prints ~12 progress lines per 99-minute pair would therefore emit nothing for hours *even with the launcher forwarding every line it receives* - which is exactly the 28-line capture: header, disk report and benchmark, all of them the launcher's own lines, and nothing from the child. The archzoo sweeps passed `PYTHONUNBUFFERED=1` explicitly, one mode at a time, and the session - the mode that matters most - was missed. It is now set once in `_bootstrap`, so every step and every child of every step is unbuffered. Pinned by **demonstration**, not assertion: the same child is read through the bootstrap's env and then through the env without the flag, and the second is shown to deliver nothing. |
| **Fix 1 - the rebuild uses the project's operator** | `_rebuild_genome` calls `act.topology.mutations.grow_expert` on a copy, so the rebuild is the mutation a real event produces rather than an approximation, and the four measurements are reachable at all. |
| **Fix 2 - the probe writes as it measures** | It saved once, after every measurement, so a 148 s death discarded everything - including a throughput it had already measured. Each measurement now lands on disk as it completes and the record carries a **`completed` list**, because a partial record must not read like a finished one. |
| **Fix 3 - the capture restarts while the kernel is live** | The tailer is a supervisor on its own thread: when a connection drops it restarts, until the session ends. Because the CLI **replays from the kernel's first line** (measured: a completed kernel's stream begins at its header, 195 identical lines twice), each attempt appends only lines the previous one did not - skipping the replay by the line count already on disk, and writing a `did not match what was captured` note rather than silently trusting an estimate it cannot verify. |
| **Fix 4 - the audit trail was kept only on the success path** | Every `_keep` call - the session log, the publish log, the per-seed summaries - sat **after** the child's exit, and `_run` raises on a non-zero child, so a session that died kept **nothing**. That is why four failures have left an empty kernel output: `kaggle kernels output` is the one command that still answers for a run the platform killed, and the files that would have answered were written on the branch a killed run never reaches. The keeps now run in a `finally`, and the child's streamed output is **teed** into `/kaggle/working/session-stream.log` line by line as it arrives - because a hard kill runs no `finally` either, so the copy that survives one has to be written during the run rather than after it. |
| **What that changes about the exit-publish trade** | The runbook accepts, deliberately and in writing, that "a session that dies before its exit publish loses every pair it trained". Its criterion for revisiting the trade is push duration against pair time: **~9 min against ~99 min**, so on that criterion the trade stands and was left alone. What changes is the *consequence*: the per-seed summaries are local handoff files, so a session that dies before its exit publish now leaves the numbers for every pair it did complete in the kernel output. The checkpoints remain dataset-only and the seed still does not count as complete, but the loss becomes measurable - the difference between "3.49 GPU-h produced nothing" and "3.49 GPU-h produced N pairs' measurements". |
| **The competing option, priced with what is known and what is guessed** | Publishing after each completed pair would make each pair durable and cost the cumulative re-upload of the staging root: ~2.8 GPU-h across the campaign against an expected saving of ~0.7 GPU-h per session at the observed 3-in-7 failure rate. Those two numbers point the same way, and **neither is good enough to act on**: the saving is derived from a failure rate that includes the driver race and the publish-under-a-live-session bug, both since fixed, and the cost rests on an *estimated* upload rate inferred from a download. The exit publish's own duration is logged (`remote publish: ...`), so the next session that reaches it supplies the missing measurement - which is what the runbook asked for ("revisit this choice with that number rather than an assumption") and what is deliberately not being invented here. |
| **Fix 5 - the driver reads a fourth channel, and takes the longest rather than the first** | `session_failure_log` now also reads the tee, and - the correction this failure forced - **keeps the fullest channel instead of the first**. Its live capture was genuine, and first in order, and held **28 lines**, so returning it short-circuited three channels that could only have been longer. An account that wins by order is worse than none, because it reads like the whole thing. The channel name is taken from the producer (`kaggle_t4.STREAM_LOG_NAME`) so the writer and the reader cannot drift, and the runbook's "reading a session that died" section now says to fetch the tee after any failure. |
| **What is still not known, and what is now bounded** | **Why the session died is still unexplained.** Four sessions have now failed and none has left a readable account; this one was the first with a capture running and that capture's own limit ended it at minute four. What *is* bounded is the shape of the loss. `PairSchedule` reproduces the benchmark's own pair time exactly from the declared queue and the measured fp16 rate (24,576,000 tokens per arm at 4410.88 tokens/s, two arms in one wave, plus the 0.1 h event overhead = **1.6477 h**, matching the artifact to six decimals), and the run's **3.48 h is 2.11 pairs' worth**. So the session very probably trained two pairs and died at or before the exit publish - which is the failure the runbook explicitly accepts ("a session that dies before its exit publish loses every pair it trained") rather than a stall in the training path. That reading also means the pair-time arithmetic is not the suspect it looked like: its inputs are measured, and they are consistent with the run. |
| **Why the campaign stays stopped** | **1.42 GPU-h** remain and a pair measures **1.65 h**: no whole pair fits before the **2026-09-26** refresh, so a push would spend the remainder on a pair it cannot finish and publish nothing. Unused hours expire, so this is the honest end of the GPU week, not caution. `--wait-for-quota/--max-waits` exists for exactly this and is left to an operator, because a five-day sleep is a deployment decision rather than a measurement. |
| **Pinned** | `tests/test_kaggle_queue.py`: a dropped capture is restarted and the replay is not duplicated (three connections, one clean transcript), a divergent replay is reported rather than hidden, and the capture tails concurrently and stops after the block. `tests/test_tpu_decision.py`: the rebuild grows exactly one expert with the project's operator and leaves the caller's declared genome untouched, the rebuilt genome is **buildable** with uniform active counts at both toy and **declared** scale, the hand-rolled edit that broke the probe still validates and still cannot be built, a partial record is written where it stands, and every measurement is saved before the next runs. `tests/test_kaggle_t4.py`: the tee receives a failing child's output, the audit trail is inside a **`finally`** and the tee write inside the stream loop - asserted over the **AST**, because the first version of both tests passed mutants that had dedented the keeps back onto the success path and moved the tee write out of the loop. `tests/test_kaggle_queue.py` also pins the new channel: the tee is read back and named, a stub capture does not mask a fuller account, and the two modules agree on the filename. All nine new pins were mutation-checked. |
| **Result** | **1610 tests pass**; **283/283 claims agree** (`scripts/verify_record.py` re-run after these changes, not carried over); queue **4 of 12**; quota **1.42 h**; campaign stopped at exit 1. |

## Every session could see one of its four completed seeds, and re-trained the others (2026-09-21)

| | |
|---|---|
| **How it was found** | The startup-restore probe - **CPU-only, so it cost no GPU quota** - eliminated its own hypothesis in 7 minutes and found something worse. The restore is **6.7 min / 25.42 GiB on disk**, so it is not why a session died after 1.32 h. But it reported `restored 1 seed version(s) from sid123123234/act-v0-5-1: [15003]` against **4 published seeds**. |
| **Why** | The live dataset stores each seed one level *deeper* than the handoff layout names it: `seed-15000/seed-15000/checkpoints/...`. A seed is uploaded as `seed-NNNN.tar` whose members already carry the `seed-NNNN/` prefix, and the service keeps both. So the handoff parent's children have no `manifests/manifest.json`, `_locate_handoff_tree` fell through to matching a single *seed* directory, and the restore returned one of four. |
| **What that cost** | The handoff's own readers (`seed_directory`, `published_seed_status`, `restore_seed`) look where the handoff *writes* seeds, and the restore put back one. Every other completed seed therefore read as **missing** and was **trained again - ~1.5 h each** - while the driver's progress read (the session log's cumulative labels) reported the same 4 of 12 either way. That is the worst shape a defect can have: real hours, no visible symptom. |
| **Fix** | `published_seed_dirs` searches the extraction for **every** seed carrying a complete manifest, at any depth (shallowest wins, so a nested seed cannot shadow a real one), and `restore_latest` places each at the name the handoff itself supplies (`_seed_dirname` asks `seed_directory` rather than spelling it, so a restore cannot put a seed where the handoff's readers will not look). Pinned by a test that models the depth the **live service returned**, not the depth we asked for. |
| **Why the suite never caught it** | The round-trip test's fake client modelled the service as extracting tar *members only* - single nesting - so the fake and the locator shared one assumption, and the pair agreed with itself. The new test is the first that models what the service actually did. |
| **Verified live, and free** | The same CPU probe re-run against the fixed reader: `restored 4 seed version(s) from sid123123234/act-v0-5-1: [15000, 15001, 15002, 15003]`, **8.5 min, 25.42 GiB written** - 4 of 4 where it was 1 of 4. The measurement is on the artifact trail as `artifacts/restore-probe.json`. A session now skips the four completed seeds instead of re-training three of them, and the campaign was relaunched on a fitted **4.5 h window** (two whole pairs, so seeds 15004-15005) rather than letting the remaining hours expire. |
| **Also measured, free of charge** | The host overlay is `7.9T, 6.9T used, 1.1T free` (**88% full, shared with other tenants**) while `/kaggle/working` is `19.5 GiB` - confirming the handoff roots belong in `/tmp` as designed, and that the shared overlay is the only disk that scales. |
| **Still not claimed** | The **1.32 h session death remains unexplained**. What is now true is that a failure of that kind is *readable* (the log fix), that the clock it is judged against is honest (`window_left`), and that a real source of silent waste has been removed. |
| **Result** | **1592 tests pass**; 283/283 claims agree; campaign still stopped, **4.9 GPU-h** left this week. |

## A third session died with nothing to read, because the driver never asked for the log (2026-09-21)

| | |
|---|---|
| **The failure** | `act-t4-session` ended **ERROR after 1.32 h of wall clock**, used **~1.3 GPU-h**, published nothing, and the platform retained **no log and no output files** for it. The primary queue is unchanged at **4 of 12 seeds** (`15000`-`15003`, last published 08:02). |
| **Ruled out by measurement, not by argument** | (a) *A second driver*: the per-queue `flock` added after the 0.29 h loss was held for the whole run (`campaign_health.py` read the holder while it ran), and a push-replacement needs the lock. (b) *A mid-run code-dataset republish*: `act-lab-code` was published **once** today, at **13:05:53**, i.e. 20 s before this session's push and 79 min before it died - and the driver re-stages the kernel body whenever the dataset moves. (c) *The test suite*: it ran during the session but every driver test replaces `_run`, so no test can push; the 1252 `/tmp/act-queue-session-*` directories that look alarming are per-test `mkdtemp` fixtures, not pushes. (d) *Disk*: the handoff roots live in the container overlay (~1 TB free), not `/kaggle/working` (measured **19.5 GiB**), so the 22.7 GiB payload does not fit the working directory - and that is by design, not the failure. |
| **The defect that made three failures unreadable** | **The driver never asked Kaggle for the execution log.** `kaggle kernels output` returns a kernel's *files* and has never returned a log, so the `"no kernel log"` branch was reached for a log it had never requested - and the three channels that could carry one were untried. `kaggle kernels logs [-f]` exists and works: verified here, it returns full logs for `act-t4-preflight` and `act-t4-archzoo`, and **nothing at all** for the failed session (so the platform keeps no log for a run it kills, while keeping them for runs that complete). That is exactly the case the live stream exists for. |
| **Fix 1 - capture it while it runs** | `capture_live_log` streams `kaggle kernels logs -f` to local disk for the whole wait, so the copy exists on this machine *before* the platform decides what to keep. This is the only channel that can survive a failure like this one, and it is the difference between explaining the next failure and guessing at it. |
| **Fix 2 - ask for it afterwards** | `fetch_kernel_log` runs `kaggle kernels logs` on the failure path, and `session_failure_log` reads three channels in order of fidelity - live capture, kernel output, post-hoc fetch - and **names which one produced the text**, because a log that is merely empty and a log that was never requested otherwise read alike. `_print_log_tail` gained the same fallback, so the diagnostics (whose whole result *is* the log) stop printing nothing. |
| **Fix 3 - stop inferring the cause from the clock** | The note took wall time from the **push**, which contains the queue wait, and inferred "it died at or before startup" from it. `_wait_for_terminal_detailed` now also reports when the status was first **RUNNING**, and the failure line states the run duration beside the wall duration. The note also no longer asserts a cause: it names what is *known* to occupy a session that long and states the two causes already excluded. |
| **What is known to occupy a session that long, and was priced nowhere** | The session's **startup restore**: `restore_latest` downloads the *entire* published dataset before a token is trained - **22.7 GiB at four seeds**, growing ~6.4 GiB per seed - so the fixed cost of every session rises for as long as the queue runs. `restore_latest` now logs the download's **duration and GiB on disk**, so the next session's (now captured) log carries the number instead of an expectation. |
| **And it proved itself on first use** | The startup-restore probe's first push died `ModuleNotFoundError: No module named 'act'`, and the driver printed **the full traceback** - the first time a failure in this campaign has been readable at all. The bug was the probe's own: it imported `act` above `_bootstrap()`, which is the line that puts the mounted `src/` on `sys.path` - the same failure mode `_bootstrap` already documents from 2026-09-15. Fixed, and now pinned by a source-level check that no `mode_*` function imports the package before it bootstraps, because this class of failure only ever appears on Kaggle. |
| **Paid for without quota** | The probe that prices that restore is **CPU-only** (`act-t4-restoreprobe`, `enable_gpu: false`): it measures a download and a local merge, which no accelerator can influence, and it reads the session queue's dataset without ever writing it - so it cannot disturb campaign progress, and it costs none of the 4.9 GPU-h left this week. |
| **Not claimed** | The **cause is not proven**. What is proven is that every explanation the driver offered for its three failures was reachable without the session's own account, and that the one channel carrying it was never opened. Re-pushing would spend another ~1.3 GPU-h on the same unknown, so the campaign stays stopped with **4.9 GPU-h left** this week. |
| **Pinned** | `tests/test_kaggle_queue.py`: the log is *requested* via `kaggle kernels logs` (a driver that only runs `kernels output` cannot tell an absent log from an unasked-for one), the live capture is preferred and named, the fetch is used when the output carries none, and the run clock ignores queue time (with the never-RUNNING case falling back rather than inventing a run). `tests/conftest.py` neutralises the tailer suite-wide so no test can reach Kaggle through the one path that spawns a process. `tests/test_kaggle_remote.py`: the restore prices its own download. All five new pins were mutation-checked. |
| **Result** | **1584 tests pass**; 283/283 claims agree; campaign stopped by design (exit 1 = "read the cause before spending more"). |

## The gate chain is declared, and what each remaining gate leaves open is now on the record (2026-09-21)

| | |
|---|---|
| **Why** | Section 99 states gates 5-7 in one line each. Gate 4 is instrumented - a pre-registered verdict, a rehearsal of it, knobs nobody may move against a measured result - because it is the gate that was reached. The rest are sentences, so they would be *defined* when their artifacts arrive, i.e. against the numbers that came back. |
| **What** | `src/act/training/gate_chain.py`: each gate's artifact, comparison, bar and required predecessor. Gate 6 requires **identity by `genome_hash`** at **held-out** seeds (a transfer that only works at the seed that found it is a seed effect); gate 7 requires a **declared scale map** and a baseline parameter-matched at the new scale. |
| **The open questions, named rather than hidden** | Gate 5: does "competitive" mean the same win threshold Gate 4 applies to its harm guard, or non-inferiority against the *searched* arm? Gate 6: must every held-out seed clear the bar, or the block in aggregate? Gate 7: **may the scale map move depth** - scaling width keeps a topology and scaling depth replaces it, so that single choice decides which claim the gate tests. A test requires each unimplemented gate to declare its own. |
| **Pinned** | The plan's text is read back out of `docs/complete_implementation_plan.md` and each declaration must match it **verbatim**, so a plan change fails the suite rather than leaving this module describing something else; the chain must be gapless with each gate requiring its predecessor; exactly one gate is implemented; every unimplemented gate must name its open questions. |
| **Scoped** | It declares **reads**, not machinery, and says so itself: `declared_but_unimplemented()` returns `(5, 6, 7)` so "the chain is declared" and "the chain runs" cannot be conflated. What exists is gate 4's verdict and gate 5's transfer path (§5.63). |
| **Result** | **§5.64** in the report; **1578 tests** pass; 283/283 claims agree. |

## Two blind spots in the campaign launcher, and a failure report that was wrong about the cost (2026-09-21)

| | |
|---|---|
| **The failure** | `act-t4-session` ended **ERROR after 0.29 h with no kernel log and no output files at all**. That is a launch-time death, not a training one - a session that dies in training leaves a log. |
| **What the evidence shows** | The campaign log holds **two drivers' lines interleaved inside each other**: one wrote `session ended with ERROR after 0.29 h ...` *inside* the other's `the platform returned no ...`. So two drivers were driving one queue at once. The rule that should prevent it - "adopt a live session instead of pushing a second" - cannot: both drivers check `active_status`, both find nothing live, both push, and a push **replaces the running kernel's version**, which is exactly a logless early ERROR and the loss of everything that session had trained. Separately, `act-lab-code` was **republished at 12:31:51 while that session had it mounted**, and a session kernel's body and the mounted `src/` have to be one revision. That publish was this session's own TPU probe push. |
| **Fix 1 - the race** | `drive()` now takes an exclusive **per-queue `flock`** (in `artifacts/.locks/`, git-ignored, per kernel so one queue cannot block another). A second driver exits **6**, naming the holder, instead of racing it. Mutual exclusion is the only thing that can close a check-then-push window. |
| **Fix 2 - the publish** | The TPU probe **refuses to republish the code dataset while a campaign session is live** (exit 5). It is a measurement; the session is the critical path, so the probe waits. |
| **Fix 3 - the report** | The log-less failure text was **unconditional**: it asserted the platform's 8 h kill and quoted **"8.05 GPU-h burned"** for a 0.29 h launch failure. That is where the belief that this run cost eight GPU-hours came from - this run spent **0.27**. The note is now selected by measured elapsed time and quotes the hours actually spent, because a report that misstates the cost is worse than no report: it is the number an operator decides with. |
| **What it actually cost** | The failed session: **0.27 GPU-h**. The week stands at 23.8 h used / 6.2 h remaining, and the durable record is unchanged at **4 of 12 seeds** - what was lost is that session's training, not any published seed. |
| **Recovery, in order** | 1. Verify the launch path with a cheap one-shot rather than hoping: `--queue preflight` (~0.07 h) completed clean - fresh 2×T4, fp16 **4,885 tok/s**, pair 1.45 h, parameter assertion OK, so push → dataset → mount → accelerator is intact. 2. Relaunch the campaign under the new lock with a **fitted 5.5 h window** (`HOURS=5.5`), because weekly quota expires at the 2026-09-26 refresh and unused hours are lost; the window is a scheduling knob, not a protocol term (fewer whole pairs, deadline stop, in-progress snapshot, resume from the last completed seed). Session is RUNNING. |
| **Fix 4 - so it cannot be silent again** | `scripts/campaign_health.py` answers "is anything driving this?" from **durable remote state** - the published dataset (seeds complete at the *registered* step count, so a part-way snapshot does not read as done), the kernel's status, the weekly quota and the driver's lock - with the verdict as its exit code: **0 RUNNING / 1 STALLED / 2 WAITING_ON_QUOTA / 3 FAILED / 4 COMPLETE**. A watchdog process was the obvious answer and is the wrong one: it dies with the terminal exactly like the driver it watches. `STALLED` is the state both losses sat in, and the campaign launcher now prints the report at every queue boundary so its own log records where it got to. Documented in `T4_RUNBOOK.md` §6. |
| **Pinned** | `tests/test_kaggle_queue.py`: a second driver is refused with exit 6 *and* the lock is released afterwards, the lock names its holder, and a 0.29 h log-less failure is not described as the platform window while an 8.0 h one is. `tests/test_push_tpu_probe.py`: the publish is refused while a session is live and `ensure_code_dataset` is never reached. `tests/test_campaign_health.py`: the padded listing parses by field (reading the stripped line matches nothing and reads as "no seeds published"), a part-way seed is not complete, the completeness threshold comes from the registered config, and each of the five states gets its own verdict. |

## Gate 5 had no path to take, and the check that gives it one was wrong once before it was right (2026-09-21)

| | |
|---|---|
| **What was found** | §99's gate 5 reads *fresh architecture remains competitive*, and every published checkpoint already carries the architecture - `training.genome` sits beside `model` in the payload. But the only code that read a checkpoint's genome was the session's **resume** branch, which rebuilds a model from that genome and then immediately restores `model`, `optimizer`, RNG and scaler state over it. So an architecture could only travel *with* its training, and gate 5 had nothing to run: Gate 4 passing would have left the next gate blocked on missing machinery rather than on compute. |
| **The primitive added** | `genome_from_checkpoint` (weights-free read; raises on a payload that is not a version-4/5 session checkpoint, and on one whose `training.genome` is absent or `None`) and `fresh_init_from_checkpoint` (rebuilds from the genome, fresh init at the config seed, optimizer and token generator untouched) in `src/act/training/v0_session.py`. |
| **The property the read depends on** | The two paths have to be **distinguishable**, or a number produced by the weights-carrying path reads as a fresh-architecture number and nothing notices. Measured against a real two-step session (seed 16000, depth 2, 20,768 params): same parameter names, same shapes, **all 20 trained parameters differ** (largest `0.125166`), while a restore of `payload["model"]` reproduces the published weights exactly, tensor for tensor. |
| **And the first version of that check was wrong** | It asserted a different config seed moves *every* parameter. It does not: 15 of the 20 are seed-sensitive and the other five are the **normalization scales**, which initialize to exactly `1.0` and therefore cannot be moved by any seed. The assertion now requires the only seed-insensitive parameters to *be* the normalization scales - a weight matrix that stopped responding to the seed is a real defect and fails it, rather than being tolerated as "some parameters are not seed-dependent". |
| **Record and pins** | `scripts/gate5_rehearsal.py` writes `artifacts/study-summaries/gate5-transfer-rehearsal.json` (10 built-in checks, all holding, the negatives among them: a `None` genome and an unknown payload version both raise, two fresh inits at one seed agree parameter-for-parameter); `tests/test_gate5_transfer.py` recomputes both paths from a real checkpoint rather than reading the record; `scripts/verify_record.py` gained **9 claims** on it, and both records' stated audit coverage was updated to the new counts. |
| **Scoped** | This is **not** a Gate-5 result. The genome transferred is a two-step dev-shape one, not a discovered architecture, and nothing here says any architecture is competitive. What §5.63 establishes is that the read exists and that violating it fails a test; the gate's own comparison and tolerance are declared with the artifact they will be applied to. Gate 6 inherits the primitive; gate 7 additionally needs capacities carried at another scale. |
| **Result** | **1534 tests pass** · 263/263 quick · 283/283 full claims agree; §5.63 in `RESEARCH_REPORT.md`; summary `gate5-transfer-rehearsal.json`. |

## The campaign's driver was dead and nothing was watching it (2026-09-21)

| | |
|---|---|
| **What was found** | The Gate-4 campaign was found with **no driver process alive** and no `artifacts/datasets/` in the tree, which reads as "nothing published" and is wrong: the durable copy is the Kaggle dataset, not the working tree (the run root is deleted at session end, `T4_RUNBOOK.md`). The published dataset says the campaign is **4 of 12 seeds in** on the primary queue (`seed-15000`-`seed-15003`, each with both arms at step 6000, published 08:04-08:08 today), and a session pushed at 12:14 today was **still live on Kaggle with nobody collecting it**. |
| **Why it stopped** | The driver sequences **one queue per invocation** and its `--wait-for-quota` sleep lives in that process, so any terminal or session that ends takes the campaign with it. The runbook's answer is an operator re-running the command; the local state shows what that costs. |
| **The fix** | `scripts/run_gate4_campaign.sh` chains `session -> control -> null -> verdict` in one process, re-running on exit 2 (budget ran out with seeds missing) and 3 (quota short with an unreadable `refreshAt`), and stopping loudly on 1 and 4 because those mean the cause has to be fixed rather than retried. The `verdict` queue is handled separately: its exit code **is** the decision (0 PASS / 1 FAIL / 2 qualification), so a FAIL is an answer, not a fault, and only 3 (incomplete block) stops. |
| **Recovery** | Re-launched; it **adopted the live session** instead of pushing a second one. |
| **TPU, measured before it is trusted** | Kaggle's TPU quota is **separate** from the GPU quota (20 TPU-h/week beside 30 GPU-h), and a v3-8's eight cores can carry eight *independent seeds* - the campaign's arms and seeds are independent, so a TPU can be used embarrassingly parallel with no collectives. But `src/act` has **no XLA anywhere**, and the two ways a port can lose are per-core throughput and rebuild cost (a mutation event and every probed candidate build a new model, i.e. a new graph, i.e. a recompile). `scripts/tpu_probe.py` measures both at the declared 285M shape - per-core throughput fp32 and bf16 with compile time separated, rebuild cost, per-core memory, and one seed per core - and `scripts/push_tpu_probe.py` runs it as `act-tpu-probe` (TPU on, GPU off, so it cannot silently answer on the wrong accelerator). No port starts until those numbers exist. |
| **And the decision was registered *before* the numbers** | A go/no-go written after the measurement is a rationalisation of it, so the rule is code: the 8 independent seeds must reach **2x the T4 pair's aggregate rate** (~10,167 tokens/s), *and* recompiles must stay under **10% of the per-core training time** (3 mutation events x (1 mutation + 4 probed candidates) = 15 rebuilds per arm, spread over 8 cores). `scripts/tpu_probe.py --evaluate <record>` applies it with no TPU present and exits **0 PORT / 2 PORT_FIXED_GRAPH_ONLY / 1 DO-NOT-PORT / 3 INCONCLUSIVE**. The fixed-graph branch is declared with the rule rather than discovered later, because recompile cost is a property of the session path and not of an arm that trains one sampled architecture to a horizon. An incomplete record is `INCONCLUSIVE`, never a pass - a probe that skips its eight-worker pass must not read as "no problem found". Pinned by `tests/test_tpu_decision.py`. |
| **Result** | Campaign driver recovered and adopting the live session; TPU probe pushed and queued; no campaign result is claimed by either. |

## Off-policy: the crossing survives the other candidate policy, its decomposition does not (2026-09-21)

| | |
|---|---|
| **The gap the arm was placed at** | Every width dose and every depth rung in this family runs the **prefix** candidate policy - invisible while the arms reported gates and prices, and load-bearing the moment they claim a *mechanism*. 5.12's 2x2 is the only policy evidence and it covers depths 1 and 4 at the registered width, i.e. exactly not the dose where the gate crosses. |
| **The arm** | One fresh cell: 5.60's crossing dose (depth 3, width 68, 173,672 params) with `probe_candidate_policy: spread` and nothing else changed. |
| **Registered outcome** | **Branch 1 - P1 and both declared predictions pass: the crossing is policy-robust.** P1: `387/576` against the bar of `313`, mean `-0.126482`, p `5.63e-17`. The ratio is **1.64** (edge `-0.323944`, pool `+0.197462`) - edge-dominated again, with the pool term still positive. And the value is `0.001510` from the prefix cell's reading, inside the declared `0.10` band by a factor of 66. |
| **On the level the two policies are indistinguishable** | Rank-sum on the per-seed value deltas: p `0.472` in one direction and p `0.528` in the other. So which dose opens the gate, and that it is edge-dominated with a positive pool, hold under both policies. |
| **But the components move - this is the finding** | The edge shifts `+0.035826` and the pool term `-0.034316` across the policy change (prefix edge larger at p `0.032`, spread pool smaller at p `0.041`): opposite directions, nearly equal magnitudes, cancelling to `+0.001510` in the contrast. So the **two-term split is a description of the balance at a dose, not an invariant of the mechanism** - a caveat every decomposition from 5.50 to 5.61 inherits, since all of them were read off prefix-policy cells. Reporting only the agreement would make the split look invariant; reporting only the moving parts would make the crossing look policy-specific. The report states both. |
| **Absolute levels move far more than the contrast** | The spread arms sit `0.31`-`0.35` nats worse in absolute loss with the arm order unchanged (`adaptive < null < random` in both). So the policy changes the level the comparison is taken at - and the comparison survives it. |
| **Pinned** | `tests/test_width_dose3_spread_summary.py` recomputes the cell, both declared predictions, **both rank-sums in both directions** for the value, the significant component shifts with their signs, the cancellation arithmetic and the absolute-level shifts; `tests/test_width_dose3_spread.py` pins the one-gene design against the prefix cell, the fresh block, the single-cell bar and the branches. `tests/test_report_navigation.py` requires the out-of-policy sentence and the caveat it produced. |
| **Mutants** | Four, all biting after one fix: replacing the caveat with "the mechanism itself" fails **1**; a perturbed edge p fails **1**; calling the agreement "marginally separated" fails **1** (my first version of that fragment was unpinned - now asserted); the earlier round's mutants still bite. |
| **Result** | **1523 tests pass** · 254/254 quick · 274/274 full claims agree; registration and frozen revision `969a251`; summary `width-d3-dose3-spread-576.json` (`study_778e4fb4004254c7`). |

## The crossing rule: my own staging claim was half wrong, and the committed ladders say so (2026-09-21)

| | |
|---|---|
| **What was wrong** | Sections 5.60 and 5.61 read the crossing's mechanism as a *division between the sides of the onset* - "below it the controller's edge crosses the gate, above it the pool term does". The onset cell itself is a counterexample: `depth4-prefix` is open (225/384) with a **positive** pool term (`+0.0701`). The claim was read off the *dose steps* above 3.80x, where the pool term happens to be negative, and generalised to the depth division it does not track. |
| **What the committed cells actually show** | Value is `(adaptive - random) + (random - null)` on the same seeds, so write the ratio: 0.96, 0.88 and 0.80 (all closed) below 1; **1.0959** at the 4.01x near miss (mean crossed, bar not); **1.55** (5.66x), **1.53** (the onset cell itself) and **1.37** (1.34x) all open. **No cell with a positive pool term and a ratio below 1 opens the gate; every cell at 1.37 or above does**, and the pool term's own sign is irrelevant once it turns negative (the 3.80x-14.82x doses and every depth rung from 6 up are open with a negative one). |
| **The corrected statement** | The crossing is located in the **ratio**, bracketed by construction to `[1.10, 1.37]`, and **what tracks the mechanism is the dose, not the side of the onset**: on either ladder the small-dose end is edge-dominated and the large-dose end is pool-dominated. What made the depth axis look pool-only is that every rung at the registered width except the onset cell sits on the far side of that transition. |
| **And it explains the near miss** | At 4.01x the ratio has passed 1 (so the *mean* has crossed, `-0.028562`) while the *win rate* has not (310/576, five short) - the boundary between the mean's crossing and the bar's is a distribution-shape effect, and the ratio table is where it becomes visible rather than surprising. |
| **Status, stated honestly** | This is a **descriptive reading of committed cells made after both arms**, not a prediction either arm made - the interval was bracketed by the doses rather than derived from a model, and the report says so in the paragraph itself. The test it invites is a dose placed to land inside the interval on a ladder that has not been swept. |
| **Pinned** | `tests/test_report_navigation.py` recomputes every ratio and every gate verdict from the summaries, asserts the closed set's maximum (`1.0959`) and the open set's minimum (`1.37`) so the bracket cannot silently widen, and requires the corrected framing and the explicit statement that it corrects 5.60/5.61. Mutants: reverting the framing to the onset split fails **1**; dropping the boundary bracket fails **1**. |
| **Result** | 254/254 quick · 274/274 full claims agree; the correction is in §10.1, with a scoped pointer from §10.3. |

## The second bisection: both declared brackets hit, and the gate's price tightens to 4.0x-7.5x (2026-09-21)

| | |
|---|---|
| **Why a second dose** | 5.60 left a 1.97x window (88,032-173,672 params) and, more usefully, a **staged mechanism**: the gate opens at 5.66x while the pool term is still positive, crossed there by the controller's edge at its ladder maximum. Nothing between 2.87x and 5.66x was measured, so one dose could not say whether that maximum is approached gradually or reached abruptly. |
| **The dose** | Width 57 - the new window's geometric midpoint (123,006 params at depth 3, 4.01x; 163,134 at depth 4, 5.32x), two fresh cells, one gene each. |
| **Registered outcome** | **Branch 1 - P1 fails with P2+P3 passing: the window narrows to (123,006, 173,672].** P2: `551/576`, mean `-0.418214`, p `1.65e-130`. P3: rank-sum U `88052.0`, p `1.55e-43`. |
| **The gate prediction obtained, and it is a near miss to state as one** | P1: `310/576` against the bar of `315` - **five seeds short** at `alpha/3` (and three short at the sibling arms' `alpha/2`), with mean `-0.028562` and p `0.0365`. So the gate is still shut at 4.01x *at the declared level*, while the mean delta has **crossed** (from `+0.061466` one dose lower). At this dose the advantage is large on some seeds and absent on the median one - which is not the same claim as "inert", and the report says both halves. |
| **Both declared brackets hit** | The depth-3 edge lies between its 2.87x and 5.66x readings (`-0.326291`: below the 2.87x reading at p `7.7e-05`, above the 5.66x reading at p `0.051`), and the pool term is positive and off its peak (`+0.297729`: below the 2.87x peak at p `0.275`, above the 5.66x reading at p `2.5e-03`). So the staging is a **gradient, not a step**. |
| **And the turning points are now located** | **The pool term peaks at 2.87x and the controller's edge peaks at 5.66x**, with the gate opening in the window between them - where the edge is still rising and the pool has come off its peak but is still positive. The honest qualification: neither local slope inside the brackets is sharp (p `0.275` and p `0.051`), so what is established is that both terms sit between their neighbours' readings, not that either turned sharply. |
| **The above-gate asymmetry is now four doses deep** | 3.80x -> 5.32x: edge `-0.194518` -> `-0.210845` (p `0.180`), pool `-0.015155` -> `-0.207369` (p `2.6e-28`). 5.32x -> 7.52x: edge p `0.453`, pool p `1.5e-63`. Four consecutive above-gate doses, the edge never moving and the pool moving decisively every time - 5.50's decomposition replicated rather than restated. |
| **The tightened price** | Width bracket (123,006, 173,672] over the depth bracket (23,280, 30,672] gives **4.0x-7.5x**: even the most capacity-favourable pairing puts the same gate at four times the price along width. The synthesis now carries that bound; 5.60's own outcome keeps the `2.9x-7.5x` interval it measured, scoped to the window it left. |
| **Pinned** | `tests/test_width_dose4_summary.py` recomputes both cells, both brackets **and their qualifications** (asserting the p-values that keep "between" distinct from "turned"), the two turning points, the four-deep above-gate asymmetry, the tightened ratio and the near-miss arithmetic (five seeds, mean crossed). `tests/test_width_dose4.py` derives the cells, the midpoint, the fresh blocks, the bar, the brackets' premises from the committed neighbours, and the branch table. `tests/test_report_navigation.py` now requires the tightened bound and the turning-point claim. |
| **Mutants** | Nine, all biting after one fix: dropping the near-miss fails **1**; a drifted outcome row fails **1**; overstating the pool bracket's qualification fails **1** (my first attempt was unpinned - the qualification is now a pinned claim); dropping the edge's marginality qualifier fails **1**; reverting the ratio fails **1**; swapping the turning points fails **1**; widening the declared bracket fails **1**; changing the dose's width fails **2**. |
| **Result** | **1510 tests pass** · 254/254 quick · 274/274 full claims agree; registration and frozen revision `04dc4b3`; summaries `width-d3-dose4-576.json` (`study_6e94b34cf411f9f1`) and `width-d4-dose4-576.json` (`study_0cb4e4bbcb3d65e9`). |

## The width bisection: the gate opens at 5.66x, at 2.9x-7.5x the price depth charges (2026-09-21)

| | |
|---|---|
| **The number the closure left open** | 10.3 closed the covariate programme on "only the depth axis moves the gate", and that closure carried a number it never closed: 5.49 measured width 48 at depth 3 (2.87x the above-onset cell, 88,032 params) as *below* the gate and 5.50 measured width 96 (11.15x, 341,952) as above it - a **3.9x window**, the widest unresolved quantity in the programme, and the difference between "depth is the gate" and "depth is the cheapest route to the same capacity threshold". |
| **The bisection** | Width 68, the window's **geometric midpoint in parameter space** (173,672 params at depth 3, 5.66x; 230,520 at depth 4, 7.52x), so one dose halves the window either way. Two fresh cells, one gene each, the registered everything else. |
| **Registered outcome** | **Branch 1 - all three primaries pass: the window narrows to (2.87x, 5.66x).** P1: `389/576` against the bar of `315`, mean `-0.127992`, p `1.31e-17`. P2: `575/576`, mean `-0.725298`, p `2.33e-171`. P3: rank-sum U `20954.0`, p `1.27e-145`. So **capacity does buy the gate at this dose**, between 88,032 and 173,672 parameters at the below-onset depth. |
| **The declared mechanism prediction missed - and the miss is the finding** | The registration declared the flip in the pool term was *above* the bisection: still positive at 5.66x **and larger** than at 2.87x. It is still positive (`+0.231778`) but significantly **smaller** (`-0.073489`, p `1.35e-04`) - the term has already **peaked at 2.87x**. The gate opened anyway, which refutes the mechanism as registered ("the gate opens when the pool term flips") and is exactly what two cells could not show. The verdict prints the state a miss lands in rather than only `False`. |
| **What replaced it: the gate opens in two stages** | At 5.66x the gate is open **while the pool term is still positive** - what crosses it is the *controller's own edge*, at its ladder maximum (`-0.359770`, having run `-0.102`, `-0.131`, `-0.244`); by 11.15x the edge has fallen back (`-0.310646`) and the pool term has crossed (`-0.585988`). One depth up the order is reversed and only the pool moves (edge `-0.194518` -> `-0.199203`, p `0.199`; pool p `4.6e-119`), which is why 5.50's two-cell read (3.80x/14.82x) attributed the crossing to the pool: it sits entirely on the above-onset side of that division. |
| **The price, both axes** | Along depth the gate opens between **23,280** and **30,672** parameters (a 1.32x step); along width at the below-onset depth, between **88,032** and **173,672**. So the ratio is bounded: **the same gate costs 2.9x-7.5x more along width than along depth**, the lower bound being the most capacity-favourable pairing. Sharpest form, needing no window: depth 9 (5.59) opens the gate at **67,632** parameters, **23% fewer** than the width dose that does not. |
| **A hypothesis named, not measured** | Every arm before this one found the controller's own edge flat or irrelevant. The bisection found the one regime where the edge is the term that crosses a threshold - wide-and-shallow, below the onset - so a transfer policy has one measured place where it might eventually matter. The report says hypothesis, and says nothing else has reproduced it. |
| **One defect of my own, caught before the run** | The registration's window table carried two **invented** parameter counts - the width-28 depth-3 cell written as 27,456 / 0.90x against the committed 31,192 / 1.02x, which 5.48's own table already stated. The column is now *computed from the profiles* by a test, so a transcribed row cannot be invented again. |
| **Pinned** | `tests/test_width_dose3_summary.py` recomputes both cells, the third primary, the bounded price ratio, the **missed** prediction (asserting the pool term is positive *and* smaller, and that the declared direction is still on the record), the two-stage mechanism, and the report's own table rows and prose. `tests/test_width_dose3.py` derives the whole window column from the profiles, checks the fresh blocks, the bar and the branch table. `tests/test_report_navigation.py` now requires the cross-axis claims §10.1/§10.3/§10.5 gained. |
| **Mutants** | Eight, all biting after one fix: reverting the prediction to a hit fails **1**; quoting the price as unbounded fails **1** (and as a single ratio fails **1**); a drifted outcome row fails **1**; a perturbed pool p fails **1**; reverting §10.1's flip location fails **1**; dropping §10.5's below-onset qualifier fails **1** (my first mutant was a no-op across a line break - re-run newline-robustly); changing a profile's width fails **2**; flipping the declared direction constant fails **1**. |
| **Result** | **1493 tests pass** · 254/254 quick · 274/274 full claims agree; registration `698a866`, frozen at the same revision; summaries `width-d3-dose3-576.json` (`study_e558d97ad872391b`) and `width-d4-dose3-576.json` (`study_ff5dfc7cb5cc6d79`). |

## The sixth rung: the third forecast missed into the seam the first two bands left (2026-09-21)

| | |
|---|---|
| **Why a sixth point** | Two registered forecasts had missed from **opposite sides** of the same series - the flat model at the fourth increment (`-0.0635` against `(-0.047, -0.033)`) and the accelerating model at the fifth (`-0.0541` against `(-0.069, -0.059)`). Two misses that bracket the axis leave **three** live readings rather than two, so this arm declared the middle one (the mild rise the last two increments showed) and constructed its band so the two it excluded are a miss. |
| **Registered outcome** | **Branch 1 - P1+P2 PASS.** P1: `559/576` against the bar of `313`, mean `-0.283891`, p `7.81e-142`. P2: rank-sum U `135354.0`, one-sided p `3.19e-08`. **The slope continues**, so the ladder is monotone across all six rungs above the onset. |
| **And the third forecast missed too - into the crack** | Declared band `(-0.069, -0.049)`; measurement **`-0.048724`** - above that band's shallow edge by `0.000276`, and *also* outside the flat band (`-0.047, -0.033`), on the same side. So it landed in the **0.002-wide seam between the two bands that preceded it**: neither the mild rise it declared nor the flat level the first model declared. `--depth9-verdict` prints that sentence on a miss rather than only reporting `False`, so the third miss reads as a location. |
| **What three misses leave standing** | The increments ending at depths 4-9 run `0.0418`, `0.0382`, `0.0422`, `0.0635`, `0.0541`, `0.0487`. The first three are flat at `0.0407`; the last three decline **monotonically** and the gap above that flat level shrinks `0.0228 -> 0.0134 -> 0.0080` - a factor of `0.59` and `0.60` per rung. So the axis's excursion above the flat regime is **decaying back toward it geometrically**, which is the first reading in this family that predicts the near future rather than describing the past: on the measured ratio, the increment ending at depth 10 falls *inside* the flat band. Stated as a prediction, not asserted - it is what the next registration can be built on instead of a fresh guess. |
| **The edge prediction hit for a fourth rung - and 5.58's marginal reading died** | `|−0.082532 − (−0.070113)| = 0.012419`, inside the declared `0.020`. That matters more than a fourth hit usually would, because the one reading that had suggested the edge moves is gone: 5.58 reported the decline reaching p `0.0244` over 3 -> 8 **as descriptive and unregistered**; over 3 -> 9 the same contrast reads p `0.121`. One more rung, no reproduction - and the discipline that refused to promote a marginal reading into the synthesis is what kept it out. |
| **The decomposition is now six rungs deep** | 3 -> 9: the controller's own edge `-0.102151` -> `-0.082532`, rank-sum p `0.121` (reversed `0.879`, flat); the churn term `-0.106794` -> `+0.201358`, p `3.7e-104`. Of the `-0.288534` improvement, the churn term carries `-0.308153` and the edge `+0.019619` **against**. The newest step decomposes the same way: churn `-0.036305` (p `3.2e-06`, 74.5%), edge `-0.012419` (p `0.831`, flat). Six added layers, and again none of the gain is selection skill. |
| **The collapse keeps growing** | `null - random` runs `-0.099659`, `-0.106794`, `-0.070072`, `-0.029226`, `+0.028013`, `+0.094421`, `+0.165053`, `+0.201358`; depth 9 repeats depth 6's arm order (`8.5254` < `8.6079` < `8.8093`), with the null arm 0.284 nats behind the adaptive one. |
| **Scrutiny before write-up** | Built-in checks 576/576 (mutations 3/3/0, zero null search FLOPs), re-derived from the shard files independently of the summary's own `built_in_checks` field; three replays of committed seeds (95404, 95704, 95944) exact at `OMP_NUM_THREADS=1`; both primaries, the increment, both decomposition rank-sums and the algebraic identity computed before any entered prose. The band and both predictions were committed at `1816e34`, before the first shard ran. |
| **Two of my own numbers were wrong first time** | The edge rank-sums went into the report *backwards* - the family prints the shallow-to-deep p with the reverse in parentheses, and I quoted the reverse as the value for both the 3 -> 9 ladder row and the newest step. `tests/test_depth9_summary.py` caught both on its first run, which is the whole reason those p-values are recomputed from the summaries in a test rather than transcribed. |
| **Pinned** | `tests/test_depth9_summary.py` recomputes the cell, both predictions (asserting **each** band excludes the measurement and that the seam between them is 0.002 wide), the eight-cell ladder, the increments and their geometric decay, the arm ordering, both decompositions and the report's own table row. `tests/test_depth9.py` pins the design, the block, the pool across depths 3-9 and the four branches; `tests/test_report_navigation.py` now requires the **decaying-excursion** text, the third miss and the 0.0244 -> 0.121 reversal. |
| **Result** | **1475 tests pass** · 254/254 quick · 274/274 full claims agree; registration and frozen revision `1816e34`; summary `depth9-dose-576.json` (`study_6f2e9b7edbd5024a`). |

## The fifth rung: both registered forecasts missed, one from each side (2026-09-21)

| | |
|---|---|
| **Why another rung** | 5.57 asked the constant-gradient model for a number and the number came back 1.5x too small (`-0.063548` against `(-0.047, -0.033)`). One point past a flat series cannot separate an **acceleration** from a **wobble**, so this arm re-asked the question with the fourth increment itself as the model - a choice between two readings, with the flat reading it was declared against sitting a full band away. |
| **Registered outcome** | **Branch 1 - P1+P2 PASS.** P1: `544/576` against the bar of `313`, mean `-0.235166`, p `1.46e-121`. P2: rank-sum U `131833.0`, one-sided p `8.12e-10`. **The slope continues**, so the ladder is monotone across all five rungs above the onset. |
| **And the second forecast missed too** | Declared band `(-0.069, -0.059)`; measurement **`-0.054124`** - a miss on the *low* side, where 5.57's was a miss on the high side. The six increments are now `-0.0162`, `-0.0418`, `-0.0382`, `-0.0422`, `-0.0635`, `-0.0541`: three steps at the flat `~0.04` level, then one well above it and one above that level but below the fourth. So the axis is **neither constant nor accelerating without bound** - it is mildly rising, and each model is refuted by the rung the other one fit. The edge prediction hit for the third rung in a row (`-0.070113` against `-0.086622`, a move of `+0.0165` inside the declared `0.020`). |
| **The decomposition, now five rungs deep** | 3 -> 8: the controller's own edge `-0.102151` -> `-0.070113`, rank-sum p **`0.0244`** (reversed `0.976`); the churn term `-0.106794` -> `+0.165053`, p `1.5e-91`. Of the `-0.239810` improvement, the churn term carries `-0.271848` and the edge `+0.032038` **against**. The newest step decomposes the same way: churn `-0.070633` (p `4.9e-14`), edge `+0.016509` (p `0.162`). |
| **One genuinely new reading** | Over 3 -> 7 the edge's decline was p `0.139`; over 3 -> 8 it reaches **p `0.0244`** - the first time this family's `edge + churn = value` split has the edge moving in the against direction at any significance level. It is a cross-cell rank-sum with no multiplicity correction and a hypothesised rather than registered direction, so it is recorded as a descriptive reading, not a test - and it makes the central point sharper: the deeper the substrate, the less of its value the controller supplies, and past depth 8 the *edge itself* is measurably shrinking. |
| **The collapse keeps accelerating** | `null - random` runs `-0.099659`, `-0.106794`, `-0.070072`, `-0.029226`, `+0.028013`, `+0.094421`, `+0.165053`: the do-nothing arm's disadvantage grows **1.75x** across the newest step, and depth 8 is `9.0747` (adaptive) < `9.1448` (random) < `9.3099` (null), with the null arm 0.235 nats behind. |
| **Scrutiny before write-up** | Built-in checks 576/576 (mutations 3/3/0, zero null search FLOPs); three replays of committed seeds (94404, 94704, 94944) exact at `OMP_NUM_THREADS=1`; both primaries, the increment, both decompositions and the algebraic identity computed before any entered prose. The band and both predictions were committed at `a21c4a9`, before the first shard ran. |
| **Pinned** | `tests/test_depth8_summary.py` recomputes the cell, both predictions (asserting **each** band excludes the measurement, so neither reading can quietly become a hit), the seven-rung ladder, the increments and their crossing, the arm ordering, both decompositions and the report's own table row. `tests/test_depth8.py` pins the design, the block, the pool across depths 3-8 and the four branches; `tests/test_report_navigation.py` now reads the gradient claim off the increments and requires the *mild-rise* text. Mutants: widening the new band to swallow the measurement fails **1**; widening 5.57's band so it would have caught the fifth fails **1**; a drifted outcome row fails **1**; a perturbed churn contribution in the prose fails **1**; and §10.1 calling the gradient constant again fails **1**. |
| **Result** | Summary `depth8-dose-576.json` (`study_fc0bd21e6e30b19f`); registration `a21c4a9`, frozen at the same revision. |

## The fourth rung: the declared forecast missed, and the miss is the result (2026-09-21)

| | |
|---|---|
| **Why a fourth point** | 5.56 turned the depth axis from a direction into a *model*: three increments of `-0.0418`, `-0.0382`, `-0.0422` (about `0.04` per layer, no decay) with a flat controller edge. A model that predicts a number is testable in a way a direction is not, so this arm was registered against the model's **own numbers** - the increment in `(-0.047, -0.033)`, and the edge within `0.020` of `-0.089482` - rather than against one more "deeper is better". |
| **Registered outcome** | **Branch 1 - P1+P2 PASS.** P1: `503/576` against the bar of `313`, mean `-0.181042`, p `2.92e-80`. P2: rank-sum U `128927.0`, one-sided p `2.95e-11` (bar `0.025`). **The slope continues**, so the ladder is monotone across all four rungs above the onset. |
| **The forecast was half right** | The **increment missed**: `-0.063548` against a band of `(-0.047, -0.033)`, i.e. **1.5x every previous step of the upper ladder** (the four increments are `-0.0418`, `-0.0382`, `-0.0422`, `-0.0635`). The **edge hit**: `-0.086622` against a declared `0.020` around `-0.089482`, a move of `+0.0029`. So the model's *mechanism* survived a second rung and its *magnitude* did not. |
| **What the miss corrects** | 5.56 read three flat increments as "the slope is a constant, not a decaying one". A fourth point shows the flatness was not yet the *shape*: the axis buys a flat `~0.04` per layer for the steps ending at depths 4, 5 and 6, and `0.0635` for the step ending at 7. The constant-gradient reading is now bounded to the three rungs that produced it, the axis is **super-linear at the fourth**, and the model understated it rather than overstating it. |
| **The decomposition holds one rung further up, with a wider margin** | Across 6 -> 7 the controller's own edge is flat (rank-sum p `0.663`) while the churn term moves decisively (p `2.6e-11`); of the `-0.063548` the newest layer buys, the churn term contributes `-0.066408` and the edge `+0.002860` **against**. Over the whole upper ladder 3 -> 7: edge p `0.139` (flat to slightly worse), churn p `9.0e-62`, and of the `-0.185686` improvement the churn term carries `-0.201215`. Four added layers, and again none of the gain is selection skill. |
| **The collapse accelerates with it** | `null - random` runs `-0.099659`, `-0.106794`, `-0.070072`, `-0.029226`, `+0.028013`, `+0.094421`: the do-nothing arm's disadvantage grows **3.4x** across the newest step alone, and depth 7 repeats depth 6's arm order (`9.7065` < `9.7932` < `9.8876`) at 1.5x the separation. |
| **Scrutiny before write-up** | Built-in checks 576/576 (mutations 3/3/0, zero null search FLOPs); three replays of committed seeds (93404, 93704, 93944) exact at `OMP_NUM_THREADS=1`; both primaries, the increment, both decomposition rank-sums and the algebraic identity computed before any entered prose. The bands and both predictions were committed at `80a28a2`/`f6f5a08`, before the first shard ran. |
| **Pinned** | `tests/test_depth7_summary.py` recomputes the cell, both predictions (asserting the band does *not* contain the measurement, so the record cannot quietly become a hit), the six-rung ladder, the increments, the arm ordering, both decompositions and the report's own table row. `tests/test_report_navigation.py` now also reads the gradient claim off the summaries: three flat steps and a fourth more than 1.4x them. |
| **Result** | **1426 tests pass** · 254/254 quick · 274/274 full claims agree; pre-registration `80a28a2`, frozen revision `f6f5a08`; summary `depth7-dose-576.json` (`study_1527be70abc860bb`). |

## The architecture record's live surfaces had drifted from its own artifacts (2026-09-21)

| | |
|---|---|
| **What was found** | `artifacts/ARCH_SEARCH_FINDINGS.md` is two documents in one: the registered arms' bodies are history, and the header, §8 deliverables table and §10 cost table are **status**. Four status surfaces had drifted: (1) the header still listed the horizon question as open - "pre-registered in its own NOTES.md and **runs on Kaggle T4 rather than the local card**" - while §3 and §9 of the same document record that arm's result (depth `+0.0209` at 10x the horizon, p `0.003086`); (2) the deliverables row called the head-paired confirmation **running** while `arch-zoo-100m-v4-head-paired-confirm/decision.md` sat committed beside it (24 runs, 3.511 GPU-h, spread 0.2412 nats); (3) the decision row quoted the head-paired spread as `0.2126` - which is the parent's *own three-seed* figure, the one `verify_record.py` recomputes - against a committed `decision.md` that is the pooled six-seed render at `0.2221`, so a reader comparing the row with the artifact saw a disagreement that was really an unlabelled subset; (4) the audit-coverage row said `verify_record.py` recomputes **119** load-bearing numbers, against the **274** it runs. |
| **Why nothing caught it** | Its numbers are audited - `verify_record.py` recomputes 274 claims and passes - but a *tense* is not a number. "running", "unresolved" and a coverage count are all claims about a state, and the one row that quotes a real figure (0.2126) quotes a correct figure for a data set the row never names. |
| **What replaced them** | The header now says the horizon arm **has run and landed**, with the measured `+0.0209` (p `0.003086`, floor `0.001543`) and the two things it did *not* settle (the 1.38x-per-doubling rate against 1.65x, and the winner changing over 10x at Spearman 0.90). The deliverables row says complete, at `3.511 GPU-h`. The decision row names both spreads and the resolution the confirmation produced (heads `-0.0553` at H=896 vs `+0.0084` at H=512). The coverage row states 274, with 254 on the `--quick` path the suite runs. |
| **Pinned both ways** | `tests/test_arch_findings_state.py` reads each surface against the artifact that decides it: the header's quoted depth effect must be the one the arm's `decision.md` records and must not use the plan tense; the deliverables row must say complete *because* the record exists (and must quote the §10 table's own cost, so the two tables cannot drift); the decision row must quote the spreads the committed `decision.md` files state. `tests/test_verify_record.py` pins the coverage count against the audit it describes, on both paths. |
| **Mutants** | Seven, all biting: dropping the header's measured effect fails **1**; reverting the header to the plan tense fails **1**; the row back to "running" fails **1**; the row's cost drifting from the §10 table fails **1**; perturbing the pooled spread fails **1**; the count reverting to `119` fails **1**; and only the `--quick` count drifting fails **1**. |
| **Result** | The architecture record's status half is now checked by the suite rather than by whoever last read it. |

## A test module had been uncollectable, so 15 of its tests had never run (2026-09-21)

| | |
|---|---|
| **What was found** | `tests/test_space_typicality.py` imported its subject as `from scripts.space_typicality import ...`. `scripts/` is not a package and the repository root is not on `sys.path` under a plain `pytest` run, so the module raised `ModuleNotFoundError` **at collection** - the suite reported one error it could not attribute to any test, and every test in the file was silently absent from every run, including the ones recorded as green. |
| **Why it survived** | A collection error is not a failure count and not a list of failing tests, so a summary line reads `N passed, 1 error` and the error is easy to read past - and no other test in the suite imports `scripts.*`, so nothing else would have hit it. The same blind spot as a stale status row: nothing that reports on itself. |
| **The fix** | The file now inserts `scripts/` on `sys.path` and imports the module by name, which is the convention every other test in this suite uses (`tests/test_depth6_summary.py`, `tests/test_depth7_summary.py`, `tests/test_verify_record.py`). |
| **Result** | **15 tests recovered and passing**; the suite now reports `1426 passed` with no collection error. |

## The synthesis carried claims the 5.48-5.56 arms had already falsified (2026-09-21)

| | |
|---|---|
| **What was found** | Sweeping every *current-state* surface in the report after correcting two synthesis claims turned up four more, and they were the kind nothing else can see: (1) §10.1's opening axis said "on the toy path, churn harm (null < random) holds at **every measured depth**" - false at depth 6, where the collapse is `+0.028013`; (2) the same axis stopped the depth ladder at four rungs, quoting `+0.0508 > +0.0208 > +0.0046 > -0.0371` as the whole dose-response; (3) §10.3 listed the depth-3/4 boundary as an open question - it was closed by 5.48-5.56 - and §10.2 said Gate 4 was "prepared, amended, and **unspent**" while the campaign was running; (4) §5's study-index table still carried `execution planned ... not yet run` for the real-scale row, and 5.14's heading still read `(designed, not yet run)`. |
| **Why nothing caught it** | A universal statement is not a number. `scripts/verify_record.py` audits claims against artifacts and the summary pins check recorded cells, so "holds at every depth" and "not yet run" were both invisible - the same blind spot as the stale header, one layer down. |
| **The line that decides what to edit** | The registered `### 5.x` bodies are **history and stay verbatim** - including the 5.13 consequence that calls the Gate 4 block "unspent" and 5.10's motivation that calls the depth-4 pass "the last pass not yet faced with a do-nothing control", both true when written. What is *not* history is the heading parenthetical (the same status marker that carries `(RAN - summary ...)`), the §5 index's status column, §4's conclusions and §10's synthesis. Those are the surfaces that get corrected. |
| **What replaced them** | §10.1 now says churn harm holds "up to 5" and does not hold at depth 6; its depth axis runs the full ladder with the constant gradient and the arm-order flip. §10.3 opens by declaring **(ii) closed and (i) in flight**, so the long narrative that follows reads as the closing rather than as a list of open questions. §10.2 says Gate 4 is *running*. The §5 index row says **IN FLIGHT** and points at the runbook; 5.14's heading says the clause is implemented and running. |
| **And the report's live numbers can no longer rot into false ones** | A campaign's progress is a snapshot, so every progress statement now says "at this writing" - which the new test enforces. It caught a fourth inconsistency on its first run: §10.2's count lacked the marker while the header's had it. |
| **Pinned** | `tests/test_report_navigation.py` now also reads §10.1's claims against the committed summaries: the ladder must be monotone in the records, the collapse sign must be negative at rungs 2-5 and positive at 6, and the index row must say `IN FLIGHT` and name the runbook. Mutants: restoring the universal churn claim fails **1**; stopping the ladder at four rungs fails **1**; a status row reverting to "not yet run" fails **1**; dropping a snapshot marker fails **1** - and two earlier versions of these checks did **not** bite (a line-wise scan is defeated by wrapping), which is why they now run on whitespace-normalized text. |
| **Result** | **1398 tests pass**; the report's synthesis is now data-checked in the two places where being wrong would mislead a reader the most. |

## The confirmation rung: the depth slope is constant, the churn regime flips, and the gain was never the controller's (2026-09-21)

| | |
|---|---|
| **Why a third point** | 5.55 left the depth slope as **two** points above the gate, and two points fit a rising line or a line already turning over between them. One rung further up the same one-gene dose (`depth` 5 -> 6, 45,456 total / 16,656 active parameters) either confirms the slope or bounds it - a plateau makes 5.55's claim a one-step statement, a reversal bounds monotonicity to depths 2..5. Both outcomes were pre-registered as informative for that reason. |
| **The design's one cleanliness check** | The candidate pool must not move with depth, or the arm changes the substrate *and* the policy's choice set at once. Pinned over the whole ladder: at `probe_candidate_cap` 4 the pool is `[grow L0, prune L0E0, prune L0E1, prune L0E2]` at depths 3, 4, 5 and 6 alike, because the cap binds at every rung (15-30 feasible candidates). |
| **Registered outcome** | **Branch 1 - P1+P2 PASS.** P1: `442/576` against the bar of `313`, mean `-0.117494`, p `1.13e-39` - the gate is open at the third rung above the onset. P2: rank-sum U `142035.0`, p `1.20e-05` (bar `0.025`) - **the slope continues**, so a depth-4 pass is a point on a rising curve rather than a first step that flattens. |
| **And the slope is a constant** | The full ladder's value (adaptive-vs-null, negative is better): depth 2 `+0.020811`, 3 `+0.004643`, 4 `-0.037147`, 5 `-0.075335`, 6 `-0.117494`. Five rungs, one direction; increments `-0.0162`, `-0.0418`, `-0.0382`, `-0.0422`. So past the first step the axis buys **roughly 0.04 nats per added layer** and the newest step is the largest since the gate itself - three points say the gradient is not turning over, which is stronger than 5.55's two-point reading could support. |
| **The churn regime flips at depth 6** | The collapse (null-vs-random, positive = the do-nothing arm is the worse one) runs `-0.099659`, `-0.106794`, `-0.070072`, `-0.029226`, **`+0.028013`**: the first rung where doing nothing is worse than churning blind. The arm levels say it plainly - depth 6 is `10.5882` (adaptive) < `10.6777` (random) < `10.7057` (null), where at every earlier rung the null arm was second or first. The depth axis does not merely make the controller better; it makes the **unmutated form** progressively less viable. |
| **The decomposition over the whole upper ladder is the result** | Value is `adaptive - null = (adaptive - random) - (null - random)`. From depth 3 to depth 6: the controller's own edge goes `-0.102151` -> `-0.089482`, rank-sum p **`0.266`** (flat, and slightly *worse*); the churn term goes `-0.106794` -> `+0.028013`, p `4.7e-29`. Of the `-0.122137` value improvement, the edge contributes **`+0.012670`** and the churn term **`-0.134807`**. 5.50 found this shape at the width crossing and 5.55 at the first depth step; with three points above the gate it is a property of the axis, and it is the sharpest form the program's central decomposition has taken. |
| **What it settles** | 5.54 established the gate is depth per se, 5.55 that the axis keeps paying, and this arm that the gradient is constant while the controller's contribution stays where it was. So a Gate 4 pass at depth 4 is a point on a rising curve with a measured gradient - and **the thing that rises is not selection skill**. Read with 5.53 (the one reachable gene reads *against* value), both tracks say the same thing from opposite sides: what this program's value tracks is the substrate's need for mutation, which the search neither creates nor enlarges. |
| **Scrutiny before write-up** | Built-in checks 576/576 (mutations 3/3/0, zero null search FLOPs); three replays of committed seeds (92400, 92700, 92958) exact at `OMP_NUM_THREADS=1`; both primaries, both decomposition rank-sums **and** the whole-ladder pair, the increments, the arm orderings and the algebraic split were all computed before any entered prose, and the identity `edge + churn == value` was verified rather than assumed. **Scoped**: the sixth layer carries its 7,392 / 2,592 parameters irreducibly, so this is the axis as a package; the rungs mix 384- and 576-seed cells and are read against each other descriptively (the `0.04`-per-layer gradient is a cross-cell descriptive, while both primaries are tests on their own cells); the increments are measured at the registered 48 steps and schedule. |
| **Result** | **1396 tests pass · 254/254 quick · 274/274 full claims agree**; pre-registration `9f31495`, frozen revision `827d294`; summary `depth6-dose-576.json` (`study_c63c14999b985dde`) pins the cell, both branches, the ladder, the flip and the decomposition. |

## The report's header and deliverables checklist had been stale for eight days (2026-09-21)

| | |
|---|---|
| **What was found** | The report's opening line read "Status: skeleton covering the narrowed first scientific milestone (Gate 3, low-fidelity Experiment 0). Updated: 2026-09-13" - eight days and three gates of recorded results behind, describing the document as a skeleton while it carried 91 section-5 subsections and a synthesis. The deliverables checklist understated shipped work in the same way: items 3-5 read `PARTIAL`/`TODO` while `src/act/training/v0_handoff.py` was already the declared path's publish-and-restore contract, and item 9 claimed "no real-corpus benchmarks" while the architecture track had a four-arm real-text series. |
| **Why nothing caught it** | Neither is a number, so no claim audit and no summary pin could see it. The report had no index either, which is the same defect from the reader's side: 8,000+ lines across 91 section-5 subsections with nothing naming what is where. |
| **The fix is structural, not editorial** | The header now states the current scope, the date and what the document is *not* (the architecture track's record, the runbook, and the implementation tracker all have their own files, named). A **Contents** table names the top-level sections and groups section 5's 91 subsections by what each range answers. The three deliverables rows now claim what the code does, with the module and layout named. |
| **Pinned so it cannot go stale again** | `tests/test_report_navigation.py`: the header date must equal the **newest date the body records** (a stale "Updated:" line now fails); every top-level section must be named in the Contents; every section-5 subsection must fall inside exactly one declared, gap-free group; every registered arm's `report_section` must lie inside a group; and the code and artifacts the checklist names must exist, with the rows required to claim `DONE for the declared path`. The row-scraping is anchored to the table rows so a subsection range like `5.1-5.3` cannot be read as the top-level range `1-5`. |
| **Result** | **1396 tests pass**; the report's own index is now a test surface rather than a description. |

## The depth dose: the gate is a steep step on a slope, and the slope is the substrate's (2026-09-21)

| | |
|---|---|
| **The gap** | 5.48-5.54 closed every covariate 10.3(ii) named, and the closing left the project's central claim resting on a two-cell comparison: depth 2 and 3 fail the bar, depth 4 passes, and **nothing has ever been measured above the upper cell**. Width was dosed (24 -> 28 -> 96), expert count was dosed (4 -> 6), the budget was dosed (48 -> 192) - depth, the one covariate that actually moves the gate, had only its bracket. So the gate could be a slope (a depth-4 pass is a point on an axis) or a cliff edge (depth 4 is the only depth that works), and nothing in the program distinguished them. |
| **The design** | One fresh cell of the registered config with `depth` 4 -> 5 and **nothing else changed** - same width 24, same expert count 4, same 48 steps, same interval (12), same cap, same policy. The family's **cleanest dose**: 5.48 moved width, 5.53 moved experts, 5.54 moved the budget, this moves one key. 576 seeds on 91384-91959. P1 is the cell's own value conjunction; P2 is a one-sided rank-sum of its value deltas against the committed above-onset cell's, non-paired. Both at `alpha/2` (bar 313). |
| **P2's ruler was rehearsed before the run** | On a measured instance of the same comparison shape - two independent cells, same depth and width, one given a strictly better substrate: `depth4-prefix` -> `width-d4-long-w24` returns U `41829`, p `2.6e-60` in the registered direction, and the reversed call is a dead null (p > 0.999). So its result here is a detection, not an artifact of the ruler. |
| **Registered outcome** | **Branch 1 - P1+P2 PASS.** P1: `375/576` against the bar of `313`, mean `-0.075335`, p `1.89e-13` - **the gate is open above the onset**. P2: rank-sum U `97584.0`, p `9.99e-04` - **a fifth layer creates more value than a fourth**, and a depth-4 pass is a point on a scaling axis rather than a knife-edge. |
| **The ladder is monotone** | Value (adaptive-vs-null, negative is better): depth 2 `+0.020811`, depth 3 `+0.004643`, depth 4 `-0.037147`, depth 5 `-0.075335`. Four rungs, one direction, no reversal; increments `-0.0162` (2 -> 3), `-0.0417` (3 -> 4, the largest) and `-0.0382` (4 -> 5). So the gate is a **steep step on a slope**, not a discontinuity in kind, and the axis keeps paying past it. The arm ordering agrees from the other side: the do-nothing arm is best at depths 2 and 3, the adaptive arm at 4 and 5. |
| **And the improvement above the onset is not the controller** | Decomposing `adaptive - null = (adaptive - random) - (null - random)`: the controller's own edge is **flat** from 4 to 5 layers (`-0.107220` -> `-0.104561`, rank-sum p `0.418`, reversed p `0.582`) and saturates by depth 3 across the ladder (`-0.0788`, `-0.1022`, `-0.1072`, `-0.1046`); what moves is the churn term (`-0.070072` -> `-0.029226`, p `1.1e-04`). A deeper substrate pays because the **unmutated form gets relatively worse**, not because the controller gets better - the same decomposition 5.50 found at the width crossing, now measured on the depth axis. |
| **What it settles** | "The gate is depth per se" now has a shape: monotone in depth, largest step at the crossing, nearly as large a step after it. A Gate 4 pass at depth 4 is therefore not fragile - the same controller on a deeper substrate of the same block would create *more* value - and equally it is **not evidence of a controller whose skill grows**. 10.4 gains the sharpest form of its gap: the axis that both gates and scales is unreachable from the search space, while the axis the search can move reads against value creation. |
| **Scrutiny before write-up** | Built-in checks 576/576 (mutations 3/3/0, zero null search FLOPs); three replays of committed seeds (91385, 91650, 91958) exact at `OMP_NUM_THREADS=1`; both primaries, both decomposition rank-sums, the ladder levels, the increments and the arm orderings computed before any entered prose. **Scoped**: the fifth layer carries 7,392 total / 2,592 active parameters irreducibly (every rung adds exactly those), so this is the depth axis *as a package* rather than a matched-parameter contrast - deliberately, since 5.48 and 5.53 measured that buying parameters with width or experts does not move the gate. Rungs are 384-seed committed cells read against a 576-seed cell descriptively; no cross-cell arithmetic enters either primary. |
| **Result** | **1367 tests pass · 254/254 quick · 274/274 full claims agree**; pre-registration `5fb7bd3`, frozen revision `3516af5`; summary `depth5-dose-576.json` (`study_3624b25a32797285`) pins the cell, the branch, both rank-sums and the ladder. |

## The total-compute covariate: 4x training is inert below the onset, and the gate is depth (2026-09-21)

| | |
|---|---|
| **The question** | 10.3(ii) named "parameter count, layer width, expert count" and then total compute as what could separate depth per se from capacity. 5.48-5.53 closed the first three at this scale, so **budget was the last candidate** - and it is the one covariate that is not a property of the substrate at all. |
| **The design** | The **one empty corner of the depth x budget 2x2 at width 24**: depth 3 at 192 steps (48 -> 192, interval 12 -> 48, the same three mutational events 5.52 ran one depth up), 576 seeds on 90000-90575. It is the family's **first arm that moves neither a parameter nor the search space** - 5.48 moved width, 5.53 moved experts, this moves only the step count - which is what makes it a compute arm rather than a second capacity one. Two primaries at `alpha/2` (bar 313): P1 the cell's own value conjunction; P2 a one-sided rank-sum of its null-vs-random deltas against the committed 48-step cell's, the family's **first non-paired test** (the two cells share no seed). |
| **The regime primary was rehearsed before it ran** | 5.52 already measured the same treatment at depth 4: the collapse goes `-0.0701` -> `+0.1638`, U `28795`, p `1.9e-84`. So the test was known to detect the regime change where the records say one exists - without which a non-detection at depth 3 would be uninterpretable - and its direction was pinned (the reversed call reads as a dead null, p `0.294`). |
| **Registered outcome** | **Branch 4 - both FAIL.** P1: `281/576` against the bar of `313`, mean `+0.039005`, p `0.734` - a win rate of `48.8%` against the committed 48-step cell's `49.0%`, i.e. four times the training does not bring the below-onset cell one step closer. P2: collapse `-0.106794` -> `-0.128388`, still negative, rank-sum U `112872.0`, p `0.706` - the regime has not moved. |
| **The finding is that the budget's effect is itself gated by depth** | The identical treatment does two large things one depth up and neither here: value `-0.037147` -> `-0.204292` (rank-sum p `2.6e-60`) and collapse `-0.070072` -> `+0.163809` (p `1.9e-84`) at depth 4, against p `0.082` and p `0.706` at depth 3. So the budget is not something the below-onset cell is short of, and not something that damages it: once there is a gate to cross, 4x training crosses further and brings a churn regime with it; where there is no gate it is inert. **All four named covariates are now measured at this scale, three do not move the gate, one removes value above it, and the one that does move it is the axis 10.1 identified.** |
| **Descriptive secondaries read as 5.53's damage cells did** | At 192 steps depth 3's best arm is the do-nothing one (`6.3063` < `6.3453` < `6.4347`), the collapse is negative (`null_vs_random` `400/576`, p `2.4e-21`) and the controller's own edge over the pool-matched random arm survives strongly (`443/576`, `-0.089383`, p `3.4e-40`). The controller still *selects* - this is not a broken search - it simply cannot beat doing nothing, at either budget, on this substrate. |
| **A tripwire fired, and it was right to** | `test_the_long_budget_arm_has_exactly_one_cell` asserted that no profile whose name ends in `-long` exists other than 5.51's cell - the name-suffix predicate that stood in for the real intent, and 5.52's own width-24 long control (`width-d4-long-w24`) never matched it. The predicate is now the **lineage**: among cells running 4x the registered steps, exactly one is width 96 (5.51's) and exactly one is the registered width (5.52's control), so a second cell of either lineage still fails loudly while a different arm's cells do not. |
| **Scrutiny before write-up** | Built-in checks 576/576 (mutations 3/3/0, zero null search FLOPs); three replays of committed seeds (90003, 90317, 90571) exact at `OMP_NUM_THREADS=1`; both primaries, both rank-sums, the 2x2 levels and the arm orderings computed before any entered prose. **The freeze window itself was fixed on the way**: `test_the_committed_summary_regenerates` raised for a frozen-but-not-yet-run profile, which is exactly the state a pre-registration passes through, so it now skips there - and the loud failure for a summary that never lands is `_committed`, which the divisor and label tests still call. |
| **Scoped** | This measures 4x the *steps* at the registered schedule, not a tuned long-budget schedule, and the same three mutational events. The only dose that has ever crossed the gate is 14.8x of *width* (5.50); a 14.8x step increase is not measured and the doses are not comparable, so the licensed claim is the registered one - 4x training is inert below the onset and not inert above it - rather than a statement about the budget as such. The 576-seed cell is read against the 384-seed corners descriptively, and no cross-cell arithmetic enters either primary. |
| **Result** | **1349 tests pass · 254/254 quick · 274/274 full claims agree**; pre-registration `64046da`, frozen revision `a43e50d`; summary `depth3-long-576.json` (`study_107b6492aa331ab7`) pins the cell, the branch, both rank-sums and the 2x2. |

## A dirty working tree could end the campaign after a session had already trained and published (2026-09-21)

| | |
|---|---|
| **What was found** | `kaggle_queue.ensure_code_dataset` refuses to publish on a dirty tree, by design - a tree that is not a clean commit cannot be pinned to a revision. But the driver calls it **before every push**, so on a two-week unattended campaign one uncommitted edit during a poll ends the driver: `SystemExit`, no session 3, and the campaign stops one session *after* the work has been published. I found it while checking whether it was safe to leave a fix uncommitted; the answer was no, and that is a property of the driver rather than of the fix. |
| **Why the refusal existed** | `stage_code_dataset` tarred files from the **working tree** while writing a marker naming `HEAD`, so a dirty checkout published uncommitted edits under a clean commit's name - and the handoff manifest records that revision, so the provenance would be wrong too. The refusal was the guard against a real defect. |
| **The fix removes the failure instead of detecting it** | Everything that ships is now read from `HEAD`: the dataset through `committed_tar` (one `git archive HEAD`, which also preserves the executable bits git recorded and needs no per-file subprocess), and the kernel body through `kernel_body`, which uses the working copy **only when that one file is unmodified** - so a local edit stays testable without a commit, while a body from one revision can never be embedded against a package from another (the mismatch the refusal existed to prevent). A dirty tree is now *reported* (`the working tree differs from <rev> in N path(s), HEAD is what ships`) rather than fatal, and `code_revision` returns the commit instead of exiting. |
| **Proved on the real tree, not just in fixtures** | Staged the actual repository while `scripts/kaggle_queue.py` and `tests/test_kaggle_queue.py` were dirty: the archive carried **238 files + the marker**, excluded `artifacts/` as designed, and its copy of `kaggle_queue.py` **lacked** the uncommitted edit - i.e. the payload was HEAD's, and the publish would have been correct rather than merely non-fatal. The staged kernel body still compiles with the window pinned at 7.0. |
| **Pinned** | `test_code_revision_reports_a_dirty_tree_instead_of_refusing` (the revision is still the commit, `dirty_paths` names the dirt, and an unresolvable repo does not raise out of the reporter on the hot path), `test_the_staged_archive_is_head_bytes_even_when_dirty` (a real one-file git repo, edited after commit, whose archive still carries the committed text), and `test_the_kernel_body_is_the_committed_copy_when_it_is_edited`. Mutants: reading the working tree again in staging fails **1**; restoring the fatal refusal fails **1**; taking the body from the working tree unconditionally fails **1**. |
| **The live driver was restarted onto it** | The driver running since 08:48 had the fatal path loaded in memory, so the fix alone would not have protected the campaign. It was stopped and relaunched (`setsid nohup ... --queue session --max-sessions 6 --wait-for-quota`), and the restart **adopted** the session already RUNNING - measured: `a session is already RUNNING; adopting it instead of pushing a second` - while republishing the code dataset at the fix commit (`298cd5a711ec`). The campaign therefore needs exactly one quota wait (the week refreshes 2026-09-26, and `--max-waits` defaults to 2) before session 3 lands the remaining 8 seeds. |
| **Result** | **1330 tests pass**; the runbook's pre-flight item 7 records the new rule beside the old one. |

## The expert-count covariate: the one gene the search can move reads against value creation (2026-09-21)

| | |
|---|---|
| **The question** | 10.3(ii) named "parameter count, layer width, expert count" as the covariates that could separate depth per se from capacity. 5.48 closed the width form at matched parameters; 5.49/5.50/5.51/5.52 closed its dose and budget forms. Expert count is what remained - and unlike the other candidate (total compute) it is decisive for this project, because expert grow/prune is the **only** operation the V0 search space can reach (10.4, pinned by `tests/test_topology_transactions.py`). |
| **The design** | Two fresh cells of the registered config with only `expert_count` changed, 4 -> 6, holding width 24 and `active_experts` 2: depth 3 x E6 (30,624 total / 9,024 active parameters) and depth 4 x E6 (40,464 / 11,664). The depth-3 cell is the family's **tightest equal-parameter partner yet** - `-0.156%` against the above-onset cell's 30,672, where 5.48's width-28 partner was `+1.695%` - **and the first that buys its parameters without buying compute**: a token still routes to two experts, so its active parameter count is **21.3% below** the above-onset cell's, where the width partner matched both axes (`+1.7%` / `+2.0%`). Two primaries at `alpha/2` (bar 313/576). |
| **Registered outcome** | **Branch 4 - both FAIL.** Neither cell creates value: `experts-d3-dose` `211/576`, mean `+0.059544`, p `1.000`; `experts-d4-dose` `260/576`, mean `+0.019507`, p `0.991`. |
| **P1 fails exactly as the width arm failed** | At 30,624 total parameters - `-0.156%` from the above-onset cell - depth 3 still does not create value, and is *further* from the bar than the committed below-onset cell already was (`188/384`). So the matched-parameter form of (ii) now has two answers that agree: the gate does not move for layer width, and it does not move for expert count. **The candidate that remains is total compute.** |
| **P2 fails the other way, and that is the result** | At E=4 the above-onset cell creates value and the adaptive arm is its *best* arm (`12.4413` < `12.4785` < `12.5485`). At E=6 the same depth-4 cell stops creating value and the ordering **inverts**: the do-nothing arm becomes the best one (`12.5379` < `12.5574` < `12.6332`). The dose is a strong, real effect, not noise - the E=4 cell's per-seed deltas are stochastically more negative than the E=6 cell's at rank-sum p `9.1e-09` (depth 4) and p `8.0e-09` (depth 3), in the direction that means the extra experts *remove* value creation. |
| **And it is not churn** | The collapse (null-vs-random) stays negative at all five cells, so there is no churn harm anywhere in this covariate, and the controller's edge over the pool-matched random arm stays negative and significant at both new cells (`391/576`, `-0.077625`, p `2.9e-18`; `372/576`, `-0.075803`, p `1.2e-12`). The controller still *selects* - what dies is its value relative to doing nothing, because at E=6 the substrate's own extra capacity is worth more than any of the three mutation policies layered on it. |
| **What it says about the search space** | 10.4 observed that the reachable gene is a different covariate from the architecture track's measured lever (residual width). This arm adds the reading on the second track's own terms: the reachable gene is a covariate that reads *against* selection value when it is raised. A V0 search that grows experts is asking a question this program has now measured a negative answer to - a bound on what a Gate 4 pass would **license**, not a verdict on Gate 4, whose question is selection against a null on a declared block. |
| **Scrutiny before write-up** | Built-in checks 576/576 per cell (3/3/0 mutations, zero null search FLOPs); six replays of committed seeds (three per cell) exact at `OMP_NUM_THREADS=1`; the cell statistics, the five-cell table, the ordering inversion and both dose rank-sums computed before any entered prose. Scoped: the expert count is a property of the *substrate* in this arm as in every 5.4x cell - no run changed its expert count mid-flight - so this bounds what a wider expert pool is worth, and does not measure whether a search that grows its own experts does better. |
| **Two record defects found on the way** | The arm also forced the summaries to be regenerated, which found mislabelled branches in three earlier arms and a CLI default that could rewrite a bar - recorded in the entry below. |
| **Result** | **1314 tests pass**; pre-registration `76d2134`, frozen revision `210ec35`; summaries `experts-d3-dose-576.json` (study `study_638083d241c0c745`) and `experts-d4-dose-576.json` (study `study_384662eb637caa38`) pin the cells, the branch, the five-cell table and the dose rank-sums. |

## The committed labels named another arm's branches, and one recorded statistic was never reproduced (2026-09-21)

| | |
|---|---|
| **How it was found** | By regenerating the committed toy summaries from their own shards instead of reading them - the method that found 5.9's alpha slip. Three defects fell out of one regeneration, and no measured number moved: every corrected field is a label or a level. |
| **1. Sections 5.48, 5.49 and 5.52 carried the 5.9 vocabulary's branch string** | That vocabulary's two primaries are null-vs-random and adaptive-vs-null *of one cell*, so it describes the arms that registered that pair (5.9, 5.10, 5.12, 5.15) and no others. The width arms registered a different pair, and the label read backwards where it mattered: `width-d3-equal` (5.48) and `width-d3-dose` (5.49) were labelled 5.9's "branch 2 (1 PASS / 2 FAIL)" - a *pass* - for the two cells their arms registered as **failures** (`252/576` and `235/576`), and `width-d4-long-w24` (5.52) was labelled "no churn harm on this substrate" where the recorded finding is a **collapse of `+0.163809`**. |
| **The fix is the one 5.50's fix started** | 5.50's own stretch replaced the 5.9 vocabulary with the arm's own per-cell label - but only for the two sections it named in a hard-coded list, so 5.48, 5.49 and 5.52 kept the old wording. The style is now **data on the profile** (`per_cell_primary`), rendered from the profile's own section, so a new arm cannot inherit another family's branch names. Regenerating the five affected summaries moves exactly one line each; 5.50's two cells and 5.51 regenerate **byte-identically**, so nothing earlier moved. |
| **2. And the level was still reachable from the command line** | `summarize`'s CLI defaulted the Bonferroni divisor to `2.0` for every profile, so regenerating one of the two 5.50 cells wrote `bonferroni_alpha 0.025` and bar **313** where the registration states `alpha/3` and bar **315** - silently, and after 5.50 had already recorded precisely this class ("printing bar 318 where the registration said 315"). This is not hypothetical: it is what the first attempt at this correction actually did. The divisor is now the profile's declaration (`bonferroni_divisor`, required on all fourteen profiles) and a mismatched override is **refused before any shard is read**; with it in place both 5.50 cells regenerate byte-identically at 315. |
| **3. Six committed summaries cannot be regenerated, and nothing was checking** | `delta_stddev` was the one comparison field no summary test recomputed. Recomputed from the stored `per_seed` triples it reproduces **47 of 53** summaries bit-exactly - all eight width arms among them - and misses exactly the six `null-control-*` records of 5.9-5.15, whose values match `statistics.pstdev` instead (an accumulation the code has since replaced, and order-sensitive, so no single formula reproduces all six in the stored order either). |
| **Recorded rather than rewritten** | The drift is bounded at `2.8e-15` relative (~12 ulp), `delta_stddev` is quoted by no report section and read by no decision, and rewriting six committed records to make a statistic nobody reads exact is not a repair. It is enumerated (the six files are named in the test) and bounded instead. What was actually missing is now pinned: the field is recomputed at all, so a change to it cannot go unnoticed again. |
| **Pinned** | `test_the_label_style_is_the_arms_own_declaration`, `test_every_committed_label_is_its_own_arms`, `test_every_profile_declares_its_bonferroni_divisor` (declaration, committed bar and stored level agree), `test_a_summary_cannot_be_written_at_a_level_its_registration_does_not_state`, `test_the_committed_summary_regenerates` (byte-exact for twelve profiles, bounded to `delta_stddev` for the six earlier records), and `test_the_committed_stddevs_are_bounded_and_the_vintage_is_enumerated`. Mutants: restoring the hard-coded section list fails **5** (exactly the five corrected summaries); letting the divisor fall back to `2.0` fails **3**, including the refusal itself. |
| **Result** | **1306 tests pass**; report §5.52.1 records the correction, and the five regenerated summaries plus the machinery carry it. |

## The long-budget width control: the collapse is capacity's, and the levels invert with the budget (2026-09-21)

| | |
|---|---|
| **The question** | 5.51 registered its own limit in the same breath as its result: at 14.8x the collapse is relieved by 55% at 4x the budget and the arms move from `15.1`-`16.9` to `3.8`-`4.5`, but "there is no 192-step width-24 control (pinned as absent) ... so no cross-width absolute claim is made at the longer budget". §5.52 is that control, and nothing else. |
| **The design** | One fresh cell: 5.51's config with only the width changed (96 -> 24), 576 seeds on 87600-88175, which completes the prefix-lineage 2x2 the collapse has to be read in (`depth4-prefix` w24@48, `width-d4-dose2` w96@48, `width-d4-dose2-long` w96@192, this cell w24@192). Every corner is prefix-policy: 5.10's depth-4 cell is the same width at the same budget but runs *spread*, so it is deliberately not the corner used (pinned by a test, because ranking against it would move width and policy at once). Two primaries at `alpha/2` (bar 313): P1, the narrow cell's collapse is stochastically smaller than the wide long cell's (one-sided rank-sum); P2, its own value-creation conjunction. Cost ~4.6 s/seed, ~5.5 min of wall clock across 20 shards, no GPU quota. |
| **Registered outcome** | **Branch 1 - P1+P2 PASS.** P1: collapse `+0.163809` against `+0.708276` (U `34.0`, p `5.7e-190`) - the collapse that survives 4x training belongs to the wide substrate, so 5.51's relief is a *capacity* effect rather than a property of training. P2: `550/576` (bar 313), mean `-0.204292`, p `3.5e-129`; adaptive `5.4605` < random `5.5010` < null `5.6648`. |
| **The 2x2, and the two directions it points in** | Collapse (null-vs-random): w24@48 `-0.0701` -> w24@192 `+0.1638`; w96@48 `+1.5739` -> w96@192 `+0.7083`. So 4x training relieves the wide substrate's collapse by 55% and *raises* the narrow one's - from a baseline where there was nothing to relieve, because at width 24 the churn harm is already absent at 48 steps. The within-width relief (difference in differences, descriptive): `+0.2339` at width 24 against `-0.8656` at width 96. |
| **The levels invert, which is what the control was for** | At 48 steps the wide substrate is the worse one (arms `15.13`/`15.29`/`16.86` against `12.44`/`12.55`/`12.48`); at 192 steps it is the better one (`3.83`/`3.84`/`4.55` against `5.46`/`5.50`/`5.66`). So 5.50's "capacity is a liability" is a fixed-budget statement in the strongest form this program can support: given 4x the training the ordering inverts. |
| **Where the value comes from changes with the width** | At 192 steps the width-24 cell's own controller edge is `-0.0405` (`390/576`, p `6.2e-18`) against the width-96 cell's `-0.0066` (`335/576`, p `5.2e-5`) - a rank-sum at p `3.85e-16` that the *narrow* one is the larger. At width 96 the value is almost entirely the mutational package (the controller is ~1% of `-0.7149`); at width 24 the controller contributes ~20% of `-0.2043`. So 5.51's "the value at 14.8x is the mutational package" is also a wide-substrate reading. |
| **Scrutiny before write-up** | Built-in checks 576/576 (mutations 3/3/0, zero null search FLOPs); three replays of committed seeds exact at `OMP_NUM_THREADS=1`; and the collapse, the value and the edge were computed before any of them entered prose. Scoped: the learning rate stays at the registered value at both widths, and the two widths differ by 14.8x in parameter count - this completes the 2x2 the width line reads, it does not turn width into a matched-budget contrast (that is 5.48's equal-parameter arm, at 48 steps). |
| **Result** | **1281 tests pass · 254/254 quick · 274/274 full claims agree**; pre-registration `f636990`, frozen revision `9264aa2`; summary `width-d4-long-w24-576.json` (study `study_cdbb2da593c23f32`) pins the cell, the branch and the four corners. |

## The platform's 8-hour cap: session 1 died inside a 9-hour window the plan had declared (2026-09-21)

| | |
|---|---|
| **What happened** | Gate 4's primary-queue session 1 (`sid123123234/act-t4-session`) was pushed 2026-09-20 10:39:54Z and ended `ERROR` after **8 h 00 m** - the driver, polling every 60 s, observed it at 18:43Z - which is *before* its own 9-hour deadline. The weekly GPU quota fell 25.11 h -> 17.06 h across it, so it billed **8.05 GPU-hours** and published nothing: the queue sits exactly where it was before the session started. |
| **The kill was total, and the emptiness is measured rather than inferred** | The platform returned an empty kernel log and no output files. That is a fact about the session and not about the query, which a control check settles: the same call against the same account returns 24,973 characters of log and 288 output files for `act-t4-archzoo`, and 5,157/249 for `act-t4-selfcheck`. So nothing of the session's own account of itself survives, and because a session publishes only at exit - the published dataset is its only durable copy - the pairs it had already finished went with the container. |
| **Why this was a plan defect and not a run of bad luck** | §5.2's session model is built on Kaggle's *documented* GPU limit: 12 h, per the 2022-01-24 product update that raised it from 9 h. `--hours 9` was therefore the registered "usable" window, and `USABLE_HOURS` still is. Rule 1 then starts a pair whenever `pair_time + 45 min` fits the remaining clock, which spends a session's *last hour* on a pair and its exit publish. At the T4's measured 1.443 h pair time a 9 h window starts a fifth pair at 5.77 h that ends at 7.2 h, leaving the publish to race the cap - and the cap is 8 h. The platform's limit is not ours to set; the declaration was, and it sat above the limit. |
| **The amendment** | `session_plan.PLATFORM_SESSION_CAP_HOURS = 8.0` records the measurement, `BOOT_AND_PREP_HOURS = 0.25` the non-training part of the window, and `DECLARED_SESSION_HOURS = 7.0` is what remains after the publish reserve (8.0 - 0.75 - 0.25). The kernel's own default window (`kaggle_t4.DEFAULT_HOURS`) and every plan-composition site move with it - the launch artifact's `_rule1` and `three_queue_*` fields and the dtype probe's plan lines - so the artifact a session writes and the command it runs are one decision again rather than two that can drift. |
| **And it is free, which is why it is not a trade-off** | At the T4's measured placement rate a 7 h window packs **4** pairs and refuses a fifth - exactly the pair the killed session was spending when it died - ending its last pair at ~5.8 h and publishing with over an hour in hand. Twelve seeds need **three sessions either way**, because `ceil(12/4) == ceil(12/5) == 3`, so the amendment gives up no session, no training hour and no quota: the campaign stays nine sessions over a 51.95 h training floor, which is the same nine the corrected plan already recorded. |
| **A second defect: the driver made the loss undiagnosable** | On a failed session `kaggle_queue.py` printed `session ended with ERROR; kernel log follows` and then *nothing at all*, because `kaggle kernels output` had returned no files - so an operator could not tell an empty log from a missing one, on a session that had just burned 8 GPU-hours. It now reports the elapsed wall clock and the quota burned, says explicitly when the platform returned nothing and that the queue restarts from its last published seed, and names what it did return when that is a partial survival rather than a total loss. |
| **Pinned** | `test_the_declared_window_is_the_measured_cap_less_its_reserve` (the plan's window and the kernel's default are one value), `test_the_artifact_plans_at_the_declared_window` (the artifact prices the campaign at the window it will be given), and two driver tests (a total loss is reported; a partial survival reads as partial). Mutants: restoring the registered 9 h window fails 1; setting `DEFAULT_HOURS` back to 9 fails 1; pricing the artifact at the registered window fails 4; and **restoring the driver's original silent path verbatim fails 2**, so the test would have caught the failure that actually happened. |
| **Result** | **1262 tests pass · 254/254 quick claims agree · 9/9 records reproduce**; the campaign was relaunched at the corrected window (`Kernel version 3`), and the driver re-started while the session was live *adopted* it rather than pushing a second - measured: "a session is already RUNNING; adopting it instead of pushing a second". One operational note from the same stretch: a detached job started with `nohup ... &` alone did **not** survive the shell that launched it here (the driver and a 20-shard sweep both died with their parent), so launches now use `setsid nohup ... &`, and "is the driver still alive" is checked after every launch rather than assumed. |
| **Validated on a real session (2026-09-21)** | The re-launched session finished **`COMPLETE`** and the driver read it back: `queue progress: 4/12 seeds complete (session completed 4 pair(s))` and `exit publish: published (4 seed version(s))`. That is exactly what the amendment predicted - a 7 h window packs 4 pairs and refuses a fifth - and it is the campaign's first session to publish anything: the queue moves 0/12 -> **4/12**, the driver pushed **kernel version 4** for session 2 within the same loop, and the run left no residue. The 8 h kill of the previous session cost 8.05 GPU-hours and published nothing; this one published all four pairs and left over an hour of its window unused. |

## The 14.8x collapse against the budget: half of it was the budget, and the controller's edge is what 4x training takes away (2026-09-20)

| | |
|---|---|
| **The question** | 5.50 registered a limit rather than only a result: at 14.8x the crossing arrives with the do-nothing arm collapsing (`+1.574`) and the arms absolutely worse (`15.1`-`16.9` against the rung's `12.4`-`12.5`), which at a fixed 48-step budget cannot be separated from "this substrate needs more training than the budget gives it". §5.51 runs the identical config at 4x the steps (192) and 4x the mutation interval (48) - the same three mutational events, asserted 3/3/0 at verdict time - with two primaries at `alpha/2`: P1, the collapse *relieved* (a one-sided rank-sum against the committed 48-step cell, the non-paired test 5.50 introduced), and P2, value creation surviving (5.50's own conjunction). Both outcomes were named before the run and the timing pilot ran with both on the table. |
| **Registered outcome** | **Branch 1 - P1+P2 PASS.** P1: the collapse falls from `+1.574` to `+0.708` (rank-sum U `2925`, p `1.7e-183`) - relieved *by 55%, not removed*, since the unmutated arm is still the worst in the longer cell (`null 4.548` against `3.833`/`3.840`). P2: `576/576`, mean `-0.7149`, p `4.0e-174` - at a workable budget the 14.8x substrate still clears the do-nothing bar at every seed. |
| **The decomposition, which is the finding** | The controller's *own* edge (adaptive-vs-random) falls from `-0.159` to `-0.0066` (reverse rank-sum p `2.1e-45`): still detectable against zero (`335/576`, p `5.2e-5`) but **24x smaller**. So at both budgets the value at 14.8x is the mutational pool (mutating at all): the collapse is largely a budget artifact and the controller's share *shrinks* as the budget grows - 5.50's "escape from a collapse, not a sharper controller" holds at the longer budget in a stronger form, with the escape itself reduced. |
| **The absolute limit, scoped** | 4x training moves the same arms from `15.1`-`16.9` to `3.8`-`4.5`, so 5.50's "capacity is a liability" is a *fixed-budget* statement and nothing more. There is no 192-step width-24 control (pinned by a test), so no cross-width absolute claim is made at the longer budget; and the learning rate was deliberately left at the registered value, so this is not a tuned longer run. |
| **Scrutiny before write-up** | Built-in checks 576/576 (3/3/0 mutations), three replays of committed seeds exact at `OMP_NUM_THREADS=1`, the relief bound stated as a *ratio* (0.45x, not "half"), and the collapse's persistence recorded beside the branch that calls it the budget's. |
| **Result** | **1257 tests pass · 254/254 quick · 274/274 full claims agree**; summary `width-d4-dose2-long-576.json` (study `study_055561a11315cf41`) pins the cell, the relief, and the shrinking edge; pre-registration `a590a7f`, frozen revision `07e66cf`. |

## The width amplifier's second dose: the gate falls at 14.8x - and the decomposition says why (2026-09-20)

| | |
|---|---|
| **The question** | §10.5(B) prices a width *mutation* by the marginal value of capacity above the onset; §5.49 left it open with one dose - capacity amplified the do-nothing contrast at 3.80x, but one dose cannot say whether it keeps paying or plateaus. Registered as §5.50 with three primaries at Bonferroni `alpha/3` (bar **315/576**): the two per-cell sign tests **plus the family's first non-paired primary** - a one-sided rank-sum between two independent fresh cells, rehearsed on the committed rungs before the run (`p = 4.5e-44` on the measured 2.84x step, `p = 0.50` on the measured +1.7% wash). |
| **The design** | Width 96 at both depths - 11.15x / 14.82x the above-onset cell, 3.90x the §5.49 rung - on fresh blocks 80000-80575 and 84000-84575. The depth-4 block's first draft was 81000, and the freshness test caught that it overlapped §5.17's reusability block (81000-81383); moved to the first free 576-window. |
| **Registered outcome** | **Branch 1 - P1+P2 PASS** at the `alpha/3` bar: `width-d3-dose2` `573/576`, mean `-0.8966`, p `1.3e-166`; `width-d4-dose2` `576/576`, mean `-1.7330`, p `4.0e-174`. At 14.8x the do-nothing bar is cleared at *both* depths - the onset is capacity-driven above a threshold between 3.8x and 14.8x, and §5.49's depth clause is bounded to doses up to 3.8x. **P3 passes in its strongest form**: the 14.8x cell's adaptive-vs-null deltas are completely separated from the 3.8x rung's (`U = 0.0`, p `4.8e-190`; verified as `max(w96) < min(w48)`). Its dose-form rider was pre-registered for the P2-only branch, which did not obtain, so the continuation is recorded as a primary result rather than as that branch's reading. |
| **The decomposition (descriptive, and the actual finding)** | The three paired contrasts across the depth-4 ladder: adaptive-vs-null `-0.0371 -> -0.0326 -> -0.2097 -> -1.7330`; **the controller's edge (adaptive-vs-random) is flat** - `-0.1072 -> -0.1207 -> -0.1945 -> -0.1591`, with the w96-vs-w48 rank-sum reading p `0.98` in the "bigger at 14.8x" direction (reverse p `0.016`); **the pool effect (random-vs-null) is what scales** - `+0.0701 -> +0.0880 -> -0.0152 -> -1.5739` (rank-sum p `6.3e-190`). At 14.8x the unmutated arm is the worst by a wide margin (`adaptive 15.127 < random 15.286 < null 16.860`). Below the gate the reverse pattern: the edge grows with every rung (`-0.102 -> -0.131 -> -0.244 -> -0.311`, w96-vs-w48 p `1.4e-3`) while the pool effect flips sign (`+0.107 -> +0.148 -> +0.305 -> -0.586`). |
| **What it changes** | **The dose buys escape from a collapse, not a sharper controller.** At the fixed 48-step budget the wide arms are absolutely *worse* (`15.1`-`16.9` against the rung's `12.4`-`12.5`), so capacity is a liability that mutation partly repairs - the report states that limit instead of claiming a quality gain. 5.49's "capacity is an amplifier" is qualified to the *bar*, §10.1's churn axis **flips sign** in width (harm at <=1.34x, absent at 3.8x, pays at 14.8x: `576/576` at depth 4, mean `-1.574`), and §10.5(B)'s measured upside becomes a *package* effect (mutation at all), leaving the transfer-policy problem (5.24, 5.40) untouched. |
| **A defect class, caught at the source again** | The first verdict run passed `alpha/3` into a helper that halved again and printed bar **318** where the registration said **315** - stricter than declared, and still §5.16.1's defect class ("a declaration whose bar and alpha disagree cannot be applied as written"). `_cell_stats` now takes the *per-primary level* directly (callers pass `alpha/2` or `alpha/3`), `summarize` takes the arm's divisor, and regenerating the two committed 5.49 summaries is **byte-identical**, so nothing earlier moved. The 5.50 summaries also carry the arm's own per-cell label instead of the 5.9 vocabulary, whose pair of primaries is a different pair. |
| **Scrutiny before write-up** | The result is a regime change, so it was checked before it was narrated: built-in checks held 576/576 per cell; six replays of committed seeds matched the shards exactly at `OMP_NUM_THREADS=1`; `U = 0.0` was verified as a strict separation; and the decomposition was computed before any of it appeared in prose. |
| **Result** | **1243 tests pass**; pre-registrations `05efa7b` (5.50) and `aa78882` (its frozen run revision); summaries `width-d3-dose2-576.json` and `width-d4-dose2-576.json` pin every quoted number, the branch, and the three descriptive readings. |

## The next gate's (A) half is registered and running: the width covariate at the toy onset (2026-09-20)

| | |
|---|---|
| **Why** | §10.5 split "what putting width under test would take" into (A) width as a gene the search is *given* (a design problem) and (B) width as a gene it can *move* (a transfer policy that does not exist anywhere). (A) is "answerable today" but had no design, and it is the arm that decides open question (ii): is the depth onset (3, 4] *depth*, or *capacity*? |
| **The design** | Two fresh cells of the registered toy config with only width changed - `width-d3-equal` (depth 3, width **28**, 576 seeds on 76000-76575) and `width-d4-equal` (depth 4, width 28, 576 seeds on 77000-77575). Width 28 is **solved**, not chosen: it is the closest integer width whose parameter count at depth 3 equals the committed above-onset cell's (depth 4 x width 24 = 30,672 total / 11,472 active; +1.7% / +2.0%), so the depth-3 cell is an equal-parameter partner of the cell that creates value, at a depth that does not. Pinned by a test against the model's own parameter count over widths 20-64. |
| **The bar, stated once** | One-sided exact sign test on adaptive-vs-null per-seed wins; alpha 0.05 with two primaries -> Bonferroni alpha/2 = **0.025**, bar **313/576** (`_gate_min_wins`, asserted). A cell creates value only when the wins clear the bar **and** the mean delta is negative - the conjunction is pre-registered because a win-rate crossing can be a tail effect (depth 1 is exactly that: 117/384 wins with the adaptive arm 0.0508 *worse* on average), and the six committed cells never disagreed on the sign. |
| **Power, sized to the committed effect** | At the committed above-onset rate (225/384 = 0.5859): exact power **0.982** at n = 576 (0.801 at 0.56, 0.641 at 0.55). A smaller true effect reads as a null by design, and that is registered rather than discovered afterwards. |
| **Four branches, fixed first** | P1+P2 -> the onset is a **capacity** onset (depth and width trade off at matched parameters); P1 only -> widening creates value below the onset and destroys it above (a depth x width interaction); P2 only -> the onset is depth-like at this width; neither -> width is harmful at both depths and the covariate is closed. Machine-checked: `width-verdict` reads both cells and prints exactly one branch. |
| **A record defect fell out of stating it** | Writing the bar once meant reading what the siblings applied. §5.9's contract says "Bonferroni alpha/2 = 0.0025" while its own sentence says alpha 0.05 with two primaries, and its committed summary applies 0.025 (bar 212). §5.16.1 already adjudicated this class and chose the stricter reading where it mattered - but its claim that the `0.0025` was *unique to 5.16* is what is wrong: §5.9 carries it too, and no §5.9 primary changes at either bar (280/384 and 117/384 pass and fail identically at 212 and 220). Both spots now carry the correction. |
| **Cost** | Measured at the new cells: ~1.2-1.4 s/seed (3 arms, 4 threads). Two cells x 576 seeds is ~25 min of single-thread CPU across 20 shards - minutes of wall clock, no GPU quota - run while the Kaggle session trains. |
| **The outcome (run 2026-09-20, revision `38f3921`, 20 shards per cell)** | **Registered branch 3 - P2 only: the onset is depth-like at this width.** `width-d3-equal` does **not** create value: 252/576 wins (bar 313), mean `+0.0176`, p `0.999` - a churn-suppression cell (null 14.9594 < adaptive 14.9771 < random 15.1078; adaptive beats random by 0.1307, p 4.2e-25, and loses to doing nothing on average). `width-d4-equal` does: 316/576, mean `-0.0326`, p `0.011` - adaptive 13.6503 < null 13.6829 < random 13.7710. The equal-parameter contrast is the first and third rows of the four-cell table: **at the same parameter count (31,192 vs 30,672, within 1.7%), depth 3 does not create value and depth 4 does** - widening does not buy what depth buys. |
| **What it closes** | The matched-parameter form of open question (ii): capacity in the residual stream is not the covariate the onset reads, so (as §10.5 registered for this outcome) a search that *moves* width asks about a different search than the value measured here. Not closed: "any width" - the contrast is one level (+1.7% params) at two depths; a dose of much wider, no-longer-matched substrates is a different registration. The two width-28 cells are also within noise of their width-24 counterparts as a *level* (descriptive two-proportion p = 0.11 and 0.25), so nothing here says width is harmful. |
| **The dose followed the same day (§5.49)** | 5.48 registered what it did not close - "any width", because +1.7% parameters is a small shift - so the dose form was registered (revision `e5dcbf6`) and run: width 48 at both depths, 88,032 / 116,640 parameters, 2.87x the above-onset cell, no longer parameter-matched. Result: **branch 3 again - P2 only**. Depth 3 fails at width 48 too (235/576, mean `+0.0615`) while depth 4 clears it with `477/576` (mean `-0.2097`, p `1.4e-60`). |
| **What the dose adds (descriptive, labelled as such)** | Two cross-cell readings, neither pre-registered: (1) below the gate the average outcome weakens at *every* width step (`+0.0046 -> +0.0176 -> +0.0615`; win rates `0.4896 -> 0.4375 -> 0.4080`), so capacity is not merely useless there, it is mildly harmful; (2) above the gate the amplifier is the *dose*, not the small step - `-0.0371 -> -0.0326` is a wash and `-0.0326 -> -0.2097` is the step (z = 8.2 between the ends) - and at width 48 the **churn harm vanishes**: null-vs-random is 266/576, mean `+0.0152`, p `0.97`, the first cell in the program where churning does not pay (0.67-0.69 win rates at both narrower cells in the same regime). So the picture sharpens to **depth gates, capacity scales**, and §10.1's churn axis is now qualified in width. |
| **Result** | **1224 tests pass**; pre-registrations `38f3921` (5.48) and `e5dcbf6` (5.49); the four summaries (`width-d3-equal-576.json`, `width-d4-equal-576.json`, `width-d3-dose-576.json`, `width-d4-dose-576.json`) pin every number, the branch, and the two descriptive trends. |

## The campaign's plan had never been composed, and the null queue was priced at dev fidelity (2026-09-20)

| | |
|---|---|
| **What was found** | The three-queue Gate 4 campaign has been running with its total cost stated only in prose, and the prose was wrong twice over. `RESEARCH_REPORT.md` §5.14 and the runbook priced a declared null seed at "~90 s of GPU (~18 min over the queue)" - a figure measured at *dev* fidelity (400 steps, dev shape: 102.9 s mean wall over the 24 sessions of `deployed_vs_null_study`, run 2026-09-20). A null session launched by the runbook's own command inherits the declared config (only `--profile` shrinks it), so a null seed trains the same 24,576,000 tokens as a pair arm: 1.343 h at the T4's measured 5,083.4 tokens/s plus the 0.1 h event overhead = **~1.44 h**, ~58x the quoted figure. And the recorded "~47 GPU-hour plan" was below the training-only floor of the corrected rate (3 queues x 12 x 1.443 h = 51.9 h): it predated the placement correction and the null queue's true cost. |
| **Why nothing caught it** | There was no composed plan to check. `session_pair_capacity` priced one session's packing, the launch artifact carried `pairs_per_session_rule1` / `sessions_for_primary_rule1` for the primary queue alone, and the three-queue total existed only in sentences - which is exactly where the two stale numbers survived four stretches of verification. |
| **The fix** | `three_queue_plan` (`session_plan.py`) composes the campaign from the runner's own start gate: primary, pool-matched control and null, each priced by rule 1 and the declared token budget. The launch artifact now carries `schedule.three_queue_sessions` / `three_queue_training_hours` (with the primary's and null's session counts beside them), and the dtype probe prints the campaign line next to each dtype's per-queue numbers, so the number a human reads and the number a budget uses cannot come apart. |
| **Corrected** | Report §5.14 (both passages, with the 90 s figure's provenance recorded); runbook §0.5 ("six with one arm per device" -> five, "two sessions" -> three, the ~47 -> ~52 + nine sessions), §2c (the null queue's real per-seed cost), and the superseded 2026-09-15 derivation marked in place; the status entries carrying the old figures now carry the correction beside them. |
| **The numbers now** | At the measured 5,083.4 tokens/s placement rate: pair 1.443 h, 5 pairs/session by rule 1, 3 sessions per queue (5+5+2) - **nine sessions and a ~52 h training floor for the campaign** (~55 billed with boot/restore/publish), about two weeks against the 30 GPU-h/week quota. On a one-device host the two pair queues halve their packing (2 pairs/session, 6 sessions each) while the null queue keeps 5/session - 15 sessions, 84.2 h training - because one arm is one wave. |
| **Pinned** | `tests/test_session_plan.py` asserts the composition (three symmetric queues, the null queue at full rate, and the falsifier `training_hours > 47.0` - any composition that prices the null queue as an afterthought fails), the one-device asymmetry, and that a zero-capacity plan is refused where it is composed. `tests/test_v0_session.py` pins that a declared null config trains 24,576,000 tokens. The probe's new campaign line is asserted in `test_kaggle_t4.py` at both placements. |
| **Result** | **1195 tests pass**, tree clean, campaign session 1 still training. |

## The next gate's two candidate arms, read against the machinery that exists (2026-09-20)

| | |
|---|---|
| **Why** | §10.4 established that the Gate-4 search cannot reach the lever the other track measured (residual width, −0.1562 nats at its factorial's exact floor), and left "what would the next gate have to build" open. Read against the code, that splits into two claims needing different work. |
| **(A) Width as a gene the search is given** | Needs a **design, not a mechanism**: the zoo already crosses width at a matched budget (`arch-zoo-100m-v3-crossed`, 24 runs / 3.28 GPU-h, `crossed_contrast` at exact floors; `head-paired` matches pair halves exactly). Running the *same controller* against substrates fixed at two widths answers whether its value depends on the width it was given — which is also the covariate open question (ii) needs to tell depth from capacity. Costable from committed per-run figures before any hours. |
| **(B) Width as a gene the search can move** | The expensive half, and the code says why: `rebuild_v0_moe_from_genome` transfers "every other parameter ... when shape-compatible", and a width change makes **every** parameter in the stream shape-incompatible — nothing transfers, so the model is re-drawn and the mutation is a restart wearing a mutation's name. The genome and the budget checks already exist; **no transfer policy exists anywhere in `src/act/model/`** (no slice, zero-pad or interpolation helper). That is the whole of the missing work, plus the shape-dependent optimizer state and the per-checkpoint payload. |
| **And the failure mode is already priced** | The one shape-incompatible transfer this project has measured is a *single* tensor — the router, whose shape moves with the expert count — and §5.24 exists because re-drawing it discarded the learned routing function mid-training. §5.40 then measured the residual cost with the repair on: doing nothing still won every fresh seed by 0.136 nats. A width mutation without a transfer policy pays that class of cost on every tensor at once, so it would demonstrate the *mutation's* harm rather than the *search's* value. |
| **The order the machinery implies** | (A) first: it is answerable today, and it decides whether (B) buys anything. Neither may be added to a running queue (the §5.5 discipline) and each needs its own pre-registration. Written up as report §10.5. |
| **Result** | **1188 tests pass · 274/274 claims agree · 9/9 records reproduce.** |

## The exit publish is now recoverable: the per-seed payloads ride the kernel output (2026-09-20)

| | |
|---|---|
| **The risk this closes** | The dataset push is the durable copy of a session's work, and it is the one step that can fail *after* the hours are already spent: a session that trains five pairs and then fails to publish leaves the driver with an incomplete queue (exit 1/4) and nothing to read back. Every number the verdict needs per seed — each arm's final loss, search FLOPs, mutation count — is in a kilobyte `seed-summary.json`, so those files are now copied into the kernel output at session end, under ``seed-<n>-summary.json``. Checkpoints (gigabytes) stay dataset-only by construction; the *decision* survives a failed push even when the weights do not. |
| **The bug the first version would have had** | Every published seed writes a file called `seed-summary.json`, so keeping them under their own names would keep exactly one and silently drop the rest — a 12-seed block would look like a 1-seed block. The target name carries the seed, and the test that pins it fails on the naive version. |
| **A second collision, found by running it against the real layout** | The publisher writes each seed's summary twice — under the seed *and* under its `reports/` directory — so a plain walk finds both, and the `reports/` copy lands under `reports-summary.json` where one seed's data sits under a name that belongs to whichever seed sorts last. The walk now takes only the `seed-<n>/` copy; the test builds both copies in its fixture, so the failure is reproduced locally rather than discovered at 03:00 in a Kaggle session. |
| **Mutants** | The naive naming fails 1 test; keeping the `reports/` copy again fails 1; deleting the call from `mode_session` fails 1 — the helper is exercised directly *and* the call site is pinned at bytecode level (`_keep_seed_summaries` must appear in `mode_session.__code__.co_names`), because an unwired helper is the same failure as no helper and a source-text match would pass on a commented-out call. |
| **Result** | **1188 tests pass · 274/274 claims agree · 9/9 records reproduce.** |

## The three-queue read-back path, validated on real published data before the real data lands (2026-09-20)

| | |
|---|---|
| **What was run** | The exact path each queue executes — benchmark artifact → dtype rule → pair time → runner → publish — with today's edits in place: `--profile` pair-session smokes on the idle local card for the **primary** (both arms), the **control** (`--probe-selection random`) and the **null** queue (`--session-arms null`), each publishing `act-checkpoints/v0-5-1/seed-*/seed-summary.json` the way a session does. Then `scripts/v0_verdict.py` was pointed at all three trees at once. |
| **What it proves** | The artifact reader accepts the artifact the benchmark now writes (including `pairs_per_session_rule1` / `sessions_for_primary_rule1`); a pair completes and publishes for all three queue shapes; the null arm's published payload books **0 mutations and 0 search FLOPs** — the §5.14 built-in checks, on real published data rather than on a fixture; and the verdict evaluates **all three clauses** (`harm_guard`, `pool_matched_control`, `null_control`) from three separate roots, pairing the null and control arms cross-queue per seed. |
| **A guard fired on the way, as designed** | Running the *real* benchmark at the declared 285M shape on this card produces no artifact at all: fp32 measures 1,817.5 tok/s (below the 2,500 floor, so the rule selects fp16) and fp16 measures 1,005.1 tok/s on Pascal, giving a **13.68 h pair against the 9 h window** — so `unusable_schedule_reason` refuses to write a plan no session could execute. That is §5.2/5.39.1's guard doing its job, and it is why the smoke artifact was built at a T4 rate. |
| **And the verdict tool reproduces §5.41 from the other side** | The smoke verdict is a cost **FAIL** with `run_scale`/`band_reading` present: at a profile-scale run the cost criterion cannot pass, because the band is derived from the declared shape. That is the effect §5.41 documented, now seen end-to-end through the tool that will read the declared block. |
| **Where it ran** | Entirely under `/tmp` (dataset roots, session dirs, outputs, `/tmp/smoke-benchmark.json`), so nothing in the repository moved: the committed state is the entry below. |
| **Result** | **1185 tests · 274/274 claims agree · 9/9 records reproduce.** |

## The Gate-4 null clause was priced before its data exists: a withheld verdict is evidence, a cleared one is not (2026-09-20)

| | |
|---|---|
| **Why this, now** | §5.14's do-nothing clause is what authorizes or withholds Gate 4, and it is a *rule*: how much it decides is a property of the rule and the noise, not of the run. Its declared block (15000-15011) is still training, so this reading was taken **before the data exists** — the interpretation of whichever verdict lands is fixed in advance rather than chosen afterwards. |
| **The rule, read from the verdict's own constants** | `NULL_WIN_LIMIT = 10` of 12: the clause withholds Gate 4 only when the adaptive arm loses-or-ties against the null on **≥10 seeds**, with ties counting against the arm seeking authorization. So a **pass needs only 3 of 12**. |
| **The level** | Under no effect the clause fires on **79/4096 = 1.93%** of blocks — a cleared clause therefore happens **98.07%** of the time when adaptive is no better than doing nothing. |
| **It is one rule, applied to three arms** | The verdict's harm guard, its pool-matched control clause and this null clause are the *same* 10-of-12 sign test on the adaptive arm, differing only in the comparison arm (random, pool control, null). Named as one table in the script, with an assertion that all three limits are equal — so giving one clause a different limit stops the analysis rather than silently describing a verdict that no longer exists. Each clause's level is 1.93%; their union is **at most 5.79%** by the union bound, and less in fact because a hard block fires them together (positive dependence). |
| **They are not redundant, and that is measured** | On §5.40's dev-fidelity seeds the **null clause fires** (the null beats adaptive on 12/12) while the **harm guard does not** (the random arm beats it on only 6/12, agreeing with §5.40's own count), and the pool-matched control arm has never been run beside them — which the reading states rather than assuming away. So a verdict can withhold on one clause and clear another: they are three conditions, not one. |
| **The power, anchored on measured data** | At the dev-fidelity effect §5.40 just measured (the null ahead by 0.13554 nats at sd 0.11288, 12/12 fresh seeds → z = 1.20) the clause fires with probability **0.848**; at z = 1.5 it is 0.958 and at z = 2.0 it is 0.998. The anchor is *read from the committed §5.40 summary*, not assumed. |
| **The reading, stated before the run reports** | (i) A withheld Gate 4 is **strong evidence** — at the only effect size this program has measured, the clause fires ~85% of the time, so it is not a coin flip that landed against the adaptive arm. (ii) A **cleared** clause is **not evidence of value**: clearing needs 3 of 12 seeds and 98% of no-effect blocks clear it, so if the clause does not fire the value claim rests on the registered cost/band/control rules exactly as registered, and the write-up must say "the clause did not fire" rather than "adaptive beats doing nothing". |
| **Where it lives** | `scripts/null_clause_power.py` (imports `NULL_WIN_LIMIT` and the declared block from `v0_verdict`, and the anchor from the §5.40 summary, so neither can drift), artifact `artifacts/null-clause-power.json`, report §5.14. |
| **Mutants** | The clause limit relaxed to 8 fails all four tests (the analysis can no longer describe the rule the verdict applies); halving the anchor's z fails 1; hand-restating the pass bar fails 1; changing a number in the artifact's **prose** fails 1 — the artifact is compared to what the script computes *now*, whole, so a stale or hand-edited field cannot look correct while the code says otherwise (the field-by-field version of that test passed with the pass bar mutated). |
| **Result** | **1185 tests pass · 274/274 claims agree · 9/9 records reproduce.** |

## §5.40 ran, and the registered prediction held — the closure survives the deployed configuration (2026-09-20)

| | |
|---|---|
| **What ran** | The block registered on 2026-09-16 and then stopped after 5 of 24 sessions: twelve **fresh** seeds (15240-15251), each a pair session (`random, adaptive`) plus a `null` session, dev shape on the real TinyStories stream, 400 steps, three events, with the deployed treatment (`preserve_router_on_rebuild=True`, `new_expert_initialization="clone"`). **24/24 sessions, ~38 min of the local card**, run detached. |
| **The blocker it was waiting on was false** | The earlier entry records that a long run "cannot be made to outlive its launching shell in this environment", so the block could only be bought in foreground chunks. Tested rather than assumed: `setsid nohup` detaches from the launching shell — the same technique the Kaggle queue driver has used all along — and the block ran to completion in the background while other work proceeded. The registration's cost estimate (~200 s/seed) was accurate. |
| **The verdict** | **Branch (b): the closure holds under the deployed configuration.** The do-nothing arm has the **lowest final loss on 12/12 fresh seeds**: `null - adaptive` = **−0.1355** (0/12 seeds in that direction, p = 1.0), `null - random` = **−0.1457** (12/12, p = 0.0005). Both registered predictions were met — P1 fails, P2 passes — so §5.13's closure is *re-established on the configuration the project actually ships*, and the repair removes the mutating arms' disadvantage without giving either an advantage (`adaptive - random` = −0.0101, 6/12, p = 0.61 — the diagnostic's +0.0007 seen again). All eleven built-in checks hold, including the registration's own post-hoc table recomputing from committed rows. |
| **What it means for Gate 4** | The declared run's quality premise now rests on the toy factor map plus the compute-efficiency protocol alone, with **no transfer claim at all** — at dev fidelity, at a depth the toy prior favours (6, against the cell's depth >= 4), with the treatment in place, doing nothing still wins every seed. The 5.14 null clause is therefore the operative bar rather than a formality, and the most likely verdict is the small-scale one. Report §10.1/§10.2/§10.3 updated to say so. |
| **A sign convention, inverted twice** | The registration's own sentence says "deltas are `null - arm`, so negative = the arm beats doing nothing". Every other use in the program (5.39's summary and its test, and 5.40's own diagnostic narrative — "the adaptive arm's harm falls from 0.248 to 0.117" for `null - adaptive <= 0`) reads the same delta as **negative = the mutating arm is worse**. §5.39.1 was the first incident; this is the second, and it is corrected in the report rather than left standing. |
| **And the first summary printed the inverted branch** | P1 and P2 are opposite claims about one delta, so they need opposite tails; the first run tested both on the churn-harm tail, making P1's `established` mean "the null is significantly better than adaptive" — P1 *failing* — and printing **branch (a), the real-text leg reopening**, on a block where the null won every seed. The numbers were right and only the label was wrong, which is why nothing else caught it: the artifact's own `per_seed` rows contradict its branch. Fixed at the source (per-primary tails, extracted `branch_from_primaries`, the convention restated in the payload with a note that the registration's sentence inverts it), and pinned by `tests/test_deployed_vs_null_summary.py`: the convention asserted convention-free (the null is the per-seed argmin), each primary's tail, the branch re-derived from the stored comparisons, and the constructed failure. |
| **Mutants** | Re-running the original bug (P1 back on the churn-harm tail) fails **3 tests** and writes branch (a) — the false headline is now loud. Flipping one seed's adaptive loss fails 4; forcing `prediction_met` to "passes" fails 1; making the derivation ignore P1 fails 1. |
| **Result** | **1179 tests pass · 274/274 claims agree · 9/9 records reproduce.** |

## The launch's own plan was priced for a one-GPU host (2026-09-20)

| | |
|---|---|
| **What was found** | The dtype probe's plan line — the number this project quotes for how long the declared run takes — was priced with `PairSchedule`'s default of **one device**, while the session's benchmark and its runner both resolve the host's placement (`arm_slots`). On Kaggle's T4 shape that is 2 devices, so the plan a human reads was **twice** the run the launch executes: the probe printed **2.871 h per pair / 3 pairs per session / 4 sessions for the primary queue**, against the true **1.443 h / 6 / 2**. |
| **Confirmed on the host, not inferred** | `--queue selfcheck` (one shot, ~1 min of the window) reports `host devices: 2`, `one arm per device: cuda:0, cuda:1`, and *all checks passed*; the re-run probe now logs `arm placement: 2 arm(s) in 1 wave(s) of slots=2 on cuda`. |
| **The re-run, with the corrected plan** | fp32 **1,451.9 tok/s** → 4.802 h per pair → **1 pair/session, 12 sessions** (an fp32 run is not merely slower, it is 6× the sessions); fp16 **5,083.4 tok/s** → **1.443 h per pair**, **6 pairs/session registered, 5 by the runner's own start gate, i.e. 3 sessions for the primary queue (5+5+2) against the 2 the registered field reads**. fp16 is **3.5×** the fp32 rate on the same host, so the pre-committed dtype rule selects it again — nothing about the decision moved, only the schedule it is priced at. |
| **And the second half of the same gap** | The plan's capacity and the runner's capacity are different quantities, and the docs read only the first. ``full_pairs_per_session`` is §5.2's arithmetic (``floor(usable / pair time)``), which the §5.2 table's rows are computed from and which the tests pin; the runner's rule 1 adds the 45-minute publish reserve to *every* start, so it begins one fewer pair whenever that reserve binds. The runbook asserted the opposite ("the plan's own arithmetic already includes the 45-minute publish reserve"), so the operator-facing expectation was one session short. Corrected at the source instead of in prose: `session_pair_capacity()` walks the deadline rules themselves, the benchmark artifact now carries `pairs_per_session_rule1` / `sessions_for_primary_rule1` beside the registered fields, the probe's plan line prints both, and the runbook says which one to plan from. Pinned by `test_the_registered_capacity_and_the_runners_gate_differ_when_the_reserve_binds`, which also shows the reserve is the whole reason (lend the session those 45 minutes of window and the two agree). |
| **Why it mattered** | Every plan number keyed to the probe was 2× too large: the primary queue's session count, the wall-clock to the primary verdict, and the window the driver's quota gate compares against. Runtime was never affected — the runner derives the pair time itself, so sessions advance at the placement-aware rate — but the plan a reader plans from was wrong in the direction that makes the run look less affordable than it is. |
| **Fixed** | The probe's plan line now comes from one function, `probe_pair_schedule`, which prices the host it measured, while the dataclass default keeps a genuinely one-device host honest. Pinned by `test_dtypeprobe_prices_the_placement_the_runner_will_use`, which also fails if the report stops quoting the numbers it computed — the first version of that test passed with the probe reverted, because it pinned the helper but not its use. |
| **Result** | **1173 tests pass · 274/274 claims agree · 9/9 records reproduce.** |

## The Gate-4 search cannot reach the one gene the other track measured as the lever (2026-09-20)

| | |
|---|---|
| **What was found** | The two tracks of this program have never been read against each other, and read together the gap is the reachable operation set. The arch-search phase measured, at a matched ~100M budget on real text, that the loss lever is **residual-stream width** (−0.1562 nats at its factorial's exact 46,656-assignment floor), that head count is parameter-free and non-significant (−0.0177, p = 0.14), that depth is real but *shallow wins*, and that the **active-parameters hypothesis is refuted** (×4.51 spread at ρ = +0.214, p = 0.62, with the fully dense design 7th of 8). The V0 session's entire search space is `feasible_expert_mutations` — grow/prune of experts — so every candidate the declared block evaluates moves the axis that phase found *uncorrelated* with loss, and none can reach the axis it found as the lever: no operation sets `hidden_size`, head count or depth, and the only assignment to `hidden_size` in `src/act/topology/` is the restore on a rejected change. |
| **What it does not mean** | Gate 4's question is about *selection* rather than about which gene — does adaptive selection beat doing nothing on the declared block — and §10.1's depth axis says selection value exists above an onset, with §10.2's null clause as the right control for it. What a pass licenses is what is bounded instead: "adaptive control of expert capacity at 15000-15011", which is not the same claim as adaptive architecture search and cannot move the other track's recommendation, because the two search spaces share no gene. |
| **Why it matters to the next gate** | The arch phase's width effect is the largest single effect either track has measured at a matched budget, and it is the covariate this program's own open question (ii) needs in order to separate depth from capacity: `arch-zoo-100m-v3-crossed` is the program's only matched-budget separation of depth from width (ρ(depth, hidden) = 0.000, balanced 2×2) and it answered *width*. A width operation is not free engineering — it reshapes every weight in the stream, so it needs the treatment `grow_expert` already gives a new expert, with re-initialisation and a parameter-budget check — which is why it is a deliberate decision after Gate 4 rather than an edit to a running queue. |
| **Written where a reader will find it** | New §10.4 of the research report, with the four-gene table, placed after 10.3's open questions because (ii) is the question it feeds. |
| **Pinned** | `tests/test_topology_transactions.py::test_no_reachable_operation_can_move_the_gene_the_search_phase_measured` — the pool's operation vocabulary is a subset of grow/prune, and applying every candidate leaves `hidden_size`, depth, `heads`, `kv_heads` and `ffn_width` unchanged — so the claim fails loudly if an operation is ever added rather than quietly going stale. |
| **Result** | **1172 tests pass · 274/274 claims agree · 9/9 records reproduce.** |

## The declared Gate-4 entry run is in flight, on a schedule measured before the spend (2026-09-20)

| | |
|---|---|
| **What was launched** | The project's critical path, per §10.2 of the research report - "Gate 4 is prepared, amended, and unspent ... the run needs a T4" - is now running on Kaggle T4: the pre-registered §5.1 compute-efficiency protocol on the declared 285M V0 config, three queue-kernels (`session`, `control`, `null`) over seeds **15000-15011**, publishing to `act-v0-5-1`. `kaggle_queue.py --queue session --max-sessions 6` is driving the primary queue detached (`artifacts/v0-5-1-launch/session.log`); nothing had been pushed since the arch-zoo kernel. |
| **Pre-flight, before any spend** | `verdict_rehearsal.py --check` applies the real verdict to every withholding clause, both absent-queue and both short-queue cases and the label-precedence combinations, and all fourteen variants match their registered label and exit code (~1 s, no GPU; the artifact it rewrites is byte-identical, so the check is idempotent). Datasets present, quota read **25.16 h remaining**, and the code dataset was republished at `92f86c0` so the pushed kernel ran this tree rather than a stale copy. |
| **The dtype probe, which decides whether a session can advance at all** | Measured on the T4 before the queue was pushed: **fp32 1,401.5 tokens/s** -> 9.842 h per pair -> **0 pairs fit a 9-hour window**, i.e. an fp32 session would idle its deadline and burn the window producing nothing - the exit-4 loop the driver refuses to re-push on. **fp16 4,926.4 tokens/s** -> **2.871 h per pair, 3 pairs per session, 4 sessions for the primary queue.** So the pre-committed throughput-gated rule selects fp16, it is **3.5x** the fp32 rate, and §5.2's committed schedule (3.585 h per pair, 6 sessions) is conservative against the measurement rather than optimistic. **(corrected 2026-09-20: those plan numbers were priced for a one-device host — see the entry above. The measured rates stand; the pair time is 1.443 h, and at that rate the primary queue needs **3** sessions, not 6 or 2 — see `sessions_for_primary_rule1`.)** |
| **What the run decides** | A pass authorizes Gate 4; a null-clause rejection closes §5.1's selection premise at its own scale; and the cost measurement is the registered blocker on scaling. The strongest real-text fact measured so far points the other way - at diagnostic fidelity the do-nothing arm beat every mutation policy, and the rebuild repair removes only 31% of the churn harm - which is why the declared run is the decisive spend rather than a formality. |
| **Why not the §11 comparison instead** | The arch-search phase's only un-run item is measured to be undecidable at its own size (§9a: 13 candidates per set needs a 0.30-nat effect, and its coverage form is invisible at every size priced), so it cannot be the next decisive spend. |
| **Verified** | The launch is quota-gated and resumable by construction: one session per queue, atomic pairs, deadline stops with in-progress snapshots, restart-from-scratch, skip-completed idempotence, and a published checkpoint dataset as the durable copy. The driver reads the weekly quota before every push and refuses when the remaining window cannot cover one. |

## The loop's price at a real horizon double-counted its confirmation (2026-09-20)

| | |
|---|---|
| **What was found** | §10 prices the search loop at two confirmation lengths: **17.0 GPU-h with the confirmation at 1,200** and **29.6 on Kaggle / ~49 on this card** with it at a 12,000-step horizon. The second figure is wrong in composition: 16.99 + 12.57 = 29.57, so it adds the horizon confirmation to the *complete* 1,200-step loop — which already contains a confirmation of the same 24 finalists — while every sentence around it says the length is a substitution ("with the confirmation at 1,200" / "with it at the horizon"; README: "what that loop costs depends entirely on **where** it confirms"). The totals are **26.4** (13.80 + 12.57) and **45.7** (13.80 + 31.92). |
| **Why nothing caught it** | The audit encoded the same composition: its claim recomputed `100×3×0.046 + 8×3×0.133 + 24×0.524 = 29.57` and compared it against the record's 29.56, so the double count was checked *against itself* and agreed. Reverting both the record and that formula is still a non-biter by construction — which is why the composition is now checked against the **driver's own planner** instead of against prose: `--plan screen` is one screen stage and one confirmation stage, and the stage whose length varies is the confirmation. |
| **Two clauses were wrong with it** | "the confirmation alone costs more than every step before it" (§9) and "the confirmation is the largest single term in the latter two" (§10) hold only on the local card: 12.57 is **91% of** the 13.80 screening half on Kaggle, and 31.92 is 2.3× it on this card. Both are now stated that way, because the point they carry — a real horizon makes the last step set the price — survives the correction while the superlative does not. |
| **The rates under every projection were unchecked too** | Both local-card rates are pooled means over the ledgers that measured them: **0.046** over the phase's 50 runs at 400 steps, **0.133** over v2's 24 flat runs at 1,200. Neither was checked, though §10's 17, 40, 26.4, 45.7 and §9's whole price list are derived from them. Added with the halves (13.80 screening, 3.19 confirmation), the quoted 17, and the corrected horizon totals — each at the precision the record quotes. |
| **And §9a's launch pre-flight** | "39 runs ≈ 5.92 GPU-hours" for the random arm is the driver's own plan projection, now recomputed from it — 13 candidates × 3 seeds flat, 95.8M tokens at the driver's conservative 4,500 tok/s — so the arm's estimate is checked before the hours rather than after. |
| **How the plan is driven** | `arch_zoo_screen.parse_args` now takes an optional `argv`, so a plan can be built and costed without a shell and without mutating the test process's own command line. The composition claims read the plan the same way `main()` does, including the candidate count from the flag — a literal `100`/`13` in the check would have made the candidates flag an equivalent mutant. |
| **Verified** | Five mutants caught: the loop planned with its confirmation at the screening length, the plan confirming 3 survivors instead of 8, the pre-flight arm sized at 26 candidates (which failed to bite until the check read the flag), the 400-step rate quoted without its half-a-place tolerance, and the 1,200-step rate read from the wrong arm. **263 → 274 claims.** |
| **Result** | **1171 tests pass · 274/274 claims agree · 9/9 records reproduce.** |

## §9a's readiness claims had no machine check, including the one that decides the statistic (2026-09-20)

| | |
|---|---|
| **What was found** | §9a states four things that decide whether the §11 comparison can be run at all and what a result would mean: that the curated sets carry no location advantage (the non-circularity condition), that equal compute is a statement about a *budget*, that the budget-matched draw is not a designed set in disguise, and that the coverage form is invisible — which is why the *set mean* rather than the winner is the pre-registered statistic. The audit recomputed §9a's mean-form power, its budget match and its typicality percentiles, and none of these four. |
| **The percentile table, whole** | Every number reproduces from the zoo definitions at the tool's own 4,000 draws: 1,059 cells at depth 8.97/3.35 and hidden 718/143; `v1` at p83/p16/p19/p64, `v2` at p68/p96/p22/p98, `v3` at p72/p100/p42/p100 — values and percentiles both. Percentiles are compared as the tool *renders* them (whole percentages) and exactly, because a relative tolerance is the wrong test on a rank: 5e-3 of 16 is 0.08, a third of a draw. `v1`'s 16th-percentile depth spread and 19th-percentile hidden size are the two numbers the non-circularity argument rests on. |
| **The budget paragraph, whole** | The curated set's 13 candidates total **1,121,409,984** and span 70.83M–123.30M (×1.7408, mean 86.26M) — checked exactly, where the audit had taken that total on trust as the denominator of every ratio beside it. The band counterexample reproduces: the *same* seed 151 at the driver's own 0.15 draws 1,282,387,008, **1.1435×**, which is why the band is part of the launch line rather than a default. And the matched draw is not the designed sets' mechanism in disguise — its structure (7.85/3.21, 718.8/137.8) sits beside its own band's population (9.49/3.50, 715.6/143.7). |
| **The reason the mean is the statistic** | A 0.60-nat better winner — twice the effect the mean form is priced at, and four fifths of the whole 0.74-nat candidate spread — leaves the `best` statistic at **0.03 power** at 13 candidates per set, **0.02** at 26 and **0.04** at 52, every one at or below the nominal 0.05. Doubling the advantage from 0.30 to 0.60 leaves the rate where it was, **0.03 both times**, so the coverage form is unsensitive rather than merely underpriced — the structural argument (half the assignments put the improved member on either side) read off the shipped test instead of asserted. The contrast that makes it informative: the same statistic sees a set-wide 0.30-nat shift at **0.94**, and an inert statistic on that design reads 0.430 against the asserted 0.8. |
| **Two comparison rules the audit did not have** | A percentile is a rank rendered to a whole percentage, so it is compared as rendered and exactly. A value quoted to one decimal carries **half a unit in the last quoted place**, not half a percent: the matched draw's depth mean is 7.846 and the record's 7.8 is that rounded — dropping the explicit tolerance makes the claim fail, so the tolerance is load-bearing rather than decorative. Both rules are pinned in the audit's own test. |
| **How it is checked** | `check_space_typicality` rebuilds the enumerated population and re-reads the three zoo definitions; the budget additions extend `check_structure_and_budget`, whose quick path still skips the 200-seed scan; and `check_set_comparison_coverage` recomputes the power figures through the shipped tool, sharing one parallel power path with `check_power_study` — generalized to take the model and statistic instead of hard-coding `shift-all`/`mean`, so a future reading cannot be computed on the wrong one. |
| **Verified** | Eight mutants, all caught: the curated totals read from the wrong definition (six claims fail), the band counterexample drawn at the matched band, the one-decimal tolerance dropped, the draw compared against the wrong population (three of four fail), the parallel power path ignoring its model/statistic (the coverage claim fails at power **1.000**), the contrast row computed on an inert statistic, the 52-per-set row transcribed as 13, and the coverage check dropped from `CHECKS` (two tests fail). The three power substitutions fail *loudly* with a `KeyError` rather than passing silently. **212 → 263 claims**, with no measurable wall clock on either the suite or the audit — both are bounded by their slowest member. |
| **Result** | **1171 tests pass · 263/263 claims agree · 9/9 records reproduce.** |

## The fifth rule gets its reading check before its arm is bought (2026-09-19)

| | |
|---|---|
| **What was done** | The reading check covered four of the phase's five pre-registered rules, and the fifth — the §11 random-vs-manual comparison — was the one that has not run. It now runs through the same shared path, and `set_comparison_decision`'s record renders the same table. |
| **Why that ordering matters** | Wiring a check in *after* a result is how a declared threshold becomes a chosen one, which is what §11's pre-registration exists to prevent. The check is part of the declaration rather than a response to anything, and by the time §11 runs, two of the four existing rules will have shown it costs something: one of them is §2c's own attribution. |
| **The mechanism had to generalize** | §11's evidence is a two-set test built from both arms' rows, not one of the zoo outcome's own blocks, so `rule_reading_stability` now takes either the outcome-block path or a caller-supplied `read(rows)`. One shared path, one set of substituted fields, one set of thresholds. |
| **One renderer, not two** | The reading table was inline in the zoo decision and absent from the §11 renderer; it is now `reading_stability_lines`, used by both. A second spelling of the table is how two arms end up describing one check differently. |
| **Verified** | Three tests on synthetic planted-effect designs — a table rendered, a branch that moves when a per-set constant is planted on the trajectory loss, and no claim at all without a reconstruction — plus the bold marker pinned on a constructed dict so the disagreeing row stays readable. Four mutants caught, including a renderer that prints the verdict without the table. **1166 → 1170 tests.** |
| **Result** | **1170 tests pass · 212/212 claims agree · 9/9 records reproduce.** |

## Every pre-registered branch now names the loss it was read on, and one of them moves (2026-09-19)

| | |
|---|---|
| **What was done** | The reading check built for the head-pairing rule was generalized to all four pre-registered rules (`replication_decision`, `horizon_decision`, `head_pairing_decision`, `finalist_order_decision`) through one shared path, so the four cannot describe the same check in four different ways. Each names the outcome blocks it actually reads, because rebuilding a block a rule never looks at costs the record's 200,000 draws and cannot change its branch. |
| **The finding it turned up** | The crossed arm's pre-registered replication rule — the branch that says §2c's attribution replicates — lands on **ATTRIBUTION FAILS TO REPLICATE** when read on the trajectory-averaged loss. Nothing about the attribution moved: both main effects keep their signs and sit at the design's Monte-Carlo floor, and the interaction stays at its 2/2⁶ floor with all six seed contrasts agreeing. Only `|hidden| ≥ 1.5 × |depth|` stops holding — 0.2255 against 0.1814, a ratio of **1.24× against the declared 1.5**. |
| **Why the condition matters, not just the label** | Both facts share one label and are different claims. The ratio is a statement about end-of-training magnitudes, and the trajectory average weights the early phase — where depth's penalty is evidently larger — more heavily (depth grows 2.20×, width 1.22×). So the readings record the condition that moved, and the record's clause names it, rather than reporting that the attribution failed. |
| **Not a six-seed accident** | The fresh three-seed confirmation arm, which shares no seeds with the pooled set, narrows its own ratio 0.1851 vs 0.0865 → 0.2265 vs 0.1879 (**1.21×**) and lands on the same alternate branch. Two independent seed sets reaching the same reading is what makes it a property of the loss. |
| **The two branches that do not move** | The horizon arm's depth-sign headline is stable on all three readings with no condition moving, and the finalist-order branch is stable while its *second* condition flips (the screening-free blocks clear 0.05 at 1/36 on the trajectory average and not on the like-for-like pair) — the case a label-only check would have called settled. |
| **Cost** | Marginal and opt-in: 0.01 s for the horizon rule, 0.16 s for finalist-order, 2.7 s for replication and 3.5 s for head-pairing, each against the analysis it belongs to. Unconditional rather than gated on a margin looking thin, because a silent skip is indistinguishable from a branch that held. |
| **Verified** | Four tests and eight claims. Two records regenerated from their own documented commands (`arch-zoo-100m-v3-crossed`, `arch-zoo-100m-v3-crossed-confirm`); `check_record_freshness.py` catches a hand-edited reading table. Four new mutants caught, including a rule re-entering its own reading check, which was an infinite recursion until the shared path passed `reading_check=False`. **204 → 212 claims.** |
| **Result** | **1166 tests pass · 212/212 claims agree · 9/9 records reproduce.** |

## The record's own branch label now carries the loss it was read on (2026-09-19)

| | |
|---|---|
| **What was found** | The previous two stretches established that the head-paired arm's branch **reverses** on a trajectory-averaged loss, and wrote that into the findings record, the arm's NOTES and the audit - but the artifact a reader actually quotes, the arm's pooled `decision.md`, still printed the branch alone. A reader could try the second loss, get a different answer, and have no way to tell which one was pre-registered. The gap was the record, not the knowledge. |
| **The fix is in the generator, not the artifact** | `head_pairing_decision` now takes the pooled rows and, when it gets them, re-reads its own rule over the other losses those ledgers carry - same rows, only the loss field substituted - then returns the branch each reading lands on plus a sentence naming any that disagrees. The decision prints that as a table. Because the alternate readings are produced by calling the *same* branch logic, the two cannot drift apart: a change to the rule moves both. |
| **What the arm's record now says** | held-out `validation_loss` and `final_train_loss` both land on `HEAD EFFECT IS REAL AT WIDE WIDTH`; `mean_train_loss` lands on `INTERACTION ABSENT - THE THREE-SEED AGREEMENT WAS SEED LUCK` and is bolded as the reading that moves. The arm's own three-seed record is `open-floor-limited` and stable on all three, which is the informative null for this check. |
| **Not gated on the margin looking thin** | The first draft skipped the re-read when no comparison sat within 2× of its threshold. That is a silent heuristic whose failure mode is indistinguishable from a branch that held - the exact defect this stretch is fixing - so it was removed. The check costs **3.6 s** against the 6.2 s 200,000-draw analysis it belongs to, and a rule applied to an outcome *without* rows reports no stability claim rather than an unverified one. |
| **Verified** | Three tests and three claims, plus an artifact-level proof. Four mutants caught: substituting the held-out loss for every field (nothing moves), dropping the reading-dependent clause, suppressing the table, and claiming stability without rows. Stripping the table out of the committed `decision.md` makes it **stop reproducing** under `check_record_freshness.py`, so the record gate covers this too. **201 -> 204 claims.** |
| **Result** | **1162 tests pass · 204/204 claims agree · 9/9 records reproduce.** |

## Every significance headline re-read on training loss: the two that move are both negatives (2026-09-19)

| | |
|---|---|
| **What was done** | The head-paired audit left an obvious question - which *other* headline had never been re-read on the training loss - so all nine committed records were checked. Five arms now carry the split check (v2, crossed, horizon, head-paired, and this stretch's finalist-order rule); the audit filled in the four that did not. |
| **The positives hold everywhere** | v2's and the crossed arm's attribution, the horizon arm's depth sign, and the two spread-only arms (`v2-horizon` at 3× and `v3-crossed-confirm`) all keep their signs and clear their thresholds on every reading. `v2-horizon`'s spread is p ≤ 5e-06 on both; the crossed confirmation's is 1.5e-05 held-out and 5e-06 on training loss. |
| **Both readings-sensitive headlines are negatives** | `arch-zoo-100m-v1-finalists-6seed` is v1's "NOT ESTABLISHED — this experiment cannot yet tell": p = 0.1111 held-out, **0.0556** on final training loss (above the declared 0.05 by 11%), and **0.0278 = 1/36, the three-block floor**, on the trajectory average. So the verdict holds on the pair the check is designed around and is reading-dependent beyond it — the same shape as the head-paired arm's negative, in the same direction. |
| **The pooled finalist-order branch is stable; its explanation is not** | The pooled six-block rule that closed §9's open item stays **order-coupled** on all three readings — 10/7776 held-out, 82/7776 on final training loss, 2/7776 on the trajectory average, against a 1/7776 floor — because the deciding condition is that two different architectures top the six blocks, which is true in every reading. Its *second* condition, "the screening-free blocks alone clear 0.05", is 4/36 and 2/36 on the like-for-like pair but **1/36, the floor**, on the trajectory average, so the clause explaining why the pooled significance is trustworthy is the part that moves. |
| **Verified** | Two tests and six claims. Pointing the rule's declared `pooled_p` at 0.005 catches the branch-stability test, tightening `fresh_p` to 0.02 and repointing `FINALIST_ORDER_FRESH_SEEDS` at the coupled blocks both catch the screening-free test, and each captures the property the other does not. **195 -> 201 claims.** |
| **Result** | **1158 tests pass · 201/201 claims agree · 9/9 records reproduce.** |

## The head-paired arm's negative headline is the one place the split check moves a decision (2026-09-19)

| | |
|---|---|
| **What was found** | The head-paired arms are the phase's only *negative* headline - "head count is free" - and they are the only arms where **no test or claim read the ledgers under a training-loss reading at all**. Run the same check the other three arms carry and the picture changes: the parent arm's head contrast is non-significant held-out (paired p = 0.140, factorial p = 0.122) and **significant on final training loss (0.044)**, while its confirmation is the reverse (0.047 held-out, 0.159 on train loss). The four numbers straddle 0.05 in both directions. |
| **The pooled contrast is thin, and now named** | The six-seed pooled head contrast is **p = 0.0115 against `HEAD_PAIRING_RULE`'s declared 0.01** - a margin of **1.15×**, about six steps of that contrast's own 1/4096 sampling resolution. The record discussed this contrast as one whose threshold is cleared; the margin is what keeps the "uniform head effect" branch out, so it is a deciding number and is now checked. |
| **The branch holds on the like-for-like pair and reverses on a third reading** | Held-out loss and final training loss are both loss at the end of training, so they are the like-for-like pair: the rule lands on **HEAD EFFECT IS REAL AT WIDE WIDTH**, five conditions of five, on both. On `mean_train_loss` - which averages the whole trajectory, so it moves with convergence speed rather than fit - one of the six per-seed interaction contrasts is **−0.0050**, the seed-blocked interaction leaves its 2/2⁶ floor for 0.0625, and the rule reports **INTERACTION ABSENT - THE THREE-SEED AGREEMENT WAS SEED LUCK** instead. |
| **What does not move, and it is what the plan spends on** | Under all three readings the width contrast is negative and at the design's floor (−0.1617 / −0.1549 / −0.1865) and the H=896 level effect is negative with **all 12 units favouring more heads at the exact 2/2¹² floor** (−0.0553 / −0.0582 / −0.0344), while H=512 stays non-significant. So the recommendation rests on the level split, which replicates, not on the pooled contrast, which does not - and the arm's headline should be read with the margin named rather than as a null. |
| **Verified** | Four tests and seven claims. The branch-stability test fails when the rule's declared `main_p` is loosened or its label changed; the margin test fails on the loosened threshold; and flipping one of the 12 wide units moves the split from 12/0 at the floor to 11/1 at p = 0.052, which the pricing test catches. The new claims run at 20,000 draws rather than the record's 200,000 - verified to return each branch unchanged, and 4× cheaper - which took the four tests from 27s to 7.5s. **188 -> 195 claims.** |
| **Result** | **1156 tests pass · 195/195 claims agree · 9/9 records reproduce.** |

## The horizon arm carried the phase's headline and none of the arm's robustness checks (2026-09-19)

| | |
|---|---|
| **What was found** | The crossed arm's attribution is backed by two checks - a re-run on training loss and a leave-one-out over candidates - and §2c states both and tests pin both. The horizon arm, whose depth-sign result at 12,000 steps is §9's first headline, had **neither**: no test read its ledgers for robustness, and §9's horizon item gave the branch and its two qualifications without saying whether either check had been run. Neither had. |
| **The split check holds, and matters more at this length** | Re-running the identical factorial at 12,000 steps on `final_train_loss` keeps both main effects at the same signs and significant: depth **+0.0249** (p = 0.0077, against +0.0209 held-out) and width **−0.0667** at the same 1296-assignment floor. A longer schedule is where a divergence in how much each architecture overfits would widen, so this is the length where the check carries the most. |
| **One check can be carried, the other cannot** | The *spread* survives leave-one-out over all 8 candidates (worst blocked p = **0.0018**, two orders below the 0.05 threshold). The *crossed attribution* cannot be checked that way and **refuses on all eight**: at 2 candidates per cell, removing one leaves 7 rows that cannot fill four cells balanced on both seeds, so the contrast returns unavailable rather than computing over a subset no design supports. A 3-replicate design would support it; v2's 8-distinct-configuration arm did. |
| **And a check nobody had needed before shows two seeds is arithmetic** | The re-split test is two-sided, so the smallest p it can report is `2/assignments`. Two seeds give 1296 assignments and a floor of 0.001543; one seed has **36** and a floor of **2/36 = 0.0556 - above the 0.05 the rule reads against**. **No single seed can reach this branch at all**, which is why the stale one-seed record the landing left behind had no verdict to state rather than merely declining to state one. What *does* replicate per seed is the effect size (depth +0.0211/+0.0207, width −0.0618/−0.0673). |
| **Verified** | Four tests against the real ledgers, each proven to bite: removing the balance enforcement catches the refusal; reporting the floor as `1/assignments` catches the two-seed check; reading the split check on the wrong loss catches the sign/size assertion; and a constructed failure where the spread rests on one candidate (the other 7 tied within each seed) drives the leave-one-out p to **1.0** and catches the threshold. Six claims added to the record check, all recomputed from the arm's ledgers. **182 -> 188 claims.** |
| **Result** | **1152 tests pass · 188/188 claims agree · 9/9 records reproduce.** |

## The search loop's most expensive step had never been priced, because no arm had ever run it (2026-09-19)

| | |
|---|---|
| **What was found** | §10 prices the search program's loop at "~17 GPU-hours" - 100 candidates x 3 seeds screened at 400 steps plus 8 finalists x 3 seeds confirmed at **1,200** steps. §9 specifies the loop as confirming finalists **at the horizon the search actually trains for**, and that step had never been run by any arm, so it was never priced. The horizon arm now measures it at two lengths on one design, and the confirmation is the largest single term in the loop at any realistic horizon. |
| **The measured step** | Both rows are from this arm's own cost file, not extrapolated: **0.0523 h/run at 1,200 steps** and **0.524 h/run at 12,000**, over 16 runs each. The ratio is **10.016x for 10x the steps**, so cost is linear in steps between two *measured* points - which is what licenses the local-card extrapolation (0.133 x 10 = 1.33 h/run) rather than assuming it, and is worth recording because a sublinear rate would have made the confirmation cheaper than this. |
| **What the loop costs at each end** | Screening 100x3 at 400 steps is 13.80 GPU-h and confirming 8x3 at 1,200 is 3.19, giving the quoted 16.99. Confirming those 24 runs at the horizon instead costs **12.57 GPU-h on Kaggle T4 fp16 or 31.92 on this card** - so the loop is **17 GPU-h** with the confirmation at 1,200, **29.6** with it at the horizon on Kaggle, and **~49** on the local card. The confirmation is a bigger term than everything before it once the horizon is real. *(Corrected 2026-09-20: those two totals double-counted the 1,200-step confirmation - 29.6 is 16.99 + 12.57 and ~49 is 16.99 + 31.92, each adding the horizon confirmation to a loop that already contained one. The totals are 26.4 and 45.7; see the entry at the top for the composition and why the audit had agreed with it.)* |
| **The like-for-like caveat, now stated** | The "~40 GPU-hours for running all 100 at full length" that the 2.4x saving is measured against is also at 1,200 steps, and at a 12,000-step horizon that baseline is ~399 GPU-h. So the loop's saving is *larger* in ratio while costing far more in absolute terms - both of which a budget has to see, and neither of which a single "~17 GPU-hours" showed. |
| **Why this also settles a platform question** | The horizon phase ran on Kaggle rather than the local card for reasons its own NOTES gave (fp16 on Turing tensor cores); this is the price of that choice at the step where it matters most, and it is **2.54x per run** on the term that dominates the loop. Recorded here because the justification was written as a throughput ratio and is really a statement about the loop's biggest cost. |
| **Verified** | Seven claims added to the record check, all recomputed from the arm's cost file or from §10's own documented rates: both per-run costs, the linearity ratio, both platform confirmation totals, the loop total, and the platform ratio. **175 -> 182 claims.** |
| **Result** | **1148 tests pass · 182/182 claims agree · 9/9 records reproduce.** |

## The capacity sweep trained a third of its arms in order to throw them away (2026-09-19)

| | |
|---|---|
| **What was found** | `capacity_sweep.py` ran **7 trainings per (scale, seed)** and reported **4**. Every probe-selection row was configured `("random", "adaptive")` so that `probe_selection` - which governs only the adaptive arm - could tell the three rows apart, and the arm without that selection was kept only to be discarded: `adaptive` and `anti` each trained a random arm they never read, and `random` trained an adaptive arm it never read. The code's own note had flagged this as "a study-design question rather than this defect" and left it. |
| **Why it is safe, measured rather than argued** | `run_v0_session` trains each arm in its **own subprocess** from the same config and seed, so the arms are independent and removing one cannot move the other. Checked directly at the dev shape (120 steps, seed 42, argmin): the adaptive arm's validation loss is **bit-identical** - 9.031953811645508 - with and without its pair, at 22.5s against 43.8s. |
| **The end-to-end A/B** | The same 120-step command with all four arms, run both ways on the real card: **64.7s paired against 38.0s single-arm, 1.70x** - and the four rows agree exactly (`null 5.626723766326904 / random 5.6035860776901245 / adaptive 5.653473734855652 / anti 5.658488988876343`, mutations 0/1/1/1). The per-row timings show the mechanism: the three probe-selection rows cost 17.8-18.2s paired against 8.5-9.3s solo, while the null row is 9.3s either way. |
| **What changed** | `make_config` now sets `arms=(ARM_MODE[arm_name],)` - the session is asked for exactly the arm the row reports - and `ARM_MODE` is what the config is built from rather than a separate lookup, so a row cannot report an arm the session was not asked for. `ARM_CONFIGS` loses its `arms` entries. |
| **Verified** | A test pins the *config* rather than the saving, because the saving follows from the config and a pair that re-grew there would be invisible in every result file; it names one training per reported row. Caught when the pair was re-grown, and the A/B above is the independent check that the rows did not move. |
| **Result** | **1148 tests pass · 175/175 claims agree · 9/9 records reproduce.** |

## The promotion threshold silently borrows a shorter stage's noise, and the record claimed it was measured there (2026-09-19)

| | |
|---|---|
| **What was found** | Now that the horizon is measured, the search program's confirmation length is known - and the promotion threshold at that length needed a seed std that the table of measured stds does not contain. `seed_std_for_steps` takes the nearest measured length *at or below* the stage, which is the conservative direction for a short screen and is the **opposite** of what a long confirmation gets: at 12,000 steps it reuses the 1,200-step 0.0148, which is a **2.64x over-estimate** against the 0.0056 this arm's own 12,000-step rows measured. Safe, and costly: the threshold is the std times `sqrt(2 / seeds)`, so the borrow buys extra seeds rather than extra safety, and nothing in the output said so. |
| **The record asserted a provenance it did not have** | Worse than the number: the decision file's caption read "the measured per-architecture seed std **at that length**", and the table printed the std with no indication of which length it came from. A 12,000-step cut therefore read as a 12,000-step measurement. The std is now rendered with the length it was measured at whenever that is not the stage's own (`0.0148 (measured at 1200)`), and a stage measured at its own length stays unannotated so the note appears only where it is true. The new key is `std_measured_at_steps`, read from `seed_std_measured_at_steps` rather than inferred, so the two cannot drift. |
| **The measured figure is deliberately not substituted** | 0.0056 is not the same measurement as 0.0148 and swapping it in would mix designs inside a table whose whole justification is that its numbers were measured the same way: a different candidate set (the crossed factorial against the matched v2 set), a different precision (fp16 against fp32), and two seeds against three. The arm's own 1,200-step anchor reads **0.0118** where that table reads 0.0148 at the same length - which is the direct measure of how unlike the two designs' noise levels are, and the reason the substitution was refused. |
| **Verified** | Two claims added to the record check, so the factor is recomputed from the arm's own rows rather than from a printed decimal (**142 -> 175 claims** across the three stretches today): a 12,000-step threshold borrows the 1,200-step std, and that borrow over-estimates the measured noise by 2.64x. Three mutants caught - a table length no longer reporting itself, the annotation removed from the renderer, and the caption restored to its false claim. No committed record renders this table, so the renderer change produced no churn in the nine committed `decision.md` files. |
| **Result** | **1147 tests pass · 175/175 claims agree · 9/9 records reproduce under `--strict-rank`.** |

## The horizon arm landed, and the file left as its record under-read it by half (2026-09-19)

| | |
|---|---|
| **The arm** | `act-t4-archzoo` completed on Kaggle T4×2: 4.83 h wall clock, `complete=True`, **32/32 rows** (8 candidates × {1200, 12000} × seeds 42/43) at ~9.22 GPU-hours token-derived against its own pre-registered 15 GPU-h cap. This is the grid whose horizon half the first attempt lost entirely (16 rows, none at 12,000), so the fix that re-ran it worked. |
| **The branch** | The pre-registered rule lands on **shallow-wins persists**: depth **+0.0209 nats, exact re-split p = 0.003086** (floor 0.001543) and width −0.0645 nats at its own floor. The depth sign is horizon-stable to 10×, so §2c's "spend at low depth" stands at the length a real budget occupies — the reading prior work predicted would reverse does not. Two qualifications come with it: the decay over 1200→12,000 is **1.38× per doubling** against the 1.65× fitted over 400→1200, so §3's rate *over-predicts* decay when extended an order of magnitude; and the **winner changes** across 10× (`W384` → `W1536`) even though the ranking holds at Spearman 0.90. Against its own fp16 anchor the spread falls 0.3085 → 0.1059 nats while seed std falls 0.0118 → 0.0056, leaving the effect at **18.88× its own seed noise** — it does not wash out. |
| **What was wrong with the landing** | The driver imports the kernel's own `decision.md`/`ranked.json`/`search_cost.json`, which are whichever sweep finished last and are **not** the arm's record. The local re-analysis that is the record launched and was killed inside the **4.2 seconds** it runs for, so the file left in place was the kernel's *single-seed* write: it reported "PROVISIONAL — single-seed evidence" and "Run at least two seeds per architecture before deciding" about a complete two-seed arm. Nothing noticed, because the watcher's log stopped mid-line with no `exit=` line to say the attempt had failed. |
| **How that is provable rather than inferred** | `decision.md`'s mtime is **older than all four of its own ledgers** (18:32:47 against 18:32:47.8–18:32:49.2), so it was not derived from the rows that landed. That is a machine-checkable property, not a timestamp curiosity: the analysis takes 4.2 s, the ledgers arrived at 18:32:47–49 and the watcher launched at 18:33:55, so the window is exactly one killed process. |
| **Fix 1 — the analysis is unbuffered** | `python3 -u` in the watcher. The redirected stdout was *block*-buffered to a file, so a kill inside that 4-second window loses every line **and any traceback** — which is why the log had no record that anything had been tried. The failure was invisible not because it was unlogged but because the log was in a buffer that died with the process. |
| **Fix 2 — the landing is verified, not assumed** | The watcher reported on an exit status alone, and 0 says the process finished, not that it wrote. It now **re-renders the record from the arm's own documented command and requires it back byte-identical** (`check_record_freshness.py --only`), which is the same check the test suite runs. Deliberately *not* a timestamp comparison, and the weaker version was tried first: asserting `decision.md` was newer than the ledgers passed the mutation probe even when its freshness test was gutted, because an unrelated ledger-to-ledger mtime comparison in the same file satisfied a substring check — and it would also fail a *correct* record on a no-op re-run where the analysis wrote nothing. Reproducibility subsumes it and names the cause. |
| **Fix 3 — the general guard, in content rather than in mtimes** | `check_record_freshness.py` now fails any arm whose committed record was read over a **seed set no documented command reads**. mtimes do not survive a fresh clone, so the durable invariant is the one the record states about itself: the horizon arm's file claimed seeds `[43]` while its own runbook globs `[42, 43]`. The check is cheap (it reads ledgers, no analysis) and it names the cause — "the committed record was read over seeds [43], but no documented command for this arm reads those" — instead of the symptom a byte diff reports after paying for a full regeneration. Equality per candidate, not containment: an arm's directory may legitimately keep a standalone and a pooled write side by side, which the v1 record does (six seeds recorded over three in its own ledgers). |
| **The same class in the other two watchers** | Both had the identical shape (`python3 … >> "$LOG" 2>&1`) and neither had failed yet, but the reasons to fix them are specific rather than prophylactic. `run_finalist_analysis_when_sweep_exits.sh` deliberately does **not** write a `decision.md` - "writing v1's decision.md is a deliberate step taken once the branch has been read" - so its log *is* the arm's only record, and a block-buffered run would discard exactly the verdict and every condition it exists to preserve. `run_confirm_analysis_when_sweep_exits.sh` ran **two** `--write` analyses and verified neither, and - unlike the other two - never resolved its own directory, so launched from anywhere but the repository root it finds no ledgers and reports a false "ledgers incomplete" against a complete arm. |
| **Why one log kept its failure and the other kept nothing, measured** | The asymmetry is buffering, and it is checkable rather than argued: the same `print` into the same redirection, killed immediately, leaves **0 bytes** on plain `python3` and **20 bytes** under `-u`, while the same line written to `stderr` leaves 17 bytes either way. So a traceback survives a kill and ordinary stdout does not - which is exactly the two logs. The finalist watcher's kept the *full traceback* of the analyzer defect it caught (stderr); the horizon watcher's kept *nothing at all*, because its run was killed before stdout flushed and it had no traceback to fall back on. |
| **The landing machinery now has tests, and the first version of them was too weak** | None of the three watchers was covered by any test, which is how a landing could be wrong without anything noticing. Five tests now pin the properties: unbuffered analyses, a write-path watcher that confirms its landing, a resolved working directory, and the content guard's behaviour plus its wiring into `check_arm`. **The first attempt at the landing guard was a substring check for `-nt`, and it passed against a deliberately gutted freshness test** - because that same watcher computes its newest ledger with an unrelated ledger-to-ledger `-nt` comparison, so the string was present whatever happened to the record check. That is the mistake the doc-index test had already recorded once (reading prose or coincidence as code), and the fix was to assert the *repository's own* record check instead: strictly stronger, and already covered by the rest of the module. **5/5 mutants caught** afterwards - the content guard disabled, the wiring removed, `-u` dropped, the landing check replaced, and the `cd` deleted. |
| **Verified** | The new guard was proven to fire by rebuilding the *exact* defect in an isolated artifact tree — the kernel-side single-seed `ranked.json` beside the real ledgers — where it failed with the message above and **0/1 records reproduced**. Across the committed arms it passes **9/9**, and it was checked against every arm *before* being added, so it is not a check that was fitted to one case. |
| **Record extended** | `verify_record.py` gained `check_horizon_10x`: **31 claims** (142 → 173) recomputed from the arm's own ledgers — the 32-row grid, both lengths' spread and seed-noise units, the decay rate and that it is below §3's 1.65×, the branch the rule returns, both main effects and their floors, the interaction at its 2-seed floor, the Spearman and that the winner is *not* preserved, the fp16 offset and that it is smaller than the effect it carries, and the phase's own cost against its own cap. The Spearman needed an explicit tolerance: quoted to two places (0.90) against a computed 0.9048, the default 5e-3 relative tolerance is 0.0045 and the difference is 0.00476 — the record is right and the audit was too tight for a two-decimal quote, the same rounding case §4a already carries. |
| **A cost decision left as a decision** | The horizon arm's 9.22 GPU-hours are **not** folded into §10's closed 23.632 total. It opened its own phase under its own 15 GPU-h cap, declared in its own pre-registration, so summing them would invent a figure neither declaration authorised. Recorded as a separate one-row table beside the phase it did not extend. |
| **Result** | **1145 tests pass · 173/173 claims agree · 9/9 records reproduce under `--strict-rank`** (all nine now report `ranked.json identical`, where the horizon arm's would have been regenerated from the wrong rows). |

## The p-value helpers' guards, and what a surviving mutant actually means (2026-09-19)

| | |
|---|---|
| **What was found** | A 22-mutant sweep over `_blocked_permutation_p`, `blocked_permutation_test` and `paired_contrast_test` - the functions every p-value quoted anywhere in this phase passes through - left 10 alive. Four were real guards, now pinned. (a) `assignments <= cap` still **enumerates**: the contract is "when the enumeration exceeds cap this samples", so a cap equal to the count leaves the reported method as `exact`, which is what the record's exact claims rest on. (b) `samples <= 0` returns `unavailable` rather than letting the `(r + 1) / (n + 1)` correction report a confident p=1.0 from **no draws at all**. (c) `len(differences) > 1` keeps the `n - 1` division out of reach for a single pair, so a one-pair contrast reports no t rather than a t with zero degrees of freedom. (d) `d < 0` means an exact tie is not counted as evidence *against* sign agreement, which would report a unanimous contrast set as inconsistent - the opposite reading. |
| **What the other six are** | Not gaps, and the distinction is worth recording because a bare "survived" reads like a hole. One is **provably** equivalent: `resolutions <= 1_000_000` can never see equality, because `resolutions` is `2 ** len(differences)` and so always a power of two, while 1,000,000 sits strictly between 2^19 and 2^20 - checked rather than argued, `<=` and `<` are the same function there. The other five differ only on an exact float equality or a specific RNG draw: the three `>= observed - 1e-12` comparisons differ from `>` only when a statistic lands exactly on `observed - 1e-12`, a value the ledger sums can produce only by coincidence and which the epsilon exists to bracket; and the Monte-Carlo coin flip `generator.random() < 0.5` differs from `<=` only on a draw of exactly 0.5, which the Mersenne Twister can make (one value in 2^53) but no seeded test does. |
| **Verified** | **4/4 real guards caught** where all four lived beforehand, and the six unreachable ones re-run and confirmed to survive - which is the evidence for that characterisation rather than an assertion about it. This is the third and fourth equivalent mutant this vein has produced, after the sign clause and the tie branch. |
| **Where the vein ends** | All five decision rules, `verdict` and its attribution, and now the p-value primitives have been swept. Every real survivor across the four sweeps was the same defect class: a threshold or guard compared at **exact equality** with no test asserting which way it decides. That is the durable finding, not any individual fix. |

## The same boundary defect in the last two rules, found by enumerating instead of hand-picking (2026-09-19)

| | |
|---|---|
| **What was found** | Rather than hand-picking edits, every comparison operator in `finalist_order_decision` and `set_comparison_decision` was swapped in turn against its dedicated test file: 18 mutants, 6 caught, 12 survived - and 6 of those 12 were inside the docstring tables (the filter skipped lines beginning with `"` but not `|`). **All six real survivors were exact-equality boundaries**, the same class as the two entries below. |
| **The five that were real** | In the finalist-order rule: `established = p_all < pooled_p` and `fresh_clears = p_fresh < fresh_p` are strict, so a p sitting on the threshold does not establish a replicated ordering and the verdict falls to *coupled* rather than *established*; and `floor_limited = pooled_floor >= pooled_p` is inclusive, since a floor equal to the threshold cannot produce a smaller p and leaves no branch reachable. In the set comparison: `ratio_ok = abs(budget_ratio - 1.0) <= budget_deviation` - the only *inclusive* condition in the codebase, and correctly so, because parity is a band and only an exceedance is rejected - and `significant = p_value < p_threshold`. |
| **The sixth survivor is an equivalent mutant, proven rather than assumed** | `branch = "manual-better" if difference < 0 else "random-better"` puts a pair whose means are exactly equal on the random side, and swapping `<` for `<=` changes nothing. That is not a missing test. The p-value is the share of permuted statistics at least as extreme as the observed one, and *every* permutation is at least as extreme as zero, so a tied pair reports **p = 1.0** and cannot clear any threshold a p-value can be compared against; the clause needs a threshold above 1.0 to be reachable at all. The invariant is now asserted (`difference_nats == 0.0` and `p == 1.0`) rather than reasoned about and filed, because a survivor being a hole or an equivalent mutant calls for opposite responses and the difference is checkable. This is the second equivalent mutant this vein has produced, after the sign clause below. |
| **What was pinned** | Five boundaries, **5/5 caught** where all five lived beforehand. The tie clause is deliberately left unpinned, with the invariant that makes it unreachable pinned in its place. |
| **Why the ratio boundary needed care** | The existing `test_boundary_budget_ratio_is_accepted` passes a ratio of `1.025`, which reads as the boundary but is not one: `abs(1.025 - 1.0)` is `0.02499999999999991`, strictly inside the declared `0.025`, so the check passes under `<=` and `<` alike and cannot see which the rule uses. That is exactly why the mutant survived. The new test takes the deviation from the ratio's own float difference, making the comparison exactly equal by construction, and says so. |

## The verdict's action named the wrong gene, and the test only checked that both genes were named (2026-09-19)

| | |
|---|---|
| **What was found** | A 9-edit mutation sweep over `verdict()` - the function that turns the evidence bundle into the headline and the *next action* - left **5 alive**. The serious one: swapping `max(resolved, key=abs(nats))` for `min` survived. That is not cosmetic. The action is an instruction about where to spend the next phase's budget - "spend the search budget there" - so attributing the effect to the *smaller* of two significant genes reverses the instruction. The existing attribution test asserted only that `` `hidden_size` `` and `` `depth` `` both appeared in the text, which they do either way: once as the lever and once as the qualification. It could not see the two swapped, so the mutant read as green. It now asserts which way round they are. |
| **Three exact-equality boundaries** | Also unpinned, and the same class as the rule boundaries fixed in the entry below: the materiality floor (`spread <= 0.01`), the significance threshold (`p_value < 0.05`), and the per-gene attribution threshold (`row["p"] < 0.05`). All three are numbers the pipeline returns *exactly* - the floor is a declared constant and both p-values come from a discrete test - and each decides **against** spending compute, so reading one generously buys work on an undecidable result. Now pinned at equality. |
| **What was pinned** | **4/4 caught**, where all four lived beforehand: the gene swap, the floor boundary, the verdict threshold, and the per-gene threshold. The attribution test was extended rather than duplicated, since the defect was in what it asserted rather than in what it covered. Each new assertion was confirmed to fail against its own mutation before being kept. |
| **A wording looseness left alone** | At a spread of exactly 0.01 the headline reads "spread 0.0100 nats is **below** the 0.01 nat floor", which is false at equality. The refusal itself is correct and is the conservative direction; only the sentence is loose, and only at an equality a continuous statistic is unlikely to land on. Left unchanged as cosmetic rather than regenerating committed decision text for it. |

## The rules decide at exact equality, and the clause that looked like a gap is an equivalent mutant (2026-09-19)

| | |
|---|---|
| **What was found** | The phase's three pre-registered rules (`replication_decision`, `head_pairing_decision`, `horizon_decision`) are the only thing between a finished arm and a published branch, and each turns on a p-value compared to a declared threshold. A 20-edit mutation sweep over those comparisons - `<=` where the code says `<`, `>` where it says `>=`, dropping a clause - left **7 alive**. Two are the boundaries: both rules decide at *exact equality*, and nothing asserted which way. The third is the largest of the seven: `REPLICATION_RULE` - the thresholds the confirmation arm's entire verdict hangs on - is referenced by **no test at all**, so `ratio: 1.5` could become `2.0` with the whole suite green. |
| **The clause that survived, and why that is a different thing** | Dropping `and consistent` from `resolved` also survived, in *both* the replication and head-paired rules. At the arm's declared seeds this is not a test gap but an **equivalent mutant**: the clause cannot change any verdict there. The interaction p comes from a sign-flip permutation test over the per-seed contrasts, and any contrast set that *mixes* signs bottoms out at **0.0625** - the lone dissenting seed always has its own sign flipped, which only makes the observed sum larger, so at least four of the 64 sign vectors count and the smallest reportable p at six seeds is exactly twice the floor (`2/64 = 0.03125`). Since `0.0625 > 0.05`, `p < 0.05` already implies agreement at six seeds, so the clause is never the deciding one. **Seven seeds is where it becomes load-bearing:** six contrasts at -0.06 and one at +0.04 clear the declared threshold at p = 0.03125 while the signs disagree, which is the *absent* branch rather than the resolved one - the two branches the clause exists to separate. Confirmed by brute force over random mixed-sign vectors for n = 4..8, where the minimum is 2x the floor in every case. |
| **What was pinned** | Five tests. (a) A p landing *exactly* on the threshold is not significant - ``p < main_p`` is strict - and a floor *equal* to the threshold still refuses the branch, since a floor that cannot produce a smaller p makes every branch unreachable. Both are numbers the discrete exact tests actually return, not boundaries the data is guaranteed to avoid, and both are asserted in the horizon rule and the replication rule. (b) `REPLICATION_RULE` is asserted equal to its declared `{main_p: 0.01, ratio: 1.5, interaction_p: 0.05}`, for the reason the module's own comment gives: a threshold that lives only in a document gets reapplied by eye once the numbers arrive. (c) The sign clause is exercised at **seven seeds in both rules**, since both carry the same clause over the same test - with the arithmetic above recorded in each docstring so a future reader knows why six seeds cannot exercise it. |
| **Verified rather than assumed** | Each new test was run against its own mutation and confirmed to fail: dropping the consistency clause in either rule, `<=` for the replication interaction threshold, `<=` for the horizon threshold, `>` for the horizon floor, and `ratio: 1.5 -> 2.0`. Six mutations, **6/6 caught**, where the same probe left all six alive beforehand. The test file also had to be changed to import `REPLICATION_RULE`, which is itself the evidence for the finding above. |

## The standing documents moved to docs/, and the layout is now checked (2026-09-19)

| | |
|---|---|
| **Why** | The repository root carried five documents of 705, 1,236, 2,700, 3,264 and 6,756 lines. A reader or an agent arriving at the root saw five walls and no statement of which was which, and nothing put them anywhere else. `docs/` now holds the four standing documents - the plan, the report, the status log and the runbook - while `artifacts/` keeps the per-arm and per-benchmark evidence beside the data it describes, which is where that belongs. `artifacts/ARCH_SEARCH_FINDINGS.md` deliberately stayed: it is the search phase's own synthesis, not a standing document. |
| **How large they actually are, and whether that is a problem** | Measured from the history rather than guessed. `RESEARCH_REPORT.md`: 557 lines on 09-13, 1,967 on 09-14, 5,349 on 09-15, 6,755 on 09-16, 6,756 on 09-19. `IMPLEMENTATION_STATUS.md`: 210 on 09-12, 501, 1,007, 2,071, 2,578, 2,700 on 09-19. So the size is *accumulated research output*, not unbounded accretion over time: the report grew 3,382 lines in one day of active study and **one line in the three days since**. Both files grow when work lands and stop when it does not, which is the right behaviour for a results narrative and an append-only log. The forward risk is rate, not age - another 09-15 adds 3,400 lines to a file already at 6,756. |
| **What had to change** | Only four places in the repository construct a path to these documents: `tests/test_doc_index.py`, `tests/test_report_artifact_links.py`, `tests/test_fused_vs_frozen_observation.py`, and `scripts/kaggle_queue.py`'s `CODE_PATHS` - the set bundled into the Kaggle code dataset, which now lists `docs` instead of the three filenames. Everything else that mentions them is prose or a docstring. The runbook flow is unaffected; `CODE_PATHS` is read at *publish* time, not on import, so the horizon arm currently running is untouched. |
| **The convention this settles** | Documents are cited by **name**, not by path - roughly 45 files say "pre-registered in `RESEARCH_REPORT.md` section 5.30", and the plan is cited only as "plan section N". Those references stay readable and unedited, including in this append-only log; the README's Documentation table is what maps a name to its location, and it now says so explicitly. Anything that *reads* a file got a real path, which is the distinction that matters. |
| **Pinned** | `tests/test_doc_index.py` enforces the layout instead of performing it once: no markdown may sit at the repository root except `README.md` and `AGENTS.md`, and every `docs/*.md` must be named in the README index. The guard was proven to fire by dropping a stray document at the root, which it named. |

## The mixer benchmark's conclusions hold; three of its stated precisions did not (2026-09-19)

| | |
|---|---|
| **What was found** | `artifacts/mixer-benchmark/FINDINGS.md` is tracked, and it is the document that rules the module-type axis out of the search space ("not on this hardware, in this implementation"). Its table and its argument check out: every `tok/s`, `s/step` and peak-memory figure matches `mixer_throughput.json` exactly, all four OOM layouts appear as OOM in the raw log, the `all_attention` control is a real `a/a/a/a/a/a/a/a` layout, and the conclusion - GDN at 4-7x the wall clock per layer, three of nine layouts unrunnable - survives every number as written. What did **not** hold were three stated precisions. |
| **The three** | (a) "All nine layouts ... land within **0.03%** of each other on total parameters": the recorded totals are `100,725,504` / `100,762,368` / `100,799,232`, a 73,728 spread - **0.073%** of the mean, 0.051% at its furthest from it. (b) The `vs attention` column **truncated** two rows where it rounded the other two, showing 6.8x for ratios of 6.850 and 6.852 against 6.9x for 6.921 - three rows that read as different were the same to within 0.07. (c) "reproduces ... to within **1%** (4,793 vs 4,733 tok/s)": the two figures are **1.27%** apart. |
| **What else the check found** | The four OOM layouts carry no `total_params` in either `mixer_throughput.json` or `bench.log`, so their totals in the table come from the block geometry rather than from a measurement - now stated. And the 4,733 in (c) is not recorded in that directory: the only throughput committed there for that shape is this benchmark's own 4,793, and the v0 zoo records **573** for `D8_H768_E8_A2`, 8x lower. The comparison can be checked arithmetically but not reproduced from the artifacts beside it. |
| **Worth recording, not fixing** | The benchmark's own log warns on **every** pattern that there was "no token file given: using uniform random tokens. Losses on this stream measure initialization, not architecture" - and `FINDINGS.md` correctly reports no loss figure at all, and the table's conclusion rests only on throughput and memory. That is the same defect `SCREENING_AUDIT.md` refuted the v0 screening for, avoided here by not reporting the losses. Verified rather than assumed. |
| **Fixed** | The three precisions are corrected to the values the data supports, each with the arithmetic shown, and the OOM-provenance and no-loss facts are stated where a reader will meet them. |

## A tracked screening analysis recommended stopping while its own audit refuted it (2026-09-19)

| | |
|---|---|
| **What was found** | `artifacts/arch-zoo/analysis.md` is tracked and reads as a live negative result: it reports a screening table, concludes "Should We Proceed with Evolutionary Search? **NO - not yet**", and recommends 10-15 GPU-hours of follow-up on the strength of it. Its sibling `SCREENING_AUDIT.md` - also tracked, and the file `ARCH_SEARCH_FINDINGS.md` sends a reader to - records five defects, each independently sufficient to invalidate the comparison, and states that "the v0 results should not be cited as a negative result about architecture". `analysis.md` itself carried no warning and nothing linked to it, so a reader arriving there first saw a decision with no hint it had been overturned. |
| **The numbers were right; the conclusion was not** | Every figure in its synthetic table is faithful to `zoo_results.json`: all 11 losses match, the range is exactly 0.0376, and the std dev of 0.0102 is the *population* std dev (sample is 0.0107 - a convention, not an error). `SCREENING_AUDIT`'s own claims check out too: `D8_H768_E8_A2` (44.1M active) and `D10_H768_E6_A2` (53.5M active) are both recorded at exactly `201451008.0` flops/token, and `artifacts/arch-zoo-real/` does hold a `zoo_definition.json` and no results file. So this was not a stale-numbers problem. |
| **Fixed** | `analysis.md` opens with a superseded header in the form its own audit established: the reason the synthetic table ranks initialization luck - uniform random tokens at context 512, every loss between **0.43% and 0.85%** above `ln(8192) = 9.0109` (the audit rounds that to "0.4% to 0.8%") - the brief's requirement of context 1024/2048, and the two real-data losses (3.5865, 3.6657) that have no artifact behind them at all because that sweep wrote results only on completion and was interrupted. |
| **Also closed here** | The `96k tokens/s` in `_MoEFeedForward`'s docstring - the one figure of the four measured last entry that sits *above* the measurement and so can still under-provision a budget - now points at `PERFORMANCE_NOTES.md` for the measured rates. |
| **Pinned** | `tests/test_doc_index.py` gained the check that the per-arm document kinds are actually covered by the index's patterns. It failed against the index as written moments earlier, naming `artifacts/mixer-benchmark/FINDINGS.md` - a real gap in the table. The existing "every document the index names exists" check was also made less brittle rather than merely satisfied: a bare filename in prose names a kind of document, so it need only exist somewhere in the project rather than at the repository root. |

## The throughput warning was stale in the other direction, and its arithmetic mattered (2026-09-19)

| | |
|---|---|
| **What was found** | Reading the tracked per-arm notes - the set the new README index marks as not machine-checked - surfaced `artifacts/arch-zoo/PERFORMANCE_NOTES.md`, which records a **75x spread** across the repository's own stated throughput for the dev shape (9.8k in the model docstring's table, 19.6k in its prose, 96k in `_MoEFeedForward`'s, 8.7k measured there) and warns that a budget drawn from 19.6k under-provisions by 2.2x and from 96k by 11x. Budgets in this phase are the binding constraint, so the warning was re-measured rather than cited: `PYTHONPATH=src python scripts/dev_cycle.py --steps 400 --tag verify-doc-throughput` in a fresh checkpoint namespace, on the current code, fp32, batch 8 x context 512, the real TinyStories stream. |
| **The measurement** | **24,299 tokens/s end to end**, 134.9 s for 3,276,800 tokens across two arms (random 24,807, adaptive 24,386). That is **2.8x** the 8.7k the note reports for the same nominal shape and **1.24x** the 19.6k docstring figure. Recorded in the note as a re-measurement, not a refutation: the two are probably not the same measurement (the note records 617 MB peak where the model docstring records 0.90 GiB for that shape, and a 20-step benchmark need not be at steady state). |
| **Why it changed the warning's arithmetic** | On the measured scale the four figures split in two directions. 8.7k, 9.8k and 19.6k all sit *below* 24.3k, so a budget drawn from any of them over-provisions - it wastes GPU-hours, which is the safe direction. The 96k in `_MoEFeedForward`'s docstring sits 4x *above* the measurement and is the only one still able to under-provision, which is the failure the note was written about. The first draft of this entry said all three flagged figures were below the measurement; that is wrong for the 96k and was corrected before committing. |
| **Deliberately not changed** | `src/act/model/v0_moe.py`'s docstrings were left alone. Its wall-time claim is right - 2.8 minutes for a two-arm 400-step dev cycle against 2.25 minutes measured - and the only figure needing a caveat, the 96k, is already covered by the tracked note. No test pins any of these strings. |

## The capacity sweep's probe arms were recording the random arm, bit-identical at every scale (2026-09-19)

| | |
|---|---|
| **What was found** | `scripts/capacity_sweep.py` read `asdict(result.arms[0])` as the sweep arm's result. Every probe-selection arm is configured `arms=("random", "adaptive")`, so `arms[0]` is the plain random-churn arm for all three of them - and the `probe_selection` that distinguishes them (`random`/`argmin`/`argmax`) governs `arms[1]`, which was never read. The `adaptive` and `anti` rows therefore carried the `random` arm's numbers. The project's own self-check already encodes the convention: `v0_session_selfcheck.py:119` reads `adaptive, random_arm = reference.arms[1], reference.arms[0]`. Found from the scratch output's signature, not from the code: `artifacts/capacity-sweep/threshold_sweep.json` records 20M/seed 42 as `random` = `adaptive` = `anti` = `4.343077778816223` exactly, with `random` taking 57.1 s and the other two 1.5 s and 1.4 s; `sweep_results.json` beside it has 8 of 12 rows carrying the 20M row's `final_loss`, `train_flops`, `search_flops` and `cost_ratio` verbatim while `scale`, `total_params` and `active_params` differ and `train_time_s` is 0.75-0.83 s, so the 40M and 60M scales were never trained. |
| **Verified before it was changed** | One real step per sweep arm returned modes `['null']`, `['random', 'adaptive']`, `['random', 'adaptive']`, `['random', 'adaptive']`, with `arms[0].mode == 'random'` in all three probe cases and the adaptive arm's own loss (`8.689111` against `8.7015`) computed and thrown away. |
| **Fixed** | `ARM_MODE` names the session mode each sweep arm reports, and `run_one` selects by mode rather than by position, raising when the session did not return it. A mode the session returns is now the only thing that can be recorded. `main` also printed a failed run and wrote `threshold_sweep.json` regardless, so a file written from no results at all read as a sweep whose arm was merely quiet - the same shape that lost the horizon arm a session. An empty result set is now refused rather than written. |
| **Pinned** | Three tests in `tests/test_capacity_sweep.py`. The arm-recording one was proven to fail on the old behaviour before being kept (`{'adaptive': 2.0} != {'adaptive': 3.0}` - the random arm's loss). A real 120-step four-arm run now gives null `5.6267` / random `5.6036` / adaptive `5.6535` / anti `5.6585`: four distinct values where `adaptive` and `anti` were both `5.6036` before, and the two probe-selection directions now separate. |
| **Scope** | The sweep is unregistered, `artifacts/capacity-sweep/` is entirely gitignored, and no committed document cites any of its numbers (each loss value was searched for). So no live claim depends on the defective rows. What it does explain is why the `analysis.md` in that directory reports "adaptive selection is indistinguishable from random" so cleanly: for the probe-selection arms it was comparing a measurement against itself. The plain random arm is still trained inside the adaptive and anti sessions and discarded there - 2x the compute for those two rows - which is left as it is because that is a study-design choice rather than this defect. **Addressed later the same day, once it was measured rather than assumed: see the entry at the top of this log on the sweep's discarded arms - 7 trainings per (scale, seed) for 4 reported rows, and the rows are bit-identical without the pair.** |
| **What finding it cost** | The end-to-end check overwrote the gitignored scratch checkpoints `artifacts/v0-sessions/20M-seed-42-{null,random}` and created `-adaptive` and `-anti` beside them. No tracked file was touched. |

## The standing documents were accurate and unindexed, and the plan's subtitle described a different plan (2026-09-19)

| | |
|---|---|
| **What was found** | The documents are in good shape where it counts. The report's section 5 numbering is complete (`5.1`-`5.47`, each number carrying its *Pre-registration* beside its *Outcome*), all 28 "plan section N" references resolve to sections the plan has, and every cross-reference between documents points at a heading that exists. What did not exist is a map. Nothing said which document owns which question, and the front door (`README.md`) linked to exactly **one** of them. The reader is handed 46 markdown files, three of which are 2,651, 3,264 and 6,756 lines, with no statement of which is prescriptive and which is a log. Separately, `complete_implementation_plan.md` was subtitled **"Kaggle-Free Compute Edition"** while its own section 3 makes Kaggle T4x2 the primary backend, section 4 the TPU second, with section 5 the Kaggle operating model, section 39 Kaggle notebook architecture and section 82 Kaggle compute allocation - 28 Kaggle mentions and four dedicated sections. The word "edition" occurs once in the file, in that line. `git show -M 75874f6` shows the file was renamed from `... - Complete Kaggle Implementation Plan.md` and the rename's only content change was a trailing newline, so the subtitle contradicted the body in the original document too. |
| **Fixed** | `README.md` carries a Documentation section: a table giving each standing document its job and what it is authoritative for, a second table for the per-arm material (which `NOTES.md` files are *executed* by `check_record_freshness.py` and which `decision.md` files are *generated* by `arch_zoo_analyze.py`), and the rule that a load-bearing number is owned by the ledger behind it. The plan's subtitle now reads `Kaggle T4x2 primary, TPU v5e-8 secondary`, which is what its own sections say. |
| **Pinned** | `tests/test_doc_index.py` makes the map checkable instead of trusted: every standing document at the repository root must be named in the index, every document the index names must exist, and every `RESEARCH_REPORT.md` section reference and every "plan section N" reference must resolve to a heading that is actually there. Each of the four guards was proven to fire on its own defect before being kept - an unindexed root document, an index entry naming a missing file, and dangling section references in both directions. |
| **Why this class matters** | It is the same defect this phase has now produced twice more: a runbook that no longer reproduced its own record, and a reproduce block that wrote somewhere the committed artifact was not. A record that cannot be read as a map fails the same way as a record that cannot be regenerated, just at the level of the whole document set rather than one arm. |

## A scratch sweep ignored its own arguments, and the reset it opened with reset nothing (2026-09-19)

| | |
|---|---|
| **What was found** | `scripts/capacity_sweep.py` has no argument parser, so `--help` began a full four-scale sweep instead of printing the flags, and a misspelled flag was dropped while the default run went ahead. It was found by running `--help` to read those flags, which started a run. The same script opens by resetting state and named `<scale>-seed-<seed>`, while every run writes `<scale>-seed-<seed>-<arm>` - and `run_v0_session` **resumes** from an existing checkpoint directory, so the reset matched nothing and each re-run continued from the previous one's weights. |
| **Fixed** | The axes are real flags (`--scales/--arms/--seeds/--steps/--out`), each validated or refused by name, and the reset now names the directory the run is handed. Both pinned in `tests/test_capacity_sweep.py` against a faked session, because the sweep has no study summary and nothing else reads it. |
| **What finding it cost** | A `--help` invocation overwrote git-ignored scratch checkpoints under `artifacts/v0-sessions/20M-seed-42-{null,random}`. No tracked file was touched (`artifacts/*` is ignored) and nothing was deleted, because the reset named a path that never existed. |
| **Scope** | The script is unregistered and referenced by no test, doc or shell script, so its output is still not part of any record. The fix is only that it no longer does something other than what it says. |

## The finalist arm resolved v1 as coupled, and getting there exposed a record that could not be regenerated (2026-09-19)

| | |
|---|---|
| **The verdict** | 9 runs, 22.1M tokens, 1.236 GPU-hours. Pooled to six seed blocks, the pre-registered rule returns **`order-coupled`**: the pooled exact test clears at p=0.001286 (floor 1/7776) while the screening-free blocks alone neither clear the threshold (p=0.1111, floor 1/36) nor agree on a winner - two different architectures finish first across the six blocks. So the pooled significance **is** the blocks that selected the finalists: nothing was replicated, and **no ranking is carried into the search**. The search's direction comes from §2c's designed factorial, which never rested on one. |
| **The record that could not be produced** | `arch_zoo_analyze.py` raised `TypeError: string indices must be integers` on **v1's own ledgers**, so the committed `decision.md` could no longer be regenerated at all. `joint_structural_fit` emits `genes` as coefficient rows on its success path but as bare gene *names* on its rank-deficient path, and `structural_signal_by_length` indexes that slot as rows. Not a corner case: `common` is the set that ran at *every* length, which for a screening sweep is its promoted finalists (three of them), and standardised columns span only n-1 = 2 dimensions, so a fit over as many predictors as architectures is dependent by construction. `genes` now has one shape, the refusal names the constraint it hit, and the consumer reads coefficient rows only from an available fit - pinned by a synthetic saturated set *and* against v1's real ledgers. |
| **The verdict read the rule's presence as its result** | `verdict()` treated "the rule is in the outcome" as "the ordering was confirmed", so the pooled p - the very number the rule reads as the screening seeds - produced "the finalists having already been confirmed on seeds never used in screening" directly above a section saying no ranking is recommended. The headline, the action and the selection caveat's closing instruction now all follow the branch; the confirming branch's wording is unchanged, and that is pinned rather than assumed. |
| **The record is now audited both ways** | `verify_record.py` recomputes each claim from the **ledgers**, which is not the same thing as re-rendering the artifact the claim is read from - that gap is why the crash survived unnoticed. `scripts/check_record_freshness.py` closes it: it parses each arm's own `NOTES.md` runbook, *executes* it against that arm's ledgers in a scratch tree, and diffs the result. **8/8 arms reproduce byte-for-byte** - `decision.md` and `ranked.json` - from the command their own documentation gives. |
| **Five runbooks did not reproduce their records** | Every committed `decision.md` was correct; the documentation was not. `v2` documented `--experiment-id arch-zoo-100m-v1 --also arch-zoo-100m-v1-confirm`, ids from a plan that was never run under that name; `v1`, `v3-crossed` and `v4-head-paired` documented the pre-rule analysis their own confirmation superseded; `v2-horizon` and `v1-finalists-6seed` documented only their *parent's* pooled write and nothing about themselves. All corrected, and `ranked.json` refreshed to the current schema so `--strict-rank` is clean. |
| **The screen could name seeds it never ran** | `write_reports` passed the *plan's* seed union to `best_available_summary`, so a flat arm's record read `seeds: [42, 43, 45, 46, 47]` against three seeds of ledgers - contradicting its own `seeds_per_architecture`, and therefore the rendered decision, which lists the rows. `arch_zoo_analyze.py` derives the set from the results, so the two writers disagreed about what the same arm had been read over. Now derived from the rows, with the flat-plan case pinned end to end. |
| **Fast enough to run casually** | The check is one subprocess per arm, so the arms run concurrently, and a candidate whose seed set cannot match the record's is ordered last rather than costing a complete analysis. **~60 s serial to 8 s**, with the analysis itself untouched: the Monte-Carlo draws must stay bit-identical or the check could not check anything. |
| **Numbers** | **1110 tests pass**; `verify_record.py` **142/142**, up from 131 - 11 new measured claims pin the branch, both p-values, both floors, the condition results, the block-top count, and the absence of the over-claim in the written headline and action. |

## v1's own open item was opened as an arm, and the rule it pre-registered was tightened before the data (2026-09-19)

| | |
|---|---|
| **Why this arm exists** | `arch-zoo-100m-v1` closed on "NOT ESTABLISHED - this experiment cannot yet tell" and named its own fix in the same file: re-run the finalists on seeds never used in screening, which removes the selection coupling and raises power. Its exact test runs 3 blocks and floors at 1/36, and its own caveat says those 3 finalists were promoted *on the seeds it tests*, so it is mildly anti-conservative in the winner's favour. `ARCH_SEARCH_FINDINGS.md` §9 has it as an open item at ~1.3 GPU-h. This arm is that item and nothing else - no new architectures, no new length, no retuned knob. |
| **What it trains** | The three v1 finalists, genomes byte-identical to the record's, flat at 1200 steps on fresh seeds 45/46/47: 9 runs, 22.1M tokens. It is read *pooled* with v1 by `--also`, so the exact test moves from 3 blocks to 6. |
| **Where, and why not Kaggle** | The local GTX 1070 is idle, so it runs beside the Kaggle horizon arm rather than competing with it. Kaggle is ~2.9x faster per token (13,930 vs 4,873 tok/s) but that advantage **is** the fp16 path, and fp16 is ruled out here: the gaps to be resolved are 0.017-0.023 nats while the measured fp16 differential reaches 0.018, so the measured ordering would be partly a numerics artefact. On fp32 the T4 is ~1.25x the 1070, which does not pay for a publish, an import and a concurrency slot beside a running kernel. |
| **The rule, and the bug found before the data** | `FINALIST_ORDER_RULE` reads the same exact test twice - all six blocks (floor 1/7776; the enumeration is still *exact* at 6^6 = 46,656, under the 1M cap) and the three screening-free blocks alone (floor 1/36) - and gates the positive branch on every block naming the same architecture first. Planting the branches to rehearse them showed the first draft unsound rather than merely loose: `blocked_permutation_test` is **omnibus**, so a design whose screening blocks rank A first while its fresh blocks rank B first is significant on the pool while having replicated nothing, and it passed that draft. Tightened in the stricter direction, with the reason written into the pre-registration, and the case pinned as a regression test. |
| **The pool was impossible until the guard was fixed** | v1's rows predate `run_config.mixed_precision` (added with autocast in `07fc36a`) and carry no field at all; the new arm records `off`. The precision guard created in the previous section read that absence as "unknown" and refused the pool - which would have blocked this arm's entire purpose. A row without the field is fp32 *by construction* (no row predating it could be fp16 or bf16), so absence now reads as `off` while any recorded mismatch is refused exactly as before; an fp16 arm pooled with an fp32 one is still refused, and that is pinned by a test. |
| **v1 stopped asking for what was already bought** | Both of `verdict()`'s closing instructions tell the reader to re-run the finalists on seeds never used in screening - which is the arm that produced the outcome. Once the rule is in the outcome the action reports what the re-run bought and points at the rule section instead. The unconfirmed wording is byte-identical, and that is checked rather than assumed: regenerating the committed `decision.md` for v1 and for the head-paired confirmation arm reproduces both files exactly. |
| **Integrity as a standing check, not a note-time claim** | Six new claims make the arm's design integrity verifiable at any point mid-run: the finalists are read off v1's own 1200-step ledgers rather than restated, the candidate genomes are compared byte-for-byte, every row sits on a screening-free seed, the plan is flat at the declared length, the validation stream is the record's holdout rather than a re-split one, and both arms' rows read as a single precision. **125 -> 131/131** claims. |
| **The run is watched** | `scripts/run_finalist_analysis_when_sweep_exits.sh` counts rows *inside* each fresh-seed ledger - the sweep writes one row per completed run, so a lost run leaves a file that exists and looks whole - and requires v1's three screening ledgers as well, since a pool partner missing turns a six-block design into a three-block one. It logs the analysis rather than writing it: `--write` supersedes v1's committed `decision.md`, which is a deliberate step once the branch has been read. Exercised against the partial arm as it stands, where it names every missing ledger and exits 2 without analysing. |
| **Pending** | The branch itself. The verdict lands in the arm's `analysis_watcher.log`; then `--write` into v1, the numeric rows in `verify_record.py`, and `ARCH_SEARCH_FINDINGS.md` §9. |

## The horizon arm trained none of its horizon, and the admission bar that refused it was the defect (2026-09-19)

| | |
|---|---|
| **What happened** | `act-t4-archzoo` ran 0.50 h and exited `COMPLETE`, and the driver reported `archzoo complete=True`. The 1200-step fp16 anchor trained; **all sixteen 12,000-step runs failed before a step** with `tokens.npy holds 40043724 tokens, need 51153024`. The per-seed logs holding the sixteen exceptions are scratch by the repo's own ignore rule (only `NOTES.md`, the decision/ranked/search-cost/matching/distance/zoo-definition files and the `*_results.jsonl` ledgers are durable), and the kernel's first version still carries the raw output on Kaggle; the failure is recorded in the arm's `NOTES.md` Status. |
| **Root cause** | The harness admission check in `architecture_eval._prepare_streams` asked for **2x** the schedule plus the 2M validation holdout, with no stated reason. The design is feasible and the arm's own cost section said so: 12,000 steps consume 24.6M tokens and use 24,000 of the stream's 37,115 windows, so no window repeats. `required_train_tokens` now states the bar as the schedule itself (24.6M here) instead of 49.2M, plus `required_holdout_tokens`/`required_stream_tokens` so a caller can check a plan against the file on disk. A one-epoch stream is admitted; one token short is still refused. |
| **Two reporting defects** | `mode_archzoo` derived `complete` from `not stopped_at_deadline` and the exit codes, but the screen catches a per-candidate exception and exits 0 even when every run in the stage failed. `complete` is now read off the requested (length, seed) grid against the ledgers, with the absent runs named; `kaggle_queue.py` warns when a re-read arm is incomplete. The screen also had no data-feasibility check, so an unservable plan was discovered inside a kernel rather than at launch; it now measures each length against the stream header **at launch** and refuses before any training, and `--dry-run` reports the same numbers without spending the hours. |
| **The landing is rehearsed, not just guarded** | The watcher's own command was run against a simulation of the landed arm - the real fp16 anchor rows plus synthesized 12,000-step rows, 32 in all, in the gate's four files - and the record it wrote was then put through `check_record_freshness.py --strict-rank`, which reproduced it **byte-for-byte** (`decision.md` and `ranked.json`) from the arm's own NOTES runbook. That closes a risk the audit tooling itself created: the checker symlinks only `*_seed*_results.jsonl` and `zoo_definition.json`, while the pre-registered `--offset-against` reads a *second* arm's ledgers, so the offset had to resolve out of that subset alone - it does. The horizon rule also read at its **declared 12,000 steps** rather than the 1200-step anchor sitting in the same directory, which is the failure the arm's first attempt would have produced had it been analysed while partial. |
| **The launch-time refusal is pinned, not just written** | The header check is only a fact until `main` acts on it, and that wiring is what the first attempt lacked - so it is tested through `main` with a 7,190-token stream: the run aborts naming `flat 12000 needs 26,600,000 tokens, stream holds 7,190`, tells the reader to lengthen the stream or shorten the plan, and leaves **no ledger** behind for anything downstream to read as a run. The existing coverage tested `stream_feasibility` returning short and the impossible-budget refusal separately; the branch that spends the quota was the one unwired. |
| **Not invalidated** | The correction relaxes an admission test, so it cannot alter a run that already passed it; `verify_record.py` reproduces **125/125** and the suite is 1068 passed. The design, candidate set, seeds, lengths and precision are unchanged, and the arm's `NOTES.md` Status section records the failure and the correction. |
| **What the attempt produced** | The reusable fp16 anchor at 1200 steps, both seeds, batch-identical to the record (same stream and holdout). Pooled: depth **+0.0742** vs fp32 **+0.0842**, hidden **-0.1918** vs **-0.1928** - a precision offset of -0.010 and +0.001 nats against effects of 0.08-0.19, so fp16 is measured rather than assumed. |
| **Re-run** | The design re-launched unchanged (~4 GPU-h at the measured 17.2k tok/s, against the 15 GPU-h horizon cap). |
| **Found by following the failure further** | `HORIZON_STEPS` now **declares** the rule's length (12,000) instead of inferring "the longest length that has rows": with only the anchor on disk, that inference resolved to 1200 and would have labelled the anchor "shallow wins persists at this horizon" from an arm that never trained a horizon step. `--also` now **refuses to pool ledgers recorded in different precisions** (the fp16 arm against the fp32 record would have produced an effect size that is part precision offset). The horizon rule's `decision.md` rendering is pinned by a test, branch and refusal both. |
| **The last unread secondary, mechanized** | The arm pre-registers five secondaries; four already had readers (width and interaction from the crossed contrast; ordering and decay from `rank_stability` and `spread_by_length` over the arm's own two lengths, inside one precision). The precision offset had none and the number had been computed once by hand, so `precision_offset` now joins two experiments' ledgers at one length, seed-matched and per-candidate, reached with `--offset-against` and rendered as a table. It refuses an unpaired length or seed set and names architectures it could not join. Rehearsed against the real fp32 ledgers with a planted +0.02 gap, recovered to the digit. |
| **The post-run read is a script, not a step** | `scripts/run_horizon_analysis_when_kernel_exits.sh` waits for the driver, **refuses to write** unless all 32 rows exist (8 candidates x {1200, 12000} x seeds 42/43), and only then runs the pre-registered analysis. The gate was exercised against the first attempt's real partial state, where it flags the two missing 12,000-step ledgers. |

## The head axis closed, the horizon arm opened on Kaggle, and the tree slimmed (2026-09-19)

| | |
|---|---|
| **Head-paired confirmation read** | Six-seed pooled analysis lands on `HEAD_PAIRING_RULE`'s **head effect is real at wide width**: −0.0553 nats at H=896 (12/12 pairs, at the 2/2¹² floor), +0.0084 at H=512 (p 0.37), interaction +0.0636 at its 2/2⁶ floor with all six seed contrasts agreeing. `verify_record.py` recomputes the six-seed rows and the branch (125/125 claims); `ARCH_SEARCH_FINDINGS.md` §2c/§9/§10/§11 and the README are updated. Phase closed at 23.632 GPU-h against the 10–20 cap, overrun pre-declared. |
| **Literature check, zero GPU** | Width winning is consistent with Levine et al. 2020; heads-only-at-wide with the low-rank head bottleneck (Bhojanapalli et al. 2020); MoE sparsity showing nothing at 2.4M tokens with routed-model scaling work (Clark 2022, Krajewski 2024). **Shallow beating deep is the one finding prior work predicts should reverse with training** (Tay et al. 2021, Petty et al. 2023). |
| **Horizon arm, pre-registered** | `arch-zoo-100m-v3-crossed-horizon10x`: the crossed depth×width factorial (byte-identical candidates, verified by dry run) at **12,000 steps** plus a 1200-step same-precision anchor, fp16, seeds 42/43, on **Kaggle T4×2** (one sweep per card). Numeric rule in `HORIZON_RULE`/`--horizon-rule`, every branch exercised on planted and real ledgers before launch; opens a horizon phase with its own 15 GPU-h cap. |
| **Kaggle plumbing** | `kaggle_t4.py` gains an `archzoo` mode (two seeds, one per device, wall-clock deadline, ledgers into the kernel output, exit 0 with `archzoo-status.json`), `kaggle_queue.py --queue archzoo` pushes and imports the ledgers into `artifacts/`, `arch_zoo_screen.py --mixed-precision` records precision per row. Token stream published as `act-lab-tinystories-tokens`. The CLI's OAuth refresh gap is worked around in `_run`. |
| **Suite and audit made fast** | `pytest` 243 s → **35 s** (pytest-xdist, one torch thread per worker via `tests/conftest.py`, three `/tmp` races moved to `tmp_path`, test-only Monte-Carlo sample counts). `verify_record.py` 63 s → **18 s** full / 27 → 14 s quick: the search-space enumeration is memoised (a 200-seed budget scan re-enumerated 19,320 cells per seed) and the audit's checks and power runs execute in a process pool. **No statistical RNG path changed**; every recorded Monte-Carlo p reproduces. |
| **Removed** | The TypeScript console scaffold (112 tracked files: routes, auth, pglite, PWA plugins, vite/eslint/prettier, `public/`, `server/`, `migrations/`, `startup.sh`), `node_modules`, screenshots. Untouched since the initial commit and representing none of the results. |

## Section 5.46: the declared launch path could not specify the rebuild treatment (2026-09-16)

Found by auditing the launch command against the adopted sections. Section 5.16 ("new work
should set `preserve_router_on_rebuild=True`") and section 5.38.3 ("a real-scale protocol
should be specified with the treatment on") both say what the declared run should carry,
and `cli._run_pair_session` had **no way to express it**: no CLI flag, no override passed
into `declared_session_config`, and no test or paragraph pinning the declared run's rebuild
policy. The protocol was registered (§5.5, 2026-09-14) before the treatment existed
(§5.16-5.28), so an unflagged launch always ran the historical rebuild.

| | |
|---|---|
| Measured consequence (§5.39 stored rows) | churn harm **0.16449** untreated vs **0.11279** repaired - the treatment is worth **1.46x** on the quantity §5.44's clause reads |
| Compared with the effect it gates | 0.0517 nats, above the clean-arm effect (0.04042) - so the missing flag can decide Gate 4 |
| Undone now (additive, no default moved) | `--preserve-router-on-rebuild`, `--new-expert {random,clone}`, threaded into `declared_session_config` for both the declared and `--profile` shapes; `clone` without preservation refused with a one-line `SystemExit`; 3 tests pin the default, the treatment, and the refusal |
| Fell out of the audit | the published seed summary already records all 35 config fields, so **which rebuild policy a run used is checkable from its own artifacts** |
| Deliberately not done | the runbook's command is unchanged - registering the declared protocol on the treatment is a sign-off decision. `T4_RUNBOOK.md` carries a **DECISION PENDING** callout at the command so T4 cannot start without noticing |

**Closed in the same section:** the verdict never compared a published run's recorded
config against the registered protocol, so a run whose configuration differed from the
registration would have been judged as though it were the registration. `v0_verdict.py`
now has `REGISTERED_RUN_CONFIG` plus a `config_conformance` block (per field: `matches` /
`differs` / `not_recorded`, with `conforms` and a reading string). It is **descriptive
only** - no label, exit code or authorization moves - and the rehearsal pins that by
building the untreated-rebuild and pre-5.5 `prefix` runs and checking they keep their
authorizations while being flagged. A pre-field artifact reads as `not_recorded`, never as
a mismatch. The two committed real-path diagnostics correctly read as non-conforming on
nine shape/budget fields (they are depth-8, 600-step runs), with their labels unchanged.

## Section 5.47: the declared run is registered with the rebuild treatment on (2026-09-16)

Delegated decision, recorded with reasoning. The declared launch command now passes
`--preserve-router-on-rebuild --new-expert clone`, and so do the §2b control and §2c null
commands - a different rebuild policy in a paired queue would make the comparison a package
of two changes instead of the decision rule alone.

| | |
|---|---|
| Grounds | (1) the untreated rebuild is an unintended router discard kept as the default only to preserve the committed record of §5.16-5.23; (2) §5.38's branch (c), which would have forbidden this, was superseded by its own correctly-sized block (§5.38.1 passed the clean arm at n=12, perm p 0.0239, clean-comparison 12/12); (3) the gap, 1.46x on churn cost, is larger than the effect claimed |
| Not claimed | the treatment does not decide Gate 4: effect is small and right-skewed (8/12 against a 10-of-12 win bar), and §5.39 shows 69% of the churn harm remains, so the §5.14 do-nothing clause stays the binding protection |
| What changes | two config fields, applied to **every** arm; no parameter count, no probe FLOPs, so the cost prediction and band are untouched |
| What does not | no criterion moves (5%, band, harm guard, §5.5 pool/quality clauses, §5.14 null clause, completeness); seeds, shape, tokens, events, dtype and probe regime all unchanged |
| Reversible | drop the two flags before session 1 and the conformance block will record the run as untreated rather than letting it pass as treated |

Record: `RESEARCH_REPORT.md` §5.47.

## Section 5.44: the quality-premise clause was shipped in one direction only (2026-09-16)

Section 5.5 change 3 was **ADOPTED** (2026-09-14) and reads "if the pool-matched arm's
quality is indistinguishable from the adaptive arm's ... withhold the Gate 4
recommendation". Only the direction where **both** arms clear the bar shipped; the
direction where **neither** does was implemented nowhere, so it authorized Gate 4.

Nothing else in the gate looks at the adaptive arm's *quality*: the harm guard is a floor
(rejects only at >= 10 losses of 12, so 3 wins clears it), and the §5.14 clause compares
churn against do-nothing rather than adaptive against the loss-blind arm.

| adaptive vs churn | loss-blind | before | after |
|---|---|---|---|
| 6/12 (tie) | 6/12 | `PASS`, exit 0, **authorized** | `PASS_COST_ONLY_NO_DEMONSTRATED_EDGE`, exit 2 |
| 9/12 | 3/12 | `PASS`, **authorized** | withheld |
| **3/12** | 0/12 | `PASS`, **authorized** | withheld |
| 2/12 | 0/12 | harm guard rejects | unchanged |
| 10/12 | 3/12 | `PASS`, authorized | unchanged |

The tie row is exactly the state the dev-fidelity evidence predicts for the real path
(§5.13, §5.39), and the 3/12 row is the harm guard's own boundary - one win above
rejection. **The amendment** completes the clause with the same `POOL_CONTROL_WIN_LIMIT`
(no new threshold) and is strictly tightening, so no spent block is reinterpreted -
verified by re-computing both committed real-path verdicts under the pre- and
post-amendment code (identical labels both sides). Gate 4 now requires the adaptive arm to
clear the bar **and** the loss-blind arm not to; the record carries a `quality_premise`
block so the reason is stored, not inferred from the label. The proxy's boundary (it also
withholds at 9-vs-3, where adaptive is clearly better) is recorded as the reading to
contest, in `registered_text_amendments[0].boundary_reading`.

Rehearsal is now **16 variants / 9 checks**, all matching; `tests/test_v0_verdict.py`
rows at the harm-guard boundary were rewritten to pin the interaction rather than
relaxed. Teeth: removing the withholding branch fails two unit rows and makes `--check`
exit 1 with three checks false; restored, 52/52 pass. Record: `RESEARCH_REPORT.md` §5.44.
Also corrected: §5.41's verification sentence misstated `local-proof`'s re-run label.

## Section 5.45: the suite's 140 s had two measurable causes, one in production code (2026-09-16)

Measured, not guessed: `--durations=35` put ~90 s of 140 s in 25 tests and 41 s of that in
four.

| | before | after |
|---|---|---|
| one CLI error-path test paying production publish backoff (`time.sleep`, injectable and already injected by the remote tests) | 15.0 s | ~0 |
| `act/evolution/study.py::_gate_min_wins` re-summing `math.comb` per step (O(n^2), ~1.06M calls for a 1536-seed block) | 16.15 s | 0.4 ms |
| full suite, `OMP_NUM_THREADS=1` (comparable with the ~140 s baseline; unpinned it is 88.3 s) | ~140 s | **98.5 s** |

The two fixes are 31.2 s of the ~42 s reduction; the rest of the suite is real work (slowest entries are fp16 session round-trips at 9.1 s and 4.4 s).

The binomial tail is now accumulated once with the exact recurrence. Because the returned
integer is a *pre-registered* analysis threshold, equality was verified rather than
assumed: **bit-identical on 199 checks** across 12 block sizes (incl. 1536), 8 alphas and
the tail-sum identity. Record: `RESEARCH_REPORT.md` §5.45.

## Section 5.43: the declared Gate-4 verdict path is rehearsed before the spend (2026-09-16)

*(Superseded in two details by §5.44: the record now holds 16 variants, and two variants
authorize - `all_clear` and the distinguishable case.)*

`scripts/verdict_rehearsal.py` builds synthetic **declared-scale** fixtures
(285,336,576 / 134,341,632, 24,576,000 tokens per arm, all three queues) and applies the
real `compute_verdict` to 14 variants - every withholding clause in isolation, both
(16 as of §5.44)
absent-queue and both short-queue cases, and the three precedence combinations. Record
`artifacts/study-summaries/gate4-verdict-rehearsal.json`, pinned by
`tests/test_gate4_rehearsal.py` (9 tests; 5 tampers each fail, restored passes).

**Why it was worth doing before the spend:** sections 5.41 and 5.42 both found defects in
this machinery by reading it, and the verdict is otherwise applied exactly once, at the end
of a multi-hour Kaggle run - so a wiring error would have surfaced as a confidently wrong
answer with nothing to compare it against.

| | settled |
|---|---|
| **Label precedence** | Verified, not just documented: cost FAIL > harm guard > pool-matched control > null > missing control > missing null > band miss. The outranked clauses' flags are recorded, so the labels are precedence and not absence. |
| **`primary_queue_short`** | An incomplete primary queue gives an ordinary `PASS`, exit 0, and `gate_4_authorized: False` - the completeness requirement lives in the flag, not the label, so a label-only reader misreads it. Now pinned. |
| **5.41 + 5.42 hold at declared scale** | Every fixture reports `run_is_at_declared_scale` with both band readings agreeing, and both short-queue variants return `..._MISSING`. |
| **Authorizing variants are named** | `all_clear`, exit 0 (two as of §5.44, the second being the distinguishable-quality case). Cost failures exit 1 and nothing else does. |

**It caught my own unit error.** A first pass expressed both cost variants as "6x the
prediction"; the prediction is itself a percentage (0.0942%), so 6x is 0.565% - inside the
5% criterion. The machinery returned `PASS_WITH_QUALIFICATION` and `HARM_GUARD_REJECTS`
where the rehearsal had written `FAIL`, and its own check reported the drift; crossing 5%
takes a multiple above 53.1. A check now ties each variant's cost to its knob - the same
family as section 5.41, committed one section after writing it up.

**Not evidence.** No session ran and every loss is fabricated; the record carries
`synthetic: true` and says it validates the decision path rather than the hypothesis, both
pinned. `--check` exits non-zero on drift, and `T4_RUNBOOK.md` step 0 runs it before
session 1.

## Section 5.42: a short pool-matched control made its own clause unfireable (2026-09-16)

Found by comparing two sibling clause evaluators that should behave alike. `_null_block`
states in its docstring that `evaluated` "requires the queue to exist AND be complete on
the declared block; anything less withholds Gate 4 rather than reading as a null result".
`_control_block` returned `evaluated: True` with a `seeds_missing` list and no guard.

**Why that is exploitable.** The clause is `wins >= POOL_CONTROL_WIN_LIMIT` (10), so on
fewer than ten pairs it cannot fire however the loss-blind arm performs, and
`compute_verdict` reads that as "clause evaluated, did not fire"
(`control["evaluated"] and control["control_passes_gate"]`). `gate_4_authorized`
consults only the **primary** queue's missingness.

| control pairs | loss-blind arm | clause | verdict | `gate_4_authorized` | exit |
|---|---|---|---|---|---|
| 4 of 12 | wins 4/4 | **unfireable** (4 < 10) | `PASS` | **True** | 0 |

So the clause whose purpose is to stop a pool/location effect from authorizing Gate 4 was
defeated by **not finishing the control queue** - the state a deadline-stopped session
produces - with nothing warning.

**Fix, mirroring `_null_block`.** `evaluated` now requires the control complete on the
declared block, recording the missing seeds and the reason; the existing
`PASS_COST_ONLY_CONTROL_MISSING` label (exit 2, Gate 4 withheld) picks it up, so no new
label and no other registered behaviour moves. A complete control behaves exactly as
registered, including 9-of-12 staying authorized. The same fixture now reads
`PASS_COST_ONLY_CONTROL_MISSING`, `gate_4_authorized: False`, exit 2.

**Boundary pinned:** a control at 11 of 12 whose loss-blind arm loses every pair is
withheld too. Two tests, teeth verified by reverting the guard (both fail; restore passes
38/38). Recorded as report section 5.42.

**Fourth instance of one shape.** Section 5.33's `forced_operation` no-op, section 5.37's
unclassified field, section 5.41's band scale, and this: **a check whose result can be
decided by something other than the quantity it claims to measure.** All four found by
reading; none by running a study.

## Section 5.41: the Gate-4 verdict's tolerance band could only be passed at declared scale (2026-09-16)

Found by reading the two committed real-path verdicts. Both record `in_tolerance_band:
false`, and the registered rules make a band miss a qualification that "forces
derivation before Gate 4" - so for two sections they have read as a miss.

**They could not have done anything else.** `compute_verdict` takes the measured cost
from the run's booked FLOPs but derives the band from `expected_probe_cost_percent`,
which is pinned to the **declared** V0 shape (285,336,576 / 134,341,632, 24,576,000
tokens/arm). The section 5.3 block ran 600 steps on a ~12.2M-parameter model -
**10.6983x** from the declared scale - so the check was measuring the distance between
the run and the declaration, not between the measurement and the identity.

**The fix is the registration's own rationale applied.** Section 5.1 already justifies
the band's slack by saying every architectural deviation "enters the identity as a
bounded multiplicative factor on `N_active` or `D`", so the factor is applied
explicitly from the run's own numbers:
`C(run) = C_declared x (T_declared / T_measured) x (N_run / N_declared)`, which reduces
to the registered comparison exactly at declared scale.

| block | scale, measured/declared | measured/predicted **at the run's own scale** | registered `in_tolerance_band` |
|---|---|---|---|
| `real-proof` (real TinyStories, 9000-9011) | 10.6983x | **1.000286** | false |
| `local-proof` (synthetic tokens, 9000-9011) | 10.6631x | **1.000278** | false |

**The only real-path execution of the compute-efficiency protocol confirms its own cost
identity to 3 parts in 10,000** (with a +0.03% systematic overshoot on both blocks),
while both artifacts report a miss.

**Additive only.** `in_tolerance_band`, `predicted_cost_percent`, `tolerance_band_percent`
and the verdict string are unchanged - verified by re-running the verdict over both
committed blocks (`PASS_COST_ONLY_HARM_GUARD_REJECTS` in each, as committed). The new
`run_scale` block is labelled `descriptive_only`, no threshold or rule moves, and the
report does not spend the reading as a criterion. What it buys is legibility on the run
that decides the gate: a declared queue that deadline-stops, resumes, or is pre-flighted
at reduced fidelity now says which reading applies. The declared numbers were also
inlined inside `expected_probe_cost_percent`; they are now named constants
(`DECLARED_TOTAL_PARAMETERS`, `DECLARED_ACTIVE_PARAMETERS`, `PROBE_TOKENS_PER_BATCH`,
`DECLARED_EVENTS`, `DECLARED_CANDIDATES`) so the re-derivation cannot drift from the
prediction - unchanged to the last digit, `0.0941636252059042`.

**Third instance of a pattern.** Same family as section 5.33's `forced_operation` no-op
and section 5.37's unclassified field: **a check that cannot be passed on the input it is
applied to.** This one sat in the artifact that decides Gate 4. Pinned by three tests in
`tests/test_v0_verdict.py` (exact reduction at declared scale, the additive contract on a
reduced-scale fixture, and the committed section 5.3 block's own numbers).

## Section 5.40: registered, deliberately not run (2026-09-16)

The deployed-configuration adaptive-vs-null block was registered and committed
(`e7e0b0f`) and then stopped after 5 of 24 sessions (seeds 15240-15242). Two measured
reasons, not a change of mind: the registered shape costs ~200 s/seed (135 s pair + 67 s
null), i.e. ~40 min of GPU for the block; and a long run **cannot** outlive its launching
shell here - a backgrounded attempt died with the shell that started it - so it can only
be bought in foreground chunks, which is time this project has twice been told is not
worth spending. No statistics are reported from the partial rows (scratch only, no
summary, no verdict). The registration stands, its prediction is sourced from the
disclosed post-hoc diagnostic (reproducible from committed rows by `--summarize`), and
the runner is resumable for whenever the GPU time is worth it.

## Section 5.39: churn harm is a property of churn, not of the rebuild (2026-09-16)

18 paired seeds (the section 5.38/5.38.1 queues), dev fidelity, real TinyStories stream: 18 new
**null-arm** sessions (~20 min GPU, 67 s each) joined to the stored churn rows. Summary
`real-path-churn-harm.json`, pinned in `tests/test_churn_harm_summary.py`. Deltas are `null - churn`,
so **negative is the harmful direction** (see the sign correction below).

| point | mean final loss |
|---|---|
| `null` (zero mutations) | **3.53398** |
| `random_base` (historical rebuild) | 3.69847 |
| `random_clone` (repaired rebuild) | 3.64678 |

| primary | mean delta | wins | permutation p | verdict |
|---|---|---|---|---|
| `null - random_base` | **-0.16449** | **18/18** | 5.00e-05 | harm established |
| `null - random_clone` | **-0.11279** | **18/18** | 5.00e-05 | harm established |
| replication `base - clone` (not a primary) | +0.05170 | 12/18 | 0.00475 | matches 5.38.2 |

**Branch (b): harm survives the repair.** Doing nothing beat churning on **every one of 18 seeds
under both rebuilds**, and the repair recovers 31.4% of the cost without removing it (0.16449 ->
0.11279). The registered red-flag direction did not occur.

**This falsifies section 5.38.3's hope** that a treatment-specified declared run would not inherit
section 5.7's harm: it inherits 69% of it. And it re-reads section 5.38.1 - the repair is 31% of a
**net loss**, not the creation of a net gain.

**Section 5.14's do-nothing clause is now the binding bar, with a measured number.** Churn costs
0.11279 nats with the repair in place, **2.2x** the +0.0517 treatment effect, so an adaptive arm that
merely matches random churn is 0.11279 nats worse than doing nothing. The declared run's null queue
is the only thing in the protocol that detects that; it is load-bearing, not insurance.

**It also moves the named next direction off the probe.** Section 5.3 scoped the real-text failure to
the probe's candidate ordering; this block shows a large part of what any ranking is fighting is that
changing the architecture at all costs ~0.11 nats at this fidelity. Two licensed follow-ups, neither
of them another probe-horizon variant: make mutation net-positive (5.26-5.28 moved this term by a
third; 5.27 put 3/4 of the gain in the cloned body, so the residual is plausibly the pruned expert and
the layers a prune re-draws), or **mutate less** (5.7's degenerate gate and 5.11's theta sweep both
showed a controller that rejects most draws beats one that spends the budget).

**Sign correction, and both errors were mine (report section 5.39.1).** The registration wrote
"positive means churn is harmful" beside the arithmetic `null - churn`, which is the reverse of its own
wording. On the inverted predicate the branch came back `c` ("not harmful") while 18/18 seeds agreed
the other way and the sign test returned p = 1.0 - the block's own checks refused to let a branch that
contradicts every seed through. The second error was in the replication check, which compared
`clone - base` against a summary whose stored delta is `base - clone` and so reported `False` on
nothing but a sign. Both fixed, neither changed a measured loss, and the registered direction is left
visible in the report rather than overwritten.

**Accounting detail, measured not asserted:** the null arm records a no-op at all **four** cadence
slots (100/200/300/400) while `mutation_budget=3` caps the churn arms at three applied mutations. A
no-op has `search_flops` 0 and leaves the architecture hash unchanged, so it cannot move the contrast;
it is recorded because "budget-matched" is a claim.

**Length caveat:** dev fidelity (depth 6, 14.9M total / 7.6M active) where the declared config is
depth 12 at 285M, on two queues 5.38/5.38.1 already spent. It decides a branch and sets the bar; the
declared run is what measures the bar.

## Section 5.38.1: the treatment's clean effect is real, small and right-skewed (2026-09-16)

Seeds 15220-15231, 24 sessions, ~27 min on the dev GPU (the section 5.37.1 checkpointing fix paid
off here: ~66 s/session against ~90 s). Summary `real-path-clone-dev-n12.json`, pooled secondary
`real-path-clone-dev-pooled.json`, both pinned in `tests/test_real_path_block_summary.py`.

| | value |
|---|---|
| `random` arm, n=12, mean `base - clone` | **+0.04042** nats (se 0.0192, t +2.10) |
| sign-flip permutation p (declared statistic) | **0.02390 -> PRIMARY PASSES** |
| exact sign test | 8/12 against a bar of 10, p 0.194 -> fails |
| the two statistics disagree | yes |
| clean-comparison guarantee | identical event sequences **12/12**, grows 21 -> 21 |
| pooled with 5.38 (post-hoc secondary, n=18) | mean +0.05170, mean-test p **0.00375**, wins 12/18 (bar 13) |

**Honest reading: the mean improves and the per-seed win rate does not.** The effect is ~0.05 nats,
about 1.4% of the ~3.6 nat dev loss, and right-skewed - seed 15224 alone contributes +0.1883, and
without it the n=12 mean falls to +0.0270. That is section 5.31.1's signature: a positive expected
effect carried by a tail, not a reliable per-seed win. Both statements answer different questions,
and the write-up keeps them apart.

**This resolves section 5.38's branch (c) instead of repeating it.** The clean arm was failing its
test because n=6, sized from the pilot's optimistic sd, could not resolve an effect this size - not
because the mechanism fails to transfer. Branch (c) is superseded for interpretation; its numbers
stand as run. The probe arm's larger effect (+0.1316, 6/6) is still a package of rebuild plus
reselection, which this design does not apportion.

**Settled:** preserve-router plus clone improves expected final loss on the declared path at dev
fidelity, on the arm that cannot be affected by probe ranking, under a pre-registered test with the
clean-comparison guarantee holding 12/12 - so section 5.28's pipeline change now has registered
real-path support rather than only toy evidence. **Carried forward as calibration:** the effect is
small, so the real-scale protocol should be specified with the treatment on without being sold an
effect size on the strength of this run. **Still open:** the rebuild/reselection split, and Gate 4
behind the compute-efficiency measurement.

**Runner note.** `scripts/real_path_block.py` now serves both blocks by profile
(`--profile 5.38 | 5.38.1`, plus `--pool`), with runs keyed by profile so a seed cannot be reused
across blocks by accident. Its t-column is descriptive only: the project's declared statistics are
the exact sign test and the permutation test, no t p-value is recorded, and the script no longer
needs scipy (which is not installed here) to say so.

## A free 1.32x on every dev cycle, straight out of the liveness contract (2026-09-16)

Section 5.37's contract established that `activation_checkpointing` is trajectory-inert, and the dev
config defaulted it to **on** - correct for the declared 285M target (section 5.2's T4 fit arithmetic
assumes checkpointed activations) and pure recompute cost at the dev shape, where memory is not a
constraint. Measured on the dev GPU at the dev shape, 60 steps on the real stream, one arm:

| setting | wall | throughput | loss |
|---|---|---|---|
| on (the dev default until now) | 13.3 s | 20,145 tok/s | 5.721975 |
| **off** | **10.1 s** | **25,760 tok/s** | **5.721975 (bit-identical)** |

The off number matches the **documented** dev throughput (25,525 tok/s), so the handbook and the
configuration had been describing different runs, and every dev cycle - including all 12 sessions of
section 5.38 - paid about 30% more than it should have.

Fixed: the field default is now `False`, and `declared_session_config()` sets `True` **explicitly** so
editing a dev default can no longer move the declared target. Both ends pinned in
`tests/test_v0_session.py`. Section 5.38's results are unaffected (the flag is inert - which is why it
was safe to act on); only its wall time was, and 5.38.1's 24 sessions now cost ~27 min instead of ~36.
The suite also got faster (155.6 s -> 138.5 s). Recorded as report section 5.37.1.

## Section 5.38: the real-path block landed on branch (c), the red-flag branch (2026-09-16)

Six fresh seeds (15210-15215), 12 sessions, ~33 min on the development GPU, base vs clone on the dev
shape and the real TinyStories stream. Runner `scripts/real_path_block.py` (resumable and
budget-capped), summary `artifacts/study-summaries/real-path-clone-dev.json`, pinned by
`tests/test_real_path_block_summary.py`.

| primary | mean `base - clone` | t | wins | perm p | verdict |
|---|---|---|---|---|---|
| `random` arm (clean) | +0.07424 | +2.06 | 4/6 | 0.0624 | **FAIL** at `alpha/2` |
| `adaptive` arm | +0.13159 | +4.92 | 6/6 | 0.0156 | **PASS** |

**Branch (c) was pre-registered as a red flag rather than a result** - a probe-driven arm improving
decisively while the clean seeded arm does not - so this is recorded as **unresolved**, not as a
transfer. The diagnostic is why: the two settings selected the **same** mutations on the random arm on
**6/6** seeds, and **different** ones on the adaptive arm on **6/6**. The adaptive arm's gain is a
package (rebuild + reselection), and only the random arm isolates the rebuild. The reselection goes in
the predicted direction - preservation makes a grow look better to the probe, so the probe arm grows
more (5 -> 6 of 18 events) while the random arm does not move (9 -> 9).

**Section 5.36 is corrected.** It reported that every one of its six arm losses improved; on six fresh
seeds **two of the six random-arm comparisons reverse** (15214, 15215), so that uniformity was a
three-seed artifact. Its numbers should not be quoted as an effect estimate.

**The failure is a sizing failure, not a null.** The pilot's sd (0.023) understated the true
dispersion (0.088, ~3.8x), so n=6 could only resolve effects above ~0.09 nats on the clean arm. The
clean mean is in the helpful direction with 4/6 agreeing. Section 5.38.1 registers the follow-up at
**n=12 fresh seeds, `random` arm only** (~36 min) and states that no result from it may resize or
reinterpret 5.38's spent seeds.

**Two disclosures.** (1) The runner's pass predicate had an inverted sign (`mean < 0` inherited from
the 5.34 studies, where the delta was treatment minus reference); it was found after the block ran and
corrected, and changes only the `passes` boolean - the deltas, wins and both p-values are computed
before it, and the verdict was hand-checked against them. (2) The event-sequence check, the per-arm
event-change counts and the grow totals were added to the summary *after* the block as structural
diagnostics, and are reported in full rather than summarised.

**What this does not do:** it is dev fidelity, GPU (so not bit-reproducible in the section-6 sense,
which is declared in the summary), one substrate, and final held-out loss rather than capability per
FLOP - so Gate 4 is untouched by it either way.

## The config surface is now checked instead of promised (2026-09-16)

Sections 5.33 and 5.34 both found the same thing the hard way: a *declared* config field that
no code path reads under the policy the block was running, so registered compute went into a
mechanism that was never applied. Nothing was checking the config surface, so the only thing that
caught either case was reading the code afterwards.

`tests/test_config_liveness.py` now checks it. Every field of `V0SessionConfig` and
`ExperimentConfig` (35 each) is classified, and each classification is **measured** at a tiny shape:
flipping a field must move the recorded trajectory where the contract says it moves and must not
where it says it must not, with decisions, losses, validation, booked FLOPs, probe losses and the
stored record separated so that "bookkeeping only" is testable. An **unclassified field fails the
completeness guard** - the check whose absence let 5.34's forced-operation block run as a duplicate
of the forced-grow block.

Measured facts now pinned rather than assumed:

- `prune_expert_policy` and `forced_operation` are **bit-identical on the probe-guided arm running
alone**, and move the random arm's trajectory. Both are scoped facts about the probe path, not dead
fields, and wiring either into the probe path now fails the table.
- `probe_scoring` is inert when the applied rebuild already *is* the historical default - the
dependency 5.34 relied on - and live otherwise.
- `probe_landscape` changes the stored record with no change to any decision and no change to booked
FLOPs, exactly as documented.
- `validation_batches` is **not** reporting-only: it consumes the arm's RNG and moves the final loss
while leaving every selected mutation identical.
- Hazards: `optimizer_reset_steps` is masked when it collides with a mutation step on a *mutating*
arm (the rebuild already discards Adam; 5.23's `fixed`-arm control is unaffected);
`probe_force_events` is a modifier, inert without `probe_force_candidate`.
- A flip is only as good as its value: `adaptive_load_threshold` moves nothing at `0.5 -> 0.9` and
everything at `0.05 -> 0.95` (6/6 seeds), so a mid-range flip would have been misread as a dead knob.

The declared path came out clean - no field on `V0SessionConfig` reaches no decision - which is now a
standing assertion rather than an assumption. 20 tests, 9.6 s. Teeth checked by tampering: making
`argmax` behave as `argmin` fails the `probe_selection` row, and adding an unclassified field fails
the completeness guard. Recorded as report section 5.37.

## The declared path had the section 5.24 defect too (2026-09-16)

While porting section 5.34's fix, reading `rebuild_v0_moe_from_genome` showed it had the
**identical** router defect that section 5.24 fixed on the toy path: expert bodies copy by
id, but the router is `hidden_size x experts_in_layer`, whose shape moves with the expert
count, so every declared mutation re-drew that layer's learned routing function. Seven
pre-registered studies were designed around a treatment the real path never had - a process
failure worth more attention than the bug, since the two paths have separate factories and
only one got fixed.

Ported: `preserve_router` / `new_expert_initialization` on the declared factory (defaults
unchanged), the same two fields on `V0SessionConfig` (rejecting `clone` without
preservation), and **one helper** that every applied rebuild in `v0_session` goes through, so
the treatment cannot be wired into some paths and forgotten in others - the failure mode that
turned section 5.33's `prune_expert_policy` and section 5.34's `forced_operation` into no-ops.
Pinned by `tests/test_v0_router_rebuild.py`.

**Dev-scale signal on the real TinyStories stream** (3 seeds, `scripts/dev_cycle.py`, ~162 s
per cycle; NOT a registered measurement): all six arm losses improved under the treatment, and
the adaptive-minus-random gap improved on all three seeds (+0.0893 -> -0.0178, -0.0260 ->
-0.0323, +0.0519 -> +0.0498). The `random` arm is the cleaner signal - it mutates on a seeded
schedule and never probes, and it improved on all three seeds. It licenses a registered
real-path block, not a claim.

**Dev-loop facts measured, not assumed:** dev cycles do **not** parallelise (four with
`xargs -P 4` failed to finish in 590 s; sequentially they are 162 s each - the path is
launch-bound, so the GPU serialises and the cores are wasted on contention). And a *resumed*
arm prints a nonsense tokens/s (687 billion in one run, because it books the full token budget
against a near-zero wall) - read the loss columns on a resume, not the throughput column.

## Section 5.34: the probe was scoring candidates in the wrong rebuild (2026-09-16)

The probe ranks candidates by the loss of a model it builds, and it was building that
model with `rebuild_model_from_genome`'s **historical defaults** - a freshly drawn router
and a random new expert - while the event then applied the winner with the arm's own
settings (`preserve_router_on_rebuild=True`, `new_expert_initialization="clone"`). A new
opt-in field `probe_scoring="applied"` makes the probe score in the rebuild it applies.

**Result (branch (b), both depths, 384 seeds each):** the fused run improves decisively
(-0.04043 and -0.03478 nats, p 5e-05, sign test passing at both depths) and the delivered
artifact does not (-0.00825 p 0.13, +0.01047 p 0.86). Value over doing nothing roughly
doubles: -0.0467 -> -0.0871 at depth 2, -0.0566 -> -0.0913 at depth 3. The two rules chose
differently in 70.3% of events and the operation mix moves the way the mismatch predicts
(grows 220 -> 382), so the controller had been pruning mostly because the default scoring
made both candidates look worse than the models the pipeline would actually build.

**Two things this session cost that it should not have.** First, an honest correction:
sections 5.32/5.33's "the cost identity must hold exactly" is a claim about *selection*
rules and it is false for a *scoring* rule, which books the scored model's active-parameter
count. Measured mean 4.62%/3.12%, max 21.86%/14.50%, exactly zero on 83/77 seeds. Second,
the declared forced-operation mechanism block was **deleted, not reported**: on the probe
path `forced_operation` never binds (`adaptive_policy="probe_loss"` returns before the
branch that reads it), so a forced grow block and a forced prune block are the same runs -
which is exactly what the numbers said. The mechanism is answered from a partition of the
stored blocks instead, and both facts are pinned by `tests/test_probe_scoring.py`.

**Fast-iteration note.** The block sizes were amended *before* the run on the disclosed
12-seed pilot (mean -0.0749, sd 0.123): at 384 paired seeds the standard error is 0.0063, so
an effect four times smaller than the pilot's still clears `alpha/4` at over 99% power.
Declaring 1536 would have bought nothing measurable. All six block kinds together ran in
6m43s wall on the 20-core box (`xargs -P 18`, `OMP_NUM_THREADS=1` per process).

Updated: 2026-09-14

This document compares the current workspace against
`Adaptive Computational Topology - Complete Kaggle Implementation Plan.md`.
It is the implementation tracker for the narrowed first scientific milestone.
The narrative companion covering the six-study factor map (six gate-style
quality tests, one pass) is `RESEARCH_REPORT.md`; all quoted statistics there
are machine-checked against `artifacts/study-summaries/`.

## Training target: dev-fast by default (2026-09-15)

The pre-registered 285M-parameter section 5.1 target is no longer the default
model. It was the default because it was the *declared* model, and the effect
was that every development cycle was a protocol-scale run (285M parameters,
24.6M tokens per arm, an hour-plus per arm on the development GPU). The declared
target is unchanged and still what a declared run uses; it is now built
explicitly:

```python
V0SessionConfig()                 # fast dev shape: ~14.9M / 7.6M active
V0SessionConfig(declared=True)    # rejected unless it IS the declared config
declared_session_config()         # the pre-registered 285M target, validated
```

``DEV_V0_CONFIG`` (vocab 8k, context 512, hidden 320, depth 6, 8 experts of 2
active, expert width 320; ~14.9M total / 7.6M active) was chosen by measurement
on the GTX 1070, not by preference, and the measurement is the interesting part:
**at these sizes the MoE block is latency-bound, not FLOP-bound** (one kernel per
expert per pass), so the smallest candidate was the *slowest* - 10.0M parameters
ran at 3.6k tokens/s while 14.9M ran at 9.8k. Cycle time is governed by tokens
per run.

That latency-bound diagnosis was then acted on: the expert block now folds its
per-expert launches into batched matmuls (``_MoEFeedForward.forward``), keeping
the eager dispatch as ``eager_forward`` and pinning the two equal by test
(float64 agreement 1e-16). Measured on the dev shape, batch 8 x context 512:
51.8 ms/step eager vs 42.5 ms/step dense (+18%).

End to end through the real session path on the **real TinyStories stream**
(``scripts/dev_cycle.py``): each arm trains at **19.6k tokens/s** (0.21 s/step),
0.9 GiB peak, and a two-arm 400-step dev cycle (1.64M tokens/arm) takes
**2.8 minutes wall** instead of an hour.

```
PYTHONPATH=src python scripts/dev_cycle.py            # 400 steps, real text
PYTHONPATH=src python scripts/dev_cycle.py --steps 40 # quick smoke
# the two settings side by side (same seed, so neither resumes the other):
PYTHONPATH=src python scripts/dev_cycle.py --seed 15201 --tag base
PYTHONPATH=src python scripts/dev_cycle.py --seed 15201 --tag clone \
    --preserve-router --new-expert clone
```

Precision stays fp32 on the development card. Measured at the dev shape,
forward+backward+AdamW: fp32 25.5k tokens/s, bf16 20.6k (emulated below compute
capability 8.0), fp16 10.5k. The fp16 penalty is the arithmetic rate, not the
guard scaler (disabling it recovers only ~1.5%), so bf16 - which needs no
scaler - does not recover it. On the declared Turing-class target fp16 is
native and correct; bf16 remains emulated there too.

The dev budget is not a protocol measurement and must not be read as one: at
dev scale the probe is 1.5% of training FLOPs, not the declared 0.0942%, and the
Gates are unchanged (Gate 4 still requires the declared target at declared
fidelity). Nothing about the pre-registered protocol moved - only which
configuration a bare ``V0SessionConfig()`` means.

The launch CLI is unaffected: ``scripts/kaggle_queue.py --queue session`` builds
``declared_session_config()`` explicitly for a real launch (pinned by test), so a
declared queue cannot silently train a dev model. Fixed while making this change:
the CLI previously relied on the defaults *being* declared, which would have
downgraded a real launch silently once the defaults changed - the test that
caught it is `test_cli_default_queue_is_the_declared_block`.

## Status Legend

- DONE: implemented and covered by a passing check.
- PARTIAL: a prototype or simulator exists, but the research contract is not complete.
- TODO: not implemented in the real Python training path.
- DEFERRED: intentionally blocked until the first scientific gate passes.

## Current Boundary

The project is a single PyTorch research path (2026-09-19: the TypeScript
"laboratory simulator" and its app-builder scaffold - auth, database, PWA plugins,
vite/eslint/prettier configs, `public/`, `server/`, `migrations/` - were removed;
they had not been touched since the initial commit and represented none of the
results below). The Python foundation has dense and static-MoE models,
deterministic training, checkpoint resume, topology genomes, safe prune/grow, and
mutation-aware MoE rebuilds, plus versioned configuration schemas, deterministic
checksum-addressed token shards, explicit CPU/CUDA device contracts, and reusable
importance signal extraction for static-MoE experts.

The first real scientific result is now decided on three fronts. (1) The
pre-registered 384-seed Gate 3 study (seeds 1000-1383) FAILED for the legacy
cheap-signal controller: 195/384 wins (50.8%), mean delta `+0.006766`, binomial
p `0.3993` - small-sample noise, no benefit. (2) A probe-guided diagnostic upper
bound (`guided` arm) PASSED its 96-seed study (seeds 2000-2095): 59/96 wins,
mean delta `-0.041325`, binomial p `0.0158`, t p `0.00025` - proving topology
selection does express at this fidelity. (3) Making bounded probe-loss selection
the adaptive controller's default decision rule (cap 4 candidates, 2 probe
batches) PASSED the re-registered 384-seed Gate 3 re-run (seeds 3000-3383):
213/384 wins (55.5%, threshold 209), mean delta `-0.025955`, binomial p
`0.018138`, paired t p `7.4e-05`, matched budgets. The same pre-registration
included a search-cost criterion (probe FLOPs <= 5% of training FLOPs) which
FAILED at 41.9%: the quality gate is passed, compute-efficiency is not, and
amortization at realistic scale is the explicit next question rather than an
assumption. (4) A cheaper pre-registered variant (cap 2, 1 probe batch,
steps=144, seeds 4004-4383) satisfied the cost criterion (4.13% <= 5%,
confirming probe cost amortizes with horizon) but FAILED quality: 181/380 wins
(47.6%, threshold 207), mean delta `+0.011747`, binomial p `0.8351`, t p
`0.8857`.The current
map: probe-rich selection wins quality but misses toy-scale cost budgets;
probe-poor selection meets cost but loses the quality edge. (5) A same-seed
paired placement study (cheap probe, steps=144, mutations moved from steps
12/24/36 to 40/80/120; seeds 4004-4383) tested whether early placement explains
that failure: late placement was better in 201/380 pairs, mean paired delta
`-0.015666`, paired t p `0.0788` - directionally suggestive but NOT significant,
so placement does not statistically explain the horizon effect, even though it
changes 39% of probe decisions. The next study should vary probe cadence, not
placement. (6) A pre-registered probe-validity analysis (counterfactual branches
forcing candidate k in {1,2,3} at the first mutation event, seeds 3000-3383)
shows probe loss IS a statistically significant predictor of realized final loss
(mean per-seed Spearman rho `0.1057`, t(383) `3.638`, one-tailed p `1.374e-04`;
pooled z-scored Pearson r `0.1041`, z `4.09`) - but weakly: argmin concordance
`0.281` is near the 0.25 chance level, and the probe-best choice mainly avoids
the worst candidates (mean worst-branch excess avoided `0.0860` vs mean loss
left on the table `0.0601`). A weak per-event signal over only 3 mutation
events is consistent with the marginal Gate 3 pass and motivated a cadence
study. (7) That cadence study (cheap probe cap=2/1-batch, steps=144, interval=24,
budget=6 = six probe-loss events, fresh seeds 5000-5383) then FAILED HARD IN
REVERSE: 137/384 wins (35.7%, threshold 209), mean delta `+0.047014`, binomial
p `1.0`, and adaptive is significantly WORSE than random (worse-than-random
sign test z `-5.61`, two-sided p ~ `2e-08`). Accumulated greedy argmin on a
noisy single-batch probe is net harmful - consistent with a winner's-curse
mechanism: selecting the probe-argmin over many events amplifies probe-batch
noise instead of signal. (8) The pre-registered variance study (cap=2, FOUR
probe batches, 3 events, steps=144, fresh seeds 6000-6383) then FAILED its
quality gate: 179/384 wins (46.6%, threshold 209), mean delta `-0.003907`,
binomial p `0.9159`, t p `0.3453`. The mean shifted in the variance-hypothesis
direction (+0.0117 -> -0.0039) but not significantly (two-sample z ~ `-1.13`).
Per the pre-registered interpretation contract, probe variance reduction alone
does not restore the adaptive edge: candidate-set richness and/or the shorter
horizon of the single passing configuration remain the differentiating axes.
(9) The pre-registered horizon replication (the PASSING cap-4/2-batch config
re-run at steps=144, fresh seeds 8000-8383) PASSED and strengthened the
original: 231/384 wins (60.2%, threshold 209), mean delta `-0.052402`,
binomial p `4.04e-05`, paired-t p `1.07e-08`, matched budgets, probe cost
`14.0%` (amortization matches the fixed-per-event prediction of 41.9%/3).
The pass is therefore NOT horizon-specific; with seven gate-style tests the
study clears Bonferroni alpha/7 ~ 0.0071 on both statistics, and the contrast
with the cap-2/1-batch failure at the same horizon pins the active axis on
candidate richness + probe quality. (10) The pre-registered richness study
then isolated that axis at fixed nominal probe tokens (192/event): arm A
cap-4/1-batch PASSED (228/384 wins, mean delta `-0.036982`, binomial p
`1.4e-04`, t p `5.0e-05`) while same-seed arm B cap-2/2-batch FAILED
(193/384, mean delta `+0.007156`, binomial p `0.47965`), and the paired
A-vs-B comparison confirms A better in 225/384 pairs (sign z `3.37`,
two-sided p `8e-04`). Candidate richness is the active axis; probe batches
matter little within the rich-cap regime (A's delta `-0.037` vs the
2-batch replication's `-0.052`, two-sample z ~ 1.1, n.s.).Gate-style quality tests now number nine, with three passes, all from the cap-4 family.
(11a) The section 5.9 do-nothing control then RE-DERIVED study 3's toy
pass: run as a three-arm block (random, adaptive, null; same registered
config, fresh block 70000-70383, `independent_arm_rng=True`), churn harm
replicated at toy scale (null < random 280/384, p `4.7e-20`) and the
adaptive arm LOST to the null arm (117/384, mean `+0.0508`, p `1.0`) -
the ordering null < adaptive < random means study 3's edge is churn
suppression, not value creation (report section 5.9). (11b) The
section-5.6 depth-4 confirmation then faced the same control (section
5.10, fresh block 71000-71383, same three-arm design): BOTH primaries
pass - churn harm replicates (null < random 254/384, p `1.2e-10`) AND
adaptive beats the null arm (218/384, mean `-0.0297`, p `4.6e-03`,
clearing alpha/2) - so the depth-4 edge is genuine value creation
(~26% of the raw edge survives the do-nothing bar) and Gate 3's depth-4
evidence stands with a third control under it.
(11) The section 5.1 protocol machinery then ran on the REAL TinyStories token
stream at reduced fidelity (`local_proof_run.py --real-data`, vocab-8192 BPE
streams, seeds 9000-9011): the cost identity held again (measured probe cost
1.0077% vs 0.9977% predicted) and arm losses were real language losses, but
the ADAPTIVE LOST 11 OF 12 PAIRS and the pre-registered harm guard engaged
(`PASS_COST_ONLY_HARM_GUARD_REJECTS`). (12) The follow-up real-data selection
diagnostic (`scripts/real_data_diagnostic.py`, same paired seed block
9000-9007, steps=600, every variant's random arm reproducing its baseline
counterpart exactly, so only selection differed) tested both toy-winning axes
on real text and NEITHER RESTORED THE EDGE: variance (cap-4/4-batch, measured
probe cost 3.99% of training FLOPs) went 0/8 with mean delta `+0.2858`;
richness (cap-8/1-batch, 1.98%) went 1/8 with mean delta `+0.1661` - though
its adaptive arm beat the baseline adaptive arm by a mean `0.109` in absolute
terms, so the richness axis itself still operates. Critically, in 5/8 seeds
the 4-batch argmin chose bit-identically to the baseline 1-batch argmin
(final-loss delta exactly `0.0`): on real text the probe ranking is STABLY
MISALIGNED, not stochastic - the winner's-curse (noise-amplification)
mechanism is excluded and the ranking is systematically anti-predictive. The
next discriminator is an argmax control (deliberately select the probe-worst
candidate). (13) That argmax control has now RUN on the same paired block
(seeds 9000-9007): 0/8 wins, mean delta vs own random `+0.3048`, mean
adaptive vs baseline adaptive `+0.0293` - argmax also loses, and by almost
exactly as much as argmin (`+0.2858`). Both tails of the probe ranking lose
to random by the same amount while a non-degenerate control (0/3 identical
picks to argmin on seed 9000; operation balance 11 grow/13 prune vs argmin's
4/20 prune bias) confirms the rule really inverted: the probe signal is
UNINFORMATIVE on real text, not inverted - probe-loss ranking carries no
candidate-ordering information at this fidelity. Per the pre-registered
discriminator contract, the fix must come from probe horizon/tokens (more or
later probe data), not from the decision rule. Gate 4 stays blocked on
real-text evidence. (14) The diagnostic now logs per-candidate probe ranks
natively: a `probe_landscape` flag on `V0SessionConfig` (off by default,
mirroring the toy path) records every probed candidate's probe loss in each
adaptive event's history, sorted best-first, plus the selected candidate's
index; the diagnostic enables it and aggregates a per-variant `probe_rank`
block (mean selected rank, rank histogram, near-top rate) into
diagnostic.json, so future mechanism claims (pick identity, operation
balance, selected rank) are directly measurable from the published seed
summaries. (15) The section-5.3 post-argmax fix candidate, the `heldout`
variant (`V0SessionConfig.probe_data_source="heldout"`; probe windows drawn
from the held-out validation stream, only the windows' source differing,
1.01% measured cost), RAN on 2026-09-14 (revision `eaa1c45`) and its
pre-registered provenance hypothesis came out FALSIFIED (outcome b):
0/8 wins, mean delta vs own random `+0.2840` (baseline argmin `+0.2858`),
adaptive arm just `+0.0085` from the baseline adaptive arm, random arms
replicating. The manipulation was real - mean selected probe loss moved
4.4195 -> 4.2848 under held-out windows and 9/24 picks changed vs the
baseline argmin (more than `var`'s extra batches moved, 7/24) - yet
outcomes did not move: the ranking is a function of the model and candidate
set, not of which iid TinyStories windows score it. Probe-data provenance
is excluded as the mechanism. (16) The long-horizon
lever ran as the `long` variant (`V0SessionConfig.probe_finetune_steps=1`,
one AdamW lookahead step per probed candidate, per-event throwaway
generators, 3.00% measured cost) and its adaptation-blindness hypothesis
also came out FALSIFIED (outcome b): 0/8 wins, mean delta `+0.3106`,
adaptive arm `+0.0350` from baseline adaptive. The lookahead worked at the
ranking level - every landscape re-ranked, grows relaxed to 5/24, and on
seed 9000 the probe top-ranked the grow at 2 of 3 events (the 4/20 prune
bias is a ranking artifact of the snapshot probe, now confirmed) - but
re-ranking did not re-rank toward what helps; across seeds the correlation
of mean selected probe loss with final harm is positive (+0.51 long,
+0.89 heldout; 8 points, indicative). Unified verdict after five
diagnostic variants: the probe ranking is real, stable, and adaptive, but
anti-correlated with downstream value (argmin +0.28 to +0.31 everywhere,
argmax +0.30, only `rich` - the cap-8 variant whose 8/24 grow share
matches the random arms' - shows any win, and it most dilutes the
ranking). Every horizon/token reading of the section-5.3 licensed axis is
exhausted; the untested reading (more probe tokens at fixed fidelity)
would sharpen an anti-correlated signal and is not recommended without a
new mechanism hypothesis. Next direction, outside the section-5.3
contract and requiring its own pre-registration: change how the ranking
is used (layer-level coarseness, or accept/reject gating of random draws).
Full suite: 285 passed (4 lookahead tests). (17) That pre-registration is
now written (report section 5.4) and implemented as the `gated` arm:
`V0SessionConfig(arms=("gated",), gate_margin=theta)` draws a random
mutation FIRST (so the chooser's load-fallback sees the same post-training
routing statistics as the random arm), scores the two-point contrast - the
no-op reference (current model) and the drawn mutation on the same probe
batches - and accepts the draw unless it loses to the reference by more
than theta nats; a rejected draw is rolled back and replaced by the
draw's own load-based grow fallback so the mutation budget stays matched.
The built-in control is exact: the gated arm shares the random arm's init
identity and streams, and at `gate_margin=inf` reproduces the random arm
BIT-FOR-BIT (pinned by test, 7 new tests, suite at 292), including the
global-RNG contract (the accepted rebuild IS the probed model, one
rebuild per accepted event) and routing-load discipline (loads captured
before any probe forward). Gated sessions are single-arm and publish
alongside the baseline block's stored random arms. Cost: two probe
evaluations per live event, ~2% of training FLOPs. The diagnostic gained
a `gated` variant (default `theta=0`) with accept/reject-rate and
draw-minus-reference aggregation. RAN 2026-09-14 - see the Verified Checks
entry below and report section 5.7.

## Substrate Audit (2026-09-14, exploratory and UNREGISTERED)

A code-level audit of the gate path found two structural defects and one
confound. The runs below are exploratory diagnostics on the fresh block
20000-20127, not pre-registered studies (full write-up: `RESEARCH_REPORT.md`
section 9).

1. **The toy path could not build depth > 1.** `StaticMoEConfig` carried a
   single `num_experts` and `StaticMoEBlock` required
   `len(expert_ids) == config.num_experts`, so any genome with unequal
   per-layer expert counts - any real topology mutation at depth > 1 - raised
   `expert_ids must match num_experts` on rebuild. Every published gate study
   is `depth=1` (confirmed in `artifacts/study-summaries/*.json`), so the
   whole factor map covers one layer with a 5-candidate mutation space. FIXED:
   each block derives its expert count from its own ids.
2. **The candidate cap truncated a layer-major list.** The bounded probe took
   `feasible_expert_mutations(...)[:cap]`, so at depth > 1 a cap of 4 reached
   only layer 0.   `probe_candidate_policy` now selects `prefix` (historical default) or
   `spread` (deterministic per-event subset of the whole feasible set), in
   both the toy path and the V0 session (`V0SessionConfig`), so the declared
   protocol can be run either way. The REGISTERED section 5.1/5.2 config is
   unchanged at the `prefix` default, so the pre-registered protocol still
   carries the confound until the policy is switched.
3. **The arms differ in candidate distribution, not only selection.** Added
   `probe_selection=random` (choose uniformly among the SAME probed candidates
   at the SAME probe cost) in both paths, extending
   `V0SessionConfig.probe_selection` from {argmin, argmax}. At
   depth 4, 128 seeds, steps=24, cap 4/1-batch, uniform baseline: argmin
   88/128 (p `1.3e-05`), random-within-pool 80/128 (p `0.0030`), argmax
   64/128 (p `0.535`). The ordering is real (monotone 88 > 80 > 64), but a
   loss-blind control on the same pool also passes, so ~40% of the argmin edge
   is a pool/location effect. Section 2's "only the selection policy differs"
   is corrected. The factor map needs re-derivation with a pool-matched
   baseline before the section 5.1 cost protocol is run.

4. **The depth-4 mechanism passes cleanly once the coupling is fixed (a
   withdrawn claim, corrected).** Under `spread` at the registered horizon with
   independent arm RNG and full power (depth 4, steps=48, 384 seeds
   50000-50383): the registered comparison is 246/384 (mean `-0.0745`, p
   `1.9e-08`); a loss-blind choice from the *identical* pool also beats uniform
   churn (217/384, p `0.0062`), so the pool accounts for ~29% of the mean gain;
   and probe ordering adds a decisive increment over the same pool at the same
   cost (222/384, mean `-0.0529`, p `0.0013`). All three clear Bonferroni
   alpha/3. An earlier draft concluded from a low-power shared-RNG run that
   "the registered comparison fails on a non-degenerate pool" - that reading is
   WITHDRAWN (65/128 coupled vs 80/128 decoupled on the same seeds; the
   64-seed steps=48 block is consistent with the powered rate). Report section
   9.7.
5. **The toy arms are coupled through the global torch RNG.** A grow draws
   fresh expert weights from the stream a prune does not, so runs whose
   treatments choose differently perturb the random arm's later rebuilds (the
   random arm is bit-identical across rules in only 70/128 seeds on the audit
   block; 384/384 once fixed). The V0 path is not affected (arms train
   sequentially with per-arm seeding). Fix implemented as the opt-in
   `ExperimentConfig.independent_arm_rng` (default `False`, so published numbers
   reproduce unchanged). Convention: every new study pins it explicitly (as
   5.6 does) rather than relying on a flipped default, because the CLI is also
   the reproduction surface for the pre-audit studies, whose records do not
   carry the field. See report section 9.8. Re-run of study 3's own block 3000-3383: the shared
   path reproduces the committed artifact exactly (213/384, mean `-0.025955`,
   p `0.01814`) and the fix gives a slightly stronger 222/384 (mean `-0.034406`,
   p `0.00128`), so the depth-1 pass survives the fix. Report section 9.8.
6. **The depth-4 mechanism is CONFIRMED out-of-sample under a pre-registered
   design.** The section 5.6 pre-registration was committed (`b876ac1`) before
   the run and executed once on the never-observed block 60000-60383 (384
   seeds, depth 4, `spread`, `independent_arm_rng=True`, cap 4/2-batch,
   steps=48). Result PASS: argmin vs uniform churn 255/384 (p `6.1e-11`),
   paired argmin vs the identical-pool loss-blind control 236/384
   (p `4.1e-06`), pool-matched control 209/384 (p `0.046`), validity control
   (`argmax`) 173/384 (p `0.977`, does not pass), random-arm invariance
   384/384. Both decision comparisons clear Bonferroni alpha/3; the pool
   accounts for ~32% of the mean gain and the selection increment for ~68%.
   This is now the project's Gate 3 evidence - the first pass on a pool (20
   candidates over 4 layers) genuinely larger than the search cap, on the
   decoupled RNG path, and out-of-sample. Report sections 5.6 and 9.9.
   Hardware note: pin `--device cpu`; the default `auto` selects CUDA and 16
   processes sharing one GPU made the same study >5 min slower (>13 min before
   being killed vs ~2.5 min on CPU shards).
7. **The published factor map has been re-run on the decoupled path.**
   Section 9.8 showed all nine studies used the drifting baseline, so section
   9.10 declared a reproduction check (published config, seed block and
   treatment arm, only `independent_arm_rng` changed) and section 9.11 records
   the result on 144 shards (~6 min wall at 16 single-threaded cores;
   `artifacts/study-summaries/factor-map-recheck.json`). **9 of 10 verdict
   labels reproduce.** Every win count moves toward the treatment (+1 to +30,
   mean +9.2, all ten positive), so the coupling was a *directional* bias that made
   every published number - passes and nulls - understate the adaptive arm.
   Studies clearing Bonferroni alpha/9 rise from **2/9 to 4/9** (horizon
   4.0e-05 -> 1.7e-09, richness A 1.4e-04 -> 9.4e-05, probe-loss 0.0181 ->
   0.00128, guided 0.0158 -> 0.00279). The single **label flip** is variance
   reduction: 179/384 FAIL -> 209/384 PASS at exactly the alpha=0.05 threshold,
   which fails alpha/9 and is therefore a null in both directions - it must not
   be quoted as evidence that more probe batches help. Cadence's
   "significantly worse than random" claim survives but weakens
   (137/384 -> 148/384). The legacy signal study was initially excluded as not
   reconstructible, but section 9.12 recovered the field its config predates
   (`adaptive_policy="signal"` - proven by reproducing the published coupled
   record to 4.8e-7 where the default sits 1.1e-1 away), so its row is now in
   the set: 195/384 FAIL -> 201/384 FAIL, +6, reproduces. Net: no headline claim
   is overturned and no verdict needs revising except variance, but the depth-1
   factor map should now be quoted from the decoupled column.
8. **The probe-validity counterfactual analysis has a committed generator, and
   re-runs weaker.** The analysis behind report section 3.2 and figure 4 had
   **no generator in the repository** - nothing produced
   `artifacts/study-summaries/probe-validity-analysis.json` - so its numbers
   were reproducible only in principle. `scripts/probe_validity_analysis.py` is
   that entry point; it recovered the definitions from the committed file
   (including two non-obvious ones: `h1_one_tailed_p` is a **normal** tail, not
   a t tail, and `pooled_zscored_pearson_r` z-scores **within** each seed
   before pooling), reproduces all 13 stored fields from the committed
   `per_seed` block, and on a fresh 4-branch x 384-seed coupled sweep
   reproduces the committed results field for field. Decoupled re-run: mean
   per-seed Spearman rho **0.1057 -> 0.0776** (p 1.4e-04 -> 4.98e-03), pooled r
   0.1041 -> 0.0809, while the avoidance statistics barely move (concordance
   0.281 -> 0.284 vs chance 0.25, excess avoided 0.0860 -> 0.0839, left on
   table 0.0601 -> 0.0592). Branch 0 reproduces the study summary's adaptive arm
   seed for seed (max drift 1.9e-6: the identity is exact in principle, not
   portably bit-exact across environments). Not pre-registered - it is a
   controlled re-measurement of a frozen `design` string, and section 9.11
   labels it as such rather than as new evidence. Report sections 3.2, 8, 9.11.
9. **The legacy signal study is reconstructible, not unknowable.** Section 9.10
   excluded `gate3-legacy-signal-384` as not reconstructible, which was right
   about the *default* - its recorded config predates `adaptive_policy`, which
   the dataclass would fill with `probe_loss`, a different experiment - but
   wrong to conclude the value was unavailable. It is recoverable three ways:
   the transition commit `d972d4c` inserted the probe branch above an untouched
   legacy chooser (so `"signal"` *is* the old controller); the raw record
   `artifacts/gate3-384.jsonl` has no probe fields in any `mutation_history`;
   and supplying `"signal"` reproduces the published coupled record to 4.8e-7
   where the default sits 1.1e-1 away (pinned as a test). Declared in section
   9.12 before the run, then re-run decoupled: **195/384 FAIL -> 201/384 FAIL,
   +6, reproduces.** The corrected figure set now carries all nine rows. A
   related defect fixed on the way: the recheck's per-shard confirmation line
   read `record['per_seed']` (the probe-validity schema), which crashed a shard
   *after* it wrote its file and made completed shards look failed.

Iteration cost for these checks: ~1.5 s/seed on CPU at depth 4 (GPU measured
slower at 2.21 s/seed for a hidden-24 model); a 128-seed study is ~85 s.

## Milestone Comparison

| Plan milestone | Status | Evidence / gap |
|---|---|---|
| 1. Foundation and baseline contract | DONE for the first gate | Python model, optimizer, training state, topology genome, hashing, model factory, and versioned checkpoint metadata are covered. Large-model and real-data contracts remain outside this gate. |
| 2. Dense baseline | PARTIAL | Dense causal LM trains on deterministic synthetic batches and resumes parameter-for-parameter. It is a tiny gate model, not the planned 256M-350M research model; validation loss, FLOPs, and real data metrics are not yet recorded. |
| 2. Static MoE baseline | DONE for the first gate | Static top-k MoE trains through the shared runtime and reports validation loss, active parameters, parameter count, and validation FLOPs. |
| 3. Topology graph and safety | PARTIAL | Python genome serialization, validation, hashing, prune/grow, merge/split/rewire guards, expert-ID weight preservation, and accepted mutation fossils exist. Full graph-aware model execution and post-mutation causal checks remain deferred. |
| 4. Random topology control | DONE for the first gate | Python Experiment 0 provides reproducible random churn with matched cadence, mutation count, training/search accounting, and report metrics. Real data and scale remain deferred. |
| 5. Importance instrumentation | DONE for the first gate | Python records routing/load, gradient norm, parameter norm, Fisher-style EMA, temporal stability, mutation count, and genome hash at structural windows. |
| 6. Adaptive controller | PARTIAL | Python exposes a deterministic, validated budget-aware `AdaptiveController` plus Pareto frontier/population selection over quality, active parameters, and search FLOPs. The validated default reaches 10/16 wins; broader multi-generation search remains. |
| 7. Experiment ledger | DONE for the first gate | Browser ledger and Python append-only JSONL registry store config, dataset/tokenizer hashes, code revision, hardware, tokens, FLOPs, arm metrics, mutation fossils, and architecture hashes. Deterministic evaluation summaries are available for report generation. |
| 8. Final gate: adaptive > random | **PASSED under a pre-registered, out-of-sample depth-4 design.** Section 5.6 was registered before the run and executed once on the fresh block 60000-60383 (384 seeds, depth 4, `spread`, `independent_arm_rng=True`, cap 4/2-batch, steps=48): registered comparison 255/384 (p `6.1e-11`), controlled same-pool selection increment 236/384 (p `4.1e-06`), loss-blind same-pool control 209/384 (p `0.046`), `argmax` validity control does not pass (173/384), random-arm invariance 384/384 - all three decision rules clear Bonferroni alpha/3 where required. This is the first gate pass on a candidate pool (20 over 4 layers) genuinely larger than the cap, with the arm-RNG coupling fixed and out-of-sample. Earlier evidence: an exploratory powered depth-4 run (246/384, p `1.9e-08`; pool 217/384; increment 222/384, p `0.0013`), and three depth-1 passes on the shared-RNG path (study 3's block reproduces exactly at 213/384 and gives 222/384 with the fix) that should now be read as a degenerate 5-candidate-space instance. See the Substrate Audit section and report sections 9.7-9.9. | Factor map (all matched budgets, objective verdicts): cap-4/2-batch, 48 steps -> PASS 213/384, p `0.0181`, at 41.9% cost. Cap-2/1-batch, 144 steps -> FAIL 181/380, p `0.8351`, at 4.13% cost. Placement (late, same seeds) -> no effect (p `0.0788`). Cap-2/1-batch, 6 events -> 137/384, significantly WORSE than random (z `-5.61`). Cap-2/4-batch -> 179/384, p `0.9159`. Horizon replication cap-4/2-batch, 144 steps -> PASS 231/384, p `4.04e-05`, t p `1.07e-08`, cost 14.0%. Richness A cap-4/1-batch, 144 steps -> PASS 228/384, p `1.4e-04`, t p `5.0e-05`, cost 7.3%; same-seed B cap-2/2-batch -> FAIL 193/384, p `0.47965`; paired A>B 225/384 (z `3.37`). Probe validity real but weak (rho `0.11`); richness is the active axis, probe batches secondary. Nine gate-style tests, three passes; Bonferroni alpha/9 ~ 0.0056 is cleared by the 144-step replication and richness-A on both statistics. Real-data diagnostic (TinyStories, seeds 9000-9007, paired): variance cap-4/4-batch 0/8 (`+0.2858`), richness cap-8/1-batch 1/8 (`+0.1661`) - neither toy axis restores the edge on real text, and the 4-batch argmin bit-reproduced the 1-batch argmin in 5/8 seeds (ranking stably misaligned, not noisy). **Re-run on the decoupled arm-RNG path (sections 9.10-9.12, Substrate Audit items 7-9): 9 of 10 labels reproduce, every win count moves toward the treatment (+1..+30, mean +9.2), studies clearing alpha/9 rise 2/9 -> 4/9 (horizon 1.7e-09, richness A 9.4e-05, probe-loss 0.00128, guided 0.00279), and the single flip is the variance study (179/384 FAIL -> 209/384 threshold PASS, failing alpha/9, i.e. a null either way). Cadence's harm claim survives at 148/384 (from 137/384). The legacy-signal study's predated `adaptive_policy` field is recoverable and its row is in the set (195/384 FAIL -> 201/384 FAIL, reproduces), so no factor-map study now rests on the coupled path alone.** So the factor-map claims stand, with corrected effect sizes; quote them from the decoupled column. |

## Plan Sections 3-6: Compute and Kaggle Operations

| Area | Status | Remaining work |
|---|---|---|
| T4x2 PyTorch/DDP | PARTIAL | Add device selection and DDP launch helpers now discover torchrun rank/world-size, initialize an appropriate process group, and wrap models only for multi-process runs; scheduler, accumulation, autocast, and activation checkpointing are available. T4x2 validation remains. |
| TPU v5e/JAX | DEFERRED | Port only after a stable architecture wins the GPU gate. |
| Kaggle artifact datasets | DONE for the 5.2 handoff | Deterministic checkpoint, evaluation, report, and manifest layout contracts exist; the V0 5.2 per-seed handoff (`act.training.v0_handoff`: atomic publish, checksum-verified restore, bit-identical resume through the dataset round trip) is implemented, and the remote half is now code: `act.ops.kaggle_remote` wraps the optional kaggle package (injectable client, zero-copy push of the handoff directory with generated metadata, restore merges manifest-complete seed versions over matching locals and keeps local-only ones) with `--kaggle-dataset owner/slug` on the launch command - restore at session start (fail-fast before any training), publish-on-exit that never raises (outcome recorded in `remote-publish-log.json`). |
| Session resume protocol | DONE for the 5.2 handoff | Local checkpoint resume is deterministic for baselines and bit-identical for V0 sessions (verified mid-run resume: losses, FLOPs counters, architecture hashes, and mutation histories all reproduce exactly); the 5.2 versioned-dataset handoff round-trips a seed working-dir -> dataset -> working-dir bit-identically, and the remote leg now exists (`--kaggle-dataset` restore at session start, publish-on-exit; remote seed versions merge over matching locals, local-only seeds are kept, partial remote versions are ignored). |
| Repository structure | PARTIAL | Core Python data, evaluation, evolution, importance, model, topology, training, and Kaggle-artifact implementations exist with focused tests; all ten plan section 39 notebooks (`00`-`09`) now exist under `notebooks/` (deterministic builder, tested), while standalone research entry points remain. |

## Plan Sections 8-24: Model and Topology

| Area | Status | Remaining work |
|---|---|---|
| V0 256M-350M model | PARTIAL | Bounded V0 construction now runs finite forward/loss probes and reports parameter/logit shapes; target-scale memory and throughput measurements remain. |
| Attention/GDN/hybrid model core | PARTIAL | Python runtime supports dense, static MoE and the gated-delta mixer (`layer.module` selects it); GDN at the ~100M zoo scale OOMs on 8 GB (`ARCH_SEARCH_FINDINGS.md` §7). |
| Full genome contract | PARTIAL | Python genome now represents heads, modules, widths, residual paths, context, and compute ceilings and can reconstruct the V0 dense/hybrid model; MoE/shared-expert reconstruction remains. |
| Weight/topology separation | DONE for the first gate | Model and optimizer remain separate from the Python genome, while versioned checkpoints and Experiment 0 fossils package genome and mutation metadata together. |
| Importance levels and lifecycle | PARTIAL | Python exposes routing, gradient, parameter, Fisher-style EMA, magnitude, temporal-stability records, temporary ablation deltas, and append-only lifecycle fossils; population aggregation remains. |
| Expert ecology and similarity graph | PARTIAL | Device-safe expert signal extraction, functional sketches, and cosine similarity primitives exist; persistent graphs and lifecycle fossils remain. |

## Plan Sections 25-42: Data, Training, and Reproducibility

| Area | Status | Remaining work |
|---|---|---|
| Real tokenizer and shards | PARTIAL | Versioned deterministic binary token shards, bounded reads, checksums, manifests, and synthetic fixtures exist; real tokenizer/corpus adapters remain. |
| Real language-model validation | PARTIAL | Add real held-out batches; bounded perplexity and deterministic named capability probes now report weighted loss, perplexity, accuracy, and token counts, while real corpora remain. |
| Complete checkpoint format | PARTIAL | Versioned local checkpoints persist model/optimizer/training RNG state plus scheduler, genome, topology history, importance, dataset position, config, and revision metadata; compatibility validation and bounded session retention are implemented, while remote/session resume remains. |
| Experiment registry | DONE for the first gate | Append-only JSONL records include experiment/config/dataset/tokenizer hashes, hardware, tokens, FLOPs, architecture hashes, and arm results. |
| Reproducibility hashes | DONE for the first gate | Genome, config, dataset, tokenizer, and complete experiment hashes are recorded. |
| Research notebooks/scripts | PARTIAL | Dense and Experiment 0 CLIs now support baseline runs, append-only records, and multi-seed gate studies; extraction, retrain, and compare entry points remain. |

## Plan Sections 43-66: Scientific Experiments

| Experiment | Status |
|---|---|
| Experiment 0: fixed vs random vs adaptive | PARTIAL | Matched-budget Python runner, metrics, append-only single/study registry records, reproducible multi-seed CLI study, and a 6/8 probe exist; universal Gate 3 is still open. |
| Conventional dense/static-MoE frontier | PARTIAL |
| Dynamic sparse prune/grow | PARTIAL | Implemented in the Python Experiment 0 runner for safe expert prune/grow; richer topology operators remain. |
| Expert ecology clone/split/merge/death | TODO |
| Module-type evolution | DEFERRED |
| Adaptive depth | PARTIAL | Python topology mutations can safely toggle effective layer depth and change attention/GDN/hybrid module metadata; full multi-generation training comparisons remain. |
| Hierarchical search | PARTIAL | Deterministic layer/module candidate ranking now supports explicit depth penalties and inspectable priors. |
| Qwen-informed search | PARTIAL | Prior interface is available for imported architecture scores; no external Qwen artifacts are bundled or claimed. |
| Teacher initialization | PARTIAL | Explicit teacher and evolved shape-safe parameter transfer policies now exist; multi-seed retraining comparisons remain. |
| Architecture reusability and fresh initialization | PARTIAL | Fresh, evolved, and rewound initialization policies report copied/skipped tensors; architecture-equivalence studies remain. |
| Cross-seed stability and architecture equivalence | PARTIAL | Strict architecture-hash equivalence and cross-seed mean/stddev/min/max summaries now exist; full retraining studies remain. |
| Compute-efficiency scaling | PRE-REGISTERED (execution path built; resumption authorized; campaign in progress) | Real-scale measurement protocol pre-registered in `RESEARCH_REPORT.md` section 5.1: declared 285M V0 config, predicted probe cost 0.094% (band 0.047-0.189%), 5% criterion + tolerance band + harm guard on 12 seed pairs. The active block is **15024-15035** (section 5.65 re-reserved it after four seeds of 15000-15011 ran under the pre-5.47 rebuild and their harm numbers were read), with primary `probe_candidate_policy=spread`, pool-matched control, and null clause. The primary has 3/12 seeds published; kernel v9 is running under the campaign driver. No verdict exists. This is not a registration amendment; PASS authorizes Gate 4. See section 5.5 and the unchanged procedure in `docs/T4_RUNBOOK.md`. The V0 training path is implemented: declared `V0MoECausalLM` reproduces 285,336,576 / 134,341,632 parameters exactly, per-layer expert genomes rebuild weight-preserving after grow/prune, and `act.training.cli --mode v0` runs resumable sessions with version-4 checkpoints (FLOPs counters, genome, generator state persisted). Session integration checks live in `scripts/v0_session_selfcheck.py` (run on demand, NOT in the pytest suite). GPU execution plan pre-committed in section 5.2: per-GPU fp32 fit ~4.5 GB of 16 GB, 3.19 GB checkpoint payloads, `max_checkpoints=2` (pair steady state ~14.9 GB of the 20 GB working dir), throughput-gated dtype rule (fp32 floor 2,500 tokens/s, fp16 fallback), parameterized session schedule with atomic pairs and deadline/publish rules, and the versioned-dataset checkpoint handoff (~45 GB ledger). The 5.2 pre-launch tooling is now code: `scripts/v0_throughput_benchmark.py` (declared-model step-loop benchmark emitting the dtype decision and schedule artifact) and `act.training.session_plan` (registered schedule/deadline/working-dir arithmetic, unit-tested against the 5.2 tables); declared sessions now train with checkpointed activations as specified, with resume re-verified bit-identical; `act.training.v0_runner.run_pair_session` ties benchmark numbers, deadline rules, session training, and the dataset handoff into the per-seed session loop (atomic pairs, in-progress snapshots on deadline stops, restart-from-scratch, skip-completed idempotence), and `--mode pair-session` exposes it as the single launch command with the dtype rule enforced against the benchmark artifact; the fp16 contingency trains under autocast + GradScaler with scaler state in the version-5 checkpoint contract. |

## Plan Sections 67-105: Scale and Final Deliverables

These remain TODO or DEFERRED until the first gate passes:

- full metric and visualization report (PARTIAL: the four pre-registered
  factor-map figures now exist under `artifacts/study-summaries/figures/`,
  generated reproducibly from the versioned summaries; trajectory plots and
  real-scale metric visualizations remain);
- topology fossils and trajectory plots;
- Multi-generation Pareto population search and richer diversity preservation;
- T4x2 scaling to 256M-350M;
- Qwen analyzer and architecture priors;
- fresh/rewound/teacher retraining;
- 500M-1B scaling;
- TPU port;
- paper-quality comparisons and final research report;
- versioned Kaggle datasets and reproducible notebooks (PARTIAL: all ten
  plan section 39 notebooks - `00_environment` plus `01`-`03` per section
  104 and `04_importance`, `05_adaptive`, `06_search`, `07_extract`,
  `08_retrain`, `09_compare` - now exist under `notebooks/`, emitted by the
  deterministic builder `scripts/build_notebooks.py` (byte-identical
  re-runs, unit-tested in `tests/test_notebooks.py`, every cell executed
  end-to-end); the remote dataset publishing is now code as well
  (`act.ops.kaggle_remote` + `--kaggle-dataset`, below) with the remote
  round trip now live-verified with real credentials); the §5.1 execution
  is operator-scripted in `T4_RUNBOOK.md` with the verdict machine-checked
  by `scripts/v0_verdict.py` - the only open piece is the T4 run itself;

## Immediate Execution Order

1. DONE: add Python metrics for validation loss, parameters, active parameters, training FLOPs,
   search FLOPs, validation FLOPs, and architecture distance.
2. DONE for the first gate: expand the Python checkpoint to include genome, mutation history,
   optimizer/scheduler, dataset position, and importance/controller state.
3. DONE for the first gate: implement a Python structural clock and random-churn controller
   using the existing prune/grow and rebuild APIs.
4. DONE for the first gate: implement cheap Python importance records: routing/load, gradient norm,
   parameter norm, Fisher-style running estimate, and temporal stability.
5. PARTIAL: use persisted routing records in the adaptive prune/grow controller while retaining
   the slower 48-step / 12-step structural clock and exact random budget. Signal sweeps did not
   yet improve the 6/8 default result.
6. DONE: add validation loss, active-parameter, and validation-FLOPs reporting to the runner.
7. RESOLVED (PASSED on quality, cost open): the cap-4/2-batch probe-loss
   controller passed the 384-seed Gate 3 re-run (213/384, p `0.018138`) at 41.9%
   search cost; a cheaper cap-2/1-batch variant at steps=144 then met the cost
   budget (4.13%) but failed quality (181/380, p `0.8351`) on fresh seeds
   4004-4383.
- `PYTHONPATH=src python -m act.training.cli --mode pair-session ...
  --kaggle-dataset owner/slug` -> the remote publish-on-exit wrapper
  (`act.ops.kaggle_remote`), **live-verified against the real Kaggle API
  with credentials**: publish -> download -> bit-identical seed handoff
  round trip (all manifest checksums re-verified on the returned copy),
  and the registered one-line launch command completes restore ->
  skip-complete -> exit publish with exit 0. The live probes established
  the upload layout the adapter implements: Kaggle datasets are flat, but
  the service auto-extracts uploaded tar/zip blobs preserving internal
  paths, and the package's default directory handling is unusable for
  nested trees (`skip` drops them silently; `tar` archives contents
  without a name prefix, so multiple seeds would collide) - the adapter
  therefore stages one tar per seed with paths rooted at `seed-NNNN/...`
  and pushes them flat. A nonexistent dataset (session 1) is the normal
  bootstrap state on restore (warn and train from scratch; atomic pairs
  make it safe) and falls back to `dataset_create_new` on publish; any
  other error is fail-fast at launch and never-raising at exit. The test
  dataset was deleted after the round trip; unit tests still inject a
  fake client, so the kaggle package is never needed by the suite.
8. RESOLVED (Gate 3 re-derived and PASSED on a non-degenerate substrate): the
   substrate audit qualified all three depth-1 passes (5-candidate single-layer
   space, depth > 1 unbuildable, a loss-blind same-pool choice reproducing part
   of the effect), so Gate 3 was re-derived rather than assumed. The section 5.6
   pre-registration was committed before the run and executed once on the fresh
   block 60000-60383: registered comparison 255/384 (p `6.1e-11`) and the
   controlled same-pool selection increment 236/384 (p `4.1e-06`), both clearing
   Bonferroni alpha/3, with the pool-matched control clearing only alpha
   (209/384, p `0.046`) and the `argmax` validity control not passing. Gate 3 is
   now established on a pool larger than the cap, out-of-sample, with the
   arm-RNG coupling fixed (report sections 5.6, 9.9; Substrate Audit item 6).
   NEXT (compute-efficiency): cap-4/1-batch cuts probe cost to
   7.3% at toy scale and amortizes with horizon, but real-scale measurement is
   the plan's deferred compute-efficiency milestone and gates Gate 4 (adaptive >
   static MoE). Controller default stays cap-4/2-batch; cap-4/1-batch is the
   validated cost-saving option. The real-scale measurement is now
   **pre-registered** (`RESEARCH_REPORT.md` section 5.1): declared V0 config
   (285,336,576 total / 134,341,632 active parameters), predicted probe cost
   0.094% of training FLOPs under the measured per-event identity, decision
   rules fixed (5% criterion + factor-2 tolerance band + anti-adaptivity harm
   guard) (+ the section 5.5 pool-matched control and, per section 5.14, the
   null queue). Running it as specified - without retuning knobs against the
   result - is the entry condition for Gate 4. **Section 5.39 gives the null
   queue a measured bar (2026-09-16)**: at dev fidelity on the real stream,
   doing nothing beat churning on 18/18 seeds under both the historical rebuild
   and the section 5.35 repair (-0.16449 and -0.11279 nats, permutation p
   5e-05), so the repair removes 31% of the churn cost rather than all of it and
   an adaptive arm that merely matches random is 0.11279 nats worse than doing
   nothing. The do-nothing clause is load-bearing, not insurance, and the
   declared run must carry the null queue. **The protocol is now REGISTERED AS
   REVISED**: `RESEARCH_REPORT.md` section 5.5 was adopted in full
   (2026-09-14) and implemented, so the launch is unblocked and the only
   remaining prerequisite is the T4 hardware. The revision was needed for
   three reasons. The registered prefix controller reaches only layer 0 at the
   declared depth; the pool-matched control arm was the missing quality check
   (its premise is now backed by the registered section 5.6 PASS); and **the
   declared seed block 9000-9011 was already spent**, so the protocol could no
   longer be a blind measurement - the section 5.3 diagnostic ran the full
   queue through the same `run_pair_session` at reduced fidelity and a 5.1
   verdict was computed over it (`artifacts/real-proof/datasets/
   v0-5-1-verdict.json`: measured cost `1.0077%`,
   `PASS_COST_ONLY_HARM_GUARD_REJECTS`, `gate_4_authorized: false`, adaptive
   losing 11 of 12 pairs), meaning the harm guard - one of the verdict
   branches - was no longer blind. Running that queue again at full fidelity
   after seeing a harm-guard rejection is the "measure again until it passes"
   pattern the protocol exists to prevent. Adoption moved the queue to the
   unused block **15000-15011** (`v0_runner.DECLARED_SEEDS`), made the primary
   controller `spread` and added the pool-matched control as a second session
   (both exposed as `--probe-candidate-policy` / `--probe-selection` on
   `--mode pair-session`), and gave `scripts/v0_verdict.py` the control block,
   the interpretation clause (`POOL_CONTROL_WIN_LIMIT` = 10, the same bar the
   harm guard uses) and a `--control-root` flag. A missing control withholds
   Gate 4 rather than being silently skipped. The section 5.2
   pre-launch tooling (throughput benchmark, session scheduler, deadline rules,
   working-dir ledger, versioned-dataset handoff) is now code and unit-tested,
   and the pieces are tied together by `act.training.v0_runner.run_pair_session`,
   the per-seed session loop over the declared queue 15000-15011 (atomic pairs,
   rule-1 gating, cooperative deadline stops with in-progress snapshots,
   restart-from-scratch, skip-completed idempotence). The launch command is
   `python -m act.training.cli --mode pair-session`: it reads the session-1
   benchmark artifact, re-derives and cross-checks the dtype decision,
   applies the registered max_checkpoints=2 pin, and runs the session loop
   with the dataset handoff. The registered fp16 contingency is executable:
   fp16 sessions train under autocast with a GradScaler (checkpoint contract
   v5 carries the scaler state; fp16 resume verified bit-identical, version-4
   payloads still load in fp32). What remains before launch is running the
   benchmark on the T4 to fix the measured dtype and schedule; the whole
   execution is now operator-scripted in `T4_RUNBOOK.md` (pre-flight,
   benchmark, the one-line launch command, queue rules, machine-checked
   verdict via `scripts/v0_verdict.py`, contingencies, artifact trail), and
   `scripts/v0_verdict.py` computes the pre-registered §5.1 verdict
   (aggregate cost vs 5% criterion, tolerance band from
   `expected_probe_cost_percent`, harm-guard tally, `gate_4_authorized`) so
   no verdict number is hand-computed. The launch
   path is now reproducible-notebook-ready: all ten plan section 39
   notebooks (`00_environment` through `09_compare`) exist under
   `notebooks/` (deliverable 2), so a session can verify the compute
   contract before `--mode pair-session` and reproduce the baseline, churn,
   gate-study, importance, search, extraction, retraining, and comparison
   claims from the same committed revision.

## Verified Checks

- **Section 5.1 T4 pre-flight DONE on a real Kaggle T4 (2026-09-15).** The
  runbook's pre-flight 4 and section 1 ran as a script kernel on a
  `NvidiaTeslaT4`: the declared config reproduced 285,336,576 total /
  134,341,632 active parameters, and the benchmark (`--steps 50`, forward +
  backward + AdamW with checkpointed activations) measured **1,958.7
tokens/s -> `dtype_decision: fp16`** (below the 2,500 floor, the registered
  contingency). Derived schedule: `pair_time_hours` 3.585,
  `full_pairs_per_session` 2, `sessions_for_primary` 6, `working_dir_headroom_gb`
  +5.12. The artifact is `artifacts/v0-benchmark.json` (gitignored) and was
  downloaded from the kernel output, so the launch command's input exists and
  is not a hand-edited file.
- **Two schedule defects found and fixed 2026-09-15; they nearly cancel, so
  the recorded total was accidentally right for the wrong reasons.**
  (a) *The schedule was built on the rule's input, not the runtime dtype.*
  Section 5.2's rule reads the **fp32** rate to choose a dtype, and the
  schedule was parameterized from that same number while the sessions train
  fp16. Measured on the T4 (`--queue dtypeprobe`): fp32 **2,066.3** tokens/s
  against fp16 **4,694.9**, i.e. 2.27x. The benchmark now measures the decided
  dtype too and records `schedule_tokens_per_second` +
  `tokens_per_second_by_dtype` beside the fp32 `measured_tokens_per_second`,
  and the launch command schedules from the former while still enforcing the
  pre-committed rule against the latter.
  (b) *A pair's arms are trained sequentially, not one per GPU.*
  `pair_time_hours` was documented as "one per GPU in parallel ... the wall
  time is one arm's training time", and `PairSchedule.arms` was left unused in
  that formula - the arithmetic tell. `run_v0_session` trains `config.arms`
  with a sequential generator on one device, measured on a real pair (arm wall
  times 0.531 s + 0.298 s added rather than overlapping), so every pair cost
  twice what the clock believed. That is the dangerous direction: the deadline
  rule would start pairs it could not finish, and section 5.2 restarts an
  interrupted pair from scratch. The multiplicity is now in the formula and in
  the runner's own deadline arithmetic.  Corrected plan, re-measured by a pre-flight on 2026-09-15 (fp32 2,124.2 /
  fp16 5,433.2, pair 2.61 h, 3 pairs/session): primary 12 x 2.61 = ~31 GPU-h,
  control ~31, null (one arm) 12 x 1.36 = ~16, so ~79 GPU-h - the recorded
  "2c ~0.3" does not follow from the runner, since an arm's token budget is
  `batch x context x steps` independent of arm count. Dtype stays an execution
  detail touching no pre-registered criterion.
  **The same pre-flight answered the device question: `gpu_inventory` reports
  two `Tesla T4`s and `nvidia-smi` shows both idle.** So the pair model's
  "one per GPU" is reachable, the second T4 has been idle for the entire run,
  and running each arm on its own device is the remaining change that takes
  the pair to 1.36 h / 6 pairs per session / 2 sessions for the primary queue -
  ~47 GPU-h for the whole run instead of 79. *(Superseded 2026-09-20: at the
  measured 5,083.4 tokens/s the pair is 1.443 h, the runner's gate packs 5 per
  session (3 per queue), and the campaign is ~52 GPU-h of training over nine
  sessions - see the entry at the top of this file.)* That is the parallelism section
  5.2 already specifies, so implementing it aligns execution with the
  registration rather than changing it. The benchmark also refuses
 to write an artifact
  whose schedule fits no pair: on a non-T4 GPU the decided fp16 measured 4.2x
  *slower* than fp32, and the artifact described a session with
  `full_pairs_per_session: 0` - billing a whole window to publish nothing,
  silently, because the deadline rules never start a pair the clock cannot
  finish.
- **Section 5.2's "one arm per GPU" is implemented (2026-09-15).** The two-T4
  measurement above made the registered parallelism reachable, so a pair whose
  host has a device per arm now trains each arm in its own process
  (`src/act/training/arm_worker.py`, driven by `v0_session._train_arms`):
  `arm_devices` places arm *i* on `cuda:i` when the host has that many devices
  and on one device otherwise, `arm_slots` reports the same placement as a
  concurrency, and `PairSchedule.pair_time_hours` is charged per *wave* of
  arms - so the plan, the runner's own deadline arithmetic and the execution all
  read the placement from one definition instead of three copies.
  A process per arm rather than threads because model init consumes global
  torch RNG and `torch.manual_seed` is global. A **deadline stop crosses the
  boundary as a stop**, not as a failure: the child exits a distinct status and
  records the step it stopped at, and the parent re-raises
  `V0SessionInterrupted`, so the sequential path's in-progress snapshot still
  happens (`[6/6]` in the battery checks this). Without it a stop would have
  been recorded as `training_failed`, losing the snapshot and reporting an
  orderly deadline as a crash. A pair that raises for any *other* reason is
  recorded as `training_failed` and stops the queue instead of crashing the
  session: the durable copy is the exit publish, which a crash never reaches,
  so an OOM at seed 6 would otherwise discard the five pairs before it. The
  driver reads that action from the session log and exits 1 rather than
  pushing a session that would repeat the failure. Correctness is equivalence: the
  arms share nothing but the seed, so `scripts/v0_session_selfcheck.py`'s fifth
  check runs both paths and requires bit-identical arm results (losses,
  architecture hashes, FLOPs, mutation histories) through the process runner
  *and* through `run_v0_session`. Every arm now records the device it used in
  its result, so the published per-seed summary carries the placement as
  evidence rather than assertion. This changes no pre-registered criterion -
  it is the configuration section 5.2 prices; the sequential path remains the
  default wherever the host lacks the devices.
  **Confirmed on the target host.** The two-device path cannot be exercised on
  the project's one-GPU box, so `--queue selfcheck` was added as a one-shot job
  that runs the battery in a Kaggle kernel. On a real two-T4 kernel
  (`sid123123234/act-t4-selfcheck`, 2026-09-15, ~2 min of GPU) it reported
  `host devices: 2` and `one arm per device: cuda:0, cuda:1` with every check
  passing - the placement section 5.2 prices, measured on the hardware that has
  it, before any session assumed it. Nothing else moved: the same tiny pairs
  produce the same losses, FLOPs and architecture hashes however the arms are
  scheduled.
- **A checkpoint is never left truncated at the name retention finds.**
  `save_v0_checkpoint` wrote straight to `checkpoint-NNNNNN.pt`. A declared
  checkpoint is ~3.4 GB, so the write takes seconds, and a process killed
  inside it (the OOM killer, a 12-hour kernel timeout) would leave a truncated
  file at the final name - which `retain_checkpoints` takes for the newest,
  `publish_seed` checksums into a `complete` version, and restore's checksum
  verification cannot distinguish from a good one. Then the seed's published
  state is unrestorable while every check says it is a complete version, and
  nothing downstream loads checkpoints (the verdict reads summaries), so no
  reader would notice. The write now goes to a sibling `.partial` name and is
  renamed into place, and a test kills the write and asserts the final name is
  still the intact checkpoint (teeth-checked: reverting the rename fails it).
  The battery's newest-checkpoint helper now calls the production
  `newest_arm_checkpoint` instead of ordering by mtime, so a resume check
  cannot compare a different file than the run would resume from.
- **A provisional verdict on a partial block is not the driver's exit code.**
  The verdict script deliberately computes a provisional verdict on an
  incomplete declared block (it records `seeds_missing`, prints a warning and
  withholds Gate 4), which is useful as a progress report - but it returns 0
  for a provisional `PASS`, and the driver mirrors that code as the run's
  outcome, so a scripted caller could read a partial block as the registered
  decision. The driver now checks `queue_complete` in the artifact and returns
  3 (`no verdict produced`) when it is false, printing the reason and the
  number of missing seeds. The script's semantics are untouched - its own tests
  still pin the provisional PASS/exit 0 - because the ambiguity is in the
  driver's contract about the *run*, not in the analysis.
- **The driver reports whether a session's pairs survived.** The exit publish
  is the only durable copy, so its outcome is the number that matters in an
  unattended multi-hour drive - and it was previously visible only to someone
  who downloaded the kernel output by hand. The driver now reads
  `remote-publish-log.json` (which `mode_session` keeps beside the session log)
  after every session, prints the status and seed count, and stops when it says
  `failed`: re-training a session's pairs to re-attempt the push costs a whole
  window, and the usual cause is transient, which is why the push already
  retried. The name is pinned against `kaggle_remote.PUBLISH_LOG_NAME` by a
  test, because a drifted constant would read as "nothing published".
- The whole run is now reproducible code, not operator lore:
  `scripts/kaggle_t4.py` drives every session kind as a script kernel
  (`ACT_T4_MODE` = `metadata` / `probe` / `preflight` / `session` / `control` /
  `null`), generates the push metadata, copies the attached code dataset into
  the writable working dir, re-runs the benchmark in-session (a fresh session
  has no `artifacts/`, and the launch command refuses a stale artifact), runs
  the registered command unchanged, and keeps the session/publish logs in the
  kernel output. It absorbs two Kaggle behaviours that each cost a failed
  run: attached datasets mount **nested** (`/kaggle/input/datasets/<owner>/<slug>/`
  on the current runtime, so the code root is found by searching the mount
  tree for `pyproject.toml` + `src/act` rather than assuming a depth), and
  `gpu_type` is **not** an accelerator pin - only `machine_shape`
  (`NvidiaTeslaT4`) is honoured, so the previous `gpu_type: t4` push ran
  whatever Kaggle assigned. Pinned by 11 tests in `tests/test_kaggle_t4.py`
  (every mount layout, the exact per-queue launch argv for sections 2/2b/2c,
  disjoint datasets/roots, and the metadata key set).
- **The Kaggle dataset API authenticates inside a kernel without secrets**
  (probed 2026-09-15 as `ACT_T4_MODE=probe`): `KaggleApi().authenticate()`
  succeeds from the session's own identity with no `KAGGLE_USERNAME`,
  `KAGGLE_KEY`, or `kaggle.json`, and `dataset_list(mine=True)` returns the
  account's datasets. So the section 2 restore/publish legs need no User
  Secrets - only `enable_internet: true`, which the generated metadata sets.
- **Quota plan for the section 5.1 queues** (measured, not assumed). Superseded
  twice, and the corrected figure is ~47 GPU-hours against the free tier's
  30 GPU-h/week - about two weeks of sessions, not three. *(Corrected
  2026-09-20: the figure is ~52 GPU-hours of training over nine sessions by the
  runner's own start gate, ~55 billed; see the entry at the top of this file.
  The ~47 predates the measured 5,083.4 tokens/s rate and priced the null queue
  at dev fidelity.)* First the 3.585 h
  pair time came from the fp32 rate the dtype *rule* reads while the arms train
  fp16 (2.27x pessimistic), then it charged two waves of arms where one arm per
  device is available on the two-T4 shape (2x). At the measured fp16 rate with
  one arm per device a pair is 1.36 h, so each of the three queues is ~16
  GPU-hours (12 pairs) and two sessions each at `--hours 9` (the registered
  usable window) place six completed pairs. (**Corrected 2026-09-21**: the
  declared window is now 7 h and packs four pairs per session over the same
  three sessions, because the platform kills this shape at 8 h - see the entry
  at the top of this file. The 9 h reading is kept as the historical record.)
  Both queues and the null queue bill
  only the time they actually train: the deadline rules stop a session early
  rather than fill the window.
- **The section 4 verdict is reachable after a Kaggle run, and its exit codes
  are now unambiguous.** That command reads *local* dataset roots, which are
  empty once the queues live in published Kaggle datasets, so
  `ACT_T4_MODE=verdict` restores all three queues (each session publishes the
  whole handoff directory, so the newest published version is cumulative) and
  then runs the registered verdict, keeping the verdict JSON in the kernel
  output; it is a CPU kernel, so it never queues behind a GPU. Auditing that
  path exposed a real defect: `scripts/v0_verdict.py` exited **1** both for
  the pre-registered `FAIL` (primary cost > 5%) and for every
  input-validation error (`raise SystemExit("...")` defaults to 1), so over
  an unattended multi-session run a wrong `--dataset-root` would have been
  recorded as "the amortization hypothesis is false". Validation errors now
  raise `InputError` and exit the distinct code **3**, writing no verdict;
  0/1/2 keep their registered meanings exactly. Verified end-to-end: a
  nonexistent root exits 3 with `no verdict produced: no handoff dataset
  under ...` and no JSON written, while the three-queue command composes over
  the reduced-fidelity `real-diag` roots (`rich` / `argmax` / `null`) and
  writes a coherent verdict. Pinned by 2 new verdict tests and a driver test
  asserting exit 3 is not treated as a verdict code.
- **The re-push loop is mechanised** (`scripts/kaggle_queue.py`), because the
  three queues are ~52 GPU-h of training against a 30 GPU-h week (three
  sessions each at the measured rate, with one arm per device - the "~47" and
  "two sessions" here were the registered packing; corrected 2026-09-20) and "re-run the same command until
  the queue completes" should not depend on someone remembering - nor on
  someone remembering to stop when the week's quota is nearly gone. It
  stages the kernel, pushes it, waits for a terminal status, reads the
  session's own log, and re-pushes until the declared block is complete (exit
  0 complete, 1 session failed, 2 session budget exhausted). Progress comes
  from the log, not the dataset: the runner labels every declared seed
  `skipped_complete` or `completed` as it walks the queue, so the count is
  cumulative and costs no multi-gigabyte download per session. Staging also
  encodes a trap that would have silently wasted a session - a Kaggle script
  kernel receives *only* `code_file` (the CLI sends that file's text and
  nothing else from the push folder, and the metadata has no environment
  field), so the staged entry script is generated with the queue mode pinned
  into its `__main__` guard, and staging fails loudly if that guard moves.
  Verified by executing a staged entry with no `ACT_T4_MODE` set: it resolves
  its pinned mode and reaches the mount check instead of the mode error.
  Pinned by 16 tests.
- **Disk budget: the run as documented could not have completed.** Measured on
  the T4 runtime (2026-09-15, `ACT_T4_MODE=probe`): `/kaggle/working` is a
  20 GB loopback mount (19.5 GiB usable) while the container's root overlay has
  7.9 TB with 1.1 TB free. A completed seed publishes its final checkpoint per
  arm *including the AdamW moments* (`save_v0_checkpoint` stores `model` plus
  `optimizer` state, ~3.19 GiB per arm), so ~6.4 GiB per seed and ~77 GiB over
  the declared block - against a 19.5 GiB working mount, and the section 5.2
  ledger (14.9 GiB steady state + 5.1 GiB headroom) accounts for the *session*
  alone, not the accumulating dataset root. A session would therefore have hit
  `ENOSPC` during its second pair, and because the per-pair local publish is
  called unguarded the kernel would have aborted *before* the exit publish -
  publishing nothing, despite a completed pair. Fixed by pointing the handoff
  roots at the overlay (`/tmp/act-lab-<mode>/artifacts/...`, mode-scoped because
  the overlay is host-shared) instead of the working directory: nothing in
  those roots needs to survive a session, since the published dataset is
  restored at session start and pushed again at exit, and the 20 GB working
  directory is left holding only the code copy and the small audit files.
  `report_resources()` now logs disk headroom and the handoff-root size at the
  start and end of every session, so the growth is measured rather than
  assumed.
- **A failed local handoff publish no longer discards completed work.** The
  exit publish never raises by design (a broken push must not destroy a
  session); the per-pair local publish needed the mirror-image guarantee, since
  it was called unguarded and any failure aborted the kernel before the exit
  publish could push the seeds that *had* completed. A failure is now recorded
  as `action: publish_failed` (never `completed`, so skip-completed and the
  queue accounting stay honest), it stops the queue rather than spending more
  GPU on a substrate that cannot publish, and the session exit-publishes
  everything already finished; the failed pair restarts from scratch next
  session. Pinned by a runner test that fails the second publish and asserts
  the first seed stays `complete`, the second does not, and the third is never
  attempted.
- **The overlay work root is collision-proof and self-cleaning.** Because the
  root lives on a host-shared overlay, it is scoped by mode *and* by run:
  `KAGGLE_CONTAINER_NAME` carries the run identity (probed 2026-09-15:
  `kaggle_<hash>-<number>-webtier`), hashed into the directory name. The queues
  are driven one session at a time by design - quota, not parallelism, is the
  constraint - so this only matters when an overlap happens by accident, and it
  makes that harmless rather than corrupting a directory another kernel is
  using. The root is deleted at session end once the exit publish reports
  anything other than `failed`, and kept when it fails (the overlay copy is
  then the only copy of that session's completions); outside a kernel the root
  is the repository itself, so the reclaim is a hard no-op there and is tested
  to be one.
- **Pushing a kernel does not supersede a running session** (measured
  2026-09-15): a second `kaggle kernels push` left the first session running,
  and the burn rate against the weekly quota line went 1.00x -> 2.00x while
  only the latest version is reported by `kernels status`. This was hit for
  real: a corrected session was pushed over a running one and both burned GPU
  for ~15 min (~0.8 GPU-h) before it was caught by sampling the quota line
  twice and differencing it - a burn rate is the only way to see a second
  session. `kaggle kernels delete <owner>/<slug> --yes` does stop the runs
  (verified: burn fell to 0.00x), but it stops every version, so recovery is
  delete-then-push-once; stopping a single version is impossible (the CLI's
  versioned form raises in argument parsing and the API has no version
  parameter). Prevented in code now: `kaggle_queue.should_adopt` makes the
  driver adopt a live session instead of adding a second, pinned by a test.
- **The measured rate was not tied to a device count, and that is now fixed.**
  Pre-flight logged `nvidia-smi` as `stdout[-800:]`, which is the *processes*
  table, so the GPU inventory table never reached the log; `machine_shape`
  cannot settle it either (free-form string, no SDK enum, and the account's own
  "T4 x2" notebook declares the same `NvidiaTeslaT4`), and training reaches the
  GPU only through `resolve_device("auto")` - plain `cuda`, i.e. device 0 - so
  a second device would be idle and invisible. `act.training.runtime
  .gpu_inventory()` now reports every visible device (`count` + `names`); the
  benchmark writes it into `artifacts/v0-benchmark.json` beside the rate and
  pre-flight logs it, so the next session settles how many T4s the shape hands
  out. It deliberately reports only host facts, not which device trained, so
  it cannot contradict the artifact's own `device` (a `--device cpu` smoke run
  on a CUDA host). If `count` is 2, the registered pairs are two independent
  arms and could train one per device for the same tokens - a *schedule*
  change (re-derives `full_pairs_per_session` from a re-run section 5.2
  benchmark), not an assumption. Pinned by 2 tests in
  `tests/test_training_distributed.py`.
- **The queue driver is quota-aware, and `--hours` now actually reaches the
  kernel.** `kaggle_queue.py` reads `kaggle quota --format json` before each
  push and refuses when remaining GPU hours are below the declared window
  (exit 3), because a session cut off by quota still bills every hour it used
  and produces nothing - the dominant failure mode of a ~52 GPU-hour plan
  against a 30 GPU-h/week line (nine sessions; the "~47" this sentence first
  carried predated the measured rate and the null queue's real cost - entry at
  the top of this file). An unreadable quota line never blocks (it can
  only spend less), and a live session is adopted rather than gated. Fixing
  this exposed the flag it depends on: `--hours` set `ACT_T4_HOURS` in the
  *driver's* process, which a script kernel never sees, so the kernel silently
  used its own default (masked only because that default is also 9). The
  window is now pinned into the staged kernel body beside the mode, so the
  gate and the session compare the same number. Pinned by 8 tests in
  `tests/test_kaggle_queue.py` (quota parsing incl. the TPU row and
  cannot-tell cases, the gate's boundary, a drive-level refusal that asserts
  no push happened, and a positive control that a coverable week still
  pushes).
- **The documented verdict command works, and the section 4 path is verified
  end to end.** `T4_RUNBOOK.md` section 3 told operators to run
  `kaggle_queue.py --queue verdict`, but `--queue` only accepted the three
  queues, so the command that produces the final answer failed with an
  argparse usage error - exactly once, after the whole quota was spent.
  `--queue verdict` (and `--queue preflight`) are now one-shot jobs: staged,
  pushed once, waited on, and reported, with no re-push loop and no quota gate.
  The verdict's registered code is mirrored as the driver's exit code (0 PASS,
  1 FAIL, 2 PASS with qualification, 3 no verdict), and to make that possible
  the decision now travels in `v0-5-1-verdict.json` as `exit_code` - the
  Kaggle-side runner exits 0 whatever the decision, so a reader could
  otherwise only re-derive PASS/FAIL from the verdict string.
  Verified by running the real thing: three full 12-seed queues produced by
  `run_pair_session` with the smoke profile, then `v0_verdict.py` over them -
  `queue_complete` true, 12 seeds complete, no strays, the pool-matched
  control and null blocks both evaluated, `measured_cost_percent` 52.2% (far
  outside the 5% criterion for a tiny config, hence the FAIL), artifact
  `exit_code` 1 matching the process exit code 1. The reader/writer coupling
  is now pinned by a test, because the existing verdict tests hand-build their
  summaries and never fed the script a summary the runner wrote.
- **The remote publish round trip is now pinned, not just asserted.** The
  adapter's docstring records that the upload layout was established by live
  probes because the *naive* upload silently produced an empty dataset and
  per-directory tarring flattened every seed to the archive root where they
  collided - but the staging that fixes it had **no test coverage at all**, and
  its failure mode is a corrupt dataset that still reports `published`. (That
  is now caught twice over: the driver's no-progress guard would also fire,
  because a later session would re-train seeds that were never really
  published.) `stage_push` is now a module-level function free of the API
  client, so it is testable without authenticating, and a test drives the
  whole joint - stage, upload, service-side extraction, download, merge - into
  a fresh root, asserting one archive per seed rooted at `seed-NNNN/`, both
  seeds restored, and every checksum from the pre-upload manifest verified
  through `restore_seed`. Confirmed to have teeth: dropping the arcname prefix
  reproduces the collision and fails the test.
- **The exit publish is retried, and can no longer raise on its own first
  step.** Two defects in the one push whose failure discards a session: the
  metadata write ran *before* the `try`, so a full disk (it writes into the
  handoff folder) made the "never raises" exit path raise - turning a finished
  session into a crashed one; and the push itself was attempted exactly once,
  so a transient API or network error was recorded as a failure. The push is
  now attempted up to `PUBLISH_ATTEMPTS` (3) with backoff, inside the same
  guard as the metadata write, and the recorded error names every attempt so a
  deterministic failure is distinguishable from a flaky one. Also corrected
  the docs' claim that a later session re-pushes a failed container's handoff:
  the handoff roots live in the container overlay, so a failed exit publish
  loses that session's pairs with no local copy left to push.
- **The driver refuses to repeat a session that advanced nothing.** A session
  whose clock never fit a pair reports *success* while writing no progress
  (rule 1 idles it), and the loop would push again - up to `--max-sessions`
  times, each repeat billing quota for the same nothing. The driver now
  compares cumulative completed seeds across sessions and stops with exit 4,
  reporting the quota the fruitless session consumed (differencing the quota
  line, the same measurement that caught two sessions running at once). This
  is the driver's own version of the burn it exists to prevent. Pinned by 3
  tests in `tests/test_kaggle_queue.py`, including a positive control that a
  session which *does* advance keeps the loop going.
- **The kernel runs the mounted code dataset, and nothing checked it.** The
  staged kernel body is embedded from the working tree, but `src/`, `scripts/`
  and `tests/` are copied from the attached `act-lab-code` dataset, so a lagging
  dataset silently executes old training code while the handoff manifest
  records a `code_revision` that did not run. Measured 2026-09-15: the dataset
  was last published at 13:14, a commit behind everything fixed afterwards - so
  the running session did not contain `c84b27b`'s guarded handoff publish, and
  2040ab1's `gpu_inventory` import would have crashed the next session against
  the older dataset. `kaggle_queue.ensure_code_dataset` runs before any push:
  it pins the dataset to `git rev-parse HEAD`, refuses a dirty or non-git tree
  (no reproducible revision), republishes when the dataset's `revision.txt`
  marker differs, and is checked before quota is spent rather than after.
  The staged dataset is the tracked Python side only - deliberately excluding
  `artifacts/`, because a session must start with no `artifacts/` directory so
  the launch command cannot read a benchmark artifact it did not measure, and
  excluding the web side (`public/`, `server/`, vite/ts configs) which the
  kernel never imports. Two follow-on defects found and fixed 2026-09-15: the
  tree is uploaded as one archive because Kaggle's uploader silently drops
  every subdirectory (a folder-shaped push published a dataset with no `src/`
  while still reporting success), and the revision marker is now written into
  the *staging* folder rather than the repo - a marker left at the repo root is
  an untracked file, so the publish refused the very next publish. The
  post-publish read-back is polled: a download issued right after a version
  push can serve the previous version, which is indistinguishable from a
  rejected push. Pinned by the code-dataset tests in
  `tests/test_kaggle_queue.py`, including that `artifacts/` never ships, that
  the archive carries the package tree, and that staging leaves the source tree
  untouched.
- **The kernel could not import its own code.** `kaggle_t4._bootstrap()`
  copies the mounted dataset into the writable working dir and returns an env
  dict with `PYTHONPATH=<work>/src` - but that dict only ever configured the
  *subprocesses* each mode launches. The pre-flight, the dtype probe and the
  verdict import `act` in this very process right after bootstrapping, so they
  died on `ModuleNotFoundError: No module named 'act'` (measured 2026-09-15,
  the dtype probe, 8 s in). The pre-flight and the verdict had it from the
  start; `2040ab1` added the same in-process import to the session mode, which
  would have killed every session pushed after it. `_bootstrap` now puts the
  copied `src/` on `sys.path` as well, and a test imports a module that only
  exists in the fake mount to prove the copied tree is importable in-process.
  The session that is running predates the session-mode import, which is why it
  is unaffected.
- **The session pipeline is proven end to end, not just in pieces.** The
  registered launch command was run at reduced fidelity on CPU
  (`--profile --seeds 15000-15001`) and its own `session-log.json` fed back
  through the driver: a fresh session logs `completed` per pair, a re-run
  against the published version logs `skipped_complete`, and both read as
  `(2, 12)` cumulative progress. That round trip is what the driver's
  "is this queue finished?" decision rests on - if a resumed session did not
  count, the driver would re-push sessions with nothing left to do - and it
  also exercises the new overlay dataset root, the atomic publish (a real
  `seed-*/manifests/manifest.json` and checksum-verified restore), and the
  resume lookup. Pinned by 1 integration test in `tests/test_kaggle_queue.py`
  (plus the 12 hand-built `queue_progress` cases).
- **Every kernel-side environment knob is now pinned into the staged body.**
  A script kernel receives only `code_file`, so anything set on the pushing
  side never arrives, and the kernel falls back to its own default. Three
  instances existed and all three only appeared to work because the default
  matched: the mode (fixed earlier), `--hours`, and - the one that would have
  been destructive - `ACT_T4_OWNER`/`ACT_T4_SLUG`, which `Queue.dataset`
  builds *inside the kernel*, so an override would have published the session
  to the default dataset while the driver believed otherwise. Staging now
  emits all of them from one helper (`pinned_environment`), and refuses if any
  of the constants is renamed in the body. Verified by executing the staged
  guard with the driver's environment set and the kernel's cleared: it
  resolves `someone-else/other-dataset` rather than the default. Pinned by a
  test that stages under an override and asserts the metadata agrees.
- `PYTHONPATH=src python -m pytest -q` -> all unit tests pass (272, including the
  26 report-figure data-assembly tests, 14 session-plan tests pinning the
  section 5.2 schedule/dtype tables, 10 handoff tests covering atomic
  publish, checksum rejection, and bit-identical resume through the dataset
  round trip, 9 runner tests covering deadline gating, cooperative
  interrupt + in-progress snapshot, restart-from-scratch, and
  skip-completed idempotence, 22 notebook-builder tests pinning determinism,
  nbformat validity, and the section 104 claims, 18 verdict tests pinning
  every branch of the section 5.1 decision rules (the 5% criterion, the
  [0.5, 2]x tolerance band, the 10-of-12 harm guard and its label
  precedence over a band miss, queue completeness
  gating `gate_4_authorized`, stray-seed exclusion, unmatched-budget and
  missing-arm aborts, and the verdict read through the real publish
  pipeline), 9 real-data session tests (window sampling, vocab guard,
  config validation, real-data training below the synthetic floor,
  bit-identical real-data resume, and the publish round trip), and fp16/CLI tests covering
  the fp16 session
  round trip, the version-4 compatibility path, and the pair-session launch
  command's artifact/dtype guards); session runs are deliberately excluded
  from the suite (they are integration checks, too slow for per-commit runs) and
  live in `scripts/v0_session_selfcheck.py`, which verifies matched budgets,
  FLOPs booking, bit-identical mid-run resume, and the version-4 checkpoint
  contract on a tiny config.
- `PYTHONPATH=src python scripts/build_notebooks.py` -> writes the ten
  notebooks (00-09) byte-identically; `--check` verifies the committed
  `.ipynb` files never drift from the builder output; every code cell of
  all ten notebooks was executed in order (cell-runner pattern, like the
  self-check): 00 reproduced the declared parameter counts, 01 proved
  train/resume/seed-reproducibility, 02 proved grow/rebuild/prune weight
  and hash round trips plus the registry append, 03 and 09 ran 4-seed
  gate studies to their recorded verdicts (demo-sized, not the powered
  384-seed studies), 04 extracted signals after a real backward pass and
  verified ablation restores parameters, 05 ran the three-arm matched-
  budget experiment with inspectable mutation histories, 06 exercised
  variants/prior ranking/frontier/Pareto selection, 07 froze and published
  the evolved genome through the checksummed manifest, and 08 compared
  fresh/rewound/teacher initializations with exact copied/skipped counts.
- `PYTHONPATH=src python -W error::UserWarning -c ...` ablation check ->
  the temporary-ablation helper now detaches losses before `float()`
  conversion (no tensor-scalar warning); covered by the notebook cell
  executions above.
- `PYTHONPATH=src python scripts/v0_session_selfcheck.py` -> re-passed after the
  declared sessions switched to checkpointed activations (resume stays
  bit-identical; recompute is deterministic because blocks carry no dropout),
  and after the fp16 path landed (step [3/4] verifies fp16 resume is
  bit-identical with scaler state restored, and that fp16 books the same
  FLOPs counters as fp32).
- `PYTHONPATH=src python scripts/v0_session_selfcheck.py --device cuda` ->
  **the declared pairing now has coverage.** Every fp16 check above ran on CPU
  (where `GradScaler("cpu")` is the native configuration) and the documented
  local-GPU pair-session run was fp32, so nothing had ever trained fp16 on
  CUDA - while the declared run does exactly that, and every session after the
  first resumes a CUDA fp16 checkpoint. The self-check now takes `--device`
  (a module constant, so no call site can compare a checkpoint from one device
  against a resume on another and pass while testing nothing) and the whole
  battery passes on CUDA: fp32 and fp16 resumes both bit-identical, fp16 FLOPs
  counters unchanged, checkpoint contract intact. The hardcoded
  `GradScaler("cpu")` in `run_v0_session` was also probed directly and is
  behaviourally identical to `GradScaler("cuda")` on CUDA tensors - infs
  detected, step skipped, scale backed off 65536 -> 32768 - so it is not a
  defect and was deliberately left alone rather than changing the numeric path
  mid-run.
- `PYTHONPATH=src python -m act.training.cli --mode pair-session --profile ...`
  -> the launch command runs the whole chain end-to-end on the local GPU:
  it consumed a real benchmark artifact (smoke profile, 26,987 tokens/s,
  fp32 decision), gated and completed all 12 declared seed pairs, published
  each to the dataset handoff, and wrote the session log; the stale-artifact
  and dtype-override guards were exercised by unit tests.
- `PYTHONPATH=src python scripts/v0_throughput_benchmark.py` -> the declared-model
  benchmark runs end-to-end on the local GPU (285,336,576 / 134,341,632 exact
  parameters, ~1,736 tokens/s with activation checkpointing -> dtype rule
  selects fp16 below the 2,500 floor) and writes `artifacts/v0-benchmark.json`;
  `--profile` smoke-checks on tiny configs or CPU. The T4 decision comes from
  the T4 run.
- `PYTHONPATH=src python scripts/local_proof_run.py` -> the local rehearsal of
  the full protocol loop: benchmarks the reduced model (12.55M/6.26M params,
  ~15.2k tokens/s on the local 1070, fp32 above the floor), then runs the
  production `run_pair_session` over the declared queue 9000-9011 (atomic
  pairs, skip-completed, deadline rules) into the local-proof dataset root and
  closes with `scripts/v0_verdict.py`. Completed 2026-09-13: 12/12 pairs,
  PASS_WITH_QUALIFICATION (cost 1.0044% <= 5%; declared band correctly not
  met by a reduced model; harm guard clear at 7/12). Re-runnable; `--dry-run`
  prints the plan without training.
- `PYTHONPATH=src python scripts/local_proof_run.py --real-data` -> the
  section 5.1 protocol machinery on the REAL TinyStories token stream (vocab
  8192 streams built by `train_real_lm.py`; `V0SessionConfig.data_path`/
  `val_data_path` route training, probe, and validation draws through random
  windows into the streams - shapes, generators, checkpointed state, and
  FLOPs booking identical to the synthetic path, so the identity and the
  bit-identical resume contract hold; unit-pinned in `tests/test_v0_real_data.py`).
  Completed 2026-09-14, 12/12 pairs: verdict
  PASS_COST_ONLY_HARM_GUARD_REJECTS - the cost identity held again (measured
  1.0077% vs 0.9977% predicted, <1% relative deviation), arm losses were real
  language losses (~3.4-3.6 nats vs the 9.01 synthetic floor), but the
  ADAPTIVE LOST 11 OF 12 PAIRS on real text at this scale: single-batch
  probe-loss argmin is anti-adaptive here,  the study-5 winner's-curse
  mechanism re-emerging on real language structure (adaptive and random arms
  chose the same operation sequences - only the chosen experts differed, so
  the probe's per-candidate ranking is what misleads). The machine-checked
  harm guard engaged exactly as pre-registered and the verdict label now
  takes the harm-guard precedence over a band miss (unit-pinned). This is a
  real-data negative result the toy studies could not show: selection value
  on real text needs richer probes or longer horizons, and Gate 4 stays
  blocked. Scratch roots `artifacts/real-proof/{sessions,datasets}`; verdict
  and session summary committed as evidence. (The follow-up diagnostic below
  revises the "richer probes" reading: neither richer probe batches nor a
  richer candidate set fixed it.)
- `PYTHONPATH=src python scripts/real_data_diagnostic.py` -> the selection
  diagnostic following the real-data harm-guard rejection: both toy-winning
  axes re-tested on the real TinyStories stream over the same paired seed
  block 9000-9007 (steps=600, production `run_pair_session`, 8 pairs per
  variant, ~5.2 min/pair on the local GPU). `var` (cap-4/4-batch; measured
  probe cost 3.99% of training FLOPs) 0/8 adaptive wins, mean delta vs own
  random `+0.2858`; `rich` (cap-8/1-batch; 1.98%) 1/8, mean delta
  `+0.1661`, mean adaptive vs baseline adaptive `-0.1094` (an absolute
  improvement over the baseline adaptive arm, still a relative loss). In
  5/8 seeds the 4-batch argmin chose bit-identically to the baseline
  1-batch argmin (final-loss delta exactly `0.0`), and every variant's
  random arm replicated its baseline counterpart - only selection differed.
  Conclusion: on real text the probe ranking is stably misaligned rather
  than noisy - variance (study 6) is excluded as the mechanism and richness
  (studies 8/9) does not transfer; next discriminator is an argmax control.
  Report: `artifacts/real-diag/diagnostic.json` (run log alongside);
  diagnostic only, not a pre-registered gate, 8 seeds unpowered.
- `PYTHONPATH=src python scripts/real_data_diagnostic.py --variants argmax`
  -> the pre-registered argmax control from the diagnostic's
  stably-misaligned finding, now code and run: `V0SessionConfig` gained
  `probe_selection` (`"argmin"` default | `"argmax"`), flipping only the
  decision rule in `_adaptive_event` - same candidates, same probe batches,
  same booked probe FLOPs (measured 1.01% vs baseline ~1.0%), and the
  mutation-history policy string (`adaptive_probe_argmin`/`..._argmax`)
  audits which rule produced each event. On the same seeds 9000-9007:
  argmax 0/8 adaptive wins, mean delta vs own random `+0.3048` (argmin
  baseline: `+0.2858`), mean adaptive vs baseline adaptive `+0.0293`;
  non-degenerate (0/3 identical picks to argmin on seed 9000) with the
  operation balance inverted (argmin chose 20 prune/4 grow, argmax
  11 prune/13 grow). Both tails of the probe ranking lose to random by
  nearly the same amount -> the probe signal is uninformative on real
  text, not inverted; per the discriminator contract the fix must come
  from probe horizon/tokens, not the decision rule. Full suite: 274
  passed (2 new argmax tests pin validation and the inverted-decision
  path). Report: `artifacts/real-diag/diagnostic.json` now carries all
  three variants; the empty `run-argmax.log` from a detached-launch
  failure was removed (the argmax run executed in the foreground).
- `PYTHONPATH=src python scripts/real_data_diagnostic.py --force` -> per-
  candidate probe ranks are now logged natively: `V0SessionConfig` gained a
  `probe_landscape` flag (bookkeeping only; off by default, mirroring the
  toy path's instrumentation; three new tests pin the sorted-ascending
  landscape, the argmax tail selection, and the off-by-default contract).
  The diagnostic enables it for every variant and aggregates a per-variant
  `probe_rank` block into `diagnostic.json` (event count, mean selected
  rank, rank histogram, near-top rate), so mechanism claims are directly
  measurable from the published seed summaries. Full suite: 277 passed.
  Re-running requires `--force` (the runner skips completed pairs; previous
  results are moved aside with a timestamp suffix). Existing artifacts
  pre-date the instrumentation and report `events_with_landscape: 0`.
- `PYTHONPATH=src python scripts/real_data_diagnostic.py --variants heldout`
  -> RUN 2026-09-14 (with `long`, revision `eaa1c45`, 8 pairs, ~1.2 h): the
  section-5.3 post-argmax fix candidate. `V0SessionConfig` gained
  `probe_data_source` ("train" default | "heldout"), pointing the probe
  windows at the held-out validation stream (same dedicated generator, same
  draw discipline, so the bit-identical resume contract holds; validation
  fail-fast requires `val_data_path` for heldout). Only the probe windows'
  source stream differs from the baseline adaptive arm - candidates, probe
  batch count, and booked probe FLOPs are baseline-matched (measured 1.01%).
  Four tests pin config validation, the real stream switch at the first
  event, cost parity, and bit-identical resume (281 at design time).
  Outcome: heldout 0/8 adaptive wins, mean delta vs own random `+0.2840`,
  random arms replicate - the pre-registered provenance hypothesis is
  FALSIFIED (outcome b) with the manipulation verified real (selected probe
  losses shifted 4.4195 -> 4.2848; 9/24 picks changed, more than `var`'s
  extra batches moved).
- `PYTHONPATH=src python scripts/real_data_diagnostic.py --variants long
  --force` -> DESIGNED, pre-registered, and RUN 2026-09-14 (same block,
  revision `eaa1c45`): the long-horizon probe as a one-step lookahead.
  `V0SessionConfig.probe_finetune_steps` (0 default | R > 0) fine-tunes
  each probed candidate for R AdamW steps on freshly drawn training batches
  (fresh optimizer, per-event throwaway generators, bit-identical resume
  pinned by test); search FLOPs book the adaptation on the full probed
  model (measured 3.00%). Outcome: 0/8 adaptive wins, mean delta vs own
  random `+0.3106` - the adaptation-blindness hypothesis is FALSIFIED
  (outcome b). Mechanism evidence: landscapes re-ranked after adaptation,
  the operation balance relaxed (5/24 grows) and seed 9000's probe
  top-ranked the grow at 2 of 3 events - the baseline's 4/20 prune bias is
  a snapshot-probe ranking artifact - yet per-seed deltas stayed in the
  harm band, and mean selected probe loss correlates POSITIVELY with final
  harm across seeds (long +0.51, heldout +0.89; 8 points, indicative).
  Both scorecards, the unified five-variant verdict, and the next-direction
  note (change how the ranking is used; own pre-registration required) are
  recorded in report section 5.3. Full suite: 285 passed.
- `PYTHONPATH=src python scripts/real_data_diagnostic.py --variants null`
  -> RAN 2026-09-14 (revision `9e5e448`; pre-registered in report section
  5.8, outcome in the same section): the do-nothing control arm
  (`V0SessionConfig.arms=("null",)`) - single-arm like gated, sharing the
  random arm's init identity and training stream, recording each event as
  a no-op, booking zero probe FLOPs. Result: **outcome (a) - churn harm is
  REAL**: null 8/8 wins vs the baseline random arms, mean delta `-0.1217`
  (per-seed -0.08 to -0.20, all negative). Built-in checks held: zero
  mutations / zero search FLOPs in 8/8 seeds, and the replication check is
  EXACT - max |null - gated| = 0.000000 on all eight seeds, the null arm
  reproducing the gated run bit-for-bit as the section-5.7 degeneracy
  reading predicted. Consequences: every section-5.3 variant's loss is
  partly churn harm; the theta sweep's control should be the null arm ("do
  accepted draws beat doing nothing?"); the honest real-text ladder for
  Gate 4 is null < random < adaptive. Verdict machine-checked by
  `scripts/null_verdict.py`. Full suite: 338 passed (6 null tests).: accept/reject gating of random draws - the
  probe's ABSOLUTE signal (does this single mutation hurt the current
  model?) as a per-event gate on the random arm's draw, abandoning the
  anti-predictive ordering entirely. Result: **8/8 wins, mean delta vs own
  random `-0.1217`** (adaptive arm `-0.3972` from the baseline adaptive
  arm; first negative real-text delta of any treatment) at 0.50% probe
  cost - but degenerate: accept rate 0/24, all 24 draws harmful by probe
  (mean draw-minus-reference `+0.1065`), all 24 decisions `reject_noop`
  (the load-fallback's `< 8` guard cannot fire at `expert_count = 8`),
  `mutation_count` 0 in 8/8 seeds, and draws bit-identical to the random
  arms' own operations (24/24 - the built-in control held). The win belongs
  to the NULL arm - zero matched-budget churn beats random churn on real
  text - not to gate discrimination: outcome (a)'s letter fired, its
  content ("absolute-harm gating carries real signal") was never exercised.
  First direct measurement of real-text churn harm, and a partial
  explanation of every earlier variant's loss (all spent the full mutation
  budget). Follow-ups need their own pre-registration: a do-nothing control
  arm (now pre-registered as report section 5.8 and implemented as
  `V0SessionConfig.arms=("null",)` - RAN, outcome (a), see below), and an
  intermediate-theta sweep (theta `0.05`/`0.1` would have
  accepted 11/24 and 21/24 of the observed draws). Single-arm sessions
  compared against the baseline block's stored random arms; per-event
  `gate_decision` records and both probe losses made the mechanism
  measurable as designed. Full suite: 332 passed.
- **Section 5.11 theta sweep (RAN 2026-09-14, revision `9bd9323`)**: the
  intermediate-theta sweep against the null arm on the fresh block
  9008-9017 (10 seeds; `scripts/theta_sweep.py`: the null control plus
  `gated` at theta {0.02, 0.05, 0.1}). Family verdict: **NO theta
  clears** (0/10 wins at every theta, p 1.0, bar 9/10 at Bonferroni
  alpha/3) - the pre-registered falsification branch fires and the
  probe-on-real-text program is CLOSED at this fidelity. Mean
  gated-minus-null delta is monotonically harmful in acceptance rate:
  +0.0027 at 6.7% accepted, +0.0236 at 43.3%, +0.0713 at 76.7% - a clean
  dose-response of churn harm. Mechanism verified from the published
  seed summaries: seeds accepting nothing are bit-identical to the null
  arm (8/10 at theta 0.02), and identical accept/reject sequences across
  thetas give bit-identical losses (9008/9015/9017) - the sweep measured
  exactly the cost of accepted churn. Secondary: the published random
  arm at 9008 (loss 4.1279 vs block-mates ~3.3) is itself a churn
  catastrophe of the section-5.8 kind. Verdict committed at
  `artifacts/real-diag/theta-sweep/verdict.json`. Next direction
  pre-registered as report section 5.12: the depth x candidate-policy
  2x2 (the 5.9-vs-5.10 contrast varies two registered fields at once).
  Full suite: 349 passed.
- **Section 5.12 pre-registration (designed, not yet run)**: the depth x
  candidate-policy 2x2 completes the factorial behind the 5.9/5.10
  contrast - depth 1 + spread (fresh block 72000-72383) and depth 4 +
  prefix (fresh block 73000-73383), same three arms and the same
  primary contract, executed as new profiles (`depth1-spread`,
  `depth4-prefix`) of `scripts/null_toy_study.py` with committed
  summaries and pinning tests like 5.9/5.10.
- **Section 5.12 2x2 (RAN 2026-09-14/15, revision `142ad44`)**: both
  cells completed 384/384 seeds, built-in checks held, verdicts
  machine-checked, summaries committed and pinned by test (suite at
  355). Result: **depth, not the candidate policy, separates churn
  suppression from value creation.** depth 1 + spread: primary 1 PASS
  (289/384), primary 2 FAIL (131/384, +0.0475) - replicates 5.9;
  depth 4 + prefix: primary 1 PASS (266/384), primary 2 PASS (225/384,
  -0.0371, p 4.4e-04) - replicates 5.10. The depth-4 edge does not
  require the spread policy and spread alone does nothing at depth 1;
  the 5.10 "richness axis" attribution is revised to depth-driven, and
  the candidate policy is exchangeable within the registered
  cap-4/2-batch budget.
- **Section 5.13 pre-registration (designed, not yet run)**: the
  routing-statistics heuristic arm - the section-5.11 contract's named
  non-probe direction: deterministic grow at the argmax-routing-load
  layer every event, no chooser draw, no probes, zero search FLOPs
  beyond the rebuild. Implemented as `V0SessionConfig(arms=("heuristic",))`
  (single-arm, shares the random arm's init identity), wired into the
  diagnostic as the `heuristic` variant, 7 new tests (suite at 362).
  Runs on the theta sweep's block 9008-9017 against its published null
  arms; primary: sign test vs null at Bonferroni alpha/2 (bar 9/10).
- **Section 5.13 outcome (RAN 2026-09-15, revision `0ef29de`)**: the
  heuristic LOSES to doing nothing on every seed - 0/10 wins, mean
  delta +0.1218 (per-seed +0.049 to +0.410, all positive), p 1.0.
  Built-in checks held: 30/30 grows at the measured argmax-load layer
  (L5 x10, L7 x6, L4 x5, L0 x4, L6 x3, L3 x2), rebuild-only search
  FLOPs. **The real-data topology program is CLOSED**: probe-ordered
  (5.3), probe-gated (5.4/5.11), and load-directed (5.13) churn all
  lose to doing nothing; the null arm is the only winning policy on
  real text at this fidelity. The project's one remaining open gate is
  Gate 4 (the section-5.5 compute-efficiency protocol, block
  15000-15011, unspent).
- **Section 5.15 pre-registered (2026-09-15, revision `1e4528d`)**: the
  **depth dose-response** - the two missing cells (depth 2 on fresh block
  74000-74383, depth 3 on 75000-75383, prefix policy) that locate where
  value creation onset begins on the depth axis the 5.12 2x2 identified.
  Same three-arm machinery, same primaries, Bonferroni alpha/2; four
  pre-registered branches (monotone onset between 1-2 / 3-4 / 2-3, or a
  non-monotone response suspending the depth-axis reading).
- **Section 5.15 outcome (RAN 2026-09-15, revision `bbad439`)**: branch
  (b) - onset between depth 3 and 4. Depth 2: adaptive vs null 161/384
  (+0.0208); depth 3: 188/384 (+0.0046); depth 4 was already 225/384
  (-0.0371). Monotone in wins and mean, sign flip just above depth 3,
  churn harm clears at every depth (p < 1e-9 everywhere). Committed
  summaries `null-control-depth2-384.json` / `null-control-depth3-384.json`
  with a monotone dose-response pinning test (suite at 387). Also:
  smoke-tested the full three-queue Gate 4 launch path (primary/control/
  null through the real CLI in profile mode + the four-clause verdict) -
  clean.
- **Section 5.14 pre-registered (2026-09-15, revision `4e26de9`)**: the
  **do-nothing clause for Gate 4**. The 5.1/5.5 verdict structure predates
  the null arm - its harm guard compares adaptive against RANDOM churn
  only, so a full-fidelity run could pass while adaptive still loses to
  doing nothing (the exact 5.9 trap). The amendment: a null-session queue
  at declared fidelity over the SAME block 15000-15011 (runbook step 2c,
  `--session-arms null`, own roots, ~90 s/seed *(corrected 2026-09-20: a null
  seed trains the declared 24.576M tokens, so it is ~1.44 h on the T4, not
  90 s - that figure was a dev-shape 400-step measurement)*), and two new verdict
  clauses - `PASS_COST_ONLY_NULL_REJECTS` (adaptive loses-or-ties >= 10/12
  vs null) and `PASS_COST_ONLY_NULL_MISSING` (the queue is mandatory), both
  exit-2 withholdings leaving the cost PASS standing. Clause order: harm
  guard > pool control > null > control-missing > null-missing > band.
  Implemented: `scripts/v0_verdict.py --null-root`, CLI `--session-arms`,
  10 new tests (suite at 372). BONUS FIX: the pair-session `--seeds` parser
  now accepts the runbook's documented `15000-15011` range form (it crashed
  on it before - a latent bug the new CLI tests exposed). The T4 runbook
  now has three queues (primary, control, null) and a 4-clause verdict
  command.
- **Section 5.16 pre-registered (2026-09-15, revision `e62a732`)**: **is the
  discovered topology a reusable artifact?** The plan's headline claim
  (section 102: extract the topology as a "reusable computational genome")
  has never been tested, because every Gate 3 comparison scores a topology
  *while it is trained*, so weights and structure are confounded. Deliverable
  10 ("Fresh-initialization results") was TODO. This is the cheapest decisive
  version: per seed one three-arm discovery block (the registered 5.10 config,
  depth 4 + spread, arm order keeping random/adaptive at indices 0/1 so the
  discovery runs are the registered ones, with the never-mutating `fixed` arm
  appended - it consumes no global RNG, pinned by test), then each arm's final
  genome transported: rebuilt with fresh weights
  (`initialize_retraining(mode="fresh")`) and retrained 48 steps on the same
  per-seed data schedule with mutation FROZEN. New machinery:
  `ArmResult.genome` (plan section 11 weight/topology separation, made usable -
  the discovered topology is now in the record and JSON-round-trips),
  `src/act/evolution/reuse.py` (`transport_genome`), and
  `scripts/reusability_study.py` (run/verdict/summary, 20-way sharding); block
  81000-81383, fresh. 13 new tests. Transport validity is the load-bearing
  built-in check: transporting the unmutated genome must reproduce that arm's
  own run exactly.
- **Section 5.16 outcome (RAN 2026-09-15, revision `e62a732`, block
  81000-81383)**: registered **branch (c) - both primaries fail**: the
  adaptive advantage at toy fidelity lives in the trained weights and/or
  initialization, NOT in the structure the controller chose. Transport
  validity held perfectly (384/384 exact reproductions, max |delta| 0.0).
  P1 transport-adaptive < transport-random: 212/384, mean -0.1016, p 2.32e-02
  - **exactly on** the sibling studies' bar (212) but below the bar this
  section declared (220 at the declared alpha 0.0025, see the 5.16.1
  amendment: the declaration was internally inconsistent and was resolved
  AGAINST the hypothesis). P2 transport-adaptive < transport-fixed: 211/384,
  mean -0.0866, p 2.94e-02 - fails at either bar. Two by-products that matter:
  (1) the discovery half replicates 5.10's depth-4 ordering OUT OF SAMPLE on a
  fresh block (adaptive beats random 259/384, p 3.5e-12; beats fixed 216/384);
  (2) discarding the weights costs the edge its *consistency* but not its
  *mean* (win rate 259/384 -> 212/384 while the mean barely moves), and a
  seed's discovery ranking predicts its transport ranking only 201/384 times
  (52%). Not capacity-confounded: adaptive topologies are usually smaller
  (lower active params in 264/384) and on the 90 capacity-matched seeds they
  beat random only 40/90 (bar 59). Committed summary
  `artifacts/study-summaries/reusability-depth4-384.json`
  (`study_f0334893263366c2`), pinned by `tests/test_reusability_summary.py`
  (which also pins the threshold resolution and the branch label).
  Consequence: the plan's section-55 initialization-mode study
  (evolved/rewound/fresh/teacher on one genome) is the next step, NOT the
  section-52 scale-up.
- **Section 5.17 pre-registered (2026-09-15, revision `e62a732`)** and
  **outcome (RAN 2026-09-15, block 82000-82383)**: plan RQ6 / section 55 - **is
  initialization or topology the important artifact?** Registered **branch (d):
  both the structure AND its initialization carry value.** Five transports per
  seed of three discovered genomes under both section-55 treatments
  (`rewound` = restore every component that existed at initialization to its
  initial weights **by MoE expert id**; grown experts keep the fresh draw;
  `fresh` = transfer nothing). P1 rewound < fresh `215/384` (mean -0.0806,
  p 1.08e-02, PASS); P2 rewound-adaptive < rewound-fixed `274/384`
  (mean -0.1123, p 1.46e-17, PASS). **This revises section 5.16's branch (c):**
  the topology has standalone value only as the *pair* (structure + the
  initialization its components actually had) - the same P2 comparison with a
  fresh draw fails again on this independent block (`198/384`, mean -0.0317,
  reproducing 5.16's `211/384`). Plan section 11's "final topology + original
  initialization ... without ambiguity" is empirically load-bearing, and the
  section-52 scale-up is triggered with the initialization policy as part of the
  declared artifact. Two pre-registered checks keep it honest: rewinding the
  *random* arm's topology does nothing (`189/384`, mean +0.0424), so the effect
  is specific to the chosen topology rather than a generic "restore good
  weights" procedure; and P1 concentrates where rewinding is total (pure-prune
  seeds `123/213`, p 1.4e-02, vs seeds with grown experts `92/171`, p 0.18),
  while P2 holds in both subgroups. Not capacity-confounded: the adaptive
  topology is *smaller* in 352/384 seeds. Unregistered observation (labelled as
  such in the report): the discovery run itself is worse than a clean retrain of
  its own topology from the original initialization (clean retrain wins
  `271/384`, mean A-B +0.0961). NEW MACHINERY: identity-aware rewinding -
  `initialize_retraining(..., renames=...)` and
  `reuse.expert_renames`/`initial_model`/`transport_genome(initialization=...)`.
  This fixes a latent bug: the pre-existing `rewound`/`evolved` copy matched
  parameters by name and shape, so rewinding a pruned topology would have handed
  a survivor the *pruned* expert's initialization (shape-compatible, silent); a
  test now asserts survivor identity on parameter values and fails if the
  mapping is dropped. A pre-run correction (report section 5.17.1) fixed
 5.17's own built-in check, which claimed B == C for pure-prune genomes - false,
  because a pruned architecture is a different architecture, so its fresh draw
  differs; B == C holds only where the topology is unchanged. Caught by the
  smoke test on a spent block before the registered block ran. Committed summary
  `artifacts/study-summaries/initialization-depth4-384.json`
  (`study_aef9c55774ab2c04`), pinned by `tests/test_initialization_summary.py`.
- **Section 5.18 pre-registered and outcome (RAN 2026-09-15, block 83000-83383,
  donor block 84000-84383, revision `137f6d0`)**: **is the genome portable, or
  irreducibly paired with its own initialization?** Registered **branch (d):
  BOTH** - and both predictions recorded before the run were wrong in the same,
  informative way. Three transports per seed: `O` = adaptive genome rewound to
  its own initial model; `D` = adaptive genome rewound to a **donor** initial
  model (seed+1000, a declared donor block); `F` = unmutated genome rewound to
  the donor's initial model, which copies every tensor and so **is** the donor
  model trained on this seed. `D` and `F` therefore share one initialization and
  differ only in topology. P1 pairing (difference-in-differences) `237/384`
  (mean -0.0830, p 2.54e-06, PASS); P2 portability (`D` < `F`) `234/384`
  (mean -0.0465, p 1.06e-05, PASS) - **the project's first support at any
  fidelity for plan section 102's "reusable computational genome" claim**.
  **The synthesis across 5.16-5.18.2: every initialization-MATCHED
  topology-vs-initial comparison wins** (own/own `293/384` here and `274/384` in
  5.17; donor/donor `234/384`) **while both MISMATCHED ones fail** (5.16 P2
  `211/384`, 5.17 `C_adaptive_vs_B_fixed` `198/384`). Sections 5.16's branch (c)
  and 5.17's failure rows were not measuring a weak topology but a control whose
  initialization was not matched to the treatment's: the seed-level
  quality difference between two draws has stdev `0.86`, five times the `~0.1`
  topology effect, which is what swamps the comparison. A pre-run correction
  (**5.18.1**, chosen from the spent smoke block for variance reasons, and whose
  smoke-block mean pointed *against* the hypothesis) replaced the contaminated
  raw contrast with the difference-in-differences; the raw contrast is still
  reported and still fails (`199/384`, mean -0.1452, p 0.254). NEW MACHINERY:
  `transport_genome(..., rewind_source=...)` for donor initializations, plus a
  test pinning that rewinding an unmutated genome to a donor yields the donor's
  state dict exactly. Committed summary
  `artifacts/study-summaries/portability-depth4-384.json`
  (`study_757e78b90de8a510`), pinned by `tests/test_portability_summary.py`
  (suite at 523).
- **Section 5.19 RAN (2026-09-15, blocks 85000-85383 depth 2 / 86000-86383 depth
  3, donors +2000).** Its committed summaries were missing while its numbers
  were already being cited; regenerated from the surviving shards as
  `portability-depth2-384.json` (`study_436e3cfb83e13fc0`) and
  `portability-depth3-384.json` (`study_ea20b7b1ace9534b`). Registered verdict:
  **branch (c)** - both primaries pass at both depths - so the matched-init
  topology value is NOT depth-limited and section 5.15's depth-4 onset is a
  property of probe-based selection, not the substrate (redirect to controller
  research; section-52 scale-up deprioritised). Recorded trap: the script's
  automatic `outcome` label uses section 5.18's branch letters, so the depth-2
  record says "branch (b)" while 5.19's registered reading is (c); pinned by
  `test_section_5_19_auto_label_letters_index_section_5_18_not_5_19`.
- **Section 5.20 RAN (2026-09-15, blocks 89000-89383 depth 2 / 91000-91383
  depth 3, `scripts/specificity_study.py`).** The selection-specificity control
  the earlier studies lacked: three rewound transports per seed on the seed's own
  initial model. **P1 `A_own` < `R_own`: 243/384 (depth 2) and 248/384 (depth 3),
  both PASS**; P2 `A_own` < `F_own` replicates at 264/384 and 265/384. Registered
  verdict **branch (a) - the shallow-depth topology value is selection-specific**
  (my declared prediction that P1 would fail at depth 2 was wrong). The
  random-churn control is neutral at depth 3 (mean -0.0012) and only nominally
  non-zero at depth 2 (mean -0.0213, 29% of the effect, failing the Bonferroni
  bar); capacity does not explain it (random-churn topologies are never the
  initial size in 0/384 seeds, yet gain nothing). Summaries
  `specificity-depth2-384.json` (`study_f10d1b1cd3812bc1`) and
  `specificity-depth3-384.json` (`study_b6455e54f8801fd8`), pinned by
  `tests/test_specificity_summary.py`. **Amendment 5.20.1:** a one-seed-in-384
  chance coincidence of the two arms' genomes tripped an all-384-differ guard and
  withheld the depth-2 block; the guard now requires coincidences to be *exact
  ties* and *rare*, which changes no statistic - the declared reading passes with
  the tie counted as a non-win. Architecture hashes are carried per seed so the
  guard is recomputable from the artifact.
- **Sections 5.21 (observation) and 5.22 (RAN 2026-09-15, blocks 92000-92383
  depth 2 / 93000-93383 depth 3).** Reanalysing three committed blocks showed
  each mutating arm's fused run losing to a frozen retrain of its own final
  genome at every depth, and - the part section 5.17.2 could not see, since it
  checked only the adaptive arm - losing for the **random** arm too (292/384,
  293/384, 258/384). That makes the ~0.1 nat penalty a property of *mutating a
  live run*, about twice the size of the topology signal these studies measure,
  which is a sufficient explanation for the weak-to-negative fused comparison at
  shallow depth. Recorded as 5.21 explicitly **unregistered** (a reanalysis of
  blocks run for other reasons), then 5.22 pre-registered, committed, and run:
  **branch (a), both primaries passing at both depths** (296/384 and 291/384
  adaptive; 286/384 and 300/384 random). **Adopted consequence: two-stage
  delivery (discover -> freeze the genome -> retrain from the original
  initialization) is the project's pipeline.** *SUPERSEDED in part by sections
  5.23-5.28 - see below: that consequence was earned on the then-current mutation
  path, which 5.24/5.26/5.27 decomposed into a router discard (a defect) and the
  fresh new expert (invertible by cloning), and 5.28 showed Freeze-and-retrain is
  a REGRESSION in the drawn regime this was adopted from. The pipeline is now
  clone-and-preserve, keeping the fused run.* Section 5.15's depth
  dose-response is re-read as being about the *fused* process. Two fresh blocks
  also replicated 5.20's primaries (256/255 and 257/272). Unseparated mechanism:
  a frozen retrain has a fresh optimizer and trains a pruned topology, so
  neither optimizer history nor per-step FLOPs are matched - declared in the
  pre-registration as uncontrolled. Summaries
  `artifacts/study-summaries/frozen-delivery-depth{2,3}-384.json`
  (`study_bc1a4fdfe99fde63`, `study_f23d93bf49e18a57`), pinned by
  `tests/test_specificity_summary.py`; the observation is pinned to its source
  summaries by `tests/test_fused_vs_frozen_observation.py`, which also fails if
  the report drops the "unregistered" label.
- **Section 5.23 RAN (2026-09-15, blocks 94000-94383 depth 2 / 95000-95383 depth
  3, `scripts/optimizer_reset_study.py`).** The confound section 5.22 left open,
  decomposed with a new opt-in knob `ExperimentConfig.optimizer_reset_steps`
  (empty by default, so no published number moves): a twin run whose genome,
  weights, seed and data schedule are identical to the fixed arm's and which
  differs **only** by discarding Adam's moments at steps 12/24/36. Registered
  verdict **branch (c)**: resets cost nothing - P1 fails at both depths *in the
  exonerating direction* (the reset twin was slightly BETTER, `F` winning only
  157/384 and 105/384, mean +0.005 / +0.016) - while the rebuild alone accounts
  for the whole penalty (`U` < `R`: 299/384 and 311/384, mean -0.1069 / -0.1276,
  against section 5.22's total -0.0996 / -0.1219 on independent blocks). **So
  section 5.22's delivery change stands unqualified and the cheap alternative is
  closed off: preserving optimizer state across mutations would buy nothing.**
  Prediction declared as correct here after four consecutive understatements, and
  the pre-registration disclosed that it was informed by a spent-block smoke that
  pointed against the mechanism the design was built to detect. What remains
  unseparated: the reset twin keeps the *initial* architecture (so `U` vs `R` is
  a between-topology contrast), and a grow's fresh expert initialization is not
  isolated at all - it is now the leading candidate. Summaries
  `artifacts/study-summaries/optimizer-reset-depth{2,3}-384.json`
  (`study_e0f1f2629785dc17`, `study_09573ffbfb828091`), pinned by
  `tests/test_optimizer_reset_summary.py` (which also pins that the rebuild's
  cost is 0.8-1.2x the 5.22 penalty, so the decomposition cannot silently stop
  closing) and `tests/test_optimizer_reset_control.py`.
- **Section 5.24 RAN (2026-09-15, blocks 96000-96383 depth 2 / 97000-97383 depth
  3, `scripts/router_preservation_study.py`) - and it found a REAL DEFECT.** A
  layer's router weight is `hidden x experts`, so its shape changes with the
  expert count and `rebuild_model_from_genome` (which carries only same-shape
  keys) **re-draws the mutated layer's entire learned routing head at random at
  every mutation** - while the experts' own weights survive by id and unmutated
  layers' routers are copied. Verified directly, not inferred. Every fused run to
  date therefore discarded a layer's learned routing function three times.
  `rebuild_model_from_genome(preserve_router=True)` copies surviving rows by id
  and leaves only a grown expert's row fresh. Registered verdict **branch (b)
  - partial**: preserving the routing head removes **55.1% / 54.3%** of the
  penalty (`Rp` < `R`: 264/384 and 270/384, mean -0.0542 / -0.0694) and a
  residual `-0.0441` / `-0.0583` survives (`Rp_own` < `Rp`: 254/384, 258/384).
  **So the dominant term of the effect section 5.22 changed the delivery path
  for was an avoidable implementation defect, not an intrinsic cost of mutating
  a live run.** The study is perfectly paired and that was checked, not assumed:
  both settings make the same architectural choices, so final genomes are
  identical in 384/384 seeds and the frozen references are bit-identical in
  384/384 (`R_own == Rp_own` exactly). **The `preserve_router=False` default is
  deliberately retained** even though the discard is unintended, because every
  committed number in sections 5.16-5.23 was measured under it and flipping it
  would silently invalidate the record; new work should set
  `preserve_router_on_rebuild=True`. Prediction (branch (b)) was right, second in
  a row, with the report stating plainly that the two spent-block smokes
  disagreed with each other. Remaining: the residual `~0.05` is either a pruned
  expert's lost capacity or a grown expert's fresh initialization, which this
  design cannot separate. Summaries
  `artifacts/study-summaries/router-preservation-depth{2,3}-384.json`
  (`study_18c56d8bf539e2f3`, `study_65103c230c610378`), pinned by
  `tests/test_router_preservation_summary.py` (which also pins the defect itself,
  teeth-checked by flipping the default).
- **Section 5.25 RAN (2026-09-15, blocks 98000-98383 prune / 99000-99383 grow,
  `scripts/residual_attribution_study.py`).** Attributes section 5.24's residual
  by *forcing* the operation rather than splitting a mix: new opt-in
  `ExperimentConfig.forced_operation` makes every non-adaptive event the named
  operation, and a forced operation the genome cannot perform is left
  unperformed rather than swapped for the fallback grow (so a profile that stops
  mutating fails its budget check instead of measuring the wrong thing).
  Registered verdict **branch (a) - the residual is BOTH operations**: the prune
  block 241/384 (mean -0.0297, p 3.25e-07) and the grow block 241/384 (mean
  -0.0476, p 3.25e-07), bar 212 at `alpha/2`. **The grow costs ~1.6x the prune**
  (0.0159 vs 0.0099 per forced event) - the opposite of the declared prediction,
  which expected the grow to be free because it only adds capacity. Fifth
  understated prediction in the sequence, and the first where the miss was in the
  *comparison* rather than in whether an effect exists. An additive check roughly
  holds: per-event costs predict `0.0386` for a 50/50 mix against section 5.24's
  measured `0.0441`, on different blocks and substrates. **Consequence:** the
  remaining lever is the *initialization of a new expert*, exactly plan section
  51's teacher treatment - now the best-motivated untested experiment in the
  programme, and still blocked on Qwen-derived artifacts this repo does not
  bundle. Summaries
  `artifacts/study-summaries/residual-{prune,grow}-384.json`
  (`study_68591d55f3a56db7`, `study_c7ec09294da043fc`), pinned by
  `tests/test_residual_attribution_summary.py` and
  `tests/test_forced_operation_control.py`.
- **Section 5.26 RAN (2026-09-15, blocks 103000-103383 depth 2 / 104000-104383
  depth 3, `scripts/new_expert_init_study.py`).** Answers section 5.25's closing
  question - is the residual penalty the *freshly drawn* new expert? - with
  something the model already contains instead of the blocked Qwen teacher: new
  opt-in `ExperimentConfig.new_expert_initialization="clone"` gives a grown expert
  the layer's **busiest surviving expert**, body **and router row**, so it starts
  as a working twin rather than as noise. Cloning is defined as copying a router
  row, so the config *refuses* `clone` with `preserve_router_on_rebuild=False`
  rather than silently cloning nothing. Registered verdict **branch (a)**:
  **P1 cloning helps** 315/384 (mean -0.1005, p 5.9e-39) at depth 2 and 307/384
  (-0.1036, p 6.4e-34) at depth 3; **P2 residual penalty FAILED IN THE REVERSED
  DIRECTION** - the penalty goes from **-0.0406/-0.0292** (fresh draw) to
  **+0.0600/+0.0745** (cloned), so cloning does not merely remove the penalty, it
  **inverts it**. The full ordering is the same at both depths,
  **`C < F < C_own < R`**: the clone-grow fused run (14.2414 / 13.3819) beats the
  unmutated fixed arm (14.2887 / 13.4362; 296/384, 291/384), the frozen retrain
  of its own final genome (14.3014 / 13.4564; 258/384, 253/384) and the historical
  random-grow run (14.3419 / 13.4855). The design was perfectly paired by
  construction - the forced grow picks its layer from the seed's own RNG and
  cloning consumes no randomness - so **the two settings chose the same final
  genome in 384/384 seeds at both depths and `C_own == R_own` bit-identically**,
  both asserted as built-in checks. **Consequence: section 5.22's two-stage
  `discover -> freeze -> retrain` pipeline is a REGRESSION on this substrate**,
  not a safeguard; the recommendation for new work is to *keep the fused run*
  with the router preserved and new experts cloned, which also makes plan section
  102's genome extraction unnecessary for performance and gives plan section 51's
  teacher treatment a *bound* (it can only pay off where a teacher beats the
  model's own busiest expert). One open mechanism, declared unresolved: cloning
  copies a trained body *and* a router row, and this design does not separate
  them. Sixth understated prediction (declared P2 pass, branch (b)). Summaries
  `artifacts/study-summaries/new-expert-init-{depth2,depth3}-384.json`
  (`study_99a12c16f910e9c0`, `study_ee833b6b284195f9`), pinned by
  `tests/test_new_expert_init.py` and `tests/test_new_expert_init_summary.py`.
  **Amendment 5.26.1:** the pre-registered branch list was written in the
  opposite letter order to section 5.24's convention and to the study's own
  script; the letters are reconciled to the script, which changes no statistic,
  no threshold and no reading (P2 failed in the exonerating direction, so the
  penalised reading was never available).
- **Section 5.27 RAN (2026-09-15, blocks 105000-105383 depth 2 / 106000-106383
  depth 3, `scripts/clone_mechanism_study.py`).** Resolves the mechanism section
  5.26 declared unresolved: cloning does two things at once (a trained *body*, and
  a copied *router row*), and a third setting separates them. New opt-in
  `new_expert_initialization="clone_body"` is identical to `"clone"` except the
  grown expert keeps a freshly drawn row, so `B - R` is the body's contribution and
  `C - B` the routing affinity's, on one genome. Registered verdict **branch (a),
  both contribute**: **P1 the body helps** 295/384 (mean -0.0744, p 3.4e-27) at
  depth 2 and 300/384 (-0.0844, p 7.0e-30) at depth 3; **P2 routing affinity adds**
  242/384 (-0.0264, p 1.9e-07) and 238/384 (-0.0220, p 1.5e-06). The split is
  **body 73.8%/79.4%, row 26.2%/20.6%** of the total clone gain, and the two terms
  partition it exactly. The ordering is four-deep and the same at both depths -
  **`C < B < F < own < R`** - so the body clone *alone* already beats the unmutated
  arm and the frozen retrain, and the row is a real, cheap second term. Three-way
  pairing asserted, not assumed: same final genome, bit-identical frozen reference
  and an identical fixed arm in **384/384 seeds**. **Consequence:** section 5.26's
  "twin" framing was right but imprecise - the body is the larger half, the row is
  what makes it a twin. It also sharpens plan section 51: a teacher must beat the
  model's own busiest expert **as a body source** for the larger term. Prediction
  was correct (second in a row) and came from the disclosed 4-seed smoke rather
  than from a prior. Declared **not** settled: no `clone_random` arm (any survivor
  vs the busiest), and the layer ends with a tied twin pair whose cost at larger
  `active_experts` is untested. Summaries
  `artifacts/study-summaries/clone-mechanism-{depth2,depth3}-384.json`
  (`study_3bcc9e86b575b7a6`, `study_872d33bdd99496a8`), pinned by
  `tests/test_clone_mechanism_summary.py` and `tests/test_new_expert_init.py`.
  Note: the toy studies stay on **CPU deliberately** - measured 2.6x faster than
  this 1070 at this shape (5.26 vs 2.04 s/seed), and every committed number in
  sections 5.16-5.27 was measured on CPU, so a device switch would stop the record
  reproducing. The real-data legs do use the GPU.
- **Section 5.28 RAN (2026-09-15, blocks 107000-107383 depth 2 / 108000-108383
  depth 3, `scripts/unforced_clone_study.py`).** Closes the gap that mattered most:
  sections 5.24/5.26/5.27 all measured on **forced** operations, but section 5.22's
  freeze-and-retrain pipeline was adopted from the **mixed, unforced** regime. Here
  `forced_operation` is left None, so the mix is drawn (622 grows / 530 prunes at
  depth 2, 548 / 604 at depth 3 - checked, so this cannot be a forced block in
  disguise). Registered verdict **branch (a)**: **P1 clone helps** 276/384 (mean
  -0.0720, p 2.3e-18) and 252/384 (-0.0651, p 4.6e-10); **P2 no residual penalty**
  FAILS in the exonerating direction again (**+0.0243** / **+0.0113**), against a
  historical penalty of -0.0477 / -0.0538. **The claim is sharper than the forced
  blocks could make it: the historical path leaves the run worse than NEVER
  MUTATING** (`F` < `R`: 251/384, -0.0444, p 8.9e-10; 236/384, -0.0403,
  p 4.1e-06), while **cloning makes mutating pay** (`C` < `F`: 238/384, -0.0275,
  p 1.5e-06; 237/384, -0.0248, p 2.5e-06). The frozen retrain ties the unmutated
  arm (186/384, p 0.75; 207/384, p 0.069), as a mixed draw should. Means
  **`C` < `own` ≈ `F` < `R`** at both depths. **Pairing was recorded, not assumed**
  - a drawn prune can fall back to a load-based grow, which reads weights cloning
  changes - and it held anyway: identical final genome **384/384**, identical
  operation mix **384/384**, `C_own == R_own` bit-identically, even though 69/72
  seeds had a layer at the 2-expert floor. Third correct prediction in a row.
  **Consequence: section 5.22's pipeline is contradicted in the regime it was
  adopted from**, and this is now a requirement rather than a preference - the
  recommended pipeline is **clone-and-preserve, keeping the fused run**
  (beats retrain 214/384 and 206/384, beats not mutating 238/384 and 237/384).
  Not tested and named as the next axis: the probe-based **adaptive** selection
  path; arm-genericity of the process penalty is cited from section 5.22, not
  re-measured. Summaries
  `artifacts/study-summaries/unforced-clone-{depth2,depth3}-384.json`
  (`study_5ece35fb3fbc8724`, `study_f8bc02ca0f048a79`), pinned by
  `tests/test_unforced_clone_summary.py`.
- **Section 5.29 RAN (2026-09-15, blocks 109000-109383 depth 2 / 110000-110383
  depth 3, `scripts/adaptive_clone_study.py`).** Tests the pipeline recommendation
  on the **probe-guided adaptive** arm - the discovery path the project actually
  uses, not the `random` arm section 5.28 used. Registered verdict **branch (c)**
  (P1 134/384 and 131/384 FAIL; P2 221/384 and 232/384 PASS), and the prediction
  (branch (a)) was WRONG - the fourth miss in a row. Pairing was not assumed: the
  probe scores candidates with the **default** rebuild while the *selected* one is
  applied with the arm's flags, so the runs can diverge. A pre-run 16-seed
  measurement found 2/16 divergent; the blocks came in at **45/384 and 40/384**.
  **But both registered verdicts are artifacts, not findings.** The adaptive arm
  grows in only **18%** of events, so **205/384 and 207/384 seeds contain NO grow**
  and cloning is then a bit-identical no-op (`C - R == 0` exactly, 0 wins). A
  full-block sign test scores those as non-wins for P1 and hands P2 the full
  historical penalty. Stratified post-hoc on whether a grow occurred, both readings
  reverse: **`C < R` 134/179 (75%) p 9.1e-12 and 131/177 (74%) p 6.0e-11**, with
  the penalty gone in the grow stratum (+0.0126 / +0.0016) and intact in the no-grow
  stratum (-0.0416 / -0.0633). This is the project's first statistic whose
  *validity* was destroyed by a structural no-op rather than by noise. Summaries
  `artifacts/study-summaries/adaptive-clone-{depth2,depth3}-384.json`
  (`study_59e969bbd7767f2d`, `study_c28ad3ce4784f3b9`). **Amendment 5.29.1** records
  that the stratified reading is post-hoc and not outcome-dependent, and
  **supersedes branch (c) for interpretation** - its numbers stand, its reading does
  not.
- **Section 5.30 RAN (2026-09-15, blocks 111000-111383 depth 2 / 112000-112383
  depth 3, `scripts/adaptive_clone_stratified_study.py`).** The pre-registered
  stratified re-test on fresh blocks, with the stratum declared in advance and
  **stratified on the HISTORICAL run's drawn mix** so the setting under test cannot
  decide who enters it. Registered verdict **branch (a), replicated at both
  depths**: grow stratum 195 and 179 seeds, **`C < R` 142/195 (73%) mean -0.0542
  p 7.1e-11 and 130/179 (73%) mean -0.0466 p 5.9e-10** (bars 114 and 105), penalty
  gone there (P2 fails: -0.0064 / +0.0144). **The artifact replicated too:** the
  full-block statistic fails on the same runs (**142/384, 130/384**) while the
  stratified one passes at p ~1e-11 - same evidence, opposite verdicts, and the
  only difference is whether the ~200 seeds that could not respond are counted.
  Positive control held: in the no-grow stratum `C == R` bit-identically and
  `C_own - C == R_own - R` exactly. **Synthesis:** the discovery path prunes 82% of
  events, so cloning repairs the grow term (-0.054/-0.047 where it binds) and
  leaves the prune term untouched, where the historical penalty stands at
  -0.055/-0.074; means are `own < C < R < F` at both depths, so the frozen retrain
  still wins overall on the prune term alone. **Section 5.28's "keep the fused run"
  holds for the `random` arm (half grows) and does NOT hold as stated for
  discovery** - cloning is necessary but not sufficient, and the residual deficit
  on prune-dominated discovery runs is a separate open problem. Summaries
  `artifacts/study-summaries/adaptive-clone-stratified-{depth2,depth3}-384.json`
  (`study_54cff7dc8d8b6f11`, `study_da528e81c3204901`), pinned by
  `tests/test_adaptive_clone_stratified_summary.py`. **Methodological conclusion
  (5.30.1):** record how often a treatment can bind before choosing the statistic,
  stratify on a pre-treatment variable, and include a positive control that the
  treatment is a no-op where it should be.
- **Section 5.31 RAN (2026-09-15, blocks 113000-113383 depth 2 / 114000-114383
  depth 3, `scripts/prune_target_study.py`).** Asks whether *which* expert a prune
  removes is a lever. New opt-in `ExperimentConfig.prune_expert_policy`
  (`"random"` default / `"busiest"` / `"quietest"`, ranked by the model's own
  recorded routing load). Registered verdict **branch (d)** - the sign test failed
  both primaries while the means ordered `Q < R < B` at both depths at 53-54% win
  rates. Summaries `artifacts/study-summaries/prune-target-depth{2,3}-384.json`
  (`study_41bbf67d9ae9bce2`, `study_131929958f11abc2`), pinned by
  `tests/test_prune_target_policy.py`, which also re-runs a committed section 5.25
  block from its own stored config and requires **bit-identical** losses - the
  guarantee that adding a defaulted knob did not move the published record.
  **Reproducibility finding recorded there and in the report:** reproduction is
  bit-exact at 1 and 4 torch threads but **not at 2** (last digits move), so
  "reproduces" is only defined at a fixed thread count and all committed numbers
  are `OMP_NUM_THREADS=1`. **5.31.1:** the binding constraint was the *statistic* -
  at a 53% win rate the paired sign test needs ~1396 seeds for what a mean test
  resolves with ~380, and this block had 384.
- **Section 5.32 RAN (2026-09-15, blocks 115000-116535 depth 2 / 117000-118535
  depth 3, `scripts/prune_target_powered_study.py`).** Section 5.31's question
  re-registered with the statistic declared in advance (one-sided paired sign-flip
  permutation test on the mean, 20k permutations, `alpha/4 = 0.0125`) on a ~4x
  block sized from 5.31's observed effect and spread. Registered verdict **branch
  (a), both primaries pass at both depths**: `fused(Q) < fused(R)` **-0.01133
  p 2.5e-04** and **-0.01250 p 1.0e-04**; `fused(R) < fused(B)` **-0.01678
  p 5.0e-05** and **-0.02435 p 5.0e-05**. **Both statistics agree everywhere in
  both blocks** (sign tests 813/1536, 823/1536, 845/1536, 846/1536 against a bar of
  813), which is 5.31.1's power claim measured rather than argued - the effect was
  always there, and 5.31's branch (d) was its statistic. Loss means order `Q < R <
  B` on the fused column and `F` sits between at depth 2 / above at depth 3.
  **Mechanism, and it is not the penalty framing's:** the lever acts on the fused
  trajectory, not the retrained topology - `own(Q) - own(R)` is **+0.0034 / +0.0008**
  (slightly *worse*) against fused -0.0113 / -0.0125, so the penalty secondary
  **fails** (p 1.00 both depths). Recovered share of the historical prune penalty
  is **27.6% / 41.6%** by targeting the quietest, and targeting the busiest *adds*
  **40.8% / 81.0%**. **Consequence: the lever is conditional on the delivery
  pipeline** - under 5.28's keep-the-fused-run recommendation it is free gain;
  under freeze-and-retrain the delivered artifact is unchanged. Against `F`,
  `fused(Q)` beats never mutating at depth 3 (`-0.01603`, p 5.0e-05) but not at
  depth 2 (`+0.00609`), so 5.30's depth-2 deficit survives. Summaries
  `artifacts/study-summaries/prune-target-powered-depth{2,3}-1536.json`
  (`study_ab45f014e8cb5e50`, `study_55d7527c71843ced`), pinned by
  `tests/test_prune_target_powered_summary.py` (which also pins the statistic's
  own calibration on symmetric heavy-tailed noise, and re-derives the stored
  comparisons from the stored per-seed rows). **5.32
  correction:** 5.31's prose read the sign-flipped `penalty = own - fused` column
  as agreeing with the fused column; it cannot, and 5.32's null own-column
  measurement is the corrected reading.
- **Section 5.33 RAN (2026-09-15, blocks 119000-120535 depth 2 / 121000-122535
  depth 3, `scripts/probe_selection_study.py`).** Two findings prompted it, both
  from section 5.32: its lever was worth 27.6-41.6% of the prune penalty on the
  **non-probe** paths, and its consequence ("the discovery controller should prefer
  the quietest expert") is **wrong**, because `prune_expert_policy` is inert on the
  probe-guided path - `_mutate_by_probe` enumerates every expert through
  `feasible_expert_mutations` and returns before the policy branch, measured as
  bit-identical losses across all three policies under `adaptive_policy="probe_loss"`
  and pinned by `test_the_knob_is_inert_on_the_probe_guided_discovery_path`. So the
  discovery controller already owns the lever; the open question is whether its own
  choice beats the cheap rule. New opt-in `probe_selection="quietest"` (both probe
  paths, so they cannot drift) isolates the TARGET: it fires only when the probe's
  own pick is a prune, and then takes the least-loaded expert in the probed pool.
  Registered verdict **branch (a) at both depths**: `fused(A)` vs `fused(M)`
  **-0.04118 p 5e-05** and **-0.04848 p 5e-05**; `fused(A)` vs `fused(Q)`
  **-0.03120 p 5e-05** and **-0.03443 p 5e-05**; both clear the sign test too. The
  ordering is three-deep and that is the interpretable part: against the same pool at
  the same probe cost the probe beats random by 0.0412/0.0485, the load rule beats
  random by 0.00998/0.0140, so the load rule recovers only **24.2%/29.0%** of the
  probe's advantage - it is not empty, and it is not enough. **Mechanism differs from
  5.32:** here `own(A) - own(Q)` is -0.03571/-0.05414, comparable to the fused
  contrast, so the probe buys a better run AND a better artifact, where 5.32's lever
  moved only the fused column. Against `F`, mutation pays in this configuration
  (`A` -0.04537/-0.05685, `Q` -0.01418/-0.02242). Treatment bound in 6110/6144 and
  6091/6144 seed-selection pairs; four distinct final genomes in 1253/1536 and
  1300/1536. Summaries `artifacts/study-summaries/probe-selection-depth{2,3}-1536.json`
  (`study_dbd63572517fd575`, `study_a24a62de8ce82d39`), pinned by
  `tests/test_probe_selection_summary.py` and
  `tests/test_probe_selection_quietest.py` (the rule's isolation contract: the
  selection must come from the probed pool, and must leave the operation alone when
  the probe's own pick was not a prune). Prediction correct, the fifth of the last
  six; the magnitude was the surprise - I expected the load rule to be comparable.
- **Report-artifact link guard (2026-09-15).** Section 5.19's missing summaries
  were a defect *class*, not an incident: a result can be cited widely while its
  artifact does not exist. `tests/test_report_artifact_links.py` now fails if any
  `study_...` id cited in `RESEARCH_REPORT.md` has no committed summary, or if
  any committed summary is referenced by neither the report nor this file
  (teeth-checked both ways). Full audit at the time of writing: **0 cited but
  missing, 0 orphaned** - section 5.19 was the only occurrence.
- `PYTHONPATH=src python scripts/train_real_lm.py` (and inference via
  `scripts/real_lm_infer.py`) -> the real-data leg the plan's data sections
  deferred: TinyStories (real English corpus) downloaded, a BPE tokenizer
  (vocab 8192) trained on the corpus, 40,043,724 real tokens encoded, and the
  project's own `V0MoECausalLM` (20,863,872 total / 10,247,040 active
  parameters) trained on GPU for 6,000 steps / 49.2M tokens (~45 min on the
  local 1070 at ~18k tokens/s). Final held-out validation loss 1.713
  (perplexity 5.55) against the uniform floor 9.011 (ppl 8192) - the model
  really learned the language. Inference produces fluent on-genre story text
  (greedy decoding deterministic; out-of-domain text scores ppl ~4000, i.e.
  the model  knows its domain). Report and samples in
  `artifacts/real-lm/report.json`; corpus/tokens/checkpoints are local
  scratch (gitignored).
- `PYTHONPATH=src python scripts/real_lm_chat.py` -> interactive streaming
  chat over the trained checkpoint: user lines are appended to the rolling
  story context (the model reacts to what you typed, verified: a "brave
  knight" line produced a lion-roar continuation), tokens print as they are
  generated, and session commands (/new with optional seed, /temp, /topk,
  /len, /show, /save, /quit) manage the loop; `--once` reads one seed line
  from stdin for piped use. Reuses `real_lm_infer.load_model` so chat and
  one-shot inference always agree on config, checkpoint, and decode.
- `PYTHONPATH=src python scripts/generate_figures.py [--set decoupled]` -> writes
  the four factor-map figures (fig1 win-rate forest, fig2 quality-vs-cost, fig3
  delta distributions, fig4 probe-rank heatmap), byte-identical across re-runs.
  `--set decoupled` plots the corrected section 9.11 record into
  `artifacts/study-summaries/decoupled/figures/` from summaries emitted by
  `factor_map_recheck.py --summaries` (each re-verified by `check_summary` at
  plot time); that set carries all nine rows and plots its own figure 4, from
  the decoupled counterfactual analysis produced by
  `scripts/probe_validity_analysis.py` (section 9.11).
- `PYTHONPATH=src python scripts/probe_validity_analysis.py --verify` ->
  recomputes every field of the committed probe-validity `results` block from
  its `per_seed` block and checks it (13/13 fields agree). Without `--verify`:
  `--shard i --shards n --out DIR [--coupled]` runs the 4-branch counterfactual
  sweep for a seed slice, and `--analyze '<DIR>/*.json' [--out FILE]` merges
  shards, cross-checks branch 0 against the named study summary, and writes the
  document in the committed schema. Ran coupled (reproduces the committed
  results) and decoupled (writes
  `artifacts/study-summaries/decoupled/probe-validity-analysis.json`) -
  section 9.11.
- `PYTHONPATH=src python scripts/factor_map_recheck.py --list` -> prints the
  recheck set (the nine gate studies, the legacy signal one reconstructed rather
  than excluded, plus the guided bound). Membership is decided by whether a
  summary records a gate verdict, not by having a config and a seed block: the
  depth, portability and probe-scoring studies carry both and record a different
  schema, so the `config and seeds` test grew the set from ten to fifty-six and
  `--list` raised `KeyError: 'gate_passed'` instead of printing it;
  `--study NAME --shard i --shards n --out DIR` re-runs a
  shard's slice with the published config and only `independent_arm_rng`
  changed, and `--compare '<DIR>/*.json'>` applies the declared label-agreement
  rule and can write  the result artifact, plus `--summaries DIR` to emit the corrected study
  summaries in the committed schema (validated by `check_summary`, so a summary
  that does not recompute cannot reach a figure). Ran as declared: 144 shards,
  ~6 min wall, `artifacts/study-summaries/factor-map-recheck.json` and
  `artifacts/study-summaries/decoupled/` (section 9.11). Unit-pinned in
  `tests/test_factor_map_recheck.py` (recheck set
  membership, only-the-RNG-changes reconstruction, shard merge order, label
  agreement and flip detection, threshold recomputation from n, and incomplete
  seed sets never reporting agreement).
- `PYTHONPATH=src python scripts/v0_verdict.py --dataset-root <root>` -> computes
  the pre-registered §5.1 verdict from the published seed summaries (verified
  against a live-published dataset and now unit-pinned in
  `tests/test_v0_verdict.py`: correctly aggregates per-arm FLOPs under the
  toy accounting, excludes stray complete seeds outside the declared block with
  a warning, refuses unmatched mutation budgets or missing arms, and returns
  FAIL with the missing-seed list until the queue completes; the harm-guard
  boundary (reject at >= 10 adaptive losses) is asserted on both sides; the
  adopted section 5.5 pool-matched clause is pinned over
  `POOL_CONTROL_WIN_LIMIT` - a control clearing the same bar withholds Gate 4
  (`PASS_COST_ONLY_POOL_MATCHED_CONTROL_PASSES`), one win below it does not,
  and a **missing** control withholds Gate 4 too
  (`PASS_COST_ONLY_CONTROL_MISSING`) rather than reading as "no pool effect").
  `--control-root` supplies the §2b control session; the declared block is
  overridable with `--seeds "A-B"` (or a comma list) without loosening any
  criterion. The CLIs now expose the adopted protocol
  (`--probe-candidate-policy` / `--probe-selection` on `--mode pair-session`),
  both defaulting to the registered values.
  Exit codes:
  0 = PASS (Gate 4 authorized only with all 12 declared seeds complete),
  1 = FAIL, 2 = PASS with qualification (band miss or harm-guard rejection).
- `PYTHONPATH=src python -m pytest -q` -> 301 passed (adds 9 substrate-audit
  tests: per-layer expert counts build and rebuild at depth > 1; prefix is
  confined to the first layer while spread reaches past it, in both the toy
  path and the V0 session; the three `probe_selection` rules differ, are
  deterministic, and are recorded in the mutation history; and a V0 resume
  that re-executes probe events redraws the identical spread subset
  bit-identically).
- Reduced-dim V0 session check (exploratory, seeds 30000-30011, depth 6 x 4
  experts, cap 4, steps=120): the declared code path's prefix policy probes
  only layers L0-L1 while `spread` probes all six - the structural difference
  confirmed outside the toy path. No adaptive edge appears at this fidelity
  (3/12, 5/12, 5/12; mean deltas ~1e-03), so the confound is not quantified
  there. Consequence recorded in `RESEARCH_REPORT.md` section 9.6: the
  section 5.1 T4 run should not launch on the current pre-registration until
  the candidate policy and a pool-matched control arm are decided.
- Substrate audit runs (exploratory, block 20000-20127): depth-4 cap-4 studies
  pass on both `prefix` and `spread` (47/64 and 48/64 at steps=24); the
  probe-rank gradient is 47/37/35 for best/middle/worst; and the pool-matched
  random control (80/128, p `0.0030`) reproduces most of the argmin edge
  (88/128, p `1.3e-05`) that argmax (64/128) does not. See
  `RESEARCH_REPORT.md` section 9.
- `PYTHONPATH=src python -m compileall -q src/act tests` -> passed.
- The frontend checks that stood here (`npm run typecheck`, `npm run build`,
  `npm test -- --run` -> 185 passed, including the grok-pwa-plugin and preview
  test-port fixes) were dropped with the TypeScript console they exercised; it
  was removed as unmaintained and representing none of the results. See the
  removal record near the top of this document, and git history for the logs.
- Python Experiment 0 mutation-budget test -> passed.
- Python registry/CLI round trip -> passed.
- Initial Gate 3 trial -> adaptive lost to random on seeds 1, 2, 3, and 42.
- Four-seed gate test -> adaptive won 3/4 seeds with equal three-mutation budgets.
- Eight-seed generalization probe -> adaptive won 6/8 seeds; Gate 3 remained open pending a stronger pass.
- Sixteen-seed baseline -> adaptive won 10/16 seeds, with matched budgets and mean loss delta `-0.032691`.
- Expert-signal sweep -> random expert selection remained strongest at 6/8; routing, gradient,
  Fisher, and parameter-norm choices were retained as configurable probes but not selected by default.
- Gate-study CLI -> `PYTHONPATH=src python -m act.evolution.cli --seeds 1,2,3,42 --steps 48`
   emits a deterministic multi-seed JSON summary.
- Gate-study registry round trip -> the same CLI appends exactly one `gate_study` JSONL record.
- Gate-study budget invariant -> study execution and registry records enforce equal random/adaptive
   mutation counts for every seed.
- Research invariants -> passed in the latest checks. (The browser console this
  line also covered was removed with the frontend, so nothing checks it now.)
- Fresh 16-seed Gate 3 study (`1,2,3,42,5,6,7,8,9,10,11,12,13,14,15,16`) -> 9/16 adaptive wins at
  48 steps, mean paired delta `-0.008127`, population standard deviation `0.162910`, and matched
  budgets; the scaling gate remains open.
- **384-seed pre-registered Gate 3 study (seeds 1000-1383, steps=48, budgets matched)**
  -> 195/384 adaptive wins (50.8%), mean paired delta `+0.006766`, stddev `0.131115`,
  exact binomial p `0.3993`, paired-t p `0.8437` (directed: treatment helps = negative
  delta); Gate 3 FAILED for the cheap-signal controller. Reproducible
  (`study_id` `48163d824f2d6852`, recorded in `artifacts/gate3-384.jsonl`).
- **96-seed guided-arm diagnostic (seeds 2000-2095, steps=48, budgets matched)**
  -> 59/96 guided wins (61.5%), mean delta `-0.041325`, binomial p `0.015774`,
  paired-t p `0.00025`; the probe-loss-greedy upper bound PASSED under the same
  verdict machinery (`study_id` `d2b76921ac40d39b`, `artifacts/guided-diagnostic-96.jsonl`).
- Gate-study verdict fields (`gate_alpha`, `gate_p_value`, `gate_min_wins`,
  `gate_passed`, `t_statistic`, `t_p_value_one_tailed`, `treatment`) added to
  `GateStudyResult` and gate-study registry records; `study_hash` now hashes the
  gate-decision inputs rather than float aggregates; the t p-value direction was
  corrected (helpful treatment = negative delta = P(T <= t)); covered by new unit tests.
- New opt-in `guided` arm (`--arms random,guided`, `--treatment guided`) with exact
  mutation-candidate enumeration, probe-loss selection, probe cost in `search_flops`,
  and probe metadata in mutation history.
- **Controller decision rule replaced**: the adaptive arm now defaults to bounded
  probe-loss selection (`adaptive_policy="probe_loss"`, `probe_candidate_cap=4`,
  `probe_batch_count=2`) via a shared probe-mutation engine; the legacy
  routing/gradient/Fisher rule remains available as `adaptive_policy="signal"`.
  Studies and registry records now include `search_flops_by_arm` and
  `treatment_search_overhead_ratio` for explicit search-cost accounting.
- **384-seed probe-loss Gate 3 re-run (seeds 3000-3383, steps=48, budgets matched)**
  -> 213/384 adaptive wins (55.5%; 209 required), mean delta `-0.025955`, binomial
  p `0.018138`, directed t p `7.4e-05`; quality PASSED, pre-registered <=5% search
  cost FAILED at 41.9% (`study_id` `27fe7e53d0533f29`,
  `artifacts/gate3-probe-loss-384.jsonl`).
- **380-seed cheap-probe study (seeds 4004-4383, steps=144, cap=2, 1 probe batch,
  budgets matched)** -> cost criterion PASSED (4.13% <= 5%; probe cost amortizes
  with horizon) but quality FAILED: 181/380 wins (47.6%), mean delta `+0.011747`,
  binomial p `0.8351`, t p `0.8857` (`study_id` `41f4b9f7cf6d82e0`,
  `artifacts/gate3-cheap-probe-380.jsonl`, summary in `artifacts/study-summaries/`).
  Confound noted: probe budget and horizon changed together vs the passing
  cap-4/2-batch/48-step study.
- **380-seed same-seed placement study (cheap probe, steps=144, interval=40 ->
  mutations at 40/80/120; seeds 4004-4383 identical to the early-placement run)**
  -> late placement better in 201/380 pairs, mean paired delta `-0.015666`,
  paired t p `0.0788`; late-study vs random alone: 198/380, binomial p `0.2208`.
  Placement changes 39% of probe decisions but does not statistically explain
  the horizon effect (`study_id` `34a9a5216a8aac36`,
  `artifacts/gate3-late-placement-380.jsonl`, summary in `artifacts/study-summaries/`).
- Added `--mutation-interval` and `--mutation-budget` to the experiment CLI
  (were config-only).
- **384-seed variance study (seeds 6000-6383, cap=2 candidates, 4 probe batches,
  3 events, steps=144, budgets matched)** -> quality FAILED: 179/384 wins (46.6%,
  threshold 209), mean delta `-0.003907`, binomial p `0.9159`, t p `0.3453`;
  probe cost 16.0% (descriptive). Probe variance reduction alone does not restore
  the adaptive edge (`study_id` `cdebaf2ed9f23f54`,
  `artifacts/gate3-variance-384.jsonl`, summary in `artifacts/study-summaries/`).
- **384-seed horizon replication (seeds 8000-8383, the passing cap-4/2-batch
  config at steps=144, budgets matched)** -> quality PASSED decisively: 231/384
  wins (60.2%, threshold 209), mean delta `-0.052402`, binomial p `4.04e-05`,
  t p `1.07e-08`; probe cost 14.0% (41.9%/3, matching the amortization model);
  clears Bonferroni alpha/7 ~ 0.0071 on both statistics
  (`study_id` `162362a15b1d6aee`, `artifacts/gate3-horizon-384.jsonl`, summary in
  `artifacts/study-summaries/`).
- **384-seed richness study (seeds 7000-7383, fixed nominal probe tokens of
  192/event, 3 events, steps=144, budgets matched)** -> arm A cap-4/1-batch
  PASSED: 228/384 wins (59.4%), mean delta `-0.036982`, binomial p `1.4e-04`,
  t p `5.0e-05`, cost 7.3% (`study_id` `994282a17955597e`); same-seed arm B
  cap-2/2-batch FAILED: 193/384, mean delta `+0.007156`, binomial p `0.47965`,
  cost 8.1% (`study_id` `de2f894d75eaedac`); paired A>B in 225/384 pairs
  (sign z `3.37`, two-sided p `8e-04`). Candidate richness is the active axis
  (`artifacts/gate3-rich-384.jsonl`, `artifacts/gate3-narrow-384.jsonl`,
  summaries in `artifacts/study-summaries/`).
- **384-seed cadence study (seeds 5000-5383, steps=144, interval=24, budget=6,
  cheap probe cap=2/1-batch, budgets matched)** -> 137/384 adaptive wins (35.7%,
  threshold 209), mean delta `+0.047014`, binomial p `1.0`; adaptive significantly
  WORSE than random (z `-5.61`, two-sided p ~ `2e-08`); probe cost 8.41%
  (descriptive only - the 5% cap was not part of this pre-registration)
  (`study_id` `964e3f08e13a3710`, `artifacts/gate3-cadence-384.jsonl`, summary in
  `artifacts/study-summaries/`).
- **Probe-landscape instrumentation** (`--probe-landscape`, `--probe-force-candidate`,
  `--probe-force-events`) records every candidate's probe loss per mutation event
  and supports deterministic counterfactual branch runs; forced index 0 is verified
  trajectory-identical to greedy selection. Probe-validity counterfactual study
  (384 seeds x 4 candidate branches at event 1) recorded in
  `artifacts/study-summaries/probe-validity-analysis.json`.
- **Report figures generated** (deliverable 12): the four pre-registered figures
  (`artifacts/study-summaries/figures/`) are produced by `scripts/generate_figures.py`
  from the versioned summaries via `act.evolution.figures_data`; the script
  recomputes every verdict (binomial p, pass flag, mean delta, probe-cost
  identity, argmin concordance) from raw per-seed records before plotting and
  fails on any mismatch. Re-runs are byte-identical. Data assembly is unit-tested
  in `tests/test_report_figures.py` (26 tests; statistics helpers pinned to
  scipy reference values); run `PYTHONPATH=src python scripts/generate_figures.py`
  (requires matplotlib, not needed for the test suite).
- Gate studies now fail fast with a clear error when the `random` baseline arm is
  missing from `--arms`.
- The two frontend fixes once recorded here (`injectGrokPwaHead` preferring the
  document title for `og:title`, and preview test ports matching `PREVIEW_PORT`)
  are spent: the code they patched was removed with the console.

## Environment Notes

- Git was initialized on 2026-09-12; public project baseline committed as `90416f3`.
- The app uses port `4173` because port `8080` is occupied by an unrelated local service.
- The installed PyTorch CUDA build does not support the local GTX 1070, so Python validation
  currently runs on CPU with a warning.
- The installed PyTorch CUDA build (2.14.0+cu126) runs kernels on the local GTX 1070
  (sm_61): `torch.cuda.is_available()` is true, matmul launches succeed, and the local
  proof run (below) trained on GPU at ~15k tokens/s. Earlier "CPU only" notes are stale
  for the current environment.
- Operational hazard (learned 2026-09-13): `V0SessionConfig` defaults are the DECLARED protocol
  dimensions, so a partial config silently instantiates the full 285M-parameter model on CPU
  (~1.1-3.4 GB checkpoints; a test run once accumulated 19 GB in /tmp before hitting the disk
  quota and hanging). Any reduced-fidelity session - tests, self-check, smoke - must set
  vocab/context/hidden/depth/expert dims explicitly; `scripts/v0_session_selfcheck.py` and the
  test suite both do. The `declared=True` guard rejects undeclared dimensions before any
  training work starts.
- Operational note (learned 2026-09-13): background processes must be launched with
  `setsid nohup ... < /dev/null &` from the terminal tool - a plain `nohup ... &` child is
  killed when the tool call that spawned it returns.
- Local proof run (2026-09-13): `scripts/local_proof_run.py` executed the section 5.1
  protocol machinery end-to-end on the local GPU with the production runner at reduced
  dims (12.55M total / 6.26M active parameters; vocab 8k, context 512, depth 8, 8 experts
  of 2 active, cap-4/1-batch primary controller; E=3; steps=600; batch 8) over the declared
  queue 9000-9011. All 12 pairs published complete; `scripts/v0_verdict.py` returned
  PASS_WITH_QUALIFICATION: measured probe cost 1.0044% (primary <=5% PASS), the declared
  285M band [0.0471%, 0.1883%] correctly not applied to a reduced model, harm guard clear
  (7/12 adaptive wins - unpowered at this scale, as expected for the cost-only protocol).
  The section 5.1 FLOPs identity S = E*k*Bp*2*N_active reproduced exactly (search FLOPs
  bit-match the formula), measured cost within 0.4% relative of the reduced-dims identity
  prediction (1.0014% vs 0.9977% on seed 9000) - the identity is dimension-independent,
  which is precisely what the tolerance band is designed to test at declared scale.
  Scratch roots: `artifacts/local-proof/{sessions,datasets}` (gitignored); verdict JSON at
  `artifacts/local-proof/datasets/v0-5-1-verdict.json`.
- `startup.sh` has a separate preview-process exit-code issue to clean up; it is not part of
  the scientific implementation and should not be solved by killing the unrelated 8080 service.

### 2026-09-25 - the code-LM leg: condensing a 1.3B code model into the project's sparse student

The TinyStories leg proved the *path* on English. It did not touch the question the
project is actually about, which is whether a large model's behaviour can be
condensed into a smaller, sparse, learned-topology network that does the heavy
lifting with a fraction of the parameters. Three files now attempt that, and the
campaign that runs them is `scripts/run_code_lm_campaign.sh`.

**The design, and the pins that make the comparison mean something.**

- Teacher: `deepseek-ai/deepseek-coder-1.3b-base` - 1.3B dense, 2T code tokens,
  public, native to `transformers`. Student: `act.model.V0MoECausalLM` at `match`,
  **166,423,296 total / 81,488,640 active** parameters (h768 d12 e8 top-2 w768).
  Reference: `bigcode/tiny_starcoder_py`, 164M dense, the same size class, trained
  on **100B tokens on 32 A100s**.
- Metric: **bits per byte on held-out Python**, not perplexity. Perplexity is a
  function of the tokenizer, so a 32k teacher, a 32k student and a 49k baseline
  cannot be compared with it; bits per byte divides by the bytes of the *text* and
  every model tokenizes that same text with its own BPE.
- The teacher's distribution is **cached**, not live: it scores a fixed grid of
  non-overlapping 512-token windows once, and the student trains against the saved
  top-32 + a residual bucket. Two reasons, one of them not about cost. The cost is
  that an 8 GiB card cannot hold a 1.3B teacher and a 166M student at once. The
  other is that non-overlapping windows make the teacher's context at a position
  *identical* to the student's, because both see the same window causally; a
  sliding window would give the teacher strictly more context and flatter the
  student by an unknown amount.
- What is stored per position is `log p_i - log p_top1`, plus `log p_top1` and the
  residual mass. Float16 resolves a difference near zero to ~1e-3 and a
  log-probability near -10 to ~1e-2. Keeping the residual means the KL is computed
  over the whole vocabulary instead of being truncated to 32 tokens and silently
  renormalised, which would make the student's own tail free (pinned by
  `tests/test_train_code_lm.py` in both directions: a student that ignores the
  teacher's tail is charged for it, and a student that *matches* it scores
  exactly zero).
- Both arms train on **the same windows in the same order from the same seed**;
  only the objective differs. This was wrong at first - the cross-entropy arm was
  reading the full 162M-token stream while the distilled arm could only see the
  scored 15.36M - so distillation would have been credited with a 10x difference in
  data. The training window range is now pinned by the requested token budget rather
  than derived from what the teacher scored.

**Measured on this box (GTX 1070, 8 GiB), not estimated.**

| quantity | measured |
|---|---|
| corpus | `codeparrot/codeparrot-clean-valid`, 61,373 files, 597 MB of Python |
| stream | 162,411,637 train tokens; 3,069 files held out (9,228,190 tokens / 28,707,445 bytes) |
| teacher load | 2.8 s, 1,346.5M parameters, 2.51 GiB |
| teacher scoring | **1,050 tokens/s** (flat across batch 2-8: logits are 511x32,256 per sequence) |
| student training | **~2,950 tokens/s** at 166M total / 81.5M active, batch 8 x 512 |
| budget | 15,360,000 teacher-labelled tokens (30,000 windows, 4.05 h), 11,250 steps per arm (46.08M student tokens = 3 passes, 4.34 h each) |

**The honest ceiling, stated in the source rather than in a footnote.**
`tiny_starcoder_py` saw 100B tokens on 32 A100s in 18 hours. This is one 1070 and
three orders of magnitude fewer tokens. Nothing here closes that gap, and a run
that landed near the teacher's published 0.18 MultiPL HumanEval pass@1 would be
evidence of a bug rather than of distillation working. What the leg can show is
whether the sparse architecture is a viable *container* for a large model's
behaviour, on the same held-out Python as every model it is compared with.

**Defects found while building it, all of which would have produced plausible numbers.**

1. **The token stream was silently invalidated by a teacher swap.** The teacher was
   changed mid-build (see below), and `tokens.npy` was then a stream of perfectly
   well-formed ids in *another* BPE. Nothing raises on that: it trains, and every
   number out of it is meaningless. The manifest now carries a `tokenizer_id`
   (repo + SHA-256 of the tokenizer bytes), `stream_cache_usable` refuses a stream
   whose id does not match, and the guard is pinned by test.
2. **The document separator was hardcoded.** `<|endoftext|>` exists in
   StarCoder-family vocabularies and does not exist in DeepSeek's, where the stage
   died with "the teacher tokenizer has no <|endoftext|> token to separate files
   with". Now a name list with a fallback to the config's `bos_token_id`/`eos_token_id`.
3. **The student was sized to the wrong vocabulary.** The teacher's tokenizer emits
   32,022 ids while the model has **32,256** output rows. Built on the smaller
   number, a teacher top-k landing on a reserved row indexes past the student's
   logits. The student is now built on the model's width, from `config.json`.
4. **`model(tokens).float()` is not how a `transformers` model returns logits.** The
   first scoring pass died with `AttributeError: 'CausalLMOutputWithPast' object has
   no attribute 'float'` - the good version of that mistake. Absorbed by
   `forward_logits` / `model_logits` so both model kinds work at every call site.
5. **A resumed target cache was resumable into the wrong shape.** Changing the token
   budget changes the array shapes; resuming would either raise or, worse, read a
   window of one grid as a window of another. A shape mismatch now rebuilds.
6. **The HuggingFace cache directories are not writable on this machine, and the
   failure names neither the model nor the fix.** `HF_HUB_CACHE` points at a
   read-only mount, so `trust_remote_code` loading died with
   `PermissionError: .../models/huggingface/modules`, and `huggingface_hub` itself
   raises on a token file it cannot read. `ensure_hf_cache()` redirects the three
   caches under `artifacts/` when the ambient ones are unusable, and
   `_nearest_writable` guards `Path.exists()`, which *raises* rather than answering
   `False` for a path under an unreadable ancestor.
7. **Single-connection HF downloads run at 150 KB/s here, and eight parallel range
   requests at 5.6 MB/s.** A 37x difference for twenty lines; `fetch_parallel`
   exists for it, and it resumes.

**A teacher was chosen twice, and the first choice is worth recording.**
`bigcode/santacoder` was the better story - a Python specialist, and the model
`tiny_starcoder_py` descends from. Its repository ships GPT-2-with-multi-query-
attention modelling code written against transformers 4.24, which no longer runs on
transformers 5.x: it imports `transformers.onnx` (removed in v5), then
`transformers.utils.model_parallel_utils` and several `add_*_docstrings` helpers
(also removed). Pinning `transformers<5` to use it resolved to **4.17.0** (Feb 2022)
and downgraded `huggingface-hub` - a worse problem than the one being solved. The
switch to `deepseek-ai/deepseek-coder-1.3b-base` fixed the version problem and
improved two other things: it is native (`LlamaForCausalLM`, loads in 2.8 s with no
remote code) and its 32k vocabulary - against `Qwen/Qwen2.5-Coder-1.5B`'s 152k -
leaves room for depth in a student that must pay for its embedding matrix.

The 4.6 GB SantaCoder download was spent before that decision and is not kept.

**The Kaggle half, and what the first push cost.**

The local card runs the campaign serially because one 8 GiB device cannot do two of
these at once, and the teacher's bits-per-byte pass sits at the *end* of that chain
while depending on nothing in it. So that pass runs on Kaggle instead, beside the
local run rather than after it (`scripts/push_code_bench.py`). Three facts decided
the shape of the job, two of them already recorded in `T4_RUNBOOK.md` because they
had each cost a run:

- **Quota, not a queue, is the binding constraint on the GPU.** Measured 2026-09-25:
  **28.70 of 30 GPU-h already spent**, 1.30 h remaining, refresh at 00:00 UTC. The
  TPU line in the same call is 0.33 of 20 h used. The push did not queue - it went
  `RUNNING` within seconds - so the earlier note about a long TPU queue is the true
  half of it, and the expensive half is the weekly GPU allowance. The driver now
  refuses to push when `kaggle_queue.quota_shortfall` says the declared window does
  not fit, before publishing anything.
- **`/kaggle/working` is 19.5 GiB against a 7.9 TB root overlay**, and this job pulls
  2.7 GB of teacher plus the corpus and its caches. All of it goes under `/tmp`; only
  the JSON results go to `/kaggle/working`.
- **A pushed script kernel receives only its entry text**, so the entry is generated
  and its placeholders are checked by test - a leftover is a `SyntaxError` in a run
  that has already queued and will still bill its minutes.

**The first push failed, and the failure was ours.** The entry passed a bare
repository id where `code_bench.py` requires `hf:<repo>`: the run installed
transformers (26 s), downloaded the 2.7 GB teacher at **36 MB/s** (74 s), built the
held-out split, and exited in three seconds with
`--model must be hf:<repo> or ours:<checkpoint path>`. Two things follow, and the
second is the one that matters. The `--model` kinds now live in one constant
(`code_bench.MODEL_PREFIXES`) that the driver's tests check the kernel's strings
against, because the test that *did* exist asserted the ids merely looked like
repository names - which is how a duplicate of the driver's list in the test file
managed to agree with the driver and disagree with the parser. And `--fetch` now
prints the kernel log as well as the output files: the first version printed only the
JSON, so an erroring kernel arrived as a summary with two empty rows while the line
naming the cause was downloaded and thrown away with the temporary directory.

Also added: `scripts/code_lm_table.py`, because three scripts write the numbers this
leg is judged on and a table assembled by hand from three places is one that can
quietly lose a row. It refuses a missing role and refuses rows measured at different
`bytes_limit` values - the second because bits per byte is only comparable on the same
held-out text, and the limit is a per-machine decision (4 MB of prefix locally,
escalating the teacher's pass from hours to minutes).

### The distillation claim is a paired test now, not two four-decimal numbers

The two student arms are scored on the *same* 512-token windows of the *same* held-out
file, with the *same* tokenizer, differing only in the objective. That is a paired
experiment, and it was being reported as two aggregates and a subtraction - a format in
which a 0.0004 bits/byte difference is indistinguishable from numerical dust, which is
exactly what the two arms' difference is expected to be at this budget. Scoring now
returns the per-window losses alongside the total (`code_bench.window_nlls`, replacing
the total-only `nll_of_ids`), each row records its window count and its tokenizer
identity, and `code_lm_table.paired_contrast` tests the differences:

- **the result is the interval, not the p.** Mean gain in bits/byte with a 95% `t`
  interval, over the paired differences, plus the count of windows that moved each way
  - so a reader can see whether the effect is carried by a few windows or by all of
  them.
- **the p-value is a sign-flip permutation**, matching `act.search.architecture_report`:
  enumerated exactly while `2 ** n` assignments fit under a 1,000,000 cap, otherwise
  200,000 Monte-Carlo draws under a fixed seed with `np.random.default_rng`, in batches
  because 2,500 windows one draw at a time is minutes. The paired `t` p is printed
  beside it as a reference.
- **it is reported as a lower bound, and says so in the output.** Windows cut from one
  file are not independent, and nothing in this pipeline measures the dependency; the
  line under the test tells the reader to read the interval as the result.
- **six ways of producing a plausible-looking but unpaired comparison are errors**, not
  notes: window lists that disagree with their own total or their own window count
  (two halves of one report from different runs), arms tokenized differently, arms with
  different window counts, arms with different held-out byte counts, a report with no
  per-window losses at all (pre-refactor, so a total), a report with no tokenizer
  identity, and fewer than 30 windows.

**The first fetch of the Kaggle kernel paid for itself, and the bug was in the scan.**
The byte pass had failed on both models, and the log names it: `nll_of_ids` collected
windows into a batch and called `np.stack` on them, but every window is 513 tokens
except the **last**, which is whatever is left of the text. The baseline's tokenizer
makes 1,106,839 tokens of the 4 MB prefix, so the tail is 407 tokens, the batch is
ragged, and numpy refuses it: `all input arrays must have the same shape`. It had
already cost a 2.7 GB download and nine of the ten minutes of that run, and it would
have failed the local campaign's bench stage the same way thirteen hours later - the
shape is a property of the text and the tokenizer, not of the device. The fix scores
the tail as a batch of its own against the empty-batch guard, and the regression test
uses those exact numbers. Worth recording about the fetch itself: `--fetch` printed
the kernel log and the two empty output files, which is the difference between "the
run produced nothing" and "the run failed at `scripts/code_bench.py` line 137" - the
version of this driver that printed only the JSON would have reported the former.
`--fetch` also gained `--save`: `kaggle kernels output` writes into a temporary
directory that the fetch then deletes, so the run that produced those two numbers had
nothing left on disk to show for the quota it spent. The report files are the artifact.
The fixed scan was pushed as kernel version 3 (0.09 GPU-h spent on the failed attempt,
1.21 h remaining before the midnight UTC refresh).

**A second failure was waiting on the same path, and running it found that one too.**
With the ragged window fixed, `code_bench.py --model ours:artifacts/...` died in
`score_ours` - the report records the checkpoint as `checkpoint.relative_to(ROOT)`, and
the campaign passes a *relative* path against an *absolute* root, which is a
`ValueError`. It would have hit the campaign's four benchmark rows the same way, after
thirteen hours. Both bugs sat on a code path the suite covered only through `hf:`
models and hand-built dicts, so the check that found them was a real run at a small
scale: `--model ours:artifacts/code-lm/checkpoints/toy-distill.pt --scale toy --suite
bytes --bytes-limit 200000` on the real held-out text, three seconds on CPU. That run
is also the producer-consumer contract verified on real data rather than on a fixture:
125 windows, the 125th being the 315-token tail scored as 314, `sum(window_nlls)`
equal to `nats` to four decimals, and the tokenizer id equal to
`train_code_lm.tokenizer_id()`, which is the string `code_lm_table` refuses to pair two
arms without.

Two more things only showed up while writing the tests. A **zero difference is free to
change sign** under the permutation, so the two-ties-two-positives case counts 8 of 16
assignments and not 4 - the first draft of the test asserted 0.25 and was wrong about
its own arithmetic, which is a good argument for the exact branch existing at all. And a
**trailing fragment of under 8 tokens is not a window**: the scan drops it rather than
weighting nine tokens of context equally with 512, and that is now pinned by a test
rather than left as a line of code nobody could see.

Not yet done, and deliberately: no claims were added to `scripts/verify_record.py` for
this leg, because the campaign's two arms are still training and there is no table to
claim from yet.

### The first numbers of this leg, from the Kaggle half

The fixed kernel (version 3, **9 minutes of GPU quota, 1.07 h remaining**) completed both
rows on a T4, on the same 4,000,000-byte held-out prefix and the same protocol the local
card will use:

| role | model | bits/byte | teacher pass took |
|---|---|---|---|
| teacher | `deepseek-ai/deepseek-coder-1.3b-base` (1.3B) | **0.4686** | 278 s |
| baseline | `bigcode/tiny_starcoder_py` (164M) | **0.5372** | 92 s |

Saved to `artifacts/code-lm/bench/kaggle/` rather than printed and deleted - and then
**committed**, because the journal quotes these numbers and a record that cites a path
the blanket `artifacts/*` rule hides is a record that cannot be re-read. `tests/
test_code_lm_evidence.py` reads them back through the same reader the table uses, pins
both numbers at the four decimals the journal writes, checks the kernel reported no
failed stage, and reconstructs the ragged tail from the artifact (1,289,028 tokens and
2,518 windows for the teacher, 1,106,839 and 2,162 for the baseline, so the last window
is a 325- and a 407-token window respectively - the shape that crashed the first run).
They are also the first real-data exercise of the whole byte path.

One asymmetry worth knowing before anyone asks for a teacher-versus-student paired test:
an `hf:` row records its tokenizer as a repository id, while a student row records
`tokenizer_id()`, which is that id plus a hash of the tokenizer file. The two student arms
agree and are the pair this leg tests; a teacher-to-student pair shares the BPE in fact
but not in that string, so the table refuses it rather than comparing window 7 of one
model with window 7 of another. Recorded in `code_lm_table.report_tokenizer` rather than
loosened, because the obvious fix - compare the part before the `@` - would accept two
different tokenizers from the same repository.

Three things to read in them. **The reference scale**: a uniform distribution over the
teacher's 32,256 tokens is about 3.8 bits/byte on ~3.9 bytes per token, so both rows are
more than seven times better than knowing nothing - and the number the student has to be
compared against is **0.5372**, not the teacher's. **The gap between the two rows is
0.069 bits/byte.** A 164M model that saw 100B tokens gets within 13% of a 1.3B model on
this text, which is the honest ceiling for a 166M student that sees 15.36M labelled
tokens: distillation transfers a *distribution* over the corpus, not the teacher's
training. **And the teacher's number is contaminated.** `codeparrot-clean-valid` is
public, in The Stack, and almost certainly in deepseek-coder's training data, so 0.4686
is an upper bound on what that model really does on unseen code. The student's number is
not contaminated in that way - it never sees the text - which means a student that
*beat* the teacher would be the first sign of a mistake, not of success. That framing
belongs in the README section when the table is written.

### The HumanEval protocol was stricter than the published one, and that is a bug

Both HumanEval generator paths (`hf:` and `ours:`) had never actually run against a
real checkpoint, so they were exercised on CPU before the campaign reaches them: the
toy student passed 0/4 problems in six seconds and the local baseline copy ran the
`hf:` path clean, which is what those runs were for. They also turned up something
worse than a crash. **This file asked for 256 generated tokens and sampled the full
multinomial at temperature 0.2, while `bigcode/tiny_starcoder_py`'s published 7.84%
pass@1 was measured by `bigcode-evaluation-harness` at `max_length_generation=512`,
`temperature=0.2`, `top_p=0.95`.** Both differences push the number down: a 164M model
often needs more than 256 tokens, and sampling a 32k tail at temperature 0.2 spends part
of the budget on tokens the harness would never draw. The baseline row exists to be
compared against 7.84%, so our protocol was measuring the protocol.

Fixed by matching it: one `sample_token` helper used by both generators (so the student
and the two `transformers` models are drawn from the same distribution of choices and
the protocol is one protocol rather than one per model family), the three settings as
named defaults, and - like `bytes_limit` - **the decoding is recorded in each report and
the table refuses rows decoded differently**, including a row that does not say how it
sampled. What cannot be matched is the estimator: the published figure is the harness's
many-sample unbiased `pass@1` and this scores one sample per problem, ±1.7 points at
that rate, so it is a scale reference rather than a target. Cost is real and accepted:
the benchmark stage's HumanEval half roughly doubles, the teacher being the slow row.

Two process notes worth carrying forward. This was found by running the paths rather
than reasoning about them, which is now three real bugs in a row on the benchmark stage -
all three would have surfaced after the campaign's nine hours of training. And running a
full 164-problem pass *while the campaign held the card* was a mistake: it competed with
the teacher-labelling stage, which is the critical path, and did not finish. Anything
that touches the GPU waits its turn.

### A KV cache cannot reproduce the student's HumanEval generation, so that lever is closed

The checklist below listed a KV cache in the student's attention as the first lever on the
1.5-4 h per arm the students' HumanEval costs. It was written in this session, measured,
and removed again before it could reach the running campaign, and the reason is worth more
than the code: **a cache cannot reproduce what that generator does, at any level of care.**

What was built: an inference-only cache of pre-rotary keys and values threaded through
`_SelfAttention` / `V0MoEBlock` / `V0MoECausalLM`, rolling with the window (`make_room`
drops the oldest entries so positions can be re-numbered from zero), refusing `gdn`/`hybrid`
layers whose recurrent state cannot be rolled back. It is exact where it looks like it
should be exact: over a `context_size=8` toy, prefill-then-decode with no roll agrees with
the uncached forward to **5.96e-08**, and after a roll the cached layer-0 *keys, values and
query* all still agree to the uncached forward's to **5.6e-08 / 1.0e-07 / 3.0e-08**. That
is the trap: every quantity that is easy to check matches.

The generation that matters, however, is 139 steps of a 64-token window: there the cache
reaches **1.03 max logit difference with 134 of 139 argmax flips** against the uncached
forward. Localised, at the first roll, on a 2-layer attention-only model: the layer-0
*output* of a token whose position changes between the two windows differs by **0.024**
(it sits at position 1 in the prefill, where it can see the token before it, and at
position 0 in the rolled window, where it cannot). So rolling the window does not only
re-number and rotate the cached tokens - it re-runs each of them against a different key
set, which changes their hidden states, which means the layer-1 keys stored during the
prefill were never the layer-1 keys the uncached forward computes. No amount of storing
or re-rotating fixes that; a cache is only available to a generator whose positions do
not move, and this one's positions are re-numbered from the window start at every step.

The alternative that *is* cacheable is monotonic positions (what the two `hf:` rows and
the published harness do). That is a different generation protocol - and at
`context_size=512` it extrapolates past every position the student was trained on, since
training always fed windows numbered from zero - so adopting it would have to be recorded
as a protocol change on one row's side of the table, mid-campaign, and would still not be
the protocol the baseline's published figure was measured under. Not worth it for the row
that exists as a scale reference. The lever list below is consequently `--limit` and
bits/byte-only; `tests/test_code_bench.py` now carries this measurement as a test so the
next session does not rebuild the cache.

Two things this leaves stated rather than implied: the student's generator keeps the
**prompt's tail** and generates into a window whose positions restart at the window start
(the teacher and the baseline generate with monotonic positions and their own caches), and
the cost of the students' HumanEval is unchanged by any of the above - the bytes half of
the benchmark is 2,518 parallel window forwards, minutes, and was never the bottleneck.

### The final table command was run before the rows it is waiting for existed

Four rows were fabricated in `/tmp` out of the two real Kaggle byte rows - the teacher's
own per-window losses rescaled by 0.98 and 1.00, relabelled with the student's
`tokenizer_id()` so the pairing the table will actually do is the pairing that was tested -
and the runbook's table command was run against them. It works end to end in **~6 s**, of
which the 200,000-draw sign-flip over 2,518 windows is 5: the exact command in item 2 below
has been executed, not just written down.

It found a defect in the output that reading the code did not: the architecture column was
`", ".join(block["architecture"] or [])`, so a report carrying a bare string printed the
model's name one character at a time - the fabricated student rows came out as
`V, 0, M, o, E, C, a, u, s, a, l, L, M`. Two things were wrong behind that. `score_ours`
recorded no architecture at all (the `hf:` rows record `config.architectures`), so the two
rows this leg is about would have read as the bare word "ours" in the one column that names
their architecture; it now records `[type(model).__name__]`. And the column now normalises
the field instead of joining whatever it finds, because the next writer to put a string
there would have produced the same garbage silently. Both are pinned by test.

The same run confirms the two states the runbook leans on. A partial directory refuses:
`no report for role 'teacher' (the 1.3B model being condensed - the target) ... a table
missing one reads as a result rather than as an unfinished run`. And the item-2 snippet,
pointed at those same two student rows alone, prints the paired line with no teacher or
baseline present at all - which is the whole reason the benchmark row is now written per
suite.

### The control arm would have started from another run's weights

An audit of the two training arms - the one stage that had not been walked through the way
the benchmark stage was, and the one that runs for eight and a half hours - found the
equivalent of this leg's first defect, one day on. Stage 2 has no `--force` and no resume
flag: `stage_train` adopts `checkpoints/<scale>-<mode>.pt` if it exists. `match-ce.pt`
existed, 2.0 GB, written at 15:32 by a **20-step smoke run** (its payload: `step: 20`, one
history entry, val loss 8.56). The distilled arm would have started from its seeded
initialisation at step 1; the control arm would have loaded those weights and optimiser
moments, logged `resumed at step 20`, trained 11,230 steps and reported the full budget. The
two arms would not have been the same experiment, and the numbers would have looked
*more* alike than they should - which is the direction nobody checks.

**Fixed twice over.** The checkpoint payload now carries the run it belongs to - scale,
mode, seed, batch size, max steps, window count - and `checkpoint_matches` refuses a
checkpoint whose run differs, and refuses one that carries no run at all rather than assuming
it is this one. Running the new guard against the actual file gives `max_steps 20 -> 11250`
(or, for the older payload, "written before checkpoints recorded which run they belong to").
The smoke checkpoint and its report moved to `artifacts/code-lm/smoke/` (nothing destroyed,
and the smoke run's own numbers are not sitting where the control arm's result will land).
And the report now says what the arm *did* as well as what it was run to: `tokens_budget` next
to `tokens_seen`, `steps_trained` and `resumed_from_step`. They are equal until something is
adopted, and that equality is the thing a reader was implicitly trusting.

**What the audit found sound, for the record.** The arms' parity does not rest on convention
only. Initialisation is one `torch.manual_seed(SEED)` immediately before the model is built,
with nothing consuming the torch RNG in between. The window draws come from a generator seeded
explicitly and used by nothing else, so the two arms' sequences cannot desynchronise even
though one of them does more work per step. The window range is pinned by the requested token
budget and deliberately not by how much the teacher scored (defect 1 of this leg, and the
reason it is now stated in the code). Optimiser, schedule, gradient clipping, evaluation
batches and checkpoint cadence are single code paths, not per-arm copies. The objective is
the only thing that branches: `(1 - 0.5) * ce + 0.5 * kl` against `ce`.

### The training report's bits per byte was smaller than its own uniform floor

The same audit found a units error in the number each arm's report is read for.
`evaluate_bits` divided a per-*token* loss by the whole held-out **byte** count, dropping the
tokens-to-bytes factor that is the entire reason the metric exists - a per-token loss and a
per-byte loss are related by how many tokens the text took. The 20-step smoke report, kept in
`artifacts/code-lm/smoke/`, states it plainly: `4.308e-07` bits/byte in the validation block,
beside `4.8154` for the uniform floor in the references block of the *same file*. Two numbers
in one report cannot be on scales nine million apart, and the factor is exactly that - the
val stream's **9,228,190 tokens over 28,682,299 bytes** (3.11 bytes/token). Its final val loss
of 8.5647 nats per token is **3.9755** bits/byte, which is a 20-step student sitting just under
the floor of 4.8154 and looks like what it is.

Nothing quoted in the record rests on it: the benchmark's bits per byte is a *total* over the
bytes of the text and always went through `scripts/code_bench.py`, whose numbers (0.4686 for
the teacher, 0.5372 for the baseline) are the ones the leg is measured by. But the arm reports
are what will be read for how the two arms trained, the training log prints this field at the
end of every arm, and the error would have been **equal in both arms** - so the comparison
would have looked fine and the number would have been wrong in the one place a reader looks
for absolute scale. `evaluate_bits` now takes the token count, both callers pass
`len(val_tokens)`, and two tests hold it: the arithmetic itself, and the invariant that a
predictor which is exactly uniform must score exactly the uniform floor.

### Sample generation is inference, and the report says which bits per byte it holds

Two smaller things from the same audit. `train_code_lm.generate` - three prompts, eighty
tokens each, at the end of four hours of training - was not marked `no_grad`, unlike every
other generation path in the project: a graph attached per step for a backward pass that
never comes. Adding it cannot change a number, and that is now pinned rather than assumed -
measured on a toy config with activation checkpointing on and the model in `train` mode, the
logits under `no_grad` are **bit-identical** to the ones carrying a graph (0.0 difference) and
identical again in `eval`. That is the property the arm reports' samples rely on, since
generation runs in whatever mode training left behind.

And the report now says **which** bits-per-byte number it holds. This leg has two: an arm's
report computes it over the *whole* held-out stream as the student tokenizes it, boundary
tokens between files included, while the table's is a fixed prefix of the same file
re-tokenized by each model. Both are "bits per byte on held-out Python" and they are not
comparable - the first is progress, the second is the measurement - so the report carries
`bits_per_byte_measured_on` saying so. Without it the natural mistake is to read an arm's
number against the baseline's published one, which is exactly the comparison the table exists
to make properly.

### A resumed arm would have replayed its windows instead of continuing them

Third finding from the same audit, found by asking what a *faithful* resume needs rather than
by watching something fail. The checkpoint carries the weights and the optimiser moments; it
did not carry the window draws. The draw generator is seeded once per launch, so a resume
restarts the sequence at its first element: the arm retrains on windows it has already seen
and never reaches some of the later ones. The window *set* is still the same and the weights
are still continuing, so nothing looks wrong - but the leg's claim is that the two arms see
the same windows **in the same order**, and a crash-and-resume on one arm would have broken
that quietly while every number stayed plausible. Given that this leg has already produced two
silent *the-arms-are-not-the-same-experiment* defects, a third that only appears if a crash
happens is worth closing before the crash rather than after.

`generator.get_state()` is now written into the payload and `checkpoint_matches` refuses a
checkpoint that does not carry it - so both requirements a resume has, that it is this run and
that the draws are where they stopped, are checked in one place and refuse in the same
direction. The test measures the property instead of the shape: the saved state must reproduce
the *next* draws, not merely equal another state tensor. And the path is now round-tripped
end to end - save, then a relaunch reading it back through `checkpoint_matches`,
`set_state`, `load_state_dict` and `restore_optimizer` - because four of those pieces were
pinned separately and none of them pinned that they still fit together, which is the shape of
every resume defect: two halves that drifted, and a run that looks finished either way.

### The arms are checked against each other now, before the benchmark stage rather than after

The audit above ends with a gap that the flags cannot close: `tests/test_code_lm_campaign.py`
pins the two arms' *invocations*, and the reports are what the arms did. So the premise of the
whole leg - same data, same order, same budget, same seed, objective the only difference - is
now a check over the artifacts, `scripts/check_code_lm_arms.py`. It compares the two reports
field by field and refuses on any difference outside a small declared set (`mode`,
`distill_weight`, `samples`, `validation`), naming the path it found: `training.seed: 11 vs 12`,
or `training.train_windows: 30000 vs 317210` - the leg's first defect, caught the same way.
Resume bookkeeping is printed rather than refused, because a resume is faithful now and hiding
it would be worse than showing it: the line says `started at step 21 (restored by name (110
parameters))`.

It needs no GPU, no teacher and no benchmark stage, which is the point of running it first: a
broken premise costs minutes instead of a night of the students' HumanEval. It writes its
comparison to JSON with `--json artifacts/code-lm/arms.json` - deliberately outside `bench/`,
which the table globs - so "the two arms differ in their objective and nothing else" is a
statement with an artifact behind it. The two `scripts/code_lm_table.py` numbers it prints are
this leg's *own* measure of the whole held-out stream, not the table's prefix, and the script
says so in its output rather than leaving a reader to notice.

### When the campaign finishes

Measured start, for the next session's estimate: the teacher-labels stage began around
15:00 and was logging **1,042-1,046 teacher tokens/s**, so the 15.36M labelled tokens
land near **19:10**; the two arms are 4.34 h each (measured ~2,950 student tokens/s). That
puts the benchmark stage's start near **03:40** and its end depends on the one number
nobody had priced: **the students' HumanEval**. The student has no KV cache, so one
generated token is a full 512-token forward, and at the published 512-token budget that
is 512 forwards per problem over 164 problems, twice. Measured on the development card
*while the campaign held it*: 89 s for four problems for the toy student (30.1M total /
19.5M active), i.e. ~22 s per problem. Scaling that to `match` by the *active* parameter ratio (4.2x) is
the optimistic end of the range; the work a forward actually does scales with depth times
width squared, which is **~8x** from `toy` to `match`, so **3-8 h per student arm** is the
honest estimate and the low end should be read as luck. The `hf:` rows are cheap by
comparison (their generate path keeps a KV cache, and the byte scans total under an hour).
Plan on the labelling stage finishing near **18:50**, the arms near **03:15**, and every
number the table reads landing **about forty minutes after that** - not by 09:00 and not at
the end of the stage. (Forty and not fifteen: the two `hf:` rows are fast, but the teacher's
own HumanEval half is ~19 min of *sequential* generation on this card - 512 single-token
forwards per problem over 164 problems - so it, not the byte scans, is what the fast rows
cost.) Two changes put it there. Each row is written to
`artifacts/code-lm/bench/` **when its byte scan finishes** rather than at the end of its
HumanEval half, and the benchmark stage now scores the two `hf:` rows first and the two
students last: the fast rows are minutes against the students' hours, and the table needs all
four. The two students' HumanEval halves are then the only thing still running, which is the
only place a failure can now cost anything - and it can no longer cost the four rows the
result is made of. Read `/tmp/code-campaign.log` for the truth; it is empty until the
labelling stage hands over.

**What this design can resolve, before the numbers arrive.** The paired interval is what the
leg's claim will be read from, so it is worth knowing what size of gain it can see. From the
two rows that already exist - per-window bits per byte, bytes per window taken as the mean
because the reports do not break it down, which is second-order since most of a window's
spread is the text's difficulty:

| row | windows | mean | sd |
|---|---|---|---|
| teacher | 2,518 | 0.4686 | 0.1627 |
| baseline | 2,162 | 0.5372 | 0.1954 |

If the two arms' window losses were **independent** - the worst case, and a wrong model, since
both arms share data, order, seed, architecture and step count - the paired 95% half-width at
2,518 windows would be `1.96 * sqrt(2) * 0.1627 / sqrt(2518)` = **0.0090 bits/byte**. The
gap this leg sits inside is 0.0690 (baseline to teacher), so a gain an eighth of that gap is
resolved, and positive correlation between the arms only sharpens the interval further. The
paired test is therefore not what limits this leg: the 15.36M labelled tokens are, and no
number of windows changes that.

If that HumanEval half turns out to be the bottleneck, the levers in order of honesty
are: scoring fewer problems with `--limit` and saying so, or leaving the students' pass@1
out and reporting bits/byte only. The KV cache that headed this list is **off it** - it was
built, measured and removed in this session, see the entry above - so `--limit` is now the
first lever and it costs evidence rather than correctness. **Not** reducing `--max-new-tokens`:
that is the protocol the baseline's published number was measured under, and the whole
reason the baseline row is in the table.

The order that turns that into a record:

1. ~~Fetch the Kaggle rows~~ - **done**, `artifacts/code-lm/bench/kaggle/` holds
   `teacher.json`, `baseline.json` and the kernel's summary. They are a *cross-machine
   check of the same protocol*, not a table input: the local campaign scores all four
   models on one card so HumanEval's sampled decoding is not split across devices, and
   the Kaggle files are two of those same two rows measured elsewhere (0.4686 and
   0.5372 below). Comparing them to the local rows is a check on the protocol, and the
   table's device column now names the card so a mix is visible rather than silent.
2. **The two arms first of all**, the moment the second one exits - it is the leg's premise and
   it is checkable before any benchmark runs. `scripts/check_code_lm_arms.py` reads the two
   reports and refuses when they disagree about anything that is not the objective, so an arm
   that resumed from another run's checkpoint, or trained over a different window range, is
   named with the field it differs on (`training.seed: 11 vs 12`) rather than being discovered
   in the table hours later:
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/check_code_lm_arms.py --json artifacts/code-lm/arms.json
   ```
3. **The paired contrast next, the table after that.** Both are available about forty minutes
   into the benchmark stage, and the paired line is the leg's headline - read it first. The
   interval is the result and the p-value is a lower bound, since windows cut from one file
   are not independent:
   ```bash
   PYTHONPATH=src .venv/bin/python - <<'PY'
   import sys; sys.path.insert(0, "scripts")
   import code_lm_table as table
   reports = table.load_reports(table.BENCH)
   tags = {role: "ours-match-" + role.split("-")[1] for role in ("ours-distill", "ours-ce")}
   arms = {role: (tags[role], reports[tags[role]]) for role in tags}
   print(table.format_paired(table.paired_contrast(arms)))
   PY
   ```
4. `PYTHONPATH=src .venv/bin/python scripts/code_lm_table.py --json artifacts/code-lm/bench/table.json`
   - the headline table, and the paired contrast. It refuses to render without all four
   roles, so a partial campaign reports itself here rather than as a shorter table.
5. Write the numbers into the README section and add the claims to
   `scripts/verify_record.py`, then run it and `scripts/check_record_freshness.py`.
   Claims last, because they are checked *against* the prose - a claim written before
   the sentence it guards is a check on a guess.
6. HumanEval pass@1 for the baseline is the one number with an outside reference
   (7.84% published at 512 tokens, temperature 0.2, top_p 0.95, many samples; ours is
   the same settings with one sample, so about ±1.7 points of noise sits on it). The
   decoding now matches the published protocol, which it did not when this checklist
   was first written - see the entry above - so a large gap from 7.84% is now a reason
   to suspect the harness or the subprocess timeout rather than the token budget. The
   check is cheap either way: `--suite humaneval --limit 20` on the baseline.

## The TPU probe's deciding measurement needed its own process, and the fetch had been deleting it (2026-09-25)

The TPU half of this project has been blocked behind one number since 2026-09-21: the
aggregate rate of eight independent seeds. `scripts/tpu_probe.py` measures it, the
pre-registered rule in the same file reads it, and `scripts/push_tpu_probe.py` runs it on
Kaggle's separate TPU quota. What this session found is that the kernel **had already
measured everything else and nobody had read the result** - and that the fetch was the
reason, not the platform.

| | |
|---|---|
| **What was on disk** | `artifacts/tpu-probe/output/tpu-probe.json`, 2,075 bytes, mtime 2026-09-22 21:02. The record of kernel **v3**: four phases completed, `independent_seeds` in `errors`, verdict `INCONCLUSIVE` (exit 3). The watcher that fetched it is dead, and its verdict was never applied or written down. |
| **What the kernel actually was** | `sid123123234/act-tpu-probe` reported `KernelWorkerStatus.COMPLETE` - v4, pushed 2026-09-22 at `1664194` after the four defects v3 exposed were fixed. It finished **three days ago** and no reader had seen it. |
| **Why nobody had** | `push_tpu_probe.py --fetch` unpacked the kernel output into a `TemporaryDirectory`, printed each file, and let the `with` block delete it. So the fetch that finally ran here printed v4's record and kept nothing - the on-disk copy stayed v3's, which is why the file and the printout were byte-identical. The same defect was fixed in `push_code_bench.py` on this same day (`--save` + `save_fetched`) and not in the other fetch. |
| **What v4 measured** | Four of its five phases: bf16 throughput **1,098.2 tokens/s** (first step including compile 67.07 s, steady mean 3.73 s, min 0.98 s), bf16 rebuild **379.4 tokens/s** (first 29.24 s, steady 10.80 s), fp32 throughput **1,121.2**, fp32 rebuild **370.2**, on `xla:0` with `xla_devices: 8`, torch 2.8.0+cpu / torch_xla 2.8.0. |
| **What it did not** | `independent_seeds`: `RuntimeError: Runtime is already initialized. Do not use the XLA device before calling xmp.spawn.` The rule needs `independent_seeds.aggregate_tokens_per_second` and `.workers`, so the run is `INCONCLUSIVE` - never a pass, which is the property `evaluate` was built with. |
| **The mechanism, verified rather than guessed** | `torch_xla._internal.pjrt.run_multiprocess` opens with `if torch_xla._XLAC._xla_runtime_is_initialized(): raise RuntimeError(...)` (torch_xla 2.8.0 source, and the same line in the traceback at pytorch/xla#9303). The probe initializes that runtime **before the aggregate's turn by design**: `_device_report` calls `xm.xla_device()` to record the hardware, and every dtype's throughput pass uses it. `independent_seeds()` then asked for the device count with `xm.get_xla_supported_devices()` immediately before spawning, which initializes it too. |
| **So the fix was not the ordering** | The phase order was already "what a failure costs the decision" - bf16 first, aggregate second, fp32 diagnostic last - and it is *registered*, so it is not what was wrong. Nothing about ordering can help: the runtime is initialized before the first phase runs, so the spawn is illegal in that process however the phases are arranged. The measurement needs a process that has done nothing else. |
| **The authoritative detail that ruled out a stale dataset** | The mounted probe was the fixed revision, not an old one: the run's warnings cite lines 445, 481, 497, 503 and 574 of `/kaggle/input/datasets/sid123123234/act-lab-code/code/scripts/tpu_probe.py`, and those are the same `xm.xla_device()` / `mark_step()` lines in the working tree. v4's four fixes - `nprocs=None`, the rank-completeness check, the steady-sum aggregate, the device-count check - were all in the tree that ran. None of them was this defect. |
| **Fixed** | `seeds_in_child()` runs the aggregate in a fresh interpreter (`sys.executable` + the same probe + `--seeds-only`), reads the child's record and returns its `independent_seeds` block with `taken_in: "child-process"` recorded beside the numbers. `seeds_only_request()` is the mechanism as data: `devices: None`. A child that dies carries its own stderr into the failure, and one that exits 0 without the block is a failure too - both were silent losses before. `spawn_arguments`' `nprocs=None` is unchanged. |
| **And the pre-spawn device count is gone** | It could only be taken by initializing the runtime the spawn needs untouched, so it was part of the cause rather than a guard. With `nprocs=None` torch_xla derives the process count from the TPU's own topology, so the *number of rank reports* is the device count that mattered; requiring all `workers` of them keeps the property (`devices=4, workers=8` and `7 of 8 ranks` are both errors, pinned) with no pre-spawn device query. |
| **And the fetch keeps what it fetches** | `--fetch --save DIR` keeps the record, fetches the kernel log beside it, names every file it kept, and prints the pre-registered verdict for the record that just landed. `save_fetched` moved to `kaggle_queue.py`, because two drivers fetch kernel output now and a second copy would be a second answer to "did anything survive the fetch" - the question that cost this run three days. |

**The reading of the four numbers that did arrive, stated before the fifth is measured.**
The aggregate is the sum of eight ranks' steady rates, and each rank has its own chip and its
own HBM, so eight cores should land near eight times the single-core rate: **1.73x the T4
pair for bf16, 1.76x for fp32**, against a declared bar of **2.0x**. The per-core rate that
would clear the bar is **1,270.8 tokens/s**; both dtypes are below it. fp32 also came out
*faster* than bf16 by 2%, so the bf16 protocol amendment that the rule assumes buys nothing
on this hardware. On the evidence in hand the v5 measurement is as likely to confirm
`DO-NOT-PORT` as to open the port - which is exactly why it is worth taking rather than
inferring: the rule reads the aggregate, this is a proxy for it, and the whole discipline of
this project is that a pre-registered measurement is not replaced by a plausible guess. The
run costs a few TPU minutes out of 19.67 h available; the kernel is pushed and a watcher is
armed, and the verdict will be applied where the record lands.

**What this changes about which resource is worth spending.** The GPU week is 1.07 h of 30
(Wednesday-to-Wednesday, refresh 2026-09-26T00:00Z) and the local card is mid-campaign until
~03:30 UTC, while the TPU week is 19.67 h of 20 -- so the TPU is the only accelerator with
room right now, and it was unusable for one reason: a probe that could not produce the number
its own rule reads. Three runs had failed to produce it, each on a defect in *our* probe - a
148 s crash with no record (09-21), an fp32 rebuild that took the whole run (09-22), and this
runtime check (09-22 evening) - and the first two are already contained by `_run_phase` and
`MEASUREMENT_ORDER`. This is the third, and it is contained by a process boundary.

## The reachable gene is inert, and a width mutation is not a mutation — it is a reparameterisation (2026-09-25)

This entry is the audit that followed the question "is the training time going into the right
thing", and it has one measurement in it that changes a design.

| | |
|---|---|
| **What was asked** | Why training takes as long as it does, and whether the next GPU week should go to the registered Gate-4 campaign or to something else. |
| **Where the time actually goes** | Three separate clocks, and only one of them is throughput. Per-token throughput is near this hardware's ceiling (a GTX 1070: measured **fp16 is 2.3x slower than fp32** - Pascal is 1/64-rate and has no tensor-core path - so precision is not a local lever at all; batch size is, at 1.53x from batch 2 to 8; and `activation_checkpointing` off was a free 1.32x). The second clock is *statistical power*: effects are 0.02-0.16 nats against 0.015-0.03 nats of seed noise, so every cell needs 3-6 seeds and every claim is replicated on fresh ones - 23.6 GPU-hours over 220 runs for the architecture phase, ~52 for the Gate-4 campaign's three queues. The third is the accelerator week, and it is binding: 30 Kaggle GPU-hours against ~18.4 per queue, currently **1.07 h remaining** with the refresh at 00:00 UTC. |
| **The measured speedups that exist and are not being taken** | The teacher's labelling pass is the clearest one: it took **4.24 h locally at ~1,006 tokens/s** for the leg's 15.36M tokens, and the same teacher was measured on Kaggle's T4 at **278 s for 1,289,028 tokens (~4,637 tokens/s)** - a **4.6x** difference on a stage that nothing depends on except the corpus. The same asymmetry is why the horizon arm ran remotely (0.524 h per 12,000-step run against 1.33 extrapolated locally). A screening loop at 400 steps is the other banked one: 1.1 GPU-hours against 3.2 for the same ordering *and* the same joint fit. |
| **What was found about the search space** | The only gene `feasible_expert_mutations` can reach is expert capacity, and the architecture phase measures it as uncorrelated with loss (rho +0.214, p = 0.62 over a x4.51 active-parameter range, fully dense 7th of 8) and, at the onset, as *removing* value when raised (5.53). The genes that pay - residual width, depth, head count at wide width - are unreachable, and the report says so itself in section 10.4. A Gate-4 pass licenses adaptive control of expert capacity and nothing else. |
| **The measurement that changes the design** | Run against `act.model.V0MoECausalLM` at the code-LM leg's `match` shape: **100.0% of the parameters - all 166,423,296 across 254 tensors - have a shape that depends on `hidden_size`**, including the 24.77M-parameter token embedding (14.9% of the model) and, on any real mutation, the AdamW moments (2x the parameters). `grow_expert` adds 3hW to one layer of twelve, which is why its penalty, its clone-inversion and its prune-dominance were measurable locally; a width mutation touches every tensor at once, so the project's whole accumulated understanding of *mutation cost* does not transfer to it. |
| **And the model will not let width move alone** | The declared-config rule is `hidden_size % 128 == 0` and `(hidden_size // heads) % 64 == 0`. Measured by constructing the model: at `head_dim = 64` every +128 of width is **+2 heads** (768/12 = 166,423,296; 832/13 = 182,847,808; 896/14 = 199,665,536), and the only way to hold `attention_heads` fixed while widening is to double `head_dim` (768 -> 1536 at 12 heads = 2.34x), which changes rotary position encoding and the GDN recurrence - **a different attention geometry, not a wider one**. Also confirmed on this model: 768/12 and 768/6 are the *same* parameter count, which is the architecture phase's "heads are free at fixed width" read in the model's own terms. So a width arm is, by construction, an arm about a joint width x heads mutation, with the head half already priced at -0.0553 nats *at wide width only*. |
| **What was written down** | `docs/width-control-design.md` - a design note, not a record. It carries the blast radius above, the constraint table, the mechanism (function-preserving widening, with **RMSNorm's width-dependence named as the obstacle** to exactness rather than discovered mid-run), the declared primaries (`adaptive - random` as the headline, the pool term as a secondary), the two fixed controls that separate capacity from compute, the admission bar, and the four outcomes with the likely one stated in advance - the pool term carrying it while the controller's edge stays flat, which is the same result the churn programme found, moved onto the gene that pays. |
| **Why it is not implemented yet, and this is the part worth carrying forward** | The code-LM campaign's two arms run from the current `src/act/model`, and the second starts when the first finishes. **Editing the model between them would make them two different experiments, and nothing in either arm's report would say so**: `run_signature` carries scale, mode, seed, batch, max_steps and windows, and no identity of the code. The hazard is not hypothetical - this session's own KV-cache experiment touched `src/act/model/v0_moe.py` and reverted it, and the only reason it is harmless is that the revert restored the file to byte-identity with `HEAD` before arm 1 started. Checked for this campaign, read-only: `HEAD` unchanged, `git status` clean over `src/act` and `scripts/train_code_lm.py`, and the newest mtime in that path (18:00 UTC) precedes arm 1's start (18:50:51 UTC) with no committed change in between. That is the substitute evidence, and it is weaker than a record that attests it. |
| **The open item** | `train_code_lm.py` should record a `code_revision` (HEAD plus a content hash of the dirty tree) in its report and in `run_signature` - so a checkpoint adopted from another revision is refused the way a checkpoint from another run already is, and two arms whose code differs are refused by `check_code_lm_arms.py` instead of being reported as "differ in their objective and nothing else". It cannot land while the arms are in flight, because the field would be present on one arm's report and absent on the other - which is *itself* the signature of the defect it guards, so the reader-side clause and the writer-side field have to land together, after the campaign. |

**The conclusion, stated as a recommendation rather than a finding.** Spend the next GPU week
on neither the registered Gate 4 nor a re-run of anything already measured, but on settling
whether the machinery has anything to select between: either the width experiment in
`docs/width-control-design.md`, or - if the preservation contract cannot be met - the honest
alternative, which is to stop describing the program as searching architecture and publish the
architecture result (width is the lever, shallow beats deep, heads are free but only at wide
width) beside the topology result (mutation helps, selection does not). Those are both
publishable; a fifth arm on expert capacity is neither.

## The campaign was stopped and would have stayed stopped: the week was about to expire unused (2026-09-25)

The Gate-4 campaign has been down since **2026-09-21 15:22** (the kernel's last run, and it ended
`ERROR`). Its cause was already in this log - the child's output was block-buffered so the capture
starved, and the three fixes that answer it (`91ac0e8` unbuffers the child, `812bce1` restarts the
capture and keeps the fullest channel rather than the first, the tees and the `finally` keeps)
**have never run on Kaggle**. What had not been done is the thing the health check asks for in as
many words - *read the cause before pushing* - and the state was that nobody was going to.

| | |
|---|---|
| **What was wrong with the report an operator acts on** | `campaign_health.py` printed `published seeds : None of None complete` for all three queues. Two defects behind one line: the live block's dataset (`act-v0-5-2`) has no versions until its first session publishes, so the listing raised; `health()` caught it and set the counts to `None`; and **the reason was dropped**, never printed. So the one line that says how far the campaign has got read like a broken report, and the cause it had in hand was invisible. |
| **What the reason actually was** | Not "no versions yet", which is what this entry's author assumed before printing it: **`403 Client Error: Forbidden`** on `datasets.DatasetApiService/ListDatasetFiles` for `act-v0-5-2`. Kaggle answers **403, not 404, for a private dataset that does not exist** - a fact the session path already encodes (`_is_missing_dataset_error` treats 403/404 as "nothing to restore yet", at both its call sites) and that the health report was throwing away. |
| **Fixed** | The report now prints `published seeds : unreadable - <reason> (of 12 declared)` when the listing fails, and the ordinary `N of 12 complete` when it does not. The denominator is a registered constant, so it survives a read failure; the numerator does not, and the two states - *unreadable* and *empty* - are the ones an operator has to tell apart. Pinned by a test that raises from the listing and asserts the reason and the denominator are printed and `None of None` is not. |
| **Verified before relaunching, on the cheap channels** | (1) **Quota**: 1.07 h against a 7 h window, so the driver must wait; `--max-waits` defaults to 2 and each wait sleeps to the payload's own `refreshAt`, so one wait clears the 00:00 UTC refresh rather than expiring first. (2) **Code dataset**: the driver republished `act-lab-code` at HEAD (`6885e02`) as its first act - a queued probe mounts whatever version is latest, and `tpu_probe.py` is unchanged since the probe's own commit, so the TPU kernel in flight is unaffected. (3) **The missing dataset is a bootstrap, not a failure**: `publish_on_exit` falls back to `dataset_create_new` on 403/404, so the first session creates `act-v0-5-2` when it publishes, and `restore_latest` treats the same 403 as "session 1 has nothing to restore" - checked at both call sites rather than assumed from the docstring. (4) **The treatment guard**: a session refuses to extend a block whose published treatment differs, so the re-reserved block cannot be mixed with the four untreated seeds. (5) **The driver's own stopping rule**: exit 1 (session failed) and exit 4 (completed without advancing) stop the campaign and say so, rather than spending the week repeating a broken session. |
| **What was done** | `scripts/run_gate4_campaign.sh` was relaunched detached. Its own log begins with the health report, republishes the code dataset, and then says: *"only 1.07 GPU-h of the weekly quota is left and a session declares 7 h ... waiting rather than billing a partial session"*. At **00:00 UTC** it pushes a session for the `session` queue, then `control`, then `null`, then the verdict - unattended, because the alternative is that the hours expire. |
| **The honest risk** | The first session of the new block is also the first run of three fixes that have never executed on Kaggle. If it dies, the driver stops at exit 1 rather than looping, and the cause is now read from four channels with the longest winning - so the failure costs one window, not the week. The block it trains is the re-reserved one (`15024-15035`, all three queues together), which is why this is worth a session rather than a fifth read of the same numbers. |
| **Cost** | ~18.4 GPU-h per queue, against 30 in the week; the three queues are ~47-52 h, so the campaign needs two weeks of refreshes and the driver sequences them. Nothing about this spends the local card, which is mid-campaign on the code-LM leg. |

**One more thing surfaced by running the suite while the campaign trains: a check that fires only
when the artifact exists.** `tests/test_report_artifact_links.py` asserts that a test cannot read
a gitignored artifact, and it fired on `tests/test_code_bench.py` naming
`artifacts/code-lm/checkpoints/match-distill.pt` - a path the *campaign had just started writing*.
The test only formats that path (`display_path` never opens it), so the report was a false positive;
but the checker cannot tell a name from a read, and its trigger is therefore data-dependent: it is
silent while the checkpoint is absent and fails once the campaign creates it, which is the worst
moment for an unrelated failure and the reason it was not seen in any earlier run today. Fixed on
the side that is wrong - the test now names a path that is not a file, and asserts it is not, so the
subject (relative-versus-absolute formatting) is unchanged and the check keeps its teeth for tests
that really read something. 1827 tests pass.

## The step loop is at half the card's peak, and the docstring that chose the expert form is 11x off (2026-09-25)

**The question this answers.** The operator asked why a leg of this project takes the hours it
takes, whether training speed is where the effort should be, and how to shorten the gap between a
run and a result. That is a fair question about a leg that spends ~4.2 h labelling and ~4.7 h per
arm, and the repository could not answer it, because its own stated speeds disagree by up to 75x.
So it was answered by derivation from the estimators the project already has, and the answer
reorders the levers.

**The loop is not the problem.** GTX 1070 fp32 peak is `1920 x 2 x 1.683e9` = 6.46 TFLOP/s.
Against it, the project's live rates convert like this:

| run | config | tokens/s | executed FLOPs/token | implied | of peak |
|---|---|---|---|---|---|
| throughput benchmark, fp32 batch 8 | `DEV_V0_CONFIG` | 8,747 | 101.4 MFLOP | 0.89 TFLOP/s | 13.7% |
| dev cycle end to end, 2026-09-19 | `DEV_V0_CONFIG` | 24,299 | 101.4 MFLOP | 2.46 TFLOP/s | 38% |
| code-LM arm 1, today 20:10 | `match` (h768 d12) | 2,731 | 1,055.2 MFLOP | 2.88 TFLOP/s | 45% |

The dev shape is launch-bound at 14.9M parameters; the code-LM shape converts to **45% of peak**.
A perfect rewrite of an eager fp32 loop already at 45% buys well under 2x, so kernel-level and
schedule-level tuning is the *smaller* half of the available speedup.

**The larger half is arithmetic the routing contract does not require.** `_MoEFeedForward.forward`
is the dense form: every expert applied to every token, then top-k selected. `estimate_flops_per_token`
and `estimate_dense_runtime_flops_per_token` separate what the contract books from what the kernels
do, and the ratio is **1.934x at `match`** (545.6 vs 1,055.2 MFLOP/token) and 1.774x at the dev
shape - the same property at both, since `active/total` is ~0.49 either way. The module already
contains the sparse formulation (`eager_forward`) and it is **pinned equivalent by test**, so the
correctness harness for switching forms exists.

**And the measurement the dense form was chosen on cannot have been taken as described.** The
docstring reports 51.8 ms/step eager vs 42.5 ms/step dense for `DEV_V0_CONFIG` at batch 8 x context
512, on this card. At that config a training step is `101.4 MFLOP x 4096` = 415 GFLOP by
construction, so the quoted pair converts to 9.77 and 8.02 TFLOP/s - **151% and 124% of the card's
physical peak** - and the benchmark's own timing of the same shape is **0.468 s/step**, which makes
the quoted 0.0425 s/step **11.0x faster than the same geometry was measured at**. This is a stronger
statement than `PERFORMANCE_NOTES.md` makes when it flags the docstring rates as mutually
inconsistent: the margin the decision rests on is unsupported in the direction that *under*-provisions
a budget. It also explains why the crossover should be re-measured rather than extrapolated in
either direction: `eager_forward`'s masking overhead scales with tokens x layers while the arithmetic
it saves scales with tokens x parameters, so from the dev shape to `match` (11.1x the parameters,
2x the layers) the overhead's share falls roughly 5x - the dev shape's +18% for dense is not evidence
about the code-LM shape.

**The second lever is much larger and it is on the benchmark path.** `train_code_lm.generate` and
`code_bench.sample_token` both sample one token per **full forward over the whole prefix** - no KV
cache, no `past_key_values`. A 512-token completion from a ~185-token prompt therefore forwards
**208,516** token-forwards where a prefill plus cached steps needs **697** (`299x`), and its attention
term is 706x larger. That is the difference between a benchmark that fits in an evening and one that
does not, at 22 s/problem measured for a *toy* student and 1.5-4 h estimated per `match` arm. It is
inference-only and cannot change what an arm learns - so it is safe for checkpoint comparability -
but it is still a change to `src/act/model/`, which every arm imports.

**What no speedup fixes, and the reason the note says so explicitly.** The leg trains 46,080,000
tokens on a 166.4M-parameter student: **0.28 tokens per parameter**. The comparator whose published
7.84% HumanEval this leg is measured against saw 100B tokens on a 164.1M model, ~609 tokens per
parameter - **2,170x** the tokens. A 1.93x step speedup moves 0.28 to 0.54 on a road three orders of
magnitude long. Speed matters because it buys tokens at all and because it decides whether the
benchmark can run per arm; it is not where the model is being lost.

**Written down, and checked.** New `docs/training-throughput.md` (indexed in the README, carrying a
"measurement note, not a record" declaration, because its rates are dated observations rather than
stored rows) with the two levers, the rejected ones (a smaller `--max-new-tokens` would change the
protocol that `code_lm_table` already refuses to break; a second GPU needs a distributed harness; fp16
autocast is a T4 lever, measured 2.3x *slower* here), and the token-budget arithmetic.
`tests/test_training_throughput.py` (7 tests) recomputes every figure from
`estimated_parameter_count` and the two estimators and refuses an edit that drops one, including the
151%-of-peak and 11.0x reproductions. Section 8 of `docs/width-control-design.md` gained the recipe
arm's shape at a matched budget with `tests/test_width_control_design.py` (5 tests) pinning it - and
writing those tests corrected the design note: an exploratory grid had allowed `active_experts=1`,
which `V0MoEConfig.__post_init__` rejects, so the count of shapes holding both budgets within 2% is
**46** on the legal grid, not 32, and the tightest joint fit (`h896 d6 H14 E7 A2 W1568`, `0.999x`
total against `1.005x` active) is better than the row the note first proposed.

**The tree stays frozen.** Nothing in `src/act/model/` or `scripts/train_code_lm.py` was edited: the
campaign's second arm starts after the first finishes, both import those files at start, and no arm
report attests which code trained it. The docstring correction is therefore part of the
re-measurement rather than a separate edit - it is replaced by the numbers that supersede it, at the
shape they were taken on.

Measured state: targets stage complete (30,000 windows, 15,271 s, 1,006 teacher tokens/s); arm 1
(`ours-match-distill`) at step 750/11,250, loss 3.5701, 2,731 tokens/s. 1839 tests pass.

**And the first lever now has a command rather than a plan.** Lever 1 was decided-on by a
measurement that does not survive checking, so `scripts/moe_dispatch_benchmark.py` takes the
measurement: it times the dense form the training loop uses, the existing `eager_forward`, and a
third - **grouped dispatch** (route once, sort the `(token, slot)` rows by expert, one contiguous
matmul pair per expert, one `index_add_` scatter) - at either geometry, in one dtype, on one device,
and it refuses to time a set of forms that disagrees. The gate has already run: grouped agrees with
dense to **5.1e-16** in float64 at the `match` geometry with a bit-identical routing load, which is
the same tolerance the dense/eager pair is pinned at in `tests/test_v0_session.py`. So the candidate
is the same function as the form the arms are training with, and the open question is only which is
faster at each shape - on a CPU smoke run at the dev geometry grouped was 2.14x dense and 1.77x
eager, which is a smoke signal about the shape of the answer rather than a number to plan from. Each
form is priced with its own estimator, so the table cannot credit a form with FLOPs it did not do,
and `of peak` is blank rather than zero off an accelerator, because a CPU run dividing by a GTX
1070's peak is the same category error as quoting a dev rate for a code-LM shape. `tests/test_moe_dispatch.py`
(5 tests) pins the identity against both existing forms, the empty-expert branch (by pigeonhole
rather than by forcing the router, since `topk` breaks ties arbitrarily), and the refusal. 1845 tests.

## Published a speedup this repository had already measured and closed (2026-09-25)

**What happened.** The entry above and the new `docs/training-throughput.md` introduced a second
lever - "generation re-forwards the whole prefix once per token, ~299x to be had from a KV cache" -
and committed it. Reading the journal's own earlier entry, *"A KV cache cannot reproduce the
student's HumanEval generation, so that lever is closed"*, and the test that was written to pin it,
shows the cache was built in this repository, measured, and removed before this session. The
arithmetic in the new note was right and the claim about its availability was wrong.

**The measurement that was already here.** The cache held pre-rotary keys and values, rolled with
the window, and refused `gdn` layers. It is exact everywhere it is easy to check - prefill-then-decode
with no roll agrees with the uncached forward to **5.96e-08**, and after a roll the cached layer-0
keys, values and query still agree to 5.6e-08 / 1.0e-07 / 3.0e-08. Over the generation that matters,
139 steps of a 64-token window, it reaches **1.03 logits of difference with 134 of 139 argmax flips**.
The mechanism is that this generator re-numbers positions from the window start every step, so rolling
re-runs each cached token against a different key set: on a 2-layer model the layer-0 output of a token
moving from position 1 to position 0 changes by **0.024**, which makes every layer-1 key stored during
the prefill stale. Only a generator whose positions do not move is cacheable, and monotonic positions
at `context_size=512` extrapolate past every position the student was trained on - a protocol change
on one row's side of the table, mid-campaign.

**The cost, and why it is small.** Nothing ran on it: the incorrect claim reached a document and a
test, both of which are now corrected in the same session. The alternative I was one command away from
- building the cache again on `src/act/model` while the campaign's second arm is still to start, and
wiring it into the students' generator - would have spent an arm's provenance on a change whose own
measurement says it generates *different tokens*, i.e. it would have replaced a benchmark row with a
different one.

**What is corrected, and what replaced it.** `docs/training-throughput.md` now records Lever 2 as
closed: it carries the tempting ratio *and* the closure, names the mechanism, and points at
`tests/test_code_bench.py::test_the_students_window_is_recomputed_because_a_cache_cannot_reproduce_it`
as the owner of the measurement, so a future session meets the closure beside the arithmetic rather
than after rebuilding it. The summary table states that a cache would be a **different generator**
rather than a faster one, and the cache heads the note's list of rejected levers. The test that had
asserted the 299x claim now pins the ratio *and* the closure and the pointer, so neither can drift.
What replaced it is the part of Lever 1 that reaches the benchmark: the experts are **68.0%** of the
parameters, so grouping them removes **48.3% of every forward pass** - and since HumanEval is pure
forward work, the same 1.93x applies to each of the 208,516 token-forwards a student's 512-token
completion costs. That is the remaining speedup on the path to a benchmark number, and it needs no
protocol change.

**Recorded as a defect class, not an incident.** The note's own opening line is the reason this was
catchable at all: the repository's numbers about its speed already disagreed by 75x, and the
disagreement was recorded rather than smoothed over. The mistake here was the same shape in the
opposite direction - a made-to-look-good ratio quoted without checking the file that had already
measured it - so the fix is to keep the closure *adjacent to* the ratio in the same document.

## The dispatcher became a library facility, off by default, after the cache turned out closed (2026-09-25)

**Where this leaves the "build the KV cache" request.** The cache is closed (entry above), so the only
speedup still open on the path to a benchmark number is the one the same note prices: the dense expert
form executes 1.934x the arithmetic the routing contract books, that is 48.3% of every forward pass, and
because the experts are 68.0% of the parameters the saving reaches HumanEval - which is pure forward
work - without touching the generation protocol.

**What was built.** `act.model.dispatch`, with `grouped_forward` plus three switches:
`use_grouped(model)` rebinds each block's MoE in place and returns how many blocks it switched,
`use_dense(model)` removes the rebinding, and `dispatch_of(model)` reports `dense`, `grouped` or
`mixed:n/total` so a run can record what it actually executed. `scripts/moe_dispatch_benchmark.py` now
imports the dispatcher from the library instead of carrying its own copy, so the thing that is measured
is the thing that would be used.

**It is not on by default, and that is the design.** The switch is in-place and reversible rather than a
config field, because it does not change what a model *is* - so it must not appear in a checkpoint, in
a config, or in the architecture column of a benchmark row. Nothing in the package switches a model
over; a caller that has the measurement says so explicitly. The switch also refuses a module with no
`blocks`/`moe` rather than silently doing nothing, because a no-op would let a caller believe it had
measured a dispatcher it never ran.

**Pinned at the level a caller sees, not only at the layer.** The layer-level identity was already
measured at 5.1e-16; `tests/test_moe_dispatch.py` now also pins a whole model's logits in float64 under
`use_grouped` (and that `use_dense` restores the class's own implementation bit-for-bit), plus a seeded
greedy completion that produces **the same token ids** under either dispatch - which is the property a
benchmark row actually is, since a row is a string of sampled tokens. Eight tests in that module.

**What still has to happen before it is used anywhere.** The crossover between the forms has to be
measured at the code-LM shape on the card, which is busy until the arms and the benchmark rows are
done. Until then the campaign's rows keep the dense path they were registered with, and the wiring
point for the student's generator is `make_ours_generator` in `scripts/code_bench.py` - one call, once
there is a number that says which form to call. 1849 tests, 328/328 claims.

## The width mutation's named obstacle is two scalars, not a wall (2026-09-25)

**The note asserted; this measures.** `docs/width-control-design.md` said a width mutation needs a
function-preserving widening, that **RMSNorm is the obstacle**, and that the test should show
**bit-identity**. Both claims are now checked against an implementation, and one of them was wrong.

**What was built, and why it could land mid-campaign.** `src/act/model/widen.py` -
`widen_config` / `widen_model` / `widening_report` - with `tests/test_width_widen.py`. It is a *new
file*: `v0_moe.py` and `train_code_lm.py` are untouched, so the two arms' provenance ("no file they
import changed") still holds where an edit to the model would have broken it. The freeze recorded in
the design note still governs anything the arms read.

**The mechanism is zero extension, not the Net2Net duplication the note proposed.** The new
coordinates are zero, the new heads emit zero, every new input column and output row is zero, so
nothing is duplicated and there is no split factor to get wrong. The one term that cannot be zeroed
away is the normalisation, and it is exactly compensable - by **two** scalars, which is where the
first version was wrong:

| state of the compensation | logits against the un-widened model |
|---|---|
| neither half | **5.79e-4** |
| weight scaled by `sqrt(h/h')`, epsilon left alone | >100x the compensated error |
| epsilon scaled by `h/h'`, weight left alone | >100x the compensated error |
| both halves, at the student's shape (768 -> 1024, 12 -> 16 heads) | **2.50e-06 absolute, 9.7e-07 relative** |

The missed half is the epsilon, and the reason is structural: `_RMSNorm` computes
`x * rsqrt(mean(x^2) + eps) * w`, and the epsilon sits *inside* the root-mean-square. Widening makes
`mean(x'^2) = mean(x^2)·h/h'`, so what is left over is `rsqrt(a/h' + eps)` against `rsqrt(a/h + eps)`
for the token's own `a = ||x||^2` - an activation-dependent discrepancy that no constant weight can
cancel. Scaling the epsilon by `h/h'` makes the two radicands proportional for every input, and the
pair is then an identity. A norm guard of `1e-6` is not negligible against a fresh model's activation
scale, which is why the weight-only version was off by 5.79e-4 rather than by rounding.

**The contract is met, but the bar as drafted was unreachable and the draft was what was wrong.**
`_RMSNorm` computes its statistic in float32 **by design**, so a widened norm rounds in the last place
of that cast, and the residue does not improve in float64 (6.90e-08 there against 5.96e-08 on a toy,
which is what identifies the floor as the model's own normalisation rather than the graft). So the
contract is *"the same function to within the model's own normalisation rounding"*, adopted with its
reason stated, rather than by relaxing an assertion until it passed. The note and its sequenced actions
now say so, and the tests pin both halves of the compensation as load-bearing inequalities against the
compensated error rather than against magic constants.

**What a graft costs at the student's shape** - and the two numbers that matter are the second and
third columns, not the first:

| step | total parameters | active parameters |
|---|---|---|
| h768 -> h896 (12 -> 14 heads) | 166,423,296 -> 199,665,536 (**1.200x**) | 81,488,640 -> 100,575,104 (**1.234x**) |
| h768 -> h1024 (12 -> 16 heads) | 166,423,296 -> 234,480,640 (**1.409x**) | 81,488,640 -> 121,234,432 (**1.488x**) |
| h768 -> h1152 (12 -> 18 heads) | 166,423,296 -> 270,868,608 (**1.628x**) | 81,488,640 -> 143,466,624 (**1.761x**) |

The active cost runs ahead of the parameter cost at every step, which is the same asymmetry section 8
records from the other direction, and it is why a width schedule's win is ambiguous between capacity and
compute unless section 5's three controls are run.

**Two properties that a search operator has to respect.** The new coordinates start at **exactly zero
gradient** - the graft is a saddle, measured for three different routes to the loss (a norm entry, a new
head's qkv rows, an expert's new input columns) - so an operator built on this owes them a perturbation,
or the controller widens a model and nothing happens. And two things are **refused rather than
approximated**: a shrinking step (pruning is a different operator with a different information loss, and
one name for both is how a mutation budget gets spent on the wrong one) and a `gdn`/`hybrid` layer,
whose `head_dim x head_dim` recurrence state changes geometry when a width step adds heads.

**What this leaves the gate.** The admission bar's item (a) - "the preservation test exists and passes" -
is now met and is a measurement rather than a plan. What remains is the *operator* itself: a mutation
that widens a partly trained model, records the graft in the run's own signature, perturbs the new
coordinates, and is costed from committed figures - plus the freeze on whatever the arms import. 1856
tests, 328/328 claims.
