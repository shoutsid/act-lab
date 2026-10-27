# Campaign session logs - the evidence behind the exit-publish numbers

## The week's last hours are lost to the clock: a slower benchmark and a 35-minute restore refuse the pair (2026-10-27)

The 2026-10-27 check spent the week's last **2.67 h** on a fitted window aimed at **15033**, and
the session's own deadline rule refused it: the window was priced against the shape measured the
night before (pair 1.54 h, restore 8.3 min), but this session benchmarked **1.62 h/pair** and its
restore ran **35.3 min** - together ~32 minutes off the clock the pair needed. Null stays
**9/12**; 15033-15035 come from the 2026-10-31 refresh, which the relaunched launcher now waits
for.

| file | what it is | sha256 |
|---|---|---|
| `session-null-20261027T140229Z-live.log` | the driver's capture: the benchmark, the 35.3 min restore, the rule-1 refusal, the no-seed exit | `e89d5f0a91983c51ef262cb5f9818ec04a55d0d8e6d01c3f68bf766fa0c68136` |
| `session-null-20261027T140229Z-kernel-log.json` | the platform log: **94 events**, last at 2,635 s (0.73 h) - a clean exit | `ebd0e2eb04a401c20db44d1c0f1136a6220281f5823999b71e774c9478145b2c` |
| `session-null-20261027T140229Z-session-log.json` | the record: **0 pairs**, `remaining_hours_at_entry: 1.93`, pair_time 1.62 h, 4,489 tokens/s | `7507eddf3363ec5914e6b992868d967e55e44a0d7b6ba8eed41a68940450a580` |

| | |
|---|---|
| Declaration | `--queue null --hours 2.65 --seeds 15033-15035 --max-sessions 1` against the week's **2.67 h**; kernel **v5**, code **`3c04bba`** |
| Startup / benchmark | 4.1 min of startup, child window 2.58 h; **fp16 4,489.5 tokens/s -> pair 1.62 h** (the 17:16Z session: 4,731.5 -> 1.54 h) |
| Restore | **28.70 GiB in 35.3 min** (12.4 MiB/s; the same dataset restored 22.32 GiB in 8.3 min the night before), nine seeds skipped `complete` |
| The refusal | `seed 15033: remaining 1.92 h does not fit pair 1.62 h + 45 min reserve (rule 1)`; 0 pairs; `remote publish skipped at exit: no seed trained this session` |
| Cost | **0.73 GPU-h**; quota **2.67 -> 1.94 of 30 h**; driver exit **4** ("session completed but the queue did not advance (0 -> 0 of 12 seeds)") |

**What this says about the fit.** The margin that priced the fire was minutes: at the previous
night's shape (1.54 h pair, 8.3 min restore) the 2.65 h window fits with ~3 minutes to spare; at
this session's shape it is short by ~27 minutes. The window was already at the quota gate's
practical ceiling (2.65 h declared against 2.67 h shown), so the reading is not that a shorter
window or a longer one would have helped - the pair time is measured per session, and this
session's was slower.

**What this does not say.** That the fitted fire cost anything but doomed hours: the 2.67 h
would have expired unused at the refresh, and the attempt billed 0.73 h of them - but it did
spend those hours on a session that trained nothing. That the slow restore is a defect: it is one
measurement of the network (12.4 MiB/s against 46 MiB/s the night before), and the restore is
inside the session clock by design. That null is finished: 15033-15035 remain and the verdict
still withholds.

## The prune holds: both re-trained pairs publish and null reaches 9/12 (2026-10-25 17:16Z)

The fire after the 200 GiB wall was read re-trained the two pairs the bare `400` had discarded
and published both. A **6 h** window fitted to a **6.46 h** week: the startup restore paid
**22.32 GiB in 8.3 min**, skipped the seven published seeds, trained **15031** and **15032**,
and published **8 seeds / 25.5 GiB** then **9 / 28.7 GiB** - the publish path the wall had
refused, now landing. The session stopped by rule 1 with a third pair **4 minutes out of
reach** (2.22 h left against 1.54 h + 45 min reserve). Null is **9/12**; the last three seeds
need the **2026-10-31** refresh.

| file | what it is | sha256 |
|---|---|---|
| `session-null-20261025T171639Z-live.log` | the driver's capture: the fitted window, the restore, both pairs, both publishes, the rule-1 stop | `4bc5b07180c6b9476927d74c5bc82179c5a8a9c61dcae86f91cae916d392d919` |
| `session-null-20261025T171639Z-kernel-log.json` | the platform log: **122 events**, last at 13,647 s (3.79 h) - a clean exit, so the session's own account survived | `6aaf5037ebe8403ac95189d7cb4b46e15a187772c5cc139c04a93eecb4010755` |
| `session-null-20261025T171639Z-session-log.json` | the record: 2 pairs (15031, 15032), `latest_published_seed: 15032`, 4732 tokens/s, pair 1.54 h, handoff 28.70 GiB | `2436b7e80f4c9501bcad56a716c8041751233db69fbc893588a62e788873d45e` |

| | |
|---|---|
| Window / startup | declared **6 h** against the week's **6.46 h**; **4.9 min** of startup, child window 5.92 h |
| Restore | `sid123123234/act-v0-5-2-null` **22.32 GiB in 8.3 min**; restored 7 seed versions `[15024-15030]`, all seven skipped `complete` |
| Pairs | **15031** `null 10.398918`; **15032** `null 10.398030`; `search_flops 0` on both |
| Publishes | 8 seeds / **25.5 GiB** (`0ffa2402fc8e`), then 9 / **28.7 GiB** (`26382a082f6f`) - both `status: published` |
| Exit publish | skipped as `already_published` (9 seed version(s)); handoff **28.70 GiB** |
| Stop | `seed 15033: remaining 2.22 h does not fit pair 1.54 h + 45 min reserve (rule 1)` - four minutes short of a third pair |
| Cost | **3.79 GPU-h**; quota **6.46 -> 2.67 of 30 h** (refresh 2026-10-31) |
| Code | the driver republished `act-lab-code` from **`3c04bba`** - the revision carrying the quota-wall fix - and pushed kernel **v4** |

**What this says about the wall.** Two publishes of 25.5 and 28.7 GiB landed after the
superseded `act-v0-5-1` checkpoints were replaced by their summaries, so the prune put the null
queue's growth back inside the account's limit. It does not re-measure the limit: the total
after this version is a derived **~180.7 GiB** (the prune's 174.31 GiB, plus 28.70 landing where
22.32 sat), not a listing sum.

**What this does not say.** That a third pair was ever reachable in this window: at the
session's own 1.54 h pair time a third pair needed 2.29 h of clock and 2.22 h remained, so the
window ended one startup's worth of minutes short - the schedule's arithmetic, not a tuning
knob. That null is finished: three seeds (15033-15035) remain, the nine published sit within
**0.001710 nats** of `ln(32768) = 10.3972`, and the verdict still withholds.

## The null queue reaches 7/12, then the 200 GiB quota rejects three publishes (2026-10-24/25)

The 2026-10-24 refresh added six null seeds in two sessions and then lost three pairs to a
wall that was in no session log: the account's **200 GiB private-dataset quota**. The
17:38Z session published **15028-15030** and took null to **7/12**; the 00:13Z session
trained **15031-15033** and every `CreateDatasetVersion` answered **`400`**, nine uploads
in all. The latest versions summed to **199.73 GiB** before the second session, and the
next null seed is **3.19 GiB**: the 7-seed version landed at the wall, the 8-seed one would
have been **202.9 GiB**.

| file | what it is | sha256 |
|---|---|---|
| `session-null-20261024T173856Z-live.log` | 17:38Z: restore 12.76 GiB, three pairs, three publishes (5/6/7 seeds), `already_published` exit | `b5e016fda44983a41504a49cc554a1f35a8967afc96c1e12c23ec46a084955c0` |
| `session-null-20261024T173856Z-kernel-log.json` | its platform log, `finished after 4.92 h` | `1e179d099da545e583da610b8403504633b2e02c135b116e0462b7c50c01a168` |
| `session-null-20261024T173856Z-session-log.json` | its record: 3 pairs (15028-15030), 4974 tokens/s, pair 1.47 h, handoff 22.32 GiB | `c77b75f5adb8d12ae971806af08482d361ab3f9a34fdeed2ad6053ea5eed54e7` |
| `session-null-20261025T001322Z-live.log` | 00:13Z: restore 22.32 GiB, three pairs, nine failed uploads, failed exit publish | `0d2967c4952eef73a4aec897e84abc6c53e31f629e92d95bba2a66c3a7b66cb4` |
| `session-null-20261025T001322Z-kernel-log.json` | its platform log, `finished after 6.52 h` | `2467499a1bf9478fef0aac5a8f16497b16b6fa07ed228026a4bdb4b1d880c8c2` |
| `session-null-20261025T001322Z-session-log.json` | its record: 3 pairs (15031-15033), 4856 tokens/s, pair 1.51 h, handoff 31.89 GiB | `051d21aea1b878cfc3ce47c58ae49ab2d84769bfdd485f3fd0eb6390a064b54e` |

| | |
|---|---|
| **17:38Z (4.92 h)** | restore **12.76 GiB in 1.3 min**; skipped 15024-15027; trained **15028** `10.398210`, **15029** `10.398393`, **15030** `10.397880`; publishes **5 seeds in 4.8 min**, **6 in 5.3**, **7 in 6.3**, all `published`; exit `already_published`; handoff 22.32 GiB. |
| **00:13Z (6.52 h)** | restore **22.32 GiB in 2.7 min**; skipped 15024-15030; trained **15031** `10.398338`, **15032** `10.398277`, **15033** `10.398332`, `search_flops 0` on every pair; every publish **`400`**: 8 seeds / **25.5 GiB**, 9 / **28.7**, 10 / **31.9**, each retried 3 times (~7-9 min per upload), **nine uploads**; exit publish failed; handoff 31.89 GiB. |
| **The wall** | The ten latest versions summed to **199.73 GiB** (44.13 + 31.80 + 44.23 + 31.75 + 25.42 + 22.32 + 0.08), and the documented private-dataset limit is **200 GB**. The 7-seed version was the last size that fit; the next seed is 3.19 GiB. |
| **Why the log has no reason** | `create_version` let `requests` raise, so the record carries the status line but not the server's body. The retry loop then treated the 400 as transient: three re-uploads per pair, ~45 min of the session, and the body was still discarded. |
| **What was freed, and how** | `act-v0-5-1`'s latest version now holds **13 files / 43,582 B** - the four 15000-15003 manifests, reports and top-level summaries plus a README - and no checkpoints. Private storage fell to **174.31 GiB**, **25.69 GiB** of headroom; the four summaries still download. |

**What this says about the wall.** It is account-wide, not per dataset: the null dataset
was 22.32 GiB and its own next version was 25.5 GiB, yet the publish was refused because
of what the *other* datasets held. It is also not a version-history limit: if archived
versions counted, the account would have crossed 200 GiB long before, so the fit is latest
versions. The 17:38Z push landing at 199.73 GiB and the 00:13Z push failing at +3.19 GiB
is one observation at the boundary, not a measured threshold with two sides.

**What this does not say.** That the refusal *was* the quota, because the body was
discarded; the size sum and the documented limit are an inference. That the four discarded
checkpoint pairs were unneeded; the README beside the summaries says what was removed, and
an analysis that needs the weights would have to retrain them. That null is complete: it
is 7/12, and the verdict still withholds.

## Control finishes 12/12 at the wall's edge, and the null queue runs on the GPU (2026-10-24)

The 2026-10-24 refresh opened with both control shards incomplete and finished them by
06:08Z: shard 1 trained **15034-15035**, shard 0 trained **15030**. Two sessions that found
nothing to train followed (one per shard), each restoring its shard and exiting
`already_published`. The control queue is now **12 of 12** by its own published records
(`campaign_health --queue control`: 12 complete, `COMPLETE`). The null queue's first GPU
session then published four seeds, and a second was pushed from seed 15028 at 12:41Z.

| file | what it is | sha256 |
|---|---|---|
| `session-control-20261024T034940Z-live.log` | shard 1: restore 19.04 GiB, two pairs, the 25.4 and 31.8 GiB pushes, the `already_published` exit | `967835871ccc3e0ec3879222ef7cb2555c3859e475091cd2871b6f9be5e5f9a2` |
| `session-control-20261024T034940Z-kernel-log.json` | shard 1's platform log (10,320 B of `stdout`) - a clean exit, `finished after 3.61 h` | `e3e0ff392ee9a7f1d7c869fc4b2883c45a06f99610b78adb89b64f196ebfab8d` |
| `session-control-20261024T034940Z-session-log.json` | shard 1's record: 2 pairs, `latest_published_seed: 15035`, 4837 tokens/s, pair_time 1.51 h | `3b69bb123f557fe714f1a9f3b698790148f0cfb0933b6922b2cea9674e73e337` |
| `session-control-20261024T040504Z-live.log` | shard 1 no-op: restore 31.75 GiB, all five `skipped_complete`, exit reused | `473bf3563547b6a7c240c9ef4861a4a022d4fc087bc406343799c9cb31293820` |
| `session-control-20261024T040504Z-kernel-log.json` | shard 1 no-op's platform log, `finished after 0.20 h` | `2804c6a7d5d44ca80fe812bda4b4ab7a12c7c20dd7f920e5d2bca7e46c9258e2` |
| `session-control-20261024T040504Z-session-log.json` | shard 1 no-op's record: 0 pairs, all five `skipped_complete` | `4780e752a1295486023db45edbd544c2de62d7c66e5608bf3535563778956067` |
| `session-control-20261024T060857Z-live.log` | shard 0: restore 37.94 GiB, 15030's pair, the **44.2 GiB** push, the `already_published` exit | `d325fa4800e217029c505500dbb409719681c1c20d62da882d67bbee06fe0b34` |
| `session-control-20261024T060857Z-kernel-log.json` | shard 0's platform log (10,515 B of `stdout`), `finished after 2.01 h` | `42c76871a10b590a74b25e1f635a6e11565b4c3aa0961def88be66504a874101` |
| `session-control-20261024T060857Z-session-log.json` | shard 0's record: 1 pair (15030), `latest_published_seed: 15030`, 5349 tokens/s | `f69428e8b8621f0945f7a02498bf22a21a84c2ab181b457cf96dd505f8484c72` |
| `session-control-20261024T062809Z-live.log` | shard 0 no-op: restore 44.23 GiB, all seven `skipped_complete`, exit reused | `549399d08c7dbef0a504e80103d99ea6230b20aeddd4506157914fa411d00c7b` |
| `session-control-20261024T062809Z-kernel-log.json` | shard 0 no-op's platform log, `finished after 0.27 h` | `c9907d606908200649157ba08badd4b95eb7c5287a60f829b1d8962785e2d583` |
| `session-control-20261024T062809Z-session-log.json` | shard 0 no-op's record: 0 pairs, all seven `skipped_complete` | `0085af0df4ac1f4af3e6ff516b43e4a015bde881bbb5332341ee3c46db200ced` |
| `session-null-20261024T124102Z-live.log` | the null queue's first GPU session: four pairs, four growing publishes, the `already_published` exit | `b7fc3549843fc2e69987b6d27941531d1571367200f9f90f070700aeabde5044` |
| `session-null-20261024T124102Z-kernel-log.json` | the null session's platform log (9,650 B of `stdout`), `finished after 5.97 h` | `c0fe6abbd96a6d4769bd3ad2a81c73154c0d9edab69bfe6eb8810e85d98818b1` |
| `session-null-20261024T124102Z-session-log.json` | the null session's record: 4 pairs (15024-15027), `latest_published_seed: 15027`, `search_flops 0` on every pair | `3e5db2ce930ddaa9af5631948711f3dc0e857d4108b65e9a55dd9c40cbcf8a01` |

| | |
|---|---|
| **Shard 1 (03:49Z)** | restore **19.04 GiB** in 3.9 min; skipped 15031-15033; trained **15034** (`random 10.397990` / `adaptive 10.398016`, **+2.6e-5**) and **15035** (`random 10.397920` / `adaptive 10.397709`, **-2.1e-4**); pushes 4 seeds / **25.4 GiB** then 5 / **31.8 GiB** in 9.4 min; exit `already_published`; handoff 31.75 GiB; **3.61 h** |
| **Shard 0 (06:08Z)** | restore **37.94 GiB** in 8.0 min; skipped 15024-15029; trained **15030** (`random 10.398272` / `adaptive 10.397173`, **-1.099e-3**); push **7 seeds / 44.2 GiB in 14.8 min**, `published`; exit `already_published`; handoff 44.23 GiB; **2.01 h** |
| **The near-wall fire** | **37.94 restored + 6.4 trained + 44.23 staged = ~88.6 GiB**, against the **95.0 GiB** that went EROFS on 2026-10-19 - it landed, 6.4 GiB below the death. The 44.2 GiB version is the largest payload this record has published. This is the falsifier the 2026-10-22 entry named, resolved in favour of the shard rule; it is a measurement, not a threshold. |
| **The two no-ops** | 0.20 h and 0.27 h; each restored its shard (31.75 / 44.23 GiB) and trained nothing. The launcher's pre-fire completeness check could not yet see the seeds its previous session had just published, and its rule is that an unreadable check must not skip work - so it fired a session that found everything complete. 0.47 GPU-h, no lost pairs. |
| **Control readout** | 12 summaries read: adaptive won **7 of 12**, mean `adaptive - random` **-0.000092 nats**, largest gap from `ln(32768) = 10.3972` **0.001281**. The 9-seed readout was +0.000020; the last three moved it down 1.1e-4. Same floor, still not a verdict. |
| **Null, first GPU session** | 5.97 h, **4/12**: 15024-15027 at `10.398030` / `10.398226` / `10.398335` / `10.398767`; no restore (`no remote version ... training from scratch`); publishes 1 / **3.2 GiB**, 2 / **6.4**, 3 / **9.6**, 4 / **12.8** (4.0 min), all `published`; exit `already_published`; handoff 12.76 GiB; measured **5326 tokens/s**, pair 1.38 h, `search_flops 0` on every pair. |
| **Cost** | control **6.09 h** + null **5.97 h** = **12.06 h**; quota **17.84 / 30 h** at 12:41Z, refresh 2026-10-31. |

**What this says about the wall.** The 2026-10-19 death was at 95.0 GiB and the projection
for 15030's fire was ~88 GiB if the restore's scratch is released before staging. It was, and
the push landed. That does not establish where the ceiling is - the bracket is still
(88.6 survived, 95.0 died) - but it removes the reading that the 44.2 GiB payload is itself
fatal: the 2026-10-19 session died staging **25.5 GiB** into p2 at a **95.0 GiB** total,
while this one staged **44.23 GiB** at **~88.6 GiB** and landed. Total footprint, not payload
size, is what the two observations have in common.

**One observation worth carrying to the verdict.** Seeds 15033 and 15035 publish the same
adaptive final loss to the last bit (`10.397709369659424`) while their random arms differ.
Each summary's own `config.seed` matches its seed, so this is convergence at the floor, not
a mislabeled record - but it is the kind of coincidence the readout should name rather than
smooth over.

## The empty shard paid off: zero restore, three seeds, every publish landed (2026-10-21 17:35Z)

Kernel **v3** ran the control queue's third session on `--seeds 15031-15035`, i.e. into
`act-v0-5-2-p2-control`, which had **no version**. The startup restore therefore downloaded
nothing (`no remote version of sid123123234/act-v0-5-2-control-p2 available yet; training from
scratch`), against the **18.99 GiB in 2.3 min** the same declaration would have paid to reach
15030 through the seven-seed shard. It trained **15031-15033** and published each pair; the
first version this shard has ever carried was written by this session.

| file | what it is | sha256 |
|---|---|---|
| `session-control-20261021T173548Z-live.log` | the driver's capture: the restore line, the three pairs, all three pair-time pushes, the rule-1 stop with 1.44 h left, and the `already_published` exit | `da9c018030a40265edaef3ed3de044efffcf0e25fe47c209a31834291ad58e39` |
| `session-control-20261021T173548Z-kernel-log.json` | the platform log (15,889 B of `stdout` events) - a clean exit, so the session's own account survived | `dd3d7d15d89b187b0d97f8f22e24937c0795b840925b63ba69c6e9a91d88db1c` |
| `session-control-20261021T173548Z-session-log.json` | the record: 3 pairs, `latest_published_seed: 15033`, `4776 tokens/s`, code revision `2b4a647` | `a7c9a937d55711a6cc239610e93e8ea96ed10ba23625cd339bc6ad99b10a7f8e` |

| | |
|---|---|
| Window / startup | declared 6.5 h, **4.3 min** of startup, run **5.06 h**; stopped by rule 1 with **1.44 h** left (a fourth pair needs 1.53 h + 45 min reserve) |
| Restore | **none** - the shard had no version; the whole session's handoff stayed at the 19.04 GiB it built, not 19 GiB restored plus 19 GiB built |
| Pairs | 15031 `random 10.398030` / `adaptive 10.397951` (**-7.9e-5**); 15032 `random 10.397685` / `adaptive 10.398204` (**+5.19e-4**); 15033 `random 10.398005` / `adaptive 10.397709` (**-2.96e-4**) |
| Pair-time pushes | 1 seed / **6.4 GiB** in 3.7 min; 2 / **12.7 GiB** in 3.9 min; 3 / **19.0 GiB** in 5.8 min - all `status: published` |
| Exit publish | **skipped** as `already_published` at fingerprint `bca4b35a3e5a`; the 19.04 GiB that was staged is the shard, and it was already up |
| Cost | **5.06 GPU-h**: quota **23.36 -> 1.57 of 30 h** (refresh 2026-10-24) |

**What this measures, and what it does not.** The scheduling claim the 2026-10-21 fire was made on
holds: a declaration that resolves to an empty shard restores nothing, publishes into a new small
shard, and never approaches the 95 GiB footprint that went read-only on 2026-10-19. What it does
*not* do is finish the queue: the 6.5 h window fits three whole pairs, so 15034-15035 were never
started (`seed 15034: remaining 1.44 h does not fit pair 1.53 h + 45 min reserve (rule 1)`), and
**15030** - the seventh seed of the other shard - was not in this declaration at all.

**The queue is now 9 of 12**: 15024-15029 in `act-v0-5-2-control`, 15031-15033 in
`act-v0-5-2-p2-control`. `campaign_health --queue control --readout`: adaptive won **5 of 9**,
mean `adaptive - random` **+0.000020 nats**, every arm within **0.001281 nats** of
`ln(32768) = 10.3972` - still the uniform floor, still not a verdict.

**Next, in the order the registration puts it.**

1. **15034-15035**, which the waiting driver (pid 147498, `--queue control --hours 6.5 --seeds
   15031-15035 --wait-for-quota`) fires at the 2026-10-24 refresh. Its declaration now resolves to
   shard 1 *with a version*, so that session restores ~19.0 GiB, skips the three this one
   published, and trains two pairs (~3.1 h) before pushing five seeds (~31.8 GiB).
2. **15030**, which needs `--seeds 15024-15030` of its own: it is the seventh seed of shard 0, so
   publishing it restores the six-seed shard (**~38 GiB**) and re-uploads all seven (**~44 GiB**).
   With the seed it trains (~6.4 GiB) that session's local footprint is ~**88 GiB**, close to the
   95 GiB that went EROFS - the one remaining control seed that carries the wall's shape.
3. The **null queue** (§2c), 0 of 12 and never fired on the GPU.

**One waste this state names before it happens.** Once 15034-15035 land, every seed in the
waiting driver's declaration is complete, so the session after that can only skip: it restores
~19.0 GiB and produces nothing, and the driver reports the no-progress exit rather than looping.
That is the exit code doing its job; the next fire is the narrowed `--seeds 15024-15030` above,
not another push of the same command.

## The control queue's first two sessions, and the shard-scoped fire that spends the last hours (2026-10-20/21)

The control queue (§2b) is at **6 of 12** after two sessions of the declared block
`15024-15030`, which is the whole of `act-v0-5-2-control` - the seven-seed shard. Both sessions
restored that shard (**18.99 GiB in 2.3 min**, three seed versions and then six) and trained
three pairs each; every pair-time and exit publish landed, and both exits were skipped as
`already_published` because the per-pair publish had already carried the state. Peak local bytes
never approached the 2026-10-19 shape: ~19 GiB restored, ~19 GiB staged.

| file | what it is | sha256 |
|---|---|---|
| `session-control-20261020T223031Z-live.log` | session 1's live capture, 3 pairs `15024-15026`, `pair_time_hours: 1.52`, `4802 tokens/s` | `f22e3a340057d3f0195c154bf6f7833cca3aca660f5ae317ed2c3dcdfe40abba` |
| `session-control-20261020T223031Z-kernel-log.json` | session 1's platform log (the child's streamed stdout) | `8ee612a7de968ebce60373bd0e49683785fbb4725f31176e7ca3be0a39fff4e2` |
| `session-control-20261020T223031Z-session-log.json` | session 1's record: 3 pairs, `latest_published_seed: 15026` | `8084a01a88121ac0e430edaf2f65aeabd5107632bd5bf2f3d0dc59ea4e8e42f2` |
| `session-control-20261021T033721Z-live.log` | session 2's live capture: three `skipped_complete` lines, then `15027-15029`; publishes at 4, 5 and 6 seeds, exit `already_published` | `ab1b78b20f7841828e73301796e01aa9a011c442d18f4409fec4b4bd79a2c0a3` |
| `session-control-20261021T033721Z-kernel-log.json` | session 2's platform log | `db20d17b1133008c4705e42dfb7787161e9776aaa1888a5c1b38e86dcb6f666e` |
| `session-control-20261021T033721Z-session-log.json` | session 2's record: 3 pairs, `latest_published_seed: 15029`, `5368 tokens/s` | `c785945222257e05089529ba2c58f30689be39b603ae610b6affb3e82d3ba45c` |

| | |
|---|---|
| Published | **6 of 12**: 15024-15029, all in `act-v0-5-2-control`; 15030 in the same shard is still missing, and 15031-15035 live in `act-v0-5-2-p2-control`, which has **no version** |
| Readout | adaptive won **3 of 6**, mean adaptive-random **+6e-6 nats**, both arms within **0.001281 nats** of `ln(32768)`, the uniform floor - the same picture as the primary block's 12 |
| Cost | quota **16.71 -> 23.36 of 30 h** across the two sessions (refresh 2026-10-24); ~3.3 GPU-h each |

**Why the next fire declares five seeds and not seven.** `campaign_health --queue control`
reported `STALLED` with **6.64 h** left and no lock: the week cannot fit the registered 7 h
window, and hours left at the refresh are lost. Two schedulings fit the window:

- `15024-15030` (what the two sessions above declared) restores the seven-seed shard (**~38 GiB
  at six published seeds**), trains 15030, and its push re-uploads the whole shard as a
  **~44 GiB** version - the largest that has landed, and the shape that has already gone
  read-only when combined with staging of the same size.
- `15031-15035` resolves to the *empty* `act-v0-5-2-p2-control`, so the startup restore is
  **0 GiB**, and the publish writes a new, small shard. It is the cheapest session the queue
  can run, and it leaves 15030 as a single-seed session of its own later.

So kernel **v3** went out as `--hours 6.5 --seeds 15031-15035` (fitted window, precedent
2026-09-27; `--wait-for-quota` so the 2026-10-24 refresh is spent without an operator). It ran
5.06 h and returned **COMPLETE** with three pairs - see the leading section for what it measured.
15030 remains the queue's one expensive seed: it can only be published by restoring the
seven-seed shard and re-uploading it whole.

## The fire that finished the block: the narrowed declaration held (2026-10-20)

The primary queue is **12 of 12** and the kernel is `COMPLETE`. The finishing session declared
`--seeds 15034-15035`, i.e. exactly the block still missing, and that declaration is what scoped
the startup restore: **shard 1 only** (`act-v0-5-2-p2`, 19.09 GiB in 19.2 min, three versions
`[15031, 15032, 15033]`), with the 44.13 GiB shard 0 never requested. The 2026-10-19 entry
below predicted a ~64 GiB peak for this shape against the 95.0 GiB that died; the session
survived both pair-time pushes and its own exit.

| file | what it is | sha256 |
|---|---|---|
| `session-session-20261020T150118Z-live.log` | the driver's live capture, 187 lines, including both pair-time pushes and the skipped exit publish | `2b1aa3346cbf2ac781736a510f1af97b633bc9eca07b84c4cab5cf1a992af52b` |
| `session-session-20261020T150118Z-kernel-log.json` | the child's full streamed output (`stdout` events): **21,905 bytes of real log**, where every killed session's was empty (`[]`, 3 B) | `cce495be598bf603303f59ded619d70f5608c7105444562fc050675f6096e994` |
| `session-session-20261020T150118Z-session-log.json` | the session record: 2 pairs, `latest_published_seed: 15035`, code revision `d12c3ca` | `f9429ad25d12f0804e55232a883210771a3062498bb0d0c8b35c7543ba778a43` |

| | |
|---|---|
| Window / startup | 6.89 h of training after 21.9 min of startup; the session ended after **4.79 h**, 2.21 h inside its clock |
| Restore | **shard 1 only**: `19.09 GiB on disk` in 19.2 min. No shard-0 download at all - the difference between this fire and the one that died (which restored 44.13 + 19.09) |
| Pairs | 15034 `random 10.397485` / `adaptive 10.397881` (**+3.95e-4**); 15035 `adaptive 10.397495` / `random 10.397844` (**-3.50e-4**) - both arms on the uniform floor, as at this fidelity |
| Pair-time push 1 | 4 seeds / **25.5 GiB** into `act-v0-5-2-p2` in 24.1 min, `status: published` |
| Pair-time push 2 | 5 seeds / **31.8 GiB** in 30.3 min, `status: published` (fingerprint `46069823f084`) |
| Exit publish | **skipped** - `already_published` at the same fingerprint, so the 44 GiB shard-0 re-upload that killed the 2026-10-19 04:10Z fire could not happen twice |
| Peak local bytes | 19.09 (restore) + 2 x 6.4 (both seeds, incl. finals) + 31.8 (staging) ~= **64 GiB**, the predicted shape, against the 95.0 GiB that went read-only |
| Cost | **4.79 GPU-h** of the week; quota after it **16.71 of 30 h** (refresh 2026-10-24) |

The clean exit is itself a check: the platform log exists and carries the child's stdout. A kill
does not flush it, which is why every earlier entry in this file is a driver capture plus silence.

## The fire that did not hit a size wall: the container's disk went read-only (2026-10-19 15:28Z)

Kernel **v21**, the first session driven after the exit-publish fix (`6401c0d`), restored
both shards, skipped all ten published seeds, trained **15034**, and then failed its
**pair-time** push to `act-v0-5-2-p2` with
`OSError: [Errno 30] Read-only file system` - three times, inside the staging directory -
and was killed. The kernel log is empty (`[]`, 3 B, the same hash as every killed
session), so the driver's live capture was again the only near-real-time account. This
time the session itself survived long enough to flush: `kaggle kernels output` still
answered and carried its **own** stream log and the summary for seed 15034, so the cause
is in the child's words rather than in an inference from silence.

| file | what it is | sha256 |
|---|---|---|
| `session-session-20261019T180432Z-live.log` | the driver's live capture, 96 lines, ending 4 min into the pair-time push | `b311d7834502bde9708ee0d67fd257f1444336de2829f48ac174ad98af6fb0bd` |
| `session-session-20261019T180432Z-stream.log` | the child's own tee (`session-stream.log`), 80 lines, ending on the traceback: **this is the account the live capture does not have** | `027c85ace6d1f77f38de895ca3f2cb11a8adb22dfa3a47a94eb65ea4b36b4269` |
| `session-session-20261019T180432Z-kernel-log.json` | the platform log: **zero events** (the kill did not flush the wrapper's stdout either) | `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570` |
| `session-session-20261019T180432Z-seed-15034-summary.json` | 15034's own record, recovered from the kernel output: both arms, both final losses, the three mutations, the treatment. What did **not** survive is the 6.4 GiB of checkpoints - so the seed has to be trained again | `7438db6d1829c7a8f5cb1fa6f7c59020d62a30e86364b206a1453d99e74c157e` |

| | |
|---|---|
| Window / startup | 6.93 h declared; **16.5 min** to restore shard 0 (44.13 GiB) and **6.8 min** for shard 1 (19.09 GiB), 32.5 min of startup in total |
| What it trained | seed **15034** only (`random 10.397681`, `adaptive 10.397773`, a 9.2e-05 difference on the uniform floor); 15035 then failed before its first step |
| What it pushed | nothing to p2; the pair-time push of **4 seeds / 25.5 GiB** began at ~17:35Z and the container died inside it |
| Quota | **2.45 GPU-h** spent, 21.5 of 30 remaining (refresh 2026-10-24); the driver reported the session ERROR after 2.62 h |
| Local bytes at the push | **95.0 GiB**: 44.13 (shard 0) + 19.09 (shard 1) + 6.4 (15034) + 25.5 (the staging tar the upload writes) |

The failing calls are `mkdir` on the staging dirs (`/tmp/act-kaggle-push-*/push`), not the
upload itself - `tmpfs`-style **ENOSPC does not appear anywhere; the filesystem was
remounted read-only** after the write failed, and the next action (`seed 15035: starting
pair`) failed writing `artifacts/v0-sessions/seed-15035`, then the session log. So the
container lost its writable layer, and the push was the write that took it there.

**This is not the 44 -> 50 GiB dataset-version wall** the 2026-10-05/06 fires were read
as. The payload that killed this container (25.5 GiB) is smaller than the one that landed
there (44.13 GiB), and the difference is the *local* footprint, which those fires also had:
v18's 7-seed push staged 44.13 GiB on top of ~44 GiB of handoff (88 GiB) and landed; its
8-seed push needed ~101 GiB and died here. Every death to date is consistent with a
container writable layer that gives out somewhere around **90-100 GiB**, and none of them
requires a per-version upload limit. The bracket is recorded as the observation it is;
the *cause* is now the child's `EROFS`, not silence.

**The cost that made this certain.** Shard 0 is 44.13 GiB that this session could only ever
read: all seven of its seeds were complete, and the exit fix (`6401c0d`) had already made
sure they would never be re-pushed. It was restored anyway, because the run declared
`--seeds 15024-15035` and the restore is scoped to the shards those seeds resolve to. The
CLI already supports a narrower block - `wanted_shards` intersects the requested seeds with
the shard map, and the treatment check reads whatever published summaries are present - so
the resuming session declares only the seeds it can still write and restores **19.1 GiB**
instead of 63.2, putting its peak at ~64 GiB with both remaining seeds trained.

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
