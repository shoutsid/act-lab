# Sharding the checkpoint handoff — a design note, before the next fire

**Status: design and implementation; not yet exercised on Kaggle.** Nothing here is a
measurement except where it says so. It exists because the publish path has a wall that
another session cannot get past: the block's handoff is pushed as **one** dataset version, and
the measured payload bracket says the whole block does not fit. The rule below was written
before the change and before the next fire, because the one thing that must not be decided
after seeing the result is which seeds go where. Section 8 records what was implemented
against it on 2026-10-18; no push has run the sharded path yet.

## 1. The gap

Measured 2026-10-05/06 (see `docs/IMPLEMENTATION_STATUS.md`, 2026-10-18):

| push | payload | outcome |
|---|---|---|
| 7 seeds, v18 2026-10-05 22:17Z | **44.13 GiB** (35 files) | **landed**, 16.9 min |
| 8 seeds, v18 2026-10-06 00:14Z | ~50.4 GiB | container killed inside the push |
| 8 seeds, v19 2026-10-06 12:54Z | ~50.4 GiB | container killed inside the push |
| 9 seeds, v15-v17 (2026-10-04/05) | ~56.7 GiB | container killed inside the exit push |

No Kaggle document in this tree names a per-version size limit, so the bracket is the evidence
and a cause is not claimed. What the bracket is enough for is arithmetic: the registered block
is 12 seeds x ~6.3 GiB = **~75.6 GiB**, so under a single-version publish seed 8 can never land
no matter how many sessions are fired. Six sessions have already trained seed 15031 and all six
pairs were discarded at that push.

## 2. The rule, pre-registered

1. **Membership is a contiguous slice of the declared queue, in declared order.** For the
   primary queue: **shard 0 = 15024-15030** (the seven seeds already published) and
   **shard 1 = 15031-15035** (five seeds, ~31.5 GiB at the measured 6.30 GiB/seed). The boundary is set from the state that
   exists today - shard 0 is exactly the version that landed - not from any future result.
   (The first draft of this rule said six seeds; it was corrected before any code shipped,
   for the migration reason in section 4. No data was seen in between.)
2. **A seed belongs to exactly one shard**, by its index in `DECLARED_SEEDS` and the declared
   shard size. Overlap is forbidden; the restore merge has no rule for two versions of the
   same seed and should never need one.
3. **Naming is derived, not chosen.** Shard 0 keeps the registered id
   `sid123123234/act-v0-5-2`. Shard *k > 0* is `<base>-p<k+1>`, e.g.
   `sid123123234/act-v0-5-2-p2`. The name is a function of the queue and the shard index, so a
   reader can compute it from `kaggle_t4.QUEUES`.
4. **A push stages only its own shard.** `stage_push` takes an include filter over the seed
   directories, so no version ever contains both shards - a version that did would hit the wall
   again and would duplicate a seed across datasets.
5. **Restore downloads every shard that exists, into the same local root, in shard order,**
   and the existing merge rule applies unchanged (a remote seed replaces a matching local one;
   local-only seeds are kept). A missing continuation shard is bootstrap, not failure: the
   existence check is the already-implemented `KaggleDatasetClient.dataset_exists`, which reads
   the owner's own listing rather than trusting a 403.
6. **The skip rule and the verdict do not change.** They read the merged local root and a
   seed's own published summary; neither learns that shards exist.

The shard size is declared **per queue** beside its dataset id (`shard_seeds` in
`kaggle_t4.QUEUES`), because the queues do not have the same payload per seed: the null queue
is single-arm (~3.2 GiB/seed, ~38 GiB for twelve) and does not need a second shard at all. A
count in the queue config keeps the rule deterministic; deriving shards from *measured* sizes
would make membership depend on the run that produced the measurement, which is the kind of
data-dependent grouping this project does not do.

## 3. Call sites, and what each one learns

| site | today | with shards |
|---|---|---|
| `act.ops.kaggle_remote.stage_push` | archives every seed dir | takes the shard's seed set (one filter argument, used by both the archive loop and the flat-file copies) |
| `act.ops.kaggle_remote.publish_handoff` | one folder -> one dataset | takes the shard's dataset id and seed set; the fingerprint is over that shard's manifests, so reuse is per shard |
| `remote-publish-log.json` | one `fingerprint`/`status` for the folder | top-level stays the **last** push (the driver reads it); add `shards: {dataset: {fingerprint, status, seeds_published}}` so a second shard's success is not overwritten by the first's, and a log without `shards` is read as the single-shard case |
| `act.training.cli._remote_publisher` | pushes the whole folder after each pair | resolves the completed seed's shard and pushes that shard only |
| `act.training.cli._publish_remote` (exit) | one push, normally `already_published` | for each shard whose recorded fingerprint is stale, one push; normally none, because the pair-time pushes already did it |
| `act.training.cli` / `kaggle_t4.mode_verdict` / `mode_restoreprobe` | `restore_latest(root, dataset=...)` | iterate the queue's shard list, stopping at the first that does not exist |
| `scripts/campaign_health.published_seeds` | lists one dataset | sums the queue's shards; a seed counted twice is a bug the rule forbids, so the counter asserts disjointness |
| `scripts/tpu_null_queue.py` | one dataset (`act-v0-5-2-null`) | unchanged: its declared `shard_seeds` covers the whole queue |

## 4. Migration of the published state

None. Shard 0 is defined as **the version that is already published** (15024-15030, 44.13 GiB);
the next push is shard 1 alone. The alternative - a 6/6 split - was rejected for two reasons: it
would need 15030 re-pushed into a second dataset (or left in both, violating rule 2), and it
would move a seed that is already a completed, published record. The choice is recorded here
before the next fire so it cannot be read as fitting the boundary to what succeeded.

## 5. Cost

- Shard 1's final push: 5 x ~6.30 GiB = **~31.5 GiB**, ~12.1 min at the 16.9 min/44.13 GiB
  rate the 7-seed push measured - 62% of the 50.4 GiB payload that kills the container.
- Per-pair pushes inside shard 1 grow 6.3 -> 31.5 GiB; the cumulative re-upload over the five
  seeds is ~94 GiB, against ~190 GiB for the same five inside a full 12-seed version. The
  per-pair trade itself is unchanged: each pair's push buys that pair's durability.
- Restore grows by one download of at most 31.5 GiB in a session that starts from shard 1;
  the restore is already priced by the session clock and logged with its duration.

Shard 1 is below the **only** push that has landed in both readings the bracket permits:
44.13 GiB / 16.9 min. So the rule does not have to bet on whether the killer is payload
size or elapsed upload time - 31.5 GiB / ~12 min is smaller than the known-good push on
either axis.

## 6. What would falsify this

- A **31.5 GiB push that dies** would falsify the size reading: the wall would not be size, and
  the next question is the upload path rather than the payload.
- A **restore that skips a seed present in shard 1** would mean the merged root was read before
  the shard downloads finished; the ordering is the contract, not an implementation detail.
- A **health count that double-counts a seed** would mean the shard rule leaked overlap;
  disjointness is asserted rather than trusted.

## 7. Open questions to settle before implementing

1. Whether the driver and the verdict should enumerate shards by the declared list or by
   listing the owner's datasets for the `<base>-p*` prefix. The declared list is deterministic
   and is the proposal; the prefix listing would answer "did an unexpected shard appear", which
   nothing currently needs.
2. What a deleted continuation shard should do to the skip decision. The proposal is to treat
   it exactly like a missing dataset today: the seeds it carried are trained again, and the
   restore's log names the shard it could not find rather than the seeds it did not see.
3. Whether the per-shard log entry should carry the shard's own payload size. Yes if the push
   heartbeat lands first (it does, 2026-10-18): the size is the number that decides when the
   next shard is needed, and it should be recorded rather than recomputed by a reader.

## 8. What was implemented (2026-10-18)

- `act.ops.kaggle_remote`: `shard_index` / `shard_members` / `shard_datasets` /
  `shard_for_seed` / `shard_dataset`; `stage_push(seeds=...)` and
  `publish_handoff(seeds=...)` so a version carries one shard; a per-dataset entry under
  `shards` in `remote-publish-log.json` (the top level stays the last push the driver
  reads); `restore_latest(datasets=[...])` merging shards in order and stopping at the first
  missing one. A `seeds=None` call is byte-for-byte the old single-dataset behaviour.
- `act.training.cli`: `--kaggle-shard-seeds` (default **0** = one version); the per-pair hook
  resolves the completed seed's shard from `DECLARED_SEEDS`; the exit path pushes each
  requested shard and stops on the first failure so the driver reads that failure.
- `scripts/kaggle_t4.py`: `Queue.shard_seeds` (7 paired / 0 null), `Queue.shards` and
  `Queue.datasets`, the flag in `session_argv`, and shard-aware restore in the verdict and
  restore probes. `scripts/campaign_health.py`: `published_seeds` merges the queue's
  datasets and refuses overlap; per-seed summary fetches resolve the seed's shard.
- `docs/T4_RUNBOOK.md` prints the flag in all three launch commands.

**What is still not established:** that a 31.5 GiB push lands. That is the next fire's
measurement, and the falsifier in section 6 is what it would mean if it did not.

## 9. The live probe the code path got (2026-10-18)

Before this landed, the sharded path was exercised against the real service at
kilobyte scale - a handoff with two fake seeds (9001, 9002), one shard each, published as
two **new** private datasets:

```
sid123123234/act-shard-probe      seed 9001 only
sid123123234/act-shard-probe-p2   seed 9002 only
```

What it established, on the live API rather than a fake client: the first-version bootstrap
(`dataset_create_new` after the 403) works for both ids; a `seeds=[...]` push stages and
uploads **only** that shard - the service listings carry `seed-9001` in one dataset and
`seed-9002` in the other, not both; a restore of `datasets=[both]` into a fresh root
returns `[9001, 9002]` with both payloads intact; and after the second shard pushed, a
repeat of the first returns `already_published` from the per-shard fingerprint, which is
the property that stops the next session re-uploading forty gigabytes. The publish log
carried both shards' records with the last push at the top level, as the driver reads it.

The CLI path was then run end to end against the same service - `--mode pair-session
--profile --seeds 15024-15025 --kaggle-shard-seeds 7 --kaggle-dataset sid123123234/act-shard-cli-probe`
- which trained two development-fidelity pairs and: bootstrapped from a dataset that did
not exist, published **1** then **2** seeds after the two pairs (the shard filter, not the
whole handoff), then returned `already_published` on the exit publish. The per-pair
`remote_publish` entries were in the session log, as the driver reads them. That probe
dataset was **deleted** afterwards (`kaggle datasets delete ... -y`): it carried
development-fidelity records for the *declared* seeds 15024/15025, and leaving those in the
account beside the real block would be a provenance hazard. The 9001/9002 datasets carry
no declared seed and are left in place.

What it does not establish is the thing the next fire is for: a **31.5 GiB** push. The
probe was kilobytes, so it says the shard mechanics are right, not that the 50 GB wall
has been cleared.

One untested assumption remains, and it is worth naming rather than discovering: both
probes created their continuation dataset from the **workstation**, while the kernel's
first shard-1 push is a `dataset_create_new` from inside a Kaggle kernel. If the platform
refuses that, the push is caught and recorded like any other failed push (the container is
not killed), but the session's pairs are not durable and the driver stops at the exit
publish. The failure would be visible in the pair's `remote_publish` entry; the fix is to
pre-create the continuation from a workstation and re-fire. This is not pre-done here
because it would make the first fire test less of the path than it can.
