# TPU session rehearsal - the evidence behind the 2026-09-28 pass

## The second fire ran the whole path (kernel v2, 2026-09-28)

`sid123123234/act-tpu-session-rehearsal` **version 2**, pushed 2026-09-28T08:46:52Z from the code
dataset republished at **`f3ed48f37ab11ff64ca5884c073bdb069f0884d8`**, reached
`KernelWorkerStatus.COMPLETE`, and the watcher fetched it at **19:36:28Z** - so it waited
**~10.8 h** in the queue and then ran in **~2.5 min**. All four phases are in `completed` and
`errors` is empty.

| | |
|---|---|
| **The phases, in order and in seconds** | From the platform log's own `time` field (seconds from kernel start): `environment` **23.2 s**, `session` **45.9 s**, `resume` **1.6 s**, `handoff` **1.3 s**, after a 60.3 s bootstrap that installs no `torch_xla` (it is already present). The record is printed at 132.2 s and the notebook ends at 150.9 s. |
| **What the session phase did** | One **null** arm, 20 development-fidelity steps on `xla:0`, checkpoint every 10: `wall_time_seconds` **40.70**, `tokens_per_second` **2,012.5**, `mutation_count` 0, `search_flops` 0, validation loss **9.066792488098145**. Both checkpoints are listed by the record itself: 179,313,571 and 179,313,635 bytes. |
| **The defect v1 found, confirmed fixed** | v1's session phase was three hours of silence and then `Killed`; v2's is **45.9 s** and returns. The same twenty dev-fidelity steps take **24 s on a CPU**, so the unmarked step was the whole difference - and this is the first measurement that `runtime.step_marker` does what it was written to do on the backend that needs it. |
| **The reading a killed run could not give** | `watch.session` has two samples: **0.0 s** at **7,045,312 kB** RSS and **30.0 s** at **11,986,972 kB**. Rising over a phase that *finished* is what working looks like; v1 ran to the kill with no samples at all. `device_bytes_used` is **null** in every sample - and in every `tpu-probe` record back to v4 - so this runtime does not expose it and the host's resident set is the growth signal that carries the watch. It worked, which is the difference between an instrument and an assumption. |
| **The resume the deadline stop depends on** | `resume.resumed_from_a_checkpoint` **true**, `same_loss_as_the_first_run` **true**, validation loss **9.066792488098145** - bit-identical to the fresh run. `session.checkpoint_tensor_devices` is **`["cpu"]`**, so `_cpu_payload` is doing its job on a lazy backend: the payload the TPU wrote is a host payload. `session.checkpoint_revision` and `manifest_code_revision` are both **`f3ed48f37ab1...`**, so a checkpoint separated from its manifest now says what wrote it. |
| **The §5.14 fields, read back off a real publish** | `handoff.null_arm` is `{final_loss 9.066792488098145, mutation_count 0, search_flops 0}` and `handoff.summary_status` is `complete`, published into a temporary dataset root with its manifest and checksums - which is what `v0_verdict`'s do-nothing clause reads through `--null-root`. |
| **What it does not establish** | The record's own `not_measured`: per-core parallelism (the probe's headline, not this), declared-fidelity throughput (the dev shape's rate is a path measurement, not a schedule input), and the two-arm pair placement. It is `rehearsal: true`, `declared_scale: false` - **not a port, not a Gate-4 result**, and no queue moved on it. |

## The second defect this record found: a resumed arm published an impossible rate

`seed-summary.json` here - the file §5.14's clause consumes - carries, for the resumed arm,
`tokens_per_second` **6871947673.6** and `wall_time_seconds` **1.1920928955078125e-05**. Both are
the same arithmetic defect: `_state_from_payload` restores `training.tokens` **cumulatively**
(81,920 = 20 steps x 8 x 512), while `wall_time` measured only the segment this process ran. The
resume phase loads a checkpoint at step 20 of 20, so the loop's `range(21, 21)` is empty, the wall
is a bare pair of `time.time()` reads, and the published rate is 6.87e9 against the **2,013** the
same arm had just measured.

That is not only the zero-step case. Any partial resume divides the checkpoint's whole token count
by one segment's seconds, and a *partial* resume is exactly what a deadline stop produces
(`V0SessionInterrupted` at a checkpoint boundary, then the queue resumes from it) - so the null
queue this record was rehearsing for would have published inflated rates on every seed that
survived a stop.

**Fixed after the fetch** (this record carries the pre-fix value): `_SessionState.wall_seconds` is
the arm's own training wall, stamped into `v0_session.wall_seconds` with **every** checkpoint
before it is written, restored on resume and summed with the segment's own - so a resumed arm
reports the rate of the whole arm (81920 / 40.70 = **2,013**) and a zero-step resume reports that
same number rather than an empty loop's. `tests/test_v0_session.py::
test_a_resumed_arm_reports_the_rate_of_the_whole_arm` pins it; run against the pre-fix tree it
fails at `KeyError: 'wall_seconds'` on the payload, which is how this note knows it is a pin
rather than a description.

## What these files are, and where they came from

The record and the platform's log were fetched by
`scripts/fetch_tpu_session_rehearsal_when_done.sh`, which keeps both on either terminal state and
printed `verdict(record)`: `all four phases ran: the registered session path runs on xla and its
checkpoints are host payloads`. The five files below are byte-for-byte what the kernel left
behind.

**There is no v1 file here, and that is a gap worth naming.** v1's fetched record and log landed
at these same fixed paths (the fetcher writes `output/tpu-session-rehearsal.json`, where the
probe's `--save` keeps one file per version), and the v2 fetch overwrote them - so v1's reading
survives only as the dated rows in `docs/IMPLEMENTATION_STATUS.md` and `docs/T4_RUNBOOK.md`
(`completed: ["environment"]`, the three hours after `--- phase session`, the `Killed`), and not as
a file a reader can check. What v1 established is not lost - the defect it found is the fix v2
tested, and v2's record is the evidence for it - but "the raw v1 record is gone" is the honest
label rather than an omission here.

| file | what it is | sha256 |
|---|---|---|
| `tpu-session-rehearsal.json` | the record: `completed`, `errors`, each phase's own entry, `runtime`, `phase_deadlines` and the folded-in `watch` samples | `48ac0fa3303963514d2ecbbeda5b0fcdb27325a7aa874afd44e783e85354d3eb` |
| `act-tpu-session-rehearsal.log` | the platform's execution log, 256 events (246 stdout, 10 stderr), `time` seconds from kernel start: **the reading v1 could not give**, and the only place the watch's samples and the phase walls are printed as they happened | `54e220117f73740831703f3ceda31e5b23189c64eecd763366527af24b786dfc` |
| `tpu-session-rehearsal.json.watch.json` | the watch's sidecar, rewritten through a rename on every sample: the file a *killed* phase still leaves whole | `b6ae1dbbaaac09583bdb90babaec4472254bc490803a5d9b9fa1595854c26ac6` |
| `manifest.json` | the handoff's published manifest for the temporary root: `code_revision f3ed48f...`, dataset `act-checkpoints/v0-5-1`, seed 51, with the checksums of what it published | `13c8a91035a89909bfe0547368db4947f8d51816710c0733fe0f39295a13db94` |
| `seed-summary.json` | seed 51's published summary - the §5.14 null clause's own input, and the artifact carrying the **pre-fix** resumed rate described above | `1ca3243a59b767f5e4d4f597fa5252b15d5ff572c5b7dcbf11376bc43ad084af` |

## Where this leaves the amendment

The TPU week stood at **16.80 of 20 hours remaining** when this record was read (3.20 h used: the
probe's 0.14, v1's 3.02, v2's ~0.04), against **1.65 of 30 GPU-h**, both refreshing
**2026-10-03T00:00:00Z**. §5.14's option 2 - the null queue, twelve single-arm seeds, **8.06 h of
TPU time at the port rule's own 2.0x bar** - fits that clock.

What did not exist before this record was any evidence that the *session path* runs on XLA. The
port verdict (`tpu_probe.py` v6, 15,034 tokens/s, 2.96x the T4 pair) measured a step loop, and the
registered objection to moving the queue was that a session adds activation checkpointing, the
optimizer and scaler state, the per-arm RNG contract, a checkpoint saved **and resumed** across a
stop, the seed handoff publish and the two-arm placement. Each of those has now run once, at
development fidelity, and the two that a lazy backend can silently break - the marked step and the
host-written checkpoint - are the two this run measured.

What it does **not** price is the declared-scale rate on the session path: 2,013 tokens/s here
against the 15,034 the probe's step loop measured, a difference that is fidelity (20 steps pay a
compile and a per-step sync that 6,000 amortize) and not something this record can settle. So the
amendment's cost figure still rests on the port rule's step loop, and a reader should treat 8.06 h
as that rule's number rather than as a measured session.

The option itself is still **not taken**, and that is a position recorded in
`docs/IMPLEMENTATION_STATUS.md` rather than a scheduling choice: moving the do-nothing arm to
different hardware and a different dtype changes what it is paired against - the clause pairs the
null loss with the **primary** adaptive loss, which is a two-T4 fp16 record - so it is a
**registered amendment**, the operator's to sign. What this run changes is that the amendment is
now decidable on evidence rather than on the assumption that a session would work.
