# TPU session price probe — the evidence behind the 2026-09-29 finding

## The fire: one session at the declared shape, and it could not start

`sid123123234/act-tpu-session-price` **version 1**, pushed **2026-09-28T20:54:37Z** from the code
dataset republished at **`98d5574e8b3f10e34f14da1b243e08a5c4be0fd3`**, reached
`KernelWorkerStatus.ERROR` and was fetched at **2026-09-29T05:36:37Z** — **8.7 h** in the queue,
then **28.09 s** of measured run. `completed` is `["environment"]`, `errors` has three entries, and
the fetch printed *"no price: the session phase did not leave a throughput to read"*.

| | |
|---|---|
| **The record** | `session`, `resume` and `handoff` each failed with the identical message, **`AttributeError: module 'torch' has no attribute 'xla'`** — at **4.5 s**, **3.1 s** and **3.7 s**. The `environment` phase passed first (16.8 s, 8 bound devices, the tensor round-trip `sum() == 8.0`), so the hardware was measured working before the path failed, and the failure is in *every* phase that trains — which is what says the defect is in constructing the loop rather than in one phase's setup. |
| **The site** | `torch/utils/checkpoint.py`'s `_get_device_module`: `if device == "meta": ... ; device_module = getattr(torch, device)`, reached from **line 1879** inside `_checkpoint_without_reentrant_generator_impl` — the implementation this repository calls (`use_reentrant=False`), where the RNG state is stashed per device. `_infer_device_type` returns `"xla"` for a checkpointed tensor on a TPU, so the lookup is `getattr(torch, "xla")`, and XLA is a separate package rather than an attribute of `torch`. |
| **Why the rehearsal could not have found it** | `activation_checkpointing` is the **declared-only** field: `declared_session_config` sets it **True** (\"section 5.2's T4 fit arithmetic assumes checkpointed activations\") while the development config the rehearsal trains has it **False**. The rehearsal ran the real session path on XLA and never entered a checkpointed block; its own record said `declared_scale: false`, and this is the first cost that shape-level gap has had. |
| **Upstream's answer** | PyTorch/XLA ships `torch_xla/utils/checkpoint.py`, a fork of that same file, whose header says why: \"PyTorch/XLA needs to add `optimization_barrier` before saving the input for the backward hence we slightly modify the upstream version\". Its lookups are `device_module = xm if device_type == \"xla\" else _get_device_module(device_type)`. Its public entry point also documents the constraint that shapes the fix: `raise ValueError(\"XLA currently does not support use_reentrant==False\")`. |
| **The fix (after the fetch)** | `src/act/model/checkpointing.py` — one dispatch, `activation_checkpoint(device, fn, *args)`: CPU and CUDA keep `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` **unchanged**, and `device.type == \"xla\"` routes to the fork **without** the `use_reentrant` keyword, since the fork rejects `False` and defaults to the only implementation it has. All four call sites go through it (`dense`, `moe`, and `v0_moe`'s block loop plus its chunked scan), so the rule lives in one place — the shape `runtime.step_marker` already has. Missing `torch_xla` is refused by name. |
| **A second finding: the amendment's dtype does not exist** | Section 5.14's option 2 prices a **TPU/bf16** arm, and `bf16` is **not a session dtype** — `V0SessionConfig` accepts `fp32` and `fp16` and raises for the rest, so that arm is a change to section 5.2's session model rather than a flag on a push command. And `fp16`, the declared dtype this kernel was fired with, is the one needing `torch.autocast(device_type=\"xla\", ...)` plus a `GradScaler` — two things no record here shows running on a TPU. The runnable registered option is **fp32**, which is what v2 prices: the slower end of the two dtypes `tpu_probe` v5 measured within 6% of each other, so the price errs against the queue. Since the fix, the probe takes `--dtype` (default `fp32`) and the record carries it. |
| **The diagnostic gap this exposed, now closed** | `tpu_probe._run_phase` recorded `f\"{type(exc).__name__}: {exc}\"` and nothing else, so a record from a queued run carried the symptom and not the site — and that particular message is ambiguous across more than a dozen `getattr(torch, <device_type>)` lookups in torch alone. Every error entry now carries **`traceback`**. This is the second time a failure here was diagnosable only from a kernel log (v1's `Killed` was the first). |
| **What the watch saw, or did not** | Samples at 0 s of each phase: RSS **528,852 kB** (environment), **7,044,976 kB** (session), **8,048,152 kB** (resume), **8,273,536 kB** (handoff) — the declared model and payload on the host. `device_bytes_used` is **null** in every sample, as in every `tpu-probe` record back to v4, so this runtime does not expose it; the host's resident set is the growth signal, and it is measured rather than assumed. Each phase's sample list has exactly one entry because each phase **failed fast** — the watch never got to its second tick. |
| **The labels, which kept this from reading as a result** | `declared_shape: true` beside `declared_seed: false` and `declared_scale: false`, `rehearsal: false`, a `scratch_root` line stating the handoff publishes nowhere near a campaign dataset, and `not_measured` naming that it is not a declared seed, not eight chips at once, not a mutation/rebuild cost, and not the amendment's bf16. `runtime.torch_xla` is `2.8.0`, `pytorch` `2.8.0+cpu`, `TPU_ACCELERATOR_TYPE v5litepod-8`, `TPU_CHIPS_PER_PROCESS_BOUNDS 2,4,1`, 8 bound devices. |

## What these files are, and where they came from

Fetched by `scripts/fetch_tpu_session_rehearsal_when_done.sh` with `KERNEL`,
`OUT_DIR` and `PUSHER` overridden for this kernel, on the terminal state (`ERROR`), keeping the
record, the log and the watch sidecar, and printing `verdict(record)` — which is the "no price"
line quoted above. No campaign queue moved: the GPU driver still holds the lock and is sleeping to
the 2026-10-03 refresh.

The fetcher writes fixed names into `output/`, so v2's fetch overwrites v1's there. **This time
both survive:** v1's three files were copied to `v1/` before the re-fire, and the hashes below are
that copy's. (The rehearsal's evidence folder could not do this - v2's fetch had already landed
when the overlap was noticed - so its note carries the apology this one does not need to make.)

| file (in `v1/`) | what it is | sha256 |
|---|---|---|
| `tpu-session-price-probe.json` | the record: `completed` (one phase), the three errors, `session.config` (the declared shape at 240 steps, fp16 as fired), `session.code_revision 98d5574...`, `runtime`, `phase_deadlines`, and the folded-in `watch` samples | `6a007a6c7ce07e75e38845298306af683fcf15f62df3d87f29216fa58a6c4cf5` |
| `act-tpu-session-price.log` | the platform's execution log, events with `time` seconds from kernel start — the phases' walls (environment ok 77.5 s, session failed 82.0 s, resume failed 85.1 s, handoff failed 88.8 s) and the record printed at 88.8 s | `bda1012fb599c8f79b8d7850e49e716f00f275a16f01d2baa1e0a54f8d0e9780` |
| `tpu-session-price-probe.json.watch.json` | the watch's sidecar: one sample, the one the fast failure allowed | `e5c749708d5fda098f663d14ab730a8bf0e872e2fde4188a272be583666cebbb` |

The record carries the **pre-fix** state of the repository: it was fired from `98d5574`, and the
checkpointing dispatch, the `traceback` field and the `dtype` option all landed after the fetch.
That is the same relationship the rehearsal's record has to its own later fixes, and it is why this
note quotes the numbers beside the code rather than after it.

## Where this leaves the amendment

The probe cost **8.7 h** of queue and about **30 s** of TPU quota. Quota after the fetch:
**TPU 16.77 of 20 h** remaining (**3.23** used: the TPU probe's 0.14, rehearsal v1's 3.02,
rehearsal v2's ~0.04, this ~0.03) against **GPU 1.65 of 30 h**, both refreshing
**2026-10-03T00:00:00Z** — so the measurement and its retry both fit the clock, which is the whole
argument for taking them before signing anything.

What the run bought is not a price. It is the discovery that the price could not have been
measured at all: the declared shape's session path had never started on a TPU, and the field that
stopped it is the one the port probe's hand-built loop did not have. Section 5.14's option 2 is
still **not taken** and is now a larger decision than it was, because the **bf16** arm it names
would need the session model changed to have a bf16 mode at all. What this record changes is that
the decision is no longer resting on a path nobody had run: the declared shape runs on XLA in a
registered dtype, and the price probe is re-armed to take it.
