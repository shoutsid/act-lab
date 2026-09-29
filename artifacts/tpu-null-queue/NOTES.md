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
| **The code it runs** | Code dataset `sid123123234/act-lab-code` at **`bf40d53984fcaba5fca47be1fa6083984093fae0`**, downloaded back and checked: `code/scripts/tpu_null_queue.py` sha256 **`9e744d9c2edce37123782d29bd9b14b7ee1cf042c0ae9077e73ca12619c1e971`** (byte-identical to the tree the local XLA:CPU dry run ran) and `code/src/act/model/checkpointing.py` sha256 **`0bf0ee2e6de98a1c947aef8ade03611d816fafd504aa5f84d6de24a8f37917d2`**. |
| **Why it has not started** | Kaggle allows **one TPU session at a time per account**, so the push was refused (`Maximum batch TPU session count of 1 reached`) while the price probe v2 held the slot. **And that refusal is invisible to an exit code:** `kaggle kernels push` prints that message and exits **0** — measured 2026-09-29, and it is what the first version of the slot waiter tripped over (it announced a fired queue for a kernel that did not exist; the account's own kernel list was the check that caught it). Success is now the CLI's own `successfully pushed` line, read by both the waiter and `push_tpu_null_queue.push_kernel` (exit 6). |
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
- **GPU quota:** 1.65 of 30 h, same refresh. The primary block stands at **6 of 12** seeds and the
  driver's lock is **not** held (its process is gone), so nothing resumes at the refresh until the
  campaign is relaunched. This run does not touch the GPU week.
