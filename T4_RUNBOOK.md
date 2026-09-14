# T4 Pre-Launch Runbook — Section 5.1 Protocol Sessions

Operator runbook for executing the pre-registered compute-efficiency protocol
(`RESEARCH_REPORT.md` §5.1) on a Kaggle T4, using the session plan committed in
§5.2. Everything below is already implemented and tested; this document adds
only the exact commands, expected outputs, and decision points.

**Pre-registration discipline: no knob is tuned at any point in this runbook.**
The dtype rule, schedule constants, seed block, probe budget, criteria, and
tolerance band are fixed in code and in §5.1/§5.2. The verdict is machine-checked.

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
   - `artifacts/v0-sessions/` — the 20 GB working-dir root (checkpoints only,
     steady state ≈ 14.9 GB, `max_checkpoints=2` pin).

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

5. **STOP — decide §5.5 before launching anything.** The registered seed
   block **9000–9011 is spent**: the §5.3 real-data diagnostic ran that whole
   queue through the same `run_pair_session` at reduced fidelity and a §5.1
   verdict already exists over it (`artifacts/real-proof/datasets/
   v0-5-1-verdict.json` — measured cost 1.0077%, `PASS_COST_ONLY_HARM_GUARD_
   REJECTS`, `gate_4_authorized: false`, adaptive losing 11 of 12 pairs). The
   harm guard is one of the three branches that decide this verdict, so a run
   on that queue is no longer blind. `RESEARCH_REPORT.md` §5.5 change 4
   re-reserves the unused block **15000–15011**; if that revision is adopted,
   pass `--seeds 15000-15011` to the launch command in §2 and to
   `scripts/v0_verdict.py` in §4 (the verdict script's flag was added for this
   and loosens no criterion). Do **not** launch on 9000–9011 in either case.
   Publish the runs under a **fresh** dataset root (`artifacts/datasets`, the
   §2 default) — never point a session at `artifacts/real-proof/datasets`,
   whose completed reduced-fidelity seed versions would be treated as
   finished work by the skip-completed idempotence rule.

---

## 1. Session-1 throughput benchmark (fixes dtype + schedule)

```bash
PYTHONPATH=src python scripts/v0_throughput_benchmark.py --steps 50
```

Writes `artifacts/v0-benchmark.json` (the artifact the launch command reads).

Read from the printed JSON and record:

| Field | Meaning |
|---|---|
| `measured_tokens_per_second` | the step-loop measurement (fwd+bwd+AdamW, activation checkpointing on, as the sessions will run) |
| `dtype_decision` | `fp32` if ≥ 2,500 tokens/s, else `fp16` (pre-committed rule; not a choice) |
| `schedule.pair_time_hours` | time per random+adaptive pair at the measured rate |
| `schedule.full_pairs_per_session` | pairs a 12-hour clock fits per the deadline rules |
| `schedule.sessions_for_primary` / `sessions_total` | expected number of sessions for the 12-seed queue |
| `schedule.working_dir_headroom_gb` | must be positive (20 GB limit − steady state) |

Expected outcomes and responses:

- **OOM**: the script exits with a message pointing at the 16 GB T4 decision
  context. Do **not** shrink the config — run on a T4. `--profile` is for local
  smoke checks only and never produces launch input.
- **dtype = fp16**: expected on a T4 if throughput lands under the 2,500
  tokens/s floor. The sessions then train under autocast + GradScaler with
  fp32 master weights (checkpoint contract version 5 carries the scaler state;
  fp16 resume is bit-identical). Nothing to configure — the launch command
  re-derives the same decision from the artifact.

The benchmark is safe to re-run per session (it re-measures; the dtype rule
re-derives the same decision for the same rate class). Each session's artifact
is part of the trail; the launch command refuses a stale or hand-edited one.

---

## 2. The launch command (one line, registered)

```bash
PYTHONPATH=src python -m act.training.cli --mode pair-session \
  --device cuda \
  --hours <remaining-clock-hours> \
  --kaggle-dataset <owner>/<slug>
```

- `--hours`: the remaining session clock at entry — Kaggle's 12-hour GPU limit
  minus boot and restore time. The deadline rules only start pairs that fit
  `pair_time + 45-min publish reserve`.
- Everything else defaults correctly: benchmark artifact
  `artifacts/v0-benchmark.json`, declared seed queue **9000–9011**,
  `--probe-batch-count 1` (the primary config), dataset root
  `artifacts/datasets`, session dir `artifacts/v0-sessions`.
- Do **not** pass `--dtype` (it must equal the measured decision and is only
  there to catch operator error) or `--tokens-per-second` (overrides the
  artifact). `--seeds` is registered too — but see pre-flight item 5: if the
  §5.5 revision is adopted, `--seeds 15000-15011` is what "registered" means,
  and the same flag must go to the verdict script.

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

## 3. Running the queue to completion

The queue is 12 declared seeds (9000–9011), one atomic pair each. Schedule
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
  cleaned up): the verdict step ignores complete seeds outside 9000–9011 and
  warns about them, but the dataset stays cleanest when strays never land.

---

## 4. The verdict (machine-checked, pre-registered)

When all 12 seeds publish `complete` versions:

```bash
PYTHONPATH=src python scripts/v0_verdict.py \
  --dataset-root artifacts/datasets \
  --probe-batch-count 1
  # + `--seeds 15000-15011` if the §5.5 revision was adopted (pre-flight item 5)
```

Exit codes and meanings:

| Exit | Verdict | Pre-registered consequence |
|---|---|---|
| 0 | `PASS` | cost ≤ 5%, inside the [0.5, 2]× band, harm guard clear → **Gate 4 authorized** (`gate_4_authorized: true` requires 12/12 seeds) |
| 1 | `FAIL` | primary cost > 5% → stop; §5.1 fails as registered |
| 2 | `PASS_WITH_QUALIFICATION` | cost ≤ 5% but outside the band → re-derive the predicted C **before** Gate 4; no knob re-tuning against the measured C |
| 2 | `PASS_COST_ONLY_HARM_GUARD_REJECTS` | harm guard hit (adaptive worse in ≥ 10 of 12 pairs) → do not proceed to Gate 4 |

The verdict JSON is written to `<dataset-root>/v0-5-1-verdict.json` and prints
the aggregate cost, predicted value, band, harm-guard tally, and per-seed
win/loss pairs. Missing seeds are listed (`queue_complete: false`) — keep
running the launch command until the queue completes before judging.

Secondary config (if pre-registered): same command with
`--probe-batch-count 2` (predicted 0.1883%, band 0.0942–0.3767%).

---

## 5. Contingencies

| Symptom | Response |
|---|---|
| OOM in benchmark or sessions on non-T4 hardware | The decision context is a 16 GB T4. Move the run; never shrink the declared config. |
| Exit publish fails (`remote_publish.ok=false`) | Never fatal by design. The local handoff under `artifacts/datasets/act-checkpoints/v0-5-1/` is intact; the next session's publish re-pushes it. Check `remote-publish-log.json` beside the dataset root. |
| Restore fails at session start | Fix credentials / `owner/slug` / connectivity; relaunch. Nothing ran, nothing published. |
| Session died mid-pair (reboot, timeout) | Relaunch the same command. The pair restarts from scratch; the stale `in-progress` version is ignored. |
| Kaggle auto-extracted archives | Expected and handled: the wrapper stages per-seed tars with `seed-NNNN/...` internal paths (live-verified bit-identical round trip). Restore also extracts blobs itself as defense-in-depth. |
| Verdict reports stray complete seeds | They are excluded from the aggregate automatically. For cleanliness, delete stray seed directories from the dataset root or use a dedicated root for smoke tests. |
| Benchmark artifact missing/stale | Re-run step 1. The launch command refuses a hand-edited or stale artifact by design. |

---

## 6. Artifact trail (what to keep per session)

- `artifacts/v0-benchmark.json` — the measured throughput and dtype decision.
- `artifacts/v0-sessions/session-log.json` — queue, per-pair actions
  (completed / interrupted / skipped), FLOPs counters, `code_revision`.
- Dataset versions — every `complete` (and forensic `in-progress`) version of
  `<owner>/<slug>`, each with a checksummed manifest.
- `v0-5-1-verdict.json` — the final machine-checked §5.1 verdict.
- `remote-publish-log.json` — the exit-publish audit trail.
