#!/usr/bin/env bash
# Fire the pre-registered horizon analysis for arch-zoo-100m-v3-crossed-horizon10x
# when its Kaggle driver exits. The arm's NOTES.md pre-registers one command, whose
# rule is declared on the 12,000-step rows while a 1200-step anchor sits in the same
# directory - so the two guards below are the difference between reading the horizon
# and reading the anchor as if it were the horizon.
#
# Guards, both learned in this project's own logs:
#   - the driver must be gone before anything is analysed, because the driver is
#     what copies the ledgers out of the kernel output; analysing mid-copy reads a
#     half-written arm;
#   - the ledgers must be complete - 8 rows at each of 1200 and 12,000 for seeds 42
#     and 43 - before --write, so a kernel that lost its horizon again produces a
#     log line rather than a decision. This is not hypothetical: the arm's first
#     attempt failed exactly that way and reported success, which is why
#     HORIZON_STEPS is declared rather than inferred and why the rule refuses an
#     outcome at any other length.
#   - launch this detached:  setsid bash scripts/run_horizon_analysis_when_kernel_exits.sh &
set -u

cd "$(dirname "$0")/.." || exit 1

EXP=arch-zoo-100m-v3-crossed-horizon10x
DIR=artifacts/$EXP
LOG=$DIR/analysis_watcher.log
DRIVER_PATTERN='kaggle_queue\.py --queue archzoo'
SEEDS="42 43"
LENGTHS="1200 12000"
EXPECTED_ROWS=8
# The fp32 record the arm's fp16 anchor is measured against, for the
# pre-registered precision offset. Read *beside* the arm, never pooled with it:
# the two were measured under different precisions, which is the quantity.
REFERENCE=arch-zoo-100m-v3-crossed

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# pgrep excludes itself, and this script's own cmdline names the file rather than
# the driver, so the only matches are the driver process.
log "watcher started (driver pid $(pgrep -f "$DRIVER_PATTERN" | head -1 || echo none))"

while pgrep -f "$DRIVER_PATTERN" >/dev/null 2>&1; do
    sleep 300
done
log "driver has exited"

complete=true
for seed in $SEEDS; do
    for length in $LENGTHS; do
        f="$DIR/flat${length}_seed${seed}_results.jsonl"
        if [ ! -f "$f" ]; then
            log "INCOMPLETE: missing $f"
            complete=false
        else
            rows=$(grep -c . "$f")
            log "seed $seed at $length steps: $rows rows"
            [ "$rows" -lt "$EXPECTED_ROWS" ] && complete=false
        fi
    done
done

if [ "$complete" != "true" ]; then
    log "NOT writing: ledgers incomplete - the arm is partial, so the horizon rule" \
        "would refuse anyway; re-push and read the driver log"
    exit 2
fi

log "ledgers complete (32 rows); running the pre-registered horizon analysis"
# `-u` is load-bearing, not cosmetic. The analysis takes ~4 seconds, and its output
# is redirected to a file, so Python block-buffers stdout: a kill inside that window
# loses every line including any traceback, and the log ends mid-line with no record
# that anything was attempted. That is exactly how this watcher failed the first time
# it ran - the process was torn down mid-analysis, `decision.md` stayed the kernel-side
# single-seed artifact the driver had imported, and the only trace was a log that
# stopped after the line above. Unbuffered, the run leaves its own trace.
PYTHONPATH=src .venv/bin/python3 -u scripts/arch_zoo_analyze.py \
    --experiment-id "$EXP" --write --horizon-rule \
    --offset-against "$REFERENCE" --offset-steps 1200 \
    >> "$LOG" 2>&1
status=$?
log "horizon analysis exit=$status"

# Exit status alone is not the landing. A 0 says the process finished; it says
# nothing about whether it wrote, and the failure that matters here is a record that
# was never written at all - leaving the driver's imported kernel-side artifact in
# place, which for this arm is a single-seed analysis of a two-seed design. The write
# is verified directly: the record must be newer than every ledger it claims to
# describe. Written after the fact, so it cannot pass on a record that predates them.
NEWEST=""
for seed in $SEEDS; do
    for length in $LENGTHS; do
        f="$DIR/flat${length}_seed${seed}_results.jsonl"
        if [ -z "$NEWEST" ] || [ "$f" -nt "$NEWEST" ]; then NEWEST="$f"; fi
    done
done

if [ ! -f "$DIR/decision.md" ]; then
    log "FAILED: no decision.md after the analysis; the record in place is whatever the"
    log "        driver imported from the kernel (for this arm: a single-seed analysis of"
    log "        a two-seed design). Re-run the command above by hand."
    exit 4
elif [ ! "$DIR/decision.md" -nt "$NEWEST" ]; then
    log "FAILED: decision.md is older than $NEWEST, so it was not written from the"
    log "        ledgers that landed - the analysis did not complete. Re-run by hand."
    exit 4
fi

if [ "$status" = "0" ]; then
    log "verified: decision.md is newer than all 32 ledgers, so it was written from them"
    log "done - read $DIR/decision.md for the branch and the length it was read at"
else
    log "analysis failed; the ledgers are complete, so re-run the command by hand"
fi
