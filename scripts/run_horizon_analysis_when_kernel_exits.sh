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
PYTHONPATH=src .venv/bin/python3 scripts/arch_zoo_analyze.py \
    --experiment-id "$EXP" --write --horizon-rule \
    >> "$LOG" 2>&1
status=$?
log "horizon analysis exit=$status"

if [ "$status" = "0" ]; then
    log "done - read $DIR/decision.md for the branch and the length it was read at"
else
    log "analysis failed; the ledgers are complete, so re-run the command by hand"
fi
