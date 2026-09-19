#!/usr/bin/env bash
# Fire the pre-registered finalist-order analysis for
# arch-zoo-100m-v1-finalists-6seed when its sweep exits. This arm is read *pooled*
# with arch-zoo-100m-v1 - the record it exists to resolve - so both arms' ledgers
# have to be complete before anything is read.
#
# Guards, the same ones the horizon arm's watcher learned:
#   - the sweep must be gone before anything is analysed. The sweep writes a ledger
#     per (length, seed) only once every candidate in it has finished, so a partial
#     arm is indetectable from the file count alone - which is why the row count
#     inside each ledger is checked as well;
#   - v1's own 1200-step ledgers must be present too. Without them the pooled design
#     is 3 blocks, the rule is declared on 6, and the rule refuses - correctly, but a
#     refusal caused by a missing pool partner should be visible in the log rather
#     than inferred from the branch.
#
# Deliberately *not* --write: this watcher's job is to make the result survive a
# dropped session, and it runs unattended. The verdict, the branch and every
# condition land in the log; writing v1's decision.md is a deliberate step taken
# once the branch has been read, because it supersedes a committed artifact.
#
#   launch detached:  setsid bash scripts/run_finalist_analysis_when_sweep_exits.sh &
set -u

cd "$(dirname "$0")/.." || exit 1

EXP=arch-zoo-100m-v1-finalists-6seed
PARENT=arch-zoo-100m-v1
DIR=artifacts/$EXP
LOG=$DIR/analysis_watcher.log
SWEEP_PATTERN='arch_zoo_screen\.py --experiment-id arch-zoo-100m-v1-finalists-6seed'
FRESH_SEEDS="45 46 47"
PARENT_SEEDS="42 43 44"
EXPECTED_ROWS=3

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

log "watcher started (sweep pid $(pgrep -f "$SWEEP_PATTERN" | head -1 || echo none))"

while pgrep -f "$SWEEP_PATTERN" >/dev/null 2>&1; do
    sleep 120
done
log "sweep has exited"

complete=true
for seed in $FRESH_SEEDS; do
    f="$DIR/flat1200_seed${seed}_results.jsonl"
    if [ ! -f "$f" ]; then
        log "INCOMPLETE: missing $f"
        complete=false
    else
        rows=$(grep -c . "$f")
        log "fresh seed $seed: $rows rows"
        [ "$rows" -lt "$EXPECTED_ROWS" ] && complete=false
    fi
done
for seed in $PARENT_SEEDS; do
    f="artifacts/$PARENT/stage3_seed${seed}_results.jsonl"
    if [ ! -f "$f" ]; then
        log "INCOMPLETE: the pool partner is missing $f"
        complete=false
    else
        log "v1 screening seed $seed: $(grep -c . "$f") rows"
    fi
done

if [ "$complete" != "true" ]; then
    log "NOT analysing: the pooled six-block design is not complete. Read the FAILED" \
        "lines in $DIR/run.log, and note that a sweep which lost runs now exits" \
        "non-zero rather than reporting success."
    exit 2
fi

log "ledgers complete; running the pre-registered analysis (pooled, no --write)"
PYTHONPATH=src .venv/bin/python3 scripts/arch_zoo_analyze.py \
    --experiment-id "$PARENT" \
    --also "$EXP" \
    --finalist-order-rule \
    >> "$LOG" 2>&1
status=$?
log "analysis exit=$status"

if [ "$status" = "0" ]; then
    log "done - read the finalist-order rule section above for the branch"
else
    log "analysis failed; the ledgers are complete, so re-run the command by hand"
fi
