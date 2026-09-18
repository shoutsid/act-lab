#!/usr/bin/env bash
# Fire the pre-registered analysis for arch-zoo-100m-v4-head-paired-confirm
# when its sweep finishes. The arm's NOTES.md pre-registers exactly two commands
# (independently first, then pooled for power); the second writes decision.md
# into the PARENT directory, which becomes the six-seed record.
#
# Guards, both learned in this project's own logs:
#   - the sweep must be gone (kill -0) before anything runs, so a still-running
#     sweep is never analysed mid-write;
#   - the ledgers must be complete (8 rows x seeds 51/52/53) before --write,
#     so a sweep that died early produces a log line, not a partial decision;
#   - the first background attempt in this repo died with its launching shell,
#     so launch this detached:  setsid bash scripts/run_confirm_analysis_when_sweep_exits.sh &
set -u

EXP=arch-zoo-100m-v4-head-paired-confirm
PARENT=arch-zoo-100m-v4-head-paired
DIR=artifacts/$EXP
LOG=$DIR/analysis_watcher.log
SWEEP_PID=556918
SEEDS="51 52 53"
EXPECTED_ROWS=8

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

log "watcher started (sweep pid $SWEEP_PID)"

while kill -0 "$SWEEP_PID" 2>/dev/null; do
    sleep 300
done
log "sweep process $SWEEP_PID has exited"

complete=true
for seed in $SEEDS; do
    f="$DIR/flat_seed${seed}_results.jsonl"
    if [ ! -f "$f" ]; then
        log "INCOMPLETE: missing $f"
        complete=false
    else
        rows=$(wc -l < "$f")
        log "seed $seed: $rows rows"
        [ "$rows" -lt "$EXPECTED_ROWS" ] && complete=false
    fi
done

if [ "$complete" != "true" ]; then
    log "NOT writing: ledgers incomplete - run the two commands by hand once the sweep is really done"
    exit 2
fi

log "ledgers complete (24 rows); running the pre-registered replication analysis"
PYTHONPATH=src .venv/bin/python3 scripts/arch_zoo_analyze.py \
    --experiment-id "$EXP" --samples 200000 --write \
    >> "$LOG" 2>&1
log "replication analysis exit=$?"

log "running the pre-registered pooled six-seed analysis with --head-rule (writes into $PARENT)"
PYTHONPATH=src .venv/bin/python3 scripts/arch_zoo_analyze.py \
    --experiment-id "$PARENT" --also "$EXP" --samples 200000 --write --head-rule \
    >> "$LOG" 2>&1
log "pooled analysis exit=$?"

log "done - read $PARENT/decision.md for the six-seed record and its branch"
