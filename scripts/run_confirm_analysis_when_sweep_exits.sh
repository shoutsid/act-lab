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
#   - the analyses run unbuffered (`python3 -u`) and both records they write are
#     confirmed by re-rendering them from the arms' own documented commands. A
#     stdout redirect to a file is
#     block-buffered, so a kill inside the analysis loses the output *and* the exit
#     line; the record that was never written then reads as a successful run. This
#     is measured rather than feared - see the horizon arm's watcher, whose single
#     analysis was killed inside 4 seconds and left the driver's imported
#     kernel-side file in place as its record, with a log that stopped mid-line.
set -u

cd "$(dirname "$0")/.." || exit 1

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
PYTHONPATH=src .venv/bin/python3 -u scripts/arch_zoo_analyze.py \
    --experiment-id "$EXP" --samples 200000 --write \
    >> "$LOG" 2>&1
status=$?
log "replication analysis exit=$status"

log "running the pre-registered pooled six-seed analysis with --head-rule (writes into $PARENT)"
PYTHONPATH=src .venv/bin/python3 -u scripts/arch_zoo_analyze.py \
    --experiment-id "$PARENT" --also "$EXP" --samples 200000 --write --head-rule \
    >> "$LOG" 2>&1
status=$?
log "pooled analysis exit=$status"

# Exit status is not the landing. A 0 says the analyses finished; it says nothing
# about whether they wrote, and the failure worth guarding is a record that was never
# written - leaving an older artifact in place while the log reports success. So both
# records this watcher owns are confirmed the strong way: each is re-rendered from its
# own arm's documented command and must come back byte-identical, by the same check the
# test suite runs. That subsumes a timestamp comparison and it names the cause when it
# fails - an arm whose committed record was read over a seed set its runbook does not
# read is reported as exactly that, with both sets printed. Non-destructive: the check
# regenerates into a scratch tree rather than over these artifacts.
log "verifying the landing: both records must reproduce from their runbooks"
if PYTHONPATH=src .venv/bin/python3 -u scripts/check_record_freshness.py \
        --only "$EXP,$PARENT" >> "$LOG" 2>&1; then
    log "verified: $EXP and $PARENT records reproduce from their documented commands"
else
    log "FAILED: a record does not reproduce from its documented command (exits $status)"
    log "        Nothing is a record until it does. Re-run the commands above by hand and"
    log "        read the check's output, which names which arm and why."
    exit 4
fi

log "done - read $PARENT/decision.md for the six-seed record and its branch"
