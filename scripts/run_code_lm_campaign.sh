#!/usr/bin/env bash
# The code-LM campaign, start to finish, on one GPU.
#
# Why this is a script rather than four commands typed in sequence: every stage is
# hours long, they share the single 8 GiB card, and a stage re-launched by hand
# after a failure is how a record ends up holding two different budgets. Every
# number is fixed here once, echoed into the log, and written into the artifacts.
#
# The design it implements, and why each pin is where it is:
#
#   score    the 1.3B teacher labels the first SCORE_TOKENS of the corpus stream.
#            15.36M tokens is 30,000 windows, which is 4.05 h at the measured
#            1,053 teacher tokens/s. This is the expensive stage: the student can
#            afford three passes over such a prefix, the teacher cannot.
#   arms     both arms train on *the same* SCORE_TOKENS. STEPS is chosen so that
#            STEPS x BATCH x 512 is SCORE_TOKENS x EPOCHS, i.e. 11,250 x 8 x 512 =
#            46.08M student tokens = three passes over the scored prefix. The only
#            difference between the arms is the objective, so a difference in the
#            result is attributable to distillation and not to data volume.
#   bench    every model is scored by scripts/code_bench.py, on the same held-out
#            text, in bits per byte and executed HumanEval pass@1. Three models
#            scored by three scripts would be three different numbers. The two
#            fast rows run before the two slow ones; see the note at stage 3.
#
# Usage:
#   nohup bash scripts/run_code_lm_campaign.sh > /tmp/code-campaign.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
export PYTHONPATH="src"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

SCALE="${SCALE:-match}"
SCORE_TOKENS="${SCORE_TOKENS:-15360000}"
EPOCHS="${EPOCHS:-3}"
BATCH="${BATCH:-8}"
STEPS="${STEPS:-$(( SCORE_TOKENS / (BATCH * 512) * EPOCHS ))}"
BENCH_BYTES="${BENCH_BYTES:-4000000}"

# Local directories, not hub ids: the teacher's 2.7 GB is fetched once by
# `train_code_lm.py --fetch-teacher` and passing the repository id here would pull
# it a second time. The reader's alternative is `--model hf:<repo-id>`, which costs
# the download and is otherwise identical.
TEACHER="artifacts/code-lm/teacher"
BASELINE="artifacts/code-lm/baseline"
CHECKPOINTS="artifacts/code-lm/checkpoints"

# The baseline can be reached by hub id if the local copy is absent. The teacher
# cannot: the student's tokenizer *is* the teacher's, so the token stream cannot be
# built without its files, and that fetch therefore happens before stage 1. Retarget
# TEACHER at a hub id only after fetching it, or shared_tokenizer() has nothing to read.
[ -f "$BASELINE/config.json" ] || BASELINE="bigcode/tiny_starcoder_py"

log() { echo "[$(date +%H:%M:%S)] === $*"; }

log "campaign: scale=$SCALE score_tokens=$SCORE_TOKENS epochs=$EPOCHS batch=$BATCH steps=$STEPS"
log "arms will each see $(( STEPS * BATCH * 512 )) student tokens over $SCORE_TOKENS labelled ones"

if [ ! -f "$TEACHER/config.json" ]; then
    log "teacher weights absent; fetching them before anything that needs the tokenizer"
    $PY scripts/train_code_lm.py --fetch-teacher
fi

log "stage 1/3: teacher targets"
$PY scripts/train_code_lm.py --stage targets --score-tokens "$SCORE_TOKENS" --targets-batch 4

log "stage 2/3a: distilled arm"
$PY scripts/train_code_lm.py --stage train --mode distill --scale "$SCALE" \
    --max-steps "$STEPS" --batch-size "$BATCH" --epochs "$EPOCHS" \
    --log-every 250 --score-tokens "$SCORE_TOKENS"

log "stage 2/3b: cross-entropy control arm (same data, same order, same seed)"
$PY scripts/train_code_lm.py --stage train --mode ce --scale "$SCALE" \
    --max-steps "$STEPS" --batch-size "$BATCH" --epochs "$EPOCHS" \
    --log-every 250 --score-tokens "$SCORE_TOKENS"

log "stage 3/3: benchmarks"
# The cheap rows go first, and that is worth about fifteen hours of waiting. Each row is
# written to artifacts/code-lm/bench/ as soon as its byte scan finishes, and the table needs
# all four - so with the two student arms first, whose HumanEval halves are hours each, the
# four-row table could not be built until this stage was nearly over. The teacher and the
# baseline are minutes (their generate path keeps a KV cache). First also means the
# baseline's pass@1 - the only number here with an outside reference, 7.84% published - is
# checked against our harness before the two long rows run rather than after them, and that a
# failure inside a student's HumanEval cannot cost the rows that do not depend on it.
$PY scripts/code_bench.py --model "hf:$TEACHER" --bytes-limit "$BENCH_BYTES" --tag "teacher"
$PY scripts/code_bench.py --model "hf:$BASELINE" --bytes-limit "$BENCH_BYTES" --tag "baseline"
$PY scripts/code_bench.py --model "ours:$CHECKPOINTS/$SCALE-distill.pt" --scale "$SCALE" \
    --bytes-limit "$BENCH_BYTES" --tag "ours-$SCALE-distill"
$PY scripts/code_bench.py --model "ours:$CHECKPOINTS/$SCALE-ce.pt" --scale "$SCALE" \
    --bytes-limit "$BENCH_BYTES" --tag "ours-$SCALE-ce"

log "campaign complete; results in artifacts/code-lm/bench/"
