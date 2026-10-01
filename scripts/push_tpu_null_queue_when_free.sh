#!/usr/bin/env bash
# Wait for the TPU slot, then push the null queue and arm its fetch watcher.
#
# Kaggle allows **one TPU session at a time per account**, and the push that fires the null queue
# was refused with exactly that on 2026-09-29 - `Maximum batch TPU session count of 1 reached` -
# because the price probe's v2 was already queued on the TPU. The wait that follows is a platform
# event, not something to watch for by hand, so this is the same shape as
# `push_tpu_probe_when_idle.sh`: a bounded poll loop that fires the push when the slot frees.
#
# It pushes the **already-staged folder** rather than invoking the pusher, and that is deliberate.
# The pusher republishes the code dataset at HEAD before it pushes; the dataset here was published
# at the commit that carries the runner, and a republish at push time would let a *later* commit -
# a docs-only one, say - become the revision the kernel mounts and the record names. The wait is
# for the slot, not for a republish, so nothing is published here at all. The pusher's own
# `push_kernel` reads the same success line this loop does, so the two cannot disagree about what
# a fired push looks like.
#
# A refused push is retried whether the refusal names the batch limit or not: the platform can
# refuse a push for its own reasons, the folder on disk is already the one that should go out, and
# the alternative is a run that never leaves because nobody was watching. Bounded, because a
# waiter that never gives up is a process nobody can tell from a hung one.
#
#   setsid nohup bash scripts/push_tpu_null_queue_when_free.sh > /tmp/tpu-null-push-watch.log 2>&1 &
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

KERNEL=${KERNEL:-sid123123234/act-tpu-null-queue}
#: The staged kernel folder: two files, written by the pusher from a committed tree.
FOLDER=${FOLDER:-artifacts/tpu-null-queue}
#: Whoever holds the TPU slot, named for the log - the limit is one at a time, so a reader of
#: this log has to be able to see who had it, and the price probe is the kernel that did.
BLOCKER=${BLOCKER:-sid123123234/act-tpu-session-price}
POLL_SECONDS=${POLL_SECONDS:-120}
#: 720 polls of 120 s is 24 h: a TPU kernel can sit in the queue for hours, and this one is
#: minutes of wall clock once the slot is free.
MAX_POLLS=${MAX_POLLS:-720}
#: The bound the *fetch* watcher gets, and it is longer than the slot waiter's on purpose.
#: Measured 2026-10-01: this kernel had sat `QUEUED` for 25 h, past the 24 h this script used to
#: arm its watcher with, so the watcher gave up on a run that had not started yet and the record
#: would have had to be noticed by hand. A queue that long is a platform-capacity event, not a
#: fault to retry - Kaggle's answer to the last capacity crunch was "we appreciate your
#: patience" - so the wait is outlasted rather than reset: 1920 polls of 180 s is 96 h.
FETCH_POLL_SECONDS=${FETCH_POLL_SECONDS:-180}
FETCH_MAX_POLLS=${FETCH_MAX_POLLS:-1920}
#: Where the fetch watcher it arms writes its own account.
FETCH_LOG=${FETCH_LOG:-/tmp/tpu-null-watch.log}

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

# The push's success is read off the CLI's own line, **not** its exit code. Measured 2026-09-29:
# `kaggle kernels push` printed `Kernel push error: Maximum batch TPU session count of 1 reached.`
# and exited **0** - so the one failure this waiter exists for is invisible to `$?`. The first
# version of this script trusted `$?` and announced a push that never left; the kernel was absent
# from the account's own list, and the fetch watcher it armed polled a 404 for two minutes.
for attempt in $(seq 1 "$MAX_POLLS"); do
  blocker=$(kaggle kernels status "$BLOCKER" 2>&1 | tail -1)
  out=$(kaggle kernels push -p "$FOLDER" 2>&1)
  if [[ "$out" == *"successfully pushed"* ]]; then
    echo "$(stamp) $KERNEL pushed on attempt $attempt ($BLOCKER was: $blocker)"
    # One poll loop serves every TPU kernel this project queues; the null queue is three values
    # overridden, because a second copy of that loop is a second place for the
    # fetch-on-both-terminal-states branch to be missing.
    KERNEL="$KERNEL" \
    OUT_DIR="$REPO_ROOT/artifacts/tpu-null-queue/output" \
    PUSHER=scripts/push_tpu_null_queue.py \
    POLL_SECONDS="$FETCH_POLL_SECONDS" MAX_POLLS="$FETCH_MAX_POLLS" \
      setsid nohup bash scripts/fetch_tpu_session_rehearsal_when_done.sh >"$FETCH_LOG" 2>&1 &
    echo "$(stamp) fetch watcher pid $! (its log: $FETCH_LOG)"
    exit 0
  fi
  echo "$(stamp) attempt $attempt/$MAX_POLLS: nothing was queued - $out"
  echo "$(stamp)   one TPU session at a time; $BLOCKER is $blocker"
  sleep "$POLL_SECONDS"
done

echo "$(stamp) gave up after $MAX_POLLS polls without a TPU slot for $KERNEL" >&2
exit 2
