#!/usr/bin/env bash
# Wait for the TPU session rehearsal kernel and collect its record.
#
# The failure that stalled the Gate-4 campaign applies to a one-shot rehearsal too: if the
# result is only visible to the terminal that launched it, then a terminal that ends takes the
# result with it - and a TPU kernel can sit in the queue for hours before it starts, so
# "watch it and fetch it" is not a thing a person does. This polls the kernel and, when it
# finishes, fetches through `push_tpu_session_rehearsal.py --fetch --save`, which keeps the
# record *and* the log and prints what the record came to (`verdict`).
#
# Both terminal states fetch, and that is deliberate. An ERROR run is not an empty one: the
# rehearsal contains each phase (`tpu_probe._run_phase`) and writes the record after every one
# of them, so a run that died inside the session still carries the environment phase's reading
# of the hardware and the exception that stopped the rest. Fetching only on COMPLETE is how a
# measurement gets left on Kaggle, which is what happened to the probe's v4 for three days.
#
#   setsid nohup bash scripts/fetch_tpu_session_rehearsal_when_done.sh > /tmp/tpu-session-watch.log 2>&1 &
#
# One poll loop serves every TPU kernel this project queues: the session price probe reuses it
# with three values overridden, because a second copy of this loop is a second place for the
# fetch-on-both-terminal-states branch to be missing.
#
#   KERNEL=sid123123234/act-tpu-session-price \
#   OUT_DIR=artifacts/tpu-session-price/output \
#   PUSHER=scripts/push_tpu_session_price_probe.py \
#     setsid nohup bash scripts/fetch_tpu_session_rehearsal_when_done.sh >/tmp/x.log 2>&1 &
set -uo pipefail

KERNEL=${KERNEL:-sid123123234/act-tpu-session-rehearsal}
#: Which pusher's ``--fetch`` collects the output, and where it keeps it.
PUSHER=${PUSHER:-scripts/push_tpu_session_rehearsal.py}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR=${OUT_DIR:-$REPO_ROOT/artifacts/tpu-session-rehearsal/output}
# 360 polls of 120 s is 12 h: a queued TPU kernel can wait hours, and the rehearsal itself is
# minutes. Bounded rather than unbounded, because a waiter that never gives up is a process
# nobody can tell from a hung one.
POLL_SECONDS=${POLL_SECONDS:-120}
MAX_POLLS=${MAX_POLLS:-360}

cd "$REPO_ROOT" || exit 1
mkdir -p "$OUT_DIR"

if [[ -n "${PYTHON:-}" ]]; then
  INTERPRETER="$PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  INTERPRETER="$REPO_ROOT/.venv/bin/python"
else
  INTERPRETER="python3"
fi

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

# One command does all of it - download, keep the record beside its log, print the record and
# the verdict - so the watcher cannot keep half of what it fetched, which is the defect this
# project has already fixed twice (the probe's fetch on 2026-09-25, the code-bench fetch
# before it).
fetch() {
  echo "$(stamp) fetching into $OUT_DIR"
  PYTHONPATH=src "$INTERPRETER" "$PUSHER" --fetch --save "$OUT_DIR" 2>&1 | tail -60
}

for poll in $(seq 1 "$MAX_POLLS"); do
  status=$(kaggle kernels status "$KERNEL" 2>&1 | tail -1)
  case "$status" in
    *COMPLETE*)
      echo "$(stamp) $status - fetching output"
      fetch
      exit 0
      ;;
    *ERROR* | *CANCELLED*)
      echo "$(stamp) $status - fetching whatever was produced"
      fetch
      exit 1
      ;;
    *)
      if (( poll % 5 == 1 )); then
        echo "$(stamp) poll $poll: $status"
      fi
      sleep "$POLL_SECONDS"
      ;;
  esac
done

echo "$(stamp) gave up after $MAX_POLLS polls" >&2
exit 2
