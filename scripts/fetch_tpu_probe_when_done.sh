#!/usr/bin/env bash
# Wait for the TPU probe kernel and collect its record.
#
# The same failure that stalled the Gate-4 campaign applies to a one-shot probe: if the
# result is only visible to the terminal that launched it, then a terminal that ends takes
# the result with it. This polls the kernel and, when it finishes, downloads its output into
# `artifacts/tpu-probe/` (git-ignored, like the rest of `artifacts/*`) and prints the record.
#
#   setsid nohup bash scripts/fetch_tpu_probe_when_done.sh > /tmp/tpu-probe-watch.log 2>&1 &
#
set -uo pipefail

KERNEL=${KERNEL:-sid123123234/act-tpu-probe}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO_ROOT/artifacts/tpu-probe/output"
POLL_SECONDS=${POLL_SECONDS:-120}
MAX_POLLS=${MAX_POLLS:-240}

cd "$REPO_ROOT" || exit 1
mkdir -p "$OUT_DIR"

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

# The record is a *file*; the reason a run failed is in the *log*, and the two come from
# different commands. Measured 2026-09-21, on the probe's first run: `kernels output`
# returned only the log file the runner had copied into /kaggle/working, and the failure
# itself was 148 s of traceback that only `kernels logs` carries. Fetch both, always.
fetch_log() {
  kaggle kernels logs "$KERNEL" > "$OUT_DIR/$(basename "$KERNEL").log" 2>&1 || true
  if [[ -s "$OUT_DIR/$(basename "$KERNEL").log" ]]; then
    echo "$(stamp) execution log: $OUT_DIR/$(basename "$KERNEL").log"
    tail -25 "$OUT_DIR/$(basename "$KERNEL").log"
  else
    echo "$(stamp) the platform kept no execution log for this run"
  fi
}

for poll in $(seq 1 "$MAX_POLLS"); do
  status=$(kaggle kernels status "$KERNEL" 2>&1 | tail -1)
  case "$status" in
    *COMPLETE*)
      echo "$(stamp) $status - fetching output"
      kaggle kernels output "$KERNEL" -p "$OUT_DIR" 2>&1 | tail -5
      fetch_log
      if [[ -f "$OUT_DIR/tpu-probe.json" ]]; then
        echo "$(stamp) record:"
        cat "$OUT_DIR/tpu-probe.json"
      else
        echo "$(stamp) no tpu-probe.json in the output; the log above says why"
      fi
      exit 0
      ;;
    *ERROR* | *CANCELLED*)
      echo "$(stamp) $status - fetching whatever was produced"
      kaggle kernels output "$KERNEL" -p "$OUT_DIR" 2>&1 | tail -5
      # The probe now writes its record as each measurement lands, so an ERROR run can
      # still carry real numbers - which is the difference between 6.5 h of queue buying
      # a measurement and buying nothing (it bought nothing on 2026-09-21).
      if [[ -f "$OUT_DIR/tpu-probe.json" ]]; then
        echo "$(stamp) PARTIAL record (this run did not finish):"
        cat "$OUT_DIR/tpu-probe.json"
      fi
      fetch_log
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
