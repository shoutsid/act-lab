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

for poll in $(seq 1 "$MAX_POLLS"); do
  status=$(kaggle kernels status "$KERNEL" 2>&1 | tail -1)
  case "$status" in
    *COMPLETE*)
      echo "$(stamp) $status - fetching output"
      kaggle kernels output "$KERNEL" -p "$OUT_DIR" 2>&1 | tail -5
      if [[ -f "$OUT_DIR/tpu-probe.json" ]]; then
        echo "$(stamp) record:"
        cat "$OUT_DIR/tpu-probe.json"
      else
        echo "$(stamp) no tpu-probe.json in the output; the kernel log says why:"
        ls -la "$OUT_DIR" | tail -10
      fi
      exit 0
      ;;
    *ERROR* | *CANCELLED*)
      echo "$(stamp) $status - fetching whatever was produced"
      kaggle kernels output "$KERNEL" -p "$OUT_DIR" 2>&1 | tail -5
      ls -la "$OUT_DIR" | tail -10
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
