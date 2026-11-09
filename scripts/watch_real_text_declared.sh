#!/usr/bin/env bash
# Wait out Kaggle's two-GPU-session cap, push wave c, then fetch and read 5.70.
#
# Waves a and b were accepted; wave c was refused with "Maximum batch GPU session
# count of 2 reached." That refusal exits 0 and prints an error, so success is the
# CLI's own "successfully pushed" line, not the status. The staged folder is pushed
# as-is: re-running the pusher would republish the code dataset at HEAD and let a
# later commit become the revision the queued kernels mount and the rows cite.
#
#   setsid nohup bash scripts/watch_real_text_declared.sh > /tmp/realtext-watch.log 2>&1 &
#
# Exit 0 once all three waves are terminal, fetched, and the twelve-seed reading is
# written; 2 if the fetch/reading failed; 1 at the deadline.
set -uo pipefail
cd "$(dirname "$0")/.."

OWNER=sid123123234
LOG=${LOG:-/tmp/realtext-watch.log}
POLL=${POLL:-120}
WATCH_HOURS=${WATCH_HOURS:-72}
DEADLINE=$(( $(date +%s) + WATCH_HOURS * 3600 ))

say() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }
status_of() { kaggle kernels status "$OWNER/$1" 2>&1 | tail -1; }
terminal() { echo "$1" | grep -qiE 'COMPLETE|ERROR|CANCEL'; }

say "watching act-realtext-a/b for a free GPU slot; c stays at revision 8fe7bbf"
pushed=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  a=$(status_of act-realtext-a)
  b=$(status_of act-realtext-b)
  say "a: ${a:-<no status>} | b: ${b:-<no status>}"

  if [ "$pushed" -eq 0 ] && { terminal "$a" || terminal "$b"; }; then
    out=$(kaggle kernels push -p artifacts/real-text-declared/kernel/c 2>&1)
    say "push c: ${out//$'\n'/ }"
    if echo "$out" | grep -q "successfully pushed"; then
      pushed=1
    fi
  fi

  if [ "$pushed" -eq 1 ]; then
    c=$(status_of act-realtext-c)
    say "c: ${c:-<no status>}"
    if terminal "$a" && terminal "$b" && terminal "$c"; then
      say "all three terminal; fetching"
      PYTHONPATH=src python scripts/push_real_text_declared.py --fetch
      fetch_code=$?
      say "fetch exit $fetch_code"
      if [ "$fetch_code" -ne 0 ]; then
        exit 2
      fi
      PYTHONPATH=src python scripts/summarize_real_text_declared.py
      read_code=$?
      say "reading exit $read_code"
      exit "$read_code"
    fi
  fi
  sleep "$POLL"
done

say "deadline reached (pushed=$pushed); push wave c later with: kaggle kernels push -p artifacts/real-text-declared/kernel/c"
exit 1
