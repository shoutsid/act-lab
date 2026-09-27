#!/usr/bin/env bash
# Drive the whole Gate-4 campaign unattended: the three queues, then the verdict.
#
# The runbook's per-queue commands (`kaggle_queue.py --queue session|control|null`,
# then `--queue verdict`) are correct but not self-sequencing: between queues, and
# after exit 2 ("session budget ran out with seeds still missing") or exit 3
# ("weekly quota cannot cover another session"), something has to re-run the next
# command. Left to an operator, the campaign stops whenever a terminal closes -
# which is the state this script exists to end.
#
# The driver is the thing that owns the decisions: the queue, the clock, the
# seeds, the deadline rules and every quota gate are its own, and `--wait-for-quota`
# makes it sleep to the payload's own `refreshAt` instead of exiting 3. This script
# only sequences the queues and re-runs on the two exit codes that mean "not
# finished" rather than "wrong".
#
# Exit codes (from the driver): 0 queue complete, 1 session failed,
# 2 session budget ran out with seeds missing, 3 quota cannot cover another
# session, 4 a session completed without advancing the queue. 2 and 3 are
# re-run; 1 and 4 mean the cause has to be fixed, so the campaign stops and
# says so rather than spending a week's budget repeating a broken session.
#
#   bash scripts/run_gate4_campaign.sh            # foreground
#   setsid nohup bash scripts/run_gate4_campaign.sh > /tmp/gate4-campaign.log 2>&1 &
#
set -uo pipefail

QUEUES=(session control null verdict)
MAX_ATTEMPTS=${MAX_ATTEMPTS:-12}
# The session window, when the week's remaining quota cannot cover the registered 7 h. The
# window is a scheduling knob, not a protocol term: a shorter one completes fewer whole
# pairs and stops by its own deadline, publishing an in-progress snapshot ("interrupted"
# is not counted as done), so the queue resumes from its last completed seed. Unused
# weekly quota expires at the refresh, so a fitted window is how the last hours of a week
# are spent on real pairs instead of lost.
HOURS=${HOURS:-}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=${PYTHON:-"$REPO_ROOT/.venv/bin/python"}
DRIVER="$REPO_ROOT/scripts/kaggle_queue.py"

cd "$REPO_ROOT" || exit 1

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

if [[ ! -x "$PYTHON" ]]; then
  echo "$(stamp) no interpreter at $PYTHON" >&2
  exit 1
fi

echo "$(stamp) campaign start: queues ${QUEUES[*]}"

for queue in "${QUEUES[@]}"; do
  # Print the durable state at every queue boundary. The campaign's two 2026-09 losses were
  # expensive partly because nothing said where it had got to, and this report reads the
  # published dataset, the kernel status and the quota rather than anything local - so the
  # campaign's own log carries a record of what it was doing even when a driver dies.
  # The window the sessions will actually use, so the quota verdict is stated against it: with
  # HOURS fitted to a short week, a report testing the kernel's declared 7 h would print
  # "cannot fit a 7.0 h session" while the driver is about to push a 6.5 h one.
  health=(--queue "$queue")
  if [[ -n "$HOURS" ]]; then
    health+=(--window "$HOURS")
  fi
  PYTHONPATH="$REPO_ROOT/src" "$PYTHON" "$REPO_ROOT/scripts/campaign_health.py" \
    "${health[@]}" 2>&1 | sed 's/^/  /' || true
  attempt=0
  while :; do
    attempt=$((attempt + 1))
    if (( attempt > MAX_ATTEMPTS )); then
      echo "$(stamp) $queue: gave up after $MAX_ATTEMPTS attempts" >&2
      exit 5
    fi
    echo "$(stamp) $queue: attempt $attempt"
    # `verdict` is a one-shot CPU job with no quota gate of its own; the flag is
    # ignored by the one-shot path, so passing it only for the three queues keeps
    # the intent explicit.
    if [[ "$queue" == "verdict" ]]; then
      PYTHONPATH="$REPO_ROOT/src" "$PYTHON" "$DRIVER" --queue "$queue"
    elif [[ -n "$HOURS" ]]; then
      echo "$(stamp) $queue: declaring a $HOURS h session window"
      PYTHONPATH="$REPO_ROOT/src" "$PYTHON" "$DRIVER" --queue "$queue" \
        --hours "$HOURS" --wait-for-quota
    else
      PYTHONPATH="$REPO_ROOT/src" "$PYTHON" "$DRIVER" --queue "$queue" --wait-for-quota
    fi
    code=$?
    if [[ "$queue" == "verdict" ]]; then
      # The verdict queue's exit code *is* the decision (runbook section 3):
      # 0 PASS, 1 FAIL, 2 PASS with qualification, 3 no verdict produced. A FAIL
      # is an answer, not a fault - re-running it would ask the same question of
      # the same artifacts - so only 3 (an incomplete declared block) stops.
      case "$code" in
        0 | 1 | 2)
          echo "$(stamp) verdict: decision produced (exit $code)"
          break
          ;;
        *)
          echo "$(stamp) verdict: no verdict produced (exit $code) - the declared block is incomplete" >&2
          exit "$code"
          ;;
      esac
    fi
    case "$code" in
      0)
        echo "$(stamp) $queue: complete"
        break
        ;;
      2)
        echo "$(stamp) $queue: session budget ran out with seeds missing; re-running"
        ;;
      3)
        # Reachable only when the refresh time is unreadable and the driver
        # refuses to sleep into a wait of unknown length (runbook section 3).
        echo "$(stamp) $queue: quota short and refreshAt unreadable; sleeping 1h"
        sleep 3600
        ;;
      1)
        echo "$(stamp) $queue: a session failed - stopping so the cause can be read" >&2
        exit 1
        ;;
      4)
        echo "$(stamp) $queue: a session completed without advancing the queue - stopping" >&2
        exit 4
        ;;
      *)
        echo "$(stamp) $queue: unexpected exit code $code; stopping" >&2
        exit "$code"
        ;;
    esac
  done
done

echo "$(stamp) campaign finished: all three queues complete and the verdict produced"
