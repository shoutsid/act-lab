#!/usr/bin/env bash
# Wait for a gap in the Gate-4 campaign, then push the TPU probe and arm its fetch watcher.
#
# The probe's own guard refuses to republish the code dataset while a campaign session is
# live, and that refusal is correct: a session kernel's body and its mounted `src/` have to
# be one revision, and the dataset is republished in place. Measured 2026-09-26: the v6 push
# was refused with `sid123123234/act-t4-session` RUNNING, so the push has to wait for a gap -
# and a gap is a platform event, not something to watch for by hand.
#
# Two properties are deliberate. **The wait condition is the guard's own function**, imported
# rather than re-implemented (`live_campaign_kernel`), so the waiter and the guard cannot
# disagree about what "idle" means - the failure mode of two copies is a waiter that pushes
# into a live mount. **It never publishes anything itself**: it invokes `push_tpu_probe.py`,
# which re-checks the campaign at its own push time, because the gap it waited for can close
# in the seconds between a poll and the publish.
#
# Bounded on purpose: a campaign queue can hold the driver's lock for many hours, and a
# waiter that never gives up is a process nobody can tell from a hung one.
#
#   setsid nohup bash scripts/push_tpu_probe_when_idle.sh > /tmp/tpu-probe-push-watch.log 2>&1 &
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

if [[ -n "${PYTHON:-}" ]]; then
  INTERPRETER="$PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  INTERPRETER="$REPO_ROOT/.venv/bin/python"
else
  INTERPRETER="python3"
fi
POLL_SECONDS=${POLL_SECONDS:-300}
WAIT_POLLS=${WAIT_POLLS:-144}
LOG=${LOG:-/tmp/tpu-probe-fetch-watch.log}

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

# The guard's own answer to "is a campaign session live?": exit 0 idle, exit 1 with the
# session named on stdout. It is `push_tpu_probe`'s function, not a copy of its logic.
campaign_is_idle() {
  PYTHONPATH=src "$INTERPRETER" - <<'PY'
import sys

sys.path.insert(0, "scripts")
import push_tpu_probe

live = push_tpu_probe.live_campaign_kernel()
if live is not None:
    print(live)
    raise SystemExit(1)
raise SystemExit(0)
PY
}

for attempt in $(seq 1 "$WAIT_POLLS"); do
  if live=$(campaign_is_idle); then
    echo "$(stamp) the campaign is between sessions - pushing the probe"
    if PYTHONPATH=src "$INTERPRETER" scripts/push_tpu_probe.py; then
      echo "$(stamp) pushed; arming the fetch watcher (its log: $LOG)"
      setsid nohup bash scripts/fetch_tpu_probe_when_done.sh >"$LOG" 2>&1 &
      echo "$(stamp) watcher pid $!"
      exit 0
    fi
    echo "$(stamp) the push refused - a session came back while waiting; will retry"
  else
    echo "$(stamp) poll $attempt/$WAIT_POLLS: $live"
  fi
  sleep "$POLL_SECONDS"
done

echo "$(stamp) gave up after $WAIT_POLLS polls without a gap in the campaign" >&2
exit 2
