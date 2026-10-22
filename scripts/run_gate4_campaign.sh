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
# The seed block each session declares, per queue, **in the order its handoff shards must
# be spent**. One session declares one shard, because the declaration is a disk budget as
# well as a training scope: the startup restore downloads every shard it resolves to, and
# the pair-time push stages the whole of the shard it belongs to. A declaration spanning
# two shards therefore holds one shard's bytes while it stages the other's - 44.13 GiB
# restored + 19.09 GiB restored + 6.4 GiB trained + 25.5 GiB staged, the 95.0 GiB
# footprint that went read-only on 2026-10-19 (docs/handoff-sharding-design.md).
#
# A step is **skipped once all its seeds are published**, read from the account rather
# than assumed: otherwise a completed step re-pushes a session that skips every seed in
# its declaration and exits 4 ("completed without advancing the queue"), which is the
# no-op the driver reports correctly but the campaign should not ask for. The empty step
# (`""`) means the queue's whole declared block - correct for the null queue, whose
# single-arm payload fits one version (`shard_seeds=0`).
SESSION_STEPS=("15024-15030" "15031-15035")
CONTROL_STEPS=("15031-15035" "15024-15030")
NULL_STEPS=("")
# 1 = print what every queue and step resolves to and push nothing.
DRY_RUN=${DRY_RUN:-0}
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

# A step's seed spec ("A-B" or a comma list; empty = the whole declared block) as seed
# numbers. The block comes from ``kaggle_t4`` rather than being spelled here, so a
# re-reserved block cannot leave the launcher declaring the old one.
step_seeds() {
  PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/scripts" "$PYTHON" - "$1" <<'PY'
import sys

import kaggle_t4

spec = (sys.argv[1] or "").strip() or kaggle_t4.SEED_BLOCK
if "-" in spec and "," not in spec:
    start, end = spec.split("-", 1)
    values = range(int(start), int(end) + 1)
else:
    values = (int(value) for value in spec.split(",") if value.strip())
print(" ".join(str(value) for value in values))
PY
}

# The handoff shards a step resolves to, one ``<dataset>:<seed span>`` per line. The same
# rule the driver and the publish path use (``act.ops.kaggle_remote``), read here because
# the launcher is what picks the declarations - a step that spans two shards must be
# refused before it costs a session, not discovered at 95 GiB.
step_shards() {
  PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/scripts" "$PYTHON" - "$1" "$2" <<'PY'
import sys

import kaggle_t4
from act.ops.kaggle_remote import shard_datasets, shard_index
from act.training.v0_runner import DECLARED_SEEDS

queue = kaggle_t4.QUEUES[sys.argv[1]]
spec = (sys.argv[2] or "").strip()
if not spec:
    seeds = tuple(DECLARED_SEEDS)
elif "-" in spec and "," not in spec:
    start, end = spec.split("-", 1)
    seeds = tuple(range(int(start), int(end) + 1))
else:
    seeds = tuple(int(value) for value in spec.split(",") if value.strip())
for index, (dataset, _members) in enumerate(
    shard_datasets(queue.dataset, DECLARED_SEEDS, queue.shard_seeds)
):
    declared = tuple(
        seed
        for seed in seeds
        if shard_index(seed, DECLARED_SEEDS, queue.shard_seeds) == index
    )
    if not declared:
        continue
    if declared == tuple(range(declared[0], declared[-1] + 1)):
        span = f"{declared[0]}-{declared[-1]}"
    else:
        span = ",".join(str(seed) for seed in declared)
    print(f"{dataset}:{span}")
PY
}

# Is every seed of this step already published? 0 = yes (skip the step), 1 = no, or the
# listing could not be read - and an unreadable listing must not silently skip work, so
# the driver is left to decide. ``campaign_health --json`` is the account's own answer.
step_is_complete() {
  local queue=$1 spec=$2 published seeds seed
  # The status is deliberately not tested: campaign_health's exit code is the queue's
  # *verdict* (4 is COMPLETE, which is what this function is looking for), so reading it
  # as failure would invert the answer. An unreadable listing leaves this empty, which
  # returns 1 below and lets the driver decide rather than skipping work.
  published=$(PYTHONPATH="$REPO_ROOT/src" "$PYTHON" "$REPO_ROOT/scripts/campaign_health.py" \
    --queue "$queue" --json 2>/dev/null \
    | "$PYTHON" -c 'import json, sys; print(" ".join(map(str, json.load(sys.stdin)["seeds"]["complete"])))' \
    2>/dev/null)
  seeds=$(step_seeds "$spec" 2>/dev/null) || return 1
  [[ -n "$seeds" && -n "$published" ]] || return 1
  for seed in $seeds; do
    [[ " $published " == *" $seed "* ]] || return 1
  done
  return 0
}

if [[ ! -x "$PYTHON" ]]; then
  echo "$(stamp) no interpreter at $PYTHON" >&2
  exit 1
fi

echo "$(stamp) campaign start: queues ${QUEUES[*]}${DRY_RUN:+ (dry run: $DRY_RUN)}"

for queue in "${QUEUES[@]}"; do
  # Print the durable state at every queue boundary. The campaign's two 2026-09 losses were
  # expensive partly because nothing said where it had got to, and this report reads the
  # published dataset, the kernel status and the quota rather than anything local - so the
  # campaign's own log carries a record of what it was doing even when a driver dies.
  # The window the sessions will actually use, so the quota verdict is stated against it: with
  # HOURS fitted to a short week, a report testing the kernel's declared 7 h would print
  # "cannot fit a 7.0 h session" while the driver is about to push a 6.5 h one.
  # campaign_health has no `verdict` queue (it reports on the three that train), so the
  # report is printed only for those.
  if [[ "$queue" != "verdict" ]]; then
    health=(--queue "$queue")
    if [[ -n "$HOURS" ]]; then
      health+=(--window "$HOURS")
    fi
    PYTHONPATH="$REPO_ROOT/src" "$PYTHON" "$REPO_ROOT/scripts/campaign_health.py" \
      "${health[@]}" 2>&1 | sed 's/^/  /' || true
  fi
  # `verdict` is a one-shot CPU job with no quota gate and no seed declaration of its
  # own; it is the last step, after the block its clauses read is complete.
  if [[ "$queue" == "verdict" ]]; then
    if (( DRY_RUN )); then
      echo "$(stamp) verdict: (dry run) would run: kaggle_queue.py --queue verdict"
      continue
    fi
    PYTHONPATH="$REPO_ROOT/src" "$PYTHON" "$DRIVER" --queue "verdict"
    code=$?
    # The verdict's exit code *is* the decision (runbook section 3): 0 PASS, 1 FAIL,
    # 2 PASS with qualification, 3 no verdict produced. A FAIL is an answer, not a
    # fault - re-running it would ask the same question of the same artifacts - so
    # only 3 (an incomplete declared block) stops.
    case "$code" in
      0 | 1 | 2)
        echo "$(stamp) verdict: decision produced (exit $code)"
        ;;
      *)
        echo "$(stamp) verdict: no verdict produced (exit $code) - the declared block is incomplete" >&2
        exit "$code"
        ;;
    esac
    continue
  fi

  case "$queue" in
    session) steps=("${SESSION_STEPS[@]}") ;;
    control) steps=("${CONTROL_STEPS[@]}") ;;
    null) steps=("${NULL_STEPS[@]}") ;;
    *) steps=("") ;;
  esac
  queue_done=0
  for seeds in "${steps[@]}"; do
    label=${seeds:-"the declared block"}
    if (( DRY_RUN )); then
      echo "$(stamp) $queue: step $label"
    fi
    # Refuse a declaration that spans two handoff shards before it costs a session:
    # the restore downloads both while the push stages one, which is the 95.0 GiB
    # footprint that went read-only on 2026-10-19. The declaration is where the fix
    # lives, so this is a launcher error, not a session's.
    span=$(step_shards "$queue" "$seeds") || {
      echo "$(stamp) $queue: could not resolve the shards of step '$label'; stopping" >&2
      exit 7
    }
    shard_count=$(printf '%s\n' "$span" | sed '/^$/d' | wc -l)
    if (( shard_count > 1 )); then
      echo "$(stamp) $queue: step '$label' spans $shard_count handoff shards:" >&2
      while IFS= read -r shard; do
        [[ -n "$shard" ]] || continue
        printf '  %s -> declare it alone with --seeds %s\n' \
          "${shard%%:*}" "${shard##*:}" >&2
      done <<< "$span"
      echo "  one session declares one shard (a two-shard declaration holds one shard's" >&2
      echo "  restored bytes while it stages the other's - the 95.0 GiB footprint that went" >&2
      echo "  read-only on 2026-10-19); split it into one ${queue^^}_STEPS entry per shard" >&2
      exit 7
    fi
    attempt=0
    while :; do
      if step_is_complete "$queue" "$seeds"; then
        echo "$(stamp) $queue: step $label already published - skipping"
        break
      fi
      attempt=$((attempt + 1))
      if (( attempt > MAX_ATTEMPTS )); then
        echo "$(stamp) $queue: gave up on step $label after $MAX_ATTEMPTS attempts" >&2
        exit 5
      fi
      echo "$(stamp) $queue: attempt $attempt (step $label)"
      if (( DRY_RUN )); then
        echo "  (dry run) would run: kaggle_queue.py --queue $queue${seeds:+ --seeds $seeds} --max-sessions 1 --wait-for-quota${HOURS:+ --hours $HOURS}"
        break
      fi
      # One push is one session, and the step is re-checked before each: so a step
      # that finishes leaves no session to skip every seed and exit 4, and a queue
      # that needs several sessions for one declaration (the null block) loops here.
      if [[ -n "$HOURS" ]]; then
        echo "$(stamp) $queue: declaring a $HOURS h session window"
        PYTHONPATH="$REPO_ROOT/src" "$PYTHON" "$DRIVER" --queue "$queue" \
          ${seeds:+--seeds "$seeds"} --max-sessions 1 --hours "$HOURS" --wait-for-quota
      else
        PYTHONPATH="$REPO_ROOT/src" "$PYTHON" "$DRIVER" --queue "$queue" \
          ${seeds:+--seeds "$seeds"} --max-sessions 1 --wait-for-quota
      fi
      code=$?
      case "$code" in
        0)
          echo "$(stamp) $queue: complete"
          queue_done=1
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
    (( queue_done )) && break
  done
done

echo "$(stamp) campaign finished: all three queues complete and the verdict produced"
