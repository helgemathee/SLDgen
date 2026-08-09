#!/usr/bin/env bash
#
# Stop the SLDgen service, without losing GPU time.
#
# The order matters. The worker is signalled first and given time to shepherd
# its running segment to a checkpoint: SLDgen finishes the current iteration,
# writes a checkpoint, writes state.json and exits 0 (Spec 2 SS2.1). The job then
# resumes from there next time rather than restarting from epoch 0, so stopping
# a 3800-iteration run costs seconds instead of fifteen minutes.
#
#   ./stop.sh                stop gracefully, waiting up to 180s for a checkpoint
#   ./stop.sh --now          do not wait for the segment; it resumes from its
#                            last checkpoint instead of its current epoch
#   ./stop.sh --keep-session leave the tmux session open (panes stay readable)

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SLDGEN_TMUX_SESSION:-sldgen-service}"
WORK_ROOT="${SLDGEN_WORK_ROOT:-$REPO_ROOT/work}"
RUN_DIR="$WORK_ROOT/run"
GRACE="${SLDGEN_STOP_GRACE:-180}"

KEEP_SESSION=0
for argument in "$@"; do
  case "$argument" in
    --now)           GRACE=5 ;;
    --keep-session)  KEEP_SESSION=1 ;;
    -h|--help)       awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *) echo "stop.sh: unknown option '$argument'" >&2; exit 2 ;;
  esac
done

# A pidfile is only evidence; the process may have died since it was written.
running() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }

read_pid() {
  local file="$1"
  [ -f "$file" ] || return 0
  local pid
  pid="$(cat "$file" 2>/dev/null || true)"
  running "$pid" && printf '%s' "$pid"
}

worker_pid="$(read_pid "$RUN_DIR/worker.pid")"
api_pid="$(read_pid "$RUN_DIR/api.pid")"

if [ -z "$worker_pid" ] && [ -z "$api_pid" ] && ! tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Nothing is running."
  exit 0
fi

# -- the worker, gracefully -------------------------------------------------

if [ -n "$worker_pid" ]; then
  echo "==> asking the worker to stop (pid $worker_pid)"
  kill -TERM "$worker_pid" 2>/dev/null || true

  waited=0
  reported=0
  while running "$worker_pid" && [ "$waited" -lt "$GRACE" ]; do
    if [ "$reported" -eq 0 ] && [ "$waited" -ge 3 ]; then
      echo "    waiting for the running segment to checkpoint (up to ${GRACE}s)"
      echo "    it finishes the current iteration first, so this is normal"
      reported=1
    fi
    sleep 1
    waited=$((waited + 1))
  done

  if running "$worker_pid"; then
    # Past the grace period something is wedged. SIGKILL loses this segment's
    # progress since its last checkpoint -- not the job, which the next worker
    # start recovers from `latest.pt` (Spec 2 SS8).
    echo "    still up after ${GRACE}s; killing it"
    kill -KILL "$worker_pid" 2>/dev/null || true
    sleep 1
  else
    echo "    worker stopped cleanly after ${waited}s"
  fi
else
  echo "==> the worker was not running"
fi

# A segment is its own process group, so it can outlive a killed worker. Leaving
# one holding the GPU would make the next start fail with an OOM that looks like
# somebody else's fault.
if pgrep -f "sldgen\.py .*--output-dir $WORK_ROOT" >/dev/null 2>&1; then
  echo "==> stopping an orphaned SLDgen segment"
  pkill -TERM -f "sldgen\.py .*--output-dir $WORK_ROOT" || true
  sleep 3
  pkill -KILL -f "sldgen\.py .*--output-dir $WORK_ROOT" 2>/dev/null || true
fi

# -- the API ----------------------------------------------------------------

if [ -n "$api_pid" ]; then
  echo "==> stopping the API (pid $api_pid)"
  kill -TERM "$api_pid" 2>/dev/null || true
  waited=0
  while running "$api_pid" && [ "$waited" -lt 15 ]; do
    sleep 1
    waited=$((waited + 1))
  done
  running "$api_pid" && kill -KILL "$api_pid" 2>/dev/null || true
else
  echo "==> the API was not running"
fi

rm -f "$RUN_DIR/worker.pid" "$RUN_DIR/api.pid"

# -- the session ------------------------------------------------------------

if tmux has-session -t "$SESSION" 2>/dev/null; then
  if [ "$KEEP_SESSION" -eq 1 ]; then
    echo "==> leaving the '$SESSION' session open (tmux attach -t $SESSION)"
  else
    echo "==> closing the '$SESSION' session"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
  fi
fi

echo
echo "Stopped. Any job that was running is checkpointed and will resume where it"
echo "left off when you next run ./start.sh."
