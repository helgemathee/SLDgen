#!/usr/bin/env bash
#
# Start the SLDgen service in a detached tmux session.
#
# Detached is the point: the session belongs to the tmux server, not to your SSH
# connection, so you can start it, close the terminal, go home, and reattach
# tomorrow with the jobs still running. See docs/RUNNING.md if tmux is new to
# you -- there are only four keystrokes worth knowing.
#
#   ./start.sh              build the UI if it is stale, then start everything
#   ./start.sh --no-build   skip the UI build (fastest restart)
#   ./start.sh --rebuild    force a UI rebuild
#
# Everything is configurable by environment variable; the defaults are what this
# machine wants.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

SESSION="${SLDGEN_TMUX_SESSION:-sldgen-service}"
WORK_ROOT="${SLDGEN_WORK_ROOT:-$REPO_ROOT/work}"
API_HOST="${SLDGEN_API_HOST:-127.0.0.1}"
API_PORT="${SLDGEN_API_PORT:-8765}"

# The worker and every SLDgen segment run in the conda env, which is the only
# place torch, pydiffvg and wiregrad exist. The API deliberately does not: a
# torch-free venv restarts in under a second (Spec 2 SS1).
CONDA_PYTHON="${SLDGEN_PYTHON:-/home/helge/miniforge3/envs/sldgen/bin/python}"
API_PYTHON="${SLDGEN_API_PYTHON:-$REPO_ROOT/.venv-service/bin/python}"
CONCORDE="${CONCORDE_PATH:-/home/helge/src/concorde/TSP/concorde}"

RUN_DIR="$WORK_ROOT/run"
WORKER_PID="$RUN_DIR/worker.pid"
API_PID="$RUN_DIR/api.pid"

BUILD_MODE=auto
for argument in "$@"; do
  case "$argument" in
    --no-build) BUILD_MODE=never ;;
    --rebuild)  BUILD_MODE=force ;;
    -h|--help)  awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *) echo "start.sh: unknown option '$argument'" >&2; exit 2 ;;
  esac
done

die() { echo "start.sh: $*" >&2; exit 1; }

# IPv6 literals need brackets to be a URL at all.
url_for() {
  case "$1" in
    *:*) echo "http://[$1]:$API_PORT" ;;
    *)   echo "http://$1:$API_PORT" ;;
  esac
}

# -- preflight --------------------------------------------------------------
#
# Every one of these has bitten someone. Failing here with the reason beats
# failing inside a tmux pane you have to go looking for.

command -v tmux >/dev/null || die "tmux is not installed (sudo apt install tmux)"
[ -x "$CONDA_PYTHON" ] || die "the conda env's python is missing: $CONDA_PYTHON
  The worker and SLDgen itself need it. Create it per CLAUDE.md, or set SLDGEN_PYTHON."
[ -x "$API_PYTHON" ] || die "the API venv is missing: $API_PYTHON
  Create it:  python -m venv .venv-service && .venv-service/bin/pip install -r requirements-service.txt"
[ -x "$CONCORDE" ] || echo "start.sh: warning: Concorde not found at $CONCORDE
  TSP initialisation will fail a few seconds into every job. Set CONCORDE_PATH." >&2

# SLDGEN_API_HOST is a comma-separated list, so "the tailnet and the LAN" is one
# variable rather than two services. The tokens below expand to whatever this
# machine's addresses are right now -- a DHCP lease or a tailnet address is not
# something anyone should have to paste in by hand each time.
#
#   loopback   127.0.0.1
#   tailscale  the tailnet address (`tailscale ip -4`)
#   lan        the first RFC1918 address on a real interface, docker0 excluded
#
#   SLDGEN_API_HOST=loopback,tailscale,lan ./start.sh
#
expand_host_token() {
  case "$1" in
    loopback) echo "127.0.0.1" ;;
    tailscale)
      command -v tailscale >/dev/null || die "SLDGEN_API_HOST asks for 'tailscale', but the tailscale CLI is not installed."
      # `|| true`: a stopped daemon exits non-zero, and under `set -e` that
      # would kill start.sh here with no message at all. Returning empty lets
      # the caller below say which token failed and how to check it.
      { tailscale ip -4 2>/dev/null | head -n1; } || true
      ;;
    lan)
      # scope global drops loopback; the grep drops docker/tailscale/bridges,
      # leaving the address the router actually handed this machine. Empty
      # output (no match) is a non-zero grep, hence `|| true` again.
      { ip -4 -o addr show scope global 2>/dev/null \
        | grep -vE '^\s*[0-9]+:\s+(docker|br-|virbr|tailscale|veth)' \
        | grep -oE '(^|\s)(192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[01])\.)[0-9.]+' \
        | tr -d ' ' | head -n1; } || true
      ;;
    *) echo "$1" ;;
  esac
}

resolved_hosts=""
IFS=',' read -ra host_entries <<< "$API_HOST"
for entry in "${host_entries[@]}"; do
  entry="$(echo "$entry" | tr -d '[:space:]')"
  [ -n "$entry" ] || continue
  case "$entry" in
    0.0.0.0|::|'*')
      die "refusing to bind $entry. The API has no authentication, so the bind
  address is the access control. List the addresses you want instead, e.g.
  SLDGEN_API_HOST=loopback,tailscale,lan"
      ;;
  esac
  resolved="$(expand_host_token "$entry")"
  [ -n "$resolved" ] || die "SLDGEN_API_HOST token '$entry' did not resolve to an address.
  For 'tailscale', check \`tailscale status\`. For 'lan', check \`ip -4 -o addr show scope global\`."
  case ",$resolved_hosts," in
    *",$resolved,"*) continue ;;   # already listed
  esac
  resolved_hosts="${resolved_hosts:+$resolved_hosts,}$resolved"
done
[ -n "$resolved_hosts" ] || die "SLDGEN_API_HOST is empty."
API_HOST="$resolved_hosts"

# The first address is "the" one: what the health check polls and what the
# banner offers as a link.
API_HOST_PRIMARY="${API_HOST%%,*}"

mkdir -p "$WORK_ROOT" "$RUN_DIR"

# -- build the UI -----------------------------------------------------------

web_is_stale() {
  local dist="$REPO_ROOT/sldgen_web/dist/index.html"
  [ -f "$dist" ] || return 0
  # Any source newer than the build means the build is out of date.
  [ -n "$(find "$REPO_ROOT/sldgen_web/src" "$REPO_ROOT/sldgen_web/index.html" \
            "$REPO_ROOT/sldgen_web/package.json" "$REPO_ROOT/sldgen_web/vite.config.ts" \
            -newer "$dist" -print -quit 2>/dev/null)" ]
}

build_web() {
  command -v npm >/dev/null || die "npm is not installed, and the UI needs building.
  Install Node, or run with --no-build to start the API without the UI."
  echo "==> building the web UI"
  ( cd "$REPO_ROOT/sldgen_web"
    [ -d node_modules ] || npm install --no-audit --no-fund
    npm run build )
}

case "$BUILD_MODE" in
  force) build_web ;;
  auto)  if web_is_stale; then build_web; else echo "==> web UI is up to date"; fi ;;
  never) echo "==> skipping the web UI build" ;;
esac

# -- start ------------------------------------------------------------------

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo
  echo "The '$SESSION' session is already running. Nothing to do."
  echo "  Attach with:  tmux attach -t $SESSION"
  echo "  Restart with: ./stop.sh && ./start.sh"
  exit 0
fi

# Exported into every pane, so the worker and the API cannot disagree about the
# root -- which is the one thing that would silently break them (Spec 2 SS3).
# The conda env is entered by calling its interpreter directly, so the variables
# `conda activate` would normally set are declared here instead. CONCORDE_PATH
# missing is the classic version of this: TSP init fails several seconds into
# every job with an unhelpful error.
common_env=(
  "SLDGEN_WORK_ROOT=$WORK_ROOT"
  "PYTHONPATH=$REPO_ROOT"
  "PYTHONUNBUFFERED=1"
)
worker_env=(
  "${common_env[@]}"
  "CONCORDE_PATH=$CONCORDE"
  "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  "TOKENIZERS_PARALLELISM=false"
  "SLDGEN_PYTHON=$CONDA_PYTHON"
)
api_env=(
  "${common_env[@]}"
  "SLDGEN_API_HOST=$API_HOST"
  "SLDGEN_API_PORT=$API_PORT"
  "SLDGEN_PYTHON=$CONDA_PYTHON"
)

rm -f "$WORKER_PID" "$API_PID"

# Build a `export A=1 B=2; ...` prefix, quoted so a path with a space cannot
# split into two variables.
exports() {
  local assignment
  for assignment in "$@"; do printf 'export %q; ' "$assignment"; done
}

# `echo $$ > pidfile; exec ...` records the daemon's own PID: exec replaces the
# shell without changing the PID, so stop.sh signals the real process rather
# than a wrapper. That is what makes a graceful shutdown actually reach the
# segment and give it time to checkpoint.
worker_cmd="echo \$\$ > $(printf '%q' "$WORKER_PID"); $(exports "${worker_env[@]}") exec $(printf '%q' "$CONDA_PYTHON") -m sldgen_worker"
api_cmd="echo \$\$ > $(printf '%q' "$API_PID"); $(exports "${api_env[@]}") exec $(printf '%q' "$API_PYTHON") -m sldgen_api"

echo "==> starting the '$SESSION' session"
tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" -n service "$worker_cmd"
tmux split-window -t "$SESSION:service" -v -c "$REPO_ROOT" "$api_cmd"
tmux split-window -t "$SESSION:service" -v -c "$REPO_ROOT"
tmux select-layout -t "$SESSION:service" even-vertical

# A dead daemon must leave its error on screen rather than closing the pane and
# taking the explanation with it.
tmux set-option -p -t "$SESSION:service.0" remain-on-exit on
tmux set-option -p -t "$SESSION:service.1" remain-on-exit on

tmux select-pane -t "$SESSION:service.0" -T "worker (GPU queue)"
tmux select-pane -t "$SESSION:service.1" -T "api  $(url_for "$API_HOST_PRIMARY")"
tmux select-pane -t "$SESSION:service.2" -T "shell"
tmux set-option -t "$SESSION" pane-border-status top 2>/dev/null || true
tmux select-pane -t "$SESSION:service.2"

# -- wait and report --------------------------------------------------------

printf '==> waiting for the API'
health=""
for _ in $(seq 1 60); do
  if health="$(curl -fsS "$(url_for "$API_HOST_PRIMARY")/api/health" 2>/dev/null)"; then
    break
  fi
  printf '.'
  sleep 0.5
done
echo

if [ -z "$health" ]; then
  echo
  echo "The API did not answer within 30s. Its output is in the middle pane:"
  echo "  tmux attach -t $SESSION"
  exit 1
fi

worker_alive=$(printf '%s' "$health" | grep -o '"worker_alive":[^,}]*' | cut -d: -f2)

web_urls=""
IFS=',' read -ra listed_hosts <<< "$API_HOST"
for listed in "${listed_hosts[@]}"; do
  web_urls="${web_urls}    Web UI      $(url_for "$listed")"$'\n'
done

cat <<BANNER

  SLDgen is running.

${web_urls%$'\n'}
    Work root   $WORK_ROOT
    Worker      $([ "$worker_alive" = "true" ] && echo "up" || echo "NOT UP -- see the top pane")

  Watch it:     tmux attach -t $SESSION
  Leave it:     press Ctrl-b then d   (it keeps running; you may close SSH)
  Stop it:      ./stop.sh

BANNER
