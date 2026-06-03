#!/usr/bin/env bash
# tools/start_train.sh — launch HER+PPO training inside a tmux session.
#
# Designed for headless server use: ``bash tools/start_train.sh -s <name>``
# creates a detached tmux session named <name>, activates the conda env, and
# runs ``framework/framework_train.py`` with the configured framework yml.
#
# Reattach with: ``tmux attach -t <name>``.

set -euo pipefail

echo "[start_train.sh] version: 2026-05-11-flare-her-v1"
echo "[start_train.sh] SHELL=$SHELL BASH_VERSION=${BASH_VERSION:-<none>}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  -s SESSION         tmux session name (default: train1)
  -e CONDA_ENV       conda environment name (default: parafoil_train_basic)
  -p PYTHON          python executable (overrides conda activation)
  -c CONFIG          path to framework config yml
                     (default: config/framework_ppo.yml; exported as
                      FRAMEWORK_CONFIG_PATH for framework_train.py)
  -d                 attach instead of detach
  -h                 help

Examples:
  bash tools/start_train.sh -s exp1
  bash tools/start_train.sh -s exp1 -e parafoil_train_basic
  bash tools/start_train.sh -s exp1 -c config/framework_ppo_debug.yml
EOF
}

SESSION="train1"
CONDA_ENV="parafoil_train_basic"
PYTHON=""
CONFIG="config/framework_ppo.yml"
DETACH=1

while [[ $# -gt 0 ]]; do
  key="$1"
  case "$key" in
    -s|--session)  SESSION="$2"; shift 2;;
    -e|--env)      CONDA_ENV="$2"; shift 2;;
    -p|--python)   PYTHON="$2"; shift 2;;
    -c|--config)   CONFIG="$2"; shift 2;;
    -d|--no-detach) DETACH=0; shift;;
    -h|--help)     usage; exit 0;;
    *) shift;;  # tolerate unknown flags (forward-compat with older HER args)
  esac
done

# Resolve paths relative to repo root so the script can be invoked from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ "$CONFIG" != /* ]]; then
  CONFIG="$REPO_ROOT/$CONFIG"
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "[start_train.sh] ERROR: config not found: $CONFIG" >&2
  exit 1
fi

# Build the inner command. framework_train.py reads FRAMEWORK_CONFIG_PATH from
# the environment, falling back to config/framework_ppo.yml.
INNER_EXPORT="export FRAMEWORK_CONFIG_PATH=\"$CONFIG\";"

if [[ -n "$PYTHON" ]]; then
  INNER_CMD="$INNER_EXPORT cd \"$REPO_ROOT\" && \"$PYTHON\" framework/framework_train.py"
else
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    ACTIVATE_CMD="source $CONDA_BASE/etc/profile.d/conda.sh; conda activate $CONDA_ENV;"
  else
    ACTIVATE_CMD="conda activate $CONDA_ENV;"
  fi
  INNER_CMD="$ACTIVATE_CMD $INNER_EXPORT cd \"$REPO_ROOT\" && python framework/framework_train.py"
fi
RUN_CMD=(bash -lc "$INNER_CMD")

# Log metadata so the user can audit what was started.
LOG_DIR="$REPO_ROOT/data/runs"
mkdir -p "$LOG_DIR"
META_FILE="$LOG_DIR/last_start_info.txt"
{
  echo "session: $SESSION"
  echo "started_at: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "config: $CONFIG"
  echo "conda_env: $CONDA_ENV"
  echo "python: ${PYTHON:-<conda>}"
  echo "cmd: ${RUN_CMD[*]}"
} > "$META_FILE"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[start_train.sh] tmux session '$SESSION' already exists." >&2
  echo "  Attach with: tmux attach -t $SESSION" >&2
  exit 1
fi

if [[ "$DETACH" -eq 1 ]]; then
  echo "[start_train.sh] Starting detached tmux session: $SESSION"
  tmux new-session -d -s "$SESSION" "${RUN_CMD[@]}"
  echo "[start_train.sh] Started. Attach with: tmux attach -t $SESSION"
  echo "[start_train.sh] Logs: $LOG_DIR/$SESSION.log (tmux output is in the tmux pane)"
else
  echo "[start_train.sh] Starting tmux session (attached): $SESSION"
  exec tmux new-session -s "$SESSION" "${RUN_CMD[@]}"
fi
