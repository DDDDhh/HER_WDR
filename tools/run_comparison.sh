#!/usr/bin/env bash
# tools/run_comparison.sh — launch the three comparison runs in separate tmux
# sessions so the paper's experiment section can be reproduced end-to-end.
#
# Usage:
#   bash tools/run_comparison.sh
#   bash tools/run_comparison.sh -e parafoil_train_basic           # conda env override
#   bash tools/run_comparison.sh -t "her_v1 vanilla_v1 naive_v1"   # session-name prefixes
set -euo pipefail

CONDA_ENV="parafoil_train_basic"
TAGS=("her_ppo" "vanilla_ppo" "her_naive")

while [[ $# -gt 0 ]]; do
  key="$1"
  case "$key" in
    -e|--env)  CONDA_ENV="$2"; shift 2;;
    -t|--tags) read -ra TAGS <<< "$2"; shift 2;;
    -h|--help)
      cat <<EOF
Usage: $(basename "$0") [options]

Launches three tmux sessions, one per config:
  ${TAGS[0]}   - config/framework_ppo.yml          (HER-WDR + PPO, our method)
  ${TAGS[1]}   - config/framework_ppo_baseline.yml (vanilla PPO, no HER)
  ${TAGS[2]}   - config/framework_ppo_her_standard.yml (naive HER, no filters)

Options:
  -e CONDA_ENV   conda environment name (default: parafoil_train_basic)
  -t "A B C"     space-separated tags for the three sessions
EOF
      exit 0;;
    *) shift;;
  esac
done

CONFIGS=(
  "config/framework_ppo.yml"
  "config/framework_ppo_baseline.yml"
  "config/framework_ppo_her_standard.yml"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START="$SCRIPT_DIR/start_train.sh"

for i in 0 1 2; do
  TAG="${TAGS[$i]}"
  CFG="${CONFIGS[$i]}"
  echo "[run_comparison.sh] starting session '$TAG' with $CFG"
  bash "$START" -s "$TAG" -e "$CONDA_ENV" -c "$CFG"
done

echo
echo "[run_comparison.sh] all sessions started."
echo "  tmux ls"
echo "  tmux attach -t ${TAGS[0]}"
