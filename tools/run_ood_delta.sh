#!/usr/bin/env bash
# tools/run_ood_delta.sh — Evaluate only the two new OOD variants
# (altitude, shear) for each of the three released checkpoints. Used to
# extend the existing tab:ood without re-running the original four
# in-distribution / OOD-strong / OOD-gusty / OOD-combined variants.
#
# Usage:
#   bash tools/run_ood_delta.sh
#
# Output: data/eval_ood_delta_<timestamp>.txt
#
# Designed to be run inside a detached tmux session:
#   tmux new-session -d -s ood_delta 'bash tools/run_ood_delta.sh'
#   tmux attach -t ood_delta            # to peek
#   tmux ls                              # to confirm it's still running
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Activate conda for headless tmux sessions
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate parafoil_train_basic
fi

OUT="data/eval_ood_delta_$(date +%F_%H-%M-%S).txt"
mkdir -p data

# tag -> checkpoint
declare -A RUNS=(
  [her_wdr]="data/logs/2026-05-11_21-59-36/model/best_model.pth"
  [vanilla]="data/logs/2026-05-13_10-08-48/model/best_model.pth"
  [her_naive]="data/logs/2026-05-13_15-40-09/model/best_model.pth"
)

# Format: "VARIANT_TAG TASK_YML SEED"
DELTAS=(
  "ood_altitude flare_landing_ood_altitude.yml 126"
  "ood_shear    flare_landing_ood_shear.yml    127"
)

{
  echo "================================================================"
  echo " OOD delta evaluation (altitude + shear only)"
  echo " Started: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "================================================================"

  for tag in her_wdr vanilla her_naive; do
    if [[ ! -f "${RUNS[$tag]}" ]]; then
      echo
      echo "[WARN] checkpoint not found, skipping: ${RUNS[$tag]}" >&2
      continue
    fi
    for entry in "${DELTAS[@]}"; do
      read -r variant task seed <<< "$entry"
      echo
      echo "----------------------------------------------------------------"
      echo "[$tag / $variant]  task=$task  seed=$seed  ckpt=${RUNS[$tag]}"
      echo "----------------------------------------------------------------"
      python tools/evaluate_model.py \
          --model "${RUNS[$tag]}" \
          --episodes 200 --workers 25 \
          --seed "$seed" --task "$task"
    done
  done

  echo
  echo "================================================================"
  echo " Finished: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo " Saved to: $OUT"
  echo "================================================================"
} 2>&1 | tee "$OUT"
