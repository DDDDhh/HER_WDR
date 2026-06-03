#!/usr/bin/env bash
# tools/run_full_comparison.sh — launch ALL comparison runs for the ablation
# study. Covers:
#   Group A: HER variant comparison (HER-WDR vs Vanilla PPO vs Naive HER)
#   Group B: Quality filter ablation (prob/step/alt only)
#   Group C: WDR noise ablation (no/low/high noise)
#   Group D: Dense vs Sparse reward (2×2 design)
#
# Usage:
#   bash tools/run_full_comparison.sh              # run all groups
#   bash tools/run_full_comparison.sh -g A         # run only group A
#   bash tools/run_full_comparison.sh -g "A B"     # run groups A and B
#   bash tools/run_full_comparison.sh -e my_env    # conda env override
set -euo pipefail

CONDA_ENV="parafoil_train_basic"
RUN_GROUPS="A B C D"

while [[ $# -gt 0 ]]; do
  key="$1"
  case "$key" in
    -e|--env)    CONDA_ENV="$2"; shift 2;;
    -g|--groups) RUN_GROUPS="$2"; shift 2;;
    -h|--help)
      cat <<EOF
Usage: $(basename "$0") [options]

Launches tmux sessions for each ablation config:
  Group A (HER variants):
    her_wdr        - config/framework_ppo.yml          (HER-WDR + PPO)
    vanilla_ppo    - config/framework_ppo_baseline.yml (vanilla PPO)
    her_naive      - config/framework_ppo_her_standard.yml (naive HER)

  Group B (Quality filter ablation):
    her_prob_only  - config/framework_ppo_her_prob_only.yml
    her_step_only  - config/framework_ppo_her_step_only.yml
    her_alt_only   - config/framework_ppo_her_alt_only.yml

  Group C (WDR noise ablation):
    wdr_no_noise   - config/framework_ppo_wdr_no_noise.yml
    wdr_low_noise  - config/framework_ppo_wdr_low_noise.yml
    wdr_high_noise - config/framework_ppo_wdr_high_noise.yml

  Group D (Dense vs Sparse reward):
    dense_her      - config/framework_ppo_dense_her.yml
    dense_baseline - config/framework_ppo_dense_baseline.yml

Options:
  -e CONDA_ENV   conda environment name (default: parafoil_train_basic)
  -g "A B C"     space-separated group tags to run (default: "A B C D")
EOF
      exit 0;;
    *) shift;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START="$SCRIPT_DIR/start_train.sh"

# Define all configs with group tags
declare -A CONFIGS=(
  # Group A: HER variants
  [her_wdr]="A:config/framework_ppo.yml"
  [vanilla_ppo]="A:config/framework_ppo_baseline.yml"
  [her_naive]="A:config/framework_ppo_her_standard.yml"
  # Group B: Quality filter ablation
  [her_prob_only]="B:config/framework_ppo_her_prob_only.yml"
  [her_step_only]="B:config/framework_ppo_her_step_only.yml"
  [her_alt_only]="B:config/framework_ppo_her_alt_only.yml"
  # Group C: WDR noise ablation
  [wdr_no_noise]="C:config/framework_ppo_wdr_no_noise.yml"
  [wdr_low_noise]="C:config/framework_ppo_wdr_low_noise.yml"
  [wdr_high_noise]="C:config/framework_ppo_wdr_high_noise.yml"
  # Group D: Dense vs Sparse
  [dense_her]="D:config/framework_ppo_dense_her.yml"
  [dense_baseline]="D:config/framework_ppo_dense_baseline.yml"
)

echo "================================================================"
echo " Full ablation study launcher"
echo " Groups: $RUN_GROUPS"
echo " Conda env: $CONDA_ENV"
echo "================================================================"

started=0
for tag in "${!CONFIGS[@]}"; do
  entry="${CONFIGS[$tag]}"
  group="${entry%%:*}"
  cfg="${entry#*:}"

  # Check if this group is requested
  if [[ ! " $RUN_GROUPS " =~ " $group " ]]; then
    continue
  fi

  echo "[run_full_comparison.sh] starting session '$tag' (group $group) with $cfg"
  bash "$START" -s "$tag" -e "$CONDA_ENV" -c "$cfg" || true
  started=$((started + 1))
done

echo
echo "[run_full_comparison.sh] $started sessions started."
echo "  tmux ls"
echo "  tmux attach -t <session_name>"
