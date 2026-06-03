#!/usr/bin/env bash
# tools/run_full_ood_eval.sh — Evaluate ALL trained models on ALL OOD variants.
# Designed to be run after all training sessions complete.
#
# Usage:
#   bash tools/run_full_ood_eval.sh
#   bash tools/run_full_ood_eval.sh -n 200 -w 25
#   bash tools/run_full_ood_eval.sh -e my_env -o data/full_ood_eval.txt
set -euo pipefail

EPISODES=200
WORKERS=25
OUT=""
CONDA_ENV="parafoil_train_basic"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--episodes) EPISODES="$2"; shift 2;;
    -w|--workers)  WORKERS="$2"; shift 2;;
    -o|--out)      OUT="$2"; shift 2;;
    -e|--env)      CONDA_ENV="$2"; shift 2;;
    -h|--help)
      cat <<EOF
Usage: $(basename "$0") [options]

Evaluate all trained models on in-distribution and OOD variants.
Expects checkpoints in data/logs/<run_name>/model/best_model.pth

Options:
  -n N           Episodes per task (default: 200)
  -w N           Parallel workers (default: 25)
  -o PATH        Output file (default: data/full_ood_eval_<timestamp>.txt)
  -e ENV         Conda environment (default: parafoil_train_basic)
EOF
      exit 0;;
    *) shift;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "$OUT" ]]; then
  TS="$(date +'%Y-%m-%d_%H-%M-%S')"
  OUT="$REPO_ROOT/data/full_ood_eval_${TS}.txt"
fi
mkdir -p "$(dirname "$OUT")"

# Activate conda
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

# Auto-discover all best_model.pth files
declare -A MODELS
for model_dir in "$REPO_ROOT"/data/logs/*/model/; do
  if [[ -f "$model_dir/best_model.pth" ]]; then
    run_name="$(basename "$(dirname "$model_dir")")"
    MODELS["$run_name"]="$model_dir/best_model.pth"
  fi
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "ERROR: No trained models found in data/logs/*/model/best_model.pth" >&2
  exit 1
fi

# OOD task variants
TASKS=(
  "in_distribution=flare_landing.yml 42"
  "ood_strong=flare_landing_ood_strong.yml 123"
  "ood_gusty=flare_landing_ood_gusty.yml 124"
  "ood_combined=flare_landing_ood_combined.yml 125"
  "ood_altitude=flare_landing_ood_altitude.yml 126"
  "ood_shear=flare_landing_ood_shear.yml 127"
)

{
  echo "================================================================"
  echo " Full OOD evaluation report"
  echo " Models       : ${#MODELS[@]} checkpoints"
  echo " Episodes/task: $EPISODES   Workers: $WORKERS"
  echo " Started      : $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "================================================================"
} | tee "$OUT"

for run_name in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort); do
  model="${MODELS[$run_name]}"
  echo "" | tee -a "$OUT"
  echo "================================================================" | tee -a "$OUT"
  echo " Model: $run_name" | tee -a "$OUT"
  echo " Path : $model" | tee -a "$OUT"
  echo "================================================================" | tee -a "$OUT"

  for entry in "${TASKS[@]}"; do
    TAG="${entry%%=*}"
    rest="${entry#*=}"
    TASK="${rest%% *}"
    SEED="${rest##* }"

    echo "" | tee -a "$OUT"
    echo "----------------------------------------------------------------" | tee -a "$OUT"
    echo " [$TAG]  task=$TASK  seed=$SEED" | tee -a "$OUT"
    echo "----------------------------------------------------------------" | tee -a "$OUT"

    python "$REPO_ROOT/tools/evaluate_model.py" \
        --model "$model" \
        --episodes "$EPISODES" \
        --workers "$WORKERS" \
        --seed "$SEED" \
        --task "$TASK" \
        2>&1 | tee -a "$OUT" | tail -25
  done
done

echo "" | tee -a "$OUT"
echo "================================================================" | tee -a "$OUT"
echo " Finished: $(date -u +'%Y-%m-%dT%H:%M:%SZ')" | tee -a "$OUT"
echo " Full log saved to: $OUT" | tee -a "$OUT"
echo "================================================================" | tee -a "$OUT"
