#!/usr/bin/env bash
# tools/run_ood_eval.sh — sequentially evaluate a trained checkpoint under the
# in-distribution task and three out-of-distribution wind variants. Designed
# for the comparison runs in the paper, so the launcher is conservative on
# concurrent worker count (25) to avoid page-file pressure when 100-worker
# training is also running.
#
# Usage:
#   bash tools/run_ood_eval.sh -m data/logs/<run>/model/best_model.pth
#   bash tools/run_ood_eval.sh -m ... -n 200 -w 25 -o data/eval_ood.txt
#
# All four evals share the same checkpoint, so any drop is purely OOD.
set -euo pipefail

MODEL=""
EPISODES=200
WORKERS=25
OUT=""
CONDA_ENV="parafoil_train_basic"

usage() {
  cat <<EOF
Usage: $(basename "$0") -m MODEL_PATH [options]

Options:
  -m PATH        Path to best_model.pth (required)
  -n N           Episodes per task (default: 200)
  -w N           Parallel workers (default: 25; keep <= cpu_count/2 on Windows)
  -o PATH        Output file for combined results (default: data/eval_<timestamp>.txt)
  -e ENV         Conda environment (default: parafoil_train_basic)
  -h             help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model)    MODEL="$2"; shift 2;;
    -n|--episodes) EPISODES="$2"; shift 2;;
    -w|--workers)  WORKERS="$2"; shift 2;;
    -o|--out)      OUT="$2"; shift 2;;
    -e|--env)      CONDA_ENV="$2"; shift 2;;
    -h|--help)     usage; exit 0;;
    *) echo "unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "ERROR: -m MODEL_PATH is required" >&2
  usage; exit 2
fi
if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: model not found: $MODEL" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "$OUT" ]]; then
  TS="$(date +'%Y-%m-%d_%H-%M-%S')"
  OUT="$REPO_ROOT/data/eval_ood_${TS}.txt"
fi
mkdir -p "$(dirname "$OUT")"

# Activate conda
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

# Each entry: TAG=TASK_YML SEED
TASKS=(
  "in_distribution=flare_landing.yml 42"
  "ood_strong_wind=flare_landing_ood_strong.yml 123"
  "ood_gusty=flare_landing_ood_gusty.yml 124"
  "ood_combined=flare_landing_ood_combined.yml 125"
  "ood_altitude=flare_landing_ood_altitude.yml 126"
  "ood_shear=flare_landing_ood_shear.yml 127"
)

{
  echo "================================================================"
  echo " OOD evaluation report"
  echo " Model        : $MODEL"
  echo " Episodes/task: $EPISODES   Workers: $WORKERS   Conda env: $CONDA_ENV"
  echo " Started      : $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "================================================================"
} | tee "$OUT"

for entry in "${TASKS[@]}"; do
  TAG="${entry%%=*}"
  rest="${entry#*=}"
  TASK="${rest%% *}"
  SEED="${rest##* }"

  echo "" | tee -a "$OUT"
  echo "----------------------------------------------------------------" | tee -a "$OUT"
  echo "[$TAG]  task=$TASK  seed=$SEED" | tee -a "$OUT"
  echo "----------------------------------------------------------------" | tee -a "$OUT"

  python "$REPO_ROOT/tools/evaluate_model.py" \
      --model "$MODEL" \
      --episodes "$EPISODES" \
      --workers "$WORKERS" \
      --seed "$SEED" \
      --task "$TASK" \
      2>&1 | tee -a "$OUT" | tail -25
done

echo "" | tee -a "$OUT"
echo "================================================================" | tee -a "$OUT"
echo " Finished: $(date -u +'%Y-%m-%dT%H:%M:%SZ')" | tee -a "$OUT"
echo " Full log saved to: $OUT" | tee -a "$OUT"
echo "================================================================" | tee -a "$OUT"
