#!/usr/bin/env bash
# tools/copy_paper_figs.sh — copy the TensorBoard-generated plots into the
# paper directory under the filenames that main.tex expects, so the .tex
# compiles against the real training curves instead of the synthetic ones.
#
# Usage:
#   bash tools/copy_paper_figs.sh <PATH_TO_PAPER_DIR>
#
# Example:
#   # local
#   bash tools/copy_paper_figs.sh ../Duzehang_parafoil_HTR/paper
#   # or copy to a remote machine via scp
#   bash tools/copy_paper_figs.sh user@workstation:/path/to/paper
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $(basename "$0") <PATH_TO_PAPER_DIR>" >&2
  exit 2
fi

DEST="$1"
SRC="paper_figs"

if [[ ! -d "$SRC" ]]; then
  echo "ERROR: $SRC/ not found. Run tools/plot_comparison.py first." >&2
  exit 1
fi

# Map TB tag PNGs to the fig filenames main.tex references.
declare -A MAP=(
  ["RL_Episode_Average_Reward.png"]="fig2_reward_curve.png"
  ["RL_real_success_rate.png"]="fig3_success_rate.png"
  ["RL_her_rate.png"]="fig6_ablation_mix.png"
)

# Detect local vs remote destination.
if [[ "$DEST" == *":"* ]]; then
  for src_name in "${!MAP[@]}"; do
    dst_name="${MAP[$src_name]}"
    echo "  scp $SRC/$src_name -> $DEST/$dst_name"
    scp "$SRC/$src_name" "${DEST%/}/$dst_name"
  done
else
  mkdir -p "$DEST"
  for src_name in "${!MAP[@]}"; do
    dst_name="${MAP[$src_name]}"
    echo "  cp $SRC/$src_name -> $DEST/$dst_name"
    cp "$SRC/$src_name" "${DEST%/}/$dst_name"
  done
fi

echo
echo "Done. The paper now references the measured training curves."
