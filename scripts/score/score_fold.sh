#!/usr/bin/env bash
#
# Score one Ego4D fold. Run inside an interactive GPU session, from the repository root:
#
#   srun --gres=gpu:rtx5090 --cpus-per-task=8 --mem=32G --time=12:00:00 --pty bash
#   ./scripts/score/score_fold.sh <fold>        e.g. ./scripts/score/score_fold.sh 1

set -euo pipefail

FOLD="${1:?usage: ./scripts/score/score_fold.sh <fold>}"

source scripts/env.sh

python3 main.py \
  "$EGO4D_DIR/fold$FOLD"/*.mp4 \
  --model-size giant \
  --sample-fps 4 \
  --context-frames 16 \
  --target-frames 8 \
  --stride-duration 6.0 \
  --segments-dir "segments$FOLD" \
  --output "results$FOLD.csv"
