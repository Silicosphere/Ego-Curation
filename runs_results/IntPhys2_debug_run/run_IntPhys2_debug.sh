#!/usr/bin/env bash

module load FFmpeg/7.0.2-GCCcore-13.3.0

export HF_HOME="/data/asem.a.abdelaziz/gp-2027a/huggingface_downloads"

export TORCH_HOME='/data/asem.a.abdelaziz/gp-2027a/torch_downloads'

source ../myenv/bin/activate

VIDEOS_DIR="/data/asem.a.abdelaziz/gp-2027a/IntPhys2_Data/Debug/Videos"

python3 main.py \
  "$VIDEOS_DIR/083861b17a8b3f4bef7e3aaad0b960318a5ea19ca98258fe2bfd700d11343e3d.mp4" \
  "$VIDEOS_DIR/56c044c79b791ff1d2eae592c3af1914772d79a23c9603dcaf9a983717c5c47c.mp4" \
  --model-size giant \
  --sample-fps 4 \
  --context-frames 16 \
  --target-frames 8 \
  --stride-duration 6.0 \
  --segments-dir IntPhys2_debug_segments \
  --output debug_results.csv \
  "$@"
