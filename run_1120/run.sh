#!/usr/bin/env bash

module load FFmpeg/7.0.2-GCCcore-13.3.0

export HF_HOME="/data/asem.a.abdelaziz/gp-2027a/huggingface_downloads"

export TORCH_HOME='/data/asem.a.abdelaziz/gp-2027a/torch_downloads'

source ../myenv/bin/activate

python3 main.py \
  /data/asem.a.abdelaziz/gp-2027a/ego4d_data/v2/full_scale/fold"$1"/*.mp4 \
  --model-size giant \
  --sample-fps 4 \
  --context-frames 16 \
  --target-frames 8 \
  --stride-duration 6.0 \
  --segments-dir segments"$1" \
