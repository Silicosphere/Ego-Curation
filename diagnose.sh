#!/usr/bin/env bash
#
# DIAGNOSTIC RUN: dump per-token, per-level prediction errors and summarise them.
# Submit with `sbatch diagnose.sh`; send back diag_fold1/summary.txt and
# diag_intphys/summary.txt. Override the subset size with `N_VIDEOS=40 sbatch diagnose.sh`.
#
#SBATCH --job-name=ego4d-diagnose
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=28G                 # default 2000 MB/CPU gets the job silently killed
#SBATCH --time=04:00:00
#SBATCH --exclude=gpu-03,gpu-08   # <16 GiB VRAM
#SBATCH --output=ego4d-diagnose-%j.out

set -euo pipefail

echo "=== host: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== started: $(date -Iseconds) ==="

# ── environment ──────────────────────────────────────────────────────────────
module load FFmpeg/7.0.2-GCCcore-13.3.0

export HF_HOME="/data/asem.a.abdelaziz/gp-2027a/huggingface_downloads"
export TORCH_HOME="/data/asem.a.abdelaziz/gp-2027a/torch_downloads"

source ../myenv/bin/activate

EGO4D_DIR="/data/asem.a.abdelaziz/gp-2027a/ego4d_data/v2/full_scale/fold1"
INTPHYS_DIR="/data/asem.a.abdelaziz/gp-2027a/IntPhys2_Data/Debug/Videos"
N_VIDEOS="${N_VIDEOS:-20}"

# Same settings as the profiling run, so the numbers match its segment scores.
COMMON=(
  --model-size giant
  --sample-fps 4
  --context-frames 16
  --target-frames 8
  --stride-duration 6.0
)

# ── collect ──────────────────────────────────────────────────────────────────
python3 diagnose.py collect "$EGO4D_DIR"/*.mp4 \
  --n-videos "$N_VIDEOS" \
  --out-dir diag_fold1 \
  "${COMMON[@]}"

python3 diagnose.py collect \
  "$INTPHYS_DIR/083861b17a8b3f4bef7e3aaad0b960318a5ea19ca98258fe2bfd700d11343e3d.mp4" \
  "$INTPHYS_DIR/56c044c79b791ff1d2eae592c3af1914772d79a23c9603dcaf9a983717c5c47c.mp4" \
  --out-dir diag_intphys \
  "${COMMON[@]}"

# ── summarise ────────────────────────────────────────────────────────────────
python3 diagnose.py summarize diag_fold1
python3 diagnose.py summarize diag_intphys

echo "=== finished: $(date -Iseconds) ==="
