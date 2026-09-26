#!/usr/bin/env bash
#
# PROFILING RUN — single fold on a single node.
# Goal: measure real peak RAM and CPU usage with sacct.
# After this job finishes, run:
#   sacct -j <JOBID> --format=JobID,JobName,Elapsed,MaxRSS,AveCPU,AllocCPUS,State
# and report back so the parallel script can be tuned.
#
#SBATCH --job-name=ego4d-profile
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12        # generous ceiling so CPU is not the bottleneck
#SBATCH --gres=gpu:1              # any available GPU; --exclude below enforces >=16 GiB
#SBATCH --mem=28G                 # close to the 29 000 MB schedulable limit on most nodes;
#                                 # a generous ceiling so the job is not OOM-killed —
#                                 # sacct will tell us what we actually used
#SBATCH --time=02:00:00           # 2-hour ceiling for a single-fold profiling run
#SBATCH --exclude=gpu-03,gpu-08   # only nodes with <16 GiB VRAM; all others qualify
#SBATCH --output=ego4d-profile-%j.out

set -euo pipefail

echo "=== host: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== started: $(date -Iseconds) ==="

# ── environment ──────────────────────────────────────────────────────────────
module load FFmpeg/7.0.2-GCCcore-13.3.0

export HF_HOME="/data/asem.a.abdelaziz/gp-2027a/huggingface_downloads"
export TORCH_HOME="/data/asem.a.abdelaziz/gp-2027a/torch_downloads"

source ../myenv/bin/activate

# ── run fold 1 ───────────────────────────────────────────────────────────────
python3 main.py \
  /data/asem.a.abdelaziz/gp-2027a/ego4d_data/v2/full_scale/fold1/*.mp4 \
  --model-size giant \
  --sample-fps 4 \
  --context-frames 16 \
  --target-frames 8 \
  --stride-duration 6.0 \
  --segments-dir segments1

echo "=== finished: $(date -Iseconds) ==="

# ── reminder ─────────────────────────────────────────────────────────────────
echo ""
echo "Run this after the job ends to get the numbers needed for the parallel script:"
echo "  sacct -j \$SLURM_JOB_ID --format=JobID,JobName,Elapsed,MaxRSS,AveCPU,AllocCPUS,State"
