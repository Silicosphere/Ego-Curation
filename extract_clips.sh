#!/usr/bin/env bash
#
# Cut the scored segments of one fold into clips, re-encoding on the GPU (NVENC).
# Run setup_nvenc.sh once first. Falls back to CPU encoding (libx264) if NVENC is not set up.
#
# Usage: sbatch extract_clips.sh <fold>        e.g. sbatch extract_clips.sh 1
#
#
#SBATCH --job-name=ego4d-extract
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4060ti:1       # NVENC only; any card works, so don't tie up the 5090
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=06:00:00              # unmeasured guess; set from sacct after the first run
#SBATCH --signal=B:USR1@600          # 10 min before the time limit: copy clips back
#SBATCH --output=ego4d-extract-%j.out

set -euo pipefail

FOLD="${1:?usage: sbatch extract_clips.sh <fold>}"
JOBS=4               # parallel ffmpeg processes (GeForce cards limit concurrent NVENC sessions)
SEGMENTS_DIR="/data/asem.a.abdelaziz/gp-2027a/Ego-Curation/runs_results/run_1184/segments$FOLD"
FINAL_OUT="/data/asem.a.abdelaziz/gp-2027a/Ego-Curation/runs_results/clips$FOLD"

module load FFmpeg/7.0.2-GCCcore-13.3.0
source ../myenv/bin/activate

# Use the GPU encoder if setup_nvenc.sh installed the library for this node's driver.
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
if [[ -e "$HOME/nvidia-libs/$DRV/libnvidia-encode.so.1" ]]; then
    export LD_LIBRARY_PATH="$HOME/nvidia-libs/$DRV:${LD_LIBRARY_PATH:-}"
    CODEC=h264_nvenc
else
    echo "WARNING: no NVENC library for driver $DRV; run setup_nvenc.sh. Using CPU." >&2
    CODEC=libx264
fi
echo "=== node: $(hostname)  job: $SLURM_JOB_ID  fold: $FOLD  encoder: $CODEC ==="

# Write clips to fast local scratch, copy them to /data at the end (or on timeout).
OUT="$SCRATCH/clips$FOLD""
copy_back() {
    mkdir -p "$FINAL_OUT"
    rsync -a "$OUT/" "$FINAL_OUT/"
    echo "=== copied clips to $FINAL_OUT ==="
}
trap 'copy_back; exit 1' USR1

# Resume: bring back clips from an earlier run of this fold and skip them.
mkdir -p "$OUT"
[[ -d "$FINAL_OUT" ]] && rsync -a "$FINAL_OUT/" "$OUT/"

python -m ego_curation.extract "$SEGMENTS_DIR"/*.csv \
    --output "$OUT" \
    --video-codec "$CODEC" \
    --no-audio \
    --jobs "$JOBS" \
    --skip-existing &
wait $!          # wait in the background so the USR1 trap can fire during the run

copy_back
