#!/usr/bin/env bash
#
# Cut the scored segments of one fold into clips, re-encoding on the GPU (NVENC).
# Run setup_nvenc.sh once first. Submit from the Ego-Curation folder:
#
# Usage: sbatch extract_clips.sh <fold>        e.g. sbatch extract_clips.sh 1
#        ALLOW_CPU=1 sbatch extract_clips.sh 1  (encode on the CPU if NVENC is not set up)
#
# Resources (fold 1: 200 videos, ~55% larger than 1080p, up to 3413x1920):
#   time: ~160 s/video measured on a 3-video test; larger videos decode slower
#   mem:  ~750 MB per ffmpeg measured; 4 run at once on the same video, more for large frames
#
#SBATCH --job-name=ego4d-extract
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4060ti:1       # NVENC only; any card works, so don't tie up the 5090
#SBATCH --cpus-per-task=6            # decoding is on the CPU: ~2.25 busy at 1080p, more above
#SBATCH --mem=12G                    # 4 ffmpeg processes on frames up to 3413x1920
#SBATCH --time=18:00:00              # ~11-16 h expected for 200 videos; resubmit resumes
#SBATCH --output=ego4d-extract-%j.out

set -euo pipefail

FOLD="${1:?usage: sbatch extract_clips.sh <fold>}"
JOBS=4               # parallel ffmpeg processes (GeForce cards limit concurrent NVENC sessions)
SEGMENTS_DIR="/data/asem.a.abdelaziz/gp-2027a/Ego-Curation/runs_results/run_1184/segments$FOLD"
FINAL_OUT="/data/asem.a.abdelaziz/gp-2027a/ego4d_clips/clips$FOLD"

# Fail early, with a clear message, instead of hours into the run.
if [[ ! -d ego_curation ]]; then
    echo "ERROR: submit from the Ego-Curation folder (current: $PWD)" >&2; exit 1
fi
shopt -s nullglob
CSVS=("$SEGMENTS_DIR"/*.csv)
if (( ${#CSVS[@]} == 0 )); then
    echo "ERROR: no segment CSVs in $SEGMENTS_DIR" >&2; exit 1
fi

module load FFmpeg/7.0.2-GCCcore-13.3.0
source ../myenv/bin/activate

# Use the GPU encoder if setup_nvenc.sh installed the library for this node's driver.
# CPU encoding of a whole fold is several times slower and would not fit the time limit,
# so it only runs when asked for explicitly.
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
if [[ -e "$HOME/nvidia-libs/$DRV/libnvidia-encode.so.1" ]]; then
    export LD_LIBRARY_PATH="$HOME/nvidia-libs/$DRV:${LD_LIBRARY_PATH:-}"
    CODEC=h264_nvenc
elif [[ "${ALLOW_CPU:-0}" == 1 ]]; then
    echo "WARNING: no NVENC library for driver $DRV; encoding on the CPU." >&2
    CODEC=libx264
else
    echo "ERROR: no NVENC library for driver $DRV on $(hostname). Run setup_nvenc.sh on" \
         "this GPU type, or resubmit with ALLOW_CPU=1." >&2
    exit 1
fi

echo "=== node: $(hostname)  job: $SLURM_JOB_ID  fold: $FOLD  encoder: $CODEC ==="
echo "=== ${#CSVS[@]} segment CSVs  started: $(date -Iseconds) ==="
df -h "$(dirname "$FINAL_OUT")" | tail -1

# Clips are written straight to /data. Each clip is renamed into place only when complete,
# so if the job times out, resubmitting resumes where it stopped (--skip-existing).
#   --max-short-side 540: same as Ego4D's own downsampled release (video_540ss). Leaves
#                         headroom for 384 px training with random-resized crops; the
#                         scoring model itself only uses a 438 px short side.
#   --delete-source:      deletes each source video once ALL its clips succeeded (the drive
#                         backup was verified with rclone check). Re-scoring or re-cutting a
#                         deleted video needs it downloaded again.
status=0
python -m ego_curation.extract "${CSVS[@]}" \
    --output "$FINAL_OUT" \
    --video-codec "$CODEC" \
    --max-short-side 540 \
    --no-audio \
    --jobs "$JOBS" \
    --skip-existing \
    --delete-source || status=$?

echo "=== finished: $(date -Iseconds)  exit status: $status ==="
du -sh "$FINAL_OUT"
exit "$status"
