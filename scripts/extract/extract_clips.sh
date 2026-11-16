#!/usr/bin/env bash
#
# Cut the scored segments of one fold into clips, re-encoding on the GPU (NVENC).
# Run setup_nvenc.sh once first. Submit from the repository root:
#
# Usage: sbatch scripts/extract/extract_clips.sh <fold>        e.g. ... extract_clips.sh 1
#        ALLOW_CPU=1 sbatch scripts/extract/extract_clips.sh 1  (encode on the CPU if NVENC
#        is not set up)
#        SEGMENTS_DIR=<dir> sbatch ...  (cut the CSVs of another run; default: run_1184)
#
# Resources (fold 1: 200 videos, ~55% larger than 1080p, up to 3413x1920):
#   time: 2.1 s/clip measured on a 2560x1920 video (worst common case), CPU-bound on decoding
#   mem:  ~800 MB per ffmpeg measured on 2560x1920; 4 at once = ~3.2 GB
#   disk: ~3 MB per clip at 540 px
#
#SBATCH --job-name=ego4d-extract
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4060ti:1       # NVENC only; any card works, so don't tie up the 5090
#SBATCH --cpus-per-task=8            # decoding is on the CPU: 5 of 6 were busy at 2560x1920
#SBATCH --mem=8G                     # 3.2 GB measured at 2560x1920; margin for 3413x1920
#SBATCH --time=18:00:00              # clips x 2.1 s is the worst case; resubmit resumes
#SBATCH --output=ego4d-extract-%j.out

set -euo pipefail

FOLD="${1:?usage: sbatch scripts/extract/extract_clips.sh <fold>}"
JOBS=4               # parallel ffmpeg processes (GeForce cards limit concurrent NVENC sessions)

# Fail early, with a clear message, instead of hours into the run (env.sh checks the folder).
source scripts/env.sh
export PYTHONUNBUFFERED=1    # write progress lines to the log immediately, not in chunks

SEGMENTS_DIR="${SEGMENTS_DIR:-runs_results/run_1184/segments$FOLD}"
FINAL_OUT="$DATA_ROOT/ego4d_clips/clips$FOLD"
shopt -s nullglob
CSVS=("$SEGMENTS_DIR"/*.csv)
if (( ${#CSVS[@]} == 0 )); then
    echo "ERROR: no segment CSVs in $SEGMENTS_DIR" >&2; exit 1
fi

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
