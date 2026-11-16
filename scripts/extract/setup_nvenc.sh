#!/usr/bin/env bash
#
# Submit from the repository root: sbatch scripts/extract/setup_nvenc.sh
#
# One-time setup: install the NVIDIA video-encoder library (libnvidia-encode) in your home
# directory, so ffmpeg's h264_nvenc encoder can use the GPU. No root needed.
#
# The library must match the node's NVIDIA driver version EXACTLY, so it is installed under
# ~/nvidia-libs/<driver version>/ and extract_clips.sh picks the matching folder at run time.
# Run this on the same GPU type you will extract on (rtx4060ti below).
#
#SBATCH --job-name=nvenc-setup
#SBATCH --partition=debug            # max 30 min; this takes a few minutes
#SBATCH --gres=gpu:rtx4060ti:1       # any NVENC card works; same node type as extract_clips.sh
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G                     # the driver installer is ~400 MB, unpacked in $SCRATCH
#SBATCH --time=00:20:00
#SBATCH --output=nvenc-setup-%j.out

set -euo pipefail

module load FFmpeg/7.0.2-GCCcore-13.3.0

echo "=== node: $(hostname)  job: $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=name,driver_version --format=csv

DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
DEST="$HOME/nvidia-libs/$DRV"
echo "=== driver version: $DRV ==="

# 1. Maybe the library is installed but not on the search path.
echo "=== system copies of libnvidia-encode (if any) ==="
ldconfig -p | grep libnvidia-encode || true
find /usr/lib /usr/lib64 /usr/local -name 'libnvidia-encode.so*' 2>/dev/null || true

# 2. Extract the matching user-space libraries from NVIDIA's official driver installer.
if [[ ! -e "$DEST/libnvidia-encode.so.1" ]]; then
    cd "$SCRATCH"
    RUN="NVIDIA-Linux-x86_64-$DRV.run"
    wget -q "https://us.download.nvidia.com/XFree86/Linux-x86_64/$DRV/$RUN" \
        || wget -q "https://us.download.nvidia.com/tesla/$DRV/$RUN" \
        || { echo "ERROR: could not download $RUN from nvidia.com" >&2; exit 1; }
    sh "$RUN" --extract-only --target "drv-$DRV" > /dev/null

    mkdir -p "$DEST"
    for lib in libnvidia-encode libnvcuvid; do
        cp "drv-$DRV/$lib.so.$DRV" "$DEST/"
        ln -sf "$lib.so.$DRV" "$DEST/$lib.so.1"
    done
    echo "=== installed to $DEST ==="
else
    echo "=== already installed in $DEST ==="
fi
ls -l "$DEST"

# 3. Test: encode 5 s of 1080p test video on the GPU.
export LD_LIBRARY_PATH="$DEST:${LD_LIBRARY_PATH:-}"
echo "=== NVENC test ==="
time ffmpeg -hide_banner -loglevel error \
    -f lavfi -i testsrc=duration=5:size=1920x1080:rate=30 -c:v h264_nvenc -f null - \
    && echo "NVENC OK" || echo "NVENC FAILED"
