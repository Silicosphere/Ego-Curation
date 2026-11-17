#!/usr/bin/env bash
#
# Pack one fold's clips folder into a single HDF5 file and delete the folder once verified.
# CPU only; no GPU needed. Submit from the repository root:
#
# Usage: sbatch scripts/extract/pack_hdf5.sh <fold>        e.g. ... pack_hdf5.sh 1
#
# Needs h5py in the venv (once): pip install h5py
# The .h5 is built and checked on the node's local scratch disk, then copied to /data as one
# file. Needs free space for one extra copy of the fold on /data until the folder is deleted.
#
#SBATCH --job-name=pack-hdf5
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G                     # reads one clip (a few MB) at a time
#SBATCH --time=06:00:00              # unmeasured guess: build, check, copy and check ~60 GB
#SBATCH --output=pack-hdf5-%j.out

set -euo pipefail

FOLD="${1:?usage: sbatch scripts/extract/pack_hdf5.sh <fold>}"
source scripts/env.sh
export PYTHONUNBUFFERED=1

CLIPS_DIR="$DATA_ROOT/ego4d_clips/clips$FOLD"
echo "=== job: $SLURM_JOB_ID  packing: $CLIPS_DIR  started: $(date -Iseconds) ==="
du -sh "$CLIPS_DIR"
df -h "$DATA_ROOT" | tail -1

python scripts/extract/pack_hdf5.py "$CLIPS_DIR" --workdir "$SCRATCH"

echo "=== finished: $(date -Iseconds) ==="
ls -lh "$CLIPS_DIR.h5"
