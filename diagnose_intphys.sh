#!/usr/bin/env bash
#
# INTPHYS2 PAIR DIAGNOSTIC: dense-stride dumps of every labelled IntPhys2 video,
# then a possible/impossible pair analysis per configuration.
# Submit with `sbatch diagnose_intphys.sh`; send back every pairs.txt under $OUT_ROOT.
# Collection resumes (finished videos are skipped), so a timed-out job can simply
# be resubmitted. `SPLITS=Debug sbatch diagnose_intphys.sh` runs only the quick split.
#
#SBATCH --job-name=intphys2-diagnose
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=28G                 # default 2000 MB/CPU gets the job silently killed
#SBATCH --time=12:00:00
#SBATCH --exclude=gpu-03,gpu-08   # <16 GiB VRAM
#SBATCH --output=intphys2-diagnose-%j.out

set -euo pipefail

echo "=== host: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== started: $(date -Iseconds) ==="

# ── environment ──────────────────────────────────────────────────────────────
module load FFmpeg/7.0.2-GCCcore-13.3.0

export HF_HOME="/data/asem.a.abdelaziz/gp-2027a/huggingface_downloads"
export TORCH_HOME="/data/asem.a.abdelaziz/gp-2027a/torch_downloads"

source ../myenv/bin/activate

# Each split directory holds metadata.csv and the Videos/ it lists.
INTPHYS_ROOT="${INTPHYS_ROOT:-/data/asem.a.abdelaziz/gp-2027a/IntPhys2_Data}"
OUT_ROOT="${OUT_ROOT:-diag_intphys2}"
SPLITS="${SPLITS:-Debug Main}"
STRIDE="${STRIDE:-0.5}"           # seconds; dense, so some target contains the event

# "name context_encoder context_frames target_frames", all at 4 fps.
# target = pipeline as is; online = context from the online encoder, as in the
# official IntPhys2 eval; c8 = 2 s context, for events early in the clip.
CONFIGS_DEBUG=("target_c16 target 16 8" "online_c16 online 16 8" "online_c8 online 8 8")
CONFIGS_MAIN=("target_c16 target 16 8" "online_c16 online 16 8")

video_list() {  # absolute paths of the videos listed in <split dir>/metadata.csv
  python3 - "$1" <<'EOF'
import csv, sys
from pathlib import Path
root = Path(sys.argv[1])
with open(root / "metadata.csv", newline="") as f:
    for row in csv.DictReader(f):
        print(root / row["file_name"])
EOF
}

for split in $SPLITS; do
  dir="$INTPHYS_ROOT/$split"
  if [[ ! -f "$dir/metadata.csv" ]]; then
    echo "missing $dir/metadata.csv; set INTPHYS_ROOT" >&2
    exit 1
  fi
  mapfile -t videos < <(video_list "$dir")
  echo "=== $split: ${#videos[@]} videos ==="
  if (( ${#videos[@]} == 0 )); then
    echo "no videos listed in $dir/metadata.csv" >&2
    exit 1
  fi

  if [[ "$split" == "Debug" ]]; then
    configs=("${CONFIGS_DEBUG[@]}")
  else
    configs=("${CONFIGS_MAIN[@]}")
  fi

  for cfg in "${configs[@]}"; do
    read -r name encoder ctx tgt <<< "$cfg"
    out="$OUT_ROOT/$split/$name"
    echo "=== $split / $name: $(date -Iseconds) ==="
    python3 diagnose.py collect "${videos[@]}" \
      --out-dir "$out" \
      --model-size giant \
      --sample-fps 4 \
      --context-frames "$ctx" \
      --target-frames "$tgt" \
      --stride-duration "$STRIDE" \
      --context-encoder "$encoder"
    python3 diagnose.py pairs "$out" --metadata "$dir/metadata.csv"
  done
done

echo "=== finished: $(date -Iseconds) ==="
