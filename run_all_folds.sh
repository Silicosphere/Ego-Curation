#!/usr/bin/env bash
#
# PARALLEL RUN — 5 folds on 5 nodes simultaneously.
#
# Measured from profiling job 1109 (fold 1, 2-hour run):
#   MaxRSS = 15.6 GiB  → --mem=20G (15.6 GiB + 20% headroom)
#   AveCPU = 01:56:38 over 12 CPUs → ~0.97 effective cores → --cpus-per-task=2
#   Elapsed = 2:00:23 (TIMEOUT) → fold 1 did not finish in 2 h; 48 h is correct
#
#SBATCH --job-name=ego4d-allfolds
#SBATCH --partition=gpu
#SBATCH --nodes=5
#SBATCH --ntasks=5                # one task per node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4         # pipeline is ~single-threaded (0.97 cores measured)
#SBATCH --gres=gpu:1              # 1 GPU per node; --exclude below enforces >=16 GiB
#SBATCH --mem=20G                 # 15.6 GiB peak RSS + 20% headroom (per node)
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu-03,gpu-08,gpu-05   # these are the only nodes with <16 GiB GPUs
#SBATCH --output=ego4d-allfolds-%j.out
#SBATCH --error=ego4d-allfolds-%j.err

set -euo pipefail

# ── environment (runs on the head node of the allocation) ────────────────────
module load FFmpeg/7.0.2-GCCcore-13.3.0

export HF_HOME="/data/asem.a.abdelaziz/gp-2027a/huggingface_downloads"
export TORCH_HOME="/data/asem.a.abdelaziz/gp-2027a/torch_downloads"

echo "=== job $SLURM_JOB_ID allocated nodes: $SLURM_JOB_NODELIST ==="
echo "=== started: $(date -Iseconds) ==="

# Build an ordered array of the allocated hostnames
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")

if [[ ${#NODES[@]} -ne 5 ]]; then
    echo "ERROR: expected 5 nodes, got ${#NODES[@]}: ${NODES[*]}" >&2
    exit 1
fi

# ── launch one fold per node, all in parallel ────────────────────────────────
for i in {0..4}; do
    NODE="${NODES[$i]}"
    FOLD=$((i + 1))
    echo "=== dispatching fold $FOLD → $NODE ==="

    srun \
      --exclusive \
      --nodes=1 \
      --ntasks=1 \
      --nodelist="$NODE" \
      bash -c "
        set -euo pipefail
        echo \"[\$(hostname)] fold $FOLD started at \$(date -Iseconds)\"
        echo \"[\$(hostname)] GPU: \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)\"

        module load FFmpeg/7.0.2-GCCcore-13.3.0
        export HF_HOME=\"$HF_HOME\"
        export TORCH_HOME=\"$TORCH_HOME\"
        source ../myenv/bin/activate

        python3 main.py \
          /data/asem.a.abdelaziz/gp-2027a/ego4d_data/v2/full_scale/fold${FOLD}/*.mp4 \
          --model-size giant \
          --sample-fps 4 \
          --context-frames 16 \
          --target-frames 8 \
          --stride-duration 6.0 \
          --segments-dir segments${FOLD} \
          --output results${FOLD}.csv

        echo \"[\$(hostname)] fold $FOLD finished at \$(date -Iseconds)\"
      " &
done

# Wait for all background sruns to complete
wait
echo "=== all folds done: $(date -Iseconds) ==="

# ── how to measure after the job ─────────────────────────────────────────────
echo ""
echo "Measure actual usage (run this after the job):"
echo "  sacct -j $SLURM_JOB_ID --format=JobID,JobName,Elapsed,MaxRSS,AveCPU,AllocCPUS,State"

