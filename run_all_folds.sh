#!/usr/bin/env bash
#
# PARALLEL RUN — 6 folds on 6 nodes simultaneously.
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │ TODO: fill in --mem and --cpus-per-task after the profiling run.         │
# │ Run profile_fold.sh first, then:                                         │
# │   sacct -j <JOBID> --format=JobID,JobName,Elapsed,MaxRSS,AveCPU,State   │
# │ and set --mem to (MaxRSS + ~20% headroom) and --cpus-per-task to what   │
# │ AveCPU saturated at.                                                     │
# └──────────────────────────────────────────────────────────────────────────┘
#
#SBATCH --job-name=ego4d-allfolds
#SBATCH --partition=gpu
#SBATCH --nodes=6
#SBATCH --ntasks=6                # one task per node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12        # TODO: tune after profiling
#SBATCH --gres=gpu:1              # 1 GPU per node; --exclude below enforces >=16 GiB
#SBATCH --mem=28G                 # TODO: tune after profiling (applied per node)
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu-03,gpu-08   # these are the only nodes with <16 GiB GPUs
#SBATCH --output=ego4d-allfolds-%j.out

set -euo pipefail

# ── environment (runs on the head node of the allocation) ────────────────────
module load FFmpeg/7.0.2-GCCcore-13.3.0

export HF_HOME="/data/asem.a.abdelaziz/gp-2027a/huggingface_downloads"
export TORCH_HOME="/data/asem.a.abdelaziz/gp-2027a/torch_downloads"

echo "=== job $SLURM_JOB_ID allocated nodes: $SLURM_JOB_NODELIST ==="
echo "=== started: $(date -Iseconds) ==="

# Build an ordered array of the 6 allocated hostnames
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")

if [[ ${#NODES[@]} -ne 6 ]]; then
    echo "ERROR: expected 6 nodes, got ${#NODES[@]}: ${NODES[*]}" >&2
    exit 1
fi

# ── launch one fold per node, all in parallel ────────────────────────────────
for i in {0..5}; do
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
          --segments-dir segments${FOLD}

        echo \"[\$(hostname)] fold $FOLD finished at \$(date -Iseconds)\"
      " &
done

# Wait for all 6 background sruns to complete
wait
echo "=== all folds done: $(date -Iseconds) ==="

# ── how to measure after the job ─────────────────────────────────────────────
echo ""
echo "Measure actual usage (run this after the job):"
echo "  sacct -j $SLURM_JOB_ID --format=JobID,JobName,Elapsed,MaxRSS,AveCPU,AllocCPUS,State"
