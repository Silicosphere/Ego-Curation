# Agent notes

## Execution environment

- The pipeline runs on a **dedicated cluster** (with GPUs), not on this machine. This machine is used only to develop the code.
- Do not try to download the V-JEPA 2.1 checkpoints or run real-weight inference locally. The checkpoints are 15–28 GB, and this machine has no GPU and limited RAM.
- Local validation is limited to random-weight or stubbed tests. Real runs (for example, `python verify_fix.py --model-size gigantic`) are done on the cluster.
- The local Python environment may lack pipeline dependencies such as `torchcodec`, `timm` and `einops`. Unresolved-import warnings in the editor are expected.

## Cluster setup (Slurm)

- Default job resources are 1 CPU and 2000 MB RAM (`DefMemPerCPU = 2000`). This is too little: runs are silently `Killed` by the job's memory limit. Always request memory, e.g. `srun --gres=gpu:rtx5090 --cpus-per-task=8 --mem=32G --time=12:00:00 --pty bash`.
- GPU nodes (e.g. `gpu-06`) have an RTX 5090 (32 GB VRAM) and about 147 GB RAM.
- FFmpeg shared libraries (needed by `torchcodec`) come from a module: `module load FFmpeg/7.0.2-GCCcore-13.3.0`.
- `TORCH_HOME` must be exported, not just set, so checkpoints go to `/data/...` instead of the home directory.
