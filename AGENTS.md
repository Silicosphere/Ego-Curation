# Agent notes

## Execution environment

- The pipeline runs on a **dedicated cluster** (with GPUs), not on this machine. This machine is used only to develop the code.
- Do not try to download the V-JEPA 2.1 checkpoints or run real-weight inference locally. The checkpoints are 15–28 GB, and this machine has no GPU and limited RAM.
- Local validation is limited to random-weight or stubbed tests. Real runs (for example, `python verify_fix.py --model-size gigantic`) are done on the cluster.
- The local Python environment may lack pipeline dependencies such as `torchcodec`, `timm` and `einops`. Unresolved-import warnings in the editor are expected.
