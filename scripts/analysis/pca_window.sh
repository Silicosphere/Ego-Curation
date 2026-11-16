#!/usr/bin/env bash
#
# PCA view of the V-JEPA features of a few windows of one video (ego_curation.visualization).
# Run inside an interactive GPU session, from the repository root:
#
#   ./scripts/analysis/pca_window.sh

set -euo pipefail

source scripts/env.sh

python3 -m ego_curation.visualization \
  "$EGO4D_DIR/fold1/03cc49c3-a7d1-445b-9a2a-545c4fae6843.mp4" \
  --start-sec 12 30.5 60 \
  --model-size giant \
  --out-dir runs_results/pca_plots
