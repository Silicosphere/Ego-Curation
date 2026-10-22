#!/usr/bin/env bash

module load FFmpeg/7.0.2-GCCcore-13.3.0

export HF_HOME="/data/asem.a.abdelaziz/gp-2027a/huggingface_downloads"

export TORCH_HOME='/data/asem.a.abdelaziz/gp-2027a/torch_downloads'

source ../myenv/bin/activate

python3 -m ego_curation.visualization \
       	/data/asem.a.abdelaziz/gp-2027a/ego4d_data/v2/full_scale/fold1/03cc49c3-a7d1-445b-9a2a-545c4fae6843.mp4 \
       	--start-sec 12 30.5 60 \
	--model-size giant \
       	--out-dir runs_results/pca_plots
