# Shared cluster environment for the scripts in scripts/. Source it from the
# repository root (every script must be run or submitted from there):
#
#   source scripts/env.sh
#
# Export DATA_ROOT or VENV beforehand to point at a different data folder or venv.

if [[ ! -d ego_curation || ! -f scripts/env.sh ]]; then
    echo "ERROR: run from the Ego-Curation folder (current: $PWD)" >&2
    return 1 2>/dev/null || exit 1
fi

DATA_ROOT="${DATA_ROOT:-/data/asem.a.abdelaziz/gp-2027a}"
EGO4D_DIR="$DATA_ROOT/ego4d_data/v2/full_scale"
VENV="${VENV:-../myenv}"

# Exported (not just set) so model downloads land on /data, not in the home directory.
export HF_HOME="$DATA_ROOT/huggingface_downloads"
export TORCH_HOME="$DATA_ROOT/torch_downloads"

# FFmpeg shared libraries, needed by torchcodec and by the ffmpeg CLI.
module load FFmpeg/7.0.2-GCCcore-13.3.0
source "$VENV/bin/activate"
