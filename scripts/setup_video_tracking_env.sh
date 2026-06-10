#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="video_tracking"
YML_FILE="environment.video_tracking.yml"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install Miniforge/Miniconda first."
  exit 1
fi

if [[ ! -f "$YML_FILE" ]]; then
  echo "ERROR: Missing $YML_FILE in repo root."
  exit 1
fi

# Ensure conda shell functions are loaded in non-interactive shells.
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Updating existing conda env: $ENV_NAME"
  conda env update -n "$ENV_NAME" -f "$YML_FILE" --prune
else
  echo "Creating conda env: $ENV_NAME"
  conda env create -f "$YML_FILE"
fi

conda activate "$ENV_NAME"
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

echo ""
echo "Environment ready."
echo "Activate with: conda activate $ENV_NAME"
echo ""