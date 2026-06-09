#!/bin/bash
# Install Segment Anything Model (SAM) in the deeplabcut conda environment

echo "Installing SAM dependencies in conda environment: deeplabcut"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate deeplabcut

# Install segment-anything
pip install git+https://github.com/facebookresearch/segment-anything.git

# Download SAM checkpoint (using vit_b for balance of speed and accuracy)
mkdir -p ../models/sam
cd ../models/sam

if [ ! -f "sam_vit_b_01ec64.pth" ]; then
    echo "Downloading SAM ViT-B checkpoint..."
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
else
    echo "SAM checkpoint already exists."
fi

echo "SAM installation complete!"
