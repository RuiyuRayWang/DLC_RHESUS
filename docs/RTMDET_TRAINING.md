# RTMDet Training for Monkey Bbox Detection

## Overview

This document describes the complete pipeline for training RTMDet (Real-Time Oriented Bounding Box Object Detection) for detecting monkey bounding boxes in the open-field task (OFT) videos.

**Key features:**
- RTMDet-m backbone (medium model size)
- COCO dataset format with spatial sampling to handle bbox concentration bias
- Multi-GPU distributed training support
- Train/val split: 80/20 for robust evaluation

---

## 1. Environment Setup (Remote Server)

### Step 1a: Create conda environment from YAML

```bash
# Copy the environment file to remote server
rsync -az environment.rtmdet.yml remote:/path/to/DLC_RHESUS/

# On remote server
cd /path/to/DLC_RHESUS
conda env create -f environment.rtmdet.yml

# Activate environment
conda activate rtmdet

# Verify installation
python -c "import mmyolo; print(mmyolo.__version__)"
```

### Step 1b: Verify NVIDIA GPU setup

```bash
# Check GPU availability
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"

# Should output: GPUs available: 8 (on your remote server)
```

---

## 2. Training Launcher: Hugging Face Accelerate

### Why Accelerate?
- **Simpler multi-GPU setup**: Automatic device detection and configuration
- **Flexible distributed training**: Supports DDP, FSDP, and other backends
- **No manual rank management**: No need for `--nproc_per_node`, `--local_rank` flags
- **Mixed precision built-in**: Easy to switch between fp32, fp16, bf16
- **Better logging**: Centralized logging from main process only

### Prerequisites
Accelerate is included in `environment.rtmdet.yml`. No additional setup needed.

### Verify Accelerate installation
```bash
conda activate rtmdet
accelerate --version
accelerate env  # Show hardware configuration
```

---

## 3. Data Preparation Workflow

### Step 3a: Directory structure

Ensure your local machine has:

```
DLC_RHESUS/
├── data/
│   └── oft_rhp/
│       └── videos_downsampled/
│           └── H1705011/2026-06-16/
│               ├── D01_*.mp4
│               ├── D02_*.mp4
│               ├── D03_*.mp4
│               └── D04_*.mp4
│
├── results/
│   └── bbox/
│       └── H1705011/2026-06-16/
│           ├── bbox_D01_*.csv
│           ├── bbox_D02_*.csv
│           ├── bbox_D03_*.csv
│           └── bbox_D04_*.csv
```

### Step 3b: Convert CSV labels to COCO format with spatial sampling

This step converts your bbox CSV files to COCO JSON format while intelligently sampling frames to balance spatial distribution.

**Why spatial sampling?**
- Your bbox annotations are concentrated in certain image regions
- Random sampling would create an imbalanced dataset
- K-means clustering identifies spatial regions
- Frames are sampled uniformly from each region

**On your local machine:**

```bash
python scripts/convert_bbox_to_coco.py \
    --csv-dir results/bbox/H1705011/2026-06-16 \
    --video-dir data/oft_rhp/videos_downsampled \
    --output-dir data/rtmdet_dataset \
    --n-spatial-regions 4 \
    --samples-per-camera 150 \
    --train-split 0.8
```

**Expected output:**
```
data/rtmdet_dataset/
├── D01/
│   └── images/
│       ├── D01_2026-06-16_000000.jpg
│       ├── D01_2026-06-16_000001.jpg
│       └── ...
├── D02/
│   └── images/
│       └── ...
├── D03/
│   └── images/
│       └── ...
├── D04/
│   └── images/
│       └── ...
├── annotations_train.json  (contains ~480 images, 4 cameras × 120)
└── annotations_val.json    (contains ~120 images, 4 cameras × 30)
```

**Key options:**
- `--n-spatial-regions`: K-means clusters for spatial balance (default: 4)
- `--samples-per-camera`: Target frames per camera (default: 150)
- `--train-split`: Train/val split ratio (default: 0.8)

### Step 3c: Rsync dataset to remote server

```bash
rsync -azv --progress data/rtmdet_dataset/ remote:/path/to/DLC_RHESUS/data/rtmdet_dataset/
```

---

## 4. Training on Remote Server

### Step 4a: Single-GPU training (for testing)

```bash
cd /path/to/DLC_RHESUS

# Activate environment
conda activate rtmdet

# Train on single GPU (Accelerate auto-detects)
python scripts/train_rtmdet.py \
    --config configs/rtmdet_m_monkey.py \
    --work-dir results/rtmdet_models/exp001 \
    --cfg-options data_root=data/rtmdet_dataset/ \
                  train_cfg.max_epochs=100 \
                  optim_wrapper.optimizer.lr=0.001
```

### Step 4b: Multi-GPU distributed training with Accelerate (recommended - 8 GPUs)

**Option 1: Interactive training with screen**
```bash
cd /path/to/DLC_RHESUS
conda activate rtmdet

# Create a new screen session
screen -S rtmdet_train

# Launch training with Accelerate (auto-detects 8 GPUs)
accelerate launch scripts/train_rtmdet.py \
    --config configs/rtmdet_m_monkey.py \
    --work-dir results/rtmdet_models/exp001 \
    --cfg-options data_root=data/rtmdet_dataset/ \
                  train_cfg.max_epochs=100

# Detach screen: Ctrl-a, d
# Reattach: screen -r rtmdet_train
```

**Option 2: Explicit 8-GPU configuration**
```bash
accelerate launch --multi_gpu --num_processes=8 scripts/train_rtmdet.py \
    --config configs/rtmdet_m_monkey.py \
    --work-dir results/rtmdet_models/exp001 \
    --cfg-options data_root=data/rtmdet_dataset/
```

**Option 3: With mixed precision (fp16 for faster training, slightly lower accuracy)**
```bash
accelerate launch --mixed_precision=fp16 scripts/train_rtmdet.py \
    --config configs/rtmdet_m_monkey.py \
    --work-dir results/rtmdet_models/exp001 \
    --cfg-options data_root=data/rtmdet_dataset/
```

### Step 4c: Using dSQ for batch job submission

If you prefer submitting through the HPC job scheduler:

```bash
# Create job submission script
cat > train_rtmdet.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=rtmdet_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=8
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --output=logs/rtmdet_train_%j.log

cd /path/to/DLC_RHESUS
source activate rtmdet

accelerate launch scripts/train_rtmdet.py \
    --config configs/rtmdet_m_monkey.py \
    --work-dir results/rtmdet_models/exp001 \
    --cfg-options data_root=data/rtmdet_dataset/ \
                  train_cfg.max_epochs=100
EOF

chmod +x train_rtmdet.sh
sbatch train_rtmdet.sh
```

---

## 5. Monitoring and Checkpoints

### View training progress

```bash
# In real-time with tensorboard
tensorboard --logdir results/rtmdet_models/exp001 --port 6006

# Then access: http://remote_ip:6006
```

### Checkpoint management

Checkpoints are saved in `results/rtmdet_models/exp001/`:

```
├── exp001/
│   ├── vis_data/
│   ├── tf_logs/
│   ├── best_coco_bbox_mAP_epoch_XX.pth  (best model)
│   ├── epoch_10.pth
│   ├── epoch_20.pth
│   └── last_checkpoint                  (pointer to latest)
```

### Resume training from checkpoint

```bash
accelerate launch scripts/train_rtmdet.py \
    --config configs/rtmdet_m_monkey.py \
    --work-dir results/rtmdet_models/exp001 \
    --resume results/rtmdet_models/exp001/epoch_50.pth
```

---

## 6. Evaluation and Inference

### Evaluate on validation set

```python
python -c "
import torch
from mmyolo.apis import init_detector

# Load trained model
checkpoint_path = 'results/rtmdet_models/exp001/best_coco_bbox_mAP_epoch_XX.pth'
config_path = 'configs/rtmdet_m_monkey.py'

model = init_detector(config_path, checkpoint_path, device='cuda:0')
print('Model loaded successfully')
"
```

### Inference on new videos

```bash
python scripts/inference_rtmdet.py \
    --config configs/rtmdet_m_monkey.py \
    --checkpoint results/rtmdet_models/exp001/best_coco_bbox_mAP_epoch_XX.pth \
    --video data/oft_rhp/videos_downsampled/H1705011/2026-06-16/D01_*.mp4 \
    --output results/rtmdet_inference/
```

---

## 7. Configuration Details

### RTMDet-m architecture

- **Backbone**: EfficientRep (depth_multiple=0.67, width_multiple=0.75)
- **Neck**: RepPAN 
- **Head**: RTMDetHead with shared convolutions
- **Input size**: 640×640 (adaptive padding)
- **Classes**: 1 (monkey)

### Training hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | SGD | momentum=0.937, weight_decay=0.0005 |
| Learning rate | 0.01 | Warmup 500 iters, then cosine annealing |
| Batch size | 16 per GPU | 128 total with 8 GPUs |
| Epochs | 100 | Validation every 5 epochs |
| Loss functions | Varifocal + GIoU + DFL | Balanced weighting |

### Data augmentation pipeline

1. **RandomFlip**: 50% probability horizontal flip
2. **RandomCrop**: 640×640 absolute crop with overflow handling
3. **HSV Augmentation**: Color jittering
4. **Resize**: Adaptive resize to 640×640 with padding
5. **FilterAnnotations**: Remove tiny bboxes (< 1×1 px)

---

## 8. Troubleshooting

### CUDA memory error

```bash
# Reduce batch size in config
# In configs/rtmdet_m_monkey.py:
# train_dataloader.batch_size = 8  (instead of 16)
```

### Slow training

```bash
# Check GPU utilization
nvidia-smi -l 1  # Update every 1 second

# Verify Accelerate detected all GPUs
accelerate env
# Should show: number_of_processes: 8
```

### Dataset not found

```bash
# Verify dataset structure
ls -la data/rtmdet_dataset/
# Should show: D01, D02, D03, D04, annotations_train.json, annotations_val.json
```

### Accelerate configuration issues

```bash
# Show Accelerate configuration
accelerate env

# Reset Accelerate config and run interactive setup
accelerate config  # Answers saved to ~/.cache/huggingface/accelerate/

# Use specific config
accelerate launch --config_file path/to/config.yaml scripts/train_rtmdet.py ...
```

### Import errors for mmyolo or accelerate

```bash
# Reinstall packages
pip install --upgrade --force-reinstall mmengine mmcv mmdet mmyolo accelerate

# Or rebuild in conda
conda env remove -n rtmdet
conda env create -f environment.rtmdet.yml
```

---

## 9. Next Steps

1. **Complete bbox annotation** for remaining cameras (D03, D04) - currently have D01, D02
2. **Experiment with hyperparameters**:
   - Learning rate scheduling
   - Augmentation strength
   - Spatial region clustering (try 6-8 regions)
3. **Cross-camera evaluation**:
   - Train per-camera models separately
   - Or ensemble model trained on all cameras
4. **Integration with DLC pipeline**:
   - Use RTMDet predictions as pre-filtering before keypoint detection
   - Improve inference speed with batch processing

---

## 10. Useful Commands

```bash
# Monitor GPU in real-time
watch -n 1 nvidia-smi

# Kill all training processes
pkill -f train_rtmdet.py

# Check disk space
du -sh data/rtmdet_dataset/

# Show Accelerate environment
accelerate env

# Create Accelerate config interactively
accelerate config

# Launch with specific config
accelerate launch --config_file /path/to/config.yaml scripts/train_rtmdet.py ...

# Count samples per camera
python -c "
import json
for cam in ['D01', 'D02', 'D03', 'D04']:
    try:
        with open(f'data/rtmdet_dataset/D{cam[-2:]}/annotations_train.json') as f:
            data = json.load(f)
            print(f'{cam}: {len(data[\"images\"])} train images')
    except: pass
"
```

---

## References

- [Hugging Face Accelerate Docs](https://huggingface.co/docs/accelerate/)
- [MMYolo Documentation](https://github.com/open-mmlab/mmyolo)
- [RTMDet Paper](https://arxiv.org/abs/2212.07784)
- [COCO Dataset Format](https://cocodataset.org/)
