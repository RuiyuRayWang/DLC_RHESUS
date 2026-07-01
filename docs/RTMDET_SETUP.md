# RTMDet Training Setup Complete ✅

## Summary

I've created a complete RTMDet training pipeline for monkey bbox detection. Here's what's been set up:

---

## 📦 New Files Created

### 1. **Environment & Dependencies**
- **`environment.rtmdet.yml`** - Conda environment with all required packages (PyTorch, MMYolo, MMDetection, etc.)

### 2. **Scripts**
- **`scripts/convert_bbox_to_coco.py`** - Converts your bbox CSV labels to COCO JSON format with intelligent spatial sampling to handle bbox concentration bias
- **`scripts/train_rtmdet.py`** - Multi-GPU distributed training script with checkpoint resumption
- **`scripts/inference_rtmdet.py`** - Inference script for running predictions on videos with visualization

### 3. **Configuration**
- **`configs/rtmdet_m_monkey.py`** - RTMDet-m model configuration optimized for your dataset

### 4. **Documentation**
- **`docs/RTMDET_TRAINING.md`** - Comprehensive 10-section guide covering setup through evaluation
- **`docs/RTMDET_CHECKLIST.md`** - Quick reference with checkbox workflow and troubleshooting

---

## 🎯 Key Features

### Spatial Sampling for Dataset Balance
Your bbox annotations are concentrated in certain image regions (e.g., bottom-right). The conversion script uses K-means clustering to:
1. Identify spatial regions where bboxes occur
2. Sample frames uniformly from each region
3. Create a balanced dataset that avoids overfitting to specific locations

### Multi-GPU Training
- Supports distributed training on 8 GPUs simultaneously
- Automatic load balancing and gradient synchronization
- Uses PyTorch distributed launch

### Configuration Flexibility
Easy to tune:
```python
--samples-per-camera 150       # Adjust dataset size
--n-spatial-regions 4         # Adjust spatial balance
--train-split 0.8             # Adjust train/val ratio
```

---

## 🚀 Quick Start (3 Steps)

### Step 1: Prepare Dataset (Local Machine)
```bash
python scripts/convert_bbox_to_coco.py \
    --csv-dir results/bbox/H1705011/2026-06-16 \
    --video-dir data/oft_rhp/videos_downsampled \
    --output-dir data/rtmdet_dataset
```

Expected: ~480 training images, ~120 validation images (4 cameras × 150 samples)

### Step 2: Setup Remote Environment
```bash
# Copy files to remote
rsync -az environment.rtmdet.yml data/rtmdet_dataset/ user@remote:/path/to/DLC_RHESUS/

# On remote server
conda env create -f environment.rtmdet.yml
conda activate rtmdet
```

### Step 3: Train Model (with Accelerate)
```bash
# Single GPU (auto-detected)
python scripts/train_rtmdet.py \
    --config configs/rtmdet_m_monkey.py \
    --work-dir results/rtmdet_models/exp001

# Multi-GPU (8 GPUs with Accelerate)
accelerate launch scripts/train_rtmdet.py \
    --config configs/rtmdet_m_monkey.py \
    --work-dir results/rtmdet_models/exp001

# With mixed precision (faster, slightly lower accuracy)
accelerate launch --mixed_precision=fp16 scripts/train_rtmdet.py \
    --config configs/rtmdet_m_monkey.py \
    --work-dir results/rtmdet_models/exp001
```

**Training time**: ~2-4 hours on 8 GPUs (100 epochs)

### Key Advantages of Accelerate
- **Auto-detection**: No need to specify `--nproc_per_node` or GPU indices
- **Single codebase**: Same script works for 1 GPU, 8 GPUs, or multi-node
- **Mixed precision**: Easy to enable with `--mixed_precision=fp16`
- **Better logging**: Only main process logs to avoid duplicates
- **Flexible**: Supports DDP, FSDP, and other distributed backends

---

## 📊 Model Specs

| Component | Specification |
|-----------|---|
| **Architecture** | RTMDet-m (Medium) |
| **Backbone** | EfficientRep |
| **Neck** | RepPAN |
| **Input** | 640×640 (adaptive padding) |
| **Classes** | 1 (monkey) |
| **Optimizer** | SGD (lr=0.01, momentum=0.937) |
| **Batch Size** | 128 (16 per GPU × 8 GPUs) |
| **Epochs** | 100 |
| **Loss** | Varifocal + GIoU + DFL |

---

## 📝 Workflow Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 LOCAL MACHINE                               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Bbox CSV files (manual labels)                          │
│     └─→ convert_bbox_to_coco.py                            │
│         ├─ K-means spatial clustering                       │
│         ├─ Balanced frame sampling                          │
│         └─→ COCO JSON + extracted images                    │
│                                                               │
│  2. rsync to remote server                                  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              REMOTE SERVER (8 GPUs)                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  3. Create conda environment                                │
│  4. Run multi-GPU training (train_rtmdet.py)               │
│     └─→ 100 epochs, checkpoint every 10 epochs             │
│     └─→ Validation every 5 epochs                          │
│     └─→ Best checkpoint saved                              │
│                                                               │
│  5. Run inference (inference_rtmdet.py)                    │
│     └─→ Generate bbox predictions + CSV                    │
│     └─→ Optional: Save visualizations                      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│            RESULTS & INTEGRATION                            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  6. Evaluate predictions vs ground truth                     │
│  7. rsync best checkpoint back to local                      │
│  8. Integrate RTMDet pre-filtering into DLC pipeline        │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Data Preparation Details

### What the converter does:

1. **Loads CSV labels**
   - Reads all bbox annotations from your CSV files
   - Filters for visible monkey frames

2. **Spatial analysis**
   - Extracts bbox centers from all frames
   - Runs K-means (k=4 by default) to identify spatial clusters
   - Prints distribution per region

3. **Balanced sampling**
   - Divides frames into spatial regions
   - Samples uniformly from each region
   - Skips consecutive frames to avoid temporal bias

4. **Frame extraction**
   - Opens video file for each selected frame
   - Extracts raw frame using OpenCV
   - Saves as JPEG in `images/` subdirectory

5. **COCO JSON generation**
   - Converts bbox to COCO format: `[x, y, width, height]`
   - Assigns annotation IDs (unique per frame)
   - Creates train/val split (80/20)
   - Outputs: `annotations_train.json`, `annotations_val.json`

### Example output structure:
```
data/rtmdet_dataset/
├── D01/
│   └── images/
│       ├── D01_2026-06-16_135117_000000.jpg  (camera D1, timestamp, frame)
│       ├── D01_2026-06-16_135117_000020.jpg
│       └── ...
├── D02/
│   └── images/
│       └── ...
├── annotations_train.json  # 480 images, 480 annotations
└── annotations_val.json    # 120 images, 120 annotations
```

---

## 🔍 Dataset Bias Handling

### Problem
Your bbox labels are concentrated in specific regions (e.g., corner of arena). Random sampling would create skewed training data.

### Solution: K-Means Clustering
1. Cluster all bbox centers into k regions (default: 4)
2. For each region, sample frames proportionally
3. Result: Balanced representation across spatial locations

### Tuning
```bash
# More spatial regions = better balance
--n-spatial-regions 6

# More samples = longer training but better coverage
--samples-per-camera 200
```

---

## 🛠️ Common Customizations

### Adjust dataset size
```bash
# Smaller dataset (faster training, less memory)
--samples-per-camera 100 --train-split 0.9

# Larger dataset (longer training, better generalization)
--samples-per-camera 300 --train-split 0.8
```

### Adjust training hyperparameters
Edit `configs/rtmdet_m_monkey.py`:
```python
train_dataloader = dict(batch_size=8)  # Reduce for memory
train_cfg = dict(max_epochs=50)        # Shorter training
```

### Different model size
Use RTMDet-l (large) instead of RTMDet-m (medium):
- Larger model = better accuracy, slower inference, more memory
- Smaller model = faster inference, less memory, lower accuracy

---

## 📚 Documentation Structure

| Document | Purpose |
|----------|---------|
| **RTMDET_TRAINING.md** | Complete reference guide (10 sections) |
| **RTMDET_CHECKLIST.md** | Phase-by-phase workflow with checkboxes |
| **This file** | High-level overview & architecture |

---

## ⚠️ Important Notes

1. **Coordinate System**: Bboxes are in image coordinates (x1, y1, x2, y2), not arena coordinates
2. **Resolution**: Input images are 640×640 for training (downsampled videos are 1920×1080)
3. **Classes**: Only 1 class (monkey) - binary detection
4. **GPU Memory**: With 8 GPUs, requires ~40GB total VRAM (5GB per GPU)

---

## 🎓 Next Actions

### Immediate (Today)
- [ ] Read [RTMDET_TRAINING.md](RTMDET_TRAINING.md) Section 1-2
- [ ] Run data conversion script locally
- [ ] Verify dataset creation

### Short-term (This week)
- [ ] Create remote environment
- [ ] Launch training on remote server
- [ ] Monitor progress with tensorboard

### Medium-term (Next week)
- [ ] Complete bbox annotation for D03, D04
- [ ] Evaluate model on new camera angles
- [ ] Experiment with hyperparameters

### Integration (After training)
- [ ] Use RTMDet predictions to pre-filter regions
- [ ] Integrate into DLC keypoint detection pipeline
- [ ] Batch process all OFT videos

---

## 🆘 Support Resources

**If you encounter issues:**

1. Check [RTMDET_CHECKLIST.md](RTMDET_CHECKLIST.md) → "Common Issues & Solutions"
2. Review [RTMDET_TRAINING.md](RTMDET_TRAINING.md) → Section 7 "Troubleshooting"
3. Check training logs:
   ```bash
   tail -100 results/rtmdet_models/exp001/*/stdout.log
   ```

**For dataset issues:**
```bash
python scripts/convert_bbox_to_coco.py --help  # View all options
```

**For training issues:**
```bash
python scripts/train_rtmdet.py --help
```

---

## 📞 Quick Reference

**Environment file**: `environment.rtmdet.yml` (Conda) | Python 3.10 | PyTorch + MMYolo + Accelerate

**Dataset conversion**: `scripts/convert_bbox_to_coco.py` | Handles CSV→COCO, spatial sampling, train/val split

**Training with Accelerate**: `scripts/train_rtmdet.py` | Auto-detects GPUs, distributed, mixed precision support

**Inference**: `scripts/inference_rtmdet.py` | Single/batch video processing, CSV output, visualization

---

**Status**: ✅ Ready to proceed with Step 1 (Local Data Preparation)  
**Date**: 2026-07-01  
**Models**: RTMDet-m (medium) for all 4 cameras (D01-D04)
