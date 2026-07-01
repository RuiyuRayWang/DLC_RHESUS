# RTMDet Training Checklist & Quick Reference

## Phase 1: Local Data Preparation ✓ (You are here)

- [ ] Verify bbox CSV files exist at `results/bbox/H1705011/2026-06-16/`
  ```bash
  ls -lh results/bbox/H1705011/2026-06-16/
  # Should show: bbox_D01_*.csv, bbox_D02_*.csv, etc.
  ```

- [ ] Verify downsampled videos exist
  ```bash
  ls -lh data/oft_rhp/videos_downsampled/H1705011/2026-06-16/
  # Should show: D01_*.mp4, D02_*.mp4, etc.
  ```

- [ ] Run data conversion script
  ```bash
  python scripts/convert_bbox_to_coco.py \
      --csv-dir results/bbox/H1705011/2026-06-16 \
      --video-dir data/oft_rhp/videos_downsampled \
      --output-dir data/rtmdet_dataset \
      --n-spatial-regions 4 \
      --samples-per-camera 150
  ```

- [ ] Verify COCO dataset created
  ```bash
  tree data/rtmdet_dataset/ -L 2 -d
  # Should show: D01/, D02/, D03/, D04/, images subdirs
  
  ls -lh data/rtmdet_dataset/*.json
  # Should show: annotations_train.json, annotations_val.json
  ```

- [ ] Check dataset statistics
  ```bash
  python -c "
  import json
  with open('data/rtmdet_dataset/annotations_train.json') as f:
      data = json.load(f)
      print(f'Train: {len(data[\"images\"])} images, {len(data[\"annotations\"])} bboxes')
  with open('data/rtmdet_dataset/annotations_val.json') as f:
      data = json.load(f)
      print(f'Val: {len(data[\"images\"])} images, {len(data[\"annotations\"])} bboxes')
  "
  ```

---

## Phase 2: Remote Server Environment Setup

- [ ] Rsync dataset to remote server
  ```bash
  rsync -azv --progress data/rtmdet_dataset/ \
      user@remote:/path/to/DLC_RHESUS/data/rtmdet_dataset/
  ```

- [ ] Rsync environment file
  ```bash
  rsync -azv environment.rtmdet.yml \
      user@remote:/path/to/DLC_RHESUS/
  ```

- [ ] On remote server: Create conda environment
  ```bash
  cd /path/to/DLC_RHESUS
  conda env create -f environment.rtmdet.yml
  ```

- [ ] Verify GPU setup on remote
  ```bash
  conda activate rtmdet
  python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
  ```

- [ ] Verify MMYolo installation
  ```bash
  python -c "import mmyolo; print(mmyolo.__version__)"
  python -c "from mmyolo.apis import init_detector, inference_detector; print('✓ APIs ready')"
  ```

---

## Phase 3: Training

### Pre-training checklist

- [ ] Verify all required files on remote
  ```bash
  ls -lh data/rtmdet_dataset/annotations*.json
  ls -lh configs/rtmdet_m_monkey.py
  ls -lh scripts/train_rtmdet.py
  ```

- [ ] Start screen session for persistent training
  ```bash
  screen -S rtmdet_train
  ```

### Launch multi-GPU training with Accelerate

- [ ] Run distributed training (8 GPUs with Accelerate)
  ```bash
  cd /path/to/DLC_RHESUS
  conda activate rtmdet
  
  accelerate launch scripts/train_rtmdet.py \
      --config configs/rtmdet_m_monkey.py \
      --work-dir results/rtmdet_models/exp001
  ```
  
  Or with explicit GPU specification:
  ```bash
  accelerate launch --multi_gpu --num_processes=8 scripts/train_rtmdet.py \
      --config configs/rtmdet_m_monkey.py \
      --work-dir results/rtmdet_models/exp001
  ```

### Monitor training

- [ ] In another terminal, monitor GPU usage
  ```bash
  watch -n 1 nvidia-smi
  ```

- [ ] Monitor training logs
  ```bash
  tail -f results/rtmdet_models/exp001/*/stdout.log
  ```

- [ ] Launch tensorboard (optional)
  ```bash
  tensorboard --logdir results/rtmdet_models/exp001 --port 6006
  # Access: http://remote_ip:6006
  ```

---

## Phase 4: Post-Training

- [ ] Find best checkpoint
  ```bash
  ls -lh results/rtmdet_models/exp001/ | grep best
  # Should show: best_coco_bbox_mAP_epoch_XX.pth
  ```

- [ ] Evaluate on test set (included in training)
  ```bash
  # Check final metrics in training log
  tail -100 results/rtmdet_models/exp001/*/stdout.log | grep -i "test"
  ```

- [ ] Rsync checkpoints back to local (optional)
  ```bash
  rsync -azv --progress \
      user@remote:/path/to/DLC_RHESUS/results/rtmdet_models/ \
      ./results/rtmdet_models/
  ```

---

## Phase 5: Inference & Evaluation

### On local machine (or remote)

- [ ] Run inference on new videos
  ```bash
  python scripts/inference_rtmdet.py \
      --config configs/rtmdet_m_monkey.py \
      --checkpoint results/rtmdet_models/exp001/best_coco_bbox_mAP_epoch_XX.pth \
      --video data/oft_rhp/videos_downsampled/H1705011/2026-06-16/*.mp4 \
      --output results/rtmdet_inference/ \
      --score-thr 0.5 \
      --save-video
  ```

- [ ] Review generated CSVs
  ```bash
  ls -lh results/rtmdet_inference/detections_*.csv
  ```

- [ ] Compare with manual annotations (optional)
  ```bash
  python scripts/evaluate_rtmdet.py \
      --predictions results/rtmdet_inference/detections_*.csv \
      --ground-truth results/bbox/H1705011/2026-06-16/*.csv
  ```

---

## Useful Quick Commands

### Monitor Training
```bash
# GPU memory/usage
nvidia-smi -l 1

# Check if training is active
ps aux | grep train_rtmdet

# View last 50 lines of training
tail -50 results/rtmdet_models/exp001/*/stdout.log

# Count images per camera
for cam in D01 D02 D03 D04; do
  echo "$cam: $(find data/rtmdet_dataset/$cam/images -type f | wc -l) images"
done
```

### Debugging
```bash
# Verify dataset format
python -c "
import json
with open('data/rtmdet_dataset/annotations_train.json') as f:
    data = json.load(f)
    img = data['images'][0]
    print('Sample image:', img)
    ann = data['annotations'][0]
    print('Sample annotation:', ann)
"

# Test single image inference
python -c "
from mmyolo.apis import init_detector, inference_detector
import cv2
model = init_detector('configs/rtmdet_m_monkey.py', 'results/rtmdet_models/exp001/best_*.pth', device='cuda:0')
img = cv2.imread('data/rtmdet_dataset/D01/images/D01_*.jpg')
result = inference_detector(model, img)
print('Detections:', result.pred_instances.scores)
"
```

### Troubleshooting
```bash
# Kill all training processes
pkill -f train_rtmdet.py

# Remove incomplete checkpoints (if training crashed)
rm results/rtmdet_models/exp001/epoch_*.pth
ls results/rtmdet_models/exp001/best*.pth  # Keep only the best

# Check available space
du -sh data/rtmdet_dataset/
df -h

# Reinitialize conda environment (if having issues)
conda env remove -n rtmdet
conda env create -f environment.rtmdet.yml
```

---

## Key Files & Locations

| File/Directory | Purpose |
|---|---|
| `environment.rtmdet.yml` | Conda environment definition |
| `scripts/convert_bbox_to_coco.py` | CSV → COCO converter with spatial sampling |
| `scripts/train_rtmdet.py` | Multi-GPU training script |
| `scripts/inference_rtmdet.py` | Inference & visualization |
| `configs/rtmdet_m_monkey.py` | RTMDet-m model config |
| `docs/RTMDET_TRAINING.md` | Full documentation |
| `data/rtmdet_dataset/` | COCO dataset (after conversion) |
| `results/rtmdet_models/` | Trained checkpoints & logs |

---

## Common Issues & Solutions

### CUDA Out of Memory
**Error:** `RuntimeError: CUDA out of memory`
- [ ] Reduce batch size: Edit `configs/rtmdet_m_monkey.py`, set `train_dataloader.batch_size = 8`
- [ ] Use mixed precision: Add `--amp` flag to training command

### Dataset Not Found
**Error:** `FileNotFoundError: No such file or directory: 'data/rtmdet_dataset/...'`
- [ ] Verify conversion completed: `ls data/rtmdet_dataset/annotations*.json`
- [ ] Verify rsync to remote: `ssh remote "ls /path/to/data/rtmdet_dataset/"`

### Import Errors
**Error:** `ModuleNotFoundError: No module named 'mmyolo'`
- [ ] Reinstall: `pip install --upgrade mmyolo`
- [ ] Or rebuild env: `conda env remove -n rtmdet && conda env create -f environment.rtmdet.yml`

### GPU Not Detected
**Error:** `CUDA device not available`
- [ ] Check driver: `nvidia-smi`
- [ ] Check PyTorch: `python -c "import torch; print(torch.cuda.is_available())"`

---

## Next Steps (After successful training)

1. **Continue bbox annotation** for remaining cameras
2. **Experiment with hyperparameters**:
   - Try `--samples-per-camera 200` or `300`
   - Adjust `--n-spatial-regions` to 6-8
3. **Evaluate model performance**:
   - Compare with manual annotations
   - Calculate IoU metrics
4. **Integration**:
   - Use RTMDet predictions to pre-filter regions before DLC keypoint detection
   - Batch process all OFT videos

---

## Contacts & References

- **RTMDet Paper**: https://arxiv.org/abs/2212.07784
- **MMYolo Docs**: https://github.com/open-mmlab/mmyolo
- **COCO API**: https://github.com/cocodataset/cocoapi

---

**Last updated**: 2026-07-01  
**Status**: Ready for Phase 1 (Local Data Preparation)
