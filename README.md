# DLC_RHESUS

Automate monkey tracking using your manual labels with artificial neural networks.

## Overview

The goal is to reconstruct real-world (cm) movement trajectories for a macaque
monkey in a **322 × 322 cm open-field arena**, using video recordings from
four wide-angle cameras (D1–D4) positioned above each quadrant.

```
(0,0) ────────── (161,0) ────────── (322,0)
  │                  │                  │
  │   D1 (top-left)  │  D2 (top-right)  │
  │                  │                  │
(0,161) ──────── (161,161) ──────── (322,161)
  │                  │                  │
  │  D3 (bottom-left)│ D4 (bottom-right)│
  │                  │                  │
(0,322) ──────── (161,322) ──────── (322,322)
```

Arena height: **105 cm**.

---

## Pipeline Summary

The current production path uses Ultralytics RT-DETR for bbox detection, then
multiview trajectory reconstruction and review.

1. Downsample raw videos.
2. Manually annotate bbox CSVs.
3. Build Ultralytics train/val dataset.
4. Train RT-DETR.
5. Extract frames and run batch inference.
6. Reconstruct trajectory.
7. Review and clean predictions/trajectory.
8. Generate multi-session report.

See:

- [docs/TRAINING.md](docs/TRAINING.md)
- [docs/INFERENCE.md](docs/INFERENCE.md)
- [docs/MULTICAM_REVIEWER.md](docs/MULTICAM_REVIEWER.md)

## Quick Start (RT-DETR)

### 1) Prepare training dataset

```bash
conda activate det
python scripts/prepare_ultralytics_bbox_dataset.py \
  --mode train \
  --videos-dir data/oft_rhp/videos_downsampled/H1705011/2026-06-16 \
  --csv-dir data/oft_rhp/annotations/H1705011/2026-06-16 \
  --out-root datasets/ultralytics_bbox/H1705011_2026-06-16 \
  --include-background \
  --background-ratio 0.5 \
  --val-ratio 0.2 \
  --seed 42
```

### 2) Train RT-DETR

```bash
conda activate det
python scripts/train_rtdetr.py \
  --data datasets/ultralytics_bbox/H1705011_2026-06-16/data.yaml \
  --model rtdetr-l.pt \
  --epochs 100 \
  --imgsz 960 \
  --batch 16 \
  --device 0 \
  --project results/ultralytics \
  --name rtdetr_monkey_bbox_H1705011_2026-06-16
```

Note:

- On Amax, place `yolo26n.pt` in the repo root before training checks
  if github.com download is blocked.

### 3) Run inference from extracted frames

```bash
conda activate det
python scripts/infer_rtdetr_batch.py \
  --model runs/detect/results/ultralytics/rtdetr_monkey_bbox_H1705011_2026-06-16_8gpu-3/weights/best.pt \
  --frames-root tmp/rtdetr_frames_2hz \
  --monkey-id H1705011 \
  --date 2026-07-01 \
  --device 0,1,2,3,4,5,6,7 \
  --imgsz 960 \
  --batch 8 \
  --chunk-size 128 \
  --out-root results/predictions \
  --overwrite
```

### 4) Reconstruct and review

```bash
conda activate video_tracking
python scripts/reconstruct_trajectory.py \
  --monkey H1705011 \
  --date 2026-07-01 \
  --predictions-dir results/predictions/H1705011/2026-07-01 \
  --cal-dir results/calibration \
  --stereo-calibration-json results/stereo_calibration/stereo_calibration_checkerboard.json \
  --edge-adjust project

python scripts/review_multicam_predictions.py --monkey H1705011 --date 2026-07-01
python scripts/generate_oft_multi_session_report.py --monkey H1705011
```

## Project Structure

```
DLC_RHESUS/
├── data/
│   ├── oft_rhp/
│   │   └── 134569R/                # Raw video files
│   └-- README.md
│
├── notebooks/
│   └── oft_video_analysis.ipynb    # Main analysis notebook
│
├── scripts/
│   ├── stereo_calibrate_checkerboard.py # Stereo calibration
│   ├── bbox_labeler.py             # Bounding box labeler
│   ├── prepare_ultralytics_bbox_dataset.py  # Build YOLO-format dataset from bbox CSV + videos
│   ├── train_rtdetr.py             # RT-DETR training entrypoint
│   ├── infer_rtdetr_batch.py       # Batch RT-DETR inference from extracted frames
│   ├── reconstruct_trajectory.py    # Multiview trajectory reconstruction
│   ├── review_multicam_predictions.py # Interactive multicam reviewer and editing
│   ├── generate_oft_multi_session_report.py # Session report generation
│   ├── downsample_ffmpeg.py        # Downsample 4K raw video to arbitrary resolution
│   └── parse_ffmpeg_joblist.py     # Parse dSQ joblist to submit on HPC cluster
│
├── results/
│   ├── oft_analysis/               # Reconstructed trajectories and reports
│   │   ├── H1705011/               # Monkey ID
│   │   │   └── 2026-05-25/         # Session date
│   │   │       ├── trajectory_realworld.csv
│   │   │       ├── movement_bouts.csv
│   │   │       ├── trajectory_artifact_candidates.csv
│   │   │       └── trajectory_summary.json
│   ├── predictions/                # RT-DETR per-video prediction CSVs
│       └── H1705011/
│           └── 2026-05-25/
│               ├── D01_*_preds.csv
│               ├── D02_*_preds.csv
│               ├── D03_*_preds.csv
│               └── D04_*_preds.csv
│
├── docs/
│   ├── DATA_PREPARATION.md
│   ├── TRAINING.md
│   ├── INFERENCE.md
│   └── MULTICAM_REVIEWER.md
├── environment.video_tracking.yml
└── .gitignore
```

## What We Verified

### Video Format
- ✅ **Codec**: HEVC (H.265) - confirmed working with OpenCV
- ✅ **Resolution**: 3840x2160 (4K)
- ✅ **Frame Rate**: 20 FPS
- ✅ **Sequential reading**: Works perfectly
- ⚠️  **Random access**: Has HEVC decoding issues (expected, not needed)
