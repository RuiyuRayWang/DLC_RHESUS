# TRAINING Workflow

Last updated: 2026-07-18

This document describes the RT-DETR training workflow used in this project.

## Overview

1. Downsample raw videos.
2. Manually annotate monkey bbox labels.
3. Prepare Ultralytics training dataset.
4. Transfer dataset to GPU server.
5. Train RT-DETR.

## 1) Downsample Video

Use scripts/downsample_ffmpeg.py to downsample HEVC 4K raw video before annotation and training dataset prep.

Typical downsample dispatch for large batches is done via scripts/parse_ffmpeg_joblist.py downsample and dSQ; see docs/INFERENCE.md for a full HPC joblist/slurm example.

## 2) Manual Annotation

Use scripts/bbox_labeler.py to annotate bbox labels for each downsampled video.

Expected output files:

- bbox_D01_xxxxxxxxxxxxxx_downsampled.csv
- bbox_D02_xxxxxxxxxxxxxx_downsampled.csv
- bbox_D03_xxxxxxxxxxxxxx_downsampled.csv
- bbox_D04_xxxxxxxxxxxxxx_downsampled.csv

Important:

- Include background data with monkey_visible=false rows (no monkey present) so training sees hard negatives.
- Keep annotations consistent across cameras and dates.

## 3) Dataset Preparation

Use scripts/prepare_ultralytics_bbox_dataset.py in train mode to build Ultralytics-format train/val datasets.

Example:

```bash
conda activate det
python scripts/prepare_ultralytics_bbox_dataset.py \
  --mode train \
  --videos-dir data/oft_rhp/videos_downsampled/H1705011/2026-06-16 \
  --csv-dir data/oft_rhp/annotations/H1705011/2026-06-16 \
  --out-root datasets/ultralytics_bbox/H1705011_2026-06-16 \
  --val-ratio 0.2 \
  --seed 42 \
  --include-background \
  --background-ratio 0.5
```

Expected output:

- datasets/ultralytics_bbox/.../images/train
- datasets/ultralytics_bbox/.../images/val
- datasets/ultralytics_bbox/.../labels/train
- datasets/ultralytics_bbox/.../labels/val
- datasets/ultralytics_bbox/.../data.yaml

## 4) Transfer Dataset to GPU Server

Transfer the generated dataset folder to your GPU training machine.

Example:

```bash
rsync -aW --info=progress2 --partial datasets/ultralytics_bbox/H1705011_2026-06-16/ ray@amax:~/DLC_RHESUS/datasets/ultralytics_bbox/H1705011_2026-06-16/
```

## 5) Train RT-DETR

Run scripts/train_rtdetr.py on the GPU server.

Example:

```bash
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

For multi-GPU, pass a device list such as --device 0,1,2,3.

## Offline Checkpoint Note (Amax)

Amax may fail to download assets directly from github.com during checks. Place this file in the project root before training:

- https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt

Expected location:

- DLC_RHESUS/yolo26n.pt

## Validation and QC

- Inspect training logs and metrics under results/ultralytics/<run_name>.
- Keep a separate holdout date/session for final sanity checks.
- After training, run inference with scripts/infer_rtdetr_batch.py and verify predictions in scripts/review_multicam_predictions.py.

## Related Docs

- docs/INFERENCE.md
- docs/DATA_PREPARATION.md
- docs/GPU_SETUP.md
- docs/MULTICAM_REVIEWER.md