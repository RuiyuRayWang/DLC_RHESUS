# Data Preparation for Ultralytics RT-DETR

This document describes how to build a detection dataset for monkey bbox training
from:

- downsampled videos
- bbox CSV annotations from `scripts/bbox_labeler.py`

The workflow is implemented in:

- `scripts/prepare_ultralytics_bbox_dataset.py`

## Input Expectations

For the current OFT session:

- videos: `data/oft_rhp/videos_downsampled/H1705011/2026-06-16`
- CSVs: `results/bbox/H1705011/2026-06-16`

Filename mapping must match:

- video: `D01_20260616135117_downsampled.mp4`
- CSV: `bbox_D01_20260616135117_downsampled.csv`

The script also verifies `video_file` inside each CSV.

## HEVC (H.265) Note

These videos use HEVC. Random frame access can be unreliable in OpenCV for HEVC MP4.
To avoid decoding mismatches, this pipeline decodes frames sequentially and writes only
annotated frames.

## Output Layout (Ultralytics Detection)

The script creates:

- `images/train/*.jpg`
- `images/val/*.jpg`
- `labels/train/*.txt`
- `labels/val/*.txt`
- `data.yaml`

Label format per line:

- `<class_id> <x_center> <y_center> <width> <height>`
- normalized to `[0, 1]`
- class id `0` is `monkey`

## Build Dataset

```bash
conda activate det
python scripts/prepare_ultralytics_bbox_dataset.py \
  --videos-dir data/oft_rhp/videos_downsampled/H1705011/2026-06-16 \
  --csv-dir results/bbox/H1705011/2026-06-16 \
  --out-root datasets/ultralytics_bbox/H1705011_2026-06-16 \
  --include-background \
  --background-ratio 0.5 \
  --val-ratio 0.25 \
  --seed 42
```

## What Is Included

Rows are included only if:

- `monkey_visible == True`
- bbox coordinates are present and valid

Rows with missing/invalid boxes are skipped.

If `--include-background` is used:

- frames with `monkey_visible == False` can be sampled as background images
- background images get empty label files (`.txt` with zero lines)
- `--background-ratio` controls max sampled backgrounds per video as a ratio of
  positive frames (for example `0.5` means up to 1 background per 2 positives)

## Quick Sanity Checks

```bash
find datasets/ultralytics_bbox/H1705011_2026-06-16/images/train -name '*.jpg' | wc -l
find datasets/ultralytics_bbox/H1705011_2026-06-16/labels/train -name '*.txt' | wc -l
cat datasets/ultralytics_bbox/H1705011_2026-06-16/data.yaml
```

## Next Step: Train RT-DETR

Use:

- `scripts/train_rtdetr.py`

Example:

```bash
conda activate det
python scripts/train_rtdetr.py \
  --data datasets/ultralytics_bbox/H1705011_2026-06-16/data.yaml \
  --model rtdetr-l.pt \
  --epochs 100 \
  --imgsz 960 \
  --batch 16 \
  --device 0
```

8-GPU example:

```bash
python scripts/train_rtdetr.py \
  --data datasets/ultralytics_bbox/H1705011_2026-06-16/data.yaml \
  --model rtdetr-l.pt \
  --epochs 100 \
  --imgsz 960 \
  --batch 64 \
  --device 0,1,2,3,4,5,6,7 \
  --workers 16
```
