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

## TL;DR

### Establish camera calibration parameters

(Note: the `docs/OFT_PIPELINE_PROGRESS.md` needs update.)
```
python scripts/calibrate_cameras.py --camera D3 --video /path/to/D03_*.mp4
```

Consider upgrade the calibration strategy with checkerboard (later).

---

Monkey positions in the OFT are traced either manually or detected by artificial neural networks.

### DeepLabCut (keypoint detection):

#### Training

- Manually label monkey head as keypoints (manual_labeler.py --mode head, launch in `video_tracking` conda env)
- Downsample videos
    - rsync videos to remote HPC server
    - use dSQ for batch downsampling (parse_ffmpeg_joblist.py & downsample_ffmpeg.py)
    - NOTE: if manual labels were generated on raw video, 
            the csv's need to be updated with the same downsampling ratio (downsample_ffmpeg.py)
    - rsync remote downsampled videos to local
- Setup DLC project (setup_deeplabcut_project.py), requires: 
    - downsampled videos (for frame extraction)
    - csv's with manual labels for monkey head
- Move DLC project to remote GPU server for training
- For each camera, train a dedicated DLC network (Resnet-50)
    - use `screen` for session control
    - edit the `pytorch_config.yaml` for further control of training sessions of the project
      `save_epochs: 10`, `batch_size: 16`, `epochs: 100`, `display_iters: 100`, `gpus: [0]` or `gpus: [1]`
    - train the network:
    ```
    (dlc)$ python
    >>> import deeplabcut
    >>> config = 'deeplabcut_projects/monkey_oft_tracking_D1-luolab-2026-06-19/config.yaml'
    >>> deeplabcut.create_training_dataset(config)  # Create training dataset
    >>> deeplabcut.train_network(config, shuffle=1)  # Train network
    ```
    - to resume from previous checkpoint (snapshot-xx.pt)
    Edit `pytorch_config.yaml`, add the `resume_training_from` key, e.g.:
    `resume_training_from: 'deeplabcut_projects/monkey_oft_tracking_D1-luolab-2026-06-19/dlc-models-pytorch/iteration-0/monkey_oft_tracking_D1Jun19-trainset95shuffle1/train/snapshot-100.pt'`
    and train with:
    ```
    deeplabcut.train_network(
      config, 
      shuffle=1,  # mandatory, or else error
      device="cuda",
      gputouse=0  # explicitly set gpu to use
    )
    ```

#### Inference

- Manually label visible/invisible frames (manual_labeler.py --mode visibility, launch in `video_tracking` conda env)
- Segment videos  # TODO
    - use dSQ for batch segmentation (parse_ffmpeg_joblist.py & segment_ffmpeg.py)
    - rsync remote segmented videos to local
- Use the pretrained DLC models for monkey head inference  # TODO

### Ultralytics (YOLO/RT-DETR for bbox detection)

- Downsample videos
    - rsync videos to remote HPC server
    - use dSQ for batch downsampling (parse_ffmpeg_joblist.py & downsample_ffmpeg.py)
    - NOTE: if manual labels were generated on raw video, 
            the csv's need to be updated with the same downsampling ratio (downsample_ffmpeg.py)
    - rsync remote downsampled videos to local
- Manually label bbox (bbox_labeler.py, launch in `video_tracking` conda env)
- Prepare Ultralytics dataset from bbox CSV + downsampled videos (see `docs/DATA_PREPARATION.md`)
  using sequential frame decoding (important for HEVC random-access instability):
  ```bash
  conda activate det
  python scripts/prepare_ultralytics_bbox_dataset.py \
    --videos-dir data/oft_rhp/videos_downsampled/H1705011/2026-06-16 \
    --csv-dir results/bbox/H1705011/2026-06-16 \
    --out-root datasets/ultralytics_bbox/H1705011_2026-06-16 \
    --include-background \
    --background-ratio 0.1 \
    --val-ratio 0.125 \
    --seed 42
  ```
  Notes:
  - This expects filename mapping like:
    `D01_20260616135117_downsampled.mp4` -> `bbox_D01_20260616135117_downsampled.csv`
  - Output follows Ultralytics detection layout:
    `images/train`, `images/val`, `labels/train`, `labels/val`, `data.yaml`
  - Only rows with `monkey_visible=True` and valid bbox coordinates are used.
  - With `--include-background`, sampled `monkey_visible=False` frames are exported as
    background samples (empty `.txt` labels).

- Train [RT-DETR](https://docs.ultralytics.com/models/rtdetr) (preferred):
  ```bash
  conda activate det
python scripts/train_rtdetr.py \
  --data datasets/ultralytics_bbox/H1705011_2026-06-16/data.yaml \
  --model rtdetr-l.pt \
  --epochs 100 \
  --imgsz 960 \
  --batch 8 \
  --device 0 \
  --project results/ultralytics \
  --name rtdetr_monkey_bbox_H1705011_2026-06-16
  ```

- Train on 8 GPUs (DDP):
  ```bash
  conda activate det
  python scripts/train_rtdetr.py \
    --data datasets/ultralytics_bbox/H1705011_2026-06-16/data.yaml \
    --model rtdetr-l.pt \
    --epochs 100 \
    --imgsz 960 \
    --batch 64 \
    --device 0,1,2,3,4,5,6,7 \
    --workers 16 \
    --project results/ultralytics \
    --name rtdetr_monkey_bbox_H1705011_2026-06-16_8gpu
  ```
  Notes:
  - Keep `batch` as global batch size; tune it to avoid OOM.
  - If OOM occurs, lower `--batch` (for example, 48 or 32).

- Inference with the trained model and integrate bbox detections into OFT tracking.

### sapiens 2 (pose-estimation)

- After detection, do pose-estimation for barrier task # TODO


## Project Structure

```
DLC_RHESUS/
├── data/
│   ├── oft_rhp/
│   │   └── 134569R/                # Raw video files
│   └-- README.md
│
├── deeplabcut_projects/
│   └── monkey_oft_tracking_D1-luolab-2026-06-19/
│       ├── config.yaml
│       ├── labeled-data/           # 141 video folders with PNGs + H5 labels
│       └── dlc-models-pytorch/
│           └── iteration-0/
│               └── monkey_oft_trackingJun16-trainset95shuffle5/
│                   ├── train/      # Training config & checkpoints
│                   │   ├── pytorch_config.yaml
│                   └── test/       # Test data
│
├── notebooks/
│   └── oft_video_analysis.ipynb    # Main analysis notebook
│
├── scripts/
│   ├── calibrate_cameras.py        # Camera calibration
│   ├── manual_labeler.py           # Monkey head labeler
│   ├── bbox_labeler.py             # Bounding box labeler
│   ├── prepare_ultralytics_bbox_dataset.py  # Build YOLO-format dataset from bbox CSV + videos
│   ├── train_rtdetr.py             # RT-DETR training entrypoint
│   ├── setup_deeplabcut_project.py # Setup DLC project
│   ├── downsample_ffmpeg.py        # Downsample 4K raw video to arbitrary resolution
│   ├── segment_ffmpeg.py           # Segment video based on visibility (previously preprocess_ffmpeg.py)
│   └── parse_ffmpeg_joblist.py     # Parse dSQ joblist to submit on HPC cluster
│
├── results/
│   ├── oft_analysis/               # Output directory, keypoint labels
│   │   ├── H1705011/               # Monkey ID
│   │   │   └── 2026-05-25/         # Session date
│   │   │       ├── D01_*.csv       # Camera D1 labels
│   │   │       ├── D02_*.csv       # Camera D2 labels
│   │   │       ├── D03_*.csv       # Camera D3 labels
│   │   │       └── D04_*.csv       # Camera D4 labels
│   │   ├── video_metadata.csv
│   │   ├── behavioral_metrics_all_videos.csv
│   │   ├── tracking_data_all_videos.csv
│   │   └── *.png figures
│   ├── visible_frames/             # Output directory, visibility labels
│   │   └── H1705011/
│   │       └── 2026-05-25/         # Same structure as in oft_analysis/
│   └── bbox/                       # Output directory, bounding-box labels
│       └── H1705011/
│           └── 2026-05-25/         # Same structure as in oft_analysis/
│
├── docs/
│   └── VIDEO_LOADING_SUMMARY.md  # Technical details
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
