# Session Log (2026-07-05)

This note summarizes the completed work in this chat so a new prompt session can continue immediately.

## Scope Completed

- Split RT-DETR workflow into 2 steps for large-scale runs:
  - Step 1: frame extraction on HPC
  - Step 2: inference on pre-extracted frames
- Added frame extraction enhancements for transfer efficiency:
  - one folder per video for extracted frames
  - optional one tarball per video (`--tar-per-video`)
- Added dSQ joblist support for extraction and tar packaging.
- Refactored RT-DETR inference to run from extracted frame folders (no input videos required).
- Added multi-GPU worker sharding for inference (`--device 0,1,2,...`).
- Fixed multi-GPU device routing issues and fail-fast behavior (no automatic fallback).
- Updated README with current extraction/inference workflow and untar instructions.
- Added bbox labeler `validate` mode for interactive prediction correction on extracted frames.
- Fixed validate-mode crash for `M` / `U` key handlers.
- Fixed labeler UI occlusion issue by moving header/footer to non-occluding bars.

## Key Script Changes

### 1) scripts/prepare_ultralytics_bbox_dataset.py

- Added `--mode inference` for frame extraction from videos.
- Inference output now follows per-video folder layout:
  - `<out_root>/<monkey_id>/<date>/<video_name>/<video_name>_f000123.jpg`
- Added tar packaging options:
  - `--tar-per-video`
  - `--tar-mode {tar,tar.gz}`
  - `--tar-remove-frames`

### 2) scripts/parse_ffmpeg_joblist.py

- Added `extract-frames` subcommand for dSQ generation.
- Added tar packaging passthrough flags for extract jobs.
- Improved `--skip-existing` logic:
  - in tar mode, skip when tar already exists
  - also skip when extracted frame folder already exists with image files

### 3) scripts/infer_rtdetr_batch.py

- Inference input changed to frame directories under `--frames-root`.
- Output remains per-video CSV:
  - `results/predictions/{monkey_id}/{date}/{video_name}_preds.csv`
- Supports multi-GPU sharding by directory across worker processes.
- Uses strict/fail-fast inference behavior for debugging.
- Current stable multi-GPU run verified by user:
  - `--device 0,1,2,3,4,5,6,7`
  - `--batch 8`
  - `--chunk-size 128`

### 4) scripts/bbox_labeler.py

- Added modes:
  - `annotate` (training label generation)
  - `validate` (interactive correction of model predictions)
- Validate mode behavior:
  - loads extracted JPG frames + prediction CSV
  - overlays predicted box per frame
  - allows frame-by-frame correction in GUI
  - saves corrected CSV
- Fixed crash in validate mode:
  - removed bad references to annotate-only state (`self.annotations` etc.)
  - `M` / `U` now modify validate prediction rows correctly
- UI fix:
  - header/footer panels no longer occlude 1920x1080 frame content

## Verified Runtime Outcomes

- Multi-GPU RT-DETR inference confirmed working with balanced 8-worker startup and outputs.
- Prediction CSVs successfully written under:
  - `results/predictions/H1705011/2026-06-27/..._preds.csv`
- GPU utilization observed on all 8 GPUs in `nvidia-smi` while inference is active.

## Canonical Commands

### Step 1: Extract frames (single monkey/date)

```bash
conda activate det
python scripts/prepare_ultralytics_bbox_dataset.py \
  --mode inference \
  --videos-dir data/oft_rhp/videos_downsampled/H1705011/2026-06-16 \
  --out-root tmp/rtdetr_frames_2hz \
  --sample-fps 2 \
  --recursive
```

### Step 1: Extract + tar per video

```bash
conda activate det
python scripts/prepare_ultralytics_bbox_dataset.py \
  --mode inference \
  --videos-dir data/oft_rhp/videos_downsampled/H1705011/2026-06-16 \
  --out-root tmp/rtdetr_frames_2hz \
  --sample-fps 2 \
  --recursive \
  --tar-per-video \
  --tar-mode tar \
  --tar-remove-frames
```

### dSQ joblist generation (extract + tar)

```bash
python scripts/parse_ffmpeg_joblist.py extract-frames \
  data/oft_rhp/videos_downsampled/H1705011 \
  -j scripts/shell/joblist_extract_frames_H1705011_tar.txt \
  --recursive \
  --script scripts/prepare_ultralytics_bbox_dataset.py \
  --videos-root data/oft_rhp/videos_downsampled \
  --frames-out-root tmp/rtdetr_frames_2hz \
  --sample-fps 2 \
  --tar-per-video \
  --tar-mode tar \
  --tar-remove-frames \
  --skip-existing
```

### Untar on GPU server (in-place)

```bash
find tmp/rtdetr_frames_2hz -type f -name '*.tar' -print0 \
  | xargs -0 -I{} sh -c 'tar -xf "$1" -C "$(dirname "$1")"' _ {}

find tmp/rtdetr_frames_2hz -type f -name '*.tar.gz' -print0 \
  | xargs -0 -I{} sh -c 'tar -xzf "$1" -C "$(dirname "$1")"' _ {}
```

If permission issues occur on external disks:

```bash
find /path/to/external/tmp/rtdetr_frames_2hz -type f -name '*.tar' -print0 \
  | sudo xargs -0 -I{} sh -c 'tar --no-same-owner -xf "$1" -C "$(dirname "$1")"' _ {}
```

### Step 2: Multi-GPU inference on extracted frames

```bash
conda activate det
python scripts/infer_rtdetr_batch.py \
  --model runs/detect/results/ultralytics/rtdetr_monkey_bbox_H1705011_2026-06-16_8gpu-3/weights/best.pt \
  --frames-root tmp/rtdetr_frames_2hz \
  --monkey-id H1705011 \
  --date 2026-06-27 \
  --device 0,1,2,3,4,5,6,7 \
  --imgsz 960 \
  --batch 8 \
  --chunk-size 128 \
  --out-root results/predictions \
  --overwrite
```

### Validate predictions interactively

```bash
python scripts/bbox_labeler.py \
  --mode validate \
  --frames-dir tmp/rtdetr_frames_2hz/H1705011/2026-06-27/D01_20260627134058_downsampled \
  --predictions-csv results/predictions/H1705011/2026-06-27/D01_20260627134058_downsampled_preds.csv
```

## Operational Notes

- JPEG transfer bottleneck is mostly file-count metadata overhead, not compression ratio.
- Per-video tar packaging significantly improves transfer throughput and manageability.
- For rsync with pre-compressed JPGs, avoid `-z` unless measured otherwise.

## Known Good Defaults (as of this session)

- Extraction sampling: `--sample-fps 2`
- Inference size: `--imgsz 960`
- Multi-GPU batch/chunk starting point: `--batch 8 --chunk-size 128`

## Suggested Next Follow-ups

- Add a short README section for bbox validate mode usage and hotkeys.
- Add optional preflight check in inference script (small test batch per GPU before full run).
- Add a utility to overlay predictions into QC videos after validate corrections.
