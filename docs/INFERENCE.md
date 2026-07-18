# INFERENCE Workflow

Last updated: 2026-07-18

This document tracks the current end-to-end OFT inference and trajectory workflow:
1. Downsample raw videos on HPC.
2. Extract RT-DETR frames on HPC (optionally tar per video).
3. Transfer frame tarballs.
4. Run multi-GPU RT-DETR inference from extracted frames.
5. Reconstruct and review trajectory.
6. Regenerate trajectory and build session-level reports.

## Documentation Updates

- The following docs are deprecated and archived under /tmp:
  - DEEPLABCUT_TRAINING.md
  - MANUAL_LABELER_REFERENCE.md
  - OFT_PIPELINE_PROGRESS.md
  - QT_SUBPROCESS_FIX.md
  - QUICK_START.md
  - VIDEO_LOADING_SUMMARY.md
- The old session note SESSION_LOG_2026-07-05.md has been consolidated into this file.
- Multi-camera reviewer usage is documented separately in docs/MULTICAM_REVIEWER.md.

## Session Progress Highlights

- Pipeline split is stable for large-scale runs:
  - Step 1: frame extraction from downsampled videos
  - Step 2: inference from pre-extracted frame folders
- Frame extraction now supports tar-per-video packaging for faster transfer.
- dSQ joblist generation supports extraction and tar workflow.
- RT-DETR inference supports multi-GPU sharding with fail-fast behavior.
- Interactive validate mode in bbox labeler is stable.
- Multi-camera reviewer gained synchronized editing and trajectory artifact review mode.
- Trajectory artifact candidates are now precomputed during reconstruction for faster reviewer launch.

## 1) Downsample on HPC

Transfer raw mp4 to HPC:

```bash
rsync -aW -R --info=progress2 --partial --ignore-existing /media/luolab/MyBook2/oft_rhp/H1705011/./2026-07-0[12]/D0*.mp4 wangruiyu@10.254.1.1:/GPFS/luominmin_lab_permanent/wangruiyu/GITHUB_REPO/DLC_RHESUS/data/oft_rhp/H1705011/
```

Parse downsample joblist:

```bash
source ~/DATA/environments/miniconda3/bin/activate det
cd ~/DATA/GITHUB_REPO/DLC_RHESUS
python scripts/parse_ffmpeg_joblist.py downsample \
  data/oft_rhp/H1705011/ \
  -j scripts/shell/joblist_downsample.txt \
  --recursive \
  --skip-existing \
  --script scripts/downsample_ffmpeg.py \
  --raw-video-root data/oft_rhp \
  --output-dir data/oft_rhp/videos_downsampled \
  --scale 0.5 \
  --gop 20 \
  --threads 32 \
  --crf 18 \
  --preset veryfast
```

Generate dSQ submission script:

```bash
module load dSQ
cd scripts/shell/
dsq --job-file joblist_downsample.txt -p q_cn -n 1 --cpus-per-task 32 --mem-per-cpu 4G
```

Edit generated slurm script and add environment/module setup:

```bash
#!/bin/bash
#SBATCH --output dsq-joblist_downsample_2026-07-04-%A_%3a-%N.out
#SBATCH --array 0-193
#SBATCH --job-name dsq-joblist_downsample_2026-07-04
#SBATCH -p q_cn -n 1 --cpus-per-task 32 --mem-per-cpu 4G

module purge
source /GPFS/luominmin_lab_permanent/wangruiyu/environments/miniconda3/bin/activate dlc
cd ~/DATA/GITHUB_REPO/DLC_RHESUS

/usr/nzx-cluster/apps/dSQ/dSQBatch.py /GPFS/luominmin_lab_permanent/wangruiyu/GITHUB_REPO/DLC_RHESUS/scripts/shell/joblist_downsample_2026-07-04.txt /GPFS/luominmin_lab_permanent/wangruiyu/GITHUB_REPO/DLC_RHESUS/scripts/shell
```

Submit:

```bash
sbatch dsq-joblist_downsample-$(date +%F).sh
```

## 2) Extract Frames on HPC

Parse extraction joblist (tar per video):

```bash
python scripts/parse_ffmpeg_joblist.py extract-frames \
  data/oft_rhp/videos_downsampled/H1705011 \
  -j scripts/shell/joblist_extract_frames_H1705011_tar.txt \
  --recursive \
  --skip-existing \
  --script scripts/prepare_ultralytics_bbox_dataset.py \
  --videos-root data/oft_rhp/videos_downsampled \
  --frames-out-root tmp/rtdetr_frames_2hz \
  --sample-fps 2 \
  --tar-per-video \
  --tar-mode tar
```

Optional filtering of stale dates:

```bash
grep -Ev '/2026-05-25/|/2026-06-23/|/2026-06-24/|/2026-06-25/|/2026-06-26/|/2026-06-27/' scripts/shell/joblist_extract_frames_H1705011_tar.txt > scripts/shell/joblist_extract_frames_H1705011_tar.filtered.txt
wc -l scripts/shell/joblist_extract_frames_H1705011_tar.filtered.txt
```

Generate dSQ script:

```bash
cd scripts/shell/
dsq --job-file joblist_extract_frames_H1705011_tar.filtered.txt -p q_cn -n 1 --cpus-per-task 2 --mem-per-cpu 3G
```

Edit slurm script and add setup:

```bash
#!/bin/bash
#SBATCH --output dsq-joblist_extract_frames_H1705011_tar-%A_%3a-%N.out
#SBATCH --array 0-480
#SBATCH --job-name dsq-joblist_extract_frames_H1705011_tar
#SBATCH -p q_cn -n 1 --cpus-per-task 2 --mem-per-cpu 3G

module purge
module load gcc/10.2.1
source /GPFS/luominmin_lab_permanent/wangruiyu/environments/miniconda3/bin/activate det
cd ~/DATA/GITHUB_REPO/DLC_RHESUS

/usr/nzx-cluster/apps/dSQ/dSQBatch.py /GPFS/luominmin_lab_permanent/wangruiyu/GITHUB_REPO/DLC_RHESUS/scripts/shell/joblist_extract_frames_H1705011_tar.txt /GPFS/luominmin_lab_permanent/wangruiyu/GITHUB_REPO/DLC_RHESUS/scripts/shell
```

Submit and check exit status:

```bash
sbatch dsq-joblist_extract_frames_H1705011_tar.filtered-$(date +%F).sh
awk '{count[$2]++} END {for (k in count) print k, count[k]}' job_14452817_status.tsv | sort -n
```

## 3) Data Transfer

Transfer tarballs back to local:

```bash
rsync -aW -R --info=progress2 --partial --ignore-existing wangruiyu@10.254.1.1:/GPFS/luominmin_lab_permanent/wangruiyu/GITHUB_REPO/DLC_RHESUS/tmp/rtdetr_frames_2hz/H1705011/./2026-07-0[12]/*.tar /media/luolab/MyBook2/rtdetr_frames_2hz/H1705011/
```

Parallel untar to SSD:

```bash
find /media/luolab/MyBook3/rtdetr_frames_2hz/H1705011/2026-07-01/ \
  -maxdepth 1 \
  -name '*_downsampled.tar' \
  -print0 | \
parallel -0 -j4 --eta '
out="/media/luolab/PortableSSD/rtdetr_frames_2hz/H1705011/2026-07-01/"
mkdir -p "$out"
echo "[$(date +%T)] Extracting {/.}"
tar --no-same-owner -xf {} -C "$out"
'
```

## 4) Multi-GPU Inference

Run inference on GPU server from extracted frames:

```bash
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

Known good defaults:

- sample-fps: 2
- imgsz: 960
- multi-GPU starting point: batch 8, chunk-size 128

## 5) Reconstruct Trajectory

Transfer predictions (if reconstructing on local workstation):

```bash
rsync -rW -R -e 'ssh -p 2026' ray@120.246.100.122:/home/ray/DLC_RHESUS/results/predictions/H1705011/./2026-07-01 /Users/ray/GITHUB_PROJECTS/DLC_RHESUS/results/predictions_bak/H1705011
```

First-time calibration GUI (if needed):

```bash
conda activate video_tracking
python scripts/stereo_calibrate_checkerboard.py \
  --images-dir results/stereo_calibration/extracted_frames \
  --board-cols 7 \
  --board-rows 4 \
  --square-size-cm 14 \
  --write-debug-overlays \
  --launch-gui \
  --gui-images-dir results/calibration
```

Trajectory reconstruction:

```bash
python scripts/reconstruct_trajectory.py \
  --monkey H1705011 \
  --date 2026-07-01 \
  --predictions-dir results/predictions/H1705011/2026-07-01 \
  --cal-dir results/calibration \
  --stereo-calibration-json results/stereo_calibration/stereo_calibration_checkerboard.json \
  --edge-adjust project
```

Artifact candidate export is generated during reconstruction and used by reviewer startup:

- trajectory_realworld.csv
- movement_bouts.csv
- trajectory_summary.json
- trajectory_artifact_candidates.csv

## 6) Review and Reporting

Launch multi-camera reviewer for interactive cleanup:

```bash
python scripts/review_multicam_predictions.py --monkey H1705011 --date 2026-07-01
```

Then rerun trajectory reconstruction after edits.

Generate multi-session HTML report:

```bash
python scripts/generate_oft_multi_session_report.py --monkey H1705011
```

## Related Docs

- docs/GPU_SETUP.md
- docs/DATA_PREPARATION.md
- docs/TRAINING.md
- docs/MULTICAM_REVIEWER.md
