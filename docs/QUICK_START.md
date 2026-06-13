# Quick Start - OFT (Daily Runbook)

Use this when you just want to run a session end-to-end.

## 0) Environment

```bash
conda activate video_tracking
cd /Users/ray/PROJECTS/PAL_ENHANCER
```

## 1) (Optional) Calibrate camera(s)

```bash
python scripts/calibrate_cameras.py --camera D3 --frame-sec 30
# or
python scripts/calibrate_cameras.py --all --frame-sec 30
```

## 2) Label videos

```bash
python scripts/manual_labeler.py --camera D2
```

For multi-video sessions, use explicit input/output:

```bash
python scripts/manual_labeler.py --camera D3 \
  --video data/oft_rhp/H1705011/2026-05-25/D03_20260525164012.mp4 \
  --output results/oft_analysis/H1705011/2026-05-25/D03_20260525164012.csv \
  --sample-every 20
```

## 3) Build and curate anchor table

```bash
python scripts/generate_time_anchor_template.py --dir results/oft_analysis/H1705011/2026-05-25
```

Then confirm `anchor_timestamp` values in:
- `results/oft_analysis/H1705011/2026-05-25/label_time_anchors.csv`

## 4) Fuse trajectory

```bash
python scripts/fuse_trajectories.py --monkey H1705011 --date 2026-05-25
```

## 5) Select analysis window

```bash
python scripts/select_time_window.py --monkey H1705011 --date 2026-05-25
```

## Main outputs

- `results/oft_analysis/H1705011/2026-05-25/trajectory_realworld.csv`
- `results/oft_analysis/H1705011/2026-05-25/movement_bouts.csv`
- `results/oft_analysis/H1705011/2026-05-25/trajectory_summary.json`
- `results/oft_analysis/H1705011/2026-05-25/selected_time_window.json`

## See also

- `docs/MULTI_VIDEO_WORKFLOW.md` for anchor-first details
- `docs/MANUAL_LABELER_REFERENCE.md` for labeler controls/options
- `docs/QUICKSTART_GUI.md` for GUI troubleshooting
