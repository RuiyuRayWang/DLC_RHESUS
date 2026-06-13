# Quick Start - GUI Tools

Use this for OpenCV GUI tools only (`calibrate_cameras.py`, `manual_labeler.py`).

## Rules

- Run commands directly in an interactive terminal.
- Do not run via `conda run`, shell wrappers, or piped commands.

## Setup

```bash
cd /Users/ray/PROJECTS/PAL_ENHANCER
conda activate video_tracking
```

## Camera calibration

```bash
python scripts/calibrate_cameras.py --camera D2 --frame-sec 30
# or
python scripts/calibrate_cameras.py --all --frame-sec 30
```

Calibration controls:
- Left-click: label and advance
- Right-click: label and stay
- N / Right: next point
- P / Left: previous point
- U: undo current point
- C: compute calibration
- S: save
- Q / ESC: save and quit

Label the 4 floor corners for the camera quadrant. Top-plane points are optional.

## Manual labeling

```bash
python scripts/manual_labeler.py --camera D2
```

Change sampling if needed:

```bash
python scripts/manual_labeler.py --camera D2 --sample-every 10
python scripts/manual_labeler.py --camera D2 --sample-every 40
```

Labeling controls:
- Left-click: label and advance
- Right-click: label and stay
- M: mark no-monkey-visible and advance
- N / Right: next frame
- P / Left: previous frame
- U: undo
- Q / ESC: save and quit

Click target: center of the monkey head (keep this consistent).

## If GUI fails

```bash
python scripts/check_opencv_display.py
```

If remote, use X11 forwarding (`ssh -X`).

## Next step after GUI

Proceed with anchor curation and fusion:
- `docs/MULTI_VIDEO_WORKFLOW.md`
