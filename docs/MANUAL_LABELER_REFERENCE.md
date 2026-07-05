# Manual Labeler Reference

## Most Common Commands

```bash
conda activate video_tracking
cd /Users/ray/PROJECTS/PAL_ENHANCER

# Simple mode
python scripts/manual_labeler.py --camera D2

# Explicit multi-video mode
python scripts/manual_labeler.py --camera D3 \
  --video data/oft_rhp/H1705011/2026-05-25/D03_20260525164012.mp4 \
  --output results/oft_analysis/H1705011/2026-05-25/D03_20260525164012.csv \
  --sample-every 20
```

## Arguments

| Argument | Short | Default | Purpose |
|---|---|---|---|
| `--camera` | `-c` | interactive | Choose `D1`/`D2`/`D3`/`D4` |
| `--video` | `-v` | auto-detect | Set input video explicitly |
| `--output` | `-o` | auto | Set output CSV path |
| `--sample-every` | `-n` | 20 | Label every Nth frame |
| `--scale` | `-s` | 0.4 | Display scaling |
| `--fps` | | 20 | FPS used for `time_sec` |

## Controls

- Left-click: label and advance
- Right-click: label and stay
- M: mark no-monkey-visible and advance
- N / Right: next frame
- P / Left: previous frame
- U: undo current frame
- Q / ESC: save and quit

## Output schema

Each row is one sampled frame.

- `camera`
- `video_file`
- `video_timestamp`
- `video_frame`
- `time_sec`
- `centroid_x`
- `centroid_y`
- `monkey_visible`

Notes:
- `centroid_x = -1` and `centroid_y = -1` means no monkey visible.
- In anchor-based workflow, final fusion timing is determined by `label_time_anchors.csv`.

## Labeling guidance

- Click the center of the monkey head consistently.
- Use `M` for occluded/out-of-view frames.
- Keep `--sample-every` consistent within a session.

## Related docs

- `docs/MULTI_VIDEO_WORKFLOW.md`
- `docs/QUICKSTART_GUI.md`
