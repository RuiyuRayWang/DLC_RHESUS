# Multi-Video Anchor Workflow

## TL;DR

```bash
conda activate video_tracking
cd /Users/ray/PROJECTS/PAL_ENHANCER

MONKEY=H1705011
DATE=2026-05-26
SESSION_DIR="results/oft_analysis/${MONKEY}/${DATE}"

# 1) Generate anchor table
python scripts/generate_time_anchor_template.py --dir "$SESSION_DIR"

# 2) Manually curate anchor_timestamp in:
#    ${SESSION_DIR}/label_time_anchors.csv
#    (VSPlayer, usually frame 0)

# 3) Fuse trajectory
python scripts/fuse_trajectories.py --monkey "$MONKEY" --date "$DATE"

# 4) Select analysis window
python scripts/select_time_window.py --monkey "$MONKEY" --date "$DATE"
```

## Why this exists

Cameras are not start-synchronized. Fusion must use confirmed absolute time.

Timing source of truth:
- `results/oft_analysis/<monkey>/<date>/label_time_anchors.csv`

## Step Details

### 1) Generate `label_time_anchors.csv`

```bash
python scripts/generate_time_anchor_template.py --dir results/oft_analysis/H1705011/2026-05-26
```

### 2) Curate `anchor_timestamp`

For each row in `label_time_anchors.csv`, verify/edit:
- `anchor_timestamp`
- `anchor_video_frame` (usually `0`)
- `method` (`human` recommended after confirmation)

### 3) Fuse trajectories

```bash
python scripts/fuse_trajectories.py --monkey H1705011 --date 2026-05-26
```

Outputs:
- `trajectory_realworld.csv`
- `movement_bouts.csv`
- `trajectory_summary.json`

### 4) Select quantification window

```bash
python scripts/select_time_window.py --monkey H1705011 --date 2026-05-26
```

Output:
- `selected_time_window.json`

## Troubleshooting

### Fusion skips files
Check for blank `anchor_timestamp` values in `label_time_anchors.csv`.

### Label CSV metadata is inconsistent (optional)

```bash
python scripts/backfill_label_timestamps.py \
  --session-dir results/oft_analysis/H1705011/2026-05-26 \
  --overwrite-existing
```
