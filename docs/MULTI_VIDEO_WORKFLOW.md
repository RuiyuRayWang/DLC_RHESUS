# Multi-Video Timestamp-Based Workflow

## Problem
The 4 cameras record **unsynchronized** videos at different times. Videos have timestamps in their filenames (e.g., `D03_20260525164012.mp4` = started at 16:40:12).

## Solution
The pipeline now uses **absolute timestamps** instead of frame numbers to synchronize trajectories across cameras.

---

## 1. Calibration (once per camera)

Calibration only needs to be done **once per camera** (applies to all videos from that camera):

```bash
python scripts/calibrate_cameras.py --camera D3 --frame-sec 30
```

Output: `results/calibration/calibration_D3.json`

---

## 2. Manual Labeling (each video separately)

### For cameras with a single video:
```bash
python scripts/manual_labeler.py --camera D2
```

### For cameras with multiple videos:
Use `--video` and `--output` flags to label each video separately:

```bash
# D3 has 7 videos - label each one:
python scripts/manual_labeler.py --camera D3 \
  --video data/oft_rhp/H1705011/2026-05-25/D03_20260525164012.mp4 \
  --output results/oft_analysis/H1705011/2026-05-25/labels_D3_164012.csv \
  --sample-every 20

python scripts/manual_labeler.py --camera D3 \
  --video data/oft_rhp/H1705011/2026-05-25/D03_20260525170340.mp4 \
  --output results/oft_analysis/H1705011/2026-05-25/labels_D3_170340.csv \
  --sample-every 20

# ... repeat for remaining D3 videos
```

**What changed:**
- Each label CSV now includes:
  - `video_file`: filename of the source video
  - `video_timestamp`: parsed start time (e.g., "2026-05-25 16:40:12")
- Output filenames include the video timestamp (e.g., `labels_D3_170340.csv`)

---

## 3. Fuse Trajectories

The fusion script now:
1. **Finds all label CSVs** per camera (e.g., `labels_D3*.csv`)
2. **Parses timestamps** from video filenames or CSV metadata
3. **Computes absolute timestamps** for each frame
4. **Merges by time** (not frame number) with a 0.1s tolerance
5. **Averages overlapping observations** from multiple cameras at the same time

```bash
python scripts/fuse_trajectories.py --monkey H1705011 --date 2026-05-25
```

Output: `results/oft_analysis/H1705011/2026-05-25/trajectory_realworld.csv`

The output trajectory has:
- `absolute_time`: actual timestamp (datetime)
- `time_sec`: relative time from session start
- `x_cm`, `y_cm`: real-world position
- `camera`: which camera(s) contributed (e.g., "D3,D4" for overlap)

---

## Key Differences from Frame-Based Sync

| Old (Frame-Based) | New (Timestamp-Based) |
|-------------------|----------------------|
| Assumed all videos start at t=0 | Each video has its own start time |
| Merged by `video_frame` number | Merges by `absolute_time` |
| One CSV per camera | Multiple CSVs per camera (one per video) |
| Manual timestamp alignment needed | Automatic timestamp parsing from filename |

---

## Example: Multiple D3 Videos

You have 7 D3 videos spanning ~3 hours:

```
D03_20260525164012.mp4  → 16:40:12  (labeled ✓)
D03_20260525170340.mp4  → 17:03:40
D03_20260525172708.mp4  → 17:27:08
D03_20260525175036.mp4  → 17:50:36
D03_20260525181406.mp4  → 18:14:06
D03_20260525183734.mp4  → 18:37:34
D03_20260525190102.mp4  → 19:01:02
```

After labeling all 7, you'll have:
```
labels_D3_164012.csv
labels_D3_170340.csv
labels_D3_172708.csv
labels_D3_175036.csv
labels_D3_181406.csv
labels_D3_183734.csv
labels_D3_190102.csv
```

`fuse_trajectories.py` will automatically:
1. Load all 7 CSVs
2. Parse their start times
3. Create one continuous trajectory across all timepoints
4. Merge with other cameras (D1, D2, D4) at matching timestamps

---

## Troubleshooting

### "Could not parse timestamp from filename"
- Ensure video filenames follow format: `D0X_YYYYMMDDHHMMSS.mp4`
- Or include `--video` explicitly when labeling

### "No labels for camera D3"
- Check that label CSVs are named `labels_D3*.csv` in the results directory
- Pattern matching looks for `labels_D3.csv`, `labels_D3_164012.csv`, etc.

### Multiple cameras at overlapping times
- The script averages their positions automatically
- Camera list shows which contributed: "D3,D4"
- Time tolerance is 0.1s (configurable in `fuse_overlapping()`)
