# Manual Labeler - Command Reference

## Quick Start

Label monkey positions for camera D2 (uses default sampling: every 20th frame):
```bash
conda activate deeplabcut
cd /home/luolab/PROJECTS/PAL_ENHANCER
python scripts/manual_labeler.py --camera D2
```

## Command-Line Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--camera` | `-c` | choice | interactive | Camera to label: D1, D2, D3, or D4 |
| `--video` | `-v` | path | auto-detect | Video file path (auto-finds from data dir) |
| `--output` | `-o` | path | auto | Output CSV path (default: results/.../labels_D#.csv) |
| `--sample-every` | `-n` | int | 20 | Label every Nth frame (stride) |
| `--scale` | `-s` | float | 0.4 | Display scale for 4K video (0.4 = 40%) |
| `--fps` | | float | 20 | Video frame rate |

## Usage Examples

### Basic Usage (Default Settings)
```bash
# Label D2 with default stride (every 20th frame = 1 Hz at 20 fps)
python scripts/manual_labeler.py --camera D2

# Label D3
python scripts/manual_labeler.py --camera D3
```

### Custom Sampling Rate
```bash
# Sample every 10th frame (2 Hz at 20 fps) - more labels, slower
python scripts/manual_labeler.py --camera D2 --sample-every 10

# Sample every 40th frame (0.5 Hz at 20 fps) - fewer labels, faster
python scripts/manual_labeler.py --camera D2 --sample-every 40

# Sample every frame (20 Hz) - maximum temporal resolution
python scripts/manual_labeler.py --camera D2 --sample-every 1
```

### Explicit Video Path
```bash
# Specify video file explicitly
python scripts/manual_labeler.py --camera D2 \
  --video data/oft_rhp/H1705011/2026-05-25/D02_20260525161934.mp4
```

### Custom Output Location
```bash
# Save labels to specific location
python scripts/manual_labeler.py --camera D2 \
  --output results/my_custom_labels_D2.csv
```

### Adjust Display Size
```bash
# Larger display (50% of 4K = 1920x1080)
python scripts/manual_labeler.py --camera D2 --scale 0.5

# Smaller display (30% of 4K = 1152x648)
python scripts/manual_labeler.py --camera D2 --scale 0.3
```

## How Sampling Works

**Default: `--sample-every 20`**
- Video has 28,162 frames
- Samples every 20th frame: frame 0, 20, 40, 60, ...
- Total frames to label: ~1,408 frames
- At 20 fps, this is 1 sample/second (1 Hz)

**Higher frequency: `--sample-every 10`**
- Samples every 10th frame: frame 0, 10, 20, 30, ...
- Total frames to label: ~2,816 frames
- At 20 fps, this is 2 samples/second (2 Hz)
- **More work, but better temporal resolution**

**Lower frequency: `--sample-every 40`**
- Samples every 40th frame: frame 0, 40, 80, 120, ...
- Total frames to label: ~704 frames
- At 20 fps, this is 0.5 samples/second (0.5 Hz)
- **Less work, but coarser temporal resolution**

## Recommended Settings

### For Behavioral Analysis (Typical)
```bash
python scripts/manual_labeler.py --camera D2 --sample-every 20
```
- Good balance between annotation effort and temporal resolution
- 1 Hz sampling is sufficient for most movement analysis
- ~1,400 frames per camera per session

### For Detailed Movement Tracking
```bash
python scripts/manual_labeler.py --camera D2 --sample-every 10
```
- Better temporal resolution for fast movements
- 2 Hz sampling captures more detail
- ~2,800 frames per camera (more work)

### For Quick Overview
```bash
python scripts/manual_labeler.py --camera D2 --sample-every 40
```
- Faster labeling for initial testing
- 0.5 Hz sampling still captures general behavior
- ~700 frames per camera (less work)

## Output Format

The script saves a CSV file with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `frame_num` | int | Original video frame number (e.g., 0, 20, 40, ...) |
| `centroid_x` | float | X coordinate in pixels (or -1 if monkey not visible) |
| `centroid_y` | float | Y coordinate in pixels (or -1 if monkey not visible) |

Example output (`labels_D2.csv`):
```csv
frame_num,centroid_x,centroid_y
0,1920.5,1080.2
20,1925.3,1082.1
40,-1.0,-1.0
60,1930.8,1085.5
```

**Note:** `centroid_x = -1` means "monkey not visible in this frame" (marked with 'M' key).

## Keyboard/Mouse Controls

| Control | Action |
|---------|--------|
| **Left-click** | Label monkey position and advance to next frame |
| **Right-click** | Label monkey position, stay on current frame |
| **M** | Mark "no monkey visible" and advance |
| **N** or **→** | Skip to next frame (don't label) |
| **P** or **←** | Go back to previous frame |
| **U** | Undo/clear label on current frame |
| **Q** or **ESC** | Save all labels and quit |

## Tips

1. **Click on the monkey's body center** - roughly where it contacts the floor
2. **Be consistent** - try to click the same anatomical point across frames
3. **Use M key liberally** - mark frames where monkey is behind bars or not visible
4. **You can quit and resume** - labels are saved automatically, script resumes where you left off
5. **Review a few frames first** - get a feel for the monkey's position before starting

## Troubleshooting

**Window doesn't open:**
- Ensure conda environment is activated: `conda activate deeplabcut`
- Check display: `echo $DISPLAY` (should show :0 or :1)
- Run diagnostic: `python scripts/check_opencv_display.py`

**Wrong video loaded:**
- Use `--video` to specify the exact video file
- Check that video path matches your data structure

**Too many/few frames:**
- Adjust `--sample-every` to change sampling rate
- Default (20) is usually good for behavioral analysis

**Display too large/small:**
- Adjust `--scale` (default 0.4 = 40% of 4K resolution)
- Try 0.5 (larger) or 0.3 (smaller)

---

**Related Documentation:**
- [OFT_PIPELINE_PROGRESS.md](OFT_PIPELINE_PROGRESS.md) - Full pipeline overview
- [QUICKSTART_GUI.md](QUICKSTART_GUI.md) - Step-by-step calibration and labeling guide
- [QT_SUBPROCESS_FIX.md](QT_SUBPROCESS_FIX.md) - GUI troubleshooting
