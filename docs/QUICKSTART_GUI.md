# Quick Start: Camera Calibration and Manual Labeling

**⚠️ IMPORTANT:** These GUI tools must be run **directly in your terminal**, not through scripts or automation tools.

## Step 0: Create the environment (first time only)

```bash
cd /Users/ray/PROJECTS/PAL_ENHANCER
bash scripts/setup_video_tracking_env.sh
conda activate video_tracking
```

## Step 1: Camera Calibration

Open a terminal and run:

```bash
cd /Users/ray/PROJECTS/PAL_ENHANCER
conda activate video_tracking
python scripts/calibrate_cameras.py --camera D2 --frame-sec 30
```

Replace `D2` with the camera you want to calibrate (D1, D2, D3, or D4).

To calibrate all cameras at once:
```bash
python scripts/calibrate_cameras.py --all --frame-sec 30
```

### Calibration Controls

- **Left-click**: Label current point and advance to next
- **Right-click**: Label current point, stay on same point
- **N** or **→**: Skip to next point
- **P** or **←**: Go back to previous point
- **U**: Undo/clear current point
- **C**: Compute calibration and show reprojection error
- **S**: Save calibration JSON now
- **Q** or **ESC**: Save and quit

### What to Label

Label the floor corners of the camera's quadrant (4 required points):
- Top-left corner
- Top-right corner  
- Bottom-left corner
- Bottom-right corner

Optionally label 4 more points on the cage top (same x,y positions, z=105cm) for distortion estimation.

---

## Step 2: Manual Labeling

After calibration, label monkey positions:

```bash
cd /Users/ray/PROJECTS/PAL_ENHANCER
conda activate video_tracking
python scripts/manual_labeler.py --camera D2
```

This uses the default sampling rate of every 20th frame, which is about 1 frame/second at 20 fps.

To change the sampling rate:
```bash
# Sample every 10th frame (2 Hz at 20 fps)
python scripts/manual_labeler.py --camera D2 --sample-every 10

# Sample every 40th frame (0.5 Hz at 20 fps)
python scripts/manual_labeler.py --camera D2 --sample-every 40
```

### Labeling Controls

- **Left-click**: Label monkey center position and advance
- **Right-click**: Label monkey center, stay on current frame
- **M**: Mark "no monkey visible" and advance
- **N** or **→**: Skip to next frame
- **P** or **←**: Go back to previous frame
- **U**: Undo label on current frame
- **Q** or **ESC**: Save and quit

### Where to Click

Click on the **center of the monkey's body** (roughly where the monkey contacts the floor). Be as consistent as possible across frames.

---

## Step 3: Trajectory Fusion

After labeling all cameras, fuse the trajectories:

```bash
python scripts/fuse_trajectories.py --monkey H1705011 --date 2026-05-25
```

This will:
1. Load calibration files from `results/calibration/`
2. Load label CSVs from `results/oft_analysis/H1705011/2026-05-25/`
3. Transform pixel coordinates to real-world cm
4. Fuse overlapping regions from multiple cameras
5. Interpolate short gaps
6. Compute velocity and movement bouts
7. Save outputs:
   - `trajectory_realworld.csv` - Frame-by-frame position (x, y in cm)
   - `movement_bouts.csv` - Detected movement episodes
   - `trajectory_summary.json` - Summary statistics

---

## Troubleshooting

### "NULL window handler" Error

This means the script was run in a subprocess (e.g., via `conda run`, piped, or in a bash script).

**Solution:** Type the commands **directly in your terminal**, don't:
- Use `conda run -n video_tracking python ...`
- Pipe output: `python script.py | tee log.txt`
- Run in a shell script with `bash -c "..."`

### No Display Available

If you're on a remote server via SSH:
```bash
ssh -X username@server  # Enable X11 forwarding
```

### Check OpenCV GUI

Run the diagnostic:
```bash
conda activate video_tracking
python scripts/check_opencv_display.py
```

This will test if OpenCV can create windows properly.

---

## File Locations

### Inputs
- Videos: `data/oft_rhp/H1705011/2026-05-25/D0*_*.mp4`

### Outputs
- Calibration: `results/calibration/calibration_D*.json`
- Labels: `results/oft_analysis/H1705011/2026-05-25/labels_D*.csv`
- Trajectories: `results/oft_analysis/H1705011/2026-05-25/trajectory_realworld.csv`
- Movement bouts: `results/oft_analysis/H1705011/2026-05-25/movement_bouts.csv`
- Summary: `results/oft_analysis/H1705011/2026-05-25/trajectory_summary.json`

---

## Full Documentation

- Pipeline overview: [docs/OFT_PIPELINE_PROGRESS.md](OFT_PIPELINE_PROGRESS.md)
- Qt6 issue details: [docs/QT_SUBPROCESS_FIX.md](QT_SUBPROCESS_FIX.md)
- Video format details: [docs/VIDEO_LOADING_SUMMARY.md](VIDEO_LOADING_SUMMARY.md)
