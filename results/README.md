# Results Directory

**Purpose:** Store analysis outputs, calibration files, and processed data.

**Note:** Large intermediate results are excluded from git. Only essential calibration files and small outputs are tracked.

---

## Directory Structure

```
results/
├── calibration/               # Camera calibration files (KEPT in git)
│   ├── calibration_D1.json   # Camera D1 calibration (homography, intrinsics)
│   ├── calibration_D2.json   # Camera D2 calibration
│   ├── calibration_D3.json   # Camera D3 calibration
│   ├── calibration_D4.json   # Camera D4 calibration
│   ├── calibration_frame_D*.png  # Reference frames (excluded: 12-13MB each)
│   └── README.md
├── oft_analysis/              # OFT trajectory outputs (excluded)
│   └── [monkey_id]/
│       └── [date]/
│           ├── labels_D*.csv          # Manual labels per camera
│           ├── trajectory_realworld.csv  # Fused trajectory (cm coordinates)
│           ├── movement_bouts.csv     # Movement episodes
│           └── trajectory_summary.json
└── [other analysis results]   # Project-specific results (excluded if large)
```

---

## What's Tracked in Git

✅ **Included:**
- Calibration JSON files (`calibration_D*.json`)
- Small summary files (<1MB)
- Analysis scripts outputs that are essential

❌ **Excluded:**
- Large CSV files (>1MB)
- Calibration frame PNGs (12-13MB each, can be regenerated)
- Intermediate analysis outputs in `oft_analysis/`
- Date-stamped result folders (`2025-*/`)

---

## Calibration Files

### Format
Calibration JSON contains:
```json
{
  "camera": "D2",
  "frame_number": 600,
  "video_path": "...",
  "floor_homography": [[...], [...], [...]],  // 3x3 matrix
  "floor_points": [...],                      // Labeled floor corners
  "cage_points": [...],                       // Optional cage top corners
  "solvepnp": {                              // Optional full camera model
    "camera_matrix": [...],
    "dist_coeffs": [...],
    "rvec": [...],
    "tvec": [...]
  },
  "reprojection_error_cm": 0.00
}
```

### How to Regenerate Calibration Frames
If you need the reference PNG images:
```bash
python scripts/calibrate_cameras.py --camera D2 --frame-sec 30
# PNG will be saved during calibration
```

---

## OFT Analysis Outputs

After running the full pipeline, you'll have:

1. **Per-camera labels** (`labels_D*.csv`):
   ```csv
   frame_num,centroid_x,centroid_y
   0,1920.5,1080.2
   20,1925.3,1082.1
   ```

2. **Fused trajectory** (`trajectory_realworld.csv`):
   ```csv
   frame_num,time_sec,x_cm,y_cm,camera,velocity_cm_s
   0,0.0,180.5,145.2,D2,0.0
   1,0.05,180.8,145.5,D2,8.5
   ```

3. **Movement bouts** (`movement_bouts.csv`):
   ```csv
   bout_id,start_frame,end_frame,duration_sec,distance_cm,mean_velocity
   1,0,120,6.0,45.2,7.5
   ```

4. **Summary** (`trajectory_summary.json`):
   ```json
   {
     "total_frames": 28162,
     "duration_sec": 1408.1,
     "total_distance_cm": 15234.5,
     "mean_velocity_cm_s": 10.8,
     "num_movement_bouts": 45
   }
   ```

---

## Regenerating Results

All results can be regenerated from raw data:

```bash
# Step 1: Calibrate (if calibration files missing)
python scripts/calibrate_cameras.py --all

# Step 2: Label frames (if labels missing)
python scripts/manual_labeler.py --camera D2
python scripts/manual_labeler.py --camera D3
python scripts/manual_labeler.py --camera D4

# Step 3: Fuse trajectories
python scripts/fuse_trajectories.py --monkey H1705011 --date 2026-05-25
```

---

## Storage Guidelines

- **Calibration files:** ~1-10 KB each (JSON) - KEEP
- **Calibration frames:** 12-13 MB each (PNG) - regenerate if needed
- **Label CSVs:** ~50-500 KB per camera - can regenerate but slow (manual work)
- **Trajectory CSVs:** 1-50 MB depending on video length - can regenerate quickly
- **Large analysis CSVs:** >50 MB - definitely exclude, regenerate as needed

---

**Last updated:** 2026-06-09
