# OFT Pipeline Status Snapshot

**Project:** PAL_ENHANCER OFT tracking  
**Environment:** `video_tracking`  
**Updated:** 2026-06-15

## Current state

The macOS OFT pipeline is operational end-to-end:

1. GUI calibration and manual labeling
2. Anchor-based multi-video time alignment
3. Real-world trajectory fusion
4. Window selection for quantification
5. Daily behavior titration plots in RStudio/Posit

## Stable workflow

Use this sequence for each session:

1. Label videos (`manual_labeler.py`)
2. Generate `label_time_anchors.csv` (`generate_time_anchor_template.py`)
3. Curate `anchor_timestamp` manually (VSPlayer)
4. Fuse trajectory (`fuse_trajectories.py`)
5. Select analysis window (`select_time_window.py`)
6. Plot daily titration metrics (`scripts/plot_pd_behavior_titration.R`)

Primary reference:
- `docs/MULTI_VIDEO_WORKFLOW.md`

## Key implementation notes

- Paths are repository-relative (`PROJECT_ROOT` style), no Linux hardcoded paths.
- Fusion uses anchor-based absolute time per file.
- Projection cleanup is active (camera-region filtering and arena clipping).
- Selector displays window metrics including distance, max velocity, movement bouts, mean bout duration, and active time.
- `selected_time_window.json` now exports active-period mean velocity, immobility time, and window-derived active/immobility durations.
- Daily titration plots are generated interactively in RStudio with explicit `...` gaps for missing dates.

## Core outputs per session

- `trajectory_realworld.csv`
- `movement_bouts.csv`
- `trajectory_summary.json`
- `selected_time_window.json`
- `pd_behavior_titration_metrics.csv` (optional QC table)

## Open items

- Keep anchor timestamps curated and auditable (`method`, `notes`).
- Re-run notebook visual QC when fusion parameters change.
- Optional: evaluate additional QC summaries for projection filtering.
- Optional: use the R titration script for a quick longitudinal summary across days.

```
Videos (D1–D4)
      │
      ▼
calibrate_cameras.py  ──►  results/calibration/calibration_D*.json
      │                          (floor homography H, optional solvePnP)
      ▼
manual_labeler.py     ──►  results/oft_analysis/.../labels_D*.csv
      │                          (pixel centroids + "no monkey" flags)
      ▼
fuse_trajectories.py  ──►  results/oft_analysis/.../trajectory_realworld.csv
                                 movement_bouts.csv
                                 trajectory_summary.json
      │
      ▼
Notebook: oft_video_analysis.ipynb
      (visualization, behavioral metrics in real units)
```

---

## What Has Been Built

### Scripts

| File | Purpose | Status |
|------|---------|--------|
| `scripts/calibrate_cameras.py` | Interactive OpenCV GUI to label known arena corners in a reference frame; computes floor homography (pixel → cm) and optionally full camera model via solvePnP | ✅ Done |
| `scripts/manual_labeler.py` | Frame-by-frame OpenCV labeling tool; camera-aware; supports "no monkey visible" marking; resumable sessions | ✅ Done |
| `scripts/fuse_trajectories.py` | Fuses per-camera CSVs into a single real-world trajectory; averages boundary-overlap frames; interpolates short gaps; computes velocity and movement bouts | ✅ Done |

### Notebook: `notebooks/oft_video_analysis.ipynb`

| Section | Content | Status |
|---------|---------|--------|
| 1. Video file organisation | Parse filenames, tabulate metadata | ✅ |
| 2. Video loading test | Sequential HEVC-safe frame reader | ✅ |
| 3. Frame extraction | `extract_frames_sequential()` helper | ✅ |
| 4. Background subtraction | Motion intensity profile via MOG2 | ✅ |
| 5. Segmentation (SAM) | SAM setup + test — kept for reference | ⚠ Not used |
| **5.1 Frame sampler** | 20-frame grid for visual inspection; select frames to test | ✅ |
| **5.2 Manual labeling** | Instructions to run the labeler; load/plot manual CSV | ✅ |
| 6. Tracking | `track_monkey()` using `segment_monkey_classical` wrapper | ✅ (SAM-based, not used) |
| 7. Behavioral metrics | Velocity, bouts, immobility from pixel-based trajectory | ✅ (pixel units) |
| 8. Batch processing | Loop all videos, save metrics CSV | ✅ |
| 9–10. Temporal analysis / Export | Trends over time, JSON summary | ✅ |

---

## Key Design Decisions

### Why SAM was abandoned
SAM (`vit_b`, automatic mask generator) consistently failed to isolate the
monkey from the cage structure:

- Generated 200–250 candidate masks per frame
- After merging, only 4–7 survived
- The cage itself was selected as the dominant mask (~75% frame area)
- The actual monkey was either filtered out (touches border, area too small)
  or never reliably segmented

Root causes: wide-angle perspective, cage bars fragmenting the monkey silhouette,
and the cage floor/wall occupying large contiguous regions that score higher than
the monkey.

### Why homography (not full 3D calibration) is the primary approach
A 3×3 floor homography is sufficient because:
- We only care about the monkey's position **on the floor** (z = 0)
- The monkey's head approximates the position of the monkey
- A homography directly maps pixel → cm without knowing camera intrinsics

The full `solvePnP` model (estimated when cage-top corners are also labeled)
is computed as a bonus and stored in the JSON, but is not required for tracking.

### Boundary overlap handling
Near the shared border between two quadrants, the monkey may be visible in two
cameras simultaneously. The fuser averages the two real-world cm positions,
which is geometrically correct when both cameras' homographies are accurate.

### "No monkey" frames
Frames where the monkey is not in the camera's field of view are explicitly
marked (M key in the labeler) and stored as `centroid_x = -1`. The fuser
drops these rows before constructing the trajectory. Short gaps (≤ 2 s by
default) are linearly interpolated.

---

## Calibration Details

### What to label per camera
| Point | Count | Known 3D position |
|-------|-------|-------------------|
| Floor corners of the quadrant | 4 (required) | (x, y, z=0) in cm |
| Cage top corners | 4 (optional) | same x, y; z=105 cm |

Minimum for a working homography: **4 floor corners**.  
Recommended for distortion estimation: 4 floor + 4 cage top (8 points total).

### Reprojection error target
| Quality | Reprojection error |
|---------|--------------------|
| Excellent | < 5 cm |
| Acceptable | 5–15 cm |
| Re-label needed | > 20 cm |

---

## How to Run the Full Pipeline

**⚠️ Important:** GUI scripts must be run with an **activated** conda environment, not through `conda run`.  
See [QT_SUBPROCESS_FIX.md](QT_SUBPROCESS_FIX.md) for details on the Qt6 subprocess limitation.

```bash
# ✅ Correct way: Activate environment first
conda activate dlc
cd /home/luolab/PROJECTS/DLC_RHESUS

# Step 1 — calibrate cameras (skip cameras without video; do once per session setup)
python scripts/calibrate_cameras.py --all --frame-sec 30
# For a missing camera, supply the video explicitly:
# python scripts/calibrate_cameras.py --camera D1 --video /path/to/D01_*.mp4

# Step 2 — label monkey position per camera (samples every 20th frame by default)
python scripts/manual_labeler.py --camera D3
python scripts/manual_labeler.py --camera D3 --sample-every 20  # explicit stride

# Step 3 — fuse into real-world trajectory
python scripts/fuse_trajectories.py

# Outputs:
#   results/calibration/calibration_D*.json
#   results/oft_analysis/H1705011/2026-05-25/labels_D*.csv
#   results/oft_analysis/H1705011/2026-05-25/trajectory_realworld.csv
#   results/oft_analysis/H1705011/2026-05-25/movement_bouts.csv
#   results/oft_analysis/H1705011/2026-05-25/trajectory_summary.json
```

### Labeler controls

| Key / Mouse | Action |
|-------------|--------|
| Left-click | Label monkey position and advance |
| Right-click | Label, stay on current frame |
| M | Mark "no monkey visible" and advance |
| N / → | Next frame (skip) |
| P / ← | Previous frame |
| U | Undo label on current frame |
| Q / ESC | Save and quit |

### Calibrator controls

| Key / Mouse | Action |
|-------------|--------|
| Left-click | Label current point and advance |
| Right-click | Label, stay |
| N / → | Next point |
| P / ← | Previous point |
| U | Undo current point |
| C | Compute and print calibration summary |
| S | Save JSON now |
| Q / ESC | Save and quit |

---

## Known Issues / TODOs

### Fixed Issues ✅
- [x] **Qt6 window creation bug** (2026-06-09): OpenCV GUI tools failed due to **special characters in window names** (em-dash `—` and pipe `|`). **Solution:** Simplified window names to alphanumeric + underscores only. Both `calibrate_cameras.py` and `manual_labeler.py` now work correctly. See [QT_SUBPROCESS_FIX.md](QT_SUBPROCESS_FIX.md) for full details.

### Open TODOs
- [x] Manual labeling for all calibrated cameras (~100 frames per camera)
- [x] `manual_labeler.py --mode visibility` V - visible, M - no monkey
- [ ] Train camera specific DLC model for automated monkey head annotation
- [ ] Lens distortion correction (solvePnP path) validation — after calibration complete

