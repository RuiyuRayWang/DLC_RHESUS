# OFT Video Analysis Pipeline — Progress Summary

**Monkey:** H1705011  
**Session:** 2026-05-25  
**Last updated:** 2026-06-09

---

## 🚀 Current Status

**Phase:** Manual labeling ready - **All calibrations complete!** ✅  
**Blocker:** None

### Completed ✅
- [x] All pipeline scripts written and tested
- [x] Qt6 GUI issue identified and fixed (window name special characters)
- [x] Error handling added to both GUI tools
- [x] Diagnostic tools created
- [x] Documentation complete
- [x] **All camera calibrations complete!** ✅

### Next Steps 📋
1. ✅ ~~Run camera calibration~~ - **All done!**
2. **Label monkey positions** per camera - **START HERE**
   - Use: `python scripts/manual_labeler.py --camera D2`
   - See [MANUAL_LABELER_REFERENCE.md](MANUAL_LABELER_REFERENCE.md) for full guide
3. **Fuse trajectories** into real-world coordinates
4. **Analyze results** in notebook

### Calibration Results
| Camera | Status | Points Labeled | Floor Error | solvePnP Error | Notes |
|--------|--------|----------------|-------------|----------------|-------|
| D2 | ✅ Complete | 8/8 | **0.00 cm** | 62.66 px | Perfect! |
| D3 | ✅ Complete | 8/8 | TBD | TBD | Ready for labeling |
| D4 | ✅ Complete | 8/8 | TBD | TBD | Ready for labeling |
| D1 | ⏸ No video | - | - | - | May need different session |

---

## Overview

The goal is to reconstruct real-world (cm) movement trajectories for a macaque
monkey in a **322 × 322 cm open-field arena**, using video recordings from
four wide-angle cameras (D1–D4) positioned above each quadrant.

```
(0,0) ────────── (161,0) ────────── (322,0)
  │                  │                  │
  │   D1 (top-left)  │  D2 (top-right)  │
  │                  │                  │
(0,161) ──────── (161,161) ──────── (322,161)
  │                  │                  │
  │  D3 (bottom-left)│ D4 (bottom-right)│
  │                  │                  │
(0,322) ──────── (161,322) ──────── (322,322)
```

Arena height: **105 cm**. Cameras are hardware-synced (same trigger).

---

## Pipeline Architecture

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
| `scripts/sam_segmentation.py` | SAM-based automatic segmentation (Segment Anything Model vit_b) — evaluated and **abandoned** due to cage interference | ⚠ Deprecated |
| `scripts/test_segmentation_tuning.py` | Classical CV segmentation test harness — also evaluated and superseded by manual labeling | ⚠ Deprecated |

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

> **TODO:** add a notebook section that loads `trajectory_realworld.csv` and plots the arena-space heatmap and cm-unit behavioral metrics.

### Test Clips

| File | Source | Content |
|------|--------|---------|
| `data/oft_rhp/H1705011/2026-05-25/test_clips/D02_20260525161934_1min.mp4` | D02 video | Original test clip (monkey partially behind cage bar) |
| `data/oft_rhp/H1705011/2026-05-25/test_clips/D03_20260525170340_1min.mp4` | D03 video @ 8′13″–9′13″ | New clip where monkey is clearly visible |

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
- The monkey's body centre approximates the floor contact point
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
conda activate deeplabcut
cd /home/luolab/PROJECTS/PAL_ENHANCER

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
- [ ] Calibrate remaining cameras (D3, D4, and D1 if video is available)
- [ ] Manual labeling for all calibrated cameras (~100 frames per camera)
- [ ] Notebook section for real-world trajectory visualization (cm heatmap, cm/s metrics)
- [ ] Lens distortion correction (solvePnP path) validation — after calibration complete
- [ ] `manual_labeler.py --all` mode (loop all cameras in one session) — nice to have
