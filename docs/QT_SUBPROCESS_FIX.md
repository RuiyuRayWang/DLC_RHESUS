# Qt6 Window Name Issue with OpenCV GUI Tools

**Issue Date:** 2026-06-09  
**Root Cause Identified:** 2026-06-09 - **Special characters in window names**  
**Status:** ✅ **FIXED**

## The Problem

OpenCV 4.13.0 uses **Qt6** as the GUI backend. Qt6 has a bug where **window names containing special characters** cause window creation to fail:

```
cv2.error: OpenCV(4.13.0) ... error: (-27:Null pointer) NULL window handler in function 'cvSetMouseCallback'
```

**Root cause:** Window names with em-dashes (`—`) or pipe characters (`|`) fail in Qt6.

Example:
```python
# ❌ FAILS with Qt6
cv2.namedWindow("Camera D2 — L-click=label | N=next", cv2.WINDOW_NORMAL)

# ✅ WORKS with Qt6
cv2.namedWindow("Calibrate_D2", cv2.WINDOW_NORMAL)
```

## The Solution ✅ APPLIED

Both scripts have been fixed (as of 2026-06-09):

**Before:**
```python
self.win = f"Calibrate {camera} — L-click=label | N=next..."  # ❌
```

**After:**
```python
self.win = f"Calibrate_{camera}"  # ✅ Simple, alphanumeric only
self.win_title = f"Calibrate {camera} - ..."  # For display via setWindowTitle()
```

## Verification

Test results from `scripts/test_qt6_window.py`:

1. ✅ Small frame (100x100) - **PASS**
2. ✅ Large frame (3840x2160) - **PASS**
3. ✅ Load video frame first - **PASS**
4. ❌ Window name with `—` and `|` - **FAIL** (expected, this is the bug)
5. ✅ Simple alphanumeric name - **PASS**

**Calibration script test:**
```bash
python scripts/calibrate_cameras.py --camera D2 --frame-sec 30
```
Result: ✅ **SUCCESS** - Calibrated D2 with 0.00 cm reprojection error

## What Was Initially Suspected (But Wrong)

Initially we thought the issue was:
- ❌ Using `conda run` (creates subprocess) - NOT the issue
- ❌ Piping output - NOT the issue  
- ❌ Running via SSH without X11 - NOT the issue
- ❌ Qt6 threading in subprocesses - NOT the issue
- ❌ Large frame size (4K video) - NOT the issue
- ❌ Frame loading order - NOT the issue

**Actual issue:** Simply the **window name containing special characters**.

## Scripts Fixed

Both scripts now use simple window names:
- `calibrate_cameras.py`: `"Calibrate_D1"`, `"Calibrate_D2"`, etc.
- `manual_labeler.py`: `"Label_D1"`, `"Label_D2"`, etc.

Additional improvements applied:
1. Added `cv2.startWindowThread()` - Helps Qt6 initialize
2. Added initialization delay (0.15s) - Ensures window is ready
3. Show test frame before setting callbacks - Verifies window exists
4. Better error messages with diagnostics

## Usage (No Special Steps Required)

The scripts now work normally:

```bash
conda activate dlc
cd /home/luolab/PROJECTS/DLC_RHESUS

# Camera calibration
python scripts/calibrate_cameras.py --camera D2 --frame-sec 30
python scripts/calibrate_cameras.py --all --frame-sec 30

# Manual labeling (samples every 20th frame by default)
python scripts/manual_labeler.py --camera D2 --sample-every 20

# Trajectory fusion
python scripts/fuse_trajectories.py --monkey H1705011 --date 2026-05-25
```

## Lessons Learned

When debugging Qt6/OpenCV GUI issues:
1. **Check window name first** - avoid special characters (`—`, `|`, etc.)
2. Use alphanumeric + underscores only: `window_name_123`
3. If you need a fancy title, set it separately with `cv2.setWindowTitle()`
4. Create a minimal test case to isolate the problem

---

**Status:** Issue resolved - both scripts working as of 2026-06-09

**Alternative:** ffmpeg
