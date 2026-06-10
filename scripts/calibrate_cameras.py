#!/usr/bin/env python3
"""
Camera calibration tool for the 4-camera monkey OFT setup.

Labels known 3D world points (arena floor corners + cage top corners) in a
single representative frame per camera, then computes:

  1. Floor homography  H  (3×3) — maps pixel coords to real-world cm on the
     floor plane (z = 0).  Used for all downstream tracking.

  2. Full camera model via solvePnP (K, distortion, R, t) — estimated when
     ≥ 6 labeled points spanning two heights are available.  This absorbs
     lens distortion and perspective in one transform.

Arena layout (bird's-eye, origin = top-left outer corner):

    (0,0) ──────── (161,0) ──────── (322,0)
      │               │               │
      │   D1 (TL)    │   D2 (TR)    │
      │               │               │
    (0,161) ────── (161,161) ────── (322,161)
      │               │               │
      │   D3 (BL)    │   D4 (BR)    │
      │               │               │
    (0,322) ────── (161,322) ────── (322,322)

Cage height: 105 cm.  Cage top corners share the same (x, y) as the floor
corners but have z = 105.

Usage:
    conda activate video_tracking
    python scripts/calibrate_cameras.py               # interactive camera pick
    python scripts/calibrate_cameras.py --camera D1
    python scripts/calibrate_cameras.py --all         # all 4 cameras
    python scripts/calibrate_cameras.py --camera D3 --video /path/to/D03_*.mp4

Controls (in window):
    Left-click   label current point, advance to next
    Right-click  label current point, stay
    N / →        next point (skip)
    P / ←        previous point
    U            undo / clear label on current point
    C            compute & preview calibration (needs ≥ 4 floor pts)
    S            save calibration JSON now
    Q / ESC      compute, save, and quit

Output:
    results/calibration/calibration_D1.json  (one file per camera)
    results/calibration/calibration_frame_D1.png   (reference frame)
"""

import cv2
import numpy as np
import json
from pathlib import Path
import argparse
import sys
from datetime import datetime
from typing import Optional, List, Tuple, Dict


# ── Constants ──────────────────────────────────────────────────────────────────

ARENA_W     = 322   # cm
ARENA_H     = 322   # cm
MID_X       = 161   # cm  (ARENA_W / 2)
MID_Y       = 161   # cm
CAGE_Z      = 105   # cm  (cage wall height)

FPS         = 20.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_BASE  = PROJECT_ROOT / 'data' / 'oft_rhp' / 'H1705011' / '2026-05-25'
OUTPUT_DIR  = PROJECT_ROOT / 'results' / 'calibration'

DISPLAY_SCALE = 0.35    # 3840×2160 * 0.35 ≈ 1344×756
SIDEBAR_W     = 430     # px — right-side info panel

# Camera → quadrant definition (all coords in cm)
CAMERA_QUADRANTS: Dict[str, dict] = {
    'D1': {'x': (0,     MID_X),   'y': (0,     MID_Y),   'label': 'top-left'},
    'D2': {'x': (MID_X, ARENA_W), 'y': (0,     MID_Y),   'label': 'top-right'},
    'D3': {'x': (0,     MID_X),   'y': (MID_Y, ARENA_H), 'label': 'bottom-left'},
    'D4': {'x': (MID_X, ARENA_W), 'y': (MID_Y, ARENA_H), 'label': 'bottom-right'},
}

# Colour palette
BG        = (30,  30,  30)
C_WHITE   = (230, 230, 230)
C_YELLOW  = (0,   220, 220)
C_GREEN   = (80,  200, 80)
C_BLUE    = (220, 150, 80)
C_GRAY    = (150, 150, 150)
C_RED     = (80,  80,  220)
C_CURRENT = (0,   200, 255)


# ── Point list builder ─────────────────────────────────────────────────────────

def _corner_desc(xx: int, yy: int, zz: int) -> str:
    x_tag = 'outer' if xx in (0, ARENA_W) else 'center-line'
    y_tag = 'outer' if yy in (0, ARENA_H) else 'center-line'
    layer = 'floor' if zz == 0 else f'cage top (z={zz})'
    return f"{layer}  {x_tag}×{y_tag}"


def build_point_list(camera: str) -> List[dict]:
    """Return ordered list of labeled points for one camera."""
    q   = CAMERA_QUADRANTS[camera]
    x0, x1 = q['x']
    y0, y1 = q['y']

    corners = [('TL', x0, y0), ('TR', x1, y0), ('BR', x1, y1), ('BL', x0, y1)]
    pts: List[dict] = []

    # Floor corners (required)
    for tag, xx, yy in corners:
        pts.append({
            'name':     f'Floor {tag} ({xx},{yy})',
            'desc':     _corner_desc(xx, yy, 0),
            'world':    [xx, yy, 0],
            'pixel':    None,
            'required': True,
        })

    # Cage-top corners (optional, improve distortion estimate)
    for tag, xx, yy in corners:
        pts.append({
            'name':     f'Top   {tag} ({xx},{yy})',
            'desc':     _corner_desc(xx, yy, CAGE_Z),
            'world':    [xx, yy, CAGE_Z],
            'pixel':    None,
            'required': False,
        })

    return pts


# ── Calibration computation ────────────────────────────────────────────────────

def compute_floor_homography(points: List[dict]) -> Tuple[Optional[np.ndarray], float]:
    """Compute pixel→cm homography using floor points (z=0) only."""
    fp = [p for p in points if p['pixel'] is not None and p['world'][2] == 0]
    if len(fp) < 4:
        return None, float('inf')

    src = np.array([[p['pixel'][0], p['pixel'][1]] for p in fp], dtype=np.float64)
    dst = np.array([[p['world'][0], p['world'][1]] for p in fp], dtype=np.float64)

    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return None, float('inf')

    # Reprojection error in cm
    src_h = np.hstack([src, np.ones((len(src), 1))])
    proj  = (H @ src_h.T).T
    proj  = proj[:, :2] / proj[:, 2:3]
    err   = float(np.mean(np.linalg.norm(proj - dst, axis=1)))
    return H, err


def compute_solvepnp(points: List[dict],
                     image_size: Tuple[int, int]) -> Optional[dict]:
    """
    Estimate full camera model via solvePnP.
    Requires ≥ 6 labeled points spanning two heights.
    Returns dict with K, dist_coeffs, rvec, tvec, reprojection_error; or None.
    """
    labeled = [p for p in points if p['pixel'] is not None]
    if len(labeled) < 6:
        return None
    heights = {p['world'][2] for p in labeled}
    if len(heights) < 2:
        return None

    obj = np.array([p['world']    for p in labeled], dtype=np.float64)
    img = np.array([p['pixel']    for p in labeled], dtype=np.float64)

    w, h = image_size
    # Initial K: focal ≈ image width (reasonable prior for a wide-angle lens)
    K = np.array([[float(w), 0, w / 2.0],
                  [0, float(w), h / 2.0],
                  [0, 0, 1.0]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)

    try:
        ok, rvec, tvec = cv2.solvePnP(
            obj, img.reshape(-1, 1, 2), K, dist,
            flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)))
        return {
            'camera_matrix':          K.tolist(),
            'dist_coeffs':            dist.tolist(),
            'rvec':                   rvec.flatten().tolist(),
            'tvec':                   tvec.flatten().tolist(),
            'reprojection_error_px':  err,
            'n_points':               len(labeled),
        }
    except cv2.error:
        return None


def pixel_to_cm(pixel_xy: Tuple[float, float],
                H: np.ndarray) -> Tuple[float, float]:
    """Apply floor homography: pixel → (x_cm, y_cm)."""
    pt = np.array([[[float(pixel_xy[0]), float(pixel_xy[1])]]], dtype=np.float64)
    r  = cv2.perspectiveTransform(pt, H)
    return float(r[0, 0, 0]), float(r[0, 0, 1])


# ── Frame extraction ───────────────────────────────────────────────────────────

def extract_frame(video_path: Path,
                  target_sec: float = 30.0,
                  fps: float = FPS) -> Tuple[np.ndarray, int]:
    """Extract frame near target_sec via sequential read (HEVC-safe)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"ERROR: Cannot open video: {video_path}")

    total        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = min(int(target_sec * fps), max(total - 1, 0))

    print(f"  Reading frame #{target_frame}  (t={target_sec:.0f}s)  "
          f"from {video_path.name}  [{total} frames total] …", end='', flush=True)

    frame = None
    for i in range(target_frame + 1):
        ret, f = cap.read()
        if not ret:
            break
        frame = f
    cap.release()

    if frame is None:
        sys.exit(f"\nERROR: Could not read frame from {video_path}")

    print(" done.")
    return frame, target_frame


def find_video(camera: str, base_dir: Path) -> Optional[Path]:
    """Auto-detect video file for a given camera (D1 → D01_*.mp4).
    Returns None (with a warning) rather than exiting if no file is found.
    """
    prefix  = f"D0{camera[-1]}_"
    matches = sorted(base_dir.glob(f"{prefix}*.mp4"))
    if not matches:
        print(f"  ⚠  No video found for camera {camera} in {base_dir} — skipping.")
        return None
    if len(matches) > 1:
        print(f"  ⚠  Multiple videos found for {camera}; using {matches[0].name}")
    return matches[0]


# ── GUI rendering ──────────────────────────────────────────────────────────────

def _put(img: np.ndarray, text: str, x: int, y: int,
         color=C_WHITE, scale: float = 0.55, thick: int = 1) -> None:
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _make_sidebar(points: List[dict], current_idx: int,
                  camera: str, H: Optional[np.ndarray], reproj_err: float,
                  panel_h: int) -> np.ndarray:
    p = np.full((panel_h, SIDEBAR_W, 3), BG, dtype=np.uint8)

    # ── Title bar ──
    cv2.rectangle(p, (0, 0), (SIDEBAR_W, 46), (50, 50, 50), -1)
    q = CAMERA_QUADRANTS[camera]
    _put(p, f"Camera {camera} — {q['label']}", 10, 30, C_YELLOW, 0.72, 2)

    # ── Stats ──
    n_all   = sum(1 for pt in points if pt['pixel'] is not None)
    n_floor = sum(1 for pt in points if pt['pixel'] is not None and pt['world'][2] == 0)
    _put(p, f"Labeled: {n_all}/{len(points)}   floor: {n_floor}/4", 10, 64)
    if H is not None:
        col = C_GREEN if reproj_err < 5 else (C_RED if reproj_err > 20 else C_YELLOW)
        _put(p, f"Reprojection error: {reproj_err:.1f} cm", 10, 84, col)
    else:
        _put(p, "Need ≥ 4 floor pts for homography", 10, 84, C_GRAY, 0.48)
    cv2.line(p, (0, 100), (SIDEBAR_W, 100), (70, 70, 70), 1)

    # ── Point list ──
    y, row_h = 112, 44
    for i, pt in enumerate(points):
        is_cur    = (i == current_idx)
        is_lab    = pt['pixel'] is not None
        is_req    = pt['required']

        # Row background
        bg = (55, 55, 85) if is_cur else (40, 40, 40)
        cv2.rectangle(p, (5, y - 26), (SIDEBAR_W - 5, y + row_h - 30), bg, -1)
        if is_cur:
            cv2.rectangle(p, (5, y - 26), (SIDEBAR_W - 5, y + row_h - 30), C_CURRENT, 1)

        icon  = '✓' if is_lab else ('►' if is_cur else ('○' if is_req else '◇'))
        ic    = C_GREEN  if is_lab else (C_CURRENT if is_cur else (C_GRAY if is_req else C_BLUE))
        nc    = C_CURRENT if is_cur else (C_GREEN if is_lab else (C_WHITE if is_req else C_BLUE))

        _put(p, icon, 12, y, ic,  0.62, 2 if is_cur else 1)
        _put(p, pt['name'], 36, y, nc, 0.48, 2 if is_cur else 1)

        if is_lab:
            px, py = pt['pixel']
            _put(p, f"  pixel ({px}, {py})", 36, y + 16, C_GREEN, 0.40)
        else:
            _put(p, f"  {pt['desc']}", 36, y + 16,
                 C_CURRENT if is_cur else C_GRAY, 0.40)

        y += row_h
        if y > panel_h - 160:
            _put(p, f"  … {len(points) - i - 1} more points below", 12, y, C_GRAY, 0.42)
            break

    # ── Controls footer ──
    fy = panel_h - 148
    cv2.line(p, (0, fy), (SIDEBAR_W, fy), (70, 70, 70), 1)
    controls = [
        ("L-click",  "label & next"),
        ("R-click",  "label, stay"),
        ("N / →",    "next point"),
        ("P / ←",    "previous point"),
        ("U",        "undo label"),
        ("C",        "compute calibration"),
        ("S",        "save JSON now"),
        ("Q / ESC",  "save & quit"),
    ]
    yc = fy + 18
    for key, desc in controls:
        _put(p, f"{key:<9}  {desc}", 10, yc, C_GRAY, 0.44)
        yc += 17

    return p


def render(frame_orig: np.ndarray, points: List[dict], current_idx: int,
           camera: str, H: Optional[np.ndarray], reproj_err: float) -> np.ndarray:
    h_o, w_o = frame_orig.shape[:2]
    dw = int(w_o * DISPLAY_SCALE)
    dh = int(h_o * DISPLAY_SCALE)
    disp = cv2.resize(frame_orig, (dw, dh), interpolation=cv2.INTER_AREA)

    # Draw labeled points
    for i, pt in enumerate(points):
        if pt['pixel'] is None:
            continue
        dx = int(pt['pixel'][0] * DISPLAY_SCALE)
        dy = int(pt['pixel'][1] * DISPLAY_SCALE)
        is_floor = (pt['world'][2] == 0)
        col      = C_CURRENT if i == current_idx else (C_GREEN if is_floor else C_BLUE)
        marker   = cv2.MARKER_CROSS if is_floor else cv2.MARKER_DIAMOND
        cv2.drawMarker(disp, (dx, dy), col, marker, markerSize=32, thickness=2)
        cv2.putText(disp, pt['name'].split('(')[0].strip(),
                    (dx + 8, dy - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44, col, 1, cv2.LINE_AA)

        # Show per-point reprojection error in cm
        if H is not None and is_floor:
            cx, cy = pixel_to_cm(pt['pixel'], H)
            wx, wy = pt['world'][0], pt['world'][1]
            ecm    = float(np.hypot(cx - wx, cy - wy))
            ec     = C_GREEN if ecm < 5 else C_RED
            cv2.putText(disp, f"{ecm:.1f}cm",
                        (dx + 8, dy + 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, ec, 1, cv2.LINE_AA)

    # Border indicating current camera
    cv2.rectangle(disp, (0, 0), (dw - 1, dh - 1), C_CURRENT, 3)

    # Header bar
    cv2.rectangle(disp, (0, 0), (dw, 44), (20, 20, 20), -1)
    cur = points[current_idx]
    wx, wy, wz = cur['world']
    header = (f"Cam {camera}  |  Point {current_idx + 1}/{len(points)}: "
              f"{cur['name']}  — world ({wx}, {wy}, {wz}) cm")
    cv2.putText(disp, header, (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_YELLOW, 1, cv2.LINE_AA)

    sidebar = _make_sidebar(points, current_idx, camera, H, reproj_err, dh)
    return np.hstack([disp, sidebar])


# ── CameraCalibrator ───────────────────────────────────────────────────────────

class CameraCalibrator:
    """Interactive per-camera calibration session."""

    def __init__(self, camera: str, video_path: Path, output_dir: Path,
                 frame_sec: float = 30.0):
        self.camera     = camera
        self.video_path = video_path
        self.output_dir = output_dir
        self.out_json   = output_dir / f'calibration_{camera}.json'
        self.out_png    = output_dir / f'calibration_frame_{camera}.png'

        print(f"\n{'='*60}")
        print(f" Camera {camera}  ({CAMERA_QUADRANTS[camera]['label']} quadrant)")
        print(f"{'='*60}")
        self.frame, self.frame_num = extract_frame(video_path, frame_sec)
        self.h, self.w = self.frame.shape[:2]
        print(f"  Frame size: {self.w}×{self.h}")

        self.points: List[dict] = build_point_list(camera)

        # Resume any previous session
        if self.out_json.exists():
            self._load_labels()

        self.current_idx          = 0
        self._advance_requested   = False
        self.H: Optional[np.ndarray] = None
        self.reproj_err: float    = float('inf')
        self._try_compute()

        # Simple window name (Qt6 can be sensitive to special characters)
        self.win = f"Calibrate_{camera}"
        self.win_title = (f"Calibrate {camera} - "
                         "L-click=label  N=next  P=prev  U=undo  Q=save&quit")

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _load_labels(self) -> None:
        with open(self.out_json) as f:
            data = json.load(f)
        lookup = {p['name']: p['pixel']
                  for p in data.get('labeled_points', [])
                  if p.get('pixel') is not None}
        loaded = 0
        for p in self.points:
            if p['name'] in lookup:
                p['pixel'] = tuple(lookup[p['name']])
                loaded += 1
        print(f"  Resumed: {loaded} existing labels loaded from {self.out_json.name}")

    def _try_compute(self) -> None:
        H, err = compute_floor_homography(self.points)
        if H is not None:
            self.H          = H
            self.reproj_err = err

    def _print_summary(self) -> None:
        n   = sum(1 for p in self.points if p['pixel'] is not None)
        nf  = sum(1 for p in self.points if p['pixel'] is not None and p['world'][2] == 0)
        print(f"\n── Calibration summary — camera {self.camera} ──")
        print(f"  Labeled: {n}/{len(self.points)}  (floor: {nf})")
        if self.H is not None:
            print(f"  Floor homography  reprojection error: {self.reproj_err:.2f} cm")
            for p in self.points:
                if p['pixel'] is not None and p['world'][2] == 0:
                    cx, cy = pixel_to_cm(p['pixel'], self.H)
                    wx, wy = p['world'][0], p['world'][1]
                    print(f"    {p['name']:<28}  error = "
                          f"{np.hypot(cx-wx, cy-wy):.2f} cm")
        else:
            print("  Floor homography: NOT COMPUTED (need ≥ 4 floor pts)")
        pnp = compute_solvepnp(self.points, (self.w, self.h))
        if pnp:
            print(f"  solvePnP pixel reprojection error: {pnp['reprojection_error_px']:.2f} px"
                  f"  (from {pnp['n_points']} pts)")

    # ── Mouse callback ──────────────────────────────────────────────────────────

    def _mouse_cb(self, event, mx, my, flags, param) -> None:
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        # Ignore clicks in the sidebar
        if mx >= int(self.w * DISPLAY_SCALE):
            return
        ox = int(mx / DISPLAY_SCALE)
        oy = int(my / DISPLAY_SCALE)
        self.points[self.current_idx]['pixel'] = (ox, oy)
        self._try_compute()
        if event == cv2.EVENT_LBUTTONDOWN:
            self._advance_requested = True

    # ── Main loop ───────────────────────────────────────────────────────────────

    def run(self) -> None:
        # Initialize Qt6 window system early
        print(f"\n  Opening window: {self.win_title}")
        
        try:
            # Try to start the window thread (helps with Qt6)
            try:
                cv2.startWindowThread()
            except:
                pass  # Not all OpenCV builds support this
            
            # Create window with simple name (Qt6 sensitive to special characters)
            cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
            
            # Set window title if possible (not all backends support this)
            try:
                cv2.setWindowTitle(self.win, self.win_title)
            except:
                pass
            
            # Resize window
            cv2.resizeWindow(self.win,
                             int(self.w * DISPLAY_SCALE) + SIDEBAR_W,
                             int(self.h * DISPLAY_SCALE))
            
            # Let Qt6 initialize the window (critical for Qt6)
            import time
            time.sleep(0.15)
            
            # Show a test frame to ensure window is ready
            test_frame = render(self.frame, self.points, self.current_idx,
                              self.camera, self.H, self.reproj_err)
            cv2.imshow(self.win, test_frame)
            cv2.waitKey(1)  # Process window events
            
            # NOW set the mouse callback (after window is fully initialized)
            cv2.setMouseCallback(self.win, self._mouse_cb)
            
        except cv2.error as e:
            print("\n❌ ERROR: Failed to create OpenCV window with Qt6.")
            print(f"   Details: {e}")
            print("\n   This is a known Qt6 issue with OpenCV 4.13.0.")
            print("\n   Possible causes:")
            print("   1. Qt6 backend has compatibility issues")
            print("   2. No X11 display available (check DISPLAY variable)")
            print("   3. Running via SSH without X11 forwarding (use ssh -X)")
            print("\n   Solutions:")
            print("   A. Check display: echo $DISPLAY")
            print("   B. Test GUI: python scripts/check_opencv_display.py")
            print("   C. Consider rebuilding OpenCV with GTK instead of Qt")
            print("   D. Or use a machine with direct display access")
            raise SystemExit(1)

        n_req = sum(1 for p in self.points if p['required'])
        print(f"\n  {n_req} required points (floor corners) + "
              f"{len(self.points) - n_req} optional (cage top).")
        print("  Click on the monkey's position for each labeled arena point.\n")

        while True:
            cv2.imshow(self.win,
                       render(self.frame, self.points, self.current_idx,
                              self.camera, self.H, self.reproj_err))

            if self._advance_requested:
                self._advance_requested = False
                if self.current_idx < len(self.points) - 1:
                    self.current_idx += 1

            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key in (ord('n'), 83):     # N or right arrow
                self.current_idx = min(len(self.points) - 1, self.current_idx + 1)
            elif key in (ord('p'), 81):     # P or left arrow
                self.current_idx = max(0, self.current_idx - 1)
            elif key == ord('u'):           # Undo
                self.points[self.current_idx]['pixel'] = None
                self._try_compute()
            elif key == ord('c'):           # Compute
                self._try_compute()
                self._print_summary()
            elif key == ord('s'):           # Save
                self.save()

        cv2.destroyAllWindows()
        self._try_compute()
        self._print_summary()
        self.save()

    # ── Save ────────────────────────────────────────────────────────────────────

    def save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        n = sum(1 for p in self.points if p['pixel'] is not None)
        q = CAMERA_QUADRANTS[self.camera]

        data = {
            'camera':            self.camera,
            'quadrant_label':    q['label'],
            'quadrant_x_cm':     list(q['x']),
            'quadrant_y_cm':     list(q['y']),
            'arena_size_cm':     [ARENA_W, ARENA_H],
            'cage_height_cm':    CAGE_Z,
            'video_path':        str(self.video_path),
            'frame_number':      self.frame_num,
            'image_size_wh':     [self.w, self.h],
            'labeled_points':    [
                {
                    'name':     p['name'],
                    'world_cm': p['world'],
                    'pixel':    list(p['pixel']) if p['pixel'] else None,
                    'required': p['required'],
                }
                for p in self.points
            ],
            'floor_homography':        self.H.tolist() if self.H is not None else None,
            'floor_reproj_error_cm':   self.reproj_err if self.H is not None else None,
            'solvepnp':                compute_solvepnp(self.points, (self.w, self.h)),
            'saved_at':                datetime.now().isoformat(),
        }

        with open(self.out_json, 'w') as f:
            json.dump(data, f, indent=2)

        # Save calibration frame with annotations for reference
        annotated = self.frame.copy()
        for p in self.points:
            if p['pixel'] is None:
                continue
            px, py = int(p['pixel'][0]), int(p['pixel'][1])
            is_floor = p['world'][2] == 0
            col = (80, 200, 80) if is_floor else (220, 150, 80)
            cv2.drawMarker(annotated, (px, py), col,
                           cv2.MARKER_CROSS if is_floor else cv2.MARKER_DIAMOND,
                           markerSize=80, thickness=4)
            cv2.putText(annotated, p['name'], (px + 20, py - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, col, 3, cv2.LINE_AA)
        cv2.imwrite(str(self.out_png), annotated)

        print(f"\n✅  Saved {n} labels → {self.out_json}")
        print(f"    Reference frame  → {self.out_png}")
        if self.H is not None:
            print(f"    Floor homography reprojection error: {self.reproj_err:.2f} cm")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Camera calibration for 4-camera OFT monkey setup',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--camera', '-c', choices=['D1', 'D2', 'D3', 'D4'],
                        help='Camera to calibrate (default: interactive)')
    parser.add_argument('--video',  '-v', default=None,
                        help='Video file path (default: auto-detect)')
    parser.add_argument('--frame-sec', '-f', type=float, default=30.0,
                        help='Time (sec) from which to pull calibration frame (default: 30)')
    parser.add_argument('--output-dir', '-o', default=str(OUTPUT_DIR),
                        help=f'Output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--all', action='store_true',
                        help='Calibrate all four cameras sequentially')
    args = parser.parse_args()

    out = Path(args.output_dir)

    # Determine which cameras to run
    if args.all:
        cameras = ['D1', 'D2', 'D3', 'D4']
    elif args.camera:
        cameras = [args.camera]
    else:
        print("Arena layout:  D1=top-left  D2=top-right  D3=bottom-left  D4=bottom-right")
        for c, q in CAMERA_QUADRANTS.items():
            print(f"  {c}: {q['label']} quadrant  x={q['x']} cm  y={q['y']} cm")
        choice = input("\nEnter camera to calibrate (D1/D2/D3/D4): ").strip().upper()
        if choice not in CAMERA_QUADRANTS:
            sys.exit(f"Invalid camera: {choice}")
        cameras = [choice]

    skipped = []
    for cam in cameras:
        if args.video and len(cameras) == 1:
            vpath = Path(args.video)
        else:
            vpath = find_video(cam, VIDEO_BASE)

        if vpath is None:
            skipped.append(cam)
            continue

        cal = CameraCalibrator(
            camera=cam,
            video_path=vpath,
            output_dir=out,
            frame_sec=args.frame_sec,
        )
        cal.run()

    if skipped:
        print(f"\n⚠  Skipped (no video): {', '.join(skipped)}")
        print(f"   To calibrate manually:  python scripts/calibrate_cameras.py "
              f"--camera <D#> --video /path/to/video.mp4")
    print(f"\n✅  Done. Calibration files in: {out}")


if __name__ == '__main__':
    main()
