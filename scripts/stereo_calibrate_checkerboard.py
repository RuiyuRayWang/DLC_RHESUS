#!/usr/bin/env python3
"""
Checkerboard-based multi-camera intrinsic calibration for OFT cameras (D01-D04).

This script:
1. Detects checkerboard corners in calibration images.
2. Calibrates intrinsics for each camera independently.

Expected input filename pattern in --images-dir:
    D01-001.jpg, D02-001.jpg, D03-017.jpg, ...

Where:
    - Prefix (D01..D04) indicates camera index.
    - Suffix number is a per-camera checkerboard pose ID.

Outputs:
    results/stereo_calibration/stereo_calibration_checkerboard.json

Usage examples:
    python scripts/stereo_calibrate_checkerboard.py \
      --board-cols 9 --board-rows 6 --square-size-cm 2.5

    python scripts/stereo_calibrate_checkerboard.py \
      --images-dir results/stereo_calibration/extracted_frames \
      --board-cols 11 --board-rows 8 --square-size-cm 2.0 \
            --write-debug-overlays
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "results" / "stereo_calibration" / "extracted_frames"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "stereo_calibration" / "stereo_calibration_checkerboard.json"
DEFAULT_DEBUG_DIR = PROJECT_ROOT / "results" / "stereo_calibration" / "debug_corners"
DEFAULT_UNDISTORTED_DIR = PROJECT_ROOT / "results" / "stereo_calibration" / "undistorted_examples"
DEFAULT_GUI_OUTPUT_DIR = PROJECT_ROOT / "results" / "calibration"

CAGE_Z = 105

FNAME_RE = re.compile(r"^(D0[1-4])-(\d+)\.(jpg|jpeg|png|bmp|tif|tiff)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate D01-D04 camera intrinsics from checkerboard images."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help=f"Directory containing checkerboard images (default: {DEFAULT_IMAGES_DIR})",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output calibration JSON (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--board-cols",
        type=int,
        required=True,
        help="Number of checkerboard inner corners along columns.",
    )
    parser.add_argument(
        "--board-rows",
        type=int,
        required=True,
        help="Number of checkerboard inner corners along rows.",
    )
    parser.add_argument(
        "--square-size-cm",
        type=float,
        required=True,
        help="Checkerboard square size in centimeters.",
    )
    parser.add_argument(
        "--min-images-per-camera",
        type=int,
        default=6,
        help="Minimum successful checkerboard detections needed for intrinsics per camera.",
    )
    parser.add_argument(
        "--write-debug-overlays",
        action="store_true",
        help="Write corner-detection overlay images under results/stereo_calibration/debug_corners.",
    )
    parser.add_argument(
        "--no-undistorted-examples",
        action="store_true",
        help="Disable writing undistorted example images per camera.",
    )
    parser.add_argument(
        "--launch-gui",
        action="store_true",
        help="Launch an interactive corner-confirmation GUI after intrinsics are calibrated.",
    )
    parser.add_argument(
        "--gui-output-dir",
        type=Path,
        default=DEFAULT_GUI_OUTPUT_DIR,
        help=f"Directory for downstream per-camera calibration outputs (default: {DEFAULT_GUI_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--gui-images-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing clean arena frames for manual corner labeling "
            "(uses D01*/D02*/D03*/D04* filenames; if omitted, falls back to undistorted checkerboard examples)."
        ),
    )
    return parser.parse_args()


GUI_CAMERA_QUADRANTS: Dict[str, dict] = {
    "D01": {"x": (0, 161), "y": (0, 161), "label": "top-left"},
    "D02": {"x": (161, 322), "y": (0, 161), "label": "top-right"},
    "D03": {"x": (0, 161), "y": (161, 322), "label": "bottom-left"},
    "D04": {"x": (161, 322), "y": (161, 322), "label": "bottom-right"},
}


def _corner_desc(xx: int, yy: int, zz: int) -> str:
    x_tag = "outer" if xx in (0, 322) else "center-line"
    y_tag = "outer" if yy in (0, 322) else "center-line"
    layer = "floor" if zz == 0 else f"cage top (z={zz})"
    return f"{layer}  {x_tag}×{y_tag}"


def _build_gui_point_list(camera: str) -> List[dict]:
    q = GUI_CAMERA_QUADRANTS[camera]
    x0, x1 = q["x"]
    y0, y1 = q["y"]

    corners = [("TL", x0, y0), ("TR", x1, y0), ("BR", x1, y1), ("BL", x0, y1)]
    pts: List[dict] = []

    for tag, xx, yy in corners:
        pts.append(
            {
                "name": f"Floor {tag} ({xx},{yy})",
                "desc": _corner_desc(xx, yy, 0),
                "world": [xx, yy, 0],
                "pixel": None,
                "required": True,
            }
        )

    for tag, xx, yy in corners:
        pts.append(
            {
                "name": f"Top   {tag} ({xx},{yy})",
                "desc": _corner_desc(xx, yy, CAGE_Z),
                "world": [xx, yy, CAGE_Z],
                "pixel": None,
                "required": False,
            }
        )

    return pts


def _compute_floor_homography(points: List[dict]) -> Tuple[Optional[np.ndarray], float]:
    fp = [p for p in points if p["pixel"] is not None and p["world"][2] == 0]
    if len(fp) < 4:
        return None, float("inf")

    src = np.array([[p["pixel"][0], p["pixel"][1]] for p in fp], dtype=np.float64)
    dst = np.array([[p["world"][0], p["world"][1]] for p in fp], dtype=np.float64)

    h_mat, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if h_mat is None:
        return None, float("inf")

    src_h = np.hstack([src, np.ones((len(src), 1))])
    proj = (h_mat @ src_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    err = float(np.mean(np.linalg.norm(proj - dst, axis=1)))
    return h_mat, err


def _compute_solvepnp(points: List[dict], image_size: Tuple[int, int]) -> Optional[dict]:
    labeled = [p for p in points if p["pixel"] is not None]
    if len(labeled) < 6:
        return None
    heights = {p["world"][2] for p in labeled}
    if len(heights) < 2:
        return None

    obj = np.array([p["world"] for p in labeled], dtype=np.float64)
    img = np.array([p["pixel"] for p in labeled], dtype=np.float64)

    w, h = image_size
    k_mat = np.array([[float(w), 0, w / 2.0], [0, float(w), h / 2.0], [0, 0, 1.0]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)

    try:
        ok, rvec, tvec = cv2.solvePnP(obj, img.reshape(-1, 1, 2), k_mat, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        proj, _ = cv2.projectPoints(obj, rvec, tvec, k_mat, dist)
        err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)))
        return {
            "camera_matrix": k_mat.tolist(),
            "dist_coeffs": dist.tolist(),
            "rvec": rvec.flatten().tolist(),
            "tvec": tvec.flatten().tolist(),
            "reprojection_error_px": err,
            "n_points": len(labeled),
        }
    except cv2.error:
        return None


def _pixel_to_cm(pixel_xy: Tuple[float, float], h_mat: np.ndarray) -> Tuple[float, float]:
    pt = np.array([[[float(pixel_xy[0]), float(pixel_xy[1])]]], dtype=np.float64)
    r = cv2.perspectiveTransform(pt, h_mat)
    return float(r[0, 0, 0]), float(r[0, 0, 1])


def _export_undistorted_example(intr: dict, out_dir: Path) -> Optional[Path]:
    """Write one undistorted example image for a successfully calibrated camera."""
    if not intr.get("success", False):
        return None

    detections = intr.get("detections", {})
    detected_pose_ids = intr.get("detected_pose_ids", [])
    if not detected_pose_ids:
        return None

    pose_id = detected_pose_ids[0]
    image_path_str = detections.get(pose_id, {}).get("image")
    if not image_path_str:
        return None

    image_path = Path(image_path_str)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return None

    k = np.array(intr["camera_matrix"], dtype=np.float64)
    d = np.array(intr["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    und = cv2.undistort(img, k, d)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{intr['camera']}_{pose_id}_undistorted.jpg"
    cv2.imwrite(str(out_path), und)
    return out_path


def _gui_put(img: np.ndarray, text: str, x: int, y: int, color=(230, 230, 230), scale: float = 0.55, thick: int = 1) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _make_gui_sidebar(points: List[dict], current_idx: int, camera: str, h_mat: Optional[np.ndarray], reproj_err: float, panel_h: int) -> np.ndarray:
    panel_w = 430
    bg = (30, 30, 30)
    c_white = (230, 230, 230)
    c_yellow = (0, 220, 220)
    c_green = (80, 200, 80)
    c_blue = (220, 150, 80)
    c_gray = (150, 150, 150)
    c_red = (80, 80, 220)
    c_current = (0, 200, 255)

    p = np.full((panel_h, panel_w, 3), bg, dtype=np.uint8)
    cv2.rectangle(p, (0, 0), (panel_w, 46), (50, 50, 50), -1)
    q = GUI_CAMERA_QUADRANTS[camera]
    _gui_put(p, f"Camera {camera} — {q['label']}", 10, 30, c_yellow, 0.72, 2)

    n_all = sum(1 for pt in points if pt["pixel"] is not None)
    n_floor = sum(1 for pt in points if pt["pixel"] is not None and pt["world"][2] == 0)
    _gui_put(p, f"Labeled: {n_all}/{len(points)}   floor: {n_floor}/4", 10, 64)
    if h_mat is not None:
        col = c_green if reproj_err < 5 else (c_red if reproj_err > 20 else c_yellow)
        _gui_put(p, f"Reprojection error: {reproj_err:.1f} cm", 10, 84, col)
    else:
        _gui_put(p, "Need ≥ 4 floor pts for homography", 10, 84, c_gray, 0.48)
    cv2.line(p, (0, 100), (panel_w, 100), (70, 70, 70), 1)

    y, row_h = 112, 44
    for i, pt in enumerate(points):
        is_cur = i == current_idx
        is_lab = pt["pixel"] is not None
        is_req = pt["required"]
        bg_row = (55, 55, 85) if is_cur else (40, 40, 40)
        cv2.rectangle(p, (5, y - 26), (panel_w - 5, y + row_h - 30), bg_row, -1)
        if is_cur:
            cv2.rectangle(p, (5, y - 26), (panel_w - 5, y + row_h - 30), c_current, 1)

        icon = "✓" if is_lab else ("►" if is_cur else ("○" if is_req else "◇"))
        ic = c_green if is_lab else (c_current if is_cur else (c_gray if is_req else c_blue))
        nc = c_current if is_cur else (c_green if is_lab else (c_white if is_req else c_blue))
        _gui_put(p, icon, 12, y, ic, 0.62, 2 if is_cur else 1)
        _gui_put(p, pt["name"], 36, y, nc, 0.48, 2 if is_cur else 1)

        if is_lab:
            px, py = pt["pixel"]
            _gui_put(p, f"  pixel ({px}, {py})", 36, y + 16, c_green, 0.40)
        else:
            _gui_put(p, f"  {pt['desc']}", 36, y + 16, c_current if is_cur else c_gray, 0.40)

        y += row_h
        if y > panel_h - 160:
            _gui_put(p, f"  … {len(points) - i - 1} more points below", 12, y, c_gray, 0.42)
            break

    fy = panel_h - 148
    cv2.line(p, (0, fy), (panel_w, fy), (70, 70, 70), 1)
    controls = [
        ("L-click", "label & next"),
        ("R-click", "label, stay"),
        ("N / →", "next point"),
        ("P / ←", "previous point"),
        ("U", "undo label"),
        ("C", "compute calibration"),
        ("S", "save JSON now"),
        ("Q / ESC", "save & quit"),
    ]
    yc = fy + 18
    for key, desc in controls:
        _gui_put(p, f"{key:<9}  {desc}", 10, yc, c_gray, 0.44)
        yc += 17
    return p


def _render_gui(frame_orig: np.ndarray, points: List[dict], current_idx: int, camera: str, h_mat: Optional[np.ndarray], reproj_err: float) -> np.ndarray:
    display_scale = 0.35
    sidebar_w = 430
    c_current = (0, 200, 255)
    c_green = (80, 200, 80)
    c_blue = (220, 150, 80)
    c_red = (80, 80, 220)
    c_yellow = (0, 220, 220)

    h_o, w_o = frame_orig.shape[:2]
    dw = int(w_o * display_scale)
    dh = int(h_o * display_scale)
    disp = cv2.resize(frame_orig, (dw, dh), interpolation=cv2.INTER_AREA)

    for i, pt in enumerate(points):
        if pt["pixel"] is None:
            continue
        dx = int(pt["pixel"][0] * display_scale)
        dy = int(pt["pixel"][1] * display_scale)
        is_floor = pt["world"][2] == 0
        col = c_current if i == current_idx else (c_green if is_floor else c_blue)
        marker = cv2.MARKER_CROSS if is_floor else cv2.MARKER_DIAMOND
        cv2.drawMarker(disp, (dx, dy), col, marker, markerSize=32, thickness=2)
        cv2.putText(disp, pt["name"].split("(")[0].strip(), (dx + 8, dy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)

        if h_mat is not None and is_floor:
            cx, cy = _pixel_to_cm(pt["pixel"], h_mat)
            wx, wy = pt["world"][0], pt["world"][1]
            ecm = float(np.hypot(cx - wx, cy - wy))
            ec = c_green if ecm < 5 else c_red
            cv2.putText(disp, f"{ecm:.1f}cm", (dx + 8, dy + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, ec, 1, cv2.LINE_AA)

    # Draw only a thin frame border; avoid overlaying any banner on top of the image
    # so cage corners remain fully visible during manual labeling.
    cv2.rectangle(disp, (0, 0), (dw - 1, dh - 1), c_current, 2)

    sidebar = _make_gui_sidebar(points, current_idx, camera, h_mat, reproj_err, dh)
    return np.hstack([disp, sidebar])


class ArenaCornerCalibrator:
    def __init__(self, camera: str, image_path: Path, output_dir: Path, source_intrinsics: dict):
        self.camera = camera
        self.image_path = image_path
        self.output_dir = output_dir
        self.source_intrinsics = source_intrinsics
        self.out_json = output_dir / f"calibration_{camera}.json"
        self.out_png = output_dir / f"calibration_frame_{camera}.png"

        self.frame = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if self.frame is None:
            raise FileNotFoundError(f"Could not read GUI image: {self.image_path}")
        self.h, self.w = self.frame.shape[:2]

        self.points: List[dict] = _build_gui_point_list(camera)
        self.current_idx = 0
        self._advance_requested = False
        self.H: Optional[np.ndarray] = None
        self.reproj_err = float("inf")
        self._try_compute()

        self.win = f"Calibrate_{camera}"
        self.win_title = f"Calibrate {camera} - L-click=label  N=next  P=prev  U=undo  Q=save&quit"

    def _try_compute(self) -> None:
        h_mat, err = _compute_floor_homography(self.points)
        if h_mat is not None:
            self.H = h_mat
            self.reproj_err = err

    def _mouse_cb(self, event, mx, my, flags, param) -> None:
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        if mx >= int(self.w * 0.35):
            return
        ox = int(mx / 0.35)
        oy = int(my / 0.35)
        self.points[self.current_idx]["pixel"] = (ox, oy)
        self._try_compute()
        if event == cv2.EVENT_LBUTTONDOWN:
            self._advance_requested = True

    def _print_summary(self) -> None:
        n = sum(1 for p in self.points if p["pixel"] is not None)
        nf = sum(1 for p in self.points if p["pixel"] is not None and p["world"][2] == 0)
        print(f"\n── Calibration summary — camera {self.camera} ──")
        print(f"  Labeled: {n}/{len(self.points)}  (floor: {nf})")
        if self.H is not None:
            print(f"  Floor homography  reprojection error: {self.reproj_err:.2f} cm")
            for p in self.points:
                if p["pixel"] is not None and p["world"][2] == 0:
                    cx, cy = _pixel_to_cm(p["pixel"], self.H)
                    wx, wy = p["world"][0], p["world"][1]
                    print(f"    {p['name']:<28}  error = {np.hypot(cx - wx, cy - wy):.2f} cm")
        else:
            print("  Floor homography: NOT COMPUTED (need ≥ 4 floor pts)")
        pnp = _compute_solvepnp(self.points, (self.w, self.h))
        if pnp:
            print(f"  solvePnP pixel reprojection error: {pnp['reprojection_error_px']:.2f} px  (from {pnp['n_points']} pts)")

    def save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        q = GUI_CAMERA_QUADRANTS[self.camera]
        data = {
            "camera": self.camera,
            "quadrant_label": q["label"],
            "quadrant_x_cm": list(q["x"]),
            "quadrant_y_cm": list(q["y"]),
            "arena_size_cm": [322, 322],
            "cage_height_cm": 105,
            "image_path": str(self.image_path),
            "image_size_wh": [self.w, self.h],
            "source_intrinsics": {
                "rms_reprojection_error": self.source_intrinsics.get("rms_reprojection_error"),
                "camera_matrix": self.source_intrinsics.get("camera_matrix"),
                "dist_coeffs": self.source_intrinsics.get("dist_coeffs"),
                "image_size": self.source_intrinsics.get("image_size"),
            },
            "labeled_points": [
                {
                    "name": p["name"],
                    "world_cm": p["world"],
                    "pixel": list(p["pixel"]) if p["pixel"] else None,
                    "required": p["required"],
                }
                for p in self.points
            ],
            "floor_homography": self.H.tolist() if self.H is not None else None,
            "floor_reproj_error_cm": self.reproj_err if self.H is not None else None,
            "solvepnp": _compute_solvepnp(self.points, (self.w, self.h)),
            "saved_at": datetime.now().isoformat(),
        }

        with open(self.out_json, "w") as f:
            json.dump(data, f, indent=2)

        annotated = self.frame.copy()
        for p in self.points:
            if p["pixel"] is None:
                continue
            px, py = int(p["pixel"][0]), int(p["pixel"][1])
            is_floor = p["world"][2] == 0
            col = (80, 200, 80) if is_floor else (220, 150, 80)
            cv2.drawMarker(annotated, (px, py), col, cv2.MARKER_CROSS if is_floor else cv2.MARKER_DIAMOND, markerSize=80, thickness=4)
            cv2.putText(annotated, p["name"], (px + 20, py - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.8, col, 3, cv2.LINE_AA)
        cv2.imwrite(str(self.out_png), annotated)

        print(f"\n✅  Saved {sum(1 for p in self.points if p['pixel'] is not None)} labels → {self.out_json}")
        print(f"    Reference frame  → {self.out_png}")
        if self.H is not None:
            print(f"    Floor homography reprojection error: {self.reproj_err:.2f} cm")

    def run(self) -> None:
        print(f"\n  Opening window: {self.win_title}")
        try:
            try:
                cv2.startWindowThread()
            except Exception:
                pass

            cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
            try:
                cv2.setWindowTitle(self.win, self.win_title)
            except Exception:
                pass

            cv2.resizeWindow(self.win, int(self.w * 0.35) + 430, int(self.h * 0.35))
            import time

            time.sleep(0.15)
            cv2.imshow(self.win, _render_gui(self.frame, self.points, self.current_idx, self.camera, self.H, self.reproj_err))
            cv2.waitKey(1)
            cv2.setMouseCallback(self.win, self._mouse_cb)
        except cv2.error as e:
            print("\n❌ ERROR: Failed to create OpenCV window with GUI.")
            print(f"   Details: {e}")
            raise SystemExit(1)

        print("\n  Click floor corners first, then optional cage-top corners.")
        while True:
            cv2.imshow(self.win, _render_gui(self.frame, self.points, self.current_idx, self.camera, self.H, self.reproj_err))
            if self._advance_requested:
                self._advance_requested = False
                if self.current_idx < len(self.points) - 1:
                    self.current_idx += 1

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key in (ord("n"), 83):
                self.current_idx = min(len(self.points) - 1, self.current_idx + 1)
            elif key in (ord("p"), 81):
                self.current_idx = max(0, self.current_idx - 1)
            elif key == ord("u"):
                self.points[self.current_idx]["pixel"] = None
                self._try_compute()
            elif key == ord("c"):
                self._try_compute()
                self._print_summary()
            elif key == ord("s"):
                self.save()

        cv2.destroyAllWindows()
        self._try_compute()
        self._print_summary()
        self.save()


def _make_object_points(cols: int, rows: int, square_size_cm: float) -> np.ndarray:
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid * float(square_size_cm)
    return objp


def _load_images_by_camera(images_dir: Path) -> Dict[str, Dict[str, Path]]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    by_cam: Dict[str, Dict[str, Path]] = {"D01": {}, "D02": {}, "D03": {}, "D04": {}}
    for p in sorted(images_dir.iterdir()):
        if not p.is_file():
            continue
        m = FNAME_RE.match(p.name)
        if not m:
            continue
        cam = m.group(1).upper()
        pose_id = m.group(2)
        by_cam[cam][pose_id] = p
    return by_cam


def _detect_corners(
    image_path: Path,
    board_size: Tuple[int, int],
    write_overlay: bool,
    overlay_out_path: Optional[Path],
) -> Tuple[bool, Optional[np.ndarray], Tuple[int, int], str]:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return False, None, (0, 0), "read_failed"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_size = (gray.shape[1], gray.shape[0])

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, board_size, flags=flags)

    if ok:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
        status = "ok"
    else:
        try:
            ok_sb, corners_sb = cv2.findChessboardCornersSB(gray, board_size, flags=0)
            if ok_sb:
                ok = True
                corners = corners_sb.astype(np.float32)
                status = "ok_sb"
            else:
                status = "not_found"
        except cv2.error:
            status = "not_found"

    if write_overlay and overlay_out_path is not None:
        overlay_out_path.parent.mkdir(parents=True, exist_ok=True)
        dbg = img.copy()
        if ok and corners is not None:
            cv2.drawChessboardCorners(dbg, board_size, corners, ok)
            cv2.putText(
                dbg,
                "DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (80, 220, 80),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                dbg,
                "NOT FOUND",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (80, 80, 220),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(overlay_out_path), dbg)

    return ok, corners, image_size, status


def _calibrate_intrinsics(
    cam: str,
    cam_images: Dict[str, Path],
    objp: np.ndarray,
    board_size: Tuple[int, int],
    write_overlays: bool,
    debug_dir: Path,
    min_images: int,
) -> dict:
    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []
    detection_by_pose: Dict[str, dict] = {}
    image_size: Optional[Tuple[int, int]] = None

    for pose_id, img_path in sorted(cam_images.items(), key=lambda kv: int(kv[0])):
        overlay_path = debug_dir / cam / f"{img_path.stem}_corners.jpg" if write_overlays else None
        ok, corners, sz, status = _detect_corners(img_path, board_size, write_overlays, overlay_path)

        detection_by_pose[pose_id] = {
            "image": str(img_path),
            "status": status,
            "found": bool(ok),
        }

        if not ok or corners is None:
            continue

        if image_size is None:
            image_size = sz
        elif image_size != sz:
            detection_by_pose[pose_id]["status"] = "size_mismatch"
            detection_by_pose[pose_id]["found"] = False
            continue

        objpoints.append(objp.copy())
        imgpoints.append(corners.reshape(-1, 2).astype(np.float32))

    found_ids = sorted(
        [pid for pid, info in detection_by_pose.items() if info["found"]],
        key=lambda x: int(x),
    )

    result = {
        "camera": cam,
        "n_images_total": len(cam_images),
        "n_detected": len(found_ids),
        "detected_pose_ids": found_ids,
        "image_size": list(image_size) if image_size is not None else None,
        "detections": detection_by_pose,
        "success": False,
        "rms_reprojection_error": None,
        "camera_matrix": None,
        "dist_coeffs": None,
        "rvecs": None,
        "tvecs": None,
    }

    if image_size is None or len(objpoints) < min_images:
        result["failure_reason"] = (
            f"insufficient detections ({len(objpoints)} < {min_images}) or no valid image size"
        )
        return result

    rms, k, d, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        [p.reshape(-1, 1, 2) for p in imgpoints],
        image_size,
        None,
        None,
    )

    result.update(
        {
            "success": True,
            "rms_reprojection_error": float(rms),
            "camera_matrix": k.tolist(),
            "dist_coeffs": d.flatten().tolist(),
            "rvecs": [rv.flatten().tolist() for rv in rvecs],
            "tvecs": [tv.flatten().tolist() for tv in tvecs],
        }
    )
    return result


def _resolve_gui_image_path(
    camera: str,
    gui_images_dir: Optional[Path],
    fallback_undistorted: Dict[str, Optional[str]],
) -> Optional[Path]:
    """Choose GUI image for one camera, preferring user-provided clean arena images."""
    if gui_images_dir is not None:
        if not gui_images_dir.exists():
            raise FileNotFoundError(f"GUI image directory not found: {gui_images_dir}")

        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        candidates: List[Path] = []
        for ext in exts:
            candidates.extend(sorted(gui_images_dir.glob(f"{camera}_*{ext[1:]}")))
            candidates.extend(sorted(gui_images_dir.glob(f"{camera}-*{ext[1:]}")))
            candidates.extend(sorted(gui_images_dir.glob(f"{camera}*{ext[1:]}")))

        # Deduplicate while preserving order.
        seen = set()
        unique: List[Path] = []
        for p in candidates:
            if p in seen:
                continue
            seen.add(p)
            unique.append(p)

        if unique:
            return unique[0]
        return None

    image_path_str = fallback_undistorted.get(camera)
    if not image_path_str:
        return None
    return Path(image_path_str)


def main() -> None:
    args = parse_args()

    if args.board_cols < 2 or args.board_rows < 2:
        raise SystemExit("ERROR: --board-cols and --board-rows must be >= 2")
    if args.square_size_cm <= 0:
        raise SystemExit("ERROR: --square-size-cm must be > 0")

    by_cam = _load_images_by_camera(args.images_dir)
    board_size = (args.board_cols, args.board_rows)
    objp = _make_object_points(args.board_cols, args.board_rows, args.square_size_cm)

    print("\n=== Checkerboard Image Inventory ===")
    for cam in ["D01", "D02", "D03", "D04"]:
        print(f"  {cam}: {len(by_cam[cam])} files")

    intrinsics: Dict[str, dict] = {}
    debug_dir = DEFAULT_DEBUG_DIR
    undistorted_dir = DEFAULT_UNDISTORTED_DIR
    undistorted_examples: Dict[str, Optional[str]] = {}
    gui_output_dir = args.gui_output_dir
    gui_images_dir = args.gui_images_dir
    print("\n=== Intrinsic Calibration ===")
    for cam in ["D01", "D02", "D03", "D04"]:
        intr = _calibrate_intrinsics(
            cam=cam,
            cam_images=by_cam[cam],
            objp=objp,
            board_size=board_size,
            write_overlays=args.write_debug_overlays,
            debug_dir=debug_dir,
            min_images=args.min_images_per_camera,
        )
        intrinsics[cam] = intr
        if intr["success"]:
            print(
                f"  ✓ {cam}: used {intr['n_detected']}/{intr['n_images_total']} images, "
                f"RMS={intr['rms_reprojection_error']:.4f}"
            )
        else:
            print(f"  ✗ {cam}: {intr.get('failure_reason', 'unknown_failure')}")

        if args.no_undistorted_examples:
            undistorted_examples[cam] = None
        else:
            out_path = _export_undistorted_example(intr, undistorted_dir)
            undistorted_examples[cam] = str(out_path) if out_path is not None else None
            if out_path is not None:
                print(f"    ↳ undistorted example: {out_path}")

    if args.launch_gui:
        print("\n=== Downstream Corner Confirmation GUI ===")
        if gui_images_dir is not None:
            print(f"  Using GUI images from: {gui_images_dir}")
        else:
            print("  Using undistorted checkerboard examples (no --gui-images-dir provided)")
        for cam in ["D01", "D02", "D03", "D04"]:
            intr = intrinsics.get(cam, {})
            if not intr.get("success", False):
                print(f"  - {cam}: skipped (intrinsics not available)")
                continue

            image_path = _resolve_gui_image_path(cam, gui_images_dir, undistorted_examples)
            if image_path is None:
                source_hint = (
                    f"no matching image in {gui_images_dir}" if gui_images_dir is not None else "no undistorted example available"
                )
                print(f"  - {cam}: skipped ({source_hint})")
                continue

            if not image_path.exists():
                print(f"  - {cam}: skipped (missing image {image_path})")
                continue

            print(f"  - {cam}: launching GUI on {image_path.name}")
            gui = ArenaCornerCalibrator(
                camera=cam,
                image_path=image_path,
                output_dir=gui_output_dir,
                source_intrinsics=intr,
            )
            gui.run()

    print("\n=== Stereo Extrinsics ===")
    print("  Skipped by design: checkerboard pose IDs are independent across cameras.")
    print("  Provide synchronized multi-camera checkerboard captures to estimate extrinsics.")

    out = {
        "meta": {
            "images_dir": str(args.images_dir),
            "board_inner_corners": [args.board_cols, args.board_rows],
            "square_size_cm": float(args.square_size_cm),
            "min_images_per_camera": int(args.min_images_per_camera),
            "undistorted_examples_enabled": not args.no_undistorted_examples,
            "launch_gui_enabled": bool(args.launch_gui),
            "gui_output_dir": str(gui_output_dir),
            "gui_images_dir": str(gui_images_dir) if gui_images_dir else None,
            "cameras": ["D01", "D02", "D03", "D04"],
            "stereo_extrinsics_computed": False,
            "stereo_extrinsics_note": (
                "Not computed because checkerboard pose IDs are independent across cameras."
            ),
        },
        "intrinsics": intrinsics,
        "undistorted_examples": undistorted_examples,
        "stereo_pairs": {},
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Done ===")
    print(f"Wrote calibration JSON: {args.output_json}")
    if not args.no_undistorted_examples:
        print(f"Wrote undistorted examples under: {undistorted_dir}")
    if args.write_debug_overlays:
        print(f"Wrote debug overlays under: {debug_dir}")


if __name__ == "__main__":
    main()
