#!/usr/bin/env python3
"""
Review synchronized 4-camera predictions with fused trajectory overlay.

This viewer is intended for date-level QC after RT-DETR prediction + manual
validate-mode refinement. It opens a 2x2 camera mosaic, overlays predicted
bounding boxes for each frame, and shows a small arena map in the lower-right
with the nearest fused trajectory point.

Inputs (date-level):
  - extracted frames:   tmp/rtdetr_frames_2hz/<monkey>/<date>/D0X_*_downsampled/
  - predictions CSVs:   results/predictions/<monkey>/<date>/D0X_*_preds.csv
  - fused trajectory:   results/oft_analysis/<monkey>/<date>/trajectory_realworld.csv

Pairing strategy:
  - For each camera (D01..D04), prediction CSVs are sorted by filename.
  - A bundle is formed by taking the i-th prediction CSV from each camera.
  - Each CSV must have a matching extracted-frame directory with the same stem
    (minus _preds.csv).

This keeps the operation minimal and robust for day-level review where files are
already organized by chronological naming.

Controls:
  - N / Right: next frame
  - P / Left: previous frame
  - K: jump +10 frames
  - J: jump -10 frames
  - M: next bundle
  - B: previous bundle
    - 1/2/3/4: toggle active edit cameras (multi-select D01/D02/D03/D04)
    - 5: toggle trajectory ignore-mask edit mode
    - 0: clear all edit selections
    - I: toggle trajectory ignore-frame mask when mode 5 is active (press again to undo)
    - U: apply "no prediction" on all active cameras
  - H: hide/show bbox overlays
  - T: hide/show trajectory tail on arena
  - Q / ESC: quit

Usage:
  conda activate video_tracking
  python scripts/review_multicam_predictions.py --monkey H1705011 --date 2026-06-26
  python scripts/review_multicam_predictions.py --monkey H1705011 --date 2026-06-26 --bundle-index 2
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FPS_DEFAULT = 20.0
CAMERAS = ["D01", "D02", "D03", "D04"]
FRAME_RE = re.compile(r"_f(\d+)$")
TS_RE = re.compile(r"(?<!\d)(\d{14})(?!\d)")
MASK_FILENAME = "trajectory_ignore_mask.csv"
FRAME_TIME_TOLERANCE_SEC = 0.08

ARENA_SIZE_CM = 322.0
PANEL_W = 800
PANEL_H = 480
PANEL_INFO_H = 52
SIDEBAR_W = 320

C_GREEN = (80, 200, 80)
C_RED = (60, 60, 220)
C_YELLOW = (0, 220, 220)
C_WHITE = (245, 245, 245)
C_GRAY = (180, 180, 180)
C_BLACK = (20, 20, 20)
C_BLUE = (220, 140, 60)


@dataclass
class CameraEntry:
    camera: str
    stem: str
    pred_csv: Path
    frames_dir: Path
    pred_df: pd.DataFrame
    frame_map: dict[int, Path]
    start_dt: Optional[datetime]
    end_dt: Optional[datetime]
    frame_indices: np.ndarray
    frame_times_sec: np.ndarray


@dataclass
class Bundle:
    index: int
    entries: dict[str, list[CameraEntry]]
    timeline: list[datetime]


class TrajectoryIndex:
    def __init__(self, traj_path: Path):
        self.path = traj_path
        self.mask_path = traj_path.parent / MASK_FILENAME
        self.df = pd.read_csv(traj_path)
        if "absolute_time" not in self.df.columns or "x_cm" not in self.df.columns or "y_cm" not in self.df.columns:
            raise ValueError(f"{traj_path}: requires absolute_time, x_cm, y_cm columns")
        self.df["absolute_time"] = pd.to_datetime(self.df["absolute_time"], errors="coerce")
        self.df = self.df.dropna(subset=["absolute_time", "x_cm", "y_cm"]).reset_index(drop=True)
        self.df["ignore_frame"] = False
        if self.df.empty:
            raise ValueError(f"{traj_path}: no valid rows after parsing absolute_time/x_cm/y_cm")
        self._load_mask_file()
        self._dirty = False
        self._refresh_visible_index()

    def _load_mask_file(self) -> None:
        if not self.mask_path.exists():
            return
        try:
            m = pd.read_csv(self.mask_path)
        except Exception:
            return

        if m.empty:
            return

        applied = np.zeros(len(self.df), dtype=bool)
        if "row_index" in m.columns:
            row_idx = pd.to_numeric(m["row_index"], errors="coerce").dropna().astype(int)
            valid = row_idx[(row_idx >= 0) & (row_idx < len(self.df))]
            if len(valid) > 0:
                self.df.loc[valid.unique(), "ignore_frame"] = True
                applied[valid.unique()] = True

        # Fallback: match by absolute_time for rows not yet covered by row index.
        if "absolute_time" in m.columns:
            ts = pd.to_datetime(m["absolute_time"], errors="coerce").dropna()
            if len(ts) > 0:
                ts_ns = ts.astype("int64").to_numpy()
                df_ns = self.df["absolute_time"].astype("int64").to_numpy()
                hit = np.isin(df_ns, ts_ns)
                self.df.loc[hit, "ignore_frame"] = True

    def _refresh_visible_index(self) -> None:
        self.visible_df = self.df.loc[~self.df["ignore_frame"]].copy()
        self.visible_df = self.visible_df.reset_index().rename(columns={"index": "_orig_idx"})
        self.times_ns_visible = self.visible_df["absolute_time"].astype("int64").to_numpy()

    def _nearest_from(self, when: datetime, use_visible: bool) -> tuple[Optional[pd.Series], float]:
        if use_visible:
            source = self.visible_df
            times_ns = self.times_ns_visible
        else:
            source = self.df.reset_index().rename(columns={"index": "_orig_idx"})
            times_ns = source["absolute_time"].astype("int64").to_numpy()

        if source.empty:
            return None, float("inf")

        ts_ns = pd.Timestamp(when).value
        idx = int(np.searchsorted(times_ns, ts_ns))
        cand = []
        if 0 <= idx < len(times_ns):
            cand.append(idx)
        if idx - 1 >= 0:
            cand.append(idx - 1)
        if not cand:
            return None, float("inf")
        best = min(cand, key=lambda i: abs(times_ns[i] - ts_ns))
        delta_sec = abs(times_ns[best] - ts_ns) / 1e9
        return source.iloc[best], float(delta_sec)

    def nearest(self, when: datetime) -> tuple[Optional[pd.Series], float]:
        # For display/analysis, prioritize non-ignored trajectory rows.
        row, dt = self._nearest_from(when, use_visible=True)
        if row is not None:
            return row, dt
        return self._nearest_from(when, use_visible=False)

    def nearest_raw(self, when: datetime) -> tuple[Optional[pd.Series], float]:
        return self._nearest_from(when, use_visible=False)

    def ignore_state_at_time(self, when: datetime) -> tuple[bool, Optional[int]]:
        row, _ = self.nearest_raw(when)
        if row is None:
            return False, None
        idx = int(row["_orig_idx"])
        return bool(self.df.at[idx, "ignore_frame"]), idx

    def set_ignore_at_time(self, when: datetime, ignore: bool) -> bool:
        row, _ = self._nearest_from(when, use_visible=False)
        if row is None:
            return False
        idx = int(row["_orig_idx"])
        self.df.at[idx, "ignore_frame"] = bool(ignore)
        self._dirty = True
        self._refresh_visible_index()
        return True

    def toggle_ignore_at_time(self, when: datetime) -> tuple[bool, bool]:
        row, _ = self._nearest_from(when, use_visible=False)
        if row is None:
            return False, False
        idx = int(row["_orig_idx"])
        new_state = not bool(self.df.at[idx, "ignore_frame"])
        self.df.at[idx, "ignore_frame"] = new_state
        self._dirty = True
        self._refresh_visible_index()
        return True, new_state

    def save_if_dirty(self) -> None:
        if not self._dirty:
            return
        masked = self.df.loc[self.df["ignore_frame"], ["absolute_time"]].copy()
        masked = masked.reset_index().rename(columns={"index": "row_index"})
        if masked.empty:
            if self.mask_path.exists():
                self.mask_path.unlink()
        else:
            masked["absolute_time"] = pd.to_datetime(masked["absolute_time"]).dt.strftime("%Y-%m-%d %H:%M:%S.%f")
            masked.to_csv(self.mask_path, index=False)
        self._dirty = False

    def tail(self, when: datetime, seconds: float = 120.0) -> pd.DataFrame:
        t0 = pd.Timestamp(when) - pd.to_timedelta(seconds, unit="s")
        t1 = pd.Timestamp(when)
        return self.visible_df[(self.visible_df["absolute_time"] >= t0) & (self.visible_df["absolute_time"] <= t1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review synchronized 4-camera predictions with fused trajectory overlay")
    parser.add_argument("--monkey", required=True, help="Monkey ID, e.g. H1705011")
    parser.add_argument("--date", required=True, help="Session date, e.g. 2026-06-26")
    parser.add_argument("--frames-root", type=Path, default=PROJECT_ROOT / "tmp" / "rtdetr_frames_2hz", help="Root for extracted frames")
    parser.add_argument("--pred-root", type=Path, default=PROJECT_ROOT / "results" / "predictions", help="Root for prediction CSVs")
    parser.add_argument("--traj-root", type=Path, default=PROJECT_ROOT / "results" / "oft_analysis", help="Root for fused trajectories")
    parser.add_argument("--bundle-index", type=int, default=0, help="Which chronological 4-camera bundle to open")
    parser.add_argument("--fps", type=float, default=FPS_DEFAULT, help="Video FPS for absolute-time reconstruction")
    return parser.parse_args()


def parse_frame_index(path: Path) -> Optional[int]:
    m = FRAME_RE.search(path.stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def parse_timestamp_from_name(name: str) -> Optional[datetime]:
    m = TS_RE.search(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def truthy(v) -> bool:
    if v is None or pd.isna(v):
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


def load_prediction_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "video_frame" not in df.columns:
        raise ValueError(f"{csv_path}: missing video_frame")
    df["video_frame"] = pd.to_numeric(df["video_frame"], errors="coerce")
    df = df.dropna(subset=["video_frame"]).copy()
    df["video_frame"] = df["video_frame"].astype(int)
    return df


def load_frame_map(frames_dir: Path) -> dict[int, Path]:
    fmap: dict[int, Path] = {}
    for p in sorted(frames_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        idx = parse_frame_index(p)
        if idx is None:
            continue
        fmap[idx] = p
    return fmap


def build_camera_entries(camera: str, pred_dir: Path, frames_dir: Path, fps: float) -> list[CameraEntry]:
    entries: list[CameraEntry] = []
    for csv_path in sorted(pred_dir.glob(f"{camera}_*_preds.csv")):
        stem = csv_path.name.replace("_preds.csv", "")
        fdir = frames_dir / stem
        if not fdir.exists():
            continue
        df = load_prediction_df(csv_path)
        fmap = load_frame_map(fdir)
        if not fmap:
            continue
        start_dt = parse_timestamp_from_name(stem)
        frame_indices = np.array(sorted(fmap.keys()), dtype=np.int64)
        if start_dt is not None and frame_indices.size > 0:
            frame_times_sec = start_dt.timestamp() + frame_indices.astype(np.float64) / float(fps)
            end_dt = datetime.fromtimestamp(float(frame_times_sec[-1]))
        else:
            frame_times_sec = np.array([], dtype=np.float64)
            end_dt = None
        entries.append(
            CameraEntry(
                camera=camera,
                stem=stem,
                pred_csv=csv_path,
                frames_dir=fdir,
                pred_df=df,
                frame_map=fmap,
                start_dt=start_dt,
                end_dt=end_dt,
                frame_indices=frame_indices,
                frame_times_sec=frame_times_sec,
            )
        )
    return entries


def build_bundles(pred_dir: Path, frames_dir: Path, fps: float) -> list[Bundle]:
    per_cam: dict[str, list[CameraEntry]] = {}
    for cam in CAMERAS:
        entries = build_camera_entries(cam, pred_dir=pred_dir, frames_dir=frames_dir, fps=fps)
        if not entries:
            raise RuntimeError(f"No usable prediction+frame entries for {cam} under {pred_dir} and {frames_dir}")
        per_cam[cam] = entries

    reference_entries = per_cam[CAMERAS[0]]
    bundles: list[Bundle] = []
    for i, ref_entry in enumerate(reference_entries):
        if ref_entry.start_dt is None or ref_entry.end_dt is None:
            continue

        anchor_start = ref_entry.start_dt.timestamp()
        anchor_end = ref_entry.end_dt.timestamp()
        entries: dict[str, list[CameraEntry]] = {}
        bundle_start = anchor_start
        bundle_end = anchor_end

        valid_map: dict[str, list[CameraEntry]] = {}
        selected_map: dict[str, set[int]] = {}

        for cam in CAMERAS:
            valid_entries = [
                e for e in per_cam[cam]
                if e.start_dt is not None and e.end_dt is not None and e.frame_times_sec.size > 0
            ]
            valid_map[cam] = valid_entries

            overlap_idx: list[int] = []
            for idx, entry in enumerate(valid_entries):
                entry_start = entry.start_dt.timestamp()
                entry_end = entry.end_dt.timestamp()
                if entry_end < anchor_start or entry_start > anchor_end:
                    continue
                overlap_idx.append(idx)

            if overlap_idx:
                selected_idx = set(overlap_idx)
                # Include immediate neighbors to avoid no-frame drops right after clip boundaries.
                lo = min(overlap_idx)
                hi = max(overlap_idx)
                if lo - 1 >= 0:
                    selected_idx.add(lo - 1)
                if hi + 1 < len(valid_entries):
                    selected_idx.add(hi + 1)
            else:
                # Fall back to the nearest clip for display continuity if nothing overlaps.
                nearest_idx = min(
                    range(len(valid_entries)),
                    key=lambda idx: min(
                        abs(valid_entries[idx].start_dt.timestamp() - anchor_start),
                        abs(valid_entries[idx].end_dt.timestamp() - anchor_end),
                    ),
                    default=None,
                )
                selected_idx = {nearest_idx} if nearest_idx is not None else set()

            if not selected_idx:
                entries = {}
                break

            selected_map[cam] = selected_idx
            for idx in sorted(selected_idx):
                entry = valid_entries[idx]
                bundle_start = min(bundle_start, entry.start_dt.timestamp())
                bundle_end = max(bundle_end, entry.end_dt.timestamp())

        if not selected_map or len(selected_map) != len(CAMERAS):
            continue

        # Second pass: expand coverage globally until no camera needs more clips.
        changed_any = True
        while changed_any:
            changed_any = False
            for cam in CAMERAS:
                valid_entries = valid_map[cam]
                selected_idx = selected_map[cam]
                if not selected_idx:
                    entries = {}
                    break

                lo = min(selected_idx)
                hi = max(selected_idx)
                cam_start = valid_entries[lo].start_dt.timestamp()
                cam_end = valid_entries[hi].end_dt.timestamp()

                if cam_start > bundle_start and lo - 1 >= 0:
                    selected_idx.add(lo - 1)
                    e = valid_entries[lo - 1]
                    bundle_start = min(bundle_start, e.start_dt.timestamp())
                    bundle_end = max(bundle_end, e.end_dt.timestamp())
                    changed_any = True

                lo = min(selected_idx)
                hi = max(selected_idx)
                cam_start = valid_entries[lo].start_dt.timestamp()
                cam_end = valid_entries[hi].end_dt.timestamp()
                if cam_end < bundle_end and hi + 1 < len(valid_entries):
                    selected_idx.add(hi + 1)
                    e = valid_entries[hi + 1]
                    bundle_start = min(bundle_start, e.start_dt.timestamp())
                    bundle_end = max(bundle_end, e.end_dt.timestamp())
                    changed_any = True

                # Ensure immediate handoff continuity at camera clip boundary.
                hi = max(selected_idx)
                if hi + 1 < len(valid_entries):
                    curr_end = valid_entries[hi].end_dt.timestamp()
                    next_start = valid_entries[hi + 1].start_dt.timestamp()
                    handoff_tol = max(FRAME_TIME_TOLERANCE_SEC, 1.0 / float(fps))
                    if next_start <= (curr_end + handoff_tol) and (hi + 1) not in selected_idx:
                        selected_idx.add(hi + 1)
                        e = valid_entries[hi + 1]
                        bundle_start = min(bundle_start, e.start_dt.timestamp())
                        bundle_end = max(bundle_end, e.end_dt.timestamp())
                        changed_any = True

        for cam in CAMERAS:
            valid_entries = valid_map[cam]
            selected_idx = selected_map[cam]
            entries[cam] = [valid_entries[idx] for idx in sorted(selected_idx)]

        if not entries or len(entries) != len(CAMERAS):
            continue

        if not entries or bundle_end < bundle_start:
            continue

        timeline_sources = []
        for cam_entries in entries.values():
            for entry in cam_entries:
                if entry.frame_times_sec.size > 0:
                    timeline_sources.append(np.rint(entry.frame_times_sec * 1e9).astype(np.int64))
        if not timeline_sources:
            continue
        timeline_ns = np.unique(np.concatenate(timeline_sources))
        if timeline_ns.size == 0:
            continue
        timeline = [datetime.fromtimestamp(float(ns) / 1e9) for ns in timeline_ns]
        bundles.append(
            Bundle(
                index=i,
                entries=entries,
                timeline=timeline,
            )
        )

    if not bundles:
        raise RuntimeError("No synchronized bundles found with overlapping time ranges across D01-D04")
    return bundles


def nearest_frame_for_time(bundle: Bundle, camera: str, when: datetime) -> tuple[Optional[CameraEntry], Optional[int], float]:
    t = when.timestamp()
    best_entry = None
    best_frame = None
    best_dt = float("inf")

    for entry in bundle.entries[camera]:
        times = entry.frame_times_sec
        idxs = entry.frame_indices
        if len(times) == 0:
            continue
        t0 = float(times[0])
        t1 = float(times[-1])
        if t < (t0 - FRAME_TIME_TOLERANCE_SEC) or t > (t1 + FRAME_TIME_TOLERANCE_SEC):
            continue
        pos = int(np.searchsorted(times, t))
        cand = []
        if 0 <= pos < len(times):
            cand.append(pos)
        if pos - 1 >= 0:
            cand.append(pos - 1)
        if not cand:
            continue
        best_local = min(cand, key=lambda i: abs(times[i] - t))
        dt = float(abs(times[best_local] - t))
        if dt < best_dt:
            best_entry = entry
            best_frame = int(idxs[best_local])
            best_dt = dt

    return best_entry, best_frame, best_dt


def get_pred_row(entry: CameraEntry, frame_idx: int) -> Optional[pd.Series]:
    rows = entry.pred_df[entry.pred_df["video_frame"] == int(frame_idx)]
    if rows.empty:
        return None
    return rows.iloc[0]


def apply_prediction_edit(entry: CameraEntry, frame_idx: int, mode: str) -> bool:
    if mode != "no_prediction":
        return False

    mask = entry.pred_df["video_frame"] == int(frame_idx)
    had_rows = bool(mask.any())

    if had_rows:
        entry.pred_df = entry.pred_df.loc[~mask].reset_index(drop=True)
        entry.pred_df.to_csv(entry.pred_csv, index=False)
        return True
    return False


def row_to_box(row: Optional[pd.Series]) -> tuple[Optional[tuple[int, int, int, int]], bool]:
    if row is None:
        return None, False

    is_det = False
    if "has_detection" in row.index:
        is_det = truthy(row.get("has_detection"))
    elif "monkey_visible" in row.index:
        is_det = truthy(row.get("monkey_visible"))

    vals = [row.get("bbox_x1"), row.get("bbox_y1"), row.get("bbox_x2"), row.get("bbox_y2")]
    if not all(pd.notna(v) for v in vals):
        return None, is_det

    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in vals]
    except Exception:
        return None, is_det

    return (x1, y1, x2, y2), is_det


def draw_arena_overlay(panel: np.ndarray, x_cm: Optional[float], y_cm: Optional[float], tail: Optional[pd.DataFrame]) -> None:
    h, w = panel.shape[:2]
    # Keep footer space for status/help text.
    size = min(w - 36, h - 190)
    x0 = (w - size) // 2
    y0 = 70

    cv2.rectangle(panel, (x0, y0), (x0 + size, y0 + size), (245, 245, 245), -1)
    cv2.rectangle(panel, (x0, y0), (x0 + size, y0 + size), (30, 30, 30), 2)

    # Quadrants
    cv2.line(panel, (x0 + size // 2, y0), (x0 + size // 2, y0 + size), (210, 210, 210), 1)
    cv2.line(panel, (x0, y0 + size // 2), (x0 + size, y0 + size // 2), (210, 210, 210), 1)

    def cm_to_px(xc: float, yc: float) -> tuple[int, int]:
        xx = int(round(x0 + (xc / ARENA_SIZE_CM) * size))
        yy = int(round(y0 + (yc / ARENA_SIZE_CM) * size))
        return xx, yy

    if tail is not None and not tail.empty:
        pts = []
        for _, r in tail.iterrows():
            if pd.isna(r.get("x_cm")) or pd.isna(r.get("y_cm")):
                continue
            pts.append(cm_to_px(float(r["x_cm"]), float(r["y_cm"])))
        for i in range(1, len(pts)):
            cv2.line(panel, pts[i - 1], pts[i], (140, 140, 140), 1)

    if x_cm is not None and y_cm is not None:
        cx, cy = cm_to_px(float(x_cm), float(y_cm))
        cv2.circle(panel, (cx, cy), 5, C_BLUE, -1)
        cv2.circle(panel, (cx, cy), 8, (255, 255, 255), 1)

    cv2.putText(panel, "Arena (fused)", (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_BLACK, 1, cv2.LINE_AA)


def make_sidebar(height: int, x_cm: Optional[float], y_cm: Optional[float], tail: Optional[pd.DataFrame],
                 abs_time: datetime, dt_sec: float, bundle_idx: int, time_idx: int, show_bbox: bool,
                 active_cameras: set[str], trajectory_mask_mode: bool,
                 velocity_cm_s: Optional[float], status_text: str,
                 raw_ignored: bool, raw_idx: Optional[int]) -> np.ndarray:
    panel = np.full((height, SIDEBAR_W, 3), 236, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (SIDEBAR_W - 1, height - 1), (60, 60, 60), 1)
    cv2.rectangle(panel, (0, 0), (SIDEBAR_W, 56), (25, 25, 25), -1)
    cv2.putText(panel, "Trajectory Review", (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, C_WHITE, 1, cv2.LINE_AA)
    cv2.putText(panel, f"bundle={bundle_idx}  t_idx={time_idx}", (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_GRAY, 1, cv2.LINE_AA)
    draw_arena_overlay(panel, x_cm, y_cm, tail)

    y = height - 128
    cv2.putText(panel, abs_time.strftime("%Y-%m-%d %H:%M:%S"), (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_BLACK, 1, cv2.LINE_AA)
    dt_text = f"traj_dt={dt_sec:.2f}s" if np.isfinite(dt_sec) else "traj_dt=nan"
    cv2.putText(panel, dt_text, (14, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_BLACK, 1, cv2.LINE_AA)
    vel_text = f"velocity={velocity_cm_s:.2f} cm/s" if velocity_cm_s is not None and np.isfinite(velocity_cm_s) else "velocity=n/a"
    cv2.putText(panel, vel_text, (14, y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_BLACK, 1, cv2.LINE_AA)
    cv2.putText(panel, f"bbox overlays={'ON' if show_bbox else 'OFF'}", (14, y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_BLACK, 1, cv2.LINE_AA)

    edit_label = "none"
    if trajectory_mask_mode:
        edit_label = "trajectory-mask"
    elif active_cameras:
        edit_label = ",".join(sorted(active_cameras))
    cv2.putText(panel, f"active edit: {edit_label}", (14, y + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_BLACK, 1, cv2.LINE_AA)

    state_text = "nearest traj row: n/a"
    state_color = (70, 70, 70)
    if raw_idx is not None:
        state_text = f"nearest traj row #{raw_idx}: {'IGNORED' if raw_ignored else 'ACTIVE'}"
        state_color = (40, 70, 170) if raw_ignored else (30, 130, 40)
    cv2.putText(panel, state_text, (14, y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.50, state_color, 1, cv2.LINE_AA)

    if status_text:
        cv2.rectangle(panel, (10, height - 36), (SIDEBAR_W - 10, height - 10), (220, 235, 248), -1)
        cv2.rectangle(panel, (10, height - 36), (SIDEBAR_W - 10, height - 10), (150, 180, 205), 1)
        cv2.putText(panel, status_text[:54], (14, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 70, 140), 1, cv2.LINE_AA)
    return panel


def resize_with_padding(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    if ih == 0 or iw == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    scale = min(target_w / iw, target_h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - nw) // 2
    y0 = (target_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def render_panel(entry: CameraEntry, frame_idx: Optional[int], show_bbox: bool, delta_t: Optional[float] = None,
                 is_active: bool = False) -> np.ndarray:
    image_h = PANEL_H - PANEL_INFO_H

    if frame_idx is None:
        panel = np.full((PANEL_H, PANEL_W, 3), 18, dtype=np.uint8)
        cv2.putText(panel, f"{entry.camera}: no frame at this time", (20, image_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_RED, 2, cv2.LINE_AA)
        return panel

    path = entry.frame_map.get(frame_idx)
    if path is None:
        panel = np.full((PANEL_H, PANEL_W, 3), 18, dtype=np.uint8)
        cv2.putText(panel, f"{entry.camera}: missing frame {frame_idx}", (20, image_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_RED, 2, cv2.LINE_AA)
        return panel

    img = cv2.imread(str(path))
    if img is None:
        panel = np.full((PANEL_H, PANEL_W, 3), 18, dtype=np.uint8)
        cv2.putText(panel, f"{entry.camera}: unreadable frame", (20, image_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_RED, 2, cv2.LINE_AA)
        return panel

    row = get_pred_row(entry, frame_idx)
    box, is_det = row_to_box(row)

    if show_bbox:
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), C_GREEN if is_det else C_YELLOW, 3)
        elif is_det:
            cv2.putText(img, "DETECTION FLAG w/o BOX", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_YELLOW, 2, cv2.LINE_AA)

    image_panel = resize_with_padding(img, PANEL_W, image_h)
    panel = np.full((PANEL_H, PANEL_W, 3), 18, dtype=np.uint8)
    panel[:image_h, :] = image_panel
    cv2.rectangle(panel, (0, image_h), (PANEL_W, PANEL_H), C_BLACK, -1)
    if is_active:
        cv2.rectangle(panel, (0, 0), (PANEL_W - 1, PANEL_H - 1), (0, 180, 255), 4)
    status = "det" if is_det else "no-det"
    dt_txt = f"  dt={delta_t:.3f}s" if delta_t is not None and np.isfinite(delta_t) else ""
    cv2.putText(panel, f"{entry.camera}  {entry.stem}", (12, image_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WHITE, 1, cv2.LINE_AA)
    cv2.putText(panel, f"frame={frame_idx}  {status}{dt_txt}", (12, image_h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_GRAY, 1, cv2.LINE_AA)
    return panel


def resolve_time_context(bundle: Bundle, abs_time: datetime) -> dict[str, tuple[CameraEntry, Optional[int], float]]:
    context: dict[str, tuple[CameraEntry, Optional[int], float]] = {}
    for c in CAMERAS:
        entry, frame_idx, dt = nearest_frame_for_time(bundle, c, abs_time)
        if entry is None:
            entry = bundle.entries[c][0]
        context[c] = (entry, frame_idx, dt)
    return context


def make_canvas(bundle: Bundle, time_idx: int, show_bbox: bool, traj_index: TrajectoryIndex, show_tail: bool,
                active_cameras: set[str], trajectory_mask_mode: bool,
                status_text: str) -> tuple[np.ndarray, datetime, dict[str, tuple[CameraEntry, Optional[int], float]]]:
    abs_time = bundle.timeline[time_idx]
    context = resolve_time_context(bundle, abs_time)

    panels = []
    for i, c in enumerate(CAMERAS, start=1):
        entry, frame_idx, dt = context[c]
        panels.append(render_panel(entry, frame_idx, show_bbox, delta_t=dt, is_active=(c in active_cameras)))

    top = np.hstack([panels[0], panels[1]])
    bot = np.hstack([panels[2], panels[3]])
    mosaic = np.vstack([top, bot])

    traj_row = None
    dt_sec = float("inf")
    tail = None
    traj_row, dt_sec = traj_index.nearest(abs_time)
    raw_ignored, raw_idx = traj_index.ignore_state_at_time(abs_time)
    if show_tail:
        tail = traj_index.tail(abs_time, seconds=120.0)

    x_cm = float(traj_row["x_cm"]) if traj_row is not None else None
    y_cm = float(traj_row["y_cm"]) if traj_row is not None else None
    velocity_cm_s = None
    if traj_row is not None and "velocity_cm_s" in traj_row.index and pd.notna(traj_row.get("velocity_cm_s")):
        velocity_cm_s = float(traj_row["velocity_cm_s"])
    sidebar = make_sidebar(
        height=mosaic.shape[0],
        x_cm=x_cm,
        y_cm=y_cm,
        tail=tail,
        abs_time=abs_time,
        dt_sec=dt_sec,
        bundle_idx=bundle.index,
        time_idx=time_idx,
        show_bbox=show_bbox,
        active_cameras=active_cameras,
        trajectory_mask_mode=trajectory_mask_mode,
        velocity_cm_s=velocity_cm_s,
        status_text=status_text,
        raw_ignored=raw_ignored,
        raw_idx=raw_idx,
    )
    canvas = np.hstack([mosaic, sidebar])
    return canvas, abs_time, context


def run(monkey: str, date: str, frames_root: Path, pred_root: Path, traj_root: Path, bundle_index: int, fps: float) -> None:
    frames_dir = frames_root / monkey / date
    pred_dir = pred_root / monkey / date
    traj_path = traj_root / monkey / date / "trajectory_realworld.csv"

    if not frames_dir.exists():
        raise SystemExit(f"ERROR: frames dir not found: {frames_dir}")
    if not pred_dir.exists():
        raise SystemExit(f"ERROR: prediction dir not found: {pred_dir}")
    if not traj_path.exists():
        raise SystemExit(f"ERROR: trajectory file not found: {traj_path}")

    bundles = build_bundles(pred_dir=pred_dir, frames_dir=frames_dir, fps=fps)
    if bundle_index < 0 or bundle_index >= len(bundles):
        raise SystemExit(f"ERROR: bundle-index out of range [0, {len(bundles)-1}]")

    traj_index = TrajectoryIndex(traj_path)

    show_bbox = True
    show_tail = True
    bidx = bundle_index
    tidx = 0
    active_cameras: set[str] = set()
    trajectory_mask_mode = False
    status_text = ""

    win = "MultiCamPredictionReview"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print("\nControls: N/P next-prev frame, J/K jump -/+10, B/M prev-next bundle, H bbox toggle, T tail toggle, Q quit")
    print("Edit: 1/2/3/4 toggle D01..D04 (multi-select), 5 toggle trajectory mask, 0 clear all, I toggle mask (mode 5), U no-prediction (all selected cameras)")

    while True:
        bundle = bundles[bidx]

        canvas, abs_time, context = make_canvas(
            bundle,
            tidx,
            show_bbox,
            traj_index,
            show_tail,
            active_cameras,
            trajectory_mask_mode,
            status_text,
        )
        cv2.imshow(win, canvas)
        key = cv2.waitKey(30) & 0xFF
        status_text = ""

        if key in (ord("q"), 27):
            break
        elif key == ord("0"):
            active_cameras.clear()
            trajectory_mask_mode = False
            status_text = "active edit cleared"
        elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
            cam = CAMERAS[int(chr(key)) - 1]
            trajectory_mask_mode = False
            if cam in active_cameras:
                active_cameras.remove(cam)
                status_text = f"{cam} deselected"
            else:
                active_cameras.add(cam)
                status_text = f"{cam} selected"
        elif key == ord("5"):
            trajectory_mask_mode = not trajectory_mask_mode
            if trajectory_mask_mode:
                active_cameras.clear()
                status_text = "active edit set to trajectory mask"
            else:
                status_text = "trajectory mask mode off"
        elif key in (ord("n"), 83):
            tidx = min(len(bundle.timeline) - 1, tidx + 1)
        elif key in (ord("p"), 81):
            tidx = max(0, tidx - 1)
        elif key == ord("k"):
            tidx = min(len(bundle.timeline) - 1, tidx + 10)
        elif key == ord("j"):
            tidx = max(0, tidx - 10)
        elif key in (ord("i"), ord("I")):
            if trajectory_mask_mode:
                ok, now_ignored = traj_index.toggle_ignore_at_time(abs_time)
                if ok:
                    status_text = "trajectory frame masked (pending save)" if now_ignored else "trajectory frame unmasked (pending save)"
                else:
                    status_text = "trajectory mask toggle failed"
                tidx = min(len(bundle.timeline) - 1, tidx + 1)
            else:
                status_text = "select 5 to use trajectory mask"
        elif key == ord("u"):
            if active_cameras:
                results = []
                for cam in sorted(active_cameras):
                    entry, frame_idx, _ = context[cam]
                    if frame_idx is None:
                        results.append(f"{cam}:no-frame")
                        continue
                    ok = apply_prediction_edit(entry, frame_idx, mode="no_prediction")
                    if ok:
                        results.append(f"{cam}:ok")
                    else:
                        results.append(f"{cam}:none")
                status_text = "U -> " + ", ".join(results)
                tidx = min(len(bundle.timeline) - 1, tidx + 1)
            elif trajectory_mask_mode:
                status_text = "U is reserved for camera no-prediction"
            else:
                status_text = "toggle 1-4 before using U"
        elif key == ord("h"):
            show_bbox = not show_bbox
        elif key == ord("t"):
            show_tail = not show_tail
        elif key == ord("m"):
            if bidx < len(bundles) - 1:
                bidx += 1
                tidx = 0
        elif key == ord("b"):
            if bidx > 0:
                bidx -= 1
                tidx = 0

    traj_index.save_if_dirty()
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise SystemExit("ERROR: --fps must be > 0")
    run(
        monkey=args.monkey,
        date=args.date,
        frames_root=args.frames_root.expanduser().resolve(),
        pred_root=args.pred_root.expanduser().resolve(),
        traj_root=args.traj_root.expanduser().resolve(),
        bundle_index=args.bundle_index,
        fps=float(args.fps),
    )


if __name__ == "__main__":
    main()
