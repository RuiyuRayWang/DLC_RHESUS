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
        - Mouse drag (single active camera): draw/adjust bbox in selected camera panel
  - H: hide/show bbox overlays
  - T: hide/show trajectory tail on arena
    - A: toggle artifact-focus mode (artifact candidate frames only)
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
ARTIFACT_FILENAME = "trajectory_artifact_candidates.csv"
FRAME_TIME_TOLERANCE_SEC = 0.08
ARTIFACT_VEL_THRESH_DEFAULT = 300.0
ARTIFACT_DVDT_THRESH_DEFAULT = 1500.0
ARTIFACT_BRIDGE_MAX_SEC_DEFAULT = 8.0

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


def sidebar_time_input_rect(height: int) -> tuple[int, int, int, int]:
    x0 = 14
    x1 = SIDEBAR_W - 14
    y0 = height - 208
    y1 = height - 182
    return x0, y0, x1, y1


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


@dataclass
class PanelMeta:
    panel_x0: int
    panel_y0: int
    image_h: int
    img_x0: int
    img_y0: int
    img_w: int
    img_h: int
    scale: float
    src_w: int
    src_h: int
    has_image: bool


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

    def artifact_row_indices(self, velocity_thresh: float, dvdt_thresh: float, bridge_max_sec: float = ARTIFACT_BRIDGE_MAX_SEC_DEFAULT) -> set[int]:
        if self.df.empty:
            return set()

        ts = self.df["absolute_time"].astype("int64").to_numpy(dtype=np.int64) / 1e9
        dt = np.diff(ts, prepend=ts[0])

        if "velocity_cm_s" in self.df.columns:
            vel = pd.to_numeric(self.df["velocity_cm_s"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        elif "x_cm" in self.df.columns and "y_cm" in self.df.columns:
            dx = pd.to_numeric(self.df["x_cm"], errors="coerce").diff().to_numpy(dtype=float)
            dy = pd.to_numeric(self.df["y_cm"], errors="coerce").diff().to_numpy(dtype=float)
            dist = np.sqrt(np.nan_to_num(dx, nan=0.0) ** 2 + np.nan_to_num(dy, nan=0.0) ** 2)
            vel = np.divide(dist, dt, out=np.zeros_like(dist), where=np.abs(dt) > 1e-9)
        else:
            vel = np.zeros(len(self.df), dtype=float)

        dvel = np.diff(vel, prepend=vel[0])
        dvdt = np.divide(np.abs(dvel), dt, out=np.zeros_like(dvel), where=np.abs(dt) > 1e-9)
        seed = (np.abs(vel) > float(velocity_thresh)) | (dvdt > float(dvdt_thresh))
        artifact_mask = seed.copy()
        seed_idx = np.flatnonzero(seed)
        if len(seed_idx) >= 2:
            for i in range(len(seed_idx) - 1):
                a = int(seed_idx[i])
                b = int(seed_idx[i + 1])
                if b <= a:
                    continue
                gap_sec = float(ts[b] - ts[a])
                if 0.0 < gap_sec <= float(bridge_max_sec):
                    artifact_mask[a:b + 1] = True
        idx = np.flatnonzero(artifact_mask)
        return set(int(i) for i in idx)


def build_artifact_tidx_map(
    bundles: list[Bundle],
    traj_index: TrajectoryIndex,
    velocity_thresh: float,
    dvdt_thresh: float,
    bridge_max_sec: float,
    artifact_csv_path: Optional[Path] = None,
) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}

    if artifact_csv_path is not None and artifact_csv_path.exists():
        try:
            art = pd.read_csv(artifact_csv_path)
        except Exception:
            art = pd.DataFrame()

        if not art.empty and 'absolute_time' in art.columns:
            ats = pd.to_datetime(art['absolute_time'], errors='coerce').dropna()
            if len(ats) > 0:
                art_ns = np.unique(ats.astype('int64').to_numpy())
                for bidx, bundle in enumerate(bundles):
                    t_ns = np.array([pd.Timestamp(t).value for t in bundle.timeline], dtype=np.int64)
                    if t_ns.size == 0:
                        continue
                    tids: list[int] = []
                    for ns in art_ns:
                        pos = int(np.searchsorted(t_ns, ns))
                        cand = []
                        if 0 <= pos < len(t_ns):
                            cand.append(pos)
                        if pos - 1 >= 0:
                            cand.append(pos - 1)
                        if not cand:
                            continue
                        idx = min(cand, key=lambda i: abs(t_ns[i] - ns))
                        if abs(t_ns[idx] - ns) <= int(FRAME_TIME_TOLERANCE_SEC * 1e9):
                            tids.append(int(idx))
                    if tids:
                        out[bidx] = sorted(set(tids))
                return out

    # Fallback for backward compatibility when precomputed artifact CSV is unavailable.
    artifact_rows = traj_index.artifact_row_indices(
        velocity_thresh=velocity_thresh,
        dvdt_thresh=dvdt_thresh,
        bridge_max_sec=bridge_max_sec,
    )
    if not artifact_rows:
        return out

    for bidx, bundle in enumerate(bundles):
        tids: list[int] = []
        for tidx, when in enumerate(bundle.timeline):
            row, _ = traj_index.nearest_raw(when)
            if row is None:
                continue
            ridx = int(row.get("_orig_idx", -1))
            if ridx in artifact_rows:
                tids.append(tidx)
        if tids:
            out[bidx] = sorted(set(tids))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review synchronized 4-camera predictions with fused trajectory overlay")
    parser.add_argument("--monkey", required=True, help="Monkey ID, e.g. H1705011")
    parser.add_argument("--date", required=True, help="Session date, e.g. 2026-06-26")
    parser.add_argument("--frames-root", type=Path, default=PROJECT_ROOT / "tmp" / "rtdetr_frames_2hz", help="Root for extracted frames")
    parser.add_argument("--pred-root", type=Path, default=PROJECT_ROOT / "results" / "predictions", help="Root for prediction CSVs")
    parser.add_argument("--traj-root", type=Path, default=PROJECT_ROOT / "results" / "oft_analysis", help="Root for fused trajectories")
    parser.add_argument("--bundle-index", type=int, default=0, help="Which chronological 4-camera bundle to open")
    parser.add_argument("--fps", type=float, default=FPS_DEFAULT, help="Video FPS for absolute-time reconstruction")
    parser.add_argument("--artifact-velocity-thresh", type=float, default=ARTIFACT_VEL_THRESH_DEFAULT, help="Velocity threshold (cm/s) for artifact candidates")
    parser.add_argument("--artifact-dvdt-thresh", type=float, default=ARTIFACT_DVDT_THRESH_DEFAULT, help="|dV/dt| threshold (cm/s^2) for artifact candidates")
    parser.add_argument("--artifact-bridge-max-sec", type=float, default=ARTIFACT_BRIDGE_MAX_SEC_DEFAULT, help="Bridge adjacent artifact edges to include sustained artifact intervals")
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


def matching_frames_for_time(bundle: Bundle, camera: str, when: datetime) -> list[tuple[CameraEntry, int, float]]:
    """Return per-clip nearest frame matches around an absolute time.

    At clip boundaries, two clips can both be valid within tolerance; callers can
    apply edits to all returned matches to avoid boundary leftovers.
    """
    t = when.timestamp()
    matches: list[tuple[CameraEntry, int, float]] = []

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
        if dt <= FRAME_TIME_TOLERANCE_SEC:
            matches.append((entry, int(idxs[best_local]), dt))

    matches.sort(key=lambda x: x[2])
    return matches


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
        if "has_detection" not in entry.pred_df.columns:
            entry.pred_df["has_detection"] = False
        entry.pred_df.loc[mask, "has_detection"] = False
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
                 time_input_text: str, time_input_active: bool,
                 artifact_mode: bool, artifact_pos: int, artifact_total: int,
                 velocity_cm_s: Optional[float], status_text: str,
                 raw_ignored: bool, raw_idx: Optional[int]) -> np.ndarray:
    panel = np.full((height, SIDEBAR_W, 3), 236, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (SIDEBAR_W - 1, height - 1), (60, 60, 60), 1)
    cv2.rectangle(panel, (0, 0), (SIDEBAR_W, 56), (25, 25, 25), -1)
    cv2.putText(panel, "Trajectory Review", (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, C_WHITE, 1, cv2.LINE_AA)
    cv2.putText(panel, f"bundle={bundle_idx}  t_idx={time_idx}", (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_GRAY, 1, cv2.LINE_AA)
    draw_arena_overlay(panel, x_cm, y_cm, tail)

    tx0, ty0, tx1, ty1 = sidebar_time_input_rect(height)
    cv2.putText(panel, "Jump to time (HH:MM:SS)", (tx0, ty0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (38, 38, 38), 1, cv2.LINE_AA)
    fill = (252, 252, 252) if not time_input_active else (232, 242, 255)
    border = (150, 150, 150) if not time_input_active else (80, 140, 200)
    cv2.rectangle(panel, (tx0, ty0), (tx1, ty1), fill, -1)
    cv2.rectangle(panel, (tx0, ty0), (tx1, ty1), border, 1)
    shown_text = time_input_text if time_input_text else "--:--:--"
    shown_color = (30, 30, 30) if time_input_text else (140, 140, 140)
    cv2.putText(panel, shown_text, (tx0 + 8, ty0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, shown_color, 1, cv2.LINE_AA)

    y = height - 128
    cv2.putText(panel, "Refresh trajectory (R)", (14, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (28, 68, 112), 1, cv2.LINE_AA)
    cv2.putText(panel, abs_time.strftime("%Y-%m-%d %H:%M:%S"), (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_BLACK, 1, cv2.LINE_AA)
    dt_text = f"traj_dt={dt_sec:.2f}s" if np.isfinite(dt_sec) else "traj_dt=nan"
    cv2.putText(panel, dt_text, (14, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_BLACK, 1, cv2.LINE_AA)
    vel_text = f"velocity={velocity_cm_s:.2f} cm/s" if velocity_cm_s is not None and np.isfinite(velocity_cm_s) else "velocity=n/a"
    cv2.putText(panel, vel_text, (14, y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_BLACK, 1, cv2.LINE_AA)
    cv2.putText(panel, f"bbox overlays={'ON' if show_bbox else 'OFF'}", (14, y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_BLACK, 1, cv2.LINE_AA)
    art_label = f"artifact focus={'ON' if artifact_mode else 'OFF'}"
    if artifact_mode:
        art_label += f" ({artifact_pos + 1}/{max(artifact_total, 1)})"
    cv2.putText(panel, art_label, (14, y + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (90, 60, 120) if artifact_mode else C_BLACK, 1, cv2.LINE_AA)

    edit_label = "none"
    if trajectory_mask_mode:
        edit_label = "trajectory-mask"
    elif active_cameras:
        edit_label = ",".join(sorted(active_cameras))
    cv2.putText(panel, f"active edit: {edit_label}", (14, y + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_BLACK, 1, cv2.LINE_AA)

    state_text = "nearest traj row: n/a"
    state_color = (70, 70, 70)
    if raw_idx is not None:
        state_text = f"nearest traj row #{raw_idx}: {'IGNORED' if raw_ignored else 'ACTIVE'}"
        state_color = (40, 70, 170) if raw_ignored else (30, 130, 40)
    cv2.putText(panel, state_text, (14, y + 128), cv2.FONT_HERSHEY_SIMPLEX, 0.50, state_color, 1, cv2.LINE_AA)

    if status_text:
        cv2.rectangle(panel, (10, height - 36), (SIDEBAR_W - 10, height - 10), (220, 235, 248), -1)
        cv2.rectangle(panel, (10, height - 36), (SIDEBAR_W - 10, height - 10), (150, 180, 205), 1)
        cv2.putText(panel, status_text[:54], (14, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 70, 140), 1, cv2.LINE_AA)
    return panel


def resize_with_padding(img: np.ndarray, target_w: int, target_h: int) -> tuple[np.ndarray, int, int, int, int, float]:
    ih, iw = img.shape[:2]
    if ih == 0 or iw == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8), 0, 0, 0, 0, 1.0
    scale = min(target_w / iw, target_h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - nw) // 2
    y0 = (target_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas, x0, y0, nw, nh, float(scale)


def render_panel(entry: CameraEntry, frame_idx: Optional[int], show_bbox: bool, delta_t: Optional[float] = None,
                 is_active: bool = False, panel_x0: int = 0, panel_y0: int = 0) -> tuple[np.ndarray, PanelMeta]:
    image_h = PANEL_H - PANEL_INFO_H
    empty_meta = PanelMeta(
        panel_x0=panel_x0,
        panel_y0=panel_y0,
        image_h=image_h,
        img_x0=0,
        img_y0=0,
        img_w=0,
        img_h=0,
        scale=1.0,
        src_w=0,
        src_h=0,
        has_image=False,
    )

    if frame_idx is None:
        panel = np.full((PANEL_H, PANEL_W, 3), 18, dtype=np.uint8)
        cv2.putText(panel, f"{entry.camera}: no frame at this time", (20, image_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_RED, 2, cv2.LINE_AA)
        return panel, empty_meta

    path = entry.frame_map.get(frame_idx)
    if path is None:
        panel = np.full((PANEL_H, PANEL_W, 3), 18, dtype=np.uint8)
        cv2.putText(panel, f"{entry.camera}: missing frame {frame_idx}", (20, image_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_RED, 2, cv2.LINE_AA)
        return panel, empty_meta

    img = cv2.imread(str(path))
    if img is None:
        panel = np.full((PANEL_H, PANEL_W, 3), 18, dtype=np.uint8)
        cv2.putText(panel, f"{entry.camera}: unreadable frame", (20, image_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_RED, 2, cv2.LINE_AA)
        return panel, empty_meta

    row = get_pred_row(entry, frame_idx)
    box, is_det = row_to_box(row)

    if show_bbox:
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), C_GREEN if is_det else C_YELLOW, 3)
        elif is_det:
            cv2.putText(img, "DETECTION FLAG w/o BOX", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_YELLOW, 2, cv2.LINE_AA)

    src_h, src_w = img.shape[:2]
    image_panel, img_x0, img_y0, img_w, img_h, scale = resize_with_padding(img, PANEL_W, image_h)
    panel = np.full((PANEL_H, PANEL_W, 3), 18, dtype=np.uint8)
    panel[:image_h, :] = image_panel
    cv2.rectangle(panel, (0, image_h), (PANEL_W, PANEL_H), C_BLACK, -1)
    if is_active:
        cv2.rectangle(panel, (0, 0), (PANEL_W - 1, PANEL_H - 1), (0, 180, 255), 4)
    status = "det" if is_det else "no-det"
    dt_txt = f"  dt={delta_t:.3f}s" if delta_t is not None and np.isfinite(delta_t) else ""
    cv2.putText(panel, f"{entry.camera}  {entry.stem}", (12, image_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WHITE, 1, cv2.LINE_AA)
    cv2.putText(panel, f"frame={frame_idx}  {status}{dt_txt}", (12, image_h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_GRAY, 1, cv2.LINE_AA)
    meta = PanelMeta(
        panel_x0=panel_x0,
        panel_y0=panel_y0,
        image_h=image_h,
        img_x0=img_x0,
        img_y0=img_y0,
        img_w=img_w,
        img_h=img_h,
        scale=scale,
        src_w=src_w,
        src_h=src_h,
        has_image=True,
    )
    return panel, meta


def point_to_camera(x: int, y: int) -> Optional[str]:
    if x < 0 or y < 0 or x >= PANEL_W * 2 or y >= PANEL_H * 2:
        return None
    col = x // PANEL_W
    row = y // PANEL_H
    idx = int(row * 2 + col)
    if 0 <= idx < len(CAMERAS):
        return CAMERAS[idx]
    return None


def update_prediction_bbox(entry: CameraEntry, frame_idx: int, bbox: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return False

    mask = entry.pred_df["video_frame"] == int(frame_idx)
    has_rows = bool(mask.any())

    if not has_rows:
        new_row = {col: np.nan for col in entry.pred_df.columns}
        new_row["video_frame"] = int(frame_idx)
        entry.pred_df = pd.concat([entry.pred_df, pd.DataFrame([new_row])], ignore_index=True)
        entry.pred_df = entry.pred_df.sort_values("video_frame").reset_index(drop=True)
        mask = entry.pred_df["video_frame"] == int(frame_idx)

    if "has_detection" in entry.pred_df.columns:
        entry.pred_df.loc[mask, "has_detection"] = True
    if "monkey_visible" in entry.pred_df.columns:
        entry.pred_df.loc[mask, "monkey_visible"] = True

    for col, val in (("bbox_x1", x1), ("bbox_y1", y1), ("bbox_x2", x2), ("bbox_y2", y2)):
        if col in entry.pred_df.columns:
            entry.pred_df.loc[mask, col] = float(val)

    w = float(x2 - x1)
    h = float(y2 - y1)
    cx = float((x1 + x2) / 2.0)
    cy = float((y1 + y2) / 2.0)
    if "bbox_w" in entry.pred_df.columns:
        entry.pred_df.loc[mask, "bbox_w"] = w
    if "bbox_h" in entry.pred_df.columns:
        entry.pred_df.loc[mask, "bbox_h"] = h
    if "bbox_cx" in entry.pred_df.columns:
        entry.pred_df.loc[mask, "bbox_cx"] = cx
    if "bbox_cy" in entry.pred_df.columns:
        entry.pred_df.loc[mask, "bbox_cy"] = cy

    entry.pred_df.to_csv(entry.pred_csv, index=False)
    return True


def panel_points_to_bbox(start_xy: tuple[int, int], end_xy: tuple[int, int], meta: PanelMeta) -> Optional[tuple[int, int, int, int]]:
    if not meta.has_image or meta.scale <= 0:
        return None

    def to_src(pxy: tuple[int, int]) -> tuple[int, int]:
        gx, gy = pxy
        lx = gx - meta.panel_x0
        ly = gy - meta.panel_y0

        lx = min(max(lx, meta.img_x0), meta.img_x0 + meta.img_w - 1)
        ly = min(max(ly, meta.img_y0), meta.img_y0 + meta.img_h - 1)

        sx = int(round((lx - meta.img_x0) / meta.scale))
        sy = int(round((ly - meta.img_y0) / meta.scale))
        sx = int(min(max(sx, 0), max(0, meta.src_w - 1)))
        sy = int(min(max(sy, 0), max(0, meta.src_h - 1)))
        return sx, sy

    sx1, sy1 = to_src(start_xy)
    sx2, sy2 = to_src(end_xy)
    x1, x2 = sorted((sx1, sx2))
    y1, y2 = sorted((sy1, sy2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


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
                time_input_text: str, time_input_active: bool,
                artifact_mode: bool, artifact_pos: int, artifact_total: int,
                status_text: str) -> tuple[np.ndarray, datetime, dict[str, tuple[CameraEntry, Optional[int], float]], dict[str, PanelMeta]]:
    abs_time = bundle.timeline[time_idx]
    context = resolve_time_context(bundle, abs_time)

    panels = []
    panel_meta: dict[str, PanelMeta] = {}
    for i, c in enumerate(CAMERAS, start=1):
        entry, frame_idx, dt = context[c]
        row = (i - 1) // 2
        col = (i - 1) % 2
        px0 = col * PANEL_W
        py0 = row * PANEL_H
        panel_img, meta = render_panel(entry, frame_idx, show_bbox, delta_t=dt, is_active=(c in active_cameras), panel_x0=px0, panel_y0=py0)
        panels.append(panel_img)
        panel_meta[c] = meta

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
        time_input_text=time_input_text,
        time_input_active=time_input_active,
        artifact_mode=artifact_mode,
        artifact_pos=artifact_pos,
        artifact_total=artifact_total,
        velocity_cm_s=velocity_cm_s,
        status_text=status_text,
        raw_ignored=raw_ignored,
        raw_idx=raw_idx,
    )
    canvas = np.hstack([mosaic, sidebar])
    return canvas, abs_time, context, panel_meta


def run(
    monkey: str,
    date: str,
    frames_root: Path,
    pred_root: Path,
    traj_root: Path,
    bundle_index: int,
    fps: float,
    artifact_velocity_thresh: float,
    artifact_dvdt_thresh: float,
    artifact_bridge_max_sec: float,
) -> None:
    frames_dir = frames_root / monkey / date
    pred_dir = pred_root / monkey / date
    traj_path = traj_root / monkey / date / "trajectory_realworld.csv"
    artifact_csv_path = traj_path.parent / ARTIFACT_FILENAME

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
    artifact_tidx_by_bundle = build_artifact_tidx_map(
        bundles,
        traj_index,
        velocity_thresh=artifact_velocity_thresh,
        dvdt_thresh=artifact_dvdt_thresh,
        bridge_max_sec=artifact_bridge_max_sec,
        artifact_csv_path=artifact_csv_path,
    )

    show_bbox = True
    show_tail = True
    bidx = bundle_index
    tidx = 0
    active_cameras: set[str] = set()
    trajectory_mask_mode = False
    status_text = ""

    win = "MultiCamPredictionReview"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    ui_state = {
        "refresh_requested": False,
        "dragging": False,
        "drag_cam": None,
        "drag_start": None,
        "drag_current": None,
        "pending_draw": None,
        "time_input_active": False,
        "time_input_text": "",
        "artifact_mode": False,
        "artifact_pos": 0,
    }

    def current_artifact_list() -> list[int]:
        return artifact_tidx_by_bundle.get(bidx, [])

    def sync_artifact_cursor_from_tidx() -> None:
        if not ui_state["artifact_mode"]:
            return
        art = current_artifact_list()
        if not art:
            return
        ui_state["artifact_pos"] = int(min(range(len(art)), key=lambda i: abs(art[i] - tidx)))

    def sync_tidx_to_artifact_mode() -> None:
        nonlocal tidx
        if not ui_state["artifact_mode"]:
            return
        art = current_artifact_list()
        if not art:
            return
        ui_state["artifact_pos"] = max(0, min(ui_state["artifact_pos"], len(art) - 1))
        tidx = int(art[ui_state["artifact_pos"]])

    def move_timeline(delta: int) -> None:
        nonlocal tidx
        if ui_state["artifact_mode"]:
            art = current_artifact_list()
            if art:
                ui_state["artifact_pos"] = max(0, min(ui_state["artifact_pos"] + delta, len(art) - 1))
                tidx = int(art[ui_state["artifact_pos"]])
                return
        tidx = max(0, min(tidx + delta, len(bundles[bidx].timeline) - 1))

    def normalize_hms_text(text: str) -> str:
        chars = [c for c in text if c.isdigit()]
        chars = chars[:6]
        if len(chars) <= 2:
            return "".join(chars)
        if len(chars) <= 4:
            return f"{''.join(chars[:2])}:{''.join(chars[2:])}"
        return f"{''.join(chars[:2])}:{''.join(chars[2:4])}:{''.join(chars[4:])}"

    def seek_tidx_by_hms(bundle: Bundle, current_abs_time: datetime, hms_text: str) -> tuple[Optional[int], str]:
        if not re.fullmatch(r"\d{2}:\d{2}:\d{2}", hms_text):
            return None, "invalid time; use HH:MM:SS"
        try:
            hh, mm, ss = [int(v) for v in hms_text.split(":")]
            target_dt = current_abs_time.replace(hour=hh, minute=mm, second=ss, microsecond=0)
        except Exception:
            return None, "invalid time components"

        ts = np.array([dt.timestamp() for dt in bundle.timeline], dtype=np.float64)
        if ts.size == 0:
            return None, "empty timeline"
        t = target_dt.timestamp()
        pos = int(np.searchsorted(ts, t))
        cand = []
        if 0 <= pos < len(ts):
            cand.append(pos)
        if pos - 1 >= 0:
            cand.append(pos - 1)
        if not cand:
            return None, "seek failed"
        idx = min(cand, key=lambda i: abs(ts[i] - t))
        return int(idx), f"jumped to {bundle.timeline[idx].strftime('%H:%M:%S')}"

    def on_mouse(event: int, x: int, y: int, flags: int, userdata) -> None:
        if x >= PANEL_W * 2:
            x_rel = x - PANEL_W * 2
            tx0, ty0, tx1, ty1 = sidebar_time_input_rect(PANEL_H * 2)
            if event == cv2.EVENT_LBUTTONDOWN:
                ui_state["time_input_active"] = (tx0 <= x_rel <= tx1 and ty0 <= y <= ty1)
            return

        if trajectory_mask_mode:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            cam = point_to_camera(x, y)
            if cam is None:
                return
            if len(active_cameras) != 1 or cam not in active_cameras:
                return
            ui_state["dragging"] = True
            ui_state["drag_cam"] = cam
            ui_state["drag_start"] = (x, y)
            ui_state["drag_current"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and ui_state["dragging"]:
            ui_state["drag_current"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and ui_state["dragging"]:
            ui_state["dragging"] = False
            ui_state["drag_current"] = (x, y)
            ui_state["pending_draw"] = (ui_state["drag_cam"], ui_state["drag_start"], ui_state["drag_current"])
            ui_state["drag_cam"] = None

    cv2.setMouseCallback(win, on_mouse)

    print("\nControls: N/P next-prev frame, J/K jump -/+10, B/M prev-next bundle, H bbox toggle, T tail toggle, R refresh trajectory, A artifact focus, Q quit")
    print("Edit: 1/2/3/4 toggle D01..D04 (multi-select), 5 toggle trajectory mask, 0 clear all, I toggle mask (mode 5), U no-prediction (all selected cameras)")

    while True:
        bundle = bundles[bidx]
        sync_tidx_to_artifact_mode()
        art_list = current_artifact_list() if ui_state["artifact_mode"] else []

        canvas, abs_time, context, panel_meta = make_canvas(
            bundle,
            tidx,
            show_bbox,
            traj_index,
            show_tail,
            active_cameras,
            trajectory_mask_mode,
            ui_state["time_input_text"],
            ui_state["time_input_active"],
            ui_state["artifact_mode"],
            ui_state["artifact_pos"],
            len(art_list),
            status_text,
        )

        if ui_state["dragging"] and ui_state["drag_start"] is not None and ui_state["drag_current"] is not None:
            x1, y1 = ui_state["drag_start"]
            x2, y2 = ui_state["drag_current"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 180, 0), 2)

        cv2.imshow(win, canvas)
        key = cv2.waitKey(30) & 0xFF
        status_text = ""

        if ui_state["time_input_active"]:
            if key in (13, 10):
                idx, msg = seek_tidx_by_hms(bundle, abs_time, ui_state["time_input_text"])
                status_text = msg
                if idx is not None:
                    tidx = idx
                    sync_artifact_cursor_from_tidx()
                ui_state["time_input_active"] = False
            elif key == 27:
                ui_state["time_input_active"] = False
            elif key in (8, 127):
                ui_state["time_input_text"] = normalize_hms_text(ui_state["time_input_text"][:-1])
            elif 32 <= key <= 126:
                ch = chr(key)
                if ch.isdigit() or ch == ":":
                    ui_state["time_input_text"] = normalize_hms_text(ui_state["time_input_text"] + ch)
            continue

        if ui_state["pending_draw"] is not None:
            cam, start_xy, end_xy = ui_state["pending_draw"]
            ui_state["pending_draw"] = None
            if cam not in context:
                status_text = "bbox draw failed: camera not in context"
            elif cam not in panel_meta:
                status_text = f"{cam}: bbox draw unavailable"
            else:
                entry, frame_idx, _ = context[cam]
                if frame_idx is None:
                    status_text = f"{cam}: no frame to edit"
                else:
                    bbox = panel_points_to_bbox(start_xy, end_xy, panel_meta[cam])
                    if bbox is None:
                        status_text = f"{cam}: bbox too small"
                    else:
                        ok = update_prediction_bbox(entry, frame_idx, bbox)
                        if ok:
                            status_text = f"{cam}: bbox updated @ frame {frame_idx}"
                        else:
                            status_text = f"{cam}: bbox update failed"

        if key in (ord("q"), 27):
            break
        elif key in (ord("a"), ord("A")):
            ui_state["artifact_mode"] = not ui_state["artifact_mode"]
            if ui_state["artifact_mode"]:
                art = current_artifact_list()
                if not art:
                    ui_state["artifact_mode"] = False
                    status_text = "artifact focus: none in this bundle"
                else:
                    sync_artifact_cursor_from_tidx()
                    sync_tidx_to_artifact_mode()
                    status_text = f"artifact focus ON ({len(art)} frames)"
            else:
                status_text = "artifact focus OFF"
        elif key in (ord("r"), ord("R")):
            ui_state["refresh_requested"] = True
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
            move_timeline(+1)
        elif key in (ord("p"), 81):
            move_timeline(-1)
        elif key == ord("k"):
            move_timeline(+10)
        elif key == ord("j"):
            move_timeline(-10)
        elif key in (ord("i"), ord("I")):
            if trajectory_mask_mode:
                ok, now_ignored = traj_index.toggle_ignore_at_time(abs_time)
                if ok:
                    status_text = "trajectory frame masked (pending save)" if now_ignored else "trajectory frame unmasked (pending save)"
                else:
                    status_text = "trajectory mask toggle failed"
                move_timeline(+1)
            else:
                status_text = "select 5 to use trajectory mask"
        elif key in (ord("u"), ord("U")):
            if active_cameras:
                results = []
                for cam in sorted(active_cameras):
                    matches = matching_frames_for_time(bundle, cam, abs_time)
                    if not matches:
                        results.append(f"{cam}:no-frame")
                        continue

                    applied = 0
                    for entry, frame_idx, _ in matches:
                        if apply_prediction_edit(entry, frame_idx, mode="no_prediction"):
                            applied += 1

                    if applied > 0:
                        results.append(f"{cam}:okx{applied}")
                    else:
                        results.append(f"{cam}:none")
                status_text = "U -> " + ", ".join(results)
                move_timeline(+1)
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
                ui_state["artifact_pos"] = 0
                if ui_state["artifact_mode"]:
                    art = current_artifact_list()
                    if art:
                        sync_tidx_to_artifact_mode()
                    else:
                        ui_state["artifact_mode"] = False
                        status_text = "artifact focus OFF (no artifacts in bundle)"
        elif key == ord("b"):
            if bidx > 0:
                bidx -= 1
                tidx = 0
                ui_state["artifact_pos"] = 0
                if ui_state["artifact_mode"]:
                    art = current_artifact_list()
                    if art:
                        sync_tidx_to_artifact_mode()
                    else:
                        ui_state["artifact_mode"] = False
                        status_text = "artifact focus OFF (no artifacts in bundle)"

        if ui_state["refresh_requested"]:
            traj_index.save_if_dirty()
            try:
                bundles = build_bundles(pred_dir=pred_dir, frames_dir=frames_dir, fps=fps)
                if not bundles:
                    raise RuntimeError("no synchronized bundles found")
                bidx = max(0, min(bidx, len(bundles) - 1))
                tidx = max(0, min(tidx, len(bundles[bidx].timeline) - 1))
                traj_index = TrajectoryIndex(traj_path)
                artifact_tidx_by_bundle = build_artifact_tidx_map(
                    bundles,
                    traj_index,
                    velocity_thresh=artifact_velocity_thresh,
                    dvdt_thresh=artifact_dvdt_thresh,
                    bridge_max_sec=artifact_bridge_max_sec,
                    artifact_csv_path=artifact_csv_path,
                )
                if ui_state["artifact_mode"]:
                    art = current_artifact_list()
                    if art:
                        sync_artifact_cursor_from_tidx()
                        sync_tidx_to_artifact_mode()
                    else:
                        ui_state["artifact_mode"] = False
                        status_text = "trajectory + overlays reloaded; artifact focus OFF"
                        ui_state["refresh_requested"] = False
                        continue
                status_text = "trajectory + overlays reloaded"
            except Exception as e:
                status_text = f"refresh load failed: {str(e)[:64]}"
            ui_state["refresh_requested"] = False

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
        artifact_velocity_thresh=float(args.artifact_velocity_thresh),
        artifact_dvdt_thresh=float(args.artifact_dvdt_thresh),
        artifact_bridge_max_sec=float(args.artifact_bridge_max_sec),
    )


if __name__ == "__main__":
    main()
