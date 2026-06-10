#!/usr/bin/env python3
"""
Fuse multi-camera manual labels into a single real-world trajectory.

Reads per-camera label CSVs (from manual_labeler.py) and per-camera calibration
JSONs (from calibrate_cameras.py), then:

  1. Applies each camera's floor homography to convert pixel → cm
  2. For frames labeled in two cameras (boundary overlap), averages the cm positions
  3. Merges all cameras into one time-ordered trajectory
  4. Linearly interpolates over short gaps (configurable)
  5. Computes per-frame velocity and movement bouts

Outputs:
    results/oft_analysis/<monkey>/<date>/trajectory_realworld.csv
    results/oft_analysis/<monkey>/<date>/trajectory_summary.json

Usage:
    conda activate video_tracking
    python scripts/fuse_trajectories.py
    python scripts/fuse_trajectories.py --monkey H1705011 --date 2026-05-25
    python scripts/fuse_trajectories.py --help
"""

import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import cv2


# ── Defaults ───────────────────────────────────────────────────────────────────

MONKEY_ID   = 'H1705011'
SESSION_DATE = '2026-05-25'
FPS          = 20.0
CAMERAS      = ['D1', 'D2', 'D3', 'D4']

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
RESULTS_BASE = PROJECT_ROOT / 'results' / 'oft_analysis'
CAL_BASE     = PROJECT_ROOT / 'results' / 'calibration'

# Interpolation: fill gaps ≤ this many frames (0 = disabled)
MAX_INTERP_FRAMES = 40   # 2 s at 20 fps

# Movement threshold for bout detection (cm/s)
VELOCITY_THRESHOLD = 5.0   # cm/s
MIN_BOUT_DURATION  = 0.5   # s

# Sentinel from manual_labeler
NO_MONKEY_SENTINEL = -1


# ── Homography helper ──────────────────────────────────────────────────────────

def pixel_to_cm(pixel_xy, H: np.ndarray):
    """Apply 3×3 floor homography: pixel → (x_cm, y_cm)."""
    pt = np.array([[[float(pixel_xy[0]), float(pixel_xy[1])]]], dtype=np.float64)
    r  = cv2.perspectiveTransform(pt, H)
    return float(r[0, 0, 0]), float(r[0, 0, 1])


# ── Load calibrations ──────────────────────────────────────────────────────────

def load_homographies(cal_dir: Path, cameras: List[str]) -> Dict[str, Optional[np.ndarray]]:
    """Load floor homography matrix for each camera. Returns None if not available."""
    Hs: Dict[str, Optional[np.ndarray]] = {}
    for cam in cameras:
        cal_file = cal_dir / f'calibration_{cam}.json'
        if not cal_file.exists():
            print(f"  ⚠  No calibration file for {cam} ({cal_file.name}) — camera will be skipped")
            Hs[cam] = None
            continue
        with open(cal_file) as f:
            data = json.load(f)
        H_raw = data.get('floor_homography')
        if H_raw is None:
            print(f"  ⚠  Calibration for {cam} has no floor_homography — camera will be skipped")
            Hs[cam] = None
            continue
        Hs[cam] = np.array(H_raw, dtype=np.float64)
        err = data.get('floor_reproj_error_cm', float('nan'))
        print(f"  ✓  {cam}: homography loaded  (reprojection error: {err:.2f} cm)")
    return Hs


# ── Load labels ────────────────────────────────────────────────────────────────

def load_labels(labels_dir: Path, cameras: List[str],
                fps: float = FPS) -> Dict[str, pd.DataFrame]:
    """Load per-camera label CSVs. Returns empty DataFrame for missing cameras."""
    dfs: Dict[str, pd.DataFrame] = {}
    for cam in cameras:
        csv_file = labels_dir / f'labels_{cam}.csv'
        if not csv_file.exists():
            print(f"  ⚠  No labels for {cam} ({csv_file.name})")
            dfs[cam] = pd.DataFrame()
            continue
        df = pd.read_csv(csv_file)
        # Keep only rows where monkey is visible
        if 'monkey_visible' in df.columns:
            df = df[df['monkey_visible'] == True].copy()
        else:
            # Legacy format: filter out sentinel values
            df = df[(df['centroid_x'] != NO_MONKEY_SENTINEL)].copy()
        df['camera'] = cam
        print(f"  ✓  {cam}: {len(df)} visible-monkey frames loaded")
        dfs[cam] = df
    return dfs


# ── Convert pixel → cm per camera ─────────────────────────────────────────────

def apply_homographies(label_dfs: Dict[str, pd.DataFrame],
                       Hs: Dict[str, Optional[np.ndarray]]) -> pd.DataFrame:
    """
    For each labeled frame convert pixel centroid → real-world cm.
    Returns a DataFrame with columns:
        video_frame, time_sec, camera, x_cm, y_cm
    """
    rows = []
    for cam, df in label_dfs.items():
        H = Hs.get(cam)
        if H is None or df.empty:
            continue
        for _, row in df.iterrows():
            x_cm, y_cm = pixel_to_cm((row['centroid_x'], row['centroid_y']), H)
            rows.append({
                'video_frame': int(row['video_frame']),
                'time_sec':    float(row['time_sec']),
                'camera':      cam,
                'x_cm':        x_cm,
                'y_cm':        y_cm,
                'pixel_x':     float(row['centroid_x']),
                'pixel_y':     float(row['centroid_y']),
            })
    return pd.DataFrame(rows)


# ── Fuse overlapping frames ────────────────────────────────────────────────────

def fuse_overlapping(cm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Where the same video_frame is labeled by multiple cameras (boundary overlap),
    average the real-world cm positions.  The 'camera' column becomes a
    comma-separated list of contributing cameras.
    """
    if cm_df.empty:
        return cm_df

    fused_rows = []
    for frame, grp in cm_df.groupby('video_frame'):
        if len(grp) == 1:
            row = grp.iloc[0].to_dict()
        else:
            row = {
                'video_frame': frame,
                'time_sec':    grp['time_sec'].iloc[0],
                'camera':      ','.join(sorted(grp['camera'].tolist())),
                'x_cm':        grp['x_cm'].mean(),
                'y_cm':        grp['y_cm'].mean(),
                'pixel_x':     float('nan'),
                'pixel_y':     float('nan'),
            }
        fused_rows.append(row)

    fused = pd.DataFrame(fused_rows).sort_values('video_frame').reset_index(drop=True)
    return fused


# ── Interpolate gaps ───────────────────────────────────────────────────────────

def interpolate_gaps(df: pd.DataFrame, fps: float,
                     max_gap_frames: int = MAX_INTERP_FRAMES) -> pd.DataFrame:
    """
    Linearly interpolate x_cm / y_cm over temporal gaps ≤ max_gap_frames.
    Inserted rows get camera = 'interpolated'.
    """
    if df.empty or max_gap_frames == 0:
        return df

    rows = df.to_dict('records')
    filled = [rows[0]]

    for i in range(1, len(rows)):
        prev, curr = rows[i - 1], rows[i]
        gap = curr['video_frame'] - prev['video_frame']
        if 1 < gap <= max_gap_frames:
            for step in range(1, gap):
                frac = step / gap
                filled.append({
                    'video_frame': prev['video_frame'] + step,
                    'time_sec':    prev['time_sec'] + step / fps,
                    'camera':      'interpolated',
                    'x_cm':        prev['x_cm'] + frac * (curr['x_cm'] - prev['x_cm']),
                    'y_cm':        prev['y_cm'] + frac * (curr['y_cm'] - prev['y_cm']),
                    'pixel_x':     float('nan'),
                    'pixel_y':     float('nan'),
                })
        filled.append(curr)

    return pd.DataFrame(filled).reset_index(drop=True)


# ── Kinematics ─────────────────────────────────────────────────────────────────

def compute_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """Add distance_cm and velocity_cm_s columns."""
    df = df.copy()
    dx = df['x_cm'].diff()
    dy = df['y_cm'].diff()
    dt = df['time_sec'].diff()
    df['distance_cm']   = np.sqrt(dx**2 + dy**2)
    df['velocity_cm_s'] = (df['distance_cm'] / dt).replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def detect_bouts(df: pd.DataFrame,
                 vel_thresh: float = VELOCITY_THRESHOLD,
                 min_dur: float = MIN_BOUT_DURATION) -> pd.DataFrame:
    """Detect movement bouts from velocity time-series."""
    df = df.copy()
    df['is_moving'] = df['velocity_cm_s'] > vel_thresh
    df['bout_change'] = df['is_moving'].astype(int).diff()

    bouts = []
    start = None
    for _, row in df.iterrows():
        if row['bout_change'] == 1:
            start = row['time_sec']
        elif row['bout_change'] == -1 and start is not None:
            dur = row['time_sec'] - start
            if dur >= min_dur:
                bouts.append({'start_sec': start, 'end_sec': row['time_sec'],
                              'duration_sec': dur})
            start = None
    return pd.DataFrame(bouts)


# ── Summary stats ──────────────────────────────────────────────────────────────

def compute_summary(traj: pd.DataFrame, bouts: pd.DataFrame) -> dict:
    total_dur    = traj['time_sec'].max() - traj['time_sec'].min()
    total_dist   = traj['distance_cm'].sum()
    mean_vel     = traj['velocity_cm_s'].mean()
    immob_frames = (traj['velocity_cm_s'] <= VELOCITY_THRESHOLD).sum()
    immob_pct    = 100.0 * immob_frames / len(traj) if len(traj) > 0 else 0.0

    return {
        'total_duration_sec':           round(float(total_dur), 2),
        'total_distance_cm':            round(float(total_dist), 2),
        'mean_velocity_cm_s':           round(float(mean_vel), 3),
        'max_velocity_cm_s':            round(float(traj['velocity_cm_s'].max()), 3),
        'immobility_pct':               round(immob_pct, 2),
        'n_movement_bouts':             len(bouts),
        'mean_bout_duration_sec':       round(float(bouts['duration_sec'].mean()), 3) if len(bouts) > 0 else 0.0,
        'total_active_time_sec':        round(float(bouts['duration_sec'].sum()), 2) if len(bouts) > 0 else 0.0,
        'n_labeled_frames':             int((traj['camera'] != 'interpolated').sum()),
        'n_interpolated_frames':        int((traj['camera'] == 'interpolated').sum()),
        'cameras_used':                 sorted(set(c for cams in traj['camera']
                                                  for c in cams.split(',')
                                                  if c != 'interpolated')),
        'velocity_threshold_cm_s':      VELOCITY_THRESHOLD,
        'max_interp_frames':            MAX_INTERP_FRAMES,
    }


# ── Main pipeline ──────────────────────────────────────────────────────────────

def fuse(monkey_id: str = MONKEY_ID, session_date: str = SESSION_DATE,
         cameras: List[str] = None, fps: float = FPS,
         max_interp: int = MAX_INTERP_FRAMES,
         cal_dir: Optional[Path] = None,
         labels_dir: Optional[Path] = None,
         output_dir: Optional[Path] = None) -> pd.DataFrame:

    cameras = cameras or CAMERAS

    labels_dir = labels_dir or (RESULTS_BASE / monkey_id / session_date)
    output_dir = output_dir or labels_dir
    cal_dir    = cal_dir    or CAL_BASE

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Fusing trajectories for {monkey_id} / {session_date}")
    print(f"{'='*60}")

    print("\n── Loading homographies ──")
    Hs = load_homographies(cal_dir, cameras)

    print("\n── Loading labels ──")
    label_dfs = load_labels(labels_dir, cameras, fps)

    print("\n── Converting pixel → cm ──")
    cm_df = apply_homographies(label_dfs, Hs)
    if cm_df.empty:
        print("ERROR: No data after homography conversion. Check calibration and labels.")
        return pd.DataFrame()
    print(f"  {len(cm_df)} total labeled positions across all cameras")

    print("\n── Fusing boundary-overlap frames ──")
    traj = fuse_overlapping(cm_df)
    n_multi = (traj['camera'].str.contains(',', na=False)).sum()
    print(f"  {len(traj)} unique frames  ({n_multi} averaged from two cameras)")

    print(f"\n── Interpolating gaps (max {max_interp} frames = {max_interp/fps:.1f}s) ──")
    traj = interpolate_gaps(traj, fps, max_interp)
    n_interp = (traj['camera'] == 'interpolated').sum()
    print(f"  {n_interp} frames interpolated  →  {len(traj)} total frames")

    print("\n── Computing kinematics ──")
    traj = compute_kinematics(traj)
    bouts = detect_bouts(traj)
    summary = compute_summary(traj, bouts)

    # Save outputs
    traj_path   = output_dir / 'trajectory_realworld.csv'
    bouts_path  = output_dir / 'movement_bouts.csv'
    summ_path   = output_dir / 'trajectory_summary.json'

    traj.to_csv(traj_path,  index=False)
    bouts.to_csv(bouts_path, index=False)
    with open(summ_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(" Summary")
    print(f"{'='*60}")
    for k, v in summary.items():
        print(f"  {k:<35} {v}")
    print(f"\n✅  trajectory_realworld.csv  → {traj_path}")
    print(f"    movement_bouts.csv        → {bouts_path}")
    print(f"    trajectory_summary.json   → {summ_path}")

    return traj


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Fuse multi-camera OFT labels into real-world trajectory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--monkey',      default=MONKEY_ID,    help='Monkey ID')
    parser.add_argument('--date',        default=SESSION_DATE, help='Session date (YYYY-MM-DD)')
    parser.add_argument('--cameras',     default=None, nargs='+', choices=['D1','D2','D3','D4'],
                        help='Cameras to include (default: all 4)')
    parser.add_argument('--fps',         type=float, default=FPS)
    parser.add_argument('--max-interp',  type=int,   default=MAX_INTERP_FRAMES,
                        help='Max gap in frames to interpolate (default: 40 = 2s)')
    parser.add_argument('--cal-dir',     default=None,
                        help=f'Calibration dir (default: {CAL_BASE})')
    parser.add_argument('--labels-dir',  default=None,
                        help='Labels dir (default: results/oft_analysis/<monkey>/<date>/)')
    parser.add_argument('--output-dir',  default=None,
                        help='Output dir (default: same as labels-dir)')
    args = parser.parse_args()

    fuse(
        monkey_id=args.monkey,
        session_date=args.date,
        cameras=args.cameras,
        fps=args.fps,
        max_interp=args.max_interp,
        cal_dir=Path(args.cal_dir) if args.cal_dir else None,
        labels_dir=Path(args.labels_dir) if args.labels_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == '__main__':
    main()
