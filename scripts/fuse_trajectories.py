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

# Camera quadrant bounds (cm) - for filtering out-of-bounds projections
CAMERA_QUADRANTS = {
    'D1': {'x_min': 0,   'x_max': 161, 'y_min': 0,   'y_max': 161},  # top-left
    'D2': {'x_min': 161, 'x_max': 322, 'y_min': 0,   'y_max': 161},  # top-right
    'D3': {'x_min': 0,   'x_max': 161, 'y_min': 161, 'y_max': 322},  # bottom-left
    'D4': {'x_min': 161, 'x_max': 322, 'y_min': 161, 'y_max': 322},  # bottom-right
}

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

def load_homographies(cal_dir: Path, cameras: List[str]):
    """
    Load floor homography matrix and calibrated pixel bounds for each camera.
    
    Returns:
        Tuple of (Hs, pixel_bounds) where:
        - Hs: Dict[str, Optional[np.ndarray]] - homography matrices
        - pixel_bounds: Dict[str, Optional[dict]] - pixel bounds with x_min, x_max, y_min, y_max
    """
    Hs: Dict[str, Optional[np.ndarray]] = {}
    pixel_bounds: Dict[str, Optional[dict]] = {}
    
    for cam in cameras:
        cal_file = cal_dir / f'calibration_{cam}.json'
        if not cal_file.exists():
            print(f"  ⚠  No calibration file for {cam} ({cal_file.name}) — camera will be skipped")
            Hs[cam] = None
            pixel_bounds[cam] = None
            continue
        
        with open(cal_file) as f:
            data = json.load(f)
        
        H_raw = data.get('floor_homography')
        if H_raw is None:
            print(f"  ⚠  Calibration for {cam} has no floor_homography — camera will be skipped")
            Hs[cam] = None
            pixel_bounds[cam] = None
            continue
        
        Hs[cam] = np.array(H_raw, dtype=np.float64)
        
        # Extract calibrated pixel bounds from floor corner points
        floor_pts = [p for p in data['labeled_points'] 
                    if p.get('pixel') and p['world_cm'][2] == 0]
        if floor_pts:
            xs = [p['pixel'][0] for p in floor_pts]
            ys = [p['pixel'][1] for p in floor_pts]
            pixel_bounds[cam] = {
                'x_min': min(xs),
                'x_max': max(xs),
                'y_min': min(ys),
                'y_max': max(ys),
            }
        else:
            pixel_bounds[cam] = None
        
        err = data.get('floor_reproj_error_cm', float('nan'))
        print(f"  ✓  {cam}: homography loaded  (reprojection error: {err:.2f} cm)")
    
    return Hs, pixel_bounds


# ── Load labels ────────────────────────────────────────────────────────────────

def load_labels(labels_dir: Path, cameras: List[str],
                fps: float = FPS) -> List[pd.DataFrame]:
    """
    Load all per-camera label CSVs (handles multiple videos per camera).
    Returns list of DataFrames, one per label CSV file.
    """
    from datetime import datetime
    import re
    
    dfs: List[pd.DataFrame] = []
    
    for cam in cameras:
        # Find all label CSVs for this camera: labels_D03.csv, labels_D03_170340.csv, etc.
        # Camera names are D1-D4 but filenames use D01-D04 (with leading zero)
        cam_num = cam[1]  # '1', '2', '3', '4'
        pattern = f'labels_D0{cam_num}*.csv'
        csv_files = sorted(labels_dir.glob(pattern))
        
        if not csv_files:
            print(f"  ⚠  No labels for {cam}")
            continue
        
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            
            # Keep only rows where monkey is visible
            if 'monkey_visible' in df.columns:
                df = df[df['monkey_visible'] == True].copy()
            else:
                # Legacy format: filter out sentinel values
                df = df[(df['centroid_x'] != NO_MONKEY_SENTINEL)].copy()
            
            if df.empty:
                continue
            
            # Compute absolute timestamp for each frame
            if 'video_timestamp' in df.columns:
                # New format: video_timestamp already in CSV
                base_time = pd.to_datetime(df['video_timestamp'].iloc[0])
            else:
                # Legacy format: try to parse from filename or video_file column
                if 'video_file' in df.columns:
                    video_name = df['video_file'].iloc[0]
                else:
                    # Fall back to CSV filename
                    video_name = csv_file.name
                
                match = re.search(r'D0[1-4]_(\d{14})', video_name)
                if match:
                    ts_str = match.group(1)
                    base_time = datetime.strptime(ts_str, '%Y%m%d%H%M%S')
                else:
                    print(f"  ⚠  Could not parse timestamp from {csv_file.name}, skipping")
                    continue
            
            # Add absolute timestamp column
            df['absolute_time'] = base_time + pd.to_timedelta(df['time_sec'], unit='s')
            df['camera'] = cam
            
            print(f"  ✓  {cam} ({csv_file.name}): {len(df)} frames, start={base_time.strftime('%H:%M:%S')}")
            dfs.append(df)
    
    return dfs


# ── Convert pixel → cm per camera ─────────────────────────────────────────────

def apply_homographies(label_dfs: List[pd.DataFrame],
                       Hs: Dict[str, Optional[np.ndarray]],
                       pixel_bounds: Dict[str, Optional[dict]],
                       filter_out_of_bounds: bool = True,
                       pixel_margin: float = 100.0,
                       world_margin_cm: float = 30.0) -> pd.DataFrame:
    """
    For each labeled frame convert pixel centroid → real-world cm.
    
    Filters labels that fall outside the calibrated pixel region to prevent
    homography extrapolation errors.
    
    Args:
        label_dfs: List of label DataFrames
        Hs: Dict of homography matrices per camera
        pixel_bounds: Dict of calibrated pixel bounds per camera
        filter_out_of_bounds: If True, reject labels outside calibrated region
        pixel_margin: Pixels beyond calibration bounds to allow (for edge tolerance)
        world_margin_cm: Additional cm beyond quadrant bounds (for boundary overlap)
    
    Returns a DataFrame with columns:
        absolute_time, camera, x_cm, y_cm, pixel_x, pixel_y
    """
    rows = []
    n_filtered_pixel = 0
    n_filtered_world = 0
    n_clipped = 0
    
    for df in label_dfs:
        if df.empty:
            continue
        
        cam = df['camera'].iloc[0]
        H = Hs.get(cam)
        if H is None:
            continue
        
        # Get pixel bounds and world quadrant bounds for this camera
        pbounds = pixel_bounds.get(cam)
        wbounds = CAMERA_QUADRANTS.get(cam)
        
        for _, row in df.iterrows():
            px, py = row['centroid_x'], row['centroid_y']
            
            # Filter based on pixel coordinates BEFORE projection
            if filter_out_of_bounds and pbounds:
                if not (pbounds['x_min'] - pixel_margin <= px <= pbounds['x_max'] + pixel_margin and
                        pbounds['y_min'] - pixel_margin <= py <= pbounds['y_max'] + pixel_margin):
                    n_filtered_pixel += 1
                    continue
            
            # Apply homography
            x_cm, y_cm = pixel_to_cm((px, py), H)
            
            # Secondary filter: world coordinate bounds (for boundary overlap)
            if filter_out_of_bounds and wbounds:
                x_min = wbounds['x_min'] - world_margin_cm
                x_max = wbounds['x_max'] + world_margin_cm
                y_min = wbounds['y_min'] - world_margin_cm
                y_max = wbounds['y_max'] + world_margin_cm
                
                if not (x_min <= x_cm <= x_max and y_min <= y_cm <= y_max):
                    n_filtered_world += 1
                    continue
            
            # Clip to arena bounds (head can extend beyond floor edge)
            x_cm_orig, y_cm_orig = x_cm, y_cm
            x_cm = np.clip(x_cm, 0, 322)
            y_cm = np.clip(y_cm, 0, 322)
            if x_cm != x_cm_orig or y_cm != y_cm_orig:
                n_clipped += 1
            
            rows.append({
                'absolute_time': row['absolute_time'],
                'camera':        cam,
                'x_cm':          x_cm,
                'y_cm':          y_cm,
                'pixel_x':       float(px),
                'pixel_y':       float(py),
            })
    
    if n_filtered_pixel > 0:
        print(f"  ⚠  Filtered {n_filtered_pixel} labels outside calibrated pixel region (margin={pixel_margin:.0f}px)")
    if n_filtered_world > 0:
        print(f"  ⚠  Filtered {n_filtered_world} projections outside world quadrant bounds (margin={world_margin_cm:.0f}cm)")
    if n_clipped > 0:
        print(f"  ℹ  Clipped {n_clipped} points to arena bounds [0, 322] cm")
    
    return pd.DataFrame(rows)


# ── Fuse overlapping frames ────────────────────────────────────────────────────

def fuse_overlapping(cm_df: pd.DataFrame, time_tolerance_sec: float = 0.1) -> pd.DataFrame:
    """
    Where multiple cameras see the monkey at approximately the same absolute time,
    average the real-world cm positions. Groups observations within time_tolerance_sec.
    The 'camera' column becomes a comma-separated list of contributing cameras.
    """
    if cm_df.empty:
        return cm_df

    # Sort by absolute time
    cm_df = cm_df.sort_values('absolute_time').reset_index(drop=True)
    
    # Group by time windows (within tolerance)
    fused_rows = []
    current_group = [cm_df.iloc[0].to_dict()]
    
    for i in range(1, len(cm_df)):
        row = cm_df.iloc[i]
        prev_time = current_group[-1]['absolute_time']
        
        # If within tolerance, add to current group
        if (row['absolute_time'] - prev_time).total_seconds() <= time_tolerance_sec:
            current_group.append(row.to_dict())
        else:
            # Finalize current group
            if len(current_group) == 1:
                fused_rows.append(current_group[0])
            else:
                # Average positions from multiple cameras
                fused_rows.append({
                    'absolute_time': current_group[0]['absolute_time'],
                    'camera':        ','.join(sorted(set(r['camera'] for r in current_group))),
                    'x_cm':          np.mean([r['x_cm'] for r in current_group]),
                    'y_cm':          np.mean([r['y_cm'] for r in current_group]),
                    'pixel_x':       float('nan'),
                    'pixel_y':       float('nan'),
                })
            # Start new group
            current_group = [row.to_dict()]
    
    # Don't forget the last group
    if len(current_group) == 1:
        fused_rows.append(current_group[0])
    else:
        fused_rows.append({
            'absolute_time': current_group[0]['absolute_time'],
            'camera':        ','.join(sorted(set(r['camera'] for r in current_group))),
            'x_cm':          np.mean([r['x_cm'] for r in current_group]),
            'y_cm':          np.mean([r['y_cm'] for r in current_group]),
            'pixel_x':       float('nan'),
            'pixel_y':       float('nan'),
        })

    fused = pd.DataFrame(fused_rows).reset_index(drop=True)
    # Add time_sec as seconds from start
    fused['time_sec'] = (fused['absolute_time'] - fused['absolute_time'].iloc[0]).dt.total_seconds()
    return fused


# ── Interpolate gaps ───────────────────────────────────────────────────────────

def interpolate_gaps(df: pd.DataFrame, fps: float,
                     max_gap_sec: float = None) -> pd.DataFrame:
    """
    Linearly interpolate x_cm / y_cm over temporal gaps ≤ max_gap_sec.
    Inserted rows get camera = 'interpolated'.
    """
    if df.empty:
        return df
    
    if max_gap_sec is None:
        max_gap_sec = MAX_INTERP_FRAMES / fps  # Convert frames to seconds
    
    if max_gap_sec == 0:
        return df

    rows = df.to_dict('records')
    filled = [rows[0]]

    for i in range(1, len(rows)):
        prev, curr = rows[i - 1], rows[i]
        gap_sec = curr['time_sec'] - prev['time_sec']
        
        if 0 < gap_sec <= max_gap_sec:
            # Interpolate at fps rate
            n_steps = int(gap_sec * fps)
            for step in range(1, n_steps):
                frac = step / n_steps
                filled.append({
                    'absolute_time': prev['absolute_time'] + pd.Timedelta(seconds=step / fps),
                    'time_sec':      prev['time_sec'] + step / fps,
                    'camera':        'interpolated',
                    'x_cm':          prev['x_cm'] + frac * (curr['x_cm'] - prev['x_cm']),
                    'y_cm':          prev['y_cm'] + frac * (curr['y_cm'] - prev['y_cm']),
                    'pixel_x':       float('nan'),
                    'pixel_y':       float('nan'),
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
    Hs, pixel_bounds = load_homographies(cal_dir, cameras)

    print("\n── Loading labels ──")
    label_dfs = load_labels(labels_dir, cameras, fps)

    print("\n── Converting pixel → cm ──")
    cm_df = apply_homographies(label_dfs, Hs, pixel_bounds,
                               filter_out_of_bounds=True,
                               pixel_margin=100.0,
                               world_margin_cm=10.0)  # Stricter world bounds
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
