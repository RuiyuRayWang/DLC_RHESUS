#!/usr/bin/env python3
"""
Reconstruct a real-world trajectory from RT-DETR bbox predictions.

This is a focused replacement for the prediction side of fuse_trajectories.py.
It keeps the stereo-intrinsic correction, quadrant fusion, interpolation, and
kinematics, but fixes time alignment by deriving each CSV's start timestamp
from its filename.

Expected prediction filename pattern:
    D01_20260626154641_downsampled_preds.csv

The timestamp portion (YYYYMMDDHHMMSS) is treated as the start time of that
prediction file. Each row's absolute_time is then:
    file_start_time + video_frame / fps

Outputs:
    results/oft_analysis/<monkey>/<date>/trajectory_realworld.csv
    results/oft_analysis/<monkey>/<date>/movement_bouts.csv
    results/oft_analysis/<monkey>/<date>/trajectory_summary.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


# ── Defaults ───────────────────────────────────────────────────────────────────

MONKEY_ID = 'H1705011'
SESSION_DATE = '2026-05-25'
FPS = 20.0
CAMERAS = ['D1', 'D2', 'D3', 'D4']

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_BASE = PROJECT_ROOT / 'results' / 'oft_analysis'
CAL_BASE = PROJECT_ROOT / 'results' / 'calibration'
STEREO_CALIB_BASE = PROJECT_ROOT / 'results' / 'stereo_calibration'
PREDICTIONS_BASE = PROJECT_ROOT / 'results' / 'predictions'
DEFAULT_STEREO_CALIB_JSON = STEREO_CALIB_BASE / 'stereo_calibration_checkerboard.json'

# Arena quadrant bounds (cm) for camera-specific filtering and overlap logic.
CAMERA_QUADRANTS = {
    'D1': {'x_min': 0,   'x_max': 161, 'y_min': 0,   'y_max': 161},  # top-left
    'D2': {'x_min': 161, 'x_max': 322, 'y_min': 0,   'y_max': 161},  # top-right
    'D3': {'x_min': 0,   'x_max': 161, 'y_min': 161, 'y_max': 322},  # bottom-left
    'D4': {'x_min': 161, 'x_max': 322, 'y_min': 161, 'y_max': 322},  # bottom-right
}

MAX_INTERP_FRAMES = 40  # 2 s at 20 fps
QUADRANT_BOUNDARY_MARGIN_CM = 15.0
ARENA_EDGE_MARGIN_CM = 10.0
VELOCITY_THRESHOLD = 5.0
MIN_BOUT_DURATION = 0.5
NO_MONKEY_SENTINEL = -1

PRED_TS_RE = re.compile(r'^(D0[1-4])_(\d{14})_.*_preds\.csv$', re.IGNORECASE)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def pixel_to_cm(pixel_xy, H: np.ndarray):
    pt = np.array([[[float(pixel_xy[0]), float(pixel_xy[1])]]], dtype=np.float64)
    r = cv2.perspectiveTransform(pt, H)
    return float(r[0, 0, 0]), float(r[0, 0, 1])


def undistort_pixel(pixel_xy, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
    pts = np.array([[[float(pixel_xy[0]), float(pixel_xy[1])]]], dtype=np.float64)
    und = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=camera_matrix)
    return float(und[0, 0, 0]), float(und[0, 0, 1])


def scale_camera_matrix(camera_matrix: np.ndarray,
                        source_size: Optional[tuple[int, int]],
                        target_size: Optional[tuple[int, int]]) -> np.ndarray:
    if source_size is None or target_size is None or tuple(source_size) == tuple(target_size):
        return camera_matrix.copy()

    src_w, src_h = source_size
    dst_w, dst_h = target_size
    if src_w <= 0 or src_h <= 0:
        return camera_matrix.copy()

    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    k_scaled = camera_matrix.copy().astype(np.float64)
    k_scaled[0, 0] *= sx
    k_scaled[0, 2] *= sx
    k_scaled[1, 1] *= sy
    k_scaled[1, 2] *= sy
    return k_scaled


def adjust_to_arena_bounds(x_cm: float, y_cm: float, mode: str = 'clip') -> tuple[float, float]:
    if mode == 'none':
        return float(x_cm), float(y_cm)

    if mode in ('clip', 'project'):
        # For an axis-aligned rectangle, projecting to the nearest point on the
        # arena is equivalent to clamping each coordinate independently.
        return float(np.clip(x_cm, 0, 322)), float(np.clip(y_cm, 0, 322))

    raise ValueError(f'Unknown arena edge adjustment mode: {mode}')


# ── File timestamp parsing ─────────────────────────────────────────────────────

def parse_prediction_start_time(csv_path: Path) -> Optional[pd.Timestamp]:
    match = PRED_TS_RE.match(csv_path.name)
    if match is None:
        fallback = re.search(r'(\d{14})', csv_path.name)
        if fallback is None:
            return None
        stamp = fallback.group(1)
    else:
        stamp = match.group(2)

    return pd.to_datetime(stamp, format='%Y%m%d%H%M%S')


# ── Load calibration ──────────────────────────────────────────────────────────

def load_stereo_intrinsics(stereo_calib_json: Path, cameras: List[str]):
    models: Dict[str, Optional[dict]] = {}
    if not stereo_calib_json.exists():
        print(f'  ⚠  No stereo calibration JSON found at {stereo_calib_json} — using raw pixels')
        for cam in cameras:
            models[cam] = None
        return models

    with open(stereo_calib_json) as f:
        data = json.load(f)

    intrinsics = data.get('intrinsics', {})
    for cam in cameras:
        intr = intrinsics.get(f'D0{cam[-1]}') or intrinsics.get(cam)
        if not intr or not intr.get('success', False):
            models[cam] = None
            continue

        models[cam] = {
            'camera_matrix': np.array(intr['camera_matrix'], dtype=np.float64),
            'dist_coeffs': np.array(intr['dist_coeffs'], dtype=np.float64).reshape(-1, 1),
            'image_size_wh': tuple(intr.get('image_size', [0, 0])) if intr.get('image_size') else None,
        }

    print(f'  ✓  Loaded stereo intrinsics from {stereo_calib_json.name}')
    return models


def load_homographies(cal_dir: Path,
                      cameras: List[str],
                      stereo_calib_json: Optional[Path] = None,
                      target_image_sizes: Optional[Dict[str, tuple[int, int]]] = None):
    Hs: Dict[str, Optional[np.ndarray]] = {}
    pixel_bounds: Dict[str, Optional[dict]] = {}
    camera_models: Dict[str, Optional[dict]] = {cam: None for cam in cameras}
    stereo_models: Dict[str, Optional[dict]] = {cam: None for cam in cameras}

    if stereo_calib_json is not None:
        stereo_models = load_stereo_intrinsics(stereo_calib_json, cameras)

    for cam in cameras:
        cam_long = f'D0{cam[-1]}'
        cal_candidates = [
            cal_dir / f'calibration_{cam}.json',
            cal_dir / f'calibration_{cam_long}.json',
        ]
        cal_file = next((p for p in cal_candidates if p.exists()), None)
        if cal_file is None:
            tried = ', '.join(p.name for p in cal_candidates)
            print(f'  ⚠  No calibration file for {cam} (tried: {tried}) — camera will be skipped')
            Hs[cam] = None
            pixel_bounds[cam] = None
            continue

        with open(cal_file) as f:
            data = json.load(f)

        floor_pts = [p for p in data['labeled_points']
                     if p.get('pixel') is not None and p['world_cm'][2] == 0]
        if len(floor_pts) < 4:
            print(f'  ⚠  Calibration for {cam} has fewer than 4 floor points — camera will be skipped')
            Hs[cam] = None
            pixel_bounds[cam] = None
            continue

        stereo_model = stereo_models.get(cam)
        target_size = target_image_sizes.get(cam) if target_image_sizes else None
        if stereo_model is not None:
            base_size = stereo_model.get('image_size_wh')
            k_scaled = scale_camera_matrix(stereo_model['camera_matrix'], base_size, target_size)
            dist = stereo_model['dist_coeffs']
            und_floor_pts = [undistort_pixel(p['pixel'], k_scaled, dist) for p in floor_pts]
            src = np.array(und_floor_pts, dtype=np.float64)
            dst = np.array([[p['world_cm'][0], p['world_cm'][1]] for p in floor_pts], dtype=np.float64)
            h_mat, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if h_mat is None:
                print(f'  ⚠  {cam}: failed to recompute undistorted homography — camera will be skipped')
                Hs[cam] = None
                pixel_bounds[cam] = None
                camera_models[cam] = None
                continue

            Hs[cam] = h_mat
            camera_models[cam] = {
                'camera_matrix': k_scaled,
                'dist_coeffs': dist,
                'image_size_wh': target_size or base_size,
            }
            xs = [pt[0] for pt in und_floor_pts]
            ys = [pt[1] for pt in und_floor_pts]
            pixel_bounds[cam] = {
                'x_min': min(xs),
                'x_max': max(xs),
                'y_min': min(ys),
                'y_max': max(ys),
            }
            print(f'  ✓  {cam}: homography loaded + undistorted with stereo intrinsics')
            continue

        H_raw = data.get('floor_homography')
        if H_raw is None:
            print(f'  ⚠  Calibration for {cam} has no floor_homography — camera will be skipped')
            Hs[cam] = None
            pixel_bounds[cam] = None
            continue

        Hs[cam] = np.array(H_raw, dtype=np.float64)
        xs = [p['pixel'][0] for p in floor_pts]
        ys = [p['pixel'][1] for p in floor_pts]
        pixel_bounds[cam] = {
            'x_min': min(xs),
            'x_max': max(xs),
            'y_min': min(ys),
            'y_max': max(ys),
        }

        err = data.get('floor_reproj_error_cm', float('nan'))
        print(f'  ✓  {cam}: homography loaded  (reprojection error: {err:.2f} cm)')

    return Hs, pixel_bounds, camera_models


# ── Load predictions ───────────────────────────────────────────────────────────

def load_prediction_csvs(predictions_dir: Path, cameras: List[str], fps: float = FPS):
    if not predictions_dir.exists():
        raise FileNotFoundError(f'Predictions directory not found: {predictions_dir}')

    dfs: List[pd.DataFrame] = []
    image_sizes: Dict[str, tuple[int, int]] = {}

    for cam in cameras:
        cam_num = cam[1]
        csv_files = sorted(predictions_dir.glob(f'D0{cam_num}_*_preds.csv'))
        if not csv_files:
            print(f'  ⚠  No prediction CSVs for {cam}')
            continue

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            if df.empty:
                continue

            if 'has_detection' in df.columns:
                df = df[df['has_detection'] == True].copy()  # noqa: E712
            else:
                df = df.copy()

            if df.empty:
                continue

            if 'bbox_cx' not in df.columns or 'bbox_cy' not in df.columns:
                raise ValueError(f'{csv_file}: missing bbox_cx / bbox_cy columns')
            if 'video_frame' not in df.columns:
                raise ValueError(f'{csv_file}: missing video_frame column')

            start_time = parse_prediction_start_time(csv_file)
            if start_time is None:
                raise ValueError(
                    f'{csv_file.name}: cannot parse start timestamp from filename '
                    '(expected YYYYMMDDHHMMSS in the name)'
                )

            if cam not in image_sizes and 'sampled_image_path' in df.columns:
                sample_series = df['sampled_image_path'].dropna().astype(str)
                for sample_path_str in sample_series:
                    sample_path = Path(sample_path_str)
                    if not sample_path.exists():
                        continue
                    sample_img = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
                    if sample_img is not None:
                        image_sizes[cam] = (sample_img.shape[1], sample_img.shape[0])
                        df['source_image_size_wh'] = [image_sizes[cam]] * len(df)
                        break

            df['absolute_time'] = start_time + pd.to_timedelta(df['video_frame'] / fps, unit='s')
            df['camera'] = cam
            df['source_csv'] = csv_file.name
            dfs.append(df)
            print(f'  ✓  {cam} ({csv_file.name}): {len(df)} detected frames')

    return dfs, image_sizes


# ── Projection to real world ──────────────────────────────────────────────────

def apply_homographies(label_dfs: List[pd.DataFrame],
                       Hs: Dict[str, Optional[np.ndarray]],
                       pixel_bounds: Dict[str, Optional[dict]],
                       camera_models: Optional[Dict[str, Optional[dict]]] = None,
                       filter_out_of_bounds: bool = True,
                       pixel_margin: float = 100.0,
                       world_margin_cm: float = 30.0,
                       pixel_x_col: str = 'centroid_x',
                       pixel_y_col: str = 'centroid_y',
                       edge_adjust_mode: str = 'clip') -> pd.DataFrame:
    rows = []
    n_adjusted = 0

    for df in label_dfs:
        if df.empty:
            continue

        cam = df['camera'].iloc[0]
        H = Hs.get(cam)
        if H is None:
            continue

        model = camera_models.get(cam) if camera_models else None
        pbounds = pixel_bounds.get(cam)
        wbounds = CAMERA_QUADRANTS.get(cam)

        for _, row in df.iterrows():
            px_raw, py_raw = row[pixel_x_col], row[pixel_y_col]
            if pd.isna(px_raw) or pd.isna(py_raw):
                continue

            px, py = float(px_raw), float(py_raw)
            if model is not None:
                px, py = undistort_pixel((px, py), model['camera_matrix'], model['dist_coeffs'])

            x_cm, y_cm = pixel_to_cm((px, py), H)
            boundary_distance_cm = min(abs(x_cm - 161.0), abs(y_cm - 161.0))

            x_cm_adj, y_cm_adj = adjust_to_arena_bounds(x_cm, y_cm, mode=edge_adjust_mode)
            if x_cm_adj != x_cm or y_cm_adj != y_cm:
                n_adjusted += 1
            x_cm, y_cm = x_cm_adj, y_cm_adj

            rows.append({
                'absolute_time': row['absolute_time'],
                'camera': cam,
                'x_cm': x_cm,
                'y_cm': y_cm,
                'pixel_x_raw': float(px_raw),
                'pixel_y_raw': float(py_raw),
                'pixel_x': float(px),
                'pixel_y': float(py),
                'boundary_distance_cm': float(boundary_distance_cm),
            })

    if n_adjusted > 0:
        print(f'  ℹ  Adjusted {n_adjusted} points to arena bounds [0, 322] cm')

    return pd.DataFrame(rows)


# ── Fuse overlaps ─────────────────────────────────────────────────────────────

def fuse_overlapping(cm_df: pd.DataFrame,
                     time_tolerance_sec: float = 0.1,
                     boundary_margin_cm: float = QUADRANT_BOUNDARY_MARGIN_CM) -> pd.DataFrame:
    if cm_df.empty:
        return cm_df

    cm_df = cm_df.sort_values('absolute_time').reset_index(drop=True)
    fused_rows = []
    current_group = [cm_df.iloc[0].to_dict()]

    for i in range(1, len(cm_df)):
        row = cm_df.iloc[i]
        prev_time = current_group[-1]['absolute_time']
        if (row['absolute_time'] - prev_time).total_seconds() <= time_tolerance_sec:
            current_group.append(row.to_dict())
        else:
            fused_rows.append(_finalize_group(current_group, boundary_margin_cm))
            current_group = [row.to_dict()]

    fused_rows.append(_finalize_group(current_group, boundary_margin_cm))

    fused = pd.DataFrame(fused_rows).reset_index(drop=True)
    fused['time_sec'] = (fused['absolute_time'] - fused['absolute_time'].iloc[0]).dt.total_seconds()
    return fused


def _finalize_group(group: List[dict], boundary_margin_cm: float) -> dict:
    if len(group) == 1:
        return group[0]

    min_boundary = min(float(r.get('boundary_distance_cm', np.inf)) for r in group)
    if min_boundary <= boundary_margin_cm:
        return {
            'absolute_time': group[0]['absolute_time'],
            'camera': ','.join(sorted(set(r['camera'] for r in group))),
            'x_cm': np.mean([r['x_cm'] for r in group]),
            'y_cm': np.mean([r['y_cm'] for r in group]),
            'pixel_x_raw': float('nan'),
            'pixel_y_raw': float('nan'),
            'pixel_x': float('nan'),
            'pixel_y': float('nan'),
            'boundary_distance_cm': float(min_boundary),
        }

    return max(group, key=lambda r: float(r.get('boundary_distance_cm', -np.inf)))


# ── Interpolation and kinematics ──────────────────────────────────────────────

def interpolate_gaps(df: pd.DataFrame, fps: float, max_gap_sec: float = None) -> pd.DataFrame:
    if df.empty:
        return df

    if max_gap_sec is None:
        max_gap_sec = MAX_INTERP_FRAMES / fps
    if max_gap_sec == 0:
        return df

    rows = df.to_dict('records')
    filled = [rows[0]]

    for i in range(1, len(rows)):
        prev, curr = rows[i - 1], rows[i]
        gap_sec = curr['time_sec'] - prev['time_sec']
        if 0 < gap_sec <= max_gap_sec:
            n_steps = int(gap_sec * fps)
            for step in range(1, n_steps):
                frac = step / n_steps
                filled.append({
                    'absolute_time': prev['absolute_time'] + pd.Timedelta(seconds=step / fps),
                    'time_sec': prev['time_sec'] + step / fps,
                    'camera': 'interpolated',
                    'x_cm': prev['x_cm'] + frac * (curr['x_cm'] - prev['x_cm']),
                    'y_cm': prev['y_cm'] + frac * (curr['y_cm'] - prev['y_cm']),
                    'pixel_x': float('nan'),
                    'pixel_y': float('nan'),
                    'pixel_x_raw': float('nan'),
                    'pixel_y_raw': float('nan'),
                    'boundary_distance_cm': float('nan'),
                })
        filled.append(curr)

    return pd.DataFrame(filled).reset_index(drop=True)


def compute_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dx = df['x_cm'].diff()
    dy = df['y_cm'].diff()
    dt = df['time_sec'].diff()
    df['distance_cm'] = np.sqrt(dx**2 + dy**2)
    df['velocity_cm_s'] = (df['distance_cm'] / dt).replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def detect_bouts(df: pd.DataFrame,
                 vel_thresh: float = VELOCITY_THRESHOLD,
                 min_dur: float = MIN_BOUT_DURATION) -> pd.DataFrame:
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
                bouts.append({'start_sec': start, 'end_sec': row['time_sec'], 'duration_sec': dur})
            start = None
    return pd.DataFrame(bouts)


# ── Summary ───────────────────────────────────────────────────────────────────

def compute_summary(traj: pd.DataFrame, bouts: pd.DataFrame) -> dict:
    total_dur = traj['time_sec'].max() - traj['time_sec'].min()
    total_dist = traj['distance_cm'].sum()
    mean_vel = traj['velocity_cm_s'].mean()
    moving_mask = traj['velocity_cm_s'] > VELOCITY_THRESHOLD
    mean_vel_moving = traj.loc[moving_mask, 'velocity_cm_s'].mean() if moving_mask.any() else 0.0
    immob_frames = (traj['velocity_cm_s'] <= VELOCITY_THRESHOLD).sum()
    immob_pct = 100.0 * immob_frames / len(traj) if len(traj) > 0 else 0.0

    return {
        'total_duration_sec': round(float(total_dur), 2),
        'total_distance_cm': round(float(total_dist), 2),
        'mean_velocity_cm_s': round(float(mean_vel), 3),
        'mean_velocity_moving_cm_s': round(float(mean_vel_moving), 3),
        'max_velocity_cm_s': round(float(traj['velocity_cm_s'].max()), 3),
        'immobility_pct': round(immob_pct, 2),
        'n_movement_bouts': len(bouts),
        'mean_bout_duration_sec': round(float(bouts['duration_sec'].mean()), 3) if len(bouts) > 0 else 0.0,
        'total_active_time_sec': round(float(bouts['duration_sec'].sum()), 2) if len(bouts) > 0 else 0.0,
        'n_labeled_frames': int((traj['camera'] != 'interpolated').sum()),
        'n_interpolated_frames': int((traj['camera'] == 'interpolated').sum()),
        'cameras_used': sorted(set(c for cams in traj['camera'] for c in cams.split(',') if c != 'interpolated')),
        'velocity_threshold_cm_s': VELOCITY_THRESHOLD,
        'max_interp_frames': MAX_INTERP_FRAMES,
    }


# ── Main prediction pipeline ──────────────────────────────────────────────────

def reconstruct_predictions(monkey_id: str = MONKEY_ID,
                            session_date: str = SESSION_DATE,
                            cameras: List[str] = None,
                            fps: float = FPS,
                            max_interp: int = MAX_INTERP_FRAMES,
                            cal_dir: Optional[Path] = None,
                            predictions_dir: Optional[Path] = None,
                            stereo_calib_json: Optional[Path] = None,
                            output_dir: Optional[Path] = None,
                            pixel_margin: float = 100.0,
                            world_margin_cm: float = 10.0,
                            edge_adjust_mode: str = 'clip') -> pd.DataFrame:
    cameras = cameras or CAMERAS
    predictions_dir = predictions_dir or (PREDICTIONS_BASE / monkey_id / session_date)
    output_dir = output_dir or (RESULTS_BASE / monkey_id / session_date)
    cal_dir = cal_dir or CAL_BASE
    stereo_calib_json = stereo_calib_json or DEFAULT_STEREO_CALIB_JSON

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f' Building bbox-center trajectory for {monkey_id} / {session_date}')
    print(f"{'='*60}")

    print('\n── Loading prediction CSVs ──')
    pred_dfs, image_sizes = load_prediction_csvs(predictions_dir, cameras, fps)
    if not pred_dfs:
        print('ERROR: No prediction CSVs were loaded.')
        return pd.DataFrame()

    print('\n── Loading homographies ──')
    Hs, pixel_bounds, camera_models = load_homographies(
        cal_dir,
        cameras,
        stereo_calib_json=stereo_calib_json,
        target_image_sizes=image_sizes,
    )

    print('\n── Converting bbox center → cm ──')
    cm_df = apply_homographies(
        pred_dfs,
        Hs,
        pixel_bounds,
        camera_models=camera_models,
        filter_out_of_bounds=True,
        pixel_margin=pixel_margin,
        world_margin_cm=world_margin_cm,
        pixel_x_col='bbox_cx',
        pixel_y_col='bbox_cy',
        edge_adjust_mode=edge_adjust_mode,
    )
    if cm_df.empty:
        print('ERROR: No data after homography conversion. Check calibration and predictions.')
        return pd.DataFrame()
    print(f'  {len(cm_df)} total detected positions across all cameras')

    print('\n── Fusing boundary-overlap frames ──')
    traj = fuse_overlapping(cm_df)
    n_multi = traj['camera'].astype(str).str.contains(',', na=False).sum()
    print(f'  {len(traj)} unique frames  ({n_multi} averaged from two or more cameras)')

    print(f'\n── Interpolating gaps (max {max_interp} frames = {max_interp/fps:.1f}s) ──')
    traj = interpolate_gaps(traj, fps, max_interp)
    n_interp = (traj['camera'] == 'interpolated').sum()
    print(f'  {n_interp} frames interpolated  →  {len(traj)} total frames')

    print('\n── Computing kinematics ──')
    traj = compute_kinematics(traj)
    bouts = detect_bouts(traj)
    summary = compute_summary(traj, bouts)

    traj_path = output_dir / 'trajectory_realworld.csv'
    bouts_path = output_dir / 'movement_bouts.csv'
    summ_path = output_dir / 'trajectory_summary.json'

    traj.to_csv(traj_path, index=False)
    bouts.to_csv(bouts_path, index=False)
    with open(summ_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(' Summary')
    print(f"{'='*60}")
    for k, v in summary.items():
        print(f'  {k:<35} {v}')
    print(f"\n✅  trajectory_realworld.csv  → {traj_path}")
    print(f'    movement_bouts.csv        → {bouts_path}')
    print(f'    trajectory_summary.json   → {summ_path}')

    return traj


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Reconstruct real-world trajectory from RT-DETR bbox predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--monkey', default=MONKEY_ID, help='Monkey ID')
    parser.add_argument('--date', default=SESSION_DATE, help='Session date (YYYY-MM-DD)')
    parser.add_argument('--cameras', default=None, nargs='+', choices=['D1', 'D2', 'D3', 'D4'],
                        help='Cameras to include (default: all 4)')
    parser.add_argument('--fps', type=float, default=FPS)
    parser.add_argument('--max-interp', type=int, default=MAX_INTERP_FRAMES,
                        help='Max gap in frames to interpolate (default: 40 = 2s)')
    parser.add_argument('--cal-dir', default=None,
                        help=f'Calibration dir (default: {CAL_BASE})')
    parser.add_argument('--stereo-calibration-json', default=None,
                        help=f'Stereo calibration JSON with intrinsics (default: {DEFAULT_STEREO_CALIB_JSON})')
    parser.add_argument('--predictions-dir', default=None,
                        help='Prediction dir (default: results/predictions/<monkey>/<date>/)')
    parser.add_argument('--output-dir', default=None,
                        help='Output dir (default: results/oft_analysis/<monkey>/<date>/)')
    parser.add_argument('--pixel-margin', type=float, default=100.0,
                        help='Pixel tolerance around calibrated bounds before filtering')
    parser.add_argument('--world-margin-cm', type=float, default=10.0,
                        help='World-space tolerance around quadrant bounds before filtering')
    parser.add_argument('--edge-adjust', choices=['clip', 'project', 'none'], default='project',
                        help='How to bring out-of-arena points back into the open field')
    args = parser.parse_args()

    reconstruct_predictions(
        monkey_id=args.monkey,
        session_date=args.date,
        cameras=args.cameras,
        fps=args.fps,
        max_interp=args.max_interp,
        cal_dir=Path(args.cal_dir) if args.cal_dir else None,
        predictions_dir=Path(args.predictions_dir) if args.predictions_dir else None,
        stereo_calib_json=Path(args.stereo_calibration_json) if args.stereo_calibration_json else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        pixel_margin=args.pixel_margin,
        world_margin_cm=args.world_margin_cm,
        edge_adjust_mode=args.edge_adjust,
    )


if __name__ == '__main__':
    main()
