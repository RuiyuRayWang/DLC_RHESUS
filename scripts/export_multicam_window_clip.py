#!/usr/bin/env python3
"""
Export a stitched 4-camera review clip with bbox + trajectory overlays.

The exporter reuses rendering logic from review_multicam_predictions.py and
writes an MP4 for a selected time window, typically from:
  results/oft_analysis/<monkey>/<date>/selected_time_window.json

Example:
  conda activate video_tracking
  python scripts/export_multicam_window_clip.py \
    --monkey H1705011 \
    --date 2026-05-27 \
    --speed-up 4
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd

from review_multicam_predictions import (
    ARTIFACT_BRIDGE_MAX_SEC_DEFAULT,
    ARTIFACT_DVDT_THRESH_DEFAULT,
    ARTIFACT_VEL_THRESH_DEFAULT,
    FPS_DEFAULT,
    PROJECT_ROOT,
    TrajectoryIndex,
    build_artifact_tidx_map,
    build_bundles,
    make_canvas,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export stitched multicamera clip for selected time window")
    p.add_argument("--monkey", required=True, help="Monkey ID, e.g. H1705011")
    p.add_argument("--date", required=True, help="Session date, e.g. 2026-05-27")
    p.add_argument("--frames-root", type=Path, default=PROJECT_ROOT / "tmp" / "rtdetr_frames_2hz")
    p.add_argument("--pred-root", type=Path, default=PROJECT_ROOT / "results" / "predictions")
    p.add_argument("--traj-root", type=Path, default=PROJECT_ROOT / "results" / "oft_analysis")
    p.add_argument("--window-json", type=Path, default=None, help="Path to selected_time_window.json")
    p.add_argument("--start-abs", default=None, help="Override window start absolute time (ISO format)")
    p.add_argument("--end-abs", default=None, help="Override window end absolute time (ISO format)")
    p.add_argument("--fps", type=float, default=FPS_DEFAULT, help="Video FPS used for timestamp reconstruction")
    p.add_argument("--output-fps", type=float, default=20.0, help="Output video FPS")
    p.add_argument("--speed-up", type=float, default=4.0, help="Playback speed-up factor")
    p.add_argument("--no-tail", action="store_true", help="Disable trajectory tail rendering")
    p.add_argument("--hide-bbox", action="store_true", help="Disable bbox overlay rendering")
    p.add_argument("--artifact-velocity-thresh", type=float, default=ARTIFACT_VEL_THRESH_DEFAULT)
    p.add_argument("--artifact-dvdt-thresh", type=float, default=ARTIFACT_DVDT_THRESH_DEFAULT)
    p.add_argument("--artifact-bridge-max-sec", type=float, default=ARTIFACT_BRIDGE_MAX_SEC_DEFAULT)
    p.add_argument("--output", type=Path, default=None, help="Output MP4 path")
    return p.parse_args()


def parse_iso_datetime(text: str) -> datetime:
    t = text.strip()
    if t.endswith("Z"):
        t = t[:-1]
    try:
        return datetime.fromisoformat(t)
    except Exception as exc:
        raise ValueError(f"invalid ISO datetime: {text}") from exc


def resolve_window_bounds(
    traj_path: Path,
    window_json: Optional[Path],
    start_abs: Optional[str],
    end_abs: Optional[str],
) -> tuple[datetime, datetime]:
    sdt: Optional[datetime] = parse_iso_datetime(start_abs) if start_abs else None
    edt: Optional[datetime] = parse_iso_datetime(end_abs) if end_abs else None

    if sdt is None or edt is None:
        if window_json is None:
            window_json = traj_path.parent / "selected_time_window.json"
        if not window_json.exists():
            raise FileNotFoundError(
                f"selected window file not found: {window_json}. "
                "Provide --window-json or --start-abs/--end-abs."
            )
        with open(window_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        sdt = sdt or parse_iso_datetime(str(payload.get("start_abs", "")))
        edt = edt or parse_iso_datetime(str(payload.get("end_abs", "")))

    if sdt is None or edt is None:
        raise ValueError("could not resolve start/end absolute times")
    if edt <= sdt:
        raise ValueError(f"invalid window: end <= start ({sdt} .. {edt})")
    return sdt, edt


def gather_window_timeline_indices(
    bundles,
    start_dt: datetime,
    end_dt: datetime,
) -> list[tuple[int, int, datetime]]:
    out: list[tuple[int, int, datetime]] = []
    seen_ns: set[int] = set()
    lo = start_dt.timestamp()
    hi = end_dt.timestamp()

    for bidx, bundle in enumerate(bundles):
        for tidx, when in enumerate(bundle.timeline):
            ts = when.timestamp()
            if ts < lo or ts > hi:
                continue
            ns = int(round(ts * 1e9))
            if ns in seen_ns:
                continue
            seen_ns.add(ns)
            out.append((bidx, tidx, when))

    out.sort(key=lambda x: x[2])
    return out


def speed_filter_indices(
    timeline_rows: list[tuple[int, int, datetime]],
    speed_up: float,
    out_fps: float,
) -> list[tuple[int, int, datetime]]:
    if not timeline_rows:
        return []
    if speed_up <= 0 or out_fps <= 0:
        return timeline_rows

    # Emit source frames with this source-time spacing to realize speed-up.
    source_step_sec = float(speed_up) / float(out_fps)
    kept = [timeline_rows[0]]
    last_t = timeline_rows[0][2].timestamp()

    for row in timeline_rows[1:]:
        t = row[2].timestamp()
        if t >= last_t + source_step_sec:
            kept.append(row)
            last_t = t

    if kept[-1] != timeline_rows[-1]:
        kept.append(timeline_rows[-1])
    return kept


def make_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter for: {path}")
    return writer


def run(args: argparse.Namespace) -> Path:
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.output_fps <= 0:
        raise ValueError("--output-fps must be > 0")
    if args.speed_up <= 0:
        raise ValueError("--speed-up must be > 0")

    monkey = args.monkey
    date = args.date
    frames_dir = args.frames_root.expanduser().resolve() / monkey / date
    pred_dir = args.pred_root.expanduser().resolve() / monkey / date
    traj_path = args.traj_root.expanduser().resolve() / monkey / date / "trajectory_realworld.csv"

    if not frames_dir.exists():
        raise FileNotFoundError(f"frames dir not found: {frames_dir}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"prediction dir not found: {pred_dir}")
    if not traj_path.exists():
        raise FileNotFoundError(f"trajectory file not found: {traj_path}")

    window_json = args.window_json.expanduser().resolve() if args.window_json is not None else None
    start_dt, end_dt = resolve_window_bounds(traj_path, window_json, args.start_abs, args.end_abs)

    print(f"[INFO] Building synchronized bundles for {monkey} / {date} ...")
    bundles = build_bundles(pred_dir=pred_dir, frames_dir=frames_dir, fps=float(args.fps))

    print(f"[INFO] Loading trajectory: {traj_path}")
    traj_index = TrajectoryIndex(traj_path)
    artifact_csv = traj_path.parent / "trajectory_artifact_candidates.csv"
    artifact_tidx = build_artifact_tidx_map(
        bundles,
        traj_index,
        velocity_thresh=float(args.artifact_velocity_thresh),
        dvdt_thresh=float(args.artifact_dvdt_thresh),
        bridge_max_sec=float(args.artifact_bridge_max_sec),
        artifact_csv_path=artifact_csv,
    )

    rows = gather_window_timeline_indices(bundles, start_dt, end_dt)
    if not rows:
        raise RuntimeError("no synchronized timeline points found in selected time window")

    rows_sped = speed_filter_indices(rows, speed_up=float(args.speed_up), out_fps=float(args.output_fps))
    if not rows_sped:
        raise RuntimeError("no frames selected after speed filter")

    if args.output is None:
        ts0 = start_dt.strftime("%H%M%S")
        ts1 = end_dt.strftime("%H%M%S")
        out_path = traj_path.parent / f"multicam_overlay_{monkey}_{date}_{ts0}_{ts1}_{int(round(args.speed_up))}x.mp4"
    else:
        out_path = args.output.expanduser().resolve()

    print(f"[INFO] Window: {start_dt.isoformat()} -> {end_dt.isoformat()}")
    print(f"[INFO] Source points in window: {len(rows)}")
    print(f"[INFO] Export frames after {args.speed_up}x speed-up: {len(rows_sped)}")
    print(f"[INFO] Output FPS: {args.output_fps}")
    print(f"[INFO] Writing: {out_path}")

    first_bidx, first_tidx, _ = rows_sped[0]
    first_canvas, _, _, _ = make_canvas(
        bundle=bundles[first_bidx],
        time_idx=int(first_tidx),
        show_bbox=not args.hide_bbox,
        traj_index=traj_index,
        show_tail=(not args.no_tail),
        active_cameras=set(),
        trajectory_mask_mode=False,
        time_input_text="",
        time_input_active=False,
        artifact_mode=False,
        artifact_pos=0,
        artifact_total=0,
        status_text="",
    )
    h, w = first_canvas.shape[:2]
    writer = make_writer(out_path, width=w, height=h, fps=float(args.output_fps))

    try:
        for i, (bidx, tidx, _) in enumerate(rows_sped, start=1):
            bundle = bundles[bidx]
            canvas, _, _, _ = make_canvas(
                bundle=bundle,
                time_idx=int(tidx),
                show_bbox=not args.hide_bbox,
                traj_index=traj_index,
                show_tail=(not args.no_tail),
                active_cameras=set(),
                trajectory_mask_mode=False,
                time_input_text="",
                time_input_active=False,
                artifact_mode=False,
                artifact_pos=0,
                artifact_total=0,
                status_text="",
            )
            writer.write(canvas)

            if i % 200 == 0 or i == len(rows_sped):
                print(f"[INFO] wrote {i}/{len(rows_sped)} frames")
    finally:
        writer.release()

    n_frames = len(rows_sped)
    out_dur = float(n_frames) / float(args.output_fps)
    src_dur = (end_dt.timestamp() - start_dt.timestamp())
    realized = (src_dur / out_dur) if out_dur > 0 else float("nan")
    print(f"[DONE] Exported clip: {out_path}")
    print(f"[DONE] Approx output duration: {out_dur:.2f}s, realized speed-up: {realized:.2f}x")

    # Surface basic metadata for tooling and downstream scripts.
    sidecar = out_path.with_suffix(".json")
    meta = {
        "monkey_id": monkey,
        "session_date": date,
        "start_abs": start_dt.isoformat(),
        "end_abs": end_dt.isoformat(),
        "speed_up_requested": float(args.speed_up),
        "speed_up_realized": round(float(realized), 4) if pd.notna(realized) else None,
        "output_fps": float(args.output_fps),
        "n_source_points": int(len(rows)),
        "n_export_frames": int(n_frames),
        "output_video": str(out_path),
    }
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[DONE] Metadata sidecar: {sidecar}")

    return out_path


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
