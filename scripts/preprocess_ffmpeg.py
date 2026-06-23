#!/usr/bin/env python3
"""
Preprocess monkey-visible video frames for DeepLabCut inference.

This script:

1. Reads a visibility CSV.
2. Parses visible 1-second bins.
3. Sequentially decodes the source HEVC video.
4. Saves only sampled visible frames as PNG.
5. Rebuilds a DLC-compatible H.264 MP4 using ffmpeg.
6. Saves metadata mapping rebuilt frames back to original video frames/times.

It does NOT run DeepLabCut.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import subprocess

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "oft_rhp" / "visible_dlc_preprocessed"

ANNOTATION_BIN_SECONDS = 1.0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess monkey-visible frames into a DLC-ready H.264 video.")
    parser.add_argument("--annotations", required=True, help="Path to visibility annotation CSV.",)
    parser.add_argument("--video", required=True, help="Path to source video.",)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.",)
    parser.add_argument("--target-fps", type=float, default=5.0, help="Sampling FPS for visible frames.",)
    parser.add_argument("--camera", default=None, help="Optional camera filter, e.g. D1. If omitted, inferred from video name.",)
    parser.add_argument("--keep-pngs", action="store_true", help="Keep extracted PNG frames after rebuilding video.",)
    return parser.parse_args()


def infer_camera_from_name(
    path_or_name: str | Path,
) -> str:
    """
    Infer camera ID from a video or CSV filename.
    
    Examples:
        D01_20260618131058.mp4      -> D1
        D01_20260618131058_vis.csv  -> D1
        D02_20260618131058.mp4      -> D2
        D03_20260618131058.mp4      -> D3
        D04_20260618131058.mp4      -> D4
    """
    
    name = Path(path_or_name).name
    
    prefix_to_camera = {
        "D01": "D1",
        "D02": "D2",
        "D03": "D3",
        "D04": "D4",
    }
    
    for prefix, camera in prefix_to_camera.items():
        if name.startswith(prefix):
            return camera
    
    raise ValueError(f"Could not infer camera from filename: {name}")


def parse_bool_series(
    s: pd.Series,
) -> pd.Series:
    """
    Robustly parse True/False values from CSV.
    """
    
    if s.dtype == bool:
        return s
    
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


def merge_intervals(
    intervals: list[tuple[float, float]],
    tolerance: float = 1e-6,
) -> list[tuple[float, float]]:
    """
    Merge overlapping or directly adjacent intervals.
    """
    
    if not intervals:
        return []
    
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        
        if start <= prev_end + tolerance:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    
    return merged


def load_visible_intervals_from_csv(
    csv_path: str | Path,
    video_file: str,
    camera: str | None = None,
    time_col: str = "time_sec",
    visible_col: str = "monkey_visible",
    video_col: str = "video_file",
    camera_col: str = "camera",
    bin_seconds: float = ANNOTATION_BIN_SECONDS,
) -> list[tuple[float, float]]:
    """
    Load visible intervals for one video.
    
    Each visible CSV row is interpreted as:
        [time_sec, time_sec + bin_seconds)
    """
    
    df = pd.read_csv(csv_path)
    
    if video_col in df.columns:
        df = df[df[video_col] == video_file].copy()
    
    if camera is not None and camera_col in df.columns:
        df = df[df[camera_col] == camera].copy()
    
    df[visible_col] = parse_bool_series(df[visible_col])
    visible = df[df[visible_col]].copy()
    
    if visible.empty:
        return []
    
    visible = visible.sort_values(time_col)
    
    intervals = [
        (float(t), float(t) + bin_seconds)
        for t in visible[time_col].to_numpy()
    ]
    
    return merge_intervals(intervals)


def timestamp_is_inside_intervals(
    t: float,
    intervals: list[tuple[float, float]],
    interval_idx: int,
) -> tuple[bool, int]:
    """
    Efficient interval membership check for monotonically increasing timestamps.
    """
    
    while interval_idx < len(intervals) and t >= intervals[interval_idx][1]:
        interval_idx += 1
    
    if interval_idx >= len(intervals):
        return False, interval_idx
    
    start, end = intervals[interval_idx]
    
    return start <= t < end, interval_idx


def extract_visible_pngs_sequential(
    video_path: str | Path,
    annotation_csv: str | Path,
    frame_dir: str | Path,
    target_fps: float,
    camera: str | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int] | None]:
    """
    Sequentially decode the video and write selected visible frames as PNG.

    This avoids random-access seeking, which is unreliable for some HEVC videos.
    """
    
    video_path = Path(video_path)
    frame_dir = Path(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    
    visible_intervals = load_visible_intervals_from_csv(
        csv_path=annotation_csv,
        video_file=video_path.name,
        camera=camera,
    )
    
    if not visible_intervals:
        print(f"No visible intervals found for {video_path.name}")
        return pd.DataFrame(), None
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if source_fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid source FPS for video: {video_path}")
    
    print("")
    print(f"Video: {video_path.name}")
    print(f"Camera: {camera}")
    print(f"Resolution: {width} x {height}")
    print(f"Source FPS: {source_fps:.6f}")
    print(f"Target FPS: {target_fps:.6f}")
    print(f"Total frames reported by OpenCV: {total_frames}")
    print(f"Visible intervals: {visible_intervals}")
    
    source_step = 1.0 / source_fps
    target_step = 1.0 / target_fps
    
    saved_idx = 0
    source_frame_idx = 0
    interval_idx = 0
    
    current_start, _ = visible_intervals[0]
    next_target_time = current_start
    
    records = []
    
    while True:
        ok, frame = cap.read()
        
        if not ok:
            break
        
        source_time_sec = source_frame_idx / source_fps
        
        inside, interval_idx = timestamp_is_inside_intervals(
            t=source_time_sec,
            intervals=visible_intervals,
            interval_idx=interval_idx,
        )
        
        if interval_idx >= len(visible_intervals):
            break
        
        if inside:
            current_start, current_end = visible_intervals[interval_idx]
            
            if next_target_time < current_start:
                next_target_time = current_start
            
            if next_target_time < current_end:
                should_save = source_time_sec + 0.5 * source_step >= next_target_time
            else:
                should_save = False
            
            if should_save:
                frame_name = f"frame_{saved_idx:06d}.png"
                frame_path = frame_dir / frame_name
                
                cv2.imwrite(str(frame_path), frame)
                
                records.append(
                    {
                        "rebuilt_frame": saved_idx,
                        "camera": camera,
                        "source_video": video_path.name,
                        "source_frame": source_frame_idx,
                        "source_time_sec": source_time_sec,
                        "target_time_sec": next_target_time,
                        "visible_interval_start": current_start,
                        "visible_interval_end": current_end,
                        "frame_file": str(frame_path),
                    }
                )
                
                saved_idx += 1
                next_target_time += target_step
        
        source_frame_idx += 1
        
        if source_frame_idx % 5000 == 0:
            print(
                f"Decoded {source_frame_idx} source frames; "
                f"saved {saved_idx} visible frames"
            )
    
    cap.release()
    
    metadata = pd.DataFrame(records)
    
    video_info = {
        "source_fps": source_fps,
        "target_fps": target_fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
    }
    
    print("")
    print(f"Finished decoding: {video_path.name}")
    print(f"Saved visible PNG frames: {len(metadata)}")
    
    return metadata, video_info


def create_video_from_pngs(
    frame_dir: str | Path,
    output_video: str | Path,
    fps: float,
) -> Path:
    """
    Rebuild a DLC-compatible H.264 MP4 from sequential PNG frames.
    """
    
    frame_dir = Path(frame_dir)
    output_video = Path(output_video)
    
    frame_pattern = str(frame_dir / "frame_%06d.png")
    
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        frame_pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "fast",
        "-crf",
        "18",
        str(output_video),
    ]
    
    print("")
    print("Creating rebuilt H.264 video:")
    print(" ".join(cmd))
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("ffmpeg failed while creating rebuilt video")
    
    return output_video


def preprocess_video(
    video_path: str | Path,
    annotation_csv: str | Path,
    output_dir: str | Path,
    target_fps: float,
    camera: str | None = None,
    keep_pngs: bool = False,
) -> Path:
    """
    Preprocess one video into a DLC-ready H.264 clip and metadata CSV.
    """
    
    video_path = Path(video_path)
    annotation_csv = Path(annotation_csv)
    output_dir = Path(output_dir)
    
    if camera is None:
        camera = infer_camera_from_name(video_path)
    
    video_out_dir = output_dir / video_path.stem
    frame_dir = video_out_dir / "frames"
    rebuilt_video = video_out_dir / f"{video_path.stem}_visible_h264.mp4"
    metadata_csv = video_out_dir / "visible_frame_metadata.csv"
    info_csv = video_out_dir / "video_info.csv"
    
    video_out_dir.mkdir(parents=True, exist_ok=True)
    
    metadata, video_info = extract_visible_pngs_sequential(
        video_path=video_path,
        annotation_csv=annotation_csv,
        frame_dir=frame_dir,
        target_fps=target_fps,
        camera=camera,
    )
    
    if metadata.empty:
        raise RuntimeError(f"No visible frames extracted for {video_path.name}")
    
    metadata.to_csv(metadata_csv, index=False)
    
    if video_info is not None:
        pd.DataFrame([video_info]).to_csv(info_csv, index=False)
    
    create_video_from_pngs(
        frame_dir=frame_dir,
        output_video=rebuilt_video,
        fps=target_fps,
    )
    
    if not keep_pngs:
        for png_path in frame_dir.glob("frame_*.png"):
            png_path.unlink()

        try:
            frame_dir.rmdir()
        except OSError:
            pass
    
    print("")
    print(f"Saved rebuilt video: {rebuilt_video}")
    print(f"Saved metadata CSV: {metadata_csv}")
    print(f"Saved video info CSV: {info_csv}")
    
    return video_out_dir


def main() -> None:
    args = parse_args()
    
    preprocess_video(
        video_path=args.video,
        annotation_csv=args.annotations,
        output_dir=args.output_dir,
        target_fps=args.target_fps,
        camera=args.camera,
        keep_pngs=args.keep_pngs,
    )


if __name__ == "__main__":
    main()
