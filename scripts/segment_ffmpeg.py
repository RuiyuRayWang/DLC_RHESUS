#!/usr/bin/env python3
"""
Segment monkey-visible video frames for DeepLabCut inference.

This script:

1. Reads a visibility CSV.
2. Parses visible 1-second bins.
3. Builds a DLC-compatible H.264 MP4 using ffmpeg.
4. Saves metadata mapping rebuilt frames back to source video frames/times.

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
DOWNSAMPLED_SUFFIX = "_downsampled"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess monkey-visible frames into a DLC-ready H.264 video."
    )
    parser.add_argument("--annotations", required=True, help="Path to visibility annotation CSV.")
    parser.add_argument("--video", required=True, help="Path to source video.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--target-fps", type=float, default=4.0, help="Sampling FPS for visible frames.")
    parser.add_argument("--camera", default=None, help="Optional camera filter, e.g. D1. If omitted, inferred from video name.")
    parser.add_argument("--threads", type=int, default=0, help="Number of threads for ffmpeg.")
    
    parser.add_argument(
        "--annotation-video-file",
        default=None,
        help=(
            "Video filename to match in the annotation CSV video_file column. "
            "If omitted, this is inferred from --video. For *_downsampled.mp4 inputs, "
            "the default inferred annotation video file strips the _downsampled suffix."
        ),
    )
    
    parser.add_argument(
        "--downsampled-suffix",
        default=DOWNSAMPLED_SUFFIX,
        help="Suffix to strip when inferring annotation video_file from a downsampled video name.",
    )
    
    return parser.parse_args()


def infer_camera_from_name(path_or_name: str | Path) -> str:
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


def infer_annotation_video_file(
    video_path: str | Path,
    annotation_video_file: str | None = None,
    downsampled_suffix: str = DOWNSAMPLED_SUFFIX,
) -> str:
    """
    Infer the value to match against the annotation CSV video_file column.
    
    For original videos:
        D01_20260525161829.mp4 -> D01_20260525161829.mp4
    
    For downsampled videos:
        D01_20260525161829_downsampled.mp4 -> D01_20260525161829.mp4
    
    This is needed because the downsampled CSV preserves the original video_file
    value while changing only centroid_x and centroid_y.
    """
    
    if annotation_video_file is not None:
        return annotation_video_file
    
    video_path = Path(video_path)
    stem = video_path.stem
    
    if downsampled_suffix and stem.endswith(downsampled_suffix):
        stem = stem[: -len(downsampled_suffix)]
    
    return f"{stem}{video_path.suffix}"


def parse_bool_series(s: pd.Series) -> pd.Series:
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


def get_video_info_with_opencv(video_path: str | Path) -> tuple[float, int, int, int]:
    """
    Read basic video metadata with OpenCV.
    """
    
    video_path = Path(video_path)
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    if source_fps <= 0:
        raise RuntimeError(f"Invalid FPS for video: {video_path}")
    
    return source_fps, total_frames, width, height


def build_ffmpeg_visible_filter(
    intervals: list[tuple[float, float]],
    target_fps: float,
) -> str:
    """
    Build an ffmpeg video filter that:
    1. samples the video at target_fps,
    2. keeps only frames inside visible intervals,
    3. resets timestamps for a continuous output video.
    """
    
    if not intervals:
        raise ValueError("No visible intervals provided")
    
    conditions = [
        f"between(t\\,{start:.6f}\\,{end:.6f})"
        for start, end in intervals
    ]
    
    select_expr = "+".join(conditions)
    
    return (
        f"fps={target_fps},"
        f"select='{select_expr}',"
        f"setpts=N/({target_fps}*TB)"
    )


def generate_visible_frame_metadata(
    video_path: str | Path,
    intervals: list[tuple[float, float]],
    source_fps: float,
    total_frames: int,
    target_fps: float,
    camera: str,
) -> pd.DataFrame:
    """
    Generate mapping from rebuilt video frames back to source video time/frame.
    
    Assumes constant-frame-rate source video.
    """
    
    video_path = Path(video_path)
    
    duration_sec = total_frames / source_fps
    target_step = 1.0 / target_fps
    
    records = []
    rebuilt_frame = 0
    
    for start, end in intervals:
        clamped_start = max(start, 0.0)
        clamped_end = min(end, duration_sec)
        
        if clamped_start >= clamped_end:
            continue
        
        t = clamped_start
        
        while t < clamped_end:
            source_frame = int(round(t * source_fps))
            
            if source_frame >= total_frames:
                break
            
            records.append(
                {
                    "rebuilt_frame": rebuilt_frame,
                    "camera": camera,
                    "source_video": video_path.name,
                    "source_frame": source_frame,
                    "source_time_sec": t,
                    "target_time_sec": t,
                    "visible_interval_start": start,
                    "visible_interval_end": end,
                }
            )
            
            rebuilt_frame += 1
            t += target_step
    
    return pd.DataFrame(records)


def create_visible_video_with_ffmpeg(
    video_path: str | Path,
    output_video: str | Path,
    intervals: list[tuple[float, float]],
    target_fps: float,
    crf: int = 18,
    preset: str = "fast",
    threads: int = 0,
) -> Path:
    """
    Create a DLC-ready visible-frame H.264 video directly from the source video.
    
    This avoids writing intermediate PNG frames.
    """
    
    video_path = Path(video_path)
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    
    vf = build_ffmpeg_visible_filter(
        intervals=intervals,
        target_fps=target_fps,
    )
    
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-threads",
        str(threads),
        str(output_video),
    ]
    
    print("")
    print("Creating visible H.264 video directly with ffmpeg:")
    print(" ".join(cmd), flush=True)
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("ffmpeg failed while creating visible video")
    
    return output_video


def preprocess_video_direct_ffmpeg(
    video_path: str | Path,
    annotation_csv: str | Path,
    output_dir: str | Path,
    target_fps: float,
    camera: str | None = None,
    threads: int = 0,
    annotation_video_file: str | None = None,
    downsampled_suffix: str = DOWNSAMPLED_SUFFIX,
) -> Path:
    """
    Preprocess one video using ffmpeg directly.
    
    This avoids Python/OpenCV frame extraction and avoids intermediate PNG files.
    """
    
    video_path = Path(video_path)
    annotation_csv = Path(annotation_csv)
    output_dir = Path(output_dir)
    
    if camera is None:
        camera = infer_camera_from_name(video_path)
    
    matched_video_file = infer_annotation_video_file(
        video_path=video_path,
        annotation_video_file=annotation_video_file,
        downsampled_suffix=downsampled_suffix,
    )
    
    video_out_dir = output_dir / video_path.stem
    video_out_dir.mkdir(parents=True, exist_ok=True)
    
    rebuilt_video = video_out_dir / f"{video_path.stem}_visible_h264.mp4"
    metadata_csv = video_out_dir / "visible_frame_metadata.csv"
    info_csv = video_out_dir / "video_info.csv"
    
    intervals = load_visible_intervals_from_csv(
        csv_path=annotation_csv,
        video_file=matched_video_file,
        camera=camera,
    )
    
    source_fps, total_frames, width, height = get_video_info_with_opencv(video_path)
    
    if not intervals:
        print(f"[INFO] No visible intervals found for {video_path.name}")
        print(f"[INFO] Annotation CSV video_file match used: {matched_video_file}")
        
        video_info = pd.DataFrame(
            [
                {
                    "source_fps": source_fps,
                    "target_fps": target_fps,
                    "width": width,
                    "height": height,
                    "total_frames": total_frames,
                    "has_visible_frames": 0,
                    "num_intervals": 0,
                    "num_frames": 0,
                    "annotation_video_file": matched_video_file,
                }
            ]
        )
        
        metadata = pd.DataFrame(
            columns=[
                "rebuilt_frame",
                "camera",
                "source_video",
                "source_frame",
                "source_time_sec",
                "target_time_sec",
                "visible_interval_start",
                "visible_interval_end",
            ]
        )
        
        metadata.to_csv(metadata_csv, index=False)
        video_info.to_csv(info_csv, index=False)
        
        print("")
        print(f"[SKIP] No visible frames. Wrote empty metadata for: {video_path.name}")
        
        return video_out_dir
    
    create_visible_video_with_ffmpeg(
        video_path=video_path,
        output_video=rebuilt_video,
        intervals=intervals,
        target_fps=target_fps,
        crf=18,
        preset="fast",
        threads=threads,
    )
    
    metadata = generate_visible_frame_metadata(
        video_path=video_path,
        intervals=intervals,
        source_fps=source_fps,
        total_frames=total_frames,
        target_fps=target_fps,
        camera=camera,
    )
    
    video_info = pd.DataFrame(
        [
            {
                "source_fps": source_fps,
                "target_fps": target_fps,
                "width": width,
                "height": height,
                "total_frames": total_frames,
                "has_visible_frames": 1,
                "num_intervals": len(intervals),
                "num_frames": len(metadata),
                "annotation_video_file": matched_video_file,
            }
        ]
    )
    
    metadata.to_csv(metadata_csv, index=False)
    video_info.to_csv(info_csv, index=False)
    
    print("")
    print(f"Saved rebuilt video: {rebuilt_video}")
    print(f"Saved metadata CSV: {metadata_csv}")
    print(f"Saved video info CSV: {info_csv}")
    
    return video_out_dir


def main() -> None:
    args = parse_args()
    
    preprocess_video_direct_ffmpeg(
        video_path=args.video,
        annotation_csv=args.annotations,
        output_dir=args.output_dir,
        target_fps=args.target_fps,
        camera=args.camera,
        threads=args.threads,
        annotation_video_file=args.annotation_video_file,
        downsampled_suffix=args.downsampled_suffix,
    )


if __name__ == "__main__":
    main()
