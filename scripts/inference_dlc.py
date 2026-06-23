#!/usr/bin/env python3
"""
Run DeepLabCut inference on preprocessed visible-frame videos.

This script:

Finds a rebuilt H.264 video created by preprocess_visible_video.py.
Infers the camera from the video name.
Selects the camera-specific DLC config.
Runs DLC inference.
Flattens the DLC CSV.
Merges predictions with visible_frame_metadata.csv.

It does NOT decode HEVC or create temporary PNGs.
"""

from future import annotations

from pathlib import Path
import argparse
import shutil

import deeplabcut
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DLC_PROJECT_ROOT = PROJECT_ROOT / "deeplabcut_projects"
DEFAULT_PREPROCESSED_ROOT = PROJECT_ROOT / "data" / "oft_rhp" / "visible_dlc_preprocessed"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "visible_dlc_predictions"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DLC on preprocessed visible-frame H.264 videos.")
    parser.add_argument("--preprocessed-dir", required=True, 
        help=(
            "Directory produced by preprocess script for one video, "
            "e.g. data/oft_rhp/visible_dlc_preprocessed/D01_20260618131058"
        ),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT), help="Output directory for DLC predictions.",)
    parser.add_argument("--dlc-project-root", default=str(DLC_PROJECT_ROOT), help="Root directory containing camera-specific DLC projects.",)
    parser.add_argument("--camera", default=None, help="Optional camera override, e.g. D1. If omitted, inferred from video name.",)
    parser.add_argument("--batch-size", type=int, default=1, help="DLC batch size.",)
    parser.add_argument("--gputouse", type=int, default=0, help="GPU index for DLC.",)
    parser.add_argument("--shuffle", type=int, default=1, help="DLC shuffle index.",)
    return parser.parse_args()

def infer_camera_from_name(
    path_or_name: str | Path,
) -> str:
    """
    Infer camera ID from a video or directory name.
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
    
    raise ValueError(f"Could not infer camera from name: {name}")

def get_dlc_config_for_camera(
    camera: str,
    dlc_project_root: str | Path = DLC_PROJECT_ROOT,
) -> Path:
    """
    Return the DLC config path for a camera-specific model.
    """
    
    dlc_project_root = Path(dlc_project_root)
    
    camera_to_project = {
        "D1": "monkey_oft_tracking_D1-luolab-2026-06-19",
        "D2": "monkey_oft_tracking_D2-luolab-2026-06-19",
        "D3": "monkey_oft_tracking_D3-luolab-2026-06-19",
        "D4": "monkey_oft_tracking_D4-luolab-2026-06-19",
    }
    
    if camera not in camera_to_project:
        raise ValueError(f"Unsupported camera: {camera}")
    
    config_path = dlc_project_root / camera_to_project[camera] / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"DLC config not found for {camera}: {config_path}")
    
    return config_path

def find_rebuilt_video(
    preprocessed_dir: str | Path,
) -> Path:
    """
    Find the rebuilt visible H.264 video in a preprocessed directory.
    """
    
    preprocessed_dir = Path(preprocessed_dir)
    candidates = sorted(preprocessed_dir.glob("*_visible_h264.mp4"))
    
    if not candidates:
        raise FileNotFoundError(
            f"No rebuilt visible H.264 video found in {preprocessed_dir}"
        )
    
    if len(candidates) > 1:
        raise RuntimeError(
            f"Expected one rebuilt video in {preprocessed_dir}, found {len(candidates)}"
        )
    
    return candidates[0]

def find_metadata_csv(
    preprocessed_dir: str | Path,
) -> Path:
    """
    Find visible_frame_metadata.csv in a preprocessed directory.
    """
    
    metadata_csv = Path(preprocessed_dir) / "visible_frame_metadata.csv"
    
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    
    return metadata_csv

def run_dlc_on_video(
    rebuilt_video: str | Path,
    config_path: str | Path,
    batch_size: int = 1,
    gputouse: int = 0,
    shuffle: int = 1,
) -> None:
    """
    Run DeepLabCut on a rebuilt visible-frame video.
    """
    
    rebuilt_video = Path(rebuilt_video)
    config_path = Path(config_path)
    
    print("")
    print(f"Running DLC")
    print(f"Video: {rebuilt_video}")
    print(f"Config: {config_path}")
    print(f"Batch size: {batch_size}")
    print(f"GPU: {gputouse}")
    print(f"Shuffle: {shuffle}")
    
    deeplabcut.analyze_videos(
        str(config_path),
        [str(rebuilt_video)],
        shuffle=shuffle,
        batch_size=batch_size,
        save_as_csv=True,
        gputouse=gputouse,
    )

def find_dlc_csv(
    search_dir: str | Path,
) -> Path:
    """
    Find DLC prediction CSV generated by analyze_videos.
    """
    
    search_dir = Path(search_dir)
    csv_files = sorted(search_dir.rglob("*DLC*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No DLC CSV found in {search_dir}")
    
    return csv_files[0]

def flatten_dlc_predictions(
    dlc_csv: str | Path,
) -> pd.DataFrame:
    """
    Flatten a DLC CSV into ordinary columns while preserving all body parts.

    DLC CSV generally has a 3-level header:
        scorer / bodypart / coord
    """
    
    df = pd.read_csv(dlc_csv, header=[0, 1, 2])
    
    flat_cols = []
    
    for col in df.columns:
        if not isinstance(col, tuple):
            flat_cols.append(str(col))
            continue
        
        parts = [
            str(x)
            for x in col
            if str(x).lower() != "nan" and not str(x).startswith("Unnamed:")
        ]
        
        flat_cols.append("__".join(parts))
    
    df.columns = flat_cols
    df.insert(0, "rebuilt_frame", np.arange(len(df), dtype=int))
    
    return df

def merge_predictions_with_metadata(
    dlc_csv: str | Path,
    metadata_csv: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    """
    Attach original source video timing/frame information to DLC predictions.
    """
    
    pred = flatten_dlc_predictions(dlc_csv)
    meta = pd.read_csv(metadata_csv)
    
    merged = pred.merge(meta, on="rebuilt_frame", how="left")
    merged.to_csv(output_csv, index=False)
    
    return merged

def analyze_preprocessed_video(
    preprocessed_dir: str | Path,
    output_dir: str | Path,
    dlc_project_root: str | Path,
    camera: str | None = None,
    batch_size: int = 1,
    gputouse: int = 0,
    shuffle: int = 1,
) -> pd.DataFrame:
    """
    Run DLC on one preprocessed visible-frame video and merge predictions.
    """
    
    preprocessed_dir = Path(preprocessed_dir)
    output_dir = Path(output_dir)
    
    rebuilt_video = find_rebuilt_video(preprocessed_dir)
    metadata_csv = find_metadata_csv(preprocessed_dir)
    
    if camera is None:
        camera = infer_camera_from_name(rebuilt_video)
    
    config_path = get_dlc_config_for_camera(
        camera=camera,
        dlc_project_root=dlc_project_root,
    )
    
    video_out_dir = output_dir / preprocessed_dir.name
    video_out_dir.mkdir(parents=True, exist_ok=True)
    
    working_video = video_out_dir / rebuilt_video.name
    working_metadata = video_out_dir / metadata_csv.name
    
    if rebuilt_video.resolve() != working_video.resolve():
        shutil.copy(rebuilt_video, working_video)
    
    if metadata_csv.resolve() != working_metadata.resolve():
        shutil.copy(metadata_csv, working_metadata)
    
    run_dlc_on_video(
        rebuilt_video=working_video,
        config_path=config_path,
        batch_size=batch_size,
        gputouse=gputouse,
        shuffle=shuffle,
    )
    
    dlc_csv = find_dlc_csv(video_out_dir)
    merged_csv = video_out_dir / f"{preprocessed_dir.name}_dlc_with_source_mapping.csv"
    
    merged = merge_predictions_with_metadata(
        dlc_csv=dlc_csv,
        metadata_csv=working_metadata,
        output_csv=merged_csv,
    )
    
    print("")
    print(f"Saved DLC CSV: {dlc_csv}")
    print(f"Saved merged predictions: {merged_csv}")
    
    return merged

def main() -> None:
    args = parse_args()
    
    analyze_preprocessed_video(
        preprocessed_dir=args.preprocessed_dir,
        output_dir=args.output_dir,
        dlc_project_root=args.dlc_project_root,
        camera=args.camera,
        batch_size=args.batch_size,
        gputouse=args.gputouse,
        shuffle=args.shuffle,
    )

if name == "main":
    main()