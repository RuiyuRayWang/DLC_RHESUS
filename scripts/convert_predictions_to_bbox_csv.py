#!/usr/bin/env python3
"""
Convert RT-DETR prediction CSVs into bbox training CSV format.

This is a minimal schema refactor intended for predictions that were manually
refined in bbox_labeler.py validate mode.

Input CSV (typical prediction columns):
    video_file, video_frame, has_detection, bbox_x1, bbox_y1, bbox_x2, bbox_y2, ...

Output CSV (bbox format expected by train pipeline):
    video_file, video_timestamp, video_frame, time_sec,
    monkey_visible, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
    bbox_width, bbox_height, image_width, image_height

Examples:
  # Convert one file
  python scripts/convert_predictions_to_bbox_csv.py \
    --csv results/barrier/H1705011/topdown/D01_20260626143617_downsampled_preds.csv

  # Convert all *_preds.csv in a folder
  python scripts/convert_predictions_to_bbox_csv.py \
    --csv-dir results/barrier/H1705011/topdown

  # Convert to a separate destination directory
  python scripts/convert_predictions_to_bbox_csv.py \
    --csv-dir results/barrier/H1705011/topdown \
    --out-dir results/barrier/H1705011/topdown_bbox
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


VALID_TRUE = {"true", "1", "yes", "y", "t"}

OUTPUT_COLUMNS = [
    "video_file",
    "video_timestamp",
    "video_frame",
    "time_sec",
    "monkey_visible",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_width",
    "bbox_height",
    "image_width",
    "image_height",
]


def parse_bool(value) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in VALID_TRUE


def parse_video_timestamp_from_name(name: str) -> str:
    match = re.search(r"(?<!\d)(\d{14})(?!\d)", name)
    if not match:
        return "unknown"
    try:
        dt = datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return "unknown"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def choose_video_file(df: pd.DataFrame, csv_path: Path) -> str:
    if "video_file" in df.columns and df["video_file"].notna().any():
        return Path(str(df["video_file"].dropna().iloc[0])).name
    if "relative_video_path" in df.columns and df["relative_video_path"].notna().any():
        return Path(str(df["relative_video_path"].dropna().iloc[0])).name
    stem = csv_path.name.replace("_preds.csv", "")
    return f"{stem}.mp4"


def as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def convert_one(csv_path: Path, out_path: Path, fps_fallback: float) -> tuple[int, int]:
    df = pd.read_csv(csv_path)
    if "video_frame" not in df.columns:
        raise ValueError(f"{csv_path}: missing required column 'video_frame'")

    out = pd.DataFrame(index=df.index)

    video_file = choose_video_file(df, csv_path)
    out["video_file"] = video_file

    if "video_timestamp" in df.columns:
        out["video_timestamp"] = df["video_timestamp"].fillna("unknown")
    else:
        out["video_timestamp"] = parse_video_timestamp_from_name(video_file)

    out["video_frame"] = as_numeric(df["video_frame"]).round().astype("Int64")

    if "time_sec" in df.columns:
        out["time_sec"] = as_numeric(df["time_sec"])
    else:
        out["time_sec"] = as_numeric(df["video_frame"]) / float(fps_fallback)

    if "monkey_visible" in df.columns:
        out["monkey_visible"] = df["monkey_visible"].map(parse_bool)
    elif "has_detection" in df.columns:
        out["monkey_visible"] = df["has_detection"].map(parse_bool)
    else:
        bbox_present = pd.Series(True, index=df.index)
        for col in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"):
            if col in df.columns:
                bbox_present &= df[col].notna()
            else:
                bbox_present &= False
        out["monkey_visible"] = bbox_present

    for col in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"):
        out[col] = as_numeric(df[col]) if col in df.columns else np.nan

    if "bbox_width" in df.columns:
        out["bbox_width"] = as_numeric(df["bbox_width"])
    elif "bbox_w" in df.columns:
        out["bbox_width"] = as_numeric(df["bbox_w"])
    else:
        out["bbox_width"] = out["bbox_x2"] - out["bbox_x1"]

    if "bbox_height" in df.columns:
        out["bbox_height"] = as_numeric(df["bbox_height"])
    elif "bbox_h" in df.columns:
        out["bbox_height"] = as_numeric(df["bbox_h"])
    else:
        out["bbox_height"] = out["bbox_y2"] - out["bbox_y1"]

    if "image_width" in df.columns:
        out["image_width"] = as_numeric(df["image_width"])
    else:
        out["image_width"] = np.nan

    if "image_height" in df.columns:
        out["image_height"] = as_numeric(df["image_height"])
    else:
        out["image_height"] = np.nan

    for col in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_width", "bbox_height"):
        out.loc[~out["monkey_visible"], col] = np.nan

    out = out[OUTPUT_COLUMNS]

    # Keep same basename style expected by training pairing logic.
    if not out_path.name.startswith("bbox_"):
        out_path = out_path.with_name(f"bbox_{out_path.name}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    kept_visible = int(out["monkey_visible"].sum())
    return len(out), kept_visible


def iter_csvs(single_csv: Path | None, csv_dir: Path | None) -> Iterable[Path]:
    if single_csv is not None:
        yield single_csv
        return

    assert csv_dir is not None
    for path in sorted(csv_dir.glob("*_preds.csv")):
        if path.is_file():
            yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert prediction CSVs to bbox training CSV format")
    parser.add_argument("--csv", type=Path, default=None, help="Path to a single prediction CSV")
    parser.add_argument("--csv-dir", type=Path, default=None, help="Directory containing *_preds.csv files")
    parser.add_argument("--out-dir", type=Path, default=None, help="Optional output directory for converted CSVs")
    parser.add_argument("--fps-fallback", type=float, default=20.0, help="FPS used when time_sec is missing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if bool(args.csv) == bool(args.csv_dir):
        raise SystemExit("ERROR: provide exactly one of --csv or --csv-dir")
    if args.fps_fallback <= 0:
        raise SystemExit("ERROR: --fps-fallback must be > 0")

    source_paths = list(iter_csvs(args.csv, args.csv_dir))
    if not source_paths:
        raise SystemExit("ERROR: no input CSVs found")

    total_rows = 0
    total_visible = 0

    for src in source_paths:
        src = src.expanduser().resolve()
        if not src.exists():
            raise SystemExit(f"ERROR: input CSV not found: {src}")

        if args.out_dir is None:
            dst = src.with_name(f"bbox_{src.name.replace('_preds.csv', '')}.csv")
        else:
            dst = args.out_dir.expanduser().resolve() / f"bbox_{src.name.replace('_preds.csv', '')}.csv"

        n_rows, n_visible = convert_one(src, dst, fps_fallback=args.fps_fallback)
        total_rows += n_rows
        total_visible += n_visible
        print(f"[OK] {src.name} -> {dst} rows={n_rows} visible={n_visible}")

    print(f"[DONE] files={len(source_paths)} rows={total_rows} visible={total_visible}")


if __name__ == "__main__":
    main()
