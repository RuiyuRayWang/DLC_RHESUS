#!/usr/bin/env python3
"""
Generate a template CSV for human/OCR-confirmed time anchors.

This template is consumed by scripts/fuse_trajectories.py via --time-anchors
(or auto-loaded as <labels_dir>/label_time_anchors.csv).

Columns:
    - csv_file: label CSV filename
    - camera: camera ID from CSV
    - video_file: source video filename when available
    - anchor_timestamp: prefilled from filename when parseable; confirm/edit as needed
    - anchor_video_frame: frame index corresponding to anchor_timestamp (default 0)
    - method: human | ocr | filename
    - notes: free text

Usage:
  python scripts/generate_time_anchor_template.py --dir results/oft_analysis/H1705011/2026-05-25
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


def is_label_csv(df: pd.DataFrame) -> bool:
    required = {"camera", "video_frame", "time_sec", "centroid_x", "centroid_y"}
    return required.issubset(df.columns)


def infer_video_file(csv_name: str) -> str:
    match = re.search(r"^(D0[1-4]_\d{14})\.csv$", csv_name)
    if not match:
        return ""
    return f"{match.group(1)}.mp4"


def infer_anchor_timestamp(name: str) -> str:
    match = re.search(r"D0[1-4]_(\d{14})", name)
    if not match:
        return ""

    dt = datetime.strptime(match.group(1), '%Y%m%d%H%M%S')
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate label time-anchor template")
    ap.add_argument("--dir", required=True, help="Directory containing label CSVs")
    ap.add_argument("--out", default=None, help="Output CSV path (default: <dir>/label_time_anchors.csv)")
    args = ap.parse_args()

    base = Path(args.dir)
    if not base.exists():
        raise SystemExit(f"ERROR: directory not found: {base}")

    out = Path(args.out) if args.out else (base / "label_time_anchors.csv")

    rows = []
    for csv_path in sorted(base.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if not is_label_csv(df):
            continue

        camera = str(df["camera"].iloc[0]) if "camera" in df.columns and len(df) else ""
        video_file = ""
        if "video_file" in df.columns and len(df):
            video_file = str(df["video_file"].iloc[0])
        if not video_file or video_file.lower() == 'nan':
            video_file = infer_video_file(csv_path.name)

        anchor_timestamp = infer_anchor_timestamp(video_file or csv_path.name)
        method = 'filename' if anchor_timestamp else ''
        notes = 'Prefilled from filename; confirm manually.' if anchor_timestamp else ''

        rows.append({
            "csv_file": csv_path.name,
            "camera": camera,
            "video_file": video_file,
            "anchor_timestamp": anchor_timestamp,
            "anchor_video_frame": 0,
            "method": method,
            "notes": notes,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out, index=False)
    print(f"Wrote {len(out_df)} rows -> {out}")


if __name__ == "__main__":
    main()
