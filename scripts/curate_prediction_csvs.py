#!/usr/bin/env python3
"""
Curate RT-DETR prediction CSV files by forcing selected frame ranges to no-detection.

For rows selected by rules, this script sets:
    - has_detection = False
    - bbox and confidence/class fields to empty values

Rule semantics are applied to a configurable frame axis:
    - video_frame (default): uses raw `video_frame`
    - row_index: uses CSV row index (sampled-frame order)
    - sampled_frame: uses video_frame / --sampled-frame-scale

Cutoff behavior:
    - before:N  -> rows with axis_value < N are forced to no-detection
    - after:N   -> rows with axis_value >= N are forced to no-detection

Example:
  python scripts/curate_prediction_csvs.py \
    --csv results/predictions/H1705011/2026-06-26/D01_20260626154641_downsampled_preds.csv \
    --csv results/predictions/H1705011/2026-06-26/D01_20260626191753_downsampled_preds.csv \
    --rule D01_20260626154641_downsampled_preds.csv:before:248 \
    --rule D01_20260626191753_downsampled_preds.csv:after:192
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BBOX_NUMERIC_COLS = [
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_cx",
    "bbox_cy",
    "bbox_w",
    "bbox_h",
    "confidence",
    "class_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate prediction CSVs by file-specific frame cutoffs")
    parser.add_argument(
        "--csv",
        type=Path,
        action="append",
        required=True,
        help="Path to a prediction CSV. Repeat for multiple files.",
    )
    parser.add_argument(
        "--rule",
        action="append",
        required=True,
        help=(
            "Rule in the format <csv_basename>:<before|after>:<N>. "
            "Example: D01_..._preds.csv:before:248"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files.",
    )
    parser.add_argument(
        "--cutoff-unit",
        choices=["sampled_frame", "video_frame", "row_index"],
        default="video_frame",
        help=(
            "Axis used to interpret rule cutoffs. "
            "video_frame (default) is robust to prior row deletions."
        ),
    )
    parser.add_argument(
        "--sampled-frame-scale",
        type=float,
        default=10.0,
        help="Scale factor between sampled frame index and video_frame. Default: 10.",
    )
    return parser.parse_args()


def parse_rule(rule_text: str) -> tuple[str, str, int]:
    parts = rule_text.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid rule '{rule_text}'. Expected <csv_basename>:<before|after>:<N>")

    basename, mode, n_text = parts
    mode = mode.strip().lower()
    if mode not in {"before", "after"}:
        raise ValueError(f"Invalid mode in rule '{rule_text}'. Use 'before' or 'after'.")

    try:
        n_value = int(n_text)
    except ValueError as exc:
        raise ValueError(f"Invalid N in rule '{rule_text}'. N must be an integer.") from exc

    if n_value < 0:
        raise ValueError(f"Invalid N in rule '{rule_text}'. N must be >= 0.")

    return basename.strip(), mode, n_value


def build_rule_map(rule_texts: list[str]) -> dict[str, list[tuple[str, int]]]:
    rule_map: dict[str, list[tuple[str, int]]] = {}
    for rule_text in rule_texts:
        basename, mode, n_value = parse_rule(rule_text)
        rule_map.setdefault(basename, []).append((mode, n_value))
    return rule_map


def mask_for_rule(axis_values: pd.Series, mode: str, n_value: int) -> np.ndarray:
    if mode == "before":
        return (axis_values < n_value).to_numpy()
    return (axis_values >= n_value).to_numpy()


def build_cutoff_axis(df: pd.DataFrame, cutoff_unit: str, sampled_frame_scale: float) -> pd.Series:
    if cutoff_unit == "row_index":
        return pd.Series(np.arange(len(df), dtype=float), index=df.index)

    if "video_frame" not in df.columns:
        raise ValueError("Missing required column 'video_frame' for selected cutoff unit")

    video_frame = pd.to_numeric(df["video_frame"], errors="coerce")
    if video_frame.isna().any():
        bad_count = int(video_frame.isna().sum())
        raise ValueError(f"Found {bad_count} non-numeric values in 'video_frame'")

    if cutoff_unit == "video_frame":
        return video_frame

    if sampled_frame_scale <= 0:
        raise ValueError("--sampled-frame-scale must be > 0")

    return video_frame / sampled_frame_scale


def clear_detection_fields(df: pd.DataFrame, mask: np.ndarray) -> None:
    if "has_detection" in df.columns:
        df.loc[mask, "has_detection"] = False

    for col in BBOX_NUMERIC_COLS:
        if col in df.columns:
            df.loc[mask, col] = np.nan

    if "class_name" in df.columns:
        if pd.api.types.is_numeric_dtype(df["class_name"]):
            df["class_name"] = df["class_name"].astype(object)
        df.loc[mask, "class_name"] = ""


def curate_file(
    csv_path: Path,
    rules: list[tuple[str, int]],
    dry_run: bool,
    cutoff_unit: str,
    sampled_frame_scale: float,
) -> None:
    df = pd.read_csv(csv_path)
    axis_values = build_cutoff_axis(
        df=df,
        cutoff_unit=cutoff_unit,
        sampled_frame_scale=sampled_frame_scale,
    )

    combined_mask = np.zeros(len(df), dtype=bool)

    for mode, n_value in rules:
        combined_mask |= mask_for_rule(axis_values, mode, n_value)

    changed_rows = int(combined_mask.sum())
    clear_detection_fields(df, combined_mask)

    if dry_run:
        print(f"[DRY-RUN] {csv_path}: rows_forced_false={changed_rows}/{len(df)}")
        return

    df.to_csv(csv_path, index=False)
    print(f"[UPDATED] {csv_path}: rows_forced_false={changed_rows}/{len(df)}")


def main() -> None:
    args = parse_args()
    rule_map = build_rule_map(args.rule)

    csv_paths = [p if p.is_absolute() else (PROJECT_ROOT / p) for p in args.csv]

    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        basename = csv_path.name
        if basename not in rule_map:
            raise ValueError(f"No rule supplied for file '{basename}'.")

        curate_file(
            csv_path=csv_path,
            rules=rule_map[basename],
            dry_run=args.dry_run,
            cutoff_unit=args.cutoff_unit,
            sampled_frame_scale=args.sampled_frame_scale,
        )


if __name__ == "__main__":
    main()