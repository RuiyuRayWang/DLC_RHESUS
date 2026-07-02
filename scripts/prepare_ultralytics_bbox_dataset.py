#!/usr/bin/env python3
"""
Build an Ultralytics detection dataset from bbox CSV annotations + videos.

This script is designed for the OFT monkey workflow where annotations are stored in
CSV files (from scripts/bbox_labeler.py) and source videos are HEVC MP4 files.
To avoid HEVC random-access issues, frames are decoded sequentially.

Expected annotation filename mapping:
    video: D01_20260616135117_downsampled.mp4
    csv:   bbox_D01_20260616135117_downsampled.csv

Output dataset structure:
    <out_root>/
      images/
        train/*.jpg
        val/*.jpg
      labels/
        train/*.txt
        val/*.txt
      data.yaml
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import cv2
import pandas as pd
import yaml


VALID_TRUE = {"true", "1", "yes", "y", "t"}


@dataclass
class RowAnno:
    frame_idx: int
    x1: float
    y1: float
    x2: float
    y2: float


def parse_bool(value) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    txt = str(value).strip().lower()
    return txt in VALID_TRUE


def clamp_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Tuple[float, float, float, float] | None:
    xa = max(0.0, min(float(width - 1), float(min(x1, x2))))
    xb = max(0.0, min(float(width - 1), float(max(x1, x2))))
    ya = max(0.0, min(float(height - 1), float(min(y1, y2))))
    yb = max(0.0, min(float(height - 1), float(max(y1, y2))))

    if xb <= xa or yb <= ya:
        return None
    return xa, ya, xb, yb


def to_yolo_line(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> str:
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def collect_annotations(csv_path: Path) -> Tuple[str, Dict[int, List[RowAnno]], int]:
    df = pd.read_csv(csv_path)

    required = {
        "video_file",
        "video_frame",
        "monkey_visible",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{csv_path}: missing required columns: {missing}")

    if df.empty:
        raise ValueError(f"{csv_path}: no rows found")

    video_files = [str(v) for v in df["video_file"].dropna().unique()]
    if len(video_files) != 1:
        raise ValueError(f"{csv_path}: expected exactly 1 video_file, got {video_files}")
    video_name = Path(video_files[0]).name

    frame_to_annos: Dict[int, List[RowAnno]] = {}
    kept = 0

    for _, row in df.iterrows():
        if not parse_bool(row.get("monkey_visible")):
            continue

        coords = [row.get("bbox_x1"), row.get("bbox_y1"), row.get("bbox_x2"), row.get("bbox_y2")]
        if any(pd.isna(v) for v in coords):
            continue

        frame_idx = int(row["video_frame"])
        anno = RowAnno(
            frame_idx=frame_idx,
            x1=float(coords[0]),
            y1=float(coords[1]),
            x2=float(coords[2]),
            y2=float(coords[3]),
        )

        frame_to_annos.setdefault(frame_idx, []).append(anno)
        kept += 1

    return video_name, frame_to_annos, kept


def collect_background_frames(
    csv_path: Path,
    positive_frames: Set[int],
    max_background: int,
    seed: int,
) -> Tuple[Set[int], int]:
    """Collect frame indices where monkey_visible is false for background images."""
    if max_background <= 0:
        return set(), 0

    df = pd.read_csv(csv_path, usecols=["video_frame", "monkey_visible"])
    neg_candidates: List[int] = []

    for _, row in df.iterrows():
        if parse_bool(row.get("monkey_visible")):
            continue
        frame_idx = int(row["video_frame"])
        if frame_idx in positive_frames:
            continue
        neg_candidates.append(frame_idx)

    unique_neg = sorted(set(neg_candidates))
    if not unique_neg:
        return set(), 0

    rng = random.Random(seed)
    if len(unique_neg) <= max_background:
        return set(unique_neg), len(unique_neg)

    sampled = sorted(rng.sample(unique_neg, k=max_background))
    return set(sampled), len(unique_neg)


def extract_video_frames(
    video_path: Path,
    frame_to_annos: Dict[int, List[RowAnno]],
    background_frames: Set[int],
    split: str,
    out_root: Path,
    image_ext: str,
) -> Tuple[int, int, int]:
    target_frames: Set[int] = set(frame_to_annos.keys()) | set(background_frames)
    if not target_frames:
        return 0, 0, 0

    images_dir = out_root / "images" / split
    labels_dir = out_root / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    stem = video_path.stem
    max_target = max(target_frames)
    found_frames = 0
    written_boxes = 0
    written_backgrounds = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if frame_idx in target_frames:
            h, w = frame.shape[:2]
            base = f"{stem}_f{frame_idx:06d}"
            image_path = images_dir / f"{base}.{image_ext}"
            label_path = labels_dir / f"{base}.txt"

            label_lines: List[str] = []
            for anno in frame_to_annos.get(frame_idx, []):
                clamped = clamp_box(anno.x1, anno.y1, anno.x2, anno.y2, w, h)
                if clamped is None:
                    continue
                label_lines.append(to_yolo_line(*clamped, w, h))

            write_image = bool(label_lines) or (frame_idx in background_frames)
            if write_image:
                if image_ext.lower() == "jpg":
                    cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                else:
                    cv2.imwrite(str(image_path), frame)

                if label_lines:
                    label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
                    written_boxes += len(label_lines)
                else:
                    # Empty label file marks a background sample for Ultralytics detection.
                    label_path.write_text("", encoding="utf-8")
                    written_backgrounds += 1

                found_frames += 1

            if frame_idx >= max_target and found_frames >= len(target_frames):
                break

        if frame_idx > max_target and found_frames >= len(target_frames):
            break

        frame_idx += 1

    cap.release()
    return found_frames, written_boxes, written_backgrounds


def find_pairs(videos_dir: Path, csv_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for csv_path in sorted(csv_dir.glob("bbox_*.csv")):
        expected_video_name = csv_path.name.replace("bbox_", "", 1).replace(".csv", ".mp4")
        video_path = videos_dir / expected_video_name
        if not video_path.exists():
            # fallback: parse video_file from CSV
            try:
                df = pd.read_csv(csv_path, usecols=["video_file"])
                names = [Path(v).name for v in df["video_file"].dropna().unique()]
            except Exception:
                names = []
            found = None
            for name in names:
                candidate = videos_dir / name
                if candidate.exists():
                    found = candidate
                    break
            if found is None:
                print(f"[WARN] Missing video for {csv_path.name}; expected {video_path.name}")
                continue
            video_path = found
        pairs.append((video_path, csv_path))
    return pairs


def write_data_yaml(out_root: Path) -> Path:
    data = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "monkey"},
    }
    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return yaml_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Ultralytics dataset from bbox CSV + videos")
    parser.add_argument("--videos-dir", type=Path, required=True, help="Directory containing downsampled MP4 videos")
    parser.add_argument("--csv-dir", type=Path, required=True, help="Directory containing bbox_*.csv files")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("datasets/ultralytics_bbox"),
        help="Output root for Ultralytics dataset",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of videos assigned to validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument(
        "--image-ext",
        choices=["jpg", "png"],
        default="jpg",
        help="Output image format",
    )
    parser.add_argument(
        "--include-background",
        action="store_true",
        help="Include monkey_visible=False frames as background samples with empty label files",
    )
    parser.add_argument(
        "--background-ratio",
        type=float,
        default=0.5,
        help="Max background images per video as ratio of positive frames (used with --include-background)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    videos_dir = args.videos_dir.resolve()
    csv_dir = args.csv_dir.resolve()
    out_root = args.out_root.resolve()

    if not videos_dir.exists():
        raise FileNotFoundError(f"videos-dir not found: {videos_dir}")
    if not csv_dir.exists():
        raise FileNotFoundError(f"csv-dir not found: {csv_dir}")

    pairs = find_pairs(videos_dir, csv_dir)
    if not pairs:
        raise RuntimeError("No valid (video, csv) pairs found")

    random.Random(args.seed).shuffle(pairs)
    n_total = len(pairs)
    n_val = max(1, int(round(n_total * args.val_ratio))) if n_total > 1 else 0
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    if not train_pairs:
        train_pairs = val_pairs
        val_pairs = []

    print(f"[INFO] Pairs found: {n_total}")
    print(f"[INFO] Train videos: {len(train_pairs)}")
    print(f"[INFO] Val videos: {len(val_pairs)}")

    total_rows = 0
    total_images = 0
    total_boxes = 0
    total_backgrounds = 0

    for split, split_pairs in (("train", train_pairs), ("val", val_pairs)):
        for video_path, csv_path in split_pairs:
            video_name, frame_to_annos, kept = collect_annotations(csv_path)
            if video_name != video_path.name:
                print(f"[WARN] video_file mismatch in {csv_path.name}: {video_name} vs {video_path.name}")

            if not frame_to_annos:
                print(f"[WARN] No positive bbox rows in {csv_path.name}")
                continue

            positive_frames = set(frame_to_annos.keys())
            background_frames: Set[int] = set()
            n_bg_candidates = 0
            if args.include_background:
                max_background = max(0, int(round(len(positive_frames) * args.background_ratio)))
                background_frames, n_bg_candidates = collect_background_frames(
                    csv_path=csv_path,
                    positive_frames=positive_frames,
                    max_background=max_background,
                    seed=args.seed,
                )

            found, boxes, backgrounds = extract_video_frames(
                video_path=video_path,
                frame_to_annos=frame_to_annos,
                background_frames=background_frames,
                split=split,
                out_root=out_root,
                image_ext=args.image_ext,
            )
            print(
                f"[INFO] {split:<5} {video_path.name}: rows={kept} images={found} boxes={boxes} "
                f"backgrounds={backgrounds} bg_candidates={n_bg_candidates}"
            )
            total_rows += kept
            total_images += found
            total_boxes += boxes
            total_backgrounds += backgrounds

    yaml_path = write_data_yaml(out_root)
    print("\n[INFO] Dataset prepared")
    print(f"[INFO] output: {out_root}")
    print(f"[INFO] data.yaml: {yaml_path}")
    print(f"[INFO] total annotation rows kept: {total_rows}")
    print(f"[INFO] total images written: {total_images}")
    print(f"[INFO] total boxes written: {total_boxes}")
    print(f"[INFO] total background images written: {total_backgrounds}")


if __name__ == "__main__":
    main()
