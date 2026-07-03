#!/usr/bin/env python3
"""
Batch RT-DETR inference for OFT downsampled videos.

Pipeline:
1) Discover videos under videos_downsampled/{monkey_id}/{date}/*.mp4
2) Sequentially decode and sample frames at target Hz (default 2Hz)
3) Save sampled frames as JPG for model input
4) Run batched RT-DETR inference using Ultralytics
5) Save per-video CSV predictions preserving source folder structure

Output CSV columns include predicted bbox coordinates and center point.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
import pandas as pd
from ultralytics import RTDETR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEOS_ROOT = PROJECT_ROOT / "data" / "oft_rhp" / "videos_downsampled"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "results" / "predictions"
DEFAULT_FRAMES_ROOT = PROJECT_ROOT / "tmp" / "rtdetr_frames_2hz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch RT-DETR inference on OFT downsampled videos")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to RT-DETR weights (for example, .../weights/best.pt)",
    )
    parser.add_argument(
        "--videos-root",
        type=Path,
        default=DEFAULT_VIDEOS_ROOT,
        help="Root folder containing monkey/date/*.mp4 videos",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Output root for per-video prediction CSV files",
    )
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=DEFAULT_FRAMES_ROOT,
        help="Where sampled JPG frames are written before inference",
    )
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Sampling rate for inference frames")
    parser.add_argument(
        "--input-fps-fallback",
        type=float,
        default=20.0,
        help="Fallback FPS if video metadata is missing/invalid",
    )
    parser.add_argument("--imgsz", type=int, default=960, help="RT-DETR inference image size")
    parser.add_argument("--batch", type=int, default=64, help="Batch size for model inference")
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device index (0), comma list (0,1,...) or cpu",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=10, help="Max detections per frame")
    parser.add_argument(
        "--monkey-id",
        default=None,
        help="Optional filter for one monkey folder (for example, H1705011)",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Optional filter for one date folder (for example, 2026-06-16)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-video CSV outputs")
    parser.add_argument(
        "--cleanup-frames",
        action="store_true",
        help="Delete sampled JPG frames after each video is processed",
    )
    return parser.parse_args()


def list_videos(videos_root: Path, monkey_id: str | None = None, date: str | None = None) -> List[Path]:
    if monkey_id and date:
        search_root = videos_root / monkey_id / date
        if not search_root.exists():
            return []
        return sorted(search_root.glob("*.mp4"))

    if monkey_id:
        search_root = videos_root / monkey_id
        if not search_root.exists():
            return []
        return sorted(search_root.rglob("*.mp4"))

    if date:
        return sorted(videos_root.glob(f"*/{date}/*.mp4"))

    return sorted(videos_root.rglob("*.mp4"))


def safe_fps(cap: cv2.VideoCapture, fallback: float) -> float:
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        return float(fallback)
    return fps


def sample_video_to_jpg(
    video_path: Path,
    frames_dir: Path,
    sample_fps: float,
    input_fps_fallback: float,
) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = safe_fps(cap, fallback=input_fps_fallback)
    sample_period = 1.0 / sample_fps
    next_sample_time = 0.0

    frames_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_time = frame_idx / fps
        if frame_time + 1e-9 >= next_sample_time:
            frame_name = f"{video_path.stem}_f{frame_idx:06d}.jpg"
            image_path = frames_dir / frame_name
            cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            rows.append(
                {
                    "video_file": video_path.name,
                    "video_frame": frame_idx,
                    "video_time_sec": frame_time,
                    "sample_fps": sample_fps,
                    "sampled_image_path": str(image_path),
                }
            )
            next_sample_time += sample_period

        frame_idx += 1

    cap.release()

    if not rows:
        return pd.DataFrame(
            columns=[
                "video_file",
                "video_frame",
                "video_time_sec",
                "sample_fps",
                "sampled_image_path",
            ]
        )

    return pd.DataFrame(rows)


def pick_best_box(result, class_names: Dict[int, str]) -> Dict[str, object]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return {
            "has_detection": False,
            "bbox_x1": np.nan,
            "bbox_y1": np.nan,
            "bbox_x2": np.nan,
            "bbox_y2": np.nan,
            "bbox_cx": np.nan,
            "bbox_cy": np.nan,
            "bbox_w": np.nan,
            "bbox_h": np.nan,
            "confidence": np.nan,
            "class_id": np.nan,
            "class_name": "",
        }

    conf = boxes.conf.detach().cpu().numpy()
    best_idx = int(conf.argmax())
    xyxy = boxes.xyxy[best_idx].detach().cpu().numpy().tolist()

    x1, y1, x2, y2 = [float(v) for v in xyxy]
    cls_id = int(boxes.cls[best_idx].item()) if boxes.cls is not None else 0
    cls_name = class_names.get(cls_id, str(cls_id))

    return {
        "has_detection": True,
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "bbox_cx": (x1 + x2) / 2.0,
        "bbox_cy": (y1 + y2) / 2.0,
        "bbox_w": x2 - x1,
        "bbox_h": y2 - y1,
        "confidence": float(conf[best_idx]),
        "class_id": cls_id,
        "class_name": cls_name,
    }


def infer_on_frames(
    model: RTDETR,
    image_paths: Sequence[Path],
    imgsz: int,
    batch: int,
    device: str,
    conf: float,
    iou: float,
    max_det: int,
) -> List[Dict[str, object]]:
    if not image_paths:
        return []

    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf,
        iou=iou,
        max_det=max_det,
        verbose=False,
    )

    names_raw = model.names
    if isinstance(names_raw, dict):
        class_names = {int(k): str(v) for k, v in names_raw.items()}
    else:
        class_names = {i: str(v) for i, v in enumerate(names_raw)}

    return [pick_best_box(result=r, class_names=class_names) for r in results]


def process_video(video_path: Path, rel_video: Path, model: RTDETR, args: argparse.Namespace) -> Path:
    out_csv = args.out_root / rel_video.parent / f"{video_path.stem}_preds.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if out_csv.exists() and not args.overwrite:
        print(f"[SKIP] {rel_video} (output exists)")
        return out_csv

    frames_dir = args.frames_root / rel_video.parent / video_path.stem
    if frames_dir.exists() and args.overwrite:
        shutil.rmtree(frames_dir)

    sampled_df = sample_video_to_jpg(
        video_path=video_path,
        frames_dir=frames_dir,
        sample_fps=args.sample_fps,
        input_fps_fallback=args.input_fps_fallback,
    )

    if sampled_df.empty:
        sampled_df.to_csv(out_csv, index=False)
        print(f"[WARN] No sampled frames found for {rel_video}; wrote empty CSV")
        return out_csv

    image_paths = [Path(p) for p in sampled_df["sampled_image_path"].tolist()]
    det_rows = infer_on_frames(
        model=model,
        image_paths=image_paths,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
    )

    pred_df = sampled_df.copy()
    pred_df.insert(0, "relative_video_path", str(rel_video))
    pred_df.insert(1, "model_path", str(args.model))
    pred_df.insert(2, "device", str(args.device))
    pred_df.insert(3, "imgsz", int(args.imgsz))
    pred_df.insert(4, "conf_thres", float(args.conf))
    pred_df.insert(5, "iou_thres", float(args.iou))

    det_df = pd.DataFrame(det_rows)
    out_df = pd.concat([pred_df.reset_index(drop=True), det_df.reset_index(drop=True)], axis=1)

    out_df.to_csv(out_csv, index=False)

    if args.cleanup_frames and frames_dir.exists():
        shutil.rmtree(frames_dir)

    n_detected = int(out_df["has_detection"].fillna(False).sum())
    print(
        f"[OK] {rel_video} -> {out_csv} "
        f"(sampled={len(out_df)} detected={n_detected})"
    )
    return out_csv


def main() -> None:
    args = parse_args()

    args.model = args.model.resolve()
    args.videos_root = args.videos_root.resolve()
    args.out_root = args.out_root.resolve()
    args.frames_root = args.frames_root.resolve()

    if args.sample_fps <= 0:
        raise ValueError("--sample-fps must be > 0")
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.videos_root.exists():
        raise FileNotFoundError(f"videos-root not found: {args.videos_root}")

    videos = list_videos(args.videos_root, monkey_id=args.monkey_id, date=args.date)
    if not videos:
        raise RuntimeError("No videos found for the given filters")

    print(
        f"[INFO] Running RT-DETR batch inference on {len(videos)} videos "
        f"(sample_fps={args.sample_fps}, batch={args.batch}, device={args.device})"
    )

    model = RTDETR(str(args.model))

    outputs: List[Path] = []
    for video_path in videos:
        rel_video = video_path.relative_to(args.videos_root)
        outputs.append(process_video(video_path=video_path, rel_video=rel_video, model=model, args=args))

    print(f"[DONE] Wrote {len(outputs)} prediction CSV files under: {args.out_root}")


if __name__ == "__main__":
    main()