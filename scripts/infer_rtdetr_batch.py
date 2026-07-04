#!/usr/bin/env python3
"""
Batch RT-DETR inference from pre-extracted OFT frame folders.

Expected input frame layout:
    <frames_root>/<monkey_id>/<date>/<video_name>/<video_name>_f000123.jpg

Output CSV layout:
    <out_root>/<monkey_id>/<date>/<video_name>_preds.csv

The script runs Ultralytics RT-DETR in batch mode per video folder.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = PROJECT_ROOT / "results" / "predictions"
DEFAULT_FRAMES_ROOT = PROJECT_ROOT / "tmp" / "rtdetr_frames_2hz"
FRAME_INDEX_RE = re.compile(r"_f(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch RT-DETR inference on extracted OFT frames")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to RT-DETR weights (for example, .../weights/best.pt)",
    )
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=DEFAULT_FRAMES_ROOT,
        help="Root folder containing extracted frames under monkey/date/video_name/",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Output root for per-video prediction CSV files",
    )
    parser.add_argument("--image-ext", choices=["jpg", "png"], default="jpg", help="Frame image extension")
    parser.add_argument("--imgsz", type=int, default=960, help="RT-DETR inference image size")
    parser.add_argument("--batch", type=int, default=64, help="Batch size for model inference")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Max number of frames sent to one predict() call; smaller values reduce peak memory",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device index (0), comma list (0,1,...) or cpu. Comma list runs one process per GPU.",
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
    return parser.parse_args()


def list_frame_dirs(frames_root: Path, monkey_id: str | None = None, date: str | None = None) -> List[Path]:
    if monkey_id and date:
        search_root = frames_root / monkey_id / date
        if not search_root.exists():
            return []
        return sorted([p for p in search_root.iterdir() if p.is_dir()])

    if monkey_id:
        search_root = frames_root / monkey_id
        if not search_root.exists():
            return []
        return sorted([p for p in search_root.glob("*/*") if p.is_dir()])

    if date:
        return sorted([p for p in frames_root.glob(f"*/{date}/*") if p.is_dir()])

    return sorted([p for p in frames_root.glob("*/*/*") if p.is_dir()])


def parse_frame_index(frame_path: Path) -> float:
    match = FRAME_INDEX_RE.search(frame_path.stem)
    if match is None:
        return np.nan
    return float(int(match.group(1)))


def collect_frame_rows(frame_dir: Path, image_ext: str) -> pd.DataFrame:
    image_paths = sorted(frame_dir.glob(f"*_f*.{image_ext}"))
    rows: List[Dict[str, object]] = []

    for image_path in image_paths:
        rows.append(
            {
                "video_name": frame_dir.name,
                "video_file": f"{frame_dir.name}.mp4",
                "video_frame": parse_frame_index(image_path),
                "sampled_image_path": str(image_path.resolve()),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["video_name", "video_file", "video_frame", "sampled_image_path"])

    df = pd.DataFrame(rows)
    return df.sort_values(by=["video_frame", "sampled_image_path"], na_position="last").reset_index(drop=True)


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
    model: Any,
    image_paths: Sequence[Path],
    imgsz: int,
    batch: int,
    chunk_size: int,
    device: str,
    conf: float,
    iou: float,
    max_det: int,
) -> List[Dict[str, object]]:
    if not image_paths:
        return []

    names_raw = model.names
    if isinstance(names_raw, dict):
        class_names = {int(k): str(v) for k, v in names_raw.items()}
    else:
        class_names = {i: str(v) for i, v in enumerate(names_raw)}

    if chunk_size <= 0:
        chunk_size = len(image_paths)

    out_rows: List[Dict[str, object]] = []

    for start in range(0, len(image_paths), chunk_size):
        chunk = image_paths[start : start + chunk_size]
        results_iter = model.predict(
            source=[str(p) for p in chunk],
            imgsz=imgsz,
            batch=max(1, int(batch)),
            device=device,
            conf=conf,
            iou=iou,
            max_det=max_det,
            stream=True,
            verbose=False,
        )
        out_rows.extend(pick_best_box(result=r, class_names=class_names) for r in results_iter)

    return out_rows


def process_frame_dir(
    frame_dir: Path,
    rel_frame_dir: Path,
    model: Any,
    inference_device: str,
    reported_device: str,
    args: argparse.Namespace,
) -> Path:
    out_csv = args.out_root / rel_frame_dir.parent / f"{frame_dir.name}_preds.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if out_csv.exists() and not args.overwrite:
        print(f"[SKIP] {rel_frame_dir} (output exists)")
        return out_csv

    sampled_df = collect_frame_rows(frame_dir=frame_dir, image_ext=args.image_ext)

    if sampled_df.empty:
        sampled_df.to_csv(out_csv, index=False)
        print(f"[WARN] No frames found for {rel_frame_dir}; wrote empty CSV")
        return out_csv

    image_paths = [Path(p) for p in sampled_df["sampled_image_path"].tolist()]
    det_rows = infer_on_frames(
        model=model,
        image_paths=image_paths,
        imgsz=args.imgsz,
        batch=args.batch,
        chunk_size=args.chunk_size,
        device=inference_device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
    )

    pred_df = sampled_df.copy()
    pred_df.insert(0, "relative_frame_dir", str(rel_frame_dir))
    pred_df.insert(1, "relative_video_path", str(rel_frame_dir.parent / f"{frame_dir.name}.mp4"))
    pred_df.insert(1, "model_path", str(args.model))
    pred_df.insert(3, "device", str(reported_device))
    pred_df.insert(4, "imgsz", int(args.imgsz))
    pred_df.insert(5, "conf_thres", float(args.conf))
    pred_df.insert(6, "iou_thres", float(args.iou))

    det_df = pd.DataFrame(det_rows)
    out_df = pd.concat([pred_df.reset_index(drop=True), det_df.reset_index(drop=True)], axis=1)

    out_df.to_csv(out_csv, index=False)

    n_detected = int(out_df["has_detection"].fillna(False).sum())
    print(
        f"[OK] {rel_frame_dir} -> {out_csv} "
        f"(sampled={len(out_df)} detected={n_detected})"
    )
    return out_csv


def parse_devices(device_arg: str) -> List[str]:
    normalized = str(device_arg).strip()
    if not normalized:
        return ["0"]
    if normalized.lower() == "cpu":
        return ["cpu"]

    parts = [p.strip() for p in normalized.split(",") if p.strip()]
    return parts if parts else ["0"]


def shard_evenly(items: Sequence[Path], n_shards: int) -> List[List[Path]]:
    shards: List[List[Path]] = [[] for _ in range(n_shards)]
    for idx, item in enumerate(items):
        shards[idx % n_shards].append(item)
    return shards


def run_worker(
    worker_id: int,
    requested_device: str,
    frame_dirs: List[Path],
    args: argparse.Namespace,
) -> int:
    try:
        # Use the requested device id directly (e.g. "1", "2", ..., or "cpu").
        inference_device = str(requested_device)

        from ultralytics import RTDETR

        model = RTDETR(str(args.model))
        outputs = 0
        for frame_dir in frame_dirs:
            rel_frame_dir = frame_dir.relative_to(args.frames_root)
            process_frame_dir(
                frame_dir=frame_dir,
                rel_frame_dir=rel_frame_dir,
                model=model,
                inference_device=inference_device,
                reported_device=requested_device,
                args=args,
            )
            outputs += 1
        print(
            f"[WORKER-{worker_id}] done on device={requested_device} "
            f"(inference_device={inference_device}); processed={outputs}"
        )
        return 0
    except Exception as exc:
        print(f"[WORKER-{worker_id}] failed on device={requested_device}: {exc}")
        traceback.print_exc()
        return 1


def worker_entry(
    worker_id: int,
    requested_device: str,
    frame_dirs: List[Path],
    args: argparse.Namespace,
    status_queue: mp.Queue,
) -> None:
    code = run_worker(worker_id=worker_id, requested_device=requested_device, frame_dirs=frame_dirs, args=args)
    status_queue.put((worker_id, code))


def main() -> None:
    args = parse_args()

    args.model = args.model.resolve()
    args.frames_root = args.frames_root.resolve()
    args.out_root = args.out_root.resolve()
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.frames_root.exists():
        raise FileNotFoundError(f"frames-root not found: {args.frames_root}")

    frame_dirs = list_frame_dirs(args.frames_root, monkey_id=args.monkey_id, date=args.date)
    if not frame_dirs:
        raise RuntimeError("No frame directories found for the given filters")

    devices = parse_devices(args.device)
    print(
        f"[INFO] Running RT-DETR batch inference on {len(frame_dirs)} frame folders "
        f"(batch={args.batch}, devices={','.join(devices)})"
    )

    if len(devices) == 1:
        status = run_worker(worker_id=0, requested_device=devices[0], frame_dirs=frame_dirs, args=args)
        if status != 0:
            raise RuntimeError("Inference worker failed")
        print(f"[DONE] Wrote predictions under: {args.out_root}")
        return

    shards = shard_evenly(frame_dirs, len(devices))
    ctx = mp.get_context("spawn")
    status_queue: mp.Queue = ctx.Queue()
    procs: List[mp.Process] = []

    for i, (device, shard) in enumerate(zip(devices, shards)):
        if not shard:
            continue
        p = ctx.Process(target=worker_entry, args=(i, device, shard, args, status_queue), daemon=False)
        p.start()
        procs.append(p)
        print(f"[INFO] started worker={i} device={device} shard={len(shard)}")

    failed = False
    for _ in range(len(procs)):
        worker_id, code = status_queue.get()
        if code != 0:
            failed = True
            print(f"[ERROR] worker={worker_id} failed")

    for p in procs:
        p.join()
        if p.exitcode != 0:
            failed = True

    if failed:
        raise RuntimeError("One or more inference workers failed")

    print(f"[DONE] Wrote predictions under: {args.out_root}")


if __name__ == "__main__":
    main()