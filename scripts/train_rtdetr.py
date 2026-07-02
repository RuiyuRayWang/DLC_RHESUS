#!/usr/bin/env python3
"""
Train an Ultralytics RT-DETR detector for monkey bbox detection.

Example:
    conda activate det
    python scripts/train_rtdetr.py \
      --data datasets/ultralytics_bbox/data.yaml \
      --model rtdetr-l.pt \
      --epochs 100 \
      --imgsz 960 \
      --batch 16 \
      --device 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import RTDETR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RT-DETR with Ultralytics")
    parser.add_argument("--data", type=Path, required=True, help="Path to Ultralytics data.yaml")
    parser.add_argument("--model", default="rtdetr-l.pt", help="Pretrained RT-DETR checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device index (0), comma list (0,1,...), or cpu",
    )
    parser.add_argument("--project", default="results/ultralytics")
    parser.add_argument("--name", default="rtdetr_monkey_bbox")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precache", action="store_true", help="Cache images in RAM for faster training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = args.data.resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    print(
        f"[INFO] Starting RT-DETR training: data={data_yaml} model={args.model} "
        f"device={args.device} batch={args.batch} workers={args.workers}"
    )

    model = RTDETR(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        cache=args.precache,
    )


if __name__ == "__main__":
    main()
