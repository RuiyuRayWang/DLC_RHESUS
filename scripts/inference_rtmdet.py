#!/usr/bin/env python3
"""
RTMDet inference script for monkey bbox detection on videos.
Supports batch processing of multiple videos.

Usage:
    python scripts/inference_rtmdet.py \
        --config configs/rtmdet_m_monkey.py \
        --checkpoint results/rtmdet_models/exp001/best_coco_bbox_mAP_epoch_XX.pth \
        --video data/oft_rhp/videos_downsampled/H1705011/2026-06-16/D01_*.mp4 \
        --output results/rtmdet_inference/ \
        --score-thr 0.5
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm
import csv

import torch
from mmyolo.apis import init_detector, inference_detector
from mmengine.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="RTMDet inference on videos")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--video", type=str, required=True, nargs="+", help="Video file(s) or pattern")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--score-thr", type=float, default=0.5, help="Score threshold for detection")
    parser.add_argument("--save-frames", action="store_true", help="Save visualization frames")
    parser.add_argument("--save-video", action="store_true", help="Save output video with detections")
    parser.add_argument("--fps", type=int, default=20, help="Output video FPS")
    
    return parser.parse_args()


class RTMDetInference:
    def __init__(self, config_path, checkpoint_path, device="cuda:0", score_thr=0.5):
        self.device = device
        self.score_thr = score_thr
        
        # Load model
        self.model = init_detector(config_path, checkpoint_path, device=device)
        print(f"Model loaded from: {checkpoint_path}")
    
    def _visualize_bbox(self, frame, result, score_thr=0.5):
        """Draw bboxes on frame."""
        if result.pred_instances.scores.numel() == 0:
            return frame
        
        bboxes = result.pred_instances.bboxes.cpu().numpy()
        scores = result.pred_instances.scores.cpu().numpy()
        
        frame_vis = frame.copy()
        for bbox, score in zip(bboxes, scores):
            if score >= score_thr:
                x1, y1, x2, y2 = bbox.astype(int)
                # Draw bbox
                cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw score
                cv2.putText(
                    frame_vis,
                    f"{score:.3f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
        
        return frame_vis
    
    def process_video(self, video_path, output_dir, save_frames=False, save_video=False, fps=20):
        """Process single video and return detections."""
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps_orig = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nProcessing: {video_path.name}")
        print(f"  Resolution: {width}×{height}, FPS: {fps_orig}, Frames: {n_frames}")
        
        # Output files
        csv_output = output_dir / f"detections_{video_path.stem}.csv"
        
        # Video writer for output video
        writer = None
        if save_video:
            video_output = output_dir / f"annotated_{video_path.stem}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(video_output), fourcc, fps, (width, height))
        
        # Process frames
        detections = []
        frame_idx = 0
        
        with tqdm(total=n_frames, desc=f"  Frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run inference
                result = inference_detector(self.model, frame)
                
                # Collect detections
                bboxes = result.pred_instances.bboxes.cpu().numpy()
                scores = result.pred_instances.scores.cpu().numpy()
                
                for bbox, score in zip(bboxes, scores):
                    if score >= self.score_thr:
                        x1, y1, x2, y2 = bbox
                        detections.append({
                            'frame': frame_idx,
                            'time_sec': frame_idx / fps_orig,
                            'bbox_x1': float(x1),
                            'bbox_y1': float(y1),
                            'bbox_x2': float(x2),
                            'bbox_y2': float(y2),
                            'width': float(x2 - x1),
                            'height': float(y2 - y1),
                            'score': float(score),
                            'image_width': width,
                            'image_height': height
                        })
                
                # Visualization
                if save_frames or save_video:
                    frame_vis = self._visualize_bbox(frame, result, self.score_thr)
                    
                    if save_frames and frame_idx % max(1, fps_orig // 10) == 0:  # Save every ~10 frames per sec
                        frame_output = output_dir / f"frames" / f"{video_path.stem}_frame_{frame_idx:06d}.jpg"
                        frame_output.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(frame_output), frame_vis)
                    
                    if save_video:
                        writer.write(frame_vis)
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        if writer:
            writer.release()
        
        # Save detections to CSV
        if detections:
            keys = detections[0].keys()
            with open(csv_output, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(detections)
            print(f"  ✓ Saved {len(detections)} detections to {csv_output.name}")
        
        if save_video:
            print(f"  ✓ Saved annotated video")
        
        return detections


def main():
    args = parse_args()
    
    # Expand video patterns
    video_files = []
    for pattern in args.video:
        matches = glob(pattern)
        if matches:
            video_files.extend(matches)
        else:
            if Path(pattern).exists():
                video_files.append(pattern)
    
    if not video_files:
        print(f"No video files found matching: {args.video}")
        return
    
    video_files.sort()
    print(f"Found {len(video_files)} video(s) to process")
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    inference_engine = RTMDetInference(
        args.config,
        args.checkpoint,
        device=args.device,
        score_thr=args.score_thr
    )
    
    # Process all videos
    all_detections = []
    for video_path in video_files:
        detections = inference_engine.process_video(
            video_path,
            args.output,
            save_frames=args.save_frames,
            save_video=args.save_video,
            fps=args.fps
        )
        all_detections.extend(detections)
    
    print(f"\n{'='*60}")
    print(f"Total detections: {len(all_detections)}")
    print(f"Output directory: {args.output}")


if __name__ == "__main__":
    main()
