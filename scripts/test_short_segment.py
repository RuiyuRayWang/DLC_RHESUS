#!/usr/bin/env python3
"""
Test script: Run inference on a short segment of video (~300 frames)
and save predictions for visual examination.

Minimal template for running DeepLabCut batch inference.
"""

import cv2
import numpy as np
from pathlib import Path
import tempfile
import subprocess
import deeplabcut
import pandas as pd
import shutil
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / 'deeplabcut_projects' / 'monkey_oft_tracking_D1-luolab-2026-06-19' / 'config.yaml'
VIDEO_PATH = '/home/ray/DLC_RHESUS/data/oft_rhp/H1705011/2026-06-17/D01_20260617131918.mp4'
NUM_FRAMES = 100
START_FRAME = 0


def parse_args():
    parser = argparse.ArgumentParser(description='Run DLC inference on a short video clip.')
    parser.add_argument('--video', default=VIDEO_PATH, help='Path to input video')
    parser.add_argument('--frames', type=int, default=NUM_FRAMES, help='Number of frames to extract')
    parser.add_argument('--start-frame', type=int, default=START_FRAME, help='Start frame index for clip extraction')
    return parser.parse_args()

def extract_first_n_frames(video_path, n_frames, output_dir, start_frame=0):
    """Extract N frames from video starting at start_frame and save as PNGs."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n📹 Video info:")
    print(f"   Resolution: {w}×{h}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"   Start frame: {start_frame}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_files = []
    
    if start_frame < 0:
        start_frame = 0
    if start_frame >= total_frames:
        raise ValueError(f"start_frame {start_frame} >= total frames {total_frames}")

    # For HEVC videos, this seek may be slow but is acceptable for short test clips.
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print(f"\n🎬 Extracting {n_frames} frames from frame {start_frame}...")
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"   Reached end of video at frame {start_frame + i}")
            break
        
        frame_path = output_dir / f"frame_{i:06d}.png"
        cv2.imwrite(str(frame_path), frame)
        frame_files.append(frame_path)
        
        if (i + 1) % 50 == 0:
            print(f"   ✅ Extracted {i + 1} frames")
    
    cap.release()
    print(f"   ✅ Total extracted: {len(frame_files)} frames")
    
    return frame_files, fps, w, h

def create_video_from_frames(frame_dir, output_video, fps):
    """Create video from PNG frames using ffmpeg."""
    frame_pattern = str(frame_dir / "frame_%06d.png")
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', frame_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'fast',  # Fast encoding
        str(output_video)
    ]
    
    print(f"\n📦 Creating test video from frames...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"   ⚠️  ffmpeg error: {result.stderr[:200]}")
        raise RuntimeError("Failed to create video")
    
    print(f"   ✅ Video created: {output_video}")
    return output_video


def find_prediction_columns(df):
    """Find x, y, likelihood columns from DLC CSV with MultiIndex header."""
    x_col, y_col, p_col = None, None, None

    for col in df.columns:
        if not isinstance(col, tuple) or len(col) < 3:
            continue
        coord = str(col[-1]).strip().lower()
        if coord == 'x' and x_col is None:
            x_col = col
        elif coord == 'y' and y_col is None:
            y_col = col
        elif coord == 'likelihood' and p_col is None:
            p_col = col

    return x_col, y_col, p_col


def create_overlay_video(input_video, output_video, xs, ys, ps, fps, pcutoff=0.5):
    """Draw predicted head points on video frames."""
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for overlay: {input_video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )

    n = min(len(xs), len(ys), len(ps))
    frame_idx = 0
    while frame_idx < n:
        ret, frame = cap.read()
        if not ret:
            break

        x, y, conf = xs[frame_idx], ys[frame_idx], ps[frame_idx]
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(conf):
            if conf >= pcutoff:
                cv2.circle(frame, (int(round(x)), int(round(y))), 10, (0, 255, 0), -1)
                cv2.putText(
                    frame,
                    f"p={conf:.2f}",
                    (max(10, int(round(x)) + 12), max(25, int(round(y)) - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.circle(frame, (int(round(x)), int(round(y))), 8, (0, 0, 255), 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

def main():
    args = parse_args()
    video_path = args.video
    num_frames = args.frames
    start_frame = args.start_frame

    print("=" * 70)
    print("DeepLabCut Test - Short Video Segment")
    print("=" * 70)
    
    if not CONFIG_PATH.exists():
        print(f"❌ Config not found: {CONFIG_PATH}")
        return
    
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    print(f"\nConfig: {CONFIG_PATH}")
    print(f"Test video: {video_path}")
    print(f"Frames to extract: {num_frames}")
    print(f"Start frame: {start_frame}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        frame_dir = tmpdir / "frames"
        test_video = tmpdir / "test_segment.mp4"
        
        # Step 1: Extract frames
        try:
            frame_files, fps, w, h = extract_first_n_frames(video_path, num_frames, frame_dir, start_frame=start_frame)
        except Exception as e:
            print(f"❌ Error extracting frames: {e}")
            return
        
        if not frame_files:
            print("❌ No frames extracted")
            return
        
        # Step 2: Create video from frames
        try:
            create_video_from_frames(frame_dir, test_video, fps)
        except Exception as e:
            print(f"❌ Error creating video: {e}")
            return
        
        # Step 3: Run inference
        print(f"\n🚀 Running inference with DLC...")
        print(f"   Shuffle: 1 (trained model)")
        print(f"   Batch size: 1 (to handle 4K resolution)")
        
        try:
            deeplabcut.analyze_videos(
                str(CONFIG_PATH),
                [str(test_video)],
                shuffle=1,
                batch_size=1,
                save_as_csv=True,
                gputouse=0
            )
            
            print(f"\n✅ Inference completed!")
            
        except Exception as e:
            print(f"\n❌ Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Step 4: Find and examine predictions
        import glob
        csv_pattern = str(tmpdir / "**/*DLC*.csv")
        csv_files = glob.glob(csv_pattern, recursive=True)
        
        if csv_files:
            print(f"\n📊 Predictions found:")
            csv_file = csv_files[0]
            print(f"   File: {Path(csv_file).name}")
            
            # Read predictions
            df = pd.read_csv(csv_file, header=[0, 1, 2])
            
            print(f"   Shape: {df.shape}")
            print(f"\n   Prediction columns: {df.columns.tolist()}")
            
            # Extract coordinates and confidence
            x_col, y_col, p_col = find_prediction_columns(df)
            if x_col is None or y_col is None or p_col is None:
                print("❌ Could not locate x/y/likelihood columns in prediction CSV")
                print(f"   Columns: {df.columns.tolist()}")
                return

            xs = pd.to_numeric(df[x_col], errors='coerce').to_numpy()
            ys = pd.to_numeric(df[y_col], errors='coerce').to_numpy()
            confs = pd.to_numeric(df[p_col], errors='coerce').to_numpy()

            print(f"\n📈 Sample predictions (first 10 frames):")
            print(f"   {'Frame':<6} {'X':<8} {'Y':<8} {'Confidence':<12}")
            print(f"   {'-'*34}")
            
            for i in range(min(10, len(df))):
                x = xs[i] if np.isfinite(xs[i]) else 0
                y = ys[i] if np.isfinite(ys[i]) else 0
                conf = confs[i] if np.isfinite(confs[i]) else 0
                print(f"   {i:<6} {x:<8.1f} {y:<8.1f} {conf:<12.3f}")
            
            # Statistics
            valid_confs = confs[np.isfinite(confs)]
            if len(valid_confs) > 0:
                print(f"\n📋 Confidence Statistics:")
                print(f"   Mean:     {np.mean(valid_confs):.3f}")
                print(f"   Std:      {np.std(valid_confs):.3f}")
                print(f"   Min:      {np.min(valid_confs):.3f}")
                print(f"   Max:      {np.max(valid_confs):.3f}")
                print(f"   Median:   {np.median(valid_confs):.3f}")
                
                # Count high-confidence predictions
                high_conf = np.sum(valid_confs > 0.9)
                print(f"\n   High confidence (>0.9): {high_conf}/{len(valid_confs)} ({100*high_conf/len(valid_confs):.1f}%)")
            
            # Save predictions to a readable CSV
            output_csv = PROJECT_ROOT / 'results' / 'test_predictions.csv'
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            flat_df = pd.DataFrame({
                'frame': np.arange(len(xs), dtype=int),
                'x': xs,
                'y': ys,
                'likelihood': confs,
            })
            flat_df.to_csv(output_csv, index=False)
            print(f"\n💾 Predictions saved to: {output_csv}")
            
            # Create labeled video using custom overlay (robust against DLC plotting issues)
            print(f"\n🎨 Creating labeled video with predictions...")
            try:
                tmp_overlay = tmpdir / 'test_segment_overlay.mp4'
                create_overlay_video(test_video, tmp_overlay, xs, ys, confs, fps, pcutoff=0.5)
                output_video = PROJECT_ROOT / 'results' / 'test_predictions_labeled.mp4'
                shutil.copy(tmp_overlay, output_video)
                print(f"   ✅ Labeled video saved to: {output_video}")
                print(f"   💡 Use: ffplay {output_video}")
                
            except Exception as e:
                print(f"   ⚠️  Could not create labeled video: {e}")
        
        else:
            print(f"❌ No prediction CSV found")
            print(f"   Checked: {csv_pattern}")
    
    print(f"\n" + "=" * 70)
    print("✅ Test complete!")
    print("=" * 70)

if __name__ == '__main__':
    main()
