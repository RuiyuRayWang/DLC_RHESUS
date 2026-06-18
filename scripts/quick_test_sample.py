#!/usr/bin/env python3
"""
Quick test of model on sampled frames only.
Extracts a few frames from the video and runs inference.
"""

import cv2
import numpy as np
from pathlib import Path
import tempfile
import deeplabcut
from PIL import Image
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / 'deeplabcut_projects' / 'monkey_oft_tracking-luolab-2026-06-15' / 'config.yaml'
VIDEO_PATH = '/media/luolab/memcore/oft_rhp/H1705011/D01_20260525164157.mp4'
SAMPLE_FRAMES = [100, 500, 1000, 5000, 10000]  # Sample frame indices

def extract_sample_frames(video_path, frame_indices, output_dir):
    """Extract specific frames from video."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Video info:")
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps}")
    
    # Filter to valid frame indices
    valid_indices = [i for i in frame_indices if i < total_frames]
    print(f"   Extracting frames: {valid_indices}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_files = []
    
    for idx in valid_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # Save as PNG
            frame_path = output_dir / f"frame_{idx:06d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_files.append(frame_path)
            print(f"   ✅ Extracted frame {idx}")
        else:
            print(f"   ❌ Failed to extract frame {idx}")
    
    cap.release()
    return frame_files

def main():
    print("=" * 70)
    print("Quick Model Test - Sample Frames")
    print("=" * 70)
    
    if not CONFIG_PATH.exists():
        print(f"❌ Config not found: {CONFIG_PATH}")
        return
    
    if not Path(VIDEO_PATH).exists():
        print(f"❌ Video not found: {VIDEO_PATH}")
        return
    
    print(f"\nConfig: {CONFIG_PATH}")
    print(f"Video: {VIDEO_PATH}")
    
    # Create temp directory for sample frames
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_dir = Path(tmpdir) / "sample_frames"
        
        # Extract sample frames
        print("\n🎬 Extracting sample frames...")
        frame_files = extract_sample_frames(VIDEO_PATH, SAMPLE_FRAMES, sample_dir)
        
        if not frame_files:
            print("❌ No frames extracted")
            return
        
        # Run inference on directory of frames
        print("\n🚀 Running inference on sample frames...")
        print(f"   Using shuffle=5 (trained model)")
        
        try:
            # Create a temporary video from the sample frames
            temp_video = Path(tmpdir) / "sample_video.mp4"
            
            # Use ffmpeg to create video from frames
            frame_pattern = str(sample_dir / "frame_%06d.png")
            os.system(f"ffmpeg -y -framerate 1 -i '{frame_pattern}' -c:v libx264 -pix_fmt yuv420p '{temp_video}' 2>/dev/null")
            
            if temp_video.exists():
                print(f"   Created sample video: {temp_video}")
                
                # Analyze the temp video
                deeplabcut.analyze_videos(
                    str(CONFIG_PATH),
                    [str(temp_video)],
                    shuffle=5,
                    batch_size=1,
                    save_as_csv=True
                )
                
                print("\n✅ Inference completed!")
                
                # Find and display the results
                import glob
                csv_files = glob.glob(str(sample_dir.parent / "**/*DLC*.csv"), recursive=True)
                
                if csv_files:
                    print(f"\n📊 Predictions:")
                    import pandas as pd
                    
                    for csv_file in csv_files[:1]:  # Show first CSV
                        df = pd.read_csv(csv_file, header=[0, 1, 2])
                        print(f"\n   File: {Path(csv_file).name}")
                        print(f"   Shape: {df.shape}")
                        print(f"\n   Sample predictions:")
                        print(df.head(10).to_string())
                        
                        # Calculate confidence stats
                        likelihood_cols = [c for c in df.columns if 'likelihood' in c]
                        if likelihood_cols:
                            likelihoods = df[likelihood_cols[0]].values
                            print(f"\n   Confidence statistics:")
                            print(f"      Mean: {np.mean(likelihoods):.3f}")
                            print(f"      Std: {np.std(likelihoods):.3f}")
                            print(f"      Min: {np.min(likelihoods):.3f}")
                            print(f"      Max: {np.max(likelihoods):.3f}")
        
        except Exception as e:
            print(f"\n❌ Error during inference: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
