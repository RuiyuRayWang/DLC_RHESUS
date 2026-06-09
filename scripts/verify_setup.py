#!/usr/bin/env python3
"""
Quick verification that OpenCV can properly read the HEVC videos sequentially.
Run this before starting the full analysis to ensure everything is working.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

def verify_video_loading():
    """Verify we can load and process HEVC videos."""
    
    video_dir = Path('/home/luolab/PROJECTS/PAL_ENHANCER/data/oft_rhp/134569R')
    test_video = list(video_dir.glob('*.mp4'))[0]
    
    print("="*70)
    print("HEVC VIDEO LOADING VERIFICATION")
    print("="*70)
    print(f"\nTest video: {test_video.name}")
    
    # Open video
    cap = cv2.VideoCapture(str(test_video))
    
    if not cap.isOpened():
        print("❌ FAILED: Could not open video")
        print("\nTroubleshooting:")
        print("  1. Verify OpenCV is installed: conda list opencv")
        print("  2. Verify ffmpeg is installed: which ffmpeg")
        print("  3. Reinstall: conda install -c conda-forge opencv ffmpeg -y")
        return False
    
    print("✅ Video opened successfully")
    
    # Get properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo Properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.1f} seconds")
    
    # Test sequential reading
    print(f"\nTesting sequential frame reading...")
    frame_count = 0
    frame_shapes = set()
    
    for i in range(100):
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame_shapes.add(frame.shape)
    
    cap.release()
    
    if frame_count < 100:
        print(f"⚠️  Warning: Only read {frame_count}/100 frames")
        return False
    
    print(f"✅ Successfully read {frame_count} frames")
    print(f"   Frame shape: {frame_shapes.pop()}")
    
    # Verify we can process frames
    print(f"\nTesting frame processing...")
    cap = cv2.VideoCapture(str(test_video))
    ret, frame = cap.read()
    
    if ret:
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply simple threshold
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        print(f"✅ Frame processing successful")
        print(f"   RGB shape: {frame_rgb.shape}")
        print(f"   Gray shape: {gray.shape}")
    else:
        print(f"❌ FAILED: Could not process frames")
        cap.release()
        return False
    
    cap.release()
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED - Ready for analysis!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Open Jupyter: jupyter lab")
    print("  2. Navigate to: notebooks/oft_video_analysis.ipynb")
    print("  3. Run the notebook cells sequentially")
    print()
    
    return True

if __name__ == '__main__':
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}\n")
    except ImportError:
        print("❌ ERROR: OpenCV not installed")
        print("\nInstall with:")
        print("  conda activate deeplabcut")
        print("  conda install -c conda-forge opencv -y")
        sys.exit(1)
    
    success = verify_video_loading()
    sys.exit(0 if success else 1)
