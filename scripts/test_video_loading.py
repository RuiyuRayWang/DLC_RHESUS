#!/usr/bin/env python3
"""
Test script to verify video loading capabilities for Hikvision .mp4 files.
These videos may require special handling as they're recorded with Hikvision cameras.
"""

import os
import sys
from pathlib import Path
import subprocess
import json

# Video file to test
VIDEO_DIR = Path('/home/luolab/PROJECTS/PAL_ENHANCER/data/oft_rhp/134569R')
TEST_VIDEO = VIDEO_DIR / 'D02_20260512094624.mp4'

print("=" * 80)
print("VIDEO LOADING VERIFICATION TEST")
print("=" * 80)
print(f"\nTest video: {TEST_VIDEO}")
print(f"File exists: {TEST_VIDEO.exists()}")

if TEST_VIDEO.exists():
    file_size = TEST_VIDEO.stat().st_size / (1024 * 1024)  # MB
    print(f"File size: {file_size:.2f} MB")

print("\n" + "=" * 80)
print("TEST 1: Check file type and codec with ffprobe")
print("=" * 80)

try:
    # Use ffprobe to get detailed video information
    cmd = [
        'ffprobe', 
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        str(TEST_VIDEO)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    
    if result.returncode == 0:
        info = json.loads(result.stdout)
        
        print("\n📹 FORMAT INFORMATION:")
        if 'format' in info:
            fmt = info['format']
            print(f"  Format: {fmt.get('format_name', 'Unknown')}")
            print(f"  Duration: {float(fmt.get('duration', 0)):.2f} seconds")
            print(f"  Bitrate: {int(fmt.get('bit_rate', 0)) / 1000:.0f} kbps")
        
        print("\n📹 STREAM INFORMATION:")
        if 'streams' in info:
            for i, stream in enumerate(info['streams']):
                if stream['codec_type'] == 'video':
                    print(f"\n  Video Stream {i}:")
                    print(f"    Codec: {stream.get('codec_name', 'Unknown')}")
                    print(f"    Codec (long): {stream.get('codec_long_name', 'Unknown')}")
                    print(f"    Profile: {stream.get('profile', 'Unknown')}")
                    print(f"    Resolution: {stream.get('width', '?')}x{stream.get('height', '?')}")
                    print(f"    FPS: {stream.get('r_frame_rate', 'Unknown')}")
                    print(f"    Pixel format: {stream.get('pix_fmt', 'Unknown')}")
                    print(f"    Total frames: {stream.get('nb_frames', 'Unknown')}")
                elif stream['codec_type'] == 'audio':
                    print(f"\n  Audio Stream {i}:")
                    print(f"    Codec: {stream.get('codec_name', 'Unknown')}")
        
        print("\n✅ ffprobe successfully read the video file")
    else:
        print(f"❌ ffprobe failed: {result.stderr}")
        
except FileNotFoundError:
    print("⚠️  ffprobe not found. Install with: sudo apt-get install ffmpeg")
except subprocess.TimeoutExpired:
    print("⚠️  ffprobe timed out")
except Exception as e:
    print(f"❌ Error running ffprobe: {e}")

print("\n" + "=" * 80)
print("TEST 2: Load video with OpenCV")
print("=" * 80)

try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
    
    cap = cv2.VideoCapture(str(TEST_VIDEO))
    
    if not cap.isOpened():
        print("❌ OpenCV failed to open the video file")
        print("\nPossible issues:")
        print("  - Video codec not supported by OpenCV build")
        print("  - File corruption")
        print("  - Proprietary Hikvision format")
    else:
        print("✅ OpenCV successfully opened the video")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        
        # Decode fourcc
        fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        print(f"\n  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Frame count: {frame_count}")
        print(f"  Duration: {frame_count/fps:.2f} seconds")
        print(f"  FourCC: {fourcc_str} ({fourcc})")
        
        # Try to read first frame
        print("\n  Attempting to read first frame...")
        ret, frame = cap.read()
        
        if ret:
            print(f"  ✅ Successfully read frame")
            print(f"     Shape: {frame.shape}")
            print(f"     Dtype: {frame.dtype}")
            print(f"     Min/Max values: {frame.min()}/{frame.max()}")
        else:
            print("  ❌ Failed to read frame")
        
        # Try to read multiple frames
        print("\n  Attempting to read first 10 frames...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        successful_reads = 0
        
        for i in range(10):
            ret, frame = cap.read()
            if ret:
                successful_reads += 1
            else:
                print(f"    Failed at frame {i}")
                break
        
        print(f"  Successfully read {successful_reads}/10 frames")
        
        # Try random access
        print("\n  Testing random frame access...")
        test_frames = [0, frame_count//4, frame_count//2, frame_count-1]
        random_access_works = True
        
        for frame_num in test_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                print(f"    ❌ Failed to read frame {frame_num}")
                random_access_works = False
            else:
                print(f"    ✅ Read frame {frame_num}")
        
        if random_access_works:
            print("  ✅ Random access works")
        else:
            print("  ⚠️  Random access issues detected")
        
        cap.release()
        
except ImportError:
    print("❌ OpenCV not installed")
    print("Install with: pip install opencv-python")
except Exception as e:
    print(f"❌ Error with OpenCV: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TEST 3: Try decord (alternative video reader)")
print("=" * 80)

try:
    import decord
    from decord import VideoReader, cpu
    
    print(f"Decord version: {decord.__version__}")
    
    vr = VideoReader(str(TEST_VIDEO), ctx=cpu(0))
    
    print(f"✅ Decord successfully opened the video")
    print(f"  Total frames: {len(vr)}")
    print(f"  FPS: {vr.get_avg_fps()}")
    
    # Try to read some frames
    print("\n  Reading sample frames...")
    frame = vr[0]
    print(f"  Frame 0 shape: {frame.shape}")
    
    # Read batch
    frames = vr.get_batch([0, 10, 20, 30]).asnumpy()
    print(f"  Batch read successful: {frames.shape}")
    print("  ✅ Decord works well with this video")
    
except ImportError:
    print("⚠️  Decord not installed")
    print("Install with: pip install decord")
except Exception as e:
    print(f"❌ Error with decord: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TEST 4: Try PyAV (another alternative)")
print("=" * 80)

try:
    import av
    
    print(f"PyAV version: {av.__version__}")
    
    container = av.open(str(TEST_VIDEO))
    
    print(f"✅ PyAV successfully opened the video")
    print(f"  Duration: {container.duration / 1000000:.2f} seconds")
    print(f"  Streams: {len(container.streams)}")
    
    for stream in container.streams:
        if stream.type == 'video':
            print(f"\n  Video stream:")
            print(f"    Codec: {stream.codec_context.name}")
            print(f"    Resolution: {stream.width}x{stream.height}")
            print(f"    FPS: {stream.average_rate}")
            print(f"    Frames: {stream.frames}")
    
    # Try to decode some frames
    print("\n  Decoding first few frames...")
    frame_count = 0
    for frame in container.decode(video=0):
        frame_count += 1
        if frame_count == 1:
            img = frame.to_ndarray(format='rgb24')
            print(f"  First frame shape: {img.shape}")
        if frame_count >= 10:
            break
    
    print(f"  ✅ Successfully decoded {frame_count} frames")
    
    container.close()
    
except ImportError:
    print("⚠️  PyAV not installed")
    print("Install with: pip install av")
except Exception as e:
    print(f"❌ Error with PyAV: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TEST 5: Check OpenCV build info")
print("=" * 80)

try:
    import cv2
    build_info = cv2.getBuildInformation()
    
    # Extract relevant lines
    relevant_lines = []
    capture = False
    for line in build_info.split('\n'):
        if 'Video I/O' in line or 'codec' in line.lower() or 'ffmpeg' in line.lower():
            capture = True
        if capture:
            relevant_lines.append(line)
            if line.strip() == '':
                capture = False
    
    if relevant_lines:
        print("\nRelevant OpenCV build info:")
        print('\n'.join(relevant_lines[:30]))  # First 30 lines
    
except Exception as e:
    print(f"Could not get build info: {e}")

print("\n" + "=" * 80)
print("SUMMARY AND RECOMMENDATIONS")
print("=" * 80)

print("""
Based on the tests above, here are the recommendations:

1. If ffprobe works but OpenCV doesn't:
   - The video uses a codec that OpenCV wasn't compiled with
   - Consider using ffmpeg to convert videos first
   - Or use PyAV/decord as alternative readers

2. If OpenCV works:
   - You're good to go with the standard pipeline
   - But watch for random access issues

3. If none work:
   - Hikvision videos may need proprietary player
   - Consider converting with ffmpeg first:
     ffmpeg -i input.mp4 -c:v libx264 -crf 18 -c:a copy output.mp4

4. Best practice for large batches:
   - Use the reader that works best from tests above
   - Consider converting all videos to a standard format first
   - This will make downstream processing much faster
""")

print("\n" + "=" * 80)
print("Next steps:")
print("  1. Review the test results above")
print("  2. If videos don't load, try: python convert_videos.py")
print("  3. If videos load OK, proceed with analysis notebook")
print("=" * 80)
