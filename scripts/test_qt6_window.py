#!/usr/bin/env python3
"""
Minimal test to isolate the Qt6 window issue.
Tests if loading a large video frame interferes with Qt6 window creation.
"""
import cv2
import numpy as np
from pathlib import Path

print("=" * 60)
print("Qt6 Window Creation Test with Large Frame")
print("=" * 60)

# Test 1: Simple window with small frame (like the diagnostic)
print("\n1. Testing with small frame (100x100)...")
try:
    cv2.startWindowThread()
    cv2.namedWindow("test_small", cv2.WINDOW_NORMAL)
    small_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(small_frame, "SMALL", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("test_small", small_frame)
    cv2.waitKey(1)
    cv2.setMouseCallback("test_small", lambda *args: None)
    print("   ✓ Small frame test passed")
    cv2.destroyWindow("test_small")
except cv2.error as e:
    print(f"   ✗ Failed: {e}")

# Test 2: Window with large frame (like calibration script)
print("\n2. Testing with large frame (3840x2160)...")
try:
    cv2.namedWindow("test_large", cv2.WINDOW_NORMAL)
    import time
    time.sleep(0.15)
    large_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    cv2.putText(large_frame, "LARGE 4K FRAME", (100, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
    cv2.imshow("test_large", large_frame)
    cv2.waitKey(1)
    cv2.setMouseCallback("test_large", lambda *args: None)
    print("   ✓ Large frame test passed")
    cv2.destroyWindow("test_large")
except cv2.error as e:
    print(f"   ✗ Failed: {e}")

# Test 3: Load actual video frame, THEN create window
print("\n3. Testing: Load video frame FIRST, then create window...")
video_path = Path("/home/luolab/PROJECTS/PAL_ENHANCER/data/oft_rhp/H1705011/2026-05-25/D02_20260525161934.mp4")
if video_path.exists():
    try:
        # Load frame first (like calibration script does in __init__)
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 600)
        ret, video_frame = cap.read()
        cap.release()
        
        if ret:
            print(f"   Loaded frame: {video_frame.shape}")
            
            # NOW try to create window (like calibration script does in run())
            cv2.namedWindow("test_video", cv2.WINDOW_NORMAL)
            import time
            time.sleep(0.15)
            cv2.imshow("test_video", video_frame)
            cv2.waitKey(1)
            cv2.setMouseCallback("test_video", lambda *args: None)
            print("   ✓ Video frame test passed")
            cv2.destroyWindow("test_video")
        else:
            print("   ✗ Failed to read video frame")
    except cv2.error as e:
        print(f"   ✗ Failed: {e}")
else:
    print("   ⚠ Video not found, skipping")

# Test 4: Window name with special characters (KNOWN TO FAIL with Qt6)
print("\n4. Testing: Window name with em-dash and pipes...")
print("   NOTE: This test is EXPECTED to fail with Qt6 backend")
try:
    bad_name = "Test — special | chars"
    cv2.namedWindow(bad_name, cv2.WINDOW_NORMAL)
    import time
    time.sleep(0.15)
    test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imshow(bad_name, test_frame)
    cv2.waitKey(1)
    cv2.setMouseCallback(bad_name, lambda *args: None)
    print("   ✓ Special characters test passed (unexpected!)")
    cv2.destroyWindow(bad_name)
except cv2.error as e:
    print(f"   ✗ Failed as expected: Qt6 doesn't support em-dash/pipe in window names")
    print(f"   → Solution: Use simple names like 'window_name' instead")

# Test 5: Simple window name (should always work)
print("\n5. Testing: Simple alphanumeric window name...")
try:
    good_name = "Calibrate_D2"
    cv2.namedWindow(good_name, cv2.WINDOW_NORMAL)
    import time
    time.sleep(0.15)
    test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(test_frame, "SIMPLE", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow(good_name, test_frame)
    cv2.waitKey(1)
    cv2.setMouseCallback(good_name, lambda *args: None)
    print("   ✓ Simple name test passed")
    cv2.destroyWindow(good_name)
except cv2.error as e:
    print(f"   ✗ Failed: {e}")

print("\n" + "=" * 60)
print("Tests complete. Press any key in any window to close...")
cv2.waitKey(3000)
cv2.destroyAllWindows()
