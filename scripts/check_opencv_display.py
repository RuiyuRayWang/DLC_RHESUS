#!/usr/bin/env python3
"""
Diagnostic tool to check OpenCV display capabilities.

Helps troubleshoot window creation issues in camera calibration and manual labeling.
"""
import os
import sys

def main():
    print("=" * 60)
    print("OpenCV Display Diagnostic")
    print("=" * 60)
    
    # Check DISPLAY environment variable
    display = os.environ.get('DISPLAY', '')
    print(f"\n1. DISPLAY environment variable: {display or '(not set)'}")
    if not display:
        print("   ⚠  No DISPLAY set — GUI windows will fail")
        print("   → Run 'export DISPLAY=:0' or enable X11 forwarding (ssh -X)")
    
    # Check if running via SSH
    ssh_conn = os.environ.get('SSH_CONNECTION', '')
    ssh_client = os.environ.get('SSH_CLIENT', '')
    if ssh_conn or ssh_client:
        print(f"\n2. Running via SSH: YES")
        print("   → Ensure X11 forwarding is enabled (ssh -X or ssh -Y)")
    else:
        print(f"\n2. Running via SSH: NO (local session)")
    
    # Try importing OpenCV
    print("\n3. OpenCV import:")
    try:
        import cv2
        print(f"   ✓ OpenCV {cv2.__version__} imported successfully")
    except ImportError as e:
        print(f"   ✗ Failed to import OpenCV: {e}")
        sys.exit(1)
    
    # Check OpenCV build info
    print("\n4. OpenCV build info:")
    build_info = cv2.getBuildInformation()
    
    # Extract GUI backend
    for line in build_info.split('\n'):
        if 'GUI:' in line or 'GTK' in line or 'QT' in line:
            print(f"   {line.strip()}")
    
    # Check if Qt or GTK is available
    has_qt = 'QT' in build_info or 'Qt' in build_info
    has_gtk = 'GTK' in build_info
    
    if has_qt:
        print("\n   Backend: Qt (can have threading issues)")
        print("   → If problems persist, rebuild OpenCV with GTK")
    elif has_gtk:
        print("\n   Backend: GTK")
    else:
        print("\n   ⚠  No GUI backend detected")
    
    # Test window creation
    print("\n5. Testing window creation:")
    try:
        window_name = "test_window"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Check if window was created
        prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
        if prop < 0:
            print("   ✗ Window creation failed (property check returned negative)")
            return False
        
        print("   ✓ namedWindow succeeded")
        
        # Try setting mouse callback
        def dummy_callback(event, x, y, flags, param):
            pass
        
        cv2.setMouseCallback(window_name, dummy_callback)
        print("   ✓ setMouseCallback succeeded")
        
        # Try showing an image
        import numpy as np
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(test_img, "TEST", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(window_name, test_img)
        print("   ✓ imshow succeeded")
        
        print("\n   Window should appear now. Press any key to close...")
        cv2.waitKey(3000)  # Wait 3 seconds or key press
        cv2.destroyAllWindows()
        
        print("\n✅ All tests passed — OpenCV GUI should work!")
        return True
        
    except cv2.error as e:
        print(f"   ✗ OpenCV error: {e}")
        return False
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
        return False
    finally:
        try:
            cv2.destroyAllWindows()
        except:
            pass
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
