#!/usr/bin/env python3
"""
Manual frame labeler for monkey OFT tracking (multi-camera aware).

Opens an OpenCV window showing one frame at a time from a selected camera.
Left-click on the monkey to record its centroid. Clicking auto-advances.

Usage:
    conda activate video_tracking
    python scripts/manual_labeler.py               # interactive camera select
    python scripts/manual_labeler.py --camera D3
    python scripts/manual_labeler.py --video path/to/video.mp4 --camera D1
    python scripts/manual_labeler.py --help

Controls:
    Left-click  : label monkey position and advance to next frame
    Right-click : label monkey position WITHOUT advancing
    M           : mark frame as "no monkey visible" and advance
    N / →       : next frame (skip without labeling)
    P / ←       : previous frame
    U           : undo label on current frame
    Q / ESC     : save and quit
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys
from typing import Dict, Optional, Tuple


# ── Defaults / constants ───────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_BASE = PROJECT_ROOT / 'data' / 'oft_rhp' / 'H1705011' / '2026-05-25'
OUTPUT_BASE = PROJECT_ROOT / 'results' / 'oft_analysis' / 'H1705011' / '2026-05-25'

# Camera → video file prefix and quadrant label
CAMERA_INFO = {
    'D1': {'prefix': 'D01_', 'quadrant': 'top-left'},
    'D2': {'prefix': 'D02_', 'quadrant': 'top-right'},
    'D3': {'prefix': 'D03_', 'quadrant': 'bottom-left'},
    'D4': {'prefix': 'D04_', 'quadrant': 'bottom-right'},
}

DEFAULT_SAMPLE_EVERY = 20   # label every 20th frame → 1 Hz at 20 fps
DEFAULT_FPS = 20
DEFAULT_SCALE = 0.4         # scale 4K → ~1536×864 for display

# Sentinel value stored in the CSV for "no monkey visible" frames
NO_MONKEY_SENTINEL = -1

# Colour palette
C_YELLOW = (0,   220, 220)
C_GREEN  = (80,  200, 80)
C_RED    = (80,  80,  220)
C_GRAY   = (180, 180, 180)


# ── Core class ─────────────────────────────────────────────────────────────────

class ManualLabeler:
    """
    OpenCV-based GUI for manually labeling monkey centroid per frame.
    Supports multiple cameras; outputs one CSV per camera.
    """

    def __init__(self, camera: str, video_path, output_csv,
                 sample_every=DEFAULT_SAMPLE_EVERY,
                 display_scale=DEFAULT_SCALE, fps=DEFAULT_FPS):
        self.camera = camera
        self.video_path = Path(video_path)
        self.output_csv = Path(output_csv)
        self.sample_every = sample_every
        self.display_scale = display_scale
        self.fps = fps

        print(f"\nCamera {camera} ({CAMERA_INFO[camera]['quadrant']} quadrant)")
        print(f"Loading frames from: {self.video_path.name}")
        print(f"  sampling every {sample_every} frames  ({fps / sample_every:.2f} Hz)")
        self.frames, self.frame_nums = self._load_frames()
        print(f"  → {len(self.frames)} frames to label\n")

        # labels: { video_frame_number -> (x, y) in ORIGINAL pixel coords }
        #   x == NO_MONKEY_SENTINEL → monkey not visible in this frame
        self.labels: Dict[int, Tuple[int, int]] = {}

        # Load existing CSV so a session can be resumed
        if self.output_csv.exists():
            self._load_existing()

        self.current_idx = 0
        self._advance_requested = False

        # Simple window name (Qt6 sensitive to special characters)
        self.win = f"Label_{camera}"
        self.win_title = (f"Camera {camera} ({CAMERA_INFO[camera]['quadrant']}) - "
                         "L-click=label  M=no-monkey  N=next  P=prev  U=undo  Q=quit")

    # ── Frame loading ──────────────────────────────────────────────────────────

    def _load_frames(self):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            sys.exit(f"ERROR: Cannot open video: {self.video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        expected = total // self.sample_every + 1
        print(f"  total frames in video: {total}  →  expecting ~{expected} samples")

        frames, frame_nums = [], []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.sample_every == 0:
                frames.append(frame)
                frame_nums.append(idx)
            idx += 1
        cap.release()
        return frames, frame_nums

    def _load_existing(self):
        existing = pd.read_csv(self.output_csv)
        loaded = 0
        for _, row in existing.iterrows():
            fn = int(row['video_frame'])
            cx, cy = int(row['centroid_x']), int(row['centroid_y'])
            self.labels[fn] = (cx, cy)
            loaded += 1
        print(f"Resumed: loaded {loaded} existing labels from {self.output_csv}")

    # ── Display ────────────────────────────────────────────────────────────────

    def _render(self):
        """Return a display-ready copy of the current frame with annotations."""
        orig = self.frames[self.current_idx]
        h, w = orig.shape[:2]
        dw = int(w * self.display_scale)
        dh = int(h * self.display_scale)
        disp = cv2.resize(orig, (dw, dh), interpolation=cv2.INTER_AREA)

        frame_num = self.frame_nums[self.current_idx]
        n_labeled    = sum(1 for v in self.labels.values() if v[0] != NO_MONKEY_SENTINEL)
        n_no_monkey  = sum(1 for v in self.labels.values() if v[0] == NO_MONKEY_SENTINEL)

        # ── Draw label marker (or "no monkey" indicator) ──
        if frame_num in self.labels:
            ox, oy = self.labels[frame_num]
            if ox == NO_MONKEY_SENTINEL:
                # "No monkey" — show red X in centre
                cx, cy = dw // 2, dh // 2
                cv2.drawMarker(disp, (cx, cy), C_RED,
                               cv2.MARKER_TILTED_CROSS, thickness=4, markerSize=80)
                cv2.putText(disp, 'NO MONKEY', (cx - 100, cy + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, C_RED, 3, cv2.LINE_AA)
            else:
                dx = int(ox * self.display_scale)
                dy = int(oy * self.display_scale)
                cv2.drawMarker(disp, (dx, dy), C_GREEN,
                               cv2.MARKER_CROSS, thickness=3, markerSize=50)
                cv2.circle(disp, (dx, dy), 30, C_GREEN, 2)

        # ── Header bar ──
        cv2.rectangle(disp, (0, 0), (dw, 50), (30, 30, 30), -1)
        status = (f"Cam {self.camera} | "
                  f"Frame {self.current_idx + 1}/{len(self.frames)}"
                  f"  #{frame_num} ({frame_num / self.fps:.1f}s)"
                  f"  |  labeled: {n_labeled}  no-monkey: {n_no_monkey}")
        cv2.putText(disp, status, (10, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, C_YELLOW, 2, cv2.LINE_AA)

        # ── Footer bar ──
        cv2.rectangle(disp, (0, dh - 35), (dw, dh), (30, 30, 30), -1)
        hint = ("L-click=label & next   R-click=label only   "
                "M=no monkey   N=next   P=prev   U=undo   Q=save & quit")
        cv2.putText(disp, hint, (10, dh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_GRAY, 1, cv2.LINE_AA)

        # ── Progress bar (green=labeled, red=no-monkey, dark=unlabeled) ──
        total = len(self.frames)
        if total > 0:
            cv2.rectangle(disp, (0, dh - 38), (dw, dh - 35), (60, 60, 60), -1)
            for fn, val in self.labels.items():
                # Find display position proportional to frame index
                try:
                    fi = self.frame_nums.index(fn)
                except ValueError:
                    continue
                bx = int(fi / total * dw)
                col = C_RED if val[0] == NO_MONKEY_SENTINEL else C_GREEN
                cv2.line(disp, (bx, dh - 38), (bx, dh - 35), col, 2)

        return disp

    # ── Mouse callback ─────────────────────────────────────────────────────────

    def _mouse_cb(self, event, mx, my, flags, param):
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            # Convert display coords → original pixel coords
            ox = int(mx / self.display_scale)
            oy = int(my / self.display_scale)
            frame_num = self.frame_nums[self.current_idx]
            self.labels[frame_num] = (ox, oy)
            if event == cv2.EVENT_LBUTTONDOWN:
                self._advance_requested = True

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self):
        # Initialize Qt6 window system early
        print(f"\n  Opening window: {self.win_title}")
        
        try:
            # Try to start the window thread (helps with Qt6)
            try:
                cv2.startWindowThread()
            except:
                pass  # Not all OpenCV builds support this
            
            # Create window with simple name
            cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
            
            # Set window title if possible
            try:
                cv2.setWindowTitle(self.win, self.win_title)
            except:
                pass
            
            # Let Qt6 initialize (critical)
            import time
            time.sleep(0.15)
            
            # Show test frame to ensure window is ready
            cv2.imshow(self.win, self._render())
            cv2.waitKey(1)
            
            # NOW set mouse callback after window is initialized
            cv2.setMouseCallback(self.win, self._mouse_cb)
            
        except cv2.error as e:
            print("\n❌ ERROR: Failed to create OpenCV window with Qt6.")
            print(f"   Details: {e}")
            print("\n   This is a known Qt6 issue with OpenCV 4.13.0.")
            print("\n   Possible causes:")
            print("   1. Qt6 backend has compatibility issues")
            print("   2. No X11 display available (check DISPLAY variable)")
            print("   3. Running via SSH without X11 forwarding (use ssh -X)")
            print("\n   Solutions:")
            print("   A. Check display: echo $DISPLAY")
            print("   B. Test GUI: python scripts/check_opencv_display.py")
            print("   C. Consider rebuilding OpenCV with GTK instead of Qt")
            raise SystemExit(1)

        print("\nWindow opened. Label the monkey by clicking on it.")
        print("Press Q or ESC to save and quit at any time.\n")

        while True:
            cv2.imshow(self.win, self._render())

            # Handle auto-advance from mouse click
            if self._advance_requested:
                self._advance_requested = False
                if self.current_idx < len(self.frames) - 1:
                    self.current_idx += 1
                else:
                    print("All frames labeled!")
                    break

            key = cv2.waitKey(30) & 0xFF

            if key in (ord('q'), 27):      # Q or ESC → save & quit
                break
            elif key in (ord('n'), 83):    # N or right arrow
                self.current_idx = min(len(self.frames) - 1, self.current_idx + 1)
            elif key in (ord('p'), 81):    # P or left arrow
                self.current_idx = max(0, self.current_idx - 1)
            elif key == ord('u'):          # U → undo
                fn = self.frame_nums[self.current_idx]
                if fn in self.labels:
                    del self.labels[fn]
            elif key == ord('m'):          # M → mark "no monkey visible"
                fn = self.frame_nums[self.current_idx]
                self.labels[fn] = (NO_MONKEY_SENTINEL, NO_MONKEY_SENTINEL)
                self._advance_requested = True

        cv2.destroyAllWindows()
        self._save()

    # ── Export ─────────────────────────────────────────────────────────────────

    def _save(self):
        # Parse video start timestamp from filename: D0X_YYYYMMDDHHMMSS.mp4
        video_timestamp = self._parse_video_timestamp(self.video_path)
        
        records = [
            {
                'camera':      self.camera,
                'video_file':  self.video_path.name,
                'video_timestamp': video_timestamp,
                'video_frame': fn,
                'time_sec':    fn / self.fps,
                'centroid_x':  x,
                'centroid_y':  y,
                'monkey_visible': (x != NO_MONKEY_SENTINEL),
            }
            for fn, (x, y) in sorted(self.labels.items())
        ]
        columns = ['camera', 'video_file', 'video_timestamp', 'video_frame',
                   'time_sec', 'centroid_x', 'centroid_y', 'monkey_visible']
        df = pd.DataFrame(records, columns=columns)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

        if df.empty:
            print(f"\n⚠  No labels recorded — output file not written.")
            return df

        df.to_csv(self.output_csv, index=False)
        n_vis  = df['monkey_visible'].sum()
        n_miss = (~df['monkey_visible']).sum()
        print(f"\n✅  Saved {len(df)} labels → {self.output_csv}")
        print(f"   monkey visible: {n_vis}   no-monkey: {n_miss}")
        print(f"   video start timestamp: {video_timestamp}")
        return df
    
    def _parse_video_timestamp(self, video_path: Path) -> str:
        """
        Extract timestamp from video filename.
        Format: D0X_YYYYMMDDHHMMSS.mp4 → YYYY-MM-DD HH:MM:SS
        """
        import re
        from datetime import datetime
        
        match = re.search(r'D0[1-4]_(\d{14})', video_path.name)
        if not match:
            return 'unknown'
        
        ts_str = match.group(1)  # e.g., '20260525164012'
        dt = datetime.strptime(ts_str, '%Y%m%d%H%M%S')
        return dt.strftime('%Y-%m-%d %H:%M:%S')  # '2026-05-25 16:40:12'


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_video(camera: str, base_dir: Path) -> Path:
    prefix  = CAMERA_INFO[camera]['prefix']      # e.g. 'D01_'
    matches = sorted(base_dir.glob(f"{prefix}*.mp4"))
    if not matches:
        sys.exit(f"ERROR: No video found for camera {camera} in {base_dir}\n"
                 f"       Pass --video explicitly.")
    if len(matches) > 1:
        print(f"  ⚠  Multiple matches for {camera}; using {matches[0].name}")
    return matches[0]


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Multi-camera manual monkey position labeler for OFT videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--camera', '-c', choices=['D1', 'D2', 'D3', 'D4'],
                        help='Camera to label (default: interactive selection)')
    parser.add_argument('--video', '-v', default=None,
                        help='Video file path (default: auto-detect from VIDEO_BASE)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output CSV path (default: results/.../labels_D#.csv)')
    parser.add_argument('--sample-every', '-n', type=int, default=DEFAULT_SAMPLE_EVERY,
                        help='Label every Nth frame (default: 20 = 1 Hz at 20 fps)')
    parser.add_argument('--scale', '-s', type=float, default=DEFAULT_SCALE,
                        help='Display scale for 4K video (default: 0.4)')
    parser.add_argument('--fps', type=float, default=DEFAULT_FPS,
                        help='Video FPS (default: 20)')
    args = parser.parse_args()

    # Resolve camera
    if args.camera:
        camera = args.camera
    else:
        print("Select camera to label:")
        for c, info in CAMERA_INFO.items():
            print(f"  {c}: {info['quadrant']} quadrant")
        camera = input("Enter camera (D1/D2/D3/D4): ").strip().upper()
        if camera not in CAMERA_INFO:
            sys.exit(f"Invalid camera: {camera}")

    # Resolve video
    video = Path(args.video) if args.video else _find_video(camera, VIDEO_BASE)

    # Resolve output CSV
    out_csv = (Path(args.output) if args.output
               else OUTPUT_BASE / f'labels_{camera}.csv')

    labeler = ManualLabeler(
        camera=camera,
        video_path=video,
        output_csv=out_csv,
        sample_every=args.sample_every,
        display_scale=args.scale,
        fps=args.fps,
    )
    labeler.run()


if __name__ == '__main__':
    main()
