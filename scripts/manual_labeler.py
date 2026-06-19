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
    python scripts/manual_labeler.py --camera D2 --mode visibility
    python scripts/manual_labeler.py --camera D2 --mode head
    python scripts/manual_labeler.py --help

Controls:
    Left-click  : label monkey position and advance to next frame
    Right-click : label monkey position WITHOUT advancing
    M           : mark frame as "no monkey visible" and advance
    V           : mark frame as "monkey visible" and advance
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
OUTPUT_BASE = PROJECT_ROOT / 'results' / 'oft_analysis'

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
                 display_scale=DEFAULT_SCALE, fps=DEFAULT_FPS,
                 mode='full'):
        self.camera = camera
        self.video_path = Path(video_path)
        self.output_csv = Path(output_csv)
        self.sample_every = sample_every
        self.display_scale = display_scale
        self.fps = fps
        self.mode = mode

        print(f"\nCamera {camera} ({CAMERA_INFO[camera]['quadrant']} quadrant)")
        print(f"Mode: {mode}")
        print(f"Loading frames from: {self.video_path.name}")
        print(f"  sampling every {sample_every} frames  ({fps / sample_every:.2f} Hz)")
        self.frames, self.frame_nums = self._load_frames()
        print(f"  → {len(self.frames)} frames to label\n")

        # labels: { video_frame_number -> (x, y) in ORIGINAL pixel coords }
        self.labels: Dict[int, Tuple[int, int]] = {}
        self.visibility: Dict[int, bool] = {}
        self.existing_rows: Dict[int, Dict[str, object]] = {}

        # Load existing CSV so a session can be resumed
        if self.output_csv.exists():
            self._load_existing()

        if self.mode == 'head':
            self._restrict_to_visible_frames()

        self.current_idx = 0
        self._advance_requested = False

        # Simple window name (Qt6 sensitive to special characters)
        self.win = f"Label_{camera}"
        self.win_title = self._window_title()

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
            visible = self._parse_visible(row.get('monkey_visible'))
            cx = row.get('centroid_x')
            cy = row.get('centroid_y')

            if visible is not None:
                self.visibility[fn] = visible

            if visible and pd.notna(cx) and pd.notna(cy):
                cx_int = int(cx)
                cy_int = int(cy)
                if cx_int != NO_MONKEY_SENTINEL and cy_int != NO_MONKEY_SENTINEL:
                    self.labels[fn] = (cx_int, cy_int)

            self.existing_rows[fn] = row.to_dict()
            loaded += 1
        print(f"Resumed: loaded {loaded} existing labels from {self.output_csv}")

    def _parse_visible(self, value):
        if pd.isna(value):
            return None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str):
            value = value.strip().lower()
            if value in {'true', '1', 'yes', 'y'}:
                return True
            if value in {'false', '0', 'no', 'n'}:
                return False
        return bool(value)

    def _restrict_to_visible_frames(self):
        visible_frames = {fn for fn, visible in self.visibility.items() if visible}
        if not visible_frames:
            sys.exit(
                'ERROR: Head mode requires an existing CSV with visible/no-monkey labels.\n'
                '       Run manual_labeler.py --mode visibility first.'
            )

        filtered = [(frame, fn) for frame, fn in zip(self.frames, self.frame_nums) if fn in visible_frames]
        if not filtered:
            sys.exit('ERROR: No visible sampled frames found for head labeling.')

        self.frames = [frame for frame, _ in filtered]
        self.frame_nums = [fn for _, fn in filtered]
        print(f"  → restricted to {len(self.frames)} visible frames for head labeling\n")

    def _window_title(self):
        if self.mode == 'visibility':
            controls = 'V=visible  M=no-monkey  N=next  P=prev  U=undo  Q=quit'
        elif self.mode == 'head':
            controls = 'L-click=label  M=no-monkey  N=next  P=prev  U=undo  Q=quit'
        else:
            controls = 'L-click=label  M=no-monkey  V=visible  N=next  P=prev  U=undo  Q=quit'
        return f"Camera {self.camera} ({CAMERA_INFO[self.camera]['quadrant']}) - {controls}"

    # ── Display ────────────────────────────────────────────────────────────────

    def _render(self):
        """Return a display-ready copy of the current frame with annotations."""
        orig = self.frames[self.current_idx]
        h, w = orig.shape[:2]
        dw = int(w * self.display_scale)
        dh = int(h * self.display_scale)
        disp = cv2.resize(orig, (dw, dh), interpolation=cv2.INTER_AREA)

        frame_num = self.frame_nums[self.current_idx]
        n_visible = sum(1 for v in self.visibility.values() if v)
        n_no_monkey = sum(1 for v in self.visibility.values() if v is False)
        n_labeled = sum(1 for fn, v in self.visibility.items() if v and fn in self.labels)
        n_pending = max(n_visible - n_labeled, 0)

        visible = self.visibility.get(frame_num)
        coords = self.labels.get(frame_num)

        # ── Draw label marker (or "no monkey" indicator) ──
        if visible is False:
            cx, cy = dw // 2, dh // 2
            cv2.drawMarker(disp, (cx, cy), C_RED,
                           cv2.MARKER_TILTED_CROSS, thickness=4, markerSize=80)
            cv2.putText(disp, 'NO MONKEY', (cx - 100, cy + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, C_RED, 3, cv2.LINE_AA)
        elif coords is not None:
            ox, oy = coords
            dx = int(ox * self.display_scale)
            dy = int(oy * self.display_scale)
            cv2.drawMarker(disp, (dx, dy), C_GREEN,
                           cv2.MARKER_CROSS, thickness=3, markerSize=50)
            cv2.circle(disp, (dx, dy), 30, C_GREEN, 2)
        elif visible is True:
            cv2.putText(disp, 'VISIBLE (PENDING HEAD LABEL)', (30, dh - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_YELLOW, 2, cv2.LINE_AA)

        # ── Header bar ──
        cv2.rectangle(disp, (0, 0), (dw, 50), (30, 30, 30), -1)
        status = (f"Cam {self.camera} | "
                  f"Frame {self.current_idx + 1}/{len(self.frames)}"
                  f"  #{frame_num} ({frame_num / self.fps:.1f}s)"
                  f"  |  labeled: {n_labeled}  visible: {n_visible}"
                  f"  pending: {n_pending}  no-monkey: {n_no_monkey}")
        cv2.putText(disp, status, (10, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, C_YELLOW, 2, cv2.LINE_AA)

        # ── Footer bar ──
        cv2.rectangle(disp, (0, dh - 35), (dw, dh), (30, 30, 30), -1)
        if self.mode == 'visibility':
            hint = 'V=visible & next   M=no monkey & next   N=next   P=prev   U=undo   Q=save & quit'
        else:
            hint = ('L-click=label & next   R-click=label only   '
                    'M=no monkey   N=next   P=prev   U=undo   Q=save & quit')
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
                col = C_GREEN
                cv2.line(disp, (bx, dh - 38), (bx, dh - 35), col, 2)
            for fn, visible in self.visibility.items():
                try:
                    fi = self.frame_nums.index(fn)
                except ValueError:
                    continue
                bx = int(fi / total * dw)
                if visible is False:
                    col = C_RED
                elif visible is True and fn not in self.labels:
                    col = C_YELLOW
                else:
                    continue
                cv2.line(disp, (bx, dh - 38), (bx, dh - 35), col, 2)

        return disp

    # ── Mouse callback ─────────────────────────────────────────────────────────

    def _mouse_cb(self, event, mx, my, flags, param):
        if self.mode == 'visibility':
            return
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            # Convert display coords → original pixel coords
            ox = int(mx / self.display_scale)
            oy = int(my / self.display_scale)
            frame_num = self.frame_nums[self.current_idx]
            self.labels[frame_num] = (ox, oy)
            self.visibility[frame_num] = True
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
                self.labels.pop(fn, None)
                self.visibility.pop(fn, None)
            elif key == ord('v'):          # V → mark visible
                fn = self.frame_nums[self.current_idx]
                self.visibility[fn] = True
                self._advance_requested = True
            elif key == ord('m'):          # M → mark "no monkey visible"
                fn = self.frame_nums[self.current_idx]
                self.visibility[fn] = False
                self.labels.pop(fn, None)
                self._advance_requested = True

        cv2.destroyAllWindows()
        self._save()

    # ── Export ─────────────────────────────────────────────────────────────────

    def _save(self):
        # Parse video start timestamp from filename: D0X_YYYYMMDDHHMMSS.mp4
        video_timestamp = self._parse_video_timestamp(self.video_path)

        all_frames = sorted(set(self.existing_rows) | set(self.visibility) | set(self.labels))
        records = []
        for fn in all_frames:
            visible = self.visibility.get(fn)
            coords = self.labels.get(fn)

            if visible is None and fn in self.existing_rows:
                visible = self._parse_visible(self.existing_rows[fn].get('monkey_visible'))

            if visible is None:
                continue

            if visible:
                if coords is not None:
                    centroid_x, centroid_y = coords
                else:
                    centroid_x, centroid_y = np.nan, np.nan
            else:
                centroid_x, centroid_y = NO_MONKEY_SENTINEL, NO_MONKEY_SENTINEL

            records.append(
                {
                    'camera': self.camera,
                    'video_file': self.video_path.name,
                    'video_timestamp': video_timestamp,
                    'video_frame': fn,
                    'time_sec': fn / self.fps,
                    'centroid_x': centroid_x,
                    'centroid_y': centroid_y,
                    'monkey_visible': visible,
                }
            )

        df = pd.DataFrame(records)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_csv, index=False)
        n_vis  = df['monkey_visible'].sum()
        n_miss = (~df['monkey_visible']).sum()
        n_pending = ((df['monkey_visible'] == True) & (df['centroid_x'].isna() | df['centroid_y'].isna())).sum()
        print(f"\n✅  Saved {len(df)} labels → {self.output_csv}")
        print(f"   monkey visible: {n_vis}   no-monkey: {n_miss}   pending head labels: {n_pending}")
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
                        help='Output CSV path (default: results/.../labels_<video_stem>.csv)')
    parser.add_argument('--sample-every', '-n', type=int, default=DEFAULT_SAMPLE_EVERY,
                        help='Label every Nth frame (default: 20 = 1 Hz at 20 fps)')
    parser.add_argument('--scale', '-s', type=float, default=DEFAULT_SCALE,
                        help='Display scale for 4K video (default: 0.4)')
    parser.add_argument('--fps', type=float, default=DEFAULT_FPS,
                        help='Video FPS (default: 20)')
    parser.add_argument('--mode', choices=['full', 'visibility', 'head'], default='full',
                        help='full=one-pass labels, visibility=visible/no-monkey only, head=head labels on visible frames only')
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
    if args.mode == 'visibility':
        out_csv = (Path(args.output) if args.output
                else OUTPUT_BASE / f'visibility_{video.stem}.csv')
    elif args.mode == 'head':
        out_csv = (Path(args.output) if args.output
                else OUTPUT_BASE / f'{video.stem}.csv')
    else:
        out_csv = (Path(args.output) if args.output
                else OUTPUT_BASE / f'{video.stem}.csv')

    labeler = ManualLabeler(
        camera=camera,
        video_path=video,
        output_csv=out_csv,
        sample_every=args.sample_every,
        display_scale=args.scale,
        fps=args.fps,
        mode=args.mode,
    )
    labeler.run()


if __name__ == '__main__':
    main()
