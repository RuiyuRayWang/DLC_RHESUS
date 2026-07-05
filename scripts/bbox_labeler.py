#!/usr/bin/env python3
"""
Bounding-box labeler for monkey behavioral videos.

This script has two modes:

1. annotate
    OpenCV GUI for drawing one monkey bounding box per sampled video frame.

2. validate
    OpenCV GUI for reviewing RT-DETR predictions on extracted JPG/PNG frames and
    correcting the output CSV in place.

It is designed for detector annotation when video filenames no longer encode camera IDs
such as D01_, D02_, etc.

Usage examples:
    conda activate video_tracking

    # Label a specific video; this is now required
    python scripts/bbox_labeler.py --video data/oft_rhp/H1705011/2026-06-15/trial_001.mp4

    # Sample 1 frame/sec from a 20 FPS video
    python scripts/bbox_labeler.py --video path/to/video.mp4 --sample-every 20

    # Denser sampling for detector training
    python scripts/bbox_labeler.py --video path/to/video.mp4 --sample-every 5

    # Export annotated positive frames as JPEGs while saving the CSV
    python scripts/bbox_labeler.py --video path/to/video.mp4 --save-images results/bbox_images/trial_001

    # Validate RT-DETR predictions on extracted frames
    python scripts/bbox_labeler.py --mode validate \
        --frames-dir tmp/rtdetr_frames_2hz/H1705011/2026-06-27/D01_20260627134058_downsampled \
        --predictions-csv results/predictions/H1705011/2026-06-27/D01_20260627134058_downsampled_preds.csv

Controls:
    Left drag       draw or replace monkey bounding box
    Enter / Space   accept current box and advance
    A               toggle auto-advance after drawing box
    M               mark frame as no monkey visible and advance
    N / Right       next frame
    P / Left        previous frame
    U               undo annotation on current frame
    S               save without quitting
    Q / ESC         save and quit

Output CSV columns:
    video_file, video_timestamp, video_frame, time_sec,
    monkey_visible, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
    bbox_width, bbox_height, image_width, image_height

Coordinates are always stored in original video pixel coordinates, not display-scaled coordinates.

Frame counting:
    When ffprobe is available, the labeler uses the decoded frame count
    (ffprobe -count_frames / nb_read_frames), not container metadata. This avoids
    crashes when an MP4 advertises more frames than can actually be decoded.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_BASE = PROJECT_ROOT / "results" / "oft_analysis"

DEFAULT_SAMPLE_EVERY = 20
DEFAULT_FPS = 20.0
DEFAULT_SCALE = 0.4
DEFAULT_CACHE_SIZE = 25
DEFAULT_AUTOSAVE_EVERY = 25
MIN_BOX_SIZE = 5
FRAME_INDEX_RE = re.compile(r"_f(\d+)$")

C_YELLOW = (0, 220, 220)
C_GREEN = (80, 200, 80)
C_RED = (80, 80, 220)
C_BLUE = (220, 160, 60)
C_GRAY = (180, 180, 180)
C_BLACK = (30, 30, 30)

BBox = Tuple[int, int, int, int]


@dataclass
class FrameAnnotation:
    visible: Optional[bool] = None
    bbox: Optional[BBox] = None


def parse_visible(value) -> Optional[bool]:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"true", "1", "yes", "y"}:
            return True
        if value in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def clamp_box(box: BBox, width: int, height: int) -> Optional[BBox]:
    x1, y1, x2, y2 = box
    x1, x2 = sorted((int(round(x1)), int(round(x2))))
    y1, y2 = sorted((int(round(y1)), int(round(y2))))
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if (x2 - x1) < MIN_BOX_SIZE or (y2 - y1) < MIN_BOX_SIZE:
        return None
    return x1, y1, x2, y2


def parse_video_timestamp(video_path: Path) -> str:
    """
    Parse a 14-digit timestamp from the filename when present.

    Accepted examples:
        trial_20260615122549.mp4
        D01_20260615122549.mp4
        20260615122549_barrier_td_D1_trial-1.mp4

    Returns 'unknown' if no 14-digit timestamp is found.
    """
    match = re.search(r"(?<!\d)(\d{14})(?!\d)", video_path.name)
    if not match:
        return "unknown"
    try:
        dt = datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return "unknown"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def parse_frame_index_from_image(frame_path: Path) -> Optional[int]:
    match = FRAME_INDEX_RE.search(frame_path.stem)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def boxes_close(box_a: Optional[BBox], box_b: Optional[BBox], atol: int = 1) -> bool:
    if box_a is None or box_b is None:
        return False
    return all(abs(int(a) - int(b)) <= atol for a, b in zip(box_a, box_b))

class FrameDecodeError(RuntimeError):
    """Raised when a requested video frame cannot be decoded."""


def get_decoded_frame_count(video_path: Path) -> Optional[int]:
    """
    Return the number of actually decodable frames using ffprobe -count_frames.

    This is more reliable than OpenCV CAP_PROP_FRAME_COUNT for some MP4 files,
    where container metadata may advertise more frames than can be decoded.
    Returns None if ffprobe is unavailable or does not report a usable count.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        return None

    if result.returncode != 0:
        return None

    value = result.stdout.strip().splitlines()
    if not value:
        return None

    # ffprobe can print N/A for some streams.
    last = value[-1].strip()
    if not last or last.upper() == "N/A":
        return None

    try:
        n = int(last)
    except ValueError:
        return None
    return n if n > 0 else None


class LazyVideo:
    """Video reader with cache plus robust fallback for MP4 seek failures.

    OpenCV frame-count metadata and random access can be unreliable for some
    H.264/H.265 MP4 files, especially near the tail of the file. This reader
    tries fast random access first, then falls back to sequential decoding.
    """

    def __init__(self, video_path: Path, cache_size: int = DEFAULT_CACHE_SIZE):
        self.video_path = Path(video_path)
        self.cache_size = cache_size
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            sys.exit(f"ERROR: Cannot open video: {self.video_path}")

        self.metadata_total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.decoded_total_frames = get_decoded_frame_count(self.video_path)
        self.total_frames = self.decoded_total_frames or self.metadata_total_frames
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.native_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()

        # Sequential reader state. This is slower than seeking but much more
        # reliable on MP4 files whose keyframe/index metadata confuses OpenCV.
        self.seq_cap = cv2.VideoCapture(str(self.video_path))
        if not self.seq_cap.isOpened():
            sys.exit(f"ERROR: Cannot open video for sequential read: {self.video_path}")
        self.seq_next_frame = 0

    def _put_cache(self, frame_num: int, frame: np.ndarray) -> None:
        self.cache[int(frame_num)] = frame
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def _read_random(self, frame_num: int) -> Optional[np.ndarray]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))
        ok, frame = self.cap.read()
        if ok and frame is not None:
            return frame
        return None

    def _reset_sequential(self) -> None:
        self.seq_cap.release()
        self.seq_cap = cv2.VideoCapture(str(self.video_path))
        if not self.seq_cap.isOpened():
            raise RuntimeError(f"Could not reopen video for sequential read: {self.video_path}")
        self.seq_next_frame = 0

    def _read_sequential(self, frame_num: int) -> Optional[np.ndarray]:
        frame_num = int(frame_num)
        if frame_num < self.seq_next_frame:
            self._reset_sequential()

        frame = None
        while self.seq_next_frame <= frame_num:
            ok, frame = self.seq_cap.read()
            if not ok or frame is None:
                return None
            if self.seq_next_frame == frame_num:
                break
            self.seq_next_frame += 1

        self.seq_next_frame = frame_num + 1
        return frame

    def read(self, frame_num: int) -> np.ndarray:
        frame_num = int(frame_num)
        if frame_num < 0:
            raise FrameDecodeError(f"Invalid negative frame index: {frame_num}")
        if self.total_frames > 0 and frame_num >= self.total_frames:
            raise FrameDecodeError(
                f"Requested frame {frame_num}, but only {self.total_frames} frames are decodable."
            )
        if frame_num in self.cache:
            frame = self.cache.pop(frame_num)
            self.cache[frame_num] = frame
            return frame.copy()

        # Fast path: OpenCV random frame seek.
        frame = self._read_random(frame_num)

        # Robust path: sequential decode. This fixes failures like
        # CAP_PROP_FRAME_COUNT reporting 1106 frames but random read failing
        # around frame 1000.
        if frame is None:
            frame = self._read_sequential(frame_num)

        if frame is None:
            raise FrameDecodeError(
                f"Could not decode frame {frame_num} from {self.video_path}. "
                "The video may be truncated or OpenCV cannot decode this MP4 reliably. "
                "Try remuxing with: ffmpeg -i input.mp4 -c copy -movflags +faststart fixed.mp4"
            )

        self._put_cache(frame_num, frame)
        return frame.copy()

    def release(self):
        self.cap.release()
        self.seq_cap.release()


class BBoxLabeler:
    def __init__(
        self,
        video_path: Path,
        output_csv: Path,
        sample_every: int = DEFAULT_SAMPLE_EVERY,
        display_scale: float = DEFAULT_SCALE,
        fps: float = DEFAULT_FPS,
        auto_advance: bool = False,
        save_images: Optional[Path] = None,
        cache_size: int = DEFAULT_CACHE_SIZE,
        autosave_every: int = DEFAULT_AUTOSAVE_EVERY,
    ):
        self.video_path = Path(video_path)
        self.output_csv = Path(output_csv)
        self.sample_every = int(sample_every)
        self.display_scale = float(display_scale)
        self.fps = float(fps)
        self.auto_advance = bool(auto_advance)
        self.save_images = Path(save_images) if save_images else None
        self.autosave_every = int(autosave_every)
        self.changes_since_save = 0

        self.video = LazyVideo(self.video_path, cache_size=cache_size)
        if self.fps <= 0:
            self.fps = self.video.native_fps if self.video.native_fps > 0 else DEFAULT_FPS

        # Use the decoded frame count, when available from ffprobe, rather than
        # OpenCV/container metadata. Valid zero-based frame indices are
        # 0 ... total_frames - 1.
        self.frame_nums = list(range(0, self.video.total_frames, self.sample_every))
        if not self.frame_nums:
            sys.exit("ERROR: No frames selected. Check video and --sample-every.")

        self.annotations: Dict[int, FrameAnnotation] = {}
        self.current_idx = 0

        self.dragging = False
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_current: Optional[Tuple[int, int]] = None
        self.last_save_count = 0

        self.win = "MonkeyBBox"
        self.win_title = f"Monkey bbox - {self.video_path.name}"

        if self.output_csv.exists():
            self._load_existing()

        print(f"\nVideo: {self.video_path}")
        print(f"Resolution: {self.video.width} x {self.video.height}")
        print(f"Container frame count: {self.video.metadata_total_frames}")
        if self.video.decoded_total_frames is not None:
            print(f"Decoded frame count: {self.video.decoded_total_frames}")
            if self.video.decoded_total_frames != self.video.metadata_total_frames:
                dropped = self.video.metadata_total_frames - self.video.decoded_total_frames
                print(f"Using decoded frame count; ignoring {dropped} non-decodable trailing metadata frames")
        else:
            print("Decoded frame count: unavailable; using OpenCV/container frame count")
        print(f"Sampling every {self.sample_every} frames -> {len(self.frame_nums)} frames to review")
        print(f"Output CSV: {self.output_csv}")
        print(f"Auto-advance after drawing: {self.auto_advance}")
        if self.save_images:
            print(f"Positive-frame JPEG export: {self.save_images}")
        if self.autosave_every > 0:
            print(f"Autosave: every {self.autosave_every} annotation changes")

    def _load_existing(self):
        df = pd.read_csv(self.output_csv)
        loaded = 0
        for _, row in df.iterrows():
            fn = int(row["video_frame"])
            visible = parse_visible(row.get("monkey_visible"))
            box = None
            if visible:
                vals = [row.get(c) for c in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")]
                if all(pd.notna(v) for v in vals):
                    raw_box = tuple(int(round(float(v))) for v in vals)  # type: ignore[arg-type]
                    box = clamp_box(raw_box, self.video.width, self.video.height)
            self.annotations[fn] = FrameAnnotation(visible=visible, bbox=box)
            loaded += 1
        print(f"Resumed: loaded {loaded} existing rows from {self.output_csv}")

    def _current_frame_num(self) -> int:
        return self.frame_nums[self.current_idx]

    def _current_ann(self) -> FrameAnnotation:
        fn = self._current_frame_num()
        if fn not in self.annotations:
            self.annotations[fn] = FrameAnnotation()
        return self.annotations[fn]

    def _display_to_original(self, x: int, y: int) -> Tuple[int, int]:
        ox = int(round(x / self.display_scale))
        oy = int(round(y / self.display_scale))
        ox = max(0, min(self.video.width - 1, ox))
        oy = max(0, min(self.video.height - 1, oy))
        return ox, oy

    def _box_to_display(self, box: BBox) -> BBox:
        x1, y1, x2, y2 = box
        return (
            int(round(x1 * self.display_scale)),
            int(round(y1 * self.display_scale)),
            int(round(x2 * self.display_scale)),
            int(round(y2 * self.display_scale)),
        )

    def _mouse_cb(self, event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start = self._display_to_original(mx, my)
            self.drag_current = self.drag_start

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.drag_current = self._display_to_original(mx, my)

        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            if self.drag_start is None:
                return
            end = self._display_to_original(mx, my)
            box = clamp_box((*self.drag_start, *end), self.video.width, self.video.height)
            self.drag_start = None
            self.drag_current = None
            if box is None:
                print("Ignored tiny box. Draw a larger rectangle.")
                return
            ann = self._current_ann()
            ann.visible = True
            ann.bbox = box
            self._note_annotation_change()
            if self.auto_advance:
                self._next_frame()

    def _note_annotation_change(self) -> None:
        self.changes_since_save += 1
        if self.autosave_every > 0 and self.changes_since_save >= self.autosave_every:
            self.save()
            self.changes_since_save = 0

    def _next_frame(self):
        self.current_idx = min(len(self.frame_nums) - 1, self.current_idx + 1)

    def _prev_frame(self):
        self.current_idx = max(0, self.current_idx - 1)

    def _annotated_counts(self) -> Tuple[int, int, int, int]:
        n_box = sum(1 for a in self.annotations.values() if a.visible is True and a.bbox is not None)
        n_visible_pending = sum(1 for a in self.annotations.values() if a.visible is True and a.bbox is None)
        n_no = sum(1 for a in self.annotations.values() if a.visible is False)
        n_any = sum(1 for a in self.annotations.values() if a.visible is not None)
        return n_box, n_visible_pending, n_no, n_any

    def _render(self) -> np.ndarray:
        fn = self._current_frame_num()
        frame = self.video.read(fn)
        disp = cv2.resize(
            frame,
            (int(self.video.width * self.display_scale), int(self.video.height * self.display_scale)),
            interpolation=cv2.INTER_AREA,
        )
        dh, dw = disp.shape[:2]

        ann = self.annotations.get(fn, FrameAnnotation())

        if ann.visible is False:
            cv2.drawMarker(disp, (dw // 2, dh // 2), C_RED, cv2.MARKER_TILTED_CROSS, 80, 4)
            cv2.putText(disp, "NO MONKEY", (dw // 2 - 110, dh // 2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, C_RED, 3)
        elif ann.bbox is not None:
            x1, y1, x2, y2 = self._box_to_display(ann.bbox)
            cv2.rectangle(disp, (x1, y1), (x2, y2), C_GREEN, 3)
            cv2.putText(disp, "monkey", (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_GREEN, 2)
        elif ann.visible is True:
            cv2.putText(disp, "VISIBLE - DRAW BOX", (30, dh - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_YELLOW, 2)

        if self.dragging and self.drag_start is not None and self.drag_current is not None:
            preview = clamp_box((*self.drag_start, *self.drag_current), self.video.width, self.video.height)
            if preview is not None:
                x1, y1, x2, y2 = self._box_to_display(preview)
                cv2.rectangle(disp, (x1, y1), (x2, y2), C_BLUE, 2)

        n_box, n_pending, n_no, n_any = self._annotated_counts()
        cv2.rectangle(disp, (0, 0), (dw, 55), C_BLACK, -1)
        header = (
            f"sample {self.current_idx + 1}/{len(self.frame_nums)} | "
            f"video frame #{fn} ({fn / self.fps:.1f}s) | "
            f"boxes {n_box} | pending {n_pending} | no-monkey {n_no} | reviewed {n_any}"
        )
        cv2.putText(disp, header, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.68, C_YELLOW, 2, cv2.LINE_AA)

        cv2.rectangle(disp, (0, dh - 42), (dw, dh), C_BLACK, -1)
        footer = "Drag=box  Enter/Space=accept+next  A=auto  M=no-monkey  N/P=next/prev  U=undo  S=save  Q=save+quit"
        if self.auto_advance:
            footer = "AUTO ON  " + footer
        cv2.putText(disp, footer, (10, dh - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_GRAY, 1, cv2.LINE_AA)

        y0 = dh - 45
        cv2.rectangle(disp, (0, y0), (dw, y0 + 4), (60, 60, 60), -1)
        for i, frame_num in enumerate(self.frame_nums):
            a = self.annotations.get(frame_num)
            if a is None or a.visible is None:
                continue
            x = int(i / max(1, len(self.frame_nums) - 1) * (dw - 1))
            if a.visible is False:
                color = C_RED
            elif a.bbox is not None:
                color = C_GREEN
            else:
                color = C_YELLOW
            cv2.line(disp, (x, y0), (x, y0 + 4), color, 2)

        return disp

    def _save_images_for_positive_boxes(self):
        if self.save_images is None:
            return
        self.save_images.mkdir(parents=True, exist_ok=True)
        written = 0
        for fn, ann in self.annotations.items():
            if ann.visible is True and ann.bbox is not None:
                out_name = f"{self.video_path.stem}_frame{fn:08d}.jpg"
                out_path = self.save_images / out_name
                if not out_path.exists():
                    frame = self.video.read(fn)
                    cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    written += 1
        if written:
            print(f"Saved {written} positive-frame JPEGs to {self.save_images}")

    def save(self) -> pd.DataFrame:
        records = []
        video_timestamp = parse_video_timestamp(self.video_path)
        for fn in sorted(self.annotations):
            ann = self.annotations[fn]
            if ann.visible is None:
                continue
            if ann.visible is True and ann.bbox is not None:
                x1, y1, x2, y2 = ann.bbox
                bw = x2 - x1
                bh = y2 - y1
            else:
                x1 = y1 = x2 = y2 = bw = bh = np.nan

            records.append(
                {
                    "video_file": self.video_path.name,
                    "video_timestamp": video_timestamp,
                    "video_frame": int(fn),
                    "time_sec": float(fn / self.fps),
                    "monkey_visible": bool(ann.visible),
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "bbox_width": bw,
                    "bbox_height": bh,
                    "image_width": self.video.width,
                    "image_height": self.video.height,
                }
            )

        df = pd.DataFrame(records)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_csv, index=False)
        self._save_images_for_positive_boxes()
        self.last_save_count = len(df)
        self.changes_since_save = 0
        n_box, n_pending, n_no, _ = self._annotated_counts()
        print(f"Saved {len(df)} reviewed frames -> {self.output_csv}")
        print(f"  boxes: {n_box}   visible-without-box: {n_pending}   no-monkey: {n_no}")
        return df


class BBoxValidateLabeler:
    def __init__(
        self,
        frames_dir: Path,
        predictions_csv: Path,
        output_csv: Optional[Path] = None,
        display_scale: float = DEFAULT_SCALE,
        autosave_every: int = DEFAULT_AUTOSAVE_EVERY,
    ):
        self.frames_dir = Path(frames_dir)
        self.predictions_csv = Path(predictions_csv)
        self.output_csv = Path(output_csv) if output_csv else self.predictions_csv
        self.display_scale = float(display_scale)
        self.autosave_every = int(autosave_every)
        self.changes_since_save = 0

        if not self.frames_dir.exists():
            raise FileNotFoundError(f"Frames directory does not exist: {self.frames_dir}")
        if not self.predictions_csv.exists():
            raise FileNotFoundError(f"Predictions CSV does not exist: {self.predictions_csv}")

        self.frame_paths = sorted(
            [p for p in self.frames_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        if not self.frame_paths:
            raise RuntimeError(f"No image frames found in: {self.frames_dir}")

        self.pred_df = pd.read_csv(self.predictions_csv)
        if "video_frame" not in self.pred_df.columns:
            raise ValueError(f"{self.predictions_csv}: missing video_frame column")

        self.frame_map = self._build_frame_map()
        self.current_idx = 0
        self.dragging = False
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_current: Optional[Tuple[int, int]] = None

        first_frame = cv2.imread(str(self.frame_paths[0]))
        if first_frame is None:
            raise RuntimeError(f"Could not read frame: {self.frame_paths[0]}")
        self.height, self.width = first_frame.shape[:2]

        self.win = "MonkeyBBoxValidate"
        self.win_title = f"Monkey bbox validate - {self.frames_dir.name}"

        print(f"\nFrames dir: {self.frames_dir}")
        print(f"Predictions CSV: {self.predictions_csv}")
        print(f"Output CSV: {self.output_csv}")
        print(f"Frames loaded: {len(self.frame_paths)}")

    def _build_frame_map(self) -> Dict[int, Path]:
        frame_map: Dict[int, Path] = {}
        for frame_path in self.frame_paths:
            frame_idx = parse_frame_index_from_image(frame_path)
            if frame_idx is None:
                continue
            frame_map[frame_idx] = frame_path
        if not frame_map:
            raise RuntimeError(f"No frame indices could be parsed from {self.frames_dir}")
        return frame_map

    def _current_frame_num(self) -> int:
        return int(parse_frame_index_from_image(self.frame_paths[self.current_idx]) or 0)

    def _pred_row(self, frame_num: int) -> pd.Series:
        matches = self.pred_df[self.pred_df["video_frame"] == frame_num]
        if matches.empty:
            return pd.Series(dtype=object)
        return matches.iloc[0]

    def _current_pred_box(self) -> Optional[BBox]:
        row = self._pred_row(self._current_frame_num())
        if row.empty:
            return None
        vals = [row.get(c) for c in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")]
        if not all(pd.notna(v) for v in vals):
            return None
        return tuple(int(round(float(v))) for v in vals)  # type: ignore[return-value]

    def _current_pred_visible(self) -> Optional[bool]:
        row = self._pred_row(self._current_frame_num())
        if row.empty:
            return None
        return parse_visible(row.get("monkey_visible"))

    def _display_to_original(self, x: int, y: int) -> Tuple[int, int]:
        ox = int(round(x / self.display_scale))
        oy = int(round(y / self.display_scale))
        ox = max(0, min(self.width - 1, ox))
        oy = max(0, min(self.height - 1, oy))
        return ox, oy

    def _box_to_display(self, box: BBox) -> BBox:
        x1, y1, x2, y2 = box
        return (
            int(round(x1 * self.display_scale)),
            int(round(y1 * self.display_scale)),
            int(round(x2 * self.display_scale)),
            int(round(y2 * self.display_scale)),
        )

    def _next_frame(self):
        self.current_idx = min(len(self.frame_paths) - 1, self.current_idx + 1)

    def _prev_frame(self):
        self.current_idx = max(0, self.current_idx - 1)

    def _note_change(self) -> None:
        self.changes_since_save += 1
        if self.autosave_every > 0 and self.changes_since_save >= self.autosave_every:
            self.save()
            self.changes_since_save = 0

    def _mouse_cb(self, event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start = self._display_to_original(mx, my)
            self.drag_current = self.drag_start
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.drag_current = self._display_to_original(mx, my)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            if self.drag_start is None:
                return
            end = self._display_to_original(mx, my)
            box = clamp_box((*self.drag_start, *end), self.width, self.height)
            self.drag_start = None
            self.drag_current = None
            if box is None:
                print("Ignored tiny box. Draw a larger rectangle.")
                return
            frame_num = self._current_frame_num()
            row_idx = self.pred_df.index[self.pred_df["video_frame"] == frame_num]
            if len(row_idx) == 0:
                new_row = {
                    "video_frame": frame_num,
                    "monkey_visible": True,
                    "bbox_x1": box[0],
                    "bbox_y1": box[1],
                    "bbox_x2": box[2],
                    "bbox_y2": box[3],
                }
                self.pred_df = pd.concat([self.pred_df, pd.DataFrame([new_row])], ignore_index=True)
            else:
                idx = row_idx[0]
                self.pred_df.at[idx, "monkey_visible"] = True
                self.pred_df.at[idx, "bbox_x1"] = box[0]
                self.pred_df.at[idx, "bbox_y1"] = box[1]
                self.pred_df.at[idx, "bbox_x2"] = box[2]
                self.pred_df.at[idx, "bbox_y2"] = box[3]
            self._note_change()

    def _render(self) -> np.ndarray:
        frame_num = self._current_frame_num()
        frame_path = self.frame_paths[self.current_idx]
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise RuntimeError(f"Could not read frame: {frame_path}")
        disp = cv2.resize(
            frame,
            (int(self.width * self.display_scale), int(self.height * self.display_scale)),
            interpolation=cv2.INTER_AREA,
        )
        dh, dw = disp.shape[:2]

        pred_visible = self._current_pred_visible()
        pred_box = self._current_pred_box()

        if pred_visible is False:
            cv2.drawMarker(disp, (dw // 2, dh // 2), C_RED, cv2.MARKER_TILTED_CROSS, 80, 4)
            cv2.putText(disp, "PRED: NO MONKEY", (20, dh // 2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_RED, 3)
        elif pred_box is not None:
            x1, y1, x2, y2 = self._box_to_display(pred_box)
            cv2.rectangle(disp, (x1, y1), (x2, y2), C_GREEN, 3)
            cv2.putText(disp, "PRED", (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_GREEN, 2)
        else:
            cv2.putText(disp, "NO PREDICTION", (20, dh - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_YELLOW, 2)

        if self.dragging and self.drag_start is not None and self.drag_current is not None:
            preview = clamp_box((*self.drag_start, *self.drag_current), self.width, self.height)
            if preview is not None:
                x1, y1, x2, y2 = self._box_to_display(preview)
                cv2.rectangle(disp, (x1, y1), (x2, y2), C_BLUE, 2)

        cv2.rectangle(disp, (0, 0), (dw, 55), C_BLACK, -1)
        header = (
            f"frame {self.current_idx + 1}/{len(self.frame_paths)} | "
            f"video frame #{frame_num} | "
            f"pred_visible={pred_visible}"
        )
        cv2.putText(disp, header, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.68, C_YELLOW, 2, cv2.LINE_AA)

        cv2.rectangle(disp, (0, dh - 42), (dw, dh), C_BLACK, -1)
        footer = "Drag=correct box  Enter/Space=next  N/P=next/prev  U=clear pred row  S=save  Q=save+quit"
        cv2.putText(disp, footer, (10, dh - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_GRAY, 1, cv2.LINE_AA)
        return disp

    def save(self) -> pd.DataFrame:
        df = self.pred_df.copy()
        if "video_frame" in df.columns:
            df = df.sort_values(by=["video_frame"]).reset_index(drop=True)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_csv, index=False)
        self.changes_since_save = 0
        print(f"Saved validated predictions -> {self.output_csv}")
        return df

    def run(self):
        print(f"\nOpening window: {self.win_title}")
        try:
            try:
                cv2.startWindowThread()
            except Exception:
                pass
            cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
            try:
                cv2.setWindowTitle(self.win, self.win_title)
            except Exception:
                pass
            cv2.imshow(self.win, self._render())
            cv2.waitKey(1)
            cv2.setMouseCallback(self.win, self._mouse_cb)
        except cv2.error as exc:
            print("\nERROR: Failed to create OpenCV GUI window.")
            print(f"Details: {exc}")
            print("Check DISPLAY, X11 forwarding, or your OpenCV GUI backend.")
            self.video.release()
            raise SystemExit(1)

        print("\nWindow opened. Draw a box around the monkey. Press Q or ESC to save and quit.\n")
        while True:
            cv2.imshow(self.win, self._render())
            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), 27):
                break
            if key in (ord("s"),):
                self.save()
            elif key in (ord("n"), 83):
                self._next_frame()
            elif key in (ord("p"), 81):
                self._prev_frame()
            elif key in (ord("u"),):
                fn = self._current_frame_num()
                self.annotations.pop(fn, None)
                self._note_annotation_change()
            elif key in (ord("m"),):
                ann = self._current_ann()
                ann.visible = False
                ann.bbox = None
                self._note_annotation_change()
                self._next_frame()
            elif key in (ord("a"),):
                self.auto_advance = not self.auto_advance
                print(f"Auto-advance after drawing: {self.auto_advance}")
            elif key in (13, 10, 32):
                self._next_frame()

        cv2.destroyAllWindows()
        df = self.save()
        self.video.release()
        return df


def main():
    parser = argparse.ArgumentParser(
        description="OpenCV GUI for annotating or validating one monkey bounding box per sampled frame",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["annotate", "validate"], default="annotate", help="annotate: create bbox labels, validate: review predictions on extracted frames")
    parser.add_argument("--video", "-v", required=False, help="Video file path for annotate mode")
    parser.add_argument("--output", "-o", default=None, help="Output CSV path for annotate mode")
    parser.add_argument("--frames-dir", default=None, help="Directory containing extracted JPG/PNG frames for validate mode")
    parser.add_argument("--predictions-csv", default=None, help="RT-DETR predictions CSV for validate mode")
    parser.add_argument("--validated-output", default=None, help="Output CSV path for validate mode (defaults to predictions CSV) ")
    parser.add_argument("--sample-every", "-n", type=int, default=DEFAULT_SAMPLE_EVERY, help="Label every Nth frame in annotate mode")
    parser.add_argument("--scale", "-s", type=float, default=DEFAULT_SCALE, help="Display scale")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="FPS used for time_sec in annotate mode; <=0 uses video metadata")
    parser.add_argument("--auto-advance", action="store_true", help="Advance immediately after drawing a box in annotate mode")
    parser.add_argument("--save-images", default=None, help="Optional directory to save positive annotated frames as JPEG")
    parser.add_argument("--cache-size", type=int, default=DEFAULT_CACHE_SIZE, help="Number of decoded frames to cache in annotate mode")
    parser.add_argument("--autosave-every", type=int, default=DEFAULT_AUTOSAVE_EVERY, help="Autosave after this many annotation changes; 0 disables")
    args = parser.parse_args()

    if args.scale <= 0:
        sys.exit("ERROR: --scale must be positive")

    if args.mode == "annotate":
        if not args.video:
            sys.exit("ERROR: --video is required in annotate mode")
        if args.sample_every <= 0:
            sys.exit("ERROR: --sample-every must be positive")

        video = Path(args.video).expanduser().resolve()
        if not video.exists():
            sys.exit(f"ERROR: Video file does not exist: {video}")

        output_csv = (Path(args.output).expanduser().resolve() if args.output
                      else OUTPUT_BASE / f"bbox_{video.stem}.csv")

        labeler = BBoxLabeler(
            video_path=video,
            output_csv=output_csv,
            sample_every=args.sample_every,
            display_scale=args.scale,
            fps=args.fps,
            auto_advance=args.auto_advance,
            save_images=Path(args.save_images) if args.save_images else None,
            cache_size=args.cache_size,
            autosave_every=args.autosave_every,
        )
        labeler.run()
        return

    if not args.frames_dir:
        sys.exit("ERROR: --frames-dir is required in validate mode")
    if not args.predictions_csv:
        sys.exit("ERROR: --predictions-csv is required in validate mode")

    frames_dir = Path(args.frames_dir).expanduser().resolve()
    predictions_csv = Path(args.predictions_csv).expanduser().resolve()
    validated_output = Path(args.validated_output).expanduser().resolve() if args.validated_output else predictions_csv

    validator = BBoxValidateLabeler(
        frames_dir=frames_dir,
        predictions_csv=predictions_csv,
        output_csv=validated_output,
        display_scale=args.scale,
        autosave_every=args.autosave_every,
    )
    validator.run()


if __name__ == "__main__":
    main()
