#!/usr/bin/env python3
"""
Test script for tuned monkey segmentation that suppresses cage detection.
Usage: conda activate deeplabcut && python scripts/test_segmentation_tuning.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Configuration
VIDEO_DIR = Path('/home/luolab/PROJECTS/PAL_ENHANCER/data/oft_rhp/H1705011/2026-05-25')
OUTPUT_DIR = Path('/home/luolab/PROJECTS/PAL_ENHANCER/results/segmentation_test')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEGMENTATION_CONFIG = {
    'gaussian_kernel': (7, 7),
    'clahe_clip_limit': 2.0,
    'clahe_grid': (8, 8),
    'adaptive_block_size': 35,
    'adaptive_c': 4,
    'open_kernel': (5, 5),
    'close_kernel': (9, 9),
    'border_margin_frac': 0.06,
    'center_weight': 0.25,
    'min_area_frac': 0.0008,
    'max_area_frac': 0.15,
    'min_solidity': 0.35,
    'min_extent': 0.12,
    'max_aspect_ratio': 5.0,
}


def _contour_touches_border(contour, width, height, margin):
    x, y, w, h = cv2.boundingRect(contour)
    return (
        x <= margin
        or y <= margin
        or (x + w) >= (width - margin)
        or (y + h) >= (height - margin)
    )


def segment_monkey_tuned(frame, config=None):
    """Tuned segmentation that suppresses cage detections."""
    cfg = SEGMENTATION_CONFIG.copy()
    if config is not None:
        cfg.update(config)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=cfg['clahe_clip_limit'], tileGridSize=cfg['clahe_grid'])
    gray_eq = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray_eq, cfg['gaussian_kernel'], 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, cfg['adaptive_block_size'], cfg['adaptive_c']
    )

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, cfg['open_kernel'])
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, cfg['close_kernel'])
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    h, w = gray.shape
    frame_area = h * w
    min_area = cfg['min_area_frac'] * frame_area
    max_area = cfg['max_area_frac'] * frame_area
    border_margin = int(cfg['border_margin_frac'] * min(h, w))
    cx_frame, cy_frame = w / 2.0, h / 2.0

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        if _contour_touches_border(contour, w, h, border_margin):
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        bbox_area = max(bw * bh, 1)
        extent = area / bbox_area

        hull = cv2.convexHull(contour)
        hull_area = max(cv2.contourArea(hull), 1.0)
        solidity = area / hull_area
        aspect_ratio = max(bw / max(bh, 1), bh / max(bw, 1))

        if solidity < cfg['min_solidity'] or extent < cfg['min_extent'] or aspect_ratio > cfg['max_aspect_ratio']:
            continue

        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        center_dist_norm = np.sqrt((cx - cx_frame)**2 + (cy - cy_frame)**2) / np.sqrt(cx_frame**2 + cy_frame**2)
        score = area * solidity * extent * (1.0 - cfg['center_weight'] * center_dist_norm)
        candidates.append((score, contour, area, (int(cx), int(cy)), (x, y, bw, bh), solidity, extent))

    if len(candidates) == 0:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_contour, best_area, centroid, bbox, best_solidity, best_extent = candidates[0]

    return {
        'contour': best_contour,
        'area': best_area,
        'bbox': bbox,
        'centroid': centroid,
        'solidity': best_solidity,
        'extent': best_extent,
        'n_candidates': len(candidates)
    }


def segment_monkey_original(frame):
    """Original naive segmentation (largest contour)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    x, y, w, h = cv2.boundingRect(largest_contour)
    M = cv2.moments(largest_contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = x + w//2, y + h//2

    return {
        'contour': largest_contour,
        'area': area,
        'bbox': (x, y, w, h),
        'centroid': (cx, cy)
    }


def extract_test_frames(video_path, n_frames=6):
    """Extract evenly spaced frames for testing."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frames = np.linspace(0, total_frames-1, n_frames, dtype=int)
    
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in target_frames:
            frames.append((frame_idx, frame))
        frame_idx += 1
        if len(frames) >= n_frames:
            break
    
    cap.release()
    return frames


def visualize_comparison(frames, video_name):
    """Compare original vs tuned segmentation side-by-side."""
    n_frames = len(frames)
    fig, axes = plt.subplots(n_frames, 3, figsize=(18, 6*n_frames))
    
    if n_frames == 1:
        axes = axes.reshape(1, -1)
    
    for i, (frame_idx, frame) in enumerate(frames):
        # Original
        axes[i, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f'Original Frame {frame_idx}')
        axes[i, 0].axis('off')
        
        # Original segmentation (naive - largest contour)
        result_orig = segment_monkey_original(frame)
        frame_orig = frame.copy()
        if result_orig:
            cv2.drawContours(frame_orig, [result_orig['contour']], -1, (0, 255, 0), 3)
            x, y, w, h = result_orig['bbox']
            cv2.rectangle(frame_orig, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cx, cy = result_orig['centroid']
            cv2.circle(frame_orig, (cx, cy), 8, (255, 0, 0), -1)
            axes[i, 1].imshow(cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB))
            axes[i, 1].set_title(f'ORIGINAL (Naive)\nArea={result_orig["area"]:.0f}')
        else:
            axes[i, 1].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            axes[i, 1].set_title('ORIGINAL (Failed)')
        axes[i, 1].axis('off')
        
        # Tuned segmentation (cage-resistant)
        result_tuned = segment_monkey_tuned(frame)
        frame_tuned = frame.copy()
        if result_tuned:
            h, w = frame.shape[:2]
            border_margin = int(SEGMENTATION_CONFIG['border_margin_frac'] * min(h, w))
            cv2.rectangle(frame_tuned, (border_margin, border_margin), 
                         (w - border_margin, h - border_margin), (255, 255, 0), 2)
            cv2.drawContours(frame_tuned, [result_tuned['contour']], -1, (0, 255, 0), 3)
            x, y, bw, bh = result_tuned['bbox']
            cv2.rectangle(frame_tuned, (x, y), (x+bw, y+bh), (255, 0, 0), 2)
            cx, cy = result_tuned['centroid']
            cv2.circle(frame_tuned, (cx, cy), 8, (255, 0, 0), -1)
            axes[i, 2].imshow(cv2.cvtColor(frame_tuned, cv2.COLOR_BGR2RGB))
            axes[i, 2].set_title(
                f'TUNED (Cage-Resistant)\nArea={result_tuned["area"]:.0f}, '
                f'Solid={result_tuned["solidity"]:.2f}, Candidates={result_tuned["n_candidates"]}'
            )
        else:
            axes[i, 2].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            axes[i, 2].set_title('TUNED (No valid contour)')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f'{video_name}_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison to: {output_path}")
    plt.close()


def main():
    print("="*70)
    print("SEGMENTATION TUNING TEST - Cage Suppression Validation")
    print("="*70)
    
    # Get test videos
    videos = sorted(VIDEO_DIR.glob('*.mp4'))
    if not videos:
        print(f"❌ ERROR: No videos found in {VIDEO_DIR}")
        sys.exit(1)
    
    print(f"\n📁 Found {len(videos)} videos in session folder")
    print(f"📂 Output directory: {OUTPUT_DIR}\n")
    
    # Test on first 3 videos (different day codes if possible)
    test_videos = videos[:3]
    
    for video_path in test_videos:
        print(f"\n{'='*70}")
        print(f"Testing: {video_path.name}")
        print(f"{'='*70}")
        
        # Extract test frames
        print("  Extracting test frames...")
        frames = extract_test_frames(video_path, n_frames=3)
        print(f"  ✓ Extracted {len(frames)} frames")
        
        # Generate comparison visualization
        print("  Generating comparison visualization...")
        visualize_comparison(frames, video_path.stem)
    
    print("\n" + "="*70)
    print("✅ SEGMENTATION TEST COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\n📊 Review the comparison images to verify:")
    print("  1. ORIGINAL column shows cage being detected (PROBLEM)")
    print("  2. TUNED column shows monkey being detected (FIXED)")
    print("  3. Yellow border shows exclusion zone for cage edges")
    print("  4. Green contour and blue bbox/centroid mark detected object")


if __name__ == '__main__':
    main()
