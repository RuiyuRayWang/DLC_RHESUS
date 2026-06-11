#!/usr/bin/env python3
"""
Generate manual labeling commands for all videos in a session.

Usage:
    python scripts/generate_label_commands.py --monkey H1705011 --date 2026-05-25
    
Output:
    Prints bash commands to label all videos, one command per video.
    You can copy-paste or pipe to a shell script.
"""

import argparse
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ['D1', 'D2', 'D3', 'D4']


def parse_video_timestamp(video_name: str) -> str:
    """Extract timestamp suffix from video filename for output CSV naming."""
    match = re.search(r'D0[1-4]_\d{8}(\d{6})', video_name)
    if match:
        return match.group(1)  # HHMMSS
    return 'unknown'


def main():
    parser = argparse.ArgumentParser(description='Generate manual labeling commands')
    parser.add_argument('--monkey', default='H1705011', help='Monkey ID')
    parser.add_argument('--date', default='2026-05-25', help='Session date (YYYY-MM-DD)')
    parser.add_argument('--sample-every', type=int, default=20, help='Frame sampling rate')
    args = parser.parse_args()

    video_dir = PROJECT_ROOT / 'data' / 'oft_rhp' / args.monkey / args.date
    output_dir = PROJECT_ROOT / 'results' / 'oft_analysis' / args.monkey / args.date

    if not video_dir.exists():
        print(f"ERROR: Video directory not found: {video_dir}")
        return

    print(f"# Manual labeling commands for {args.monkey} / {args.date}")
    print(f"# Generated from: {video_dir}")
    print(f"# Outputs will be saved to: {output_dir}\n")
    print("# Copy-paste these commands to label all videos:\n")

    for cam in CAMERAS:
        cam_num = cam[1]  # '1', '2', '3', '4'
        cam_prefix = f"D0{cam_num}_"  # D1 -> D01_
        videos = sorted(video_dir.glob(f"{cam_prefix}*.mp4"))

        if not videos:
            print(f"# No videos found for {cam}")
            continue

        if len(videos) == 1:
            # Single video - simple command
            video = videos[0]
            output_csv = output_dir / f'labels_{cam}.csv'
            print(f"# {cam}: {video.name}")
            print(f"python scripts/manual_labeler.py --camera {cam} \\")
            print(f"  --video {video.relative_to(PROJECT_ROOT)} \\")
            print(f"  --output {output_csv.relative_to(PROJECT_ROOT)} \\")
            print(f"  --sample-every {args.sample_every}\n")
        else:
            # Multiple videos - label each with timestamp suffix
            print(f"\n# {cam}: {len(videos)} videos")
            for video in videos:
                ts_suffix = parse_video_timestamp(video.name)
                output_csv = output_dir / f'labels_{cam}_{ts_suffix}.csv'
                print(f"# {video.name} → {output_csv.name}")
                print(f"python scripts/manual_labeler.py --camera {cam} \\")
                print(f"  --video {video.relative_to(PROJECT_ROOT)} \\")
                print(f"  --output {output_csv.relative_to(PROJECT_ROOT)} \\")
                print(f"  --sample-every {args.sample_every}\n")


if __name__ == '__main__':
    main()
