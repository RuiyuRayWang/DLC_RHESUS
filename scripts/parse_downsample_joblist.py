#!/usr/bin/env python
"""
Create a dSQ joblist for video downsampling with ffmpeg.

Each output line is one command that processes one video using downsample_ffmpeg.py.

Example
-------
python parse_downsample_joblist.py \
  data/oft_rhp/H1705011/2026-05-25 \
  -j scripts/shell/joblist_downsample.txt \
  --raw-video-root data/oft_rhp \
  --output-dir data/oft_rhp/videos_downsampled \
  --scale 0.5 \
  --threads 32 \
  --downsample-csv

Then submit with dSQ, for example:
dsq --job-file scripts/shell/joblist_downsample.txt --cpus-per-task 32 --mem 8G -t 02:00:00
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a dSQ joblist for downsample_ffmpeg.py."
    )

    parser.add_argument(
        "video_dir",
        type=Path,
        help="Directory containing input videos. Can be a monkey/date folder or a higher-level root.",
    )

    parser.add_argument(
        "-j",
        "--joblist",
        type=Path,
        default=None,
        help="Output dSQ joblist. If omitted, print to stdout.",
    )

    parser.add_argument(
        "--script",
        type=str,
        default="downsample_ffmpeg.py",
        help="Path to the single-job downsampling script.",
    )

    parser.add_argument(
        "--python",
        type=str,
        default="python",
        help="Python executable or command to use in each job line.",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.mp4",
        help="Video filename glob pattern.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for videos under video_dir.",
    )

    parser.add_argument(
        "--raw-video-root",
        type=Path,
        default=Path("data/oft_rhp"),
        help="Raw video root passed to downsample_ffmpeg.py.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/oft_rhp/videos_downsampled"),
        help="Output root for downsampled videos.",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Scale factor passed to downsample_ffmpeg.py.",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=32,
        help="ffmpeg thread count passed to downsample_ffmpeg.py.",
    )

    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="H.264 CRF quality passed to downsample_ffmpeg.py.",
    )

    parser.add_argument(
        "--preset",
        type=str,
        default="veryfast",
        help="x264 preset passed to downsample_ffmpeg.py.",
    )

    parser.add_argument(
        "--downsample-csv",
        action="store_true",
        help="Add --downsample-csv to each job command.",
    )

    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=Path("results/oft_analysis"),
        help="Annotation CSV root passed to downsample_ffmpeg.py.",
    )

    parser.add_argument(
        "--csv-output-dir",
        type=Path,
        default=Path("results/oft_analysis/downsampled"),
        help="Downsampled CSV output root passed to downsample_ffmpeg.py.",
    )

    parser.add_argument(
        "--require-csv",
        action="store_true",
        help="Add --require-csv to each job command.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Add --overwrite to each job command.",
    )

    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Add --keep-audio to each job command.",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Do not emit commands for videos whose expected downsampled output already exists. "
            "This check uses --raw-video-root and --output-dir."
        ),
    )

    return parser.parse_args()


def quote_path(path: str | Path) -> str:
    return shlex.quote(str(path))


def expected_output_path(
    video_path: Path,
    raw_video_root: Path,
    output_dir: Path,
    suffix: str = "_downsampled",
) -> Path:
    video_path_resolved = video_path.resolve()
    raw_video_root_resolved = raw_video_root.resolve()

    try:
        rel_path = video_path_resolved.relative_to(raw_video_root_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Video path is not under raw_video_root.\n"
            f"video_path: {video_path_resolved}\n"
            f"raw_video_root: {raw_video_root_resolved}"
        ) from exc

    return output_dir / rel_path.parent / f"{video_path.stem}{suffix}.mp4"


def build_downsample_command(
    video_path: Path,
    python_cmd: str,
    script: str,
    raw_video_root: Path,
    output_dir: Path,
    scale: float,
    threads: int,
    crf: int,
    preset: str,
    downsample_csv: bool,
    analysis_root: Path,
    csv_output_dir: Path,
    require_csv: bool,
    overwrite: bool,
    keep_audio: bool,
) -> str:
    cmd = [
        python_cmd,
        script,
        "--video-path",
        str(video_path),
        "--raw-video-root",
        str(raw_video_root),
        "--output-dir",
        str(output_dir),
        "--scale",
        str(scale),
        "--threads",
        str(threads),
        "--crf",
        str(crf),
        "--preset",
        preset,
    ]

    if downsample_csv:
        cmd.extend(
            [
                "--downsample-csv",
                "--analysis-root",
                str(analysis_root),
                "--csv-output-dir",
                str(csv_output_dir),
            ]
        )

    if require_csv:
        cmd.append("--require-csv")

    if overwrite:
        cmd.append("--overwrite")

    if keep_audio:
        cmd.append("--keep-audio")

    return " ".join(shlex.quote(part) for part in cmd)


def parse_joblist(
    video_dir: Path,
    joblist: Path | None,
    script: str,
    python_cmd: str,
    pattern: str,
    recursive: bool,
    raw_video_root: Path,
    output_dir: Path,
    scale: float,
    threads: int,
    crf: int,
    preset: str,
    downsample_csv: bool,
    analysis_root: Path,
    csv_output_dir: Path,
    require_csv: bool,
    overwrite: bool,
    keep_audio: bool,
    skip_existing: bool,
) -> int:
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    if recursive:
        video_paths = sorted(video_dir.rglob(pattern))
    else:
        video_paths = sorted(video_dir.glob(pattern))

    if not video_paths:
        raise FileNotFoundError(f"No videos found in {video_dir} matching {pattern}")

    if joblist is not None:
        joblist.parent.mkdir(parents=True, exist_ok=True)
        ostream = open(joblist, "w")
    else:
        ostream = sys.stdout

    n_written = 0

    try:
        for video_path in video_paths:
            if skip_existing:
                output_video = expected_output_path(
                    video_path=video_path,
                    raw_video_root=raw_video_root,
                    output_dir=output_dir,
                )
                if output_video.exists():
                    continue

            command = build_downsample_command(
                video_path=video_path,
                python_cmd=python_cmd,
                script=script,
                raw_video_root=raw_video_root,
                output_dir=output_dir,
                scale=scale,
                threads=threads,
                crf=crf,
                preset=preset,
                downsample_csv=downsample_csv,
                analysis_root=analysis_root,
                csv_output_dir=csv_output_dir,
                require_csv=require_csv,
                overwrite=overwrite,
                keep_audio=keep_audio,
            )

            print(command, file=ostream)
            n_written += 1

    finally:
        if joblist is not None:
            ostream.close()

    return n_written


def main() -> None:
    args = parse_args()

    n_written = parse_joblist(
        video_dir=args.video_dir,
        joblist=args.joblist,
        script=args.script,
        python_cmd=args.python,
        pattern=args.pattern,
        recursive=args.recursive,
        raw_video_root=args.raw_video_root,
        output_dir=args.output_dir,
        scale=args.scale,
        threads=args.threads,
        crf=args.crf,
        preset=args.preset,
        downsample_csv=args.downsample_csv,
        analysis_root=args.analysis_root,
        csv_output_dir=args.csv_output_dir,
        require_csv=args.require_csv,
        overwrite=args.overwrite,
        keep_audio=args.keep_audio,
        skip_existing=args.skip_existing,
    )

    if args.joblist is not None:
        print(f"Wrote {n_written} jobs to {args.joblist}", file=sys.stderr)


if __name__ == "__main__":
    main()
