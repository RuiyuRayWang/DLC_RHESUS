#!/usr/bin/env python
"""
Create dSQ joblists for ffmpeg-based OFT video workflows.

This script supports two subcommands:

1. downsample
   Emit one command per video for downsample_ffmpeg.py.

2. preprocess
   Emit one command per video/annotation pair for preprocess_ffmpeg.py.

Examples
--------
Downsample joblist:

python scripts/parse_oft_ffmpeg_joblist.py downsample \
  data/oft_rhp/H1705011 \
  -j scripts/shell/joblist_downsample_H1705011.txt \
  --recursive \
  --skip-existing \
  --script scripts/downsample_ffmpeg.py \
  --raw-video-root data/oft_rhp \
  --output-dir data/oft_rhp/videos_downsampled \
  --scale 0.5 \
  --threads 32 \
  --downsample-csv \
  --module ffmpeg/9.1

Preprocess original videos:

python scripts/parse_oft_ffmpeg_joblist.py preprocess \
  data/oft_rhp/H1705011 \
  -j scripts/shell/joblist_preprocess_H1705011.txt \
  --recursive \
  --script scripts/preprocess_ffmpeg.py \
  --video-root data/oft_rhp \
  --annotation-root results/oft_analysis \
  --output-dir data/oft_rhp/visible_dlc_preprocessed \
  --target-fps 4 \
  --threads 32 \
  --module ffmpeg/9.1

Preprocess downsampled videos:

python scripts/parse_oft_ffmpeg_joblist.py preprocess \
  data/oft_rhp/videos_downsampled/H1705011 \
  -j scripts/shell/joblist_preprocess_downsampled_H1705011.txt \
  --recursive \
  --script scripts/preprocess_ffmpeg.py \
  --video-root data/oft_rhp/videos_downsampled \
  --annotation-root results/oft_analysis/downsampled \
  --output-dir data/oft_rhp/visible_dlc_preprocessed_downsampled \
  --target-fps 4 \
  --threads 32 \
  --module ffmpeg/9.1
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path


def add_common_joblist_args(parser: argparse.ArgumentParser) -> None:
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
        required=True,
        help="Path to the workflow script to run in each job.",
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
        "--threads",
        type=int,
        default=32,
        help="Thread count passed to the workflow script.",
    )
    parser.add_argument(
        "--module",
        action="append",
        default=[],
        help="Module to load before each job command. Can be used multiple times.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos whose expected output already exists.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create dSQ joblists for OFT ffmpeg workflows."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    downsample = subparsers.add_parser(
        "downsample",
        help="Create joblist for downsample_ffmpeg.py.",
    )
    add_common_joblist_args(downsample)
    downsample.add_argument(
        "--raw-video-root",
        type=Path,
        default=Path("data/oft_rhp"),
        help="Raw video root passed to downsample_ffmpeg.py.",
    )
    downsample.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/oft_rhp/videos_downsampled"),
        help="Output root for downsampled videos.",
    )
    downsample.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Scale factor passed to downsample_ffmpeg.py.",
    )
    downsample.add_argument(
        "--threads",
        type=int,
        default=32,
        help="ffmpeg thread count passed to downsample_ffmpeg.py.",
    )
    downsample.add_argument(
        "--crf",
        type=int,
        default=18,
        help="H.264 CRF quality passed to downsample_ffmpeg.py.",
    )
    downsample.add_argument(
        "--preset",
        type=str,
        default="veryfast",
        help="x264 preset passed to downsample_ffmpeg.py.",
    )
    downsample.add_argument(
        "--downsample-csv",
        action="store_true",
        help="Add --downsample-csv to each job command.",
    )
    downsample.add_argument(
        "--analysis-root",
        type=Path,
        default=Path("results/oft_analysis"),
        help="Annotation CSV root passed to downsample_ffmpeg.py.",
    )
    downsample.add_argument(
        "--csv-output-dir",
        type=Path,
        default=Path("results/oft_analysis/downsampled"),
        help="Downsampled CSV output root passed to downsample_ffmpeg.py.",
    )
    downsample.add_argument(
        "--require-csv",
        action="store_true",
        help="Add --require-csv to each job command.",
    )
    downsample.add_argument(
        "--overwrite",
        action="store_true",
        help="Add --overwrite to each job command.",
    )
    downsample.add_argument(
        "--keep-audio",
        action="store_true",
        help="Add --keep-audio to each job command.",
    )

    preprocess = subparsers.add_parser(
        "preprocess",
        help="Create joblist for preprocess_ffmpeg.py.",
    )
    add_common_joblist_args(preprocess)
    preprocess.add_argument(
        "--video-root",
        type=Path,
        default=Path("data/oft_rhp"),
        help="Video root used to infer relative monkey/date layout.",
    )
    preprocess.add_argument(
        "--annotation-root",
        type=Path,
        default=Path("results/oft_analysis"),
        help="Root containing annotation CSVs.",
    )
    preprocess.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/oft_rhp/visible_dlc_preprocessed"),
        help="Output root passed to preprocess_ffmpeg.py.",
    )
    preprocess.add_argument(
        "--target-fps",
        type=float,
        default=4.0,
        help="Sampling FPS passed to preprocess_ffmpeg.py.",
    )
    preprocess.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Optional camera argument passed to preprocess_ffmpeg.py, e.g. D1.",
    )
    preprocess.add_argument(
        "--skip-missing-annotations",
        action="store_true",
        help="Skip videos whose inferred annotation CSV does not exist.",
    )
    preprocess.add_argument(
        "--annotation-video-file-mode",
        choices=["auto", "basename", "strip-downsampled"],
        default="auto",
        help=(
            "How to add --annotation-video-file to preprocess jobs. "
            "auto: do not pass it and let patched preprocess_ffmpeg.py infer. "
            "basename: pass video_path.name. "
            "strip-downsampled: strip --downsampled-suffix from video basename."
        ),
    )
    preprocess.add_argument(
        "--downsampled-suffix",
        type=str,
        default="_downsampled",
        help="Suffix used by strip-downsampled mode and passed to preprocess_ffmpeg.py in auto mode.",
    )

    return parser.parse_args()


def collect_video_paths(
    video_dir: Path,
    pattern: str,
    recursive: bool,
) -> list[Path]:
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    if recursive:
        video_paths = sorted(video_dir.rglob(pattern))
    else:
        video_paths = sorted(video_dir.glob(pattern))

    if not video_paths:
        raise FileNotFoundError(f"No videos found in {video_dir} matching {pattern}")

    return video_paths


def wrap_with_modules(command: str, modules: list[str]) -> str:
    if not modules:
        return command

    setup_cmds = [f"module load {shlex.quote(module)}" for module in modules]
    full_cmd = " && ".join(setup_cmds + [command])
    return "bash -lc " + shlex.quote(full_cmd)


def write_commands(
    commands: list[str],
    joblist: Path | None,
) -> None:
    if joblist is None:
        for command in commands:
            print(command)
        return

    joblist.parent.mkdir(parents=True, exist_ok=True)
    with open(joblist, "w") as ostream:
        for command in commands:
            print(command, file=ostream)


def expected_downsample_output_path(
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
    args: argparse.Namespace,
) -> str:
    cmd = [
        args.python,
        args.script,
        "--video-path",
        str(video_path),
        "--raw-video-root",
        str(args.raw_video_root),
        "--output-dir",
        str(args.output_dir),
        "--scale",
        str(args.scale),
        "--threads",
        str(args.threads),
        "--crf",
        str(args.crf),
        "--preset",
        args.preset,
    ]

    if args.downsample_csv:
        cmd.extend(
            [
                "--downsample-csv",
                "--analysis-root",
                str(args.analysis_root),
                "--csv-output-dir",
                str(args.csv_output_dir),
            ]
        )

    if args.require_csv:
        cmd.append("--require-csv")

    if args.overwrite:
        cmd.append("--overwrite")

    if args.keep_audio:
        cmd.append("--keep-audio")

    command = " ".join(shlex.quote(part) for part in cmd)
    return wrap_with_modules(command, args.module)


def run_downsample(args: argparse.Namespace) -> int:
    video_paths = collect_video_paths(
        video_dir=args.video_dir,
        pattern=args.pattern,
        recursive=args.recursive,
    )

    commands: list[str] = []
    n_existing = 0

    for video_path in video_paths:
        if args.skip_existing:
            output_video = expected_downsample_output_path(
                video_path=video_path,
                raw_video_root=args.raw_video_root,
                output_dir=args.output_dir,
            )
            if output_video.exists():
                n_existing += 1
                continue

        commands.append(build_downsample_command(video_path, args))

    write_commands(commands, args.joblist)

    if args.joblist is not None:
        print(f"Wrote {len(commands)} downsample jobs to {args.joblist}", file=sys.stderr)
        if n_existing:
            print(f"Skipped {n_existing} existing downsample outputs.", file=sys.stderr)

    return len(commands)


def infer_preprocess_annotation_csv(
    video_path: Path,
    video_root: Path,
    annotation_root: Path,
) -> Path:
    video_path_resolved = video_path.resolve()
    video_root_resolved = video_root.resolve()

    try:
        rel_path = video_path_resolved.relative_to(video_root_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Video path is not under video_root.\n"
            f"video_path: {video_path_resolved}\n"
            f"video_root: {video_root_resolved}"
        ) from exc

    return annotation_root / rel_path.with_suffix(".csv")


def expected_preprocess_output_path(
    video_path: Path,
    output_dir: Path,
) -> Path:
    return output_dir / video_path.stem / f"{video_path.stem}_visible_h264.mp4"


def infer_annotation_video_file_for_job(
    video_path: Path,
    mode: str,
    downsampled_suffix: str,
) -> str | None:
    if mode == "auto":
        return None

    if mode == "basename":
        return video_path.name

    if mode == "strip-downsampled":
        stem = video_path.stem
        if downsampled_suffix and stem.endswith(downsampled_suffix):
            stem = stem[: -len(downsampled_suffix)]
        return f"{stem}{video_path.suffix}"

    raise ValueError(f"Unsupported annotation-video-file mode: {mode}")


def build_preprocess_command(
    video_path: Path,
    annotation_csv: Path,
    args: argparse.Namespace,
) -> str:
    cmd = [
        args.python,
        args.script,
        "--video",
        str(video_path),
        "--annotations",
        str(annotation_csv),
        "--output-dir",
        str(args.output_dir),
        "--target-fps",
        str(args.target_fps),
        "--threads",
        str(args.threads),
    ]

    if args.camera is not None:
        cmd.extend(["--camera", args.camera])

    annotation_video_file = infer_annotation_video_file_for_job(
        video_path=video_path,
        mode=args.annotation_video_file_mode,
        downsampled_suffix=args.downsampled_suffix,
    )
    if annotation_video_file is not None:
        cmd.extend(["--annotation-video-file", annotation_video_file])
    elif args.downsampled_suffix:
        cmd.extend(["--downsampled-suffix", args.downsampled_suffix])

    command = " ".join(shlex.quote(part) for part in cmd)
    return wrap_with_modules(command, args.module)


def run_preprocess(args: argparse.Namespace) -> int:
    video_paths = collect_video_paths(
        video_dir=args.video_dir,
        pattern=args.pattern,
        recursive=args.recursive,
    )

    commands: list[str] = []
    n_missing_annotations = 0
    n_existing = 0

    for video_path in video_paths:
        annotation_csv = infer_preprocess_annotation_csv(
            video_path=video_path,
            video_root=args.video_root,
            annotation_root=args.annotation_root,
        )

        if not annotation_csv.exists():
            n_missing_annotations += 1
            if args.skip_missing_annotations:
                continue

        if args.skip_existing:
            output_video = expected_preprocess_output_path(
                video_path=video_path,
                output_dir=args.output_dir,
            )
            if output_video.exists():
                n_existing += 1
                continue

        commands.append(build_preprocess_command(video_path, annotation_csv, args))

    write_commands(commands, args.joblist)

    if args.joblist is not None:
        print(f"Wrote {len(commands)} preprocess jobs to {args.joblist}", file=sys.stderr)
        if n_missing_annotations:
            print(f"Detected {n_missing_annotations} videos with missing annotation CSVs.", file=sys.stderr)
            if not args.skip_missing_annotations:
                print("Those commands were still emitted and may fail if the CSVs remain absent.", file=sys.stderr)
        if n_existing:
            print(f"Skipped {n_existing} existing preprocess outputs.", file=sys.stderr)

    return len(commands)


def main() -> None:
    args = parse_args()

    if args.mode == "downsample":
        run_downsample(args)
    elif args.mode == "preprocess":
        run_preprocess(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
