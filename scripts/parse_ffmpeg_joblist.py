#!/usr/bin/env python3
"""
Create dSQ joblists for OFT video preprocessing workflows.

This script supports two subcommands:

1. downsample
   Emit one command per video for downsample_ffmpeg.py.

2. segment
   Emit one command per video/visibility-annotation pair for segment_ffmpeg.py.

3. extract-frames
    Emit one command per video for prepare_ultralytics_bbox_dataset.py in
    inference-frame extraction mode.

The deprecated subcommand name "preprocess" is retained as an alias for "segment"
to avoid breaking older commands.

Examples
--------
Downsample videos to 0.5 scale / 1920x1080 from 4K sources:

python scripts/parse_ffmpeg_joblist.py downsample \
  data/oft_rhp/H1705011 \
  -j scripts/shell/joblist_downsample_H1705011.txt \
  --recursive \
  --skip-existing \
  --script scripts/downsample_ffmpeg.py \
  --raw-video-root data/oft_rhp \
  --output-dir data/oft_rhp/videos_downsampled \
  --scale 0.5 \
  --gop 20 \
  --threads 32 \
  --crf 18 \
  --preset veryfast

Segment original videos:

python scripts/parse_ffmpeg_joblist.py segment \
  data/oft_rhp/H1705011 \
  -j scripts/shell/joblist_segment_H1705011.txt \
  --recursive \
  --script scripts/segment_ffmpeg.py \
  --video-root data/oft_rhp \
  --annotation-root results/oft_analysis \
  --output-dir data/oft_rhp/visible_dlc_preprocessed \
  --target-fps 4 \
  --threads 32

Segment downsampled videos:

python scripts/parse_ffmpeg_joblist.py segment \
  data/oft_rhp/videos_downsampled/H1705011 \
  -j scripts/shell/joblist_segment_downsampled_H1705011.txt \
  --recursive \
  --script scripts/segment_ffmpeg.py \
  --video-root data/oft_rhp/videos_downsampled \
  --annotation-root results/oft_analysis/downsampled \
  --output-dir data/oft_rhp/visible_dlc_preprocessed_downsampled \
  --target-fps 4 \
  --threads 32

Extract sampled frames for RT-DETR inference:

python scripts/parse_ffmpeg_joblist.py extract-frames \
    data/oft_rhp/videos_downsampled/H1705011 \
    -j scripts/shell/joblist_extract_frames_H1705011.txt \
    --recursive \
    --script scripts/prepare_ultralytics_bbox_dataset.py \
    --videos-root data/oft_rhp/videos_downsampled \
    --frames-out-root tmp/rtdetr_frames_2hz \
    --sample-fps 2
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
        "--skip-existing",
        action="store_true",
        help="Skip videos whose expected output already exists.",
    )


def add_segment_subparser(
    subparsers: argparse._SubParsersAction,
    name: str,
    help_text: str,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(name, help=help_text)
    add_common_joblist_args(parser)
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("data/oft_rhp"),
        help="Video root used to infer relative monkey/date layout.",
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=Path("results/oft_analysis"),
        help="Root containing visibility annotation CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/oft_rhp/visible_dlc_preprocessed"),
        help="Output root passed to segment_ffmpeg.py.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=4.0,
        help="Sampling FPS passed to segment_ffmpeg.py.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Optional camera argument passed to segment_ffmpeg.py, e.g. D1.",
    )
    parser.add_argument(
        "--skip-missing-annotations",
        action="store_true",
        help="Skip videos whose inferred annotation CSV does not exist.",
    )
    parser.add_argument(
        "--annotation-video-file-mode",
        choices=["auto", "basename", "strip-downsampled"],
        default="auto",
        help=(
            "How to add --annotation-video-file to segment jobs. "
            "auto: do not pass it and let segment_ffmpeg.py infer. "
            "basename: pass video_path.name. "
            "strip-downsampled: strip --downsampled-suffix from video basename."
        ),
    )
    parser.add_argument(
        "--downsampled-suffix",
        type=str,
        default="_downsampled",
        help="Suffix used by strip-downsampled mode and passed to segment_ffmpeg.py in auto mode.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create dSQ joblists for OFT ffmpeg workflows.")
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
        help="Scale factor passed to downsample_ffmpeg.py. Used when --width/--height are omitted.",
    )
    downsample.add_argument(
        "--width",
        type=int,
        default=None,
        help="Explicit output width passed to downsample_ffmpeg.py. Must be used with --height.",
    )
    downsample.add_argument(
        "--height",
        type=int,
        default=None,
        help="Explicit output height passed to downsample_ffmpeg.py. Must be used with --width.",
    )
    downsample.add_argument(
        "--pad-to-size",
        action="store_true",
        help="With --width/--height, preserve aspect ratio and pad to the requested canvas.",
    )
    downsample.add_argument(
        "--force-original-aspect-ratio",
        action="store_true",
        help="With --width/--height, preserve aspect ratio without padding.",
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
        "--gop",
        type=int,
        default=20,
        help="H.264 keyframe interval passed to downsample_ffmpeg.py. Use 0 for x264 default.",
    )
    downsample.add_argument(
        "--suffix",
        type=str,
        default="_downsampled",
        help="Output suffix passed to downsample_ffmpeg.py and used for skip-existing checks.",
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

    add_segment_subparser(
        subparsers,
        name="segment",
        help_text="Create joblist for segment_ffmpeg.py.",
    )
    add_segment_subparser(
        subparsers,
        name="preprocess",
        help_text="Deprecated alias for segment.",
    )

    extract_frames = subparsers.add_parser(
        "extract-frames",
        help="Create joblist for inference-frame extraction with prepare_ultralytics_bbox_dataset.py.",
    )
    add_common_joblist_args(extract_frames)
    extract_frames.add_argument(
        "--videos-root",
        type=Path,
        default=Path("data/oft_rhp/videos_downsampled"),
        help="Video root used to preserve relative monkey/date layout.",
    )
    extract_frames.add_argument(
        "--frames-out-root",
        type=Path,
        default=Path("tmp/rtdetr_frames_2hz"),
        help="Output root for extracted inference frames.",
    )
    extract_frames.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="Sampling FPS passed to prepare_ultralytics_bbox_dataset.py in inference mode.",
    )
    extract_frames.add_argument(
        "--input-fps-fallback",
        type=float,
        default=20.0,
        help="Fallback FPS passed to prepare_ultralytics_bbox_dataset.py.",
    )
    extract_frames.add_argument(
        "--image-ext",
        choices=["jpg", "png"],
        default="jpg",
        help="Image extension for extracted frames.",
    )
    extract_frames.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted frames for each video.",
    )
    extract_frames.add_argument(
        "--tar-per-video",
        action="store_true",
        help="Create one tar archive per extracted video frame folder.",
    )
    extract_frames.add_argument(
        "--tar-mode",
        choices=["tar", "tar.gz"],
        default="tar",
        help="Archive format passed to prepare_ultralytics_bbox_dataset.py.",
    )
    extract_frames.add_argument(
        "--tar-overwrite",
        action="store_true",
        help="Overwrite existing tar archive when --tar-per-video is enabled.",
    )
    extract_frames.add_argument(
        "--tar-remove-frames",
        action="store_true",
        help="Remove extracted frame files after tar creation.",
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


def quote_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


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
    output_dir = Path(output_dir)

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
        "--threads",
        str(args.threads),
        "--crf",
        str(args.crf),
        "--preset",
        args.preset,
        "--gop",
        str(args.gop),
        "--suffix",
        args.suffix,
    ]

    if args.width is not None or args.height is not None:
        if args.width is None or args.height is None:
            raise ValueError("--width and --height must be provided together.")
        cmd.extend(["--width", str(args.width), "--height", str(args.height)])
    else:
        cmd.extend(["--scale", str(args.scale)])

    if args.pad_to_size:
        cmd.append("--pad-to-size")

    if args.force_original_aspect_ratio:
        cmd.append("--force-original-aspect-ratio")

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

    return quote_command(cmd)


def run_downsample(args: argparse.Namespace) -> int:
    if (args.width is None) != (args.height is None):
        raise ValueError("--width and --height must be provided together.")
    if args.pad_to_size and args.force_original_aspect_ratio:
        raise ValueError("Use only one of --pad-to-size or --force-original-aspect-ratio.")

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
                suffix=args.suffix,
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


def infer_segment_annotation_csv(
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


def expected_segment_output_path(
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


def build_segment_command(
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

    return quote_command(cmd)


def run_segment(args: argparse.Namespace) -> int:
    video_paths = collect_video_paths(
        video_dir=args.video_dir,
        pattern=args.pattern,
        recursive=args.recursive,
    )

    commands: list[str] = []
    n_missing_annotations = 0
    n_existing = 0

    for video_path in video_paths:
        annotation_csv = infer_segment_annotation_csv(
            video_path=video_path,
            video_root=args.video_root,
            annotation_root=args.annotation_root,
        )

        if not annotation_csv.exists():
            n_missing_annotations += 1
            if args.skip_missing_annotations:
                continue

        if args.skip_existing:
            output_video = expected_segment_output_path(
                video_path=video_path,
                output_dir=args.output_dir,
            )
            if output_video.exists():
                n_existing += 1
                continue

        commands.append(build_segment_command(video_path, annotation_csv, args))

    write_commands(commands, args.joblist)

    label = "segment" if args.mode == "segment" else "preprocess/segment"
    if args.joblist is not None:
        print(f"Wrote {len(commands)} {label} jobs to {args.joblist}", file=sys.stderr)
        if n_missing_annotations:
            print(f"Detected {n_missing_annotations} videos with missing annotation CSVs.", file=sys.stderr)
            if not args.skip_missing_annotations:
                print("Those commands were still emitted and may fail if the CSVs remain absent.", file=sys.stderr)
        if n_existing:
            print(f"Skipped {n_existing} existing segment outputs.", file=sys.stderr)

    return len(commands)


def expected_extract_output_dir(
    video_path: Path,
    videos_root: Path,
    frames_out_root: Path,
) -> Path:
    video_path_resolved = video_path.resolve()
    videos_root_resolved = videos_root.resolve()

    try:
        rel_path = video_path_resolved.relative_to(videos_root_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Video path is not under videos_root.\n"
            f"video_path: {video_path_resolved}\n"
            f"videos_root: {videos_root_resolved}"
        ) from exc

    return frames_out_root / rel_path.parent / rel_path.stem


def expected_extract_tar_path(
    video_path: Path,
    videos_root: Path,
    frames_out_root: Path,
    tar_mode: str,
) -> Path:
    output_dir = expected_extract_output_dir(
        video_path=video_path,
        videos_root=videos_root,
        frames_out_root=frames_out_root,
    )
    suffix = ".tar.gz" if tar_mode == "tar.gz" else ".tar"
    return output_dir.parent / f"{video_path.stem}{suffix}"


def build_extract_frames_command(
    video_path: Path,
    args: argparse.Namespace,
) -> str:
    cmd = [
        args.python,
        args.script,
        "--mode",
        "inference",
        "--videos-dir",
        str(args.videos_root),
        "--video-path",
        str(video_path),
        "--videos-root",
        str(args.videos_root),
        "--out-root",
        str(args.frames_out_root),
        "--sample-fps",
        str(args.sample_fps),
        "--input-fps-fallback",
        str(args.input_fps_fallback),
        "--image-ext",
        args.image_ext,
    ]
    if args.tar_per_video:
        cmd.extend(["--tar-per-video", "--tar-mode", args.tar_mode])
        if args.tar_overwrite:
            cmd.append("--tar-overwrite")
        if args.tar_remove_frames:
            cmd.append("--tar-remove-frames")
    if args.overwrite:
        cmd.append("--overwrite")
    return quote_command(cmd)


def run_extract_frames(args: argparse.Namespace) -> int:
    video_paths = collect_video_paths(
        video_dir=args.video_dir,
        pattern=args.pattern,
        recursive=args.recursive,
    )

    commands: list[str] = []
    n_existing = 0
    img_ext = args.image_ext.lower()

    for video_path in video_paths:
        if args.skip_existing and not args.overwrite:
            if args.tar_per_video:
                tar_path = expected_extract_tar_path(
                    video_path=video_path,
                    videos_root=args.videos_root,
                    frames_out_root=args.frames_out_root,
                    tar_mode=args.tar_mode,
                )
                if tar_path.exists():
                    n_existing += 1
                    continue
            else:
                output_dir = expected_extract_output_dir(
                    video_path=video_path,
                    videos_root=args.videos_root,
                    frames_out_root=args.frames_out_root,
                )
                existing = list(output_dir.glob(f"{video_path.stem}_f*.{img_ext}"))
                if existing:
                    n_existing += 1
                    continue

        commands.append(build_extract_frames_command(video_path, args))

    write_commands(commands, args.joblist)

    if args.joblist is not None:
        print(f"Wrote {len(commands)} extract-frames jobs to {args.joblist}", file=sys.stderr)
        if n_existing:
            print(f"Skipped {n_existing} videos with existing extracted frames.", file=sys.stderr)

    return len(commands)


def main() -> None:
    args = parse_args()

    if args.mode == "downsample":
        run_downsample(args)
    elif args.mode in {"segment", "preprocess"}:
        run_segment(args)
    elif args.mode == "extract-frames":
        run_extract_frames(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
