from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RAW_VIDEO_ROOT = PROJECT_ROOT / "data" / "oft_rhp"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "oft_rhp" / "videos_downsampled"

DEFAULT_ANALYSIS_ROOT = PROJECT_ROOT / "results" / "oft_analysis"
DEFAULT_CSV_OUTPUT_DIR = DEFAULT_ANALYSIS_ROOT / "downsampled"

DEFAULT_SCALE = 0.5
DEFAULT_CRF = 18
DEFAULT_PRESET = "veryfast"
DEFAULT_THREADS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downsample one 4K video for DeepLabCut training/inference.")
    parser.add_argument("--video-path", type=Path, required=True, help="Path to one input video. Intended for one HPC array task.",)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Root output directory for downsampled videos.",)
    parser.add_argument("--raw-video-root", type=Path, default=DEFAULT_RAW_VIDEO_ROOT, help="Raw video root used to preserve monkey/date layout.",)
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE, help="Scale factor. Example: 0.5 converts 3840x2160 to 1920x1080.",)
    parser.add_argument("--width", type=int, default=None, help="Explicit output width. Must be used with --height.",)
    parser.add_argument("--height", type=int, default=None, help="Explicit output height. Must be used with --width.",)
    parser.add_argument("--crf", type=int, default=DEFAULT_CRF, help="H.264 CRF quality. Lower is better quality/larger file.",)
    parser.add_argument("--preset", type=str, default=DEFAULT_PRESET, help="x264 preset: ultrafast, superfast, veryfast, faster, fast, medium, slow.",)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="ffmpeg thread count. Use the CPU allocation from the scheduler.",)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing video/CSV outputs.",)
    parser.add_argument("--keep-audio", action="store_true", help="Keep audio track. Default is to remove audio.",)
    parser.add_argument("--downsample-csv", action="store_true", help="Also downsample the matching annotation CSV centroid columns.",)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT, help="Root directory containing original annotation CSVs.",)
    parser.add_argument("--csv-output-dir", type=Path, default=DEFAULT_CSV_OUTPUT_DIR, help="Root output directory for downsampled annotation CSVs.",)
    parser.add_argument("--require-csv", action="store_true", help="Raise an error if the matching annotation CSV is missing.",)
    return parser.parse_args()


def downsampled_output_path(
    video_path: str | Path,
    raw_video_root: str | Path = DEFAULT_RAW_VIDEO_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_DIR,
    suffix: str = "_downsampled",
    output_ext: str = ".mp4",
) -> Path:
    """
    Compose output path for one downsampled video while preserving monkey/date layout.

    Input:
        data/oft_rhp/H1705011/2026-05-25/D01_20260525164157.mp4

    Output:
        data/oft_rhp/videos_downsampled/H1705011/2026-05-25/D01_20260525164157_downsampled.mp4
    """

    video_path = Path(video_path).resolve()
    raw_video_root = Path(raw_video_root).resolve()
    output_root = Path(output_root).resolve()

    try:
        rel_path = video_path.relative_to(raw_video_root)
    except ValueError as exc:
        raise ValueError(
            f"Video path is not under raw_video_root.\n"
            f"video_path: {video_path}\n"
            f"raw_video_root: {raw_video_root}"
        ) from exc

    return output_root / rel_path.parent / f"{video_path.stem}{suffix}{output_ext}"


def downsampled_csv_paths_from_video(
    video_path: str | Path,
    raw_video_root: str | Path = DEFAULT_RAW_VIDEO_ROOT,
    analysis_root: str | Path = DEFAULT_ANALYSIS_ROOT,
    csv_output_root: str | Path = DEFAULT_CSV_OUTPUT_DIR,
    suffix: str = "_downsampled",
) -> tuple[Path, Path]:
    """
    Infer input/output annotation CSV paths from a video path.

    Video:
        data/oft_rhp/H1705011/2026-05-25/D01_20260525164157.mp4

    Input CSV:
        results/oft_analysis/H1705011/2026-05-25/D01_20260525164157.csv

    Output CSV:
        results/oft_analysis/downsampled/H1705011/2026-05-25/D01_20260525164157_downsampled.csv
    """

    video_path = Path(video_path).resolve()
    raw_video_root = Path(raw_video_root).resolve()
    analysis_root = Path(analysis_root).resolve()
    csv_output_root = Path(csv_output_root).resolve()

    try:
        rel_video_path = video_path.relative_to(raw_video_root)
    except ValueError as exc:
        raise ValueError(
            f"Video path is not under raw_video_root.\n"
            f"video_path: {video_path}\n"
            f"raw_video_root: {raw_video_root}"
        ) from exc

    rel_csv_path = rel_video_path.with_suffix(".csv")
    input_csv = analysis_root / rel_csv_path
    output_csv = csv_output_root / rel_csv_path.parent / f"{rel_csv_path.stem}{suffix}.csv"

    return input_csv, output_csv


def run_ffmpeg(cmd: list[str]) -> None:
    """
    Run ffmpeg and raise a useful error if it fails.
    """

    print("Running:")
    print(" ".join(cmd), flush=True)

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed.\n\n"
            f"Command:\n{' '.join(cmd)}\n\n"
            f"STDERR:\n{result.stderr}"
        )


def downsample_annotation_csv_for_video(
    video_path: str | Path,
    scale: float,
    raw_video_root: str | Path = DEFAULT_RAW_VIDEO_ROOT,
    analysis_root: str | Path = DEFAULT_ANALYSIS_ROOT,
    csv_output_root: str | Path = DEFAULT_CSV_OUTPUT_DIR,
    overwrite: bool = False,
    require_csv: bool = False,
) -> Path | None:
    """
    Downsample centroid_x and centroid_y in the sidecar annotation CSV.

    Only centroid_x and centroid_y are modified. Negative sentinel values
    such as -1 are preserved.
    """

    input_csv, output_csv = downsampled_csv_paths_from_video(
        video_path=video_path,
        raw_video_root=raw_video_root,
        analysis_root=analysis_root,
        csv_output_root=csv_output_root,
    )

    if not input_csv.exists():
        msg = f"Annotation CSV not found, skipping CSV preprocessing: {input_csv}"
        if require_csv:
            raise FileNotFoundError(msg)
        print(msg, flush=True)
        return None

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if output_csv.exists() and not overwrite:
        print(f"Output CSV already exists, skipping: {output_csv}", flush=True)
        return output_csv

    df = pd.read_csv(input_csv)

    required_cols = {"centroid_x", "centroid_y"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required centroid columns in {input_csv}: {sorted(missing_cols)}"
        )

    for col in ("centroid_x", "centroid_y"):
        values = pd.to_numeric(df[col], errors="coerce")
        valid = values >= 0
        df.loc[valid, col] = values.loc[valid] * scale

    df.to_csv(output_csv, index=False)

    print(f"Saved downsampled annotation CSV: {output_csv}", flush=True)

    return output_csv


def downsample_one_video_job(
    video_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    raw_video_root: str | Path = DEFAULT_RAW_VIDEO_ROOT,
    scale: float = DEFAULT_SCALE,
    width: int | None = None,
    height: int | None = None,
    crf: int = DEFAULT_CRF,
    preset: str = DEFAULT_PRESET,
    threads: int = DEFAULT_THREADS,
    overwrite: bool = False,
    keep_audio: bool = False,
    downsample_csv: bool = False,
    analysis_root: str | Path = DEFAULT_ANALYSIS_ROOT,
    csv_output_dir: str | Path = DEFAULT_CSV_OUTPUT_DIR,
    require_csv: bool = False,
) -> Path:
    """
    Process one HPC job: one video, optionally one matching annotation CSV.
    """

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    output_video = downsampled_output_path(
        video_path=video_path,
        raw_video_root=raw_video_root,
        output_root=output_dir,
    )
    output_video.parent.mkdir(parents=True, exist_ok=True)

    if (width is None) != (height is None):
        raise ValueError("width and height must either both be provided or both be None.")

    if width is not None and height is not None:
        scale_filter = f"scale={width}:{height}"
        if downsample_csv:
            raise ValueError(
                "CSV coordinate downsampling currently requires --scale. "
                "Use --scale instead of --width/--height when --downsample-csv is enabled."
            )
    else:
        if scale <= 0 or scale > 1:
            raise ValueError(f"scale must be in (0, 1], got {scale}")

        scale_filter = (
            "scale="
            f"trunc(iw*{scale}/2)*2:"
            f"trunc(ih*{scale}/2)*2"
        )

    if output_video.exists() and not overwrite:
        print(f"Output video already exists, skipping ffmpeg: {output_video}", flush=True)
    else:
        cmd = ["ffmpeg"]

        if overwrite:
            cmd.append("-y")
        else:
            cmd.append("-n")

        cmd.extend(
            [
                "-i",
                str(video_path),
                "-vf",
                scale_filter,
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-preset",
                preset,
                "-pix_fmt",
                "yuv420p",
            ]
        )

        if threads is not None and threads >= 0:
            cmd.extend(["-threads", str(threads)])

        if keep_audio:
            cmd.extend(["-c:a", "aac"])
        else:
            cmd.append("-an")

        cmd.append(str(output_video))

        run_ffmpeg(cmd)

    if downsample_csv:
        downsample_annotation_csv_for_video(
            video_path=video_path,
            scale=scale,
            raw_video_root=raw_video_root,
            analysis_root=analysis_root,
            csv_output_root=csv_output_dir,
            overwrite=overwrite,
            require_csv=require_csv,
        )

    return output_video


def main() -> None:
    args = parse_args()

    output_video = downsample_one_video_job(
        video_path=args.video_path,
        output_dir=args.output_dir,
        raw_video_root=args.raw_video_root,
        scale=args.scale,
        width=args.width,
        height=args.height,
        crf=args.crf,
        preset=args.preset,
        threads=args.threads,
        overwrite=args.overwrite,
        keep_audio=args.keep_audio,
        downsample_csv=args.downsample_csv,
        analysis_root=args.analysis_root,
        csv_output_dir=args.csv_output_dir,
        require_csv=args.require_csv,
    )

    print(f"\nDone: {output_video}", flush=True)


if __name__ == "__main__":
    main()
