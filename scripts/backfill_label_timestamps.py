#!/usr/bin/env python3
"""
Backfill missing label CSV metadata from label_time_anchors.csv.

This repairs legacy per-video label CSVs that are missing `video_file` and/or
`video_timestamp`. The anchor table is treated as authoritative. The repaired
`video_timestamp` is written per row as:

    anchor_timestamp + (video_frame - anchor_video_frame) / fps

so at 20 FPS, every 20 frames advances the timestamp by 1 second.

Usage:
    conda activate video_tracking
    python scripts/backfill_label_timestamps.py \
        --session-dir results/oft_analysis/H1705011/2026-05-25

    python scripts/backfill_label_timestamps.py \
        --session-dir results/oft_analysis/H1705011/2026-05-25 \
        --dry-run
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FPS = 20.0
TIMESTAMP_FMT = '%Y-%m-%d %H:%M:%S'
LABEL_CSV_RE = re.compile(r'^(labels_)?D0[1-4].*\.csv$', re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Backfill missing video_file/video_timestamp columns from time anchors.'
    )
    parser.add_argument(
        '--session-dir',
        required=True,
        help='Session directory containing label CSVs and label_time_anchors.csv',
    )
    parser.add_argument(
        '--anchors',
        default=None,
        help='Optional path to label_time_anchors.csv (default: <session-dir>/label_time_anchors.csv)',
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=DEFAULT_FPS,
        help='Video frame rate used for frame/time conversion (default: 20)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Report planned changes without writing files',
    )
    parser.add_argument(
        '--overwrite-existing',
        action='store_true',
        help='Rewrite existing non-empty metadata columns from anchors too',
    )
    return parser.parse_args()


def resolve_session_dir(session_dir_arg: str) -> Path:
    session_dir = Path(session_dir_arg)
    if not session_dir.is_absolute():
        session_dir = PROJECT_ROOT / session_dir
    return session_dir.resolve()


def normalize_anchor_table(anchors_path: Path) -> dict[str, dict[str, object]]:
    anchors_df = pd.read_csv(anchors_path)
    required = {'csv_file', 'anchor_timestamp'}
    missing = required.difference(anchors_df.columns)
    if missing:
        raise ValueError(f'{anchors_path} is missing required columns: {sorted(missing)}')

    anchors: dict[str, dict[str, object]] = {}
    for _, row in anchors_df.iterrows():
        csv_name = str(row['csv_file']).strip()
        if not csv_name:
            continue

        anchor_timestamp = pd.to_datetime(row['anchor_timestamp'])
        if pd.isna(anchor_timestamp):
            raise ValueError(f'Anchor timestamp is blank for {csv_name}')

        video_file = ''
        if 'video_file' in anchors_df.columns and pd.notna(row.get('video_file')):
            video_file = str(row['video_file']).strip()
        if not video_file:
            video_file = Path(csv_name).with_suffix('.mp4').name

        anchor_frame = 0
        if 'anchor_video_frame' in anchors_df.columns and pd.notna(row.get('anchor_video_frame')):
            anchor_frame = int(row['anchor_video_frame'])

        anchors[csv_name] = {
            'video_file': video_file,
            'anchor_timestamp': anchor_timestamp,
            'anchor_video_frame': anchor_frame,
        }
    return anchors


def is_label_csv(path: Path) -> bool:
    return LABEL_CSV_RE.match(path.name) is not None


def is_missing_series(series: pd.Series) -> pd.Series:
    normalized = series.astype('string').str.strip().str.lower()
    return series.isna() | normalized.isin(['', 'unknown', 'nan', 'nat'])


def build_video_timestamp(df: pd.DataFrame, anchor_timestamp: pd.Timestamp, anchor_frame: int, fps: float) -> pd.Series:
    if 'video_frame' not in df.columns:
        raise ValueError('Missing required column: video_frame')

    offsets = (df['video_frame'].astype(float) - float(anchor_frame)) / float(fps)
    repaired = anchor_timestamp + pd.to_timedelta(offsets, unit='s')
    return repaired.dt.strftime(TIMESTAMP_FMT)


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        'camera',
        'video_file',
        'video_timestamp',
        'video_frame',
        'time_sec',
        'centroid_x',
        'centroid_y',
        'monkey_visible',
    ]
    ordered = [col for col in preferred if col in df.columns]
    remainder = [col for col in df.columns if col not in ordered]
    return df[ordered + remainder]


def process_file(csv_path: Path, anchor: dict[str, object], fps: float, dry_run: bool, overwrite_existing: bool) -> dict[str, object]:
    df = pd.read_csv(csv_path)
    original_columns = list(df.columns)

    if 'video_file' not in df.columns:
        df.insert(1 if 'camera' in df.columns else 0, 'video_file', pd.NA)
    if 'video_timestamp' not in df.columns:
        insert_at = df.columns.get_loc('video_file') + 1 if 'video_file' in df.columns else 1
        df.insert(insert_at, 'video_timestamp', pd.NA)

    fill_video_file = overwrite_existing or is_missing_series(df['video_file']).any()
    fill_video_timestamp = overwrite_existing or is_missing_series(df['video_timestamp']).any()

    changed = False
    if fill_video_file:
        if overwrite_existing:
            df['video_file'] = str(anchor['video_file'])
        else:
            mask = is_missing_series(df['video_file'])
            df.loc[mask, 'video_file'] = str(anchor['video_file'])
        changed = True

    if fill_video_timestamp:
        repaired_timestamps = build_video_timestamp(
            df,
            anchor_timestamp=pd.Timestamp(anchor['anchor_timestamp']),
            anchor_frame=int(anchor['anchor_video_frame']),
            fps=fps,
        )
        if overwrite_existing:
            df['video_timestamp'] = repaired_timestamps
        else:
            mask = is_missing_series(df['video_timestamp'])
            df.loc[mask, 'video_timestamp'] = repaired_timestamps[mask]
        changed = True

    df = reorder_columns(df)

    if changed and not dry_run:
        df.to_csv(csv_path, index=False)

    return {
        'file': csv_path.name,
        'changed': changed,
        'columns_before': original_columns,
        'columns_after': list(df.columns),
        'row_count': len(df),
        'anchor_timestamp': pd.Timestamp(anchor['anchor_timestamp']).strftime(TIMESTAMP_FMT),
        'first_row_timestamp': str(df['video_timestamp'].iloc[0]) if 'video_timestamp' in df.columns and not df.empty else '',
        'last_row_timestamp': str(df['video_timestamp'].iloc[-1]) if 'video_timestamp' in df.columns and not df.empty else '',
    }


def main() -> int:
    args = parse_args()
    session_dir = resolve_session_dir(args.session_dir)
    anchors_path = Path(args.anchors).resolve() if args.anchors else session_dir / 'label_time_anchors.csv'

    if not session_dir.exists():
        raise FileNotFoundError(f'Session directory not found: {session_dir}')
    if not anchors_path.exists():
        raise FileNotFoundError(f'Anchor file not found: {anchors_path}')

    anchors = normalize_anchor_table(anchors_path)
    changed_count = 0
    skipped_count = 0

    print(f'Session: {session_dir}')
    print(f'Anchors: {anchors_path}')
    print(f'FPS: {args.fps:g}')
    if args.dry_run:
        print('Mode: dry-run')
    elif args.overwrite_existing:
        print('Mode: overwrite-existing')

    for csv_name, anchor in sorted(anchors.items()):
        csv_path = session_dir / csv_name
        if not csv_path.exists():
            print(f'SKIP  missing file: {csv_name}')
            skipped_count += 1
            continue
        if not is_label_csv(csv_path):
            print(f'SKIP  non-label file: {csv_name}')
            skipped_count += 1
            continue

        result = process_file(
            csv_path=csv_path,
            anchor=anchor,
            fps=args.fps,
            dry_run=args.dry_run,
            overwrite_existing=args.overwrite_existing,
        )
        if result['changed']:
            changed_count += 1
            print(
                f"FIX   {result['file']}: rows={result['row_count']} anchor={result['anchor_timestamp']} "
                f"first={result['first_row_timestamp']} last={result['last_row_timestamp']}"
            )
        else:
            print(f"OK    {result['file']}: already has metadata")

    print(f'\nSummary: changed={changed_count} skipped={skipped_count} total_anchors={len(anchors)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())