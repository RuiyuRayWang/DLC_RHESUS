#!/usr/bin/env python3
"""
Split a combined DeepLabCut CollectedData H5 into per-folder annotation files.

This repairs projects where only training-datasets/iteration-0/CollectedData_*.h5
exists, but labeled-data/<video_stem>/CollectedData_*.h5 files are missing.

Usage:
  python scripts/split_combined_h5_to_labeled_folders.py
  python scripts/split_combined_h5_to_labeled_folders.py \
      --project /home/luolab/PROJECTS/PAL_ENHANCER/deeplabcut_projects/monkey_oft_tracking-luolab-2026-06-17
"""

from pathlib import Path
import argparse
import pandas as pd

PROJECT_DEFAULT = Path('/home/luolab/PROJECTS/PAL_ENHANCER/deeplabcut_projects/monkey_oft_tracking-luolab-2026-06-17')
SCORER = 'luolab'


def find_combined_h5(project_dir: Path, scorer: str) -> Path:
    candidates = [
        project_dir / 'training-datasets' / 'iteration-0' / f'CollectedData_{scorer}.h5',
        project_dir / f'CollectedData_{scorer}.h5',
    ]
    for c in candidates:
        if c.exists():
            return c

    matches = sorted(project_dir.rglob(f'CollectedData_{scorer}.h5'))
    if matches:
        # Prefer file under training-datasets if present
        for m in matches:
            if 'training-datasets' in m.parts:
                return m
        return matches[0]

    raise FileNotFoundError(f'No combined CollectedData_{scorer}.h5 found under {project_dir}')


def parse_folder_from_index(idx: str) -> str:
    p = Path(str(idx))
    # Expected format: labeled-data/<folder>/<image>
    if len(p.parts) >= 3 and p.parts[0] == 'labeled-data':
        return p.parts[1]
    # Fallback: use parent as folder when possible
    if len(p.parts) >= 2:
        return p.parts[-2]
    raise ValueError(f'Cannot parse folder from index: {idx}')


def parse_folder_from_any_index(idx) -> str:
    """Parse folder name from either string index or tuple-style MultiIndex row."""
    if isinstance(idx, tuple) and len(idx) >= 2:
        # Expected tuple form like ('labeled-data', 'D03_xxx/frame.png')
        rel = str(idx[1])
        p = Path(rel)
        if len(p.parts) >= 2:
            return p.parts[0]
    return parse_folder_from_index(str(idx))


def to_multiindex_rows(df: pd.DataFrame, folder_name: str) -> pd.DataFrame:
    """Convert row index to DLC-compatible 2-level MultiIndex."""
    image_names = [Path(str(i)).name for i in df.index]
    level0 = ['labeled-data'] * len(image_names)
    level1 = [f'{folder_name}/{img}' for img in image_names]
    mi = pd.MultiIndex.from_arrays(
        [level0, level1],
        names=['set', 'image'],
    )
    out = df.copy()
    out.index = mi
    return out


def main():
    parser = argparse.ArgumentParser(description='Split combined DLC H5 into per-folder annotation files.')
    parser.add_argument('--project', type=Path, default=PROJECT_DEFAULT, help='Path to DLC project directory')
    parser.add_argument('--scorer', type=str, default=SCORER, help='DLC scorer name, default luolab')
    args = parser.parse_args()

    project_dir = args.project.resolve()
    scorer = args.scorer

    if not project_dir.exists():
        raise FileNotFoundError(f'Project directory not found: {project_dir}')

    combined_h5 = find_combined_h5(project_dir, scorer)
    print(f'Using combined H5: {combined_h5}')

    df = pd.read_hdf(combined_h5)
    if len(df) == 0:
        raise RuntimeError('Combined H5 is empty')

    folder_map = {}
    for idx in df.index:
        folder = parse_folder_from_any_index(idx)
        folder_map.setdefault(folder, []).append(idx)

    written = 0
    for folder, indices in sorted(folder_map.items()):
        folder_dir = project_dir / 'labeled-data' / folder
        if not folder_dir.exists():
            print(f'WARNING: folder missing, skipping: {folder_dir}')
            continue

        out_df = df.loc[indices].copy()
        out_df = to_multiindex_rows(out_df, folder)

        out_h5 = folder_dir / f'CollectedData_{scorer}.h5'
        out_csv = folder_dir / f'CollectedData_{scorer}.csv'

        out_df.to_hdf(out_h5, key='df_with_missing', mode='w')
        out_df.to_csv(out_csv)
        written += 1

    print(f'Wrote per-folder annotation files for {written} folders')


if __name__ == '__main__':
    main()
