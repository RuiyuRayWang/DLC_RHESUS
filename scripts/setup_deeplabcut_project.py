#!/usr/bin/env python3
"""
Convert manual labels to DeepLabCut format and create DLC project.

This script:
1. Reads your manual label CSVs
2. Extracts labeled frames as images
3. Creates a DeepLabCut project with proper structure
4. Converts labels to DLC's HDF5 format
5. Prepares the project for training

Usage:
    conda install -c conda-forge deeplabcut
    python scripts/setup_deeplabcut_project.py

After running this, you can train with:
    import deeplabcut
    deeplabcut.train_network(config_path)
"""

import argparse
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import yaml
import shutil
import json
from datetime import datetime
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = PROJECT_ROOT / 'results' / 'oft_analysis' / 'downsampled'
DLC_PROJECT_DIR = PROJECT_ROOT / 'deeplabcut_projects'

# Video directories (search in order of priority)
VIDEO_DIRS = [
    Path('/media/luolab/MyBook2/oft_rhp/videos_downsampled/H1705011'),  # External drive (fallback)
    # Path('/media/luolab/memcore/oft_rhp/videos_downsampled/H1705011'),  # Local fast storage (primary)
    # PROJECT_ROOT / 'data' / 'oft_rhp' / 'videos_downsampled' / 'H1705011',               # Project data (fallback)
]

# DeepLabCut project settings
PROJECT_NAME = f'monkey_oft_tracking'
EXPERIMENTER = 'luolab'
BODYPART = 'head'  # single point: monkey head

# Training parameters
TRAIN_FRACTION = 0.95  # use 95% for training, 5% for testing
TARGET_TOTAL_FRAMES = 7000  # global cap for extracted labeled frames
MAX_FRAMES_PER_VIDEO = 600  # secondary guardrail per video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("camera", type=str, help="Camera identifier",)
    return parser.parse_args()


def load_label_config(camera):
    """Load selected label files from config."""
    config_path = PROJECT_ROOT / f'deeplabcut_label_config_{camera}.json'
    
    if not config_path.exists():
        print("❌ No label file configuration found!")
        print("\nPlease run the file selector first:")
        print("   python scripts/select_label_files.py")
        print("\nThis will open a GUI to select which CSV files contain labels.")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        selected_files = config.get('selected_files', [])
        if not selected_files:
            print("❌ No files selected in config!")
            print("\nPlease run the file selector:")
            print("   python scripts/select_label_files.py")
            return None
        
        # Convert to Path objects and verify they exist
        csv_files = []
        for file_path in selected_files:
            p = Path(file_path)
            if p.exists():
                csv_files.append(p)
            else:
                print(f"⚠️  File not found (skipping): {p}")
        
        return csv_files
    
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None


def find_label_csvs():
    """Find all manual label CSV files (legacy - not used anymore)."""
    csvs = list(LABEL_DIR.rglob('*.csv'))
    # Filter out trajectory CSVs (these are from fusion, not manual labels)
    csvs = [c for c in csvs if 'trajectory' not in c.name and 'manual_labels' not in c.name]
    return csvs


def load_labels(csv_path):
    """Load labels from manual_labeler CSV."""
    try:
        df = pd.read_csv(csv_path)
        
        # Check for required columns
        required_cols = {'video_file', 'centroid_x', 'centroid_y', 'monkey_visible'}
        missing_cols = required_cols - set(df.columns)
        
        if missing_cols:
            print(f"   ⚠️  Skipping (missing columns: {missing_cols})")
            return None
        
        # Filter to only visible monkey frames
        df = df[df['monkey_visible'] == True].copy()

        # Visibility-only pass can mark frames without head coordinates yet.
        # DLC localization training must exclude those rows.
        before_coords = len(df)
        df = df.dropna(subset=['centroid_x', 'centroid_y']).copy()
        df = df[(df['centroid_x'] != -1) & (df['centroid_y'] != -1)].copy()
        skipped_missing_coords = before_coords - len(df)
        
        # Check if any frames remain after filtering
        if len(df) == 0:
            print(f"   ⚠️  Skipping (no visible frames)")
            return None

        if skipped_missing_coords > 0:
            print(f"   ℹ️  Skipped {skipped_missing_coords} visible frames without head coordinates")
        
        return df
    
    except Exception as e:
        print(f"   ⚠️  Error reading file: {e}")
        return None


def allocate_frame_budgets(video_items, target_total_frames):
    """Allocate per-video frame budgets with a global cap."""
    if not video_items:
        return []

    total_visible = sum(len(item['df']) for item in video_items)
    if total_visible <= 0:
        for item in video_items:
            item['budget'] = 0
        return video_items

    # If dataset is already smaller than target, keep all frames.
    if total_visible <= target_total_frames:
        for item in video_items:
            item['budget'] = len(item['df'])
        return video_items

    # Proportional allocation with at least 1 frame per video.
    base = []
    for item in video_items:
        proportional = int(round(len(item['df']) / total_visible * target_total_frames))
        budget = max(1, min(len(item['df']), proportional, MAX_FRAMES_PER_VIDEO))
        item['budget'] = budget
        base.append(item)

    # Adjust to match target_total_frames after rounding and per-video clipping.
    current = sum(item['budget'] for item in base)
    if current < target_total_frames:
        # Add frames to videos with remaining capacity.
        for item in sorted(base, key=lambda x: len(x['df']) - x['budget'], reverse=True):
            if current >= target_total_frames:
                break
            capacity = min(len(item['df']), MAX_FRAMES_PER_VIDEO) - item['budget']
            if capacity <= 0:
                continue
            add = min(capacity, target_total_frames - current)
            item['budget'] += add
            current += add
    elif current > target_total_frames:
        # Remove frames from videos with the largest budgets first.
        for item in sorted(base, key=lambda x: x['budget'], reverse=True):
            if current <= target_total_frames:
                break
            removable = item['budget'] - 1
            if removable <= 0:
                continue
            drop = min(removable, current - target_total_frames)
            item['budget'] -= drop
            current -= drop

    return base


# ── Extract labeled frames ─────────────────────────────────────────────────────

def extract_labeled_frames(df, video_path, output_dir, max_frames=None):
    """
    Extract frames from video at labeled frame numbers.
    
    HEVC codec requires sequential reading - cannot seek directly.
    
    Returns:
        list of (image_path, frame_num, x, y) tuples
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️  Could not open video: {video_path}")
        return []
    
    total_frames = len(df)
    if max_frames and total_frames > max_frames:
        # Sample uniformly to stay under limit
        df = df.sample(n=max_frames, random_state=42)
        print(f"   Sampling {max_frames}/{total_frames} frames")
    
    # Sort by frame number for sequential reading
    df = df.sort_values('video_frame').reset_index(drop=True)
    
    # Create set of target frame numbers for fast lookup
    target_frames = set(df['video_frame'].astype(int))
    
    # Create mapping from frame_num to (x, y) coordinates
    frame_coords = {}
    for _, row in df.iterrows():
        frame_num = int(row['video_frame'])
        frame_coords[frame_num] = (int(row['centroid_x']), int(row['centroid_y']))
    
    # Read frames sequentially
    extracted = []
    current_frame = 0
    max_target_frame = max(target_frames)
    
    # Progress bar
    pbar = tqdm(total=max_target_frame + 1, desc=f"   Reading {video_path.stem}", 
                unit="frame", leave=False)
    
    while current_frame <= max_target_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame if it's a labeled one
        if current_frame in target_frames:
            x, y = frame_coords[current_frame]
            
            img_name = f"{video_path.stem}_frame{current_frame:06d}.png"
            img_path = output_dir / img_name
            cv2.imwrite(str(img_path), frame)
            
            extracted.append((img_path, current_frame, x, y))
            pbar.set_postfix({'extracted': len(extracted)})
        
        current_frame += 1
        pbar.update(1)
    
    pbar.close()
    
    cap.release()
    
    if len(extracted) < len(target_frames):
        print(f"   ⚠️  Warning: Only extracted {len(extracted)}/{len(target_frames)} frames")
    
    return extracted


# ── Create DeepLabCut project ──────────────────────────────────────────────────

def create_dlc_project(camera):
    """Create DeepLabCut project directory structure."""
    
    # Create project directory
    date_str = datetime.now().strftime('%Y-%m-%d')
    project_dir = DLC_PROJECT_DIR / f"{PROJECT_NAME}_{camera}_downsampled-{EXPERIMENTER}-{date_str}"
    
    if project_dir.exists():
        print(f"⚠️  Project already exists: {project_dir}")
        response = input("   Overwrite? (y/n): ")
        if response.lower() != 'y':
            return None
        shutil.rmtree(project_dir)
    
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (project_dir / 'labeled-data').mkdir()
    (project_dir / 'videos').mkdir()
    (project_dir / 'training-datasets').mkdir()
    
    return project_dir


def create_config_yaml(project_dir, video_paths):
    """Create DeepLabCut config.yaml file."""

    video_sets = {}
    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video to read dimensions: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid video dimensions for {video_path}: {width}x{height}")

        video_sets[str(video_path)] = {'crop': f'0, {width}, 0, {height}'}
    
    config = {
        'Task': PROJECT_NAME,
        'scorer': EXPERIMENTER,
        'date': datetime.now().strftime('%b%d'),
        'project_path': str(project_dir),
        'engine': 'pytorch',
        
        # Single bodypart: monkey head
        'bodyparts': [BODYPART],
        'skeleton': [],  # no skeleton connections for single point
        
        # Video settings
        'video_sets': video_sets,
        
        # Training parameters
        'TrainingFraction': [TRAIN_FRACTION],
        'iteration': 0,
        'default_net_type': 'resnet_50',
        'snapshotindex': -1,
        'batch_size': 16,
        
        # Data augmentation
        'multianimalproject': False,
        'dotsize': 12,
        'alphavalue': 0.7,
        'colormap': 'rainbow',
        
        # Network config
        'pos_dist_thresh': 17,
        'global_scale': 0.8,
        'scale_jitter_lo': 0.5,
        'scale_jitter_up': 1.25,
    }
    
    config_path = project_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path


def build_dlc_dataframe(extracted_frames, include_folder_in_index=True):
    """Build DLC-format dataframe from extracted frame tuples."""
    records = []
    for img_path, frame_num, x, y in extracted_frames:
        if include_folder_in_index:
            image_idx = f"labeled-data/{img_path.parent.name}/{img_path.name}"
        else:
            image_idx = img_path.name
        records.append({
            'image': image_idx,
            f'{EXPERIMENTER}_{BODYPART}_x': x,
            f'{EXPERIMENTER}_{BODYPART}_y': y,
            '_folder': img_path.parent.name,
        })

    df = pd.DataFrame(records)
    df = df.set_index('image')

    # Create MultiIndex columns (scorer, bodypart, coords)
    columns = pd.MultiIndex.from_tuples([
        (EXPERIMENTER, BODYPART, 'x'),
        (EXPERIMENTER, BODYPART, 'y'),
    ], names=['scorer', 'bodyparts', 'coords'])

    value_df = df[[f'{EXPERIMENTER}_{BODYPART}_x', f'{EXPERIMENTER}_{BODYPART}_y']].copy()
    value_df.columns = columns
    return value_df, df['_folder']


def ensure_multiindex_rows(df, folder_series):
    """Ensure row index is DLC-compatible: ('labeled-data', 'folder/image')."""
    image_names = [Path(str(idx)).name for idx in df.index]
    folder_vals = [str(v) for v in folder_series]
    level0 = ['labeled-data'] * len(image_names)
    level1 = [f"{folder}/{img}" for folder, img in zip(folder_vals, image_names)]
    mi = pd.MultiIndex.from_arrays([level0, level1], names=['set', 'image'])
    out = df.copy()
    out.index = mi
    return out


def convert_to_dlc_format(extracted_frames, project_dir, output_dir):
    """
    Convert manual labels to DeepLabCut HDF5 format.
    
    DLC expects a pandas DataFrame with MultiIndex columns:
    - Level 0: scorer name
    - Level 1: bodypart name  
    - Level 2: coordinate ('x' or 'y')
    
    Index: relative image paths
    """
    
    # Combined table for inspection and tooling
    df, folders = build_dlc_dataframe(extracted_frames, include_folder_in_index=True)

    # Use MultiIndex rows for compatibility with DLC dataset builder.
    combined_df = ensure_multiindex_rows(df, folders)

    h5_path = output_dir / f'CollectedData_{EXPERIMENTER}.h5'
    combined_df.to_hdf(h5_path, key='df_with_missing', mode='w')

    csv_path = output_dir / f'CollectedData_{EXPERIMENTER}.csv'
    combined_df.to_csv(csv_path)

    # Per-folder annotation files expected by deeplabcut.create_training_dataset
    folder_count = 0
    for folder in sorted(folders.unique()):
        folder_mask = folders == folder
        folder_df = df.loc[folder_mask].copy()
        folder_folders = folders.loc[folder_mask]
        folder_df = ensure_multiindex_rows(folder_df, folder_folders)

        folder_dir = project_dir / 'labeled-data' / folder
        folder_h5 = folder_dir / f'CollectedData_{EXPERIMENTER}.h5'
        folder_csv = folder_dir / f'CollectedData_{EXPERIMENTER}.csv'

        folder_df.to_hdf(folder_h5, key='df_with_missing', mode='w')
        folder_df.to_csv(folder_csv)
        folder_count += 1

    print(f"✅  Converted {len(df)} labels → {h5_path}")
    print(f"✅  Wrote per-folder annotation files in {folder_count} labeled-data folders")
    return h5_path


# ── Main workflow ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    
    print("=" * 70)
    print("DeepLabCut Project Setup from Manual Labels")
    print("=" * 70)
    
    # 1. Load selected CSV files from config
    print("\n1. Loading selected label files from config...")
    csv_files = load_label_config(args.camera)
    
    if csv_files is None:
        return
    
    print(f"   Found {len(csv_files)} selected label files")
    for csv in csv_files:
        print(f"   - {csv.relative_to(PROJECT_ROOT)}")
    
    if not csv_files:
        print("❌ No label files found!")
        return
    
    # 2. Create DLC project
    print("\n2. Creating DeepLabCut project...")
    project_dir = create_dlc_project(args.camera)
    if not project_dir:
        print("❌ Project creation cancelled")
        return
    print(f"   Project: {project_dir}")
    
    # 3. Prepare per-video label pools and global frame budgets
    print("\n3. Preparing frame pools and global sampling budgets...")
    video_items = []
    for i, csv_path in enumerate(csv_files, 1):
        print(f"\n   [{i}/{len(csv_files)}] Processing: {csv_path.name}")
        df = load_labels(csv_path)
        if df is None or len(df) == 0:
            continue

        print(f"   - {len(df)} visible monkey frames")

        video_file = csv_path.stem + '.mp4'
        date = csv_path.parent.stem
        video_path = VIDEO_DIRS[0] / date / video_file
        
        # video_file = df['video_file'].iloc[0]
        # video_path = None
        # for video_dir in VIDEO_DIRS:
        #     if not video_dir.exists():
        #         continue

        #     direct_path = video_dir / video_file
        #     if direct_path.exists():
        #         video_path = direct_path
        #         break

        #     for vp in video_dir.rglob(video_file):
        #         video_path = vp
        #         break

        #     if video_path:
        #         break

        if not video_path or not video_path.exists():
            print(f"   ⚠️  Video not found: {video_file}")
            print(f"      Searched in: {[str(d) for d in VIDEO_DIRS if d.exists()]}")
            continue

        video_items.append({'csv_path': csv_path, 'df': df, 'video_path': video_path})

    if not video_items:
        print("❌ No valid videos with labels found!")
        return

    total_visible = sum(len(item['df']) for item in video_items)
    video_items = allocate_frame_budgets(video_items, TARGET_TOTAL_FRAMES)
    planned_total = sum(item['budget'] for item in video_items)

    print("\n   Sampling plan:")
    print(f"   - Total visible labeled frames: {total_visible}")
    print(f"   - Target total frames: {TARGET_TOTAL_FRAMES}")
    print(f"   - Planned extracted frames: {planned_total}")
    print(f"   - Per-video cap: {MAX_FRAMES_PER_VIDEO}")

    # 4. Extract frames according to global budgets
    print("\n4. Extracting labeled frames...")
    print("   ⏱️  This may take 10-30 minutes for HEVC videos (sequential reading required)")

    all_extracted = []
    video_paths = []

    import time
    start_time = time.time()

    for i, item in enumerate(video_items, 1):
        video_path = item['video_path']
        df = item['df']
        budget = item['budget']

        video_paths.append(video_path)
        print(f"\n   [{i}/{len(video_items)}] {video_path.name}")
        print(f"   - Visible frames: {len(df)} | Budget: {budget}")

        video_label_dir = project_dir / 'labeled-data' / video_path.stem
        extracted = extract_labeled_frames(df, video_path, video_label_dir, max_frames=budget)
        all_extracted.extend(extracted)
        print(f"   ✅ Extracted {len(extracted)} frames → {video_label_dir.name}")

        elapsed = time.time() - start_time
        avg_time = elapsed / i
        remaining = avg_time * (len(video_items) - i)
        print(f"   ⏱️  Elapsed: {elapsed/60:.1f}m | Est. remaining: {remaining/60:.1f}m")
    
    elapsed_total = time.time() - start_time
    
    if not all_extracted:
        print("❌ No frames extracted!")
        return
    
    print(f"\n   ✅ Total frames extracted: {len(all_extracted)} in {elapsed_total/60:.1f} minutes")
    
    # 5. Create config.yaml
    print("\n5. Creating config.yaml...")
    config_path = create_config_yaml(project_dir, video_paths)
    print(f"   Config: {config_path}")
    
    # 6. Convert labels to DLC format
    print("\n6. Converting labels to DeepLabCut format...")
    training_dir = project_dir / 'training-datasets' / 'iteration-0'
    training_dir.mkdir(parents=True, exist_ok=True)
    h5_path = convert_to_dlc_format(all_extracted, project_dir, training_dir)
    
    # 7. Summary and next steps
    print("\n" + "=" * 70)
    print("✅ DeepLabCut Project Setup Complete!")
    print("=" * 70)
    print(f"\nProject directory: {project_dir}")
    print(f"Config file: {config_path}")
    print(f"Labeled frames: {len(all_extracted)}")
    print(f"Videos: {len(video_paths)}")
    
    print("\n" + "-" * 70)
    print("Next steps:")
    print("-" * 70)
    print("\n1. Install DeepLabCut (if not already installed):")
    print("   conda install -c conda-forge deeplabcut")
    print("\n2. Create training dataset:")
    print("   import deeplabcut")
    print(f"   config = r'{config_path}'")
    print("   deeplabcut.create_training_dataset(config)")
    print("\n3. Train the network:")
    print("   deeplabcut.train_network(config, shuffle=1)")
    print("\n4. Evaluate the network:")
    print("   deeplabcut.evaluate_network(config)")
    print("\n5. Analyze new videos:")
    print("   deeplabcut.analyze_videos(config, ['/path/to/new/video.mp4'])")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
