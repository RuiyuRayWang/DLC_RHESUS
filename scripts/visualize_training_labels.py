#!/usr/bin/env python3
"""
Visualize random samples from the training dataset to verify annotations.

Shows frames with labeled points overlaid, similar to the manual labeler.
"""

import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import random

def find_labeled_data_folders():
    """Find all labeled-data folders in the DLC project."""
    project_root = Path(__file__).resolve().parents[1]
    dlc_projects = project_root / 'deeplabcut_projects'
    
    if not dlc_projects.exists():
        print(f"❌ No deeplabcut_projects folder found")
        return []
    
    # Find the most recent project
    projects = [p for p in dlc_projects.iterdir() if p.is_dir() and not p.name.startswith('.')]
    if not projects:
        print(f"❌ No DLC projects found")
        return []
    
    project = max(projects, key=lambda p: p.stat().st_mtime)
    labeled_data_dir = project / 'labeled-data'
    
    if not labeled_data_dir.exists():
        print(f"❌ No labeled-data folder found in {project}")
        return []
    
    folders = [f for f in labeled_data_dir.iterdir() if f.is_dir()]
    return folders

def load_labels(label_folder):
    """Load labels from a video folder."""
    h5_file = label_folder / f"CollectedData_luolab.h5"
    if not h5_file.exists():
        return None, None
    
    # Load labels
    df = pd.read_hdf(h5_file)
    
    # Get list of labeled frames
    png_files = list(label_folder.glob("*.png"))
    
    return df, png_files

def visualize_sample(img_path, df, bodypart='head'):
    """Visualize a frame with its label."""
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Build the relative path that matches the H5 index format
    # Index format: 'labeled-data/D01_20260527160156/D01_20260527160156_frame000000.png'
    video_name = img_path.parent.name
    frame_name = img_path.name
    rel_path = f"labeled-data/{video_name}/{frame_name}"
    
    # Find coordinates in dataframe
    try:
        # Get the row for this image
        if rel_path in df.index:
            coords = df.loc[rel_path]
            x = coords[('luolab', bodypart, 'x')]
            y = coords[('luolab', bodypart, 'y')]
            
            # Draw circle at labeled point
            if not pd.isna(x) and not pd.isna(y):
                x_int, y_int = int(x), int(y)
                cv2.circle(img, (x_int, y_int), 10, (255, 0, 0), -1)  # Red filled circle
                cv2.circle(img, (x_int, y_int), 12, (255, 255, 0), 2)  # Yellow outline
                
                # Add text label
                label_text = f"({x_int}, {y_int})"
                cv2.putText(img, label_text, (x_int + 15, y_int - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                return img, (x_int, y_int), True
    except Exception as e:
        print(f"   Warning: Could not get label for {rel_path}: {e}")
    
    return img, None, False

def main():
    print("=" * 70)
    print("DeepLabCut Training Data Visualization")
    print("=" * 70)
    
    # Find labeled data folders
    folders = find_labeled_data_folders()
    if not folders:
        print("❌ No labeled data folders found")
        return
    
    print(f"\n✅ Found {len(folders)} labeled video folders")
    
    # Collect all valid samples
    all_samples = []
    print("\n📊 Scanning labeled data...")
    
    for folder in folders:
        df, png_files = load_labels(folder)
        if df is None or not png_files:
            continue
        
        for png_file in png_files:
            all_samples.append((png_file, df))
    
    print(f"   Total frames: {len(all_samples)}")
    
    if not all_samples:
        print("❌ No labeled frames found")
        return
    
    # Sample random frames
    n_samples = min(16, len(all_samples))
    samples = random.sample(all_samples, n_samples)
    
    print(f"\n🎯 Visualizing {n_samples} random samples...")
    
    # Create figure
    cols = 4
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    valid_count = 0
    for idx, (img_path, df) in enumerate(samples):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        img, coords, has_label = visualize_sample(img_path, df)
        
        if img is not None:
            ax.imshow(img)
            if has_label and coords:
                title = f"{img_path.parent.name}\n{img_path.name}\n({coords[0]}, {coords[1]})"
                valid_count += 1
            else:
                title = f"{img_path.parent.name}\n{img_path.name}\n(No label)"
            ax.set_title(title, fontsize=8)
        else:
            ax.text(0.5, 0.5, f"Failed to load\n{img_path.name}",
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(n_samples, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).resolve().parents[1] / 'results'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'training_data_samples.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved visualization to: {output_path}")
    print(f"   {valid_count}/{n_samples} frames have valid labels")
    
    # Show statistics
    print("\n📈 Label Statistics:")
    all_coords = []
    for img_path, df in all_samples[:1000]:  # Sample 1000 for stats
        video_name = img_path.parent.name
        frame_name = img_path.name
        rel_path = f"labeled-data/{video_name}/{frame_name}"
        if rel_path in df.index:
            try:
                coords = df.loc[rel_path]
                x = coords[('luolab', 'head', 'x')]
                y = coords[('luolab', 'head', 'y')]
                if not pd.isna(x) and not pd.isna(y):
                    all_coords.append((x, y))
            except:
                pass
    
    if all_coords:
        all_coords = np.array(all_coords)
        print(f"   X range: {all_coords[:, 0].min():.1f} - {all_coords[:, 0].max():.1f}")
        print(f"   Y range: {all_coords[:, 1].min():.1f} - {all_coords[:, 1].max():.1f}")
        print(f"   X mean ± std: {all_coords[:, 0].mean():.1f} ± {all_coords[:, 0].std():.1f}")
        print(f"   Y mean ± std: {all_coords[:, 1].mean():.1f} ± {all_coords[:, 1].std():.1f}")
    
    # Display
    print("\n💡 Opening visualization...")
    plt.show()

if __name__ == '__main__':
    main()
