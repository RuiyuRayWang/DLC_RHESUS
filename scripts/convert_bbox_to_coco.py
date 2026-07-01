#!/usr/bin/env python3
"""
Convert bbox CSV labels to COCO format with spatial sampling for balanced dataset.
Handles dataset bias by sampling frames uniformly from spatial regions.
"""

import os
import json
import csv
import cv2
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

class CocoDatasetBuilder:
    def __init__(self, output_dir, image_width=1920, image_height=1080, n_spatial_regions=4):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_width = image_width
        self.image_height = image_height
        self.n_spatial_regions = n_spatial_regions
        
        # COCO format
        self.images = []
        self.annotations = []
        self.categories = [{"id": 1, "name": "monkey"}]
        self.image_id = 1
        self.annotation_id = 1
        
        # Track spatial distribution
        self.spatial_centers = None
        self.frame_to_region = {}
        
    def _get_spatial_region(self, bbox_center):
        """Assign bbox to nearest spatial region using KMeans."""
        if self.spatial_centers is None:
            return 0
        distances = np.linalg.norm(self.spatial_centers - bbox_center, axis=1)
        return int(np.argmin(distances))
    
    def add_frame_from_video(self, video_path, frame_num, bbox_data, camera_id, timestamp):
        """Extract frame from video and add to dataset."""
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return False
        
        # Save frame
        frame_name = f"{camera_id}_{timestamp}_{frame_num:06d}.jpg"
        frame_path = self.output_dir / "images" / frame_name
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(frame_path), frame)
        
        # Add image entry
        image_entry = {
            "id": self.image_id,
            "file_name": f"images/{frame_name}",
            "height": self.image_height,
            "width": self.image_width
        }
        self.images.append(image_entry)
        
        # Add annotation
        x1, y1, x2, y2 = bbox_data
        width = x2 - x1
        height = y2 - y1
        
        annotation = {
            "id": self.annotation_id,
            "image_id": self.image_id,
            "category_id": 1,
            "bbox": [x1, y1, width, height],
            "area": width * height,
            "iscrowd": 0
        }
        self.annotations.append(annotation)
        
        # Track spatial region
        bbox_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        region = self._get_spatial_region(bbox_center)
        self.frame_to_region[(self.image_id, video_path.name)] = region
        
        self.image_id += 1
        self.annotation_id += 1
        return True
    
    def load_csv_labels(self, csv_path):
        """Load bbox labels from CSV."""
        frames = defaultdict(list)
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['monkey_visible'] == 'True':
                    key = (row['video_file'], int(row['video_frame']))
                    bbox = (
                        float(row['bbox_x1']),
                        float(row['bbox_y1']),
                        float(row['bbox_x2']),
                        float(row['bbox_y2'])
                    )
                    frames[key].append({
                        'bbox': bbox,
                        'timestamp': row['video_timestamp'],
                        'time_sec': float(row['time_sec'])
                    })
        return frames
    
    def sample_frames_balanced(self, frames_dict, video_dir, camera_id, n_samples=None, sample_every_n_frames=1):
        """
        Sample frames with spatial balance to handle bbox concentration.
        
        Strategy:
        1. Cluster all bbox centers into spatial regions
        2. Sample frames proportionally from each region
        3. Skip consecutive frames to avoid temporal bias
        """
        # Collect all bbox centers for clustering
        centers = []
        frame_list = []
        
        for (video_file, frame_num), data_list in frames_dict.items():
            for data in data_list:
                bbox = data['bbox']
                center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                centers.append(center)
                frame_list.append((video_file, frame_num, data))
        
        if len(centers) == 0:
            return
        
        # K-means clustering for spatial regions
        centers = np.array(centers)
        kmeans = KMeans(n_clusters=min(self.n_spatial_regions, len(centers)), random_state=42, n_init=10)
        labels = kmeans.fit_predict(centers)
        self.spatial_centers = kmeans.cluster_centers_
        
        # Sample frames from each region
        regions = defaultdict(list)
        for idx, label in enumerate(labels):
            regions[label].append((frame_list[idx], idx))
        
        if n_samples is None:
            n_samples = len(frame_list)
        
        samples_per_region = max(1, n_samples // self.n_spatial_regions)
        
        print(f"[{camera_id}] Sampling {n_samples} frames from {len(frame_list)} candidates")
        print(f"[{camera_id}] Spatial regions: {self.n_spatial_regions}, samples/region: {samples_per_region}")
        
        sampled = 0
        for region_id in sorted(regions.keys()):
            region_frames = regions[region_id]
            print(f"[{camera_id}] Region {region_id}: {len(region_frames)} frames -> sample {samples_per_region}")
            
            # Sort by frame number for temporal sampling
            region_frames.sort(key=lambda x: x[0][1])
            
            # Sample every n-th frame
            step = max(1, len(region_frames) // samples_per_region)
            for i in range(0, len(region_frames), step):
                if sampled >= samples_per_region * self.n_spatial_regions:
                    break
                
                (video_file, frame_num, data), _ = region_frames[i]
                video_path = Path(video_dir) / camera_id / video_file
                
                if video_path.exists():
                    success = self.add_frame_from_video(
                        video_path, 
                        frame_num, 
                        data['bbox'],
                        camera_id,
                        data['timestamp']
                    )
                    if success:
                        sampled += 1
    
    def save_coco_json(self, output_file):
        """Save annotations in COCO format."""
        coco_data = {
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories
        }
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        print(f"Saved {len(self.images)} images with {len(self.annotations)} annotations to {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert bbox CSV to COCO format with spatial sampling")
    parser.add_argument("--csv-dir", required=True, help="Directory containing bbox CSV files")
    parser.add_argument("--video-dir", required=True, help="Directory containing downsampled videos")
    parser.add_argument("--output-dir", required=True, help="Output directory for COCO dataset")
    parser.add_argument("--n-spatial-regions", type=int, default=4, help="Number of spatial regions for balanced sampling")
    parser.add_argument("--samples-per-camera", type=int, default=200, help="Target number of frames per camera")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train/val split ratio")
    
    args = parser.parse_args()
    
    csv_dir = Path(args.csv_dir)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each camera
    cameras = ['D01', 'D02', 'D03', 'D04']
    all_images = []
    all_annotations = []
    image_id_offset = 1
    ann_id_offset = 1
    
    for camera_id in cameras:
        print(f"\n{'='*60}")
        print(f"Processing camera: {camera_id}")
        print(f"{'='*60}")
        
        # Find CSV files for this camera
        csv_files = list(csv_dir.glob(f"*{camera_id}*.csv"))
        if not csv_files:
            print(f"No CSV files found for {camera_id}")
            continue
        
        builder = CocoDatasetBuilder(
            output_dir / camera_id,
            n_spatial_regions=args.n_spatial_regions
        )
        
        # Load and merge all CSV files for this camera
        all_frames = defaultdict(list)
        for csv_file in csv_files:
            print(f"  Loading: {csv_file.name}")
            frames = builder.load_csv_labels(csv_file)
            for key, data_list in frames.items():
                all_frames[key].extend(data_list)
        
        # Sample frames with spatial balance
        builder.sample_frames_balanced(
            all_frames,
            video_dir,
            camera_id,
            n_samples=args.samples_per_camera
        )
        
        # Update IDs and collect data
        for img in builder.images:
            img_copy = img.copy()
            img_copy['id'] = image_id_offset
            img_copy['camera'] = camera_id
            all_images.append(img_copy)
            image_id_offset += 1
        
        for ann in builder.annotations:
            ann_copy = ann.copy()
            ann_copy['id'] = ann_id_offset
            # Update image_id reference
            img_idx = builder.images.index(next(i for i in builder.images if i['id'] == ann['image_id']))
            ann_copy['image_id'] = all_images[-len(builder.images) + img_idx]['id']
            all_annotations.append(ann_copy)
            ann_id_offset += 1
    
    # Create train/val split
    n_images = len(all_images)
    split_idx = int(n_images * args.train_split)
    indices = np.arange(n_images)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_indices = set(indices[:split_idx])
    
    train_images = [all_images[i] for i in range(n_images) if i in train_indices]
    val_images = [all_images[i] for i in range(n_images) if i not in train_indices]
    
    train_image_ids = {img['id'] for img in train_images}
    val_image_ids = {img['id'] for img in val_images}
    
    train_annotations = [ann for ann in all_annotations if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in all_annotations if ann['image_id'] in val_image_ids]
    
    # Save annotations
    train_file = output_dir / "annotations_train.json"
    val_file = output_dir / "annotations_val.json"
    
    train_data = {"images": train_images, "annotations": train_annotations, "categories": [{"id": 1, "name": "monkey"}]}
    val_data = {"images": val_images, "annotations": val_annotations, "categories": [{"id": 1, "name": "monkey"}]}
    
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Dataset Summary")
    print(f"{'='*60}")
    print(f"Total images: {n_images}")
    print(f"Training: {len(train_images)} images, {len(train_annotations)} annotations")
    print(f"Validation: {len(val_images)} images, {len(val_annotations)} annotations")
    print(f"Saved to: {output_dir}")
    print(f"  - {train_file}")
    print(f"  - {val_file}")


if __name__ == "__main__":
    main()
