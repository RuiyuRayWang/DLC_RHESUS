#!/usr/bin/env python3
"""
Evaluation script for RTMDet predictions against ground truth bbox annotations.
Computes IoU, precision, recall, and generates comparison visualizations.

Usage:
    python scripts/evaluate_rtmdet.py \
        --predictions results/rtmdet_inference/detections_*.csv \
        --ground-truth results/bbox/H1705011/2026-06-16/*.csv \
        --output results/rtmdet_evaluation/ \
        --iou-threshold 0.5
"""

import argparse
import csv
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt


def iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    if xi_max < xi_min or yi_max < yi_min:
        return 0.0
    
    intersection = (xi_max - xi_min) * (yi_max - yi_min)
    
    # Union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def load_detections(csv_files):
    """Load detections from CSV files."""
    detections = defaultdict(list)
    
    for csv_file in csv_files:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_key = f"{Path(csv_file).stem}_{row.get('frame', row.get('time_sec', '0'))}"
                bbox = (
                    float(row.get('bbox_x1', row.get('x1', 0))),
                    float(row.get('bbox_y1', row.get('y1', 0))),
                    float(row.get('bbox_x2', row.get('x2', 0))),
                    float(row.get('bbox_y2', row.get('y2', 0)))
                )
                detections[frame_key].append({
                    'bbox': bbox,
                    'score': float(row.get('score', 1.0)),
                    'source': Path(csv_file).stem
                })
    
    return detections


def evaluate(predictions, ground_truth, iou_threshold=0.5):
    """Evaluate predictions against ground truth."""
    
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    iou_scores = []
    
    # Match predictions to ground truth
    matched_gt = set()
    
    for frame_key in ground_truth.keys():
        gt_boxes = ground_truth[frame_key]
        pred_boxes = predictions.get(frame_key, [])
        
        # Sort predictions by score (descending)
        pred_boxes = sorted(pred_boxes, key=lambda x: x['score'], reverse=True)
        
        # Match predictions to ground truth
        for pred in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                
                box_iou = iou(pred['bbox'], gt['bbox'])
                if box_iou > best_iou:
                    best_iou = box_iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
                iou_scores.append(best_iou)
            else:
                fp += 1
        
        # Unmatched ground truth
        fn += len(gt_boxes) - sum(1 for idx in range(len(gt_boxes)) if idx in matched_gt)
    
    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = np.mean(iou_scores) if iou_scores else 0.0
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_iou': mean_iou,
        'iou_scores': iou_scores
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RTMDet predictions")
    parser.add_argument("--predictions", type=str, required=True, nargs="+",
                       help="Prediction CSV file(s)")
    parser.add_argument("--ground-truth", type=str, required=True, nargs="+",
                       help="Ground truth CSV file(s)")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                       help="IoU threshold for matching (default: 0.5)")
    parser.add_argument("--score-threshold", type=float, default=0.0,
                       help="Minimum prediction score to consider")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization plots")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictions and ground truth
    print("Loading predictions...")
    predictions = load_detections(args.predictions)
    
    print("Loading ground truth...")
    ground_truth = load_detections(args.ground_truth)
    
    # Filter by score threshold
    for key in predictions:
        predictions[key] = [p for p in predictions[key] if p['score'] >= args.score_threshold]
    
    # Evaluate
    print(f"\nEvaluating with IoU threshold: {args.iou_threshold}")
    metrics = evaluate(predictions, ground_truth, iou_threshold=args.iou_threshold)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"True Positives:  {metrics['tp']}")
    print(f"False Positives: {metrics['fp']}")
    print(f"False Negatives: {metrics['fn']}")
    print(f"\nPrecision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"Mean IoU:  {metrics['mean_iou']:.4f}")
    print("="*60 + "\n")
    
    # Save results to JSON
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'tp': int(metrics['tp']),
            'fp': int(metrics['fp']),
            'fn': int(metrics['fn']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'mean_iou': float(metrics['mean_iou']),
            'iou_threshold': args.iou_threshold,
            'score_threshold': args.score_threshold,
            'total_predictions': len([p for preds in predictions.values() for p in preds]),
            'total_ground_truth': len([g for gts in ground_truth.values() for g in gts])
        }, f, indent=2)
    
    print(f"✓ Results saved to {results_file}")
    
    # Visualization
    if args.visualize and metrics['iou_scores']:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # IoU histogram
        axes[0, 0].hist(metrics['iou_scores'], bins=30, edgecolor='black')
        axes[0, 0].set_xlabel('IoU')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'IoU Distribution (mean={metrics["mean_iou"]:.3f})')
        axes[0, 0].axvline(args.iou_threshold, color='r', linestyle='--', label='Threshold')
        axes[0, 0].legend()
        
        # Precision-Recall-F1
        metrics_names = ['Precision', 'Recall', 'F1']
        metrics_values = [metrics['precision'], metrics['recall'], metrics['f1']]
        axes[0, 1].bar(metrics_names, metrics_values)
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Performance Metrics')
        for i, v in enumerate(metrics_values):
            axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # Confusion matrix
        confusion = np.array([
            [metrics['tp'], metrics['fp']],
            [metrics['fn'], 0]
        ])
        im = axes[1, 0].imshow(confusion, cmap='Blues')
        axes[1, 0].set_xticks([0, 1])
        axes[1, 0].set_yticks([0, 1])
        axes[1, 0].set_xticklabels(['Positive', 'Negative'])
        axes[1, 0].set_yticklabels(['True', 'False'])
        axes[1, 0].set_title('Confusion Matrix')
        for i in range(2):
            for j in range(2):
                axes[1, 0].text(j, i, str(confusion[i, j]), ha='center', va='center')
        
        # Detection counts
        axes[1, 1].text(0.1, 0.7, f"Total Ground Truth: {metrics['fn'] + metrics['tp']}", fontsize=12)
        axes[1, 1].text(0.1, 0.6, f"Total Predictions: {metrics['tp'] + metrics['fp']}", fontsize=12)
        axes[1, 1].text(0.1, 0.5, f"True Positives: {metrics['tp']}", fontsize=12, color='green')
        axes[1, 1].text(0.1, 0.4, f"False Positives: {metrics['fp']}", fontsize=12, color='red')
        axes[1, 1].text(0.1, 0.3, f"False Negatives: {metrics['fn']}", fontsize=12, color='orange')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'evaluation_plots.png', dpi=150)
        print(f"✓ Plots saved to evaluation_plots.png")


if __name__ == "__main__":
    main()
