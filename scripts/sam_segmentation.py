"""
SAM-based segmentation module for monkey OFT video analysis.
Uses Segment Anything Model for robust segmentation that suppresses cage artifacts.
"""

import cv2
import numpy as np
from pathlib import Path
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# SAM Configuration
SAM_CONFIG = {
    'model_type': 'vit_b',  # vit_b, vit_l, or vit_h
    'checkpoint_path': '/home/luolab/PROJECTS/PAL_ENHANCER/models/sam/sam_vit_b_01ec64.pth',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Automatic mask generation parameters - VERY PERMISSIVE to generate many candidates
    'points_per_side': 32,  # More sample points
    'pred_iou_thresh': 0.75,  # Very low to catch everything
    'stability_score_thresh': 0.85,  # Very low to allow uncertain masks
    'crop_n_layers': 1,
    'crop_n_points_downscale_factor': 2,
    'min_mask_region_area': 500,  # Very small minimum
    
    # Spatial priors to suppress cage/edges
    'border_margin_frac': 0.05,  # Minimal border rejection
    'center_weight': 0.5,  # Prefer central objects
    'merge_gap_frac': 0.08,  # Conservative merging (only close parts, not everything)
    'enable_merging': True,  # Set to False to disable mask merging for debugging
    
    # Geometry filters (relative to frame area) - RELAXED for difficult cases
    'min_area_frac': 0.001,  # Minimum visible object (very small)
    'max_area_frac': 0.50,  # Allow very large objects (up to half the frame)
    'min_solidity': 0.25,  # Very permissive minimum
    'max_solidity': 0.96,  # Reject only near-perfect circles
    'min_extent': 0.10,  # Very relaxed extent
    'max_extent': 0.92,  # Very relaxed max extent
    'max_aspect_ratio': 6.0,  # Allow very elongated shapes
}


class SAMMonkeySegmenter:
    """SAM-based monkey segmentation that suppresses cage detections."""
    
    def __init__(self, config=None):
        self.config = SAM_CONFIG.copy()
        if config is not None:
            self.config.update(config)
        
        # Store device
        self.device = self.config['device']
        self.model_type = self.config['model_type']
        self.checkpoint_path = self.config['checkpoint_path']
        
        # Initialize SAM model
        print(f"Loading SAM model ({self.model_type}) on {self.device}...")
        sam = sam_model_registry[self.model_type](
            checkpoint=self.checkpoint_path
        )
        sam.to(device=self.device)
        
        # Create mask generator
        self.mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=self.config['points_per_side'],
            pred_iou_thresh=self.config['pred_iou_thresh'],
            stability_score_thresh=self.config['stability_score_thresh'],
            crop_n_layers=self.config['crop_n_layers'],
            crop_n_points_downscale_factor=self.config['crop_n_points_downscale_factor'],
            min_mask_region_area=self.config['min_mask_region_area']
        )
        print("SAM model loaded successfully!")
    
    def _mask_touches_border(self, mask, border_margin):
        """Check if mask touches image borders."""
        h, w = mask.shape
        edges = (
            mask[0, :].any() or
            mask[-1, :].any() or
            mask[:, 0].any() or
            mask[:, -1].any() or
            mask[:border_margin, :].any() or
            mask[h-border_margin:, :].any() or
            mask[:, :border_margin].any() or
            mask[:, w-border_margin:].any()
        )
        return edges
    
    def _merge_nearby_masks(self, masks, max_gap_frac=0.05):
        """
        Merge masks that are close to each other (handles objects split by thin obstacles).
        
        Args:
            masks: List of mask dictionaries from SAM
            max_gap_frac: Maximum gap between masks as fraction of frame diagonal
            
        Returns:
            List of merged mask dictionaries
        """
        if len(masks) == 0:
            return masks
        
        # Get frame dimensions from first mask
        h, w = masks[0]['segmentation'].shape
        frame_diagonal = np.sqrt(h**2 + w**2)
        max_gap = max_gap_frac * frame_diagonal
        
        # Calculate centroids for all masks
        mask_info = []
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            coords = np.argwhere(mask)
            if len(coords) == 0:
                continue
            cy, cx = coords.mean(axis=0)
            area = mask_data['area']
            mask_info.append({
                'idx': i,
                'centroid': (cx, cy),
                'area': area,
                'mask_data': mask_data
            })
        
        if len(mask_info) == 0:
            return masks
        
        # Build adjacency graph of nearby masks
        n = len(mask_info)
        merged = [False] * n
        merged_masks = []
        
        for i in range(n):
            if merged[i]:
                continue
                
            # Start a new merged group with mask i
            group = [i]
            merged[i] = True
            
            # Find all masks close to anything in the group
            changed = True
            while changed:
                changed = False
                for g_idx in list(group):
                    cx1, cy1 = mask_info[g_idx]['centroid']
                    for j in range(n):
                        if merged[j]:
                            continue
                        cx2, cy2 = mask_info[j]['centroid']
                        dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                        
                        if dist < max_gap:
                            group.append(j)
                            merged[j] = True
                            changed = True
            
            # Merge all masks in the group
            if len(group) == 1:
                # No merging needed
                merged_masks.append(mask_info[group[0]]['mask_data'])
            else:
                # Combine masks with OR operation
                combined_mask = np.zeros_like(mask_info[group[0]]['mask_data']['segmentation'])
                total_area = 0
                max_stability = 0
                max_iou = 0
                
                for g_idx in group:
                    mask_data = mask_info[g_idx]['mask_data']
                    combined_mask = np.logical_or(combined_mask, mask_data['segmentation'])
                    total_area += mask_data['area']
                    max_stability = max(max_stability, mask_data.get('stability_score', 0))
                    max_iou = max(max_iou, mask_data.get('predicted_iou', 0))
                
                # Create merged mask data
                merged_data = {
                    'segmentation': combined_mask,
                    'area': int(combined_mask.sum()),
                    'stability_score': max_stability,
                    'predicted_iou': max_iou,
                    'bbox': None  # Will be recalculated in main function
                }
                merged_masks.append(merged_data)
        
        return merged_masks
    
    def segment(self, frame, debug=False):
        """
        Segment monkey from frame using SAM.
        
        Args:
            frame: RGB image (numpy array)
            debug: If True, return visualization
            
        Returns:
            result: Dict with mask, bbox, centroid, area, etc.
            debug_frame: Visualization (if debug=True)
        """
        # Convert RGB to BGR for SAM
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Get frame dimensions for debug output
        h, w = frame.shape[:2]
        
        # Generate masks
        masks = self.mask_generator.generate(frame_bgr)
        
        if debug:
            print(f"\n📊 SAM generated {len(masks)} initial masks")
            if len(masks) > 0:
                areas = [m['area'] / (h * w) for m in masks]
                print(f"   Area range: {min(areas):.1%} to {max(areas):.1%}")
        
        if len(masks) == 0:
            return None, None
        
        # Merge nearby masks (handles objects split by thin obstacles like cage bars)
        if self.config.get('enable_merging', True):
            pre_merge_count = len(masks)
            masks = self._merge_nearby_masks(masks, max_gap_frac=self.config['merge_gap_frac'])
            
            if debug and len(masks) != pre_merge_count:
                print(f"   After merging: {len(masks)} masks ({pre_merge_count - len(masks)} groups merged)")
        elif debug:
            print(f"   Merging disabled")
        
        if len(masks) == 0:
            return None, None
        
        # Use frame dimensions for calculations
        frame_area = h * w
        min_area = self.config['min_area_frac'] * frame_area
        max_area = self.config['max_area_frac'] * frame_area
        border_margin = int(self.config['border_margin_frac'] * min(h, w))
        cx_frame, cy_frame = w / 2.0, h / 2.0
        
        # Score and filter masks
        candidates = []
        rejected = []  # Track rejected masks for debugging
        
        for mask_data in masks:
            mask = mask_data['segmentation']
            area = mask_data['area']
            area_frac = area / frame_area
            
            rejection_reason = None
            
            # Area filter
            if area < min_area or area > max_area:
                rejection_reason = f"area ({area_frac:.3f}, min={self.config['min_area_frac']:.3f}, max={self.config['max_area_frac']:.3f})"
                if debug:
                    rejected.append({'area_frac': area_frac, 'reason': rejection_reason})
                continue
            
            # Border filter
            if self._mask_touches_border(mask, border_margin):
                rejection_reason = "touches_border"
                if debug:
                    rejected.append({'area_frac': area_frac, 'reason': rejection_reason})
                continue
            
            # Get bounding box
            coords = np.argwhere(mask)
            if len(coords) == 0:
                continue
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            bbox_w = x_max - x_min + 1
            bbox_h = y_max - y_min + 1
            bbox = (int(x_min), int(y_min), int(bbox_w), int(bbox_h))
            
            # Calculate shape metrics
            bbox_area = max(bbox_w * bbox_h, 1)
            extent = area / bbox_area
            
            # Calculate solidity (mask area / convex hull area)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if len(contours) == 0:
                continue
            contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(contour)
            hull_area = max(cv2.contourArea(hull), 1.0)
            solidity = area / hull_area
            
            aspect_ratio = max(bbox_w / max(bbox_h, 1), bbox_h / max(bbox_w, 1))
            
            # Geometry filters with rejection tracking
            if solidity < self.config['min_solidity']:
                if debug:
                    rejected.append({'area_frac': area_frac, 'reason': f"solidity_low ({solidity:.3f} < {self.config['min_solidity']:.3f})"})
                continue
            if solidity > self.config.get('max_solidity', 1.0):
                if debug:
                    rejected.append({'area_frac': area_frac, 'reason': f"solidity_high ({solidity:.3f} > {self.config.get('max_solidity', 1.0):.3f})"})
                continue
            if extent < self.config['min_extent']:
                if debug:
                    rejected.append({'area_frac': area_frac, 'reason': f"extent_low ({extent:.3f} < {self.config['min_extent']:.3f})"})
                continue
            if extent > self.config.get('max_extent', 1.0):
                if debug:
                    rejected.append({'area_frac': area_frac, 'reason': f"extent_high ({extent:.3f} > {self.config.get('max_extent', 1.0):.3f})"})
                continue
            if aspect_ratio > self.config['max_aspect_ratio']:
                if debug:
                    rejected.append({'area_frac': area_frac, 'reason': f"aspect_ratio ({aspect_ratio:.2f} > {self.config['max_aspect_ratio']:.2f})"})
                continue
            
            # Calculate centroid
            M = cv2.moments(mask.astype(np.uint8))
            if M['m00'] == 0:
                continue
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            centroid = (int(cx), int(cy))
            
            # Score: prefer moderately compact, central, large, high-quality objects
            center_dist_norm = np.sqrt((cx - cx_frame) ** 2 + (cy - cy_frame) ** 2) / np.sqrt(cx_frame ** 2 + cy_frame ** 2)
            stability = mask_data.get('stability_score', 0.5)
            predicted_iou = mask_data.get('predicted_iou', 0.5)
            
            # Compactness score: prefer moderate values (organic shapes, not perfect circles)
            # Very forgiving for merged masks and difficult cases
            ideal_solidity = 0.60  # Lower ideal for merged/irregular masks
            solidity_score = 1.0 - min(1.0, abs(solidity - ideal_solidity) / 0.8)
            ideal_extent = 0.40  # Lower ideal for merged masks
            extent_score = 1.0 - min(1.0, abs(extent - ideal_extent) / 0.8)
            compactness = max(0.05, solidity_score * extent_score)  # Lower minimum
            
            # Combined score: favor large, moderately compact, central, high-quality objects
            quality = stability * predicted_iou  # SAM's confidence scores
            centrality = 1.0 - self.config['center_weight'] * center_dist_norm
            score = area * compactness * quality * centrality
            
            candidates.append({
                'score': score,
                'mask': mask,
                'contour': contour,
                'area': area,
                'bbox': bbox,
                'centroid': centroid,
                'solidity': solidity,
                'extent': extent,
                'stability_score': stability,
                'predicted_iou': predicted_iou
            })
        
        if len(candidates) == 0:
            if debug:
                print(f"\n⚠️  No masks passed all filters!")
                print(f"Total masks after merging: {len(masks)}")
                print(f"Rejected: {len(rejected)}")
                if rejected:
                    print("\nRejection reasons:")
                    reason_counts = {}
                    for r in rejected:
                        key = r['reason'].split('(')[0].strip()
                        reason_counts[key] = reason_counts.get(key, 0) + 1
                    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
                        print(f"  {count}x {reason}")
                    print("\nTop rejected masks:")
                    for r in rejected[:3]:
                        print(f"  - Area {r['area_frac']:.1%}: {r['reason']}")
            
            # FALLBACK: If no candidates, relax filters and return best mask anyway
            print("\n🔧 Applying fallback: returning best mask regardless of filters...")
            fallback_candidates = []
            for mask_data in masks:
                mask = mask_data['segmentation']
                area = mask_data['area']
                
                # Skip only impossibly extreme cases
                if area < min_area * 0.1:  # Truly tiny
                    continue
                if area > frame_area * 0.8:  # Nearly entire frame
                    continue
                
                # Calculate all metrics without filtering
                coords = np.argwhere(mask)
                if len(coords) == 0:
                    continue
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                bbox_w = x_max - x_min + 1
                bbox_h = y_max - y_min + 1
                bbox = (int(x_min), int(y_min), int(bbox_w), int(bbox_h))
                
                bbox_area = max(bbox_w * bbox_h, 1)
                extent = area / bbox_area
                
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if len(contours) == 0:
                    continue
                contour = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(contour)
                hull_area = max(cv2.contourArea(hull), 1.0)
                solidity = area / hull_area
                
                M = cv2.moments(mask.astype(np.uint8))
                if M['m00'] == 0:
                    continue
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                centroid = (int(cx), int(cy))
                
                # Score
                center_dist_norm = np.sqrt((cx - cx_frame) ** 2 + (cy - cy_frame) ** 2) / np.sqrt(cx_frame ** 2 + cy_frame ** 2)
                stability = mask_data.get('stability_score', 0.5)
                predicted_iou = mask_data.get('predicted_iou', 0.5)
                quality = stability * predicted_iou
                centrality = 1.0 - 0.3 * center_dist_norm  # Reduced center weight
                score = area * quality * centrality  # Simplified score
                
                fallback_candidates.append({
                    'score': score,
                    'mask': mask,
                    'contour': contour,
                    'area': area,
                    'bbox': bbox,
                    'centroid': centroid,
                    'solidity': solidity,
                    'extent': extent,
                    'stability_score': stability,
                    'predicted_iou': predicted_iou
                })
            
            if len(fallback_candidates) == 0:
                print("❌ No fallback candidates found")
                return None, None
            
            fallback_candidates.sort(key=lambda x: x['score'], reverse=True)
            best = fallback_candidates[0]
            print(f"✓ Fallback selected: area={best['area']/frame_area:.1%}, solidity={best['solidity']:.2f}, extent={best['extent']:.2f}")
            candidates = fallback_candidates  # Use fallback candidates
        else:
            # Select best candidate from normal filtering
            candidates.sort(key=lambda x: x['score'], reverse=True)
            best = candidates[0]
            
            if debug:
                print(f"\n✓ Found {len(candidates)} valid candidates")
                print(f"Best: area={best['area']/frame_area:.1%}, solidity={best['solidity']:.2f}, extent={best['extent']:.2f}")
        
        # Build result from best candidate
        best = candidates[0]
        
        result = {
            'mask': best['mask'].astype(np.uint8) * 255,
            'contour': best['contour'],
            'area': best['area'],
            'bbox': best['bbox'],
            'centroid': best['centroid'],
            'solidity': best['solidity'],
            'extent': best['extent'],
            'stability_score': best['stability_score'],
            'predicted_iou': best['predicted_iou'],
            'n_candidates': len(candidates)
        }
        
        if debug:
            debug_frame = frame.copy()
            x, y, bw, bh = best['bbox']
            cx, cy = best['centroid']
            
            # Draw mask overlay
            mask_overlay = np.zeros_like(frame)
            mask_overlay[best['mask']] = [0, 255, 0]
            debug_frame = cv2.addWeighted(debug_frame, 0.7, mask_overlay, 0.3, 0)
            
            # Draw contour and bbox
            cv2.drawContours(debug_frame, [best['contour']], -1, (0, 255, 0), 3)
            cv2.rectangle(debug_frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
            cv2.circle(debug_frame, (cx, cy), 6, (255, 0, 0), -1)
            
            # Draw border exclusion zone
            cv2.rectangle(
                debug_frame,
                (border_margin, border_margin),
                (w - border_margin, h - border_margin),
                (255, 255, 0),
                2
            )
            
            return result, debug_frame
        
        return result, None


def segment_monkey_sam(frame, segmenter, debug=False):
    """Wrapper function for compatibility with existing code."""
    return segmenter.segment(frame, debug=debug)
