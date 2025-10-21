#!/usr/bin/env python3
"""
Hole Detector - Core component for patch-based architecture
Analyzes geometric hole masks and determines patch centers for processing

Features: connected components analysis, configurable parameters, efficient algorithms
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class HoleDetectorConfig:
    """Hole detector configuration"""
    min_hole_area: int = 16          # Min hole area to process (pixels)
    patch_size: int = 128            # Patch size
    merge_distance: int = 64         # Distance threshold for hole merging
    max_patches_per_image: int = 16  # Max patches per image
    boundary_margin: int = 8         # Boundary safety margin


@dataclass
class PatchInfo:
    """Patch information structure"""
    center_x: int                    # Patch center x coordinate
    center_y: int                    # Patch center y coordinate
    hole_area: int                   # Corresponding hole area
    patch_id: int                    # Unique patch identifier
    boundary_valid: bool = True      # Whether within valid boundaries
    merged_holes: int = 1            # Number of merged holes


class HoleDetector:
    """
    Hole Detector - Analyzes geometric hole masks and determines patch centers
    
    Features:
    1. Connected components analysis for hole regions
    2. Area-based filtering of valid holes
    3. Optimal patch center calculation
    4. Nearby hole merging to reduce patch count
    5. Boundary validation for patch validity
    """
    
    def __init__(self, config: Optional[HoleDetectorConfig] = None):
        """Initialize hole detector
        
        Args:
            config: Detector config, uses default if None
        """
        self.config = config or HoleDetectorConfig()
        
        # Pre-compute common parameters
        self.half_patch = self.config.patch_size // 2
        self.merge_distance_sq = self.config.merge_distance ** 2  # Squared distance to avoid sqrt
    
    def detect_patch_centers(self, holes_mask: np.ndarray) -> List[PatchInfo]:
        """Detect patch centers
        
        Args:
            holes_mask: Binary hole mask [H,W], 0=background, 1=hole
            
        Returns:
            patch_info_list: List of patch info with centers and areas
        """
        # Input validation
        if holes_mask is None or holes_mask.size == 0:
            return []
        
        # Ensure binary input
        if holes_mask.dtype != np.uint8:
            holes_mask = (holes_mask > 0).astype(np.uint8)
        
        # 1. Connected components analysis
        hole_regions = self._find_connected_holes(holes_mask)
        
        if len(hole_regions) == 0:
            return []
        
        # 2. Area filtering
        valid_holes = self._filter_holes_by_area(hole_regions)
        
        if len(valid_holes) == 0:
            return []
        
        # 3. Merge nearby holes
        merged_holes = self._merge_nearby_holes(valid_holes)
        
        # 4. Boundary validation
        final_patches = self._validate_patch_bounds(merged_holes, holes_mask.shape)
        
        # 5. Limit patch count
        if len(final_patches) > self.config.max_patches_per_image:
            # Sort by hole area, keep largest patches
            final_patches.sort(key=lambda x: x.hole_area, reverse=True)
            final_patches = final_patches[:self.config.max_patches_per_image]
        
        return final_patches
    
    def _find_connected_holes(self, holes_mask: np.ndarray) -> List[Dict]:
        """Find all hole regions using connected components analysis
        
        Args:
            holes_mask: Binary hole mask
            
        Returns:
            hole_regions: List of hole region info
        """
        # Use OpenCV connected components analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            holes_mask, connectivity=8
        )
        
        hole_regions = []
        
        # Skip background label (label 0)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Extract region info
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Centroid coordinates
            center_x = int(centroids[i, 0])
            center_y = int(centroids[i, 1])
            
            hole_regions.append({
                'label': i,
                'area': area,
                'center_x': center_x,
                'center_y': center_y,
                'bbox': (x, y, w, h),
                'original': True  # Mark as original hole, not merged
            })
        
        return hole_regions
    
    def _filter_holes_by_area(self, hole_regions: List[Dict]) -> List[Dict]:
        """Filter holes by area
        
        Args:
            hole_regions: Original hole region list
            
        Returns:
            valid_holes: Holes meeting area criteria
        """
        valid_holes = []
        
        for hole in hole_regions:
            if hole['area'] >= self.config.min_hole_area:
                valid_holes.append(hole)
        
        return valid_holes
    
    def _merge_nearby_holes(self, valid_holes: List[Dict]) -> List[Dict]:
        """Merge nearby holes to reduce patch count
        
        Args:
            valid_holes: Valid hole list
            
        Returns:
            merged_holes: Merged hole list
        """
        if len(valid_holes) <= 1:
            return valid_holes
        
        # Use union-find algorithm for hole merging
        n = len(valid_holes)
        parent = list(range(n))  # Union-find parent array
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Calculate hole distances and merge nearby holes
        for i in range(n):
            for j in range(i + 1, n):
                hole1, hole2 = valid_holes[i], valid_holes[j]
                
                # Calculate squared distance between centers
                dx = hole1['center_x'] - hole2['center_x']
                dy = hole1['center_y'] - hole2['center_y']
                distance_sq = dx * dx + dy * dy
                
                # Merge holes if distance is below threshold
                if distance_sq <= self.merge_distance_sq:
                    union(i, j)
        
        # Collect holes by merge groups
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(valid_holes[i])
        
        # Generate merged hole info
        merged_holes = []
        for group_holes in groups.values():
            if len(group_holes) == 1:
                # Single hole, keep original info
                merged_holes.append(group_holes[0])
            else:
                # Multiple holes merged, calculate new center and total area
                total_area = sum(hole['area'] for hole in group_holes)
                
                # Weighted average for new center
                weighted_x = sum(hole['center_x'] * hole['area'] for hole in group_holes)
                weighted_y = sum(hole['center_y'] * hole['area'] for hole in group_holes)
                
                new_center_x = int(weighted_x / total_area)
                new_center_y = int(weighted_y / total_area)
                
                merged_hole = {
                    'area': total_area,
                    'center_x': new_center_x,
                    'center_y': new_center_y,
                    'original': False,  # Mark as merged hole
                    'merged_count': len(group_holes)
                }
                
                merged_holes.append(merged_hole)
        
        return merged_holes
    
    def _validate_patch_bounds(self, holes: List[Dict], image_shape: Tuple[int, int]) -> List[PatchInfo]:
        """Validate patch boundaries to ensure patches are within image bounds
        
        Args:
            holes: Hole info list
            image_shape: Image shape (H, W)
            
        Returns:
            patch_info_list: Validated patch info list
        """
        H, W = image_shape
        patch_info_list = []
        
        for i, hole in enumerate(holes):
            center_x, center_y = hole['center_x'], hole['center_y']
            
            # Calculate patch boundaries
            x1 = center_x - self.half_patch
            y1 = center_y - self.half_patch
            x2 = center_x + self.half_patch
            y2 = center_y + self.half_patch
            
            # Check if within valid boundaries (considering safety margin)
            margin = self.config.boundary_margin
            boundary_valid = (
                x1 >= -margin and y1 >= -margin and 
                x2 <= W + margin and y2 <= H + margin
            )
            
            # Adjust center to safe range if out of bounds
            if not boundary_valid:
                center_x = np.clip(center_x, self.half_patch, W - self.half_patch)
                center_y = np.clip(center_y, self.half_patch, H - self.half_patch)
                boundary_valid = True  # Mark as valid after adjustment
            
            # Create PatchInfo object
            patch_info = PatchInfo(
                center_x=center_x,
                center_y=center_y,
                hole_area=hole['area'],
                patch_id=i,
                boundary_valid=boundary_valid,
                merged_holes=hole.get('merged_count', 1)
            )
            
            patch_info_list.append(patch_info)
        
        return patch_info_list
    
    def get_patch_bounds(self, patch_info: PatchInfo) -> Tuple[int, int, int, int]:
        """Get patch boundary coordinates
        
        Args:
            patch_info: Patch info
            
        Returns:
            (x1, y1, x2, y2): Patch boundary coordinates
        """
        x1 = patch_info.center_x - self.half_patch
        y1 = patch_info.center_y - self.half_patch
        x2 = x1 + self.config.patch_size
        y2 = y1 + self.config.patch_size
        
        return x1, y1, x2, y2
    
    def visualize_patches(self, holes_mask: np.ndarray, patch_infos: List[PatchInfo]) -> np.ndarray:
        """Visualize detected patches for debugging
        
        Args:
            holes_mask: Original hole mask
            patch_infos: Patch info list
            
        Returns:
            vis_image: Visualization image showing holes and patch boundaries
        """
        # Create 3-channel visualization image
        vis_image = np.stack([holes_mask * 255] * 3, axis=2).astype(np.uint8)
        
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green  
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
        
        for i, patch_info in enumerate(patch_infos):
            color = colors[i % len(colors)]
            
            # Draw patch boundary
            x1, y1, x2, y2 = self.get_patch_bounds(patch_info)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cv2.circle(vis_image, (patch_info.center_x, patch_info.center_y), 3, color, -1)
            
            # Add patch ID label
            cv2.putText(vis_image, f'P{patch_info.patch_id}', 
                       (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis_image


def test_hole_detector():
    """Simple test function"""
    # Create test hole mask
    test_mask = np.zeros((270, 480), dtype=np.uint8)
    
    # Add test holes
    test_mask[50:80, 100:150] = 1    # Hole 1
    test_mask[150:170, 200:240] = 1  # Hole 2  
    test_mask[200:220, 350:380] = 1  # Hole 3
    
    # Create detector
    detector = HoleDetector()
    
    # Detect patch centers
    patch_infos = detector.detect_patch_centers(test_mask)
    
    print(f"Detected {len(patch_infos)} patches:")
    for patch_info in patch_infos:
        print(f"  Patch {patch_info.patch_id}: center=({patch_info.center_x}, {patch_info.center_y}), "
              f"area={patch_info.hole_area}, boundary_valid={patch_info.boundary_valid}")
    
    # Generate visualization
    vis_image = detector.visualize_patches(test_mask, patch_infos)
    return vis_image, patch_infos


if __name__ == "__main__":
    # Run test
    vis_image, patch_infos = test_hole_detector()
    print("SUCCESS: HoleDetector test completed")