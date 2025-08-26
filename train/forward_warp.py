#!/usr/bin/env python3
"""
Forward Warp Implementation - Motion vector based forward projection.

Implements forward splatting for generating intermediate frames from motion vectors.
Python version of the GPU module for training data generation.
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from numba import jit, prange


def forward_warp_vectorized(source_image: np.ndarray, 
                          motion_vectors: np.ndarray,
                          method: str = 'splatting') -> Tuple[np.ndarray, np.ndarray]:
    """
    Motion vector based forward projection warping.
    
    Simulates GPU compute shader forward projection:
    1. Calculate target position for each source pixel using motion vectors
    2. Splat source pixels to target positions (Forward Splatting)
    3. Handle conflicts when multiple sources project to same target
    4. Generate coverage mask for valid projection regions
    
    Args:
        source_image: Source image [3, H, W] or [H, W, 3]
        motion_vectors: Motion vectors [2, H, W] or [H, W, 2]  
        method: Projection method ('splatting', 'nearest', 'bilinear')
        
    Returns:
        warped_image: Forward projected image [3, H, W]
        coverage_mask: Coverage mask [H, W], higher values = more source coverage
    """
    # Normalize input format to CHW
    if source_image.ndim == 3 and source_image.shape[2] in [1, 3]:
        source_image = source_image.transpose(2, 0, 1)  # HWC -> CHW
    
    if motion_vectors.ndim == 3 and motion_vectors.shape[2] == 2:
        motion_vectors = motion_vectors.transpose(2, 0, 1)  # HWC -> CHW
    
    C, H, W = source_image.shape
    
    # Initialize output
    warped_image = np.zeros_like(source_image)
    coverage_mask = np.zeros((H, W), dtype=np.float32)
    depth_buffer = np.full((H, W), np.inf, dtype=np.float32)  # For projection conflict resolution
    
    if method == 'splatting':
        warped_image, coverage_mask = _forward_splatting_optimized(
            source_image, motion_vectors, warped_image, coverage_mask, depth_buffer
        )
    elif method == 'nearest':
        warped_image, coverage_mask = _forward_nearest(
            source_image, motion_vectors, warped_image, coverage_mask
        )
    elif method == 'bilinear':
        warped_image, coverage_mask = _forward_bilinear(
            source_image, motion_vectors, warped_image, coverage_mask
        )
    else:
        raise ValueError(f"Unknown projection method: {method}")
    
    return warped_image, coverage_mask


def _forward_splatting_optimized(source_image: np.ndarray,
                               motion_vectors: np.ndarray,
                               warped_image: np.ndarray,
                               coverage_mask: np.ndarray,
                               depth_buffer: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized forward splatting implementation.
    
    Simulates GPU atomic depth testing:
    - Calculate target positions for each source pixel
    - Use depth testing for multiple source conflicts
    - Closer pixels override distant ones (Z-buffer style)
    """
    C, H, W = source_image.shape
    
    # Create source coordinate grid
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    
    # Calculate target coordinates
    target_x = x_coords + motion_vectors[0]  # u component
    target_y = y_coords + motion_vectors[1]  # v component
    
    # Calculate depth (simplified as distance from image center)
    center_x, center_y = W // 2, H // 2
    source_depth = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    
    # Find valid projections (within image bounds)
    valid_mask = ((target_x >= 0) & (target_x < W) & 
                  (target_y >= 0) & (target_y < H))
    
    if not np.any(valid_mask):
        return warped_image, coverage_mask
    
    # Get valid projection coordinates
    valid_source_y, valid_source_x = np.where(valid_mask)
    valid_target_x = target_x[valid_mask]
    valid_target_y = target_y[valid_mask]
    valid_depths = source_depth[valid_mask]
    
    # Convert to integer coordinates
    target_x_int = np.clip(valid_target_x.astype(np.int32), 0, W-1)
    target_y_int = np.clip(valid_target_y.astype(np.int32), 0, H-1)
    
    # Use Numba accelerated core projection loop
    _splatting_core_numba(
        source_image, warped_image, coverage_mask, depth_buffer,
        valid_source_x, valid_source_y, 
        target_x_int, target_y_int, valid_depths
    )
    
    return warped_image, coverage_mask


@jit(nopython=True, parallel=True)
def _splatting_core_numba(source_image, warped_image, coverage_mask, depth_buffer,
                         src_x, src_y, tgt_x, tgt_y, depths):
    """
    Numba accelerated splatting core loop.
    
    Simulates GPU compute shader atomic operations.
    """
    C = source_image.shape[0]
    
    for i in prange(len(src_x)):
        sx, sy = src_x[i], src_y[i]
        tx, ty = tgt_x[i], tgt_y[i]
        depth = depths[i]
        
        # Atomic operation simulation: depth test
        if depth < depth_buffer[ty, tx]:
            depth_buffer[ty, tx] = depth
            
            # Update pixel value
            for c in range(C):
                warped_image[c, ty, tx] = source_image[c, sy, sx]
        
        # Always increment coverage count regardless of depth test
        coverage_mask[ty, tx] += 1.0


def _forward_nearest(source_image: np.ndarray,
                    motion_vectors: np.ndarray,
                    warped_image: np.ndarray,
                    coverage_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Nearest neighbor forward projection.
    """
    C, H, W = source_image.shape
    
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    target_x = x_coords + motion_vectors[0]
    target_y = y_coords + motion_vectors[1]
    
    valid_mask = ((target_x >= 0) & (target_x < W) & 
                  (target_y >= 0) & (target_y < H))
    
    if np.any(valid_mask):
        target_x_int = np.clip(target_x[valid_mask].astype(np.int32), 0, W-1)
        target_y_int = np.clip(target_y[valid_mask].astype(np.int32), 0, H-1)
        source_y, source_x = np.where(valid_mask)
        
        for i in range(len(source_x)):
            sx, sy = source_x[i], source_y[i]
            tx, ty = target_x_int[i], target_y_int[i]
            
            warped_image[:, ty, tx] = source_image[:, sy, sx]
            coverage_mask[ty, tx] += 1.0
    
    return warped_image, coverage_mask


def _forward_bilinear(source_image: np.ndarray,
                     motion_vectors: np.ndarray,
                     warped_image: np.ndarray,
                     coverage_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bilinear interpolation forward projection.
    """
    C, H, W = source_image.shape
    
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    target_x = x_coords + motion_vectors[0]
    target_y = y_coords + motion_vectors[1]
    
    valid_mask = ((target_x >= 0) & (target_x < W-1) & 
                  (target_y >= 0) & (target_y < H-1))
    
    if not np.any(valid_mask):
        return warped_image, coverage_mask
    
    # Get valid pixels
    source_y, source_x = np.where(valid_mask)
    tx = target_x[valid_mask]
    ty = target_y[valid_mask]
    
    # Bilinear interpolation weights
    tx_floor = np.floor(tx).astype(np.int32)
    ty_floor = np.floor(ty).astype(np.int32)
    tx_ceil = tx_floor + 1
    ty_ceil = ty_floor + 1
    
    wx = tx - tx_floor
    wy = ty - ty_floor
    
    # Weights for four corner points
    w00 = (1 - wx) * (1 - wy)
    w01 = (1 - wx) * wy
    w10 = wx * (1 - wy)  
    w11 = wx * wy
    
    # Distribute to four corner points
    for i in range(len(source_x)):
        sx, sy = source_x[i], source_y[i]
        pixel_value = source_image[:, sy, sx]
        
        # Four target positions
        positions = [(ty_floor[i], tx_floor[i]), (ty_ceil[i], tx_floor[i]),
                    (ty_floor[i], tx_ceil[i]), (ty_ceil[i], tx_ceil[i])]
        weights = [w00[i], w01[i], w10[i], w11[i]]
        
        for (py, px), weight in zip(positions, weights):
            if 0 <= py < H and 0 <= px < W and weight > 0:
                warped_image[:, py, px] += pixel_value * weight
                coverage_mask[py, px] += weight
    
    return warped_image, coverage_mask


def create_test_motion_vectors(H: int, W: int, motion_type: str = 'translation') -> np.ndarray:
    """
    Create test motion vectors.
    
    Args:
        H, W: Image dimensions
        motion_type: Motion type ('translation', 'rotation', 'radial', 'random')
        
    Returns:
        motion_vectors: Motion vectors [2, H, W]
    """
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    center_x, center_y = W // 2, H // 2
    
    if motion_type == 'translation':
        # Global translation
        motion_u = np.full((H, W), 5.0, dtype=np.float32)
        motion_v = np.full((H, W), -3.0, dtype=np.float32)
        
    elif motion_type == 'rotation':
        # Rotation around center
        dx = x_coords - center_x
        dy = y_coords - center_y
        angle = 0.1  # radians
        
        motion_u = -dy * np.sin(angle) + dx * (np.cos(angle) - 1)
        motion_v = dx * np.sin(angle) + dy * (np.cos(angle) - 1)
        
    elif motion_type == 'radial':
        # Radial motion (scaling effect)
        dx = x_coords - center_x
        dy = y_coords - center_y
        scale = 0.05
        
        motion_u = dx * scale
        motion_v = dy * scale
        
    elif motion_type == 'random':
        # Random motion
        motion_u = np.random.randn(H, W) * 2.0
        motion_v = np.random.randn(H, W) * 2.0
        
    else:
        raise ValueError(f"Unknown motion type: {motion_type}")
    
    return np.stack([motion_u, motion_v], axis=0)


def visualize_warp_result(source_image: np.ndarray,
                         warped_image: np.ndarray, 
                         coverage_mask: np.ndarray,
                         motion_vectors: np.ndarray,
                         save_path: Optional[str] = None):
    """
    Visualize warp results.
    """
    import matplotlib.pyplot as plt
    
    # Convert to display format
    if source_image.shape[0] <= 3:  # CHW format
        source_vis = source_image.transpose(1, 2, 0)
        warped_vis = warped_image.transpose(1, 2, 0)
    else:
        source_vis = source_image
        warped_vis = warped_image
    
    # Ensure values in valid range
    source_vis = np.clip(source_vis, 0, 1)
    warped_vis = np.clip(warped_vis, 0, 1)
    
    # Handle single channel images
    if source_vis.shape[2] == 1:
        source_vis = np.repeat(source_vis, 3, axis=2)
        warped_vis = np.repeat(warped_vis, 3, axis=2)
    
    # Calculate motion vector magnitude
    mv_magnitude = np.sqrt(motion_vectors[0]**2 + motion_vectors[1]**2)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(source_vis)
    axes[0, 0].set_title('Source Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(warped_vis)
    axes[0, 1].set_title('Warped Image (Forward Projection)')
    axes[0, 1].axis('off')
    
    im1 = axes[1, 0].imshow(coverage_mask, cmap='hot')
    axes[1, 0].set_title('Coverage Mask')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
    
    im2 = axes[1, 1].imshow(mv_magnitude, cmap='jet')
    axes[1, 1].set_title('Motion Vector Magnitude')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    """Test forward warp implementation."""
    print("[TEST] Testing forward warp implementation...")
    
    # Create test image
    H, W = 256, 256
    
    # Create a test image with features
    test_image = np.zeros((3, H, W), dtype=np.float32)
    
    # Add geometric shapes
    cv2.circle(test_image.transpose(1, 2, 0), (64, 64), 30, (1.0, 0.0, 0.0), -1)    # Red circle
    cv2.rectangle(test_image.transpose(1, 2, 0), (120, 100), (180, 140), (0.0, 1.0, 0.0), -1)  # Green rectangle
    cv2.circle(test_image.transpose(1, 2, 0), (180, 180), 25, (0.0, 0.0, 1.0), -1)  # Blue circle
    
    # Test different motion types
    motion_types = ['translation', 'rotation', 'radial']
    
    for motion_type in motion_types:
        print(f"\n[TEST] Testing motion type: {motion_type}")
        
        # Create motion vectors
        motion_vectors = create_test_motion_vectors(H, W, motion_type)
        
        # Execute forward warp
        warped_image, coverage_mask = forward_warp_vectorized(
            test_image, motion_vectors, method='splatting'
        )
        
        print(f"[SUCCESS] Warp completed:")
        print(f"   Coverage area: {np.sum(coverage_mask > 0) / (H*W) * 100:.1f}%")
        print(f"   Average coverage: {np.mean(coverage_mask[coverage_mask > 0]):.2f}")
        print(f"   Max coverage: {np.max(coverage_mask):.2f}")
        
        # Visualize results
        try:
            visualize_warp_result(
                test_image, warped_image, coverage_mask, motion_vectors,
                save_path=f"warp_test_{motion_type}.png"
            )
        except Exception as e:
            print(f"Visualization failed: {e}")
    
    print("\n[COMPLETE] Forward warp testing completed!")