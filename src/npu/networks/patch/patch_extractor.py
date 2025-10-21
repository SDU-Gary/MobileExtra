#!/usr/bin/env python3
"""
Patch Extractor - Core component for patch-based architecture

Extracts 128x128 regions from full data based on hole center points.
Supports numpy and torch formats with intelligent boundary handling.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Union, Optional
from dataclasses import dataclass
from enum import Enum

# Import HoleDetector data structures
from .hole_detector import PatchInfo


class PaddingMode(Enum):
    """Padding mode enumeration"""
    MIRROR = "mirror"        # Mirror padding (recommended)
    CONSTANT = "constant"    # Constant padding
    EDGE = "edge"           # Edge replication padding
    WRAP = "wrap"           # Wrap padding


@dataclass
class PatchExtractorConfig:
    """Patch extractor configuration"""
    patch_size: int = 128
    padding_mode: PaddingMode = PaddingMode.MIRROR
    constant_value: float = 0.0
    enable_batching: bool = True
    preserve_dtype: bool = True


@dataclass 
class PatchPosition:
    """Patch位置信息"""
    patch_id: int                   # patch唯一标识
    center_x: int                   # 原始中心x坐标
    center_y: int                   # 原始中心y坐标
    extract_x1: int                 # 实际提取区域左上角x
    extract_y1: int                 # 实际提取区域左上角y  
    extract_x2: int                 # 实际提取区域右下角x
    extract_y2: int                 # 实际提取区域右下角y
    pad_left: int = 0               # 左侧padding像素数
    pad_top: int = 0                # 顶部padding像素数
    pad_right: int = 0              # 右侧padding像素数
    pad_bottom: int = 0             # 底部padding像素数
    original_shape: Tuple[int, int] = None  # 原始图像尺寸 (H,W)


class PatchExtractor:
    """Patch extractor for extracting regions from full data
    
    Features:
    - Extract 128x128 patches based on center coordinates
    - Intelligent boundary handling with padding strategies
    - Support for numpy and torch data formats
    - Batch extraction optimization
    - Precise position tracking for fusion
    """
    
    def __init__(self, config: Optional[PatchExtractorConfig] = None):
        """Initialize patch extractor
        
        Args:
            config: Extractor configuration, uses default if None
        """
        self.config = config or PatchExtractorConfig()
        self.half_patch = self.config.patch_size // 2
    
    def extract_patches(self, 
                       full_data: Union[np.ndarray, torch.Tensor], 
                       patch_infos: List[PatchInfo]) -> Tuple[Union[np.ndarray, torch.Tensor], List[PatchPosition]]:
        """Batch extract patches
        
        Args:
            full_data: Full data [C,H,W] or [B,C,H,W]
            patch_infos: List of patch information
            
        Returns:
            patches: Extracted patches [N,C,patch_size,patch_size]
            positions: List of patch position information
        """
        if len(patch_infos) == 0:
            # Return empty result, maintain data type consistency
            if isinstance(full_data, torch.Tensor):
                empty_patches = torch.empty(0, full_data.shape[-3], self.config.patch_size, self.config.patch_size, 
                                          dtype=full_data.dtype, device=full_data.device)
            else:
                empty_patches = np.empty((0, full_data.shape[-3], self.config.patch_size, self.config.patch_size), 
                                       dtype=full_data.dtype)
            return empty_patches, []
        
        # Handle batch dimension
        if full_data.ndim == 4:
            # Batch data, take first sample
            data = full_data[0]  # [C,H,W]
        else:
            data = full_data  # [C,H,W]
        
        C, H, W = data.shape
        
        patches = []
        positions = []
        
        for patch_info in patch_infos:
            patch, position = self._extract_single_patch(data, patch_info, (H, W))
            patches.append(patch)
            positions.append(position)
        
        # Batch assembly
        if self.config.enable_batching and len(patches) > 0:
            if isinstance(data, torch.Tensor):
                patches = torch.stack(patches, dim=0)  # [N,C,H,W]
            else:
                patches = np.stack(patches, axis=0)  # [N,C,H,W]
        
        return patches, positions
    
    def _extract_single_patch(self, 
                             data: Union[np.ndarray, torch.Tensor], 
                             patch_info: PatchInfo,
                             image_shape: Tuple[int, int]) -> Tuple[Union[np.ndarray, torch.Tensor], PatchPosition]:
        """
        提取单个patch
        
        Args:
            data: 图像数据 [C,H,W]
            patch_info: patch信息
            image_shape: 图像尺寸 (H,W)
            
        Returns:
            patch: 提取的patch [C,patch_size,patch_size]
            position: 位置信息
        """
        H, W = image_shape
        C = data.shape[0]
        
        # 计算理想的提取区域
        ideal_x1 = patch_info.center_x - self.half_patch
        ideal_y1 = patch_info.center_y - self.half_patch
        ideal_x2 = ideal_x1 + self.config.patch_size
        ideal_y2 = ideal_y1 + self.config.patch_size
        
        # 计算实际可提取区域（限制在图像边界内）
        extract_x1 = max(0, ideal_x1)
        extract_y1 = max(0, ideal_y1)
        extract_x2 = min(W, ideal_x2)
        extract_y2 = min(H, ideal_y2)
        
        # 计算需要的padding
        pad_left = extract_x1 - ideal_x1
        pad_top = extract_y1 - ideal_y1
        pad_right = ideal_x2 - extract_x2
        pad_bottom = ideal_y2 - extract_y2
        
        # 提取实际区域
        extracted_region = data[:, extract_y1:extract_y2, extract_x1:extract_x2]
        
        # 应用padding得到最终patch
        patch = self._apply_padding(
            extracted_region, 
            (C, self.config.patch_size, self.config.patch_size),
            (pad_top, pad_bottom, pad_left, pad_right)
        )
        
        # 创建位置信息
        position = PatchPosition(
            patch_id=patch_info.patch_id,
            center_x=patch_info.center_x,
            center_y=patch_info.center_y,
            extract_x1=extract_x1,
            extract_y1=extract_y1,
            extract_x2=extract_x2,
            extract_y2=extract_y2,
            pad_left=pad_left,
            pad_top=pad_top,
            pad_right=pad_right,
            pad_bottom=pad_bottom,
            original_shape=image_shape
        )
        
        return patch, position
    
    def _apply_padding(self, 
                      region: Union[np.ndarray, torch.Tensor],
                      target_shape: Tuple[int, int, int],
                      padding: Tuple[int, int, int, int]) -> Union[np.ndarray, torch.Tensor]:
        """
        对提取区域应用padding
        
        Args:
            region: 提取的区域 [C,H,W]
            target_shape: 目标形状 (C,H,W)
            padding: padding参数 (top, bottom, left, right)
            
        Returns:
            padded: padding后的patch
        """
        pad_top, pad_bottom, pad_left, pad_right = padding
        
        # 如果不需要padding，直接返回
        if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
            return region
        
        if isinstance(region, torch.Tensor):
            # PyTorch padding格式：(left, right, top, bottom)
            padded = torch.nn.functional.pad(
                region, 
                (pad_left, pad_right, pad_top, pad_bottom),
                mode=self._get_torch_padding_mode(),
                value=self.config.constant_value if self.config.padding_mode == PaddingMode.CONSTANT else 0
            )
        else:
            # NumPy padding
            pad_width = (
                (0, 0),  # 通道维度不padding
                (pad_top, pad_bottom),  # 高度方向padding
                (pad_left, pad_right)   # 宽度方向padding
            )
            
            # 根据padding模式决定是否传递constant_values参数
            if self.config.padding_mode == PaddingMode.CONSTANT:
                padded = np.pad(
                    region,
                    pad_width,
                    mode=self._get_numpy_padding_mode(),
                    constant_values=self.config.constant_value
                )
            else:
                padded = np.pad(
                    region,
                    pad_width,
                    mode=self._get_numpy_padding_mode()
                )
        
        return padded
    
    def _get_torch_padding_mode(self) -> str:
        """获取PyTorch padding模式字符串"""
        mode_map = {
            PaddingMode.MIRROR: 'reflect',
            PaddingMode.CONSTANT: 'constant',
            PaddingMode.EDGE: 'replicate',
            PaddingMode.WRAP: 'circular'
        }
        return mode_map[self.config.padding_mode]
    
    def _get_numpy_padding_mode(self) -> str:
        """获取NumPy padding模式字符串"""
        mode_map = {
            PaddingMode.MIRROR: 'reflect',
            PaddingMode.CONSTANT: 'constant',
            PaddingMode.EDGE: 'edge',
            PaddingMode.WRAP: 'wrap'
        }
        return mode_map[self.config.padding_mode]
    
    def reconstruct_patch_positions(self, 
                                   patches: Union[np.ndarray, torch.Tensor],
                                   positions: List[PatchPosition],
                                   output_shape: Tuple[int, int, int]) -> Union[np.ndarray, torch.Tensor]:
        """Reconstruct patches to original positions (for visualization and debugging)
        
        Args:
            patches: Patches data [N,C,H,W]
            positions: List of position information
            output_shape: Output image shape (C,H,W)
            
        Returns:
            reconstructed: Reconstructed image
        """
        C, H, W = output_shape
        
        if isinstance(patches, torch.Tensor):
            reconstructed = torch.zeros((C, H, W), dtype=patches.dtype, device=patches.device)
        else:
            reconstructed = np.zeros((C, H, W), dtype=patches.dtype)
        
        for i, (patch, position) in enumerate(zip(patches, positions)):
            # Remove padding to get original extraction region
            patch_content = self._remove_padding(patch, position)
            
            # Place at corresponding position
            reconstructed[:, 
                         position.extract_y1:position.extract_y2, 
                         position.extract_x1:position.extract_x2] = patch_content
        
        return reconstructed
    
    def _remove_padding(self, 
                       patch: Union[np.ndarray, torch.Tensor], 
                       position: PatchPosition) -> Union[np.ndarray, torch.Tensor]:
        """
        从patch中移除padding，获得原始内容
        
        Args:
            patch: 带padding的patch [C,H,W]
            position: 位置信息
            
        Returns:
            content: 移除padding后的内容
        """
        # Calculate content region boundaries
        top = position.pad_top
        bottom = self.config.patch_size - position.pad_bottom if position.pad_bottom > 0 else self.config.patch_size
        left = position.pad_left
        right = self.config.patch_size - position.pad_right if position.pad_right > 0 else self.config.patch_size
        
        return patch[:, top:bottom, left:right]
    
    def get_patch_statistics(self, positions: List[PatchPosition]) -> Dict[str, float]:
        """Get patch extraction statistics
        
        Args:
            positions: List of position information
            
        Returns:
            stats: Statistics dictionary
        """
        if len(positions) == 0:
            return {}
        
        total_padding = sum(p.pad_left + p.pad_right + p.pad_top + p.pad_bottom for p in positions)
        avg_padding = total_padding / len(positions)
        
        padded_patches = sum(1 for p in positions if p.pad_left + p.pad_right + p.pad_top + p.pad_bottom > 0)
        
        stats = {
            'total_patches': len(positions),
            'padded_patches': padded_patches,
            'padding_ratio': padded_patches / len(positions),
            'avg_padding_pixels': avg_padding,
            'max_padding': max(p.pad_left + p.pad_right + p.pad_top + p.pad_bottom for p in positions),
        }
        
        return stats


def test_patch_extractor():
    """Simple test function"""
    # Create test data
    test_data = np.random.randn(7, 270, 480).astype(np.float32)
    
    # Create test patch information
    from .hole_detector import PatchInfo
    patch_infos = [
        PatchInfo(center_x=100, center_y=100, hole_area=500, patch_id=0),
        PatchInfo(center_x=400, center_y=200, hole_area=300, patch_id=1),
        PatchInfo(center_x=50, center_y=50, hole_area=200, patch_id=2),   # Boundary test
    ]
    
    # Create extractor
    extractor = PatchExtractor()
    
    # Extract patches
    patches, positions = extractor.extract_patches(test_data, patch_infos)
    
    print(f"Extracted {len(patches)} patches:")
    print(f"Patches shape: {patches.shape}")
    
    for i, (patch_info, position) in enumerate(zip(patch_infos, positions)):
        print(f"  Patch {i}: 中心=({position.center_x}, {position.center_y}), "
              f"padding=(T:{position.pad_top}, B:{position.pad_bottom}, L:{position.pad_left}, R:{position.pad_right})")
    
    # Statistics
    stats = extractor.get_patch_statistics(positions)
    print(f"Extraction statistics: {stats}")
    
    # Reconstruction test
    reconstructed = extractor.reconstruct_patch_positions(patches, positions, test_data.shape)
    print(f"Reconstructed image shape: {reconstructed.shape}")
    
    return patches, positions, stats


if __name__ == "__main__":
    # Run test
    patches, positions, stats = test_patch_extractor()
    print("SUCCESS: PatchExtractor test completed")