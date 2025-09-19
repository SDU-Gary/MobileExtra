#!/usr/bin/env python3
"""
Simple Grid Patch Extractor - 简单可靠的4x4网格切分策略

为训练提供简单、可预测、无复杂性的patch提取方案：
- 将完整图像切分为4x4网格 (16个patch)
- 每个patch大小: 270x480 (适配1080x1920图像)
- 跳过复杂的hole detection和动态patch策略
- 提供最大的训练稳定性和可预测性
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# 兼容现有结构
try:
    import sys
    sys.path.append('./src/npu/networks/patch')
    from patch_extractor import PatchPosition
except ImportError:
    @dataclass
    class PatchPosition:
        """Patch position information"""
        x: int
        y: int
        width: int
        height: int
        patch_id: int


@dataclass
class SimpleGridConfig:
    """Simple grid patch extractor configuration"""
    
    # Grid parameters
    grid_rows: int = 4                  # 短边分割数量
    grid_cols: int = 4                  # 长边分割数量
    
    # Expected input size (1080x1920 → 270x480 per patch)
    expected_height: int = 1080         # 预期输入图像高度
    expected_width: int = 1920          # 预期输入图像宽度
    
    # Patch size (automatically calculated)
    patch_height: int = 270             # 每个patch高度 (1080/4)
    patch_width: int = 480              # 每个patch宽度 (1920/4)
    
    # Padding strategy for non-standard sizes
    padding_mode: str = 'reflect'       # 'reflect', 'edge', 'constant'
    
    # Debug and validation
    enable_size_validation: bool = True # 验证输入尺寸
    enable_debug_info: bool = False     # 打印debug信息


class SimplePatchExtractor:
    """
    Simple Grid Patch Extractor
    
    将完整图像简单切分为4x4网格，提供：
    1. 完全可预测的patch数量 (固定16个)
    2. 均匀的图像覆盖 (无遗漏、无重叠)  
    3. 极简的实现 (无复杂算法)
    4. 最大的训练稳定性
    
    适用场景：
    - 训练阶段需要稳定可靠的patch提取
    - 不依赖hole detection或动态策略
    - 需要快速、可预测的处理
    """
    
    def __init__(self, config: Optional[SimpleGridConfig] = None):
        """Initialize simple patch extractor"""
        self.config = config or SimpleGridConfig()
        
        # Pre-compute grid parameters
        self.total_patches = self.config.grid_rows * self.config.grid_cols
        
        if self.config.enable_debug_info:
            print(f"🔧 SimplePatchExtractor initialized:")
            print(f"   Grid: {self.config.grid_rows}x{self.config.grid_cols} = {self.total_patches} patches")
            print(f"   Expected input: {self.config.expected_height}x{self.config.expected_width}")
            print(f"   Patch size: {self.config.patch_height}x{self.config.patch_width}")
    
    def extract_patches(self, 
                       image_data: np.ndarray, 
                       holes_mask: Optional[np.ndarray] = None) -> Tuple[List[np.ndarray], List[PatchPosition]]:
        """
        Extract 4x4 grid patches from complete image
        
        Args:
            image_data: Complete image [H, W, C] or [C, H, W]
            holes_mask: Holes mask (optional, will be split accordingly)
            
        Returns:
            patches: List of 16 image patches
            positions: List of 16 corresponding patch positions
        """
        # Input validation and format conversion
        image_data = self._validate_and_prepare_input(image_data)
        H, W = image_data.shape[:2]
        
        if self.config.enable_debug_info:
            print(f"📊 Processing image: {H}x{W}")
        
        # Calculate actual patch dimensions
        patch_h = H // self.config.grid_rows
        patch_w = W // self.config.grid_cols
        
        patches = []
        positions = []
        
        # Extract grid patches
        for row in range(self.config.grid_rows):
            for col in range(self.config.grid_cols):
                # Calculate patch coordinates
                y_start = row * patch_h
                y_end = y_start + patch_h
                x_start = col * patch_w  
                x_end = x_start + patch_w
                
                # Handle potential size mismatch for last row/col
                if row == self.config.grid_rows - 1:
                    y_end = H  # Include any remaining pixels
                if col == self.config.grid_cols - 1:
                    x_end = W  # Include any remaining pixels
                
                # Extract patch
                if len(image_data.shape) == 3:  # [H, W, C]
                    patch = image_data[y_start:y_end, x_start:x_end, :]
                else:  # [H, W]
                    patch = image_data[y_start:y_end, x_start:x_end]
                
                # Create position info
                patch_id = row * self.config.grid_cols + col
                position = PatchPosition(
                    x=x_start,
                    y=y_start, 
                    width=x_end - x_start,
                    height=y_end - y_start,
                    patch_id=patch_id
                )
                
                patches.append(patch)
                positions.append(position)
        
        if self.config.enable_debug_info:
            print(f"✅ Extracted {len(patches)} patches")
            for i, pos in enumerate(positions[:4]):  # Show first 4
                print(f"   Patch {i}: ({pos.x}, {pos.y}) {pos.width}x{pos.height}")
            if len(positions) > 4:
                print(f"   ... and {len(positions)-4} more")
        
        return patches, positions
    
    def extract_patches_with_masks(self,
                                  image_data: np.ndarray,
                                  holes_mask: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], List[PatchPosition]]:
        """
        Extract patches along with corresponding hole masks
        
        Args:
            image_data: Complete image [H, W, C]
            holes_mask: Holes mask [H, W] 
            
        Returns:
            image_patches: List of image patches
            mask_patches: List of corresponding hole mask patches  
            positions: List of patch positions
        """
        # Extract image patches
        image_patches, positions = self.extract_patches(image_data)
        
        # Extract corresponding mask patches
        if holes_mask is not None:
            mask_patches, _ = self.extract_patches(holes_mask)
        else:
            # Create dummy masks if not provided
            mask_patches = [np.zeros((pos.height, pos.width), dtype=np.uint8) for pos in positions]
        
        return image_patches, mask_patches, positions
    
    def _validate_and_prepare_input(self, image_data: np.ndarray) -> np.ndarray:
        """Validate and prepare input image"""
        if image_data is None:
            raise ValueError("Input image_data cannot be None")
        
        # Convert torch tensor if needed
        if torch.is_tensor(image_data):
            image_data = image_data.detach().cpu().numpy()
        
        # Ensure numpy array
        image_data = np.asarray(image_data)
        
        # Handle different input formats
        if len(image_data.shape) == 4:  # [B, C, H, W] or [B, H, W, C]
            if image_data.shape[0] == 1:
                image_data = image_data[0]  # Remove batch dimension
            else:
                raise ValueError(f"Batch processing not supported, got shape {image_data.shape}")
        
        # Convert [C, H, W] to [H, W, C] if needed
        # 🔧 FIX: 支持7通道输入 (warped RGB + holes + occlusion + motion vectors)
        if len(image_data.shape) == 3 and image_data.shape[0] in [1, 3, 4, 7]:  # Support 7-channel input
            # 🔧 FIX: 修复判断逻辑 - 7通道时[7,1080,1920]应该转换为[1080,1920,7]
            # 通道数通常 <= 7，高度和宽度通常 > 100
            if image_data.shape[0] <= 7 and image_data.shape[1] > 100 and image_data.shape[2] > 100:  # [C, H, W]
                image_data = np.transpose(image_data, (1, 2, 0))
        
        # Size validation
        if self.config.enable_size_validation:
            H, W = image_data.shape[:2]
            if H != self.config.expected_height or W != self.config.expected_width:
                print(f"⚠️  Size mismatch: got {H}x{W}, expected {self.config.expected_height}x{self.config.expected_width}")
                print(f"    Patches will be: {H//self.config.grid_rows}x{W//self.config.grid_cols}")
        
        return image_data
    
    def get_patch_info(self) -> Dict:
        """Get information about patch configuration"""
        return {
            'total_patches': self.total_patches,
            'grid_size': f"{self.config.grid_rows}x{self.config.grid_cols}",
            'patch_size': f"{self.config.patch_height}x{self.config.patch_width}",
            'expected_input_size': f"{self.config.expected_height}x{self.config.expected_width}",
            'extraction_type': 'simple_grid',
            'predictable': True,
            'coverage': '100%'
        }


def create_default_config() -> SimpleGridConfig:
    """Create default configuration for common use cases"""
    return SimpleGridConfig(
        grid_rows=4,
        grid_cols=4,
        expected_height=1080,
        expected_width=1920,
        patch_height=270,
        patch_width=480,
        enable_size_validation=True,
        enable_debug_info=False
    )


if __name__ == "__main__":
    # Test the simple patch extractor
    print("🧪 Testing SimplePatchExtractor")
    
    # Create test image (1080x1920x3)
    test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    test_holes = np.random.randint(0, 2, (1080, 1920), dtype=np.uint8)
    
    # Test extractor
    extractor = SimplePatchExtractor(create_default_config())
    
    patches, positions = extractor.extract_patches(test_image)
    
    print(f"✅ Test completed:")
    print(f"   Input: {test_image.shape}")
    print(f"   Patches extracted: {len(patches)}")
    print(f"   Patch sizes: {[p.shape for p in patches[:3]]}...")  # Show first 3
    
    # Test with masks
    img_patches, mask_patches, positions = extractor.extract_patches_with_masks(test_image, test_holes)
    print(f"   Mask patches: {len(mask_patches)}")
    
    # Show patch info
    info = extractor.get_patch_info()
    print(f"   Info: {info}")
    
    print("✅ SimplePatchExtractor ready for training!")