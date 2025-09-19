#!/usr/bin/env python3
"""Patch-based inpainting network for mobile frame interpolation hole filling."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass

try:
    from .patch import (
        HoleDetector, HoleDetectorConfig, PatchInfo,
        PatchExtractor, PatchExtractorConfig, PatchPosition, PaddingMode,
        PatchNetwork
    )
    from .mobile_inpainting_network import MobileInpaintingNetwork
except ImportError:
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    from patch import (
        HoleDetector, HoleDetectorConfig, PatchInfo,
        PatchExtractor, PatchExtractorConfig, PatchPosition, PaddingMode,
        PatchNetwork
    )
    from mobile_inpainting_network import MobileInpaintingNetwork


@dataclass
class PatchInpaintingConfig:
    """Configuration for patch-based inpainting network."""
    enable_patch_mode: bool = True
    hole_detector: HoleDetectorConfig = None
    patch_extractor: PatchExtractorConfig = None
    patch_network_channels: int = 24  #  REVERT: 回退到稳定的24通道
    fusion_mode: str = "weighted_replace"  # weighted_replace, alpha_blend
    fusion_feather_size: int = 4
    enable_color_correction: bool = True
    max_batch_size: int = 8
    enable_performance_stats: bool = False


class PatchFusion:
    """Fuses repaired patches back into full image with feathering."""
    
    def __init__(self, config: PatchInpaintingConfig):
        self.config = config
        self.feather_size = config.fusion_feather_size
    
    def fuse_patches(self, 
                    original_image: torch.Tensor,
                    repaired_patches: torch.Tensor,
                    positions: List[PatchPosition],
                    holes_mask: torch.Tensor) -> torch.Tensor:
        """Fuse repaired patches into original image."""
        if len(positions) == 0:
            return original_image
        
        B, C, H, W = original_image.shape
        result = original_image.clone()
        
        # Process each patch
        for i, (patch, position) in enumerate(zip(repaired_patches, positions)):
            x1, y1 = position.extract_x1, position.extract_y1
            x2, y2 = position.extract_x2, position.extract_y2
            
            patch_content = self._remove_padding(patch, position)
            hole_region = holes_mask[0, 0, y1:y2, x1:x2]
            
            if self.config.fusion_mode == "weighted_replace":
                fused_region = self._weighted_replace_fusion(
                    result[0, :, y1:y2, x1:x2],
                    patch_content,
                    hole_region
                )
            else:
                fused_region = self._alpha_blend_fusion(
                    result[0, :, y1:y2, x1:x2], 
                    patch_content, 
                    hole_region
                )
            
            result[0, :, y1:y2, x1:x2] = fused_region
        
        if self.config.enable_color_correction:
            result = self._apply_color_correction(result, original_image, holes_mask)
        
        return result
    
    def _remove_padding(self, patch: torch.Tensor, position: PatchPosition) -> torch.Tensor:
        """移除patch的padding，获得有效内容"""
        # 计算有效区域边界
        top = position.pad_top
        bottom = 128 - position.pad_bottom if position.pad_bottom > 0 else 128
        left = position.pad_left  
        right = 128 - position.pad_right if position.pad_right > 0 else 128
        
        return patch[:, top:bottom, left:right]
    
    def _weighted_replace_fusion(self, 
                                original: torch.Tensor, 
                                patch: torch.Tensor, 
                                hole_mask: torch.Tensor) -> torch.Tensor:
        """加权替换融合"""
        # 创建羽化掩码
        feather_mask = self._create_feather_mask(hole_mask, self.feather_size)
        
        # 扩展掩码到3通道
        feather_mask_3d = feather_mask.unsqueeze(0).expand(3, -1, -1)  # [3,H,W]
        
        # 加权融合
        fused = original * (1 - feather_mask_3d) + patch * feather_mask_3d
        
        return fused
    
    def _alpha_blend_fusion(self, 
                           original: torch.Tensor, 
                           patch: torch.Tensor, 
                           hole_mask: torch.Tensor) -> torch.Tensor:
        """Alpha blend fusion."""
        alpha = hole_mask.unsqueeze(0).expand(3, -1, -1)
        
        fused = original * (1 - alpha) + patch * alpha
        
        return fused
    
    def _create_feather_mask(self, mask: torch.Tensor, feather_size: int) -> torch.Tensor:
        """Create feathered mask using Gaussian blur."""
        if feather_size <= 0:
            return mask.float()
        
        from torch.nn.functional import conv2d
        
        kernel_size = feather_size * 2 + 1
        sigma = feather_size / 3.0
        
        x = torch.arange(kernel_size, dtype=torch.float32) - feather_size
        gaussian_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gaussian_1d /= gaussian_1d.sum()
        
        gaussian_2d = gaussian_1d.view(-1, 1) * gaussian_1d.view(1, -1)
        gaussian_2d = gaussian_2d.unsqueeze(0).unsqueeze(0)
        
        if mask.device != gaussian_2d.device:
            gaussian_2d = gaussian_2d.to(mask.device)
        
        mask_expanded = mask.unsqueeze(0).unsqueeze(0).float()
        feathered = conv2d(mask_expanded, gaussian_2d, padding=feather_size)
        
        return feathered.squeeze(0).squeeze(0)
    
    def _apply_color_correction(self, 
                               fused_image: torch.Tensor, 
                               original_image: torch.Tensor, 
                               holes_mask: torch.Tensor) -> torch.Tensor:
        """Apply color correction using statistics matching."""
        non_hole_mask = (1 - holes_mask).expand_as(original_image)
        
        if non_hole_mask.sum() > 0:
            orig_mean = (original_image * non_hole_mask).sum(dim=[2, 3], keepdim=True) / non_hole_mask.sum(dim=[2, 3], keepdim=True)
            orig_std = torch.sqrt(((original_image - orig_mean) ** 2 * non_hole_mask).sum(dim=[2, 3], keepdim=True) / non_hole_mask.sum(dim=[2, 3], keepdim=True))
            
            fused_mean = (fused_image * non_hole_mask).sum(dim=[2, 3], keepdim=True) / non_hole_mask.sum(dim=[2, 3], keepdim=True)
            fused_std = torch.sqrt(((fused_image - fused_mean) ** 2 * non_hole_mask).sum(dim=[2, 3], keepdim=True) / non_hole_mask.sum(dim=[2, 3], keepdim=True))
            
            hole_region = fused_image * holes_mask.expand_as(fused_image)
            corrected_hole = (hole_region - fused_mean) * (orig_std / (fused_std + 1e-8)) + orig_mean
            
            corrected_image = fused_image * (1 - holes_mask.expand_as(fused_image)) + corrected_hole * holes_mask.expand_as(fused_image)
            
            return corrected_image
        
        return fused_image


class PatchBasedInpainting(nn.Module):
    """Patch-based inpainting network with automatic hole detection and processing."""
    
    def __init__(self, 
                 input_channels: int = 7, 
                 output_channels: int = 3,
                 config: Optional[PatchInpaintingConfig] = None):
        super(PatchBasedInpainting, self).__init__()
        
        self.config = config or PatchInpaintingConfig()
        
        # Initialize sub-configurations
        if self.config.hole_detector is None:
            self.config.hole_detector = HoleDetectorConfig()
        if self.config.patch_extractor is None:
            self.config.patch_extractor = PatchExtractorConfig()
        
        # Create core components
        if self.config.enable_patch_mode:
            self.hole_detector = HoleDetector(self.config.hole_detector)
            self.patch_extractor = PatchExtractor(self.config.patch_extractor)
            self.patch_network = PatchNetwork(
                input_channels=input_channels,
                output_channels=output_channels,
                base_channels=self.config.patch_network_channels
            )
            self.patch_fusion = PatchFusion(self.config)
        
        # Performance statistics
        self.performance_stats = {
            'patch_mode_count': 0,
            'total_patches_processed': 0,
            'avg_patches_per_image': 0.0
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using patch-based processing.
        
        Args:
            x: Input tensor [B,7,H,W] - WarpedRGB(3) + holes(1) + occlusion(1) + ResidualMV(2)
        Returns:
            Repaired RGB image [B,3,H,W]
        """
        B, C, H, W = x.shape
        
        holes_mask = x[:, 3:4, :, :]
        return self._patch_forward(x, holes_mask)
    
    def _patch_forward(self, x: torch.Tensor, holes_mask: torch.Tensor) -> torch.Tensor:
        """Patch模式前向传播 - 纯patch处理"""
        B, C, H, W = x.shape
        
        # 当前只处理batch_size=1的情况，后续可扩展
        if B != 1:
            # 对于batch size > 1，处理第一个样本
            # TODO: 未来可以支持真正的batch处理
            x = x[:1]
            holes_mask = holes_mask[:1] 
            B = 1
        
        # Hole detection
        holes_mask_np = holes_mask[0, 0].cpu().numpy()
        patch_infos = self.hole_detector.detect_patch_centers(holes_mask_np)
        
        if len(patch_infos) == 0:
            # No valid holes, create default center patch to ensure processing
            center_x = W // 2
            center_y = H // 2
            patch_infos = [PatchInfo(center_x=center_x, center_y=center_y, hole_area=1000, patch_id=0)]
        
        # Extract patches
        patches, positions = self.patch_extractor.extract_patches(x[0], patch_infos)
        
        if isinstance(patches, np.ndarray):
            patches = torch.from_numpy(patches).to(x.device)
        
        # Process patches in batches
        repaired_patches = self._process_patches_in_batches(patches)
        
        # Fuse patches
        fused_result = self.patch_fusion.fuse_patches(
            x[:, :3],  # Only use RGB channels as original image
            repaired_patches,
            positions,
            holes_mask
        )
        
        # Update statistics
        self.performance_stats['patch_mode_count'] += 1
        self.performance_stats['total_patches_processed'] += len(patch_infos)
        self.performance_stats['avg_patches_per_image'] = (
            self.performance_stats['total_patches_processed'] / 
            max(self.performance_stats['patch_mode_count'], 1)
        )
        
        return fused_result
    
    def _process_patches_in_batches(self, patches: torch.Tensor) -> torch.Tensor:
        """Process patches in batches to avoid memory overflow."""
        N, C, H, W = patches.shape
        batch_size = self.config.max_batch_size
        
        repaired_patches = []
        
        for i in range(0, N, batch_size):
            batch_end = min(i + batch_size, N)
            batch_patches = patches[i:batch_end]
            
            with torch.no_grad() if not self.training else torch.enable_grad():
                batch_repaired = self.patch_network(batch_patches)
                repaired_patches.append(batch_repaired)
        
        return torch.cat(repaired_patches, dim=0)
    
    def get_performance_stats(self) -> Dict[str, Union[int, float]]:
        """Get performance statistics."""
        total_calls = self.performance_stats['patch_mode_count']
        
        stats = self.performance_stats.copy()
        stats['total_forward_calls'] = total_calls
        stats['patch_mode_ratio'] = 1.0  # Only patch mode now
        
        return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'patch_mode_count': 0,
            'total_patches_processed': 0,
            'avg_patches_per_image': 0.0
        }
    
    def set_patch_mode(self, enabled: bool):
        """Dynamically toggle patch mode."""
        self.config.enable_patch_mode = enabled
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        info = {
            'patch_mode_enabled': self.config.enable_patch_mode,
            'patch_network_params': self.patch_network.get_parameter_count() if self.config.enable_patch_mode else None,
        }
        
        if self.config.enable_patch_mode:
            patch_info = self.patch_network.get_model_info()
            info.update({
                'patch_network_info': patch_info,
                'compression_ratio': patch_info['compression_ratio']
            })
        
        return info


def test_patch_based_inpainting():
    """Test PatchBasedInpainting network."""
    # Create test data
    batch_size = 1
    test_input = torch.randn(batch_size, 7, 270, 480)
    
    # Create meaningful hole mask
    holes_mask = torch.zeros(batch_size, 1, 270, 480)
    holes_mask[0, 0, 50:100, 100:200] = 1  # Add rectangular hole
    holes_mask[0, 0, 150:180, 300:350] = 1  # Add another hole
    test_input[:, 3:4, :, :] = holes_mask
    
    # Create configuration and network
    config = PatchInpaintingConfig(
        enable_patch_mode=True,
        enable_performance_stats=True
    )
    network = PatchBasedInpainting(config=config)
    
    # Forward pass
    with torch.no_grad():
        output = network(test_input)
    
    # Get info
    stats = network.get_performance_stats()
    model_info = network.get_model_info()
    
    print("=== PatchBasedInpainting Test Results ===")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Patch mode used: {stats['patch_mode_count'] > 0}")
    print(f"Patches processed: {stats['total_patches_processed']}")
    print(f"Avg patches/image: {stats['avg_patches_per_image']:.1f}")
    
    if model_info['patch_network_info']:
        print(f"Patch network params: {model_info['patch_network_info']['total_parameters']:,}")
        print(f"Compression ratio: {model_info['patch_network_info']['compression_ratio']:.1f}x")
    
    return network, output, stats


if __name__ == "__main__":
    network, output, stats = test_patch_based_inpainting()
    print("SUCCESS: PatchBasedInpainting test completed")