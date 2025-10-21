#!/usr/bin/env python3
"""Patch-based inpainting network for mobile frame interpolation hole filling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    residual_scale_factor: float = 1.0

    # Input normalization (match training: per-patch scalar scale on RGB)
    enable_input_normalization: bool = True
    norm_method: str = "per_patch_percentile"  # reserved for future
    norm_percentile: float = 0.99
    norm_min_scale: float = 1.0
    norm_max_scale: float = 512.0

    # Global standardization (single mu, sigma)
    enable_global_standardization: bool = True
    gs_mu: float = 0.0
    gs_sigma: float = 1.0
    gs_apply_to_channels: str = "rgb"
    # Unified normalization selection
    normalization_type: str = "global"  # options: none|per_patch|global|log
    log_epsilon: float = 1.0e-8
    log_delta_scale: float = 0.1
    log_delta_abs_max: float = 0.0
    log_delta_alpha: float = 1.0
    log_apply_delta_in_holes: bool = True
    log_delta_mask_ring_scale: float = 0.5
    log_delta_mask_ring_kernel: int = 3
    log_dither_enable: bool = False
    log_dither_scale: float = 1.0e-6


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
                base_channels=self.config.patch_network_channels,
                residual_scale_factor=self.config.residual_scale_factor
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
                scale = torch.as_tensor(self.config.residual_scale_factor, dtype=batch_patches.dtype, device=batch_patches.device)
                # Inference: prefer global standardization if enabled; else optional per-patch normalization
                if getattr(self.config, 'normalization_type', 'global') == 'global' and getattr(self.config, 'enable_global_standardization', True):
                    warped_rgb = batch_patches[:, :3]
                    mu = torch.as_tensor(self.config.gs_mu, dtype=warped_rgb.dtype, device=warped_rgb.device)
                    sigma = torch.as_tensor(self.config.gs_sigma, dtype=warped_rgb.dtype, device=warped_rgb.device)
                    Xn = (warped_rgb - mu) / torch.clamp(sigma, min=1e-6)
                    batch_patches_norm = batch_patches.clone()
                    batch_patches_norm[:, :3] = Xn
                    batch_residual_norm = self.patch_network(batch_patches_norm)
                    repaired = (Xn + batch_residual_norm * scale) * sigma + mu
                elif getattr(self.config, 'normalization_type', '') == 'per_patch' and getattr(self.config, 'enable_input_normalization', False):
                    warped_rgb = batch_patches[:, :3]
                    B = warped_rgb.shape[0]
                    # Compute per-patch scale b using percentile; fallback to amax if needed
                    try:
                        q = torch.quantile(warped_rgb.view(B, -1), self.config.norm_percentile, dim=1)
                    except Exception:
                        q = warped_rgb.view(B, -1).amax(dim=1)
                    b = q.view(-1, 1, 1, 1)
                    b = torch.clamp(b, min=self.config.norm_min_scale, max=self.config.norm_max_scale)

                    # Normalize RGB channels; keep masks/MV unchanged
                    batch_patches_norm = batch_patches.clone()
                    batch_patches_norm[:, :3] = batch_patches_norm[:, :3] / b

                    # Predict residual in normalized domain
                    batch_residual_norm = self.patch_network(batch_patches_norm)

                    # Reconstruct in normalized domain, then de-normalize back to HDR
                    repaired = (warped_rgb / b + batch_residual_norm * scale) * b
                elif getattr(self.config, 'normalization_type', '') == 'log':
                    warped_rgb = batch_patches[:, :3]
                    eps = torch.as_tensor(self.config.log_epsilon, dtype=warped_rgb.dtype, device=warped_rgb.device)
                    warped_pos = torch.clamp(warped_rgb, min=0.0)
                    if getattr(self.config, 'log_dither_enable', False):
                        scale = float(getattr(self.config, 'log_dither_scale', 1.0e-6))
                        noise = (torch.rand_like(warped_pos) - 0.5) * scale
                        warped_pos = torch.clamp(warped_pos + noise, min=0.0)
                    log_img = torch.log(warped_pos + eps)
                    B = warped_rgb.shape[0]
                    min_log = torch.amin(log_img.view(B, -1), dim=1).view(-1, 1, 1, 1)
                    max_log = torch.amax(log_img.view(B, -1), dim=1).view(-1, 1, 1, 1)
                    denom = torch.clamp(max_log - min_log, min=1e-6)
                    Xn = (log_img - min_log) / denom
                    batch_patches_norm = batch_patches.clone()
                    batch_patches_norm[:, :3] = Xn
                    batch_delta_log = self.patch_network(batch_patches_norm)

                    if float(getattr(self.config, 'log_delta_abs_max', 0.0)) > 0.0:
                        alpha = float(getattr(self.config, 'log_delta_alpha', 1.0))
                        delta_log = alpha * torch.tanh(batch_delta_log) * float(self.config.log_delta_abs_max) * scale
                    else:
                        beta = float(getattr(self.config, 'log_delta_scale', 0.1))
                        delta_log = (beta * denom) * torch.tanh(batch_delta_log) * scale

                    if getattr(self.config, 'log_apply_delta_in_holes', True):
                        holes_mask = torch.clamp(batch_patches[:, 3:4], 0.0, 1.0)
                        if holes_mask.shape[2:] != log_img.shape[2:]:
                            holes_mask = F.interpolate(holes_mask, size=log_img.shape[2:], mode='nearest')
                        try:
                            kernel_size = int(getattr(self.config, 'log_delta_mask_ring_kernel', 3))
                            if kernel_size % 2 == 0:
                                kernel_size += 1
                            pad = kernel_size // 2
                            ring = F.max_pool2d(holes_mask, kernel_size=kernel_size, stride=1, padding=pad) - holes_mask
                            ring = torch.clamp(ring, 0.0, 1.0)
                        except Exception:
                            ring = torch.zeros_like(holes_mask)
                        scale = float(getattr(self.config, 'log_delta_mask_ring_scale', 0.5))
                        mask_weight = torch.clamp(holes_mask + scale * ring, 0.0, 1.0)
                        delta_log = delta_log * mask_weight

                    log_hat = log_img + delta_log
                    repaired = torch.exp(log_hat) - eps
                else:
                    # Original path: direct residual prediction in HDR domain
                    batch_residual = self.patch_network(batch_patches) * scale
                    try:
                        from train.residual_learning_helper import ResidualLearningHelper
                    except Exception:
                        import sys, os
                        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'train'))
                        from residual_learning_helper import ResidualLearningHelper
                    repaired = ResidualLearningHelper.reconstruct_from_residual(batch_patches[:, :3], batch_residual)

                repaired_patches.append(repaired)

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
