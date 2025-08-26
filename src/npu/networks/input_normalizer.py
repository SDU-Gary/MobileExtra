#!/usr/bin/env python3
"""
Unified Input Normalizer - Handles heterogeneous data normalization (RGB/Mask/MV)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class UnifiedInputNormalizer(nn.Module):
    """
    Unified Input Normalizer - HDR->LDR conversion for RGB, preserves masks/MV
    Strategy: HDR RGB -> LDR -> [0,1], masks/MV stay unchanged
    """
    
    def __init__(self, 
                 rgb_method: str = "hdr_to_ldr",      # HDR->LDR normalization
                 tone_mapping: str = "reinhard",      # Tone mapping method
                 normalize_masks: bool = False,       # Keep masks raw
                 normalize_mv: bool = False,          # Keep MV raw  
                 gamma: float = 2.2):                 # Gamma correction
        super().__init__()
        
        self.rgb_method = rgb_method
        self.tone_mapping = tone_mapping
        self.normalize_masks = normalize_masks
        self.normalize_mv = normalize_mv  
        self.gamma = gamma
        
        # Store normalization states
        self.register_buffer('masks_normalized', torch.tensor(normalize_masks))
        self.register_buffer('mv_normalized', torch.tensor(normalize_mv))
        
    def forward(self, x: torch.Tensor, update_stats: bool = True) -> torch.Tensor:
        """
        Process 7-channel input: warped_RGB(3) + holes(1) + occlusion(1) + residual_MV(2)
        
        Returns:
            [B,7,H,W] - RGB:[0,1] normalized LDR, Masks:[0,1] raw, MV: pixel offsets
        """
        
        processed = torch.zeros_like(x)
        
        # Process RGB: HDR -> LDR -> [0,1]
        rgb_channels = x[:, 0:3, :, :]
        processed[:, 0:3, :, :] = self._process_rgb(rgb_channels)
        
        # Process masks: keep [0,1] range
        holes_mask = x[:, 3:4, :, :]
        occlusion_mask = x[:, 4:5, :, :]
        processed[:, 3:4, :, :] = torch.clamp(holes_mask, 0.0, 1.0)
        processed[:, 4:5, :, :] = torch.clamp(occlusion_mask, 0.0, 1.0)
        
        # Process MV: preserve pixel offset values
        mv_channels = x[:, 5:7, :, :]
        if self.normalize_mv:
            processed[:, 5:7, :, :] = self._normalize_mv_legacy(mv_channels)
        else:
            processed[:, 5:7, :, :] = mv_channels
            if update_stats:
                mv_min, mv_max = mv_channels.min().item(), mv_channels.max().item()
                if abs(mv_max) > 500 or abs(mv_min) > 500:
                    print(f"MV range warning: [{mv_min:.1f}, {mv_max:.1f}]")
        
        return processed
    
    def _process_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        """HDR -> LDR conversion with tone mapping"""
        
        if torch.any(rgb < 0):
            print(f"HDR negative values clipped: [{rgb.min().item():.3f}, {rgb.max().item():.3f}]")
            rgb = torch.clamp(rgb, min=0.0)
        
        if self.rgb_method == "hdr_to_ldr":
            ldr_rgb = self._hdr_to_ldr(rgb, self.tone_mapping, self.gamma)
        else:
            ldr_rgb = torch.clamp(rgb, 0.0, 1.0)
        
        normalized_rgb = torch.clamp(ldr_rgb, 0.0, 1.0)
        
        rgb_min, rgb_max = normalized_rgb.min().item(), normalized_rgb.max().item()
        rgb_mean = normalized_rgb.mean().item()
        if rgb_max > 0.98 or rgb_mean < 0.02:
            print(f"RGB processing warning - range: [{rgb_min:.3f}, {rgb_max:.3f}], mean: {rgb_mean:.3f}")
        
        return normalized_rgb
    
    def _hdr_to_ldr(self, hdr_image: torch.Tensor, 
                    tone_mapping: str = "reinhard", 
                    gamma: float = 2.2) -> torch.Tensor:
        """HDR to LDR conversion with tone mapping"""
        
        hdr_positive = torch.clamp(hdr_image, min=0.0)
        
        if tone_mapping == "reinhard":
            ldr = hdr_positive / (1.0 + hdr_positive)
        elif tone_mapping == "aces":
            a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
            ldr = torch.clamp((hdr_positive * (a * hdr_positive + b)) / 
                            (hdr_positive * (c * hdr_positive + d) + e), 0.0, 1.0)
        elif tone_mapping == "exposure":
            exposure = 1.0
            ldr = 1.0 - torch.exp(-hdr_positive * exposure)
        elif tone_mapping == "filmic":
            ldr = hdr_positive / (hdr_positive + 1.0) * 1.02
            ldr = torch.clamp(ldr, 0.0, 1.0)
        else:  # "clamp"
            ldr = torch.clamp(hdr_positive, 0.0, 1.0)
        
        # Apply gamma correction
        if gamma != 1.0:
            ldr = torch.pow(torch.clamp(ldr, 1e-8, 1.0), 1.0 / gamma)
        
        return torch.clamp(ldr, 0.0, 1.0)
    
    def _normalize_mv_legacy(self, mv: torch.Tensor, mv_pixel_range: float = 100.0) -> torch.Tensor:
        """Legacy MV normalization method (backup usage only)"""
        mv_max = torch.max(torch.abs(mv)).item()
        if mv_max > mv_pixel_range * 2:
            print(f"MV range [{mv.min().item():.1f}, {mv.max().item():.1f}] exceeds expected ±{mv_pixel_range:.1f}")
        return torch.clamp(mv / mv_pixel_range, -1.0, 1.0)
    
    def denormalize_output(self, network_output: torch.Tensor) -> torch.Tensor:
        """Denormalize network output (handles [-1,1] to [0,1] conversion if needed)"""
        output_min, output_max = network_output.min().item(), network_output.max().item()
        
        if output_min >= -1.1 and output_max <= 1.1:
            # Network uses tanh activation, convert [-1,1] to [0,1]
            print("Network output in [-1,1] range, converting to [0,1]")
            return (network_output + 1.0) / 2.0
        else:
            return torch.clamp(network_output, 0.0, 1.0)
    
    def hdr_to_ldr_for_display(self, hdr_image: torch.Tensor, 
                              tone_mapping: str = "adaptive_reinhard", 
                              gamma: float = 1.8) -> torch.Tensor:
        """HDR to LDR conversion for TensorBoard display with adaptive exposure"""
        
        hdr_positive = torch.clamp(hdr_image, min=0.0)
        
        if tone_mapping == "adaptive_reinhard":
            # Adaptive exposure + Reinhard
            mean_luminance = hdr_positive.mean() + 1e-8
            exposure_factor = torch.clamp(0.5 / mean_luminance, 0.2, 8.0)
            exposed_hdr = hdr_positive * exposure_factor
            ldr = exposed_hdr / (1.0 + exposed_hdr)
            
            print(f"Adaptive Reinhard: exposure={exposure_factor.item():.2f}, orig_mean={mean_luminance.item():.4f}")
            
        elif tone_mapping == "log_compress":
            # Log compression to preserve details
            log_compressed = torch.log(1.0 + hdr_positive * 4.0) / torch.log(torch.tensor(5.0))
            ldr = log_compressed
            
            print(f"Log compress: output_range=[{ldr.min():.3f}, {ldr.max():.3f}]")
            
        elif tone_mapping == "reinhard":
            ldr = hdr_positive / (1.0 + hdr_positive)
        elif tone_mapping == "aces":
            a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
            ldr = torch.clamp((hdr_positive * (a * hdr_positive + b)) / 
                            (hdr_positive * (c * hdr_positive + d) + e), 0.0, 1.0)
        elif tone_mapping == "exposure":
            exposure = 2.0  # Increased exposure
            ldr = 1.0 - torch.exp(-hdr_positive * exposure)
        else:  # "clamp"
            ldr = torch.clamp(hdr_positive, 0.0, 1.0)
        
        # Gamma correction (lower gamma for higher brightness)
        if gamma != 1.0:
            ldr = torch.pow(torch.clamp(ldr, 1e-8, 1.0), 1.0 / gamma)
        
        # Brightness boost for certain tone mapping methods
        if tone_mapping in ["adaptive_reinhard", "log_compress"]:
            ldr = torch.clamp(ldr * 1.1, 0.0, 1.0)
        
        result = torch.clamp(ldr, 0.0, 1.0)
        
        # Diagnostic info (10% chance)
        if torch.rand(1).item() < 0.1:
            dark_pixels = (result < 0.1).sum().item() / result.numel() * 100
            print(f"HDR→LDR: method={tone_mapping}, dark_pixels={dark_pixels:.1f}%, avg_brightness={result.mean():.3f}")
        
        return result
    
    def prepare_for_tensorboard(self, image: torch.Tensor, 
                               data_type: str = "rgb", 
                               is_normalized: bool = False) -> torch.Tensor:
        """Prepare image data for TensorBoard display"""
        
        if data_type == "rgb":
            if is_normalized:
                return self.denormalize_output(image)
            else:
                return torch.clamp(image, 0.0, 1.0)
        elif data_type == "mask":
            return torch.clamp(image, 0.0, 1.0)
        elif data_type == "mv":
            return self._visualize_mv_for_tensorboard(image)
        else:
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                return (image - img_min) / (img_max - img_min)
            else:
                return torch.zeros_like(image)
    
    def _visualize_mv_for_tensorboard(self, mv: torch.Tensor) -> torch.Tensor:
        """Convert residual MV to pseudo-color visualization"""
        
        # Calculate motion magnitude and direction
        mv_magnitude = torch.sqrt(mv[:, 0:1]**2 + mv[:, 1:2]**2 + 1e-8)
        mv_angle = torch.atan2(mv[:, 1:2], mv[:, 0:1])  # [-π, π]
        
        # Normalize for visualization
        magnitude_max = torch.max(mv_magnitude)
        if magnitude_max > 1e-6:
            magnitude_norm = mv_magnitude / magnitude_max
        else:
            magnitude_norm = torch.zeros_like(mv_magnitude)
        
        angle_norm = (mv_angle + 3.14159) / (2 * 3.14159)  # [-π,π] → [0,1]
        
        # Create RGB visualization: R=magnitude, G=angle, B=constant
        B, _, H, W = mv.shape
        mv_vis = torch.zeros(B, 3, H, W, dtype=mv.dtype, device=mv.device)
        
        mv_vis[:, 0:1] = magnitude_norm  # R: motion magnitude
        mv_vis[:, 1:2] = angle_norm      # G: motion direction
        mv_vis[:, 2:3] = 0.5             # B: constant value
        
        return torch.clamp(mv_vis, 0.0, 1.0)
    
    def get_normalization_stats(self) -> dict:
        """Get normalization statistics"""
        stats = {
            'rgb_method': self.rgb_method,
            'tone_mapping': self.tone_mapping,
            'normalize_masks': self.normalize_masks,
            'normalize_mv': self.normalize_mv,
            'gamma': self.gamma,
            'masks_normalized': self.masks_normalized.item(),
            'mv_normalized': self.mv_normalized.item()
        }
        
        return stats


def create_input_normalizer(config: dict) -> UnifiedInputNormalizer:
    """Create input normalizer from config"""
    
    normalizer_config = config.get('input_normalizer', {})
    
    return UnifiedInputNormalizer(
        rgb_method=normalizer_config.get('rgb_method', 'hdr_to_ldr'),
        tone_mapping=normalizer_config.get('tone_mapping', 'reinhard'),
        normalize_masks=normalizer_config.get('normalize_masks', False),
        normalize_mv=normalizer_config.get('normalize_mv', False),
        gamma=normalizer_config.get('gamma', 2.2)
    )


if __name__ == "__main__":
    # Test input normalizer
    print("Testing unified input normalizer")
    
    batch_size, height, width = 1, 64, 64
    test_input = torch.zeros(batch_size, 7, height, width)
    
    # HDR RGB data
    test_input[:, 0:3, :, :] = torch.rand(batch_size, 3, height, width) * 5.0
    # Mask data [0,1]
    test_input[:, 3:5, :, :] = torch.rand(batch_size, 2, height, width)
    # MV data (pixel offsets)
    test_input[:, 5:7, :, :] = torch.randn(batch_size, 2, height, width) * 50.0
    
    print(f"Input ranges:")
    print(f"  HDR RGB: [{test_input[:, 0:3].min():.3f}, {test_input[:, 0:3].max():.3f}]")
    print(f"  Masks: [{test_input[:, 3:5].min():.3f}, {test_input[:, 3:5].max():.3f}]")
    print(f"  MV: [{test_input[:, 5:7].min():.1f}, {test_input[:, 5:7].max():.1f}]")
    
    normalizer = UnifiedInputNormalizer(
        rgb_method="hdr_to_ldr",
        tone_mapping="reinhard",
        normalize_masks=False,
        normalize_mv=False,
        gamma=2.2
    )
    
    processed = normalizer(test_input)
    
    print(f"\nProcessed ranges:")
    print(f"  LDR RGB: [{processed[:, 0:3].min():.3f}, {processed[:, 0:3].max():.3f}]")
    print(f"  Masks: [{processed[:, 3:5].min():.3f}, {processed[:, 3:5].max():.3f}]")
    print(f"  MV: [{processed[:, 5:7].min():.1f}, {processed[:, 5:7].max():.1f}]")
    
    # Test TensorBoard visualization
    rgb_vis = normalizer.prepare_for_tensorboard(processed[:, 0:3], "rgb")
    mask_vis = normalizer.prepare_for_tensorboard(processed[:, 3:4], "mask")
    mv_vis = normalizer.prepare_for_tensorboard(processed[:, 5:7], "mv")
    
    print(f"\nVisualization ranges:")
    print(f"  RGB: [{rgb_vis.min():.3f}, {rgb_vis.max():.3f}]")
    print(f"  Mask: [{mask_vis.min():.3f}, {mask_vis.max():.3f}]")
    print(f"  MV: [{mv_vis.min():.3f}, {mv_vis.max():.3f}]")
    
    # Test output denormalization
    test_output = torch.rand(batch_size, 3, height, width)
    denormalized = normalizer.denormalize_output(test_output)
    
    print(f"\nOutput denormalization:")
    print(f"  Network output: [{test_output.min():.3f}, {test_output.max():.3f}]")
    print(f"  Denormalized: [{denormalized.min():.3f}, {denormalized.max():.3f}]")
    
    print("\nInput normalizer test completed")
    print("Strategy: HDR RGB → LDR [0,1], preserve masks/MV ranges")