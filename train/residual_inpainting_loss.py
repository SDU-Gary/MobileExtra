#!/usr/bin/env python3
"""
Residual Inpainting Loss Functions
Specialized for residual MV-guided selective inpainting tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
import torchvision.models as models


class VGGFeatureExtractor(nn.Module):
    """VGG16-based feature extractor for perceptual loss."""
    
    def __init__(self, feature_layers: list = [1, 6, 11, 20, 29]):
        super().__init__()
        
        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        
        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.feature_layers = feature_layers
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input from [-1,1] to VGG range."""
        x = (x + 1.0) / 2.0  # [-1,1] -> [0,1]
        return (x - self.mean) / self.std
    
    def forward(self, x: torch.Tensor) -> list:
        """Extract multi-layer VGG features."""
        x = self.normalize_input(x)
        
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        
        return features


class ResidualInpaintingLoss(nn.Module):
    """Residual inpainting loss for selective hole filling."""
    
    def __init__(self, device: torch.device, config: Optional[Dict] = None):
        super().__init__()
        
        self.device = device
        
        # Loss component weights
        default_weights = {
            'residual_mse': 1.0,        # Core residual MSE
            'residual_l1': 0.5,         # Residual L1
            'spatial_weighted': 0.8,    # Focus on repair regions
            'preservation': 0.3,        # Preserve non-repair areas
            'edge_preservation': 0.2,   # Edge consistency
            'perceptual': 0.1,         # VGG perceptual
            'attention_supervision': 0.1 # Attention sparsity
        }
        
        # Load config weights
        if config:
            loss_config = config.get('loss', {})
            for key in default_weights:
                if key + '_weight' in loss_config:
                    default_weights[key] = loss_config[key + '_weight']
        
        self.loss_weights = default_weights
        
        # Core loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Sobel edge kernels
        self.register_buffer('sobel_x', torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ], dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1, 1))
        
        self.register_buffer('sobel_y', torch.tensor([
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        ], dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1, 1))
        
        # VGG perceptual extractor
        try:
            self.vgg_extractor = VGGFeatureExtractor()
            self.vgg_available = True
        except Exception as e:
            print(f"[WARN] VGG init failed, fallback to SSIM: {e}")
            self.vgg_available = False
            self.register_buffer('ssim_window', self._create_ssim_window(11, 3))
        
        # Device setup
        self.to(device)
    
    def _create_ssim_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Create Gaussian window for SSIM."""
        sigma = 1.5
        gauss = torch.Tensor([
            np.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _compute_residual_mse_loss(self, predicted_output: torch.Tensor, 
                                  target: torch.Tensor, 
                                  warped_rgb: torch.Tensor) -> torch.Tensor:
        """MSE loss on prediction residuals."""
        target_residual = target - warped_rgb
        predicted_residual = predicted_output - warped_rgb
        
        return self.mse_loss(predicted_residual, target_residual)
    
    def _compute_residual_l1_loss(self, predicted_output: torch.Tensor, 
                                 target: torch.Tensor, 
                                 warped_rgb: torch.Tensor) -> torch.Tensor:
        """L1 loss on prediction residuals."""
        target_residual = target - warped_rgb
        predicted_residual = predicted_output - warped_rgb
        
        return self.l1_loss(predicted_residual, target_residual)
    
    def _compute_spatial_weighted_loss(self, predicted_output: torch.Tensor, 
                                     target: torch.Tensor, 
                                     spatial_attention: torch.Tensor) -> torch.Tensor:
        """Spatially weighted loss focusing on repair regions."""
        # Weight by repair attention
        weighted_pred = predicted_output * spatial_attention
        weighted_target = target * spatial_attention
        
        return self.l1_loss(weighted_pred, weighted_target)
    
    def _compute_preservation_loss(self, predicted_output: torch.Tensor, 
                                 warped_rgb: torch.Tensor, 
                                 spatial_attention: torch.Tensor) -> torch.Tensor:
        """Preservation loss for non-repair regions."""
        preservation_mask = 1.0 - spatial_attention
        
        preserved_pred = predicted_output * preservation_mask
        preserved_original = warped_rgb * preservation_mask
        
        return self.l1_loss(preserved_pred, preserved_original)
    
    def _compute_edge_preservation_loss(self, predicted_output: torch.Tensor, 
                                      target: torch.Tensor) -> torch.Tensor:
        """Edge-aware loss using Sobel gradients."""
        # Sobel kernel device check
        sobel_x = self.sobel_x.to(predicted_output.device)
        sobel_y = self.sobel_y.to(predicted_output.device)
        
        # Edge computation
        pred_edges_x = F.conv2d(predicted_output, sobel_x, padding=1, groups=3)
        pred_edges_y = F.conv2d(predicted_output, sobel_y, padding=1, groups=3)
        pred_edges = torch.sqrt(pred_edges_x**2 + pred_edges_y**2 + 1e-8)
        
        target_edges_x = F.conv2d(target, sobel_x, padding=1, groups=3)
        target_edges_y = F.conv2d(target, sobel_y, padding=1, groups=3)
        target_edges = torch.sqrt(target_edges_x**2 + target_edges_y**2 + 1e-8)
        
        return self.l1_loss(pred_edges, target_edges)
    
    def _compute_ssim_loss(self, predicted_output: torch.Tensor, 
                          target: torch.Tensor) -> torch.Tensor:
        """Structural similarity loss."""
        # Window device check
        window = self.ssim_window.to(predicted_output.device)
        
        # Mean computation
        mu1 = F.conv2d(predicted_output, window, padding=5, groups=3)
        mu2 = F.conv2d(target, window, padding=5, groups=3)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Variance & covariance
        sigma1_sq = F.conv2d(predicted_output * predicted_output, window, padding=5, groups=3) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=5, groups=3) - mu2_sq
        sigma12 = F.conv2d(predicted_output * target, window, padding=5, groups=3) - mu1_mu2
        
        # SSIM stabilization
        C1 = 0.01**2
        C2 = 0.03**2
        
        # SSIM calculation
        ssim = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim.mean()
    
    def _compute_vgg_perceptual_loss(self, predicted_output: torch.Tensor, 
                                   target: torch.Tensor) -> torch.Tensor:
        """VGG-based perceptual loss."""
        if not self.vgg_available:
            # SSIM fallback
            return self._compute_ssim_loss(predicted_output, target)
        
        try:
            # Feature extraction
            pred_features = self.vgg_extractor(predicted_output)
            target_features = self.vgg_extractor(target)
            
            # Multi-layer MSE
            perceptual_loss = 0.0
            layer_weights = [1.0, 1.0, 1.0, 1.0, 1.0]  # Per-layer weights
            
            for i, (pred_feat, target_feat) in enumerate(zip(pred_features, target_features)):
                if i < len(layer_weights):
                    layer_loss = self.mse_loss(pred_feat, target_feat)
                    perceptual_loss += layer_weights[i] * layer_loss
            
            return perceptual_loss
            
        except Exception as e:
            print(f"[WARN] VGG failed, SSIM fallback: {e}")
            return self._compute_ssim_loss(predicted_output, target)
    
    def _compute_attention_supervision_loss(self, network_attention: torch.Tensor,
                                          input_data: torch.Tensor) -> torch.Tensor:
        """Attention sparsity supervision loss."""
        # Target attention from masks
        holes_mask = input_data[:, 3:4, :, :]
        occlusion_mask = input_data[:, 4:5, :, :]
        
        # Combine repair regions
        target_attention = torch.clamp(holes_mask + occlusion_mask, 0.0, 1.0)
        
        # L1 supervision
        attention_l1 = self.l1_loss(network_attention, target_attention)
        
        # Sparsity regularization
        sparsity_loss = torch.mean(network_attention)  # Sparsity penalty
        
        # Combined attention loss
        return attention_l1 + 0.1 * sparsity_loss
    
    def _extract_spatial_attention(self, input_data: torch.Tensor) -> torch.Tensor:
        """Generate spatial attention from input masks."""
        # Extract masks
        holes_mask = input_data[:, 3:4, :, :]
        occlusion_mask = input_data[:, 4:5, :, :]
        
        # Combine hole/occlusion masks
        base_mask = torch.clamp(holes_mask + occlusion_mask, 0, 1)
        
        return base_mask
    
    def forward(self, predicted_output: torch.Tensor, 
                target: torch.Tensor, 
                input_data: torch.Tensor,
                network_spatial_attention: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute residual inpainting loss.
        
        Args:
            predicted_output: [B,3,H,W] Network output
            target: [B,3,H,W] Ground truth
            input_data: [B,7,H,W] Input channels
            network_spatial_attention: [B,1,H,W] Learned attention (optional)
            
        Returns:
            total_loss: Combined loss
            loss_dict: Loss components
        """
        
        # Extract warped RGB
        warped_rgb = input_data[:, 0:3, :, :]
        
        # Use network attention or fallback to masks
        if network_spatial_attention is not None:
            spatial_attention = network_spatial_attention
        else:
            spatial_attention = self._extract_spatial_attention(input_data)
        
        # Loss computation
        loss_dict = {}
        
        # 1. Core residual MSE
        residual_mse = self._compute_residual_mse_loss(predicted_output, target, warped_rgb)
        loss_dict['residual_mse'] = residual_mse.item()
        
        # 2. Residual L1
        residual_l1 = self._compute_residual_l1_loss(predicted_output, target, warped_rgb)
        loss_dict['residual_l1'] = residual_l1.item()
        
        # 3. Spatial weighting
        spatial_weighted = self._compute_spatial_weighted_loss(
            predicted_output, target, spatial_attention
        )
        loss_dict['spatial_weighted'] = spatial_weighted.item()
        
        # 4. Area preservation
        preservation = self._compute_preservation_loss(
            predicted_output, warped_rgb, spatial_attention
        )
        loss_dict['preservation'] = preservation.item()
        
        # 5. Edge preservation
        edge_preservation = self._compute_edge_preservation_loss(predicted_output, target)
        loss_dict['edge_preservation'] = edge_preservation.item()
        
        # 6. Perceptual loss
        perceptual = self._compute_vgg_perceptual_loss(predicted_output, target)
        loss_dict['perceptual'] = perceptual.item()
        
        # 7. Attention supervision
        attention_supervision = torch.tensor(0.0, device=predicted_output.device)
        if network_spatial_attention is not None:
            attention_supervision = self._compute_attention_supervision_loss(
                network_spatial_attention, input_data
            )
            loss_dict['attention_supervision'] = attention_supervision.item()
        else:
            loss_dict['attention_supervision'] = 0.0
        
        # Total weighted loss
        total_loss = (
            self.loss_weights['residual_mse'] * residual_mse +
            self.loss_weights['residual_l1'] * residual_l1 +
            self.loss_weights['spatial_weighted'] * spatial_weighted +
            self.loss_weights['preservation'] * preservation +
            self.loss_weights['edge_preservation'] * edge_preservation +
            self.loss_weights['perceptual'] * perceptual +
            self.loss_weights['attention_supervision'] * attention_supervision
        )
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Return current loss weights."""
        return self.loss_weights.copy()
    
    def update_loss_weights(self, new_weights: Dict[str, float]):
        """Update loss component weights."""
        for key, value in new_weights.items():
            if key in self.loss_weights:
                self.loss_weights[key] = value


def create_residual_inpainting_loss(config: Dict, device: torch.device) -> ResidualInpaintingLoss:
    """Factory function for residual inpainting loss."""
    return ResidualInpaintingLoss(device, config)


if __name__ == "__main__":
    # Test loss function
    print("[TEST] Testing residual inpainting loss function")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test data generation
    batch_size, height, width = 1, 64, 64
    
    predicted_output = torch.randn(batch_size, 3, height, width, device=device)
    target = torch.randn(batch_size, 3, height, width, device=device)
    input_data = torch.randn(batch_size, 7, height, width, device=device)
    
    # Simulate realistic input
    input_data[:, 0:3, :, :] = target + torch.randn_like(target) * 0.1  # warped_rgb
    input_data[:, 3:4, :, :] = torch.rand(batch_size, 1, height, width, device=device) > 0.8  # holes
    input_data[:, 4:5, :, :] = torch.rand(batch_size, 1, height, width, device=device) > 0.9  # occlusion
    
    # Loss function setup
    test_config = {
        'loss': {
            'residual_mse_weight': 1.0,
            'residual_l1_weight': 0.5,
            'spatial_weighted_weight': 0.8,
            'preservation_weight': 0.3,
            'edge_preservation_weight': 0.2,
            'perceptual_weight': 0.1
        }
    }
    
    loss_fn = create_residual_inpainting_loss(test_config, device)
    
    # Loss computation test
    total_loss, loss_dict = loss_fn(predicted_output, target, input_data)
    
    print(f"[SUCCESS] Residual inpainting loss function test successful")
    print(f"[INFO] Total loss: {total_loss.item():.6f}")
    print(f"[INFO] Detailed losses:")
    for key, value in loss_dict.items():
        print(f"   {key}: {value:.6f}")
    
    print(f"\n[INFO] Loss weights:")
    for key, value in loss_fn.get_loss_weights().items():
        print(f"   {key}: {value}")
    
    print(f"\n[SUCCESS] Residual inpainting loss function test completed")
    print("[INFO] Key features:")
    print("   - Residual learning: Evaluates predicted - warped vs target - warped")
    print("   - Spatial selectivity: Focus on regions needing repair")
    print("   - Preservation: Ensures non-repair regions remain intact")
    print("   - Multi-dimensional constraints: MSE + L1 + edge + perceptual losses")