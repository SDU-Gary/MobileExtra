#!/usr/bin/env python3
"""
Feature-level PatchGAN Discriminator for perceptual discrimination.

Instead of operating on raw RGB pixels, this discriminator operates on VGG features,
enabling better perceptual quality assessment and reducing pixel-level artifacts.

Key advantages:
1. Perceptual similarity: Operates in feature space aligned with human perception
2. Reduced artifacts: Less sensitive to minor pixel shifts
3. Better texture learning: Focuses on semantic/structural features
4. Complementary to pixel-level discrimination: Can be used alongside standard discriminator
"""

from typing import Optional, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGFeatureExtractor(nn.Module):
    """
    Extract VGG features for feature-level discrimination.

    Uses pretrained VGG16 to extract intermediate features at specified layers.
    Features are extracted in eval mode and frozen (no gradient updates).
    """

    def __init__(
        self,
        feature_layers: Optional[List[str]] = None,
        requires_grad: bool = False
    ):
        """
        Args:
            feature_layers: List of layer names to extract features from.
                           Default: ['relu3_3'] for mid-level features
                           Options: 'relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'
            requires_grad: Whether to compute gradients for VGG (usually False)
        """
        super().__init__()

        if feature_layers is None:
            feature_layers = ['relu3_3']  # Default: mid-level features

        self.feature_layers = feature_layers

        # Load pretrained VGG16 (multiple fallback strategies, same as training framework)
        try:
            # Try new API with DEFAULT weights (torchvision >= 0.13)
            vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        except (AttributeError, TypeError):
            try:
                # Try with IMAGENET1K_V1 weights
                vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            except (AttributeError, TypeError):
                # Fallback to old API (torchvision < 0.13)
                vgg16 = models.vgg16(pretrained=True)

        self.features = vgg16.features

        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = requires_grad

        # Layer name to index mapping (VGG16 features)
        self.layer_name_mapping = {
            'relu1_1': 1,
            'relu1_2': 3,
            'relu2_1': 6,
            'relu2_2': 8,
            'relu3_1': 11,
            'relu3_2': 13,
            'relu3_3': 15,
            'relu4_1': 18,
            'relu4_2': 20,
            'relu4_3': 22,
            'relu5_1': 25,
            'relu5_2': 27,
            'relu5_3': 29,
        }

        # Validate feature layers
        for layer in self.feature_layers:
            if layer not in self.layer_name_mapping:
                raise ValueError(f"Unknown layer: {layer}. Available: {list(self.layer_name_mapping.keys())}")

        # ImageNet normalization
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract VGG features at specified layers.

        Args:
            x: Input tensor [B, 3, H, W] in range [0, 1]

        Returns:
            Dictionary mapping layer names to feature tensors
        """
        # Normalize to ImageNet statistics
        x = (x - self.mean) / self.std

        output = {}
        for name, layer_idx in self.layer_name_mapping.items():
            if name in self.feature_layers:
                # Extract features up to this layer
                x = self.features[:layer_idx + 1](x)
                output[name] = x

        return output


class FeaturePatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator operating on VGG feature maps.

    Architecture:
    - Input: VGG feature maps (e.g., relu3_3: 256 channels)
    - Process: Lightweight conv layers to reduce to patch predictions
    - Output: Patch authenticity logits

    This is significantly smaller than pixel-level discriminator since:
    1. Input resolution is already downsampled by VGG
    2. Input has rich semantic features (no need for deep feature extraction)
    """

    def __init__(
        self,
        in_channels: int = 256,  # relu3_3 has 256 channels
        base_channels: int = 128,
        num_layers: int = 3,
        norm_type: str = "instance"
    ):
        """
        Args:
            in_channels: Number of input feature channels (depends on VGG layer)
            base_channels: Base number of channels for discriminator
            num_layers: Number of conv layers (3 recommended for feature-level)
            norm_type: Normalization type ('instance', 'batch', 'none')
        """
        super().__init__()

        self.in_channels = in_channels

        layers = []
        nf = base_channels

        # First layer: feature channels → base_channels
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, nf, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )

        # Subsequent layers
        for i in range(1, num_layers):
            prev_nf = nf
            nf = min(base_channels * (2 ** i), 512)
            stride = 2 if i < num_layers - 1 else 1

            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        prev_nf, nf,
                        kernel_size=4, stride=stride, padding=1,
                        bias=(norm_type == "none")
                    ),
                    self._make_norm(norm_type, nf),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )

        # Output layer: map to 1 channel logits
        layers.append(nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)
        self._init_weights()

    @staticmethod
    def _make_norm(norm_type: str, num_features: int) -> nn.Module:
        """Create normalization layer."""
        norm_type = (norm_type or "instance").lower()
        if norm_type == "instance":
            return nn.InstanceNorm2d(num_features, affine=True)
        if norm_type == "batch":
            return nn.BatchNorm2d(num_features)
        if norm_type in ("none", "identity"):
            return nn.Identity()
        raise ValueError(f"Unsupported norm type: {norm_type}")

    def _init_weights(self):
        """Initialize weights with normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: VGG feature tensor [B, C, H, W]

        Returns:
            Patch authenticity logits [B, 1, H', W']
        """
        return self.model(features)


class FeatureLevelDiscriminator(nn.Module):
    """
    Complete feature-level discriminator: VGG feature extraction + PatchGAN discrimination.

    Supports both symmetric and asymmetric modes:
    - Symmetric: Same input channels for real and fake (e.g., both 3ch RGB)
    - Asymmetric: Different channels (real: 3ch RGB, fake: 4ch RGB+mask)

    In asymmetric mode, the mask channel is dropped before VGG feature extraction,
    since VGG expects 3-channel RGB input.
    """

    def __init__(
        self,
        vgg_layer: str = 'relu3_3',
        asymmetric: bool = False,
        real_channels: int = 3,
        fake_channels: int = 4,
        base_channels: int = 128,
        num_layers: int = 3,
        norm_type: str = "instance"
    ):
        """
        Args:
            vgg_layer: Which VGG layer to extract features from
                      (relu3_3: 256ch, relu4_3: 512ch, relu5_3: 512ch)
            asymmetric: Whether to use asymmetric mode (different channels for real/fake)
            real_channels: Number of channels for real samples (3 for RGB)
            fake_channels: Number of channels for fake samples (4 for RGB+mask)
            base_channels: Base channels for patch discriminator
            num_layers: Number of conv layers in patch discriminator
            norm_type: Normalization type
        """
        super().__init__()

        self.vgg_layer = vgg_layer
        self.asymmetric = asymmetric
        self.real_channels = real_channels
        self.fake_channels = fake_channels

        # VGG feature extractor (shared for all inputs)
        self.vgg_extractor = VGGFeatureExtractor(
            feature_layers=[vgg_layer],
            requires_grad=False  # Freeze VGG
        )

        # Determine feature channels for this VGG layer
        vgg_feature_channels = {
            'relu1_2': 64,
            'relu2_2': 128,
            'relu3_3': 256,
            'relu4_3': 512,
            'relu5_3': 512,
        }

        if vgg_layer not in vgg_feature_channels:
            raise ValueError(f"Unsupported VGG layer: {vgg_layer}")

        feature_channels = vgg_feature_channels[vgg_layer]

        # Patch discriminator on features
        self.patch_disc = FeaturePatchDiscriminator(
            in_channels=feature_channels,
            base_channels=base_channels,
            num_layers=num_layers,
            norm_type=norm_type
        )

        # Set to eval mode and freeze VGG
        self.vgg_extractor.eval()

    def _prepare_input(self, x: torch.Tensor, is_real: bool = False) -> torch.Tensor:
        """
        Prepare input for VGG feature extraction.

        Args:
            x: Input tensor [B, C, H, W]
            is_real: Whether this is a real sample

        Returns:
            RGB tensor [B, 3, H, W] suitable for VGG
        """
        if self.asymmetric:
            if is_real or x.shape[1] == self.real_channels:
                # Real samples: should already be 3ch RGB
                return x[:, :3]
            else:
                # Fake samples: drop mask channel, keep RGB
                return x[:, :3]
        else:
            # Symmetric mode: take first 3 channels (RGB)
            return x[:, :3]

    def forward(self, x: torch.Tensor, is_real: bool = False) -> torch.Tensor:
        """
        Forward pass through feature-level discriminator.

        Args:
            x: Input tensor
               - Symmetric mode: [B, 3, H, W] or [B, 4, H, W]
               - Asymmetric mode: [B, 3, H, W] if real, [B, 4, H, W] if fake
            is_real: Whether input is real sample (used in asymmetric mode)

        Returns:
            Patch authenticity logits [B, 1, H', W']
        """
        # Prepare RGB input for VGG
        rgb = self._prepare_input(x, is_real)

        # Ensure input is in [0, 1] range (VGG expects this)
        rgb = torch.clamp(rgb, 0.0, 1.0)

        # Extract VGG features (no gradient)
        with torch.no_grad():
            features = self.vgg_extractor(rgb)[self.vgg_layer]

        # Discriminate on features
        return self.patch_disc(features)


def build_feature_discriminator(config: Optional[Dict] = None):
    """
    Factory function to build feature-level discriminator from config.

    Args:
        config: Configuration dictionary

    Returns:
        FeatureLevelDiscriminator instance

    Example config:
        {
            'feature_discriminator': {
                'vgg_layer': 'relu3_3',
                'asymmetric': True,
                'real_channels': 3,
                'fake_channels': 4,
                'base_channels': 128,
                'num_layers': 3,
                'norm_type': 'instance'
            }
        }
    """
    cfg = config or {}
    disc_cfg = cfg.get('feature_discriminator', {})

    vgg_layer = disc_cfg.get('vgg_layer', 'relu3_3')
    asymmetric = disc_cfg.get('asymmetric', False)
    real_channels = disc_cfg.get('real_channels', 3)
    fake_channels = disc_cfg.get('fake_channels', 4)
    base_channels = disc_cfg.get('base_channels', 128)
    num_layers = disc_cfg.get('num_layers', 3)
    norm_type = disc_cfg.get('norm_type', 'instance')

    return FeatureLevelDiscriminator(
        vgg_layer=vgg_layer,
        asymmetric=asymmetric,
        real_channels=real_channels,
        fake_channels=fake_channels,
        base_channels=base_channels,
        num_layers=num_layers,
        norm_type=norm_type
    )
