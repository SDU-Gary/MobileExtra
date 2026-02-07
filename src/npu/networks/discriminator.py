#!/usr/bin/env python3
"""
Patch-based discriminators for GAN training.

Currently implements a PatchGAN style discriminator commonly used for
inpainting and translation tasks. The discriminator expects conditional
inputs `[rgb, hole_mask]` in the tone-mapped LDR domain.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


def _make_norm(norm: str, num_features: int) -> nn.Module:
    norm = (norm or "instance").lower()
    if norm == "instance":
        return nn.InstanceNorm2d(num_features, affine=True)
    if norm == "batch":
        return nn.BatchNorm2d(num_features)
    if norm in ("none", "identity"):
        return nn.Identity()
    raise ValueError(f"Unsupported norm type: {norm}")


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator (70x70 receptive field for 256x256 inputs).

    Args:
        in_channels: number of input channels (e.g. 3 RGB + 1 mask).
        base_channels: number of channels in first conv layer.
        num_layers: number of downsampling layers (>=3 recommended).
        norm_type: normalization per conv block ("instance", "batch", "none").
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        num_layers: int = 5,
        norm_type: str = "instance",
    ):
        super().__init__()

        layers = []
        nf = base_channels

        # First layer: conv + LeakyReLU (no norm)
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, nf, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )
        )

        # Subsequent layers: conv + norm + LeakyReLU
        total_layers = max(3, num_layers)
        for i in range(1, total_layers):
            prev_nf = nf
            nf = min(base_channels * (2 ** i), 512)
            stride = 1 if i == total_layers - 1 else 2
            layers.append(
                nn.Sequential(
                    nn.Conv2d(prev_nf, nf, kernel_size=4, stride=stride, padding=1, bias=(norm_type == "none")),
                    _make_norm(norm_type, nf),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )

        # Output conv: map to 1 channel logits
        layers.append(nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, C, H, W]
        Returns:
            Patch authenticity logits [B, 1, H', W']
        """
        return self.model(x)


class AsymmetricPatchGANDiscriminator(nn.Module):
    """
    Asymmetric PatchGAN discriminator for inpainting tasks.

    Supports different input channels for real and fake samples:
    - Real samples: 3 channels (RGB only, no holes)
    - Fake samples: 4 channels (RGB + hole mask, shows where inpainting occurred)

    This design is more suitable for inpainting because:
    1. Real images don't have holes (perfect, complete images)
    2. Fake images have holes that were filled by the generator
    3. Discriminator learns to distinguish real textures from generated ones

    Args:
        real_channels: number of channels for real samples (default: 3 for RGB).
        fake_channels: number of channels for fake samples (default: 4 for RGB+mask).
        base_channels: number of channels in first conv layer.
        num_layers: number of downsampling layers (>=3 recommended).
        norm_type: normalization per conv block ("instance", "batch", "none").
    """

    def __init__(
        self,
        real_channels: int = 3,
        fake_channels: int = 4,
        base_channels: int = 64,
        num_layers: int = 5,
        norm_type: str = "instance",
    ):
        super().__init__()

        self.real_channels = real_channels
        self.fake_channels = fake_channels
        nf = base_channels

        # Separate input branches for real (3ch) and fake (4ch)
        self.real_input = nn.Sequential(
            nn.Conv2d(real_channels, nf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fake_input = nn.Sequential(
            nn.Conv2d(fake_channels, nf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Shared layers after first conv
        shared_layers = []
        total_layers = max(3, num_layers)
        for i in range(1, total_layers):
            prev_nf = nf
            nf = min(base_channels * (2 ** i), 512)
            stride = 1 if i == total_layers - 1 else 2
            shared_layers.append(
                nn.Sequential(
                    nn.Conv2d(prev_nf, nf, kernel_size=4, stride=stride, padding=1, bias=(norm_type == "none")),
                    _make_norm(norm_type, nf),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )

        # Output conv: map to 1 channel logits
        shared_layers.append(nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1))

        self.shared_model = nn.Sequential(*shared_layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor, is_real: bool = False) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, C, H, W]
                - If is_real=True: C=3 (RGB)
                - If is_real=False: C=4 (RGB+mask)
            is_real: Whether input is from real samples (True) or fake samples (False)
        Returns:
            Patch authenticity logits [B, 1, H', W']
        """
        # Route through appropriate input branch
        if is_real or x.shape[1] == self.real_channels:
            feat = self.real_input(x)
        else:
            feat = self.fake_input(x)

        # Process through shared layers
        return self.shared_model(feat)


class DualDiscriminator(nn.Module):
    """
    Dual discriminator wrapper combining pixel-level and feature-level discrimination.

    Args:
        pixel_disc: Pixel-level discriminator (e.g., AsymmetricPatchGANDiscriminator)
        feature_disc: Feature-level discriminator (e.g., FeatureLevelDiscriminator)
        lambda_pixel: Weight for pixel-level discriminator loss
        lambda_feature: Weight for feature-level discriminator loss
    """
    def __init__(
        self,
        pixel_disc: nn.Module,
        feature_disc: nn.Module,
        lambda_pixel: float = 1.0,
        lambda_feature: float = 0.5,
    ):
        super().__init__()
        self.pixel_disc = pixel_disc
        self.feature_disc = feature_disc
        self.lambda_pixel = lambda_pixel
        self.lambda_feature = lambda_feature

    def forward(self, x: torch.Tensor, is_real: bool = False, return_individual: bool = False):
        """
        Forward pass through both discriminators.

        Args:
            x: Input tensor
            is_real: Whether input is real or fake
            return_individual: If True, return individual discriminator outputs

        Returns:
            If return_individual=False: weighted combined output
            If return_individual=True: (pixel_out, feature_out, combined_out)
        """
        pixel_out = self.pixel_disc(x, is_real=is_real)
        feature_out = self.feature_disc(x, is_real=is_real)

        # Combined output (weighted average)
        combined_out = self.lambda_pixel * pixel_out + self.lambda_feature * feature_out

        if return_individual:
            return pixel_out, feature_out, combined_out
        else:
            return combined_out


def build_discriminator(config: Optional[Dict[str, any]] = None):
    """Factory helper to create a discriminator from config dict.

    Supports:
    - Single discriminator: PatchGANDiscriminator or AsymmetricPatchGANDiscriminator
    - Dual discriminator: Pixel-level + Feature-level

    Returns:
        Discriminator module (single or dual)
    """
    cfg = config or {}

    # Check if dual discriminator mode is enabled
    dual_cfg = cfg.get("dual_discriminator", {})
    if dual_cfg.get("enable", False):
        # Import feature discriminator
        try:
            from feature_discriminator import FeatureLevelDiscriminator
        except ImportError:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(__file__))
            from feature_discriminator import FeatureLevelDiscriminator

        # Build pixel-level discriminator
        pixel_cfg = dual_cfg.get("pixel", {})
        pixel_disc = AsymmetricPatchGANDiscriminator(
            real_channels=int(pixel_cfg.get("real_channels", 3)),
            fake_channels=int(pixel_cfg.get("fake_channels", 4)),
            base_channels=int(pixel_cfg.get("base_channels", 64)),
            num_layers=int(pixel_cfg.get("num_layers", 5)),
            norm_type=pixel_cfg.get("norm_type", "instance"),
        )

        # Build feature-level discriminator
        feature_cfg = dual_cfg.get("feature", {})
        feature_disc = FeatureLevelDiscriminator(
            vgg_layer=feature_cfg.get("vgg_layer", "relu3_3"),
            asymmetric=bool(feature_cfg.get("asymmetric", True)),
            real_channels=int(feature_cfg.get("real_channels", 3)),
            fake_channels=int(feature_cfg.get("fake_channels", 4)),
            base_channels=int(feature_cfg.get("base_channels", 128)),
            num_layers=int(feature_cfg.get("num_layers", 3)),
            norm_type=feature_cfg.get("norm_type", "instance"),
        )

        # Combine into dual discriminator
        lambda_pixel = float(pixel_cfg.get("lambda", 1.0))
        lambda_feature = float(feature_cfg.get("lambda", 0.5))

        dual_disc = DualDiscriminator(
            pixel_disc=pixel_disc,
            feature_disc=feature_disc,
            lambda_pixel=lambda_pixel,
            lambda_feature=lambda_feature,
        )

        print(f"✅ Dual discriminator created:")
        print(f"   - Pixel-level: AsymmetricPatchGAN ({pixel_cfg.get('base_channels', 64)}ch, {pixel_cfg.get('num_layers', 5)} layers, λ={lambda_pixel})")
        print(f"   - Feature-level: VGG {feature_cfg.get('vgg_layer', 'relu3_3')} + PatchGAN ({feature_cfg.get('base_channels', 128)}ch, {feature_cfg.get('num_layers', 3)} layers, λ={lambda_feature})")

        return dual_disc

    # Original single discriminator logic
    disc_cfg = cfg.get("discriminator", {}) if isinstance(cfg.get("discriminator", {}), dict) else {}

    # Check if asymmetric mode is enabled
    use_asymmetric = disc_cfg.get("asymmetric", cfg.get("asymmetric", False))

    base_channels = int(disc_cfg.get("base_channels", cfg.get("base_channels", 64)))
    num_layers = int(disc_cfg.get("num_layers", cfg.get("num_layers", 5)))
    norm_type = disc_cfg.get("norm_type", cfg.get("norm_type", "instance"))

    if use_asymmetric:
        # Asymmetric discriminator for inpainting (real: 3ch, fake: 4ch)
        real_channels = int(disc_cfg.get("real_channels", 3))
        fake_channels = int(disc_cfg.get("fake_channels", 4))

        return AsymmetricPatchGANDiscriminator(
            real_channels=real_channels,
            fake_channels=fake_channels,
            base_channels=base_channels,
            num_layers=num_layers,
            norm_type=norm_type,
        )
    else:
        # Standard symmetric discriminator
        in_channels = int(disc_cfg.get("input_channels", cfg.get("input_channels", 4)))

        return PatchGANDiscriminator(
            in_channels=in_channels,
            base_channels=base_channels,
            num_layers=num_layers,
            norm_type=norm_type,
        )

