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


def build_discriminator(config: Optional[Dict[str, any]] = None) -> PatchGANDiscriminator:
    """Factory helper to create a PatchGAN discriminator from config dict."""
    cfg = config or {}
    disc_cfg = cfg.get("discriminator", {}) if isinstance(cfg.get("discriminator", {}), dict) else {}
    in_channels = int(disc_cfg.get("input_channels", cfg.get("input_channels", 4)))
    base_channels = int(disc_cfg.get("base_channels", cfg.get("base_channels", 64)))
    num_layers = int(disc_cfg.get("num_layers", cfg.get("num_layers", 5)))
    norm_type = disc_cfg.get("norm_type", cfg.get("norm_type", "instance"))

    return PatchGANDiscriminator(
        in_channels=in_channels,
        base_channels=base_channels,
        num_layers=num_layers,
        norm_type=norm_type,
    )

