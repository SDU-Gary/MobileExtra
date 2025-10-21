#!/usr/bin/env python3
"""
GAN loss utilities (hinge, etc.) for adversarial training.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class HingeGANLoss(nn.Module):
    """
    Standard hinge loss for GANs.

    Discriminator:
        L = E[max(0, 1 - D(real))] + E[max(0, 1 + D(fake))]
    Generator:
        L = -E[D(fake)]
    """

    def __init__(self):
        super().__init__()

    def discriminator_loss(self, d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
        loss_real = F.relu(1.0 - d_real)
        loss_fake = F.relu(1.0 + d_fake)
        return loss_real.mean() + loss_fake.mean()

    def generator_loss(self, d_fake: torch.Tensor) -> torch.Tensor:
        return -d_fake.mean()


def build_gan_loss(config: Dict) -> nn.Module:
    """Factory for GAN loss modules (currently only hinge)."""
    loss_type = (config or {}).get("loss", "hinge").lower()
    if loss_type in ("hinge", "hingegan", "hinge_loss"):
        return HingeGANLoss()
    raise ValueError(f"Unsupported GAN loss type: {loss_type}")

