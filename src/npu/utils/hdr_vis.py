#!/usr/bin/env python3
"""
HDR Tone-Mapping Utilities

Provides common tone-mapping functions for linear HDR tensors.

All functions expect PyTorch tensors with shape [..., C, H, W] or [C, H, W].
Outputs are clamped to [0, 1] for display or perceptual loss consumption.
"""

import torch
from typing import Optional


def _ensure_chw(x: torch.Tensor) -> torch.Tensor:
    # Accept [C,H,W] or [B,C,H,W]; passthrough otherwise.
    if x.dim() == 3:
        return x
    elif x.dim() == 4:
        return x
    else:
        raise ValueError(f"Unsupported tensor shape for tone-mapping: {tuple(x.shape)}")


def _compute_adaptive_exposure(x: torch.Tensor, cfg: Optional[dict]) -> float:
    """Compute adaptive exposure factor based on luminance statistics.
    cfg keys: { enable: bool, target_luminance: float, percentile: float }
    Returns scalar exposure multiplier.
    """
    if not cfg or not cfg.get('enable', False):
        return 1.0
    target = float(cfg.get('target_luminance', 0.18))
    perc = float(cfg.get('percentile', 0.9))
    # Luminance from RGB (assume first 3 channels)
    if x.dim() == 3:
        rgb = x[:3]
        Y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    elif x.dim() == 4:
        rgb = x[:, :3]
        Y = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    else:
        return 1.0
    try:
        y_flat = Y.reshape(-1)
        stat = torch.quantile(y_flat, torch.tensor(perc, device=y_flat.device))
        stat = float(max(stat.item(), 1e-6))
    except Exception:
        stat = float(max(Y.mean().item(), 1e-6))
    return target / stat


def tone_map_reinhard(hdr: torch.Tensor, gamma: float = 2.2, exposure: float = 1.0,
                      adaptive_exposure: Optional[dict] = None) -> torch.Tensor:
    """Apply Reinhard global tone mapping then gamma correction.

    Args:
        hdr: linear HDR tensor (non-negative recommended)
        gamma: display gamma (>= 1.0)
    Returns:
        ldr: tensor in [0,1], same shape as input
    """
    x = _ensure_chw(hdr)
    # Apply exposure (static and/or adaptive)
    exp = float(exposure) * _compute_adaptive_exposure(x, adaptive_exposure)
    if exp != 1.0:
        x = x * exp
    # HDR -> LDR (Reinhard)
    ldr = x / (1.0 + x)
    # Gamma correction
    if gamma != 1.0:
        ldr = torch.pow(torch.clamp(ldr, 1e-8, 1.0), 1.0 / gamma)
    return torch.clamp(ldr, 0.0, 1.0)


def tone_map(hdr: torch.Tensor, method: str = "reinhard", gamma: float = 2.2,
             exposure: float = 1.0, adaptive_exposure: Optional[dict] = None) -> torch.Tensor:
    """Generic tone-mapping dispatcher.

    Currently supports:
        - reinhard
        
    """
    m = (method or "reinhard").lower()
    if m == "reinhard":
        return tone_map_reinhard(hdr, gamma=gamma, exposure=exposure, adaptive_exposure=adaptive_exposure)
    # Fallback to Reinhard for unknown methods
    return tone_map_reinhard(hdr, gamma=gamma, exposure=exposure, adaptive_exposure=adaptive_exposure)
