"""
Student Patch Network variants (S1/S2):
 - S1: base_channels=16, 5-level U-Net, all depthwise separable + Hardswish
 - S2: base_channels=12, 4-level U-Net (set num_levels=4), depthwise separable + Hardswish

NEW: Attention-Enhanced Variants (Phase 2):
 - S1_attn_bc18: 0.58M params, 4.01× compression, LightweightSelfAttention @ bottleneck
 - S1_attn_bc20: 0.72M params, 3.25× compression, RECOMMENDED (85-90% accuracy target)
 - S1_attn_bc22: 0.86M params, 2.69× compression, higher capacity

Designed for faster inference while keeping the same I/O contract:
  input:  [B, 7, H, W]
  output: residual [B, 3, H, W]  (reconstructed optional)

Key differences vs V2:
  * Smaller base channels
  * Depthwise separable on all encoder/decoder blocks
  * Optional 4-level topology to cut one down/up stage
  * Optional LightweightSelfAttention at bottleneck for global context (use_attention=True)
  * Still keep 7→8 input padding and 8→3 output projection for INT8 friendliness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lightweight_attention import LightweightSelfAttention


class DWHSBlock(nn.Module):
    """Depthwise separable conv block with Hardswish, optional boundary mask."""

    def __init__(self, channels: int, use_boundary_aware: bool = False):
        super().__init__()
        self.use_boundary_aware = use_boundary_aware
        self.dw = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x, boundary_mask=None):
        out = self.dw(x)
        out = self.pw(out)
        out = self.act(out)
        if self.use_boundary_aware and boundary_mask is not None:
            if boundary_mask.shape[-2:] != out.shape[-2:]:
                boundary_mask = F.interpolate(boundary_mask, size=out.shape[-2:], mode="nearest")
            out = out * boundary_mask
        return out


class HighFreqEnhanceBlock(nn.Module):
    """High-frequency detail recovery module - TEXTURE FIX

    Uses standard convolution (not depthwise) to enable cross-channel texture aggregation.

    Design:
    1. Standard 3×3 conv allows RGB channel interaction for texture learning
    2. Residual connection preserves original features
    3. Lightweight: single conv layer + activation
    4. Scale factor 0.2 to avoid over-correction
    """
    def __init__(self, channels: int):
        super().__init__()
        # Standard convolution: enables cross-channel high-freq aggregation
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x):
        # Residual connection: only learn high-frequency residual
        residual = self.act(self.bn(self.conv(x)))
        return x + 0.2 * residual  # 0.2 scaling factor


class StudentPatchNetwork(nn.Module):
    def __init__(self, input_channels: int = 7, output_channels: int = 3,
                 base_channels: int = 16, residual_scale_factor: float = 1.0,
                 num_levels: int = 5, use_attention: bool = False,
                 decoder_scale: float = 1.0):
        super().__init__()
        assert num_levels in (4, 5), "num_levels must be 4 or 5"
        self.base_channels = base_channels
        self.num_levels = num_levels
        self.use_attention = use_attention
        self.decoder_scale = decoder_scale

        # Encoder channel schedule (always full capacity)
        self.ch1 = base_channels
        self.ch2 = int(base_channels * 1.5)  # floor
        self.ch3 = base_channels * 2
        self.ch4 = base_channels * 3
        self.ch5 = base_channels * 4

        # Decoder channel schedule (scaled independently for asymmetric design)
        self.ch1_dec = int(self.ch1 * decoder_scale) if decoder_scale < 1.0 else self.ch1
        self.ch2_dec = int(self.ch2 * decoder_scale) if decoder_scale < 1.0 else self.ch2
        self.ch3_dec = int(self.ch3 * decoder_scale) if decoder_scale < 1.0 else self.ch3
        self.ch4_dec = int(self.ch4 * decoder_scale) if decoder_scale < 1.0 else self.ch4
        # Bottleneck channel remains same (ch5)

        # Input padding 7->8 (INT8 friendly)
        self.input_padding = nn.Conv2d(input_channels, 8, 1)
        self.input_proj = nn.Conv2d(8, self.ch1, 3, padding=1)

        # Encoder
        self.enc1 = DWHSBlock(self.ch1, use_boundary_aware=True)
        self.down1 = nn.Conv2d(self.ch1, self.ch2, 3, stride=2, padding=1)

        self.enc2 = DWHSBlock(self.ch2, use_boundary_aware=True)
        self.down2 = nn.Conv2d(self.ch2, self.ch3, 3, stride=2, padding=1)

        self.enc3 = DWHSBlock(self.ch3, use_boundary_aware=True)
        if num_levels == 5:
            self.down3 = nn.Conv2d(self.ch3, self.ch4, 3, stride=2, padding=1)
            self.enc4 = DWHSBlock(self.ch4, use_boundary_aware=True)
            self.down4 = nn.Conv2d(self.ch4, self.ch5, 3, stride=2, padding=1)
            bottleneck_channels = self.ch5
        else:
            bottleneck_channels = self.ch3

        # Bottleneck - TEXTURE FIX: Standard conv for better cross-channel high-freq aggregation
        # Changed from DWHSBlock (depthwise) to standard conv for enhanced texture learning
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.Hardswish(inplace=True),
        )

        # Optional Attention Module at bottleneck (for global context modeling)
        self.attention = None
        if use_attention:
            self.attention = LightweightSelfAttention(bottleneck_channels, enable_position_encoding=False)

        # Decoder (asymmetric architecture support)
        if num_levels == 5:
            # Skip connection adapters (1x1 conv) for channel matching when asymmetric
            self.skip_adapt4 = nn.Conv2d(self.ch4, self.ch4_dec, 1) if self.ch4 != self.ch4_dec else nn.Identity()
            self.skip_adapt3 = nn.Conv2d(self.ch3, self.ch3_dec, 1) if self.ch3 != self.ch3_dec else nn.Identity()
            self.skip_adapt2 = nn.Conv2d(self.ch2, self.ch2_dec, 1) if self.ch2 != self.ch2_dec else nn.Identity()
            self.skip_adapt1 = nn.Conv2d(self.ch1, self.ch1_dec, 1) if self.ch1 != self.ch1_dec else nn.Identity()

            self.up1 = nn.Sequential(nn.Conv2d(self.ch5, self.ch5 * 4, 3, padding=1), nn.PixelShuffle(2))
            self.dec1_conv = nn.Conv2d(self.ch5 + self.ch4_dec, self.ch4_dec, 3, padding=1)
            self.dec1 = DWHSBlock(self.ch4_dec, use_boundary_aware=True)

            self.up2 = nn.Sequential(nn.Conv2d(self.ch4_dec, self.ch4_dec * 4, 3, padding=1), nn.PixelShuffle(2))
            self.dec2_conv = nn.Conv2d(self.ch4_dec + self.ch3_dec, self.ch3_dec, 3, padding=1)
            self.dec2 = DWHSBlock(self.ch3_dec, use_boundary_aware=True)

            self.up3 = nn.Sequential(nn.Conv2d(self.ch3_dec, self.ch3_dec * 4, 3, padding=1), nn.PixelShuffle(2))
            self.dec3_conv = nn.Conv2d(self.ch3_dec + self.ch2_dec, self.ch2_dec, 3, padding=1)
            self.dec3 = DWHSBlock(self.ch2_dec, use_boundary_aware=True)
            self.hfr3 = HighFreqEnhanceBlock(self.ch2_dec)  # TEXTURE FIX: High-freq enhance at 1/2 scale

            self.up4 = nn.Sequential(nn.Conv2d(self.ch2_dec, self.ch2_dec * 4, 3, padding=1), nn.PixelShuffle(2))
            self.dec4_conv = nn.Conv2d(self.ch2_dec + self.ch1_dec, self.ch1_dec, 3, padding=1)
            self.dec4 = DWHSBlock(self.ch1_dec, use_boundary_aware=True)
            self.hfr4 = HighFreqEnhanceBlock(self.ch1_dec)  # TEXTURE FIX: High-freq enhance at full scale
        else:
            # 4-level decoder (asymmetric support)
            self.skip_adapt2 = nn.Conv2d(self.ch2, self.ch2_dec, 1) if self.ch2 != self.ch2_dec else nn.Identity()
            self.skip_adapt1 = nn.Conv2d(self.ch1, self.ch1_dec, 1) if self.ch1 != self.ch1_dec else nn.Identity()

            self.up1 = nn.Sequential(nn.Conv2d(self.ch3, self.ch3 * 4, 3, padding=1), nn.PixelShuffle(2))
            self.dec1_conv = nn.Conv2d(self.ch3 + self.ch2_dec, self.ch2_dec, 3, padding=1)
            self.dec1 = DWHSBlock(self.ch2_dec, use_boundary_aware=True)

            self.up2 = nn.Sequential(nn.Conv2d(self.ch2_dec, self.ch2_dec * 4, 3, padding=1), nn.PixelShuffle(2))
            self.dec2_conv = nn.Conv2d(self.ch2_dec + self.ch1_dec, self.ch1_dec, 3, padding=1)
            self.dec2 = DWHSBlock(self.ch1_dec, use_boundary_aware=True)

        # Output projection  -> 8 -> 3 (uses decoder output channels)
        self.output_expand = nn.Conv2d(self.ch1_dec, 8, 1)
        self.output_act = nn.Hardswish(inplace=True)
        self.output_conv = nn.Conv2d(8, output_channels, 1)

        self.register_buffer('residual_scale_factor', torch.tensor(float(residual_scale_factor)))

    def forward(self, x, return_full_image: bool = False, boundary_override=None, return_intermediates=False):
        boundary_mask = boundary_override
        if boundary_mask is None:
            boundary_mask = self._gen_boundary_mask(x)

        x_pad = self.input_padding(x)
        x0 = self.input_proj(x_pad)

        e1 = self.enc1(x0, boundary_mask)
        d1 = self.down1(e1)

        e2 = self.enc2(d1, boundary_mask)
        d2 = self.down2(e2)

        e3 = self.enc3(d2, boundary_mask)

        if self.num_levels == 5:
            d3 = self.down3(e3)
            e4 = self.enc4(d3, boundary_mask)
            d4 = self.down4(e4)
            e5 = d4  # encoder5 output before bottleneck
            bn = self.bottleneck(e5)

            # Apply attention at bottleneck for global context modeling
            if self.attention is not None:
                bn = self.attention(bn)

            # Apply skip connection adapters for asymmetric architecture
            u1 = self._upsample_to(self.up1(bn), e4)
            u1 = self.dec1_conv(torch.cat([u1, self.skip_adapt4(e4)], dim=1))
            u1 = self.dec1(u1, boundary_mask)

            u2 = self._upsample_to(self.up2(u1), e3)
            u2 = self.dec2_conv(torch.cat([u2, self.skip_adapt3(e3)], dim=1))
            u2 = self.dec2(u2, boundary_mask)

            u3 = self._upsample_to(self.up3(u2), e2)
            u3 = self.dec3_conv(torch.cat([u3, self.skip_adapt2(e2)], dim=1))
            u3 = self.dec3(u3, boundary_mask)
            u3 = self.hfr3(u3)  # TEXTURE FIX: Apply high-freq enhancement

            u4 = self._upsample_to(self.up4(u3), e1)
            u4 = self.dec4_conv(torch.cat([u4, self.skip_adapt1(e1)], dim=1))
            u4 = self.dec4(u4, boundary_mask)
            u4 = self.hfr4(u4)  # TEXTURE FIX: Apply high-freq enhancement
            feat = u4
        else:
            bn = self.bottleneck(e3)

            # Apply attention at bottleneck for global context modeling
            if self.attention is not None:
                bn = self.attention(bn)

            # Apply skip connection adapters for asymmetric architecture
            u1 = self._upsample_to(self.up1(bn), e2)
            u1 = self.dec1_conv(torch.cat([u1, self.skip_adapt2(e2)], dim=1))
            u1 = self.dec1(u1, boundary_mask)

            u2 = self._upsample_to(self.up2(u1), e1)
            u2 = self.dec2_conv(torch.cat([u2, self.skip_adapt1(e1)], dim=1))
            u2 = self.dec2(u2, boundary_mask)
            feat = u2

        out = self.output_expand(feat)
        out = self.output_act(out)
        residual = self.output_conv(out)

        if residual.shape[2:] != x.shape[2:]:
            residual = F.interpolate(residual, size=x.shape[2:], mode='bilinear', align_corners=False)

        # 返回中间层特征（用于知识蒸馏）
        if return_intermediates and self.num_levels == 5:
            intermediates = {
                'encoder3': e3,      # [B, 40, H/4, W/4] for base_channels=20
                'encoder5': e5,      # [B, 80, H/16, W/16]
                'bottleneck': bn,    # [B, 80, H/16, W/16]
                'decoder2': u2,      # [B, 40, H/4, W/4] (may be scaled if asymmetric)
                'decoder4': u4,      # [B, 20, H, W]
            }
            return residual, intermediates

        if return_full_image:
            reconstructed = x[:, :3] + residual * self.residual_scale_factor
            return residual, reconstructed
        return residual

    @staticmethod
    def _upsample_to(x, ref):
        """Bilinear upsample tensor x to the spatial size of ref."""
        if x.shape[2:] != ref.shape[2:]:
            return F.interpolate(x, size=ref.shape[2:], mode='bilinear', align_corners=False)
        return x

    def _gen_boundary_mask(self, x):
        # simple gradient-based edge map on RGB
        if x.shape[1] >= 3:
            rgb = x[:, :3]
            gray = torch.mean(rgb, dim=1, keepdim=True)
            kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
            edges = F.conv2d(gray, kernel, padding=1)
            return torch.sigmoid(torch.abs(edges))
        return torch.ones_like(x[:, :1])

    @staticmethod
    def from_variant(name: str):
        name = name.lower()
        # Original variants (without attention)
        if name == 's1':
            return StudentPatchNetwork(base_channels=16, num_levels=5, use_attention=False)
        if name == 's2':
            return StudentPatchNetwork(base_channels=12, num_levels=4, use_attention=False)

        # NEW: Attention-enhanced variants (Phase 2 implementation)
        if name == 's1_attn_bc18':
            return StudentPatchNetwork(base_channels=18, num_levels=5, use_attention=True)
        if name == 's1_attn_bc20':
            # RECOMMENDED: 0.72M params, 3.25× compression, 85-90% accuracy target
            return StudentPatchNetwork(base_channels=20, num_levels=5, use_attention=True)
        if name == 's1_attn_bc22':
            return StudentPatchNetwork(base_channels=22, num_levels=5, use_attention=True)

        # Asymmetric U-Net variant (reduced decoder for faster inference)
        if name == 's1_asymmetric':
            # Encoder: 20→30→40→60→80, Decoder: 80→48→32→20→16 (~40% reduction)
            # Target: ~0.5M params, ~30% MACs reduction vs s1_attn_bc20
            return StudentPatchNetwork(base_channels=20, num_levels=5, use_attention=True,
                                      decoder_scale=0.6)

        # Symmetric U-Net variant (balanced encoder-decoder for better texture recovery)
        if name == 's1_symmetric_bc32':
            # Encoder: 32→48→64→96→128, Decoder: 128→96→64→48→32 (symmetric)
            # Target: ~1.5M params, balanced capacity, better quality vs s1_asymmetric
            # Designed for staged training with selective distillation + balanced GAN
            return StudentPatchNetwork(base_channels=32, num_levels=5, use_attention=True,
                                      decoder_scale=1.0)

        raise ValueError(f"Unknown student variant: {name}")
