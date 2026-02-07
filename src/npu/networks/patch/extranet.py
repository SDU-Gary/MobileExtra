"""
ExtraNet-like architecture implementation (per paper description):

Key points from the paper:
- U-Net style encoder/decoder.
- Encoder: 11 layers of lightweight gated convolutions (LWG Conv.).
- Decoder: 7 layers of standard convolutions (no gating in upsampling).
- Lightweight gated conv: gate branch produces a **single-channel** mask M,
  applied to feature branch F: O = sigmoid(M) * F. (Original gated conv would
  output C-channel mask; single-channel reduces compute.)
- Inputs include warped frames, hole mask, G-buffers, history features. Here we
  mirror the interface of PatchNetwork: input [B,7,H,W] (RGB + mask/occlusion/MV
  placeholders). Missing buffers can be zero-filled by caller; we keep I/O shape
  consistent with existing pipeline.
- Output: residual RGB [B,3,H,W]; reconstructed = input_rgb + residual (optional).

Notes / Simplifications:
- History encoder in the paper processes three historical frames; here we expose
  an optional history tensor `hist_feats` that can be concatenated at bottleneck
  if provided. If absent, we use zeros.
- Channel schedule follows a lightweight pattern similar to the paper figure:
  start with base_ch, grow moderately. Default base_ch=32 to match paper-style
  capacity (tunable).
- Normalization: BatchNorm in encoder/decoder convs (as per Fig.7 history encoder
  description uses BN+ReLU). For simplicity we use BN+ReLU for both encoder/decoder.
- Upsampling: bilinear + conv.
- No attention or extra modules; focus on faithful LWG gating and layer counts.

This module is intended for experimentation/ablation; training stability and
performance should be validated separately.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightGatedConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.feature = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        # single-channel gate for lightweight version
        self.mask = nn.Conv2d(in_ch, 1, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        f = self.feature(x)
        m = self.mask(x)
        g = torch.sigmoid(m)  # [B,1,H,W]
        f = self.bn(f)
        f = self.act(f)
        return f * g  # broadcast g over channels


class StdConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ExtraNet(nn.Module):
    def __init__(self, input_channels=7, output_channels=3, base_channels=32, residual_scale=1.0):
        super().__init__()
        self.base = base_channels
        self.residual_scale = residual_scale

        # Encoder channel schedule (11 LWG convs)
        # We define 5 stages with downsample convs; total LWG layers = 11.
        ch1 = base_channels
        ch2 = base_channels * 2
        ch3 = base_channels * 4
        ch4 = base_channels * 4
        ch5 = base_channels * 4

        self.enc1 = nn.Sequential(
            LightweightGatedConv(input_channels, ch1),
            LightweightGatedConv(ch1, ch1),
        )  # 2 LWG
        self.down1 = nn.Conv2d(ch1, ch2, 3, 2, 1)

        self.enc2 = nn.Sequential(
            LightweightGatedConv(ch2, ch2),
            LightweightGatedConv(ch2, ch2),
        )  # +2 => 4
        self.down2 = nn.Conv2d(ch2, ch3, 3, 2, 1)

        self.enc3 = nn.Sequential(
            LightweightGatedConv(ch3, ch3),
            LightweightGatedConv(ch3, ch3),
            LightweightGatedConv(ch3, ch3),
        )  # +3 => 7
        self.down3 = nn.Conv2d(ch3, ch4, 3, 2, 1)

        self.enc4 = nn.Sequential(
            LightweightGatedConv(ch4, ch4),
            LightweightGatedConv(ch4, ch4),
        )  # +2 => 9
        self.down4 = nn.Conv2d(ch4, ch5, 3, 2, 1)

        self.enc5 = nn.Sequential(
            LightweightGatedConv(ch5, ch5),
            LightweightGatedConv(ch5, ch5),
        )  # +2 => 11 LWG total

        # Bottleneck can concatenate history features; keep projection ready
        self.hist_proj = nn.Conv2d(ch5 * 2, ch5, 1)  # if hist provided

        # Decoder: 7 standard convs total (spread across up blocks)
        self.up1 = StdConvBlock(ch5, ch4)
        self.dec1 = nn.Sequential(
            StdConvBlock(ch4 + ch4, ch4),
            StdConvBlock(ch4, ch4),
        )

        self.up2 = StdConvBlock(ch4, ch3)
        self.dec2 = nn.Sequential(
            StdConvBlock(ch3 + ch3, ch3),
            StdConvBlock(ch3, ch3),
        )

        self.up3 = StdConvBlock(ch3, ch2)
        self.dec3 = nn.Sequential(
            StdConvBlock(ch2 + ch2, ch2),
        )

        self.up4 = StdConvBlock(ch2, ch1)
        self.dec4 = nn.Sequential(
            StdConvBlock(ch1 + ch1, ch1),
        )

        self.out_conv = nn.Conv2d(ch1, output_channels, 1)

    def forward(self, x, hist_feats=None, return_full_image=False):
        # Encoder
        e1 = self.enc1(x)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        e3 = self.enc3(d2)
        d3 = self.down3(e3)

        e4 = self.enc4(d3)
        d4 = self.down4(e4)

        e5 = self.enc5(d4)

        if hist_feats is not None:
            # expect hist_feats shape [B, ch5, H/16, W/16]; if not, resize
            if hist_feats.shape[2:] != e5.shape[2:]:
                hist_feats = F.interpolate(hist_feats, size=e5.shape[2:], mode='bilinear', align_corners=False)
            bn = self.hist_proj(torch.cat([e5, hist_feats], dim=1))
        else:
            bn = e5

        # Decoder
        u1 = self._upsample_to(self.up1(bn), e4)
        u1 = self.dec1(torch.cat([u1, e4], dim=1))

        u2 = self._upsample_to(self.up2(u1), e3)
        u2 = self.dec2(torch.cat([u2, e3], dim=1))

        u3 = self._upsample_to(self.up3(u2), e2)
        u3 = self.dec3(torch.cat([u3, e2], dim=1))

        u4 = self._upsample_to(self.up4(u3), e1)
        u4 = self.dec4(torch.cat([u4, e1], dim=1))

        residual = self.out_conv(u4)
        if residual.shape[2:] != x.shape[2:]:
            residual = F.interpolate(residual, size=x.shape[2:], mode='bilinear', align_corners=False)

        if return_full_image:
            recon = x[:, :3] + residual * self.residual_scale
            return residual, recon
        return residual

    @staticmethod
    def _upsample_to(x, ref):
        if x.shape[2:] != ref.shape[2:]:
            return F.interpolate(x, size=ref.shape[2:], mode='bilinear', align_corners=False)
        return x

