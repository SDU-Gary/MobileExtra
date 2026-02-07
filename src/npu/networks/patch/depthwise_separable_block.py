#!/usr/bin/env python3
"""
Depthwise Separable Convolution Blocks - Phase 2 Optimization

轻量级卷积模块，用于替代PatchGatedConv以提升推理速度：
- Depthwise Separable Conv: ~1/9 计算量 vs 标准Conv3x3
- 相比Gated Conv: ~1/18 计算量（Gated = 2× 标准Conv）
- 保持边界感知能力，简化实现

预期提速：
- Encoder1: 103.7ms → ~8ms (13×)
- Decoder4: 103.4ms → ~8ms (13×)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise Separable Convolution - 轻量级卷积替代方案

    标准Conv3x3的FLOPs: C_in × C_out × 9 × H × W
    Depthwise Separable:
      - Depthwise:  C_in × 9 × H × W (groups=C_in)
      - Pointwise:  C_in × C_out × H × W
      Total: (9 + C_out) × C_in × H × W

    当C_in ≈ C_out时，约为标准Conv的 1/9 计算量
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 bias=False, activation=None):
        super().__init__()

        # Depthwise convolution (每个通道独立卷积)
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # 关键: groups=C_in
            bias=bias
        )

        # Pointwise convolution (1x1卷积混合通道)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

        # 激活函数
        self.activation = activation if activation is not None else nn.LeakyReLU(0.2, inplace=True)

        # 边界增强权重（简化版，可选）
        self.boundary_weight = nn.Parameter(torch.ones(1, out_channels, 1, 1) * 0.3)

    def forward(self, x, boundary_mask=None):
        """
        Args:
            x: [B, C_in, H, W]
            boundary_mask: [B, 1, H, W] 可选的边界掩码
        Returns:
            [B, C_out, H, W]
        """
        # Depthwise + Pointwise
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.activation(out)

        # 边界感知增强（如果提供mask）
        if boundary_mask is not None:
            if boundary_mask.shape[-2:] != out.shape[-2:]:
                boundary_mask = F.interpolate(boundary_mask, size=out.shape[-2:], mode='nearest')
            boundary_enhancement = self.boundary_weight * boundary_mask
            out = out * (1.0 + boundary_enhancement)

        return out


class DepthwiseSeparableConvBlock(nn.Module):
    """Depthwise Separable残差块 - 替代PatchGatedConvBlock

    PatchGatedConvBlock计算量分析 (C=24, H=1080, W=1920):
      - 2× PatchGatedConv2d (每个 = feature_conv + mask_conv)
      - Total: 4× 标准Conv3x3 = 4 × (24×24×9×2.07M) = 43 GFLOPs

    DepthwiseSeparableConvBlock:
      - 2× DepthwiseSeparableConv
      - Total: 2 × [(9+24)×24×2.07M] = 3.3 GFLOPs
      - 比例: 3.3 / 43 = 0.077 (7.7%)

    预期提速: ~13× (对于全分辨率层)
    """

    def __init__(self, channels, use_boundary_aware=True, norm_type='groupnorm'):
        super().__init__()

        self.channels = channels
        self.use_boundary_aware = use_boundary_aware

        # 第一个Depthwise Separable Conv
        self.conv1 = DepthwiseSeparableConv2d(
            channels, channels,
            kernel_size=3, stride=1, padding=1,
            activation=nn.LeakyReLU(0.2, inplace=True)
        )

        # 第二个Depthwise Separable Conv (残差分支)
        self.conv2 = DepthwiseSeparableConv2d(
            channels, channels,
            kernel_size=3, stride=1, padding=1,
            activation=nn.Identity()  # 残差连接前不激活
        )

        # 归一化层
        self.norm_type = norm_type
        if norm_type == 'groupnorm':
            # GroupNorm: 计算group数量
            def _calc_groups(c: int, cpg: int = 16) -> int:
                g = max(1, c // cpg)
                while g > 1 and (c % g != 0):
                    g -= 1
                return g
            groups = _calc_groups(channels, 16)
            self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=channels)
            self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=channels)
        elif norm_type == 'batchnorm':
            self.norm1 = nn.BatchNorm2d(channels)
            self.norm2 = nn.BatchNorm2d(channels)
        else:  # none
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        # 最终激活
        self.final_activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, boundary_mask=None):
        """
        Args:
            x: [B, C, H, W]
            boundary_mask: [B, 1, H, W] 可选
        Returns:
            [B, C, H, W]
        """
        identity = x

        # Conv1 → Norm1
        out = self.conv1(x, boundary_mask if self.use_boundary_aware else None)
        out = self.norm1(out)

        # Conv2 → Norm2
        out = self.conv2(out, boundary_mask if self.use_boundary_aware else None)
        out = self.norm2(out)

        # 残差连接 + 激活
        out = out + identity
        out = self.final_activation(out)

        return out


class HybridConvBlock(nn.Module):
    """混合卷积块 - 同时支持Gated Conv和Depthwise Separable Conv

    用于灵活切换和对比测试，方便消融实验。
    """

    def __init__(self, channels, use_boundary_aware=True, conv_type='depthwise_separable'):
        super().__init__()

        self.conv_type = conv_type

        if conv_type == 'depthwise_separable':
            self.block = DepthwiseSeparableConvBlock(channels, use_boundary_aware)
        elif conv_type == 'gated':
            # 导入原始的PatchGatedConvBlock
            try:
                from .patch_network import PatchGatedConvBlock
                self.block = PatchGatedConvBlock(channels, use_boundary_aware)
            except ImportError:
                raise ImportError("Cannot import PatchGatedConvBlock for hybrid mode")
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")

    def forward(self, x, boundary_mask=None):
        return self.block(x, boundary_mask)


def test_depthwise_separable():
    """测试Depthwise Separable Conv模块"""
    print("=== Depthwise Separable Conv测试 ===\n")

    # 测试配置
    batch_size = 2
    channels = 24
    height, width = 1080, 1920  # 全分辨率

    # 创建测试数据
    test_input = torch.randn(batch_size, channels, height, width)
    boundary_mask = torch.rand(batch_size, 1, height, width)

    print(f"输入形状: {test_input.shape}")
    print(f"Boundary mask: {boundary_mask.shape}\n")

    # 1. 测试单个Depthwise Separable Conv
    print("1. 测试DepthwiseSeparableConv2d:")
    conv = DepthwiseSeparableConv2d(channels, channels)

    with torch.no_grad():
        output = conv(test_input, boundary_mask)

    print(f"   输出形状: {output.shape}")
    print(f"   参数量: {sum(p.numel() for p in conv.parameters()):,}")

    # 计算FLOPs
    depthwise_flops = channels * 9 * height * width
    pointwise_flops = channels * channels * height * width
    total_flops = depthwise_flops + pointwise_flops
    print(f"   FLOPs: {total_flops/1e9:.2f} GFLOPs")
    print(f"   (Depthwise: {depthwise_flops/1e9:.2f}G, Pointwise: {pointwise_flops/1e9:.2f}G)\n")

    # 2. 测试Depthwise Separable Block
    print("2. 测试DepthwiseSeparableConvBlock:")
    block = DepthwiseSeparableConvBlock(channels)

    with torch.no_grad():
        output = block(test_input, boundary_mask)

    print(f"   输出形状: {output.shape}")
    print(f"   参数量: {sum(p.numel() for p in block.parameters()):,}")

    block_flops = 2 * total_flops  # 2个conv
    print(f"   FLOPs: {block_flops/1e9:.2f} GFLOPs\n")

    # 3. 对比标准Conv3x3
    print("3. 与标准Conv3x3对比:")
    standard_conv_flops = channels * channels * 9 * height * width
    print(f"   标准Conv3x3 FLOPs: {standard_conv_flops/1e9:.2f} GFLOPs")
    print(f"   Depthwise Separable FLOPs: {total_flops/1e9:.2f} GFLOPs")
    print(f"   加速比: {standard_conv_flops/total_flops:.2f}×\n")

    # 4. 对比Gated Conv
    print("4. 与PatchGatedConv对比:")
    gated_conv_flops = 2 * standard_conv_flops  # feature + mask
    gated_block_flops = 2 * gated_conv_flops  # 2个gated conv
    print(f"   PatchGatedConvBlock FLOPs: {gated_block_flops/1e9:.2f} GFLOPs")
    print(f"   DepthwiseSeparableConvBlock FLOPs: {block_flops/1e9:.2f} GFLOPs")
    print(f"   加速比: {gated_block_flops/block_flops:.2f}×\n")

    # 5. 性能估算
    print("5. Encoder1/Decoder4性能估算:")
    print(f"   当前耗时: ~103ms (实测)")
    estimated_new_time = 103 / (gated_block_flops/block_flops)
    print(f"   预期耗时: ~{estimated_new_time:.1f}ms")
    print(f"   预期提速: {103/estimated_new_time:.1f}×")

    return conv, block


if __name__ == "__main__":
    conv, block = test_depthwise_separable()
    print("\n✅ Depthwise Separable Conv模块测试完成")
    print("可用于Phase 2优化: 替换Encoder1和Decoder4")
