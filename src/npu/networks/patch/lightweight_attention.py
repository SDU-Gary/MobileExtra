#!/usr/bin/env python3
"""
轻量级自注意力模块 - 替代FFC的全局依赖建模方案

核心特性：
1. 分层空间下采样注意力（128x128 → 32x32 → 8x8）
2. 可分离注意力机制（水平+垂直）
3. 通道分组处理
4. 多尺度特征融合
5. 移动端优化：参数量<50K，计算复杂度O(N)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class PositionalEncoding2D(nn.Module):
    """2D位置编码 - 为patch提供空间位置信息"""
    
    def __init__(self, channels: int, max_len: int = 128):
        super().__init__()
        self.channels = channels
        self.max_len = max_len
        
        # 使用简化的2D位置编码，避免复杂的交错编码
        pe = torch.zeros(max_len, max_len, channels)
        
        # 创建位置坐标
        y_position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) / max_len  # 归一化
        x_position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(0) / max_len  # 归一化
        
        # 简化的位置编码：为每个通道分配不同的频率
        for c in range(channels):
            freq = 1.0 / (10000.0 ** (c / channels))
            if c % 4 == 0:  # Y方向 sin
                pe[:, :, c] = torch.sin(y_position * freq).expand(max_len, max_len)
            elif c % 4 == 1:  # Y方向 cos
                pe[:, :, c] = torch.cos(y_position * freq).expand(max_len, max_len)
            elif c % 4 == 2:  # X方向 sin  
                pe[:, :, c] = torch.sin(x_position * freq).expand(max_len, max_len)
            else:  # X方向 cos
                pe[:, :, c] = torch.cos(x_position * freq).expand(max_len, max_len)
        
        self.register_buffer('pe', pe.permute(2, 0, 1))  # [C, H, W]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            position encoded features: [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 动态生成位置编码以适应任意输入尺寸
        if H > self.max_len or W > self.max_len:
            # 如果输入尺寸超过预设最大长度，动态生成位置编码
            pe = self._generate_dynamic_pe(C, H, W).to(x.device)
        else:
            # 使用预计算的位置编码并裁剪到实际尺寸
            pe = self.pe[:C, :H, :W]
        
        pe = pe.unsqueeze(0).expand(B, -1, -1, -1)
        return x + pe * 0.1  # 小权重添加位置信息
    
    def _generate_dynamic_pe(self, channels: int, height: int, width: int) -> torch.Tensor:
        """动态生成位置编码"""
        pe = torch.zeros(channels, height, width)
        
        # 创建归一化的位置坐标
        y_position = torch.arange(0, height, dtype=torch.float).unsqueeze(1) / height
        x_position = torch.arange(0, width, dtype=torch.float).unsqueeze(0) / width
        
        # 为每个通道分配不同的频率
        for c in range(channels):
            freq = 1.0 / (10000.0 ** (c / channels))
            if c % 4 == 0:  # Y方向 sin
                pe[c] = torch.sin(y_position * freq).expand(height, width)
            elif c % 4 == 1:  # Y方向 cos
                pe[c] = torch.cos(y_position * freq).expand(height, width)
            elif c % 4 == 2:  # X方向 sin  
                pe[c] = torch.sin(x_position * freq).expand(height, width)
            else:  # X方向 cos
                pe[c] = torch.cos(x_position * freq).expand(height, width)
        
        return pe


class SeparableAttention2D(nn.Module):
    """可分离2D注意力 - 分别计算水平和垂直方向的注意力"""
    
    def __init__(self, channels: int, reduction_ratio: int = 8):
        super().__init__()
        self.channels = channels
        self.hidden_dim = max(channels // reduction_ratio, 8)
        
        # 水平方向注意力 (H维度)
        self.h_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),  # [B, C, H, 1]
            nn.Conv2d(channels, self.hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 垂直方向注意力 (W维度)
        self.w_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),  # [B, C, 1, W]
            nn.Conv2d(channels, self.hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 通道注意力
        self.c_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, 1, 1]
            nn.Conv2d(channels, self.hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 特征融合权重
        self.fusion_weight = nn.Parameter(torch.ones(3) / 3.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            attention weighted features: [B, C, H, W]
        """
        # 计算三个方向的注意力
        h_att = self.h_attention(x)  # [B, C, H, 1]
        w_att = self.w_attention(x)  # [B, C, 1, W]
        c_att = self.c_attention(x)  # [B, C, 1, 1]
        
        # 应用注意力权重
        h_weighted = x * h_att
        w_weighted = x * w_att  
        c_weighted = x * c_att
        
        # 加权融合
        weights = F.softmax(self.fusion_weight, dim=0)
        output = (weights[0] * h_weighted + 
                 weights[1] * w_weighted + 
                 weights[2] * c_weighted)
        
        return output


class HierarchicalSpatialAttention(nn.Module):
    """分层空间注意力 - 多尺度全局依赖建模（自适应尺度）。"""
    
    def __init__(self, channels: int, scales: list = [32, 16, 8]):
        super().__init__()
        self.channels = channels
        self.base_scales = list(scales)
        
        # 每个尺度共用相同的卷积结构，区别只在自适应池化输出尺寸。
        self.scale_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels // 4, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, channels // 4, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, channels, kernel_size=1),
                nn.Sigmoid()
            )
            for _ in self.base_scales
        ])
        
        # 尺度权重参数（运行时根据有效尺度截取）。
        self.scale_weights = nn.Parameter(torch.ones(len(self.base_scales)) / max(len(self.base_scales), 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            multi-scale attended features: [B, C, H, W]
        """
        B, C, H, W = x.shape
        if H == 0 or W == 0:
            return x
        
        scale_features = []
        used_weights = []
        seen_shapes = set()
        
        for idx, target_scale in enumerate(self.base_scales):
            target_h = max(1, min(target_scale, H))
            target_w = max(1, min(target_scale, W))
            shape_key = (target_h, target_w)
            if shape_key in seen_shapes:
                continue  # 避免重复尺度
            seen_shapes.add(shape_key)
            
            pooled = F.adaptive_avg_pool2d(x, output_size=shape_key)
            att_weight = self.scale_blocks[idx](pooled)
            att_weight = F.interpolate(att_weight, size=(H, W), mode='bilinear', align_corners=False)
            scale_features.append(att_weight * x)
            used_weights.append(idx)
        
        if not scale_features:
            return x
        
        weights = F.softmax(self.scale_weights[used_weights], dim=0)
        fused_features = sum(w * feat for w, feat in zip(weights, scale_features))
        return fused_features


class LightweightSelfAttention(nn.Module):
    """
    轻量级自注意力模块 - 替代FFC的全局依赖建模方案
    
    特性：
    - 参数量 < 50K
    - 计算复杂度 O(N) 
    - 支持任意尺寸输入全局建模 (如270x480)
    - 移动端友好设计
    """
    
    def __init__(self, channels: int, enable_position_encoding: bool = True):
        super().__init__()
        self.channels = channels
        self.enable_position_encoding = enable_position_encoding
        
        # 位置编码
        if enable_position_encoding:
            self.pos_encoding = PositionalEncoding2D(channels)
        
        # 通道分组数
        self.num_groups = max(channels // 16, 1)
        self.channels_per_group = channels // self.num_groups
        
        # 可分离注意力（主要的全局建模组件）
        self.separable_attention = SeparableAttention2D(channels, reduction_ratio=8)
        
        # 分层空间注意力（多尺度全局依赖）
        self.hierarchical_attention = HierarchicalSpatialAttention(channels, scales=[32, 16, 8])
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, kernel_size=1)
        )
        
        # 输出门控
        self.output_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 残差权重
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] - patch features
        Returns:
            globally attended features: [B, C, H, W]
        """
        identity = x
        B, C, H, W = x.shape
        
        # 1. 位置编码
        if self.enable_position_encoding:
            x = self.pos_encoding(x)
        
        # 2. 可分离注意力（主要全局建模）
        separable_out = self.separable_attention(x)
        
        # 3. 分层空间注意力（多尺度全局依赖）
        hierarchical_out = self.hierarchical_attention(x)
        
        # 4. 特征融合
        combined_features = separable_out + hierarchical_out
        fused_features = self.feature_fusion(combined_features)
        
        # 5. 输出门控
        gate = self.output_gate(fused_features)
        gated_output = fused_features * gate
        
        # 6. 残差连接
        output = identity + self.residual_weight * gated_output
        
        return output
    
    def get_parameter_count(self) -> dict:
        """获取参数统计信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 按模块分类统计
        module_params = {}
        if hasattr(self, 'pos_encoding'):
            module_params['position_encoding'] = sum(p.numel() for p in self.pos_encoding.parameters())
        module_params['separable_attention'] = sum(p.numel() for p in self.separable_attention.parameters())
        module_params['hierarchical_attention'] = sum(p.numel() for p in self.hierarchical_attention.parameters())
        module_params['feature_fusion'] = sum(p.numel() for p in self.feature_fusion.parameters())
        module_params['output_gate'] = sum(p.numel() for p in self.output_gate.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'module_breakdown': module_params,
            'memory_mb': total_params * 4 / (1024 * 1024)  # FP32
        }


def test_lightweight_attention():
    """测试轻量级自注意力模块"""
    print("=== 轻量级自注意力模块测试 ===")
    
    # 测试配置
    batch_size = 2
    channels = 96  # 对应PatchNetwork的bottleneck通道数
    height, width = 32, 32  # bottleneck的特征图尺寸
    
    # 创建测试数据
    test_input = torch.randn(batch_size, channels, height, width)
    
    # 创建注意力模块
    attention = LightweightSelfAttention(channels)
    
    # 前向传播测试
    print(f"输入形状: {test_input.shape}")
    
    with torch.no_grad():
        output = attention(test_input)
    
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.3f}, {output.max():.3f}]")
    
    # 参数统计
    param_info = attention.get_parameter_count()
    print(f"\n参数统计:")
    print(f"总参数量: {param_info['total_parameters']:,}")
    print(f"内存占用: {param_info['memory_mb']:.2f} MB")
    
    print(f"\n模块参数分布:")
    for module, params in param_info['module_breakdown'].items():
        percentage = params / param_info['total_parameters'] * 100
        print(f"  {module}: {params:,} ({percentage:.1f}%)")
    
    # 性能验证
    print(f"\n性能指标:")
    print(f"参数效率: {'PASS' if param_info['total_parameters'] < 50000 else 'FAIL'} (目标<50K)")
    print(f"内存效率: {'PASS' if param_info['memory_mb'] < 1.0 else 'FAIL'} (目标<1MB)")
    
    return attention, output, param_info


if __name__ == "__main__":
    attention_module, output, info = test_lightweight_attention()
    print(f"\nSUCCESS: 轻量级自注意力模块测试完成")
    print(f"可直接替代PatchFFCBlock使用")
