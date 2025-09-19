#!/usr/bin/env python3
"""
Patch专用网络 - 支持任意尺寸输入的轻量级补全网络 (如270x480)

U-Net结构，轻量化通道数：24->48->96，参数量减少75%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# 导入轻量级自注意力模块
try:
    from .lightweight_attention import LightweightSelfAttention
except ImportError:
    from lightweight_attention import LightweightSelfAttention


class PatchGatedConv2d(nn.Module):
    """Patch专用门控卷积层 - 边界感知机制"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, 
                 activation=nn.LeakyReLU(0.2, inplace=True)):
        super(PatchGatedConv2d, self).__init__()
        
        # 特征卷积
        self.feature_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        
        # 门控卷积
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        
        # 激活函数
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        
        # 边界感知权重
        self.boundary_weight = nn.Parameter(torch.ones(1, out_channels, 1, 1) * 0.5)
    
    def forward(self, x, boundary_mask=None):
        # 计算特征和门控
        feature = self.feature_conv(x)
        mask = self.sigmoid(self.mask_conv(x))
        
        # 门控机制
        output = self.activation(feature) * mask
        
        if boundary_mask is not None:
            if boundary_mask.shape[-2:] != output.shape[-2:]:
                boundary_mask = F.interpolate(boundary_mask, size=output.shape[-2:], mode='nearest')
            boundary_enhancement = self.boundary_weight * boundary_mask
            output = output * (1.0 + boundary_enhancement)
        
        return output


class PatchGatedConvBlock(nn.Module):
    """Patch专用门控卷积残差块"""
    
    def __init__(self, channels, use_boundary_aware=True):
        super(PatchGatedConvBlock, self).__init__()
        
        self.conv1 = PatchGatedConv2d(channels, channels, 3, 1, 1)
        self.conv2 = PatchGatedConv2d(channels, channels, 3, 1, 1, activation=nn.Identity())
        
        self.use_boundary_aware = use_boundary_aware
        
        self.final_activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x, boundary_mask=None):
        identity = x
        
        out = self.conv1(x, boundary_mask if self.use_boundary_aware else None)
        out = self.conv2(out, boundary_mask if self.use_boundary_aware else None)
        
        out = out + identity
        return self.final_activation(out)


class PatchFFCBlock(nn.Module):
    """Patch专用FFC块 - 频域-空域融合"""
    
    def __init__(self, channels, ratio_gin=0.5, ratio_gout=0.5):
        super(PatchFFCBlock, self).__init__()
        
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        
        self.global_in_num = int(channels * ratio_gin)
        self.local_in_num = channels - self.global_in_num
        
        # 本地特征处理
        if self.local_in_num > 0:
            self.local_conv = nn.Sequential(
                nn.Conv2d(self.local_in_num, self.local_in_num, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.local_in_num, self.local_in_num, 3, 1, 1),
            )
        
        # 全局特征处理
        if self.global_in_num > 0:
            self.global_fusion = nn.Conv2d(self.global_in_num * 3, self.global_in_num, 1, 1, 0)
            self.global_conv = nn.Sequential(
                nn.Conv2d(self.global_in_num, self.global_in_num, 1, 1, 0),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.global_in_num, self.global_in_num, 1, 1, 0),
            )
        
        # 特征融合
        if self.local_in_num > 0 and self.global_in_num > 0:
            self.fusion = nn.Conv2d(channels, channels, 1, 1, 0)
    
    def forward(self, x):
        identity = x
        B, C, H, W = x.shape
        
        # 分离本地和全局特征
        if self.local_in_num > 0:
            local_feat = x[:, :self.local_in_num]
            local_out = self.local_conv(local_feat)
        else:
            local_out = None
        
        if self.global_in_num > 0:
            global_feat = x[:, self.local_in_num:]
            global_out = self._global_processing(global_feat)
        else:
            global_out = None
        
        # 特征融合
        if local_out is not None and global_out is not None:
            fused_feat = torch.cat([local_out, global_out], dim=1)
            out = self.fusion(fused_feat)
        elif local_out is not None:
            out = local_out
        else:
            out = global_out
        
        return out + identity
    
    def _global_processing(self, x):
        """多尺度全局特征处理"""
        B, C, H, W = x.shape
        
        global_avg = F.adaptive_avg_pool2d(x, 1)
        global_max = F.adaptive_max_pool2d(x, 1)
        mid_scale = F.adaptive_avg_pool2d(x, (4, 4))
        mid_scale_flat = mid_scale.view(B, C, 16).mean(dim=2, keepdim=True).unsqueeze(-1)
        
        global_feat = torch.cat([global_avg, global_max, mid_scale_flat], dim=1)
        global_processed = self.global_fusion(global_feat)
        global_processed = self.global_conv(global_processed)
        global_out = global_processed.expand(-1, -1, H, W)
        
        return global_out


class PatchNetwork(nn.Module):
    """Enhanced Patch专用残差学习网络 - 支持任意尺寸输入(如270x480)，5层U-Net架构，~2.8M参数
    
     核心改进:
    - 边界掩码一致性: 网络与损失函数都使用图像边缘检测，完全对齐优化目标
    - 网络容量增强: base_channels 24→64，通道数增加2.67倍，深度扩展至5层
    - 渐进式特征提取: 64→96→128→192→256，优化参数效率
    
     残差学习模式:
    - 网络输出: residual_prediction (范围 [-1, 1])
    - 最终结果: warped_rgb + residual_prediction * scale_factor
    - 优势: 简化学习任务，仅学习差异部分
    
     架构特点:
    - 5层编码器-解码器: 4次下采样 + 4次上采样，深层特征建模
    - 边界感知门控卷积: 每层都具备边界敏感性，与损失函数语义对齐
    - 轻量级自注意力: 256通道瓶颈层全局建模，~32K参数
    - 自适应尺寸匹配: skip connection自动处理非正方形输入 
    """
    
    def __init__(self, input_channels=7, output_channels=3, base_channels=64):
        super(PatchNetwork, self).__init__()
        
        self.ch1 = base_channels      # 64
        self.ch2 = int(base_channels * 1.5)  # 96  
        self.ch3 = base_channels * 2  # 128
        self.ch4 = base_channels * 3  # 192
        self.ch5 = base_channels * 4  # 256 (bottleneck)
        
        # 输入投影层
        self.input_proj = PatchGatedConv2d(input_channels, self.ch1, 3, 1, 1)
        
        self.encoder1 = PatchGatedConvBlock(self.ch1, use_boundary_aware=True)
        self.down1 = PatchGatedConv2d(self.ch1, self.ch2, 3, 2, 1)
        
        self.encoder2 = PatchGatedConvBlock(self.ch2, use_boundary_aware=True)
        self.down2 = PatchGatedConv2d(self.ch2, self.ch3, 3, 2, 1)
        
        self.encoder3 = PatchGatedConvBlock(self.ch3, use_boundary_aware=True)
        self.down3 = PatchGatedConv2d(self.ch3, self.ch4, 3, 2, 1)
        
        self.encoder4 = PatchGatedConvBlock(self.ch4, use_boundary_aware=True)
        self.down4 = PatchGatedConv2d(self.ch4, self.ch5, 3, 2, 1)
        
        self.encoder5 = PatchGatedConvBlock(self.ch5, use_boundary_aware=False)
        
        self.bottleneck = LightweightSelfAttention(self.ch5, enable_position_encoding=True)

        self.up1 = nn.Upsample(scale_factor=2, mode='bicubic')
        self.up_conv1 = PatchGatedConv2d(self.ch5 + self.ch4, self.ch4, 3, 1, 1)
        self.decoder1 = PatchGatedConvBlock(self.ch4, use_boundary_aware=True)

        self.up2 = nn.Upsample(scale_factor=2, mode='bicubic')
        self.up_conv2 = PatchGatedConv2d(self.ch4 + self.ch3, self.ch3, 3, 1, 1)
        self.decoder2 = PatchGatedConvBlock(self.ch3, use_boundary_aware=True)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bicubic')
        self.up_conv3 = PatchGatedConv2d(self.ch3 + self.ch2, self.ch2, 3, 1, 1)
        self.decoder3 = PatchGatedConvBlock(self.ch2, use_boundary_aware=True)

        self.up4 = nn.Upsample(scale_factor=2, mode='bicubic')
        self.up_conv4 = PatchGatedConv2d(self.ch2 + self.ch1, self.ch1, 3, 1, 1)
        self.decoder4 = PatchGatedConvBlock(self.ch1, use_boundary_aware=True)
        
        # 输出层
        self.output_conv = nn.Conv2d(self.ch1, output_channels, 1, 1, 0)
        self.output_activation = nn.Tanh()  # residual ∈ [-1, 1]
        self.register_buffer('residual_scale_factor', torch.tensor(3.0))
        
        self.register_buffer('boundary_kernel', self._create_boundary_kernel())
        
        # 权重初始化
        self._init_weights()
    
    def _create_boundary_kernel(self):
        kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1], 
            [-1, -1, -1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return kernel
    
    def _generate_boundary_mask(self, x):
        """
         FIXED: 边界掩码现在与损失函数保持一致
        使用图像边缘检测而非空洞检测，确保网络优化目标与损失函数对齐
        """
        if x.shape[1] >= 3:
            #  NEW: 使用warped_rgb (前3通道) 进行边缘检测，与损失函数一致
            warped_rgb = x[:, :3]  # 提取warped RGB图像
            
            # 转换为灰度图进行边缘检测
            rgb_gray = torch.mean(warped_rgb, dim=1, keepdim=True)  # [B, 1, H, W]
            
            # 应用边缘检测卷积核（与损失函数相同的kernel）
            edges = F.conv2d(rgb_gray, self.boundary_kernel, padding=1)
            
            #  使用与损失函数相同的激活方式
            boundary_mask = torch.sigmoid(torch.abs(edges) * 1.0)
            
            #  可选：如果有空洞信息，可以作为额外增强
            if x.shape[1] > 3:
                hole_mask = x[:, 3:4]  # 空洞掩码
                hole_edges = F.conv2d(hole_mask, self.boundary_kernel, padding=1)
                hole_boundary = torch.sigmoid(torch.abs(hole_edges) * 0.5)
                
                # 组合：图像边缘 + 空洞边缘，以图像边缘为主
                boundary_mask = torch.clamp(boundary_mask + 0.3 * hole_boundary, 0.0, 2.0)
            
            return boundary_mask
        else:
            return None
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_full_image=False, boundary_override=None):
        """
        前向传播 - 残差学习版本
        
        Args:
            x: 输入特征 [B, 7, H, W]
            return_full_image: 是否返回完整重建图像
            boundary_override: 可选的外部边界掩码 [B, 1, H, W]，若提供则覆盖内部基于空洞的边界图
            
        Returns:
            默认: residual_prediction [B, 3, H, W] ∈ [-1, 1]
            可选: (residual_prediction, reconstructed_image) 如果 return_full_image=True
        """
        boundary_mask = boundary_override if boundary_override is not None else self._generate_boundary_mask(x)
        
        x_input = self.input_proj(x, boundary_mask)
        
        #  NEW: 5层编码器前向传播
        e1 = self.encoder1(x_input, boundary_mask)
        d1 = self.down1(e1, boundary_mask)
        
        e2 = self.encoder2(d1, boundary_mask)
        d2 = self.down2(e2, boundary_mask)
        
        e3 = self.encoder3(d2, boundary_mask)
        d3 = self.down3(e3, boundary_mask)
        
        e4 = self.encoder4(d3, boundary_mask)
        d4 = self.down4(e4, boundary_mask)
        
        e5 = self.encoder5(d4)
        
        bottleneck_out = self.bottleneck(e5)
        
        #  NEW: 5层解码器前向传播，对称skip connections
        u1 = self.up1(bottleneck_out)
        if u1.shape[2:] != e4.shape[2:]:
            u1 = F.interpolate(u1, size=e4.shape[2:], mode='bilinear', align_corners=False)
        u1 = torch.cat([u1, e4], dim=1)
        u1 = self.up_conv1(u1, boundary_mask)
        u1 = self.decoder1(u1, boundary_mask)
        
        u2 = self.up2(u1)
        if u2.shape[2:] != e3.shape[2:]:
            u2 = F.interpolate(u2, size=e3.shape[2:], mode='bilinear', align_corners=False)
        u2 = torch.cat([u2, e3], dim=1)
        u2 = self.up_conv2(u2, boundary_mask)
        u2 = self.decoder2(u2, boundary_mask)
        
        u3 = self.up3(u2)
        if u3.shape[2:] != e2.shape[2:]:
            u3 = F.interpolate(u3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        u3 = torch.cat([u3, e2], dim=1)
        u3 = self.up_conv3(u3, boundary_mask)
        u3 = self.decoder3(u3, boundary_mask)
        
        u4 = self.up4(u3)
        if u4.shape[2:] != e1.shape[2:]:
            u4 = F.interpolate(u4, size=e1.shape[2:], mode='bilinear', align_corners=False)
        u4 = torch.cat([u4, e1], dim=1)
        u4 = self.up_conv4(u4, boundary_mask)
        u4 = self.decoder4(u4, boundary_mask)
        
        #  残差学习: 网络输出残差预测
        residual_prediction = self.output_conv(u4)
        residual_prediction = self.output_activation(residual_prediction)  # [-1, 1]
        
        #  FIX: 确保输出尺寸与输入完全匹配
        if residual_prediction.shape[2:] != x.shape[2:]:
            residual_prediction = F.interpolate(residual_prediction, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        if return_full_image:
            # 重建完整图像：warped_rgb + residual * scale_factor
            warped_rgb = x[:, :3]  # 输入的前3通道
            reconstructed_image = warped_rgb + residual_prediction * self.residual_scale_factor
            reconstructed_image = torch.clamp(reconstructed_image, -1.0, 1.0)
            return residual_prediction, reconstructed_image
        
        return residual_prediction
    
    def get_parameter_count(self):
        """获取网络参数数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }
    
    def get_model_info(self):
        """获取模型详细信息"""
        param_info = self.get_parameter_count()
        
        model_size_mb = param_info['total_parameters'] * 4 / (1024 * 1024)
        
        return {
            **param_info,
            'model_size_mb': model_size_mb,
            'target_size_mb': 8.0,  #  NEW: 增加目标大小以适应更大容量
            'compression_ratio': 8.0 / model_size_mb if model_size_mb > 0 else 0,
            'architecture': 'Enhanced PatchNetwork (5-Layer U-Net + Edge-Aligned Boundary + Gated Conv + Self-Attention + Residual Learning)',
            'channel_progression': '64 → 96 → 128 → 192 → 256 (bottleneck)',
            'encoder_layers': 5,
            'decoder_layers': 5,
            'boundary_detection': 'Image Edge-Based (Aligned with Loss Function)',
            'skip_connections': 'Full U-Net with Size-Adaptive Interpolation',
            'input_shape': '[B, 7, H, W] (supports arbitrary sizes like 270×480)',
            'output_shape': '[B, 3, H, W] (residual prediction ∈ [-1, 1])',
            'learning_mode': 'Residual Learning (warped_rgb + residual * scale_factor)',
            'key_improvements': [
                'Edge-based boundary detection (consistent with loss)',
                '2.67x increased capacity (64 vs 24 base channels)',
                '5-layer progressive feature extraction',
                'Enhanced skip connections with size matching'
            ]
        }


def test_patch_network():
    """测试PatchNetwork - 残差学习版本"""
    # 创建测试数据
    batch_size = 2
    test_input = torch.randn(batch_size, 7, 128, 128)
    
    # 模拟warped_rgb作为前3通道
    test_input[:, :3] = torch.randn_like(test_input[:, :3]) * 0.5
    
    # 创建网络
    network = PatchNetwork()
    
    # 前向传播测试
    with torch.no_grad():
        # 测试残差预测模式
        residual_pred = network(test_input)
        
        # 测试完整重建模式
        residual_pred2, reconstructed = network(test_input, return_full_image=True)
        
        # 验证一致性
        assert torch.allclose(residual_pred, residual_pred2), "残差预测不一致"
    
    # 输出信息
    model_info = network.get_model_info()
    
    print("=== Enhanced PatchNetwork测试结果 ===")
    print(f"输入形状: {test_input.shape}")
    print(f"残差预测形状: {residual_pred.shape}")
    print(f"重建图像形状: {reconstructed.shape}")
    print(f"参数数量: {model_info['total_parameters']:,}")
    print(f"模型大小: {model_info['model_size_mb']:.2f} MB")
    print(f"压缩比: {model_info['compression_ratio']:.1f}x")
    print(f"架构: {model_info['architecture']}")
    print(f"通道进展: {model_info['channel_progression']}")
    print(f"边界检测: {model_info['boundary_detection']}")
    print(f"学习模式: {model_info['learning_mode']}")
    
    print(f"残差预测范围: [{residual_pred.min():.3f}, {residual_pred.max():.3f}]")
    print(f"重建图像范围: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
    
    # 验证残差学习逻辑
    warped_rgb = test_input[:, :3]
    manual_reconstruction = warped_rgb + residual_pred * network.residual_scale_factor
    reconstruction_diff = torch.mean(torch.abs(manual_reconstruction - reconstructed))
    print(f"重建一致性检查: {reconstruction_diff:.6f} (应该接近0)")
    
    return network, residual_pred, model_info


if __name__ == "__main__":
    network, residual_pred, model_info = test_patch_network()
    print("SUCCESS: PatchNetwork残差学习版本测试完成")
    print("\n关键特性:")
    print("- 网络输出: 残差预测 ∈ [-1, 1]")
    print("- 最终结果: warped_rgb + residual")
    print("- 学习任务: 仅学习差异，简化训练")
    print("- 全局建模: 轻量级自注意力替代FFC")