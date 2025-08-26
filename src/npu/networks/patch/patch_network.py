#!/usr/bin/env python3
"""
Patch专用网络 - 128x128输入的轻量级补全网络

U-Net结构，轻量化通道数：24->48->96，参数量减少75%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


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
    """Patch专用补全网络 - 128x128输入，U-Net架构，~750K参数"""
    
    def __init__(self, input_channels=7, output_channels=3, base_channels=24):
        super(PatchNetwork, self).__init__()
        
        self.ch1 = base_channels
        self.ch2 = base_channels * 2
        self.ch3 = base_channels * 4
        
        # 输入投影层
        self.input_proj = PatchGatedConv2d(input_channels, self.ch1, 3, 1, 1)
        
        self.encoder1 = PatchGatedConvBlock(self.ch1, use_boundary_aware=True)
        self.down1 = PatchGatedConv2d(self.ch1, self.ch2, 3, 2, 1)
        
        self.encoder2 = PatchGatedConvBlock(self.ch2, use_boundary_aware=True)
        self.down2 = PatchGatedConv2d(self.ch2, self.ch3, 3, 2, 1)
        
        self.encoder3 = PatchGatedConvBlock(self.ch3, use_boundary_aware=False)
        
        self.bottleneck = PatchFFCBlock(self.ch3, ratio_gin=0.5, ratio_gout=0.5)
        
        # 解码器
        self.up1 = nn.Upsample(scale_factor=2, mode='bicubic')
        self.up_conv1 = PatchGatedConv2d(self.ch3 + self.ch2, self.ch2, 3, 1, 1)
        self.decoder1 = PatchGatedConvBlock(self.ch2, use_boundary_aware=True)

        self.up2 = nn.Upsample(scale_factor=2, mode='bicubic')
        self.up_conv2 = PatchGatedConv2d(self.ch2 + self.ch1, self.ch1, 3, 1, 1)
        self.decoder2 = PatchGatedConvBlock(self.ch1, use_boundary_aware=True)
        
        # 输出层
        self.output_conv = nn.Conv2d(self.ch1, output_channels, 1, 1, 0)
        self.output_activation = nn.Tanh()
        
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
        if x.shape[1] > 3:
            hole_mask = x[:, 3:4]
            boundary = F.conv2d(hole_mask, self.boundary_kernel, padding=1)
            boundary_mask = torch.sigmoid(boundary * 0.1)
            return boundary_mask
        else:
            return None
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        boundary_mask = self._generate_boundary_mask(x)
        
        x = self.input_proj(x, boundary_mask)
        
        e1 = self.encoder1(x, boundary_mask)
        d1 = self.down1(e1, boundary_mask)
        
        e2 = self.encoder2(d1, boundary_mask)
        d2 = self.down2(e2, boundary_mask)
        
        e3 = self.encoder3(d2)
        
        bottleneck_out = self.bottleneck(e3)
        
        u1 = self.up1(bottleneck_out)
        u1 = torch.cat([u1, e2], dim=1)
        u1 = self.up_conv1(u1, boundary_mask)
        u1 = self.decoder1(u1, boundary_mask)
        
        u2 = self.up2(u1)
        u2 = torch.cat([u2, e1], dim=1)
        u2 = self.up_conv2(u2, boundary_mask)
        u2 = self.decoder2(u2, boundary_mask)
        
        output = self.output_conv(u2)
        output = self.output_activation(output)
        
        return output
    
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
            'target_size_mb': 3.0,
            'compression_ratio': 3.0 / model_size_mb if model_size_mb > 0 else 0,
            'architecture': 'PatchNetwork (U-Net + Gated Conv + FFC)',
            'input_shape': '[B, 7, 128, 128]',
            'output_shape': '[B, 3, 128, 128]',
        }


def test_patch_network():
    """测试PatchNetwork"""
    # 创建测试数据
    batch_size = 2
    test_input = torch.randn(batch_size, 7, 128, 128)
    
    # 创建网络
    network = PatchNetwork()
    
    # 前向传播
    with torch.no_grad():
        output = network(test_input)
    
    # 输出信息
    model_info = network.get_model_info()
    
    print("=== PatchNetwork测试结果 ===")
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数数量: {model_info['total_parameters']:,}")
    print(f"模型大小: {model_info['model_size_mb']:.2f} MB")
    print(f"压缩比: {model_info['compression_ratio']:.1f}x")
    
    print(f"输出范围: [{output.min():.3f}, {output.max():.3f}]")
    
    return network, output, model_info


if __name__ == "__main__":
    network, output, model_info = test_patch_network()
    print("SUCCESS: PatchNetwork test completed")